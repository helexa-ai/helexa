//! Hyper-connections: `qwen4_exp`'s replacement for the pre-norm
//! residual stream.
//!
//! Every architecture we serve today carries one residual stream of
//! `hidden_size` and wraps each sublayer as
//! `x = x + sublayer(norm(x))`. `qwen4_exp` carries **four** streams,
//! concatenated into a single `hc_count * hidden_size` tensor, and
//! there is no `input_layernorm`, no `post_attention_layernorm` and no
//! final `model.norm` anywhere in the checkpoint — the `hc_norm` inside
//! each hyper-connection does all three jobs.
//!
//! Per sublayer:
//!
//! ```text
//! h [.., 10240]
//!   │
//!   ├─ hn   = hc_norm(h)                       grouped over 4x2560
//!   ├─ w    = sigmoid(up(silu(down(hn) / 4)))  10240 -> 320 -> 10240
//!   ├─ x    = mean over the 4 streams of (w * hn)        -> [.., 2560]
//!   ├─ inj  = 2 * sigmoid(block_inject(hn) / 4)          -> [.., 4]
//!   │
//!   │  y = sublayer(x)                                   -> [.., 2560]
//!   │
//!   └─ h' = h + (y (x) inj).flatten(-2)                  -> [.., 10240]
//! ```
//!
//! Three details are easy to get subtly wrong, and all three produce
//! fluent-but-worse output rather than a crash, so each is pinned by a
//! test below:
//!
//! 1. Both gates divide by `hc_count` **before** the nonlinearity.
//! 2. The stream mix is a `mean`, not a `sum`.
//! 3. The tensor carried into the residual add is the **un-normalised**
//!    `h`. Only the mixing and injection paths see `hn`.
//!
//! See `doc/qwen4_exp-port-spec.md` §1.

use anyhow::{Context, Result};
use candle_core::{D, Module, Tensor};
use candle_nn::Linear;
use candle_nn::var_builder::ShardedVarBuilder;

use crate::harness::arch::qwen3_5::rmsnorm::Qwen3_5RmsNorm;

/// What a hyper-connection hands its sublayer.
pub struct HyperSplit {
    /// The mixed, normalised input for the sublayer: `[.., hidden]`.
    pub input: Tensor,
    /// The **un-normalised** stream tensor to add back: `[.., hidden*hc]`.
    pub residual: Tensor,
    /// Per-stream injection weights: `[.., hc]`. `None` on the final
    /// mixer, which has no `block_inject_weight` and never rejoins.
    pub inject: Option<Tensor>,
}

/// `Qwen4ExpTextGatedResidual`.
pub struct HyperConnection {
    hc_norm: Qwen3_5RmsNorm,
    mix_down: Linear,
    mix_up: Linear,
    /// Absent on the top-level `hyper_connection_mixer`, which collapses
    /// the streams for the LM head instead of feeding a sublayer.
    block_inject: Option<Linear>,
    hidden_size: usize,
    hc_count: usize,
}

impl HyperConnection {
    pub fn load(
        vb: &ShardedVarBuilder,
        hidden_size: usize,
        hc_count: usize,
        hc_lowrank: usize,
        eps: f64,
        use_combine: bool,
    ) -> Result<Self> {
        let wide = hidden_size * hc_count;
        let hc_norm = Qwen3_5RmsNorm::load_grouped(&vb.pp("hc_norm"), wide, hidden_size, eps)
            .with_context(|| format!("load '{}/hc_norm'", vb.prefix()))?;
        let mix_down = linear(vb, "input_mix_weight_down", wide, hc_lowrank)?;
        let mix_up = linear(vb, "input_mix_weight_up", hc_lowrank, wide)?;
        let block_inject = if use_combine {
            Some(linear(vb, "block_inject_weight", wide, hc_count)?)
        } else {
            None
        };
        Ok(Self {
            hc_norm,
            mix_down,
            mix_up,
            block_inject,
            hidden_size,
            hc_count,
        })
    }

    /// Split the stream tensor into a sublayer input plus what is needed
    /// to rejoin.
    pub fn split(&self, h: &Tensor) -> candle_core::Result<HyperSplit> {
        let wide = self.hidden_size * self.hc_count;
        debug_assert_eq!(h.dims()[h.rank() - 1], wide);

        let hn = self.hc_norm.forward(h)?;
        let hc = self.hc_count as f64;

        // silu(down(hn) / hc) -> sigmoid(up(.)) : a low-rank per-stream,
        // per-channel gate over the normalised streams.
        let w = self.mix_down.forward(&hn)?;
        let w = candle_nn::ops::silu(&(w / hc)?)?;
        let w = candle_nn::ops::sigmoid(&self.mix_up.forward(&w)?)?;

        // Mean over the stream axis — NOT a sum.
        let input = (unflatten_streams(&w, self.hc_count, self.hidden_size)?
            * unflatten_streams(&hn, self.hc_count, self.hidden_size)?)?
        .mean(D::Minus2)?;

        let inject = match &self.block_inject {
            Some(bi) => {
                let raw = (bi.forward(&hn)? / hc)?;
                Some((candle_nn::ops::sigmoid(&raw)? * 2.0)?)
            }
            None => None,
        };

        Ok(HyperSplit {
            input,
            residual: h.clone(),
            inject,
        })
    }

    /// Scatter a sublayer's output back across the streams and add it to
    /// the residual.
    pub fn combine(
        &self,
        residual: &Tensor,
        block_out: &Tensor,
        inject: &Tensor,
    ) -> candle_core::Result<Tensor> {
        // outer product: [.., 1, hidden] * [.., hc, 1] -> [.., hc, hidden]
        let scattered = block_out
            .unsqueeze(D::Minus2)?
            .broadcast_mul(&inject.unsqueeze(D::Minus1)?)?;
        residual.add(&scattered.flatten_from(D::Minus2)?)
    }

    /// The top-level collapse before the LM head: mix the streams down
    /// to one `hidden_size` vector and discard the rest. There is no
    /// separate final norm in this architecture; `hc_norm` is it.
    pub fn collapse(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        Ok(self.split(h)?.input)
    }
}

fn unflatten_streams(x: &Tensor, hc: usize, hidden: usize) -> candle_core::Result<Tensor> {
    let mut shape = x.dims().to_vec();
    shape.pop();
    shape.push(hc);
    shape.push(hidden);
    x.reshape(shape)
}

fn linear(vb: &ShardedVarBuilder, name: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = vb
        .pp(name)
        .get((out_dim, in_dim), "weight")
        .with_context(|| format!("load '{}/{name}/weight'", vb.prefix()))?;
    Ok(Linear::new(weight, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Build a hyper-connection whose three projections are all zero.
    /// That makes the whole gate analytic:
    ///   down(hn) = 0 -> silu(0) = 0 -> up(0) = 0 -> sigmoid(0) = 0.5
    ///   block_inject(hn) = 0 -> 2 * sigmoid(0) = 1.0
    /// so the mix weight is exactly 0.5 everywhere and every stream is
    /// injected with weight 1. Any error in the /hc_count divisions, the
    /// sigmoid constants, or mean-vs-sum shows up as a wrong number.
    fn zeroed(hidden: usize, hc: usize, lowrank: usize) -> HyperConnection {
        let dev = Device::Cpu;
        let wide = hidden * hc;
        let z = |o: usize, i: usize| {
            Linear::new(Tensor::zeros((o, i), DType::F32, &dev).unwrap(), None)
        };
        HyperConnection {
            hc_norm: Qwen3_5RmsNorm::from_weight(
                Tensor::zeros(wide, DType::F32, &dev).unwrap(),
                1e-6,
                Some(hidden),
            ),
            mix_down: z(lowrank, wide),
            mix_up: z(wide, lowrank),
            block_inject: Some(z(hc, wide)),
            hidden_size: hidden,
            hc_count: hc,
        }
    }

    #[test]
    fn split_matches_hand_computed_forward() {
        let dev = Device::Cpu;
        let hc = zeroed(2, 2, 2);
        // Two streams whose magnitudes differ 10x, so the grouped norm
        // has to bring them to the same scale before mixing.
        let h = Tensor::new(&[[[3.0f32, 4.0, 30.0, 40.0]]], &dev).unwrap();

        let split = hc.split(&h).unwrap();

        // hn = [3,4]/rms([3,4]) ++ [30,40]/rms([30,40]) — identical pairs.
        let a = (12.5f32).sqrt();
        // mix weight is 0.5, and the two normalised streams are equal,
        // so mean over streams leaves them unchanged.
        let want = [0.5 * 3.0 / a, 0.5 * 4.0 / a];
        let got: Vec<f32> = split.input.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "input: got {got:?} want {want:?}");
        }

        let inj: Vec<f32> = split
            .inject
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(inj.len(), 2);
        for v in &inj {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "inject should be 2*sigmoid(0) = 1, got {inj:?}"
            );
        }
    }

    /// The residual handed back for the skip connection is the raw
    /// input, not the normalised one. Getting this wrong still produces
    /// fluent output, so assert it exactly.
    #[test]
    fn residual_is_the_unnormalised_input() {
        let dev = Device::Cpu;
        let hc = zeroed(2, 2, 2);
        let h = Tensor::new(&[[[3.0f32, 4.0, 30.0, 40.0]]], &dev).unwrap();
        let got: Vec<f32> = hc
            .split(&h)
            .unwrap()
            .residual
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(got, vec![3.0, 4.0, 30.0, 40.0]);
    }

    #[test]
    fn combine_scatters_the_block_output_across_streams() {
        let dev = Device::Cpu;
        let hc = zeroed(2, 2, 2);
        let residual = Tensor::new(&[[[3.0f32, 4.0, 30.0, 40.0]]], &dev).unwrap();
        let block_out = Tensor::new(&[[[1.0f32, 2.0]]], &dev).unwrap();
        // Distinct per-stream weights, so a transposed outer product
        // would give a different answer.
        let inject = Tensor::new(&[[[10.0f32, 100.0]]], &dev).unwrap();

        let got: Vec<f32> = hc
            .combine(&residual, &block_out, &inject)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // stream 0 gets 10*[1,2]; stream 1 gets 100*[1,2].
        assert_eq!(
            got,
            vec![3.0 + 10.0, 4.0 + 20.0, 30.0 + 100.0, 40.0 + 200.0]
        );
    }

    /// A round trip through split + combine must preserve the stream
    /// width, or the next layer sees the wrong shape.
    #[test]
    fn split_combine_preserves_stream_width() {
        let dev = Device::Cpu;
        let hc = zeroed(4, 4, 3);
        let h = Tensor::rand(0f32, 1f32, (2, 5, 16), &dev).unwrap();
        let split = hc.split(&h).unwrap();
        assert_eq!(split.input.dims(), &[2, 5, 4]);
        assert_eq!(split.inject.as_ref().unwrap().dims(), &[2, 5, 4]);
        let out = hc
            .combine(
                &split.residual,
                &split.input,
                split.inject.as_ref().unwrap(),
            )
            .unwrap();
        assert_eq!(out.dims(), &[2, 5, 16]);
    }

    /// The final mixer has no block_inject_weight and collapses to one
    /// hidden-size vector for the LM head.
    #[test]
    fn collapse_has_no_injection_and_narrows_to_hidden() {
        let dev = Device::Cpu;
        let mut hc = zeroed(4, 4, 3);
        hc.block_inject = None;
        let h = Tensor::rand(0f32, 1f32, (1, 3, 16), &dev).unwrap();
        assert!(hc.split(&h).unwrap().inject.is_none());
        assert_eq!(hc.collapse(&h).unwrap().dims(), &[1, 3, 4]);
    }
}
