//! A linear layer that is either dense or quantized in situ (#385).
//!
//! The vendored modules build their projections through `linear_b` /
//! `linear_no_bias`, the same names `candle_transformers::with_tracing`
//! exports, so switching them over is an import change and the vendored
//! code stays diffable against upstream.
//!
//! Whether a layer is quantized is decided while it is built, by
//! [`with_quant`]. A scope rather than a parameter, because the setting
//! would otherwise have to be threaded through every constructor of the
//! DiT, the condition encoder, the text encoder and the planner. Loads
//! run on one thread (the device worker), so the scope cannot leak into
//! another model's load.
//!
//! A quantized layer runs through candle's `QMatMul`, which wants f32
//! activations: build the component with an f32 `VarBuilder`.

use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::cell::Cell;
use std::sync::Arc;

thread_local! {
    static QUANT: Cell<Option<GgmlDType>> = const { Cell::new(None) };
}

/// Build with `quant` applied to every linear layer `f` creates. A layer
/// whose input width does not divide the block size stays dense.
pub fn with_quant<R>(quant: Option<GgmlDType>, f: impl FnOnce() -> R) -> R {
    let previous = QUANT.with(|q| q.replace(quant));
    let out = f();
    QUANT.with(|q| q.set(previous));
    out
}

#[derive(Debug, Clone)]
pub enum Linear {
    Dense(candle_nn::Linear),
    Quantized {
        weight: QMatMul,
        bias: Option<Tensor>,
    },
}

impl Linear {
    /// A dense layer from existing weights (tied heads, identity
    /// projections). Never quantized: the weight is shared or synthetic.
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self::Dense(candle_nn::Linear::new(weight, bias))
    }

    fn build(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let quant = QUANT.with(Cell::get);
        let width = weight.dims().last().copied().unwrap_or(0);
        match quant {
            Some(dtype) if width > 0 && width.is_multiple_of(dtype.block_size()) => {
                let qt = crate::harness::tp::isq::quantize_parallel(&weight, dtype)
                    .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
                Ok(Self::Quantized {
                    weight: QMatMul::from_arc(Arc::new(qt))?,
                    bias,
                })
            }
            _ => Ok(Self::Dense(candle_nn::Linear::new(weight, bias))),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized { weight, bias } => {
                // The quantized kernels read a contiguous (batch, rows, k)
                // or (rows, k) input.
                let y = match x.dims() {
                    [a, b, c, k] => weight
                        .forward(&x.contiguous()?.reshape((a * b, *c, *k))?)?
                        .reshape((*a, *b, *c, ()))?,
                    _ => weight.forward(&x.contiguous()?)?,
                };
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
        }
    }
}

pub fn linear_b(d1: usize, d2: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((d2, d1), "weight")?;
    let bias = if bias {
        Some(vb.get(d2, "bias")?)
    } else {
        None
    };
    Linear::build(weight, bias)
}

pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    linear_b(d1, d2, false, vb)
}

pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    linear_b(d1, d2, true, vb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn vb_with(weight: Tensor) -> VarBuilder<'static> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), weight);
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn quantization_is_scoped_and_skips_widths_that_do_not_tile() -> Result<()> {
        let w = Tensor::randn(0f32, 1.0, (64, 64), &Device::Cpu)?;
        let dense = linear_no_bias(64, 64, vb_with(w.clone()))?;
        assert!(!dense.is_quantized(), "no scope, no quantization");

        let q = with_quant(Some(GgmlDType::Q8_0), || {
            linear_no_bias(64, 64, vb_with(w.clone()))
        })?;
        assert!(q.is_quantized());

        let narrow = Tensor::randn(0f32, 1.0, (64, 6), &Device::Cpu)?;
        let kept = with_quant(Some(GgmlDType::Q8_0), || {
            linear_no_bias(6, 64, vb_with(narrow))
        })?;
        assert!(!kept.is_quantized(), "6 does not tile a 32-block");

        let after = linear_no_bias(64, 64, vb_with(w))?;
        assert!(!after.is_quantized(), "the scope must end with the call");
        Ok(())
    }

    #[test]
    fn q8_output_tracks_the_dense_layer() -> Result<()> {
        let w = Tensor::randn(0f32, 0.05, (128, 256), &Device::Cpu)?;
        let x = Tensor::randn(0f32, 1.0, (3, 256), &Device::Cpu)?;
        let dense = linear_no_bias(256, 128, vb_with(w.clone()))?.forward(&x)?;
        let q = with_quant(Some(GgmlDType::Q8_0), || {
            linear_no_bias(256, 128, vb_with(w))
        })?
        .forward(&x)?;
        let err = (dense.clone() - q)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = dense.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err / scale < 0.02, "q8_0 relative error {}", err / scale);
        Ok(())
    }
}
