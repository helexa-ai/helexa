//! HeartCodec decoder (#392): HeartMuLa's 12.5 Hz RVQ codes to 48 kHz
//! stereo audio.
//!
//! 1. **RVQ lookup.** Eight 8192-entry codebooks (dim 32) are summed and
//!    projected to 512, then `cond_feature_emb` and nearest 2× upsampling
//!    give the 25 Hz flow condition `mu`.
//! 2. **Flow matching.** Ten Euler steps from noise at t = 0 to latents at
//!    t = 1, with CFG 1.25 against a zero condition. The estimator is a
//!    DiT-style transformer: 24 blocks at 1536 and 6 at 3072, each with
//!    AdaLN-single timestep modulation and interleaved RoPE.
//! 3. **Scalar decoder.** Each 256-dim latent splits into two 128-dim
//!    halves, one per stereo channel. A scalar quantiser (`round(9x)/9`)
//!    is followed by a causal conv decoder: transposed-conv upsampling
//!    5·4·4·4·3, residual PReLU units, then a 2× sample-repeat
//!    post-processor, 1920× in all.
//!
//! Songs longer than one 29.76 s segment are decoded as overlapping
//! segments. Each later segment takes the previous one's last 104 latent
//! frames as in-context latents, and the waveforms crossfade linearly.
//!
//! Reference: heartlib `heartcodec/models/{decoder,flow_matching,
//! transformer,sq_codec}.py` @ 5858ca8f. The codec runs in f32 in every
//! heartlib configuration.

use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear, linear_no_bias};
use serde::Deserialize;

/// The fields of `config.json` the decoder uses.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub dim: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub num_quantizers: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub num_layers_2: usize,
    pub out_channels: usize,
    pub num_samples: usize,
    pub upsample_factors: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub latent_hidden_dim: usize,
    pub delay_kernel_size: usize,
    pub default_kernel_size: usize,
    pub init_channel: usize,
    pub res_kernel_size: usize,
    pub sample_rate: usize,
}

/// One segment, in seconds (heartlib's `duration` default).
pub const SEGMENT_SECS: f64 = 29.76;
/// LM frames per second.
pub const CODE_RATE: f64 = 12.5;
/// Default Euler steps and guidance (heartlib's `detokenize` defaults).
pub const DEFAULT_STEPS: usize = 10;
pub const DEFAULT_GUIDANCE: f64 = 1.25;

// ── shared pieces ───────────────────────────────────────────────────

/// heartlib's RMSNorm: `weight · x / sqrt(mean(x²) + eps)`.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps: 1e-6,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        x.broadcast_div(&(var + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)
    }
}

/// LayerNorm without affine parameters.
fn layer_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)
}

/// `ProjectLayer`: a k-wide "same" Conv1d over time scaled by `k^-0.5`,
/// then a Linear. Input and output are `(b, t, c)`.
struct ProjectLayer {
    conv_w: Tensor,
    conv_b: Tensor,
    k: usize,
    lin: Linear,
}

impl ProjectLayer {
    fn new(cin: usize, cout: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv_w: vb.get((cout, cin, k), "ffn_1.weight")?,
            conv_b: vb.get(cout, "ffn_1.bias")?,
            k,
            lin: linear(cout, cout, vb.pp("ffn_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x
            .transpose(1, 2)?
            .contiguous()?
            .conv1d(&self.conv_w, self.k / 2, 1, 1, 1)?
            .broadcast_add(&self.conv_b.reshape((1, (), 1))?)?
            .transpose(1, 2)?;
        let y = (y * (self.k as f64).powf(-0.5))?;
        self.lin.forward(&y)
    }
}

/// RoPE tables for the estimator: base 10 000 over the full head, f32,
/// rotated as interleaved pairs.
fn rope_tables(head_dim: usize, len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for p in 0..len {
        for &f in &inv {
            let a = p as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, half), device)?,
        Tensor::from_vec(sin, (len, half), device)?,
    ))
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(dim: usize, heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = heads * head_dim;
        Ok(Self {
            q: linear_no_bias(dim, inner, vb.pp("q_proj"))?,
            k: linear_no_bias(dim, inner, vb.pp("k_proj"))?,
            v: linear_no_bias(dim, inner, vb.pp("v_proj"))?,
            o: linear_no_bias(inner, dim, vb.pp("o_proj"))?,
            heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let shape = |y: Tensor| -> Result<Tensor> {
            y.reshape((b, t, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (cos, sin) = rope;
        let q = candle_nn::rotary_emb::rope_i(&shape(self.q.forward(x)?)?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope_i(&shape(self.k.forward(x)?)?, cos, sin)?;
        let v = shape(self.v.forward(x)?)?;
        let att = (q.matmul(&k.t()?)? * (1.0 / (self.head_dim as f64).sqrt()))?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.o.forward(&out)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // heartlib's LlamaMLP sizing: 4·dim, ×2/3, rounded up to 256.
        let hidden = (2 * 4 * dim) / 3;
        let hidden = 256 * hidden.div_ceil(256);
        Ok(Self {
            gate: linear_no_bias(dim, hidden, vb.pp("gate"))?,
            up: linear_no_bias(dim, hidden, vb.pp("up"))?,
            down: linear_no_bias(hidden, dim, vb.pp("down"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        self.down.forward(&(g * self.up.forward(x)?)?)
    }
}

/// A transformer block with AdaLN-single modulation.
struct Block {
    attn_norm: RmsNorm,
    attn: Attention,
    mlp_norm: RmsNorm,
    mlp: Mlp,
    table: Tensor,
    dim: usize,
}

impl Block {
    fn new(dim: usize, heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: RmsNorm::new(dim, vb.pp("attn_norm"))?,
            attn: Attention::new(dim, heads, head_dim, vb.pp("attn"))?,
            mlp_norm: RmsNorm::new(dim, vb.pp("mlp_norm"))?,
            mlp: Mlp::new(dim, vb.pp("mlp"))?,
            table: vb.get((6, dim), "scale_shift_table")?,
            dim,
        })
    }

    /// `tmod` is `(b, 6·dim)`.
    fn forward(&self, x: &Tensor, tmod: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        let b = x.dim(0)?;
        let m = self
            .table
            .unsqueeze(0)?
            .broadcast_add(&tmod.reshape((b, 6, self.dim))?)?;
        let part = |i: usize| m.narrow(1, i, 1);
        let (shift_a, scale_a, gate_a) = (part(0)?, part(1)?, part(2)?);
        let (shift_m, scale_m, gate_m) = (part(3)?, part(4)?, part(5)?);

        let n = self.attn_norm.forward(x)?;
        let n = n
            .broadcast_mul(&(scale_a + 1.0)?)?
            .broadcast_add(&shift_a)?;
        let h = self.attn.forward(&n, rope)?.broadcast_mul(&gate_a)?;
        let x = (x + h)?;
        let n = self.mlp_norm.forward(&x)?;
        let n = n
            .broadcast_mul(&(scale_m + 1.0)?)?
            .broadcast_add(&shift_m)?;
        let h = self.mlp.forward(&n)?.broadcast_mul(&gate_m)?;
        x + h
    }
}

/// AdaLN-single: sinusoidal timestep features → MLP → `(modulation,
/// embedding)`.
struct AdaLnSingle {
    linear_1: Linear,
    linear_2: Linear,
    linear: Linear,
}

const FLOW_T_SIZE: usize = 512;

impl AdaLnSingle {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let te = vb.pp("emb.timestep_embedder");
        Ok(Self {
            linear_1: linear(FLOW_T_SIZE, dim, te.pp("linear_1"))?,
            linear_2: linear(dim, dim, te.pp("linear_2"))?,
            linear: linear(dim, 6 * dim, vb.pp("linear"))?,
        })
    }

    fn forward(&self, t: &[f32], device: &Device) -> Result<(Tensor, Tensor)> {
        let half = FLOW_T_SIZE / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp())
            .collect();
        let mut feats = Vec::with_capacity(t.len() * FLOW_T_SIZE);
        for &tt in t {
            let args: Vec<f32> = freqs.iter().map(|f| tt * f * 1000.0).collect();
            feats.extend(args.iter().map(|a| a.cos()));
            feats.extend(args.iter().map(|a| a.sin()));
        }
        let x = Tensor::from_vec(feats, (t.len(), FLOW_T_SIZE), device)?;
        let emb = self
            .linear_2
            .forward(&candle_nn::ops::silu(&self.linear_1.forward(&x)?)?)?;
        let tmod = self.linear.forward(&candle_nn::ops::silu(&emb)?)?;
        Ok((tmod, emb))
    }
}

/// The flow-matching velocity estimator (heartlib `LlamaTransformer`).
struct Estimator {
    proj_in: ProjectLayer,
    blocks: Vec<Block>,
    blocks_2: Vec<Block>,
    connection_proj: ProjectLayer,
    table: Tensor,
    table_2: Tensor,
    proj_out: ProjectLayer,
    adaln: AdaLnSingle,
    adaln_2: AdaLnSingle,
    head_dim: usize,
}

impl Estimator {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let heads = cfg.num_attention_heads;
        let inner = heads * cfg.attention_head_dim;
        let inner2 = inner * 2;
        let blocks = (0..cfg.num_layers)
            .map(|i| {
                Block::new(
                    inner,
                    heads,
                    cfg.attention_head_dim,
                    vb.pp(format!("transformer_blocks.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let blocks_2 = (0..cfg.num_layers_2)
            .map(|i| {
                Block::new(
                    inner2,
                    heads,
                    cfg.attention_head_dim * 2,
                    vb.pp(format!("transformer_blocks_2.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_in: ProjectLayer::new(cfg.in_channels, inner, 3, vb.pp("proj_in"))?,
            blocks,
            blocks_2,
            connection_proj: ProjectLayer::new(
                cfg.in_channels + inner,
                inner2,
                3,
                vb.pp("connection_proj"),
            )?,
            table: vb.get((2, inner), "scale_shift_table")?,
            table_2: vb.get((2, inner2), "scale_shift_table_2")?,
            proj_out: ProjectLayer::new(inner2, cfg.out_channels, 3, vb.pp("proj_out"))?,
            adaln: AdaLnSingle::new(inner, vb.pp("adaln_single"))?,
            adaln_2: AdaLnSingle::new(inner2, vb.pp("adaln_single_2"))?,
            head_dim: cfg.attention_head_dim,
        })
    }

    fn modulate_out(s: &Tensor, table: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let m = table.unsqueeze(0)?.broadcast_add(&emb.unsqueeze(1)?)?;
        let shift = m.narrow(1, 0, 1)?;
        let scale = m.narrow(1, 1, 1)?;
        layer_norm(s, 1e-6)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)
    }

    /// `x` is `(b, t, in_channels)`, `t` one timestep per row.
    fn forward(&self, x: &Tensor, t: &[f32]) -> Result<Tensor> {
        let dev = x.device();
        let len = x.dim(1)?;
        let rope = rope_tables(self.head_dim, len, dev)?;
        let rope_2 = rope_tables(self.head_dim * 2, len, dev)?;

        let mut s = self.proj_in.forward(x)?;
        let (tmod, emb) = self.adaln.forward(t, dev)?;
        for blk in &self.blocks {
            s = blk.forward(&s, &tmod, &rope)?;
        }
        let s = Self::modulate_out(&s, &self.table, &emb)?;

        let mut y = self
            .connection_proj
            .forward(&Tensor::cat(&[x, &s], D::Minus1)?)?;
        let (tmod_2, emb_2) = self.adaln_2.forward(t, dev)?;
        for blk in &self.blocks_2 {
            y = blk.forward(&y, &tmod_2, &rope_2)?;
        }
        let y = Self::modulate_out(&y, &self.table_2, &emb_2)?;
        self.proj_out.forward(&y)
    }
}

// ── scalar decoder ──────────────────────────────────────────────────

/// A weight-normalised kernel: `g · v / ‖v‖`, the norm taken per slice of
/// dim 0 (torch `weight_norm(dim=0)`).
fn weight_norm(vb: &VarBuilder, shape: (usize, usize, usize)) -> Result<Tensor> {
    let g = vb.get((shape.0, 1, 1), "parametrizations.weight.original0")?;
    let v = vb.get(shape, "parametrizations.weight.original1")?;
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    v.broadcast_mul(&g.broadcast_div(&norm)?)
}

fn prelu(x: &Tensor, a: &Tensor) -> Result<Tensor> {
    // max(0, x) + a · min(0, x), with `a` a single learned slope.
    let pos = x.relu()?;
    let neg = (x - &pos)?;
    pos + neg.broadcast_mul(a)
}

/// A Conv1d, causal (left-padded) or "same".
struct Conv {
    w: Tensor,
    b: Tensor,
    k: usize,
    dilation: usize,
    causal: bool,
}

impl Conv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let span = self.dilation * (self.k - 1);
        let (x, pad) = if self.causal {
            (x.pad_with_zeros(2, span, 0)?, 0)
        } else {
            (x.clone(), span / 2)
        };
        x.conv1d(&self.w, pad, 1, self.dilation, 1)?
            .broadcast_add(&self.b.reshape((1, (), 1))?)
    }
}

struct ResidualUnit {
    conv1: Conv,
    conv2: Conv,
    act1: Tensor,
    act2: Tensor,
}

impl ResidualUnit {
    fn new(c: usize, k: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: Conv {
                w: weight_norm(&vb.pp("conv1"), (c, c, k))?,
                b: vb.get(c, "conv1.bias")?,
                k,
                dilation,
                causal: true,
            },
            conv2: Conv {
                w: weight_norm(&vb.pp("conv2"), (c, c, 1))?,
                b: vb.get(c, "conv2.bias")?,
                k: 1,
                dilation: 1,
                causal: true,
            },
            act1: vb.get(1, "activation1.weight")?,
            act2: vb.get(1, "activation2.weight")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = prelu(&self.conv1.forward(x)?, &self.act1)?;
        let y = prelu(&self.conv2.forward(&y)?, &self.act2)?;
        y + x
    }
}

/// `conv_transpose1d` one batch row at a time. candle's CPU kernel (at the
/// pinned fork rev) gets every row after the first wrong when the batch is
/// larger than one; `conv_transpose_rows_match_a_direct_reference` pins it.
fn conv_transpose1d_rows(x: &Tensor, w: &Tensor, stride: usize) -> Result<Tensor> {
    let b = x.dim(0)?;
    if b == 1 {
        return x.conv_transpose1d(w, 0, 0, stride, 1, 1);
    }
    let rows = (0..b)
        .map(|r| x.narrow(0, r, 1)?.conv_transpose1d(w, 0, 0, stride, 1, 1))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&rows, 0)
}

/// Causal transposed-conv upsampling (kernel = 2 · stride) and five
/// dilated residual units.
struct UpBlock {
    up_w: Tensor,
    up_b: Tensor,
    stride: usize,
    units: Vec<ResidualUnit>,
}

impl UpBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = conv_transpose1d_rows(x, &self.up_w, self.stride)?
            .broadcast_add(&self.up_b.reshape((1, (), 1))?)?;
        let len = y.dim(2)? - self.stride;
        let mut y = y.narrow(2, 0, len)?;
        for u in &self.units {
            y = u.forward(&y)?;
        }
        Ok(y)
    }
}

/// The ScalarModel decoder: 128-dim latents at 25 Hz → mono samples.
pub struct ScalarDecoder {
    head: Conv,
    blocks: Vec<UpBlock>,
    post: Conv,
    post_act: Tensor,
    num_samples: usize,
    out: Conv,
    /// Output samples per latent frame.
    pub hop: usize,
    /// Latent frames of history any output sample can depend on.
    receptive_frames: usize,
}

const RES_DILATIONS: [usize; 5] = [1, 3, 5, 7, 9];

impl ScalarDecoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_up = cfg.upsample_factors.len();
        let top = cfg.init_channel << n_up;
        let head = Conv {
            w: weight_norm(
                &vb.pp("0"),
                (top, cfg.latent_hidden_dim, cfg.delay_kernel_size),
            )?,
            b: vb.get(top, "0.bias")?,
            k: cfg.delay_kernel_size,
            dilation: 1,
            causal: false,
        };
        let mut blocks = Vec::with_capacity(n_up);
        for (i, (&stride, &k)) in cfg
            .upsample_factors
            .iter()
            .zip(&cfg.upsample_kernel_sizes)
            .enumerate()
        {
            let cin = cfg.init_channel << (n_up - i);
            let cout = cfg.init_channel << (n_up - i - 1);
            let bvb = vb.pp(format!("{}", i + 1));
            let units = RES_DILATIONS
                .iter()
                .enumerate()
                .map(|(j, &d)| {
                    ResidualUnit::new(cout, cfg.res_kernel_size, d, bvb.pp(format!("convs.{j}")))
                })
                .collect::<Result<Vec<_>>>()?;
            blocks.push(UpBlock {
                up_w: weight_norm(&bvb.pp("up_conv.layer"), (cin, cout, k))?,
                up_b: bvb.get(cout, "up_conv.layer.bias")?,
                stride,
                units,
            });
        }
        let c = cfg.init_channel;
        let p = n_up + 1;
        let post = Conv {
            w: vb.get((c, c, cfg.default_kernel_size), &format!("{p}.conv.weight"))?,
            b: vb.get(c, &format!("{p}.conv.bias"))?,
            k: cfg.default_kernel_size,
            dilation: 1,
            causal: true,
        };
        let post_act = vb.get(1, &format!("{p}.activation.weight"))?;
        let o = n_up + 2;
        let out = Conv {
            w: weight_norm(&vb.pp(format!("{o}")), (1, c, cfg.default_kernel_size))?,
            b: vb.get(1, &format!("{o}.bias"))?,
            k: cfg.default_kernel_size,
            dilation: 1,
            causal: true,
        };
        let hop = cfg.upsample_factors.iter().product::<usize>() * cfg.num_samples;
        // How far back the causal stack reaches, in latent frames: each
        // transposed conv looks ceil(k / stride) frames back at its input
        // rate, each block's residual units sum(dilation · (k − 1)) samples
        // at its output rate, the post-processor (k − 1) samples at the
        // full rate, and the head conv half its kernel.
        let mut rf = (cfg.delay_kernel_size / 2) as f64;
        let mut rate = 1usize;
        for (&s, &k) in cfg.upsample_factors.iter().zip(&cfg.upsample_kernel_sizes) {
            rf += k.div_ceil(s) as f64 / rate as f64;
            rate *= s;
            let res: usize = RES_DILATIONS
                .iter()
                .map(|d| d * (cfg.res_kernel_size - 1))
                .sum();
            rf += res as f64 / rate as f64;
        }
        rf += 2.0 * (cfg.default_kernel_size - 1) as f64 / hop as f64;
        let receptive_frames = rf.ceil() as usize + 1;
        Ok(Self {
            head,
            blocks,
            post,
            post_act,
            num_samples: cfg.num_samples,
            out,
            hop,
            receptive_frames,
        })
    }

    /// `x` `(b, 128, t)` → `(b, t · hop)`.
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        // The scalar quantiser. candle rounds half away from zero where
        // torch rounds half to even; the two only differ on exact ties.
        let x = ((x * 9.0)?.round()? / 9.0)?;
        let mut y = self.head.forward(&x)?;
        for b in &self.blocks {
            y = b.forward(&y)?;
        }
        // Repeat every sample `num_samples` times along time.
        let (b, c, t) = y.dims3()?;
        let y = y
            .unsqueeze(3)?
            .broadcast_as((b, c, t, self.num_samples))?
            .reshape((b, c, t * self.num_samples))?;
        let y = prelu(&self.post.forward(&y)?, &self.post_act)?;
        self.out.forward(&y)?.squeeze(1)
    }

    /// Latent frames of history any output sample can depend on.
    pub fn receptive_frames(&self) -> usize {
        self.receptive_frames
    }

    /// The same as [`Self::decode`], in chunks of `chunk` latent frames, so
    /// peak memory follows the chunk rather than the song. Every layer is
    /// causal except the first (a 5-wide "same" conv), so the stack's
    /// receptive field of history and 2 frames of look-ahead make each
    /// chunk's samples equal to a whole-song decode.
    pub fn decode_chunked(&self, x: &Tensor, chunk: usize) -> Result<Tensor> {
        let t = x.dim(2)?;
        if t <= chunk {
            return self.decode(x);
        }
        let left = self.receptive_frames;
        let right = 2;
        let mut parts = Vec::new();
        let mut a = 0;
        while a < t {
            let b = (a + chunk).min(t);
            let lo = a.saturating_sub(left);
            let hi = (b + right).min(t);
            let y = self.decode(&x.narrow(2, lo, hi - lo)?)?;
            parts.push(y.narrow(1, (a - lo) * self.hop, (b - a) * self.hop)?);
            a = b;
        }
        Tensor::cat(&parts, 1)
    }
}

// ── the decoder as a whole ──────────────────────────────────────────

/// Residual-VQ lookup: summed codebook vectors, projected to `dim`.
struct Rvq {
    codebooks: Vec<Tensor>,
    project_out: Linear,
}

impl Rvq {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let codebooks = (0..cfg.num_quantizers)
            .map(|i| {
                vb.get(
                    (1, cfg.codebook_size, cfg.codebook_dim),
                    &format!("layers.{i}._codebook.embed"),
                )?
                .squeeze(0)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            codebooks,
            project_out: linear(cfg.codebook_dim, cfg.dim, vb.pp("project_out"))?,
        })
    }

    /// `codes` `(quantizers, t)` → `(1, t, dim)`.
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum: Option<Tensor> = None;
        for (q, book) in self.codebooks.iter().enumerate() {
            let e = book.index_select(&codes.i(q)?.contiguous()?, 0)?;
            sum = Some(match sum {
                None => e,
                Some(s) => (s + e)?,
            });
        }
        let sum = sum.ok_or_else(|| candle_core::Error::Msg("no codebooks".into()))?;
        self.project_out.forward(&sum)?.unsqueeze(0)
    }
}

/// What one flow solve produced, for parity checks.
pub struct FlowTrace {
    pub rvq: Tensor,
    pub mu: Tensor,
    pub t_span: Vec<f32>,
    pub step1_v: Tensor,
    pub step5_x: Tensor,
}

/// torch's `linspace` for f32: forward from `start` in the first half,
/// back from `end` in the second, so both ends are exact.
pub fn torch_linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let halfway = steps / 2;
    (0..steps)
        .map(|i| {
            if i < halfway {
                start + step * i as f32
            } else {
                end - step * (steps - i - 1) as f32
            }
        })
        .collect()
}

/// The loaded codec decoder.
pub struct HeartCodec {
    pub config: Config,
    rvq: Rvq,
    cond_feature_emb: Linear,
    estimator: Estimator,
    pub scalar: ScalarDecoder,
    device: Device,
}

/// Overlap between segments, in latent frames (52 codes, ×2).
const SEGMENT_CODES: usize = 372;
const HOP_CODES: usize = SEGMENT_CODES / 93 * 80;
const OVERLAP_CODES: usize = SEGMENT_CODES - HOP_CODES;

impl HeartCodec {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let fm = vb.pp("flow_matching");
        Ok(Self {
            rvq: Rvq::new(&config, fm.pp("vq_embed"))?,
            cond_feature_emb: linear(config.dim, config.dim, fm.pp("cond_feature_emb"))?,
            estimator: Estimator::new(&config, fm.pp("estimator"))?,
            scalar: ScalarDecoder::new(&config, vb.pp("scalar_model.decoder"))?,
            device: vb.device().clone(),
            config,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The flow condition `mu` `(1, 2t, dim)` for one segment's codes.
    fn condition(&self, codes: &Tensor) -> Result<(Tensor, Tensor)> {
        let rvq = self.rvq.decode(codes)?;
        let cond = self.cond_feature_emb.forward(&rvq)?;
        let (b, t, d) = cond.dims3()?;
        let mu = cond
            .unsqueeze(2)?
            .broadcast_as((b, t, 2, d))?
            .reshape((b, t * 2, d))?;
        Ok((rvq, mu))
    }

    /// Euler from `noise` at t = 0 to latents at t = 1. `incontext` holds a
    /// previous segment's tail latents and how many frames of it to keep.
    fn solve(
        &self,
        noise: &Tensor,
        incontext: Option<(&Tensor, usize)>,
        mu: &Tensor,
        steps: usize,
        guidance: f64,
        mut trace: Option<&mut FlowTrace>,
    ) -> Result<Tensor> {
        let t_span = torch_linspace(0.0, 1.0, steps + 1);
        let zeros = noise.zeros_like()?;
        let (ic, ic_len) = match incontext {
            Some((ic, n)) => (ic.clone(), n),
            None => (zeros.clone(), 0),
        };
        let cfg = guidance > 1.0;
        let mu_rows = if cfg {
            Tensor::cat(&[&mu.zeros_like()?, mu], 0)?
        } else {
            mu.clone()
        };
        let mut x = noise.clone();
        let mut t = t_span[0];
        let mut dt = t_span[1] - t_span[0];
        for step in 1..t_span.len() {
            if ic_len > 0 {
                let a = 1.0 - (1.0 - 1e-6f32) * t;
                let head = ((noise.narrow(1, 0, ic_len)? * a as f64)?
                    + (ic.narrow(1, 0, ic_len)? * t as f64)?)?;
                let tail = x.narrow(1, ic_len, x.dim(1)? - ic_len)?;
                x = Tensor::cat(&[head, tail], 1)?;
            }
            let v = if cfg {
                let inp = Tensor::cat(
                    &[
                        &Tensor::cat(&[&x, &x], 0)?,
                        &Tensor::cat(&[&ic, &ic], 0)?,
                        &mu_rows,
                    ],
                    2,
                )?;
                let out = self.estimator.forward(&inp, &[t, t])?;
                let vu = out.i(0..1)?;
                let vc = out.i(1..2)?;
                (&vu + ((vc - &vu)? * guidance)?)?
            } else {
                let inp = Tensor::cat(&[&x, &ic, &mu_rows], 2)?;
                self.estimator.forward(&inp, &[t])?
            };
            x = (x + (&v * dt as f64)?)?;
            t += dt;
            if step < t_span.len() - 1 {
                dt = t_span[step + 1] - t;
            }
            if let Some(tr) = trace.as_deref_mut() {
                if step == 1 {
                    tr.step1_v = v.clone();
                }
                if step == 5 {
                    tr.step5_x = x.clone();
                }
            }
        }
        if ic_len > 0 {
            let tail = x.narrow(1, ic_len, x.dim(1)? - ic_len)?;
            x = Tensor::cat(&[ic.narrow(1, 0, ic_len)?, tail], 1)?;
        }
        Ok(x)
    }

    /// One segment's latents `(1, 2·segment, out_channels)`, and its trace.
    pub fn segment_latents(
        &self,
        codes: &Tensor,
        noise: &Tensor,
        incontext: Option<(&Tensor, usize)>,
        steps: usize,
        guidance: f64,
    ) -> Result<(Tensor, FlowTrace)> {
        let (rvq, mu) = self.condition(codes)?;
        let mut trace = FlowTrace {
            rvq,
            mu: mu.clone(),
            t_span: torch_linspace(0.0, 1.0, steps + 1),
            step1_v: Tensor::zeros(1, DType::F32, &self.device)?,
            step5_x: Tensor::zeros(1, DType::F32, &self.device)?,
        };
        let lat = self.solve(noise, incontext, &mu, steps, guidance, Some(&mut trace))?;
        Ok((lat, trace))
    }

    /// Latents `(1, t, 256)` → stereo `(2, t · hop)`: the first half of
    /// each latent is the left channel, the second the right.
    pub fn latents_to_audio(&self, lat: &Tensor, chunk: Option<usize>) -> Result<Tensor> {
        let (_, t, c) = lat.dims3()?;
        let x = lat
            .reshape((t, 2, c / 2))?
            .permute((1, 2, 0))?
            .contiguous()?;
        match chunk {
            Some(n) => self.scalar.decode_chunked(&x, n),
            None => self.scalar.decode(&x),
        }
    }

    /// Decode a whole song, as heartlib's `detokenize`. `codes` is
    /// `(quantizers, frames)`; `noise(len)` supplies each segment's
    /// initial noise `(1, len, out_channels)`.
    pub fn detokenize(
        &self,
        codes: &Tensor,
        steps: usize,
        guidance: f64,
        noise: &mut dyn FnMut(usize) -> Result<Tensor>,
        chunk: Option<usize>,
    ) -> Result<Tensor> {
        let frames = codes.dim(1)?;
        let target_len = (frames as f64 / CODE_RATE * self.config.sample_rate as f64) as usize;
        let codes = tile_codes(codes)?;
        let latent_len = 2 * SEGMENT_CODES;
        let overlap_frames = 2 * OVERLAP_CODES;

        let mut latents: Vec<Tensor> = Vec::new();
        let mut start = 0;
        while start + HOP_CODES <= codes.dim(1)? {
            let seg = codes.narrow(1, start, SEGMENT_CODES)?;
            let n = noise(latent_len)?;
            let lat = match latents.last() {
                None => self.segment_latents(&seg, &n, None, steps, guidance)?.0,
                Some(prev) => {
                    let tail = prev.narrow(1, latent_len - overlap_frames, overlap_frames)?;
                    let ic = tail.pad_with_zeros(1, 0, latent_len - overlap_frames)?;
                    self.segment_latents(&seg, &n, Some((&ic, overlap_frames)), steps, guidance)?
                        .0
                }
            };
            latents.push(lat);
            start += HOP_CODES;
        }

        let seg_samples = (SEGMENT_SECS * self.config.sample_rate as f64) as usize;
        let hop_samples = seg_samples / 93 * 80;
        let ovl = seg_samples - hop_samples;
        let mut out: Option<Tensor> = None;
        for lat in &latents {
            let wav = self.latents_to_audio(lat, chunk)?;
            let wav = wav.narrow(1, 0, wav.dim(1)?.min(seg_samples))?;
            out = Some(match out {
                None => wav,
                Some(prev) => crossfade(&prev, &wav, ovl)?,
            });
        }
        let out = out.ok_or_else(|| candle_core::Error::Msg("no segments".into()))?;
        out.narrow(1, 0, target_len.min(out.dim(1)?))
    }
}

/// Repeat the codes until they fill whole segments, as heartlib does: by
/// doubling, then trimming to the length the segment hop needs.
pub fn tile_codes(codes: &Tensor) -> Result<Tensor> {
    let mut c = codes.clone();
    if c.dim(1)? < SEGMENT_CODES {
        while c.dim(1)? < SEGMENT_CODES {
            c = Tensor::cat(&[&c, &c], 1)?;
        }
        c = c.narrow(1, 0, SEGMENT_CODES)?;
    }
    let len = c.dim(1)?;
    if !(len - OVERLAP_CODES).is_multiple_of(HOP_CODES) {
        let want = (len - OVERLAP_CODES).div_ceil(HOP_CODES) * HOP_CODES + OVERLAP_CODES;
        while c.dim(1)? < want {
            c = Tensor::cat(&[&c, &c], 1)?;
        }
        c = c.narrow(1, 0, want)?;
    }
    Ok(c)
}

/// Linear crossfade of `next`'s first `ovl` samples over `prev`'s last.
fn crossfade(prev: &Tensor, next: &Tensor, ovl: usize) -> Result<Tensor> {
    if ovl == 0 {
        return Tensor::cat(&[prev, next], 1);
    }
    let fade_in: Vec<f32> = (0..ovl)
        .map(|i| (i as f64 / (ovl - 1) as f64) as f32)
        .collect();
    let fade_out: Vec<f32> = fade_in.iter().map(|w| 1.0 - w).collect();
    let dev = prev.device();
    let fin = Tensor::from_vec(fade_in, (1, ovl), dev)?;
    let fout = Tensor::from_vec(fade_out, (1, ovl), dev)?;
    let keep = prev.dim(1)? - ovl;
    let mixed = (prev.narrow(1, keep, ovl)?.broadcast_mul(&fout)?
        + next.narrow(1, 0, ovl)?.broadcast_mul(&fin)?)?;
    Tensor::cat(
        &[
            prev.narrow(1, 0, keep)?,
            mixed,
            next.narrow(1, ovl, next.dim(1)? - ovl)?,
        ],
        1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn linspace_matches_torch() {
        let t = torch_linspace(0.0, 1.0, 11);
        assert_eq!(t.len(), 11);
        assert_eq!(t[0], 0.0);
        assert_eq!(t[10], 1.0);
        // torch computes the second half back from the end.
        assert_eq!(t[7], 1.0 - 0.1f32 * 3.0);
        assert_eq!(t[3], 0.1f32 * 3.0);
    }

    #[test]
    fn tiling_fills_segments() {
        let d = Device::Cpu;
        let codes = Tensor::arange(0u32, 151, &d).unwrap().unsqueeze(0).unwrap();
        let t = tile_codes(&codes).unwrap();
        assert_eq!(t.dims(), &[1, 372]);
        let v = t.i(0).unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(v[151], 0);
        assert_eq!(v[371], (371 % 151) as u32);
        // Longer than a segment: padded to 52 + k·320.
        let long = Tensor::arange(0u32, 400, &d).unwrap().unsqueeze(0).unwrap();
        assert_eq!(tile_codes(&long).unwrap().dims(), &[1, 692]);
        let exact = Tensor::arange(0u32, 692, &d).unwrap().unsqueeze(0).unwrap();
        assert_eq!(tile_codes(&exact).unwrap().dims(), &[1, 692]);
    }

    #[test]
    fn crossfade_is_linear_and_keeps_length() {
        let d = Device::Cpu;
        let a = Tensor::ones((2, 10), DType::F32, &d).unwrap();
        let b = Tensor::zeros((2, 10), DType::F32, &d).unwrap();
        let out = crossfade(&a, &b, 5).unwrap();
        assert_eq!(out.dims(), &[2, 15]);
        let v = out.i(0).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&v[..5], &[1.0; 5]);
        assert_eq!(&v[5..10], &[1.0, 0.75, 0.5, 0.25, 0.0]);
        assert_eq!(&v[10..], &[0.0; 5]);
    }

    /// A randomly initialised decoder in the real topology, scaled down.
    fn small_decoder() -> ScalarDecoder {
        let cfg = Config {
            dim: 8,
            codebook_size: 4,
            codebook_dim: 2,
            num_quantizers: 1,
            attention_head_dim: 2,
            in_channels: 4,
            num_attention_heads: 1,
            num_layers: 1,
            num_layers_2: 1,
            out_channels: 2,
            num_samples: 2,
            upsample_factors: vec![2, 3],
            upsample_kernel_sizes: vec![4, 6],
            latent_hidden_dim: 3,
            delay_kernel_size: 5,
            default_kernel_size: 7,
            init_channel: 2,
            res_kernel_size: 7,
            sample_rate: 48_000,
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        // `get` creates zeros, and weight norm of a zero kernel is 0/0: build
        // once to register the variables, randomise them, then build again.
        ScalarDecoder::new(&cfg, vb.clone()).unwrap();
        for var in vm.all_vars() {
            let r = Tensor::randn(0f32, 0.5, var.shape(), &Device::Cpu).unwrap();
            var.set(&r).unwrap();
        }
        ScalarDecoder::new(&cfg, vb).unwrap()
    }

    /// A direct transposed convolution: `out[b, o, i·s + k] += x[b, c, i] · w[c, o, k]`.
    fn conv_transpose_reference(x: &Tensor, w: &Tensor, s: usize) -> Vec<f32> {
        let (b, cin, l) = x.dims3().unwrap();
        let (_, cout, k) = w.dims3().unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let wv = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let lout = (l - 1) * s + k;
        let mut out = vec![0f32; b * cout * lout];
        for bi in 0..b {
            for c in 0..cin {
                for i in 0..l {
                    let xi = xv[(bi * cin + c) * l + i];
                    for o in 0..cout {
                        for kk in 0..k {
                            out[(bi * cout + o) * lout + i * s + kk] +=
                                xi * wv[(c * cout + o) * k + kk];
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn conv_transpose_rows_match_a_direct_reference() {
        let d = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (3, 6, 50), &d).unwrap();
        let w = Tensor::randn(0f32, 1.0, (6, 4, 8), &d).unwrap();
        let got = conv_transpose1d_rows(&x, &w, 4)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = conv_transpose_reference(&x, &w, 4);
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-4, "{worst}");
    }

    #[test]
    fn chunked_decode_equals_whole_decode() {
        let dec = small_decoder();
        assert_eq!(dec.hop, 12);
        // Long enough for several chunks beyond the receptive field.
        let t = dec.receptive_frames() * 3 + 5;
        let x = Tensor::randn(0f32, 0.3, (2, 3, t), &Device::Cpu).unwrap();
        let whole = dec.decode(&x).unwrap();
        assert_eq!(whole.dims(), &[2, t * 12]);
        let chunked = dec.decode_chunked(&x, 7).unwrap();
        let diff = (whole - chunked)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "{diff}");
    }
}
