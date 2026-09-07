//! `qwen4_exp` — the architecture behind `Qwen/Qwen3.8-Flash-Next`.
//!
//! A Qwen 4 experimental generation rather than a Qwen3 variant. The
//! hybrid skeleton (36 linear-attention layers to 12 full-attention,
//! `full_attention_interval: 4`) and the linear-attention tensor layout
//! are shared with [`super::qwen3_5`], so much of that implementation
//! carries over. What is new is the residual structure — four streams
//! joined by hyper-connections, with no layernorms of the usual kind —
//! plus QSA sparse attention and PLE hashed n-gram embeddings.
//!
//! The written specification lives at `doc/qwen4_exp-port-spec.md`;
//! it is the reference for every dimension and dataflow here, and it
//! records what was measured from the checkpoint rather than inferred.
//!
//! Built in dependency order, so each piece can be parity-tested before
//! anything stacks on it:
//! - [`config`] — the checkpoint's own `config.json`, and what its
//!   fields mean where they are not what they look like.
//! - [`hyper`] — hyper-connections, the residual structure itself.
//! - [`decoder`] — the composition: four residual streams, PLE on
//!   layer 1, and no layernorm outside the hyper-connections.
//! - [`full_attn`] — one layer in four, plus the indexer that
//!   narrows what it may attend.
//! - [`linear_attn`] — the other three layers in four, reused whole
//!   from `qwen3_5` with two settings corrected.
//! - [`model`] — embeddings in, logits out; four streams, one
//!   n-gram lookup, and no final norm.
//! - [`mtp`] — the shipped draft head: one more decoder layer, fed
//!   the target's pre-mixer stream and the next token's embedding.
//! - [`moe`] — the FFN every layer has, loaded from fused expert
//!   tensors instead of 512 modules.
//! - [`ple`] — the hashed n-gram table on layer 1: how its rows
//!   are addressed, and how the gathered rows are consumed.
//! - [`qsa`] — which blocks of the past a full-attention layer is
//!   allowed to look at.

pub mod config;
pub mod decoder;
pub mod full_attn;
pub mod hyper;
pub mod linear_attn;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod ple;
pub mod qsa;

/// What a load needs to know beyond the config and the weights.
///
/// Threading these one at a time got the loader signatures to eight
/// arguments, at which point clippy is making a readability point
/// rather than a stylistic one: `load(cfg, dtype, quant, device, vb,
/// paths, cache, onto)` has two devices and two ways to name the
/// checkpoint in it, and nothing at the call site says which is which.
///
/// They also belong together. Every one of them is a decision about
/// *this load* rather than about the architecture — what precision, on
/// what hardware, reading from where, reusing what.
pub struct LoadOptions<'a> {
    /// In-situ quantisation type (#315). `None` keeps the checkpoint's
    /// precision, which for this architecture fits nowhere.
    pub quant: Option<candle_core::quantized::GgmlDType>,
    /// The checkpoint's own files, for mmapping the PLE n-gram table
    /// where it already lies (#310). Empty falls back to loading it
    /// through the VarBuilder, which puts 102.4 GB wherever that
    /// points.
    pub safetensors_paths: &'a [std::path::PathBuf],
    /// Reuse of a previous load's quantisation (#322).
    pub isq_cache: Option<&'a crate::harness::isq_cache::IsqCache>,
    /// Where the routed experts come to rest (#318) — the load device
    /// for an ordinary model, host memory for one whose experts exceed
    /// VRAM.
    pub experts_onto: &'a candle_core::Device,
    /// A VarBuilder rooted at the checkpoint but bound to
    /// [`Self::experts_onto`], used to read the fused expert tensors
    /// when that is not the load device.
    ///
    /// Without it, host-resident experts still arrive *through* VRAM:
    /// `get` puts the layer's 5.03 GB fused pair on the load device,
    /// the quantiser copies it to the host, and the source is dropped
    /// — 48 times, 241 GB of PCIe traffic, against a pool that has to
    /// absorb two multi-gigabyte blocks and 1536 small ones per layer.
    /// Measured: the load reached layer 22 of 48 and ran out with 31 GB
    /// nominally free, which is fragmentation rather than occupancy —
    /// the same failure the image path trims after every job to avoid.
    ///
    /// Reading them straight into host memory removes the transit
    /// rather than managing it.
    pub experts_vb: Option<&'a candle_nn::var_builder::ShardedVarBuilder<'a>>,
}
