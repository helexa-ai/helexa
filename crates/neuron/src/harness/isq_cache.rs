//! On-disk cache for in-situ quantisation (#322).
//!
//! Quantising a model's weights at load is a pure function of the
//! checkpoint bytes, the requested type, and the code that does it.
//! None of those change between loads, and yet neuron recomputed the
//! answer every time and threw it away on unload. For `qwen4_exp` that
//! is **~32 s per decoder layer, about 26 minutes for a full load**,
//! measured on beast — paid again on every restart and every eviction
//! that is later reversed.
//!
//! So the quantised tensors are written out once and read back after.
//!
//! ## The key must include our own version
//!
//! `(checkpoint revision, quant name)` is not enough. `bf1e80ed`
//! changed what `quant = "q4k"` *means* for a 640-wide row — k-quants
//! cannot block it, so it now falls back to a 32-block type — and an
//! artifact written before that commit and read after it would be
//! silently wrong in a way no checksum of the inputs could catch.
//! [`QUANTISER_VERSION`] is part of the path, and bumping it orphans
//! every stale artifact rather than misreading one.
//!
//! ## What is cached, and what is not
//!
//! One GGUF per layer, holding that layer's quantised experts. GGUF
//! because it stores a **type per tensor**, which is exactly what the
//! per-row fallback produces — a layer whose `down_proj` is `Q5_0`
//! while its `gate`/`up` are `Q4K` is representable without inventing
//! a container for it — and because neuron already reads the format.
//!
//! Per layer rather than per model because it makes the cache
//! **incrementally useful**: a load that runs out of VRAM at layer 16
//! has still written sixteen layers, and the next attempt starts
//! there. On hardware where the model does not yet fit at all (#318),
//! that is the difference between the cache being useless and being
//! the thing that makes iteration bearable.
//!
//! ## Where it lives
//!
//! Inside the source's own cache root — `<hf cache>/neuron-isq/…` —
//! beside the weights it is derived from, and not behind a config key
//! naming a path.
//!
//! That is deliberate and it is the fleet's existing design. beast
//! already keeps its hot model weights on NVMe by symlinking them out
//! of the spinning `/archive3` cache into `/archive1`, with no config
//! mapping paths to mounts. An operator who wants these artifacts on
//! faster storage does the same thing to one directory. A second,
//! differently-shaped way to express the same intent would be a second
//! place to get it wrong, and the first one already works.
//!
//! Size is the thing to know before symlinking: an artifact is roughly
//! the served size of the model's quantised weights — ~73 GB for
//! `qwen4_exp` at q4k.
//!
//! ## It is an optimisation, and never load-bearing
//!
//! Every failure here is swallowed: a miss, a corrupt file, a full
//! disk, a directory nobody can write. The load proceeds by
//! quantising, exactly as it did before this module existed. A cache
//! that can fail a load is worse than no cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::Device;
use candle_core::quantized::{QTensor, gguf_file};

/// Bump when the *meaning* of a quant name changes: a different
/// fallback table, a different rounding, a different layout. Artifacts
/// written by an older quantiser are then ignored rather than
/// misread, and get overwritten in place as layers are re-quantised.
///
/// 1: the initial format. Per-layer GGUF, tensors named
///    `e{index}.{gate,up,down}`, llama.cpp's `tensor_type_fallback`
///    table for rows that do not divide the requested block size.
pub const QUANTISER_VERSION: u32 = 1;

/// Refuse a write that would leave the filesystem below this fraction
/// of itself free.
///
/// A proportion rather than a fixed size, because the cache lives
/// wherever the weights do and that varies: `/archive3` on beast is
/// 7.3 TB, a symlinked NVMe target might be 900 GB, and a developer's
/// checkout is neither. Two percent of a 7 TB volume is 146 GB of
/// headroom and of a 32 GB one is 640 MB — in both cases "do not be
/// the reason this filesystem filled up", which is the actual rule.
/// An artifact for one 180 B model is ~73 GB, so this is not
/// hypothetical.
const MIN_FREE_FRACTION: f64 = 0.02;

/// A cache scoped to one (model, revision, quant, quantiser version).
pub struct IsqCache {
    dir: PathBuf,
}

impl IsqCache {
    /// `revision` should identify the checkpoint bytes — the snapshot
    /// directory name, for an hf-hub cache. `None` disables the cache
    /// rather than guessing, because a wrong revision is the one
    /// failure mode that produces a confidently wrong model.
    pub fn new(base: &Path, model_id: &str, revision: Option<&str>, quant: &str) -> Option<Self> {
        let revision = revision?;
        let safe = |s: &str| {
            s.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                        c
                    } else {
                        '-'
                    }
                })
                .collect::<String>()
        };
        let dir = base
            .join(safe(model_id))
            .join(safe(revision))
            .join(format!("{}-v{QUANTISER_VERSION}", safe(quant)));
        Some(Self { dir })
    }

    /// Read a layer's quantised tensors onto `device`.
    ///
    /// `None` for a miss and for anything that goes wrong reading —
    /// a truncated artifact from an interrupted write reads as absent,
    /// and is rewritten by the load that follows.
    pub fn load(&self, key: &str, device: &Device) -> Option<HashMap<String, QTensor>> {
        let path = self.dir.join(format!("{key}.gguf"));
        let mut file = std::fs::File::open(&path).ok()?;
        let content = match gguf_file::Content::read(&mut file) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e,
                    "isq cache: unreadable artifact, re-quantising this layer");
                return None;
            }
        };
        let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        let mut out = HashMap::with_capacity(names.len());
        for name in names {
            match content.tensor(&mut file, &name, device) {
                Ok(t) => {
                    out.insert(name, t);
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), tensor = %name, error = %e,
                        "isq cache: artifact tensor failed to read, re-quantising this layer");
                    return None;
                }
            }
        }
        Some(out)
    }

    /// Write a layer's quantised tensors.
    ///
    /// Through a temporary file and a rename, so an interrupted write
    /// cannot leave a half-artifact that a later load would read as
    /// real. Errors are logged and swallowed.
    pub fn store(&self, key: &str, tensors: &[(&str, &QTensor)]) {
        if let Err(e) = self.try_store(key, tensors) {
            tracing::warn!(dir = %self.dir.display(), key, error = %e,
                "isq cache: could not write this layer; the load continues uncached");
        }
    }

    fn try_store(&self, key: &str, tensors: &[(&str, &QTensor)]) -> anyhow::Result<()> {
        let bytes: usize = tensors.iter().map(|(_, t)| t.storage_size_in_bytes()).sum();
        std::fs::create_dir_all(&self.dir)?;
        if let Some((total, free)) = capacity(&self.dir) {
            let floor = (total as f64 * MIN_FREE_FRACTION) as u64;
            if free.saturating_sub(bytes as u64) < floor {
                anyhow::bail!(
                    "writing {:.1} GiB would leave under the {:.1} GiB floor \
                     ({:.0}% of the filesystem)",
                    bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    floor as f64 / (1024.0 * 1024.0 * 1024.0),
                    MIN_FREE_FRACTION * 100.0,
                );
            }
        }
        let final_path = self.dir.join(format!("{key}.gguf"));
        let tmp = self.dir.join(format!(".{key}.{}.tmp", std::process::id()));
        {
            let mut f = std::fs::File::create(&tmp)?;
            gguf_file::write(&mut f, &[], tensors)?;
        }
        std::fs::rename(&tmp, &final_path)?;
        Ok(())
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// `(total, free)` bytes on the filesystem holding `path`, or `None`
/// if it cannot be determined — in which case the caller writes anyway
/// rather than refusing on a guess.
fn capacity(path: &Path) -> Option<(u64, u64)> {
    let out = std::process::Command::new("df")
        .arg("-kP")
        .arg(path)
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let mut fields = text.lines().nth(1)?.split_whitespace();
    let total = fields.nth(1)?.parse::<u64>().ok()?;
    let free = fields.nth(1)?.parse::<u64>().ok()?;
    Some((total * 1024, free * 1024))
}

/// The revision an hf-hub-cached checkpoint was resolved at, taken
/// from the snapshot directory its files live in.
///
/// `.../models--org--name/snapshots/<revision>/model-00001-of-N.safetensors`
/// — the parent directory *is* the revision, which is what makes it a
/// safe cache key: two checkpoints with different bytes cannot share
/// one. Returns `None` for any layout that is not that, so an
/// unrecognised path disables the cache instead of colliding.
pub fn revision_from_paths(paths: &[PathBuf]) -> Option<&str> {
    let first = paths.first()?;
    let dir = first.parent()?;
    let name = dir.file_name()?.to_str()?;
    let looks_like_snapshot = dir.parent()?.file_name()?.to_str()? == "snapshots";
    (looks_like_snapshot && !name.is_empty()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use candle_core::quantized::GgmlDType;

    /// A cached layer must dequantise to the same numbers the fresh
    /// quantisation produced.
    ///
    /// This is the whole contract. A cache that returns *something*
    /// plausible is worse than no cache, because the model loads, it
    /// serves, and its weights are quietly not the ones it was asked
    /// for. Round-tripping the dequantised values rather than the
    /// bytes is deliberate: it is what the model actually consumes.
    #[test]
    fn a_cached_tensor_dequantises_to_what_was_stored() {
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let cache = IsqCache::new(dir.path(), "org/model", Some("rev0"), "q4k").unwrap();

        // 256-wide so the k-quant can block it at all.
        let src = Tensor::rand(-1f32, 1f32, (8, 256), &dev).unwrap();
        let q = QTensor::quantize(&src, GgmlDType::Q4K).unwrap();
        let before: Vec<f32> = q
            .dequantize(&dev)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        cache.store("layer-0", &[("e0.gate", &q)]);
        let got = cache
            .load("layer-0", &dev)
            .expect("the artifact just written");
        let after: Vec<f32> = got["e0.gate"]
            .dequantize(&dev)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(
            before, after,
            "a cached tensor changed value on the round trip"
        );
        assert_eq!(got["e0.gate"].dtype(), GgmlDType::Q4K, "and kept its type");
    }

    /// The key separates everything that changes the answer.
    ///
    /// The quantiser version is the one that is easy to leave out and
    /// impossible to detect: `bf1e80ed` changed what `q4k` means for a
    /// 640-wide row, and an artifact written before it is wrong after
    /// it in a way no checksum of the *inputs* would reveal.
    #[test]
    fn the_key_separates_model_revision_quant_and_quantiser() {
        let root = std::path::Path::new("/cache");
        let base = IsqCache::new(root, "org/model", Some("rev0"), "q4k").unwrap();
        let others = [
            IsqCache::new(root, "org/other", Some("rev0"), "q4k").unwrap(),
            IsqCache::new(root, "org/model", Some("rev1"), "q4k").unwrap(),
            IsqCache::new(root, "org/model", Some("rev0"), "q6k").unwrap(),
        ];
        for other in &others {
            assert_ne!(base.dir(), other.dir());
        }
        assert!(
            base.dir()
                .to_str()
                .unwrap()
                .contains(&format!("v{QUANTISER_VERSION}")),
            "the quantiser version must be in the path: {}",
            base.dir().display()
        );
    }

    /// No revision, no cache. A key that might collide across
    /// checkpoints is worse than recomputing.
    #[test]
    fn an_unidentifiable_revision_disables_the_cache() {
        assert!(IsqCache::new(std::path::Path::new("/cache"), "org/model", None, "q4k").is_none());

        // An hf-hub snapshot path yields its revision; anything else
        // does not, rather than yielding a directory name that happens
        // to be there.
        let hf = PathBuf::from(
            "/cache/models--org--model/snapshots/abc123/model-00001-of-00002.safetensors",
        );
        assert_eq!(revision_from_paths(&[hf]), Some("abc123"));
        assert_eq!(
            revision_from_paths(&[PathBuf::from("/tmp/model.safetensors")]),
            None
        );
        assert_eq!(revision_from_paths(&[]), None);
    }

    /// A miss and a corrupt artifact are the same thing to the caller:
    /// re-quantise. Neither may propagate as an error.
    #[test]
    fn a_missing_or_corrupt_artifact_reads_as_absent() {
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let cache = IsqCache::new(dir.path(), "org/model", Some("rev0"), "q4k").unwrap();
        assert!(cache.load("layer-0", &dev).is_none(), "a miss");

        std::fs::create_dir_all(cache.dir()).unwrap();
        std::fs::write(cache.dir().join("layer-0.gguf"), b"not a gguf file at all").unwrap();
        assert!(cache.load("layer-0", &dev).is_none(), "a corrupt artifact");
    }
}
