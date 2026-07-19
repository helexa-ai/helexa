//! Placement feasibility check that runs before any device allocation,
//! NCCL handshake, or weight download.
//!
//! The loader path in `candle.rs` historically discovers an
//! incompatibility *after* it has already started fetching files —
//! "fetch config.json from HauhauCS/...: 404 Not Found" surfaces hours
//! after operators set `tensor_parallel = 2` on a GGUF-only repo, with
//! no hint about what's actually wrong. Preflight closes that gap:
//!
//! 1. one `repo.info()` round-trip (siblings listing, no blob fetch)
//! 2. classify the repo: GGUF-only, dense safetensors, mixed, empty
//! 3. apply the feasibility table against the requested
//!    `ModelSpec` (tp_size, quant)
//! 4. return a structured `PreflightError` the API layer can map to
//!    422 + JSON, or `Ok(PlacementPlan)` carrying the decisions the
//!    downstream load path needs (which GGUF file to fetch, etc.).
//!
//! Phase 2 of plan-source-aware-loader-preflight. The Phase 1 scheme
//! work — `ModelSourceId` and per-scheme `SourceConfig` — is a
//! separate PR; preflight runs against the single configured
//! HuggingFace source for now and the scheme threading drops in
//! cleanly when Phase 1 lands.

use cortex_core::harness::ModelSpec;
use cortex_core::source::ModelSourceId;
use hf_hub::api::tokio::Api;
use serde::Serialize;

/// What the repo's siblings listing tells us about how to load it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceFormat {
    /// Only GGUF files present. Single-GPU load path. `quants` is the
    /// lowercased filename list so the operator can be told what's
    /// actually available when their `quant=` choice doesn't match.
    Gguf { quants: Vec<String> },
    /// Dense safetensors (single-file or sharded via index.json).
    /// Goes through `load_arch_dense` on single-GPU, or `load_tp` (with
    /// optional in-situ quantization) when `tensor_parallel > 1`.
    DenseSafetensors { sharded: bool },
    /// Both safetensors and GGUF present — prefer the dense path
    /// because it composes with TP and ISQ. We surface the GGUF
    /// filenames anyway so operators with a strong preference can
    /// see they exist.
    Mixed { gguf_quants: Vec<String> },
    /// No recognised weight files. Either a tokenizer-only repo
    /// (e.g. some base-model repos that only host `tokenizer.json` and
    /// expect the operator to use a `-GGUF` sibling repo) or a
    /// genuinely empty entry.
    Empty,
}

/// Output of `preflight` for a load that can proceed. Carries the
/// decisions downstream resolve_* paths would otherwise re-derive.
#[derive(Debug, Clone, Serialize)]
pub struct PlacementPlan {
    pub model_id: String,
    pub format: SourceFormat,
    pub tp_size: u32,
    /// Filename of the GGUF to fetch, populated when `format` is
    /// `Gguf` and a single-GPU load was requested. None for the
    /// dense/TP path.
    pub picked_quant_file: Option<String>,
}

/// Structured failure modes. Each variant carries the fields the API
/// layer needs to produce an actionable 422 body.
#[derive(Debug, Clone, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PreflightError {
    /// `repo.info()` failed. Captures the underlying cause as a string
    /// so the operator log shows whether it's auth, 404, or transport.
    #[error("failed to fetch repo info for '{model_id}': {cause}")]
    RepoFetchFailed { model_id: String, cause: String },

    /// The repo exists but has no recognised weight files.
    #[error(
        "repo '{model_id}' has no recognised weight files (no .gguf, no .safetensors); \
         a tokenizer-only repo cannot be loaded directly"
    )]
    EmptyRepo { model_id: String },

    /// Operator asked for `tensor_parallel > 1` on a GGUF-only repo.
    /// The TP path requires safetensors+config for in-situ
    /// quantization; GGUF-TP isn't implemented (see CLAUDE.md).
    #[error(
        "cannot load '{model_id}' with tensor_parallel={tp_size}: repo is GGUF-only \
         ({} .gguf files); TP requires dense safetensors. {suggestion}",
        gguf_quants.len()
    )]
    TpRequiresSafetensors {
        model_id: String,
        tp_size: u32,
        gguf_quants: Vec<String>,
        suggestion: String,
    },

    /// Operator asked for a GGUF quant whose substring doesn't match
    /// any filename in the repo. `nearest` is a best-effort Levenshtein
    /// suggestion against the available quant names.
    #[error(
        "no GGUF file in '{model_id}' matches quant '{requested}'; \
         available: {available:?}{}",
        nearest.as_ref().map(|n| format!("; did you mean '{n}'?")).unwrap_or_default()
    )]
    QuantNotFound {
        model_id: String,
        requested: String,
        available: Vec<String>,
        nearest: Option<String>,
    },
}

/// Run the placement check.
///
/// One network round-trip (`repo.info()`); no blob fetches. Returns
/// `Ok(PlacementPlan)` when the requested combination is feasible, or
/// a structured `PreflightError` describing what's wrong.
///
/// When `repo.info()` fails but the repo has a snapshot in the local
/// hf-hub cache, the siblings listing is reconstructed from the
/// snapshot directory and preflight proceeds offline (#189) — a
/// cold-booting host with warm caches must not depend on WAN/DNS
/// being up. Downstream `repo.get()` calls short-circuit on cached
/// files, so the whole load stays offline.
///
/// `api` and `cache` must already be configured for the scheme
/// `source_id` belongs to — caller (typically
/// `CandleHarness::load_model`) builds them via
/// `hf_api_for(&source_id.scheme)` / `hf_cache_for(&source_id.scheme)`.
/// Only the `org/name` portion of the id is sent to the registry.
pub async fn preflight(
    api: &Api,
    cache: &hf_hub::Cache,
    source_id: &ModelSourceId,
    spec: &ModelSpec,
) -> Result<PlacementPlan, PreflightError> {
    let repo = api.model(source_id.repo_path());
    let owned_filenames: Vec<String> = match repo.info().await {
        Ok(info) => info.siblings.into_iter().map(|s| s.rfilename).collect(),
        Err(e) => match cached_snapshot_files(cache.path(), &source_id.repo_path()) {
            Some(cached) => {
                tracing::warn!(
                    model = %source_id,
                    error = %e,
                    files = cached.len(),
                    "repo info fetch failed; proceeding from local hf-hub cache snapshot"
                );
                cached
            }
            None => {
                return Err(PreflightError::RepoFetchFailed {
                    model_id: source_id.to_string(),
                    cause: format!("{e}"),
                });
            }
        },
    };

    let filenames: Vec<&str> = owned_filenames.iter().map(String::as_str).collect();
    let format = classify(&filenames);
    let tp_size = spec.tensor_parallel.unwrap_or(1);

    match (&format, tp_size, spec.quant.as_deref()) {
        // No weights at all — nothing to do.
        (SourceFormat::Empty, _, _) => Err(PreflightError::EmptyRepo {
            model_id: source_id.to_string(),
        }),

        // GGUF-only + TP: not supported. Today's HauhauCS failure.
        (SourceFormat::Gguf { quants }, tp, _) if tp > 1 => {
            Err(PreflightError::TpRequiresSafetensors {
                model_id: source_id.to_string(),
                tp_size: tp,
                gguf_quants: quants.clone(),
                suggestion: format!(
                    "Set tensor_parallel=1 and pick a quant from {quants:?}, \
                     or use a dense safetensors release of this model."
                ),
            })
        }

        // GGUF-only + single-GPU: pick the file that matches the
        // operator's quant. Empty quant matches the first GGUF.
        (SourceFormat::Gguf { quants }, _, requested) => {
            let picked = pick_gguf_file(&filenames, requested.unwrap_or(""));
            match picked {
                Some(fname) => Ok(PlacementPlan {
                    model_id: source_id.to_string(),
                    format: format.clone(),
                    tp_size,
                    picked_quant_file: Some(fname),
                }),
                None => Err(PreflightError::QuantNotFound {
                    model_id: source_id.to_string(),
                    requested: requested.unwrap_or("").to_string(),
                    available: quants.clone(),
                    nearest: nearest_quant(requested.unwrap_or(""), quants),
                }),
            }
        }

        // Dense or mixed: dense path handles both single-GPU and TP.
        // The architecture compatibility check stays where it is —
        // `check_dense_config_supported` runs once `config.json` is
        // on disk, since it needs the parsed JSON.
        (SourceFormat::DenseSafetensors { .. } | SourceFormat::Mixed { .. }, _, _) => {
            Ok(PlacementPlan {
                model_id: source_id.to_string(),
                format: format.clone(),
                tp_size,
                picked_quant_file: None,
            })
        }
    }
}

/// List the files of a repo's cached snapshot, mirroring hf-hub's
/// cache layout: `<cache>/models--{org}--{name}/refs/main` names the
/// commit, `snapshots/<commit>/` holds the per-file symlinks. Returns
/// relative `/`-separated paths — the same shape `repo.info()` reports
/// in `siblings[].rfilename` — or `None` when the repo has no usable
/// snapshot (never cached, or the ref/snapshot is missing).
pub fn cached_snapshot_files(cache_path: &std::path::Path, repo_path: &str) -> Option<Vec<String>> {
    let repo_dir = cache_path.join(format!("models--{}", repo_path.replace('/', "--")));
    let commit = std::fs::read_to_string(repo_dir.join("refs").join("main")).ok()?;
    let snapshot = repo_dir.join("snapshots").join(commit.trim());
    let mut files = Vec::new();
    collect_snapshot_files(&snapshot, &snapshot, &mut files);
    files.sort();
    if files.is_empty() { None } else { Some(files) }
}

/// Recursive walk of a snapshot directory. Snapshot entries are
/// symlinks into `blobs/`; `is_dir`/`exists` follow them, so a symlink
/// whose blob was pruned is skipped rather than reported as present.
fn collect_snapshot_files(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_snapshot_files(root, &path, out);
        } else if path.exists()
            && let Ok(rel) = path.strip_prefix(root)
        {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// Classify a siblings file list into a `SourceFormat`. Pulled out so
/// the unit tests can exercise it against fixture JSON without
/// spinning up an Api.
pub fn classify(filenames: &[&str]) -> SourceFormat {
    let mut gguf_quants: Vec<String> = filenames
        .iter()
        .filter(|f| f.to_lowercase().ends_with(".gguf"))
        .map(|f| f.to_lowercase())
        .collect();
    gguf_quants.sort();
    gguf_quants.dedup();

    let has_safetensors = filenames.iter().any(|f| f.ends_with(".safetensors"));
    let sharded = filenames
        .iter()
        .any(|f| f.ends_with("model.safetensors.index.json"));

    match (has_safetensors, gguf_quants.is_empty()) {
        (true, true) => SourceFormat::DenseSafetensors { sharded },
        (true, false) => SourceFormat::Mixed { gguf_quants },
        (false, false) => SourceFormat::Gguf {
            quants: gguf_quants,
        },
        (false, true) => SourceFormat::Empty,
    }
}

/// Mirror of the quant-matching logic in `candle.rs::resolve_files` so
/// preflight picks the same file the downstream loader would. Empty
/// quant returns the first `.gguf` (any quant). Lowercased substring
/// match otherwise.
fn pick_gguf_file(filenames: &[&str], quant_lc: &str) -> Option<String> {
    filenames
        .iter()
        .filter(|f| f.to_lowercase().ends_with(".gguf"))
        .find(|f| quant_lc.is_empty() || f.to_lowercase().contains(quant_lc))
        .map(|f| f.to_string())
}

/// Best-effort suggestion when the operator's quant name doesn't
/// substring-match any filename. Extracts the quant-ish token from
/// each `.gguf` filename and picks the one with the smallest
/// Levenshtein distance to the requested string. Returns None when
/// the input is empty or no candidates exist.
fn nearest_quant(requested: &str, candidates: &[String]) -> Option<String> {
    if requested.is_empty() || candidates.is_empty() {
        return None;
    }
    // Pull the "Q6_K_P"/"IQ4_XS"-ish token out of each filename for a
    // fairer comparison. Filenames look like
    // `Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf`, so the
    // quant is the last `-`-separated segment before the extension,
    // lowercased.
    let tokens: Vec<(String, String)> = candidates
        .iter()
        .map(|f| (extract_quant_token(f), f.clone()))
        .collect();

    let req_lc = requested.to_lowercase();
    tokens
        .into_iter()
        .min_by_key(|(token, _)| levenshtein(&req_lc, token))
        .map(|(token, _)| token)
}

fn extract_quant_token(filename: &str) -> String {
    let stem = filename
        .rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or(filename);
    let token = stem.rsplit('-').next().unwrap_or(stem);
    token.to_lowercase()
}

/// Iterative Levenshtein. Small inputs (quant names are <=12 chars),
/// no need for the `levenshtein` crate.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(model_id: &str, tp: Option<u32>, quant: Option<&str>) -> ModelSpec {
        ModelSpec {
            model_id: model_id.into(),
            harness: "candle".into(),
            quant: quant.map(String::from),
            tensor_parallel: tp,
            devices: None,
        }
    }

    #[test]
    fn cached_snapshot_files_lists_relative_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("models--Qwen--Qwen3-8B");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), "abc123\n").unwrap();
        let snap = repo.join("snapshots").join("abc123");
        std::fs::create_dir_all(snap.join("nested")).unwrap();
        std::fs::write(snap.join("config.json"), "{}").unwrap();
        std::fs::write(snap.join("model.safetensors"), "x").unwrap();
        std::fs::write(snap.join("nested").join("tokenizer.json"), "{}").unwrap();

        let files = cached_snapshot_files(tmp.path(), "Qwen/Qwen3-8B").unwrap();
        assert_eq!(
            files,
            vec!["config.json", "model.safetensors", "nested/tokenizer.json"]
        );
    }

    #[test]
    fn cached_snapshot_files_absent_or_empty_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(cached_snapshot_files(tmp.path(), "No/Repo").is_none());

        // Ref exists but the snapshot directory is empty — no usable
        // listing, the caller must surface the original fetch error.
        let repo = tmp.path().join("models--Empty--Repo");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), "sha").unwrap();
        std::fs::create_dir_all(repo.join("snapshots").join("sha")).unwrap();
        assert!(cached_snapshot_files(tmp.path(), "Empty/Repo").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn cached_snapshot_files_skips_broken_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("models--Pruned--Blobs");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), "sha").unwrap();
        let snap = repo.join("snapshots").join("sha");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("config.json"), "{}").unwrap();
        std::os::unix::fs::symlink(
            repo.join("blobs").join("gone"),
            snap.join("model.safetensors"),
        )
        .unwrap();

        let files = cached_snapshot_files(tmp.path(), "Pruned/Blobs").unwrap();
        assert_eq!(files, vec!["config.json"]);
    }

    #[test]
    fn classify_gguf_only() {
        let files = [
            "README.md",
            ".gitattributes",
            "Qwen3.6-27B-Q6_K_P.gguf",
            "Qwen3.6-27B-Q4_K_P.gguf",
        ];
        match classify(&files) {
            SourceFormat::Gguf { quants } => {
                assert_eq!(quants.len(), 2);
                assert!(quants.iter().any(|q| q.contains("q6_k_p")));
            }
            other => panic!("expected Gguf, got {other:?}"),
        }
    }

    #[test]
    fn classify_dense_sharded() {
        let files = [
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        assert_eq!(
            classify(&files),
            SourceFormat::DenseSafetensors { sharded: true }
        );
    }

    #[test]
    fn classify_dense_single_file() {
        let files = ["config.json", "tokenizer.json", "model.safetensors"];
        assert_eq!(
            classify(&files),
            SourceFormat::DenseSafetensors { sharded: false }
        );
    }

    #[test]
    fn classify_mixed() {
        let files = [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "model-Q4_K_M.gguf",
        ];
        match classify(&files) {
            SourceFormat::Mixed { gguf_quants } => {
                assert_eq!(gguf_quants, vec!["model-q4_k_m.gguf"]);
            }
            other => panic!("expected Mixed, got {other:?}"),
        }
    }

    #[test]
    fn classify_empty() {
        let files = ["README.md", "tokenizer.json"];
        assert_eq!(classify(&files), SourceFormat::Empty);
    }

    #[test]
    fn pick_gguf_substring_match() {
        let files = ["model-Q4_K_M.gguf", "model-Q6_K.gguf", "model-Q8_0.gguf"];
        assert_eq!(
            pick_gguf_file(&files, "q6_k"),
            Some("model-Q6_K.gguf".into())
        );
    }

    #[test]
    fn pick_gguf_empty_returns_first() {
        let files = ["model-Q4_K_M.gguf", "model-Q6_K.gguf"];
        assert_eq!(pick_gguf_file(&files, ""), Some("model-Q4_K_M.gguf".into()));
    }

    #[test]
    fn pick_gguf_no_match() {
        let files = ["model-Q4_K_M.gguf", "model-Q6_K.gguf"];
        assert_eq!(pick_gguf_file(&files, "iq2_xs"), None);
    }

    #[test]
    fn nearest_quant_suggests_close_match() {
        // Today's HauhauCS scenario: operator wrote "q6k", actual
        // filename token is "q6_k_p". Should suggest the latter.
        let candidates = vec![
            "qwen-q4_k_p.gguf".to_string(),
            "qwen-q5_k_p.gguf".to_string(),
            "qwen-q6_k_p.gguf".to_string(),
            "qwen-q8_k_p.gguf".to_string(),
        ];
        assert_eq!(nearest_quant("q6k", &candidates), Some("q6_k_p".into()));
    }

    #[test]
    fn nearest_quant_empty_input() {
        assert_eq!(nearest_quant("", &[]), None);
        assert_eq!(nearest_quant("q6k", &[]), None);
        assert_eq!(nearest_quant("", &["model-q4.gguf".into()]), None);
    }

    #[test]
    fn extract_quant_handles_typical_filenames() {
        assert_eq!(extract_quant_token("Qwen3.6-27B-Q6_K_P.gguf"), "q6_k_p");
        assert_eq!(extract_quant_token("model-IQ4_XS.gguf"), "iq4_xs");
        assert_eq!(extract_quant_token("simple.gguf"), "simple");
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("q6k", "q6_k_p"), 3);
        assert_eq!(levenshtein("q6k", "q4_k_p"), 4);
    }

    // Higher-level preflight tests below exercise the full feasibility
    // table via a thin wrapper that bypasses the network — we hand it
    // a pre-built `SourceFormat` and request shape, then drive the
    // same decision logic. The end-to-end test with a mock HTTP
    // server lives in tests/preflight.rs (integration).

    /// Mirror of the `match` in `preflight()` but takes a classified
    /// `SourceFormat` directly. Lets us unit-test the feasibility
    /// table without making the API trait object-safe / boxable.
    fn decide(
        spec: &ModelSpec,
        format: &SourceFormat,
        filenames: &[&str],
    ) -> Result<PlacementPlan, PreflightError> {
        // Tests parse spec.model_id with the default scheme so the
        // assertions can keep comparing against bare "org/name".
        let source_id: ModelSourceId = spec
            .model_id
            .parse::<ModelSourceId>()
            .expect("test spec.model_id must parse");
        let tp_size = spec.tensor_parallel.unwrap_or(1);
        match (format, tp_size, spec.quant.as_deref()) {
            (SourceFormat::Empty, _, _) => Err(PreflightError::EmptyRepo {
                model_id: source_id.to_string(),
            }),
            (SourceFormat::Gguf { quants }, tp, _) if tp > 1 => {
                Err(PreflightError::TpRequiresSafetensors {
                    model_id: source_id.to_string(),
                    tp_size: tp,
                    gguf_quants: quants.clone(),
                    suggestion: format!(
                        "Set tensor_parallel=1 and pick a quant from {quants:?}, \
                         or use a dense safetensors release of this model."
                    ),
                })
            }
            (SourceFormat::Gguf { quants }, _, requested) => {
                let picked = pick_gguf_file(filenames, requested.unwrap_or(""));
                match picked {
                    Some(fname) => Ok(PlacementPlan {
                        model_id: source_id.to_string(),
                        format: format.clone(),
                        tp_size,
                        picked_quant_file: Some(fname),
                    }),
                    None => Err(PreflightError::QuantNotFound {
                        model_id: source_id.to_string(),
                        requested: requested.unwrap_or("").to_string(),
                        available: quants.clone(),
                        nearest: nearest_quant(requested.unwrap_or(""), quants),
                    }),
                }
            }
            (SourceFormat::DenseSafetensors { .. } | SourceFormat::Mixed { .. }, _, _) => {
                Ok(PlacementPlan {
                    model_id: source_id.to_string(),
                    format: format.clone(),
                    tp_size,
                    picked_quant_file: None,
                })
            }
        }
    }

    #[test]
    fn feasibility_gguf_tp_rejected() {
        let files = ["Qwen-Q6_K_P.gguf", "Qwen-Q4_K_P.gguf"];
        let fmt = classify(&files);
        let s = spec("HauhauCS/Qwen3.6", Some(2), Some("q6k"));
        match decide(&s, &fmt, &files).unwrap_err() {
            PreflightError::TpRequiresSafetensors {
                model_id,
                tp_size,
                gguf_quants,
                ..
            } => {
                assert_eq!(model_id, "HauhauCS/Qwen3.6");
                assert_eq!(tp_size, 2);
                assert_eq!(gguf_quants.len(), 2);
            }
            other => panic!("expected TpRequiresSafetensors, got {other:?}"),
        }
    }

    #[test]
    fn feasibility_gguf_single_gpu_bad_quant() {
        let files = [
            "Qwen-Q4_K_P.gguf",
            "Qwen-Q5_K_P.gguf",
            "Qwen-Q6_K_P.gguf",
            "Qwen-Q8_K_P.gguf",
        ];
        let fmt = classify(&files);
        let s = spec("HauhauCS/Qwen3.6", Some(1), Some("q6k"));
        match decide(&s, &fmt, &files).unwrap_err() {
            PreflightError::QuantNotFound {
                requested,
                nearest,
                available,
                ..
            } => {
                assert_eq!(requested, "q6k");
                assert_eq!(nearest.as_deref(), Some("q6_k_p"));
                assert_eq!(available.len(), 4);
            }
            other => panic!("expected QuantNotFound, got {other:?}"),
        }
    }

    #[test]
    fn feasibility_gguf_single_gpu_good_quant() {
        let files = ["Qwen-Q4_K_M.gguf", "Qwen-Q6_K.gguf"];
        let fmt = classify(&files);
        let s = spec("Qwen/Q-GGUF", Some(1), Some("q6_k"));
        let plan = decide(&s, &fmt, &files).unwrap();
        assert_eq!(plan.picked_quant_file.as_deref(), Some("Qwen-Q6_K.gguf"));
    }

    #[test]
    fn feasibility_dense_tp_ok() {
        let files = [
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
        ];
        let fmt = classify(&files);
        let s = spec("Qwen/Q3-30B", Some(2), Some("q5k"));
        let plan = decide(&s, &fmt, &files).unwrap();
        assert_eq!(plan.tp_size, 2);
        assert!(plan.picked_quant_file.is_none());
        assert!(matches!(
            plan.format,
            SourceFormat::DenseSafetensors { sharded: true }
        ));
    }

    #[test]
    fn feasibility_empty_rejected() {
        let files = ["README.md", "tokenizer.json"];
        let fmt = classify(&files);
        let s = spec("Empty/Repo", Some(1), None);
        match decide(&s, &fmt, &files).unwrap_err() {
            PreflightError::EmptyRepo { model_id } => assert_eq!(model_id, "Empty/Repo"),
            other => panic!("expected EmptyRepo, got {other:?}"),
        }
    }

    #[test]
    fn error_serialization_carries_kind_field() {
        let err = PreflightError::TpRequiresSafetensors {
            model_id: "x/y".into(),
            tp_size: 2,
            gguf_quants: vec!["q6_k_p".into()],
            suggestion: "...".into(),
        };
        let v: serde_json::Value = serde_json::to_value(&err).unwrap();
        assert_eq!(v["kind"], "tp_requires_safetensors");
        assert_eq!(v["model_id"], "x/y");
        assert_eq!(v["tp_size"], 2);
    }
}
