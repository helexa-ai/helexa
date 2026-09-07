//! Numerical validation against the HF transformers reference (#15).
//!
//! Replays the fixtures captured by `script/dump_reference.py` (token
//! ids, image, reference tensors) through neuron's own qwen3_5
//! implementation and compares. This is what pins the README's
//! "implemented in this repository, ported against the HuggingFace
//! reference" claim to numbers.
//!
//! Needs the model weights on disk, so it self-skips unless
//! `NEURON_REF_MODEL_PATH` points at the HF snapshot directory the
//! fixtures were captured from (see each fixture's `manifest.json`).
//! Run on a host with the snapshot (CUDA used when available):
//!
//! ```sh
//! NEURON_REF_MODEL_PATH=/path/to/models--Qwen--Qwen3.5-0.8B/snapshots/<rev> \
//!     cargo test -p neuron --test numerical_reference -- --nocapture
//! ```
//!
//! The text prompt is longer than 64 tokens on purpose: the replay
//! prefill goes through the chunked delta-rule path, so the
//! comparison validates the production prefill math, not just the
//! per-token recurrence.
//!
//! Fixtures are captured in **f32** (script default) so the
//! comparison pins the math itself: observed f32-vs-f32 agreement is
//! text max_abs 0.000 / cosine 1.000000 and vision-tower cosine
//! 0.999998 (worst patch 0.99994), so the thresholds below sit far
//! above noise and far below any real bug (a wrong RoPE base, a
//! missing projector bias, an off-by-one in position handling).
//! For context: comparing across dtypes is dominated by bf16
//! rounding chaos through the 27-layer tower (global cosine ~0.997,
//! worst patch ~0.92, worst index unstable across dtypes) — that is
//! production-dtype noise, not implementation error, and is why the
//! fixtures are not captured in bf16.

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Manifest {
    case: String,
    token_ids: Vec<u32>,
    #[serde(default)]
    image_grid_thw: Option<Vec<usize>>,
    files: std::collections::HashMap<String, FileEntry>,
}

#[derive(Deserialize)]
struct FileEntry {
    file: String,
    shape: Vec<usize>,
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/numerical")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert!(bytes.len().is_multiple_of(4), "truncated f32 file {path:?}");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

struct Comparison {
    max_abs: f32,
    cosine: f64,
    argmax_ours: usize,
    argmax_ref: usize,
}

fn compare(ours: &[f32], reference: &[f32]) -> Comparison {
    assert_eq!(ours.len(), reference.len(), "length mismatch");
    let mut max_abs = 0f32;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&a, &b) in ours.iter().zip(reference) {
        max_abs = max_abs.max((a - b).abs());
        dot += a as f64 * b as f64;
        na += a as f64 * a as f64;
        nb += b as f64 * b as f64;
    }
    let argmax = |xs: &[f32]| {
        xs.iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    Comparison {
        max_abs,
        cosine: dot / (na.sqrt() * nb.sqrt()),
        argmax_ours: argmax(ours),
        argmax_ref: argmax(reference),
    }
}

/// bf16 on CUDA (matching production and the reference capture);
/// f32 on CPU, where candle has no bf16 matmul — the comparison
/// tolerances absorb the reference's bf16 rounding either way.
fn load_dtype(device: &Device) -> DType {
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

fn load_model(
    model_path: &str,
    device: &Device,
) -> neuron::harness::arch::qwen3_5::Qwen3_5ForCausalLM {
    let cfg_text =
        std::fs::read_to_string(Path::new(model_path).join("config.json")).expect("config.json");
    let cfg: neuron::harness::arch::qwen3_5::Config =
        serde_json::from_str(&cfg_text).expect("parse config");
    let index_text =
        std::fs::read_to_string(Path::new(model_path).join("model.safetensors.index.json"));
    let paths: Vec<PathBuf> = match index_text {
        Ok(text) => {
            let v: serde_json::Value = serde_json::from_str(&text).expect("parse index");
            let mut names: Vec<String> = v["weight_map"]
                .as_object()
                .expect("weight_map")
                .values()
                .filter_map(|x| x.as_str().map(String::from))
                .collect();
            names.sort();
            names.dedup();
            names
                .into_iter()
                .map(|n| Path::new(model_path).join(n))
                .collect()
        }
        Err(_) => vec![Path::new(model_path).join("model.safetensors")],
    };
    // SAFETY: mmap of read-only snapshot files, same justification as
    // the production loader.
    let vb = unsafe {
        candle_nn::var_builder::ShardedSafeTensors::var_builder(&paths, load_dtype(device), device)
            .expect("var_builder")
    };
    neuron::harness::arch::qwen3_5::Qwen3_5ForCausalLM::new(cfg, vb).expect("build model")
}

fn pick_device() -> Device {
    Device::new_cuda(0).unwrap_or(Device::Cpu)
}

fn ref_model_path() -> Option<String> {
    match std::env::var("NEURON_REF_MODEL_PATH") {
        Ok(p) if !p.is_empty() => Some(p),
        _ => {
            eprintln!("NEURON_REF_MODEL_PATH unset — skipping numerical reference test");
            None
        }
    }
}

#[test]
fn text_logits_match_reference() {
    let Some(model_path) = ref_model_path() else {
        return;
    };
    let fixture = fixture_root().join("qwen3_5-0.8b-text");
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest.case, "text");
    let reference = read_f32(&fixture.join(&manifest.files["logits"].file));

    let device = pick_device();
    let mut model = load_model(&model_path, &device);
    let input = Tensor::new(manifest.token_ids.as_slice(), &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    // Single full-prompt forward; the prompt is >64 tokens so the
    // GDN layers take the chunked prefill path.
    let logits = model.forward(&input, 0).unwrap();
    let ours: Vec<f32> = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let c = compare(&ours, &reference);
    eprintln!(
        "text: max_abs={:.4} cosine={:.6} argmax ours={} ref={}",
        c.max_abs, c.cosine, c.argmax_ours, c.argmax_ref
    );
    assert_eq!(c.argmax_ours, c.argmax_ref, "argmax token diverged");
    assert!(c.cosine > 0.9995, "cosine {:.6} too low", c.cosine);
    assert!(c.max_abs < 0.1, "max abs diff {:.4} too high", c.max_abs);
}

#[test]
fn vision_tower_and_logits_match_reference() {
    let Some(model_path) = ref_model_path() else {
        return;
    };
    let fixture = fixture_root().join("qwen3_5-0.8b-vision");
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest.case, "vision");
    let ref_visual = read_f32(&fixture.join(&manifest.files["visual_out"].file));
    let ref_logits = read_f32(&fixture.join(&manifest.files["logits"].file));
    let visual_shape = manifest.files["visual_out"].shape.clone();

    let device = pick_device();
    let model = load_model(&model_path, &device);
    let tower = model.vision().expect("model has a vision tower");
    let image_token_id = model.image_token_id().expect("image_token_id");

    // Same preprocessing path production requests take. The fixture
    // image is 448×448 (factor-aligned) so resize is the identity and
    // any mismatch below is normalization/patchify/tower math.
    let img = image::open(fixture.join("image.png")).expect("open fixture image");
    let profile = neuron::harness::preprocess::PreprocessProfile::qwen3_6();
    let (pixels, h, w) = neuron::harness::preprocess::preprocess(&img, &profile).unwrap();
    let image = Tensor::from_vec(pixels, (3, h as usize, w as usize), &device).unwrap();

    let embeds = tower.forward(&image).unwrap();
    assert_eq!(
        embeds.dims(),
        visual_shape.as_slice(),
        "tower output shape vs reference"
    );
    let ours_visual: Vec<f32> = embeds
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let cv = compare(&ours_visual, &ref_visual);
    // Per-patch cosine: a positional bug (pos-embed interpolation,
    // rotary grid, merger order) concentrates error in specific
    // patches; dtype noise spreads uniformly.
    let hidden = visual_shape[1];
    let mut worst = (1.0f64, 0usize);
    for (i, (a, b)) in ours_visual
        .chunks(hidden)
        .zip(ref_visual.chunks(hidden))
        .enumerate()
    {
        let c = compare(a, b);
        if c.cosine < worst.0 {
            worst = (c.cosine, i);
        }
    }
    eprintln!(
        "vision tower: max_abs={:.4} cosine={:.6} worst_patch={} (cosine {:.6})",
        cv.max_abs, cv.cosine, worst.1, worst.0
    );
    assert!(cv.cosine > 0.9995, "tower cosine {:.6} too low", cv.cosine);
    assert!(
        worst.0 > 0.995,
        "patch {} cosine {:.6} — positional divergence",
        worst.1,
        worst.0
    );

    // Full LM forward with the splice — the fixture token ids are
    // already pad-expanded by the HF processor. The LM grid is the
    // post-merge grid: grid_thw / spatial_merge.
    let grid = manifest.image_grid_thw.as_ref().expect("grid in manifest");
    let lm_grid = (grid[1] / 2, grid[2] / 2);
    let mut model = model;
    let input = Tensor::new(manifest.token_ids.as_slice(), &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let logits = model
        .forward_with_vision(&input, 0, &embeds, image_token_id, &[lm_grid])
        .unwrap();
    let ours_logits: Vec<f32> = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let cl = compare(&ours_logits, &ref_logits);
    eprintln!(
        "vision logits: max_abs={:.4} cosine={:.6} argmax ours={} ref={}",
        cl.max_abs, cl.cosine, cl.argmax_ours, cl.argmax_ref
    );
    assert_eq!(cl.argmax_ours, cl.argmax_ref, "argmax token diverged");
    assert!(cl.cosine > 0.9995, "logits cosine {:.6} too low", cl.cosine);
    assert!(cl.max_abs < 0.1, "max abs diff {:.4} too high", cl.max_abs);

    // #18: the chunked single-GPU vision prefill must produce the same
    // logits as the single-shot path. chunk_size 64 over a 217-token
    // prompt forces 4 chunks, and the ~196-token image-pad run spans
    // them — exercising the per-chunk splice + img_off accounting and
    // the GDN/KV cross-chunk state carry. Comparing to BOTH the
    // single-shot result and the HF reference pins chunked == single-
    // shot == reference. Re-encodes the image internally (same tower),
    // so it takes pixel tensors, not the pre-encoded `embeds`.
    model.clear_kv_cache();
    let chunked = model
        .prefill_with_images_chunked(
            manifest.token_ids.as_slice(),
            0,
            std::slice::from_ref(&image),
            image_token_id,
            64,
        )
        .unwrap();
    let chunked_logits: Vec<f32> = chunked
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let cc = compare(&chunked_logits, &ours_logits);
    let cr = compare(&chunked_logits, &ref_logits);
    eprintln!(
        "vision chunked(64): vs single-shot max_abs={:.4} cosine={:.6}; vs ref argmax={}",
        cc.max_abs, cc.cosine, cr.argmax_ours
    );
    assert_eq!(
        cc.argmax_ours, cl.argmax_ours,
        "chunked vision argmax diverged from single-shot"
    );
    assert!(
        cc.cosine > 0.99995,
        "chunked vs single-shot cosine {:.6} too low — chunking changed the math",
        cc.cosine
    );
    assert_eq!(cr.argmax_ours, cr.argmax_ref, "chunked argmax vs reference");
}

/// `qwen4_exp` logits against the upstream reference (#323).
///
/// Unlike the fixtures above this one needs no `NEURON_REF_MODEL_PATH`
/// and no snapshot: the model is 44 KB of random weights in the real
/// architecture, generated *by* `transformers` itself
/// (`script/generate_qwen4_exp_fixture.py`), so it is checked in and
/// this runs on every push. A tensor name we cannot find is therefore a
/// finding about our loader rather than a fixture that drifted.
///
/// f32 on both sides, for the reason in this file's header: cross-dtype
/// comparison measures bf16 rounding, not implementation error.
///
/// What it covers that the unit tests cannot: the four novel blocks
/// composed. Hyper-connections carrying four residual streams with no
/// layernorm of the usual kind, the PLE n-gram gather on layer 1, QSA
/// on layer 3, and the final mixer collapse with no `model.norm` before
/// `lm_head` — every one of which was written from prose and tested
/// against our own reading of it.
#[test]
fn qwen4_exp_logits_match_reference() {
    // The recurrent GDN step and a 12-token prompt: agreement is
    // essentially exact (observed 2e-6), so the bound sits just off the
    // floor. At 1e-3 this passed with the fused expert's gate and up
    // halves swapped (2.4e-4) — a bound wide enough to feel safe is
    // wide enough to hide what it was written for.
    qwen4_exp_case("qwen4_exp-tiny", 1e-5);
}

/// The same comparison over a prompt long enough, and shaped
/// deliberately enough, to reach two paths the short case cannot.
///
/// **80 tokens** puts the GatedDeltaNet layers on the *chunked* prefill
/// algorithm (#23), which engages at 64 and is a different
/// implementation from the recurrent step the 12-token case exercises —
/// the same reason the qwen3_5 fixture above is over 64 tokens. It also
/// gives QSA up to 40 blocks against a budget of 4, so selection is
/// discarding almost everything rather than one block at the boundary.
///
/// **An eos at position 37** puts a segment boundary inside the n-gram
/// window. `_shift_right_ignore_eos` refuses to read an n-gram across
/// one, and that logic — a cummax over eos positions, per row — is
/// intricate enough to be worth comparing rather than reasoning about.
/// The short case has no eos at all, on purpose: a fixture that happens
/// to contain one tests segment handling by accident.
#[test]
fn qwen4_exp_long_prompt_logits_match_reference() {
    // Looser than the short case, and for a stated reason rather than
    // to go green: at 80 tokens the GDN layers take the chunked delta
    // rule, which is a different *factorisation* of the same
    // recurrence, so its f32 rounding differs from the recurrent step
    // by construction. Observed 1.3e-5 against the 2e-6 of the short
    // case. The bound is still four orders of magnitude below what the
    // mutations produce — gate/up swapped moves this fixture by 2.7 —
    // so it discriminates exactly as well.
    qwen4_exp_case("qwen4_exp-tiny-long", 1e-4);
}

fn qwen4_exp_case(name: &str, max_abs_bound: f32) {
    use candle_nn::var_builder::ShardedSafeTensors;
    use neuron::harness::arch::qwen4_exp::config::Config as Qwen4ExpConfig;
    use neuron::harness::arch::qwen4_exp::model::Qwen4ExpForCausalLM;

    let fixture = fixture_root().join(name);
    let weights = fixture.join("model.safetensors");
    let device = Device::Cpu;

    let cfg = Qwen4ExpConfig::from_config_json(
        &std::fs::read_to_string(fixture.join("config.json")).unwrap(),
    )
    .expect("parse the fixture config");

    let vb = unsafe { ShardedSafeTensors::var_builder(&[&weights], DType::F32, &device).unwrap() };
    let mut model = Qwen4ExpForCausalLM::load(
        &cfg,
        DType::F32,
        &device,
        &vb,
        &neuron::harness::arch::qwen4_exp::LoadOptions {
            quant: None,
            safetensors_paths: std::slice::from_ref(&weights),
            isq_cache: None,
            experts_onto: &device,
            experts_vb: None,
        },
    )
    .expect("load the fixture through the production loader");

    let expected = candle_core::safetensors::load(fixture.join("expected.safetensors"), &device)
        .expect("read the reference output");
    let ids = expected["input_ids"].to_dtype(DType::U32).unwrap();
    let reference: Vec<f32> = expected["logits"].flatten_all().unwrap().to_vec1().unwrap();

    let logits = model.forward(&ids, 0).expect("forward");
    let ours: Vec<f32> = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let c = compare(&ours, &reference);
    eprintln!(
        "{name}: max_abs={:.6} cosine={:.6} argmax ours={} ref={}",
        c.max_abs, c.cosine, c.argmax_ours, c.argmax_ref
    );
    // On failure, break the error down by position before asserting.
    // Whether the divergence is confined to one position or spread
    // across all of them is the first thing worth knowing: localised
    // means a discrete choice went differently (an expert, a QSA
    // block), spread means the arithmetic drifted. Finding that out
    // cost an hour the first time.
    if c.max_abs >= max_abs_bound {
        let v = cfg.text_config.vocab_size;
        for t in 0..(ours.len() / v) {
            let pc = compare(&ours[t * v..(t + 1) * v], &reference[t * v..(t + 1) * v]);
            eprintln!(
                "  pos {t:2}: max_abs={:.6} cosine={:.6}",
                pc.max_abs, pc.cosine
            );
        }
    }
    assert_eq!(c.argmax_ours, c.argmax_ref, "argmax token diverged");
    assert!(c.cosine > 0.999_999, "cosine {:.6} too low", c.cosine);
    assert!(
        c.max_abs < max_abs_bound,
        "max abs diff {:.6} exceeds {max_abs_bound:e}",
        c.max_abs
    );
}
