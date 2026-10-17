//! Parity with heartlib on the CPU in f32, against the #390 traces.
//!
//! Ignored by default: they need the checkpoints and minutes of CPU. Lay
//! the snapshot out as heartlib does —
//! `<ckpt>/{HeartMuLa-oss-3B,HeartCodec-oss,tokenizer.json,gen_config.json}`
//! — and run
//!
//!   HEARTMULA_CKPT=<ckpt> cargo test --release -p neuron --lib \
//!     -- --ignored heartmula --nocapture --test-threads=1
//!
//! `HEARTMULA_CASES` narrows the cases (comma-separated).

use super::codec::{self, HeartCodec};
use super::lm::{self, HeartMuLa};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn ckpt() -> Option<PathBuf> {
    match std::env::var("HEARTMULA_CKPT") {
        Ok(p) => Some(p.into()),
        Err(_) => {
            eprintln!("HEARTMULA_CKPT not set; skipping");
            None
        }
    }
}

fn cases(default: &[&str]) -> Vec<String> {
    match std::env::var("HEARTMULA_CASES") {
        Ok(c) => c.split(',').map(str::to_string).collect(),
        Err(_) => default.iter().map(|s| s.to_string()).collect(),
    }
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/harness/testdata/heartmula")
}

/// Named tensors from a trace. The files also hold U8 masks, which are
/// skipped (only the requested names are read).
fn load(case: &str, names: &[&str]) -> HashMap<String, Tensor> {
    let path = fixture_dir().join(format!("{case}__cpu_f32.safetensors"));
    let file = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }.unwrap();
    names
        .iter()
        .map(|n| (n.to_string(), file.load(n, &Device::Cpu).unwrap()))
        .collect()
}

fn rel_l2(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let d = (&a - &b).unwrap().sqr().unwrap().sum_all().unwrap();
    let n = b.sqr().unwrap().sum_all().unwrap();
    (d.to_scalar::<f32>().unwrap() / n.to_scalar::<f32>().unwrap().max(1e-30)).sqrt()
}

fn shards(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    v.sort();
    v
}

fn read_json<T: serde::de::DeserializeOwned>(p: &Path) -> T {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

/// Teacher-forced LM parity: the prompt matrix exactly, then frames 0 and
/// 1 with the recorded codes fed back, comparing every codebook's guided
/// logits and the backbone's last hidden state.
#[test]
#[ignore = "needs HEARTMULA_CKPT and minutes of CPU"]
fn heartmula_lm_matches_reference_cpu_f32() {
    let Some(root) = ckpt() else { return };
    let mula_dir = root.join("HeartMuLa-oss-3B");
    let config: lm::Config = read_json(&mula_dir.join("config.json"));
    let gen_cfg: lm::GenConfig = read_json(&root.join("gen_config.json"));
    let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
    let t = std::time::Instant::now();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shards(&mula_dir), DType::F32, &Device::Cpu).unwrap()
    };
    let mut model = HeartMuLa::new(config, vb).unwrap();
    eprintln!("LM loaded in {:.1}s", t.elapsed().as_secs_f32());

    for case in cases(&["instr_12s", "vocal_en_12s", "vocal_es_12s"]) {
        let meta: serde_json::Value =
            read_json(&fixture_dir().join(format!("{case}__cpu_f32.json")));
        let c = &meta["case"];
        let prompt = lm::build_prompt(
            &tokenizer,
            c["tags"].as_str().unwrap(),
            c["lyrics"].as_str().unwrap(),
            &gen_cfg,
        )
        .unwrap();
        let want = load(
            &case,
            &[
                "prompt_tokens",
                "lm_greedy_frames",
                "lm_greedy_logits",
                "lm_greedy_backbone_last_h",
                "lm_sampled_frames",
                "lm_sampled_logits",
                "lm_sampled_backbone_last_h",
            ],
        );
        let rows: Vec<Vec<i64>> = want["prompt_tokens"].to_vec2().unwrap();
        let got: Vec<Vec<i64>> = prompt
            .tokens
            .iter()
            .map(|r| r.iter().map(|&x| x as i64).collect())
            .collect();
        assert_eq!(got, rows, "{case}: prompt tokens");
        assert_eq!(prompt.muq_idx, meta["muq_idx"].as_u64().unwrap() as usize);

        for sampler in ["greedy", "sampled"] {
            let frames: Vec<Vec<i64>> = want[&format!("lm_{sampler}_frames")].to_vec2().unwrap();
            let logits_ref = &want[&format!("lm_{sampler}_logits")];
            let h_ref = &want[&format!("lm_{sampler}_backbone_last_h")];
            let cfg = meta["lm"][sampler]["cfg_scale"].as_f64().unwrap();
            let plen = prompt.tokens.len();
            model.reset(plen + 4);
            let t = std::time::Instant::now();
            let mut worst = 0f32;
            for f in 0..2usize {
                let mut got_logits: Vec<Vec<f32>> = Vec::new();
                let mut choose = |cb: usize, l: &[f32]| {
                    got_logits.push(l.to_vec());
                    frames[cb][f] as u32
                };
                let (codes, last_h) = if f == 0 {
                    let mask: Vec<[bool; 9]> = prompt
                        .tokens
                        .iter()
                        .map(|_| {
                            let mut m = [false; 9];
                            m[8] = true;
                            m
                        })
                        .collect();
                    model
                        .step(
                            &prompt.tokens,
                            &mask,
                            0,
                            Some(prompt.muq_idx),
                            cfg,
                            &mut choose,
                        )
                        .unwrap()
                } else {
                    let mut row = [gen_cfg.empty_id; 9];
                    for (cb, slot) in row.iter_mut().take(8).enumerate() {
                        *slot = frames[cb][0] as u32;
                    }
                    let mut m = [true; 9];
                    m[8] = false;
                    model
                        .step(&[row], &[m], plen, None, cfg, &mut choose)
                        .unwrap()
                };
                for (cb, &code) in codes.iter().enumerate() {
                    assert_eq!(code as i64, frames[cb][f]);
                }
                for (cb, l) in got_logits.iter().enumerate() {
                    let ours = Tensor::new(l.as_slice(), &Device::Cpu).unwrap();
                    let theirs = logits_ref.get(f).unwrap().get(cb).unwrap();
                    let e = rel_l2(&ours, &theirs);
                    worst = worst.max(e);
                    assert!(
                        e < 1e-4,
                        "{case}/{sampler} frame {f} cb {cb}: logits rel L2 {e}"
                    );
                }
                let e = rel_l2(&last_h, &h_ref.get(f).unwrap());
                worst = worst.max(e);
                assert!(e < 1e-4, "{case}/{sampler} frame {f}: last_h rel L2 {e}");
            }
            eprintln!(
                "{case}/{sampler}: worst rel L2 {worst:.2e} over 2 frames × 8 codebooks ({:.1}s)",
                t.elapsed().as_secs_f32()
            );
        }
    }
}

/// Codec parity from the recorded greedy codes and noise: the RVQ sum, the
/// flow condition, the velocity at step 1, the state after step 5, the
/// final latents, and the first 3 s of waveform.
#[test]
#[ignore = "needs HEARTMULA_CKPT and minutes of CPU"]
fn heartmula_codec_matches_reference_cpu_f32() {
    let Some(root) = ckpt() else { return };
    let dir = root.join("HeartCodec-oss");
    let config: codec::Config = read_json(&dir.join("config.json"));
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shards(&dir), DType::F32, &Device::Cpu).unwrap()
    };
    let model = HeartCodec::new(config, vb).unwrap();

    for case in cases(&["instr_12s", "vocal_en_12s"]) {
        let want = load(
            &case,
            &[
                "codec_codes",
                "codec_rvq_quantized",
                "codec_flow_mu",
                "codec_flow_noise",
                "codec_flow_t_span",
                "codec_flow_step1_v",
                "codec_flow_step5_x",
                "codec_flow_final",
                "codec_waveform",
            ],
        );
        let codes = want["codec_codes"].to_dtype(DType::U32).unwrap();
        let frames = codes.dim(1).unwrap();
        let tiled = codec::tile_codes(&codes).unwrap();
        let t = std::time::Instant::now();
        let (lat, trace) = model
            .segment_latents(
                &tiled.narrow(1, 0, 372).unwrap(),
                &want["codec_flow_noise"],
                None,
                codec::DEFAULT_STEPS,
                codec::DEFAULT_GUIDANCE,
            )
            .unwrap();
        let flow_s = t.elapsed().as_secs_f32();
        let t_span: Vec<f32> = want["codec_flow_t_span"].to_vec1().unwrap();
        assert_eq!(trace.t_span, t_span, "{case}: t_span");
        let checks = [
            ("rvq", rel_l2(&trace.rvq, &want["codec_rvq_quantized"])),
            ("mu", rel_l2(&trace.mu, &want["codec_flow_mu"])),
            (
                "step1_v",
                rel_l2(&trace.step1_v, &want["codec_flow_step1_v"]),
            ),
            (
                "step5_x",
                rel_l2(&trace.step5_x, &want["codec_flow_step5_x"]),
            ),
            ("final", rel_l2(&lat, &want["codec_flow_final"])),
        ];

        let t = std::time::Instant::now();
        let wav = model.latents_to_audio(&lat, Some(128)).unwrap();
        let decode_s = t.elapsed().as_secs_f32();
        assert_eq!(wav.dim(1).unwrap(), 744 * 1920);
        let target = frames * 48_000 * 2 / 25;
        assert!(target <= wav.dim(1).unwrap());
        let wref = &want["codec_waveform"];
        let n = wref.dim(1).unwrap();
        let wave = rel_l2(&wav.narrow(1, 0, n).unwrap(), wref);

        eprintln!(
            "{case}: {} | waveform {wave:.2e} | flow {flow_s:.1}s, decode {decode_s:.1}s",
            checks
                .iter()
                .map(|(n, e)| format!("{n} {e:.2e}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for (name, e) in checks {
            assert!(e < 1e-4, "{case}: {name} rel L2 {e}");
        }
        assert!(wave < 1e-3, "{case}: waveform rel L2 {wave}");
    }
}

/// The scalar decoder alone, from the reference's final latents, whole and
/// chunked: isolates the waveform stage from the flow.
#[test]
#[ignore = "needs HEARTMULA_CKPT and minutes of CPU"]
fn heartmula_scalar_decoder_matches_reference_cpu_f32() {
    let Some(root) = ckpt() else { return };
    let dir = root.join("HeartCodec-oss");
    let config: codec::Config = read_json(&dir.join("config.json"));
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shards(&dir), DType::F32, &Device::Cpu).unwrap()
    };
    let model = HeartCodec::new(config, vb).unwrap();
    for case in cases(&["instr_12s"]) {
        let want = load(&case, &["codec_flow_final", "codec_waveform"]);
        let wref = &want["codec_waveform"];
        let n = wref.dim(1).unwrap();
        for chunk in [None, Some(128)] {
            let t = std::time::Instant::now();
            let wav = model
                .latents_to_audio(&want["codec_flow_final"], chunk)
                .unwrap();
            let e = rel_l2(&wav.narrow(1, 0, n).unwrap(), wref);
            eprintln!(
                "{case} chunk {chunk:?}: waveform rel L2 {e:.2e} ({:.1}s)",
                t.elapsed().as_secs_f32()
            );
            assert!(e < 1e-3, "{case} chunk {chunk:?}: waveform rel L2 {e}");
        }
    }
}

/// Free-running greedy generation: frame 0 must equal the reference's
/// (nothing has been sampled yet), later frames may drift on near-ties
/// (#390). Reports the per-frame time.
#[test]
#[ignore = "needs HEARTMULA_CKPT and minutes of CPU"]
fn heartmula_generates_greedily() {
    let Some(root) = ckpt() else { return };
    let mula_dir = root.join("HeartMuLa-oss-3B");
    let config: lm::Config = read_json(&mula_dir.join("config.json"));
    let gen_cfg: lm::GenConfig = read_json(&root.join("gen_config.json"));
    let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shards(&mula_dir), DType::F32, &Device::Cpu).unwrap()
    };
    let mut model = HeartMuLa::new(config, vb).unwrap();
    for case in cases(&["vocal_en_12s"]) {
        let meta: serde_json::Value =
            read_json(&fixture_dir().join(format!("{case}__cpu_f32.json")));
        let c = &meta["case"];
        let prompt = lm::build_prompt(
            &tokenizer,
            c["tags"].as_str().unwrap(),
            c["lyrics"].as_str().unwrap(),
            &gen_cfg,
        )
        .unwrap();
        let want: Vec<Vec<i64>> = load(&case, &["lm_greedy_frames"])["lm_greedy_frames"]
            .to_vec2()
            .unwrap();
        let mut sampler = lm::Sampler::new(1.0, 1, 0);
        let extra = 24;
        let t = std::time::Instant::now();
        let frames = model
            .generate(&prompt, extra, 1.5, &gen_cfg, &mut sampler)
            .unwrap();
        let secs = t.elapsed().as_secs_f32();
        assert_eq!(frames.len(), extra + 1);
        let same = frames
            .iter()
            .enumerate()
            .take_while(|(f, codes)| (0..8).all(|cb| codes[cb] as i64 == want[cb][*f]))
            .count();
        eprintln!(
            "{case}: {} frames ({:.2}s of audio) in {secs:.1}s, {:.2}s/frame; first {same} frames equal the reference",
            frames.len(),
            frames.len() as f64 / 12.5,
            secs / frames.len() as f32
        );
        assert!(same >= 1, "{case}: frame 0 differs from the reference");
    }
}
