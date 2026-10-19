//! Music generation engines behind the device worker: ACE-Step 1.5 (#383)
//! and HeartMuLa (#393).
//!
//! The ACE-Step repo carries every component as a subfolder of one
//! snapshot — the turbo DiT (with its `silence_latent.pt`), the song
//! planner LM, the Qwen3 text encoder and the VAE — so the loader takes
//! the snapshot root and finds them by name.
//!
//! Same discipline as the image engine (`image.rs`): the pipeline lives in
//! the worker's audio slab, one `Job::GenerateAudio` runs a whole song, and
//! the reply is CPU-side samples. WAV encoding happens on the async side.

use super::arch::acestep::linear::with_quant;
use super::arch::acestep::pipeline::{AceStepPipeline, GenerationParams, LmMetadata};
use super::arch::acestep::{AceStepConfig, VaeConfig, lm::LmConfig, qwen3, sampling};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The turbo DiT's subfolder.
pub const DIT_DIR: &str = "acestep-v15-turbo";
/// The text encoder's subfolder.
pub const TEXT_ENCODER_DIR: &str = "Qwen3-Embedding-0.6B";
/// The VAE's subfolder.
pub const VAE_DIR: &str = "vae";
/// Planner LMs in order of preference; the first the snapshot has is used.
pub const PLANNER_DIRS: [&str; 2] = ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B"];

/// What ACE-Step 1.5 turbo serves (#383). Upstream allows 10 minutes; the
/// planner's code budget (3072 tokens at 5 Hz) covers that. No `steps`
/// knob: the turbo schedule is a fixed eight steps.
pub const ACESTEP_LIMITS: cortex_core::audio::AudioLimits = cortex_core::audio::AudioLimits {
    min_duration: 10.0,
    max_duration: 600.0,
    default_duration: 30.0,
    vocals: true,
    sample_rate: 48_000,
    channels: 2,
    max_steps: None,
};

/// The turbo schedule's default step count.
pub const DEFAULT_STEPS: usize = 8;
/// Turbo models are trained with timestep shift 3.
const TURBO_SHIFT: f64 = 3.0;
/// Upstream's default output peak, in dBFS.
const PEAK_DB: f64 = -1.0;

/// Files every ACE-Step snapshot needs, relative to its root.
pub fn required_files() -> Vec<String> {
    vec![
        format!("{DIT_DIR}/config.json"),
        format!("{DIT_DIR}/model.safetensors"),
        format!("{DIT_DIR}/silence_latent.pt"),
        format!("{TEXT_ENCODER_DIR}/config.json"),
        format!("{TEXT_ENCODER_DIR}/model.safetensors"),
        format!("{TEXT_ENCODER_DIR}/tokenizer.json"),
        format!("{VAE_DIR}/config.json"),
        format!("{VAE_DIR}/diffusion_pytorch_model.safetensors"),
    ]
}

/// Files a planner LM needs, relative to the snapshot root.
pub fn planner_files(dir: &str) -> Vec<String> {
    ["config.json", "model.safetensors", "tokenizer.json"]
        .iter()
        .map(|f| format!("{dir}/{f}"))
        .collect()
}

/// True for a repo laid out like ACE-Step 1.5: the turbo DiT and the VAE as
/// subfolders. Checked before the generic safetensors sniff, which would
/// otherwise take the root `config.json` for a text model.
pub fn is_acestep_layout(filenames: &[&str]) -> bool {
    filenames.contains(&format!("{DIT_DIR}/config.json").as_str())
        && filenames.contains(&format!("{VAE_DIR}/diffusion_pytorch_model.safetensors").as_str())
}

/// A resolved snapshot: the root directory every component sits under,
/// which planner (if any) it has, and how this host lays it out.
#[derive(Debug, Clone)]
pub struct AceStepFiles {
    pub root: PathBuf,
    pub planner: Option<String>,
    pub layout: AceStepLayout,
}

/// Where a pipeline component runs (#385).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Place {
    /// The model's device.
    #[default]
    Gpu,
    /// The host CPU, in f32.
    Cpu,
}

impl Place {
    fn parse(component: &str, value: Option<&str>) -> Result<Self> {
        match value.map(str::to_ascii_lowercase).as_deref() {
            None | Some("gpu") | Some("cuda") => Ok(Self::Gpu),
            Some("cpu") => Ok(Self::Cpu),
            Some(other) => anyhow::bail!(
                "audio placement for {component} is '{other}'; expected \"gpu\" or \"cpu\" \
                 (a component on another card of the host is not supported yet)"
            ),
        }
    }
}

/// How an ACE-Step pipeline is laid out on this host (#385): weight
/// quantization, compute precision, where each component runs, and the
/// memory budget it must fit.
#[derive(Debug, Clone, Default)]
pub struct AceStepLayout {
    /// Quantizes the DiT's, the text encoder's and the planner's linear
    /// layers. Quantized pipelines compute in f32.
    pub quant: Option<candle_core::quantized::GgmlDType>,
    /// f32 compute on the GPU even without quantization.
    pub f32_compute: bool,
    pub text_encoder: Place,
    pub planner: Place,
    pub dit: Place,
    pub vae: Place,
    pub vram_budget_mb: Option<u64>,
    pub enforce_vram_budget: bool,
}

impl AceStepLayout {
    pub fn from_config(cfg: &crate::config::AudioConfig) -> Result<Self> {
        let quant = match cfg.quant.as_deref().map(str::to_ascii_lowercase).as_deref() {
            None | Some("") | Some("none") => None,
            Some("q8_0") | Some("q8") => Some(candle_core::quantized::GgmlDType::Q8_0),
            Some(other) => {
                anyhow::bail!("unsupported audio quant '{other}'; ACE-Step serves q8_0 or none")
            }
        };
        let f32_compute = match cfg
            .compute
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            None | Some("bf16") => false,
            Some("f32") => true,
            Some(other) => {
                anyhow::bail!("unsupported audio compute '{other}'; expected bf16 or f32")
            }
        };
        if cfg.enforce_vram_budget && cfg.vram_budget_mb.is_none() {
            anyhow::bail!("enforce_vram_budget needs vram_budget_mb");
        }
        let p = &cfg.placement;
        Ok(Self {
            quant,
            f32_compute,
            text_encoder: Place::parse("text_encoder", p.text_encoder.as_deref())?,
            planner: Place::parse("planner", p.planner.as_deref())?,
            dit: Place::parse("dit", p.dit.as_deref())?,
            vae: Place::parse("vae", p.vae.as_deref())?,
            vram_budget_mb: cfg.vram_budget_mb,
            enforce_vram_budget: cfg.enforce_vram_budget,
        })
    }

    /// The device and dtype a component placed at `place` runs with.
    fn component(&self, place: Place, gpu: &Device) -> (Device, DType) {
        match place {
            Place::Cpu => (Device::Cpu, DType::F32),
            Place::Gpu if gpu.is_cpu() || self.quant.is_some() || self.f32_compute => {
                (gpu.clone(), DType::F32)
            }
            Place::Gpu => (gpu.clone(), DType::BF16),
        }
    }

    /// VAE decode chunk, in latent frames. A budget halves the default
    /// chunk, which roughly halves the decoder's transient memory.
    fn vae_chunk_frames(&self) -> Option<usize> {
        self.vram_budget_mb.map(|_| 64)
    }
}

/// One generation's parameters, already validated against the model's limits.
#[derive(Debug, Clone)]
pub struct AudioGenParams {
    pub prompt: String,
    /// Empty for an instrumental.
    pub lyrics: String,
    pub duration_secs: f64,
    pub seed: u64,
    pub steps: Option<usize>,
    pub guidance_scale: Option<f64>,
    /// Run the song planner first (when the model has one).
    pub planner: bool,
    pub temperature: Option<f64>,
    /// Lyric language code; `None` lets the planner (or `"unknown"`) decide.
    pub language: Option<String>,
    /// Top-k sampling (autoregressive models).
    pub top_k: Option<usize>,
}

/// One generation's output: CPU-side stereo samples in `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct AudioGenResult {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
    pub timing: cortex_core::audio::AudioTiming,
}

impl AudioGenResult {
    pub fn duration_secs(&self) -> f64 {
        self.left.len() as f64 / self.sample_rate.max(1) as f64
    }
}

/// The loaded pipeline, owned by the device worker.
pub struct AceStepEngine {
    pipeline: AceStepPipeline,
    has_planner: bool,
    vae_chunk_frames: Option<usize>,
    /// Device memory held so the pipeline sees only its budget
    /// (`enforce_vram_budget`). Dropped with the engine.
    _ballast: Option<candle_core::Tensor>,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(f).with_context(|| format!("parse {}", path.display()))
}

impl AceStepEngine {
    /// Build every component where `files.layout` places it: bf16 on a
    /// GPU, f32 on the CPU or when quantized or asked for.
    pub fn load(files: &AceStepFiles, device: &Device) -> Result<Self> {
        let layout = &files.layout;
        let ballast = match (layout.enforce_vram_budget, layout.vram_budget_mb) {
            (true, Some(budget)) => hold_all_but(device, budget)?,
            _ => None,
        };
        let (dit_device, dtype) = layout.component(layout.dit, device);
        let (text_device, text_dtype) = layout.component(layout.text_encoder, device);
        let (vae_device, vae_dtype) = layout.component(layout.vae, device);
        let (planner_device, planner_dtype) = layout.component(layout.planner, device);
        let root = &files.root;
        let dit = root.join(DIT_DIR);
        let te = root.join(TEXT_ENCODER_DIR);
        let vae = root.join(VAE_DIR);

        let dit_config: AceStepConfig = read_json(&dit.join("config.json"))?;
        let vae_config: VaeConfig = read_json(&vae.join("config.json"))?;
        let text_config: qwen3::Config = read_json(&te.join("config.json"))?;

        // SAFETY: the snapshot is a read-only cache the daemon owns; the
        // files are not modified while mapped.
        let dit_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dit.join("model.safetensors")],
                dtype,
                &dit_device,
            )?
        };
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[vae.join("diffusion_pytorch_model.safetensors")],
                vae_dtype,
                &vae_device,
            )?
        };
        // The embedding model's tensors carry a `model.` prefix the bare
        // qwen3 encoder does not expect.
        let text_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[te.join("model.safetensors")],
                text_dtype,
                &text_device,
            )?
        }
        .rename_f(|n| n.strip_prefix("model.").unwrap_or(n).to_string());

        let silence_latent = super::arch::acestep::pipeline::load_silence_latent(
            &dit.join("silence_latent.pt"),
            dit_config.audio_acoustic_hidden_dim,
            dtype,
            &dit_device,
        )?;
        // The VAE builds outside the scope: its decoder is convolutions, and
        // the quantization scope only touches linear layers anyway.
        let mut pipeline = with_quant(layout.quant, || {
            AceStepPipeline::new(
                &dit_config,
                &vae_config,
                &text_config,
                dit_vb,
                vae_vb,
                text_vb,
                &te.join("tokenizer.json"),
                silence_latent,
                false,
            )
        })?;

        // The planner is optional and runs only when asked for, so a planner
        // that will not load leaves a working engine without one rather
        // than failing the model.
        let has_planner = match &files.planner {
            Some(dir) => match with_quant(layout.quant, || {
                Self::load_planner(
                    &mut pipeline,
                    &root.join(dir),
                    planner_dtype,
                    &planner_device,
                )
            }) {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(planner = %dir, error = %format!("{e:#}"), "song planner failed to load; serving without it");
                    false
                }
            },
            None => false,
        };
        tracing::info!(
            quant = ?layout.quant,
            dit = ?(dit_device.location(), dtype),
            text_encoder = ?(text_device.location(), text_dtype),
            vae = ?(vae_device.location(), vae_dtype),
            planner = ?has_planner.then(|| (planner_device.location(), planner_dtype)),
            budget_mb = ?layout.vram_budget_mb,
            "ACE-Step pipeline laid out"
        );
        Ok(Self {
            pipeline,
            has_planner,
            vae_chunk_frames: layout.vae_chunk_frames(),
            _ballast: ballast,
        })
    }

    pub fn has_planner(&self) -> bool {
        self.has_planner
    }

    /// Load the planner LM into the pipeline. The output head is tied to
    /// the embeddings.
    fn load_planner(
        pipeline: &mut AceStepPipeline,
        dir: &Path,
        dtype: DType,
        device: &Device,
    ) -> Result<()> {
        let lm_config: qwen3::Config = read_json(&dir.join("config.json"))?;
        let lm_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)?
        };
        // The 1.7B checkpoint stores the decoder without the `model.`
        // prefix; the 0.6B one keeps it. Strip it only where it is absent.
        let lm_vb = if lm_vb.contains_tensor("model.embed_tokens.weight") {
            lm_vb
        } else {
            lm_vb.rename_f(|n| n.strip_prefix("model.").unwrap_or(n).to_string())
        };
        pipeline.load_lm(&lm_config, lm_vb, &dir.join("tokenizer.json"))?;
        Ok(())
    }

    /// Render one song.
    pub fn generate(&mut self, p: &AudioGenParams) -> Result<AudioGenResult> {
        let mut timing = cortex_core::audio::AudioTiming::default();

        let (lm_hints, lm_metadata) = if p.planner && self.has_planner {
            let t = Instant::now();
            let cfg = LmConfig {
                temperature: p.temperature.unwrap_or(0.85),
                seed: p.seed,
                ..Default::default()
            };
            let (hints, meta) =
                self.pipeline
                    .generate_lm_hints(&p.prompt, &p.lyrics, p.duration_secs, &cfg)?;
            timing.plan_ms = t.elapsed().as_millis() as u64;
            (Some(hints), Some(lm_metadata(&meta)))
        } else {
            (None, None)
        };

        let params = GenerationParams {
            duration_secs: p.duration_secs,
            seed: Some(p.seed),
            num_steps: Some(p.steps.unwrap_or(DEFAULT_STEPS)),
            guidance_scale: p
                .guidance_scale
                .unwrap_or(GenerationParams::default().guidance_scale),
            shift: TURBO_SHIFT,
            is_turbo: true,
            lm_hints_25hz: lm_hints,
            lm_metadata,
            infer_method: sampling::InferMethod::Ode,
            language: p.language.clone(),
            vae_chunk_frames: self.vae_chunk_frames,
            ..Default::default()
        };
        let out = self.pipeline.text2music(&p.prompt, &p.lyrics, &params)?;
        let audio = out.audio.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        timing.encode_ms = out.timing.encode_ms;
        timing.generate_ms = out.timing.denoise_ms;
        timing.decode_ms = out.timing.decode_ms;

        let mut left = audio.get(0)?.to_vec1::<f32>()?;
        let mut right = audio.get(1)?.to_vec1::<f32>()?;
        if left.iter().chain(&right).any(|s| !s.is_finite()) {
            anyhow::bail!("generation produced non-finite samples");
        }
        peak_normalize(&mut left, &mut right, PEAK_DB);
        Ok(AudioGenResult {
            left,
            right,
            sample_rate: out.sample_rate,
            timing,
        })
    }
}

/// The plan's fields the DiT prompt carries.
fn lm_metadata(meta: &std::collections::BTreeMap<String, String>) -> LmMetadata {
    LmMetadata {
        bpm: meta.get("bpm").cloned(),
        timesignature: meta.get("timesignature").cloned(),
        keyscale: meta.get("keyscale").cloned(),
        caption: meta.get("caption").cloned(),
        language: meta
            .get("language")
            .or_else(|| meta.get("vocal_language"))
            .cloned(),
    }
}

/// Allocate all of `device`'s free memory except `budget_mb`, so what
/// loads next sees a card of the budget's size (#385). `None` on the CPU,
/// or when the device already has no more than the budget free.
#[cfg(feature = "cuda")]
fn hold_all_but(device: &Device, budget_mb: u64) -> Result<Option<candle_core::Tensor>> {
    if device.is_cpu() {
        return Ok(None);
    }
    // The device worker bound this device's context to the calling thread.
    let (free, _) = candle_core::cuda::cudarc::driver::result::mem_get_info()
        .map_err(|e| anyhow::anyhow!("mem_get_info: {e:?}"))?;
    let hold = (free as u64).saturating_sub(budget_mb * 1024 * 1024);
    if hold == 0 {
        return Ok(None);
    }
    tracing::warn!(
        held_mb = hold / (1024 * 1024),
        budget_mb,
        "holding device memory so the audio pipeline sees only its budget"
    );
    let t = candle_core::Tensor::zeros(hold as usize, DType::U8, device)?;
    Ok(Some(t))
}

#[cfg(not(feature = "cuda"))]
fn hold_all_but(_device: &Device, _budget_mb: u64) -> Result<Option<candle_core::Tensor>> {
    Ok(None)
}

// ── HeartMuLa (#393) ─────────────────────────────────────────────────

/// What HeartMuLa oss-3B serves: songs of 10 s to 4 min, as the evaluation
/// beta did. It is autoregressive (12.5 frames per second), so there is no
/// step knob.
pub const HEARTMULA_LIMITS: cortex_core::audio::AudioLimits = cortex_core::audio::AudioLimits {
    min_duration: 10.0,
    max_duration: 240.0,
    default_duration: 30.0,
    vocals: true,
    sample_rate: 48_000,
    channels: 2,
    max_steps: None,
};

/// The codec and the generation config live in sibling repos of the LM.
pub const HEARTCODEC_REPO: &str = "HeartMuLa/HeartCodec-oss-20260123";
pub const HEARTMULA_GEN_REPO: &str = "HeartMuLa/HeartMuLaGen";

/// heartlib's defaults, as the beta used them.
const HEART_CFG: f64 = 1.5;
const HEART_TEMPERATURE: f64 = 1.0;
const HEART_TOP_K: usize = 50;
/// Section tags with no words: how an instrumental is asked of HeartMuLa.
const HEART_INSTRUMENTAL: &str =
    "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Bridge]\n\n[Chorus]\n\n[Outro]";
/// Latent frames per scalar-decoder chunk: bounds decode memory at any length.
const HEART_DECODE_CHUNK: usize = 256;

/// True when a `config.json` describes a HeartMuLa LM. Its weight files
/// look like any dense LLM's, so the config is what tells them apart.
pub fn is_heartmula_config(config_json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(config_json)
        .ok()
        .and_then(|v| {
            v.get("model_type")
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .is_some_and(|t| t == "heartmula")
}

/// A resolved HeartMuLa model: the LM, the codec, and the generation files.
#[derive(Debug, Clone)]
pub struct HeartMuLaFiles {
    pub lm_dir: PathBuf,
    pub codec_dir: PathBuf,
    pub tokenizer: PathBuf,
    pub gen_config: PathBuf,
}

/// HeartMuLa's LM and codec, owned by the device worker. Both stay
/// resident: the LM in bf16 on a GPU (f32 on the CPU), the codec in f32 as
/// heartlib recommends.
pub struct HeartMuLaEngine {
    lm: super::arch::heartmula::lm::HeartMuLa,
    codec: super::arch::heartmula::codec::HeartCodec,
    tokenizer: tokenizers::Tokenizer,
    gen_cfg: super::arch::heartmula::lm::GenConfig,
}

fn safetensors_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    v.sort();
    anyhow::ensure!(!v.is_empty(), "no safetensors in {}", dir.display());
    Ok(v)
}

impl HeartMuLaEngine {
    pub fn load(files: &HeartMuLaFiles, device: &Device) -> Result<Self> {
        use super::arch::heartmula::{codec, lm};
        let lm_dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };
        let lm_config: lm::Config = read_json(&files.lm_dir.join("config.json"))?;
        // SAFETY: read-only snapshot files the daemon owns.
        let lm_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_in(&files.lm_dir)?, lm_dtype, device)?
        };
        let lm = lm::HeartMuLa::new(lm_config, lm_vb)?;
        let codec_config: codec::Config = read_json(&files.codec_dir.join("config.json"))?;
        let codec_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &safetensors_in(&files.codec_dir)?,
                DType::F32,
                device,
            )?
        };
        let codec = codec::HeartCodec::new(codec_config, codec_vb)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        let gen_cfg: lm::GenConfig = read_json(&files.gen_config)?;
        Ok(Self {
            lm,
            codec,
            tokenizer,
            gen_cfg,
        })
    }

    /// Render one song: tags and lyrics through the LM, then the codec.
    pub fn generate(&mut self, p: &AudioGenParams) -> Result<AudioGenResult> {
        use super::arch::heartmula::{codec, lm};
        let mut timing = cortex_core::audio::AudioTiming::default();
        let lyrics = if p.lyrics.trim().is_empty() {
            HEART_INSTRUMENTAL
        } else {
            p.lyrics.as_str()
        };
        let t = Instant::now();
        let prompt = lm::build_prompt(&self.tokenizer, &p.prompt, lyrics, &self.gen_cfg)?;
        timing.encode_ms = t.elapsed().as_millis() as u64;

        let t = Instant::now();
        let max_frames = (p.duration_secs * codec::CODE_RATE).round() as usize;
        let mut sampler = lm::Sampler::new(
            p.temperature.unwrap_or(HEART_TEMPERATURE),
            p.top_k.unwrap_or(HEART_TOP_K),
            p.seed,
        );
        let frames = self.lm.generate(
            &prompt,
            max_frames,
            p.guidance_scale.unwrap_or(HEART_CFG),
            &self.gen_cfg,
            &mut sampler,
        )?;
        timing.generate_ms = t.elapsed().as_millis() as u64;
        anyhow::ensure!(!frames.is_empty(), "the model produced no audio frames");

        let t = Instant::now();
        let n = frames.len();
        let flat: Vec<u32> = (0..8)
            .flat_map(|q| frames.iter().map(move |f| f[q]))
            .collect();
        let codes = candle_core::Tensor::from_vec(flat, (8, n), self.codec.device())?;
        let device = self.codec.device().clone();
        let out_channels = self.codec.config.out_channels;
        let mut segment = 0u64;
        let seed = p.seed;
        let mut noise = |len: usize| {
            segment += 1;
            super::arch::acestep::model::seeded_normal(
                Some(seed.wrapping_add(segment)),
                (1, len, out_channels),
            )?
            .to_device(&device)
        };
        let audio = self.codec.detokenize(
            &codes,
            codec::DEFAULT_STEPS,
            codec::DEFAULT_GUIDANCE,
            &mut noise,
            Some(HEART_DECODE_CHUNK),
        )?;
        let audio = audio.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        timing.decode_ms = t.elapsed().as_millis() as u64;

        let mut left = audio.get(0)?.to_vec1::<f32>()?;
        let mut right = audio.get(1)?.to_vec1::<f32>()?;
        if left.iter().chain(&right).any(|s| !s.is_finite()) {
            anyhow::bail!("generation produced non-finite samples");
        }
        for s in left.iter_mut().chain(right.iter_mut()) {
            *s = s.clamp(-1.0, 1.0);
        }
        Ok(AudioGenResult {
            left,
            right,
            sample_rate: self.codec.config.sample_rate as u32,
            timing,
        })
    }
}

// ── Engine selection ─────────────────────────────────────────────────

/// What the worker is asked to load.
#[derive(Debug, Clone)]
pub enum AudioFiles {
    AceStep(AceStepFiles),
    HeartMuLa(HeartMuLaFiles),
}

/// A loaded music engine in the worker's audio slab.
pub enum AudioEngine {
    AceStep(Box<AceStepEngine>),
    HeartMuLa(Box<HeartMuLaEngine>),
}

impl AudioEngine {
    pub fn load(files: &AudioFiles, device: &Device) -> Result<Self> {
        Ok(match files {
            AudioFiles::AceStep(f) => Self::AceStep(Box::new(AceStepEngine::load(f, device)?)),
            AudioFiles::HeartMuLa(f) => {
                Self::HeartMuLa(Box::new(HeartMuLaEngine::load(f, device)?))
            }
        })
    }

    pub fn generate(&mut self, p: &AudioGenParams) -> Result<AudioGenResult> {
        match self {
            Self::AceStep(e) => e.generate(p),
            Self::HeartMuLa(e) => e.generate(p),
        }
    }

    pub fn has_planner(&self) -> bool {
        match self {
            Self::AceStep(e) => e.has_planner(),
            Self::HeartMuLa(_) => false,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::AceStep(_) => "ace-step",
            Self::HeartMuLa(_) => "heartmula",
        }
    }
}

/// Scale both channels so the loudest sample sits at `target_db` dBFS, as
/// upstream does before saving. Near-silence is left alone rather than
/// amplified into noise.
pub fn peak_normalize(left: &mut [f32], right: &mut [f32], target_db: f64) {
    let peak = left
        .iter()
        .chain(right.iter())
        .fold(0f32, |m, s| m.max(s.abs()));
    if peak < 1e-6 {
        return;
    }
    let gain = (10f64.powf(target_db / 20.0) / peak as f64) as f32;
    for s in left.iter_mut().chain(right.iter_mut()) {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
}

/// Encode stereo samples as 16-bit FLAC. Lossless at the 16-bit depth the
/// samples are quantised to, and roughly half the size of the same PCM.
pub fn encode_flac_pcm16(left: &[f32], right: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let frames = left.len().min(right.len());
    let pcm = |s: f32| (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i32;
    let mut interleaved = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        interleaved.push(pcm(left[i]));
        interleaved.push(pcm(right[i]));
    }
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, e)| anyhow::anyhow!("flac encoder config: {e:?}"))?;
    let source =
        flacenc::source::MemSource::from_samples(&interleaved, 2, 16, sample_rate as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| anyhow::anyhow!("flac encode: {e:?}"))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| anyhow::anyhow!("flac write: {e:?}"))?;
    Ok(sink.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(toml: &str) -> Result<AceStepLayout> {
        let cfg: crate::config::AudioConfig = toml::from_str(toml).unwrap();
        AceStepLayout::from_config(&cfg)
    }

    #[test]
    fn the_default_layout_is_bf16_on_the_gpu() {
        let l = layout("").unwrap();
        assert_eq!(l.quant, None);
        // A CPU stands in for the GPU here; what matters is the dtype rule.
        let gpu = Device::Cpu;
        assert_eq!(
            l.component(Place::Gpu, &gpu).1,
            DType::F32,
            "CPU is always f32"
        );
        assert_eq!(l.vae_chunk_frames(), None);
    }

    #[test]
    fn the_pascal_recipe_parses_to_q8_with_f32_compute_and_a_budget() {
        let l = layout(
            r#"
            quant = "q8_0"
            vram_budget_mb = 8192
            enforce_vram_budget = true
            [placement]
            text_encoder = "cpu"
            "#,
        )
        .unwrap();
        assert_eq!(l.quant, Some(candle_core::quantized::GgmlDType::Q8_0));
        assert_eq!(l.text_encoder, Place::Cpu);
        assert_eq!(l.dit, Place::Gpu);
        assert_eq!(l.vae_chunk_frames(), Some(64));
    }

    #[test]
    fn layout_errors_name_the_problem() {
        let e = format!("{:#}", layout(r#"quant = "q4_k""#).unwrap_err());
        assert!(e.contains("q4_k"), "{e}");
        let e = format!("{:#}", layout(r#"compute = "f16""#).unwrap_err());
        assert!(e.contains("f16"), "{e}");
        let e = format!("{:#}", layout("[placement]\ndit = \"cuda:1\"").unwrap_err());
        assert!(e.contains("dit") && e.contains("cuda:1"), "{e}");
        let e = format!("{:#}", layout("enforce_vram_budget = true").unwrap_err());
        assert!(e.contains("vram_budget_mb"), "{e}");
    }

    #[test]
    fn heartmula_is_recognised_by_its_config() {
        assert!(is_heartmula_config(
            r#"{"architectures": ["HeartMuLa"], "model_type": "heartmula"}"#
        ));
        assert!(!is_heartmula_config(r#"{"model_type": "qwen3"}"#));
        assert!(!is_heartmula_config("not json"));
    }

    #[test]
    fn layout_detection() {
        let mut files: Vec<String> = required_files();
        files.push("config.json".into());
        let refs: Vec<&str> = files.iter().map(String::as_str).collect();
        assert!(is_acestep_layout(&refs));
        assert!(!is_acestep_layout(&["config.json", "model.safetensors"]));
        assert!(!is_acestep_layout(&["acestep-v15-turbo/config.json"]));
    }

    #[test]
    fn peak_normalize_hits_target_and_spares_silence() {
        let (mut l, mut r) = (vec![0.5, -0.25], vec![0.1, 0.0]);
        peak_normalize(&mut l, &mut r, -1.0);
        let target = 10f32.powf(-1.0 / 20.0);
        assert!((l[0] - target).abs() < 1e-6);
        assert!((l[1] + target / 2.0).abs() < 1e-6);
        let (mut l, mut r) = (vec![1e-8, 0.0], vec![0.0, -1e-8]);
        peak_normalize(&mut l, &mut r, -1.0);
        assert_eq!(l, vec![1e-8, 0.0]);
    }

    #[test]
    fn flac_round_trips_the_16_bit_samples() {
        // A second of a decaying tone, plus the clamp at both rails.
        let n = 48_000;
        let left: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.05).sin() * (1.0 - i as f32 / n as f32))
            .collect();
        let mut right: Vec<f32> = left.iter().map(|s| s * -0.5).collect();
        right[0] = 2.0;
        right[1] = -2.0;
        let flac = encode_flac_pcm16(&left, &right, 48_000).unwrap();
        assert_eq!(&flac[..4], b"fLaC");
        assert!(
            flac.len() < n * 4,
            "{} bytes is no smaller than PCM",
            flac.len()
        );

        let mut reader = claxon::FlacReader::new(std::io::Cursor::new(&flac)).unwrap();
        let info = reader.streaminfo();
        assert_eq!(
            (info.sample_rate, info.channels, info.bits_per_sample),
            (48_000, 2, 16)
        );
        assert_eq!(info.samples, Some(n as u64));
        let decoded: Vec<i32> = reader.samples().map(Result::unwrap).collect();
        let pcm = |s: f32| (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i32;
        let expected: Vec<i32> = (0..n).flat_map(|i| [pcm(left[i]), pcm(right[i])]).collect();
        assert_eq!(decoded, expected);
        assert_eq!(
            (decoded[1], decoded[3]),
            (i16::MAX as i32, -(i16::MAX as i32))
        );
    }
}

/// End-to-end render against real weights, on the CPU. Ignored by default:
/// it needs an ACE-Step 1.5 snapshot on disk and minutes of CPU time.
///
///   ACESTEP_SNAPSHOT=/path/to/Ace-Step1.5 \
///     cargo test -p neuron --lib -- --ignored acestep_renders_on_cpu --nocapture
#[cfg(test)]
mod hardware {
    use super::*;
    use crate::harness::arch::acestep::pipeline::AceStepPipeline;

    #[test]
    #[ignore = "needs ACESTEP_SNAPSHOT and minutes of CPU"]
    fn acestep_renders_on_cpu() {
        let Ok(root) = std::env::var("ACESTEP_SNAPSHOT") else {
            eprintln!("ACESTEP_SNAPSHOT not set; skipping");
            return;
        };
        let files = AceStepFiles {
            root: root.into(),
            planner: None,
            layout: AceStepLayout::default(),
        };
        let t = Instant::now();
        let mut engine = AceStepEngine::load(&files, &Device::Cpu).expect("load");
        eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());
        let params = AudioGenParams {
            prompt: "Warm lo-fi hip hop beat with mellow Rhodes chords, 82 BPM".into(),
            lyrics: String::new(),
            duration_secs: 10.0,
            seed: 42,
            steps: None,
            guidance_scale: None,
            planner: false,
            temperature: None,
            language: None,
            top_k: None,
        };
        let t = Instant::now();
        let out = engine.generate(&params).expect("generate");
        eprintln!(
            "rendered {:.2}s of audio in {:.1}s ({:?})",
            out.duration_secs(),
            t.elapsed().as_secs_f32(),
            out.timing
        );
        assert_eq!(out.sample_rate, 48_000);
        assert!(
            (out.duration_secs() - 10.0).abs() < 0.5,
            "{}",
            out.duration_secs()
        );
        let rms = (out.left.iter().map(|s| s * s).sum::<f32>() / out.left.len() as f32).sqrt();
        assert!(rms > 1e-3, "silent render, rms {rms}");
        if let Ok(path) = std::env::var("ACESTEP_OUT") {
            std::fs::write(
                &path,
                encode_flac_pcm16(&out.left, &out.right, out.sample_rate).expect("encode flac"),
            )
            .expect("write flac");
            eprintln!("wrote {path}");
        }
        // Same seed, same song.
        let again = engine.generate(&params).expect("generate again");
        assert_eq!(out.left, again.left, "same seed must render identically");
    }

    /// Relative L2 distance between two tensors of the same shape.
    fn rel_l2(a: &candle_core::Tensor, b: &candle_core::Tensor) -> f32 {
        let a = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let b = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let diff = (&a - &b)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let norm = b
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        (diff / norm.max(1e-20)).sqrt()
    }

    /// Parity with the upstream PyTorch reference (#381) on the CPU in f32,
    /// from the reference's own noise: prompt token ids exactly, then the
    /// denoised latents and the decoded waveform within float tolerance.
    #[test]
    #[ignore = "needs ACESTEP_SNAPSHOT and minutes of CPU"]
    fn acestep_matches_reference_cpu_f32() {
        let Ok(root) = std::env::var("ACESTEP_SNAPSHOT") else {
            eprintln!("ACESTEP_SNAPSHOT not set; skipping");
            return;
        };
        let cases = std::env::var("ACESTEP_CASES")
            .unwrap_or_else(|_| "instr_10s_noplan,vocal_es_20s_noplan".into());
        // ACESTEP_LAYOUT (a `[harness.candle.audio]` body) and
        // ACESTEP_DEVICE=cuda measure another layout's drift from the
        // reference (#385). Only the default layout on the CPU is held to
        // parity; any other is reported, and held to a drift bound.
        let layout_toml = std::env::var("ACESTEP_LAYOUT").unwrap_or_default();
        let cfg: crate::config::AudioConfig = toml::from_str(&layout_toml).unwrap();
        let device = match std::env::var("ACESTEP_DEVICE").as_deref() {
            Ok("cuda") => Device::new_cuda(0).unwrap(),
            _ => Device::Cpu,
        };
        let parity = layout_toml.trim().is_empty() && device.is_cpu();
        // (case, this layout's drift, upstream bf16's drift)
        let mut drifts: Vec<(String, f32, f32)> = Vec::new();
        // A planned case (`*_plan`) runs the planner greedily first, as the
        // trace did; its plan and codes then condition the DiT (#384).
        let planned = cases.split(',').any(|c| c.ends_with("_plan"));
        let mut engine = AceStepEngine::load(
            &AceStepFiles {
                root: root.into(),
                planner: planned.then(|| PLANNER_DIRS[0].to_string()),
                layout: AceStepLayout::from_config(&cfg).unwrap(),
            },
            &device,
        )
        .expect("load");
        assert_eq!(engine.has_planner(), planned, "planner load");
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/harness/testdata/acestep");
        for case in cases.split(',') {
            let meta: serde_json::Value = serde_json::from_slice(
                &std::fs::read(dir.join(format!("{case}__cpu_f32.json"))).unwrap(),
            )
            .unwrap();
            // Read tensors by name: the file also holds BOOL masks, which
            // candle cannot load.
            let file = unsafe {
                candle_core::safetensors::MmapedSafetensors::new(
                    dir.join(format!("{case}__cpu_f32.safetensors")),
                )
            }
            .unwrap();
            let want: std::collections::HashMap<&str, candle_core::Tensor> = [
                "text_token_ids",
                "lyric_token_ids",
                "noise",
                "target_latents",
                "vae_waveform",
            ]
            .into_iter()
            .map(|k| (k, file.load(k, &Device::Cpu).unwrap()))
            .collect();
            let c = &meta["case"];
            let caption = c["caption"].as_str().unwrap();
            let lyrics = c["lyrics"].as_str().unwrap();
            let language = c["language"].as_str().unwrap();
            let duration = c["duration"].as_f64().unwrap();

            let (hints, metadata) = if c["planner"].as_bool() == Some(true) {
                let cfg = crate::harness::arch::acestep::lm::LmConfig {
                    temperature: 0.0,
                    ..Default::default()
                };
                // ACESTEP_PLAN_FROM_TRACE takes the trace's own codes and
                // plan instead of running the planner, to test what follows
                // the planner on its own (and in seconds rather than
                // minutes).
                let (hints, meta) = if std::env::var("ACESTEP_PLAN_FROM_TRACE").is_ok() {
                    let codes = file
                        .load("planner_audio_codes", &Device::Cpu)
                        .unwrap()
                        .to_vec1::<i64>()
                        .unwrap();
                    let cot = meta["planner"][0]["output_text"].as_str().unwrap();
                    (
                        engine.pipeline.audio_codes_to_latents(&codes).unwrap(),
                        crate::harness::arch::acestep::lm::LmPipeline::parse_metadata(cot),
                    )
                } else {
                    engine
                        .pipeline
                        .generate_lm_hints(caption, lyrics, duration, &cfg)
                        .unwrap()
                };
                let want_hints = file
                    .load("gen_precomputed_lm_hints_25Hz", &Device::Cpu)
                    .unwrap();
                eprintln!(
                    "{case}: lm hints rel L2 {:.2e}",
                    rel_l2(&hints.to_device(&Device::Cpu).unwrap(), &want_hints)
                );
                (Some(hints), Some(lm_metadata(&meta)))
            } else {
                (None, None)
            };
            let instruction = if hints.is_some() {
                crate::harness::arch::acestep::pipeline::COVER_INSTRUCTION
            } else {
                crate::harness::arch::acestep::pipeline::TEXT2MUSIC_INSTRUCTION
            };
            let (text, lyric) = AceStepPipeline::format_prompts(
                instruction,
                caption,
                lyrics,
                duration,
                metadata.as_ref(),
                Some(language),
            );
            let (text_ids, lyric_ids) = engine.pipeline.prompt_token_ids(&text, &lyric).unwrap();
            let ids = |k: &str| -> Vec<u32> {
                want[k]
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<i64>()
                    .unwrap()
                    .into_iter()
                    .map(|v| v as u32)
                    .collect()
            };
            assert_eq!(text_ids, ids("text_token_ids"), "{case}: text token ids");
            assert_eq!(lyric_ids, ids("lyric_token_ids"), "{case}: lyric token ids");

            let params = GenerationParams {
                duration_secs: duration,
                seed: None,
                num_steps: Some(DEFAULT_STEPS),
                shift: TURBO_SHIFT,
                is_turbo: true,
                infer_method: sampling::InferMethod::Ode,
                initial_noise: Some(want["noise"].clone()),
                language: Some(language.to_string()),
                lm_hints_25hz: hints,
                lm_metadata: metadata,
                ..Default::default()
            };
            let t = Instant::now();
            let out = engine
                .pipeline
                .text2music(caption, lyrics, &params)
                .unwrap();
            let cpu = |t: &candle_core::Tensor| t.to_device(&Device::Cpu).unwrap();
            let latents = rel_l2(&cpu(&out.latents), &want["target_latents"]);
            let wave_ref = &want["vae_waveform"];
            let n = wave_ref.dim(2).unwrap();
            let wave = cpu(&out.audio)
                .narrow(1, 0, n)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let waveform = rel_l2(&wave, wave_ref);
            eprintln!(
                "{case} [{layout_toml:?} on {:?}]: latents rel L2 {latents:.2e}, waveform rel L2 {waveform:.2e} ({:.1}s)",
                device.location(),
                t.elapsed().as_secs_f32()
            );
            if parity {
                assert!(latents < 1e-3, "{case}: latents rel L2 {latents}");
                assert!(waveform < 1e-2, "{case}: waveform rel L2 {waveform}");
            } else {
                // The yardstick is upstream's own: how far its bf16 GPU run
                // of the same case lands from its f32 CPU run. (Its f16 run
                // of a vocal case is NaN.)
                let envelope_file = unsafe {
                    candle_core::safetensors::MmapedSafetensors::new(
                        dir.join(format!("{case}__cuda_bf16.safetensors")),
                    )
                }
                .unwrap();
                let upstream_bf16 = envelope_file.load("target_latents", &Device::Cpu).unwrap();
                let envelope = rel_l2(&upstream_bf16, &want["target_latents"]);
                eprintln!("{case}: upstream bf16 drifts {envelope:.2e}");
                drifts.push((case.to_string(), latents, envelope));
            }
        }
        // Turbo denoising is chaotic: per-case drift swings with tiny
        // numeric differences, so a layout is judged on its mean across
        // cases against upstream bf16's mean, and every case must be
        // finite.
        if !drifts.is_empty() {
            let n = drifts.len() as f32;
            let mean = drifts.iter().map(|d| d.1).sum::<f32>() / n;
            let envelope = drifts.iter().map(|d| d.2).sum::<f32>() / n;
            eprintln!("mean latents rel L2 {mean:.2e}; upstream bf16 mean {envelope:.2e}");
            assert!(
                drifts.iter().all(|d| d.1.is_finite()),
                "non-finite drift: {drifts:?}"
            );
            assert!(
                mean <= envelope,
                "mean drift {mean} beyond upstream bf16's {envelope}: {drifts:?}"
            );
        }
    }

    /// End-to-end HeartMuLa render on the CPU from the real checkpoints,
    /// laid out as heartlib does (`<ckpt>/{HeartMuLa-oss-3B,HeartCodec-oss,
    /// tokenizer.json,gen_config.json}`). About 15.7 GB of f32 LM plus the
    /// codec, so a host with 32 GB or more.
    #[test]
    #[ignore = "needs HEARTMULA_CKPT, ~24 GB of RAM and minutes of CPU"]
    fn heartmula_renders_on_cpu() {
        let Ok(root) = std::env::var("HEARTMULA_CKPT") else {
            eprintln!("HEARTMULA_CKPT not set; skipping");
            return;
        };
        let root = PathBuf::from(root);
        let files = HeartMuLaFiles {
            lm_dir: root.join("HeartMuLa-oss-3B"),
            codec_dir: root.join("HeartCodec-oss"),
            tokenizer: root.join("tokenizer.json"),
            gen_config: root.join("gen_config.json"),
        };
        let t = Instant::now();
        let mut engine =
            AudioEngine::load(&AudioFiles::HeartMuLa(files), &Device::Cpu).expect("load");
        eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());
        let params = AudioGenParams {
            prompt: "indie pop, bright guitars, female vocals".into(),
            lyrics: "[Verse]\nStreetlights hum a quiet tune".into(),
            duration_secs: 10.0,
            seed: 7,
            steps: None,
            guidance_scale: None,
            planner: false,
            temperature: None,
            language: None,
            top_k: None,
        };
        let t = Instant::now();
        let out = engine.generate(&params).expect("generate");
        eprintln!(
            "rendered {:.2}s in {:.1}s ({:?})",
            out.duration_secs(),
            t.elapsed().as_secs_f32(),
            out.timing
        );
        assert_eq!(out.sample_rate, 48_000);
        assert!(
            out.duration_secs() > 1.0 && out.duration_secs() <= 10.5,
            "{}",
            out.duration_secs()
        );
        let rms = (out.left.iter().map(|s| s * s).sum::<f32>() / out.left.len() as f32).sqrt();
        assert!(rms > 1e-3, "silent render, rms {rms}");
        if let Ok(path) = std::env::var("HEARTMULA_OUT") {
            std::fs::write(
                &path,
                encode_flac_pcm16(&out.left, &out.right, out.sample_rate).unwrap(),
            )
            .unwrap();
        }
        let again = engine.generate(&params).expect("generate again");
        assert_eq!(out.left, again.left, "same seed must render identically");
    }

    /// Planner parity with upstream (#384): greedy decoding in f32 over the
    /// `vocal_en_30s_plan` case must reproduce upstream's CoT text exactly,
    /// and its audio codes (CFG 2.0, 150 of them). Ignored: needs
    /// ACESTEP_SNAPSHOT; ACESTEP_DEVICE=cuda runs it on the GPU.
    #[test]
    #[ignore = "needs ACESTEP_SNAPSHOT"]
    fn acestep_planner_matches_reference() {
        use crate::harness::arch::acestep::lm::{LmConfig, LmPipeline, TokenIds};
        let Ok(root) = std::env::var("ACESTEP_SNAPSHOT") else {
            return;
        };
        let device = match std::env::var("ACESTEP_DEVICE").as_deref() {
            Ok("cuda") => Device::new_cuda(0).unwrap(),
            _ => Device::Cpu,
        };
        let dir = PathBuf::from(root).join(PLANNER_DIRS[0]);
        let cfg: qwen3::Config = read_json(&dir.join("config.json")).unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &device,
            )
            .unwrap()
        }
        .rename_f(|n| n.strip_prefix("model.").unwrap_or(n).to_string());
        let model = qwen3::ModelForCausalLM::new(&cfg, vb).unwrap();
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let encode = |s: &str| -> candle_core::Result<Vec<u32>> {
            Ok(tok.encode(s, false).unwrap().get_ids().to_vec())
        };
        let decode =
            |ids: &[u32]| -> candle_core::Result<String> { Ok(tok.decode(ids, false).unwrap()) };
        let ids = TokenIds::discover(encode).unwrap();
        let mut lm = LmPipeline::new(model, ids, device);

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/harness/testdata/acestep/vocal_en_30s_plan__cpu_f32.json");
        let f: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let case = &f["case"];
        let config = LmConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let t = Instant::now();
        let out = lm
            .generate(
                case["caption"].as_str().unwrap(),
                case["lyrics"].as_str().unwrap(),
                case["duration"].as_f64().unwrap(),
                &config,
                &decode,
                &encode,
            )
            .unwrap();
        eprintln!("planner ran in {:.1}s", t.elapsed().as_secs_f32());

        let want_cot = f["planner"][0]["output_text"].as_str().unwrap();
        // raw_text is the CoT, a newline, then the re-dumped plan and codes.
        let got_cot = &out.raw_text[..want_cot.len().min(out.raw_text.len())];
        assert_eq!(got_cot, want_cot, "CoT text");

        let want_codes = crate::harness::arch::acestep::lm::parse_audio_codes_from_text(
            f["planner"][1]["output_text"].as_str().unwrap(),
        );
        let same = out
            .audio_codes
            .iter()
            .zip(&want_codes)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "codes: {} generated, {} expected, first {same} identical",
            out.audio_codes.len(),
            want_codes.len()
        );
        assert_eq!(out.audio_codes.len(), want_codes.len(), "code count");
        assert_eq!(out.audio_codes, want_codes, "audio codes");
    }

    /// The planner LM loads from the real snapshot (its tensors carry no
    /// `model.` prefix). Ignored: needs ACESTEP_SNAPSHOT.
    #[test]
    #[ignore = "needs ACESTEP_SNAPSHOT"]
    fn acestep_planner_loads() {
        let Ok(root) = std::env::var("ACESTEP_SNAPSHOT") else {
            return;
        };
        let dir = PathBuf::from(root).join(PLANNER_DIRS[0]);
        let cfg: qwen3::Config = read_json(&dir.join("config.json")).unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::BF16,
                &Device::Cpu,
            )
            .unwrap()
        }
        .rename_f(|n| n.strip_prefix("model.").unwrap_or(n).to_string());
        qwen3::ModelForCausalLM::new(&cfg, vb).expect("planner loads");
    }

    /// Peak device memory and latency per song length, for a layout
    /// (#385). Run on a CUDA host:
    ///
    ///   ACESTEP_SNAPSHOT=… ACESTEP_LAYOUT='quant = "q8_0"' \
    ///   ACESTEP_PLANNER=acestep-5Hz-lm-1.7B ACESTEP_DURATIONS=30,180 \
    ///     cargo test -p neuron --features cuda --lib -- --ignored \
    ///     acestep_profile_gpu --nocapture
    ///
    /// `ACESTEP_PLANNER` is a snapshot subfolder or an absolute path; set,
    /// the planner runs. Peak memory is this process's, as nvidia-smi
    /// reports it, CUDA context included, so it is what a card of that
    /// size would have to hold. The pool is trimmed between songs, as the
    /// device worker does after every job.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs ACESTEP_SNAPSHOT and a CUDA device"]
    fn acestep_profile_gpu() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        let Ok(root) = std::env::var("ACESTEP_SNAPSHOT") else {
            eprintln!("ACESTEP_SNAPSHOT not set; skipping");
            return;
        };
        // The engine reports a planner that fails to load as a warning and
        // carries on without it; show it.
        let _ = tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .with_env_filter(tracing_subscriber::EnvFilter::new("neuron=info"))
            .try_init();
        let cfg: crate::config::AudioConfig =
            toml::from_str(&std::env::var("ACESTEP_LAYOUT").unwrap_or_default()).unwrap();
        let planner = std::env::var("ACESTEP_PLANNER").ok();
        let durations: Vec<f64> = std::env::var("ACESTEP_DURATIONS")
            .unwrap_or_else(|_| "30,120,240,480".into())
            .split(',')
            .map(|d| d.trim().parse().unwrap())
            .collect();

        let pid = std::process::id();
        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let sampler = {
            let (peak, stop) = (peak.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let out = std::process::Command::new("nvidia-smi")
                        .args([
                            "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits",
                        ])
                        .output()
                        .unwrap();
                    for line in String::from_utf8_lossy(&out.stdout).lines() {
                        let mut f = line.split(',').map(str::trim);
                        if f.next() == Some(pid.to_string().as_str())
                            && let Some(mb) = f.next().and_then(|m| m.parse::<u64>().ok())
                        {
                            peak.fetch_max(mb, Ordering::Relaxed);
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            })
        };
        let trim = || {
            use candle_core::cuda::cudarc::driver::result::{ctx, device, mem_pool};
            unsafe {
                ctx::synchronize().unwrap();
                let pool = device::get_default_mem_pool(device::get(0).unwrap()).unwrap();
                mem_pool::trim_to(pool, 0).unwrap();
            }
        };

        let device = Device::new_cuda(0).unwrap();
        let files = AceStepFiles {
            root: root.into(),
            planner: planner.clone(),
            layout: AceStepLayout::from_config(&cfg).unwrap(),
        };
        let t = Instant::now();
        let mut engine = AceStepEngine::load(&files, &device).expect("load");
        assert_eq!(
            engine.has_planner(),
            planner.is_some(),
            "the planner asked for did not load"
        );
        trim();
        std::thread::sleep(std::time::Duration::from_millis(300));
        eprintln!(
            "PROFILE load layout={:?} planner={:?} secs={:.1} resident_mb={}",
            std::env::var("ACESTEP_LAYOUT").unwrap_or_default(),
            planner,
            t.elapsed().as_secs_f32(),
            peak.swap(0, Ordering::Relaxed)
        );
        for d in durations {
            let params = AudioGenParams {
                prompt: "Upbeat synth-pop with bright female vocals, punchy drums, 120 BPM".into(),
                lyrics: "[verse]\nCity lights are calling out my name\nEvery street a spark, every night a flame\n\
                         [chorus]\nWe run, we run, until the morning comes\nHearts like drums, we run\n"
                    .into(),
                duration_secs: d,
                seed: 7,
                steps: None,
                guidance_scale: None,
                planner: planner.is_some(),
                temperature: None,
                language: Some("en".into()),
                top_k: None,
            };
            let t = Instant::now();
            let res = engine.generate(&params);
            let secs = t.elapsed().as_secs_f32();
            std::thread::sleep(std::time::Duration::from_millis(300));
            let mb = peak.swap(0, Ordering::Relaxed);
            match res {
                Ok(out) => {
                    let rms = (out.left.iter().map(|s| s * s).sum::<f32>()
                        / out.left.len().max(1) as f32)
                        .sqrt();
                    eprintln!(
                        "PROFILE song secs={d} wall={secs:.1} plan_ms={} encode_ms={} denoise_ms={} decode_ms={} peak_mb={mb} rms={rms:.4}",
                        out.timing.plan_ms,
                        out.timing.encode_ms,
                        out.timing.generate_ms,
                        out.timing.decode_ms
                    );
                    if let Ok(dir) = std::env::var("ACESTEP_OUT") {
                        std::fs::write(
                            format!("{dir}/song_{d}.flac"),
                            encode_flac_pcm16(&out.left, &out.right, out.sample_rate).unwrap(),
                        )
                        .unwrap();
                    }
                }
                Err(e) => {
                    eprintln!("PROFILE song secs={d} wall={secs:.1} peak_mb={mb} error={e:#}")
                }
            }
            trim();
        }
        stop.store(true, Ordering::Relaxed);
        sampler.join().unwrap();
    }
}
