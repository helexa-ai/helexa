#!/usr/bin/env python3
"""Record ACE-Step 1.5 reference traces for neuron's parity tests (#381).

Runs the upstream ACE-Step 1.5 pipeline (`acestep.inference.generate_music`,
the same entry point its own CLI and API server use) over a fixed set of
cases. It hooks every component boundary a port has to reproduce and saves
what crosses each one. Per case and per trace it writes:

  - the DiT text prompt and lyric prompt, exactly as tokenized, with token ids
  - the text-encoder output (Qwen3-Embedding-0.6B `last_hidden_state`) and
    the lyric embeddings (the same model's `embed_tokens` lookup only)
  - the lyric-encoder output and the packed condition-encoder output
    (lyrics + timbre + projected text), with its mask
  - the planner (5 Hz LM): each phase's formatted prompt, input ids, output
    ids and decoded text, plus the parsed metadata and audio codes
  - `generate_audio` inputs: src/context latents, chunk masks, cover flags,
    LM hints, and the resolved sampler settings (schedule, shift, DCW)
  - the initial noise, then DiT input (`xt`) and output (`vt`) per step
  - the final latent, the VAE input, and the decoded waveform (truncated)

Reference revisions (what the committed fixtures were recorded against):

    code     https://github.com/ace-step/ACE-Step-1.5 @ ca1e85fe9430179831e6bc6be790c332190a3866
    weights  ACE-Step/Ace-Step1.5 @ 19671f406d603126926c1b7e2adc169acbcade22
             (acestep-v15-turbo, acestep-5Hz-lm-1.7B, Qwen3-Embedding-0.6B, vae)
    torch 2.10.0+cu128, transformers 4.57.6, pytorch_wavelets 1.3.0

Traces (`--trace`):

    cpu_f32     float32 on CPU: the parity reference, deterministic
    cuda_bf16   what upstream runs on an Ampere+ card (bf16, SDPA)
    cuda_f32    true float32 on the GPU, TF32 off: the Pascal compute recipe
    cuda_f16    upstream's own pre-Ampere path (fp16 + eager attention),
                forced on a newer card by reporting no bf16 support

GPU traces start from the cpu_f32 trace's noise (record that one first), so
every trace of a case is directly comparable; each GPU trace's JSON carries
its drift against cpu_f32.

Usage (inside the ACE-Step venv, with checkpoints already on disk):

    python script/acestep-reference.py --src ~/ACE-Step-1.5 --trace cpu_f32 \
        --out crates/neuron/src/harness/testdata/acestep
    python script/acestep-reference.py --src ... --trace cuda_bf16 --out ...
    python script/acestep-reference.py --index --out ...   # rebuild index.json

The pipeline downloads missing checkpoints by itself, so fetch them through
whichever Hugging Face-compatible registry you use before running this; the
script refuses to start if a component is missing.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

CASES = [
    {
        "name": "instr_10s_noplan",
        "caption": "warm lo-fi hip hop beat, dusty vinyl drums, mellow rhodes chords, relaxed",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "language": "unknown",
        "duration": 10.0,
        "planner": False,
        "seed": 1234,
    },
    {
        "name": "vocal_en_30s_noplan",
        "caption": "upbeat indie pop, bright electric guitars, punchy drums, female vocals",
        "lyrics": (
            "[Verse]\n"
            "Streetlights hum a quiet tune\n"
            "We chase the shadows of the moon\n"
            "[Chorus]\n"
            "Hold on, hold on, the night is young\n"
            "Every word we never sung\n"
        ),
        "instrumental": False,
        "language": "en",
        "duration": 30.0,
        "planner": False,
        "seed": 4242,
    },
    {
        "name": "vocal_en_30s_plan",
        "caption": "upbeat indie pop, bright electric guitars, punchy drums, female vocals",
        "lyrics": (
            "[Verse]\n"
            "Streetlights hum a quiet tune\n"
            "We chase the shadows of the moon\n"
            "[Chorus]\n"
            "Hold on, hold on, the night is young\n"
            "Every word we never sung\n"
        ),
        "instrumental": False,
        "language": "en",
        "duration": 30.0,
        "planner": True,
        "seed": 4242,
    },
    {
        "name": "vocal_es_20s_noplan",
        "caption": "latin ballad, nylon string guitar, soft percussion, male vocals",
        "lyrics": (
            "[Verse]\n"
            "Bajo la luna del mar\n"
            "Te vuelvo a recordar\n"
            "[Chorus]\n"
            "Canta conmigo esta noche\n"
        ),
        "instrumental": False,
        "language": "es",
        "duration": 20.0,
        "planner": False,
        "seed": 777,
    },
    {
        "name": "vocal_en_120s_noplan",
        "caption": "anthemic rock, driving drums, distorted guitars, male vocals, stadium",
        "lyrics": (
            "[Verse]\n"
            "Engines roar across the plain\n"
            "Thunder rolling in the rain\n"
            "[Chorus]\n"
            "We are the fire, we are the flame\n"
            "Nothing will ever be the same\n"
            "[Verse]\n"
            "Iron hearts and open roads\n"
            "Carry all our heavy loads\n"
            "[Chorus]\n"
            "We are the fire, we are the flame\n"
            "Nothing will ever be the same\n"
        ),
        "instrumental": False,
        "language": "en",
        "duration": 120.0,
        "planner": False,
        "seed": 99,
    },
]

# Which traces record which cases. GPU traces skip what the CPU reference
# already pins down, to keep the committed fixtures small.
TRACE_CASES = {
    "cpu_f32": [c["name"] for c in CASES],
    "cuda_bf16": [c["name"] for c in CASES],
    "cuda_f32": ["instr_10s_noplan", "vocal_en_30s_noplan", "vocal_es_20s_noplan", "vocal_en_120s_noplan"],
    "cuda_f16": ["instr_10s_noplan", "vocal_en_30s_noplan", "vocal_es_20s_noplan", "vocal_en_120s_noplan"],
}

# What each trace keeps, to hold the committed fixtures to a few tens of MB.
# The CPU reference keeps every sampler step for cases up to ALL_STEPS_MAX_FRAMES
# 25 Hz frames, the first, middle and last step up to LONG_FRAMES, and the first
# and last beyond that. GPU traces keep only what an envelope check needs: the
# noise, the first and last velocity, the sampler output and the waveform.
# Every dropped tensor still has its stats (shape, NaN/Inf counts, absmax,
# mean, std) in the JSON. step0_xt is never saved: it is `noise`.
ALL_STEPS_MAX_FRAMES = 250
LONG_FRAMES = 750
# Waveform kept per trace: the first WAVE_KEEP_S seconds.
WAVE_KEEP_S = 0.5

DIT_CONFIG = "acestep-v15-turbo"
LM_MODEL = "acestep-5Hz-lm-1.7B"


def to_saved(t, keep_dtype):
    t = t.detach().to("cpu")
    if t.is_floating_point() and not keep_dtype:
        t = t.float()
    return t.contiguous().clone()


def stats(t):
    f = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "nan": int(torch.isnan(f).sum()),
        "inf": int(torch.isinf(f).sum()),
        "absmax": float(f.nan_to_num().abs().max()) if f.numel() else 0.0,
        "mean": float(f.nan_to_num().mean()) if f.numel() else 0.0,
        "std": float(f.nan_to_num().std()) if f.numel() > 1 else 0.0,
    }


class Recorder:
    """Collects tensors and metadata from the hooks for one case."""

    def __init__(self, keep_dtype):
        self.keep_dtype = keep_dtype
        self.reset()

    def reset(self):
        self.tensors = {}
        self.meta = {"stats": {}, "planner": [], "decoder_calls": 0}
        self.step_inputs = []
        self.step_outputs = []
        self.step_timesteps = []
        self.decode_depth = 0

    def put(self, name, t, save=True):
        if t is None:
            return
        self.meta["stats"][name] = stats(t)
        if save:
            self.tensors[name] = to_saved(t, self.keep_dtype)


def install_hooks(dit, llm, rec):
    model = dit.model

    orig_text = dit.infer_text_embeddings

    def infer_text_embeddings(ids):
        out = orig_text(ids)
        if "text_token_ids" not in rec.tensors:
            rec.put("text_token_ids", ids)
            rec.put("text_hidden_states", out)
            rec.meta["text_prompt"] = dit.text_tokenizer.decode(ids[0])
        return out

    dit.infer_text_embeddings = infer_text_embeddings

    orig_lyric = dit.infer_lyric_embeddings

    def infer_lyric_embeddings(ids):
        out = orig_lyric(ids)
        rec.put("lyric_token_ids", ids)
        rec.put("lyric_embeds", out)
        rec.meta["lyric_prompt"] = dit.text_tokenizer.decode(ids[0])
        return out

    dit.infer_lyric_embeddings = infer_lyric_embeddings

    def lyric_hook(_m, _args, out):
        rec.put("lyric_encoder_out", out.last_hidden_state)

    model.encoder.lyric_encoder.register_forward_hook(lyric_hook)

    def encoder_hook(_m, _args, out):
        rec.put("encoder_hidden_states", out[0])
        rec.put("encoder_attention_mask", out[1])

    model.encoder.register_forward_hook(encoder_hook)

    orig_noise = model.prepare_noise

    def prepare_noise(context_latents, seed=None):
        noise = orig_noise(context_latents, seed)
        if rec.forced_noise is not None and rec.forced_noise.shape == noise.shape:
            # GPU traces start from the CPU reference's noise, so every trace
            # of a case is directly comparable against cpu_f32.
            noise = rec.forced_noise.to(device=noise.device, dtype=noise.dtype)
            rec.meta["noise_source"] = "cpu_f32"
        rec.put("noise", noise)
        rec.meta["noise_seed"] = seed if not isinstance(seed, torch.Tensor) else seed.tolist()
        return noise

    model.prepare_noise = prepare_noise

    def decoder_hook(_m, _args, kwargs, out):
        rec.meta["decoder_calls"] += 1
        rec.step_inputs.append(kwargs["hidden_states"].detach().to("cpu"))
        rec.step_outputs.append(out[0].detach().to("cpu"))
        rec.step_timesteps.append(float(kwargs["timestep"].flatten()[0]))
        if rec.meta["decoder_calls"] == 1:
            rec.put("decoder_attention_mask", kwargs["attention_mask"])
            rec.put("context_latents", kwargs["context_latents"])

    model.decoder.register_forward_hook(decoder_hook, with_kwargs=True)

    orig_gen = model.generate_audio

    def generate_audio(**kwargs):
        scalars = {}
        for k, v in kwargs.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                scalars[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
                scalars[k] = list(v)
        rec.meta["generate_audio_kwargs"] = scalars
        for k in ("src_latents", "chunk_masks", "is_covers", "precomputed_lm_hints_25Hz",
                  "attention_mask", "text_attention_mask", "lyric_attention_mask",
                  "refer_audio_acoustic_hidden_states_packed", "refer_audio_order_mask"):
            v = kwargs.get(k)
            if isinstance(v, torch.Tensor):
                rec.put(f"gen_{k}", v, save=k in ("chunk_masks", "is_covers", "precomputed_lm_hints_25Hz"))
        out = orig_gen(**kwargs)
        rec.put("target_latents", out["target_latents"])
        return out

    model.generate_audio = generate_audio

    orig_tiled = dit.tiled_decode

    def tiled_decode(z, *a, **kw):
        rec.decode_depth += 1
        try:
            rec.put("vae_input", z)
            rec.meta["vae_path"] = "tiled_decode"
            wav = orig_tiled(z, *a, **kw)
        finally:
            rec.decode_depth -= 1
        rec.put("vae_output", wav, save=False)
        rec.raw_wave = wav.detach().float().cpu()
        return wav

    dit.tiled_decode = tiled_decode

    orig_decode = dit.vae.decode

    def vae_decode(z, *a, **kw):
        out = orig_decode(z, *a, **kw)
        if rec.decode_depth == 0:
            rec.put("vae_input", z)
            rec.meta["vae_path"] = "vae.decode"
            rec.put("vae_output", out.sample, save=False)
            rec.raw_wave = out.sample.detach().float().cpu()
        return out

    dit.vae.decode = vae_decode

    if llm is None:
        return

    orig_single = llm._run_pt_single

    def run_pt_single(formatted_prompt, temperature, cfg_scale, *a, **kw):
        entry = {
            "phase": kw.get("generation_phase"),
            "formatted_prompt": formatted_prompt,
            "temperature": temperature,
            "cfg_scale": cfg_scale,
            "negative_prompt": kw.get("negative_prompt", a[0] if a else None),
            "target_duration": kw.get("target_duration"),
        }
        rec.meta["planner"].append(entry)
        text = orig_single(formatted_prompt, temperature, cfg_scale, *a, **kw)
        entry["output_text"] = text
        return text

    llm._run_pt_single = run_pt_single

    for name in ("_generate_with_cfg_custom", "_generate_with_constrained_decoding"):
        orig = getattr(llm, name)

        def wrapped(*a, _orig=orig, _name=name, **kw):
            ids = kw.get("batch_input_ids", kw.get("input_ids"))
            out = _orig(*a, **kw)
            idx = len(rec.meta["planner"]) - 1
            rec.put(f"planner{idx}_input_ids", ids)
            seq = out if isinstance(out, torch.Tensor) else out[0]
            rec.put(f"planner{idx}_output_ids", seq[..., ids.shape[-1]:])
            rec.meta["planner"][idx]["loop"] = _name
            return out

        setattr(llm, name, wrapped)


def load(args, trace, need_lm):
    sys.path.insert(0, str(args.src))
    from acestep import gpu_config
    from acestep.core.generation.handler import init_service_orchestrator, init_service_loader
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    device = "cpu" if trace == "cpu_f32" else "cuda"
    if trace == "cuda_f16":
        # Upstream's own pre-Ampere path: fp16 weights and eager attention.
        no_bf16 = lambda *a, **k: False  # noqa: E731
        gpu_config.cuda_supports_bfloat16 = no_bf16
        init_service_orchestrator.gpu_config.cuda_supports_bfloat16 = no_bf16
        init_service_loader.gpu_config.cuda_supports_bfloat16 = no_bf16

    dit = AceStepHandler()
    # cuda_f32 loads on the CPU (where upstream builds everything in float32)
    # and then moves to the GPU. Loading on CUDA and casting afterwards is not
    # the same thing: the CUDA load rounds the RoPE `inv_freq` buffers and the
    # float32 silence latent to bf16, and a later cast cannot restore them.
    load_device = "cpu" if trace == "cuda_f32" else device
    msg, ok = dit.initialize_service(
        project_root=str(args.src), config_path=DIT_CONFIG, device=load_device, use_flash_attention=False
    )
    if not ok:
        raise SystemExit(f"DiT init failed: {msg}")
    if trace == "cuda_f32":
        for m in (dit.model, dit.vae, dit.text_encoder):
            m.to(device)
        dit.silence_latent = dit.silence_latent.to(device)
        dit.device = device
        # Pascal has no TF32; keep every matmul and convolution in true f32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    llm = None
    if need_lm:
        llm = LLMHandler()
        lm_dtype = torch.float32 if trace in ("cpu_f32", "cuda_f32") else None
        msg, ok = llm.initialize(
            checkpoint_dir=str(args.src / "checkpoints"), lm_model_path=LM_MODEL,
            backend="pt", device=device, dtype=lm_dtype,
        )
        if not ok:
            raise SystemExit(f"LM init failed: {msg}")
    return dit, llm


def run_case(case, trace, dit, llm, rec, out_dir):
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from safetensors.torch import save_file

    from safetensors.torch import load_file

    rec.reset()
    rec.raw_wave = None
    rec.forced_noise = None
    ref = None
    ref_path = out_dir / f"{case['name']}__cpu_f32.safetensors"
    if trace != "cpu_f32" and ref_path.exists():
        ref = load_file(str(ref_path))
        rec.forced_noise = ref["noise"]
    plan = case["planner"]
    params = GenerationParams(
        caption=case["caption"],
        lyrics=case["lyrics"],
        instrumental=case["instrumental"],
        vocal_language=case["language"],
        duration=case["duration"],
        inference_steps=8,
        shift=3.0,
        seed=case["seed"],
        thinking=plan,
        use_cot_metas=plan,
        use_cot_caption=plan,
        use_cot_language=plan,
        lm_temperature=0.0,  # greedy: the planner's output is a fixture, not a sample
        lm_top_p=1.0,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[case["seed"]], audio_format="flac")
    random.seed(case["seed"])
    np.random.seed(case["seed"])
    torch.manual_seed(case["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(case["seed"])
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = generate_music(dit, llm if plan else None, params, config, save_dir=None)
    wall = time.time() - t0
    if not result.success:
        # Upstream refuses to decode NaN latents. That failure is itself the
        # finding (the fp16 pre-Ampere trace), so record everything up to it.
        if "noise" not in rec.tensors:
            raise RuntimeError(f"{case['name']}/{trace}: {result.error}")
        rec.meta["error"] = result.error

    frames = rec.tensors["noise"].shape[1]
    n = len(rec.step_inputs)
    gpu_trace = trace != "cpu_f32"
    if gpu_trace:
        keep_vt = set() if frames > LONG_FRAMES else {0, n - 1}
        keep_xt = set()
    elif frames <= ALL_STEPS_MAX_FRAMES:
        keep_vt, keep_xt = set(range(n)), set(range(1, n))
    elif frames <= LONG_FRAMES:
        keep_vt, keep_xt = {0, n // 2, n - 1}, {n // 2, n - 1}
    else:
        keep_vt, keep_xt = {0}, set()
    # step0_xt is never kept: it is the noise.
    # Where does a NaN first appear? Keep that step's tensors whatever the policy.
    first_nan = None
    for name in ("text_hidden_states", "lyric_embeds", "lyric_encoder_out", "encoder_hidden_states",
                 "context_latents", "noise"):
        if rec.meta["stats"].get(name, {}).get("nan"):
            first_nan = name
            break
    for i in range(n):
        if first_nan:
            break
        if torch.isnan(rec.step_inputs[i]).any():
            first_nan = f"step{i}_xt"
            keep_xt.add(i)
            keep_vt.add(i - 1)
        elif torch.isnan(rec.step_outputs[i]).any():
            first_nan = f"step{i}_vt"
            keep_vt.add(i)
            if i:
                keep_xt.add(i)
    rec.meta["first_nan"] = first_nan

    # Drift against the CPU f32 reference, from the same noise. A planner
    # case's LM can take a different greedy path in reduced precision, after
    # which its conditioning differs and the comparison measures that instead.
    if ref is not None:
        drift = {}
        pairs = [("target_latents", rec.tensors.get("target_latents"))]
        pairs += [(f"step{i}_vt", rec.step_outputs[i]) for i in range(n)]
        for name, t in pairs:
            if t is None or name not in ref:
                continue
            a, b = t.float(), ref[name].float()
            if a.shape != b.shape:
                continue
            diff = (a - b).nan_to_num(nan=float("inf"))
            drift[name] = {
                "max_abs": float(diff.abs().max()),
                "rel_l2": float(diff.norm() / b.norm().clamp(min=1e-12)),
                "cos": float(torch.nn.functional.cosine_similarity(
                    a.flatten().nan_to_num(), b.flatten(), dim=0)),
            }
        rec.meta["drift_vs_cpu_f32"] = drift
    for i in range(n):
        rec.put(f"step{i}_xt", rec.step_inputs[i], save=i in keep_xt)
        rec.put(f"step{i}_vt", rec.step_outputs[i], save=i in keep_vt)
    rec.meta["timesteps"] = rec.step_timesteps
    rec.meta["steps_saved"] = {"xt": sorted(keep_xt), "vt": sorted(keep_vt)}

    # GPU traces drop what the CPU reference already carries in full.
    if gpu_trace:
        for k in ("text_hidden_states", "lyric_embeds", "lyric_encoder_out", "encoder_hidden_states",
                  "encoder_attention_mask", "context_latents", "decoder_attention_mask",
                  "gen_chunk_masks", "vae_input", "text_token_ids", "lyric_token_ids"):
            rec.tensors.pop(k, None)
    # The condition encoder packs lyrics first; with an all-true mask the
    # lyric encoder's output is exactly the head of encoder_hidden_states.
    lyr, enc = rec.tensors.get("lyric_encoder_out"), rec.tensors.get("encoder_hidden_states")
    if lyr is not None and enc is not None and torch.equal(enc[:, : lyr.shape[1]], lyr):
        rec.tensors.pop("lyric_encoder_out")
        rec.meta["lyric_encoder_out_is_encoder_prefix"] = True
    if frames > LONG_FRAMES:
        rec.tensors.pop("context_latents", None)
        rec.tensors.pop("vae_input", None)  # equals target_latents (transposed)

    if result.success:
        audio = result.audios[0]
    else:
        audio = {"sample_rate": 48000, "tensor": torch.zeros(2, 1), "params": {}}
    sr = audio["sample_rate"]
    final = audio["tensor"].float()
    rec.meta["stats"]["final_audio_full"] = stats(final)
    if rec.raw_wave is not None:
        rec.put("vae_waveform", rec.raw_wave[..., : int(WAVE_KEEP_S * sr)])
        # The pipeline peak-normalises the VAE output to -1 dBFS; record the
        # gain so final_audio = vae_waveform * gain without storing both.
        rec.meta["normalisation_gain"] = float(final.abs().max() / rec.raw_wave.abs().max())

    p = audio.get("params", {})
    rec.meta.update({
        "case": case,
        "trace": trace,
        "sample_rate": sr,
        "latent_frames": frames,
        "wall_s": round(wall, 2),
        "planner_metadata": {k: p.get(k) for k in p if k.startswith("cot_")},
        "nan_anywhere": not result.success or any(s["nan"] or s["inf"] for s in rec.meta["stats"].values()),
        "dtype": str(dit.dtype).replace("torch.", ""),
        "time_costs": (result.extra_outputs or {}).get("time_costs"),
    })
    if torch.cuda.is_available() and gpu_trace:
        rec.meta["vram_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    codes = (result.extra_outputs or {}).get("lm_generated_audio_codes") or p.get("audio_codes")
    if plan and isinstance(codes, str):
        import re
        ids = [int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", codes)]
        rec.put("planner_audio_codes", torch.tensor(ids, dtype=torch.int64))

    # A GPU trace that started from the CPU reference's noise does not need its
    # own copy; reduced-precision traces keep stats, not the waveform.
    if rec.meta.get("noise_source") == "cpu_f32":
        rec.tensors.pop("noise", None)
    if trace in ("cuda_bf16", "cuda_f16"):
        rec.tensors.pop("vae_waveform", None)

    stem = f"{case['name']}__{trace}"
    save_file({k: v for k, v in rec.tensors.items()}, str(out_dir / f"{stem}.safetensors"))
    (out_dir / f"{stem}.json").write_text(json.dumps(rec.meta, indent=1, default=str) + "\n")
    print(f"{stem}: {wall:.1f}s frames={frames} nan={rec.meta['nan_anywhere']}", flush=True)


def build_index(out_dir):
    index = {"traces": {}}
    for f in sorted(out_dir.glob("*__*.json")):
        meta = json.loads(f.read_text())
        st = out_dir / f.name.replace(".json", ".safetensors")
        index["traces"][f.stem] = {
            "case": meta["case"]["name"],
            "trace": meta["trace"],
            "dtype": meta["dtype"],
            "latent_frames": meta["latent_frames"],
            "nan_anywhere": meta["nan_anywhere"],
            "first_nan": meta.get("first_nan"),
            "target_latents_rel_l2_vs_cpu_f32": (
                meta.get("drift_vs_cpu_f32", {}).get("target_latents", {}).get("rel_l2")),
            "error": meta.get("error"),
            "wall_s": meta["wall_s"],
            "vram_peak_gib": meta.get("vram_peak_gib"),
            "bytes": st.stat().st_size if st.exists() else None,
        }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path.home() / "ACE-Step-1.5")
    ap.add_argument("--trace", choices=sorted(TRACE_CASES))
    ap.add_argument("--cases", nargs="*", help="subset of case names (default: all for the trace)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", action="store_true", help="only rebuild index.json")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.index:
        build_index(args.out)
        return
    if not args.trace:
        ap.error("--trace is required unless --index")
    for component in (DIT_CONFIG, LM_MODEL, "vae", "Qwen3-Embedding-0.6B"):
        path = args.src / "checkpoints" / component
        if not path.is_dir() or not any(path.glob("*.safetensors")):
            raise SystemExit(f"missing checkpoint {path}; fetch it before recording")

    names = args.cases or TRACE_CASES[args.trace]
    cases = [c for c in CASES if c["name"] in names]
    need_lm = any(c["planner"] for c in cases)
    dit, llm = load(args, args.trace, need_lm)
    # bf16/fp16 traces are stored in their own dtype: exact, and half the size.
    rec = Recorder(keep_dtype=args.trace in ("cuda_bf16", "cuda_f16"))
    install_hooks(dit, llm, rec)
    for case in cases:
        run_case(case, args.trace, dit, llm, rec, args.out)
    build_index(args.out)


if __name__ == "__main__":
    main()
