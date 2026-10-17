#!/usr/bin/env python3
"""Record HeartMuLa reference traces for neuron's parity tests (#390).

Runs heartlib's own model code (`HeartMuLaGenPipeline.preprocess` and
`_forward`, `HeartCodec.detokenize`) over a fixed set of cases, hooking each
boundary a port has to reproduce. Per case and per trace it writes:

  - the tags and lyric prompts as tokenized, with token ids, and the
    9-stream prompt token matrix the LM actually consumes
  - LM: the backbone's last hidden state and the guided (post-CFG) logits of
    every codebook for the first few frames, and the full sampled frame
    stream `[8, frames]`, for greedy decoding (top-k 1) and for the default
    sampler (cfg 1.5, temperature 1.0, top-k 50) from a fixed seed
  - codec (CPU f32 only): the codes as given, the RVQ embedding after
    `cond_feature_emb` and 2x upsampling (the flow-matching condition), the
    initial noise, the flow state after step 5, the final latents, and the
    decoded waveform, truncated

Reference revisions (what the committed fixtures were recorded against):

    code     https://github.com/HeartMuLa/heartlib @ 5858ca8f1ffe58d62be7c007ea55f1033c5ca456
    weights  HeartMuLa/HeartMuLa-oss-3B-happy-new-year @ 41f6fc68490e11dc43fdabaa6b5767946408c903
             HeartMuLa/HeartCodec-oss-20260123 @ f889dab0532cfa4bf459f2a3367eb6d346b8eeda
             HeartMuLa/HeartMuLaGen (tokenizer.json, gen_config.json)
    torch 2.10.0+cu128, transformers 4.57.0, torchtune 0.4.0

Traces (`--trace`):

    cpu_f32     float32 on CPU: the parity reference; LM and codec
    cuda_bf16   what upstream runs: bf16 weights, bf16 autocast; LM only
    cuda_f32    true float32 on the GPU, TF32 off; LM only

The codec is float32 in every upstream configuration (heartlib warns that a
bf16 codec degrades audio), so only the CPU trace records it; a port checks
its codec against `cpu_f32` from the recorded codes.

The MuQ-MuLan reference-audio encoder is never loaded: heartlib builds the
pipeline with `muq_mulan=None` and conditions on a zero embedding.

Usage (inside heartlib's venv, checkpoints already on disk):

    python script/heartmula-reference.py --ckpt ~/heart/ckpt --trace cpu_f32 \
        --out crates/neuron/src/harness/testdata/heartmula
    python script/heartmula-reference.py --ckpt ... --trace cuda_bf16 --out ...
    python script/heartmula-reference.py --compare --out ... # GPU vs CPU, index.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

INSTRUMENTAL = "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Bridge]\n\n[Chorus]\n\n[Outro]"

CASES = [
    {
        "name": "instr_12s",
        "tags": "lo-fi hip hop, mellow, piano, vinyl crackle, relaxed, instrumental",
        "lyrics": INSTRUMENTAL,
        "duration": 12.0,
        "seed": 1234,
        "codec": True,
    },
    {
        "name": "vocal_en_12s",
        "tags": "pop, female vocals, acoustic guitar, upbeat, bright",
        "lyrics": (
            "[Verse]\n"
            "Streetlights hum a quiet tune\n"
            "We chase the shadows of the moon\n"
            "[Chorus]\n"
            "Hold on, hold on, the night is young\n"
        ),
        "duration": 12.0,
        "seed": 4242,
        "codec": True,
    },
    {
        "name": "vocal_es_12s",
        "tags": "latin ballad, male vocals, nylon string guitar, soft percussion",
        "lyrics": (
            "[Verse]\n"
            "Bajo la luna del mar\n"
            "Te vuelvo a recordar\n"
            "[Chorus]\n"
            "Canta conmigo esta noche\n"
        ),
        "duration": 12.0,
        "seed": 777,
        "codec": False,
    },
]

# Decoding setups recorded for the LM. Both use heartlib's default CFG.
SAMPLERS = {
    "greedy": {"cfg_scale": 1.5, "temperature": 1.0, "topk": 1},
    "sampled": {"cfg_scale": 1.5, "temperature": 1.0, "topk": 50},
}
TRACE_SAMPLERS = {
    "cpu_f32": ["greedy", "sampled"],
    "cuda_bf16": ["greedy", "sampled"],
    "cuda_f32": ["greedy"],
}
LOGIT_FRAMES = 2  # frames whose per-codebook logits are recorded
WAVE_SECONDS = 3.0  # waveform kept, from the start
FLOW_STEP = 5  # flow-matching step whose state is recorded


def stats(t: torch.Tensor) -> dict:
    f = t.detach().float()
    finite = torch.isfinite(f)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "nan": int(torch.isnan(f).sum()),
        "inf": int(torch.isinf(f).sum()),
        "absmax": float(f[finite].abs().max()) if finite.any() else None,
        "mean": float(f[finite].mean()) if finite.any() else None,
    }


def load(ckpt: Path, trace: str):
    from heartlib import HeartMuLaGenPipeline
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec
    from heartlib.heartmula.modeling_heartmula import HeartMuLa

    if trace == "cpu_f32":
        device, dtype = torch.device("cpu"), torch.float32
    elif trace == "cuda_bf16":
        device, dtype = torch.device("cuda"), torch.bfloat16
    else:
        device, dtype = torch.device("cuda"), torch.float32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    pipe = HeartMuLaGenPipeline.from_pretrained(
        str(ckpt),
        device={"mula": device, "codec": torch.device("cpu")},
        dtype={"mula": dtype, "codec": torch.float32},
        version="3B",
        lazy_load=True,
    )
    # `from_pretrained` leaves the LM in the checkpoint's fp32 whatever dtype
    # it is given; cast explicitly. No torch.compile: eager is the reference.
    mula = HeartMuLa.from_pretrained(pipe.mula_path, device_map="cpu", dtype=dtype)
    pipe._mula = mula.to(dtype).to(device).eval()
    pipe._codec = None
    if trace == "cpu_f32":
        pipe._codec = HeartCodec.from_pretrained(
            pipe.codec_path, device_map="cpu", dtype=torch.float32
        ).eval()
    pipe.lazy_load = False
    return pipe


def record_lm(pipe, case: dict, sampler: str, out: dict, meta: dict):
    import heartlib.heartmula.modeling_heartmula as hm

    cfg = SAMPLERS[sampler]
    calls = {"n": 0}
    logits, hidden = [], []
    orig_sample = hm.sample_topk

    def sample_topk(lg, topk, temperature):
        frame = calls["n"] // 8
        if frame < LOGIT_FRAMES:
            logits.append(lg.detach().float().cpu())
        calls["n"] += 1
        return orig_sample(lg, topk, temperature)

    def pre_hook(_module, args):
        if len(hidden) < LOGIT_FRAMES:
            hidden.append(args[0].detach().float().cpu())

    hm.sample_topk = sample_topk
    handle = pipe._mula.codebook0_head.register_forward_pre_hook(pre_hook)
    try:
        torch.manual_seed(case["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(case["seed"])
        t0 = time.time()
        with torch.no_grad():
            inputs = pipe.preprocess(
                {"tags": case["tags"], "lyrics": case["lyrics"]}, cfg_scale=cfg["cfg_scale"]
            )
            frames = pipe._forward(
                inputs,
                max_audio_length_ms=int(case["duration"] * 1000),
                temperature=cfg["temperature"],
                topk=cfg["topk"],
                cfg_scale=cfg["cfg_scale"],
            )["frames"]
        wall = time.time() - t0
    finally:
        hm.sample_topk = orig_sample
        handle.remove()

    p = f"lm_{sampler}_"
    out[p + "frames"] = frames.detach().cpu().to(torch.int64)
    # [frame, codebook, vocab] — the guided logits each sample was drawn from.
    out[p + "logits"] = torch.stack(
        [torch.stack(logits[f * 8 : f * 8 + 8]).squeeze(1) for f in range(LOGIT_FRAMES)]
    )
    # [frame, batch, dim] — the backbone's last hidden state, both CFG rows.
    out[p + "backbone_last_h"] = torch.stack(hidden)
    meta["lm"][sampler] = {
        **cfg,
        "seed": case["seed"],
        "frames": int(frames.shape[-1]),
        "stopped_early": int(frames.shape[-1]) < int(case["duration"] * 1000) // 80 + 1,
        "wall_s": round(wall, 2),
        "frames_stats": stats(frames),
        "logits_stats": stats(out[p + "logits"]),
    }
    return inputs, frames


def record_codec(pipe, frames: torch.Tensor, out: dict, meta: dict, seed: int):
    import heartlib.heartcodec.models.flow_matching as fm

    codec = pipe._codec
    rec = {}
    orig_solve = fm.FlowMatching.solve_euler
    orig_cond = codec.decoder.flow_matching.cond_feature_emb.forward

    def cond_forward(x):
        y = orig_cond(x)
        if "rvq_cond" not in rec:
            rec["rvq_quantized"] = x.detach().clone()
            rec["rvq_cond"] = y.detach().clone()
        return y

    def solve_euler(self, x, incontext_x, incontext_length, t_span, mu, guidance_scale):
        # heartlib's solver, verbatim, with the states recorded.
        if "flow_noise" not in rec:
            rec["flow_noise"] = x.detach().clone()
            rec["flow_mu"] = mu.detach().clone()
            rec["flow_t_span"] = t_span.detach().clone()
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        noise = x.clone()
        sol = []
        for step in range(1, len(t_span)):
            x[:, 0:incontext_length, :] = (1 - (1 - 1e-6) * t) * noise[
                :, 0:incontext_length, :
            ] + t * incontext_x[:, 0:incontext_length, :]
            if guidance_scale > 1.0:
                dphi_dt = self.estimator(
                    torch.cat(
                        [
                            torch.cat([x, x], 0),
                            torch.cat([incontext_x, incontext_x], 0),
                            torch.cat([torch.zeros_like(mu), mu], 0),
                        ],
                        2,
                    ),
                    timestep=t.unsqueeze(-1).repeat(2),
                )
                dphi_dt_uncond, dhpi_dt_cond = dphi_dt.chunk(2, 0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (dhpi_dt_cond - dphi_dt_uncond)
            else:
                dphi_dt = self.estimator(torch.cat([x, incontext_x, mu], 2), timestep=t.unsqueeze(-1))
            if step == 1 and "flow_step1_v" not in rec:
                rec["flow_step1_v"] = dphi_dt.detach().clone()
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step == FLOW_STEP and "flow_step5_x" not in rec:
                rec["flow_step5_x"] = x.detach().clone()
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        if "flow_final" not in rec:
            rec["flow_final"] = sol[-1].detach().clone()
        return sol[-1]

    fm.FlowMatching.solve_euler = solve_euler
    codec.decoder.flow_matching.cond_feature_emb.forward = cond_forward
    try:
        torch.manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            wav = codec.detokenize(frames.to(torch.long))
        wall = time.time() - t0
    finally:
        fm.FlowMatching.solve_euler = orig_solve
        codec.decoder.flow_matching.cond_feature_emb.forward = orig_cond

    keep = int(WAVE_SECONDS * 48000)
    out["codec_codes"] = frames.detach().cpu().to(torch.int64)
    for k, v in rec.items():
        out["codec_" + k] = v.detach().cpu().float()
    out["codec_waveform"] = wav[:, :keep].detach().cpu().float()
    meta["codec"] = {
        "seed": seed,
        "detokenize": {"duration": 29.76, "num_steps": 10, "guidance_scale": 1.25},
        "segments": 1,
        "codes_frames": int(frames.shape[-1]),
        "waveform_samples": int(wav.shape[-1]),
        "waveform_kept_s": WAVE_SECONDS,
        "wall_s": round(wall, 2),
        "stats": {k: stats(out["codec_" + k]) for k in rec},
        "waveform_stats": stats(wav),
    }


def run(args):
    from safetensors.torch import save_file

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = load(Path(args.ckpt).expanduser(), args.trace)
    ref = {}
    for case in CASES:
        if args.cases and case["name"] not in args.cases:
            continue
        out, meta = {}, {"case": case, "trace": args.trace, "lm": {}}
        # Prompt ids, exactly as preprocess builds them.
        tok = pipe.text_tokenizer
        tags = case["tags"].lower()
        tags = f"<tag>{tags}</tag>"
        meta["tags_prompt"] = tags
        meta["lyrics_prompt"] = case["lyrics"].lower()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for sampler in TRACE_SAMPLERS[args.trace]:
            inputs, frames = record_lm(pipe, case, sampler, out, meta)
            if "prompt_tokens" not in out:
                tokens = inputs["tokens"][0]
                out["prompt_tokens"] = tokens.to(torch.int64)
                out["prompt_tokens_mask"] = inputs["tokens_mask"][0].to(torch.uint8)
                meta["muq_idx"] = int(inputs["muq_idx"][0])
                meta["prompt_len"] = int(tokens.shape[0])
                text = tokens[:, -1].tolist()
                meta["tags_ids"] = text[: meta["muq_idx"]]
                meta["lyrics_ids"] = text[meta["muq_idx"] + 1 :]
            if sampler == "greedy" and args.trace == "cpu_f32" and case["codec"]:
                record_codec(pipe, frames, out, meta, case["seed"])
        if torch.cuda.is_available() and args.trace != "cpu_f32":
            meta["vram_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        meta["nan_anywhere"] = any(
            torch.isnan(v.float()).any().item() for v in out.values() if v.is_floating_point()
        )
        save_file(
            {k: v.contiguous() for k, v in out.items()},
            str(out_dir / f"{case['name']}__{args.trace}.safetensors"),
        )
        (out_dir / f"{case['name']}__{args.trace}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(case["name"], args.trace, json.dumps(meta["lm"]), flush=True)


def compare(args):
    """Stamp every GPU trace's JSON with its agreement with `cpu_f32`."""
    from safetensors.torch import load_file

    out_dir = Path(args.out)
    for f in sorted(out_dir.glob("*__cuda_*.json")):
        case, _trace = f.stem.split("__")
        cpu = out_dir / f"{case}__cpu_f32.safetensors"
        if not cpu.exists():
            continue
        meta = json.loads(f.read_text())
        got = load_file(str(out_dir / f"{f.stem}.safetensors"))
        want = load_file(str(cpu))
        for sampler in meta["lm"]:
            a, b = got[f"lm_{sampler}_frames"], want[f"lm_{sampler}_frames"]
            n = min(a.shape[-1], b.shape[-1])
            diff = (a[:, :n] != b[:, :n]).any(0).nonzero()
            first = int(diff[0]) if len(diff) else None
            la, lb = got[f"lm_{sampler}_logits"], want[f"lm_{sampler}_logits"]
            ha, hb = got[f"lm_{sampler}_backbone_last_h"], want[f"lm_{sampler}_backbone_last_h"]
            rel = lambda x, y: float((x - y).norm() / y.norm())  # noqa: E731
            meta["lm"][sampler]["vs_cpu_f32"] = {
                "frames_equal": first is None and a.shape == b.shape,
                "first_divergent_frame": first,
                # Frame 0, codebook 0 precede any sampled token, so they are
                # the like-for-like numerical comparison; everything after
                # depends on tokens that may already differ.
                "frame0_cb0_logits_rel_l2": rel(la[0, 0], lb[0, 0]),
                "frame0_cb0_argmax_equal": bool(la[0, 0].argmax() == lb[0, 0].argmax()),
                "frame0_backbone_last_h_rel_l2": rel(ha[0], hb[0]),
                "logits_rel_l2": rel(la, lb),
                "backbone_last_h_rel_l2": rel(ha, hb),
            }
        f.write_text(json.dumps(meta, indent=2) + "\n")


def index(args):
    out_dir = Path(args.out)
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        if f.name == "index.json":
            continue
        m = json.loads(f.read_text())
        rows.append(
            {
                "file": f.stem,
                "case": m["case"]["name"],
                "trace": m["trace"],
                "lm": {
                    k: {kk: v.get(kk) for kk in ("frames", "stopped_early", "wall_s", "vs_cpu_f32")}
                    for k, v in m["lm"].items()
                },
                "codec": bool(m.get("codec")),
                "nan_anywhere": m.get("nan_anywhere"),
                "vram_peak_gib": m.get("vram_peak_gib"),
                "bytes": (out_dir / f"{f.stem}.safetensors").stat().st_size,
            }
        )
    (out_dir / "index.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/spike/heart/ckpt"))
    ap.add_argument("--trace", choices=list(TRACE_SAMPLERS), default="cpu_f32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(args)
        index(args)
    elif args.index:
        index(args)
    else:
        torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count() or 8)))
        run(args)
        index(args)


if __name__ == "__main__":
    main()
