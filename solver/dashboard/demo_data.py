"""Synthetic run journals for previewing the dashboard before the engine exists.

Scales to the full benchmark: if `problems/` is populated it simulates a run
over all fetched problems (real names, families derived from the taxonomy);
otherwise a small hand-picked set. A single-flight GPU, agents thinking in
parallel, improving convergence, rejects/dups, family chains (siblings seeded
from the exemplar's best), and rental windows with un-rented gaps.
Deterministic (seeded RNG).
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

from ..journal import Journal

_AGENTS = ["sonnet", "haiku", "opus"]

# family classifier — keyword → family (ordered; first match wins).
_FAMILY_RULES = [
    (("nvfp4",), "quant-nvfp4"),
    (("fp8",), "quant-fp8"),
    (("moe", "expert"), "moe"),
    (("mamba", "ssm", "selective_scan", "segsum"), "ssm"),
    (("hyena", "fft"), "fft-conv"),
    (("flash_attention", "flash_attn"), "flash-attn"),
    (("mla", "latent_attention"), "mla"),
    (("backward",), "backward"),
    (("attention", "sdpa", "gqa", "attn"), "attention"),
    (("rope", "rotary", "position_embedding"), "rope"),
    (("rms_norm", "rmsnorm", "rms"), "rmsnorm"),
    (("layer_norm", "layernorm", "group_norm", "groupnorm", "ada", "grn", "norm"), "norm"),
    (("swiglu", "geglu", "gelu", "silu", "mlp", "gated"), "mlp-act"),
    (("conv",), "conv"),
    (("gemm", "matmul", "projection", "linear", "lm_head"), "gemm"),
    (("rope", "embedding"), "embedding"),
]

_STRATEGIES = [
    ("triton fused single-pass", "triton",
     "import triton\nimport triton.language as tl\n\n@triton.jit\ndef _k(X, R, W, Y, H: tl.constexpr, BLOCK: tl.constexpr):\n    row = tl.program_id(0)\n    off = tl.arange(0, BLOCK)\n    x = tl.load(X + row * H + off).to(tl.float32)\n    r = tl.load(R + row * H + off).to(tl.float32)\n    x = x + r\n    var = tl.sum(x * x) / H\n    y = x * tl.rsqrt(var + 1e-5) * tl.load(W + off)\n    tl.store(Y + row * H + off, y)\n\ndef run(hidden_states, residual, weight, eps, output):\n    n = hidden_states.numel() // hidden_states.shape[-1]\n    _k[(n,)](hidden_states, residual, weight, output,\n             hidden_states.shape[-1], BLOCK=1024)\n"),
    ("vectorized 128-bit loads + fp32 accum", "triton",
     "# v2: 8-wide vector loads, fp32 accumulator, weight cached in smem\n# (same structure as v1; BLOCK tuned per H, num_warps=8)\n"),
    ("torch.compile fused baseline", "pytorch",
     "import torch\n\n@torch.compile(mode='max-autotune')\ndef _fused(h, r, w, eps):\n    x = (h + r).float()\n    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(h.dtype) * w\n\ndef run(hidden_states, residual, weight, eps, output):\n    output[:] = _fused(hidden_states, residual, weight, eps)\n"),
    ("two-pass smem reduction, per-shape BLOCK dispatch", "triton",
     "# v4: per-shape BLOCK dispatch table {131:128, 2048:1024, 8192:2048}\n"),
    ("cuda_cpp warp-shuffle reduction", "cuda_cpp",
     "#include <torch/extension.h>\n// warp-shuffle rowwise reduction; float4 loads; grid-stride\nvoid run(torch::Tensor h, torch::Tensor r, torch::Tensor w,\n         double eps, torch::Tensor out) { /* kernel launch */ }\n"),
    ("online-softmax tiling + TF32 dot", "triton",
     "# flash-style online softmax; tl.dot(allow_tf32=True); fp32 accum\n"),
    ("cutlass block-scaled NVFP4 GEMM (ex.72b)", "cutlass",
     "// CollectiveBuilder<Sm100, nv_float4_t<e2m1>, ...>; block-scaled tcgen05.mma\n"),
    ("cuDNN graph SDPA fusion", "cudnn_frontend",
     "# cudnn.pygraph SDPA fprop; TANH softcapping between BMM1 and softmax\n"),
]

_SEED_CODE = ("import torch\n\n# DPS wrapper delegating to the inlined reference (seed)\n"
              "def run(*args):\n    out = args[-1]\n    out[:] = _reference_run(*args[:-1])\n")

_HARDCODED = [
    (230, "021_rmsnorm_h128", "rmsnorm"),
    (231, "022_rmsnorm_h512", "rmsnorm"),
    (69, "069_rms_norm", "rmsnorm"),
    (67, "067_flash_attention_gqa_ultralong", "flash-attn"),
]


def _classify(name: str) -> str:
    n = name.lower()
    for kws, fam in _FAMILY_RULES:
        if any(k in n for k in kws):
            return fam
    return "other"


def discover(problems_dir: Path) -> list[tuple[int, str, str]]:
    out = []
    for d in sorted((p for p in Path(problems_dir).iterdir() if p.is_dir() and p.name.isdigit()),
                    key=lambda p: int(p.name)):
        try:
            name = json.loads((d / "definition.json").read_text())["name"]
        except Exception:
            continue
        out.append((int(d.name), name, _classify(name)))
    return out


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _solution(lang: str, code: str, name: str) -> dict:
    entry = "kernel.cu::run" if lang == "cuda_cpp" else (
        "kernel.cpp::run" if lang in ("cutlass", "cudnn", "cublas") else "kernel.py::run")
    path = entry.split("::")[0]
    return {"name": name, "definition": "demo", "author": "demo",
            "spec": {"languages": [lang], "target_hardware": ["B200"],
                     "entry_point": entry, "destination_passing_style": True},
            "sources": [{"path": path, "content": code}]}


def build_demo(runs_dir: Path, seed: int = 7, limit: int | None = None,
               problems_dir: Path = Path("problems")) -> Path:
    rng = random.Random(seed)
    runs_dir = Path(runs_dir)
    discovered = discover(problems_dir) if Path(problems_dir).exists() else _HARDCODED
    if not discovered:
        discovered = _HARDCODED
    if limit:
        discovered = discovered[:limit]

    # family chains: first-seen id per family is the exemplar; the rest seed higher
    seen_family: set[str] = set()
    probs = []
    for task, name, family in discovered:
        exemplar = family not in seen_family
        seen_family.add(family)
        agent = _AGENTS[task % len(_AGENTS)]
        # difficulty by family: attention/moe/quant harder (lower ceiling)
        hard = family in ("flash-attn", "attention", "mla", "moe", "quant-nvfp4",
                          "backward", "ssm", "fft-conv")
        ceil = rng.uniform(0.70, 0.86) if hard else rng.uniform(0.82, 0.96)
        start = (rng.uniform(0.30, 0.48) if exemplar
                 else min(ceil - 0.05, rng.uniform(0.55, 0.75)))  # sibling template seed
        evals = rng.randint(4, 10) if not exemplar else rng.randint(8, 20)
        running = rng.random() < 0.12          # ~12% still in flight
        if running:
            evals = max(2, evals // 2)
        probs.append({"task": task, "name": name, "family": family, "agent": agent,
                      "ceil": ceil, "start": start, "evals_left": evals,
                      "running": running, "exemplar": exemplar})

    base = dt.datetime.now(dt.timezone.utc).timestamp() - 3.5 * 3600
    gpu_free = base
    clocks: dict[int, float] = {}
    js: dict[int, Journal] = {}
    st: dict[int, dict] = {}

    # rental gaps: two un-rented windows during the span
    gaps = [(base + 3000, base + 3600), (base + 7200, base + 7800)]

    def run_gpu(task: int, gpu_s: float) -> tuple[float, float, float]:
        nonlocal gpu_free
        enq = clocks[task]
        start_t = max(enq, gpu_free) + rng.uniform(0.2, 1.0)
        for gs, ge in gaps:
            if start_t < ge and start_t + gpu_s > gs:
                start_t = ge
        done_t = start_t + gpu_s
        gpu_free = done_t
        return enq, start_t, done_t

    # stagger starts so the fleet ramps up rather than all-at-once
    for i, p in enumerate(probs):
        task = p["task"]
        j = Journal(runs_dir / str(task) / "journal.jsonl", task)
        t0 = base + i * rng.uniform(2, 9)
        j.append("run_started", ts=_iso(t0), name=p["name"], family=p["family"], agent=p["agent"])
        t0 += rng.uniform(30, 110)
        j.append("design_done", ts=_iso(t0), dur_s=round(rng.uniform(40, 120), 1))
        js[task] = j
        clocks[task] = t0
        st[task] = {**p, "best": None, "eval_i": 0, "cand_i": 0, "frontier": 0}

    order = [p["task"] for p in probs]

    # bootstrap: seed eval for each
    for task in order:
        s, j = st[task], js[task]
        s["cand_i"] += 1
        cand = f"c{s['cand_i']:03d}-seed"
        gpu_s = rng.uniform(30, 75)
        enq, sT, d = run_gpu(task, gpu_s)
        job = f"{task}-j{s['eval_i']:03d}"
        j.append("exec_enqueued", ts=_iso(enq), job=job, cand=cand)
        j.append("exec_started", ts=_iso(sT), job=job)
        s["best"] = s["start"]
        j.append("exec_done", ts=_iso(d), job=job, cand=cand, gpu_s=round(gpu_s, 1),
                 all_passed=True, sol_score=round(s["best"], 4), statuses={"PASSED": 16},
                 strategy="seed: reference wrapper baseline (DPS)",
                 solution=_solution("pytorch", _SEED_CODE, f"{task}-seed"))
        s["frontier"] = 1
        j.append("accept", ts=_iso(d + 0.4), cand=cand, verdict="entered",
                 best=round(s["best"], 4), frontier=1)
        s["eval_i"] += 1
        s["evals_left"] -= 1
        clocks[task] = d + rng.uniform(4, 14)

    active = list(order)
    while active:
        for task in list(active):
            s, j = st[task], js[task]
            if s["evals_left"] <= 0:
                if not s["running"]:
                    j.append("terminated", ts=_iso(clocks[task]),
                             reason=rng.choice(["plateau", "budget"]))
                active.remove(task)
                continue
            s["cand_i"] += 1
            cand = f"c{s['cand_i']:03d}"
            dur = rng.uniform(25, 90)
            clocks[task] += dur
            strat, lang, code = _STRATEGIES[(s["cand_i"] - 2) % len(_STRATEGIES)]
            j.append("plan_done", ts=_iso(clocks[task]), cand=cand, parent="frontier",
                     model=s["agent"], dur_s=round(dur, 1),
                     tok_in=rng.randint(4000, 12000), tok_out=rng.randint(800, 3000),
                     strategy=strat, solution=_solution(lang, code, f"{task}-{cand}"))
            roll = rng.random()
            if roll < 0.08:
                j.append("check", ts=_iso(clocks[task]), cand=cand, ok=False)
                continue
            j.append("check", ts=_iso(clocks[task]), cand=cand, ok=True)
            if roll < 0.18:
                j.append("novelty", ts=_iso(clocks[task]), cand=cand, verdict="cosmetic-duplicate")
                continue
            j.append("novelty", ts=_iso(clocks[task]), cand=cand, verdict="materially-new")
            gpu_s = rng.uniform(28, 80)
            enq, sT, d = run_gpu(task, gpu_s)
            job = f"{task}-j{s['eval_i']:03d}"
            j.append("exec_enqueued", ts=_iso(enq), job=job, cand=cand)
            j.append("exec_started", ts=_iso(sT), job=job)
            if rng.random() < 0.15:
                j.append("exec_done", ts=_iso(d), job=job, cand=cand, gpu_s=round(gpu_s, 1),
                         all_passed=False, sol_score=None,
                         statuses={"PASSED": rng.randint(8, 14),
                                   "INCORRECT_NUMERICAL": rng.randint(1, 4)})
                verdict = "dominated"
            else:
                gap = s["ceil"] - s["best"]
                improved = rng.random() < 0.55
                score = s["best"] + (gap * rng.uniform(0.15, 0.45) if improved
                                     else -rng.uniform(0.01, 0.05))
                score = max(0.05, min(score, s["ceil"]))
                j.append("exec_done", ts=_iso(d), job=job, cand=cand, gpu_s=round(gpu_s, 1),
                         all_passed=True, sol_score=round(score, 4), statuses={"PASSED": 16})
                if score > s["best"]:
                    s["best"] = score
                    s["frontier"] = min(s["frontier"] + rng.choice([0, 1]), 6)
                    verdict = "entered"
                else:
                    verdict = "dominated"
            j.append("accept", ts=_iso(d + 0.4), cand=cand, verdict=verdict,
                     best=round(s["best"], 4), frontier=s["frontier"])
            s["eval_i"] += 1
            s["evals_left"] -= 1
            clocks[task] = d + rng.uniform(4, 18)

    # rental windows around the gaps
    span_end = gpu_free + 60
    windows = []
    prev = base - 60
    labels = ["pod-A", "pod-B", "pod-C"]
    for k, (gs, ge) in enumerate(gaps):
        windows.append({"start": _iso(prev), "end": _iso(gs), "label": labels[k]})
        prev = ge
    windows.append({"start": _iso(prev), "end": _iso(span_end), "label": labels[len(gaps) % len(labels)]})
    (runs_dir / "gpu_rentals.jsonl").write_text(
        "\n".join(json.dumps(w) for w in windows) + "\n")
    return runs_dir
