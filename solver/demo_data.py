"""Synthetic run journals for previewing the dashboard before the engine exists.

Simulates a small fleet with the real event vocabulary: a single-flight GPU
(round-robin-ish interleaving falls out of the simulated queue), agents
thinking in parallel, improving convergence curves, rejects/dups, and one
family chain (rmsnorm exemplar → siblings via template seeding).
Deterministic (seeded RNG).
"""

from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

from .journal import Journal

_PROBLEMS = [
    # (task, name, family, evals, start_best, ceiling, agent)
    (230, "021_rmsnorm_h128", "rmsnorm", 14, 0.42, 0.93, "sonnet"),
    (231, "022_rmsnorm_h512", "rmsnorm", 6, 0.71, 0.95, "sonnet"),   # seeded from 230
    (232, "023_rmsnorm_h1536", "rmsnorm", 5, 0.74, 0.94, "haiku"),   # seeded from 230
    (69, "069_rms_norm", "fused-norm", 16, 0.38, 0.88, "sonnet"),
    (67, "067_flash_attention_gqa_ultralong", "attention-fwd", 18, 0.31, 0.79, "opus"),
]


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat().replace("+00:00", "Z")


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
     "#include <torch/extension.h>\n// warp-shuffle rowwise rmsnorm; float4 loads; grid-stride\nvoid run(torch::Tensor h, torch::Tensor r, torch::Tensor w,\n         double eps, torch::Tensor out) { /* kernel launch */ }\n"),
    ("online-softmax tiling + TF32 dot", "triton",
     "# flash-style online softmax; tl.dot(allow_tf32=True); fp32 accum\n"),
]


def _solution(lang: str, code: str, name: str) -> dict:
    entry = "kernel.cu::run" if lang == "cuda_cpp" else "kernel.py::run"
    path = "kernel.cu" if lang == "cuda_cpp" else "kernel.py"
    return {"name": name, "definition": "demo", "author": "demo",
            "spec": {"languages": [lang], "target_hardware": ["B200"],
                     "entry_point": entry, "destination_passing_style": True},
            "sources": [{"path": path, "content": code}]}


_SEED_CODE = ("import torch\n\n# DPS wrapper delegating to the inlined reference (seed)\n"
              "def run(*args):\n    out = args[-1]\n    out[:] = _reference_run(*args[:-1])\n")


def build_demo(runs_dir: Path, seed: int = 7) -> Path:
    rng = random.Random(seed)
    runs_dir = Path(runs_dir)
    base = dt.datetime.now(dt.timezone.utc).timestamp() - 2.5 * 3600

    gpu_free = base          # single-flight GPU cursor
    clocks = {}              # per-task local clock
    journals = {}
    states = {}

    for i, (task, name, family, evals, start, ceil, agent) in enumerate(_PROBLEMS):
        j = Journal(runs_dir / str(task) / "journal.jsonl", task)
        # family chain: siblings 231/232 start after the exemplar's span midpoint
        t0 = base + i * 40 + (2400 if task in (231, 232) else 0)
        j.append("run_started", ts=_iso(t0), name=name, family=family, agent=agent)
        t0 += rng.uniform(60, 140)
        j.append("design_done", ts=_iso(t0), dur_s=round(rng.uniform(45, 120), 1))
        journals[task] = j
        clocks[task] = t0
        states[task] = {"best": None, "start": start, "ceil": ceil, "evals_left": evals,
                        "eval_i": 0, "frontier": 0, "cand_i": 0, "agent": agent}

    # two rental windows with an un-rented gap between them
    gap_start, gap_end = base + 1500, base + 2400

    def run_gpu(task: int, gpu_s: float) -> tuple[float, float, float]:
        nonlocal gpu_free
        enq = clocks[task]
        start_t = max(enq, gpu_free) + rng.uniform(0.3, 1.2)
        if gap_start <= start_t < gap_end or start_t + gpu_s > gap_start > start_t:
            start_t = max(start_t, gap_end)   # GPU not rented during the gap
        done_t = start_t + gpu_s
        gpu_free = done_t
        return enq, start_t, done_t

    # seed evals first (bootstrap), then iterate every task until budget spent
    order = [t for t, *_ in _PROBLEMS]
    for task in order:
        st, j = states[task], journals[task]
        st["cand_i"] += 1
        cand = f"c{st['cand_i']:03d}-seed"
        gpu_s = rng.uniform(35, 70)
        enq, s, d = run_gpu(task, gpu_s)
        job = f"{task}-j{st['eval_i']:03d}"
        j.append("exec_enqueued", ts=_iso(enq), job=job, cand=cand)
        j.append("exec_started", ts=_iso(s), job=job)
        st["best"] = st["start"]
        j.append("exec_done", ts=_iso(d), job=job, cand=cand, gpu_s=round(gpu_s, 1),
                 all_passed=True, sol_score=st["best"],
                 statuses={"PASSED": 16},
                 strategy="seed: reference wrapper baseline (DPS)",
                 solution=_solution("pytorch", _SEED_CODE, f"{task}-seed"))
        st["frontier"] = 1
        j.append("accept", ts=_iso(d + 0.5), cand=cand, verdict="entered",
                 best=round(st["best"], 4), frontier=1)
        st["eval_i"] += 1
        st["evals_left"] -= 1
        clocks[task] = d + rng.uniform(5, 15)

    active = [t for t, *_ in _PROBLEMS]
    while active:
        for task in list(active):
            st, j = states[task], journals[task]
            if st["evals_left"] <= 0:
                j.append("terminated", ts=_iso(clocks[task]),
                         reason=rng.choice(["plateau", "budget"]))
                active.remove(task)
                continue
            # plan (agent thinking — advances only this task's clock)
            st["cand_i"] += 1
            cand = f"c{st['cand_i']:03d}"
            dur = rng.uniform(25, 90)
            clocks[task] += dur
            strat_name, lang, code = _STRATEGIES[(st["cand_i"] - 2) % len(_STRATEGIES)]
            j.append("plan_done", ts=_iso(clocks[task]), cand=cand, parent="frontier",
                     model=st["agent"], dur_s=round(dur, 1),
                     tok_in=rng.randint(4000, 12000), tok_out=rng.randint(800, 3000),
                     strategy=strat_name,
                     solution=_solution(lang, code, f"{task}-{cand}"))
            roll = rng.random()
            if roll < 0.08:
                j.append("check", ts=_iso(clocks[task]), cand=cand, ok=False)
                continue
            j.append("check", ts=_iso(clocks[task]), cand=cand, ok=True)
            if roll < 0.18:
                j.append("novelty", ts=_iso(clocks[task]), cand=cand,
                         verdict="cosmetic-duplicate")
                continue
            j.append("novelty", ts=_iso(clocks[task]), cand=cand, verdict="materially-new")
            # evaluate
            gpu_s = rng.uniform(30, 80)
            enq, s, d = run_gpu(task, gpu_s)
            job = f"{task}-j{st['eval_i']:03d}"
            j.append("exec_enqueued", ts=_iso(enq), job=job, cand=cand)
            j.append("exec_started", ts=_iso(s), job=job)
            failed = rng.random() < 0.15
            if failed:
                score = None
                j.append("exec_done", ts=_iso(d), job=job, cand=cand,
                         gpu_s=round(gpu_s, 1), all_passed=False, sol_score=None,
                         statuses={"PASSED": rng.randint(8, 14),
                                   "INCORRECT_NUMERICAL": rng.randint(1, 4)})
                verdict = "dominated"
            else:
                gap = st["ceil"] - st["best"]
                improved = rng.random() < 0.55
                score = st["best"] + (gap * rng.uniform(0.15, 0.45) if improved
                                      else -rng.uniform(0.01, 0.05))
                score = max(0.05, min(score, st["ceil"]))
                j.append("exec_done", ts=_iso(d), job=job, cand=cand,
                         gpu_s=round(gpu_s, 1), all_passed=True,
                         sol_score=round(score, 4), statuses={"PASSED": 16})
                if score > st["best"]:
                    st["best"] = score
                    st["frontier"] = min(st["frontier"] + rng.choice([0, 1]), 5)
                    verdict = "entered"
                else:
                    verdict = "dominated"
            j.append("accept", ts=_iso(d + 0.5), cand=cand, verdict=verdict,
                     best=round(st["best"], 4), frontier=st["frontier"])
            j.append("reflect_done", ts=_iso(d + rng.uniform(10, 30)), cand=cand,
                     tier="full" if verdict == "entered" else "brief",
                     dur_s=round(rng.uniform(15, 45), 1))
            st["eval_i"] += 1
            st["evals_left"] -= 1
            clocks[task] = d + rng.uniform(5, 20)

    # rental windows file (later: written automatically by the GPU executor)
    import json
    (runs_dir / "gpu_rentals.jsonl").write_text(
        json.dumps({"start": _iso(base - 60), "end": _iso(gap_start),
                    "label": "pod-A"}) + "\n" +
        json.dumps({"start": _iso(gap_end), "end": _iso(gpu_free + 60),
                    "label": "pod-B"}) + "\n")
    return runs_dir
