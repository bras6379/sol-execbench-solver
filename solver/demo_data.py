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

    def run_gpu(task: int, gpu_s: float) -> tuple[float, float, float]:
        nonlocal gpu_free
        enq = clocks[task]
        start_t = max(enq, gpu_free) + rng.uniform(0.3, 1.2)
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
                 statuses={"PASSED": 16})
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
            j.append("plan_done", ts=_iso(clocks[task]), cand=cand, parent="frontier",
                     model=st["agent"], dur_s=round(dur, 1),
                     tok_in=rng.randint(4000, 12000), tok_out=rng.randint(800, 3000))
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

    return runs_dir
