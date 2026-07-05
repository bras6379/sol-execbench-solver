"""SOL-ExecBench score, vendored from the official harness.

The scoring FORMULA is a stable, pure-Python closed form, so we vendor it
verbatim to use on the laptop (no GPU / no heavy deps) for design-phase
"if I hit X ms, my score is Y" reasoning. Everything else in the grader
(correctness checking, timing, input generation, reward-hack detection) is
NOT reimplemented here — it runs on the GPU via the real harness (installed
as the ``bench`` optional dependency; see pyproject.toml).

Source: github.com/NVIDIA/SOL-ExecBench  src/sol_execbench/sol_score.py
Pinned: commit 2d852a30914d4ef7f9fac92696e7fc8eea630f52 (2026-05-28)
License: Apache-2.0. If the pinned harness changes this formula, re-vendor.
"""

from __future__ import annotations


def sol_score(t_k: float, t_b: float, t_sol: float) -> float:
    """Anchored score S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL)).

    S = 0.5 when the candidate matches the baseline (t_k == t_b); S = 1.0 at
    the Speed-of-Light bound (t_k == t_sol); S -> 0 as the candidate slows.

    Args:
        t_k:   candidate kernel runtime (ms)
        t_b:   optimized-PyTorch scoring baseline runtime (ms)
        t_sol: Speed-of-Light runtime (ms)
    """
    denom_gap = t_b - t_sol
    if denom_gap <= 0:
        return 1.0 if t_k <= t_sol else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / denom_gap)


def geomean(values: list[float]) -> float | None:
    """Geometric mean — the benchmark's per-problem latency aggregation
    (verified: the leaderboard SOL-row latency equals the geomean of the
    per-workload sol_ms). Benchmark-level score is the arithmetic mean of
    per-problem scores, correctness-gated (paper). See
    kb/benchmark-grader.md § Aggregation."""
    import math
    vals = [v for v in values if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def score_from_metadata(candidate_ms: list[float], sol: dict) -> dict:
    """Score a candidate's per-workload latencies against a problem's SOL data.

    ``sol`` is the ``metadata.json['sol']`` block written by ``solver fetch``
    (per_workload entries carry ``sol_ms`` and ``baseline_latency_ms``).
    Returns per-workload scores and the mean. Workloads with missing SOL data
    are skipped.
    """
    per = sol.get("per_workload", []) or []
    scores: list[float] = []
    rows: list[dict] = []
    for i, t_k in enumerate(candidate_ms):
        if i >= len(per):
            break
        t_sol = per[i].get("sol_ms")
        t_b = per[i].get("baseline_latency_ms")
        if t_sol is None or t_b is None:
            continue
        s = sol_score(t_k, t_b, t_sol)
        scores.append(s)
        rows.append({"index": i, "t_k": t_k, "t_b": t_b, "t_sol": t_sol, "score": s})
    mean = sum(scores) / len(scores) if scores else None
    return {"mean_score": mean, "per_workload": rows}
