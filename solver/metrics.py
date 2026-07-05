"""Aggregate journal events into the metrics the dashboard renders.

Pure read-side: consumes `journal.read_all()` output, produces plain dicts.
The exec_enqueued/started/done triple is the measurement backbone:
queue wait = started - enqueued; GPU busy = done - started; the merged,
sorted job list across problems gives the global GPU timeline + utilization.
"""

from __future__ import annotations

import datetime as dt
import statistics
from typing import Any

OUTCOMES = ("accepted", "dominated", "incorrect", "rejected", "duplicate", "error")


def _t(ts: str) -> float:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round(p * (len(vals) - 1))))
    return vals[idx]


def problem_metrics(task_id: int, events: list[dict]) -> dict[str, Any]:
    jobs: dict[str, dict] = {}
    convergence: list[tuple[int, float]] = []   # (gpu_eval_index, best_so_far)
    outcomes = {k: 0 for k in OUTCOMES}
    agent = {"plan": {"n": 0, "dur": 0.0, "tok": 0},
             "reflect": {"n": 0, "dur": 0.0, "tok": 0},
             "design": {"n": 0, "dur": 0.0, "tok": 0}}
    iters = evals = 0
    best = None
    frontier = 0
    terminated = None
    name = family = model = ""
    first_ts = last_ts = None

    for e in events:
        ts = e.get("ts")
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        ev = e.get("ev")
        if ev == "run_started":
            name, family, model = e.get("name", ""), e.get("family", ""), e.get("agent", "")
        elif ev == "agent_changed":
            model = e.get("model", model)
        elif ev == "design_done":
            agent["design"]["n"] += 1
            agent["design"]["dur"] += e.get("dur_s", 0.0)
        elif ev == "plan_done":
            iters += 1
            agent["plan"]["n"] += 1
            agent["plan"]["dur"] += e.get("dur_s", 0.0)
            agent["plan"]["tok"] += e.get("tok_in", 0) + e.get("tok_out", 0)
        elif ev == "check" and not e.get("ok", True):
            outcomes["rejected"] += 1
        elif ev == "novelty" and e.get("verdict") != "materially-new":
            outcomes["duplicate"] += 1
        elif ev == "exec_enqueued":
            jobs[e["job"]] = {"task": task_id, "enq": _t(ts)}
        elif ev == "exec_started":
            jobs.setdefault(e["job"], {"task": task_id})["start"] = _t(ts)
        elif ev == "exec_done":
            j = jobs.setdefault(e["job"], {"task": task_id})
            j["done"] = _t(ts)
            j["gpu_s"] = e.get("gpu_s")
            evals += 1
            if e.get("solution_status") == "error":
                outcomes["error"] += 1
            elif not e.get("all_passed", False):
                outcomes["incorrect"] += 1
        elif ev == "accept":
            if e.get("verdict") == "entered":
                outcomes["accepted"] += 1
            elif e.get("verdict") == "dominated":
                outcomes["dominated"] += 1
            if e.get("best") is not None:
                best = e["best"]
                convergence.append((evals, best))
            frontier = e.get("frontier", frontier)
        elif ev == "reflect_done":
            agent["reflect"]["n"] += 1
            agent["reflect"]["dur"] += e.get("dur_s", 0.0)
        elif ev == "terminated":
            terminated = e.get("reason")

    waits = [j["start"] - j["enq"] for j in jobs.values() if "start" in j and "enq" in j]
    return {
        "task": task_id, "name": name, "family": family, "model": model,
        "iters": iters, "evals": evals, "best": best, "frontier": frontier,
        "terminated": terminated, "convergence": convergence,
        "outcomes": outcomes, "agent": agent, "jobs": list(jobs.values()),
        "wait_p50": _pct(waits, 0.5), "wait_p95": _pct(waits, 0.95),
        "first_ts": first_ts, "last_ts": last_ts,
    }


def fleet_metrics(per_problem: list[dict]) -> dict[str, Any]:
    jobs = [j for p in per_problem for j in p["jobs"]
            if "start" in j and "done" in j]
    jobs.sort(key=lambda j: j["start"])
    busy = sum(j["done"] - j["start"] for j in jobs)
    if jobs:
        span_start = min(j.get("enq", j["start"]) for j in jobs)
        span_end = max(j["done"] for j in jobs)
        span = max(span_end - span_start, 1e-9)
    else:
        span_start = span_end = span = 0.0
    waits = [j["start"] - j["enq"] for j in jobs if "enq" in j]
    return {
        "jobs": jobs, "busy_s": busy, "span_s": span,
        "span_start": span_start, "span_end": span_end,
        "gpu_util": (busy / span) if span else 0.0,
        "wait_p50": _pct(waits, 0.5), "wait_p95": _pct(waits, 0.95),
        "total_evals": len(jobs),
        "agent_calls": sum(p["agent"][k]["n"] for p in per_problem for k in p["agent"]),
        "agent_tokens": sum(p["agent"][k]["tok"] for p in per_problem for k in p["agent"]),
        "done": sum(1 for p in per_problem if p["terminated"]),
        "active": sum(1 for p in per_problem if not p["terminated"]),
        "mean_best": (statistics.mean([p["best"] for p in per_problem if p["best"] is not None])
                      if any(p["best"] is not None for p in per_problem) else None),
    }


def collect(journals: dict[int, list[dict]]) -> dict[str, Any]:
    per_problem = [problem_metrics(t, evs) for t, evs in sorted(journals.items())]
    return {"problems": per_problem, "fleet": fleet_metrics(per_problem)}
