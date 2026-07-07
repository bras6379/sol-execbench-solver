"""Aggregate journal events into the metrics the dashboard renders.

Pure read-side: consumes `journal.read_all()` output, produces plain dicts.
The exec_enqueued/started/done triple is the measurement backbone:
queue wait = started - enqueued; GPU busy = done - started; the merged,
sorted job list across problems gives the global GPU timeline.

GPU rentals: if `<runs_dir>/gpu_rentals.jsonl` exists (one JSON per line:
{"start": iso, "end": iso|null, "label": str}), utilization is computed
against RENTED time only and the timeline is drawn per rental window
(un-rented gaps compressed). Written by hand for now; later the
GpuQueueExecutor can append these automatically on pod connect/disconnect.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any

OUTCOMES = ("accepted", "dominated", "incorrect", "rejected", "duplicate", "flaky", "error")


def _t(ts: str) -> float:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round(p * (len(vals) - 1))))
    return vals[idx]


def load_rentals(runs_dir: Path) -> list[dict]:
    """Rental windows [{start, end, label}] (epoch seconds), sorted."""
    path = Path(runs_dir) / "gpu_rentals.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            out.append({"start": _t(r["start"]),
                        "end": _t(r["end"]) if r.get("end") else None,
                        "label": r.get("label", "")})
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return sorted(out, key=lambda r: r["start"])


def submission_summary(runs_dir: Path, task_id: int) -> dict | None:
    """The best REAL leaderboard submission for a problem (runs/<task>/submissions.jsonl):
    highest actual SOL + its board rank/size. None if never submitted. This is the
    ground truth (not the calibrated estimate) — surfaced on the problems table."""
    sf = Path(runs_dir) / str(task_id) / "submissions.jsonl"
    if not sf.exists():
        return None
    subs: dict = {}
    for line in sf.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = e.get("submission_id") or e.get("id")
        if sid is not None:
            subs.setdefault(sid, {}).update(e)
    if not subs:
        return None
    scored = [e for e in subs.values() if e.get("sol_score") is not None]
    best = max(scored, key=lambda e: e["sol_score"]) if scored else list(subs.values())[-1]
    return {
        "sol": best.get("sol_score"), "rank": best.get("board_rank"),
        "n": best.get("board_n"), "top_sol": best.get("board_top_sol"),
        "status": best.get("status"), "sid": best.get("submission_id") or best.get("id"),
    }


def problem_metrics(task_id: int, events: list[dict]) -> dict[str, Any]:
    jobs: dict[str, dict] = {}
    convergence: list[tuple[int, float]] = []   # (gpu_eval_index, best_so_far)
    accept_times: list[tuple[float, float]] = []  # (ts, best) for fleet-over-time
    outcomes = {k: 0 for k in OUTCOMES}
    agent = {"plan": {"n": 0, "dur": 0.0, "tok": 0},
             "design": {"n": 0, "dur": 0.0, "tok": 0}}
    iters = evals = 0
    best = None
    best_cal = None            # leaderboard-estimate score of the best candidate
    frontier = 0
    terminated = None
    last_improve_ts = None
    name = family = model = ""
    best_model = ""            # producer of the best candidate (who actually won)
    best_model_score = -1.0
    first_ts = last_ts = None
    candidates: dict[str, dict] = {}   # cand id -> progression record

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
            # Two agents can produce the SAME kernel (same content hash = same cand
            # id). Keep the FIRST occurrence — the one that got evaluated/accepted —
            # so a later exact-duplicate plan_done doesn't clobber the winner's row.
            candidates.setdefault(e["cand"], {
                "cand": e["cand"], "ts": ts, "model": e.get("model", model),
                "parent": e.get("parent"), "strategy": e.get("strategy", ""),
                "solution": e.get("solution"), "status": "planned",
                "sol_score": None,
            })
        elif ev == "check" and not e.get("ok", True):
            outcomes["rejected"] += 1
            if e.get("cand") in candidates:
                candidates[e["cand"]]["status"] = "rejected"
        elif ev == "novelty" and e.get("verdict") != "materially-new":
            outcomes["duplicate"] += 1
            c = candidates.get(e.get("cand"))
            if c and c["status"] == "planned":   # a re-generated exact dup must NOT
                c["status"] = "duplicate"        # downgrade the accepted/scored winner
        elif ev == "exec_enqueued":
            jobs[e["job"]] = {"task": task_id, "enq": _t(ts)}
        elif ev == "exec_started":
            jobs.setdefault(e["job"], {"task": task_id})["start"] = _t(ts)
        elif ev == "exec_done":
            j = jobs.setdefault(e["job"], {"task": task_id})
            j["done"] = _t(ts)
            j["gpu_s"] = e.get("gpu_s")
            evals += 1
            c = candidates.get(e.get("cand"))
            if c is None and e.get("cand"):   # seeds evaluate without plan_done
                c = candidates[e["cand"]] = {
                    "cand": e["cand"], "ts": ts, "model": model, "parent": None,
                    "strategy": e.get("strategy", "seed: reference wrapper baseline"),
                    "solution": e.get("solution"), "status": "planned",
                    "sol_score": None,
                }
            if e.get("solution_status") == "error":
                outcomes["error"] += 1
                if c:
                    c["status"] = "error"
            elif not e.get("all_passed", False):
                outcomes["incorrect"] += 1
                if c:
                    c["status"] = "incorrect"
            if c:
                c["sol_score"] = e.get("sol_score")
                c["sol_score_cal"] = e.get("sol_score_cal")
        elif ev == "accept":
            c = candidates.get(e.get("cand"))
            if e.get("verdict") == "entered":
                outcomes["accepted"] += 1
                if ts:
                    last_improve_ts = _t(ts)
                if c and c["status"] not in ("incorrect", "error"):
                    c["status"] = "accepted"
                if c and c.get("sol_score") is not None and c["sol_score"] >= best_model_score:
                    best_model_score = c["sol_score"]
                    best_model = c.get("model") or model
            elif e.get("verdict") == "dominated":
                outcomes["dominated"] += 1
                if c and c["status"] == "planned":
                    c["status"] = "dominated"
            if e.get("best") is not None:
                best = e["best"]
                if e.get("best_cal") is not None:
                    best_cal = e["best_cal"]
                # the fleet-over-time + convergence charts plot EXPECTED SOL (the
                # leaderboard estimate), falling back to raw when uncalibrated.
                show = best_cal if best_cal is not None else best
                convergence.append((evals, show))
                if ts:
                    accept_times.append((_t(ts), show))
            if c:
                c["best_after"] = best
            frontier = e.get("frontier", frontier)
        elif ev == "verify_started":
            jobs.setdefault(e["job"], {"task": task_id})["start"] = _t(ts)
        elif ev == "verify_done":                     # re-verification of a would-be frontier entry
            j = jobs.setdefault(e["job"], {"task": task_id})
            j["done"] = _t(ts)
            j["gpu_s"] = e.get("gpu_s")
            j["verify"] = True
            evals += 1
        elif ev == "flaky":                           # passed once, failed a fresh re-run → rejected
            outcomes["flaky"] += 1
            c = candidates.get(e.get("cand"))
            if c:
                c["status"] = "flaky"
        elif ev == "terminated":
            terminated = e.get("reason")
        elif ev == "reopened":
            terminated = None            # a reopened run is running again, not "budget:*"

    waits = [j["start"] - j["enq"] for j in jobs.values() if "start" in j and "enq" in j]
    return {
        "task": task_id, "name": name, "family": family, "model": best_model or model,
        "iters": iters, "evals": evals, "best": best, "best_cal": best_cal, "frontier": frontier,
        "terminated": terminated, "convergence": convergence,
        "accept_times": accept_times, "last_improve_ts": last_improve_ts,
        "outcomes": outcomes, "agent": agent, "jobs": list(jobs.values()),
        "candidates": sorted(candidates.values(), key=lambda c: c["ts"] or ""),
        "wait_p50": _pct(waits, 0.5), "wait_p95": _pct(waits, 0.95),
        "first_ts": first_ts, "last_ts": last_ts,
    }


def fleet_metrics(per_problem: list[dict], rentals: list[dict] | None = None) -> dict[str, Any]:
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

    # resolve rentals: open-ended windows close at span_end
    windows = []
    for r in (rentals or []):
        end = r["end"] if r["end"] is not None else max(span_end, r["start"])
        if end > r["start"]:
            windows.append({"start": r["start"], "end": end, "label": r["label"]})
    rented_s = sum(w["end"] - w["start"] for w in windows)

    waits = [j["start"] - j["enq"] for j in jobs if "enq" in j]
    return {
        "jobs": jobs, "busy_s": busy, "span_s": span,
        "span_start": span_start, "span_end": span_end,
        "windows": windows, "rented_s": rented_s,
        "gpu_util": (busy / rented_s) if rented_s else ((busy / span) if span else 0.0),
        "util_basis": "rented" if rented_s else "observed span",
        "wait_p50": _pct(waits, 0.5), "wait_p95": _pct(waits, 0.95),
        "total_evals": len(jobs),
        "agent_calls": sum(p["agent"][k]["n"] for p in per_problem for k in p["agent"]),
        "agent_tokens": sum(p["agent"][k]["tok"] for p in per_problem for k in p["agent"]),
        "done": sum(1 for p in per_problem if p["terminated"]),
        "active": sum(1 for p in per_problem if not p["terminated"]),
        "mean_best": (statistics.mean([p["best"] for p in per_problem if p["best"] is not None])
                      if any(p["best"] is not None for p in per_problem) else None),
        "mean_best_cal": (statistics.mean([p["best_cal"] for p in per_problem if p.get("best_cal") is not None])
                          if any(p.get("best_cal") is not None for p in per_problem) else None),
    }


def fleet_score_series(per_problem: list[dict]) -> list[tuple[float, float]]:
    """Stepwise fleet mean-of-best over wall time (problems count once seen)."""
    events = []
    for p in per_problem:
        for ts, best in p["accept_times"]:
            events.append((ts, p["task"], best))
    events.sort()
    cur: dict[int, float] = {}
    series = []
    for ts, task, best in events:
        cur[task] = best
        series.append((ts, sum(cur.values()) / len(cur)))
    return series


def score_histogram(per_problem: list[dict], bins: int = 20) -> list[int]:
    counts = [0] * bins
    for p in per_problem:
        if p["best"] is None:
            continue
        b = min(bins - 1, int(max(0.0, min(p["best"], 0.9999)) * bins))
        counts[b] += 1
    return counts


def family_rollup(per_problem: list[dict]) -> list[dict]:
    fams: dict[str, list[dict]] = {}
    for p in per_problem:
        fams.setdefault(p["family"] or "?", []).append(p)
    out = []
    for fam, ps in sorted(fams.items()):
        bests = [p["best"] for p in ps if p["best"] is not None]
        out.append({
            "family": fam, "n": len(ps),
            "done": sum(1 for p in ps if p["terminated"]),
            "mean_best": statistics.mean(bests) if bests else None,
            "evals": sum(p["evals"] for p in ps),
        })
    out.sort(key=lambda r: -(r["mean_best"] or 0))
    return out


def top_movers(per_problem: list[dict], k: int = 8) -> list[dict]:
    """The k most-recently-improving problems (for the convergence chart)."""
    ranked = sorted(per_problem,
                    key=lambda p: (p["last_improve_ts"] or 0, p["evals"]),
                    reverse=True)
    return [p for p in ranked if p["convergence"]][:k]


def collect(journals: dict[int, list[dict]], runs_dir: Path | None = None) -> dict[str, Any]:
    per_problem = [problem_metrics(t, evs) for t, evs in sorted(journals.items())]
    if runs_dir:                                   # attach the real leaderboard result per problem
        for p in per_problem:
            p["lb"] = submission_summary(runs_dir, p["task"])
    rentals = load_rentals(runs_dir) if runs_dir else []
    return {
        "problems": per_problem,
        "fleet": fleet_metrics(per_problem, rentals),
        "fleet_series": fleet_score_series(per_problem),
        "histogram": score_histogram(per_problem),
        "families": family_rollup(per_problem),
        "movers": top_movers(per_problem),
    }
