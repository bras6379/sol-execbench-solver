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

OUTCOMES = ("accepted", "dominated", "incorrect", "rejected", "duplicate", "no_op", "flaky", "error")

# Non-terminal "something is supposed to still be happening to this candidate"
# statuses. If the owning problem isn't actually live anymore (see collect()),
# nothing will ever resolve these — an abrupt stop (SIGTERM/SIGKILL) can land
# in the gap between a GPU eval finishing and its accept/frontier decision
# being journaled, permanently orphaning the candidate at whatever it was
# mid-transition to. Left alone, it looks identical to a live, healthy call.
_INFLIGHT_STATUSES = {"planned", "gpu_queued", "gpu_running", "reviewing", "repairing", "revised",
                      "verify_queued", "verifying"}


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
    # the EXPECTED sol of the exact kernel we submitted (from its candidate record) —
    # lets the dashboard say "current best beats what we submitted → re-submit".
    cand = best.get("cand_id")
    submitted_expected = None
    if cand:
        cf = Path(runs_dir) / str(task_id) / "candidates" / f"{cand}.json"
        if cf.exists():
            try:
                submitted_expected = json.loads(cf.read_text()).get("sol_score_calibrated")
            except (json.JSONDecodeError, OSError):
                pass
    return {
        "sol": best.get("sol_score"), "rank": best.get("board_rank"),
        "n": best.get("board_n"), "top_sol": best.get("board_top_sol"),
        "status": best.get("status"), "sid": best.get("submission_id") or best.get("id"),
        "cand": cand, "submitted_expected": submitted_expected,
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
    candidates: dict[str, dict] = {}   # cand id -> progression record (a repair round's id is an
                                        # ALIAS onto its lineage's shared row — see plan_done/check
                                        # below — so every id ever seen in one review-repair chain
                                        # resolves here without touching the other handlers)
    pending_repairs: dict[str, dict] = {}   # repair cand id -> staged row content, applied only
                                             # once `check` confirms it's actually valid — a repair
                                             # that fails check never ships, so its content must
                                             # never overwrite what the row is about to show
    candidate_roots: dict[str, str] = {}    # any cand id -> the lineage's root id (== itself for
                                             # a non-repair candidate; a repair round's id maps to
                                             # its chain's original root)
    cost = {"plan": 0.0, "review": 0.0, "diagnose": 0.0, "design": 0.0}   # $ by call-type
    cost_by_model: dict[str, dict] = {}    # model -> {cost, in, out, cached} — pricing (and
                                            # in/out/cache ratios) differ per provider, so raw
                                            # tokens matter alongside $
    noop_cost = 0.0                                         # $ spent on iterations that changed nothing
    reflect_health = {"success": 0, "fail": 0}               # diagnose_cost outcomes

    def _spend(kind: str, model: str, usd: float, tok_in: int = 0, tok_out: int = 0,
               tok_cached: int = 0) -> None:
        if not usd and not tok_in and not tok_out and not tok_cached:
            return
        cost[kind] = cost.get(kind, 0.0) + usd
        m = cost_by_model.setdefault(model, {"cost": 0.0, "in": 0, "out": 0, "cached": 0})
        m["cost"] += usd
        m["in"] += tok_in
        m["out"] += tok_out
        m["cached"] += tok_cached

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
            agent["design"]["tok"] += e.get("tok_in", 0) + e.get("tok_out", 0)
            _spend("design", e.get("model", "?"), e.get("cost_usd", 0.0),
                   e.get("tok_in", 0), e.get("tok_out", 0), e.get("tok_cached", 0))
        elif ev == "plan_done":
            iters += 1
            agent["plan"]["n"] += 1
            agent["plan"]["dur"] += e.get("dur_s", 0.0)
            agent["plan"]["tok"] += e.get("tok_in", 0) + e.get("tok_out", 0)
            plan_model = e.get("model", model)
            _spend("plan", plan_model, e.get("cost_usd", 0.0),
                   e.get("tok_in", 0), e.get("tok_out", 0), e.get("tok_cached", 0))
            if e.get("no_op"):
                noop_cost += e.get("cost_usd", 0.0)
            cid = e["cand"]
            if e.get("repair"):
                # A repair round is the SAME logical attempt as its pre-repair parent,
                # continuing (via a resumed CLI session) toward one eventual GPU
                # submission or abandonment — it must not appear as a separate
                # "recent attempt" row. Stage its content; `check` below adopts it
                # into the lineage's ONE shared row only if it's actually valid (a
                # repair that fails check never ships, so its content must never be
                # what the row shows — see the `check` handler).
                root_id = candidate_roots.get(e.get("parent"), e.get("parent"))
                if root_id in candidates:
                    candidate_roots[cid] = root_id
                    pending_repairs[cid] = {
                        "model": e.get("model", model), "strategy": e.get("strategy", ""),
                        "solution": e.get("solution"), "context_read": e.get("context_read"),
                    }
                else:
                    # parent's own row is missing (shouldn't normally happen — e.g. a
                    # truncated journal) — fail safe as its own root rather than lose
                    # the row entirely.
                    candidate_roots[cid] = cid
                    candidates[cid] = {
                        "cand": cid, "ts": ts, "model": e.get("model", model),
                        "parent": e.get("parent"), "strategy": e.get("strategy", ""),
                        "solution": e.get("solution"), "status": "planned",
                        "sol_score": None, "context_read": e.get("context_read"),
                        "reviewer_context_read": None, "repairs": 0,
                    }
            else:
                candidate_roots[cid] = cid
                # Two agents can produce the SAME kernel (same content hash = same cand
                # id). Keep the FIRST occurrence — the one that got evaluated/accepted —
                # so a later exact-duplicate plan_done doesn't clobber the winner's row.
                candidates.setdefault(cid, {
                    "cand": cid, "ts": ts, "model": e.get("model", model),
                    "parent": e.get("parent"), "strategy": e.get("strategy", ""),
                    "solution": e.get("solution"), "status": "planned",
                    # None (key absent — journal predates this feature) is kept distinct
                    # from [] (tracked, and the model genuinely read no kb/ files) —
                    # collapsing them would make every pre-2026-07-08 candidate falsely
                    # look like it read nothing.
                    "sol_score": None, "context_read": e.get("context_read"),
                    "reviewer_context_read": None, "repairs": 0,
                })
        elif ev == "check":
            cid = e.get("cand")
            if not e.get("ok", True):
                outcomes["rejected"] += 1
                if cid in candidates:
                    candidates[cid]["status"] = "rejected"
                pending_repairs.pop(cid, None)   # this repair's content never ships
            else:
                staged = pending_repairs.pop(cid, None)
                if staged is not None:
                    row = candidates.get(candidate_roots.get(cid, cid))
                    if row is not None:
                        row.update(staged)
                        row["cand"], row["ts"] = cid, ts
                        row["status"] = "planned"          # a fresh, not-yet-(re)reviewed attempt
                        row["reviewer_context_read"] = None
                        row["repairs"] = row.get("repairs", 0) + 1
                        candidates[cid] = row               # alias: later events keyed by cid resolve here
        elif ev == "review":
            _spend("review", e.get("reviewer", "?"), e.get("cost_usd", 0.0),
                   e.get("tok_in", 0), e.get("tok_out", 0), e.get("tok_cached", 0))
            c = candidates.get(e.get("cand"))
            if c is not None:
                c["reviewer_context_read"] = e.get("context_read")
                # A "revise" verdict sends the SAME writer back for a repair turn — this
                # candidate is done, superseded by whatever plan_done comes next in the
                # repair loop. Without this, it sits at "planned" forever (indistinguishable
                # from genuinely in-flight work) since nothing else ever revisits its status.
                if e.get("verdict") == "revise" and c["status"] == "planned":
                    c["status"] = "revised"
        elif ev == "diagnose_cost":
            _spend("diagnose", e.get("model", "?"), e.get("cost_usd", 0.0),
                   e.get("tok_in", 0) or 0, e.get("tok_out", 0) or 0, e.get("tok_cached", 0) or 0)
            reflect_health["success" if e.get("success") else "fail"] += 1
        elif ev == "novelty" and e.get("verdict") == "no_op":
            # the agent's output hashed EXACTLY to its own parent — it changed
            # nothing. Distinct from a generic duplicate: this is the dominant
            # real-world waste mode (measured live: ~97% of iterations on a
            # stuck problem) and, done paid-for on every call, a direct cost sink.
            outcomes["no_op"] += 1
            c = candidates.get(e.get("cand"))
            if c and c["status"] == "planned":
                c["status"] = "no_op"
        elif ev == "novelty" and e.get("verdict") != "materially-new":
            outcomes["duplicate"] += 1
            c = candidates.get(e.get("cand"))
            if c and c["status"] == "planned":   # a re-generated exact dup must NOT
                c["status"] = "duplicate"        # downgrade the accepted/scored winner
        elif ev == "exec_enqueued":
            jobs[e["job"]] = {"task": task_id, "enq": _t(ts)}
            # queued for the single-flight GPU — may sit here a while if another
            # problem currently holds it. Distinct from "gpu_running" below so a
            # long queue wait isn't mistaken for a stuck/hung eval.
            c = candidates.get(e.get("cand"))
            if c and c["status"] == "planned":
                c["status"] = "gpu_queued"
        elif ev == "exec_started":
            jobs.setdefault(e["job"], {"task": task_id})["start"] = _t(ts)
            # exec_started carries `job` only (== the cand id at both call sites),
            # not `cand` — look it up by job.
            c = candidates.get(e.get("job"))
            if c and c["status"] in ("planned", "gpu_queued"):
                c["status"] = "gpu_running"
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
                    "sol_score": None, "context_read": None, "reviewer_context_read": None,
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
                # Same permissive guard as "entered" above: by the time accept fires,
                # a correct candidate's status is "gpu_running" (set at exec_started),
                # not "planned" — a strict == "planned" check here (the original bug)
                # left every dominated-but-correct candidate stuck showing "gpu:
                # running" forever, alongside its real (final) score.
                if c and c["status"] not in ("incorrect", "error"):
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
        elif ev == "verify_enqueued":
            # This candidate already has a real score (from exec_done) but isn't
            # accepted yet — re-verification competes for the SAME single-flight
            # GPU lock as every other problem, so it can sit queued for a while.
            # Without this, the row keeps showing the stale "gpu: running" from
            # the eval that already finished — confirmed live (2026-07-08) this
            # made two DIFFERENT problems look like they were both "running" on
            # the GPU at once, when only one actually held the lock.
            c = candidates.get(e.get("cand"))
            if c and c["status"] in ("planned", "gpu_running"):
                c["status"] = "verify_queued"
        elif ev == "verify_started":
            jobs.setdefault(e["job"], {"task": task_id})["start"] = _t(ts)
            c = candidates.get(e.get("cand"))
            if c and c["status"] in ("planned", "gpu_running", "verify_queued"):
                c["status"] = "verifying"
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
    # A repair round's id is an ALIAS onto its lineage's shared row (see plan_done/
    # check above), so candidates.values() yields that same row once per alias —
    # dedupe by object identity so a review-repair chain surfaces as ONE row.
    seen_rows: set[int] = set()
    uniq_candidates = []
    for c in candidates.values():
        if id(c) not in seen_rows:
            seen_rows.add(id(c))
            uniq_candidates.append(c)
    return {
        "task": task_id, "name": name, "family": family, "model": best_model or model,
        "iters": iters, "evals": evals, "best": best, "best_cal": best_cal, "frontier": frontier,
        "terminated": terminated, "convergence": convergence,
        "accept_times": accept_times, "last_improve_ts": last_improve_ts,
        "outcomes": outcomes, "agent": agent, "jobs": list(jobs.values()),
        "candidates": sorted(uniq_candidates, key=lambda c: c["ts"] or ""),
        "wait_p50": _pct(waits, 0.5), "wait_p95": _pct(waits, 0.95),
        "first_ts": first_ts, "last_ts": last_ts,
        "cost": cost, "cost_by_model": cost_by_model, "noop_cost": noop_cost,
        "reflect_health": reflect_health,
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

    cost_total = {"plan": 0.0, "review": 0.0, "diagnose": 0.0, "design": 0.0}
    cost_by_model: dict[str, dict] = {}   # model -> {cost, in, out, cached}, summed fleet-wide
    noop_cost = 0.0
    reflect_health = {"success": 0, "fail": 0}
    for p in per_problem:
        for k, v in (p.get("cost") or {}).items():
            cost_total[k] = cost_total.get(k, 0.0) + v
        for m, d in (p.get("cost_by_model") or {}).items():
            agg = cost_by_model.setdefault(m, {"cost": 0.0, "in": 0, "out": 0, "cached": 0})
            for k in ("cost", "in", "out", "cached"):
                agg[k] += (d or {}).get(k, 0)
        noop_cost += p.get("noop_cost", 0.0)
        for k, v in (p.get("reflect_health") or {}).items():
            reflect_health[k] = reflect_health.get(k, 0) + v
    total_cost = sum(cost_total.values())

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
        "total_cost": total_cost, "cost_by_kind": cost_total, "cost_by_model": cost_by_model,
        "noop_cost": noop_cost, "reflect_health": reflect_health,
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


def _age_s(ts) -> float | None:
    """Seconds since an ISO timestamp, or None if unparseable."""
    if not ts:
        return None
    try:
        d = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return (dt.datetime.now(dt.timezone.utc) - d).total_seconds()
    except (ValueError, TypeError):
        return None


def _live_state(p: dict, active_set: set, active_fresh: bool) -> str:
    """running (holds a concurrency slot / actively worked) · waiting (started but
    queued for a slot) · pending (not started yet) · else the terminal reason.
    Prefers the engine's exact working-set (runs/_active.json); falls back to the
    recency of the last journal event when that file is absent or stale."""
    # Holding a slot RIGHT NOW is ground truth — it overrides a stale terminal reason
    # left in the journal by a prior run that is now being resumed/reopened.
    if active_fresh and p["task"] in active_set:
        return "running"
    if p["terminated"]:
        return p["terminated"]
    if active_fresh:
        return "pending" if p["evals"] == 0 else "waiting"
    age = _age_s(p.get("last_ts"))                 # fallback: no engine active-file
    if age is not None and age < 720:              # a slot-holder logs within ~12 min
        return "running"
    return "pending" if p["evals"] == 0 else "waiting"


def collect(journals: dict[int, list[dict]], runs_dir: Path | None = None) -> dict[str, Any]:
    per_problem = [problem_metrics(t, evs) for t, evs in sorted(journals.items())]
    if runs_dir:                                   # attach the real leaderboard result per problem
        board = {}                                 # cached leaderboard #1 per problem (solver poll --all)
        bf = Path(runs_dir) / "leaderboard.json"
        if bf.exists():
            try:
                board = json.loads(bf.read_text())
            except (json.JSONDecodeError, OSError):
                board = {}
        for p in per_problem:
            p["lb"] = submission_summary(runs_dir, p["task"])
            p["board"] = board.get(str(p["task"]))   # {top_sol, top_user, n, sol_bound, scores} = the #1 to beat + full distribution for rank projection
    # live working-set: which problems currently hold a concurrency slot (from the
    # engine's runs/_active.json) so status shows running vs waiting vs pending.
    active_set, active_fresh, live = set(), False, None
    task_phase: dict[str, dict] = {}
    if runs_dir:
        af = Path(runs_dir) / "_active.json"
        if af.exists():
            try:
                a = json.loads(af.read_text())
                active_set = set(a.get("active", []))
                age = _age_s(a.get("ts"))
                active_fresh = age is not None and age < 180
                live = {"phase": a.get("phase"), "reflect": a.get("reflect"),
                        "cap": a.get("cap"), "n_active": len(active_set),
                        "fresh": active_fresh, "age_s": None if age is None else round(age)}
                if active_fresh:      # a stale file's phase snapshot is almost certainly wrong
                    task_phase = a.get("task_phase") or {}
            except (json.JSONDecodeError, OSError):
                pass
    _PHASE_TO_STATUS = {"review": "reviewing", "repair": "repairing"}
    for p in per_problem:
        p["live_state"] = _live_state(p, active_set, active_fresh)
        # "what is this problem's agent doing RIGHT NOW" — design/plan/review/repair,
        # which model, since when — so the dashboard can show real-time activity
        # instead of forcing a guess from process list / journal recency.
        tp = task_phase.get(str(p["task"]))
        p["current_phase"] = {**tp, "elapsed_s": _age_s(tp.get("started"))} if tp else None
        # A review/repair call in flight is happening TO a specific existing
        # candidate (tp["cand"]) — surface that on its row too, so "planned"
        # doesn't look identical to "actively being re-reviewed right now".
        # (A "plan" phase has no row yet — its candidate doesn't exist until the
        # call finishes — so only review/repair correlate to an existing row.)
        new_status = _PHASE_TO_STATUS.get((tp or {}).get("phase"))
        if new_status and tp.get("cand"):
            for c in p["candidates"]:
                if c["cand"] == tp["cand"] and c["status"] in ("planned", "gpu_queued", "revised"):
                    c["status"] = new_status
                    break
        # This problem isn't currently live — nothing is going to advance an
        # in-flight status further. Say so honestly instead of leaving a stale
        # "gpu: running"/"under review"/etc that looks like live activity.
        if p["live_state"] != "running":
            for c in p["candidates"]:
                if c["status"] in _INFLIGHT_STATUSES:
                    c["status"] = "interrupted"
    rentals = load_rentals(runs_dir) if runs_dir else []
    return {
        "problems": per_problem,
        "live": live,
        "fleet": fleet_metrics(per_problem, rentals),
        "fleet_series": fleet_score_series(per_problem),
        "histogram": score_histogram(per_problem),
        "families": family_rollup(per_problem),
        "movers": top_movers(per_problem),
    }
