"""Durable per-problem artifacts (docs/orchestration.md §8): every evaluated
candidate, the ε-Pareto frontier, and the submittable best — written *live* by
`solve_problem` so all of it survives a crash/resume and any candidate can be
inspected or submitted straight to the leaderboard.

Layout under `runs/<task>/`:
    candidates/<cid>.json   full record: raw engine candidate, per-workload
                            results, score/vector, verdict, trajectory pointer,
                            AND `submit` = a harness-format solution.json.
    candidates/index.jsonl  one compact line per candidate (fast listing).
    frontier.json           the current Pareto set (members → candidate files).
    best_solution.json      the best member's harness solution.json — submit this.

Seed solutions are captured here (the journal doesn't carry them), so the store
is the authoritative candidate archive, not a journal derivative.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .executor import EvalResult
from .frontier import Frontier
from .harness import solution_to_harness_json


def _write(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)                                  # atomic publish


def _submit_form(solution: dict, task_id: int, problems_dir, cid: str) -> dict | None:
    """The harness solution.json for a candidate, or None if it can't be built
    (no sources / problem not fetched / a stub candidate)."""
    try:
        if solution and solution.get("sources"):
            return solution_to_harness_json(solution, task_id, problems_dir, name=f"t{task_id}_{cid}")
    except Exception:
        pass
    return None


def record_candidate(runs_dir, task_id: int, cid: str, solution: dict | None,
                     result: EvalResult, *, strategy: str = "", agent: str = "",
                     model: str = "", parent: str | None = None, verdict: str = "",
                     trajectory=None, problems_dir="problems",
                     cross_op_patterns_shown: list[str] | None = None) -> None:
    """Persist one evaluated candidate (idempotent: overwrites <cid>.json,
    indexes it once)."""
    cdir = Path(runs_dir) / str(task_id) / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / f"{cid}.json"
    is_new = not cfile.exists()
    per = [{"index": w.index, "correct": w.correct, "latency_ms": w.latency_ms,
            "sol_ms": w.sol_ms, "baseline_latency_ms": w.baseline_latency_ms,
            "sol_score": w.sol_score, "sol_score_cal": w.calibrated_sol_score(),
            "error": w.error, "detail": w.detail} for w in result.per_workload]
    rec = {
        "cand_id": cid, "task_id": task_id, "verdict": verdict,
        "sol_score": result.sol_score, "sol_score_calibrated": result.calibrated_sol_score(),
        "correct": result.correct, "vector": result.vector(),
        "strategy": strategy, "agent": agent, "model": model, "parent": parent,
        "trajectory": str(trajectory) if trajectory else None,
        "gpu_s": result.raw.get("gpu_s"), "job_id": result.raw.get("job_id"),
        "asi": result.asi, "per_workload": per,
        # technique tags whose cross-op notes were shown for THIS attempt (docs/
        # context-architecture-plan.md Part B) — lets a later pass compare
        # first-attempt correctness/error rates with vs without notes shown.
        "cross_op_patterns_shown": cross_op_patterns_shown,
        "solution": solution,                                       # raw engine candidate
        "submit": _submit_form(solution, task_id, problems_dir, cid),  # ready-to-submit
    }
    _write(cfile, rec)
    if is_new:
        with (cdir / "index.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"cand_id": cid, "sol_score": result.sol_score,
                                "sol_score_cal": result.calibrated_sol_score(),
                                "correct": result.correct, "verdict": verdict,
                                "agent": agent, "model": model, "strategy": strategy}) + "\n")


def record_frontier(runs_dir, task_id: int, frontier: Frontier, *,
                    problems_dir="problems", family: str = "", name: str = "") -> None:
    """Write frontier.json (the Pareto set) + best_solution.json (submit this)."""
    base = Path(runs_dir) / str(task_id)
    base.mkdir(parents=True, exist_ok=True)
    best = frontier.best()
    members = sorted(frontier.members, key=lambda m: m.mean, reverse=True)
    _write(base / "frontier.json", {
        "task_id": task_id, "family": family, "name": name,
        "epsilon": frontier.epsilon, "size": len(frontier.members),
        "best_cand": best.cand_id if best else None,
        "best_score": best.mean if best else None,
        "best_score_cal": best.sol_score_cal if best else None,   # leaderboard estimate
        "members": [{
            "cand_id": m.cand_id, "sol_score": m.mean, "sol_score_cal": m.sol_score_cal,
            "all_passed": m.all_passed, "shapes_won": _shapes_won(m, members),
            "vector": list(m.vector), "strategy": m.strategy, "agent": m.agent, "model": m.model,
            "candidate": f"candidates/{m.cand_id}.json",
        } for m in members],
    })
    # best_solution.json = the submittable form of the best member, pulled from
    # its candidate record (robust to Member.solution being None for seeds).
    if best:
        cf = base / "candidates" / f"{best.cand_id}.json"
        submit = None
        if cf.exists():
            submit = json.loads(cf.read_text()).get("submit")
        if submit:
            _write(base / "best_solution.json", submit)


def record_playbook(runs_dir, task_id: int, playbook: list[dict], *, name: str = "") -> None:
    """Write the per-problem `playbook.md`: higher-ceiling ideas that accepted
    kernels flagged but did NOT ship (each banked when its author entered the
    frontier). Human-browsable, and the same list is fed to the next agent's
    context so reserve plays accumulate instead of dying in the trajectory."""
    if not playbook:
        return
    base = Path(runs_dir) / str(task_id)
    base.mkdir(parents=True, exist_ok=True)
    lines = [f"# Playbook — task {task_id}" + (f" · {name}" if name else ""), "",
             "Higher-ceiling ideas that accepted kernels flagged but did NOT ship.",
             "Banked when each author entered the frontier; the next agent reads these.", ""]
    for i, e in enumerate(playbook, 1):
        strat = (e.get("strategy") or "").strip()
        lines.append(f"## {i}. from `{e['cand'][:8]}`" + (f" — {strat}" if strat else ""))
        lines += [e["handoff"].strip(), ""]
    tmp = base / "playbook.md.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(base / "playbook.md")


def _shapes_won(m, members) -> int:
    """How many per-shape columns this member is (co-)best on — why it's on the set."""
    if not m.vector:
        return 0
    won = 0
    for i in range(len(m.vector)):
        top = max(o.vector[i] for o in members if len(o.vector) == len(m.vector))
        if m.vector[i] >= top - 1e-12:
            won += 1
    return won
