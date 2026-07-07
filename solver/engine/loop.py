"""The solve loop and the fleet (docs/orchestration.md §2).

`solve_problem` is one problem's GEPA loop: bootstrap (design + seed the
frontier) → repeat (round-robin the current tier's pool → plan → gates → GPU
eval → ε-Pareto accept → reflect) until a budget, target, or a terminating
plateau. `run_fleet` runs many concurrently, guarded so one failure stops only
its own problem. Everything is async; the executor is the single serialized GPU.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from .. import journal as journal_mod
from . import store
from .agent import Agent, solution_hash
from .config import Config, Perspective
from .context import RunContext
from .executor import EvalResult, Executor

SeedsFn = Callable[[int], list[dict]]
CheckFn = Callable[[dict, int], "tuple[bool, list[str]]"]


def _default_seeds(task_id: int) -> list[dict]:
    """Fallback seed: a single baseline shape (used only when no reference is
    available, e.g. unit tests)."""
    return [{"__eval__": {"scores": [0.5]}}]


def reference_seed(problems_dir: str | Path = "problems") -> SeedsFn:
    """Seed the frontier with the real PyTorch **reference** (the DPS baseline),
    so the reference impl is candidate #0 and always sits on/beside the frontier.
    Falls back to `_default_seeds` if the reference isn't fetched."""
    problems_dir = Path(problems_dir)

    def seeds_fn(task_id: int) -> list[dict]:
        ref = problems_dir / str(task_id) / "reference.py"
        if not ref.exists():
            return _default_seeds(task_id)
        return [{"spec": {"languages": ["pytorch"]},
                 "sources": [{"path": "reference.py", "content": ref.read_text()}]}]

    return seeds_fn


def _persist_reference(runs_dir: str | Path, task_id: int, problems_dir: str | Path) -> None:
    """Copy the reference impl into the run dir so the ground truth always sits
    beside the frontier kernels (runs/<task>/reference.py + definition.json)."""
    pdir = Path(problems_dir) / str(task_id)
    dest = Path(runs_dir) / str(task_id)
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("reference.py", "definition.json"):
        src = pdir / fname
        if src.exists():
            (dest / fname).write_text(src.read_text())


def _default_check(solution: dict, task_id: int) -> tuple[bool, list[str]]:
    """Stub gate: honour the agent's `__invalid__` marker and reject an empty
    Solution (a CLI agent that wrote no kernel). The real gate wraps
    `solver.check.check_solution` against the problem definition."""
    if solution.get("__invalid__"):
        return False, ["stub: marked invalid"]
    if "sources" in solution and not solution.get("sources"):
        return False, ["no source files produced"]
    return True, []


def _statuses(result: EvalResult) -> list[str]:
    return [("PASSED" if w.correct else (w.error or "FAILED")) for w in result.per_workload]


def _persist(ctx: RunContext, task_id: int, cid: str, solution: dict | None,
             result: EvalResult, *, runs_dir, problems_dir, family: str, name: str,
             **meta) -> None:
    """Best-effort durable store: the candidate + the refreshed frontier/best.
    A store failure warns but never aborts the solve."""
    try:
        store.record_candidate(runs_dir, task_id, cid, solution, result,
                               problems_dir=problems_dir, **meta)
        store.record_frontier(runs_dir, task_id, ctx.frontier,
                              problems_dir=problems_dir, family=family, name=name)
    except Exception as exc:                       # pragma: no cover - defensive
        ctx.record("store_error", cand=cid, error=repr(exc)[:200])


async def solve_problem(
    task_id: int,
    executor: Executor,
    agents: dict[Perspective, Agent],
    cfg: Config,
    *,
    runs_dir: str | Path = "runs",
    seed: int = 0,
    seeds_fn: SeedsFn | None = None,
    check_fn: CheckFn | None = None,
    knowledge=None,
    problems_dir: str | Path = "problems",
    family: str = "",
    name: str = "",
) -> RunContext:
    ctx = RunContext.load(task_id, cfg, runs_dir, seed=seed)
    ctx.reopen_if_capped()             # a cap-terminated run continues if the caps now allow it
    seeds_fn = seeds_fn or _default_seeds
    check_fn = check_fn or _default_check

    # ---- bootstrap (consumes GPU evals; committed by the `bootstrapped` marker) ----
    if ctx.fresh():
        ctx.record("run_started", agent=str(cfg.design_model), name=name, family=family)
        _persist_reference(runs_dir, task_id, problems_dir)   # ground truth beside the frontier
        design = await agents[cfg.design_model].design(task_id)
        ctx.record("design_done", text=design, dur_s=0.0)
        # sibling seeding (best same-family Solution so far) then the scaffold seed
        seeds = (knowledge.sibling_seed(task_id, family) if knowledge else []) + seeds_fn(task_id)
        for sol in seeds:
            cid = solution_hash(sol)[:12]
            ctx.record("exec_enqueued", job=cid, cand=cid)
            result = await executor.evaluate(sol, task_id)
            ctx.record("exec_started", job=cid, ts=result.raw.get("started"))
            ctx.record("exec_done", job=cid, cand=cid, ts=result.raw.get("ended"),
                       gpu_s=result.raw.get("gpu_s", 0.0), all_passed=result.correct,
                       sol_score=result.sol_score, sol_score_cal=result.calibrated_sol_score(),
                       scores=result.vector(), statuses=_statuses(result))
            verdict = ctx.accept_candidate(cid)
            _persist(ctx, task_id, cid, sol, result, runs_dir=runs_dir, problems_dir=problems_dir,
                     family=family, name=name, strategy="seed", verdict=verdict)
        ctx.record("bootstrapped")

    # ---- the loop ----
    while not ctx.done():
        if ctx.tier_plateaued():
            if not ctx.escalate():
                break
        persp = ctx.current_perspective()
        agent = agents[persp]
        parent = ctx.frontier.select(ctx.rng)
        cand = await agent.plan(parent, ctx)

        is_dup_hash = cand.cand_id in ctx.seen
        tok = cand.tokens or {}
        ctx.record("plan_done", cand=cand.cand_id, parent=cand.parent, agent=persp.agent,
                   model=persp.model, strategy=cand.strategy, solution=cand.solution,
                   dur_s=0.0, tok_in=(tok.get("in") or 0), tok_out=(tok.get("out") or 0),
                   trajectory=cand.trajectory)

        ok, _errs = check_fn(cand.solution, task_id)
        ctx.record("check", cand=cand.cand_id, ok=ok)
        if not ok:
            ctx.record("iter", n=ctx.iters, outcome="rejected")
            continue
        if is_dup_hash:
            ctx.record("novelty", cand=cand.cand_id, verdict="duplicate")
            ctx.record("iter", n=ctx.iters, outcome="duplicate")
            continue
        verdict_nov = await agent.judge(cand, parent, ctx.frontier)
        ctx.record("novelty", cand=cand.cand_id, verdict=verdict_nov)
        if verdict_nov != "materially-new":
            ctx.record("iter", n=ctx.iters, outcome="duplicate")
            continue

        ctx.record("exec_enqueued", job=cand.cand_id, cand=cand.cand_id)
        result = await executor.evaluate(cand.solution, task_id)
        ctx.record("exec_started", job=cand.cand_id, ts=result.raw.get("started"))
        ctx.record("exec_done", job=cand.cand_id, cand=cand.cand_id, ts=result.raw.get("ended"),
                   gpu_s=result.raw.get("gpu_s", 0.0), all_passed=result.correct,
                   sol_score=result.sol_score, sol_score_cal=result.calibrated_sol_score(),
                   scores=result.vector(), statuses=_statuses(result))
        verdict = ctx.accept_candidate(cand.cand_id)
        _persist(ctx, task_id, cand.cand_id, cand.solution, result, runs_dir=runs_dir,
                 problems_dir=problems_dir, family=family, name=name, strategy=cand.strategy,
                 agent=persp.agent, model=persp.model, parent=cand.parent,
                 verdict=verdict, trajectory=cand.trajectory)
        await agent.reflect(cand, result, verdict)
        ctx.record("reflect_done", cand=cand.cand_id, dur_s=0.0)
        ctx.record("iter", n=ctx.iters, outcome=verdict)

    if ctx.terminated_reason is None:
        if ctx.done():
            reason = ctx.done_reason()
        elif ctx.frontier.best_score() >= cfg.escalate_ceiling:
            reason = "converged:ceiling"
        else:
            reason = "converged:last-tier"
        ctx.record("terminated", reason=reason)

    try:                                                   # final flush: frontier.json + best_solution.json
        store.record_frontier(runs_dir, task_id, ctx.frontier,
                              problems_dir=problems_dir, family=family, name=name)
    except Exception:                                      # pragma: no cover - defensive
        pass

    if knowledge is not None:                              # serialized curator (§8)
        await knowledge.curate(ctx, family, name)
    return ctx


def exemplar_first(ids: list[int], families: dict[int, str] | None = None) -> list[int]:
    """Static launch order (Phase E refines this to exemplar-before-siblings)."""
    return list(ids)


async def run_fleet(
    ids: list[int],
    executor: Executor,
    agents: dict[Perspective, Agent],
    cfg: Config,
    *,
    runs_dir: str | Path = "runs",
    seed: int = 0,
    seeds_fn: SeedsFn | None = None,
    check_fn: CheckFn | None = None,
    knowledge=None,
    problems_dir: str | Path = "problems",
    families: dict[int, str] | None = None,
    names: dict[int, str] | None = None,
) -> None:
    families = families or {}
    names = names or {}

    async def guarded(t: int) -> None:
        try:
            await solve_problem(t, executor, agents, cfg, runs_dir=runs_dir, seed=seed,
                                seeds_fn=seeds_fn, check_fn=check_fn, knowledge=knowledge,
                                problems_dir=problems_dir,
                                family=families.get(t, ""), name=names.get(t, f"task-{t}"))
        except Exception as exc:  # crash isolation: one problem's failure is journaled, not fatal
            journal_mod.Journal(Path(runs_dir) / str(t) / "journal.jsonl", t).append(
                "solver_error", error=repr(exc))

    await asyncio.gather(*(guarded(t) for t in exemplar_first(ids, families)))
