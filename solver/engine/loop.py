"""The solve loop and the fleet (docs/orchestration.md §2).

`solve_problem` is one problem's GEPA loop: bootstrap (design + seed the
frontier) → repeat (round-robin the current tier's pool → plan → gates → GPU
eval → ε-Pareto accept → bank the plan's handoff into the playbook) until a
budget, target, or a terminating plateau. `run_fleet` runs many concurrently,
guarded so one failure stops only its own problem. Everything is async; the
executor is the single serialized GPU.
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import Awaitable, Callable

from .. import journal as journal_mod
from . import store
from .agent import Agent, solution_hash
from .config import Config, Perspective
from .context import RunContext
from .executor import EvalResult, Executor
from .knowledge import op_key_of

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


def _dominant_failure(result: EvalResult) -> str:
    """The most common failure kind across a candidate's workloads (for feedback)."""
    from collections import Counter
    if (result.asi or {}).get("solution_status"):
        return result.asi["solution_status"]                  # COMPILE_ERROR / REWARD_HACK
    errs = [w.error for w in result.per_workload if not w.correct and w.error]
    return Counter(errs).most_common(1)[0][0] if errs else "INCORRECT"


async def _reverify(ctx: RunContext, executor: Executor, task_id: int, cid: str,
                    solution: dict, runs: int) -> int | None:
    """Re-run a would-be frontier entry `runs`-1 more times as FRESH evals (same
    seed/config as the grader — just more correctness rounds). Returns the attempt
    that DISAGREED on correctness (→ flaky/racy kernel), or None if all re-runs
    passed. Each re-run costs a GPU eval and is journaled (verify_started/_done)
    so resume replays it. Gated by the caller onto would-be frontier entries only,
    so most candidates never pay for it."""
    for attempt in range(1, runs):
        job = f"{cid}-v{attempt}"
        r = await executor.evaluate(solution, task_id, attempt=attempt)
        ctx.record("verify_started", cand=cid, attempt=attempt, job=job,
                   ts=r.raw.get("started"))
        ctx.record("verify_done", cand=cid, attempt=attempt, job=job,
                   ts=r.raw.get("ended"), gpu_s=r.raw.get("gpu_s", 0.0),
                   all_passed=r.correct, scores=r.vector(), statuses=_statuses(r))
        if not r.correct:
            return attempt
    return None


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
        store.record_playbook(runs_dir, task_id, ctx.playbook, name=name)
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
    if cfg.time_limit_s:               # wall-clock budget for THIS run (resume gets a fresh one)
        ctx.deadline = time.monotonic() + cfg.time_limit_s
    seeds_fn = seeds_fn or _default_seeds
    check_fn = check_fn or _default_check

    # ---- bootstrap (consumes GPU evals; committed by the `bootstrapped` marker) ----
    if ctx.fresh():
        ctx.record("run_started", agent=str(cfg.design_model), name=name, family=family)
        _persist_reference(runs_dir, task_id, problems_dir)   # ground truth beside the frontier
        try:
            design = await agents[cfg.design_model].design(task_id)
        except Exception as exc:                              # a slow/failed design must not abort
            design = ""
            ctx.record("design_error", error=repr(exc)[:200])
        ctx.record("design_done", text=design, dur_s=0.0)
        # cross-problem transfer: hand the agent the best SAME-OP sibling's kernel as a
        # warm start to adapt (NOT eval'd — a sibling kernel usually hardcodes its shape).
        if knowledge is not None:
            ctx.sibling_hint = knowledge.sibling_hint(op_key_of(task_id, problems_dir),
                                                      exclude_task=task_id)
        seeds = seeds_fn(task_id)
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
        if persp is None:                          # every agent in this tier is dead (circuit-broken)
            if not ctx.route_around_dead_tier():   # → downgrade to a tier that still has a live agent
                ctx.record("terminated", reason="agents-unavailable")
                break
            persp = ctx.current_perspective()
        agent = agents[persp]
        parent = ctx.frontier.select(ctx.rng)
        try:
            cand = await agent.plan(parent, ctx)
        except Exception as exc:
            # agent timed out / wrote no kernel: skip THIS iteration, keep the
            # problem's frontier, advance to the next perspective. Never abort.
            ctx.record("plan_error", agent=persp.agent, model=persp.model, error=repr(exc)[:200])
            ctx.record("iter", n=ctx.iters, outcome="agent_error")
            continue

        is_dup_hash = cand.cand_id in ctx.seen
        tok = cand.tokens or {}
        ctx.record("plan_done", cand=cand.cand_id, parent=cand.parent, agent=persp.agent,
                   model=persp.model, strategy=cand.strategy, solution=cand.solution,
                   dur_s=0.0, tok_in=(tok.get("in") or 0), tok_out=(tok.get("out") or 0),
                   trajectory=cand.trajectory, handoff=cand.handoff)

        ok, _errs = check_fn(cand.solution, task_id)
        ctx.record("check", cand=cand.cand_id, ok=ok)
        if not ok:
            ctx.record("iter", n=ctx.iters, outcome="rejected")
            continue
        if is_dup_hash:                          # exact same kernel already seen → skip (free)
            ctx.record("novelty", cand=cand.cand_id, verdict="duplicate")
            ctx.record("iter", n=ctx.iters, outcome="duplicate")
            continue
        # NO LLM novelty pre-filter. The ε-Pareto frontier IS the novelty gate: it
        # keeps a candidate only if its MEASURED perf is non-dominated, and discards
        # a near-duplicate that doesn't actually improve. Pre-judging from one-line
        # strategy strings wrongly threw away real variants (FP16-vs-TF32, tile/warp
        # autotuning) after we'd already paid to generate them. We measure; the
        # frontier decides.

        ctx.record("exec_enqueued", job=cand.cand_id, cand=cand.cand_id)
        result = await executor.evaluate(cand.solution, task_id)
        ctx.record("exec_started", job=cand.cand_id, ts=result.raw.get("started"))
        ctx.record("exec_done", job=cand.cand_id, cand=cand.cand_id, ts=result.raw.get("ended"),
                   gpu_s=result.raw.get("gpu_s", 0.0), all_passed=result.correct,
                   sol_score=result.sol_score, sol_score_cal=result.calibrated_sol_score(),
                   scores=result.vector(), statuses=_statuses(result))
        if not result.correct:                              # feed the mistake back to future agents
            ctx.note_failure(cand.strategy, _dominant_failure(result), cand.cand_id)
            verdict = ctx.accept_candidate(cand.cand_id)
        else:
            # A candidate that PASSED and would improve the frontier gets re-verified
            # (10 more correctness rounds per re-run) to catch flaky/racy kernels that
            # pass once but fail the leaderboard's single run. Cheap: only frontier-
            # entering candidates pay, and we can't lean on the leaderboard (throttled).
            flaky_at = None
            if cfg.verify_runs > 1 and ctx.frontier.would_enter(tuple(result.vector())):
                flaky_at = await _reverify(ctx, executor, task_id, cand.cand_id,
                                           cand.solution, cfg.verify_runs)
            if flaky_at is not None:
                ctx.record("flaky", cand=cand.cand_id, attempt=flaky_at)
                ctx.note_failure(cand.strategy, "FLAKY_NONDETERMINISTIC", cand.cand_id)
                verdict = "flaky"                           # rejected: never enters the frontier
            else:
                verdict = ctx.accept_candidate(cand.cand_id)
        _persist(ctx, task_id, cand.cand_id, cand.solution, result, runs_dir=runs_dir,
                 problems_dir=problems_dir, family=family, name=name, strategy=cand.strategy,
                 agent=persp.agent, model=persp.model, parent=cand.parent,
                 verdict=verdict, trajectory=cand.trajectory)
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
        await knowledge.curate(ctx, op_key_of(task_id, problems_dir), name)
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
    max_concurrency: int = 0,
    shuffle: bool = False,
    reflect_first: bool = False,
    reflect_every_min: float = 0,
    reflect_model: str = "",
) -> None:
    families = families or {}
    names = names or {}

    # Cross-run reflection (the "Coach"): regenerate every problem's reflection.md
    # coach card from the accumulated journals BEFORE the fleet starts, so a restart
    # begins with each agent already knowing what's been tried / where it's stuck /
    # where the loss is. The deterministic detectors are free; when reflect_model is
    # set (e.g. claude-fable-5), a strong model ALSO reads the tried kernels' source
    # and adds a why-it's-stuck + one-lever diagnosis — but only for STUCK problems
    # whose state moved (deduped), so fable spend stays bounded. Best-effort: a
    # failure here (incl. fable out of credits) never aborts the run.
    async def _reflect(tag: str) -> None:
        try:
            from . import reflection
            refls = await asyncio.to_thread(reflection.reflect_all, runs_dir, ids, names=names)
            if reflect_model:
                from . import diagnose
                await diagnose.diagnose_stuck(runs_dir, refls, model=reflect_model, log=print)
        except Exception as exc:
            print(f"[reflect:{tag}] skipped: {exc!r}")
    if reflect_first:
        await _reflect("startup")
    # Cap how many problems are ACTIVE at once. Each active problem holds ≤1 agent
    # call in flight (its loop is sequential), so this bounds concurrent agent CLIs
    # + provider streams — the real limit is the laptop and the provider's rate
    # limit, NOT the single-flight GPU. 0 = unbounded (the old behaviour). Excess
    # problems queue and start as slots free (a resumable rolling window over huge
    # id ranges), so `solve 1-100 --max-concurrency 10` never spawns 100 CLIs.
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None

    async def guarded(t: int) -> None:
        try:
            if sem is not None:
                async with sem:
                    await solve_problem(t, executor, agents, cfg, runs_dir=runs_dir, seed=seed,
                                        seeds_fn=seeds_fn, check_fn=check_fn, knowledge=knowledge,
                                        problems_dir=problems_dir,
                                        family=families.get(t, ""), name=names.get(t, f"task-{t}"))
            else:
                await solve_problem(t, executor, agents, cfg, runs_dir=runs_dir, seed=seed,
                                    seeds_fn=seeds_fn, check_fn=check_fn, knowledge=knowledge,
                                    problems_dir=problems_dir,
                                    family=families.get(t, ""), name=names.get(t, f"task-{t}"))
        except Exception as exc:  # crash isolation: one problem's failure is journaled, not fatal
            journal_mod.Journal(Path(runs_dir) / str(t) / "journal.jsonl", t).append(
                "solver_error", error=repr(exc))

    # Launch order determines which problems fill the concurrency window first (and,
    # as slots free, which enter next). Default = exemplar-first. `shuffle` randomizes
    # it (seeded → reproducible) so a --max-concurrency window is a RANDOM sample of a
    # big id range rather than always the lowest ids — fairer coverage, and on a resume
    # it stops the already-strong low ids from starving the underworked high ids.
    order = exemplar_first(ids, families)
    if shuffle:
        random.Random(seed).shuffle(order)

    # Optional periodic refresh: rebuild coach cards every N minutes so long runs
    # keep reflecting on fresh results, not just the startup snapshot.
    async def _refresher() -> None:
        while True:
            await asyncio.sleep(max(30.0, reflect_every_min * 60))
            await _reflect("periodic")
    refresher = asyncio.create_task(_refresher()) if reflect_every_min > 0 else None
    try:
        await asyncio.gather(*(guarded(t) for t in order))
    finally:
        if refresher is not None:
            refresher.cancel()
