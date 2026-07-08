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
import contextlib
import datetime
import json
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
    Solution (a CLI agent that wrote no kernel)."""
    if solution.get("__invalid__"):
        return False, ["stub: marked invalid"]
    if "sources" in solution and not solution.get("sources"):
        return False, ["no source files produced"]
    return True, []


def default_check_fn(problems_dir: str | Path) -> CheckFn:
    """The real pre-GPU gate: `_default_check`'s cheap markers, THEN the static
    schema/DPS-signature/reward-hack checker (`solver.bench.check`) against the
    exact JSON the harness would receive. Pure Python, no GPU — catches malformed
    solutions (wrong entry signature, mixed C++/Python sources, reward-hack
    patterns) before a single GPU eval is spent. Fails OPEN (never blocks a run)
    if the check itself can't run, e.g. a test solution with no real `sources`."""
    from ..bench.check import check_solution
    from .harness import solution_to_harness_json
    problems_dir = Path(problems_dir)

    def check(solution: dict, task_id: int) -> tuple[bool, list[str]]:
        ok, errs = _default_check(solution, task_id)
        if not ok:
            return ok, errs
        try:
            harness_sol = solution_to_harness_json(solution, task_id, problems_dir)
            defn_path = problems_dir / str(task_id) / "definition.json"
            definition = json.loads(defn_path.read_text()) if defn_path.is_file() else None
        except Exception:
            return True, []                            # can't build a real check → don't block
        report = check_solution(harness_sol, definition)
        return report.ok, report.errors

    return check


def _statuses(result: EvalResult) -> list[str]:
    return [("PASSED" if w.correct else (w.error or "FAILED")) for w in result.per_workload]


def _dominant_failure(result: EvalResult) -> str:
    """The most common failure kind across a candidate's workloads (for feedback)."""
    from collections import Counter
    if (result.asi or {}).get("solution_status"):
        return result.asi["solution_status"]                  # COMPILE_ERROR / REWARD_HACK
    errs = [w.error for w in result.per_workload if not w.correct and w.error]
    return Counter(errs).most_common(1)[0][0] if errs else "INCORRECT"


def _fmt_idxs(idxs: list[int]) -> str:
    """Compress a workload-index list to ranges: [0,1,2,3,7] -> '0-3,7'."""
    if not idxs:
        return "-"
    idxs = sorted(set(idxs))
    out, start, prev = [], idxs[0], idxs[0]
    for i in idxs[1:] + [None]:
        if i == prev + 1:
            prev = i
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = i
    return ",".join(out)


def _failure_detail(result: EvalResult) -> str:
    """An ACTIONABLE per-workload failure breakdown fed back to the agent so it can
    fix an incorrect kernel: which workloads failed, with what error, and which
    passed. E.g. '12/15 workloads FAILED — RUNTIME_ERROR on #0-11; TOLERANCE on
    #12,13. PASSED: #14'. Empty string if there's no per-workload detail."""
    from collections import defaultdict
    pw = result.per_workload or []
    if not pw:
        return ""
    failed = [w for w in pw if not w.correct]
    if not failed:
        return ""
    passed = [w.index for w in pw if w.correct]
    by_err: dict[str, list[int]] = defaultdict(list)
    for w in failed:
        by_err[w.error or "INCORRECT_OUTPUT"].append(w.index)
    parts = [f"{err} on #{_fmt_idxs(idxs)}"
             for err, idxs in sorted(by_err.items(), key=lambda kv: -len(kv[1]))]
    tail = f". PASSED: #{_fmt_idxs(passed)}" if passed else " (ALL workloads failed)"
    return f"{len(failed)}/{len(pw)} workloads FAILED — " + "; ".join(parts) + tail


def pick_reviewer(perspectives: list[Perspective], writer: Perspective, key: str) -> Perspective:
    """Pick a reviewer DIFFERENT from the writer — cross-model review catches blind
    spots a model has about its own code. Deterministic on `key` (e.g. a per-round
    seed) so a resumed run reproduces the same reviewer for the same round; falls
    back to the writer itself only when it's the sole perspective in the pool."""
    others = [p for p in perspectives if p != writer]
    pool = others or [writer]
    return random.Random(key).choice(pool)


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
    on_phase: Callable[[str | None, dict | None], None] | None = None,
) -> RunContext:
    ctx = RunContext.load(task_id, cfg, runs_dir, seed=seed)
    ctx.reopen_if_capped()             # a cap-terminated run continues if the caps now allow it
    if cfg.time_limit_s:               # wall-clock budget for THIS run (resume gets a fresh one)
        ctx.deadline = time.monotonic() + cfg.time_limit_s
    seeds_fn = seeds_fn or _default_seeds
    check_fn = check_fn or default_check_fn(problems_dir)

    @contextlib.contextmanager
    def _phase(phase_name: str, persp=None, cand_id: str | None = None):
        """Publish 'what is this problem's agent doing RIGHT NOW' (design / plan /
        review / repair) to run_fleet's live status — purely informational for the
        dashboard (see runs/_active.json's task_phase), never read back by the
        engine itself, so it can't affect replay/resume determinism."""
        if on_phase is not None:
            on_phase(phase_name, {"agent": getattr(persp, "agent", None),
                                  "model": getattr(persp, "model", None), "cand": cand_id,
                                  "started": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        try:
            yield
        finally:
            if on_phase is not None:
                on_phase(None, None)

    # ---- bootstrap (consumes GPU evals; committed by the `bootstrapped` marker) ----
    if ctx.fresh():
        ctx.record("run_started", agent=str(cfg.design_model), name=name, family=family)
        _persist_reference(runs_dir, task_id, problems_dir)   # ground truth beside the frontier
        dtok: dict = {}
        try:
            with _phase("design", cfg.design_model):
                design, dtok = await agents[cfg.design_model].design(task_id)
        except Exception as exc:                              # a slow/failed design must not abort
            design = ""
            ctx.record("design_error", error=repr(exc)[:200])
        ctx.record("design_done", text=design, dur_s=0.0, model=str(cfg.design_model),
                   tok_in=(dtok.get("in") or 0), tok_out=(dtok.get("out") or 0),
                   tok_cached=(dtok.get("cached") or 0), cost_usd=(dtok.get("cost_usd") or 0.0))
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
            with _phase("plan", persp, getattr(parent, "cand_id", None)):
                cand = await agent.plan(parent, ctx)
        except Exception as exc:
            # agent timed out / wrote no kernel: skip THIS iteration, keep the
            # problem's frontier, advance to the next perspective. Never abort.
            ctx.record("plan_error", agent=persp.agent, model=persp.model, error=repr(exc)[:200])
            ctx.record("iter", n=ctx.iters, outcome="agent_error")
            continue

        is_dup_hash = cand.cand_id in ctx.seen
        # A NO-OP is a stronger, more specific signal than a generic duplicate: the
        # agent's output hashes EXACTLY to the parent it was given to improve on —
        # it changed nothing at all (no kernel edit, no strategy.txt). This is the
        # dominant real-world failure mode (measured: ~97% of iterations on a
        # stuck problem), and every model does it at similar rates once a strong
        # candidate dominates the frontier — it's usually the agent correctly
        # judging there's nothing left to improve, just with no clean way to say
        # so. Left undetected, it silently burns a paid API call per iteration
        # forever; detected, N agents agreeing is real ceiling evidence.
        is_noop = parent is not None and getattr(parent, "cand_id", None) == cand.cand_id
        tok = cand.tokens or {}
        ctx.record("plan_done", cand=cand.cand_id, parent=cand.parent, agent=persp.agent,
                   model=persp.model, strategy=cand.strategy, solution=cand.solution,
                   dur_s=0.0, tok_in=(tok.get("in") or 0), tok_out=(tok.get("out") or 0),
                   tok_cached=(tok.get("cached") or 0), cost_usd=(tok.get("cost_usd") or 0.0),
                   no_op=is_noop, trajectory=cand.trajectory, handoff=cand.handoff,
                   context_read=cand.context_read or [])

        ok, _errs = check_fn(cand.solution, task_id)
        ctx.record("check", cand=cand.cand_id, ok=ok)
        if not ok:
            ctx.noop_streak = 0                  # a real (if invalid) attempt — not a no-op
            ctx.record("iter", n=ctx.iters, outcome="rejected")
            continue
        if is_noop:
            ctx.noop_streak += 1
            ctx.record("novelty", cand=cand.cand_id, verdict="no_op")
            ctx.record("iter", n=ctx.iters, outcome="no_op")
            if cfg.ceiling_consensus and ctx.noop_streak >= cfg.ceiling_consensus:
                ctx.record("terminated", reason="ceiling_consensus")
                break
            continue
        ctx.noop_streak = 0                      # a genuinely new (or re-derived) candidate
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

        # ---- pre-GPU code review: an INDEPENDENT model (never the writer) reads the
        # kernel against reference.py + workloads.md and judges ship/revise BEFORE a
        # GPU eval is spent — the single-flight GPU makes a bad candidate expensive
        # (a hang burns the full timeout for the whole fleet). On "revise", the SAME
        # writer gets the critique back for a repair turn and the cycle repeats —
        # effectively unlimited, bounded only by review_max_rounds as a safety valve
        # so one stubborn candidate can't stall the problem forever (it ships as-is
        # if the cap is hit — never worse than review being off).
        if cfg.review_enabled:
            round_n = 0
            while True:
                review_persp = pick_reviewer(cfg.perspectives, persp,
                                             f"{seed}:{task_id}:{cand.cand_id}:{round_n}")
                try:
                    with _phase("review", review_persp, cand.cand_id):
                        verdict = await agents[review_persp].review(cand, ctx)
                except Exception as exc:
                    ctx.record("review_error", cand=cand.cand_id, reviewer=str(review_persp),
                              error=repr(exc)[:200])
                    break                          # review itself failed → fail open, ship as-is
                vtok = verdict.tokens or {}
                ctx.record("review", cand=cand.cand_id, reviewer=str(review_persp),
                          verdict=verdict.verdict, issues=verdict.issues, round=round_n,
                          cost_usd=verdict.cost_usd, tok_in=(vtok.get("in") or 0),
                          tok_out=(vtok.get("out") or 0), tok_cached=(vtok.get("cached") or 0),
                          context_read=verdict.context_read or [])
                if verdict.ship or round_n >= cfg.review_max_rounds:
                    break
                round_n += 1
                ctx.review_critique = verdict.issues_text()
                try:
                    # repair() RESUMES the writer's own CLI session in place (same
                    # model memory) instead of cold-starting a fresh plan() call —
                    # falls back to plan() itself when resume isn't available.
                    with _phase("repair", persp, cand.cand_id):
                        repaired = await agents[persp].repair(cand, ctx.review_critique, ctx)
                except Exception as exc:
                    ctx.record("plan_error", agent=persp.agent, model=persp.model,
                              error=repr(exc)[:200])
                    break                          # repair failed → ship the last GOOD candidate
                finally:
                    ctx.review_critique = None
                rtok = repaired.tokens or {}
                ctx.record("plan_done", cand=repaired.cand_id, parent=repaired.parent,
                          agent=persp.agent, model=persp.model, strategy=repaired.strategy,
                          solution=repaired.solution, dur_s=0.0, tok_in=(rtok.get("in") or 0),
                          tok_out=(rtok.get("out") or 0), tok_cached=(rtok.get("cached") or 0),
                          cost_usd=(rtok.get("cost_usd") or 0.0),
                          repair=True, trajectory=repaired.trajectory, handoff=repaired.handoff,
                          context_read=repaired.context_read or [])
                rok, _rerrs = check_fn(repaired.solution, task_id)
                ctx.record("check", cand=repaired.cand_id, ok=rok)
                if not rok:
                    break                          # repair broke the schema → ship the pre-repair cand
                if repaired.cand_id == cand.cand_id:
                    # the repair changed nothing (the prompt explicitly forbids this,
                    # but a model can still no-op it) — another review pass would just
                    # re-judge identical bytes and risk a non-deterministic flip-flop
                    # (confirmed live: same reviewer, same bytes, revise then ship
                    # seconds apart). Stop here and ship as-is — the same fallback
                    # guarantee as hitting review_max_rounds.
                    ctx.record("iter", n=ctx.iters, outcome="repair_no_op")
                    break
                cand = repaired

        ctx.record("exec_enqueued", job=cand.cand_id, cand=cand.cand_id)
        result = await executor.evaluate(cand.solution, task_id)
        ctx.record("exec_started", job=cand.cand_id, ts=result.raw.get("started"))
        ctx.record("exec_done", job=cand.cand_id, cand=cand.cand_id, ts=result.raw.get("ended"),
                   gpu_s=result.raw.get("gpu_s", 0.0), all_passed=result.correct,
                   sol_score=result.sol_score, sol_score_cal=result.calibrated_sol_score(),
                   scores=result.vector(), statuses=_statuses(result))
        if not result.correct:                              # feed the mistake back to future agents
            ctx.note_failure(cand.strategy, _dominant_failure(result), cand.cand_id,
                             detail=_failure_detail(result))
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
    reflect_model: str | list[str] = "",
) -> None:
    families = families or {}
    names = names or {}
    # A pool rotates across the stuck list in diagnose_stuck (see there); the
    # dashboard just needs a readable label, not the raw list.
    reflect_label = (reflect_model if isinstance(reflect_model, str)
                     else "+".join(m.rsplit("/", 1)[-1] for m in reflect_model))

    # Cross-run reflection (the "Coach") is wired below, AFTER the working-set
    # publisher. Critically it must NEVER block the GPU: the cheap deterministic cards
    # are written fast up front, but the expensive fable diagnosis runs in the
    # BACKGROUND so problems start being worked immediately (not after ~15 min of
    # fable). Live status (agents running / reflecting) is published to _active.json.

    # Cap how many problems are ACTIVE at once. Each active problem holds ≤1 agent
    # call in flight (its loop is sequential), so this bounds concurrent agent CLIs
    # + provider streams — the real limit is the laptop and the provider's rate
    # limit, NOT the single-flight GPU. 0 = unbounded (the old behaviour). Excess
    # problems queue and start as slots free (a resumable rolling window over huge
    # id ranges), so `solve 1-100 --max-concurrency 10` never spawns 100 CLIs.
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None

    # Working set: which problems currently hold a concurrency slot (are actively
    # being iterated) vs queued. Published to runs/_active.json (with a heartbeat ts)
    # so the dashboard can show running / waiting / pending instead of "running" for
    # every not-yet-terminated problem — the single-flight GPU means a slot-holder
    # can sit idle for a while, so recency alone can't tell active from queued.
    active: set[int] = set()
    active_path = Path(runs_dir) / "_active.json"
    status = {"phase": "starting", "reflect": None}   # live fleet status for the dashboard
    # Per-task "what is the agent doing right now" (design/plan/review/repair, who,
    # since when) — fed by solve_problem's on_phase hook (see there). None once a
    # task has no call in flight (between rounds, or queued/done). This answers
    # "is it healthy or just slow?" without guessing from process list/journal age.
    task_phase: dict[int, dict | None] = {}

    def _write_active() -> None:
        try:
            active_path.write_text(json.dumps({
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "active": sorted(active), "cap": max_concurrency,
                "phase": status["phase"], "reflect": status["reflect"],
                "task_phase": {str(t): v for t, v in task_phase.items() if v is not None}}))
        except OSError:
            pass

    async def _diagnose_bg(tag: str) -> None:
        """The Coach: deterministic cards (fast) + fable diagnosis on STUCK problems.
        Runs in its own task so it NEVER blocks the fleet; publishes reflect progress
        to _active.json so the dashboard shows 'reflecting X/Y'."""
        try:
            from . import reflection
            refls = await asyncio.to_thread(reflection.reflect_all, runs_dir, ids, names=names)
            if not reflect_model:
                return
            from . import diagnose as diag
            stuck = [r for r in refls.values() if r.status in diag.STUCK]
            status["reflect"] = {"done": 0, "total": len(stuck), "model": reflect_label} if stuck else None
            _write_active()

            def _prog(done: int) -> None:
                status["reflect"] = ({"done": done, "total": len(stuck), "model": reflect_label}
                                     if done < len(stuck) else None)
                _write_active()
            await diag.diagnose_stuck(runs_dir, refls, model=reflect_model,
                                      progress=_prog, log=print)
        except Exception as exc:
            print(f"[reflect:{tag}] skipped: {exc!r}")
        finally:
            status["reflect"] = None
            _write_active()

    async def guarded(t: int) -> None:
        def _on_phase(phase_name: str | None, info: dict | None) -> None:
            task_phase[t] = {"phase": phase_name, **info} if phase_name else None
            _write_active()

        async def _do() -> None:
            active.add(t)
            _write_active()
            try:
                await solve_problem(t, executor, agents, cfg, runs_dir=runs_dir, seed=seed,
                                    seeds_fn=seeds_fn, check_fn=check_fn, knowledge=knowledge,
                                    problems_dir=problems_dir,
                                    family=families.get(t, ""), name=names.get(t, f"task-{t}"),
                                    on_phase=_on_phase)
            finally:
                active.discard(t)
                task_phase.pop(t, None)
                _write_active()
        try:
            if sem is not None:
                async with sem:
                    await _do()
            else:
                await _do()
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

    # Write the cheap deterministic cards up front (fast) so the first agent plans
    # already have them — but do NOT wait on fable here.
    if reflect_first:
        try:
            from . import reflection
            await asyncio.to_thread(reflection.reflect_all, runs_dir, ids, names=names)
        except Exception as exc:
            print(f"[reflect:startup] cards skipped: {exc!r}")

    # Optional periodic refresh: rebuild coach cards + re-diagnose every N minutes so
    # long runs keep reflecting on fresh results (in its own task — never blocks).
    async def _refresher() -> None:
        while True:
            await asyncio.sleep(max(30.0, reflect_every_min * 60))
            await _diagnose_bg("periodic")
    # Heartbeat: keep _active.json's ts fresh (the set changes only when a problem
    # starts/finishes, but the dashboard needs a recent ts to trust it as live).
    async def _heartbeat() -> None:
        while True:
            _write_active()
            await asyncio.sleep(25)

    status["phase"] = "running"
    _write_active()
    heartbeat = asyncio.create_task(_heartbeat())
    refresher = asyncio.create_task(_refresher()) if reflect_every_min > 0 else None
    # startup fable diagnosis runs in the BACKGROUND, concurrently with the fleet
    startup_diag = asyncio.create_task(_diagnose_bg("startup")) if reflect_first else None
    try:
        await asyncio.gather(*(guarded(t) for t in order))
    finally:
        if refresher is not None:
            refresher.cancel()
        if startup_diag is not None:
            startup_diag.cancel()
        heartbeat.cancel()
        status["phase"] = "done"
        active.clear()
        _write_active()                # publish an empty, fresh set → all idle on exit
