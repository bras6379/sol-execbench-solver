"""Acceptance tests for the engine (docs/orchestration.md §12), stub-powered.

Determinism is the strategy: StubExecutor + StubAgent make every run
reproducible, so routing / frontier / budget / plateau / escalation / resume
invariants are exactly assertable — no GPU, no model.
"""

from __future__ import annotations

import asyncio
import random

from solver import journal as journal_mod
from solver.engine import (
    Config,
    Frontier,
    Member,
    Perspective,
    StubExecutor,
    Tier,
    solve_problem,
    stub_agents,
    run_fleet,
)
from solver.engine.frontier import _dominates

run = asyncio.run


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def cfg(tiers, **kw):
    return Config(tiers=tiers, **kw)


def one_tier(**kw):
    return cfg([Tier("cheap", [Perspective("claude", "haiku")])], **kw)


def two_tier(**kw):
    return cfg([Tier("cheap", [Perspective("claude", "haiku")]),
                Tier("strong", [Perspective("claude", "opus")])], **kw)


def scripted(scripts):
    """planner popping the next spec per perspective (its own sequential counter)."""
    counters: dict[str, int] = {}

    def planner(persp, parent, ctx):
        k = str(persp)
        i = counters.get(k, 0)
        counters[k] = i + 1
        seq = scripts[k]
        return seq[min(i, len(seq) - 1)]
    return planner


def state(ctx):
    """State fingerprint used for resume equivalence (ts-independent)."""
    return (
        sorted(tuple(round(x, 6) for x in m.vector) for m in ctx.frontier.members),
        round(ctx.frontier.best_score(), 6),
        ctx.iters,
        ctx.evals,
        ctx.tier_idx,
        ctx.terminated_reason,
    )


def events(path):
    return [e.get("ev") for e in journal_mod.read(path)]


# --------------------------------------------------------------------------- #
# test 3 — frontier correctness (property test vs brute force, ε = 0)
# --------------------------------------------------------------------------- #
def test_frontier_matches_brute_force():
    rng = random.Random(0)
    for _ in range(200):
        vecs = [tuple(rng.randint(0, 3) for _ in range(4)) for _ in range(rng.randint(1, 12))]
        fr = Frontier(epsilon=0.0)
        for i, v in enumerate(vecs):
            fr.accept(Member(cand_id=f"c{i}", vector=v, all_passed=True))
        got = sorted(m.vector for m in fr.members)
        brute = sorted(v for i, v in enumerate(vecs)
                       if not any(_dominates(u, v, 0.0) for j, u in enumerate(vecs) if j != i))
        assert got == brute


def test_frontier_keeps_specialists():
    fr = Frontier(epsilon=0.0)
    fr.accept(Member("a", (1.0, 0.0), True))    # wins shape 0
    fr.accept(Member("b", (0.0, 1.0), True))    # wins shape 1
    fr.accept(Member("c", (0.4, 0.4), True))    # dominated by neither
    assert len(fr.members) == 3
    fr.accept(Member("d", (1.0, 1.0), True))    # dominates all
    assert [m.cand_id for m in fr.members] == ["d"]


# --------------------------------------------------------------------------- #
# test 2 — budget exactness
# --------------------------------------------------------------------------- #
def test_budget_iterations_cap(tmp_path):
    # constant-improving unique candidates never plateau → cap must stop it.
    c = one_tier(max_iterations=5, plateau_cycles=999, escalate_ceiling=1.1)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}))
    ctx = run(solve_problem(1, StubExecutor(), agents, c, runs_dir=tmp_path))
    assert ctx.iters == 5
    assert ctx.evals == 6                      # 1 seed + 5 loop evals
    assert ctx.done_reason() == "budget:iterations"


def test_budget_evals_cap(tmp_path):
    c = one_tier(max_iterations=999, max_gpu_evals=4, plateau_cycles=999, escalate_ceiling=1.1)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}))
    ctx = run(solve_problem(1, StubExecutor(), agents, c, runs_dir=tmp_path))
    assert ctx.evals == 4                      # 1 seed + 3 loop evals, then cap
    assert ctx.done_reason() == "budget:evals"


def test_cap_terminated_reopens_but_converged_does_not(tmp_path):
    # a budget-capped run continues when the eval cap is raised...
    d = tmp_path / "capped"
    c1 = one_tier(max_iterations=99, max_gpu_evals=4, plateau_cycles=999, escalate_ceiling=1.1)
    ctx1 = run(solve_problem(1, StubExecutor(),
                             stub_agents(c1.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]})),
                             c1, runs_dir=d))
    assert ctx1.done_reason() == "budget:evals" and ctx1.evals == 4
    c2 = one_tier(max_iterations=99, max_gpu_evals=8, plateau_cycles=999, escalate_ceiling=1.1)
    ctx2 = run(solve_problem(1, StubExecutor(),
                             stub_agents(c2.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]})),
                             c2, runs_dir=d))
    assert ctx2.evals == 8                       # reopened and continued 4 → 8
    assert "reopened" in events(ctx2.path)

    # ...but a converged run stays done even with headroom in the caps.
    d2 = tmp_path / "converged"
    script = {"claude:haiku": [{"scores": [0.95]}, {"scores": [0.9]}, {"scores": [0.9]}, {"scores": [0.9]}],
              "claude:opus": [{"scores": [0.99]}]}
    cc = two_tier(plateau_cycles=2, escalate_ceiling=0.9, max_iterations=99, max_gpu_evals=99)
    x1 = run(solve_problem(5, StubExecutor(), stub_agents(cc.perspectives, scripted(script)), cc, runs_dir=d2))
    assert x1.terminated_reason == "converged:ceiling"
    x2 = run(solve_problem(5, StubExecutor(), stub_agents(cc.perspectives, scripted(script)), cc, runs_dir=d2))
    assert x2.evals == x1.evals                  # did NOT reopen
    assert "reopened" not in events(x2.path)


# --------------------------------------------------------------------------- #
# test 5 — gates never touch the executor
# --------------------------------------------------------------------------- #
def test_gates_skip_the_gpu(tmp_path):
    script = {"claude:haiku": [
        {"scores": [0.6], "duplicate": True},   # i0: novel → eval, becomes the frontier's parent
        {"scores": [0.6], "duplicate": True},   # i1: hashes IDENTICAL to its own parent (i0) → no_op
        {"scores": [0.7], "invalid": True},     # i2: check fail → reject, no eval
        {"scores": [0.65]},                     # i3: novel → eval
    ]}
    c = one_tier(max_iterations=4, plateau_cycles=999, escalate_ceiling=1.1, ceiling_consensus=0)
    ex = StubExecutor()
    agents = stub_agents(c.perspectives, scripted(script))
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))
    assert ex.calls == 3                        # 1 seed + i0 + i3 (no_op & invalid skipped)
    assert ctx.evals == 3
    evs = events(ctx.path)
    assert evs.count("iter") == 4               # every plan committed one iteration
    # one rejected, one no_op outcome recorded (i1 hash-matches its own parent i0 exactly —
    # a stronger, more specific signal than a generic "duplicate of some unrelated candidate")
    outcomes = [e for e in journal_mod.read(ctx.path) if e["ev"] == "iter"]
    kinds = [o["outcome"] for o in outcomes]
    assert "rejected" in kinds and "no_op" in kinds


def test_no_novelty_prefilter_measures_candidates(tmp_path):
    # There is no LLM novelty pre-filter: every non-exact-duplicate is MEASURED and
    # the ε-Pareto frontier decides on real performance (exact-hash dups still skip).
    c = one_tier(max_iterations=3, plateau_cycles=999, escalate_ceiling=1.1)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}))
    ex = StubExecutor()
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))
    assert ex.calls > 1                          # candidates are measured, not bounced
    assert ctx.frontier.best_score() >= 0.6      # a 0.6 kernel entered over the 0.5 seed


# --------------------------------------------------------------------------- #
# re-verification — a flaky (racy) kernel that passes once but fails a fresh
# re-run must NOT enter the frontier when --verify-runs > 1.
# --------------------------------------------------------------------------- #
def test_verify_runs_rejects_flaky_candidate(tmp_path):
    script = {"claude:haiku": [
        {"scores": [0.8], "flaky_on": [1]},   # passes attempt 0, fails fresh re-run 1 → flaky
        {"scores": [0.7]},                     # clean → passes every re-run → accepted
    ]}
    c = one_tier(max_iterations=2, plateau_cycles=999, escalate_ceiling=1.1, verify_runs=3)
    ex = StubExecutor()
    agents = stub_agents(c.perspectives, scripted(script))
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))
    # the flaky 0.8 is rejected; only the clean 0.7 enters → best is 0.7, never 0.8
    assert round(ctx.frontier.best_score(), 4) == 0.7
    assert all(round(m.mean, 4) != 0.8 for m in ctx.frontier.members)
    evs = journal_mod.read(ctx.path)
    assert any(e["ev"] == "flaky" for e in evs)
    assert "flaky" in [e["outcome"] for e in evs if e["ev"] == "iter"]
    # cost: seed(1) + flaky[primary+1 verify=2, stops on first disagree] + clean[primary+2 verify=3]
    assert ex.calls == 6 and ctx.evals == 6


def test_verify_runs_off_by_default_accepts_flaky(tmp_path):
    # verify_runs=1 (default): no re-verification, so a flaky kernel slips in — this
    # is the exact gap --verify-runs closes; it stays opt-in to keep GPU cost down.
    script = {"claude:haiku": [{"scores": [0.8], "flaky_on": [1]}]}
    c = one_tier(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1)
    ex = StubExecutor()
    agents = stub_agents(c.perspectives, scripted(script))
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))
    assert round(ctx.frontier.best_score(), 4) == 0.8
    assert not any(e["ev"] == "flaky" for e in journal_mod.read(ctx.path))


def test_handoff_accumulates_into_playbook(tmp_path):
    # Each ACCEPTED candidate's handoff (a reserve play it flagged but didn't ship)
    # is banked into ctx.playbook — deduped by text, skipping dominated candidates
    # and empty handoffs — so the next agent inherits reserve plays, not silence.
    script = {"claude:haiku": [
        {"scores": [0.60], "handoff": "A radix-sort segmented reduction"},  # enters → bank A
        {"scores": [0.55], "handoff": "B dominated idea"},                  # dominated → NOT banked
        {"scores": [0.65], "handoff": "C warp-specialized epilogue"},       # enters → bank C
        {"scores": [0.70], "handoff": "A radix-sort segmented reduction"},  # enters but DUP → not re-banked
        {"scores": [0.72]},                                                 # enters, no handoff → nothing
    ]}
    c = one_tier(max_iterations=5, plateau_cycles=999, escalate_ceiling=1.1)
    ctx = run(solve_problem(1, StubExecutor(), stub_agents(c.perspectives, scripted(script)),
                            c, runs_dir=tmp_path))
    assert [p["handoff"] for p in ctx.playbook] == [
        "A radix-sort segmented reduction", "C warp-specialized epilogue"]
    # survives resume: the playbook is journal-derived, so a replay rebuilds it
    ctx2 = run(solve_problem(1, StubExecutor(), stub_agents(c.perspectives, scripted(script)),
                             c, runs_dir=tmp_path))
    assert [p["handoff"] for p in ctx2.playbook] == [p["handoff"] for p in ctx.playbook]


def test_verify_runs_resume_is_identical(tmp_path):
    # a run with re-verification replays bit-identically (verify_started/_done/flaky
    # are journaled with no hidden state), so resume reconstructs the same frontier.
    script = {"claude:haiku": [{"scores": [0.8], "flaky_on": [1]}, {"scores": [0.7]}]}
    c = one_tier(max_iterations=2, plateau_cycles=999, escalate_ceiling=1.1, verify_runs=3)
    x1 = run(solve_problem(1, StubExecutor(), stub_agents(c.perspectives, scripted(script)),
                           c, runs_dir=tmp_path))
    x2 = run(solve_problem(1, StubExecutor(), stub_agents(c.perspectives, scripted(script)),
                           c, runs_dir=tmp_path))            # resume: fully replayed, no new work
    assert state(x1) == state(x2)


# --------------------------------------------------------------------------- #
# test 4 — plateau → escalation → termination (+ headroom gate, round-robin)
# --------------------------------------------------------------------------- #
def test_plateau_escalates_with_headroom(tmp_path):
    # cheap tier tops out at 0.6 (< ceiling) then stalls → escalate; strong tier
    # reaches 0.95 (≥ ceiling) then stalls → terminate.
    script = {
        "claude:haiku": [{"scores": [0.6]}, {"scores": [0.55]}, {"scores": [0.55]}, {"scores": [0.55]}],
        "claude:opus":  [{"scores": [0.95]}, {"scores": [0.9]}, {"scores": [0.9]}, {"scores": [0.9]}],
    }
    c = two_tier(plateau_cycles=2, escalate_ceiling=0.9, max_iterations=99)
    agents = stub_agents(c.perspectives, scripted(script))
    ctx = run(solve_problem(1, StubExecutor(), agents, c, runs_dir=tmp_path))
    assert ctx.terminated_reason == "converged:ceiling"
    assert ctx.tier_idx == 1                     # climbed one tier
    assert round(ctx.frontier.best_score(), 4) == 0.95
    evs = journal_mod.read(ctx.path)
    esc = [e for e in evs if e["ev"] == "agent_changed"]
    assert len(esc) == 1 and esc[0]["trigger"] == "escalation" and esc[0]["tier"] == 1


def test_headroom_gate_no_escalate(tmp_path):
    # cheap tier already at/above ceiling when it stalls → terminate, don't climb.
    script = {
        "claude:haiku": [{"scores": [0.95]}, {"scores": [0.9]}, {"scores": [0.9]}, {"scores": [0.9]}],
        "claude:opus":  [{"scores": [0.99]}],
    }
    c = two_tier(plateau_cycles=2, escalate_ceiling=0.9, max_iterations=99)
    agents = stub_agents(c.perspectives, scripted(script))
    ctx = run(solve_problem(1, StubExecutor(), agents, c, runs_dir=tmp_path))
    assert ctx.terminated_reason == "converged:ceiling"
    assert ctx.tier_idx == 0                     # never escalated
    assert not any(e["ev"] == "agent_changed" for e in journal_mod.read(ctx.path))


def test_round_robin_covers_pool_before_escalation(tmp_path):
    # a 2-model cheap tier: both models must produce a candidate before escalation.
    tier0 = Tier("cheap", [Perspective("claude", "haiku"), Perspective("openai", "gpt")])
    tier1 = Tier("strong", [Perspective("claude", "opus")])
    c = cfg([tier0, tier1], plateau_cycles=2, escalate_ceiling=0.9, max_iterations=99)
    # everyone stalls at a dominated score so the tier plateaus.
    script = {
        "claude:haiku": [{"scores": [0.6]}] + [{"scores": [0.4]}] * 8,
        "openai:gpt":   [{"scores": [0.4]}] * 8,
        "claude:opus":  [{"scores": [0.95]}] + [{"scores": [0.9]}] * 8,
    }
    agents = stub_agents(c.perspectives, scripted(script))
    ctx = run(solve_problem(1, StubExecutor(), agents, c, runs_dir=tmp_path))
    plans = [e for e in journal_mod.read(ctx.path) if e["ev"] == "plan_done"]
    before_esc = []
    for e in journal_mod.read(ctx.path):
        if e["ev"] == "agent_changed":
            break
        if e["ev"] == "plan_done":
            before_esc.append(e["model"])
    assert "haiku" in before_esc and "gpt" in before_esc     # both cheap models tried
    assert ctx.tier_idx == 1


# --------------------------------------------------------------------------- #
# fleet load-spreading + graceful downgrade (mixing cheap + premium models)
# --------------------------------------------------------------------------- #
def test_pool_order_is_per_problem_shuffle(tmp_path):
    # one pool, all models: the rotation is a per-problem SHUFFLE — deterministic
    # (replay-safe) but different across problems, so the fleet doesn't hit one
    # provider (Claude/GPT) in lockstep. Each order is still a full permutation.
    from solver.engine.context import RunContext
    pool = [Perspective(a, "m") for a in ("claude", "gpt", "glm", "deepseek", "kimi")]
    c = cfg([Tier("all", pool)])
    twice = [tuple(str(p) for p in RunContext(1, c, tmp_path, seed=0)._pool_order()) for _ in range(2)]
    assert twice[0] == twice[1]                               # deterministic per problem (replay-safe)
    orders = {tuple(str(p) for p in RunContext(t, c, tmp_path, seed=0)._pool_order()) for t in range(6)}
    assert len(orders) > 1                                    # desynchronized across the fleet
    assert all(sorted(o) == sorted(str(p) for p in pool) for o in orders)   # every model once (coverage)


def test_circuit_breaker_disables_dead_agent_and_downgrades(tmp_path):
    # a dead agent (out of credits → always errors) is circuit-broken after
    # agent_fail_limit consecutive failures; the healthy model carries the run.
    from solver.engine.agent import StubAgent
    dead, good = Perspective("dead", "x"), Perspective("good", "y")
    c = cfg([Tier("all", [dead, good])], max_iterations=16, plateau_cycles=999,
            escalate_ceiling=1.1, agent_fail_limit=3)
    plan = lambda persp, parent, ctx: {"scores": [0.6]}
    agents = {dead: StubAgent(dead, plan, raise_on=lambda *a: True),
              good: StubAgent(good, plan)}
    ctx = run(solve_problem(1, StubExecutor(), agents, c, runs_dir=tmp_path))
    assert "dead:x" in ctx.disabled and "good:y" not in ctx.disabled
    assert ctx.frontier.best_score() >= 0.6                   # progressed via the healthy agent
    assert ctx.terminated_reason != "agents-unavailable"


def test_max_concurrency_caps_active_problems(tmp_path):
    # --max-concurrency bounds how many problems run at once (each holds <=1 agent
    # call in flight), so a big id range doesn't spawn a CLI per problem all at once.
    from solver.engine.agent import Candidate, solution_hash

    class CountingAgent:                       # tracks peak concurrent design() calls
        def __init__(self, persp, tracker):
            self.perspective = persp
            self._t = tracker
        async def design(self, task_id):
            self._t["n"] += 1
            self._t["peak"] = max(self._t["peak"], self._t["n"])
            await asyncio.sleep(0.02)
            self._t["n"] -= 1
            return "d"
        async def plan(self, parent, ctx):
            sol = {"__eval__": {"scores": [0.6]}, "__uid__": f"{self.perspective}:{getattr(ctx, 'iters', 0)}"}
            return Candidate(cand_id=solution_hash(sol)[:12], solution=sol, parent=None,
                             agent=self.perspective.agent, model=self.perspective.model, strategy="s")

    persp = Perspective("x", "1")
    c = cfg([Tier("t", [persp])], max_iterations=2, plateau_cycles=999, escalate_ceiling=1.1)
    capped = {"n": 0, "peak": 0}
    run(run_fleet([1, 2, 3, 4], StubExecutor(delay=0.003), {persp: CountingAgent(persp, capped)},
                  c, runs_dir=tmp_path / "cap", max_concurrency=1))
    assert capped["peak"] == 1                 # cap=1 ⇒ strictly serial, never two agents at once
    unbounded = {"n": 0, "peak": 0}
    run(run_fleet([1, 2, 3, 4], StubExecutor(delay=0.003), {persp: CountingAgent(persp, unbounded)},
                  c, runs_dir=tmp_path / "unb", max_concurrency=0))
    assert unbounded["peak"] >= 2              # unbounded ⇒ problems overlap


def test_retune_to_fewer_tiers_does_not_crash(tmp_path):
    # tuning the system between sessions (fewer tiers / different models) must never
    # crash a resume — tier access clamps to the new config.
    from solver.engine.context import RunContext
    two = cfg([Tier("cheap", [Perspective("a", "1")]), Tier("strong", [Perspective("b", "2")])])
    ctx = RunContext(1, two, tmp_path)
    ctx.tier_idx = 1
    assert ctx.tier.name == "strong"
    ctx.cfg = cfg([Tier("new", [Perspective("c", "9")])])     # resume under a 1-tier config
    assert ctx.tier.name == "new"                             # clamped, no IndexError
    assert ctx.pool_size == 1 and ctx.current_perspective() is not None


def test_reopen_resumes_any_non_final_termination(tmp_path):
    # only converged/target are final; a killed ('stopped') or capped run resumes,
    # so the journal keeps going across retunes.
    from solver.engine.context import RunContext
    c = one_tier(max_iterations=100, plateau_cycles=999)
    cases = {"budget:iterations": True, "budget:time": True, "stopped": True,
             "agents-unavailable": True, "converged:ceiling": False,
             "converged:last-tier": False, "target": False}
    for reason, should in cases.items():
        ctx = RunContext(1, c, tmp_path / reason.replace(":", "_"))
        ctx.terminated_reason = reason
        assert ctx.reopen_if_capped() is should, reason


def test_shuffle_randomizes_launch_order(tmp_path):
    # --shuffle launches problems in a seeded-random order, so a concurrency window
    # samples random ids (not always the lowest), and every problem still runs.
    from solver.engine.agent import Candidate, solution_hash
    seen = []

    class RecAgent:
        def __init__(self, persp):
            self.perspective = persp
        async def design(self, task_id):
            seen.append(task_id)
            return "d"
        async def plan(self, parent, ctx):
            sol = {"__eval__": {"scores": [0.6]}, "__uid__": f"{self.perspective}:{getattr(ctx, 'iters', 0)}"}
            return Candidate(cand_id=solution_hash(sol)[:12], solution=sol, parent=None,
                             agent=self.perspective.agent, model=self.perspective.model, strategy="s")

    persp = Perspective("x", "1")
    c = cfg([Tier("t", [persp])], max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1)
    ids = list(range(1, 9))
    run(run_fleet(ids, StubExecutor(), {persp: RecAgent(persp)}, c, runs_dir=tmp_path,
                  max_concurrency=1, shuffle=True, seed=0))         # cap=1 ⇒ serial ⇒ launch order observable
    assert sorted(seen) == ids                                     # every problem ran (coverage)
    assert seen != ids                                             # order was shuffled, not 1..8


def test_all_agents_dead_terminates_cleanly(tmp_path):
    from solver.engine.agent import StubAgent
    dead = Perspective("dead", "x")
    c = cfg([Tier("all", [dead])], max_iterations=16, plateau_cycles=999,
            escalate_ceiling=1.1, agent_fail_limit=3)
    agents = {dead: StubAgent(dead, lambda *a: {"scores": [0.6]}, raise_on=lambda *a: True)}
    ctx = run(solve_problem(1, StubExecutor(), agents, c, runs_dir=tmp_path))
    assert "dead:x" in ctx.disabled
    assert ctx.terminated_reason == "agents-unavailable"


# --------------------------------------------------------------------------- #
# test 6 — agent errors are non-fatal (a timeout skips the iteration, keeps the
# problem), and the fleet stays isolated
# --------------------------------------------------------------------------- #
def test_agent_error_is_nonfatal(tmp_path):
    c = one_tier(max_iterations=3, plateau_cycles=999, escalate_ceiling=1.1)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}),
                         raise_on=lambda self, parent, ctx: ctx.task_id == 2)
    run(run_fleet([1, 2, 3], StubExecutor(), agents, c, runs_dir=tmp_path))
    ev2 = events(tmp_path / "2" / "journal.jsonl")
    assert "plan_error" in ev2 and "solver_error" not in ev2   # skipped, NOT aborted
    assert "terminated" in ev2                                 # still finished cleanly
    assert "accept" in ev2                                     # its seed frontier survived
    for good in (1, 3):                                        # other problems unaffected
        evs = events(tmp_path / str(good) / "journal.jsonl")
        assert "terminated" in evs and "solver_error" not in evs


# --------------------------------------------------------------------------- #
# test 7 — concurrency / single-flight under forced interleaving
# --------------------------------------------------------------------------- #
def test_single_flight_under_races(tmp_path):
    c = one_tier(max_iterations=3, plateau_cycles=999, escalate_ceiling=1.1)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}))
    ex = StubExecutor(delay=0.001)               # delay forces overlap attempts
    run(run_fleet([1, 2, 3, 4, 5], ex, agents, c, runs_dir=tmp_path))
    assert ex.max_concurrent == 1                # re-entrancy assertion never tripped
    assert ex.calls == 5 * 4                     # 5 problems × (1 seed + 3 evals)


# --------------------------------------------------------------------------- #
# test 1 — kill/resume safety: identical final state after a kill at every line
# --------------------------------------------------------------------------- #
def test_kill_resume_is_identical(tmp_path):
    # deterministic-from-ctx planner: first candidate improves, the rest are
    # dominated → the single tier plateaus and terminates.
    def planner(persp, parent, ctx):
        return {"scores": [round(0.8 - 0.01 * ctx.iters, 4)]}
    c = one_tier(plateau_cycles=2, escalate_ceiling=0.9, max_iterations=50)

    def build_agents():
        return stub_agents(c.perspectives, planner)

    # 1) uninterrupted run
    base_dir = tmp_path / "base"
    ctx0 = run(solve_problem(1, StubExecutor(), build_agents(), c, runs_dir=base_dir))
    final = state(ctx0)
    lines = (base_dir / "1" / "journal.jsonl").read_text().splitlines()
    assert final[5] is not None                  # it terminated

    # 2) kill after every journal line, resume, assert identical final state
    for L in range(1, len(lines) + 1):
        d = tmp_path / f"resume_{L}"
        (d / "1").mkdir(parents=True)
        (d / "1" / "journal.jsonl").write_text("\n".join(lines[:L]) + "\n")
        ctx = run(solve_problem(1, StubExecutor(), build_agents(), c, runs_dir=d))
        assert state(ctx) == final, f"resume after {L} lines diverged"


# --------------------------------------------------------------------------- #
# on_phase — live "what is this agent doing right now" for the dashboard
# (runs/_active.json's task_phase). Purely informational: never read back by
# the engine, so it can't affect replay/resume — verified separately by the
# kill/resume test above still passing unmodified.
# --------------------------------------------------------------------------- #
def test_on_phase_reports_design_then_plan_and_always_clears(tmp_path):
    seen = []

    def on_phase(name, info):
        seen.append((name, info.get("agent") if info else None, info.get("model") if info else None))

    c = one_tier(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1, review_enabled=False)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}))
    run(solve_problem(1, StubExecutor(), agents, c, runs_dir=tmp_path, on_phase=on_phase))

    assert seen == [("design", "claude", "haiku"), (None, None, None),
                    ("plan", "claude", "haiku"), (None, None, None)]


def test_on_phase_is_optional_and_defaults_to_no_reporting(tmp_path):
    """Every existing caller (nothing passes on_phase) must keep working exactly
    as before — this is an opt-in side channel, not a required parameter."""
    c = one_tier(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}))
    ctx = run(solve_problem(1, StubExecutor(), agents, c, runs_dir=tmp_path))
    assert ctx.iters >= 1


def test_run_fleet_publishes_task_phase_to_the_active_file(tmp_path):
    """run_fleet threads on_phase through to _active.json's task_phase, keyed by
    task id, and removes the entry once the problem finishes (whether it ends
    mid-phase or between rounds)."""
    import json as _json

    c = one_tier(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1, review_enabled=False)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}))
    run(run_fleet([1], StubExecutor(), agents, c, runs_dir=tmp_path))
    active = _json.loads((tmp_path / "_active.json").read_text())
    assert active.get("task_phase") == {}          # problem finished -> no call in flight
