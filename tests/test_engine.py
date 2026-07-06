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
        {"scores": [0.6], "duplicate": True},   # i0: novel → eval
        {"scores": [0.6], "duplicate": True},   # i1: identical hash → dup, no eval
        {"scores": [0.7], "invalid": True},     # i2: check fail → reject, no eval
        {"scores": [0.65]},                     # i3: novel → eval
    ]}
    c = one_tier(max_iterations=4, plateau_cycles=999, escalate_ceiling=1.1)
    ex = StubExecutor()
    agents = stub_agents(c.perspectives, scripted(script))
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))
    assert ex.calls == 3                        # 1 seed + i0 + i3 (dup & invalid skipped)
    assert ctx.evals == 3
    evs = events(ctx.path)
    assert evs.count("iter") == 4               # every plan committed one iteration
    # one rejected, one duplicate outcome recorded
    outcomes = [e for e in journal_mod.read(ctx.path) if e["ev"] == "iter"]
    kinds = [o["outcome"] for o in outcomes]
    assert "rejected" in kinds and "duplicate" in kinds


def test_judge_cosmetic_bounces(tmp_path):
    c = one_tier(max_iterations=3, plateau_cycles=999, escalate_ceiling=1.1)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}),
                         judge_fn=lambda cand, parent, fr: "cosmetic")
    ex = StubExecutor()
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))
    assert ex.calls == 1                        # only the seed; every plan bounced at novelty
    assert ctx.frontier.best_score() == 0.5     # nothing but the seed entered


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
# test 6 — crash isolation
# --------------------------------------------------------------------------- #
def test_crash_isolation(tmp_path):
    c = one_tier(max_iterations=3, plateau_cycles=999, escalate_ceiling=1.1)
    agents = stub_agents(c.perspectives, scripted({"claude:haiku": [{"scores": [0.6]}]}),
                         raise_on=lambda self, parent, ctx: ctx.task_id == 2)
    run(run_fleet([1, 2, 3], StubExecutor(), agents, c, runs_dir=tmp_path))
    assert "solver_error" in events(tmp_path / "2" / "journal.jsonl")
    for good in (1, 3):
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
