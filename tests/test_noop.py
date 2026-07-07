"""Tests for no-op / ceiling-consensus detection (solver/engine/loop.py).

A no-op is an agent turn whose output hashes EXACTLY to the parent it was given
— it changed nothing. Measured live: ~97% of iterations on a stuck problem were
this, silently absorbed as a generic 'duplicate' with no signal and no cost
control. These tests assert it's detected distinctly and, once N models agree
in a row, the problem auto-terminates instead of burning turns forever.

`"duplicate": True` in a StubAgent spec skips the automatic `__uid__` tag (see
agent.py), making the resulting solution hash a pure function of `scores` — so
returning the SAME scores list every call reproduces the SAME cand_id, exactly
simulating an agent that keeps handing back its parent's kernel unchanged.
"""

from __future__ import annotations

import asyncio

from solver.engine import Config, Perspective, StubExecutor, Tier, solve_problem, stub_agents
from solver import journal as journal_mod

run = asyncio.run

A = Perspective("claude", "haiku")
B = Perspective("claude", "opus")

NOOP_SPEC = {"scores": [0.6], "duplicate": True}   # deterministic hash every call


def two_persp(**kw):
    return Config(tiers=[Tier("t", [A, B])], **kw)


def events(path):
    return list(journal_mod.read(path))


def test_noop_detected_distinctly_from_generic_duplicate(tmp_path):
    """An agent that keeps handing back the same (parent's) content is journaled
    as 'no_op', not the generic 'duplicate' — a stronger, more specific signal."""
    agents = stub_agents([A, B], lambda persp, parent, ctx: dict(NOOP_SPEC))
    ex = StubExecutor()
    c = two_persp(max_iterations=2, plateau_cycles=999, escalate_ceiling=1.1,
                  ceiling_consensus=0)                # disabled — just check labeling
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    outcomes = [e["outcome"] for e in events(ctx.path) if e["ev"] == "iter"]
    assert outcomes.count("no_op") >= 1
    assert "duplicate" not in outcomes
    novelty = [e for e in events(ctx.path) if e["ev"] == "novelty"]
    assert any(n["verdict"] == "no_op" for n in novelty)
    assert ex.calls == 2                              # seed + the one genuinely-new first candidate


def test_ceiling_consensus_terminates_after_n_consecutive_noops(tmp_path):
    agents = stub_agents([A, B], lambda persp, parent, ctx: dict(NOOP_SPEC))
    ex = StubExecutor()
    c = two_persp(max_iterations=20, plateau_cycles=999, escalate_ceiling=1.1,
                  ceiling_consensus=3)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert ctx.terminated_reason == "ceiling_consensus"
    outcomes = [e["outcome"] for e in events(ctx.path) if e["ev"] == "iter"]
    assert outcomes.count("no_op") == 3                # stopped exactly at the threshold, not later
    assert ex.calls == 2                               # seed + the first (accepted) real candidate


def test_a_real_attempt_resets_the_noop_streak(tmp_path):
    """No-op, no-op, then a genuinely NEW candidate, then more no-ops — must NOT
    trigger at streak=3 total; only 3 CONSECUTIVE no-ops count."""
    calls = {"n": 0}

    def planner(persp, parent, ctx):
        calls["n"] += 1
        if calls["n"] == 3:
            return {"scores": [0.7], "duplicate": True}   # a real, DIFFERENT attempt
        return dict(NOOP_SPEC)

    agents = stub_agents([A, B], planner)
    ex = StubExecutor()
    c = two_persp(max_iterations=4, plateau_cycles=999, escalate_ceiling=1.1,
                  ceiling_consensus=3)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert ctx.terminated_reason != "ceiling_consensus"   # streak reset by the real attempt
    outcomes = [e["outcome"] for e in events(ctx.path) if e["ev"] == "iter"]
    assert outcomes.count("no_op") >= 1
    assert any(o in ("entered", "dominated") for o in outcomes)   # the real attempt was evaluated


def test_a_rejected_candidate_also_resets_the_streak(tmp_path):
    """An invalid (check_fn-rejected) candidate is a real attempt, just a bad one
    — it must reset the streak the same as a real accepted/dominated candidate."""
    calls = {"n": 0}

    def planner(persp, parent, ctx):
        calls["n"] += 1
        if calls["n"] == 2:
            return {"scores": [0.7], "invalid": True}
        return dict(NOOP_SPEC)

    agents = stub_agents([A, B], planner)
    ex = StubExecutor()
    c = two_persp(max_iterations=3, plateau_cycles=999, escalate_ceiling=1.1,
                  ceiling_consensus=2)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert ctx.terminated_reason != "ceiling_consensus"   # the rejected turn broke the streak


def test_ceiling_consensus_disabled_by_zero_never_terminates(tmp_path):
    agents = stub_agents([A, B], lambda persp, parent, ctx: dict(NOOP_SPEC))
    ex = StubExecutor()
    c = two_persp(max_iterations=10, plateau_cycles=999, escalate_ceiling=1.1,
                  ceiling_consensus=0)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert ctx.terminated_reason != "ceiling_consensus"
    outcomes = [e["outcome"] for e in events(ctx.path) if e["ev"] == "iter"]
    assert outcomes.count("no_op") == 9                # 1 real + 9 no-ops over the full budget


def test_ceiling_consensus_is_reopenable_on_resume(tmp_path):
    """A ceiling_consensus stop is not final — a later restart (e.g. after a
    config/model change) should get another shot, same as budget:* caps."""
    agents = stub_agents([A, B], lambda persp, parent, ctx: dict(NOOP_SPEC))
    ex = StubExecutor()
    c = two_persp(max_iterations=3, plateau_cycles=999, escalate_ceiling=1.1,
                  ceiling_consensus=3)
    run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    # resume with a planner that produces genuinely new candidates each time
    def real_planner(persp, parent, ctx):
        return {"scores": [0.6 + 0.01 * ctx.iters]}
    agents2 = stub_agents([A, B], real_planner)
    c2 = two_persp(max_iterations=10, plateau_cycles=999, escalate_ceiling=1.1,
                   ceiling_consensus=3)
    ctx2 = run(solve_problem(1, StubExecutor(), agents2, c2, runs_dir=tmp_path))
    assert any(e["ev"] == "reopened" for e in events(ctx2.path))
    assert ctx2.evals > 0                                  # it actually got to try again
