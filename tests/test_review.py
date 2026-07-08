"""Tests for the pre-GPU code review gate (solver/engine/loop.py's review/repair
cycle + pick_reviewer). Stub-powered — no GPU, no model. See docs/orchestration.md §12
for the stub contract this suite follows.

The tier's pool order is a per-problem SEEDED SHUFFLE (docs/orchestration.md), so
these tests never assume which of the two perspectives round-robin picks as the
writer for a given task_id — they discover it from the journal and assert
structural properties (one consistent writer across repair rounds, the reviewer
always excludes whoever that writer turned out to be) instead.
"""

from __future__ import annotations

import asyncio

from solver.engine import (
    Config,
    Perspective,
    ReviewVerdict,
    StubExecutor,
    Tier,
    pick_reviewer,
    solve_problem,
    stub_agents,
)
from solver import journal as journal_mod

run = asyncio.run

A = Perspective("claude", "haiku")
B = Perspective("claude", "opus")


def two_persp(**kw):
    return Config(tiers=[Tier("t", [A, B])], **kw)


def _planner(scores=(0.6,)):
    def planner(persp, parent, ctx):
        return {"scores": list(scores)}
    return planner


def events(path):
    return list(journal_mod.read(path))


def plan_dones(path):
    return [e for e in events(path) if e["ev"] == "plan_done"]


def reviews(path):
    return [e for e in events(path) if e["ev"] == "review"]


# --------------------------------------------------------------------------- #
# pick_reviewer
# --------------------------------------------------------------------------- #
def test_pick_reviewer_excludes_the_writer():
    pool = [A, B]
    for i in range(20):
        r = pick_reviewer(pool, A, key=f"k{i}")
        assert r == B                          # only one other choice — always picked


def test_pick_reviewer_deterministic_on_key():
    pool = [A, B, Perspective("codex", "gpt-5.5")]
    a = pick_reviewer(pool, A, key="fixed")
    b = pick_reviewer(pool, A, key="fixed")
    assert a == b and a != A


def test_pick_reviewer_falls_back_to_writer_when_sole_perspective():
    assert pick_reviewer([A], A, key="x") == A


# --------------------------------------------------------------------------- #
# the review/repair cycle inside solve_problem — both A and B get the SAME
# scripted reviewer so the test doesn't care which one round-robin picks as the
# writer for this task_id.
# --------------------------------------------------------------------------- #
def test_ship_verdict_sends_straight_to_gpu(tmp_path):
    """A reviewer that always ships costs one review call and zero repairs."""
    agents = stub_agents([A, B], _planner())    # default reviewer: always ship
    ex = StubExecutor()
    c = two_persp(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))
    assert ex.calls == 1 + 1                            # seed + the one accepted candidate
    revs = reviews(ctx.path)
    assert len(revs) == 1 and revs[0]["verdict"] == "ship"
    assert agents[A].calls + agents[B].calls == 1       # no repair — plan() called once total


def test_revise_triggers_repair_by_the_same_writer(tmp_path):
    """First review says revise; the SAME writer is re-invoked with the critique;
    second review (now on the repaired candidate) ships."""
    calls = {"n": 0}

    def reviewer(persp, cand, ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            assert ctx.review_critique is None          # not set before the first review
            return ReviewVerdict(verdict="revise", issues=["off-by-one at line 4"])
        return ReviewVerdict(verdict="ship")             # repair round: ship

    agents = stub_agents([A, B], _planner())
    agents[A]._reviewer = reviewer                       # scripted on BOTH — whichever plays
    agents[B]._reviewer = reviewer                       # reviewer, it behaves the same

    ex = StubExecutor()
    c = two_persp(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert calls["n"] == 2                              # revise once, then ship
    assert agents[A].calls + agents[B].calls == 2        # original plan + one repair plan
    assert agents[A].calls == 0 or agents[B].calls == 0  # but from the SAME writer, not both
    revs = reviews(ctx.path)
    assert [r["verdict"] for r in revs] == ["revise", "ship"]
    assert [r["round"] for r in revs] == [0, 1]
    plans = plan_dones(ctx.path)
    assert len(plans) == 2                              # original + repair, both journaled
    assert plans[0]["model"] == plans[1]["model"]        # same writer both times
    # the reviewer field never names the writer that round
    assert all(r["reviewer"] != f"{p['agent']}:{p['model']}" for r, p in zip(revs, plans))


def test_critique_is_visible_to_the_repair_call(tmp_path):
    """ctx.review_critique is set DURING the repair plan() call and cleared after."""
    seen = {}
    call_n = {"n": 0}

    def planner(persp, parent, ctx):
        call_n["n"] += 1
        if call_n["n"] == 2:                     # the repair call (2nd plan() this iteration)
            seen["critique_during_repair"] = ctx.review_critique
        return {"scores": [0.6]}

    def reviewer(persp, cand, ctx):
        if call_n["n"] == 1:                      # reviewing the ORIGINAL candidate
            return ReviewVerdict(verdict="revise", issues=["fix the mask"])
        return ReviewVerdict(verdict="ship")       # reviewing the repaired one

    agents = stub_agents([A, B], planner)
    agents[A]._reviewer = reviewer
    agents[B]._reviewer = reviewer
    ex = StubExecutor()
    c = two_persp(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1)
    run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert seen.get("critique_during_repair") and "fix the mask" in seen["critique_during_repair"]


def test_review_max_rounds_caps_the_repair_loop(tmp_path):
    """A reviewer that ALWAYS says revise must not loop forever — capped at
    review_max_rounds, then the candidate ships as-is (exactly one GPU eval)."""
    def reviewer(persp, cand, ctx):
        return ReviewVerdict(verdict="revise", issues=["still wrong"])

    agents = stub_agents([A, B], _planner())
    agents[A]._reviewer = reviewer
    agents[B]._reviewer = reviewer
    ex = StubExecutor()
    c = two_persp(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1,
                  review_max_rounds=3)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    revs = reviews(ctx.path)
    assert len(revs) == 4                               # rounds 0,1,2,3 — capped, not infinite
    assert [r["round"] for r in revs] == [0, 1, 2, 3]
    assert all(r["verdict"] == "revise" for r in revs)
    assert ex.calls == 1 + 1                             # seed + exactly ONE GPU eval despite 4 reviews


def test_review_disabled_skips_the_gate_entirely(tmp_path):
    reviewer_calls = {"n": 0}

    def reviewer(persp, cand, ctx):
        reviewer_calls["n"] += 1
        return ReviewVerdict(verdict="ship")

    agents = stub_agents([A, B], _planner())
    agents[A]._reviewer = reviewer
    agents[B]._reviewer = reviewer
    ex = StubExecutor()
    c = two_persp(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1,
                  review_enabled=False)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert reviewer_calls["n"] == 0
    assert not reviews(ctx.path)
    assert ex.calls == 1 + 1                             # unaffected — still evals normally


def test_review_error_fails_open_ships_as_is(tmp_path):
    """A reviewer that crashes must not block the candidate — it ships as-is,
    matching the system-wide 'never let a side-check permanently block' rule."""
    def reviewer(persp, cand, ctx):
        raise RuntimeError("reviewer CLI timed out")

    agents = stub_agents([A, B], _planner())
    agents[A]._reviewer = reviewer
    agents[B]._reviewer = reviewer
    ex = StubExecutor()
    c = two_persp(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert ex.calls == 1 + 1                             # still evaluated despite the review crash
    errs = [e for e in events(ctx.path) if e["ev"] == "review_error"]
    assert len(errs) == 1


def test_repair_no_op_ships_without_a_further_review_round(tmp_path):
    """If a repair round leaves the content byte-identical to what it was asked to
    fix (the writer's own prompt forbids this, but a model can still do it — this
    is exactly the failure confirmed live, 2026-07-08 problem #18: a cold-start
    repair produced identical bytes, and the reviewer flip-flopped from 'revise'
    to 'ship' reviewing the SAME content seconds later), stop immediately and ship
    as-is rather than looping for another (non-deterministic) review pass."""
    fixed_solution = {"solution": {"__fixed__": True, "__eval__": {"scores": [0.6]}}}
    reviews_seen = {"n": 0}

    def reviewer(persp, cand, ctx):
        reviews_seen["n"] += 1
        return ReviewVerdict(verdict="revise" if reviews_seen["n"] == 1 else "ship",
                             issues=["still wrong"])

    # both the original plan and the "repair" (StubAgent.repair delegates to
    # plan()) return the SAME fixed solution dict -> identical content hash
    script = {"claude:haiku": [fixed_solution, fixed_solution],
              "claude:opus": [fixed_solution, fixed_solution]}
    agents = stub_agents([A, B], lambda persp, parent, ctx: script[str(persp)].pop(0))
    agents[A]._reviewer = reviewer
    agents[B]._reviewer = reviewer
    ex = StubExecutor()
    c = two_persp(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1, review_max_rounds=6)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert reviews_seen["n"] == 1                    # no second review on the identical repair
    revs = reviews(ctx.path)
    assert len(revs) == 1 and revs[0]["verdict"] == "revise"
    noops = [e for e in events(ctx.path) if e["ev"] == "iter" and e.get("outcome") == "repair_no_op"]
    assert len(noops) == 1
    assert ex.calls == 1 + 1                          # seed + exactly one GPU eval (shipped as-is)


def test_on_phase_reports_review_then_repair_around_a_revise_cycle(tmp_path):
    """The live 'what's happening now' side channel (runs/_active.json's
    task_phase) must show review/repair distinctly, each cleared immediately
    after — dashboard visibility into a phase that used to be invisible until
    the whole review-repair cycle finished and journaled its result."""
    seen = []

    def on_phase(name, info):
        seen.append(name)

    calls = {"n": 0}

    def reviewer(persp, cand, ctx):
        calls["n"] += 1
        return ReviewVerdict(verdict="revise", issues=["x"]) if calls["n"] == 1 \
               else ReviewVerdict(verdict="ship")

    agents = stub_agents([A, B], _planner())
    agents[A]._reviewer = reviewer
    agents[B]._reviewer = reviewer
    ex = StubExecutor()
    c = two_persp(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1)
    run(solve_problem(1, ex, agents, c, runs_dir=tmp_path, on_phase=on_phase))

    assert seen == ["design", None, "plan", None, "review", None,
                    "repair", None, "review", None]


def test_repair_that_breaks_schema_ships_the_pre_repair_candidate(tmp_path):
    """If the repair round produces an invalid solution (fails check_fn), the
    ORIGINAL (pre-repair) candidate is shipped rather than the broken repair."""
    call_n = {"n": 0}

    def planner(persp, parent, ctx):
        call_n["n"] += 1
        if call_n["n"] == 2:                      # the repair call
            return {"scores": [0.6], "invalid": True}    # repair produces junk
        return {"scores": [0.6]}

    def reviewer(persp, cand, ctx):
        return ReviewVerdict(verdict="revise", issues=["x"]) if call_n["n"] == 1 \
               else ReviewVerdict(verdict="ship")

    agents = stub_agents([A, B], planner)
    agents[A]._reviewer = reviewer
    agents[B]._reviewer = reviewer
    ex = StubExecutor()
    c = two_persp(max_iterations=1, plateau_cycles=999, escalate_ceiling=1.1)
    ctx = run(solve_problem(1, ex, agents, c, runs_dir=tmp_path))

    assert call_n["n"] == 2                              # the repair WAS attempted
    assert ex.calls == 1 + 1                             # the pre-repair candidate still got evaluated
