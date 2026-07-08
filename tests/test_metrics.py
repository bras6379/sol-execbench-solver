"""solver.dashboard.metrics — problem_metrics candidate-status derivation.

No prior coverage existed for this despite the dashboard's whole per-candidate
table depending on it. Added after a live incident (2026-07-08): a candidate
sent back for repair ("revise") never had its status updated away from the
default "planned" — on the dashboard this made a review-repair cycle's
superseded rounds look identical to genuinely in-flight/pending work.
"""

from __future__ import annotations

from solver.dashboard import metrics


def _cand_status(events, cand_id):
    for e in events:
        e.setdefault("ts", "2026-01-01T00:00:00Z")
    m = metrics.problem_metrics(1, events)
    c = next(c for c in m["candidates"] if c["cand"] == cand_id)
    return c["status"]


def test_a_revised_candidate_is_not_left_planned_forever():
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "first try"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "review", "cand": "a1", "reviewer": "r", "verdict": "revise", "issues": ["bug"]},
    ]
    assert _cand_status(events, "a1") == "revised"


def test_a_shipped_candidate_still_reaches_its_real_outcome():
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "first try"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "review", "cand": "a1", "reviewer": "r", "verdict": "ship"},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1"},
        {"ev": "exec_started", "job": "a1"},
        {"ev": "exec_done", "job": "a1", "cand": "a1", "all_passed": True,
         "sol_score": 0.6, "scores": [0.6]},
        {"ev": "accept", "cand": "a1", "verdict": "entered", "best": 0.6, "frontier": 1},
    ]
    assert _cand_status(events, "a1") == "accepted"


def test_revise_never_overwrites_an_already_resolved_status():
    """A stray/duplicate review event for a candidate that already reached a
    real outcome must not regress it back to "revised"."""
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "first try"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1"},
        {"ev": "exec_started", "job": "a1"},
        {"ev": "exec_done", "job": "a1", "cand": "a1", "all_passed": True,
         "sol_score": 0.6, "scores": [0.6]},
        {"ev": "accept", "cand": "a1", "verdict": "entered", "best": 0.6, "frontier": 1},
        {"ev": "review", "cand": "a1", "reviewer": "r", "verdict": "revise"},
    ]
    assert _cand_status(events, "a1") == "accepted"
