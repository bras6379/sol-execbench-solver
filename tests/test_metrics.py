"""solver.dashboard.metrics — problem_metrics candidate-status derivation.

No prior coverage existed for this despite the dashboard's whole per-candidate
table depending on it. Added after a live incident (2026-07-08): a candidate
sent back for repair ("revise") never had its status updated away from the
default "planned" — on the dashboard this made a review-repair cycle's
superseded rounds look identical to genuinely in-flight/pending work.
"""

from __future__ import annotations

import datetime
import json

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


def test_a_repair_round_collapses_into_its_lineages_one_row():
    """A review->repair->review->ship chain is ONE logical attempt (the writer's
    own resumed session, continuing toward one eventual GPU submission) — it must
    surface as ONE dashboard row reflecting the FINAL outcome, not two rows where
    the pre-repair one sits stuck at "revised" forever."""
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "first try"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "review", "cand": "a1", "reviewer": "r", "verdict": "revise", "issues": ["bug"]},
        {"ev": "plan_done", "cand": "a2", "parent": "a1", "model": "m",
         "strategy": "fixed the bug", "repair": True},
        {"ev": "check", "cand": "a2", "ok": True},
        {"ev": "review", "cand": "a2", "reviewer": "r", "verdict": "ship", "round": 1},
        {"ev": "exec_enqueued", "job": "a2", "cand": "a2"},
        {"ev": "exec_started", "job": "a2"},
        {"ev": "exec_done", "job": "a2", "cand": "a2", "all_passed": True,
         "sol_score": 0.7, "scores": [0.7]},
        {"ev": "accept", "cand": "a2", "verdict": "entered", "best": 0.7, "frontier": 1},
    ]
    for e in events:
        e.setdefault("ts", "2026-01-01T00:00:00Z")
    m = metrics.problem_metrics(1, events)
    assert len(m["candidates"]) == 1                       # not two ghost rows
    row = m["candidates"][0]
    assert row["cand"] == "a2" and row["status"] == "accepted"
    assert row["strategy"] == "fixed the bug"              # shows the FINAL content, not the flagged one
    assert row["repairs"] == 1


def test_a_repair_that_fails_check_never_adopts_its_content():
    """loop.py ships the ORIGINAL (pre-repair) candidate when a repair round
    breaks the schema — the row must keep showing a1's content/outcome, not a2's
    invalid one, and a2 (which never really existed as a shippable thing) must
    not appear as its own row either."""
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "first try"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "review", "cand": "a1", "reviewer": "r", "verdict": "revise", "issues": ["bug"]},
        {"ev": "plan_done", "cand": "a2", "parent": "a1", "model": "m",
         "strategy": "broken repair", "repair": True},
        {"ev": "check", "cand": "a2", "ok": False},
        # repair broke the schema -> loop.py ships the PRE-repair candidate (a1)
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1"},
        {"ev": "exec_started", "job": "a1"},
        {"ev": "exec_done", "job": "a1", "cand": "a1", "all_passed": True,
         "sol_score": 0.5, "scores": [0.5]},
        {"ev": "accept", "cand": "a1", "verdict": "entered", "best": 0.5, "frontier": 1},
    ]
    for e in events:
        e.setdefault("ts", "2026-01-01T00:00:00Z")
    m = metrics.problem_metrics(1, events)
    assert len(m["candidates"]) == 1
    row = m["candidates"][0]
    assert row["cand"] == "a1" and row["strategy"] == "first try"
    assert row["status"] == "accepted"


# --------------------------------------------------------------------------- #
# current_phase — live "what is this agent doing right now", sourced from
# runs/_active.json's task_phase (published by loop.py's on_phase hook). This
# is what makes "is it running or stuck?" answerable without process-list /
# journal-age guessing (see the 2026-07-08 dashboard-visibility request).
# --------------------------------------------------------------------------- #
def test_current_phase_is_attached_from_a_fresh_active_file(tmp_path):
    now = datetime.datetime.now(datetime.timezone.utc)
    started = (now - datetime.timedelta(seconds=42)).isoformat()
    (tmp_path / "_active.json").write_text(json.dumps({
        "ts": now.isoformat(), "active": [1], "cap": 5, "phase": "running", "reflect": None,
        "task_phase": {"1": {"phase": "plan", "agent": "claude", "model": "haiku",
                             "cand": "abc123def", "started": started}},
    }))
    journals = {1: [{"ev": "plan_done", "cand": "x", "ts": now.isoformat(), "model": "haiku"}]}
    data = metrics.collect(journals, runs_dir=tmp_path)
    cp = data["problems"][0]["current_phase"]
    assert cp["phase"] == "plan" and cp["agent"] == "claude" and cp["cand"] == "abc123def"
    assert 40 <= cp["elapsed_s"] <= 45


def test_current_phase_is_absent_when_the_active_file_is_stale(tmp_path):
    stale_ts = (datetime.datetime.now(datetime.timezone.utc)
               - datetime.timedelta(minutes=10)).isoformat()
    (tmp_path / "_active.json").write_text(json.dumps({
        "ts": stale_ts, "active": [1], "cap": 5, "phase": "running", "reflect": None,
        "task_phase": {"1": {"phase": "plan", "agent": "claude", "model": "haiku",
                             "cand": "abc", "started": stale_ts}},
    }))
    journals = {1: [{"ev": "plan_done", "cand": "x", "ts": stale_ts}]}
    data = metrics.collect(journals, runs_dir=tmp_path)
    assert data["problems"][0]["current_phase"] is None


def test_current_phase_is_none_without_an_active_file(tmp_path):
    journals = {1: [{"ev": "plan_done", "cand": "x", "ts": "2026-01-01T00:00:00Z"}]}
    data = metrics.collect(journals, runs_dir=tmp_path)
    assert data["problems"][0]["current_phase"] is None


# --------------------------------------------------------------------------- #
# "planned" sub-states — gpu_queued / gpu_running / reviewing / repairing.
# Without these, a candidate genuinely stuck (nothing left in flight) looks
# IDENTICAL to one queued for the single-flight GPU, actively running on it, or
# mid-review — you can't tell "is it running or stuck?" from the status alone.
# --------------------------------------------------------------------------- #
def test_a_candidate_queued_for_the_gpu_shows_gpu_queued():
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1"},
    ]
    assert _cand_status(events, "a1") == "gpu_queued"


def test_a_candidate_actively_executing_shows_gpu_running():
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1"},
        {"ev": "exec_started", "job": "a1"},   # no `cand` field at this call site — matched via job
    ]
    assert _cand_status(events, "a1") == "gpu_running"


def test_gpu_running_still_resolves_to_the_real_outcome_once_done():
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1"},
        {"ev": "exec_started", "job": "a1"},
        {"ev": "exec_done", "job": "a1", "cand": "a1", "all_passed": True,
         "sol_score": 0.6, "scores": [0.6]},
        {"ev": "accept", "cand": "a1", "verdict": "entered", "best": 0.6, "frontier": 1},
    ]
    assert _cand_status(events, "a1") == "accepted"


def test_gpu_running_still_resolves_to_dominated_once_done():
    """Regression: a CORRECT-but-dominated candidate's status is "gpu_running" by
    the time `accept` fires (exec_done never touches status on a pass) — the
    dominated branch's guard used to require status == "planned" exactly, which
    no longer matched once gpu_queued/gpu_running existed, leaving it stuck
    showing "gpu: running" forever alongside its real (final) score."""
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1"},
        {"ev": "exec_started", "job": "a1"},
        {"ev": "exec_done", "job": "a1", "cand": "a1", "all_passed": True,
         "sol_score": 0.6, "scores": [0.6]},
        {"ev": "accept", "cand": "a1", "verdict": "dominated", "best": 0.7, "frontier": 1},
    ]
    assert _cand_status(events, "a1") == "dominated"


def test_incorrect_status_is_not_overwritten_by_a_dominated_verdict():
    """An INCORRECT candidate can still get a 'dominated' accept verdict (it
    can't beat anything) — the real failure reason must win, not get replaced."""
    events = [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s"},
        {"ev": "check", "cand": "a1", "ok": True},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1"},
        {"ev": "exec_started", "job": "a1"},
        {"ev": "exec_done", "job": "a1", "cand": "a1", "all_passed": False,
         "sol_score": None, "scores": [None]},
        {"ev": "accept", "cand": "a1", "verdict": "dominated", "best": 0.7, "frontier": 1},
    ]
    assert _cand_status(events, "a1") == "incorrect"


def test_current_phase_review_marks_the_matching_candidate_reviewing(tmp_path):
    now = datetime.datetime.now(datetime.timezone.utc)
    started = (now - datetime.timedelta(seconds=10)).isoformat()
    (tmp_path / "_active.json").write_text(json.dumps({
        "ts": now.isoformat(), "active": [1], "cap": 5, "phase": "running", "reflect": None,
        "task_phase": {"1": {"phase": "review", "agent": "claude", "model": "sonnet",
                             "cand": "a1", "started": started}},
    }))
    journals = {1: [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s", "ts": now.isoformat()},
        {"ev": "check", "cand": "a1", "ok": True, "ts": now.isoformat()},
    ]}
    data = metrics.collect(journals, runs_dir=tmp_path)
    row = next(c for c in data["problems"][0]["candidates"] if c["cand"] == "a1")
    assert row["status"] == "reviewing"


def test_current_phase_repair_marks_the_matching_candidate_repairing(tmp_path):
    now = datetime.datetime.now(datetime.timezone.utc)
    started = (now - datetime.timedelta(seconds=10)).isoformat()
    (tmp_path / "_active.json").write_text(json.dumps({
        "ts": now.isoformat(), "active": [1], "cap": 5, "phase": "running", "reflect": None,
        "task_phase": {"1": {"phase": "repair", "agent": "claude", "model": "sonnet",
                             "cand": "a1", "started": started}},
    }))
    journals = {1: [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s", "ts": now.isoformat()},
        {"ev": "check", "cand": "a1", "ok": True, "ts": now.isoformat()},
        {"ev": "review", "cand": "a1", "reviewer": "r", "verdict": "revise", "ts": now.isoformat()},
    ]}
    data = metrics.collect(journals, runs_dir=tmp_path)
    row = next(c for c in data["problems"][0]["candidates"] if c["cand"] == "a1")
    assert row["status"] == "repairing"          # NOT stuck at "revised" while the repair runs


def test_current_phase_plan_does_not_touch_any_existing_row(tmp_path):
    """A 'plan' phase's cand is the PARENT being improved on, not a new row — it
    must never be mistaken for 'this existing candidate is being re-planned'."""
    now = datetime.datetime.now(datetime.timezone.utc)
    (tmp_path / "_active.json").write_text(json.dumps({
        "ts": now.isoformat(), "active": [1], "cap": 5, "phase": "running", "reflect": None,
        "task_phase": {"1": {"phase": "plan", "agent": "claude", "model": "sonnet",
                             "cand": "a1", "started": now.isoformat()}},
    }))
    journals = {1: [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s", "ts": now.isoformat()},
        {"ev": "check", "cand": "a1", "ok": True, "ts": now.isoformat()},
        {"ev": "review", "cand": "a1", "reviewer": "r", "verdict": "ship", "ts": now.isoformat()},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1", "ts": now.isoformat()},
        {"ev": "exec_started", "job": "a1", "ts": now.isoformat()},
        {"ev": "exec_done", "job": "a1", "cand": "a1", "all_passed": True,
         "sol_score": 0.6, "scores": [0.6], "ts": now.isoformat()},
        {"ev": "accept", "cand": "a1", "verdict": "entered", "best": 0.6,
         "frontier": 1, "ts": now.isoformat()},
    ]}
    data = metrics.collect(journals, runs_dir=tmp_path)
    row = next(c for c in data["problems"][0]["candidates"] if c["cand"] == "a1")
    assert row["status"] == "accepted"           # untouched by the unrelated 'plan' phase


# --------------------------------------------------------------------------- #
# "interrupted" — an abrupt stop (SIGTERM/SIGKILL) can land in the gap between
# a GPU eval finishing and its accept/frontier decision being journaled,
# permanently orphaning that candidate mid-transition (confirmed live,
# 2026-07-08: exec_done fired with a real score, no accept ever followed).
# Nothing will ever advance it once the problem isn't live anymore — must not
# keep showing a label that implies active work.
# --------------------------------------------------------------------------- #
def test_a_stuck_in_flight_candidate_shows_interrupted_when_the_problem_is_not_live(tmp_path):
    old_ts = "2020-01-01T00:00:00Z"        # far past -> definitely not "running"
    journals = {1: [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s", "ts": old_ts},
        {"ev": "check", "cand": "a1", "ok": True, "ts": old_ts},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1", "ts": old_ts},
        {"ev": "exec_started", "job": "a1", "ts": old_ts},
        {"ev": "exec_done", "job": "a1", "cand": "a1", "all_passed": True,
         "sol_score": 0.6, "scores": [0.6], "ts": old_ts},
        # no "accept" — the process died right here, exactly like the live incident
    ]}
    data = metrics.collect(journals, runs_dir=tmp_path)   # no _active.json at all
    row = next(c for c in data["problems"][0]["candidates"] if c["cand"] == "a1")
    assert data["problems"][0]["live_state"] != "running"
    assert row["status"] == "interrupted"


def test_a_live_in_flight_candidate_is_not_marked_interrupted(tmp_path):
    now = datetime.datetime.now(datetime.timezone.utc)
    (tmp_path / "_active.json").write_text(json.dumps({
        "ts": now.isoformat(), "active": [1], "cap": 5, "phase": "running", "reflect": None,
        "task_phase": {},
    }))
    journals = {1: [
        {"ev": "plan_done", "cand": "a1", "model": "m", "strategy": "s", "ts": now.isoformat()},
        {"ev": "check", "cand": "a1", "ok": True, "ts": now.isoformat()},
        {"ev": "exec_enqueued", "job": "a1", "cand": "a1", "ts": now.isoformat()},
    ]}
    data = metrics.collect(journals, runs_dir=tmp_path)
    row = next(c for c in data["problems"][0]["candidates"] if c["cand"] == "a1")
    assert data["problems"][0]["live_state"] == "running"
    assert row["status"] == "gpu_queued"          # genuinely still in flight, left alone
