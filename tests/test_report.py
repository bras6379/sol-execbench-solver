"""solver.dashboard.report — live-transcript file discovery.

No prior coverage exists for report.py (pure HTML string generation, by
established precedent in this codebase). This one function is different: it's
a real filesystem heuristic (most-recently-modified trajectory file under a
problem's work/ tree, with a staleness guard) with genuine room to get wrong,
added to back the "open the live transcript of that agent" feature.
"""

from __future__ import annotations

import datetime
import os

from solver.dashboard import report


def _touch(path, mtime_offset_s=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"type": "thread.started", "thread_id": "t1"}\n')
    t = 1_700_000_000 + mtime_offset_s   # fixed epoch base -> deterministic, no wall-clock dependency
    os.utime(path, (t, t))
    return t


def test_finds_the_most_recently_modified_trajectory_file(tmp_path):
    work = tmp_path / "7" / "work"
    _touch(work / "cand1" / "trajectory.jsonl", mtime_offset_s=0)
    _touch(work / "review-abc-1" / "trajectory.jsonl", mtime_offset_s=100)   # newest
    _touch(work / "cand2" / "trajectory.repair-1.jsonl", mtime_offset_s=50)
    found = report._find_live_trajectory(7, tmp_path, started_iso=None)
    assert found == work / "review-abc-1" / "trajectory.jsonl"


def test_returns_none_when_the_work_dir_does_not_exist(tmp_path):
    assert report._find_live_trajectory(7, tmp_path, started_iso=None) is None


def test_returns_none_when_no_trajectory_files_exist(tmp_path):
    (tmp_path / "7" / "work" / "cand1").mkdir(parents=True)
    assert report._find_live_trajectory(7, tmp_path, started_iso=None) is None


def test_ignores_a_stale_file_older_than_the_phase_start(tmp_path):
    """The newest file under work/ predates when the CURRENT phase reportedly
    started — e.g. a leftover from a much earlier call — so it must not be
    shown as if it were this call's live transcript."""
    work = tmp_path / "7" / "work"
    real_now = datetime.datetime.now(datetime.timezone.utc)
    old_mtime = report._age_s_local(real_now.isoformat())  # ~0
    p = work / "cand1" / "trajectory.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("{}\n")
    # file's mtime is "now minus 2 hours" (old); the phase claims to have
    # started "now" (recent) -> the file predates the phase by ~2h, way past
    # the 30s slack -> must be rejected.
    stale_epoch = (real_now - datetime.timedelta(hours=2)).timestamp()
    os.utime(p, (stale_epoch, stale_epoch))
    found = report._find_live_trajectory(7, tmp_path, started_iso=real_now.isoformat())
    assert found is None


def test_accepts_a_file_modified_after_the_phase_started(tmp_path):
    work = tmp_path / "7" / "work"
    now = datetime.datetime.now(datetime.timezone.utc)
    started = (now - datetime.timedelta(seconds=5)).isoformat()   # phase started 5s ago
    p = work / "cand1" / "trajectory.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("{}\n")   # mtime = now (freshly written) -> after started
    found = report._find_live_trajectory(7, tmp_path, started_iso=started)
    assert found == p
