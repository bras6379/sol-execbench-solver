"""Tests for the fable-5 diagnosis layer — gating, dedup, and attachment.

The real fable call is monkeypatched, so these assert the COST-CONTROL contract
(only stuck problems, deduped on state, graceful on failure) without spending.
"""

from __future__ import annotations

import asyncio
import json

from solver.engine import diagnose, reflection as R
from solver.dashboard.report import _md_to_html

run = asyncio.run


def _mk_problem(runs, task, scores, strategies, status_expect):
    p = runs / str(task)
    (p / "candidates").mkdir(parents=True)
    ev = []
    for i, (s, strat) in enumerate(zip(scores, strategies)):
        cid = f"c{task}_{i:02d}"
        ev.append({"ev": "plan_done", "cand": cid, "model": "opus", "strategy": strat})
        ev.append({"ev": "exec_done", "cand": cid, "sol_score_cal": s})
        c = {"cand_id": cid, "strategy": strat, "model": "opus", "correct": True,
             "sol_score_calibrated": s, "solution": {"sources": [{"path": "k.py", "content": f"#{cid}\nx=1"}]}}
        (p / "candidates" / f"{cid}.json").write_text(json.dumps(c))
    (p / "journal.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    return p


def test_diagnose_only_stuck_deduped_and_attached(tmp_path, monkeypatch):
    calls = []

    async def fake_call(prompt, model, timeout, cwd=None):
        calls.append((model, prompt))
        return "**Root cause:** memory-bound.\n**Spent:** bf16 cublas.\n**The one lever:** try fp8.", model

    monkeypatch.setattr(diagnose, "_call_fable", fake_call)

    # a regressing (stuck) problem + a climbing one
    _mk_problem(tmp_path, 3, [0.70, 0.55, 0.60, 0.52, 0.58, 0.50],
                ["seed"] + ["bf16 cublas"] * 5, "regressing")
    _mk_problem(tmp_path, 9, [0.40, 0.55, 0.70], ["a", "b", "c"], "climbing")

    refls = R.reflect_all(tmp_path, [3, 9])
    assert refls[3].status == "regressing" and refls[9].status == "climbing"

    n = run(diagnose.diagnose_stuck(tmp_path, refls, model="claude-fable-5"))
    assert n == 1                                   # only the stuck one
    assert {m for m, _ in calls} == {"claude-fable-5"}
    assert len(calls) == 1                          # climbing problem NOT diagnosed
    # diagnosis persisted + attached to the card
    assert (tmp_path / "3" / "diagnosis.json").is_file()
    assert not (tmp_path / "9" / "diagnosis.json").exists()
    card = (tmp_path / "3" / "reflection.md").read_text()
    assert R.DIAGNOSIS_HEADER in card and "one lever" in card

    # second pass, unchanged state → NO new fable spend (dedup on fingerprint)
    refls2 = R.reflect_all(tmp_path, [3, 9])
    run(diagnose.diagnose_stuck(tmp_path, refls2, model="claude-fable-5"))
    assert len(calls) == 1                          # still one — deduped


def test_attach_is_idempotent(tmp_path, monkeypatch):
    async def fake_call(p, m, t, cwd=None):
        return "**The one lever:** try split-K.", m
    monkeypatch.setattr(diagnose, "_call_fable", fake_call)
    _mk_problem(tmp_path, 3, [0.70, 0.55, 0.60, 0.52, 0.58, 0.50], ["seed"] + ["bf16"] * 5, "regressing")
    refls = R.reflect_all(tmp_path, [3])
    run(diagnose.diagnose_stuck(tmp_path, refls, model="fable"))
    R.attach_diagnosis(tmp_path, 3)
    R.attach_diagnosis(tmp_path, 3)                 # twice
    card = (tmp_path / "3" / "reflection.md").read_text()
    assert card.count(R.DIAGNOSIS_HEADER) == 1      # not duplicated


def test_fable_failure_is_graceful(tmp_path, monkeypatch):
    async def fail(p, m, t, cwd=None):
        return None, ""                              # e.g. quota exceeded (all fallbacks too)
    monkeypatch.setattr(diagnose, "_call_fable", fail)
    _mk_problem(tmp_path, 3, [0.70, 0.55, 0.60, 0.52, 0.58, 0.50], ["seed"] + ["bf16"] * 5, "regressing")
    refls = R.reflect_all(tmp_path, [3])
    n = run(diagnose.diagnose_stuck(tmp_path, refls, model="fable"))
    assert n == 0                                   # no diagnosis, no crash
    assert not (tmp_path / "3" / "diagnosis.json").exists()
    # the deterministic card still stands
    assert "REGRESSING" in (tmp_path / "3" / "reflection.md").read_text()


def test_call_fable_falls_back_when_primary_provider_fails(monkeypatch):
    """Reflection must not go dark just because one provider (e.g. claude, rate-
    limited) is unavailable — it should walk FALLBACK_CHAIN to a working one."""
    monkeypatch.setattr(diagnose.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    attempts = []

    async def fake_try(prompt, model, timeout, cwd, env):
        attempts.append((model, env.get("ANTHROPIC_BASE_URL")))
        if env.get("ANTHROPIC_BASE_URL") is None:
            return None                     # primary (native claude) fails — e.g. rate-limited
        return f"diagnosis from {model}"

    monkeypatch.setattr(diagnose, "_try_model", fake_try)
    prose, used = run(diagnose._call_fable("prompt", "claude-sonnet-5", 10))
    assert prose == "diagnosis from deepseek/deepseek-v4-pro"
    assert used == "openrouter:deepseek/deepseek-v4-pro"
    # tried native claude first, then the first fallback that has a key configured
    assert attempts[0] == ("claude-sonnet-5", None)
    assert attempts[1][0] == "deepseek/deepseek-v4-pro"


def test_call_fable_skips_fallback_with_no_api_key(monkeypatch):
    monkeypatch.setattr(diagnose.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    async def always_fail(prompt, model, timeout, cwd, env):
        return None

    monkeypatch.setattr(diagnose, "_try_model", always_fail)
    prose, used = run(diagnose._call_fable("prompt", "claude-sonnet-5", 10))
    assert prose is None and used == ""      # graceful — no crash when every rung is unavailable


def test_markdown_renders_structure():
    h = _md_to_html("## Title\n\n**bold** and `code`\n- one\n- two\n\n```py\nx=1\n```")
    assert "<h4>Title</h4>" in h
    assert "<strong>bold</strong>" in h and "<code>code</code>" in h
    assert "<ul><li>one</li><li>two</li></ul>" in h
    assert "md-code" in h and "x=1" in h


def test_markdown_escapes_html():
    h = _md_to_html("a <script>alert(1)</script> b")
    assert "<script>" not in h and "&lt;script&gt;" in h


def test_live_state_from_active_set():
    from solver.dashboard.metrics import _live_state
    active = {13, 16}
    # terminated always wins
    assert _live_state({"task": 4, "terminated": "budget:time", "evals": 8}, active, True) == "budget:time"
    # fresh active file: in the set = running; not in set = waiting (has evals) / pending (none)
    assert _live_state({"task": 13, "terminated": None, "evals": 5}, active, True) == "running"
    assert _live_state({"task": 28, "terminated": None, "evals": 5}, active, True) == "waiting"
    assert _live_state({"task": 30, "terminated": None, "evals": 0}, active, True) == "pending"


def test_failure_detail_is_actionable():
    from solver.engine.executor import EvalResult, WorkloadResult
    from solver.engine.loop import _failure_detail, _fmt_idxs

    assert _fmt_idxs([0, 1, 2, 3, 7]) == "0-3,7"
    assert _fmt_idxs([]) == "-"

    result = EvalResult(task_id=1, correct=False, sol_score=None, per_workload=[
        WorkloadResult(index=0, correct=False, error="RUNTIME_ERROR"),
        WorkloadResult(index=1, correct=False, error="RUNTIME_ERROR"),
        WorkloadResult(index=2, correct=False, error="TOLERANCE"),
        WorkloadResult(index=3, correct=True),
    ])
    detail = _failure_detail(result)
    assert "3/4 workloads FAILED" in detail
    assert "RUNTIME_ERROR on #0-1" in detail
    assert "TOLERANCE on #2" in detail
    assert "PASSED: #3" in detail

    all_fail = EvalResult(task_id=1, correct=False, sol_score=None, per_workload=[
        WorkloadResult(index=0, correct=False, error="COMPILE_ERROR")])
    assert "ALL workloads failed" in _failure_detail(all_fail)

    ok = EvalResult(task_id=1, correct=True, sol_score=0.5,
                    per_workload=[WorkloadResult(index=0, correct=True)])
    assert _failure_detail(ok) == ""


def test_live_state_recency_fallback():
    import datetime as dt
    from solver.dashboard.metrics import _live_state
    now = dt.datetime.now(dt.timezone.utc)
    recent = (now - dt.timedelta(minutes=3)).isoformat()
    stale = (now - dt.timedelta(minutes=40)).isoformat()
    # no engine active-file (active_fresh=False) → recency of last_ts decides
    assert _live_state({"task": 1, "terminated": None, "evals": 4, "last_ts": recent}, set(), False) == "running"
    assert _live_state({"task": 2, "terminated": None, "evals": 4, "last_ts": stale}, set(), False) == "waiting"
    assert _live_state({"task": 3, "terminated": None, "evals": 0, "last_ts": stale}, set(), False) == "pending"
