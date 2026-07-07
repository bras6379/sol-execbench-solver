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
        return "**Root cause:** memory-bound.\n**Spent:** bf16 cublas.\n**The one lever:** try fp8."

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
        return "**The one lever:** try split-K."
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
        return None                                 # e.g. quota exceeded
    monkeypatch.setattr(diagnose, "_call_fable", fail)
    _mk_problem(tmp_path, 3, [0.70, 0.55, 0.60, 0.52, 0.58, 0.50], ["seed"] + ["bf16"] * 5, "regressing")
    refls = R.reflect_all(tmp_path, [3])
    n = run(diagnose.diagnose_stuck(tmp_path, refls, model="fable"))
    assert n == 0                                   # no diagnosis, no crash
    assert not (tmp_path / "3" / "diagnosis.json").exists()
    # the deterministic card still stands
    assert "REGRESSING" in (tmp_path / "3" / "reflection.md").read_text()


def test_markdown_renders_structure():
    h = _md_to_html("## Title\n\n**bold** and `code`\n- one\n- two\n\n```py\nx=1\n```")
    assert "<h4>Title</h4>" in h
    assert "<strong>bold</strong>" in h and "<code>code</code>" in h
    assert "<ul><li>one</li><li>two</li></ul>" in h
    assert "md-code" in h and "x=1" in h


def test_markdown_escapes_html():
    h = _md_to_html("a <script>alert(1)</script> b")
    assert "<script>" not in h and "&lt;script&gt;" in h
