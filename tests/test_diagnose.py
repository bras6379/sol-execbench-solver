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
        return ("**Root cause:** memory-bound.\n**Spent:** bf16 cublas.\n**The one lever:** try fp8.",
                model, {"cost_usd": 0.01})

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
        return "**The one lever:** try split-K.", m, {"cost_usd": 0.01}
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
        return None, "", {}                          # e.g. quota exceeded (all fallbacks too)
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
            return None, {}                  # primary (native claude) fails — e.g. rate-limited
        return f"diagnosis from {model}", {"cost_usd": 0.02}

    monkeypatch.setattr(diagnose, "_try_model", fake_try)
    prose, used, usage = run(diagnose._call_fable("prompt", "claude-sonnet-5", 10))
    assert prose == "diagnosis from deepseek/deepseek-v4-pro"
    assert used == "openrouter:deepseek/deepseek-v4-pro"
    assert usage["cost_usd"] == 0.02
    # tried native claude first, then the first fallback that has a key configured
    assert attempts[0] == ("claude-sonnet-5", None)
    assert attempts[1][0] == "deepseek/deepseek-v4-pro"


def test_parse_reflect_model_splits_agent_prefix():
    assert diagnose._parse_reflect_model("openrouter/z-ai/glm-4.7-flash") == \
        ("openrouter", "z-ai/glm-4.7-flash")
    assert diagnose._parse_reflect_model("deepseek/deepseek-v4-pro") == \
        ("deepseek", "deepseek-v4-pro")
    assert diagnose._parse_reflect_model("claude-sonnet-5") == ("claude", "claude-sonnet-5")
    assert diagnose._parse_reflect_model("sonnet") == ("claude", "sonnet")
    # codex is not a valid reflect spec — falls through to a literal (invalid) claude
    # model name rather than silently routing to the wrong CLI binary
    assert diagnose._parse_reflect_model("codex/gpt-5.5") == ("claude", "codex/gpt-5.5")


def test_call_fable_routes_to_the_actually_requested_model(monkeypatch):
    """Regression: --reflect-model with an OpenRouter model NOT in FALLBACK_CHAIN
    used to be silently discarded — the fallback rungs used FALLBACK_CHAIN's own
    hardcoded model names instead of the one the caller actually asked for."""
    monkeypatch.setattr(diagnose.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    attempts = []

    async def fake_try(prompt, model, timeout, cwd, env):
        attempts.append((model, env.get("ANTHROPIC_BASE_URL")))
        return f"diagnosis from {model}", {"cost_usd": 0.01}

    monkeypatch.setattr(diagnose, "_try_model", fake_try)
    prose, used, usage = run(diagnose._call_fable("prompt", "openrouter/z-ai/glm-4.7-flash", 10))
    assert attempts[0][0] == "z-ai/glm-4.7-flash"          # the ACTUAL requested model, not a
    assert used == "openrouter:z-ai/glm-4.7-flash"          # FALLBACK_CHAIN placeholder
    assert prose == "diagnosis from z-ai/glm-4.7-flash"


def test_diagnose_stuck_rotates_across_a_model_pool(tmp_path, monkeypatch):
    """A --reflect-model POOL rotates round-robin over the stuck list, so a
    problem still stuck after one model's take gets a different one next time."""
    seen_models = []

    async def fake_diagnose_one(runs_dir, r, *, model, timeout, kb_dir="kb", log=lambda *_: None):
        seen_models.append((r.task_id, model))
        return True

    monkeypatch.setattr(diagnose, "diagnose_one", fake_diagnose_one)
    monkeypatch.setattr(R, "attach_diagnosis", lambda *a, **k: None)
    refls = {
        t: R.ProblemReflection(task_id=t, status="plateaued", name="", family="",
                               headline="", n_evals=1, best=0.5)
        for t in (1, 2, 3, 4)
    }
    run(diagnose.diagnose_stuck(tmp_path, refls, model=["model-a", "model-b"]))
    picked = dict(seen_models)
    assert picked[1] == "model-a" and picked[2] == "model-b"
    assert picked[3] == "model-a" and picked[4] == "model-b"


def test_call_fable_skips_fallback_with_no_api_key(monkeypatch):
    monkeypatch.setattr(diagnose.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    async def always_fail(prompt, model, timeout, cwd, env):
        return None, {}

    monkeypatch.setattr(diagnose, "_try_model", always_fail)
    prose, used, usage = run(diagnose._call_fable("prompt", "claude-sonnet-5", 10))
    assert prose is None and used == "" and not usage     # graceful — no crash, no rungs available


def test_diagnose_cost_is_persisted_and_journaled(tmp_path, monkeypatch):
    """Real $ cost must survive to both diagnosis.json (latest) and the problem's
    own journal.jsonl (cumulative, dashboard-scannable) — on success AND failure,
    since a failed-but-billed call (e.g. 402 mid-fallback) must still count."""
    async def fake_call(prompt, model, timeout, cwd=None):
        return "**The one lever:** try split-K.", model, {"cost_usd": 0.05, "in": 100, "out": 40, "cached": 10}
    monkeypatch.setattr(diagnose, "_call_fable", fake_call)
    _mk_problem(tmp_path, 3, [0.70, 0.55, 0.60, 0.52, 0.58, 0.50], ["seed"] + ["bf16"] * 5, "regressing")
    refls = R.reflect_all(tmp_path, [3])
    run(diagnose.diagnose_stuck(tmp_path, refls, model="fable"))

    d = json.loads((tmp_path / "3" / "diagnosis.json").read_text())
    assert d["cost_usd"] == 0.05
    assert d["tok_in"] == 100 and d["tok_out"] == 40 and d["tok_cached"] == 10
    lines = [json.loads(l) for l in (tmp_path / "3" / "journal.jsonl").read_text().splitlines()]
    cost_ev = [e for e in lines if e.get("ev") == "diagnose_cost"]
    assert len(cost_ev) == 1 and cost_ev[0]["cost_usd"] == 0.05 and cost_ev[0]["success"] is True


def test_diagnose_failed_but_billed_cost_still_journaled(tmp_path, monkeypatch):
    async def fake_call(prompt, model, timeout, cwd=None):
        return None, "", {"cost_usd": 0.03}   # e.g. a 402 that still billed a partial request
    monkeypatch.setattr(diagnose, "_call_fable", fake_call)
    _mk_problem(tmp_path, 3, [0.70, 0.55, 0.60, 0.52, 0.58, 0.50], ["seed"] + ["bf16"] * 5, "regressing")
    refls = R.reflect_all(tmp_path, [3])
    n = run(diagnose.diagnose_stuck(tmp_path, refls, model="fable"))

    assert n == 0                       # no successful diagnosis
    assert not (tmp_path / "3" / "diagnosis.json").exists()
    lines = [json.loads(l) for l in (tmp_path / "3" / "journal.jsonl").read_text().splitlines()]
    cost_ev = [e for e in lines if e.get("ev") == "diagnose_cost"]
    assert len(cost_ev) == 1 and cost_ev[0]["cost_usd"] == 0.03 and cost_ev[0]["success"] is False


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


def test_failure_detail_includes_the_harness_diagnostic_not_just_the_status_code():
    """Confirmed live (2026-07-08): the real harness reports an actual Triton/CUDA
    traceback (trace.jsonl's `log`) or the measured error magnitude for a failed
    workload — this was being parsed then discarded, so the next agent only ever
    saw a bare 'RUNTIME_ERROR' with nothing to act on. Must surface a real
    example, not just the status."""
    from solver.engine.executor import EvalResult, WorkloadResult
    from solver.engine.loop import _failure_detail

    result = EvalResult(task_id=1, correct=False, sol_score=None, per_workload=[
        WorkloadResult(index=0, correct=False, error="RUNTIME_ERROR",
                       detail="User function failed: at 22:11: BLOCK_H undefined"),
        WorkloadResult(index=1, correct=False, error="RUNTIME_ERROR",
                       detail="User function failed: at 22:11: BLOCK_H undefined"),
        WorkloadResult(index=2, correct=False, error="INCORRECT_NUMERICAL",
                       detail="max_abs_error=0.4, max_rel_error=0.23"),
        WorkloadResult(index=3, correct=True),
    ])
    detail = _failure_detail(result)
    assert "User function failed: at 22:11: BLOCK_H undefined" in detail
    assert "max_abs_error=0.4, max_rel_error=0.23" in detail


def test_failure_detail_falls_back_to_the_compile_error_stderr_snippet():
    """COMPILE_ERROR fails before any workload runs, so there's no per-workload
    trace/log — the only diagnostic available is the SSH-captured stderr stored
    on the EvalResult itself (asi.error, see ssh_exec.py)."""
    from solver.engine.executor import EvalResult, WorkloadResult
    from solver.engine.loop import _failure_detail

    result = EvalResult(task_id=1, correct=False, sol_score=None,
                        asi={"error": "nvcc: error: identifier \"foo\" is undefined"},
                        per_workload=[
                            WorkloadResult(index=0, correct=False, error="COMPILE_ERROR"),
                            WorkloadResult(index=1, correct=False, error="COMPILE_ERROR"),
                        ])
    detail = _failure_detail(result)
    assert 'nvcc: error: identifier "foo" is undefined' in detail


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
