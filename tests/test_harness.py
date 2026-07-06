"""Harness Trace→EvalResult mapping (F2) — real task-230 metadata, fake driver, no GPU.

Proves the §4b metric round-trip: per-shape Traces + the problem's SOL metadata
score into an EvalResult whose per-shape sol_score matches the vendored formula,
whose vector is frontier-safe, and whose failures/compile-errors are handled.
"""

from __future__ import annotations

import json
from pathlib import Path

from solver import scoring
from solver.engine.harness import map_traces_to_result, pod_harness

ROOT = Path("problems")
N230 = len(json.loads((ROOT / "230" / "metadata.json").read_text())["sol"]["per_workload"])


def _traces(status="PASSED", latency=0.001, n=N230):
    return [{"index": i, "status": status, "latency_ms": latency, "matched_ratio": 1.0}
            for i in range(n)]


def test_all_passed_scores_match_the_formula():
    r = map_traces_to_result(230, _traces(latency=0.001), problems_dir=ROOT)
    assert r.correct and r.sol_score is not None
    assert len(r.per_workload) == N230
    per = json.loads((ROOT / "230" / "metadata.json").read_text())["sol"]["per_workload"]
    for wr, sm in zip(r.per_workload, per):
        expected = scoring.sol_score(0.001, sm["baseline_latency_ms"], sm["sol_ms"])
        assert abs(wr.sol_score - expected) < 1e-9
    assert abs(r.sol_score - sum(w.sol_score for w in r.per_workload) / N230) < 1e-9


def test_baseline_latency_scores_half():
    per = json.loads((ROOT / "230" / "metadata.json").read_text())["sol"]["per_workload"]
    traces = [{"index": i, "status": "PASSED", "latency_ms": sm["baseline_latency_ms"],
               "matched_ratio": 1.0} for i, sm in enumerate(per)]
    r = map_traces_to_result(230, traces, problems_dir=ROOT)
    assert all(abs(w.sol_score - 0.5) < 1e-9 for w in r.per_workload)   # matching T_b → 0.5


def test_failed_shape_is_non_scored_and_zero_in_vector():
    traces = _traces()
    traces[3]["status"] = "INCORRECT_NUMERICAL"
    r = map_traces_to_result(230, traces, problems_dir=ROOT)
    assert not r.correct and r.sol_score is None
    assert r.per_workload[3].correct is False and r.per_workload[3].error == "INCORRECT_NUMERICAL"
    assert r.vector()[3] == 0.0 and r.vector()[0] > 0.0                 # specialist signal intact


def test_compile_error_is_frontier_safe():
    r = map_traces_to_result(230, [], solution_status="COMPILE_ERROR", problems_dir=ROOT)
    assert not r.correct and r.sol_score is None
    assert len(r.vector()) == N230 and all(v == 0.0 for v in r.vector())  # full-length zeros
    assert r.asi["solution_status"] == "COMPILE_ERROR"


def test_pod_harness_materializes_and_drives(tmp_path):
    seen = {}

    def fake_driver(workdir, task_id):
        seen["kernel"] = (workdir / "kernel.py").read_text()          # the kernel was written
        return {"traces": _traces(latency=0.0008), "solution_status": None,
                "asi": {"memory": {"peak_alloc_mb": 12.3}}}

    h = pod_harness(problems_dir=ROOT, driver=fake_driver, workdir_root=tmp_path)
    sol = {"sources": [{"path": "kernel.py", "content": "def run(*t): return t[-1]"}]}
    r = h(sol, 230)
    assert "def run" in seen["kernel"]
    assert r.correct and r.sol_score is not None
    assert r.asi["memory"]["peak_alloc_mb"] == 12.3                    # Tier-1 asi passes through
