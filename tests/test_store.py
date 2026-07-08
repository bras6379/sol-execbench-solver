"""Durable candidate + frontier persistence (store.py) — no GPU.

Proves solve_problem's artifacts: every candidate gets a full record with a
ready-to-submit harness solution.json; the frontier + best_solution.json are
written and point back at the candidate files; stub/source-less candidates don't
break the store.
"""

from __future__ import annotations

import json
from pathlib import Path

from solver.engine import store
from solver.engine.executor import EvalResult, WorkloadResult
from solver.engine.frontier import Frontier, Member
from solver.engine.harness import map_traces_to_result, traces_from_jsonl

FIXTURE = Path("tests/fixtures/trace_230.jsonl").read_text()
TRITON = {"spec": {"languages": ["triton"], "entry_point": "kernel.py::run"},
          "sources": [{"path": "kernel.py", "content": "def run(x, w):\n    return x"}]}


def _real_result():
    return map_traces_to_result(230, traces_from_jsonl(FIXTURE), problems_dir="problems")


def test_record_candidate_writes_record_index_and_submit(tmp_path):
    r = _real_result()
    store.record_candidate(tmp_path, 230, "abc123def456", TRITON, r,
                           strategy="fused", agent="claude", model="haiku",
                           verdict="entered", problems_dir="problems")
    rec = json.loads((tmp_path / "230/candidates/abc123def456.json").read_text())
    assert rec["correct"] and abs(rec["sol_score"] - r.sol_score) < 1e-9
    assert len(rec["per_workload"]) == 14 and len(rec["vector"]) == 14
    assert rec["solution"]["sources"]                              # raw candidate kept
    assert rec["submit"]["definition"] == "021_rmsnorm_h128"       # harness-format, submittable
    assert rec["submit"]["spec"]["entry_point"] == "kernel.py::run"
    lines = (tmp_path / "230/candidates/index.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["cand_id"] == "abc123def456"


def test_record_candidate_persists_the_per_workload_detail(tmp_path):
    """The harness's actual diagnostic (a Triton/CUDA traceback line, or the
    measured error magnitude — see WorkloadResult.detail) must survive into the
    archived candidate record, not just the bare status code — otherwise a
    later investigation into "why did this fail" has nothing to read (confirmed
    live, 2026-07-08: this field existed on WorkloadResult but store.py's
    explicit field list silently dropped it when persisting)."""
    per = [WorkloadResult(index=0, correct=False, error="RUNTIME_ERROR",
                          detail="User function failed: at 22:11: BLOCK_H undefined")]
    r = EvalResult(task_id=230, correct=False, sol_score=None, per_workload=per)
    store.record_candidate(tmp_path, 230, "failcid", TRITON, r,
                           strategy="fused", verdict="dominated", problems_dir="problems")
    rec = json.loads((tmp_path / "230/candidates/failcid.json").read_text())
    assert rec["per_workload"][0]["detail"] == "User function failed: at 22:11: BLOCK_H undefined"


def test_record_candidate_index_is_idempotent(tmp_path):
    r = _real_result()
    for _ in range(3):                                            # e.g. a resume re-persists
        store.record_candidate(tmp_path, 230, "dup", TRITON, r, problems_dir="problems")
    lines = (tmp_path / "230/candidates/index.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1                                        # indexed once


def test_sourceless_candidate_does_not_crash(tmp_path):
    r = EvalResult(task_id=1, correct=True, sol_score=0.5,
                   per_workload=[WorkloadResult(index=0, correct=True, score=0.5)])
    store.record_candidate(tmp_path, 1, "stub", {"__eval__": {"scores": [0.5]}}, r,
                           problems_dir="problems")
    rec = json.loads((tmp_path / "1/candidates/stub.json").read_text())
    assert rec["submit"] is None                                 # nothing to submit, but stored


def test_record_frontier_writes_frontier_and_best(tmp_path):
    r = _real_result()
    # a weaker seed and the stronger triton candidate, both persisted first
    store.record_candidate(tmp_path, 230, "seedcid", TRITON, _half_result(), strategy="seed",
                           verdict="entered", problems_dir="problems")
    store.record_candidate(tmp_path, 230, "bestcid", TRITON, r, strategy="fused",
                           verdict="entered", problems_dir="problems")
    fr = Frontier(0.02)
    fr.accept(Member("seedcid", tuple([0.5] * 14), True, strategy="seed"))
    fr.accept(Member("bestcid", tuple(r.vector()), True, strategy="fused"))
    store.record_frontier(tmp_path, 230, fr, problems_dir="problems", family="rmsnorm", name="rmsnorm_230")

    front = json.loads((tmp_path / "230/frontier.json").read_text())
    assert front["size"] == len(fr.members) and front["family"] == "rmsnorm"
    assert front["members"][0]["candidate"].startswith("candidates/")
    assert front["best_cand"] == fr.best().cand_id
    best = json.loads((tmp_path / "230/best_solution.json").read_text())
    assert best["definition"] == "021_rmsnorm_h128"              # the submittable best


def _half_result():
    per = [WorkloadResult(index=i, correct=True, score=0.5) for i in range(14)]
    return EvalResult(task_id=230, correct=True, sol_score=0.5, per_workload=per)
