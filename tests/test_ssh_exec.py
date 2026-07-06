"""SshExecutor + harness-format helpers (F2b) — offline, against a fake pod.

The real end-to-end (provision → bootstrap → eval on a B200) was validated live;
these lock the pure control flow: candidate → solution.json, trace.jsonl → score,
single-flight, idempotent job reuse, and the no-trace → COMPILE_ERROR path — all
driven by a `FakePodConn` (an in-memory stand-in for `PodConn`), no GPU/SSH.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from solver.engine.gpu_run import harness_ref
from solver.engine.harness import (map_traces_to_result, solution_to_harness_json,
                                    traces_from_jsonl)
from solver.engine.loop import reference_seed
from solver.engine.ssh_exec import SshExecutor

ROOT = Path("problems")
FIXTURE = Path("tests/fixtures/trace_230.jsonl").read_text()
TRITON = {"spec": {"languages": ["triton"], "entry_point": "kernel.py::rmsnorm_fwd",
                   "destination_passing_style": True},
          "sources": [{"path": "kernel.py", "content": "def rmsnorm_fwd(x,w,o): ..."}]}


# ----------------------------- builder + parser ----------------------------- #
def test_solution_builder_fills_harness_fields():
    seed = reference_seed(ROOT)(230)[0]
    sol = solution_to_harness_json(seed, 230, ROOT)
    assert sol["definition"] == "021_rmsnorm_h128"          # == definition.json name
    assert sol["spec"]["entry_point"] == "reference.py::run"
    assert sol["spec"]["target_hardware"] == ["B200", "LOCAL"]
    assert sol["spec"]["dependencies"] == ["torch"]
    assert sol["name"] and sol["author"]                    # NonEmptyString fields present


def test_solution_builder_respects_spec_overrides():
    sol = solution_to_harness_json(TRITON, 230, ROOT)
    assert sol["spec"]["languages"] == ["triton"]
    assert sol["spec"]["entry_point"] == "kernel.py::rmsnorm_fwd"
    assert sol["spec"]["destination_passing_style"] is True
    assert set(sol["spec"]["dependencies"]) == {"torch", "triton"}


def test_solution_builder_rejects_empty_sources():
    try:
        solution_to_harness_json({"spec": {}, "sources": []}, 230, ROOT)
        assert False, "should reject empty sources"
    except ValueError:
        pass


def test_parse_real_trace_scores_match_map():
    rows = traces_from_jsonl(FIXTURE)
    assert len(rows) == 14 and {r["status"] for r in rows} == {"PASSED"}
    assert [r["index"] for r in rows] == list(range(14))    # positional == metadata index
    r = map_traces_to_result(230, rows, problems_dir=ROOT)
    assert r.correct and r.sol_score is not None and len(r.vector()) == 14


def test_harness_ref_reads_pinned_sha():
    sha = harness_ref("pyproject.toml")
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


# --------------------------- SshExecutor control flow ----------------------- #
class FakePodConn:
    """In-memory pod: `run(run_eval.sh ...)` writes the fixture trace to <out>."""

    def __init__(self, trace=FIXTURE, *, runs=None):
        self.fs: dict[str, str] = {}
        self.trace = trace
        self.run_calls = 0
        self._runs = runs

    async def exists(self, path):
        return path in self.fs

    async def write_text(self, path, text):
        self.fs[path] = text

    async def read_text(self, path):
        return self.fs.get(path)

    async def mkdir(self, path):
        pass

    async def run(self, cmd, *, timeout=600, stdin_data=None):
        if "run_eval.sh" in cmd:
            self.run_calls += 1
            out = cmd.split()[-1]                            # last arg is the trace path
            if self.trace is not None:
                self.fs[out] = self.trace                    # the harness "produced" a trace
        return (0, "", "")


def _run(coro):
    return asyncio.run(coro)


def test_ssh_executor_scores_a_candidate():
    conn = FakePodConn()
    ex = SshExecutor(conn, problems_dir=ROOT)
    r = _run(ex.evaluate(reference_seed(ROOT)(230)[0], 230))
    assert r.correct and r.sol_score is not None
    assert len(r.per_workload) == 14 and ex.max_concurrent == 1
    assert conn.run_calls == 1                               # ran the harness once
    assert r.raw["job_id"].startswith("230-")


def test_ssh_executor_is_idempotent():
    conn = FakePodConn()
    ex = SshExecutor(conn, problems_dir=ROOT)
    seed = reference_seed(ROOT)(230)[0]
    _run(ex.evaluate(seed, 230))
    _run(ex.evaluate(seed, 230))                            # 2nd time: trace already there
    assert conn.run_calls == 1                              # never re-ran the harness


def test_ssh_executor_no_trace_is_compile_error():
    conn = FakePodConn(trace=None)                          # harness produced nothing
    ex = SshExecutor(conn, problems_dir=ROOT)
    r = _run(ex.evaluate(reference_seed(ROOT)(230)[0], 230))
    assert not r.correct and r.sol_score is None
    assert r.asi.get("solution_status") == "COMPILE_ERROR"
    assert len(r.vector()) == 14 and all(v == 0.0 for v in r.vector())  # frontier-safe
