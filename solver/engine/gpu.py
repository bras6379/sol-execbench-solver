"""GpuQueueExecutor (Phase F1) — the real Executor's transport + durability core.

Runs candidates through a file-queue to a worker (docs/gpu-execution.md). Now:
a **LocalPod** — a `FileQueueTransport` (jobs/ + results/ dirs) + a `Worker`
that runs a fake harness on localhost, no GPU. Later (F2): the same
`GpuQueueExecutor` against an SSH-synced dir + a real `eval_driver` worker on a
rented B200 — no engine change.

Durability = **hash-keyed idempotent jobs**: `job_id = <task>-<Solution.hash>`,
and both the executor (before push) and the worker (before run) reuse an
existing `results/<job_id>` instead of recomputing. So recovery is free — a
laptop crash mid-eval re-evaluates the same candidate → same job_id → the cached
result is recovered, never re-run; a pod death → the result is gone → re-run.
All of that is provable on the laptop with the fake harness (tests/test_gpu.py).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Callable, Protocol

from .agent import solution_hash
from .executor import EvalResult, WorkloadResult

Harness = Callable[[dict, int], EvalResult]        # (solution, task_id) -> EvalResult


# --------------------------------------------------------------------------- #
# EvalResult <-> JSON  (the results/<job>.json contract, docs §4)
# --------------------------------------------------------------------------- #
def eval_result_to_dict(r: EvalResult) -> dict:
    return {"task_id": r.task_id, "correct": r.correct, "sol_score": r.sol_score,
            "per_workload": [dict(vars(w)) for w in r.per_workload],
            "asi": r.asi, "raw": r.raw}


def eval_result_from_dict(d: dict) -> EvalResult:
    rows = [WorkloadResult(**w) for w in d.get("per_workload", [])]
    return EvalResult(task_id=d["task_id"], correct=d["correct"], sol_score=d.get("sol_score"),
                      per_workload=rows, asi=d.get("asi", {}), raw=d.get("raw", {}))


def _write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)                                  # atomic publish


def _iso(t: dt.datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Transport — the laptop↔pod boundary (LocalPod now; SSH-synced dir later)
# --------------------------------------------------------------------------- #
class Transport(Protocol):
    async def push(self, job_id: str, payload: dict) -> None: ...
    async def result(self, job_id: str) -> dict | None: ...


class FileQueueTransport:
    """Dir-based jobs/results. Localhost for F1; an SSH-synced dir for F2 —
    same interface, so `GpuQueueExecutor` doesn't change."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        (self.root / "results").mkdir(parents=True, exist_ok=True)

    async def push(self, job_id: str, payload: dict) -> None:
        jf = self.root / "jobs" / f"{job_id}.json"
        rf = self.root / "results" / f"{job_id}.json"
        if jf.exists() or rf.exists():                 # idempotent — never duplicate a job
            return
        _write_json(jf, payload)

    async def result(self, job_id: str) -> dict | None:
        rf = self.root / "results" / f"{job_id}.json"
        if not rf.exists():
            return None
        try:
            return json.loads(rf.read_text())
        except json.JSONDecodeError:                   # mid-write; try next poll
            return None


# --------------------------------------------------------------------------- #
# Worker — the pod side. F1: runs a fake harness in-process for tests.
# (F2's real worker is a pod-side process wrapping build_ext + eval_driver.)
# --------------------------------------------------------------------------- #
class Worker:
    def __init__(self, root: str | Path, harness: Harness) -> None:
        self.root = Path(root)
        self.harness = harness

    def run_pending(self) -> int:
        jobs, results = self.root / "jobs", self.root / "results"
        n = 0
        for jf in sorted(jobs.glob("*.json")):
            job_id = jf.stem
            rf = results / f"{job_id}.json"
            if rf.exists():                            # idempotent: already done
                continue
            try:
                payload = json.loads(jf.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                continue
            t0 = time.monotonic()
            r = self.harness(payload["solution"], payload["task_id"])
            d = eval_result_to_dict(r)
            d.setdefault("raw", {})["gpu_s"] = round(time.monotonic() - t0, 6)
            _write_json(rf, d)                         # durable
            n += 1
        return n

    async def serve(self, stop: asyncio.Event, interval: float = 0.005) -> None:
        while not stop.is_set():
            self.run_pending()
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass


# --------------------------------------------------------------------------- #
# GpuQueueExecutor — implements the Executor interface (async evaluate)
# --------------------------------------------------------------------------- #
class GpuQueueExecutor:
    def __init__(self, transport: Transport, *, poll_interval: float = 0.01,
                 timeout: float = 600.0) -> None:
        self.transport = transport
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._lock = asyncio.Lock()                    # single-flight (the GPU)
        self._in_flight = False
        self.max_concurrent = 0

    async def evaluate(self, solution: dict, task_id: int, *, profile: bool = False) -> EvalResult:
        job_id = f"{task_id}-{solution_hash(solution)[:12]}"
        async with self._lock:
            assert not self._in_flight, "GPU re-entered: single-flight violated"
            self._in_flight = True
            self.max_concurrent = max(self.max_concurrent, 1)
            try:
                started = dt.datetime.now(dt.timezone.utc)
                await self.transport.push(job_id, {"task_id": task_id, "solution": solution})
                d = await self._poll(job_id)
                r = eval_result_from_dict(d)
                r.raw = {**(r.raw or {}), "job_id": job_id, "started": _iso(started),
                         "ended": _iso(dt.datetime.now(dt.timezone.utc))}
                r.raw.setdefault("gpu_s", 0.0)
                return r
            finally:
                self._in_flight = False

    async def _poll(self, job_id: str) -> dict:
        t0 = time.monotonic()
        while True:
            d = await self.transport.result(job_id)
            if d is not None:
                return d
            if time.monotonic() - t0 > self.timeout:
                raise RuntimeError(f"GPU eval stuck (job {job_id}, >{self.timeout}s)")
            await asyncio.sleep(self.poll_interval)
