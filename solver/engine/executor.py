"""The Executor: the single serialized boundary where candidates are run.

Every evaluation goes through ONE Executor whose `evaluate` is single-flight —
modelling the GPU lock even in the stub, so the loop is written against a
locked, async executor from day one.

- `Executor` — the interface: `async evaluate(solution, task_id) -> EvalResult`.
- `StubExecutor` — no GPU: an async, single-flight stub with a scenario API and
  a re-entrancy assertion (docs/orchestration.md §12 stub contract). Outcomes
  come from a pluggable `outcome` callable: `embedded_outcome` reads scores the
  agent stamped on the candidate (file-free, for unit tests); `metadata_outcome`
  synthesizes from a fetched problem's SOL metadata. The real `GpuQueueExecutor`
  (later) implements the same interface and nothing else in the engine changes.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from ..bench import check as check_mod
from .. import scoring


@dataclass
class WorkloadResult:
    index: int
    correct: bool
    latency_ms: float | None = None
    sol_ms: float | None = None
    baseline_latency_ms: float | None = None
    matched_ratio: float | None = None
    error: str | None = None
    score: float | None = None          # explicit score (stub); else derived from latencies

    @property
    def sol_score(self) -> float | None:
        if self.score is not None:
            return self.score
        if self.latency_ms is None or self.sol_ms is None or self.baseline_latency_ms is None:
            return None
        return scoring.sol_score(self.latency_ms, self.baseline_latency_ms, self.sol_ms)


@dataclass
class EvalResult:
    task_id: int
    correct: bool                       # correct on every workload
    sol_score: float | None             # mean over workloads (None if incorrect / no data)
    per_workload: list[WorkloadResult] = field(default_factory=list)
    asi: dict = field(default_factory=dict)   # actionable side info for reflection
    raw: dict = field(default_factory=dict)

    @property
    def evaluated(self) -> bool:
        """False when the candidate never ran (e.g. failed static check)."""
        return bool(self.per_workload) or self.correct

    def vector(self) -> list[float]:
        """Per-shape score vector for the frontier; non-PASSED shape → 0.0."""
        return [(w.sol_score if (w.correct and w.sol_score is not None) else 0.0)
                for w in self.per_workload]


class Executor(Protocol):
    async def evaluate(self, solution: dict, task_id: int, *, profile: bool = False) -> EvalResult: ...


Outcome = Callable[[dict, int], EvalResult]


def embedded_outcome(solution: dict, task_id: int) -> EvalResult:
    """Read scores the agent stamped on the candidate (`__eval__.scores`).

    `None` in the score list = that shape failed. File-free — the unit-test path.
    """
    scores = (solution.get("__eval__") or {}).get("scores")
    if scores is None:
        scores = [0.5]                  # default: exactly the baseline, one shape
    rows = [
        WorkloadResult(index=i, correct=(s is not None), score=s,
                       error=None if s is not None else "STUB_FAIL")
        for i, s in enumerate(scores)
    ]
    all_passed = all(s is not None for s in scores)
    mean = sum(scores) / len(scores) if all_passed and scores else None
    return EvalResult(task_id=task_id, correct=all_passed, sol_score=mean,
                      per_workload=rows, asi={"stage": "stub", "note": "embedded scores"})


def _baseline_speed_model(task_id: int, solution: dict, index: int, baseline_ms: float) -> float:
    return baseline_ms


def metadata_outcome(problems_dir: Path = Path("problems"),
                     speed_model: Callable[[int, dict, int, float], float] = _baseline_speed_model) -> Outcome:
    """Outcome that synthesizes latencies from a fetched problem's SOL metadata."""
    problems_dir = Path(problems_dir)

    def outcome(solution: dict, task_id: int) -> EvalResult:
        pdir = problems_dir / str(task_id)
        meta = json.loads((pdir / "metadata.json").read_text())
        per = (meta.get("sol") or {}).get("per_workload", []) or []
        rows: list[WorkloadResult] = []
        for w in per:
            base = w.get("baseline_latency_ms")
            latency = (speed_model(task_id, solution, w["index"], base)
                       if base is not None else None)
            rows.append(WorkloadResult(index=w["index"], correct=True, latency_ms=latency,
                                       sol_ms=w.get("sol_ms"), baseline_latency_ms=base,
                                       matched_ratio=1.0))
        scores = [r.sol_score for r in rows if r.sol_score is not None]
        mean = sum(scores) / len(scores) if scores else None
        return EvalResult(task_id=task_id, correct=True, sol_score=mean, per_workload=rows,
                          asi={"stage": "stub", "note": "synthesized from metadata"})

    return outcome


class StubExecutor:
    """GPU-free, async, single-flight Executor for building/testing the engine."""

    def __init__(self, outcome: Outcome | None = None, *, delay: float = 0.0) -> None:
        self._outcome = outcome or embedded_outcome
        self._delay = delay
        self._lock = asyncio.Lock()
        self._in_flight = False          # re-entrancy tripwire (§12 test 7)
        self.calls = 0                   # GPU-equivalent evaluations
        self.max_concurrent = 0          # observed peak; must stay ≤ 1

    async def evaluate(self, solution: dict, task_id: int, *, profile: bool = False) -> EvalResult:
        # NB: the timed window opens *after* the lock is acquired, so `gpu_s` /
        # started..ended measure actual GPU work — never lock-wait. The loop
        # journals these as exec_started/exec_done ts, keeping queue-wait and
        # busy-time cleanly separated on the dashboard.
        async with self._lock:           # single-flight, like the GPU
            assert not self._in_flight, "GPU re-entered: single-flight violated"
            self._in_flight = True
            self.max_concurrent = max(self.max_concurrent, 1)
            try:
                started = dt.datetime.now(dt.timezone.utc)
                t0 = time.perf_counter()
                if self._delay:
                    await asyncio.sleep(self._delay)   # force interleavings in concurrency tests
                self.calls += 1
                result = self._outcome(solution, task_id)
                result.raw = {**(result.raw or {}),
                              "started": _iso(started),
                              "ended": _iso(dt.datetime.now(dt.timezone.utc)),
                              "gpu_s": round(time.perf_counter() - t0, 6)}
                return result
            finally:
                self._in_flight = False


def _iso(t: dt.datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")
