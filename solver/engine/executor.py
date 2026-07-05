"""The Executor: the single serialized boundary where candidates are run.

Every candidate evaluation in the engine goes through ONE Executor instance
whose `evaluate` is serialized (concurrency 1) — modelling the GPU lock even
in the stub, so the loop is written against a locked executor from day one.

- `Executor` — the interface. `evaluate(solution, task_id)` -> `EvalResult`.
- `StubExecutor` — no GPU: runs the static `check`, then synthesizes latencies
  from the problem's SOL metadata so the loop, scoring, and transfer can be
  built and tested deterministically. The real `GpuQueueExecutor` (later, a
  single-flight queue to the harness on a GPU box) implements the same
  interface, and nothing else in the engine changes.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .. import check as check_mod
from .. import scoring


@dataclass
class WorkloadResult:
    index: int
    correct: bool
    latency_ms: float | None
    sol_ms: float | None
    baseline_latency_ms: float | None
    matched_ratio: float | None = None
    error: str | None = None

    @property
    def sol_score(self) -> float | None:
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
    raw: dict = field(default_factory=dict)   # full underlying payload

    @property
    def evaluated(self) -> bool:
        """False when the candidate never ran (e.g. failed static check)."""
        return bool(self.per_workload) or self.correct


class Executor(Protocol):
    def evaluate(self, solution: dict, task_id: int, *, profile: bool = False) -> EvalResult: ...


# A speed model maps (task_id, solution, workload_index, baseline_ms) -> latency_ms.
SpeedModel = Callable[[int, dict, int, float], float]


def _baseline_speed_model(task_id: int, solution: dict, index: int, baseline_ms: float) -> float:
    """Default stub speed: the candidate is exactly the baseline (score ~0.5)."""
    return baseline_ms


class StubExecutor:
    """GPU-free Executor for building/testing the engine.

    Serializes like the real one (a lock), validates via `solver.check`, and
    synthesizes per-workload latencies with an injectable `speed_model` so
    loop dynamics (accept/reject, Pareto, termination) can be exercised
    deterministically.
    """

    def __init__(
        self,
        problems_dir: Path = Path("problems"),
        *,
        speed_model: SpeedModel = _baseline_speed_model,
    ) -> None:
        self.problems_dir = Path(problems_dir)
        self.speed_model = speed_model
        self._lock = threading.Lock()
        self.calls = 0  # count of GPU-equivalent evaluations

    def evaluate(self, solution: dict, task_id: int, *, profile: bool = False) -> EvalResult:
        with self._lock:  # single-flight, like the GPU
            return self._evaluate(solution, task_id)

    def _evaluate(self, solution: dict, task_id: int) -> EvalResult:
        pdir = self.problems_dir / str(task_id)
        definition = json.loads((pdir / "definition.json").read_text())
        meta = json.loads((pdir / "metadata.json").read_text())

        # Static check first — a failure never reaches the GPU, so it does not
        # consume a GPU-evaluation from the budget.
        report = check_mod.check_solution(solution, definition)
        if not report.ok:
            return EvalResult(
                task_id=task_id,
                correct=False,
                sol_score=None,
                asi={"stage": "static_check", "check_errors": report.errors,
                     "check_warnings": report.warnings},
            )

        self.calls += 1  # a real (GPU-equivalent) run
        per = (meta.get("sol") or {}).get("per_workload", []) or []
        rows: list[WorkloadResult] = []
        for w in per:
            base = w.get("baseline_latency_ms")
            sol_ms = w.get("sol_ms")
            latency = (
                self.speed_model(task_id, solution, w["index"], base)
                if base is not None
                else None
            )
            rows.append(
                WorkloadResult(
                    index=w["index"],
                    correct=True,
                    latency_ms=latency,
                    sol_ms=sol_ms,
                    baseline_latency_ms=base,
                    matched_ratio=1.0,
                )
            )
        scores = [r.sol_score for r in rows if r.sol_score is not None]
        mean = sum(scores) / len(scores) if scores else None
        return EvalResult(
            task_id=task_id,
            correct=True,
            sol_score=mean,
            per_workload=rows,
            asi={"stage": "stub", "note": "synthesized latencies; no GPU",
                 "check_warnings": report.warnings},
        )
