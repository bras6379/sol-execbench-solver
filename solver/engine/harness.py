"""The pod-side harness wrapper + Trace→EvalResult mapping (Phase F2).

docs/gpu-execution.md §4b/§4. The worker turns a candidate into an `EvalResult`:
materialize the kernel → run the SOL-ExecBench harness (build_ext + eval_driver)
→ per-shape Traces → **score against the problem's SOL metadata** (`sol_ms`,
`baseline_latency_ms`) → `EvalResult`. The harness call is behind an injectable
`driver` so the pure mapping is testable on the laptop (real metadata + a fake
driver, no GPU); the real driver runs on the pod against the `[bench]` harness.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable

from .executor import EvalResult, WorkloadResult

# driver(workdir, task_id) -> {"traces":[{index,status,latency_ms,matched_ratio}],
#                              "solution_status": COMPILE_ERROR|REWARD_HACK|None, "asi": {...}}
Driver = Callable[[Path, int], dict]


def _sol_per_workload(task_id: int, problems_dir: Path) -> list[dict]:
    meta = json.loads((problems_dir / str(task_id) / "metadata.json").read_text())
    return ((meta.get("sol") or {}).get("per_workload") or [])


def map_traces_to_result(task_id: int, traces: list[dict], *, solution_status: str | None = None,
                         asi: dict | None = None, problems_dir: str | Path = "problems") -> EvalResult:
    """Score per-shape Traces against the problem's SOL metadata → EvalResult.

    Always emits one `WorkloadResult` per metadata shape (missing/failed shapes
    score 0 in the frontier vector), so the result is frontier-safe even for a
    COMPILE_ERROR / all-failed candidate.
    """
    per = _sol_per_workload(task_id, Path(problems_dir))
    by_index = {t["index"]: t for t in traces}
    rows: list[WorkloadResult] = []
    for sm in per:
        i = sm["index"]
        t = by_index.get(i)
        passed = solution_status is None and t is not None and t.get("status") == "PASSED"
        rows.append(WorkloadResult(
            index=i, correct=passed,
            latency_ms=(t or {}).get("latency_ms"),
            sol_ms=sm.get("sol_ms"), baseline_latency_ms=sm.get("baseline_latency_ms"),
            matched_ratio=(t or {}).get("matched_ratio"),
            error=None if passed else (solution_status or (t or {}).get("status") or "NO_TRACE")))
    all_passed = solution_status is None and len(rows) > 0 and all(r.correct for r in rows)
    scores = [r.sol_score for r in rows if r.correct and r.sol_score is not None]
    mean = sum(scores) / len(scores) if all_passed and scores else None
    a = {"stage": "harness", **(asi or {})}
    if solution_status:
        a["solution_status"] = solution_status
    return EvalResult(task_id=task_id, correct=all_passed, sol_score=mean, per_workload=rows, asi=a)


def pod_harness(problems_dir: str | Path = "problems", *, driver: Driver,
                workdir_root: str | Path | None = None):
    """A `Harness` (solution, task_id) -> EvalResult for the gpu Worker: materialize
    the kernel into a workdir, run `driver` (real eval_driver on the pod / a fake
    in tests), map its Traces + metadata → EvalResult."""
    problems_dir = Path(problems_dir)
    root = Path(workdir_root) if workdir_root else None

    def harness(solution: dict, task_id: int) -> EvalResult:
        base = root / str(task_id) if root else Path(tempfile.mkdtemp(prefix="eval-"))
        base.mkdir(parents=True, exist_ok=True)
        for s in solution.get("sources", []):
            (base / s["path"]).write_text(s.get("content", ""))
        out = driver(base, task_id) or {}
        return map_traces_to_result(task_id, out.get("traces", []),
                                    solution_status=out.get("solution_status"),
                                    asi=out.get("asi"), problems_dir=problems_dir)

    return harness


# The real pod-side driver (F2, runs on the GPU against the `[bench]` harness) is
# written when a pod is available — it invokes build_ext then eval_driver over the
# problem's workloads and returns Traces + the Tier-1 asi (§8c). Its exact call
# shape is pinned against the installed sol-execbench version. Everything above is
# GPU-free and covered by tests/test_harness.py.
