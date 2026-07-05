"""Orchestrator engine for optimizing SOL-ExecBench problems.

See docs/orchestrator-engine.md. Phase A (this): the Executor abstraction —
the single serialized GPU-lock boundary — plus a StubExecutor for building and
testing the loop without a GPU.
"""

from .executor import EvalResult, Executor, StubExecutor, WorkloadResult

__all__ = ["EvalResult", "Executor", "StubExecutor", "WorkloadResult"]
