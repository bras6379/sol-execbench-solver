"""Orchestrator engine for optimizing SOL-ExecBench problems.

See docs/orchestration.md. Phase A: the Executor abstraction (the single
serialized GPU-lock boundary). Phase B: the async solve loop — RunContext +
journal replay/resume, the ε-Pareto frontier, the tier ladder with
headroom-gated escalation, the novelty gates, and the fleet. Everything runs
against StubExecutor + StubAgent (no GPU, no API); see the §12 stub contract.
"""

from .agent import Agent, Candidate, StubAgent, solution_hash, stub_agents
from .cli_agent import SPECS, CliAgent, CliSpec, make_agents
from .config import Config, Perspective, Tier, default_config
from .context import RunContext
from .executor import (
    EvalResult,
    Executor,
    StubExecutor,
    WorkloadResult,
    embedded_outcome,
    metadata_outcome,
)
from .frontier import Frontier, Member
from .gpu import FileQueueTransport, GpuQueueExecutor, Worker
from .knowledge import KnowledgeStore
from .loop import exemplar_first, reference_seed, run_fleet, solve_problem

__all__ = [
    "Agent", "Candidate", "StubAgent", "solution_hash", "stub_agents",
    "CliAgent", "CliSpec", "SPECS", "make_agents",
    "Config", "Perspective", "Tier", "default_config",
    "RunContext",
    "EvalResult", "Executor", "StubExecutor", "WorkloadResult",
    "embedded_outcome", "metadata_outcome",
    "Frontier", "Member",
    "FileQueueTransport", "GpuQueueExecutor", "Worker",
    "KnowledgeStore",
    "exemplar_first", "reference_seed", "run_fleet", "solve_problem",
]
