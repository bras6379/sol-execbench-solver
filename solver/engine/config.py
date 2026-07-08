"""Run configuration for the engine (docs/orchestration.md §6b, §6).

A **perspective** is a `(agent, model)` pair; a **tier** is an ordered pool of
perspectives; the **ladder** is an ordered list of tiers (cheap → strong).
Escalation is governed by two knobs — `plateau_cycles` (M, patience) and
`escalate_ceiling` (ambition). Everything here is plain data so a run is fully
described by config + journal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Perspective:
    """One `(agent, model)` — the identity that produces a candidate."""

    agent: str
    model: str

    def __str__(self) -> str:  # used as a stable dict key / journal value
        return f"{self.agent}:{self.model}"


@dataclass
class Tier:
    name: str
    pool: list[Perspective]

    def __post_init__(self) -> None:
        if not self.pool:
            raise ValueError(f"tier {self.name!r} has an empty pool")


@dataclass
class Config:
    tiers: list[Tier]
    design_model: Perspective | None = None      # strong one-shot design (§6b); default: strongest
    escalate_ceiling: float = 0.9                 # ambition: plateau at/above this → stop, not climb
    plateau_cycles: int = 2                       # M: full pool-cycles with no ε-gain → tier stuck
    epsilon: float = 0.02                         # ε for the Pareto frontier
    max_iterations: int = 60                      # hard cap (counts every plan)
    max_gpu_evals: int = 40                       # hard cap (the scarce resource)
    time_limit_s: float | None = None             # wall-clock budget per problem (per run); None = off
    verify_runs: int = 1                          # re-run a would-be frontier entry this many times;
                                                  # reject if any run disagrees (catches flaky/racy kernels)
    agent_fail_limit: int = 3                     # consecutive plan failures before a perspective is
                                                  # circuit-broken (dead agent, e.g. out of credits) and skipped
    score_target: float | None = None            # optional early stop (off by default)
    review_enabled: bool = True                   # pre-GPU code review gate (a DIFFERENT model than the
                                                  # writer reads the kernel + reference + workloads and
                                                  # judges ship/revise BEFORE it spends a GPU eval)
    review_max_rounds: int = 2                    # repair-loop safety valve: "revise" sends the SAME
                                                  # writer back (via a resumed session, not a cold start)
                                                  # with the critique for another attempt, up to this many
                                                  # rounds, then ships as-is (never blocks forever on one
                                                  # stubborn candidate). 2 real attempts with full context
                                                  # is plenty — if the writer can't fix it by then, more
                                                  # rounds mostly burn reviewer-model cost and wall clock
                                                  # rather than actually converging.
    ceiling_consensus: int = 2                    # N consecutive iterations where the agent left the
                                                  # kernel byte-identical to its parent (a "no-op") is a
                                                  # real signal the problem is at ceiling — auto-terminate
                                                  # rather than keep paying for turns that produce nothing.
                                                  # 0 disables. Lowered from 3->2: ~50% of all plan turns
                                                  # measured live were no-ops, so 3 was letting stuck
                                                  # problems burn a 3rd wasted turn before catching it.

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("config needs at least one tier")
        if self.design_model is None:
            # convention: tiers run cheap → strong, so the last tier is strongest
            self.design_model = self.tiers[-1].pool[0]

    @property
    def perspectives(self) -> list[Perspective]:
        """Every distinct perspective referenced (pools + design_model)."""
        seen: set[Perspective] = set()
        out: list[Perspective] = []
        for tier in self.tiers:
            for p in tier.pool:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        if self.design_model is not None and self.design_model not in seen:
            out.append(self.design_model)
        return out


def default_config() -> Config:
    """v1 default: a single Claude tier (escalation off until a 2nd tier)."""
    return Config(tiers=[Tier("claude", [Perspective("claude", "haiku")])])
