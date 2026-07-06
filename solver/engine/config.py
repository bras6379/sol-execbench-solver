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
    score_target: float | None = None            # optional early stop (off by default)

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
