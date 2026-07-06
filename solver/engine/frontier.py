"""The ε-Pareto frontier over per-workload-shape scores (docs/orchestration.md §6).

Vector = per-shape `sol_score`, non-PASSED shape = 0.0. A ε-dominates B iff
A ≥ B−ε everywhere and A > B+ε somewhere. Partial specialists survive (they win
some shapes), so the set is *not* top-k by mean. `select` samples a parent
weighted by shapes-won; `best` is the argmax-mean among all-correct members
(the deliverable invariant).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Member:
    cand_id: str
    vector: tuple[float, ...]          # per-shape score, non-PASSED = 0.0
    all_passed: bool
    solution: dict | None = None
    strategy: str = ""
    agent: str = ""
    model: str = ""
    reflection: str | None = None

    @property
    def mean(self) -> float:
        return sum(self.vector) / len(self.vector) if self.vector else 0.0


def _dominates(a: tuple[float, ...], b: tuple[float, ...], eps: float) -> bool:
    """a ε-dominates b: no worse than b−ε everywhere, strictly better than b+ε somewhere."""
    ge = all(x >= y - eps for x, y in zip(a, b))
    gt = any(x > y + eps for x, y in zip(a, b))
    return ge and gt


class Frontier:
    def __init__(self, epsilon: float = 0.02) -> None:
        self.epsilon = epsilon
        self.members: list[Member] = []

    def best(self) -> Member | None:
        correct = [m for m in self.members if m.all_passed]
        return max(correct, key=lambda m: m.mean) if correct else None

    def best_score(self) -> float:
        b = self.best()
        return b.mean if b else 0.0

    def accept(self, m: Member) -> str:
        """Insert `m`, pruning what it dominates. Returns the verdict.

        'entered'   — m is non-dominated and joins the frontier
        'dominated' — an existing member ε-dominates m; m is discarded
        """
        if self.members and len(m.vector) != len(self.members[0].vector):
            raise ValueError("frontier vectors must share length within a problem")
        for other in self.members:
            if _dominates(other.vector, m.vector, self.epsilon):
                return "dominated"
        # m survives — drop everything it ε-dominates, then add it
        self.members = [o for o in self.members
                        if not _dominates(m.vector, o.vector, self.epsilon)]
        self.members.append(m)
        return "entered"

    def select(self, rng: random.Random) -> Member | None:
        """Weighted-random parent: weight = shapes won (+1 smoothing)."""
        if not self.members:
            return None
        n = len(self.members[0].vector)
        weights = [1.0] * len(self.members)          # +1 so every member can be picked
        for i in range(n):
            col = [m.vector[i] for m in self.members]
            top = max(col)
            winners = [j for j, v in enumerate(col) if v >= top - 1e-12]
            for j in winners:
                weights[j] += 1.0 / len(winners)
        return rng.choices(self.members, weights=weights, k=1)[0]
