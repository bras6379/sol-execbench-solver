"""The Agent interface + a deterministic StubAgent (docs/orchestration.md §2, §12).

An `Agent` is any coding agent bound to a model, behind four methods:
`design` (one-shot per problem), `plan` (mutate → a new candidate), `reflect`
(the "text gradient"), `judge` (novelty verdict). The engine is written against
this interface only; real backends (Claude Agent SDK, OpenAI-compatible) come
later. `StubAgent` is the deterministic, scriptable stand-in the §12 tests drive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from .config import Perspective


@dataclass
class Candidate:
    cand_id: str
    solution: dict
    parent: str | None
    agent: str
    model: str
    strategy: str = ""
    reflection: str | None = None


def solution_hash(solution: dict) -> str:
    """Stable content hash — the novelty gate's identity (stands in for the
    harness's `Solution.hash()` until the real Solution model is wired)."""
    payload = json.dumps(solution, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(payload).hexdigest()


class Agent(Protocol):
    perspective: Perspective

    async def design(self, task_id: int) -> str: ...
    async def plan(self, parent: Any, ctx: Any) -> Candidate: ...
    async def reflect(self, cand: Candidate, result: Any, verdict: str) -> str: ...
    async def judge(self, cand: Candidate, parent: Any, frontier: Any) -> str: ...


# A planner turns (perspective, parent, ctx) into a *spec* dict the StubAgent
# renders into a Candidate. Spec keys:
#   scores: list[float|None]   per-shape score (None = that shape failed)
#   invalid: bool              produce a check-failing candidate
#   duplicate: bool            reuse fixed content → identical hash (novelty dup)
#   strategy: str              one-line TL;DR
Spec = dict
Planner = Callable[[Perspective, Any, Any], Spec]


class StubAgent:
    """Deterministic Agent for tests — see the §12 stub contract.

    `planner(perspective, parent, ctx) -> Spec` scripts what gets produced;
    `judge_fn` scripts novelty verdicts; `raise_on(self, parent, ctx)` forces a
    crash (crash-isolation test). Same perspective ⇒ same tags on every candidate.
    """

    def __init__(
        self,
        perspective: Perspective,
        planner: Planner,
        *,
        design_text: str = "stub design",
        judge_fn: Callable[[Candidate, Any, Any], str] | None = None,
        raise_on: Callable[["StubAgent", Any, Any], bool] | None = None,
    ) -> None:
        self.perspective = perspective
        self._planner = planner
        self._design_text = design_text
        self._judge_fn = judge_fn or (lambda cand, parent, frontier: "materially-new")
        self._raise_on = raise_on
        self.calls = 0

    async def design(self, task_id: int) -> str:
        return self._design_text

    async def plan(self, parent: Any, ctx: Any) -> Candidate:
        self.calls += 1
        if self._raise_on and self._raise_on(self, parent, ctx):
            raise RuntimeError(f"stub agent {self.perspective} boom")
        spec = self._planner(self.perspective, parent, ctx)
        solution = spec.get("solution")
        if solution is None:
            solution = {"__eval__": {"scores": list(spec.get("scores", [0.5]))}}
            if spec.get("invalid"):
                solution["__invalid__"] = True
            if not spec.get("duplicate"):
                # distinct hash per candidate; keyed on ctx.iters (restored on
                # replay) so a resumed run reproduces identical candidates
                uid = getattr(ctx, "iters", self.calls)
                solution["__uid__"] = f"{self.perspective}:{uid}"
        return Candidate(
            cand_id=solution_hash(solution)[:12],
            solution=solution,
            parent=getattr(parent, "cand_id", None),
            agent=self.perspective.agent,
            model=self.perspective.model,
            strategy=spec.get("strategy", ""),
        )

    async def reflect(self, cand: Candidate, result: Any, verdict: str) -> str:
        return f"reflect[{self.perspective}]: {verdict}"

    async def judge(self, cand: Candidate, parent: Any, frontier: Any) -> str:
        return self._judge_fn(cand, parent, frontier)


def stub_agents(perspectives, planner: Planner, **kwargs) -> dict:
    """Build `{perspective: StubAgent}` for every perspective a config references."""
    return {p: StubAgent(p, planner, **kwargs) for p in perspectives}
