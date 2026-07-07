"""The Agent interface + a deterministic StubAgent (docs/orchestration.md §2, §12).

An `Agent` is any coding agent bound to a model, behind two methods: `design`
(one-shot per problem) and `plan` (mutate → a new candidate). A plan also emits a
**handoff** — the higher-ceiling idea it did NOT ship — which the engine
accumulates into a per-problem playbook and feeds to the next agent as the "text
gradient". The engine is written against this interface only. `StubAgent` is the
deterministic, scriptable stand-in the §12 tests drive.
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
    handoff: str | None = None          # higher-ceiling idea NOT shipped → the per-problem playbook
    tokens: dict | None = None          # {in, out, reasoning, cached, cost_usd} from the agent stream
    trajectory: str | None = None       # path to the persisted agent trajectory (jsonl)


@dataclass
class ReviewVerdict:
    """A pre-GPU code review's verdict on a candidate — read the kernel against
    the reference + graded shapes and judge whether it's worth a GPU eval."""
    verdict: str                        # "ship" | "revise"
    issues: list[str] = field(default_factory=list)
    reviewer: str = ""                  # the Perspective that produced this verdict

    @property
    def ship(self) -> bool:
        return self.verdict == "ship"

    def issues_text(self) -> str:
        return "\n".join(f"- {i}" for i in self.issues)


def solution_hash(solution: dict) -> str:
    """Stable content hash — the novelty gate's identity (stands in for the
    harness's `Solution.hash()` until the real Solution model is wired)."""
    payload = json.dumps(solution, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(payload).hexdigest()


class Agent(Protocol):
    perspective: Perspective

    async def design(self, task_id: int) -> str: ...
    async def plan(self, parent: Any, ctx: Any) -> Candidate: ...
    async def review(self, cand: Candidate, ctx: Any) -> ReviewVerdict: ...


# A planner turns (perspective, parent, ctx) into a *spec* dict the StubAgent
# renders into a Candidate. Spec keys:
#   scores: list[float|None]   per-shape score (None = that shape failed)
#   invalid: bool              produce a check-failing candidate
#   duplicate: bool            reuse fixed content → identical hash (novelty dup)
#   strategy: str              one-line TL;DR
#   handoff: str               higher-ceiling idea not shipped (→ playbook)
Spec = dict
Planner = Callable[[Perspective, Any, Any], Spec]
# A reviewer turns (perspective, candidate, ctx) into a ReviewVerdict. None (the
# default) means "always ship" — existing tests that don't exercise review are
# unaffected.
ReviewFn = Callable[[Perspective, Candidate, Any], "ReviewVerdict"]


class StubAgent:
    """Deterministic Agent for tests — see the §12 stub contract.

    `planner(perspective, parent, ctx) -> Spec` scripts what gets produced;
    `raise_on(self, parent, ctx)` forces a crash (crash-isolation test). Same
    perspective ⇒ same tags on every candidate.
    """

    def __init__(
        self,
        perspective: Perspective,
        planner: Planner,
        *,
        design_text: str = "stub design",
        raise_on: Callable[["StubAgent", Any, Any], bool] | None = None,
        reviewer: ReviewFn | None = None,
    ) -> None:
        self.perspective = perspective
        self._planner = planner
        self._design_text = design_text
        self._raise_on = raise_on
        self._reviewer = reviewer
        self.calls = 0
        self.review_calls = 0

    async def design(self, task_id: int) -> str:
        return self._design_text

    async def review(self, cand: Candidate, ctx: Any) -> ReviewVerdict:
        self.review_calls += 1
        if self._reviewer is None:
            return ReviewVerdict(verdict="ship", reviewer=str(self.perspective))
        return self._reviewer(self.perspective, cand, ctx)

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
            if spec.get("flaky_on"):                # attempts on which a re-run "fails"
                solution["__flaky_on__"] = list(spec["flaky_on"])
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
            handoff=spec.get("handoff"),
        )


def stub_agents(perspectives, planner: Planner, **kwargs) -> dict:
    """Build `{perspective: StubAgent}` for every perspective a config references."""
    return {p: StubAgent(p, planner, **kwargs) for p in perspectives}
