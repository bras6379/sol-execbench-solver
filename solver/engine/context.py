"""RunContext: per-problem engine state, journal replay, and resume.

State lives here and is mutated through exactly one path: `record(ev, **f)` =
append the journal line **and** `apply()` it. Replay calls `apply()` alone over
the journal. Because live and replay share `apply()`, the in-memory state after
writing a line is identical to replaying up to that line — the basis for
bitwise-identical resume (docs/orchestration.md §12.1).

Resume truncates the journal to the last **commit boundary** (`bootstrapped` /
`iter` / `agent_changed` / `terminated`), dropping a partially-journaled
iteration so it re-runs cleanly rather than double-counting.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from .. import journal as journal_mod
from .config import Config
from .frontier import Frontier, Member

# Events that end a committed unit of work; everything after the last one is a
# partial iteration and is dropped on resume.
_BOUNDARIES = {"bootstrapped", "iter", "agent_changed", "terminated", "reopened"}


class RunContext:
    def __init__(self, task_id: int, cfg: Config, runs_dir: str | Path = "runs",
                 *, seed: int = 0) -> None:
        self.task_id = task_id
        self.cfg = cfg
        self.runs_dir = Path(runs_dir)
        self.path = self.runs_dir / str(task_id) / "journal.jsonl"
        self.journal = journal_mod.Journal(self.path, task_id)
        self.rng = random.Random(seed)
        self.frontier = Frontier(cfg.epsilon)
        self.tier_idx = 0
        self.cursor = 0
        self.iters = 0
        self.evals = 0
        self.planned_since_gain = 0
        self.seen: set[str] = set()
        self.design: str | None = None
        self.bootstrapped = False
        self.terminated_reason: str | None = None
        self._pending: dict[str, dict] = {}
        self._replaying = False

    # ---- tier / perspective ----
    @property
    def tier(self):
        return self.cfg.tiers[self.tier_idx]

    @property
    def pool_size(self) -> int:
        return len(self.tier.pool)

    def current_perspective(self):
        return self.tier.pool[self.cursor % self.pool_size]

    # ---- lifecycle predicates ----
    def fresh(self) -> bool:
        return not self.bootstrapped

    def done(self) -> bool:
        if self.terminated_reason is not None:
            return True
        if self.iters >= self.cfg.max_iterations:
            return True
        if self.evals >= self.cfg.max_gpu_evals:
            return True
        if self.cfg.score_target is not None and self.frontier.best_score() >= self.cfg.score_target:
            return True
        return False

    def done_reason(self) -> str:
        if self.terminated_reason is not None:
            return self.terminated_reason
        if self.cfg.score_target is not None and self.frontier.best_score() >= self.cfg.score_target:
            return "target"
        if self.evals >= self.cfg.max_gpu_evals:
            return "budget:evals"
        if self.iters >= self.cfg.max_iterations:
            return "budget:iterations"
        return "stopped"

    def tier_plateaued(self) -> bool:
        # M full pool-cycles with no ε-gain ⇒ every pool member had ≥ M shots.
        return self.planned_since_gain >= self.cfg.plateau_cycles * self.pool_size

    def escalate(self) -> bool:
        """Advance a tier iff headroom remains and a stronger tier is left.
        Returns False when the plateau should terminate the run instead."""
        if self.frontier.best_score() >= self.cfg.escalate_ceiling:
            return False
        if self.tier_idx + 1 >= len(self.cfg.tiers):
            return False
        nxt = self.tier_idx + 1
        p = self.cfg.tiers[nxt].pool[0]
        self.record("agent_changed", tier=nxt, agent=p.agent, model=p.model,
                    trigger="escalation")
        return True

    # ---- the single mutation path ----
    def record(self, ev: str, **fields) -> dict:
        entry = self.journal.append(ev, **fields)
        self.apply(entry)
        return entry

    def apply(self, e: dict) -> None:
        ev = e.get("ev")
        if ev == "design_done":
            self.design = e.get("text")
        elif ev == "bootstrapped":
            self.bootstrapped = True
        elif ev == "plan_done":
            self.iters += 1
            self.planned_since_gain += 1
            self.cursor += 1
            cid = e.get("cand")
            if cid:
                self.seen.add(cid)
                self._pending[cid] = {
                    "solution": e.get("solution"), "scores": None, "all_passed": False,
                    "agent": e.get("agent", ""), "model": e.get("model", ""),
                    "strategy": e.get("strategy", ""), "parent": e.get("parent"),
                }
            if self._replaying:
                self.frontier.select(self.rng)      # advance rng exactly as the live select() did
        elif ev == "exec_done":
            self.evals += 1
            cid = e.get("cand")
            if cid:
                self.seen.add(cid)                    # seeds have no plan_done; dedup them too
            p = self._pending.get(cid)
            if p is not None:
                p["scores"] = list(e.get("scores") or [])
                p["all_passed"] = bool(e.get("all_passed"))
                p["sol_score_cal"] = e.get("sol_score_cal")
            elif cid:                                 # a seed (no plan_done precedes it)
                self._pending[cid] = {
                    "solution": None, "scores": list(e.get("scores") or []),
                    "all_passed": bool(e.get("all_passed")), "strategy": "seed",
                    "agent": "", "model": "", "parent": None,
                    "sol_score_cal": e.get("sol_score_cal"),
                }
        elif ev == "accept":
            cid = e.get("cand")
            if self._replaying:
                p = self._pending.get(cid)
                if p and p["scores"] is not None:
                    self.frontier.accept(self._member(cid, p))
            if e.get("verdict") == "entered":
                self.planned_since_gain = 0
        elif ev == "agent_changed":
            if e.get("trigger") == "escalation":
                self.tier_idx = int(e.get("tier", self.tier_idx + 1))
                self.cursor = 0
                self.planned_since_gain = 0
        elif ev == "terminated":
            self.terminated_reason = e.get("reason")
        elif ev == "reopened":
            self.terminated_reason = None                 # a cap-terminated run resumes
        # iter / check / novelty / exec_enqueued / exec_started / reflect_done /
        # run_started / solver_error carry no core-state delta.

    # ---- helpers used by the loop ----
    def _member(self, cand_id: str, p: dict) -> Member:
        return Member(cand_id=cand_id, vector=tuple(p["scores"]), all_passed=p["all_passed"],
                      solution=p.get("solution"), strategy=p.get("strategy", ""),
                      agent=p.get("agent", ""), model=p.get("model", ""),
                      sol_score_cal=p.get("sol_score_cal"))

    def reopen_if_capped(self) -> bool:
        """A `budget:*`-terminated run continues when the caps now allow more
        work (e.g. resumed with a higher `--max-evals`); `converged:*`/`target`
        runs stay done. Journals `reopened` so replay reconstructs it."""
        if (self.terminated_reason and self.terminated_reason.startswith("budget:")
                and self.iters < self.cfg.max_iterations
                and self.evals < self.cfg.max_gpu_evals):
            self.record("reopened", from_reason=self.terminated_reason)
            return True
        return False

    def accept_candidate(self, cand_id: str) -> str:
        """Live-accept a candidate whose scores are already recorded (exec_done)."""
        p = self._pending[cand_id]
        verdict = self.frontier.accept(self._member(cand_id, p))   # live mutation
        best = self.frontier.best()
        self.record("accept", cand=cand_id, verdict=verdict,
                    best=self.frontier.best_score(),
                    best_cal=(best.sol_score_cal if best else None),
                    frontier=len(self.frontier.members))
        return verdict

    # ---- load / resume ----
    @classmethod
    def load(cls, task_id: int, cfg: Config, runs_dir: str | Path = "runs",
             *, seed: int = 0) -> "RunContext":
        ctx = cls(task_id, cfg, runs_dir, seed=seed)
        events = journal_mod.read(ctx.path)
        kept = _truncate_to_boundary(events)
        if len(kept) != len(events):
            _rewrite(ctx.path, kept)                 # physically drop the partial trailing iteration
        ctx._replaying = True
        for e in kept:
            ctx.apply(e)
        ctx._replaying = False
        return ctx


def _truncate_to_boundary(events: list[dict]) -> list[dict]:
    last = -1
    for i, e in enumerate(events):
        if e.get("ev") in _BOUNDARIES:
            last = i
    return events[: last + 1]


def _rewrite(path: Path, events: list[dict]) -> None:
    path = Path(path)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
