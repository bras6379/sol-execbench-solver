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
import time
from collections import Counter
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
        self.seed = seed
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
        self.deadline: float | None = None          # monotonic wall-clock stop (live-only, per run)
        self.recent_failures: list[dict] = []       # last few INCORRECT attempts (fed back to agents)
        self.review_critique: str | None = None     # live-only: this round's pre-GPU review issues,
                                                     # fed to the SAME writer for a repair turn
        self.noop_streak = 0                        # live-only: consecutive iterations where the agent
                                                     # left the kernel unchanged from its parent
        self.sibling_hint: dict | None = None       # best same-op sibling's kernel to adapt (transfer)
        self.playbook: list[dict] = []              # accepted candidates' handoffs = reserve plays
        self._playbook_seen: set[str] = set()       # dedup playbook entries by handoff text
        self.disabled: set[str] = set()             # perspectives circuit-broken (repeated agent failures)
        self._fail_streak: dict[str, int] = {}      # consecutive plan failures per perspective
        self._pending: dict[str, dict] = {}
        self._replaying = False

    # ---- tier / perspective ----
    @property
    def tier(self):
        # Clamp: retuning to FEWER tiers between sessions must never crash a resume.
        # The journal keeps going regardless of how the config is tuned.
        return self.cfg.tiers[min(self.tier_idx, len(self.cfg.tiers) - 1)]

    @property
    def pool_size(self) -> int:
        return len(self.tier.pool)

    def _pool_order(self) -> list:
        """A per-problem *shuffled* permutation of the current tier's pool —
        deterministic (seeded by seed+task_id+tier), so replay is identical, but
        DIFFERENT across problems, so the fleet doesn't march in lockstep and hammer
        one provider (e.g. Claude/GPT) simultaneously. Round-robin over this
        permutation still covers every model once per cycle (plateau logic intact)."""
        pool = list(self.tier.pool)
        random.Random(f"{self.seed}:{self.task_id}:{self.tier_idx}").shuffle(pool)
        return pool

    def current_perspective(self):
        """Next perspective in the shuffled rotation, skipping any that have been
        circuit-broken (disabled). None ⇒ every model in this tier is dead."""
        order = self._pool_order()
        n = len(order)
        for i in range(n):
            p = order[(self.cursor + i) % n]
            if str(p) not in self.disabled:
                return p
        return None

    def route_around_dead_tier(self) -> bool:
        """The current tier has no live agent (all circuit-broken). Switch to any
        OTHER tier that still has one — this is how a run gracefully **downgrades**
        when premium agents (Claude/GPT) run out of credits and only the cheap
        providers remain. Returns False if no tier anywhere has a live agent."""
        for idx, tier in enumerate(self.cfg.tiers):
            if idx == self.tier_idx:
                continue
            live = next((p for p in tier.pool if str(p) not in self.disabled), None)
            if live is not None:
                self.record("agent_changed", tier=idx, agent=live.agent, model=live.model,
                            trigger="route")
                return True
        return False

    # ---- lifecycle predicates ----
    def fresh(self) -> bool:
        return not self.bootstrapped

    def done(self) -> bool:
        if self.terminated_reason is not None:
            return True
        if self.deadline is not None and time.monotonic() >= self.deadline:
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
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return "budget:time"
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
        elif ev == "plan_error":
            # an agent call failed (e.g. timed out, or no credits) — advance exactly
            # like a plan so the loop moves on (next perspective, counts toward the
            # cap) and the problem keeps its frontier instead of aborting. Track the
            # failure streak: repeated failures circuit-break the perspective so a
            # dead agent (e.g. Claude/GPT out of credits) stops being scheduled.
            self.iters += 1
            self.planned_since_gain += 1
            self.cursor += 1
            persp = f"{e.get('agent', '')}:{e.get('model', '')}"
            self._fail_streak[persp] = self._fail_streak.get(persp, 0) + 1
            if self._fail_streak[persp] >= self.cfg.agent_fail_limit:
                self.disabled.add(persp)
        elif ev == "plan_done":
            self.iters += 1
            self.planned_since_gain += 1
            self.cursor += 1
            self._fail_streak[f"{e.get('agent', '')}:{e.get('model', '')}"] = 0   # a success clears the streak
            cid = e.get("cand")
            if cid:
                self.seen.add(cid)
                self._pending[cid] = {
                    "solution": e.get("solution"), "scores": None, "all_passed": False,
                    "agent": e.get("agent", ""), "model": e.get("model", ""),
                    "strategy": e.get("strategy", ""), "parent": e.get("parent"),
                    "handoff": e.get("handoff"),
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
                p = self._pending[cid]
            # note_failure() itself is a plain in-memory call (loop.py calls it
            # directly, live-only) — reconstruct the SAME effect here so a
            # resumed run doesn't silently start recent_failures empty, throwing
            # away "don't repeat X" context right when a restart just happened
            # (confirmed live, 2026-07-08).
            if cid and p is not None and not e.get("all_passed"):
                statuses = [s for s in (e.get("statuses") or []) if s and s != "PASSED"]
                reason = Counter(statuses).most_common(1)[0][0] if statuses else "INCORRECT"
                self.note_failure(p.get("strategy", ""), reason, cid, detail=e.get("detail", ""))
        elif ev == "verify_done":
            self.evals += 1                           # a re-verification consumed a GPU eval;
            # deliberately does NOT touch _pending/seen/frontier — it re-runs an
            # already-scored candidate purely to confirm correctness stability.
        elif ev == "accept":
            cid = e.get("cand")
            if self._replaying:
                p = self._pending.get(cid)
                if p and p["scores"] is not None:
                    self.frontier.accept(self._member(cid, p))
            if e.get("verdict") == "entered":
                self.planned_since_gain = 0
                self._playbook_add(cid)          # bank the handoff (live and replay alike)
        elif ev == "agent_changed":
            # escalation (plateau → stronger tier) OR route (current tier's agents all
            # circuit-broken → switch to a tier that still has a live one).
            if e.get("trigger") in ("escalation", "route"):
                self.tier_idx = int(e.get("tier", self.tier_idx + 1))
                self.cursor = 0
                self.planned_since_gain = 0
        elif ev == "terminated":
            self.terminated_reason = e.get("reason")
        elif ev == "reopened":
            self.terminated_reason = None                 # a non-final run resumes
            self.disabled.clear()                         # fresh circuit-breaker each session:
            self._fail_streak.clear()                     # retuning gives every agent a new chance
            if self.tier_idx >= len(self.cfg.tiers):      # config shrank → don't point past the end
                self.tier_idx = 0
        # iter / check / novelty / flaky / verify_started / exec_enqueued /
        # exec_started / run_started / solver_error carry no core-state delta.

    # ---- helpers used by the loop ----
    def _member(self, cand_id: str, p: dict) -> Member:
        return Member(cand_id=cand_id, vector=tuple(p["scores"]), all_passed=p["all_passed"],
                      solution=p.get("solution"), strategy=p.get("strategy", ""),
                      agent=p.get("agent", ""), model=p.get("model", ""),
                      sol_score_cal=p.get("sol_score_cal"))

    def reopen_if_capped(self) -> bool:
        """Resume a run that isn't truly finished. Only `converged:*`/`target` are
        final (a different prior won't help / the goal is met). Everything else —
        a killed run (`stopped`), a cap (`budget:*`), a dead-agent stop
        (`agents-unavailable`) — continues when the caps allow, so you can retune
        the system (models/tiers/budgets) and the journal keeps going. Journals
        `reopened` so replay reconstructs it."""
        r = self.terminated_reason
        if (r and not r.startswith("converged") and r != "target"
                and self.iters < self.cfg.max_iterations
                and self.evals < self.cfg.max_gpu_evals):
            self.record("reopened", from_reason=r)
            return True
        return False

    def note_failure(self, strategy: str, reason: str, cand_id: str, detail: str = "") -> None:
        """Remember an INCORRECT attempt so the next agent context can warn against
        repeating it AND tell it exactly which workloads failed and how (live-only;
        keeps the last few)."""
        self.recent_failures.append({"strategy": strategy or "", "reason": reason,
                                     "cand": cand_id, "detail": detail})
        self.recent_failures = self.recent_failures[-4:]

    def _playbook_add(self, cand_id: str) -> None:
        """A candidate ENTERED the frontier — bank its handoff (the higher-ceiling
        idea it flagged but didn't ship) into the per-problem playbook, deduped by
        text. Runs in both live and replay (state is journal-derived), so the next
        agent gets accumulated reserve plays instead of losing them to trajectory."""
        p = self._pending.get(cand_id) or {}
        text = (p.get("handoff") or "").strip()
        if not text:
            return
        key = " ".join(text.split()).lower()[:200]
        if key in self._playbook_seen:
            return
        self._playbook_seen.add(key)
        self.playbook.append({"cand": cand_id, "strategy": p.get("strategy", ""), "handoff": text})

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
