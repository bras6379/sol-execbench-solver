"""Knowledge store + serialized curator (docs/orchestration.md §8).

Two jobs: distill each *finished* problem's run into transferable family/global
markdown, and offer the cheapest transfer channel — **bootstrap sibling
seeding** (a new problem starts from the best Solution a same-family sibling has
found so far). The curator is **globally serialized** (one lock) so the shared
family file is never concurrently written; per-family locks are deferred.

v1 distillation is a deterministic summary line per problem. The LLM distiller
(which reads the journal and writes real insights) lands with the real Agent —
this is the store + serialization + injection plumbing it will slot into.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


class KnowledgeStore:
    def __init__(self, knowledge_dir: str | Path = "knowledge") -> None:
        self.dir = Path(knowledge_dir)
        (self.dir / "families").mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()                       # serialize the curator
        self._sibling_best: dict[str, tuple[float, dict]] = {}   # family -> (mean, best Solution)

    def sibling_seed(self, task_id: int, family: str) -> list[dict]:
        """Best Solution found by a finished same-family sibling (or none yet)."""
        entry = self._sibling_best.get(family)
        return [entry[1]] if entry and entry[1] else []

    async def curate(self, ctx, family: str, name: str) -> None:
        async with self._lock:                            # one at a time; no clobber
            best = ctx.frontier.best()
            score = round(best.mean, 4) if best else None
            self._append_family(family, ctx.task_id, name, score, ctx.tier_idx,
                                ctx.terminated_reason, best.strategy if best else "")
            if best and best.solution is not None:
                cur = self._sibling_best.get(family)
                if cur is None or best.mean > cur[0]:
                    self._sibling_best[family] = (best.mean, best.solution)

    def _append_family(self, family: str, task: int, name: str, score, tier: int,
                       reason, strategy: str) -> None:
        fam_path = self.dir / "families" / f"{family}.md"
        if not fam_path.exists():
            fam_path.write_text(
                f"# Family: {family}\n\nOne distilled line per finished problem.\n\n")
        with fam_path.open("a", encoding="utf-8") as f:
            f.write(f'- task {task} ({name}): best={score} tier={tier} '
                    f'via "{strategy}" [{reason}]\n')
        glob = self.dir / "global.md"
        if not glob.exists():
            glob.write_text("# Global learnings\n\n")
        with glob.open("a", encoding="utf-8") as f:
            f.write(f"- [{family}] task {task}: best={score} tier={tier}\n")
