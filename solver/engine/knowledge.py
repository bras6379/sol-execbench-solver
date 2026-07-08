"""Knowledge store + serialized curator (docs/orchestration.md §8).

Cross-problem transfer, done narrowly so it *adds up*: it only transfers between
genuine siblings — the **same op at a different shape** (rmsnorm_h128 →
rmsnorm_h512, gemm_n128 → gemm_n256). The op key is parsed from the definition
name (drop the `0NN_` index + the trailing shape params), so a softmax never
seeds a conv.

Three channels:
- **best-kernel persistence** — the winning Solution per op is written to
  `knowledge/best/<op>.json` and **loaded on startup**, so transfer survives
  across separate runs (not just within one process).
- **sibling warm-start** — a new problem is handed the best same-op sibling's
  kernel + approach as a *starting point to adapt* (written to `sibling_kernel.py`
  + summarized in CONTEXT). It is NOT auto-evaluated as a seed, because a sibling
  kernel usually hardcodes its own shape (`assert H==128`) and would just fail.
- **cross-operator technique patterns** (`knowledge/patterns.json`) — unlike the
  two channels above, this transfers a TECHNIQUE (e.g. "split-K") across
  DIFFERENT operators, not a same-op kernel. Indexed by the technique tags in
  reflection.py's `TECHNIQUES` taxonomy, never by op. Never injected as
  ready-to-copy code (see docs/context-architecture-plan.md Part B) — only a
  natural-language summary + known pitfalls + a pointer to read and re-derive.

The curator is globally serialized (one lock) so the shared files never clobber.
A human-readable `families/<op>.md` + `global.md` summary is also kept.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
from pathlib import Path


def _clip_note(s: str, n: int = 170) -> str:
    """Trim a strategy string to ~n chars on a word boundary, for patterns.json
    notes (mirrors reflection.py's `_clip`, not imported to avoid coupling the
    two modules more than `loop.py`'s call site already does)."""
    s = " ".join((s or "").split())
    if len(s) <= n:
        return s
    cut = s[:n]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > n * 0.6 else cut).rstrip() + "…"


def op_key_of(task_id: int, problems_dir: str | Path = "problems") -> str:
    """Reliable op family from the definition name: `021_rmsnorm_h128` → `rmsnorm`,
    `004_gemm_n128_k2048` → `gemm`, `001_fused_add_rmsnorm_h2048` →
    `fused_add_rmsnorm`. Only true siblings share a key."""
    try:
        d = json.loads((Path(problems_dir) / str(task_id) / "definition.json").read_text())
    except Exception:
        return ""
    if d.get("op_type"):
        return str(d["op_type"])
    name = str(d.get("name", "") or "")
    name = re.sub(r"^\d+_", "", name)                       # drop the "021_" index prefix
    name = re.sub(r"(_[a-z]{1,6}\d+\w*)+$", "", name)       # drop trailing shape params
    return name or "?"


class KnowledgeStore:
    def __init__(self, knowledge_dir: str | Path = "knowledge") -> None:
        self.dir = Path(knowledge_dir)
        (self.dir / "families").mkdir(parents=True, exist_ok=True)
        (self.dir / "best").mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()                         # serialize the curator
        self._best: dict[str, dict] = {}                    # op -> {score,task,name,strategy,solution}
        self._load_best()
        self._patterns: dict = self._load_patterns()

    def _load_best(self) -> None:
        """Load persisted best-per-op kernels so transfer works across runs."""
        for p in (self.dir / "best").glob("*.json"):
            try:
                e = json.loads(p.read_text())
                if e.get("solution") and e.get("score") is not None:
                    self._best[p.stem] = e
            except (json.JSONDecodeError, OSError):
                continue

    def _load_patterns(self) -> dict:
        p = self.dir / "patterns.json"
        try:
            d = json.loads(p.read_text()) if p.is_file() else None
        except (json.JSONDecodeError, OSError):
            d = None
        return d if isinstance(d, dict) and "tags" in d else {"schema_version": 1, "tags": {}}

    def sibling_hint(self, op: str, exclude_task: int | None = None) -> dict | None:
        """Best same-op sibling's kernel + approach (a warm start to ADAPT), from a
        DIFFERENT problem than `exclude_task`. None if no sibling yet."""
        e = self._best.get(op)
        if not e or not e.get("solution") or e.get("task") == exclude_task:
            return None
        return {"op": op, "sibling": e.get("name"), "score": e.get("score"),
                "strategy": e.get("strategy", ""), "sources": (e["solution"] or {}).get("sources", [])}

    def pattern_notes(self, tags: list[str], *, exclude_task: int | None = None) -> dict[str, dict]:
        """Cross-OPERATOR technique notes for `tags` (reflection.py's TECHNIQUES
        keys), for injection into a DIFFERENT problem's context — never the
        problem's own in-progress attempts (`exclude_task`). Pitfalls are only
        surfaced once replicated across >=2 DISTINCT op families — a single
        implementation bug on one op is not durable cross-op evidence (see
        docs/context-architecture-plan.md Part B); the raw <2-op evidence still
        accumulates on disk so it can be promoted once a second op replicates it.
        Read-only, synchronous, no lock needed (writer publishes atomically)."""
        out: dict[str, dict] = {}
        tags_map = self._patterns.get("tags", {})
        for tag in tags:
            entry = tags_map.get(tag)
            if not entry:
                continue
            confirmed = [e for e in entry.get("confirmed", []) if e.get("task") != exclude_task]
            pitfalls = [e for e in entry.get("pitfalls", []) if len(set(e.get("ops", []))) >= 2]
            if confirmed or pitfalls:
                out[tag] = {"confirmed": confirmed, "pitfalls": pitfalls}
        return out

    async def curate(self, ctx, op: str, name: str, *, refl=None) -> None:
        async with self._lock:                              # one at a time; no clobber
            best = ctx.frontier.best()
            score = round(best.mean, 4) if best else None
            self._append_family(op, ctx.task_id, name, score, ctx.tier_idx,
                                ctx.terminated_reason, best.strategy if best else "")
            if best and best.solution is not None:
                cur = self._best.get(op)
                if cur is None or best.mean > cur.get("score", -1):
                    entry = {"op": op, "task": ctx.task_id, "name": name, "score": best.mean,
                             "strategy": best.strategy or "", "solution": best.solution}
                    self._best[op] = entry
                    self._write_best(op, entry)             # persist for future runs
            if refl is not None and refl.tech_events:
                self._upsert_patterns(op, ctx.task_id, refl.tech_events)
                self._write_patterns()

    def _write_best(self, op: str, entry: dict) -> None:
        path = self.dir / "best" / f"{op}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(entry, f)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(path)

    def _upsert_patterns(self, op: str, task_id: int, tech_events: list[dict]) -> None:
        """Fold this run's tech_events into self._patterns, in place. UPSERT, not
        append: a `reopened`-and-reterminated run re-processing the same task_id
        replaces its own prior 'confirmed' entry and merges into its own prior
        'pitfall' op-membership, instead of double-counting on every replay (the
        bug already present in `_append_family`'s unconditional append — not
        copied here)."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tags = self._patterns.setdefault("tags", {})
        for ev in tech_events:
            tag_entry = tags.setdefault(ev["tag"], {"confirmed": [], "pitfalls": []})
            if ev["kind"] == "confirmed":
                confirmed = [e for e in tag_entry["confirmed"] if e.get("task") != task_id]
                confirmed.append({"op": op, "task": task_id, "score": ev["score"],
                                  "note": _clip_note(ev["strategy"]), "cand": ev["cand"], "ts": now})
                confirmed.sort(key=lambda e: -(e["score"] or 0))
                tag_entry["confirmed"] = confirmed[:5]
            else:                                            # "pitfall"
                fam = ev["error"] or "UNKNOWN"
                pf = next((p for p in tag_entry["pitfalls"] if p["error_family"] == fam), None)
                if pf is None:
                    pf = {"error_family": fam, "ops": [], "note": "", "example_cand": None, "ts": now}
                    tag_entry["pitfalls"].append(pf)
                if op not in pf["ops"]:
                    pf["ops"].append(op)
                pf["note"] = _clip_note(ev["strategy"])
                pf["example_cand"] = ev["cand"]
                pf["ts"] = now
                tag_entry["pitfalls"].sort(key=lambda p: -len(set(p["ops"])))
                tag_entry["pitfalls"] = tag_entry["pitfalls"][:5]

    def _write_patterns(self) -> None:
        path = self.dir / "patterns.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._patterns, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(path)

    def _append_family(self, op: str, task: int, name: str, score, tier: int,
                       reason, strategy: str) -> None:
        fam_path = self.dir / "families" / f"{op}.md"
        if not fam_path.exists():
            fam_path.write_text(f"# Op: {op}\n\nOne distilled line per finished problem.\n\n")
        with fam_path.open("a", encoding="utf-8") as f:
            f.write(f'- task {task} ({name}): best={score} tier={tier} via "{strategy}" [{reason}]\n')
        glob = self.dir / "global.md"
        if not glob.exists():
            glob.write_text("# Global learnings\n\n")
        with glob.open("a", encoding="utf-8") as f:
            f.write(f"- [{op}] task {task}: best={score} tier={tier}\n")
