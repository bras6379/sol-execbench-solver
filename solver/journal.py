"""Append-only run journal: the engine's single source of truth.

Schema v1 (pinned in docs/orchestration.md §Instrumentation): every line is
one JSON object `{v, ts, task, ev, ...}` — ISO-8601 UTC timestamp, fsync'd on
write. A crash can truncate at most the trailing line; `read()` drops it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


class Journal:
    """Writer for one problem's journal (runs/<task>/journal.jsonl)."""

    def __init__(self, path: Path, task_id: int):
        self.path = Path(path)
        self.task_id = task_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, ev: str, *, ts: str | None = None, **fields) -> dict:
        entry = {"v": SCHEMA_VERSION, "ts": ts or _utc_now(),
                 "task": self.task_id, "ev": ev, **fields}
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return entry


def read(path: Path) -> list[dict]:
    """Read a journal, tolerating a truncated trailing line."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            break  # truncated tail — drop it and everything after
    return out


def read_all(runs_dir: Path) -> dict[int, list[dict]]:
    """Read every problem journal under runs_dir → {task_id: events}."""
    runs_dir = Path(runs_dir)
    out: dict[int, list[dict]] = {}
    if not runs_dir.exists():
        return out
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir() and d.name.isdigit():
            events = read(d / "journal.jsonl")
            if events:
                out[int(d.name)] = events
    return out
