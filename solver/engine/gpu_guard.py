"""Tracks whether a GPU eval-to-accept sequence is currently in flight, so a
shutdown (a manual restart, or the --gpu-max-hours cap firing) can wait for it
to reach a safe stopping point instead of orphaning a result that already
spent real GPU time but never got recorded.

Confirmed live (2026-07-08): a SIGTERM landing between exec_done and a
candidate's accept+persist step left a fully-scored candidate permanently
stuck showing as still "running" — real, billed GPU time spent, with no
outcome ever recorded, and nothing will ever revisit it (a resumed run just
starts a fresh iteration). Since restarting already costs several minutes for
new agent calls to spin back up regardless, there's no real cost to waiting a
short, bounded time for any GPU work already in flight to finish first.
"""

from __future__ import annotations

import asyncio
import time


class GpuWorkGuard:
    """An in-flight counter + bounded async wait — not a mutex. Multiple
    problems can each be inside their own eval-to-accept window "at once" in
    asyncio's cooperative sense (only one physically executes a GPU call at a
    time, via the executor's own single-flight lock, but one problem's
    CPU-only accept+persist step can overlap with another's GPU wait), so this
    tracks "is ANY of that in flight anywhere", not "who holds the GPU"."""

    def __init__(self) -> None:
        self._n = 0

    def __enter__(self) -> "GpuWorkGuard":
        self._n += 1
        return self

    def __exit__(self, *exc: object) -> None:
        self._n = max(0, self._n - 1)

    @property
    def busy(self) -> bool:
        return self._n > 0

    async def wait_idle(self, timeout: float, *, poll_s: float = 0.1) -> bool:
        """Poll until no eval-to-accept sequence is in flight, or `timeout`
        elapses. Returns True if it went idle in time, False if the bound was
        hit — the caller proceeds either way; this is a grace period, not a
        guarantee, and must never hang a shutdown forever."""
        deadline = time.monotonic() + timeout
        while self.busy:
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll_s)
        return True
