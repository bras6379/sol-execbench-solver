"""GpuWorkGuard — tracks whether a GPU eval-to-accept sequence is in flight, so
a shutdown can wait for it instead of orphaning a result that already spent
real GPU time (see solver/engine/gpu_guard.py's module docstring for the live
incident this fixes).
"""

from __future__ import annotations

import asyncio

from solver.engine.gpu_guard import GpuWorkGuard

run = asyncio.run


def test_starts_idle():
    g = GpuWorkGuard()
    assert not g.busy


def test_busy_while_a_section_is_open():
    g = GpuWorkGuard()
    with g:
        assert g.busy
    assert not g.busy


def test_stays_busy_while_nested_sections_overlap():
    """Multiple problems can each be mid eval-to-accept "at once" in asyncio's
    cooperative sense — busy must reflect ANY of them, not just the first."""
    g = GpuWorkGuard()
    with g:
        with g:
            assert g.busy
        assert g.busy       # outer section still open
    assert not g.busy


def test_exit_never_goes_negative_on_an_unbalanced_exit():
    g = GpuWorkGuard()
    g.__exit__()
    g.__exit__()
    assert not g.busy


def test_wait_idle_returns_true_immediately_when_already_idle():
    g = GpuWorkGuard()
    assert run(g.wait_idle(timeout=1.0)) is True


def test_wait_idle_returns_true_once_the_section_closes_in_time():
    g = GpuWorkGuard()

    async def _scenario():
        async def _hold_briefly():
            with g:
                await asyncio.sleep(0.05)
        task = asyncio.create_task(_hold_briefly())
        await asyncio.sleep(0.01)          # let it actually acquire
        assert g.busy
        ok = await g.wait_idle(timeout=2.0, poll_s=0.01)
        await task
        return ok

    assert run(_scenario()) is True


def test_wait_idle_times_out_and_returns_false_without_hanging():
    g = GpuWorkGuard()

    async def _scenario():
        async def _hold_forever():
            with g:
                await asyncio.sleep(10)
        task = asyncio.create_task(_hold_forever())
        await asyncio.sleep(0.01)
        ok = await g.wait_idle(timeout=0.1, poll_s=0.01)
        task.cancel()
        return ok

    assert run(_scenario()) is False
