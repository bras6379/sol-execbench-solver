"""Pod lifecycle (F2) — "never strand a pod" proofs, MockProvider, no spend.

Every path that can end a run must terminate the pod: normal exit, an exception
mid-run, a boot that never reaches RUNNING, a pod that dies while booting; plus
reaping stragglers from a prior crash on startup.
"""

from __future__ import annotations

import asyncio

from solver.engine.pod import MockProvider, PodSession, PodSpec

run = asyncio.run
SPEC = PodSpec(gpu_type="NVIDIA B200", tag="test-run")


def _session(provider, **kw):
    return PodSession(provider, SPEC, arm_signals=False, ready_timeout=0.2, poll_s=0.005, **kw)


def test_terminates_on_normal_exit():
    async def scenario():
        p = MockProvider()
        async with _session(p) as pod:
            assert pod.status == "RUNNING" and pod.ssh_host
        return p
    p = run(scenario())
    assert p.created == 1 and p.terminated == 1        # created for the run, gone after


def test_terminates_on_exception():
    async def scenario():
        p = MockProvider()
        try:
            async with _session(p):
                raise ValueError("mid-run boom")
        except ValueError:
            pass
        return p
    p = run(scenario())
    assert p.created == 1 and p.terminated == 1        # exception did not strand the pod


def test_reaps_orphans_before_creating():
    async def scenario():
        p = MockProvider()
        p.seed_orphan("test-run")                      # a straggler from a prior crashed run
        p.seed_orphan("test-run")
        async with _session(p):
            pass
        return p
    p = run(scenario())
    assert p.terminated == 3                           # 2 orphans reaped + our own pod


def test_ready_timeout_terminates():
    async def scenario():
        p = MockProvider(boot_status="CREATING")       # never reaches RUNNING
        try:
            async with _session(p):
                assert False, "should have timed out"
        except RuntimeError as e:
            assert "not RUNNING" in str(e)
        return p
    p = run(scenario())
    assert p.created == 1 and p.terminated == 1        # the stuck pod is not leaked


def test_pod_dies_while_booting():
    async def scenario():
        p = MockProvider(boot_status="EXITED")         # preempted during boot
        try:
            async with _session(p):
                assert False
        except RuntimeError as e:
            assert "died before READY" in str(e)
        return p
    p = run(scenario())
    assert p.created == 1 and p.terminated == 1


def test_terminate_is_idempotent():
    async def scenario():
        p = MockProvider()
        s = _session(p)
        async with s:
            pass
        await s.terminate()                            # explicit extra terminate
        s._terminate_sync()                            # last-resort path too
        return p
    p = run(scenario())
    assert p.terminated == 1                           # counted once, no double-terminate
