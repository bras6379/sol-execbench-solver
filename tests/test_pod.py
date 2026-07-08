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


# ---- --gpu-reuse-pod: adopt-in-place instead of reap+recreate on restart ----

def test_reuse_adopts_a_single_running_pod_instead_of_recreating():
    async def scenario():
        p = MockProvider()
        existing = p.seed_orphan("test-run")           # a pod left running from a prior launch
        async with _session(p, reuse=True) as pod:
            assert pod.id == existing.id
        return p
    p = run(scenario())
    assert p.created == 0                              # never created a new one
    assert p.terminated == 1                           # normal exit still tears it down (no signal)


def test_reuse_creates_fresh_when_no_pod_exists():
    async def scenario():
        p = MockProvider()
        s = _session(p, reuse=True)
        async with s as pod:
            assert s.adopted is False
        return p, s
    p, s = run(scenario())
    assert p.created == 1 and p.terminated == 1


def test_reuse_falls_back_to_reap_and_create_on_ambiguous_state():
    """More than one live tagged pod is an unexpected/ambiguous state — reap all
    and start clean rather than guessing which one to adopt."""
    async def scenario():
        p = MockProvider()
        p.seed_orphan("test-run")
        p.seed_orphan("test-run")
        s = _session(p, reuse=True)
        async with s:
            assert s.adopted is False
        return p
    p = run(scenario())
    assert p.terminated == 3                           # 2 reaped + the fresh one, on normal exit


def test_reuse_survives_a_signal_but_not_normal_completion():
    """The whole point of --gpu-reuse-pod: a caught SIGINT/SIGTERM restart leaves
    the pod running for the next launch to adopt. Normal completion (no signal)
    must still terminate — reuse only changes the restart-in-place path."""
    async def scenario():
        p = MockProvider()
        s = _session(p, reuse=True, terminate_on_signal=False)
        async with s:
            s._signaled = True                         # simulate a caught SIGTERM mid-run
        return p, s
    p, s = run(scenario())
    assert p.terminated == 0                            # left running, not torn down
    assert s._terminated is False

    async def normal_scenario():
        p = MockProvider()
        async with _session(p, reuse=True, terminate_on_signal=False):
            pass                                        # no signal this time
        return p
    p2 = run(normal_scenario())
    assert p2.terminated == 1                            # normal exit still tears it down


def test_parse_rented_at_extracts_runpod_timestamp():
    from solver.engine.pod import _parse_rented_at
    iso = _parse_rented_at("Rented by User: Wed Jul 08 2026 01:13:03 GMT+0000 (Coordinated Universal Time)")
    assert iso == "2026-07-08T01:13:03+00:00"
    assert _parse_rented_at(None) is None
    assert _parse_rented_at("garbage") is None
