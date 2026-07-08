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


# --------------------------------------------------------------------------- #
# Graceful shutdown — a signal must wait for any in-flight GPU eval-to-accept
# span (GpuWorkGuard) before tearing anything down, instead of interrupting it
# mid-way (the live incident this fixes: a scored candidate that never got its
# accept recorded because a SIGTERM landed right after exec_done). These call
# `_on_signal_async`/`_graceful_shutdown` DIRECTLY rather than sending a real
# OS signal — `arm_signals=False` throughout, so no process-wide handler is
# ever installed by this test file, and `os.kill`/`signal.signal` are
# monkeypatched so a test never actually delivers a signal to the test runner.
# --------------------------------------------------------------------------- #
def test_graceful_shutdown_waits_for_the_gpu_guard_before_terminating(monkeypatch):
    import signal as signal_mod

    from solver.engine.gpu_guard import GpuWorkGuard

    killed = []
    monkeypatch.setattr(signal_mod, "signal", lambda *a: None)
    monkeypatch.setattr("os.kill", lambda *a: killed.append(a))

    guard = GpuWorkGuard()
    order = []

    async def scenario():
        p = MockProvider()
        s = _session(p, gpu_guard=guard, shutdown_grace_s=5.0)
        async with s:
            async def _hold_then_release():
                with guard:
                    order.append("holding")
                    await asyncio.sleep(0.05)
                order.append("released")
            task = asyncio.create_task(_hold_then_release())
            await asyncio.sleep(0.01)
            assert guard.busy
            await s._graceful_shutdown(signal_mod.SIGTERM)
            order.append("terminated")
            await task
        return p, s

    p, s = run(scenario())
    assert order == ["holding", "released", "terminated"]   # waited for release BEFORE terminating
    assert p.terminated == 1                                 # _terminate_sync actually ran
    assert killed                                            # handed off to re-raise, didn't just return


def test_graceful_shutdown_proceeds_anyway_once_the_grace_period_elapses(monkeypatch):
    """Never hang a shutdown forever — a bound, not a guarantee."""
    import signal as signal_mod

    from solver.engine.gpu_guard import GpuWorkGuard

    monkeypatch.setattr(signal_mod, "signal", lambda *a: None)
    monkeypatch.setattr("os.kill", lambda *a: None)

    guard = GpuWorkGuard()

    async def scenario():
        p = MockProvider()
        s = _session(p, gpu_guard=guard, shutdown_grace_s=0.05)
        async with s:
            async def _hold_forever():
                with guard:
                    await asyncio.sleep(10)
            task = asyncio.create_task(_hold_forever())
            await asyncio.sleep(0.01)
            await s._graceful_shutdown(signal_mod.SIGTERM)   # must return despite guard still busy
            task.cancel()
        return p

    p = run(scenario())
    assert p.terminated == 1


def test_graceful_shutdown_skips_the_wait_without_a_guard(monkeypatch):
    """No gpu_guard configured (e.g. a caller that doesn't care) -> terminate
    immediately, same as before this feature existed."""
    import signal as signal_mod

    monkeypatch.setattr(signal_mod, "signal", lambda *a: None)
    monkeypatch.setattr("os.kill", lambda *a: None)

    async def scenario():
        p = MockProvider()
        s = _session(p)                     # no gpu_guard
        async with s:
            await s._graceful_shutdown(signal_mod.SIGTERM)
        return p

    p = run(scenario())
    assert p.terminated == 1


def test_on_signal_async_only_schedules_one_shutdown_for_repeat_signals(monkeypatch):
    """A second Ctrl-C while already shutting down must not spawn a second
    graceful-shutdown task — the original (already bounded) wait just keeps
    going."""
    import signal as signal_mod

    calls = []

    async def scenario():
        p = MockProvider()
        s = _session(p)

        async def _fake_graceful_shutdown(sig):
            calls.append(sig)

        s._graceful_shutdown = _fake_graceful_shutdown
        async with s:
            s._on_signal_async(signal_mod.SIGTERM)
            s._on_signal_async(signal_mod.SIGTERM)          # repeat — must be ignored
            await asyncio.sleep(0.01)                        # let the scheduled task run
        return calls

    calls = run(scenario())
    assert calls == [signal_mod.SIGTERM]
