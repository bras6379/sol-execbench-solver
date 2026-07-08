"""solve_on_gpu (F2b) end-to-end pod-lifecycle integration — offline, MockProvider.

Nothing exercised this function directly before (only its pieces: PodSession in
test_pod.py, SshExecutor in test_ssh_exec.py) — a real bug (`pod.adopted` on a
PodHandle instead of the PodSession) shipped and crashed every real `--gpu`
launch before being caught live. These lock the actual wiring: which object the
`async with` binds, adopt-vs-create logging, and that --gpu-reuse-pod threads
through without raising.
"""

from __future__ import annotations

import asyncio

import pytest

from solver.engine import gpu_run
from solver.engine.config import Config, Perspective, Tier
from solver.engine.pod import MockProvider, PodSpec

run = asyncio.run
SPEC = PodSpec(gpu_type="NVIDIA B200", tag="test-run")


@pytest.fixture(autouse=True)
def _stub_ssh(monkeypatch):
    """No real SSH/bootstrap — solve_on_gpu's pod-lifecycle wiring is what's under
    test here, not the bootstrap shell recipe (covered live) or the fleet loop
    (covered by test_engine.py's StubExecutor suite)."""
    async def fake_wait_ssh_ready(provider, pod_id, **kw):
        h = await provider.status(pod_id)
        h.ssh_host, h.ssh_port = "10.0.0.1", 22
        return h

    async def fake_wait_ssh_login(conn, **kw):
        pass

    async def fake_bootstrap(conn, **kw):
        pass

    async def fake_run_fleet(*a, **kw):
        pass

    monkeypatch.setattr(gpu_run, "wait_ssh_ready", fake_wait_ssh_ready)
    monkeypatch.setattr(gpu_run, "wait_ssh_login", fake_wait_ssh_login)
    monkeypatch.setattr(gpu_run, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(gpu_run, "run_fleet", fake_run_fleet)


def _cfg():
    return Config(tiers=[Tier("t", [Perspective("fake", "m")])])


def test_solve_on_gpu_creates_and_tears_down_a_fresh_pod(tmp_path):
    p = MockProvider()
    run(gpu_run.solve_on_gpu([1], {}, _cfg(), runs_dir=tmp_path, seeds_fn=lambda t: [],
                             knowledge=None, families={}, names={}, provider=p, spec=SPEC,
                             log=lambda *_: None))
    assert p.created == 1 and p.terminated == 1


def test_solve_on_gpu_adopts_existing_pod_with_reuse_flag(tmp_path):
    p = MockProvider()
    existing = p.seed_orphan("test-run")
    run(gpu_run.solve_on_gpu([1], {}, _cfg(), runs_dir=tmp_path, seeds_fn=lambda t: [],
                             knowledge=None, families={}, names={}, provider=p, spec=SPEC,
                             reuse_pod=True, log=lambda *_: None))
    assert p.created == 0                     # adopted, never created a new one
    assert p.terminated == 1                  # normal completion still tears it down (no signal)


def test_solve_on_gpu_logs_adoption_not_creation(tmp_path, capsys=None):
    p = MockProvider()
    p.seed_orphan("test-run")
    logs = []
    run(gpu_run.solve_on_gpu([1], {}, _cfg(), runs_dir=tmp_path, seeds_fn=lambda t: [],
                             knowledge=None, families={}, names={}, provider=p, spec=SPEC,
                             reuse_pod=True, log=logs.append))
    assert any("adopted existing pod" in l for l in logs)
    assert not any("created; waiting for SSH" in l for l in logs)
