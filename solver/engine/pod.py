"""Pod lifecycle (Phase F2) — auto-provision, run, guaranteed self-terminate.

docs/gpu-execution.md §6. `solver solve --gpu` creates an ephemeral RunPod pod,
runs the fleet, and terminates it — with **belt-and-suspenders teardown** so a
crash never strands a paid pod: the `async with` `finally`, plus `atexit` +
SIGINT/SIGTERM as last resort, plus **reap** of stragglers on startup. (The
pod-side dead-man's-switch and lifetime/idle caps ride on top; §6.)

The RunPod API sits behind `PodProvider` — `MockProvider` (in-memory, no spend,
what the tests drive) and `RunPodProvider` (the real SDK). The lifecycle logic
is provider-agnostic, so "never strand a pod" is proven on the laptop.
"""

from __future__ import annotations

import asyncio
import atexit
import datetime
import re
import signal
import time
from dataclasses import dataclass
from typing import Protocol


# Custom-image SSH bootstrap (validated live, docs/gpu-execution.md §3b). The base
# CUDA image ships no sshd; this start command installs it, injects the account
# key RunPod exposes as $PUBLIC_KEY, and keeps the container alive. NOTE: **no
# double-quotes** — the runpod SDK wraps dockerArgs in unescaped GraphQL quotes,
# so a literal " would close the string and expose $. An SSH key has no
# consecutive spaces, so unquoted `echo $PUBLIC_KEY` is safe.
SSH_START = ("bash -c 'apt-get update -qq; "
             "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server >/dev/null 2>&1; "
             "mkdir -p /run/sshd /root/.ssh; chmod 700 /root/.ssh; "
             "echo $PUBLIC_KEY >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; "
             "service ssh start; sleep infinity'")

# The pinned harness runs on CUDA 13 + Blackwell; this devel image has nvcc 13.1
# and cuDNN. Bootstrap adds build-essential + python3-dev (Triton's runtime cc).
DEFAULT_IMAGE = "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04"


@dataclass
class PodSpec:
    gpu_type: str = "NVIDIA B200"
    image: str = DEFAULT_IMAGE
    network_volume_id: str | None = None
    tag: str = "sol-solver"                 # every pod we create carries this → reap can find them
    name: str = ""
    cloud_type: str = "SECURE"              # SECURE supports the public IP we SSH over
    container_disk_gb: int = 60             # torch(cu13)+cutlass+cudnn ≈ 25GB installed
    ports: str = "22/tcp"
    start_cmd: str = SSH_START
    support_public_ip: bool = True


@dataclass
class PodHandle:
    id: str
    status: str = "CREATING"                # CREATING | RUNNING | EXITED | TERMINATED
    ssh_host: str | None = None
    ssh_port: int | None = None
    cost_per_hr: float | None = None
    tag: str = ""
    rented_at: str | None = None            # ISO ts, RunPod's own record — see _parse_rented_at


class PodProvider(Protocol):
    async def create(self, spec: PodSpec) -> PodHandle: ...
    async def status(self, pod_id: str) -> PodHandle: ...
    async def terminate(self, pod_id: str) -> None: ...
    async def list_tagged(self, tag: str) -> list[PodHandle]: ...
    def terminate_sync(self, pod_id: str) -> None: ...     # for atexit/signal (no event loop)


class MockProvider:
    """In-memory provider for tests — no RunPod, no spend, deterministic."""

    def __init__(self, *, ssh=("10.0.0.1", 22), cost_per_hr=5.99, boot_status="RUNNING") -> None:
        self._pods: dict[str, PodHandle] = {}
        self._seq = 0
        self._ssh = ssh
        self._cost = cost_per_hr
        self._boot = boot_status
        self.created = 0
        self.terminated = 0

    async def create(self, spec: PodSpec) -> PodHandle:
        self._seq += 1
        pid = f"pod-{self._seq}"
        self._pods[pid] = PodHandle(id=pid, status=self._boot, ssh_host=self._ssh[0],
                                    ssh_port=self._ssh[1], cost_per_hr=self._cost, tag=spec.tag)
        self.created += 1
        return self._pods[pid]

    async def status(self, pod_id: str) -> PodHandle:
        return self._pods.get(pod_id) or PodHandle(id=pod_id, status="TERMINATED")

    async def terminate(self, pod_id: str) -> None:
        self.terminate_sync(pod_id)

    def terminate_sync(self, pod_id: str) -> None:
        h = self._pods.get(pod_id)
        if h and h.status != "TERMINATED":
            h.status = "TERMINATED"
            self.terminated += 1

    async def list_tagged(self, tag: str) -> list[PodHandle]:
        return [h for h in self._pods.values() if h.tag == tag and h.status != "TERMINATED"]

    # ---- test hooks ----
    def seed_orphan(self, tag: str) -> PodHandle:
        self._seq += 1
        pid = f"orphan-{self._seq}"
        self._pods[pid] = PodHandle(id=pid, status="RUNNING", tag=tag)
        return self._pods[pid]

    def kill(self, pod_id: str) -> None:        # simulate preemption / a pod dying under us
        h = self._pods.get(pod_id)
        if h:
            h.status = "EXITED"


class RunPodProvider:
    """Real RunPod (wraps the `runpod` SDK; needs RUNPOD_API_KEY + credit).

    NOTE: exact SDK arg/field names vary by `runpod` version — verify against the
    installed one on first real use. The `PodProvider` interface + MockProvider
    are what the durability tests cover; this is the thin real adapter.
    """

    def __init__(self, api_key: str) -> None:
        import runpod                       # lazy — only when actually provisioning
        runpod.api_key = api_key
        self._rp = runpod

    async def create(self, spec: PodSpec) -> PodHandle:
        kw = dict(name=spec.name or spec.tag, image_name=spec.image,
                  gpu_type_id=spec.gpu_type, cloud_type=spec.cloud_type,
                  support_public_ip=spec.support_public_ip, start_ssh=True,
                  container_disk_in_gb=spec.container_disk_gb, ports=spec.ports,
                  docker_args=spec.start_cmd)
        if spec.network_volume_id:           # persistent harness volume (optional)
            kw["network_volume_id"] = spec.network_volume_id
        pod = await asyncio.to_thread(self._rp.create_pod, **kw)
        return _from_runpod(pod, spec.tag)

    async def status(self, pod_id: str) -> PodHandle:
        return _from_runpod(await asyncio.to_thread(self._rp.get_pod, pod_id), "")

    async def terminate(self, pod_id: str) -> None:
        await asyncio.to_thread(self.terminate_sync, pod_id)

    def terminate_sync(self, pod_id: str) -> None:
        self._rp.terminate_pod(pod_id)

    async def list_tagged(self, tag: str) -> list[PodHandle]:
        pods = await asyncio.to_thread(self._rp.get_pods)
        return [_from_runpod(p, tag) for p in pods
                if tag in (p.get("name", "") or "") and p.get("desiredStatus") != "TERMINATED"]


_RENTED_AT_RE = re.compile(r"Rented by User: (\w+ \w+ \d+ \d+ \d+:\d+:\d+) GMT\+0000")


def _parse_rented_at(last_status_change: str | None) -> str | None:
    """RunPod's `lastStatusChange` is a human string like 'Rented by User: Wed Jul
    08 2026 01:13:03 GMT+0000 (...)' — the only authoritative record of when a pod
    was actually created (no clean createdAt field in the API). Used to anchor
    --gpu-max-hours to the pod's REAL age across a --gpu-reuse-pod restart, so a
    string of quick restarts can't reset the safety cap's clock. Returns an ISO
    string, or None if the field is missing/unparseable (caller must not assume a
    fresh pod in that case — see solve_on_gpu)."""
    if not last_status_change:
        return None
    m = _RENTED_AT_RE.match(last_status_change)
    if not m:
        return None
    try:
        dt = datetime.datetime.strptime(m.group(1), "%a %b %d %Y %H:%M:%S")
        return dt.replace(tzinfo=datetime.timezone.utc).isoformat()
    except ValueError:
        return None


def _from_runpod(p: dict, tag: str) -> PodHandle:
    rt = (p or {}).get("runtime") or {}
    ports = rt.get("ports") or []
    ssh = next((x for x in ports if x.get("privatePort") == 22), {})
    return PodHandle(id=p.get("id", ""), status=p.get("desiredStatus", "CREATING"),
                     ssh_host=ssh.get("ip"), ssh_port=ssh.get("publicPort"),
                     cost_per_hr=p.get("costPerHr"), tag=tag,
                     rented_at=_parse_rented_at(p.get("lastStatusChange")))


class PodSession:
    """Create a pod on enter, guarantee its termination on exit (any path) —
    UNLESS `reuse=True` and the exit was a caught SIGINT/SIGTERM, in which case
    the pod is deliberately left running for the next launch to adopt (see
    `--gpu-reuse-pod`). Normal completion (the `async with` block finishing, or
    `--gpu-max-hours`'s own internal timeout) always still terminates — `reuse`
    only changes what happens on a manual restart-in-place, never the safety
    backstop for "the run is actually done" or "a real crash happened outside
    the signal path"."""

    def __init__(self, provider: PodProvider, spec: PodSpec, *,
                 ready_timeout: float = 600.0, poll_s: float = 2.0,
                 arm_signals: bool = True, reuse: bool = False,
                 terminate_on_signal: bool = True) -> None:
        self.provider = provider
        self.spec = spec
        self.ready_timeout = ready_timeout
        self.poll_s = poll_s
        self.arm_signals = arm_signals
        self.reuse = reuse
        self.terminate_on_signal = terminate_on_signal
        self.pod: PodHandle | None = None
        self.adopted = False
        self._terminated = False
        self._signaled = False

    async def __aenter__(self) -> PodHandle:
        live = [h for h in await self.provider.list_tagged(self.spec.tag)
                if h.status == "RUNNING"] if self.reuse else []
        if self.reuse and len(live) == 1:
            self.pod = live[0]                           # adopt — skip reap AND create entirely
            self.adopted = True
        else:
            await self.reap()                            # kill stragglers from a prior crash first
            self.pod = await self._create_and_wait()
        if self.arm_signals:
            self._arm_last_resort()
        return self.pod

    async def __aexit__(self, *exc) -> None:
        if self._signaled and not self.terminate_on_signal:
            return                          # deliberate restart-in-place: leave the pod for reuse
        await self.terminate()

    async def reap(self) -> int:
        n = 0
        for h in await self.provider.list_tagged(self.spec.tag):
            await self.provider.terminate(h.id)
            n += 1
        return n

    async def terminate(self) -> None:
        if self.pod and not self._terminated:
            self._terminated = True
            await self.provider.terminate(self.pod.id)

    async def _create_and_wait(self) -> PodHandle:
        h = await self.provider.create(self.spec)
        t0 = time.monotonic()
        while h.status != "RUNNING":
            if h.status in ("EXITED", "TERMINATED"):
                await self.provider.terminate(h.id)     # clean up a pod that died booting
                raise RuntimeError(f"pod {h.id} died before READY ({h.status})")
            if time.monotonic() - t0 > self.ready_timeout:
                await self.provider.terminate(h.id)     # don't leak a stuck pod
                raise RuntimeError(f"pod {h.id} not RUNNING after {self.ready_timeout}s")
            await asyncio.sleep(self.poll_s)
            h = await self.provider.status(h.id)
        return h

    def _arm_last_resort(self) -> None:
        """atexit + signal handlers: a synchronous terminate if the async
        `finally` never runs (hard crash / kill). No-op once terminated — and,
        when `reuse` is on, no-op on a CAUGHT signal specifically (a crash that
        doesn't go through this handler still terminates normally; only a
        deliberate SIGINT/SIGTERM restart-in-place is spared)."""
        atexit.register(self._terminate_sync)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig)

                def handler(signum, frame, _prev=prev):
                    self._signaled = True
                    self._terminate_sync()
                    if callable(_prev):
                        _prev(signum, frame)
                    else:
                        raise KeyboardInterrupt
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass                                    # not the main thread — atexit still covers us

    def _terminate_sync(self) -> None:
        if self._signaled and not self.terminate_on_signal:
            return                          # deliberate restart-in-place: leave the pod for reuse
        if self.pod and not self._terminated:
            self._terminated = True
            try:
                self.provider.terminate_sync(self.pod.id)
            except Exception:
                pass
