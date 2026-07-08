"""End-to-end GPU orchestration (Phase F2b): `solver solve --gpu`.

Ties the pieces together into one autonomous run: provision an ephemeral B200
(`PodSession`, guaranteed teardown) → wait for SSH → **bootstrap** the harness
(the exact sequence validated live on a real pod, §3b) → drive the fleet through
an `SshExecutor` → terminate. A crash anywhere still tears the pod down
(PodSession's `finally` + atexit + signal last-resort).

The bootstrap is deliberately a flat list of the shell steps that were run by
hand during discovery — apt tools (incl. build-essential + python3-dev for
Triton's runtime cc, and rsync), `uv`, clone the pinned harness, `uv sync`,
install `run_eval.sh` + config, and rsync only the problems this run needs.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import time
from pathlib import Path

from .loop import run_fleet
from .pod import PodHandle, PodProvider, PodSession, PodSpec
from .ssh_exec import RUN_EVAL_SH, PodConn, SshExecutor

HARNESS_REPO = "https://github.com/NVIDIA/SOL-ExecBench.git"
# cold-L2 + median (the scored conditions) are baked into the pinned eval_driver,
# so those already match the leaderboard. Clock-locking is the one thing we can't
# match: the leaderboard pins the B200 to 1500/3996 MHz, but a RunPod *container*
# has no privilege to set GPU clocks — and the harness rejects every workload if
# lock_clocks=True without real locking (eval_driver §356). So we always run
# UNLOCKED (boost) → a roughly constant ~10-20% optimistic offset vs the
# leaderboard, which we record per-eval (run_eval.sh samples the real clocks) and
# calibrate against submissions. Ranking is unaffected by a constant offset.
DEFAULT_CONFIG = {"warmup_runs": 10, "iterations": 50, "lock_clocks": False, "seed": 200}


def harness_ref(pyproject: str | Path = "pyproject.toml") -> str:
    """The exact harness commit pinned in the `[bench]` extra — so the pod runs
    the same grader the engine was built against (no drift)."""
    text = Path(pyproject).read_text()
    m = re.search(r"SOL-ExecBench\.git@([0-9a-f]{40})", text)
    if not m:
        raise RuntimeError("could not find the pinned SOL-ExecBench SHA in pyproject.toml")
    return m.group(1)


async def wait_ssh_ready(provider: PodProvider, pod_id: str, *,
                         timeout: float = 600, poll: float = 6, log=print) -> PodHandle:
    """Poll until the pod exposes its SSH host/port (runtime up)."""
    t0 = time.monotonic()
    while True:
        h = await provider.status(pod_id)
        if h.ssh_host and h.ssh_port:
            log(f"[gpu] pod {pod_id} ssh ready at {h.ssh_host}:{h.ssh_port} (${h.cost_per_hr}/hr)")
            return h
        if h.status in ("EXITED", "TERMINATED"):
            raise RuntimeError(f"pod {pod_id} died before SSH ({h.status})")
        if time.monotonic() - t0 > timeout:
            raise RuntimeError(f"pod {pod_id} exposed no SSH within {timeout}s")
        await _sleep(poll)


async def wait_ssh_login(conn: PodConn, *, timeout: float = 300, poll: float = 5, log=print) -> None:
    """Poll until sshd actually answers (the start command's apt-install finished)."""
    t0 = time.monotonic()
    while True:
        try:
            rc, out, _ = await conn.run("echo ok", timeout=15)
            if rc == 0 and "ok" in out:
                log("[gpu] sshd is answering")
                return
        except Exception:
            pass
        if time.monotonic() - t0 > timeout:
            raise RuntimeError(f"sshd never answered within {timeout}s")
        await _sleep(poll)


async def bootstrap(conn: PodConn, *, problems_dir: str | Path, ids: list[int],
                    config: dict | None = None, pyproject: str | Path = "pyproject.toml",
                    log=print) -> None:
    """Install the harness + stage problems on a fresh pod (validated recipe)."""
    sha = harness_ref(pyproject)
    log("[gpu] bootstrap: apt tools (build-essential, python3-dev, rsync, git) ...")
    await conn.run("export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; "
                   "apt-get install -y -qq curl git ca-certificates rsync git-lfs "
                   "build-essential python3-dev", timeout=420)
    log("[gpu] bootstrap: uv ...")
    await conn.run("command -v uv >/dev/null 2>&1 || "
                   "(curl -LsSf https://astral.sh/uv/install.sh | sh)", timeout=180)
    log(f"[gpu] bootstrap: clone harness @ {sha[:7]} ...")
    await conn.run(
        f"cd /root && (test -d SOL-ExecBench || git clone --depth 1 -q {HARNESS_REPO} SOL-ExecBench) && "
        f"cd SOL-ExecBench && git fetch -q --depth 1 origin {sha} && git checkout -q {sha}", timeout=300)
    log("[gpu] bootstrap: uv sync (torch cu13 + cutlass + cupti — a few minutes) ...")
    await conn.run(
        "cd /root/SOL-ExecBench && PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH "
        "CUDA_HOME=/usr/local/cuda uv sync", timeout=1800)
    log("[gpu] bootstrap: install run_eval.sh + config + stage problems ...")
    await conn.write_text("/root/run_eval.sh", RUN_EVAL_SH)
    await conn.write_text("/root/config.json", json.dumps(config or DEFAULT_CONFIG))
    await conn.mkdir("/root/problems")
    await conn.mkdir("/root/solver-run")
    problems_dir = Path(problems_dir)
    dirs = [problems_dir / str(t) for t in ids if (problems_dir / str(t)).is_dir()]
    if dirs:
        await conn.rsync_up(dirs, "/root/problems/")
    log(f"[gpu] bootstrap complete — {len(dirs)} problem(s) staged")


async def solve_on_gpu(ids, agents, cfg, *, runs_dir, seeds_fn, knowledge, families, names,
                       provider: PodProvider, spec: PodSpec | None = None,
                       problems_dir: str | Path = "problems", config: dict | None = None,
                       key: str = "~/.ssh/id_ed25519", max_concurrency: int = 0,
                       max_lifetime_min: float | None = None, shuffle: bool = False,
                       reflect_first: bool = False, reflect_every_min: float = 0,
                       reflect_model: str | list[str] = "", reuse_pod: bool = False,
                       log=print) -> None:
    """Provision → bootstrap → run the fleet on the pod → guaranteed teardown.

    `max_lifetime_min` is a HARD wall-clock cap on total pod uptime (create →
    terminate, bootstrap included): when hit, the fleet is cancelled and the pod is
    terminated via the API (the only reliable way to stop RunPod billing). The run
    is resumable, so a cut-off is safe. Primary cap; a cron backstop reaps the pod
    even if this process is hard-killed.

    `reuse_pod=True` (--gpu-reuse-pod) adopts an already-RUNNING tagged pod instead
    of reaping+recreating one, and leaves it running on a manual SIGINT/SIGTERM
    restart (e.g. after a prompt/code fix) instead of tearing it down — `bootstrap`
    still runs unconditionally (it's near-instant on a warm pod: `uv sync`/git
    clone/apt-install all no-op when already satisfied) so the freshest run_eval.sh
    /config/problem set is always pushed. The cap is anchored to the POD's own
    RunPod-reported rental start time (`rented_at`), not this process's launch
    time, so a string of quick restarts never resets/extends the safety clock."""
    spec = spec or PodSpec()
    t_start = time.monotonic()
    session = PodSession(provider, spec, reuse=reuse_pod, terminate_on_signal=not reuse_pod)
    async with session as pod:
        if session.adopted:
            log(f"[gpu] adopted existing pod {pod.id} (--gpu-reuse-pod) — skipping create/reap")
        else:
            log(f"[gpu] pod {pod.id} created; waiting for SSH ...")
        h = await wait_ssh_ready(provider, pod.id)
        conn = PodConn(host=h.ssh_host, port=int(h.ssh_port), key=key,
                       control_path="~/.ssh/cm/%r@%h:%p")
        await wait_ssh_login(conn)
        await bootstrap(conn, problems_dir=problems_dir, ids=ids, config=config, log=log)
        executor = SshExecutor(conn, problems_dir=problems_dir)
        log(f"[gpu] running fleet over {len(ids)} problem(s) on the B200 ...")
        fleet = run_fleet(ids, executor, agents, cfg, runs_dir=runs_dir, seeds_fn=seeds_fn,
                          knowledge=knowledge, families=families, names=names,
                          max_concurrency=max_concurrency, shuffle=shuffle,
                          reflect_first=reflect_first, reflect_every_min=reflect_every_min,
                          reflect_model=reflect_model)
        if max_lifetime_min:
            elapsed = _pod_age_s(h, t_start)
            remaining = max(1.0, max_lifetime_min * 60 - elapsed)
            log(f"[gpu] hard {max_lifetime_min:.0f}-min pod cap ({'pod age' if h.rented_at else 'this launch'}"
                f" = {elapsed/60:.0f} min so far) — fleet has ~{remaining/60:.0f} min before forced teardown")
            try:
                await asyncio.wait_for(fleet, timeout=remaining)
                log("[gpu] fleet done — terminating pod")
            except asyncio.TimeoutError:
                log(f"[gpu] {max_lifetime_min:.0f}-min cap reached — cancelling fleet + terminating pod")
        else:
            await fleet
            log("[gpu] fleet done — terminating pod")


def _pod_age_s(h: PodHandle, fallback_t_start: float) -> float:
    """Seconds since the pod was actually rented (RunPod's own record), so
    --gpu-max-hours reflects true pod age across a --gpu-reuse-pod restart chain
    rather than resetting to 0 on each relaunch. Falls back to this process's own
    elapsed time if RunPod didn't report a parseable rental timestamp."""
    if h.rented_at:
        try:
            rented = datetime.datetime.fromisoformat(h.rented_at)
            return max(0.0, (datetime.datetime.now(datetime.timezone.utc) - rented).total_seconds())
        except ValueError:
            pass
    return time.monotonic() - fallback_t_start


async def _sleep(s: float) -> None:
    import asyncio
    await asyncio.sleep(s)
