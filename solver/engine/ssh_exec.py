"""SshExecutor (Phase F2b) — the real Executor over SSH to a rented B200 pod.

Implements the same `Executor.evaluate` as StubExecutor/GpuQueueExecutor, so the
solve loop is unchanged. Per candidate it: builds a harness `solution.json`
(`solution_to_harness_json`), ships it to the pod, runs the `sol-execbench` CLI
there via `/root/run_eval.sh` (the validated recipe, docs/gpu-execution.md §3),
pulls the trace JSONL back, and **scores it on the laptop** against the problem's
SOL metadata (`map_traces_to_result`). The pod holds only the harness; every
result is laptop-authoritative.

Durability: single-flight (the GPU lock) + **idempotent per-candidate job dirs**
keyed by `task-<Solution.hash>`. A re-eval of the same candidate reuses the
pod-side `trace.jsonl` instead of re-running — so a laptop crash mid-eval
recovers for free. The pod-side driver (`run_eval.sh`) exits 1 when not every
workload passes; that is normal, so we key off the *trace file*, never the exit
code.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import solution_hash
from .executor import EvalResult
from .harness import map_traces_to_result, solution_to_harness_json, traces_from_jsonl


def _iso(t: dt.datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


@dataclass
class PodConn:
    """A thin async SSH/rsync client to one pod (ControlMaster-multiplexed)."""

    host: str
    port: int
    key: str = "~/.ssh/id_ed25519"
    user: str = "root"
    control_path: str | None = None
    connect_timeout: int = 15

    def _opts(self) -> list[str]:
        opts = ["-i", os.path.expanduser(self.key), "-p", str(self.port),
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR", "-o", f"ConnectTimeout={self.connect_timeout}",
                "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]
        if self.control_path:
            opts += ["-o", "ControlMaster=auto", "-o", f"ControlPath={os.path.expanduser(self.control_path)}",
                     "-o", "ControlPersist=15m"]
        return opts

    async def run(self, remote_cmd: str, *, timeout: float = 600.0,
                  stdin_data: str | bytes | None = None) -> tuple[int, str, str]:
        argv = ["ssh", *self._opts(), f"{self.user}@{self.host}", remote_cmd]
        stdin = asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=stdin, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        data = stdin_data.encode() if isinstance(stdin_data, str) else stdin_data
        try:
            out, err = await asyncio.wait_for(proc.communicate(input=data), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"ssh command timed out after {timeout}s: {remote_cmd[:80]}")
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")

    async def exists(self, remote_path: str) -> bool:
        rc, _, _ = await self.run(f"test -f {shlex.quote(remote_path)}", timeout=self.connect_timeout + 15)
        return rc == 0

    async def write_text(self, remote_path: str, text: str) -> None:
        rc, _, err = await self.run(f"cat > {shlex.quote(remote_path)}", stdin_data=text, timeout=120)
        if rc != 0:
            raise RuntimeError(f"failed writing {remote_path}: {err[:200]}")

    async def read_text(self, remote_path: str) -> str | None:
        rc, out, _ = await self.run(f"cat {shlex.quote(remote_path)} 2>/dev/null", timeout=120)
        return out if rc == 0 else None

    async def mkdir(self, remote_path: str) -> None:
        await self.run(f"mkdir -p {shlex.quote(remote_path)}", timeout=self.connect_timeout + 15)

    async def rsync_up(self, locals_: list[str | Path], remote_dir: str, *, timeout: float = 900) -> None:
        rsh = "ssh " + " ".join(shlex.quote(o) for o in self._opts())
        argv = ["rsync", "-az", "-e", rsh, *[str(p) for p in locals_],
                f"{self.user}@{self.host}:{remote_dir}"]
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"rsync timed out after {timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(f"rsync failed ({proc.returncode}): {err.decode(errors='replace')[:300]}")


@dataclass
class SshExecutor:
    """Real Executor: evaluate candidates on a rented B200 over SSH."""

    conn: PodConn
    problems_dir: str | Path = "problems"
    remote_root: str = "/root/solver-run"
    remote_problems: str = "/root/problems"
    run_script: str = "/root/run_eval.sh"
    config_path: str = "/root/config.json"
    timeout: float = 900.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _in_flight: bool = field(default=False, repr=False)
    max_concurrent: int = 0

    async def evaluate(self, solution: dict, task_id: int, *, profile: bool = False) -> EvalResult:
        job_id = f"{task_id}-{solution_hash(solution)[:12]}"
        jobdir = f"{self.remote_root}/{job_id}"
        trace_path = f"{jobdir}/trace.jsonl"
        async with self._lock:                         # single-flight — the GPU
            assert not self._in_flight, "GPU re-entered: single-flight violated"
            self._in_flight = True
            self.max_concurrent = max(self.max_concurrent, 1)
            try:
                started = dt.datetime.now(dt.timezone.utc)
                t0 = time.monotonic()
                err = ""
                if not await self.conn.exists(trace_path):        # idempotent: reuse a prior trace
                    soljson = solution_to_harness_json(solution, task_id, self.problems_dir,
                                                       name=f"t{task_id}_{job_id}")
                    await self.conn.mkdir(jobdir)
                    await self.conn.write_text(f"{jobdir}/solution.json",
                                               __import__("json").dumps(soljson))
                    # exit 1 == "not all workloads passed" — normal; we key off the trace file.
                    _, _, err = await self.conn.run(
                        f"bash {self.run_script} {self.remote_problems}/{task_id} "
                        f"{jobdir}/solution.json {self.config_path} {trace_path}",
                        timeout=self.timeout)
                text = await self.conn.read_text(trace_path)
                if not text or not text.strip():
                    # no traces → compile/validation crash before eval. Frontier-safe, carries the log.
                    r = map_traces_to_result(task_id, [], solution_status="COMPILE_ERROR",
                                             asi={"error": (err or "no trace produced")[-2000:]},
                                             problems_dir=self.problems_dir)
                else:
                    rows = traces_from_jsonl(text)
                    r = map_traces_to_result(task_id, rows, problems_dir=self.problems_dir)
                env_text = await self.conn.read_text(f"{jobdir}/env.json")   # GPU conditions
                if env_text:
                    try:
                        r.asi = {**(r.asi or {}), "gpu": __import__("json").loads(env_text)}
                    except Exception:
                        pass
                r.raw = {**(r.raw or {}), "job_id": job_id, "started": _iso(started),
                         "ended": _iso(dt.datetime.now(dt.timezone.utc)),
                         "gpu_s": round(time.monotonic() - t0, 3)}
                return r
            finally:
                self._in_flight = False


# The pod-side driver the executor invokes. Written by bootstrap (§3b); kept here
# so the exact validated invocation lives next to the code that calls it. It also
# samples the GPU's clocks/temp/power *while the eval runs* → env.json, so every
# measurement records the conditions it was taken under (the clock the kernel
# actually ran at is what makes us optimistic vs the leaderboard's locked 1500MHz).
RUN_EVAL_SH = r"""#!/bin/bash
# run_eval.sh <problem_dir> <solution_json> <config_json> <out_trace>
# Exit 1 == not all workloads passed (normal); the caller reads the trace, not $?.
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/SOL-ExecBench
JOBDIR=$(dirname "$4")
( while true; do nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw \
    --format=csv,noheader,nounits; sleep 0.25; done ) >"$JOBDIR/clocks.csv" 2>/dev/null &
SAMPLER=$!
uv run sol-execbench "$1" --solution "$2" --config "$3" --output "$4" \
    --compile-timeout 180 --timeout 600
RC=$?
kill "$SAMPLER" 2>/dev/null
python3 - "$JOBDIR/clocks.csv" "$JOBDIR/env.json" <<'PY'
import sys, json, statistics
rows = []
for line in open(sys.argv[1]).read().splitlines():
    p = [x.strip() for x in line.split(",")]
    try: rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
    except (ValueError, IndexError): continue
col = lambda i: [r[i] for r in rows]
med = lambda i: statistics.median(col(i)) if rows else None
json.dump({"sm_mhz_median": med(0), "sm_mhz_max": max(col(0)) if rows else None,
           "mem_mhz_median": med(1), "temp_c_max": max(col(2)) if rows else None,
           "power_w_max": max(col(3)) if rows else None, "samples": len(rows),
           "harness_lock_preset_mhz": 1500}, open(sys.argv[2], "w"))
PY
exit $RC
"""
