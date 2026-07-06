"""GpuQueueExecutor (F1) durability tests — LocalPod + fake harness, no GPU/SSH.

Proves the two guarantees before renting anything:
  · kill-laptop-mid-eval → recover the cached result, never re-run
  · kill-pod-mid-eval → resubmit on a new pod; no double-pay when the result survived
plus single-flight and that it drives the real solve loop.
"""

from __future__ import annotations

import asyncio
import shutil

from solver.engine import (
    Config,
    FileQueueTransport,
    GpuQueueExecutor,
    Perspective,
    StubAgent,
    Tier,
    Worker,
    solve_problem,
    stub_agents,
)
from solver.engine.sim import hash_score_outcome

run = asyncio.run


class Counter:
    """Wrap a harness to count how many times the 'GPU' actually ran."""

    def __init__(self, harness):
        self._h = harness
        self.calls = 0

    def __call__(self, solution, task_id):
        self.calls += 1
        return self._h(solution, task_id)


def _sol(tag: str) -> dict:
    return {"spec": {"languages": ["pytorch"]}, "sources": [{"path": "k.py", "content": tag}]}


async def _worker(pod, harness):
    stop = asyncio.Event()
    task = asyncio.create_task(Worker(pod, harness).serve(stop, interval=0.002))
    return stop, task


def test_no_double_pay_when_result_exists(tmp_path):
    async def scenario():
        h = Counter(hash_score_outcome())
        pod = tmp_path / "pod"
        stop, wt = await _worker(pod, h)
        ex = GpuQueueExecutor(FileQueueTransport(pod), poll_interval=0.002)
        sol = _sol("kernel-A")
        r1 = await ex.evaluate(sol, 1)
        r2 = await ex.evaluate(sol, 1)              # same job_id → cached, no re-run
        stop.set(); await wt
        return r1, r2, h.calls
    r1, r2, calls = run(scenario())
    assert calls == 1                                # the GPU ran exactly once
    assert r1.sol_score == r2.sol_score is not None


def test_kill_laptop_mid_eval_recovers(tmp_path):
    async def scenario():
        h = Counter(hash_score_outcome())
        pod = tmp_path / "pod"
        stop, wt = await _worker(pod, h)
        sol = _sol("kernel-B")
        r1 = await GpuQueueExecutor(FileQueueTransport(pod), poll_interval=0.002).evaluate(sol, 1)
        # laptop "crashes" and restarts: a brand-new executor, same (surviving) pod
        r2 = await GpuQueueExecutor(FileQueueTransport(pod), poll_interval=0.002).evaluate(sol, 1)
        stop.set(); await wt
        return r1, r2, h.calls
    r1, r2, calls = run(scenario())
    assert calls == 1                                # recovered the cached result, no re-run
    assert r1.sol_score == r2.sol_score


def test_kill_pod_mid_eval_resubmits(tmp_path):
    async def scenario():
        h = Counter(hash_score_outcome())
        pod = tmp_path / "pod"
        stop, wt = await _worker(pod, h)
        sol = _sol("kernel-C")
        await GpuQueueExecutor(FileQueueTransport(pod), poll_interval=0.002).evaluate(sol, 1)
        # pod dies: stop the worker and wipe the pod (ephemeral disk gone)
        stop.set(); await wt
        shutil.rmtree(pod)
        # a fresh pod comes up
        stop2, wt2 = await _worker(pod, h)
        await GpuQueueExecutor(FileQueueTransport(pod), poll_interval=0.002).evaluate(sol, 1)
        stop2.set(); await wt2
        return h.calls
    assert run(scenario()) == 2                       # re-run on the new pod (result was lost with the pod)


def test_single_flight_under_concurrency(tmp_path):
    async def scenario():
        h = Counter(hash_score_outcome())
        pod = tmp_path / "pod"
        stop, wt = await _worker(pod, h)
        ex = GpuQueueExecutor(FileQueueTransport(pod), poll_interval=0.001)
        await asyncio.gather(*(ex.evaluate(_sol(f"k{i}"), 1) for i in range(6)))
        stop.set(); await wt
        return ex.max_concurrent, h.calls
    peak, calls = run(scenario())
    assert peak == 1                                 # re-entrancy assertion never tripped
    assert calls == 6                                # six distinct kernels, each run once


def test_gpu_executor_drives_the_loop(tmp_path):
    async def scenario():
        h = Counter(hash_score_outcome())
        pod = tmp_path / "pod"
        stop, wt = await _worker(pod, h)
        ex = GpuQueueExecutor(FileQueueTransport(pod), poll_interval=0.002)
        cfg = Config(tiers=[Tier("t", [Perspective("claude", "haiku")])],
                     max_iterations=3, max_gpu_evals=9, plateau_cycles=999, escalate_ceiling=1.1)
        # unique candidate per iter (distinct sources → distinct job_id)
        planner = lambda p, parent, ctx: {"solution": _sol(f"cand-{ctx.iters}"), "strategy": "s"}
        agents = stub_agents(cfg.perspectives, planner)
        seeds = lambda t: [_sol("seed")]
        ctx = await solve_problem(1, ex, agents, cfg, runs_dir=tmp_path / "runs",
                                  seeds_fn=seeds, check_fn=lambda s, t: (True, []))
        stop.set(); await wt
        return ctx, h.calls
    ctx, calls = run(scenario())
    assert ctx.iters == 3
    assert calls == ctx.evals                        # every GPU eval = one harness run, no re-runs
