"""CliAgent adapter tests — driven by a fake CLI (docs/agent.md).

A tiny fake "coding agent" (writes a kernel file on `edit`, echoes on `ask`)
exercises workdir setup → subprocess run → collect → Solution and the four
methods, with no real CLI, network, auth, or GPU.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from solver import journal as journal_mod
from solver.engine import (
    CliAgent,
    CliSpec,
    Config,
    Perspective,
    StubExecutor,
    Tier,
    make_agents,
    solve_problem,
)

run = asyncio.run

_FAKE = '''\
import sys, os, pathlib, hashlib
mode, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "edit":
    tag = hashlib.sha1(prompt.encode()).hexdigest()[:8]
    pathlib.Path("kernel.py").write_text(f"# {model} {tag}\\ndef run(*t):\\n    return t[-1]\\n")
    print("STRATEGY: fused elementwise path")
else:
    if "materially different" in prompt:          # judge
        print(os.environ.get("FAKE_JUDGE", "materially-new"))
    else:                                          # reflect / design
        print("Try vectorized 128-bit loads to raise achieved bandwidth.")
'''


def _fake_spec(tmp_path):
    fake = tmp_path / "fake_cli.py"
    fake.write_text(_FAKE)
    return CliSpec("fake",
                   edit=[sys.executable, str(fake), "edit", "{model}", "{prompt}"],
                   ask=[sys.executable, str(fake), "ask", "{model}", "{prompt}"])


def _fake_ctx(task_id=7):
    return SimpleNamespace(task_id=task_id, design="op graph + roofline",
                           iters=0, frontier=SimpleNamespace(members=[]))


def test_plan_collects_a_solution(tmp_path):
    agent = CliAgent(_fake_spec(tmp_path), "gpt-5.5", runs_dir=tmp_path / "runs", timeout=30)
    cand = run(agent.plan(parent=None, ctx=_fake_ctx()))
    assert cand.agent == "fake" and cand.model == "gpt-5.5"
    assert cand.solution["sources"][0]["path"] == "kernel.py"
    assert "gpt-5.5" in cand.solution["sources"][0]["content"]
    assert cand.solution["spec"]["languages"] == ["pytorch"]
    assert cand.strategy == "fused elementwise path"
    # the §8 context was seeded into the workdir the agent read
    wd = tmp_path / "runs" / "7" / "work" / "cand1"
    assert (wd / "DESIGN.md").read_text().startswith("op graph")
    assert (wd / "CONTEXT.md").exists()


def test_reflect_and_judge_read_stdout(tmp_path):
    spec = _fake_spec(tmp_path)
    agent = CliAgent(spec, "gpt-5.5", runs_dir=tmp_path / "runs", timeout=30)
    refl = run(agent.reflect(SimpleNamespace(cand_id="c1", strategy="s"),
                             SimpleNamespace(sol_score=0.7), "dominated"))
    assert "128-bit" in refl
    assert run(agent.judge(SimpleNamespace(cand_id="c1", strategy="s"),
                           SimpleNamespace(strategy="p"), None)) == "materially-new"
    # the cosmetic branch (parsed from stdout)
    cosmetic = CliAgent(spec, "gpt-5.5", runs_dir=tmp_path / "runs", timeout=30,
                        env={"FAKE_JUDGE": "this is cosmetic"})
    assert run(cosmetic.judge(SimpleNamespace(cand_id="c2", strategy="s"),
                              SimpleNamespace(strategy="p"), None)) == "cosmetic"


def test_timeout_raises(tmp_path):
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(5)\n")
    spec = CliSpec("slow", edit=[sys.executable, str(slow)], ask=[sys.executable, str(slow)])
    agent = CliAgent(spec, "m", runs_dir=tmp_path / "runs", timeout=0.3)
    try:
        run(agent.plan(parent=None, ctx=_fake_ctx()))
        assert False, "expected a timeout"
    except RuntimeError as e:
        assert "timed out" in str(e)


def test_plan_raises_when_no_kernel_produced(tmp_path):
    noop = tmp_path / "noop.py"
    noop.write_text("import sys\nsys.stderr.write('bad model\\n')\nsys.exit(1)\n")
    spec = CliSpec("noop", edit=[sys.executable, str(noop)], ask=[sys.executable, str(noop)])
    agent = CliAgent(spec, "gpt-5.5", runs_dir=tmp_path / "runs", timeout=30)
    try:
        run(agent.plan(parent=None, ctx=_fake_ctx()))
        assert False, "expected a no-kernel error"
    except RuntimeError as e:
        assert "no kernel" in str(e) and "bad model" in str(e)


def test_cli_agent_drives_the_real_loop(tmp_path):
    spec = _fake_spec(tmp_path)
    cfg = Config(tiers=[Tier("t", [Perspective("fake", "gpt-5.5")])],
                 max_iterations=3, max_gpu_evals=9, plateau_cycles=999, escalate_ceiling=1.1)
    agents = make_agents(cfg, {"fake": spec}, runs_dir=tmp_path / "runs", timeout=30)
    seeds_fn = lambda t: [{"spec": {"languages": ["pytorch"]},
                           "sources": [{"path": "kernel.py", "content": "def run(*t): return t[-1]"}]}]
    check_fn = lambda sol, t: (bool(sol.get("sources")), [])
    ctx = run(solve_problem(7, StubExecutor(), agents, cfg, runs_dir=tmp_path / "runs",
                            seeds_fn=seeds_fn, check_fn=check_fn))
    evs = journal_mod.read(ctx.path)
    kinds = [e["ev"] for e in evs]
    assert "plan_done" in kinds and "terminated" in kinds
    assert any(e.get("agent") == "fake" and e.get("model") == "gpt-5.5"
               for e in evs if e["ev"] == "plan_done")
