"""CliAgent adapter tests — driven by a fake CLI (docs/agent.md).

A fake "coding agent" writes results to the known files (kernel.py + strategy.txt
+ handoff.md / design.md) and emits a codex-style JSON event stream, so the whole
contract — workdir setup → run → collect files → Solution, trajectory persistence,
token parsing — is exercised with no real CLI/auth/GPU.
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

# A fake CLI: reads the prompt, writes the requested known file(s), streams codex JSONL.
_FAKE = '''\
import sys, pathlib, hashlib, json
model, prompt = sys.argv[1], sys.argv[2]
def ev(o): print(json.dumps(o))
ev({"type": "thread.started", "thread_id": "t1"})
if "kernel.py" in prompt:                       # plan
    tag = hashlib.sha1(prompt.encode()).hexdigest()[:8]
    pathlib.Path("kernel.py").write_text(f"# {model} {tag}\\ndef run(*t): return t[-1]\\n")
    pathlib.Path("strategy.txt").write_text("fused elementwise path")
    pathlib.Path("handoff.md").write_text("reserve play: radix-sort segmented reduction")
elif "design.md" in prompt:
    pathlib.Path("design.md").write_text("# design\\nroofline")
ev({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 50,
     "reasoning_output_tokens": 20, "cached_input_tokens": 0}})
'''


def _fake_spec(tmp_path):
    fake = tmp_path / "fake_cli.py"
    fake.write_text(_FAKE)
    return CliSpec("fake", cmd=[sys.executable, str(fake), "{model}", "{prompt}"], stream="codex")


def _agent(spec, tmp_path, **kw):
    return CliAgent(spec, "gpt-5.5", runs_dir=tmp_path / "runs",
                    problems_dir=tmp_path / "none", timeout=30, **kw)


def _fake_ctx(task_id=7):
    return SimpleNamespace(task_id=task_id, design="op graph + roofline",
                           iters=0, frontier=SimpleNamespace(members=[]))


def test_plan_writes_files_and_persists_trajectory(tmp_path):
    agent = _agent(_fake_spec(tmp_path), tmp_path)
    cand = run(agent.plan(parent=None, ctx=_fake_ctx()))
    assert cand.agent == "fake" and cand.model == "gpt-5.5"
    assert cand.solution["sources"][0]["path"] == "kernel.py"
    assert "gpt-5.5" in cand.solution["sources"][0]["content"]
    assert cand.solution["spec"]["languages"] == ["pytorch"]
    assert cand.strategy == "fused elementwise path"          # read from strategy.txt, not stdout
    assert cand.handoff == "reserve play: radix-sort segmented reduction"   # from handoff.md
    assert cand.tokens["in"] == 100 and cand.tokens["out"] == 50   # parsed from the stream
    # kernel + trajectory + inputs persist together under work/<cand_id>/
    wd = tmp_path / "runs" / "7" / "work" / cand.cand_id
    assert (wd / "kernel.py").exists() and (wd / "trajectory.jsonl").exists()
    assert (wd / "DESIGN.md").read_text().startswith("op graph")
    assert cand.trajectory == str(wd / "trajectory.jsonl")


def test_design_reads_file(tmp_path):
    agent = _agent(_fake_spec(tmp_path), tmp_path)
    assert "roofline" in run(agent.design(7))                 # from design.md


def test_context_md_renders_reserve_plays():
    # the per-problem playbook (reserve plays) is fed to the next agent via CONTEXT.md
    from solver.engine.cli_agent import _context_md
    ctx = SimpleNamespace(design="", sibling_hint=None, recent_failures=[],
                          frontier=SimpleNamespace(members=[]),
                          playbook=[{"cand": "abc12345", "strategy": "atomic scatter",
                                     "handoff": "radix-sort + atomic-free segmented reduction"}])
    md = _context_md(parent=None, ctx=ctx)
    assert "Reserve plays" in md
    assert "radix-sort + atomic-free segmented reduction" in md
    assert "atomic scatter" in md                             # the strategy that flagged it


def test_timeout_raises(tmp_path):
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(5)\n")
    spec = CliSpec("slow", cmd=[sys.executable, str(slow)], stream="codex")
    agent = CliAgent(spec, "m", runs_dir=tmp_path / "runs", problems_dir=tmp_path / "none", timeout=0.3)
    try:
        run(agent.plan(parent=None, ctx=_fake_ctx()))
        assert False, "expected a timeout"
    except RuntimeError as e:
        assert "timed out" in str(e)


def test_plan_raises_when_no_kernel_written(tmp_path):
    noop = tmp_path / "noop.py"
    noop.write_text("import sys\nsys.stderr.write('bad model\\n')\nsys.exit(1)\n")
    spec = CliSpec("noop", cmd=[sys.executable, str(noop), "{model}", "{prompt}"], stream="codex")
    agent = _agent(spec, tmp_path)
    try:
        run(agent.plan(parent=None, ctx=_fake_ctx()))
        assert False, "expected a no-kernel error"
    except RuntimeError as e:
        assert "no kernel" in str(e) and "bad model" in str(e)


def test_cli_agent_drives_the_real_loop(tmp_path):
    spec = _fake_spec(tmp_path)
    cfg = Config(tiers=[Tier("t", [Perspective("fake", "gpt-5.5")])],
                 max_iterations=3, max_gpu_evals=9, plateau_cycles=999, escalate_ceiling=1.1)
    agents = make_agents(cfg, {"fake": spec}, runs_dir=tmp_path / "runs",
                         problems_dir=tmp_path / "none", timeout=30)
    seeds_fn = lambda t: [{"spec": {"languages": ["pytorch"]},
                           "sources": [{"path": "kernel.py", "content": "def run(*t): return t[-1]"}]}]
    check_fn = lambda sol, t: (bool(sol.get("sources")), [])
    ctx = run(solve_problem(7, StubExecutor(), agents, cfg, runs_dir=tmp_path / "runs",
                            seeds_fn=seeds_fn, check_fn=check_fn))
    evs = journal_mod.read(ctx.path)
    kinds = [e["ev"] for e in evs]
    assert "plan_done" in kinds and "terminated" in kinds
    pd = next(e for e in evs if e["ev"] == "plan_done")
    assert pd["agent"] == "fake" and pd["model"] == "gpt-5.5" and pd["tok_in"] == 100
    assert pd.get("trajectory", "").endswith("trajectory.jsonl")
