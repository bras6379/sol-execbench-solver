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

import pytest

from solver import journal as journal_mod
from solver.engine import (
    Candidate,
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


def test_plan_collision_preserves_every_trajectory(tmp_path):
    """A hash collision (always true for a no-op — the parent's workdir already
    exists) must not destroy the fresh call's own trajectory: it used to be
    rmtree'd, silently losing the only evidence of what that specific call did
    and cost. Now it survives as a numbered sibling file."""
    agent = _agent(_fake_spec(tmp_path), tmp_path)
    ctx = _fake_ctx()
    cand1 = run(agent.plan(parent=None, ctx=ctx))
    cand2 = run(agent.plan(parent=None, ctx=ctx))   # same model+prompt -> same fake tag -> same hash
    assert cand1.cand_id == cand2.cand_id           # confirms this exercises the collision path

    dest = tmp_path / "runs" / "7" / "work" / cand1.cand_id
    assert (dest / "trajectory.jsonl").exists()             # first call's trajectory untouched
    assert (dest / "trajectory.dup-1.jsonl").exists()       # second call's trajectory preserved, not deleted
    assert cand2.trajectory == str(dest / "trajectory.dup-1.jsonl")
    assert cand2.tokens["in"] == 100                        # second call's own tokens still captured


def test_plan_workdir_clears_stale_content_from_a_prior_process_lifetime(tmp_path):
    """Regression (2026-07-08): `cand{seq}` is only unique WITHIN one process —
    self._seq resets to 0 on every restart, so a candidate that errored before
    _rekey_workdir ever ran (leaving its temp dir un-renamed) can have its name
    reused by a LATER process's totally unrelated attempt. Live incident: a
    leftover Triton kernel.py from a prior day survived into a fresh CUDA repair
    attempt and got correctly flagged as 'mixed C++/Python' — for content the
    current agent never wrote. The workdir must start clean every time."""
    agent = _agent(_fake_spec(tmp_path), tmp_path)
    stale = tmp_path / "runs" / "7" / "work" / "cand1"
    stale.mkdir(parents=True)
    (stale / "kernel.cu").write_text("// leftover from a completely different, unrelated attempt\n")
    (stale / "strategy.txt").write_text("stale strategy from a prior process lifetime")

    cand = run(agent.plan(parent=None, ctx=_fake_ctx()))          # this call also lands on cand1

    paths = {s["path"] for s in cand.solution["sources"]}
    assert paths == {"kernel.py"}                                 # only what THIS call wrote
    assert cand.strategy != "stale strategy from a prior process lifetime"


def test_design_reads_file(tmp_path):
    agent = _agent(_fake_spec(tmp_path), tmp_path)
    text, tokens = run(agent.design(7))                       # from design.md
    assert "roofline" in text
    assert tokens["in"] == 100 and tokens["out"] == 50         # cost/tokens must not be discarded


def _seed_knowledge(knowledge_dir, tag="fusion"):
    """Seed a cross-op pattern for `tag` on a DIFFERENT task/op than the one
    under test (docs/context-architecture-plan.md Part B), via the real
    curate()/tech_events path — not by hand-writing patterns.json."""
    import asyncio as _asyncio

    from solver.engine.frontier import Frontier, Member
    from solver.engine.knowledge import KnowledgeStore
    from solver.engine.reflection import ProblemReflection

    class _Ctx:
        task_id, tier_idx, terminated_reason = 999, 0, "budget:time"
        frontier = Frontier(0.02)

    ev = [{"tag": tag, "kind": "confirmed", "score": 0.8, "error": None,
          "cand": "deadbeef", "strategy": f"{tag} fused epilogue"}]
    refl = ProblemReflection(task_id=999, family="other_op", tech_events=ev)
    ks = KnowledgeStore(knowledge_dir)
    _asyncio.run(ks.curate(_Ctx(), "other_op", "other_op_999", refl=refl))


def test_design_writes_patterns_md_when_notes_exist(tmp_path):
    _seed_knowledge(tmp_path / "knowledge")
    agent = _agent(_fake_spec(tmp_path), tmp_path, knowledge_dir=tmp_path / "knowledge")
    run(agent.design(7))
    wd = tmp_path / "runs" / "7" / "work" / "design"
    assert (wd / "PATTERNS.md").is_file()
    assert "fusion" in (wd / "PATTERNS.md").read_text()
    assert "other_op" in (wd / "PATTERNS.md").read_text()


def test_plan_records_cross_op_patterns_shown_on_candidate(tmp_path):
    _seed_knowledge(tmp_path / "knowledge")
    agent = _agent(_fake_spec(tmp_path), tmp_path, knowledge_dir=tmp_path / "knowledge")
    cand = run(agent.plan(parent=None, ctx=_fake_ctx(task_id=7)))
    assert cand.cross_op_patterns_shown == ["fusion"]
    wd = tmp_path / "runs" / "7" / "work" / cand.cand_id
    assert "Cross-op technique notes" in (wd / "CONTEXT.md").read_text()


def test_cross_op_patterns_flag_disables_rendering(tmp_path):
    _seed_knowledge(tmp_path / "knowledge")
    agent = _agent(_fake_spec(tmp_path), tmp_path, knowledge_dir=tmp_path / "knowledge",
                   cross_op_patterns=False)
    run(agent.design(7))
    assert not (tmp_path / "runs" / "7" / "work" / "design" / "PATTERNS.md").exists()

    cand = run(agent.plan(parent=None, ctx=_fake_ctx(task_id=8)))
    assert not cand.cross_op_patterns_shown
    wd = tmp_path / "runs" / "8" / "work" / cand.cand_id
    assert "Cross-op technique notes" not in (wd / "CONTEXT.md").read_text()


def test_provider_routing_injects_anthropic_endpoint(monkeypatch, tmp_path):
    # A cheap provider (OpenRouter/GLM/DeepSeek/Kimi) runs the SAME claude CLI but
    # routed at its Anthropic-compatible endpoint via env; the real claude spec is
    # NOT routed (keeps subscription auth) — so a mixed tier ladder stays isolated.
    from solver.engine.cli_agent import OPENROUTER, CLAUDE
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    a = CliAgent(OPENROUTER, "z-ai/glm-5.2", runs_dir=tmp_path / "r", problems_dir=tmp_path / "p")
    assert a._provider_env() == {"ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
                                 "ANTHROPIC_AUTH_TOKEN": "sk-or-test"}
    assert a.perspective.agent == "openrouter" and a.model == "z-ai/glm-5.2"
    b = CliAgent(CLAUDE, "opus", runs_dir=tmp_path / "r", problems_dir=tmp_path / "p")
    assert b._provider_env() == {}                        # real Claude subscription, not routed


def test_provider_missing_key_fails_fast(monkeypatch, tmp_path):
    from solver.engine.cli_agent import DEEPSEEK
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        CliAgent(DEEPSEEK, "deepseek-chat", runs_dir=tmp_path / "r", problems_dir=tmp_path / "p")


def test_context_md_renders_reserve_plays():
    # the per-problem playbook (reserve plays) is fed to the next agent via CONTEXT.md
    from solver.engine.cli_agent import _context_md
    ctx = SimpleNamespace(design="", sibling_hint=None, recent_failures=[],
                          frontier=SimpleNamespace(members=[]),
                          playbook=[{"cand": "abc12345", "strategy": "atomic scatter",
                                     "handoff": "radix-sort + atomic-free segmented reduction"}])
    md, _shown = _context_md(parent=None, ctx=ctx)
    assert "Reserve plays" in md
    assert "radix-sort + atomic-free segmented reduction" in md
    assert "atomic scatter" in md                             # the strategy that flagged it


def test_context_md_flags_seed_only_frontier_for_risk_sequencing():
    """When nothing but the seed has been accepted yet, the writer must be told
    to bank a safe correctness win before reaching for the design's highest-
    ceiling approach (the #44 compile-hang incident: an agent went straight for
    the riskiest ranked approach on its very first real attempt and burned the
    single-flight GPU for 10 minutes on a COMPILE_ERROR)."""
    from solver.engine.cli_agent import _context_md
    seed_member = SimpleNamespace(strategy="seed", mean=0.1, sol_score_cal=None,
                                  cand_id="seed0000")
    ctx = SimpleNamespace(design="", sibling_hint=None, recent_failures=[],
                          frontier=SimpleNamespace(members=[seed_member]), playbook=[])
    md, _shown = _context_md(parent=None, ctx=ctx)
    assert "Nothing real has been accepted yet" in md
    assert "SAFEST" in md


def test_context_md_silent_once_a_real_candidate_is_accepted():
    from solver.engine.cli_agent import _context_md
    seed_member = SimpleNamespace(strategy="seed", mean=0.1, sol_score_cal=None,
                                  cand_id="seed0000")
    real_member = SimpleNamespace(strategy="fused Triton kernel", mean=0.6,
                                  sol_score_cal=0.55, cand_id="real0001")
    ctx = SimpleNamespace(design="", sibling_hint=None, recent_failures=[],
                          frontier=SimpleNamespace(members=[seed_member, real_member]), playbook=[])
    md, _shown = _context_md(parent=None, ctx=ctx)
    assert "Nothing real has been accepted yet" not in md


def test_run_writes_the_trajectory_incrementally_not_just_at_the_end(tmp_path):
    """The dashboard's live-transcript view needs real content to show WHILE a
    call is still running (see report.py's live-transcript lookup) — trajectory
    .jsonl must grow as output arrives, not appear all at once at process exit."""
    fake = tmp_path / "slow_streamer.py"
    fake.write_text(
        "import sys, json, time, pathlib\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 't1'}), flush=True)\n"
        "time.sleep(0.3)\n"
        "pathlib.Path('kernel.py').write_text('def run(*t): return t[-1]\\n')\n"
        "pathlib.Path('strategy.txt').write_text('s')\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {}}), flush=True)\n"
    )
    spec = CliSpec("slow", cmd=[sys.executable, str(fake), "{model}", "{prompt}"], stream="codex")
    agent = CliAgent(spec, "m", runs_dir=tmp_path / "runs", problems_dir=tmp_path / "none", timeout=5)

    async def _check():
        task = asyncio.create_task(agent.plan(parent=None, ctx=_fake_ctx()))
        wd = tmp_path / "runs" / "7" / "work" / "cand1"
        traj = wd / "trajectory.jsonl"
        for _ in range(100):
            if traj.exists() and traj.read_text().strip():
                break
            await asyncio.sleep(0.02)
        mid_content = traj.read_text() if traj.exists() else ""
        assert not task.done()                        # still running when we captured this
        assert "thread.started" in mid_content
        assert "turn.completed" not in mid_content    # second line hasn't landed yet
        await task

    run(_check())


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


def test_parse_review_treats_hand_trace_line_as_not_an_issue():
    """Line 2 is the required hand-trace summary, not a bullet issue — it must
    not leak into verdict.issues on either a ship or a revise."""
    from solver.engine.cli_agent import _parse_review

    ship = _parse_review(
        "VERDICT: SHIP\n"
        "traced M=17 (odd, smallest graded shape) through the mask — matches reference.\n",
        reviewer="r")
    assert ship.ship and ship.issues == []

    revise = _parse_review(
        "VERDICT: REVISE\n"
        "traced M=17 — accumulation dtype step couldn't be confirmed against reference.\n"
        "- kernel.py:42 accumulates in fp16, reference accumulates in fp32\n"
        "- workloads.md shape (17, 2048) triggers the tail mask, unverified\n",
        reviewer="r")
    assert not revise.ship
    assert len(revise.issues) == 2
    assert "fp16" in revise.issues[0]


def test_extract_context_read_claude_schema_from_tool_use_blocks():
    """Auditable evidence of what the model actually consulted, not an assumption
    — claude reports file reads as structured Read tool_use blocks."""
    from solver.engine.cli_agent import _extract_context_read
    import json as _json

    raw = "\n".join(_json.dumps(e) for e in [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read",
             "input": {"file_path": "/work/cand1/reference.py"}},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read",
             "input": {"file_path": "/work/cand1/kb/README.md"}},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read",
             "input": {"file_path": "/work/cand1/kb/b200-hardware.md"}},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read",       # a repeat — must not duplicate
             "input": {"file_path": "/work/cand1/kb/README.md"}},
        ]}},
    ])
    assert _extract_context_read("claude", raw) == ["README.md", "b200-hardware.md"]


def test_extract_context_read_codex_schema_from_shell_commands():
    """codex has no Read tool at all — it reads files via plain shell
    (nl/cat/rg/sed), so this must grep command TEXT, not tool_use blocks, or it
    silently undercounts codex to zero even when it read half the kb (confirmed
    live, 2026-07-08 — the naive claude-only check showed 20/20 codex calls with
    'zero kb reads' when they were reading kb/*.md via `nl -ba kb/...`)."""
    from solver.engine.cli_agent import _extract_context_read
    import json as _json

    raw = "\n".join(_json.dumps(e) for e in [
        {"type": "item.completed", "item": {"type": "command_execution",
         "command": "/bin/zsh -lc 'nl -ba kb/README.md'"}},
        {"type": "item.completed", "item": {"type": "command_execution",
         "command": "/bin/zsh -lc \"rg -n 'num_warps' kb/*.md\""}},
        {"type": "item.completed", "item": {"type": "command_execution",
         "command": "/bin/zsh -lc 'nl -ba kb/fusion-patterns.md'"}},
        {"type": "item.completed", "item": {"type": "command_execution",
         "command": "/bin/zsh -lc 'nl -ba reference.py'"}},        # not kb/ — excluded
    ])
    assert _extract_context_read("codex", raw) == ["README.md", "fusion-patterns.md"]


def test_extract_context_read_empty_when_nothing_consulted():
    from solver.engine.cli_agent import _extract_context_read
    assert _extract_context_read("claude", "") == []
    assert _extract_context_read("codex", "") == []


# --------------------------------------------------------------------------- #
# session resume — repair() continues the writer's OWN CLI session (real model
# memory) instead of cold-starting, so it can actually act on a critique instead
# of reinterpreting a stranger's code from a text file (see CliAgent.repair).
# --------------------------------------------------------------------------- #
def test_extract_session_id_claude_schema():
    from solver.engine.cli_agent import _extract_session_id
    import json as _json

    raw = "\n".join(_json.dumps(e) for e in [
        {"type": "system", "subtype": "init", "session_id": "abc-123"},
        {"type": "assistant", "session_id": "abc-123", "message": {"content": []}},
    ])
    assert _extract_session_id("claude", raw) == "abc-123"


def test_extract_session_id_codex_schema():
    from solver.engine.cli_agent import _extract_session_id
    import json as _json

    raw = "\n".join(_json.dumps(e) for e in [
        {"type": "thread.started", "thread_id": "thread-xyz"},
        {"type": "turn.started"},
    ])
    assert _extract_session_id("codex", raw) == "thread-xyz"


def test_extract_session_id_none_when_absent():
    from solver.engine.cli_agent import _extract_session_id
    assert _extract_session_id("claude", "") is None
    assert _extract_session_id("codex", '{"type": "turn.started"}') is None


def test_render_stream_handles_a_partial_trailing_line():
    """render_stream backs the dashboard's LIVE transcript view, which reads
    whatever has landed so far — including a cut-off, not-yet-complete final
    line. It must render the complete lines and not blow up on the partial one
    (the vendored jq helper wraps each line in try/catch for exactly this)."""
    import json as _json

    from solver.engine.cli_agent import render_stream

    complete = _json.dumps({"type": "thread.started", "thread_id": "t1"})
    raw = complete + "\n" + '{"type": "item.completed", "item": {"type": "agent_mess'  # cut off mid-line
    text = render_stream("codex", raw)
    assert text != ""                 # didn't blow up / return nothing useful
    assert "unparsed" in text or "thread" in text.lower()


def test_plan_captures_session_id_for_later_resume(tmp_path):
    agent = _agent(_fake_spec(tmp_path), tmp_path)
    cand = run(agent.plan(parent=None, ctx=_fake_ctx()))
    assert cand.session_id == "t1"        # the fake CLI's thread.started id, from _extract_session_id


# A fake CLI that behaves differently when invoked via a resume-style extra argv
# (the session id) — writes a REAL, distinguishable fix and records which session
# id it was actually resumed with, so tests can verify the writer's own session
# (not a fresh one) is what gets continued.
_FAKE_RESUMABLE = '''\
import sys, pathlib, json
model, prompt = sys.argv[1], sys.argv[2]
session_id = sys.argv[3] if len(sys.argv) > 3 else ""
def ev(o): print(json.dumps(o))
ev({"type": "thread.started", "thread_id": "sess-fixed-1"})
if session_id:
    pathlib.Path("seen_resume_session_id.txt").write_text(session_id)
    pathlib.Path("kernel.py").write_text("def run(*t): return t[-1]  # v2 fixed\\n")
    pathlib.Path("strategy.txt").write_text("fused elementwise path (bounds-checked)")
elif "kernel.py" in prompt:
    pathlib.Path("kernel.py").write_text("def run(*t): return t[-1]  # v1\\n")
    pathlib.Path("strategy.txt").write_text("fused elementwise path")
    pathlib.Path("handoff.md").write_text("reserve play: radix-sort segmented reduction")
ev({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5,
     "reasoning_output_tokens": 0, "cached_input_tokens": 0}})
'''


def _fake_resumable_spec(tmp_path):
    fake = tmp_path / "fake_resumable_cli.py"
    fake.write_text(_FAKE_RESUMABLE)
    return CliSpec("fake", cmd=[sys.executable, str(fake), "{model}", "{prompt}"],
                   resume_cmd=[sys.executable, str(fake), "{model}", "{prompt}", "{session_id}"],
                   stream="codex")


def test_repair_resumes_the_writers_own_session_and_fixes_the_bug(tmp_path):
    agent = _agent(_fake_resumable_spec(tmp_path), tmp_path)
    ctx = _fake_ctx()
    cand = run(agent.plan(parent=None, ctx=ctx))
    assert cand.session_id == "sess-fixed-1"
    wd = tmp_path / "runs" / "7" / "work" / cand.cand_id

    repaired = run(agent.repair(cand, "fix the off-by-one", ctx))
    assert repaired.cand_id != cand.cand_id            # a real change -> new content hash
    assert "v2 fixed" in repaired.solution["sources"][0]["content"]
    assert repaired.parent == cand.cand_id
    # resumed IN PLACE — same directory, never renamed/recreated (needed so a
    # LATER repair round in the same lineage can resume the same session again)
    assert repaired.trajectory == str(wd / "trajectory.repair-1.jsonl")
    assert (wd / "trajectory.jsonl").exists()          # original round's trajectory untouched
    # the fake CLI actually received the ORIGINAL session id on the resume call
    assert (wd / "seen_resume_session_id.txt").read_text() == cand.session_id


def test_repair_without_a_resume_cmd_falls_back_to_cold_start(tmp_path):
    """A spec with no resume_cmd (or a candidate with no captured session id, e.g.
    replayed from a journal that predates this feature) must still get repaired —
    via the old cold-start plan() call — never hard-fail."""
    agent = _agent(_fake_spec(tmp_path), tmp_path)   # plain fake spec: no resume_cmd
    ctx = _fake_ctx()
    cand = run(agent.plan(parent=None, ctx=ctx))
    assert cand.session_id == "t1"                   # captured, but this spec can't resume
    repaired = run(agent.repair(cand, "fix it", ctx))
    assert repaired.solution["sources"][0]["path"] == "kernel.py"


def test_repair_falls_back_to_cold_start_for_a_candidate_with_no_session_id(tmp_path):
    agent = _agent(_fake_resumable_spec(tmp_path), tmp_path)
    ctx = _fake_ctx()
    cand = Candidate(cand_id="deadbeef0000", solution={"sources": []}, parent=None,
                     agent="fake", model="gpt-5.5", session_id=None)
    repaired = run(agent.repair(cand, "fix it", ctx))
    assert "v1" in repaired.solution["sources"][0]["content"]   # cold-start plan(), not resume


def test_repair_falls_back_to_cold_start_when_workdir_is_missing(tmp_path):
    import shutil

    agent = _agent(_fake_resumable_spec(tmp_path), tmp_path)
    ctx = _fake_ctx()
    cand = run(agent.plan(parent=None, ctx=ctx))
    shutil.rmtree(tmp_path / "runs" / "7" / "work" / cand.cand_id)
    repaired = run(agent.repair(cand, "fix it", ctx))
    assert "v1" in repaired.solution["sources"][0]["content"]   # fell back, didn't crash


def test_collect_default_keeps_mixed_language_files(tmp_path):
    agent = _agent(_fake_spec(tmp_path), tmp_path)
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "kernel.py").write_text("old")
    (wd / "kernel.cu").write_text("new")
    sol = agent._collect(wd)
    assert {s["path"] for s in sol["sources"]} == {"kernel.py", "kernel.cu"}


def test_collect_resolve_conflict_keeps_only_the_freshest_language(tmp_path):
    """repair() reuses an existing directory across rounds (needed for session
    resume) instead of starting clean — if the model switches language without
    deleting the old file, keep only the freshest one rather than shipping a
    mixed-language 'solution' (mirrors the _workdir() staleness fix)."""
    import os

    agent = _agent(_fake_spec(tmp_path), tmp_path)
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "kernel.py").write_text("stale")
    (wd / "kernel.cu").write_text("fresh")
    os.utime(wd / "kernel.py", (1000, 1000))
    os.utime(wd / "kernel.cu", (2000, 2000))
    sol = agent._collect(wd, resolve_conflict=True)
    assert [s["path"] for s in sol["sources"]] == ["kernel.cu"]
    assert sol["sources"][0]["content"] == "fresh"
