"""CliAgent — drive the **codex** and **claude** CLIs behind the Agent interface.

We don't implement an agent; we shell out to one (docs/agent.md). Scoped to
codex and claude only; other agent types would be integrated separately.

Two disciplines make this robust:
- **Results come from known files, never parsed stdout.** The agent is told to
  write its output to fixed filenames in the workdir — `kernel.<ext>` +
  `strategy.txt` + `handoff.md` (plan), `design.md` — and we read those files.
  The model's prose is never parsed for the answer.
- **The event stream is the trajectory.** We run the CLI in streaming JSON mode,
  persist the raw stream as `trajectory.jsonl` (+ a readable `trajectory.txt`),
  and parse it only for token usage. Each plan's workdir is keyed by its
  `cand_id`, so the kernel, its trajectory, and its inputs persist together.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .agent import Candidate, ReviewVerdict, solution_hash
from .config import Config, Perspective

_HELPERS = Path(__file__).resolve().parent.parent / "agent_helpers"
_EXT_LANG = {"py": "pytorch", "cu": "cuda_cpp", "cuh": "cuda_cpp",
             "cpp": "cuda_cpp", "cc": "cuda_cpp"}

# Known output files — the agent writes these; we read them (no stdout parsing).
KERNELS = "kernel.*"
F_STRATEGY = "strategy.txt"
F_DESIGN = "design.md"
F_HANDOFF = "handoff.md"
F_REVIEW = "review.md"


@dataclass
class CliSpec:
    """A supported agent CLI. One streaming, write-capable command template; its
    JSON event schema (`stream`) is parsed for tokens.

    `base_url` + `api_key_env` route the *same* CLI at a different provider: the
    `claude` CLI speaks the Anthropic Messages protocol, and GLM / DeepSeek / Kimi
    / OpenRouter all expose an Anthropic-compatible endpoint, so a cheap model is
    just `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` (per-spec, so a mixed tier
    ladder — cheap on OpenRouter, strong on the real Claude subscription — keeps
    each perspective's auth isolated)."""

    name: str
    cmd: list[str]                  # {model}/{prompt} substituted; runs with write access + JSON stream
    stream: str                     # event-stream schema: "codex" | "claude"
    kernels: str = KERNELS
    lang: str | None = None
    base_url: str | None = None     # Anthropic-compatible endpoint (routes the claude CLI elsewhere)
    api_key_env: str | None = None  # env var holding the provider key → ANTHROPIC_AUTH_TOKEN
    prompt_via_stdin: bool = False  # feed the prompt on stdin instead of as an argv (codex)


CODEX = CliSpec(
    "codex",   # `codex exec [PROMPT]`: prompt as an argv makes codex ALSO read stdin and
    # intermittently die with "Reading additional input from stdin" (exit 1) before the
    # model runs. Feed the prompt on stdin with `-` — the canonical programmatic form.
    cmd=["codex", "exec", "--json", "-m", "{model}", "-s", "workspace-write",
         "--skip-git-repo-check", "-"],
    stream="codex", prompt_via_stdin=True,
)
# The Claude Code CLI in print mode + stream-json events. The same command, pointed
# at an Anthropic-compatible endpoint via base_url, drives cheap third-party models.
_CLAUDE_CMD = ["claude", "-p", "{prompt}", "--model", "{model}",
               "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"]
CLAUDE = CliSpec("claude", cmd=_CLAUDE_CMD, stream="claude")   # your real Claude subscription auth

# Cheap providers over the SAME claude CLI (Anthropic-compatible endpoints). The
# model string comes from `--tier` (e.g. openrouter/z-ai/glm-5.2, deepseek/deepseek-chat).
OPENROUTER = CliSpec("openrouter", cmd=_CLAUDE_CMD, stream="claude",
                     base_url="https://openrouter.ai/api", api_key_env="OPENROUTER_API_KEY")
GLM = CliSpec("glm", cmd=_CLAUDE_CMD, stream="claude",
              base_url="https://api.z.ai/api/anthropic", api_key_env="ZAI_API_KEY")
DEEPSEEK = CliSpec("deepseek", cmd=_CLAUDE_CMD, stream="claude",
                   base_url="https://api.deepseek.com/anthropic", api_key_env="DEEPSEEK_API_KEY")
KIMI = CliSpec("kimi", cmd=_CLAUDE_CMD, stream="claude",
               base_url="https://api.moonshot.ai/anthropic", api_key_env="MOONSHOT_API_KEY")
SPECS: dict[str, CliSpec] = {"codex": CODEX, "claude": CLAUDE, "openrouter": OPENROUTER,
                             "glm": GLM, "deepseek": DEEPSEEK, "kimi": KIMI}


_PLAN = (
    "You are optimizing a GPU kernel for NVIDIA B200 (SOL-ExecBench).\n"
    "reference.py is the PyTorch reference you must stay numerically equivalent to;\n"
    "definition.json is the spec; workloads.md lists the EXACT shapes you are graded\n"
    "on — TUNE FOR THEM (tile sizes, grid/launch config, CUDA-graph-vs-not, split-K,\n"
    "per-shape code paths); classify each shape by roofline and specialize. You are\n"
    "not given the input tensor VALUES, so write a correct, shape-specialized kernel.\n"
    "DESIGN.md and CONTEXT.md hold the plan and prior\n"
    "attempts. The kb/ directory is a B200 optimization knowledge base (start at\n"
    "kb/README.md; e.g. optimization-recipe.md, profiling-guide.md, b200-hardware.md,\n"
    "fusion-patterns.md) — consult the files relevant to this op before you write.\n"
    "Write your optimized implementation to a file named kernel.py or\n"
    "kernel.cu (one language), and a one-line summary of the approach to strategy.txt.\n"
    "CONTRACT: the kernel file MUST define a top-level function named `run` with the\n"
    "SAME parameter names and order as reference.py's `run`, and it MUST RETURN the\n"
    "output tensor(s) (do not write in place). It is the entry point the grader calls\n"
    "(kernel.py::run); a mismatch scores zero. Any imports (triton, torch) go in that\n"
    "file. Follow the ladder torch -> Triton -> CuTe/CUTLASS -> C++/PTX; escalate only\n"
    "when needed.\n"
    "CORRECTNESS — over a third of candidates fail here, so check these BEFORE you\n"
    "consider the kernel done: (1) never mix a .py and a .cu/.cpp source file in one\n"
    "solution — pick ONE language; (2) grid/block launch dims and bounds checks must\n"
    "cover every element for EVERY graded shape in workloads.md, not just the common\n"
    "case — an out-of-bounds index is the #1 failure mode; (3) match the reference's\n"
    "reduction order, masking, and accumulation dtype exactly (fp32 accumulate unless\n"
    "the reference doesn't) — silent numerical drift fails the tolerance gate; (4) the\n"
    "output shape/dtype/device must exactly match what reference.py returns; (5) never\n"
    "use threading, monkey-patch torch.cuda.Event.elapsed_time, or wrap the entry\n"
    "function body in try/except — these are rejected as reward-hacking, not scored.\n"
    "Re-read your kernel against this list before finishing.\n"
    "NEVER leave the kernel file byte-identical to the parent's — a silent no-change\n"
    "is indistinguishable from a crash and wastes this whole turn for no signal. If you\n"
    "genuinely believe no further improvement is possible, you MUST still: (1) try ONE\n"
    "untried technique from CONTEXT.md's ladder even if you doubt it'll help — a real\n"
    "attempt beats no attempt — and (2) say explicitly in handoff.md why you think this\n"
    "is at ceiling. Only true, hard-won ceilings look like this; don't reach for it early.\n"
    "Also write handoff.md: 1-2 sentences naming the HIGHER-CEILING idea you did NOT\n"
    "ship this round and the trigger to try it (e.g. 'if this only ties the baseline,\n"
    "switch to a radix-sort + atomic-free segmented reduction that writes output once').\n"
    "This is fed verbatim to the next agent as a reserve play — make it a concrete,\n"
    "actionable next kernel, not a platitude. Leave it empty only if you truly shipped\n"
    "the ceiling.\n"
    "HOW THIS WORKS — your ONLY job is to write the kernel file, strategy.txt, and\n"
    "handoff.md. Do NOT try to run, benchmark, or numerically test the kernel yourself:\n"
    "there is NO GPU, torch, or triton in your environment, so any local execution will\n"
    "fail — that is EXPECTED and not your concern. After you finish, the SYSTEM ships\n"
    "your kernel to a real B200, runs it in the official harness (correctness + latency),\n"
    "and scores it; the next round's agent sees that score, the frontier, and your\n"
    "handoff. So spend all your effort making the kernel correct and fast — not on\n"
    "verifying it locally. Do not print the code — only write the files."
)
_DESIGN = (
    "Analyze reference.py and definition.json for NVIDIA B200. workloads.md lists the\n"
    "EXACT shapes this kernel is graded on — compute the roofline (arithmetic intensity,\n"
    "memory- vs compute- vs launch-bound) for THOSE specific shapes, and note where they\n"
    "split into regimes needing different strategies. Consult the kb/ knowledge base\n"
    "(kb/README.md index; optimization-recipe.md, profiling-guide.md, b200-hardware.md)\n"
    "for the hardware limits and the optimization ladder. Write a short markdown design\n"
    "(op graph, per-shape roofline, 3 ranked approaches) to a file named design.md."
)
_REVIEW = (
    "You are a SECOND, INDEPENDENT reviewer for a GPU kernel about to be sent to a real\n"
    "B200 for grading — a rented, single-flight GPU, so a bad kernel wastes real GPU\n"
    "time for the whole fleet (a hang can burn 10 minutes). You did NOT write this\n"
    "kernel; read it with fresh, skeptical eyes.\n"
    "reference.py is the ground truth it must match; workloads.md lists the EXACT\n"
    "graded shapes/tolerances; definition.json is the spec. CONTEXT.md has this\n"
    "problem's history (what's already been tried and failed) if useful. The kb/\n"
    "directory is the same B200 knowledge base the writer had (kb/README.md index) —\n"
    "check it for KNOWN correctness pitfalls specific to this op's technique (e.g. a\n"
    "documented Triton masking gotcha, a CUDA-graph static-buffer trap, or the grader\n"
    "comparing against a bf16-ROUNDED reference rather than fp32 — a plain fp32 port can\n"
    "fail tolerance purely from rounding-order mismatch) before you flag something as a\n"
    "guess.\n"
    "REQUIRED STEP — do not skip: pick the single trickiest graded shape in workloads.md\n"
    "(smallest, most irregular, or highest-risk for the launch config used) and manually\n"
    "trace it by hand — write out the actual grid/block dims, loop bounds, and mask\n"
    "conditions the kernel would compute for THAT shape's real numbers. Most real bugs\n"
    "only show up on one specific shape, not in the general logic, and 'the logic looks\n"
    "right' without doing this trace is not a verified SHIP.\n"
    "Read the kernel file(s) line by line against reference.py and workloads.md. Look\n"
    "specifically for: (1) mixed C++/Python source files (instant reject); (2) the\n"
    "entry function's name/params not matching reference.py's `run` exactly; (3)\n"
    "grid/block launch config or bounds checks that don't cover every graded shape in\n"
    "workloads.md — this is the #1 real failure mode, confirmed by your hand trace above;\n"
    "(4) reduction order / masking / accumulation dtype that would silently drift from\n"
    "the reference's numerics (fp16/bf16 accumulation without an fp32 accumulator,\n"
    "non-deterministic atomic-add ordering across thread blocks, a rounding step done in\n"
    "a different order/dtype than the reference); (5) output shape/dtype mismatches; (6)\n"
    "threading, timer monkey-patching, or try/except around the entry body (reward-hack\n"
    "rejects).\n"
    "You have NO GPU either — this is a careful READ, not an execution test. Because of\n"
    "that, treat 'I traced it and it holds' and 'I couldn't fully verify this' as\n"
    "DIFFERENT outcomes: if the hand trace surfaces ANY step you can't confidently follow\n"
    "through to the reference's exact numeric result — not just a definite bug — that is\n"
    "grounds for REVISE with a specific question, not a ship on faith. Reserve SHIP for\n"
    "when you actually completed the trace and it checked out; do not revise on pure\n"
    "style with no numeric stake.\n"
    "Write your verdict to review.md in EXACTLY this format: the first line is either\n"
    "'VERDICT: SHIP' or 'VERDICT: REVISE'; the second line is your one-sentence hand-trace\n"
    "result (the shape you chose and what it confirmed or couldn't confirm); if REVISE,\n"
    "followed by a markdown bullet list of the SPECIFIC issues found (name the exact\n"
    "line/shape/variable — 'looks risky' is not actionable). Write nothing else; only the\n"
    "review.md file."
)


@dataclass
class _Run:
    stderr: str
    rc: int
    tokens: dict           # {in, out, reasoning, cached, cost_usd} where the stream provides them
    trajectory: Path       # persisted raw event stream (trajectory.jsonl)


def _parse_tokens(schema: str, raw: str) -> dict:
    """Token usage from the codex/claude event stream (results come from files)."""
    tokens: dict = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if schema == "codex" and e.get("type") == "turn.completed":
            u = e.get("usage") or {}
            tokens = {"in": u.get("input_tokens"), "out": u.get("output_tokens"),
                      "reasoning": u.get("reasoning_output_tokens"),
                      "cached": u.get("cached_input_tokens")}
        elif schema == "claude" and e.get("type") == "result":
            u = e.get("usage") or {}
            cache_read = u.get("cache_read_input_tokens") or 0
            cache_creation = u.get("cache_creation_input_tokens") or 0
            tokens = {"in": u.get("input_tokens"), "out": u.get("output_tokens"),
                      "cached": (cache_read + cache_creation) or None,
                      "cost_usd": e.get("total_cost_usd")}
    return {k: v for k, v in tokens.items() if v is not None}


class CliAgent:
    """An Agent that drives a coding CLI (codex, or claude — optionally pointed at a
    cheap provider's Anthropic-compatible endpoint) via subprocess."""

    def __init__(self, spec: CliSpec, model: str, *, runs_dir: str | Path = "runs",
                 problems_dir: str | Path = "problems", timeout: float = 1800.0,
                 kb_dir: str | Path = "kb", env: dict | None = None) -> None:
        self.spec = spec
        self.model = model
        self.perspective = Perspective(spec.name, model)
        self.runs_dir = Path(runs_dir)
        self.problems_dir = Path(problems_dir)
        self.kb_dir = Path(kb_dir)
        self.timeout = timeout
        self.env = {**os.environ, **(env or {}), **self._provider_env()}
        self._seq = 0

    def _provider_env(self) -> dict:
        """Route the claude CLI at a third-party Anthropic-compatible endpoint when
        the spec names one (GLM / DeepSeek / Kimi / OpenRouter). Fail fast with a
        clear message if the provider key isn't in the environment (.env)."""
        if not self.spec.base_url:
            return {}
        key = os.environ.get(self.spec.api_key_env or "")
        if not key:
            raise SystemExit(f"{self.spec.name} needs {self.spec.api_key_env} set "
                             f"(add it to .env) to reach {self.spec.base_url}")
        return {"ANTHROPIC_BASE_URL": self.spec.base_url, "ANTHROPIC_AUTH_TOKEN": key}

    async def design(self, task_id: int) -> tuple[str, dict]:
        wd = self._workdir(task_id, "design")
        self._write_problem(wd, task_id)
        self._write_kb(wd)
        res = await self._run(wd, _DESIGN)
        f = wd / F_DESIGN
        text = f.read_text().strip() if f.exists() else "(no design produced)"
        return text, (res.tokens or {})

    async def plan(self, parent, ctx) -> Candidate:
        self._seq += 1
        wd = self._workdir(ctx.task_id, f"cand{self._seq}")
        self._seed_workdir(wd, parent, ctx)
        res = await self._run(wd, _PLAN)
        solution = self._collect(wd)
        if not solution["sources"]:                    # loud, not a silent baseline candidate
            raise RuntimeError(f"{self.spec.name}/{self.model} wrote no kernel "
                               f"(exit {res.rc}): {res.stderr[:400]}")
        strat = wd / F_STRATEGY
        strategy = (strat.read_text().strip()[:120] if strat.exists() else "") or "cli agent"
        hf = wd / F_HANDOFF
        handoff = (hf.read_text().strip()[:600] if hf.exists() else "") or None
        cand_id = solution_hash(solution)[:12]
        # kernel + trajectory + inputs, keyed by cand — traj_path is THIS call's own
        # trajectory even on a hash collision (a no-op/duplicate), so every attempt
        # stays debuggable instead of being silently deleted (see _rekey_workdir).
        wd, traj_path = self._rekey_workdir(wd, ctx.task_id, cand_id)
        return Candidate(cand_id=cand_id, solution=solution,
                         parent=getattr(parent, "cand_id", None),
                         agent=self.spec.name, model=self.model, strategy=strategy,
                         handoff=handoff, tokens=res.tokens or None,
                         trajectory=str(traj_path))

    async def review(self, cand: Candidate, ctx) -> ReviewVerdict:
        """Pre-GPU code review: an INDEPENDENT (agent,model) — never the one that
        wrote `cand` — reads the kernel against reference.py + workloads.md and
        judges ship/revise before a single GPU eval is spent."""
        wd = self._workdir(getattr(ctx, "task_id", ""), f"review-{cand.cand_id}-{self._seq}")
        self._seq += 1
        (wd / "CONTEXT.md").write_text(_context_md(cand, ctx))
        self._write_problem(wd, getattr(ctx, "task_id", ""))
        self._write_kb(wd)                     # same knowledge base the writer consulted
        for s in (cand.solution or {}).get("sources", []):
            (wd / s["path"]).write_text(s.get("content", ""))
        res = await self._run(wd, _REVIEW)
        f = wd / F_REVIEW
        verdict = _parse_review(f.read_text() if f.exists() else "", reviewer=f"{self.spec.name}:{self.model}")
        verdict.cost_usd = (res.tokens or {}).get("cost_usd") or 0.0
        verdict.tokens = res.tokens or {}
        return verdict

    # ---- helpers ----
    def _workdir(self, key, sub: str) -> Path:
        wd = self.runs_dir / str(key) / "work" / sub
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    def _rekey_workdir(self, wd: Path, task_id, cand_id: str) -> tuple[Path, Path]:
        """Rename the plan workdir to be keyed by cand_id, so the kernel and its
        trajectory persist together under runs/<task>/work/<cand_id>/. Returns
        (canonical_dir, this_call's_trajectory_path).

        On a hash collision (this content already has a workdir — ALWAYS true for
        a no-op, since the parent's own directory already exists by definition)
        the fresh run used to be discarded via rmtree, silently destroying the one
        piece of evidence that could explain why the model produced nothing new —
        and misattributing the surviving candidate's trajectory link to whichever
        earlier call happened to create `dest`. Now the fresh trajectory is kept
        as a numbered sibling file instead of being deleted."""
        dest = self.runs_dir / str(task_id) / "work" / cand_id
        if dest.resolve() == wd.resolve():
            return wd, wd / "trajectory.jsonl"
        if dest.exists():                              # duplicate hash: keep the trajectory, drop the rest
            n = sum(1 for _ in dest.glob("trajectory.dup-*.jsonl")) + 1
            dup_path = dest / f"trajectory.dup-{n}.jsonl"
            src_traj = wd / "trajectory.jsonl"
            copied = src_traj.is_file()
            if copied:
                shutil.copyfile(src_traj, dup_path)
            shutil.rmtree(wd, ignore_errors=True)
            return dest, (dup_path if copied else dest / "trajectory.jsonl")
        wd.rename(dest)
        return dest, dest / "trajectory.jsonl"

    def _seed_workdir(self, wd: Path, parent, ctx) -> None:
        (wd / "DESIGN.md").write_text(getattr(ctx, "design", "") or "")
        (wd / "CONTEXT.md").write_text(_context_md(parent, ctx))
        self._write_problem(wd, getattr(ctx, "task_id", ""))
        self._write_kb(wd)
        psol = getattr(parent, "solution", None)
        for s in (psol or {}).get("sources", []):     # starting point = the parent's kernel
            (wd / s["path"]).write_text(s.get("content", ""))
        for s in (getattr(ctx, "sibling_hint", None) or {}).get("sources", []):   # cross-op warm start
            (wd / f"sibling_{s['path']}").write_text(s.get("content", ""))

    def _write_problem(self, wd: Path, task_id) -> None:
        pdir = self.problems_dir / str(task_id)
        for fname in ("reference.py", "definition.json"):
            src = pdir / fname
            if src.exists():
                (wd / fname).write_text(src.read_text())
        self._write_workloads(wd, pdir)

    def _write_workloads(self, wd: Path, pdir: Path) -> None:
        """Write the EXACT graded shapes (each workload's `axes` + tolerance) so the
        agent can tune tile/launch/algorithm per shape — the biggest lever for a
        fixed-shape benchmark. Deliberately EXCLUDES the input tensors: shapes enable
        optimization, values would open the door to hardcoding outputs (kept shut; the
        correctness gate would catch it anyway, but we don't even hand over the inputs)."""
        wl = pdir / "workload.jsonl"
        if not wl.exists():
            return
        rows = []
        for line in wl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append((w.get("axes") or {}, w.get("tolerance")))
        if not rows:
            return
        out = [
            "# Workloads — the EXACT shapes you are graded on",
            "",
            f"This kernel is scored ONLY on these {len(rows)} shapes. Optimize for them:",
            "choose tile sizes, grid/launch config, CUDA-graph-vs-not, split-K, and",
            "per-shape code paths for these dims; classify each by roofline (small =",
            "launch/latency-bound, large = memory/compute-bound) and specialize. You do",
            "NOT get the input tensor values — write a correct, shape-specialized kernel",
            "(correctness is re-checked against the reference on fresh inputs).",
            "",
            "| # | shape (axes) | tolerance |",
            "|---|---|---|",
        ]
        for i, (axes, tol) in enumerate(rows):
            ax = ", ".join(f"{k}={v}" for k, v in axes.items()) or "(scalar)"
            if isinstance(tol, dict):
                ts = ", ".join(f"{k}={v}" for k, v in tol.items()) or "default"
            else:
                ts = str(tol) if tol not in (None, "") else "default"
            out.append(f"| {i} | {ax} | {ts} |")
        (wd / "workloads.md").write_text("\n".join(out) + "\n")

    def _write_kb(self, wd: Path) -> None:
        """Drop the B200 optimization knowledge base into the workdir so the
        agentic CLI can read the recipes/profiling/hardware notes it needs."""
        if self.kb_dir.is_dir() and not (wd / "kb").exists():
            shutil.copytree(self.kb_dir, wd / "kb",
                            ignore=shutil.ignore_patterns("__pycache__", ".*"))

    def _collect(self, wd: Path) -> dict:
        files = [f for f in sorted(wd.glob(self.spec.kernels)) if f.is_file()]
        sources = [{"path": f.name, "content": f.read_text(errors="replace")} for f in files]
        langs = sorted({self.spec.lang or _EXT_LANG.get(f.suffix.lstrip("."), "cuda_cpp")
                        for f in files}) or ["cuda_cpp"]
        return {"spec": {"languages": langs}, "sources": sources}

    async def _run(self, wd: Path, prompt: str) -> _Run:
        cmd = [t.replace("{model}", self.model).replace("{prompt}", prompt) for t in self.spec.cmd]
        # Feed the prompt on stdin (codex) or as an argv with stdin closed (claude).
        # codex with an argv prompt + DEVNULL stdin flakily dies "Reading additional
        # input from stdin"; giving it the prompt ON stdin (cmd ends with `-`) fixes it.
        stdin_data = prompt.encode() if self.spec.prompt_via_stdin else None
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(wd), env=self.env,
            stdin=(asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(input=stdin_data), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"{self.spec.name} timed out after {self.timeout}s")
        raw = out.decode("utf-8", "replace")
        traj = wd / "trajectory.jsonl"
        traj.write_text(raw)                        # the trajectory: the raw agent event stream
        await asyncio.to_thread(self._render, raw, wd / "trajectory.txt")   # readable render (best-effort)
        return _Run(err.decode("utf-8", "replace"), proc.returncode or 0,
                    _parse_tokens(self.spec.stream, raw), traj)

    def _render(self, raw: str, out_path: Path) -> None:
        """Render the raw stream to readable text via the vendored jq wrapper."""
        helper = _HELPERS / f"{self.spec.stream}_stream.sh"
        if not raw.strip() or not helper.exists() or not shutil.which("jq") or not shutil.which("bash"):
            return
        try:
            r = subprocess.run(["bash", str(helper)], input=raw, capture_output=True,
                               text=True, timeout=30)
            if r.stdout:
                out_path.write_text(r.stdout)
        except Exception:
            pass


def _member_cal(m):
    return getattr(m, "sol_score_cal", None)


def _score_note(m) -> str:
    """`leaderboard-est (raw N)` — est is the number that ranks you on the board
    (0.5 = optimized-PyTorch baseline, 1.0 = speed-of-light); raw is our local measure."""
    cal = _member_cal(m)
    return f"{cal:.3f} (raw {m.mean:.3f})" if cal is not None else f"{m.mean:.3f} (raw)"


def _context_md(parent, ctx) -> str:
    lines = ["# Context", "",
             "Scores are the **leaderboard estimate** (0.5 = optimized-PyTorch baseline,",
             "1.0 = speed-of-light). Beat 0.5 to score on the board; the raw local measure",
             "is in parentheses.", ""]
    if parent is not None and getattr(parent, "solution", None):
        lines.append("The parent kernel to improve on is in this directory's kernel file(s).")
    critique = getattr(ctx, "review_critique", None)
    if critique:
        lines += ["", "## Pre-submission code review — FIX these before this ships to the GPU",
                  "An independent reviewer found concrete issues in the kernel you're improving on.",
                  "Fix EVERY issue below; do not ignore one because you're focused on speed:",
                  critique.strip()]
    # Coach card first — the cross-run reflection is the highest-signal directive
    # (you are stuck / here's what's been tried / where the loss actually is).
    card = getattr(ctx, "reflection", None)
    if card is None:
        rd, tid = getattr(ctx, "runs_dir", None), getattr(ctx, "task_id", None)
        if rd is not None and tid is not None:
            cf = Path(rd) / str(tid) / "reflection.md"
            card = cf.read_text() if cf.is_file() else ""
    if card:
        lines += ["", card.strip()]
    prior = getattr(ctx, "runs_dir", None)
    if prior is not None and getattr(ctx, "task_id", None) is not None and \
            (Path(prior) / str(ctx.task_id) / "prior").is_dir():
        lines += ["", "The `prior/` directory holds the top earlier kernels (named by score) "
                  "if you want to inspect exactly what a past approach did — read them on demand."]
    hint = getattr(ctx, "sibling_hint", None)
    if hint and hint.get("sources"):
        est = hint.get("score")
        lines += ["", "## Warm start — a SIBLING problem (same op) is already solved",
                  f"Sibling '{hint.get('sibling')}' (op `{hint.get('op')}`) scored ~"
                  f"{est:.3f} raw with: {hint.get('strategy')}." if isinstance(est, (int, float))
                  else f"Sibling '{hint.get('sibling')}' (op `{hint.get('op')}`): {hint.get('strategy')}.",
                  "Its kernel is in `sibling_kernel.py` — this is your best STARTING POINT.",
                  "ADAPT it to THIS problem's shapes/constants (it likely hardcodes the sibling's",
                  "shape, e.g. a fixed hidden size); do not just copy it verbatim."]
    fr = getattr(ctx, "frontier", None)
    members = list(getattr(fr, "members", None) or []) if fr else []
    if members:
        members.sort(key=lambda m: (_member_cal(m) if _member_cal(m) is not None else m.mean),
                     reverse=True)                                     # best first
        lines += ["", "## Frontier — best first (leaderboard-est (raw) · strategy)"]
        lines += [f"- {m.cand_id[:8]}  {_score_note(m)}  {m.strategy}" for m in members[:8]]
    plays = getattr(ctx, "playbook", None) or []
    if plays:
        lines += ["", "## Reserve plays — higher-ceiling ideas flagged but NOT yet shipped",
                  "Banked by prior accepted kernels. Cross-check the frontier above: if one is",
                  "still unexplored, executing it is likely the biggest win — otherwise beat it."]
        for e in plays[-6:][::-1]:                                   # most recent first
            strat = (e.get("strategy") or "").strip()
            tag = f"  (flagged after: {strat[:60]})" if strat else ""
            lines.append(f"- {e['handoff']}{tag}")
    fails = getattr(ctx, "recent_failures", None)
    if fails:
        lines += ["", "## Recent FAILED attempts — INCORRECT. Fix the exact workloads that failed",
                  "(each line: the failure + WHICH workloads broke; their shapes are in `workloads.md`):"]
        for f in fails[-4:]:
            detail = (f.get("detail") or "").strip()
            lines.append(f"- [{f['reason']}] {f['strategy']}")
            if detail:
                lines.append(f"    → {detail}")
    return "\n".join(lines) + "\n"


def _parse_review(text: str, *, reviewer: str) -> ReviewVerdict:
    """Parse review.md's fixed format. Fails OPEN (verdict=ship) on anything
    malformed/missing — a reviewer that doesn't follow the format must never
    permanently block a candidate from ever reaching the GPU. Line 2 is always
    the required hand-trace summary (not an issue bullet), whether the verdict
    is ship or revise; only line 3+ are the SPECIFIC-issue bullets."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].upper().startswith("VERDICT:"):
        return ReviewVerdict(verdict="ship", reviewer=reviewer)
    verdict = "revise" if "REVISE" in lines[0].upper() else "ship"
    issues = [ln.lstrip("-* ").strip() for ln in lines[2:] if ln.lstrip("-* ").strip()]
    return ReviewVerdict(verdict=verdict, issues=issues, reviewer=reviewer)


def make_agents(cfg: Config, specs: dict[str, CliSpec] = SPECS, **kwargs) -> dict:
    """Map every perspective in `cfg` to a CliAgent for its agent's CLI spec."""
    out = {}
    for p in cfg.perspectives:
        spec = specs.get(p.agent)
        if spec is None:
            raise KeyError(f"no CLI spec for agent {p.agent!r} (supported: {sorted(specs)})")
        out[p] = CliAgent(spec, p.model, **kwargs)
    return out
