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
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .agent import Candidate, ReviewVerdict, solution_hash
from .config import Config, Perspective
from .knowledge import KnowledgeStore, op_key_of
from .reflection import relevant_techniques

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
    resume_cmd: list[str] | None = None   # {session_id}/{model}/{prompt} template that CONTINUES
    # an existing session in place, instead of a cold start — used for review-repair turns so the
    # writer keeps its own reasoning about what it wrote and why (see CliAgent.repair). MUST run
    # in the exact cwd the session started in — both CLIs key session storage off it (verified
    # empirically: resuming from a different directory fails with "No conversation found").


CODEX = CliSpec(
    "codex",   # `codex exec [PROMPT]`: prompt as an argv makes codex ALSO read stdin and
    # intermittently die with "Reading additional input from stdin" (exit 1) before the
    # model runs. Feed the prompt on stdin with `-` — the canonical programmatic form.
    cmd=["codex", "exec", "--json", "-m", "{model}", "-s", "workspace-write",
         "--skip-git-repo-check", "-"],
    stream="codex", prompt_via_stdin=True,
    resume_cmd=["codex", "exec", "resume", "{session_id}", "-", "--json",
                "-m", "{model}", "--skip-git-repo-check"],
)
# The Claude Code CLI in print mode + stream-json events. The same command, pointed
# at an Anthropic-compatible endpoint via base_url, drives cheap third-party models.
_CLAUDE_CMD = ["claude", "-p", "{prompt}", "--model", "{model}",
               "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"]
_CLAUDE_RESUME_CMD = ["claude", "-p", "{prompt}", "--resume", "{session_id}", "--model", "{model}",
                      "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"]
CLAUDE = CliSpec("claude", cmd=_CLAUDE_CMD, stream="claude",
                 resume_cmd=_CLAUDE_RESUME_CMD)   # your real Claude subscription auth

# Cheap providers over the SAME claude CLI (Anthropic-compatible endpoints). The
# model string comes from `--tier` (e.g. openrouter/z-ai/glm-5.2, deepseek/deepseek-chat).
OPENROUTER = CliSpec("openrouter", cmd=_CLAUDE_CMD, stream="claude", resume_cmd=_CLAUDE_RESUME_CMD,
                     base_url="https://openrouter.ai/api", api_key_env="OPENROUTER_API_KEY")
GLM = CliSpec("glm", cmd=_CLAUDE_CMD, stream="claude", resume_cmd=_CLAUDE_RESUME_CMD,
              base_url="https://api.z.ai/api/anthropic", api_key_env="ZAI_API_KEY")
DEEPSEEK = CliSpec("deepseek", cmd=_CLAUDE_CMD, stream="claude", resume_cmd=_CLAUDE_RESUME_CMD,
                   base_url="https://api.deepseek.com/anthropic", api_key_env="DEEPSEEK_API_KEY")
KIMI = CliSpec("kimi", cmd=_CLAUDE_CMD, stream="claude", resume_cmd=_CLAUDE_RESUME_CMD,
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
    "BEFORE you write a single line of the kernel, do this reasoning explicitly (in your\n"
    "own thinking, not a separate file): (1) name WHICH of DESIGN.md's ranked approaches\n"
    "you're implementing and WHY that one now — see the frontier-state note in CONTEXT.md\n"
    "for whether this should be a safe correctness-first approach or the ambitious one;\n"
    "(2) pick the single trickiest graded shape in workloads.md (smallest, most irregular,\n"
    "or highest-risk for the launch config you're about to use) and manually trace it —\n"
    "the actual grid/block dims, loop bounds, and mask conditions your kernel will compute\n"
    "for THAT shape's real numbers. Most real bugs only show up on one specific shape, not\n"
    "in the general logic — 'the logic looks right' without doing this trace is how a third\n"
    "of candidates fail. Only start writing kernel.py once that trace actually checks out;\n"
    "if it doesn't, that's a sign to pick a different (usually simpler) approach, not to\n"
    "write the risky one anyway and hope the reviewer catches it.\n"
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
_REPAIR = (
    "An independent reviewer read the kernel you just wrote (still in this directory) and\n"
    "found issues before it ships to a real B200 GPU — a rented, single-flight GPU, so a bad\n"
    "kernel wastes real time for the whole fleet. Fix EVERY issue below; don't dismiss one\n"
    "because you're focused on speed:\n"
    "\n"
    "{critique}\n"
    "\n"
    "Edit the kernel file(s) IN PLACE in this directory (do not start over from reference.py)\n"
    "unless the review shows your whole approach is broken, in which case say so and rewrite.\n"
    "Update strategy.txt if your approach changed, and handoff.md if your reserve idea changed.\n"
    "NEVER leave the kernel byte-identical to what's already in this directory, and a\n"
    "comment-only or cosmetic edit does not count as a fix — if you think an issue is a false\n"
    "positive, still make a REAL defensive code change addressing it (e.g. an explicit bounds\n"
    "check or an extra guard condition): a no-op or cosmetic change is indistinguishable from\n"
    "not having read the review at all, and wastes this turn for no signal.\n"
    "Do NOT try to run or benchmark the kernel yourself — there is still no GPU/torch/triton in\n"
    "your environment. Only write the files; do not print the code."
)


@dataclass
class _Run:
    stderr: str
    rc: int
    tokens: dict           # {in, out, reasoning, cached, cost_usd} where the stream provides them
    trajectory: Path       # persisted raw event stream (trajectory.jsonl)
    error_hint: str = ""   # best-effort failure message pulled from STDOUT (see _extract_error_hint)
    context_read: list | None = None   # kb/*.md files actually consulted (see _extract_context_read)
    session_id: str | None = None      # this call's session/thread id (see _extract_session_id) —
                                        # lets a LATER call resume the exact same conversation


def _extract_error_hint(schema: str, raw: str) -> str:
    """Best-effort short error message pulled from the agent's STDOUT event stream.
    Both codex and claude report real failures (quota exceeded, auth, rate limits)
    as JSON events on stdout, not stderr — a call that dies this way leaves
    `res.stderr` completely empty even though the actual cause is sitting right
    there in the trajectory. Confirmed live (2026-07-07): a codex 'wrote no kernel'
    error with an empty stderr turned out to be a plain 'Quota exceeded' event —
    invisible until someone reads trajectory.jsonl by hand."""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if schema == "codex" and e.get("type") in ("error", "turn.failed"):
            msg = e.get("message") or (e.get("error") or {}).get("message")
            if msg:
                return str(msg)
        elif schema == "claude" and e.get("type") == "result" and e.get("is_error"):
            msg = e.get("result")
            if msg:
                return str(msg)
    return ""


_KB_PATH_RE = re.compile(r"(?:^|[\s'\"])((?:\./)?kb/[\w.\-]+\.md)")


def _extract_context_read(schema: str, raw: str) -> list[str]:
    """Which `kb/*.md` files this call actually consulted — auditable evidence
    for "is the model using the context we hand it", not just an assumption.
    claude schema: `Read` tool_use blocks (a structured file_path). codex schema
    has no Read tool at all — it reads files via plain shell (`command_execution`
    items running `nl`/`cat`/`rg`/`sed` etc.), so this greps the COMMAND TEXT for
    `kb/*.md` references instead; a naive claude-only check undercounts codex to
    zero even when it read half the knowledge base. Returns the distinct
    filenames in first-seen order (not full paths — every call's cwd differs)."""
    seen: list[str] = []
    seen_set: set[str] = set()

    def _add(name: str) -> None:
        name = name.rsplit("/", 1)[-1]
        if name not in seen_set:
            seen_set.add(name)
            seen.append(name)

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if schema == "claude" and e.get("type") == "assistant":
            for c in (e.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "Read":
                    fp = (c.get("input") or {}).get("file_path", "")
                    if "/kb/" in fp or fp.startswith("kb/"):
                        _add(fp)
        elif schema == "codex" and e.get("type") == "item.completed":
            item = e.get("item") or {}
            if item.get("type") == "command_execution":
                for m in _KB_PATH_RE.finditer(item.get("command", "")):
                    _add(m.group(1))
    return seen


def _extract_session_id(schema: str, raw: str) -> str | None:
    """The session/thread id this call ran under, so a LATER call can `--resume`/
    `resume` the exact same conversation instead of cold-starting (see
    CliAgent.repair). claude schema: every event carries `session_id` once the
    session is established. codex schema: `{"type": "thread.started",
    "thread_id": ...}` is the first line of the stream."""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if schema == "codex" and e.get("type") == "thread.started":
            tid = e.get("thread_id")
            if tid:
                return str(tid)
        elif schema == "claude":
            sid = e.get("session_id")
            if sid:
                return str(sid)
    return None


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
                 kb_dir: str | Path = "kb", knowledge_dir: str | Path = "knowledge",
                 cross_op_patterns: bool = True, env: dict | None = None) -> None:
        self.spec = spec
        self.model = model
        self.perspective = Perspective(spec.name, model)
        self.runs_dir = Path(runs_dir)
        self.problems_dir = Path(problems_dir)
        self.kb_dir = Path(kb_dir)
        # A plain path, not a live KnowledgeStore instance — read fresh off disk
        # every call (mirrors how reflection.md is read directly off disk below,
        # never memoized), so a long-running fleet always sees the latest
        # cross-op corpus without needing a shared object across agents/processes.
        self.knowledge_dir = Path(knowledge_dir)
        self.cross_op_patterns = cross_op_patterns
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
        if self.cross_op_patterns:
            # Cross-op notes at design()-time — BEFORE any candidate exists — are
            # the highest-leverage injection point: this is where the ranked
            # approach that determines which technique gets tried FIRST is
            # decided (design() never calls _context_md(), unlike plan()/
            # review(), so without this the corpus is unreachable until the
            # strategic ranking is already locked in).
            self._write_patterns(wd, op_key_of(task_id, self.problems_dir), task_id)
        res = await self._run(wd, _DESIGN)
        f = wd / F_DESIGN
        text = f.read_text().strip() if f.exists() else "(no design produced)"
        return text, (res.tokens or {})

    async def plan(self, parent, ctx) -> Candidate:
        self._seq += 1
        wd = self._workdir(ctx.task_id, f"cand{self._seq}")
        shown_tags = self._seed_workdir(wd, parent, ctx)
        res = await self._run(wd, _PLAN)
        solution = self._collect(wd)
        if not solution["sources"]:                    # loud, not a silent baseline candidate
            detail = res.error_hint or res.stderr[:400]
            raise RuntimeError(f"{self.spec.name}/{self.model} wrote no kernel "
                               f"(exit {res.rc}): {detail}")
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
                         trajectory=str(traj_path), context_read=res.context_read or [],
                         session_id=res.session_id,
                         cross_op_patterns_shown=shown_tags or None)

    async def review(self, cand: Candidate, ctx) -> ReviewVerdict:
        """Pre-GPU code review: an INDEPENDENT (agent,model) — never the one that
        wrote `cand` — reads the kernel against reference.py + workloads.md and
        judges ship/revise before a single GPU eval is spent."""
        wd = self._workdir(getattr(ctx, "task_id", ""), f"review-{cand.cand_id}-{self._seq}")
        self._seq += 1
        text, _shown_tags = _context_md(cand, ctx, self)   # reviewer's own shown-tags not tracked —
        (wd / "CONTEXT.md").write_text(text)               # cand's are already recorded from plan()
        self._write_problem(wd, getattr(ctx, "task_id", ""))
        self._write_kb(wd)                     # same knowledge base the writer consulted
        for s in (cand.solution or {}).get("sources", []):
            (wd / s["path"]).write_text(s.get("content", ""))
        res = await self._run(wd, _REVIEW)
        f = wd / F_REVIEW
        verdict = _parse_review(f.read_text() if f.exists() else "", reviewer=f"{self.spec.name}:{self.model}")
        verdict.cost_usd = (res.tokens or {}).get("cost_usd") or 0.0
        verdict.tokens = res.tokens or {}
        verdict.context_read = res.context_read or []
        return verdict

    async def repair(self, cand: Candidate, critique: str, ctx) -> Candidate:
        """Fix a reviewer-flagged candidate by RESUMING the writer's OWN CLI session,
        in its own persisted directory, instead of cold-starting a fresh plan() call.
        A fresh process given only a critique text file has no memory of why the
        original kernel looks the way it does — confirmed live (2026-07-08, problem
        #18): a cold-start repair produced a BYTE-IDENTICAL kernel to the one just
        flagged, and the reviewer, re-reviewing the same bytes seconds later,
        flip-flopped from 'revise' to 'ship'. Resuming gives the model its own
        reasoning back instead of asking it to reinterpret a stranger's code from a
        critique alone.
        `--resume`/`resume` are scoped to the EXACT working directory the session
        started in (verified empirically: resuming from a different cwd fails with
        "No conversation found"), so this reuses cand's own workdir UNCHANGED —
        never rename it, or a later repair round in the same lineage can't resume
        it either.
        Falls back to a cold-start plan() when resume isn't available (no captured
        session id, no resume_cmd for this spec, or the workdir is gone) — repair
        must never hard-fail just because the fast path isn't available."""
        wd = self.runs_dir / str(getattr(ctx, "task_id", "")) / "work" / cand.cand_id
        if not cand.session_id or not self.spec.resume_cmd or not wd.is_dir():
            return await self.plan(cand, ctx)
        n = sum(1 for _ in wd.glob("trajectory.repair-*.jsonl")) + 1
        res = await self._run(wd, _REPAIR.format(critique=critique.strip()),
                              session_id=cand.session_id, traj_name=f"trajectory.repair-{n}")
        # the resumed session edits kernel.py/kernel.cu IN PLACE in a directory we
        # deliberately never clear between rounds (needed for resume) — if it
        # switches language without deleting the old file, keep only the freshest
        # one rather than shipping a mixed-language "solution" (see _collect).
        solution = self._collect(wd, resolve_conflict=True)
        if not solution["sources"]:
            detail = res.error_hint or res.stderr[:400]
            raise RuntimeError(f"{self.spec.name}/{self.model} repair wrote no kernel "
                               f"(exit {res.rc}): {detail}")
        strat = wd / F_STRATEGY
        strategy = (strat.read_text().strip()[:120] if strat.exists() else "") or cand.strategy
        hf = wd / F_HANDOFF
        handoff = (hf.read_text().strip()[:600] if hf.exists() else "") or cand.handoff
        new_id = solution_hash(solution)[:12]
        return Candidate(cand_id=new_id, solution=solution, parent=cand.cand_id,
                         agent=self.spec.name, model=self.model, strategy=strategy,
                         handoff=handoff, tokens=res.tokens or None,
                         trajectory=str(res.trajectory), context_read=res.context_read or [],
                         cross_op_patterns_shown=cand.cross_op_patterns_shown,   # resumed session,
                                              # no fresh CONTEXT.md — inherit what plan() showed it
                         session_id=cand.session_id)

    # ---- helpers ----
    def _workdir(self, key, sub: str) -> Path:
        """A fresh scratch dir for one attempt. `sub` (e.g. `cand{seq}`) is only
        unique WITHIN one process lifetime — `self._seq` resets to 0 on every
        restart, so the same name gets reused across restarts. A candidate whose
        plan() raised before `_rekey_workdir` ever ran (e.g. the "wrote no kernel"
        RuntimeError) leaves its dir behind un-renamed; `mkdir(exist_ok=True)`
        alone would silently inherit that leftover content into the NEXT attempt
        that reuses the same name. Confirmed live (2026-07-08): a stale Triton
        kernel.py from a prior day survived into a fresh CUDA repair attempt,
        got flagged as 'mixed C++/Python' (correctly, given what was actually in
        the directory) and check-rejected — which then shipped the ORIGINAL,
        reviewer-flagged-buggy candidate to the GPU as the fallback. Clearing
        first guarantees every attempt starts from nothing but what THIS call
        writes."""
        wd = self.runs_dir / str(key) / "work" / sub
        if wd.exists():
            shutil.rmtree(wd, ignore_errors=True)
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

    def _seed_workdir(self, wd: Path, parent, ctx) -> list[str]:
        (wd / "DESIGN.md").write_text(getattr(ctx, "design", "") or "")
        text, shown_tags = _context_md(parent, ctx, self)
        (wd / "CONTEXT.md").write_text(text)
        self._write_problem(wd, getattr(ctx, "task_id", ""))
        self._write_kb(wd)
        psol = getattr(parent, "solution", None)
        for s in (psol or {}).get("sources", []):     # starting point = the parent's kernel
            (wd / s["path"]).write_text(s.get("content", ""))
        for s in (getattr(ctx, "sibling_hint", None) or {}).get("sources", []):   # cross-op warm start
            (wd / f"sibling_{s['path']}").write_text(s.get("content", ""))
        return shown_tags

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

    def _write_patterns(self, wd: Path, family: str, task_id=None) -> list[str]:
        """PATTERNS.md: cross-OPERATOR technique notes relevant to this op's
        compute-bound class (docs/context-architecture-plan.md Part B). Needs
        ZERO run history — usable from design() before any candidate exists,
        unlike the coach card. Returns the tags that had non-empty notes (for
        Candidate.cross_op_patterns_shown's later measurement), writing nothing
        if there's nothing to show."""
        tags = relevant_techniques(family)
        notes = KnowledgeStore(self.knowledge_dir).pattern_notes(tags, exclude_task=task_id)
        if notes:
            (wd / "PATTERNS.md").write_text(_render_patterns_md(notes))
        return sorted(notes)

    def _collect(self, wd: Path, *, resolve_conflict: bool = False) -> dict:
        """Gather the kernel file(s) a call wrote. `resolve_conflict` is for
        repair(), which deliberately reuses an existing directory across rounds
        (needed for session resume) instead of starting from a clean one: if the
        model switched language (e.g. kernel.py → kernel.cu) without deleting the
        old file, keep only the MOST RECENTLY WRITTEN language rather than
        shipping a mixed-language "solution" — the same one-language rule the
        writer/reviewer prompts enforce, resolved by recency instead of a reject."""
        files = [f for f in sorted(wd.glob(self.spec.kernels)) if f.is_file()]
        if resolve_conflict and len({f.suffix for f in files}) > 1:
            newest_ext = max(files, key=lambda f: f.stat().st_mtime).suffix
            files = [f for f in files if f.suffix == newest_ext]
        sources = [{"path": f.name, "content": f.read_text(errors="replace")} for f in files]
        langs = sorted({self.spec.lang or _EXT_LANG.get(f.suffix.lstrip("."), "cuda_cpp")
                        for f in files}) or ["cuda_cpp"]
        return {"spec": {"languages": langs}, "sources": sources}

    async def _run(self, wd: Path, prompt: str, *, session_id: str | None = None,
                   traj_name: str = "trajectory") -> _Run:
        template = self.spec.resume_cmd if session_id else self.spec.cmd
        if session_id and not template:
            raise RuntimeError(f"{self.spec.name} has no resume_cmd configured")
        cmd = [t.replace("{model}", self.model).replace("{prompt}", prompt)
                .replace("{session_id}", session_id or "") for t in template]
        # Feed the prompt on stdin (codex) or as an argv with stdin closed (claude).
        # codex with an argv prompt + DEVNULL stdin flakily dies "Reading additional
        # input from stdin"; giving it the prompt ON stdin (cmd ends with `-`) fixes it.
        stdin_data = prompt.encode() if self.spec.prompt_via_stdin else None
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(wd), env=self.env,
            stdin=(asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        traj = wd / f"{traj_name}.jsonl"
        traj.write_text("")   # exists immediately — a live-transcript viewer has
                               # something to tail before the call finishes
        out_chunks: list[bytes] = []

        async def _feed_stdin() -> None:
            if stdin_data is None:
                return
            proc.stdin.write(stdin_data)
            await proc.stdin.drain()
            proc.stdin.close()

        async def _drain_stdout() -> None:
            # Written INCREMENTALLY (as chunks arrive, flushed) rather than
            # buffered to the end — the dashboard's live-transcript view reads
            # whatever has landed so far while the call is still running.
            # Reads fixed-size CHUNKS, not lines: StreamReader.readline() has a
            # default 64KB-per-line buffer limit and raises ValueError
            # ("Separator is found, but chunk is longer than limit") on
            # anything longer — confirmed live (2026-07-08): a single event
            # line over 64KB (e.g. a large kernel embedded in one JSON message)
            # killed the whole plan/review call as a bare exception, a real
            # regression from the very first version of this change. A torn
            # line at a chunk boundary is harmless — render_stream's per-line
            # try/catch already tolerates a cut-off trailing line.
            with open(traj, "a", encoding="utf-8") as f:
                while True:
                    chunk = await proc.stdout.read(65536)
                    if not chunk:
                        break
                    out_chunks.append(chunk)
                    f.write(chunk.decode("utf-8", "replace"))
                    f.flush()

        async def _drain_stderr() -> bytes:
            return await proc.stderr.read()

        try:
            _, _, err = await asyncio.wait_for(
                asyncio.gather(_feed_stdin(), _drain_stdout(), _drain_stderr()),
                timeout=self.timeout)
            await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"{self.spec.name} timed out after {self.timeout}s")
        raw = b"".join(out_chunks).decode("utf-8", "replace")
        await asyncio.to_thread(self._render, raw, wd / f"{traj_name}.txt")   # readable render (best-effort)
        return _Run(err.decode("utf-8", "replace"), proc.returncode or 0,
                    _parse_tokens(self.spec.stream, raw), traj,
                    _extract_error_hint(self.spec.stream, raw),
                    _extract_context_read(self.spec.stream, raw),
                    _extract_session_id(self.spec.stream, raw))

    def _render(self, raw: str, out_path: Path) -> None:
        """Render the raw stream to readable text via the vendored jq wrapper."""
        text = render_stream(self.spec.stream, raw)
        if text:
            out_path.write_text(text)


# Schema each spec's event stream follows — used by the dashboard to render a
# LIVE (still-growing) trajectory the same way a finished one is rendered,
# without needing a CliAgent instance (see render_stream / solver/dashboard).
AGENT_STREAM: dict[str, str] = {name: spec.stream for name, spec in SPECS.items()}


def render_stream(schema: str, raw: str) -> str:
    """Render a raw agent event stream to readable text via the vendored jq
    wrapper. Safe to call on a PARTIAL stream (e.g. a call still in progress):
    each helper script wraps its per-line parse in try/catch, so one incomplete
    trailing line renders as '[unparsed] ...' instead of failing the whole
    render. Returns "" (never raises) if jq/bash aren't available or the
    render fails outright."""
    helper = _HELPERS / f"{schema}_stream.sh"
    if not raw.strip() or not helper.exists() or not shutil.which("jq") or not shutil.which("bash"):
        return ""
    try:
        r = subprocess.run(["bash", str(helper)], input=raw, capture_output=True,
                           text=True, timeout=30)
        return r.stdout or ""
    except Exception:
        return ""


def _member_cal(m):
    return getattr(m, "sol_score_cal", None)


def _score_note(m) -> str:
    """`leaderboard-est (raw N)` — est is the number that ranks you on the board
    (0.5 = optimized-PyTorch baseline, 1.0 = speed-of-light); raw is our local measure."""
    cal = _member_cal(m)
    return f"{cal:.3f} (raw {m.mean:.3f})" if cal is not None else f"{m.mean:.3f} (raw)"


def _render_patterns_md(notes: dict[str, dict]) -> str:
    """Shared renderer for PATTERNS.md (design()-time) and CONTEXT.md's cross-op
    section (plan()/review()-time) — same content, different filenames. Pitfalls
    render as traps to AVOID while re-deriving the technique, never as a reason
    to avoid the technique itself (docs/context-architecture-plan.md Part B) —
    a COMPILE_ERROR/RUNTIME_ERROR is overwhelmingly an implementation bug, not a
    verdict on the underlying idea."""
    lines = ["## Cross-op technique notes", "",
             "Patterns confirmed or flagged on OTHER, unrelated operators — NOT this",
             "problem's kernel. A technique that worked elsewhere can still fail here on",
             "IMPLEMENTATION details even when the algorithmic idea is sound — re-derive",
             "and re-validate for THIS op's shapes/dtypes, do not copy verbatim.", ""]
    for tag in sorted(notes):
        entry = notes[tag]
        lines.append(f"### `{tag}`")
        for e in entry.get("confirmed", []):
            lines.append(f"- confirmed on `{e['op']}` (task {e['task']}, score {e['score']:.3f}): "
                         f"{e['note']} — see `runs/{e['task']}/candidates/{e['cand']}.json`")
        for p in entry.get("pitfalls", []):
            ops = ", ".join(sorted(set(p.get("ops", []))))
            lines.append(f"- **pitfall** ({p['error_family']}, seen on {ops}): {p['note']} "
                         f"— avoid THIS bug, the technique itself isn't the problem")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _cross_op_section(ctx, agent) -> tuple[str, list[str]]:
    """Fresh `pattern_notes()` lookup every call — NEVER memoized on `ctx` (same
    reasoning as the `ctx.reflection` dead-code fallthrough a few lines below:
    a long fleet run or a resume must always see the LATEST cross-op corpus, and
    unlike `ctx.sibling_hint` this isn't meant to be bootstrap-only). Returns
    (markdown, tags_shown) — the tags feed Candidate.cross_op_patterns_shown for
    later measurement. `agent` is the calling CliAgent instance (holds
    knowledge_dir/problems_dir/cross_op_patterns; module-level so tests can call
    `_context_md` without one — see cli_agent.py's own docstring convention)."""
    if agent is None or not getattr(agent, "cross_op_patterns", True):
        return "", []
    task_id = getattr(ctx, "task_id", None)
    if task_id is None:
        return "", []
    family = op_key_of(task_id, agent.problems_dir)
    tags = relevant_techniques(family)
    notes = KnowledgeStore(agent.knowledge_dir).pattern_notes(tags, exclude_task=task_id)
    if not notes:
        return "", []
    return _render_patterns_md(notes), sorted(notes)


def _context_md(parent, ctx, agent=None) -> tuple[str, list[str]]:
    lines = ["# Context", "",
             "Scores are the **leaderboard estimate** (0.5 = optimized-PyTorch baseline,",
             "1.0 = speed-of-light). Beat 0.5 to score on the board; the raw local measure",
             "is in parentheses.", ""]
    if parent is not None and getattr(parent, "solution", None):
        lines.append("The parent kernel to improve on is in this directory's kernel file(s).")
    fr = getattr(ctx, "frontier", None)
    members = list(getattr(fr, "members", None) or []) if fr else []
    if members and all((m.strategy or "") == "seed" for m in members):
        lines += ["", "## Nothing real has been accepted yet — sequence risk accordingly",
                  "The frontier is still just the unoptimized reference seed; no agent has banked",
                  "a genuine correctness win on this problem. Per DESIGN.md's ranked approaches,",
                  "implement the SAFEST/most-reliable one now to get a real, correct score on the",
                  "board — not the highest-ceiling one. A failed ambitious attempt leaves the",
                  "frontier exactly where it is now (worse than a modest real win); once a working",
                  "baseline exists, later rounds can escalate toward the higher-ceiling approach",
                  "with a fallback already secured."]
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
    patterns_md, shown_tags = _cross_op_section(ctx, agent)
    if patterns_md:
        lines += ["", patterns_md.strip()]
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
    return "\n".join(lines) + "\n", shown_tags


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
