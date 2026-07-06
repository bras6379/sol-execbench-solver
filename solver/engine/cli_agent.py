"""CliAgent — drive the **codex** and **claude** CLIs behind the Agent interface.

We don't implement an agent; we shell out to one (docs/agent.md). Scoped to
codex and claude only; other agent types would be integrated separately.

Two disciplines make this robust:
- **Results come from known files, never parsed stdout.** The agent is told to
  write its output to fixed filenames in the workdir — `kernel.<ext>` +
  `strategy.txt` (plan), `design.md`, `reflection.txt`, `verdict.txt` — and we
  read those files. The model's prose is never parsed for the answer.
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
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .agent import Candidate, solution_hash
from .config import Config, Perspective

_HELPERS = Path(__file__).resolve().parent.parent / "agent_helpers"
_EXT_LANG = {"py": "pytorch", "cu": "cuda_cpp", "cuh": "cuda_cpp",
             "cpp": "cuda_cpp", "cc": "cuda_cpp"}

# Known output files — the agent writes these; we read them (no stdout parsing).
KERNELS = "kernel.*"
F_STRATEGY = "strategy.txt"
F_DESIGN = "design.md"
F_REFLECT = "reflection.txt"
F_VERDICT = "verdict.txt"


@dataclass
class CliSpec:
    """A supported agent CLI (codex or claude). One streaming, write-capable
    command template; its JSON event schema (`stream`) is parsed for tokens."""

    name: str
    cmd: list[str]                  # {model}/{prompt} substituted; runs with write access + JSON stream
    stream: str                     # event-stream schema: "codex" | "claude"
    kernels: str = KERNELS
    lang: str | None = None


CODEX = CliSpec(
    "codex",   # from `codex exec --help`: --json stream, -m model, -s sandbox, positional prompt
    cmd=["codex", "exec", "--json", "-m", "{model}", "-s", "workspace-write",
         "--skip-git-repo-check", "{prompt}"],
    stream="codex",
)
CLAUDE = CliSpec(
    "claude",  # Claude Code print mode, stream-json events (needs --verbose)
    cmd=["claude", "-p", "{prompt}", "--model", "{model}",
         "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"],
    stream="claude",
)
SPECS: dict[str, CliSpec] = {"codex": CODEX, "claude": CLAUDE}


_PLAN = (
    "You are optimizing a GPU kernel for NVIDIA B200 (SOL-ExecBench).\n"
    "reference.py is the PyTorch reference you must stay numerically equivalent to;\n"
    "definition.json is the spec; DESIGN.md and CONTEXT.md hold the plan and prior\n"
    "attempts. Write your optimized implementation to a file named kernel.py or\n"
    "kernel.cu (one language), and a one-line summary of the approach to strategy.txt.\n"
    "Follow the ladder torch -> Triton -> CuTe/CUTLASS -> C++/PTX; escalate only when\n"
    "needed. Do not print the code — only write the files."
)
_DESIGN = (
    "Analyze reference.py and definition.json for NVIDIA B200. Write a short markdown\n"
    "design (op graph, per-shape roofline memory- vs compute-bound, 3 ranked\n"
    "approaches) to a file named design.md."
)


def _reflect_prompt(cand: Candidate, result, verdict: str) -> str:
    return (f"A GPU kernel (strategy: '{cand.strategy}') scored "
            f"{getattr(result, 'sol_score', None)} (frontier verdict: {verdict}). Write a "
            f"2-3 sentence diagnosis of the single most promising next optimization to a "
            f"file named reflection.txt.")


def _judge_prompt(cand: Candidate, parent) -> str:
    return (f"Candidate B strategy: '{cand.strategy}'. Parent A strategy: "
            f"'{getattr(parent, 'strategy', '')}'. Is B a materially different kernel "
            f"implementation (algorithm/layout/fusion/precision/launch) from A, or a "
            f"cosmetic variant? Write exactly one word — materially-new or cosmetic — to a "
            f"file named verdict.txt.")


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
            tokens = {"in": u.get("input_tokens"), "out": u.get("output_tokens"),
                      "cost_usd": e.get("total_cost_usd")}
    return {k: v for k, v in tokens.items() if v is not None}


class CliAgent:
    """An Agent that drives codex or claude via subprocess."""

    def __init__(self, spec: CliSpec, model: str, *, runs_dir: str | Path = "runs",
                 problems_dir: str | Path = "problems", timeout: float = 600.0,
                 env: dict | None = None) -> None:
        self.spec = spec
        self.model = model
        self.perspective = Perspective(spec.name, model)
        self.runs_dir = Path(runs_dir)
        self.problems_dir = Path(problems_dir)
        self.timeout = timeout
        self.env = {**os.environ, **(env or {})}
        self._seq = 0

    async def design(self, task_id: int) -> str:
        wd = self._workdir(task_id, "design")
        self._write_problem(wd, task_id)
        await self._run(wd, _DESIGN)
        f = wd / F_DESIGN
        return f.read_text().strip() if f.exists() else "(no design produced)"

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
        cand_id = solution_hash(solution)[:12]
        wd = self._rekey_workdir(wd, ctx.task_id, cand_id)   # kernel + trajectory + inputs, keyed by cand
        return Candidate(cand_id=cand_id, solution=solution,
                         parent=getattr(parent, "cand_id", None),
                         agent=self.spec.name, model=self.model, strategy=strategy,
                         tokens=res.tokens or None,
                         trajectory=str(wd / "trajectory.jsonl"))

    async def reflect(self, cand: Candidate, result, verdict: str) -> str:
        with tempfile.TemporaryDirectory(prefix="agent-reflect-") as d:   # text output, ephemeral
            await self._run(Path(d), _reflect_prompt(cand, result, verdict))
            f = Path(d) / F_REFLECT
            return f.read_text().strip() if f.exists() else ""

    async def judge(self, cand: Candidate, parent, frontier) -> str:
        with tempfile.TemporaryDirectory(prefix="agent-judge-") as d:
            await self._run(Path(d), _judge_prompt(cand, parent))
            f = Path(d) / F_VERDICT
            txt = f.read_text().lower() if f.exists() else ""
            return "cosmetic" if "cosmetic" in txt else "materially-new"

    # ---- helpers ----
    def _workdir(self, key, sub: str) -> Path:
        wd = self.runs_dir / str(key) / "work" / sub
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    def _rekey_workdir(self, wd: Path, task_id, cand_id: str) -> Path:
        """Rename the plan workdir to be keyed by cand_id, so the kernel and its
        trajectory persist together under runs/<task>/work/<cand_id>/."""
        dest = self.runs_dir / str(task_id) / "work" / cand_id
        if dest.resolve() == wd.resolve():
            return wd
        if dest.exists():                              # duplicate hash: discard the redundant workdir
            shutil.rmtree(wd, ignore_errors=True)
            return dest
        wd.rename(dest)
        return dest

    def _seed_workdir(self, wd: Path, parent, ctx) -> None:
        (wd / "DESIGN.md").write_text(getattr(ctx, "design", "") or "")
        (wd / "CONTEXT.md").write_text(_context_md(parent, ctx))
        self._write_problem(wd, getattr(ctx, "task_id", ""))
        psol = getattr(parent, "solution", None)
        for s in (psol or {}).get("sources", []):     # starting point = the parent's kernel
            (wd / s["path"]).write_text(s.get("content", ""))

    def _write_problem(self, wd: Path, task_id) -> None:
        pdir = self.problems_dir / str(task_id)
        for fname in ("reference.py", "definition.json"):
            src = pdir / fname
            if src.exists():
                (wd / fname).write_text(src.read_text())

    def _collect(self, wd: Path) -> dict:
        files = [f for f in sorted(wd.glob(self.spec.kernels)) if f.is_file()]
        sources = [{"path": f.name, "content": f.read_text(errors="replace")} for f in files]
        langs = sorted({self.spec.lang or _EXT_LANG.get(f.suffix.lstrip("."), "cuda_cpp")
                        for f in files}) or ["cuda_cpp"]
        return {"spec": {"languages": langs}, "sources": sources}

    async def _run(self, wd: Path, prompt: str) -> _Run:
        cmd = [t.replace("{model}", self.model).replace("{prompt}", prompt) for t in self.spec.cmd]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(wd), env=self.env,
            stdin=asyncio.subprocess.DEVNULL,      # non-interactive: the CLI must not read stdin
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
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


def _context_md(parent, ctx) -> str:
    lines = ["# Context", ""]
    if parent is not None and getattr(parent, "solution", None):
        lines.append("The current best kernel is in this directory's kernel file(s).")
        if getattr(parent, "reflection", None):
            lines += ["", "## Reflection on it", parent.reflection]
    fr = getattr(ctx, "frontier", None)
    members = getattr(fr, "members", None) if fr else None
    if members:
        lines += ["", "## Frontier (score · strategy)"]
        lines += [f"- {m.cand_id[:8]}  {m.mean:.3f}  {m.strategy}" for m in members[:8]]
    return "\n".join(lines) + "\n"


def make_agents(cfg: Config, specs: dict[str, CliSpec] = SPECS, **kwargs) -> dict:
    """Map every perspective in `cfg` to a CliAgent for its agent's CLI spec."""
    out = {}
    for p in cfg.perspectives:
        spec = specs.get(p.agent)
        if spec is None:
            raise KeyError(f"no CLI spec for agent {p.agent!r} (supported: {sorted(specs)})")
        out[p] = CliAgent(spec, p.model, **kwargs)
    return out
