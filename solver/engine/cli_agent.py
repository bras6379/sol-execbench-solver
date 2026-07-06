"""CliAgent — abstract existing coding-agent CLIs behind the Agent interface.

We don't implement an agent; we shell out to one (docs/agent.md). Each CLI is a
`CliSpec` (two command templates + a kernel glob); `CliAgent` drives it as a
subprocess: seed a workdir with the §8 context, run the CLI, collect the kernel
files it wrote into a Solution (plan/design), or read stdout (reflect/judge).
Language-agnostic, sandboxable, no per-CLI code.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .agent import Candidate, solution_hash
from .config import Config, Perspective

_EXT_LANG = {"py": "pytorch", "cu": "cuda_cpp", "cuh": "cuda_cpp",
             "cpp": "cuda_cpp", "cc": "cuda_cpp", "cutlass": "cutlass"}


@dataclass
class CliSpec:
    name: str
    edit: list[str]                 # file-editing template (plan/design); {model}/{prompt} substituted
    ask: list[str]                  # text-answer template (reflect/judge)
    kernels: str = "kernel.*"       # glob for the files the agent writes
    lang: str | None = None         # force a language, else infer from extension


CODEX = CliSpec(
    "codex",   # grounded in `codex exec --help`: -m model, -s sandbox, positional prompt
    edit=["codex", "exec", "-m", "{model}", "-s", "workspace-write",
          "--skip-git-repo-check", "{prompt}"],
    ask=["codex", "exec", "-m", "{model}", "-s", "read-only",
         "--skip-git-repo-check", "{prompt}"],
)
CLAUDE = CliSpec(
    "claude",
    edit=["claude", "-p", "{prompt}", "--model", "{model}", "--dangerously-skip-permissions"],
    ask=["claude", "-p", "{prompt}", "--model", "{model}"],
)
SPECS: dict[str, CliSpec] = {"codex": CODEX, "claude": CLAUDE}


_PLAN = (
    "You are optimizing a CUDA kernel for NVIDIA B200 (SOL-ExecBench).\n"
    "Read DESIGN.md and CONTEXT.md in this directory. Improve on the current kernel\n"
    "and write your implementation to kernel.<ext> (e.g. kernel.cu or kernel.py).\n"
    "Follow the abstraction ladder torch -> Triton -> CuTe/CUTLASS -> C++/PTX; escalate\n"
    "only when needed. Write ONLY the kernel file(s); print one line 'STRATEGY: <summary>'."
)
_DESIGN = (
    "Analyze this SOL-ExecBench problem for NVIDIA B200. Produce a short design as\n"
    "markdown: op graph, per-shape roofline (memory- vs compute-bound), and 3 ranked\n"
    "candidate approaches. Output only the design text."
)


def _reflect_prompt(cand: Candidate, result, verdict: str) -> str:
    score = getattr(result, "sol_score", None)
    return (f"A GPU kernel using strategy '{cand.strategy}' scored {score} "
            f"(frontier verdict: {verdict}). In 2-3 sentences, name the single most "
            f"promising next optimization. Output only the diagnosis.")


def _judge_prompt(cand: Candidate, parent) -> str:
    return (f"Candidate B strategy: '{cand.strategy}'. Parent A strategy: "
            f"'{getattr(parent, 'strategy', '')}'. Is B a materially different kernel "
            f"implementation (algorithm/layout/fusion/precision/launch) from A, or a "
            f"cosmetic variant? Answer exactly 'materially-new' or 'cosmetic'.")


class CliAgent:
    """An Agent that drives an existing coding-agent CLI via subprocess."""

    def __init__(self, spec: CliSpec, model: str, *, runs_dir: str | Path = "runs",
                 timeout: float = 600.0, env: dict | None = None) -> None:
        self.spec = spec
        self.model = model
        self.perspective = Perspective(spec.name, model)
        self.runs_dir = Path(runs_dir)
        self.timeout = timeout
        self.env = {**os.environ, **(env or {})}
        self._seq = 0

    async def design(self, task_id: int) -> str:
        wd = self._workdir(task_id, "design")
        out, _err, _rc = await self._run(self.spec.ask, wd, _DESIGN)
        return out.strip() or "(no design produced)"

    async def plan(self, parent, ctx) -> Candidate:
        self._seq += 1
        wd = self._workdir(ctx.task_id, f"cand{self._seq}")
        self._seed_workdir(wd, parent, ctx)
        stdout, stderr, rc = await self._run(self.spec.edit, wd, _PLAN)
        solution, strategy = self._collect(wd, stdout)
        if not solution["sources"]:                    # loud, not a silent baseline candidate
            raise RuntimeError(f"{self.spec.name}/{self.model} produced no kernel "
                               f"(exit {rc}): {(stderr or stdout)[:400]}")
        return Candidate(cand_id=solution_hash(solution)[:12], solution=solution,
                         parent=getattr(parent, "cand_id", None),
                         agent=self.spec.name, model=self.model, strategy=strategy)

    async def reflect(self, cand: Candidate, result, verdict: str) -> str:
        wd = self._workdir(getattr(cand, "cand_id", "x"), "reflect")
        out, _e, _r = await self._run(self.spec.ask, wd, _reflect_prompt(cand, result, verdict))
        return out.strip()

    async def judge(self, cand: Candidate, parent, frontier) -> str:
        wd = self._workdir(getattr(cand, "cand_id", "x"), "judge")
        out, _e, _r = await self._run(self.spec.ask, wd, _judge_prompt(cand, parent))
        return "cosmetic" if "cosmetic" in out.lower() else "materially-new"

    # ---- helpers ----
    def _workdir(self, key, sub: str) -> Path:
        wd = self.runs_dir / str(key) / "work" / sub
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    def _seed_workdir(self, wd: Path, parent, ctx) -> None:
        (wd / "DESIGN.md").write_text(getattr(ctx, "design", "") or "")
        (wd / "CONTEXT.md").write_text(_context_md(parent, ctx))
        psol = getattr(parent, "solution", None)
        for s in (psol or {}).get("sources", []):     # starting point = the parent's kernel
            (wd / s["path"]).write_text(s.get("content", ""))

    def _collect(self, wd: Path, stdout: str) -> tuple[dict, str]:
        files = [f for f in sorted(wd.glob(self.spec.kernels)) if f.is_file()]
        sources = [{"path": f.name, "content": f.read_text(errors="replace")} for f in files]
        langs = sorted({self.spec.lang or _EXT_LANG.get(f.suffix.lstrip("."), "cuda_cpp")
                        for f in files}) or ["cuda_cpp"]
        solution = {"spec": {"languages": langs}, "sources": sources}
        return solution, _strategy_from(stdout)

    async def _run(self, template: Sequence[str], wd: Path, prompt: str) -> tuple[str, str, int]:
        cmd = [t.replace("{model}", self.model).replace("{prompt}", prompt) for t in template]
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
        return (out.decode("utf-8", "replace"), err.decode("utf-8", "replace"),
                proc.returncode or 0)


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


def _strategy_from(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.strip().upper().startswith("STRATEGY:"):
            return line.split(":", 1)[1].strip()[:120]
    first = next((l.strip() for l in stdout.splitlines() if l.strip()), "")
    return first[:120] or "cli agent"


def make_agents(cfg: Config, specs: dict[str, CliSpec] = SPECS, **kwargs) -> dict:
    """Map every perspective in `cfg` to a CliAgent for its agent's CLI spec."""
    out = {}
    for p in cfg.perspectives:
        spec = specs.get(p.agent)
        if spec is None:
            raise KeyError(f"no CLI spec for agent {p.agent!r} "
                           f"(known: {sorted(specs)})")
        out[p] = CliAgent(spec, p.model, **kwargs)
    return out
