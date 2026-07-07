# Agent backends: drive the codex & claude CLIs

**Status: `CliAgent` built (laptop-verified with a fake CLI). Scoped to codex
and claude only** — other agent types would be integrated separately, not by
adding a spec here.

We don't implement an agent; we shell out to one (`solver/engine/cli_agent.py`).
`CliAgent` seeds a workdir with the §8 context, runs the CLI, and turns its
output into a `Candidate`.

```
Perspective("codex", "gpt-5.5")   →   CliAgent(CODEX, "gpt-5.5")
Perspective("claude", "opus")     →   CliAgent(CLAUDE, "opus")
```

## Two disciplines that make it robust

**1. Results come from known files — never parsed stdout.** The prompt tells the
agent exactly which file to write; we read that file. The model's prose is never
parsed for the answer.

| Method | Agent writes | We read |
|---|---|---|
| `plan` | `kernel.<ext>` (+ `strategy.txt` + `handoff.md`) | glob `kernel.*` → Solution; `strategy.txt`; `handoff.md` |
| `design` | `design.md` | `design.md` |

`handoff.md` is the plan agent's own reserve play — the higher-ceiling idea it did
NOT ship. On accept it's banked into the per-problem **playbook** (`runs/<task>/
playbook.md`) and fed to the next agent's `CONTEXT.md`, so forward-looking reasoning
accumulates instead of dying in the trajectory. (There is no separate `reflect`/
`judge` call — their output was discarded; the ε-Pareto frontier is the novelty gate
and the playbook is the "text gradient".)

A `plan` that writes no kernel raises (exit code + stderr) → the fleet journals a
`solver_error` for that problem; the default `check` gate also rejects an empty
Solution. Loud, never a silent baseline candidate.

**2. The event stream is the trajectory.** Each CLI runs in **streaming JSON
mode** (`codex exec --json`, `claude -p --output-format stream-json --verbose`).
We persist the raw stream as `trajectory.jsonl`, render a readable
`trajectory.txt` via the vendored `solver/agent_helpers/{codex,claude}_stream.sh`
jq wrappers, and parse it **only for token usage** (`turn.completed` / `result`).
Each plan's workdir is renamed to its `cand_id`, so the kernel, its trajectory,
and its inputs persist together:

```
runs/<task>/work/<cand_id>/
  kernel.py | kernel.cu        the produced kernel (→ Solution.sources)
  strategy.txt                 one-line approach (→ Candidate.strategy)
  trajectory.jsonl             raw agent event stream (how the kernel was made)
  trajectory.txt               readable render (jq wrapper)
  reference.py definition.json the problem it optimized
  DESIGN.md CONTEXT.md         the §8 context it was given
```

`plan_done` journals `tok_in`/`tok_out` and the `trajectory` path, so every
kernel is linked to its trajectory and cost.

## CliSpec — one command template per CLI

```python
CODEX  = CliSpec("codex",
    cmd=["codex","exec","--json","-m","{model}","-s","workspace-write","--skip-git-repo-check","{prompt}"],
    stream="codex")
CLAUDE = CliSpec("claude",
    cmd=["claude","-p","{prompt}","--model","{model}","--output-format","stream-json","--verbose","--dangerously-skip-permissions"],
    stream="claude")
```

One write-capable, streaming command; `{model}`/`{prompt}` substituted (exec
form, no shell). codex's `workspace-write` sandbox lets it write into the workdir
but nowhere else — the sandboxing we want, for free.

## Gotchas (handled)

- **Non-interactive** — `stdin=DEVNULL` so the CLI never waits on stdin; codex
  `exec`/claude `-p` + auto-approve flags run headless.
- **Auth** — env / logged-in CLI, per perspective (`.env` → `OPENAI_API_KEY` for
  codex; `claude setup-token` for claude). `codex doctor` verifies.
- **Timeout/retry** — per-call subprocess timeout (kill → the fleet
  retries/replans). Reasoning models are slow on hard kernels — set `--timeout`
  generously.

## Determinism & resume

Real CLIs are nondeterministic, so resume is **no-loss / no-double-pay**, not
bitwise (orchestration.md §12.1). Producing `(agent, model)` + trajectory are
journaled per candidate, so the dashboard attributes every win.

## Testability

Verified with a **fake CLI** — a script that writes the known files and emits a
codex-style JSON stream — exercising workdir → run → collect → Solution,
trajectory persistence, and token parsing with no network/auth/GPU.

## Wiring

`make_agents(cfg, specs)` maps each `Perspective` to `CliAgent(specs[p.agent],
p.model)`. `solver solve --agent codex --model gpt-5.5` runs the real engine
against codex (needs it installed + authed; StubExecutor → no GPU scoring yet).
Default `--agent sim` is the GPU-free deterministic demo.
