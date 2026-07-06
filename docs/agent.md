# Agent backends: shell out to existing coding-agent CLIs

**Status: `CliAgent` adapter built (laptop-verified with a fake CLI); real
`codex`/`claude` drop in behind a spec once installed + authed.**

We don't implement an agent — we **abstract existing agent CLIs** behind the
engine's `Agent` interface (docs/orchestration.md §2). Every CLI already has a
non-interactive mode; our adapter drives it as a subprocess, so the agent's
implementation language is irrelevant and sandboxing is just "run the subprocess
in a container" (Phase F).

```
Perspective(agent="codex", model="gpt-5.5")   →   CliAgent(CODEX_spec, "gpt-5.5")
Perspective(agent="claude", model="opus")     →   CliAgent(CLAUDE_spec, "opus")
```

## One class, CLIs are data

`CliAgent` implements `design/plan/reflect/judge`. Each CLI is a `CliSpec` — two
command templates (`{model}`/`{prompt}` substituted, exec-form so no shell
injection) and a glob for the files it writes:

```python
CODEX  = CliSpec("codex",   # from `codex exec --help`: -m model, -s sandbox, positional prompt
    edit=["codex","exec","-m","{model}","-s","workspace-write","--skip-git-repo-check","{prompt}"],
    ask =["codex","exec","-m","{model}","-s","read-only","--skip-git-repo-check","{prompt}"])
CLAUDE = CliSpec("claude",
    edit=["claude","-p","{prompt}","--model","{model}","--dangerously-skip-permissions"],
    ask =["claude","-p","{prompt}","--model","{model}"])
```

Adding an agent = add a `CliSpec`. Zero new code. (`codex exec`'s
`workspace-write` sandbox lets it write the kernel into the workdir but nowhere
else — the sandboxing we want, for free. Verify each CLI's exact flags with its
own `--help`; the `claude` spec above is illustrative.)

**Failures are loud, not silent.** If a `plan` run produces no kernel file (bad
model id, auth failure, non-zero exit), `CliAgent.plan` raises with the exit
code + stderr → the fleet journals a `solver_error` for that problem (the rest
keep running) instead of accepting an empty baseline candidate. The default
`check` gate also rejects an empty Solution as a backstop.

## The workdir contract (the whole integration)

```
plan / design  →  _run(spec.edit) in a per-candidate workdir seeded with:
                    DESIGN.md   (ctx.design)            } the §8 context,
                    CONTEXT.md  (parent reflection,     } as files the agent
                                 frontier capsules,     } reads
                                 top-K insights)        }
                    <parent kernel files>  (the starting point)
                  the prompt tells it to write kernel.<ext>
                  →  collect kernel.*  →  Solution{spec.languages, sources}  →  check() gate
reflect / judge →  _run(spec.ask)  →  read stdout  (judge parses "materially-new"/"cosmetic")
```

The engine stays the loop: it seeds the workdir with curated context (never raw
journals), the `check` gate validates the produced Solution, and reflection/
frontier/escalation are the engine's, not the CLI's. So an `Agent` call is one
targeted generation, not a second autonomous loop.

## Four gotchas (handled)

1. **Non-interactive/permissions** — each CLI needs its auto-approve flag
   (`codex --full-auto`, `claude --dangerously-skip-permissions`); safe because
   Phase F runs the subprocess sandboxed.
2. **Output collection** — two channels: files-in-workdir (kernels, `plan`/
   `design`) vs stdout (text, `reflect`/`judge`). The prompt names the file to
   write; the spec globs it back.
3. **Auth** — env / logged-in CLI (`codex login` or `OPENAI_API_KEY`;
   `claude setup-token`), per perspective; passed through `env`.
4. **Timeout/retry** — subprocess timeout (kill on expiry → the engine
   retries/replans); a run that writes nothing valid → `check` rejects → replan.

## Determinism & resume

Real CLIs are nondeterministic, so resume is **no-loss / no-double-pay**, not
bitwise (docs/orchestration.md §12.1). The producing `(agent, model)` is
journaled per candidate, so the dashboard still attributes every win.

## Testability (laptop, no real CLI)

The adapter is verified against a **fake CLI** — a tiny script that writes a
`kernel.*` (edit) or echoes a verdict (ask). This exercises workdir setup → run
→ collect → Solution → `check` deterministically, no network/auth/GPU. Real
`codex`/`claude` slot behind the identical spec. Kernel *quality* is only
scorable once the real GPU executor lands (Phase F); until then a real-agent run
on `StubExecutor` verifies plumbing (candidates score at the baseline).

## Wiring

`make_agents(cfg, specs)` maps each `Perspective` to `CliAgent(specs[p.agent],
p.model)`. `solver solve --agent codex --model gpt-5.5` runs the real engine
against a codex/GPT-5.5 tier (needs `codex` installed + authed); the default
`--agent sim` keeps the GPU-free deterministic demo.
