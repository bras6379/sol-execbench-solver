# SOL-ExecBench Solver

An autonomous solver for the [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench)
GPU-kernel benchmark on NVIDIA **B200**. Given a problem's PyTorch reference, it
drives coding agents (Claude, GPT-5.5, or any OpenRouter-routed model) to write
optimized kernels, code-reviews each candidate BEFORE spending a GPU eval,
evaluates the survivors on a **real B200** through the official harness, keeps a
per-shape **ε-Pareto frontier**, tracks real $ spend per problem/model, and can
submit the best straight to the leaderboard — end to end, one command.

```bash
solver solve --gpu 1-10 --tier main=claude/opus,codex/gpt-5.5 --time-limit-min 120
solver submit 8 --poll          # → real leaderboard SOL + our rank + the #1
solver report --watch 15        # live dashboard at out/index.html
```

**Validated live on a rented B200, with real leaderboard submissions accepted:**
task-18 reduction at SOL 0.868 (rank #5/8), task-12 fused-embedding at SOL 0.892
(rank #10/18), task-4 attention-backward at SOL 0.67 (rank #4/16). A naive
reference kernel typically scores ~0.02–0.20; agent kernels land 0.4–0.9.

## Layout

- `solver/` — the Python package:
  - `engine/` — the optimization engine: the async `solve_problem` loop + fleet
    (`loop.py`), the ε-Pareto `frontier`, `context` (journal replay/resume,
    `RunContext`), the `--tier` ladders (`config.py`), the coding-agent adapter
    (`cli_agent.py`, `agent.py`), cross-run reflection + expert diagnosis
    (`reflection.py`, `diagnose.py`), cross-problem `knowledge` transfer, the
    real GPU transport (`ssh_exec.py`, `pod.py`, `gpu_run.py`), the harness
    bridge (`harness.py`), and durable `store`.
  - `bench/` — problem fetching, the candidate Solution model/validation, and the
    `leaderboard` client (submit/poll/board).
  - `dashboard/` — the static, live-updating performance site (`solver report`),
    including a cost/efficiency panel (real $ by call-type and model, no-op
    waste, reflection health).
  - `scoring.py` (the vendored SOL formula + the leaderboard calibration factor),
    `journal.py`, `cli.py` (the entry point).
- `kb/` — hand-curated B200/Blackwell kernel-engineering knowledge base (24 files
  — roofline method, autotuning, per-architecture hardware notes, fusion
  patterns, benchmarking discipline, reward-hack lints); **fed into every agent
  workdir** (writer AND reviewer) so agents consult the recipes instead of
  re-deriving them. Start at `kb/README.md`.
- `knowledge/` — cross-*problem* transfer, populated automatically as runs
  complete: `best/<op>.json` (the winning Solution per operator family, loaded
  on startup so transfer survives across separate `solve` invocations),
  `families/<op>.md` + `global.md` (human-readable summaries of what worked).
- `docs/` — `orchestration.md` (engine design), `gpu-execution.md` (the GPU path
  + the measured calibration table), `agent.md` (the `CliAgent` contract),
  `oncall-runbook.md` (operational playbook: health checks, safe restarts,
  live incidents and their diagnoses/fixes).
- `problems/<N>/` — fetched problem packs.

## Setup

```bash
uv venv --python 3.12
uv pip install -e '.[dev,gpu]'      # engine + tests + the runpod SDK (auto-provisioning)
cp .env.example .env                # then fill in the keys below
```

`.env` (gitignored): `OPENAI_API_KEY` (codex/gpt-5.5), `RUNPOD_API_KEY` (rent B200s),
`SOLBENCH_TOKEN` (leaderboard submit/poll), optionally `OPENROUTER_API_KEY` /
`ZAI_API_KEY` / `DEEPSEEK_API_KEY` / `MOONSHOT_API_KEY` for cheaper model pools.
The `claude` agent uses your local `claude` CLI auth. The pinned harness itself
installs on the pod only.

## How it works

Each problem runs its own instance of a GEPA-style loop — mutate the kernel,
measure it for real, keep only genuine improvements — until a budget, a
plateau, or a "this is at ceiling" signal stops it:

```
                    ┌──────────────────────────────┐
                    │ design(task) — ONE-SHOT      │
                    │ a strong model reads the     │
                    │ reference + kb/, writes a    │
                    │ ranked, roofline-backed plan │
                    └──────────────────────────────┘
                                              │
                    ┌─────────────────────────────────┐
                    │ seed the frontier: eval the raw │
                    │ PyTorch reference on the GPU    │
                    └─────────────────────────────────┘
                                               │
  ┌───────────────────────────────────────────────────────────────────┐
  │  per-iteration loop                                               │
  │                                                                   │
  │  1. pick a parent from the ε-Pareto frontier (weighted by         │
  │     shapes-won — partial specialists get picked too)              │
  │  2. agent.plan(parent, ctx) → a Candidate (the writer sees the    │
  │     full CONTEXT.md — assembled below)                            │
  │                                                                   │
  │  same hash as its own parent? ──yes──▶ NO-OP (streak++; N in a    │
  │    │no                                  row ⇒ ceiling_consensus:  │
  │    ▼                                     this problem is at       │
  │  static check_fn                         ceiling — stop)          │
  │  (schema / DPS-signature / reward-hack lints, no GPU)             │
  │    │ok                                                            │
  │    ▼                                                              │
  │  seen this exact hash before? ──yes──▶ duplicate (skip)           │
  │    │no                                                            │
  │    ▼                                                              │
  │          ┌────────────────────────────────────────────┐           │
  │          │ PRE-GPU REVIEW — a DIFFERENT model reads   │           │
  │          │ the kernel against reference.py +          │           │
  │          │ workloads.md and hand-traces the trickiest │           │
  │          │ graded shape.                              │           │
  │          │                                            │           │
  │          │ "SHIP"   ───────────────────────▶ GPU      │           │
  │          │ "REVISE" ─▶ the SAME writer repairs with   │           │
  │          │    the critique, loop (≤ review_max_       │           │
  │          │    rounds, then ships as-is — never        │           │
  │          │    worse than review being off)            │           │
  │          └────────────────────────────────────────────┘           │
  │    │                                                              │
  │    ▼                                                              │
  │  executor.evaluate() — the ONE single-flight GPU                  │
  │    │                                                              │
  │    ▼                                                              │
  │  correct? ──no──▶ note_failure (exact failing shapes fed back to  │
  │    │yes            the NEXT agent, so it does not repeat it)      │
  │    ▼                                                              │
  │  would this ENTER the frontier? ──yes──▶ re-verify ×N (catches    │
  │    │                                      flaky/racy kernels      │
  │    ▼                                      before they ship)       │
  │  ε-Pareto accept / dominate / reject                              │
  │    └─ entered? → bank handoff.md into playbook.md (the higher-    │
  │       ceiling idea it flagged but didn't ship — the next agent    │
  │       inherits it)                                                │
  └───────────────────────────────────────────────────────────────────┘
                                     │ loop until: budget, plateau→escalate
                                     │ (or terminate), score_target, or
                                     │ ceiling_consensus
                                     ▼
                    ┌─────────────────────────────┐
                    │ terminated: reason recorded │
                    │ (reopens on the next resume │
                    │ unless converged/target)    │
                    └─────────────────────────────┘
```

A `run_fleet` runs many problems concurrently, but the GPU itself never is:

```
run_fleet(ids, ..., max_concurrency=N)
┌──────────────────────────────────────────────────────────────────────┐
│  Semaphore(N) — only N problems hold a concurrency slot at once      │
│                                                                      │
│  problem A ─┐                                                        │
│  problem B ─┼─ each runs its OWN loop above, sequentially within     │
│  problem C ─┤   itself (≤1 agent call in flight per problem — the    │
│    ...      │   provider rate limit is the cap, not the GPU)         │
│  problem N ─┘                                                        │
│                                                                      │
│  all N funnel into ONE queue                                         │
│          │                                                           │
│          ▼                                                           │
│          ┌─────────────────────────┐                                 │
│          │ single-flight GPU queue │  ← the REAL                     │
│          │ (never two kernels at   │    throughput                   │
│          │  once — required for    │    bottleneck,                  │
│          │  clean latency measure) │    not concurrency              │
│          └─────────────────────────┘                                 │
│                                                                      │
│  problems beyond N queue and enter as a slot frees — a resumable     │
│  rolling window, so `solve 1-235 --max-concurrency 12` never spawns  │
│  235 CLIs at once. `--shuffle` randomizes launch order (seeded, so   │
│  a resume is identical) so the window samples fairly across a big    │
│  id range instead of always the lowest ids.                          │
└──────────────────────────────────────────────────────────────────────┘
```

## GPU execution — `solver solve --gpu`

One command provisions an ephemeral B200 on RunPod, bootstraps the pinned harness,
runs the fleet, and terminates the pod:

```bash
solver solve --gpu 1-10 \
  --tier main=claude/opus,codex/gpt-5.5 \   # models round-robin within a tier (or cheap→strong ladders)
  --time-limit-min 120 \                    # wall-clock budget PER problem (lifts iter/eval caps)
  --gpu-iterations 50 \                     # harness timed iters/workload
  --verify-runs 3 \                         # re-run a would-be frontier entry 3× → reject flaky kernels
  --review --review-max-rounds 6 \          # pre-GPU code review + repair loop (default: on)
  --ceiling-consensus 2                     # N consecutive no-ops → auto-stop (default: 2, 0 disables)
```

- **Resumes from journals** — re-running the same ids continues where it left off,
  keeping every frontier (kill the process to pause; `docs/oncall-runbook.md`
  §2 has the safe restart procedure — SIGTERM is handled and guarantees pod
  teardown even on a hard kill, unless `--gpu-reuse-pod` is set).
- **`--gpu-reuse-pod`** — a restart for a code/prompt fix doesn't need a new pod.
  Leaves the pod running on a SIGINT/SIGTERM instead of tearing it down; the next
  launch adopts it directly, skipping the ~5-10 min `uv sync`/bootstrap (it still
  runs, but every step no-ops in seconds on an already-warm pod, which is also
  how an updated `RUN_EVAL_SH`/config gets pushed). `--gpu-max-hours` anchors to
  the pod's own real rental time, so a chain of quick restarts can't reset the
  safety cap. Normal completion or the cap being hit still tears the pod down —
  this only changes what a *deliberate* restart does.
- **`--tier NAME=agent/model[,agent/model]`** (repeatable) — one tier shuffles its
  models per problem; multiple tiers escalate cheap→strong on plateau.
- **Cheap models, one CLI.** Claude/GPT are pricey, so any provider with an
  Anthropic-compatible endpoint runs through the same `claude` CLI binary,
  re-pointed via environment variables — no per-provider code:
  ```
      ┌───────────────────────────────────────────────────────────────────┐
      │ CliAgent(spec, model)                                             │
      │   spec.base_url unset → real Claude, your own auth                │
      │   spec.base_url set   → ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN │
      │                          injected into the subprocess env         │
      └───────────────────────────────────────────────────────────────────┘

      claude/opus, claude/sonnet, claude/haiku   ─▶ api.anthropic.com
      openrouter/<any model on OpenRouter>       ─▶ openrouter.ai (one key, every model)
      glm/<model>, deepseek/<model>, kimi/<model> ─▶ their own direct endpoints
      codex/gpt-5.5                               ─▶ a SEPARATE binary (`codex exec`),
                                                      its own auth, its own stream schema
  ```
  A GLM plan is roughly 1/6th the cost of GPT-5.5; DeepSeek less. Mix them in one pool:
  ```bash
  solver solve --gpu 1-10 --tier all=claude/opus,codex/gpt-5.5,\
    openrouter/z-ai/glm-5.2,openrouter/deepseek/deepseek-v4-pro,openrouter/moonshotai/kimi-k2.7-code
  ```
  Keys go in `.env` (`OPENROUTER_API_KEY`, or `ZAI_API_KEY`/`DEEPSEEK_API_KEY`/`MOONSHOT_API_KEY`).
  Each of these is an **independent billing account** — OpenRouter, OpenAI/codex,
  and your Claude subscription can each run out of credit/quota on its own
  schedule, with no shared cause (`docs/oncall-runbook.md` §4 has the live
  incident where all three were checked and codex's quota exhaustion turned out
  to be unrelated to OpenRouter's).
- **Load-spreading + graceful downgrade.** A pool is rotated in a **per-problem shuffle**
  (deterministic → replay-safe, but different across problems) so the fleet doesn't hit
  one provider in lockstep. An agent that keeps failing (e.g. out of credits) is
  **circuit-broken** after `--agent-fail-limit` (default 3) consecutive failures and
  skipped — the run downgrades to the healthy models automatically; if a whole tier dies
  it routes to a live one, or ends cleanly (`agents-unavailable`).
- **`--max-concurrency N`** — how many problems run at once (see the diagram
  above — this bounds concurrent agent CLIs, not GPU throughput). `0` = unbounded.
- **`--verify-runs N`** (default 1 = off) — the harness checks correctness for 10 rounds,
  but a rare non-deterministic (racy) kernel can pass locally yet fail the leaderboard's
  single run. With `N>1`, a candidate that would *enter the frontier* is re-run `N−1` more
  times (fresh evals, same grader config = `10N` correctness rounds); if any run disagrees
  it's marked **flaky** and rejected. Targeted (only frontier-entering candidates pay) and
  fully local — the leaderboard is throttled, so we don't lean on it for correctness.
- **Compile time bounded, off the GPU where it matters.** Each eval is capped at
  90s compile / 300s total (`ssh_exec.py`'s `RUN_EVAL_SH`) so one pathological
  candidate (e.g. an overambitious custom megakernel that never finishes
  compiling) can't hog the single-flight GPU for more than 5 minutes — confirmed
  live, see `docs/oncall-runbook.md` §3. A bigger, deferred idea (a second cheap
  GPU as a compile/correctness smoke-test gate, decoupled from the B200 timing
  lock) is noted as a TODO right above `RUN_EVAL_SH`, but isn't worth the extra
  always-on rental unless compile-stalls become a recurring pattern.
- **Calibration.** RunPod containers can't lock GPU clocks, so we measure *unlocked*
  (boost) — a constant **~1.19×** faster than the leaderboard's locked 1500 MHz
  (measured across 6 submissions; `docs/gpu-execution.md` §8). Every score therefore
  carries a **leaderboard estimate** (`scoring.LEADERBOARD_LATENCY_FACTOR`, env
  `SOLBENCH_CALIBRATION_FACTOR`) shown to the agent and the dashboard. A constant
  factor preserves ranking; the leaderboard is the authoritative gate.

## Engine

`docs/orchestration.md` is the full design. What an agent actually sees each
turn is assembled into one `CONTEXT.md`, in priority order:

```
   ┌───────────────────────────────────────────────────────────────────┐
   │ 1. score legend (0.5 = optimized-PyTorch baseline, 1.0 = SOL)     │
   │ 2. pre-submission review critique — only on a repair turn: fix    │
   │    EVERY issue before re-submitting                               │
   │ 3. Coach card — reflection.py's deterministic status (climbing /  │
   │    plateaued / regressing / rabbit_hole / broken) + diagnose.py's │
   │    expert prose on STUCK ones (see below)                         │
   │ 4. prior/ pointer — top earlier kernels, named by score, read ON  │
   │    DEMAND rather than force-fed                                   │
   │ 5. sibling warm-start — the best SAME-OP sibling's kernel from a  │
   │    DIFFERENT problem (`knowledge.py` cross-problem transfer),     │
   │    handed over to ADAPT, never auto-evaluated as a seed           │
   │ 6. frontier, best-first — up to 8 members (id · leaderboard-est   │
   │    (raw) · one-line strategy)                                     │
   │ 7. playbook — up to 6 most-recent reserve plays: higher-ceiling   │
   │    ideas an ACCEPTED candidate flagged but didn't ship            │
   │ 8. recent FAILED attempts — up to 4, with the EXACT failing       │
   │    shapes (so the next try doesn't repeat the same mistake)       │
   └───────────────────────────────────────────────────────────────────┘
```
Each plan writes its own `handoff.md` naming its higher-ceiling reserve play,
banked into `runs/<task>/playbook.md` only when that candidate is ACCEPTED —
so forward-looking reasoning compounds across many rounds instead of dying in
the trajectory. In practice this produces multi-generation idea chains: e.g. a
scatter-add problem's playbook threading "atomic scatter (0.844)" → "try
atomic-free sort+segmented-reduction, gated by T" → the next accepted kernel
explicitly checking whether that lever moved the frontier before proposing the
next one.

**The Coach — cross-run reflection + expert diagnosis.** Every
`--reflect-every-min` minutes (and once at startup), EVERY problem gets a fast,
free, deterministic classification from its own journal:

```
   journal.jsonl ─▶ classify: climbing / plateaued / regressing / rabbit_hole
                             / broken / thin
                    └─▶ reflection.md (fed into CONTEXT.md §3 above)
```
Problems classified **STUCK** (plateaued / regressing / rabbit_hole / broken)
additionally get a paid, background diagnosis from a strong model — gated on a
state fingerprint so it never re-spends on a problem that hasn't actually moved:
```
   ┌─────────────────────────────────────────────────────────────────┐
   │ diagnose_one(): reads the best kernel + reference + the full    │
   │ attempt ledger, in prose — the WHY it's stuck + the ONE untried │
   │ lever. Runs in the BACKGROUND (never blocks the GPU or the      │
   │ fleet). Real $ cost tracked per call.                           │
   │                                                                 │
   │ FALLBACK_CHAIN on failure (rate-limit / no credits):            │
   │   native claude (--reflect-model) ──▶ openrouter:deepseek       │
   │                                    ──▶ openrouter:kimi          │
   │ (skips the native rung entirely when --reflect-model isn't      │
   │  claude-shaped, so a fully-OpenRouter reflect-model never       │
   │  touches Claude auth at all)                                    │
   └─────────────────────────────────────────────────────────────────┘
```
The written header always names the model that actually produced a given
diagnosis (not a fixed label) — `--reflect-model` gets retuned across restarts
far more often than any hardcoded string would track.

**No-op / ceiling detection.** An agent that hands back a kernel byte-identical
to its own parent has, empirically, changed nothing ~50% of the time once a
problem nears its ceiling — every model does this at similar rates, usually
because it's correctly concluded there's nothing left to improve with no clean
way to say so. `ceiling_consensus` (default 2) auto-terminates a problem after
that many CONSECUTIVE no-ops, rather than paying for turn after turn that
produces nothing — reopenable on the next resume, unlike a genuine convergence.

**Real $ cost, tracked per problem/kind/model.** Every design/plan/review/
diagnose call's token usage and $ cost (where the CLI reports it) is journaled
and rolled up on the dashboard: total spend, spend by call-kind, spend by
model (in/out/cached tokens too — pricing differs a lot across providers), and
a **no-op waste** figure (paid calls that changed nothing). `codex/gpt-5.5`'s
own stream schema doesn't report a cost field, so its spend is currently
invisible in the $ totals — a known gap, not a bug.

Agents are existing coding-agent CLIs, shelled out to — no per-agent code
(`CliAgent`, `docs/agent.md`): `claude` and `codex` today. A timed-out or
failed agent call *skips the iteration*, it never aborts the problem; if the
underlying failure is a quota/auth error, the real message is now pulled out
of the CLI's stdout event stream (not just stderr, which is often empty on
these) so it shows up readably in the journal instead of a blank string.

**Laptop-first + tested.** The whole loop runs against a `StubExecutor`/`StubAgent`
(no GPU, no API), so every routing / frontier / budget / escalation / resume /
review / no-op invariant is deterministically asserted:

```bash
python -m pytest -q          # 104 tests, no GPU / no API
```

## Leaderboard

The loop is closed in-tool (`SOLBENCH_TOKEN` required; `kernel_id == task_id`):

```bash
solver submit 8 --poll       # upload runs/8/best_solution.json, wait for the score
solver poll 8                # our SOL + our rank + the current #1
solver poll --all            # re-poll EVERY recorded submission (catches queued ones as they resolve)
solver export                # bundle every best_solution.json → submissions/ + manifest.json
```

## Durable artifacts + dashboard

`solve` writes to `runs/<task>/`: `journal.jsonl`, `candidates/<cid>.json` (every
candidate + a ready-to-submit `submit` form), `frontier.json`, `best_solution.json`,
`playbook.md` (accumulated reserve plays), `submissions.jsonl`. The dashboard renders
it live (no server/CDN — self-contained):

```bash
solver report --runs-dir runs --out-dir out --watch 15   # regenerates out/ every 15s
python -m http.server 8765 --directory out               # view at localhost:8765
```
(Python fully buffers stdout when redirected to a log file — a freshly-launched
`solve`/`report --watch` can show an empty log for several minutes even though
it's healthy; check `runs/_active.json`'s heartbeat or the journal instead of
the log file, see `docs/oncall-runbook.md` §5.)

The top of the page is a **System diagnostics** section — fleet GPU utilization,
eval counts, and a **Cost & efficiency** panel (total spend, spend by
call-kind/model with in/out/cached tokens, no-op waste, reflection health) —
before the per-problem table. Problems are sorted by leaderboard estimate; the
table also carries each problem's **real SOL** and **leaderboard rank**
(`#4 of 16`) from any actual submission. Each problem's detail page shows every
candidate's **agent · score · kernel · trajectory**, the Coach card and its
diagnosis history, and the real leaderboard submissions.

## Candidates, scoring, fetching

A **candidate** is a *Solution* (harness JSON: build `spec` + `sources`, entry
`kernel.py::run`). Score: `S = 1 / (1 + (T_k − T_SOL) / (T_b − T_SOL))` — 0.5 at the
optimized baseline `T_b`, 1.0 at Speed-of-Light `T_SOL`; both in each problem's
`metadata.json`. The formula is vendored (`solver/scoring.py`); the **real grader**
(correctness, cold-L2 CUPTI timing, reward-hack defenses — `kb/benchmark-grader.md`)
is the pinned `NVIDIA/SOL-ExecBench` harness, installed on the pod (`.[bench]`).

```bash
solver fetch 1 2 5-10        # by global task number 1–235 (L1 1–94, L2 95–176,
solver fetch --all           #   Quant 177–209, FlashInfer-Bench 210–235)
solver check <solution.json> # static pre-flight (schema, DPS signature, reward-hack lints)
solver scaffold 69           # a signature-correct baseline to start from
```

Each `fetch` writes `problems/<N>/`: `definition.json`, `reference.py`,
`workload.jsonl` (byte-identical to the official unpack, incl. tolerances), and
`metadata.json` (SOL/baseline targets from the website API). Authoritative source:
the [`nvidia/SOL-ExecBench`](https://huggingface.co/datasets/nvidia/SOL-ExecBench)
HuggingFace dataset.

## Status

**Built and validated live on a B200** (`docs/gpu-execution.md`): auto-provision →
bootstrap → pre-GPU review gate → real harness scoring → guaranteed teardown →
leaderboard submission, with real $ cost tracked end to end. 104 tests pass on
the laptop (no GPU/API needed). `docs/oncall-runbook.md` covers day-to-day
operation: health checks, safe restarts, and diagnoses for the incidents that
have actually happened live (a stalled single-flight GPU from a runaway
compile, a provider's credits/quota running out, a forgotten `--reflect-model`
still pointed at a dead provider after the rest of the pool moved).

Deferred: network-volume harness caching (fresh `uv sync` each run, ~5 min), ncu
deep-profiling, a second cheap-GPU compile/correctness gate (only worth it if
compile-stalls recur), and same-op transfer only helps runs with sibling shapes
(the 2xx FlashInfer groups / full 235).
