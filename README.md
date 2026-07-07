# SOL-ExecBench Solver

An autonomous solver for the [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench)
GPU-kernel benchmark on NVIDIA **B200**. Given a problem's PyTorch reference, it
drives coding agents (Claude, GPT-5.5) to write optimized kernels, evaluates each
on a **real B200** through the official harness, keeps a per-shape **ε-Pareto
frontier**, and can submit the best straight to the leaderboard — end to end,
one command.

```bash
solver solve --gpu 1-10 --tier main=claude/opus,codex/gpt-5.5 --time-limit-min 120
solver submit 8 --poll          # → real leaderboard SOL + our rank + the #1
solver report --watch 15        # live dashboard at out/index.html
```

**Validated live on a rented B200.** A naive reference rmsnorm scores ~0.05; agent
kernels reach 0.5–0.87. Submissions are accepted by the real leaderboard (e.g.
task-4 attention-backward at SOL 0.66, task-8 reduction ~0.84 est).

## Layout

- `solver/` — the Python package:
  - `engine/` — the optimization engine: the async `solve_problem` loop + fleet,
    the ε-Pareto `frontier`, `context` (journal replay/resume), the `--tier`
    ladders, the real `SshExecutor` + pod lifecycle (`ssh_exec.py`, `pod.py`,
    `gpu_run.py`), the harness bridge (`harness.py`), durable `store`, and the
    cross-problem `knowledge` transfer.
  - `bench/` — problem fetching, the candidate Solution model/validation, and the
    `leaderboard` client (submit/poll/board).
  - `dashboard/` — the static, live-updating performance site (`solver report`).
  - `scoring.py` (the vendored SOL formula + the leaderboard calibration factor),
    `journal.py`, `cli.py` (the entry point).
- `kb/` — hand-curated B200/Blackwell kernel-engineering knowledge base (21 files);
  **fed into every agent workdir** so agents consult the recipes. Start at `kb/README.md`.
- `docs/` — `orchestration.md` (engine design), `gpu-execution.md` (the GPU path +
  the measured calibration table), `agent.md`.
- `problems/<N>/` — fetched problem packs.

## Setup

```bash
uv venv --python 3.12
uv pip install -e '.[dev,gpu]'      # engine + tests + the runpod SDK (auto-provisioning)
cp .env.example .env                # then fill in the keys below
```

`.env` (gitignored): `OPENAI_API_KEY` (codex/gpt-5.5), `RUNPOD_API_KEY` (rent B200s),
`SOLBENCH_TOKEN` (leaderboard submit/poll). The `claude` agent uses your local
`claude` CLI auth. The pinned harness itself installs on the pod only.

## The pipeline

```
  laptop (orchestrator, authoritative)          RunPod B200 (ephemeral, per run)
  ------------------------------------          -------------------------------
  solve --gpu  ── auto-provision ────────────▶  create pod, bootstrap the pinned
                                                 harness (uv sync torch/cu13 + …)
  agents write kernel.py (Claude/GPT) ──────▶   run in the SOL-ExecBench harness
  ε-Pareto frontier · score · playbook ◀────    (correctness + cold-L2 CUPTI timing)
  … iterate until the time budget …
  terminate the pod (guaranteed teardown)  ──▶  pod destroyed, nothing left billing
  submit best → leaderboard
```

The laptop is authoritative (scoring, the frontier, durable artifacts); the pod
only holds the harness and is torn down on any exit path (finally + atexit +
signals + reap of stragglers).

## GPU execution — `solver solve --gpu`

One command provisions an ephemeral B200 on RunPod, bootstraps the pinned harness,
runs the fleet, and terminates the pod:

```bash
solver solve --gpu 1-10 \
  --tier main=claude/opus,codex/gpt-5.5 \   # models round-robin within a tier (or cheap→strong ladders)
  --time-limit-min 120 \                    # wall-clock budget PER problem (lifts iter/eval caps)
  --gpu-iterations 50 \                      # harness timed iters/workload
  --verify-runs 3                            # re-run a would-be frontier entry 3× → reject flaky kernels
```

- **Resumes from journals** — re-running the same ids continues where it left off,
  keeping every frontier (kill the process to pause).
- **`--tier NAME=agent/model[,agent/model]`** (repeatable) — one tier round-robins its
  models; multiple tiers escalate cheap→strong on plateau.
- **`--verify-runs N`** (default 1 = off) — the harness checks correctness for 10 rounds,
  but a rare non-deterministic (racy) kernel can pass locally yet fail the leaderboard's
  single run. With `N>1`, a candidate that would *enter the frontier* is re-run `N−1` more
  times (fresh evals, same grader config = `10N` correctness rounds); if any run disagrees
  it's marked **flaky** and rejected. Targeted (only frontier-entering candidates pay) and
  fully local — the leaderboard is throttled, so we don't lean on it for correctness.
- **Calibration.** RunPod containers can't lock GPU clocks, so we measure *unlocked*
  (boost) — a constant **~1.19×** faster than the leaderboard's locked 1500 MHz
  (measured across 6 submissions; `docs/gpu-execution.md` §8). Every score therefore
  carries a **leaderboard estimate** (`scoring.LEADERBOARD_LATENCY_FACTOR`, env
  `SOLBENCH_CALIBRATION_FACTOR`) shown to the agent and the dashboard. A constant
  factor preserves ranking; the leaderboard is the authoritative gate.

## Engine

`docs/orchestration.md` is the full design: a GEPA-style loop per problem
(mutate the kernel → evaluate → ε-Pareto accept), a fleet over one single-flight GPU,
fully resumable via journal replay. What the agents get each iteration: the problem
reference + `kb/`, the current **frontier** (best-first, with the leaderboard estimate),
the **playbook** of reserve plays (higher-ceiling ideas prior accepted kernels flagged
but didn't ship — `runs/<task>/playbook.md`), **recent FAILED attempts** (so they don't
repeat mistakes), and — for same-op siblings — a **warm-start kernel** to adapt
(cross-problem transfer, `knowledge.py`). Each plan writes a `handoff.md` naming its own
reserve play, which is banked into that playbook on accept — so forward-looking reasoning
accumulates across rounds instead of dying in the trajectory.

Agents are existing coding-agent CLIs, shelled out to — no per-agent code (`CliAgent`,
`docs/agent.md`): `claude` and `codex` today. A timed-out or failed agent call *skips
the iteration*, it never aborts the problem.

**Laptop-first + tested.** The whole loop runs against a `StubExecutor`/`StubAgent`
(no GPU, no API), so every routing / frontier / budget / escalation / resume invariant
is deterministically asserted:

```bash
python -m pytest -q          # 55 tests, no GPU / no API
```

## Leaderboard

The loop is closed in-tool (`SOLBENCH_TOKEN` required; `kernel_id == task_id`):

```bash
solver submit 8 --poll       # upload runs/8/best_solution.json, wait for the score
solver poll 8                # our SOL + our rank + the current #1
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

Problems are sorted by leaderboard estimate; the table also carries each problem's
**real SOL** and **leaderboard rank** (`#4 of 16`) from any actual submission. Each
problem's detail page shows every candidate's **agent · score · kernel · trajectory**
and the real leaderboard submissions.

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
bootstrap → real harness scoring → guaranteed teardown → leaderboard submission.
Deferred: network-volume harness caching (fresh `uv sync` each run, ~5 min), ncu
deep-profiling, and same-op transfer only helps runs with sibling shapes (the 2xx
FlashInfer groups / full 235). 55 tests pass on the laptop.
