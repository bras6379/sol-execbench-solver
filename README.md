# SOL-ExecBench Solver

Engine for solving the [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench)
GPU-kernel benchmark on B200. Runs locally (laptop); no pod/GPU required for
the current stage.

- `kb/` — B200 / Blackwell kernel-engineering knowledge base (21 files,
  built from four fact-checked deep-research passes). Start at `kb/README.md`.
- `.claude/skills/design-kernel/` — recipe for turning a problem's PyTorch
  reference into optimized kernel candidate designs.
- `designs/` — per-problem design docs.
- `solver/` — the Python package, organized by concern:
  - `engine/` — the optimization engine (loop, frontier, tiers, resume).
  - `bench/` — problem fetching + candidate Solution model/validation.
  - `dashboard/` — the static performance site (`solver report`).
  - `journal.py` / `scoring.py` — shared primitives; `cli.py` — the entry point.
- `docs/orchestration.md` — the orchestrator engine design (see **Engine** below).
- `problems/<N>/` — fetched problem packs (see below).

## Setup

```bash
uv venv --python 3.12
uv pip install -e .          # laptop base: just the fetcher + scoring (pyarrow)
```

## Execution model

The laptop **never runs a workload**. Local tooling prepares candidates and,
later, ingests results; the actual run (correctness + timing) happens on a
GPU via the official harness, which returns data we then handle.

```
  laptop                                   GPU box
  ------                                   -------
  fetch problems         ─────────────▶
  design (design-kernel skill)
  scaffold candidate  ──┐
  check (static, no GPU)│
                        └── solution ────▶  run in SOL-ExecBench harness
                                            (correctness + cold-L2 timing)
  handle results / score  ◀────────────    returns latencies + correctness
  iterate
```

The GPU-run transport (submit solution → run harness → return data) is a
later build phase; everything below runs on the laptop today.

## Engine (orchestrator)

`docs/orchestration.md` is the full design for the optimization engine: a
GEPA-style reflective loop per problem (reflect on the harness trace → mutate
the kernel → evaluate → keep a per-shape **ε-Pareto frontier**), a fleet of
these running concurrently over one single-flight GPU, with knowledge
transferred across problems (sibling seeding + curated family learnings).
Diverse `(agent, model)` **tiers** (Claude / GPT / DeepSeek / GLM / … pools)
round-robin for exploration and escalate cheap→strong when a problem plateaus
with headroom left. It runs autonomously and is fully resumable (journal
replay).

**Laptop-first:** the loop is built and tested entirely against a StubExecutor
(no GPU) and StubAgent (no API / credits) — the §12 stub contract makes every
routing / frontier / budget / escalation / resume invariant deterministically
assertable. Real agents and GPU transport are later phases.

```bash
uv pip install -e '.[dev]'   # pytest
python -m pytest tests/      # the §12 acceptance tests (no GPU, no API)
```

Drive real runs through the engine (stub sim agent, no GPU/model) and view them:

```bash
solver solve 1-40 --runs-dir runs                 # deterministic sim agent (no GPU/model)
solver solve 1-40 --agent codex --model gpt-5.5   # a real agent CLI (needs it installed + authed)
solver status --runs-dir runs                     # per-problem summary
solver report --runs-dir runs                     # → out/ dashboard site
```

Status: **Phases A–E built** — the engine (`solver/engine/`): the async
`solve_problem` loop, RunContext with journal replay/resume, the ε-Pareto
frontier, the tier ladder with headroom-gated escalation, novelty gates, the
fleet, the serialized knowledge curator, and bootstrap sibling seeding — all
covered by the §12 stub tests and verified end-to-end on the dashboard
(`docs/screenshots/real-*.png`). **Real agents** plug in by shelling out to
existing coding-agent CLIs (`CliAgent`; `docs/agent.md`) — a `CliSpec` per CLI,
no per-agent code. Next: **GPU execution** (`docs/gpu-execution.md`, designed) —
a `GpuQueueExecutor` that runs kernels in the harness on a rented B200 over SSH;
Phase F1 (transport + durability) is testable on-laptop with a fake harness
before renting a GPU.

Run progress is inspectable as a static dashboard (no server/CDN):

```bash
solver report --demo      # synthetic 235-problem run -> .cache/demo/out/index.html
solver report             # a real run under runs/ -> publishable out/ site
```

## Candidates

A **candidate** is a *Solution* (the harness's JSON: build `spec` + `sources`,
called in Destination Passing Style — inputs then pre-allocated outputs,
written in place). See `docs/solution.md` in the upstream repo; the format is
captured in `solver/bench/solution.py`.

```bash
solver scaffold 69                 # correct PyTorch DPS baseline for task 69
solver scaffold 69 --lang triton   # signature-correct Triton stub to build from
solver check <solution.json>       # static pre-flight (schema, DPS signature, reward-hack lints)
```

`scaffold` writes to `problems/<id>/candidates/<name>.json` (git-ignored;
regenerable). `check` runs no GPU and executes no candidate code — it only
AST-parses — so it's the cheap gate before spending a GPU run.

## Scoring and the grader

The score is `S = 1 / (1 + (T_k − T_SOL) / (T_b − T_SOL))` — 0.5 when a
candidate matches the baseline `T_b`, 1.0 at the Speed-of-Light `T_SOL`. Both
numbers are in each problem's `metadata.json` (`sol.per_workload`). The
formula is vendored (`solver/scoring.py`) so you can project scores during
design without a GPU:

```python
from solver.scoring import sol_score, score_from_metadata
```

The **real grader** (correctness, cold-L2 CUPTI timing, input generation,
reward-hack defenses — see `kb/benchmark-grader.md`) is the official
`NVIDIA/SOL-ExecBench` harness. It runs on a **Linux + CUDA-13 + Blackwell
GPU** box, not the laptop, and is wired as a pinned optional dependency:

```bash
uv pip install -e '.[bench]'   # GPU box only; pulls torch(cu130), cupti, cutlass-dsl, ...
```

Driving that harness on the GPU (candidate → measured latency → score) is the
next build phase; nothing in the current laptop flow requires it.

## Fetching problems

Problems are addressed by their **global task number 1–235**, which maps
onto the four benchmark subsets (L1 1–94, L2 95–176, Quant 177–209,
FlashInfer-Bench 210–235).

```bash
solver fetch 69            # one problem
solver fetch 1 2 5-10      # lists and ranges
solver fetch --all         # all 235
solver fetch 69 --refresh  # re-download, ignore cache
solver fetch 67 --no-sol   # skip the website SOL-baseline enrichment
solver list --all          # print task-id -> subset / name
```

Each `solver fetch` writes `problems/<N>/`:

| File | Source | Contents |
|---|---|---|
| `definition.json` | dataset | name, hf_id, description, axes, inputs/outputs, reference |
| `reference.py` | dataset | PyTorch reference + input generator (verbatim) |
| `workload.jsonl` | dataset | one workload per line: axes, inputs, **tolerance** |
| `metadata.json` | + website API | provenance + Speed-of-Light baselines (`sol_ms`) |

**Data source.** The authoritative problem data is the
[`nvidia/SOL-ExecBench`](https://huggingface.co/datasets/nvidia/SOL-ExecBench)
HuggingFace dataset (four parquet files, cached under `.cache/`). The
`workload.jsonl` we write is byte-identical to the official unpack. The
public website API (`/api/kernels/<id>`) is used only to enrich
`metadata.json` with per-workload SOL targets and the reference baseline
latency — it does **not** carry tolerances, so it is enrichment, not the
source of truth.

Note: 18 of the 26 FlashInfer-Bench problems (rmsnorm/gemm/moe: ids 210–220,
229–235) carry no `tolerance` field; the 8 paged/ragged GQA/MLA attention
problems (221–228) do. `metadata.json` records this per problem as
`workloads_have_tolerance`. All L1/L2/Quant problems have tolerance.
