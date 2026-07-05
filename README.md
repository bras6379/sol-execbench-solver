# SOL-ExecBench Solver

Engine for solving the [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench)
GPU-kernel benchmark on B200. Runs locally (laptop); no pod/GPU required for
the current stage.

- `kb/` — B200 / Blackwell kernel-engineering knowledge base (21 files,
  built from four fact-checked deep-research passes). Start at `kb/README.md`.
- `.claude/skills/design-kernel/` — recipe for turning a problem's PyTorch
  reference into optimized kernel candidate designs.
- `designs/` — per-problem design docs.
- `solver/` — the engine (Python package).
- `problems/<N>/` — fetched problem packs (see below).

## Setup

```bash
uv venv --python 3.12
uv pip install -e .          # laptop base: just the fetcher + scoring (pyarrow)
```

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
