# Solution Format & Build/Run Pipeline (Ground Truth)

How a solution is *submitted* and how the harness *builds and runs* it — read
from `NVIDIA/SOL-ExecBench` source (`core/data/solution.py`, `driver/
templates/{eval_driver,build_ext}.py`, `scripts/run_dataset.py`,
`examples/`, `docs/{solution,trace}.md`) on 2026-07-05. All ✅ (from source).
Companion to [benchmark-grader.md](benchmark-grader.md) (scoring/timing/
correctness) — this file is the *artifact + build/run* half.

## The Solution artifact

JSON: `{name, definition, author, spec, sources, description?}`. Immutable;
carries a **content hash** (`Solution.hash()` — SHA1 over name, definition,
languages, entry_point, binding, dependencies, and every source path+content).
**Use that hash for dedup and build-artifact caching** — don't invent our own.

`spec` (BuildSpec):
- `languages: [..]` — from the 9 below; **C++ and Python cannot be mixed**.
- `target_hardware: [..]` — `B200` and/or `LOCAL` (min 1).
- `entry_point: "file::func"` — the function the driver calls; the file must
  be in `sources`; its suffix must match the language family.
- `dependencies: [..]` — e.g. `cublas`, `cudnn`, `cutlass`, `triton >= 2.3`.
- `destination_passing_style: bool = true` — **per-solution**, not global.
- `binding: torch | null` — for C++/CUDA (defaults to `torch`); ignored for py.
- `compile_options: {cflags:[], cuda_cflags:[-O3,--use_fast_math], ld_flags:[-lcuda]}`
  — C++ only; passed to `torch.utils.cpp_extension.load`.

`sources`: **list** of `{path, content}`, ≥1, unique *relative* paths (no
absolute, no `..`), entry file must be present. **Multi-file is normal** — the
real cuda_cpp rmsnorm example is `kernel.h` + `kernel.cu` + `main.cpp` with
`entry_point: main.cpp::run`.

## The 9 languages — two families

| Family | entry suffix | languages | how it runs |
|---|---|---|---|
| **Python** | `.py` | pytorch, triton, cute_dsl, cutile, cudnn_frontend | imported as a module; JIT DSLs compile on the warmup call |
| **C++** | `.cu/.cpp/.cc/.cxx/.c/.h/.hpp/.cuh` | cuda_cpp, cutlass, cudnn, cublas | compiled to a `.so` via torch cpp_extension (binding=torch) |

Validation (real pydantic model) enforces: no py+C++ mix; entry suffix matches
the family. **I originally designed Python-only — wrong; all 9 are valid
submissions.**

## DPS (destination passing style) — per solution

- `true` (default): evaluator pre-allocates outputs, passes them as the **last
  positional args**; `run` writes in-place, returns nothing.
- `false`: `run` **returns** the outputs (the real cuda_cpp example uses this).
- Arg order = `Definition.inputs` keys, then (if DPS) `Definition.outputs`
  keys. The driver branches on the flag (`_call_and_collect_outputs`).

## Build → eval pipeline (how it actually runs)

Local, one command: `sol-execbench <problem_dir> --solution sol.json
[--config cfg.json] [--compile-timeout S] [--timeout S] [--output trace.jsonl]`.
Hosted benchmark: **two servers** — a compile server, then a GPU eval server.

1. **Stage** sources into a temp dir (per `path`).
2. **Compile** (C++ only, `build_ext.py`): `cpp_extension.load(sources,
   extra_cuda_cflags, extra_cflags, extra_ldflags)` → `.so`. Failable
   (`COMPILE_ERROR`), bounded by `--compile-timeout`. Python: no compile.
   **`cpp_extension.load/load_inline` is BLOCKED inside the eval server** —
   compilation is a *separate phase*; artifacts are cached by `Solution.hash()`.
3. **Eval subprocess** (`eval_driver.py`, bounded by `--timeout`): load the
   entry (compiled `.so` for C++, imported module for Python), then per
   workload — generate **seeded** inputs (random-heuristic / scalar /
   safetensors / custom), pre-allocate outputs if DPS, **warmup 10**, **time
   50 iters** (CUPTI, cold-L2), validate vs the reference (tolerance +
   matched-ratio), run reward-hack checks.
4. **Emit a `Trace`** per workload.

Config (`bench_config.example.json`): `warmup_runs=10, iterations=50, seed=200,
lock_clocks, compile_timeout, timeout`.

## Trace — the result we consume

One `Trace` per (solution, definition, **workload**); a full evaluation = the
set of Traces over the problem's ~16 workloads. Fields: `definition`,
`solution`, `workload`, `evaluation`. The **status enum** (this is the real
taxonomy — mirror it):

```
PASSED | INCORRECT_SHAPE | INCORRECT_NUMERICAL | INCORRECT_DTYPE
      | RUNTIME_ERROR | COMPILE_ERROR | TIMEOUT | REWARD_HACK | INVALID_REFERENCE
```

`evaluation` also carries the `log` (stdout/stderr), latency stats, and the
correctness summary. `sol_score` is computed from the workload latency vs its
baseline `T_b` and SOL `T_SOL` (see [benchmark-grader.md](benchmark-grader.md)).

## Implications for our engine

- **Candidate = Solution**: multi-file, any of the 9 languages, DPS optional.
  The seed is the reference-as-solution (matches the harness's own
  `build_reference_solution`); generated candidates are free-form.
- **Execute node (real, Phase F) = drive `sol-execbench` / eval_driver** with
  our solution + problem + config, then parse the emitted `Trace` JSONL into
  `EvalResult`. Compile is a real sub-stage; `COMPILE_ERROR` and `TIMEOUT` are
  first-class outcomes, and `EvalResult.status` should mirror the Trace enum
  above.
- **Dedup / build cache by `Solution.hash()`** (harness-provided).
- **`solver/check.py` is a subset** of the real pydantic `Solution` validation
  (which runs in the harness) — our static gate catches the cheap failures
  pre-GPU; the harness is authoritative.
- When the `.[bench]` extra is installed (GPU box), import
  `sol_execbench.core.Solution` for real validation + hashing instead of our
  lightweight mirror.
