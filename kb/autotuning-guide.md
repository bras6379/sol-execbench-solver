# Autotuning Guide

Tune spaces, search strategy, and what tuning is actually worth.
Tags per kb/README.md. (Note: this angle had the least adversarial
verification — most items ⚠️ from practitioner/competition sources.)

## What tuning is worth (set expectations first)

- Structural changes dominate: tiling structure took Boehm's SGEMM from 12.8%
  → 68.7% of cuBLAS; exhaustive tile autotuning added only ~5–8% at the end
  ⚠️. Same conclusion from the AMD-challenge winner and CUB-vs-custom results
  (see [optimization-recipe.md](optimization-recipe.md) Step 5).
- BUT an *unconfigured* kernel is catastrophic: adding a missing
  `@triton.autotune` decorator alone gave 95× over a default-config baseline
  ⚠️. Tuning is a small multiplier on a good kernel and a huge one on an
  untuned kernel — always tune *something*, never hand-pick a single config.
- Optimal configs are GPU-specific and shape-specific ⚠️ (multiple sources);
  configs cannot be derived analytically — they must be measured.

## The tune space

| Axis | Typical values | Notes |
|---|---|---|
| BLOCK_M/N/K | 64–256 / 64–256 / 32–256 | B200: M,N ≥ 128 for TC peak ✅; bK spans 4 UMMA atoms for block-scaled (128 B → bK 128 for 8-bit, 256 for 4-bit) ⚠️ |
| num_stages | 2–6 | SMEM capacity bound (228 KB ✅) |
| num_warps | 4–16 | Triton; 8 was optimal in the Gluon tcgen05 tutorial ⚠️ |
| cluster size / CTA pair | 1, 2, (4) | 2-SM UMMA only M∈{128,256} ✅; clusters shrink schedulable SMs (4→132, 8→120) ⚠️ |
| swizzle / smem layout | arch-specific | bank-conflict padding (e.g. 128→132) ⚠️ |
| grid strategy | data-parallel vs persistent vs Stream-K | CUTLASS Heuristic mode picks reliably ⚠️ |
| epilogue variant | fused vs separate | measure, don't assume |

## Search strategy

1. **Prune with a model, then measure few** ⚠️ (the documented B200 pipeline
   from PyTorch's CuTeDSL backend): enumerate compatible configs (hundreds
   via cutlass_api), score with NVIDIA's **nvMatmulHeuristics** analytical
   model (tile efficiency, bandwidth, occupancy), keep ~5, compile and
   benchmark only those.
2. Small spaces (≤ ~400 configs, e.g. Triton tile axes): exhaustive sweep is
   fine — it's a one-time cost per problem since **benchmark shapes are
   fixed** (each problem's `workload.jsonl` enumerates its shapes).
3. Per-shape specialization is where the headroom is ⚠️: CuTeDSL backend won
   up to 1.73× at M=8–64 but only parity at large shapes; tune per workload
   shape and dispatch on shape. (Competitions score geomean across shapes
   precisely to punish single-shape overfit ⚠️ — check how SOL-ExecBench
   aggregates its workloads before over-specializing one.)
4. JIT-specialize when possible ⚠️: cuda.compute/CUB's wins came from
   JIT-compiled shape-specialized kernels + LTO; DeepGEMM is JIT-everything
   (NVRTC). For fixed benchmark shapes, compile-time constants beat runtime
   parameters (enables unrolling ⚠️).

## Persist everything

⚠️ Two-level caching (from the Inductor/CuTeDSL practice): store (a) the
compiled kernel artifact, (b) the winning config per (problem, shape) —
skip both recompilation and re-benchmarking on re-runs. For our solver:
commit tune results per problem into the repo (e.g.
`problems/<N>/tune_cache.json`) so iterations and future agents inherit them.
Pin shape hints when shapes are dynamic — tuning at the wrong shape mistunes
the kernel ⚠️.

## Tuning discipline

- **`@triton.autotune`'s `key=` must include every axis that varies across
  this problem's workloads, not just the one that changes the compiled
  kernel's shape.** Measured bug: a kernel keyed `key=["S"]` alone silently
  let workloads that share an `S` but differ in `B` (e.g. S=256 used by
  B∈{1,8,16,32}) reuse ONE tuned config — forcing a small, launch-bound shape
  and a large, bandwidth-bound shape sharing that `S` to share a config tuned
  for only one of them. Fixing the key to `["B","S"]` (all axes present in
  `workload.jsonl`) let every workload tune independently; the win was
  substantial on the previously-mistuned shapes. Autotune candidates run
  during the grader's untimed warmup, so widening the key/config space is
  free at scoring time — there's no reason not to key on every axis that can
  plausibly matter.
- Tune AFTER the structure is right (right algorithm/fusion/pipeline), not
  before — retune after any structural change.
- One benchmark job at a time on the pod (contention wrecks measurements —
  see [benchmarking-discipline.md](benchmarking-discipline.md)).
- Keep the tune metric identical to the scored metric (median latency under
  harness conditions), not ncu-reported time ✅.
- Verify the winning config passes correctness at ALL workload shapes, not
  just the tuned one — tile-size edge cases (partial tiles, masking) are the
  classic autotune-then-fail bug.
- Log every measured config, not just the winner — the loss surface tells
  the next agent which axes matter for this problem family.
