# Kernel Fusion Patterns

The math and the menu of fusion strategies. Tags per kb/README.md.

## The bytes-saved arithmetic (why fusion wins) — ⚠️ [Horace He, "brrr"]

- Fusing two pointwise ops halves global traffic (4 reads/writes → 2) ≈ 2×
  speedup in the memory-bound regime.
- In that regime extra compute is ~free: fused `cos(cos(x))` ≈ cost of one
  `cos`; all activations cost the same once fused.
- Break-even: A100 could afford hundreds of FLOPs/byte of recompute; **B200's
  ratio is even more extreme** (2250e12 / 7.7e12 ≈ 292 FLOP/byte at BF16 —
  see [optimization-playbook.md](optimization-playbook.md)). A pointwise chain
  needs ~64 fused repeats before it turns compute-bound.
- Recompute-vs-save in backward: recomputing cheap activations can beat
  saving/reloading them on BOTH memory and wall-clock — bandwidth is the
  scarce resource. Directly applicable to the benchmark's 46 backward
  problems.

## Fusion strategy menu (escalating scope)

1. **Epilogue fusion** (GEMM + pointwise tail): post-process while the
   accumulator is still in registers/TMEM — saves the GMEM round-trip AND a
   kernel launch ⚠️ [Colfax EVT tutorial]. Vehicles: cuBLASLt epilogues ✅,
   CUTLASS EVT (❌ **Hopper-only as of Oct 2024 — verify SM100 status before
   relying on it**), CUTLASS collective-builder custom epilogues (the SM100
   path used by example 72b ✅), Triton (just write the tail in the kernel).
2. **Flash-style online tiling** (fuse across a softmax/normalization
   barrier): online softmax / Welford running stats let you fuse
   GEMM→softmax→GEMM or norm→matmul without materializing the intermediate.
   The pattern behind FlashAttention and every fused-attention-variant
   problem in the benchmark.
3. **Horizontal fusion** (MoE experts): batch/group many small GEMMs into one
   launch — grouped GEMM (CUTLASS 4.5 ✅, cuDNN frontend ✅, DeepGEMM ⚠️
   M-axis-grouped with contiguous/masked layouts).
4. **Persistent kernels + tile schedulers**: fixed grid ≈ SM count; each CTA
   iterates work units ⚠️ [Colfax]. CUTLASS tile-scheduler abstraction
   (`get_initial_tile` / `is_valid` / `get_next_tile`) ⚠️. **Stream-K** kills
   wave quantization by splitting along K; CUTLASS has deterministic
   (turnstile) and atomic reduction modes, and its Heuristic mode picks
   DataParallel vs Stream-K reliably — don't hand-force ⚠️.
5. **Megakernels** (fuse a whole forward pass): HazyResearch fused all of
   Llama-1B decode into one persistent kernel — <680 µs/forward on B200,
   >3.5× vLLM, >1.5× SGLang single-sequence; 78% of HBM bandwidth on H100 vs
   ~50% for serving stacks ⚠️. Built as an on-GPU interpreter: per-SM
   instruction streams, global-memory counters for cross-instruction deps,
   paged shared memory to overlap loads across instruction boundaries ⚠️.
   Code: github.com/HazyResearch/Megakernels. **This is the ceiling for the
   L2 whole-decoder-layer problems.**
- ⚠️ B200-specific note from the megakernel work: tensor cores become
  *marginally worthwhile* even for small memory-bound ops on Blackwell where
  CUDA cores were preferable on Hopper.

## When one fused kernel beats library-call + elementwise kernel

Fuse when (a) the intermediate is written to GMEM only to be re-read once,
(b) the tail op is pointwise or a row/col reduction, (c) launch overhead
matters (small tensors, decode-style shapes). Don't fuse when the library
GEMM is at 75%+ of peak and the tail is a tiny fraction of runtime — measure
the elementwise kernel's share first (profiling beats theory).

## Backward-pass fusion strategies (46 benchmark problems)

- Ride the GEMM: cuBLASLt dBias/dReLU epilogues ✅; cuDNN grouped-GEMM Wgrad ✅.
- Fused dgrad chains: chain-rule pointwise tails fused into the producing
  matmul (e.g., dGELU into the dgrad GEMM epilogue).
- Recompute-vs-save per the bytes math above; TE constraint for FP8 blockwise
  training: 1D-block-quantized tensors **cannot be transposed** — rowwise and
  columnwise versions must be quantized separately from the hi-precision
  source; 2D-scaled tensors CAN transpose with their scales ⚠️. See
  [quantized-kernels.md](quantized-kernels.md).
- wgrad reductions: atomics vs split-K/turnstile determinism — benchmark
  tolerance may allow atomic mode (faster); check `workload.jsonl` tolerances
  per problem.
- Study set: NVIDIA's open d=256 SDPA bprop CuTe-DSL kernels ✅, FlashMLA's
  SM100 dense-prefill backward (per-warpgroup register allocation 152/128/96
  for reduce/compute/MMA, TMA-reduction-add stores for dQ) ⚠️, quack's
  RMSNorm/softmax/cross-entropy bwd kernels ⚠️.

## Atomic scatter / index_add kernels (MoE combine, expert-output accumulation)

For scatter-add / `index_add_` problems (routing MoE expert outputs back to
token positions, weighted combine, etc.), the optimized-PyTorch baseline is
often **~60× slower than SOL** (measured: T=2048 baseline 0.220 ms vs sol_ms
0.0037 ms) — PyTorch's bf16 `index_add_` serializes through per-element atomic
**CAS loops** under heavy contention, achieving only ~6% of HBM bandwidth. Any
native-atomic vectorized Triton kernel beats this baseline by a wide margin
without needing to approach SOL itself (the stated `sol_ms` is often physically
unreachable, several× beyond HBM peak — treat a few× `sol_ms` as the practical
ceiling, not 1×).

This op family also tends to carry **looser tolerances** than the 1%/1% default
(e.g. `max_atol` ~0.07–0.13, `max_rtol` 0.05) — check `workload.jsonl`, don't
assume the default. Loose enough that a **bf16-direct accumulator** (the
output tensor IS the bf16 accumulator, no fp32/fp16 shadow buffer) matches a
nondeterministic bf16-atomic reference at ~100% match, so skip the shadow-
buffer complexity.

**Winning recipe:** `output = base.clone()` (mandatory — the harness reuses
inputs across timing iterations, so accumulating in place corrupts later
iterations), then a Triton kernel with `tl.atomic_add` doing a coalesced
row-load + scatter into `output[idx[row]]`. Two eviction/ordering levers, both
pure upside once measured:
- **`eviction_policy='evict_first'`** on any input tile that's read once and
  never revisited (e.g. the values being scattered) — without it, streaming a
  large once-read tensor through L2 evicts the *resident, repeatedly-atomic'd*
  output buffer via normal LRU pressure, forcing the atomics out to HBM
  (~7.7 TB/s) instead of staying in L2 (~21 TB/s on B200). Keeping the output
  buffer L2-resident only works while it's small enough to fit (check against
  your GPU's L2 size); this stops being free once the output exceeds L2.
- **`sem='relaxed'`** on the atomic add (Triton's default is `acq_rel`) — a
  commutative sum needs atomicity, not ordering, so relaxed semantics drop the
  per-atomic memory fence. Neutral at small sizes, a real win once atomic
  throughput is the bottleneck.

**Launch-bound small shapes need a different lever than bandwidth ones:** at
large token counts the kernel is bandwidth/atomic-contention-bound and already
near ceiling; at small token counts the grid is launch/scheduling-bound, not
bandwidth-bound (e.g. one measured shape needed ~1µs of bandwidth but took
~8µs). The fix that helps small shapes without regressing large ones: **one
program per output ROW, walking every column/H-tile in a fully-unrolled
`tl.static_range` loop**, instead of one program per (row, tile) — this cuts
the launched block count several-fold (proportional to the tile count per row)
at every shape, and reduces to loading the row's scatter index once instead of
once per tile. Monotone-safe: at large shapes the grid is still large enough
to saturate the GPU, so bandwidth is unaffected; at small shapes fewer blocks
directly cuts scheduling overhead.

**A tempting fusion that doesn't work:** fusing the `clone()` seed and the
scatter into a single kernel to save the extra launch is **race-unsafe** —
Triton has no grid-wide barrier, so a scatter program can atomic-add into an
output row before that row's clone/seed write has landed, reading garbage.
A CAS-based "designated seeder" flag needs a spin-wait (deadlock risk) plus its
own init launch, netting no win. The 2-launch clone-then-scatter is the
correctness-safe floor for this pattern — don't chase fusing it away.

## Launch-overhead layer

Even without fusing, CUDA graphs / `torch.compile reduce-overhead` removes
launch gaps (1.81× e2e on batch-1 NVFP4 QwenImage ⚠️). For benchmark problems
measured at small sizes, launch overhead can dominate — check whether the
harness timing includes launch (it does: latency per call).
