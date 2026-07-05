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

## Launch-overhead layer

Even without fusing, CUDA graphs / `torch.compile reduce-overhead` removes
launch gaps (1.81× e2e on batch-1 NVFP4 QwenImage ⚠️). For benchmark problems
measured at small sizes, launch overhead can dominate — check whether the
harness timing includes launch (it does: latency per call).
