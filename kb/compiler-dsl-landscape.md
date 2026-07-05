# Compiler / DSL Landscape on B200

Which authoring path for which op class. Tags per kb/README.md.

## Ranking by op class (working hypothesis)

| Op class | Best path today | Fallback |
|---|---|---|
| Memory-bound (norm/softmax/rope/elementwise fusions) | CuTe-DSL (quack-style, "speed of light" claims ⚠️) or Triton | plain CUDA |
| Attention fwd/bwd | CuTe-DSL (FA4, cuDNN open kernels) | cuDNN graph API; Triton (slower) |
| Dense GEMM-shaped custom (fused epilogues) | CUTLASS C++ / CuTe-DSL | Gluon |
| Block-scaled FP4/FP8 | CUTLASS collective builders | CuTe-DSL example, Triton (MX support) |
| MoE grouped GEMM | CUTLASS 4.5 / cuDNN frontend | Triton (grouped launch) |
| Scan/sort | CUB | Triton |

## Triton on sm_100

- ✅ (medium) Runs on Blackwell **with no code changes**; up to 1.5× FP16
  flash-attention vs Hopper "for free"; MXFP8 GEMMs ≈ plain-FP8 speed with
  scaling done natively in the tensor core; MXFP4 ≈ 2× FP8 throughput.
- ✅ (medium) Known gaps (as of the NVIDIA/OpenAI joint blog, Feb 2025):
  sub-byte formats (MXFP4) need **manual layout/packing**; low matmul
  utilization when GEMM_K is small (manual sub-tiling mitigates; automatic
  warp specialization was "planned"); FP8 attention still being optimized.
- ✅ On plain dense BF16 GEMM, autotuned Triton 3.4 reaches only **62–70% of
  cuBLAS on B200** — don't use it for plain GEMM.
- ❌ Refuted: "Triton's Blackwell tensor-core optimizations apply automatically
  to any tl.dot kernel yielding strong perf" — too generous as stated.
- ❌ Refuted (1-2): "Triton FA2-style attention 526 TFLOP/s beats FA2 CUDA
  (401) on B200" — plausible but unconfirmed; benchmark before believing.
- ⚠️ FlashInfer's Triton fallback path emits no tcgen05 → reduced throughput
  vs tcgen05-native kernels. Triton is fine for bandwidth-bound; suspect for
  tensor-core-bound.

## Gluon (same repo as Triton)

- ✅ Lower-level language on the **same compiler stack/frontend/JIT as
  Triton** — the escape hatch when Triton codegen underperforms on sm_100,
  without leaving the toolchain.
- ✅ Exposes what Triton hides: explicit tile layouts, explicit shared memory
  (`shared_memory_descriptor`), explicit warp specialization, TMA, tcgen05 /
  TMEM ops. Compiles straight to ttg IR (skips tt IR) ⚠️.
- ⚠️ Cost: architecture-specific, non-portable; and low-level control ≠ free
  perf — a naive Gluon memcpy hit 666 GB/s of ~8 TB/s on GB200. Budget real
  tuning time.
- Tutorial 06-tcgen05 measured 1113 TFLOPS pipelined FP16 matmul (see
  [software-stack.md](software-stack.md)).

## CuTe-DSL (CUTLASS Python)

- ✅ Launched with CUTLASS 4.0 (2025-06); by 4.4 supports CUDA 13.1 + AOT
  compilation; TVM-FFI integration (4.3) cuts host launch overhead ⚠️.
- **The proven sm_100 path**: FlashAttention-4 ✅, NVIDIA's open SDPA
  fprop/bprop kernels ✅, quack's near-peak memory-bound kernels ⚠️, FlashInfer
  SM100 kernels ⚠️ are all written in it.
- ⚠️ Block-scaled persistent GEMM example
  (`dense_blockscaled_gemm_persistent.py`) exists but is explicitly a
  starting point, not tuned peak.

## CuTile (NVIDIA tile DSL)

- ⚠️ Compiles tiny Python (86-line MoE kernel) into Blackwell-native code:
  auto single-thread MMA issue via `elect.sync`, multi-mbarrier pipelining,
  TMEM alloc with nanosleep-backoff contention handling — tcgen05 idioms you'd
  otherwise hand-write. [Toulmé analysis]
- ⚠️ BUT: the MLIR passes doing the magic are **proprietary** (not in the
  public cutile-python repo); no public quantitative benchmarks; bf16/fp32
  only in the analysis.
- ✅ Independent measurements: CuTile GEMM = 52–79% of cuBLAS on B200; its
  compiler (tileiras, CUDA 13.1) is tuned for sm_100 — the same kernel gets
  5.6× worse relative perf on sm_120. Validate per-chip.
- ❌ Refuted: "CuTile attention 1,007 TFLOP/s, 2.51× FA2, fastest measured on
  B200" — killed 0-3. Treat CuTile as promising, not proven.

## torch.compile / TorchInductor

- ⚠️ For small-batch quantized inference, `mode='reduce-overhead'` (CUDA
  graphs) is a real lever: QwenImage+NVFP4 batch-1 went to 1.81× e2e.
- ✅ In the academic B200 study, torch.compile added ~nothing over eager
  FA2 for the tested model — it's a launch-overhead/fusion tool, not a
  kernel-quality tool on Blackwell today.
- Useful as the *baseline to beat* per problem and for quick fusion of
  bandwidth-bound chains.

## ThunderKittens

- ✅ (medium) B200 GEMM/attention kernels at/near cuBLAS/cuDNN (self-reported;
  "up to 2×" figures are vs H100 hardware, largely the hw uplift). C++
  template library; 2-SM MMA via `ncta=2` ✅. See
  [optimization-playbook.md](optimization-playbook.md) for its TK2 tuning
  lessons (TMEM double-buffering, fence removal).

## Practical routing rule for the solver

1. Baseline: PyTorch eager → torch.compile.
2. Library call if the decision table in
   [library-playbook.md](library-playbook.md) covers the op.
3. Triton for bandwidth-bound fusion candidates (fast to write, good enough).
4. CuTe-DSL or CUTLASS when tensor-core peak matters (GEMM-shaped, attention,
   quant).
5. Gluon when Triton is close but leaves sm_100 features unused.
6. Raw CUDA/PTX only when profiling proves everything else short.
