# Playbook — task 10 · rmsnorm_10

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `90b8da0e` — Shape-dispatched split-K Triton for T<=1024 (spreads K across 16 slices to light many SMs and fuses the [B,8,S,128] stor
If this ties cuBLAS on small T or lags SOL on the compute-bound tail, the higher ceiling is a CuTe-DSL SM100 tcgen05 single-kernel GEMM with a custom head-scatter epilogue that writes [B,8,S,128] directly from TMEM, eliminating the fp32 scratch/reduction pass. Trigger: raw score remains below ~0.70 or large-T (T >= 2048) is more than 1.25x SOL.

## 2. from `94b0d20c` — Split-K Triton GEMM with fused atomic [B,8,S,128] store for small T (fills all SMs to stream the 10.5MB weight at HBM BW
Higher-ceiling idea NOT shipped: a CuTe-DSL SM100 tcgen05 single-kernel GEMM with a custom head-scatter epilogue that writes [B,8,S,128] straight from TMEM (no fp32 atomic scratch, no separate bf16 cast). This is the only path that hits SOL on BOTH ends: tcgen05 MMA ~80%+ on the compute-bound tail AND single-launch for small T.

Trigger: if this kernel only ties cuBLAS (~0.51) at small T, OR the large-T tail (T>=2048) stays above ~1.25x SOL (raw score below ~0.70). Also worth trying if the timing gate keeps falling back to cuBLAS at T=1024 — that means split-K's atomic contention / M-tiling ov

## 3. from `f5cd7a12` — Per-shape cold-L2-gated dispatch for GQA value-proj: fused single-launch bf16 Triton GEMM (direct [B,8,S,128] store) or
HIGHER-CEILING RESERVE (not shipped this round):

CuTe-DSL SM100 tcgen05 single-kernel GEMM (M=T, N=1024, K=5120) with a
custom head-scatter epilogue that writes [B,8,S,128] straight from TMEM — no
fp32 scratch, no separate cast. This is the ONLY path that beats cuBLAS on the
compute-bound tail (T>=2048): tcgen05 MMA ~80%+ of peak AND single-launch for
small T. Study/port the open CuTe-DSL GEMM kernels in NVIDIA cudnn-frontend
(python/cudnn/grouped_gemm, rmsnorm_rht_amax) and CUTLASS ex.
71_blackwell_gemm_with_collective_builder; SM100 EVT is refuted so hand-code the
per-head store. It was NOT

## 4. from `55dfafb5` — FP8 weight quantization (e4m3, per-row fp32 scales, cached) halves the cold 10.5 MB weight traffic for the memory-bound
Higher-ceiling reserve: CuTe-DSL SM100 tcgen05 single-kernel GEMM (M=T, N=1024, K=5120) with a custom head-scatter epilogue that writes [B,8,S,128] directly from TMEM — no fp32 scratch, no separate bf16 cast, no atomic-add reduction. This is the only path that hits SOL on both ends: tcgen05 MMA ~80%+ of peak on the compute-bound tail AND single-launch for small T. Study the open CuTe-DSL GEMM kernels in NVIDIA cudnn-frontend (python/cudnn/grouped_gemm, rmsnorm_rht_amax) and CUTLASS ex. 71_blackwell_gemm_with_collective_builder; SM100 EVT is refuted so hand-code the per-head store.

Trigger to

