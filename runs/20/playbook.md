# Playbook — task 20 · rmsnorm_20

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `3db3660b` — Fused Triton LayerNorm+spatial-shuffle (GPU-built permutation, no per-grid .item() loop) writing the [M,6144] merged ten
Higher-ceiling play not shipped: fuse FC1's bias+GELU (and ideally FC1->FC2) into a custom Triton/CuTe GEMM epilogue so the [M,6144] FC1 output never round-trips through GMEM and the separate GELU pass disappears (Approach 3 megakernel; both weights, 119.5MB, are L2-resident). Trigger: if this only modestly beats the 0.5 baseline at medium/large M, profile — if the FC1-output write + standalone GELU pass is >10% of time, replace `F.linear+F.gelu` for FC1 with a fused matmul+GELU-epilogue kernel (keep cuBLAS FC2 first, then attempt the full TMEM-resident FC1->FC2 chain).

## 2. from `79a31401` — Single fused LN+spatial-shuffle Triton kernel with INLINE per-program source-index computation (no perm tensor, no perm-
Higher-ceiling idea NOT shipped: fuse the 2x2 shuffle into FC1's input load (Approach 2) — a custom Triton/CuTe FC1 GEMM that gathers the 4 source LN'd rows per merged row directly into shared memory and applies bias+GELU in the epilogue, so the [M,6144] `shuffled` intermediate is never written/re-read and GELU never round-trips GMEM (saves ~2*N*1536*2 bytes prelude + M*6144*2*2 bytes GELU traffic — meaningful at large M shapes #7/#10/#11).

Trigger: if this sync-free version reaches ~0.5 but profiling shows (a) the `shuffled` write + FC1 read or (b) the standalone F.gelu pass is >10% of wall-

## 3. from `ed3d1f81` — Same fused single-launch LN+shuffle Triton kernel (inline shuffle-index derivation, zero host syncs) + cuBLAS FC1/GELU/F
If this unmasked-chunk fix only ties 0.639 (i.e. the mask wasn't actually costing bandwidth — ptxas may already fully predicate off the uniform trailing warp), the load/store pattern is not the lever; stop touching the LN+shuffle kernel body and instead tune its launch config per shape-regime (num_warps ∈ {1,2} for the small-M shapes #2/#14/#4 where grid=4*M is launch-latency-bound, vs num_warps=8 + multiple rows/program for the large-M shapes #7/#11 to raise occupancy/ILP) — this axis is untried and doesn't touch the GEMM fusion path, which has failed correctness 4/4 times (RUNTIME_ERROR/COMP

## 4. from `623ec554` — Fused LN+shuffle Triton kernel (3×512 unmasked loads) + cuBLAS FC1/FC2, optimized with memory hierarchy hints (evict_fir
# Handoff: Next Higher-Ceiling Ideas

## Primary reserve play: Approach 2 (Fused FC1 with shuffle-addressed input loading)

If this eviction-policy optimization only marginally improves upon 0.651 (e.g., reaches ~0.66–0.67), the hidden bottleneck is likely the [M,6144] shuffled tensor write+read or the standalone F.gelu round-trip. Implement Approach 2 from DESIGN.md: a custom Triton FC1 GEMM that reads LN'd hidden using 4-row shuffle-addressed gathers directly into shared memory, applies FC1 matmul, then fuses bias+GELU in the epilogue before writing output. This eliminates the shuffled inter

## 5. from `777972e5` — Kept the proven 0.651 fused LN+shuffle Triton kernel unchanged (3x512 unmasked loads, eviction hints); fused FC1's bias+
If torch._addmm_activation only ties ~0.651 (cuBLASLt didn't pick the fused GELU epilogue on B200, or the saved traffic was already hidden under compute at medium/large M), the next lever is CUDA-graph capture of the whole 3-launch pipeline (LN+shuffle Triton kernel + FC1 + FC2), keyed by (N, ngrids), targeting the launch-latency-bound small-M shapes #1/#2/#12/#14 (M<=256) and #4 (M=400) — this exact lever (graph-collapsing launch overhead on small/launch-bound shapes) already won on problems 005 and 007 in this benchmark family and is untried here. Must clone the output tensor out of the stat

