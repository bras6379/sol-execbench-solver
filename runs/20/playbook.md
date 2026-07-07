# Playbook — task 20 · rmsnorm_20

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `3db3660b` — Fused Triton LayerNorm+spatial-shuffle (GPU-built permutation, no per-grid .item() loop) writing the [M,6144] merged ten
Higher-ceiling play not shipped: fuse FC1's bias+GELU (and ideally FC1->FC2) into a custom Triton/CuTe GEMM epilogue so the [M,6144] FC1 output never round-trips through GMEM and the separate GELU pass disappears (Approach 3 megakernel; both weights, 119.5MB, are L2-resident). Trigger: if this only modestly beats the 0.5 baseline at medium/large M, profile — if the FC1-output write + standalone GELU pass is >10% of time, replace `F.linear+F.gelu` for FC1 with a fused matmul+GELU-epilogue kernel (keep cuBLAS FC2 first, then attempt the full TMEM-resident FC1->FC2 chain).

## 2. from `79a31401` — Single fused LN+spatial-shuffle Triton kernel with INLINE per-program source-index computation (no perm tensor, no perm-
Higher-ceiling idea NOT shipped: fuse the 2x2 shuffle into FC1's input load (Approach 2) — a custom Triton/CuTe FC1 GEMM that gathers the 4 source LN'd rows per merged row directly into shared memory and applies bias+GELU in the epilogue, so the [M,6144] `shuffled` intermediate is never written/re-read and GELU never round-trips GMEM (saves ~2*N*1536*2 bytes prelude + M*6144*2*2 bytes GELU traffic — meaningful at large M shapes #7/#10/#11).

Trigger: if this sync-free version reaches ~0.5 but profiling shows (a) the `shuffled` write + FC1 read or (b) the standalone F.gelu pass is >10% of wall-

