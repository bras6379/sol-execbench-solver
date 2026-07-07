# Playbook — task 20 · rmsnorm_20

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `3db3660b` — Fused Triton LayerNorm+spatial-shuffle (GPU-built permutation, no per-grid .item() loop) writing the [M,6144] merged ten
Higher-ceiling play not shipped: fuse FC1's bias+GELU (and ideally FC1->FC2) into a custom Triton/CuTe GEMM epilogue so the [M,6144] FC1 output never round-trips through GMEM and the separate GELU pass disappears (Approach 3 megakernel; both weights, 119.5MB, are L2-resident). Trigger: if this only modestly beats the 0.5 baseline at medium/large M, profile — if the FC1-output write + standalone GELU pass is >10% of time, replace `F.linear+F.gelu` for FC1 with a fused matmul+GELU-epilogue kernel (keep cuBLAS FC2 first, then attempt the full TMEM-resident FC1->FC2 chain).

