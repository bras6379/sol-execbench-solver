# Playbook — task 45 · attention_45

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `b2917e63` — Fixed critical correctness bug in torch.compile approach: added .clone() to return (CUDA graph buffers are static and ge
If this clone() fix doesn't improve scoring beyond 0.243 baseline: The 3-kernel Triton approach (DESIGN.md §7) remains the recommended path to close the recompute-vs-materialize gap (16–20× fewer bytes). The prior RUNTIME_ERROR suggests a grid/bounds issue — rebuild with careful per-shape grid sizing: N_tiles = ceil(N / tile_size), grid = min(N_tiles, 148*occupancy), bounds-check every index before access. Alternatively, if eager debugging is needed, fall back to DESIGN.md candidate #3 (cuBLASLt fused epilogue + torch.compile GRN + cuBLASLt GEMM2 in a CUDA graph).

## 2. from `62b668e7` — Shape-specialized torch.compile with TF32 precision and per-shape caching: max-autotune for launch-bound shapes (N<25k),
# Handoff — Next Lever if Score < 0.35

If the torch.compile+TF32 approach doesn't beat the 0.355 frontier significantly, the next lever is a **2-pass Triton recompute kernel**:

**Strategy**: Implement a cleaner 2-pass approach that avoids the 3-kernel complexity:
- **Pass 1** (reduction kernel): For each of B batches and hidden_dim channels in parallel, accumulate sum-of-squares across all H×W spatial positions. Each (b,c) block iterates sequentially through the H×W positions, recomputing GELU(linear1) per position (cheap, no materialization). Use thread-level reduction within each block, th

