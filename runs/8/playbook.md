# Playbook — task 8 · reduction_8

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `29623a03` — Fused Triton bf16 atomic scatter-add (H=3072), coarsened 4 eo-rows/program: clone-seed output=fhs, grid=N/4, each progra
Higher-ceiling idea NOT shipped: the atomic-free **sort + segmented-reduction** kernel (DESIGN #3), which writes `output` exactly once and eliminates *all* the L2 atomic-RMW traffic and the clone launch. Concretely: `sorted_idx, perm = token_indices.sort()`; `offsets = torch.searchsorted(sorted_idx, torch.arange(T+1))` to get each token's contiguous segment; then an **output-centric** Triton kernel (grid = T tokens x 3 tiles) that fp32-accumulates `fhs[t] + sum(eo[perm[j]])` over the segment and writes `output[t]` once (fp32 accum is safely within this problem's loose atol~0.1 vs the reference

## 2. from `9cf39188` — Fused Triton bf16 atomic scatter-add (grid=N, one program per eo row, BLOCK=1024, 4 warps): clone-seed output, 3×1024 ho
Higher-ceiling idea NOT shipped: atomic-free sort + segmented reduction hybrid. For T ≤ 2048 (eo ≤ 100 MB fits in 126 MB L2), sort token_indices (CUB radix sort), compute segment offsets (searchsorted), then output-centric Triton kernel (grid=T×3 tiles) fp32-accumulates fhs[t] + Σeo[perm[j]] and writes output once — eliminates the clone launch AND all L2 atomic-RMW traffic. Scattered eo reads hit L2 (21 TB/s) at small T, beating the coalesced-HBM + atomic-RMW path. Gate by T: sort path for T ≤ 2048, atomic path for T ≥ 4096. Trigger: if this kernel stays at the ~0.844 frontier, the sort path i

