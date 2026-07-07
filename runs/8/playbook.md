# Playbook — task 8 · reduction_8

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `29623a03` — Fused Triton bf16 atomic scatter-add (H=3072), coarsened 4 eo-rows/program: clone-seed output=fhs, grid=N/4, each progra
Higher-ceiling idea NOT shipped: the atomic-free **sort + segmented-reduction** kernel (DESIGN #3), which writes `output` exactly once and eliminates *all* the L2 atomic-RMW traffic and the clone launch. Concretely: `sorted_idx, perm = token_indices.sort()`; `offsets = torch.searchsorted(sorted_idx, torch.arange(T+1))` to get each token's contiguous segment; then an **output-centric** Triton kernel (grid = T tokens x 3 tiles) that fp32-accumulates `fhs[t] + sum(eo[perm[j]])` over the segment and writes `output[t]` once (fp32 accum is safely within this problem's loose atol~0.1 vs the reference

