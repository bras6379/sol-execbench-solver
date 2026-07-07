# Playbook — task 6 · moe_6

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `a94c6b77` — Two-path fused Hyena short-filter (depthwise causal conv1d k=3 lookback-2 + 768->3x256 split + v*x0 gate): proven straig
Higher-ceiling idea NOT shipped: **cold-L2-aware config selection.** Triton's
autotuner times each config WARM (data resident in L2 at 21 TB/s), but the grader
scores COLD (2xL2 flushed each iter -> HBM at ~7.7 TB/s). So warm-autotune tends
to pick high-occupancy small-block configs that look fast warm but under-pipeline
cold, which is exactly why the mem-bound shapes stall at 1.5-2.5x the HBM byte
floor (idx10 25MB: 8.35us vs 3.4us floor). This round's persistent kernel adds the
pipelined structure but still lets warm-autotune choose among its configs.

TRIGGER: if this persistent kernel only

## 2. from `68e868dc` — Single persistent Triton kernel w/ CHANNEL_BLOCK∈{1,2,4} coarsening, load-compute-store reordered for num_stages SW pipe
2D persistent grid: outer dim = `min(B * ceil(DM/CHANNEL_BLOCK), NUM_SM * PROG_MULT)` selects (batch, channel-group) pairs; inner loop = grid-stride only over time tiles. Each CTA stays on the same channel(s) for its lifetime, preloading weights+bias once outside the time loop and keeping the 3 input rows hot in L1 across all time tiles. The current flat 1D stride causes L1 thrashing when channels change every few iterations. Trigger: if overall score < 0.78 after this round, the L1-thrash from channel-hopping is the bottleneck — the 2D grid also enables warp-specialized producer-consumer doub

