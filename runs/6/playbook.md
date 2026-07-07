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

## 3. from `4f190950` — Vectorized fused Hyena short-filter: same proven two-path (simple<4096, persistent>=4096) conv1d k=3 + split + v*x0, but
Higher-ceiling idea NOT shipped: **shared-memory halo staging.** This round vectorized the
unique HBM stream but the two causal lookback loads (a_{t-1}, a_{t-2}) are still predicated
per-lane global loads — cache hits, but they burn LSU-instruction throughput, which may be the
true limiter on the small-S mem-bound shapes (idx 2 B=64/S=128, idx 10 B=32/S=128) that are
single-tile (BLOCK_T=128) so nothing vectorizes for them. Next kernel: load the BLOCK_T+2 window
of each of the 3 input rows ONCE into shared memory with a single coalesced vectorized load, then
read all three taps (t, t-1, t-2) f

## 4. from `bdcc783f` — Fused Triton depthwise causal conv1d(k=3) + 768->3x256 split + v*x0 gate (two-path: simple launch-bound / persistent gri
Higher-ceiling idea NOT shipped: the two smallest single-tile mem-bound shapes (idx2 B=64/S=128, idx10 B=32/S=128) always have `tb==0` (S==BLOCK_T, one tile per channel), so this round's interior-tile mask elimination cannot help them — they're stuck with predicated lookback loads. Trigger: if overall score doesn't clear ~0.75, retry the 0.720 attempt's channel-affine 2D persistent schedule (CTA pins to one channel, preloads weights once) but with PROG_MULT tuned much higher (near-degenerate, many CTAs) instead of the low PROG_MULT it used — isolate the "avoid L1 thrash from channel-hopping" b

