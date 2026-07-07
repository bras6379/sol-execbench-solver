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

