# Playbook — task 18 · reduction_18

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `de2f87a3` — One-launch Triton fusion of RMSNorm, RoPE, and KV-cache update over 8-head tiles, reusing each sin/cos vector across hea
If this underperforms on prefill shapes, switch to a two-stage design that precomputes one bf16 sin/cos tile per `(batch, seq)` and then runs separate Q and KV vectorized kernels; try it when SFU utilization or libdevice range-reduction dominates the fused kernel despite the saved launch.

## 2. from `d3f34260` — One-launch Triton fusion of per-head RMSNorm, RoPE, and KV-cache scatter-write with per-(batch,seq) cos/sin computed onc
If this kernel only ties the frontier despite autotune, the next reserve is a shape-dispatch decode megakernel (one CTA per batch, all 96 Q + 8 KV heads, S≤16) to cut multi-block launch overhead, and, for SFU-bound prefill shapes, a two-stage cos/sin precompute so each (batch,seq) angle is computed only once and reused across all head tiles.

