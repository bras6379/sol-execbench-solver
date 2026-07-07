# Playbook — task 18 · reduction_18

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `de2f87a3` — One-launch Triton fusion of RMSNorm, RoPE, and KV-cache update over 8-head tiles, reusing each sin/cos vector across hea
If this underperforms on prefill shapes, switch to a two-stage design that precomputes one bf16 sin/cos tile per `(batch, seq)` and then runs separate Q and KV vectorized kernels; try it when SFU utilization or libdevice range-reduction dominates the fused kernel despite the saved launch.

