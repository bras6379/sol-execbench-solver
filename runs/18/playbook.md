# Playbook — task 18 · reduction_18

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `de2f87a3` — One-launch Triton fusion of RMSNorm, RoPE, and KV-cache update over 8-head tiles, reusing each sin/cos vector across hea
If this underperforms on prefill shapes, switch to a two-stage design that precomputes one bf16 sin/cos tile per `(batch, seq)` and then runs separate Q and KV vectorized kernels; try it when SFU utilization or libdevice range-reduction dominates the fused kernel despite the saved launch.

## 2. from `d3f34260` — One-launch Triton fusion of per-head RMSNorm, RoPE, and KV-cache scatter-write with per-(batch,seq) cos/sin computed onc
If this kernel only ties the frontier despite autotune, the next reserve is a shape-dispatch decode megakernel (one CTA per batch, all 96 Q + 8 KV heads, S≤16) to cut multi-block launch overhead, and, for SFU-bound prefill shapes, a two-stage cos/sin precompute so each (batch,seq) angle is computed only once and reused across all head tiles.

## 3. from `b2a6470d` — Reload the 0.848 one-launch Triton fusion (RMSNorm+RoPE+KV-cache scatter) and apply the untried vectorization lever: las
If this only ties or barely beats 0.848, the next lever is to add `tl.multiple_of`/`tl.max_contiguous` alignment hints on top of the constexpr unit-stride fix — but with rank-matched tuple arguments (e.g. `tl.multiple_of(ptr, (1, 16))` for a `(HEAD_BLOCK, HALF)` 2D pointer tile), not a bare scalar; a prior attempt passed a scalar to a 2D tile and crashed all 13 workloads with RUNTIME_ERROR, which is why this round skipped manual hints entirely. If instead the win is concentrated on large prefill shapes but tiny decode shapes (S=1, workloads 0/2) stay flat, switch to a decode-specific megakerne

