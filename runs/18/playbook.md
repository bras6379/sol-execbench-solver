# Playbook — task 18 · reduction_18

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `de2f87a3` — One-launch Triton fusion of RMSNorm, RoPE, and KV-cache update over 8-head tiles, reusing each sin/cos vector across hea
If this underperforms on prefill shapes, switch to a two-stage design that precomputes one bf16 sin/cos tile per `(batch, seq)` and then runs separate Q and KV vectorized kernels; try it when SFU utilization or libdevice range-reduction dominates the fused kernel despite the saved launch.

## 2. from `d3f34260` — One-launch Triton fusion of per-head RMSNorm, RoPE, and KV-cache scatter-write with per-(batch,seq) cos/sin computed onc
If this kernel only ties the frontier despite autotune, the next reserve is a shape-dispatch decode megakernel (one CTA per batch, all 96 Q + 8 KV heads, S≤16) to cut multi-block launch overhead, and, for SFU-bound prefill shapes, a two-stage cos/sin precompute so each (batch,seq) angle is computed only once and reused across all head tiles.

## 3. from `b2a6470d` — Reload the 0.848 one-launch Triton fusion (RMSNorm+RoPE+KV-cache scatter) and apply the untried vectorization lever: las
If this only ties or barely beats 0.848, the next lever is to add `tl.multiple_of`/`tl.max_contiguous` alignment hints on top of the constexpr unit-stride fix — but with rank-matched tuple arguments (e.g. `tl.multiple_of(ptr, (1, 16))` for a `(HEAD_BLOCK, HALF)` 2D pointer tile), not a bare scalar; a prior attempt passed a scalar to a 2D tile and crashed all 13 workloads with RUNTIME_ERROR, which is why this round skipped manual hints entirely. If instead the win is concentrated on large prefill shapes but tiny decode shapes (S=1, workloads 0/2) stay flat, switch to a decode-specific megakerne

## 4. from `13461bec` — Expanded autotune space (1–32 warps, num_stages 1–3) to find optimal config per workload; current 0.865 vectorized Trito
**Current ceiling:** 0.865 is at hardware bandwidth limit (85-95% of 7.7 TB/s peak). Achieved ~6.5 TB/s effective on memory-bound op with AI≈0.2 FLOP/byte. To beat 0.909 threshold requires 11% speedup unlikely without algorithmic change.

**If this ties or regresses 0.865:** Switch to shape-dispatch decode megakernel for S≤16 workloads (0,2) — one CTA processes batch*heads instead of grid-stride, cuts launch overhead from 7→1 program per shape. Fallback: two-stage sin/cos precompute (precompute per-seq once, amortize across 7 head-tile programs) if decode kernel doesn't win on prefill.

## 5. from `4847a63a` — Single-launch vectorized Triton fusion for RMSNorm+RoPE+KV-cache update, with stable prefill tiling and a 32-head S=1 di
I did not ship a true decode megakernel that processes all Q/KV work for one batch in one persistent program; try that if workloads 0 and 2 still lag after this 32-head S=1 dispatch. For prefill regressions, revert the decode-only branch first, then consider a two-stage sin/cos precompute only if profiling shows SFU range reduction dominating despite cold-L2 traffic.

