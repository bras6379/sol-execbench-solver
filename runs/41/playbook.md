# Playbook — task 41 · rope_41

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `24d9fda0` — cuDNN NHWC + benchmark autotune: enable tensor-core conv routing, per-shape algorithm cache during warmup, avoid output-
# Handoff: Next Unshipped Lever

**Unshipped idea:** Approach 2 — Fused FP16 output emission in cuDNN conv kernel.

**Why:** This op's arithmetic intensity (~48 FLOP/byte fp32 I/O) is shape-invariant and below the memory-bound ridge for both TF32 and FP16 rates. Fusing fp16 output directly into the conv epilogue (accumulate fp32/TF32, cast only in the store) saves ~17% of total HBM traffic (5N vs 6N bytes per input element), uniform across all 16 shapes. Do NOT do a separate fp32→fp16 cast pass (that adds 50% extra traffic).

**Trigger:** If Approach 1 (this kernel) scores >0.5 but <0.75 on th

## 2. from `783503e3` — Proven frontier baseline: cuDNN NHWC layout + benchmark autotune for per-shape algorithm caching, no spurious output cop
# Handoff: Problem 041 Status and Next Steps

## Current State
**Frontier score: 0.505 (raw) / 0.460 (leaderboard-est)** — NHWC layout + cudnn.benchmark autotune.

The kernel is well-optimized for the fp32 path: empirically beats NCHW baseline (0.492) and seed (0.498) because cuDNN's tensor-core routing is significantly more BW-efficient on NHWC, even though the layout conversion costs a full read/write pass of the input.

## Bottleneck Analysis

**All 16 workloads are memory-bound** at fixed arithmetic intensity ≈48 FLOP/byte (shape-invariant due to fixed Cin=32, Cout=64, K=3). This sits far

