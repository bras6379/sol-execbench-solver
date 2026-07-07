# Playbook — task 35 · attention_35

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `4f5c0feb` — fp16-weight-cache keyed by tensor object identity (id(weight)) to halve memory traffic (113 MB weight) while avoiding "s
Batched GEMV kernel with per-output-row coalesced 128-bit weight streaming for memory-bound shapes (B=5–211): Triton kernel that assigns one CTA per output row, accumulates K=3072 dot products with SIMD-friendly layout, and writes fp32 accumulators directly. Expected 10–15% BW gain by avoiding skinny-M tensor-core underutilization in cuBLAS and achieving higher HBM bandwidth saturation for B≤211.

## 2. from `0fa4aa07` — Replace reference's two-op matmul+bias-add with a single fused torch.nn.functional.linear (cuBLASLt TF32 GEMM with bias
If this plateaus below ~0.6 (especially on the memory-bound B<=211 shapes), the next lever is the untried Triton batched-GEMV reserve play in DESIGN.md: one CTA per output row streaming the 226MB weight with wide coalesced loads, since cuBLAS's TF32 tensor-core GEMM is underutilized at such small M — but validate it in isolation on one shape before trusting all 8, since every custom/cached rewrite of this op has zeroed the whole problem so far.

Do NOT retry any weight/bias caching keyed by `data_ptr()` or `id()` across calls — 4/4 attempts failed (mostly all-16 or 13/16 workloads INCORRECT),

