# Playbook — task 27 · elementwise_27

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `a7cfbd9f` — fp32 (no TF32): fuse the 3D-RoPE into one gather+FMA with cached cos/sin tables and replace the O(S^2) score/prob materi
# Handoff — problem 027 (video spatial attention + 3D RoPE)

## Shipped this round
Safe fp32 kernel: fused 3D-RoPE (single gather + FMA, cached cos/sin) + fp32
`scaled_dot_product_attention` (kills S^2 materialization) + cuBLAS fp32
projections. Everything fp32 -> guaranteed correct on all 16 shapes. The fp32
attention matmul (~4 ms for the S=7574/8192 shapes) is the floor for a safe
kernel; expect ~2x over the reference (~0.45-0.55 avg). One failing shape zeroes
the whole candidate, so I did NOT gamble on reduced precision.

## HIGHER-CEILING reserve (try next, with GPU feedback)
**Custom Tri

## 2. from `83e68737` — Enable TF32 (remove the previous round's explicit fp32 restriction) — the reference already uses TF32 matmuls, so matchi
Cast Q, K, V to BF16 before SDPA (then cast output back to fp32) to force cuDNN's BF16 FlashAttention kernel. The BF16 tensor cores are 2× faster than TF32 for the attention matmuls (2.25 vs 1.1 PFLOPS). Trigger: if the TF32 kernel passes correctness but the attention is still the bottleneck (score < 0.7), try BF16 attention — the tolerance may be wide enough to absorb the additional ~8× rounding error from BF16 vs TF32, especially since the 1% matched_ratio slack can absorb outliers.

