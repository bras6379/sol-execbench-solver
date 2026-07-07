# Playbook — task 4 · gemm_4

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `b2fe61b3` — Sequential bf16 tensor-core GEMMs (fp32 accumulation) with a zero-copy transposed (B,H,S,D) view for grad_attn_output, a
If this single-stream view strategy only ties the frontier, the next lever is a CUTLASS/CuTe-DSL custom epilogue for the dgrad GEMM that scatters outputs directly into contiguous (B,H,S,D) layout, removing the intermediate M×2048 buffer.

## 2. from `e0b86351` — bf16 cuBLAS tensor-core GEMMs (fp32-accum) for wgrad+dgrad with grad_attn_output returned as a zero-copy transposed (B,3
Higher-ceiling idea NOT shipped: CUDA-graph the small-M path. At M<=1024 the
bottleneck is kernel-launch overhead (~2 cuBLAS launches + sync, ~10-20us) that
dominates the <20us of actual GEMM work, which concurrent streams alone cannot
remove. Trigger: if this gated-stream kernel only TIES the frontier (~0.710),
launch overhead -- not SM underfill -- is the small-M ceiling. Next kernel:
per-M CUDA-graph cache -- prime the cuBLAS workspace once, capture
{wgrad-on-side-stream + dgrad-on-current-stream} into one graph, replay it, and
CLONE the two static output buffers before returning (never ali

