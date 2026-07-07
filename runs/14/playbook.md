# Playbook — task 14 · gemm_14

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `c9d2ba4f` — Single fused Triton kernel: load grad_cos/grad_sin halves, sum duplicates, apply sin/cos chain rule with attention scali
If this only ties baseline or Triton's large-shape bandwidth is under ~6 TB/s, switch to a persistent CUDA/CuTe kernel with a 148-CTA grid, float4/ushort4 vector loads, and a shared-memory inv_freq broadcast to remove launch overhead and push HBM utilization.

## 2. from `bb4eb238` — Chunk half_dim (64) into BLOCK_D-sized tiles to cut register pressure from O(BLOCK_M×64) to O(BLOCK_M×BLOCK_D), unblocki
If this still scores below ~0.93, escalate to a persistent CUDA kernel: 148-CTA grid (one per SM), each CTA work-steals tiles of BLOCK_M=256 rows from a global atomic counter. Use float4 loads for emb (fp32), ushort4 for grad_cos/grad_sin (bf16), pre-broadcast inv_freq into __shared__ (256 bytes), and warp-shuffle the 64-element dot-product reduction. The trigger is that Triton's codegen overhead (non-ideal instruction scheduling, missed vectorization of bf16 loads) leaves ~5-10% bandwidth on the table vs hand-tuned CUDA on this strictly memory-bound, low-AI op.

