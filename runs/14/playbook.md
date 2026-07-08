# Playbook — task 14 · gemm_14

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `c9d2ba4f` — Single fused Triton kernel: load grad_cos/grad_sin halves, sum duplicates, apply sin/cos chain rule with attention scali
If this only ties baseline or Triton's large-shape bandwidth is under ~6 TB/s, switch to a persistent CUDA/CuTe kernel with a 148-CTA grid, float4/ushort4 vector loads, and a shared-memory inv_freq broadcast to remove launch overhead and push HBM utilization.

## 2. from `bb4eb238` — Chunk half_dim (64) into BLOCK_D-sized tiles to cut register pressure from O(BLOCK_M×64) to O(BLOCK_M×BLOCK_D), unblocki
If this still scores below ~0.93, escalate to a persistent CUDA kernel: 148-CTA grid (one per SM), each CTA work-steals tiles of BLOCK_M=256 rows from a global atomic counter. Use float4 loads for emb (fp32), ushort4 for grad_cos/grad_sin (bf16), pre-broadcast inv_freq into __shared__ (256 bytes), and warp-shuffle the 64-element dot-product reduction. The trigger is that Triton's codegen overhead (non-ideal instruction scheduling, missed vectorization of bf16 loads) leaves ~5-10% bandwidth on the table vs hand-tuned CUDA on this strictly memory-bound, low-AI op.

## 3. from `dc061e75` — Shape-dispatch between the 0.885 fused Triton frontier autotune set and a coalesced no-BLOCK_D=8 autotune set for the ex
If this exact-shape hybrid only ties or regresses, try a one-language `kernel.cu` retry of the 2D-grid CUDA kernel with the prior compile fixes: include `ATen/cuda/CUDAContext.h` and `c10/cuda/CUDAException.h`, remove any Python source, keep strides or require contiguous only after measuring, and specialize B=64/S=3011,8192 with vectorized bf16/fp32 row loads plus shared `inv_freq`.

## 4. from `7b7eab33` — Manual 2-stage software prefetch for cold-L2 streaming: pre-load chunk 0, then iterate loading chunk i while processing
# Handoff: Next Kernel Iteration

If this manual 2-stage prefetch regresses or only ties the 0.885 frontier, or if workloads #8/#14 still score below ~0.72-0.75, escalate to a persistent CUDA kernel: 148-CTA grid (one per SM), each CTA work-steals tiles of BLOCK_M=256 rows from a global atomic counter, with float4/ushort4 vectorized loads for emb/grad_cos_sin, shared-memory inv_freq broadcast (256 bytes per CTA), and warp-shuffle 64-element dot-product reduction. This removes launch overhead and explicit Triton codegen suboptimalities, pushing achieved HBM BW from current ~5-6 TB/s into the 6.

