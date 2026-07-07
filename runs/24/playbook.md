# Playbook — task 24 · gemm_24

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `21e2e2ef` — Algebraic collapse: scatter+matmul folds into per-patch weighted reduction; single fused Triton kernel eliminates ~10 Py
Wrap the fused Triton kernel in a `torch.cuda.CUDAGraph` to squeeze out the remaining single-kernel-launch overhead (~5–15 µs). The trigger: if this kernel scores below 0.7 (i.e., the single-launch overhead still dominates), capture the kernel call into a per-shape CUDA graph and replay it. Pre-allocate the output buffer outside the graph; the graph only captures the kernel dispatch. For the tiny shapes (P ≤ 256), the graph replay plus the fused kernel should approach the ~2 µs cold-L2 transfer floor.

## 2. from `d97cfa3e` — Algebraically collapse index_add+matmul into one per-patch weighted reduction over 18 frequencies, then cache and replay
If this graph-wrapped kernel scores below ~0.7, the next ceiling is a warp-specialized layout: assign one warp per patch so the 72 fp32 columns of grad_cos/grad_sin/emb are read coalesced, compute per-lane partials for the 18 outputs, warp-shuffle reduce within the warp, and use a single block-level reduction + 18 global atomics (or a second tiny reduction kernel) to eliminate the 18 per-j block reductions and scattered strided loads in the current kernel.

## 3. from `2d494aab` — Fused algebraic collapse kernel: per-patch weighted reduction replaces scatter+matmul, single Triton kernel eliminates 1
Wrap the fused Triton kernel in torch.cuda.CUDAGraph with per-shape grid capture to eliminate remaining single-kernel-launch overhead; trigger if score < 0.7 (expect 1.2-2× gain on tiny/medium shapes where launch still dominates).

