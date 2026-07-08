# Playbook — task 13 · layernorm_13

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `3b40d0a8` — Triton implementation: full-row RMSNorm-backward dx kernel with non-atomic tiled grad_weight reductions, plus a one-laun
Higher-ceiling reserve: replace the Python/Triton split with a C++ CUDA persistent kernel that computes row means and bf16 dx while accumulating per-CTA grad_weight in shared memory, followed by a CUB-style block reduction that writes grad_weight once. Try it if this is correct but still trails the PyTorch baseline or if the three-launch large-row path is the measured bottleneck.

## 2. from `58a48f49` — Three-tier shape-specialized fused Triton kernels: tiny (n_rows≤8) single-program one-pass, small (≤128) per-row two-pas
Higher-ceiling reserve: single-pass shared-memory-buffered kernel. Buffer go*w and norm (20 KB per row) in shared memory during the mean-reduction pass, then compute grad_x from SMEM without re-reading HBM — cuts HBM traffic from ~16 B/elem to ~10 B/elem (1.6×). Use BLOCK_M=1, atomics for grad_weight (2560-element array fits in L2). On B200 228 KB SMEM, fits ~11 CTAs/SM at 5-warps each. Trigger: if this kernel ties or only marginally beats the frontier (0.653), or if profiling shows the large-shape path is HBM-bandwidth-bound at <60% of peak.

## 3. from `21b91ed0` — Keep proven 0.653 Triton kernel structure; add per-shape CUDA graphs (n_rows<=16K) with pre-allocated buffers and output
Single-pass SMEM-buffered kernel: buffer go*w and norm (20 KB per row) in shared memory during the mean-reduction pass, then compute grad_x from SMEM without re-reading HBM — cuts HBM traffic from ~16 B/elem to ~10 B/elem (1.6×). Use BLOCK_M=1, pre-allocate a per-CTA grad_weight partial buffer in global memory (not atomics — those failed 3×), and write each CTA's partial sum to a unique slot. A second kernel reduces the partial buffer. This eliminates the second read of grad_output and normalized for the grad_weight pass. Trigger: if the CUDA-graph approach doesn't lift the large-shape scores

## 4. from `8fe2822c` — Fuse the proven 0.653 kernel's separate row_dx and grad_weight-partial Triton launches into one loop-based kernel (BLOCK
CUDA graphs are a dead end here — already tried twice (`prior/0.509_21b91ed0.py`, correct but slower; `prior/0.000_3aee3613.py`, NO_TRACE): per-workload data shows the required `copy_()` of fresh inputs into static graph buffers costs more than the launch overhead it saves on 15/16 shapes (only #14 improved). Don't retry it unless a copy-free capture (relying on stable input addresses across cold-L2 reps, as validated on problem 005) can be made to work — that was never gotten correct.

If this fused 2-launch kernel only ties 0.653, the next lever is raising MAX_BLOCK_M (currently capped at 32

## 5. from `ca52c317` — Single-CTA exact 5x512 fast path for ≤512 rows eliminates the second launch and partial buffer, while the large path use
Higher ceiling: replace the large path with a single-pass exact 5×512 CTA (buffer grad_norm and normalized in registers, no padded reductions), raise MAX_BLOCK_M to 128, and capture the whole two-kernel sequence in a persistent per-shape CUDA graph if the timed reps use stable input pointers.

## 6. from `24f62c4d` — Lower SMALL_ROWS_MAX 512→64 (single-CTA path is bandwidth-starved; multi-CTA large path is faster down to ~20 rows), rai
Higher-ceiling reserve: rewrite as a single-kernel CUDA persistent grid with cooperative-groups grid.sync() — each CTA computes dx + gw partial for its rows, then the last block reduces all partials into grad_weight in a single launch. Eliminates the second kernel launch entirely. Trigger: if this kernel only ties 0.716 or the reduce-kernel launch overhead is still the measured bottleneck on small shapes (B×S < 4096).

