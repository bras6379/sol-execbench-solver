# Playbook — task 3 · layernorm_3

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `4c05cd72` — Hybrid LM-head GEMM: Triton N-tiled streaming kernel (each weight row read once, hidden L2-resident, fp32 accum, BLOCK_N
Higher-ceiling idea NOT shipped: extend the streaming kernel to the M=293
(non-power-of-2) skinny workload. Shipped version routes M>256 to torch, so
M=293 stays on cuBLAS at ~1.5x SOL. Trigger: if this beats the seed on the
M<=256 workloads but the mean is still dragged by M=293, split M=293 as
one BLOCK_M=256 block + a masked BLOCK_M=64 block along a 2D (pid_m,pid_n)
grid so the MMA waste is ~1.1x (not 1.75x from a single BLOCK_M=512 pad) --
accept the ~1.5x weight re-read across the two M-blocks (still << cuBLAS's
bandwidth gap) OR use a Stream-K / persistent scheme that keeps each weight
t

