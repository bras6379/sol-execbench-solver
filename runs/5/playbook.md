# Playbook — task 5 · attention_5

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `a049e306` — 2D F.linear (torch.addmm fused-bias path) + expanded CUDA-graph gate M<=2560 (covers M=2048/2164) + clone-safety + extra
If this only ties the baseline (~0.757), the remaining gap is the BCx round-trip: GEMM1 writes (M,3H) to HBM and the middle kernel reads it back. The next lever is a persistent megakernel that tiles along the M dimension, keeps BCx and y in SMEM/TMEM, and does GEMM→gate→conv→gate→GEMM in one launch with zero HBM intermediates. Trigger: still launch/weight-BW bound at M=256 (decode-like shapes) after this round. Concrete approach: CUTLASS 4.5 persistent kernel with two tcgen05 GEMM stages + a CUDA-core middle epilogue, 2-SM CTA pairs, tile M=128 with a 3-token causal halo in SMEM.

## 2. from `e987d5d9` — cuBLAS in_proj/out_proj GEMMs + int32-indexed fused Triton middle (gate→causal-conv K=4→gate, pre-laid-out for GEMM2) +
# Handoff — reserve play (higher ceiling, NOT shipped this round)

This round shipped only a safe polish (int32 offsets in the memory-bound middle
kernel) on top of the proven cuBLAS+Triton-middle+CUDA-graph structure. The real
remaining lever is untouched: **the BCx HBM round-trip.** At M=8192 GEMM1 writes
BCx (M,6144)=~100 MB and the middle kernel reads all ~100 MB back — ~29 µs of
avoidable traffic on top of ~122 µs of GEMM, i.e. the whole ~15% large-M gap to SOL.

**Next kernel to try (highest ceiling): fuse the middle INTO out_proj as one
CUTLASS/CuTe-DSL prologue-GEMM, keeping cuBLAS-qua

