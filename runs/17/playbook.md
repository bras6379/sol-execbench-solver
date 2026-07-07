# Playbook — task 17 · elementwise_17

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `ffda16b6` — Use bf16 cuBLAS matmuls for all six fixed GEMMs, with Triton kernels fusing the SwiGLU backward pointwise chain and fina
Higher ceiling: replace the cuBLAS chain with a CUTLASS/CuTe SM100 persistent fused pipeline that computes grad_intermediate, applies SwiGLU backward in the GEMM epilogue, and feeds grad_x plus the two Wgrad reductions without materializing grad_gate/grad_up. Try it if this submission only matches PyTorch/cuBLAS latency or if small-token workloads are launch-bound.

