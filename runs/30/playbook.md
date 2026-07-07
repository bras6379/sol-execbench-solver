# Playbook — task 30 · rmsnorm_30

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `152b0e65` — Single cuBLASLt GEMM via torch.addmm(beta=1.0) with the residual pre-loaded as C, fusing the matmul and residual add int
If this only matches baseline, the next ceiling is a shape-specialized Triton fused GEMM+residual kernel with smaller M-tiles (BM=64) and split-K to fill SMs on the underfilled small-M shapes, replayed under CUDA graph for the harness's repeated iterations. If wave-quantization tails persist on the largest shapes, escalate to a CUTLASS 4 SM100 persistent/Stream-K kernel with a TMA-fused residual epilogue.

