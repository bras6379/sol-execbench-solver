# Playbook — task 32 · softmax_32

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `ea0086f0` — Correctness-first layout fusion: use cuBLAS bmm into final interleaved strides for B<=4 shapes and a direct-output Trito
Higher-ceiling reserve: replace the high-batch Triton path with a CUTLASS/CuTe SM100 grouped GEMM or persistent direct-layout kernel that uses the final `(B,S,40,128)` C strides natively. Try it if this only ties the baseline, especially on the B=8/16/32/64 power-of-two shapes where Triton matmul quality may lag cuBLAS.

