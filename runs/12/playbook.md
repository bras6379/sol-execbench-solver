# Playbook — task 12 · softmax_12

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `e9827f4b` — Triton fused single-kernel: loads each fp32 frequency once, computes cos+sin+scale+bf16-cast in registers, writes each v
If bandwidth <80% peak at large shapes (B≥4, S≥16384), escalate to CUDA: use `__sincosf` intrinsic for true SFU fusion (one instruction vs two), `uint4` 256-bit loads (Blackwell-native), `__nv_bfloat162` packed stores, and a persistent grid of 148 SMs with grid-stride loop. Trigger: achieved HBM bandwidth below 7 TB/s on the largest workload shapes — means Triton's codegen left a measurable gap.

