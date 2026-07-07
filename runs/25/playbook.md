# Playbook — task 25 · attention_25

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `270e508f` — Fused Triton element-wise kernel: 1 read + 1 write (10.5x fewer bytes vs 9-op reference), autotuned per-shape for BLOCK_
If Triton BW < 85% peak on memory-bound shapes (score < 0.8), escalate to custom CUDA: float4 vectorized LDG.E.128/STG.E.128 (4x fewer memory instructions), `__tanhf()` intrinsic (bit-exact SFU path matching reference `torch.tanh`), persistent grid-stride launch sized to 148 SMs with 8-way warp coarsening (8 elements/thread, 2x float4 per iteration). Trigger: achieved HBM bandwidth below 6.4 TB/s on the 134M-element shapes.

