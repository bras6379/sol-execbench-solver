# Playbook — task 13 · layernorm_13

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `3b40d0a8` — Triton implementation: full-row RMSNorm-backward dx kernel with non-atomic tiled grad_weight reductions, plus a one-laun
Higher-ceiling reserve: replace the Python/Triton split with a C++ CUDA persistent kernel that computes row means and bf16 dx while accumulating per-CTA grad_weight in shared memory, followed by a CUB-style block reduction that writes grad_weight once. Try it if this is correct but still trails the PyTorch baseline or if the three-launch large-row path is the measured bottleneck.

