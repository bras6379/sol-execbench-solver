# Playbook — task 44 · gemm_44

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `0a6c1b78` — Counting-sort routing (argsort+scatter_add+cumsum+searchsorted, no host sync) into a BLOCK_M=64-padded expert-major layo
Untried higher-ceiling idea: fuse all 3 GEMMs (gate+up+SiLU+mul+down) into one persistent Triton megakernel that keeps the (BLOCK_M,2048) SwiGLU intermediate in registers/smem and never round-trips `intermediate`/`down_out` through HBM (DESIGN.md approach 1, DeepGEMM Mega-MoE style) — estimated ~4% of the 22.5GB weight-streaming floor today from those two buffers' writes+reads, so worth it once this round's 3-kernel version is confirmed correct and its achieved-BW% is measured. Trigger: if this round scores well below the ~3.5ms floor (score << ~0.85-0.9) with correctness passing.

Note on the

