# Playbook — task 9 · conv_9

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `1f68f8d9` — Eager torch, targeted TF32: run grad_gated (G2) + the 3 K=N wgrad GEMMs (G1/G3/G5) in TF32 (fp32 operands, 10-bit mantis
Higher-ceiling idea NOT shipped: a custom Triton wgrad kernel for G1/G3/G5 that
takes fp32/TF32 inputs but writes the (F,H)/(H,F) output ONCE as bf16 via a fused
cast epilogue — plus a Triton G2 that loads w2 as bf16 (117MB) and upcasts on-chip
to TF32 instead of the eager 234MB fp32 w2 read. Eager currently pays ~1GB of
avoidable traffic (three fp32 wgrad outputs round-tripped through .to(bf16) + the
w2 fp32 cast), which dominates the 13 memory-bound small-N shapes whose SOL is the
~0.10ms weight floor.

TRIGGER: if this passes all 16 (unlocks sol_score) but the memory-bound shapes
(N<=~300)

