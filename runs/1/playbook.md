# Playbook — task 1 · rope_1

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `ced4eee4` — Frontier bodies verbatim (exact f32 delta, ratio 1.0) + swap dV grid to (Skv/BLOCK_J, B*KVH) so kv-block is the fast dim
This round shipped the SAFE, math-identical dV grid swap `(Skv/BLOCK_J, B*KVH)`
(kv-block fast → each group's dO stays L2-resident) to kill the HBM thrash that
pinned the largest-batch shape (idx5, ~B8/S4096) at ~0.50. First check the
per-shape breakdown: if idx5 (and other big-B shapes) jumped but the big
LOW-HEADROOM shapes (idx15 S=4096, idx2 S=2048, idx11 S=1024; base/sol only 4–6×,
most score-sensitive) are still stuck, the remaining ceiling is the dS-path
single-pass.

HIGHER-CEILING PLAY (not shipped): a CuTe/CUDA SMEM-resident single-pass dS
kernel that holds the (BLOCK_Q × Skv) row's

