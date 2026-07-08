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

## 2. from `c9dca185` — Measured frontier two-kernel Triton fusion with L2-friendly dV grid order, plus autotuned Skv<=256 full-row dS while kee
Higher ceiling not shipped: implement a CuTe/CUDA dS kernel that stages a BLOCK_Q x Skv row of P/mask and dP chunks in shared memory, accumulates the exact f32 delta, then writes dS without the second P/mask read. Try it if the Skv>256 workloads remain around the current 0.5-0.65 score band; keep delta from separate P and mask, and do not derive it from rounded P_drop.

## 3. from `7f1d48d8` — Two fused Triton kernels: single-pass full-row dS for Skv≤512, two-pass L2-resident dS for larger Skv; dV uses fast-dim
Higher ceiling not shipped: a CuTe/CUDA SMEM-resident single-pass dS kernel that explicitly stages a BLOCK_Q×Skv row of P/mask and dP chunks in shared memory, computes the full f32 delta once, then writes dS without the second P/mask read. Trigger if the large-Skv shapes (idx2 S=2048, idx11 S=1024, idx15 S=4096) still sit below ~0.75—these are the workloads where the Triton two-pass re-reads P/mask from HBM and Triton cannot hold mutable scratch across tile loops.

## 4. from `a29ca61a` — l2_persist eviction-policy hints on the 0.624 two-kernel Triton fusion: evict_last on P/mask (two-pass dS) and dO (dV),
Higher-ceiling not shipped: a CuTe/CUDA SMEM-resident single-pass dS kernel that stages a BLOCK_Q×Skv row of P/mask and dP chunks in shared memory, computes the full f32 delta once, then writes dS without the second P/mask read. Trigger: if the large-Skv shapes (idx2 S=2048, idx11 S=1024, idx15 S=4096) still sit below ~0.75 after this round's eviction hints — those are the workloads where the Triton two-pass re-reads P/mask from HBM even with evict_last, and Triton cannot hold mutable scratch across tile loops.

