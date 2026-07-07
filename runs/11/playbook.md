# Playbook — task 11 · rope_11

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `28b87eae` — Fused Triton RoPE computes the first batch's BF16 cos/sin table once and returns an expanded batch view because generate
If the grader rejects non-contiguous expanded outputs or this only ties the baseline, switch to a CUDA/C++ materializing kernel that uses __sincosf per (seq,freq), packs each cos/sin BF16 pair into one 32-bit store, and vectorizes the duplicated h and h+64 writes across all batches.

## 2. from `1bed36f1` — Fused Triton RoPE: compute batch-0 cos/sin once (all batches share arange positions), pack each cos/sin bf16 pair into o
Higher-ceiling idea NOT shipped: a CUDA/C++ kernel using `__sincosf` (computes cos+sin in one SFU sequence, ~halving transcendental cost) with 128-bit vectorized packed stores of the already-int32-packed pairs. Trigger: if this Triton packed-store version is SFU-bound on large prefill S (raw score stalls below ~0.75 and profiling shows SFU, not HBM, as the limiter) or only marginally beats 0.669, drop to CUDA `__sincosf` to cut the cos/sin cost in half.

