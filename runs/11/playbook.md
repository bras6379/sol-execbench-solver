# Playbook — task 11 · rope_11

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `28b87eae` — Fused Triton RoPE computes the first batch's BF16 cos/sin table once and returns an expanded batch view because generate
If the grader rejects non-contiguous expanded outputs or this only ties the baseline, switch to a CUDA/C++ materializing kernel that uses __sincosf per (seq,freq), packs each cos/sin BF16 pair into one 32-bit store, and vectorizes the duplicated h and h+64 writes across all batches.

## 2. from `1bed36f1` — Fused Triton RoPE: compute batch-0 cos/sin once (all batches share arange positions), pack each cos/sin bf16 pair into o
Higher-ceiling idea NOT shipped: a CUDA/C++ kernel using `__sincosf` (computes cos+sin in one SFU sequence, ~halving transcendental cost) with 128-bit vectorized packed stores of the already-int32-packed pairs. Trigger: if this Triton packed-store version is SFU-bound on large prefill S (raw score stalls below ~0.75 and profiling shows SFU, not HBM, as the limiter) or only marginally beats 0.669, drop to CUDA `__sincosf` to cut the cos/sin cost in half.

## 3. from `b8cf3de6` — Fused Triton RoPE: compute batch-0 cos/sin once (all batches share arange positions) and .expand() for free; pack each c
HIGHER-CEILING PLAY NOT SHIPPED: hand-rolled shared-range-reduction sincos in
Triton (pure arithmetic, no libdevice sincos exists). Compute the angle once,
do ONE Cody-Waite range reduction (reduce by pi/4, 3-term DP1/DP2/DP3 as in
Julien Pommier's sincos_ps / Cephes sinf; max angle here is ~256 rad so 3-term
fp32 CW is accurate to ~1e-6), then evaluate the sin AND cos minimax polynomials
on the same reduced argument and pick/sign by octant. This shares the expensive
range reduction between cos and sin, ~halving the FP32-pipeline transcendental
cost that limits the batch=1 large-S shapes (1,3,

## 4. from `65e7371f` — Shared Cody-Waite range reduction sincos in Triton: replace two independent tl.cos/tl.sin calls (each does full range re
Higher-ceiling idea NOT shipped: CUDA __sincosf kernel (hardware sincos in one SFU sequence, zero polynomial overhead). Write kernel.cu — the compile server handles .cu (load_inline is blocked at eval). The CUDA __sincosf intrinsic does the shared range reduction in hardware with a single SFU instruction, avoiding the polynomial evaluation and octant selection overhead of the software CW approach. Trigger: if this Triton kernel scores below 0.80 and nsys shows >30% of kernel time in the polynomial evaluation (not the range reduction), the CUDA path removes that overhead entirely.

