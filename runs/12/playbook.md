# Playbook — task 12 · softmax_12

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `e9827f4b` — Triton fused single-kernel: loads each fp32 frequency once, computes cos+sin+scale+bf16-cast in registers, writes each v
If bandwidth <80% peak at large shapes (B≥4, S≥16384), escalate to CUDA: use `__sincosf` intrinsic for true SFU fusion (one instruction vs two), `uint4` 256-bit loads (Blackwell-native), `__nv_bfloat162` packed stores, and a persistent grid of 148 SMs with grid-stride loop. Trigger: achieved HBM bandwidth below 7 TB/s on the largest workload shapes — means Triton's codegen left a measurable gap.

## 2. from `ff735bec` — Row-specialized Triton RoPE kernel: load each contiguous 64-float row once, compute cos/sin/scale in one launch, cast to
If this only ties the prior Triton frontier or large-shape bandwidth stays below roughly 7 TB/s, switch to a CUDA kernel using `__sincosf`, packed `__nv_bfloat162`/vectorized stores, and a 148-SM persistent grid-stride loop over rows to cut Triton codegen and launch-shape overhead.

## 3. from `00e3ba87` — Triton fused flat RoPE kernel (frontier-winning layout) with an expanded autotune superset — adds large BLOCK_SIZE 4096/
Higher-ceiling idea NOT shipped: a **fixed-grid cuda_cpp kernel that does NOT over-spread**. The prior CUDA attempt (75ffe60e, 0.8986 raw) lost only because it *forced* grid>=148 blocks (shrinking block to 32) on the tiny shapes, adding block-scheduling overhead — the exact opposite of what these SOL~0.5us / L2-resident workloads want. A CUDA kernel that instead uses a natural `grid = cdiv(elems, block)` with a per-shape block heuristic matching Triton's autotune winners (few large blocks for tiny shapes, more blocks only for idx4/B=64), plus `__nv_bfloat162` packed 128-bit stores and accurate

## 4. from `996a46d3` — Fixed-grid CUDA RoPE: 2 freqs/thread via float2 load + packed __nv_bfloat162 stores to both output halves, natural grid
Higher-ceiling idea NOT shipped: **4-freqs/thread** — one `float4` load + two 64-bit
packed stores (4 bf16 as a `uint2`/`int2`) per half, instead of this round's
2-freq `float2`+`bfloat162`. Trigger: if this CUDA kernel BEATS the Triton 0.894
frontier but the one DRAM-bound shape (idx4, B=64/S=541) still trails SOL, widen to
float4/int2 — it only helps that single memory-bound shape (~1%), tiny shapes are
launch-floor-limited and won't move.

FALLBACK if this round REGRESSES or fails to compile: the proven Triton flat kernel
(frontier e9827f4b, 0.894) is still the retained frontier and the nex

## 5. from `d3257b2c` — Replace FP64 range reduction (x - 2π*rint(x/2π)) with fp32 Cody-Waite (nearbyint + FMA) to shorten per-thread latency ch
4-freq/thread with float4 loads + int2 packed stores, using the same Cody-Waite reduction. The prior 4-freq CUDA attempt (0.000_6e334853) had a fatal index bug: `base = (row << 6) | (lane << 1)` should be `base = (row << 5) | (lane << 1)` — the int2 stores index into 32 slots/row (128 bf16 / 4), not 64. Fix that, keep the grid-stride loop, and use the Cody-Waite reduction from this round. Trigger: if this round only ties 0.898 or the large shape (B=64/S=541) is the one workload holding the score down, the 4-freq approach halves per-element instruction count and could move the bandwidth-bound s

