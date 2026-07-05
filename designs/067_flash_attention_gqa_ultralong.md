# Problem 67 — 067_flash_attention_gqa_ultralong

Smoke-test design doc produced via `.claude/skills/design-kernel/SKILL.md`.
Source model: nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct.

## Op graph (from reference.py)

Full attention block, **all fp32**: 3 separate projections (Q: 4096→4096,
K/V: 4096→1024 each) → reshape/transpose (stride-only, free) → RoPE
(rotate-half, cos/sin are *inputs*) on Q and K → **GQA KV expansion 4×
materialized** via expand+reshape → S = Q·Kᵀ·scale → **full S×S causal mask
+ softmax materialized** → P·V → transpose+reshape → O-projection
(4096→4096).

Reference waste, in order of severity:
1. Materialized S×S attention matrix, fp32: at b1 s16384 that is
   32 heads · 16384² · 4 B ≈ **34 GB written+read twice** — catastrophic;
   flash-style fusion is mandatory, not optional, at long seq.
2. GQA expansion materializes K/V 4× (avoidable by head-index mapping).
3. RoPE materializes rotated copies + concat.
4. `weights are runtime inputs` — nothing (e.g. QKV weight concat) can be
   precomputed offline; any repacking cost lands on the measured clock.

## Shapes / dtypes / tolerances

18 workloads: b1–b64, s128–s16384. fp32 throughout. Tolerance
**atol ≈ 2.1–2.6e-3 (shape-dependent), rtol = 1e-5**. That atol is far above
fp32 noise — the grader clearly budgeted for reduced-precision matmul
(TF32-class error ~1e-3 relative). This is the single biggest design lever,
and also the biggest risk: with 4096-deep dot products, output magnitudes
~O(√4096·σ²) could make 2.6e-3 absolute tight. **Must be validated on GPU
before committing** (skill rule: ⚠️ assumption → open question).

## Roofline (per shape)

Two regimes:

- **Small s (128–512), small b**: AI 63–240; LB_mem ≈ 25–29 µs ≥ LB_tf32 —
  memory/launch-bound; projections dominate (weights alone are 168 MB read).
- **Large s (2k–16k) or large b**: AI 450–4950; TC-bound if TF32 is usable.
  LB_tf32: 188 µs (s2053) → 3.25 ms (s16384). If stuck on fp32 CUDA cores
  (75 TF): 2.8 ms → **47.6 ms** — a 15× gap. Everything hinges on the dtype
  question.

FLOPs split: projections ≈ 2·T·4096·10240; attention ≈ 2·s²·b·32·128
(causal ≈ half). At s≥2048 attention dominates; at s≤512 projections do.

## Candidates (ranked)

1. **cuBLAS projections + custom Triton flash-attention core (TF32
   `tl.dot`, fp32 accumulate, GQA-aware)** — effort M, expected 5–15× vs
   eager at long seq (S×S elimination + TC math), risk = tolerance
   (mitigation below). Design: 3 cuBLAS GEMMs (or one cuBLASLt call on
   pre-concatenated weights if the ~100 MB concat copy < GEMM savings —
   measure); small fused RoPE kernel writing Q,K in attention-friendly
   layout; flash kernel with online softmax, causal block skipping,
   `kv_head = q_head // 4` indexing (zero KV expansion); epilogue writes
   the [b,s,4096] layout directly so O-projection consumes it with no
   transpose kernel; O-projection via cuBLAS.
2. **Same structure, bf16 compute in the attention core** (inputs cast to
   bf16, fp32 accumulate) — effort M (same kernel, different dtype),
   ~2× the attention TFLOPS ceiling of TF32 (2.25 PF vs 1.1 PF). Strictly a
   tolerance gamble; keep as a config flag on candidate 1 and let GPU
   validation decide TF32 vs bf16 per the atol results.
3. **SDPA-based rewrite (no custom kernel)** — effort S, risk low:
   `F.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)`
   under torch.compile. Kills the S×S materialization and KV expansion via
   the mem-efficient backend (fp32-capable). Expect 2–4× vs eager at long
   seq but far from TC roofline. This is the fallback AND the bar for #1.
4. **Full-custom fused megablock (CuTe-DSL)** — fold RoPE into the flash
   prologue and O-proj into its epilogue. Only worth it if #1 profiles show
   the small kernels/launches dominating at small s. Effort L; defer.

Recommended: **#1 with #2 as a dtype flag**; #3 implemented first as
baseline (it's ~20 lines).

## Implementation plan (recommended)

- Kernels: [RoPE+layout] → [flash-attn core] + 4 cuBLAS calls (3 proj + O).
  5 launches total; at s≤293 consider merging RoPE into a Triton QKV-GEMM
  epilogue later (launch-bound regime).
- Flash core: BLOCK_M×BLOCK_N from {64,128}×{64,128}, fp32 softmax
  accumulators, causal early-exit per block-row, heads-parallel grid
  (b·32·⌈s/BLOCK_M⌉ CTAs — at b1 s128 only 32–64 CTAs → underfill; pick
  small BLOCK_M for small shapes via per-shape config).
- Numerics ladder per shape (decided by validation): fp32-dot → TF32-dot →
  bf16-dot, take the fastest that passes both atol and rtol on 2+ seeds.
- Autotune later: block sizes, num_warps, stages per the two shape regimes;
  weight-concat-vs-3-GEMMs decision.

## Open questions for GPU validation

1. Does TF32 (then bf16) attention+projection math stay within the
   per-shape atol? (Decides between a 15× and a 1× compute ceiling.)
2. What does eager reference actually cost at s8192/s16384 (OOM risk of the
   34 GB S matrix on the 180 GB card at b2 s8192 ≈ 17 GB ×2 — fits, but
   slow)? Sets the real speedup denominator.
3. cuBLAS fp32 GEMM on B200: does it internally use TF32/emulation (affects
   whether projections need explicit TF32 opt-in)?
4. Concat-weights-per-call vs 3 GEMMs crossover.
