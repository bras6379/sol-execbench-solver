# Problem 69 — 069_rms_norm (fused residual + RMSNorm)

Smoke-test design doc produced via `.claude/skills/design-kernel/SKILL.md`.
Source model: nvidia/Llama-3.3-Nemotron-70B-Select.

## Op graph (from reference.py)

`x = residual + hidden_states` (bf16) → cast fp32 → `var = mean(x², -1)` →
`x * rsqrt(var + eps)` (fp32) → cast bf16 → `weight * (·)` (bf16×bf16).
Intermediates materialized by eager: x (bf16), x_fp32, variance, x_normalized
(fp32!), bf16 cast, output — ~5 extra HBM round-trips, two of them fp32
(2× bytes). Reference traffic ≈ 26 B/element vs minimum 6 B/element →
**~4× fusion prize**.

Semantics notes: reduction in fp32 (keep); final multiply is bf16×bf16 after
the downcast — mimic the cast order only if tolerances bite (they won't:
rtol=0.05 is loose, atol=1e-5).

## Shapes / dtypes / tolerances

16 workloads, H=8192 const, tokens (b·s) from 131 to 8192. All bf16 (+ fp32
eps scalar). Tolerance atol 1e-5 / **rtol 0.05** — loose; numerics are not a
constraint beyond fp32 accumulation of the mean.

## Roofline (per shape)

Pure memory-bound (AI < 1 FLOP/B). Min bytes = 6·n + 16 KB weight.
Lower bounds at 7.0 TB/s: 57.5 µs (8192 tokens) down to **0.9 µs (131
tokens)**. Ten of 16 shapes have LB < 20 µs → **launch-overhead territory**;
three are sub-4 µs, where the launch itself (~3–5 µs) is the floor and
measurement noise exceeds kernel time.

## Candidates (ranked)

1. **Single fused Triton kernel** — gain ~3–4× vs eager on large shapes,
   effort S, risk minimal. One row (H=8192) per CTA: load hs+res as 8-wide
   vectorized bf16, add, accumulate Σx² in fp32 (one pass, keep x in
   registers/smem — 8192 fp32 = 32 KB, fits), rsqrt, scale, multiply by
   weight (cached in smem), store bf16. Row-count = tokens → 131-row shapes
   slightly under-fill 148 SMs; acceptable. Triton is terminal for
   memory-bound ops (verified stop rule) — no reason to go lower-level.
2. **Port quack RMSNorm / vLLM fused_add_rms_norm** — same expected perf as
   #1 (both claim near-peak bandwidth), effort S (adapt: vLLM's variant also
   rewrites residual in place; ours returns only `output`), risk minimal.
   Do this if #1 measures below ~85% of bandwidth roofline.
3. **torch.compile** — baseline/fallback; Inductor should fuse this whole
   chain into one kernel; expect within ~10–20% of #1 on big shapes, worse
   on tiny shapes (guard overhead).

Recommended: **#1**, with #3 measured first as the bar.

## Implementation plan (#1)

- One Triton kernel, `BLOCK_H` covering 8192 via 1–4 iterations
  (autotune num_warps 4–16, vector width). Compile-time H.
- Grid = (tokens,). fp32 accumulator; `tl.rsqrt`; weight loaded once per CTA.
- Nothing else to fuse — problem is a single fusion island by construction.
- Autotune later: num_warps, BLOCK size per the 4 token-count clusters
  (~131–512, ~1–2k, ~4k, ~8k).

## Open questions for GPU validation

- Whether the harness's timing floor makes the sub-4 µs shapes pure launch
  noise (if so, all correct implementations tie there; win the big shapes).
- Achieved GB/s vs the ~7.0–7.5 TB/s target on the 8192-token shapes.
