---
name: design-kernel
description: Design an optimized B200 kernel for a SOL-ExecBench problem, starting from its PyTorch reference implementation. Use when asked to analyze a bench problem, propose a kernel approach, or turn a reference.py into candidate kernel designs. Covers analysis and design only — not GPU execution, validation, or the iterate loop.
---

# Design a Kernel for a SOL-ExecBench Problem

You are given a problem folder containing `definition.json` (op description,
axes, I/O shapes/dtypes), `reference.py` (PyTorch reference + input
generator), and `workload.jsonl` (the exact shapes and tolerances the
benchmark scores). Your output is a **design**: a bottleneck analysis, a
ranked list of 3–5 candidate approaches, and a concrete implementation plan
for the top candidates. You do NOT run anything on a GPU in this skill.

The knowledge base at `kb/` is the canonical reference. Load only the files
this recipe points you to — not all 21.

## Step 1 — Read the problem completely

Read all three files. Extract and write down:

1. **The op graph**: read `reference.py` line by line and list every
   operation in order (matmuls, softmaxes, norms, elementwise chains,
   reshapes/transposes, scatter/gather, dropout, dtype casts). Note which
   intermediates are materialized. Reshapes/transposes that only change
   strides are free candidates for folding into adjacent ops.
2. **Every workload shape** from `workload.jsonl` — not just the axes from
   `definition.json`. The benchmark scores fixed shapes; the design must
   target them specifically (compile-time constants, per-shape dispatch).
3. **Dtypes**: fp32-only → tensor cores are irrelevant beyond TF32, this is
   a bandwidth/fusion problem. bf16 → tensor-core game for matmul-shaped
   parts. fp8/fp4 (+ scale tensors) → block-scaled tcgen05 territory; read
   the reference's scale handling exactly (granularity, transpose behavior,
   where dequant happens).
4. **Forward or backward**: `_backward` problems need chain-rule fusion,
   recompute-vs-save decisions, and wgrad reduction strategy (atomics vs
   deterministic split — check tolerances).
5. **Tolerances** from `workload.jsonl` (full grader in
   [../../../kb/benchmark-grader.md](../../../kb/benchmark-grader.md)): each
   workload has `max_atol`, `max_rtol`, **`required_matched_ratio` (default
   0.99 — only 99% of elements must pass; a few outliers are tolerated
   unless `max_error_cap` is set)**, `max_error_cap`, `allow_negative_inf`.
   Tight bands forbid aggressive-precision tricks (quantized intermediates,
   FP16 accumulation, nondeterministic atomics); loose ones open them up.
   Read the atol/rtol *pattern*, not just the values: an atol far above the
   dtype's noise floor (e.g. atol ~2e-3 on an fp32 problem), especially
   shape-dependent atol, means the grader budgeted for reduced-precision
   math — TF32/bf16 compute is probably intended and is likely the problem's
   main lever. Hard fails regardless of tolerance: any inf/nan in the output,
   or an all-zeros output when the reference is non-zero. Still treat the
   precision question as an open item for GPU validation, never a design
   certainty.
6. **Semantics traps**: dropout (RNG reproducibility requirements?), masking
   conventions, GQA head-expansion (`repeat_kv` is often avoidable — index
   instead of materialize), padding vs varlen (`cu_seqlens`).
7. **What is an input vs a constant**: everything in `inputs` (including
   weights and cos/sin tables) arrives fresh on every measured call —
   nothing can be precomputed offline. Any repacking/concat/layout
   conversion of inputs costs measured time; weigh it explicitly (e.g.
   concatenating QKV weights to enable one GEMM may or may not pay).
8. **Input distributions are structured, not noise**: the grader generates
   `random` inputs with role heuristics (valid softmax outputs, real RoPE
   cos/sin, causal masks, norm weights, positive tensors, fp8/fp4 clamped
   draws — see [../../../kb/benchmark-grader.md](../../../kb/benchmark-grader.md)).
   You may rely on those properties (a "softmax output" input sums to 1),
   but must be correct for any seeded draw of the shape.

## Step 2 — Roofline classification, per shape

Follow `kb/optimization-recipe.md` Step 0. For EACH workload shape compute:

- Total FLOPs and minimum bytes moved (inputs read once + outputs written
  once; intermediates count only if they must materialize).
- Arithmetic intensity vs B200 ridge points: **BF16 TC ~292 FLOP/B, FP8
  ~584, FP4 ~1168, FP32 ~10**. Memory peak 7.7 TB/s (plan for ~80–90%
  achievable). 148 SMs.
- **Lower-bound latency**: max(bytes / 7.0 TB/s, FLOPs / dense-peak). Then
  cross-check against the **authoritative SOL target in `metadata.json`**
  (`sol.per_workload[i].sol_ms`) if present — that is the grader's own
  Speed-of-Light number; use it as the north star and treat your hand
  estimate as a sanity check. The grader also gives the baseline `T_b`
  (`baseline_latency_ms`): the score is
  `S = 1/(1 + (T_k−T_SOL)/(T_b−T_SOL))`, so **beating `T_b` earns S > 0.5**
  and closing the gap to `sol_ms` earns the rest (see
  [../../../kb/benchmark-grader.md](../../../kb/benchmark-grader.md)). Rank
  effort by the `T_b`→`sol_ms` gap per shape.
- Classify each shape: memory-bound / tensor-core-bound / **SFU-bound**
  (attention-class: count exp ops — B200 SFU does ~16 exp/cycle/SM vs ~8192
  TC ops/cycle, so softmax-heavy kernels bottleneck on exp) /
  launch-overhead-bound (total work so small the lower bound is <20 µs).
- The same op can flip class across shapes — if it does, plan per-shape
  strategies, not one kernel.

Also estimate the reference's waste: count the HBM round-trips the PyTorch
implementation makes (each unfused op = read inputs + write output). The gap
between reference-bytes and minimum-bytes is the fusion prize.

## Step 3 — Prior-art sweep (before designing anything)

In this order, and stop early when something fits:

1. **Library call** — check the decision table in `kb/library-playbook.md`:
   cuBLASLt fused epilogues (bias/GELU/dgrad), cuDNN graph API SDPA (incl.
   backward d≤256, softcapping via pointwise DAG, varlen), cuDNN MoE grouped
   GEMM (+SwiGLU/+Wgrad/+quantize), CUTLASS 4.5 grouped GEMM, CUB
   (sort/scan/reduce — beats hand-written 2–4×), cuFFT (Hyena).
2. **OSS kernel to port/adapt** — check `kb/oss-kernel-catalog.md` and
   `kb/chinese-oss-kernels.md`: FlashAttention-4 (feature list covers GQA,
   varlen, KV-cache+rope, softcapping), FlashMLA (mind the support matrix:
   dense decode is SM90-only), DeepGEMM (FP8/FP4, grouped, Mega MoE),
   FlashInfer, quack (norms/softmax/CE fwd+bwd in CuTe-DSL), Liger (Triton
   fwd+bwd), NVIDIA's open SDPA bprop kernels in cudnn-frontend, official
   Mamba/causal-conv1d. All permissively licensed. Verify B200 status in
   the catalog — FA-3 and anything `compute_90a`/wgmma does not run.
3. **Recipe from papers** — for quantized attention, `kb/sageattention-and-
   quant-papers.md` (recipes port; the kernels do NOT — SA3 is sm_120-
   validated). For quant GEMM scaling schemes, `kb/quantized-kernels.md`.
4. Only if nothing above covers the op graph (or covers it with obvious
   waste left): design a custom kernel.

A port with shape-specific tuning usually beats a from-scratch kernel on
both time and risk. "Custom" often means: OSS/library core + custom fused
prologue/epilogue.

## Step 4 — Choose the abstraction

Use `kb/compiler-dsl-landscape.md` routing:

- Memory-bound fusion → **Triton** (fast to write, near-terminal bandwidth;
  lower DSLs gain nothing here — verified stop rule).
- Tensor-core-bound / attention / block-scaled quant → **CuTe-DSL or CUTLASS
  C++** (the proven sm_100 path: FA4, quack, NVIDIA's own kernels), or start
  from a CUTLASS SM100 example (72b for NVFP4).
- Triton close but leaving tcgen05/TMEM unused → **Gluon**.
- Plain dense GEMM with no fusion opportunity → don't write one; cuBLAS
  wins (Triton reaches only 62–70% of it on B200).
- Raw CUDA/PTX only when the design demonstrably needs something the DSLs
  can't express (consult the Zhihu tcgen05 worklog details in
  `kb/chinese-community-insights.md` before going there).

## Step 5 — Design the kernel(s)

### Memory-bound design checklist
- Fusion plan: which ops merge into one kernel; target = minimum-bytes from
  Step 2. Recompute cheap values instead of storing (B200 affords ~290
  FLOP/byte of recompute — it's free).
- Coalesced, 128-bit (or 256-bit) vectorized access; watch register
  pressure. Contiguous innermost dim; fold transposes into indexing.
- One row/CTA-sized reduction pattern per norm/softmax (Welford / online
  softmax when fusing across a normalization barrier).
- Grid sized to saturate 148 SMs at the actual shapes; grid-stride loops for
  small shapes.
- Backward: fuse dgrad chains into one kernel where the forward was fused;
  choose deterministic vs atomic reductions from tolerances.

### Tensor-core-bound design checklist (`kb/tcgen05-and-tmem.md` + `kb/optimization-playbook.md`)
- Tiles M,N ≥ 128 (128×128 systolic behavior; M=64 runs ~half rate).
  Largest UMMA atom 128×256×16; its fp32 accumulator = half of TMEM →
  double-buffer accumulators.
- 3-stage overlap: TMA→SMEM, tcgen05→TMEM, epilogue TMEM→GMEM. Single-thread
  MMA issue; 4-warp epilogue (TMEM lane restriction). TMEM kernels get
  1 CTA/SM — latency hiding must come from async pipelining, not occupancy.
- CTA pairs (2-SM, M∈{128,256} only) when B-operand reuse justifies it —
  expect ~8% not 2× (it halves per-SM B smem traffic, nothing more).
- Persistent kernel + tile scheduler; Stream-K if wave quantization shows.
- Fuse the epilogue (bias/activation/residual/quantize-output) — never a
  separate elementwise kernel after a custom GEMM.
- Attention-class: budget exp — pipeline softmax against MMAs, keep exp off
  the critical path (FA4 pattern: polynomial exp on FMA units, correction
  warpgroup, skip-rescale-unless-max-moved).

### Quantized design checklist (`kb/quantized-kernels.md`)
- Match the reference's scale granularity exactly first; only then optimize.
- NVFP4 = 16-elem blocks + ue4m3 scales via `mxf4nvf4`; scales need the
  128×4 (512 B) K-major basic-block layout, staged SMEM→TMEM via tcgen05.cp
  from the MMA warp. The scale-layout conversion is where the bugs live.
- FP8: cuBLASLt first; DeepGEMM-style (SM100 wants packed-UE8M0 scales) if
  granularity doesn't fit cuBLASLt. Two-level accumulation if accuracy
  drifts.
- Fuse output-quantization into the epilogue when the output feeds another
  quantized op.

### Launch-bound design checklist
- Merge kernels; CUDA graphs; persistent megakernel for whole-block (L2
  collection) problems — see `kb/fusion-patterns.md`.

## Step 6 — Write the design doc

Save it as `designs/<problem_name>.md` in the repo. Produce, per problem:

```markdown
## Problem N — <name>
- Op graph: ...
- Shapes × dtypes × tolerance summary; fwd/bwd
- Roofline: per-shape class, lower-bound latency, reference-waste estimate
- Candidates (ranked, 3–5):
  1. <approach> — expected gain vs reference (from lower bound), effort
     (S/M/L), risk (correctness traps, unverified assumptions), abstraction,
     prior art to reuse (file/repo pointers)
  2. ...
- Recommended: #1 (+ #2 as hedge if #1 has a risky assumption)
- Implementation plan for recommended: kernel count, fusion boundaries,
  tile/block plan per shape, epilogue plan, what to autotune later
- Open questions to resolve on GPU (assumptions the design bets on)
```

Rank candidates by expected-value: (potential gain × probability it works) /
effort. Always include one low-risk candidate (library call or
torch.compile) even if unexciting — it's the fallback and the baseline the
fancy design must beat.

## Rules

- Never trust the reference's structure as optimal — it's the *semantics*
  spec, not the implementation spec. Anything producing the same outputs
  within tolerance is legal.
- Never design around a ⚠️-tagged KB number without flagging it as an
  open question in the design doc.
- Check `kb/optimization-playbook.md` § "Refuted / suspect claims" before
  using any performance folklore (e.g., EVT-on-SM100, 11-cycle tcgen05
  latency, INT8-faster-than-FP8 are all refuted/false).
- Small kernels: if lower-bound latency < 20 µs, launch overhead dominates —
  design for kernel-count reduction before micro-optimizing.
- If two candidates are within ~10% expected gain, prefer the one that's
  simpler to validate.
