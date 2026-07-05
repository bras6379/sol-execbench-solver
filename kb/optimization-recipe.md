# The Optimization Recipe (Master Flowchart)

The per-problem procedure for optimizing one kernel on B200. Built from
NVIDIA's documented process (APOD, Nsight guided analysis), practitioner
worklogs (Boehm, Lei Mao, salykova, GPU MODE), and competition/LLM-loop
evidence. Tags per kb/README.md (✅ 3-vote verified or high-confidence
synthesis · ⚠️ extracted, unverified).

## Step 0 — Classify the bottleneck BEFORE writing code ✅

Compute arithmetic intensity (AI) = FLOPs / bytes moved, from the problem's
`definition.json` shapes. Compare to B200 ridge points (dense):

| Regime | Ridge (approx) | Classification |
|---|---|---|
| BF16 tensor core | 2250e12 / 7.7e12 ≈ **292 FLOP/B** | AI ≪ 292 → memory-bound |
| FP32 (no TC) | 75e12 / 7.7e12 ≈ **10 FLOP/B** | most fp32 problems still memory-bound |
| FP8 / FP4 | 584 / 1168 FLOP/B | quant GEMMs are compute-bound at benchmark shapes |

Reference AIs ⚠️: residual add ≈ 0.17, ReLU ≈ 0.25, elementwise ≈ 0.08–0.5,
batchnorm ≈ O(10), matmul ≈ 0.167·N (fp32) — **the same op flips class with
shape; classify per-workload-shape, not per-operator** ⚠️. Caveat ✅: ridge
comparison doesn't detect *latency/launch-bound* kernels — if total runtime
is tens of µs, suspect overhead-bound first (see ladder C).

Third dimension on B200 ✅: for attention-class kernels, roofline must include
**SFU and SMEM ceilings**, not just FLOPs and HBM — B200 doubled tensor cores
but kept SFU (~16 exp/cycle/SM) and SMEM bandwidth (~128 B/cycle) flat, so
kernels compute-bound on H100 become SFU-bound (fwd softmax) or SMEM-bound
(bwd) on B200. FA4's fixes (polynomial exp on FMA units, softmax/MMA overlap,
~10× fewer rescales → 1605 TFLOPS) are the worked example.

## Step 1 — Establish the baseline chain

1. PyTorch eager (the reference itself) — the score to beat.
2. torch.compile (+ `mode='reduce-overhead'` for small shapes ⚠️).
3. Library call per [library-playbook.md](library-playbook.md) decision table.
4. Existing OSS kernel per [oss-kernel-catalog.md](oss-kernel-catalog.md).

Measure each per [benchmarking-discipline.md](benchmarking-discipline.md).
Only proceed to custom kernels if the best of these leaves a gap worth the
effort (Step 4 stop rules).

## Step 2 — The iterate loop ✅ (APOD Optimize phase, verbatim NVIDIA)

> baseline → profile → **one hypothesis** → apply → measure → keep/revert → repeat

- Profile first is a labeled **High Priority** recommendation ✅.
- Score memory-bound kernels by **achieved effective bandwidth**, not raw
  time ✅ ((Br+Bw)/time vs ~7.5 TB/s peak).
- Let Nsight's **Prioritized Rules** (ordered by Est. Speedup) pick the next
  move instead of hand-triaging counters ✅ — see
  [profiling-guide.md](profiling-guide.md).
- Keep/revert discipline: accept only changes ≥ ~1.05× measured ⚠️ (the
  profiling-guided LLM-loop paper's arbiter threshold); an automated loop
  without this oscillates among rewrites that change code but not perf ⚠️.
- **Time-box**: iteration shows strong diminishing returns — ~80% of wins in
  round 1 falling to ~20% by round 4; cap at ~8 iterations per approach ⚠️.

## Step 3 — Optimization ladders by bottleneck class

### Ladder A: MEMORY-BOUND (most norm/rope/elementwise/fp32 problems)

Ordered, with expected magnitudes from measured worklogs:

1. **Coalescing** ✅ — High Priority; warp accesses coalesce into minimum
   32-byte sectors (32-B sector confirmed unchanged on B200 ✅). Worklogs:
   ~6.5× on naive matmul (Boehm 15→110 GB/s; Lei Mao 6.4×) ⚠️.
2. **Vectorized 128-bit access** ✅ — float4 → LDG.E.128, 4× fewer load
   instructions; default-on per NVIDIA **unless register-limited** (raises
   register pressure) ✅; needs 16-B alignment; Blackwell adds 256-bit loads ✅.
   ~1.5× on a tiled GEMM ⚠️.
3. **Fusion** — the dominant lever for this benchmark's fused-op problems;
   bytes-saved math in [fusion-patterns.md](fusion-patterns.md). Includes
   dtype-shrinking (write bf16 intermediates, not fp32) ⚠️.
4. **Tiling/privatization/coarsening** ⚠️ — GPU MODE lecture-8 ordering;
   thread coarsening measured 10–11.5× on toy kernels; divergence removal ~3×.
5. Occupancy is a **latency-hiding mechanism, not a target** ⚠️ (multiple
   independent sources): 50–75% is usually fine; raising per-thread work
   (tiling → higher AI) beats chasing occupancy.
6. L2 persistence / grid-stride tuning — last resorts, shape-dependent.

Stop when achieved bandwidth ≈ 85–95% of the ~7.5 TB/s STREAM-real peak.

### Ladder B: TENSOR-CORE-BOUND (GEMM-shaped, attention, quant)

Design constraints first (all ✅ unless noted):

- Accumulators in TMEM (256 KB/SM); largest UMMA atom 128×256×16, its FP32
  accumulator = exactly half of TMEM → **double-buffer accumulators**.
- Single-thread MMA issue; full warpgroup only for epilogue (per-warp ¼-TMEM
  lane limit) — warp-specialization shape differs from Hopper.
- 2-SM UMMA supports **M ∈ {128, 256} only**, accumulator always split across
  the CTA pair along M — constrains the tile/cluster tune space.
- ❌ Refuted nuance: 2-SM UMMA does **not** halve total operand movement /
  double arithmetic intensity (0-3 vote). What's real (pass-1 ✅): the B
  operand is shared so each SM's SMEM loads half of B. Don't credit CTA pairs
  with more than that.
- TMA multicast: N-CTA multicast group cuts TMA loads of a shared tile by N
  — but per group along one cluster dim (2×2 cluster → 2× per operand, not
  4×), and it saves L2 traffic more than DRAM traffic (medium ✅).
- Keep M/N tiles ≥ 128 (systolic behavior); 3-stage TMA→MMA→epilogue overlap;
  persistent kernel + (Stream-K when wave quantization bites) — see
  [fusion-patterns.md](fusion-patterns.md) and
  [optimization-playbook.md](optimization-playbook.md) checklist.
- Ladder within the class ⚠️ (siboehm/salykova/Lei Mao, pre-TC but the
  structure transfers): tiling hierarchy → register/thread tiling (the
  biggest single step, ~3–5×) → vectorized smem access → bank-conflict
  padding (e.g. LD 128→132) → async copies → double buffering → autotune
  (last ~5–15%).
- For block-scaled NVFP4/MX GEMM ✅: stock CuTeDSL example is "far from
  optimized"; the **GPU MODE × NVIDIA Blackwell competition winning entries**
  are the documented performance reference — mine them before writing.

### Ladder C: LATENCY/OVERHEAD-BOUND (small kernels, decode shapes)

⚠️ (unverified tail of the research — treat as leads): CUDA graphs /
`reduce-overhead` (measured 1.81× e2e at batch-1 NVFP4); kernel merging;
custom launch paths (~3× launch-overhead cut in AMD-challenge winner, 120→40
µs); PDL (programmatic dependent launch); persistent megakernels
([fusion-patterns.md](fusion-patterns.md)). If per-call time is <10 µs, also
distrust your measurements (see
[benchmarking-discipline.md](benchmarking-discipline.md)).

## Step 4 — Abstraction ladder + stop rules

Routing per [compiler-dsl-landscape.md](compiler-dsl-landscape.md). Evidence-
based stop rules:

- Memory-bound op & Triton already near terminal bandwidth → **stop; lower
  DSLs gain nothing** ✅ (PyTorch/CuTeDSL-backend blog: lower levels pay off
  only for tensor-core-bound ops).
- Standard primitive (sort/scan/reduce/histogram) → **use CUB and stop** ✅ —
  CUB beat the next-best GPU MODE submission by 2–4×, including on B200.
- Achieved ≥ ~90% of the applicable roofline ceiling (bandwidth or TC) →
  polishing is low-EV; move to the next problem.
- Library GEMM ≥ ~75–80% of dense peak and epilogue is a small fraction of
  runtime → don't hand-write the GEMM; fuse the epilogue or stop.
- Headroom concentrates in **small-M/decode and tall-skinny shapes** (CuTeDSL
  backend: up to 1.73× over ATen at M=8–64, parity at large shapes) ⚠️ —
  spend custom-kernel effort there, not on big square GEMMs.

## Step 5 — Where wins actually come from (allocate effort accordingly)

Convergent evidence: **structural/algorithmic changes dominate parameter
tuning.**

- Boehm: blocktiling steps 12.8%→68.7% of cuBLAS; autotuning ~5–8% ⚠️.
- Profiling-guided LLM loop: biggest wins were adding `@triton.autotune`
  (95× over unconfigured baseline) and algorithmic restructuring (broadcast→
  tile-wise reduction, 1.74×); counter improvements ≠ wall-clock gains ⚠️.
- AMD challenge winner: comm/compute overlap, fusion, topology-aware
  remapping — not micro-tuning ⚠️.
- CUB sort: library/algorithm choice 2–4× > micro-optimization ✅.

Priority order for any problem: (1) right algorithm/library, (2) right
fusion, (3) right tile/pipeline structure, (4) parameter autotune
([autotuning-guide.md](autotuning-guide.md)), (5) micro-tuning.
