# B200 / Blackwell Kernel-Engineering Knowledge Base

Reference material for writing maximally optimized kernels (GEMM, attention,
elementwise, reductions, quantized ops) on the NVIDIA B200 for SOL-ExecBench.

Built 2026-07-04 from four multi-source deep-research passes:
- **Hardware/architecture**: 20 sources, 100 claims, 25 verified
  (20 confirmed / 5 refuted), 10-finding synthesis.
- **Kernel-implementation space**: 24 sources, 120 claims, 25 verified
  (19 confirmed / 6 refuted), 12-finding synthesis.
- **Chinese ecosystem & papers**: 25 sources, 115 claims, 25 verified
  (23 confirmed / 2 refuted), 13-finding synthesis.
- **Optimization methodology**: 25 sources, 123 claims, 25 verified
  (24 confirmed / 1 refuted), 13-finding synthesis.

## Confidence legend

Every fact in these files carries a tag:

- ✅ **verified** — survived 3-vote adversarial verification against the cited source.
- ⚠️ **unverified** — extracted from the cited source but not individually
  adversarially re-checked (only the top 25 of 100 claims were). Treat as
  probably right; re-check before betting a design on an exact number.
- ❌ **refuted** — a claim that failed verification; kept only in
  `optimization-playbook.md` § "Refuted / suspect claims" so nobody
  reintroduces it.

## Files

### Hardware & architecture

| File | Contents |
|---|---|
| [b200-hardware.md](b200-hardware.md) | SMs, dies, memory hierarchy, occupancy limits, clocks, throughput tables |
| [tcgen05-and-tmem.md](tcgen05-and-tmem.md) | 5th-gen tensor cores: tcgen05 instruction family, Tensor Memory, CTA pairs |
| [hopper-to-blackwell.md](hopper-to-blackwell.md) | What carries over from H100 kernel practice and what breaks |
| [software-stack.md](software-stack.md) | CUDA/PTX versions, sm_100/sm_100a, CUTLASS, Triton/Gluon status |
| [optimization-playbook.md](optimization-playbook.md) | Rooflines, achieved-TFLOPS targets, pitfalls, pipeline patterns, refuted claims |

### Kernel implementation space

| File | Contents |
|---|---|
| [library-playbook.md](library-playbook.md) | Per-op-class decision table: cuBLAS(Lt), cuDNN frontend/graph API, CUTLASS 4.x, TE — and when to go custom |
| [compiler-dsl-landscape.md](compiler-dsl-landscape.md) | Triton, Gluon, CuTe-DSL, CuTile, torch.compile, ThunderKittens — Blackwell quality of each |
| [fusion-patterns.md](fusion-patterns.md) | Bytes-saved math, epilogue fusion, flash-style tiling, persistent/Stream-K, megakernels, backward strategies |
| [oss-kernel-catalog.md](oss-kernel-catalog.md) | FA4, FlashMLA, FlashInfer, DeepGEMM, quack, Liger, Mamba — B200-readiness + licenses |
| [quantized-kernels.md](quantized-kernels.md) | FP8/MXFP8/NVFP4 recipes, scale-factor layouts, TE/torchao status, Quant-collection guidance |

### Chinese ecosystem & research literature

| File | Contents |
|---|---|
| [chinese-oss-kernels.md](chinese-oss-kernels.md) | DeepGEMM (Mega MoE, UE8M0 scales), FlashMLA SM100 matrix, SGLang B200 stack, DeepEP + its PTX landmine |
| [sageattention-and-quant-papers.md](sageattention-and-quant-papers.md) | SageAttention 1/2/2++/3 + SpargeAttn recipes; the sm_120-not-B200 caveat |
| [chinese-community-insights.md](chinese-community-insights.md) | Zhihu tcgen05 worklog (98% of cuBLAS ladder), reed's CuTe series, TMEM/L2 deep-dives |
| [llm-kernel-generation.md](llm-kernel-generation.md) | KernelBench-family literature, reward-hacking catalog, design rules for the auto-research loop |
| [transferable-hopper-tricks.md](transferable-hopper-tricks.md) | DeepSeek H800 tricks → B200 applicability; verified transfers and anti-patterns |

### Methodology (how to work a problem)

| File | Contents |
|---|---|
| [optimization-recipe.md](optimization-recipe.md) | The master per-problem flowchart: classify → baseline → iterate loop → ladders → stop rules |
| [profiling-guide.md](profiling-guide.md) | Nsight Compute triage order, exact metric names, profiler-vs-reality gotchas |
| [benchmarking-discipline.md](benchmarking-discipline.md) | Timing/machine-state rules, trial counts, anti-Sakana correctness checklist |
| [autotuning-guide.md](autotuning-guide.md) | Tune spaces, model-pruned search, per-shape specialization, caching |

### Benchmark

| File | Contents |
|---|---|
| [benchmark-problems.md](benchmark-problems.md) | SOL-ExecBench problem taxonomy: op families, dtypes, collections, solver implications |
| [benchmark-grader.md](benchmark-grader.md) | The official grader (read from source): SOL score formula, cold-L2 timing, correctness/tolerance rules, input heuristics, reward-hack defenses |
| [solution-format.md](solution-format.md) | How a solution is submitted & run: the Solution schema, 9 languages (Python + C++ families), DPS, multi-file sources, the compile→eval pipeline, and the Trace result |
| [sources.md](sources.md) | All sources from all research passes with quality ratings |

## The 30-second summary

The B200 is a dual-die, 148-SM GPU (compute capability 10.0, `sm_100`) that
presents as a single CUDA device. For kernel authors the headline changes vs
Hopper are:

1. **`wgmma` is gone.** Hopper's warpgroup MMA does not run on Blackwell; peak
   tensor-core throughput requires the new `tcgen05.*` instruction family.
2. **Tensor Memory (TMEM).** A new 256 KB per-SM on-chip memory (128 lanes ×
   512 columns × 32-bit) holds MMA accumulators instead of registers. It is
   explicitly allocated/deallocated and has hard per-warp lane-access rules.
3. **Single-thread MMA issue.** `tcgen05.mma` is issued by one thread (like
   TMA), async, completed via mbarriers — no more warpgroup-collective issue
   or register-pressure fights for accumulators.
4. **CTA pairs (2-SM MMA).** Two adjacent CTAs in a cluster can cooperatively
   execute one MMA spanning both SMs' TMEM, doubling the M dimension and
   halving per-SM operand traffic.
5. **FP4/FP6 + microscaling (MX/NVF4) formats** are new, with block-scaled MMA
   variants at up to 4× Hopper's FP8 throughput.
6. **Occupancy math is Hopper-like** (64 warps/SM, 64K regs, 228 KB smem) but
   any TMEM-using kernel is limited to 1 CTA/SM ⚠️.

Roofline anchors (HGX B200, dense): ~2.25 PFLOPS BF16, ~4.5 PFLOPS FP8, ~9
PFLOPS FP4, 7.7 TB/s HBM3e ✅. Realistic achieved: cuBLAS BF16 ≈ 1763 TFLOPS
(~78% of peak) ⚠️; best hand-written kernels land at 95–98% of cuBLAS.
