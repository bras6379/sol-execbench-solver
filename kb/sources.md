# Sources

Two research passes on 2026-07-04. Quality: primary = NVIDIA docs /
first-party author / peer-reviewed; secondary = reputable analysis; blog =
individual writeups.

## Pass 1 — hardware/architecture (20 sources)

## NVIDIA primary docs

- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) — smem/occupancy/L2/cluster limits (most ✅ facts). Notably does NOT document tcgen05/TMEM/SM counts.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) — tcgen05 semantics, TMEM structure/alloc/lane rules (✅ facts).
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) — CUDA 12.8, sm_100/sm_100a, PTX-JIT rules.
- [Blackwell Architecture page/whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture) — dual-die, NV-HBI, datatypes, Transformer Engine 2.
- [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) — SM100 schedules, narrow-precision types, alignment rules.
- [HGX B200 datasheet (PDF mirror)](https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf) — throughput/memory/power tables (✅; sparsity footnote).

## Papers

- [arxiv 2512.02189 (v1/v3) — B200 microbenchmarking](https://arxiv.org/html/2512.02189v3) — the correct B200 paper. Topology facts (148 SMs, 8 GPCs, dual-die) verified; but its tcgen05 latency and several throughput measurements **failed adversarial verification** — trust structure, re-check numbers.
- [arxiv 2507.10789 — "Dissecting Blackwell" ISA study](https://arxiv.org/pdf/2507.10789) — ⚠️ **benchmarks consumer GB203 (RTX 5080, sm_120a), NOT B200** ✅ verified: zero B200/TMEM/tcgen05 measurements (tcgen05 unsupported on sm_120a). Useful only for PTX-level rules (.kind::f8f6f4) and SASS naming (OMMA/QMMA). Never transpose its perf numbers to B200.

## Engineering writeups (first-party authors)

- [Tri Dao — FlashAttention-4 on Blackwell](https://tridao.me/blog/2026/flash4/) — FA4 numbers, exp-unit bottleneck, UMMA/TMEM usage.
- [HazyResearch — ThunderKittens on Blackwell](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell) and [TK2](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2) — dataflow model, TMEM double-buffering, fence removal, cluster-vs-SM-count table, 1 CTA/SM TMEM limit.
- [Triton Gluon tcgen05 tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/06-tcgen05.py) — operand placement, implicit pipelining, measured TFLOPS.
- [Colfax — CUTLASS GEMM with TMEM](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) — best CUTLASS-on-Blackwell walkthrough.
- [Modular — Matmul on Blackwell series](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction) — 3-stage pipeline, cuBLAS 1763 TFLOPS baseline. (Its 192 MB L2 figure conflicts with NVIDIA's 126 MB — trust NVIDIA.)
- [gau-nernst — hand-written tcgen05 matmul](https://gau-nernst.github.io/tcgen05/) — v1→v6 progression to 98% of cuBLAS; SMEM descriptor (LBO/SBO) details.

## Secondary analysis

- [SemiAnalysis — Blackwell perf/TCO](https://newsletter.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis) — B100/B200/GB200 SKU table, "30×" debunk.
- [SemiAnalysis — Tensor core evolution Volta→Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell) — MMA.2SM mechanics, MX formats, why smem didn't grow.
- [Chips and Cheese — B200 deep dive](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut) — 148/160 SM yield, per-SM MAC rates, FP16-vector regression.
- [0xsero Blackwell GPU wiki](https://0xsero.github.io/blackwell-gpu-wiki/blackwell/tcgen05-and-tmem/) — useful overview; contains at least one error (PTX ISA 8.4) — double-check anything sourced only here.

## Pass 2 — kernel-implementation space (24 sources)

### NVIDIA primary
- [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) — instruction throughput ratings, block-scale layouts, dtype tables.
- [CUTLASS example 72b — NVFP4×NVFP4 GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu) — the canonical sm_100 FP4 starting point.
- [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md) — 4.5.0 grouped GEMM, CuTe-DSL timeline.
- [cudnn-frontend repo](https://github.com/NVIDIA/cudnn-frontend) — open SDPA/MoE/norm kernels, SM100 backward support matrix (MIT).
- [cuDNN 9 transformers blog](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/) — graph-API SDPA fusion mechanics (H200-era numbers only).
- [cuBLAS 12.0 blog](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/) — epilogue surface, workspace/heuristics-cache advice.
- [Triton-on-Blackwell blog (NVIDIA/OpenAI)](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/) — capabilities + known gaps, Feb 2025.
- [Gluon tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html) — official positioning + GB200 examples.
- [TransformerEngine FP8 blockwise docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html) — DeepSeek-recipe emulation on Blackwell, transpose constraints.
- [PyTorch blog — MXFP8/NVFP4 diffusion on Blackwell](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/) — torchao paths, e2e speedups, deployment heuristics.

### Repos (first-party)
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — FA-4 = Blackwell (CuTe-DSL), FA-3 = Hopper-only, feature list, BSD-3.
- [Dao-AILab/quack](https://github.com/Dao-AILab/quack) — CuTe-DSL memory-bound kernels + Blackwell GEMM, Apache-2.0.
- [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) — B200 since v0.4.0, multi-backend dispatcher, FP8/FP4 GEMM+MoE, Apache-2.0.
- [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — SM100 FP8/FP4/BF16 + grouped + wgrad, ue8m0 packed scales on SM100, MIT.
- [FlashMLA SM100 kernels (deepwiki)](https://deepwiki.com/deepseek-ai/FlashMLA/5.2-sm100-kernels-(blackwell)) — secondary writeup of csrc/sm100 kernel structure.

### Engineering writeups
- [HazyResearch — megakernels ("no bubbles")](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) — Llama-1B single-kernel decode, B200 numbers.
- [Colfax — EVT tutorial](https://research.colfax-intl.com/epilogue_visitor_tree/) — epilogue-fusion mechanics; EVT was Hopper-only as of Oct 2024.
- [Colfax — persistent kernels & Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) — tile schedulers, wave quantization.
- [Colfax — Blackwell block scaling](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/) — mxf4nvf4 kinds, scale TMEM staging, CuTe-DSL example.
- [Horace He — gpus go brrr](https://horace.io/brrr_intro.html) — fusion bytes-saved arithmetic, recompute logic.
- [arxiv 2604.23466 — DSL benchmark study](https://arxiv.org/pdf/2604.23466) — cuBLAS vs Triton vs CuTile on B200 (BF16/FP16 only; FA-2 baseline).
- [Toulmé — CuTile on Blackwell](https://patricktoulme.substack.com/p/cutile-on-blackwell-nvidias-compiler) — codegen analysis; MLIR passes proprietary; no benchmarks.
- [Lei Zhang — Gluon explicit performance](https://www.lei.chat/posts/gluon-explicit-performance/) — Gluon vs Triton IR levels, explicit controls.
- [0xsero wiki — tcgen05/TMEM](https://0xsero.github.io/blackwell-gpu-wiki/blackwell/tcgen05-and-tmem/) — (also in pass 1; same caution applies).

## Pass 3 — optimization methodology (25 sources)

### NVIDIA primary
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) — SOL sections, metric naming, clock/cache-control defaults, roofline sections, Est. Speedup rules.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — APOD, High Priority recommendations, coalescing rule, timing rules.
- [Nsight roofline blog](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/) — arithmetic intensity / ridge-point method.
- [Vectorized memory access pro-tip](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) — LDG.E.64/128, register-pressure tradeoff.
- [GPU MODE leaderboard via cuda.compute/CUB](https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute/) — library-first rule for primitives, B200 wins.
- [PyTorch blog — TorchInductor CuTeDSL GEMM backend](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/) — nvMatmulHeuristics-pruned autotuning, decode-shape headroom, abstraction stop rule.

### Practitioner worklogs (ladder + magnitudes)
- [Simon Boehm — CUDA matmul worklog](https://siboehm.com/articles/22/CUDA-MMM) — canonical ladder 1.3%→93.7% of cuBLAS with per-rung numbers.
- [Lei Mao — CUDA matmul optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) — coalescing 6.4×, thread-tiling dominance.
- [salykova — SGEMM on GPU](https://salykova.github.io/sgemm-gpu) — beat cuBLAS at locked clocks; power-throttling confounder; bank-conflict padding.
- [GPU MODE lecture 8 notes](https://christianjmills.com/posts/cuda-mode-notes/lecture-008/) — ordered checklist + measured magnitudes.
- [vrushankdes — CUDA kernel tips](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-ii---cuda-kernel-optimization-tips) — per-shape bottleneck classification.
- Colfax tutorials (TMEM GEMM, clusters, block scaling, FA4 co-design) — Blackwell TC-bound design facts; FA4 SFU/SMEM asymmetry analysis.

### Benchmarking discipline
- [Standard Kernel — high-fidelity benchmarking](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/) — sub-10µs unmeasurability, clock-lock vs power cap, contention numbers, pacing effect.
- [Speechmatics — timing PyTorch ops](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch) — events vs host timers, warmup causes, queue saturation.
- [jan.ai — how we benchmark kernels](https://www.jan.ai/post/how-we-benchmark-kernels) — >100% SOL from unflushed L2; ncu clock-control distortion; median-of-20 harness.
- [guillesanbri — CUDA benchmarks](https://guillesanbri.com/CUDA-Benchmarks/) — real-kernel warmup, clock locking, median+percentile reporting.
- [miru_why on Sakana exploits](https://x.com/miru_why/status/1892703900425486539) — memory-reuse reward hack + tolerance-masked missing-conv kernel.
- [arxiv 2512.09196 — profiling-guided LLM kernel optimization](https://arxiv.org/html/2512.09196v1) — profiling feedback ~2× success rate, diminishing returns per round, 1.05× keep/revert arbiter, oscillation failure mode.
- [Yotta Labs — AMD Developer Challenge 2025](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025) — structural wins over tuning; custom launcher 3×; geomean scoring.

## Benchmark harness (read directly from source, 2026-07-04)

The official grader in [github.com/NVIDIA/SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench)
(`src/sol_execbench/`, Apache-2.0) — authoritative, not research-derived:
- `sol_score.py` — `S(T_k)=1/(1+(T_k−T_SOL)/(T_b−T_SOL))`.
- `core/bench/timing.py` — CUPTI timing, cold-L2 flush, warmup=10.
- `core/bench/correctness.py` — fp32 compare, tolerance + 0.99 matched-ratio, inf/nan + all-zeros hard fails.
- `core/bench/reward_hack.py` — monkey-patch / thread-injection / lazy-output / eval-integrity defenses.
- `core/bench/io.py` — role-heuristic input generation.
- `core/data/workload.py` — Workload / ToleranceSpec / InputSpec schema.
Data: [nvidia/SOL-ExecBench](https://huggingface.co/datasets/nvidia/SOL-ExecBench) (HF dataset, parquet).
Captured in [benchmark-grader.md](benchmark-grader.md).

## Pass 4 — Chinese ecosystem & papers (25 sources)

### Lab repos (primary)
- [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — SM100-native details, UE8M0 scales, Mega MoE megakernel, layouts ✅.
- [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) — SM100 support matrix + B200 TFLOPS ✅.
- [deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe) — bubble math; hardware-agnostic scheduling.
- [DeepEP issue #136](https://github.com/deepseek-ai/DeepEP/issues/136) — the `ld.global.nc.L1::no_allocate` unsafety analysis; maintainer-accepted.
- [SGLang GB200 tracking issue #7227](https://github.com/sgl-project/sglang/issues/7227) + [#9145](https://github.com/sgl-project/sglang/issues/9145) — Blackwell bringup timeline, fork pins, early parity caveat.
- [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) — arch support, sm89-reuse-on-sm100 structural detail ✅.

### Papers (primary)
- SageAttention [2410.02367](https://arxiv.org/abs/2410.02367) · SageAttention2 [2411.10958](https://arxiv.org/abs/2411.10958) · SageAttention2++ [2505.21136](https://arxiv.org/abs/2505.21136) · SageAttention3 [2505.11594](https://arxiv.org/abs/2505.11594) · SpargeAttn [2502.18137](https://arxiv.org/abs/2502.18137) — quantized-attention recipes ✅; SA3-on-B200 "direct applicability" refuted ✅.
- [DeepSeek-V3 technical report 2412.19437](https://arxiv.org/abs/2412.19437) — FP8 recipe, two-level accumulation, 20-SM comm allocation.
- [Kevin 2507.11948](https://arxiv.org/html/2507.11948v1) · [TritonRL 2510.17891](https://arxiv.org/pdf/2510.17891) · [LLM-kernel-gen survey 2601.15727](https://arxiv.org/pdf/2601.15727) — RL kernel generation, reward-hacking catalog; survey Table 2 places SOL-ExecBench (2603.19173) in the benchmark lineage.
- [Sakana admission tweet](https://x.com/SakanaAILabs/status/1892992938013270019) — eval-exploit retraction ✅.
- [PyTorch/torchtitan — MXFP8 + DeepEP on B200](https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/) — verified H800→B200 transfer numbers.

### Chinese community (blog; access often anti-bot-gated)
- [reed's CuTe series (Zhihu p/661182311)](https://zhuanlan.zhihu.com/p/661182311) — the CUTLASS layout-algebra curriculum.
- [B200 tcgen05 GEMM worklog (Zhihu p/2007020183127094121)](https://zhuanlan.zhihu.com/p/2007020183127094121) — 98%-of-cuBLAS ladder with per-step TFLOPS; SMEM-descriptor/core-matrix details.
- [Blackwell TMEM deep-dive (Zhihu p/1930631771835326592)](https://zhuanlan.zhihu.com/p/1930631771835326592) — rated *unreliable* by fetcher; L2-partition latency claims conflict with other sources.
- [FA4 explainer (Zhihu p/2032510402437891223)](https://zhuanlan.zhihu.com/p/2032510402437891223) + [CUTLASS-Blackwell slides walkthrough (p/2008547341972574634)](https://zhuanlan.zhihu.com/p/2008547341972574634) — anti-bot-blocked; mined via search-index renderings.
- [antigravity.codes DeepGEMM guide](https://antigravity.codes/blog/deepgemm-guide) — DeepGEMM API/layout summary.
