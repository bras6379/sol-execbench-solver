# SageAttention Family & Quantized-Attention Papers

Tsinghua thu-ml's quantized attention line — directly relevant to the
benchmark's quantized and attention problems. All code Apache-2.0 in
github.com/thu-ml/SageAttention ✅. Tags per kb/README.md.

## The critical B200 caveat (read first) ✅

**"SageAttention3 is directly applicable to B200" was REFUTED (0-3).**
Verified reality:

- SA3's benchmarks are on **consumer Blackwell (RTX 5090, sm_120)** — 1038
  TOPS ≈ 62% of *that card's* FP4 peak; ~5× the fastest FlashAttention
  *there*; e2e ~3×.
- The repo's `setup.py` DOES emit `compute_100a/sm_100a` for CC 10.0 ✅ — it
  compiles for B200 — but on sm100/sm120 the SageAttention2 INT8/FP8 path
  **reuses the sm89 Ada mma kernel module (no tcgen05-native kernels)**;
  Blackwell-native code exists only in the `sageattention3_blackwell` FP4
  subpackage, and even that is sm_120-validated. ✅
- Also refuted (0-3): "SpargeAttn explicitly supports Blackwell."
- **Treat the whole family as porting targets + recipe documentation for
  B200, not drop-ins.** The recipes map naturally onto tcgen05 block-scaled
  kinds but that work is ours to do.

## The recipes (what to actually reuse)

| Paper | Venue | Recipe | Key numbers |
|---|---|---|---|
| SageAttention (2410.02367) ✅ | ICLR'25 | Q,K → INT8 with outlier **smoothing**; **PV stays FP16** in v1 | ~2.1× FA2, ~2.7× xformers OPS; near-lossless e2e across LLM/image/video |
| SageAttention2 (2411.10958) ✅ | ICML'25 | Q,K → **per-thread INT4** with Q-smoothing; P̃,V → FP8 with **two-level accumulation** | INT8+FP8 variant matches FA3-FP8 speed on Hopper at higher accuracy; INT4 results are sm89 (Hopper lacks INT4 TC) |
| SageAttention2++ (2505.21136) ✅ | — | SA2 + FP8 MMA with **FP16 accumulation** | 3.9× FA2 on RTX 4090 — BUT the 2× FP16-acc advantage is a **consumer-GPU trait**; H100/B200 run FP32-acc FP8 at full rate → re-validate before assuming any B200 gain |
| SageAttention3 (2505.11594) ✅ | NeurIPS'25 Spotlight | **Microscaling FP4** attention (both matmuls quantized) | 1038 TOPS on RTX 5090; authors recommend SA2 for precision-sensitive uses; plug-and-play inference (not training) |
| SpargeAttn (2502.18137) ✅ | ICML'25 | Training-free sparse: 2-stage online filter (attention-map prediction → softmax-aware skip) **on top of SageAttention kernels** | Block sparsity composes orthogonally with INT8/FP8 quant in one kernel; follow-ups: PISA (2602.01077) |

## Details worth stealing for our kernels

- ✅ Smoothing (subtracting per-block means from Q/K) is the accuracy trick
  that makes INT8/INT4 QK^T viable — applicable to any quantized-attention
  candidate we write.
- ✅ Two-level accumulation for FP8 PV (accumulate in higher precision
  periodically) mirrors DeepSeek's FP8 GEMM trick — see
  [transferable-hopper-tricks.md](transferable-hopper-tricks.md).
- ✅ APIs to study: `sageattn_qk_int8_pv_fp8_cuda`,
  `sageattn_qk_int8_pv_fp8_cuda_sm90` (Hopper-tuned variant),
  `spas_sage2_attn_meansim_topk_cuda` (sparse+quant composed).
- ✅ Honest-baseline note: the repo's headline "2–5× over FlashAttention" is
  vs FA2; on H20 the e2e CogVideoX table shows FA3-FP8 nearly matching
  SageAttention (12'14'' vs 12'07'') — always compare against the strongest
  baseline, not the advertised one.
- ✅ Toolchain: CUDA ≥12.8 for Blackwell targets; SUPPORTED_ARCHS
  8.0/8.6/8.9/9.0/10.0/12.0/12.1.

## When to reach for this on the benchmark

- Quant-collection attention problems (nvfp4_grouped_query_attention etc.):
  SA3's microscaling-FP4 attention design is the closest published prior art
  — port its quantization structure onto tcgen05 `mxf4nvf4`, don't reuse its
  sm_120 kernels.
- bf16 attention problems where tolerances are loose: INT8 QK^T + smoothing
  (SA1/SA2 recipe) is a legitimate speed lever IF `workload.jsonl` tolerances
  allow — validate per the anti-Sakana checklist in
  [benchmarking-discipline.md](benchmarking-discipline.md).
