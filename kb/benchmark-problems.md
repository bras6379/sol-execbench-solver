# SOL-ExecBench Problem Taxonomy

Surveyed 2026-07-04 from the 200 locally-cloned problem packs in
`~/monorepo/tests/sol-execbench-solver/problems/` (source:
https://research.nvidia.com/benchmarks/sol-execbench/problems).
Machine-readable summary: each problem has `definition.json` (axes, I/O
shapes/dtypes), `reference.py` (PyTorch reference + input generator),
`workload.jsonl` (benchmark axes + tolerances).

## Collections (website/global IDs)

| IDs | Collection | Local status |
|---|---|---|
| 1–94 | L1 (single ops / small fusions) | 94, all fetched |
| 95–176 | L2 (whole blocks / large fusions) | 82, all fetched |
| 177–209 | Quant (FP8 / NVFP4) | 33, all fetched |
| 210–235 | FlashInfer-Bench | 26, all fetched |

All 235 fetched locally via `solver fetch --all` (HF dataset
`nvidia/SOL-ExecBench`). 3,957 workloads total. Tolerance is present on every
L1/L2/Quant workload; **18 FlashInfer-Bench problems (rmsnorm/gemm/moe:
210–220, 229–235) have no tolerance**, the 8 paged/ragged GQA/MLA attention
ones (221–228) do — see [benchmark-grader.md](benchmark-grader.md).

Problems are extracted from 95 distinct real models (DeepSeek-V3/R1, Qwen3-VL/
Omni, Wan2.2 video, Parakeet ASR, SAM-HQ, Gemma-3n, Nemotron, FLUX, Whisper,
Mamba/Hyena hybrids…), so shapes are production LLM/diffusion/ASR shapes, not
synthetic.

## Op-family distribution (by name/description keywords)

| Family | ~Count | Notes |
|---|---|---|
| Attention variants | ~70 | GQA, MLA, cross/joint/window attention, qk-norm, softcapping, varlen `cu_seqlens`, KV-cache update, ultralong flash |
| RoPE family | ~25 | standard, YaRN, 2D/3D/multimodal grids, inverse-freq computation, cos/sin generation, many backwards |
| MoE | ~30 | top-k / group-limited routing, token scatter/dispatch/capacity, batched & grouped expert GEMMs, weighted accumulation, shared experts, radix-sort + prefix-sum token sorting |
| Norm + residual fusions | ~25 | RMSNorm, LayerNorm, GroupNorm, AdaLN(-Zero), GRN, AdaIN, pre/post-norm chains |
| MLP / activations | ~20 | SwiGLU, GEGLU, GELU (approx), SiLU, gated projections, forward + backward |
| SSM / alt-arch | ~12 | Mamba1/2 selective scan, chunk scan, segsum, dt-softplus, conv1d+gating, Hyena FFT (rfft) blocks, gated delta rule linear attention, time-decay stabilization |
| Conv / vision / VAE | ~15 | conv2d/3d blocks, depthwise conv1d, ConvNeXtV2 (NHWC), patch embedding, VAE encoder/decoder blocks, feature pyramids |
| Full decoder/encoder layers (L2) | ~20 | entire transformer blocks: norm→attn→residual→norm→MLP→residual |
| Quantized projections | 24 | FP8-e4m3 and NVFP4 linear/QKV/MoE/Mamba projections |
| Misc | ~10 | LM head + logit slicing, embeddings, losses, sampling-adjacent |

## Key cross-cutting facts

- **46/200 problems are backward passes** — dgrad/wgrad fusion chains,
  recompute-vs-save decisions, and backward flash-attention matter as much as
  forward kernels.
- **Dtypes**: 65 problems are pure fp32 (mostly vision/conv/SSM); ~100 involve
  bf16; Quant problems use `float8_e4m3fn` and `float4_e2m1fn_x2` (+ scales).
  fp32-heavy problems get **no tensor-core help beyond TF32** — they're
  bandwidth/fusion problems, not tcgen05 problems.
- **L2 problems are fusion problems**: the win comes from eliminating
  intermediate HBM round-trips across a whole block, not from a faster GEMM.
- **Quant problems map directly to tcgen05 block-scaled kinds**
  (`mxf4nvf4` for NVFP4, FP8 GEMM + fused dequant epilogues) — see
  [tcgen05-and-tmem.md](tcgen05-and-tmem.md) and the quantized-kernels KB file.
- Correctness is checked against `reference.py` with per-workload tolerances
  in `workload.jsonl`; latency on real B200 is the score signal.

## Implications for the solver

1. A per-family playbook beats per-problem improvisation: ~6 kernel families
   cover ~90% of problems.
2. The bandwidth-bound majority (norms, RoPE, scatter, elementwise chains,
   fp32 ops) needs the fusion/roofline playbook; only GEMM-shaped work
   justifies tcgen05-level effort.
3. Backward passes are a distinct skill: expect chain-rule fusion, atomics vs
   split-reduction for wgrad, and recompute tradeoffs.
4. Existing OSS kernels (FlashAttention, FlashInfer, Liger, Mamba official,
   DeepGEMM…) likely already solve many problems — porting + adapting can be
   faster and safer than writing from scratch (license/benchmark rules
   permitting).
