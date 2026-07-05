# Open-Source Kernel Catalog (B200-readiness)

What exists, what's actually sm_100-ready, licenses, and what to port vs study.
Tags per kb/README.md.

## Attention

| Project | B200 status | Covers | License |
|---|---|---|---|
| **FlashAttention-4** (Dao-AILab/flash-attention) | ✅ B200-ready, written in CuTe-DSL (`pip install flash-attn-4`, cu13 extra for best perf ⚠️) | The FA generation for Blackwell; FA-3 is **Hopper-only** ✅ | BSD-3 ⚠️ |
| FA library features ⚠️ | — | MQA/GQA, varlen (`flash_attn_varlen_func`), KV-cache update **with fused rotary** (`flash_attn_with_kvcache`), paged KV, softcapping, deterministic backward option | |
| FA-3 constraint ✅ | Hopper | FP8 **forward only**; backward is FP16/BF16 only | |
| **FlashMLA** (deepseek-ai) | ✅ support matrix verified: Dense Decoding **SM90-only**; Sparse Decoding SM90+SM100; Dense Prefill **SM100-only**; Sparse Prefill SM90+SM100 (SM100 MHA kernels contributed by NVIDIA 2025-08); CUDA ≥12.9 | MLA (K=576/V=512); B200: 1460 TFLOPS prefill fwd / 1000 bwd ✅, 1450 sparse prefill ✅, 350 sparse FP8 decode ("not really optimized yet" per repo ✅) | MIT |
| **NVIDIA open SDPA kernels** (cudnn-frontend) | ✅ Blackwell fprop+bprop, d=256, CuTe-DSL | study/port for attention-backward problems | MIT |
| xFormers / FA-2 | legacy | FA-2 is the *baseline to beat*, not the target | BSD |

FlashMLA implementation details worth stealing ⚠️: 3-warpgroup specialization
(384 threads), 64×64 tiles, double buffering, explicit TMEM partitioning
(acc in TMEM regions 0–255, scores 256–287); FP8 KV layout = 512 B FP8 V +
16 B of 4 float scales (4×128 groups) + bf16 RoPE features per token.

## GEMM / quantized GEMM

| Project | B200 status | Covers | License |
|---|---|---|---|
| **DeepGEMM** (deepseek-ai) | ✅ Blackwell-native since July 2025 refactor; CUDA ≥12.9; SM100 gets all four layouts (NT/TN/NN/TT) vs NT-only on SM90 ✅ | FP8 blockwise (128) GEMM; **SM100 scales = packed UE8M0 (4/int32), not FP32** ✅; grouped GEMM M-axis-only, contiguous (train/prefill) + masked (decode, CUDA-graph-safe) ✅, K-grouped for MoE bwd ⚠️; **Mega MoE megakernel (EP dispatch→FP8×FP4 linear→SwiGLU→linear→EP combine in ONE kernel, 2026-04)** ✅; JIT/NVRTC | MIT |
| **CUTLASS examples** | ✅ 72b NVFP4×NVFP4 (see [quantized-kernels.md](quantized-kernels.md)), 84 sparse, 71 builder | dense block-scaled | BSD-3 |
| **quack** (Dao-AILab) | ✅ targets H100, B200/B300, RTX50; CUDA ≥12.9, Py 3.12 | Pure CuTe-DSL: RMSNorm fwd+bwd, softmax fwd+bwd, cross-entropy fwd+bwd, LayerNorm fwd; Blackwell-specific GEMM w/ fused epilogue (separate from Hopper impl) ✅; claims "speed-of-light" memory-bound perf ⚠️ | Apache-2.0 |

## Inference-stack kernel libraries

| Project | B200 status | Covers | License |
|---|---|---|---|
| **FlashInfer** | ✅ B200/B300 (SM 10.0/10.3) since v0.4.0 (2025-10-08); SM100 kernels in CuTe-DSL, CUDA-13 extra | MLA, paged/ragged KV decode/prefill/append, cascade attention, fused MoE (DeepSeek-V3/Llama-4 routing), LLaMA-3.1 RoPE, fused RMSNorm/LayerNorm/Gemma norms; FP8 GEMM (per-tensor + groupwise), FP4 GEMM (NVFP4 + MXFP4), quantized MoE ⚠️ | Apache-2.0 |
| FlashInfer architecture ⚠️ | — | multi-backend dispatcher over FA-2/3, cuDNN, CUTLASS, TRT-LLM kernels — one API aggregating everything; **inference-oriented (no backward)**; Triton fallback path emits no tcgen05 | |
| vLLM / SGLang kernels | check per-kernel | fused rope, rmsnorm, moe align/topk — small CUDA kernels, easy to read/port; both consume FlashInfer for the heavy ops | Apache-2.0 |
| Liger-Kernel | Triton-based | RMSNorm, RoPE, SwiGLU, GEGLU, cross-entropy, fused-linear-CE — fwd+bwd, portable Triton (no sm_100 specifics) | BSD-2 |

## SSM / conv

- **state-spaces/mamba** official kernels + **causal-conv1d** (Dao): the
  reference implementations for selective scan / chunk scan / conv1d+gating
  problems. B200 status unverified — likely "runs but Hopper-tuned" ⚠️;
  cuDNN frontend now has causal-conv1d+SiLU fused as an alternative ⚠️.
- Apex fused kernels (LayerNorm etc.): legacy; superseded by quack/Liger for
  our purposes.

## Megakernels

- **HazyResearch/Megakernels**: Llama-1B decode as one persistent kernel,
  B200 numbers in [fusion-patterns.md](fusion-patterns.md) ⚠️. Template for
  L2 whole-block problems.

## How to use this catalog per problem

1. Match problem family → project above; read the kernel before writing one.
2. Check the B200 column — Hopper-only code (FA-3, wgmma-based kernels) won't
   even JIT to sm_100 (`compute_90a` — see
   [hopper-to-blackwell.md](hopper-to-blackwell.md)).
3. Licenses are all permissive (MIT/BSD/Apache) — porting into benchmark
   submissions is fine; verify the benchmark's own rules on external code.
4. Ports still need shape-specific tuning: these kernels are tuned for
   serving shapes, not necessarily each problem's `workload.jsonl` axes.
