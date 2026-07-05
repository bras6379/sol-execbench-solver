# NVIDIA Library Playbook for B200

Which vendor library wins which op class, and when to go custom.
Tags: ✅ verified (3-vote panel or synthesized high-confidence) · ⚠️ unverified · ❌ refuted. See kb/README.md.

## Decision table (start here per problem)

| Op class | First choice | Why / caveats |
|---|---|---|
| Plain dense GEMM (bf16/fp16/tf32) | cuBLAS / torch.matmul | ✅ 1,672 TFLOP/s at 8192³ on B200; Triton reaches only 62–70%, CuTile 52–79%. Only one known hand-written kernel ever beat it (~106%). Don't rewrite plain GEMM. |
| GEMM + bias/activation (fwd) | cuBLASLt fused epilogue | ✅ bias (bf16/fp16), ReLU, GELU epilogues incl. FP8 GEMMs, with optional aux outputs |
| GEMM backward bias/ReLU | cuBLASLt dgrad epilogues | ✅ `DRELU_BGRAD`, `BGRADA/BGRADB` — dBias/dReLU ride the GEMM instead of a separate elementwise kernel. Carries to Blackwell; cuBLAS 12.9 exposes block-scaled FP8/FP4 via same API |
| Attention fwd (std/GQA/varlen/softcap) | cuDNN graph API (via frontend) | ✅ SDPA subgraph + arbitrary pointwise ops between BMM1 and softmax (TANH covers softcapping); usually compiles to ONE fused kernel |
| Attention backward | cuDNN ≥9.23 (SM100, d≤256) ✅ or open CuTe-DSL kernels | NVIDIA open-sourced Blackwell SDPA fprop+bprop d=256 kernels in cudnn-frontend (CuTe-DSL) — study/port them |
| MoE grouped expert GEMM | cuDNN frontend MoE kernels or CUTLASS 4.5 grouped GEMM | ✅ Grouped GEMM+GLU/SwiGLU, +Wgrad, +quantize (SM100); CUTLASS 4.5.0 grouped GEMM matches torch `grouped_mm` interface and beats torch 2.10/cu13 on B200 (mxfp8 1.29–1.41×, nvfp4 ~1.11×, bf16 ~1.15×, worst 0.96–0.98) |
| RoPE, causal-conv1d+SiLU, RMSNorm+SiLU | cuDNN frontend fused primitives ⚠️ | RoPE is a native cudnn op; causal conv1d w/ fused SiLU; RMSNorm+SiLU engine spans SM80–SM103 |
| NVFP4 / MX GEMM | CUTLASS example 72b / collective builders | ✅ no hand-written PTX needed — see [quantized-kernels.md](quantized-kernels.md) |
| Scan / sort / prefix-sum (MoE token sort) | CUB device primitives | standard answer for radix-sort + prefix-sum problems; benchmark has these explicitly |
| FFT conv (Hyena) | cuFFT (+ custom gating fusion) | benchmark's Hyena problems do rfft with padding; fuse the pointwise chain around cuFFT calls |
| Conv2d/3d + norm + act (VAE) | cuDNN graph API fusion engines ⚠️ | conv+pointwise fusion; NHWC layout matters (ConvNeXt problem is explicitly NHWC) |

## cuBLAS / cuBLASLt specifics

- ✅ FP8 GEMM epilogues since cuBLAS 12.0: bias (BF16/FP16), ReLU, GELU, ± aux
  buffers. FP16 GEMMs additionally: dBias, dReLU dgrad epilogues (uses the
  bitmask saved by the forward `RELU_AUX_BIAS` epilogue).
- ⚠️ Give cuBLASLt **≥ 32 MiB workspace** or it silently restricts algorithm
  selection.
- ⚠️ Heuristics cache memoizes shape→kernel mapping — repeated same-shape calls
  (exactly the benchmark setting) get low dispatch overhead.

## cuDNN 9 / cuDNN Frontend (github.com/NVIDIA/cudnn-frontend, MIT)

- ✅ Now NVIDIA's official open-source entry point AND a kernel collection:
  `python/cudnn/sdpa/`, `grouped_gemm/`, `discrete_grouped_gemm/`,
  `rmsnorm_rht_amax/` — real CuTe-DSL kernel sources targeting
  H100/H200/B200/GB200/GB300, FP16/BF16/FP8/MXFP8. Backend lib stays closed.
- ✅ SDPA backward on SM100 up to head dim 256 (cuDNN ≥ 9.23); open-source
  Blackwell bprop kernels: `fmha_dkdv_d256_sm100.py`, `fmha_dq_d256_sm100.py`,
  `fmha_forward_sm100_d256.py` (note: open CuTe-DSL versions are d=256 only ⚠️).
- ✅ MoE surface: Grouped GEMM+GLU, +SwiGLU (dense and discrete
  per-expert-pointer layouts, no weight packing), Grouped GEMM Wgrad
  (`GroupedGemmWgradSm100`), Grouped GEMM+Quant with dynamic shapes, plus a
  PyTorch custom op wrapping MoE Grouped GEMM (v1.22.1+).
- ✅ Graph API attention: custom variants via arbitrary pointwise DAG between
  BMM1 and softmax; fusion engine emits one fused kernel or a small set of
  cooperative kernels.
- ⚠️ SDPA varlen `cu_seqlens` packed batches supported.
- ❌ Refuted: "cuDNN SDPA is up to 2× (BF16) / 3× (FP8) faster than PyTorch
  eager attention" — do not cite; benchmark it yourself per problem.
- ⚠️ Its FP8 fwd SDPA hit 1.2 PFLOPS on **H200** (May 2024, pre-Blackwell
  data); B200-era numbers must come from newer releases.

## CUTLASS 4.x on SM100

- ✅ Grouped GEMM (4.5.0, 2026-05-01): torch `grouped_mm`-aligned, expert-wise
  tensormap setup via ~2 µs helper kernel, beats torch on B200 nearly
  everywhere (see decision table).
- ✅ Block-scaled MMA: NVFP4, MXF8×MXF4, MXF8×MXF6 mixed precision (since
  3.8.0, extended through 4.4/4.5).
- ⚠️ Examples to start from: `72_blackwell_narrow_precision_gemm` (dense
  FP4/FP8), `84_blackwell_narrow_precision_sparse_gemm`,
  `71_blackwell_gemm_with_collective_builder`.
- ❌ Refuted: "EVT (Epilogue Visitor Tree) fusion works on SM100
  grouped/pointer-array GEMMs" — Colfax (Oct 2024) documents EVT as
  **Hopper-only, warp-specialized kernels only**. SM100 EVT status must be
  verified directly against current CUTLASS before designing around it.
- ❌ Refuted as stated: specifics of "CUTLASS example 77 Blackwell FMHA
  (dims 32/64/128, varlen bwd, 5-MMA fused backward in 4.3.0)" — an FMHA
  example exists but check the repo for actual current coverage.
- ⚠️ Dispatch policies: `KernelTmaWarpSpecialized1SmSm100` / `2SmSm100`
  (see [tcgen05-and-tmem.md](tcgen05-and-tmem.md)).

## TransformerEngine

- ✅ On Blackwell, TE's DeepSeek-style `Float8BlockScaling` (128-block) is
  **emulated via MXFP8, not native** — NVIDIA says MXFP8 is the preferred
  recipe on Blackwell. Details in [quantized-kernels.md](quantized-kernels.md).

## When the library is NOT the answer

- Bandwidth-bound chains (norm+residual+rope+…) that no library fuses end to
  end → custom fused kernel (see [fusion-patterns.md](fusion-patterns.md)).
- The 65 fp32-only benchmark problems: tensor-core libraries mostly
  irrelevant; it's coalescing + fusion.
- L2-collection whole-block problems: library calls leave inter-op HBM
  round-trips on the table; fusion or megakernel wins.
- Weird shapes (skinny M, tiny K): ⚠️ Triton and even cuBLAS degrade;
  hand-tuned tiling can win.
