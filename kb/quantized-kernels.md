# Quantized Kernels on B200 (FP8 / NVFP4 / MX)

For the Quant collection (problems 177–209: fp8-e4m3 + nvfp4 projections).
Tags per kb/README.md.

## Format cheat sheet

| Format | Element | Block size | Scale type | Hardware path |
|---|---|---|---|---|
| FP8 per-tensor / rowwise | e4m3/e5m2 | tensor/row | fp32 | tcgen05 f8f6f4; cuBLASLt FP8 API |
| MXFP8 | e4m3/e5m2 | 32 | ue8m0 (pow-2) | `kind::mxf8f6f4` ✅ |
| MXFP4 | e2m1 | 32 | ue8m0 | `kind::mxf4` ✅ |
| **NVFP4** | e2m1 | **16** | **ue4m3** | `kind::mxf4nvf4` ✅ (the general 4-bit kind) |
| DeepSeek blockwise FP8 | e4m3 | 1×128 acts / 128×128 weights | fp32 (pow-2 on Blackwell) | **emulated via MXFP8 on B200** ✅ |

- ✅ NVFP4 recipe confirmed twice: one ue4m3 scale per 16-element block
  (CUTLASS `nv_float4_t` SF vector size 16 dense / 32 sparse; example 72b
  uses `InputSFVectorSize = OutputSFVectorSize = 16`).
- ⚠️ Max representable in an nvf4 block: 6 × 448 = 2688. [Colfax]
- ✅ Rated throughput: `mxf4nvf4.block_scale` = 4× Hopper FP8; f16/tf32/i8
  tcgen05 = 2× Hopper. Dense datasheet peaks in
  [b200-hardware.md](b200-hardware.md).

## The scale-factor layout tax (main implementation pitfall)

- ⚠️ SM100 block-scaled GEMM requires scale factors in a fixed basic-block
  structure: **128 rows (M/N) × 4 scales (K), 512-byte K-major blocks** —
  you must produce this layout when quantizing. [CUTLASS docs]
- ⚠️ Scales are staged SMEM→TMEM via `tcgen05.cp` issued by the same warp as
  the MMA (same internal pipeline = implicit ordering); simplest fast setup
  stores scales in GMEM already in the (32×16-byte tiled) TMEM layout so TMA
  loads them directly. [Colfax]
- ✅ DeepGEMM on SM100 wants scales as **packed ue8m0, 4 per int32** (SM90
  took fp32) — porting DeepSeek-style kernels to B200 means reworking scale
  handling.
- ⚠️ Alignment: unpacked fp4/fp6 operands need 128-element alignment; fp8
  16-element when mixed with 4/6-bit. [CUTLASS docs]

## Ready-made building blocks

- ✅ **CUTLASS example 72b**: NVFP4×NVFP4 GEMM via CollectiveBuilder — no
  hand PTX. Design: block-scaled tcgen05.mma + TMEM + warp-specialized
  MMA/epilogue + cluster-launch-control dynamic scheduler. MMA tile
  128×128×256, cluster 1×1×1 ⚠️; needs CUDA ≥12.8, CC 10.0/10.1/10.3 ⚠️.
- ⚠️ 72b can emit **quantized FP4 output D + ue8m0 scales** — fused
  output-quantization epilogue for chained low-precision GEMMs (relevant to
  nvfp4 problems whose outputs feed the next quantized layer).
- ⚠️ CuTe-DSL: `dense_blockscaled_gemm_persistent.py` with MmaMXF8Op /
  MmaMXF4Op / MmaMXF4NVF4Op; bK = 128 B (8-bit) / 256 (4-bit) so a K-tile
  spans 4 UMMA atoms; min bM = 128. Explicitly a starting point, not tuned.
- ✅ CUTLASS 4.5 grouped GEMM: nvfp4/mxfp8/bf16 MoE shapes beating torch on
  B200 (numbers in [library-playbook.md](library-playbook.md)).
- ⚠️ FlashInfer: FP8 GEMM (per-tensor + groupwise), FP4 GEMM (NVFP4+MXFP4),
  quantized MoE with blockwise-scaled FP8/FP4 expert weights.
- ⚠️ FlashMLA: in-kernel dequant of blockwise FP8 KV cache (layout in
  [oss-kernel-catalog.md](oss-kernel-catalog.md)).

## Recipes / framework status

- ✅ TransformerEngine on Blackwell: `Float8BlockScaling` (DeepSeek 128-block)
  is **emulated via MXFP8**; MXFP8 is the stated preferred Blackwell recipe.
  Details: 1D (1×128) acts/grads + 2D (128×128) weights by default; 2D×2D
  GEMMs unsupported (≤1 operand 2D); Blackwell allows only power-of-2 scales;
  blockwise uses e4m3 for both fwd and bwd (unlike per-tensor's e5m2 grads)
  ⚠️; last dim and remaining-dims product must divide by 128 ⚠️.
- ⚠️ Backward constraint: 1D-block-quantized tensors can't be transposed —
  quantize rowwise and columnwise separately from the hi-precision source
  (2D-scaled tensors transpose fine). Shapes kernel design for FP8 training
  problems.
- ✅ torchao end-to-end reality check (diffusion, B200): NVFP4
  dynamic-act+weight up to ~1.68× e2e (LTX-2 b8); MXFP8 ~1.26× (Flux b8);
  path is `torchao.prototype.mx_formats` ⚠️; NVFP4 needs CC ≥ 10.0 ✅.
- ⚠️ Deployment heuristics that held: skip Linears with min(M,K,N) < 1024 and
  accuracy-sensitive layers (embeddings, norms) — better quality at ~no
  latency cost; add CUDA graphs (`reduce-overhead`) for small batch.

## Solver guidance for the 24 Quant problems

1. The reference implementations define required numerics — read
   `reference.py` scale handling before choosing a kernel; tolerances in
   `workload.jsonl` decide how loose you can be.
2. FP8 projection problems: try cuBLASLt FP8 (+ fused epilogue) first, then
   CUTLASS/DeepGEMM if the reference's scaling granularity doesn't match
   cuBLASLt's supported modes.
3. NVFP4 problems: start from CUTLASS 72b / FlashInfer FP4 GEMM; budget time
   for the scale-factor layout conversion — that's where the bugs live.
4. Watch for dequant-fused-epilogue opportunities: output-quantize in the
   epilogue instead of a separate kernel.
