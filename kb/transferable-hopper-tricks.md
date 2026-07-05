# Transferable Hopper/H800 Tricks → B200

DeepSeek's published H800-constrained techniques and which have single-GPU
B200 analogues. Tags per kb/README.md; each notes B200 applicability.

## From the DeepSeek-V3 technical report (arXiv 2412.19437) ⚠️

- **FP8 recipe — 1×128 activation tiles / 128×128 weight blocks**: the exact
  granularity DeepGEMM implements. B200: directly applicable as a
  quantization scheme, but note TE emulates this via MXFP8 on Blackwell and
  NVIDIA prefers MXFP8 there (✅, see
  [quantized-kernels.md](quantized-kernels.md)); DeepGEMM's SM100 path is
  the native-style implementation (UE8M0 packed scales).
- **Two-level accumulation**: H800 FP8 tensor cores keep only ~14 bits of
  accumulation precision → promote partial sums to FP32 on CUDA cores every
  N_C=128 elements. B200: **needs re-evaluation** — tcgen05 changes the
  accumulation path (accumulators in TMEM), and the report explicitly asked
  vendors for better TC accumulation. Echoed by SageAttention2's two-level
  FP8-PV accumulation ✅. If a Quant problem shows accuracy drift, this is
  the pattern to reach for.
- **SM allocation for comms** (20 of 132 SMs, warp-specialized into 10
  channels, custom PTX, auto-tuned chunk sizes): multi-GPU technique, but
  the single-GPU B200 analogue is real — **dedicating warps/CTAs to a
  role** (copy engines, correction warpgroups, epilogue drains) is exactly
  the Blackwell dataflow style ([hopper-to-blackwell.md](hopper-to-blackwell.md));
  the AMD-challenge winner used the same pattern (56 CTAs for comm) ⚠️.
- **DualPipe**: bidirectional pipeline scheduling — hardware-agnostic
  scheduling math (MIT, pure PyTorch), multi-GPU only. B200: background
  theory; its lesson for kernels is the general one — hide communication/
  memory phases behind compute phases within one pipeline.

## Verified B200 transfer results ⚠️ [PyTorch/torchtitan blog]

- DeepEP on 256×B200: **+32%** DeepSeek-V3 671B pretraining throughput by
  itself; **+41%** combined with MXFP8 grouped GEMMs via torchao (651→918
  tok/s). The H800 comm stack transfers, including tunable SM allocation.
- MXFP8 on B200 is native (E8M0 per 32-elem block via tcgen05.mma), full FP8
  throughput, no emulation; 16B-MoE convergence matched BF16 over 1500
  steps. Caveat: quantization overhead is O(MK) vs O(GMNK) compute — small
  GEMMs can lose; matches the "skip Linears with min dim < 1024" heuristic
  in [quantized-kernels.md](quantized-kernels.md).

## Anti-patterns — do NOT transfer ✅

- **`ld.global.nc.L1::no_allocate` for cross-SM polling** (DeepEP's
  documented trick): unsafe on every arch sm70+ including Blackwell — no
  hit-on-miss check means stale L1 lines never refetch. Use `ld.global.cg`.
  DeepSeek accepted the analysis (2025-04-28). Details in
  [chinese-oss-kernels.md](chinese-oss-kernels.md).
- SageAttention2++'s FP16-accumulated FP8 MMA 2× trick: consumer-GPU
  (Ada/GeForce) rate asymmetry; H100/B200 run FP32-acc FP8 at full rate ✅.
- Hopper wgmma anything (`compute_90a` PTX) — won't even load
  ([hopper-to-blackwell.md](hopper-to-blackwell.md)).
- Hopper-era L2 assumptions: Hopper's LRC equalized L2 partition latency ⚠️;
  B200 partition behavior is disputed (2 vs 4 partitions, possible ~300/800
  cycle near/far asymmetry ⚠️ — see
  [chinese-community-insights.md](chinese-community-insights.md)).
  Microbenchmark before relying on L2 locality either way.

## Transfer heuristic

When porting any Hopper-era kernel or trick, check it against three B200
deltas: (1) wgmma→tcgen05/TMEM (issue model, accumulator location, epilogue
warpgroup shape), (2) accumulation precision differences (FP8 paths), and
(3) the flat-SFU/SMEM-vs-2×-TC asymmetry (a Hopper compute-bound kernel may
re-bottleneck on exp or SMEM — see
[optimization-recipe.md](optimization-recipe.md) Step 0).
