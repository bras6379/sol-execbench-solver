# B200 Optimization Playbook

Tags: ✅ verified · ⚠️ unverified · ❌ refuted (see kb/README.md legend).

## Roofline anchors

Compute (dense, per GPU) ✅: BF16 2.25 PF · FP8/FP6 4.5 PF · FP4 9 PF ·
TF32 1.1 PF · FP32 75 TF · FP64 37 TF (note: FP64 is *lower* than H100's 67 TF ✅).
Memory ✅: 7.7 TB/s HBM3e (HGX B200); achievable ~80–90% of peak ✅
(~7.48 TB/s STREAM ⚠️).
L2 ✅: 126 MB; ⚠️ ~21 TB/s. TMEM/SMEM per SM: 256 KB / 228 KB.

Arithmetic-intensity break-even (dense BF16): 2250e12 / 7.7e12 ≈ **292
FLOP/byte** — everything short of GEMM/attention-forward is bandwidth-bound.
For elementwise/reduction kernels the only game is coalescing, vector widths,
and fusion; tensor cores are irrelevant.

## Achieved-performance targets (what "good" looks like)

| Kernel | Result | Source |
|---|---|---|
| cuBLAS BF16 GEMM | ~1763 TFLOPS (~78% of peak) ⚠️ | Modular |
| Hand-written tcgen05 BF16 GEMM (M=N=K=4096) | 1476 TFLOPS = 98% of cuBLAS ⚠️ | gau-nernst |
| Gluon FP16 matmul, pipelined | 1113 TFLOPS ⚠️ | Triton tutorial |
| ThunderKittens BF16/FP8 GEMM | cuBLAS-level ⚠️ | HazyResearch |
| FlashAttention-4 BF16 | 1605 TFLOPS ≈ 71% util; 1.1–1.3× cuDNN 9.13 ⚠️ | Tri Dao |
| FP64 DGEMM | 36.3 TFLOPS = 80.7% of peak ⚠️ | arxiv 2512.02189v3 |

Interpretation: ~75–80% of dense peak is the practical ceiling for GEMM;
95–98% **of cuBLAS** is achievable by hand. Attention tops out ~70% because
of the exp/SFU bottleneck.

## Checklist for a peak-throughput GEMM-class kernel

1. Target `sm_100a`, use tcgen05 (via CUTLASS SM100 collectives, Gluon, or
   raw PTX). wgmma/mma.sync leaves 2–4× on the table ⚠️.
2. Tile M/N ≥ 128 — the tensor core acts like a 128×128 systolic array;
   64-wide tiles run at ¼ rate ⚠️.
3. Use CTA pairs (`.cta_group::2`, 2-SM MMA) for large GEMMs: doubles M to
   256, halves per-SM B-operand smem traffic ⚠️/✅.
4. Three-stage overlap: TMA→SMEM, MMA→TMEM, epilogue TMEM→GMEM ⚠️.
5. Double-buffer accumulators in TMEM — a 128×256 FP32 accumulator is exactly
   half of TMEM, so two fit (~100 TFLOPS win on TK's BF16 GEMM) ⚠️.
6. Persistent kernel + tile scheduler; grid sized to 148 SMs ✅ (fewer if
   using clusters: 4→132, 8→120, 16→112 ⚠️).
7. Epilogue: full warpgroup (4 warps) for TMEM drain (lane restriction ✅);
   overlap drain with next tile's MMAs.
8. Exploit implicit tcgen05 pipelining (back-to-back MMAs, cp+mma from one
   thread: ~10% on NVFP4 GEMM) ⚠️; drop redundant proxy fences when mbarrier
   acquire already orders TMA writes vs tensor-core reads (20–30 TFLOPS) ⚠️.
9. Quantized ops: block-scaled kinds (mxf8f6f4/mxf4/mxf4nvf4) with hardware
   scale application; respect 128-element alignment for unpacked FP4/FP6 ⚠️.
10. Attention: budget the softmax — exp SFU is ~512× slower than BF16 MACs
    per SM; pipeline exp against MMAs (FA4's core trick) ⚠️.

## Pitfalls

- Forgetting `tcgen05.dealloc` before kernel exit ✅ (required by PTX ISA).
- Assuming datasheet TFLOPS are dense — they include 2× sparsity except
  FP64 ✅.
- Porting `compute_90a` (wgmma) PTX — will not run ⚠️.
- Reusing consumer-Blackwell (sm_120) measurements or code paths — no
  TMEM/tcgen05 there ⚠️, and CC 12.0 has different smem/occupancy limits ✅.
- Expecting INT8 to beat FP8 (same rate ✅) or FP16-vector to beat FP32-vector
  (no longer 2× ⚠️).
- TMEM kernels get 1 CTA/SM ⚠️ — latency hiding must come from async
  pipelines within the CTA, not from multiple resident CTAs.
- Small wins (<3%) need reruns before trusting — carried over from prior
  SOL-ExecBench experience.

## ❌ Refuted / suspect claims — do not reuse

These failed adversarial verification:

1. "TMEM has ~8 TB/s per-SM read + 8 TB/s write bandwidth and ~420-cycle
   latency" (0-3) — the numbers could not be confirmed in the cited paper.
   TMEM's existence/size is solid ✅; its bandwidth is unknown.
2. "tcgen05.mma issue latency ~11 cycles vs 32–128 for Hopper wgmma" (0-3 in
   one form, 1-2 in another) — do not design around specific issue-latency
   cycle counts; rely on the async/pipelined programming model instead.
3. "Measured B200 peaks: FP4 7702 TFLOPS, FP8 3851 TFLOPS, FP16 964 TFLOPS…"
   (0-3) — could not be confirmed as stated (mixed sparse/dense, wrong
   attribution). Use the datasheet dense numbers.
4. The composite "FP8 = exactly 2× FP4-dense … 192 GB @ 7.7 TB/s @ 1000 W"
   claim (0-3) — mixed HGX/NVL72 configs. Per-config numbers in
   [b200-hardware.md](b200-hardware.md) are the verified ones.
5. "tcgen05 introduced in PTX ISA 8.4" (0xsero wiki) — it's PTX ISA 8.6 ✅.

**Source-hygiene warning ✅**: arXiv 2507.10789 ("Dissecting the NVIDIA
Blackwell Architecture with Microbenchmarks") benchmarks the **consumer**
GB203 (RTX 5080, sm_120a) vs H100 PCIe — zero B200 measurements, no
TMEM/tcgen05 data. Never transpose its numbers to B200. The correct B200
microbenchmark paper is arXiv 2512.02189 (and even there, several measured
claims failed verification — trust its topology facts more than its
latency/bandwidth numbers).

## Open questions to resolve on the pod

- Toolkit/driver on the pod (`nvcc --version`, `nvidia-smi`): need CUDA ≥
  12.8, ideally 13.x, for sm_100a/tcgen05 work.
- Actual HBM capacity/bandwidth of the rented B200 config (180 vs 186 GB).
- TMEM bandwidth characteristics (unknown — microbenchmark if a kernel seems
  TMEM-bound).
- L2 partition count (4 vs 2 — sources disagree; only matters for extreme
  L2-bandwidth tuning).
- Whether the benchmark's baseline is cuBLAS/cuDNN or something weaker — sets
  how much tcgen05 effort each problem deserves.
