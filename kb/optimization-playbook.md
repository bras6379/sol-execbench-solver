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

## Picking a dtype to beat an fp32-reference conv/GEMM problem

For a benchmark problem whose PyTorch reference runs in fp32 (conv/GEMM-heavy:
VAE blocks, plain matmuls), PyTorch's fp32 cuDNN/cuBLAS default path itself
runs in **TF32** (10-bit mantissa) — so the reference the grader checks against
is effectively TF32-precision, not true fp32. The highest-EV tensor-core dtype
to match it is **FP16, not BF16**, and this is easy to get backwards: FP16 and
BF16 have the **same peak throughput** on B200 (both are 16-bit, ~2.25 PF dense
✅ — the roofline table above states this once for "BF16" but the number is
identical for FP16), so throughput alone gives no reason to prefer one. The
reason to prefer FP16 here is **precision**: FP16 has a 10-bit mantissa —
identical relative rounding to TF32 (~5×10⁻⁴) — while BF16 has only a 7-bit
mantissa (~4×10⁻³ error), which is large enough to blow a typical
`atol≈1e-2`-scale tolerance on a compute-heavy reduction. cuDNN/cuBLAS
accumulate FP16 inputs in fp32 internally, matching TF32's accumulation
behavior. Measured: a TF32 baseline on an fp32-referenced conv problem scored
0.451 — *below* the 0.5 baseline-parity mark — purely because it only matched
the TF32 baseline's own speed rather than beating it with a faster
same-precision dtype; switching the matmul/conv path to FP16 (keeping norms,
reductions, and the residual add in fp32) beat that baseline ~2× at identical
precision.

**When this doesn't apply:** FP16's usable range is exponent-limited (±65504)
where BF16 matches fp32's exponent range — if any intermediate can plausibly
exceed ~6×10⁴ (unnormalized pre-activation sums, large-scale losses), BF16 or a
scaled/mixed approach is safer despite the coarser mantissa. For the O(1)–
O(hundreds)-magnitude activations typical of normalized conv/GEMM blocks
(randn inputs, post-GroupNorm/LayerNorm activations), FP16's range is not a
practical constraint. Always confirm the actual `atol`/`rtol` for the specific
workload in `workload.jsonl` before assuming BF16 is disqualified — a loose
enough tolerance can still pass with BF16's throughput-equivalent, cheaper
mantissa.

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
