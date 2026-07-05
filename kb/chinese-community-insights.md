# Chinese Community Deep-Dives (Zhihu / WeChat / CSDN)

High-quality Chinese-language sources with insight beyond NVIDIA docs.
This angle got the least adversarial verification — most items ⚠️ — but two
sources stand out enough to mine deeply. Tags per kb/README.md.

## The B200 tcgen05 GEMM worklog (Zhihu p/2007020183127094121) ⚠️ — highest value

A from-scratch pure CUDA/PTX tcgen05 BF16 GEMM on B200 reaching **98% of
cuBLAS** (1475.93 vs 1506.74 TFLOPS at 4096³, PyTorch 2.9.1 + CUDA 13; this
is the Chinese-community twin of gau-nernst's English writeup — likely the
same lineage, DeepGEMM cited as the structural reference). Its measured
**Blackwell optimization ladder**:

| Step | TFLOPS | Gain |
|---|---|---|
| naive tcgen05 + 2D 16B TMA | 254.6 | — |
| 128B-swizzled 3D TMA | 695.4 | **2.7×** |
| pipelining | 939.6 | 1.35× |
| warp specialization | 1208.8 | 1.29× |
| 2-SM MMA | 1302.3 | ~1.08× |
| persistent kernel | 1475.9 | 1.13× |

Unique implementation details not in NVIDIA docs:

- The 2.7× swizzle gain is argued to come from **wider TMA innermost
  dimensions**, not bank-conflict elimination (the non-swizzled 8×16B
  core-matrix layout was already conflict-free).
- SMEM descriptor semantics (LBO/SBO) hinge on an undocumented **8×16B "core
  matrix"** unit whose columns must be contiguous in SMEM — explained only in
  Modular's Blackwell series, absent from PTX docs.
- Descriptor encoding: bits 61–63 = swizzle mode (2 = 128B swizzle);
  `desc_encode(addr) | (desc_encode(SBO)<<32) | (1<<46) | (2<<61)`.
- **Correctness bug**: `__syncthreads()` after the main loop does NOT
  guarantee MMA completion (only issue) — needs an extra mbarrier signaled
  by `tcgen05.commit`. Classic silent-wrong-answer source.
- 2-SM MMA gave only ~8% here — consistent with it being a per-SM-B-traffic
  optimization, not a 2× lever (matches the refuted-claim nuance in
  [tcgen05-and-tmem.md](tcgen05-and-tmem.md)).
- TMEM alloc returns its address **into shared memory** (observed 0 for a
  single allocation); author's verdict: Blackwell TC programming is
  tile-level, easier than Hopper because the viable design space is small.

## reed's CuTe series (Zhihu, from p/661182311) ⚠️ — the CUTLASS curriculum

- The famous Chinese CuTe tutorial series; author reed (GitHub reed-lau),
  acknowledged in CUTLASS discussion #1252. Covers Layout algebra → Swizzle →
  Copy → MMA → GEMM pipeline as a complete curriculum.
- Core abstraction: Layout = Shape+Stride as **recursively nested tuples**
  (hierarchical layouts are what express TC operand tilings and swizzled
  SMEM that flat shape/stride cannot).
- Examples are SM80/Hopper-era — **layout algebra transfers to B200; the
  arch-specific examples need porting**. No Blackwell installment found.
- Seeded the Chinese CuTe learning community (WeChat study groups active
  into 2026) — follow-on Blackwell content clusters around this author
  network; worth re-searching when stuck on CuTe layout puzzles.

## Blackwell TMEM deep-dive (Zhihu p/1930631771835326592) ⚠️ (source rated *unreliable* by the fetcher — verify anything load-bearing)

Mostly consistent with our verified facts (TMEM 128×512×32-bit, per-warp
32-lane limit, single-thread UMMA issue, no more Hopper ping-pong). Unique
claims to treat as leads only:

- CUTLASS 2-SM pair sync detail: odd CTA finds leader's mbarrier via
  `Sm100MmaPeerBitMask = 0xFEFFFFFF` (clearing bit 24 of the cluster SMEM
  address = bit 0 of CTA-id-in-pair).
- **L2 partition locality**: claims B200's 126 MB L2 is two ~63 MB partitions
  with ~300-cycle near / ~800-cycle far miss latency (vs 200/550 on A100) —
  would make L2 locality matter more than on Hopper. CONFLICTS with the
  "4 partitions" (2-1 vote) and "LRC equalizes latency on Hopper" claims
  elsewhere — the L2 partition question is genuinely open; microbenchmark on
  the pod if a kernel looks L2-latency-sensitive.
- Peak math from 148 SMs @ ~1.97 GHz: ~4.8 PF FP16 / ~9.55 PF FP8 / ~19.1 PF
  FP4 (with sparsity, presumably) — cross-check against datasheet dense
  numbers in [b200-hardware.md](b200-hardware.md) before use.

## FA4-on-Blackwell explainer (Zhihu p/2032510402437891223) ⚠️ (page anti-bot-blocked; content from search-index rendering)

Chinese explainer of FlashAttention-4's Blackwell design; consistent with
the Colfax/Tri Dao material already in
[optimization-recipe.md](optimization-recipe.md): two Q-tiles ping-ponged
across two softmax warpgroups scheduled so their exp phases never overlap
(avoiding MUFU/SFU contention); a dedicated "correction" warpgroup does
rescaling off the critical path; P staged in TMEM; cubic-polynomial 2^f
approximation with IEEE-754 bit tricks; relaxed online softmax that skips
rescale unless the running max moves beyond a threshold.

## Access note

Zhihu blocks unauthenticated scraping (HTTP 403 + zse-ck challenge;
`need_login` API). For future mining: use browser-based access or mirrors;
several articles here were only readable via search-index renderings.
