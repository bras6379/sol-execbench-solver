# Hopper → Blackwell: What Carries Over, What Breaks

Tags: ✅ verified · ⚠️ unverified (see kb/README.md legend).

## Carries over unchanged

- ✅ Occupancy envelope: 64 warps/SM, 64K×32-bit registers, 255 regs/thread,
  32 blocks/SM, 228 KB smem — identical to H100, so Hopper occupancy
  reasoning transfers.
- TMA (bulk async copies, mbarrier-driven producer/consumer pipelines) —
  still the way data moves; tcgen05 deliberately mirrors TMA's
  single-thread-issue style.
- Thread block clusters + distributed shared memory (portable max still 8 ⚠️,
  nonportable 16 ⚠️).
- Warp specialization (producer/consumer warps) — still the dominant kernel
  organization, but roles shift (see below).
- Persistent-kernel + tile-scheduler patterns.

## Breaks / changes

| Hopper practice | Blackwell reality |
|---|---|
| `wgmma.mma_async` warpgroup MMA | ✅ Not supported. Use `tcgen05.mma`. |
| Accumulators in registers, wgmma thread-value layouts | ⚠️ Accumulators in TMEM, transparent row-major; registers freed for other work [Colfax, Modular] |
| Warpgroup issues MMA collectively | ✅ One thread issues the MMA |
| Register pressure limits tile size | ⚠️ TMEM capacity + its 1-CTA/SM limit are the new constraints [TK2] |
| `compute_90a` PTX JITs forward | ⚠️ **No** — arch-conditional Hopper PTX (`compute_90a`, i.e. anything wgmma) will not run on Blackwell [compatibility guide, central claim] |
| Grids sized for 132 SMs (H100) | 148 SMs ✅; minus cluster-size losses ⚠️ (4→132, 8→120, 16→112 active) |
| L2 50 MB | 126 MB ✅ — rethink what's "L2-resident"; tiling/blocking assumptions change ⚠️ |
| HBM3 3.4 TB/s (H100) / 4.8 (H200) | 7.7–8 TB/s HBM3e ✅ — bandwidth-bound kernels get ~2.3× for free |
| FP8 is the narrowest format | FP4/FP6 + MX/NVF4 block scaling ✅, at 2–4× FP8 rates ⚠️ |
| Packed-half vector math 2× FP32 | ⚠️ No longer — FP16 vector rate ≈ FP32 rate [Chips and Cheese] |
| FP64 tensor core 67 TF (H100) | ✅ **Regressed to 37–40 TF** — FP64 problems get no Blackwell tensor uplift |

## Programming-model shift (how kernels are organized now)

⚠️ [HazyResearch TK Blackwell; Modular; gau-nernst]

- Hopper: consumer warpgroups own accumulators in registers and issue wgmma;
  producers feed SMEM via TMA.
- Blackwell: kernels become more of a **dataflow machine** —
  - TMA producer warps fill SMEM;
  - a single "MMA-launcher" thread/warp issues async tcgen05 MMAs that
    accumulate in TMEM, with MMA→producer signaling via mbarriers;
  - epilogue warpgroup(s) drain TMEM (`tcgen05.ld`, 4 warps required by the
    lane restriction) and write out, overlapped with the next tile's loads.
- The canonical GEMM is a 3-stage concurrent pipeline: TMA → SMEM,
  tcgen05 SMEM→TMEM, epilogue TMEM→GMEM, all overlapped. [Modular]
- Well-tuned pipelines have essentially one bubble (first accumulator
  read-out) every few hundred μs. [TK Blackwell]
- CTA pairs replace some of Hopper's cluster tricks: pairing halves per-SM B
  traffic, which is why smem didn't need to grow. [SemiAnalysis]

## Performance expectations vs H100

- ⚠️ Tensor cores ~2–2.5× H100 per GPU (BF16 ~1 → ~2.25 PFLOPS dense);
  FP4 adds another ~2× over FP8. NVIDIA's "30×" inference claim is
  cherry-picked (FP4-vs-FP8, huge TP, long-context); expect per-precision
  ~2–2.5×. [SemiAnalysis; Tri Dao]
- ⚠️ HBM ~2.3× H100 → elementwise/reduction/decode kernels scale by
  bandwidth, not tensor cores.
