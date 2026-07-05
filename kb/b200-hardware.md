# B200 Hardware Architecture

Tags: ✅ verified · ⚠️ unverified (see kb/README.md legend).

## Chip layout

- ✅ Dual-die design: 208 B transistors on TSMC 4NP, two reticle-limit dies
  joined by the 10 TB/s NV-HBI chip-to-chip interconnect into **one fully
  coherent chip** — kernel authors see a single CUDA device, not two.
  [NVIDIA Blackwell architecture page; B200 datasheet]
- ✅ **148 SMs** across 8 GPCs. [arxiv 2512.02189]
  - ⚠️ 74 SMs enabled per die out of 80 physically present — size grids for
    148, not 160. [Chips and Cheese]
- ✅ Compute capability **10.0** (`sm_100`); datacenter only. Consumer
  Blackwell is CC 12.0 with very different limits — numbers in this KB are
  for CC 10.0 unless stated.
- ⚠️ Die area ~1600 mm² total vs Hopper's ~800 mm² monolithic. [SemiAnalysis]

## Memory hierarchy

| Level | Size | Bandwidth | Notes |
|---|---|---|---|
| Registers | 64K × 32-bit per SM ✅ | — | 255 regs/thread cap ✅ |
| TMEM | 256 KB per SM ✅ | — | new; tensor-core accumulators only, see [tcgen05-and-tmem.md](tcgen05-and-tmem.md) |
| Shared memory | 228 KB per SM, 227 KB max per block ✅ | — | same as Hopper (H100: 228 KB) |
| L2 | 126 MB ✅ (tuning guide; Chips and Cheese measured ~126 MB on B200) | ~21 TB/s measured ⚠️ [HazyResearch TK2] | partition count uncertain: one paper says 4 (2-1 vote), Chips and Cheese speculates 2; H100 was 50 MB |
| HBM3e | HGX B200: 180 GB @ 7.7 TB/s ✅ · GB200 NVL72: 186 GB @ 8 TB/s ✅ | STREAM-measured ~7.48 TB/s ⚠️; plan on ~80–90% of peak achievable ✅ | the "192 GB" figure was GTC-2024 marketing, superseded by the 180 GB production spec ✅; use what the pod reports |

- ⚠️ Chip-to-chip (die-to-die) measured bandwidth ~10 TB/s. [HazyResearch TK2]
- Conflict note: the Modular blog claims 192 MB L2 — outweighed by the NVIDIA
  tuning guide and Chips and Cheese (126 MB). Use **126 MB**.

## Occupancy limits (CC 10.0) — all ✅ [NVIDIA Blackwell tuning guide]

- Max 64 concurrent warps per SM (2048 threads).
- Register file 64K × 32-bit per SM; 255 registers per thread max.
- Max 32 thread blocks per SM.
- Same envelope as Hopper → Hopper occupancy reasoning carries over.
- ⚠️ BUT: kernels that allocate TMEM are hard-limited to **1 CTA per SM**.
  [HazyResearch TK2]

## Thread block clusters

- ⚠️ Max portable cluster size 8; nonportable up to 16 with distributed
  shared memory. [tuning guide]
- ⚠️ Larger clusters reduce schedulable SMs on the 148-SM part:
  cluster 4 → 132 active SMs, 8 → 120, 16 → 112. [HazyResearch TK2]

## Tensor-core / ALU throughput (per GPU, HGX B200)

✅ Datasheet figures are quoted WITH sparsity (except FP64); dense = half.

| Format | Sparse (datasheet) | **Dense (use this)** |
|---|---|---|
| FP4 Tensor Core | 18 PFLOPS | **9 PFLOPS** |
| FP8 / FP6 Tensor Core | 9 PFLOPS | **4.5 PFLOPS** |
| INT8 Tensor Core | 9 POPS | **4.5 POPS** |
| FP16/BF16 Tensor Core | 4.5 PFLOPS | **2.25 PFLOPS** |
| TF32 Tensor Core | 2.2 PFLOPS | **1.1 PFLOPS** |
| FP32 (vector) | — | 75 TFLOPS |
| FP64 / FP64 Tensor Core | — | 37 TFLOPS |

- ✅ FP8 and FP6 share physical circuits → identical throughput. INT8 gains
  nothing over FP8 on this generation. (Also stated by SemiAnalysis ⚠️.)
- ✅ **FP64 tensor throughput regressed vs Hopper**: 37–40 TF on B200 vs 67 TF
  on H100. FP64-heavy problems will not see Blackwell uplift from tensor
  cores. Also, "FP32" in NVIDIA's datatype list means CUDA-core/accumulator
  support — tensor cores take TF32 inputs, not FP32.
- ✅ FP8 dense uplift vs H100 is ~2.27× (4500 vs 1979 TFLOPS) — the "~2–2.5×
  per precision" rule of thumb is exact here.
- ✅ Supported tensor-core datatypes: FP64, FP32, TF32, FP16, BF16, FP8, INT8,
  FP6, FP4, including OCP microscaling (MX) formats.
- ⚠️ Per-SM ceiling: CTA-level matrix ops sustain 1024 16-bit MACs/cycle per
  SM partition. [Chips and Cheese]
- ⚠️ SFU asymmetry: BF16 tensor cores do 8192 ops/cycle/SM but the exponential
  unit does only **16 ops/cycle/SM** (~512× gap) — softmax `exp` is the
  bottleneck attention kernels must pipeline around. [Tri Dao, FA4 blog]
- ⚠️ Non-tensor FP16 vector throughput is **no longer 2× FP32** — packed-half
  tricks in elementwise/reduction kernels stop paying. [Chips and Cheese]
- ⚠️ Measured FP64 DGEMM: 36.3 TFLOPS (80.7% of peak), 1.92× H200.
  [arxiv 2512.02189v3]

## Interconnect / power (context)

- ⚠️ NVLink 5: 18 links × 50 GB/s/dir = 1.8 TB/s bidirectional per GPU (2×
  Hopper). PCIe Gen5 128 GB/s. TDP up to 1000 W (HGX B200) / 1200 W (NVL72).
