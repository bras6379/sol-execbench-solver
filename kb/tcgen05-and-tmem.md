# 5th-Gen Tensor Cores: tcgen05 and Tensor Memory (TMEM)

Tags: ✅ verified · ⚠️ unverified (see kb/README.md legend).

## The headline change

- ✅ Hopper's `wgmma` instructions **do not run on Blackwell**. Warp-group
  matrix computation uses the new `tcgen05.*` family instead.
  [compatibility guide (`compute_90a` PTX won't JIT); Colfax ("WGMMA is
  deprecated on Blackwell")]
- ✅ `tcgen05.mma` has **single-thread issue semantics** — one thread of the
  CTA (or of the CTA pair for `.cta_group::2`) initiates the whole async MMA
  (like TMA), unlike collective `mma.sync` / `wgmma.mma_async`. [PTX ISA]
- ✅ Introduced in **PTX ISA 8.6** (CUDA 12.7, driver r565); available only on
  architecture-specific targets **sm_100a/sm_101a** — NOT base sm_100
  (family-conditional sm_100f from PTX ISA 8.8). See
  [software-stack.md](software-stack.md).
- ❌ The widely-repeated "~11-cycle tcgen05 issue latency vs 32-cycle wgmma"
  microbenchmark claim **failed adversarial verification** (0-3 / 1-2 votes) —
  do not design around specific issue-latency cycle counts.
- ⚠️ NVIDIA/CUTLASS: the 7 new tcgen05.mma instructions are 2–4× faster than
  WGMMA; block-scaled FP4 variants are 4× Hopper FP8 throughput, and
  tf32/f16/i8/f8f6f4 variants 2× their Hopper counterparts. [CUTLASS docs]

## Tensor Memory (TMEM)

- ✅ Dedicated on-chip memory for 5th-gen tensor cores on sm_100a/sm_100f:
  **512 columns × 128 lanes × 32-bit cells per CTA = 256 KB per SM**.
  [PTX ISA; corroborated by Colfax, SemiAnalysis, Chips and Cheese]
- ✅ TMEM addresses encode **lane in bits 31:16, column in bits 15:0**.
  Accessible only via `tcgen05.ld/st/cp/mma` — it is not a general-purpose
  register file. [PTX ISA]
- Same capacity as the register file (64K × 32-bit) — think of it as a second
  register layer dedicated to accumulators ⚠️. [Triton Gluon tutorial;
  HazyResearch]

### Allocation — ✅ [PTX ISA]

- Allocated dynamically by a **single warp** per CTA via `tcgen05.alloc`,
  in units of **32 columns**; column count must be a **power of 2**.
- Allocating a column allocates all 128 lanes of it.
- **All TMEM must be explicitly deallocated (`tcgen05.dealloc`) before kernel
  exit** — forgetting this hangs/fails the kernel.
- ⚠️ Allocation does not directly reduce occupancy, but blocks when TMEM is
  exhausted; TMEM-using kernels get 1 CTA/SM in practice. [Gluon tutorial;
  HazyResearch TK2]

### Per-warp lane restriction — ✅ [PTX ISA]

CTA TMEM is split into 4 lane chunks; each warp of a warpgroup accesses only
its own 32 lanes (warp 0: lanes 0–31, warp 1: 32–63, warp 2: 64–95, warp 3:
96–127). All warps see all columns. Consequences:

- Epilogues need a **full warpgroup (4 warps)** to read a 128-lane
  accumulator out of TMEM (`tcgen05.ld`). CUTLASS `make_tmem_copy` is
  hardcoded to 4 warps ⚠️. [Colfax]

### Operand placement — ⚠️ [Gluon tutorial; Colfax; SemiAnalysis]

- Accumulator (D): **must live in TMEM**, transparent row-major layout — no
  registers consumed, no wgmma-style thread-value layout puzzles.
- A operand: SMEM or TMEM. B operand: SMEM only (NVMMASharedLayout in
  Gluon terms).
- SMEM operands are described by a shared-memory matrix descriptor with LBO
  (leading byte offset) / SBO (stride byte offset) fields — poorly documented;
  see gau-nernst's writeup.

## MMA shapes and CTA pairs (2-SM MMA)

- ⚠️ 1-CTA shapes: M ∈ {64, 128} × N up to 256 × K16; largest 1-SM atom
  128×256×16 — 2× the largest wgmma atom. [Colfax]
- ✅ `.cta_group::2`: one MMA spans the TMEM of both CTAs of a **CTA pair** —
  two CTAs in a cluster whose `%cluster_ctarank` differs only in the last
  bit. [PTX ISA]
- ⚠️ The pair maps onto the two SMs of a TPC; it doubles M (each CTA keeps
  BLOCK_M=128 → effective 256) and **splits/shares the B operand across both
  SMs, halving per-SM SMEM loads of B**. This is why smem didn't grow vs
  Hopper. [SemiAnalysis; gau-nernst; Tri Dao]
- ❌ Do NOT overstate this: "2-SM UMMA halves total operand movement /
  doubles arithmetic intensity vs two independent 1-CTA MMAs" was refuted
  0-3 in the methodology pass. The benefit is per-SM B-operand SMEM traffic,
  not a 2× AI gain.
- ✅ 2-SM UMMA supports only **M ∈ {128, 256}**, with the accumulator always
  split across the CTA pair along M — constrains valid tile/cluster combos
  in the tune space. [methodology pass, Colfax]
- ⚠️ CUTLASS 2-SM MMA tile shapes range 128×64 up to 256×256. [CUTLASS docs]
- ✅ ThunderKittens exposes 2-SM MMA via template parameter `ncta=2`
  (emits `tcgen05.*.cta_group::2`) — verified in their code.

## Asynchrony and synchronization

- ✅ `tcgen05.mma` is async; completion is tracked with mbarriers
  (`tcgen05.commit`). [PTX ISA — issue semantics ✅; pipeline details ⚠️]
- ⚠️ Implicit pipelining guarantees (no explicit sync needed) between:
  same-shape/same-acc-dtype back-to-back MMAs; MMA followed by
  `tcgen05.commit`; `tcgen05.cp` adjacent to MMA. [Gluon tutorial]
- ⚠️ `tcgen05.cp` is implicitly pipelined w.r.t. `tcgen05.mma` — merging
  scale-copy + MMA issue into one thread recovered ~500 TFLOPS (~10%) on an
  NVFP4 GEMM. [HazyResearch TK2]
- ⚠️ PTX proxy fences between TMA writes and tensor-core reads are unnecessary
  when synchronized via mbarrier acquire; removing them gained 20–30 TFLOPS.
  [HazyResearch TK2]

## Instruction family cheat sheet — ✅ [PTX ISA 8.6/8.7 changelogs]

`tcgen05.alloc / .dealloc / .relinquish_alloc_permit` (TMEM lifecycle),
`.mma / .mma.sp / .mma.ws` (MMA, sparse, weight-stationary), `.ld / .st`
(TMEM ↔ registers), `.cp` (SMEM → TMEM), `.commit` (mbarrier completion),
`.wait`, `.fence`, `.shift`. Async tcgen05 ops are **unordered** unless they
form a documented implicit pipeline or are synchronized via `tcgen05.commit`.

MMA kinds: `.kind::f16`, `.kind::tf32`, `.kind::f8f6f4`, `.kind::i8`, and
block-scaled `.kind::mxf8f6f4`, `.kind::mxf4`, `.kind::mxf4nvf4`
(e2m1 = 2 exp + 1 mantissa bits). Scale-factor types `ue8m0`/`ue4m3`, scale
vector size 16 or 32; block-scaled math:
`D_ij = C_ij + Σ_k (A_ik · SFA_{i,k/SV}) · (B_jk · SFB_{j,k/SV})`.
[CUTLASS docs]

- ⚠️ tcgen05 compiles to precision-specific SASS: HMMA (f16), QMMA (fp8),
  OMMA (fp4), IMMA (int), DMMA (fp64). [arxiv 2512.02189]

## Systolic-array behavior

- ✅ (medium confidence — empirical inference, quantitatively confirmed by two
  independent microbenchmarks) The 5th-gen tensor core behaves like a 128×128
  systolic array: a 64×64×64 MMA runs at ¼ the rate of 128×128×64, and 1-SM
  tcgen05 MMA with M=64 caps at ~50% of peak while M=128 reaches ~100%.
  **Keep M/N tile dims at 128 (or 256 via CTA pairs).** Skinny-M workloads
  (decode GEMMs) inherently underutilize the tensor cores.
  [HazyResearch TK Blackwell; arxiv 2512.02189; gau-nernst]
