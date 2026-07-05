# Software Stack for B200 Kernels

Tags: ✅ verified · ⚠️ unverified (see kb/README.md legend).

## CUDA toolkit & compute capability

- ✅ **tcgen05 requires architecture-specific targets `sm_100a`/`sm_101a`** —
  it is NOT available on base `sm_100`. Family-conditional `sm_100f` targets
  exist from PTX ISA 8.8 (later PTX adds sm_103a/sm_110a for Blackwell
  Ultra). Build tcgen05 kernels with `-arch=sm_100a`-style flags.
- ⚠️ The compatibility guide describes CUDA 12.8 as the first toolkit emitting
  native Blackwell cubins; the PTX ISA changelog maps ISA 8.6 (first with
  sm_100/sm_100a + tcgen05) to CUDA 12.7 / driver r565 ✅. In practice: use
  **CUDA ≥ 12.8**, ideally 13.x.
- ⚠️ Base targets: `sm_100` (cubin) / `compute_100` (PTX) for
  non-arch-conditional code.
- ⚠️ Recommended flags:
  `-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_100,code=compute_100`
  (add `sm_100a` when using tcgen05). `CUDA_FORCE_PTX_JIT=1` verifies PTX-JIT
  compatibility.
- ⚠️ Old apps run on Blackwell only via embedded PTX; cubins are
  backward-compatible only within a major CC. `compute_90a` PTX (wgmma) does
  **not** JIT to Blackwell.
- ⚠️ CUDA 13.0: preliminary TMEM/CTA-pair support in tooling; gau-nernst's
  tcgen05 work was benchmarked on CUDA 13. [arxiv 2512.02189; gau-nernst]

## PTX ISA versions — resolved ✅

**PTX ISA 8.6 (CUDA 12.7, driver r565) introduced the full tcgen05 family**
(alloc, dealloc, ld, st, wait, cp, shift, mma, mma.sp, mma.ws, fence, commit)
on sm_100a/sm_101a. sm_100f family targets from PTX ISA 8.8.
(❌ "PTX ISA 8.4" from the 0xsero wiki is wrong; "8.7" from arxiv 2507.10789
was also off.)

- ⚠️ As of CUDA 12.9, FP4 (e2m1) / FP6 (e2m3, e3m2) `mma.sync` requires the
  explicit `.kind::f8f6f4` suffix; omitting it or using these formats on
  Hopper is a PTX error. [arxiv 2507.10789 — note this paper measured
  *consumer* sm_120a, but this is a PTX-level rule; sanity-check on sm_100a]

## Datacenter vs consumer Blackwell

- ⚠️ **tcgen05 and TMEM exist only on datacenter Blackwell (sm_100)**.
  Consumer/workstation Blackwell (sm_120) has no TMEM, no clusters > 1, no
  tcgen05; CC 12.0 also has only 128 KB smem/SM and 48 warps/SM ✅.
  Anything you read that was measured on an RTX 50-series does not transfer.

## CUTLASS

⚠️ [CUTLASS Blackwell functionality docs]

- Full SM100 support: 1-SM and 2-SM GEMM schedules via dispatch policies
  `KernelTmaWarpSpecialized1SmSm100` / `KernelTmaWarpSpecialized2SmSm100`;
  2-SM MMA tiles 128×64 … 256×256.
- Narrow-precision types: `float_e2m1_t` (FP4), `float_e2m3_t`/`float_e3m2_t`
  (FP6), FP8 `e4m3`/`e5m2`, scale-factor types `float_ue8m0_t`/`float_ue4m3_t`
  for MX/NVF4 block scaling (scale vector 16 or 32).
- **Alignment pitfall**: unpacked float4_t/float6_t operands need 128-element
  alignment; float8_t needs 16-element alignment when mixed with 4/6-bit
  types.
- CUTLASS keeps separate template trees for SM100 (tcgen05) and SM120. ⚠️
  [0xsero]
- Colfax's tutorial series ("Writing GEMM kernels using Tensor Memory") is
  the best walkthrough of CUTLASS-on-Blackwell internals.

## Triton

- ⚠️ Plain Triton lowers to Blackwell but its attention kernel was 2.1–2.7×
  slower than FlashAttention-4 [Tri Dao] — the compiler doesn't yet exploit
  the full tcgen05 pipeline on its own.
- ⚠️ **Gluon** (Triton's lower-level API, `python/tutorials/gluon/06-tcgen05.py`)
  exposes tcgen05 directly: TMEM alloc, `tcgen05_mma`, `tcgen05_commit`
  mbarrier completion, implicit-pipelining rules, operand placement rules
  (acc in TMEM, A in SMEM/TMEM, B in SMEM + NVMMASharedLayout).
  Requires CC major == 10. Measured FP16 matmul (8192×8192×16384):
  ~1020 TFLOPS unpipelined → 1113 TFLOPS with software pipelining
  (BLOCK 128×128×64, 8 warps). blockN=128 is typically the optimal tcgen05
  shape.

## Libraries

- cuBLAS: BF16 GEMM ≈ 1763 TFLOPS on B200 (~78% of dense peak) ⚠️ [Modular] —
  this is the bar SOL-ExecBench baselines likely sit near.
- cuDNN 9.13 attention: FA4 beats it by 1.1–1.3× ⚠️ [Tri Dao].
- ThunderKittens has Blackwell GEMM/attention templates at cuBLAS/cuDNN
  level ⚠️ [HazyResearch].
