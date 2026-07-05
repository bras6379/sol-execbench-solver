# Chinese Lab OSS Kernels — B200 Status & Techniques

Deltas beyond [oss-kernel-catalog.md](oss-kernel-catalog.md), from the
Chinese-ecosystem research pass (verified against repos 2026-07-04).
Tags per kb/README.md; each item notes B200 applicability.

## DeepGEMM (deepseek-ai, MIT) — B200: directly applicable ✅

- ✅ Blackwell-native since the 2025-07-20 refactor (CUDA ≥12.9 for SM100),
  NOT a Hopper port. SM100-specific behavior:
  - **All four memory layouts (NT/TN/NN/TT)** on SM100 vs NT-only on SM90.
  - Scales in **packed UE8M0 (4 per torch.int)**, not FP32 — use the
    `get_mn_major_t...` helper; callers must adapt ⚠️.
- ✅ **Mega MoE megakernel (2026-04-16, shipped in `deep_gemm/mega`)**: fuses
  EP dispatch → linear1 (FP8×FP4) → SwiGLU → linear2 (FP8×FP4) → EP combine
  into ONE kernel, overlapping NVLink comm with tensor-core compute.
  FP8×FP4 MMA is SM100/tcgen05-only. Needs PyTorch ≥2.9 ⚠️. This is the
  state-of-the-art reference for our MoE fusion problems.
- ✅ Grouped-GEMM layouts (2-1 vote on framing): contiguous
  (`m_grouped_fp8_gemm_{nt,nn}_contiguous`, training-fwd/prefill, M-segments
  aligned to block size) and masked (`m_grouped_fp8_gemm_nt_masked`, decode
  under CUDA graphs, composes with DeepEP). Groups along M only; N,K fixed.
  Plus a K-grouped variant for MoE backward ⚠️.
- ⚠️ Recipe details: e4m3 + FP32 accumulation, one scale per 128 K-columns;
  core kernel ~300 lines; JIT-everything with per-shape/arch cache in
  `$HOME/.deep_gemm`; NVRTC backend ~10× faster compiles; 1350+ FP8 TFLOPS
  on Hopper.

## FlashMLA (deepseek-ai, MIT) — B200: directly applicable ✅

- ✅ Support matrix: Dense Decoding **SM90-only**; Sparse Decoding SM90+SM100;
  Dense Prefill **SM100-only**; Sparse Prefill SM90+SM100. SM100 MHA kernels
  were contributed by NVIDIA (2025-08-01).
- ✅ B200 numbers: MHA prefill up to 1460 TFLOPS fwd / 1000 TFLOPS bwd;
  sparse MLA prefill up to 1450 TFLOPS; sparse FP8-KV decoding up to 350
  TFLOPS **with the repo's own caveat "not really optimized yet"** — i.e.
  sparse decode has known headroom.
- ✅ Token-level sparse attention kernels released 2025-09-29 with
  DeepSeek-V3.2.

## SGLang / sgl-kernel — B200: directly applicable ✅

- ✅ GB200 bringup done 2025-06-16 (PD disaggregation + large-scale EP);
  13,149 prefill tok/s/GPU at ISL 4096; later official results: 18,471
  (BF16 attn / FP8 MoE) and **26,156 t/s/GPU (FP8 attn / NVFP4 MoE)** at ISL
  2000 — NVFP4 MoE in production.
- ✅ (medium) Dependency reality: SGLang's Blackwell stack runs on a
  **DeepGEMM fork** (sgl-project/DeepGEMM; historical `blackwell` branch now
  deleted; as of 2026-07 it's `sgl-deep-gemm==0.1.4` wheels from fork
  release branches — still fork-not-upstream) + DeepEP. **Check SGLang's
  pinned versions, not upstream HEADs, when reproducing.**
- ⚠️ Cautionary datapoint (Aug 2025, tracking issue): DeepSeek-V3 FP8 on
  B200 via FlashInfer backend + TP4 initially only matched H100 throughput —
  B200 wins are not automatic; they came from the kernel/EP work above.
  Docker channel: `lmsysorg/sglang:b200-cu129`.

## DeepEP — B200: works, one PTX landmine ⚠️→✅

- ✅ Runs on B200 clusters: +32% pretraining throughput for DeepSeek-V3 671B
  on 256×B200 (651→859 tok/s); +41% combined with MXFP8 grouped GEMMs
  (918 tok/s). Techniques that transfer: pipelined NVLink→RDMA forwarding,
  GPU-initiated RDMA (NVSHMEM/IBGDA), **configurable SM allocation for comm
  kernels** (the H800 20-SM trick, tunable on B200). [PyTorch/torchtitan blog]
- ✅ **PTX landmine**: DeepEP's documented
  `ld.global.nc.L1::no_allocate.L2::256B` trick for polling volatile flags
  is architecturally unsafe on ALL sm70+ including Blackwell —
  `L1::no_allocate` skips hit-on-miss, so a stale L1 line never refetches;
  use `ld.global.cg` for cross-SM polling. DeepSeek maintainer accepted the
  analysis (2025-04-28) and committed to fixing. **Never copy this pattern
  into our kernels**; check DeepEP version if reusing its loops.

## Cross-references

- SageAttention family → [sageattention-and-quant-papers.md](sageattention-and-quant-papers.md)
- DeepSeek-V3-report tricks (two-level accumulation, SM allocation) →
  [transferable-hopper-tricks.md](transferable-hopper-tricks.md)
- Kimi/MiniMax/Alibaba/ByteDance kernels: searched; nothing survived into the
  verified set this pass — revisit if a problem matches their ops (lightning/
  linear attention).
