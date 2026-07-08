# Playbook — task 49 · conv_49

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `2140585c` — Head-stacking reshape: (B,4,S,D)→(B,4S,D) and (B,1,S,D)→(B,S,D) for one efficient M=4S GEMM per batch; bf16-native (no f
If this scores below frontier, switch to custom Triton kernel (Approach 1) with persistent grid-stride loop and explicit strided-batch GEMM via TMA multicasting, targeting only the 4 genuine bandwidth-bound shapes (#2/#3/#10/#14) while CUDA-graphing the 12 launch-bound shapes.

## 2. from `683c1963` — Head-stacked (B,4S,D)x(B,D,S) bf16 GEMM with power-of-two scaling folded into K, CUDA-graphed for the 12 launch-bound sh
If this still fails correctness on any workload (i.e. disabling `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` doesn't fully close the gap the reviewer measured), fall back to the reviewer's own verified-exact fix: upcast `query_flat`/`key_scaled` to fp32 before `torch.matmul` in `_compute` (0.00e+00 error confirmed by the reviewer on shape #5) — this sacrifices bf16 tensor-core throughput on the compute-heavier shapes (roughly half the 16, not just the 4 bandwidth-bound ones — several of the "launch-bound" shapes like #7/#8/#9/#13 have large enough FLOP counts that fp32-v

## 3. from `95650786` — Single-launch bf16 GEMM: free head-stacking reshapes (no K materialization) + scaling fused into torch.baddbmm's alpha e
If this round's score improves but shapes #2/#3/#10/#14 (the genuine bandwidth-bound ones) still sit notably below ~0.6-0.7 raw, cuBLAS's batched-GEMM heuristic isn't hitting peak achievable HBM% on this skinny-K=256 shape (best on-hardware run so far was still ~1.3-2x off t_SOL there) — switch to DESIGN.md Approach 1: a custom persistent Triton kernel with explicit tile/split-K bandwidth tuning (and optionally TMA multicast on the K operand) targeting only those 4 shapes, while keeping this single-launch baddbmm path for the 12 launch-bound ones.

## 4. from `140b67e1` — Single-launch bf16 GEMM with head-stacking (no materialized K repeat) and scaling fused via baddbmm's alpha; removed tor
If score ties or slightly improves on launch-bound shapes (#5,#6,#12,#1,#11,#15,#4,#0,#13,#7,#9,#8) but bandwidth-bound shapes (#2,#3,#10,#14) remain below 0.6-0.7 raw score, cuBLAS's heuristic isn't saturating HBM bandwidth on these skinny-K=256 shapes. Switch to DESIGN.md Approach 1: custom persistent Triton kernel targeting only the 4 bandwidth-bound shapes with explicit tile/split-K tuning and TMA multicast on K (exploiting L2 sharing across the cluster), while keeping this baddbmm path for all 12 launch-bound shapes (where dispatch matters, not bandwidth).

