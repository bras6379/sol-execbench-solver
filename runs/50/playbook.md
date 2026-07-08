# Playbook — task 50 · rmsnorm_50

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `d88ead87` — Single autotuned Triton kernel computes Q/K/V in one launch via a virtual-concat 12-tile grid (8 Q-tiles + 2 K-tiles + 2
Did NOT ship CUDA-graph capture this round (skipped deliberately: correctness-critical, unverifiable without GPU access, and every workload here is launch-overhead-bound per DESIGN.md's roofline table, so it's the biggest remaining lever). Next round: wrap `_launch` in a per-(M, input-data_ptr-tuple) cache of a captured `torch.cuda.CUDAGraph` (warm up 2-3 eager calls first so triton.autotune's search finishes outside capture, then capture once), replay when all 7 input pointers still match the capture-time pointers, else re-capture/fall back to eager `_fused_qkv_kernel` — never return the grap

## 2. from `2622a4ba` — Expanded Triton autotuning: 12 configs covering BLOCK_M ∈ {16,32,64,128,256}, BLOCK_K ∈ {64,128}, num_warps ∈ {2,4,8}; s
CUDA-graph capture for M ≤ 2048 is the remaining lever. Capture per (M, input-data-ptr-tuple) on first call with pointer-guarded replay on subsequent calls; clone outputs (never return static buffers). Gate graphs to M ≤ 2048 since compute time (4.6-9.1 µs) becomes comparable to launch overhead (2-10 µs) at larger M. Ship if GPU testing confirms input/output pointer stability within a workload's 60-call loop.

