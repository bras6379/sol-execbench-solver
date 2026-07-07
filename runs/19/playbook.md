# Playbook — task 19 · conv_19

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `4ef589bc` — Single Triton kernel fusing RoPE backward, pairing dims d/d+64 to compute grad_q, grad_k, and head-reduced grad_embeddin
If this only ties the optimized PyTorch baseline, try a CUDA C++ persistent grid-stride kernel with one CTA per sequence row pair-block, explicit float4 vectorized loads/stores, and staged sincos/embedding values to cut Triton register pressure and address arithmetic.

## 2. from `dde06f2f` — Autotuned single fused Triton RoPE-backward (traffic already at the 5-read/3-write floor); tune the occupancy<->MLP spec
HIGHER-CEILING RESERVE (not shipped): this is a pure streaming kernel with zero
intra-kernel reuse, so add L2-bypass cache hints to stop polluting L2 with
read-once data — put `eviction_policy="evict_first"` (or `cache_modifier=".cg"`)
on the 8 big input `tl.load`s and streaming stores on the 4 grad_q/grad_k writes,
and fold that choice into the autotune space (an EVICT: tl.constexpr flag).
TRIGGER: if this autotuned kernel plateaus at/below ~0.70 raw on the large
seq_len shapes (>=4096) where it's HBM-bound — the win there is achieved-BW%,
and evicting streamed data frees L2 for the writes.
D

## 3. from `635aeb80` — Fused Triton RoPE-backward with per-seq_len autotuned tile sizes and an L2-bypass EVICT variant (evict_first on the five
If this only matches the frontier (~0.70 raw) or is still bandwidth/launch-bound on small seq_len, switch to a persistent grid-stride kernel: a fixed ~SM-count grid atomically pulls BLOCK_S chunks, combines explicit float4/128-bit vectorized I/O with register-staged sincos, and adds num_stages pipelining to amortize launch overhead on the latency-bound shapes.

