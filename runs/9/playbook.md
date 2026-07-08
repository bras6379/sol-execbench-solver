# Playbook — task 9 · conv_9

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `1f68f8d9` — Eager torch, targeted TF32: run grad_gated (G2) + the 3 K=N wgrad GEMMs (G1/G3/G5) in TF32 (fp32 operands, 10-bit mantis
Higher-ceiling idea NOT shipped: a custom Triton wgrad kernel for G1/G3/G5 that
takes fp32/TF32 inputs but writes the (F,H)/(H,F) output ONCE as bf16 via a fused
cast epilogue — plus a Triton G2 that loads w2 as bf16 (117MB) and upcasts on-chip
to TF32 instead of the eager 234MB fp32 w2 read. Eager currently pays ~1GB of
avoidable traffic (three fp32 wgrad outputs round-tripped through .to(bf16) + the
w2 fp32 cast), which dominates the 13 memory-bound small-N shapes whose SOL is the
~0.10ms weight floor.

TRIGGER: if this passes all 16 (unlocks sol_score) but the memory-bound shapes
(N<=~300)

## 2. from `a49df0b8` — Targeted TF32 eager kernel: run G2 and the K=N wgrads G3/G5 in TF32 tensor-core GEMMs to eliminate the N>=215 rounding f
If this passes all 16 but the memory-bound shapes (N <= ~300) stay above ~1.15x SOL, the next lift is a custom Triton wgrad kernel for G1/G3/G5 that reads bf16 activations and weights, upcasts them inside the dot, and writes the (H,F)/(F,H) weight-grad outputs directly as bf16, plus a Triton G2 that loads w2 as bf16 and upcasts on-chip to TF32—this removes the eager fp32 copies of gated_output, selected_tokens, and w2_weight.

## 3. from `3afa044f` — Keep the proven 0.721 TF32-selective precision plan (G2/G3/G5 TF32, G1/G4/G6 bf16) unchanged; wrap everything except the
If this only ties or barely beats 0.721, the CUDA-graph replay path is likely never triggering (i.e. `entry["ptrs"] == ptrs` fails every call) -- check whether the harness actually reuses the same input tensor objects across a workload's warmup+timed loop, or regenerates fresh allocations per call; if the latter, switch to a copy-in static-buffer graph (copy live inputs into persistent capture buffers each call, like problem-007's cuFFT pattern) instead of binding directly to live pointers.

If it beats 0.721 but the 3 compute-bound shapes (idx 7/10/13, N>=496) are still the laggards, retry th

## 4. from `f3e2efa4` — Targeted-TF32 eager cuBLAS path from the best frontier kernel, with final bf16 dgrad accumulation folded through addmm a
Higher-ceiling idea not shipped: build a fixed-shape copy-in CUDA graph that stages live inputs into persistent buffers, then replays the proven targeted-TF32 op sequence so launch overhead is removed even if the harness changes input allocations. Try this if the memory-bound N<=296 shapes still tie the 0.721 eager kernel or if a direct live-pointer graph does not replay.

## 5. from `65d58c25` — Targeted-TF32 cuBLAS path: G2/G3/G5 use fp32 operands with TF32 tensor cores for correctness, G1/G4/G6 stay bf16, and G6
Higher-ceiling idea not shipped: keep G2 in TF32 but add shape-bucket precision for G3/G5, using bf16 wgrad inputs only for N=64/128/196 and TF32 for N>=215. Try it only if the current TF32 frontier ties the parent again and per-shape correctness logs confirm the small-N wgrad path has enough tolerance headroom.

