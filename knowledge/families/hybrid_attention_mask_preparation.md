# Op: hybrid_attention_mask_preparation

One distilled line per finished problem.

- task 28 (reduction_28): best=0.8621 tier=0 via "Single per-shape CUDA-graph-cached Triton kernel builds only the unique [T,S] full-causal bool pattern with per-shape ti" [budget:time]
- task 28 (reduction_28): best=0.8621 tier=0 via "Single per-shape CUDA-graph-cached Triton kernel builds only the unique [T,S] full-causal bool pattern with per-shape ti" [ceiling_consensus]
- task 28 (reduction_28): best=0.8646 tier=0 via "Per-shape CUDA-graph-cached Triton kernel with simplified store and improved tile-size thresholds; removed .cs cache mod" [ceiling_consensus]
