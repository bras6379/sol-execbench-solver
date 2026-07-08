# Op: multimodal_rope_position_computation_with_grid_based_indexing

One distilled line per finished problem.

- task 23 (layernorm_23): best=0.0591 tier=0 via "seed" [budget:time]
- task 23 (layernorm_23): best=0.7675 tier=0 via "Precompute freqs=outer(seq,inv_freq) in meta, single output buffer [T,1792] → 1 clone instead of 3: 5→3 launches (copy+g" [budget:time]
- task 23 (layernorm_23): best=0.8133 tier=0 via "Sync-once vectorized PyTorch + double-buffered CUDA graph: one graph launch computes bilinear gather/sum and MRoPE cos/s" [ceiling_consensus]
- task 23 (layernorm_23): best=0.9369 tier=0 via "Same double-buffered CUDA graph + fused-Triton-bilinear-gather pipeline as the 0.891 parent, but the RoPE cos/sin chain " [ceiling_consensus]
