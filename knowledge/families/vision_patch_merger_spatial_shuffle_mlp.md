# Op: vision_patch_merger_spatial_shuffle_mlp

One distilled line per finished problem.

- task 20 (rmsnorm_20): best=0.7187 tier=0 via "Single fused LN+spatial-shuffle Triton kernel with INLINE per-program source-index computation (no perm tensor, no perm-" [budget:time]
- task 20 (rmsnorm_20): best=0.7187 tier=0 via "Single fused LN+spatial-shuffle Triton kernel with INLINE per-program source-index computation (no perm tensor, no perm-" [ceiling_consensus]
- task 20 (rmsnorm_20): best=0.7325 tier=0 via "Same fused single-launch LN+shuffle Triton kernel (inline shuffle-index derivation, zero host syncs) + cuBLAS FC1/GELU/F" [ceiling_consensus]
