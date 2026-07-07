# Op: fused_rope_with_qk_norm_and_kv_cache_update

One distilled line per finished problem.

- task 18 (reduction_18): best=0.8811 tier=0 via "One-launch Triton fusion of per-head RMSNorm, RoPE, and KV-cache scatter-write with per-(batch,seq) cos/sin computed onc" [budget:time]
- task 18 (reduction_18): best=0.8972 tier=0 via "Reload the 0.848 one-launch Triton fusion (RMSNorm+RoPE+KV-cache scatter) and apply the untried vectorization lever: las" [budget:time]
- task 18 (reduction_18): best=0.8972 tier=0 via "Reload the 0.848 one-launch Triton fusion (RMSNorm+RoPE+KV-cache scatter) and apply the untried vectorization lever: las" [ceiling_consensus]
