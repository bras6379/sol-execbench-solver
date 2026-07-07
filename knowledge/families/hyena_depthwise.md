# Op: hyena_depthwise

One distilled line per finished problem.

- task 6 (moe_6): best=0.7566 tier=0 via "Two-path fused Hyena short-filter (depthwise causal conv1d k=3 lookback-2 + 768->3x256 split + v*x0 gate): proven straig" [budget:time]
- task 6 (moe_6): best=0.7677 tier=0 via "Vectorized fused Hyena short-filter: same proven two-path (simple<4096, persistent>=4096) conv1d k=3 + split + v*x0, but" [budget:time]
- task 6 (moe_6): best=0.7677 tier=0 via "Vectorized fused Hyena short-filter: same proven two-path (simple<4096, persistent>=4096) conv1d k=3 + split + v*x0, but" [ceiling_consensus]
