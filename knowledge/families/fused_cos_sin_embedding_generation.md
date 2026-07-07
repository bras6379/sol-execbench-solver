# Op: fused_cos_sin_embedding_generation

One distilled line per finished problem.

- task 12 (softmax_12): best=0.9184 tier=0 via "Fixed-grid CUDA RoPE: 2 freqs/thread via float2 load + packed __nv_bfloat162 stores to both output halves, natural grid " [budget:time]
- task 12 (softmax_12): best=0.9195 tier=0 via "Replace FP64 range reduction (x - 2π*rint(x/2π)) with fp32 Cody-Waite (nearbyint + FMA) to shorten per-thread latency ch" [ceiling_consensus]
