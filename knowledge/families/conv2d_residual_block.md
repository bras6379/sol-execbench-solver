# Op: conv2d_residual_block

One distilled line per finished problem.

- task 40 (rmsnorm_40): best=0.5621 tier=0 via "fp16 + channels_last cuDNN convs (fp32-accumulate) — half-precision intermediate halves the dominant HBM term & 2x tenso" [budget:time]
- task 40 (rmsnorm_40): best=0.6507 tier=0 via "Single CUDA graph capture of fp16 channels_last convs (no NCHW fallback) for all 16 workloads." [ceiling_consensus]
