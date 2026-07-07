# Playbook — task 40 · rmsnorm_40

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `b13c5842` — fp16 + channels_last cuDNN convs (fp32-accumulate) — half-precision intermediate halves the dominant HBM term & 2x tenso
Higher-ceiling idea NOT shipped: a single fused kernel (Triton, then CUDA/CuTe) that
tiles the output spatially, keeps BOTH weight matrices (~7 KB) in registers/smem,
computes conv_in -> (round to fp16) -> conv_out -> residual entirely on-chip, and
writes the output ONCE — eliminating the (B,32,H,W) intermediate GMEM round-trip
(10.7x the I/O), pushing arithmetic intensity from ~11 to ~144 (the fp16/TF32 ridge).
Process conv_in in output-channel groups (e.g. 8 at a time), immediately feeding
conv_out so the 32-ch intermediate never fully materializes. Round the intermediate
to fp16 at the conv

## 2. from `d8d5b038` — FP16 cuDNN tensor-core convs with per-regime torch.compile dispatch: reduce-overhead CUDA graphs for launch-bound small
If this kernel only ties or marginally beats the 0.520 frontier, the next play is a spatially-tiled Triton (or CuTe) fused direct-conv kernel: keep both 3x3 weight sets in smem, compute conv_in's 32 hidden channels in groups of 8, immediately accumulate conv_out's 3 output channels in registers, add the residual, and write the output once—eliminating the (B,32,H,W) fp16 intermediate GMEM round-trip entirely.

## 3. from `48b17b8e` — fp16 channels-last cuDNN residual block captured in a per-shape CUDA graph with eager fallback; inputs copy to static bu
Reserve play: if this CUDA-graph-only version fails to reach ~0.55 or shows copy/static-buffer overhead eating the savings, switch to a spatially-tiled fused Triton/CuTe direct-conv megakernel. Keep both 3x3 weight sets (~7 KB) in smem, compute conv_in's 32 hidden channels in groups of 8, immediately accumulate conv_out's 3 output channels in registers, round the intermediate to fp16 on-chip, add the residual, and write the output once—eliminating the (B,32,H,W) fp16 GMEM round-trip entirely. This pushes arithmetic intensity from ~11 to ~144 and is the only remaining lever after launch overhea

## 4. from `6a82558c` — Single CUDA graph capture of fp16 channels_last convs (no NCHW fallback) for all 16 workloads.
If this kernel only ties or marginally beats 0.520, switch to a spatially-tiled fused Triton/CuTe direct-conv megakernel: keep both 3x3 weight sets (~7 KB) in smem, compute conv_in's 32 hidden channels in groups of 8, immediately accumulate conv_out's 3 output channels in registers, round intermediate to fp16, add residual, and write output once—eliminating the (B,32,H,W) fp16 GMEM round-trip entirely.

