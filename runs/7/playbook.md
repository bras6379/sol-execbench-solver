# Playbook — task 7 · elementwise_7

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `ccbdf21e` — DFT-as-GEMM for small prime-L (Bluestein) shapes + cuFFT R2C with fused Triton epilogue (div+split) for rest; no CUDA gr
cuFFT R2C **store callback** to fuse the div+split directly into cuFFT's output write, eliminating the intermediate complex HBM buffer (saves ~1 full read+write of the (B,256,L+1) complex array — ~2 GB on idx6). The callback writes `element.x/2L` and `element.y/2L` directly into the two output arrays. Trigger to try: if this kernel still leaves the power-of-2 memory-bound shapes (L=4096,8192,32768) with S<0.7. Requires testing on B200/CUDA-13 for LTO callback compatibility with both Cooley-Tukey and Bluestein paths. If callbacks fail, fall back to a hand-written Stockham R2C FFT in CUDA for po

