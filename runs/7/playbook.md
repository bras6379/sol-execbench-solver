# Playbook — task 7 · elementwise_7

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `ccbdf21e` — DFT-as-GEMM for small prime-L (Bluestein) shapes + cuFFT R2C with fused Triton epilogue (div+split) for rest; no CUDA gr
cuFFT R2C **store callback** to fuse the div+split directly into cuFFT's output write, eliminating the intermediate complex HBM buffer (saves ~1 full read+write of the (B,256,L+1) complex array — ~2 GB on idx6). The callback writes `element.x/2L` and `element.y/2L` directly into the two output arrays. Trigger to try: if this kernel still leaves the power-of-2 memory-bound shapes (L=4096,8192,32768) with S<0.7. Requires testing on B200/CUDA-13 for LTO callback compatibility with both Cooley-Tukey and Bluestein paths. If callbacks fail, fall back to a hand-written Stockham R2C FFT in CUDA for po

## 2. from `8c9be313` — cuFFT R2C + single fused Triton epilogue (1/(2L) scale + de-interleave split into two contiguous fp32 outputs); no CUDA
cuFFT R2C **store callback** to fuse div+split directly into cuFFT's output write, eliminating the intermediate (B,256,L+1) complex HBM buffer entirely — saves ~1 GB of HBM round-trip on the largest workload (B64 L8192). The callback receives each complex FFT output element and writes `real/2L` to out_r and `imag/2L` to out_i directly, collapsing the current 3.5 GB HBM traffic to the theoretical 1.5 GB minimum. Trigger: if the memory-bound power-of-2 shapes (L=4096,8192,32768) still score S<0.7 after this round. Implementation: write a .cu kernel that uses the cuFFT C API (`cufftXtSetCallback`

## 3. from `7536acee` — DFT-as-GEMM for small/Bluestein shapes (self-validates at 1e-5 and keeps it only if on-device timing beats cuFFT after a
Higher-ceiling idea not shipped: a cuFFT LTO store callback (or a hand-written fused Stockham R2C kernel for the pow2 sizes) that writes the scaled real/imag outputs directly during the FFT store, eliminating the intermediate (B,256,L+1) complex HBM buffer entirely. Trigger to try it: if the large power-of-2 memory-bound shapes (L = 4096, 8192, 32768) still score below ~0.7 SOL after this round.

