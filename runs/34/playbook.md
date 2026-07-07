# Playbook — task 34 · gemm_34

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `2ba6b142` — Single-fused Triton kernel with 1D tiling over seq_len, 3-branch axis computation, and interleaved cos/sin store pattern
If score < ~0.95, wrap the fused kernel with torch.compile(mode='reduce-overhead') to eliminate the single remaining kernel launch overhead, targeting ~0.99 SOL on the 3 largest shapes (seq_len=4093, 4096, 3967).

## 2. from `659f93b5` — Single fused Triton kernel with precomputed freq_bands (axis 1/2 shared), 1D tiling over seq_len (BLOCK=64), and in-stor
If this scores below ~0.95, wrap the run() function body with torch.compile(mode='reduce-overhead', fullgraph=True) — the kernel is launch-bound and the single remaining ~5-7 µs kernel launch is the dominant cost; CUDA graph replay drops it to ~0.2 µs. Trigger: any shape scores < 0.95 SOL after this fused kernel lands.

