# Playbook — task 43 · layernorm_43

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `a1d0cda7` — Keep the 3 dense GEMMs on cuBLAS (F.linear) but replace the reference's ~7 elementwise/copy ops (5-op RMSNorm, 2x .conti
Higher-ceiling idea not shipped: a custom epilogue-fused GEMM2 (1536->24576, the 71%-of-FLOPs GEMM) that writes q_nope/q_pe directly per-head from registers, skipping the (M,24576) `q` materialization entirely — saves ~2x full-tensor HBM traversal (~124us at M=8192, ~25% of that shape's compute time) but requires hand-matching cuBLAS's ~78%-of-peak throughput, which is the real risk. Trigger: if this round's fused-launch (cuBLAS + Triton RMSNorm/split) version only marginally beats the 0.303 baseline on the 13 compute-bound shapes, attempt the CUTLASS/Triton epilogue-fused GEMM2 next, validate

