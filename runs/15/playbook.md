# Playbook — task 15 · attention_15

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `2ba9d07b` — Cached concatenated QKV projection with cuBLAS, PyTorch SDPA native causal GQA to avoid materializing attention weights,
If this only ties the PyTorch SDPA baseline, replace it with a CuTe-DSL/FA4-style fused forward kernel that computes QKV tiles, applies per-head RMSNorm+RoPE in the producer, and runs online causal GQA softmax without writing Q/K/V or SxS scores to HBM. Trigger that path when profiling shows SDPA plus Q/K/V HBM traffic dominates after the projection GEMMs.

## 2. from `37985d2c` — Concat-QKV cuBLAS GEMM + SDPA flash (native causal GQA) + fused RMSNorm/RoPE, wrapped in hand-managed raw CUDA graphs (p
Higher-ceiling idea NOT shipped: fuse the QKV-projection epilogue (reshape -> per-head RMSNorm -> RoPE -> flash-layout write of Q/K/V) into ONE Triton producer kernel so the elementwise chain and its HBM round-trips are removed from inside the captured graph, then feed a hand-written flash-attention (or cuDNN graph-API SDPA) core. Compute RMSNorm in fp32 and round to bf16 exactly where the reference does; RoPE is rotate-half (out[:64]=x0*cos0-x1*sin0, out[64:]=x1*cos1+x0*sin1). Trigger: if this raw-CUDA-graph version still lands below ~0.6, the remaining gap is the unfused RMSNorm/RoPE/transpo

## 3. from `96a39648` — Fusion of QKV projection epilogue with elementwise ops (RMSNorm+RoPE) using cached concatenated weights, PyTorch SDPA wi
Higher-ceiling idea NOT shipped this round: fuse the QKV-projection epilogue (reshape -> per-head RMSNorm -> RoPE -> flash-layout write of Q/K/V) into ONE Triton producer kernel so the elementwise chain and its HBM round-trips are removed from inside the captured graph, then feed a hand-written flash-attention (or cuDNN graph-API SDPA) core.

Trigger: if this raw-CUDA-graph version (with CUDA-graph-wrapped pre-processing kernel) still lands below ~0.6, the remaining gap is in the RMSNorm/RoPE transpose+HBM traffic between the pre-processing kernel and the attention core. To beat ~0.7 SOL, impl

## 4. from `68a78981` — Reload the frontier-best (2ba9d07b, raw 0.416): cached concatenated QKV cuBLAS GEMM + fused RMSNorm/RoPE + native SDPA f
Higher-ceiling idea NOT shipped: the two exact-fallback shapes (B=32,S=128 and B=1,S=8192) still materialize the full (B,32,S,S) fp32 scores tensor in HBM, which is far slower than SDPA and will score near-zero on those two workloads. If this round passes correctness but the average is still capped well below ~0.55, replace `_exact_attention` with a small two-pass Triton kernel (block-wise max/sum in fp32, then a second pass that rounds P to bf16 per-tile before the bf16 P@V accumulate) so those two shapes get flash-level bandwidth without losing the bf16-rounding bit-match to the reference —

## 5. from `c65ae90c` — Cached concatenated QKV cuBLAS GEMM + fused per-head RMSNorm/RoPE (reference-exact rounding) + PyTorch SDPA (flash, is_c
Higher-ceiling idea NOT shipped: the 8 exact-path shapes (esp. (1,8192), which materializes a ~4GB fp32/bf16 (32,S,S) scores tensor across several HBM passes) still pay full O(B*S^2) memory traffic instead of flash-level bandwidth. If this round passes correctness but the average score is capped well below ~0.5 and profiling shows (1,8192) or the other exact shapes dominating total time, replace `_exact_attention` with a two-pass Triton kernel per exact shape: pass 1 computes block-wise row max/sum in fp32 (online, no full materialization), pass 2 recomputes each tile's P, rounds it to bf16 (m

