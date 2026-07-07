# Playbook — task 15 · attention_15

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `2ba9d07b` — Cached concatenated QKV projection with cuBLAS, PyTorch SDPA native causal GQA to avoid materializing attention weights,
If this only ties the PyTorch SDPA baseline, replace it with a CuTe-DSL/FA4-style fused forward kernel that computes QKV tiles, applies per-head RMSNorm+RoPE in the producer, and runs online causal GQA softmax without writing Q/K/V or SxS scores to HBM. Trigger that path when profiling shows SDPA plus Q/K/V HBM traffic dominates after the projection GEMMs.

