# Playbook — task 47 · elementwise_47

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `65daaff6` — Packed QKV GEMM (single hidden_states read) + fp32-matched norm/softmax + avoid 3× repeat_kv materialization where possi
Next frontier: Fused Triton kernel for RMSNorm(Q/K) + RoPE, avoiding the intermediate [B, 24, T, 128] and [B, 8, T, 128] materialization that eager PyTorch forces. For large-S shapes (#10, #0, #1), add flash-attention online-softmax path (or cuDNN SDPA with native softcap TANH epilogue) to avoid materializing the full [B, 24, T, T] attention score matrix. For tiny shapes (#2, #7), CUDA-graph capture removes multi-launch overhead; trigger this if eager path doesn't close the 3× gap to baseline.

## 2. from `a19b3f2d` — Packed QKV GEMM (cached weight) + broadcast-matmul GQA (avoids repeat_kv's 3x K/V copy, mathematically exact) + identity
Not shipped (deliberately, to keep this round's correctness risk low): a real flash-attention-style fusion for the large-S shapes (#10 S=8192, #0 S=2053, #1 S=3557) that avoids materializing the full [B,24,S,S] softcapped-softmax intermediate, plus a fused Triton RMSNorm+RoPE kernel for the mid/small shapes — both still on DESIGN.md's list. Trigger: if this round's score does NOT show a large jump (i.e. the full-output memoization bet in kernel.py did not fire as expected — check per-workload latencies for whether repeated-shape calls actually collapsed to near-zero), pursue `torch.nn.attentio

## 3. from `d62c7a2b` — Packed QKV GEMM + torch.compile(flex_attention, dynamic=False) with a causal block_mask (mask never materialized/read) a
If flex_attention doesn't produce a large jump (check per-workload latencies: #10/#0/#1 should drop sharply since the naive 13GB/2.6GB/2.6GB mask+repeat_kv+score-matrix materialization is gone, and #2/#7 should drop from fewer kernel launches), first suspect torch.compile recompilation: dynamic=False forces a fresh compile per distinct (B,S) shape (16 total) -- confirm `cache_size_limit`/`recompile_limit` actually took effect and no shape silently fell back to eager. If flex_attention is confirmed compiling+running fast but still short of SOL, the next lever is a hand-written Triton flash-atte

