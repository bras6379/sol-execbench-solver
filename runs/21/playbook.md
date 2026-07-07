# Playbook — task 21 · rope_21

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `d75971ec` — Fuse the entire op under torch.compile reduce-overhead and replace the per-sequence Python loop with a single F.scaled_d
Reserve play: if measured SOL score stays below ~0.7 on the largest shapes (S ≥ 2304) or the O(S²) dense mask build/memory traffic becomes a visible bottleneck, switch to a cuDNN-frontend varlen SDPA that accepts cu_seqlens directly (or a Triton FlashAttention-style fused attention) to eliminate the dense mask entirely and avoid its HBM round-trip.

