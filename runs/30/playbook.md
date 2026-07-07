# Playbook — task 30 · rmsnorm_30

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `152b0e65` — Single cuBLASLt GEMM via torch.addmm(beta=1.0) with the residual pre-loaded as C, fusing the matmul and residual add int
If this only matches baseline, the next ceiling is a shape-specialized Triton fused GEMM+residual kernel with smaller M-tiles (BM=64) and split-K to fill SMs on the underfilled small-M shapes, replayed under CUDA graph for the harness's repeated iterations. If wave-quantization tails persist on the largest shapes, escalate to a CUTLASS 4 SM100 persistent/Stream-K kernel with a TMA-fused residual epilogue.

## 2. from `ab07051b` — Reload the best-scoring seed (plain torch.matmul + separate add, 0.363) unchanged in its GEMM structure — keeping beta=0
If this still only ties/loses to 0.363, the real ceiling is a CUTLASS 4 SM100 GEMM built with the collective builder's **Heuristic** tile scheduler (auto-picks DataParallel vs Stream-K per shape — don't hand-force split-K) plus a TMA-fused residual epilogue, dispatched only for M<=1024 (shapes #1,7,8,9,12,13,15,0's siblings); every hand-rolled Triton split-K attempt so far died on boundary masking for non-power-of-2 M (128/293/512/586/997/1571/2053/4106/7976), so re-deriving that path again is unlikely to pay off without CUTLASS's tested tile scheduler.

## 3. from `dab79370` — Reload the proven-best seed (plain torch.matmul + separate add_, beta=0) unchanged, plus two zero-risk config knobs — CU
If the workspace/reduced-precision-reduction knobs don't clear ~0.40+, the next real lever is a CUTLASS 4 SM100 GEMM built via the collective builder's **Heuristic** tile scheduler (auto-picks DataParallel vs Stream-K per shape — do not hand-force split-K) with a TMA-fused residual epilogue, dispatched only for the underfilled M<=1024 shapes (#1,7,8,9,12,13,15 and siblings); every hand-rolled Triton split-K attempt so far has died on boundary masking for non-power-of-2 M (128/293/512/586/997/1571/2053) or scored below the plain baseline even when correct, so re-deriving that path again is not

