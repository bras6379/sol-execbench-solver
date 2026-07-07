# Playbook — task 2 · softmax_2

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `c3dd0c43` — Per-regime body dispatch (element-wise max of the two tied FP16 frontier kernels): FP16 channels-last cuDNN tensor-core
Higher-ceiling reserve play NOT shipped: replace the `mode="reduce-overhead"` small-shape path with a MANUAL `torch.cuda.CUDAGraph` capture using a private/dedicated memory pool, so the small-shape graph pool can never pollute the plain-compiled B==1-large path. Trigger: if this regime-split scores >= ~0.58 overall but the B==1 large workloads (wl10 1x1024x1024, wl11 1x293x293, wl15 1x768x768) are STILL weak (< ~0.49) — that means torch.compile's cudagraph-tree pool is bleeding into the plain-compile path in-process, and a manual graph with its own pool + a warmup replay is the fix. Beyond tha

## 2. from `5f5e10a4` — NCHW-layout body for B==1 (cuDNN emits NCHW for tall-skinny B==1 implicit GEMMs, so matching NCHW weights avoids the int
Higher-ceiling reserve: fuse the entire block (conv1+GN1+SiLU1+conv2+GN2+SiLU2+add) into a single cuDNN operation-graph node via the cuDNN frontend Python API (cudnn-frontend), which on SM100 can JIT-compile the full chain into ONE fused kernel — eliminating ALL intermediate HBM roundtrips between convs and tails, not just the elementwise tail fusion torch.compile provides. Trigger: if this revision leaves any B>=2 workload raw-score < ~0.65 (meaning cuDNN implicit-GEMM alone is utilization-limited and the GN+SiLU+add memory traffic is the binding constraint), the cuDNN graph API is the next l

## 3. from `8c84d614` — Split the small-shape CUDA-graph path by batch: B==1 uses a pure-NCHW reduce-overhead graph to avoid the NCHW->NHWC tran
Higher-ceiling reserve not shipped: fuse the whole block (conv1→GN1→SiLU1→conv2→GN2→SiLU2→add) into a single cuDNN frontend operation graph so the backend can elide all intermediate HBM round-trips between the two convs and the normalizations, not just the tail fusion torch.compile provides. Trigger: if any B>=2 large workload still scores raw < ~0.65 after this dispatch tuning, the remaining gap is likely inter-op memory traffic and the cuDNN graph API is the next move; wrap it with a torch.compile fallback in case cudnn-frontend is unavailable.

## 4. from `18fc2385` — Exhaustive cuDNN algorithm search (benchmark_limit=0) + max-autotune for large NHWC shapes to autotune Inductor's fused
Fuse the whole block (conv1→GN1→SiLU→conv2→GN2→SiLU→add) into a single cuDNN frontend operation graph via the cudnn-frontend Python API. On SM100 the backend can JIT-compile the full chain into one fused kernel, eliminating all intermediate HBM round-trips. Even though GroupNorm is a reduction barrier, the graph API can manage the entire execution as a single node with fused pointwise segments and cooperative buffer reuse. Trigger: if this revision leaves any B>=2 large workload raw-score < ~0.65, the remaining gap is inter-op memory traffic — deploy the cuDNN frontend graph with a torch.compi

## 5. from `42623ceb` — Reorder dispatch checks so b==1 is evaluated before p<=32768 for small product, eliminating redundant condition evaluati
Fuse the entire residual block (conv1→GN1→SiLU→conv2→GN2→SiLU→add) into a single cuDNN frontend operation graph via cudnn-frontend Python API, eliminating all intermediate HBM round-trips between the two convs and normalizations. Trigger: if this revision leaves any B>=2 large workload raw-score < ~0.65 (meaning cuDNN implicit-GEMM alone is utilization-limited and the GN+SiLU+add memory traffic is the binding constraint), the cuDNN graph API is the next move.

## 6. from `aa162b65` — Keep the 0.556 frontier for small shapes and B==1 large, and add a new large-B>=2 path using custom Triton kernels that
This round ships a real attempt at the untried Triton-fused-NHWC-tail lever for large B>=2 workloads: two single-CTA-per-(sample,group) Triton kernels collapse GN+SiLU (first tail) and GN+SiLU+residual-add+layout-to-NCHW (second tail), with fallback to the proven 0.556 torch.compile paths if Triton fails.

Higher-ceiling idea NOT shipped: fuse the whole residual block end-to-end with the cuDNN frontend operation graph (or a persistent single-kernel Triton/Welford reduction that keeps conv1 output resident in L2 between stats and apply). Trigger to try it: if this revision still lands at or bel

