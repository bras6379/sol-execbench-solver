# Playbook — task 2 · softmax_2

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `c3dd0c43` — Per-regime body dispatch (element-wise max of the two tied FP16 frontier kernels): FP16 channels-last cuDNN tensor-core
Higher-ceiling reserve play NOT shipped: replace the `mode="reduce-overhead"` small-shape path with a MANUAL `torch.cuda.CUDAGraph` capture using a private/dedicated memory pool, so the small-shape graph pool can never pollute the plain-compiled B==1-large path. Trigger: if this regime-split scores >= ~0.58 overall but the B==1 large workloads (wl10 1x1024x1024, wl11 1x293x293, wl15 1x768x768) are STILL weak (< ~0.49) — that means torch.compile's cudagraph-tree pool is bleeding into the plain-compile path in-process, and a manual graph with its own pool + a warmup replay is the fix. Beyond tha

## 2. from `5f5e10a4` — NCHW-layout body for B==1 (cuDNN emits NCHW for tall-skinny B==1 implicit GEMMs, so matching NCHW weights avoids the int
Higher-ceiling reserve: fuse the entire block (conv1+GN1+SiLU1+conv2+GN2+SiLU2+add) into a single cuDNN operation-graph node via the cuDNN frontend Python API (cudnn-frontend), which on SM100 can JIT-compile the full chain into ONE fused kernel — eliminating ALL intermediate HBM roundtrips between convs and tails, not just the elementwise tail fusion torch.compile provides. Trigger: if this revision leaves any B>=2 workload raw-score < ~0.65 (meaning cuDNN implicit-GEMM alone is utilization-limited and the GN+SiLU+add memory traffic is the binding constraint), the cuDNN graph API is the next l

