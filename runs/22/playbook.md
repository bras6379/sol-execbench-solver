# Playbook — task 22 · softmax_22

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `ed1c1143` — Move the 4 fp32 GEMMs onto tensor cores (fp16/cuBLAS for large S, TF32-no-cast for small S) and fuse GELU-backward into
HIGHER-CEILING RESERVE (not shipped this round):

1) Fuse GELU-backward into GEMM-B's epilogue with a single Triton matmul
   (grad_output @ fc2_weight -> gelu'(fc1_output) applied in-register -> write
   grad_fc1_output fp16 directly). This kills the grad_gelu_output HBM
   round-trip (~29-58 MB written + reread at S=4096). TRIGGER: if the large-S
   shapes (S>=997) still trail SOL by >15% — the residual is that extra pass
   plus fc1_output's 58 MB read; a fused epilogue removes the intermediate.

2) For small S (<=256, launch-bound): wrap the whole run in a per-shape CUDA
   graph (copy fre

## 2. from `396a2b8a` — Keep fused Triton GELU bwd + torch.compile(reduce-overhead) per-shape cache for small S (64-256): 7 eager launches cost
Fuse GELU-backward into GEMM-B's epilogue with a single Triton matmul (grad_output @ fc2_weight -> gelu'(fc1_output) applied in-register -> write grad_fc1_output fp16 directly). This kills the grad_gelu_output HBM round-trip (~29-58 MB written + reread at S=4096). Trigger: if the large-S shapes (S>=997) still trail SOL by >15% — the residual is that extra pass plus fc1_output's 58 MB read; a fused epilogue removes the intermediate.

