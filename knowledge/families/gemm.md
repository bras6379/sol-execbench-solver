# Family: gemm

One distilled line per finished problem.

- task 234 (gemm_234): best=0.2879 tier=0 via "Fused Triton kernel combining cast, square-accumulate, rsqrt, and scaled output in single pass to achieve minimum-byte b" [budget:evals]
- task 4 (gemm_4): best=0.7969 tier=0 via "bf16 cuBLAS tensor-core GEMMs for wgrad and dgrad, returning dgrad as a transposed (B,32,S,64) view to avoid materializi" [budget:iterations]
