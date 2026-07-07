# Family: rope

One distilled line per finished problem.

- task 231 (rope_231): best=0.5466 tier=0 via "Fused Triton kernel: one block per row with parallel reduction for mean-of-squares, rsqrt, and fused scaling by inv_rms*" [budget:evals]
- task 211 (rope_211): best=0.6784 tier=0 via "Triton fused kernel with two-pass approach: first computes sum-of-squares reduction for all batch elements, then applies" [budget:evals]
