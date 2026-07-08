# Op: expert_token_scatter_with_weighted_forward_backward

One distilled line per finished problem.

- task 9 (conv_9): best=0.7582 tier=0 via "Targeted TF32 eager kernel: run G2 and the K=N wgrads G3/G5 in TF32 tensor-core GEMMs to eliminate the N>=215 rounding f" [budget:time]
- task 9 (conv_9): best=0.7582 tier=0 via "Targeted TF32 eager kernel: run G2 and the K=N wgrads G3/G5 in TF32 tensor-core GEMMs to eliminate the N>=215 rounding f" [ceiling_consensus]
- task 9 (conv_9): best=0.7606 tier=0 via "Targeted-TF32 cuBLAS path: G2/G3/G5 use fp32 operands with TF32 tensor cores for correctness, G1/G4/G6 stay bf16, and G6" [ceiling_consensus]
