# Op: expert_output_weighted_index_add_accumulation

One distilled line per finished problem.

- task 8 (reduction_8): best=0.653 tier=0 via "seed" [budget:iterations]
- task 8 (reduction_8): best=0.8676 tier=0 via "Fused Triton bf16 atomic scatter-add (H=3072): seed output=fhs.clone(), grid=N=8T one program per expert row, hoist all " [budget:time]
- task 8 (reduction_8): best=0.8683 tier=0 via "Proven fused Triton bf16 atomic scatter-add (clone-seed output=fhs, grid=N one program per expert row, BLOCK=1024 x3 til" [budget:time]
- task 8 (reduction_8): best=0.8683 tier=0 via "Proven fused Triton bf16 atomic scatter-add (clone-seed output=fhs, grid=N one program per expert row, BLOCK=1024 x3 til" [ceiling_consensus]
