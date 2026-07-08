# Op: mla_fused_qkv_rope_split

One distilled line per finished problem.

- task 43 (layernorm_43): best=0.5927 tier=0 via "Keep the 3 dense GEMMs on cuBLAS (F.linear) but replace the reference's ~7 elementwise/copy ops (5-op RMSNorm, 2x .conti" [budget:time]
