# Playbook — task 48 · reduction_48

Higher-ceiling ideas that accepted kernels flagged but did NOT ship.
Banked when each author entered the frontier; the next agent reads these.

## 1. from `661548b3` — Same two cuBLAS GEMMs as the reference (bit-identical), but replace the reference's ~9 separate fp32 elementwise ops for
Investigation before shipping (worth reading before writing a 4th custom-Triton
epilogue): I dug into `runs/48/candidates/*.json` (per-workload `error` field)
and `runs/48/work/review-*/review.md` for the three prior RUNTIME_ERROR
attempts (`fc740b0d0377` full fused dual-tl.dot, `d2b6ffe578fe` cuBLASx2 +
Triton elementwise epilogue with a manual `tl.exp2`-based tanh, `358350ab439e`
cuBLASx2 + Triton elementwise epilogue with `tl.math.tanh`/`tl.math.pow`) —
all 3 crashed identically on all 16 workloads despite passing independent
code review (hand-traced, no bug found), and despite using 3 stru

