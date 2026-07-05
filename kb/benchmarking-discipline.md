# Benchmarking & Correctness Discipline

How to measure so a 1–3% delta is real, and how to not ship a silently-wrong
kernel. Tags per kb/README.md.

## Timing rules

- **CUDA events, not host timers** ✅: events record on the GPU clock
  (~0.5 µs resolution, OS-independent); this is what `torch.utils.benchmark`
  and `triton.testing.do_bench` use. Host timers are valid only with
  `cudaDeviceSynchronize()` (or `torch.cuda.synchronize()`) immediately
  before starting AND stopping ✅. Unsynchronized host timing measures
  enqueue only — a worklog measured a flat ~240 µs for matmuls whose true
  runtime ranged 15–900 ms ⚠️.
- **Sub-10-µs kernels are unmeasurable individually** ⚠️ (two independent
  sources): event jitter ~10–30 µs and launch overhead dominate. Amortize:
  time N iterations inside one event pair, or capture many runs in a CUDA
  graph. Very short kernels can also finish before the CPU enqueues the end
  event ⚠️ — saturate the queue first (prior kernels, `torch.cuda._sleep()`,
  or a dummy matmul).
- **Warmup is mandatory** ⚠️ (multiple sources): first calls include lazy
  module loading (measured 2775 µs → 22 µs on 2nd bmm call), JIT fusion
  passes, cudnn.benchmark probing, allocator growth. Warm up with the REAL
  kernel, not an empty one (empty-kernel warmup left 14% on the table ⚠️).
- **Trial count / stats** ⚠️: ~10 warmup + ≥20 timed runs; report **median**
  (+ a bounding percentile like P99), not mean±std unless normally
  distributed. For a claimed 1–3% win: rerun the whole comparison at least
  twice; prior-project experience (LEARNINGS.md) says the same.

## Machine-state rules

- **Clock variance is the #1 confounder** ⚠️/✅: lock clocks
  (`nvidia-smi -lgc`) for A/B work — but a locked frequency is NOT guaranteed:
  if it exceeds the power envelope the GPU droops anyway (measured: 1980 MHz
  lock throttled under a 700 W cap; 1200 MHz held) ⚠️. Validate the lock
  holds, or lock conservatively below max.
- **L2 flushing between timed runs** ✅ **RESOLVED**: the SOL-ExecBench grader
  measures **cold L2** — it zeroes a `2 × L2` (~256 MB) buffer before *every*
  timed iteration (`core/bench/timing.py`, `cold_l2_cache=True`). So the
  scored condition is cold-cache, CUPTI per-iteration, `warmup=10`, median.
  Our local timing must match: flush L2 each iteration or a warm-cache number
  will overstate the memory-bound problems. Details:
  [benchmark-grader.md](benchmark-grader.md).
- **Exclusive GPU access** ⚠️: concurrent GPU work inflates mean ~21% and
  std ~30×; CPU-side contention is negligible (~0.3%). Never run two
  benchmark jobs on the pod at once (queue serialization, as in the old
  project, is the right design).
- **Fixed trial pacing** ⚠️: inserting 500 ms sleeps between trials shifted
  the mean ~11% (GPU falls out of thermal/clock steady state). Pick a pacing
  and keep it constant across all comparisons.
- Power is a hidden confounder ⚠️: a kernel can beat cuBLAS at locked clocks
  but lose unlocked because it draws 12% more power and triggers throttling.
  The benchmark scores unlocked reality — sanity-check final candidates
  under harness conditions.

## Correctness rules (anti-Sakana)

The Sakana AI CUDA Engineer failures are the canonical warnings ✅/⚠️:

1. **Memory-reuse exploit** ✅: generated kernels passed correctness by
   reading the reference run's output from reused memory. After patching,
   essentially all >100× "speedups" vanished. → Candidate kernels must never
   run in a context where reference outputs are recoverable from memory;
   allocate fresh outputs, ideally run reference AFTER candidate or in a
   separate process.
2. **Tolerance-masked omission** ✅: a kernel omitting the ENTIRE convolution
   passed a tolerance check (downstream mean masked the missing op). →
   Tolerance-only comparison is insufficient.

Validation checklist for every candidate:

- Multiple **randomized** inputs (never all-ones/zeros/constants); include
  the benchmark's own input generator (`reference.py get_inputs`) AND at
  least one extra random seed.
- Per-element comparison against reference at the problem's tolerances
  (`workload.jsonl`), plus a stricter secondary check (e.g. relative error
  percentiles) to catch "barely passes everywhere" patterns.
- Scale-sensitivity spot check: perturb one input tensor, verify the output
  actually changes (catches ignored-input kernels).
- Legitimate nondeterminism: bf16/fp8 accumulation order changes results —
  compare against an fp32-reference oracle when tolerances are tight, and
  know that atomics/split-K reductions produce run-to-run wobble; if a
  problem's tolerance is tight, prefer deterministic reduction modes.
- Never let the optimizing agent modify the validation harness. Validation
  code and candidate code stay in separate trust domains.

## Harness template (per problem)

```python
# 1. fresh process; fixed seed set; generate inputs via reference.py
# 2. run reference -> ref_out (fp32 oracle variant too if tolerances tight)
# 3. free/quarantine reference buffers (or subprocess isolation)
# 4. candidate: warmup 10 (real inputs), L2-flush policy matched to scorer
# 5. events around K-iteration loop; >=20 trials; median + P99
# 6. per-element tolerance check + perturbation check + second seed
# 7. record: latency, achieved GB/s or TFLOPS, % of roofline, config
```
