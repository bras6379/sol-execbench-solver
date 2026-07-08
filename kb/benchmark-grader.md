# The SOL-ExecBench Grader (Ground Truth)

Exactly how a candidate is scored, read from the official harness
(`github.com/NVIDIA/SOL-ExecBench`, `src/sol_execbench/`, Apache-2.0). This is
the un-hackable reference we must satisfy and should **reuse, not
reimplement**, when we build validation. Tags per kb/README.md — everything
here is ✅ (read directly from source 2026-07-04).

## Score formula (`sol_score.py`)

```
S(T_k) = 1 / (1 + (T_k − T_SOL) / (T_b − T_SOL))
```

- `T_k` = candidate latency, `T_b` = optimized-PyTorch **baseline**, `T_SOL`
  = Speed-of-Light bound (all ms, per workload).
- **S = 0.5 when T_k = T_b** (matching the baseline scores 0.5), **S = 1.0
  when T_k = T_SOL**, S → 0 as the candidate gets slower than baseline.
- If `T_b ≤ T_SOL` (baseline already at SOL): S = 1 if `T_k ≤ T_SOL` else 0.
- Implication: **beating `T_b` is the threshold for S > 0.5**; the remaining
  points come from closing the gap to `T_SOL`. Both numbers are in each
  problem's `metadata.json` (`sol.per_workload[i].baseline_latency_ms` and
  `sol_ms`) after `solver fetch`.

## Aggregation (how 16 workloads → one number → the leaderboard)

- **Benchmark level (paper, ✅):** arithmetic mean of per-problem scores,
  correctness-gated — `S̄ = (1/N) Σ C_j·S_j` (C_j ∈ {0,1}).
- **Per-problem latency (empirical, ✅):** the leaderboard's per-kernel
  latency is the **geometric mean over the ~16 per-workload latencies** —
  verified on kernel 1: the "SOL Bound" row's latency (0.08172 ms) equals
  the geomean of the 16 `sol_ms` values exactly.
- **Per-problem sol_score (⚠️ open):** the paper doesn't specify whether it's
  the mean of per-workload S values or S applied to geomean latencies; on
  kernel 1's rank-1 row, S-of-geomeans gives 0.7354 vs the published 0.7237
  (~1.6% off), suggesting **mean-of-per-workload-S** — our engine uses that
  (already what `solver/scoring.score_from_metadata` computes) and treats
  the exact website formula as a calibration question for the real harness.
- Dashboard conventions: score ↑ toward 1.0 (0.5 = baseline); latency-side
  view = geomean(T_k)/geomean(T_SOL) ↓ toward 1.0.

## Timing (`core/bench/timing.py`)

- **CUPTI per-iteration GPU timing**; `warmup = 10` (untimed) then `rep`
  measured iterations; returns mean or **median**.
- **Cold-L2 cache is the scored default** (`cold_l2_cache=True`): the harness
  zeroes a `2 × L2` (~256 MB) buffer to flush L2 **before every iteration**.
  So kernels are measured cold-cache — a warm-cache local number will
  overstate performance on the memory-bound problems. This resolves the open
  question in [benchmarking-discipline.md](benchmarking-discipline.md): the
  scored condition is **cold L2, CUPTI, 10 warmup, median**.
- Clock locking handled by `core/bench/clock_lock.py`.

**Why memory-bound kernels cap well below 1.0 — cold-L2 vs. an L2-BW-based
`T_SOL`.** For a bandwidth-bound problem, `metadata.json`'s stated `sol_ms` can
be derived from L2 bandwidth (~21 TB/s on B200) even though the scored
condition is cold-L2, i.e. every real measured iteration can only achieve HBM
bandwidth (~7.7 TB/s) — a candidate physically cannot reach `T_SOL` no matter
how optimal, because `T_SOL` assumes a cache tier the cold-L2 protocol
deliberately denies it. Measured on a memory-bound depthwise-conv problem: the
perfect-achieved-HBM-bandwidth ceiling on the large (bandwidth-bound) shapes is
S≈0.76, not 1.0; whole-problem ceiling (mixing in the smaller, launch-bound
shapes that DO have headroom) lands around ≈0.87. **Don't chase S→1.0 as a
correctness-of-approach signal on a memory-bound op** — check whether the
per-workload achieved bandwidth is already near HBM peak before assuming a
kernel is under-optimized; if it is, the remaining gap is the grading
methodology, not something more fusion/tuning can close. On the same problem,
CUPTI's device-only timing also means CPU launch overhead is mostly uncounted,
so CUDA-graph capture buys little on the large shapes — the lever that
actually moves the number is achieved HBM bandwidth % (elements/thread,
block-size tuning), not fancier fusion or graph capture.

### Eval-loop module reuse — cache derived values, NEVER skip the kernel launch

The eval subprocess imports your kernel module **once**, then per workload runs
`warmup=10` (untimed) followed by the timed iterations — all within that one
process/module-load, all with the **identical** input scalars for that
workload. Module-level state (a dict, a cache) genuinely persists across every
one of those calls; that part is real and is fine to use.

**The trap:** it is tempting to key a cache on the call signature (e.g.
`(seq_len, scaling)`) and, once cached, return the stored result **without
launching any kernel at all** on the timed iterations — since the inputs are
identical every call, the output is too, so this looks like a free win.
**Confirmed empirically dead — twice, on two different problems:** any
solution that skips real GPU kernel execution on a repeat call (returning a
previously-computed tensor with zero kernel launch, or skipping a
`cuda_graph.replay()`) scores **RUNTIME_ERROR on every workload** ("Timing
failed: No kernel activities recorded for iteration N" — the harness's
profiler requires real per-iteration GPU stream activity; an iteration with
none is a hard failure, not a free 0 ms). This is **not** one of the
documented reward-hack detectors below (no monkey-patch, no thread injection,
no FakeTensor) — it's a separate timing-instrumentation invariant.

**What's actually safe:** caching *derived Python objects that don't change
the GPU work performed* — a computed grid/tile config, a `torch.as_strided`
view object, a pre-built CUDA graph you still `.replay()` every call. The
boundary is exactly this: skipping the **launch/replay itself** is the
failure; caching *inputs to* that launch is not. If you're tempted by a
module-level cache keyed on scalar inputs, make sure the cached path still
issues a real kernel launch (or graph replay) every single call.

## Correctness (`core/bench/correctness.py`)

Compared in **fp32** (both candidate and reference cast to float32). Order:

1. **Sanity (`check_tensor_sanity`)** — hard fails, independent of tolerance:
   - Any **inf or nan in either** tensor → fail. Even *matching* non-finite
     values fail, unless `allow_negative_inf` is set AND both are −inf at
     that position. (nan is reported as `has_nan`; inf-without-nan as
     `has_inf`.)
   - **All-zeros solution while the reference is non-zero → fail.** (Kills
     the "return zeros" degenerate kernel.)
2. **Per-element tolerance** — `|x − y| ≤ max_atol + max_rtol·|y|`
   (torch.allclose style). `matched_ratio` = fraction of elements passing.
3. **Pass iff `matched_ratio ≥ required_matched_ratio`** (default **0.99** —
   up to 1% of elements may exceed tolerance).
4. **`max_error_cap`** (optional) — if set, any single element's abs error
   above the cap fails *regardless* of matched_ratio (cuDNN-style outlier
   guard).

### Tolerance schema (`ToleranceSpec`, per workload)

| Field | Default | Meaning |
|---|---|---|
| `max_atol` | 1e-2 | absolute band |
| `max_rtol` | 1e-2 | relative band |
| `required_matched_ratio` | **0.99** | fraction of elements that must pass |
| `max_error_cap` | none | hard per-element abs-error ceiling |
| `allow_negative_inf` | false | tolerate matching −inf positions (e.g. causal masks) |

The 99% default is a real lever: a kernel with a few tolerance-exceeding
outliers still passes unless `max_error_cap` is set. Read the actual per-
workload values from `workload.jsonl`. Note: **18 of the 26 FlashInfer-Bench
problems carry no tolerance** — the rmsnorm/gemm/moe ones (ids 210–220,
229–235); the 8 paged/ragged GQA/MLA attention problems (221–228) do have
tolerance. Every L1/L2/Quant problem has tolerance on every workload.

### bf16 reference rounding — match it, don't out-precision it

For a bf16 op chain, the reference materializes **each intermediate as a bf16
tensor** before the next op reads it — not one fp32 computation cast to bf16 at
the end. A fused kernel that keeps an intermediate in fp32 through steps the
reference rounds to bf16 will *diverge* from the reference, not converge to it,
even though the fp32 path is "more accurate": each skipped bf16 rounding shifts
the value by ~1 ulp, and propagated through a downstream reduction (e.g. a
several-thousand-channel GEMM) that's enough to push a meaningful fraction of
elements outside the tolerance band. Measured on a short-conv fusion problem:
keeping the middle activations in fp32 → matched_ratio ~0.83 (FAIL against the
0.99 default); rounding each intermediate to bf16 at exactly the points the
reference does (elementwise mul → bf16, conv → bf16, elementwise mul → bf16,
fp32 accumulation only *within* each step) → matched_ratio 1.0000.

**How to apply:** find every point in the reference PyTorch implementation
where an op's output is a bf16 tensor (i.e. every op boundary, if the model is
bf16 end-to-end), and insert `.to(bf16).to(fp32)` at the equivalent point in
your fused kernel — accumulate in fp32 *inside* a step, round to bf16 *between*
steps. When validating on CPU before a GPU run, don't trust plain CPU bf16
matmul (`F.linear` on bf16 tensors accumulates in bf16 on CPU, which is noisy —
~3% spurious error); simulate the real GPU path explicitly with
`(a.float() @ b.float().T).to(bf16)` (fp32 accumulate, bf16 round) instead.

## Input generation (`core/bench/io.py`)

Inputs are **seeded** (`set_seed`, `core/bench/correctness.py`) so the
reference and the candidate receive **identical** tensors. Input specs
(`core/data/workload.py`): `random`, `scalar` (fixed literal, e.g.
`eps=1e-5`), `safetensors` (from file), `custom` (needs
`custom_inputs_entrypoint`; all-or-nothing per workload).

**`random` is NOT uniform noise.** `io.py` inspects each tensor's
name/shape/description and generates **role-appropriate** data via heuristics
(`_generate_heuristic_tensor`): weight matrices, norm weights, norm biases,
causal attention masks, binary masks, RoPE cos/sin tables, softmax outputs,
positive tensors, SSM decay, etc. Dtype-specific draws (`_rand_tensor`):
fp32/fp16/bf16 → `randn`; fp8 → `randn().clamp(±2)→fp8`; fp4 → packed
`e2m1fn_x2`; bool → `randint(0,2)`; ints → ranged `randint`.

Consequences for kernel design:
- A "softmax output" input really is a valid softmax; a cos/sin input is a
  valid rotary table — your kernel can rely on those properties but must be
  correct for **any** seeded draw of that shape (no overfitting to values).
- Local validation must reproduce these heuristics (or call the harness),
  not feed plain `randn` — plain noise can pass/fail differently than the
  scored inputs.

## Reward-hack defenses (`core/bench/reward_hack.py`)

The harness actively detects the Sakana-class exploits. Do not attempt any of
these — they raise `RewardHackDetected`:

- **Monkey-patch detection**: `torch.cuda.Event.elapsed_time` identity is
  captured at import (before user code); replacing it is caught.
- **Thread injection**: `threading.active_count()` before/after the call must
  not increase (no background-thread timing tricks).
- **Lazy outputs**: strict `type(t) is torch.Tensor` — FakeTensor / proxy /
  any subclass is rejected (no deferred/never-executed compute).
- **Eval integrity**: `id()` of critical eval-driver functions snapshotted
  pre-import and re-checked (no replacing the scorer).

This is the concrete implementation of the anti-Sakana guidance in
[llm-kernel-generation.md](llm-kernel-generation.md) — our own validation
harness should keep the same posture. A RELATED but separate trap — skipping
the kernel launch entirely via a module-level cache — isn't one of these
detectors but still hard-fails; see "Eval-loop module reuse" above.

### Correctness dominates marginal latency tricks — two concrete traps

- **Never return a CUDA graph's static output buffer directly.** The harness
  reuses inputs and re-invokes `run()` across many timed iterations; if your
  kernel captures a graph once and returns its static output tensor by
  reference, a *later* `graph.replay()` overwrites the memory the grader is
  still holding from an *earlier* call, corrupting a result the caller assumed
  was final. Clone the output before returning it (ideally one fused clone of
  a stacked/concatenated output buffer, not N separate clones, to keep the
  extra HBM traffic small).
- **Don't rely on `try/except` or `threading` inside the entry `run` function.**
  This repo's own pre-flight lint additionally flags `try/except` **inside the
  entry point itself** and any `threading` usage, independent of the harness's
  own reward-hack detectors above — keep any fallback/dispatch logic in a
  helper function `run` calls, not inline in `run`.

## Harness map (reuse these)

| File | Role |
|---|---|
| `core/data/workload.py` | Workload / ToleranceSpec / InputSpec schema |
| `core/bench/io.py` | input generation + role heuristics |
| `core/bench/correctness.py` | seeding + tolerance/matched-ratio scoring |
| `core/bench/timing.py` | CUPTI timing, cold-L2 flush, warmup/rep |
| `core/bench/clock_lock.py` | clock locking |
| `core/bench/reward_hack.py` | anti-cheat detection |
| `sol_score.py` | the S(T_k) score |
| `driver/problem_packager.py` | packaging |

**When we build the validation/run phase, drive the real harness** (it's
Apache-2.0 and pip-installable from the repo) rather than reimplementing
scoring — reimplementation risks diverging from the grader, which is itself a
reward-hacking failure mode.
