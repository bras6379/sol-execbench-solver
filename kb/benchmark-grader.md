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
harness should keep the same posture.

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
