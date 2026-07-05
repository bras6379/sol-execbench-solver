# Orchestration Design (forward-looking)

Design note for the multi-agent optimization loop. **Not built yet** — this
captures the intended shape so the pieces we build now (fetch, scaffold,
check, scoring, and the future GPU-run transport) fit it. The core idea:
drive kernel optimization as a **GEPA-style reflective evolutionary search**
whose evaluator is a **single-flight, GPU-locked** harness run.

## The loop (execution graph)

```
   plan/scaffold ─▶ [ GPU evaluate ] ─▶ reflect ─▶ mutate ─▶ [ GPU evaluate ] ─▶ …
                          │(locked)                                │(locked)
                          ▼                                        ▼
                    Pareto frontier over workload shapes  ◀────────┘
```

planning → execution → reflection → execution, matching the user's sketch.
Execution is the only node that touches the GPU, and it is serialized.

## Why GEPA fits (mapping)

GEPA = Genetic-Pareto reflective text evolution (ICLR 2026 oral,
arXiv 2507.19457; github.com/gepa-ai/gepa). Its `optimize_anything` API takes
a seed candidate + an evaluator and evolves the candidate via LLM reflection
on execution traces, keeping a Pareto frontier and minimizing evaluations.
The mapping to our solver is nearly one-to-one:

| GEPA concept | Our solver |
|---|---|
| Candidate (text artifact) | a kernel **Solution** (`solver/solution.py`) — the source |
| `evaluate(candidate)` → score | ship Solution to the GPU box, run the SOL-ExecBench harness, return correctness + per-workload latency; score via `solver/scoring.py` (SOL score) |
| Actionable Side Information (ASI) | harness output: matched_ratio / max error / inf-nan flags, per-workload latency vs SOL & baseline, **+ Nsight profile** if collected |
| `make_reflective_dataset()` | structure that harness+profile output (plus the design doc & relevant `kb/` slices) for the reflection LLM |
| Reflect (the "text gradient") | LLM diagnoses *why*: "matched_ratio 0.97 < 0.99 → precision too aggressive, accumulate in fp32"; "40% of SOL, memory-bound, uncoalesced → vectorize" |
| Mutate | generate an improved Solution from the diagnosis (grounded in `kb/optimization-recipe.md` ladders) |
| Pareto frontier over instances | **per-workload-shape** frontier — a problem has ~16 shapes; keep candidates each best on a subset |
| System-aware merge | merge shape-specialist candidates into one **per-shape dispatch** kernel (matches the design skill's per-shape specialization) |
| "Minimal rollouts" (100–500 vs 25k) | GPU runs are the scarce, serialized resource — this is the whole reason GEPA fits over RL |

## The GPU lock (the load-bearing constraint)

**All agents run in parallel; the only shared, serialized resource is GPU
execution — exactly one run at a time.** Every `evaluate()` is a real GPU run
and must serialize (concurrent GPU work inflates latency ~21% and std ~30×;
see `kb/benchmarking-discipline.md`). So:

- **Fan out everything except the GPU.** Planning, reflection, and mutation
  agents (all LLM/CPU) run fully concurrently across many problems and many
  candidates. Their candidate evaluations all funnel through **one
  single-flight GPU executor** — a shared lock/queue (à la the old repo's
  global GPU job daemon). Agents block only when they reach the GPU, and only
  behind other GPU work, never behind each other's thinking.
- The executor is the GPU-run transport (next build phase): submit Solution →
  run harness on the GPU box → return data. GEPA's `evaluate` is a thin client
  over that queue; it blocks until its turn completes.
- Correctness + timing happen **only** there; the laptop never runs a workload
  (see README "Execution model").

## Recommended approach

> **Superseded (2026-07-05):** the engine decision is a **custom lightweight
> loop** implementing GEPA's concepts (reflection, ε-Pareto frontier, merge)
> rather than the `gepa` library API — see
> [orchestrator-engine.md](orchestrator-engine.md) "Decisions". The mapping
> below remains valid as the conceptual reference.

Original note — use the `gepa` library's `optimize_anything` + a custom
`GEPAAdapter`:

- `evaluate` = enqueue on the GPU-locked executor, wait, return
  `(sol_score, ASI)`.
- `make_reflective_dataset` = harness result + Nsight section + the problem's
  design doc + the KB slices the design classified as relevant.
- `reflection_lm` = a strong model prompted with `kb/optimization-recipe.md`
  and the family-specific KB, told to propose one concrete change (honoring
  the keep/revert + time-box discipline already in the KB).
- seed candidate = `solver scaffold <id>` (the correct PyTorch DPS baseline).
- budget = a hard cap on GPU evaluations per problem (they are the cost).

This reuses the design skill (initial candidates + KB grounding), the scoring
(SOL score as the metric), and the pre-flight `check` (a candidate that fails
static check never reaches the GPU — saves an evaluation).

## Open questions (resolve when building)

- **Profiling in the eval path**: does each GPU run also return an Nsight
  section? Reflection quality depends on it, but it costs GPU time — maybe
  profile only on plateau, not every run.
- **Pareto → dispatch**: mechanics of merging shape-specialist kernels into
  one submission with per-shape dispatch (and whether the harness allows it).
- **Reward-hack posture in the loop**: the reflection LLM must be fenced from
  gaming the metric; the harness already detects timer/thread/lazy-output
  hacks, and `solver check` lints statically — keep both in the loop.
- **Cross-problem memory**: feed durable learnings (per-family what worked)
  back in, GEPA-merge-style, so problem N+1 starts smarter than N.
- **Concurrency knobs**: GEPA's rollout parallelism must be pinned to 1 at the
  GPU executor even if reflection/mutation run many-at-once.

## GRPO vs GEPA (why reflection, not RL)

Both improve an LLM system against a reward, but differently:

- **GRPO** (Group Relative Policy Optimization, DeepSeek) is RL: it updates
  the model **weights** via policy gradient. It samples a group of rollouts
  per input, scores each, and pushes weights toward the above-average ones
  (advantage = reward minus group mean — no critic network). It's
  sample-hungry (thousands–tens-of-thousands of rollouts) and the signal is a
  **scalar** reward.
- **GEPA** keeps the model **frozen** and evolves the **text candidate**
  (here: the kernel source) via **natural-language reflection** on execution
  traces — a rich, per-trace diagnosis instead of a scalar — plus Pareto
  selection and merge. It reaches better results with **100–500 evaluations
  vs 25k+** (up to 35× fewer).

For us, each evaluation is a scarce, serialized GPU run, and each run yields a
trace full of signal (matched_ratio, latency-vs-SOL, profiler output). GEPA
turns each of those into a targeted fix; GRPO would collapse it to one number
and need a training loop over thousands of GPU runs. So we **evolve the
kernel, not a model** — GEPA, not GRPO.

## References

- GEPA paper: arXiv 2507.19457 (ICLR 2026 oral) —
  "Reflective Prompt Evolution Can Outperform Reinforcement Learning".
- Library: github.com/gepa-ai/gepa (`optimize_anything`, `GEPAAdapter`).
- Our KB: `kb/optimization-recipe.md` (the per-problem loop the reflection
  step should follow), `kb/llm-kernel-generation.md` (reward-hacking failure
  modes + design rules), `kb/benchmark-grader.md` (what `evaluate` returns).
