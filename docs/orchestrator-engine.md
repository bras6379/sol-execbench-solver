# Orchestrator Engine Design

Concrete engine on top of the [orchestration loop](orchestration.md). Goal:
**select problems → optimize each until it wins the bench → accumulate and
transfer knowledge across problems.** Most of this is buildable now with the
GPU abstracted behind an interface (stub today, real transport later).

## Component map

```
                 ┌──────────────────────────────────────────────┐
   selected ids  │  Orchestrator (fleet)                         │
   ────────────▶ │   budget allocation across problems           │
                 │   ┌───────────────┐  ┌───────────────┐  ...   │
                 │   │ ProblemSolver │  │ ProblemSolver │        │  ← run in parallel
                 │   │  (GEPA loop)  │  │  (GEPA loop)  │        │
                 │   └──────┬────────┘  └──────┬────────┘        │
                 └──────────┼──────────────────┼────────────────┘
                            │ evaluate()        │ evaluate()
                            ▼                   ▼
                 ┌──────────────────────────────────────────────┐
                 │  Executor  (SINGLE-FLIGHT GPU LOCK, conc=1)   │  ← the only serialized resource
                 │   StubExecutor (now)  |  GpuQueueExecutor(later)│
                 └──────────────────────────────────────────────┘
        reads/writes ▲                              ▲ writes
                 ┌───────────────────────┐   ┌──────────────────┐
                 │ Knowledge (2 layers)  │   │ Run state        │
                 │  static kb/ (priors)  │   │ runs/<id>/journal│
                 │  dynamic knowledge/   │   │ frontier, best   │
                 └───────────────────────┘   └──────────────────┘
```

## 1. Executor — the GPU lock (interface, swappable)

The one serialized resource. Everything else fans out.

```python
class Executor(Protocol):
    def evaluate(self, solution: dict, task_id: int, *, profile: bool = False) -> EvalResult: ...

@dataclass
class EvalResult:
    correct: bool
    sol_score: float | None                 # via solver/scoring.py
    per_workload: list[WorkloadResult]       # matched_ratio, latency_ms, sol_ms, error
    asi: dict                                # actionable side info: errors, profile, notes
    raw: dict                                # full harness payload
```

- `StubExecutor` (now): validates via `solver.check`, returns mock latencies
  (e.g. baseline × a factor drawn from the candidate's declared technique) so
  the whole loop + KB + transfer can be built and tested without a GPU.
- `GpuQueueExecutor` (later): enqueues on a single-flight queue to the GPU box
  running the real SOL-ExecBench harness; blocks for the result. Concurrency
  pinned to **1**. This is the future "GPU-run transport" phase.

All ProblemSolvers share **one** Executor instance → the lock is structural.

## 2. ProblemSolver — per-question GEPA loop

Optimizes one problem to a win (or budget). Holds a **Pareto frontier of
candidate Solutions over the problem's ~16 workload shapes** (a kernel best on
big shapes and one best on small shapes can both survive → later merged into a
per-shape dispatch submission).

```
seed  = scaffold(id) (correct PyTorch DPS baseline)  ∪  transferred templates
loop until win or budget or plateau(K):
    parent   = select_from_frontier()          # Pareto-aware
    context  = retrieve_knowledge(id)          # static kb slices + family learnings + sibling bests
    candidate= mutate_agent(parent, reflection, context)   # LLM proposes a new Solution
    if not check(candidate): record(reject); continue      # cheap static gate, no GPU
    result   = executor.evaluate(candidate)    # THE GPU LOCK
    frontier.accept_if_pareto(candidate, result)
    reflection = reflect_agent(candidate, result)          # "why": the text gradient
    journal.append(candidate, result, reflection)
```

Agents (mutate/reflect/plan) are LLM calls grounded in `kb/optimization-recipe.md`
and the family KB; they run concurrently with every other problem's agents.

## 3. Orchestrator — the fleet

Runs N ProblemSolvers concurrently over the selected ids; allocates the shared
GPU budget (by headroom to SOL, round-robin, or priority). Because agents are
parallel and only `evaluate()` serializes, the GPU is never idle waiting on
thinking — while one candidate is timed, every other problem's agents keep
planning/reflecting.

## 4. Knowledge base — two layers

- **Static** (`kb/`, already built): B200/kernel priors + the grader spec.
  Read-only.
- **Dynamic** (`knowledge/`, written during runs) — three tiers:
  - **Per-problem journal** (`runs/<id>/journal.jsonl`): every candidate, its
    technique tag, diff, score, trace, verdict. The raw search history.
  - **Per-family learnings** (`knowledge/families/<family>.md`): curated
    distillation — what wins / what fails for `rmsnorm`, `attention-bwd`,
    `moe-grouped-gemm`, `fp8-projection`, … Keyed by the taxonomy in
    `kb/benchmark-problems.md`. **This is the transferable unit.**
  - **Global learnings** (`knowledge/global.md`): cross-cutting facts
    ("TF32 clears tolerance on the fp32 attention problems"; "cold-L2 makes
    small rmsnorm launch-bound"). Curated, concise.

A **curator step** (LLM) runs when a problem finishes: it reads the journal
and updates the family + global learnings (merge/rewrite, not append). Bounded
size, like `kb/` itself.

## 5. Knowledge transfer (the crux)

Sample-efficiency comes from not re-deriving what a sibling already solved.
Mechanisms, cheapest first:

1. **Sibling templating.** Many problems are shape-variants (230–235 are
   rmsnorm at H=128…7168; 213–220 are GEMMs at different N/K). A winning
   Solution for one is a near-direct template for the next — port constants,
   don't re-search. The engine detects "same family + same op, different
   axes" and **seeds the frontier from the sibling's best**, so its first GPU
   run is often already near-winning.
2. **Family learnings retrieval.** At plan/mutate time, inject the family's
   distilled learnings + the current family bests into the agent context.
3. **Static KB slices.** The design classification (from the `design-kernel`
   skill) selects which `kb/` files are relevant; those ride along.
4. **Cross-problem Pareto/merge.** A technique that won on one problem seeds
   mutations on related ones (GEPA's system-aware merge, across problems).
5. **Curation feedback.** Each finished problem improves the family learnings,
   so problem N+1 in a family starts strictly smarter than N. Order the fleet
   to solve one exemplar per family first, then its cheap siblings.

Net effect: the first problem in a family is expensive (real search); the rest
are cheap (template + tune). That is where the GPU-budget savings live.

## 6. "Win the bench" / termination (per problem)

Stop a ProblemSolver when any holds:
- **Target reached** — `sol_score ≥ TARGET`. Options for TARGET (decision
  needed): beat the current leaderboard rank-1 (requires fetching the public
  leaderboard per problem), or a fixed threshold (e.g. 0.90), or "best-effort,
  maximize within budget."
- **Budget exhausted** — GPU-evaluation cap for this problem.
- **Plateau** — K consecutive accepted-or-not iterations with no frontier
  improvement.

Best correct candidate (or merged per-shape dispatch) becomes
`runs/<id>/best_solution.json`.

## 7. State & CLI

- `runs/<id>/`: journal.jsonl, frontier/, best_solution.json, status.json —
  fully resumable.
- `knowledge/`: families/*.md, global.md.
- CLI: `solver solve <ids|--all> [--budget N] [--target ...] [--executor stub|gpu]`
  and `solver status`.

## 8. Build order (laptop-first)

| Phase | Buildable now? | Piece |
|---|---|---|
| A | ✅ | Executor interface + StubExecutor; EvalResult model |
| B | ✅ | ProblemSolver loop + Pareto frontier + journal/state |
| C | ✅ | Knowledge store (dynamic KB) + retrieval + curator |
| D | ✅ | Orchestrator fleet + budget + CLI (`solve`, `status`) |
| E | ✅ | Transfer: sibling detection + templating |
| F | ⛔ later | GpuQueueExecutor (real GPU transport + harness) |
| G | ⛔ later | Profiling in the eval path (Nsight → ASI) |

A–E give a fully testable engine against the stub; F swaps in the GPU with no
change to the loop, KB, or transfer logic.

## Decisions (defaults chosen 2026-07-05, revisable)

Picked to unblock the foundation; revisit before the agent layer is built.

1. **Agent substrate** → **Anthropic API direct**, behind a swappable `Agent`
   interface (StubAgent for deterministic tests). Standalone engine; not
   coupled to the Claude Code runtime.
2. **Search driver** → **custom lightweight loop**. Small, full control over
   the GPU lock / per-shape Pareto / cross-problem transfer; no library to bend.
3. **"Win" definition** → **beat leaderboard rank-1** (fetch the public
   leaderboard per problem), with **budget** and **plateau(K)** as fallbacks.
   Configurable to a fixed threshold or best-effort.

Foundation (Phase A: Executor interface + StubExecutor + result model) is
decision-independent and is built first. The contentious layer (agents /
search loop, phases B–D) waits for confirmation.
