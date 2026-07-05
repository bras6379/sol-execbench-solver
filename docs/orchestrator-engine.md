# Orchestrator Engine Design

Concrete engine on top of the [orchestration loop](orchestration.md). Goal:
**select problems → optimize each to the best kernel we can find → accumulate
and transfer knowledge across problems.** Most of this is buildable now with
the GPU abstracted behind an interface (stub today, real transport later).

**Scope:** the engine *finds* the best solution per problem. It does **not**
submit to the SOL-ExecBench website/leaderboard — no submission code. The
deliverable per problem is `best_solution.json` + its measured `sol_score`.
The public leaderboard is used, if at all, only as a read-only reference to
gauge how competitive our best is.

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

## Nodes: the unit of composition

Every step is a **Node**: a typed transform `run(ctx, inputs) -> outputs`
whose **role** (its input/output contract) is separate from its
**implementation** (how the work is done). The per-problem loop is a *cyclic
graph* of nodes; the Orchestrator runs many such graphs in parallel. Adding a
capability later — submission, a new profiler, a different planner — means
**add a node (or swap an implementation) and wire an edge**; no existing node
changes. This is why "add submission later" is safe by construction.

### Edge types (the typed artifacts that flow)

Nodes share a `RunContext` (problem, Pareto frontier, journal, knowledge
handles, budget) and pass typed artifacts:

- `Candidate` — a Solution (`solver/solution.py`)
- `CheckReport` (`solver/check.py`)
- `EvalResult` — per-workload score + ASI (`solver/engine`)
- `Reflection` — diagnosis + next-step hint
- `KnowledgeContext` — retrieved kb slices + family learnings + sibling bests
- `TerminationDecision`

### Node catalog (role → implementations)

| Node (role) | inputs → outputs | implementations |
|---|---|---|
| Select | frontier → parent(s) | Pareto / random |
| Plan / Mutate | parent, reflection, knowledge → Candidate | StubAgent / Claude Agent SDK |
| Check | Candidate → CheckReport | static check (built) — cheap gate |
| **Execute (GPU LOCK)** | Candidate → EvalResult | **StubExecutor (built)** / GpuQueueExecutor (later) |
| Reflect | Candidate, EvalResult → Reflection | Stub / Claude Agent SDK |
| Research | problem, reflection → KnowledgeContext | kb retrieval / sibling lookup / web |
| Accept | Candidate, EvalResult, frontier → frontier′ | Pareto-accept |
| Route | state → next node | budget / plateau / optional target |
| Curate | journal → knowledge′ | Claude Agent SDK curator |
| **Submit (LATER)** | best_solution + score → receipt | website submit — *just add this node* |

### The graph (per problem)

```
  select ─▶ plan ─▶ check ──fail──▶ plan
                      └──pass──▶ execute ─▶ accept ─▶ reflect ─▶ route
                                  (GPU lock)                       │
                          ┌──────────── continue ──────────────────┤
                          ▼                                    stop │
                        select                                      ▼
                                                          curate ─▶ (submit?)
```

Cyclic control flow. A small **Driver** runs a node, then a **router** picks
the next node from `(node, output, ctx)` — no heavyweight framework. A config
**binds each role to an implementation**, so the *same graph* runs as
`{StubExecutor + StubAgent}` on the laptop or `{GpuQueueExecutor + Claude
Agent SDK}` for real, with no change to the graph itself.

### Submission, later

`Submit` is a terminal node: input = `best_solution` + score, output = a
receipt. Wire it after `curate` (or as an opt-in branch). Nothing upstream
changes — that is the whole point of the node-graph shape. We still don't
build it now; the graph just leaves a socket for it.

`Executor` (Phase A) is exactly the **Execute** node's role with its first two
implementations. Every other role follows the same interface-per-role pattern.

## 1. Execute node — the GPU lock (interface, swappable)

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

Optimizes one problem to the best kernel (or budget). Holds a **Pareto
frontier of candidate Solutions over the problem's ~16 workload shapes**.

GEPA selection, precisely: the frontier keeps every candidate that is best on
**at least one individual workload shape** — *not* the top-k by aggregate
score. A kernel that wins only the smallest shape survives even with a poor
average, because it preserves a *specialist* worth keeping or merging. This
avoids the local optimum that mutating-from-the-single-best falls into, and it
enables **system-aware merge**: a small-shape winner + a large-shape winner →
one per-shape dispatch kernel that beats both. Parent selection samples from
this frontier; the Reflect node supplies the reflective "text gradient" for
the next mutation.

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

## 6. Termination (per problem)

Objective: **maximize `sol_score`** (find the best correct kernel). No
submission — we stop when we've found the best we can within limits, not when
we clear a submission bar. Stop a ProblemSolver when any holds:
- **Budget exhausted** — GPU-evaluation cap for this problem (the scarce
  resource).
- **Plateau** — K consecutive iterations with no frontier improvement.
- **Target reached (optional)** — a `sol_score` target if set, either a fixed
  threshold (e.g. 0.90) or the current leaderboard rank-1 as a read-only
  reference to stop early once we're clearly competitive. Off by default;
  budget + plateau are the primary stops.

Best correct candidate (or merged per-shape dispatch) becomes
`runs/<id>/best_solution.json`, with its `sol_score` and per-workload
latencies recorded. What happens to it afterward (submit or not) is outside
the engine.

## 7. State & CLI

- `runs/<id>/`: journal.jsonl, frontier/, best_solution.json, status.json —
  fully resumable.
- `knowledge/`: families/*.md, global.md.
- CLI: `solver solve <ids|--all> [--budget N] [--target ...] [--executor stub|gpu]`
  and `solver status`.

## 8. Build order (laptop-first)

| Phase | Buildable now? | Piece (nodes) |
|---|---|---|
| A | ✅ done | Node/edge-type base + **Execute** node (StubExecutor); EvalResult |
| B | ✅ | Driver + router + **Select/Plan/Check/Accept/Reflect/Route** nodes (Stub agent), Pareto frontier, journal/state |
| C | ✅ | **Research/Curate** nodes + knowledge store (dynamic KB) |
| D | ✅ | Orchestrator fleet + budget + CLI (`solve`, `status`) |
| E | ✅ | Transfer: sibling detection + templating (a Research-node strategy) |
| F | ⛔ later | GpuQueueExecutor (real Execute-node impl: GPU transport + harness) |
| —  | ⛔ later | **Submit** node (website) — add node + edge, no loop change |
| G | ⛔ later | Profiling in the eval path (Nsight → ASI) |

A–E give a fully testable engine against the stub; F swaps in the GPU with no
change to the loop, KB, or transfer logic.

## Decisions (defaults chosen 2026-07-05, revisable)

Picked to unblock the foundation; revisit before the agent layer is built.

1. **Agent substrate** → **Claude Agent SDK (Python) on a subscription OAuth
   token**, behind a swappable `Agent` interface (StubAgent for deterministic
   tests; API-key and CLI options also pluggable). `pip install
   claude-agent-sdk`; async `query()` + `ClaudeAgentOptions`; run the fleet
   with `asyncio.gather()`. Auth WITHOUT an API key / API billing: `claude
   setup-token` → `export CLAUDE_CODE_OAUTH_TOKEN=<token>`; the SDK uses it
   automatically and draws from the subscription. **Caveat (live since
   2026-06-15):** programmatic use draws a separate monthly non-interactive
   credit pool (Pro $20 / Max-5x $100 / Max-20x $200); past it, API rates
   apply. → keep the loop GEPA-frugal (few high-value agent calls per problem;
   budget/time-box already enforce this). Standalone engine, not coupled to
   the Claude Code runtime; tmux-of-CLIs is an equivalent-but-messier fallback.
2. **Search driver** → **custom lightweight loop**. Small, full control over
   the GPU lock / per-shape Pareto / cross-problem transfer; no library to bend.
3. **Termination** → **maximize `sol_score`; stop on budget or plateau(K)**.
   No submission (out of scope). Optional read-only `sol_score` target
   (fixed threshold or leaderboard rank-1 reference) can stop early, off by
   default.

Foundation (Phase A: Executor interface + StubExecutor + result model) is
decision-independent and is built first. The contentious layer (agents /
search loop, phases B–D) waits for confirmation.
