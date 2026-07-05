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

### Candidate discipline (clean artifact, any language)

Two invariants for the Solutions the loop generates (the "pareto iterations"):

1. **The Solution stays clean — engine bookkeeping never leaks in.** A
   `Candidate`'s `spec`/`sources` are *only* the kernel the harness runs.
   Status tags, parent, technique, scores, and reflections live in the
   engine's own `meta.json`/`result.json`/journal *beside* the solution, never
   inside `solution.json`. What ships to the GPU is exactly a kernel, nothing
   of ours.
2. **The baseline is a seed, not a mold.** `scaffold` produces one starting
   candidate — a correct PyTorch DPS wrapper that delegates to the inlined
   reference (`_reference_run`, score ≈ 0.5). Generated candidates do **not**
   inherit that shape: they drop the reference delegation and are free-form
   real kernels, in **any language the harness supports** — pytorch, triton,
   cute_dsl, cutile, cudnn_frontend, or C++ (`cuda_cpp`/`cutlass`/`cudnn`/
   `cublas`), including **multi-file** `sources` with `compile_options` /
   `dependencies`. Only the seed is Python-and-reference-shaped; the search is
   not. (So no C++ scaffold is needed — the seed is always the Python
   baseline; mutations cross into C++.)

Consequences the nodes must honour:
- **Check** validates *by language*: DPS parameter names for python entries;
  the `void run(torch::Tensor …)` signature + compilable structure for C++.
- **Execute** for C++ has a **compile step**, so `EvalResult` distinguishes
  `compile_error` / `runtime_error` / `incorrect` / `correct` — each drives a
  different reflection (a compile error is a code fix, not a perf idea).
- **Observability** materializes the *actual* entry file(s) (`kernel.cu`,
  `kernel.cpp`, or `kernel.py`) and every `source`, not a hardcoded
  `kernel.py`.

## 1. Execute node — the GPU lock (interface, swappable)

The one serialized resource. Everything else fans out.

```python
class Executor(Protocol):
    def evaluate(self, solution: dict, task_id: int, *, profile: bool = False) -> EvalResult: ...

@dataclass
class EvalResult:
    status: str                              # correct | incorrect | compile_error | runtime_error
    sol_score: float | None                 # via solver/scoring.py; None unless correct
    per_workload: list[WorkloadResult]       # matched_ratio, latency_ms, sol_ms, error
    asi: dict                                # actionable side info: errors, profile, notes
    raw: dict                                # full harness payload
```

(The stub only produces `correct`/`incorrect` since it doesn't compile; the
`compile_error`/`runtime_error` states appear with the real GPU executor and
C++ candidates. `correct` is a convenience for `status == "correct"`.)

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

**Exact frontier rule (Accept node contract):** score each candidate as a
vector — its `sol_score` (or latency) on each of the ~16 shapes; incorrect
shapes score 0. Candidate A **dominates** B iff A ≥ B on *every* shape and A >
B on *at least one*. After each evaluation: add the new candidate, then drop
any candidate that is dominated by another. The survivors are the frontier —
equivalently, the set of candidates each of which is the (co-)best on at least
one shape. **Select** samples a parent from this set (e.g. weighted by how
many shapes it wins). This is per-shape non-domination, deliberately *not*
top-k by aggregate.

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

## 7. Durability & resume

**Principle: the journal is the single source of truth; everything else is a
rebuildable cache.** The engine may be killed at any instant; on restart it
replays the journal and continues from the last completed step, **never
re-paying for GPU or agent work it already did.** This falls out of the
node-graph shape: persist at every node boundary, and make each node
idempotent on replay.

### What's persisted (per problem, `runs/<id>/`)

- **`journal.jsonl` — append-only, the authority.** One line per transition,
  written and **fsync'd before the engine acts on it** (write-ahead):
  `node_start`, `node_done` (with an output ref), `frontier_update`,
  `budget`, `route`, `terminated`. A crash truncates at most the last line,
  which is dropped on parse error during replay.
- **`cache/<hash>.json` — content-addressed node outputs**, keyed by
  `hash(node_role, impl_version, inputs)`. Memoizes every node result →
  idempotent replay: a node whose inputs already have a cached output is
  **not** re-run. This is what prevents re-spending a GPU evaluation or an
  agent call. `impl_version` in the key invalidates stale outputs when a node
  implementation changes.
- **`frontier/`, `best_solution.json`, `status.json` — derived snapshots**,
  rebuildable from the journal; written atomically (temp + `os.replace`) for
  fast startup, never authoritative.
- **`candidates/<cand_id>/` — browsable per candidate** (not opaque JSON):
  the **actual entry source file(s)** materialized from the Solution
  (`kernel.cu` / `kernel.cpp` / `kernel.py` — whatever the candidate is — so
  you can just *open and read it*), `solution.json` (the canonical artifact,
  kernel-only, no engine bookkeeping), `result.json` (per-shape scores,
  sol_score, correctness, ASI), `meta.json` (parent, technique tag, and
  **status**: `frontier` | `dominated` | `rejected` (failed static check) |
  `incorrect` | `compile_error` | `runtime_error`).

**Git policy:** run outputs (`journal.jsonl`, `candidates/`, `frontier/`,
`best_solution.json`, `status.json`) are **git-visible** — check them in when
you want to snapshot/share progress. Only regenerable churn is ignored:
`.cache/` (parquet) and `runs/*/cache/` (the content-addressed dedup blobs).

### The resumable driver

```
restart:  state = replay(journal); node = state.next or graph.start
loop:
  key = hash(node, impl_version, inputs)
  if cache.has(key): output = cache.get(key)                 # already done → reuse
  else:
    append(journal, node_start{node, key})                   # write-ahead intent
    output = node.run(ctx, inputs)
    cache.put(key, output); append(journal, node_done{key})  # + fsync
  ctx.apply(output)                                           # frontier/budget (journaled)
  node, inputs = router.next(node, output, ctx); append(journal, route{node})
```

A crash leaves at most a trailing `node_start` with no `node_done`. On restart
that node is simply re-run — safe because content-addressed caching makes
every node idempotent, **except** the GPU node, which is reconciled by job-id.

### The GPU node — the one non-idempotent, expensive step

The Execute node must never silently re-run an evaluation we already paid for,
and must survive a crash *during* a run. It is backed by a durable queue with
**stable job-ids** (the old repo's `gpu_jobs/` pending→processing→completed
pattern):

- enqueue: write `gpu_jobs/pending/<job_id>` (candidate + task) and journal
  `execute_submitted{job_id, candidate_hash}` **before** waiting.
- the GPU-side worker moves it to `completed/<job_id>.json` durably —
  **independently of the engine process**.
- on restart, for any `execute_submitted` with no `execute_done`: find
  `job_id` in `completed/` → recover the result (no re-run); if still in
  `processing/` → wait; if lost → resubmit. `candidate_hash` also dedups
  identical candidate sources so the same kernel is never evaluated twice.

Because results are persisted queue-side, killing the engine mid-evaluation
loses no GPU work. (`StubExecutor` today is in-process/cheap, so its "queue"
is trivial; this matters when `GpuQueueExecutor` lands.)

### The fleet

Each `runs/<id>/` is self-contained. On restart the Orchestrator scans run
dirs to reconstruct the active set (done / in-progress / pending) and resumes
the in-progress ones; its own allocation state is a small journaled file,
likewise rebuildable.

## 7c. Observability (watch the search happen)

A first-class requirement: you should be able to *see* the optimization
progressing — every kernel tried, which were kept, which were beaten, and the
live Pareto frontier. Three surfaces, all reading the journal:

- **Browse on disk.** `runs/<id>/candidates/*/kernel.py` are readable source
  files; each `meta.json` carries the candidate's status
  (`frontier`/`dominated`/`rejected`/`incorrect`). Sort by `result.json`
  sol_score, grep by status — no tooling required.
- **CLI views** (read-only over the journal):
  - `solver status [ids]` — per problem: iterations, GPU-evals spent/budget,
    best sol_score vs baseline & SOL, frontier size, terminated?.
  - `solver journal <id>` — the timeline: each candidate as it was created →
    checked → evaluated → accepted/dominated/rejected, with the reflection.
  - `solver frontier <id>` — the current Pareto set: each surviving candidate
    and which workload shapes it wins (the "why it's kept").
  - `solver candidates <id> [--status dominated|frontier|rejected]` — list/
    filter every kernel tried, so "what got ignored" is one query.
- **Report (optional, later).** Render a self-contained HTML progress
  dashboard (score-over-iterations, the frontier, per-shape heatmap, diffs
  between a candidate and its parent) as an Artifact.

Everything is derived from `journal.jsonl`, so these are pure read views — safe
to run against a live or a finished run.

## 7d. Knowledge store & CLI

- `knowledge/`: families/*.md, global.md — the dynamic KB (§4); curated writes
  are journaled so a crash mid-curation is recoverable.
- Run CLI: `solver solve <ids|--all> [--budget N] [--resume] [--executor stub|gpu]`.
  `--resume` is the default; re-running `solve` on the same ids continues
  rather than restarts. View CLI is in §7c.

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
