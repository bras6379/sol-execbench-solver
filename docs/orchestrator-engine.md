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
handles, budgets) and pass typed artifacts:

- `Candidate` — a Solution (`solver/solution.py`) + lineage (parent id)
- `CheckReport` (`solver/check.py`)
- `NoveltyVerdict` — materially-new | cosmetic-duplicate(of X) | exact-duplicate
- `EvalResult` — per-workload Trace statuses + ASI (`solver/engine`)
- `Reflection` — diagnosis + next-step hint, **attached to its candidate**
- `KnowledgeContext` — design doc + kb slices + family learnings + sibling bests
- `TerminationDecision`

### Node catalog (role → implementations)

| Node (role) | inputs → outputs | implementations |
|---|---|---|
| Select | frontier → (parent, parent.reflection) | Pareto-weighted / random |
| Plan / Mutate | parent, parent's reflection, knowledge → Candidate | StubAgent / Claude Agent SDK; iteration 0 consumes/produces the **design doc** (design-kernel skill) |
| Check | Candidate → CheckReport | static check (built) — cheap gate |
| **Novelty (gate)** | Candidate, lineage/frontier → NoveltyVerdict | tiered: exact hash → normalized hash → **LLM judge** (see below) |
| **Execute (GPU LOCK)** | Candidate → EvalResult | **StubExecutor (built)** / GpuQueueExecutor (later); optional **screen mode** (subset of shapes) |
| Reflect | Candidate, EvalResult → Reflection (attached to candidate) | Stub / Claude Agent SDK |
| Research | problem, reflection → KnowledgeContext | kb retrieval / sibling lookup / web |
| Accept | Candidate, EvalResult, frontier → frontier′ | ε-Pareto-accept |
| **Merge** | frontier specialists → Candidate (per-shape dispatch) | same-family mechanical merge / Plan-rewrite across families |
| Route | state → next node | budgets / plateau / optional target |
| Curate | journal → knowledge′ | Claude Agent SDK curator (**serialized** — single queue) |
| **Submit (LATER)** | best_solution + score → receipt | website submit — *just add this node* |

### The graph (per problem)

```
  select ─▶ plan ─▶ check ──fail──▶ plan       (bounce w/ errors)
                      └─pass─▶ novelty ──dup──▶ plan  (bounce w/ "cosmetic" verdict)
                                  └─new─▶ execute ─▶ accept ─▶ reflect ─▶ route
                                           (GPU lock)                      │
                          ┌──────────────── continue ──────────────────────┤
                          ▼                                           stop │
                        select                                             ▼
                                                       merge? ─▶ curate ─▶ (submit?)
```

Cyclic control flow. A small **Driver** runs a node, then a **router** picks
the next node from `(node, output, ctx)` — no heavyweight framework. A config
**binds each role to an implementation**, so the *same graph* runs as
`{StubExecutor + StubAgent}` on the laptop or `{GpuQueueExecutor + Claude
Agent SDK}` for real, with no change to the graph itself.

**Every pass through Plan counts as an iteration** — including check-fails and
novelty-bounces — so plateau/budget stops fire even when nothing reaches the
GPU (no spin-forever).

### The Novelty gate (dedup is semantic, not just hashes)

Exact hashing alone is wrong for dedup: a renamed variable or reshuffled
comments yields a new hash but the same kernel — and a wasted GPU run. Tiered
gate, cheapest first; each tier only runs if the previous passed:

1. **Exact**: the harness `Solution.hash()` (SHA1 over languages, entry, deps,
   sources). Free; catches byte-identical.
2. **Normalized**: hash after stripping comments/whitespace (and
   AST-normalizing Python). Free; catches formatting/rename-only churn.
3. **LLM judge** (cheap model): "is this a *materially different*
   implementation from its parent / nearest frontier member — new algorithm,
   layout, fusion, precision strategy, or launch config — or a cosmetic
   variant?" A judge call costs orders of magnitude less than the GPU eval it
   protects. Verdict `cosmetic-duplicate` bounces to Plan **with the verdict
   as feedback** ("that was cosmetic — propose a materially different
   change"), which is better than silently skipping.

(Same rule the old repo's prompts used: renames/comments/formatting do not
count as materially different candidates.)

### Candidate discipline (clean artifact, any language)

Two invariants for the Solutions the loop generates:

1. **The Solution stays clean — engine bookkeeping never leaks in.** A
   `Candidate`'s `spec`/`sources` are *only* the kernel the harness runs.
   Status tags, parent, technique, scores, and reflections live in the
   engine's own `meta.json`/`result.json`/journal *beside* the solution, never
   inside `solution.json`. What ships to the GPU is exactly a kernel.
2. **The baseline is a seed, not a mold.** `scaffold` produces one starting
   candidate — a correct PyTorch DPS wrapper delegating to the inlined
   reference. Generated candidates do **not** inherit that shape: free-form
   real kernels in **any harness language** — pytorch, triton, cute_dsl,
   cutile, cudnn_frontend, or C++ (`cuda_cpp`/`cutlass`/`cudnn`/`cublas`),
   including **multi-file** `sources` with `compile_options`/`dependencies`.
   (No C++ scaffold needed — the seed is always the Python baseline; mutations
   cross into C++.) Note: the seed's *measured* score may be **below 0.5** —
   `T_b` is an *optimized* PyTorch baseline, not the raw reference wrapper.

Consequences the nodes must honour:
- **Check** validates *by language*: DPS parameter names for python entries;
  the `void run(torch::Tensor …)` signature + compilable structure for C++.
- **Execute** for C++ has a **compile sub-stage** (`COMPILE_ERROR` is
  solution-level: nothing ran).
- **Observability** materializes the *actual* entry file(s) (`kernel.cu`,
  `kernel.cpp`, or `kernel.py`) and every `source`.

## 1. Execute node — the GPU lock (interface, swappable)

The one serialized resource. Everything else fans out.

```python
class Executor(Protocol):
    def evaluate(self, solution: dict, task_id: int, *,
                 shapes: list[int] | None = None,   # None = all; subset = screen mode
                 profile: bool = False) -> EvalResult: ...

@dataclass
class WorkloadResult:
    index: int
    status: str            # per-workload Trace enum: PASSED | INCORRECT_SHAPE |
                           # INCORRECT_NUMERICAL | INCORRECT_DTYPE | RUNTIME_ERROR |
                           # TIMEOUT | REWARD_HACK | INVALID_REFERENCE
    latency_ms: float | None
    sol_ms: float | None
    baseline_latency_ms: float | None
    matched_ratio: float | None

@dataclass
class EvalResult:
    solution_status: str | None   # solution-level only: COMPILE_ERROR | REWARD_HACK | None
    per_workload: list[WorkloadResult]   # one Trace per evaluated shape
    all_passed: bool                     # derived: every evaluated shape PASSED
    sol_score: float | None              # mean over evaluated shapes; None if not all_passed
    gpu_seconds: float                   # actual GPU time consumed (budget input)
    asi: dict                            # logs, profile, notes for reflection
    raw: dict                            # full Trace payloads
```

**Status is per-workload** (the harness emits one Trace per shape — a kernel
can PASS 14 shapes and TIMEOUT on 2, which is exactly the partial-specialist
signal Accept and Reflect need). Only `COMPILE_ERROR` (nothing ran) and
`REWARD_HACK` (quarantine) are solution-level. Each status drives a different
reflection: `COMPILE_ERROR` → code fix; `INCORRECT_NUMERICAL` →
precision/tolerance; `TIMEOUT` → perf/hang.

**Screen mode (GEPA minibatch analogue):** `shapes=[...]` evaluates a 2–3
shape subset first (the harness CLI accepts a custom workload file). Full
16-shape runs are spent only on candidates that survive the screen. Budgets
are therefore counted in **workload-runs and GPU-seconds**, not eval calls.

**REWARD_HACK policy:** the candidate is quarantined — never selected, never
merged, and its trace is **not** given to Reflect as an optimization signal
(the one thing reflection must never learn is evasion). The router bounces to
Plan with a neutral "disallowed pattern; regenerate from parent" note, and the
event is journaled loudly.

- `StubExecutor` (now): validates via `solver.check`, then synthesizes
  latencies with a **specified stub model**: deterministic pseudo-random
  improvement factor keyed on `Solution.hash()` (the stub sees only the clean
  Solution — no engine metadata), plus an optional **noise term** (to exercise
  ε-domination) and optional scripted scenarios for tests.
- `GpuQueueExecutor` (later): single-flight queue driving the real harness —
  compile sub-stage (`build_ext.py` → `cpp_extension.load` → `.so`) then the
  eval subprocess (`eval_driver.py`) — parsing emitted **Trace JSONL** into
  `EvalResult`. Dedup/build-cache by `Solution.hash()`. Concurrency **1**.
  Details: [kb/solution-format.md](../kb/solution-format.md).

All ProblemSolvers share **one** Executor instance → the lock is structural.

## 2. ProblemSolver — per-question GEPA loop

Optimizes one problem to the best kernel (or budget). Holds a **Pareto
frontier of candidate Solutions over the problem's ~16 workload shapes**.

GEPA selection, precisely: the frontier keeps every candidate that is best on
**at least one individual workload shape** — *not* the top-k by aggregate
score. A kernel that wins only the smallest shape survives even with a poor
average: it preserves a *specialist* worth keeping or merging. This avoids the
local optimum of mutating-from-the-single-best, and enables **merge** (§6).

**Exact frontier rule (Accept node contract):** score each candidate as a
vector — `sol_score` per shape; any non-PASSED shape scores 0. Candidate A
**ε-dominates** B iff A ≥ B − ε on *every* shape and A > B + ε on *at least
one* (ε ≈ measurement noise, config; default a few % relative). After each
evaluation: add the new candidate, drop any candidate ε-dominated by another.
**Confirm-before-promote:** a candidate whose wins are all within ε of the
incumbent is re-measured (or held) rather than churning the frontier — single
measurements must not decide near-ties. **Select** samples a parent from the
frontier (e.g. weighted by shapes won), returning the parent **with its own
attached reflection** — a reflection is a diagnosis of *that* candidate and
never floats free to be applied to a different parent.

```
seed  = scaffold(id) (reference wrapper)  ∪  transferred sibling templates
iter 0: plan consumes/produces the DESIGN DOC (design-kernel skill output)
loop until budgets or plateau(K) or optional target:
    parent, parent_reflection = select_from_frontier()
    context   = retrieve_knowledge(id)      # design doc + kb slices + family learnings
    candidate = plan(parent, parent_reflection, context)     # LLM mutation
    if not check(candidate):   journal(reject);  continue    # iteration counts
    if not novel(candidate):   journal(dup);     continue    # iteration counts, bounce w/ verdict
    result    = execute(candidate, screen-then-full)         # THE GPU LOCK
    accept_if_eps_pareto(candidate, result)
    candidate.reflection = reflect(candidate, result)         # attached to lineage
    journal(candidate, result, reflection)
```

Agents (plan/reflect) are LLM calls grounded in `kb/optimization-recipe.md`
and the family KB; they run concurrently with every other problem's agents.

## 3. Orchestrator — the fleet

Runs N ProblemSolvers concurrently over the selected ids; allocates the shared
GPU budget across problems — default weight ∝ **headroom** = 1 − best
all-correct `sol_score` (round-robin and priority as alternatives). Because
agents are parallel and only `evaluate()` serializes, the GPU is never idle
waiting on thinking.

## 3b. Budgets (three, any one stops the run)

| Budget | Unit | Why |
|---|---|---|
| GPU work | **workload-runs + GPU-seconds** | eval-count undercounts: a TIMEOUT candidate burns up to 600 s; a screen run is ~5× cheaper than a full run |
| Agent credit | tokens / $ (subscription non-interactive pool) | a chatty reflect/mutate loop can exhaust the monthly pool without touching the GPU |
| Iterations | count (incl. rejects and novelty bounces) | guarantees termination even if nothing reaches the GPU |

`RunContext` tracks all three; Route stops on whichever trips first. Every
journal entry records the budget deltas.

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
  - **Global learnings** (`knowledge/global.md`): cross-cutting facts.

A **curator step** (LLM) runs when a problem finishes: reads the journal and
updates family + global learnings (merge/rewrite, not append; bounded size).
**Curation is serialized** — one curator queue for the whole fleet — so two
same-family problems finishing together can't clobber each other's writes
(same pattern as the GPU lock, applied to the knowledge store).

## 5. Knowledge transfer (the crux)

Sample-efficiency comes from not re-deriving what a sibling already solved.
Mechanisms, cheapest first:

1. **Sibling templating.** Many problems are shape-variants (230–235 are
   rmsnorm at H=128…7168; 213–220 are GEMMs at different N/K). A winning
   Solution for one is a near-direct template for the next — port constants,
   don't re-search. The engine detects "same family + same op, different
   axes" and **seeds the frontier from the sibling's best**.
2. **Family learnings retrieval.** At plan time, inject the family's
   distilled learnings + current family bests into the agent context.
3. **Design doc + static KB slices.** Iteration 0's Plan runs the
   `design-kernel` recipe (or ingests an existing `designs/<name>.md`): op
   graph, per-shape roofline, ranked approaches, open questions — this is the
   grounding for the first real candidates, not generic retrieval.
4. **Cross-problem Pareto/merge.** A technique that won on one problem seeds
   mutations on related ones.
5. **Curation feedback.** Each finished problem improves the family
   learnings; solve one exemplar per family first, then its cheap siblings.

## 6. Merge & the deliverable (all-correct invariant)

**Invariant:** `best_solution.json` must be **PASSED on every workload**. The
frontier may hold partially-correct specialists (they score 0 on failed
shapes but win others — useful genetic material), but the deliverable is the
argmax of mean `sol_score` **among all-correct candidates**. The seed
(reference wrapper) guarantees at least one all-correct candidate always
exists.

**Merge node:** combines frontier specialists into one per-shape-dispatch
kernel (branch on runtime shape inside `run`). Two modes:
- **Same-language-family** (both Python or both C++): mechanical merge —
  dispatch wrapper + both kernels in one Solution's `sources`.
- **Cross-family** (e.g. a Triton winner + a cuda_cpp winner): **cannot be
  merged mechanically — the harness forbids mixing Python and C++ languages
  in one Solution.** Merge degrades to a Plan-node rewrite: "port the losing
  side's technique into the winning side's language."
Run Merge before finalization (and opportunistically on plateau); a merged
candidate goes through the same novelty→execute→accept path as any other.

## 7. Termination (per problem)

Objective: **maximize `sol_score`**. Stop on any of: **budgets** (§3b),
**plateau** — K consecutive iterations (of any kind) without ε-improvement of
the frontier — or the **optional target** (fixed threshold or read-only
leaderboard reference; off by default). Then Merge → finalize
`best_solution.json` (all-correct argmax) → Curate.

## 8. Durability & resume

**Principle: the journal is the single source of truth; everything else is a
rebuildable cache.** Kill at any instant; restart replays the journal and
continues, **never re-paying for completed GPU or agent work.**

### Two caching mechanisms — do not conflate

1. **Crash-replay cache** (`runs/<id>/cache/`, content-addressed
   `hash(node_role, impl_version, inputs, attempt_nonce)`): exact, mechanical,
   LLM-free. Purpose: after a crash, a node that already completed returns its
   recorded output instead of re-running. **Generative nodes (Plan/Reflect)
   include an attempt nonce** in the key: the cache replays a *specific past
   attempt*; it never makes a *fresh* attempt return a stale result. (Without
   the nonce, a plateau would deadlock: same inputs → same cached mutation →
   dup → same inputs → …)
2. **Novelty gate** (§ above): semantic dedup *of candidates* before the GPU —
   hashes first, LLM judge last. This is where "is the code really the same?"
   is decided; the crash cache never decides that.

### What's persisted (per problem, `runs/<id>/`)

- **`journal.jsonl` — append-only, the authority.** One fsync'd line per
  transition (`node_start`, `node_done`, `frontier_update`, `budget`,
  `route`, `terminated`). A crash truncates at most the last line (dropped on
  replay).
- **`cache/`** — the crash-replay cache (gitignored; regenerable).
- **`frontier/`, `best_solution.json`, `status.json`, `candidates/`** —
  **derived, journal-rebuildable snapshots** for humans and fast startup;
  written atomically. `candidates/<cand_id>/` materializes the *actual*
  source files (`kernel.py`/`kernel.cu`/…), `solution.json` (clean),
  `result.json` (per-shape statuses/scores), `meta.json` (parent, technique,
  status: `frontier | dominated | rejected | duplicate | incorrect |
  compile_error | runtime_error | timeout | reward_hack`).

**Git policy:** run outputs (journal, candidates, frontier, best_solution,
status) are **git-visible**; only regenerable churn is ignored (`.cache/`,
`runs/*/cache/`).

### The resumable driver

```
restart:  state = replay(journal); node = state.next or graph.start
loop:
  key = hash(node, impl_version, inputs [, attempt_nonce if generative])
  if cache.has(key): output = cache.get(key)                 # crash replay → reuse
  else:
    append(journal, node_start{node, key})                   # write-ahead intent
    output = node.run(ctx, inputs)
    cache.put(key, output); append(journal, node_done{key})  # + fsync
  ctx.apply(output)                                           # frontier/budgets (journaled)
  node, inputs = router.next(node, output, ctx); append(journal, route{node})
```

### The GPU node — reconciled by job-id

Backed by a durable queue with stable job-ids (pending→processing→completed,
the old repo's pattern): journal `execute_submitted{job_id, solution_hash}`
before waiting; the GPU-side worker persists `completed/<job_id>.json`
independently of the engine. On restart: completed → recover result;
processing → wait; lost → resubmit. `Solution.hash()` (the harness's own
SHA1) is the dedup/build-cache key throughout. Killing the engine
mid-evaluation loses no GPU work. (Stub: trivial in-process queue.)

### The fleet

Each `runs/<id>/` is self-contained; the Orchestrator rescans run dirs on
restart and resumes in-progress problems. Its allocation state is a small
journaled file.

## 9. Observability (watch the search happen)

- **Browse on disk:** `runs/<id>/candidates/*/` — real source files + status
  in `meta.json`; sort by `result.json` scores. No tooling required.
- **CLI views** (read-only over the journal): `solver status [ids]`
  (iterations, budgets spent, best vs baseline & SOL, frontier size);
  `solver journal <id>` (the timeline incl. novelty bounces and reflections);
  `solver frontier <id>` (who survives and which shapes each wins);
  `solver candidates <id> [--status …]` (everything tried, filterable).
- **Report (later):** self-contained HTML dashboard (score-over-iterations,
  frontier, per-shape heatmap, parent→child diffs) as an Artifact.

## 10. Build order (laptop-first)

| Phase | Buildable now? | Piece (nodes) |
|---|---|---|
| A | ✅ done | Execute node role + StubExecutor (needs: per-workload status, screen mode, stub speed/noise model) |
| B | ✅ | Driver + router + journal/replay/cache (attempt nonce) + Select/Plan/Check/Novelty/Accept/Reflect/Route (Stub agent) + ε-frontier |
| C | ✅ | Research/Curate nodes + knowledge store (serialized curator) |
| D | ✅ | Orchestrator fleet + triple budgets + CLI (`solve`, views) |
| E | ✅ | Transfer: sibling detection + templating; Merge node (same-family) |
| F | ⛔ later | GpuQueueExecutor (compile sub-stage + eval + Trace parse) |
| — | ⛔ later | Submit node (website) — add node + edge, no loop change |
| G | ⛔ later | Profiling in the eval path (Nsight → ASI) |

A–E give a fully testable engine against the stub; F swaps in the GPU with no
change to the loop, KB, or transfer logic.

## Decisions (updated 2026-07-05)

1. **Agent substrate** → **Claude Agent SDK (Python) on a subscription OAuth
   token**, behind a swappable `Agent` interface (StubAgent for tests).
   `claude setup-token` → `CLAUDE_CODE_OAUTH_TOKEN`; no API key. Caveat: the
   non-interactive monthly credit pool (Pro $20 / Max-5x $100 / Max-20x $200)
   → agent credit is a **tracked budget** (§3b), and the loop stays
   GEPA-frugal. The Novelty judge uses a cheap model.
2. **Search driver** → **custom lightweight loop** (this document); the gepa
   library's *concepts* (reflection, ε-Pareto, merge) without bending its API
   around our GPU lock. (Supersedes the "use the gepa library" note in
   orchestration.md.)
3. **Termination** → maximize `sol_score`; stop on budgets or plateau(K);
   optional read-only target off by default. No submission.
4. **Dedup** → crash-replay caching is exact/mechanical; **candidate novelty
   is semantic** — tiered hash → normalized-hash → LLM judge (user decision
   2026-07-05).
