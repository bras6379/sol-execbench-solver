# Orchestrator Engine Design

Concrete engine on top of the [orchestration loop](orchestration.md). Goal:
**select problems → optimize each to the best kernel we can find → accumulate
and transfer knowledge across problems.** Most of this is buildable now with
the GPU abstracted behind an interface (stub today, real transport later).

**Scope:** the engine *finds* the best solution per problem. It does **not**
submit to the SOL-ExecBench website/leaderboard — no submission code. The
deliverable per problem is `best_solution.json` + its measured `sol_score`.
The public leaderboard is used, if at all, only as a read-only reference.

## Component map

```
                 ┌──────────────────────────────────────────────┐
   selected ids  │  Orchestrator (fleet)                         │
   ────────────▶ │   budget allocation across problems           │
                 │   ┌───────────────┐  ┌───────────────┐  ...   │
                 │   │ ProblemSolver │  │ ProblemSolver │        │  ← async tasks, crash-isolated
                 │   │  (GEPA loop)  │  │  (GEPA loop)  │        │
                 │   └──────┬────────┘  └──────┬────────┘        │
                 └──────────┼──────────────────┼────────────────┘
                            │ evaluate()        │ evaluate()
                            ▼                   ▼
                 ┌──────────────────────────────────────────────┐
                 │  Executor:  Compile (parallel) ─▶ Run (GPU    │
                 │  LOCK, conc=1, priority queue)                │  ← GPU run is the only serialized resource
                 │   StubExecutor (now)  |  GpuQueueExecutor(later)│
                 └──────────────────────────────────────────────┘
        reads/writes ▲                              ▲ writes
                 ┌───────────────────────┐   ┌──────────────────┐
                 │ Knowledge (2 layers)  │   │ Run state        │
                 │  static kb/ (priors)  │   │ runs/<id>/journal│
                 │  dynamic knowledge/   │   │ frontier, best   │
                 └───────────────────────┘   └──────────────────┘
```

## 0. Execution model (async throughout)

**One asyncio process.** Each ProblemSolver is an asyncio task; agent calls
(Claude Agent SDK `query()`) are awaited; the GPU lock is an `asyncio.Lock`
(or an awaitable queue), and any genuinely blocking work (subprocess waits,
file I/O bursts) runs in `asyncio.to_thread`. **No synchronous lock may sit
on the event-loop path** — a blocking `evaluate()` under a `threading.Lock`
would freeze every other problem's agents, violating the core property
(agents never wait on each other, only on the GPU). The built
`StubExecutor`'s `threading.Lock` is a Phase-A artifact to be converted.

**Fleet failure containment:** a ProblemSolver that raises is caught,
journaled (`solver_error`), and only that problem stops; the fleet continues.
Each problem is independently resumable, so a crashed solver can be retried
without touching the others.

## Nodes: the unit of composition

Every step is a **Node**: a typed transform `run(ctx, inputs) -> outputs`
whose **role** (contract) is separate from its **implementation**. The
per-problem loop is a cyclic graph of nodes; the Orchestrator runs many
graphs in parallel. Adding a capability later — submission, a profiler, a
different planner — means add a node or swap an implementation; nothing
upstream changes.

### Edge types

Nodes share a `RunContext` (problem, frontier, journal, knowledge handles,
budgets) and pass typed artifacts:

- `Candidate` — a Solution (`solver/solution.py`) + lineage (parent id)
- `CheckReport` (`solver/check.py`)
- `NoveltyVerdict` — materially-new | cosmetic-duplicate(of X) | exact-duplicate
- `EvalResult` — per-workload Trace statuses + stats + ASI
- `Reflection` — diagnosis + next-step hint, **attached to its candidate**
- `KnowledgeContext` — design doc + kb slices + family learnings + sibling bests
- `TerminationDecision`

### Node catalog (role → implementations)

| Node (role) | inputs → outputs | implementations |
|---|---|---|
| Select | frontier → (parent, parent.reflection) | Pareto-weighted / random |
| Plan / Mutate | parent, parent's reflection, knowledge → Candidate | StubAgent / Claude Agent SDK; iteration 0 consumes/produces the **design doc**; governed by the **mutation policy** (below) |
| Check | Candidate → CheckReport | static check (built) — cheap gate |
| **Novelty (gate)** | Candidate, lineage/frontier → NoveltyVerdict | tiered: exact hash → normalized hash → **LLM judge** (cheap model) |
| **Execute** | Candidate → EvalResult | **Compile stage (parallel, cached)** + **Run stage (GPU LOCK)**; StubExecutor (built) / GpuQueueExecutor (later); **screen mode** |
| Reflect | Candidate, EvalResult → Reflection | tiered (below): SDK full / cheap-model brief / none |
| Research | problem, reflection → KnowledgeContext | kb retrieval / sibling lookup / web (**off by default**, see Security) |
| Accept | Candidate, EvalResult, frontier → frontier′ | ε-Pareto-accept (**full evals only**) |
| **Merge** | frontier specialists → Candidate (per-shape dispatch) | same-family mechanical / Plan-rewrite across families |
| Route | state → next node | budgets / plateau / circuit breaker / optional target |
| Curate | journal → knowledge′ | Claude Agent SDK curator (**serialized** queue) |
| **Submit (LATER)** | best_solution + score → receipt | website submit — just add this node |

### The graph (per problem)

```
  select ─▶ plan ─▶ check ──fail──▶ plan       (bounce w/ errors)
                      └─pass─▶ novelty ──dup──▶ plan  (bounce w/ "cosmetic" verdict)
                                  └─new─▶ execute ─▶ accept ─▶ reflect ─▶ route
                                     (compile ∥ ; run under GPU lock)      │
                          ┌──────────────── continue ──────────────────────┤
                          ▼                                           stop │
                        select                                             ▼
                                                       merge? ─▶ curate ─▶ (submit?)
```

A small **Driver** runs a node; a **router** picks the next from
`(node, output, ctx)`. A config binds each role to an implementation, so the
same graph runs `{StubExecutor + StubAgent}` on the laptop or
`{GpuQueueExecutor + Claude Agent SDK}` for real.

**Every pass through Plan counts as an iteration** — including check-fails
and novelty-bounces — so plateau/budget stops fire even when nothing reaches
the GPU.

### The Novelty gate (dedup is semantic, not just hashes)

Tiered, cheapest first: (1) exact `Solution.hash()`; (2) normalized hash
(strip comments/whitespace, AST-normalize Python) — catches rename/format
churn; (3) **LLM judge** (cheap model): materially different implementation
(algorithm/layout/fusion/precision/launch-config) or cosmetic variant? A
judge call costs orders of magnitude less than the GPU eval it protects.
`cosmetic-duplicate` bounces to Plan **with the verdict as feedback**.

### Candidate discipline (clean artifact, any language)

1. **The Solution stays clean.** `spec`/`sources` are only the kernel;
   status, parent, technique, scores, reflections live in engine files
   beside it, never inside `solution.json`.
2. **The baseline is a seed, not a mold.** The seed is the correct PyTorch
   DPS reference wrapper; generated candidates are free-form kernels in any
   harness language (python family or C++ family, multi-file, own
   `compile_options`/`dependencies`). The seed's measured score may be
   **below 0.5** (T_b is an *optimized* baseline, not the raw reference).
3. Check validates per-language (DPS names for py; `void run(torch::Tensor…)`
   + structure for C++). Observability materializes the actual source files.

### Mutation policy (what Plan is allowed to try, and when)

Encodes the KB's abstraction ladder so the agent doesn't jump to exotic
backends on iteration 1:

- **Ladder order** (default): torch-level rewrite → Triton → CuTe-DSL /
  CUTLASS → C++/PTX. **Escalation requires justification**: plateau at the
  current rung, profiling evidence, or an explicit design-doc recommendation.
  C++ rungs also pay compile time — the policy accounts for that.
- **Explore/exploit knob**: mostly mutate frontier parents (exploit);
  a configurable fraction of iterations instead *seeds fresh* from the design
  doc's next-ranked approach (explore) — protection against a frontier full
  of one mediocre idea's descendants.
- **Context budget per Plan call**: parent source + parent's own reflection +
  top-K curated insights + the design doc section for the chosen approach.
  Never the full lineage history (the journal holds history; the prompt
  doesn't grow with iterations).

### Reflection tiering (credit economics)

- Frontier-entrants and near-misses (within ~2ε): **full** reflection (SDK).
- Clear failures/regressions: **brief** one-liner via the cheap model
  (status + first error line → hint).
- Exact/cosmetic duplicates and quarantined candidates: **no reflection**.

## 1. Execute node — compile in parallel, run under the lock

**The GPU lock covers only the timed run.** C++ compilation (30–120+ s of
CPU) happens in a **parallel compile stage** — mirroring the harness's own
compile-server/eval-server split — keyed by `Solution.hash()` so rebuilds are
free. The GPU never idles waiting for `nvcc`.

```python
class Executor(Protocol):
    async def evaluate(self, solution: dict, task_id: int, *,
                       shapes: list[int] | None = None,   # None = all; subset = screen
                       profile: bool = False) -> EvalResult: ...

@dataclass
class WorkloadResult:
    index: int
    status: str            # per-workload Trace enum: PASSED | INCORRECT_SHAPE |
                           # INCORRECT_NUMERICAL | INCORRECT_DTYPE | RUNTIME_ERROR |
                           # TIMEOUT | REWARD_HACK | INVALID_REFERENCE
    latency_ms: float | None       # central estimate (median)
    latency_spread: float | None   # measured dispersion (e.g. IQR/std over iters)
    sol_ms: float | None
    baseline_latency_ms: float | None
    matched_ratio: float | None

@dataclass
class EvalResult:
    solution_status: str | None   # solution-level only: COMPILE_ERROR | REWARD_HACK | None
    per_workload: list[WorkloadResult]
    all_passed: bool
    sol_score: float | None       # mean over evaluated shapes; None if not all_passed
    gpu_seconds: float            # budget input
    env: dict                     # environment fingerprint (below)
    asi: dict                     # logs, profile, notes for reflection
    raw: dict                     # full Trace payloads
```

- **Status is per-workload** (one Trace per shape); only `COMPILE_ERROR`
  (nothing ran) and `REWARD_HACK` (quarantine) are solution-level.
- **Screen mode** (GEPA minibatch analogue): evaluate 2–3 shapes first; full
  16-shape runs only for candidates that survive the screen. **Screen
  results never enter the frontier** — admission requires a full eval;
  screens are only a gate. Budgets count **workload-runs + GPU-seconds**.
- **Queue discipline** (GpuQueueExecutor): priority queue — screens before
  full runs, per-problem fairness (no starvation). **Per-problem circuit
  breaker**: repeated TIMEOUT/RUNTIME_ERROR evals suspend that problem
  (journaled) instead of eating the shared budget.
- **REWARD_HACK policy**: quarantine — never selected, never merged, trace
  NOT fed to Reflect (evasion must never be learned); bounce to Plan with a
  neutral note; journal loudly.
- `StubExecutor` (now): `solver.check`, then synthesized latencies —
  deterministic pseudo-random factor keyed on `Solution.hash()` (stub sees
  only the clean Solution), optional **noise term** (exercises ε-logic),
  scripted scenarios for tests.
- `GpuQueueExecutor` (later): compile stage (`build_ext.py` →
  `cpp_extension.load` → `.so`, cached) then eval subprocess
  (`eval_driver.py`); parse Trace JSONL → `EvalResult`.
  Details: [kb/solution-format.md](../kb/solution-format.md).

### Environment fingerprint & score calibration

- Every `EvalResult.env` records: GPU UUID/model, driver/CUDA versions,
  clock-lock state, harness version. **Scores are comparable only within a
  fingerprint.** On fingerprint change (new pod), the engine flags frontier
  scores stale and **re-baselines** (re-measures frontier members) before
  ε-comparisons resume.
- **Website calibration:** `T_b`/`T_SOL` were measured on NVIDIA's rig; our
  pod differs systematically (the old repo observed exactly this). On first
  contact with a pod, measure the seed + (optionally) a known baseline,
  journal a **calibration note** (our-latency vs website `T_b`), and report
  both raw and calibrated sol_scores. Search decisions use local relative
  numbers; headline scores are labeled calibrated-vs-website.

## 2. ProblemSolver — per-question GEPA loop

Holds a **Pareto frontier over the problem's ~16 workload shapes**: every
candidate that is best on **at least one shape** survives (not top-k by
aggregate) — specialists are preserved for Merge.

**Frontier rule (Accept contract):** vector = `sol_score` per shape
(non-PASSED shape = 0). **Admission requires a full eval** (never screens).
A **ε-dominates** B iff A ≥ B − ε_s on every shape s and A > B + ε_s on at
least one — **ε_s is per-shape and empirical**, derived from measured
dispersion (`latency_spread`), not a global constant (small shapes are
µs-scale and noisy; large shapes are stable). **Confirm-before-promote:**
near-ties (all wins within ε_s) are re-measured rather than churning the
frontier. Select samples a parent (weighted by shapes won) and returns it
**with its own attached reflection** — reflections never float to a
different parent.

```
seed  = scaffold(id) (reference wrapper)  ∪  transferred sibling templates
iter 0: plan consumes/produces the DESIGN DOC (design-kernel skill)
loop until budgets or plateau(K) or circuit-breaker or optional target:
    parent, parent_reflection = select_from_frontier()
    context   = retrieve_knowledge(id)
    candidate = plan(parent, parent_reflection, context)     # mutation policy applies
    if not check(candidate):   journal(reject);  continue    # iteration counts
    if not novel(candidate):   journal(dup);     continue    # iteration counts
    result    = await execute(candidate, screen→full)        # compile ∥, run locked
    accept_if_eps_pareto(candidate, result)                  # full evals only
    candidate.reflection = reflect(candidate, result)        # tiered
    journal(candidate, result, reflection)
```

## 3. Orchestrator — the fleet

N ProblemSolvers as crash-isolated async tasks; GPU budget allocated by
**headroom** = 1 − best all-correct sol_score (round-robin/priority as
alternatives). While one candidate is on the GPU, every other problem keeps
planning/reflecting.

## 3b. Budgets (three, any one stops the run)

| Budget | Unit | Why |
|---|---|---|
| GPU work | **workload-runs + GPU-seconds** | eval-count undercounts: a TIMEOUT burns up to 600 s; a screen is ~5× cheaper than a full run |
| Agent credit | tokens / $ (subscription non-interactive pool) | a chatty loop can exhaust the monthly pool without touching the GPU |
| Iterations | count (incl. rejects and bounces) | guarantees termination even if nothing reaches the GPU |

`RunContext` tracks all three; every journal entry records budget deltas.

## 4. Knowledge base — two layers

- **Static** (`kb/`): B200/kernel priors + grader + solution-format specs.
- **Dynamic** (`knowledge/`): per-problem journal (raw history) →
  **per-family learnings** (`knowledge/families/<family>.md` — the
  transferable unit) → global learnings. A **serialized curator** (single
  queue for the fleet) distills each finished problem into family + global
  files (merge/rewrite, bounded size) — no concurrent clobbering.

## 5. Knowledge transfer

Cheapest first: (1) **sibling templating** — same family + same op,
different axes → seed the frontier from the sibling's best; (2) family
learnings injected at plan time; (3) **design doc + KB slices** — iteration
0 runs the design-kernel recipe (or ingests `designs/<name>.md`): op graph,
per-shape roofline, ranked approaches drive the first real candidates;
(4) cross-problem technique seeding; (5) curation feedback — solve one
exemplar per family first, then its cheap siblings.

## 6. Merge & the deliverable (all-correct invariant)

**Invariant:** `best_solution.json` must be **PASSED on every workload** —
the argmax of mean sol_score **among all-correct candidates** (the seed
guarantees one exists). The frontier may hold partially-correct specialists
as genetic material, but they can't be the deliverable unless merged into
something all-correct.

**Merge node:** per-shape dispatch (branch on runtime shape inside `run`).
Same-language-family → mechanical merge (dispatch wrapper + both kernels in
one Solution). **Cross-family is impossible** (harness forbids mixing Python
and C++ languages in one Solution) → degrade to a Plan-rewrite ("port the
losing side's technique"). Merged candidates go through
novelty→execute→accept like any other. Run Merge before finalization and
opportunistically on plateau.

## 7. Termination (per problem)

Maximize `sol_score`; stop on budgets, plateau (K iterations of any kind
without ε-improvement), circuit breaker, or optional target (off by
default). Then Merge → finalize `best_solution.json` → Curate.

## 8. Durability & resume

**The journal is the single source of truth; everything else is a
rebuildable cache.** Kill at any instant; restart replays and continues,
never re-paying completed GPU or agent work.

### Two caching mechanisms — do not conflate

1. **Crash-replay cache** (`runs/<id>/cache/`, key =
   `hash(node_role, impl_version, inputs, attempt_nonce)`): exact,
   mechanical, LLM-free. Generative nodes (Plan/Reflect) include an
   **attempt nonce** — the cache replays a specific past attempt, never
   makes a fresh attempt return a stale result (kills plateau-deadlock).
2. **Novelty gate**: semantic dedup of candidates before the GPU — hashes
   first, LLM judge last. The crash cache never decides "is the code the
   same"; the gate never replaces crash replay.

### What's persisted (per problem, `runs/<id>/`)

- **`journal.jsonl`** — append-only authority; one fsync'd line per
  transition. **Every line carries a `schema_version`**, and state-changing
  entries (e.g. `frontier_update`) record the **full delta** (who entered,
  who dropped, with scores) so replay is deterministic even across engine
  versions — replay applies recorded deltas, it does not recompute them.
- **`cache/`** — crash-replay cache (gitignored; regenerable).
- **`frontier/`, `best_solution.json`, `status.json`, `candidates/`** —
  derived, journal-rebuildable snapshots. `candidates/<cand_id>/` holds the
  real source files (`kernel.py`/`kernel.cu`/…), clean `solution.json`,
  `result.json` (per-shape statuses/stats), `meta.json` (parent, technique,
  status: frontier | dominated | rejected | duplicate | incorrect |
  compile_error | runtime_error | timeout | reward_hack).

**Git policy:** run outputs are git-visible; only regenerable churn is
ignored (`.cache/`, `runs/*/cache/`).

### The resumable driver

```
restart:  state = replay(journal); node = state.next or graph.start
loop:
  key = hash(node, impl_version, inputs [, attempt_nonce if generative])
  if cache.has(key): output = cache.get(key)                 # crash replay → reuse
  else:
    append(journal, node_start{node, key})                   # write-ahead intent
    output = await node.run(ctx, inputs)
    cache.put(key, output); append(journal, node_done{key})  # + fsync
  ctx.apply(output)                                           # journaled deltas
  node, inputs = router.next(node, output, ctx); append(journal, route{node})
```

### The GPU node — reconciled by job-id

Durable queue with stable job-ids (pending→processing→completed):
journal `execute_submitted{job_id, solution_hash}` before waiting; the
GPU-side worker persists results independently of the engine. Restart:
completed → recover; processing → wait; lost → resubmit. `Solution.hash()`
is the dedup/build-cache key throughout.

### The fleet

Each `runs/<id>/` is self-contained; the Orchestrator rescans run dirs on
restart. Allocation state is a small journaled file.

## 9. Security (LLM-generated code on our own hardware)

- **Sandbox the worker.** On our GPU box the harness executes LLM-generated
  code with worker privileges. The eval worker runs candidates in a
  least-privilege container: no network, workdir-only filesystem, resource
  limits (the hosted bench sandboxes for NVIDIA; on our pod that's our job).
  Phase-F requirement, designed now.
- **Web research is an injection chain** (web content → code-writing agent →
  code on our GPU). The Research node's web option is **off by default**;
  when enabled: allowlisted domains, and web-derived text enters prompts
  quoted/labelled as untrusted data, never as instructions.
- Validation code and candidate code remain separate trust domains
  (kb/benchmark-grader.md posture); the agent never modifies harness or
  engine files.

## 10. Observability & steering

- **Browse on disk:** `runs/<id>/candidates/*/` — real source files, status
  in `meta.json`.
- **CLI views** (read the journal): `solver status [ids]`, `solver journal
  <id>` (timeline incl. bounces + reflections), `solver frontier <id>` (who
  survives, which shapes each wins), `solver candidates <id> [--status …]`.
- **Human-in-the-loop:** per-problem `runs/<id>/hints.md` — read by the Plan
  node each iteration (steer the search by editing a file); a `pause` flag
  honored by the router. Both journaled.
- **Report (later):** self-contained HTML dashboard as an Artifact.

## 11. Engine acceptance tests (Phase B deliverable, stub-powered)

The loop's invariants get an explicit test matrix before any agent/GPU spend:

1. **Kill/resume equivalence** — kill after *every* journal line; resumed run
   reaches a state identical to an uninterrupted run (same frontier, budgets,
   journal suffix).
2. **Budget exactness** — rejects, bounces, screens, TIMEOUTs account
   correctly in all three budgets.
3. **Frontier correctness** — ε-Pareto set matches brute-force reference on
   synthetic score sets (incl. noise churn scenarios).
4. **Plateau termination** — with the attempt nonce, a plateau terminates by
   iteration budget; no spin.
5. **Quarantine** — REWARD_HACK candidates never re-enter selection/merge;
   their traces never reach Reflect.
6. **Screen isolation** — screened-only candidates never appear in the
   frontier.
7. **Fingerprint change** — frontier flagged stale; re-baseline path runs.

## 12. Build order (laptop-first)

| Phase | Buildable now? | Piece |
|---|---|---|
| A | ✅ done (revise) | Execute role + StubExecutor → **convert to async**, per-workload status, screen mode, stub speed/noise model |
| B | ✅ | Async driver + router + journal (versioned, delta-carrying) + replay + nonce cache + Select/Plan/Check/Novelty/Accept(ε, full-only)/Reflect(tiered)/Route + **test matrix §11** |
| C | ✅ | Research/Curate nodes + knowledge store (serialized curator) + hints.md |
| D | ✅ | Orchestrator fleet (crash-isolated) + triple budgets + CLI views |
| E | ✅ | Transfer (sibling templating) + Merge node (same-family) + mutation policy |
| F | ⛔ later | GpuQueueExecutor: compile-∥-run split, priority queue, circuit breaker, sandbox, env fingerprint + calibration |
| — | ⛔ later | Submit node |
| G | ⛔ later | Nsight → ASI profiling |

## Decisions (updated 2026-07-05)

1. **Agent substrate** → Claude Agent SDK (Python) on a subscription OAuth
   token behind a swappable `Agent` interface; agent credit is a tracked
   budget; Novelty judge + brief reflections use a cheap model.
2. **Search driver** → custom lightweight loop (GEPA concepts: reflection,
   ε-Pareto, merge). Supersedes the gepa-library note in orchestration.md.
3. **Termination** → maximize sol_score; budgets/plateau/circuit-breaker;
   optional target off by default. No submission.
4. **Dedup** → crash-replay caching is exact/mechanical; candidate novelty is
   semantic (hash → normalized hash → LLM judge).
5. **Execution model** → async throughout; GPU lock is awaitable; compile
   never holds the GPU lock.
6. **Measurement honesty** → per-shape empirical ε; full-eval-only frontier;
   environment fingerprints; website-calibration notes on reported scores.
