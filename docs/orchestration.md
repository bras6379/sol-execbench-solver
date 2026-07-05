# Orchestration: the Solver Engine

Single design doc for the multi-agent optimization engine. **Status: design;
Phase A (Executor stub) built.** Goal: **select problems → optimize each to
the best kernel we can find → accumulate and transfer knowledge across
problems.** The GPU is abstracted behind an interface (stub today, real
harness transport later). The engine *finds* best solutions; it does **not**
submit to the website (a submit step can be added at the end of the loop
later — it's one function call).

Design stance: **solid but not over-engineered.** The v1 core below is the
minimum that is correct, resumable, and observable; everything else lives in
[Deferred](#deferred-designed-not-built) with its trigger condition.

---

## 1. Concept: a GEPA loop with a GPU-locked evaluator

GEPA (Genetic-Pareto reflective text evolution; arXiv 2507.19457, ICLR 2026
oral) evolves a *text artifact* — for us, a kernel **Solution** — through:
reflect on execution traces (the "text gradient") → mutate → evaluate →
keep a **Pareto frontier per instance** (per workload shape, *not* top-k by
aggregate) → merge specialists. It reaches strong results in 100–500
evaluations instead of RL's tens of thousands — the reason it fits a world
where every evaluation is a scarce, serialized GPU run. (GRPO-style RL would
collapse each rich trace — per-shape statuses, matched-ratio, logs — into a
scalar and need thousands of runs to learn from it; GEPA turns each trace
into one targeted fix. We evolve the kernel, not a model.)

| GEPA concept | Here |
|---|---|
| Candidate | a Solution (`solver/solution.py`) + lineage (parent id) |
| evaluate() | run in the SOL-ExecBench harness on the GPU → per-workload Traces |
| Reflection / ASI | diagnosis from statuses, matched_ratio, latency-vs-SOL, logs |
| Pareto frontier | per-workload-shape non-domination (specialists survive) |
| Merge | per-shape dispatch kernel from specialists (deferred until data shows specialist-rich frontiers) |

## 2. Architecture (v1): a fleet of async loop functions

**Many problems run at the same time.** Concurrency comes from the async
runtime, not from a framework: each problem is one asyncio task running a
plain loop function; the only shared, serialized resource is the GPU.

```python
async def solve_problem(task_id, executor, agent, knowledge):
    ctx = RunContext.load(task_id)                    # journal replay → resume
    # ---- bootstrap (consumes GPU evals from the budget) ----
    if ctx.fresh():
        ctx.design = await agent.design(task_id)      # design-kernel recipe (or load designs/<name>.md)
        for seed in [scaffold(task_id), *sibling_templates(task_id, knowledge)]:
            result = await executor.evaluate(seed, task_id)
            ctx.accept(seed, result)                  # frontier now non-empty; seed eval doubles
                                                      # as the §10 calibration probe (local vs T_b)
    # ---- the loop ----
    while not ctx.should_stop():                      # caps / plateau / target
        parent = ctx.frontier.select()                # Pareto-weighted, with its reflection
        cand   = await agent.plan(parent, ctx)        # concurrent across problems
        if not check(cand):        ctx.journal("reject", cand); continue
        if not await novel(cand):  ctx.journal("dup", cand);    continue
        result  = await executor.evaluate(cand, task_id)  # ← THE GPU LOCK
        verdict = ctx.accept(cand, result)            # ε-Pareto, journaled delta
        cand.reflection = await agent.reflect(cand, result, verdict)  # verdict drives tiering
        ctx.journal("iter", cand, result)
    finalize(ctx)                                     # best all-correct → best_solution.json
    await knowledge.curate(ctx)                       # serialized curator

async def solve_family(chain, executor, agent, knowledge):
    for task_id in chain:              # exemplar first, then its siblings, in order
        await guarded(solve_problem(task_id, executor, agent, knowledge))

async def main(ids):
    executor  = make_executor(cfg)     # one shared instance; awaitable lock inside
    agent     = make_agent(cfg)        # Claude Agent SDK (or stub); concurrent sessions
    knowledge = KnowledgeStore()       # curator serialized internally
    chains    = chains_by_family(ids)  # one ordered chain per family
    await asyncio.gather(*(solve_family(c, executor, agent, knowledge)
                           for c in chains))   # families concurrent; NO cross-family barrier
```

**Work-conserving family chains.** There are **no global rounds**: all
families run concurrently, always. Within a family, chain order (exemplar →
siblings) is a **priority, not a gate** — it exists purely for GPU-budget
economics: a sibling started after its exemplar tunes a template (~3–5
evals) instead of searching from scratch (~20–50), and since the GPU is
single-flight, redundant sibling exploration burns shared budget without
finishing anything sooner. But the chain never holds work back while
capacity is free: **if the GPU would otherwise idle** (few families
selected, everyone mid-think), the scheduler starts the next pending problem
early. Late starters bootstrap from whatever knowledge exists at start time
and **live-read running siblings' frontier bests** mid-run, so any
same-family overlap still transfers continuously. Net behavior: everything
runs whenever there's capacity; ordering only shapes who gets started first.

**Agent failure policy.** Transient agent errors → retry with backoff.
**Quota-exhausted (subscription credit pool dry) is fleet-wide**, not a
per-problem crash: the fleet suspends cleanly (`suspended: credit exhausted`
journaled once), and `solver solve` resumes it later — never twenty
`solver_error`s from one shared cause.

**Agent swapping across sessions.** The agent is a per-session construct;
run state is model-agnostic (Solutions, reflections, scores). So a run can
be stopped and **resumed with a different agent/model** — e.g. grind with a
cheap model, resume the hard tail with an expensive one, or stop an
expensive run and continue cheaper. Mechanics: every generative journal
entry records the **agent identity** (model/config); each candidate's
`meta.json` (and its Solution `author` field — deliberately outside
`Solution.hash()`) records the producing model, so wins and credit spend are
attributable per model; on resume with a different agent an `agent_changed`
event is journaled and the **plateau counter resets** (a no-improvement
streak under model A is not evidence about model B).

Rules that make this work:

- **Async throughout.** Agent calls and `evaluate` are awaited; blocking work
  goes to `asyncio.to_thread`; no sync lock ever sits on the event-loop path.
  (Phase A's `threading.Lock` stub gets converted.)
- **Two swappable interfaces** — the only abstraction that earns its keep:
  - `Executor.evaluate(solution, task_id) -> EvalResult` — StubExecutor (now)
    / GpuQueueExecutor (later). All solvers share ONE instance; strictly one
    job on the GPU at a time, dispatched **fair round-robin across
    problems**: per-problem FIFO queues, one job per problem per turn. This
    prevents a fast-thinking problem (or a multi-seed bootstrap) from
    monopolizing the GPU while other problems' evals wait — each solver has
    ≤1 eval in flight, so RR over problems is exact fairness. (Phase F's
    priority refinements — screens first, circuit breaker — extend this same
    dispatcher.)
  - `Agent.design/plan/reflect/judge(...)` — StubAgent (deterministic tests)
    / Claude Agent SDK on the subscription OAuth token (`claude setup-token`
    → `CLAUDE_CODE_OAUTH_TOKEN`; no API key). Judge + brief reflections use
    a cheap model.
- **Crash isolation:** one solver failing is journaled and stops only that
  problem.
- "Nodes" (select/plan/check/novelty/execute/accept/reflect) remain the
  *vocabulary* — steps of the loop with typed inputs/outputs — but there is
  **no driver/router/graph framework**; the loop is a function we edit.

## 3. Candidates

- A candidate is a harness **Solution**: any of the 9 languages (python
  family: pytorch/triton/cute_dsl/cutile/cudnn_frontend; C++ family:
  cuda_cpp/cutlass/cudnn/cublas — families can't mix), multi-file `sources`,
  own `compile_options`/`dependencies`, DPS optional. See
  [kb/solution-format.md](../kb/solution-format.md).
- **Clean-artifact invariant:** `solution.json` contains only the kernel.
  Engine bookkeeping (status, parent, technique, scores, reflection) lives
  beside it (`meta.json`/`result.json`/journal), never inside.
- **Seed, not mold:** the seed is `scaffold(id)` — the correct PyTorch DPS
  reference wrapper (its measured score may be < 0.5; `T_b` is an *optimized*
  baseline). Generated candidates are free-form; no C++ scaffold needed.
- **Mutation policy lives in the Plan prompt**, not engine machinery: follow
  the KB abstraction ladder (torch → Triton → CuTe-DSL/CUTLASS → C++/PTX),
  escalate only on plateau/evidence/design-doc recommendation; context per
  call = parent source + parent's own reflection + top-K insights + relevant
  design-doc section (never full lineage history).

## 4. The gates before the GPU

1. **Check** (static, free): schema + per-language entry validation (DPS
   names for `.py`; `void run(torch::Tensor…)` shape for C++). Built.
2. **Novelty** (semantic dedup — the GPU's last gate), two tiers with
   distinct scopes:
   - exact `Solution.hash()` (the harness's own SHA1) checked against **the
     full journal history** — free set-lookup; a candidate identical to
     *anything ever evaluated* (including long-dominated ones) never re-pays
     a GPU run;
   - **LLM judge** (cheap model) against **parent + current frontier**:
     materially different implementation (algorithm/layout/fusion/precision/
     launch config) vs cosmetic variant? A judge call costs orders of
     magnitude less than the GPU run it protects. `cosmetic` bounces to Plan
     *with the verdict as feedback*.
3. Every pass through Plan — including rejects and bounces — **counts as an
   iteration**, so termination caps always fire (no spin).

## 5. Evaluation results

```python
@dataclass
class WorkloadResult:
    index: int
    status: str            # per-workload Trace enum: PASSED | INCORRECT_SHAPE |
                           # INCORRECT_NUMERICAL | INCORRECT_DTYPE | RUNTIME_ERROR |
                           # TIMEOUT | REWARD_HACK | INVALID_REFERENCE
    latency_ms: float | None
    latency_spread: float | None      # recorded now; adaptive ε later
    sol_ms: float | None
    baseline_latency_ms: float | None
    matched_ratio: float | None

@dataclass
class EvalResult:
    solution_status: str | None       # solution-level only: COMPILE_ERROR | REWARD_HACK | None
    per_workload: list[WorkloadResult]
    all_passed: bool
    sol_score: float | None           # mean over shapes; None unless all_passed
    env: dict                         # fingerprint: GPU/driver/clock/harness (recorded now)
    asi: dict                         # logs + notes for reflection
```

- **Status is per-workload** (one harness Trace per shape — a kernel can PASS
  14 and TIMEOUT 2; that partial-specialist signal is what the frontier
  needs). Only `COMPILE_ERROR` / `REWARD_HACK` are solution-level.
- **REWARD_HACK → quarantine:** never selected or merged; its trace is *not*
  fed to reflection (evasion must never be learned); Plan gets a neutral
  "disallowed pattern, regenerate" note; journaled loudly.
- **StubExecutor** synthesizes latencies deterministically keyed on
  `Solution.hash()` (it sees only the clean Solution) + optional noise term +
  scripted scenarios — enough to test the loop, frontier, and resume.

## 6. Frontier, budgets, termination

- **ε-Pareto frontier over the ~16 shapes.** Vector = per-shape `sol_score`
  (non-PASSED shape = 0). A ε-dominates B iff A ≥ B−ε everywhere and
  A > B+ε somewhere. v1: **one configurable relative ε (default ~2%)**;
  Select samples the frontier weighted by shapes won, and always returns the
  parent **with its own reflection** (reflections attach to lineage, never
  float).
- **Budgets: two enforced caps + one observed metric.**
  - `max_iterations` per problem (counts everything; guarantees termination);
  - `max_gpu_evals` per problem (full harness runs — the scarce resource;
    per-eval cost is already bounded by the harness timeout). Counted when a
    candidate **reaches the run stage**: check-fails, novelty bounces, and
    (Phase F) `COMPILE_ERROR`s never touched the GPU and don't consume it;
  - agent tokens/credit: **logged per call**, not enforced — the
    subscription's non-interactive pool is a hard stop on Anthropic's side;
    we watch the logs and add enforcement only if needed.
- **Termination:** caps, or plateau (K iterations of any kind without
  ε-improvement), or optional score target (off by default).
- **Deliverable invariant:** `best_solution.json` = argmax mean sol_score
  **among all-correct candidates** (PASSED on every shape). The seed
  guarantees one exists. Partially-correct specialists stay on the frontier
  as genetic material but can't be the deliverable.

## 7. Persistence & resume (journal + results store)

**The journal is the source of truth for *what happened*; an append-only
results store holds the *irreplaceable measurements*.** No node-output
cache: replay *is* resume. A crash mid-agent-call re-pays one cheap call;
GPU work is protected separately (below).

Storage rule — three classes, no ambiguity:

1. **Inline in the journal**: small artifacts — Solutions (few KB of text),
   check/novelty verdicts, reflections, frontier deltas, budget deltas.
2. **`runs/<id>/results/` — append-only, AUTHORITATIVE** (referenced from the
   journal by hash): raw harness Traces/logs. These are measurements — the
   one thing that **cannot be rebuilt without re-paying a GPU run** — so
   they are never classed as a derived view.
3. **Derived views, journal-rebuildable**: `frontier/`, `best_solution.json`,
   `status.json`, `candidates/`. `candidates/<cand_id>/` materializes the
   **real source files** (`kernel.py`/`kernel.cu`/…), the clean
   `solution.json`, `result.json` (per-shape statuses/scores), and
   `meta.json` (parent, technique, status: frontier | dominated | rejected |
   duplicate | incorrect | compile_error | runtime_error | timeout |
   reward_hack).

- `runs/<id>/journal.jsonl` — append-only, one fsync'd line per step; every
  line has a `schema_version`; state changes record **full deltas** (who
  entered/left the frontier, with scores) so replay is deterministic across
  engine versions. A crash truncates at most the trailing line (dropped on
  replay).
- **GPU work is never lost or re-paid:** the (Phase F) executor journals
  `execute_submitted{job_id, solution_hash}` before waiting; the GPU-side
  worker persists results durably by job-id (pending→processing→completed).
  Restart reconciles: completed → recover; processing → wait; lost →
  resubmit.
- **Git policy:** run outputs (journal, candidates, frontier, best_solution)
  are git-visible — check in to snapshot progress. Only regenerable churn is
  ignored (`.cache/`).
- Re-running `solver solve` on the same ids **resumes by default**. Pausing =
  kill the process; resume continues (no pause flag needed).

## 8. Knowledge & transfer

- **Static KB** (`kb/`): read-only priors — B200, kernels, grader,
  solution format.
- **Dynamic KB** (`knowledge/`): journal (raw, per problem) → **family
  learnings** (`knowledge/families/<family>.md` — the transferable unit) →
  `global.md`. A **serialized curator** (one queue for the fleet) distills a
  finished problem's journal into family/global files — merge/rewrite,
  bounded size, no concurrent clobbering.
- **Family mapping is concrete, not conceptual:** a checked-in
  `knowledge/families.json` maps every task id → family id, generated from
  definition-name/description keywords (the taxonomy in
  `kb/benchmark-problems.md`) and **human-overridable**. Sibling detection
  and `families/<family>.md` both key on it.
- **Transfer, cheapest first:** (1) **sibling templating** — same family +
  op, different axes (e.g. rmsnorm 230–235) → seed the new frontier from the
  sibling's best; (2) family learnings + current bests injected at plan
  time (including **live-reads of running siblings' frontiers**); (3)
  **design doc at bootstrap** — Plan consumes/produces the `design-kernel`
  output (op graph, per-shape roofline, ranked approaches); (4)
  cross-problem technique seeding; (5) curation feedback — enforced by the
  fleet's **exemplar-first waves** (§2).

## 9. Observability & steering

- **Disk is browsable:** real kernel sources + status per candidate under
  `runs/<id>/candidates/`.
- **CLI views** (read-only over journals): `solver status [ids]`
  (iterations, caps spent, best vs baseline & SOL, frontier size);
  `solver journal <id>` (timeline incl. bounces + reflections);
  `solver frontier <id>` (survivors + which shapes each wins);
  `solver candidates <id> [--status …]`.
- **Steering:** edit `runs/<id>/hints.md` — the Plan prompt includes it every
  iteration.

### Instrumentation (built now; the loop emits these from day one)

Every journal event carries `{v, ts, task, ev, …}` (ISO-8601 UTC, fsync'd).
The v1 event vocabulary — pinned here so Phase B implements against it:

`run_started{agent} · design_done{dur_s} · plan_done{cand, parent, model,
dur_s, tok_in, tok_out} · check{cand, ok} · novelty{cand, verdict} ·
exec_enqueued{job, cand} · exec_started{job} · exec_done{job, cand, gpu_s,
all_passed, sol_score, statuses} · accept{cand, verdict, best, frontier} ·
reflect_done{cand, tier, dur_s} · agent_changed{model} · terminated{reason}`

The executor lifecycle triple (`exec_enqueued/started/done`) is the
measurement backbone: queue wait = started−enqueued, GPU busy = done−started;
merging these across all journals yields the global GPU timeline and
utilization — no separate metrics file needed.

**`solver report`** renders a **self-contained static HTML dashboard** (no
server, no CDN; light/dark) from the journals: convergence per problem (best
sol_score vs GPU evals), GPU busy % + job timeline colored by problem, queue
wait percentiles, iteration-outcome mix (accepted/dominated/rejected/dup/…),
agent call/token spend, and a per-problem status table. Flags: `--out`,
`--runs-dir`, `--refresh N` (meta-refresh), `--watch N` (regenerate loop),
`--demo` (synthetic runs under `.cache/demo/` to preview the dashboard
without the engine).

## 10. Measurement honesty (v1 posture)

- Frontier admission requires a **full eval** (all shapes) — v1 has no
  partial evals at all, so this is trivially true (screen mode is deferred).
- `env` fingerprint (GPU/driver/clock/harness) is **recorded on every
  result** from day one — comparisons across different pods are flagged;
  automated re-baselining is deferred.
- Website calibration: on first contact with a real pod, measure the seed,
  journal local-vs-`T_b` discrepancy, report raw + calibrated scores
  (search decisions use local relative numbers).

## 11. Security posture

- Candidates are LLM-generated code. On our GPU box (Phase F) the eval
  worker runs them **sandboxed**: no network, workdir-only FS, resource
  limits. Designed now, built with the worker.
- The Research/web option is **off by default**; when on: allowlisted
  domains, and web text enters prompts as quoted untrusted data, never
  instructions (injection chain: web → code-writing agent → our GPU).
- Agents never modify the harness, the engine, or validation code.

## 12. Acceptance tests (v1, stub-powered)

1. **Kill/resume safety** — kill after every journal line. With
   **deterministic stubs**: the resumed run is *identical* to an
   uninterrupted one (bitwise state equivalence). With real agents resume is
   legitimately nondeterministic (an in-flight agent call re-runs and may
   differ), so the guarantee tested is **no-loss / no-double-pay**: every
   journaled GPU result and result-store trace is retained, budgets are
   exact, and no completed GPU eval is ever re-run.
2. **Budget exactness** — rejects, bounces, evals, timeouts all count
   correctly against both caps.
3. **Frontier correctness** — ε-Pareto set matches brute force on synthetic
   score sets, including noisy scenarios.
4. **Plateau termination** — plateaus terminate by cap; no spin (bounces
   count as iterations).

## 13. Build order

| Phase | Now? | Piece |
|---|---|---|
| A | ✅ done (revise) | Executor interface + stub → **async**, per-workload status, hash-keyed stub speed/noise model |
| B | ✅ | `solve_problem` loop + RunContext + journal/replay + ε-frontier + novelty (hash + judge) + caps/plateau + **tests §12** |
| C | ✅ | Knowledge store + serialized curator + hints.md + design-doc-at-iter-0 |
| D | ✅ | Fleet (`main`, crash isolation) + CLI (`solve`, status/journal/frontier/candidates views) |
| E | ✅ | Sibling templating transfer |
| F | ⛔ GPU | GpuQueueExecutor: job-id queue, compile-∥-run split, sandbox, calibration; then the deferred items below as data demands |

## Deferred (designed, not built)

Each with its trigger:

- **Screen mode** (subset-of-shapes smoke evals; GEPA minibatch analogue) —
  when the GPU bill shows full evals dominating; requires the
  full-eval-only-frontier rule already stated.
- **Per-shape empirical ε + confirm-before-promote** — when real
  measurements show per-shape noise profiles (spread field already
  recorded).
- **Merge node** (per-shape dispatch; same-family mechanical, cross-family =
  Plan-rewrite since the harness forbids mixing language families) — when
  real frontiers hold divergent specialists.
- **Priority queue + per-problem circuit breaker** — with the real GPU queue
  (Phase F).
- **Env re-baselining automation** — when running across multiple pods.
- **Agent-credit enforcement** — if usage logs show the pool at risk.
- **Reflection tiering beyond two levels; normalized-hash novelty tier** —
  if judge/reflect costs show up in logs.
- **Submit node** — if/when we decide to submit; one function call at the
  end of `finalize`.
- **Nsight → ASI profiling in the eval path** — Phase F+; profile on
  plateau, not every run.
- **HTML dashboard** — nice-to-have.

## Decision log

1. **Agents:** Claude Agent SDK on subscription OAuth token, behind the
   `Agent` interface; cheap model for judge/brief-reflections. Credit pool
   (Pro $20 / Max-5x $100 / Max-20x $200 monthly, non-interactive) is logged,
   not enforced.
2. **Search:** custom lightweight loop implementing GEPA's concepts
   (reflection, ε-Pareto, merge); no gepa-library dependency; **no graph
   framework** — concurrency comes from asyncio, extensibility from editing
   a 25-line loop.
3. **Termination:** maximize sol_score; caps + plateau; optional target off
   by default; no submission.
4. **Dedup:** crash-replay = journal replay (exact, mechanical); candidate
   novelty = semantic (hash → LLM judge).
5. **Execution model:** async throughout; GPU lock awaitable; compile (Phase
   F) never holds the GPU lock.
6. **Persistence:** journal (what happened, small artifacts inline) + an
   append-only authoritative results store (irreplaceable GPU traces); no
   node-output cache (an agent call re-paid after a crash costs cents; GPU
   work is protected by the job-id queue).
