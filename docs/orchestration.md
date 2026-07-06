# Orchestration: the Solver Engine

Single design doc for the multi-agent optimization engine. **Status: Phases
A + B built (engine core: loop, frontier, tiers, resume, §12 tests — all
laptop/stub); Phases C–F designed.** Goal: **select problems → optimize each to
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
| Candidate | a Solution (`solver/bench/solution.py`) + lineage (parent id) |
| evaluate() | run in the SOL-ExecBench harness on the GPU → per-workload Traces |
| Reflection / ASI | diagnosis from statuses, matched_ratio, latency-vs-SOL, logs |
| Pareto frontier | per-workload-shape non-domination (specialists survive) |
| Merge | per-shape dispatch kernel from specialists (deferred until data shows specialist-rich frontiers) |

## 2. Architecture (v1): a fleet of async loop functions

**Many problems run at the same time.** Concurrency comes from the async
runtime, not from a framework: each problem is one asyncio task running a
plain loop function; the only shared, serialized resource is the GPU.

```python
async def solve_problem(task_id, executor, agents, knowledge):
    ctx = RunContext.load(task_id)                    # journal replay → resume (incl. tier index)
    # ---- bootstrap (consumes GPU evals from the budget) ----
    if ctx.fresh():
        ctx.design = await agents[cfg.design_model].design(task_id)  # ONE strong design call (§6b), not round-robin
        for seed in [scaffold(task_id), *sibling_seed(task_id, knowledge)]:  # sibling's BEST, not its whole frontier
            result = await executor.evaluate(seed, task_id)
            ctx.accept(seed, result)                  # frontier now non-empty; seed eval doubles
                                                      # as the §10 calibration probe (local vs T_b)
    # ---- the loop ----
    while not ctx.done():                             # budgets/target, or a terminating plateau (§6)
        if ctx.tier_plateaued():                      # M full pool-cycles, no ε-gain (§6b)
            if not ctx.escalate():  break             # escalate iff headroom (best<ceiling) & a tier remains; else STOP
        agent  = agents[ctx.tier.next()]              # round-robin THIS tier's pool → per-iter diversity (§6b)
        parent = ctx.frontier.select()                # Pareto-weighted parent (full) + frontier capsules
        cand   = await agent.plan(parent, ctx)        # concurrent across problems
        if not check(cand):        ctx.journal("reject", cand); continue
        if not await novel(cand):  ctx.journal("dup", cand);    continue
        result  = await executor.evaluate(cand, task_id)  # ← THE GPU LOCK
        verdict = ctx.accept(cand, result)            # ε-Pareto, journaled delta
        cand.reflection = await agent.reflect(cand, result, verdict)  # reflection BY the planning agent (§6b); no tiering
        ctx.journal("iter", cand, result)
    finalize(ctx)                                     # best all-correct → best_solution.json
    await knowledge.curate(ctx)                       # serialized curator

async def main(ids):
    executor  = make_executor(cfg)     # one shared instance; awaitable lock inside
    agents    = make_agents(cfg)       # {(agent,model): Agent} — the tier pools (§6b); concurrent sessions
    knowledge = KnowledgeStore()       # curator serialized internally
    order     = exemplar_first(ids)    # static launch order: each family's exemplar ahead of its siblings
    await asyncio.gather(*(guarded(solve_problem(t, executor, agents, knowledge))
                           for t in order))    # ALL problems concurrent; GPU round-robin is the only serialization
```

**Flat concurrent fleet.** No rounds, no family gates: every problem is its
own asyncio task and they all run at once; the single-flight GPU (round-robin
across problems) is the only serialization. Cross-problem transfer rides two
cheap, best-effort channels: (1) **sibling seeding** — a problem seeds its
frontier from the sibling's **single best Solution** (not its whole frontier;
§8) that exists *at bootstrap time*, falling back to `scaffold(id)` if none
yet; (2) **curated
family/global learnings** injected at plan time (§8). To make (1) land more
often, problems launch in a **static exemplar-first order** (each family's
exemplar ahead of its siblings) — a soft head-start, not a gate: nothing is
ever held back and the GPU never idles waiting on ordering. That's the whole
scheduler. (Dynamic work-conserving rebalancing and **mid-run live-reads of a
running sibling's frontier** are Deferred: at 235-problems / 1-GPU the eval
queue is permanently backed up, so the idle-avoidance they buy never fires,
and the live-read's cross-task coupling isn't worth its marginal gain over
bootstrap templating + curated files.)

**Agent failure policy.** Transient agent errors → retry with backoff.
**Quota-exhausted (subscription credit pool dry) is fleet-wide**, not a
per-problem crash: the fleet suspends cleanly (`suspended: credit exhausted`
journaled once), and `solver solve` resumes it later — never twenty
`solver_error`s from one shared cause.

**Agent swapping across sessions.** The agent is a per-session construct;
run state is model-agnostic (Solutions, reflections, scores). So a run can
be stopped and **resumed with a different agent/model (or a whole different
ladder/pool config)** — e.g. grind with a cheap model, resume the hard tail
with an expensive one, or stop an expensive run and continue cheaper. Mechanics: every generative journal
entry records the **agent identity** (model/config); each candidate's
`meta.json` (and its Solution `author` field — deliberately outside
`Solution.hash()`) records the producing model, so wins and credit spend are
attributable per model; on resume with a different agent an `agent_changed`
event is journaled and the **plateau counter resets** (a no-improvement
streak under model A is not evidence about model B). **§6b (the tier ladder)
is the automatic, within-run form of this same swap** — same
model-agnostic state, same `agent_changed`/reset mechanics, but triggered by
a plateau mid-run instead of by you stopping and resuming.

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
  - `Agent.design/plan/reflect/judge(...)` — the interface is **framework- and
    provider-agnostic**: an `Agent` is any coding agent bound to a model.
    Impls: StubAgent (deterministic tests; scripting API in §12), Claude Agent
    SDK on the subscription OAuth token (`claude setup-token` → `CLAUDE_CODE_OAUTH_TOKEN`;
    no API key), and **one generic OpenAI-compatible backend** (any
    `base_url`/`model` — GPT / DeepSeek / GLM / Qwen / Kimi, native or via an
    aggregator). A `(agent, model)` pair is a **perspective**; a pool of them
    is a **tier**; the tier ladder is §6b. The **novelty judge** uses a cheap
    fixed model; **reflection is done by the planning agent** (so it matches
    the current tier — strong reflection once you've escalated).
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
  escalating language only on plateau/evidence. The exact context a `plan()`
  call receives is the **§8 context-assembly** block (parent source + its
  reflection + frontier capsules + top-K insights + design-doc section —
  never full lineage history).

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
  scripted scenarios — enough to test the loop, frontier, and resume. Its full
  scenario API + re-entrancy assertion are the **§12 stub contract**.

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
- **Termination:** caps, or optional score target (off by default), or a
  plateau (no ε-improvement over the plateau window — measured in **full pool-
  cycles** for a pooled tier, §6b) that **either is on the last tier, or is at
  a score with no headroom left** (`best ≥ escalate_ceiling`). Otherwise the
  plateau *escalates* to the next tier (§6b) rather than terminating — so a
  near-optimal problem stops instead of climbing to the expensive tier.
- **Deliverable invariant:** `best_solution.json` = argmax mean sol_score
  **among all-correct candidates** (PASSED on every shape). The seed
  guarantees one exists. Partially-correct specialists stay on the frontier
  as genetic material but can't be the deliverable.

## 6b. Tier ladder — diverse agents/models to unblock plateaus

A single `(agent, model)` has one fixed prior; once it plateaus, *more of its
own mutations* rarely help. So diversity is built in at **two levels**: within
a tier several different priors are *always* contributing, and across tiers a
plateau escalates to a more capable pool. Cheap-first, escalate-when-stuck:
easy problems finish in the cheap tier and never touch the expensive one; only
the stubborn tail climbs.

- **A perspective = `(agent, model)`; a tier = an ordered *pool* of them; the
  ladder = an ordered list of tiers** (cheap/fast → strong/diverse). A
  perspective is any coding agent bound to any model behind the §2 `Agent`
  interface, and a pool mixes model **families** on purpose (Anthropic,
  OpenAI, DeepSeek, Zhipu-GLM, Moonshot-Kimi, Qwen) — different training
  corpora surface different kernel tricks.
- **Within a tier: round-robin the pool** (deterministic, by iteration index →
  replay-safe). Consecutive candidates for the same problem come from
  different models/agents, all feeding the *same* per-shape frontier — so
  **diversity is continuous**, not just something that happens at escalation.
  `plan_done{agent, model}` tags each candidate, so the dashboard shows exactly
  which model produced which win, per family.
- **Escalate on a *tier* plateau — but only while headroom remains.** The
  plateau window is **M full round-robin cycles** (one candidate per pool
  member) with no ε-gain, so it scales with pool size and every model gets a
  fair shot before the tier is judged stuck. On a tier plateau: if
  `best < escalate_ceiling` (headroom to SOL remains) **and** a tier is left →
  advance to the next tier; otherwise **terminate** (near the ceiling a
  different prior won't help — *this* is what keeps easy problems on the cheap
  tier). The new tier inherits the **full frontier + journal** (model-agnostic
  state, §2), so it **continues, it does not restart**. An `agent_changed{tier,
  trigger:"escalation"}` event is journaled and the **plateau counter resets**
  (a stall under tier A is not evidence about tier B). Within-tier rotation is
  *not* a reset; only tier boundaries are.
- **Terminate only when the *last* tier plateaus** (or a cap/target trips, §6).
  Effort deepens only as far as a problem actually needs.
- **Ratchet up (v1).** Once escalated, a problem stays in the higher tier for
  the rest of its run. (Drop-back-to-cheap and an evidence trigger that
  escalates immediately on repeated `COMPILE_ERROR`/`INCORRECT` are Deferred
  variants — v1 is pure tier-plateau escalation.)
- **Orthogonal to the abstraction ladder.** This ladder is over *who writes the
  kernel* (agent/model). The §3 abstraction ladder (torch → Triton →
  CuTe/CUTLASS → C++/PTX) is over *what language the kernel is in*. They
  compose independently.

**Escalation knobs (two config values).** They answer two different questions
at a plateau — *when is a tier stuck?* and *once stuck, climb or stop?*

- **`M` — patience** (default `2`): the plateau window in **round-robin
  cycles**, where one cycle = one candidate per pool member. A tier is *stuck*
  after `M` consecutive cycles with no ε-gain. Counted in cycles (not raw
  iterations) so every model gets a fair shot first; bigger `M` = more
  thorough/costly, smaller = quit a tier sooner. A single-model tier degenerates
  to the classic "`M` iterations without improvement."
- **`escalate_ceiling` — ambition** (default `0.9`): the "good enough" line on
  `sol_score` (0.5 = baseline `T_b`, 1.0 = SOL; 0.9 ≈ 89% of the baseline→SOL
  gap closed). At a plateau, `best < ceiling` (headroom left) → **climb**;
  `best ≥ ceiling` → **stop**. `= 1.0` means "nothing is ever good enough" (climb
  the whole ladder); lower → only the badly-stuck escalate.

At every plateau: stuck after `M` cycles → if `best ≥ escalate_ceiling` **or**
no stronger tier remains, terminate; else escalate to the next tier (inherit
frontier, reset the counter). Both are best tuned against real convergence
curves once the GPU is live.

**Model strategy (all config).** Two `Agent` backends cover everything: the
**Claude Agent SDK** (native, subscription) for Claude models, and one
**generic OpenAI-compatible backend** (any `base_url`/`model` — GPT / DeepSeek
/ GLM / Qwen / Kimi, native endpoints or an aggregator like OpenRouter) for the
rest. **v1 ships a Claude-only pool** (cheap tier `haiku`, strong tier `opus`);
adding cross-family diversity is a config entry + a key, no architecture change.
Illustrative pools (verify model IDs at run time — the landscape churns
monthly): cheap = `{claude-haiku, deepseek-v4-flash, glm-5.2, qwen3.6}`,
strong = `{claude-opus, gpt-5.5, deepseek-v4-pro, kimi-k2.6}`.

**Cost note:** metered non-Claude providers spend real per-token money (unlike
the Claude subscription's non-interactive pool, which Anthropic hard-stops), so
per-provider spend is logged and **credit enforcement (Deferred) moves up in
priority the moment a pool includes a metered backend.**

The ladder is **config** (`tiers: [{name, pool:[{agent, model}, …]}, …]`, plus
`escalate_ceiling` and a `design_model`); the default is a single Claude tier
(escalation off) — the mechanism adds nothing to a plain run until you
configure a second tier or a bigger pool. The one-shot bootstrap `design()`
call uses `design_model` (default: the strongest configured), decoupled from
the round-robin: it sets the whole search trajectory, so it's worth a strong
model even on an otherwise-cheap run.

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
3. **Journal-rebuildable, but materialized eagerly + checked in**:
   `frontier/`, `best_solution.json`, `status.json`, and — **first-class in
   practice** — `candidates/<cand_id>/`, the durable, browsable home for each
   (expensive-to-produce) kernel **and** its measured perf:
   - the **real source files** (`kernel.py`/`kernel.cu`/… — the multi-file
     `sources` as written);
   - `solution.json` — the clean Solution (+ its `Solution.hash()`);
   - `result.json` — **the GPU/metrics perf after it runs**: per-shape
     `status` / `latency_ms` / `sol_ms` / `baseline_latency_ms` /
     `matched_ratio` / `latency_spread`; solution-level `all_passed` +
     `sol_score`; the `gpu_s` wall time; and the `env` fingerprint (GPU/driver/
     clock/harness) — so a stored kernel carries exactly how fast it ran and
     against what;
   - `meta.json` — parent lineage, strategy TL;DR, producing `(agent, model)`,
     and status (frontier | dominated | rejected | duplicate | incorrect |
     compile_error | runtime_error | timeout | reward_hack).

   Rebuildable *in principle* (source is also inline-authoritative in the
   journal, raw traces in `results/`), so "replay is resume" still holds — but
   written **eagerly and git-checked-in** so an expensive kernel + its numbers
   is never lost and is always laid out on disk to inspect and reuse.

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
  `global.md`. A **globally-serialized curator** (one queue for the whole
  fleet) runs **once per *finished problem***, distilling that journal into an
  atomic read-merge-write of its family file + `global.md` (bounded size).
  Because it's a single queue, the shared family file is **never concurrently
  written — no per-family lock needed**; incremental per-problem curation means
  an early-finishing sibling immediately enriches the file for later ones.
  (Per-family parallel curation is Deferred — unneeded while curation is cheap
  vs. a GPU run.)
- **Family mapping is concrete, not conceptual:** a checked-in
  `knowledge/families.json` maps every task id → family id, generated from
  definition-name/description keywords (the taxonomy in
  `kb/benchmark-problems.md`) and **human-overridable**. Sibling detection
  and `families/<family>.md` both key on it.
- **Transfer, cheapest first:** (1) **sibling seeding at bootstrap** — same
  family + op, different axes (e.g. rmsnorm 230–235) → seed the new frontier
  from the sibling's **single best all-correct Solution** (not its whole
  specialist frontier: differing shapes make importing specialists speculative,
  and each import costs a bootstrap eval — multi-seed import is a config knob,
  default off); (2) family learnings + latest curated bests injected at plan
  time; (3) **design doc at bootstrap** — Plan consumes/produces the
  `design-kernel` output (op graph, per-shape roofline, ranked approaches);
  (4) cross-problem technique seeding; (5) curation feedback — biased by the
  fleet's **static exemplar-first launch order** (§2), so an exemplar usually
  has results before its siblings bootstrap.

**Context assembly (what a `plan()` call actually receives).** The agent never
sees raw journals or full lineage — just a **bounded, curated context**:

```
                 ┌──────────── PLAN() context (bounded & curated) ────────────┐
 STATIC  kb/ ───►│ • relevant KB slice — B200 tricks for THIS op class        │
 DESIGN  design ►│ • design-doc section — op graph, per-shape roofline, ideas  │
 THIS RUN ──────►│ • parent Solution (full source, the code to improve)        │──► agent
 (warm)          │ • parent's reflection ("text gradient": what to fix next)   │    writes a
                 │ • frontier capsules — every survivor: shapes-won, score,    │    candidate
                 │   one-line descriptor (only the selected parent is in full) │
 PAST RUNS ─────►│ • top-K distilled insights — knowledge/families/<fam>.md    │
 (cold/curated)  │   + global.md  (curator-COMPRESSED, never raw journals)     │
                 └──────────────────────────────────────────────────────────────┘
   NEVER injected: full lineage history · raw journals · other problems' traces
```

Both "pasts" are compressed: **this problem's own** past → parent + its
reflection + frontier capsules (bounded, not full lineage); **other problems'**
past → the curator's family/global markdown, top-K relevant slice only. The
whole frontier is *summarized* (capsules) so coverage gaps are visible without
every survivor's full source bloating the prompt. There is **no human/`hints`
layer — the system runs autonomously** (§9).

## 9. Observability

- **Disk is browsable:** real kernel sources + status per candidate under
  `runs/<id>/candidates/`.
- **CLI views** (read-only over journals): `solver status [ids]`
  (iterations, caps spent, best vs baseline & SOL, frontier size);
  `solver journal <id>` (timeline incl. bounces + reflections);
  `solver frontier <id>` (survivors + which shapes each wins);
  `solver candidates <id> [--status …]`.
- **Autonomous — no in-loop human steering.** These views are read-only; the
  system runs unattended (no `hints.md`). Behavior is changed by **config
  between runs** (tiers/pools, budgets, ε), never by editing a live run.

### Instrumentation (built now; the loop emits these from day one)

Every journal event carries `{v, ts, task, ev, …}` (ISO-8601 UTC, fsync'd).
The v1 event vocabulary — pinned here so Phase B implements against it:

`run_started{agent} · design_done{dur_s} · plan_done{cand, parent, agent,
model, dur_s, tok_in, tok_out, strategy, solution} · check{cand, ok} ·
novelty{cand, verdict} · exec_enqueued{job, cand} · exec_started{job} ·
exec_done{job, cand, gpu_s, all_passed, sol_score, statuses} ·
accept{cand, verdict, best, frontier} · reflect_done{cand, dur_s} ·
agent_changed{tier, agent, model, trigger} · terminated{reason}`

`plan_done{agent, model}` tags every candidate with the perspective that made
it (it rotates within a tier's pool). `agent_changed` fires only at a **tier
boundary** — a within-run tier escalation (`trigger:"escalation"`) or a
cross-session swap (`trigger:"resume"`) — and *that* is what resets the plateau
counter, so the dashboard can show which tier produced which candidate and
which one broke a plateau (§6b).

`plan_done.strategy` is a one-line TL;DR of the approach and
`plan_done.solution` is the full (clean) Solution inline — together they make
the dashboard's per-problem "solution progression" and per-candidate code
pages renderable from the journal alone.

The executor lifecycle triple (`exec_enqueued/started/done`) is the
measurement backbone: queue wait = started−enqueued, GPU busy = done−started;
merging these across all journals yields the global GPU timeline and
utilization — no separate metrics file needed.

**`solver report`** renders a **publishable static site into `out/`** (no
server, no CDN; light/dark): `index.html` = fleet hub (SOL-score tiles,
fleet-score-over-time, top-movers convergence, GPU occupancy per rental
window, score histogram, family rollup, sortable/filterable problems table —
scales to all 235 problems by showing distributions + top-K, never 235
lines), `p/<task>.html` = per-problem deep-dive (convergence + **solution
progression**: every candidate in order with strategy TL;DR, status, score),
`p/<task>/<cand>.html` = the candidate's **code page**. Flags: `--out-dir`,
`--runs-dir`, `--refresh N`, `--watch N`, `--demo` (synthetic runs under
`.cache/demo/`).

**GPU rentals:** `<runs>/gpu_rentals.jsonl` (one `{start, end, label}` per
line; hand-written now, executor-written later) scopes GPU-utilization to
**rented time only** and draws the occupancy timeline per rental window with
un-rented gaps compressed.

**Filtering:** the hub is rendered client-side from an embedded per-problem
data blob, so a filter bar (family chips + a task/name/family/agent search,
comma = OR) **scopes the whole view** — fleet-score-over-time, convergence
top-movers, tiles, histogram, family rollup, waits, outcomes, and table all
recompute for the selected subset (GPU occupancy stays fleet-wide). Deep-
linkable via `?fam=…&q=…`. Candidate code opens in an in-page **modal**
(inline `<template>`, `?code=<cand>` deep-link) — static-safe under `file://`.

**Aggregation note (does the mean change the engine? No):** Accept/Select
consume the per-shape score *vector*; the mean-of-S is derived — used only
for reporting and the finalize argmax. At finalize both aggregates
(mean-of-S and S-of-geomean-latencies) are recorded, so the open question
about the website's exact per-problem formula (kb/benchmark-grader.md
§Aggregation) can relabel the winner but can never alter the search.

## 10. Measurement honesty (v1 posture)

- Frontier admission requires a **full eval** (all shapes) — v1 has no
  partial evals at all, so this is trivially true (screen mode is deferred).
- `env` fingerprint (GPU/driver/clock/harness) is **recorded on every
  result** from day one (stub-empty until Phase F's real pod) — comparisons
  across different pods are flagged; automated re-baselining is deferred.
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

Determinism is the whole strategy: the two non-deterministic dependencies (GPU,
LLM) sit behind interfaces, so with deterministic stubs the engine's routing /
frontier / budget / plateau / persistence behavior is *fully determined* and
exactly assertable. Correctness here is **orchestration logic, not kernel
quality** — none of these tests need a real GPU or model.

**Stub contract (build the stubs test-ready in Phase B).** The tests are only
as strong as the stubs' scriptability:

- **StubExecutor** — deterministic, keyed on `Solution.hash()`, with a
  **scenario API**: map a hash → per-shape outcome (score, or `TIMEOUT` /
  `COMPILE_ERROR` / `REWARD_HACK` / partial-specialist), an optional **noise
  term** (fixed seed), and an optional **dispatch delay** (to force async
  interleavings). It also **asserts it is never entered re-entrantly** — any
  accidental parallel GPU access fails a test on the spot.
- **StubAgent** — deterministic, with a **scripting API**: a scripted candidate
  sequence, **context-dependent** responses (so tier N behaves differently from
  tier N−1), produce-**duplicate** / produce-**invalid** on demand, a
  per-`(agent, model)` **identity**, and **raise-on-demand** (crash tests).
- **Injectable clock + seed** — a controllable event-loop clock and one RNG
  seed, so every run is reproducible and specific interleavings are forceable.

**Tests:**

1. **Kill/resume safety** — kill after every journal line. With deterministic
   stubs the resumed run is *identical* to an uninterrupted one (bitwise state
   equivalence). With real agents resume is legitimately nondeterministic (an
   in-flight agent call re-runs), so the guarantee is **no-loss /
   no-double-pay**: every journaled GPU result and result-store trace is
   retained, budgets are exact, no completed GPU eval is re-run.
2. **Budget exactness** — rejects, bounces, evals, timeouts all count correctly
   against both caps; check-fails and novelty-bounces never charge
   `max_gpu_evals`.
3. **Frontier correctness** — property test: the ε-Pareto set matches brute
   force over random score vectors + ε, including injected-noise scenarios (the
   frontier stays stable under the stub's noise term).
4. **Plateau → escalation → termination** — with a stubbed score **below**
   `escalate_ceiling`, a tier plateau (M full pool-cycles, *every* pool member
   having produced ≥1 candidate — verified via stub identities in
   `plan_done.{agent, model}`) emits `agent_changed{tier, trigger:"escalation"}`
   and resets the counter; with a stubbed score **at/above** the ceiling, or on
   the last tier, it **terminates** instead (a near-optimal problem never
   climbs). No spin (bounces count as iterations). A single-tier, single-model
   ladder degenerates to plain plateau termination.
5. **Gates** — a hash-identical candidate never re-pays a GPU run; a
   stub-judged "cosmetic" candidate bounces to Plan carrying the verdict; a
   check-invalid candidate is rejected — each without touching the executor.
   `REWARD_HACK` is quarantined (never selected, never fed to reflection).
6. **Crash isolation** — a StubAgent that raises on one problem stops *only*
   that problem (`solver_error` journaled once) while the rest of the fleet
   finishes; fleet-wide credit-exhaustion suspends cleanly (one `suspended`
   line), not N per-problem errors.
7. **Concurrency / single-flight** — under injected dispatch delays that force
   races, the StubExecutor's re-entrancy assertion never trips (exactly one
   eval on the GPU at a time), fair round-robin holds (each problem ≤1 eval in
   flight; RR order across problems), and the curator queue never interleaves
   writes. This is the async layer the other tests don't exercise.

**Out of scope for stubs** (other layers, by design): whether the real agent
writes good/valid kernels; whether the stub's scoring matches real GPU noise
(injected scenarios *approximate* it, they don't validate it); real
GPU-transport failure modes (a fake queue *simulates* pod-crash-mid-eval, but
the real transport is validated only at Phase F).

## 13. Build order

Status legend: ✅ built · ⬜ v1, planned · ⛔ GPU-blocked. **Only Phase A
exists today; B–E are v1 scope, not yet written.**

| Phase | Status | Piece |
|---|---|---|
| A | ✅ built | Executor interface + **async** StubExecutor (scenario API + re-entrancy assertion, §12) |
| B | ✅ built | `solve_problem` loop + RunContext + journal/replay + ε-frontier + novelty (hash + judge) + caps/plateau + **tier ladder / headroom-gated escalate (§6b)** + StubAgent + **§12 tests (12 passing)** |
| C | ⬜ v1 | Knowledge store + serialized curator (per-finished-problem) + design-doc-at-iter-0 |
| D | ⬜ v1 | Fleet (`main`, crash isolation, **static exemplar-first launch order**) + CLI (`solve`, status/journal/frontier/candidates views) |
| E | ⬜ v1 | **Bootstrap** sibling-templating transfer (§2) |
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
- **Agent-credit enforcement** — logged not enforced for the Claude
  subscription (Anthropic hard-stops it); **required sooner once a tier pool
  includes a metered provider** (GPT / DeepSeek / GLM / … spend real money
  per token).
- **Per-family parallel curation** (§8) — a per-family lock so different
  families curate concurrently, beyond the single global curator queue — if
  curation ever bottlenecks fleet throughput.
- **Multi-seed sibling import** (§8) — seed a new problem from several
  structurally-distinct sibling frontier members, not just the single best —
  if bootstrap diversity proves worth the extra evals.
- **Reflection tiering** (deeper/multi-level reflection on interesting
  results; v1 does one reflection, by the planning agent) **+ normalized-hash
  novelty tier** — if judge/reflect costs show up in logs.
- **Dynamic family scheduling + live sibling reads** (§2) — work-conserving
  rebalancing and mid-run reads of a running sibling's frontier, beyond the
  static exemplar-first launch order + bootstrap templating — when the fleet
  is small enough that the GPU actually idles, or siblings routinely finish
  far apart.
- **Evidence-triggered escalation** (§6b) — escalate to the next tier
  immediately on repeated `COMPILE_ERROR`/`INCORRECT` a tier can't fix,
  without waiting out the full plateau window — when logs show tiers stuck on
  fixable errors.
- **Submit node** — if/when we decide to submit; one function call at the
  end of `finalize`.
- **Nsight → ASI profiling in the eval path** — Phase F+; profile on
  plateau, not every run.
- **Parallel perspective panel** (§6b) — on plateau, run several perspectives
  *concurrently on the same parent* and keep the best, beyond within-tier
  round-robin + sequential tier escalation. More diverse, but multiplies
  GPU/credit per stuck problem — turn on when the tail is wide *and* the budget
  allows.
- **Drop-back-to-cheap after a win** (§6b) — de-escalate to the cheapest tier
  once a higher tier lands an improvement (the unblock may reopen cheap
  gains). Adds churn; enable if the dashboard shows expensive tiers winning
  gains a cheap tier could then have continued.
- **HTML dashboard** — nice-to-have.

## Decision log

1. **Agents:** two backends behind the `Agent` interface — Claude Agent SDK on
   the subscription OAuth token (default) and one generic OpenAI-compatible
   backend for GPT / DeepSeek / GLM / Qwen / Kimi. The novelty judge uses a
   cheap fixed model; reflection is done by the planning agent. Perspectives
   pool into **tiers** (§6b); **v1 ships Claude-only**. Claude credit pool
   (Pro $20 / Max-5x $100 / Max-20x $200 monthly, non-interactive) is logged
   not enforced; metered providers bump enforcement.
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
7. **Tier ladder (§6b):** a **tier** (pool of `(agent, model)` perspectives,
   round-robined for continuous diversity) escalates to the next tier on a tier
   plateau **only while headroom remains** (`best < escalate_ceiling`);
   otherwise it terminates — so easy problems stay cheap. Plateau window = M
   full pool-cycles. Per-problem, ratchet-up. The one-shot `design()` uses a
   strong `design_model`. Backends framework/provider-agnostic (Claude Agent
   SDK + OpenAI-compatible); v1 Claude-only. Parallel panel, drop-back,
   evidence trigger Deferred.
8. **Autonomous:** no in-loop human steering (no `hints.md`); operators tune by
   config between runs. Observability views (§9) are read-only.
