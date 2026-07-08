# Cross-operator technique transfer + context-architecture fixes

## Context

You asked why the fleet's kernels keep coming back incorrect/plateaued and whether
better "context engineering" (e.g. feeding agents related papers) could help,
then sharpened it to the real question: how do top SOL-ExecBench leaderboard
entries reach near-1.0 while this fleet plateaus, and what's the right context
architecture to close that gap? You asked me to research SOTA approaches first
(RAG-for-code, memory-augmented coding agents, library learning, GPU-kernel-specific
automated optimization) and design a plan before touching any code.

The research (95-agent deep-research pass, full results in
`docs/research/context-architecture-deep-research.{md,json}`) came back mostly
**not** domain-specific: every claim sourced from GPU-kernel literature (Sakana's
CUDA Engineer, KernelBench/GPU MODE, FunSearch/AlphaEvolve, a system called cuPilot
with an explicit RAG kernel-strategy-pool) was either not found or refuted under
adversarial verification. What *did* survive, from general code-RAG and
library-learning research, points at a specific, disciplined design rather than a
generic "add a vector DB":

1. Granularity for a reusable unit should be decided mechanically, not by LLM
   judgment (DreamCoder/Stitch delegate this to symbolic compression search).
2. Validation must be aggressive and cheap — unfiltered abstraction mining is
   >99.99% garbage on real code (Leroy: 6 valid out of ~45,000 candidates).
3. **The single most important negative finding**: mined abstractions do not
   reliably transfer outside the distribution they were learned in, even within
   the *same* toy DSL under a mild shift (AbstractBeam) — direct warning against
   assuming a technique from one operator "just works" on an unrelated one.
4. CODESKILL (closest architectural analogue) retrieves via task-level +
   event/trigger-based skills and gates every candidate through an explicit
   add/merge/drop policy — a good template, though its own reported ROI over a
   no-library baseline didn't survive verification.
5. At this corpus scale (dozens–low hundreds of entries), nothing confirms a
   vector-embedding RAG system is worth it over a simple tag-indexed lookup.

Two exploration passes over this actual codebase then found that **most of the
needed plumbing already exists** — it's just disconnected across operators — plus
two concrete, verified problems that make the case concrete rather than
theoretical:

- **A live knowledge-base leak.** Agent `context_read` telemetry shows agents
  trying to open kb-file-shaped paths that don't exist on disk
  (`bf16-reference-rounding-match.md`, `sol-execbench-eval-loop-memoization.md`,
  `scatter-add-family.md`, `correctness-gating.md`). These are near-exact matches
  to *my own private cross-session memory* for this project — genuine
  confirmed-technique/confirmed-dead-end writeups from prior sessions that never
  got transcribed into the shared `kb/` the worker agents can actually read.
- **A verified, currently-live labeling bug.** `solver/cli.py:162-163` computes
  every problem's `family`/`name` via `sim.family_of(task_id)` — a
  `task_id % 10` round-robin over 10 generic labels — **unconditionally,
  including the real `--gpu` path**, not just the sim agent. Confirmed on disk:
  `knowledge/families/gemm.md` contains task 234, which is actually an RMSNorm
  kernel (`op_key_of(234) == "rmsnorm"`), filed under `gemm` purely because
  `234 % 10 == 4` and `_FAMILIES[4] == "gemm"`. This also silently corrupts the
  *already-shipped* `_transfer_line()` sibling nudge in every coach card today.

The plan below fixes the verified bug first, closes the memory-transcription gap
(cheap, zero architecture risk), then builds the actual cross-operator technique
transfer mechanism by **extending** `KnowledgeStore`/`reflection.py` rather than
inventing new infrastructure — directly matching item (4) "cross-problem technique
seeding" in `docs/orchestration.md` §8's own transfer list, which was designed but
never built.

## Part 0 — fix the `family` mislabeling bug (ship first, alone)

`op_key_of()` (`solver/engine/knowledge.py:31-44`) is already correct and already
used for same-op sibling matching. Make it the single source of truth everywhere:

- `solver/cli.py:162-163`: replace
  `families = {t: sim.family_of(t) for t in ids}` /
  `names = {t: f"{sim.family_of(t)}_{t}" for t in ids}`
  with `op_key_of(t, args.problems_dir)` for both — **except** inside the literal
  `args.agent == "sim"` branch (line 165), which has no real `definition.json` to
  key off and should keep `sim.family_of` for its own synthetic ids.
- `solver/engine/reflection.py`'s `_COMPUTE_FAMILIES` exact-match check (line 251)
  breaks once `family` is a compound real op key
  (`attention_output_projection_with_reshape_backward`, not `attention`). Replace
  with substring containment against an expanded keyword list:
  ```python
  _COMPUTE_KEYWORDS = ("gemm", "matmul", "attention", "moe", "conv", "linear",
                       "projection", "expert", "mlp", "swiglu", "geglu")
  def _compute_bound(family: str) -> bool:
      return any(k in (family or "").lower() for k in _COMPUTE_KEYWORDS)
  ```
- This also fixes `_transfer_line()`'s sibling nudge for free — it currently can
  tell an agent working on a norm that an unrelated GEMM problem is its "sibling."
- No fallout elsewhere: the dashboard only ever treats `family` as an opaque
  grouping string (verified), and `tests/test_reflection.py` doesn't hardcode the
  old 10 literal family values.
- Existing `knowledge/families/*.md` and `knowledge/global.md` content is
  contaminated by the bug; they're append-only logs, not replayed state, so no
  migration is required — just let them start filing correctly going forward
  (old entries stay as historical noise, acceptable given they were never read
  back into a prompt anyway, per the exploration findings).

## Part A — close the private-memory → shared-`kb/` transcription gap

Manual, one-time-then-periodic editorial pass (not automatable — worker agents
never see my private memory store; this is a human/orchestrator bridge). Not
gated on anything else in this plan; do anytime.

- Filter: transcribe only entries that are (a) genuinely about a kernel-writing
  technique or a B200/Triton/CUDA pitfall, not repo/infra trivia (pod bootstrap,
  harness flakiness, CLI flags — those belong in `docs/`, not `kb/`); (b) phrased
  generally enough to be useful outside the one session that found it.
- Follow the existing `kb/` convention exactly: purpose line, inline ✅/⚠️/❌ tags
  per claim (`kb/README.md`'s legend), cite the source, cross-link related files.
  Confirmed-dead-end material goes in a "## ❌ Refuted / suspect claims" section
  matching `kb/optimization-playbook.md`'s existing graveyard pattern.
- Prioritize the memories whose slugs already leaked into agent telemetry
  (bf16 rounding, eval-loop memoization, scatter-add family, correctness gating)
  since there's direct evidence agents are already reaching for this content.
- Cadence: revisit whenever my private memory picks up a meaningful number of
  new project-memory entries since the last pass — not schedulable via any code
  in this repo.

## Part B — cross-operator technique pattern library

Extends `KnowledgeStore` (`solver/engine/knowledge.py`), reusing its existing
single-writer lock and its one call site, rather than building new infrastructure.

**Design choices, and why:**
- Atomic unit = the existing `TECHNIQUES` tag taxonomy
  (`reflection.py:32-51`, 18 tags, plain substring classification) — sidesteps the
  DreamCoder/Stitch granularity problem entirely since it's already hand-designed
  and small.
- "Confirmed working" entries are gated on `verdict == "entered"` — the same gate
  `store.record_playbook()` already uses. Zero incremental GPU cost.
- "Pitfall" entries (from correctness failures) are **never** framed as "this
  technique doesn't work" — only as a named implementation trap to avoid, and only
  promoted after replication across **≥2 different op families** (not just
  different task ids). Rationale: a `COMPILE_ERROR`/`RUNTIME_ERROR` failure is
  overwhelmingly evidence of an implementation bug, not a technique verdict —
  exactly the failure mode this whole project is trying to fix, so the pitfall
  ledger must not itself become "confirmed dead end" folklore. `TOLERANCE`
  failures are the only kind allowed to hint at a genuine numerics/technique
  issue, and even then framed as a hypothesis (the AbstractBeam finding says
  cross-distribution transfer failure is common even within one DSL — one
  family's numeric failure is weak evidence for a different family).
- Cross-op entries are **never** injected as ready-to-copy seed code (unlike the
  same-op `sibling_hint`, which is safe only because same-op implies same
  distribution). Only a natural-language summary + explicit known pitfalls + a
  pointer to `runs/<task>/candidates/<cand>.json` for the agent to read and
  re-derive. This directly targets the diagnosed failure mode (implementation
  details, not the algorithmic idea) and is closer to CODESKILL's
  trigger/event-matched skill than a generic snippet library.
- Retrieval = a plain dict lookup keyed by technique tag, no embeddings — matches
  what the research could actually confirm holds at this corpus scale, and stays
  fully auditable.

**Data schema** — one file, following the codebase's own JSON=authoritative /
MD=human-view convention (`best/<op>.json`+`families/<op>.md`):

```jsonc
// knowledge/patterns.json — atomic tmp+rename write, upsert keyed by (tag, task_id)
{
  "schema_version": 1,
  "tags": {
    "split_k": {
      "confirmed": [   // verdict == "entered"; capped [:5], sorted by score desc
        {"op": "gemm", "task": 213, "score": 0.79, "note": "K=4096 3-way split, "
         "atomic epilogue add; K<512 not worth it", "cand": "a1b2c3d4",
         "ts": "2026-07-06T18:05:00Z"}
      ],
      "pitfalls": [    // grouped by error_family; capped [:5] by n_ops_seen desc
        {"error_family": "RUNTIME_ERROR", "n_ops_seen": 3,
         "ops": ["gemm", "moe_expert_computation"],
         "note": "atomic_add epilogue needs accumulator dtype == output dtype "
         "exactly, or Triton silently truncates — 3 independent kernels hit this "
         "as a last-tile RUNTIME_ERROR", "example_cand": "e5f6a7b8",
         "ts": "2026-07-07T09:12:00Z"}
      ]
    }
    // one key per TECHNIQUES tag, created lazily on first write
  }
}
```
`knowledge/patterns.md` is a deterministically-regenerated human view of the same
file (never read by any agent path) — same relationship as `best/`→`families/`.

**Concrete hooks:**

- `reflection.py`: extract the inline compute-bound/ladder filter
  (lines 251-255) into a standalone, history-independent
  `relevant_techniques(family) -> tuple[str, ...]`, usable both from
  `analyze()` (unchanged behavior) and from `design()`/`_context_md()` with zero
  run history. Add `ProblemReflection.tech_events: list[dict]` — every
  `(technique-tag, verdict, score/error, cand, strategy)` seen this run, populated
  in the same loop that already builds `.ledger`/`.failed`, no new I/O.
- `knowledge.py`: `KnowledgeStore._load_patterns()`/`_write_patterns()` (atomic
  tmp+rename, mirrors `_write_best`); new read method
  `pattern_notes(tags, *, exclude_task=None) -> dict[tag, {"confirmed":[...],
  "pitfalls":[...]}]`; `curate(ctx, op, name, *, refl=None)` gains the optional
  `refl` param and, when present, calls `_upsert_patterns(refl)` — **upsert keyed
  by `(tag, task_id)`, not append**, so a `reopened`-and-reterminated run replaces
  its own prior entry instead of double-counting (the existing `_append_family`
  has this duplication bug already — don't copy it into new code).
- `loop.py:504-511` (the existing, only `curate()` call site, once per finished
  problem, inside the existing lock): compute
  `refl = reflection.from_runs_dir(runs_dir, task_id, name=name, family=family)`
  (best-effort, never blocks `curate()` on failure) and pass it through. Purely
  CPU-side on already-flushed local files — **no interaction with the GPU
  single-flight lock**.
- `cli_agent.py`:
  - `CliAgent.design()` (currently never calls `_context_md()` at all — the
    highest-leverage and currently-unreached injection point, since this is
    where the strategic ranking that determines which technique gets tried
    *first* is locked in) gains `_write_patterns(wd, family)`, writing
    `PATTERNS.md` from `relevant_techniques(family)` + `pattern_notes(...)` —
    zero run history required.
  - `_context_md()` gains a new "## Cross-op notes" section (`_cross_op_section`),
    inserted right after the coach card, same priority tier. **Never memoized on
    `ctx`** — always a fresh `pattern_notes()` read per call, exactly mirroring
    why `ctx.reflection` is already dead code (`cli_agent.py:772-777`'s comment)
    and unlike `ctx.sibling_hint`, which is deliberately bootstrap-only.
  - Pitfall notes render as traps to avoid ("re-derive this, but watch for X"),
    never as bans — matches the schema's own framing.
- `Config` gains `cross_op_patterns: bool = True` — a feature flag so the write
  side (corpus accumulation) can run independently of whether the read side is
  rendered into prompts, needed for the staged rollout below.
- `Candidate` gains `cross_op_patterns_shown: list[str] | None` (which tags had
  non-empty notes for this specific call) so effect can be measured later against
  `candidates/<cid>.json`'s existing `strategy`/`correct`/`per_workload[].error`.

## Part C — fleet-wide "untried rung" visibility (deferred)

A pure read-side query over Part B's own `patterns.json` (cross-referenced with
`relevant_techniques()` per known op) — **not** a second write path, no new
`global.md` aggregation logic needed. Explicitly **do not build this until Part B
shows a measurable signal** (see Verification below): the research found zero
domain-specific evidence either way on whether "nobody has shipped X" nudges
agents toward higher-ceiling techniques or just reinforces "this is risky, everyone
avoided it." If built, frame as opportunity ("no confirmed nvfp4 kernel yet fleet-
wide — this op-class's ceiling suggests it may be necessary"), never as
absence-shaming, and recompute on `reflect_all()`'s existing timer cadence so it
stays live across a still-running fleet.

## Rollout sequencing

1. **Part 0** — ship alone first. Independently valuable bug fix; unblocks
   everything else (Part B's compute-bound gating and pitfall replication-by-
   op-family both depend on `family` being real).
2. **Part B, write-side only, dark** (`Config.cross_op_patterns` need not even be
   wired to the read side yet) — let `knowledge/patterns.json` accumulate across
   a batch of runs. Validate offline: entries correctly tagged, pitfall/confirmed
   split behaves, no duplicate entries across a `reopened` run.
3. **Part B, read-side on** — turn on rendering into `PATTERNS.md`/`CONTEXT.md`.
   Part A can run in parallel any time, independent of 1–2.
4. **Part C** — only if step 3 shows a non-null signal.

## Verification

- Part 0: after the fix, run `solver solve --gpu` (or check the next problem that
  finishes) and confirm `knowledge/families/<real-op>.md` files under the correct
  key — spot check against `op_key_of()` directly, same way the bug was confirmed
  here (`.venv/bin/python -c "from solver.engine.knowledge import op_key_of; ..."`).
- Part B write-side: after a handful of problems finish, inspect
  `knowledge/patterns.json` by hand — confirm tags are populated, `confirmed`/
  `pitfalls` entries look sane, re-run the same finished problem's `curate()` path
  (or just check a `reopened` run) and confirm no duplicate `(tag, task_id)`
  entries.
- Part B read-side: compare, per technique tag, first-attempt correctness rate
  and `COMPILE_ERROR`/`RUNTIME_ERROR` frequency for candidates where
  `cross_op_patterns_shown` was non-empty for that tag vs. candidates where it
  was empty (feature flag off, or tag had no notes yet) — the concrete, cheap
  "is this actually helping" check, using data already collected by this plan.
- Existing test suite (`tests/`) must still pass — `tests/test_reflection.py`
  covers `analyze()`/`render_card()`, add coverage for `relevant_techniques()`
  and the new `tech_events` field; `tests/test_store.py`-style fixture pattern
  extends naturally to a new `tests/test_knowledge.py` for `pattern_notes()`/
  `_upsert_patterns()`'s upsert-not-append behavior.

## Explicitly out of scope (and why)

- **No vector-embedding RAG.** Nothing in the verified research confirms it's
  worth it at this corpus scale (dozens–low hundreds of entries); a tag lookup is
  simpler, fully auditable, and matches the one confirmed finding that lexical/tag
  and semantic signals are complementary, not that semantic is required.
- **No automated code-block/abstraction mining** (DreamCoder/Stitch-style
  symbolic search over kernel source). Leroy's finding (>99.99% of raw candidates
  are garbage even on simple Python) makes this a bad bet for numerically-
  sensitive, hardware-constrained Triton/CUDA source, and the existing
  hand-designed `TECHNIQUES` taxonomy already gives a small, safe unit of
  granularity for free.
- **No forced escalation toward "riskier" techniques** in this pass (that's
  Part C, explicitly deferred). The risk-aversion-in-search hypothesis has no
  domain-specific evidence behind it yet; ship the lower-risk, better-evidenced
  parts first and only build the speculative piece if Part B's data suggests it's
  worth it.
