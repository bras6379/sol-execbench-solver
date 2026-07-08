# Deep research: cross-operator technique transfer for the kernel-writing fleet

95-agent deep-research pass (search → fetch → adversarial 3-vote verification →
synthesis), run 2026-07-07/08. Full raw output (question, findings, refuted
claims, sources, stats) in `context-architecture-deep-research.json` next to this
file. This is the source material behind
`../context-architecture-plan.md`.

## Question researched

For a fleet of LLM coding agents that write GPU kernels (Triton/CUDA) to optimize
~15-50 diverse operators against SOL-ExecBench, evaluated by real GPU runs with
correctness+latency scoring — what does SOTA research say about building and
using a persistent, reusable library of validated code patterns/techniques that
later agents can retrieve and adapt when tackling a *different* operator that
needs the *same* technique, instead of re-deriving (and re-breaking) it from
scratch each time? Five areas: (1) retrieval-augmented code generation, (2)
memory-augmented/experience-replay coding agents, (3) library-learning/program
synthesis research, (4) GPU-kernel-specific automated optimization research, (5)
practical retrieval/indexing design at small corpus scale.

## Bottom line

None of the evidence that survived adversarial verification is GPU-kernel-specific.
Every claim sourced from GPU-kernel-optimization literature (Sakana AI's CUDA
Engineer, KernelBench/GPU MODE retrospectives, FunSearch/AlphaEvolve, and a system
called cuPilot with an explicit RAG kernel-strategy-pool) was either not found or
refuted under adversarial fact-checking. Any design decision for this fleet is an
extrapolation from general-purpose code-RAG and library-learning research, not
direct domain evidence — treat confidence accordingly.

## Confirmed findings (survived 3-vote adversarial verification)

1. **Hybrid retrieval beats either signal alone, and the two signals are
   complementary, not redundant.** For code completion, similarity/embedding
   retrieval substantially beats identifier/keyword retrieval, and combining
   BM25 (lexical) + embeddings beats either alone, with the advantage growing at
   larger model scale. BM25 and semantic retrievers return mostly *disjoint* top
   candidates (64–76% non-overlap out of 100 queries) — they capture genuinely
   different notions of code similarity.
   *Source: WeChat/Tencent industrial study, 26 LLMs (0.5B–671B), 1,669 internal
   C++ repos (arXiv:2507.18515). High confidence, but all three sub-claims come
   from one paper/one codebase — a strong single data point, not triangulated.*

2. **Retrieval signals beyond plain semantic similarity can help, especially on
   hard problems — but the underlying "semantic similarity misses algorithmic
   similarity" diagnosis was NOT confirmed.** A bespoke solution-aware retriever
   (SolveRank) beat semantic/lexical baselines on hard problems specifically, and
   a graph-structured repo-level retriever beat text-similarity RAG. But the
   claim that plain embeddings fundamentally can't capture technique-level
   similarity was refuted (1-2 vote) — don't read this as "embeddings are the
   wrong tool," only as "specialized retrievers can do better on hard cases."
   *Sources: arXiv:2509.01129 (SolveRank, EMNLP 2025 Findings), arXiv:2504.10046
   (CodeRAG/GraphCodeAgent). Medium confidence — self-reported benchmarks, no
   independent replication, small hard-tier sample (~n=36) in SolveRank.*

3. **Granularity for a reusable code abstraction should be decided mechanically,
   not by LLM judgment.** DreamCoder and Stitch both delegate this to symbolic/
   algorithmic compression search (MDL/Bayesian objectives, corpus-guided
   synthesis) over a corpus of already-correct programs. LILO explicitly puts the
   LLM's role *after* extraction — naming/documenting an abstraction Stitch has
   already chosen, never choosing which abstractions to keep.
   *Sources: LILO (arXiv:2310.19791, ICLR 2024), Stitch (arXiv:2211.16605, POPL
   2023), DreamCoder (dl.acm.org/10.1145/3453483.3454080, PLDI 2021). High
   confidence on the general framing; a specific claim about Stitch's exact
   utility function (size × reuse-sites) was refuted — the general MDL/compression
   framing holds, the precise formula doesn't.*

4. **Auto-generated documentation of a mined abstraction measurably improves its
   later reuse, not just its readability — but isn't sufficient alone.** LILO's
   controlled ablation: adding AutoDoc to an already-fixed search raised solve
   rates on REGEX (+9.73pp) and CLEVR (+2.27pp), zero gain on LOGO. Full LILO
   (search + AutoDoc) is needed to beat the no-library baseline in every domain —
   AutoDoc alone doesn't get there.
   *Source: arXiv:2310.19791. Medium confidence — single paper, but a clean
   controlled ablation.*

5. **Unfiltered automated abstraction mining on real code is overwhelmingly
   noisy, and even after heavy filtering the net benefit is marginal.** Leroy:
   generic Stitch search on 122 real Python programs produced ~45,000 raw
   candidates; only 6 valid, non-trivial abstractions survived filtering
   (>99.99% garbage). Overall corpus compression was only 1.04x, and the corpus
   actually *grew* 1.2% in AST nodes once the library itself was counted, because
   correctness-preserving "closing" (adding params for scoping) inflates
   abstraction size in imperative languages.
   *Source: arXiv:2410.06438 (Leroy, HATRA/SPLASH 2024 workshop). Medium
   confidence — one small "worst-case" corpus at a workshop-tier venue, but the
   qualitative lesson (raw symbolic search needs heavy filtering; DSL-benchmark
   gains don't transfer to real code) is unlikely to be a corpus artifact alone.*

6. **The single most important negative finding: mined abstractions do not
   reliably transfer outside the distribution they were learned on — even within
   the *same* toy DSL under a mild shift.** AbstractBeam showed a real,
   statistically significant in-domain gain over its no-library baseline
   (p<0.05), but on an out-of-distribution task set using the *identical* DSL and
   *identical* 28 primitives, the gain vanished and search got *slower* (3.62s vs
   1.37s). The paper's own conclusion: abstractions "do not generalize beyond the
   domain they are derived from." This is a direct, serious warning against
   assuming a technique mined from one operator (e.g., split-K GEMM) will "just
   work" on an unrelated operator (e.g., RoPE backward) without deliberate
   re-derivation and re-validation.
   *Sources: DreamCoder (dl.acm.org/10.1145/3453483.3454080), AbstractBeam
   (arXiv:2405.17514). High confidence.*

7. **CODESKILL is the closest architectural analogue to what this fleet needs,
   but its own claimed ROI didn't survive verification.** Retrieves reusable
   skills via dense embedding similarity at two granularities — task-level
   skills matched to the task goal, and event/trigger-based skills matched to the
   agent's *live* reasoning/error state — and gates every candidate skill
   through a trained (RL) add/merge/drop policy before it enters the persistent
   library, explicitly screening out anything "overly specific, or unlikely to
   transfer." Treat this as a design template, not proof it works: the paper's
   claimed +9.69pp over no-skill and +4.01pp over the best alternative memory
   approach was explicitly refuted (1-2 vote).
   *Source: arXiv:2605.25430 (~May 2026, likely pre-peer-review). High confidence
   on the architecture description, explicitly unconfirmed on its effectiveness
   claim.*

## Refuted claims (do not reuse)

- Semantic-similarity retrieval "misses" algorithmic/technique-level similarity
  as its core failure mode (1-2, arXiv:2509.01129).
- A specific 6.31-point Pass@1 ablation drop from removing graph-reasoning
  retrieval (0-3, arXiv:2504.10046).
- Stitch's granularity objective is precisely "abstraction-size ×
  number-of-reuse-sites" (1-2, arXiv:2211.16605).
- A minimum-complexity gate (≥2 non-trivial sub-expressions) before admitting a
  mined pattern (1-2, arXiv:2405.17514).
- CODESKILL's +9.69pp/+4.01pp improvement over baselines (1-2, arXiv:2605.25430).
- **cuPilot** (the one system that looked like a direct GPU-kernel-RAG analogue):
  its persistent RAG-indexed kernel-strategy database (0-3), its 54.1%
  latency-reduction ablation (0-3), and its strategy/implementation-granularity
  decoupling claim (0-3) — all three refuted. No GPU-kernel-specific claim from
  any source survived.

## Open questions this research did not resolve

- Does cross-operator technique transfer need a different retrieval signal than
  code-similarity embeddings (structured tags for technique/hardware-constraint/
  failure-mode, or a bespoke trained retriever like SolveRank)? Neither confirmed
  nor cleanly refuted.
- What do GPU-kernel-specific systems (Sakana's CUDA Engineer, KernelBench/GPU
  MODE, FunSearch/AlphaEvolve) actually report about cross-kernel pattern reuse
  and why top solutions reach near-speed-of-light? Zero claims from these sources
  survived verification — a dedicated, narrowly-scoped research pass on just
  these named sources would be needed to answer this.
- At this fleet's actual corpus scale (dozens–low hundreds of patterns), does a
  lightweight tag/keyword index capture most of the benefit more cheaply than
  full vector RAG? Finding 1's scale-dependence (embedding advantage grows with
  model/corpus size) suggests this could go either way at small scale — no
  direct evidence either way.
- What would an automatic correctness gate for GPU kernel patterns look like
  (must mean "passes real GPU tolerance/perf tests on the new target," not just
  syntactic validity), and given Leroy's >99.99% rejection rate even on simple
  Python, how expensive would equivalent validation be in GPU-hours?

## Caveats on how far this can inform the design decision

1. Zero confirmed domain-specific (GPU-kernel) evidence — every relevant claim
   was refuted or not found. Any GPU-kernel design choice here is inference from
   general-purpose code/library-learning research.
2. Area 5 (practical small-corpus retrieval design) has no directly-confirmed
   claim at all; conclusions there are reasonable inferences from Finding 1, not
   cited results.
3. Concentration risk: Finding 1's three sub-claims all come from one paper/one
   codebase (WeChat/Tencent, C++) — strong but singular evidence.
4. The two claims most directly on-point for this fleet's stated pain point
   (semantic similarity "misses" technique similarity; a CODESKILL-style library
   beats no-library) were both refuted, not confirmed. The case for building
   anything has to rest on indirect, mechanism-level findings, not a validated
   end-to-end success story.
5. External validity to GPU kernels is unproven: DreamCoder/Stitch/LILO/
   AbstractBeam operate on small DSLs (list processing, LOGO, grid-world) with
   clean functional semantics; none of this touches numerically-sensitive,
   hardware-constrained, correctness-gated Triton/CUDA code. If anything, the
   AbstractBeam transfer-failure finding argues for *more* caution extrapolating
   to a much larger domain gap (split-K GEMM → RoPE backward).
6. CODESKILL (arXiv:2605.25430) is dated ~May 2026, likely pre-peer-review —
   treat as an illustrative current-research-direction template, not settled
   practice.

## Stats

95 agent calls · 15 sources fetched · 74 claims extracted · 25 adversarially
verified (17 confirmed, 8 killed, 0 left unverified) · 7 findings after
dedup/synthesis.
