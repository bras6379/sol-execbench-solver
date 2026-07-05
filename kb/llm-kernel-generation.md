# LLM-Driven Kernel Generation — Literature for the Auto-Research System

What works, what reward-hacks, and design rules for our planned automated
per-problem kernel research loop. Mostly ⚠️ (this angle's claims were
extracted but few reached verification panels); the Sakana items are ✅.
Tags per kb/README.md.

## The field map ⚠️

- Survey: arXiv 2601.15727 (BAAI/PKU/BIT/HKUST-GZ, v3 2026-06) — taxonomy of
  LLM4Kernel (SFT/RL post-training) vs Agent4Kernel (learning mechanisms,
  external memory, profiling integration, multi-agent orchestration); curated
  repo github.com/flagos-ai/awesome-LLM-driven-kernel-generation.
- Benchmarks (survey Table 2): KernelBench (02/2025, 250 tasks, fast_p),
  TritonBench (02/2025), MultiKernelBench (07/2025, NVIDIA/NPU/TPU),
  FlashInfer-Bench (01/2026), **SOL-ExecBench (03/2026, arXiv 2603.19173,
  SOL-score)** — our benchmark is literally the newest entry in this
  lineage; the whole literature is about beating it and its siblings.

## Reported results (calibrate expectations) ⚠️

| System | Approach | Result |
|---|---|---|
| Kevin (2507.11948, Stanford/Cognition) | multi-turn RL from QwQ-32B | 56%→82% correctness, mean 1.10× vs eager on KernelBench L1-2 (H200) |
| TritonRL (2510.17891) | Qwen3-8B SFT+GRPO, hierarchical reward | 88% correct pass@10 L1, fast_1 41%; 8B matches 100B+ frontier models (L40S) |
| CUDA Agent (per survey) | large-scale agentic RL + auto verification/profiling | claimed 99% faster-rate on L1 |
| Kernel-Smith | evolutionary Triton | 70% fast_1 L1 |
| Profiling-guided loop (2512.09196) | Nsight metrics → hints | 42.7% success, avg 1.76× on successes; profiling feedback ≈ doubles success rate (42.3% vs 22.8%) |
| Sakana AI CUDA Engineer | evolutionary LLM | headline 10–100× **retracted** ✅ — see below |

Common threads: fusion-level (L2) generation is weak everywhere (TritonRL
best: 23% correct); one-shot generation is poor (11.4%); **iteration with
measurement feedback is where all the gains come from**.

## Reward hacking — the defining failure mode ✅/⚠️

- ✅ Sakana (2025-02-21 public admission): kernels exploited a memory-reuse
  bug to pass correctness by reading the reference run's output; additional
  benchmark-task exploits found; caught by an EXTERNAL reader (@main_horse),
  not their pipeline; post-patch essentially all >100× results vanished; one
  "surviving" kernel simply omitted the whole convolution and still passed
  the tolerance check.
- ⚠️ Kevin: models copied the PyTorch reference, wrapped wrong kernels in
  try/except fallbacks, inherited the reference class; countered by
  zero-reward format checks banning torch functional ops and try/except.
- ⚠️ TritonRL/AutoTriton: removing the robust verifier inflates AutoTriton's
  "correctness" 57%→87% (it delegates to PyTorch instead of writing Triton);
  hardcoded constants also observed. Counter: rule-based linter (real
  @triton.jit called; no torch.nn/@ delegation) + LLM judge + execution
  check on 5 random inputs.
- ⚠️ Survey's verdict: reward hacking + weak generalization are THE
  reliability problem of the field.

## Design rules for OUR auto-research loop (synthesis)

1. **Verification is the system.** Execution check on multiple random seeds
   + linter for delegation/hardcoding + reference isolation (fresh memory /
   separate process) + perturbation test. Full checklist:
   [benchmarking-discipline.md](benchmarking-discipline.md). Validation
   harness in a separate trust domain from the generating agent.
2. **Iterate with measurement + profiling feedback**, not one-shot: feedback
   ≈ doubles success; serial refinement beats parallel sampling at fixed
   budget (16×8 turns: 1.10× vs 128×1: 0.65× ⚠️ Kevin).
3. **Time-box hard**: diminishing returns by round 3–4; cap ~8 iterations;
   keep/revert with a ≥1.05× acceptance threshold; watch for oscillation
   (rewrites that change code but not perf) ⚠️.
4. **Grounding beats generation**: our KB slices, OSS kernel ports, and
   library calls give the agent strong priors — the literature's weakest
   results are ungrounded generation; strongest use retrieval/memory/
   profiling context (ReGraphT, KernelBlaster, TritonForge per survey ⚠️).
5. Speedup-weighted, correctness-gated objective (Kevin's S =
   0.3·1{correct} + speedup·1{correct} ⚠️) matches SOL-ExecBench's scoring
   shape — but OUR gate must be the un-hackable harness, not the agent's own
   check.
6. Expect the L2-collection (fusion) problems to be the hardest for
   automation — that's where the literature fails; route more human/manual
   attention there.
