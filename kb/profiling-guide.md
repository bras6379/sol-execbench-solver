# Profiling Guide (Nsight Compute on B200)

Metric names, triage order, and profiler gotchas. Tags per kb/README.md.

## Triage order (guided top-down workflow)

1. **Speed of Light (SOL) section first** ✅ — per-unit (compute/memory)
   achieved throughput as % of theoretical peak, with sub-metric breakdowns
   identifying the top contributor. This is the bottleneck classifier:
   - High SM%, low MEM% → compute-bound; low SM%, high MEM% → memory-bound;
   - both low → latency/occupancy/launch-bound.
2. **Roofline chart** ✅ — `--set detailed/full` or
   `--section SpeedOfLight_RooflineChart` (hierarchical cache-aware variants
   like `SpeedOfLight_HierarchicalDoubleRooflineChart` in the `roofline`/
   `full` sets). The ceiling the kernel sits against = the bottleneck unit.
   If on the sloped (bandwidth) part → investigate memory access patterns
   first, not compute ✅.
3. **Prioritized Rules / Recommendations** ✅ — ncu auto-applies all rules on
   collection; recommendations carry **Est. Speedup** (global runtime
   decrease) and **Est. Local Speedup** (per-unit efficiency); the Summary
   page orders rules by estimated speedup. Use this ordering as the
   "what to try next" list.
4. **Warp stalls are second-order** ✅ (pass-2-verified in earlier batch):
   only analyze stall reasons if schedulers fail to issue every cycle.

## Metric cheat sheet ✅

| What | Metric |
|---|---|
| Compute (SM) throughput % | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| Memory throughput % | `gpu__compute_memory_throughput...` (SOL Memory) / DRAM: `dram__throughput...` |
| Naming scheme | throughputs: `.pct_of_peak_sustained_active` / `_elapsed`; counters: `.sum/.avg/.min/.max` |
| Occupancy limiters | `launch__occupancy_limit_registers`, `launch__occupancy_limit_shared_mem`, plus barriers/blocks/warps variants |
| Tensor-core utilization | SM pipe throughput breakdowns in SOL Compute (pipe_tensor) |

For per-bottleneck targets: memory-bound → achieved-bandwidth % (score
against ~7.5 TB/s real peak); TC-bound → tensor-pipe % of peak; L2-resident
work → L2 hit rate + L2 throughput (~21 TB/s measured ⚠️ on B200).

## Profiler-vs-reality gotchas ✅

ncu's defaults produce *controlled*, not *realistic*, numbers:

- **Locks SM clocks to base** (`--clock-control base`) and **flushes all GPU
  caches between replay passes** (`--cache-control all`). Nsight Systems does
  neither.
- Therefore ncu times ≠ leaderboard latency. One measured example ⚠️: a
  kernel at 676.6 µs under ncu ran 575 µs with `--clock-control none`.
- For A/B timing use the benchmark harness (CUDA events, warm cache,
  production clocks); use ncu for *why*, not *how fast*.
- When cache/clock realism matters inside ncu: `--cache-control none`,
  `--clock-control none`, or lock clocks externally via `nvidia-smi -lgc`
  (NVIDIA's own recommendation when comparing across tools ✅).

## B200-specific profiling notes

- Per-SM "feeds and speeds" accounting of NON-tensor units is mandatory for
  attention-class kernels ✅ — check SFU (xu pipe) and shared-memory
  throughput sections, not just tensor pipe: B200's asymmetric scaling makes
  these the real ceilings (see
  [optimization-recipe.md](optimization-recipe.md) Step 0).
- Wave/tile quantization: grid should fill 148 SMs (fewer with clusters);
  ncu flags tile-quantization inefficiency — keep matrix dims multiples of
  the tile sizes ⚠️.
- Remote-pod workflow: `ncu --set detailed -o report.ncu-rep` on the pod,
  copy the report back, read with `ncu -i` CLI or the UI locally. Profile a
  SINGLE representative workload shape per problem (ncu replay is slow);
  time all shapes with the harness instead.

## Reading the profile → action mapping ⚠️ (from the profiling-guided-loop paper)

- Memory stalls / low memory throughput % → better tiling/staging, coalescing.
- Low occupancy limited by registers (e.g. 12.5% with
  `launch__occupancy_limit_registers` binding) → reduce tile size / spill
  pressure.
- High throughput % on both SM and MEM but slow wall-clock → suspect launch
  overhead or serialization between kernels (Nsight *Systems* territory).
- Counter improvements don't always move wall-clock ⚠️ — always confirm with
  the harness measurement before keeping a change.
