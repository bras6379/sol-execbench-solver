# GPU execution (Phase F): running kernels in the harness on a rented B200

**Status: F1 + F2 built and VALIDATED LIVE on a rented B200 (RunPod).** The full
path works autonomously: `solver solve --gpu 230 --agent claude --model haiku`
auto-provisioned a CUDA-13 B200, bootstrapped the pinned harness (`uv sync`
torch/cu13 + cutlass + cupti), scored candidates through the real `sol-execbench`
CLI, and terminated the pod — no orphans. The naive reference rmsnorm scored
~0.05; a fused Triton kernel (the agent found the same one) scored **0.54** at
26× the speed, so scoring tracks real performance. Covered by 42 laptop tests
(durability, teardown, Trace→EvalResult mapping, the solution/​trace helpers, and
the `SshExecutor` control flow against a fake pod) plus the live run above.
Real-GPU pieces: `SshExecutor` + `PodConn` (`ssh_exec.py`), `bootstrap` +
`solve_on_gpu` (`gpu_run.py`), the harness-format helpers (`harness.py`), and the
`RunPodProvider` create args (`pod.py`). Deferred (not needed for a working run):
network-volume harness caching (fresh `uv sync` each run, ~5 min), Nsight/​ncu
deep-profiling (§8b Tier 2), and the pod-side dead-man's-switch (laptop-side
teardown is the proven primary; §6).
The engine already treats the GPU as an interface (`Executor.evaluate`), stubbed
on the laptop. This is the *real* executor: get a
GPU (RunPod, over SSH), run each candidate in the SOL-ExecBench harness, return
measured latencies. Nothing in the engine changes — that's the point of the
interface (orchestration.md §2).

The hard problem isn't running the harness; it's the **transport** (laptop ↔
ephemeral pod) and its **durability** (a candidate that cost a GPU run is never
lost or re-paid), plus **compile-off-the-GPU-lock**, **sandboxing** untrusted
kernels, and **rental lifecycle**.

---

## 1. Where it plugs in — no engine change

`GpuQueueExecutor` implements the exact interface the loop already awaits:

```python
class GpuQueueExecutor:                       # swaps in for StubExecutor
    async def evaluate(self, solution, task_id, *, profile=False) -> EvalResult: ...
```

The loop, frontier, tiers, journal, dashboard, and §12 stub tests are untouched.
`solver solve --gpu` (a boolean; reads pod creds from `.env`, §3b) selects it;
without it, the StubExecutor path.

## 2. Topology

```
  LAPTOP  (engine + AUTHORITATIVE store)          RUNPOD  B200 · Linux · CUDA-13
  ─────────────────────────────────────           ────────────────────────────
  GpuQueueExecutor                                 gpu-worker (daemon)
   · single-flight asyncio lock (the GPU)           · claims jobs/, runs harness
   · submit(job) · poll(result) · reconcile         · build_ext.py  (compile, C++)
   · pulls results → runs/<id>/results/  ◀──rsync──▶ · eval_driver.py (cold-L2 timing)
                     (durable)             (SSH)     · writes results/<job>.json (fsync)
```

The pod is **ephemeral and expendable**; the **laptop is the durable store**
(runs/<id>/results/, §7). Everything on the pod can be rebuilt from the laptop.

## 3. The harness on the pod

`pip install 'sol-execbench @ …'` (the pinned dep). Two phases per candidate:
- **build** — `build_ext.py` compiles C++-family sources (cuda_cpp/cutlass/…) →
  `.so`. Python-family (pytorch/triton/cute_dsl) is JIT — no build step.
- **eval** — `eval_driver.py` runs all ~16 workload shapes: cold-L2 flush,
  10 warmup / 50 iters / seed 200, 99% matched-ratio, reward-hack defenses →
  one `Trace` per shape (status, latency, sol_ms, baseline, matched_ratio).

The **workloads/tolerances** come from each problem's dataset pack; we
**`rsync problems/` to the pod at bootstrap** (§3b) rather than fetch on the
pod, so the sandboxed eval never needs the network.

## 3b. Pod bootstrap & provisioning (setting up the GPU)

Provisioning is **automatic** (assuming credit on the account): `solver solve
--gpu` calls the RunPod API to **create an ephemeral pod for the run**, sets it
up over SSH **idempotently**, and **terminates it when done** (§6) — you never
touch the console, and you pay only for the run (per-second billing). First use
does a full install; with a network volume every start is a fast verify. Bootstrap:

1. **Connect** — the API `create_pod` returns the pod's SSH host/port; connect
   with your key. Verify SSH reachability *and* API status `RUNNING`. (Manual
   BYO fallback: set `GPU_SSH_*` in `.env` and skip create/terminate.)
2. **Transfer** — `rsync` the self-contained `gpu-worker` module + the problem
   packs (`problems/`) to `~/solver-gpu/`. (No repo clone; the worker is one
   module.)
3. **Install** — `pip install 'sol-execbench @ …'` with the `[bench]` extras
   (torch cu130, cutlass-dsl, cupti; `ncu` when Tier-2 profiling lands, §8b).
   Slow once; a version marker + a **network volume** skip it on warm starts.
4. **Start** — launch `gpu-worker` as a daemon polling `jobs/`, with a
   **heartbeat** the laptop feeds (the dead-man's-switch, §6).
5. **Ready** — append `gpu_rentals.jsonl{start}` and reconcile in-flight jobs (§4).

Every step is **checkpointed** (`~/solver-gpu/state/<step>.done`), so a dropped
connection resumes where it left off and a warm pod (volume) jumps to step 5.
Logged to `runs/gpu-setup.log`.

**What lives where** (this is why warm starts are seconds and per-eval payloads
are tiny):
- **Network volume** (set up once by `solver gpu init-volume`, survives every
  pod): the **installed harness venv** (`sol-execbench[bench]` + torch cu130 +
  cutlass + cupti + `ncu`) *and* the static **`problems/`** packs. Big,
  unchanging — never re-installed.
- **Per run** (rsync'd fresh, ~KB): just the `gpu-worker` module, so code
  updates land without re-initing the volume.
- **Per eval** (rides `jobs/<id>.json`, a few KB): just **the kernel** — the
  candidate's Solution sources. The worker materializes them into a workdir and
  runs them against *that* problem's workloads from the volume.

So a warm pod = attach volume → activate venv → start worker → ready in seconds;
`init-volume` (the slow pip install + problem sync) is a one-time cost.

**Config — `.env` (gitignored):**
```
RUNPOD_API_KEY=...            # auto-provision (create/terminate); the only thing you must set
GPU_TYPE="NVIDIA B200"        GPU_IMAGE=...            NETWORK_VOLUME_ID=...   # optional (fast starts)
GPU_MAX_LIFETIME_MIN=120      GPU_IDLE_TIMEOUT_MIN=15  GPU_SSH_KEY=~/.ssh/runpod
# manual BYO fallback: GPU_SSH_HOST / GPU_SSH_PORT / GPU_SSH_USER  (then create/terminate is skipped)
```
Loaded like the agent keys (`_load_dotenv`). `solver gpu status` shows the live
pod; `solver gpu reap` terminates stragglers (§6).

## 4. Job queue + durability (no-loss / no-double-pay)

The guarantee (orchestration.md §7): a candidate that reached the GPU is never
lost or re-run. It falls out of **hash-keyed idempotent jobs** + the engine's
existing resume — **no separate reconciliation pass, no new events**.

- **Idempotent job.** `job_id = <task_id>-<Solution.hash()>`. Both the executor
  (before pushing) and the worker (before running) check for an existing
  `results/<job_id>.json`; if present it's reused, never recomputed.
- **State by directory** on the pod (`~/solver-gpu/`): `jobs/<id>.json` (queued)
  → worker claims → `results/<id>.json` (**fsync'd**). Single-flight ⇒ ≤1 job in
  flight, so the "queue" is shallow (a real queue only matters at the deferred
  multi-GPU stage).
- **`evaluate`**: push `jobs/<id>` (skip if job/result already there) → poll
  `results/<id>` → pull into `runs/<id>/results/` (authoritative) → return.
- **Recovery is the resume path, for free.** The loop already journals
  `exec_enqueued` *before* awaiting `evaluate` — that is the durable "submitted"
  marker (no separate event). On a **laptop crash** mid-eval, resume
  re-evaluates the same candidate → same `job_id` → the cached pod result is
  recovered, **not re-run**. On a **pod death**, the result is gone → the same
  re-evaluate re-runs it fresh (**resubmit**). Deterministic agents make this
  exact; a real-agent re-plan may differ (a new job, the old result orphaned on
  the expendable pod) — the §12.1 *no-double-pay-of-the-same-eval* guarantee
  still holds.
- **Ephemeral caveat.** A completed result lives on the pod until the tight poll
  pulls it; a pod death in that narrow window re-runs. A persistent volume
  closes it (deferred).

**Result contract** — `results/<job_id>.json` deserializes into `EvalResult`:
```
{ task_id, solution_status: COMPILE_ERROR|REWARD_HACK|null, all_passed,
  sol_score, per_workload:[{index, status, latency_ms, sol_ms,
  baseline_latency_ms, matched_ratio}], env:{gpu,driver,cuda,harness_commit},
  asi:{logs, compile_log?} }
```
Per-shape `status` is what the frontier needs for specialists (non-PASSED → 0).

## 4b. The metric round-trip — what runs, what returns, how it pipes in

**On the GPU (per job `{task_id, solution}`):** the worker loads
`problems/<task>/` — `workload.jsonl` (~16 shapes: axes + inputs + tolerance),
`reference.py`, `metadata.json` — then `build_ext` (C++) → `eval_driver`, which
per shape does cold-L2 timing (10 warmup / 50 iters / seed 200) + correctness vs
the reference (matched-ratio ≥ 99% within tolerance) → one **Trace**:
`{status, latency_ms (T_k), matched_ratio}`. The **SOL targets are not measured**
— `sol_ms` (T_SOL) and `baseline_latency_ms` (T_b) are known dataset values in
`metadata.json` (e.g. task 230 shape 0: `sol_ms=0.0004`, `baseline=0.003712`).

**Scoring (worker-side; the formula already exists):** per shape
`S = 1/(1 + (T_k−T_SOL)/(T_b−T_SOL))` (`solver/scoring.py::sol_score`;
`score_from_metadata` maps measured latencies + the metadata `sol` block →
per-shape scores + mean). So a shape at `T_k=0.001ms` scores ≈0.85; matching the
baseline = 0.5; SOL = 1.0; a non-PASSED shape = 0.

**→ `EvalResult`** (`results/<job>.json`, §4), then **into the orchestrator —
identical to the stub path (already built + §12-tested):**
```
EvalResult ─.vector()─▶ per-shape sol_score (non-PASSED→0) ─▶ frontier.accept(Member)   ← specialists
   ├─ all_passed, sol_score ─▶ journal exec_done{…,scores,statuses} + accept{best,verdict,frontier}
   └───────────────────────▶ metrics.problem_metrics → dashboard (convergence · best · per-shape)
```
The per-shape **vector** is the ε-Pareto specialist signal; the **mean-of-S** is
the reported best/deliverable; **geomean-of-latencies** is recorded at finalize
(orchestration.md §9 aggregation). **Nothing downstream of `EvalResult` knows or
cares whether it came from the stub or the GPU** — the GPU only fills in the
measured `latency_ms`. F2's only new metric code is the pod-side worker turning
raw Traces + metadata into that `EvalResult` (≈ `eval_driver` +
`score_from_metadata`).

## 5. Single-flight, and compile

The single GPU serializes **eval**. In **v1, build+eval are one job under the
single-flight lock** — the worker builds (C++) then evals in one shot. Simple
and correct.

- The executor's **asyncio single-flight lock** (re-entrancy-asserted, reused
  from the stub) keeps ≤1 job in flight → one GPU run at a time.
- **`COMPILE_ERROR` never reaches the eval** → it doesn't consume
  `max_gpu_evals` (orchestration.md §6); the build failure returns early with the
  compiler log in `asi` for reflection. Python-family (torch/triton) has no
  build step.

**Deferred — compile off the lock.** nvcc is CPU-bound and slow, so holding the
GPU lock through a C++ build wastes the GPU. The refinement is a **parallel
off-lock build pool** on the pod (pre-compile a candidate's `.so` before it
takes the eval lock; orchestration.md decision-log #5). Triggered when logs show
nvcc time dominating GPU-idle; only C++-family candidates are affected.

## 6. Pod lifecycle — auto-provision, run, self-terminate (RunPod API)

`solver solve --gpu` owns the whole GPU lifecycle; you never open the console:

1. **Reap** — terminate any orphaned pods tagged from a prior crashed run
   (`solver gpu reap`, run on startup).
2. **Create** — `runpod.create_pod(gpu_type=GPU_TYPE, image=GPU_IMAGE,
   network_volume=…, name=<run-tag>)` → poll `get_pod` until `RUNNING` + SSH
   ready. B200 on-demand ~$5–6/hr, **per-second billing** — you pay only for the run.
3. **Bootstrap** (§3b) — fast if a volume holds the installed harness; full
   install only first time. `gpu_rentals.jsonl{start}`.
4. **Run** the fleet; pull each result to `runs/<id>/results/` (authoritative)
   as it lands.
5. **Terminate** — `runpod.terminate_pod()` in a `finally`; `gpu_rentals.jsonl{end}`.
   **Billing stops.**

**Never strand a pod** (a leaked pod silently drains credit). Layered teardown,
belt-and-suspenders:
- `finally` + `atexit` + SIGINT/SIGTERM handlers → terminate on *any* laptop exit.
- A **pod-side dead-man's-switch**: the `gpu-worker` self-terminates (via the
  RunPod API from the pod, or `shutdown`) if it hasn't seen a laptop **heartbeat**
  in `N` min — so even a hard laptop crash can't leave the pod running.
- **Caps**: `GPU_MAX_LIFETIME_MIN` (hard stop) and `GPU_IDLE_TIMEOUT_MIN`
  (no jobs → terminate) — a hung run auto-stops.
- **`solver gpu reap`** terminates stragglers by run-tag (recovery after a crash
  that skipped `finally`).

**Health monitor**: a background poller reads the RunPod API every ~30s
(`RUNNING`?, uptime, GPU, `costPerHr`) to detect a stop/preemption *before* an
SSH timeout would and to write rentals; on unexpected death the fleet **suspends**
cleanly (§2) and in-flight evals reconcile/resubmit on the next pod (§4).

**Cost**: create/terminate timestamps × the API's `costPerHr` = exact **GPU $
per run**, in `gpu_rentals.jsonl` + the dashboard (alongside agent tokens). A
cheap **network volume** ($0.07/GB·mo) optionally persists the installed env +
results for fast, cheap restarts; without it the pod is fully ephemeral (results
already pulled to the laptop before terminate, so nothing is lost).

**Manual BYO** (fallback): set `GPU_SSH_*` in `.env`; create/terminate/reap are
skipped, everything else identical.

## 7. Sandbox — untrusted kernels on the GPU box

The eval worker runs LLM-generated code (orchestration.md §11):
- **no network** (the harness/problems are pre-fetched; the eval namespace has
  networking disabled), **workdir-only FS**, **CPU/mem/time limits** (`ulimit`,
  a hard per-job timeout matching the harness), run under a **restricted user**
  or `bwrap`/container-in-pod.
- The harness's own **reward-hack** defenses stand; a `REWARD_HACK` trace is
  quarantined by the engine (never fed to reflection).
- The pod is already an isolated container; per-job sandboxing is defence in
  depth for the injection chain *web/agent → kernel → our GPU*.

## 8. Calibration & env honesty (measured against the real leaderboard)

**Confirmed root cause of the score gap = GPU clock state.** Submitting the
task-230 fused Triton kernel scored **0.481** on the leaderboard vs **~0.53** in
our CLI. cold-L2, median, warmup, iterations, and the harness commit already
match (it's the *same pinned `eval_driver`*), so none of those is the gap. The
harness pins the B200 to **1500/3996 MHz** while scoring (`device_config.py`
preset); we measured **unlocked** (boost ~1830 MHz) → latency ratio
`0.00695 / 0.008469 ≈ 1.22×`, exactly the boost/1500 clock ratio. So we run
~10-20% optimistic — a **roughly constant offset** that does not change kernel
ranking (the frontier stays correct).

**Why we can't just lock clocks:** a RunPod *container* has no privilege to set
GPU clocks, and the harness **rejects every workload** when `lock_clocks=True`
without real locking (`eval_driver.py:356`). So locking is not a knob we have on
RunPod — we removed the flag rather than keep a fallback that always fires.
Leaderboard-exact timing needs a privileged / bare-metal GPU.

What we do instead, to stay honest and calibratable:
- **Record the real conditions per eval.** `run_eval.sh` samples
  `clocks.sm / clocks.mem / temperature / power` *while the eval runs* and folds
  the summary into the result `asi.gpu` (median/max SM clock, etc.) — so every
  stored candidate carries the clock it was actually measured at. Cross-pod or
  anomalous-clock measurements are then visible, not silent.
- **Calibrate empirically against submissions.** The offset is measured, not
  guessed: our score vs the leaderboard's, per submission. The engine optimizes
  on the (consistent) local number; the **leaderboard is the authoritative gate**.
- **Reference-as-seed** is the always-present local probe (measure it every run).

## 8b. GPU observability & profiling — what we capture, and when

Latency says a kernel is slow; **profiling says *why*** — and "why" is what makes
`reflect()` (the text gradient) actionable ("memory-bound at 45% of peak BW,
register-limited occupancy → vectorize loads, cut registers" ≫ "it's slow").
Two rules from `kb/profiling-guide.md` shape the design:

- **`ncu` times ≠ leaderboard latency** (it locks clocks + flushes caches) — the
  **harness scores, `ncu` only diagnoses**. Never score off a profiler run.
- **`ncu` replay is slow** (10–100×) — profile **one representative shape, on
  demand**, not every eval.

The leaderboard scores **latency** (geomean, §4b) — that's what we optimize
*for*. Everything below is what the agent needs to *get there*, plus what the
score never shows (memory footprint, *how* wrong a failure is, measurement
noise). This mirrors 2026 kernel-agent practice — roofline-first bottleneck
classification then profiling-guided edits (KernelPro, AutoKernel; "micro-
profiling as expert surrogate for LLM kernel optimization", arXiv 2606.26453).

So GPU-side data is **two cost tiers**:

**Tier 1 — every eval (cheap; rides `EvalResult.asi`):**
- *Score data* (built): per-shape `status`, `latency_ms`, `matched_ratio`, sol_score.
- *Timing quality*: per-shape latency **variance** (std / CV, p50/p95/p99,
  n_trials) from the harness iters — so the engine knows if a 1–3% delta is real
  or noise (`kb/benchmarking-discipline.md`); what a future per-shape ε /
  confirm-before-promote keys on (`latency_spread` field already reserved).
- *Clock/thermal*: SM clock, power, temperature during the run (`nvidia-smi`) —
  catch a "slow" result that's actually throttling, not the kernel.
- *Correctness detail* (when not PASSED): max abs/rel error, ULP, mismatch
  fraction, which output — turns a bare `INCORRECT_NUMERICAL` into "off by 1e-3
  (bf16 accumulation order)" vs "wrong shape/dtype" vs "NaN".
- *Memory footprint*: peak GPU memory (`torch.cuda.max_memory_allocated`, reset
  per iter; `nvidia-smi` for the true peak incl. allocator fragmentation).
  **Not scored** by SOL-ExecBench (latency is), but a **constraint** (OOM →
  `RUNTIME_ERROR`) and a **lever** — excess intermediates flag a fusion
  opportunity, and a smaller footprint buys bigger tiles / higher occupancy.
- *Build info* (C++): `ptxas -v` — registers/thread, shared-mem/block, **spills**,
  warnings, nvcc time; + static launch dims → theoretical occupancy + wave/tile
  quantization (do we fill the 148 SMs?). All from `build_ext`, no profiler.

**Tier 2 — deep profile, on demand (expensive; a separate profile job):**
`ncu --set detailed -o report.ncu-rep` on **one representative shape**, then the
top-down triage (`kb/profiling-guide.md`):
- **SOL**: SM% vs MEM% → the **bottleneck class** (compute- / memory- /
  latency-bound).
- Achieved **DRAM bandwidth %** (vs ~7.5 TB/s), **tensor-pipe %** (TC-bound),
  **L2 hit rate**.
- **Achieved occupancy** + limiter (registers / shared / blocks).
- **Warp-stall** breakdown (top reason) — second-order, only if issue-starved.
- `ncu`'s own **Recommendations** with *Est. Speedup* — the ranked "try next".
- B200: **TMEM / TMA / cluster** usage, tile-quantization flags.
We **digest** this into a compact, agent-readable summary (the ASI), not the raw
report; the raw `.ncu-rep` is persisted to the candidate dir for the human.

**Into the loop:**
- Tier 1 rides every `EvalResult.asi` → `reflect()` gets rich why-signal, and the
  variance gates promotion.
- Tier 2 is **profile-on-plateau** (orchestration.md deferred): when a problem
  stalls, profile its frontier-best → the bottleneck digest enters the next
  `plan()`'s context and family knowledge → the agent targets the actual ceiling.
  Gated because `ncu` is slow.
- `ncu` joins the pod-bootstrap install (§3b) when Tier 2 lands.

**v1 posture**: **Tier 1 in F2** (cheap, always-on, high-value for reflection);
**Tier 2 (`ncu`) deferred** to F3+/on-plateau — the existing "Nsight → ASI"
deferred item, now specified.

## 8c. The `asi` schema — every-eval diagnostic bundle (F2 checklist)

Per-shape *scored* fields stay on `WorkloadResult` (index · status · `latency_ms`
· `latency_spread` · `sol_ms` · `baseline_latency_ms` · `matched_ratio`); `asi`
carries the cheap Tier-1 diagnostics `reflect()` reads. The worker fills what's
available and leaves the rest null.

```
asi = {
  stage:  "gpu",
  env:    { gpu, driver, cuda, harness_commit },        # the §10 fingerprint (§4 lists it top-level; F2 picks)
  timing: { n_trials, cv, cold_l2: true },              # per-shape std → WorkloadResult.latency_spread
  memory: { peak_alloc_mb, peak_reserved_mb },          # max_memory_allocated / nvidia-smi — NOT scored
  device: { sm_clock_mhz, mem_clock_mhz, power_w, temp_c, clock_locked },   # throttle detection
  launch: { grid, block, registers_per_thread, theoretical_occupancy,
            sm_fill_frac, tile_quantized },             # static; no profiler; do we fill 148 SMs?
  build:  { registers_per_thread, shared_mem_bytes, spill_stores, spill_loads,
            ptxas_warnings:[…], nvcc_seconds } | null,  # C++ only (ptxas -v)
  correctness: [ { index, kind:"numerical|shape|dtype|nan|inf",
                   max_abs_err, max_rel_err, mismatch_frac } ],   # only non-PASSED shapes; else []
  logs:   { stdout_tail, stderr_tail, compile_log },    # bounded (~KB) — rides the journal
  profile: null,                                        # Tier 2 (§8b), filled only on profile-on-plateau
}
```
- **Always**: `env`, `timing`, `memory`, `device`, `launch`, `logs`.
  **Conditional**: `build` (C++), `correctness` (failed shapes), `profile`
  (plateau). Everything **bounded** so it fits the journal.
- The bundle is what the loop feeds `reflect()`; the family curator distills
  recurring patterns ("rmsnorm keeps spilling at tile 256").

**Tier-2 `profile` sub-schema** (populated by the on-plateau `ncu` job):
```
profile = {
  bottleneck: "memory|compute|latency|mixed",           # the SOL classifier
  sm_pct, mem_pct, dram_bw_pct, l2_hit_pct, tensor_pipe_pct,
  achieved_occupancy, occupancy_limiter: "registers|shared|blocks",
  top_stall: "long_scoreboard|barrier|mio|…",
  recommendations: [ { rule, est_speedup } ],           # ncu's ranked "try next"
  ncu_report: "candidates/<cand>/report.ncu-rep",       # raw, for the human
}
```

## 9. Transport mechanism

- **rsync/ssh file-queue** (chosen): `jobs/`/`results/` synced over SSH. Robust,
  reconnect-safe, no custom protocol — a polling laptop + a polling worker.
- **SSH `ControlMaster` (connection multiplexing) is essential.** Without it a
  fresh SSH handshake per poll/pull is seconds of GPU-idle *per eval* on the one
  GPU. With a persistent master socket, poll via `ssh cat results/<id>` and pull
  via rsync are both cheap; tune the poll interval to the eval time.
- **Two timeout layers**: the harness's per-workload `TIMEOUT` (a slow kernel),
  and the executor's **per-eval wall-clock timeout** (a hung worker/kernel that
  never writes `results/`) → the job is declared stuck → resubmit/fail.
- **Deferred**: a persistent-SSH line-protocol RPC (job in / result out) if even
  multiplexed poll latency dominates; sshfs (flaky under disconnect).

## 10. Testability — no real pod, no CUDA

Same stub discipline as the engine (§12): make the transport testable on the
laptop.

- **LocalPodExecutor**: run `gpu-worker` on `localhost` (or a mock-ssh that
  `exec`s locally) against a **fake harness** that returns synthetic Traces
  (hash-keyed, like `hash_score_outcome`) — no CUDA. Tests the queue, single-
  flight, durability, and reconciliation deterministically.
- **Acceptance tests**: (a) kill the laptop mid-eval → resume recovers the
  result, no re-run; (b) kill the pod mid-eval → resubmit on reconnect, result
  idempotent, no double-pay; (c) single-flight holds under concurrent problems
  (the re-entrancy assertion, reused); (d) `COMPILE_ERROR` returns early without
  a GPU eval.
- A **real-harness-on-localhost** smoke (if a CUDA box is handy) validates the
  eval_driver parsing before renting a B200.

## 11. Build order

| Phase | Piece |
|---|---|
| F1 | `GpuQueueExecutor` + file-queue transport + hash-keyed idempotent jobs + resume-based recovery + **LocalPod fake-harness tests (§10)** — all laptop-testable, no GPU |
| F2a ✅ | Pod lifecycle + **guaranteed teardown** (§6; `PodSession`/`MockProvider`, laptop-tested) + harness **Trace→EvalResult mapping** (§4b; metadata + fake driver) |
| F2b ✅ | **Live on a B200.** `SshExecutor` + `PodConn` (rsync/ssh, single-flight, idempotent job dirs) instead of a pod-side worker — all scoring laptop-authoritative; `bootstrap` (§3b, validated recipe); the real driver = the `sol-execbench` **CLI** (`run_eval.sh`) + `solution_to_harness_json`/`traces_from_jsonl`; **`solver solve --gpu`** (`solve_on_gpu`: provision→bootstrap→run→terminate). Deferred: `init-volume` caching, RunPod cost monitor, pod-side dead-man's-switch (laptop teardown is primary) |
| F3 | Timing calibration vs the leaderboard (§8; cold-L2/clock-lock, our CLI latency vs the published `sol_ms`/`baseline`), network-volume harness caching, ncu deep-profiling (§8b), compile-off-lock split (§5) + sandbox hardening (§7) as data demands |

## Decision log

1. **Interface unchanged**: `GpuQueueExecutor` behind `Executor`; the engine is
   GPU-agnostic. Stub ↔ real is a config swap.
2. **Durability = hash-keyed idempotent jobs + laptop-authoritative results**;
   the pod is expendable. Reconcile on connect; resubmit only incomplete evals.
3. **Single-flight**: ≤1 eval at a time (re-entrancy-asserted lock); **v1
   combines build+eval** under it — the off-lock compile split is deferred.
4. **Transport = rsync file-queue** (chosen); persistent-SSH RPC deferred as a
   latency upgrade.
5. **Ephemeral pods** (chosen — cheapest); a persistent volume is the deferred
   upgrade for zero re-pay across pod restarts.
6. **Auto-provisioned, self-terminating pods** (default): `solver solve --gpu`
   creates an ephemeral pod via the RunPod API, bootstraps it (§3b), runs, and
   **terminates it in a `finally`** — with a pod-side dead-man's-switch +
   lifetime/idle caps + `reap` so a crash never strands a paid pod (§6). You pay
   only for the run. Manual BYO (`GPU_SSH_*`) is the fallback.
7. **Workloads rsync'd** to the pod at bootstrap (chosen — network-free eval
   sandbox), not fetched pod-side.
8. **GPU cost** tracked from the RunPod API `costPerHr` × rental windows
   (`gpu_rentals`), reported alongside agent tokens.

## Deferred (with trigger)

- **Multi-GPU / multi-pod** (parallel evals) — when one B200's throughput caps
  the fleet; needs per-pod queues + the executor to fan out (breaks strict
  single-flight into per-GPU single-flight).
- **Persistent-SSH RPC transport** — when rsync-poll latency dominates.
- **Compile/eval split** — when build time shows up as GPU-idle in the logs.
- **Spot / interruptible pods** (cheaper than on-demand but preemptible
  mid-run) — auto-provisioning already does on-demand create/terminate;
  spot-bidding is the cost cut once the reconcile path is battle-tested.
- **Nsight profiling in the eval path** (orchestration.md deferred) — profile on
  plateau, on the pod.
