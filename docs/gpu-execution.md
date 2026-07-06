# GPU execution (Phase F): running kernels in the harness on a rented B200

**Status: design.** The engine already treats the GPU as an interface
(`Executor.evaluate`), stubbed on the laptop. This is the *real* executor: get a
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
`solver solve` gains `--gpu <pod-config>`; without it, the StubExecutor path.

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

Provisioning is **manual** — you rent the pod on RunPod and drop its SSH details
in `.env`; the engine **sets the pod up over SSH**, **idempotently**, so first
contact does a full install and every reconnect is a fast verify. `solver gpu
setup` (also run automatically on first connect) does:

1. **Connect** — SSH creds + RunPod API key/pod-id from `.env`; verify SSH
   reachability *and* that the RunPod API reports the pod `RUNNING` (§6).
2. **Transfer** — `rsync` the self-contained `gpu-worker` module + the problem
   packs (`problems/`) to `~/solver-gpu/` on the pod. (No repo clone needed; the
   worker is one module. `git clone` of the solver repo is an alternative if the
   worker ever grows.)
3. **Install** — `pip install 'sol-execbench @ …'` with the `[bench]` extras
   (torch cu130, cutlass-dsl, cupti). Slow once; guarded by a version marker so
   warm pods skip it. One-time extension builds happen here (pip compiles).
4. **Start** — launch `gpu-worker` as a durable daemon (tmux/nohup/systemd)
   polling `jobs/`, with a heartbeat file for health.
5. **Ready** — append `gpu_rentals.jsonl{start}` and **reconcile** in-flight
   jobs (§4).

Every step is **checkpointed** (`~/solver-gpu/state/<step>.done`), so a dropped
connection resumes setup where it left off and a warm pod jumps to step 5.
Bootstrap is logged to `runs/gpu-setup.log`.

**Config — `.env` (gitignored; you fill it after renting):**
```
GPU_SSH_HOST=...            GPU_SSH_PORT=...       GPU_SSH_USER=root
GPU_SSH_KEY=~/.ssh/runpod   RUNPOD_API_KEY=...     RUNPOD_POD_ID=...
```
Loaded like the agent keys (`_load_dotenv`). `solver gpu status` prints SSH
reachability + the live pod details from the API; `solver gpu setup` (re)provisions.

## 4. Job queue + durability (no-loss / no-double-pay)

The design's guarantee (orchestration.md §7): a candidate that reached the GPU
is never lost and never re-run.

- **Hash-keyed, idempotent jobs.** `job_id = <task_id>-<Solution.hash()>`. The
  worker checks `results/<job_id>.json` *before* running — if present, it
  returns it without re-executing. So a resubmit (after any crash) is free.
- **State by directory** on the pod (`~/solver-gpu/`):
  `jobs/<id>.json` (pending) → atomic-rename → `processing/<id>` (claimed) →
  `results/<id>.json` (**fsync'd**, completed).
- **Laptop protocol** (inside `evaluate`): journal `execute_submitted{job_id,
  hash}` → push `jobs/<id>.json` → poll `results/<id>.json` → pull it into
  `runs/<id>/results/` (authoritative) → journal `exec_done`.
- **Reconciliation on (re)connect**, for every `execute_submitted` without an
  `exec_done`: result already local → recover (journal exec_done); result on pod
  → pull; job still `jobs/`/`processing/` → wait; none of these (pod wiped) →
  **resubmit** (idempotent, so at most a re-run of an *incomplete* eval).
- **Ephemeral caveat.** A completed result lives on the pod until pulled; the
  tight poll pulls it immediately, so a pod death after completion but before
  pull is the only re-pay window (rare). A **persistent volume** closes it
  (results survive pod restart).

## 5. Single-flight + compile off the GPU lock

The single GPU serializes **eval**, not compile (orchestration.md decision-log
#5). nvcc is CPU-bound and slow (seconds–minutes); holding the GPU lock through
it wastes the GPU.

- The executor's **asyncio single-flight lock wraps only the eval submit+poll**.
- **Compile runs off the lock**: a candidate's C++ build is submitted to a
  parallel build pool on the pod (N concurrent nvcc) *before* it takes the GPU
  lock; by the time it acquires the lock, the `.so` is ready. Python-family
  skips straight to eval.
- **`COMPILE_ERROR` never touches the GPU** → doesn't consume `max_gpu_evals`
  (already stated in §6). Build failures return early with the compiler log in
  `asi` for reflection.

**v1: combined** (build+eval as one job under the lock) for simplicity — the
worker builds then evals in one shot. The split (compile in a parallel off-lock
build pool) is **Deferred**, triggered when the logs show nvcc time dominating
GPU-idle. Python-family (torch/triton) has no build step, so the waste only
applies to C++-family candidates.

## 6. Pod lifecycle, rentals & health (RunPod API)

- **Connect** → §3b bootstrap → `gpu_rentals.jsonl{start}` → reconcile (§4).
- **Run**: submit / poll evals.
- **Health monitor (RunPod API)**: a background poller hits the RunPod API
  (`RUNPOD_API_KEY` from `.env`) every ~30s for the pod's **live state** —
  `desiredStatus`/`RUNNING`, `runtime.uptimeInSeconds`, GPU type, cost/hr. This
  is how we notice an **ephemeral/spot pod being preempted or stopped** *before*
  an SSH timeout would: the API status flips → we suspend cleanly rather than
  hang on a dead socket. (RunPod pods have no fixed expiry; the signal is
  status-change + uptime, not a countdown.)
- **Disconnect / death** (preemption, manual stop, SSH loss): append
  `gpu_rentals.jsonl{end}`; the fleet **suspends** (no GPU — the same clean
  suspend as credit-exhaustion, §2) and resumes when a new pod's `.env` is set
  and `solver gpu setup` runs. In-flight evals are reconciled / resubmitted on
  the next pod (§4); a death mid-run is crash-isolated.
- **Manual provisioning, API for reads only**: you rent/stop the pod on RunPod
  and fill `.env`; the API is used only to **read** instance state (status,
  uptime, GPU, cost) for the rental log and death-detection — never to
  create/destroy. Auto-provisioning (RunPod API create/terminate, spot bidding)
  is Deferred.
- `gpu_rentals.jsonl` is now **executor-written** from the API's uptime/status
  → the dashboard's rented-window utilization becomes real.

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

## 8. Calibration & env honesty

- **First real eval**: measure the **seed (reference)**, journal
  local-measured vs the website `T_b`; report raw + calibrated scores (search
  uses local relative numbers). The reference-as-seed (now built) makes this a
  natural, always-present calibration probe.
- **`env` fingerprint** (GPU / driver / clock / harness commit) on **every**
  result; cross-pod comparisons flagged; automated re-baselining deferred.

## 9. Transport mechanism

- **Primary: rsync file-queue** (`jobs/`, `results/` dirs synced over SSH).
  Robust, reconnect-safe, no custom protocol — just a polling worker + rsync.
- **Latency option: a persistent SSH session** running the worker with a
  line-protocol (job JSON in, result JSON out), *plus* the durable `results/`
  dir as the crash-recovery source of truth. Adopt if rsync-poll latency
  (~seconds) shows up against sub-second evals.
- sshfs (mount the pod dir) is possible but flaky under disconnect — not primary.

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
| F1 | `GpuQueueExecutor` + rsync file-queue + hash-keyed idempotent jobs + reconciliation + **LocalPod fake-harness tests (§10)** |
| F2 | **Idempotent SSH bootstrap (§3b)** (rsync worker + problems, install harness, start `gpu-worker`) + **RunPod API health/rental monitor (§6)** + `.env` config + `solver gpu setup`/`status` |
| F3 | Real `eval_driver` result parsing + calibration (§8) on an actual B200; then compile-off-lock split (§5) + sandbox hardening (§7) as data demands |

## Decision log

1. **Interface unchanged**: `GpuQueueExecutor` behind `Executor`; the engine is
   GPU-agnostic. Stub ↔ real is a config swap.
2. **Durability = hash-keyed idempotent jobs + laptop-authoritative results**;
   the pod is expendable. Reconcile on connect; resubmit only incomplete evals.
3. **Single-flight = eval only**; compile runs off the lock in a build pool.
4. **Transport = rsync file-queue** (chosen); persistent-SSH RPC deferred as a
   latency upgrade.
5. **Ephemeral pods** (chosen — cheapest); a persistent volume is the deferred
   upgrade for zero re-pay across pod restarts.
6. **Provisioning manual, API read-only**: you rent the pod and fill `.env`
   (SSH creds + `RUNPOD_API_KEY` + `RUNPOD_POD_ID`); the executor **sets it up
   over SSH idempotently** (§3b) and **reads** live pod state via the RunPod API
   (§6) for the rental log + death-detection. Auto-provisioning deferred.
7. **Workloads rsync'd** to the pod at bootstrap (chosen — network-free eval
   sandbox), not fetched pod-side.
8. **Build+eval combined** under the lock for v1 (chosen); off-lock compile
   split deferred.

## Deferred (with trigger)

- **Multi-GPU / multi-pod** (parallel evals) — when one B200's throughput caps
  the fleet; needs per-pod queues + the executor to fan out (breaks strict
  single-flight into per-GPU single-flight).
- **Persistent-SSH RPC transport** — when rsync-poll latency dominates.
- **Compile/eval split** — when build time shows up as GPU-idle in the logs.
- **Auto-provisioning** (RunPod API create/destroy on demand, spot bidding) —
  when manual pod management is the bottleneck.
- **Nsight profiling in the eval path** (orchestration.md deferred) — profile on
  plateau, on the pod.
