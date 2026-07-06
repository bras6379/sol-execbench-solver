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

The **workloads/tolerances** come from each problem's dataset pack. Sync the
problems once at bootstrap (`rsync problems/ pod:`), or let the pod fetch from
HF at bootstrap and cache them — so the sandboxed eval needs no network.

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

*v1 may start combined* (build+eval as one job under the lock) for simplicity;
split it once logs show compile time dominating GPU-idle. (Deferred trigger.)

## 6. Pod lifecycle & rentals

- **Connect / bootstrap**: provision a pod (RunPod API or manual), read SSH
  creds; install the harness, sync problems, start `gpu-worker`; append
  `gpu_rentals.jsonl{start,label}`; **reconcile** in-flight jobs (§4).
- **Run**: submit / poll.
- **Disconnect** (rental ends / pod dies): append `gpu_rentals.jsonl{end}`; the
  fleet **suspends** (no GPU) and resumes on the next pod — same clean suspend
  as credit-exhaustion (§2). A pod death mid-run is crash-isolated; reconcile on
  the next pod.
- `gpu_rentals.jsonl` becomes **executor-written** (was hand-written) → the
  dashboard's rented-window utilization is finally real.

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
| F2 | Real pod bootstrap (RunPod: provision, install harness, sync problems, `gpu-worker`) + `eval_driver` result parsing + calibration + `gpu_rentals` writing |
| F3 | Compile-off-lock split (§5) + sandbox hardening (§7) + rental automation / suspend-resume |

## Decision log

1. **Interface unchanged**: `GpuQueueExecutor` behind `Executor`; the engine is
   GPU-agnostic. Stub ↔ real is a config swap.
2. **Durability = hash-keyed idempotent jobs + laptop-authoritative results**;
   the pod is expendable. Reconcile on connect; resubmit only incomplete evals.
3. **Single-flight = eval only**; compile runs off the lock in a build pool.
4. **Transport = rsync file-queue** first; persistent-SSH RPC as a latency
   upgrade.
5. **Ephemeral pods** by default (cheapest); a persistent volume is the upgrade
   for zero re-pay across pod restarts.

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
