# Oncall runbook — running the sol-execbench fleet

Operational playbook for babysitting a live `solver solve --gpu` fleet: what to
check, what's normal vs a real problem, and how to fix the incidents that have
actually happened. Complements `docs/gpu-execution.md` (architecture) and
`docs/orchestration.md` (engine design) — this doc is about running the thing,
not building it.

## 1. The standard health check

Run this every ~5 min while a fleet is live (this is what the recurring
health-check cron does):

1. **Fleet process**: `pgrep -f "solver solve --gpu"` — don't assume a fixed
   pid, it gets restarted often. If gone, tail its log file (buffering means
   `cat`/`tail` on the redirect target can show nothing for the first several
   minutes even though the process is fine — see §5). First ~10 min after a
   (re)start is bootstrap (pod creation, SSH, `uv sync`); no journal writes yet
   is normal, not a stall.
2. **Pod count**: load `.env`, then
   `RunPodProvider(api).list_tagged(PodSpec().tag)` (async). Confirm **exactly
   one** pod RUNNING. Never `solver reap` or `provider.terminate` while a run
   is live — that kills the pod and loses the in-flight GPU rental.
3. **Journal progress**: `runs/<task>/journal.jsonl` per problem. Check which
   problems have moved, best raw/calibrated scores, count of `solver_errors`
   / `flaky` / `ceiling_consensus` terminations / `no_op` outcomes. A
   `ceiling_consensus` termination is a healthy signal (the fix that auto-stops
   a problem after N consecutive no-op turns) — not a bug.
4. **Agent health**: are all pool models producing `plan_done` events? Cluster
   of `plan_error` across every OpenRouter-routed model (deepseek/kimi/glm) at
   once → check credits FIRST: `curl -s https://openrouter.ai/api/v1/credits
   -H "Authorization: Bearer $OPENROUTER_API_KEY"`. `total_usage >=
   total_credits` means the account is exhausted — every OpenRouter-routed
   call will 402 until more credit is added. `codex/gpt-5.5` failures are a
   *different*, unrelated provider (no shared root cause with OpenRouter).
5. **Laptop stress**: `pgrep -af "claude -p|codex exec"` count, `uptime`,
   `memory_pressure 2>/dev/null | tail -3`. Flag sustained load >12. A load
   spike isn't automatically fleet-caused — check process-level CPU/mem before
   blaming the fleet (this laptop has had real load spikes unrelated to it).
6. **2h cap on track**: pod self-terminates at launch-time + `--gpu-max-hours`.
   Shifts on every restart — recompute from the actual pid's start time
   (`ps -p <pid> -o lstart`), don't trust a stale cron description.
7. **Poll tracked leaderboard submissions**: `solver poll --all` (safe,
   read-only against the leaderboard API; never touches the fleet/pod).

**When cron descriptions go stale**: the recurring health-check/leaderboard
crons carry a fixed prompt describing "the fleet" as of whenever they were set
up (problem range, model pool, expected cap time). The fleet gets restarted
with different problems/pools far more often than the cron text is updated —
always verify against the ACTUAL running process (`pgrep`, `ps -p <pid>
-o lstart`, the actual `--tier`/`--gpu` flags in its command line), not the
cron's stale description. Say so plainly in the report when they've diverged.

## 2. Restarting the fleet

Needed whenever a flag/code change should take effect (Python doesn't
hot-reload a running process) — e.g. switching `--reflect-model`, tightening
`--review`, changing the model pool.

1. `git status` — never discard uncommitted work by accident.
2. `kill -TERM <pid>`. This is safe: `PodSession` (`solver/engine/pod.py`)
   arms a SIGINT/SIGTERM handler (`_arm_last_resort`) plus an `atexit` hook
   that synchronously terminates the pod even on a hard kill — confirmed live
   (2026-07-08): SIGTERM → pod gone within ~3s, verified via
   `list_tagged`.
3. **Verify pod teardown** via `list_tagged` before relaunching — should be
   `count: 0`. If it isn't, something didn't clean up; investigate before
   spending on a second pod.
4. **Sweep orphaned agent CLI processes** — a SIGTERM'd fleet can leave
   headless `claude -p ...`/`codex exec ...` calls running (`ppid=1`):
   ```
   ps -o pid,ppid,command -u $(whoami) | grep "claude -p\|codex exec" \
     | grep -v grep | awk '$2==1{print $1}' | xargs -r kill -TERM
   ```
   Verify these are genuinely headless (ppid=1) before killing — never kill
   your own interactive Claude Code session.
5. Relaunch with `nohup ... > logfile 2>&1 & disown`, then verify:
   `ps -p <newpid> -o pid,etime` and `list_tagged` shows exactly one new pod.
6. **The new log file will be EMPTY for a while** — see §5, this is expected,
   not a sign the launch failed. Cross-check via `runs/_active.json` (a live
   heartbeat the engine writes) or `list_tagged` instead of waiting on the log.

## 3. Incident: GPU executor stalled (queue backed up, nothing executing)

**Symptom**: several problems sit at `exec_enqueued` with no `exec_started`
for many minutes; `runs/_active.json`'s `ts` is fresh (engine loop is alive)
but no new `exec_done` events land anywhere.

**Cause (confirmed live, 2026-07-08, problem 44)**: the GPU executor is
single-flight by design (only one eval runs at a time — required so two
kernels never share the GPU and corrupt each other's latency measurement).
One candidate's `sol-execbench` invocation ran all the way to its `--timeout`
ceiling (was 600s) before failing `COMPILE_ERROR`, and every other queued
problem waited behind it the whole time.

**How to diagnose**:
1. `runs/_active.json` — if `ts` is recent, the engine loop itself is fine;
   the stall is specifically in the GPU executor.
2. SSH directly into the pod and check what's actually running:
   ```
   ssh -p <port> root@<host> "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv; \
     ps aux | grep -i 'run_eval\|eval_driver' | grep -v grep"
   ```
   `0% utilization` + a long-running `eval_driver.py` with high cumulative
   CPU time (`ps -o pid,etimes,time`) confirms a compile-bound stall, not a
   crashed/hung-forever process — the harness's own `--timeout` will still
   fire and the queue will self-unblock; it isn't infinite.
3. Cross-check the problem's own `journal.jsonl` for the pending `plan_done`
   candidate's design text — an ambitious "custom fused megakernel" approach
   is the highest compile-risk pattern (the review gate can't catch this: it's
   a static code read with no compiler, so "will this JIT-compile in time" is
   fundamentally invisible to it).

**Fix applied**: `solver/engine/ssh_exec.py`'s `RUN_EVAL_SH` lowered
`--compile-timeout 180 --timeout 600` → `90`/`300` (2026-07-08) so a
pathological compile now costs at most 5 min of blocked queue instead of 10.
See the `TODO(compile-off-the-lock)` comment right above `RUN_EVAL_SH` for the
deferred bigger fix (a second cheap GPU as a compile/correctness smoke-test
gate, decoupled from the single-flight B200 timing lock) — worth doing only if
compile stalls turn out to be a recurring pattern, not a one-off.

**Do NOT**: reap/terminate the pod over this — it resolves on its own once the
harness's timeout fires. Just wait it out (bounded now to ≤5 min) and confirm
the queue drains afterward.

## 4. Incident: a provider's credits/quota are exhausted

There are THREE independent billing accounts in play — OpenRouter (deepseek/
kimi/glm), OpenAI/codex (gpt-5.5), and the user's own Claude subscription
(claude/sonnet, claude/haiku). Any one of them can run dry without affecting
the others — don't assume a failure on one model implicates the others.

**Symptom**: `plan_error` cluster on the affected model(s); the journal's
`error` field was, until 2026-07-08, often uselessly empty
(`"wrote no kernel (exit 1): "`) — `CliAgent.plan()` only read subprocess
**stderr**, but both codex and claude report real failures (quota, auth, rate
limits) as a JSON event on **stdout**, so the actual cause was invisible
without reading `trajectory.jsonl` by hand. Fixed: `_extract_error_hint()` in
`cli_agent.py` now pulls the real message (codex's `{"type":"error",
"message":...}` / `"turn.failed"`, claude's `{"type":"result","is_error":true,
"result":...}`) into the exception — check the journal's `error` field first
now, it should be self-explanatory.

**Diagnose**:
- OpenRouter: `curl -s https://openrouter.ai/api/v1/credits -H "Authorization:
  Bearer $OPENROUTER_API_KEY"` → `total_usage >= total_credits` confirms it.
  Confirmed recurring in one session (burn rate has hit ~$27/40min); budget
  accordingly without an auto-recharge in place.
- OpenAI/codex: confirmed live (2026-07-07) — raw trajectory showed
  `{"type":"error","message":"Quota exceeded. Check your plan and billing
  details."}`. This happened at the SAME time as an OpenRouter exhaustion,
  which is what made the original ~40% `gpt-5.5` `plan_error` rate look like
  a mysterious unrelated bug — it wasn't; it was quota on a third, separate
  account failing independently.
- Claude subscription: no credits API to poll — watch the session-limit
  message (`"You've hit your session limit · resets <time>"`) in a
  trajectory, or just track cumulative `cost_usd` from the journal (§ below)
  against known plan limits.

**Fix**: top up the exhausted account, or reroute the affected role(s) to
whichever provider still has headroom. If falling back to native Claude
(`claude/sonnet`, `claude/haiku`), remember to change `--reflect-model` too,
not just the writer/reviewer `--tier` pool — reflection is a separate flag and
easy to forget (a live incident where reflection was silently 100%-failing
for ~10 min because it was still pointed at
`openrouter/deepseek/deepseek-v4-pro` after the rest of the pool moved to
Claude). If ALL THREE are exhausted at once, there is currently no cheap
fallback — that's a real "wait for credit or pause the fleet" situation, not
something to route around.

## 5. Gotcha: stdout buffering on redirected logs

`nohup solver solve ... > file.log 2>&1 &` fully block-buffers Python's
stdout (vs. line-buffering to a terminal) — a freshly-launched fleet's log
file can sit completely empty for minutes even though the process, pod, and
journal are all healthy. **Don't diagnose "no log output" as a hang** — check
`ps -p <pid>`, `list_tagged`, and `runs/<task>/journal.jsonl` / `runs/
_active.json` instead, which update immediately regardless of stdout
buffering.

## 6. Gotcha: stale dashboard watcher

A background `solver report --runs-dir runs --out-dir out --watch 15` holds
the OLD Python code in memory. After **any** edit to `solver/dashboard/*.py`,
the running watcher will keep silently re-rendering with stale logic until
it's killed and relaunched:
```
pkill -f "solver report --runs-dir runs --out-dir out --watch"
nohup solver report --runs-dir runs --out-dir out --watch 15 > dash.log 2>&1 & disown
```
Same principle applies to the live `solver solve` process for any
`solver/engine/*.py` change (§2 restart procedure) — it does NOT hot-reload.

## 7. Known display/attribution quirks (not bugs)

- **Coach diagnosis header names whichever model actually produced it**
  (`## Coach — expert diagnosis (via <model>)`), which can look surprising —
  e.g. showing `openrouter:deepseek/deepseek-v4-pro` well after `--reflect-model`
  was switched to `claude-sonnet-5`. That's correct: it's the diagnosis's own
  historical `diagnosis.json` timestamp/model, not live-recomputed. Check the
  file's mtime vs. the current fleet's start time before assuming it's stale
  or wrong. The "how the diagnosis evolved" history table on a problem's page
  is a frozen point-in-time capture — old snapshots legitimately show
  whatever label was live when they were written (including the old hardcoded
  "(fable-5)" text from before this was fixed), and are not retroactively
  rewritten.
- **`--max-concurrency` < total problem count**: the Nth-and-later problems
  queue-starved, not stuck — they simply haven't been dispatched a slot yet.
  Only resolves once another problem reaches a terminal state (accept,
  ceiling_consensus, budget cap) and frees its slot.

## 8. Known cost/coverage gaps (tracked, not yet fixed)

- **codex/gpt-5.5 spend is invisible** in `$` cost tracking — the codex CLI's
  stream-json schema never reports a cost field (only `cli_agent.py`'s claude
  schema branch does), so `total_cost`/`cost_by_model` undercounts true spend
  whenever gpt-5.5 is in the pool.
- **Review gate can't catch two whole classes of failure**: (1) numeric
  correctness that only shows up at runtime (race conditions, accumulation
  order) — it's a static code read with no GPU; (2) compile-time failures —
  it has no compiler either. Both require an actual GPU eval to surface.
  Measured live: ~71% of candidates the reviewer explicitly "shipped" still
  failed on the real GPU/tolerance check — treat review as a cheap partial
  filter, not a correctness guarantee.
- **`_rekey_workdir` collision handling** (fixed 2026-07-08, `cli_agent.py`):
  used to `rmtree` a candidate's own trajectory whenever its hash collided
  with an existing one (always true for a no-op). Now preserved as
  `trajectory.dup-N.jsonl` — if debugging an OLD no-op from before this fix,
  the evidence genuinely doesn't exist.

## Quick reference — commands

```bash
# health check essentials
pgrep -af "solver solve --gpu"
ps -p <pid> -o pid,etime,lstart
cat runs/_active.json | python3 -m json.tool

# pod listing (non-destructive)
set -a && source .env && set +a
python3 -c "
import asyncio, os
from solver.engine.pod import PodSpec, RunPodProvider
async def go():
    p = RunPodProvider(os.environ['RUNPOD_API_KEY'])
    for pod in await p.list_tagged(PodSpec().tag): print(pod)
asyncio.run(go())"

# OpenRouter credit check
curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY"

# leaderboard refresh (safe, read-only)
solver poll --all

# orphan agent-CLI sweep after a restart
ps -o pid,ppid,command -u $(whoami) | grep "claude -p\|codex exec" \
  | grep -v grep | awk '$2==1{print $1}' | xargs -r kill -TERM

# dashboard watcher refresh after any solver/dashboard/*.py edit
pkill -f "solver report --runs-dir runs --out-dir out --watch"
nohup solver report --runs-dir runs --out-dir out --watch 15 > dash.log 2>&1 & disown
```
