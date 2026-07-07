"""Fable-5 expert diagnosis layer on top of the deterministic reflection cards.

The detectors in `reflection.py` say WHAT is stuck (regressing / rabbit-hole /
plateaued) and WHERE the loss is. This layer adds the WHY + the one untried lever,
in prose, from a strong model (claude-fable-5) reading the best kernel + the
reference + the attempt ledger. That prose is appended to the coach card and fed
to every agent.

Fable is expensive, so this is gated hard:
  - only STUCK problems (regressing / plateaued / rabbit_hole / broken) — never a
    problem that's still climbing on its own.
  - deduped on a state fingerprint (status · best · #evals): a 20-min tick only
    re-diagnoses problems whose state actually MOVED since the last diagnosis.
  - capped concurrency, bounded timeout, and fully graceful: if fable has no
    credits / errors / times out, the deterministic card stands alone (no crash).

The result is persisted to `<task>/diagnosis.json` {fingerprint, model, ts, prose}
so `reflection.reflect_all` can re-attach it to the card on every (cheap) pass
without re-spending on fable.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
from pathlib import Path

from . import reflection as R

STUCK = {"regressing", "plateaued", "rabbit_hole", "broken"}

_PROMPT = """You are a senior GPU performance engineer. An automated fleet is STUCK optimizing a \
kernel for NVIDIA B200 (SOL-ExecBench). Score = latency vs a PyTorch reference under a numeric \
tolerance; 0.5 = optimized-PyTorch baseline, 1.0 = hardware speed-of-light. Higher is better.

Problem #{task} ({name}), op family `{family}`. Fleet status: {status}.

What the deterministic reflection already found:
{card}

The current BEST kernel (score {best}) — this is what to beat:
```python
{best_src}
```

The PyTorch reference it must match (truncated):
```python
{ref_src}
```

The actual SOURCE of the top already-tried approaches (named by score). Do NOT trust the \
one-line strategy titles — READ THE CODE and derive for yourself what each really did, what it \
got wrong, and why it capped:
{prior_src}

In UNDER 220 words, blunt and concrete, output exactly these three sections:

**Root cause:** the bottleneck class (memory-bandwidth-bound / tensor-core-bound / \
launch-latency-bound) and WHY this specific op at these shapes is capped where it is.

**Spent — stop retrying:** which already-tried families (name them from the ledger) are \
genuinely dead-ends here, so agents quit re-deriving them.

**The one lever:** the single most promising UNTRIED direction, concrete enough to implement — \
name the technique and why it fits THIS op's shapes/tolerance. Prefer a rung from the untried list \
if one applies. If you believe the problem is at a true ceiling and no lever remains, say so plainly \
so the fleet stops spending evals here.

No preamble, no restating the question."""


def fingerprint(r: R.ProblemReflection) -> str:
    """State signature — a diagnosis is stale only when this moves."""
    return f"{r.status}:{r.best}:{r.n_evals}"


async def _call_fable(prompt: str, model: str, timeout: float) -> str | None:
    """One-shot claude CLI call (real Anthropic auth), returns the final text or
    None on any failure — quota/credits, timeout, non-zero exit, empty output."""
    if not shutil.which("claude"):
        return None
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "stream-json",
           "--verbose", "--dangerously-skip-permissions"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={**os.environ})
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        try:
            proc.kill()
        except Exception:
            pass
        return None
    if proc.returncode:
        return None
    return _extract_text(out.decode("utf-8", "replace"))


def _extract_text(raw: str) -> str | None:
    """Pull the assistant's final text out of the stream-json event stream."""
    result, chunks = None, []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result" and isinstance(e.get("result"), str):
            result = e["result"]
        elif e.get("type") == "assistant":
            for blk in (e.get("message", {}) or {}).get("content", []) or []:
                if blk.get("type") == "text" and blk.get("text"):
                    chunks.append(blk["text"])
    text = (result or "\n".join(chunks)).strip()
    return text or None


def _best_source(runs_dir: Path, r: R.ProblemReflection) -> str:
    cands = R._load_candidates(runs_dir / str(r.task_id))
    c = cands.get(r.best_cand) or {}
    srcs = (c.get("solution") or {}).get("sources") or []
    body = "\n\n".join(s.get("content", "") for s in srcs)
    return body[:6000] if body else "(source unavailable)"


def _prior_sources(runs_dir: Path, task_id: int, top: int = 4, budget: int = 9000) -> str:
    """The actual code of the top distinct-family prior attempts (staged under
    <task>/prior/ by reflection). Lets fable derive its own read from the source
    instead of trusting the strategy titles. Bounded to a token budget."""
    pdir = runs_dir / str(task_id) / "prior"
    if not pdir.is_dir():
        return "(no prior kernels staged)"
    files = sorted(pdir.glob("*.py"), key=lambda f: f.name, reverse=True)[:top]
    out, used = [], 0
    for f in files:
        try:
            body = f.read_text(errors="replace")
        except Exception:
            continue
        snippet = body[:budget // max(1, len(files))]
        out.append(f"```python\n# {f.stem}\n{snippet}\n```")
        used += len(snippet)
        if used >= budget:
            break
    return "\n\n".join(out) if out else "(no prior kernels staged)"


def _reference_source(runs_dir: Path, task_id: int) -> str:
    # any candidate's workdir carries the reference; grab the first we find
    wroot = runs_dir / str(task_id) / "work"
    if wroot.is_dir():
        for ref in wroot.glob("*/reference.py"):
            try:
                return ref.read_text(errors="replace")[:3500]
            except Exception:
                break
    return "(reference unavailable)"


async def diagnose_one(runs_dir: Path, r: R.ProblemReflection, *, model: str,
                       timeout: float, log=lambda *_: None) -> bool:
    """Diagnose one stuck problem with fable; persist to diagnosis.json. Returns
    True if a fresh diagnosis was written. Deduped on the state fingerprint."""
    pdir = Path(runs_dir) / str(r.task_id)
    fp = fingerprint(r)
    dfile = pdir / "diagnosis.json"
    if dfile.is_file():
        try:
            if json.loads(dfile.read_text()).get("fingerprint") == fp:
                return False                            # unchanged since last diagnosis → no spend
        except Exception:
            pass
    card = pdir / "reflection.md"
    prompt = _PROMPT.format(
        task=r.task_id, name=r.name or "?", family=r.family or "?", status=r.status,
        card=(card.read_text(errors="replace") if card.is_file() else r.headline),
        best=("?" if r.best is None else f"{r.best:.3f}"),
        best_src=_best_source(pdir.parent, r), ref_src=_reference_source(pdir.parent, r.task_id),
        prior_src=_prior_sources(pdir.parent, r.task_id))
    prose = await _call_fable(prompt, model, timeout)
    if not prose:
        log(f"[diagnose] task {r.task_id}: fable unavailable/failed — deterministic card stands")
        return False
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dfile.write_text(json.dumps({"fingerprint": fp, "model": model, "ts": ts, "prose": prose}))
    log(f"[diagnose] task {r.task_id}: fresh fable diagnosis ({len(prose)} chars)")
    return True


async def diagnose_stuck(runs_dir: str | Path, refls: dict[int, R.ProblemReflection], *,
                         model: str = "claude-fable-5", timeout: float = 240,
                         max_concurrency: int = 4, log=lambda *_: None) -> int:
    """Run fable on every STUCK problem whose state moved since its last diagnosis.
    Returns how many fresh diagnoses were written. After writing, re-attach the
    stored prose to each card so agents pick it up on the next plan."""
    runs_dir = Path(runs_dir)
    stuck = [r for r in refls.values() if r.status in STUCK]
    if not stuck:
        return 0
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _guard(r):
        async with sem:
            try:
                return await diagnose_one(runs_dir, r, model=model, timeout=timeout, log=log)
            except Exception as exc:                    # one bad diagnosis never aborts the sweep
                log(f"[diagnose] task {r.task_id}: {exc!r}")
                return False

    results = await asyncio.gather(*(_guard(r) for r in stuck))
    for r in refls.values():
        R.attach_diagnosis(runs_dir, r.task_id)
    n = sum(1 for x in results if x)
    log(f"[diagnose] {n} fresh diagnosis(es) across {len(stuck)} stuck problem(s)")
    return n
