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
  - capped concurrency, bounded timeout, and fully graceful: if the requested model
    is unavailable (no credits, rate-limited, CLI error), FALLBACK_CHAIN tries
    OpenRouter-routed models before giving up — one blocked provider never stalls
    the Coach; only total exhaustion leaves the deterministic card standing alone.

The result is persisted to `<task>/diagnosis.json` {fingerprint, model, ts, prose,
cost_usd, tok_in, tok_out, tok_cached} so `reflection.reflect_all` can re-attach it
to the card on every (cheap) pass without re-spending on fable.
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

# Fallback order when the primary reflection model is unavailable (rate-limited, out
# of credits, CLI error): try the requested model on native claude auth first, then
# fall through to OpenRouter-routed models already proven productive in this fleet.
# Mirrors the agent pool's own graceful-downgrade pattern (cli_agent._provider_env)
# so a blocked reflection model never stalls the Coach — it just costs a cheaper
# model's diagnosis instead of none at all.
FALLBACK_CHAIN: list[tuple[str, str]] = [
    ("claude", ""),                              # "" = use the caller's requested --reflect-model
    ("openrouter", "deepseek/deepseek-v4-pro"),
    ("openrouter", "moonshotai/kimi-k2.7-code"),
]

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

Relevant excerpts from the curated B200 optimization knowledge base (embedded directly below — \
you have NO file/tool access in this call, so reason from what's given here, don't try to Read \
anything else). Ground your recommendation in these where relevant and cite the file you used:
{kb_index}

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


def _provider_env(spec_name: str) -> dict | None:
    """Env for a fallback provider (ANTHROPIC_BASE_URL/AUTH_TOKEN via the same
    CliSpecs the agent pool uses), or None if its API key isn't configured — the
    caller skips that rung rather than crashing."""
    from .cli_agent import SPECS
    spec = SPECS.get(spec_name)
    if spec is None:
        return None
    if not spec.base_url:
        return dict(os.environ)                          # native claude auth
    key = os.environ.get(spec.api_key_env or "")
    return {**os.environ, "ANTHROPIC_BASE_URL": spec.base_url, "ANTHROPIC_AUTH_TOKEN": key} if key else None


def _add_usage(a: dict, b: dict) -> dict:
    return {k: (a.get(k) or 0) + (b.get(k) or 0) for k in (set(a) | set(b))}


async def _try_model(prompt: str, model: str, timeout: float, cwd: str | None,
                     env: dict) -> tuple[str | None, dict]:
    """One-shot claude-CLI call against a single (already-resolved) provider env.
    Returns (text_or_None, usage) where usage is {cost_usd, in, out, cached} — all
    0 whenever the CLI doesn't report them (e.g. the call failed before producing
    a result). None text on any failure — quota/credits, timeout, non-zero exit,
    empty output."""
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "stream-json",
           "--verbose", "--dangerously-skip-permissions"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=env, cwd=cwd)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        try:
            proc.kill()
        except Exception:
            pass
        return None, {}
    raw = out.decode("utf-8", "replace")
    usage = _extract_usage(raw)
    if proc.returncode:
        return None, usage                   # a failed call can still have billed something
    return _extract_text(raw), usage


async def _call_fable(prompt: str, model: str, timeout: float,
                      cwd: str | None = None) -> tuple[str | None, str, dict]:
    """Try the requested reflection model, then walk FALLBACK_CHAIN so a single
    blocked provider (rate limit, no credits) never stalls the Coach. `cwd` is
    optional and unused by the diagnose caller (kb content is embedded directly
    in the prompt — see _kb_index — so no file/tool access is needed for this
    call); kept as a parameter for callers that do want a specific cwd.
    Returns (prose_or_None, label of the model that actually produced it, total
    usage {cost_usd, in, out, cached} summed across every rung tried — including
    failed ones that billed)."""
    if not shutil.which("claude"):
        return None, "", {}
    seen: set[tuple[str, str]] = set()
    total_usage: dict = {}
    for spec_name, fallback_model in FALLBACK_CHAIN:
        m = fallback_model or model
        # Skip the native-claude rung entirely when the requested model isn't a
        # claude-style name (e.g. --reflect-model deepseek/deepseek-v4-pro) — avoids
        # a wasted round-trip against native auth for a model it was never meant to
        # serve, so a fully-OpenRouter reflect-model never touches Claude at all.
        if spec_name == "claude" and fallback_model == "" and not model.startswith("claude"):
            continue
        if (spec_name, m) in seen:
            continue
        seen.add((spec_name, m))
        env = _provider_env(spec_name)
        if env is None:
            continue                                     # required key missing — skip this rung
        text, usage = await _try_model(prompt, m, timeout, cwd, env)
        total_usage = _add_usage(total_usage, usage)
        if text:
            return text, (m if spec_name == "claude" else f"{spec_name}:{m}"), total_usage
    return None, "", total_usage


def _extract_usage(raw: str) -> dict:
    """Pull {cost_usd, in, out, cached} off the stream's terminal 'result' event —
    same schema as cli_agent.py's claude branch of _parse_tokens."""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            u = e.get("usage") or {}
            cache_read = u.get("cache_read_input_tokens") or 0
            cache_creation = u.get("cache_creation_input_tokens") or 0
            usage = {"in": u.get("input_tokens"), "out": u.get("output_tokens"),
                     "cached": (cache_read + cache_creation) or None,
                     "cost_usd": e.get("total_cost_usd")}
            return {k: v for k, v in usage.items() if v is not None}
    return {}


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


# Broadly useful across every op family — embedded directly in the prompt so the
# diagnosis never needs a tool-call round trip to read them. A prior version told
# the model to "use your file tools" on kb/ paths; live testing showed real
# diagnose calls (large prompt + multi-file tool use) reliably HUNG past a 240s
# timeout with zero output, while an identical trivial no-tool-use prompt
# completed in seconds. Embedding content directly removes the round trip
# entirely regardless of the exact cause — a single-turn text-in/text-out call.
_CORE_KB_DOCS = ("README.md", "optimization-playbook.md", "benchmark-grader.md")


def _kb_index(kb_dir: Path, budget: int = 6000) -> str:
    """Embed the CONTENT of a few broadly-useful kb docs directly (bounded by a
    char budget), not just a filename index — no tool access needed to use it."""
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        return "(no knowledge base found)"
    per_doc = budget // max(1, len(_CORE_KB_DOCS))
    parts = []
    for name in _CORE_KB_DOCS:
        f = kb_dir / name
        if not f.is_file():
            continue
        text = f.read_text(errors="replace")[:per_doc]
        parts.append(f"--- kb/{name} ---\n{text}")
    return "\n\n".join(parts) if parts else "(no knowledge base found)"


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
                       timeout: float, kb_dir: str | Path = "kb",
                       log=lambda *_: None) -> bool:
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
    kb_dir = Path(kb_dir)
    prompt = _PROMPT.format(
        task=r.task_id, name=r.name or "?", family=r.family or "?", status=r.status,
        card=(card.read_text(errors="replace") if card.is_file() else r.headline),
        best=("?" if r.best is None else f"{r.best:.3f}"),
        best_src=_best_source(pdir.parent, r), ref_src=_reference_source(pdir.parent, r.task_id),
        prior_src=_prior_sources(pdir.parent, r.task_id), kb_index=_kb_index(kb_dir))
    # No cwd pinned to the kb root anymore — kb content is embedded directly in the
    # prompt (see _kb_index), so this call needs no file/tool access at all; running
    # with no special cwd avoids inviting any exploratory tool use.
    prose, used_model, usage = await _call_fable(prompt, model, timeout)
    cost = usage.get("cost_usd", 0.0)
    # Cost/tokens are journaled into the PROBLEM's own journal.jsonl (same file the
    # dashboard already scans for plan/review costs) whether this attempt succeeded
    # or not — a failed-but-billed call (e.g. a 402 mid-fallback-chain) still needs
    # to show up in total spend, or the dashboard would silently under-count real cost.
    if usage:
        from .. import journal as journal_mod
        journal_mod.Journal(pdir / "journal.jsonl", r.task_id).append(
            "diagnose_cost", model=(used_model or model), cost_usd=cost, success=bool(prose),
            tok_in=usage.get("in"), tok_out=usage.get("out"), tok_cached=usage.get("cached"))
    if not prose:
        billed = f" (billed ${cost:.4f})" if cost else ""
        log(f"[diagnose] task {r.task_id}: {model} + all fallbacks unavailable/failed — "
            f"deterministic card stands{billed}")
        return False
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dfile.write_text(json.dumps({"fingerprint": fp, "model": used_model, "ts": ts,
                                 "prose": prose, "cost_usd": cost,
                                 "tok_in": usage.get("in"), "tok_out": usage.get("out"),
                                 "tok_cached": usage.get("cached")}))
    note = "" if used_model == model else f" [fell back from {model} to {used_model}]"
    log(f"[diagnose] task {r.task_id}: fresh diagnosis via {used_model} "
        f"({len(prose)} chars, ${cost:.4f}){note}")
    return True


async def diagnose_stuck(runs_dir: str | Path, refls: dict[int, R.ProblemReflection], *,
                         model: str = "claude-fable-5", timeout: float = 360,
                         kb_dir: str | Path = "kb", max_concurrency: int = 4,
                         progress=None, log=lambda *_: None) -> int:
    """Run fable on every STUCK problem whose state moved since its last diagnosis.
    Returns how many fresh diagnoses were written. After writing, re-attach the
    stored prose to each card so agents pick it up on the next plan. `progress(done)`
    is called after each problem completes (for a live 'reflecting X/Y' indicator)."""
    runs_dir = Path(runs_dir)
    stuck = [r for r in refls.values() if r.status in STUCK]
    if not stuck:
        return 0
    sem = asyncio.Semaphore(max(1, max_concurrency))
    done = 0

    async def _guard(r):
        nonlocal done
        async with sem:
            try:
                res = await diagnose_one(runs_dir, r, model=model, timeout=timeout,
                                         kb_dir=kb_dir, log=log)
            except Exception as exc:                    # one bad diagnosis never aborts the sweep
                log(f"[diagnose] task {r.task_id}: {exc!r}")
                res = False
        done += 1
        if progress:
            try:
                progress(done)
            except Exception:
                pass
        return res

    results = await asyncio.gather(*(_guard(r) for r in stuck))
    for r in refls.values():
        R.attach_diagnosis(runs_dir, r.task_id)
    n = sum(1 for x in results if x)
    log(f"[diagnose] {n} fresh diagnosis(es) across {len(stuck)} stuck problem(s)")
    return n
