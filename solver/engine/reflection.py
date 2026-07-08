"""Cross-run reflection: turn a problem's journal + candidates into a *coach card*.

The fleet runs ~30 problems as independent, amnesiac agents — each re-derives the
same ideas blind to what the other attempts already learned. Problem 3 is the
archetype: the *first* attempt scored 0.703 and the next 15 evals never beat it,
all re-deriving one family ("bf16 LM-head GEMM"), while two specific workloads
(#1, #12) carried the entire loss the whole time.

This module extracts that structure **deterministically** (no LLM) so it can be
injected back into every agent's context as a "you are stuck, here is what's been
tried and where the loss actually is" directive. A stronger (but CHEAP —
sonnet/haiku or an OpenRouter model, never an expensive frontier model) model
layers a *why + which untried lever* diagnosis on top (see `diagnose.py`), but
the signals below stand on their own and cost nothing.

Pure core: `analyze(events, candidates)` → `ProblemReflection`; `render_card()` →
markdown. `from_runs_dir()` is the thin IO loader.
"""

from __future__ import annotations

import datetime
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# The standard B200 optimization ladder, as coarse technique tags. Used two ways:
# (1) signature an attempt's strategy to detect family lock-in ("rabbit hole"), and
# (2) name ladder rungs that have NOT been tried yet (a factual nudge, not advice).
TECHNIQUES: dict[str, tuple[str, ...]] = {
    "cublas":      ("cublas", "cublaslt", "torch.matmul", "torch.mm", "f.linear", "addmm"),
    "triton":      ("triton", "tl.", "@triton"),
    "cuda_graph":  ("cuda graph", "cudagraph", "cuda-graph", "graph capture", "make_graphed"),
    "split_k":     ("split-k", "split k", "splitk", "stream-k", "streamk"),
    "fusion":      ("fuse", "fused", "epilogue", "one-shot", "megakernel", "single kernel"),
    "streaming":   ("streaming", "weight-once", "n-tiled", "grid-stride", "persistent"),
    "fp8":         ("fp8", "e4m3", "e5m2"),
    "nvfp4":       ("nvfp4", "fp4", "mxfp", "block-scal", "microscal"),
    "bf16":        ("bf16", "bfloat16"),
    "fp16":        ("fp16", "float16", "half"),
    "tf32":        ("tf32",),
    "vectorized":  ("vectoriz", "128-bit", "float4", "int4 pack", "packed"),
    "l2_persist":  ("l2 persist", "l2-persist", "carveout", "access policy"),
    "async_copy":  ("async copy", "cp.async", "async_copy", "pipeline", "num_stages"),
    "tma":         ("tma", "tensor memory accel", "cp.async.bulk"),
    "scatter":     ("scatter", "index_add", "atomic", "gather"),
    "reduction":   ("reduction", "reduce", "two-pass", "welford", "tree"),
    "slicing":     ("slice", "logits_to_keep", "last-row", "keep <", "collapse"),
}
# Order to surface UNTRIED rungs in — roughly "biggest lever first" for this bench.
_LADDER = ("fusion", "cuda_graph", "split_k", "fp16", "fp8", "nvfp4",
           "streaming", "vectorized", "async_copy", "l2_persist", "slicing")
# Rungs that only make sense on tensor-core / GEMM-bound ops — hidden for memory-bound
# elementwise/reduction ops (RoPE, norms, softmax), where they're dead ends.
_GEMM_ONLY_RUNGS = {"split_k", "fp8", "nvfp4"}
# Substring match, not exact-set membership: a real op key from op_key_of() is a
# compound string (e.g. "attention_output_projection_with_reshape_backward"), not
# a bare "attention" — exact membership would silently never match a real run.
_COMPUTE_KEYWORDS = ("gemm", "matmul", "attention", "moe", "conv", "linear",
                     "projection", "expert", "mlp", "swiglu", "geglu")


def _compute_bound(family: str) -> bool:
    return any(k in (family or "").lower() for k in _COMPUTE_KEYWORDS)


def relevant_techniques(family: str) -> tuple[str, ...]:
    """Ladder rungs applicable to this op's compute-bound class — a pure function
    of the op family alone, needing no run history (usable at design()-time,
    before any candidate exists, unlike ProblemReflection.techniques_untried)."""
    compute_bound = _compute_bound(family)
    return tuple(t for t in _LADDER if compute_bound or t not in _GEMM_ONLY_RUNGS)


def _clip(s: str, n: int) -> str:
    """Trim to ~n chars on a word boundary (no mid-word cut mid-sentence)."""
    s = " ".join((s or "").split())
    if len(s) <= n:
        return s
    cut = s[:n]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > n * 0.6 else cut).rstrip() + "…"

_STOP = {"a", "the", "an", "to", "of", "for", "and", "or", "with", "via", "in",
         "on", "by", "as", "one", "full", "single", "custom", "use", "using",
         "kernel", "gemm", "each", "then", "keep", "read", "reads"}


def _techniques(text: str) -> frozenset[str]:
    """Coarse technique tags present in a strategy string."""
    t = (text or "").lower()
    return frozenset(tag for tag, kws in TECHNIQUES.items() if any(k in t for k in kws))


def _sig(text: str) -> frozenset[str]:
    """A family signature: technique tags, else salient content words. Two attempts
    with the same signature are 'the same idea' for rabbit-hole detection."""
    tech = _techniques(text)
    if tech:
        return tech
    words = [w for w in re.findall(r"[a-z0-9\-]+", (text or "").lower())
             if len(w) > 2 and w not in _STOP]
    return frozenset(words[:4])


@dataclass
class Attempt:
    order: int
    score: float          # calibrated (leaderboard-est) score, or raw if uncal
    cand_id: str
    strategy: str
    model: str
    correct: bool


@dataclass
class ProblemReflection:
    task_id: int
    name: str = ""
    family: str = ""
    status: str = "unknown"        # climbing|plateaued|regressing|rabbit_hole|broken|thin
    n_evals: int = 0
    best: float | None = None
    best_order: int | None = None  # 1-based eval index that produced the best
    stale_evals: int = 0           # evals since the best (wasted exploration)
    best_strategy: str = ""
    best_cand: str = ""
    ledger: list[dict] = field(default_factory=list)     # distinct families + best score + n
    weak_workloads: list[dict] = field(default_factory=list)  # {index, score} dragging the mean
    workload_median: float | None = None
    techniques_tried: list[str] = field(default_factory=list)
    techniques_untried: list[str] = field(default_factory=list)
    dominant_family: tuple[str, ...] = ()
    dominant_share: float = 0.0
    failed: list[dict] = field(default_factory=list)  # approaches that FAILED correctness (don't retry)
    # Every (technique-tag, verdict) pair this run produced, for the cross-op
    # KnowledgeStore curator (knowledge.py) — same underlying attempts as
    # .ledger/.failed, flattened to single tags instead of family signatures.
    # {tag, kind: "confirmed"|"pitfall", score, error, cand, strategy}
    tech_events: list[dict] = field(default_factory=list)
    headline: str = ""             # one-line directive for the agent


def _dominant_error(candidates: dict, cand_id: str) -> str | None:
    """The raw dominant per-workload error code of a failed candidate
    (RUNTIME_ERROR / TOLERANCE / COMPILE_ERROR / …), or None if unknown."""
    from collections import Counter
    c = candidates.get(cand_id) or {}
    pw = c.get("per_workload") or []
    errs = [w.get("error") for w in pw if isinstance(w, dict) and w.get("correct") is False and w.get("error")]
    return Counter(errs).most_common(1)[0][0] if errs else None


def _cand_error(candidates: dict, cand_id: str) -> str:
    """The dominant per-workload error of a failed candidate (RUNTIME_ERROR /
    TOLERANCE / COMPILE_ERROR / …), so the card names the FAILURE MODE, not just
    'it failed'."""
    from collections import Counter
    c = candidates.get(cand_id) or {}
    pw = c.get("per_workload") or []
    errs = [w.get("error") for w in pw if isinstance(w, dict) and w.get("correct") is False and w.get("error")]
    if errs:
        err, k = Counter(errs).most_common(1)[0]
        return f"{err} (×{k})"
    return ""


def _failed_ledger(bad: list, candidates: dict | None = None) -> list[dict]:
    """Distinct approaches that FAILED correctness (score 0), grouped by family, with
    the dominant failure mode — so the card can say 'these don't even pass, and HOW
    they broke — don't retry'."""
    candidates = candidates or {}
    fam: dict[frozenset[str], dict] = {}
    for a in bad:
        sig = _sig(a.strategy)
        rec = fam.setdefault(sig, {"n": 0, "strategy": a.strategy, "cand": a.cand_id})
        rec["n"] += 1
    out = [{"family": " + ".join(sorted(sig)) or "misc", "n": rec["n"],
            "cand": rec["cand"][:8], "strategy": _clip(rec["strategy"], 150),
            "error": _cand_error(candidates, rec["cand"])}
           for sig, rec in fam.items()]
    out.sort(key=lambda d: -d["n"])
    return out


def _tech_events(ok: list, bad: list, candidates: dict) -> list[dict]:
    """Flatten this run's attempts to single-technique-tag events for the
    cross-op KnowledgeStore curator (knowledge.py) — same underlying attempts
    as .ledger/.failed, but tagged at the single-tag granularity Part B indexes
    by, not the multi-tag family signature .ledger/.failed group by.

    'confirmed' is gated on the candidate having actually ENTERED the frontier
    (verdict == 'entered', the same gate store.record_playbook() uses) — a
    correct-but-dominated attempt is not durable cross-op evidence. 'pitfall'
    events carry the RAW error code (not the formatted '(×N)' string) so the
    curator can group/replicate by error_family itself."""
    out: list[dict] = []
    for a in ok:
        c = candidates.get(a.cand_id) or {}
        if c.get("verdict") != "entered":
            continue
        for tag in _techniques(a.strategy):
            out.append({"tag": tag, "kind": "confirmed", "score": a.score,
                       "error": None, "cand": a.cand_id[:8], "strategy": a.strategy})
    for a in bad:
        err = _dominant_error(candidates, a.cand_id)
        for tag in _techniques(a.strategy):
            out.append({"tag": tag, "kind": "pitfall", "score": None,
                       "error": err, "cand": a.cand_id[:8], "strategy": a.strategy})
    return out


def analyze(events: list[dict], candidates: dict[str, dict], *,
            task_id: int = 0, name: str = "", family: str = "",
            stale_threshold: int = 4, weak_gap: float = 0.25) -> ProblemReflection:
    """Deterministic reflection over one problem's history. `candidates` maps
    cand_id -> candidate record (for per-workload vectors of the best)."""
    r = ProblemReflection(task_id=task_id, name=name, family=family)

    # exec_done carries the score + order; strategy/model/correct live on the matching
    # plan_done event and the candidate record — join them by cand_id.
    meta: dict[str, dict] = {}
    for e in events:
        if e.get("ev") == "plan_done" and e.get("cand"):
            meta[e["cand"]] = {"strategy": (e.get("strategy") or "").strip(),
                               "model": e.get("model", "?")}
    for cid, c in candidates.items():
        m = meta.setdefault(cid, {})
        if not m.get("strategy"):
            m["strategy"] = (c.get("strategy") or "").strip()
        if not m.get("model") or m.get("model") == "?":
            m["model"] = c.get("model") or "?"
        m["correct"] = bool(c.get("correct", True))

    attempts: list[Attempt] = []
    for e in events:
        if e.get("ev") != "exec_done":
            continue
        cid = e.get("cand", "")
        m = meta.get(cid, {})
        s = e.get("sol_score_cal")
        if not isinstance(s, (int, float)):
            s = e.get("sol_score")
        if not isinstance(s, (int, float)):
            # No score: a correctness FAILURE (correct=False) is a tried-and-failed
            # attempt (score 0) — the strongest 'don't retry'. A genuinely unscored /
            # pending candidate (correct unknown) is skipped.
            if m.get("correct") is False:
                s = 0.0
            else:
                continue
        attempts.append(Attempt(
            order=len(attempts) + 1, score=float(s), cand_id=cid,
            strategy=m.get("strategy", ""), model=m.get("model", "?"),
            correct=bool(m.get("correct", True))))
    r.n_evals = len(attempts)
    # An attempt that FAILED correctness scores 0 with correct=False — a stronger
    # "don't retry" than a merely-slow one. Split them out so the main ledger reflects
    # real ceilings and the failures get their own dead-end list.
    ok = [a for a in attempts if a.correct and a.score > 0]
    bad = [a for a in attempts if not a.correct or a.score <= 0]
    if not ok:
        r.status = "broken" if any(e.get("ev") == "plan_done" for e in events) else "thin"
        r.headline = ("No correct kernel yet — the whole score is gated on producing "
                      "ONE that passes all workloads. Prioritize correctness over speed.")
        r.failed = _failed_ledger(bad, candidates)
        r.tech_events = _tech_events(ok, bad, candidates)
        return r

    best_a = max(ok, key=lambda a: a.score)
    r.best, r.best_order = round(best_a.score, 4), best_a.order
    r.best_strategy, r.best_cand = best_a.strategy, best_a.cand_id
    r.stale_evals = r.n_evals - best_a.order
    r.failed = _failed_ledger(bad, candidates)
    r.tech_events = _tech_events(ok, bad, candidates)

    # --- attempt ledger: distinct families, their ceiling, how many times tried ---
    # (correct attempts only — failures are in r.failed, not the "best score each" list)
    fam_best: dict[frozenset[str], dict] = {}
    fam_counts: dict[frozenset[str], int] = {}
    for a in ok:
        sig = _sig(a.strategy)
        fam_counts[sig] = fam_counts.get(sig, 0) + 1
        cur = fam_best.get(sig)
        if cur is None or a.score > cur["score"]:
            fam_best[sig] = {"score": a.score, "strategy": a.strategy, "n": 0, "cand": a.cand_id}
    for sig, rec in fam_best.items():
        rec["n"] = fam_counts[sig]
        r.ledger.append({"family": " + ".join(sorted(sig)) or "misc",
                         "best": round(rec["score"], 4), "n": rec["n"],
                         "cand": rec["cand"][:8],
                         "strategy": _clip(rec["strategy"], 170)})
    r.ledger.sort(key=lambda d: -d["best"])

    # --- rabbit hole: does one family dominate the attempts? ---
    if fam_counts:
        dom_sig, dom_n = max(fam_counts.items(), key=lambda kv: kv[1])
        r.dominant_family = tuple(sorted(dom_sig))
        r.dominant_share = dom_n / r.n_evals

    # --- technique coverage: which ladder rungs tried vs untried ---
    # Only surface rungs that APPLY to this op's bottleneck class — suggesting fp8 /
    # split-k / nvfp4 (tensor-core/GEMM levers) on a memory-bound elementwise op like
    # RoPE or a norm is a red herring that sends agents down dead ends.
    tried = set()
    for a in attempts:
        tried |= _techniques(a.strategy)
    r.techniques_tried = sorted(tried)
    # "cublas" in tried is a runtime signal independent of (and stronger than) the
    # family label — if an attempt already used cublas, this op is compute-bound
    # regardless of what op_key_of() named it.
    compute_bound = _compute_bound(r.family) or "cublas" in tried
    allowed = _LADDER if compute_bound else tuple(t for t in _LADDER if t not in _GEMM_ONLY_RUNGS)
    r.techniques_untried = [t for t in allowed if t not in tried][:6]

    # --- per-workload bottleneck: the shapes dragging the mean, from the best cand ---
    cand = candidates.get(best_a.cand_id) or {}
    vec = cand.get("vector")
    pw = cand.get("per_workload")
    scores: list[float] = []
    if isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec):
        scores = [float(x) for x in vec]
    elif isinstance(pw, list) and pw:
        scores = [float(w.get("sol_score", 0)) for w in pw if isinstance(w, dict)]
    if len(scores) >= 3:
        med = statistics.median(scores)
        r.workload_median = round(med, 3)
        weak = [{"index": i, "score": round(s, 3)}
                for i, s in enumerate(scores) if s < med - weak_gap]
        r.weak_workloads = sorted(weak, key=lambda d: d["score"])[:6]

    # --- classify + headline ---
    recent = attempts[-min(5, len(attempts)):]
    recent_mean = statistics.mean(a.score for a in recent)
    improving = r.best_order >= r.n_evals - 1        # best is the latest (or nearly)
    if not best_a.correct and r.best is not None and r.best <= 0.001:
        r.status = "broken"
    elif improving:
        r.status = "climbing"
    elif r.stale_evals >= stale_threshold and recent_mean < r.best - 0.03:
        r.status = "regressing"
    elif r.dominant_share >= 0.55 and r.n_evals >= 6:
        r.status = "rabbit_hole"
    else:
        r.status = "plateaued"

    r.headline = _headline(r)
    return r


def _headline(r: ProblemReflection) -> str:
    if r.status == "regressing":
        return (f"Your BEST kernel ({r.best:.3f}) came from eval #{r.best_order} and the "
                f"{r.stale_evals} evals since have all been worse. STOP exploring — reload "
                f"cand {r.best_cand[:8]} and improve THAT, don't start from scratch.")
    if r.status == "rabbit_hole":
        fam = " + ".join(r.dominant_family) or "one family"
        return (f"{int(r.dominant_share*100)}% of {r.n_evals} attempts are the same family "
                f"[{fam}] and it's capped at {r.best:.3f}. Break out: try an approach whose "
                f"technique set is NOT {fam}.")
    if r.status == "plateaued":
        return (f"{r.n_evals} evals, stuck at {r.best:.3f}. The obvious approaches are spent; "
                f"the win now is a different axis (see untried rungs + weak workloads below).")
    if r.status == "climbing":
        return f"Improving — best {r.best:.3f} is recent. Keep pushing the current line."
    return r.headline


def render_card(r: ProblemReflection) -> str:
    """Markdown coach card, injected verbatim into the agent's CONTEXT.md."""
    if r.status in ("thin",):
        return ""
    tag = {"regressing": "⚠ REGRESSING", "rabbit_hole": "⚠ RABBIT HOLE",
           "plateaued": "◦ PLATEAUED", "climbing": "↑ CLIMBING",
           "broken": "✖ NO CORRECT KERNEL"}.get(r.status, r.status)
    L = [f"## Coach — cross-run reflection  [{tag}]", "", r.headline, ""]

    if r.weak_workloads and r.workload_median is not None:
        idxs = ", ".join(f"#{w['index']} ({w['score']:.2f})" for w in r.weak_workloads)
        L += [f"**Where the loss is:** workloads {idxs} score far below the median "
              f"({r.workload_median:.2f}). These specific shapes are most of your gap — "
              f"specialize for them (their axes are in `workloads.md`).", ""]

    if r.ledger:
        L += ["**Already tried — don't re-derive these (best score each). Read the exact",
              "kernel in `prior/<score>_<cand>.py` before assuming what it did:**"]
        for d in r.ledger[:8]:
            L.append(f"- `{d['best']:.3f}` ×{d['n']}  [{d['family']}]  {d['strategy']}  "
                     f"→ `prior/{d['best']:.3f}_{d['cand']}.py`")
        L.append("")

    if r.failed:
        total = sum(d["n"] for d in r.failed)
        L += [f"**Tried and FAILED correctness — do NOT retry ({total} attempt(s) that don't even",
              "pass the tolerance/compile gate; the kernel in `prior/0.000_<cand>.py` shows the bug):**"]
        for d in r.failed[:6]:
            err = f"  ({d['error']})" if d.get("error") else ""
            L.append(f"- ×{d['n']}  [{d['family']}]{err}  {d['strategy']}  → `prior/0.000_{d['cand']}.py`")
        L.append("")

    if r.techniques_untried:
        L += [f"**Untried rungs that fit this op:** {', '.join(r.techniques_untried)} "
              f"— at least one is likely the axis you haven't explored (tailored to this "
              f"op's bottleneck class, not a generic list).", ""]
    return "\n".join(L).rstrip() + "\n"


def _load_candidates(pdir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    cdir = pdir / "candidates"
    if not cdir.is_dir():
        return out
    for f in cdir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        cid = d.get("cand_id") or f.stem
        out[cid] = d
    return out


def from_runs_dir(runs_dir: str | Path, task_id: int, *, name: str = "",
                  family: str = "") -> ProblemReflection:
    """Load a problem's journal + candidates and analyze it."""
    pdir = Path(runs_dir) / str(task_id)
    events = []
    jf = pdir / "journal.jsonl"
    if jf.is_file():
        for line in jf.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    if not family:
        family = next((e.get("family") for e in events if e.get("family")), "") or ""
    return analyze(events, _load_candidates(pdir), task_id=task_id, name=name, family=family)


def _transfer_line(r: ProblemReflection, fam_best: dict) -> str:
    """A cross-problem nudge: the best OTHER problem sharing this op family."""
    sib = fam_best.get(r.family)
    if not sib or sib["task"] == r.task_id or r.best is None or sib["best"] <= (r.best + 0.02):
        return ""
    return (f"**Transfer:** sibling problem #{sib['task']} (same family `{r.family}`) reached "
            f"{sib['best']:.3f} with: {sib['strategy'][:90]} — see if that lever ports here.")


def reflect_all(runs_dir: str | Path, task_ids: list[int] | None = None, *,
                names: dict[int, str] | None = None, dump_prior: int = 8,
                log=lambda *_: None) -> dict[int, ProblemReflection]:
    """Regenerate every problem's `reflection.md` coach card (deterministic, no LLM).
    Also stages the top prior kernels under `<task>/prior/` for on-demand inspection.
    Returns the reflections keyed by task id. Safe to call at fleet startup and on a
    timer; a config change never crashes it (best-effort per problem)."""
    runs_dir = Path(runs_dir)
    names = names or {}
    if task_ids is None:
        task_ids = sorted(int(p.name) for p in runs_dir.glob("*")
                          if p.is_dir() and p.name.isdigit())
    refls: dict[int, ProblemReflection] = {}
    for t in task_ids:
        try:
            refls[t] = from_runs_dir(runs_dir, t, name=names.get(t, ""))
        except Exception as exc:                        # one bad problem must not abort the sweep
            log(f"[reflect] task {t}: {exc!r}")
    # fleet-level family-best map for cross-problem transfer
    fam_best: dict[str, dict] = {}
    for r in refls.values():
        if r.family and r.best is not None:
            cur = fam_best.get(r.family)
            if cur is None or r.best > cur["best"]:
                fam_best[r.family] = {"task": r.task_id, "best": r.best,
                                      "strategy": r.best_strategy or "?"}
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for t, r in refls.items():
        card = render_card(r)
        if card:
            tl = _transfer_line(r, fam_best)
            if tl:
                card = card.rstrip() + "\n\n" + tl + "\n"
            (runs_dir / str(t) / "reflection.md").write_text(card)
            attach_diagnosis(runs_dir, t)               # re-attach any stored fable prose (cheap)
            _append_snapshot(runs_dir / str(t), r, card, now_iso)
        if dump_prior:
            _dump_prior(runs_dir, t, r)
    log(f"[reflect] wrote {sum(1 for r in refls.values() if r.status not in ('thin',))} "
        f"coach card(s) across {len(refls)} problem(s)")
    return refls


DIAGNOSIS_HEADER = "## Coach — expert diagnosis"          # stable prefix — see attach_diagnosis


def attach_diagnosis(runs_dir: str | Path, task_id: int) -> bool:
    """Append the stored diagnosis prose (diagnosis.json, written by the diagnose
    layer) to `<task>/reflection.md`, idempotently. Called on every reflect pass
    so the deterministic card always carries the latest diagnosis WITHOUT
    re-spending on it. No-op if there's no diagnosis yet.

    The header names the model that ACTUALLY produced the prose (read from
    diagnosis.json — it varies: native claude-sonnet-5, an OpenRouter fallback,
    or whichever `--reflect-model`/FALLBACK_CHAIN landed on for that run), not a
    fixed label — `--reflect-model` has changed several times across restarts
    and a hardcoded name here would silently lie about the source. DIAGNOSIS_HEADER
    itself stays a stable, model-less prefix so the stale-block strip below finds
    and replaces ANY prior diagnosis, regardless of which model wrote it."""
    pdir = Path(runs_dir) / str(task_id)
    dfile, card = pdir / "diagnosis.json", pdir / "reflection.md"
    if not dfile.is_file() or not card.is_file():
        return False
    try:
        d = json.loads(dfile.read_text())
        prose = (d.get("prose") or "").strip()
        model = d.get("model") or "?"
    except Exception:
        return False
    if not prose:
        return False
    text = card.read_text(errors="replace")
    body = text.split("\n" + DIAGNOSIS_HEADER, 1)[0].rstrip()   # drop any stale diagnosis block
    header = f"{DIAGNOSIS_HEADER} (via {model})"
    card.write_text(f"{body}\n\n{header}\n\n{prose}\n")
    return True


def _append_snapshot(pdir: Path, r: ProblemReflection, card: str, ts: str) -> None:
    """Append a timestamped coach-card snapshot to `<task>/reflections.jsonl`, but
    only when the card actually changed — so the dashboard timeline shows how the
    diagnosis evolved (not one row per 20-min tick that said the same thing)."""
    log = pdir / "reflections.jsonl"
    if log.is_file():
        lines = log.read_text().splitlines()
        if lines:
            try:
                if json.loads(lines[-1]).get("card", "").strip() == card.strip():
                    return
            except Exception:
                pass
    with log.open("a") as f:
        f.write(json.dumps({"ts": ts, "task": r.task_id, "status": r.status,
                            "headline": r.headline, "card": card}) + "\n")


def _dump_prior(runs_dir: Path, task_id: int, r: ProblemReflection) -> None:
    """Stage EXACTLY the kernels the coach card references (each ledger entry + each
    failed dead-end) under `<task>/prior/<score>_<cand>.py`, so every `→ prior/…`
    link in the card resolves and an agent can read what an approach really did."""
    cands = _load_candidates(runs_dir / str(task_id))
    by_prefix: dict[str, tuple] = {}
    for cid, c in cands.items():
        by_prefix.setdefault(cid[:8], (cid, c))
    pdir = runs_dir / str(task_id) / "prior"

    def _stage(cand8: str, score: float) -> None:
        item = by_prefix.get(cand8)
        if not item:
            return
        cid, c = item
        srcs = (c.get("solution") or {}).get("sources") or []
        if not srcs:
            return
        pdir.mkdir(parents=True, exist_ok=True)
        body = "\n\n".join(f"# --- {s.get('path')} ---\n{s.get('content','')}" for s in srcs)
        (pdir / f"{score:.3f}_{cand8}.py").write_text(
            f"# score={score:.4f}  strategy: {c.get('strategy','')[:200]}\n\n{body}")

    for d in r.ledger[:8]:
        _stage(d["cand"], d["best"])
    for d in r.failed[:6]:
        _stage(d["cand"], 0.0)
