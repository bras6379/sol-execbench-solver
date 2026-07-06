"""A deterministic 'simulated search' for laptop demo runs.

`solver solve` has no real Agent/GPU backend yet, so it drives the *real* engine
(loop, frontier, tiers, escalation, journaling, resume) against this stub. Each
problem gets a hidden difficulty; the planner proposes candidates that improve
toward the current tier's ceiling with diminishing returns — so easy problems
converge on the cheap tier, hard ones plateau and escalate, and the whole system
is exercisable and *viewable* without a GPU or model. Fully deterministic (keyed
on ints), so runs are reproducible and resumable.
"""

from __future__ import annotations

N_SHAPES = 6

_FAMILIES = ["rmsnorm", "rope", "softmax", "layernorm", "gemm", "attention",
             "moe", "elementwise", "reduction", "conv"]

_STRATEGIES = ["torch reference wrapper", "fuse elementwise epilogue",
               "vectorized 128-bit loads", "Triton tiled kernel",
               "warp-specialized pipeline", "CUTLASS epilogue fusion",
               "persistent kernel + PDL", "CUDA-graph capture"]


def _h(*ints: int) -> float:
    """Deterministic hash of ints → [0, 1) (no PYTHONHASHSEED dependence)."""
    v = 2166136261
    for x in ints:
        v = ((v ^ (int(x) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return v / 0xFFFFFFFF


def family_of(task_id: int) -> str:
    return _FAMILIES[task_id % len(_FAMILIES)]


def _ceilings(task_id: int) -> tuple[float, float]:
    """(cheap-tier ceiling, strong-tier ceiling) by difficulty band."""
    return {0: (0.95, 0.97), 1: (0.82, 0.95), 2: (0.60, 0.85)}[task_id % 3]


def sim_seeds(task_id: int) -> list[dict]:
    return [{"__eval__": {"scores": [0.5] * N_SHAPES}}]


def _lang(strategy: str) -> tuple[str, str]:
    if "Triton" in strategy:
        return "triton", "py"
    if "CUTLASS" in strategy or "PDL" in strategy or "pipeline" in strategy:
        return "cuda_cpp", "cu"
    return "pytorch", "py"


def _source(task_id: int, step: int, persp, strategy: str, scores: list[float]) -> dict:
    lang, ext = _lang(strategy)
    code = (f"# task {task_id} · iter {step} · {persp} · {strategy}\n"
            f"# per-shape sol_score ~ {[round(s, 2) for s in scores]}\n"
            "import torch\n\n"
            "def run(*tensors):\n"
            f"    # {strategy}: fused / vectorized path (simulated)\n"
            "    out = tensors[-1]\n"
            "    # ... kernel body ...\n"
            "    return out\n")
    return {"spec": {"languages": [lang]},
            "sources": [{"path": f"kernel.{ext}", "content": code}]}


def sim_planner(persp, parent, ctx) -> dict:
    cheap_c, strong_c = _ceilings(ctx.task_id)
    ceiling = strong_c if ctx.tier_idx >= 1 else cheap_c
    step = ctx.iters
    if step % 9 == 8:                                   # occasional bounce → outcome variety
        return {"scores": [0.3] * N_SHAPES, "invalid": True,
                "strategy": "malformed launch config"}
    cur = ctx.frontier.best_score()
    base = cur + (ceiling - cur) * 0.4                  # diminishing returns toward the ceiling
    scores = [round(min(ceiling, max(0.0, base + (_h(ctx.task_id, s, step) - 0.5) * 0.06)), 4)
              for s in range(N_SHAPES)]                 # ±0.03 per-shape jitter → specialists
    strategy = _STRATEGIES[min(step, len(_STRATEGIES) - 1)]
    solution = {**_source(ctx.task_id, step, persp, strategy, scores),
                "__eval__": {"scores": scores},
                "__uid__": f"{persp}:{step}"}           # unique hash even in the plateau region
    return {"solution": solution, "strategy": strategy}
