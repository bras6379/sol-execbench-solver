"""Cross-problem transfer (KnowledgeStore) — op-key parsing + best-kernel
persistence + sibling warm-start hint + cross-operator technique patterns
(docs/context-architecture-plan.md Part B). No GPU."""

from __future__ import annotations

import asyncio

from solver.engine.frontier import Frontier, Member
from solver.engine.knowledge import KnowledgeStore, op_key_of
from solver.engine.reflection import ProblemReflection


def test_op_key_only_groups_true_siblings():
    assert op_key_of(230, "problems") == "rmsnorm"              # 021_rmsnorm_h128
    assert op_key_of(231, "problems") == "rmsnorm"              # 022_rmsnorm_h512  (sibling!)
    assert op_key_of(213, "problems") == "gemm"                 # 004_gemm_n128_k2048
    assert op_key_of(210, "problems") == "fused_add_rmsnorm"    # 001_fused_add_rmsnorm_h2048
    assert op_key_of(230, "problems") != op_key_of(213, "problems")   # rmsnorm ≠ gemm


class _Ctx:
    def __init__(self, task, fr):
        self.task_id, self.frontier, self.tier_idx, self.terminated_reason = task, fr, 0, "budget:time"


def _frontier(score, content="def run(x):\n    return x"):
    fr = Frontier(0.02)
    sol = {"spec": {"languages": ["triton"]}, "sources": [{"path": "kernel.py", "content": content}]}
    fr.accept(Member("c1", (score,), True, solution=sol, strategy="fused triton rmsnorm"))
    return fr


def test_best_kernel_persists_and_transfers_across_runs(tmp_path):
    ks = KnowledgeStore(tmp_path / "k")
    asyncio.run(ks.curate(_Ctx(230, _frontier(0.8)), "rmsnorm", "021_rmsnorm_h128"))
    assert (tmp_path / "k/best/rmsnorm.json").exists()             # persisted the winning kernel

    fresh = KnowledgeStore(tmp_path / "k")                          # a NEW process/run
    h = fresh.sibling_hint("rmsnorm", exclude_task=231)            # a DIFFERENT rmsnorm shape
    assert h and h["sibling"] == "021_rmsnorm_h128"
    assert h["sources"] and "def run" in h["sources"][0]["content"]  # the actual kernel to adapt
    assert h["strategy"] == "fused triton rmsnorm"


def test_hint_excludes_self_and_other_ops(tmp_path):
    ks = KnowledgeStore(tmp_path / "k")
    asyncio.run(ks.curate(_Ctx(230, _frontier(0.8)), "rmsnorm", "021_rmsnorm_h128"))
    assert ks.sibling_hint("rmsnorm", exclude_task=230) is None    # no self-hint
    assert ks.sibling_hint("gemm") is None                         # no cross-op contamination


def test_best_kernel_only_upgrades(tmp_path):
    ks = KnowledgeStore(tmp_path / "k")
    asyncio.run(ks.curate(_Ctx(230, _frontier(0.8, "GOOD")), "rmsnorm", "021_rmsnorm_h128"))
    asyncio.run(ks.curate(_Ctx(231, _frontier(0.5, "WORSE")), "rmsnorm", "022_rmsnorm_h512"))
    h = ks.sibling_hint("rmsnorm", exclude_task=999)
    assert "GOOD" in h["sources"][0]["content"] and h["score"] == 0.8   # kept the better one


def _refl(task_id, family, tech_events):
    return ProblemReflection(task_id=task_id, family=family, tech_events=tech_events)


def test_pattern_notes_empty_for_unknown_tags(tmp_path):
    ks = KnowledgeStore(tmp_path / "k")
    assert ks.pattern_notes(["split_k", "fusion"]) == {}


def test_curate_persists_confirmed_pattern_across_runs(tmp_path):
    ks = KnowledgeStore(tmp_path / "k")
    refl = _refl(213, "gemm", [{"tag": "split_k", "kind": "confirmed", "score": 0.79,
                                "error": None, "cand": "a1b2c3d4", "strategy": "3-way split-K"}])
    asyncio.run(ks.curate(_Ctx(213, _frontier(0.79)), "gemm", "004_gemm_n128", refl=refl))
    assert (tmp_path / "k/patterns.json").exists()

    fresh = KnowledgeStore(tmp_path / "k")                      # a NEW process/run
    notes = fresh.pattern_notes(["split_k"])
    assert notes["split_k"]["confirmed"][0]["op"] == "gemm"
    assert notes["split_k"]["confirmed"][0]["task"] == 213


def test_pattern_notes_excludes_the_querying_problems_own_entry(tmp_path):
    ks = KnowledgeStore(tmp_path / "k")
    refl = _refl(213, "gemm", [{"tag": "split_k", "kind": "confirmed", "score": 0.79,
                                "error": None, "cand": "a1b2c3d4", "strategy": "split-K"}])
    asyncio.run(ks.curate(_Ctx(213, _frontier(0.79)), "gemm", "004_gemm_n128", refl=refl))
    assert ks.pattern_notes(["split_k"], exclude_task=213) == {}
    assert ks.pattern_notes(["split_k"], exclude_task=999)["split_k"]["confirmed"]


def test_pitfall_requires_two_distinct_op_families_before_surfacing(tmp_path):
    ks = KnowledgeStore(tmp_path / "k")
    ev = [{"tag": "split_k", "kind": "pitfall", "score": None,
          "error": "RUNTIME_ERROR", "cand": "e5f6a7b8", "strategy": "split-K atomic epilogue"}]
    asyncio.run(ks.curate(_Ctx(213, _frontier(0.79)), "gemm", "004_gemm", refl=_refl(213, "gemm", ev)))
    # only one op family has hit this so far — must NOT be surfaced yet
    assert ks.pattern_notes(["split_k"]) == {}

    asyncio.run(ks.curate(_Ctx(16, _frontier(0.5)), "moe_expert_computation", "moe_16",
                          refl=_refl(16, "moe_expert_computation", ev)))
    # a second, DIFFERENT op family replicated it — now it's durable evidence
    notes = ks.pattern_notes(["split_k"])
    pf = notes["split_k"]["pitfalls"][0]
    assert pf["error_family"] == "RUNTIME_ERROR" and set(pf["ops"]) == {"gemm", "moe_expert_computation"}


def test_reopened_run_upserts_instead_of_duplicating(tmp_path):
    """A `reopened`-and-reterminated run re-calling curate() for the SAME task
    must replace its own prior entries, not double-count them (the bug already
    present in `_append_family`'s unconditional append)."""
    ks = KnowledgeStore(tmp_path / "k")
    confirmed_ev = [{"tag": "split_k", "kind": "confirmed", "score": 0.70,
                     "error": None, "cand": "aaaaaaaa", "strategy": "first pass"}]
    pitfall_ev = [{"tag": "split_k", "kind": "pitfall", "score": None,
                  "error": "RUNTIME_ERROR", "cand": "bbbbbbbb", "strategy": "trap"}]
    asyncio.run(ks.curate(_Ctx(213, _frontier(0.7)), "gemm", "004_gemm",
                         refl=_refl(213, "gemm", confirmed_ev + pitfall_ev)))
    # re-run: same task, IMPROVED score
    confirmed_ev2 = [{"tag": "split_k", "kind": "confirmed", "score": 0.85,
                      "error": None, "cand": "cccccccc", "strategy": "improved pass"}]
    asyncio.run(ks.curate(_Ctx(213, _frontier(0.85)), "gemm", "004_gemm",
                         refl=_refl(213, "gemm", confirmed_ev2 + pitfall_ev)))

    notes = ks.pattern_notes(["split_k"], exclude_task=999)
    confirmed = notes["split_k"]["confirmed"]
    assert len(confirmed) == 1 and confirmed[0]["score"] == 0.85   # replaced, not appended
    # the SAME op re-reporting the same pitfall must not inflate replication count
    assert notes["split_k"]["pitfalls"] == []   # still only 1 distinct op -> not surfaced yet
