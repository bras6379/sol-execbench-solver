"""Cross-problem transfer (KnowledgeStore) — op-key parsing + best-kernel
persistence + sibling warm-start hint. No GPU."""

from __future__ import annotations

import asyncio

from solver.engine.frontier import Frontier, Member
from solver.engine.knowledge import KnowledgeStore, op_key_of


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
