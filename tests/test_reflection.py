"""Tests for the cross-run reflection ("Coach") layer.

Deterministic over synthetic journals: the detectors are pure functions of the
event stream + candidate records, so every classification / directive is exactly
assertable — no GPU, no model.
"""

from __future__ import annotations

import json
from pathlib import Path

from solver.engine import reflection as R


def _events(scores, strategies):
    """Interleaved plan_done+exec_done for each (score, strategy), like the real
    journal: strategy lives on plan_done, the score on exec_done, joined by cand."""
    ev = []
    cands = {}
    for i, (s, strat) in enumerate(zip(scores, strategies)):
        cid = f"cand{i:03d}"
        ev.append({"ev": "plan_done", "cand": cid, "model": "opus", "strategy": strat})
        ev.append({"ev": "exec_done", "cand": cid, "sol_score_cal": s, "sol_score": s + 0.3})
        cands[cid] = {"cand_id": cid, "strategy": strat, "model": "opus", "correct": True,
                      "sol_score_calibrated": s}
    return ev, cands


def test_regression_detected_and_reload_directive():
    # best is the FIRST attempt; everything after is worse → regressing, reload it.
    scores = [0.70, 0.55, 0.60, 0.52, 0.58, 0.50]
    ev, cands = _events(scores, ["seed"] + ["bf16 cublas gemm"] * 5)
    r = R.analyze(ev, cands, task_id=3)
    assert r.status == "regressing"
    assert r.best == 0.70 and r.best_order == 1
    assert r.stale_evals == 5
    card = R.render_card(r)
    assert "REGRESSING" in card and "cand000"[:8] in card
    assert "don't start from scratch" in card.lower()


def test_rabbit_hole_when_one_family_dominates():
    # many attempts, one family, no late improvement → rabbit hole.
    scores = [0.40, 0.41, 0.39, 0.42, 0.40, 0.41, 0.40, 0.42]
    ev, cands = _events(scores, ["bf16 cublas matmul streaming"] * 8)
    r = R.analyze(ev, cands, task_id=1)
    assert r.status == "rabbit_hole"
    assert r.dominant_share == 1.0
    assert "RABBIT HOLE" in R.render_card(r)


def test_per_workload_bottleneck_from_vector():
    ev, cands = _events([0.6], ["triton fused"])
    # two workloads far below the median drag the score
    cands["cand000"]["vector"] = [0.8, 0.14, 0.79, 0.82, 0.13, 0.77, 0.8]
    r = R.analyze(ev, cands, task_id=5)
    idxs = {w["index"] for w in r.weak_workloads}
    assert 1 in idxs and 4 in idxs
    assert "Where the loss is" in R.render_card(r)


def test_ledger_dedups_families_and_keeps_best():
    scores = [0.30, 0.50, 0.45]
    ev, cands = _events(scores, ["bf16 cublas", "triton fused", "bf16 cublas"])
    r = R.analyze(ev, cands, task_id=2)
    fams = {d["family"]: d for d in r.ledger}
    # the two bf16+cublas attempts collapse to one ledger row at their best score
    bf = fams.get("bf16 + cublas")
    assert bf is not None and bf["n"] == 2 and bf["best"] == 0.45


def test_untried_ladder_rungs_surface():
    ev, cands = _events([0.4], ["bf16 cublas matmul"])
    r = R.analyze(ev, cands, task_id=9)
    assert "fp8" in r.techniques_untried and "cuda_graph" in r.techniques_untried
    assert "bf16" not in r.techniques_untried      # it WAS tried


def test_no_correct_kernel_is_broken():
    r = R.analyze([{"ev": "plan_done", "cand": "x", "strategy": "s"}], {}, task_id=7)
    assert r.status == "broken"
    assert "correctness" in r.headline.lower()


def test_reflect_all_writes_cards_and_prior(tmp_path):
    runs = tmp_path
    p = runs / "3"
    (p / "candidates").mkdir(parents=True)
    scores = [0.70, 0.55, 0.60, 0.52, 0.58, 0.50]
    ev, cands = _events(scores, ["seed"] + ["bf16 cublas", "triton streaming"] * 2 + ["bf16 cublas"])
    (p / "journal.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    for cid, c in cands.items():
        c["solution"] = {"sources": [{"path": "k.py", "content": f"# {cid}"}]}
        (p / "candidates" / f"{cid}.json").write_text(json.dumps(c))
    refls = R.reflect_all(runs, [3])
    assert refls[3].status == "regressing"
    assert (p / "reflection.md").is_file()
    assert "REGRESSING" in (p / "reflection.md").read_text()
    # top prior kernels staged for on-demand inspection, named by score
    priors = list((p / "prior").glob("*.py"))
    assert priors and any("0.700" in f.name for f in priors)


def test_reflect_all_survives_a_bad_problem(tmp_path):
    (tmp_path / "5").mkdir()
    (tmp_path / "5" / "journal.jsonl").write_text("not json\n{bad\n")
    # must not raise — best-effort per problem
    refls = R.reflect_all(tmp_path, [5])
    assert 5 in refls
