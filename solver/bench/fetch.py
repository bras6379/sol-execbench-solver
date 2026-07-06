"""Fetch SOL-ExecBench problems by global task number onto the local disk.

Writes, per problem, into ``<out_dir>/<task_id>/``:
  - definition.json   (official schema: name, hf_id, description, axes,
                       custom_inputs_entrypoint, inputs, outputs, reference)
  - reference.py      (the PyTorch reference + input generator, verbatim)
  - workload.jsonl    (one workload per line, verbatim from the dataset —
                       carries per-workload axes, inputs, and tolerance)
  - metadata.json     (provenance + optional Speed-of-Light baselines)

The dataset is the authoritative source for tolerances/inputs; the public
website API (optional) adds the per-workload SOL target (``sol_ms``) and the
reference baseline latency, which the dataset does not include.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import dataset as ds
from . import problems as pb

DEFAULT_OUT_DIR = Path("problems")
_SOL_API = "https://research.nvidia.com/benchmarks/sol-execbench/api/kernels/{id}"

# Files that must all exist for a problem to count as already fetched.
_REQUIRED_FILES = ("definition.json", "reference.py", "workload.jsonl", "metadata.json")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _build_definition(row: dict) -> dict:
    """Assemble definition.json exactly as the official downloader does."""
    return {
        "name": row["name"],
        "hf_id": row.get("hf_id"),
        "description": row["description"],
        "axes": row["axes"],
        "custom_inputs_entrypoint": row.get("custom_inputs_entrypoint"),
        "inputs": row["inputs"],
        "outputs": row["outputs"],
        "reference": row["reference"],
    }


def _fetch_sol(task_id: int, n_workloads: int) -> dict:
    """Best-effort SOL baseline from the public website API. Never raises."""
    try:
        url = _SOL_API.format(id=task_id)
        req = urllib.request.Request(url, headers={"User-Agent": "sol-execbench-solver"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read()).get("data", {})
    except Exception as exc:  # network/HTTP/JSON — SOL is optional enrichment
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    api_workloads = data.get("workloads", []) or []
    per_workload = [
        {
            "index": i,
            "axes": w.get("axes"),
            "sol_ms": w.get("sol_ms"),
            "baseline_latency_ms": w.get("baseline_latency_ms"),
        }
        for i, w in enumerate(api_workloads)
    ]
    return {
        "available": True,
        "baseline_latency_ms": data.get("baseline_latency_ms"),
        "latency_unit": data.get("latency_unit"),
        "sol_is_dummy": data.get("sol_is_dummy"),
        "workload_count_matches": len(api_workloads) == n_workloads,
        "per_workload": per_workload,
    }


def _is_complete(problem_dir: Path) -> bool:
    return all((problem_dir / f).exists() for f in _REQUIRED_FILES)


@dataclass
class FetchResult:
    task_id: int
    subset: str
    name: str
    path: Path
    status: str  # "written" | "skipped"


def fetch(
    ids: list[int],
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    refresh: bool = False,
    with_sol: bool = True,
    revision: str = ds.DEFAULT_REVISION,
    cache_dir: Path = ds.DEFAULT_CACHE_DIR,
) -> list[FetchResult]:
    """Fetch the given global task ids. Idempotent unless ``refresh``."""
    out_dir = Path(out_dir)
    ids = sorted(dict.fromkeys(ids))

    # Group by subset so each parquet is downloaded/read at most once.
    by_subset: dict[str, list[int]] = {}
    for task_id in ids:
        by_subset.setdefault(pb.subset_of(task_id), []).append(task_id)

    sha = ds.resolved_sha(revision)
    results: list[FetchResult] = []

    for subset, subset_ids in by_subset.items():
        index: dict[int, dict] | None = None  # lazily loaded on first miss
        for task_id in subset_ids:
            _, local = pb.resolve(task_id)
            problem_dir = out_dir / str(task_id)

            if not refresh and _is_complete(problem_dir):
                name = _read_name(problem_dir)
                results.append(FetchResult(task_id, subset, name, problem_dir, "skipped"))
                continue

            if index is None:
                index = ds.load_subset(
                    subset, revision=revision, cache_dir=cache_dir, refresh=refresh
                )
            if local not in index:
                raise KeyError(
                    f"task {task_id} ({subset} #{local}) not found in dataset subset {subset}"
                )
            row = index[local]
            results.append(
                _write_problem(
                    task_id, subset, local, row, problem_dir,
                    with_sol=with_sol, revision=revision, sha=sha,
                )
            )
    return results


def _read_name(problem_dir: Path) -> str:
    try:
        return json.loads((problem_dir / "definition.json").read_text()).get("name", "?")
    except Exception:
        return "?"


def _write_problem(
    task_id: int,
    subset: str,
    local: int,
    row: dict,
    problem_dir: Path,
    *,
    with_sol: bool,
    revision: str,
    sha: str | None,
) -> FetchResult:
    problem_dir.mkdir(parents=True, exist_ok=True)

    definition = _build_definition(row)
    (problem_dir / "definition.json").write_text(json.dumps(definition, indent=4) + "\n")
    (problem_dir / "reference.py").write_text(row["reference"])

    workloads = row["workloads"]
    with (problem_dir / "workload.jsonl").open("w") as f:
        for workload in workloads:
            f.write(json.dumps(workload) + "\n")

    metadata = {
        "task_id": task_id,
        "subset": subset,
        "collection_index": local,
        "name": row["name"],
        "hf_id": row.get("hf_id"),
        "num_workloads": len(workloads),
        "workloads_have_tolerance": all("tolerance" in w for w in workloads),
        "source": {
            "dataset": ds.DATASET_ID,
            "revision": revision,
            "resolved_sha": sha,
            "subset_parquet": f"data/{subset}.parquet",
        },
        "fetched_at": _utc_now(),
    }
    if with_sol:
        metadata["sol"] = _fetch_sol(task_id, len(workloads))

    (problem_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return FetchResult(task_id, subset, row["name"], problem_dir, "written")
