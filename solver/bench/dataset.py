"""Fetch and index the nvidia/SOL-ExecBench HuggingFace dataset (parquet).

The authoritative problem data lives in four parquet files (one per subset).
Each row is one problem with columns: name, description, hf_id, axes, inputs,
outputs, reference, custom_inputs_entrypoint, workloads (all JSON strings
except name/description/hf_id/reference). This module downloads the parquet
files (cached on disk) and returns rows keyed by local NNN index.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

DATASET_ID = "nvidia/SOL-ExecBench"
_RESOLVE = "https://huggingface.co/datasets/{ds}/resolve/{rev}/data/{subset}.parquet"
_DATASET_API = "https://huggingface.co/api/datasets/{ds}"

DEFAULT_REVISION = "main"
DEFAULT_CACHE_DIR = Path(".cache/solexecbench")

# Columns that hold JSON-encoded strings in the parquet.
_JSON_COLUMNS = ("axes", "inputs", "outputs", "workloads")


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sol-execbench-solver"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def resolved_sha(revision: str = DEFAULT_REVISION) -> str | None:
    """Best-effort: the dataset commit sha for reproducibility metadata."""
    try:
        meta = json.loads(_http_get(_DATASET_API.format(ds=DATASET_ID)))
        return meta.get("sha")
    except Exception:
        return None


def parquet_path(
    subset: str,
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> Path:
    """Return a local path to the subset parquet, downloading if needed."""
    cache_dir = Path(cache_dir) / revision
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{subset}.parquet"
    if refresh or not dest.exists():
        url = _RESOLVE.format(ds=DATASET_ID, rev=revision, subset=subset)
        dest.write_bytes(_http_get(url))
    return dest


def load_subset(
    subset: str,
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> dict[int, dict]:
    """Return {local_index: row_dict} for a subset.

    row_dict has string columns as-is and JSON columns decoded to Python
    objects. local_index is parsed from the ``NNN_`` name prefix.
    """
    path = parquet_path(
        subset, revision=revision, cache_dir=cache_dir, refresh=refresh
    )
    table = pq.read_table(path)
    rows = table.to_pylist()
    out: dict[int, dict] = {}
    for row in rows:
        local = int(str(row["name"]).split("_", 1)[0])
        decoded = dict(row)
        for col in _JSON_COLUMNS:
            if isinstance(decoded.get(col), str):
                decoded[col] = json.loads(decoded[col])
        out[local] = decoded
    return out
