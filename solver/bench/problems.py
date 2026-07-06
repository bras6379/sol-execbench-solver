"""Task-number <-> benchmark-subset mapping and problem-spec parsing.

Global task numbers 1-235 map contiguously onto the four SOL-ExecBench
subsets. Within a subset each problem folder is named ``NNN_<slug>`` where NNN
is the 1-based index inside that subset; the global id is just that index
offset by the subset's start. This layout is verified against the HuggingFace
dataset (row counts 94/82/33/26, each subset NNN-contiguous).
"""

from __future__ import annotations

# (subset, first_global_id, last_global_id) — order matters, contiguous, covers 1..235.
COLLECTION_RANGES: tuple[tuple[str, int, int], ...] = (
    ("L1", 1, 94),
    ("L2", 95, 176),
    ("Quant", 177, 209),
    ("FlashInfer-Bench", 210, 235),
)

MIN_ID = COLLECTION_RANGES[0][1]
MAX_ID = COLLECTION_RANGES[-1][2]


def all_ids() -> list[int]:
    return list(range(MIN_ID, MAX_ID + 1))


def resolve(task_id: int) -> tuple[str, int]:
    """Return (subset, local_index) for a global task number.

    local_index is 1-based within the subset and equals the NNN name prefix.
    """
    for subset, start, end in COLLECTION_RANGES:
        if start <= task_id <= end:
            return subset, task_id - start + 1
    raise ValueError(
        f"task {task_id} is outside the supported range {MIN_ID}-{MAX_ID}"
    )


def subset_of(task_id: int) -> str:
    return resolve(task_id)[0]


def parse_specs(values: list[str]) -> list[int]:
    """Parse task-id specs like ``["1", "2", "5-10"]`` (commas also allowed).

    Returns a sorted, de-duplicated list. Validates every id is in range.
    """
    ids: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s), int(end_s)
                if end < start:
                    raise ValueError(f"invalid range: {part!r}")
                ids.extend(range(start, end + 1))
            else:
                ids.append(int(part))
    ordered = sorted(dict.fromkeys(ids))
    bad = [i for i in ordered if i < MIN_ID or i > MAX_ID]
    if bad:
        raise ValueError(
            f"task ids out of range {MIN_ID}-{MAX_ID}: {bad}"
        )
    return ordered
