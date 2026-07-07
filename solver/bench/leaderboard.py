"""SOL-ExecBench leaderboard client — submit a solution and poll its score.

The official leaderboard (research.nvidia.com/benchmarks/sol-execbench) closes the
loop: `solver solve` → `runs/<task>/best_solution.json` → `submit` → the real
score. Auth is a JWT in `SOLBENCH_TOKEN` (loaded from .env). The leaderboard's
`kernel_id` is the same numbering as our `task_id` (verified: kernel 230 =
021_rmsnorm_h128), so a task submits as `kernel_id = task_id`. Stdlib only.
"""

from __future__ import annotations

import json
import mimetypes
import os
import secrets
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://research.nvidia.com/benchmarks/sol-execbench"
TERMINAL = {"COMPLETED", "FAILED", "ERROR", "REJECTED", "CANCELLED"}

# Fields the poll endpoint returns — the same columns the website shows.
RESULT_KEYS = ("id", "kernel_id", "kernel_name", "status", "submission_mode",
               "is_correct", "sol_score", "latency_ms", "fast_1_count",
               "fast_1_total", "avg_speedup", "submitted_at", "finished_at",
               "result_available_at", "error_log")


def _token(token: str | None) -> str:
    token = token or os.environ.get("SOLBENCH_TOKEN")
    if not token:
        raise SystemExit("SOLBENCH_TOKEN not set — add it to .env (see .env.example)")
    return token


def _send(req: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            obj = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"leaderboard HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"leaderboard unreachable: {e}")
    return obj.get("data", obj)


def _multipart(fields: dict, file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = "----solbench-" + secrets.token_hex(16)
    parts = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="{name}"\r\n\r\n{value}\r\n'.encode())
    ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{file_field}"; filename="{file_path.name}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n".encode())
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def submit(kernel_id: int, file_path: str | Path, *, token: str | None = None,
           gpu: str = "B200", mode: str = "private") -> dict:
    """Upload a solution.json to the leaderboard. Returns {submission_id, ...}."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise SystemExit(f"solution file not found: {file_path}")
    body, ctype = _multipart({"kernel_id": str(kernel_id), "gpu_type": gpu,
                              "submission_mode": mode}, "file", file_path)
    req = urllib.request.Request(
        f"{BASE_URL}/api/submissions/upload", data=body, method="POST",
        headers={"Authorization": f"Bearer {_token(token)}", "Content-Type": ctype,
                 "Content-Length": str(len(body))})
    return _send(req)


def poll(submission_id: int, *, token: str | None = None) -> dict:
    """Fetch a submission's current status/score."""
    req = urllib.request.Request(
        f"{BASE_URL}/api/submissions/{submission_id}",
        headers={"Authorization": f"Bearer {_token(token)}"})
    d = _send(req)
    return {k: d.get(k) for k in RESULT_KEYS if k in d}


def board(kernel_id: int, *, gpu: str = "B200", token: str | None = None) -> dict:
    """Public leaderboard for a kernel: ranked entries + the #1 and SOL bound.
    Endpoint: /api/leaderboard/kernel/<id>/<gpu>."""
    req = urllib.request.Request(
        f"{BASE_URL}/api/leaderboard/kernel/{kernel_id}/{gpu}",
        headers={"Authorization": f"Bearer {_token(token)}"})
    d = _send(req)
    real = [e for e in (d.get("rankings") or [])
            if not e.get("is_reference") and e.get("rank") is not None and e.get("sol_score") is not None]
    real.sort(key=lambda e: e["rank"])
    top = real[0] if real else None
    return {"rankings": real, "n": len(real),
            "top_sol": (top or {}).get("sol_score"), "top_user": (top or {}).get("username"),
            "sol_bound": (d.get("sol_entry") or {}).get("sol_score")}


def rank_of(score: float | None, rankings: list[dict]) -> int | None:
    """1-based position `score` would take on the board (higher SOL = better)."""
    if score is None:
        return None
    return 1 + sum(1 for e in rankings if (e.get("sol_score") or 0) > score)


def format_result(row: dict) -> str:
    sid = row.get("id")
    score = row.get("sol_score")
    lat = row.get("latency_ms")
    fast = (f"{row['fast_1_count']}/{row['fast_1_total']}"
            if row.get("fast_1_total") else "-")
    return (f"#{sid} {row.get('status','-'):<10} correct={row.get('is_correct')}  "
            f"SOL={'-' if score is None else f'{score:.4f}'}  "
            f"lat={'-' if lat is None else f'{lat:.6f}ms'}  fast={fast}  "
            f"speedup={row.get('avg_speedup','-')}")
