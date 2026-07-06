"""Benchmark access: fetching problems and modelling/validating candidate Solutions.

- `dataset` / `fetch` / `problems` — download problem packs and map the global
  task number (1–235) onto the four benchmark subsets.
- `solution` — the harness Solution model + `scaffold`.
- `check` — static, GPU-free pre-flight of a candidate Solution.
"""
