"""Command-line interface for the SOL-ExecBench solver engine.

    solver fetch 1 2 5-10        # fetch specific tasks / ranges
    solver fetch --all           # fetch all 235 problems
    solver fetch 69 --refresh    # re-download, ignoring the cache
    solver fetch 67 --no-sol     # skip the website SOL-baseline enrichment
    solver list --all            # print task-id -> subset / name mapping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import check as check_mod
from . import dataset as ds
from . import fetch as fetch_mod
from . import problems as pb
from . import solution as sol_mod


def _resolve_ids(args) -> list[int]:
    if getattr(args, "all", False):
        if args.tasks:
            raise SystemExit("pass either task ids or --all, not both")
        return pb.all_ids()
    if not args.tasks:
        raise SystemExit("no task ids given (use ids/ranges like '1 5-10' or --all)")
    return pb.parse_specs(args.tasks)


def _cmd_fetch(args) -> None:
    ids = _resolve_ids(args)
    results = fetch_mod.fetch(
        ids,
        out_dir=Path(args.out_dir),
        refresh=args.refresh,
        with_sol=not args.no_sol,
        revision=args.revision,
        cache_dir=Path(args.cache_dir),
    )
    written = sum(r.status == "written" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    for r in results:
        print(f"[{r.status:7s}] {r.task_id:>3}  {r.subset:<16} {r.name}  -> {r.path}")
    print(f"\n{written} written, {skipped} skipped, {len(results)} total")


def _cmd_list(args) -> None:
    ids = _resolve_ids(args)
    # Group by subset; load each subset index once for the names.
    by_subset: dict[str, list[int]] = {}
    for task_id in ids:
        by_subset.setdefault(pb.subset_of(task_id), []).append(task_id)
    for subset, subset_ids in by_subset.items():
        index = ds.load_subset(
            subset, revision=args.revision, cache_dir=Path(args.cache_dir)
        )
        for task_id in subset_ids:
            _, local = pb.resolve(task_id)
            name = index.get(local, {}).get("name", "?")
            print(f"{task_id:>3}  {subset:<16} #{local:<3}  {name}")


def _cmd_scaffold(args) -> None:
    problem_dir = Path(args.out_dir_problems) / str(args.task)
    if not problem_dir.exists():
        raise SystemExit(f"problem {args.task} not fetched (run: solver fetch {args.task})")
    sol = sol_mod.scaffold(
        problem_dir, lang=args.lang, author=args.author, name=args.name
    )
    dest = (
        Path(args.out)
        if args.out
        else problem_dir / "candidates" / f"{sol.name}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(sol.to_json())
    print(f"scaffolded {args.lang} solution -> {dest}")
    # immediate self-check
    report = check_mod.check_solution(
        sol.to_dict(), json.loads((problem_dir / "definition.json").read_text())
    )
    _print_report(report, dest)


def _cmd_check(args) -> None:
    report = check_mod.check_solution_file(
        Path(args.path), problems_dir=Path(args.out_dir_problems)
    )
    _print_report(report, Path(args.path))
    if not report.ok:
        raise SystemExit(1)


def _print_report(report, path) -> None:
    for w in report.warnings:
        print(f"  warn: {w}")
    for e in report.errors:
        print(f"  ERROR: {e}")
    print(f"{'OK' if report.ok else 'FAILED'}: {path}"
          + (f" ({len(report.warnings)} warning(s))" if report.warnings else ""))


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("tasks", nargs="*", help="task ids or ranges, e.g. 1 2 5-10")
    p.add_argument("--all", action="store_true", help="select all 235 problems")
    p.add_argument("--revision", default=ds.DEFAULT_REVISION, help="dataset revision")
    p.add_argument(
        "--cache-dir", default=str(ds.DEFAULT_CACHE_DIR), help="parquet cache directory"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="solver", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download problems by task number")
    _add_selection_args(p_fetch)
    p_fetch.add_argument("--out-dir", default=str(fetch_mod.DEFAULT_OUT_DIR))
    p_fetch.add_argument("--refresh", action="store_true", help="re-download, ignore cache")
    p_fetch.add_argument("--no-sol", action="store_true", help="skip website SOL enrichment")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_list = sub.add_parser("list", help="print task-id -> subset / name mapping")
    _add_selection_args(p_list)
    p_list.set_defaults(func=_cmd_list)

    p_scaffold = sub.add_parser("scaffold", help="generate a candidate Solution for a task")
    p_scaffold.add_argument("task", type=int, help="global task id (1-235)")
    p_scaffold.add_argument("--lang", default="pytorch", choices=sorted(sol_mod.ALL_LANGS))
    p_scaffold.add_argument("--author", default="solver")
    p_scaffold.add_argument("--name", default=None, help="solution name (default: <def>__<lang>_baseline)")
    p_scaffold.add_argument("--out", default=None, help="output path (default: problems/<id>/candidates/<name>.json)")
    p_scaffold.add_argument("--out-dir-problems", default=str(fetch_mod.DEFAULT_OUT_DIR))
    p_scaffold.set_defaults(func=_cmd_scaffold)

    p_check = sub.add_parser("check", help="static pre-flight a candidate Solution (no GPU)")
    p_check.add_argument("path", help="path to a Solution JSON")
    p_check.add_argument("--out-dir-problems", default=str(fetch_mod.DEFAULT_OUT_DIR))
    p_check.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
