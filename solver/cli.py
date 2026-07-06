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

from .bench import check as check_mod
from .bench import dataset as ds
from .bench import fetch as fetch_mod
from .bench import problems as pb
from .bench import solution as sol_mod


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


def _cmd_report(args) -> None:
    import time

    from .dashboard import report as report_mod

    runs_dir = Path(args.runs_dir)
    if args.demo:
        from .dashboard import demo_data
        runs_dir = Path(".cache/demo/runs")
        if not runs_dir.exists():
            demo_data.build_demo(runs_dir)
            print(f"demo journals -> {runs_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(".cache/demo/out") if args.demo else Path("out"))
    if args.watch:
        print(f"watching: regenerating {out_dir}/ every {args.watch}s (Ctrl-C to stop)")
        while True:
            report_mod.render(runs_dir, out_dir, refresh=args.refresh or args.watch)
            time.sleep(args.watch)
    path = report_mod.render(runs_dir, out_dir, refresh=args.refresh)
    print(f"dashboard site -> {path}")


def _cmd_solve(args) -> None:
    import asyncio

    from . import journal as J
    from .dashboard import metrics
    from .engine import (Config, KnowledgeStore, Perspective, StubExecutor, Tier,
                         make_agents, run_fleet, sim, stub_agents)

    ids = _resolve_ids(args)
    runs_dir = Path(args.runs_dir)
    knowledge = KnowledgeStore(args.knowledge_dir)
    families = {t: sim.family_of(t) for t in ids}
    names = {t: f"{sim.family_of(t)}_{t}" for t in ids}

    if args.agent == "sim":
        cfg = Config(
            tiers=[Tier("cheap", [Perspective("claude", "haiku"), Perspective("openai", "gpt")]),
                   Tier("strong", [Perspective("claude", "opus")])],
            plateau_cycles=2, escalate_ceiling=0.9, epsilon=0.02,
            max_iterations=args.max_iters, max_gpu_evals=args.max_evals,
        )
        agents = stub_agents(cfg.perspectives, sim.sim_planner)
        seeds_fn = sim.sim_seeds
        print(f"solving {len(ids)} problem(s) with the stub sim agent (no GPU/model) -> {runs_dir}/")
    else:
        cfg = Config(
            tiers=[Tier(args.agent, [Perspective(args.agent, args.model)])],
            plateau_cycles=2, escalate_ceiling=0.9, epsilon=0.02,
            max_iterations=args.max_iters, max_gpu_evals=args.max_evals,
        )
        agents = make_agents(cfg, runs_dir=runs_dir, timeout=args.timeout)
        seeds_fn = None       # _default_seeds; a real scaffold seed lands with the GPU executor
        print(f"solving {len(ids)} problem(s) with `{args.agent}`/{args.model} "
              f"(StubExecutor — no GPU scoring yet) -> {runs_dir}/")

    executor = StubExecutor(delay=args.delay)   # delay simulates GPU busy time → a real timeline
    asyncio.run(run_fleet(ids, executor, agents, cfg, runs_dir=runs_dir,
                          seeds_fn=seeds_fn, knowledge=knowledge,
                          families=families, names=names))
    js = J.read_all(runs_dir)
    ms = [metrics.problem_metrics(t, evs) for t, evs in sorted(js.items()) if t in ids]
    scored = [m["best"] for m in ms if m["best"] is not None]
    mean = sum(scored) / len(scored) if scored else 0.0
    esc = sum(1 for m in ms if any(e.get("ev") == "agent_changed" for e in js.get(m["task"], [])))
    print(f"done: {len(ms)} runs · fleet mean best {mean:.3f} · {esc} escalated to a stronger tier")
    print(f"view: solver report --runs-dir {runs_dir}   |   solver status --runs-dir {runs_dir}")


def _run_metrics(runs_dir: Path, task_id: int):
    from . import journal as J
    from .dashboard import metrics
    return metrics.problem_metrics(task_id, J.read(runs_dir / str(task_id) / "journal.jsonl"))


def _cmd_status(args) -> None:
    from . import journal as J
    runs_dir = Path(args.runs_dir)
    js = J.read_all(runs_dir)
    ids = pb.parse_specs(args.tasks) if args.tasks else sorted(js)
    from .dashboard import metrics
    print(f"{'task':>4}  {'family':<11} {'best':>6} {'iters':>5} {'evals':>5} {'front':>5}  status")
    for t in ids:
        if t not in js:
            continue
        m = metrics.problem_metrics(t, js[t])
        best = f"{m['best']:.3f}" if m["best"] is not None else "  -  "
        print(f"{t:>4}  {m['family']:<11} {best:>6} {m['iters']:>5} {m['evals']:>5} "
              f"{m['frontier']:>5}  {m['terminated'] or 'running'}")


def _cmd_journal(args) -> None:
    from . import journal as J
    evs = J.read(Path(args.runs_dir) / str(args.task) / "journal.jsonl")
    keys = ("cand", "model", "verdict", "ok", "best", "outcome", "tier", "trigger", "reason", "strategy")
    for e in evs:
        bits = " ".join(f"{k}={e[k]}" for k in keys if k in e and e[k] not in (None, ""))
        print(f"{(e.get('ts') or '')[:19]}  {e.get('ev',''):<14} {bits}")


def _cmd_frontier(args) -> None:
    m = _run_metrics(Path(args.runs_dir), args.task)
    accepted = [c for c in m["candidates"] if c["status"] == "accepted"]
    print(f"task {args.task} ({m['family']}) — best {m['best']:.3f}, frontier size {m['frontier']}, "
          f"{len(accepted)} candidate(s) entered")
    for c in accepted:
        sc = f"{c['sol_score']:.3f}" if c["sol_score"] is not None else "  -  "
        print(f"  {c['cand'][:10]}  score={sc}  {c['model']:<6}  {c['strategy']}")


def _cmd_candidates(args) -> None:
    m = _run_metrics(Path(args.runs_dir), args.task)
    for c in m["candidates"]:
        if args.status and c["status"] != args.status:
            continue
        sc = f"{c['sol_score']:.3f}" if c["sol_score"] is not None else "  -  "
        print(f"{c['cand'][:10]}  {c['status']:<10} score={sc}  {c['model']:<6}  {c['strategy']}")


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

    p_solve = sub.add_parser("solve", help="run the engine over selected tasks")
    _add_selection_args(p_solve)
    p_solve.add_argument("--agent", default="sim",
                         help="sim (deterministic, no GPU) | codex | claude | <any CliSpec>")
    p_solve.add_argument("--model", default="gpt-5.5", help="model for a real --agent (e.g. gpt-5.5)")
    p_solve.add_argument("--runs-dir", default="runs")
    p_solve.add_argument("--knowledge-dir", default="knowledge")
    p_solve.add_argument("--max-iters", type=int, default=40, help="per-problem iteration cap")
    p_solve.add_argument("--max-evals", type=int, default=30, help="per-problem GPU-eval cap")
    p_solve.add_argument("--timeout", type=float, default=600.0, help="per agent-call timeout (s)")
    p_solve.add_argument("--delay", type=float, default=0.006,
                         help="simulated per-eval GPU time (spreads the timeline)")
    p_solve.set_defaults(func=_cmd_solve)

    p_stat = sub.add_parser("status", help="per-problem summary over a runs dir")
    p_stat.add_argument("tasks", nargs="*", help="task ids/ranges (default: all in runs dir)")
    p_stat.add_argument("--runs-dir", default="runs")
    p_stat.set_defaults(func=_cmd_status)

    p_jrnl = sub.add_parser("journal", help="print a problem's event timeline")
    p_jrnl.add_argument("task", type=int)
    p_jrnl.add_argument("--runs-dir", default="runs")
    p_jrnl.set_defaults(func=_cmd_journal)

    p_front = sub.add_parser("frontier", help="print a problem's frontier (accepted candidates)")
    p_front.add_argument("task", type=int)
    p_front.add_argument("--runs-dir", default="runs")
    p_front.set_defaults(func=_cmd_frontier)

    p_cand = sub.add_parser("candidates", help="list a problem's candidates")
    p_cand.add_argument("task", type=int)
    p_cand.add_argument("--status", default=None, help="filter by status (accepted/dominated/rejected/...)")
    p_cand.add_argument("--runs-dir", default="runs")
    p_cand.set_defaults(func=_cmd_candidates)

    p_report = sub.add_parser("report", help="render the run dashboard (static site)")
    p_report.add_argument("--runs-dir", default="runs")
    p_report.add_argument("--out-dir", default=None,
                          help="output site dir (default out/, publishable as-is)")
    p_report.add_argument("--refresh", type=int, default=None, help="meta auto-refresh seconds")
    p_report.add_argument("--watch", type=int, default=None, help="regenerate every N seconds")
    p_report.add_argument("--demo", action="store_true",
                          help="render synthetic demo runs under .cache/demo/")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
