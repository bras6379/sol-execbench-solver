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
import os
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
            try:
                report_mod.render(runs_dir, out_dir, refresh=args.refresh or args.watch)
            except Exception as exc:                    # a transient render error must not kill the watcher
                print(f"[watch] render failed ({exc!r}); retrying next cycle")
            time.sleep(args.watch)
    path = report_mod.render(runs_dir, out_dir, refresh=args.refresh)
    print(f"dashboard site -> {path}")


def _parse_tiers(specs: list[str]):
    """Parse `--tier NAME=agent/model[,agent/model...]` into Tier objects."""
    from .engine import Perspective, Tier
    tiers = []
    for s in specs:
        name, sep, pool_s = s.partition("=")
        if not sep or not pool_s.strip():
            raise SystemExit(f"bad --tier {s!r}; use NAME=agent/model[,agent/model]")
        pool = []
        for pm in pool_s.split(","):
            agent, _, model = pm.strip().partition("/")
            if not agent or not model:
                raise SystemExit(f"bad perspective {pm!r} in --tier; use agent/model")
            pool.append(Perspective(agent.strip(), model.strip()))
        tiers.append(Tier(name.strip(), pool))
    return tiers


def _cmd_solve(args) -> None:
    import asyncio

    from . import journal as J
    from .dashboard import metrics
    from .engine import (Config, KnowledgeStore, Perspective, StubExecutor, Tier,
                         make_agents, reference_seed, run_fleet, sim, stub_agents)

    ids = _resolve_ids(args)
    runs_dir = Path(args.runs_dir)
    knowledge = KnowledgeStore(args.knowledge_dir)
    families = {t: sim.family_of(t) for t in ids}
    names = {t: f"{sim.family_of(t)}_{t}" for t in ids}

    if args.agent == "sim" and not args.tier:
        if args.gpu:
            raise SystemExit("--gpu needs a real --agent (codex/claude), not the sim agent")
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
        if args.tier:
            # multi-tier ladder (cheap → strong), escalates on plateau (§6b)
            tiers = _parse_tiers(args.tier)
        else:
            tiers = [Tier(args.agent, [Perspective(args.agent, args.model)])]
        tl = args.time_limit_min * 60 if args.time_limit_min else None
        big = 10 ** 9
        cfg = Config(
            tiers=tiers,
            # with a wall-clock budget, keep iterating until time runs out: don't let
            # the iter/eval caps or a plateau stop the problem early.
            plateau_cycles=(big if tl else args.plateau_cycles), escalate_ceiling=0.9,
            epsilon=0.02, max_iterations=(big if tl else args.max_iters),
            max_gpu_evals=(big if tl else args.max_evals), time_limit_s=tl,
            verify_runs=args.verify_runs, agent_fail_limit=args.agent_fail_limit,
        )
        agents = make_agents(cfg, runs_dir=runs_dir, timeout=args.timeout)
        seeds_fn = reference_seed()   # seed the frontier with the real reference impl
        ladder = " → ".join(t.name + "(" + ",".join(str(p) for p in t.pool) + ")" for t in cfg.tiers)
        print(f"ladder: {ladder}  ·  design: {cfg.design_model}")

    if args.gpu:
        # Real end-to-end: rent an ephemeral B200, bootstrap the harness, score
        # every candidate on the GPU, then terminate the pod (guaranteed teardown).
        from .engine.gpu_run import solve_on_gpu
        from .engine.pod import PodSpec, RunPodProvider
        api = os.environ.get("RUNPOD_API_KEY")
        if not api:
            raise SystemExit("RUNPOD_API_KEY not set (add it to .env)")
        spec = PodSpec(gpu_type=args.gpu_type, cloud_type=args.gpu_cloud)
        hcfg = {"warmup_runs": 10, "iterations": args.gpu_iterations,
                "lock_clocks": False, "seed": 200}   # containers can't lock; we measure unlocked
        who = args.tier and "the ladder above" or f"`{args.agent}`/{args.model}"
        print(f"solving {len(ids)} problem(s) with {who} on a rented "
              f"{args.gpu_type} via RunPod (auto-provision → bootstrap → run → terminate) -> {runs_dir}/")
        asyncio.run(solve_on_gpu(ids, agents, cfg, runs_dir=runs_dir, seeds_fn=seeds_fn,
                                 knowledge=knowledge, families=families, names=names,
                                 provider=RunPodProvider(api), spec=spec, config=hcfg,
                                 max_concurrency=args.max_concurrency, shuffle=args.shuffle,
                                 reflect_first=args.reflect_first, reflect_every_min=args.reflect_every_min,
                                 reflect_model=args.reflect_model,
                                 max_lifetime_min=(args.gpu_max_hours * 60 if args.gpu_max_hours else None)))
    else:
        if args.agent != "sim":
            print(f"  (StubExecutor — no GPU scoring; add --gpu for real B200 evaluation)")
        # --fake-scores: score real agent kernels by content hash so the frontier /
        # convergence exercises without a GPU (real agents have no embedded scores).
        outcome = sim.hash_score_outcome() if args.fake_scores else None
        executor = StubExecutor(outcome, delay=args.delay)   # delay → a real timeline
        asyncio.run(run_fleet(ids, executor, agents, cfg, runs_dir=runs_dir,
                              seeds_fn=seeds_fn, knowledge=knowledge,
                              families=families, names=names,
                              max_concurrency=args.max_concurrency, shuffle=args.shuffle,
                              reflect_first=args.reflect_first, reflect_every_min=args.reflect_every_min,
                              reflect_model=args.reflect_model))
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
    base = Path(args.runs_dir) / str(args.task)
    for label, f in (("frontier", base / "frontier.json"),
                     ("submit  ", base / "best_solution.json"),
                     ("candidates", base / "candidates")):
        if f.exists():
            print(f"  {label}: {f}")


def _submissions_log(runs_dir: Path, task: int) -> Path:
    return Path(runs_dir) / str(task) / "submissions.jsonl"


def _record_submission(runs_dir: Path, task: int, event: dict) -> None:
    p = _submissions_log(runs_dir, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _submitted_cand(runs_dir: Path, task: int) -> dict:
    """Which frontier kernel is in best_solution.json right now (cand + strategy)."""
    fr = runs_dir / str(task) / "frontier.json"
    if not fr.exists():
        return {}
    d = json.loads(fr.read_text())
    m = next((x for x in d.get("members", []) if x.get("cand_id") == d.get("best_cand")), {})
    return {"cand_id": d.get("best_cand"), "cand_strategy": m.get("strategy"),
            "cand_agent": m.get("model")}


def _board_enrich(row: dict, kernel: int) -> dict:
    """Add leaderboard rank + #1 score to a poll row (best-effort GET, no crash)."""
    from .bench import leaderboard as lb
    try:
        b = lb.board(kernel)
        return {**row, "board_rank": lb.rank_of(row.get("sol_score"), b["rankings"]),
                "board_n": b["n"], "board_top_sol": b["top_sol"], "board_top_user": b["top_user"]}
    except Exception:
        return row


def _cmd_submit(args) -> None:
    import time

    from .bench import leaderboard as lb

    task = args.task
    kernel = args.kernel if args.kernel is not None else task     # kernel_id == task_id
    runs_dir = Path(args.runs_dir)
    file_path = Path(args.file) if args.file else runs_dir / str(task) / "best_solution.json"
    if not file_path.exists():
        raise SystemExit(f"no solution to submit at {file_path} (run `solver solve --gpu {task}` first, "
                         f"or pass --file)")
    print(f"submitting {file_path}  →  kernel {kernel} ({args.mode}, {args.gpu})")
    resp = lb.submit(kernel, file_path, gpu=args.gpu, mode=args.mode)
    sid = resp.get("submission_id") or resp.get("id")
    _record_submission(runs_dir, task, {"event": "submit", "submission_id": sid,
                                        "kernel_id": kernel, "file": str(file_path),
                                        "mode": args.mode, "message": resp.get("message"),
                                        **_submitted_cand(runs_dir, task)})   # which kernel
    print(f"submitted → #{sid}")
    if args.poll and sid:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            row = lb.poll(sid)
            print("  " + lb.format_result(row))
            if row.get("status") in lb.TERMINAL:
                row = _board_enrich(row, kernel)
                if row.get("board_rank"):
                    print(f"  → would rank #{row['board_rank']} of {row['board_n']}; "
                          f"#1 is {row['board_top_user']} at SOL {row['board_top_sol']}")
                _record_submission(runs_dir, task, {"event": "poll", **row})
                break
            time.sleep(args.poll_interval)


def _cmd_poll(args) -> None:
    from .bench import leaderboard as lb

    runs_dir = Path(args.runs_dir)
    if getattr(args, "all", False):
        # Refresh EVERY recorded submission against the leaderboard (dashboard's real
        # SOL / rank columns read the recorded poll results). Keeps queued submissions
        # up to date as NVIDIA's worker processes them.
        seen: dict[int, int] = {}                    # submission_id -> task
        for sf in sorted(runs_dir.glob("*/submissions.jsonl")):
            try:
                task = int(sf.parent.name)
            except ValueError:
                continue
            for line in sf.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = e.get("submission_id") or e.get("id")
                if sid:
                    seen[int(sid)] = task
        if not seen:
            print("no recorded submissions found"); return
        done = queued = failed = 0
        print(f"refreshing {len(seen)} submission(s) against the leaderboard ...")
        for sid, task in sorted(seen.items()):
            try:
                row = lb.poll(int(sid))
                if row.get("kernel_id") is not None:
                    row = _board_enrich(row, row["kernel_id"])
                _record_submission(runs_dir, task, {"event": "poll", **row})
                st = row.get("status", "?")
                done += st == "COMPLETED"; queued += st in ("QUEUED", "PENDING_RESULT", "RUNNING")
                rank = f"  → rank #{row['board_rank']}/{row['board_n']}" if row.get("board_rank") else ""
                print(f"  task {task:>3}  " + lb.format_result(row) + rank)
            except Exception as exc:                 # a dead/foreign-account submission must not abort the sweep
                failed += 1
                print(f"  task {task:>3}  #{sid}  poll failed: {repr(exc)[:80]}")
        print(f"done: {done} completed, {queued} still queued, {failed} unreachable")
        # cache each problem's leaderboard #1 (the target to beat) for the dashboard
        boards = {}
        for pdir in sorted((p for p in runs_dir.glob("*") if p.is_dir() and p.name.isdigit()),
                           key=lambda p: int(p.name)):
            t = int(pdir.name)
            try:
                b = lb.board(t)
                if b.get("top_sol") is not None:
                    # cache the full sol_score distribution (desc) so the dashboard can
                    # project what rank a given expected SOL would take on this board.
                    scores = sorted((e["sol_score"] for e in b.get("rankings") or []
                                     if e.get("sol_score") is not None), reverse=True)
                    boards[str(t)] = {"top_sol": b["top_sol"], "top_user": b["top_user"],
                                      "n": b["n"], "sol_bound": b.get("sol_bound"),
                                      "scores": scores}
            except Exception:
                continue
        if boards:
            (runs_dir / "leaderboard.json").write_text(json.dumps(boards, indent=2))
            print(f"cached leaderboard #1 for {len(boards)} problem(s) -> {runs_dir}/leaderboard.json")
        return

    sids = args.ids
    if args.task and not sids:                       # latest submission for a task
        evs = [json.loads(l) for l in _submissions_log(Path(args.runs_dir), args.task).read_text().splitlines()] \
            if _submissions_log(Path(args.runs_dir), args.task).exists() else []
        sids = [max((e["submission_id"] for e in evs if e.get("event") == "submit" and e.get("submission_id")),
                    default=None)]
        sids = [s for s in sids if s]
    if not sids:
        raise SystemExit("give submission id(s) or --task <n> (with a prior submit)")
    for sid in sids:
        row = lb.poll(int(sid))
        if row.get("kernel_id") is not None:
            row = _board_enrich(row, row["kernel_id"])
        print(lb.format_result(row)
              + (f"  → rank #{row['board_rank']}/{row['board_n']}, #1={row['board_top_user']} "
                 f"@ {row['board_top_sol']}" if row.get("board_rank") else ""))
        if args.task:
            _record_submission(Path(args.runs_dir), args.task, {"event": "poll", **row})


def _cmd_export(args) -> None:
    """Collect every problem's best_solution.json into one submission dir + manifest."""
    import shutil

    runs_dir = Path(args.runs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for best in sorted(runs_dir.glob("*/best_solution.json"), key=lambda p: int(p.parent.name)):
        task = int(best.parent.name)
        fr = best.parent / "frontier.json"
        meta = json.loads(fr.read_text()) if fr.exists() else {}
        name = meta.get("name") or f"task-{task}"
        dest = out / f"{task:03d}_{Path(name).name}.json"
        shutil.copyfile(best, dest)
        manifest.append({"task": task, "name": name, "family": meta.get("family"),
                         "best_score": meta.get("best_score"), "best_cand": meta.get("best_cand"),
                         "frontier_size": meta.get("size"), "solution": dest.name})
        print(f"{task:>3}  {name:<28} score={meta.get('best_score')}  -> {dest}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    scored = [m["best_score"] for m in manifest if m["best_score"] is not None]
    mean = sum(scored) / len(scored) if scored else 0.0
    print(f"\n{len(manifest)} solution(s) -> {out}/  ·  mean best {mean:.3f}  ·  manifest.json written")


def _cmd_reap(args) -> None:
    """Kill switch / cron backstop: terminate EVERY B200 pod this tool created
    (the `sol-solver` tag) via the RunPod API, so billing stops even if a solve
    process died uncleanly. Safe to run anytime — no-op if nothing is running."""
    import asyncio
    from .engine.pod import PodSpec, RunPodProvider
    api = os.environ.get("RUNPOD_API_KEY")
    if not api:
        raise SystemExit("RUNPOD_API_KEY not set (add it to .env)")
    tag = args.tag or PodSpec().tag
    provider = RunPodProvider(api)

    async def go() -> int:
        pods = await provider.list_tagged(tag)
        for p in pods:
            print(f"terminating pod {p.id} (tag {tag}) ...")
            await provider.terminate(p.id)
        return len(pods)

    n = asyncio.run(go())
    print(f"reaped {n} pod(s) with tag {tag!r}")


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


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ (never overrides an
    already-set var; skips blanks/comments/empty values). Agent CLIs inherit it."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if sep and key and val and key not in os.environ:
            os.environ[key] = val


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
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
    p_solve.add_argument("--tier", action="append", default=None,
                         help="multi-tier ladder (repeatable), cheap→strong; escalates on plateau. "
                              "NAME=agent/model[,agent/model]. E.g. "
                              "--tier cheap=claude/haiku,codex/gpt-5.5 --tier strong=claude/opus")
    p_solve.add_argument("--plateau-cycles", type=int, default=2,
                         help="M: full pool-cycles with no ε-gain before escalating a tier")
    p_solve.add_argument("--max-iters", type=int, default=500, help="per-problem iteration cap (large; use --time-limit-min to gate on time instead)")
    p_solve.add_argument("--max-evals", type=int, default=500, help="per-problem GPU-eval cap")
    p_solve.add_argument("--time-limit-min", type=float, default=None,
                         help="wall-clock budget per problem (minutes); when set it's the ONLY stop "
                              "condition — iter/eval caps and plateau are lifted so it keeps trying")
    p_solve.add_argument("--verify-runs", type=int, default=1,
                         help="re-run a would-be frontier entry this many times (fresh evals, same "
                              "grader config) and reject if any run disagrees on correctness; catches "
                              "flaky/racy kernels that pass locally but fail the leaderboard. 1=off, 2-3=harden")
    p_solve.add_argument("--agent-fail-limit", type=int, default=3,
                         help="consecutive plan failures before a perspective is circuit-broken and "
                              "skipped — a dead agent (e.g. Claude/GPT out of credits) stops being used "
                              "and the run downgrades to the healthy models in the pool")
    p_solve.add_argument("--shuffle", action="store_true",
                         help="randomize problem launch order (seeded) so a --max-concurrency window "
                              "is a RANDOM sample of the id range, not always the lowest ids — fairer "
                              "coverage; on a resume, stops strong low ids from starving underworked high ids")
    p_solve.add_argument("--max-concurrency", type=int, default=0,
                         help="cap how many problems run at once (each holds <=1 agent call in flight, "
                              "so this bounds concurrent CLIs + provider streams — the real limit is the "
                              "laptop/rate-limit, NOT the single-flight GPU). 0=unbounded. Excess ids queue "
                              "and start as slots free, so a big range like 1-100 rolls through safely")
    p_solve.add_argument("--timeout", type=float, default=1800.0,
                         help="per agent-call timeout (s); a timeout now skips the iteration, "
                              "not the whole problem")
    p_solve.add_argument("--reflect-first", dest="reflect_first", action="store_true", default=True,
                         help="(default on) regenerate every problem's reflection.md coach card from the "
                              "accumulated journals BEFORE the fleet starts — a restart begins with each "
                              "agent already knowing what's been tried / where it's stuck / where the loss is")
    p_solve.add_argument("--no-reflect-first", dest="reflect_first", action="store_false",
                         help="skip the startup reflection pass")
    p_solve.add_argument("--reflect-every-min", type=float, default=20.0,
                         help="also rebuild coach cards every N minutes during the run so long runs keep "
                              "reflecting on fresh results (0=only at startup)")
    p_solve.add_argument("--reflect-model", default="claude-sonnet-5",
                         help="strong model that reads the tried kernels' SOURCE and adds a why-it's-stuck "
                              "+ one-untried-lever diagnosis to the coach card of STUCK problems (deduped on "
                              "state so spend is bounded; runs in the BACKGROUND, never blocks the GPU; at "
                              "startup + every --reflect-every-min). Uses the same native-auth claude CLI as "
                              "the opus agents. Empty string = deterministic cards only, no LLM spend")
    p_solve.add_argument("--fake-scores", action="store_true",
                         help="score real agent kernels by content hash (no GPU) so the loop exercises")
    p_solve.add_argument("--delay", type=float, default=0.006,
                         help="simulated per-eval GPU time (spreads the timeline)")
    p_solve.add_argument("--gpu", action="store_true",
                         help="rent an ephemeral B200 on RunPod, score candidates for real, then terminate it")
    p_solve.add_argument("--gpu-type", default="NVIDIA B200", help="RunPod GPU type id")
    p_solve.add_argument("--gpu-cloud", default="SECURE", choices=["SECURE", "COMMUNITY", "ALL"],
                         help="RunPod cloud type (SECURE supports the public IP we SSH over)")
    p_solve.add_argument("--gpu-max-hours", type=float, default=None,
                         help="HARD cap on total B200 pod uptime (hours). When hit, the fleet is "
                              "cancelled and the pod terminated via the API (stops billing). The run "
                              "is resumable, so a cut-off is safe. Strongly recommended for real runs")
    p_solve.add_argument("--gpu-iterations", type=int, default=50,
                         help="harness timed iterations per workload (leaderboard uses 50; lower = noisier)")
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

    p_submit = sub.add_parser("submit", help="submit a problem's best_solution.json to the leaderboard")
    p_submit.add_argument("task", type=int, help="task id (also the leaderboard kernel_id)")
    p_submit.add_argument("--file", default=None, help="solution.json (default: runs/<task>/best_solution.json)")
    p_submit.add_argument("--kernel", type=int, default=None, help="leaderboard kernel_id (default: = task)")
    p_submit.add_argument("--gpu", default="B200")
    p_submit.add_argument("--mode", default="private", choices=["private", "release"])
    p_submit.add_argument("--runs-dir", default="runs")
    p_submit.add_argument("--poll", action="store_true", help="wait and poll until the result is ready")
    p_submit.add_argument("--poll-interval", type=float, default=10.0)
    p_submit.add_argument("--timeout", type=float, default=600.0, help="max seconds to poll")
    p_submit.set_defaults(func=_cmd_submit)

    p_poll = sub.add_parser("poll", help="poll leaderboard submission status/score")
    p_poll.add_argument("ids", nargs="*", type=int, help="submission id(s)")
    p_poll.add_argument("--task", type=int, default=None, help="poll the latest submission for this task")
    p_poll.add_argument("--all", action="store_true",
                        help="refresh EVERY recorded submission (all runs/*/submissions.jsonl) against the "
                             "leaderboard and update the dashboard's real-SOL/rank columns")
    p_poll.add_argument("--runs-dir", default="runs")
    p_poll.set_defaults(func=_cmd_poll)

    p_reap = sub.add_parser("reap", help="terminate all B200 pods this tool created (kill switch / cron backstop)")
    p_reap.add_argument("--tag", default=None, help="pod tag to reap (default: sol-solver)")
    p_reap.set_defaults(func=_cmd_reap)

    p_export = sub.add_parser("export", help="collect each problem's best_solution.json into a submission dir")
    p_export.add_argument("--runs-dir", default="runs")
    p_export.add_argument("--out-dir", default="submissions", help="where to write the bundle (default: submissions/)")
    p_export.set_defaults(func=_cmd_export)

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
