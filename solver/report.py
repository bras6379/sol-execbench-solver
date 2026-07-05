"""Render the run dashboard: journals -> one self-contained static HTML.

No server, no CDN, no JS libraries — inline CSS + SVG + ~60 lines of vanilla
JS (tooltips, theme override). Light/dark via prefers-color-scheme, with
``?theme=light|dark`` override for screenshot testing. Palette = validated
reference instance (see kb dataviz notes); series identity is fixed by
problem order, never cycled.
"""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

from . import journal as journal_mod
from . import metrics as metrics_mod

# Validated categorical slots (light, dark) — fixed order, identity per problem.
SERIES = [
    ("#2a78d6", "#3987e5"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500"),
    ("#008300", "#008300"), ("#4a3aa7", "#9085e9"), ("#e34948", "#e66767"),
    ("#e87ba4", "#d55181"), ("#eb6834", "#d95926"),
]
OUTCOME_COLORS = {          # semantic, fixed; never reused for series
    "accepted": ("#0ca30c", "#0ca30c"),
    "dominated": ("#9c9a92", "#7c7a72"),
    "rejected": ("#eb6834", "#d95926"),
    "duplicate": ("#eda100", "#c98500"),
    "incorrect": ("#e87ba4", "#d55181"),
    "error": ("#d03b3b", "#d03b3b"),
}


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _fmt_s(sec: float | None) -> str:
    if sec is None:
        return "–"
    if sec < 90:
        return f"{sec:.1f}s"
    if sec < 5400:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.1f}h"


def _tile(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="tile"><div class="tile-v">{value}</div>'
            f'<div class="tile-l">{_esc(label)}</div>'
            + (f'<div class="tile-s">{_esc(sub)}</div>' if sub else "") + "</div>")


# ---------------------------------------------------------------- charts ----

def _convergence_svg(problems: list[dict]) -> str:
    """Best sol_score vs GPU-eval index, one line per problem."""
    W, H, L, R, T, B = 960, 320, 46, 150, 16, 30
    pw, ph = W - L - R, H - T - B
    max_x = max((p["convergence"][-1][0] for p in problems if p["convergence"]),
                default=1) or 1

    def X(x): return L + pw * x / max_x
    def Y(y): return T + ph * (1 - y)

    grid = "".join(
        f'<line x1="{L}" y1="{Y(v)}" x2="{L + pw}" y2="{Y(v)}" class="grid"/>'
        f'<text x="{L - 8}" y="{Y(v) + 4}" class="ax" text-anchor="end">{v:.2f}</text>'
        for v in (0, 0.25, 0.5, 0.75, 1.0))
    # reference lines: baseline 0.5 and SOL 1.0
    refs = (f'<line x1="{L}" y1="{Y(0.5)}" x2="{L + pw}" y2="{Y(0.5)}" class="ref"/>'
            f'<text x="{L + pw + 6}" y="{Y(0.5) + 4}" class="ax">baseline 0.5</text>'
            f'<line x1="{L}" y1="{Y(1.0)}" x2="{L + pw}" y2="{Y(1.0)}" class="ref sol"/>'
            f'<text x="{L + pw + 6}" y="{Y(1.0) + 4}" class="ax">SOL 1.0</text>')
    lines, labels = [], []
    for i, p in enumerate(problems):
        if not p["convergence"]:
            continue
        c = f"var(--s{i % len(SERIES) + 1})"
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in p["convergence"])
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{c}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
        lx, ly = p["convergence"][-1]
        labels.append(f'<circle cx="{X(lx):.1f}" cy="{Y(ly):.1f}" r="3.5" fill="{c}"/>'
                      f'<text x="{X(lx) + 7:.1f}" y="{Y(ly) - 6:.1f}" class="dl" '
                      f'fill="{c}">#{p["task"]} {ly:.2f}</text>')
    # hover columns: one per eval index with all series values
    hovers = []
    for xi in range(0, max_x + 1):
        vals = []
        for p in problems:
            seq = dict(p["convergence"])
            best = None
            for k in range(xi, -1, -1):
                if k in seq:
                    best = seq[k]
                    break
            if best is not None:
                vals.append(f"#{p['task']}: {best:.3f}")
        tip = f"eval {xi} • " + " · ".join(vals) if vals else f"eval {xi}"
        x0 = X(max(xi - 0.5, 0))
        x1 = X(min(xi + 0.5, max_x))
        hovers.append(f'<rect x="{x0:.1f}" y="{T}" width="{x1 - x0:.1f}" '
                      f'height="{ph}" class="hover-col" data-tip="{_esc(tip)}"/>')
    xt = "".join(f'<text x="{X(v)}" y="{H - 10}" class="ax" text-anchor="middle">{v}</text>'
                 for v in range(0, max_x + 1, max(1, max_x // 8)))
    xcap = (f'<text x="{W - 4}" y="{H - 10}" class="ax" text-anchor="end">'
            f'GPU evals →</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="convergence">'
            f'{grid}{refs}{"".join(lines)}{"".join(labels)}{"".join(hovers)}{xt}{xcap}</svg>')


def _timeline_svg(fleet: dict, problems: list[dict]) -> str:
    """One-row GPU occupancy strip: segments per job, colored by problem."""
    W, H, L, R = 960, 96, 46, 20
    T, BH = 22, 34
    jobs = fleet["jobs"]
    if not jobs:
        return '<p class="muted">no GPU jobs yet</p>'
    t0, t1 = fleet["span_start"], fleet["span_end"]
    span = max(t1 - t0, 1e-9)
    pw = W - L - R
    idx = {p["task"]: i for i, p in enumerate(problems)}

    def X(t): return L + pw * (t - t0) / span
    segs = []
    for j in jobs:
        i = idx.get(j["task"], 0) % len(SERIES)
        x0, x1 = X(j["start"]), X(j["done"])
        wait = (j["start"] - j["enq"]) if "enq" in j else None
        tip = (f"#{j['task']} • run {_fmt_s(j['done'] - j['start'])}"
               + (f" • waited {_fmt_s(wait)}" if wait is not None else ""))
        segs.append(f'<rect x="{x0:.1f}" y="{T}" width="{max(x1 - x0, 1.2):.1f}" '
                    f'height="{BH}" rx="2" fill="var(--s{i + 1})" class="seg" '
                    f'data-tip="{_esc(tip)}"/>')
    ticks = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        tt = t0 + frac * span
        label = dt.datetime.fromtimestamp(tt).strftime("%H:%M")
        ticks.append(f'<text x="{L + pw * frac}" y="{H - 10}" class="ax" '
                     f'text-anchor="middle">{label}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="gpu timeline">'
            f'<rect x="{L}" y="{T}" width="{pw}" height="{BH}" class="lane"/>'
            f'{"".join(segs)}{"".join(ticks)}</svg>')


def _waits_svg(problems: list[dict]) -> str:
    """Per-problem queue wait: p50 bar with a thin p95 whisker."""
    rows = [p for p in problems if p["wait_p50"] is not None]
    if not rows:
        return '<p class="muted">no queue data yet</p>'
    W, L, R, RH = 960, 170, 60, 26
    H = 16 + RH * len(rows) + 8
    maxv = max(p["wait_p95"] or p["wait_p50"] for p in rows) or 1
    pw = W - L - R
    out = []
    for r, p in enumerate(rows):
        y = 12 + r * RH
        i = problems.index(p) % len(SERIES)
        w50 = pw * (p["wait_p50"] / maxv)
        w95 = pw * ((p["wait_p95"] or p["wait_p50"]) / maxv)
        tip = f"#{p['task']} wait p50 {_fmt_s(p['wait_p50'])} • p95 {_fmt_s(p['wait_p95'])}"
        out.append(
            f'<text x="{L - 8}" y="{y + 13}" class="ax" text-anchor="end">#{p["task"]} {p["name"][:16]}</text>'
            f'<line x1="{L}" y1="{y + 9}" x2="{L + w95:.1f}" y2="{y + 9}" class="whisk"/>'
            f'<rect x="{L}" y="{y + 2}" width="{max(w50, 1.5):.1f}" height="14" rx="3" '
            f'fill="var(--s{i + 1})" class="seg" data-tip="{_esc(tip)}"/>'
            f'<text x="{L + w95 + 6:.1f}" y="{y + 13}" class="dl2">{_fmt_s(p["wait_p50"])}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="queue waits">{"".join(out)}</svg>'


def _outcomes_svg(problems: list[dict]) -> str:
    """Per-problem stacked bar of iteration outcomes (2px gaps)."""
    rows = [p for p in problems if sum(p["outcomes"].values())]
    if not rows:
        return '<p class="muted">no iterations yet</p>'
    W, L, R, RH = 960, 170, 60, 26
    H = 16 + RH * len(rows) + 8
    maxn = max(sum(p["outcomes"].values()) for p in rows)
    pw = W - L - R
    out = []
    for r, p in enumerate(rows):
        y = 12 + r * RH
        x = float(L)
        total = sum(p["outcomes"].values())
        for key in metrics_mod.OUTCOMES:
            n = p["outcomes"][key]
            if not n:
                continue
            w = pw * n / maxn
            tip = f"#{p['task']} {key}: {n}"
            out.append(f'<rect x="{x:.1f}" y="{y + 2}" width="{max(w - 2, 1.2):.1f}" '
                       f'height="14" rx="3" fill="var(--o-{key})" class="seg" '
                       f'data-tip="{_esc(tip)}"/>')
            x += w
        out.append(
            f'<text x="{L - 8}" y="{y + 13}" class="ax" text-anchor="end">#{p["task"]} {p["name"][:16]}</text>'
            f'<text x="{x + 6:.1f}" y="{y + 13}" class="dl2">{total}</text>')
    legend = "".join(
        f'<span class="lg"><i style="background:var(--o-{k})"></i>{k}</span>'
        for k in metrics_mod.OUTCOMES)
    return (f'<div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="outcomes">{"".join(out)}</svg>')


def _problem_table(problems: list[dict]) -> str:
    rows = []
    for i, p in enumerate(problems):
        best = p["best"]
        bar = ""
        if best is not None:
            pct = max(0.0, min(best, 1.0)) * 100
            bar = (f'<div class="bar"><i style="width:{pct:.1f}%;'
                   f'background:var(--s{i % len(SERIES) + 1})"></i><b></b></div>')
        status = p["terminated"] or "running"
        rows.append(
            "<tr>"
            f'<td><span class="dot" style="background:var(--s{i % len(SERIES) + 1})"></span>'
            f'#{p["task"]}</td>'
            f"<td>{_esc(p['name'])}</td><td>{_esc(p['family'])}</td>"
            f"<td>{_esc(p['model'])}</td>"
            f'<td class="{ "done" if p["terminated"] else "run"}">{_esc(status)}</td>'
            f"<td>{p['iters']}</td><td>{p['evals']}</td><td>{p['frontier']}</td>"
            f"<td>{'' if best is None else f'{best:.3f}'}{bar}</td>"
            f"<td>{_fmt_s(p['wait_p50'])}</td>"
            "</tr>")
    return ("<table><thead><tr><th>task</th><th>name</th><th>family</th>"
            "<th>agent</th><th>status</th><th>iters</th><th>evals</th>"
            "<th>frontier</th><th>best SOL score</th><th>wait p50</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


# ------------------------------------------------------------------ page ----

_CSS = """
:root{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a887f;
--grid:#e8e7e2;--ref:#c9c7bf;
--s1:#2a78d6;--s2:#1baf7a;--s3:#eda100;--s4:#008300;--s5:#4a3aa7;--s6:#e34948;--s7:#e87ba4;--s8:#eb6834;
--o-accepted:#0ca30c;--o-dominated:#9c9a92;--o-rejected:#eb6834;--o-duplicate:#eda100;--o-incorrect:#e87ba4;--o-error:#d03b3b;}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--panel:#222221;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8a887f;
--grid:#33322f;--ref:#4a4945;
--s1:#3987e5;--s2:#199e70;--s3:#c98500;--s4:#008300;--s5:#9085e9;--s6:#e66767;--s7:#d55181;--s8:#d95926;
--o-dominated:#7c7a72;--o-rejected:#d95926;--o-duplicate:#c98500;--o-incorrect:#d55181;}}
:root[data-theme=light]{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a887f;--grid:#e8e7e2;--ref:#c9c7bf;
--s1:#2a78d6;--s2:#1baf7a;--s3:#eda100;--s4:#008300;--s5:#4a3aa7;--s6:#e34948;--s7:#e87ba4;--s8:#eb6834;
--o-dominated:#9c9a92;--o-rejected:#eb6834;--o-duplicate:#eda100;--o-incorrect:#e87ba4;}
:root[data-theme=dark]{--surface:#1a1a19;--panel:#222221;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8a887f;--grid:#33322f;--ref:#4a4945;
--s1:#3987e5;--s2:#199e70;--s3:#c98500;--s4:#008300;--s5:#9085e9;--s6:#e66767;--s7:#d55181;--s8:#d95926;
--o-dominated:#7c7a72;--o-rejected:#d95926;--o-duplicate:#c98500;--o-incorrect:#d55181;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:24px}
h1{font-size:19px;margin:0 0 2px}
h2{font-size:14px;font-weight:600;margin:0 0 10px;color:var(--ink)}
.sub{color:var(--ink3);font-size:12px;margin-bottom:20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.tile{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:14px 16px}
.tile-v{font-size:24px;font-weight:650;letter-spacing:-.02em}
.tile-l{color:var(--ink2);font-size:12px;margin-top:2px}
.tile-s{color:var(--ink3);font-size:11px;margin-top:2px}
.panel{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:16px 18px;margin-bottom:16px;overflow-x:auto}
svg{display:block;width:100%;height:auto}
.grid{stroke:var(--grid);stroke-width:1}
.ref{stroke:var(--ref);stroke-width:1;stroke-dasharray:4 4}
.ref.sol{stroke:var(--o-accepted)}
.lane{fill:var(--grid);opacity:.35;rx:3}
.ax{fill:var(--ink3);font-size:11px}
.dl{font-size:11px;font-weight:600}
.dl2{fill:var(--ink2);font-size:11px}
.seg{cursor:pointer}
.seg:hover{opacity:.82}
.hover-col{fill:transparent;cursor:crosshair}
.hover-col:hover{fill:var(--ink);opacity:.05}
.whisk{stroke:var(--ink3);stroke-width:2;opacity:.55;stroke-linecap:round}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;font-size:12px;color:var(--ink2)}
.lg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{color:var(--ink3);font-weight:600;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
th,td{padding:7px 12px 7px 0;border-bottom:1px solid var(--grid);white-space:nowrap}
td .bar{position:relative;height:5px;background:var(--grid);border-radius:3px;margin-top:4px;min-width:130px}
td .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
td .bar b{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--ref)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.done{color:var(--ink3)} .run{color:var(--o-accepted);font-weight:600}
.muted{color:var(--ink3)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);
padding:5px 9px;border-radius:6px;font-size:12px;opacity:0;transition:opacity .08s;z-index:9;max-width:420px;white-space:normal}
"""

_JS = """
const q=new URLSearchParams(location.search).get('theme');
if(q)document.documentElement.dataset.theme=q;
const tip=document.createElement('div');tip.id='tip';document.body.appendChild(tip);
document.addEventListener('mousemove',e=>{
  const el=e.target.closest('[data-tip]');
  if(el){tip.textContent=el.dataset.tip;tip.style.opacity=1;
    const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);
    tip.style.left=x+'px';tip.style.top=(e.clientY+16)+'px';}
  else tip.style.opacity=0;});
"""


def build_html(data: dict, *, refresh: int | None = None,
               generated_at: str | None = None) -> str:
    problems, fleet = data["problems"], data["fleet"]
    mean_best = fleet["mean_best"]
    gen = generated_at or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    tiles = "".join([
        _tile("fleet SOL score (mean of best)",
              "–" if mean_best is None else f"{mean_best:.3f}",
              "0.5 = baseline · 1.0 = SOL"),
        _tile("GPU utilization", f"{fleet['gpu_util'] * 100:.0f}%",
              f"busy {_fmt_s(fleet['busy_s'])} of {_fmt_s(fleet['span_s'])}"),
        _tile("queue wait p50 / p95",
              f"{_fmt_s(fleet['wait_p50'])} / {_fmt_s(fleet['wait_p95'])}"),
        _tile("GPU evals", str(fleet["total_evals"])),
        _tile("problems", f"{fleet['active']} active · {fleet['done']} done"),
        _tile("agent calls / tokens",
              f"{fleet['agent_calls']} / {fleet['agent_tokens']:,}"),
    ])
    legend = "".join(
        f'<span class="lg"><i style="background:var(--s{i % len(SERIES) + 1})"></i>'
        f"#{p['task']} {_esc(p['name'][:22])}</span>"
        for i, p in enumerate(problems))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{meta}
<title>SOL-ExecBench solver — run dashboard</title>
<style>{_CSS}</style></head><body>
<h1>SOL-ExecBench solver — run dashboard</h1>
<div class="sub">generated {gen}{' · auto-refresh ' + str(refresh) + 's' if refresh else ''}</div>
<div class="tiles">{tiles}</div>
<div class="panel"><h2>Convergence — best SOL score per problem (↑ toward 1.0)</h2>
<div class="legend">{legend}</div>{_convergence_svg(problems)}</div>
<div class="panel"><h2>GPU occupancy — one job at a time, colored by problem</h2>
{_timeline_svg(fleet, problems)}</div>
<div class="panel"><h2>Queue wait per problem (bar = p50, whisker = p95)</h2>
{_waits_svg(problems)}</div>
<div class="panel"><h2>Iteration outcomes</h2>{_outcomes_svg(problems)}</div>
<div class="panel"><h2>Problems</h2>{_problem_table(problems)}</div>
<script>{_JS}</script></body></html>"""


def render(runs_dir: Path, out: Path, *, refresh: int | None = None) -> Path:
    journals = journal_mod.read_all(Path(runs_dir))
    data = metrics_mod.collect(journals)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(build_html(data, refresh=refresh))
    tmp.replace(out)
    return out
