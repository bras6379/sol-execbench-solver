"""Render the run dashboard: journals -> static HTML (hub + per-problem pages).

Scales to all 235 problems: the hub shows distributions and top movers (never
235 lines on one chart); every problem links to its own detail page. GPU
utilization and the occupancy timeline are computed against RENTED windows
when `<runs>/gpu_rentals.jsonl` exists (un-rented gaps compressed).

Self-contained: inline CSS + SVG + small vanilla JS (tooltips, table
sort/filter, ?theme= override). Palette = validated reference instance;
series identity fixed by problem order, never cycled.
"""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

from . import journal as journal_mod
from . import metrics as metrics_mod

SERIES_N = 8
OUTCOME_KEYS = metrics_mod.OUTCOMES


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


def _hm(t: float) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%H:%M")


def _slot(i: int) -> str:
    return f"var(--s{i % SERIES_N + 1})"


def _tile(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="tile"><div class="tile-v">{value}</div>'
            f'<div class="tile-l">{_esc(label)}</div>'
            + (f'<div class="tile-s">{_esc(sub)}</div>' if sub else "") + "</div>")


# ---------------------------------------------------------------- charts ----

def _line_chart(series: list[dict], *, x_label: str, x_is_time: bool = False,
                height: int = 300, right: int = 150) -> str:
    """Generic multi-line chart. series item: {label, color, points[(x,y)]}."""
    W, H, L, R, T, B = 960, height, 46, right, 14, 30
    pw, ph = W - L - R, H - T - B
    xs = [x for s in series for x, _ in s["points"]]
    if not xs:
        return '<p class="muted">no data yet</p>'
    x0, x1 = min(xs), max(xs)
    if x1 <= x0:
        x1 = x0 + 1

    def X(x): return L + pw * (x - x0) / (x1 - x0)
    def Y(y): return T + ph * (1 - y)

    grid = "".join(
        f'<line x1="{L}" y1="{Y(v)}" x2="{L + pw}" y2="{Y(v)}" class="grid"/>'
        f'<text x="{L - 8}" y="{Y(v) + 4}" class="ax" text-anchor="end">{v:.2f}</text>'
        for v in (0, 0.25, 0.5, 0.75, 1.0))
    refs = (f'<line x1="{L}" y1="{Y(0.5)}" x2="{L + pw}" y2="{Y(0.5)}" class="ref"/>'
            f'<text x="{L + pw + 6}" y="{Y(0.5) + 4}" class="ax">baseline 0.5</text>'
            f'<line x1="{L}" y1="{Y(1.0)}" x2="{L + pw}" y2="{Y(1.0)}" class="ref sol"/>'
            f'<text x="{L + pw + 6}" y="{Y(1.0) + 4}" class="ax">SOL 1.0</text>')
    body, labels = [], []
    for s in series:
        pts = s["points"]
        step = "".join(  # stepwise: hold previous y until next x
            (f"M {X(pts[0][0]):.1f} {Y(pts[0][1]):.1f} " if i == 0 else
             f"H {X(x):.1f} V {Y(y):.1f} ")
            for i, (x, y) in enumerate(pts))
        body.append(f'<path d="{step}" fill="none" stroke="{s["color"]}" '
                    f'stroke-width="2" stroke-linejoin="round"/>')
        lx, ly = pts[-1]
        labels.append(f'<circle cx="{X(lx):.1f}" cy="{Y(ly):.1f}" r="3.5" fill="{s["color"]}"/>'
                      f'<text x="{X(lx) + 7:.1f}" y="{Y(ly) - 6:.1f}" class="dl" '
                      f'fill="{s["color"]}">{_esc(s["label"])} {ly:.2f}</text>')
    if x_is_time:
        ticks = "".join(
            f'<text x="{L + pw * f}" y="{H - 10}" class="ax" text-anchor="middle">'
            f'{_hm(x0 + f * (x1 - x0))}</text>' for f in (0, 0.25, 0.5, 0.75, 1.0))
    else:
        import math
        n = int(x1)
        stepn = max(1, n // 8)
        first = math.ceil(x0)
        ticks = "".join(
            f'<text x="{X(v)}" y="{H - 10}" class="ax" text-anchor="middle">{v}</text>'
            for v in range(first, n + 1, stepn))
    xcap = f'<text x="{W - 4}" y="{H - 10}" class="ax" text-anchor="end">{_esc(x_label)} →</text>'
    return (f'<svg viewBox="0 0 {W} {H}" role="img">'
            f'{grid}{refs}{"".join(body)}{"".join(labels)}{ticks}{xcap}</svg>')


def _timeline_svg(fleet: dict, order: dict[int, int]) -> str:
    """GPU occupancy. With rental windows: one lane per window, un-rented
    gaps compressed; without: single observed-span lane."""
    jobs = fleet["jobs"]
    if not jobs:
        return '<p class="muted">no GPU jobs yet</p>'
    windows = fleet["windows"] or [
        {"start": fleet["span_start"], "end": fleet["span_end"], "label": "observed"}]
    W, L, R, T, BH, SEP = 960, 46, 20, 26, 34, 14
    pw = W - L - R
    total = sum(w["end"] - w["start"] for w in windows) or 1e-9
    # piecewise x-mapping: windows laid side by side with SEP gaps
    n_sep = max(0, len(windows) - 1)
    scale = (pw - n_sep * SEP) / total
    segs, lanes, ticks = [], [], []
    xoff = float(L)
    for wi, w in enumerate(windows):
        wpx = (w["end"] - w["start"]) * scale
        lanes.append(f'<rect x="{xoff:.1f}" y="{T}" width="{wpx:.1f}" height="{BH}" class="lane"/>')
        if wpx > 70:   # only label windows wide enough to avoid overlap
            lbl = (f'{_esc(w["label"] or "rental")} · {_hm(w["start"])}–{_hm(w["end"])}'
                   if wpx > 150 else _esc(w["label"] or "rental"))
            ticks.append(f'<text x="{xoff + 2:.1f}" y="{T - 8}" class="ax">{lbl}</text>')
        for j in jobs:
            if j["start"] >= w["start"] and j["start"] <= w["end"]:
                x0 = xoff + (j["start"] - w["start"]) * scale
                x1 = xoff + (min(j["done"], w["end"]) - w["start"]) * scale
                i = order.get(j["task"], 0)
                wait = (j["start"] - j["enq"]) if "enq" in j else None
                tip = (f"#{j['task']} • run {_fmt_s(j['done'] - j['start'])}"
                       + (f" • waited {_fmt_s(wait)}" if wait is not None else ""))
                segs.append(f'<rect x="{x0:.1f}" y="{T}" width="{max(x1 - x0, 1.2):.1f}" '
                            f'height="{BH}" rx="2" fill="{_slot(i)}" class="seg" '
                            f'data-tip="{_esc(tip)}"/>')
        xoff += wpx
        if wi < len(windows) - 1:
            segs.append(f'<text x="{xoff + SEP / 2:.1f}" y="{T + BH / 2 + 4}" '
                        f'class="ax" text-anchor="middle">⋯</text>')
            xoff += SEP
    H = T + BH + 26
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="gpu timeline">'
            f'{"".join(lanes)}{"".join(ticks)}{"".join(segs)}</svg>')


def _histogram_svg(counts: list[int]) -> str:
    if not any(counts):
        return '<p class="muted">no scored problems yet</p>'
    W, H, L, R, T, B = 960, 190, 46, 20, 12, 34
    pw, ph = W - L - R, H - T - B
    n = len(counts)
    maxc = max(counts)
    bw = pw / n
    bars = []
    for i, c in enumerate(counts):
        if not c:
            continue
        h = ph * c / maxc
        x = L + i * bw
        lo, hi = i / n, (i + 1) / n
        bars.append(f'<rect x="{x + 1:.1f}" y="{T + ph - h:.1f}" width="{bw - 2:.1f}" '
                    f'height="{h:.1f}" rx="3" fill="var(--seq)" class="seg" '
                    f'data-tip="score {lo:.2f}–{hi:.2f}: {c} problem(s)"/>'
                    f'<text x="{x + bw / 2:.1f}" y="{T + ph - h - 5:.1f}" class="dl2" '
                    f'text-anchor="middle">{c}</text>')
    ticks = "".join(
        f'<text x="{L + pw * v:.1f}" y="{H - 10}" class="ax" text-anchor="middle">{v:.1f}</text>'
        for v in (0, 0.25, 0.5, 0.75, 1.0))
    base = f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" class="grid"/>'
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="score histogram">'
            f'{base}{"".join(bars)}{ticks}</svg>')


def _outcomes_svg(problems: list[dict], order: dict[int, int]) -> str:
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
        for key in OUTCOME_KEYS:
            n = p["outcomes"][key]
            if not n:
                continue
            w = pw * n / maxn
            out.append(f'<rect x="{x:.1f}" y="{y + 2}" width="{max(w - 2, 1.2):.1f}" '
                       f'height="14" rx="3" fill="var(--o-{key})" class="seg" '
                       f'data-tip="#{p["task"]} {key}: {n}"/>')
            x += w
        out.append(
            f'<text x="{L - 8}" y="{y + 13}" class="ax" text-anchor="end">#{p["task"]} {_esc(p["name"][:16])}</text>'
            f'<text x="{x + 6:.1f}" y="{y + 13}" class="dl2">{total}</text>')
    legend = "".join(
        f'<span class="lg"><i style="background:var(--o-{k})"></i>{k}</span>'
        for k in OUTCOME_KEYS)
    return (f'<div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img">{"".join(out)}</svg>')


def _waits_svg(problems: list[dict], order: dict[int, int]) -> str:
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
        w50 = pw * (p["wait_p50"] / maxv)
        w95 = pw * ((p["wait_p95"] or p["wait_p50"]) / maxv)
        tip = f"#{p['task']} wait p50 {_fmt_s(p['wait_p50'])} • p95 {_fmt_s(p['wait_p95'])}"
        out.append(
            f'<text x="{L - 8}" y="{y + 13}" class="ax" text-anchor="end">#{p["task"]} {_esc(p["name"][:16])}</text>'
            f'<line x1="{L}" y1="{y + 9}" x2="{L + w95:.1f}" y2="{y + 9}" class="whisk"/>'
            f'<rect x="{L}" y="{y + 2}" width="{max(w50, 1.5):.1f}" height="14" rx="3" '
            f'fill="{_slot(order.get(p["task"], 0))}" class="seg" data-tip="{_esc(tip)}"/>'
            f'<text x="{L + w95 + 6:.1f}" y="{y + 13}" class="dl2">{_fmt_s(p["wait_p50"])}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img">{"".join(out)}</svg>'


# ------------------------------------------------------------------ page ----

_CSS = """
:root{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a887f;
--grid:#e8e7e2;--ref:#c9c7bf;--seq:#2a78d6;
--s1:#2a78d6;--s2:#1baf7a;--s3:#eda100;--s4:#008300;--s5:#4a3aa7;--s6:#e34948;--s7:#e87ba4;--s8:#eb6834;
--o-accepted:#0ca30c;--o-dominated:#9c9a92;--o-rejected:#eb6834;--o-duplicate:#eda100;--o-incorrect:#e87ba4;--o-error:#d03b3b;}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--panel:#222221;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8a887f;
--grid:#33322f;--ref:#4a4945;--seq:#3987e5;
--s1:#3987e5;--s2:#199e70;--s3:#c98500;--s4:#008300;--s5:#9085e9;--s6:#e66767;--s7:#d55181;--s8:#d95926;
--o-dominated:#7c7a72;--o-rejected:#d95926;--o-duplicate:#c98500;--o-incorrect:#d55181;}}
:root[data-theme=light]{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a887f;--grid:#e8e7e2;--ref:#c9c7bf;--seq:#2a78d6;
--s1:#2a78d6;--s2:#1baf7a;--s3:#eda100;--s4:#008300;--s5:#4a3aa7;--s6:#e34948;--s7:#e87ba4;--s8:#eb6834;
--o-dominated:#9c9a92;--o-rejected:#eb6834;--o-duplicate:#eda100;--o-incorrect:#e87ba4;}
:root[data-theme=dark]{--surface:#1a1a19;--panel:#222221;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8a887f;--grid:#33322f;--ref:#4a4945;--seq:#3987e5;
--s1:#3987e5;--s2:#199e70;--s3:#c98500;--s4:#008300;--s5:#9085e9;--s6:#e66767;--s7:#d55181;--s8:#d95926;
--o-dominated:#7c7a72;--o-rejected:#d95926;--o-duplicate:#c98500;--o-incorrect:#d55181;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:24px}
h1{font-size:19px;margin:0 0 2px}
h2{font-size:14px;font-weight:600;margin:0 0 10px;color:var(--ink)}
.sub{color:var(--ink3);font-size:12px;margin-bottom:20px}
.sub a{color:var(--ink2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.tile{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:14px 16px}
.tile-v{font-size:24px;font-weight:650;letter-spacing:-.02em}
.tile-l{color:var(--ink2);font-size:12px;margin-top:2px}
.tile-s{color:var(--ink3);font-size:11px;margin-top:2px}
.panel{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:16px 18px;margin-bottom:16px;overflow-x:auto}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:980px){.cols{grid-template-columns:1fr}}
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
.whisk{stroke:var(--ink3);stroke-width:2;opacity:.55;stroke-linecap:round}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;font-size:12px;color:var(--ink2)}
.lg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{color:var(--ink3);font-weight:600;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;cursor:pointer;user-select:none}
th:hover{color:var(--ink)}
th,td{padding:7px 12px 7px 0;border-bottom:1px solid var(--grid);white-space:nowrap}
td .bar{position:relative;height:5px;background:var(--grid);border-radius:3px;margin-top:4px;min-width:120px}
td .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
td .bar b{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--ref)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.done{color:var(--ink3)} .run{color:var(--o-accepted);font-weight:600}
.muted{color:var(--ink3)}
a{color:inherit}
input.flt{background:var(--surface);border:1px solid var(--grid);border-radius:7px;
color:var(--ink);padding:6px 10px;font-size:13px;margin-bottom:10px;width:260px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);
padding:5px 9px;border-radius:6px;font-size:12px;opacity:0;transition:opacity .08s;z-index:9;max-width:420px;white-space:normal}
.chip{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;font-weight:600;color:#fff}
td.strat{white-space:normal;max-width:380px;color:var(--ink2)}
pre{margin:0 0 12px;overflow-x:auto;background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:12px 14px}
pre code{font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink)}
button.link{background:none;border:none;color:var(--s1);cursor:pointer;font:inherit;padding:0}
button.link:hover{text-decoration:underline}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:20;display:flex;
align-items:flex-start;justify-content:center;padding:40px 16px;overflow:auto}
.modal-card{background:var(--panel);border:1px solid var(--grid);border-radius:12px;
width:min(920px,100%);max-height:90vh;display:flex;flex-direction:column;box-shadow:0 12px 40px rgba(0,0,0,.35)}
.modal-bar{display:flex;justify-content:space-between;align-items:center;gap:16px;
padding:14px 18px;border-bottom:1px solid var(--grid);position:sticky;top:0;background:var(--panel);border-radius:12px 12px 0 0}
#modal-body{padding:16px 18px;overflow:auto}
.modal-meta{display:flex;gap:14px;align-items:center;color:var(--ink2);font-size:12px;margin-bottom:12px;flex-wrap:wrap}
.src-h{font-size:12px;font-weight:600;color:var(--ink2);margin:4px 0}
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
// table sort + filter
document.querySelectorAll('table.sortable').forEach(tb=>{
  tb.querySelectorAll('th').forEach((th,ci)=>th.addEventListener('click',()=>{
    const rows=[...tb.tBodies[0].rows];
    const dir=th.dataset.dir=th.dataset.dir==='a'?'d':'a';
    rows.sort((r1,r2)=>{
      const a=r1.cells[ci].dataset.v??r1.cells[ci].textContent.trim();
      const b=r2.cells[ci].dataset.v??r2.cells[ci].textContent.trim();
      const na=parseFloat(a),nb=parseFloat(b);
      const c=(!isNaN(na)&&!isNaN(nb))?na-nb:a.localeCompare(b);
      return dir==='a'?c:-c;});
    rows.forEach(r=>tb.tBodies[0].appendChild(r));}));});
const flt=document.getElementById('flt');
if(flt)flt.addEventListener('input',()=>{
  const v=flt.value.toLowerCase();
  document.querySelectorAll('table.sortable tbody tr').forEach(r=>
    r.style.display=r.textContent.toLowerCase().includes(v)?'':'none');});
// candidate code modal (inline templates -> no fetch; works under file://)
const modal=document.getElementById('modal');
function openCode(id){
  const tpl=document.querySelector('template[data-code="'+CSS.escape(id)+'"]');
  if(!tpl||!modal)return;
  document.getElementById('modal-title').textContent=tpl.dataset.title||id;
  document.getElementById('modal-body').replaceChildren(tpl.content.cloneNode(true));
  modal.hidden=false;history.replaceState(null,'','?code='+encodeURIComponent(id));}
function closeCode(){if(modal){modal.hidden=true;history.replaceState(null,'',location.pathname);}}
if(modal){
  document.addEventListener('click',e=>{
    const b=e.target.closest('[data-code]');
    if(b&&b.tagName==='BUTTON'){openCode(b.dataset.code);}
    else if(e.target===modal||e.target.id==='modal-x'){closeCode();}});
  document.addEventListener('keydown',e=>{if(e.key==='Escape')closeCode();});
  const pre=new URLSearchParams(location.search).get('code');
  if(pre)openCode(pre);
}
"""


def _shell(title: str, sub: str, body: str, refresh: int | None) -> str:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{meta}
<title>{_esc(title)}</title><style>{_CSS}</style></head><body>
<h1>{_esc(title)}</h1><div class="sub">{sub}</div>{body}
<script>{_JS}</script></body></html>"""


def _problem_table(problems: list[dict], order: dict[int, int], detail_dir: str) -> str:
    rows = []
    for p in problems:
        i = order.get(p["task"], 0)
        best = p["best"]
        bar = ""
        if best is not None:
            pct = max(0.0, min(best, 1.0)) * 100
            bar = (f'<div class="bar"><i style="width:{pct:.1f}%;'
                   f'background:{_slot(i)}"></i><b></b></div>')
        status = p["terminated"] or "running"
        rows.append(
            "<tr>"
            f'<td data-v="{p["task"]}"><span class="dot" style="background:{_slot(i)}"></span>#{p["task"]}</td>'
            f'<td><a href="{detail_dir}/{p["task"]}.html">{_esc(p["name"])}</a></td>'
            f"<td>{_esc(p['family'])}</td><td>{_esc(p['model'])}</td>"
            f'<td class="{ "done" if p["terminated"] else "run"}">{_esc(status)}</td>'
            f"<td>{p['iters']}</td><td>{p['evals']}</td><td>{p['frontier']}</td>"
            f'<td data-v="{best if best is not None else -1}">'
            f"{'' if best is None else f'{best:.3f}'}{bar}</td>"
            f'<td data-v="{p["wait_p50"] or -1}">{_fmt_s(p["wait_p50"])}</td>'
            "</tr>")
    return ('<input id="flt" class="flt" placeholder="filter problems…">'
            '<table class="sortable"><thead><tr><th>task</th><th>name</th>'
            "<th>family</th><th>agent</th><th>status</th><th>iters</th>"
            "<th>evals</th><th>frontier</th><th>best SOL score</th>"
            "<th>wait p50</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def _family_table(families: list[dict]) -> str:
    rows = []
    for f in families:
        mb = "–" if f["mean_best"] is None else f"{f['mean_best']:.3f}"
        rows.append(
            "<tr>"
            f"<td>{_esc(f['family'])}</td><td>{f['n']}</td><td>{f['done']}</td>"
            f'<td data-v="{f["mean_best"] if f["mean_best"] is not None else -1}">{mb}</td>'
            f"<td>{f['evals']}</td></tr>")
    return ('<table class="sortable"><thead><tr><th>family</th><th>problems</th>'
            "<th>done</th><th>mean best</th><th>evals</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def build_hub(data: dict, *, refresh: int | None, detail_dir: str = "problems_html") -> str:
    problems, fleet = data["problems"], data["fleet"]
    order = {p["task"]: i for i, p in enumerate(problems)}
    movers = data["movers"]
    gen = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tiles = "".join([
        _tile("fleet SOL score (mean of best)",
              "–" if fleet["mean_best"] is None else f"{fleet['mean_best']:.3f}",
              "0.5 = baseline · 1.0 = SOL"),
        _tile(f"GPU utilization (of {fleet['util_basis']})",
              f"{fleet['gpu_util'] * 100:.0f}%",
              f"busy {_fmt_s(fleet['busy_s'])}"
              + (f" of rented {_fmt_s(fleet['rented_s'])}" if fleet["rented_s"]
                 else f" of {_fmt_s(fleet['span_s'])}")),
        _tile("queue wait p50 / p95",
              f"{_fmt_s(fleet['wait_p50'])} / {_fmt_s(fleet['wait_p95'])}"),
        _tile("GPU evals", str(fleet["total_evals"])),
        _tile("problems", f"{fleet['active']} active · {fleet['done']} done"),
        _tile("agent calls / tokens",
              f"{fleet['agent_calls']} / {fleet['agent_tokens']:,}"),
    ])

    fleet_line = _line_chart(
        [{"label": "fleet mean", "color": "var(--seq)",
          "points": data["fleet_series"]}],
        x_label="wall time", x_is_time=True, height=240) \
        if data["fleet_series"] else '<p class="muted">no accepted results yet</p>'

    mv_series = [{"label": f"#{p['task']}", "color": _slot(order[p["task"]]),
                  "points": p["convergence"]} for p in movers]
    mv_legend = "".join(
        f'<span class="lg"><i style="background:{_slot(order[p["task"]])}"></i>'
        f"#{p['task']} {_esc(p['name'][:22])}</span>" for p in movers)
    more = len([p for p in problems if p["convergence"]]) - len(movers)
    mv_note = (f'<div class="sub">top {len(movers)} most-recently-improving; '
               f'{more} more in the table below</div>' if more > 0 else "")

    body = f"""
<div class="tiles">{tiles}</div>
<div class="panel"><h2>Fleet SOL score over time (mean of per-problem best, ↑)</h2>{fleet_line}</div>
<div class="panel"><h2>Convergence — top movers (best SOL score vs GPU evals)</h2>
{mv_note}<div class="legend">{mv_legend}</div>
{_line_chart(mv_series, x_label="GPU evals") if mv_series else '<p class="muted">no data yet</p>'}</div>
<div class="panel"><h2>GPU occupancy — rented windows, one job at a time</h2>
{_timeline_svg(fleet, order)}</div>
<div class="cols">
<div class="panel"><h2>Best-score distribution (all problems)</h2>{_histogram_svg(data["histogram"])}</div>
<div class="panel"><h2>Families</h2>{_family_table(data["families"])}</div>
</div>
<div class="panel"><h2>Queue wait — slowest 10 (bar = p50, whisker = p95)</h2>
{_waits_svg(sorted([p for p in problems if p["wait_p50"]], key=lambda p: -(p["wait_p95"] or 0))[:10], order)}</div>
<div class="panel"><h2>Iteration outcomes — 10 most-iterated</h2>
{_outcomes_svg(sorted(problems, key=lambda p: -sum(p["outcomes"].values()))[:10], order)}</div>
<div class="panel"><h2>All problems</h2>{_problem_table(problems, order, detail_dir)}</div>
"""
    sub = (f"generated {gen}"
           + (f" · auto-refresh {refresh}s" if refresh else "")
           + " · click a problem name for its detail page")
    return _shell("SOL-ExecBench solver — run dashboard", sub, body, refresh)


_STATUS_CHIP = {
    "accepted": "o-accepted", "dominated": "o-dominated", "rejected": "o-rejected",
    "duplicate": "o-duplicate", "incorrect": "o-incorrect", "error": "o-error",
    "planned": "o-dominated",
}


def _code_blocks(p: dict) -> str:
    """Hidden per-candidate code, revealed by the modal (inline = static-safe,
    works under file://). One <template> per candidate, keyed by cand id."""
    out = []
    for c in p["candidates"]:
        sol = c.get("solution")
        if not sol:
            continue
        spec = sol.get("spec", {})
        score = "–" if c.get("sol_score") is None else f"{c['sol_score']:.3f}"
        chip = _STATUS_CHIP.get(c["status"], "o-dominated")
        meta = (f'<div class="modal-meta">'
                f'<span class="chip" style="background:var(--{chip})">{_esc(c["status"])}</span>'
                f'<span>score <b>{score}</b></span>'
                f'<span>{_esc(", ".join(spec.get("languages", ["?"])))}</span>'
                f'<span>{_esc(c.get("model") or "")}</span></div>')
        srcs = "".join(
            f'<div class="src-h">{_esc(s.get("path", "?"))}</div>'
            f'<pre><code>{_esc(s.get("content", ""))}</code></pre>'
            for s in sol.get("sources", []))
        out.append(
            f'<template data-code="{_esc(c["cand"])}" '
            f'data-title="{_esc(c["cand"])} — {_esc(c.get("strategy") or "")}">'
            f'{meta}{srcs}</template>')
    return "".join(out)


def _progression_table(p: dict) -> str:
    """Solution progression over time: strategy TL;DR + status + score + code link."""
    if not p["candidates"]:
        return '<p class="muted">no candidates yet</p>'
    rows = []
    for c in p["candidates"]:
        t = (c["ts"] or "")[11:19]
        chip = (f'<span class="chip" style="background:var(--{_STATUS_CHIP.get(c["status"], "o-dominated")})">'
                f'{_esc(c["status"])}</span>')
        score = "–" if c.get("sol_score") is None else f"{c['sol_score']:.3f}"
        best = "" if c.get("best_after") is None else f"{c['best_after']:.3f}"
        code = (f'<button class="link" data-code="{_esc(c["cand"])}">view code</button>'
                if c.get("solution") else '<span class="muted">–</span>')
        rows.append(
            "<tr>"
            f'<td>{t}</td><td>{_esc(c["cand"])}</td><td>{_esc(c.get("model") or "")}</td>'
            f'<td class="strat">{_esc(c.get("strategy") or "")}</td>'
            f'<td>{chip}</td><td data-v="{c.get("sol_score") or -1}">{score}</td>'
            f'<td data-v="{c.get("best_after") or -1}">{best}</td><td>{code}</td>'
            "</tr>")
    return ('<table class="sortable"><thead><tr><th>time</th><th>cand</th>'
            "<th>agent</th><th>strategy</th><th>status</th><th>score</th>"
            "<th>best after</th><th>code</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def build_detail(p: dict, slot_i: int) -> str:
    conv = _line_chart(
        [{"label": f"#{p['task']}", "color": _slot(slot_i), "points": p["convergence"]}],
        x_label="GPU evals", right=120) if p["convergence"] else '<p class="muted">no evals yet</p>'
    order = {p["task"]: slot_i}
    stats = "".join([
        _tile("best SOL score", "–" if p["best"] is None else f"{p['best']:.3f}"),
        _tile("status", p["terminated"] or "running"),
        _tile("iterations / evals", f"{p['iters']} / {p['evals']}"),
        _tile("frontier size", str(p["frontier"])),
        _tile("wait p50 / p95", f"{_fmt_s(p['wait_p50'])} / {_fmt_s(p['wait_p95'])}"),
        _tile("agent", p["model"] or "–"),
    ])
    modal = ('<div id="modal" class="modal" hidden><div class="modal-card">'
             '<div class="modal-bar"><b id="modal-title"></b>'
             '<button id="modal-x" class="link" aria-label="close">✕ close</button></div>'
             '<div id="modal-body"></div></div></div>')
    body = f"""
<div class="tiles">{stats}</div>
<div class="panel"><h2>Convergence</h2>{conv}</div>
<div class="panel"><h2>Solution progression — every candidate, in order</h2>
{_progression_table(p)}</div>
<div class="panel"><h2>Iteration outcomes</h2>{_outcomes_svg([p], order)}</div>
{_code_blocks(p)}{modal}
"""
    sub = f'{_esc(p["family"])} · <a href="../index.html">← back to fleet dashboard</a>'
    return _shell(f"#{p['task']} {p['name']}", sub, body, None)


def render(runs_dir: Path, out_dir: Path, *, refresh: int | None = None) -> Path:
    """Render the publishable static site into out_dir (as-is):
    out_dir/index.html + out_dir/p/<task>.html (code shown in-page via modal)."""
    runs_dir = Path(runs_dir)
    journals = journal_mod.read_all(runs_dir)
    data = metrics_mod.collect(journals, runs_dir=runs_dir)
    out_dir = Path(out_dir)
    p_dir = out_dir / "p"
    p_dir.mkdir(parents=True, exist_ok=True)

    order = {p["task"]: i for i, p in enumerate(data["problems"])}
    for p in data["problems"]:
        (p_dir / f"{p['task']}.html").write_text(build_detail(p, order[p["task"]]))

    index = out_dir / "index.html"
    tmp = index.with_suffix(".tmp")
    tmp.write_text(build_hub(data, refresh=refresh, detail_dir="p"))
    tmp.replace(index)
    return index
