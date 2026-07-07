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
import json
import os
from pathlib import Path

from .. import journal as journal_mod
from . import metrics as metrics_mod

SERIES_N = 8
OUTCOME_KEYS = metrics_mod.OUTCOMES


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _localtime(ts: str | None) -> str:
    """UTC journal timestamp → local HH:MM:SS (the dashboard is viewed where it's
    rendered, so this is the user's timezone)."""
    if not ts:
        return ""
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return ts[11:19]


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
--o-accepted:#0ca30c;--o-dominated:#9c9a92;--o-rejected:#eb6834;--o-duplicate:#eda100;--o-incorrect:#e87ba4;--o-flaky:#a855c7;--o-error:#d03b3b;}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--panel:#222221;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8a887f;
--grid:#33322f;--ref:#4a4945;--seq:#3987e5;
--s1:#3987e5;--s2:#199e70;--s3:#c98500;--s4:#008300;--s5:#9085e9;--s6:#e66767;--s7:#d55181;--s8:#d95926;
--o-dominated:#7c7a72;--o-rejected:#d95926;--o-duplicate:#c98500;--o-incorrect:#d55181;--o-flaky:#c07de0;}}
:root[data-theme=light]{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a887f;--grid:#e8e7e2;--ref:#c9c7bf;--seq:#2a78d6;
--s1:#2a78d6;--s2:#1baf7a;--s3:#eda100;--s4:#008300;--s5:#4a3aa7;--s6:#e34948;--s7:#e87ba4;--s8:#eb6834;
--o-dominated:#9c9a92;--o-rejected:#eb6834;--o-duplicate:#eda100;--o-incorrect:#e87ba4;--o-flaky:#a855c7;}
:root[data-theme=dark]{--surface:#1a1a19;--panel:#222221;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8a887f;--grid:#33322f;--ref:#4a4945;--seq:#3987e5;
--s1:#3987e5;--s2:#199e70;--s3:#c98500;--s4:#008300;--s5:#9085e9;--s6:#e66767;--s7:#d55181;--s8:#d95926;
--o-dominated:#7c7a72;--o-rejected:#d95926;--o-duplicate:#c98500;--o-incorrect:#d55181;--o-flaky:#c07de0;}
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
td.real b{color:var(--o-accepted)}  /* the real, submitted leaderboard SOL (ground truth) */
.up{color:var(--o-accepted);font-weight:700;cursor:help}  /* ↑ = worth (re)submitting: beats our last submission, or never submitted */
td.proj{color:var(--ink2);font-style:italic}  /* projected board SOL + projected rank (a data-grounded estimate) */
pre{margin:0 0 12px;overflow-x:auto;background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:12px 14px}
pre.traj{font-size:11px;line-height:1.5;white-space:pre-wrap;word-break:break-word;max-height:70vh;overflow:auto}
pre code{font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink)}
pre.coach-card{white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.55;
  background:var(--bg);border:1px solid var(--grid);border-left:3px solid var(--s1);
  border-radius:8px;padding:12px 14px;max-height:44vh;overflow:auto;margin:0}
h3.coach-h{font-size:13px;color:var(--ink2);margin:16px 0 8px}
td.views{white-space:nowrap}
button.link{background:none;border:none;color:var(--s1);cursor:pointer;font:inherit;padding:0}
button.link:hover{text-decoration:underline}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:20;display:flex;
align-items:flex-start;justify-content:center;padding:40px 16px;overflow:auto}
.modal[hidden]{display:none}
.filterbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
.chip-btn{border:1px solid var(--grid);background:var(--panel);color:var(--ink2);
border-radius:16px;padding:4px 11px;font-size:12px;cursor:pointer}
.chip-btn.on{color:#fff;border-color:transparent}
.fcount{color:var(--ink3);font-size:12px;margin-left:4px}
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


_HUB_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const RECS = D.recs, OC = D.outcomes, DETAIL = D.detail;
const OCC = {accepted:'--o-accepted',dominated:'--o-dominated',incorrect:'--o-incorrect',
             rejected:'--o-rejected',duplicate:'--o-duplicate',flaky:'--o-flaky',error:'--o-error'};
const sel = new Set();          // selected families
let query = '';
const SL = i => 'var(--s'+(i%8+1)+')';
const esc = s => String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fs = s => s==null?'–':(s<90?s.toFixed(1)+'s':(s<5400?(s/60).toFixed(1)+'m':(s/3600).toFixed(1)+'h'));
const pct=(a,p)=>{if(!a.length)return null;const v=[...a].sort((x,y)=>x-y);return v[Math.min(v.length-1,Math.max(0,Math.round(p*(v.length-1))))]};
const bigN=n=>{n=n||0;const a=Math.abs(n);return a>=1e9?(n/1e9).toFixed(1)+'B':a>=1e6?(n/1e6).toFixed(1)+'M':a>=1e3?(n/1e3).toFixed(1)+'K':String(n);};
const idx = {}; RECS.forEach((r,i)=>idx[r.t]=i);   // stable color slot by task order

function matches(r){
  if(sel.size && !sel.has(r.f)) return false;
  if(!query) return true;
  const hay = ('#'+r.t+' '+r.n+' '+r.f+' '+r.a).toLowerCase();
  return query.split(',').map(s=>s.trim()).filter(Boolean).some(tok=>hay.includes(tok));
}

// ---- SVG builders (mirror the server's; identity color = task's stable slot) ----
function lineChart(series,{xTime=false,xLabel='',right=150,height=300}={}){
  const W=960,H=height,L=46,R=right,T=14,B=30,pw=W-L-R,ph=H-T-B;
  const xs=series.flatMap(s=>s.pts.map(p=>p[0])); if(!xs.length) return '<p class="muted">no data</p>';
  let x0=Math.min(...xs),x1=Math.max(...xs); if(x1<=x0)x1=x0+1;
  const X=x=>L+pw*(x-x0)/(x1-x0), Y=y=>T+ph*(1-y);
  let g=''; [0,.25,.5,.75,1].forEach(v=>g+=`<line x1="${L}" y1="${Y(v)}" x2="${L+pw}" y2="${Y(v)}" class="grid"/><text x="${L-8}" y="${Y(v)+4}" class="ax" text-anchor="end">${v.toFixed(2)}</text>`);
  g+=`<line x1="${L}" y1="${Y(.5)}" x2="${L+pw}" y2="${Y(.5)}" class="ref"/><text x="${L+pw+6}" y="${Y(.5)+4}" class="ax">baseline 0.5</text>`;
  g+=`<line x1="${L}" y1="${Y(1)}" x2="${L+pw}" y2="${Y(1)}" class="ref sol"/><text x="${L+pw+6}" y="${Y(1)+4}" class="ax">SOL 1.0</text>`;
  let body='',lab='';
  series.forEach(s=>{
    const p=s.pts; let d='';
    p.forEach((pt,i)=>{d+= i===0?`M ${X(pt[0]).toFixed(1)} ${Y(pt[1]).toFixed(1)} `:`H ${X(pt[0]).toFixed(1)} V ${Y(pt[1]).toFixed(1)} `;});
    body+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
    const [lx,ly]=p[p.length-1];
    lab+=`<circle cx="${X(lx).toFixed(1)}" cy="${Y(ly).toFixed(1)}" r="3.5" fill="${s.color}"/><text x="${(X(lx)+7).toFixed(1)}" y="${(Y(ly)-6).toFixed(1)}" class="dl" fill="${s.color}">${esc(s.label)} ${ly.toFixed(2)}</text>`;
  });
  let tk='';
  if(xTime){[0,.25,.5,.75,1].forEach(f=>{const t=new Date((x0+f*(x1-x0))*1000);tk+=`<text x="${L+pw*f}" y="${H-10}" class="ax" text-anchor="middle">${String(t.getHours()).padStart(2,'0')}:${String(t.getMinutes()).padStart(2,'0')}</text>`;});}
  else {const n=Math.floor(x1),st=Math.max(1,Math.floor(n/8)); for(let v=Math.ceil(x0);v<=n;v+=st)tk+=`<text x="${X(v)}" y="${H-10}" class="ax" text-anchor="middle">${v}</text>`;}
  return `<svg viewBox="0 0 ${W} ${H}">${g}${body}${lab}${tk}<text x="${W-4}" y="${H-10}" class="ax" text-anchor="end">${esc(xLabel)} →</text></svg>`;
}
function histChart(sub){
  const bins=20,c=Array(bins).fill(0);
  sub.forEach(r=>{if(r.bc!=null)c[Math.min(bins-1,Math.floor(Math.max(0,Math.min(r.bc,.9999))*bins))]++;});
  if(!c.some(x=>x))return '<p class="muted">no scored problems</p>';
  const W=960,H=190,L=46,R=20,T=12,B=34,pw=W-L-R,ph=H-T-B,mx=Math.max(...c),bw=pw/bins;
  let bars=''; c.forEach((v,i)=>{if(!v)return;const h=ph*v/mx,x=L+i*bw;bars+=`<rect x="${(x+1).toFixed(1)}" y="${(T+ph-h).toFixed(1)}" width="${(bw-2).toFixed(1)}" height="${h.toFixed(1)}" rx="3" fill="var(--seq)" class="seg" data-tip="score ${(i/bins).toFixed(2)}–${((i+1)/bins).toFixed(2)}: ${v}"/><text x="${(x+bw/2).toFixed(1)}" y="${(T+ph-h-5).toFixed(1)}" class="dl2" text-anchor="middle">${v}</text>`;});
  let tk=''; [0,.25,.5,.75,1].forEach(v=>tk+=`<text x="${(L+pw*v).toFixed(1)}" y="${H-10}" class="ax" text-anchor="middle">${v.toFixed(1)}</text>`);
  return `<svg viewBox="0 0 ${W} ${H}"><line x1="${L}" y1="${T+ph}" x2="${L+pw}" y2="${T+ph}" class="grid"/>${bars}${tk}</svg>`;
}
function waitsChart(sub){
  const rows=sub.filter(r=>r.w5!=null).sort((a,b)=>(b.w9||b.w5)-(a.w9||a.w5)).slice(0,10);
  if(!rows.length)return '<p class="muted">no queue data</p>';
  const W=960,L=170,R=60,RH=26,H=16+RH*rows.length+8,mx=Math.max(...rows.map(r=>r.w9||r.w5))||1,pw=W-L-R;
  let o=''; rows.forEach((r,i)=>{const y=12+i*RH,w50=pw*(r.w5/mx),w95=pw*((r.w9||r.w5)/mx);
    o+=`<text x="${L-8}" y="${y+13}" class="ax" text-anchor="end">#${r.t} ${esc(r.n.slice(0,16))}</text><line x1="${L}" y1="${y+9}" x2="${(L+w95).toFixed(1)}" y2="${y+9}" class="whisk"/><rect x="${L}" y="${y+2}" width="${Math.max(w50,1.5).toFixed(1)}" height="14" rx="3" fill="${SL(idx[r.t])}" class="seg" data-tip="#${r.t} p50 ${fs(r.w5)} • p95 ${fs(r.w9)}"/><text x="${(L+w95+6).toFixed(1)}" y="${y+13}" class="dl2">${fs(r.w5)}</text>`;});
  return `<svg viewBox="0 0 ${W} ${H}">${o}</svg>`;
}
function ocChart(sub){
  const rows=sub.filter(r=>r.o.reduce((a,b)=>a+b,0)).sort((a,b)=>b.o.reduce((x,y)=>x+y)-a.o.reduce((x,y)=>x+y)).slice(0,10);
  if(!rows.length)return '<p class="muted">no iterations</p>';
  const W=960,L=170,R=60,RH=26,H=16+RH*rows.length+8,mx=Math.max(...rows.map(r=>r.o.reduce((a,b)=>a+b,0))),pw=W-L-R;
  let o=''; rows.forEach((r,i)=>{const y=12+i*RH; let x=L,tot=r.o.reduce((a,b)=>a+b,0);
    r.o.forEach((n,k)=>{if(!n)return;const w=pw*n/mx;o+=`<rect x="${x.toFixed(1)}" y="${y+2}" width="${Math.max(w-2,1.2).toFixed(1)}" height="14" rx="3" fill="var(${OCC[OC[k]]})" class="seg" data-tip="#${r.t} ${OC[k]}: ${n}"/>`;x+=w;});
    o+=`<text x="${L-8}" y="${y+13}" class="ax" text-anchor="end">#${r.t} ${esc(r.n.slice(0,16))}</text><text x="${(x+6).toFixed(1)}" y="${y+13}" class="dl2">${tot}</text>`;});
  return `<svg viewBox="0 0 ${W} ${H}">${o}</svg>`;
}
function fleetSeries(sub){
  const ev=[]; sub.forEach(r=>r.ac.forEach(([ts,b])=>ev.push([ts,r.t,b])));
  ev.sort((a,b)=>a[0]-b[0]); const cur={},out=[];
  ev.forEach(([ts,t,b])=>{cur[t]=b;let s=0,n=0;for(const k in cur){s+=cur[k];n++;}out.push([ts,s/n]);});
  return out;
}
function famRollup(sub){
  const m={}; sub.forEach(r=>{(m[r.f]=m[r.f]||[]).push(r);});
  return Object.entries(m).map(([f,rs])=>{const bs=rs.filter(r=>r.bc!=null).map(r=>r.bc);
    return {f,n:rs.length,done:rs.filter(r=>r.s!=='running').length,
      mb:bs.length?bs.reduce((a,b)=>a+b)/bs.length:null,ev:rs.reduce((a,r)=>a+r.e,0)};})
    .sort((a,b)=>(b.mb||0)-(a.mb||0));
}
function tile(l,v,s){return `<div class="tile"><div class="tile-v">${v}</div><div class="tile-l">${esc(l)}</div>${s?`<div class="tile-s">${esc(s)}</div>`:''}</div>`;}

function render(){
  const sub=RECS.filter(matches);
  document.getElementById('fcount').textContent=`${sub.length} / ${RECS.length} problems`;
  const bcs=sub.filter(r=>r.bc!=null).map(r=>r.bc);
  const mbc=bcs.length?(bcs.reduce((a,b)=>a+b)/bcs.length):null;
  const w5=pct(sub.filter(r=>r.w5!=null).map(r=>r.w5),.5), w9=pct(sub.filter(r=>r.w9!=null).map(r=>r.w9),.95);
  const scoped = sub.length!==RECS.length;
  document.getElementById('tiles').innerHTML=[
    tile('expected SOL (mean of best)', mbc==null?'–':mbc.toFixed(3),'leaderboard estimate · 0.5 baseline · 1.0 SOL'),
    tile('GPU utilization (of '+D.util_basis+')',(D.gpu_util*100).toFixed(0)+'%','busy '+fs(D.busy_s)+(D.rented_s?' of rented '+fs(D.rented_s):'')+'  (fleet)'),
    tile('queue wait p50 / p95', fs(w5)+' / '+fs(w9)+(scoped?'  (selected)':'')),
    tile('GPU evals'+(scoped?' (selected)':''), String(sub.reduce((a,r)=>a+r.e,0))),
    tile('problems', sub.filter(r=>r.s==='running').length+' active · '+sub.filter(r=>r.s!=='running').length+' done'),
    tile('agent calls / tokens'+(scoped?' (sel)':''), sub.reduce((a,r)=>a+r.an,0)+' / '+bigN(sub.reduce((a,r)=>a+r.ak,0))),
  ].join('');
  // fleet-score-over-time (scoped)
  const fsr=fleetSeries(sub);
  document.getElementById('c-fleet').innerHTML=fsr.length?lineChart([{label:'mean',color:'var(--seq)',pts:fsr}],{xTime:true,height:240}):'<p class="muted">no accepted results</p>';
  // convergence — top movers in the selection
  const mv=sub.filter(r=>r.c.length).sort((a,b)=>(b.li-a.li)||(b.e-a.e)).slice(0,8);
  document.getElementById('c-conv').innerHTML=mv.length?lineChart(mv.map(r=>({label:'#'+r.t,color:SL(idx[r.t]),pts:r.c})),{xLabel:'GPU evals'}):'<p class="muted">no data</p>';
  document.getElementById('lg-conv').innerHTML=mv.map(r=>`<span class="lg"><i style="background:${SL(idx[r.t])}"></i>#${r.t} ${esc(r.n.slice(0,22))}</span>`).join('');
  const more=sub.filter(r=>r.c.length).length-mv.length;
  document.getElementById('note-conv').textContent=more>0?`top ${mv.length} most-recently-improving of the selection; ${more} more in the table`:'';
  // histogram / families / waits / outcomes
  document.getElementById('c-hist').innerHTML=histChart(sub);
  document.getElementById('t-fam').innerHTML=(()=>{const r=famRollup(sub);
    return '<table class="sortable"><thead><tr><th>family</th><th>problems</th><th>done</th><th>mean best</th><th>evals</th></tr></thead><tbody>'
    +r.map(f=>`<tr><td>${esc(f.f)}</td><td>${f.n}</td><td>${f.done}</td><td data-v="${f.mb??-1}">${f.mb==null?'–':f.mb.toFixed(3)}</td><td>${f.ev}</td></tr>`).join('')+'</tbody></table>';})();
  document.getElementById('c-waits').innerHTML=waitsChart(sub);
  document.getElementById('c-oc').innerHTML=ocChart(sub);
  // problems table
  document.getElementById('t-prob').innerHTML=(()=>{
    const rows=[...sub].sort((a,b)=>((b.bc??-1)-(a.bc??-1))).map(r=>{const i=idx[r.t];const bar=r.bc==null?'':`<div class="bar"><i style="width:${(Math.max(0,Math.min(r.bc,1))*100).toFixed(1)}%;background:${SL(i)}"></i><b></b></div>`;
      const lb=r.lb||{};const rank=lb.rank?`#${lb.rank} of ${lb.n??'?'}`:'';
      const b1=r.b1,b1s=(b1==null?'':b1.toFixed(3));const subE=lb.submitted_expected;
      const neverSub=(subE==null&&lb.sol==null);
      const prk=r.prk;const projRank=prk?`<span class="muted"> → #${prk.rank} of ${prk.n}</span>`:'';
      const resub=(subE!=null&&r.bc!=null&&lb.sol!=null&&r.bc>subE+0.01)
        ?` <span class="up" title="current best expected (${r.bc.toFixed(3)}) beats what we submitted (~${subE.toFixed(3)}) — re-submit">↑</span>`
        :(neverSub&&r.bc!=null)
          ?` <span class="up" title="never submitted — projected board SOL ${r.pr==null?'?':'~'+r.pr.toFixed(3)}${prk?', proj. rank #'+prk.rank+' of '+prk.n:''} — worth a first submit">↑</span>`
          :'';
      const subCell=(lb.sol==null&&subE==null)?'':`${subE==null?'–':subE.toFixed(3)}<span class="muted"> → </span><b class="real">${lb.sol==null?'–':lb.sol.toFixed(4)}</b>`;
      return `<tr><td data-v="${r.t}"><span class="dot" style="background:${SL(i)}"></span>#${r.t}</td><td><a href="${DETAIL}/${r.t}.html">${esc(r.n)}</a></td><td>${esc(r.f)}</td><td>${esc(r.a)}</td><td class="${r.s==='running'?'run':'done'}">${esc(r.s)}</td><td>${r.it}</td><td>${r.e}</td><td>${r.fr}</td><td data-v="${r.bc??-1}"><b>${r.bc==null?'':r.bc.toFixed(3)}</b>${resub}${bar}</td><td data-v="${prk?prk.rank:9999}" class="proj" title="projected board SOL = best expected × observed est→real ratio; projected rank = where that SOL would place on the live board">${r.pr==null?'':'~'+r.pr.toFixed(3)+projRank}</td><td data-v="${lb.sol??-1}">${subCell}</td><td data-v="${r.b1??-1}">${b1s}</td><td data-v="${lb.rank||9999}">${rank}</td><td data-v="${r.w5??-1}">${fs(r.w5)}</td></tr>`;}).join('');
    return '<table class="sortable"><thead><tr><th>task</th><th>name</th><th>family</th><th>agent</th><th>status</th><th>iters</th><th>evals</th><th>frontier</th><th>best expected SOL ▼</th><th>proj. board</th><th>submitted (est→real)</th><th>#1 SOL</th><th>leaderboard</th><th>wait p50</th></tr></thead><tbody>'+rows+'</tbody></table>';})();
  bindSort();
}
function bindSort(){
  document.querySelectorAll('table.sortable').forEach(tb=>{
    tb.querySelectorAll('th').forEach((th,ci)=>{if(th._b)return;th._b=1;th.addEventListener('click',()=>{
      const rows=[...tb.tBodies[0].rows],dir=th.dataset.dir=th.dataset.dir==='a'?'d':'a';
      rows.sort((r1,r2)=>{const a=r1.cells[ci].dataset.v??r1.cells[ci].textContent.trim(),b=r2.cells[ci].dataset.v??r2.cells[ci].textContent.trim();const na=parseFloat(a),nb=parseFloat(b);const c=(!isNaN(na)&&!isNaN(nb))?na-nb:String(a).localeCompare(b);return dir==='a'?c:-c;});
      rows.forEach(r=>tb.tBodies[0].appendChild(r));});});});
}
document.getElementById('q').addEventListener('input',e=>{query=e.target.value.toLowerCase();render();});
document.getElementById('clearf').addEventListener('click',()=>{sel.clear();query='';document.getElementById('q').value='';document.querySelectorAll('.chip-btn[data-fam]').forEach(b=>b.classList.remove('on'));render();});
function toggleFam(f,on){
  const b=document.querySelector('.chip-btn[data-fam="'+CSS.escape(f)+'"]'); if(!b)return;
  if(on){sel.add(f);b.classList.add('on');b.style.background=b.style.getPropertyValue('--c');}
  else{sel.delete(f);b.classList.remove('on');b.style.background='';}
}
document.querySelectorAll('.chip-btn[data-fam]').forEach(b=>b.addEventListener('click',()=>{
  toggleFam(b.dataset.fam,!sel.has(b.dataset.fam));render();}));
// deep-link / screenshot support: ?fam=a,b  ?q=text
const up=new URLSearchParams(location.search);
if(up.get('fam'))up.get('fam').split(',').map(s=>s.trim()).filter(Boolean).forEach(f=>toggleFam(f,true));
if(up.get('q')){query=up.get('q').toLowerCase();document.getElementById('q').value=up.get('q');}
render();
"""


def _shell(title: str, sub: str, body: str, refresh: int | None,
           extra_js: str = "") -> str:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{meta}
<title>{_esc(title)}</title><style>{_CSS}</style></head><body>
<h1>{_esc(title)}</h1><div class="sub">{sub}</div>{body}
<script>{_JS}</script>{f'<script>{extra_js}</script>' if extra_js else ''}</body></html>"""


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


def build_hub(data: dict, *, refresh: int | None, detail_dir: str = "p") -> str:
    problems, fleet = data["problems"], data["fleet"]
    order = {p["task"]: i for i, p in enumerate(problems)}
    gen = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Projected board SOL: our "expected SOL" vs the real board result differs a bit
    # per problem (calibration isn't uniform). Where we've submitted we KNOW that
    # ratio (real / expected@submit); apply it to the current best to project what a
    # (re)submission would score. Unsubmitted problems use the mean ratio across all
    # completed submissions. Simple first-order estimate, but grounded in real data.
    _ratios = [((p.get("lb") or {})["sol"] / (p.get("lb") or {})["submitted_expected"])
               for p in problems
               if (p.get("lb") or {}).get("sol") and (p.get("lb") or {}).get("submitted_expected")]
    _gr = (sum(_ratios) / len(_ratios)) if _ratios else 1.0

    def _proj(p):
        lb = p.get("lb") or {}
        r = (lb["sol"] / lb["submitted_expected"]) if (lb.get("sol") and lb.get("submitted_expected")) else _gr
        return round(p["best_cal"] * r, 4) if p.get("best_cal") is not None else None

    def _proj_rank(p, proj):
        """1-based rank the projected board SOL would take (higher SOL = better).
        A never-submitted entry JOINS the board (field grows by one); a resubmission
        just moves our existing slot (field size unchanged). None if uncached."""
        board = p.get("board") or {}
        scores = board.get("scores")
        if proj is None or not scores:
            return None
        beat = sum(1 for s in scores if s > proj)          # entries strictly better than us
        submitted = bool((p.get("lb") or {}).get("sol"))
        n = (board.get("n") or len(scores)) + (0 if submitted else 1)
        return {"rank": beat + 1, "n": n}

    # compact per-problem records for the client-side filter/renderer
    recs = [{
        "t": p["task"], "n": p["name"], "f": p["family"] or "?", "a": p["model"],
        "bc": p.get("best_cal"), "s": p["terminated"] or "running", "e": p["evals"],
        "it": p["iters"], "fr": p["frontier"], "lb": p.get("lb"),
        "b1": (p.get("board") or {}).get("top_sol"), "pr": _proj(p),
        "prk": _proj_rank(p, _proj(p)),
        "w5": p["wait_p50"], "w9": p["wait_p95"], "li": p["last_improve_ts"] or 0,
        "c": [[x, round(y, 4)] for x, y in p["convergence"]],
        "ac": [[round(ts, 1), round(y, 4)] for ts, y in p["accept_times"]],
        "o": [p["outcomes"][k] for k in OUTCOME_KEYS],
        "an": sum(p["agent"][k]["n"] for k in p["agent"]),
        "ak": sum(p["agent"][k]["tok"] for k in p["agent"]),
    } for p in problems]
    fams = [f["family"] for f in data["families"]]
    fam_chips = "".join(
        f'<button class="chip-btn" data-fam="{_esc(f)}" '
        f'style="--c:{_slot(fams.index(f))}">{_esc(f)}</button>' for f in fams)

    import json as _json
    payload = _json.dumps({
        "recs": recs, "outcomes": list(OUTCOME_KEYS),
        "detail": detail_dir,
        "gpu_util": fleet["gpu_util"], "util_basis": fleet["util_basis"],
        "busy_s": fleet["busy_s"], "rented_s": fleet["rented_s"], "span_s": fleet["span_s"],
    }, separators=(",", ":")).replace("<", "\\u003c")

    filterbar = (
        '<div class="filterbar">'
        '<input id="q" class="flt" placeholder="filter: task id, name, family, agent — comma = OR (e.g. 67, 231, rmsnorm)">'
        f'{fam_chips}'
        '<button id="clearf" class="chip-btn">clear</button>'
        '<span id="fcount" class="fcount"></span></div>')

    body = f"""
{filterbar}
<div id="tiles" class="tiles"></div>
<div class="panel"><h2 id="h-tbl">Problems <span class="fcount">(sorted by expected SOL ▼)</span></h2><div id="t-prob"></div></div>
<div class="panel"><h2 id="h-fleet">Fleet expected SOL over time (mean of per-problem best, ↑)</h2>
<div id="c-fleet"></div></div>
<div class="panel"><h2 id="h-conv">Convergence — top movers (expected SOL vs GPU evals)</h2>
<div id="note-conv" class="sub"></div><div id="lg-conv" class="legend"></div><div id="c-conv"></div></div>
<div class="panel"><h2>GPU occupancy — rented windows, one job at a time <span class="fcount">(fleet-wide)</span></h2>
{_timeline_svg(fleet, order)}</div>
<div class="cols">
<div class="panel"><h2 id="h-hist">Best-score distribution</h2><div id="c-hist"></div></div>
<div class="panel"><h2>Families</h2><div id="t-fam"></div></div>
</div>
<div class="panel"><h2>Queue wait — slowest 10 (bar = p50, whisker = p95)</h2><div id="c-waits"></div></div>
<div class="panel"><h2>Iteration outcomes — 10 most-iterated</h2>
<div class="legend">{"".join(f'<span class="lg"><i style="background:var(--o-{k})"></i>{k}</span>' for k in OUTCOME_KEYS)}</div>
<div id="c-oc"></div></div>
<script id="data" type="application/json">{payload}</script>
"""
    sub = (f"generated {gen}"
           + (f" · auto-refresh {refresh}s" if refresh else "")
           + " · filter with the bar above · click a problem name to deep-dive")
    return _shell("SOL-ExecBench solver — run dashboard", sub, body, refresh,
                  extra_js=_HUB_JS)


_STATUS_CHIP = {
    "accepted": "o-accepted", "dominated": "o-dominated", "rejected": "o-rejected",
    "duplicate": "o-duplicate", "incorrect": "o-incorrect", "flaky": "o-flaky",
    "error": "o-error", "planned": "o-dominated",
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


_ART_LABEL = {"ctx": "context", "design": "design", "strat": "strategy", "handoff": "handoff"}


def _progression_table(p: dict, has_traj: set | None = None,
                       avail: dict | None = None) -> str:
    """The per-problem timeline: every candidate in chronological order, each with
    drill-downs into EVERYTHING it saw and produced — the exact context passed to
    the agent (CONTEXT.md, incl. the coach card), its design, strategy, the kernel,
    the trajectory, and any handoff. Open a row's links to inspect that artifact."""
    if not p["candidates"]:
        return '<p class="muted">no candidates yet</p>'
    has_traj = has_traj or set()
    avail = avail or {}
    rows = []
    for c in p["candidates"]:
        t = _localtime(c["ts"])
        chip = (f'<span class="chip" style="background:var(--{_STATUS_CHIP.get(c["status"], "o-dominated")})">'
                f'{_esc(c["status"])}</span>')
        cal = "–" if c.get("sol_score_cal") is None else f"{c['sol_score_cal']:.3f}"
        best = "" if c.get("best_after") is None else f"{c['best_after']:.3f}"
        have = avail.get(c["cand"], set())
        links = []
        if "ctx" in have:   # lead with context — "what was this agent actually told?"
            links.append(f'<button class="link" data-code="ctx:{_esc(c["cand"])}">context</button>')
        if c.get("solution"):
            links.append(f'<button class="link" data-code="{_esc(c["cand"])}">kernel</button>')
        if c["cand"] in has_traj:
            links.append(f'<button class="link" data-code="traj:{_esc(c["cand"])}">trajectory</button>')
        for key in ("design", "strat", "handoff"):
            if key in have:
                links.append(f'<button class="link" data-code="{key}:{_esc(c["cand"])}">{_ART_LABEL[key]}</button>')
        links_html = " · ".join(links) or '<span class="muted">–</span>'
        rows.append(
            "<tr>"
            f'<td>{t}</td><td>{_esc(c["cand"])}</td><td>{_esc(c.get("model") or "")}</td>'
            f'<td class="strat">{_esc(c.get("strategy") or "")}</td>'
            f'<td>{chip}</td>'
            f'<td data-v="{c.get("sol_score_cal") or -1}"><b>{cal}</b></td>'
            f'<td data-v="{c.get("best_after") or -1}">{best}</td><td class="views">{links_html}</td>'
            "</tr>")
    return ('<table class="sortable"><thead><tr><th>time</th><th>cand</th>'
            "<th>agent</th><th>strategy</th><th>status</th><th>expected SOL</th>"
            "<th>best after</th><th>inspect</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def _traj_blocks(p: dict, runs_dir: Path) -> tuple[str, set]:
    """Agent-trajectory templates (reasoning + tool calls) per candidate, read from
    runs/<task>/work/<cand>/trajectory.txt. Returns (html, {cands with a trajectory})."""
    out, have = [], set()
    wroot = Path(runs_dir) / str(p["task"]) / "work"
    for c in p["candidates"]:
        tf = wroot / str(c["cand"]) / "trajectory.txt"
        if not tf.exists():
            continue
        txt = tf.read_text(errors="replace")
        if len(txt) > 80000:                              # bound the page
            txt = "…(truncated to last 80k chars)…\n" + txt[-80000:]
        have.add(c["cand"])
        out.append(
            f'<template data-code="traj:{_esc(c["cand"])}" '
            f'data-title="{_esc(c["cand"])} — trajectory · {_esc(c.get("model") or "")}">'
            f'<pre class="traj"><code>{_esc(txt)}</code></pre></template>')
    return "".join(out), have


# Per-candidate artifacts we surface in the timeline: the EXACT context the agent
# saw (CONTEXT.md = coach card + frontier + playbook + failures) and what it did.
_ARTIFACTS = [("ctx", "CONTEXT.md", "context"), ("design", "DESIGN.md", "design"),
              ("strat", "strategy.txt", "strategy"), ("handoff", "handoff.md", "handoff")]


def _artifact_blocks(p: dict, runs_dir: Path) -> tuple[str, dict]:
    """Templates for each candidate's context artifacts, revealed in the modal.
    Returns (html, {cand -> set of artifact keys present})."""
    out: list[str] = []
    avail: dict[str, set] = {}
    wroot = Path(runs_dir) / str(p["task"]) / "work"
    for c in p["candidates"]:
        wd = wroot / str(c["cand"])
        if not wd.is_dir():
            continue
        for key, fname, label in _ARTIFACTS:
            f = wd / fname
            if not f.is_file():
                continue
            txt = f.read_text(errors="replace")
            if not txt.strip():
                continue
            if len(txt) > 80000:
                txt = "…(truncated to last 80k chars)…\n" + txt[-80000:]
            avail.setdefault(c["cand"], set()).add(key)
            out.append(
                f'<template data-code="{key}:{_esc(c["cand"])}" '
                f'data-title="{_esc(c["cand"])} — {label} · {_esc(c.get("model") or "")}">'
                f'<pre class="traj"><code>{_esc(txt)}</code></pre></template>')
    return "".join(out), avail


def _coach_panel(p: dict, runs_dir: Path) -> tuple[str, str]:
    """The cross-run reflection for this problem: the CURRENT coach card + a timeline
    of how the diagnosis evolved (from reflections.jsonl). Returns (panel, templates)."""
    pdir = Path(runs_dir) / str(p["task"])
    rf = pdir / "reflection.md"
    if not rf.is_file():
        return "", ""
    card = rf.read_text(errors="replace")
    snaps = []
    hf = pdir / "reflections.jsonl"
    if hf.is_file():
        for line in hf.read_text().splitlines():
            try:
                snaps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    hist, tmpls = "", ""
    if len(snaps) > 1:
        rows = []
        for i, s in enumerate(snaps):
            rows.append(
                f'<tr><td>{_localtime(s.get("ts",""))}</td>'
                f'<td>{_esc(s.get("status",""))}</td>'
                f'<td class="strat">{_esc((s.get("headline") or "")[:110])}</td>'
                f'<td><button class="link" data-code="reflect:{i}">view card</button></td></tr>')
            tmpls += (f'<template data-code="reflect:{i}" '
                      f'data-title="coach card @ {_localtime(s.get("ts",""))}">'
                      f'<pre class="traj"><code>{_esc(s.get("card",""))}</code></pre></template>')
        hist = ('<h3 class="coach-h">how the diagnosis evolved</h3>'
                '<table class="sortable"><thead><tr><th>time</th><th>status</th>'
                '<th>headline</th><th>card</th></tr></thead><tbody>'
                + "".join(rows) + "</tbody></table>")
    panel = (f'<div class="panel"><h2>Coach — cross-run reflection (fed to every agent as context)</h2>'
             f'<pre class="coach-card"><code>{_esc(card)}</code></pre>{hist}</div>')
    return panel, tmpls


def _submissions_panel(p: dict, runs_dir: Path) -> str:
    """Real leaderboard submissions for this problem (runs/<task>/submissions.jsonl)."""
    sf = Path(runs_dir) / str(p["task"]) / "submissions.jsonl"
    if not sf.exists():
        return ""
    subs: dict = {}
    for line in sf.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = e.get("submission_id") or e.get("id")
        if sid is not None:
            subs.setdefault(sid, {}).update(e)
    if not subs:
        return ""
    have_code = {c["cand"] for c in p["candidates"] if c.get("solution")}
    exp_by_cand = {c["cand"]: c.get("sol_score_cal") for c in p["candidates"]}   # expected SOL per kernel
    cur_exp = p.get("best_cal")                        # our current best expected SOL
    rows = []
    for sid, e in sorted(subs.items()):
        sc = e.get("sol_score")
        fast = f"{e['fast_1_count']}/{e['fast_1_total']}" if e.get("fast_1_total") else "–"
        cid = e.get("cand_id")
        # which kernel we submitted (strategy + a code link if we still have it)
        kern = _esc((e.get("cand_strategy") or "")[:52]) or (_esc(cid[:10]) if cid else "–")
        if cid in have_code:
            kern += f' <button class="link" data-code="{_esc(cid)}">code</button>'
        # our position on the public board + the current #1
        rank = e.get("board_rank")
        rank_s = f"#{rank} of {e.get('board_n','?')}" if rank else "–"
        top = e.get("board_top_sol")
        top_s = f"{top:.4f}" if top is not None else "–"
        cid_s = _esc(cid[:12]) if cid else "–"          # matches the CAND column in the solutions table
        exp_sub = exp_by_cand.get(cid)                  # expected SOL of the exact kernel we submitted
        exp_s = "–" if exp_sub is None else f"{exp_sub:.3f}"
        # do we have a better kernel now than the one we submitted? → re-submit
        resub = (' <span class="up" title="current best expected ('
                 f'{cur_exp:.3f}) beats this submission — re-submit">↑</span>'
                 if (exp_sub is not None and cur_exp is not None and cur_exp > exp_sub + 0.01) else "")
        rows.append(
            f"<tr><td>#{sid}</td><td>{cid_s}</td><td class='strat'>{kern}</td><td>{_esc(e.get('status', '–'))}</td>"
            f"<td data-v='{exp_sub or -1}'>{exp_s}{resub}</td>"
            f"<td data-v='{sc or -1}'><b>{'–' if sc is None else f'{sc:.4f}'}</b></td>"
            f"<td data-v='{rank or 999}'>{rank_s}</td><td data-v='{top or -1}'>{top_s}</td>"
            f"<td>{fast}</td></tr>")
    return ('<div class="panel"><h2>Leaderboard submissions (real, not estimate)</h2>'
            '<table class="sortable"><thead><tr><th>submission</th><th>cand</th><th>kernel</th><th>status</th>'
            '<th>expected@submit</th><th>real SOL</th><th>rank</th><th>leaderboard #1</th>'
            '<th>fast</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def build_detail(p: dict, slot_i: int, runs_dir: Path | None = None) -> str:
    conv = _line_chart(
        [{"label": f"#{p['task']}", "color": _slot(slot_i), "points": p["convergence"]}],
        x_label="GPU evals", right=120) if p["convergence"] else '<p class="muted">no evals yet</p>'
    order = {p["task"]: slot_i}
    stats = "".join([
        _tile("expected SOL (best)", "–" if p.get("best_cal") is None else f"{p['best_cal']:.3f}",
              "leaderboard estimate"),
        _tile("status", p["terminated"] or "running"),
        _tile("iterations / evals", f"{p['iters']} / {p['evals']}"),
        _tile("frontier size", str(p["frontier"])),
        _tile("wait p50 / p95", f"{_fmt_s(p['wait_p50'])} / {_fmt_s(p['wait_p95'])}"),
        _tile("agent", p["model"] or "–"),
    ])
    traj_html, has_traj = _traj_blocks(p, runs_dir) if runs_dir else ("", set())
    art_html, avail = _artifact_blocks(p, runs_dir) if runs_dir else ("", {})
    coach_panel, coach_tmpls = _coach_panel(p, runs_dir) if runs_dir else ("", "")
    subs = _submissions_panel(p, runs_dir) if runs_dir else ""
    modal = ('<div id="modal" class="modal" hidden><div class="modal-card">'
             '<div class="modal-bar"><b id="modal-title"></b>'
             '<button id="modal-x" class="link" aria-label="close">✕ close</button></div>'
             '<div id="modal-body"></div></div></div>')
    body = f"""
<div class="tiles">{stats}</div>
{coach_panel}
<div class="panel"><h2>Solution progression — timeline · inspect any candidate's context · kernel · trajectory</h2>
{_progression_table(p, has_traj, avail)}</div>
{subs}
<div class="panel"><h2>Convergence</h2>{conv}</div>
<div class="panel"><h2>Iteration outcomes</h2>{_outcomes_svg([p], order)}</div>
{_code_blocks(p)}{traj_html}{art_html}{coach_tmpls}{modal}
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
        (p_dir / f"{p['task']}.html").write_text(build_detail(p, order[p["task"]], runs_dir))

    index = out_dir / "index.html"
    tmp = index.with_suffix(f".{os.getpid()}.tmp")   # unique per process → no cross-render collision
    tmp.write_text(build_hub(data, refresh=refresh, detail_dir="p"))
    tmp.replace(index)
    return index
