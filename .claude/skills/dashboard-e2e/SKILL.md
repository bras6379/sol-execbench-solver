---
name: dashboard-e2e
description: E2E visual test loop for the solver run dashboard (solver report). Use whenever the dashboard HTML/CSS/SVG in solver/report.py changes, or when asked to verify/screenshot the dashboard. Renders demo data, screenshots light+dark with headless Chrome, inspects the images, and iterates until clean.
---

# Dashboard E2E: render → screenshot → look → fix → repeat

The dashboard is a self-contained static HTML (`solver/report.py`). The only
trustworthy test is *looking at the rendered pixels* — the palette validator
checks color, not layout. This skill is the loop that catches invisible
strokes, label collisions, overflow, and theme bugs before a human sees them.

## The loop

1. **Render from demo data** (deterministic, no engine/GPU needed):
   ```bash
   .venv/bin/solver report --demo        # site -> .cache/demo/out/
   ```
   To force fresh demo journals: `rm -rf .cache/demo` first. Real runs:
   `solver report --runs-dir runs` → publishable static site in `out/`
   (`index.html`, `p/<task>.html`, `p/<task>/<cand>.html`).

2. **Screenshot BOTH themes and EVERY page type** with headless Chrome. Pages
   honor `?theme=light|dark` (JS sets `data-theme`) and problem pages honor
   `?code=<cand>` to auto-open the candidate **code modal** — both make the
   modal and theme states screenshot-able without CDP:
   ```bash
   CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
   S=docs/screenshots; O="file://$PWD/.cache/demo/out"
   "$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size=1440,3200 \
     --screenshot=$S/dashboard-light.png "$O/index.html?theme=light"
   # + ?theme=dark ; a problem page ($O/p/<task>.html) ; the modal
   # ($O/p/<task>.html?theme=dark&code=<cand>). Bugs live in every page type.
   ```
   The hub is **JS-rendered from embedded data** and **filterable**: pages
   honor `?fam=a,b` (select families) and `?q=text` (task/name/family/agent,
   comma = OR) to pre-apply a filter — use these to screenshot a *scoped*
   state (fleet score, convergence, tiles, tables all recompute for the
   subset). Tall pages (the 235-problem hub) need `--window-size=1440,3200`.
   Ignore Chrome's "Failed parsing certificate policies" stderr. Targets:
   `dashboard-light/dark.png`, `dashboard-filtered.png`,
   `problem-detail.png`, `candidate-modal.png`.

   **Watch for `[hidden]` overridden by `display:`** — a `.modal{display:flex}`
   beats the UA `[hidden]{display:none}` (specificity), so a hidden modal
   renders as an empty overlay. The screenshot loop caught exactly this; the
   fix is an explicit `.modal[hidden]{display:none}`. Always screenshot a
   problem page *without* `?code=` to confirm no stray overlay.

3. **Actually look at the PNGs** (Read tool renders them). Check every item:
   - every mark visible — a CSS class with no fill/stroke renders as
     *nothing* (this loop caught an invisible p95 whisker);
   - no text collisions (axis captions vs tick labels; direct labels vs
     lines) and no overflow/clipping at panel edges;
   - both themes: readable ink, series distinguishable, no light-theme
     colors leaking into dark;
   - legends present for multi-series charts + direct labels on line ends;
   - numbers sane vs the demo data (score in [0,1], baseline tick at 0.5,
     utilization plausible, table consistent with charts).

4. **Fix `solver/report.py`, go to 1.** Iterate until a pass finds nothing.

5. **Commit the final screenshots** to `docs/screenshots/` — they are the
   visual record for review (git-visible on purpose).

## Notes

- Hover tooltips are JS-driven and invisible in static screenshots — verify
  them by opening the file in a real browser (`open .cache/demo/report.html`)
  when tooltip code changes.
- The demo generator is `solver/demo_data.py` (seeded, deterministic). If a
  new chart needs a data shape the demo doesn't produce (e.g. errors,
  suspended problems), extend the generator so the visual state is testable.
- Palette changes must re-run the dataviz validator (see the dataviz skill)
  *in addition to* this loop — validator for color, this loop for layout.
