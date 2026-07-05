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
   .venv/bin/solver report --demo        # -> .cache/demo/report.html
   ```
   To force fresh demo journals: `rm -rf .cache/demo` first. Real runs:
   `solver report --runs-dir runs`.

2. **Screenshot BOTH themes** with headless Chrome. The page honors a
   `?theme=light|dark` query override (JS sets `data-theme`), so no CDP
   media-emulation is needed:
   ```bash
   CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
   "$CHROME" --headless=new --disable-gpu --hide-scrollbars \
     --window-size=1440,2000 \
     --screenshot=docs/screenshots/dashboard-light.png \
     "file://$PWD/.cache/demo/report.html?theme=light"
   "$CHROME" --headless=new ... --screenshot=docs/screenshots/dashboard-dark.png \
     "file://$PWD/.cache/demo/report.html?theme=dark"
   ```
   (Ignore Chrome's "Failed parsing certificate policies" stderr noise.
   Bump `--window-size` height if the page grows taller than the capture.)

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
