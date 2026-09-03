# Presentation figures

Conference-talk figures. Every script imports `slidestyle.use_slide_style()`
first and writes its PNG into this directory. Nothing outside `presentation/`
is read for style or written to.

Rebuild everything:

```bash
cd presentation/figs
for f in f2_feasibility_heatmap f3_frontload_saving f4_two_cancer_axes \
         f5_evidence_tiers f6_gamma_warp f7_fixed_vs_coupled; do
  PYTHONIOENCODING=utf-8 python3 $f.py
done
```

Each script asserts its headline numbers against the values below and fails
loudly if the underlying data has moved.

## `slidestyle.py`

Shared style. Exports `use_slide_style()`, `ACCENT` (`#0584a4`, teal),
`FST_COLORS` (ordered I..VI, light to dark), `FST_LABELS`, `GREY_INFEASIBLE`
(`#bdbdbd`), `GREYS`, `TEXT_COLOR` (`#0e2a47`, navy), `relative_luminance()`,
`fst_luminances()`, `pct()`.

**2026-08-30 display pass** (deck feedback: false precision, jargon on axes
and annotations): `ACCENT` moved from the original red `#c0392b` to teal
`#0584a4`; `TEXT_COLOR` is now navy `#0e2a47`, to match the decks' navy +
teal palette (its prior value isn't recorded anywhere in this repo -- see
the reconstruction note below -- so this is a set, not a documented move).
`FST_COLORS` and `GREY_INFEASIBLE` are unchanged.
Added `pct(x, decimals=0, show_plus=False)`, a small formatter so every
script rounds percentages the same way and uses a proper minus sign (`−`,
U+2212) rather than a hyphen. No script's numeric asserts changed — this
pass only changes what gets printed on the images. Per-figure display
changes below.

Note for future maintenance: `slidestyle.py` was found **missing from disk**
when this pass started, and was never in git history — it had a
`presentation/figs/slidestyle.py` entry in `.gitignore` until that block was
removed in the "deleted some files" commit, so it was gitignored-and-then-lost
with no tracked copy to recover from. It's currently untracked (not
gitignored), i.e. `git status` will offer to add it; whether it should be
committed or gitignored again is a repo-policy call left to the user, not
decided here. The file here was reconstructed from this README's documented
API, the FST luminance table below (which pins down `FST_COLORS` and the
luminance formula exactly, since both reproduce the table to 4 dp), and usage
in the six figure scripts. `GREYS` (the four tier greys F5 draws with) could
not be recovered the same way — no script or doc records its old values — so
its values in the current file are a fresh choice (`#595959` .. `#b8b8b8`,
darkest first), not a reconstruction. If the original greys mattered for some
other reason, they are not recoverable from this repo.

`use_slide_style()` sets DPI 220, white background, and
`savefig.bbox="tight"` — the exact left/right/top/bottom margins the
original file used are one of the things that couldn't be recovered, so
`bbox="tight"` (auto-crop to content) was chosen as the closest reproducible
behaviour, and it is sensitive to label text length. Current PNG sizes
against the previously recorded pixel dimensions: F2 2052x1138 (target
2056x1138, -0.2%/0%), F3 1692x1004 (1696x1020, -0.2%/-1.6%), F4 1721x1273
(1740x1273, -1.1%/0%), F7 1701x1004 (1719x1020, -1.0%/-1.6%) — all within a
couple percent. F5 2077x780 (1921x783, **+8.1%**/-0.4%) and F6 2077x665
(2363x683, **-12.1%**/-2.6%) fall outside that: F5's tier labels got longer
under the plain-English rule ("self-consistency" vs "INTERNAL"), and F6's
row labels got shorter ("Front-loaded" vs "front-loaded (gamma = 1.4)");
both directly shift how much the tight-crop keeps on the label side of the
figure. Visually both are unclipped and well-proportioned (see the PNGs) —
the deviation is in overall canvas size, not layout — but if the decks
place these two at fixed coordinates, check the fit before dropping them in.

The FST ramp is verified at import time: WCAG relative luminance must decrease
monotonically I -> VI with every adjacent gap at least 0.06, so the six read as
skin type in colour and stay separable in greyscale. Measured luminances:

| FST | hex | relative luminance | gap to next |
|---|---|---|---|
| I   | `#f2d8c3` | 0.7193 | 0.1994 |
| II  | `#e2b78e` | 0.5199 | 0.1686 |
| III | `#ce945f` | 0.3513 | 0.1309 |
| IV  | `#b47239` | 0.2203 | 0.0894 |
| V   | `#90582c` | 0.1309 | 0.0644 |
| VI  | `#67401f` | 0.0665 | - |

---

## F2 - `f2_feasibility_heatmap.png`, `f2_feasibility_heatmap_free.png`

Script: `f2_feasibility_heatmap.py`
Data: `tables/primary_minimum_schedule.csv`

Cumulative annual UV (`total_sed`) over FST I..VI (rows, I at bottom,
`origin='lower'`) by session budget (columns 52, 104, 146, 183, 260, 365).
Infeasible cells are solid `GREY_INFEASIBLE` with a black x. `YlOrRd`
colourbar, labelled "Annual UV needed (SED)". Each output covers one
`spacing` mode.

Asserted:
- `spacing == "uniform"`: exactly **9 of 36** cells infeasible.
- `spacing == "free"`: exactly **7 of 36** cells infeasible.
- exactly one CSV row per (fst, n_sessions, spacing) combination.

**Display (2026-08-30):** cell values rounded to whole SED (`576.0` -> `576`,
not two decimals); colourbar label reworded off "cumulative"; teal/navy
palette from `slidestyle.py`. Asserted values above are unchanged.

## F3 - `f3_frontload_saving.png` (hero)

Script: `f3_frontload_saving.py`
Data: `tables/primary_minimum_schedule.csv`, `n_sessions == 365`,
`spacing` `uniform` vs `free`

Percent change in `total_sed` from uniform to free spacing, by FST. Horizontal
bars, FST I at top, value labels rounded (see below).

Asserted (each value must match to 4 decimal places):

| FST | percent change in `total_sed` |
|---|---|
| I   | -29.5604% |
| II  | -29.7758% |
| III | -30.2468% |
| IV  | -32.0548% |
| V   | -33.9544% |
| VI  | -35.5031% |

**Display (2026-08-30):** labels use `pct(v, decimals=1)`, not the whole-number
default — at 0 decimals, FST I/II/III (-29.5604, -29.7758, -30.2468) all round
to the same "-30%" and collapse onto each other, so this figure is the one
place the "1 decimal where whole numbers collide" clause actually fires.
Rendered labels: -29.6%, -29.8%, -30.2%, -32.1%, -34.0%, -35.5%. X-axis label
reworded from "Reduction in cumulative annual UV to hold 50 nmol/L (%)" to
"Reduction in annual UV needed to stay sufficient (%)". Proper minus sign
throughout. Asserted values above are unchanged.

## F4 - `f4_two_cancer_axes.png`

Script: `f4_two_cancer_axes.py`
Data: `tables/joint_schedule_optimization.csv`

Scatter of all three schedule families on the two cancer-risk axes: each
non-uniform family (`two_phase` with `free_spacing == True`, i.e.
front-loaded; and `exponential`, always `free_spacing == True`) merged
against its own evenly-spaced (`two_phase`, `free_spacing == False`)
baseline for the same (`fst`, `n_sessions`), keeping only pairs feasible on
both sides. x = percent change in `scc_index` (cumulative-dose risk),
y = percent change in `intermittent_index` (burn-driven risk), both
relative to evenly-spaced, so evenly-spaced itself sits at the origin by
construction. Front-loaded points are teal circles, exponential-taper
points are rust triangles, evenly-spaced is a single grey square at (0, 0).

Asserted:
- **22** feasible evenly-spaced rows, **18** front-loaded pairs, **4**
  exponential pairs (feasible on both sides after the (`fst`,`n_sessions`)
  merge).
- front-loaded median: **+7.65%** cumulative-dose / **-23.50%** burn-driven.
- exponential median: **+7.27%** cumulative-dose / **-19.71%** burn-driven.

**Display (2026-09-03):** three revisions in one sitting. (1) Redrawn from
a slope chart (each pair as a line between two categorical axis positions)
to a scatter, after feedback that the crossing slope lines didn't read
clearly. (2) A quadrant-scatter pass added corner labels and two-line axis
subtitles explaining the sign convention; cut back out as too much text on
the plot itself. (3) Split the single "front-loaded" series into its two
underlying schedule families (front-loaded / exponential-taper) plus an
explicit evenly-spaced reference point, after noticing the original slope
chart's "free spacing" pairs had silently mixed both families together —
4 of the original 22 pairs were actually `exponential`, not `two_phase`.
Splitting them out is why the front-loaded intermittent-risk median moved
from -21.36% (22 mixed pairs) to -23.50% (18 pure front-loaded pairs); the
cumulative-dose median (+7.65%) happened not to move. The figure now
carries only axis labels and a 3-entry legend, no crosshair or corner
text. See the sign-convention note in `RESULTS.md` sec. 13c — this
figure's convention (percent change in the index itself) is the one to use
everywhere; `RESULTS.md`'s own prose numbers use the opposite sign.

## F5 - `f5_evidence_tiers.png`

**Not used on any slide as of 2026-08-30** — cut from both decks (the
reviewer called the internal tier taxonomy the most machine-generated thing
in either one). Retained here, buildable, for the write-up and Q&A.

Script: `f5_evidence_tiers.py`
Data: none - the tier counts are given and hard-coded in the script.

What the 14 validation checks actually test, strongest evidence first. The
top tier carries `ACCENT`; the other four are graded greys.

| tier (internal name) | display label | n | gloss |
|---|---|---|---|
| QUANTITATIVE | external, numeric | 2 | external data, real numeric comparison |
| DIRECTIONAL  | direction only | 3 | external data, direction only |
| INTERNAL     | self-consistency | 4 | self-consistency, not biology |
| CIRCULAR     | fitted to this data | 3 | data the model was fitted to |
| NOT_EVIDENCE | not vitamin-D evidence | 2 | its own docstring says it is not vitamin-D evidence |

Asserted: the five tier counts sum to **14**.

**Display (2026-08-30):** the internal tier constants (`QUANTITATIVE`,
`NOT_EVIDENCE`, etc.) no longer appear on the figure itself — the bars are
labelled with the plain-English "display label" column above instead. Teal
+ graded-grey palette from `slidestyle.py`'s `GREYS`. Counts and their sum
are unchanged.

## F6 - `f6_gamma_warp.png`

Script: `f6_gamma_warp.py`
Data: none - session days come from `vitd_optimization.warp_days` (repo root
put on `sys.path` by the script).

Two 365-day timelines: `warp_days(N=52, T_days=365, gamma=1.0)` (uniform) above
`warp_days(N=52, T_days=365, gamma=1.4)` (front-loaded). N = 52 rather than 365
so individual session ticks resolve on a slide.

Asserted: both rows carry exactly **52** ticks. Reported gaps: uniform first
gap 7.0192 d; front-loaded first gap 1.4451 d, last gap 9.7125 d.

**Display (2026-08-30):** row labels reworded off the exponent —
"uniform (gamma = 1.0)" -> "Evenly spaced", "front-loaded (gamma = 1.4)" ->
"Front-loaded". Gamma is off the slides entirely now, so it no longer
appears on the figure either. Asserted values above are unchanged.

## F7 - `f7_fixed_vs_coupled.png` (+ `f7_fixed_vs_coupled_caption.txt`)

Script: `f7_fixed_vs_coupled.py`
Data: none - two live `vitd_model.simulate` runs (repo root put on `sys.path`
by the script). No file outside `presentation/` is touched.

FST VI, 180 days, one session every 3 days at a fixed **4.0 SED** per session
(60 sessions, 240.0 SED cumulative), from `D0 = 25.0 nmol/L`. 4.0 SED is 26.3%
of the FST VI MED of 15.2 SED used in `vitd_model.compute_safety_sidecar`, so
every session is sub-erythemal.

The two runs differ only in whether the adaptive skin states move. The pinned
run uses documented `simulate` keyword overrides - `alpha_override=0.0`,
`alpha_SC_override=0.0`, `k_damage_override=0.0` - which zero the per-session
melanin and stratum-corneum update terms. Nothing in the model is patched.

Asserted:
- pinned run: melanin stays at `ModelParams().M_baseline(6)` = 1 throughout.
- pinned run: stratum corneum stays at 0 throughout.
- coupled run: melanin does rise above baseline.
- 60 sessions, cumulative SED 240.0.

Reported: day-180 25(OH)D is 51.0596 nmol/L coupled vs 58.0184 nmol/L pinned,
a shortfall of 6.9588 nmol/L (11.99% of the pinned value). Melanin rises
1.0000 -> 1.0760; stratum corneum 0 -> 0.3225.

Caveat carried in the caption file: absolute predictions here rest on the
uncalibrated SC priors (`SC_max`, `alpha_SC`, `K_SC`). Read the size of the gap
as illustrative of the mechanism, not as a calibrated quantity.

**Display (2026-08-30):** legend reworded off internal state names — "skin
states held at baseline" -> "If skin never adapted", "melanin + stratum
corneum allowed to adapt" -> "Real skin (tans and thickens)". End-gap
annotation rounded: "7.0 nmol/L lost to adaptation" -> "7 nmol/L lost to
adaptation". Added markers for where each curve first reaches the 50 nmol/L
floor, computed by linear interpolation between the simulation's own 0.5-day
samples (`crossing_day()` in the script) rather than hard-coded: pinned day
84.0000, coupled day 147.0000, labelled "day 84" / "day 147" — this is the
number the talk narration actually uses (63-day gap) and it was previously
only in speech, not on the figure. Caption file's last paragraph rewritten
out of jargon: "uncalibrated SC priors (SC_max, alpha_SC, K_SC)" -> "our
estimate of how fast the outer skin layer thickens is still rough". All
asserted values and the day-180 numbers above are unchanged.
