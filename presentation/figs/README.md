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

Shared style. Exports `use_slide_style()`, `ACCENT` (`#c0392b`), `FST_COLORS`
(ordered I..VI, light to dark), `FST_LABELS`, `GREY_INFEASIBLE` (`#bdbdbd`),
`GREYS`, `TEXT_COLOR`, `relative_luminance()`, `fst_luminances()`.

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
colourbar, labelled "Cumulative annual UV (SED)". Each output covers one
`spacing` mode.

Asserted:
- `spacing == "uniform"`: exactly **9 of 36** cells infeasible.
- `spacing == "free"`: exactly **7 of 36** cells infeasible.
- exactly one CSV row per (fst, n_sessions, spacing) combination.

## F3 - `f3_frontload_saving.png` (hero)

Script: `f3_frontload_saving.py`
Data: `tables/primary_minimum_schedule.csv`, `n_sessions == 365`,
`spacing` `uniform` vs `free`

Percent change in `total_sed` from uniform to free spacing, by FST. Horizontal
bars, FST I at top, value labels to two decimals.

Asserted (each value must match to 4 decimal places):

| FST | percent change in `total_sed` |
|---|---|
| I   | -29.5604% |
| II  | -29.7758% |
| III | -30.2468% |
| IV  | -32.0548% |
| V   | -33.9544% |
| VI  | -35.5031% |

## F4 - `f4_two_cancer_axes.png`

Script: `f4_two_cancer_axes.py`
Data: `tables/joint_schedule_optimization.csv`

`free_spacing == False` rows merged against `free_spacing == True` rows on
(`fst`, `n_sessions`), keeping only pairs feasible on both sides. Slope chart:
each pair is a line from its SCC-index percent change to its intermittent-index
percent change, so the two axes visibly diverge across the zero line.

Asserted:
- exactly **22** pairs feasible on both sides.
- median SCC change rounds to **+7.65%** (computed 7.6546%, worse).
- median intermittent change rounds to **-21.36%** (computed -21.3580%, better).

## F5 - `f5_evidence_tiers.png`

Script: `f5_evidence_tiers.py`
Data: none - the tier counts are given and hard-coded in the script.

What the 14 validation checks actually test, strongest evidence first.
QUANTITATIVE carries `ACCENT`; the other four are graded greys.

| tier | n | gloss |
|---|---|---|
| QUANTITATIVE | 2 | external data, real numeric comparison |
| DIRECTIONAL  | 3 | external data, direction only |
| INTERNAL     | 4 | self-consistency, not biology |
| CIRCULAR     | 3 | data the model was fitted to |
| NOT_EVIDENCE | 2 | its own docstring says it is not vitamin-D evidence |

Asserted: the five tier counts sum to **14**.

## F6 - `f6_gamma_warp.png`

Script: `f6_gamma_warp.py`
Data: none - session days come from `vitd_optimization.warp_days` (repo root
put on `sys.path` by the script).

Two 365-day timelines: `warp_days(N=52, T_days=365, gamma=1.0)` (uniform) above
`warp_days(N=52, T_days=365, gamma=1.4)` (front-loaded). N = 52 rather than 365
so individual session ticks resolve on a slide.

Asserted: both rows carry exactly **52** ticks. Reported gaps: uniform first
gap 7.0192 d; front-loaded first gap 1.4451 d, last gap 9.7125 d.

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
