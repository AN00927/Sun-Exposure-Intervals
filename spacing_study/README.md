# `spacing_study/` — is daily exposure actually optimal for vitamin D?

A self-contained sub-study answering one question:

> **Ignoring all risk and harm, is going outside every single day optimal for
> pure vitamin D synthesis — or is there a spacing (every 2 days, every 3
> days, some burst pattern) that yields more serum 25(OH)D?**

The parent repo had assumed the answer was trivially "daily", because serum
25(OH)D response is monotone in UV dose. That assumption had never been
tested against the model's own photoadaptation and feedback machinery, which
pull in the opposite direction.

**Answer, and the numbers behind it: [`results/FINDINGS.md`](results/FINDINGS.md).**

## Running it

```
PYTHONIOENCODING=utf-8 python3 spacing_study/run_spacing_study.py
```

Takes about 4 minutes. It writes only into `spacing_study/results/` and
`spacing_study/figs/`, and modifies nothing else in the repo. It does not
import or invoke `run_all.py`.

## What this is

| file | what it does |
|---|---|
| `spacing_model.py` | thin wrapper over `vitd_model.simulate`: schedule + FST + horizon + D0 -> outcome metrics. Also defines the mechanism-ablation variants. |
| `spacing_experiments.py` | the experiment grid (A-G) and the parameter sweeps. |
| `run_spacing_study.py` | entry point: runs everything, writes CSVs, figures and `FINDINGS.md`. |

The physiology is **imported, not reimplemented** — `vitd_model.py` and
`vitd_physiology.py` are the forward model, unchanged.

## Experiments

| | question |
|---|---|
| **A** | Matched **per-session dose**: same SED every session, vary the interval. The advisor's literal question. |
| **B** | Matched **total UV budget**: same total SED over the horizon, distributed over fewer/larger or more/smaller sessions. Isolates dose-response concavity against photoadaptation. |
| **C** | Matched **minutes outdoors** per week, split daily vs every-other-day vs weekly. |
| **D** | **Non-uniform patterns**: burst-then-rest (5 on / 2 off, 10 on / 10 off, ...) plus an exhaustive search over every rotation-distinct 14-day on/off pattern at a fixed duty cycle. |
| **E** | **Mechanism decomposition**: photoadaptation off, SC off, melanin adaptation off, feedback `g(D)` off, D decay off — plus a sweep of the retired `K_ATTEN` coefficient and a threshold scan on `K_SC`. |
| **F** | **Skin type sweep**, FST I-VI, with FST V-VI reported explicitly per the repo's dark-skin priority directive. |
| **G** | **Outcome-metric sensitivity**: final D, mean D, minimum D, and fraction of days above 50 nmol/L, for every comparison. |

## What is deliberately stripped out

This is a **risk-free, fixed-ambient** study by design. Removed relative to the
main pipeline:

- skin cancer, erythema, MED caps, CPD/DNA-damage constraints — **all of it**
- latitude, solar zenith angle, season, cloud, time of day
- behaviour adjustment (shade, clothing, time-outdoors multipliers)
- oral/dietary intake (already absent from the parent model)

Several schedules that "win" in these tables deliver per-session doses far
above an MED. Nothing here is a recommendation.

## Stated assumptions

All are module-level constants at the top of `spacing_model.py`, easy to
change and re-run. Parameter *values* are not restated here — the canonical
source for every scientific parameter is the `vitd_model.py` module docstring.

1. **One constant ambient UV level, every session, every day of the year.**
   There is no seasonal or geometric variation. This is the single biggest
   simplification and the one most likely to matter: a seasonal ambient
   profile could create a non-daily optimum for reasons that have nothing to
   do with photoadaptation (dose simply is not available on some days). That
   is a different question and is not addressed here.
2. **One fixed exposed-BSA fraction**, set to the model's own reference BSA so
   the BSA scaling factor is exactly 1.0 and cannot confound the comparison.
3. **Solar spectral weighting** taken from the repo's default sun entry in
   `SOURCE_CALIBRATION_TABLE`. Per repo directive, `E_source` is never treated
   as universal across sources; this study is sun-only.
4. **A fixed SED-per-minute rate** for experiment C only, to convert minutes
   outdoors into dose. It is a stated constant, not a modelled quantity.
5. **One starting 25(OH)D** by default (a mildly insufficient adult baseline),
   swept 15-130 nmol/L in the sensitivity section.
6. **Horizons of 90, 365 and 728 days.** The 728-day run supplies
   *steady-state* metrics averaged over the final full schedule period. This
   matters: the serum half-life in this model is of order months, so a
   365-day comparison is still on its transient, and a burst schedule that
   front-loads doses inside each period picks up a spurious mean-D advantage
   that vanishes once the system settles.

## Limitations

The full list is section 6 of `results/FINDINGS.md`. The three that matter
most:

- **This is a within-model result.** It reports what `vitd_model.py` implies,
  not what a human body does.
- **Absolute levels inherit the repo's standing flags** — `SC_max`,
  `alpha_SC` and `K_SC` are uncalibrated priors, and `A` is a light-skin
  extrapolation wherever it is applied to FST III-VI. The *rankings* are much
  more robust than the levels.
- **Melanin photoadaptation is structurally inert in the shipped model.**
  `K_ATTEN` is a retired parameter pinned at zero, so tanning has no effect on
  synthesis and this study cannot test the tanning hypothesis on its merits —
  only bound how large the coefficient would have to be for it to win. See
  `vitd_model.RETIRED_PARAMETERS`.

## Calibration status

**Nothing here is calibrated.** Every parameter sweep varies a value to
measure how the conclusion depends on it and then discards it;
`calibration_approval.yaml` was not touched, and no shipped parameter changed.
The `K_ATTEN` sweep runs inside `vitd_model.allow_retired_parameter_sweep()`,
the context manager that exists for exactly this purpose.
