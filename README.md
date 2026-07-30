# Vitamin D Scheduling Optimization Model

Determines optimal sun exposure schedules for maximizing
serum vitamin D (25(OH)D) in people with darker skin (Fitzpatrick types V-VI),
accounting for photoadaptation (tanning) dynamics.

## Overview

Existing models treat skin pigmentation as static and simulate single
sessions. This one adds dynamic photoadaptation: repeated UV exposure raises
effective melanin, progressively reducing vitamin D synthesis from subsequent
sessions. The optimizer picks the inter-session interval that maximizes
absolute serum D or efficiency per UV dose.

The model uses exact analytical step updates for the D, M, and SC states
between UV pulses, with 0.5-day sampling. Melanin and
stratum-corneum pulse updates use stable exponential saturation forms. The
tanning rise rate depends on dose through a Hill-style alpha(UV) term.

Start with `vitd_objectives.py` — the three decision problems people actually
face (fixed session budget, deficiency-vs-cancer Pareto frontier, minutes of
sun) — rather than the parameter sweeps in `vitd_analyses.py`. See
`MODEL_AND_EVIDENCE.md` for the current model description and per-paper/dataset
evidence, and `OPEN_WORK.md` for open questions, known issues, data requests,
and the parameter change log. Module docstrings are the source of truth for
what each file does; this list is deliberately not repeated here.

## How to run

```bash
pip install numpy scipy pandas matplotlib pytest
PYTHONIOENCODING=utf-8 python3 run_all.py   # full pipeline, rewrites RESULTS.md
python3 -m pytest                           # 196 tests
```

`PYTHONIOENCODING=utf-8` matters on Windows: the report contains non-ASCII
characters that crash the default cp1252 codec.

## Current calibration

Parameter values and their citations live in **one** place: the `vitd_model.py`
module docstring. They are deliberately not restated here — nine separate
restatements of `K_ATTEN` across the docs is how the numbers drifted out of sync
in the first place.

Structural points worth knowing before reading results:

- **D decay is state-dependent** (Datta 2017 power law), so 25(OH)D does *not*
  halve in one nominal half-life — clearance slows as D falls. The analytic
  solution is in `tests/test_model_invariants.py`.
- **Melanin has a small effect** on synthesis, per Young 2020 (FST II:VI ratio
  ~1.3-1.4). Do not expect pigmentation to dominate model behaviour; it does not.
- **The `(1 - D/D_max)` feedback term dominates** scheduling behaviour, and
  `D_max` is the least-supported number in the model. See `OPEN_WORK.md`
  known issue #3.

## Read this before quoting any "optimal interval"

The per-SED efficiency optimum reported below is **a property of the feedback
term, not of photoadaptation**, and it is not a usable clinical
recommendation. Three findings from ablation:

1. Setting `K_ATTEN = 0` (no melanin attenuation) changes the interval ranking
   by ~0.3%. Setting `K_ATTEN = 0` *and* `K_SC = 0` leaves the ranking intact.
   The ordering comes from `(1 - D/D_max)`: sparse dosing keeps mean D low, so
   each pulse meets less inhibition.
2. `gross_synth_per_SED` therefore has **no interior optimum**. It increases
   monotonically as dosing approaches zero (21 d → 1.027, 42 d → 1.049,
   60 d → 1.056 nmol/L per SED for FST V).
3. At the reported optima, final D is *below* the starting baseline. FST V at
   1 SED every 21 d ends at 24.0 nmol/L from a 25.0 start — **a net loss** of
   vitamin D, scored as maximally "efficient."

Using `net_final_D_change_per_SED` instead — already implemented in
`vitd_model.py` — gives a genuine interior optimum at **~3 days** (+0.525
nmol/L per SED for FST V). Monte Carlo over all documented prior ranges puts the
optimum at **2-4 days in 86% of draws** (`vitd_sensitivity.py`), so that answer
is robust to parameter uncertainty in a way the 21-37 day figure never was.

Better still, don't optimise a per-SED ratio at all. `vitd_objectives.py` poses
the same questions with cumulative UV **held fixed**, so no schedule can win by
dosing less:

| Question | Function |
|---|---|
| "I can get outside N times in 12 weeks — when?" | `optimal_spacing_fixed_sessions` |
| "Stay sufficient at least cancer risk" | `deficiency_vs_risk_frontier` |
| "How many minutes of sun, for my skin type?" | `minutes_for_target` |

## Results, open questions, and evidence

See `RESULTS.md` (auto-generated, rewritten on every `run_all.py` run) for the
full validation status and analysis output. See `MODEL_AND_EVIDENCE.md` for the
current model description and per-paper/dataset evidence cards, and
`OPEN_WORK.md` for known issues, open gaps, data requests, assumptions, and the
parameter change log.
