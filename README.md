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

## Files

- **`vitd_model.py`** — Core three-state analytical model (D, M, and SC).
  All parameters carry literature citations in the module docstring.
- **`vitd_validation.py`** — Twelve checks: three sanity guards, seven
  literature/model validation checks (de Boer 2025, SC validation, Bogh,
  Farrar, Bogh baseline/BSA, Karppinen 2017, metric ordering, K_ATTEN
  seasonal amplitude), and one NHANES seasonal direction check
  (warning-only).
- **`vitd_analyses.py`** — Twelve analyses (A–L): alpha sweep, K_ATTEN sweep,
  beta sweep, two-framing schedule comparison, fine-grid interval sweep,
  scipy.optimize continuous optimization, maximum interval for sufficiency,
  loading dose optimization, maintenance dose splitting, sun exposure
  scheduling, example trajectories, constant-sessions optimization.
- **`vitd_genetics.py`** — Genetics presets for common vitamin-D-pathway
  variants (CYP2R1, VDR, GC, CYP24A1). Flags confidence caveats given
  individual variability (Datta 2019).
- **`vitd_optimization.py`** — Constant-sessions and best-interval analyses
  with AUC deficiency below 50 nmol/L as a clinical feasibility metric.
- **`vitd_physiology.py`** — Mechanistic sub-models: body-weight volumetric
  dilution, previtamin-D3 photodegradation, biphasic 25-hydroxylation, and the
  quantitative skin-cancer risk model. All opt-in and separately ablatable.
- **`vitd_objectives.py`** — The three decision problems people actually face
  (fixed session budget, deficiency-vs-cancer Pareto frontier, minutes of sun).
  **Start here** rather than with the parameter sweeps.
- **`vitd_benchmark_webbkift.py`** — Held-out external benchmark. Nothing in the
  repo is calibrated to it; the model currently fails its dose-ratio check.
- **`vitd_sensitivity.py`** — Which parameters matter, which pairs are
  unidentifiable, and headline numbers as intervals rather than point estimates.
- **`PAPER_REQUESTS.md`** — Sources not in the repo that would improve accuracy,
  ranked, with what to extract from each and what it would change.
- **`vitd_dose_conversion.py`** — Spectral weighting: physical dose → effective
  dose via lamp and action spectra.
- **`vitd_protocols/`** — Published trial protocols, including the Bogh 2012
  dose schedule and its J/cm² → erythemal SED conversion.
- **`run_all.py`** — Runs validation + all 12 analyses, writes `RESULTS.md`.
- **`RESULTS.md`** — Auto-generated summary with figures and interpretation.
- **`KNOWN_ISSUES.md`** — Current defects, caveats, and resolved-with-caveat items.
- **`OPEN_QUESTIONS.md`** — Dataset provenance cards, open data gaps, unresolved
  conflicts. (Replaces the retired `SCIENTIFIC_AUDIT_HANDOFF.md`.)
- **`PAPER_SUMMARIES.md`** — Per-paper evidence cards (46 papers).
- **`ASSUMPTION_REGISTER.md`**, **`PARAMETER_CHANGES.md`** — What we assume and
  what changed, with sources.
- **`figs/`**, **`tables/`** — Generated plots and sweep CSVs. Tracked
  deliberately so each commit records what the model produced.
- **`audit_outputs/`** — Generated dataset profiles, PDF inventories, and
  extraction snippets.

## How to run

```bash
pip install numpy scipy pandas matplotlib pytest
PYTHONIOENCODING=utf-8 python3 run_all.py   # full pipeline, rewrites RESULTS.md
python3 -m pytest                           # 129 tests
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
  `D_max = 150` is the least-supported number in the model. See KNOWN_ISSUES #3.

## Analysis Index

Analyses are organized by letter. Current modules and their coverage:

| Module | Analyses | Coverage |
|--------|----------|----------|
| `vitd_analyses.py` | A–L | Alpha, K_ATTEN, beta sweeps; two-framing; fine-grid; scipy optimize; max interval; loading dose; maintenance split; sun exposure; trajectories; constant-sessions |
| `vitd_optimization.py` | Best-interval, constant-sessions | AUC deficiency below 50 nmol/L for clinical feasibility |

## Headline results

**Validation**: 9/10 required checks pass, 2/2 warning-only checks pass
(last verified 2026-07-30; regenerate with
`PYTHONIOENCODING=utf-8 python3 run_all.py`). Key results:

- **Bogh 2012**: Passes. Model delta **+56.0** vs observed **+55.8**
  (error 0.2 nmol/L, tolerance ±10). This was previously recorded as a
  structural failure; the cause was a dose-unit error — absolute J/cm² was
  passed to `simulate()` as erythemal SED, omitting the ~5.5× TL-01
  conversion, and the 35% reference BSA was used for a whole-body protocol.
  **No parameter was tuned.** See KNOWN_ISSUES #1.
- **Farrar 2013 FST V**: D_final and delta are in a plausible window.
- **Bogh baseline/BSA**: direction check passes.
- **de Boer 2025 photoadaptation**: melanin and SC respond in the right direction.
- **Sanity checks**: single-bolus, ordering, and photoadaptation reduction all pass.

### Read this before quoting any "optimal interval"

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

### Sensitivity: the best-documented parameter barely matters

Across each parameter's full plausible range, swing in steady-state 25(OH)D:

| parameter | swing | note |
|---|---|---|
| `A` | **36%** | synthesis scale |
| `K_const` | 19% | constitutive pigment penalty |
| `K_SC` | 10% | stratum corneum |
| `D_max` | 8% | *origin unconfirmed* |
| `K_ATTEN` | **0.3%** | longest citation trail in the repo |

`K_ATTEN` has the most documentation, the most caveats, and the least influence.
Three pairs — `alpha_SC~SC_max`, `K_const~K_ATTEN`, `K_ATTEN~K_SC` — are jointly
**unidentifiable** (>99% of the response surface on a single ridge), so only
their combinations are constrained by current data.

### Skin cancer is scored, not just flagged

SCC tracks **cumulative** dose; melanoma and BCC track **intermittent** bursts
(Elwood 1997: intermittent OR 1.71, heavy occupational OR 0.86 — *protective*).
At a fixed UV budget, splitting 52 SED into weekly 1-SED sessions instead of two
26-SED holidays leaves SCC risk identical and cuts the melanoma/BCC index
**~680×**. Frequent-and-small beats rare-and-large — and for once this agrees
with the vitamin D objective instead of opposing it.

### The held-out benchmark fails

Webb/Kift 2018 report FST V needing 2.5-3.0× the dose of white Caucasians; the
model gives **1.40×**. `K_const` is calibrated to Young 2020's lamp-based,
matched-BSA ratio of 1.3-1.4, which is a different quantity from a whole-body
naturalistic response. Every real-world minutes recommendation for darker skin
is therefore optimistic. This is the highest-value calibration target in the
repo (`OPEN_QUESTIONS.md` gaps 22-23).

### Sweep results (efficiency framing, for reference)

**Alpha sweep**: The optimal interval (21 d under efficiency framing, 1 d under
absolute framing) holds across alpha = 0.01 to 0.40 for FST V and VI — because
melanin barely participates (see above), not because alpha is well constrained.
At 1 SED/session, M moves only +0.019 over 84 days.

**Beta sweep**: Robust across melanin decay half-lives of 21-120 days. Beta
shifts absolute D but does not change interval ranking.

**K_ATTEN sweep**: Same robustness, for the same reason.

**Two-framing finding**: Under matched per-session dose (1 SED/session),
daily dosing wins absolute D and sparse dosing wins gross efficiency. The
trade-off is real; the specific sparse optimum is not.

**Fine-grid (1-30 d) and scipy.optimize**: efficiency optima at 30 d and ~59 d
respectively, drifting upward with the search bound — the signature of a
monotone objective, not a peak. Absolute-D optimum ~2 d.

Best-interval and constant-sessions analyses report **AUC deficiency below
50 nmol/L**, which — unlike per-SED efficiency — penalises the net-loss
schedules above.

## Open questions

1. **Source / dose-unit calibration**: Per-study calibration via
   `SOURCE_CALIBRATION_TABLE` is implemented; `E_source=1.0` is the solar
   broadband reference and per-study values (e.g. `tl01_nb_uvb=2.44`) cover lab
   sources. The remaining risk is *unit* handling upstream of `E_source`, which
   is what broke Bogh 2012: physical J/cm² must be converted to erythemal SED
   before it reaches `simulate()`. Lloyd 2004 found TL-01 cabinet calibration
   varies ±60% between departments, so lab-source doses carry ~2× uncertainty.
2. **Baseline-dependent dose-response [C4]**: Bogh 2011/P5 derives
   `delta = 7.1 - 0.223*baseline + 11.0*ln(UVB_dose) + 5.5*ln(%BSA)` (R2=0.47).
   The model captures the baseline direction via feedback; a full BSA term is
   not incorporated.
3. **SC thickening calibration [C8]**: SC(t) is active with fitted alpha_SC=0.12
   and FST-stratified SC_max (de Boer 2025: +23% FP I-III, +36% FP IV-VI).
   k_damage=0.0 is a known limitation — the week-5 SC dip in de Boer 2025 is
   not modeled. Individual-level fitting still needed.
4. **Individual variability / M_baseline [C7]**: Datta 2019 (P13) reports a
   2.9-139 nmol/L response range under identical UVB with <3% variance explained
   by pigmentation. M_baseline_by_FST is anchored to ITA_Unexposed means from
   trial CSVs (N=44) as group-level priors; genetics and baseline serum D
   dominate individual outcomes.

See `RESULTS.md` for model-run interpretation, `OPEN_QUESTIONS.md` for the
dataset provenance cards and the full list of open gaps, `KNOWN_ISSUES.md` for
current defects and caveats, and `PAPER_SUMMARIES.md` for per-paper evidence
cards.
