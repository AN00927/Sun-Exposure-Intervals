# Vitamin D Scheduling Optimization Model

> **Health-safety notice.** This model answers two questions with very
> different reliability:
>
> - **"How often should I go outside?"** (the interval) -- reliable at
>   moderate-to-typical daily doses. Interval rankings depend on the *shape*
>   of the dose-response curve (absolute SED, not fractional MED), not its
>   scale, and survive roughly a 7x error in the absolute dose -- **at
>   budgets of roughly 0.5 SED/day or more.** At budgets at or below
>   0.25 SED/day the ranking is *not* protected by the same evidence: the
>   published per-session dose-response curvature straddles linearity rather
>   than confirming it, and the flip point falls at or below the cleanest
>   single measured exponent (Shih 2018). See `tests/test_concavity_assumption.py`
>   and `run_all.py`'s "## 17. Interpretation" section for the current margins.
> - **"How many minutes?"** -- not reliable yet. An older ~7.57x band
>   was withdrawn (decision B5, 2026-09-23) when it could not be reproduced
>   from the current `vitd_scenario.py` constants. The only range
>   `vitd_scenario.py` itself varies is exposed body-surface area (10%-85%),
>   about a **2.34x** minutes ratio at FST III at the settled `K_const`/`A`
>   point. Read the current band from `RESULTS.md` "Scope and reference
>   conditions" after a full `run_all.py` run. Treat it as a floor.
>
> **Every validation check against real, unsimulated sunlight either fails
> outright or passes on a band wide enough that it could hardly fail.** Every
> check that passes *quantitatively* uses a lamp or solar simulator, not
> sunlight.
>
> **Treat the interval it recommends as the useful output. Do not treat any
> absolute-minutes number as a dosing instruction.**
>
> See the `vitd_model.py` module docstring for the full derivation.

## Scope and limitations

**As of 2026-08-21, this repository does not model location, date, weather, or
posture.** A scope audit on that date found the location-based
dosimetry chain — solar position -> ambient UV index -> posture/site geometry
-> exposed BSA — stacked four independent uncertainties into one number that
had never been validated as a product, even though its individual links were
mostly sound. That chain was retired to `exploratory/` (still present, still
tested, no longer wired into any headline number; see `exploratory/README.md`).

In its place, every schedule in the core pipeline is evaluated under one
**frozen reference scenario** (`vitd_scenario.py`) — fixed ambient UV index,
fixed geometry factor, fixed exposed body-surface area, held constant on every
exposure day. The values were chosen so no parameter needed recalibrating: each
was already the reference condition the model was calibrated at.

`vitd_scenario.LIMITATION_STATEMENT`, verbatim:

> All results assume a fixed ambient UV index of 6, a horizontal-to-personal
> geometry factor of 0.35, and 35% exposed body surface area (t-shirt and
> shorts with partial coverage (face, neck, forearms, hands and part of the
> legs bare)), held constant on every exposure day. Real ambient UV varies with
> latitude, season, time of day, altitude, cloud and ozone, and real personal
> dose varies with posture and clothing. We do not model that variation. Every
> duration reported here should therefore be read as a duration under these
> reference conditions, not as a calendar recommendation for a specific place
> or date.

Two consequences worth stating plainly:

- The model **cannot** tell you how long to stand outside in a specific city on
  a specific date. It answers "how long under reference conditions."
- Two new reference points narrow what "the model" means in any given claim:
  `basics/melanin_only.py` and `basics/sc_only.py` isolate one mechanism each
  against one dataset each, and `spacing_study/` is a self-contained sub-study
  on whether daily exposure is actually optimal for vitamin D synthesis (see
  `spacing_study/results/FINDINGS.md`). Neither is fitted to anything, and
  neither replaces the full model.

## General Information

- **Title:** Vitamin D Scheduling Optimization Model
- **Authors:**
  - Ahaan Nigam (independent researcher) -- ahaannigam27@gmail.com, corresponding
  - Rabira Tusi (Carnegie Mellon University) -- rtusi@cmu.edu
  - Pulkit Grover (Carnegie Mellon University) -- pulkitgrover@cmu.edu
  - Rachita Mathur (Jaipur, Rajasthan, India)
- **Date of model development:** 2026 (ongoing)
- **Keywords:** vitamin D, UV exposure scheduling, photoadaptation, melanin,
  stratum corneum, serum 25(OH)D, dose-response, ODE model
- **Language:** English
- **Funding:** [to be filled]
- **Related documentation:**
  - `vitd_model.py` module docstring -- canonical parameter values, citations
    and assumptions (see `CLAUDE.md`, "Where facts live")
  - `calibration_approval.yaml` -- the approval record for every calibration
    decision, with its rationale
  - `RESULTS.md` -- auto-generated validation and analysis summary (regenerated
    by `run_all.py`; do not hand-edit)
  - `paper/README.md` -- manuscript status, what is drafted versus stubbed
  - `CLAUDE.md` -- repository conventions and agent guidelines
  - `docs/MELANIN_MODEL.md` -- the melanin sub-model: equations, per-parameter
    provenance, limitations
  - `docs/SC_MODEL.md` -- the stratum corneum sub-model, same treatment
  - `docs/SC_MELANIN_OPEN_ITEMS.md` -- the open items from those two, and how
    each was resolved
  - `exploratory/README.md` -- what was retired, why, and how to un-retire it
  - `basics/README.md` -- the two Tier-1 single-mechanism reference models
  - `spacing_study/README.md` -- the daily-vs-spaced-exposure sub-study

## Data and File Overview

### Source data

| Path | Description | Format |
|------|-------------|--------|
| `data/spectra/tl01_spectrum.csv` | TL-01 lamp relative spectral irradiance | CSV (wavelength_nm, weight) |
| `data/spectra/cie_erythema_action_spectrum.csv` | CIE 1987 erythema action spectrum | CSV (wavelength_nm, weight) |
| `data/spectra/vitd_action_spectrum_cie.csv` | CIE-174 previtamin-D3 action spectrum | CSV (wavelength_nm, weight) |
| `data/spectra/vitd_action_spectrum_young2021.csv` | Young 2021 blue-shifted action spectrum (5 nm shift of CIE-174) | CSV (wavelength_nm, weight) |
| `data/spectra/vitd_action_spectrum_young2021_MEASURED.csv` | Same values with full provenance in header | CSV (wavelength_nm, weight) |
| `data/vernez2015_table_s1_vis_by_site_posture.csv` | Vernez 2015 UV index by body site and posture | CSV |

See `data/spectra/README.md` for detailed provenance and caveats on the action
spectra.

### Generated outputs

| Path | Description | Regenerated by |
|------|-------------|----------------|
| `RESULTS.md` | Validation suite output, parameter sweeps, interpretation | `run_all.py` |
| `tables/*.csv` | 20 analysis tables (sweeps, optimisations, schedule comparisons) | `run_all.py` |
| `figs/*.png` | 17 figures (trajectories, sweeps, dose-response, skin-type breakdown) | `run_all.py` |
| `audit_outputs/` | Extracted datasets, summaries, provenance artifacts | Various scripts |

### Repository layout

| Path | Purpose |
|------|---------|
| `vitd_model.py` | Model state, parameters, Young 2021 action spectrum calibration, canonical parameter values |
| `vitd_scenario.py` | The frozen reference scenario (fixed ambient UV index, geometry factor, exposed BSA) that every schedule in the core pipeline is evaluated under -- single source of exposure-scenario constants |
| `vitd_calibration.py` | Calibration suite (Matsuoka 1990, Young 2020 per-person ratio, De Boer 2025 cross-validation) |
| `vitd_objectives.py` | Decision problems and schedule objectives |
| `vitd_protocols.py` | Named protocol helpers for study-specific simulations |
| `vitd_validation.py` | Validation suite wiring |
| `vitd_physiology.py` | Body-weight dilution, photodegradation, hydroxylation efficiency, cancer risk |
| `vitd_sensitivity.py` | Parameter sensitivity analysis |
| `vitd_optimization.py` | Schedule optimisation routines |
| `vitd_dose_conversion.py` | Dose unit conversions (SED, MED, J/cm^2) |
| `vitd_analyses.py` | Analysis helpers |
| `vitd_benchmark_webbkift.py` | Webb/Kift benchmark |
| `run_all.py` | Full pipeline entry point |
| `scripts/` | Standalone analyses, parameter fitting, diagnostics |
| `tests/` | Pytest test suite for the core pipeline |
| `tests/exploratory/` | Test suite for `exploratory/`, with its own `conftest.py` that puts `exploratory/` on `sys.path` -- the core suite must pass without it (see `CLAUDE.md`) |
| `paper/` | Elsevier elsarticle manuscript skeleton |
| `exploratory/` | Retired location/genetics/behaviour/NHANES modules -- still import, still tested, no longer wired into any headline number. See `exploratory/README.md` |
| `basics/` | Two Tier-1 single-mechanism reference models (`melanin_only.py`, `sc_only.py`). See `basics/README.md` |
| `spacing_study/` | Self-contained sub-study on exposure spacing. See `spacing_study/README.md` |
| `docs/MELANIN_MODEL.md` | The melanin sub-model: equations, per-parameter provenance, limitations |
| `docs/SC_MODEL.md` | The stratum corneum sub-model, same treatment |
| `docs/SC_MELANIN_OPEN_ITEMS.md` | Open items from those two documents and how each was resolved |
| `docs/INDEX.md` | Navigation root: every tracked Markdown document, what it answers, and its status |

## Methodological Information

### Model description

A three-state ODE model of serum 25(OH)D dynamics under UV exposure schedules,
accounting for photoadaptation (melanin upregulation and stratum corneum
thickening) in darker skin types.

**State variables:**

| Variable | Description | Units |
|----------|-------------|-------|
| D(t) | Effective serum 25(OH)D | nmol/L |
| M(t) | Effective melanin index (Datta 2019 reflectance scale) | dimensionless 0--1 |
| SC(t) | Additional stratum corneum thickness (above baseline 0) | dimensionless |

**Governing equations:**

```
dD/dt  = S(UV, M, SC, D, F, bsa)  -  k_decay_D(D) * D
dM/dt  = alpha * UV * (M_max - M)  -  beta * (M - M_baseline(F))
dSC/dt = alpha_SC * UV * (SC_max - SC)  -  k_decay_SC * SC  -  k_damage * UV * SC
```

The synthesis function S uses a logarithmic dose-response (Webb 2021 form) with
melanin and stratum corneum attenuation, BSA scaling, and homeostatic feedback
inhibition. Full equation and parameter derivations are in the `vitd_model.py`
module docstring, which is the single canonical source for all parameter
values (see `CLAUDE.md`).

### Calibration

The model is calibrated against:

- **Matsuoka 1990:** Within-subject tanning attenuation (direction check; `vitd_calibration.py`)
- **Young 2020:** Per-person FST II:VI synthesis ratio (n=23, TL-01 arm)
- **de Boer 2025:** Stratum corneum thickening cross-validation
- **Ala-Houhala:** Finland reversal check

Calibration changes require explicit user approval via `calibration_approval.yaml`.
Bug fixes (unit errors, state-update errors, inverted signs) do not.

### Validation status

**9 / 10 required checks passed, plus 4 / 4 warning-only**, across a
**14**-check registry (`vitd_validation.CHECK_REGISTRY`). Measured on the
`run_all.py` run of 2026-09-21; see `RESULTS.md` for the current count --
do not quote without regenerating.

The required count is real but flatters the model. The same 14 checks split
by what they are *evidence of*: INTERNAL 4/4 (self-consistency and wiring, not
biology), DIRECTIONAL 3/3 (external data, direction or ordering only),
QUANTITATIVE 1/2 (external data, a real numeric comparison), CIRCULAR 3/3
(external data the model or one of its parameters was fitted to -- a pass is
expected by construction). **One quantitative external check out of two is the
honest headline.**

Validation uses:
- **Literature benchmarks:** Bogh 2012 (NB-UVB), Farrar 2013 (FST V),
  Bogh 2010 (baseline feedback), Karppinen 2017 (calibration target),
  Kift 2013 (UK South Asian naturalistic)
- **Held-out benchmarks:** Kallioglu 2024 (skin-type gradient, warning-only),
  Gallagher 2012 (oral plateau, warning-only)

**NHANES is not in `vitd_validation.CHECK_REGISTRY`.** The population-level
NHANES checks that used to run there (including a "Part B" summer
steady-state check) were retired; there is no NHANES entry in the 14-check
registry above at all. Population-level NHANES plausibility is instead
covered by the `RECURRING`/`PIPELINE` macro diagnostic suite in `scripts/`
(`nhanes_macro_validation.py`, `nhanes_dietary_partitioning.py`,
`nhanes_weight_dilution_check.py` -- see `CLAUDE.md` "Agent guidelines" and
`docs/SCRIPTS.md`), which is separate from the required/warning-only tally
above and is not itself pass/fail against a fixed band. Do not quote a
specific pass count for the old "Part B" check; it no longer exists to be
counted.

### Quick Start

```bash
pip install -r requirements.txt
PYTHONIOENCODING=utf-8 python3 run_all.py
python3 -m pytest
```

On Windows, `PYTHONIOENCODING=utf-8` matters because generated reports use
Unicode characters. `run_all.py` takes ~10 minutes (dominated by
differential-evolution optimisations). Do not run it just to refresh results.

### Key scripts

- **Fitting K_ATTEN:** `python scripts/fit_katten_young2020.py`
- **NHANES Weighted Analysis:** `python scripts/nhanes_weighted_check.py`
  (auto-downloads missing CDC XPT files)
- **Calibration Suite:** `python -m vitd_calibration`
- **Parameter Optima:** `python scripts/parameter_optima.py`
- **Metric Redundancy:** `python scripts/metric_redundancy.py`
- **Melanin-SC Coupling:** `python scripts/melanin_sc_coupling.py`

### Dose framing

The model supports four dose framings (labelled in all scenario outputs):

| Framing | Description |
|---------|-------------|
| Fixed absolute SED | Each session delivers the same SED regardless of MED |
| Fixed cumulative SED | Total UV budget split across sessions |
| Fixed per-session SED | 1 SED/session; cumulative varies with frequency |
| Fixed fractional MED | Each session is N% of the individual MED |

Shih/Farrar/Rhodes 2018 shows fractional-MED dosing removes the apparent
skin-type effect for acute 25(OH)D gain. Fixed absolute SED can leave South
Asian/FST V participants below sufficiency. **Do not treat these framings as
interchangeable.**

### Quality assurance

- Validation suite: `python -m pytest` or `run_all.py`
- Audit change tags (C1--C10) track scientific modifications
- `calibration_approval.yaml` gates all calibration changes
- Parameter values live in one place: `vitd_model.py` module docstring

## Data-specific Information

### Model parameters (summary)

Full parameter values, citations, and evidence status are in the `vitd_model.py`
module docstring. **Do not copy values into this file** -- link to the code.

| Parameter | Description | Units | Source |
|-----------|-------------|-------|--------|
| A | Synthesis scaling factor | nmol/L per ln(SED) | Calibrated to 1.27 nmol/L per 1 SED at FST III |
| K_ATTEN | Melanin attenuation coefficient | dimensionless | Young 2020 per-person fit (n=23) |
| K_const | Baseline inhibition coefficient | dimensionless | Young 2020 FST II:VI ratio |
| K_SC | Stratum corneum attenuation coefficient | dimensionless | Uncalibrated prior |
| D_max | Homeostatic feedback ceiling | nmol/L | Origin unconfirmed (see paper/README.md) |
| SC_max | SC thickening ceiling | dimensionless | Prior 0.20--0.40 (de Boer 2025) |
| alpha_SC | SC induction rate | dimensionless | Uncalibrated prior |
| k_decay_SC | SC regression rate | 1/days | ~28 day half-life |
| k_damage | UV-driven SC damage rate | 1/SED | 0.0 (structurally unidentifiable) |
| alpha | Melanin induction rate | dimensionless | Miyamura 2011 (light skin) |
| beta | Melanin decay rate | 1/days | Ravnbak 2008 |
| UV_halfmax_by_FST | Hill constant for melanin induction | SED | Ravnbak 2008 Table IV |
| M_baseline_by_FST | Constitutive melanin index by FST | dimensionless | ITA_Unexposed means (ArimedB irradiated + ARIMED controls pooled, n=48) |

### UV source calibration

Per-source D3-effective dose weighting factors (`E_source`):

| Source | E_source (mean) | Range | Reference |
|--------|----------------|-------|-----------|
| Solar summer (default) | 1.89 | 1.79--2.00 | Webb 2021 (Manchester) |
| Solar summer NZ | 2.01 | -- | Webb 2021 |
| Solar winter NZ | 1.22 | -- | Webb 2021 (lowest sun value) |
| TL-01 NB-UVB | 2.44 | 2.24--2.63 | Webb 2021 / Grigalavicius 2015 |
| TL-12 BB-UVB | 1.24 | 1.23--1.25 | Webb 2021 / Grigalavicius 2015 |
| UV-6 broadband | 1.54 | 1.54--1.55 | Webb 2021 / Grigalavicius 2015 |
| Arimed B | 1.83 | -- | Webb 2021 |
| Sunbed UVB | 1.18 | -- | Grigalavicius 2015 |

**E_source must be a per-study calibration input, NOT a fixed multiplier.**
Use values from `SOURCE_CALIBRATION_TABLE` in `vitd_model.py`.

### CSV file format

All spectra files use:

```csv
wavelength_nm,weight
280,0.0
281,0.0
```

- For lamp spectra: `weight` = relative spectral irradiance (normalised internally)
- For action spectra: `weight` = dimensionless biological weighting (source and
  normalization documented in file header)

### Missing data

If a spectrum is missing, scientific dose conversion must be marked
`BLOCKED_MISSING_SPECTRUM`. Do not fabricate spectrum values.

### Abbreviations and codes

| Term | Meaning |
|------|---------|
| SED | Standard erythemal dose (1 SED = 100 J/m^2 erythemal UV) |
| MED | Minimal erythemal dose |
| FST I--VI | Fitzpatrick skin type (I = most fair, VI = most dark) |
| BSA | Body surface area (fraction exposed to UV) |
| D | Serum 25(OH)D concentration (nmol/L) |
| M | Melanin index (dimensionless, Datta 2019 scale) |
| SC | Stratum corneum additional thickness (dimensionless) |
| VDE | Vitamin D-effective (spectral weighting) |
| E_source | Source-specific D3-effective dose weighting factor |
| K_ATTEN | Melanin attenuation coefficient |
| NB-UVB | Narrowband UVB (peak ~311 nm) |
| BB-UVB | Broadband UVB |
| NHANES | National Health and Nutrition Examination Survey |
| CIE | International Commission on Illumination |
| ITA | Individual typology angle (skin colour metric) |

### Audit change tags

Tags used throughout the codebase to track scientific modifications:

| Tag | Meaning |
|-----|---------|
| C1 | exposed_bsa / clothing_cover support |
| C2 | SOURCE_CALIBRATION_TABLE replaces ad hoc E_source scalar |
| C3 | E_source must be per-study calibration input |
| C4 | Dose framing labels added to simulate() output |
| C5 | K_ATTEN small-penalty evidence caveat |
| C7 | SC_max reparameterised as 0.20--0.40 prior |
| C8 | SC state active, calibration pending |
| C9 | Cholesterol/7-DHC added to future covariates |
| C10-PRIOR | K_ATTEN and M_baseline_by_FST tagged as coarse priors |
