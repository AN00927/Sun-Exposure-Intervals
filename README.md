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
- **`vitd_validation.py`** — Nine checks: three sanity guards, five
  literature/model validation checks (de Boer 2025, Bogh, Farrar,
  Bogh baseline/BSA, Karppinen 2017), and one metric-ordering check.
- **`vitd_analyses.py`** — Five analyses: alpha sweep, K_ATTEN sweep,
  beta sweep, two-framing schedule comparison, example trajectories.
- **`run_all.py`** — Runs validation + all analyses, writes `RESULTS.md`.
- **`RESULTS.md`** — Auto-generated summary with figures and interpretation.
- **`SCIENTIFIC_AUDIT_HANDOFF.md`** — Dataset/paper synthesis and recommended
  future fixes, generated from local CSV/PDF analysis artifacts.
- **`figs/`** — PNG plots from each analysis.
- **`tables/`** — CSV outputs of every parameter sweep.
- **`audit_outputs/`** — Generated dataset profiles, PDF inventories,
  compact extraction snippets, and computed summary tables used by
  `SCIENTIFIC_AUDIT_HANDOFF.md`.

## How to run

```bash
pip install numpy scipy pandas matplotlib
python3 run_all.py
```

## Current calibration

- **D decay is state-dependent.** Half-life follows a power law fit to Datta
  2017: `t_half(D) = 2953 * D^(-0.72)`. At high D (~132 nmol/L), half-life is
  ~89 days. At low D (~43 nmol/L), half-life is ~199 days. Body preserves D
  when levels are low and clears it faster when levels are high.
- **Synthesis coefficient** `A = 2.353`. With current K_const and M_baseline,
  1 SED in untanned FST III at D=25 produces ~1.27 nmol/L rise.
- **K_const = 0.33** sets the constitutive melanin penalty so that the FST
  II:VI synthesis ratio matches Young 2020 (~1.36×).
- **M_baseline_by_FST** anchored to spectrometer measurements from trial
  participants (ITA angle from ArimedB CSVs, N=48):
  `1:0.0, 2:0.07, 3:0.21, 4:0.44, 5:0.78, 6:1.00`.
- **K_ATTEN = 0.37** (was 2.0). Young 2020 shows melanin has a small effect
  on D synthesis. The old 2.0 was a coarse prior with no paper support.
- **Matsuoka test replaced** with de Boer 2025 photoadaptation check (the
  local Matsuoka PDF was a sunscreen paper, not the tanned-vs-untanned paper).
- All other parameters unchanged: `alpha_by_FST`, `SC`, `beta`, `E_source`,
  `D_max`, `BSA`. See `vitd_model.py` docstring for full citations.

## Headline results

**Validation**: 8/8 required + 1/1 warning checks pass. Key results:

- **Bogh 2012**: model delta +41.6 vs observed +55.8 (within ±15 tolerance). The
  ~14 nmol/L gap is expected — the study used FST I-III Danes, lighter than the
  model's FST III default.
- **Farrar 2013 FST V**: D_final and delta are in a plausible window.
- **Bogh baseline/BSA**: direction check passes.
- **de Boer 2025 photoadaptation**: melanin and SC respond in the right direction.
- **Sanity checks**: single-bolus, ordering, and photoadaptation reduction all pass.

**Alpha sweep**: The optimal interval (21 days under efficiency framing,
1 day under absolute framing) holds across alpha = 0.01 to 0.40
for both FST V and VI. The handoff flagged alpha as the largest scheduling
uncertainty; this sweep shows otherwise.

**Beta sweep**: The 21-day efficiency optimum and daily absolute-D optimum
remain robust across melanin decay half-lives of 21-120 days. Beta shifts
absolute D but does not change interval ranking.

**K_ATTEN sweep**: Same robustness. K_ATTEN=0.37 (Young 2020-backed) is the
default. Steady-state D is ~67 nmol/L under daily dosing (FST V). Interval
ranking does not change with K_ATTEN.

**Two-framing finding**: Under matched per-session dose (1 SED/session),
- shortest interval (daily) wins absolute D (FST V: ~72, FST VI: ~69 nmol/L)
- longest interval (every 21 days) wins gross efficiency (~1.05 vs ~0.79 nmol/L per SED)

UV is a constrained resource (skin cancer risk), so the efficiency framing
carries more clinical weight. But sufficiency targets (50 nmol/L) require dose
density. You face a trade-off, not a single optimum.

## Open questions

1. **Source / spectral calibration [C3]**: Young 2020 (P20) shows the standard
   CIE previtamin-D3 action spectrum needs a 5 nm blue shift. Osmancevic 2009
   (P18) shows broadband UVB can outperform NB-UVB per SED in psoriasis.
   `E_source` needs to be a per-study calibration input, not a fixed multiplier.
2. **Baseline-dependent dose-response [C4]**: Bogh 2011/P5 derives
   `delta = 7.1 - 0.223*baseline + 11.0*ln(UVB_dose) + 5.5*ln(%BSA)` (R2=0.47).
   The model captures the baseline direction via feedback; a full BSA term is
   not incorporated.
3. **SC thickening calibration [C8]**: SC(t) is active. De Boer 2025 supports
   +23% (FP I-III) and +36% (FP IV-VI) SC thickening; current uniform SC
   parameters are not FST-stratified and require individual-level data fitting.
4. **Individual variability / M_baseline [C7]**: Datta 2019 (P13) reports a
   2.9-139 nmol/L response range under identical UVB with <3% variance explained
   by pigmentation. M_baseline_by_FST is a group-level prior only; genetics and
   baseline serum D dominate individual outcomes.

See `RESULTS.md` for model-run interpretation and
`SCIENTIFIC_AUDIT_HANDOFF.md` for the dataset/literature audit.
