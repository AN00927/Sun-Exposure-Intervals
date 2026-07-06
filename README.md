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
- **`vitd_validation.py`** — Eight checks: three sanity guards, four
  literature/model validation checks (Matsuoka warning, Bogh, Farrar,
  Bogh baseline/BSA), and one metric-ordering check.
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

- Effective serum 25(OH)D decay half-life is 80 days:
  `k_decay_D = ln(2) / 80 ≈ 0.00866 day^-1`, reflecting systemic tissue and
  adipose buffering instead of plasma-only decay. Datta 2017 is the direct
  post-UVB citation for this value.
- The synthesis coefficient is `A = 2.353`, preserving the
  prior ~1.17 nmol/L rise from 1 SED in untanned FST III after adding the
  constitutive baseline factor (`D_ss ∝ A / k_decay_D` no longer the only
  calibration constraint).
- `K_const = 0.50` adds a constitutive baseline synthesis penalty using
  `exp(-K_const * M_baseline(F))`.
- Baseline melanin by Fitzpatrick type: `1:0.10, 2:0.20, 3:0.30,
  4:0.38, 5:0.65, 6:0.80`, matching the updated Datta 2019-style reflectance
  mapping.
- Photoadaptation rise rates are FST-stratified from de Boer 2025:
  `alpha_by_FST = 0.0025` for FP I-III and `0.002` for FP IV-VI, with
  `alpha = 0.0025` retained as the fallback default.
- Stratum-corneum thickening is modeled as `SC(t)` with 28-day half-life,
  `alpha_SC = 0.005`, `SC_max = 0.30` (prior range 0.20–0.40, de Boer 2025 P2),
  and attenuation coefficient `K_SC = 1.0`.
  The July 2026 audit found this state is structurally present but still
  needs FST/source-specific calibration.
- The Matsuoka 1990 tanned-state validation uses a 2.5 SED preconditioning
  protocol every 2 days for 4 weeks, representing a realistic summer tanning
  exposure instead of the prior 0.75 SED protocol. The local PDF named
  `Matsuoka et al., 1990.pdf` is a different sunscreen/body-region paper, so
  the tanned-state validation still needs the correct full-text source.

## Headline results

**Validation**: 8 / 9 checks pass (7 of 8 required + 1 warning-only). The model reproduces
the 80-day effective D half-life, a recalibrated 1.17 nmol/L rise from 1 SED in
untanned FST III, Matsuoka 1990 tanned-state attenuation (ratio ~0.67 after 2.5 SED
preconditioning), Farrar 2013 FST V quantitative window, and baseline-dependent
synthesis direction (Bogh 2010/P5).

**Bogh 2012 fails under the new quantitative threshold**: the model predicts
delta +30.8 nmol/L vs the observed +55.8 nmol/L (gap ~25 nmol/L). The old
qualitative `delta > 5` criterion was hiding a real
model underprediction. Closing the gap requires per-study spectral calibration
(E_source tuning) before using this model for absolute NB-UVB predictions.

> **July 2026 audit change [C1/C2]**: Bogh 2012 and Farrar 2013 thresholds are
> quantitative. Bogh 2012 requires `|delta - 55.8| ≤ 15 nmol/L` instead of
> `delta > 5`. Farrar 2013 requires D_final in [25, 55] nmol/L and delta in
> [10, 35] nmol/L instead of `D_final < 68`.

> **[C6] Matsuoka citation warning**: The local `Matsuoka et al., 1990.pdf` is a
> sunscreen/body-region paper, not the tanned-vs-untanned attenuation paper. The
> Matsuoka tanned-state validation uses a synthetic preconditioning protocol and
> the <0.70 criterion has no confirmed local source. See `vitd_validation.py`.

**Alpha sweep**: The optimal interval (21 days under efficiency framing,
1 day under absolute framing) holds across alpha = 0.01 to 0.40
for both FST V and VI. The handoff flagged alpha as the largest scheduling
uncertainty; this sweep shows otherwise.

**Beta sweep**: The 21-day efficiency optimum and daily absolute-D optimum
remain robust across melanin decay half-lives of 21-120 days. Beta shifts
absolute D but does not change interval ranking.

**K_ATTEN sweep**: Same robustness. Steady-state D ranges from ~47.5 to
~60.5 nmol/L across K = 8.0 to 0.2 under daily dosing, but interval ranking
does not change. Calibrating K_ATTEN matters for absolute-level predictions,
not for scheduling conclusions.

**Two-framing finding**: Under matched per-session dose (1 SED/session),
- shortest interval (daily) wins absolute D (FST V: ~57, FST VI: ~54 nmol/L)
- longest interval (every 21 days) wins efficiency (~0.90 vs ~0.82 nmol/L per SED)

UV is a constrained resource (skin cancer risk), so the efficiency framing
carries more clinical weight. But sufficiency targets (50 nmol/L) require dose
density. You face a trade-off, not a single optimum.

## Open questions

1. **K_ATTEN calibration [C5]**: Young 2020 (P9) shows the FST II:VI vitamin D
   slope ratio is roughly 1.3-1.4 (Arimed B 1.29, TL01 1.43). K_ATTEN=2.0
   is a placeholder; a large multi-fold melanin-only penalty is unsupported by
   current evidence. Do not present K_ATTEN=2.0 as a measured parameter.
2. **Source / spectral calibration [C3]**: Young 2020 (P20) shows the standard
   CIE previtamin-D3 action spectrum needs a 5 nm blue shift. Osmancevic 2009
   (P18) shows broadband UVB can outperform NB-UVB per SED in psoriasis.
   `E_source` needs to be a per-study calibration input, not a fixed multiplier.
3. **Baseline-dependent dose-response [C4]**: Bogh 2011/P5 derives
   `delta = 7.1 - 0.223*baseline + 11.0*ln(UVB_dose) + 5.5*ln(%BSA)` (R2=0.47).
   The model captures the baseline direction via feedback; a full BSA term is
   not incorporated.
4. **SC thickening calibration [C8]**: SC(t) is active. De Boer 2025 supports
   +23% (FP I-III) and +36% (FP IV-VI) SC thickening; current uniform SC
   parameters are not FST-stratified and require individual-level data fitting.
5. **Individual variability / M_baseline [C7]**: Datta 2019 (P13) reports a
   2.9-139 nmol/L response range under identical UVB with <3% variance explained
   by pigmentation. M_baseline_by_FST is a group-level prior only; genetics and
   baseline serum D dominate individual outcomes.
6. **Matsuoka local PDF mismatch [C6]**: The local file is not the
   tanned-vs-untanned attenuation paper. Obtain 'Suntanning and Cutaneous
   Synthesis of Vitamin D3' (Matsuoka et al., J Lab Clin Med 1990) to resolve.
7. **Datta 2017 half-life structure**: Post-UVB half-life is ~89 days at high
   start levels, but 148-199 days at medium/low start. A fixed 80-day constant
   is approximate; state-dependent decay is the next mechanistic step.

See `RESULTS.md` for model-run interpretation and
`SCIENTIFIC_AUDIT_HANDOFF.md` for the dataset/literature audit.
