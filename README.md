# Vitamin D Scheduling Optimization Model

Computational model to determine optimal sun exposure schedules for maximizing
serum vitamin D (25(OH)D) in people with darker skin (Fitzpatrick types V-VI),
accounting for photoadaptation (tanning) dynamics.

## What this is

Most existing models treat skin pigmentation as a static parameter and only
look at single-session synthesis. None update synthesis efficiency based on
recent UV history. This model adds a dynamic photoadaptation term: repeated
UV exposure raises effective melanin, which progressively reduces vitamin D
synthesis from subsequent sessions. The model then optimizes the
inter-session interval to maximize either absolute serum D or efficiency per
unit UV dose.

The model now uses exact analytical step updates for the D, M, and SC states
between UV pulses, with 0.5-day sampling for smooth trajectories. Melanin and
stratum-corneum pulse updates use stable exponential saturation forms, and the
tanning rise rate is dose-dependent through a Hill-style alpha(UV) term.

## Files

- **`vitd_model.py`** — Core three-state analytical model (D, M, and SC).
  All parameters carry literature citations in the module docstring.
- **`vitd_validation.py`** — Six checks: three sanity (numerical correctness)
  and three literature validation (Matsuoka, Bogh, Farrar).
- **`vitd_analyses.py`** — Five analyses: alpha sweep, K_ATTEN sweep,
  beta sweep, two-framing schedule comparison, example trajectories.
- **`run_all.py`** — Runs validation + all analyses, writes `RESULTS.md`.
- **`RESULTS.md`** — Auto-generated summary with figures and interpretation.
- **`SCIENTIFIC_AUDIT_HANDOFF.md`** — Dataset/paper synthesis and recommended
  future fixes, generated from local CSV/PDF analysis artifacts.
- **`figs/`** — PNG plots from each analysis.
- **`tables/`** — CSV outputs of every parameter sweep.
- **`audit_outputs/`** — Generated dataset profiles, extracted PDF text, and
  computed summary tables used by `SCIENTIFIC_AUDIT_HANDOFF.md`.

## How to run

```bash
pip install numpy scipy pandas matplotlib
python3 run_all.py
```

## Current calibration

- Effective serum 25(OH)D decay half-life is now 80 days:
  `k_decay_D = ln(2) / 80 ≈ 0.00866 day^-1`, reflecting systemic tissue and
  adipose buffering rather than plasma-only decay. Datta 2017 is the direct
  post-UVB citation to verify for this value.
- The synthesis coefficient is currently `A = 2.353`, preserving the
  prior ~1.17 nmol/L rise from 1 SED in untanned FST III after adding the
  constitutive baseline factor (`D_ss ∝ A / k_decay_D` no longer the only
  calibration constraint).
- `K_const = 0.50` adds a constitutive baseline synthesis penalty using
  `exp(-K_const * M_baseline(F))`.
- Baseline melanin by Fitzpatrick type is now `1:0.10, 2:0.20, 3:0.30,
  4:0.38, 5:0.65, 6:0.80`, matching the updated Datta 2019-style reflectance
  mapping.
- Photoadaptation rise rates are FST-stratified from de Boer 2025:
  `alpha_by_FST = 0.0025` for FP I-III and `0.002` for FP IV-VI, with
  `alpha = 0.0025` retained as the fallback default.
- Stratum-corneum thickening is modeled as `SC(t)` with 28-day half-life,
  `alpha_SC = 0.005`, `SC_max = 0.50`, and attenuation coefficient `K_SC = 1.0`.
  The July 2026 audit found that this state is structurally present but still
  needs FST/source-specific calibration.
- The Matsuoka 1990 tanned-state validation uses a 2.5 SED preconditioning
  protocol every 2 days for 4 weeks, representing a realistic summer tanning
  exposure rather than the prior 0.75 SED protocol. The local PDF named
  `Matsuoka et al., 1990.pdf` is a different sunscreen/body-region paper, so
  the tanned-state validation still needs the correct full-text source.

## Headline results

**Validation**: 6 / 7 checks pass (3 sanity + 4 literature). The model reproduces
the 80-day effective D half-life, a recalibrated 1.17 nmol/L rise from 1 SED in
untanned FST III, Matsuoka 1990 tanned-state attenuation (ratio ~0.67 after 2.5 SED
preconditioning), Farrar 2013 FST V quantitative window, and baseline-dependent
synthesis direction (Bogh 2010/P5).

**Bogh 2012 now FAILS under the new quantitative threshold** — the model predicts
delta +30.8 nmol/L vs the observed +55.8 nmol/L (gap ~25 nmol/L). This FAIL is
scientifically correct: the old qualitative `delta > 5` criterion was hiding a real
model underprediction. The gap must be closed by per-study spectral calibration
(E_source tuning) before using this model for absolute NB-UVB predictions.

> **July 2026 audit change [C1/C2]**: Bogh 2012 and Farrar 2013 thresholds are
> now QUANTITATIVE. Bogh 2012 requires `|delta - 55.8| ≤ 15 nmol/L` (not just
> `delta > 5`). Farrar 2013 requires D_final in [25, 55] nmol/L and delta in
> [10, 35] nmol/L (not just `D_final < 68`).

> **[C6] Matsuoka citation warning**: The local `Matsuoka et al., 1990.pdf` is a
> sunscreen/body-region paper, NOT the tanned-vs-untanned attenuation paper. The
> Matsuoka tanned-state validation uses a synthetic preconditioning protocol and
> the <0.70 criterion has no confirmed local source. See `vitd_validation.py`.

**Alpha sweep**: The optimal interval (21 days under efficiency framing,
1 day under absolute framing) is **robust** across alpha = 0.01 to 0.40
for both FST V and VI. The handoff flagged alpha as the largest scheduling
uncertainty; this sweep shows it is not.

**Beta sweep**: The 21-day efficiency optimum and daily absolute-D optimum
remain robust across melanin decay half-lives of 21-120 days. Beta changes
absolute D modestly, but not interval ranking.

**K_ATTEN sweep**: Same robustness. K_ATTEN moves the magnitude of
steady-state D from ~47.5 to ~60.5 nmol/L across K = 8.0 to 0.2 under daily
dosing, but does not change interval ranking. Calibrating K_ATTEN
accurately matters for absolute-level predictions, not for scheduling
conclusions.

**Two-framing finding**: Under matched per-session dose (1 SED/session),
- shortest interval (daily) wins absolute D (FST V: ~57, FST VI: ~54 nmol/L)
- longest interval (every 21 days) wins efficiency (~0.90 vs ~0.82 nmol/L per SED)

This is the central finding for clinical relevance: UV is the constrained
resource (skin cancer risk), so the efficiency framing matters. But
sufficiency targets (50 nmol/L) are only achievable through dose density.
Trade-off, not optimum.

## Open questions

1. **K_ATTEN calibration [C5]**: Young 2020 (P9) shows the FST II:VI vitamin D
   slope ratio is only about 1.3-1.4 (Arimed B 1.29, TL01 1.43). K_ATTEN=2.0
   is a placeholder; a large multi-fold melanin-only penalty is unsupported by
   current evidence. Do NOT present K_ATTEN=2.0 as a measured parameter.
2. **Source / spectral calibration [C3]**: Young 2020 (P20) shows the standard
   CIE previtamin-D3 action spectrum needs a 5 nm blue shift. Osmancevic 2009
   (P18) shows broadband UVB can outperform NB-UVB per SED in psoriasis.
   `E_source` must become a per-study calibration input, not a fixed multiplier.
3. **Baseline-dependent dose-response [C4]**: Bogh 2011/P5 derives
   `delta = 7.1 - 0.223*baseline + 11.0*ln(UVB_dose) + 5.5*ln(%BSA)` (R2=0.47).
   The model captures the baseline direction via feedback; a full BSA term is
   not yet incorporated.
4. **SC thickening calibration [C8]**: SC(t) is active. De Boer 2025 supports
   +23% (FP I-III) and +36% (FP IV-VI) SC thickening; current uniform SC
   parameters are not FST-stratified and need individual-level data fitting.
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
