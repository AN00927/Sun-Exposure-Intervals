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

The model now uses exact analytical step updates for the D and M states
between UV pulses, with 0.5-day sampling for smooth trajectories. Melanin
pulse updates use a stable exponential saturation form, and the tanning rise
rate is dose-dependent through a Hill-style alpha(UV) term.

## Files

- **`vitd_model.py`** — Core two-state analytical model (D and M).
  All parameters carry literature citations in the module docstring.
- **`vitd_validation.py`** — Six checks: three sanity (numerical correctness)
  and three literature validation (Matsuoka, Bogh, Farrar).
- **`vitd_analyses.py`** — Five analyses: alpha sweep, K_ATTEN sweep,
  beta sweep, two-framing schedule comparison, example trajectories.
- **`run_all.py`** — Runs validation + all analyses, writes `RESULTS.md`.
- **`RESULTS.md`** — Auto-generated summary with figures and interpretation.
- **`figs/`** — PNG plots from each analysis.
- **`tables/`** — CSV outputs of every parameter sweep.

## How to run

```bash
pip install numpy scipy pandas matplotlib
python3 run_all.py
```

## Current calibration

- Effective serum 25(OH)D decay half-life is now 80 days:
  `k_decay_D = ln(2) / 80 ≈ 0.00866 day^-1`, reflecting systemic tissue and
  adipose buffering rather than plasma-only decay.
- The synthesis coefficient is recalibrated to `A = 1.6875`, preserving the
  prior steady-state level under baseline conditions after the longer D
  half-life (`D_ss ∝ A / k_decay_D`).
- Baseline melanin by Fitzpatrick type is now `1:0.10, 2:0.20, 3:0.30,
  4:0.38, 5:0.46, 6:0.60`, matching spectrophotometer reflectance data from
  Ravnbak 2008 / Datta 2019.
- The Matsuoka 1990 tanned-state validation uses a 2.5 SED preconditioning
  protocol every 2 days for 4 weeks, representing a realistic summer tanning
  exposure rather than the prior 0.75 SED protocol.

## Headline results

**Validation**: 6 / 6 checks pass. The model reproduces the 80-day effective
D half-life, a recalibrated 1.17 nmol/L rise from 1 SED in untanned FST III,
Matsuoka 1990 tanned-state attenuation (ratio 0.67 after 2.5 SED
preconditioning), qualitative Bogh 2012 rise, and Farrar 2013
sufficiency-failure for FST V.

**Alpha sweep**: The optimal interval (21 days under efficiency framing,
1 day under absolute framing) is **robust** across alpha = 0.01 to 0.40
for both FST V and VI. The handoff flagged alpha as the largest scheduling
uncertainty; this sweep shows it is not.

**Beta sweep**: The 21-day efficiency optimum and daily absolute-D optimum
remain robust across melanin decay half-lives of 21-120 days. Beta changes
absolute D modestly, but not interval ranking.

**K_ATTEN sweep**: Same robustness. K_ATTEN moves the magnitude of
steady-state D from ~27 to ~110 nmol/L across K = 8.0 to 0.2 under daily
dosing, but does not change interval ranking. Calibrating K_ATTEN
accurately matters for absolute-level predictions, not for scheduling
conclusions.

**Two-framing finding**: Under matched per-session dose (1 SED/session),
- shortest interval (daily) wins absolute D (FST V: ~67, FST VI: ~69 nmol/L)
- longest interval (every 21 days) wins efficiency (~1.16 vs ~1.00-1.05 nmol/L per SED)

This is the central finding for clinical relevance: UV is the constrained
resource (skin cancer risk), so the efficiency framing matters. But
sufficiency targets (50 nmol/L) are only achievable through dose density.
Trade-off, not optimum.

## Open questions

1. K_ATTEN calibration for FST V-VI. Young 2020 implies it should be lower
   than the default 2.0; if so, sufficiency is more achievable than the
   current model predicts.
2. NB-UVB vs broadband UV conversion. The model is calibrated to broadband
   per Webb 2018, so direct comparison to phototherapy studies (Bogh,
   Karppinen) underpredicts in absolute terms.
3. Baseline-dependent dose-response. Webb 2018 notes the response is
   steepest at low starting D. The model doesn't yet incorporate this.
4. Constitutive FST effects. The current attenuation term only applies to
   tanning above baseline melanin; it does not yet model the smaller
   constitutive baseline synthesis difference between light and dark skin.

See `RESULTS.md` for full interpretation.
