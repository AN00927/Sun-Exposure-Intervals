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

## Files

- **`vitd_model.py`** — Core two-state ODE model (D and M).
  All parameters carry literature citations in the module docstring.
- **`vitd_validation.py`** — Six checks: three sanity (numerical correctness)
  and three literature validation (Matsuoka, Bogh, Farrar).
- **`vitd_analyses.py`** — Four analyses: alpha sweep, K_ATTEN sweep,
  two-framing schedule comparison, example trajectories.
- **`run_all.py`** — Runs validation + all analyses, writes `RESULTS.md`.
- **`RESULTS.md`** — Auto-generated summary with figures and interpretation.
- **`figs/`** — PNG plots from each analysis.
- **`tables/`** — CSV outputs of every parameter sweep.

## How to run

```bash
pip install numpy scipy pandas matplotlib
python3 run_all.py
```

## Headline results

**Validation**: 6 / 6 checks pass. Model reproduces 30-day effective D
half-life, Webb 2018 light-skin per-SED yield (3.12 nmol/L for 1 SED in
untanned FST III), Matsuoka 1990 tanned-state attenuation, qualitative Bogh
2012 rise, and Farrar 2013 sufficiency-failure for FST V.

**Alpha sweep**: The optimal interval (21 days under efficiency framing,
1 day under absolute framing) is **robust** across alpha = 0.01 to 0.40
for both FST V and VI. The handoff flagged alpha as the largest scheduling
uncertainty; this sweep shows it is not.

**K_ATTEN sweep**: Same robustness. K_ATTEN moves the magnitude of
steady-state D from ~27 to ~110 nmol/L across K = 8.0 to 0.2 under daily
dosing, but does not change interval ranking. Calibrating K_ATTEN
accurately matters for absolute-level predictions, not for scheduling
conclusions.

**Two-framing finding**: Under matched per-session dose,
- shortest interval (daily) wins absolute D (FST V: ~77, FST VI: ~93 nmol/L)
- longest interval (every 21 days) wins efficiency (3.03 vs 2.13 nmol/L per SED)

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

See `RESULTS.md` for full interpretation.
