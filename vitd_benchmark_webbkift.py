"""
vitd_benchmark_webbkift.py
==========================
HELD-OUT external benchmark. Nothing in this repo is calibrated to it.

Source
------
Webb, Kift, Rhodes et al., "Colour Counts: Sunlight and Skin Type as Drivers of
Vitamin D Deficiency at UK Latitudes", Nutrients 2018;10(4):457. PMC5946242.

Reported findings used here:
  - White Caucasian adults need ~9 min/day lunchtime sun, March-September
  - Skin type V adults need ~25 min/day over the same period
  - Skin type V needs 2.5-3.0x the simulated-sunlight DOSE of white Caucasians
    for an equivalent 25(OH)D increase
  - Target: end-summer 85.8 nmol/L (so 97.5% of the population stay >= 25
    nmol/L through February)
  - Exposed BSA: 10% (hands + face) or 35% (adds forearms + lower legs)
  - Their stated acceptable daily dose: 2.75 SED

Why this is the right benchmark
-------------------------------
Every study already in `vitd_validation.py` is either a calibration target or an
explicit debug target, so the suite's pass rate says nothing about
out-of-sample skill (OPEN_QUESTIONS gap 22). This is an independent group,
independent cohort, and a naturalistic (not lamp) protocol.

What can and cannot be compared
-------------------------------
ABSOLUTE minutes are NOT a clean test. They depend on assumptions Webb/Kift make
that we cannot recover from the abstract: the ambient UVI climatology actually
used (UK lunchtime Mar-Sep averages ~3-4, peaking ~7 in June), and the
ambient-to-personal dose geometry factor. Their own numbers are not mutually
reconcilable without those (89.6 SED summer total / 2.75 SED per day = 33
day-equivalents, not the ~210 days of Mar-Sep -- so 2.75 SED is a safety cap,
not the 9-minute dose).

The DOSE RATIO between skin types IS a clean test: UVI and geometry cancel, and
both groups are treated identically. That is what this module asserts on.
"""

from __future__ import annotations

import numpy as np

from vitd_model import ModelParams, simulate, synthesis_per_session
from vitd_objectives import minutes_for_target

# Webb/Kift reported values
WK_MINUTES_WHITE = 9.0
WK_MINUTES_FST_V = 25.0
WK_DOSE_RATIO_LOW = 2.5
WK_DOSE_RATIO_HIGH = 3.0
WK_TARGET_NMOL_L = 85.8
WK_BSA_35 = 0.35

# Fitzpatrick assignment: "white Caucasian" in UK cohorts is predominantly
# FST II (Webb/Kift's white cohorts were FST I-III, mode II).
WK_FST_WHITE = 2
WK_FST_DARK = 5


def dose_ratio_for_equal_response(params: ModelParams | None = None,
                                  D0: float = 25.0,
                                  reference_sed: float = 1.0) -> float:
    """Dose multiple FST V needs to match FST II's 25(OH)D gain per session.

    Compared against Webb/Kift's reported 2.5-3.0x. Geometry- and
    climate-independent, so this is the falsifiable part of the benchmark.
    """
    if params is None:
        params = ModelParams()

    m_white = params.M_baseline(WK_FST_WHITE)
    target = synthesis_per_session(reference_sed, m_white, m_white,
                                   D0, 0.0, params)
    m_dark = params.M_baseline(WK_FST_DARK)

    lo, hi = 1e-4, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if synthesis_per_session(mid, m_dark, m_dark, D0, 0.0, params) < target:
            lo = mid
        else:
            hi = mid
    return float(hi / reference_sed)


def minutes_ratio(uv_index: float = 3.5,
                  params: ModelParams | None = None) -> dict:
    """FST V : FST II ratio of required daily minutes to hit 85.8 nmol/L."""
    out = {}
    for label, fst in (("white", WK_FST_WHITE), ("dark", WK_FST_DARK)):
        r = minutes_for_target(fst=fst, uv_index=uv_index,
                               exposed_bsa=WK_BSA_35,
                               frequency_per_week=7.0,
                               target=WK_TARGET_NMOL_L,
                               params=params,
                               max_minutes=600.0)
        out[label] = r
    mw = out["white"]["minutes"]
    md = out["dark"]["minutes"]
    ratio = (md / mw) if (mw and md) else None
    return {"white": out["white"], "dark": out["dark"], "ratio": ratio,
            "reported_ratio": WK_MINUTES_FST_V / WK_MINUTES_WHITE}


def run(verbose: bool = True) -> dict:
    """Run the held-out benchmark. Returns pass/fail per sub-check."""
    params = ModelParams()

    model_dose_ratio = dose_ratio_for_equal_response(params)
    dose_ratio_ok = WK_DOSE_RATIO_LOW <= model_dose_ratio <= WK_DOSE_RATIO_HIGH

    mr = minutes_ratio(uv_index=3.5, params=params)
    # Accept the minutes ratio within +/-40% of 2.78: the FST->minutes mapping
    # compounds the dose ratio with feedback and BSA nonlinearity, so it is a
    # looser test than the dose ratio itself.
    ratio_ok = (mr["ratio"] is not None and
                abs(mr["ratio"] - mr["reported_ratio"]) / mr["reported_ratio"] <= 0.40)

    if verbose:
        print("[HELD-OUT] Webb/Kift 2018 (Nutrients 10:457) -- NOT calibrated to")
        print(f"  FST V : FST II dose ratio for equal response")
        print(f"    model    = {model_dose_ratio:.2f}x")
        print(f"    reported = {WK_DOSE_RATIO_LOW}-{WK_DOSE_RATIO_HIGH}x")
        print(f"    {'PASS' if dose_ratio_ok else 'FAIL'}")
        print(f"  FST V : FST II required-minutes ratio (UVI 3.5, 35% BSA, target 85.8)")
        if mr["ratio"] is not None:
            print(f"    model    = {mr['ratio']:.2f}  "
                  f"({mr['white']['minutes']:.1f} vs {mr['dark']['minutes']:.1f} min)")
        else:
            print(f"    model    = target unreachable for at least one skin type")
        print(f"    reported = {mr['reported_ratio']:.2f} (25 min / 9 min)")
        print(f"    {'PASS' if ratio_ok else 'FAIL'} (+/-40%)")
        if not dose_ratio_ok:
            print("  NOTE: model under-predicts the skin-type penalty. Young 2020's")
            print("  FST II:VI ratio of 1.3-1.4 is per-unit-dose at MATCHED BSA under")
            print("  lamps; Webb/Kift's 2.5-3.0x is a whole-body naturalistic response.")
            print("  These measure different things, and K_const is calibrated to the")
            print("  former. See OPEN_QUESTIONS gap 23.")

    return {
        "webbkift_dose_ratio": dose_ratio_ok,
        "webbkift_minutes_ratio": ratio_ok,
        "model_dose_ratio": model_dose_ratio,
        "model_minutes_ratio": mr["ratio"],
    }


if __name__ == "__main__":
    run(verbose=True)
