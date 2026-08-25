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
    for an equivalent 25(OH)D increase (Webb 2018's citation of Farrar 2013)
  - Target: end-summer 85.8 nmol/L (so 97.5% of the population stay >= 25
    nmol/L through February)
  - Exposed BSA: 10% (hands + face) or 35% (adds forearms + lower legs)
  - Their stated acceptable daily dose: 2.75 SED

Why this is the right benchmark
-------------------------------
Every study already in `vitd_validation.py` is either a calibration target or
an explicit debug target, so the suite's pass rate says nothing about
out-of-sample skill (MODEL_AND_EVIDENCE.md, 'Remaining gaps', gap 22). This
is an independent group and independent cohort.

CORRECTION 2026-08-17: the paragraph above used to call this a "naturalistic
(not lamp) protocol" and, in `run()`'s failure note, contrasted it with
Young 2020 as "matched-dose lab vs whole-body naturalistic response". Wrong.
Webb 2018 attributes its 2.5-3.0x figure to reference [11] = Farrar 2013,
which is itself a SIMULATED-SUNLIGHT (lamp) cabinet trial at matched ~35% BSA
against a matched-dose white comparator (Rhodes 2010, same group, same Arimed
B source, same latitude) -- matched-dose, lamp-based, containing NO
behavioural exposure term. Re-verified 2026-08-17 (MODEL_AND_EVIDENCE.md gap
23) and independently reproduced in `scripts/fst_gap_decomposition.py` S1b.
The genuine tension is Young 2020 (lamp, ratio 1.3-1.4x) vs Farrar/Rhodes
2010 (lamp, ratio ~2.3x on primary cells) -- a LAMP-VS-LAMP conflict, not a
naturalistic-vs-matched-dose one. K_const cannot be "the matched-dose answer"
and simultaneously fail a matched-dose benchmark; the open question is which
matched-dose study the constant should track.

What can and cannot be compared
--------------------------------
ABSOLUTE minutes are NOT a clean test. They depend on assumptions Webb/Kift
make that we cannot recover from the abstract: the ambient UVI climatology
actually used (UK lunchtime Mar-Sep averages ~3-4, peaking ~7 in June), and
the ambient-to-personal dose geometry factor. Their own numbers are not
mutually reconcilable without those (89.6 SED summer total / 2.75 SED per day
= 33 day-equivalents, not the ~210 days of Mar-Sep -- so 2.75 SED is a safety
cap, not the 9-minute dose).

The DOSE RATIO between skin types IS a clean test: UVI and geometry cancel,
and both groups are treated identically. That is what this module asserts on.

ONE NUMBER, NOT TWO. An earlier version of this module asserted separately on
(a) the model's dose ratio for equal response against Webb 2018's reported
2.5-3.0x, and (b) the model's required-minutes ratio against Webb 2018's
reported 25 min / 9 min = 2.78x, as if these were two independent pieces of
literature evidence. They are not -- they are the same fact stated twice.
Webb 2018 Results, verbatim: "The intervention study [11] showed that skin
type V needs a dose ... 2.5-3 times that required by white Caucasians ...
FOR THIS REASON, the dose for which the 'safe exposure time' for skin type V
was set was 2.75 SED", and: "The safe exposure time ... was 25 min." Minutes
are proportional to SED at a fixed irradiance, so the white cap implied by
their own 9 min is 2.75 * 9/25 = 0.99 SED, giving a dose-cap ratio of
2.75/0.99 = 2.78 -- identical, by construction, to the 25/9 = 2.78 minutes
ratio. The 2.75 SED cap was derived FROM the 2.5-3.0x dose ratio, and the
25-minute figure follows from the cap: one underlying measurement (Farrar
2013's 2.5-3.0x), not two. `run()` therefore GATES on the dose ratio only
(`dose_ratio_for_equal_response`, the geometry/climate-independent quantity)
and reports the minutes ratio as diagnostic context, not a second pass/fail.
"""

from __future__ import annotations

import numpy as np

from vitd_model import ModelParams, simulate, synthesis_per_session
from vitd_objectives import minutes_for_target

# Webb/Kift reported values.
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
    climate-independent -- the falsifiable part of the benchmark.
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
    """FST V : FST II ratio of required daily minutes to hit 85.8 nmol/L.

    DIAGNOSTIC ONLY -- not a second independent check. Its `reported_ratio`
    (25 min / 9 min = 2.78) is the same underlying Webb 2018 measurement as
    `WK_DOSE_RATIO_LOW`-`WK_DOSE_RATIO_HIGH` (2.5-3.0x), re-expressed via
    the 2.75 SED safe-exposure-time cap; see the module docstring's "ONE
    NUMBER, NOT TWO" section for the derivation. `run()` does not gate on
    this value. Computed and printed because it captures the model's feedback
    and BSA nonlinearity that `dose_ratio_for_equal_response` (a
    single-session comparison) does not -- useful context even though not an
    independent literature target.
    """
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
    """Run the held-out benchmark. Returns pass/fail for the ONE gating check.

    COLLAPSED 2026-08-17 (was two asserted sub-checks). The dose ratio and
    required-minutes ratio each used to get their own PASS/FAIL against
    Webb 2018's 2.5-3.0x and 25min/9min=2.78x respectively. Those two
    reported numbers are the same measurement (see module docstring, "ONE
    NUMBER, NOT TWO"), so asserting on both double-counted one piece of
    evidence as two. Only the dose ratio -- geometry- and climate-independent,
    per the module docstring -- gates pass/fail. The minutes ratio is computed
    and printed for context (it reflects feedback and BSA nonlinearity the
    single-session dose ratio does not) but no longer produces an independent
    verdict.
    """
    params = ModelParams()

    model_dose_ratio = dose_ratio_for_equal_response(params)
    dose_ratio_ok = WK_DOSE_RATIO_LOW <= model_dose_ratio <= WK_DOSE_RATIO_HIGH

    mr = minutes_ratio(uv_index=3.5, params=params)

    if verbose:
        print("[HELD-OUT] Webb/Kift 2018 (Nutrients 10:457) -- NOT calibrated to")
        print(f"  FST V : FST II dose ratio for equal response (the ONE gating check;")
        print(f"  see module docstring for why the minutes ratio below is not a second one)")
        print(f"    model    = {model_dose_ratio:.2f}x")
        print(f"    reported = {WK_DOSE_RATIO_LOW}-{WK_DOSE_RATIO_HIGH}x")
        print(f"    {'PASS' if dose_ratio_ok else 'FAIL'}")
        print(f"  [diagnostic, not gated] FST V : FST II required-minutes ratio "
              f"(UVI 3.5, 35% BSA, target 85.8)")
        if mr["ratio"] is not None:
            print(f"    model    = {mr['ratio']:.2f}  "
                  f"({mr['white']['minutes']:.1f} vs {mr['dark']['minutes']:.1f} min)")
        else:
            print(f"    model    = target unreachable for at least one skin type")
        print(f"    reported = {mr['reported_ratio']:.2f} (25 min / 9 min -- "
              f"algebraically the same fact as the 2.5-3.0x above)")
        if not dose_ratio_ok:
            # Direction is COMPUTED, not asserted. This line read "under-predicts"
            # unconditionally until 2026-08-25, which was simply wrong at the
            # values it was printing beside: the model's 3.01x sits ABOVE the
            # 2.5-3.0x band, and the diagnostic minutes ratio (5.21) is likewise
            # above the reported 2.78. Both over-predict the skin-type penalty.
            _dir = ("over" if model_dose_ratio > WK_DOSE_RATIO_HIGH else "under")
            print(f"  NOTE: model {_dir}-predicts the skin-type penalty. "
                  f"Farrar 2013 (the")
            print("  source of Webb/Kift's 2.5-3.0x) is ITSELF a matched-dose, simulated-")
            print("  sunlight (lamp) trial at ~35% BSA against a matched-dose white")
            print("  comparator (Rhodes 2010) -- it contains NO behavioural exposure term,")
            print("  contrary to what an earlier version of this note claimed. Young 2020's")
            print("  FST II:VI ratio of 1.3-1.4 is also matched-dose, lamp-based. The two")
            print("  disagree with each other despite both being matched-dose lamp studies")
            print("  (a lamp-vs-lamp conflict, not naturalistic-vs-matched-dose), and")
            print("  K_const is calibrated to Young 2020. Re-verified 2026-08-17; see")
            print("  MODEL_AND_EVIDENCE.md gap 23 and scripts/fst_gap_decomposition.py S1b.")

    return {
        "webbkift_dose_ratio": dose_ratio_ok,
        "model_dose_ratio": model_dose_ratio,
        "model_minutes_ratio": mr["ratio"],
        "model_minutes_ratio_is_diagnostic_only": True,
    }


if __name__ == "__main__":
    run(verbose=True)
