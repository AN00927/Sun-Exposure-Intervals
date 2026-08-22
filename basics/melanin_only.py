"""
basics/melanin_only.py
======================
TIER-1 REFERENCE MODEL: constitutive melanin, and nothing else.

WHY THIS FILE EXISTS
--------------------
The full model has ~25 parameters and 13 validation checks, and a reviewer
cannot tell from it which term is carrying the skin-type result. This file
strips everything away except the one mechanism that produces the skin-type
gradient, so that mechanism can be checked on its own, against its own data,
in a file short enough to read in one sitting.

It is a REFERENCE model, not a replacement. It makes no absolute 25(OH)D
prediction and is not fitted to anything here. See
`docs/REPO_SCOPE_AUDIT.md` section 6.

THE MODEL, IN FULL
------------------
    response(FST, dose) = ln(1 + dose * E_source) * exp(-K_const * M(FST))

That is the entire model. No stratum corneum, no facultative tanning, no
product feedback g(D), no serum decay, no BSA term, no geometry, no risk.

The synthesis scalar A and the BSA factor are deliberately ABSENT: this model
only ever reports RATIOS between skin types at matched dose, and both cancel
exactly in a ratio. Dropping them is not an approximation, it is an algebraic
identity, and it is the point of the file -- it means nothing in the
comparison below depends on `A`, which is the model's single largest source
of uncertainty and is anchored on FST I-II data only.

WHAT THE RATIO COLLAPSES TO
---------------------------
At matched dose the log term cancels too, leaving

    ratio(a : b) = exp(-K_const * M(a)) / exp(-K_const * M(b))
                 = exp(K_const * (M(b) - M(a)))

So the ENTIRE skin-type gradient of this repository rests on exactly two
things: the scalar `K_const`, and the `M_baseline_by_FST` lookup table. That
is the finding this file is built to make unmissable.

PARAMETER VALUES
----------------
Read from `vitd_model.ModelParams` at runtime -- never restated here. The
canonical source, with derivations and caveats, is the `vitd_model.py` module
docstring (see `CLAUDE.md`).

VALIDATION DATA
---------------
Four within-study, matched-dose or behaviour-corrected routes, reproduced
from `scripts/kconst_recalibration.py` and `scripts/fst_gap_decomposition.py`
(run 2026-08-17). These are the SAME four routes `K_const` was fitted to, so
this is a consistency check, NOT independent validation -- stated plainly
rather than dressed up. Its value is that it makes the SPREAD between the
routes visible, which the aggregate fit hides.

DARK-SKIN PROVENANCE (standing directive, CLAUDE.md)
-----------------------------------------------------
Every FST V datapoint behind these routes is Indian-subcontinent ancestry at
UK latitude; every FST VI datapoint is lamp-based. `M_baseline(VI) = 1.00` is
an unvalidated group-level prior. Any VI ratio below is an extrapolation onto
that prior, not a measurement.

USAGE
-----
    PYTHONIOENCODING=utf-8 python3 basics/melanin_only.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams  # noqa: E402


def M_baseline(fst: int, params: ModelParams) -> float:
    """Constitutive melanin index for `fst`. Thin accessor for
    ModelParams.M_baseline so the formulas below read like the maths.
    """
    return float(params.M_baseline(fst))

FST_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

# ---------------------------------------------------------------------------
# Observed within-study optical factors (FST V : FST II unless noted).
#
# Reproduced verbatim from scripts/kconst_recalibration.py (HELD_OUT_FIELD,
# YOUNG_PUBLISHED_ADJUSTED) and its Felton contrast. "behaviour removed" means
# the study's measured personal-dosimetry difference was divided out first, so
# what remains is intended to be optics only -- see exploratory/vitd_behaviour.py
# for that separation argument.
#
# NOTE the Young 2020 entries are FST II : FST VI, not V : II. They are kept in
# their published form and converted for comparison rather than silently
# rescaled.
# ---------------------------------------------------------------------------
OBSERVED_ROUTES: Dict[str, Tuple[int, int, float, str]] = {
    # label: (darker FST, lighter FST, observed response ratio lighter:darker, method)
    # 2.14 is Felton's OBSERVED matched-dose FST II:V response ratio, quoted in
    # the vitd_model.py K_const docstring ("model FST II:V 0.96 vs observed
    # 2.14" at the pre-recalibration K_const = 0.33). Not to be confused with
    # Felton's measured erythema MED ratio of 3.95, which is a different
    # endpoint and is excluded from the fit for that reason.
    "Felton 2016":  (5, 2, 2.14,
                     "lab, matched dose, within-study, both baselines printed"),
    "Young 2020 (Arimed B)": (6, 2, 1.29,
                              "lab, matched dose, within-study, ANCOVA-adjusted"),
    "Young 2020 (TL-01)":    (6, 2, 1.43,
                              "lab, matched dose, within-study, ANCOVA-adjusted"),
    "Kift 2013 / Webb 2010": (5, 2, 1.75,
                              "field dosimetry, behaviour removed"),
    "Darling 2013":          (5, 2, 2.68,
                              "field dosimetry, behaviour removed"),
}

#: The approved target band for the FST V : II optical factor
#: (scripts/kconst_recalibration.py DIAGNOSED_BAND).
DIAGNOSED_BAND = (1.7, 2.7)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def attenuation(fst: int, params: ModelParams | None = None) -> float:
    """Constitutive melanin transmission factor for `fst`.

    exp(-K_const * M_baseline(fst)). 1.0 for FST I (M = 0), falling
    monotonically with pigmentation. This is the whole mechanism.
    """
    p = params or ModelParams()
    return float(np.exp(-p.K_const * M_baseline(fst, p)))


def response(fst: int, dose_sed: float, e_source: float = 1.0,
             params: ModelParams | None = None) -> float:
    """Relative per-session response. Arbitrary units -- only ratios mean
    anything, because the synthesis scalar A is deliberately omitted."""
    return float(np.log1p(dose_sed * e_source)) * attenuation(fst, params)


def ratio(lighter_fst: int, darker_fst: int,
          params: ModelParams | None = None) -> float:
    """Predicted matched-dose response ratio, lighter : darker.

    The dose term cancels exactly, so this is
    exp(K_const * (M(darker) - M(lighter))) and depends on nothing else.
    """
    p = params or ModelParams()
    return float(np.exp(p.K_const * (M_baseline(darker_fst, p)
                                     - M_baseline(lighter_fst, p))))


def implied_K_const(lighter_fst: int, darker_fst: int, observed_ratio: float,
                    params: ModelParams | None = None) -> float:
    """Invert the ratio: what K_const would reproduce `observed_ratio` exactly?

    K = ln(ratio) / (M(darker) - M(lighter)). Undefined when the two skin
    types share a baseline melanin index.
    """
    p = params or ModelParams()
    dM = M_baseline(darker_fst, p) - M_baseline(lighter_fst, p)
    if abs(dM) < 1e-12:
        return float("nan")
    return float(np.log(observed_ratio) / dM)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> dict:
    p = ModelParams()
    out: dict = {"K_const": p.K_const, "routes": {}, "gradient": {}}

    if verbose:
        print("=" * 74)
        print("TIER-1 REFERENCE MODEL: constitutive melanin only")
        print("=" * 74)
        print("  response(FST, dose) = ln(1 + dose*E) * exp(-K_const * M(FST))")
        print("  At matched dose the dose term cancels, so every ratio below")
        print("  is exp(K_const * dM) and depends on NOTHING ELSE -- not A,")
        print("  not BSA, not feedback, not stratum corneum, not decay.")
        print()
        print(f"  K_const in force  : {p.K_const:.4f}")
        print("  M_baseline table  : "
              + ", ".join(f"{FST_ROMAN[f]}={M_baseline(f, p):.2f}"
                          for f in range(1, 7)))
        print()

    # --- the predicted gradient -------------------------------------------
    if verbose:
        print("-" * 74)
        print("1. The whole skin-type gradient this model produces")
        print("-" * 74)
        print(f"  {'FST':>4} {'M_base':>7} {'transmission':>13} "
              f"{'response vs FST I':>19}")
    t1 = attenuation(1, p)
    for f in range(1, 7):
        t = attenuation(f, p)
        out["gradient"][f] = t
        if verbose:
            print(f"  {FST_ROMAN[f]:>4} {M_baseline(f, p):>7.2f} "
                  f"{t:>13.4f} {t / t1:>18.3f}x")
    if verbose:
        print()
        print(f"  FST V : II  = {ratio(2, 5, p):.3f}x    "
              f"FST VI : I = {ratio(1, 6, p):.3f}x")
        print()

    # --- consistency against the four fitting routes ----------------------
    if verbose:
        print("-" * 74)
        print("2. Against the routes K_const was fitted to")
        print("-" * 74)
        print("  NOT independent validation -- these are the SAME routes the")
        print("  aggregate fit used. The point is the SPREAD, which the")
        print("  aggregate value hides.")
        print()
        print(f"  {'route':<24} {'contrast':>9} {'obs':>6} {'model':>7} "
              f"{'err':>7} {'implied K':>10}")

    implied = []
    for label, (dark, light, obs, _method) in OBSERVED_ROUTES.items():
        mod = ratio(light, dark, p)
        k = implied_K_const(light, dark, obs, p)
        implied.append(k)
        out["routes"][label] = {"observed": obs, "model": mod,
                                "implied_K_const": k}
        if verbose:
            contrast = f"{FST_ROMAN[light]}:{FST_ROMAN[dark]}"
            print(f"  {label:<24} {contrast:>9} {obs:>6.2f} {mod:>7.2f} "
                  f"{100 * (mod - obs) / obs:>6.0f}% {k:>10.3f}")

    implied = np.array(implied, dtype=float)
    out["implied_K_mean"] = float(np.mean(implied))
    out["implied_K_sd"] = float(np.std(implied, ddof=1))
    out["implied_K_min"] = float(np.min(implied))
    out["implied_K_max"] = float(np.max(implied))

    if verbose:
        print()
        print(f"  Implied K_const across routes: mean {np.mean(implied):.3f}, "
              f"SD {np.std(implied, ddof=1):.3f}, "
              f"range {np.min(implied):.3f}-{np.max(implied):.3f}")
        print(f"  In force: {p.K_const:.3f}")
        print()
        print("  READ THIS AS: the routes disagree by a factor of "
              f"{np.max(implied) / max(np.min(implied), 1e-9):.1f} on the one")
        print("  parameter that carries the entire skin-type result. The value")
        print("  in force is their centre, not a reconciliation of them.")
        print()

    # --- the verdict -------------------------------------------------------
    v_ii = ratio(2, 5, p)
    in_band = DIAGNOSED_BAND[0] <= v_ii <= DIAGNOSED_BAND[1]
    out["V_II_ratio"] = v_ii
    out["V_II_in_diagnosed_band"] = bool(in_band)

    if verbose:
        print("-" * 74)
        print("3. Verdict")
        print("-" * 74)
        print(f"  FST V : II optical factor = {v_ii:.3f}, "
              f"approved band {DIAGNOSED_BAND[0]}-{DIAGNOSED_BAND[1]}: "
              f"{'IN BAND' if in_band else 'OUT OF BAND'}")
        print()
        print("  SUPPORTED  : the DIRECTION of the gradient. Every one of the")
        print("               five routes agrees darker skin responds less at")
        print("               matched dose. That is unanimous and robust.")
        print("  NOT SUPPORTED : the SIZE of the gradient as a point estimate.")
        print("               Two matched-dose LAMP studies (Felton, Young)")
        print("               disagree with each other, which no amount of")
        print("               modelling can reconcile. Report a range.")
        print("  EXTRAPOLATED : everything involving FST VI. M_baseline(VI) =")
        print("               1.00 is an unvalidated group-level prior and")
        print("               every VI ratio above is computed onto it.")
        print()
    return out


if __name__ == "__main__":
    run(verbose=True)
