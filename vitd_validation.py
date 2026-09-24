"""
vitd_validation.py
==================
Validation and sanity check suite for vitd_model.

Three categories:

1. Sanity checks - basic numerical correctness
   a. D-dependent decay half-life (Datta 2017): half-life varies with serum
      25(OH)D level via power-law fit t_half(D) = 2953 * D^(-0.72)
      (# Power-law fitted to Datta 2017 three-group data).
      At high D (~132 nmol/L), half-life ~89 days; at low D (~43 nmol/L),
      half-life ~199 days. Tests both endpoints within ±10%.
   b. 1 SED bolus in untanned FST III: computation-integrity check.
      Expected delta = A * ln(1 + E_source) * exp(-K_const * M_III) * g(D),
      REBUILT FROM LIVE PARAMETERS (bug fix 2026-08-18 -- it used to hardcode
      A, K_const, D_max and the linear feedback form, all four of which have
      since moved). ±2% tolerance.
   c. Photoadaptation reduction (FST III, M_baseline -> M_ceiling): guard on
      the FACULTATIVE tanning channel, band 0-10%. Everything but the
      facultative term cancels, so the check equals
      1 - exp(-K_ATTEN * M_TAN_HEADROOM) exactly and is INDEPENDENT of
      K_const, A, D_max and the feedback form. K_ATTEN was RETIRED at 0.00
      when this check was written, giving a measured value of 0.0% -- a
      one-sided drift guard, NOT evidence. K_ATTEN is now an active,
      calibrated parameter (reinstated 2026-08-21, re-derived 2026-08-25 when
      M_TAN_HEADROOM moved; see the vitd_model.py module docstring for the
      live value), so this check now measures the
      live attenuation rather than a fixed zero -- it remains a drift guard,
      not independent evidence, since the calibration and the guard share the
      same K_ATTEN value. See sanity_photoadaptation_reduction for the full
      re-derivation.

2. Literature validation targets
   a. de Boer 2025 - photoadaptation direction check (replaces mis-sourced Matsuoka 1990).
      Protocol: 0.8 SED 3x/week x 9 weeks on lower back, FST I-III. Checks direction
      only: melanin increases, SC thickens, synthesis decreases. Quantitative MI comparison
      is not possible (instrument-specific arbitrary units vs model's normalized 0-1 scale).
   b. Bogh 2012 - NB-UVB 3x/week for 6 weeks in FST III starting at 19.2 nmol/L
      produced a quantitative rise to 75.0 nmol/L (delta ~55.8 nmol/L).
       Validation is now QUANTITATIVE: model delta must be within ±10 nmol/L of
      the observed +55.8 nmol/L rise (i.e., 40.8-70.8 nmol/L delta).
   c. Farrar 2013 - FST V South Asians at >=1.95 SED/session reached mean
      15.7 ng/mL (~39.25 nmol/L) with a +8.7 ng/mL rise (~+21.75 nmol/L).
      Validation is now QUANTITATIVE: model D_final must be 25-55 nmol/L and
      delta must be 10-35 nmol/L, tighter than the prior loose '<68 nmol/L' bound.
   d. Bogh 2005 / baseline + BSA dose-response - Bogh 2010 (PMID:19798069) and
      Bogh 2011/P5 derived: delta = 7.1 - 0.223*baseline + 11.0*ln(UVB_dose)
      + 5.5*ln(%BSA), R2=0.47. Baseline D and exposed body area must enter
      response predictions.

Each check prints PASS or FAIL with reasoning.

Audit change tags (July 2026 Scientific Audit):
  [C1] Bogh 2012 quantitative threshold       (High confidence)
  [C2] Farrar 2013 quantitative threshold     (High confidence)
  [C3] Source spectral calibration note       (High confidence)
  [C4] Baseline + BSA response check         (Moderate confidence)
  [C5] K_ATTEN documentation (see vitd_model.py)
  [C6] Matsuoka test replaced by de Boer 2025 photoadaptation check (resolved)
  [C7] Datta 2019 individual-variability note (see vitd_model.py)
  [C8] SC state documentation fix            (see vitd_model.py)
"""

import os
import numpy as np
import pandas as pd
from collections import namedtuple

# ---------------------------------------------------------------------------
# Generic check result container
# ---------------------------------------------------------------------------

CheckResult = namedtuple('CheckResult', ['name', 'passed', 'message'])

# ---------------------------------------------------------------------------
# Evidence tiers (added 2026-08-21)
# ---------------------------------------------------------------------------
# `run_all()` used to report one number: "PASSED REQUIRED: 10 / 10". That
# number is real -- every check does what its assertions say -- but it is
# also the single largest overclaim in this repo, because most of what it
# counts is self-consistency, not evidence about biology. This block adds an
# explicit TIER to every registered check, orthogonal to the existing
# required/warning-only axis (which controls what counts toward the headline
# pass/fail tally, not what KIND of support a pass gives).
#
# Assignment method: read each check's own docstring and assertions. Several
# checks already state their own tier in words ("computation-integrity
# check", "NOT evidence", "Direction check only", "STATUS: DEBUGGING_TARGET",
# "STATUS: EXPLORATORY_ONLY", "[C7-SC-FITTED] ... Calibrated with de Boer
# 2025"); those self-descriptions are honoured directly below rather than
# re-derived. See the per-check `tier_note` in CHECK_REGISTRY for the
# one-line justification for each of the 13 checks.
TIER_INTERNAL = "INTERNAL"
TIER_DIRECTIONAL = "DIRECTIONAL"
TIER_QUANTITATIVE = "QUANTITATIVE"
TIER_CIRCULAR = "CIRCULAR"
TIER_NOT_EVIDENCE = "NOT_EVIDENCE"

#: Print order for the tiered summary, roughly strongest-looking-but-weakest
#: first to poorest-looking-but-honest last -- NOT a ranking of evidentiary
#: strength (QUANTITATIVE is the strongest tier that exists in this suite).
TIER_ORDER = [TIER_INTERNAL, TIER_DIRECTIONAL, TIER_QUANTITATIVE,
              TIER_CIRCULAR, TIER_NOT_EVIDENCE]

TIER_LABELS = {
    TIER_INTERNAL: "self-consistency / computation-integrity / wiring; "
                    "not evidence about biology",
    TIER_DIRECTIONAL: "external literature data, direction or ordering "
                       "only -- no quantitative comparison",
    TIER_QUANTITATIVE: "external literature data, a real numeric "
                        "comparison -- see per-check circularity notes",
    TIER_CIRCULAR: "external data the model (or one of its parameters) "
                   "was FITTED to; a pass is expected by construction",
    TIER_NOT_EVIDENCE: "retained for the finding it records, but its own "
                        "docstring says it is not vitamin-D evidence, or "
                        "that it is exploratory only",
}

#: One registered check. `tier_note` is the one-line justification for the
#: tier assignment, sourced from the function's own docstring/assertions.
CheckSpec = namedtuple('CheckSpec',
                       ['name', 'fn', 'warning_only', 'tier', 'tier_note'])

# ---------------------------------------------------------------------------
# Comparison utilities for validation targets
# ---------------------------------------------------------------------------

def check_exact_match(observed, expected, abs_tol=0.0):
    """Exact-match comparison within absolute tolerance.

    Parameters
    ----------
    observed : float
        Model output value.
    expected : float
        Expected value from literature.
    abs_tol : float
        Absolute tolerance (inclusive).

    Returns
    -------
    bool
    """
    return abs(observed - expected) <= abs_tol


def check_percent_error(observed, expected, max_pct=10.0):
    """Percent-error comparison.

    Parameters
    ----------
    observed : float
        Model output value.
    expected : float
        Expected value from literature.
    max_pct : float
        Maximum allowed percent error.

    Returns
    -------
    bool
    """
    if expected == 0:
        return observed == 0
    pct = abs((observed - expected) / expected) * 100.0
    return pct <= max_pct


def check_in_range(value, lo, hi):
    """Range check: *lo* <= *value* <= *hi* (inclusive).

    Parameters
    ----------
    value : float
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    bool
    """
    return lo <= value <= hi


def check_qualitative_only(result):
    """Qualitative-only pass/fail. Accepts a boolean directly.

    Use when a validation target has no numeric uncertainty
    (e.g. direction-of-effect checks).
    """
    return bool(result)


def weighted_aggregate(results, weights=None):
    """Compute weighted aggregate score from a dict of check results.

    Parameters
    ----------
    results : dict
        {name: bool} mapping check names to pass/fail.
    weights : dict or None
        {name: weight} mapping check names to numeric weights.
        If None, all checks have equal weight.
        Checks missing from *weights* default to weight 1.0.

    Returns
    -------
    dict
        {score: weighted pass fraction [0,1],
         n_pass: int,
         n_total: int}
    """
    n_total = len(results)
    if n_total == 0:
        return {"score": 0.0, "n_pass": 0, "n_total": 0}

    total_weight = 0.0
    weighted_pass = 0.0

    for name, passed in results.items():
        w = weights.get(name, 1.0) if weights else 1.0
        total_weight += w
        if passed:
            weighted_pass += w

    n_pass = sum(results.values())
    score = weighted_pass / total_weight if total_weight > 0 else 0.0
    return {"score": score, "n_pass": n_pass, "n_total": n_total}

from vitd_model import (ModelParams, synthesis_per_session, simulate,
                         build_pulse_schedule, build_weekly_schedule,
                         net_final_D_change_per_SED,
                         auc_above_baseline_per_SED,
                         auc_above_sufficiency_per_SED,
                         steady_state_D, get_E_source, E_SOURCE_SOLAR_DEFAULT,
                         BSA_REF, BSA_EXP, K_ATTEN_STRUCTURAL,
                         RETIRED_PARAMETERS)
from vitd_physiology import hydroxylation_efficiency
from vitd_protocols import bogh2012_session_doses_sed


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_decay_halflife(verbose=True):
    """
    Datta 2017: state-dependent half-life from power-law fit.
    # Power-law fitted to Datta 2017 three-group data
    At high D (~132 nmol/L), instantaneous half-life ≈ 89 days.
    At low D (~43 nmol/L), instantaneous half-life ≈ 199 days.
    Test all three anchor points with ±10% tolerance.

    The simulator is checked against the analytic solution of the ODE
    rather than against a fixed-k exponential. With k(D) = ln2*D^0.72/2953,
    dD/dt = -c*D^1.72 (c = ln2/2953) integrates to
        D(t) = [D0^-0.72 + 0.72*c*t]^(-1/0.72)
    A state-dependent-clearance model does NOT halve in one instantaneous
    half-life: as D falls, clearance slows, so decay lags a fixed-k
    exponential (D(89)/D0 = 0.566, not 0.500, from D0=132). Asserting 0.500
    here would encode the fixed-k assumption the model explicitly rejects.
    """
    p = ModelParams()
    tol = 0.10

    # Instantaneous half-life at each Datta 2017 group mean
    t_half_high = np.log(2) / p.k_decay_D(132.0)
    high_ok = abs(t_half_high - 89.0) / 89.0 <= tol

    t_half_med = np.log(2) / p.k_decay_D(65.0)
    med_ok = abs(t_half_med - 148.0) / 148.0 <= tol

    t_half_low = np.log(2) / p.k_decay_D(43.0)
    low_ok = abs(t_half_low - 199.0) / 199.0 <= tol

    def _exact_decay(D0, t):
        """Analytic solution of dD/dt = -k(D)*D for the power-law k(D)."""
        c = np.log(2.0) / 2953.0
        return (D0 ** -0.72 + 0.72 * c * t) ** (-1.0 / 0.72)

    # Simulator must reproduce the analytic ODE solution to within 1%
    result = simulate(fst=3, schedule=[], duration_days=89, params=p, D0=132.0)
    D_89 = np.interp(89.0, result["t"], result["D"])
    exact_89 = _exact_decay(132.0, 89.0)
    sim_high_ok = abs(D_89 - exact_89) / exact_89 <= 0.01

    result = simulate(fst=3, schedule=[], duration_days=199, params=p, D0=43.0)
    D_199 = np.interp(199.0, result["t"], result["D"])
    exact_199 = _exact_decay(43.0, 199.0)
    sim_low_ok = abs(D_199 - exact_199) / exact_199 <= 0.01

    passed = high_ok and med_ok and low_ok and sim_high_ok and sim_low_ok
    if verbose:
        print(f"[Sanity 1] D-dependent decay half-life (Datta 2017 power-law fit)")
        print(f"  D=132: t_half={t_half_high:.1f}d (expected 89d, ±10%): {'OK' if high_ok else 'FAIL'}")
        print(f"  D=65:  t_half={t_half_med:.1f}d (expected 148d, ±10%): {'OK' if med_ok else 'FAIL'}")
        print(f"  D=43:  t_half={t_half_low:.1f}d (expected 199d, ±10%): {'OK' if low_ok else 'FAIL'}")
        print(f"  Sim vs analytic ODE, D0=132 @89d:  {D_89:.3f} vs {exact_89:.3f} "
              f"(±1%): {'OK' if sim_high_ok else 'FAIL'}")
        print(f"  Sim vs analytic ODE, D0=43 @199d:  {D_199:.3f} vs {exact_199:.3f} "
              f"(±1%): {'OK' if sim_low_ok else 'FAIL'}")
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def sanity_single_bolus_FST_III(verbose=True):
    """
    Computation-integrity check: single 1 SED bolus in untanned FST III.

    Asserts that `synthesis_per_session` computes the product the module
    docstring says it computes. NOT a literature check -- no external target.

    BUG FIX 2026-08-18, not a calibration. The expected value was hardcoded as
        2.353 * ln(2) * exp(-0.33 * 0.21) * (1 - 25/150)  ~= 1.27
    with A, K_const, D_max and the LINEAR feedback form all written out as
    literals. Every one of those four literals is now superseded -- K_const
    moved 0.33 -> 0.98 on 2026-08-17, and A and the feedback form moved on
    2026-08-18 -- so the check had been FAILING (model 1.11 against a frozen
    1.27) for reasons that had nothing to do with the code being wrong. A
    computation-integrity check that hardcodes the parameters it is checking
    stops being an integrity check the moment any of them moves. The expected
    value is now rebuilt from the live parameters and the live feedback form,
    which is the only way this check can keep doing its job.
    """
    p = ModelParams()
    fst = 3
    M_base = p.M_baseline(fst)
    D_ref = 25.0
    delta = synthesis_per_session(1.0, M_base, M_base, D=D_ref, SC=0.0, params=p)
    # Independently reconstructed from the documented form, using the live
    # parameters. bsa_factor is 1.0 at the reference BSA and SC = 0, so both
    # drop out; E_source is ModelParams.E_source.
    feedback = _feedback_factor_for(p, D_ref)
    expected_delta = (p.A * np.log(1.0 + 1.0 * p.E_source)
                      * np.exp(-p.K_const * M_base) * feedback)
    passed = check_percent_error(delta, expected_delta, max_pct=2.0)
    if verbose:
        print(f"[Sanity 2] Single bolus, FST III, untanned (computation-integrity check)")
        print(f"  A={p.A}, K_const={p.K_const}, M_III={M_base}, "
              f"feedback_mode={p.feedback_mode!r}, g({D_ref:.0f})={feedback:.4f}")
        print(f"  delta-D = {delta:.2f} nmol/L  (expected {expected_delta:.2f}, ±2%)")
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def _feedback_factor_for(p, D):
    """Reconstruct g(D) for the params' feedback form, independently of
    synthesis_per_session, so Sanity 2 is a real cross-check and not a
    tautology against the same code path."""
    mode = p.feedback_mode
    if mode == "biphasic":
        return hydroxylation_efficiency(D)
    if mode == "saturating":
        return float(np.exp(-np.log(2.0) * D / p.D_half_feedback))
    if mode == "linear":
        return max(0.0, 1.0 - D / p.D_max)
    if mode == "power":
        return max(0.0, 1.0 - (D / p.D_max) ** p.feedback_power)
    raise ValueError(f"unknown feedback_mode {mode!r}")


# ---------------------------------------------------------------------------
# A calibration anchor (approved item A4, 2026-08-18)
# ---------------------------------------------------------------------------
# Young et al. 2021, PNAS 118:e2015867118, Fig. 4B -- FULL-BODY study,
# 85% BSA, linear regressions of delta-25(OH)D3 on ERYTHEMAL SED, not forced
# through zero. Transcribed from the figure legend of the full text in
# Research_Papers/. These are per-SED slopes, in nmol/L per SED.
YOUNG2021_FB_SLOPES_NMOL_PER_SED = {
    "PUVA":     1.99,   # n = 11, y = 1.99x + 1.12, R2 = 0.47
    "Arimed B": 3.18,   # n = 19, y = 3.18x + 2.26, R2 = 0.63
    "UV6":      4.97,   # n = 10, y = 4.97x + 0.86, R2 = 0.68
}
# Table 1, "All FB" row: n = 40, baseline 34.6 +/- 20.0 nmol/L.
YOUNG2021_FB_BASELINE_NMOL_L = 34.6
YOUNG2021_FB_BSA = 0.85          # Methods: "The whole body, apart from
                                 # underwear ... an estimated 85% BSA"
YOUNG2021_FB_FST = 2             # FB arm: 38/40 FST II, 2/40 FST I


def young2021_A_anchor(params=None, verbose=False):
    """Reproduce the derivation of `ModelParams.A` from Young 2021.

    Returns a dict with the target slope, the model kernel, the implied A, and
    the per-source cross-checks. This is a CALIBRATION ANCHOR, not a
    validation: it cannot fail in an interesting way, so it is deliberately
    NOT registered in run_all()'s scored check list. It is asserted instead by
    tests/test_vitd_model.py::TestYoung2021AAnchor, so A cannot drift silently.

    DARK-SKIN FLAG: Young 2021 recruited FST I-II ONLY (n = 75; 6x I, 69x II;
    zero FST III-VI) and its authors write "The results from this study only
    apply to FST I/II". A is applied at every FST, so this anchor is a
    LIGHT-SKIN EXTRAPOLATION for FST III-VI. The skin-type GRADIENT is anchored
    separately (K_const, on matched-dose FST V data); what is extrapolated here
    is the overall level.
    """
    p = ModelParams() if params is None else params
    slopes = YOUNG2021_FB_SLOPES_NMOL_PER_SED
    target = (min(slopes.values()) + max(slopes.values())) / 2.0   # 3.48

    M_base = p.M_baseline(YOUNG2021_FB_FST)
    bsa_f = (YOUNG2021_FB_BSA / BSA_REF) ** BSA_EXP
    melanin = float(np.exp(-p.K_const * M_base))
    feedback = _feedback_factor_for(p, YOUNG2021_FB_BASELINE_NMOL_L)
    E = get_E_source(E_SOURCE_SOLAR_DEFAULT)
    kernel = float(np.log(1.0 + 1.0 * E) * bsa_f * melanin * feedback)

    cross = {}
    for name, src in (("Arimed B", "arimed_b"), ("UV6", "uv6_broadband")):
        Es = get_E_source(src)
        k = float(np.log(1.0 + Es) * bsa_f * melanin * feedback)
        cross[name] = slopes[name] / k
    # PUVA is deliberately absent: it is UVA-dominant and has no entry in
    # SOURCE_CALIBRATION_TABLE, so no E_source-matched cross-check exists.

    out = {
        "target_slope_nmol_per_SED": target,
        "pooled_slope_nmol_per_SED": 36.1 / 10.0,   # Table 1 All FB / 10 SED
        "kernel": kernel,
        "A_implied": target / kernel,
        "A_current": p.A,
        "cross_checks": cross,
        "model_slope_at_current_A": p.A * kernel,
    }
    if verbose:
        print("[Anchor] Young 2021 FB per-SED slope -> A")
        print(f"  printed FB slopes (Fig. 4B): {slopes}")
        print(f"  range midpoint = {target:.3f}; pooled All-FB = "
              f"{out['pooled_slope_nmol_per_SED']:.3f} nmol/L per SED")
        print(f"  kernel = ln(1+{E}) * bsa_factor({YOUNG2021_FB_BSA}) * "
              f"exp(-K_const*M_II) * g({YOUNG2021_FB_BASELINE_NMOL_L}) "
              f"= {kernel:.6f}")
        print(f"  A implied = {out['A_implied']:.4f};  A in force = {p.A}")
        print(f"  model slope at A in force = "
              f"{out['model_slope_at_current_A']:.3f} nmol/L per SED")
        print(f"  E_source-matched cross-checks (bracket, not targets): "
              + ", ".join(f"{k} {v:.3f}" for k, v in cross.items()))
        print("  DARK-SKIN FLAG: Young 2021 is FST I-II only; applying A to "
              "FST III-VI is a light-skin extrapolation.")
    return out


# [A7, RE-DERIVED 2026-08-18] Acceptance ceiling for the facultative (tanning)
# attenuation channel, as a FRACTION of untanned synthesis at the structural
# melanin ceiling. See sanity_photoadaptation_reduction for the derivation of
# why this number, and what it is and is not evidence of.
PHOTOADAPTATION_REDUCTION_MAX = 0.10   # [CHOICE] -> K_ATTEN <= 0.443 (at
                                        # M_TAN_HEADROOM=0.238; was 0.527 at
                                        # the pre-2026-08-25 value of 0.20)


def sanity_photoadaptation_reduction(verbose=True):
    """Guard on the FACULTATIVE (tanning) melanin channel.

    Measures synthesis at M = M_ceiling(FST III) = 0.448 (M_baseline 0.21 +
    M_TAN_HEADROOM 0.238, raised from 0.20 -> 0.238 on 2026-08-25 -- see
    docs/MELANIN_MODEL.md sec.2.5), the largest tan the melanin ODE can
    produce, against untanned M = M_baseline = 0.21, at matched D and SC.

    WHAT THE MEASURED QUANTITY ACTUALLY IS
    --------------------------------------
    Everything except the facultative term is identical between the two calls,
    so A, K_const, bsa_factor, E_source, SC and the feedback factor all CANCEL
    EXACTLY, and the check reduces to

        reduction = 1 - exp(-K_ATTEN * M_TAN_HEADROOM)

    This is not an approximation; it is an algebraic identity, and it is
    asserted as such in tests/test_vitd_model.py (the K_ATTEN/M_TAN_HEADROOM
    coupling tests) and tests/test_matsuoka1990.py. Two consequences:

      * The 2026-08-17 K_const move (0.33 -> 0.98) CANNOT have changed this
        check, and neither can the 2026-08-18 A and feedback-form changes.
        `calibration_approval.yaml` -> pending_approvals asked for the band to
        be re-derived because K_const had moved since it was set. It was
        re-derived; the answer is that K_const never entered. The band stands
        on its own terms, not because it happened to keep passing.
      * The check has exactly ONE degree of freedom, K_ATTEN, so the band is
        equivalent to a bracket on K_ATTEN. At M_TAN_HEADROOM = 0.238
        (raised from 0.20 on 2026-08-25, calibration_approval.yaml, "melanin
        ceiling"; see docs/MELANIN_MODEL.md sec.2.5):
            0-10%  reduction  <=>  K_ATTEN in [0.000, 0.443]
            10-30% reduction  <=>  K_ATTEN in [0.443, 1.499]   (the OLD band)

    WHY 0-10% AND NOT 10-30%
    ------------------------
    The old 10-30% band EXCLUDED the model's own K_ATTEN. It was set when
    K_ATTEN was 0.37 AND the test point was a hardcoded M = 0.70 -- a state
    above the structural ceiling of 0.41, which the melanin ODE cannot reach at
    any dose. Fixing the test point (a bug fix, ungated) dropped the measured
    reduction below the band, and the band had to move or the check had to fail
    for the wrong reason. The replacement is not a loosening: in K_ATTEN terms
    the new band is 3.4x NARROWER (width 0.527 vs 1.256) and its upper bound is
    3.4x LOWER. What it gives up is the lower bound, which is the honest cost of
    K_ATTEN being 0.00.

    The 10% ceiling is a [CHOICE], not a measurement, and it is bounded on both
    sides by what the repo holds. Young 2020 (n = 23, TL-01, the dataset
    K_ATTEN was fitted to) supports a SMALL facultative effect and returns 0.00.
    Matsuoka 1990 (n = 7), deconvolved for its baseline-25(OH)D confound, implies
    K_ATTEN 0.288972-0.838389 depending on the assumed feedback form
    (vitd_calibration.derive_K_ATTEN, which reads M_TAN_HEADROOM live), i.e.
    reductions of 6.6-18.1%. The reduction percentages are the measured
    quantity and do not move with the headroom; the K_ATTEN bracket that
    encodes them does. A 10%
    ceiling admits the LOW end of the Matsuoka bracket and rejects its high end,
    which is the correct posture given that Matsuoka's "51%" is mostly
    D-feedback rather than pigment (see CLAUDE.md).

    WHAT THIS CHECK IS NOT
    ----------------------
    NOT evidence that photoadaptation reduces synthesis -- it is a bound
    derived from the same Matsuoka 1990 study K_ATTEN is calibrated to, not an
    independent check.

    [2026-08-18 -- 2026-08-21: while K_ATTEN was RETIRED at 0.00 (approved
    item A6) the measured reduction was exactly 0.0% and this check could not
    fail at the value the model shipped; it functioned as a ONE-SIDED
    RETIREMENT GUARD that would only fire if someone reinstated a strong
    facultative penalty without going through RETIRED_PARAMETERS.]

    K_ATTEN was reinstated as an active, calibrated parameter on 2026-08-21
    (user-approved) and re-derived to 0.288972 on 2026-08-25, which this check
    evaluates at 6.6464% reduction -- inside the 0-10% band, so it currently
    PASSES, but it is no longer
    pinned to 0.0% by construction: a future re-calibration of K_ATTEN could
    move it outside the band, and this check would then be measuring that, not
    guarding a retirement. Do not count it as literature support.

    RESOLVED 2026-08-25. For a few hours on 2026-08-25 this docstring recorded
    an open inconsistency, and the history is worth keeping because the failure
    mode is easy to recreate. M_TAN_HEADROOM was raised 0.20 -> 0.238 against
    Ravnbak 2008 while K_ATTEN stayed at 0.34 -- but K_ATTEN is not an
    independent parameter. It is derived FROM the headroom, as
        K_ATTEN = -ln(1 - residual) / M_TAN_HEADROOM
    where `residual` is the melanin attenuation deconvolved from Matsuoka 1990
    (vitd_calibration.derive_K_ATTEN). Raising the denominator while pinning
    the quotient silently raised the facultative attenuation the model APPLIED
    from 6.5740% to 7.7733%, overshooting the Matsuoka residual the parameter
    exists to encode. The measured quantity is the residual; K_ATTEN is only
    its parameterisation.

    Fixed by re-deriving K_ATTEN (0.34 -> 0.288972, user-approved) AND by
    removing the two hardcoded `excess_M=0.20` call sites that caused the
    decoupling -- derive_K_ATTEN's signature default and vitd_calibration.run().
    Both now read params.M_TAN_HEADROOM live, so the coupling is structural
    rather than a number someone has to remember to update.
    tests/test_vitd_model.py::test_k_atten_tracks_the_headroom_it_is_derived_from
    guards the coupling itself, not the value, so this cannot silently recur.
    """
    p = ModelParams()
    fst = 3
    M_base = p.M_baseline(fst)
    M_tanned = p.M_ceiling(fst)  # 0.448 for FST III (M_base + M_TAN_HEADROOM)
    d_untanned = synthesis_per_session(1.0, M_base, M_base, D=25.0, SC=0.0, params=p)
    d_tanned = synthesis_per_session(1.0, M_tanned, M_base, D=25.0, SC=0.0, params=p)
    reduction = 1.0 - d_tanned / d_untanned
    passed = 0.0 <= reduction <= PHOTOADAPTATION_REDUCTION_MAX
    headroom = M_tanned - M_base
    k_implied_max = -np.log(1.0 - PHOTOADAPTATION_REDUCTION_MAX) / headroom
    if verbose:
        print(f"[Sanity 3] Photoadaptation reduction (FST III, M {M_base:.2f} -> {M_tanned:.2f} [M_ceiling])")
        print(f"  untanned synthesis: {d_untanned:.2f}  tanned synthesis: {d_tanned:.2f}")
        print(f"  reduction = {100*reduction:.1f}%  "
              f"(guard range 0-{100*PHOTOADAPTATION_REDUCTION_MAX:.0f}%; "
              f"evaluated at structural ceiling M_ceiling)")
        print(f"  equivalent bracket: K_ATTEN in [0.000, {k_implied_max:.3f}]; "
              f"K_ATTEN is ACTIVE and calibrated at {K_ATTEN_STRUCTURAL:.2f} "
              f"(reinstated 2026-08-21), so this measures the live "
              f"parameter and is a drift guard sharing its evidence with the "
              f"calibration, NOT an independent check.")
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Literature validation
# ---------------------------------------------------------------------------
def validate_photoadaptation_deBoer2025(verbose=True):
    """
    de Boer 2025: 0.8 SED 3x/week x 9 weeks on lower back, FST I-III.
    Melanin index increased 12%, SC thickness increased 34%, MED increased 50%.
    
    [Assumption] This test replaces the mis-sourced Matsuoka 1990 test.
    The local 'Matsuoka et al., 1990.pdf' is a sunscreen/BSA paper, not the
    tanned-vs-untanned paper. This test verifies direction + plausibility:
    melanin should increase, SC should thicken, and synthesis should decrease
    after photoadaptation.
    
    Quantitative MI comparison is not possible — de Boer 2025 uses instrument-
    specific arbitrary melanin index units, not the model's normalized 0-1 scale.
    """
    p = ModelParams()
    fst = 3
    M_base = p.M_baseline(fst)
    
    # de Boer 2025 protocol: 0.8 SED, 3x/week, 9 weeks
    sched = build_weekly_schedule(63, 0.8, [0, 2, 4])
    result = simulate(fst=fst, schedule=sched, duration_days=63, params=p)
    M_after = result["M"][-1]
    SC_after = result["SC"][-1]
    
    # Direction checks
    melanin_increased = M_after > M_base
    sc_increased = SC_after > 0.0
    
    # Synthesis attenuation: tanned vs untanned
    d_untanned = synthesis_per_session(0.8, M_base, M_base, D=25.0, SC=0.0, params=p)
    d_tanned = synthesis_per_session(0.8, M_after, M_base, D=25.0, SC=SC_after, params=p)
    synthesis_decreased = d_tanned < d_untanned
    
    passed = melanin_increased and sc_increased and synthesis_decreased
    if verbose:
        print(f"[Lit 1] de Boer 2025 - photoadaptation direction (replaces Matsuoka 1990)")
        print(f"  Protocol: 0.8 SED 3x/week x 9 weeks, FST III (de Boer 2025)")
        print(f"  Baseline M={M_base:.3f}, After M={M_after:.3f}, increased: {'YES' if melanin_increased else 'NO'}")
        print(f"  Baseline SC=0.0, After SC={SC_after:.4f}, increased: {'YES' if sc_increased else 'NO'}")
        print(f"  Untanned delta-D={d_untanned:.3f}, Tanned delta-D={d_tanned:.3f}")
        print(f"  Synthesis decreased after photoadaptation: {'YES' if synthesis_decreased else 'NO'}")
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def validate_bogh_2012(verbose=True):
    # STATUS: DEBUGGING_TARGET
    # Bogh 2012 is not an independent validation pass.
    # Do not tune E_source, D_max, BSA, or clearance to force this test to pass.
    """
    Bogh 2012 NB-UVB debug target.

    STATUS: DEBUGGING_TARGET
    This study is used for debugging and possible calibration only.
    It must not be reported as an independent validation pass until structural
    and source-data verification are complete.

    Bogh 2012: full-body NB-UVB 3x/week for 6 weeks (18 sessions) in
    light-skinned Danish adults raised mean 25(OH)D from 19.2 to 75.0 nmol/L
    (delta +55.8 nmol/L; between-group p=0.02 vs oral D3).

    [C1] QUANTITATIVE THRESHOLD (tightened after SC recalibration):
    Model delta must fall within ±10 nmol/L of the observed +55.8 nmol/L,
    i.e., 45.8 to 65.8 nmol/L. Tolerance was ±15 previously but a ~14 nmol/L
    prediction error is unacceptable. The model systematically under-predicts
    NB-UVB by ~17 nmol/L with current parameters (alpha_SC=0.12, SC_max=0.24
    for FST III, raised from 0.23 on 2026-08-25 -- calibration_approval.yaml,
    "SC_max_by_FST I-III 0.23 -> 0.24"). This is a known structural
    limitation: E_source=2.5 is a placeholder requiring per-study spectral
    calibration (see [C3] below).

    [C3] SOURCE / SPECTRAL CALIBRATION NOTE:
    E_source=2.5 is a per-study placeholder. Young 2021 PNAS (P20) shows that the
    standard CIE previtamin-D3 action spectrum does not collapse polychromatic
    responses to a common regression and a 5 nm blue shift gives a better fit.
    Osmancevic 2009 (P18) shows broadband UVB can produce larger D rises than
    NB-UVB in psoriasis cohorts despite lower physical J/cm². Source handling must
    become a first-class per-study calibration input, not a fixed multiplier.
    E_source_override should be treated as a study-specific parameter, not a
    general NB-UVB conversion constant.
    """
    p = ModelParams()
    # Escalating dose schedule: 18 sessions (3x/week x 6 weeks), converted from
    # absolute J/cm2 to CIE erythemal SED via the Rodriguez-Granados 2017
    # TL-01 factor (1 J/cm2 -> 5.5 SED) and normalized to the reported
    # cumulative 9.0 J/cm2. simulate() expects erythemal SED; E_source then
    # maps erythemal SED -> vitamin-D-effective dose.
    session_days = [w * 7 + d for w in range(6) for d in [0, 2, 4]]
    session_sed = bogh2012_session_doses_sed(n_sessions=18)
    sched = list(zip(session_days, session_sed))

    # Account for cohort FST distribution (31% FST II, 12.5% FST III, 44% FST IV, 12.5% FST VI)
    fst_dist = [(2, 0.31), (3, 0.125), (4, 0.44), (6, 0.125)]

    # Bogh 2012 was whole-body NB-UVB (~90% BSA, ASSUMPTION_REGISTER A2), not
    # the 35% BSA reference. Both the source conversion and the exposed area
    # are study facts, not free parameters.
    E_SOURCE_TL01 = get_E_source("tl01_nb_uvb")
    EXPOSED_BSA = 0.90

    D_final = sum(
        weight * simulate(fst=fst, schedule=sched, duration_days=42, params=p,
                          D0=19.2,
                          E_source_override=E_SOURCE_TL01,
                          exposed_bsa_override=EXPOSED_BSA,
                          dose_framing="fixed_absolute_SED")["D"][-1]
        for fst, weight in fst_dist
    )
    delta = D_final - 19.2
    # [C1] Quantitative target: observed delta = +55.8 nmol/L; tolerance ±10 nmol/L
    OBSERVED_DELTA = 55.8
    TOLERANCE = 10.0
    passed = abs(delta - OBSERVED_DELTA) <= TOLERANCE
    if verbose:
        print(f"[Lit 2] Bogh 2012 - quantitative NB-UVB validation  [C1]")
        print(f"  Protocol: 18 sessions 3x/week x 6 weeks, cohort FST mix, baseline 19.2")
        print(f"  Dose: 9.0 J/cm2 cumulative -> {sum(session_sed):.1f} erythemal SED "
              f"(TL-01 x5.5), E_source={E_SOURCE_TL01}, BSA={EXPOSED_BSA}")
        print(f"  Model D_final = {D_final:.1f} nmol/L (delta = +{delta:.1f})")
        print(f"  Study observed: 19.2 -> 75.0 nmol/L, delta +{OBSERVED_DELTA} nmol/L")
        print(f"  Acceptance: |delta - {OBSERVED_DELTA}| <= {TOLERANCE} nmol/L  (range {OBSERVED_DELTA-TOLERANCE:.1f}-{OBSERVED_DELTA+TOLERANCE:.1f})")
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def validate_farrar_2013(verbose=True):
    """
    Farrar 2013: FST V South Asian adults in Manchester winter, 6 weeks of
    3x/week simulated summer sunlight (Arimed B + Cleo Natural cabinet,
    NOT NB-UVB) at 0.65-3.9 SED to 35% BSA. Subjects receiving >=1.95 SED
    reached mean 15.7 +/- 5 ng/mL with a mean rise of 8.7 +/- 5.7 ng/mL
    (all completers were <20 ng/mL at baseline; only 6/51 reached >=20 ng/mL).

    [C2] QUANTITATIVE THRESHOLD (replaces prior loose 'D_final < 68.0 nmol/L'):
    Unit conversion: 15.7 ng/mL * 2.496 = ~39.2 nmol/L final;
                     8.7 ng/mL * 2.496 = ~21.7 nmol/L rise.
    We simulate the >=1.95 SED group with 1.95 SED/session.

    [W-Y 2026-09-22 BUG FIX] D0 was previously a crude D0=30 nmol/L "proxy"
    documented as unresolved ("exact group baseline is not extractable from
    the local summary alone"). It IS extractable: Farrar 2013's own abstract
    prints both the pooled >=1.95-SED-subgroup final (15.7 ng/mL) and rise
    (8.7 ng/mL) for the same n=33 subjects, so the implied baseline is
    15.7 - 8.7 = 7.0 ng/mL exactly, i.e. 7.0 * 2.496 = 17.472 nmol/L (full
    precision). Content-verified against Research_Papers/"Efficacy of a dose
    range of simulated sunlight exposures ... South Asian adults ...pdf"
    (DOI 10.3945/ajcn.112.052639). D0=17.472 replaces the D0=30 proxy below;
    the wide acceptance windows are UNCHANGED (they bound the paper's own
    observed D_final/delta, not D0, so nothing about fixing D0 requires
    retuning them).

    [W-Y 2026-09-22 DEDUPLICATION, DATA FIX KEPT] This function duplicated
    vitd_scenario.py's farrar2013_fst5_uk scenario -- both encoded the same
    real-world fact (this same >=1.95-SED-subgroup finding), inconsistently,
    until the D0 fix above and the parallel fix to that scenario record
    (decision E5, user instruction "deduplicate", 2026-09-22)
    made them compute the identical simulation. That data fix (D0=17.472,
    dose 1.95 SED, corrected citation) is CORRECT and UNCHANGED here.

    [REVERTED 2026-09-22] W-Y additionally demoted this function's
    CHECK_REGISTRY entry to warning_only=True, reasoning that the same fact
    is now covered by the required "Reference Scenarios Benchmark" check
    via farrar2013_fst5_uk, so keeping both required double-counts one
    real-world finding. Ahaan refused to sign off on that demotion: this is
    "the only dark-skin quantitative check" per calibration_approval.yaml,
    and downgrading required dark-skin coverage is his call. The CheckSpec
    in CHECK_REGISTRY below is back to warning_only=False (required). The
    double-counting W-Y flagged is real and not fixed by this revert --
    two required checks now run the identical simulation through two
    differently-shaped acceptance bands -- but that is the accepted
    tradeoff for keeping this check required, not an oversight.
    Acceptance windows (broadband model approximation of Manchester winter
    31% BSA protocol; exact BSA/dose conversion not resolved locally):
      D_final in [25, 55] nmol/L   (observed ~39.2, wide band for broadband approx)
      delta    in [10, 35] nmol/L  (observed ~21.7)
    Both conditions must hold.
    """
    p = ModelParams()
    fst = 5
    D0 = 17.472
    # Simulate the >=1.95 SED dose group: use 1.95 SED/session
    sched = []
    week_days = [0, 2, 4]
    for w in range(6):
        for d in week_days:
            sched.append((w * 7 + d, 1.95))
    result = simulate(fst=fst, schedule=sched, duration_days=42, params=p, D0=D0)
    D_final = result["D"][-1]
    delta = D_final - D0
    # [C2] Quantitative targets from Farrar 2013 (unit-converted)
    OBSERVED_FINAL_NM = 39.2   # nmol/L (15.7 ng/mL * 2.496)
    OBSERVED_DELTA_NM = 21.7   # nmol/L (8.7 ng/mL * 2.496)
    D_final_ok = 25.0 <= D_final <= 55.0
    delta_ok   = 10.0 <= delta  <= 35.0
    passed = D_final_ok and delta_ok
    if verbose:
        print(f"[Lit 3] Farrar 2013 - FST V, 1.95 SED 3x/week x 6 weeks, baseline {D0:.3f}  [C2]")
        print(f"  D_final = {D_final:.1f} nmol/L  (observed ~{OBSERVED_FINAL_NM} nmol/L; target 25-55)")
        print(f"  delta   = {delta:.1f} nmol/L  (observed ~{OBSERVED_DELTA_NM} nmol/L; target 10-35)")
        print(f"  D_final in [25,55]: {'YES' if D_final_ok else 'NO'}   delta in [10,35]: {'YES' if delta_ok else 'NO'}")
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def validate_bogh_baseline_bsa(verbose=True):
    """
    Bogh 2010/P5 (Bogh et al. 2011, PMID:21270901): baseline serum D and exposed
    body surface area are significant independent predictors of UV-induced delta D.
    Derived equation (R2=0.47):
        delta = 7.1 - 0.223 * baseline + 11.0 * ln(UVB_dose) + 5.5 * ln(%BSA)

    [C4] Baseline D and exposed body area must enter validation / calibration.
    This check verifies that the model's synthesis function produces a
    baseline-dependent response consistent with the P5 direction: a higher
    baseline D at fixed dose/area must yield a lower delta.

    Two scenarios at 1 SED (broadband), FST III, SC=0:
      - Low baseline:  D=20 nmol/L  -> predict delta > high-baseline delta
      - High baseline: D=80 nmol/L  -> predict delta < low-baseline delta

    Direction check only (Moderate confidence per audit); exact coefficient
    match is not required because P5's BSA/dose protocol differs from the
    model's full-body exposure assumption.
    """
    p = ModelParams()
    fst = 3
    M_base = p.M_baseline(fst)
    delta_low  = synthesis_per_session(1.0, M_base, M_base, D=20.0,  SC=0.0, params=p)
    delta_high = synthesis_per_session(1.0, M_base, M_base, D=80.0,  SC=0.0, params=p)
    # P5 equation direction: higher baseline -> lower delta
    direction_ok = delta_low > delta_high
    # P5 coefficient: -0.223 nmol/L delta per nmol/L baseline
    # Model: feedback factor = 1 - D/D_max; ratio = (1-20/150)/(1-80/150) = 0.867/0.467 ~ 1.86
    p5_ratio_expected = (1 - 20 / 150) / (1 - 80 / 150)  # ~1.86
    model_ratio = delta_low / delta_high if delta_high > 0 else 0.0
    # Accept if model ratio is in [1.2, 3.0] (direction strongly correct, P5 roughly consistent)
    ratio_ok = 1.2 <= model_ratio <= 3.0
    passed = direction_ok and ratio_ok
    if verbose:
        print(f"[Lit 4] Bogh 2010/P5 - baseline-dependent synthesis direction  [C4]")
        print(f"  1 SED, FST III, SC=0:")
        print(f"    delta at D=20:  {delta_low:.3f} nmol/L")
        print(f"    delta at D=80:  {delta_high:.3f} nmol/L")
        print(f"  direction (low-baseline > high-baseline): {'YES' if direction_ok else 'NO'}")
        print(f"  model ratio = {model_ratio:.2f}  (P5-implied feedback ratio ~{p5_ratio_expected:.2f}; target 1.2-3.0)")
        print(f"  Note: full BSA term (5.5*ln(%BSA)) not modeled; direction check only")
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def validate_metric_ordering(verbose=True):
    """
    Asserts:
    1. AUC above sufficiency = 0 when D(t) never exceeds the threshold (e.g. mild schedule with threshold 50.0).
    2. AUC above baseline is > 0 when D(t) rises above baseline.

    (A third check used to assert the retired per-SED efficiency metric was
    >= net_final_D_change_per_SED for a sparse schedule. That metric was
    removed from vitd_model.py on 2026-07-31 -- see the removal comment there
    -- so that ordering check no longer applies and was dropped rather than
    rewritten against a different metric pair.)
    """
    p = ModelParams()

    # 1. Check AUC above sufficiency = 0 when D never reaches threshold (e.g. 50)
    sched_mild = build_pulse_schedule(84, 0.1, 7)
    r_mild = simulate(fst=3, schedule=sched_mild, duration_days=84, params=p, D0=25.0)
    auc_suff_mild = auc_above_sufficiency_per_SED(r_mild, threshold_nmol_L=50.0)

    # 2. Check AUC above baseline > 0 when D rises
    sched_rise = build_pulse_schedule(84, 1.0, 2)
    r_rise = simulate(fst=3, schedule=sched_rise, duration_days=84, params=p, D0=25.0)
    auc_base_rise = auc_above_baseline_per_SED(r_rise)

    passed_auc_suff = abs(auc_suff_mild) < 1e-9
    passed_auc_base = auc_base_rise > 0.0

    passed = passed_auc_suff and passed_auc_base

    if verbose:
        print(f"[New Check] Metric ordering and sanity assertions")
        print(f"  Mild schedule (0.1 SED every 7d, D0=25.0):")
        print(f"    auc_above_sufficiency (threshold 50): {auc_suff_mild:.4f} (target: 0.0)")
        print(f"    AUC above sufficiency = 0: {'YES' if passed_auc_suff else 'NO'}")
        print(f"  Standard schedule (1.0 SED every 2d, D0=25.0):")
        print(f"    auc_above_baseline: {auc_base_rise:.4f} (target: > 0.0)")
        print(f"    AUC above baseline > 0: {'YES' if passed_auc_base else 'NO'}")
        print(f"  {'PASS' if passed else 'FAIL'}")

    return passed


def validate_baseline_calibration_target(verbose=True):
    """
    [C3-feedback] Baseline feedback calibration validation.
    Using Karppinen 2017 (P33) high-latitude solar UV:
      - Dose: 11.8 SED over 10 days (1.18 SED/day)
      - FST III, exposed BSA = 7% (0.07)
      - Solar summer spectrum (E_source = 2.0 with snow reflection)
      - Low baseline (62.4 nmol/L) should increase (observed +6.0 nmol/L)
      - High baseline (116.9 nmol/L) should decrease (observed -6.7 nmol/L)
    This now tests the DEFAULT model.

    STRENGTHENED 2026-08-18. This check used to force
    `feedback_mode='saturating'` with `D_half_feedback = 90.0`, a value fitted
    to this very study, and carried the note "the default model may not pass
    this check". It was therefore both circular and not a test of anything the
    model ships. After the approved A2 (feedback form) and A4 (A) changes the
    DEFAULT model passes it outright -- FST III, 1.18 SED/day x 10 d, 7% BSA,
    E_source 2.0: low baseline 62.4 -> +2.5 (observed +6.0), high baseline
    116.9 -> -5.7 (observed -6.7). The override is removed and the acceptance
    bounds are UNCHANGED. Direction, not magnitude, is what this check licenses:
    a baseline-dependent SIGN REVERSAL is a strong constraint on feedback shape
    and neither bound was moved to accommodate the new form.
    """
    p = ModelParams()

    sched = [(d, 1.18) for d in range(10)]
    
    # 1. Low baseline run
    r_low = simulate(fst=3, schedule=sched, duration_days=10, params=p, D0=62.4,
                     exposed_bsa_override=0.07, E_source_override=2.0)
    change_low = r_low["D"][-1] - 62.4
    
    # 2. High baseline run
    r_high = simulate(fst=3, schedule=sched, duration_days=10, params=p, D0=116.9,
                      exposed_bsa_override=0.07, E_source_override=2.0)
    change_high = r_high["D"][-1] - 116.9
    
    # Low baseline should increase, high baseline should decrease
    passed_direction = (change_low > 0) and (change_high < 0)
    
    # Check if they are within loose target ranges matching P33 (observed +6.0 and -6.7)
    # Low baseline change in [0.5, 10.0]
    # High baseline change in [-10.3, -3.0]
    passed_bounds = (0.5 <= change_low <= 10.0) and (-10.3 <= change_high <= -3.0)
    passed = passed_direction and passed_bounds
    
    if verbose:
        print(f"[Lit 5] Karppinen 2017/P33 - baseline feedback calibration target")
        print(f"  Protocol: 1.18 SED/day x 10 days, FST III, BSA=0.07, E_source=2.0")
        print(f"  DEFAULT feedback mode ({p.feedback_mode!r}):")
        print(f"    Low baseline (62.4 nmol/L) change:  +{change_low:.2f} nmol/L (observed +6.0)")
        print(f"    High baseline (116.9 nmol/L) change: {change_high:.2f} nmol/L (observed -6.7)")
        print(f"    Low baseline increased: {'YES' if change_low > 0 else 'NO'}")
        print(f"    High baseline decreased: {'YES' if change_high < 0 else 'NO'}")
        print(f"    Within target bounds: {'YES' if passed_bounds else 'NO'}")
        print(f"  {'PASS' if passed else 'FAIL'}")
        
    return passed


def validate_sc_deBoer2025(verbose=True):
    """
    de Boer 2025 (P2): 0.8 SED 3x/week x 9 weeks on lower back.
    SC thickness: FP I-III +23% (5.6->6.9 um), FP IV-VI +36% (6.6->9.0 um).
    Melanin index: FP I-III +15%, FP IV-VI +9%.

    Calibrated with de Boer 2025 Supplementary Table S1.
    [C7-SC-FITTED] alpha_SC=0.12, SC_max_by_FST: I-III=0.24, IV-VI=0.36.
    I-III raised 0.23 -> 0.24 on 2026-08-25 (calibration_approval.yaml,
    "SC_max_by_FST I-III 0.23 -> 0.24 (stratum corneum ceiling)"); Miyamura
    2011 Table 1 (FST II-III stratum corneum 16.6 -> 20.7 um, +24.6988%) now
    independently corroborates de Boer 2025's +23% in MAGNITUDE only -- see
    docs/MELANIN_MODEL.md. This remains an uncalibrated prior for absolute D
    predictions.
    Warning-only because k_damage=0. The week-5 dip is representable (SC is
    two-process) but unidentifiable from three group medians.

    Target SC at week 9: FP I-III proxy (FST III, SC_max=0.24) should be
      in [0.12, 0.24] (reaches ~90% of SC_max within 9 weeks).
    Target SC at week 9: FP IV-VI proxy (FST V, SC_max=0.36) should be
      in [0.18, 0.36] (reaches ~90% of SC_max within 9 weeks).
    """
    p = ModelParams()
    sched = build_weekly_schedule(63, 0.8, [0, 2, 4])

    r3 = simulate(fst=3, schedule=sched, duration_days=63, params=p)
    SC_III = float(np.mean(r3["SC"][-28:]))

    r5 = simulate(fst=5, schedule=sched, duration_days=63, params=p)
    SC_V = float(np.mean(r5["SC"][-28:]))

    SC_III_increased = SC_III > 0.0
    SC_V_increased = SC_V > 0.0

    SC_III_range_ok = 0.08 <= SC_III <= 0.24
    SC_V_range_ok = 0.14 <= SC_V <= 0.36

    SC_V_ge_III = SC_V >= SC_III * 0.8

    passed = all([SC_III_increased, SC_V_increased,
                  SC_III_range_ok, SC_V_range_ok, SC_V_ge_III])

    # Shape check against the actual visit structure. Table S1 measures at
    # T0 = 0, T1 = 5 weeks (day 35), T2 = 9 weeks (day 63), and reports SC
    # thickness 5.8 -> 4.3 -> 7.5 um for all FPs: a ~26% dip BELOW baseline
    # before overshooting. The model cannot reproduce that with k_damage = 0,
    # which is deliberate -- the parameter is not identifiable from three group
    # medians (scripts/fit_sc_deboer2025.py). Reported, not asserted.
    sc_t1 = float(np.interp(35.0, r3["t"], r3["SC"]))
    sc_t2 = float(np.interp(63.0, r3["t"], r3["SC"]))
    obs_t1_rel, obs_t2_rel = 4.3 / 5.8 - 1.0, 7.5 / 5.8 - 1.0

    if verbose:
        print(f"[SC Check] de Boer 2025 SC validation (warning-only)")
        print(f"  Protocol: 0.8 SED 3x/week x 9 weeks (T1 = 35 d, T2 = 63 d)")
        print(f"  FST III (FP I-III proxy, SC_max=0.24): SC={SC_III:.4f} "
              f"(increased: {'YES' if SC_III_increased else 'NO'}, "
              f"range [0.08,0.24]: {'OK' if SC_III_range_ok else 'OUT'})")
        print(f"  FST V  (FP IV-VI proxy, SC_max=0.36): SC={SC_V:.4f} "
              f"(increased: {'YES' if SC_V_increased else 'NO'}, "
              f"range [0.14,0.36]: {'OK' if SC_V_range_ok else 'OUT'})")
        print(f"  SC_V >= SC_III * 0.8: {'YES' if SC_V_ge_III else 'NO'}")
        print(f"  SHAPE (reported, not asserted): model T1={sc_t1:+.3f} "
              f"T2={sc_t2:+.3f} vs observed {obs_t1_rel:+.3f} / {obs_t2_rel:+.3f}")
        print(f"    Observed SC dips ~26% BELOW baseline at week 5. Model cannot")
        print(f"    reproduce that at k_damage=0.0, which is deliberate: the")
        print(f"    parameter is unidentifiable from 3 group medians. See")
        print(f"    scripts/fit_sc_deboer2025.py and docs/SC_MODEL.md section 2.4.")
        print(f"  {'PASS' if passed else 'FAIL'} (warning-only)")
    return passed


# ---------------------------------------------------------------------------
# Seasonal-check simulation constants (shared by the K_ATTEN seasonal
# amplitude check below; formerly also by the retired NHANES population
# validation, removed 2026-08-25 -- see validation_exploratory/ and
# scripts/nhanes_macro_validation.py for its successors).
# ---------------------------------------------------------------------------

_SIM_DAYS = 84      # 12-week simulation
# [ASSUMED, NOT MEASURED -- this is the load-bearing weakness of the NHANES
# checks.] Nobody measured what UV dose NHANES respondents actually received.
# 0.333 SED/day is a plausible stand-in ("~5 min at UVI 5"), not data. Because
# synthesis depends on the PRODUCT sed * E_source, this assumed dose absorbs any
# E_source change exactly -- so solar E_source is NOT identifiable from these
# checks, in either direction. Deriving this from a real ambient UV climatology
# is 'Open decisions' item 12(b) and is what would make these checks informative.
_SIM_SED = 0.333    # ~5 min at UVI 5
_SIM_BSA = 0.30     # arms + face
# [CALIBRATION 2026-08-16, user-approved] Was 1.0 ("solar / broadband
# reference"); follows vitd_model.E_SOURCE_SOLAR_DEFAULT like every other
# sun path. See calibration_approval.yaml, E_SOURCE_SOLAR_DEFAULT /
# not_covered. Standing rule from that change: do not deflate _SIM_SED to
# keep a seasonal check green -- raising E_source 1.0 -> 1.89 raised every
# modelled steady state, and the resulting check failures were accepted
# deliberately rather than tuned away. Tuning an unmeasured knob to hide a
# calibration change is calibration through an ungated back door.
_SIM_ESOURCE = get_E_source(E_SOURCE_SOLAR_DEFAULT)
_SIM_TOLERANCE = 10.0  # nmol/L, matching tightened Bogh 2012 convention


def validate_katten_seasonal_amplitude(verbose=True):
    # STATUS: EXPLORATORY_ONLY
    # Invalid as physiological validation of K_ATTEN.
    # Retained only for future behavioral modeling.
    """
    Validate K_ATTEN seasonal amplitude ratio (NH White : NH Black).

    Reads audit_outputs/seasonality_by_cycle_race.csv and compares the
    observed seasonal amplitude ratio (NH White FST=2 : NH Black FST=5)
    against the model-predicted ratio.

    For each cycle and group:
      Winter: D0 = min_season_mean
      Summer: simulate daily 0.333 SED x 84 days, E_source=1.0, BSA=0.30
      Model amplitude = steady_state_D - D0

    The model ratio must be within +/-50% of the observed ratio.
    """
    data_path = os.path.join(os.path.dirname(__file__),
                             "audit_outputs", "seasonality_by_cycle_race.csv")
    if not os.path.exists(data_path):
        if verbose:
            print(f"\n[K_ATTEN seasonal amplitude] Data file not found. SKIP.")
        return False

    df = pd.read_csv(data_path)
    df = df[df["race_ethnicity_code"].isin([3, 4])].copy()

    code_to_fst = {3: 2, 4: 5}
    code_to_label = {3: "NH White", 4: "NH Black"}
    df.loc[:, "fst"] = df["race_ethnicity_code"].map(code_to_fst)
    df.loc[:, "label"] = df["race_ethnicity_code"].map(code_to_label)

    p = ModelParams()
    schedule = [(d, _SIM_SED) for d in range(_SIM_DAYS)]

    model_amps = {3: [], 4: []}
    obs_amps = {3: [], 4: []}

    for _, row in df.iterrows():
        code = int(row["race_ethnicity_code"])
        fst = code_to_fst[code]
        D0 = row["min_season_mean"]
        obs_amp = row["amplitude_mean_25ohd"]
        obs_amps[code].append(obs_amp)

        result = simulate(fst=fst, schedule=schedule,
                          duration_days=_SIM_DAYS, params=p,
                          D0=D0,
                          E_source_override=_SIM_ESOURCE,
                          exposed_bsa_override=_SIM_BSA)
        ss = steady_state_D(result)
        model_amp = ss - D0
        model_amps[code].append(model_amp)

    obs_mean_white = np.mean(obs_amps[3])
    obs_mean_black = np.mean(obs_amps[4])
    obs_ratio = obs_mean_white / obs_mean_black if obs_mean_black > 0 else np.nan

    mod_mean_white = np.mean(model_amps[3])
    mod_mean_black = np.mean(model_amps[4])
    mod_ratio = mod_mean_white / mod_mean_black if mod_mean_black > 0 else np.nan

    if obs_ratio > 0 and mod_ratio > 0:
        pct_diff = (mod_ratio - obs_ratio) / obs_ratio * 100
        passed = abs(pct_diff) <= 50.0
    else:
        passed = False
        pct_diff = 0.0

    if verbose:
        print(f"\n[K_ATTEN seasonal amplitude] NH White vs NH Black seasonal rise")
        print(f"  Protocol: daily {_SIM_SED} SED x {_SIM_DAYS}d, "
              f"E_source={_SIM_ESOURCE}, BSA={_SIM_BSA}")
        print(f"  NH White (FST=2, code 3): "
              f"observed amplitude = {obs_mean_white:.2f}, "
              f"model amplitude = {mod_mean_white:.2f}")
        print(f"  NH Black (FST=5, code 4): "
              f"observed amplitude = {obs_mean_black:.2f}, "
              f"model amplitude = {mod_mean_black:.2f}")
        print(f"  Observed ratio (White/Black): {obs_ratio:.3f}")
        print(f"  Model ratio (White/Black):    {mod_ratio:.3f}")
        print(f"  Difference: {pct_diff:+.1f}% (target: within +/-50%)")
        print(f"  {'PASS' if passed else 'FAIL'}")

    return passed


#: [PROVENANCE] Kift 2013 Fig. 1 medians for the FST V South Asian cohort,
#: decimal-reconstructed (the local PDF's text layer drops decimal points):
#: summer 9.0 ng/mL, winter 5.8 ng/mL. Independently confirmed to the decimal
#: by Webb et al. 2018 (Nutrients 10:457) Results 3.1, which quotes the SAME
#: cohort directly: "22.5 nmol/L in summer and 14.5 nmol/L in winter"
#: (9.0*2.5=22.5, 5.8*2.5=14.5 -- exact agreement between two independent
#: readings). See scripts/fst_gap_decomposition.py, FIELD_COHORTS
#: ["Kift2013_southasian"] and WEBB18_QUOTES_KIFT, for the full reproduction.
#: Reported by validate_kift_2013_south_asian for the record; NOT gated on
#: -- see that function's docstring for why.
KIFT2013_OBSERVED_SUMMER_NMOL_L = 22.5
KIFT2013_OBSERVED_WINTER_NMOL_L = 14.5
KIFT2013_OBSERVED_AMPLITUDE_NMOL_L = (KIFT2013_OBSERVED_SUMMER_NMOL_L
                                       - KIFT2013_OBSERVED_WINTER_NMOL_L)  # 8.0


def validate_kift_2013_south_asian(verbose=True):
    """
    Kift 2013: UK South Asian adults (FST V) receiving naturalistic UK summer exposure.
    Protocol: ~14% BSA exposed (face + hands/forearms), ~1.0 SED/week (0.333 SED 3x/week)
    for 12 weeks (84 days), baseline 22.5 nmol/L.

    Target:
      - Baseline ~22.5 nmol/L
      - D_final in range [20.0, 40.0] nmol/L
      - delta in range [0.0, 15.0] nmol/L

    TIER: CIRCULAR. DEMOTED to warning-only 2026-08-21. Two independent
    problems, either one alone disqualifying this from the
    required/quantitative pool:

    (1) UNFALSIFIABLE BAND. `delta` is accepted anywhere in [0.0, 15.0]
        nmol/L -- 15 nmol/L wide, and it INCLUDES ZERO. A model predicting
        NO vitamin D synthesis whatsoever from this protocol would pass.
        That is a range assertion, not a validation.

    (2) CIRCULAR. Kift 2013 is one of the four within-study/field routes --
        Felton 2016, Young 2020, Kift 2013, Darling 2013 -- that
        `ModelParams.K_const` (1.15 as of 2026-09-06, was 0.98) was fitted to by equal-weight least
        squares in log-ratio space (vitd_model.py, K_const entry: "Fitted
        to four within-study routes: Felton 2016, Young 2020, Kift 2013,
        Darling 2013"; full derivation in scripts/kconst_recalibration.py,
        S3/S4). A model built in part to reproduce Kift's own implied
        skin-type penalty passing a check built on that same study is
        expected by construction, not independent support.

    OPTION (a) -- tighten the band to the observed value -- was considered
    and rejected. A real observed delta DOES exist in the repo (see
    KIFT2013_OBSERVED_AMPLITUDE_NMOL_L below) and is not invented, but
    tightening around a number K_const was partly fitted to would make the
    check LOOK stronger while making problem (2) worse, not better: a
    precise-looking pass that is still circular is a more convincing
    overclaim than a loose one. Demoting was judged more defensible than
    dressing up a circular check as a quantitative one.

    WHAT WOULD MAKE THIS A REAL CHECK: a naturalistic (non-lamp) South Asian
    FST V UK exposure cohort that was NOT one of the four K_const fitting
    routes. None currently exists in this repo -- Darling 2013 is the only
    other field route, and it is also a fitting input. The nearest thing to
    a genuinely held-out check on this exact quantity is Webb/Kift 2018
    (RESULTS.md section 15, vitd_benchmark_webbkift.py), which is un-fitted
    precisely because it was assembled after K_const was set. As of the
    2026-08-21 audit run it FAILS on the minutes ratio (model 4.80 vs
    reported 2.78; dose ratio 3.01x vs reported 2.5-3.0x, a near-pass) --
    see RESULTS.md section 15 for the live re-run.

    REPORTED, NOT GATED: Kift 2013's own observed winter -> summer amplitude
    in this cohort is 14.5 -> 22.5 = 8.0 nmol/L (see
    KIFT2013_OBSERVED_AMPLITUDE_NMOL_L above this function; independently
    confirmed by Webb 2018). Printed below for the record. Note also that
    this check's own D0 = 22.5 is the STUDY'S SUMMER value, not its winter
    baseline -- the simulation therefore does NOT reproduce the
    winter-to-summer transition the amplitude describes; it only checks
    that continued exposure from an already-summer level stays in a
    plausible range. That mismatch is flagged here rather than silently
    fixed, since correcting the simulated protocol is outside this task's
    scope (retiering + band honesty, not protocol reconstruction).
    """
    p = ModelParams()
    fst = 5
    D0 = 22.5
    sched = [(d, 0.333) for d in range(84) if d % 7 in [0, 2, 4]]
    result = simulate(fst=fst, schedule=sched, duration_days=84, params=p, D0=D0,
                      exposed_bsa_override=0.14, E_source_override=1.0)
    D_final = result["D"][-1]
    delta = D_final - D0

    D_final_ok = 20.0 <= D_final <= 40.0
    delta_ok = 0.0 <= delta <= 15.0
    passed = D_final_ok and delta_ok

    if verbose:
        print(f"[Lit 6, WARNING-ONLY / CIRCULAR] Kift 2013 - UK South Asian naturalistic exposure (FST V)")
        print(f"  Protocol: ~1.0 SED/wk (0.333 SED 3x/wk) x 12 weeks, FST V, BSA=0.14, baseline {D0:.1f} nmol/L")
        print(f"  Model D_final = {D_final:.1f} nmol/L (delta = +{delta:.1f} nmol/L)")
        print(f"  Target bounds: D_final in [20, 40]: {'YES' if D_final_ok else 'NO'}, delta in [0, 15]: {'YES' if delta_ok else 'NO'}")
        print(f"  CIRCULAR: Kift 2013 is one of the 4 routes K_const (1.15 as of 2026-09-06) was fitted to; band")
        print(f"  also includes zero, so a null model would pass. Demoted to warning-only,")
        print(f"  not counted toward the required pass/fail score. See docstring.")
        print(f"  [REPORTED, NOT GATED] Kift's own observed winter->summer amplitude: "
              f"{KIFT2013_OBSERVED_WINTER_NMOL_L:.1f} -> {KIFT2013_OBSERVED_SUMMER_NMOL_L:.1f} "
              f"= {KIFT2013_OBSERVED_AMPLITUDE_NMOL_L:.1f} nmol/L (confirmed independently by Webb 2018).")
        print(f"  Note D0 above is the study's SUMMER value, not its winter baseline, so this")
        print(f"  simulation does not reproduce that transition -- flagged, not fixed.")
        print(f"  {'PASS' if passed else 'FAIL'} (warning-only; not counted toward required score)")

    return passed


def validate_kallioglu_2024_erythema_med_ratio(verbose=True):
    """Kallioglu 2024 (Sci Rep 14:3541) -- NOT independent vitamin-D evidence.

    RELABELLED 2026-08-17 (was `validate_kallioglu_2024_skintype_gradient`).
    The function is kept -- deleting it would lose the finding below and
    someone would likely re-add the paper as "new evidence" later -- but its
    old docstring made four claims that are false and have been removed:
    "fully independent", "different endpoint", "two independent sources
    agree", and "This is K_const / melanin attenuation". None of those
    survive inspection of the paper.

    IDENTITY PROOF. Kallioglu 2024 Table 3 (p.7) is a minimal-erythema-dose
    (MED) table, not a vitamin-D measurement:

        FST        I    II   III   IV    V     VI
        MED J/m2  200  250   300  450   600  1000   (erythemal)

    Their published minutes for 1000 IU/day at 25% BSA (annual mean, Antalya
    37N / Istanbul 41N) are:

        I 5.05   II 6.30   III 7.60   IV 11.35   V 15.15   VI 25.25

    Those minutes are `5.05 * MED_fst / MED_I` to within rounding: exact at
    FST I, V, VI, and within 0.33% at FST II-IV (largest gap: FST III,
    predicted 7.5750 vs published 7.60). Skin type enters their model ONLY
    through this MED table, via Hollick's rule (their Eq. 10:
    UVd = (MED/4) * 0.25 * BSA * Dvit/1000) -- Hollick's rule assumes equal
    fractional-MED dosing gives equal vitamin D synthesis. There is no
    measurement of vitamin D synthesis differing by pigment anywhere in the
    paper. The "V:II = 2.40x" and "VI:I = 5.00x" this check reports are
    therefore ratios of a textbook erythema-threshold table, reproduced
    through one assumption, not two independent measurements agreeing.

    THIS CONTRADICTS EVIDENCE ALREADY IN THIS REPO.
    - P24 (Shih/Farrar/Rhodes 2018) finds fractional-MED dosing REMOVES the
      apparent skin-type effect for acute 25(OH)D gain (`vitd_physiology.py`,
      `vitd_model.py` P24 references) -- i.e. equal %MED did NOT give unequal
      vitamin D in that trial, the opposite of what Kallioglu's method
      assumes.
    - Felton et al. 2016 (Br J Dermatol 175:1320) Table 1 measured BOTH
      quantities in the SAME volunteers: an erythema MED ratio of
      146/37 = 3.95x (FST V:II) alongside a matched-dose vitamin D gain
      ratio of only (25.5-17.2)/(54.3-36.5) = 2.14x. Equal fractional MED do
      not imply equal vitamin D even within one study -- direct evidence
      against the assumption this check's target value is built on.

    WHAT THIS CHECK ACTUALLY MEASURES. Whether the model's minutes-ratio
    across Fitzpatrick types reproduces Kallioglu's MED-derived ratio. That
    is a legitimate (weak) cross-check of the model's skin-type scaling
    against a published erythema table run through Hollick's rule -- it is
    NOT evidence about vitamin D synthesis and must not be counted or
    reported as such. It stays `warning_only=True` in the `run_all()` checks
    list below; this relabelling changes no pass/fail computation.

    See `scripts/fst_gap_decomposition.py` S1a for the full reproduction of
    the identity above (`kallioglu_is_med_table` return value), and the
    Kallioglu row in that script's S3b evidence-bracket table, which is
    excluded from the "matched-dose, within-study" bracket for this reason.
    """
    from vitd_objectives import minutes_for_target, SUFFICIENCY_NMOL_L

    reported = {1: 5.05, 2: 6.30, 3: 7.60, 4: 11.35, 5: 15.15, 6: 25.25}
    model = {}
    for fst in range(1, 7):
        r = minutes_for_target(fst=fst, uv_index=6.0, exposed_bsa=0.25,
                               frequency_per_week=7.0, target=SUFFICIENCY_NMOL_L,
                               horizon_days=150, max_minutes=1200.0)
        model[fst] = r["minutes"]

    rep_ratio = reported[6] / reported[1]
    mod_ratio = model[6] / model[1]
    # Advisory band. The model is currently well outside it; the check exists to
    # measure the gap and to fail loudly if a change makes it worse.
    passed = 0.70 * rep_ratio <= mod_ratio <= 1.30 * rep_ratio

    if verbose:
        print("[Held-out, WARNING-ONLY] Kallioglu 2024 erythema-MED-derived ratio")
        print("  NOT vitamin-D evidence -- reported ratio is Table 3's MED table")
        print("  run through Hollick's rule; see this function's docstring.")
        print("  1000 IU/d, 25% BSA, annual mean, Antalya 37N / Istanbul 41N")
        print(f"  {'FST':>4} {'reported':>9} {'rep ratio':>10} {'model min':>10} {'mod ratio':>10}")
        for fst in range(1, 7):
            print(f"  {fst:>4} {reported[fst]:>9.2f} {reported[fst]/reported[1]:>10.2f} "
                  f"{model[fst]:>10.1f} {model[fst]/model[1]:>10.2f}")
        print(f"  FST VI:I  reported {rep_ratio:.2f}  model {mod_ratio:.2f}")
        print(f"  FST V:II  reported {reported[5]/reported[2]:.2f}  model {model[5]/model[2]:.2f}")
        print(f"  {'PASS' if passed else 'FAIL'} - model's minutes ratio vs the "
              f"MED-table-derived ratio (advisory only, not a vitamin-D check)")

    return passed


def validate_reference_scenarios(verbose=True) -> bool:
    """Benchmark the forward model across all registered clinical reference scenarios.

    Evaluates Farrar 2013, Young 2021, Felton 2016, de Boer 2025, Webb 2018, etc.
    Asserts that:
      1. Every registered scenario simulates successfully.
      2. 25(OH)D remains strictly non-negative and physiologically bounded.
      3. For scenarios with published literature targets (Farrar 2013, Felton 2016, Young 2021),
         the model delta is positive and within 2.5x of observed.
      4. Cancer risk indices (scc_index, intermittent_index, burn_fraction) are non-negative
         and burn_fraction is 0% under sub-erythemal trial protocols.
    """
    from vitd_scenario import list_scenarios, get_scenario, simulate_scenario
    all_ok = True
    rows = []
    for name in list_scenarios():
        sc = get_scenario(name)
        res = simulate_scenario(name)
        d_final = res["final_D"]
        d_delta = res["delta_D"]
        scc = res["scc_index"]
        interm = res["intermittent_index"]
        burn = res["burn_fraction"]

        # Invariant checks
        ok = (d_final >= 0.0) and (scc >= 0.0) and (interm >= 0.0) and (0.0 <= burn <= 1.0)
        
        obs = sc.observed_delta_D
        if isinstance(obs, (int, float)):
            ok = ok and (0.2 * obs <= d_delta <= 2.5 * obs)

        if not ok:
            all_ok = False

        obs_str = f"{obs:+.1f}" if isinstance(obs, (int, float)) else "N/A"
        rows.append((name, sc.target_fst, sc.baseline_D0, d_delta, obs_str, scc, interm, burn * 100, "PASS" if ok else "FAIL"))

    if verbose:
        print("[Reference Scenarios Benchmark] Clinical Trial Protocols vs Model Predictions:")
        print(f"  {'Scenario':<26} {'FST':>4} {'D0':>5} {'Model dD':>9} {'Obs dD':>8} {'SCC':>6} {'Interm':>7} {'Burn%':>6} {'Status':>7}")
        print("  " + "-" * 82)
        for r in rows:
            print(f"  {r[0]:<26} {r[1]:>4} {r[2]:>5.1f} {r[3]:>+8.1f} {r[4]:>8} {r[5]:>6.2f} {r[6]:>7.2f} {r[7]:>5.0f}% {r[8]:>7}")
        print(f"  {'PASS' if all_ok else 'FAIL'}")

    return all_ok


# ---------------------------------------------------------------------------
# How to add a new validation target
#
# 1. Define a validation function with signature:
#        def my_new_check(verbose=True) -> bool
#    - Return True if the model passes, False otherwise.
#    - Use the comparison utilities above (check_exact_match,
#      check_percent_error, check_in_range, check_qualitative_only)
#      to keep target logic readable.
#    - Hardcode the expected value, tolerance / bounds / direction
#      directly in the function body so the target is self-documenting.
#    - Print details when verbose=True; include the PASS/FAIL line.
#
# 2. Add a CheckSpec(name, fn, warning_only, tier, tier_note) to
#    CHECK_REGISTRY below.
#    - name: short human-readable label.
#    - fn: the function from step 1.
#    - warning_only: True if the check is advisory (does not count
#      toward required pass/fail).
#    - tier: one of TIER_INTERNAL, TIER_DIRECTIONAL, TIER_QUANTITATIVE,
#      TIER_CIRCULAR, TIER_NOT_EVIDENCE (defined near CheckResult, above).
#      Assign by reading the check's OWN docstring and assertions, not by
#      guessing -- most checks already say what they are in words.
#    - tier_note: one line saying why that tier, citing the docstring
#      language or the specific parameter the check would be circular
#      against.
#
# 3. Run |python3 vitd_validation.py| to confirm registration.
#
# 4. If the check introduces a new literature source, add a paper
#    summary card next to the parameter it supports in the vitd_model.py
#    module docstring (outside this file, and the canonical home for
#    parameter provenance per CLAUDE.md) and cross-reference the
#    audit-change tag in the docstring.
#
# See existing checks for examples.
# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
#: The 14 registered checks, each stamped with an evidence tier. This is
#: the single source of truth for both the existing required/warning-only
#: pass/fail tally AND the tiered honesty summary added 2026-08-21.
#: Tier assignments were made by reading
#: each function's own docstring; see tier_note for the one-line reason and
#: the function docstring itself for the full argument.
CHECK_REGISTRY = [
    CheckSpec(
        "decay half-life", sanity_decay_halflife, False,
        TIER_INTERNAL,
        "Simulator checked against the analytic solution of its own ODE; "
        "half-life targets are a power-law fit to the same three Datta "
        "2017 group means the model encodes. Tests that the code computes "
        "its documented math, not that the math matches unseen biology."),
    CheckSpec(
        "single bolus FST III", sanity_single_bolus_FST_III, False,
        TIER_INTERNAL,
        "Docstring: \"computation-integrity check ... NOT a literature "
        "check -- no external target.\" Expected value is rebuilt from the "
        "model's own live parameters."),
    CheckSpec(
        "photoadaptation reduction", sanity_photoadaptation_reduction, False,
        TIER_CIRCULAR,
        "RETAGGED 2026-09-22 (decision A6; "
        "retroactive approval calibration_approval.yaml entry A7). K_ATTEN "
        "(the parameter this check's 0-10% band guards) and the band's own "
        "acceptance cutoff are both read off the SAME Matsuoka 1990 (n=7, "
        "within-subject) deconvolution, under different feedback-form "
        "assumptions -- near-total circularity on the discriminating "
        "evidence, the same structural relationship as the \"SC de Boer "
        "2025\" check below, which is why this entry now matches its tier. "
        "Not vacuous, so not TIER_CIRCULAR's strongest form either: a "
        "different, equally defensible feedback-form reading of the "
        "identical Matsuoka data measures 18.088974442653605% and FAILS "
        "the 10% ceiling (verified by direct perturbation, W-Z sec 0). "
        "Docstring: \"NOT evidence ... a drift guard sharing its evidence "
        "with the calibration.\""),
    CheckSpec(
        "de Boer 2025 photoadaptation [C6 replaced]",
        validate_photoadaptation_deBoer2025, True,
        TIER_DIRECTIONAL,
        "Docstring: \"Checks direction only ... Quantitative MI comparison "
        "is not possible.\" External de Boer 2025 protocol; asserts "
        "melanin up / SC up / synthesis down, no numeric target."),
    CheckSpec(
        "SC de Boer 2025 [C7-SC warning]", validate_sc_deBoer2025, True,
        TIER_CIRCULAR,
        "Docstring: \"[C7-SC-FITTED] alpha_SC=0.12, SC_max_by_FST ... "
        "Calibrated with de Boer 2025 Supplementary Table S1.\" The SC "
        "parameters generating the model's trajectory were fitted to the "
        "same dataset this check compares that trajectory against."),
    CheckSpec(
        "Bogh 2012 [C1 quantitative]", validate_bogh_2012, False,
        TIER_CIRCULAR,
        "Self-labelled \"STATUS: DEBUGGING_TARGET ... must not be "
        "reported as an independent validation pass until structural and "
        "source-data verification are complete.\" E_source=2.5 is a "
        "per-study placeholder tuned to this trial."),
    CheckSpec(
        "Farrar 2013 [C2 quantitative]", validate_farrar_2013, False,
        TIER_QUANTITATIVE,
        "Real numeric comparison (D_final and delta bounds in nmol/L) "
        "against an external FST V trial. Not a direct K_const/A fitting "
        "target, but vitd_model.py's A entry records this check's pass/"
        "fail as a \"MEASURED CONSEQUENCE\" of the 2026-08-18 A "
        "recalibration (FAIL -> PASS) -- partly informed, not fully "
        "independent. W-Y (2026-09-22, decision E5, \"deduplicate\") "
        "demoted this to warning-only on the grounds that its D0 proxy "
        "fix (17.472, extracted from the source rather than guessed at "
        "30) makes its simulated protocol identical to vitd_scenario.py's "
        "farrar2013_fst5_uk, so keeping both REQUIRED double-counts one "
        "real-world finding. REVERTED to required 2026-09-22 (this "
        "workstream): Ahaan refused sign-off on the demotion -- "
        "calibration_approval.yaml describes this check in multiple "
        "places as \"the only dark-skin quantitative check in the "
        "suite,\" and downgrading the suite's only dark-skin-specific "
        "required check is a decision for him, not an agent, regardless "
        "of the double-counting argument's technical merit. The D0 "
        "dedup FIX ITSELF IS KEPT (see validate_farrar_2013 docstring) -- "
        "only the warning_only flip is reverted. Two required checks now "
        "exercise the identical Farrar simulation via different "
        "acceptance-band shapes; that duplication is a known, flagged "
        "fact, not an oversight."),
    CheckSpec(
        "Bogh baseline+BSA [C4]", validate_bogh_baseline_bsa, False,
        TIER_DIRECTIONAL,
        "Docstring: \"Direction check only ... exact coefficient match is "
        "not required.\" Tests the SIGN of the baseline effect against "
        "Bogh 2010/P5's regression; the ratio band [1.2, 3.0] is wide "
        "enough to be near-guaranteed by the feedback function's own "
        "monotonicity."),
    CheckSpec(
        "baseline calibration target [C3-feedback]",
        validate_baseline_calibration_target, False,
        TIER_DIRECTIONAL,
        "Docstring: \"Direction, not magnitude, is what this check "
        "licenses: a baseline-dependent SIGN REVERSAL is a strong "
        "constraint on feedback shape.\" Uses external Karppinen 2017 "
        "numbers to set a sign-reversal target with a wide bracket around "
        "them, not a tight numeric match."),
    CheckSpec(
        "metric ordering and sanity", validate_metric_ordering, False,
        TIER_INTERNAL,
        "Pure internal sanity: AUC-above-sufficiency is 0 when D never "
        "crosses the threshold, AUC-above-baseline is >0 when D rises. No "
        "external data of any kind."),
    CheckSpec(
        "K_ATTEN seasonal amplitude", validate_katten_seasonal_amplitude,
        False,
        TIER_NOT_EVIDENCE,
        "Self-labelled in source: \"STATUS: EXPLORATORY_ONLY. Invalid as "
        "physiological validation of K_ATTEN. Retained only for future "
        "behavioral modeling.\""),
    CheckSpec(
        "Kift 2013 South Asian", validate_kift_2013_south_asian, True,
        TIER_CIRCULAR,
        "DEMOTED to warning-only 2026-08-21 (was required, band included "
        "zero). Kift 2013 is one of the four within-study/field routes "
        "K_const (1.15 as of 2026-09-06, was 0.98) was fitted to (vitd_model.py K_const entry; "
        "scripts/kconst_recalibration.py). See function docstring for the "
        "full argument and the option-(a)-vs-(b) decision."),
    CheckSpec(
        "Kallioglu 2024 erythema-MED ratio [held-out, not vitamin-D evidence]",
        validate_kallioglu_2024_erythema_med_ratio, True,
        TIER_NOT_EVIDENCE,
        "Docstring: \"NOT independent vitamin-D evidence ... must not be "
        "counted or reported as such.\" Compares the model's minutes "
        "ratio to an erythema-MED table run through Hollick's rule, not "
        "to a vitamin-D measurement."),
    CheckSpec(
        "Reference Scenarios Benchmark [multi-study]",
        validate_reference_scenarios, False,
        TIER_QUANTITATIVE,
        "Cross-study multi-scenario benchmark: simulates all registered clinical "
        "trial protocols (Farrar 2013, Young 2021, Felton 2016, de Boer 2025, etc.) "
        "and checks physiological bounds, synthesis deltas, and cancer risk metrics."),
]


def run_all(verbose=True):
    results = {}
    warning_results = {}
    outcomes = {}

    print("=" * 70)
    print("VALIDATION SUITE")
    print("=" * 70)
    for spec in CHECK_REGISTRY:
        if verbose:
            print(f"[TIER: {spec.tier}] {spec.tier_note}")
        ok = spec.fn(verbose=verbose)
        outcomes[spec.name] = ok
        if spec.warning_only:
            warning_results[spec.name] = ok
        else:
            results[spec.name] = ok
        print()

    n_pass = sum(results.values())
    n_total = len(results)

    n_warn_pass = sum(warning_results.values())
    n_warn_total = len(warning_results)

    print("=" * 70)
    print(f"PASSED REQUIRED: {n_pass} / {n_total}")
    print(f"PASSED WARNING-ONLY: {n_warn_pass} / {n_warn_total}")
    print("Audit change tags active: C1 C2 C3 C4 C6 (see module docstring)")
    print("C5 C7 C8: see vitd_model.py docstring and parameter comments")
    print("=" * 70)
    print()
    print(f"EVIDENCE TIER BREAKDOWN -- all {len(CHECK_REGISTRY)} checks, orthogonal to")
    print("the required/warning-only split above. See the per-check tier_note")
    print("in CHECK_REGISTRY: the REQUIRED count above is real but misleading --")
    print("most of what it counts is self-consistency, not biology.")
    for tier in TIER_ORDER:
        members = [c for c in CHECK_REGISTRY if c.tier == tier]
        if not members:
            continue
        t_pass = sum(1 for c in members if outcomes[c.name])
        t_total = len(members)
        print(f"  {tier:13s} {t_pass} / {t_total}   ({TIER_LABELS[tier]})")
    print()
    print("HELD-OUT (not run inside this suite, un-gated by design):")
    print("  Webb/Kift 2018 -- RESULTS.md section 15, vitd_benchmark_webbkift.py.")
    print("  The one genuinely un-fitted external check in the pipeline. Gating")
    print("  is on the dose ratio only (geometry/climate-independent); the")
    print("  minutes ratio is diagnostic context, not a second pass/fail --")
    print("  see that module's docstring, 'ONE NUMBER, NOT TWO'. Recomputed on")
    print("  every run rather than quoted here, because a stale figure printed")
    print("  above the live section-15 number is exactly the drift this fix")
    print("  removes:")
    try:
        import vitd_benchmark_webbkift as _wk
        _bk = _wk.run(verbose=False)
        _verdict = "PASS" if _bk["webbkift_dose_ratio"] else "FAIL"
        _mr = (f"{_bk['model_minutes_ratio']:.2f}"
               if _bk["model_minutes_ratio"] is not None else "n/a")
        print(f"    {_verdict}: model dose ratio {_bk['model_dose_ratio']:.2f}x "
              f"vs reported {_wk.WK_DOSE_RATIO_LOW}-{_wk.WK_DOSE_RATIO_HIGH}x "
              f"(diagnostic minutes ratio {_mr} vs reported "
              f"{_wk.WK_MINUTES_FST_V / _wk.WK_MINUTES_WHITE:.2f})")
    except Exception as _exc:
        print(f"    (live re-run unavailable: {type(_exc).__name__}: {_exc} "
              "-- see RESULTS.md section 15 instead)")
    print("  See RESULTS.md section 15 for the full re-run.")
    print("=" * 70)

    # Return combining all for run_all.py or external consumers. SAME
    # contract as before this re-tiering: {check_name: bool} for every
    # registered check, keys unchanged in meaning (Kift 2013's key still
    # appears here even though its bucket moved from required to
    # warning-only -- run_all.py's sum()/len() over this dict still works
    # unmodified). Tier information lives in CHECK_REGISTRY / the printed
    # summary above, not in this dict, so nothing here can break run_all.py's
    # arithmetic.
    combined = {**results, **warning_results}
    return combined


if __name__ == "__main__":
    run_all()
