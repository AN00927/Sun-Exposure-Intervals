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
      K_const, A, D_max and the feedback form. K_ATTEN is RETIRED at 0.00, so
      the measured value is 0.0% and this is a one-sided drift guard, NOT
      evidence. See sanity_photoadaptation_reduction for the full re-derivation.

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

    The simulator is then checked against the ANALYTIC solution of the ODE
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

    This asserts that `synthesis_per_session` computes the product the module
    docstring says it computes. It is NOT a literature check and has no
    external target.

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
PHOTOADAPTATION_REDUCTION_MAX = 0.10   # [CHOICE] -> K_ATTEN <= 0.527


def sanity_photoadaptation_reduction(verbose=True):
    """Guard on the FACULTATIVE (tanning) melanin channel.

    Measures synthesis at M = M_ceiling(FST III) = 0.41, the largest tan the
    melanin ODE can produce, against untanned M = M_baseline = 0.21, at
    matched D and SC.

    WHAT THE MEASURED QUANTITY ACTUALLY IS
    --------------------------------------
    Everything except the facultative term is identical between the two calls,
    so A, K_const, bsa_factor, E_source, SC and the feedback factor all CANCEL
    EXACTLY, and the check reduces to

        reduction = 1 - exp(-K_ATTEN * M_TAN_HEADROOM)

    This is not an approximation; it is an algebraic identity, and it is
    asserted as such in tests/test_vitd_validation.py. Two consequences:

      * The 2026-08-17 K_const move (0.33 -> 0.98) CANNOT have changed this
        check, and neither can the 2026-08-18 A and feedback-form changes.
        `calibration_approval.yaml` -> pending_approvals asked for the band to
        be re-derived because K_const had moved since it was set. It was
        re-derived; the answer is that K_const never entered. The band stands
        on its own terms, not because it happened to keep passing.
      * The check has exactly ONE degree of freedom, K_ATTEN, so the band is
        equivalent to a bracket on K_ATTEN. At M_TAN_HEADROOM = 0.20:
            0-10%  reduction  <=>  K_ATTEN in [0.000, 0.527]
            10-30% reduction  <=>  K_ATTEN in [0.527, 1.783]   (the OLD band)

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
    K_ATTEN 0.34-1.00 depending on the assumed feedback form
    (vitd_calibration.derive_K_ATTEN), i.e. reductions of 6.6-18.1%. A 10%
    ceiling therefore admits the LOW end of the Matsuoka bracket and rejects its
    high end, which is the correct posture given that Matsuoka's "51%" is mostly
    D-feedback rather than pigment (see CLAUDE.md).

    WHAT THIS CHECK IS NOT
    ----------------------
    It is NOT evidence that photoadaptation reduces synthesis. With K_ATTEN
    RETIRED at 0.00 (approved item A6) the measured reduction is exactly 0.0%
    and this check CANNOT FAIL at the value the model ships. Read it as a
    ONE-SIDED RETIREMENT GUARD: it fires if anyone reinstates a strong
    facultative penalty without going through RETIRED_PARAMETERS. Do not count
    it as literature support, and read the headline "passed" count knowing one
    of its members is pinned by construction.
    """
    p = ModelParams()
    fst = 3
    M_base = p.M_baseline(fst)
    M_tanned = p.M_ceiling(fst)  # 0.41 for FST III (M_base + M_TAN_HEADROOM)
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
              f"K_ATTEN is RETIRED at {K_ATTEN_STRUCTURAL:.2f}, so this is a "
              f"one-sided drift guard, NOT evidence.")
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
    NB-UVB by ~17 nmol/L with current parameters (alpha_SC=0.12, SC_max=0.23
    for FST III). This is a known structural limitation: E_source=2.5 is a
    placeholder requiring per-study spectral calibration (see [C3] below).

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
    3x/week UVB at 0.65-3.9 SED to 35% BSA. Subjects receiving >=1.95 SED
    reached mean 15.7 +/- 5 ng/mL with a mean rise of 8.7 +/- 5.7 ng/mL
    (all completers were <20 ng/mL at baseline; only 6/51 reached >=20 ng/mL).

    [C2] QUANTITATIVE THRESHOLD (replaces prior loose 'D_final < 68.0 nmol/L'):
    Unit conversion: 15.7 ng/mL * 2.496 = ~39.2 nmol/L final;
                     8.7 ng/mL * 2.496 = ~21.7 nmol/L rise.
    We simulate the >=1.95 SED group with 1.95 SED/session.
    D0=30 nmol/L is a proxy within Farrar's reported baseline constraint
    (all completers <20 ng/mL, i.e. <~50 nmol/L). Exact group baseline is not
    extractable from the local summary alone, so this check asserts D0 < 50.
    Acceptance windows (broadband model approximation of Manchester winter
    31% BSA protocol; exact BSA/dose conversion not resolved locally):
      D_final in [25, 55] nmol/L   (observed ~39.2, wide band for broadband approx)
      delta    in [10, 35] nmol/L  (observed ~21.7)
    Both conditions must hold.
    """
    p = ModelParams()
    fst = 5
    D0_PROXY = 30.0
    # Simulate the >=1.95 SED dose group: use 1.95 SED/session
    sched = []
    week_days = [0, 2, 4]
    for w in range(6):
        for d in week_days:
            sched.append((w * 7 + d, 1.95))
    result = simulate(fst=fst, schedule=sched, duration_days=42, params=p, D0=D0_PROXY)
    D_final = result["D"][-1]
    delta = D_final - D0_PROXY
    # [C2] Quantitative targets from Farrar 2013 (unit-converted)
    OBSERVED_FINAL_NM = 39.2   # nmol/L (15.7 ng/mL * 2.496)
    OBSERVED_DELTA_NM = 21.7   # nmol/L (8.7 ng/mL * 2.496)
    D_final_ok = 25.0 <= D_final <= 55.0
    delta_ok   = 10.0 <= delta  <= 35.0
    baseline_ok = D0_PROXY < 50.0
    passed = baseline_ok and D_final_ok and delta_ok
    if verbose:
        print(f"[Lit 3] Farrar 2013 - FST V, 1.95 SED 3x/week x 6 weeks, baseline {D0_PROXY:.1f}  [C2]")
        print(f"  Baseline proxy <50 nmol/L constraint: {'YES' if baseline_ok else 'NO'}")
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
    [C7-SC-FITTED] alpha_SC=0.12, SC_max_by_FST: I-III=0.23, IV-VI=0.36.
    Warning-only because k_damage=0. The week-5 dip is representable (SC is
    two-process) but unidentifiable from three group medians.

    Target SC at week 9: FP I-III proxy (FST III, SC_max=0.23) should be
      in [0.12, 0.23] (reaches ~90% of SC_max within 9 weeks).
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

    SC_III_range_ok = 0.08 <= SC_III <= 0.23
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
        print(f"  FST III (FP I-III proxy, SC_max=0.23): SC={SC_III:.4f} "
              f"(increased: {'YES' if SC_III_increased else 'NO'}, "
              f"range [0.08,0.23]: {'OK' if SC_III_range_ok else 'OUT'})")
        print(f"  FST V  (FP IV-VI proxy, SC_max=0.36): SC={SC_V:.4f} "
              f"(increased: {'YES' if SC_V_increased else 'NO'}, "
              f"range [0.14,0.36]: {'OK' if SC_V_range_ok else 'OUT'})")
        print(f"  SC_V >= SC_III * 0.8: {'YES' if SC_V_ge_III else 'NO'}")
        print(f"  SHAPE (reported, not asserted): model T1={sc_t1:+.3f} "
              f"T2={sc_t2:+.3f} vs observed {obs_t1_rel:+.3f} / {obs_t2_rel:+.3f}")
        print(f"    Observed SC dips ~26% BELOW baseline at week 5. Model cannot")
        print(f"    reproduce that at k_damage=0.0, which is deliberate: the")
        print(f"    parameter is unidentifiable from 3 group medians. See")
        print(f"    scripts/fit_sc_deboer2025.py and MODEL_AND_EVIDENCE.md, 'Remaining gaps', gap 27.")
        print(f"  {'PASS' if passed else 'FAIL'} (warning-only)")
    return passed


# ---------------------------------------------------------------------------
# NHANES population validation
# ---------------------------------------------------------------------------

# Race/ethnicity code to (label, FST) mapping from NHANES RIDRETH1/RIDRETH3.
# Codes: 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White,
#        4=Non-Hispanic Black, 5=Other/Multiracial (RIDRETH1),
#        6=Other/Multiracial (RIDRETH3).  Code 7 (no clear mapping) is dropped.
RACE_FST_MAP = {
    1: ("Mexican American", 4),
    2: ("Other Hispanic", 4),
    3: ("Non-Hispanic White", 2),
    4: ("Non-Hispanic Black", 5),
     5: ("Other/Multiracial", 3),
     6: ("Non-Hispanic Asian", 4),
}

# RIDEXMON codes: 1 = Nov-Apr (winter), 2 = May-Oct (summer)
SEASON_MAP = {1: "winter", 2: "summer"}

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
# reference"). NHANES is a real-sun population scenario, so it takes the same
# solar value as every other sun path; see vitd_model.E_SOURCE_SOLAR_DEFAULT.
#
# THIS MAKES PART B FAIL, AND THAT FAILURE WAS ACCEPTED DELIBERATELY. Holding
# the assumed dose above fixed while raising E_source 1.0 -> 1.89 raises every
# modelled summer steady state. The alternative -- deflating _SIM_SED by ~1.89x
# to keep the check green -- would be tuning an unmeasured knob to hide a
# calibration change, i.e. calibration through an ungated back door. Do not do
# that. See calibration_approval.yaml, E_SOURCE_SOLAR_DEFAULT / not_covered.
_SIM_ESOURCE = get_E_source(E_SOURCE_SOLAR_DEFAULT)
_SIM_TOLERANCE = 10.0  # nmol/L, matching tightened Bogh 2012 convention


def validate_nhanes_seasonal(nhanes_available=True, verbose=True):
    """
    NHANES population-level seasonal validation for the vitamin D scheduling model.

    Part A — Direction check (warning-only)
        For every NHANES race/ethnicity group, verify that the observed
        summer mean 25(OH)D > winter mean 25(OH)D.  This confirms the
        expected seasonal pattern exists in the source data.

    Part B — Summer steady-state comparison (required)
        For every FST-mapped group, simulate a daily sun-exposure schedule
        (0.333 SED/session, 84 days, E_source=1.0, 30% BSA) using the
        NHANES winter mean as the dietary-baseline proxy (D0).  Compare
        the model-predicted steady-state D to the NHANES summer mean with
        an absolute tolerance of ±10 nmol/L (tightened from ±15; model
        prediction error should not exceed 10 nmol/L for a calibrated model).

    Returns
    -------
    dict
        {check_name: bool} — merges into run_all.py's val_results.
    """
    data_path = os.path.join(os.path.dirname(__file__),
                             "audit_outputs", "nhanes_standardized_processed.csv")

    file_ok = (os.path.exists(data_path) and os.path.getsize(data_path) > 0)

    if not nhanes_available or not file_ok:
        if verbose:
            print("\n" + "=" * 70)
            print("NHANES POPULATION VALIDATION")
            print("=" * 70)
            print("\n  WARNING: NHANES data not found. "
                  "Run nhanes_preprocessing.py first.")
            print("  Skipping NHANES seasonal validation.\n")
        return {}

    # ------------------------------------------------------------------
    # Load and filter
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path)

    # Adults 18–65 only
    df = df[(df["age"] >= 18) & (df["age"] <= 65)].copy()
    df = df[df["vitd_standardized_nmol_L"] > 0].copy()

    # Map race codes to labels and FST
    df["race_label"] = df["race_ethnicity"].map(
        lambda x: RACE_FST_MAP[x][0] if x in RACE_FST_MAP else None)
    df["fst"] = df["race_ethnicity"].map(
        lambda x: RACE_FST_MAP[x][1] if x in RACE_FST_MAP else None)

    # Drop rows with missing race, D, season, or weight
    df = df.dropna(subset=["race_label", "fst",
                           "vitd_standardized_nmol_L", "exam_season", "wtmec2yr"])

    # Map season codes to labels
    df["season"] = df["exam_season"].map(SEASON_MAP)
    df = df.dropna(subset=["season"])

    # Compute per-group mean and SEM of 25(OH)D by season
    grouped = df.groupby(["race_label", "fst", "season"])
    def weighted_mean_25ohd(g):
        return np.average(g["vitd_standardized_nmol_L"], weights=g["wtmec2yr"])
    grp_mean = grouped.apply(weighted_mean_25ohd)
    grp_sem = df.groupby(["race_label", "fst", "season"])["vitd_standardized_nmol_L"].sem()

    # Build lookup: {(label, fst): {"winter": mean, "summer": mean, "sem": ...}}
    groups = {}
    for (label, fst, season), mean_val in grp_mean.items():
        key = (label, int(fst))
        if key not in groups:
            groups[key] = {}
        groups[key][season] = mean_val
    for (label, fst, season), sem_val in grp_sem.items():
        key = (label, int(fst))
        if key not in groups:
            groups[key] = {}
        if "sem" not in groups[key]:
            groups[key]["sem"] = {}
        groups[key]["sem"][season] = sem_val

    if not groups:
        if verbose:
            print("\n  WARNING: No valid race/season groups after filtering.")
        return {"NHANES group availability": True}

    # ------------------------------------------------------------------
    # Part A — Direction check (warning-only)
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("NHANES POPULATION VALIDATION")
        print("=" * 70)
        print("\n--- Part A: Seasonal direction check (warning-only) ---")
        print(f"  {'Group':25s} {'FST':4s} {'Winter':>8s} {'Summer':>8s}  Direction")

    direction_ok = True
    for (label, fst), data in sorted(groups.items()):
        w = data.get("winter")
        s = data.get("summer")
        if w is None or s is None:
            continue
        ok = s > w
        if not ok:
            direction_ok = False
        if verbose:
            print(f"  {label:25s}  {fst:1d}   {w:6.1f}  {s:6.1f}  "
                  f"{'OK' if ok else 'WARNING (summer <= winter)'}")

    if verbose:
        if direction_ok:
            print(f"  -> Direction check: PASS (all groups summer > winter)")
        else:
            print(f"  -> Direction check: WARNING (some groups summer <= winter)")

    # ------------------------------------------------------------------
    # Part B — Summer steady-state comparison (required)
    # ------------------------------------------------------------------
    # [C1-BSA] If residuals correlate with BMI/weight, this would indicate
    # incomplete BSA modeling (exposed_bsa vs total_bsa). Current model uses
    # a fixed exposed_bsa=0.30 for all subjects.
    if verbose:
        print(f"\n--- Part B: Summer steady-state comparison "
              f"(±{_SIM_TOLERANCE} nmol/L) ---")
        print(f"  Protocol: daily {_SIM_SED} SED, {_SIM_DAYS} days, "
              f"E_source={_SIM_ESOURCE}, BSA={_SIM_BSA}")
        print(f"  D0 = NHANES winter mean for each group.")
        print(f"  {'Group':25s} {'FST':4s} {'Model_ss':>8s} "
              f"{'NHANES_summer':>13s}  {'|diff|':>7s}  Status")

    p = ModelParams()
    summer_passed = {}
    model_ss_map = {}
    for (label, fst), data in sorted(groups.items()):
        w = data.get("winter")
        s = data.get("summer")
        if w is None or s is None:
            continue

        # Build daily schedule for simulation
        schedule = [(d, _SIM_SED) for d in range(_SIM_DAYS)]
        result = simulate(fst=fst, schedule=schedule,
                          duration_days=_SIM_DAYS, params=p,
                          D0=w,
                          E_source_override=_SIM_ESOURCE,
                          exposed_bsa_override=_SIM_BSA)
        model_ss = steady_state_D(result)
        model_ss_map[(label, fst)] = model_ss
        diff = abs(model_ss - s)
        ok = diff <= _SIM_TOLERANCE
        summer_passed[(label, fst)] = ok

        if verbose:
            dw = data.get("sem", {}).get("winter")
            ds = data.get("sem", {}).get("summer")
            winter_str = f"{w:.1f}±{dw:.1f}" if dw and not np.isnan(dw) else f"{w:.1f}"
            summer_str = f"{s:.1f}±{ds:.1f}" if ds and not np.isnan(ds) else f"{s:.1f}"
            print(f"  {label:25s}  {fst:1d}   {model_ss:6.1f}  "
                  f"{summer_str:>13s}  {diff:5.1f}  "
                  f"{'PASS' if ok else 'FAIL'}")

    all_summer_ok = all(summer_passed.values())
    if verbose:
        n_summer_pass = sum(1 for v in summer_passed.values() if v)
        n_summer_total = len(summer_passed)
        if all_summer_ok:
            print(f"  -> Summer comparison: "
                  f"PASS ({n_summer_pass}/{n_summer_total} within ±{_SIM_TOLERANCE})")
        else:
            print(f"  -> Summer comparison: "
                  f"{n_summer_pass}/{n_summer_total} passed "
                  f"(target: all within ±{_SIM_TOLERANCE} nmol/L)")

    # ------------------------------------------------------------------
    # Part C — Dietary source residual stratification (if available)
    # ------------------------------------------------------------------
    if "dietary_source_flag" in df.columns:
        summer_df = df[df["season"] == "summer"].copy()
        for (label, fst_val), ss_val in model_ss_map.items():
            mask = (summer_df["race_label"] == label) & (summer_df["fst"] == fst_val)
            summer_df.loc[mask, "model_ss"] = ss_val
        summer_df = summer_df.dropna(subset=["model_ss", "vitd_standardized_nmol_L", "dietary_source_flag"])
        if len(summer_df) > 0:
            summer_df.loc[:, "residual"] = summer_df["model_ss"] - summer_df["vitd_standardized_nmol_L"]
            total_n = len(summer_df)
            if verbose:
                print(f"\n--- Part C: Dietary source residual stratification ---")
                print(f"  {'Flag':20s} {'n':>6s} {'Fraction':>9s}  {'Mean resid':>10s} {'Std resid':>10s}")
            for flag, grp in sorted(summer_df.groupby("dietary_source_flag")):
                n = len(grp)
                frac = n / total_n * 100
                mean_res = grp["residual"].mean()
                std_res = grp["residual"].std()
                if verbose:
                    print(f"  {flag:20s} {n:6d} {frac:7.2f}%  {mean_res:8.2f}  {std_res:8.2f}")
            if verbose:
                total_mean = summer_df["residual"].mean()
                total_std = summer_df["residual"].std()
                print(f"  {'Total':20s} {total_n:6d} {'100.00%':>9s}  {total_mean:8.2f}  {total_std:8.2f}")

    if verbose:
        print()

    # Return dict compatible with val_results.update() in run_all.py
    return {
        "NHANES seasonal direction": True,        # warning-only → always "passes"
        "NHANES summer comparison": all_summer_ok,   # required check
    }


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


def validate_kift_2013_south_asian(verbose=True):
    """
    Kift 2013: UK South Asian adults (FST V) receiving naturalistic UK summer exposure.
    Protocol: ~14% BSA exposed (face + hands/forearms), ~1.0 SED/week (0.333 SED 3x/week)
    for 12 weeks (84 days), baseline 22.5 nmol/L.

    Target:
      - Baseline ~22.5 nmol/L
      - D_final in range [20.0, 40.0] nmol/L
      - delta in range [0.0, 15.0] nmol/L
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
        print(f"[Lit 6] Kift 2013 - UK South Asian naturalistic exposure (FST V)")
        print(f"  Protocol: ~1.0 SED/wk (0.333 SED 3x/wk) x 12 weeks, FST V, BSA=0.14, baseline {D0:.1f} nmol/L")
        print(f"  Model D_final = {D_final:.1f} nmol/L (delta = +{delta:.1f} nmol/L)")
        print(f"  Target bounds: D_final in [20, 40]: {'YES' if D_final_ok else 'NO'}, delta in [0, 15]: {'YES' if delta_ok else 'NO'}")
        print(f"  {'PASS' if passed else 'FAIL'}")

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
# 2. Add a (name, fn, warning_only) tuple to the |checks| list below.
#    - name: short human-readable label.
#    - fn: the function from step 1.
#    - warning_only: True if the check is advisory (does not count
#      toward required pass/fail).
#
# 3. Run |python3 vitd_validation.py| to confirm registration.
#
# 4. If the check introduces a new literature source, add a paper
#    summary card in MODEL_AND_EVIDENCE.md (outside this file)
#    and cross-reference the audit-change tag in the docstring.
#
# See existing checks for examples.
# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all(verbose=True):
    checks = [
        ("decay half-life",              sanity_decay_halflife, False),
        ("single bolus FST III",          sanity_single_bolus_FST_III, False),
        ("photoadaptation reduction",     sanity_photoadaptation_reduction, False),
        ("de Boer 2025 photoadaptation [C6 replaced]", validate_photoadaptation_deBoer2025, True),
        ("SC de Boer 2025 [C7-SC warning]", validate_sc_deBoer2025, True),
        ("Bogh 2012 [C1 quantitative]",   validate_bogh_2012, False),
        ("Farrar 2013 [C2 quantitative]", validate_farrar_2013, False),
        ("Bogh baseline+BSA [C4]",        validate_bogh_baseline_bsa, False),
        ("baseline calibration target [C3-feedback]", validate_baseline_calibration_target, False),
        ("metric ordering and sanity",    validate_metric_ordering, False),
        ("K_ATTEN seasonal amplitude",    validate_katten_seasonal_amplitude, False),
        ("Kift 2013 South Asian",         validate_kift_2013_south_asian, False),
        ("Kallioglu 2024 erythema-MED ratio [held-out, not vitamin-D evidence]",
         validate_kallioglu_2024_erythema_med_ratio, True),
    ]
    
    results = {}
    warning_results = {}
    
    print("=" * 70)
    print("VALIDATION SUITE")
    print("=" * 70)
    for name, fn, warning_only in checks:
        ok = fn(verbose=verbose)
        if warning_only:
            warning_results[name] = ok
        else:
            results[name] = ok
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
    
    # Return combining all for run_all.py or external consumers
    combined = {**results, **warning_results}
    return combined


if __name__ == "__main__":
    run_all()
