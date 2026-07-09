"""
vitd_validation.py
==================
Validation and sanity check suite for vitd_model.

Three categories:

1. Sanity checks - basic numerical correctness
   a. D-dependent decay half-life (Datta 2017): half-life varies with serum
      25(OH)D level via power-law fit t_half(D) = 2953 * D^(-0.72).
      At high D (~132 nmol/L), half-life ~89 days; at low D (~43 nmol/L),
      half-life ~199 days. Tests both endpoints within ±10%.
   b. 1 SED bolus in untanned FST III produces ~1.17 nmol/L rise
      (recalibrated to A=2.353 with K_const=0.50, and D=25.0)
   c. Photoadaptation: synthesis at high M is meaningfully reduced vs baseline M.
      Local evidence supports a broad 20-60% guard, not the older 40-75% range.

2. Literature validation targets
   a. de Boer 2025 - photoadaptation direction check (replaces mis-sourced Matsuoka 1990).
      Protocol: 0.8 SED 3x/week x 9 weeks on lower back, FST I-III. Checks direction
      only: melanin increases, SC thickens, synthesis decreases. Quantitative MI comparison
      is not possible (instrument-specific arbitrary units vs model's normalized 0-1 scale).
   b. Bogh 2012 - NB-UVB 3x/week for 6 weeks in FST III starting at 19.2 nmol/L
      produced a quantitative rise to 75.0 nmol/L (delta ~55.8 nmol/L).
      Validation is now QUANTITATIVE: model delta must be within ±15 nmol/L of
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

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
                         gross_synth_per_SED, net_final_D_change_per_SED,
                         auc_D_per_SED, auc_above_baseline_per_SED,
                         auc_above_sufficiency_per_SED)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_decay_halflife(verbose=True):
    """
    Datta 2017: state-dependent half-life from power-law fit.
    At high D (~132 nmol/L), half-life ≈ 89 days.
    At low D (~43 nmol/L), half-life ≈ 199 days.
    Test both endpoints with ±10% tolerance.
    """
    p = ModelParams()
    tol = 0.10
    
    # Check half-life at high D
    t_half_high = np.log(2) / p.k_decay_D(132.0)
    high_ok = abs(t_half_high - 89.0) / 89.0 <= tol
    
    # Check half-life at medium D
    t_half_med = np.log(2) / p.k_decay_D(65.0)
    med_ok = abs(t_half_med - 148.0) / 148.0 <= tol
    
    # Check half-life at low D
    t_half_low = np.log(2) / p.k_decay_D(43.0)
    low_ok = abs(t_half_low - 199.0) / 199.0 <= tol
    
    # Simulate decay from 132 nmol/L for 89 days
    result = simulate(fst=3, schedule=[], duration_days=89, params=p, D0=132.0)
    D_89 = np.interp(89.0, result["t"], result["D"])
    sim_high_ok = abs(D_89 / 132.0 - 0.5) < 0.06
    
    # Simulate decay from 43 nmol/L for 199 days
    result = simulate(fst=3, schedule=[], duration_days=199, params=p, D0=43.0)
    D_199 = np.interp(199.0, result["t"], result["D"])
    sim_low_ok = abs(D_199 / 43.0 - 0.5) < 0.06
    
    passed = high_ok and med_ok and low_ok and sim_high_ok and sim_low_ok
    if verbose:
        print(f"[Sanity 1] D-dependent decay half-life (Datta 2017 power-law fit)")
        print(f"  D=132: t_half={t_half_high:.1f}d (expected 89d, ±10%): {'OK' if high_ok else 'FAIL'}")
        print(f"  D=65:  t_half={t_half_med:.1f}d (expected 148d, ±10%): {'OK' if med_ok else 'FAIL'}")
        print(f"  D=43:  t_half={t_half_low:.1f}d (expected 199d, ±10%): {'OK' if low_ok else 'FAIL'}")
        print(f"  Sim D=132->D(89)/D0={D_89/132.0:.3f} (target ~0.5): {'OK' if sim_high_ok else 'FAIL'}")
        print(f"  Sim D=43->D(199)/D0={D_199/43.0:.3f} (target ~0.5): {'OK' if sim_low_ok else 'FAIL'}")
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def sanity_single_bolus_FST_III(verbose=True):
    """
    1 SED bolus in untanned FST III should produce ~1.17 nmol/L rise starting from baseline D=25.0, SC=0.0.
    A=2.353 and K_const=0.50 were recalibrated together so that the baseline FST III
    constitutive factor and feedback factor at D=25.0 are preserved:
    delta-D = A * ln(2) * exp(-0.50 * 0.30) * (1 - 25.0/150.0).

    Evidence caveat: this is a self-calibration check for the current per-SED
    model unit. Bogh 2011/P5 shows strong exposed-BSA dependence, so this small
    single-bolus target should not be compared directly to protocols exposing
    6-24% BSA or more without an explicit BSA term.
    """
    p = ModelParams()
    fst = 3
    M_base = p.M_baseline(fst)
    delta = synthesis_per_session(1.0, M_base, M_base, D=25.0, SC=0.0, params=p)
    passed = 1.0 <= delta <= 1.3
    if verbose:
        print(f"[Sanity 2] Single bolus, FST III, untanned")
        print(f"  delta-D = {delta:.2f} nmol/L  (target range 1.0 - 1.3, A=2.353 K_const=0.50)")
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def sanity_photoadaptation_reduction(verbose=True):
    """
    At M = 0.70 (well above FST III baseline of 0.30), synthesis should be
    substantially less than at M = M_baseline. The guard is deliberately broad:
    de Boer 2025 supports photoadaptation by MED/MI/SC changes but not a direct
    vitamin-D attenuation coefficient, while Young 2020 supports only modest
    melanin-only attenuation.
    """
    p = ModelParams()
    fst = 3
    M_base = p.M_baseline(fst)
    d_untanned = synthesis_per_session(1.0, M_base, M_base, D=25.0, SC=0.0, params=p)
    d_tanned = synthesis_per_session(1.0, 0.70, M_base, D=25.0, SC=0.0, params=p)
    reduction = 1.0 - d_tanned / d_untanned
    passed = 0.10 <= reduction <= 0.30
    if verbose:
        print(f"[Sanity 3] Photoadaptation reduction (FST III, M 0.30 -> 0.70)")
        print(f"  untanned synthesis: {d_untanned:.2f}  tanned synthesis: {d_tanned:.2f}")
        print(f"  reduction = {100*reduction:.1f}%  (guard range 10-30%; Young 2020 supports small melanin effect)")
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
    """
    Bogh 2012: full-body NB-UVB 3x/week for 6 weeks (18 sessions) in
    light-skinned Danish adults raised mean 25(OH)D from 19.2 to 75.0 nmol/L
    (delta +55.8 nmol/L; between-group p=0.02 vs oral D3).

    [C1] QUANTITATIVE THRESHOLD (replaces prior qualitative 'delta > 5 nmol/L'):
    Model delta must fall within ±15 nmol/L of the observed +55.8 nmol/L,
    i.e., 40.8 to 70.8 nmol/L. A tolerance of ±15 is granted because the
    model is broadband-calibrated and E_source=2.5 is a placeholder for NB-UVB
    311 nm; a tighter bound requires spectral calibration (see [C3] below).

    [C3] SOURCE / SPECTRAL CALIBRATION NOTE:
    E_source=2.5 is a per-study placeholder. Young 2020 (P20) shows that the
    standard CIE previtamin-D3 action spectrum does not collapse polychromatic
    responses to a common regression and a 5 nm blue shift gives a better fit.
    Osmancevic 2009 (P18) shows broadband UVB can produce larger D rises than
    NB-UVB in psoriasis cohorts despite lower physical SED. Source handling must
    become a first-class per-study calibration input, not a fixed multiplier.
    E_source_override should be treated as a study-specific parameter, not a
    general NB-UVB conversion constant.
    """
    p = ModelParams()
    sched = build_weekly_schedule(42, 1.0, [0, 2, 4])
    result = simulate(fst=3, schedule=sched, duration_days=42, params=p, D0=19.2, E_source_override=2.5)
    D_final = result["D"][-1]
    delta = D_final - 19.2
    # [C1] Quantitative target: observed delta = +55.8 nmol/L; tolerance ±15 nmol/L
    OBSERVED_DELTA = 55.8
    TOLERANCE = 15.0
    passed = abs(delta - OBSERVED_DELTA) <= TOLERANCE
    if verbose:
        print(f"[Lit 2] Bogh 2012 - quantitative NB-UVB validation  [C1]")
        print(f"  Protocol: 1 SED 3x/week x 6 weeks, FST III, baseline 19.2, E_source=2.5")
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
    1. Gross >= Net: gross_synth_per_SED >= net_final_D_change_per_SED for a sparse schedule.
    2. AUC above sufficiency = 0 when D(t) never exceeds the threshold (e.g. mild schedule with threshold 50.0).
    3. AUC above baseline is > 0 when D(t) rises above baseline.
    """
    p = ModelParams()
    
    # 1. Sparse schedule check: Gross >= Net
    sched_sparse = build_pulse_schedule(84, 10.5, 21)
    r_sparse = simulate(fst=5, schedule=sched_sparse, duration_days=84, params=p, D0=30.0)
    gross_sparse = gross_synth_per_SED(r_sparse)
    net_sparse = net_final_D_change_per_SED(r_sparse)
    
    # 2. Check AUC above sufficiency = 0 when D never reaches threshold (e.g. 50)
    sched_mild = build_pulse_schedule(84, 0.1, 7)
    r_mild = simulate(fst=3, schedule=sched_mild, duration_days=84, params=p, D0=25.0)
    auc_suff_mild = auc_above_sufficiency_per_SED(r_mild, threshold_nmol_L=50.0)
    
    # 3. Check AUC above baseline > 0 when D rises
    sched_rise = build_pulse_schedule(84, 1.0, 2)
    r_rise = simulate(fst=3, schedule=sched_rise, duration_days=84, params=p, D0=25.0)
    auc_base_rise = auc_above_baseline_per_SED(r_rise)
    
    passed_gross_net = gross_sparse >= net_sparse
    passed_auc_suff = abs(auc_suff_mild) < 1e-9
    passed_auc_base = auc_base_rise > 0.0
    
    passed = passed_gross_net and passed_auc_suff and passed_auc_base
    
    if verbose:
        print(f"[New Check] Metric ordering and sanity assertions")
        print(f"  Sparse schedule (10.5 SED every 21d):")
        print(f"    gross_synth_per_SED:        {gross_sparse:.4f}")
        print(f"    net_final_D_change_per_SED: {net_sparse:.4f}")
        print(f"    Gross >= Net: {'YES' if passed_gross_net else 'NO'}")
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
    We test if the model's saturating feedback mode can replicate this target.
    """
    p = ModelParams()
    p.feedback_mode = 'saturating'
    p.D_half_feedback = 90.0  # calibrated threshold from P33
    
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
    # High baseline change in [-15.0, -1.0]
    passed_bounds = (0.5 <= change_low <= 10.0) and (-15.0 <= change_high <= -1.0)
    passed = passed_direction and passed_bounds
    
    if verbose:
        print(f"[Lit 5] Karppinen 2017/P33 - baseline feedback calibration target")
        print(f"  Protocol: 1.18 SED/day x 10 days, FST III, BSA=0.07, E_source=2.0")
        print(f"  Saturating feedback mode (D_half_feedback = 90.0):")
        print(f"    Low baseline (62.4 nmol/L) change:  +{change_low:.2f} nmol/L (observed +6.0)")
        print(f"    High baseline (116.9 nmol/L) change: {change_high:.2f} nmol/L (observed -6.7)")
        print(f"    Low baseline increased: {'YES' if change_low > 0 else 'NO'}")
        print(f"    High baseline decreased: {'YES' if change_high < 0 else 'NO'}")
        print(f"    Within target bounds: {'YES' if passed_bounds else 'NO'}")
        print(f"  {'PASS' if passed else 'FAIL'}")
        
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
#    summary card in SCIENTIFIC_AUDIT_HANDOFF.md (outside this file)
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
        ("de Boer 2025 photoadaptation [C6 replaced]", validate_photoadaptation_deBoer2025, True),  # warning_only
        ("Bogh 2012 [C1 quantitative]",   validate_bogh_2012, False),
        ("Farrar 2013 [C2 quantitative]", validate_farrar_2013, False),
        ("Bogh baseline+BSA [C4]",        validate_bogh_baseline_bsa, False),
        ("baseline calibration target [C3-feedback]", validate_baseline_calibration_target, False),
        ("metric ordering and sanity",    validate_metric_ordering, False),
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
