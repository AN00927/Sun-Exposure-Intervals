"""Guards on the 2026-08-17 K_const recalibration.

These are REGRESSION guards, not a re-derivation. The derivation lives in
`scripts/kconst_recalibration.py` and its diagnosis in
`scripts/fst_gap_decomposition.py`; both are slow and read a trial spreadsheet,
so neither belongs in the unit suite. What is guarded here is the pair of
properties the recalibration exists to establish, plus the invariants that
would silently break if someone later moved the number back.
"""

import numpy as np
import pytest

from vitd_model import ModelParams, simulate, get_E_source

# The approved band for the FST V:II OPTICAL factor
# (calibration_approval.yaml, a1_diagnosis_result_2026_08_17).
APPROVED_BAND = (1.7, 2.7)
# What dumping the behavioural share into K_const would have required. The
# calibration must stay clear of it -- that is the whole point of the split.
BEHAVIOUR_BLIND_FLOOR = 4.28


def optical(K, dark, light=2):
    p = ModelParams()
    return float(np.exp(K * (p.M_baseline(dark) - p.M_baseline(light))))


# ---------------------------------------------------------------------------
# The value and the band
# ---------------------------------------------------------------------------

def test_k_const_is_the_calibrated_value():
    assert ModelParams().K_const == pytest.approx(0.98, abs=1e-9)


def test_fst_v_ii_optical_factor_inside_the_approved_band():
    v = optical(ModelParams().K_const, 5)
    assert APPROVED_BAND[0] <= v <= APPROVED_BAND[1], (
        f"FST V:II optical factor {v:.2f}x is outside the approved "
        f"{APPROVED_BAND[0]}-{APPROVED_BAND[1]}x band")


def test_does_not_absorb_the_behavioural_share():
    """A K_const large enough to reproduce the raw field gap would exceed every
    matched-dose measurement in existence. Behaviour belongs in
    vitd_behaviour.py, not here."""
    assert optical(ModelParams().K_const, 5) < BEHAVIOUR_BLIND_FLOOR


def test_penalty_is_monotone_in_skin_type():
    p = ModelParams()
    factors = [optical(p.K_const, f, 1) for f in range(1, 7)]
    assert factors == sorted(factors)
    assert factors[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The sign fix -- the sharpest single result of the diagnosis
# ---------------------------------------------------------------------------
# Felton et al. 2016, Br J Dermatol 175:1320, Table 1. Within-study, matched
# dose (1.3 SED x3/wk x6 wk), matched ~35% BSA, same cabinet both arms, both
# baselines printed. Phototype II 36.5 -> 54.3; phototype V 17.2 -> 25.5.
FELTON_D0 = {2: 36.5, 5: 17.2}
FELTON_GAIN = {2: 17.8, 5: 8.3}
FELTON_OBSERVED_RATIO = 17.8 / 8.3      # 2.14


def _felton_gain(fst, K):
    sched = [(w * 7 + k * 2, 1.3) for w in range(6) for k in range(3)]
    res = simulate(fst=fst, schedule=sched, duration_days=42,
                   params=ModelParams(), D0=FELTON_D0[fst],
                   K_const_override=K,
                   E_source_override=get_E_source("arimed_b"),
                   exposed_bsa_override=0.35,
                   dose_framing="fixed_absolute_SED")
    return float(res["D"][-1]) - FELTON_D0[fst]


def test_felton_direction_is_no_longer_inverted():
    """At the old K_const = 0.33 the model predicted FST V gaining MORE than
    FST II on Felton's own protocol (ratio 0.96). That must not recur."""
    K = ModelParams().K_const
    ratio = _felton_gain(2, K) / _felton_gain(5, K)
    assert ratio > 1.0, (
        f"model predicts FST V gaining more than FST II on Felton 2016 "
        f"(II:V = {ratio:.2f}); the sign error the recalibration fixed has "
        f"returned")


def test_old_k_const_did_invert_felton():
    """Pins the defect the recalibration was for, so the guard above cannot be
    passed by a change that merely makes the protocol un-simulatable."""
    ratio = _felton_gain(2, 0.33) / _felton_gain(5, 0.33)
    assert ratio < 1.0


def test_felton_ratio_moved_toward_the_observation():
    old = _felton_gain(2, 0.33) / _felton_gain(5, 0.33)
    new = _felton_gain(2, ModelParams().K_const) / _felton_gain(5, ModelParams().K_const)
    assert abs(np.log(new / FELTON_OBSERVED_RATIO)) < abs(
        np.log(old / FELTON_OBSERVED_RATIO))


def test_felton_fst_v_over_prediction_is_recorded_not_forgotten():
    """The K_const fit matched RATIOS; both absolute arms are over-predicted.

    BOUNDS REBASED 2026-08-18, and the direction is WORSE, not better. Recorded
    here rather than hidden, because this is the clearest measured cost of the
    approved A2 + A4 changes.

        config                          FST II err   FST V err   model II:V
        K_const 0.33 (pre-Phase-3)        +24.3%      +178.5%      0.957
        K_const 0.98 (Phase 3)            +18.6%       +70.6%      1.491
        + A2 feedback form only            +0.7%       +70.6%      1.266
        + A4 A = 2.930 only               +47.6%      +111.9%      1.494
        + A2 and A4 together (shipped)    +31.5%      +121.7%      1.271
                                                   observed II:V = 2.145

    Two separable effects, and they are NOT the same one:
      A4 (A up x1.32) raises BOTH absolute gains and leaves the II:V ratio
        essentially untouched (1.491 -> 1.494). Absolute level is what A4 was
        approved to change, and Felton is not the study it was anchored to.
      A2 (biphasic feedback) is what moves the RATIO, 1.491 -> 1.271, away from
        the observation. Mechanism: the biphasic curve is steeper than the old
        linear one across 36-55 nmol/L, where Felton's FST II arm sits, and
        nearly identical to it across 17-26 nmol/L, where the FST V arm sits.
        So it damps the light-skin arm harder and closes part of the gap
        K_const had opened. It also, for the same reason, all but eliminates
        the FST II absolute error.

    The sign fix that Phase 3 was for SURVIVES -- the model still predicts
    FST II gaining more than FST V, and test_felton_ratio_moved_toward_the_
    observation still holds -- but roughly a third of the Phase 3 ratio gain
    (in log terms) is given back. Tighten these bounds if a later change
    improves the numbers; do not widen them.
    """
    K = ModelParams().K_const
    err5 = _felton_gain(5, K) / FELTON_GAIN[5] - 1.0
    err2 = _felton_gain(2, K) / FELTON_GAIN[2] - 1.0
    assert 0.25 < err2 < 0.40
    assert 1.05 < err5 < 1.35


def test_felton_ratio_regression_from_the_feedback_form_is_pinned():
    """The A2 cost above, isolated and pinned so it cannot be forgotten.

    Holding K_const and A at their shipped values and flipping ONLY the
    feedback form back to the retired linear one recovers most of the ratio.
    That attributes the regression to the form and not to A, and it fails if
    anyone later claims the two changes were ratio-neutral.
    """
    K = ModelParams().K_const
    shipped = _felton_gain(2, K) / _felton_gain(5, K)

    legacy = ModelParams()
    legacy.feedback_mode = "linear"
    sched = [(w * 7 + k * 2, 1.3) for w in range(6) for k in range(3)]

    def gain(fst):
        r = simulate(fst=fst, schedule=sched, duration_days=42, params=legacy,
                     D0=FELTON_D0[fst], K_const_override=K,
                     E_source_override=get_E_source("arimed_b"),
                     exposed_bsa_override=0.35,
                     dose_framing="fixed_absolute_SED")
        return float(r["D"][-1]) - FELTON_D0[fst]

    linear_ratio = gain(2) / gain(5)
    assert shipped == pytest.approx(1.271, abs=0.02)
    assert linear_ratio == pytest.approx(1.494, abs=0.02)
    assert linear_ratio > shipped, (
        "the feedback-form regression on Felton's II:V ratio has changed sign; "
        "re-derive it rather than editing this bound")


# ---------------------------------------------------------------------------
# Behaviour stayed out of the optical term
# ---------------------------------------------------------------------------

def test_behaviour_module_is_neutral_by_default():
    import vitd_behaviour as vb
    assert vb.behaviour_factor() == 1.0


def test_optical_and_behavioural_factors_are_separable():
    """Their product is what the behaviour-blind field inversion measured
    (4.3-5.7x). Each on its own must be well below that."""
    import vitd_behaviour as vb
    opt = optical(ModelParams().K_const, 5)
    beh = vb.measured_dose_ratio("kift2013")
    assert opt < BEHAVIOUR_BLIND_FLOOR
    assert opt * beh > BEHAVIOUR_BLIND_FLOOR   # together they over-explain it
