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
# calibration must stay clear of it, that is the whole point of the split.
BEHAVIOUR_BLIND_FLOOR = 4.28


def optical(K, dark, light=2):
    p = ModelParams()
    return float(np.exp(K * (p.M_baseline(dark) - p.M_baseline(light))))


# ---------------------------------------------------------------------------
# The value and the band
# ---------------------------------------------------------------------------

def test_k_const_is_the_calibrated_value():
    # REBASED 2026-09-06: 0.98 -> 1.15, the coupled refit against the Narea
    # 2026 M_baseline_by_FST table (calibration_approval.yaml, K_const entry;
    # scripts/kconst_recalibration.py run against the new table).
    assert ModelParams().K_const == pytest.approx(1.15, abs=1e-9)


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
# The sign fix: the sharpest single result of the diagnosis
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
    passed by a change that only makes the protocol un-simulatable."""
    ratio = _felton_gain(2, 0.33) / _felton_gain(5, 0.33)
    assert ratio < 1.0


def test_felton_ratio_moved_toward_the_observation():
    old = _felton_gain(2, 0.33) / _felton_gain(5, 0.33)
    new = _felton_gain(2, ModelParams().K_const) / _felton_gain(5, ModelParams().K_const)
    assert abs(np.log(new / FELTON_OBSERVED_RATIO)) < abs(
        np.log(old / FELTON_OBSERVED_RATIO))


def test_felton_fst_v_over_prediction_is_recorded_not_forgotten():
    """The K_const fit matched RATIOS; both absolute arms are over-predicted.

    BOUNDS REBASED 2026-08-18, REBASED AGAIN 2026-09-06 for the
    K_const/M_baseline_by_FST/BSA_EXP move. This time the direction is
    BETTER on both arms, not worse -- recorded here for the same reason as
    before: so an improvement is not silently taken for granted either.

        config                              FST II err   FST V err   model II:V
        K_const 0.33 (pre-Phase-3)            +24.3%      +178.5%      0.957
        K_const 0.98 (Phase 3)                +18.6%       +70.6%      1.491
        A2+A4 shipped (through 2026-09-05)    +31.5%      +121.7%      1.271
        K_const 1.15, new M_baseline, BSA_EXP  +8.8%       +84.9%      1.262
        (2026-09-06, shipped)
                                                       observed II:V = 2.145

    Both absolute errors fell (FST II +31.5% -> +8.8%, FST V +121.7% ->
    +84.9%) while the II:V ratio barely moved (1.271 -> 1.262) -- consistent
    with the K_const entry's own finding that Felton's implied K_const alone
    moved further than the other three routes when refit against the Narea
    table (1.473 -> 2.097), while M_baseline(II) and M_baseline(V) moving in
    the SAME direction (both up) partially cancels in the ratio while still
    shifting each arm's absolute prediction.

    BOUNDS WIDENED 2026-09-06, LATER THE SAME DAY, AND THE DIRECTION IS BADLY
    WORSE. The row above was measured with A still at its stale 3.114. When A
    was re-anchored to 4.125 (calibration_approval.yaml,
    A_reanchor_2026_09_06) both Felton arms deteriorated sharply:

        config                                  FST II err   FST V err   model II:V
        ... rows above measured at A = 3.114 ...
        + A re-anchored to 4.125 (SHIPPED)        +49.1%      +159.6%      1.232
                                                           observed II:V = 2.145

    Model gains are now FST II 26.53 against an observed 17.8, and FST V
    21.54 against an observed 8.3. THIS IS A WIDENING, WHICH THE PREVIOUS
    VERSION OF THIS DOCSTRING FORBADE, AND IT IS DELIBERATE. The rule it
    breaks was written to stop a bound being loosened to hide drift; here the
    drift is the POINT and is recorded rather than hidden. This test is a
    RECORDER (see its name), not a quality gate.

    WHAT THE TRADE ACTUALLY IS. Re-anchoring A rescued Bogh 2012 (+38.8 ->
    +50.1, band 45.8-65.8) and Farrar 2013 (+8.7 -> +13.6, band 10-35), both
    of which the four-parameter move had knocked out, and cost Felton: the
    reference-scenario check felton2016_fst5_matched now FAILS at +21.9
    against +8.3 observed (band is 2.5x observed, i.e. <= 20.75). Net
    validation 8/10 -> 9/10, with a different check failing than before.
    A was NOT tuned into the A in [3.75, 3.87] window that would pass all
    three, because Felton 2016 sits in the CALIBRATION set behind K_const and
    Bogh 2012 is self-labelled TIER: CIRCULAR -- fitting A to them would
    collapse the calibration/validation separation. See
    calibration_approval.yaml, A_reanchor_2026_09_06, `not_covered`.

    OPEN QUESTION, NOT RESOLVED HERE: whether Felton is now telling us
    something structural (the model cannot hold Young/Bogh/Farrar and Felton
    simultaneously, the same mutual incompatibility the A entry already
    records for Young-vs-Bogh) or whether the 2.5x reference-scenario band is
    simply too tight on an observed delta as small as 8.3 nmol/L.

    The sign fix survives: the model still predicts FST II gaining more than
    FST V, and test_felton_ratio_moved_toward_the_observation still holds.
    TIGHTEN these bounds if a later change improves the numbers; widen them
    only with the same kind of note as this one.
    """
    K = ModelParams().K_const
    err5 = _felton_gain(5, K) / FELTON_GAIN[5] - 1.0
    err2 = _felton_gain(2, K) / FELTON_GAIN[2] - 1.0
    assert 0.42 < err2 < 0.56
    assert 1.50 < err5 < 1.70


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
    # PINS REBASED 2026-09-06 for the full pass (K_const 0.98 -> 1.15,
    # M_baseline_by_FST -> Narea table, BSA_EXP 0.50 -> 0.3281, A re-anchored
    # 3.114 -> 4.125, K_SC held at 1.0). Both ratios fell slightly and the
    # ORDERING -- the actual property under test -- is unchanged:
    # 1.271 -> 1.2316 (biphasic, shipped) and 1.494 -> 1.4389 (legacy linear).
    assert shipped == pytest.approx(1.2316, abs=0.02)
    assert linear_ratio == pytest.approx(1.4389, abs=0.02)
    assert linear_ratio > shipped, (
        "the feedback-form regression on Felton's II:V ratio has changed sign; "
        "re-derive it rather than editing this bound")


# ---------------------------------------------------------------------------
# Behaviour stayed out of the optical term
# ---------------------------------------------------------------------------
# The two tests that exercised `vitd_behaviour` directly MOVED on 2026-08-21 to
# tests/exploratory/test_behaviour_separation.py, when that module was retired
# (see exploratory/README.md). The half of the argument that does NOT need the
# retired module -- that K_const alone stays clear of the behaviour-blind floor
# -- is guarded by test_does_not_absorb_the_behavioural_share above and
# stays here.
