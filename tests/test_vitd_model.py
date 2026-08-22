import pytest
import numpy as np
from vitd_model import (
    simulate, build_pulse_schedule, steady_state_D,
    synthesis_per_session, ModelParams, auc_deficit,
    net_final_D_change_per_SED,
)


def test_empty_schedule_decay():
    params = ModelParams()
    result = simulate(fst=5, schedule=[], duration_days=84, D0=50, params=params)
    D = result["D"]
    assert D[-1] < D[0], f"final D {D[-1]:.3f} >= initial D {D[0]:.3f}"
    assert all(D > 0)


def _d0_spread(fst, days):
    schedule = build_pulse_schedule(days, 1.0, 7)
    ss = [steady_state_D(simulate(fst=fst, schedule=schedule,
                                  duration_days=days, D0=D0))
          for D0 in (15, 25, 50)]
    return max(ss) - min(ss)


def test_steady_state_independent_of_D0():
    """Steady state must not remember the starting value.

    Horizon rebased 2026-08-17: 365 -> 730 days, and FST VI added.
    Tolerance unchanged at 5.0 nmol/L; the change is the horizon needed to
    reach it. Recalibrating K_const 0.33 -> 0.98 weakens synthesis in
    pigmented skin, so the approach to steady state is slower there and 365
    days no longer washes out the initial condition for FST V (spread 5.4).
    A convergence rate change, not a loss of the property being asserted:
    the memory of D0 decays to nothing, and
    test_steady_state_memory_decays_with_horizon below shows it does.

    Widening the 5.0 tolerance instead would have made the test pass without
    testing anything -- see the pending_approvals entry in
    calibration_approval.yaml about a band widened until a check passed by
    construction.
    """
    for fst in (2, 5, 6):
        spread = _d0_spread(fst, 730)
        assert spread <= 5.0, f"FST={fst}: max steady-state diff = {spread:.3f} nmol/L"


def test_steady_state_memory_decays_with_horizon():
    """D0 memory shrinks monotonically as the run lengthens.

    What "independent of D0" means, stated so it cannot be satisfied by
    picking a convenient horizon.
    """
    for fst in (2, 5, 6):
        spreads = [_d0_spread(fst, d) for d in (365, 730, 1095)]
        assert spreads == sorted(spreads, reverse=True), (
            f"FST={fst}: D0 memory does not decay: {spreads}")
        assert spreads[-1] < 1.0, (
            f"FST={fst}: D0 memory still {spreads[-1]:.3f} nmol/L at 3 years")


def test_build_pulse_schedule_counts():
    assert len(build_pulse_schedule(84, 1.0, 7)) == 12
    assert len(build_pulse_schedule(84, 1.0, 1)) == 84
    assert len(build_pulse_schedule(84, 1.0, 100)) == 1
    assert len(build_pulse_schedule(0, 1.0, 7)) == 0


def test_D_never_negative():
    schedule = build_pulse_schedule(365, 10.0, 1)
    result = simulate(fst=5, schedule=schedule, duration_days=365, D0=25)
    assert all(result["D"] >= 0)


def test_M_stays_in_bounds():
    params = ModelParams()
    schedule = build_pulse_schedule(84, 1.0, 1)
    result = simulate(fst=2, schedule=schedule, duration_days=84, params=params)
    M_baseline = params.M_baseline_by_FST[2]
    assert all(result["M"] >= M_baseline - 1e-10)
    assert all(result["M"] <= 0.95 + 1e-10)


def test_SC_stays_in_bounds():
    schedule = build_pulse_schedule(84, 1.0, 1)
    result = simulate(fst=2, schedule=schedule, duration_days=84)
    assert all(result["SC"] >= -1e-10)
    assert all(result["SC"] <= 0.30 + 1e-10)


def test_synthesis_zero_sed():
    result = synthesis_per_session(sed=0, M=0.5, M_baseline=0.21, D=50, SC=0.1, params=ModelParams())
    assert result == 0.0


def test_synthesis_at_high_D_is_suppressed_but_never_zero():
    """Rewritten 2026-08-18 (approved item A2).

    This used to assert `synthesis == 0` at D = 250, which was true only under
    the withdrawn max(0, 1 - D/D_max) form. That assertion encoded the exact
    claim the approval rejected: that cutaneous synthesis stops outright above
    some serum level. It does not, and D_max = 150.0 never had a located
    source. The property worth testing: feedback is strong and monotone, not
    absolute.
    """
    p = ModelParams()
    at_250 = synthesis_per_session(sed=1.0, M=0.5, M_baseline=0.21, D=250, SC=0.1, params=p)
    at_25 = synthesis_per_session(sed=1.0, M=0.5, M_baseline=0.21, D=25, SC=0.1, params=p)
    assert at_250 > 0.0, "synthesis must never switch off outright"
    assert at_250 < 0.10 * at_25, "feedback at 250 nmol/L must still be strong"


def test_legacy_linear_mode_still_zeroes_at_D_max():
    """The retired form stays working so older results stay reproducible."""
    p = ModelParams()
    p.feedback_mode = "linear"
    r = synthesis_per_session(sed=1.0, M=0.5, M_baseline=0.21, D=250, SC=0.1, params=p)
    assert r == pytest.approx(0.0, abs=1e-12)


def test_D_max_override_is_inert_in_the_default_mode():
    """A silently-ignored override is a trap, so its inertness is asserted
    rather than left to be discovered. D_max is legacy: only 'linear' and
    'power' read it."""
    p = ModelParams()
    assert p.feedback_mode == "biphasic"
    sched = build_pulse_schedule(84, 1.0, 3)
    a = simulate(fst=3, schedule=sched, duration_days=84, params=p, D0=25.0)
    b = simulate(fst=3, schedule=sched, duration_days=84, params=p, D0=25.0,
                 D_max_override=90.0)
    assert a["D"][-1] == pytest.approx(b["D"][-1], rel=1e-12)


class TestFeedbackFormA2:
    """Approved item A2: the hard ceiling is withdrawn as default."""

    def test_default_is_biphasic(self):
        assert ModelParams().feedback_mode == "biphasic"

    def test_inflection_is_the_scholze_value(self):
        from vitd_physiology import (HYDROXYLATION_INFLECTION_NMOL_L,
                                     HYDROXYLATION_HILL_N)
        # Scholze et al. 2026, Commun Med 6:105. [ESTIMATE] 56.0 is an output
        # of a fitted breakpoint on the vitamin D3 axis, not a regressor.
        # [CHOICE] the Hill exponent is not fittable from that paper; 2.0 is
        # the gentlest integer knee.
        assert HYDROXYLATION_INFLECTION_NMOL_L == 56.0
        assert HYDROXYLATION_HILL_N == 2.0

    def test_every_mode_is_monotone_decreasing_in_D(self):
        """The property that produces the interval ranking."""
        for mode in ("biphasic", "saturating", "linear", "power"):
            p = ModelParams()
            p.feedback_mode = mode
            vals = [synthesis_per_session(1.0, 0.21, 0.21, D=d, SC=0.0, params=p)
                    for d in (10, 30, 60, 90)]
            assert all(a > b for a, b in zip(vals, vals[1:])), mode

    def test_interval_optimum_is_not_an_artefact_of_the_linear_form(self):
        """run_all.py section 17 attributes the interval ranking to the
        `(1 - D/D_max)` term specifically. At fixed A the optimum is 3 d
        under linear, saturating, and biphasic alike, so the ordering
        comes from having a decreasing feedback factor, not from that term.
        """
        for mode in ("biphasic", "saturating", "linear"):
            p = ModelParams()
            p.feedback_mode = mode
            p.A = 2.353                      # the pre-A4 value, held fixed
            v = [net_final_D_change_per_SED(
                    simulate(fst=3, schedule=build_pulse_schedule(84, 1.0, iv),
                             duration_days=84, params=p, D0=25.0), 25.0)
                 for iv in range(1, 16)]
            assert int(np.argmax(v)) + 1 == 3, mode


class TestYoung2021AAnchor:
    """Approved item A4: A is anchored to Young 2021's printed FB slopes.

    Kept in the unit suite rather than in run_all()'s scored validation list:
    it is a calibration anchor and cannot fail in an interesting way, so
    counting it as a passed validation would inflate the score by construction
    -- the mistake calibration_approval.yaml records for Sanity 3.
    """

    def test_A_is_the_anchored_value(self):
        assert ModelParams().A == pytest.approx(3.114, abs=1e-9)

    def test_A_reproduces_the_anchor_arithmetic(self):
        from vitd_validation import young2021_A_anchor
        out = young2021_A_anchor()
        assert out["A_implied"] == pytest.approx(ModelParams().A, abs=5e-4)
        assert out["model_slope_at_current_A"] == pytest.approx(3.48, abs=0.01)

    def test_printed_slopes_are_transcribed(self):
        from vitd_validation import YOUNG2021_FB_SLOPES_NMOL_PER_SED as S
        # Young 2021 PNAS 118:e2015867118, Fig. 4B legend, FB panel.
        assert S == {"PUVA": 1.99, "Arimed B": 3.18, "UV6": 4.97}

    def test_the_two_readings_of_midpoint_agree(self):
        """Range midpoint (3.48) vs the n-weighted pooled FB arm (3.61)."""
        from vitd_validation import young2021_A_anchor
        out = young2021_A_anchor()
        a, b = out["target_slope_nmol_per_SED"], out["pooled_slope_nmol_per_SED"]
        assert abs(b / a - 1.0) < 0.05

    def test_the_source_matched_cross_checks_bracket_the_adopted_value(self):
        """Young's three arms disagree by 2.5x because SED is a poor spectral
        predictor -- the paper's own headline. A must sit between the two
        E_source-matched routes, not on one of them."""
        from vitd_validation import young2021_A_anchor
        c = young2021_A_anchor()["cross_checks"]
        assert min(c.values()) < ModelParams().A < max(c.values())


class TestKAttenCalibration:
    """K_ATTEN is calibrated to 0.34 based on Matsuoka 1990 deconvolution."""

    def test_k_atten_default_is_calibrated_value(self):
        from vitd_model import K_ATTEN_DEFAULT
        assert K_ATTEN_DEFAULT == 0.34
        assert ModelParams().K_ATTEN == 0.34

    def test_assignment_and_construction_allowed(self):
        p = ModelParams(K_ATTEN=0.5)
        assert p.K_ATTEN == 0.5
        p.K_ATTEN = 0.8
        assert p.K_ATTEN == 0.8

    def test_the_mechanism_is_active(self):
        """Tanning attenuates synthesis relative to untanned baseline."""
        p = ModelParams()
        tanned = synthesis_per_session(1.0, 0.41, 0.21, D=25.0, SC=0.0, params=p)
        untanned = synthesis_per_session(1.0, 0.21, 0.21, D=25.0, SC=0.0, params=p)
        assert tanned < untanned

    def test_k_atten_range_immateriality(self):
        """Across the full plausible range [0.0, 1.0], K_ATTEN moves steady-state
        D by well under 1%, against ~36% for A."""
        sched = build_pulse_schedule(365, 1.0, 3)
        ss = []
        for k in (0.0, 0.34, 1.0):
            p = ModelParams(K_ATTEN=k)
            ss.append(steady_state_D(simulate(fst=5, schedule=sched,
                                              duration_days=365, params=p,
                                              D0=25.0)))
        assert (max(ss) - min(ss)) / max(ss) < 0.01, ss


def test_feedback_monotonicity():
    params = ModelParams()
    high = synthesis_per_session(sed=1.0, M=0.5, M_baseline=0.21, D=20, SC=0.1, params=params)
    low = synthesis_per_session(sed=1.0, M=0.5, M_baseline=0.21, D=80, SC=0.1, params=params)
    assert high > low


def test_auc_deficit_above_threshold_is_zero():
    D = np.array([60, 62, 65, 70])
    t = np.array([0, 10, 20, 30])
    result = auc_deficit(D, t, threshold=50)
    assert result == 0.0, f"Expected 0, got {result}"


def test_auc_deficit_below_threshold_positive():
    D = np.array([40, 35, 30, 25])
    t = np.array([0, 10, 20, 30])
    result = auc_deficit(D, t, threshold=50)
    assert result > 0, f"Expected positive deficit, got {result}"


def test_auc_deficit_strictly_increasing_with_longer_deficit():
    D_short = np.array([40, 45, 55, 60])
    t_short = np.array([0, 5, 10, 15])
    D_long = np.array([40, 35, 30, 45])
    t_long = np.array([0, 5, 10, 15])
    short_def = auc_deficit(D_short, t_short, threshold=50)
    long_def = auc_deficit(D_long, t_long, threshold=50)
    assert long_def > short_def, f"Longer deficit should give larger AUC"


def test_auc_deficit_matches_manual_calculation():
    D = np.array([40, 40, 60, 60])
    t = np.array([0, 10, 10, 20])
    expected = 100.0
    result = auc_deficit(D, t, threshold=50)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"


def test_bsa_factor_bounds_and_monotonicity():
    p = ModelParams()
    assert p.bsa_factor(0) == 0.0
    assert p.bsa_factor(1.0) == pytest.approx(1.69, abs=0.01)
    assert p.bsa_factor(0.5) < p.bsa_factor(0.7)
