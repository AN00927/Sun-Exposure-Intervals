import pytest
import numpy as np
from vitd_model import (
    simulate, build_pulse_schedule, steady_state_D,
    synthesis_per_session, ModelParams, auc_deficit,
)


def test_empty_schedule_decay():
    params = ModelParams()
    result = simulate(fst=5, schedule=[], duration_days=84, D0=50, params=params)
    D = result["D"]
    assert D[-1] < D[0], f"final D {D[-1]:.3f} >= initial D {D[0]:.3f}"
    assert all(D > 0)


def test_steady_state_independent_of_D0():
    schedule = build_pulse_schedule(365, 1.0, 7)
    for fst in [2, 5]:
        ss_values = []
        for D0 in [15, 25, 50]:
            result = simulate(fst=fst, schedule=schedule, duration_days=365, D0=D0)
            ss_values.append(steady_state_D(result))
        max_diff = max(ss_values) - min(ss_values)
        assert max_diff <= 5.0, f"FST={fst}: max steady-state diff = {max_diff:.3f} nmol/L"


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


def test_synthesis_at_Dmax():
    result = synthesis_per_session(sed=1.0, M=0.5, M_baseline=0.21, D=250, SC=0.1, params=ModelParams())
    assert result == pytest.approx(0.0, abs=1e-12)


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
