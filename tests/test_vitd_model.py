import pytest
import numpy as np
from vitd_model import (
    simulate, build_pulse_schedule, steady_state_D,
    gross_synth_per_SED, net_final_D_change_per_SED,
    synthesis_per_session, ModelParams,
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


def test_efficiency_ordering():
    schedule = build_pulse_schedule(84, 1.0, 7)
    result = simulate(fst=2, schedule=schedule, duration_days=84)
    gross = gross_synth_per_SED(result)
    net = net_final_D_change_per_SED(result)
    assert gross >= net - 1e-12
    assert gross >= 0
    assert net > 0


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


def test_bsa_factor_bounds_and_monotonicity():
    p = ModelParams()
    assert p.bsa_factor(0) == 0.0
    assert p.bsa_factor(1.0) == pytest.approx(1.69, abs=0.01)
    assert p.bsa_factor(0.5) < p.bsa_factor(0.7)
