"""Tests for the model-ablation ladder (`vitd_ablation`).

The ladder exists so an analysis can say which mechanisms were active when it
produced a number. These tests pin the two properties that claim rests on:
`melanin_only` really does switch the SC channel off, and `melanin_plus_sc`
really is the untouched default model.
"""
import numpy as np
import pytest

import vitd_model
from vitd_model import build_pulse_schedule, simulate
from vitd_ablation import (
    RUNGS, list_rungs, reports_risk, rung_label, rung_overrides, simulate_rung,
)


# Multi-session schedule: 3x/week for 12 weeks. Long enough that SC
# hyperplasia is unambiguously built up under the default model.
DURATION_DAYS = 84
SCHEDULE = build_pulse_schedule(DURATION_DAYS, 1.0, 2)
FST = 3


def test_ladder_order():
    assert list_rungs() == ["melanin_only", "melanin_plus_sc", "melanin_sc_risk"]
    assert list(RUNGS) == list_rungs()


@pytest.mark.parametrize("rung", list_rungs())
def test_each_rung_runs(rung):
    result = simulate_rung(rung, fst=FST, schedule=SCHEDULE,
                           duration_days=DURATION_DAYS, D0=50)
    assert isinstance(result, dict)
    for key in ("t", "D", "M", "SC", "cancer_risk"):
        assert key in result, f"{rung} result missing {key!r}"
    assert result["ablation_rung"] == rung
    assert len(result["D"]) == len(result["t"])
    assert np.all(np.isfinite(result["D"]))
    # cancer_risk is computed unconditionally at every rung -- the top rung is
    # a reporting distinction, not a dynamical one.
    assert "combined_index" in result["cancer_risk"]


def test_default_model_builds_nonzero_sc():
    """Guard for the test below: the ablation must have something to remove."""
    sc = simulate(fst=FST, schedule=SCHEDULE, duration_days=DURATION_DAYS,
                  D0=50)["SC"]
    assert np.max(sc) > 0.1, f"default SC never exceeded {np.max(sc):.4f}"


def test_melanin_only_sc_is_identically_zero():
    result = simulate_rung("melanin_only", fst=FST, schedule=SCHEDULE,
                           duration_days=DURATION_DAYS, D0=50)
    sc = result["SC"]
    assert np.count_nonzero(sc) == 0, (
        f"SC not identically zero under melanin_only; "
        f"max |SC| = {np.max(np.abs(sc)):.3e}")
    assert result["SC_build_final"] == 0.0
    assert result["SC_damage_final"] == 0.0


def test_melanin_only_sc_zero_even_with_nonzero_k_damage():
    """k_damage is a separate channel from SC_max/alpha_SC.

    It defaults to 0.0, so the three SC-build overrides alone would pass by
    luck. With k_damage > 0 and no k_damage override, SC would go negative.
    """
    params = vitd_model.ModelParams()
    params.k_damage = 0.1
    sc = simulate_rung("melanin_only", fst=FST, schedule=SCHEDULE,
                       duration_days=DURATION_DAYS, D0=50, params=params)["SC"]
    assert np.count_nonzero(sc) == 0, (
        f"SC not identically zero with k_damage > 0; "
        f"min SC = {np.min(sc):.3e}")


def test_melanin_only_still_has_melanin_dynamics():
    result = simulate_rung("melanin_only", fst=FST, schedule=SCHEDULE,
                           duration_days=DURATION_DAYS, D0=50)
    M = result["M"]
    assert np.max(M) > M[0], "melanin never rose under melanin_only"


def test_melanin_plus_sc_reproduces_plain_simulate():
    """The middle rung must be the untouched default model, bit for bit."""
    plain = simulate(fst=FST, schedule=SCHEDULE, duration_days=DURATION_DAYS,
                     D0=50)
    rung = simulate_rung("melanin_plus_sc", fst=FST, schedule=SCHEDULE,
                         duration_days=DURATION_DAYS, D0=50)
    assert np.array_equal(rung["D"], plain["D"])
    assert np.array_equal(rung["SC"], plain["SC"])
    assert np.array_equal(rung["M"], plain["M"])
    assert rung_overrides("melanin_plus_sc") == {}


def test_risk_rung_matches_sc_rung_dynamically():
    """melanin_sc_risk is a REPORTING rung -- the trajectory is unchanged."""
    sc_rung = simulate_rung("melanin_plus_sc", fst=FST, schedule=SCHEDULE,
                            duration_days=DURATION_DAYS, D0=50)
    risk_rung = simulate_rung("melanin_sc_risk", fst=FST, schedule=SCHEDULE,
                              duration_days=DURATION_DAYS, D0=50)
    assert np.array_equal(risk_rung["D"], sc_rung["D"])
    assert np.array_equal(risk_rung["SC"], sc_rung["SC"])


def test_reports_risk_flags():
    assert reports_risk("melanin_sc_risk") is True
    assert reports_risk("melanin_only") is False
    assert reports_risk("melanin_plus_sc") is False


def test_rung_labels_are_distinct_and_short():
    labels = [rung_label(r) for r in list_rungs()]
    assert len(set(labels)) == len(labels)
    assert all(l and len(l) <= 32 for l in labels)


@pytest.mark.parametrize("fn,args", [
    (rung_label, ()),
    (reports_risk, ()),
    (rung_overrides, ()),
])
def test_unknown_rung_raises(fn, args):
    with pytest.raises(ValueError, match="unknown ablation rung"):
        fn("melanin_and_vibes", *args)


def test_unknown_rung_raises_in_simulate():
    with pytest.raises(ValueError, match="unknown ablation rung"):
        simulate_rung("no_such_rung", fst=FST, schedule=SCHEDULE,
                      duration_days=DURATION_DAYS)


@pytest.mark.parametrize("kw", ["K_SC_override", "alpha_SC_override",
                                "SC_max_override", "k_damage_override"])
def test_conflicting_override_raises(kw):
    with pytest.raises(ValueError, match="already sets"):
        simulate_rung("melanin_only", fst=FST, schedule=SCHEDULE,
                      duration_days=DURATION_DAYS, **{kw: 0.5})


def test_non_conflicting_override_passes_through():
    """A rung that sets no overrides must not block any keyword."""
    result = simulate_rung("melanin_plus_sc", fst=FST, schedule=SCHEDULE,
                           duration_days=DURATION_DAYS, D0=50,
                           K_SC_override=0.0)
    assert result["ablation_rung"] == "melanin_plus_sc"


def test_mutating_rungs_copy_does_not_leak():
    overrides = rung_overrides("melanin_only")
    overrides["K_SC_override"] = 99.0
    assert RUNGS["melanin_only"]["K_SC_override"] == 0.0
