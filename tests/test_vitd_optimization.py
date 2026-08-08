import pytest
import numpy as np
from vitd_model import ModelParams, simulate
from vitd_optimization import (
    schedule_uniform,
    schedule_frontloaded,
    schedule_backloaded,
    schedule_random,
    optimal_interval_risk_constrained,
)
from vitd_objectives import SUFFICIENCY_NMOL_L


def test_schedule_uniform_correct_count():
    sched = schedule_uniform(10, 84, 1.0)
    assert len(sched) == 10, f"Expected 10 sessions, got {len(sched)}"


def test_schedule_uniform_ordered():
    sched = schedule_uniform(10, 84, 1.0)
    days = [d for d, s in sched]
    assert all(days[i] <= days[i+1] for i in range(len(days)-1))


def test_schedule_uniform_within_bounds():
    sched = schedule_uniform(10, 84, 1.0)
    days = [d for d, s in sched]
    assert min(days) >= 0
    assert max(days) < 84


def test_schedule_frontloaded_first_interval_smaller():
    sched = schedule_frontloaded(10, 84, 1.0, skew=2.0)
    days = [d for d, s in sched]
    first_interval = days[1] - days[0]
    last_interval = days[-1] - days[-2]
    assert last_interval > first_interval, (
        f"Front-loaded: last interval {last_interval} should be > first {first_interval}"
    )


def test_schedule_backloaded_first_interval_larger():
    sched = schedule_backloaded(10, 84, 1.0, skew=2.0)
    days = [d for d, s in sched]
    first_interval = days[1] - days[0]
    last_interval = days[-1] - days[-2]
    assert first_interval > last_interval, (
        f"Back-loaded: first interval {first_interval} should be > last {last_interval}"
    )


def test_schedule_random_generates_correct_count():
    samples = list(schedule_random(10, 84, 1.0, n_samples=5))
    assert len(samples) == 5
    for sched in samples:
        assert len(sched) == 10


def test_schedule_uniform_all_same_sed():
    sched = schedule_uniform(10, 84, 1.0)
    seds = [s for d, s in sched]
    assert all(s == 1.0 for s in seds)


# ---------------------------------------------------------------------------
# Risk-constrained interval recommendation (constraint C9)
# ---------------------------------------------------------------------------
HORIZON = 200
RAMP = 90
GRID = (1, 3, 7, 14)


def _recommend(fst=3, **kw):
    kw.setdefault("horizon_days", HORIZON)
    kw.setdefault("ramp_in_days", RAMP)
    kw.setdefault("interval_grid", GRID)
    kw.setdefault("max_minutes", 120.0)
    return optimal_interval_risk_constrained(fst=fst, **kw)


class TestRiskConstrainedInterval:
    """A recommender that returns a schedule which does not actually work is
    worse than no recommender. These check the constraints it claims to satisfy
    are satisfied by an INDEPENDENT re-simulation, not by its own bookkeeping."""

    def test_recommendation_actually_holds_the_floor(self):
        r = _recommend(fst=3)
        rec = r["recommended"]
        assert rec is not None
        days = list(range(0, HORIZON, rec["interval_days"]))
        sim = simulate(fst=3,
                       schedule=[(float(d), rec["sed_per_session"]) for d in days],
                       duration_days=HORIZON, params=ModelParams(), D0=25.0,
                       dose_framing="fixed_per_session_SED")
        t, D = np.asarray(sim["t"]), np.asarray(sim["D"])
        assert float(np.min(D[t >= RAMP])) >= SUFFICIENCY_NMOL_L - 1e-6

    def test_recommendation_never_burns(self):
        """C1. A schedule that requires burning is not a schedule."""
        from vitd_physiology import _med_sed_by_fst
        from vitd_objectives import GAMMA_MED
        for fst in (1, 3, 6):
            rec = _recommend(fst=fst)["recommended"]
            assert rec is not None
            assert rec["sed_per_session"] <= GAMMA_MED * _med_sed_by_fst(fst) + 1e-9
            assert rec["burn_fraction"] == 0.0

    def test_picks_minimum_risk_among_feasible(self):
        r = _recommend(fst=3, risk_cap=None)
        rec = r["recommended"]
        feasible = [e for e in r["evaluated"] if e["feasible"]]
        assert feasible
        assert rec["combined_risk"] == pytest.approx(
            min(e["combined_risk"] for e in feasible))

    def test_respects_risk_cap(self):
        loose = _recommend(fst=3)["recommended"]
        cap = loose["combined_risk"] * 1.05
        r = _recommend(fst=3, risk_cap=cap)
        assert r["recommended"] is not None
        assert r["recommended"]["combined_risk"] <= cap

    def test_tightening_the_cap_never_raises_realised_risk(self):
        """The correct monotonicity direction. Note it is NOT 'a tighter cap
        widens the interval': longer intervals need a larger per-session dose to
        hold the floor, and the intermittent term is steeply increasing in dose,
        so tightening the cap pushes toward MORE frequent, smaller sessions --
        the model's frequent-and-small result. Asserting the opposite would lock
        in a claim the model contradicts."""
        base = _recommend(fst=3)["recommended"]["combined_risk"]
        prev = None
        for mult in (2.0, 1.5, 1.2, 1.0):
            r = _recommend(fst=3, risk_cap=base * mult)
            rec = r["recommended"]
            assert rec is not None, f"cap {mult}x should stay satisfiable"
            assert rec["combined_risk"] <= base * mult + 1e-12
            if prev is not None:
                assert rec["combined_risk"] <= prev + 1e-12
            prev = rec["combined_risk"]

    def test_feasible_set_shrinks_monotonically_with_the_cap(self):
        base = _recommend(fst=3)["recommended"]["combined_risk"]
        counts = []
        for mult in (10.0, 3.0, 1.0):
            r = _recommend(fst=3, risk_cap=base * mult)
            counts.append(sum(1 for e in r["evaluated"]
                              if e["feasible"] and e["within_risk_cap"]))
        assert counts == sorted(counts, reverse=True)

    def test_impossible_cap_reports_rather_than_fakes(self):
        r = _recommend(fst=3, risk_cap=1e-9)
        assert r["recommended"] is None
        assert r["note"] and "risk cap" in r["note"]

    def test_unreachable_target_reports_rather_than_fakes(self):
        r = _recommend(fst=6, target=500.0)
        assert r["recommended"] is None
        assert r["note"] and "unreachable" in r["note"]
        assert all(not e["feasible"] for e in r["evaluated"])

    def test_evaluated_covers_the_whole_grid(self):
        r = _recommend(fst=3)
        assert [e["interval_days"] for e in r["evaluated"]] == list(GRID)

    def test_session_length_cap_binds(self):
        """C7. Capping minutes can only lower the per-session dose."""
        loose = _recommend(fst=6, max_minutes=None)["recommended"]
        tight = _recommend(fst=6, max_minutes=20.0)["recommended"]
        assert loose is not None and tight is not None
        assert tight["sed_per_session"] <= loose["sed_per_session"] + 1e-9
        assert tight["minutes_per_session"] <= 20.0 + 1e-9

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            optimal_interval_risk_constrained(fst=0)
        with pytest.raises(ValueError):
            optimal_interval_risk_constrained(fst=3, interval_grid=())
        with pytest.raises(ValueError):
            optimal_interval_risk_constrained(fst=3, interval_grid=(0, 7))
