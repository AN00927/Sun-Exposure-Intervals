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
        """C7. Capping minutes can only lower the per-session dose.

        FIXTURE REBASED 2026-08-17. The old fixture was FST VI at a 20-minute
        ceiling. Under K_const = 0.33 that cap sat ABOVE FST VI's requirement
        (~16 min/day), so it never actually bound and the assertion held
        trivially. Under K_const = 0.98 FST VI needs ~42 min/day, so a 20-min
        ceiling makes the schedule INFEASIBLE -- see
        TestRecalibratedFeasibility, which asserts that directly.

        The property under test is the optimiser's monotonicity in the cap, so
        it is now exercised at operating points where both arms are still
        feasible. The cap value is not a tolerance and nothing was widened:
        60 min for FST VI and 20 min for FST IV are both STRICTER than "no
        cap", which is the direction that makes the assertion meaningful.
        """
        for fst, mm in ((6, 60.0), (4, 20.0), (3, 20.0)):
            loose = _recommend(fst=fst, max_minutes=None)["recommended"]
            tight = _recommend(fst=fst, max_minutes=mm)["recommended"]
            assert loose is not None and tight is not None, (fst, mm)
            assert tight["sed_per_session"] <= loose["sed_per_session"] + 1e-9
            assert tight["minutes_per_session"] <= mm + 1e-9

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            optimal_interval_risk_constrained(fst=0)
        with pytest.raises(ValueError):
            optimal_interval_risk_constrained(fst=3, interval_grid=())
        with pytest.raises(ValueError):
            optimal_interval_risk_constrained(fst=3, interval_grid=(0, 7))


# ---------------------------------------------------------------------------
# Variable-dose profiles and optimisation
# ---------------------------------------------------------------------------
from vitd_optimization import (            # noqa: E402
    session_days, dose_uniform, dose_two_phase, dose_exponential,
    dose_seasonal, dose_gap_coupled, dose_knots, build_variable_schedule,
    optimise_variable_dose, VARIABLE_DOSE_FAMILIES,
)
from vitd_objectives import time_to_sufficiency, GAMMA_MED  # noqa: E402
from vitd_physiology import (_med_sed_by_fst, cancer_risk,  # noqa: E402
                             INTERMITTENT_BURN_EXPONENT)


class TestDoseProfiles:
    def test_families_return_one_dose_per_session(self):
        days = session_days(20, 365)
        assert len(dose_uniform(days, 1.0)) == 20
        assert len(dose_two_phase(days, 2.0, 0.5, 100.0)) == 20
        assert len(dose_exponential(days, 1.0, 2.0, 30.0)) == 20
        assert len(dose_seasonal(days, 1.0, 0.4)) == 20
        assert len(dose_gap_coupled(days, 1.0, 1.0)) == 20
        assert len(dose_knots(days, [1.0, 2.0, 0.5], T_days=365)) == 20

    def test_constant_dose_is_recovered_at_the_null_parameter(self):
        """Each family must nest the constant-dose baseline, or a comparison
        against it is not a comparison -- the search could lose to uniform
        purely by being unable to express it."""
        days = session_days(30, 365)
        assert np.allclose(dose_gap_coupled(days, 1.5, 0.0), 1.5)
        assert np.allclose(dose_seasonal(days, 1.5, 0.0), 1.5)
        assert np.allclose(dose_two_phase(days, 1.5, 1.5, 100.0), 1.5)
        assert np.allclose(dose_exponential(days, 1.5, 0.0, 30.0), 1.5)
        assert np.allclose(dose_knots(days, [1.5, 1.5, 1.5], T_days=365), 1.5)

    def test_two_phase_loads_then_drops(self):
        days = session_days(40, 365)
        seds = dose_two_phase(days, 2.0, 0.5, 100.0)
        assert np.all(seds[days < 100.0] == 2.0)
        assert np.all(seds[days >= 100.0] == 0.5)

    def test_exponential_decays_monotonically_to_the_asymptote(self):
        days = session_days(40, 365)
        seds = dose_exponential(days, 1.0, 3.0, 30.0)
        assert np.all(np.diff(seds) < 0)
        assert seds[-1] == pytest.approx(1.0, abs=1e-3)

    def test_knots_do_not_overshoot_between_control_points(self):
        """PCHIP is shape-preserving; a cubic spline would ring here and could
        emit negative doses."""
        days = session_days(60, 365)
        seds = dose_knots(days, [0.1, 3.0, 0.1], T_days=365)
        assert seds.min() >= 0.0
        assert seds.max() <= 3.0 + 1e-9


class TestBuildVariableSchedule:
    def test_clips_to_the_erythema_ceiling(self):
        days = session_days(10, 365)
        sched = build_variable_schedule(days, np.full(10, 99.0), fst=3)
        cap = GAMMA_MED * _med_sed_by_fst(3)
        assert all(s <= cap + 1e-12 for _, s in sched)

    def test_zero_dose_drops_the_session(self):
        """This is what lets session count be optimised without integer
        variables: d_i = 0 IS 'no session that day'."""
        days = session_days(10, 365)
        seds = np.array([1.0, 0.0] * 5)
        sched = build_variable_schedule(days, seds, fst=3)
        assert len(sched) == 5

    def test_negative_and_nan_doses_do_not_propagate(self):
        days = session_days(4, 365)
        sched = build_variable_schedule(days, np.array([1.0, -2.0, np.nan, 0.5]),
                                        fst=3)
        assert all(np.isfinite(s) and s > 0 for _, s in sched)


class TestVariableDoseOptimisation:
    # FIXTURE REBASED 2026-08-17: N 104 -> 146. These tests are about
    # PROPERTIES OF THE OPTIMISER (a nested family cannot lose to the family it
    # nests; the returned profile really holds the floor), and they need a
    # feasible operating point to exercise them on. FST III at N = 104 stopped
    # being one when K_const was recalibrated 0.33 -> 0.98: holding 50 nmol/L
    # on 104 sessions now needs more than 70% MED per session in FST III.
    #
    # NO CONSTRAINT WAS RELAXED to do this. The sufficiency floor, GAMMA_MED
    # and the ramp-in are untouched; only the session budget the fixture uses
    # moved, to one that is still feasible. The infeasibility itself is a
    # RESULT and is asserted explicitly in TestRecalibratedFeasibility below,
    # rather than being made to disappear.
    HORIZON, RAMP, N = 365, 90, 146

    def _opt(self, family, fst=3):
        return optimise_variable_dose(fst=fst, n_sessions=self.N, family=family,
                                      horizon_days=self.HORIZON,
                                      ramp_in_days=self.RAMP)

    def test_returned_profile_actually_holds_the_floor(self):
        """Re-simulated independently of the optimiser's own bookkeeping."""
        for fam in ("uniform", "two_phase", "exponential"):
            res = self._opt(fam)
            assert res is not None, f"{fam} found nothing feasible"
            assert res["trough_after_ramp"] >= SUFFICIENCY_NMOL_L - 1e-6

    def test_returned_profile_never_burns(self):
        for fam in VARIABLE_DOSE_FAMILIES:
            res = self._opt(fam)
            if res is None:
                continue
            assert res["max_sed"] <= GAMMA_MED * _med_sed_by_fst(3) + 1e-9

    def test_variable_dose_is_never_worse_than_uniform(self):
        """Every family nests constant dose, so the search can only tie or win.
        A loss would mean the optimiser cannot find a point it can express."""
        base = self._opt("uniform")
        assert base is not None
        for fam in VARIABLE_DOSE_FAMILIES:
            res = self._opt(fam)
            if res is None:
                continue
            assert res["combined_risk"] <= base["combined_risk"] * 1.02, (
                f"{fam} lost to uniform, which it can express exactly")

    def test_gain_comes_from_a_smaller_budget_not_from_reshaping(self):
        """The Jensen argument, checked on the model rather than asserted: at
        fixed total dose AND session count, equal doses minimise the
        intermittency term, so any real gain must show up as less total UV."""
        base = self._opt("uniform")
        best = self._opt("two_phase")
        assert base is not None and best is not None
        if best["combined_risk"] < base["combined_risk"] * 0.99:
            assert best["total_sed"] < base["total_sed"]

    def test_equal_doses_minimise_the_intermittency_term(self):
        """Direct Jensen check at fixed budget and count."""
        n, budget = 60, 60.0
        equal = np.full(n, budget / n)
        base = cancer_risk(equal, 3)["intermittent_index"]
        rng = np.random.default_rng(0)
        for _ in range(25):
            w = rng.random(n) + 0.05
            uneven = budget * w / w.sum()
            assert cancer_risk(uneven, 3)["intermittent_index"] >= base - 1e-12
        assert INTERMITTENT_BURN_EXPONENT > 1.0   # why the above holds

    def test_unreachable_target_returns_none_rather_than_a_near_miss(self):
        assert optimise_variable_dose(fst=3, n_sessions=104, target=500.0,
                                      horizon_days=self.HORIZON,
                                      ramp_in_days=self.RAMP) is None

    def test_rejects_unknown_family(self):
        with pytest.raises(ValueError):
            optimise_variable_dose(fst=3, n_sessions=20, family="nope")


class TestJointTimingAndDose:
    """Joint spacing + dose search."""

    # FIXTURE REBASED 2026-08-17: N 104 -> 146, for the reason given on
    # TestVariableDoseOptimisation. Same operating point, so the two classes
    # stay comparable.
    HORIZON, RAMP, N = 365, 90, 146

    def _opt(self, free_spacing, fst=3, n=None, seed=0):
        from vitd_optimization import optimise_joint_schedule
        return optimise_joint_schedule(
            fst=fst, n_sessions=n or self.N, horizon_days=self.HORIZON,
            ramp_in_days=self.RAMP, free_spacing=free_spacing, seed=seed,
            maxiter=8, popsize=6)

    def test_warp_recovers_uniform_spacing_at_gamma_one(self):
        from vitd_optimization import warp_days
        assert np.allclose(warp_days(40, 365, 1.0), session_days(40, 365))

    def test_warp_is_monotone_and_within_horizon(self):
        from vitd_optimization import warp_days
        for gamma in (0.4, 1.0, 2.5):
            d = warp_days(50, 365, gamma)
            assert np.all(np.diff(d) > 0)
            assert d[0] == 0.0 and d[-1] < 365

    def test_gamma_above_one_front_loads(self):
        from vitd_optimization import warp_days
        front = warp_days(50, 365, 2.0)
        assert front[1] - front[0] < front[-1] - front[-2]

    def test_returned_schedule_holds_the_floor_and_never_burns(self):
        for free in (False, True):
            res = self._opt(free)
            assert res is not None
            assert res["trough_after_ramp"] >= SUFFICIENCY_NMOL_L - 1e-6
            assert res["max_sed"] <= GAMMA_MED * _med_sed_by_fst(3) + 1e-9

    def test_control_arm_really_is_uniformly_spaced(self):
        res = self._opt(free_spacing=False)
        assert res is not None
        assert res["gamma"] == pytest.approx(1.0)

    def test_rebuild_reproduces_the_scored_schedule(self):
        """The stored parameters must regenerate exactly what was scored, or
        every downstream figure and table describes a different schedule."""
        from vitd_optimization import rebuild_joint_schedule
        from vitd_model import simulate as _sim
        res = self._opt(free_spacing=True)
        assert res is not None
        days, seds = rebuild_joint_schedule(res, horizon_days=self.HORIZON)
        sched = build_variable_schedule(days, seds, fst=int(res["fst"]))
        r = _sim(fst=int(res["fst"]), schedule=sched,
                 duration_days=self.HORIZON, D0=25.0,
                 dose_framing="fixed_per_session_SED")
        assert r["cancer_risk"]["combined_index"] == pytest.approx(
            res["combined_risk"], rel=1e-6)
        assert r["cumulative_SED"] == pytest.approx(res["total_sed"], rel=1e-6)

    def test_free_spacing_gets_no_larger_search_budget(self):
        """Both arms must optimise the same number of parameters. Otherwise
        differential_evolution (population = popsize * n_params) hands the free
        arm more evaluations, and any gain is confounded with search effort."""
        import inspect
        from vitd_optimization import optimise_joint_schedule
        src = inspect.getsource(optimise_joint_schedule)
        assert "if free_spacing:\n            bounds = bounds + " not in src, (
            "spacing dimension must be present in both arms")

    def test_free_spacing_is_never_worse_than_uniform(self):
        """gamma = 1 is inside the bounds, so the free arm can express the
        control exactly and should not lose beyond solver noise."""
        fixed = self._opt(free_spacing=False)
        free = self._opt(free_spacing=True)
        assert fixed is not None and free is not None
        assert free["combined_risk"] <= fixed["combined_risk"] * 1.10

    def test_infeasible_budget_returns_none(self):
        assert self._opt(free_spacing=True, n=1) is None


class TestTimeToSufficiency:
    def test_returns_nan_when_target_is_never_reached(self):
        r = simulate(fst=3, schedule=[(0.0, 0.1)], duration_days=60, D0=25.0,
                     dose_framing="fixed_per_session_SED")
        assert np.isnan(time_to_sufficiency(r, 500.0))

    def test_crossing_is_interpolated_not_quantised_to_the_grid(self):
        r = simulate(fst=3, schedule=[(float(d), 2.0) for d in range(0, 200, 3)],
                     duration_days=200, D0=25.0,
                     dose_framing="fixed_per_session_SED")
        ttf = time_to_sufficiency(r, SUFFICIENCY_NMOL_L)
        assert np.isfinite(ttf)
        t, D = np.asarray(r["t"]), np.asarray(r["D"])
        assert t[0] <= ttf <= t[-1]
        # D at the reported time is the target, to within the local slope
        assert float(np.interp(ttf, t, D)) == pytest.approx(
            SUFFICIENCY_NMOL_L, abs=0.5)

    def test_a_faster_schedule_reaches_sufficiency_sooner(self):
        """Both arms must actually reach the target, or this compares a number
        against NaN and passes/fails for the wrong reason. At 1 SED every 5 days
        FST III never reaches 50 nmol/L at all."""
        slow = simulate(fst=3, schedule=[(float(d), 2.0) for d in range(0, 300, 5)],
                        duration_days=300, D0=25.0,
                        dose_framing="fixed_per_session_SED")
        fast = simulate(fst=3, schedule=[(float(d), 2.0) for d in range(0, 300, 3)],
                        duration_days=300, D0=25.0,
                        dose_framing="fixed_per_session_SED")
        t_slow = time_to_sufficiency(slow, SUFFICIENCY_NMOL_L)
        t_fast = time_to_sufficiency(fast, SUFFICIENCY_NMOL_L)
        assert np.isfinite(t_slow) and np.isfinite(t_fast)
        assert t_fast < t_slow


class TestRecalibratedFeasibility:
    """Schedules the 2026-08-17 K_const recalibration (0.33 -> 0.98) removed,
    and the 2026-08-18 A recalibration (2.353 -> 3.114) partially restored.

    These are RESULTS, asserted so they cannot be quietly undone. Every
    constraint here is the standing one -- the 50 nmol/L floor, the 90-day
    ramp-in, GAMMA_MED, the session-length ceiling. Nothing is relaxed.

    THE COMBINED EFFECT IS NOT K_const ALONE. K_const 0.33 -> 0.98 in
    isolation pushes FST III-VI all infeasible at N = 104 (twice weekly). The
    subsequent A recalibration (2.353 -> 3.114, raising overall synthesis
    efficiency) pulls FST III and IV back to feasible at N = 104, but not FST
    V or VI: the melanin penalty at K_const = 0.98 is too large for the
    higher synthesis rate to offset within a non-burning, <=120-minute
    session. The dark-skin conclusion (FST V-VI need more than twice-weekly
    exposure to hold the floor) survives the combined calibration; the FST
    III/IV conclusion the class used to assert does not, and is corrected
    below rather than deleted.

    Full map, all six skin types x five session budgets, with the binding
    constraint per cell: scripts/feasibility_map_kconst.py.
    """

    HORIZON, RAMP = 365, 90

    def _uniform(self, fst, n, params=None):
        from vitd_optimization import optimise_variable_dose
        return optimise_variable_dose(fst=fst, n_sessions=n, family="uniform",
                                      horizon_days=self.HORIZON,
                                      ramp_in_days=self.RAMP, params=params)

    @pytest.mark.parametrize("fst", [5, 6])
    def test_twice_weekly_no_longer_holds_the_floor_for_fst_v_and_vi(self, fst):
        """N = 104 (twice weekly) is infeasible for FST V and VI under the
        CURRENT combined calibration (K_const = 0.98, A = 3.114).

        It was feasible for all of III-VI at K_const = 0.33 (either A). FST
        III and IV were ALSO infeasible under K_const = 0.98 with the prior
        A = 2.353, but recovered when A moved to 3.114 -- see
        test_fst_iii_and_iv_recover_under_the_combined_calibration below. FST
        V and VI do not recover at this budget: the higher synthesis rate is
        not enough to offset their melanin penalty. FST I and II are
        unaffected -- II because its M_baseline is 0.07, I because it was
        already infeasible at this budget for an unrelated reason (its low
        MED caps the per-session dose).
        """
        assert self._uniform(fst, 104) is None

    def test_fst_ii_twice_weekly_still_holds(self):
        """The contrast that shows this is a PIGMENT effect and not a
        general tightening: the least-pigmented feasible type is untouched."""
        assert self._uniform(2, 104) is not None

    def test_fst_iii_and_iv_recover_under_the_combined_calibration(self):
        """FST III and IV are feasible at N = 104 under the CURRENT
        parameters (K_const = 0.98, A = 3.114) -- the attribution the old,
        misnamed test_the_infeasibility_is_caused_by_k_const was trying to
        make no longer holds for these two skin types.

        At K_const = 0.98 with the PRIOR A = 2.353, both are infeasible at
        this budget: K_const raised alone would have left all of III-VI
        infeasible at N = 104. The A recalibration is what pulls III and IV
        back, and does not extend to V/VI (see
        test_fst_v_infeasibility_is_a_joint_kconst_and_a_effect below).
        """
        intermediate = ModelParams()
        intermediate.K_const, intermediate.A = 0.98, 2.353
        for fst in (3, 4):
            assert self._uniform(fst, 104, params=intermediate) is None, (
                f"FST {fst} should still be infeasible at K_const=0.98, A=2.353")
            assert self._uniform(fst, 104) is not None, (
                f"FST {fst} should be feasible at the current K_const=0.98, A=3.114")

    def test_fst_v_infeasibility_is_a_joint_kconst_and_a_effect(self):
        """2x2 attribution over (K_const, A) for FST V, the cell where the A
        recalibration does NOT rescue feasibility (unlike III/IV above).

        Raising K_const alone (A held at the old 2.353) makes N = 104
        infeasible. Raising A alone (K_const held at the old 0.33) leaves it
        feasible -- A is not itself a tightening. Only when BOTH move to
        their current values does the cell stay infeasible, so pinning this
        on "K_const" alone (the old test name) was already an
        oversimplification even before A moved: it is the K_const penalty
        that FST III/IV happen to escape via the A increase and FST V does
        not, which is a joint statement, not a single-parameter one.
        """
        k033_a2353 = ModelParams()
        k033_a2353.K_const, k033_a2353.A = 0.33, 2.353
        k098_a2353 = ModelParams()
        k098_a2353.K_const, k098_a2353.A = 0.98, 2.353
        k033_a3114 = ModelParams()
        k033_a3114.K_const, k033_a3114.A = 0.33, 3.114
        # current default ModelParams() is K_const=0.98, A=3.114

        assert self._uniform(5, 104, params=k033_a2353) is not None
        assert self._uniform(5, 104, params=k098_a2353) is None
        assert self._uniform(5, 104, params=k033_a3114) is not None
        assert self._uniform(5, 104) is None

    @pytest.mark.parametrize("fst,n_min,n_below", [
        (1, 104, 52), (2, 104, 52), (3, 104, 52), (4, 104, 52),
        (5, 146, 104), (6, 183, 146),
    ])
    def test_minimum_feasible_session_budget_by_skin_type(self, fst, n_min, n_below):
        """The headline scheduling result: how many sessions a year each skin
        type needs to hold sufficiency on sun alone, within a non-burning dose
        and a 120-minute session. Monotone in pigmentation, as it must be.

        Pinned from BOTH sides on the TRI_SESSION_BUDGETS grid: feasible
        (holds the floor AND fits the 120-minute ceiling) at n_min, and NOT
        both of those at the next budget down. FST V and VI are the cases
        where the binding constraint differs: FST VI at n=146 already holds
        the 50 nmol/L floor, but needs ~159 min/day, over the 120-minute
        ceiling -- its minimum is set by the session-length ceiling, not by
        the erythema/floor constraint that sets FST I-V.
        """
        from vitd_objectives import sed_from_minutes
        res = self._uniform(fst, n_min)
        assert res is not None, f"FST {fst} infeasible at its stated minimum"
        minutes = res["max_sed"] / sed_from_minutes(1.0, 6.0)
        assert minutes <= 120.0 + 1e-9

        below = self._uniform(fst, n_below)
        below_ok = (below is not None and
                    below["max_sed"] / sed_from_minutes(1.0, 6.0) <= 120.0 + 1e-9)
        assert not below_ok, (
            f"FST {fst} should NOT be feasible-and-under-120-min at n={n_below}")

    def test_fst_vi_cannot_be_scheduled_in_twenty_minute_sessions(self):
        """Dark-skin priority, stated as a constraint rather than a footnote:
        under the recalibrated penalty no interval on the grid holds the floor
        for FST VI within a 20-minute session. It needs ~42 min daily."""
        r = optimal_interval_risk_constrained(fst=6, max_minutes=20.0)
        assert r["recommended"] is None
        assert r["note"]
        loose = optimal_interval_risk_constrained(fst=6)["recommended"]
        assert loose is not None
        assert loose["minutes_per_session"] > 20.0


# ---------------------------------------------------------------------------
# PRIMARY objective: minimum dose / minimum sessions to reach and hold
# ---------------------------------------------------------------------------
class TestPrimaryMinimumSchedule:
    """The headline objective. A recommender whose returned schedule does not
    actually hold the floor is worse than none, so the floor is re-checked by
    an INDEPENDENT simulation rather than by the optimiser's own bookkeeping.
    """

    HORIZON, RAMP = 365, 90
    BUDGETS = (104, 183, 365)
    UVI, MAXMIN = 6.0, 120.0

    def _run(self, fst, **kw):
        from vitd_optimization import minimum_schedule_for_sufficiency
        kw.setdefault("session_budgets", self.BUDGETS)
        kw.setdefault("horizon_days", self.HORIZON)
        kw.setdefault("ramp_in_days", self.RAMP)
        kw.setdefault("uv_index", self.UVI)
        kw.setdefault("max_minutes", self.MAXMIN)
        return minimum_schedule_for_sufficiency(fst=fst, **kw)

    def test_returned_schedule_actually_holds_the_floor(self):
        from vitd_optimization import session_days, dose_uniform, build_variable_schedule
        res = self._run(3)
        rec = res["fewest_sessions"]
        assert rec is not None
        days = session_days(rec["n_sessions"], self.HORIZON)
        sched = build_variable_schedule(
            days, dose_uniform(days, rec["sed_per_session"]), fst=3)
        sim = simulate(fst=3, schedule=sched, duration_days=self.HORIZON,
                       params=ModelParams(), D0=25.0,
                       dose_framing="fixed_per_session_SED")
        t, D = np.asarray(sim["t"]), np.asarray(sim["D"])
        assert float(np.min(D[t >= self.RAMP])) >= SUFFICIENCY_NMOL_L - 1e-6

    def test_never_burns_and_fits_the_session_ceiling(self):
        """C1 and C7 on every feasible row, for every skin type."""
        from vitd_physiology import _med_sed_by_fst
        from vitd_objectives import GAMMA_MED
        for fst in range(1, 7):
            for row in self._run(fst)["rows"]:
                if not row["feasible"]:
                    continue
                assert row["sed_per_session"] <= GAMMA_MED * _med_sed_by_fst(fst) + 1e-9
                assert row["minutes_per_session"] <= self.MAXMIN + 1e-6

    def test_infeasible_cells_are_returned_not_dropped(self):
        """Item A3(a): an infeasible cell is a result. The grid must come back
        whole, with a binding constraint attached to every failure."""
        res = self._run(6)
        assert len(res["rows"]) == len(self.BUDGETS)
        for row in res["rows"]:
            if not row["feasible"]:
                assert row["binding_constraint"], row

    def test_binding_constraint_is_one_of_the_declared_ones(self):
        for fst in range(1, 7):
            for row in self._run(fst)["rows"]:
                if row["feasible"]:
                    continue
                assert ("C1" in row["binding_constraint"]
                        or "C7" in row["binding_constraint"]), row

    def test_fewest_sessions_really_is_the_smallest_feasible_budget(self):
        for fst in (2, 5):
            res = self._run(fst)
            ok = [r["n_sessions"] for r in res["rows"] if r["feasible"]]
            if not ok:
                assert res["fewest_sessions"] is None
                continue
            assert res["fewest_sessions"]["n_sessions"] == min(ok)

    def test_least_total_dose_really_is_the_smallest_total(self):
        res = self._run(3)
        ok = [r["total_sed"] for r in res["rows"] if r["feasible"]]
        assert ok
        assert res["least_total_dose"]["total_sed"] == pytest.approx(min(ok))

    def test_fewest_sessions_costs_more_UV_than_least_dose(self):
        """The two minima are different questions, and the report says so.
        Per-session yield saturates, so squeezing the year into fewer, larger
        sessions buys the same floor with more cumulative UV."""
        res = self._run(3)
        fw, ld = res["fewest_sessions"], res["least_total_dose"]
        assert fw is not None and ld is not None
        assert fw["total_sed"] >= ld["total_sed"] - 1e-9

    def test_minimum_budget_is_monotone_in_pigmentation(self):
        """Darker skin cannot need FEWER sessions. A violation would mean the
        melanin penalty had inverted, which is exactly the failure the
        2026-08-17 K_const recalibration was made to remove."""
        mins = []
        for fst in range(1, 7):
            res = self._run(fst)
            fw = res["fewest_sessions"]
            mins.append(fw["n_sessions"] if fw else float("inf"))
        assert mins == sorted(mins), mins

    def test_fst_vi_is_reported_even_where_infeasible(self):
        """Dark-skin priority, as a constraint rather than a footnote: FST VI
        must produce rows at every budget whether or not any of them work."""
        res = self._run(6)
        assert res["fst"] == 6
        assert len(res["rows"]) == len(self.BUDGETS)
        if not res["feasible"]:
            assert res["note"]

    def test_free_spacing_never_raises_the_minimum_budget(self):
        """gamma = 1 is in the grid, so the free search can always reproduce
        the uniform answer and cannot do worse. Where it does BETTER that is a
        result -- C3 binds at the ramp-in boundary, so reaching target sooner
        can rescue a budget uniform spacing misses."""
        from vitd_optimization import TRI_GAMMA_GRID
        assert 1.0 in TRI_GAMMA_GRID
        for fst in (3, 6):
            uni = self._run(fst)["fewest_sessions"]
            free = self._run(fst, gammas=TRI_GAMMA_GRID)["fewest_sessions"]
            if uni is None:
                continue
            assert free is not None
            assert free["n_sessions"] <= uni["n_sessions"]

    def test_rejects_bad_input(self):
        from vitd_optimization import minimum_schedule_for_sufficiency
        with pytest.raises(ValueError):
            minimum_schedule_for_sufficiency(fst=0)
        with pytest.raises(ValueError):
            minimum_schedule_for_sufficiency(fst=3, session_budgets=())
        with pytest.raises(ValueError):
            minimum_schedule_for_sufficiency(fst=3, session_budgets=(0,))


# ---------------------------------------------------------------------------
# Problem 1+4: location/clothing-specific geometry threaded through both
# schedule functions. `geometry_factor`/`exposed_bsa` default to None, so
# every pre-existing caller must be a byte-for-byte no-op; a non-None
# geometry_factor must move the result exactly the way calling
# sed_from_minutes with that same geometry_factor would -- never twice.
# ---------------------------------------------------------------------------
class TestLocationAwareGeometryWiring:
    def test_none_geometry_factor_is_a_no_op_for_risk_constrained(self):
        explicit = optimal_interval_risk_constrained(fst=3, uv_index=6.0,
                                                      geometry_factor=None)
        default = optimal_interval_risk_constrained(fst=3, uv_index=6.0)
        assert explicit["recommended"] == default["recommended"]

    def test_none_geometry_and_bsa_is_a_no_op_for_minimum_schedule(self):
        # Rows can carry NaN (infeasible cells), and NaN != NaN under `==`,
        # so a plain dict-equality check would fail even for identical rows.
        # Compare field-by-field, treating NaN as equal to NaN.
        from math import isnan
        from vitd_optimization import minimum_schedule_for_sufficiency
        explicit = minimum_schedule_for_sufficiency(fst=3, uv_index=6.0,
                                                     exposed_bsa=None,
                                                     geometry_factor=None)
        default = minimum_schedule_for_sufficiency(fst=3, uv_index=6.0)
        assert len(explicit["rows"]) == len(default["rows"])
        for row_a, row_b in zip(explicit["rows"], default["rows"]):
            assert row_a.keys() == row_b.keys()
            for key in row_a:
                a, b = row_a[key], row_b[key]
                if isinstance(a, float) and isinstance(b, float) and isnan(a) and isnan(b):
                    continue
                assert a == b, (key, a, b)

    def test_higher_geometry_factor_lowers_minutes_needed_risk_constrained(self):
        """More horizontal UV actually reaching skin should let the same
        capped dose be delivered in fewer minutes -- the same relationship
        sed_from_minutes itself embodies (max_minutes=None isolates this from
        the session-length ceiling, which does depend on geometry_factor)."""
        low = optimal_interval_risk_constrained(fst=3, uv_index=6.0,
                                                 geometry_factor=0.2,
                                                 max_minutes=None)["recommended"]
        high = optimal_interval_risk_constrained(fst=3, uv_index=6.0,
                                                  geometry_factor=0.6,
                                                  max_minutes=None)["recommended"]
        assert low is not None and high is not None
        assert high["minutes_per_session"] < low["minutes_per_session"]

    def test_risk_constrained_geometry_factor_matches_manual_sed_from_minutes(self):
        """No double-application: with max_minutes=None the erythema-capped
        sed_per_session is independent of geometry_factor, so the ratio
        between two geometry_factor runs' minutes_per_session must equal the
        ratio sed_from_minutes gives directly for those two factors."""
        from vitd_objectives import sed_from_minutes
        r1 = optimal_interval_risk_constrained(fst=3, uv_index=6.0,
                                               geometry_factor=0.2,
                                               interval_grid=(7,), max_minutes=None)
        r2 = optimal_interval_risk_constrained(fst=3, uv_index=6.0,
                                               geometry_factor=0.6,
                                               interval_grid=(7,), max_minutes=None)
        row1, row2 = r1["evaluated"][0], r2["evaluated"][0]
        assert row1["sed_per_session"] == pytest.approx(row2["sed_per_session"], rel=1e-9)
        expected_ratio = (sed_from_minutes(1.0, 6.0, 0.2)
                          / sed_from_minutes(1.0, 6.0, 0.6))
        actual_ratio = row2["minutes_per_session"] / row1["minutes_per_session"]
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-9)

    def test_exposed_bsa_changes_minimum_schedule_dose(self):
        """exposed_bsa scales synthesis -- a different mechanism than
        geometry_factor, which scales dose per minute -- confirm it actually
        reaches simulate(): more exposed skin needs less dose per session to
        reach the same target."""
        from vitd_optimization import minimum_schedule_for_sufficiency
        small_bsa = minimum_schedule_for_sufficiency(
            fst=3, uv_index=6.0, exposed_bsa=0.10, max_minutes=None,
            session_budgets=(365,))["rows"][0]
        large_bsa = minimum_schedule_for_sufficiency(
            fst=3, uv_index=6.0, exposed_bsa=0.60, max_minutes=None,
            session_budgets=(365,))["rows"][0]
        assert small_bsa["feasible"] and large_bsa["feasible"]
        assert large_bsa["sed_per_session"] < small_bsa["sed_per_session"]

    def test_minimum_schedule_geometry_factor_matches_manual_sed_from_minutes(self):
        from vitd_optimization import minimum_schedule_for_sufficiency
        from vitd_objectives import sed_from_minutes
        r1 = minimum_schedule_for_sufficiency(
            fst=3, uv_index=6.0, geometry_factor=0.2, max_minutes=None,
            session_budgets=(365,))["rows"][0]
        r2 = minimum_schedule_for_sufficiency(
            fst=3, uv_index=6.0, geometry_factor=0.6, max_minutes=None,
            session_budgets=(365,))["rows"][0]
        assert r1["feasible"] and r2["feasible"]
        assert r1["sed_per_session"] == pytest.approx(r2["sed_per_session"], rel=1e-6)
        expected_ratio = (sed_from_minutes(1.0, 6.0, 0.2)
                          / sed_from_minutes(1.0, 6.0, 0.6))
        actual_ratio = r2["minutes_per_session"] / r1["minutes_per_session"]
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)


# ---------------------------------------------------------------------------
# SECONDARY objective: the tri-objective frontier
# ---------------------------------------------------------------------------
class TestTriObjective:
    """Best SED at least spacing, fewest sessions, accounting for skin cancer.
    Efficacy is a constraint here, so the checks are that every point on the
    frontier really does hold the floor, and that the frontier really is
    nondominated on all three axes."""

    BUDGETS = (183, 365)
    GAMMAS = (1.0, 1.4)

    def _run(self, fst):
        from vitd_optimization import tri_objective_frontier
        return tri_objective_frontier(fst=fst, session_budgets=self.BUDGETS,
                                      gammas=self.GAMMAS)

    def test_max_gap_is_minimised_by_uniform_spacing(self):
        """The A1-vs-A2 tension the frontier exists to show: at a fixed budget
        no warp beats uniform spacing on the largest-gap axis."""
        from vitd_optimization import warp_days, max_gap_days
        for n in (52, 183):
            uni = max_gap_days(warp_days(n, 365, 1.0), 365)
            for g in (0.5, 0.7, 1.4, 2.0, 2.5):
                assert max_gap_days(warp_days(n, 365, g), 365) >= uni - 1e-9

    def test_max_gap_includes_the_tail(self):
        """A front-loaded schedule that stops in autumn has a real gap to the
        end of the horizon; excluding it would make gamma > 1 look free."""
        from vitd_optimization import warp_days, max_gap_days
        days = warp_days(10, 365, 2.5)
        assert max_gap_days(days, 365) >= 365.0 - float(days[-1]) - 1e-9

    def test_pareto_front_is_internally_nondominated(self):
        from vitd_optimization import pareto_front_nd
        pts = [{"a": 1.0, "b": 5.0}, {"a": 2.0, "b": 2.0},
               {"a": 3.0, "b": 9.0}, {"a": 2.0, "b": 2.0}]
        front = pareto_front_nd(pts, ("a", "b"))
        assert {"a": 3.0, "b": 9.0} not in front
        assert {"a": 1.0, "b": 5.0} in front

    def test_pareto_front_keeps_a_strictly_better_point(self):
        from vitd_optimization import pareto_front_nd
        pts = [{"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 2.0, "b": 2.0, "c": 2.0}]
        front = pareto_front_nd(pts, ("a", "b", "c"))
        assert front == [pts[0]]

    def test_every_frontier_point_holds_the_floor(self):
        res = self._run(3)
        assert res["frontier"]
        for r in res["frontier"]:
            assert r["feasible"]
            assert r["trough_after_ramp"] >= SUFFICIENCY_NMOL_L - 1e-6

    def test_frontier_points_are_mutually_nondominated(self):
        res = self._run(3)
        axes = res["axes"]
        for p in res["frontier"]:
            for q in res["frontier"]:
                if p is q:
                    continue
                assert not (all(q[k] <= p[k] for k in axes)
                            and any(q[k] < p[k] for k in axes))

    def test_infeasible_cells_are_reported_not_dropped(self):
        res = self._run(6)
        assert len(res["rows"]) == len(self.BUDGETS) * len(self.GAMMAS)
        assert all("feasible" in r for r in res["rows"])

    def test_all_three_axes_are_present_on_every_row(self):
        res = self._run(3)
        for r in res["rows"]:
            for k in ("n_sessions", "max_gap_days", "combined_risk"):
                assert k in r

    def test_rejects_bad_skin_type(self):
        from vitd_optimization import tri_objective_frontier
        with pytest.raises(ValueError):
            tri_objective_frontier(fst=7)


# ---------------------------------------------------------------------------
# DIAGNOSTIC tier: why gross_synth_per_SED is background only
# ---------------------------------------------------------------------------
class TestGrossSynthDiagnostic:
    """Item A3(2). These assert the CURRENT model still reproduces both
    failures. If a recalibration ever makes gross_synth_per_SED well-behaved,
    these fail and the hierarchy gets revisited on purpose rather than by
    accident."""

    def test_argmax_tracks_the_search_bound(self):
        """A real interior optimum does not move when the bound moves."""
        from vitd_optimization import gross_synth_per_SED_diagnostic
        d = gross_synth_per_SED_diagnostic(fst=3, intervals=tuple(range(1, 91)),
                                           search_bounds=(30, 60, 90))
        assert d["argmax_tracks_search_bound"], d["argmax_by_search_bound"]
        assert d["argmax_by_search_bound"][30] < d["argmax_by_search_bound"][90]

    def test_metric_rises_as_dosing_falls(self):
        from vitd_optimization import gross_synth_per_SED_diagnostic
        d = gross_synth_per_SED_diagnostic(fst=3, intervals=tuple(range(1, 61)))
        assert d["spearman_rho_interval_vs_metric"] > 0.9

    def test_its_selected_schedule_ends_below_baseline(self):
        """The failure that disqualifies it as advice: a net LOSS of vitamin D
        scored as maximal efficiency."""
        from vitd_optimization import gross_synth_per_SED_diagnostic
        d = gross_synth_per_SED_diagnostic(fst=3, intervals=tuple(range(1, 91)))
        assert d["net_loss_at_selected"]
        assert d["final_D_at_selected"] < d["D0"]

    def test_it_is_labelled_diagnostic(self):
        from vitd_optimization import gross_synth_per_SED_diagnostic
        from vitd_objectives import TIER_DIAGNOSTIC
        assert gross_synth_per_SED_diagnostic(
            fst=3, intervals=(7, 14, 28))["tier"] == TIER_DIAGNOSTIC


# ---------------------------------------------------------------------------
# Item A3(a): the 13c table must cover all six skin types
# ---------------------------------------------------------------------------
class TestJointScheduleCoverage:

    def test_default_skin_types_span_i_to_vi(self):
        """FST VI is this project's priority population and FST I is the other
        end of the range; the headline optimisation table ran neither before
        Phase 4b."""
        import inspect
        from vitd_optimization import analysis_joint_schedule
        default = inspect.signature(
            analysis_joint_schedule).parameters["fsts"].default
        assert tuple(default) == (1, 2, 3, 4, 5, 6)

    def test_infeasible_budget_still_yields_a_row_shape(self):
        """optimise_joint_schedule returns None for an infeasible cell; the
        analysis layer must turn that into a flagged ROW rather than a gap.
        Checked on the contract both sides share."""
        from vitd_optimization import optimise_joint_schedule
        assert optimise_joint_schedule(fst=6, n_sessions=1) is None
