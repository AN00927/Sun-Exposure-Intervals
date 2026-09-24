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
    are satisfied by independent re-simulation, not by its own bookkeeping."""

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
        """The correct monotonicity direction. Tighter cap pushes toward more
        frequent, smaller sessions -- the model's frequent-and-small result.
        Longer intervals need a larger per-session dose to hold the floor,
        and the intermittent term is steeply increasing in dose."""
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

        Fixture rebased 2026-08-17, REBASED AGAIN 2026-09-06 for the
        K_const/M_baseline_by_FST/BSA_EXP move. FST IV's own natural
        requirement under `optimal_interval_risk_constrained` (which picks a
        SINGLE interval from a fixed grid, not the variable-dose optimiser
        `TestRecalibratedFeasibility` uses) rose from ~16 min/day to
        ~22.4 min/day, pushing it past the previous 20-minute fixture cap
        into infeasibility. This optimiser's cap does not reduce
        minutes-per-session below the interval's natural requirement -- it
        only accepts or rejects each interval on the grid -- so a "tight" cap
        that stays feasible must sit AT OR ABOVE that natural value; there is
        no intermediate cap that is both stricter than the natural minutes
        and still feasible (verified 2026-09-06 by sweeping FST IV's cap from
        15 to 22 minutes: every value returns infeasible). FST IV's cap moves
        20.0 -> 25.0 (natural requirement ~22.4 min); FST III and VI are
        unaffected (~13.7 and ~54.9 min/day respectively, both still under
        their existing caps).
        """
        for fst, mm in ((6, 60.0), (4, 25.0), (3, 20.0)):
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
        """d_i = 0 is 'no session that day', letting session count be optimised
        without integer variables."""
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
    # FIXTURE REBASED 2026-08-17: N 104 -> 146. These tests exercise
    # properties of the optimiser (a nested family cannot lose to the family it
    # nests; the returned profile holds the floor), and they need a
    # feasible operating point. FST III at N = 104 stopped being one when
    # K_const was recalibrated 0.33 -> 0.98: holding 50 nmol/L
    # on 104 sessions now needs more than 70% MED per session in FST III.
    #
    # No constraint was relaxed to do this. The sufficiency floor, GAMMA_MED
    # and the ramp-in are untouched; only the session budget the fixture uses
    # moved, to one that is still feasible. The infeasibility itself is a
    # result and is asserted explicitly in TestRecalibratedFeasibility below,
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
        """The Jensen argument, checked on the model: at fixed total dose and
        session count, equal doses minimise the intermittency term, so any real
        gain must show up as less total UV."""
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

    def test_unreachable_target_returns_none(self):
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

    def test_free_spacing_gets_no_larger_search_budget(self, monkeypatch):
        """Both arms must optimise the same number of parameters. Otherwise
        differential_evolution (population = popsize * n_params) hands the free
        arm more evaluations, and any gain is confounded with search effort.

        REWRITTEN (was an `inspect.getsource` substring match on a single
        literal indentation, which any reformat of `optimise_joint_schedule`
        would silently defeat (mutation-tested 2026-09-22; introducing the
        bug this test guards against was confirmed to fail it)). This version
        patches `scipy.optimize.differential_evolution` (imported locally
        inside `optimise_joint_schedule` on every call) and records the
        actual `bounds` length the solver is invoked with for each arm, so
        it tests the real runtime search-space size rather than the source
        text that produces it.
        """
        import scipy.optimize
        from vitd_optimization import optimise_joint_schedule

        class _StopAfterFirstCall(Exception):
            pass

        captured = {}

        def fake_differential_evolution(func, bounds, **kwargs):
            captured["n_bounds"] = len(bounds)
            raise _StopAfterFirstCall()

        monkeypatch.setattr(scipy.optimize, "differential_evolution",
                            fake_differential_evolution)

        n_bounds_by_arm = {}
        for free in (False, True):
            captured.clear()
            with pytest.raises(_StopAfterFirstCall):
                optimise_joint_schedule(
                    fst=3, n_sessions=self.N, horizon_days=self.HORIZON,
                    ramp_in_days=self.RAMP, free_spacing=free, seed=0,
                    maxiter=8, popsize=6)
            assert "n_bounds" in captured, (
                "differential_evolution was never called for "
                f"free_spacing={free}")
            n_bounds_by_arm[free] = captured["n_bounds"]

        assert n_bounds_by_arm[True] == n_bounds_by_arm[False], (
            f"free_spacing=True searches {n_bounds_by_arm[True]} parameters "
            f"vs {n_bounds_by_arm[False]} for free_spacing=False; both arms "
            "must search the same number of parameters so a risk gain "
            "cannot be confounded with a larger search budget "
            "(differential_evolution sizes its population as "
            "popsize * n_params)")

    def test_free_spacing_is_never_worse_than_uniform(self):
        """gamma = 1 is inside the bounds, so the free arm can express the
        control exactly and should not lose beyond solver noise.

        Tolerance: `optimise_joint_schedule`'s own docstring (and the
        duplicate note in `vitd_optimization_reports.py`, ~line 281)
        documents a directly-measured, single-cell (FST III, N=146),
        5-seed spread of **6.3%** in `combined_risk` -- "[reference purged
        2026-09-19]" but the figure itself was retained inline. This test
        runs a single seed per arm (matching production's default
        `n_seeds=1`), so its observed ratio is one draw from that ~6.3%-wide
        distribution, not a repeatable point estimate. The previous 10%
        slack was only ~1.5x that documented spread (decision E3
        calls this "~1.5sigma", i.e. flaky-prone under the repo's own
        stated convention of treating the documented spread as one sigma
        unit). Widened here to 3x the documented 6.3% figure (18.9% slack,
        `1.0 + 3 * 0.063 = 1.189`), a conservative multiple chosen because:
        (a) it uses the OLDER, LARGER of the two measurements on record --
        `RESULTS.md` (~line 2146)'s most recent regenerated run measured a
        tighter 2.3% spread at the same cell, but that figure is a
        single re-run, not a second controlled replication, so the more
        conservative 6.3% is kept as the basis; (b) this test's solver
        settings (maxiter=8, popsize=6) are smaller than production's
        (maxiter=18, popsize=10), which plausibly widens rather than
        narrows true noise, so erring conservative is the safer direction.
        Seed-averaging (`n_seeds>1`, supported directly by
        `optimise_joint_schedule`) was considered instead of/alongside
        widening, but measured at ~39s per single-seed call of this test
        (2026-09-22, this machine); n_seeds=3 would roughly triple that for
        one test, which was judged not worth the wall-clock for a test that
        already has headroom via the tolerance widening alone.
        """
        fixed = self._opt(free_spacing=False)
        free = self._opt(free_spacing=True)
        assert fixed is not None and free is not None
        assert free["combined_risk"] <= fixed["combined_risk"] * 1.189

    def test_infeasible_budget_returns_none(self):
        assert self._opt(free_spacing=True, n=1) is None


class TestTimeToSufficiency:
    def test_returns_nan_when_target_is_never_reached(self):
        r = simulate(fst=3, schedule=[(0.0, 0.1)], duration_days=60, D0=25.0,
                     dose_framing="fixed_per_session_SED")
        assert np.isnan(time_to_sufficiency(r, 500.0))

    def test_crossing_is_interpolated(self):
        r = simulate(fst=3, schedule=[(float(d), 2.0) for d in range(0, 200, 3)],
                     duration_days=200, D0=25.0,
                     dose_framing="fixed_per_session_SED")
        ttf = time_to_sufficiency(r, SUFFICIENCY_NMOL_L)
        assert np.isfinite(ttf)
        t, D = np.asarray(r["t"]), np.asarray(r["D"])
        assert t[0] <= ttf <= t[-1]
        # The reported time must BRACKET the crossing: everything strictly
        # before it is below target, and the target is reached at it.
        #
        # This used to assert np.interp(ttf, t, D) == target +/- 0.5, which is
        # not a well-posed test of this function. simulate() emits two samples
        # at the same t for every session (pre- and post-pulse), so `t` is
        # non-decreasing rather than increasing and np.interp is undefined on
        # the jump -- it returns the post-pulse value. When the crossing falls
        # on a session, D steps ACROSS the target discontinuously and never
        # equals it: here D goes 49.33 -> 50.62 at t = 60.0 with dt = 0. The
        # old assertion passed only while the crossing happened to land on a
        # continuous stretch, and a parameter change that moved it onto a pulse
        # exposed that. time_to_sufficiency itself is correct.
        i = int(np.flatnonzero(D >= SUFFICIENCY_NMOL_L)[0])
        assert D[:i].max() < SUFFICIENCY_NMOL_L
        assert D[i] >= SUFFICIENCY_NMOL_L
        assert t[i - 1] <= ttf <= t[i]

    def test_a_faster_schedule_reaches_sufficiency_sooner(self):
        """Both arms must reach the target, or this compares a number
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
    """REBASED 2026-09-06 for the literature-evidence-pass calibration:
    K_const 0.98 -> 1.15 (coupled refit), M_baseline_by_FST moved to the
    Narea 2026 table, BSA_EXP 0.50 -> 0.3281 (all `calibration_approval.yaml`,
    2026-09-06 entries). Supersedes the 2026-08-17/18 rebase this class used
    to describe (K_const 0.33 -> 0.98, A 2.353 -> 3.114); that history is
    preserved in git blame, not restated here.

    These are results, asserted so they cannot be quietly undone. Every
    constraint here is the standing one -- the 50 nmol/L floor, the 90-day
    ramp-in, GAMMA_MED, the session-length ceiling. Nothing is relaxed.

    SUPERSEDED AND REWRITTEN LATER THE SAME DAY (2026-09-06), AFTER THE A
    RE-ANCHOR. An earlier version of this class recorded a dramatic tightening
    -- FST II newly infeasible at N = 104, the FST III/IV "recovery" fully
    undone, FST VI needing N = 365 to fit under 120 min. THAT WAS MEASURED
    WITH A STILL AT ITS STALE 3.114 AND IS NO LONGER TRUE. When A was
    re-anchored to 4.125 (calibration_approval.yaml, A_reanchor_2026_09_06 --
    the Young 2021 anchor recomputed after BSA_EXP and K_const/M_baseline
    moved its own denominator), the tightening was undone almost exactly.

    THE HEADLINE, RESTATED. The four-parameter move (K_const, M_baseline_by_FST,
    BSA_EXP, K_SC) all push synthesis DOWN; A carries the absolute level and
    had not yet been recomputed, so the intermediate state overstated the
    tightening. With A at its anchored value the minimum
    feasible-and-under-120-min budgets are:

        FST      I    II   III    IV     V     VI
        N_min  104   104   104   104   146    183

    which is EXACTLY the pre-2026-09-06 (2026-08-18) landscape. Measured by
    direct `optimise_variable_dose` calls, 2026-09-06, minutes/day at UV
    index 6 over a 365-day horizon with a 90-day ramp:

        N        52      104      146      183      260      365
        I    infeas   26.5m    17.3m    12.9m     8.6m     6.0m
        II   infeas   33.4m    21.3m    15.7m    10.4m     7.1m
        III  infeas   44.2m    27.3m    19.9m    12.9m     8.8m
        IV   infeas   76.4m    43.7m    30.7m    19.2m    12.8m
        V    infeas  135.9m!   70.5m    47.3m    28.3m    18.3m
        VI   infeas  291.4m!  127.9m!   79.6m    44.3m    27.4m
        (! = holds the 50 nmol/L floor but exceeds the 120-minute ceiling)

    SO THE PIGMENT GRADIENT SURVIVES AND THE SCHEDULING CONCLUSION IS
    UNCHANGED: twice-weekly (N = 104) cannot be scheduled within 120 minutes
    for FST V or VI, while FST I-IV can. The dark-skin conclusion this class
    exists to pin -- that FST V-VI need more than twice-weekly exposure to
    hold the floor on sun alone -- is intact, and is now reached with a
    LARGER melanin penalty (K_const 1.15 vs 0.98) and a LARGER synthesis
    scalar (A 4.125 vs 3.114) that happen to offset.

    WHAT THIS EPISODE SHOWS, worth keeping: feasibility here is a JOINT
    property of the pigment penalty and the absolute synthesis level, and
    reading either one alone produces a wrong headline. The intermediate
    8/10-validation state was internally consistent and completely
    misleading about the scheduling result.

    Full map, all six skin types x five session budgets, with the binding
    constraint per cell: scripts/feasibility_map_kconst.py (NOT re-run for
    this rebase -- it derives K_const from its own literal, not from
    ModelParams(), so its printed numbers are stale; the facts asserted here
    come from direct `optimise_variable_dose` calls, reproduced above).

    REBASED AGAIN 2026-09-23, AND THIS TIME THE DARK-SKIN HEADLINE MOVED.
    The 2026-09-22 parameter pass (K_const 1.17 -> 2.096070737357932 under
    random-effects pooling of the LIVE-re-inverted field routes, the A3
    ITA -> M clip release, SC_max IV-VI 0.36 -> 0.24, A re-anchored to
    4.359560972608079 at the joint fixed point, and the body-weight dilution
    term made live) tightens FST VI and leaves FST V where it was. Measured
    by direct `optimise_variable_dose` calls at UV index 6 over a 365-day
    horizon with a 90-day ramp, minutes per session (! = over the
    120-minute ceiling):

        N        52      104      146      183      260      365
        I    infeas    26.8m    17.4m    13.0m     8.7m     6.0m
        II   infeas    35.8m    22.6m    16.7m    11.0m     7.5m
        III  infeas    51.8m    31.4m    22.6m    14.6m     9.9m
        IV   infeas    87.6m    49.4m    34.4m    21.4m    14.1m
        V    infeas   infeas    93.2m    60.8m    35.4m    22.4m
        VI   infeas   infeas   269.4m!  148.4m!   74.1m    43.1m

        FST      I    II   III    IV     V     VI
        N_min  104   104   104   104   146    260      (was 183 for FST VI)

    TWO CHANGES A READER WOULD NOTICE. (1) FST VI's minimum feasible budget
    moves 183 -> 260 sessions a year, i.e. from roughly every other day to
    roughly five days a week. (2) At N = 104 FST V and VI are now
    FLOOR-INFEASIBLE rather than feasible-but-over-the-ceiling, so the
    twice-weekly exclusion is stronger than the version this class used to
    record. Both follow from the K_const move, which is itself dominated by
    a BUG FIX (the field routes were a stale 2026-08-17 cache) rather than
    by new evidence -- see vitd_model.py's K_const docstring entry.

    AND THE UNCERTAINTY THAT TRAVELS WITH THEM. The leave-one-out envelope
    on K_const is 1.588440 to 3.183999, and with A re-anchored at each end
    FST V's N_min runs 104 (low end) to 260 (high end) and FST VI's 183 to
    "no feasible budget up to daily". No number in this class may be quoted
    as a bare fact; see calibration_approval.yaml,
    K_const_leave_one_out_envelope_2026_09_22.
    """

    HORIZON, RAMP = 365, 90

    # Read from the code rather than restated, so an attribution row can
    # never silently describe a parameter set that is no longer shipped --
    # which is exactly how the stale-A rows below came to be written.
    K_SHIPPED = ModelParams().K_const
    A_SHIPPED = ModelParams().A

    def _uniform(self, fst, n, params=None):
        from vitd_optimization import optimise_variable_dose
        return optimise_variable_dose(fst=fst, n_sessions=n, family="uniform",
                                      horizon_days=self.HORIZON,
                                      ramp_in_days=self.RAMP, params=params)

    @pytest.mark.parametrize("fst", [5, 6])
    def test_twice_weekly_cannot_be_scheduled_within_120_min_for_fst_v_and_vi(self, fst):
        """N = 104 (twice weekly) cannot be scheduled for FST V or VI.

        THE BINDING CONSTRAINT CHANGED ON 2026-09-22, AND IT IS A HARDER
        EXCLUSION THAN BEFORE, NOT A SOFTER ONE. History, because this test
        has now said three different things and each was measured, not
        argued:
          * pre-2026-09-06 and again after the A re-anchor: `is None`, i.e.
            floor-infeasible;
          * 2026-09-06 to 2026-09-21: FEASIBLE but over the ceiling -- the
            optimiser returned a schedule holding the 50 nmol/L floor at
            N = 104 needing 135.9 (V) and 291.4 (VI) min/day, so the binding
            constraint was session LENGTH;
          * from the 2026-09-22 parameter pass (K_const 1.17 ->
            2.096070737357932 under random-effects pooling of the
            live-re-inverted field routes; ITA -> M clip released; SC_max
            IV-VI 0.36 -> 0.24; body-weight dilution now live): back to
            FLOOR-INFEASIBLE. At N = 104 no dose profile inside the erythema
            ceiling holds 50 nmol/L for FST V or VI at all, at any session
            length.
        Both halves are asserted, so a future change that restores
        feasibility-but-over-the-ceiling fails here and has to say so rather
        than passing quietly on the weaker of the two claims.
        """
        res = self._uniform(fst, 104)
        assert res is None, (
            f"FST {fst} at N=104 is now expected to be FLOOR-INFEASIBLE "
            f"inside the erythema ceiling, not merely over the 120-minute "
            f"session ceiling; re-derive this test rather than relaxing it")
        # ... and the contrast that makes it a pigment result: the same
        # budget at the same parameters is comfortably feasible for FST I-IV
        # (test_twice_weekly_still_works_for_fst_i_through_iv).

    @pytest.mark.parametrize("fst", [1, 2, 3, 4])
    def test_twice_weekly_still_works_for_fst_i_through_iv(self, fst):
        """The contrast that makes the above a PIGMENT result rather than a
        general tightening: at the same N = 104, FST I-IV are feasible AND
        inside the 120-minute ceiling (26.5 / 33.4 / 44.2 / 76.4 min/day).
        Added 2026-09-06 alongside the narrowing above."""
        from vitd_objectives import sed_from_minutes
        res = self._uniform(fst, 104)
        assert res is not None
        assert res["max_sed"] / sed_from_minutes(1.0, 6.0) <= 120.0

    def test_fst_i_twice_weekly_still_holds(self):
        """The one skin type this combined calibration leaves untouched at
        N = 104: FST I's own low MED already capped its per-session dose
        before any of K_const/M_baseline/BSA_EXP moved, and M_baseline(I)
        barely moves (0.0 -> 0.0381)."""
        assert self._uniform(1, 104) is not None

    def _under_ceiling(self, fst, n, kc, a):
        """Feasible AND inside the 120-minute ceiling, at an explicit
        (K_const, A). Returns (bool, minutes_or_None). Added 2026-09-06:
        after the A re-anchor, `result is None` no longer separates the
        cases -- FST V/VI hold the floor at N=104 while needing 136-291
        min/day -- so attribution has to test the ceiling explicitly.
        """
        from vitd_objectives import sed_from_minutes
        p = ModelParams()
        p.K_const, p.A = kc, a
        r = self._uniform(fst, n, params=p)
        if r is None:
            return False, None
        minutes = r["max_sed"] / sed_from_minutes(1.0, 6.0)
        return minutes <= 120.0, minutes

    def test_fst_iii_and_iv_recovery_survives_the_full_calibration(self):
        """FST III and IV remain schedulable at N = 104 under the shipped
        2026-09-06 parameters -- but ONLY because K_const and A moved
        together.

        REWRITTEN 2026-09-06 (was
        test_fst_iii_and_iv_recovery_is_undone_by_m_baseline_and_bsa_exp).
        The "undone" version was measured with A stale at 3.114 and is
        false once A is re-anchored.

        REBASED 2026-09-23 onto the shipped (K_const, A) pair, which is now
        READ FROM ModelParams() rather than written out -- the rows below
        that named a literal 1.15 were describing a K_const that had not
        been shipped for two days by the time they failed. Measured
        attribution at N = 104 (minutes/session at UV index 6):

            K_const     A        FST III        FST IV
            1.17      4.139      39.7m ok       51.1m ok   (2026-09-21 pair)
            2.0961    3.114      infeasible     infeasible (incomplete edit!)
            2.0961    4.3596     51.8m ok       87.6m ok   (SHIPPED)
            2.0961    2.353      infeasible     infeasible (pre-A4 A)

        The middle row is still the whole point: raising the pigment penalty
        WITHOUT recomputing the synthesis scalar it invalidated makes both
        skin types infeasible. That state was never a defensible model, only
        an incomplete edit. The recovery genuinely depends on A being large.
        """
        for fst in (3, 4):
            for kc, a, label, expect in [
                (1.17, 4.139, "the 2026-09-21 shipped pair", True),
                (self.K_SHIPPED, 3.114, "new K_const, STALE A -- incomplete edit", False),
                (self.K_SHIPPED, self.A_SHIPPED, "shipped pair", True),
                (self.K_SHIPPED, 2.353, "new K_const, pre-A4 A", False),
            ]:
                got, minutes = self._under_ceiling(fst, 104, kc, a)
                assert got == expect, (
                    f"FST {fst} at K_const={kc}, A={a} ({label}): expected "
                    f"{'schedulable' if expect else 'not schedulable'}, got "
                    f"{'schedulable' if got else 'not schedulable'} "
                    f"({'%.1f min' % minutes if minutes else 'infeasible'})")

    def test_fst_v_feasibility_is_a_joint_kconst_and_a_effect(self):
        """Attribution over (K_const, A) for FST V at N = 104, on the shipped
        M_baseline_by_FST and BSA_EXP.

        REBASED 2026-09-06 to the ceiling criterion (see _under_ceiling).
        REBASED 2026-09-23 onto the shipped pair, read from ModelParams().
        Measured minutes/session at N = 104, UV index 6:

            K_const     A        FST V
            2.0961    3.114     infeasible
            1.17      4.139      72.7m  -> schedulable (2026-09-21 pair)
            1.0       4.3596     56.1m  -> schedulable
            1.5       4.3596     92.6m  -> schedulable
            2.0961    5.3       119.6m  -> schedulable, just inside
            2.0961    5.2       124.8m  -> NOT schedulable, just outside
            2.0961    4.3596    floor-infeasible (SHIPPED)

        The joint story is intact and is now sharper still. Holding the
        shipped K_const, twice-weekly FST V becomes schedulable only once A
        reaches 5.292416573008921 -- the 5.3/5.2 pair above brackets that
        threshold -- against a shipped A of 4.359560972608079, i.e. the
        exclusion holds with 1.213979253017023x headroom in A. And holding
        the shipped A, it becomes schedulable again at a K_const somewhere
        below 1.5, which the leave-one-out envelope's low end (1.588440)
        does not quite reach but the 2026-09-21 value (1.17) comfortably
        did. So FST V's twice-weekly exclusion is a genuine pigment result
        at the shipped parameters, and it is one that the pigment penalty's
        own uncertainty can overturn -- which is why the envelope travels
        with it.
        """
        for kc, a, expect in [
            (self.K_SHIPPED, 3.114, False),
            (1.17, 4.139, True),                  # the 2026-09-21 pair
            (1.0, self.A_SHIPPED, True),
            (1.5, self.A_SHIPPED, True),
            (self.K_SHIPPED, 5.3, True),               # just above the threshold
            (self.K_SHIPPED, 5.2, False),              # just below it
            (self.K_SHIPPED, self.A_SHIPPED, False),        # shipped
        ]:
            got, minutes = self._under_ceiling(5, 104, kc, a)
            assert got == expect, (
                f"FST V at K_const={kc}, A={a}: expected {expect}, got {got} "
                f"({'%.1f min' % minutes if minutes else 'infeasible'})")

    @pytest.mark.parametrize("fst,n_min,n_below", [
        (1, 104, 52), (2, 104, 52), (3, 104, 52), (4, 104, 52),
        (5, 146, 104), (6, 260, 183),
    ])
    def test_minimum_feasible_session_budget_by_skin_type(self, fst, n_min, n_below):
        """The headline scheduling result: how many sessions a year each skin
        type needs to hold sufficiency on sun alone, within a non-burning dose
        and a 120-minute session. Monotone in pigmentation, as it must be.

        REBASED 2026-09-23: FST VI's minimum moves 183 -> 260 sessions a
        year (and 183 is now over the ceiling at 148.4 min, not under it),
        while FST I-V are unchanged at 104/104/104/104/146. This is the one
        headline budget the 2026-09-22 parameter pass actually moved, and it
        moves in the burdensome direction for the darkest skin type: about
        five days a week rather than every other day. The full minutes
        table, and the leave-one-out envelope it must be quoted with, are in
        this class's own docstring.

        REBASED TWICE ON 2026-09-06, AND IT ENDED BACK WHERE IT STARTED.
        The first rebase recorded 104/146/146/146/183/365 -- every skin type
        except FST I needing MORE sessions after the
        M_baseline_by_FST/BSA_EXP/K_const move. That was measured with A
        still stale at 3.114. Re-anchoring A to 4.125
        (calibration_approval.yaml, A_reanchor_2026_09_06) restores
        104/104/104/104/146/183, which is the pre-2026-09-06 landscape
        exactly.

        This is not a no-op dressed up as one. The same budgets are now
        reached with a LARGER pigment penalty (K_const 0.98 -> 1.15, and
        every M_baseline cell raised) and a LARGER synthesis scalar
        (A 3.114 -> 4.125). Those two offset at the level of this table
        while the underlying parameters are better anchored than before --
        M_baseline on n=1005 rather than n=48, A on a recomputed rather than
        stale kernel. The scheduling conclusion is unchanged; its provenance
        is not.
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
        """Under the recalibrated penalty no interval on the grid holds the
        floor for FST VI within a 20-minute session. It needs ~42 min daily."""
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
    independent simulation rather than by the optimiser's own bookkeeping.
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
        """Darker skin cannot need fewer sessions. A violation would mean the
        melanin penalty had inverted, which is exactly the failure the
        2026-08-17 K_const recalibration was made to remove."""
        mins = []
        for fst in range(1, 7):
            res = self._run(fst)
            fw = res["fewest_sessions"]
            mins.append(fw["n_sessions"] if fw else float("inf"))
        assert mins == sorted(mins), mins

    def test_fst_vi_is_reported_even_where_infeasible(self):
        """FST VI must produce rows at every budget whether or not any of them
        work."""
        res = self._run(6)
        assert res["fst"] == 6
        assert len(res["rows"]) == len(self.BUDGETS)
        if not res["feasible"]:
            assert res["note"]

    def test_free_spacing_never_raises_the_minimum_budget(self):
        """gamma = 1 is in the grid, so the free search can always reproduce
        the uniform answer and cannot do worse. Where it does better, that is a
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
        """More horizontal UV reaching skin should let the same capped dose be
        delivered in fewer minutes -- the same relationship sed_from_minutes
        itself embodies (max_minutes=None isolates this from the session-length
        ceiling, which does depend on geometry_factor)."""
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
        geometry_factor, which scales dose per minute. More exposed skin needs
        less dose per session to reach the same target."""
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
    frontier holds the floor, and that the frontier is nondominated on all
    three axes."""

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
    """Item A3(2). These assert the current model still reproduces both
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
        """The failure that disqualifies it as advice: a net loss of vitamin D
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
        analysis layer must turn that into a flagged row rather than a gap.
        Checked on the contract both sides share."""
        from vitd_optimization import optimise_joint_schedule
        assert optimise_joint_schedule(fst=6, n_sessions=1) is None
