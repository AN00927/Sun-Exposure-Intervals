"""Structural invariants of the vitamin D model.

These tests encode properties that must hold for ANY parameter setting, as
opposed to calibration checks against specific studies (which live in
vitd_validation.py). Each test here corresponds to a real bug that reached
main and was only found by running the model rather than reading it.
"""

import numpy as np
import pytest

from vitd_model import (ModelParams, simulate, build_pulse_schedule,
                        build_weekly_schedule, DIETARY_UG_TO_NMOL_PER_L)


ALL_FST = [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Regression: dietary intake drove D to zero
# ---------------------------------------------------------------------------
# The sampling loop in simulate() computed delta_time = ti - current_t but
# never advanced current_t, so delta_time grew to the whole simulation window.
# The dietary term was added once per sample without being decayed, while the
# recorded value was decayed over an ever-longer interval. Net effect:
# dietary_D_daily_ug=25 gave D(365d) = 0.0 nmol/L (underflow from ~1e-267)
# while the internal accumulator reached 232,849.
class TestDietaryPathway:

    def test_dietary_intake_raises_D_without_UV(self):
        """1000 IU/day with no UV must raise D, not collapse it to zero."""
        p = ModelParams(dietary_D_daily_ug=25.0)
        r = simulate(fst=3, schedule=[], duration_days=365, params=p, D0=25.0)
        assert r["D"][-1] > 25.0, "dietary intake must raise 25(OH)D"
        assert np.all(np.isfinite(r["D"]))

    def test_dietary_response_is_monotone_in_dose(self):
        """More dietary D3 must give a higher steady state."""
        finals = []
        for ug in [0.0, 10.0, 25.0, 50.0]:
            p = ModelParams(dietary_D_daily_ug=ug)
            r = simulate(fst=3, schedule=[], duration_days=365, params=p, D0=25.0)
            finals.append(r["D"][-1])
        assert finals == sorted(finals), f"non-monotone dietary response: {finals}"

    def test_dietary_magnitude_is_physiological(self):
        """1000 IU/day should land in a plausible band, not orders out.

        Supplementation trials (Cashman 2008/2011) put 25 ug/day at roughly
        +1.5-2.0 nmol/L per ug/day at steady state from a low baseline.
        Band is deliberately wide -- this guards against unit errors, not
        calibration drift.
        """
        p = ModelParams(dietary_D_daily_ug=25.0)
        r = simulate(fst=3, schedule=[], duration_days=365, params=p, D0=25.0)
        assert 60.0 < r["D"][-1] < 250.0, f"implausible dietary steady state {r['D'][-1]}"

    def test_dietary_constant_is_documented(self):
        assert DIETARY_UG_TO_NMOL_PER_L > 0


# ---------------------------------------------------------------------------
# Regression: UV made FST VI skin LIGHTER
# ---------------------------------------------------------------------------
# A single global M_max=0.95 sat below M_baseline_by_FST[6]=1.00, so the
# saturating pulse update M <- M_max - (M_max - M)*exp(-alpha*sed) DECREASED
# melanin for FST VI. FST V/VI is this project's target population.
class TestMelaninMonotonicity:

    @pytest.mark.parametrize("fst", ALL_FST)
    def test_UV_increases_melanin_for_every_fst(self, fst):
        p = ModelParams()
        sched = build_pulse_schedule(84, 1.0, 1)
        r = simulate(fst=fst, schedule=sched, duration_days=84, params=p, D0=25.0)
        assert r["M"][-1] > p.M_baseline(fst), (
            f"FST {fst}: UV must raise melanin, got "
            f"{p.M_baseline(fst)} -> {r['M'][-1]}"
        )

    @pytest.mark.parametrize("fst", ALL_FST)
    def test_melanin_ceiling_strictly_above_baseline(self, fst):
        p = ModelParams()
        assert p.M_ceiling(fst) > p.M_baseline(fst)

    @pytest.mark.parametrize("fst", ALL_FST)
    def test_melanin_never_exceeds_ceiling(self, fst):
        """Even under absurd dosing, M must stay bounded by its ceiling."""
        p = ModelParams()
        sched = build_pulse_schedule(365, 20.0, 1)
        r = simulate(fst=fst, schedule=sched, duration_days=365, params=p, D0=25.0)
        assert r["M"].max() <= p.M_ceiling(fst) + 1e-9

    @pytest.mark.parametrize("fst", ALL_FST)
    def test_melanin_never_falls_below_baseline(self, fst):
        p = ModelParams()
        sched = build_weekly_schedule(120, 1.0, [0, 3])
        r = simulate(fst=fst, schedule=sched, duration_days=120, params=p, D0=25.0)
        assert r["M"].min() >= p.M_baseline(fst) - 1e-9


# ---------------------------------------------------------------------------
# Regression: Bogh 2012 read J/cm2 as if it were SED
# ---------------------------------------------------------------------------
class TestBoghDoseUnits:

    def test_session_doses_in_sed_reproduce_reported_cumulative(self):
        from vitd_protocols.bogh2012 import (
            bogh2012_session_doses_sed,
            BOGH2012_REPORTED_CUMULATIVE_J_PER_CM2,
            TL01_J_PER_CM2_TO_ERYTHEMAL_SED,
        )
        sed = bogh2012_session_doses_sed(n_sessions=18)
        expected = (BOGH2012_REPORTED_CUMULATIVE_J_PER_CM2
                    * TL01_J_PER_CM2_TO_ERYTHEMAL_SED)
        assert sum(sed) == pytest.approx(expected, rel=1e-6)

    def test_sed_doses_exceed_raw_j_per_cm2(self):
        """Guards the exact bug: SED must not equal the J/cm2 number."""
        from vitd_protocols.bogh2012 import (bogh2012_session_doses,
                                             bogh2012_session_doses_sed)
        raw = bogh2012_session_doses(n_sessions=18)
        sed = bogh2012_session_doses_sed(n_sessions=18)
        assert sum(sed) > 3.0 * sum(raw), (
            "erythemal SED must be several times the absolute J/cm2 value"
        )

    def test_doses_escalate_then_plateau(self):
        from vitd_protocols.bogh2012 import bogh2012_session_doses_sed
        sed = bogh2012_session_doses_sed(n_sessions=18)
        assert sed == sorted(sed), "escalating protocol must be non-decreasing"
        assert sed[-1] > sed[0]


# ---------------------------------------------------------------------------
# General conservation / limit behaviour
# ---------------------------------------------------------------------------
class TestConservationAndLimits:

    def test_no_UV_no_diet_decays_monotonically(self):
        p = ModelParams()
        r = simulate(fst=3, schedule=[], duration_days=200, params=p, D0=100.0)
        assert np.all(np.diff(r["D"]) <= 1e-9)
        assert r["D"][-1] < r["D"][0]

    def test_decay_matches_analytic_ode_solution(self):
        """dD/dt = -k(D)*D with k(D)=ln2*D^0.72/2953 integrates in closed form.

        A state-dependent-clearance model does NOT halve in one instantaneous
        half-life; asserting that would encode a fixed-k assumption the model
        explicitly rejects.
        """
        p = ModelParams()
        c = np.log(2.0) / 2953.0
        for D0, t in [(132.0, 89.0), (43.0, 199.0), (65.0, 148.0)]:
            exact = (D0 ** -0.72 + 0.72 * c * t) ** (-1.0 / 0.72)
            r = simulate(fst=3, schedule=[], duration_days=int(t), params=p, D0=D0)
            got = np.interp(t, r["t"], r["D"])
            assert got == pytest.approx(exact, rel=0.01)

    def test_zero_dose_pulses_are_inert(self):
        p = ModelParams()
        a = simulate(fst=5, schedule=[], duration_days=60, params=p, D0=40.0)
        b = simulate(fst=5, schedule=build_pulse_schedule(60, 0.0, 3),
                     duration_days=60, params=p, D0=40.0)
        assert b["D"][-1] == pytest.approx(a["D"][-1], rel=1e-9)

    @pytest.mark.parametrize("fst", ALL_FST)
    def test_D_stays_below_D_max(self, fst):
        p = ModelParams()
        sched = build_pulse_schedule(365, 15.0, 1)
        r = simulate(fst=fst, schedule=sched, duration_days=365, params=p,
                     D0=25.0, exposed_bsa_override=1.0)
        assert r["D"].max() <= p.D_max + 1e-6

    def test_all_states_finite_under_extreme_dosing(self):
        p = ModelParams(dietary_D_daily_ug=100.0)
        sched = build_pulse_schedule(365, 50.0, 1)
        r = simulate(fst=6, schedule=sched, duration_days=365, params=p, D0=10.0)
        for key in ("D", "M", "SC"):
            assert np.all(np.isfinite(r[key])), f"{key} went non-finite"

    def test_SC_bounded_by_its_max(self):
        p = ModelParams()
        for fst in ALL_FST:
            sched = build_weekly_schedule(365, 2.0, [0, 2, 4])
            r = simulate(fst=fst, schedule=sched, duration_days=365, params=p, D0=25.0)
            cap = p.SC_max_by_FST.get(fst, p.SC_max)
            assert r["SC"].max() <= cap + 1e-9
            assert r["SC"].min() >= -1e-12

    def test_higher_baseline_gives_smaller_rise(self):
        """Feedback direction (Bogh 2011): higher baseline D -> smaller delta."""
        p = ModelParams()
        sched = build_pulse_schedule(42, 1.0, 3)
        deltas = []
        for D0 in [20.0, 50.0, 80.0]:
            r = simulate(fst=3, schedule=sched, duration_days=42, params=p, D0=D0)
            deltas.append(r["D"][-1] - D0)
        assert deltas[0] > deltas[1] > deltas[2]

    def test_cumulative_SED_is_tracked_exactly(self):
        p = ModelParams()
        sched = build_pulse_schedule(84, 1.3, 7)
        r = simulate(fst=4, schedule=sched, duration_days=84, params=p, D0=25.0)
        assert r["cumulative_SED"] == pytest.approx(sum(s for _, s in sched))


class TestScheduleGridIndependence:
    """Pulses must be applied even when they fall off the 0.5-day sample grid."""

    def test_offgrid_pulses_are_all_applied(self):
        p = ModelParams()
        sched = [(0.3, 1.0), (1.7, 1.0), (5.15, 1.0)]
        r = simulate(fst=3, schedule=sched, duration_days=10, params=p, D0=25.0)
        assert len(r["synthesis_log"]) == 3
        assert r["cumulative_SED"] == pytest.approx(3.0)

    def test_darker_skin_synthesises_less_at_equal_dose(self):
        p = ModelParams()
        sched = build_pulse_schedule(42, 1.0, 3)
        finals = []
        for fst in ALL_FST:
            r = simulate(fst=fst, schedule=sched, duration_days=42, params=p, D0=25.0)
            finals.append(r["D"][-1])
        assert finals == sorted(finals, reverse=True), (
            f"D should decrease monotonically with FST, got {finals}"
        )
