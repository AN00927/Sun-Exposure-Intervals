"""Tests for the mechanistic sub-models added in the July 2026 literature pass.

Each test states the literature fact it encodes, so a future change that breaks
one has to argue with a paper rather than with a number.
"""

import numpy as np
import pytest

from vitd_model import (ModelParams, simulate, build_pulse_schedule,
                        build_weekly_schedule, synthesis_per_session)
from vitd_physiology import (body_weight_dilution_factor,
                             photodegradation_factor,
                             hydroxylation_efficiency,
                             cancer_risk,
                             BODY_WEIGHT_REF_KG)


class TestBodyWeightDilution:
    """Drincic 2012: hyperbolic in total body WEIGHT, not linear, not BMI."""

    def test_neutral_when_unknown_or_reference(self):
        assert body_weight_dilution_factor(None) == 1.0
        assert body_weight_dilution_factor(BODY_WEIGHT_REF_KG) == pytest.approx(1.0)

    def test_heavier_dilutes_lighter_concentrates(self):
        assert body_weight_dilution_factor(100.0) < 1.0
        assert body_weight_dilution_factor(55.0) > 1.0

    def test_hyperbolic_not_linear(self):
        """1/weight: doubling weight must halve the factor exactly."""
        assert (body_weight_dilution_factor(140.0)
                == pytest.approx(0.5 * body_weight_dilution_factor(70.0)))

    def test_applies_to_dietary_route_too(self):
        """Dilution is a distribution-volume effect, not a skin effect."""
        light = ModelParams(dietary_D_daily_ug=25.0, body_weight_kg=55.0)
        heavy = ModelParams(dietary_D_daily_ug=25.0, body_weight_kg=110.0)
        rl = simulate(fst=3, schedule=[], duration_days=200, params=light, D0=25.0)
        rh = simulate(fst=3, schedule=[], duration_days=200, params=heavy, D0=25.0)
        assert rl["D"][-1] > rh["D"][-1]

    def test_obese_needs_roughly_double_dose(self):
        """Ekwaru 2014: obese need 2-3x, overweight ~1.5x the oral dose."""
        ratio = (body_weight_dilution_factor(70.0)
                 / body_weight_dilution_factor(140.0))
        assert 1.8 <= ratio <= 2.2


class TestPhotodegradation:
    """Holick 1981 plateau; van Dijk 2016 net production cancelled by ~7.5 SED."""

    def test_neutral_at_vanishing_dose(self):
        assert photodegradation_factor(0.0) == 1.0
        assert photodegradation_factor(1e-9) == pytest.approx(1.0, abs=1e-6)

    def test_monotone_decreasing_in_dose(self):
        vals = [photodegradation_factor(s) for s in [0.5, 1, 2, 4, 8, 16, 32]]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_absolute_yield_saturates(self):
        """Total per-session yield must plateau."""
        yields = [s * photodegradation_factor(s) for s in [1, 10, 100, 1000]]
        assert all(a < b for a, b in zip(yields, yields[1:])), "still monotone"
        assert yields[-1] / yields[0] < 10.0

    def test_splitting_a_budget_beats_one_session(self):
        """With photodegradation on, N small sessions must beat 1 large one
        at IDENTICAL cumulative dose. This is a real interior optimum, unlike
        the feedback artifact in MODEL_AND_EVIDENCE.md, 'Remaining gaps',
        gap 19."""
        p = ModelParams(use_photodegradation=True)
        total = 24.0
        one = simulate(fst=3, schedule=[(0.0, total)], duration_days=84,
                       params=p, D0=25.0)
        many = simulate(fst=3,
                        schedule=[(float(i * 3), total / 8) for i in range(8)],
                        duration_days=84, params=p, D0=25.0)
        synth_one = sum(x[2] for x in one["synthesis_log"])
        synth_many = sum(x[2] for x in many["synthesis_log"])
        assert synth_many > synth_one
        assert one["cumulative_SED"] == pytest.approx(many["cumulative_SED"])

    def test_off_by_default(self):
        assert ModelParams().use_photodegradation is False


class TestHydroxylationFeedback:
    """Comms Med 2025: biphasic 25-hydroxylation, inflection ~56 nmol/L."""

    def test_half_efficiency_at_inflection(self):
        assert hydroxylation_efficiency(56.0) == pytest.approx(0.5, rel=1e-6)

    def test_monotone_decreasing(self):
        vals = [hydroxylation_efficiency(d) for d in [10, 30, 56, 100, 200]]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_never_reaches_zero(self):
        """Unlike (1 - D/D_max), cutaneous synthesis never stops outright."""
        assert hydroxylation_efficiency(500.0) > 0.0
        assert hydroxylation_efficiency(10_000.0) > 0.0

    def test_biphasic_mode_differs_from_linear_above_dmax(self):
        pl, pb = ModelParams(), ModelParams()
        pb.feedback_mode = "biphasic"
        m = pl.M_baseline(3)
        assert synthesis_per_session(1.0, m, m, 200.0, 0.0, pl) == 0.0
        assert synthesis_per_session(1.0, m, m, 200.0, 0.0, pb) > 0.0

    def test_linear_remains_default(self):
        assert ModelParams().feedback_mode == "linear"


class TestTransitDelay:
    """Previtamin->D3 ~8 h; serum D3 t_half ~4 h; 25(OH)D sampled at 24 h."""

    def test_off_by_default(self):
        assert ModelParams().use_transit_delay is False

    def test_delays_the_rise(self):
        p = ModelParams(use_transit_delay=True)
        q = ModelParams(use_transit_delay=False)
        a = simulate(fst=3, schedule=[(0.0, 5.0)], duration_days=10, params=p, D0=25.0)
        b = simulate(fst=3, schedule=[(0.0, 5.0)], duration_days=10, params=q, D0=25.0)
        i = int(np.argmin(np.abs(a["t"] - 0.5)))
        assert a["D"][i] < b["D"][i], "delayed run must lag at 12 h"

    def test_pool_drains_completely(self):
        p = ModelParams(use_transit_delay=True)
        r = simulate(fst=3, schedule=[(0.0, 5.0)], duration_days=90, params=p, D0=25.0)
        assert r["D3_pool_final"] < 1e-9

    def test_no_D_lost_relative_to_instant(self):
        """Delay should redistribute in time, not destroy vitamin D. Over a long
        horizon the delayed run ends slightly LOWER only via extra clearance."""
        p = ModelParams(use_transit_delay=True)
        q = ModelParams(use_transit_delay=False)
        a = simulate(fst=3, schedule=[(0.0, 5.0)], duration_days=90, params=p, D0=25.0)
        b = simulate(fst=3, schedule=[(0.0, 5.0)], duration_days=90, params=q, D0=25.0)
        assert a["D"][-1] == pytest.approx(b["D"][-1], rel=0.05)


class TestCancerRisk:
    """Elwood 1997: intermittent OR 1.71 for melanoma, heavy occupational
    OR 0.86 (protective). SCC tracks cumulative dose (Calzavara-Pinton 2015)."""

    def test_scc_depends_only_on_cumulative_dose(self):
        a = cancer_risk([1.0] * 52, 2)
        b = cancer_risk([2.0] * 26, 2)
        c = cancer_risk([26.0] * 2, 2)
        assert a["total_sed"] == pytest.approx(b["total_sed"])
        assert a["scc_index"] == pytest.approx(b["scc_index"])
        assert a["scc_index"] == pytest.approx(c["scc_index"])

    def test_intermittent_risk_penalises_bursts(self):
        """THE key scheduling result: at fixed total dose, many small sessions
        carry far less melanoma/BCC risk than a few large ones."""
        small = cancer_risk([1.0] * 52, 2)
        large = cancer_risk([26.0] * 2, 2)
        assert large["intermittent_index"] > 10 * small["intermittent_index"]

    def test_burn_fraction_tracks_med(self):
        low = cancer_risk([0.5] * 10, 2)   # well under FST II MED of 2.6
        high = cancer_risk([5.0] * 10, 2)  # well over
        assert low["burn_fraction"] == 0.0
        assert high["burn_fraction"] == 1.0

    def test_darker_skin_lower_risk(self):
        risks = [cancer_risk([1.0] * 52, f)["combined_index"] for f in range(1, 7)]
        assert risks == sorted(risks, reverse=True)

    def test_combined_index_is_scale_free(self):
        """Regression: an arithmetic mean of the two indices was ~100%
        intermittent risk, because that index is cubic in dose and spans ~0.1-450
        while SCC spans ~0.3-5. Geometric mean keeps both objectives visible."""
        r = cancer_risk([2.0] * 365, 5)
        expected = np.sqrt(r["scc_index"] * r["intermittent_index"])
        assert r["combined_index"] == pytest.approx(expected)
        lo, hi = sorted([r["scc_index"], r["intermittent_index"]])
        assert lo <= r["combined_index"] <= hi

    def test_combined_index_monotone_in_dose(self):
        vals = [cancer_risk([s] * 365, 5)["combined_index"]
                for s in (0.25, 0.5, 1.0, 2.0, 4.0)]
        assert vals == sorted(vals)

    def test_empty_schedule_is_zero_risk(self):
        r = cancer_risk([], 3)
        assert r["scc_index"] == 0.0
        assert r["intermittent_index"] == 0.0
        assert r["total_sed"] == 0.0

    def test_exposed_in_simulate_output(self):
        r = simulate(fst=5, schedule=build_pulse_schedule(84, 1.0, 7),
                     duration_days=84, params=ModelParams(), D0=25.0)
        assert "cancer_risk" in r
        assert r["cancer_risk"]["total_sed"] == pytest.approx(r["cumulative_SED"])


class TestBackwardCompatibility:
    """All new mechanisms default off/neutral."""

    def test_defaults_neutral(self):
        p = ModelParams()
        assert p.body_weight_kg is None
        assert p.use_photodegradation is False
        assert p.use_transit_delay is False
        assert p.feedback_mode == "linear"

    def test_default_run_is_stable(self):
        """Guards the headline scenario against accidental drift."""
        r = simulate(fst=5, schedule=build_pulse_schedule(84, 1.0, 1),
                     duration_days=84, params=ModelParams(), D0=25.0)
        assert r["D"][-1] == pytest.approx(63.75, abs=0.5)


class TestTwoProcessSC:
    """de Boer 2025 Supp Table S1: SC falls ~26% BELOW baseline at week 5
    (5.8 -> 4.3 -> 7.5 um) before overshooting at week 9.

    The old single multiplicative form (`SC *= exp(-k_damage*sed)`) could only
    shrink SC toward zero, so no k_damage could ever reproduce that. These tests
    lock in the corrected structure -- and lock in that k_damage stays 0.0,
    because fixing the structure did NOT make the parameter identifiable.
    """

    def test_k_damage_still_zero_by_default(self):
        """Structure was fixed; the parameter is still unidentifiable from
        three group medians. See scripts/fit_sc_deboer2025.py."""
        assert ModelParams().k_damage == 0.0

    def test_default_SC_trajectory_unchanged(self):
        """Backward compatibility: at k_damage=0 the two-process form must
        reproduce the old single-process numbers exactly."""
        p = ModelParams()
        r = simulate(fst=3, schedule=build_weekly_schedule(63, 0.8, [0, 2, 4]),
                     duration_days=63, params=p, D0=25.0)
        assert r["SC"][-1] == pytest.approx(0.1360, abs=1e-3)

    def test_SC_can_go_below_baseline(self):
        """The capability the old parameterisation lacked entirely."""
        p = ModelParams()
        p.k_damage = 0.10
        r = simulate(fst=3, schedule=build_weekly_schedule(63, 0.8, [0, 2, 4]),
                     duration_days=63, params=p, D0=25.0)
        assert r["SC"].min() < 0.0

    def test_old_form_could_not_go_below_zero(self):
        """Guards the diagnosis: a multiplicative damage term is sign-preserving,
        so it can only ever shrink a non-negative SC toward zero."""
        sc = 0.2
        for _ in range(200):
            sc = sc * np.exp(-0.5 * 0.8)
        assert sc > 0.0

    def test_components_exposed_and_consistent(self):
        p = ModelParams()
        p.k_damage = 0.05
        r = simulate(fst=3, schedule=build_weekly_schedule(63, 0.8, [0, 2, 4]),
                     duration_days=63, params=p, D0=25.0)
        assert r["SC"][-1] == pytest.approx(
            r["SC_build_final"] - r["SC_damage_final"], abs=1e-9)

    def test_more_damage_lowers_net_SC(self):
        finals = []
        for kd in (0.0, 0.05, 0.10, 0.20):
            p = ModelParams()
            p.k_damage = kd
            r = simulate(fst=3, schedule=build_weekly_schedule(63, 0.8, [0, 2, 4]),
                         duration_days=63, params=p, D0=25.0)
            finals.append(r["SC"][-1])
        assert finals == sorted(finals, reverse=True)

    def test_damage_repairs_when_dosing_stops(self):
        """Acute damage must clear on a days timescale once UV stops."""
        p = ModelParams()
        p.k_damage = 0.10
        sched = build_weekly_schedule(21, 0.8, [0, 2, 4])
        r = simulate(fst=3, schedule=sched, duration_days=120, params=p, D0=25.0)
        assert r["SC_damage_final"] < 1e-3

    def test_fit_script_reports_structural_failure(self):
        """The fit is expected to FAIL. If it ever starts passing, either the
        model gained a mechanism or someone widened the tolerance -- both need
        a human look."""
        from scripts.fit_sc_deboer2025 import run as fit_run
        out = fit_run(verbose=False)
        assert out["any_acceptable"] is False
        for f in out["fits"].values():
            assert f["relative_rmse"] > 0.5

    def test_fractional_med_insult_falls(self):
        """Constant 0.8 SED against a rising MED means the relative insult
        declines ~23-36% over the study."""
        from scripts.fit_sc_deboer2025 import fractional_med_trend
        fr = fractional_med_trend(verbose=False)
        for stratum, (f0, f1, f2) in fr.items():
            assert f2 < f0, stratum
