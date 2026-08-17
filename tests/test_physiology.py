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
                             personal_geometry_factor,
                             hydroxylation_efficiency,
                             cancer_risk,
                             _med_sed_by_fst,
                             INTERMITTENT_BURN_EXPONENT,
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

    def test_applies_to_the_cutaneous_route(self):
        """Dilution is a distribution-volume effect, not a skin effect.

        Previously asserted via the oral route, which was removed on
        2026-08-16. Re-expressed through UV, which is the only input this
        model has now -- and is the route that actually matters here.
        """
        light = ModelParams(body_weight_kg=55.0)
        heavy = ModelParams(body_weight_kg=110.0)
        sched = build_weekly_schedule(200, 1.0, [0, 2, 4])
        rl = simulate(fst=3, schedule=sched, duration_days=200, params=light, D0=25.0)
        rh = simulate(fst=3, schedule=sched, duration_days=200, params=heavy, D0=25.0)
        assert rl["D"][-1] > rh["D"][-1]

    def test_heavier_body_needs_roughly_double_dose(self):
        """Ekwaru 2014: obese need 2-3x, overweight ~1.5x the dose.

        Ekwaru measured this orally; the factor here is a distribution-volume
        term and applies to cutaneous input for the same reason.
        """
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


class TestElwoodIntermittencyAnchor:
    """Ties INTERMITTENT_BURN_EXPONENT back to Elwood & Jopson 1997.

    The exponent is OR-ANCHORED, not fitted: the ORs give one equation in two
    unknowns (the exponent and the burst:continuous dose contrast, which the
    studies never report). These lock in the anchoring that was actually used,
    so that moving the exponent without revisiting the derivation fails loudly.
    Derivation: vitd_calibration.elwood1997_intermittency_check.
    """

    def test_model_reproduces_the_elwood_or_ratio_at_equal_cumulative_dose(self):
        """THE calibration check. At equal cumulative SED, the model's burst:
        continuous intermittent-index ratio must land inside the range of OR
        ratios Elwood & Jopson actually observed."""
        from vitd_objectives import GAMMA_MED
        from vitd_calibration import (ELWOOD_OR_INTERMITTENT, ELWOOD_OR_SUNBURN,
                                      ELWOOD_OR_OCCUPATIONAL)
        fst = 2
        med = _med_sed_by_fst(fst)
        # Burst arm at the erythema threshold, continuous arm at the
        # sub-erythemal ceiling, matched on cumulative SED (70 * med each).
        burst = cancer_risk([1.0 * med] * 70, fst)
        continuous = cancer_risk([GAMMA_MED * med] * 100, fst)
        assert burst["total_sed"] == pytest.approx(continuous["total_sed"])

        ratio = burst["intermittent_index"] / continuous["intermittent_index"]
        lo = ELWOOD_OR_INTERMITTENT / ELWOOD_OR_OCCUPATIONAL   # 1.71 / 0.86
        hi = ELWOOD_OR_SUNBURN / ELWOOD_OR_OCCUPATIONAL        # 1.91 / 0.86
        assert lo <= ratio <= hi, (
            f"model burst:continuous index ratio {ratio:.3f} falls outside the "
            f"Elwood & Jopson observed OR-ratio range [{lo:.3f}, {hi:.3f}]")

    def test_exponent_sits_inside_the_anchored_bracket(self):
        from vitd_calibration import elwood1997_intermittency_check
        r = elwood1997_intermittency_check(verbose=False)
        lo, hi = r["anchor_bracket"]
        assert lo <= INTERMITTENT_BURN_EXPONENT <= hi
        assert r["consistent"] is True

    def test_derivation_inverts_correctly(self):
        """k = 1 + ln(R)/ln(C) must round-trip: an exponent k fed back through
        the equal-cumulative-dose ratio C**(k-1) recovers R."""
        from vitd_calibration import elwood_intermittency_exponent
        for R, C in ((2.221, 1.4286), (1.988, 2.0), (3.0, 5.0)):
            k = elwood_intermittency_exponent(R, 1.0, C)
            assert C ** (k - 1.0) == pytest.approx(R, rel=1e-9)

    def test_derivation_rejects_degenerate_contrast(self):
        """C = 1 means the two arms had the same dose, under which no exponent
        can explain any OR difference."""
        from vitd_calibration import elwood_intermittency_exponent
        with pytest.raises(ValueError):
            elwood_intermittency_exponent(1.91, 0.86, 1.0)
        with pytest.raises(ValueError):
            elwood_intermittency_exponent(1.91, 0.0, 1.43)

    def test_susceptibility_cancels_from_within_skin_type_rankings(self):
        """The FST susceptibility is now derived rather than a prior (see
        TestSkinTypeSusceptibility), but this invariant must survive that: it is
        a pure multiplier, so schedule RANKINGS for one person cannot depend on
        it. Only cross-FST comparisons of absolute index values do."""
        a = cancer_risk([1.0] * 52, 2)
        b = cancer_risk([2.0] * 26, 2)
        c = cancer_risk([1.0] * 52, 5)
        d = cancer_risk([2.0] * 26, 5)
        # Same schedule pair, different FST: the RATIO must be identical.
        assert (a["intermittent_index"] / b["intermittent_index"]) == pytest.approx(
            c["intermittent_index"] / d["intermittent_index"])


class TestSkinTypeSusceptibility:
    """The FST susceptibility is a DERIVED basal-layer damage fraction.

    Fitted to exactly one number -- Fajuyigbe 2018's basal:whole CPD protection
    ratio (59.0/8.0) -- and checked against three sources it was not fitted to.
    These tests pin the fit, the checks, and the physical form, so that moving
    kappa or the melanin table without redoing the derivation fails loudly.
    Derivation: vitd_physiology section 4a; audit:
    vitd_calibration.skin_type_susceptibility_check.
    """

    def test_reproduces_the_fajuyigbe_ratio_it_was_fitted_to(self):
        from vitd_physiology import basal_damage_fraction, FAJUYIGBE2018_PROTECTION
        target = FAJUYIGBE2018_PROTECTION["basal"] / FAJUYIGBE2018_PROTECTION["whole"]
        assert basal_damage_fraction(1) / basal_damage_fraction(6) == pytest.approx(
            target, rel=1e-3)

    def test_predicts_gandini_2005_skin_type_I_vs_IV(self):
        """OUT-OF-SAMPLE. A CPD biomarker in two extreme phenotypes must land
        inside a melanoma meta-analysis of 60 studies: RR 2.09 (1.67-2.58)."""
        from vitd_calibration import (skin_type_susceptibility_check,
                                      GANDINI2005_RR_SKINTYPE_I_VS_IV_CI)
        r = skin_type_susceptibility_check(verbose=False)
        lo, hi = GANDINI2005_RR_SKINTYPE_I_VS_IV_CI
        assert lo <= r["derived_I_vs_IV"] <= hi
        assert r["gandini_consistent"] is True

    def test_premise_holds_med_ratio_matches_measured_whole_epidermis_protection(self):
        """OUT-OF-SAMPLE. The construction assumes whole-epidermal CPD depends
        only on fractional MED. If so, Shih's MED ratio (7.24) must equal
        Fajuyigbe's measured equal-dose whole-epidermis protection (8.0)."""
        from vitd_calibration import skin_type_susceptibility_check
        r = skin_type_susceptibility_check(verbose=False)
        assert r["premise_consistent"] is True
        assert r["premise_error"] < 0.25

    def test_consistent_with_shih_pooled_basal_to_whole_ratio(self):
        from vitd_calibration import skin_type_susceptibility_check
        r = skin_type_susceptibility_check(verbose=False)
        assert r["shih_consistent"] is True

    def test_stays_below_confounded_population_incidence_ratio(self):
        """A CONSTITUTIVE factor must be smaller than the observed population
        gap, which also contains behaviour, exposure and ascertainment."""
        from vitd_calibration import skin_type_susceptibility_check
        r = skin_type_susceptibility_check(verbose=False)
        assert r["below_population_ratio"] is True

    def test_beats_the_retired_prior_on_the_anchor_it_can_be_scored_on(self):
        from vitd_calibration import (skin_type_susceptibility_check,
                                      GANDINI2005_RR_SKINTYPE_I_VS_IV)
        r = skin_type_susceptibility_check(verbose=False)
        target = GANDINI2005_RR_SKINTYPE_I_VS_IV
        assert abs(r["derived_I_vs_IV"] - target) < abs(r["prior_I_vs_IV"] - target)

    def test_monotone_decreasing_and_bounded(self):
        from vitd_physiology import basal_damage_fraction
        vals = [basal_damage_fraction(f) for f in range(1, 7)]
        assert vals == sorted(vals, reverse=True)
        assert vals[0] == pytest.approx(1.0)   # unpigmented: uniform with depth
        assert all(0.0 < v <= 1.0 for v in vals)

    def test_conservative_where_the_studies_saw_nothing(self):
        """Shih found basal CPD UNDETECTABLE in FST IV-VI. The curve still
        charges them risk -- the safe direction. If this ever goes to zero the
        model would declare dark skin risk-free, which no source supports."""
        from vitd_physiology import basal_damage_fraction
        for f in (4, 5, 6):
            assert basal_damage_fraction(f) > 0.05

    def test_out_of_range_fst_does_not_explode(self):
        from vitd_physiology import basal_damage_fraction
        assert basal_damage_fraction(0) == pytest.approx(basal_damage_fraction(3))
        assert basal_damage_fraction(9) == pytest.approx(basal_damage_fraction(3))


class TestIntermittentIndexNormalisation:
    """Regression tests for the 2026-08-09 normalisation bug fix.

    The reference schedule used to be per-FST and to include the susceptibility
    factor, so both cancelled out of the ratio. The index was then exactly 1.0
    for every skin type on the reference schedule, and ranked dark skin as
    hugely riskier than light skin at equal fractional MED -- backwards
    relative to Shih 2018, this module's own MED source.
    """

    def test_susceptibility_survives_normalisation(self):
        """The bug: this was 1.0 for every FST."""
        vals = [cancer_risk([1.0] * 52, f)["intermittent_index"] for f in range(1, 7)]
        assert len({round(v, 9) for v in vals}) == 6
        assert vals == sorted(vals, reverse=True)

    def test_equal_fractional_med_ranks_darker_skin_lower(self):
        """Shih 2018: at equal fractional MED, whole-epidermal damage is equal
        across FST I-VI and basal-layer damage FALLS with skin darkness. The
        burst-driven index must follow, since it is already in fractional-MED
        units."""
        vals = []
        for f in range(1, 7):
            med = _med_sed_by_fst(f)
            vals.append(cancer_risk([0.5 * med] * 52, f)["intermittent_index"])
        assert vals == sorted(vals, reverse=True)
        # And the spread must be the susceptibility ratio, nothing else.
        from vitd_physiology import basal_damage_fraction
        assert vals[0] / vals[5] == pytest.approx(
            basal_damage_fraction(1) / basal_damage_fraction(6), rel=1e-9)

    def test_reference_schedule_scores_one_for_fst_I_on_both_axes(self):
        r = cancer_risk([1.0] * 52, 1)
        assert r["scc_index"] == pytest.approx(1.0)
        assert r["intermittent_index"] == pytest.approx(1.0)

    def test_within_fst_burst_penalty_is_unchanged_by_the_fix(self):
        """The fix must not touch within-person rankings -- the burst:spread
        ratio at fixed cumulative dose is C**(k-1) regardless of skin type."""
        ratios = []
        for f in (1, 3, 6):
            spread = cancer_risk([1.0] * 52, f)["intermittent_index"]
            burst = cancer_risk([26.0] * 2, f)["intermittent_index"]
            ratios.append(burst / spread)
        assert ratios[0] == pytest.approx(ratios[1])
        assert ratios[1] == pytest.approx(ratios[2])


class TestSCCDoseExponent:
    """SCC is super-linear in cumulative dose (biological amplification factor).

    Moan 1989 (Cancer Res 49:5207) gives Ab = 1.6-1.8 for SCC, erythemally
    weighted -- the same weighting this model's SED carries. The term used to be
    linear (Ab = 1), which understated the cost of accumulating dose.
    """

    def test_exponent_inside_the_published_range(self):
        from vitd_calibration import moan1989_scc_exponent_check
        r = moan1989_scc_exponent_check(verbose=False)
        assert r["in_moan_range"] is True
        assert r["in_literature_range"] is True

    def test_scc_is_super_linear_in_cumulative_dose(self):
        """Doubling cumulative dose must MORE than double the SCC index."""
        one = cancer_risk([1.0] * 52, 2)
        two = cancer_risk([2.0] * 52, 2)
        assert two["total_sed"] == pytest.approx(2 * one["total_sed"])
        assert two["scc_index"] > 2.0 * one["scc_index"]

    def test_still_depends_only_on_the_cumulative_total(self):
        """Super-linear, but in the TOTAL: how the dose is split must not matter
        on this axis. That separation is what makes the two indices distinct."""
        a = cancer_risk([1.0] * 52, 2)
        b = cancer_risk([26.0] * 2, 2)
        assert a["scc_index"] == pytest.approx(b["scc_index"])

    def test_reference_schedule_normalisation_preserved(self):
        """1 SED/week for a year is the unit schedule: index = susceptibility."""
        from vitd_physiology import basal_damage_fraction
        r = cancer_risk([1.0] * 52, 3)
        assert r["scc_index"] == pytest.approx(basal_damage_fraction(3))


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


class TestPersonalGeometryFactor:
    """Schmalwieser & Siani 2018; Parisi 2000/2003: geometry factor vs solar elevation."""

    def test_default_when_elevation_unknown(self):
        assert personal_geometry_factor(None) == 0.35

    def test_solar_noon_lower_ratio(self):
        """At 60 deg solar elevation, vertical surfaces receive ~0.30 ratio."""
        assert personal_geometry_factor(60.0) == pytest.approx(0.30, abs=1e-2)

    def test_low_elevation_higher_ratio(self):
        """At 0 deg solar elevation, ratio rises to 0.45."""
        assert personal_geometry_factor(0.0) == pytest.approx(0.45, abs=1e-2)

