"""Tests for the mechanistic sub-models added in the July 2026 literature pass.

Each test states the literature fact it encodes. Wrong code fails against a
paper, not a magic number.
"""

import numpy as np
import pytest

from vitd_model import (ModelParams, simulate, build_pulse_schedule,
                        build_weekly_schedule, synthesis_per_session,
                        BODY_WEIGHT_DILUTION_EXPONENT)
from vitd_physiology import (body_weight_dilution_factor,
                             photodegradation_factor,
                             personal_geometry_factor,
                             vernez_exposure_ratio,
                             hydroxylation_efficiency,
                             cancer_risk,
                             _med_sed_by_fst,
                             INTERMITTENT_BURN_EXPONENT,
                             BODY_WEIGHT_REF_KG,
                             vis_table_s1,
                             activity_erta,
                             activity_erta_interval,
                             ACTIVITY_ERTA_TABLES,
                             SCHMALWIESER_2025_JOGGING_ERTA,
                             SCHMALWIESER_2025_JOGGING_ERTA_MAX,
                             SCHMALWIESER_2025_ERTA_ELEVATIONS_DEG,
                             SCHMALWIESER_2025_ERTA_REL_UNCERTAINTY,
                             SCHMALWIESER_2025_ERTA_VALID_ELEVATION_DEG,
                             VERNEZ_TABLE_S1_VIS,
                             VERNEZ_TABLE_S1_POSTURES,
                             VERNEZ_REPORTED_SITE_ER,
                             VERNEZ_REF_SZA_DEG,
                             weihs2013_prose_ratio,
                             WEIHS_2013_PROSE_SITE_RATIOS,
                             WEIHS_2013_REPORTED_ER_RANGE,
                             WEIHS_2013_MOUNTING_ACCURACY,
                             WEIHS_2013_SOLAR_ELEVATION_BAND_DEG)


class TestBodyWeightDilution:
    """Drincic 2012: hyperbolic in total body WEIGHT, not linear, not BMI."""

    def test_neutral_when_unknown_or_reference(self):
        assert body_weight_dilution_factor(None) == 1.0
        assert body_weight_dilution_factor(BODY_WEIGHT_REF_KG) == pytest.approx(1.0)

    def test_heavier_dilutes_lighter_concentrates(self):
        assert body_weight_dilution_factor(100.0) < 1.0
        assert body_weight_dilution_factor(55.0) > 1.0

    def test_hyperbolic_not_linear(self):
        """1/weight: doubling weight must halve the factor exactly.

        [PIN FIX 2026-09-22, mechanical, A5 wire-in design-trap fix] This
        test's own subject is the LITERAL Drincic/Ekwaru hyperbolic form
        (exponent exactly 1.0). That form is no longer the function's
        default -- the calibrated BODY_WEIGHT_DILUTION_EXPONENT now is, per
        vitd_physiology.py's own comment ("Callers wanting the literal
        Drincic/Ekwaru hyperbolic form must now ask for it explicitly with
        exponent=1.0") -- so this test must ask for it explicitly rather than
        rely on the default. Nothing about what this test checks changed.
        """
        assert (body_weight_dilution_factor(140.0, exponent=1.0)
                == pytest.approx(0.5 * body_weight_dilution_factor(70.0, exponent=1.0)))

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

        [PIN FIX 2026-09-22, mechanical] Same reason as
        test_hyperbolic_not_linear above: this checks the LITERAL
        exponent=1.0 form against the Ekwaru literature ratio, which is no
        longer the function's default, so it must be requested explicitly.
        """
        ratio = (body_weight_dilution_factor(70.0, exponent=1.0)
                 / body_weight_dilution_factor(140.0, exponent=1.0))
        assert 1.8 <= ratio <= 2.2


class TestBodyWeightDilutionExponentCalibration:
    """A5, 2026-09-22 (user-approved): the exponent is now overridable, and
    vitd_model.py's own synthesis path overrides it with an NHANES-measured
    value -- see calibration_approval.yaml BODY_WEIGHT_DILUTION_EXPONENT and
    the BODY_WEIGHT_DILUTION_EXPONENT entry in vitd_model.py's docstring.

    These tests validate the NEW `exponent=` parameter itself. They do NOT
    touch TestBodyWeightDilution above, but that class's two tests of the
    literal hyperbolic form (test_hyperbolic_not_linear,
    test_heavier_body_needs_roughly_double_dose) were themselves updated
    2026-09-22 to pass `exponent=1.0` explicitly -- see the design-trap-fix
    comment below.

    [DESIGN CHANGED 2026-09-22, decision A5 wire-in] This class's own
    docstring, and the test immediately below, previously asserted the
    OPPOSITE of what is now true: that the function's default stayed at the
    literal exponent 1.0. That was itself the design trap CLAUDE.md's A5
    brief named ("fix the function's default exponent still being 1.0") --
    vitd_model.py's one call site passed the calibrated exponent explicitly
    while the function's own default silently stayed at the superseded
    value, so any OTHER caller got the wrong answer by omission. The
    function's default is now BODY_WEIGHT_DILUTION_EXPONENT itself; a caller
    wanting the literal Drincic/Ekwaru form must ask for `exponent=1.0`
    explicitly (see vitd_physiology.body_weight_dilution_factor's docstring).
    """

    def test_default_is_the_calibrated_exponent_not_one(self):
        """The function's own default must now be the calibrated exponent.

        REPLACES test_default_exponent_unchanged_at_one (2026-09-22, decision
        A5 wire-in), which asserted the design this repository deliberately
        moved away from -- see the class docstring above. The old assertion
        (default == exponent=1.0) is now FALSE by construction and would be
        the wrong thing to guard; this test guards the new invariant instead:
        calling with no `exponent=` must reproduce calling with
        `exponent=BODY_WEIGHT_DILUTION_EXPONENT` explicitly, and must NOT
        reproduce the literal hyperbolic form.
        """
        assert (body_weight_dilution_factor(140.0)
                == pytest.approx(body_weight_dilution_factor(
                    140.0, exponent=BODY_WEIGHT_DILUTION_EXPONENT)))
        assert (body_weight_dilution_factor(140.0)
                != pytest.approx(body_weight_dilution_factor(140.0, exponent=1.0)))

    def test_calibrated_constant_matches_nhanes_measurement(self):
        """BODY_WEIGHT_DILUTION_EXPONENT is the full-precision NHANES
        unsupplemented-adults estimate (magnitude of gamma_ln_weight_kg),
        tables/nhanes_weight_dilution_check.csv, not a rounded figure."""
        assert BODY_WEIGHT_DILUTION_EXPONENT == pytest.approx(0.24226563069970553,
                                                               abs=1e-12)

    def test_calibrated_exponent_dilutes_less_steeply_than_default(self):
        """0 < BODY_WEIGHT_DILUTION_EXPONENT < 1: the NHANES-measured
        exponent is between flat (0, no weight association) and the old
        exact-hyperbolic default (1) -- a real, weaker-than-hyperbolic
        weight association, not a null result and not the prior value."""
        assert 0.0 < BODY_WEIGHT_DILUTION_EXPONENT < 1.0

    def test_calibrated_exponent_used_by_the_model(self):
        """simulate() with body_weight_kg set must reproduce
        body_weight_dilution_factor's own value at the calibrated exponent,
        not at the function's default of 1.0."""
        assert (body_weight_dilution_factor(140.0, exponent=BODY_WEIGHT_DILUTION_EXPONENT)
                != pytest.approx(body_weight_dilution_factor(140.0, exponent=1.0)))


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
        the feedback artifact that
        vitd_optimization.gross_synth_per_SED_diagnostic recomputes on
        every pipeline run."""
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
        pl.feedback_mode = "linear"          # LEGACY, must be asked for now
        pb.feedback_mode = "biphasic"
        m = pl.M_baseline(3)
        assert synthesis_per_session(1.0, m, m, 200.0, 0.0, pl) == 0.0
        assert synthesis_per_session(1.0, m, m, 200.0, 0.0, pb) > 0.0

    def test_biphasic_is_now_the_default(self):
        """UPDATED 2026-08-18 (approved item A2). This assertion used to read
        `feedback_mode == "linear"`. The hard-ceiling form was withdrawn as the
        default: D_max = 150.0 has no located source, and 25(OH)D regulation
        acts on hydroxylation and CYP24A1-driven catabolism rather than on a
        switch that stops cutaneous synthesis outright. 'linear' and 'power'
        are retained so earlier results remain reproducible on request."""
        assert ModelParams().feedback_mode == "biphasic"

    def test_unknown_feedback_mode_raises_instead_of_silently_going_linear(self):
        """The dispatch used to fall through to the linear form on any
        unrecognised string. Harmless while linear WAS the default; a trap now,
        because a typo would silently reinstate the withdrawn hard ceiling."""
        p = ModelParams()
        p.feedback_mode = "bipahsic"
        with pytest.raises(ValueError):
            synthesis_per_session(1.0, 0.21, 0.21, 25.0, 0.0, p)


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
    """All new mechanisms default off or neutral."""

    def test_defaults_neutral(self):
        p = ModelParams()
        assert p.body_weight_kg is None
        assert p.use_photodegradation is False
        assert p.use_transit_delay is False
        # feedback_mode is NOT in this list any more: it is a scientific choice
        # with a source, not an opt-in mechanism. See
        # TestHydroxylationFeedback::test_biphasic_is_now_the_default.

    def test_default_run_is_stable(self):
        """Guards the headline scenario against accidental drift.

        Pin REBASED 2026-08-17: 63.75 -> 49.85 nmol/L. This is a pinned
        regression value, not an independent measurement, and it moved because
        `ModelParams.K_const` was recalibrated 0.33 -> 0.98 under the approval
        in calibration_approval.yaml. The scenario is FST V, so constitutive
        synthesis falls by exp(-(0.98 - 0.33) * M_baseline(V)) = 0.60 and the
        84-day endpoint falls with it. Nothing else in the scenario changed.

        Pin REBASED AGAIN 2026-08-18: 49.85 -> 53.64 nmol/L, from the two
        approved changes landing together. A: 2.353 -> 3.114 (item A4, Young
        2021 per-SED anchor) raises synthesis by x1.32; the feedback form moved
        from max(0, 1 - D/150) to the biphasic hydroxylation efficiency (item
        A2), which is the STEEPER of the two across the 40-55 nmol/L band this
        scenario occupies and gives most of that back. Net +7.6%. Neither the
        scenario nor the tolerance changed.

        Pin REBASED AGAIN 2026-09-06: 53.64 -> 48.32 nmol/L. K_const
        0.98 -> 1.15 and M_baseline(V) 0.78 -> 0.7539 both move together
        (calibration_approval.yaml, K_const and M_baseline_by_FST entries);
        this scenario uses default exposed_bsa (bsa_factor = 1.0), so
        BSA_EXP's move does not touch it. Net effect on FST V constitutive
        attenuation: exp(-1.15*0.7539) / exp(-0.98*0.78) = 0.9025, a further
        9.75% fall on top of the 2026-08-18 pin (the 84-day endpoint falls
        somewhat more than that, since feedback and decay compound over the
        course rather than scaling the single multiplicative factor exactly).
        Neither the scenario nor the tolerance changed.

        Pin REBASED ONCE MORE 2026-09-06 (same day): 48.32 -> 57.77 nmol/L,
        and this time the pin goes UP. The 48.32 above was measured with
        A still at its stale 3.114 and K_SC briefly at 1.509. Two later
        same-day decisions moved it: A re-anchored 3.114 -> 4.125
        (calibration_approval.yaml, A_reanchor_2026_09_06 -- the Young 2021
        anchor recomputed after BSA_EXP and K_const/M_baseline moved its own
        denominator), and K_SC backed off 1.509 -> 1.0 (the 1.509 optical
        value is an upper bound that assumes corneocyte thickening, while
        SC_max is calibrated on de Boer's edema-attributed change). Both raise
        the endpoint. Net across the whole 2026-09-06 pass: 53.64 -> 57.77,
        i.e. the melanin term's fall is more than offset by the larger A.
        Neither the scenario nor the tolerance changed.

        Pin REBASED 2026-09-23: 57.77 -> 54.44427408025138 nmol/L, for the
        2026-09-22 parameter pass. FOUR changes land on this FST V scenario
        at once and they do not all point the same way:
          K_const   1.17 -> 2.096070737357932   (down)
          M_baseline(V) 0.7539 -> 0.5295867768595042  (up, the A3 ITA -> M
                        re-scale; the pair is a reparameterisation, so the
                        NET melanin term barely moves)
          A         4.139 -> 4.359560972608079  (up)
          body weight  inert -> BODY_WEIGHT_BY_FST[5] = 83.337561944132 kg
                        (down: dilution factor
                        (70/83.337561944132)**0.24226563069970553)
        The body-weight term is the one that was not there before at all:
        before 2026-09-22 nothing set a weight and the factor was exactly
        1.0 everywhere, so a default FST V run is now stated at a US
        population body rather than at the 70 kg reference. Neither the
        scenario nor the tolerance changed.
        """
        r = simulate(fst=5, schedule=build_pulse_schedule(84, 1.0, 1),
                     duration_days=84, params=ModelParams(), D0=25.0)
        assert r["D"][-1] == pytest.approx(54.44427408025138, abs=0.5)


class TestTwoProcessSC:
    """de Boer 2025 Supp Table S1: SC falls ~26% BELOW baseline at week 5
    (5.8 -> 4.3 -> 7.5 um) before overshooting at week 9.

    The old single multiplicative form (`SC *= exp(-k_damage*sed)`) could only
    shrink SC toward zero, so no k_damage could ever reproduce that. These tests
    pin the corrected structure -- and pin that k_damage stays 0.0, because
    fixing the structure did not make the parameter identifiable.
    """

    def test_k_damage_still_zero_by_default(self):
        """Structure was fixed; the parameter is still unidentifiable from
        three group medians. See scripts/fit_sc_deboer2025.py."""
        assert ModelParams().k_damage == 0.0

    def test_default_SC_trajectory_unchanged(self):
        """Backward compatibility: at k_damage=0 the two-process form must
        reproduce the old single-process numbers exactly.

        Expected value updated 2026-08-25 with the SC_max_by_FST I-III
        recalibration 0.23 -> 0.24 (Miyamura 2011 corroboration of de Boer
        2025; see calibration_approval.yaml). This is a ceiling rescale and
        nothing else: 0.136 * 0.24/0.23 = 0.141913 against an observed
        0.141905, so the trajectory shape is untouched and this test still
        guards what it was written to guard.
        """
        p = ModelParams()
        r = simulate(fst=3, schedule=build_weekly_schedule(63, 0.8, [0, 2, 4]),
                     duration_days=63, params=p, D0=25.0)
        assert r["SC"][-1] == pytest.approx(0.141905, abs=1e-3)

    def test_SC_can_go_below_baseline(self):
        """The capability the old parameterisation lacked entirely."""
        p = ModelParams()
        p.k_damage = 0.10
        r = simulate(fst=3, schedule=build_weekly_schedule(63, 0.8, [0, 2, 4]),
                     duration_days=63, params=p, D0=25.0)
        assert r["SC"].min() < 0.0

    def test_old_form_could_not_go_below_zero(self):
        """Guards the diagnosis: a multiplicative damage term preserves sign,
        so it shrinks a non-negative SC toward zero but never below it."""
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
    """Vernez 2015 Eq. 1 (doi 10.1038/jes.2014.6), full text in the repo.

    The previous `0.15 + 0.30*cos(theta)` form was withdrawn 2026-08-16: its
    cited source contradicts it (gap 26). These tests pin the replacement to
    numbers printed in the Vernez paper.
    """

    def test_default_when_vis_unknown(self):
        assert personal_geometry_factor(None) == 0.35

    def test_elevation_alone_warns_and_does_not_change_result(self):
        """The withdrawn form must not silently come back."""
        with pytest.warns(UserWarning, match="withdrawn"):
            assert personal_geometry_factor(60.0) == 0.35

    def test_activity_route_returns_the_measured_erta(self):
        """activity= must reach the Schmalwieser 2025 table, not the scalar."""
        assert personal_geometry_factor(40.0, activity="jogging",
                                        body_site="shoulders") == pytest.approx(0.82)
        assert personal_geometry_factor(60.0, activity="jogging",
                                        body_site="thighs") == pytest.approx(0.34)

    def test_activity_defaults_to_the_across_site_mean(self):
        assert (personal_geometry_factor(40.0, activity="jogging")
                == pytest.approx(activity_erta("jogging",
                                               "mean_measured_sites", 40.0)))

    def test_vis_percent_and_activity_together_raise(self):
        """Two competing estimates of one ratio: the caller must choose. Silently
        preferring either would hide which source produced the answer."""
        with pytest.raises(ValueError, match="not both"):
            personal_geometry_factor(40.0, vis_percent=58.0, activity="jogging")

    def test_body_site_without_activity_raises(self):
        """The Vernez route selects a site through vis_table_s1(), not here."""
        with pytest.raises(ValueError, match="body_site"):
            personal_geometry_factor(40.0, vis_percent=58.0, body_site="forehead")

    def test_flat_fallback_is_still_0_35_after_the_2025_papers(self):
        """[REGRESSION, 2026-08-20] Schmalwieser 2025 and Weihs 2013 were both
        integrated WITHOUT moving the Cockell scalar, and that decision is the
        substance of the calibration approval -- so it is pinned.

        Jogging ERTAs sit above 0.35; Weihs reports ERs from 0.05 to 1.0 across
        six sites and four postures, straddling it in both directions. The
        spread across activities and sites is wider than the distance from 0.35
        to either, so no single measured number replaces the fallback.

        [CORRECTED 2026-08-20] this docstring previously cited a Weihs "walking
        site mean (~0.26)" taken from a Figure 6 digitization since withdrawn as
        unreadable. Conclusion and value unchanged; see TestWeihs2013ProseRatios.
        """
        from vitd_objectives import PERSONAL_DOSE_GEOMETRY_FACTOR
        assert PERSONAL_DOSE_GEOMETRY_FACTOR == 0.35
        assert personal_geometry_factor() == 0.35
        jogging_mean = activity_erta("jogging", "mean_measured_sites", 40.0)
        assert jogging_mean > 0.35, (
            "if this ever fails the 'jogging sits above the scalar' half of the "
            "argument has changed and the approval needs revisiting")

    def test_intercept_reproduces_at_centring_point(self):
        """Vernez centres on Vis=58%; there ER == the printed intercept 56.991%."""
        from vitd_physiology import VERNEZ_REF_SZA_DEG
        assert vernez_exposure_ratio(58.0, VERNEZ_REF_SZA_DEG) == pytest.approx(
            0.56991, abs=1e-4)

    def test_low_visibility_site(self):
        """Vis=10% collapses to ~11.5% ER -- a heavily self-shaded site."""
        from vitd_physiology import VERNEZ_REF_SZA_DEG
        assert vernez_exposure_ratio(10.0, VERNEZ_REF_SZA_DEG) == pytest.approx(
            0.11534, abs=1e-4)

    def test_vis_dominates_sza(self):
        """The paper: "The Vis parameter is largely predominant".

        Sky visibility must move ER far more than solar geometry does.
        """
        sza_span = (vernez_exposure_ratio(58.0, 89.0)
                    / vernez_exposure_ratio(58.0, 0.0))
        vis_span = (vernez_exposure_ratio(95.0, 45.0)
                    / vernez_exposure_ratio(5.0, 45.0))
        assert sza_span < 1.3, f"SZA term should be a small trim, got {sza_span}"
        assert vis_span > 8.0, f"Vis term should dominate, got {vis_span}"

    def test_er_increases_with_visibility(self):
        vals = [vernez_exposure_ratio(v, 45.0) for v in (5, 20, 40, 58, 80, 95)]
        assert vals == sorted(vals)

    def test_er_bounded(self):
        for v in (0.5, 1, 10, 50, 99, 100):
            assert 0.0 <= vernez_exposure_ratio(v, 30.0) <= 1.0

    def test_rejects_out_of_range_visibility(self):
        for bad in (0.0, -5.0, 101.0):
            with pytest.raises(ValueError):
                vernez_exposure_ratio(bad, 45.0)

    def test_unshaded_ceiling_exceeds_behavioural_dosimetry(self):
        """Vernez ER assumes no shade; it must sit ABOVE the Cockell 0.35 scalar
        for an averagely-exposed site. This is the caveat, encoded: the two
        numbers are not interchangeable."""
        assert vernez_exposure_ratio(58.0, 45.0) > 0.35


class TestSchmalwieser2025JoggingERTA:
    """Schmalwieser et al. 2025, Photochem Photobiol, doi 10.1111/php.14114,
    "Erythemal UV radiation exposure during jogging". Full text in
    Research_Papers/Schmalwieser_2025_Photo.pdf.

    The constants are DIGITIZED FROM FIGURE 5 (the paper prints no ERTA table),
    so every test below cross-checks them against a statement the paper makes in
    PROSE. A failure here should mean someone re-digitized the figure, never
    that someone re-fitted anything.

    n = 1 subject, one posture, Vienna, solar elevation 20-60 deg, stated
    measurement uncertainty +/-15%.
    """

    def test_seven_measured_sites_and_five_elevations(self):
        """The paper measured 12 dosimeters at 7 sites; left/right were pooled."""
        assert len(SCHMALWIESER_2025_JOGGING_ERTA) == 7
        assert SCHMALWIESER_2025_ERTA_ELEVATIONS_DEG == (20.0, 30.0, 40.0, 50.0, 60.0)
        for site, vals in SCHMALWIESER_2025_JOGGING_ERTA.items():
            assert len(vals) == 5, site

    def test_the_sourced_table_is_not_polluted_by_the_derived_mean(self):
        """'mean_measured_sites' is COMPUTED and lives only in the lookup
        registry. The transcribed constant must keep exactly the paper's sites."""
        assert "mean_measured_sites" not in SCHMALWIESER_2025_JOGGING_ERTA
        assert "mean_measured_sites" in ACTIVITY_ERTA_TABLES["jogging"]
        assert len(ACTIVITY_ERTA_TABLES["jogging"]) == 8

    def test_shoulders_are_the_most_exposed_site_at_every_elevation(self):
        """Abstract: "the top of the shoulder is the most exposed body site".
        Discussion: "at all solar elevations and for all running directions"."""
        for i, elev in enumerate(SCHMALWIESER_2025_ERTA_ELEVATIONS_DEG):
            shoulders = SCHMALWIESER_2025_JOGGING_ERTA["shoulders"][i]
            for site, vals in SCHMALWIESER_2025_JOGGING_ERTA.items():
                if site != "shoulders":
                    assert vals[i] < shoulders, (site, elev)

    def test_shoulders_average_80_percent_of_ambient_and_are_flat(self):
        """Abstract: "receiving 80% of ambient UVR on average". Results: "highest
        on average at a level of 0.8 with NO DEPENDENCE on solar elevation"."""
        vals = SCHMALWIESER_2025_JOGGING_ERTA["shoulders"]
        assert sum(vals) / len(vals) == pytest.approx(0.80, abs=0.02)
        assert max(vals) - min(vals) <= 0.03, "must stay flat across elevation"

    def test_forearm_and_forehead_levels(self):
        """Results: forearms and forehead show "no clear tendency ... at a level
        of 0.5 and 0.3, respectively"."""
        fore = SCHMALWIESER_2025_JOGGING_ERTA["forearms"]
        head = SCHMALWIESER_2025_JOGGING_ERTA["forehead"]
        assert sum(fore) / len(fore) == pytest.approx(0.5, abs=0.06)
        assert sum(head) / len(head) == pytest.approx(0.3, abs=0.05)
        assert min(head) == min(SCHMALWIESER_2025_JOGGING_ERTA[s][i]
                                for s in SCHMALWIESER_2025_JOGGING_ERTA
                                for i in range(5)), "forehead holds the minimum"

    def test_four_sites_fall_with_rising_solar_elevation(self):
        """Results: upper arms, calves, thighs and nape "decrease with increasing
        solar elevation, starting with a value of 0.5 at 20 deg down to around
        0.4 and 0.3 at 60 deg".

        This is the result the flat 0.35 scalar could not express: for a tilted
        anatomical surface the ratio to a HORIZONTAL reference falls as the sun
        climbs. This is geometry, not spectrum -- see the double-counting note in
        vitd_physiology.
        """
        for site in ("upper_arms", "calves", "nape", "thighs"):
            vals = SCHMALWIESER_2025_JOGGING_ERTA[site]
            assert vals[0] == pytest.approx(0.5, abs=0.06), site
            assert 0.30 <= vals[-1] <= 0.45, site
            assert vals[-1] < vals[0], site

    def test_all_random_path_means_stay_at_or_below_ambient(self):
        """Mean over a random pathway must not exceed 1.0; only the DIRECTIONAL
        maximum does, and that is a separate constant."""
        for site, vals in SCHMALWIESER_2025_JOGGING_ERTA.items():
            for v in vals:
                assert 0.0 < v <= 1.0, (site, v)
        assert SCHMALWIESER_2025_JOGGING_ERTA_MAX["shoulders"] > 1.0

    def test_directional_shoulder_maximum_exceeds_ambient(self):
        """Discussion: "When jogging toward the sun, the UVR exposure may reach
        up to 117% ... caused by the torso bending slightly forward"."""
        assert SCHMALWIESER_2025_JOGGING_ERTA_MAX["shoulders"] == pytest.approx(1.17)

    def test_interpolates_between_measured_elevations(self):
        mid = activity_erta("jogging", "nape", 45.0)
        assert mid == pytest.approx(0.5 * (0.45 + 0.38))

    def test_clamps_outside_the_measured_band_with_a_warning(self):
        """The authors: results "may not be necessarily applicable at higher
        solar elevations as found at subtropical and tropical latitudes". Silent
        extrapolation to a tropical noon is exactly what must not happen."""
        lo, hi = SCHMALWIESER_2025_ERTA_VALID_ELEVATION_DEG
        with pytest.warns(UserWarning, match="outside the measured band"):
            assert (activity_erta("jogging", "calves", 85.0)
                    == activity_erta("jogging", "calves", hi))
        with pytest.warns(UserWarning, match="outside the measured band"):
            assert (activity_erta("jogging", "calves", 5.0)
                    == activity_erta("jogging", "calves", lo))

    def test_interval_uses_the_papers_own_stated_uncertainty(self):
        """+/-15%, derived by the authors from left/right disagreement on runs
        straight toward and away from the sun (0.5-30%, worst at the shoulders)."""
        assert SCHMALWIESER_2025_ERTA_REL_UNCERTAINTY == 0.15
        lo, hi = activity_erta_interval("jogging", "shoulders", 40.0)
        mid = activity_erta("jogging", "shoulders", 40.0)
        assert lo == pytest.approx(mid * 0.85)
        assert hi == pytest.approx(mid * 1.15)

    def test_across_site_mean_is_computed_not_typed(self):
        for i, elev in enumerate(SCHMALWIESER_2025_ERTA_ELEVATIONS_DEG):
            expected = (sum(v[i] for v in SCHMALWIESER_2025_JOGGING_ERTA.values())
                        / len(SCHMALWIESER_2025_JOGGING_ERTA))
            assert activity_erta("jogging", "mean_measured_sites", elev) == pytest.approx(expected)

    def test_across_site_mean_is_not_bsa_weighted(self):
        """Encodes the caveat rather than the number: the mean counts the
        shoulders (~2% of BSA, most exposed site) equally with everything else
        and represents the trunk not at all -- the sternum was dropped as a
        mounting position. It must therefore sit well above the site-count-free
        truth for a mostly-limb exposure."""
        mean40 = activity_erta("jogging", "mean_measured_sites", 40.0)
        limbs = [activity_erta("jogging", s, 40.0)
                 for s in ("forearms", "upper_arms", "calves", "thighs")]
        assert mean40 > sum(limbs) / len(limbs)

    def test_unknown_activity_site_or_missing_elevation_raise(self):
        with pytest.raises(KeyError):
            activity_erta("cycling", "forehead", 40.0)
        with pytest.raises(KeyError):
            activity_erta("jogging", "elbow", 40.0)
        with pytest.raises(ValueError):
            activity_erta("jogging", "nape", None)

    def test_body_site_names_normalise_like_the_vernez_table(self):
        assert (activity_erta("jogging", "Upper arms", 40.0)
                == activity_erta("jogging", "upper_arms", 40.0))

    def test_geometry_and_spectral_terms_stay_separable(self):
        """The double-counting guard, encoded. ERTA is a GEOMETRY factor that
        multiplies the erythemal dose; E_source (VDE:EE) is a SPECTRAL factor
        applied downstream. sed_from_minutes must return an erythemal SED that
        is exactly linear in the ERTA and carries no spectral weighting at all,
        so the two never collapse into one another.
        """
        from vitd_objectives import sed_from_minutes
        base = sed_from_minutes(30.0, 6.0, geometry_factor=1.0)
        for site in ("shoulders", "thighs"):
            erta = activity_erta("jogging", site, 40.0)
            got = sed_from_minutes(30.0, 6.0, solar_elevation_deg=40.0,
                                   activity="jogging", body_site=site)
            assert got == pytest.approx(base * erta)


class TestWeihs2013ProseRatios:
    """Weihs, Schmalwieser, Reinisch, Meraner, Walisch & Harald 2013, Photochem
    Photobiol 89:1004-1007, "Measurements of Personal UV Exposure on Different
    Parts of the Body During Various Activities". Full text in Research_Papers/.

    UNLIKE the Schmalwieser class above, NOTHING here is digitized. Every number
    is transcribed from a sentence in the paper's Results or Discussion, because
    its Figure 6 cannot be read to per-site precision in the copy held (see the
    PROVENANCE block in vitd_physiology.py; an earlier Figure 6 digitization was
    withdrawn). A failure here should mean someone retyped the prose wrong.

    These are RATIOS BETWEEN SITES, never exposure ratios to ambient. The paper
    states no absolute ER for any single site and no elevation dependence, which
    is exactly why there is no Weihs entry in ACTIVITY_ERTA_TABLES -- pinned by
    test_weihs_must_not_become_an_erta_table below.
    """

    def test_walking_shoulders_are_70_percent_above_chest_and_forehead(self):
        """Results: "During walking, the study subject receives the highest
        exposures on the shoulders which is 70% higher than the exposure of the
        chest and the forehead"."""
        assert weihs2013_prose_ratio("walking", "shoulders", "chest") == 1.70
        assert weihs2013_prose_ratio("walking", "shoulders", "forehead") == 1.70

    def test_sitting_thighs_and_shoulders_are_60_percent_above(self):
        """Results: "During sitting the highest UV exposures are received by the
        thighs and the shoulders. Exposures are 60% higher than on the forehead
        and on the chest"."""
        for site in ("thighs", "shoulders"):
            for ref in ("forehead", "chest"):
                assert weihs2013_prose_ratio("sitting", site, ref) == 1.60

    def test_lying_and_sitting_up_are_led_by_the_lower_legs(self):
        """Results: "During sitting up and lying the exposures are highest on
        the lower legs (respectively 115% and 55% higher than on forehead)".

        Corroborated by the Discussion, which is why this survived while the
        figure read did not: "The highest ERs occurred on the lower legs while
        lying and sitting up. In these positions the lower legs are oriented
        horizontally like the device for ambient UV measurements."
        """
        assert weihs2013_prose_ratio("sitting_up", "lower_legs", "forehead") == 2.15
        assert weihs2013_prose_ratio("lying", "lower_legs", "forehead") == 1.55

    def test_the_posture_reorders_the_body(self):
        """The finding that makes a single geometry scalar indefensible: which
        site leads DEPENDS ON THE POSTURE. Shoulders lead upright locomotion,
        the lower legs lead once the body is horizontal."""
        assert (weihs2013_prose_ratio("sitting_up", "lower_legs", "forehead")
                > weihs2013_prose_ratio("walking", "shoulders", "forehead"))
        for activity, table in WEIHS_2013_PROSE_SITE_RATIOS.items():
            for (site, ref), ratio in table.items():
                assert ratio > 1.0, (activity, site, ref)

    def test_reported_range_accuracy_and_elevation_band(self):
        """Discussion: ERs "between 0.05 and 1"; inclinations adjustable to
        "roughly +/-10%". Methods: "solar elevations between 46 and 64 degrees".

        The 0.05-1.0 spread is what carries the non-promotion argument for
        PERSONAL_DOSE_GEOMETRY_FACTOR: it straddles 0.35 in both directions.
        """
        lo, hi = WEIHS_2013_REPORTED_ER_RANGE
        assert (lo, hi) == (0.05, 1.0)
        assert lo < 0.35 < hi, "the straddle IS the argument; see vitd_objectives"
        assert WEIHS_2013_MOUNTING_ACCURACY == 0.10
        assert WEIHS_2013_SOLAR_ELEVATION_BAND_DEG == (46.0, 64.0)

    def test_weihs_must_not_become_an_erta_table(self):
        """[REGRESSION, 2026-08-20] The prose gives ratios with NO absolute
        anchor and NO elevation resolution, so it cannot populate an ERTA table.
        Building one would require inventing an absolute value for some site --
        a calibration, gated by calibration_approval.yaml. If this fails, either
        that approval was granted (check the WEIHS-2013-VOR open item) or
        someone anchored the ratios to a guess.
        """
        for activity in WEIHS_2013_PROSE_SITE_RATIOS:
            assert activity not in ACTIVITY_ERTA_TABLES, activity
        assert set(ACTIVITY_ERTA_TABLES) == {"jogging"}
        with pytest.raises(KeyError):
            activity_erta("walking", "shoulders", 50.0)

    def test_accessor_normalises_site_names_and_refuses_the_rest(self):
        """Raises rather than inverting or chaining: the paper prints only the
        pairs it prints."""
        assert (weihs2013_prose_ratio("walking", "Shoulders", "Forehead")
                == weihs2013_prose_ratio("walking", "shoulders", "forehead"))
        with pytest.raises(KeyError, match="Do not invert or chain"):
            weihs2013_prose_ratio("walking", "forehead", "shoulders")
        with pytest.raises(KeyError, match="Do not invert or chain"):
            weihs2013_prose_ratio("walking", "thighs", "forehead")
        with pytest.raises(KeyError, match="No Weihs 2013 prose ratios"):
            weihs2013_prose_ratio("cycling", "shoulders", "forehead")

    def test_cross_study_shoulders_forehead_contrast(self):
        """Recomputes the cross-study comparison written above
        weihs2013_prose_ratio in vitd_physiology.py, so that comment cannot go
        stale if the Schmalwieser table is ever re-digitized.

        Overlap band: Weihs 46-64 deg, Schmalwieser 20-60 deg -> 46-60 deg.
        """
        ratio = lambda e: (activity_erta("jogging", "shoulders", e)
                           / activity_erta("jogging", "forehead", e))
        assert ratio(46.0) == pytest.approx(2.61, abs=0.01)
        assert ratio(50.0) == pytest.approx(2.76, abs=0.01)
        assert ratio(60.0) == pytest.approx(2.22, abs=0.01)
        band = [ratio(e) for e in np.linspace(46.0, 60.0, 15)]
        assert sum(band) / len(band) == pytest.approx(2.52, abs=0.01)

    def test_two_studies_agree_on_ordering_not_on_magnitude(self):
        """The verdict, encoded. Both papers put the shoulders far above the
        forehead during upright locomotion -- independent corroboration of
        Schmalwieser's headline finding, which otherwise rests on n = 1. They do
        NOT agree on how far: the jogging contrast is ~1.5x the walking one, a
        gap well outside either paper's stated measurement uncertainty
        (+/-15% mounting for Schmalwieser, +/-10% for Weihs). Posture, not
        measurement error. Evidence AGAINST borrowing one activity's table for
        another.
        """
        walking = weihs2013_prose_ratio("walking", "shoulders", "forehead")
        jogging = [activity_erta("jogging", "shoulders", e)
                   / activity_erta("jogging", "forehead", e)
                   for e in np.linspace(46.0, 60.0, 15)]
        jogging_mean = sum(jogging) / len(jogging)
        assert walking > 1.0 and min(jogging) > 1.0, "ordering must agree"
        combined_uncertainty = (1.0 + SCHMALWIESER_2025_ERTA_REL_UNCERTAINTY
                                + WEIHS_2013_MOUNTING_ACCURACY)
        assert jogging_mean / walking > combined_uncertainty, (
            "if these ever reconcile inside the stated uncertainties, the "
            "'activity and posture dominate' argument for keeping the flat "
            "0.35 fallback needs re-examining")
        assert jogging_mean / walking == pytest.approx(1.48, abs=0.02)


class TestVernezTableS1:
    """Vernez 2015 Supplementary Table S1 (doi 10.1038/jes.2014.6), extracted
    2026-08-17 to data/vernez2015_table_s1_vis_by_site_posture.csv.

    These pin the REAL, sourced Vis-by-site-and-posture values -- the paper's
    actual inputs -- replacing the back-derived VIS_PRESETS_PLACEHOLDER dict
    that lived in vitd_physiology.py from 2026-08-16 to 2026-08-17. Numbers
    below are transcribed straight from the CSV; a change here should mean the
    CSV changed, not that someone re-fit anything.
    """

    def test_thirteen_sites_loaded(self):
        assert len(VERNEZ_TABLE_S1_VIS) == 13

    def test_five_postures(self):
        assert len(VERNEZ_TABLE_S1_POSTURES) == 5
        for site, postures in VERNEZ_TABLE_S1_VIS.items():
            assert set(postures) == set(VERNEZ_TABLE_S1_POSTURES), site

    def test_pins_face_values(self):
        assert vis_table_s1("face", "seated") == pytest.approx(53.7)
        assert vis_table_s1("face", "kneeling") == pytest.approx(28.7)
        assert vis_table_s1("face", "standing_erect_arms_down") == pytest.approx(46.6)
        assert vis_table_s1("face", "standing_erect_arms_up") == pytest.approx(44.9)
        assert vis_table_s1("face", "standing_bowing") == pytest.approx(19.2)

    def test_pins_forearm_and_neck(self):
        """Sanity spot-check on two more rows so this isn't just the Face row."""
        assert vis_table_s1("forearm_external", "standing_erect_arms_down") == pytest.approx(49.4)
        assert vis_table_s1("neck_back", "kneeling") == pytest.approx(84.3)
        assert vis_table_s1("low_back", "standing_bowing") == pytest.approx(86.9)

    def test_body_site_name_normalises(self):
        """Raw CSV headers ('Forearm (external)', 'Top of shoulders', ...) and
        a caller's own casing/spacing must resolve to the same row."""
        assert (vis_table_s1("Forearm (external)", "seated")
                == vis_table_s1("forearm_external", "seated"))
        assert (vis_table_s1("Top of shoulders", "seated")
                == vis_table_s1("top_of_shoulders", "seated"))

    def test_unknown_site_raises(self):
        with pytest.raises(KeyError):
            vis_table_s1("elbow", "seated")

    def test_unknown_posture_raises(self):
        with pytest.raises(KeyError):
            vis_table_s1("face", "lying_prone")

    def test_top_of_shoulders_and_upper_back_are_identical(self):
        """[VERIFY] Reproduced verbatim from the source table across all five
        postures. This is NOT a lookup bug -- do not 'fix' it by making the
        rows differ. See the CSV header and the module comment above
        VERNEZ_TABLE_S1_VIS for the caveat this test encodes."""
        for posture in VERNEZ_TABLE_S1_POSTURES:
            assert (vis_table_s1("top_of_shoulders", posture)
                    == vis_table_s1("upper_back", posture))
        assert vis_table_s1("top_of_shoulders", "seated") == pytest.approx(35.9)
        assert vis_table_s1("top_of_shoulders", "kneeling") == pytest.approx(50.3)
        assert vis_table_s1("top_of_shoulders", "standing_erect_arms_down") == pytest.approx(48.6)
        assert vis_table_s1("top_of_shoulders", "standing_erect_arms_up") == pytest.approx(45.7)
        assert vis_table_s1("top_of_shoulders", "standing_bowing") == pytest.approx(85.3)

    def test_all_values_are_valid_vis_percentages(self):
        """Every Table S1 entry must be a legal vernez_exposure_ratio() input."""
        for site, postures in VERNEZ_TABLE_S1_VIS.items():
            for posture, vis in postures.items():
                assert 0.0 < vis <= 100.0, (site, posture, vis)
                er = vernez_exposure_ratio(vis, VERNEZ_REF_SZA_DEG)
                assert 0.0 <= er <= 1.0

    def test_face_bowing_and_neck_back_reproduce_the_papers_own_reported_er(self):
        """Cross-check against VERNEZ_REPORTED_SITE_ER (mean ERs printed in the
        paper's Results TEXT, not Table S1): feeding the real Table S1 Vis for
        the matching posture through Eq. 1 should land close to the paper's
        own reported point value for the same site. This is the honest version
        of the check the old back-derivation short-circuited by construction."""
        face_bowing_vis = vis_table_s1("face", "standing_bowing")
        er = vernez_exposure_ratio(face_bowing_vis, VERNEZ_REF_SZA_DEG) * 100.0
        assert er == pytest.approx(VERNEZ_REPORTED_SITE_ER["face_bowing"], abs=2.0)

        neck_vis = vis_table_s1("neck_back", "standing_erect_arms_down")
        er = vernez_exposure_ratio(neck_vis, VERNEZ_REF_SZA_DEG) * 100.0
        assert er == pytest.approx(VERNEZ_REPORTED_SITE_ER["back_of_neck"], abs=3.0)

    def test_top_of_shoulders_standing_does_not_reproduce_the_reported_89_percent(self):
        """Documents a real discrepancy (not a bug to fix here): the paper's
        text reports 'top of shoulders' ER = 89% for the standing posture, but
        Table S1's standing_erect Vis values (45.7 / 48.6) give an Eq. 1 ER of
        only ~46-48%. The reported 89% is far closer to Table S1's
        standing_bowing Vis (85.3, ER~85%). Recorded so nobody 'fixes' this
        table to force agreement -- see the geometry/BSA report for context."""
        vis_down = vis_table_s1("top_of_shoulders", "standing_erect_arms_down")
        vis_up = vis_table_s1("top_of_shoulders", "standing_erect_arms_up")
        er_down = vernez_exposure_ratio(vis_down, VERNEZ_REF_SZA_DEG) * 100.0
        er_up = vernez_exposure_ratio(vis_up, VERNEZ_REF_SZA_DEG) * 100.0
        reported = VERNEZ_REPORTED_SITE_ER["top_of_shoulders"]
        assert reported - er_down > 30.0
        assert reported - er_up > 30.0

        vis_bow = vis_table_s1("top_of_shoulders", "standing_bowing")
        er_bow = vernez_exposure_ratio(vis_bow, VERNEZ_REF_SZA_DEG) * 100.0
        assert er_bow == pytest.approx(reported, abs=5.0)

