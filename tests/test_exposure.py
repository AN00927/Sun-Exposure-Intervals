"""
tests/test_exposure.py
========================
Tests for vitd_exposure.py -- the PHASE 1 entry point joining vitd_solar.py
(where is the sun), vitd_ambient.py (what is the ambient UV Index) and
vitd_physiology.py/vitd_objectives.py (personal-dose geometry) into
(latitude, longitude, datetime, exposed sites, posture) -> SED.

NO NETWORK ACCESS. exposure_sed() takes an already-known ambient UV Index
(no network at all). exposure_sed_live() is tested with an injected
`http_get` stub reusing the SAME real recorded fixture
(tests/fixtures/open_meteo_forecast_la_20260817.json) that
tests/test_ambient.py uses, exactly the pattern documented in that module's
own "TESTING WITHOUT NETWORK" section -- not a second, separately-invented
fixture.

Test groups
-----------
1. BSA table integrity   -- the Lund-Browder-derived BSA_SITE_FRACTION table
                             sums, excludes the right regions, and covers
                             exactly the 13 Vernez Table S1 sites.
2. Area-weighted Vis/BSA -- vis_percent_for_site_set / exposed_bsa_for_site_set
                             arithmetic, checked against hand-computed values.
3. Structural consistency-- the repo open item 5b fix: ExposureGeometry ties
                             vis_percent and exposed_bsa to ONE site set and
                             cannot be desynced (immutability, dedup, single
                             construction path).
4. Presets                -- every named preset resolves, and its (vis, bsa)
                             pair is pinned against values computed directly
                             from vis_table_s1 + BSA_SITE_FRACTION so this
                             test would fail if the preset definitions or the
                             weighting arithmetic ever drift apart.
5. exposure_sed()          -- the deterministic end-to-end chain: solar
                             position + geometry + ambient UVI -> SED, and
                             its input-validation failure modes.
6. exposure_sed_live()     -- same chain via a stubbed vitd_ambient fetch,
                             matching a real recorded UVI value exactly.
"""

import datetime as dt
from pathlib import Path

import pytest

import vitd_exposure as ve
from vitd_ambient import AmbientUVIUnavailable
from vitd_objectives import SED_PER_MIN_PER_UVI
from vitd_physiology import vis_table_s1

UTC = dt.timezone.utc
FIXTURES = Path(__file__).parent / "fixtures"
LA_FIXTURE = (FIXTURES / "open_meteo_forecast_la_20260817.json").read_text(encoding="utf-8")


def _stub(text: str):
    def _get(url, timeout):
        return text
    return _get


# ---------------------------------------------------------------------------
# 1. BSA table integrity
# ---------------------------------------------------------------------------
class TestBSATableIntegrity:
    def test_lund_browder_sums_to_100(self):
        assert sum(ve.LUND_BROWDER_ADULT_BSA_PERCENT.values()) == pytest.approx(100.0)

    def test_excluded_regions_are_buttock_genitalia_foot(self):
        assert set(ve.BSA_EXCLUDED_LUND_BROWDER_REGIONS) == {"buttock", "genitalia", "foot"}
        assert ve.BSA_EXCLUDED_PERCENT_TOTAL == pytest.approx(5.0 + 1.0 + 7.0)

    def test_excluded_regions_never_appear_in_site_fraction_table(self):
        for region in ve.BSA_EXCLUDED_LUND_BROWDER_REGIONS:
            assert region not in ve.BSA_SITE_FRACTION

    def test_bsa_site_fraction_covers_exactly_the_13_vernez_sites(self):
        # Every site vis_table_s1 can look up must have a BSA weight, and
        # vice versa -- otherwise vis_percent_for_site_set and
        # exposed_bsa_for_site_set would silently disagree on domain.
        known_vernez_sites = {
            "face", "skull", "forearm_external", "upper_arm_external",
            "neck_back", "top_of_shoulders", "belly", "upper_back",
            "hand_back", "shoulder", "upper_leg_front", "lower_leg_back",
            "low_back",
        }
        assert set(ve.BSA_SITE_FRACTION) == known_vernez_sites
        assert len(ve.BSA_SITE_FRACTION) == 13

    def test_bsa_site_fraction_total_is_87_percent(self):
        # 100% minus buttock(5) + genitalia(1) + foot(7) = 87%.
        assert ve.BSA_SITE_FRACTION_TOTAL == pytest.approx(0.87)

    def test_every_bsa_fraction_is_positive_and_less_than_one(self):
        for site, frac in ve.BSA_SITE_FRACTION.items():
            assert 0.0 < frac < 1.0, site


# ---------------------------------------------------------------------------
# 2. Area-weighted Vis / BSA arithmetic
# ---------------------------------------------------------------------------
class TestAreaWeightedArithmetic:
    def test_vis_percent_single_site_equals_its_own_vis(self):
        v = ve.vis_percent_for_site_set(["face"], "seated")
        assert v == pytest.approx(vis_table_s1("face", "seated"))

    def test_vis_percent_two_sites_is_bsa_weighted_mean(self):
        # Hand-computed against the source CSV values and BSA weights.
        face_vis = vis_table_s1("face", "seated")       # 53.7
        hand_vis = vis_table_s1("hand_back", "seated")  # 59.2
        w_face = ve.BSA_SITE_FRACTION["face"]            # 0.035
        w_hand = ve.BSA_SITE_FRACTION["hand_back"]        # 0.05
        expected = (w_face * face_vis + w_hand * hand_vis) / (w_face + w_hand)
        got = ve.vis_percent_for_site_set(["face", "hand_back"], "seated")
        assert got == pytest.approx(expected)
        assert got == pytest.approx(56.935294, abs=1e-4)

    def test_vis_percent_empty_raises(self):
        with pytest.raises(ValueError):
            ve.vis_percent_for_site_set([], "seated")

    def test_vis_percent_unknown_site_raises_keyerror(self):
        with pytest.raises(KeyError):
            ve.vis_percent_for_site_set(["not_a_real_site"], "seated")

    def test_exposed_bsa_sums_fractions(self):
        bsa = ve.exposed_bsa_for_site_set(["face", "hand_back"])
        assert bsa == pytest.approx(ve.BSA_SITE_FRACTION["face"] + ve.BSA_SITE_FRACTION["hand_back"])

    def test_exposed_bsa_duplicate_site_counted_once(self):
        # A body site cannot become "more bare" by being named twice.
        once = ve.exposed_bsa_for_site_set(["face"])
        twice = ve.exposed_bsa_for_site_set(["face", "face", "Face"])
        assert once == pytest.approx(twice)

    def test_exposed_bsa_empty_raises(self):
        with pytest.raises(ValueError):
            ve.exposed_bsa_for_site_set([])

    def test_exposed_bsa_unknown_site_raises_keyerror(self):
        with pytest.raises(KeyError):
            ve.exposed_bsa_for_site_set(["not_a_real_site"])

    def test_all_sites_bare_hits_the_87_percent_ceiling(self):
        bsa = ve.exposed_bsa_for_site_set(ve.BSA_SITE_FRACTION.keys())
        assert bsa == pytest.approx(0.87)


# ---------------------------------------------------------------------------
# 3. Structural consistency -- repo open item 5b
# ---------------------------------------------------------------------------
class TestGeometryConsistency:
    def test_vis_and_bsa_come_from_the_same_call(self):
        sites = ("face", "forearm_external", "hand_back")
        g = ve.geometry_for_site_set(sites, "standing_erect_arms_down")
        assert g.vis_percent == pytest.approx(
            ve.vis_percent_for_site_set(sites, "standing_erect_arms_down"))
        assert g.exposed_bsa == pytest.approx(ve.exposed_bsa_for_site_set(sites))

    def test_geometry_is_immutable(self):
        g = ve.geometry_for_site_set(["face"], "seated")
        with pytest.raises(Exception):
            g.vis_percent = 999.0
        with pytest.raises(Exception):
            g.exposed_bsa = 0.5

    def test_low_vis_site_cannot_be_paired_with_a_larger_high_vis_bsa(self):
        # The scenario the task calls out explicitly: a caller claiming a
        # single low-Vis site (face, 46.6% standing) while wanting the
        # larger exposed_bsa of higher-Vis sites (forearm 49.4%, neck
        # 67.6%). With this API that is impossible -- there is no
        # constructor that accepts vis_percent and exposed_bsa
        # independently; both always come from resolving ONE `sites` tuple.
        face_only = ve.geometry_for_site_set(["face"], "standing_erect_arms_down")
        larger_set = ve.geometry_for_site_set(
            ["face", "forearm_external", "neck_back"], "standing_erect_arms_down")
        # The larger site set's exposed_bsa is bigger...
        assert larger_set.exposed_bsa > face_only.exposed_bsa
        # ...and its vis_percent reflects the ADDED higher-Vis sites, not the
        # face-only value -- there is no way to get the small face-only Vis
        # together with the larger set's BSA out of this API.
        assert larger_set.vis_percent != pytest.approx(face_only.vis_percent)
        assert larger_set.vis_percent > face_only.vis_percent  # forearm/neck > face here

    def test_duplicate_sites_deduped_and_order_preserving(self):
        g = ve.geometry_for_site_set(["face", "face", "hand_back"], "seated")
        assert g.sites == ("face", "hand_back")

    def test_unknown_posture_raises(self):
        with pytest.raises(KeyError):
            ve.geometry_for_site_set(["face"], "lounging")

    def test_skull_included_gets_a_note(self):
        g = ve.geometry_for_site_set(["face", "skull"], "seated")
        assert any("skull" in n for n in g.notes)

    def test_skull_excluded_no_note(self):
        g = ve.geometry_for_site_set(["face"], "seated")
        assert g.notes == ()


# ---------------------------------------------------------------------------
# 4. Presets
# ---------------------------------------------------------------------------
class TestPresets:
    @pytest.mark.parametrize("name", list(ve.CLOTHING_PRESETS))
    def test_preset_resolves_and_matches_direct_computation(self, name):
        g = ve.geometry_for_preset(name, "standing_erect_arms_down")
        sites = ve.CLOTHING_PRESETS[name]
        assert g.vis_percent == pytest.approx(
            ve.vis_percent_for_site_set(sites, "standing_erect_arms_down"))
        assert g.exposed_bsa == pytest.approx(ve.exposed_bsa_for_site_set(sites))
        assert name in ve.CLOTHING_PRESET_DESCRIPTIONS

    def test_presets_ordered_by_increasing_bsa(self):
        # face_and_hands < short_sleeves < shorts_and_tshirt < swimwear, by
        # construction (each preset's site set is a superset expansion of
        # the previous one plus legs/torso).
        order = ["face_and_hands", "short_sleeves", "shorts_and_tshirt", "swimwear"]
        bsas = [ve.geometry_for_preset(n, "standing_erect_arms_down").exposed_bsa
               for n in order]
        assert bsas == sorted(bsas)

    def test_swimwear_is_the_87_percent_ceiling_minus_skull(self):
        g = ve.geometry_for_preset("swimwear", "standing_erect_arms_down")
        assert g.exposed_bsa == pytest.approx(0.87 - ve.BSA_SITE_FRACTION["skull"])

    def test_no_preset_includes_skull(self):
        for sites in ve.CLOTHING_PRESETS.values():
            assert "skull" not in sites

    def test_no_preset_includes_excluded_regions(self):
        # Sanity: none of buttock/genitalia/foot's absence should silently
        # let an excluded region sneak in under a different site name.
        for sites in ve.CLOTHING_PRESETS.values():
            for s in sites:
                assert s in ve.BSA_SITE_FRACTION

    def test_unknown_preset_raises_keyerror(self):
        with pytest.raises(KeyError):
            ve.geometry_for_preset("nudist_colony", "standing_erect_arms_down")


# ---------------------------------------------------------------------------
# 5. exposure_sed() -- deterministic end-to-end chain
# ---------------------------------------------------------------------------
class TestExposureSED:
    WHEN = dt.datetime(2026, 7, 1, 12, 0, tzinfo=UTC)

    def test_basic_scenario_positive_sed(self):
        r = ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=30,
                            ambient_uv_index=8.0, preset="short_sleeves")
        assert r.sed > 0.0
        assert 0.0 < r.geometry_factor <= 1.0
        assert r.exposed_bsa == pytest.approx(
            ve.geometry_for_preset("short_sleeves", "standing_erect_arms_down").exposed_bsa)

    def test_zero_minutes_gives_zero_sed(self):
        r = ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=0,
                            ambient_uv_index=8.0, preset="swimwear")
        assert r.sed == pytest.approx(0.0)

    def test_zero_uv_index_gives_zero_sed(self):
        r = ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=60,
                            ambient_uv_index=0.0, preset="swimwear")
        assert r.sed == pytest.approx(0.0)

    def test_negative_minutes_raises(self):
        with pytest.raises(ValueError):
            ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=-5,
                            ambient_uv_index=8.0, preset="swimwear")

    def test_negative_uv_index_raises(self):
        with pytest.raises(ValueError):
            ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=30,
                            ambient_uv_index=-1.0, preset="swimwear")

    def test_neither_sites_nor_preset_raises(self):
        with pytest.raises(ValueError):
            ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=30, ambient_uv_index=8.0)

    def test_both_sites_and_preset_raises(self):
        with pytest.raises(ValueError):
            ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=30, ambient_uv_index=8.0,
                            sites=["face"], preset="swimwear")

    def test_more_bsa_preset_yields_higher_sed_same_inputs(self):
        # Higher-Vis site sets (swimwear pulls in higher-Vis limbs/trunk
        # sites) should raise the geometry factor and therefore the SED,
        # for identical minutes/UVI/time/place.
        r_small = ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=30,
                                  ambient_uv_index=8.0, preset="face_and_hands")
        r_large = ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=30,
                                  ambient_uv_index=8.0, preset="swimwear")
        assert r_large.sed >= r_small.sed

    def test_sed_matches_manual_geometry_factor_computation(self):
        r = ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=45,
                            ambient_uv_index=6.0, preset="shorts_and_tshirt")
        manual = 45 * 6.0 * SED_PER_MIN_PER_UVI * r.geometry_factor
        assert r.sed == pytest.approx(manual)

    def test_naive_datetime_requires_utc_offset(self):
        naive = dt.datetime(2026, 7, 1, 12, 0)
        # naive datetime with no utc_offset_hours -> vitd_solar raises ValueError
        with pytest.raises(ValueError):
            ve.exposure_sed(34.05, -118.25, naive, minutes=30,
                            ambient_uv_index=8.0, preset="swimwear")

    def test_naive_datetime_with_utc_offset_works(self):
        naive = dt.datetime(2026, 7, 1, 5, 0)  # 05:00 local, UTC-7 -> 12:00 UTC
        r = ve.exposure_sed(34.05, -118.25, naive, minutes=30,
                            ambient_uv_index=8.0, preset="swimwear",
                            utc_offset_hours=-7.0)
        assert r.sed > 0.0
        assert r.timestamp_utc.startswith("2026-07-01T12:00")

    def test_custom_site_set_matches_preset_with_same_sites(self):
        r_preset = ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=30,
                                   ambient_uv_index=8.0, preset="face_and_hands")
        r_custom = ve.exposure_sed(34.05, -118.25, self.WHEN, minutes=30,
                                   ambient_uv_index=8.0,
                                   sites=ve.CLOTHING_PRESETS["face_and_hands"])
        assert r_preset.sed == pytest.approx(r_custom.sed)
        assert r_preset.exposed_bsa == pytest.approx(r_custom.exposed_bsa)


# ---------------------------------------------------------------------------
# 6. exposure_sed_live() -- network path, stubbed with a real fixture
# ---------------------------------------------------------------------------
class TestExposureSEDLive:
    def test_matches_recorded_la_uvi(self):
        # Same instant/place as tests/test_ambient.py's happy-path test:
        # uv_index_live == 8.10 at 2026-08-17T20:00Z. Confirms
        # exposure_sed_live() actually threads the fetched UVI through,
        # not a placeholder.
        when = dt.datetime(2026, 8, 17, 20, 0, tzinfo=UTC)
        r = ve.exposure_sed_live(34.05, -118.25, when, minutes=30,
                                 preset="short_sleeves",
                                 http_get=_stub(LA_FIXTURE))
        assert r.ambient_uv_index == pytest.approx(8.10)
        direct = ve.exposure_sed(34.05, -118.25, when, minutes=30,
                                 ambient_uv_index=8.10, preset="short_sleeves")
        assert r.sed == pytest.approx(direct.sed)

    def test_propagates_ambient_unavailable(self):
        when = dt.datetime(2026, 8, 17, 20, 0, tzinfo=UTC)
        with pytest.raises(AmbientUVIUnavailable):
            ve.exposure_sed_live(999.0, -118.25, when, minutes=30,
                                 preset="short_sleeves",
                                 http_get=_stub(LA_FIXTURE))


# ---------------------------------------------------------------------------
# Regression: the 5b guarantee must hold on EVERY construction path
# ---------------------------------------------------------------------------
# ADDED 2026-08-18. The original module claimed "there is no constructor path
# that accepts vis_percent and exposed_bsa independently", resting on
# frozen=True. That claim was false: frozen blocks attribute assignment and
# nothing else. Both paths below were verified to produce a low single-site Vis
# carried on a BSA belonging to a different, higher-Vis site set -- precisely
# the inconsistency repo open item 5b exists to eliminate. Enforcement now
# lives in ExposureGeometry.__post_init__, so it covers every path.
import dataclasses as _dataclasses

import pytest as _pytest

from vitd_exposure import ExposureGeometry as _EG, geometry_for_preset as _gfp

_POSTURE = "standing_erect_arms_down"


class TestGeometryConsistencyIsStructural:

    def test_dataclasses_replace_cannot_desync_vis_from_sites(self):
        g = _gfp("short_sleeves", _POSTURE)
        with _pytest.raises(ValueError, match="not derivable from sites"):
            _dataclasses.replace(g, vis_percent=10.0)

    def test_dataclasses_replace_cannot_desync_bsa_from_sites(self):
        g = _gfp("short_sleeves", _POSTURE)
        with _pytest.raises(ValueError, match="not derivable from sites"):
            _dataclasses.replace(g, exposed_bsa=0.60)

    def test_direct_construction_cannot_pair_low_vis_with_large_bsa(self):
        # The concrete hazard: face-only Vis (~46.6%) claimed over a BSA that
        # would have to include forearms/neck/shoulders (all higher Vis).
        g = _gfp("short_sleeves", _POSTURE)
        with _pytest.raises(ValueError, match="not derivable from sites"):
            _EG(sites=g.sites, posture=_POSTURE, vis_percent=10.0,
                exposed_bsa=0.60, per_site_vis=g.per_site_vis, notes=g.notes)

    def test_blessed_constructors_are_unaffected(self):
        for preset in ("face_and_hands", "short_sleeves",
                       "shorts_and_tshirt", "swimwear"):
            g = _gfp(preset, _POSTURE)
            assert 0.0 < g.exposed_bsa <= 0.87
            assert 0.0 < g.vis_percent <= 100.0

    def test_posture_change_must_go_through_a_rebuild(self):
        # Same sites, different posture => different Vis. Swapping only the
        # posture field must fail rather than silently keep the old Vis.
        g = _gfp("short_sleeves", _POSTURE)
        with _pytest.raises(ValueError, match="not derivable from sites"):
            _dataclasses.replace(g, posture="standing_bowing")


# ---------------------------------------------------------------------------
# 8. End-to-end: exposure_sed() -> the schedule functions (Problem 1+4 wiring)
# ---------------------------------------------------------------------------
# ADDED 2026-08-19. Before this, nothing in the repo fed an ExposureResult's
# .exposed_bsa/.geometry_factor into minimum_schedule_for_sufficiency /
# optimal_interval_risk_constrained -- confirmed by grep at the time this plan
# was written. These are the regression tests that pin the join.
from vitd_optimization import (            # noqa: E402
    minimum_schedule_for_sufficiency, optimal_interval_risk_constrained,
)
from vitd_objectives import sed_from_minutes as _sed_from_minutes  # noqa: E402

_LAGOS_LAT, _LAGOS_LON = 6.5244, 3.3792
_LAGOS_WHEN = dt.datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


class TestExposureFeedsScheduleFunctions:
    def test_minimum_schedule_accepts_live_exposure_result(self):
        r = ve.exposure_sed(_LAGOS_LAT, _LAGOS_LON, _LAGOS_WHEN, minutes=30,
                            ambient_uv_index=9.0, preset="face_and_hands")
        res = minimum_schedule_for_sufficiency(
            fst=5, uv_index=r.ambient_uv_index, exposed_bsa=r.exposed_bsa,
            geometry_factor=r.geometry_factor, session_budgets=(365,))
        assert res["fst"] == 5
        assert len(res["rows"]) == 1

    def test_risk_constrained_accepts_live_exposure_result(self):
        # interval_grid includes 1 day: at FST 5 with a small face_and_hands
        # exposed_bsa, weekly+ intervals are genuinely infeasible under the
        # erythema cap (a real result of the model, not a wiring bug -- see
        # TestPrimaryMinimumSchedule's monotone-in-pigmentation tests for the
        # same shape of constraint), so this only asserts SOME interval on
        # the grid resolves, not that every interval does.
        r = ve.exposure_sed(_LAGOS_LAT, _LAGOS_LON, _LAGOS_WHEN, minutes=30,
                            ambient_uv_index=9.0, preset="face_and_hands")
        res = optimal_interval_risk_constrained(
            fst=5, uv_index=r.ambient_uv_index, exposed_bsa=r.exposed_bsa,
            geometry_factor=r.geometry_factor, interval_grid=(1, 7, 14),
            max_minutes=None)
        assert res["recommended"] is not None

    def test_no_double_application_of_geometry_or_E_source(self):
        """A schedule built from ExposureResult.sed must match one built by
        calling sed_from_minutes directly with the SAME geometry_factor --
        i.e. exposure_sed()'s geometry correction and the optimizer's own
        sed_from_minutes(1.0, uv_index, geometry_factor) call must agree
        exactly, not compound. This is the audit-table regression test."""
        r = ve.exposure_sed(_LAGOS_LAT, _LAGOS_LON, _LAGOS_WHEN, minutes=30,
                            ambient_uv_index=9.0, preset="face_and_hands")
        manual_sed_per_min = _sed_from_minutes(1.0, r.ambient_uv_index,
                                               r.geometry_factor)
        assert r.sed == pytest.approx(30 * manual_sed_per_min, rel=1e-9)

        # And the optimizer, given that SAME geometry_factor, must derive an
        # identical per-minute conversion -- not a second, independent one.
        res = optimal_interval_risk_constrained(
            fst=5, uv_index=r.ambient_uv_index, geometry_factor=r.geometry_factor,
            interval_grid=(7,), max_minutes=None)
        row = res["evaluated"][0]
        assert row["minutes_per_session"] * manual_sed_per_min == pytest.approx(
            row["sed_per_session"], rel=1e-6)

    def test_more_exposed_preset_needs_fewer_minutes_same_location(self):
        # Daily interval (grid=(1,)): weekly+ intervals are infeasible at
        # FST 5 for these presets under the erythema cap (see the comment on
        # test_risk_constrained_accepts_live_exposure_result above), so this
        # isolates the exposed_bsa effect at an interval both presets can hit.
        r_small = ve.exposure_sed(_LAGOS_LAT, _LAGOS_LON, _LAGOS_WHEN, minutes=1,
                                  ambient_uv_index=9.0, preset="face_and_hands")
        r_large = ve.exposure_sed(_LAGOS_LAT, _LAGOS_LON, _LAGOS_WHEN, minutes=1,
                                  ambient_uv_index=9.0, preset="shorts_and_tshirt")
        res_small = optimal_interval_risk_constrained(
            fst=5, uv_index=9.0, exposed_bsa=r_small.exposed_bsa,
            geometry_factor=r_small.geometry_factor, interval_grid=(1,),
            max_minutes=None)["recommended"]
        res_large = optimal_interval_risk_constrained(
            fst=5, uv_index=9.0, exposed_bsa=r_large.exposed_bsa,
            geometry_factor=r_large.geometry_factor, interval_grid=(1,),
            max_minutes=None)["recommended"]
        assert res_small is not None and res_large is not None
        assert res_large["minutes_per_session"] < res_small["minutes_per_session"]


# ---------------------------------------------------------------------------
# 9. exposure_across_months() -- optional, caller-supplied-only seasonal mode
# ---------------------------------------------------------------------------
_LAGOS_UTC_OFFSET = 1.0  # West Africa Time, no network involved -- just a plain offset


class TestExposureAcrossMonths:
    def test_matches_exposure_sed_called_per_month(self):
        monthly = {1: 3.0, 7: 8.5}
        out = ve.exposure_across_months(_LAGOS_LAT, _LAGOS_LON, 2026, monthly,
                                        minutes=30, preset="face_and_hands",
                                        utc_offset_hours=_LAGOS_UTC_OFFSET)
        assert set(out) == {1, 7}
        for month, uvi in monthly.items():
            when = dt.datetime(2026, month, 15, 12, 0)
            direct = ve.exposure_sed(_LAGOS_LAT, _LAGOS_LON, when, minutes=30,
                                     ambient_uv_index=uvi, preset="face_and_hands",
                                     utc_offset_hours=_LAGOS_UTC_OFFSET)
            assert out[month].sed == pytest.approx(direct.sed)

    def test_exposed_bsa_constant_across_months_geometry_factor_can_vary(self):
        # Same clothing every month by construction -> exposed_bsa must be
        # identical; geometry_factor is allowed to differ because solar
        # elevation genuinely changes month to month.
        monthly = {1: 5.0, 4: 5.0, 7: 5.0, 10: 5.0}
        out = ve.exposure_across_months(_LAGOS_LAT, _LAGOS_LON, 2026, monthly,
                                        minutes=30, preset="shorts_and_tshirt",
                                        utc_offset_hours=_LAGOS_UTC_OFFSET)
        bsas = {r.exposed_bsa for r in out.values()}
        assert len(bsas) == 1

    def test_each_month_independent_of_the_others(self):
        # Dropping a month from the input must not change any other month's
        # result -- no cross-month interaction (no dose banking).
        full = ve.exposure_across_months(_LAGOS_LAT, _LAGOS_LON, 2026,
                                         {1: 4.0, 7: 9.0}, minutes=30,
                                         preset="face_and_hands",
                                         utc_offset_hours=_LAGOS_UTC_OFFSET)
        partial = ve.exposure_across_months(_LAGOS_LAT, _LAGOS_LON, 2026,
                                            {7: 9.0}, minutes=30,
                                            preset="face_and_hands",
                                            utc_offset_hours=_LAGOS_UTC_OFFSET)
        assert full[7].sed == pytest.approx(partial[7].sed)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            ve.exposure_across_months(_LAGOS_LAT, _LAGOS_LON, 2026, {},
                                      minutes=30, preset="face_and_hands",
                                      utc_offset_hours=_LAGOS_UTC_OFFSET)

    def test_out_of_range_month_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            ve.exposure_across_months(_LAGOS_LAT, _LAGOS_LON, 2026,
                                      {0: 5.0, 13: 5.0}, minutes=30,
                                      preset="face_and_hands",
                                      utc_offset_hours=_LAGOS_UTC_OFFSET)
