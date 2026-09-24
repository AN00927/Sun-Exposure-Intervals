"""Tests for vitd_scenario module: dataclass, registry, aliases, helpers."""

import pytest
from vitd_scenario import ExposureScenario


class TestExposureScenario:
    """Test the ExposureScenario dataclass construction and properties."""

    def _make(self, **kw):
        defaults = dict(
            name="test", description="desc",
            study_citation="Author 2024", pmid_or_doi="10.1/test",
        )
        defaults.update(kw)
        return ExposureScenario(**defaults)

    def test_construction_defaults(self):
        s = self._make()
        assert s.uvi == 6.0
        assert s.exposed_bsa == 0.35
        assert s.geometry_factor == 0.35
        assert s.e_source_name == "solar_summer"
        assert s.target_fst == 3
        assert s.baseline_D0 == 25.0
        assert s.protocol_duration_days == 84
        assert s.protocol_schedule == ()
        assert s.observed_delta_D is None

    def test_construction_full(self):
        sched = ((0.0, 1.3), (2.0, 1.3), (4.0, 1.3))
        s = self._make(
            uvi=8.0, exposed_bsa=0.85, target_fst=2,
            baseline_D0=34.6, protocol_duration_days=10,
            protocol_schedule=sched, observed_delta_D=(36.1, 30.0, 42.0),
        )
        assert s.uvi == 8.0
        assert s.observed_delta_D == (36.1, 30.0, 42.0)
        assert len(s.protocol_schedule) == 3

    def test_frozen(self):
        s = self._make()
        with pytest.raises(AttributeError):
            s.uvi = 10.0

    def test_hashable(self):
        s = self._make()
        assert hash(s) == hash(self._make())
        d = {s: "value"}
        assert d[self._make()] == "value"

    def test_sed_per_minute(self):
        from vitd_objectives import SED_PER_MIN_PER_UVI
        s = self._make(uvi=6.0, geometry_factor=0.35)
        expected = SED_PER_MIN_PER_UVI * 6.0 * 0.35
        assert s.sed_per_minute == pytest.approx(expected)

    def test_sed_for_minutes(self):
        s = self._make()
        assert s.sed_for_minutes(10.0) == pytest.approx(10.0 * s.sed_per_minute)

    def test_sed_for_minutes_zero(self):
        s = self._make()
        assert s.sed_for_minutes(0.0) == 0.0

    def test_sed_for_minutes_negative(self):
        s = self._make()
        assert s.sed_for_minutes(-5.0) == 0.0

    def test_minutes_for_sed(self):
        s = self._make()
        assert s.minutes_for_sed(2.0) == pytest.approx(2.0 / s.sed_per_minute)

    def test_minutes_for_sed_zero(self):
        s = self._make()
        assert s.minutes_for_sed(0.0) == 0.0

    def test_minutes_for_sed_negative(self):
        s = self._make()
        assert s.minutes_for_sed(-3.0) == 0.0

    def test_roundtrip(self):
        s = self._make()
        for sed_val in (0.5, 1.0, 2.0, 5.0):
            mins = s.minutes_for_sed(sed_val)
            back = s.sed_for_minutes(mins)
            assert abs(back - sed_val) < 1e-10


class TestScenarioRegistry:
    """Test SCENARIO_REGISTRY contents and parameter values."""

    def test_registry_has_all_scenarios(self):
        from vitd_scenario import SCENARIO_REGISTRY
        expected = [
            "summer_midday_reference",
            "farrar2013_fst5_uk",
            "young2021_fst2_solar",
            "felton2016_fst2_matched",
            "felton2016_fst5_matched",
            "deboer2025_barrier",
            "webb2018_uk_guidance",
        ]
        assert sorted(SCENARIO_REGISTRY.keys()) == sorted(expected)

    def test_registry_values_are_dataclass(self):
        from vitd_scenario import SCENARIO_REGISTRY
        for name, s in SCENARIO_REGISTRY.items():
            assert isinstance(s, ExposureScenario), f"{name} is not ExposureScenario"
            assert s.name == name

    def test_farrar2013_parameters(self):
        from vitd_scenario import SCENARIO_REGISTRY
        s = SCENARIO_REGISTRY["farrar2013_fst5_uk"]
        assert s.target_fst == 5
        # REBASED 2026-09-22 (W-Y, decision E5): 17.2 was Felton 2016's FST V
        # baseline (Table 1), copy-pasted into this record in error -- this
        # record was bit-identical to felton2016_fst5_matched below. The
        # correct value, derived from Farrar 2013's own printed pooled
        # >=1.95-SED-subgroup stats (15.7 - 8.7 = 7.0 ng/mL x 2.496 =
        # 17.472 nmol/L), is used instead. See vitd_scenario.py's inline
        # W-Y comment on this record for the full derivation.
        assert s.baseline_D0 == 17.472
        assert s.exposed_bsa == 0.35
        assert s.observed_delta_D == 21.7
        assert len(s.protocol_schedule) == 18

    def test_young2021_parameters(self):
        from vitd_scenario import SCENARIO_REGISTRY
        s = SCENARIO_REGISTRY["young2021_fst2_solar"]
        assert s.target_fst == 2
        assert s.baseline_D0 == 34.6
        assert s.exposed_bsa == 0.85
        assert s.e_source_name == "arimed_b"
        assert s.observed_delta_D == 36.1
        assert len(s.protocol_schedule) == 5

    def test_deboer2025_bsa(self):
        from vitd_scenario import SCENARIO_REGISTRY
        s = SCENARIO_REGISTRY["deboer2025_barrier"]
        assert s.exposed_bsa == 0.10
        assert s.target_fst == 3
        assert len(s.protocol_schedule) == 27

    def test_all_schedules_positive_sed(self):
        from vitd_scenario import SCENARIO_REGISTRY
        for name, s in SCENARIO_REGISTRY.items():
            for day, sed_val in s.protocol_schedule:
                assert sed_val > 0, f"{name}: SED {sed_val} at day {day}"

    def test_all_scenarios_have_citations(self):
        from vitd_scenario import SCENARIO_REGISTRY
        for name, s in SCENARIO_REGISTRY.items():
            assert s.study_citation, f"{name} missing citation"
            assert s.pmid_or_doi, f"{name} missing doi/pmid"


class TestBackwardCompatibility:
    """Test backward-compatible aliases delegating to default scenario."""

    def test_uv_index_matches_default(self):
        from vitd_scenario import UV_INDEX, SCENARIO_REGISTRY
        default = SCENARIO_REGISTRY["summer_midday_reference"]
        assert UV_INDEX == default.uvi

    def test_geometry_factor_matches_default(self):
        from vitd_scenario import GEOMETRY_FACTOR, SCENARIO_REGISTRY
        default = SCENARIO_REGISTRY["summer_midday_reference"]
        assert GEOMETRY_FACTOR == default.geometry_factor

    def test_exposed_bsa_matches_default(self):
        from vitd_scenario import EXPOSED_BSA, SCENARIO_REGISTRY
        default = SCENARIO_REGISTRY["summer_midday_reference"]
        assert EXPOSED_BSA == default.exposed_bsa

    def test_sed_per_minute_matches_default(self):
        from vitd_scenario import SED_PER_MINUTE, SCENARIO_REGISTRY
        default = SCENARIO_REGISTRY["summer_midday_reference"]
        assert SED_PER_MINUTE == default.sed_per_minute

    def test_limitation_statement_is_string(self):
        from vitd_scenario import LIMITATION_STATEMENT
        assert isinstance(LIMITATION_STATEMENT, str)
        assert "UV index" in LIMITATION_STATEMENT

    def test_describe_returns_string(self):
        from vitd_scenario import describe
        result = describe()
        assert isinstance(result, str)
        assert "FROZEN REFERENCE SCENARIO" in result

    def test_sed_for_minutes_matches_default(self):
        from vitd_scenario import sed_for_minutes, SCENARIO_REGISTRY
        default = SCENARIO_REGISTRY["summer_midday_reference"]
        assert sed_for_minutes(10.0) == default.sed_for_minutes(10.0)

    def test_minutes_for_sed_matches_default(self):
        from vitd_scenario import minutes_for_sed, SCENARIO_REGISTRY
        default = SCENARIO_REGISTRY["summer_midday_reference"]
        assert minutes_for_sed(1.0) == default.minutes_for_sed(1.0)


class TestHelperFunctions:
    """Test get_scenario, list_scenarios, and simulate_scenario."""

    def test_list_scenarios(self):
        from vitd_scenario import list_scenarios
        names = list_scenarios()
        assert len(names) == 7
        assert names == sorted(names)

    def test_get_scenario(self):
        from vitd_scenario import get_scenario
        for name in ["farrar2013_fst5_uk", "young2021_fst2_solar"]:
            s = get_scenario(name)
            assert isinstance(s, ExposureScenario)
            assert s.name == name

    def test_get_scenario_unknown_raises(self):
        from vitd_scenario import get_scenario
        with pytest.raises(KeyError):
            get_scenario("nonexistent")

    def test_simulate_scenario_returns_dict(self):
        from vitd_scenario import simulate_scenario
        result = simulate_scenario("summer_midday_reference")
        assert isinstance(result, dict)

    def test_simulate_scenario_has_trajectories(self):
        from vitd_scenario import simulate_scenario
        result = simulate_scenario("summer_midday_reference")
        assert "t" in result and "D" in result
        assert len(result["t"]) == len(result["D"])
        assert len(result["t"]) > 0

    def test_simulate_scenario_has_cancer_risk(self):
        from vitd_scenario import simulate_scenario
        result = simulate_scenario("summer_midday_reference")
        assert "scc_index" in result
        assert "intermittent_index" in result
        assert "burn_fraction" in result
        assert result["scc_index"] >= 0
        assert result["intermittent_index"] >= 0
        assert 0.0 <= result["burn_fraction"] <= 1.0

    def test_simulate_scenario_has_metadata(self):
        from vitd_scenario import simulate_scenario
        result = simulate_scenario("farrar2013_fst5_uk")
        assert result["scenario_name"] == "farrar2013_fst5_uk"
        assert "scenario_metadata" in result
        assert result["scenario_metadata"]["citation"].startswith("Farrar")

    def test_simulate_scenario_D_never_negative(self):
        from vitd_scenario import simulate_scenario
        for name in ["summer_midday_reference", "farrar2013_fst5_uk",
                      "felton2016_fst2_matched", "deboer2025_barrier"]:
            result = simulate_scenario(name)
            assert all(result["D"] >= 0), f"{name}: D went negative"

    def test_simulate_scenario_cumulative_sed_positive(self):
        from vitd_scenario import simulate_scenario
        for name in ["summer_midday_reference", "young2021_fst2_solar"]:
            result = simulate_scenario(name)
            assert result["cumulative_SED"] > 0

    @pytest.mark.parametrize("name", [
        "summer_midday_reference",
        "farrar2013_fst5_uk",
        "young2021_fst2_solar",
        "felton2016_fst2_matched",
        "felton2016_fst5_matched",
        "deboer2025_barrier",
        "webb2018_uk_guidance",
    ])
    def test_simulate_all_scenarios(self, name):
        from vitd_scenario import simulate_scenario
        result = simulate_scenario(name)
        assert isinstance(result, dict)
        assert len(result["t"]) > 0
        assert all(result["D"] >= 0)
        assert result["cumulative_SED"] > 0
        assert "scc_index" in result

    def test_farrar2013_delta_plausible(self):
        from vitd_scenario import simulate_scenario
        result = simulate_scenario("farrar2013_fst5_uk")
        # REBASED 2026-09-22 (W-Y, decision E5): matches the scenario's
        # corrected baseline_D0 (was 17.2, Felton's value copy-pasted in
        # error; see test_farrar2013_parameters and vitd_scenario.py).
        D0 = 17.472
        D_final = result["D"][-1]
        delta = D_final - D0
        assert 5.0 < delta < 50.0, f"Farrar delta {delta:.1f} out of range"
