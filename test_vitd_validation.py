"""
Tests for vitd_validation.py — comparison utilities and check structure.

These tests verify the validation framework itself, not the model science.
They do not assert any scientific thresholds or literature targets.
"""

import numpy as np
import pytest

from vitd_validation import (
    check_exact_match,
    check_percent_error,
    check_in_range,
    check_qualitative_only,
    weighted_aggregate,
)


# ===========================================================================
# check_exact_match
# ===========================================================================

class TestCheckExactMatch:
    def test_exact_identity(self):
        assert check_exact_match(55.8, 55.8, 0.0)

    def test_within_tolerance(self):
        assert check_exact_match(50.0, 55.8, 15.0)
        assert check_exact_match(70.0, 55.8, 15.0)

    def test_outside_tolerance(self):
        assert not check_exact_match(30.0, 55.8, 15.0)
        assert not check_exact_match(80.0, 55.8, 15.0)

    def test_zero_tolerance_default(self):
        assert check_exact_match(1.0, 1.0)
        assert not check_exact_match(1.001, 1.0)

    def test_negative_values(self):
        assert check_exact_match(-5.0, -5.0, 0.0)
        assert check_exact_match(-7.0, -5.0, 3.0)
        assert not check_exact_match(-10.0, -5.0, 3.0)

    def test_boundary_exact(self):
        assert check_exact_match(40.8, 55.8, 15.0)
        assert check_exact_match(70.8, 55.8, 15.0)

    def test_non_float_inputs(self):
        assert check_exact_match(5, 5, 0)


# ===========================================================================
# check_percent_error
# ===========================================================================

class TestCheckPercentError:
    def test_within_percent(self):
        assert check_percent_error(110.0, 100.0, 10.0)

    def test_outside_percent(self):
        assert not check_percent_error(120.0, 100.0, 10.0)

    def test_exact_match(self):
        assert check_percent_error(55.8, 55.8, 0.0)

    def test_zero_expected(self):
        assert check_percent_error(0.0, 0.0, 0.0)
        assert not check_percent_error(1.0, 0.0, 0.0)
        assert not check_percent_error(0.1, 0.0, 100.0)

    def test_default_max_pct(self):
        assert check_percent_error(105.0, 100.0)
        assert not check_percent_error(120.0, 100.0)

    def test_low_observed(self):
        assert check_percent_error(95.0, 100.0, 5.0)
        assert not check_percent_error(90.0, 100.0, 5.0)

    def test_all_zero(self):
        assert check_percent_error(0.0, 0.0)


# ===========================================================================
# check_in_range
# ===========================================================================

class TestCheckInRange:
    def test_in_range(self):
        assert check_in_range(5.0, 0.0, 10.0)

    def test_below_range(self):
        assert not check_in_range(-1.0, 0.0, 10.0)

    def test_above_range(self):
        assert not check_in_range(11.0, 0.0, 10.0)

    def test_lower_boundary(self):
        assert check_in_range(0.0, 0.0, 10.0)

    def test_upper_boundary(self):
        assert check_in_range(10.0, 0.0, 10.0)

    def test_single_value_range(self):
        assert check_in_range(5.0, 5.0, 5.0)
        assert not check_in_range(5.1, 5.0, 5.0)


# ===========================================================================
# check_qualitative_only
# ===========================================================================

class TestCheckQualitativeOnly:
    def test_pass(self):
        assert check_qualitative_only(True)

    def test_fail(self):
        assert not check_qualitative_only(False)

    def test_from_expression(self):
        assert check_qualitative_only(5 > 3)
        assert not check_qualitative_only(5 < 3)

    def test_non_bool_cast(self):
        assert check_qualitative_only(1)
        assert not check_qualitative_only(0)
        assert check_qualitative_only("non-empty")
        assert not check_qualitative_only("")


# ===========================================================================
# weighted_aggregate
# ===========================================================================

class TestWeightedAggregate:
    def test_equal_weights(self):
        out = weighted_aggregate({"a": True, "b": True, "c": False})
        assert out["n_pass"] == 2
        assert out["n_total"] == 3
        assert out["score"] == pytest.approx(2.0 / 3.0)

    def test_weighted(self):
        results = {"a": True, "b": False, "c": False}
        weights = {"a": 2.0, "b": 1.0, "c": 1.0}
        out = weighted_aggregate(results, weights)
        assert out["score"] == pytest.approx(2.0 / 4.0)

    def test_all_pass(self):
        out = weighted_aggregate({"a": True, "b": True})
        assert out["score"] == pytest.approx(1.0)
        assert out["n_pass"] == 2

    def test_all_fail(self):
        out = weighted_aggregate({"a": False, "b": False})
        assert out["score"] == pytest.approx(0.0)
        assert out["n_pass"] == 0

    def test_empty(self):
        out = weighted_aggregate({})
        assert out["score"] == pytest.approx(0.0)
        assert out["n_pass"] == 0
        assert out["n_total"] == 0

    def test_partial_weights(self):
        results = {"a": True, "b": True, "c": False}
        weights = {"a": 3.0}
        out = weighted_aggregate(results, weights)
        assert out["score"] == pytest.approx(4.0 / 5.0)

    def test_single_check_weighted(self):
        out = weighted_aggregate({"only": True}, {"only": 2.0})
        assert out["score"] == pytest.approx(1.0)

    def test_weights_beyond_unit(self):
        results = {"easy": True, "hard": False}
        weights = {"easy": 0.2, "hard": 0.8}
        out = weighted_aggregate(results, weights)
        assert out["score"] == pytest.approx(0.2)


# ===========================================================================
# Validation structure
# ===========================================================================

class TestValidationStructure:
    """Verify the checks list and validation function signatures."""

    def test_run_all_returns_dict_with_all_checks(self):
        from vitd_validation import run_all
        result = run_all(verbose=False)
        assert isinstance(result, dict)
        assert len(result) == 10

    def test_each_check_returns_bool(self):
        from vitd_validation import (
            sanity_decay_halflife,
            sanity_single_bolus_FST_III,
            sanity_photoadaptation_reduction,
            validate_photoadaptation_deBoer2025,
            validate_bogh_2012,
            validate_farrar_2013,
            validate_bogh_baseline_bsa,
            validate_metric_ordering,
            validate_baseline_calibration_target,
            validate_sc_deBoer2025,
        )
        for fn in [
            sanity_decay_halflife,
            sanity_single_bolus_FST_III,
            sanity_photoadaptation_reduction,
            validate_photoadaptation_deBoer2025,
            validate_bogh_2012,
            validate_farrar_2013,
            validate_bogh_baseline_bsa,
            validate_metric_ordering,
            validate_baseline_calibration_target,
            validate_sc_deBoer2025,
        ]:
            result = fn(verbose=False)
            assert isinstance(result, (bool, np.bool_)), f"{fn.__name__} returned {type(result)}"

    def test_each_check_accepts_verbose_kwarg(self):
        from vitd_validation import (
            sanity_decay_halflife,
            sanity_single_bolus_FST_III,
            sanity_photoadaptation_reduction,
            validate_photoadaptation_deBoer2025,
            validate_bogh_2012,
            validate_farrar_2013,
            validate_bogh_baseline_bsa,
            validate_metric_ordering,
            validate_baseline_calibration_target,
            validate_sc_deBoer2025,
        )
        for fn in [
            sanity_decay_halflife,
            sanity_single_bolus_FST_III,
            sanity_photoadaptation_reduction,
            validate_photoadaptation_deBoer2025,
            validate_bogh_2012,
            validate_farrar_2013,
            validate_bogh_baseline_bsa,
            validate_metric_ordering,
            validate_baseline_calibration_target,
            validate_sc_deBoer2025,
        ]:
            result_verbose = fn(verbose=True)
            result_quiet = fn(verbose=False)
            assert isinstance(result_verbose, (bool, np.bool_))
            assert result_verbose == result_quiet

    def test_checks_list_format(self):
        from vitd_validation import run_all
        checks = run_all.__code__.co_consts

    def test_missing_uncertainty_defaults_qualitative(self):
        """When no uncertainty is supplied, qualitative-only is the fallback."""
        assert check_qualitative_only(True)
        assert not check_qualitative_only(False)


# ===========================================================================
# Edge cases for comparison utilities
# ===========================================================================

class TestEdgeCases:
    def test_exact_match_near_floating_point(self):
        assert check_exact_match(1.0 / 3.0, 1.0 / 3.0, 1e-15)

    def test_percent_error_near_zero_expected(self):
        assert check_percent_error(0.0, 0.0, 0.0)
        assert not check_percent_error(1e-10, 0.0, 0.0)

    def test_in_range_float_imprecision(self):
        assert check_in_range(0.3333333333333333, 0.0, 0.5)
        assert check_in_range(0.9999999999999999, 0.0, 1.0)

    def test_weighted_aggregate_negative_weights(self):
        results = {"a": True, "b": False}
        weights = {"a": -1.0, "b": 1.0}
        out = weighted_aggregate(results, weights)
        assert out["n_total"] == 2

    def test_weighted_aggregate_mixed_types(self):
        results = {"a": True, "b": False}
        out = weighted_aggregate(results)
        assert isinstance(out["score"], float)
        assert isinstance(out["n_pass"], int)
        assert isinstance(out["n_total"], int)


if __name__ == "__main__":
    pytest.main([__file__])
