"""
Structural tests for vitd_validation.py's CHECK_REGISTRY and run_all().

This file did not exist before 2026-09-22 -- a 2026-09-22 audit flagged that
tests/test_validation_suite_integrity.py's own scope check silently only
ever covered half its stated scope (vitd_validation.py, but not this file)
as a result. It fills that gap.

These tests verify the validation *framework*: registry integrity (every
CheckSpec well-formed), the required/warning-only tier split, the run_all()
return contract, and that every registered check is callable and returns
its documented field types. They deliberately do NOT pin any check's
PASS/FAIL outcome or any simulated numeric value -- per CLAUDE.md, model
parameters (K_const, A, SC_max, ...) are being recalibrated concurrently by
another workstream, so a scientific verdict measured today may legitimately
flip tomorrow. A structural test that broke on that flip would be testing
the wrong thing.

The one exception, by design, is `test_farrar_2013_is_required`: it does
not assert a scientific outcome, it asserts a governance fact (Ahaan
refused sign-off on demoting this CheckSpec to warning-only, reverted
2026-09-22 -- see vitd_validation.py's CHECK_REGISTRY entry "Farrar 2013
[C2 quantitative]"), which is a
structural property of the registry (a `warning_only` flag), not a
parameter-dependent number.
"""

import inspect
import re

import numpy as np
import pytest

import vitd_validation as vv
from vitd_validation import CHECK_REGISTRY, CheckSpec


# ===========================================================================
# CHECK_REGISTRY structural integrity
# ===========================================================================

class TestCheckRegistryIntegrity:
    """Every entry in CHECK_REGISTRY must be a well-formed CheckSpec."""

    def test_registry_nonempty(self):
        assert len(CHECK_REGISTRY) > 0

    def test_every_entry_is_a_checkspec(self):
        for entry in CHECK_REGISTRY:
            assert isinstance(entry, CheckSpec), (
                f"{entry!r} is not a CheckSpec instance"
            )

    def test_every_entry_has_the_five_documented_fields(self):
        # CheckSpec = namedtuple('CheckSpec',
        #                        ['name', 'fn', 'warning_only', 'tier', 'tier_note'])
        for entry in CHECK_REGISTRY:
            assert entry._fields == (
                "name", "fn", "warning_only", "tier", "tier_note"
            )

    def test_names_are_nonempty_strings(self):
        for entry in CHECK_REGISTRY:
            assert isinstance(entry.name, str) and entry.name.strip(), (
                f"bad name: {entry.name!r}"
            )

    def test_names_are_unique(self):
        names = [entry.name for entry in CHECK_REGISTRY]
        dupes = sorted({n for n in names if names.count(n) > 1})
        assert not dupes, f"duplicate CHECK_REGISTRY names: {dupes}"

    def test_every_fn_is_callable(self):
        for entry in CHECK_REGISTRY:
            assert callable(entry.fn), f"{entry.name}'s fn is not callable"

    def test_every_fn_accepts_a_verbose_kwarg(self):
        for entry in CHECK_REGISTRY:
            sig = inspect.signature(entry.fn)
            assert "verbose" in sig.parameters, (
                f"{entry.name} ({entry.fn.__name__}) has no verbose parameter"
            )

    def test_warning_only_is_a_bool(self):
        for entry in CHECK_REGISTRY:
            assert isinstance(entry.warning_only, bool), (
                f"{entry.name}.warning_only is {type(entry.warning_only)}, "
                "not bool"
            )

    def test_tier_note_is_a_nonempty_string(self):
        for entry in CHECK_REGISTRY:
            assert isinstance(entry.tier_note, str) and entry.tier_note.strip(), (
                f"{entry.name} has an empty tier_note"
            )


# ===========================================================================
# Evidence-tier tagging
# ===========================================================================

class TestTierTags:
    """Every CheckSpec.tier must be one of the module's recognised tiers,
    and every tier that appears in the registry must have a print label."""

    def test_every_tier_is_recognised(self):
        valid_tiers = set(vv.TIER_ORDER)
        for entry in CHECK_REGISTRY:
            assert entry.tier in valid_tiers, (
                f"{entry.name} has unrecognised tier {entry.tier!r}; "
                f"expected one of {sorted(valid_tiers)}"
            )

    def test_every_tier_has_a_label(self):
        for entry in CHECK_REGISTRY:
            assert entry.tier in vv.TIER_LABELS, (
                f"{entry.name}'s tier {entry.tier!r} has no TIER_LABELS entry"
            )

    def test_tier_order_and_tier_labels_cover_the_same_set(self):
        assert set(vv.TIER_ORDER) == set(vv.TIER_LABELS.keys())


# ===========================================================================
# Required vs warning-only bucketing
# ===========================================================================

class TestRequiredWarningTiers:
    def test_registry_has_both_required_and_warning_checks(self):
        flags = [entry.warning_only for entry in CHECK_REGISTRY]
        assert any(not w for w in flags), "no required (warning_only=False) checks"
        assert any(w for w in flags), "no warning-only checks"

    def test_required_plus_warning_only_equals_total(self):
        n_required = sum(1 for e in CHECK_REGISTRY if not e.warning_only)
        n_warning = sum(1 for e in CHECK_REGISTRY if e.warning_only)
        assert n_required + n_warning == len(CHECK_REGISTRY)

    def test_farrar_2013_is_required(self):
        """
        The user has refused sign-off on demoting validate_farrar_2013 to
        warning-only (reverted 2026-09-22): calibration_approval.yaml
        describes it in several places as "the only dark-skin quantitative
        check in the suite." This is the specific structural invariant
        this file exists to guard.

        Mutation-proved: flipping this CheckSpec's warning_only back to
        True (the 2026-09-22 demotion proposal) makes this test fail;
        reverting restores the pass.
        """
        farrar_entries = [
            entry for entry in CHECK_REGISTRY
            if entry.fn is vv.validate_farrar_2013
        ]
        assert len(farrar_entries) == 1, (
            "expected exactly one CHECK_REGISTRY entry for "
            f"validate_farrar_2013, found {len(farrar_entries)}"
        )
        assert farrar_entries[0].warning_only is False, (
            "validate_farrar_2013's CheckSpec must have warning_only=False "
            "(required). The user refused sign-off on the 2026-09-22 "
            "demotion proposal to warning-only -- see "
            "vitd_validation.py's CHECK_REGISTRY entry \"Farrar 2013 "
            "[C2 quantitative]\"."
        )

    def test_passed_required_denominator_matches_required_count(self, capsys):
        """The printed 'PASSED REQUIRED: n / N' denominator N must equal the
        number of non-warning-only entries in CHECK_REGISTRY -- a structural
        fact about the registry, independent of which checks pass today."""
        required_count = sum(1 for e in CHECK_REGISTRY if not e.warning_only)
        vv.run_all(verbose=True)
        out = capsys.readouterr().out
        m = re.search(r"PASSED REQUIRED:\s*(\d+)\s*/\s*(\d+)", out)
        assert m is not None, "PASSED REQUIRED line not found in run_all() output"
        n_pass, n_total = int(m.group(1)), int(m.group(2))
        assert n_total == required_count
        assert 0 <= n_pass <= n_total

    def test_passed_warning_only_denominator_matches_warning_count(self, capsys):
        warning_count = sum(1 for e in CHECK_REGISTRY if e.warning_only)
        vv.run_all(verbose=True)
        out = capsys.readouterr().out
        m = re.search(r"PASSED WARNING-ONLY:\s*(\d+)\s*/\s*(\d+)", out)
        assert m is not None, "PASSED WARNING-ONLY line not found in run_all() output"
        n_pass, n_total = int(m.group(1)), int(m.group(2))
        assert n_total == warning_count
        assert 0 <= n_pass <= n_total


# ===========================================================================
# run_all() return schema
# ===========================================================================

class TestRunAllReturnSchema:
    """run_all() must return {check_name: bool} for every registered check,
    regardless of warning_only bucket (run_all.py and other external
    consumers rely on this exact contract -- see the comment immediately
    above `combined = {**results, **warning_results}` in vitd_validation.py).
    """

    def test_returns_a_dict(self):
        result = vv.run_all(verbose=False)
        assert isinstance(result, dict)

    def test_keys_match_registry_names_exactly(self):
        result = vv.run_all(verbose=False)
        expected = {entry.name for entry in CHECK_REGISTRY}
        assert set(result.keys()) == expected

    def test_length_equals_registry_length(self):
        """Guards against a silent name collision: run_all() combines a
        required-dict and a warning-dict via **-unpacking, which would
        silently drop one entry (not raise) if two CheckSpecs shared a
        name across the required/warning-only split."""
        result = vv.run_all(verbose=False)
        assert len(result) == len(CHECK_REGISTRY)

    def test_all_values_are_bool(self):
        result = vv.run_all(verbose=False)
        for name, verdict in result.items():
            assert isinstance(verdict, (bool, np.bool_)), (
                f"{name}: verdict is {type(verdict)}, not bool"
            )


# ===========================================================================
# Every registered check is callable and returns the documented type
# ===========================================================================

@pytest.mark.parametrize(
    "entry", CHECK_REGISTRY, ids=[e.name for e in CHECK_REGISTRY]
)
def test_registered_check_is_callable_and_returns_bool(entry):
    """Calls every CHECK_REGISTRY function directly (not just the subset
    hand-listed elsewhere in tests/test_validation_helpers.py, which
    currently omits validate_katten_seasonal_amplitude,
    validate_kift_2013_south_asian,
    validate_kallioglu_2024_erythema_med_ratio, and
    validate_reference_scenarios). No outcome (True/False) is asserted --
    only that the call succeeds and returns the documented bool type."""
    result = entry.fn(verbose=False)
    assert isinstance(result, (bool, np.bool_)), (
        f"{entry.name} ({entry.fn.__name__}) returned {type(result)}, not bool"
    )


@pytest.mark.parametrize(
    "entry", CHECK_REGISTRY, ids=[e.name for e in CHECK_REGISTRY]
)
def test_registered_check_verbose_and_quiet_agree(entry):
    """verbose=True must only affect printing, never the returned verdict."""
    quiet = entry.fn(verbose=False)
    loud = entry.fn(verbose=True)
    assert bool(quiet) == bool(loud), (
        f"{entry.name}: verbose=True changed the verdict "
        f"({loud!r} vs verbose=False's {quiet!r})"
    )


if __name__ == "__main__":
    pytest.main([__file__])
