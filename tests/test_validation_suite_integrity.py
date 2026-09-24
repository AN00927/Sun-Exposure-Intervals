"""
Validation suite integrity tests.

Excludes invalid tests from the main validation suite.
"""

import importlib.util
import inspect
import os


# Anchored to this file's location, not the current working directory.
# REWRITTEN (was `glob.glob("vitd_validation.py")` / bare
# `os.path.join("validation_exploratory", ...)`): glob.glob on a relative
# pattern silently returns [] -- not an error -- when pytest is invoked from
# anywhere but the repo root, which made test_no_invalid_tests_in_main_suite
# pass vacuously (the for-loop body, and its only assertion, never ran) under
# any other CWD. Empirically verified by running this file with `-c
# /path/to/tmp/empty` (pytest's rootdir override) from outside the repo.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_invalid_tests_in_main_suite():
    """No file in the main validation suite contains STATUS: INVALID_FOR_PHYSIOLOGICAL_VALIDATION."""
    main_files = [
        os.path.join(REPO_ROOT, "vitd_validation.py"),
        os.path.join(REPO_ROOT, "tests", "test_vitd_validation.py"),
    ]

    checked = []
    for filepath in main_files:
        # REWRITTEN 2026-09-22 (this workstream): this loop used to
        # `continue` past a missing file, tolerantly. That tolerance is
        # exactly what let tests/test_vitd_validation.py's absence go
        # unnoticed -- this check silently only ever covered
        # vitd_validation.py, half its stated main_files scope, for as
        # long as the second file didn't exist. Now that
        # tests/test_vitd_validation.py exists, both entries in
        # main_files are required to exist -- a future deletion of either
        # must fail loudly here, not silently shrink this test's scope
        # again.
        assert os.path.exists(filepath), (
            f"{filepath} does not exist. Every path in main_files is "
            "expected to exist -- if this file was intentionally removed "
            "or renamed, update main_files to match; do not let this loop "
            "silently skip it."
        )
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        assert "INVALID_FOR_PHYSIOLOGICAL_VALIDATION" not in content, (
            f"{filepath} contains INVALID_FOR_PHYSIOLOGICAL_VALIDATION marker. "
            "Invalid tests must be in validation_exploratory/ only."
        )
        checked.append(filepath)

    # Closes the vacuous-pass hole directly: even with paths correctly
    # anchored, a future rename/deletion of every entry in main_files would
    # otherwise silently reduce this test to "for-loop over nothing, assert
    # never runs, PASS" again. At least one real file must have been read.
    assert checked, (
        "none of the main-suite files exist at their repo-root-anchored "
        f"paths {main_files!r}; this check has nothing to guard and would "
        "otherwise pass having tested nothing"
    )
    assert len(checked) == len(main_files), (
        "not every main_files path was checked -- see the per-path "
        "assertion above for which one is missing"
    )


def _load_test_vitd_validation_module():
    """Load tests/test_vitd_validation.py as a standalone module (there is
    no tests/__init__.py, so `import tests.test_vitd_validation` is not
    reliable) purely to inspect its test classes/functions -- not to run
    them as part of this test's own assertions."""
    path = os.path.join(REPO_ROOT, "tests", "test_vitd_validation.py")
    spec = importlib.util.spec_from_file_location(
        "_test_vitd_validation_integrity_probe", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vitd_validation_test_module_is_collectible():
    """tests/test_vitd_validation.py must exist and import cleanly.

    This is the other half of test_no_invalid_tests_in_main_suite's
    stated scope, made explicit rather than tolerated as absent: this file
    did not exist for some time (2026-09-22), so the integrity check above
    never actually exercised it. A load
    failure here (missing file, import error, syntax error) fails loudly
    instead of being silently skipped.
    """
    module = _load_test_vitd_validation_module()
    assert module is not None
    # Sanity: it should actually reference the registry it's meant to test,
    # not just be an empty or unrelated file that happens to exist.
    assert hasattr(module, "CHECK_REGISTRY"), (
        "tests/test_vitd_validation.py does not import CHECK_REGISTRY from "
        "vitd_validation -- it may no longer be testing the check registry."
    )


def test_vitd_validation_test_module_guards_farrar_required():
    """The specific governance invariant this workstream was tasked with
    restoring -- validate_farrar_2013 must be REQUIRED, not warning-only,
    because the user refused sign-off on the 2026-09-22 demotion proposal
    (see vitd_validation.py's CHECK_REGISTRY entry "Farrar 2013 [C2
    quantitative]") -- must
    have a live test guarding it in tests/test_vitd_validation.py, not
    merely a comment. This test fails if that guard is ever deleted or
    renamed away from referencing both 'farrar' and 'required'.
    """
    module = _load_test_vitd_validation_module()
    test_classes = [
        obj for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
    ]
    farrar_guard_methods = []
    for cls in test_classes:
        farrar_guard_methods += [
            name for name in vars(cls)
            if name.startswith("test_") and "farrar" in name.lower()
        ]
    # Also allow a module-level (non-class) test function.
    farrar_guard_functions = [
        name for name, obj in vars(module).items()
        if name.startswith("test_") and callable(obj) and "farrar" in name.lower()
    ]
    assert farrar_guard_methods or farrar_guard_functions, (
        "tests/test_vitd_validation.py has no test with 'farrar' in its "
        "name -- the required-status guard for validate_farrar_2013 "
        "(reverted to required 2026-09-22 after the user refused sign-off "
        "on a demotion proposal) may have been removed."
    )


def test_nhanes_invalid_test_exists():
    """The archived NHANES test exists in validation_exploratory/."""
    archived = os.path.join(REPO_ROOT, "validation_exploratory",
                            "validate_nhanes_seasonal_INVALID.py")
    assert os.path.exists(archived), f"Archived NHANES test not found at {archived}"


def test_young2020_check_exists():
    """The Young 2020 calibration check function is callable and importable."""
    from vitd_calibration import young2020_calibration_check
    assert callable(young2020_calibration_check)