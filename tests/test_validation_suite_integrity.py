"""
Validation suite integrity tests.

Excludes invalid tests from the main validation suite.
"""

import os
import glob


def test_no_invalid_tests_in_main_suite():
    """No file in the main validation suite contains STATUS: INVALID_FOR_PHYSIOLOGICAL_VALIDATION."""
    main_files = glob.glob("vitd_validation.py") + glob.glob("tests/test_vitd_validation.py")

    for filepath in main_files:
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            content = f.read()
        assert "INVALID_FOR_PHYSIOLOGICAL_VALIDATION" not in content, (
            f"{filepath} contains INVALID_FOR_PHYSIOLOGICAL_VALIDATION marker. "
            "Invalid tests must be in validation_exploratory/ only."
        )


def test_nhanes_invalid_test_exists():
    """The archived NHANES test exists in validation_exploratory/."""
    archived = os.path.join("validation_exploratory", "validate_nhanes_seasonal_INVALID.py")
    assert os.path.exists(archived), f"Archived NHANES test not found at {archived}"


def test_young2020_check_exists():
    """The Young 2020 calibration check function is callable and importable."""
    from vitd_calibration import young2020_calibration_check
    assert callable(young2020_calibration_check)