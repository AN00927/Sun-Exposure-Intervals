"""
NHANES seasonal amplitude test.

STATUS: INVALID_FOR_PHYSIOLOGICAL_VALIDATION

This test is not a valid validation of K_ATTEN because NHANES seasonal
amplitude reflects behavior, clothing, latitude, supplement use,
self-reported race, and baseline differences.

Retained only for exploratory behavioral modeling.
"""

from __future__ import annotations


def validate_nhanes_seasonal_archived():
    """
    Archived NHANES seasonal amplitude validation.

    STATUS: INVALID_FOR_PHYSIOLOGICAL_VALIDATION

    This function is retained for reference only. It must NOT be included
    in the main validation suite or used for pass/fail validation.

    The NHANES seasonal amplitude test conflates:
    - behavioral differences (outdoor time, shade-seeking)
    - clothing cover variations
    - latitude effects
    - supplement use
    - self-reported race as a proxy for pigmentation
    - baseline 25(OH)D differences

    None of these are controlled physiological measurements of K_ATTEN.
    """
    print("NHANES seasonal amplitude test is INVALID for physiological validation.")
    print("Retained for exploratory behavioral modeling only.")
    return {"status": "INVALID_FOR_PHYSIOLOGICAL_VALIDATION"}
