"""Tests for the Matsuoka 1990 K_ATTEN derivation.

The data are in the repo (abstract, PMID 2165512) and always were. These tests
lock in the arithmetic and, more importantly, the finding that the widely cited
"51% tanning attenuation" is confounded by a baseline 25(OH)D difference.
"""

import numpy as np
import pytest

from vitd_model import ModelParams
from vitd_calibration.matsuoka1990_check import (
    derive_K_ATTEN, run,
    UNTANNED_RISE, TANNED_RISE, NAIVE_ATTENUATION,
    UNTANNED_25OHD, TANNED_25OHD,
)


class TestAbstractArithmetic:

    def test_rises_match_abstract(self):
        assert UNTANNED_RISE == pytest.approx(9.3)
        assert TANNED_RISE == pytest.approx(4.7)

    def test_naive_attenuation_reproduces_cited_51_percent(self):
        assert NAIVE_ATTENUATION == pytest.approx(0.495, abs=0.01)

    def test_tanned_subjects_started_higher(self):
        """The confound: tanned baseline 25(OH)D was 64% higher, so the naive
        ratio conflates melanin attenuation with D-feedback."""
        assert TANNED_25OHD > UNTANNED_25OHD
        assert TANNED_25OHD / UNTANNED_25OHD == pytest.approx(1.64, abs=0.02)


class TestDeconvolution:

    def test_feedback_absorbs_part_of_the_blunting(self):
        """Under any real feedback form, melanin-only attenuation must be
        SMALLER than the naive 49.5%."""
        for mode in ("linear", "biphasic", "saturating"):
            r = derive_K_ATTEN(mode)
            assert r["valid"], f"{mode} produced an invalid residual"
            assert r["melanin_attenuation"] < NAIVE_ATTENUATION

    def test_no_feedback_recovers_the_naive_number(self):
        r = derive_K_ATTEN("none")
        assert r["melanin_attenuation"] == pytest.approx(NAIVE_ATTENUATION)

    def test_result_is_feedback_model_dependent(self):
        """The abstract constrains the PRODUCT of feedback and melanin, so the
        split depends on the assumed feedback form. Reporting one number
        without stating the form would be overclaiming."""
        ks = [derive_K_ATTEN(m)["K_ATTEN"]
              for m in ("linear", "biphasic", "saturating")]
        assert max(ks) / min(ks) > 2.0, "forms should disagree materially"

    def test_smaller_assumed_tan_implies_larger_K(self):
        """K_ATTEN and excess_M trade off: the same observed attenuation needs a
        bigger coefficient if the tan is assumed smaller."""
        big = derive_K_ATTEN("linear", excess_M=0.20)["K_ATTEN"]
        small = derive_K_ATTEN("linear", excess_M=0.10)["K_ATTEN"]
        assert small > big

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            derive_K_ATTEN("nonsense")


class TestConsistencyWithModel:

    def test_current_K_ATTEN_within_implied_range(self):
        """The headline check: K_ATTEN=0.37 was derived from Young 2020's
        CONSTITUTIVE FST ratio, a different quantity from Matsuoka's
        within-subject FACULTATIVE effect. It should still be defensible."""
        out = run(verbose=False)
        lo, hi = out["implied_range"]
        assert lo <= out["current_K_ATTEN"] <= hi

    def test_run_returns_all_forms(self):
        out = run(verbose=False)
        modes = {r["feedback_mode"] for r in out["results"]}
        assert modes == {"linear", "biphasic", "saturating", "none"}
