"""
tests/test_benchmark_webbkift.py
==================================
Coverage for vitd_benchmark_webbkift.py (previously zero tests). This is the
HELD-OUT external benchmark run_all.py section 15 quotes verbatim (see the
module's own docstring: "Nothing in this repo is calibrated to it"), and it
runs through `capture_log_safe` like every other section on that path, so a
silent exception here would degrade that report section rather than fail
the pipeline loudly.

CHARACTERISATION vs VALIDATION
-------------------------------------------------------------------------
  VALIDATION        The expected value follows from the module's own
                    documented arithmetic (the "ONE NUMBER, NOT TWO"
                    derivation in the module docstring: reported_ratio =
                    WK_MINUTES_FST_V / WK_MINUTES_WHITE; the PASS/FAIL gate
                    is WK_DOSE_RATIO_LOW <= model_dose_ratio <=
                    WK_DOSE_RATIO_HIGH; the bisection in
                    `dose_ratio_for_equal_response` must actually bracket
                    the target) -- independent of the model's specific
                    numeric prediction.

  CHARACTERISATION  The expected value is a NUMBER PINNED FROM RUNNING THE
                    CURRENT MODEL against Webb/Kift's reported constants
                    (the model's own dose ratio, minutes ratio). This is the
                    benchmark's whole POINT -- it is deliberately not
                    calibrated to match -- so a characterisation test here
                    pins "what the model currently says", not "what is
                    correct". A change in `K_const`, `K_ATTEN`, or any
                    synthesis parameter is expected to move these numbers;
                    re-derive the pin, do not treat a failure as proof the
                    fix broke something.

Runtime: `dose_ratio_for_equal_response` runs a 200-iteration bisection over
single `synthesis_per_session` evaluations; `minutes_ratio` calls
`minutes_for_target` (a bounded search) twice. The whole file runs in well
under a second.
"""

from __future__ import annotations

import pytest

import vitd_benchmark_webbkift as bk
from vitd_model import ModelParams


# ---------------------------------------------------------------------------
# Module-level reported constants -- "ONE NUMBER, NOT TWO" derivation
# ---------------------------------------------------------------------------

def test_validation_reported_ratio_matches_one_number_not_two_derivation():
    """VALIDATION: the module docstring derives the white safe-exposure dose
    cap algebraically from the SAME Webb 2018 measurement as the 2.5-3.0x
    dose ratio (2.75 * 9/25 = 0.99 SED cap, so 2.75/0.99 = 25/9 = 2.78...).
    `minutes_ratio`'s `reported_ratio` field must equal
    WK_MINUTES_FST_V / WK_MINUTES_WHITE exactly, per that derivation.
    """
    mr = bk.minutes_ratio(uv_index=3.5)
    assert mr["reported_ratio"] == pytest.approx(
        bk.WK_MINUTES_FST_V / bk.WK_MINUTES_WHITE)
    # And that ratio must fall inside the WK_DOSE_RATIO band -- it is
    # "algebraically the same fact", per the module docstring, so it had
    # better land in the same 2.5-3.0x window as the dose ratio it derives
    # from.
    assert bk.WK_DOSE_RATIO_LOW <= mr["reported_ratio"] <= bk.WK_DOSE_RATIO_HIGH


def test_validation_dose_ratio_bisection_brackets_the_target():
    """VALIDATION: `dose_ratio_for_equal_response` bisects on
    `synthesis_per_session(mid, m_dark, ...) < target`. Re-implementing the
    bracket check independently (not calling the function under test)
    confirms the returned ratio actually straddles the white-skin target
    dose within the search's own tolerance, i.e. the bisection converged to
    what it claims to have converged to.
    """
    from vitd_model import synthesis_per_session
    params = ModelParams()
    D0 = 25.0
    m_white = params.M_baseline(bk.WK_FST_WHITE)
    target = synthesis_per_session(1.0, m_white, m_white, D0, 0.0, params)
    m_dark = params.M_baseline(bk.WK_FST_DARK)

    ratio = bk.dose_ratio_for_equal_response(params, D0=D0, reference_sed=1.0)
    achieved = synthesis_per_session(ratio, m_dark, m_dark, D0, 0.0, params)
    # 200 bisection iterations over a 1e-4..1e4 bracket converges far tighter
    # than this; a loose relative tolerance just confirms "converged to the
    # right root", not exact reproduction of the search's own arithmetic.
    assert achieved == pytest.approx(target, rel=1e-3)


def test_validation_run_gate_matches_documented_threshold():
    res = bk.run(verbose=False)
    expected_ok = bk.WK_DOSE_RATIO_LOW <= res["model_dose_ratio"] <= bk.WK_DOSE_RATIO_HIGH
    assert res["webbkift_dose_ratio"] == expected_ok
    assert res["model_minutes_ratio_is_diagnostic_only"] is True
    assert set(res.keys()) == {
        "webbkift_dose_ratio", "model_dose_ratio",
        "model_minutes_ratio", "model_minutes_ratio_is_diagnostic_only",
    }


def test_validation_run_is_verbose_safe(capsys):
    """The verbose branch (including the direction-of-miss NOTE, which is
    COMPUTED rather than hardcoded per a 2026-08-25 fix noted in the module)
    must not raise regardless of whether the gate currently passes or fails.
    """
    res = bk.run(verbose=True)
    out = capsys.readouterr().out
    assert "Webb/Kift 2018" in out
    if not res["webbkift_dose_ratio"]:
        assert ("over-predicts" in out) or ("under-predicts" in out)


# ---------------------------------------------------------------------------
# Characterisation: current model output against the held-out benchmark
# ---------------------------------------------------------------------------

def test_characterisation_current_model_dose_ratio_pinned():
    """PINNED CURRENT VALUE, default ModelParams(). NOT a claim that this is
    correct -- the whole point of this module is that nothing is calibrated
    to it. Currently INSIDE the reported 2.5-3.0x band (PASS), close to its
    upper edge; a small calibration change could easily flip this gate, at
    which point this pin (and the module's own printed direction-of-miss
    NOTE) needs re-deriving, not defending.
    """
    res = bk.run(verbose=False)
    assert res["model_dose_ratio"] == pytest.approx(2.873850719007945, abs=1e-9)
    assert bk.WK_DOSE_RATIO_LOW <= res["model_dose_ratio"] <= bk.WK_DOSE_RATIO_HIGH
    assert res["webbkift_dose_ratio"] is True


def test_characterisation_current_model_minutes_ratio_pinned():
    """PINNED CURRENT VALUE (diagnostic only, per module docstring -- not
    gated). UV index 3.5, 35% BSA, target 85.8 nmol/L, default ModelParams(),
    passed explicitly to match how `run()` calls this (bare `params=None`
    takes a different, solar-E_source code path inside `minutes_for_target`
    that roughly halves the minutes -- not what the benchmark's own `run()`
    reports).
    """
    mr = bk.minutes_ratio(uv_index=3.5, params=ModelParams())
    assert mr["white"]["minutes"] == pytest.approx(63.533823296510505, abs=1e-6)
    assert mr["dark"]["minutes"] == pytest.approx(270.85415457212605, abs=1e-6)
    assert mr["ratio"] == pytest.approx(4.263148989288077, abs=1e-6)
