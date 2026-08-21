"""Tests for dose conversion utilities."""

import pandas as pd
import pytest


def test_effective_j_per_m2_to_sed():
    from vitd_dose_conversion import effective_j_per_m2_to_sed
    assert effective_j_per_m2_to_sed(100.0) == 1.0


def test_physical_to_j_per_m2_conversion():
    """1 J/cm2 = 10000 J/m2. Flat action spectrum of weight 1.0, normalized flat lamp spectrum."""
    from vitd_dose_conversion import physical_dose_to_effective_j_per_m2, normalize_spectrum
    wl = list(range(280, 321, 5))
    spectrum = pd.DataFrame({"wavelength_nm": wl, "weight": [1.0] * len(wl)})
    action = pd.DataFrame({"wavelength_nm": wl, "weight": [1.0] * len(wl)})
    result = physical_dose_to_effective_j_per_m2(1.0, spectrum, action)
    # Flat normalized spectrum and flat action=1.0, weighting sum = 1.0
    # So effective = 1.0 * 10000 * 1.0 = 10000
    assert result == pytest.approx(10000.0)


def test_missing_spectrum_returns_blocked():
    from vitd_dose_conversion import convert_bogh_dose_to_effective_units
    result = convert_bogh_dose_to_effective_units(
        0.36, "nonexistent_spectrum.csv", "nonexistent_action.csv"
    )
    assert result["status"] == "BLOCKED_MISSING_SPECTRUM"
    assert result["effective_dose_j_per_m2"] is None
    assert result["sed_equivalent"] is None


def test_normalize_spectrum():
    from vitd_dose_conversion import normalize_spectrum
    df = pd.DataFrame({"wavelength_nm": [280, 290, 300], "weight": [1.0, 2.0, 1.0]})
    result = normalize_spectrum(df)
    assert result["normalized_weight"].sum() == pytest.approx(1.0)
    assert result.loc[result["wavelength_nm"] == 290, "normalized_weight"].values[0] == pytest.approx(0.5)
