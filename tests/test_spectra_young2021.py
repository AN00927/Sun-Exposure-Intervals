"""
Tests for data/spectra/vitd_action_spectrum_young2021_MEASURED.csv.

Context: Young et al. 2021 (PNAS 118:e2015867118) publishes no independently
digitized or re-shaped per-wavelength action spectrum. What it publishes is a
printed transformation rule -- shift the existing CIE-174 (2006) previtamin D3
action spectrum -5 nm ("blue shift"), shape unchanged -- validated by a grid
search over integer-nm shifts (Table 3; best fit at 5 nm, P = 0.911 partial-body,
P = 0.326 full-body). See the CSV header for full quotes and page references.

These tests pin that relationship rather than raw plot-derived numbers, since
no such numbers exist in the paper. They do NOT assert this is an independent
measurement -- see data/spectra/README.md and the CSV header for that
distinction.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

SPECTRA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "spectra")

CIE_PATH = os.path.join(SPECTRA_DIR, "vitd_action_spectrum_cie.csv")
YOUNG_MEASURED_PATH = os.path.join(
    SPECTRA_DIR, "vitd_action_spectrum_young2021_MEASURED.csv"
)
YOUNG_SYNTHETIC_PATH = os.path.join(SPECTRA_DIR, "vitd_action_spectrum_young2021.csv")


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    assert list(df.columns) == ["wavelength_nm", "weight"]
    return df


@pytest.fixture(scope="module")
def cie_spectrum() -> pd.DataFrame:
    return _load(CIE_PATH)


@pytest.fixture(scope="module")
def young_measured() -> pd.DataFrame:
    return _load(YOUNG_MEASURED_PATH)


@pytest.fixture(scope="module")
def young_synthetic() -> pd.DataFrame:
    return _load(YOUNG_SYNTHETIC_PATH)


def test_file_exists_and_parses(young_measured: pd.DataFrame) -> None:
    assert not young_measured.empty


def test_peak_is_292nm_matching_5nm_blue_shift(young_measured: pd.DataFrame) -> None:
    """CIE-174 previtamin D3 peaks at 297 nm (paper, Discussion, p.7, quoting
    MacLaughlin 1982: "the peak of 297 nm from the action spectrum for
    previtamin D3 in ex vivo human skin"). A 5 nm blue shift moves the peak
    to 292 nm."""
    peak_row = young_measured.loc[young_measured["weight"].idxmax()]
    assert peak_row["wavelength_nm"] == pytest.approx(292.0)
    assert peak_row["weight"] == pytest.approx(1.0)


def test_matches_cie_shifted_5nm_exactly(
    young_measured: pd.DataFrame, cie_spectrum: pd.DataFrame
) -> None:
    """Pin the paper's own stated rule: weight(w) == CIE_previtD3(w + 5) for
    every wavelength where both curves have data (Results, p.3-4: "incremental
    1-nm blue shifts ... without changing the shape of CIE action spectrum";
    Table 3 identifies 5 nm as best fit)."""
    cie_lookup = dict(zip(cie_spectrum["wavelength_nm"], cie_spectrum["weight"]))
    checked = 0
    for _, row in young_measured.iterrows():
        target = row["wavelength_nm"] + 5.0
        if target in cie_lookup:
            assert row["weight"] == pytest.approx(cie_lookup[target]), (
                f"wavelength {row['wavelength_nm']}: expected CIE({target}) = "
                f"{cie_lookup[target]}, got {row['weight']}"
            )
            checked += 1
    # Sanity: the meaningful (non-padding) part of the curve should be checked.
    assert checked >= 40


@pytest.mark.parametrize(
    "wavelength,expected_weight",
    [
        (292.0, 1.0000),  # peak, shifted from CIE's 297 nm peak
        (291.0, 0.8000),
        (293.0, 0.9800),
        (284.0, 0.0010),  # onset of support
        (327.0, 0.0001),  # tail
    ],
)
def test_pinned_values(
    young_measured: pd.DataFrame, wavelength: float, expected_weight: float
) -> None:
    row = young_measured[young_measured["wavelength_nm"] == wavelength]
    assert not row.empty, f"wavelength {wavelength} missing from spectrum"
    assert row["weight"].iloc[0] == pytest.approx(expected_weight)


def test_support_range_shifted_5nm_from_cie(
    young_measured: pd.DataFrame, cie_spectrum: pd.DataFrame
) -> None:
    cie_nonzero = cie_spectrum[cie_spectrum["weight"] > 0]["wavelength_nm"]
    young_nonzero = young_measured[young_measured["weight"] > 0]["wavelength_nm"]
    assert cie_nonzero.min() == pytest.approx(289.0)
    assert cie_nonzero.max() == pytest.approx(332.0)
    assert young_nonzero.min() == pytest.approx(284.0)
    assert young_nonzero.max() == pytest.approx(327.0)


def test_integral_against_flat_source_matches_cie(
    young_measured: pd.DataFrame, cie_spectrum: pd.DataFrame
) -> None:
    """A pure axis translation with unchanged shape preserves the integral
    against a flat (uniform-weight) source -- the shift redistributes where
    the weighting falls on the wavelength axis but not its total mass."""

    def integral(df: pd.DataFrame) -> float:
        d = df.sort_values("wavelength_nm").reset_index(drop=True)
        total = 0.0
        for i in range(len(d) - 1):
            w0, v0 = d.loc[i, "wavelength_nm"], d.loc[i, "weight"]
            w1, v1 = d.loc[i + 1, "wavelength_nm"], d.loc[i + 1, "weight"]
            total += (v0 + v1) / 2 * (w1 - w0)
        return total

    cie_integral = integral(cie_spectrum)
    young_integral = integral(young_measured)
    assert young_integral == pytest.approx(cie_integral, rel=1e-6)


def test_measured_file_agrees_with_existing_synthetic_file(
    young_measured: pd.DataFrame, young_synthetic: pd.DataFrame
) -> None:
    """The pre-existing vitd_action_spectrum_young2021.csv (flagged in
    data/spectra/README.md as a "synthetic relabelling") turns out to already
    implement the paper's exact stated transform correctly -- its defect was
    provenance/labeling (it read as an independent digitization that does not
    exist), not its numeric content. Both files should therefore agree on
    every wavelength they share."""
    measured_lookup = dict(
        zip(young_measured["wavelength_nm"], young_measured["weight"])
    )
    synthetic_lookup = dict(
        zip(young_synthetic["wavelength_nm"], young_synthetic["weight"])
    )
    shared = set(measured_lookup) & set(synthetic_lookup)
    assert len(shared) >= 40
    for w in shared:
        assert measured_lookup[w] == pytest.approx(synthetic_lookup[w]), (
            f"wavelength {w} disagrees between the two young2021 files"
        )
