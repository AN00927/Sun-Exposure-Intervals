"""
Tests for nhanes_preprocessing.py's harmonization no-op fix.

Background: `harmonize_ria_to_lcms()` implements the CDC/NCHS P27 regression
equations for converting raw DiaSorin RIA 25(OH)D values to LC-MS/MS-
equivalent concentrations, but the NHANES 2001-2006 files this repo actually
loads (VID_B.csv, VID_C.csv, VID_D.csv) contain only the already-standardized
`LBDVIDMS` column, so the function is never invoked in practice.
`verify_no_harmonization_needed()` makes that fact an explicit, checked
assertion instead of a silent assumption. See nhanes_preprocessing.py's
module docstring and CLAUDE.md's NHANES pooling rule.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nhanes_preprocessing import (harmonize_ria_to_lcms,
                                  verify_no_harmonization_needed,
                                  preprocess_nhanes_data)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "Papers_and_Datasets")


# ---------------------------------------------------------------------------
# harmonize_ria_to_lcms: pure-function correctness (kept as a documented,
# guarded fallback -- see its docstring -- so it must stay correct even
# though it is not exercised by the current data files).
# ---------------------------------------------------------------------------

def test_harmonize_ria_to_lcms_known_equations():
    # CDC/NCHS P27 regression equations, verbatim from the function's
    # in-code comment.
    assert harmonize_ria_to_lcms(50.0, "2001-2002") == pytest.approx(
        6.43435 + 0.95212 * 50.0)
    assert harmonize_ria_to_lcms(50.0, "2003-2004") == pytest.approx(
        1.72786 + 0.98284 * 50.0)
    assert harmonize_ria_to_lcms(50.0, "2005-2006") == pytest.approx(
        8.36753 + 0.97012 * 50.0)


def test_harmonize_ria_to_lcms_nan_passthrough():
    assert np.isnan(harmonize_ria_to_lcms(np.nan, "2001-2002"))


def test_harmonize_ria_to_lcms_unknown_cycle_passthrough():
    # Cycles outside 2001-2006 (e.g. already-LC-MS/MS eras) are passed
    # through unchanged -- this branch exists for robustness, not because
    # it is expected to be hit with a non-RIA cycle in practice.
    assert harmonize_ria_to_lcms(50.0, "2007-2008") == 50.0


# ---------------------------------------------------------------------------
# verify_no_harmonization_needed: the explicit no-op check.
# ---------------------------------------------------------------------------

def _make_df(methods_2001_2002, raw=None, standardized=None):
    n = len(methods_2001_2002)
    if raw is None:
        raw = [50.0] * n
    if standardized is None:
        standardized = raw
    return pd.DataFrame({
        "cycle": ["2001-2002"] * n,
        "harmonized_method": methods_2001_2002,
        "vitd_raw_nmol_L": raw,
        "vitd_standardized_nmol_L": standardized,
    })


def test_verify_passes_when_already_standardized_and_identical():
    df = _make_df(["LC-MS/MS Direct (already standardized)"] * 5,
                  raw=[10.0, 20.0, 30.0, 40.0, 50.0])
    # Must not raise.
    verify_no_harmonization_needed(df)


def test_verify_passes_with_missing_label():
    df = _make_df(["Missing"] * 3, raw=[np.nan] * 3, standardized=[np.nan] * 3)
    verify_no_harmonization_needed(df)


def test_verify_raises_if_harmonize_path_was_taken():
    df = _make_df(["DiaSorin RIA (CDC Standardized)"] * 3)
    with pytest.raises(AssertionError, match="harmonize_ria_to_lcms"):
        verify_no_harmonization_needed(df)


def test_verify_raises_if_raw_and_standardized_diverge_despite_label():
    df = _make_df(["LC-MS/MS Direct (already standardized)"] * 3,
                  raw=[10.0, 20.0, 30.0],
                  standardized=[10.0, 20.0, 31.0])  # one mismatch
    with pytest.raises(AssertionError, match="numerically violated"):
        verify_no_harmonization_needed(df)


def test_verify_ignores_non_ria_cycles():
    # 2007+ rows should not be checked by this function at all -- it only
    # gates the RIA-era cycles.
    df = pd.DataFrame({
        "cycle": ["2007-2008"],
        "harmonized_method": ["some future label that would fail if checked"],
        "vitd_raw_nmol_L": [10.0],
        "vitd_standardized_nmol_L": [999.0],
    })
    verify_no_harmonization_needed(df)  # must not raise


# ---------------------------------------------------------------------------
# Integration: the real NHANES files, if present (gitignored -- absent on a
# fresh clone; CLAUDE.md notes these are user-disk-only datasets).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(DATA_DIR),
                    reason="Papers_and_Datasets/ not present (gitignored)")
def test_real_nhanes_data_harmonization_is_a_verified_noop(tmp_path):
    df = preprocess_nhanes_data(DATA_DIR, str(tmp_path))
    assert df is not None
    # preprocess_nhanes_data() already calls verify_no_harmonization_needed()
    # internally and would have raised; re-running it here pins the contract
    # even if that internal call is ever removed by accident.
    verify_no_harmonization_needed(df)

    ria_cycles = {"2001-2002", "2003-2004", "2005-2006"}
    ria_rows = df[df["cycle"].isin(ria_cycles)]
    assert not ria_rows.empty
    assert set(ria_rows["harmonized_method"].unique()) <= {
        "LC-MS/MS Direct (already standardized)", "Missing"}

    both = (ria_rows["vitd_raw_nmol_L"].notna() &
           ria_rows["vitd_standardized_nmol_L"].notna())
    assert (ria_rows.loc[both, "vitd_raw_nmol_L"] ==
           ria_rows.loc[both, "vitd_standardized_nmol_L"]).all()
