"""
Dose conversion utilities.

This module converts physical UV doses into biologically weighted doses.

No scientific parameter fitting is performed here.
All spectra and action spectra must come from documented sources.
"""

from __future__ import annotations

import os
import pandas as pd

SPECTRA_DIR = os.path.join(os.path.dirname(__file__), "data", "spectra")


def _load_spectrum(filename: str) -> pd.DataFrame | None:
    """Load a spectrum CSV from the data/spectra directory. Returns None if missing or empty."""
    path = os.path.join(SPECTRA_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or "wavelength_nm" not in df.columns or "weight" not in df.columns:
        return None
    return df


def normalize_spectrum(
    spectrum: pd.DataFrame,
    wavelength_min: float = 280.0,
    wavelength_max: float = 320.0,
) -> pd.DataFrame:
    """
    Normalize a lamp spectrum so that the integrated relative irradiance equals 1.

    Input columns:
        wavelength_nm
        weight

    Returns a copy with columns:
        wavelength_nm
        weight
        normalized_weight
    """
    mask = (spectrum["wavelength_nm"] >= wavelength_min) & (spectrum["wavelength_nm"] <= wavelength_max)
    subset = spectrum.loc[mask].copy()
    total = subset["weight"].sum()
    if total == 0:
        subset["normalized_weight"] = 0.0
    else:
        subset["normalized_weight"] = subset["weight"] / total
    return subset[["wavelength_nm", "weight", "normalized_weight"]]


def physical_dose_to_effective_j_per_m2(
    physical_dose_j_per_cm2: float,
    spectrum: pd.DataFrame,
    action_spectrum: pd.DataFrame,
    wavelength_min: float = 280.0,
    wavelength_max: float = 320.0,
) -> float:
    """
    Convert physical dose in J/cm2 to effective dose in J/m2.

    Formula:
        D_physical_J_per_m2 = physical_dose_j_per_cm2 * 10000

        D_effective_J_per_m2 =
            D_physical_J_per_m2 *
            sum_lambda(
                normalized_spectrum(lambda) *
                action_spectrum(lambda)
            )
    """
    normed = normalize_spectrum(spectrum, wavelength_min, wavelength_max)
    merged = normed.merge(action_spectrum, on="wavelength_nm", how="inner", suffixes=("_spec", "_action"))
    if merged.empty:
        return 0.0
    weighting = (merged["normalized_weight"] * merged["weight_action"]).sum()
    physical_j_per_m2 = physical_dose_j_per_cm2 * 10000.0
    return physical_j_per_m2 * weighting


def effective_j_per_m2_to_sed(effective_j_per_m2: float) -> float:
    """
    Convert erythemally effective J/m2 to SED.

    1 SED = 100 J/m2 erythemally weighted.
    """
    return effective_j_per_m2 / 100.0


def convert_bogh_dose_to_effective_units(
    physical_dose_j_per_cm2: float,
    spectrum_name: str,
    action_spectrum_name: str,
) -> dict:
    """
    Return a dictionary containing:

        physical_dose_j_per_cm2
        physical_dose_j_per_m2
        effective_dose_j_per_m2
        sed_equivalent
        spectrum_name
        action_spectrum_name
        status

    If required spectrum files are missing or empty, return:

        status = BLOCKED_MISSING_SPECTRUM

    Do not fabricate spectra.
    """
    spectrum = _load_spectrum(spectrum_name)
    action = _load_spectrum(action_spectrum_name)

    if spectrum is None or action is None:
        return {
            "physical_dose_j_per_cm2": physical_dose_j_per_cm2,
            "physical_dose_j_per_m2": physical_dose_j_per_cm2 * 10000.0,
            "effective_dose_j_per_m2": None,
            "sed_equivalent": None,
            "spectrum_name": spectrum_name,
            "action_spectrum_name": action_spectrum_name,
            "status": "BLOCKED_MISSING_SPECTRUM",
        }

    eff = physical_dose_to_effective_j_per_m2(physical_dose_j_per_cm2, spectrum, action)
    sed = effective_j_per_m2_to_sed(eff)

    return {
        "physical_dose_j_per_cm2": physical_dose_j_per_cm2,
        "physical_dose_j_per_m2": physical_dose_j_per_cm2 * 10000.0,
        "effective_dose_j_per_m2": eff,
        "sed_equivalent": sed,
        "spectrum_name": spectrum_name,
        "action_spectrum_name": action_spectrum_name,
        "status": "OK",
    }
