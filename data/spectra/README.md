# Spectra

Required files:

- tl01_spectrum.csv
- cie_erythema_action_spectrum.csv
- vitd_action_spectrum_cie.csv
- vitd_action_spectrum_young2021.csv

Additional reference:

- tl01_conversion_factor.md — TL-01 physical-to-SED conversion factor (Rodríguez-Granados 2017): 1 J/cm² absolute ≈ 5.5 SED for Waldmann UV7002 with 21 TL-01 tubes. WARNING: Lloyd 2004 shows calibration factors vary ±60% across departments.

CSV format:

```csv
wavelength_nm,weight
280,0.0
281,0.0
...
```

For lamp spectra:

- `weight` means relative spectral irradiance.
- The spectrum will be normalized internally.

For action spectra:

- `weight` means dimensionless biological weighting.
- Source and normalization must be documented.

Do not fabricate spectrum values.

If a spectrum is missing, scientific dose conversion must be marked BLOCKED_MISSING_SPECTRUM.
