# Spectra

## `vitd_action_spectrum_young2021.csv` — values CORRECT, provenance was misleading

**Investigated and resolved 2026-08-17. An intermediate note here (same day)
called this file "synthetic" and said it "must never be used". That was an
overcorrection and is withdrawn — the numbers are right.**

The file is `vitd_action_spectrum_cie.csv` shifted exactly 5 nm to the blue:
**53 exact matches, 0 mismatches**, `young2021(w) == cie(w + 5)` throughout;
peak 297 → 292 nm; support 289-332 → 284-327 nm; integral against a flat source
unchanged (pure translation).

**That is exactly what Young et al. 2021 prescribes.** The paper (*PNAS*
118:e2015867118) does NOT publish an independently measured curve. It publishes
a transformation *rule*: shift the CIE-174 wavelength axis by an integer number
of nm "without changing the shape of CIE action spectrum", then grid-search
which shift (0-9 nm) best reconciles serum 25(OH)D3 dose-response slopes across
five UVR sources. Their Table 3 gives fit statistics per shift; 5 nm wins. Their
words: *"We therefore propose a 5-nm blue shift for the CIE spectrum based on
our data."*

So the file implements the paper faithfully. The defect was **labelling**: it
reads as an independent digitisation of a "Young 2021 spectrum" that does not
exist as such.

**Consequence for the "which action spectrum is in force" question (A5):** it is
not a choice between two datasets. There is one curve and a **one-parameter
shift**. The paper's own non-significant band is 3-6 nm (PB) / 4-5 nm (FB), so
5 nm is a point estimate inside a range, not a constant.

**Dark-skin caveat, in the authors' own words** (Discussion, p. 7): *"A weakness
is the lack of data on melanized skin types... The results from this study only
apply to FST I/II, and it will be important to assess the effect of skin melanin
on the action spectrum for 25(OH)D3."* Population was **n = 75, 6x FST I +
69x FST II, zero FST III-VI**. The authors explicitly disclaim generalisation.
See `calibration_approval.yaml` → `standing_directives.dark_skin_priority`.

`vitd_action_spectrum_young2021_MEASURED.csv` regenerates the same values
arithmetically from CIE-174 with full provenance in its header. Note "MEASURED"
refers to the *shift amount* being regression-fit to human 25(OH)D3 data, not to
independently measured per-wavelength weights.

**Also corrected 2026-08-17:** CIE-174 is *not* "exactly zero above 315 nm", as
several places in this repo claimed. `vitd_action_spectrum_cie.csv` has support
to 332 nm and 18 nonzero entries at/above 315 nm (0.045 at 315 nm). The
argument that depended on it survives; the stated cut-off does not.

Required files:

- tl01_spectrum.csv
- cie_erythema_action_spectrum.csv
- vitd_action_spectrum_cie.csv
- vitd_action_spectrum_young2021.csv — **see warning above**

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
