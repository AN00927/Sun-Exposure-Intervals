#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")
from vitd_dose_conversion import convert_bogh_dose_to_effective_units

# Test with sample dose
result = convert_bogh_dose_to_effective_units(
    physical_dose_j_per_cm2=0.025,
    spectrum_name="tl01_spectrum.csv",
    action_spectrum_name="cie_erythema_action_spectrum.csv"
)
print(f"Status: {result['status']}")
print(f"Effective dose: {result['effective_dose_j_per_m2']}")
print(f"SED equivalent: {result['sed_equivalent']}")
