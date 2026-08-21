"""
Bogh 2012 protocol representation.

STATUS: PROVISIONAL until source verification is complete.
"""

from __future__ import annotations


def bogh2012_session_doses(
    start_j_per_cm2: float = 0.36,
    escalation_fraction: float = 0.15,
    every_n_sessions: int = 2,
    cap_j_per_cm2: float = 1.20,
    n_sessions: int = 18,
) -> list[float]:
    """
    Provisional Bogh 2012 dose schedule.

    Rule:
        - Sessions 1 and 2 use start_j_per_cm2.
        - Every 2 sessions, multiply the current dose by (1 + escalation_fraction).
        - Never exceed cap_j_per_cm2.
        - Once the cap is reached, all remaining sessions use cap_j_per_cm2.

    PROVISIONAL until verified from the primary source.
    """
    doses = []
    current_dose = start_j_per_cm2

    for session_index in range(1, n_sessions + 1):
        dose = min(current_dose, cap_j_per_cm2)
        doses.append(dose)

        if session_index % every_n_sessions == 0:
            current_dose = current_dose * (1.0 + escalation_fraction)
            if current_dose > cap_j_per_cm2:
                current_dose = cap_j_per_cm2

    return doses


# TL-01 absolute physical dose -> CIE erythemal SED.
# Rodriguez-Granados 2017, Waldmann UV7002 with 21x F79/120W-TL01 tubes:
#   absolute irradiance 4.73 mW/cm2, erythemal-weighted 0.26 mW/cm2
#   ratio 0.055 -> 55 mJ/cm2 erythemal per J/cm2 absolute
#   1 SED = 100 J/m2 = 10 mJ/cm2 erythemal  =>  1 J/cm2 absolute = 5.5 SED
# See data/spectra/tl01_conversion_factor.md for the full derivation.
#
# CAVEAT (Lloyd 2004): TL-01 calibration factors vary +/-60% (up to 2.7x)
# across expert medical-physics departments, because TL-01's narrow 311 nm
# peak defeats broadband meter filters. This factor is cabinet-specific and
# is a PRIOR, not a universal constant. Treat any TL-01 conclusion as
# carrying at least a 2x dose uncertainty.
TL01_J_PER_CM2_TO_ERYTHEMAL_SED: float = 5.5

# Bogh 2012 reports mean cumulative 9.0 J/cm2 over the 18-session course.
# The stated escalation rule (+15% every 2nd session from 0.36, capped 1.20)
# integrates to 12.09 J/cm2, which does NOT reproduce the reported total --
# the two published facts are mutually inconsistent (see ASSUMPTION_REGISTER
# A3). The cumulative dose is the better-constrained quantity (it is a
# directly reported trial mean, not a protocol rule subject to per-patient
# MED titration), so we rescale the escalation shape to match it.
BOGH2012_REPORTED_CUMULATIVE_J_PER_CM2: float = 9.0


def bogh2012_session_doses_sed(
    n_sessions: int = 18,
    normalize_to_reported_cumulative: bool = True,
) -> list[float]:
    """Bogh 2012 per-session doses in CIE erythemal SED.

    The ONLY correct entry point for driving the model with the Bogh 2012
    protocol. `simulate()` expects erythemal SED, and E_source then
    converts erythemal SED -> vitamin-D-effective dose. Passing the raw
    J/cm2 numbers straight in as SED (as earlier validation code did)
    understates the dose by the 5.5x factor above.
    """
    doses_j = bogh2012_session_doses(n_sessions=n_sessions)
    if normalize_to_reported_cumulative:
        total = sum(doses_j)
        if total > 0:
            scale = BOGH2012_REPORTED_CUMULATIVE_J_PER_CM2 / total
            doses_j = [d * scale for d in doses_j]
    return [d * TL01_J_PER_CM2_TO_ERYTHEMAL_SED for d in doses_j]
