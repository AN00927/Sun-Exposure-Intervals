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

    This rule is PROVISIONAL until verified from the primary source.
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
