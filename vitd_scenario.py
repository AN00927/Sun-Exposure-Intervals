"""
vitd_scenario.py
================
The FROZEN REFERENCE SCENARIO. One fixed set of exposure conditions, used for
every day of every schedule in the core pipeline.

WHY THIS EXISTS
---------------
This module replaces the location-based dosimetry chain (solar position ->
ambient UV index -> posture/site geometry -> exposed BSA), retired to
`exploratory/` on 2026-08-21. That chain multiplied four independent
uncertainties into one number which then entered `ln(1 + UV * E_source)`, and
no *product* of the chain had ever been validated against measured personal
dosimetry. Its individual links were mostly fine; the stack was not defensible.

See `docs/REPO_SCOPE_AUDIT.md` for the full argument, and
`exploratory/README.md` for what was retired and why.

WHAT THIS BUYS, AND WHAT IT COSTS
---------------------------------
BUYS: reproducibility (no network call, no date dependence, no posture
assumption), and a much smaller surface to defend.

COSTS: the model can no longer answer "how long should I stand outside in
Manchester on 15 March". It answers "how long under reference conditions".
That is a real loss of capability and is stated as a limitation, not hidden.

THE VALUES, AND WHY EACH ONE REQUIRED NO RECALIBRATION
------------------------------------------------------
Every constant below was ALREADY the reference value somewhere in the
codebase. Freezing them changes no calibrated parameter and moves no
validation check. That was the selection criterion.

  UV_INDEX = 6.0
      Already the reference condition printed in the RESULTS.md section 0
      header ("Reference conditions for the minutes column: UV index 6.0").

  GEOMETRY_FACTOR = 0.35
      Already `vitd_objectives.PERSONAL_DOSE_GEOMETRY_FACTOR` (Cockell 2001
      scalar), which is the fallback the whole minutes machinery uses when no
      posture or activity is supplied. Imported from there rather than
      restated, so the two cannot drift apart.

  EXPOSED_BSA = vitd_model.BSA_REF
      Already the model's reference BSA, at which `bsa_factor(bsa) == 1.0`
      exactly. Every existing calibration (A, K_const) was performed at this
      value, so freezing here leaves all of them untouched.

      THE "USUAL AMOUNT OF BODY PARTS" QUESTION. Computing the retired
      `vitd_exposure.CLOTHING_PRESETS` at standing posture gives
      `short_sleeves` = 0.165 (face, neck, forearms, hands bare) and
      `shorts_and_tshirt` = 0.495 (adds both legs). Their midpoint is 0.33 --
      within 6% of BSA_REF. So "a t-shirt and shorts, with partial coverage"
      is already what the model's reference BSA encodes. The plain-English
      description below is therefore a description of the existing value, not
      a new choice.

  SED_PER_MINUTE
      DERIVED, not asserted:
          SED_PER_MIN_PER_UVI * UV_INDEX * GEOMETRY_FACTOR
      `SED_PER_MIN_PER_UVI` is owned by `vitd_objectives.py` and follows from
      the WHO/WMO UVI definition (1 UVI = 25 mW/m^2 erythemal) and the SED
      definition (1 SED = 100 J/m^2 erythemal). It is not a free parameter.

THE LIMITATION, FOR THE PAPER
-----------------------------
Reproduce `LIMITATION_STATEMENT` below verbatim in any write-up. Real ambient
UV varies with latitude, season, time of day, altitude, cloud and ozone; real
personal dose varies with posture and clothing. None of that is modelled here.
Every duration this repository reports is a duration UNDER REFERENCE
CONDITIONS, not a calendar recommendation for a specific place.

The returned SED is ERYTHEMAL. The spectral step to vitamin-D-effective dose
is `E_source` (`vitd_model.SOURCE_CALIBRATION_TABLE`), applied downstream and
deliberately not folded in here.
"""

from vitd_model import BSA_REF
from vitd_objectives import PERSONAL_DOSE_GEOMETRY_FACTOR, SED_PER_MIN_PER_UVI

# ---------------------------------------------------------------------------
# The frozen scenario
# ---------------------------------------------------------------------------

UV_INDEX: float = 6.0
GEOMETRY_FACTOR: float = PERSONAL_DOSE_GEOMETRY_FACTOR
EXPOSED_BSA: float = BSA_REF

#: Erythemal SED delivered per minute of exposure under the frozen scenario.
#: Derived from the three constants above; never edit this directly.
SED_PER_MINUTE: float = SED_PER_MIN_PER_UVI * UV_INDEX * GEOMETRY_FACTOR

#: Plain-English description of EXPOSED_BSA, for report headers. See the
#: module docstring for why this describes the existing BSA_REF rather than
#: choosing a new value.
CLOTHING_DESCRIPTION: str = (
    "t-shirt and shorts with partial coverage (face, neck, forearms, hands "
    "and part of the legs bare)"
)

LIMITATION_STATEMENT: str = (
    "All results assume a fixed ambient UV index of {uvi:g}, a "
    "horizontal-to-personal geometry factor of {geom:g}, and {bsa:.0%} exposed "
    "body surface area ({clothing}), held constant on every exposure day. Real "
    "ambient UV varies with latitude, season, time of day, altitude, cloud and "
    "ozone, and real personal dose varies with posture and clothing. We do not "
    "model that variation. Every duration reported here should therefore be "
    "read as a duration under these reference conditions, not as a calendar "
    "recommendation for a specific place or date."
).format(uvi=UV_INDEX, geom=GEOMETRY_FACTOR, bsa=EXPOSED_BSA,
         clothing=CLOTHING_DESCRIPTION)


def sed_for_minutes(minutes: float) -> float:
    """Erythemal SED delivered by `minutes` of exposure under the frozen
    scenario. Negative durations clamp to zero."""
    return max(0.0, float(minutes)) * SED_PER_MINUTE


def minutes_for_sed(sed: float) -> float:
    """Minutes of exposure needed to deliver `sed` erythemal SED under the
    frozen scenario. Negative doses clamp to zero."""
    return max(0.0, float(sed)) / SED_PER_MINUTE


def describe() -> str:
    """One-block human-readable summary, for report headers."""
    return (
        "FROZEN REFERENCE SCENARIO\n"
        f"  ambient UV index      : {UV_INDEX:g}\n"
        f"  geometry factor       : {GEOMETRY_FACTOR:g} "
        "(horizontal ambient -> personal, Cockell 2001)\n"
        f"  exposed BSA           : {EXPOSED_BSA:.0%} "
        f"({CLOTHING_DESCRIPTION})\n"
        f"  erythemal SED/minute  : {SED_PER_MINUTE:.4f} (derived)\n"
        "  held constant on every exposure day; see vitd_scenario.py"
    )


if __name__ == "__main__":
    print(describe())
    print()
    print(LIMITATION_STATEMENT)
