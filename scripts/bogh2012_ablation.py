"""
Bogh 2012 Ablation Study — Phase 3 Debug Run

Runs 21 ablation configurations (A00-A20) against the Bogh 2012 NB-UVB protocol.
Originally used to isolate which model factors contributed to a prediction gap;
that gap was a dose-unit error (absolute J/cm2 passed as erythemal SED, and 35%
BSA used for a whole-body protocol), not a physiological one. Once units and
BSA are corrected, the default SOURCE_CALIBRATION_TABLE value (tl01_nb_uvb=2.44)
gives error 0.2 nmol/L with no search. Retained as a sensitivity tool; per-config
error figures from historical runs are not reproducible and should not be cited.

STATUS: PROVISIONAL — requires human review before calibration.
"""

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vitd_model import ModelParams, simulate, build_pulse_schedule
from vitd_protocols import bogh2012_session_doses
from vitd_dose_conversion import convert_bogh_dose_to_effective_units


GIT_HASH = "620e428"
MODEL_VERSION = "debug_rev1"
STUDY_ID = "Bogh2012"
BASELINE_D = 19.2
OBSERVED_DELTA = 55.8
FST = 3
DURATION_DAYS = 42

SESSION_DAYS = [
    1, 2, 4, 5, 8, 9, 11, 12,
    15, 16, 18, 19, 22, 23, 25, 26,
    29, 30,
]


def build_bogh2012_schedule():
    """Return list of (day, J/cm2) tuples for the 18-session Bogh 2012 protocol."""
    doses_j = bogh2012_session_doses()
    schedule = list(zip(SESSION_DAYS, doses_j))
    return schedule, doses_j


def sed_from_j_per_cm2(j_per_cm2):
    """Provisional conversion: 1 MED ≈ 0.025 J/cm2 for FST III (NB-UVB 311 nm).

    This is a rough placeholder. Erythemal SED = J/cm2 / MED_FST3.
    """
    med_fst3 = 0.025
    return j_per_cm2 / med_fst3


def convert_schedule_to_sed(schedule, spectrum_name, action_spectrum_name):
    """Convert a (day, J/cm²) schedule to (day, SED) using spectral weighting.

    Returns (converted_schedule, conversion_info) where conversion_info
    contains per-session details and overall status.
    """
    converted = []
    session_info = []
    all_ok = True

    for day, dose_j in schedule:
        result = convert_bogh_dose_to_effective_units(
            dose_j, spectrum_name, action_spectrum_name
        )
        if result["status"] != "OK":
            all_ok = False
        converted.append((day, result["sed_equivalent"]))
        session_info.append({
            "day": day,
            "physical_j_per_cm2": dose_j,
            "sed": result["sed_equivalent"],
            "status": result["status"],
        })

    overall_status = "OK" if all_ok else "BLOCKED_MISSING_SPECTRUM"
    return converted, {"sessions": session_info, "status": overall_status}


def run_config(config_id, config_name, description, schedule, params,
               overrides, dose_conversion="none", bsa_form="none",
               bsa_fraction=0.35, dmax=150.0, feedback=True,
               half_life_days=None, tanning=True, sc=True,
               E_source=1.0, A_synthesis=2.353, notes=""):
    """Run a single ablation configuration and return result dict."""
    effective_overrides = {}

    D_max_val = dmax
    if not feedback:
        D_max_val = 99999

    effective_overrides["D_max_override"] = D_max_val
    effective_overrides["E_source_override"] = E_source
    effective_overrides["exposed_bsa_override"] = bsa_fraction

    if not tanning:
        effective_overrides["alpha_override"] = 0.0
    if not sc:
        effective_overrides["alpha_SC_override"] = 0.0

    if half_life_days is not None:
        k_decay = math.log(2) / half_life_days
        effective_overrides["k_decay_M_override"] = k_decay

    result = simulate(
        fst=FST,
        schedule=schedule,
        duration_days=DURATION_DAYS,
        params=params,
        D0=BASELINE_D,
        M0=None,
        SC0=0.0,
        **effective_overrides,
    )

    final_D = float(result["D"][-1])
    predicted_delta = final_D - BASELINE_D

    return {
        "git_hash": GIT_HASH,
        "parameter_hash": "NA",
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": config_id,
        "study_id": STUDY_ID,
        "configuration_name": config_name,
        "changed_factors_json": json.dumps(overrides),
        "dose_conversion": dose_conversion,
        "bsa_form": bsa_form,
        "bsa_fraction": bsa_fraction,
        "dmax_nmol_L": D_max_val if not feedback else dmax,
        "feedback_enabled": feedback,
        "clearance_half_life_days": half_life_days if half_life_days else 197.0,
        "tanning_enabled": tanning,
        "sc_enabled": sc,
        "E_source": E_source,
        "A_synthesis": A_synthesis,
        "baseline_nmol_L": BASELINE_D,
        "predicted_final_nmol_L": round(final_D, 2),
        "predicted_delta_nmol_L": round(predicted_delta, 2),
        "observed_delta_nmol_L": OBSERVED_DELTA,
        "absolute_error": round(abs(predicted_delta - OBSERVED_DELTA), 2),
        "relative_error": "NA",
        "status": "OK",
        "notes": notes,
    }


def main():
    schedule, doses_j = build_bogh2012_schedule()
    params = ModelParams()

    results = []

    base_result = run_config(
        "A00", "base_current",
        "Use current model defaults (E_source=1.0, BSA=0.90, D_max=150, feedback linear, tanning ON, SC ON)",
        schedule, params,
        overrides={"E_source": 1.0, "BSA": 0.90, "D_max": 150, "feedback": "linear", "tanning": True, "SC": True},
        E_source=1.0,
        bsa_fraction=0.90,  # Bogh 2012 used ~90% BSA exposed
        notes="Baseline run with BSA=0.90 for Bogh 2012 study",
    )
    results.append(base_result)

    base_delta = base_result["predicted_delta_nmol_L"]

    schedule_result = run_config(
        "A01", "schedule_only",
        "Use exact 18-session schedule with current model",
        schedule, params,
        overrides={"schedule": "18_session_bogh2012", "BSA": 0.90},
        bsa_fraction=0.90,
        notes="Same as A00 but explicitly using the 18-session schedule",
    )
    results.append(schedule_result)

    a02_sched, a02_conv = convert_schedule_to_sed(
        schedule, "tl01_spectrum.csv", "cie_erythema_action_spectrum.csv"
    )
    a02_result = run_config(
        "A02", "schedule_plus_erythemal_dose",
        "A01 + erythemal SED conversion",
        a02_sched, params,
        overrides={"dose_conversion": "erythemal_SED", "BSA": 0.90},
        dose_conversion="erythemal_SED",
        bsa_fraction=0.90,
        notes="Erythemal SED via TL-01 spectrum + CIE erythema action spectrum",
    )
    if a02_conv["status"] != "OK":
        a02_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a02_result)

    a03_sched, a03_conv = convert_schedule_to_sed(
        schedule, "tl01_spectrum.csv", "vitd_action_spectrum_cie.csv"
    )
    a03_result = run_config(
        "A03", "schedule_plus_vitd_cie_dose",
        "A01 + vitamin-D-effective conversion using CIE action spectrum",
        a03_sched, params,
        overrides={"dose_conversion": "vitd_CIE", "BSA": 0.90},
        dose_conversion="vitd_CIE",
        bsa_fraction=0.90,
        notes="D3-effective SED via TL-01 spectrum + CIE action spectrum",
    )
    if a03_conv["status"] != "OK":
        a03_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a03_result)

    a04_sched, a04_conv = convert_schedule_to_sed(
        schedule, "tl01_spectrum.csv", "vitd_action_spectrum_young2021.csv"
    )
    a04_result = run_config(
        "A04", "schedule_plus_vitd_young_dose",
        "A01 + vitamin-D-effective conversion using Young 2021 action spectrum",
        a04_sched, params,
        overrides={"dose_conversion": "vitd_Young2021", "BSA": 0.90},
        dose_conversion="vitd_Young2021",
        bsa_fraction=0.90,
        notes="D3-effective SED via TL-01 spectrum + Young 2021 action spectrum",
    )
    if a04_conv["status"] != "OK":
        a04_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a04_result)

    # A05 removed - BSA=0.90 is now baseline (was schedule_plus_bsa90_powerlaw)
    
    a05_result = run_config(
        "A05", "schedule_plus_bsa90_log",
        "A01 + provisional Bogh log BSA additive term",
        schedule, params,
        overrides={"exposed_bsa": 0.90, "bsa_form": "log_provisional"},
        bsa_form="log_provisional",
        bsa_fraction=0.90,
        notes="Bogh 2011 equation direct prediction — separate model, not ODE modifier",
    )
    a05_result["status"] = "BLOCKED_UNVERIFIED_BSA_EQUATION"
    results.append(a05_result)

    a06_result = run_config(
        "A06", "schedule_plus_no_feedback",
        "A01 + D_max=99999 (effectively no feedback)",
        schedule, params,
        overrides={"D_max": 99999, "feedback": "disabled"},
        feedback=False,
        dmax=99999,
        notes="Removes homeostatic feedback ceiling",
    )
    results.append(a06_result)

    a07_result = run_config(
        "A07", "schedule_plus_dmax_200",
        "A01 + D_max=200",
        schedule, params,
        overrides={"D_max": 200},
        dmax=200,
        notes="Raises feedback ceiling from 150 to 200 nmol/L",
    )
    results.append(a07_result)

    a08_result = run_config(
        "A08", "schedule_plus_dmax_300",
        "A01 + D_max=300",
        schedule, params,
        overrides={"D_max": 300},
        dmax=300,
        notes="Raises feedback ceiling from 150 to 300 nmol/L",
    )
    results.append(a08_result)

    a09_result = run_config(
        "A09", "schedule_plus_half_life_28",
        "A01 + k_decay overridden for 28-day half-life",
        schedule, params,
        overrides={"half_life_days": 28},
        half_life_days=28,
        notes="Fixed 28-day half-life replaces D-dependent Datta 2017 decay",
    )
    results.append(a09_result)

    a10_result = run_config(
        "A10", "schedule_plus_half_life_42",
        "A01 + k_decay overridden for 42-day half-life",
        schedule, params,
        overrides={"half_life_days": 42},
        half_life_days=42,
        notes="Fixed 42-day half-life (slower decay than Datta 2017 at baseline D)",
    )
    results.append(a10_result)

    a11_result = run_config(
        "A11", "schedule_plus_no_tanning",
        "A01 + alpha=0 (no melanin change)",
        schedule, params,
        overrides={"alpha": 0, "tanning": False},
        tanning=False,
        notes="Melanin remains at baseline throughout",
    )
    results.append(a11_result)

    a12_result = run_config(
        "A12", "schedule_plus_no_sc",
        "A01 + alpha_SC=0 (no SC change)",
        schedule, params,
        overrides={"alpha_SC": 0, "sc": False},
        sc=False,
        notes="Stratum corneum remains at baseline throughout",
    )
    results.append(a12_result)

    a13_result = run_config(
        "A13", "combo_dose_ery_bsa_power",
        "A02 + BSA 0.90",
        a02_sched, params,
        overrides={"dose_conversion": "erythemal_SED", "exposed_bsa": 0.90, "bsa_form": "power_law"},
        dose_conversion="erythemal_SED",
        bsa_form="power_law",
        bsa_fraction=0.90,
        notes="Combines erythemal dose conversion with BSA 0.90",
    )
    if a02_conv["status"] != "OK":
        a13_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a13_result)

    a14_result = run_config(
        "A14", "combo_dose_vitd_cie_bsa_power",
        "A03 + BSA 0.90",
        a03_sched, params,
        overrides={"dose_conversion": "vitd_CIE", "exposed_bsa": 0.90, "bsa_form": "power_law"},
        dose_conversion="vitd_CIE",
        bsa_form="power_law",
        bsa_fraction=0.90,
        notes="Combines CIE dose conversion with BSA 0.90",
    )
    if a03_conv["status"] != "OK":
        a14_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a14_result)

    a15_result = run_config(
        "A15", "combo_dose_vitd_young_bsa_power",
        "A04 + BSA 0.90",
        a04_sched, params,
        overrides={"dose_conversion": "vitd_Young2021", "exposed_bsa": 0.90, "bsa_form": "power_law"},
        dose_conversion="vitd_Young2021",
        bsa_form="power_law",
        bsa_fraction=0.90,
        notes="Combines Young 2021 dose conversion with BSA 0.90",
    )
    if a04_conv["status"] != "OK":
        a15_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a15_result)

    a16_result = run_config(
        "A16", "combo_vitd_cie_bsa_power_no_feedback",
        "A14 + no feedback",
        a03_sched, params,
        overrides={"dose_conversion": "vitd_CIE", "exposed_bsa": 0.90, "bsa_form": "power_law", "D_max": 99999, "feedback": "disabled"},
        dose_conversion="vitd_CIE",
        bsa_form="power_law",
        bsa_fraction=0.90,
        feedback=False,
        dmax=99999,
        notes="CIE dose + BSA 0.90 + no feedback",
    )
    if a03_conv["status"] != "OK":
        a16_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a16_result)

    a17_result = run_config(
        "A17", "combo_vitd_cie_bsa_power_half_life_42",
        "A14 + 42d half-life",
        a03_sched, params,
        overrides={"dose_conversion": "vitd_CIE", "exposed_bsa": 0.90, "bsa_form": "power_law", "half_life_days": 42},
        dose_conversion="vitd_CIE",
        bsa_form="power_law",
        bsa_fraction=0.90,
        half_life_days=42,
        notes="CIE dose + BSA 0.90 + 42d half-life",
    )
    if a03_conv["status"] != "OK":
        a17_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a17_result)

    a18_result = run_config(
        "A18", "combo_vitd_cie_bsa_power_no_tanning_no_sc",
        "A14 + no tanning + no SC",
        a03_sched, params,
        overrides={"dose_conversion": "vitd_CIE", "exposed_bsa": 0.90, "bsa_form": "power_law", "alpha": 0, "alpha_SC": 0, "tanning": False, "sc": False},
        dose_conversion="vitd_CIE",
        bsa_form="power_law",
        bsa_fraction=0.90,
        tanning=False,
        sc=False,
        notes="CIE dose + BSA 0.90 + no tanning + no SC",
    )
    if a03_conv["status"] != "OK":
        a18_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a18_result)

    a19_result = run_config(
        "A19", "combo_vitd_young_bsa_power_no_feedback",
        "A15 + no feedback",
        a04_sched, params,
        overrides={"dose_conversion": "vitd_Young2021", "exposed_bsa": 0.90, "bsa_form": "power_law", "D_max": 99999, "feedback": "disabled"},
        dose_conversion="vitd_Young2021",
        bsa_form="power_law",
        bsa_fraction=0.90,
        feedback=False,
        dmax=99999,
        notes="Young 2021 dose + BSA 0.90 + no feedback",
    )
    if a04_conv["status"] != "OK":
        a19_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a19_result)

    a20_result = run_config(
        "A20", "combo_vitd_young_bsa_power_half_life_42",
        "A15 + 42d half-life",
        a04_sched, params,
        overrides={"dose_conversion": "vitd_Young2021", "exposed_bsa": 0.90, "bsa_form": "power_law", "half_life_days": 42},
        dose_conversion="vitd_Young2021",
        bsa_form="power_law",
        bsa_fraction=0.90,
        half_life_days=42,
        notes="Young 2021 dose + BSA 0.90 + 42d half-life",
    )
    if a04_conv["status"] != "OK":
        a20_result["status"] = "BLOCKED_MISSING_SPECTRUM"
    results.append(a20_result)

    output_dir = Path(__file__).resolve().parent.parent / "tables" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bogh2012_ablation.csv"

    fieldnames = [
        "git_hash", "parameter_hash", "model_version", "timestamp",
        "run_id", "study_id", "configuration_name", "changed_factors_json",
        "dose_conversion", "bsa_form", "bsa_fraction", "dmax_nmol_L",
        "feedback_enabled", "clearance_half_life_days", "tanning_enabled",
        "sc_enabled", "E_source", "A_synthesis", "baseline_nmol_L",
        "predicted_final_nmol_L", "predicted_delta_nmol_L",
        "observed_delta_nmol_L", "absolute_error", "relative_error",
        "status", "notes",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"CSV written to {csv_path}")

    ok_results = [r for r in results if r["status"] == "OK"]
    ok_results_sorted = sorted(ok_results, key=lambda r: r["absolute_error"])

    print("\n=== Ablation Results (non-blocked, sorted by error) ===")
    for r in ok_results_sorted:
        print(f"  {r['run_id']}: {r['configuration_name']} -> "
              f"delta={r['predicted_delta_nmol_L']:.2f}, "
              f"error={r['absolute_error']:.2f}")

    print(f"\nTotal configurations: {len(results)}")
    print(f"OK: {len(ok_results)}")
    print(f"BLOCKED: {len(results) - len(ok_results)}")

    return results


if __name__ == "__main__":
    main()
