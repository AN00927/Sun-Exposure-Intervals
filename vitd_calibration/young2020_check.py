"""
Young 2020 calibration check.

STATUS: CALIBRATION_CHECK_ONLY — not independent validation.

This script checks whether the model's FST II:VI synthesis ratio is
consistent with Young 2020 observations. If K_ATTEN was derived from
Young 2020, this is a calibration check, not independent validation.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone

import numpy as np


def young2020_calibration_check():
    """
    Run the Young 2020 FST II:VI calibration check.

    Young 2020 reports FST II:VI vitamin D slope ratios of approximately
    1.29-1.43 (Arimed B: 1.29, TL01: 1.43).

    If the model's ratio falls within this range, it passes the calibration check.
    However, this is NOT independent validation if K_ATTEN was derived from Young 2020.
    """
    try:
        from vitd_model import ModelParams, simulate, build_weekly_schedule
    except ImportError:
        print("ERROR: vitd_model not found. Cannot run calibration check.")
        return

    params = ModelParams()

    # Standard test: 1 SED, 3x/week (Mon,Wed,Fri) for 4 weeks, untanned
    schedule = build_weekly_schedule(
        duration_days=28,
        sed_per_session=1.0,
        days_of_week=[0, 2, 4]  # Mon=0, Wed=2, Fri=4
    )

    results = {}
    for fst in [2, 3, 4, 5, 6]:
        result = simulate(
            fst=fst,
            schedule=schedule,
            duration_days=30,
            params=params,
            D0=25.0,
            dose_framing="fixed_per_session_SED",
        )
        final_D = result["D"][-1]
        delta = final_D - 25.0
        results[fst] = {"final_D": final_D, "delta": delta}

    # FST II:VI ratio
    fst_ii_delta = results[2]["delta"]
    fst_vi_delta = results[6]["delta"]

    if fst_vi_delta > 0:
        ratio = fst_ii_delta / fst_vi_delta
    else:
        ratio = float("inf")

    # Young 2020 target range
    target_low = 1.29
    target_high = 1.43

    if target_low <= ratio <= target_high:
        result_status = "CALIBRATION_CHECK_PASS"
    else:
        result_status = "CALIBRATION_CHECK_FAIL"

    # Check provenance
    provenance_path = os.path.join("data_provenance", "young_2020.yaml")
    source_status = "UNVERIFIED"
    if os.path.exists(provenance_path):
        with open(provenance_path) as f:
            content = f.read()
            if "verification_status: VERIFIED" in content:
                source_status = "VERIFIED"

    if source_status != "VERIFIED":
        result_status = "UNVERIFIED"

    # Write CSV
    os.makedirs("tables/debug", exist_ok=True)
    csv_path = os.path.join("tables/debug", "young2020_check.csv")

    import subprocess
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.dirname(__file__)) or "."
    ).decode().strip()

    timestamp = datetime.now(timezone.utc).isoformat()

    with open(csv_path, "w") as f:
        f.write("git_hash,parameter_hash,model_version,timestamp,study_id,metric,predicted_value,target_low,target_high,source_status,result,notes\n")
        f.write(f"{git_hash},NA,debug_rev1,{timestamp},Young2020,FST_II_VI_ratio,{ratio:.4f},{target_low},{target_high},{source_status},{result_status},Calibration check only - not independent validation\n")

    print(f"Young 2020 calibration check: {result_status}")
    print(f"  FST II:VI ratio = {ratio:.4f} (target: {target_low}-{target_high})")
    print(f"  Source status: {source_status}")
    print(f"  Output: {csv_path}")

    return {
        "ratio": ratio,
        "target_low": target_low,
        "target_high": target_high,
        "result": result_status,
        "source_status": source_status,
    }


if __name__ == "__main__":
    young2020_calibration_check()