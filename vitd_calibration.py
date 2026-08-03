"""
vitd_calibration.py
====================
Calibration checks against published data. These are CALIBRATION checks, not
independent validation, wherever a parameter was derived from the same source
being checked -- each function's own docstring says which.

Two checks live here:
  - derive_K_ATTEN / run(): Matsuoka 1990 K_ATTEN deconvolution.
  - young2020_calibration_check(): Young 2020 FST II:VI ratio check.

Run as a module for the Matsuoka report: `python3 -m vitd_calibration`.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from vitd_model import ModelParams
from vitd_physiology import hydroxylation_efficiency


# ===========================================================================
# Matsuoka 1990 -- K_ATTEN derivation
# ===========================================================================
#
# Source (IN THE REPO, abstract):
#     Papers_and_Datasets/
#       Matsuoka 1990 JLabClinMed PMID2165512 - ABSTRACT ONLY (has the tanning data).txt
#     Matsuoka LY, Wortsman J, Hollis BW.
#     "Suntanning and cutaneous synthesis of vitamin D3."
#     J Lab Clin Med. 1990 Jul;116(1):87-90. PMID 2165512.
#
# Design: n=7 healthy subjects, WITHIN-SUBJECT. Whole-body UVB in a phototherapy
# unit, same dose both times. Blood immediately before and 24 h after.
#   midwinter (untanned): serum D3  1.7 +/- 0.4 -> 11.0 +/- 1.5 ng/ml (p<0.001)
#   midsummer (tanned):   serum D3  9.6 +/- 2.8 -> 14.3 +/- 4.1 ng/ml (p>0.1, ns)
#   baseline 25(OH)D:     22.5 +/- 2.9 (untanned) vs 36.9 +/- 4.7 (tanned), p<0.02
#
# The file was previously named "(NO PAPER FOUND)", which was true of the full
# text but badly misleading about the data: every number needed to constrain
# K_ATTEN is in the abstract, and has been in this repository all along. Earlier
# docs claimed the paper was "missing from the repo" and listed it as an
# interlibrary-loan request. That was wrong. Before declaring any paper missing,
# check Papers_and_Datasets/ -- filenames there are unreliable.
#
# WHY THE FAMOUS "51%" IS NOT K_ATTEN
# ------------------------------------
# The naive reading -- untanned rise 9.3 vs tanned rise 4.7 ng/ml, so tanning
# blunts synthesis by 49.5% -- is the number cited throughout the literature and
# in this repo's earlier docs. It is confounded.
#
# Matsuoka's own sentence is "resetting of the vitamin D3 synthetic mechanism":
# the tanned subjects ALSO started with 64% higher 25(OH)D (36.9 vs 22.5 ng/ml =
# 92.1 vs 56.2 nmol/L). Since this model (and the underlying physiology) makes
# synthesis baseline-dependent, part of the blunting is D-feedback and only the
# remainder is melanin. Most of the 51% is feedback, not melanin: melanin-only
# attenuation is 7-18% depending on feedback form, implying K_ATTEN 0.34-1.00.
#
# So the abstract constrains the PRODUCT (feedback x melanin), and splitting it
# requires committing to a feedback form. That makes the derived K_ATTEN
# feedback-model-dependent, which this module reports explicitly rather than
# hiding behind a single number.
#
# CAVEATS
# -------
# - n=7, group means only, no individual data, no measured melanin index. The
#   mapping from "midsummer tan" to the model's excess_M is an assumption.
# - Serum D3 (the parent compound) is the measured outcome, not 25(OH)D. The
#   model's synthesis term is in 25(OH)D units, so this assumes proportionality.
# - Seasonal confounds beyond tanning (diet, other sun exposure, weight change).
# - Still the ONLY within-subject tanned-vs-untanned study in existence
#   (Neville 2020 confirms the field lacks replication).

NG_ML_TO_NMOL_L = 2.496

# Abstract values, serum vitamin D3 in ng/ml
UNTANNED_D3_PRE, UNTANNED_D3_POST = 1.7, 11.0
TANNED_D3_PRE, TANNED_D3_POST = 9.6, 14.3
# Baseline 25(OH)D in ng/ml
UNTANNED_25OHD_NG, TANNED_25OHD_NG = 22.5, 36.9

UNTANNED_RISE = UNTANNED_D3_POST - UNTANNED_D3_PRE   # 9.3
TANNED_RISE = TANNED_D3_POST - TANNED_D3_PRE         # 4.7
NAIVE_ATTENUATION = 1.0 - TANNED_RISE / UNTANNED_RISE  # 0.495, the cited "51%"

UNTANNED_25OHD = UNTANNED_25OHD_NG * NG_ML_TO_NMOL_L  # 56.2 nmol/L
TANNED_25OHD = TANNED_25OHD_NG * NG_ML_TO_NMOL_L      # 92.1 nmol/L


def _feedback_factor(D: float, mode: str, params: ModelParams) -> float:
    if mode == "linear":
        return max(0.0, 1.0 - D / params.D_max)
    if mode == "biphasic":
        return hydroxylation_efficiency(D)
    if mode == "saturating":
        k = np.log(2.0) / params.D_half_feedback
        return float(np.exp(-k * D))
    if mode == "none":
        return 1.0
    raise ValueError(f"unknown feedback mode {mode!r}")


def derive_K_ATTEN(feedback_mode: str = "linear",
                   excess_M: float = 0.20,
                   params: ModelParams | None = None) -> dict:
    """Back out K_ATTEN from the Matsuoka data under a given feedback form.

    excess_M is the assumed (M - M_baseline) of a midsummer tan in fair skin.
    Default 0.20 = ModelParams.M_TAN_HEADROOM, i.e. a fully developed tan.
    """
    if params is None:
        params = ModelParams()

    fb_untanned = _feedback_factor(UNTANNED_25OHD, feedback_mode, params)
    fb_tanned = _feedback_factor(TANNED_25OHD, feedback_mode, params)
    if fb_untanned <= 0:
        return {"feedback_mode": feedback_mode, "K_ATTEN": float("nan"),
                "melanin_ratio": float("nan"), "valid": False,
                "note": "feedback factor collapsed to zero at untanned baseline"}

    observed_ratio = TANNED_RISE / UNTANNED_RISE          # 0.505
    feedback_ratio = fb_tanned / fb_untanned
    melanin_ratio = observed_ratio / feedback_ratio       # residual

    valid = 0.0 < melanin_ratio < 1.0
    K = -np.log(melanin_ratio) / excess_M if valid else float("nan")
    return {
        "feedback_mode": feedback_mode,
        "excess_M": excess_M,
        "observed_ratio": observed_ratio,
        "feedback_ratio": feedback_ratio,
        "melanin_ratio": melanin_ratio,
        "melanin_attenuation": 1.0 - melanin_ratio if valid else float("nan"),
        "K_ATTEN": K,
        "valid": valid,
        "note": ("" if valid else
                 "residual implies tanning INCREASED synthesis -- feedback form "
                 "over-explains the blunting"),
    }


def run(verbose: bool = True) -> dict:
    """Report the K_ATTEN range implied by Matsuoka across feedback forms."""
    params = ModelParams()
    results = [derive_K_ATTEN(m, 0.20, params)
               for m in ("linear", "biphasic", "saturating", "none")]
    valid_Ks = [r["K_ATTEN"] for r in results
                if r["valid"] and r["feedback_mode"] != "none"]

    current = params.K_ATTEN
    consistent = bool(valid_Ks) and (min(valid_Ks) <= current <= max(valid_Ks))

    if verbose:
        print("[Matsuoka 1990] K_ATTEN derivation from PMID 2165512 (abstract, n=7)")
        print(f"  Untanned D3 rise {UNTANNED_RISE:.1f} ng/ml, "
              f"tanned {TANNED_RISE:.1f} ng/ml")
        print(f"  Naive attenuation = {NAIVE_ATTENUATION:.1%}  <- the widely cited '51%'")
        print(f"  BUT baseline 25(OH)D differed: {UNTANNED_25OHD:.1f} vs "
              f"{TANNED_25OHD:.1f} nmol/L, so part of the blunting is FEEDBACK.")
        print()
        print(f"  {'feedback form':16} {'melanin resid':>14} {'melanin atten':>14} {'K_ATTEN':>9}")
        for r in results:
            k = f"{r['K_ATTEN']:.2f}" if r["valid"] else "n/a"
            ma = f"{r['melanin_attenuation']:.1%}" if r["valid"] else "n/a"
            print(f"  {r['feedback_mode']:16} {r['melanin_ratio']:14.3f} "
                  f"{ma:>14} {k:>9}")
        print()
        if valid_Ks:
            print(f"  Implied K_ATTEN range (excluding 'none'): "
                  f"{min(valid_Ks):.2f} - {max(valid_Ks):.2f}")
        print(f"  Model currently uses K_ATTEN = {current}")
        print(f"  {'CONSISTENT' if consistent else 'INCONSISTENT'}: current value "
              f"{'lies within' if consistent else 'lies OUTSIDE'} the implied range")
        if not consistent:
            print("  -> K_ATTEN=0.37 was derived from Young 2020's FST II:VI")
            print("     CONSTITUTIVE ratio, which is a different quantity from")
            print("     Matsuoka's within-subject FACULTATIVE (tanning) effect.")
            print("     Note sensitivity analysis shows K_ATTEN swings steady-state")
            print("     D by only ~0.3% over 0-2.0, so this changes little in")
            print("     practice -- but the citation should match the number.")
        print("  Caveats: n=7, group means, serum D3 not 25(OH)D, excess_M assumed,")
        print("  seasonal confounds. Only within-subject study in existence.")

    return {
        "results": results,
        "implied_range": (min(valid_Ks), max(valid_Ks)) if valid_Ks else None,
        "current_K_ATTEN": current,
        "consistent": consistent,
        "naive_attenuation": NAIVE_ATTENUATION,
    }


# ===========================================================================
# Young 2020 -- FST II:VI calibration check
# ===========================================================================
#
# STATUS: CALIBRATION_CHECK_ONLY -- not independent validation.

def young2020_calibration_check():
    """
    Run the Young 2020 FST II:VI calibration check.

    Young 2020 reports FST II:VI vitamin D slope ratios of approximately
    1.29-1.43 (Arimed B: 1.29, TL01: 1.43).

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

    # Young 2020 target range (xlsx per-person fit range)
    target_low = 1.29
    target_high = 1.50

    if target_low <= ratio <= target_high:
        result_status = "CALIBRATION_CHECK_PASS"
    else:
        result_status = "CALIBRATION_CHECK_FAIL"

    # (Previously this substring-matched
    # "verification_status: VERIFIED" against a hard-coded per-study YAML path;
    # merging the per-study files without this fix would have made the check
    # silently degrade to UNVERIFIED regardless of the actual status, since a
    # naive substring match against a missing path fails closed with no error.
    # This does a real parse keyed by study_id instead.)
    import yaml

    provenance_path = "data_provenance.yaml"
    source_status = "UNVERIFIED"
    if os.path.exists(provenance_path):
        with open(provenance_path, encoding="utf-8") as f:
            provenance = yaml.safe_load(f)
        study = (provenance or {}).get("studies", {}).get("Young2020", {})
        source_status = study.get("verification_status", "UNVERIFIED")

    if source_status != "VERIFIED":
        result_status = "UNVERIFIED"

    # Write CSV
    os.makedirs("tables/debug", exist_ok=True)
    csv_path = os.path.join("tables/debug", "young2020_check.csv")

    import subprocess
    from datetime import datetime, timezone

    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__)) or "."
    ).decode().strip()

    timestamp = datetime.now(timezone.utc).isoformat()

    with open(csv_path, "w", encoding="utf-8") as f:
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


def alahouhala_reversal_check():
    """Verify biphasic mode reproduces Finland UV reversal at high baseline.
    
    Targets (approximate from Ala-Houhala Finland vernal study):
      Low baseline D0 ≈ 37 nmol/L: predict delta 25(OH)D > 0 after 8 weeks
      High baseline D0 ≈ 116 nmol/L: predict delta 25(OH)D < 0 after 8 weeks
    
    Protocol proxy: 3x/week spring solar UV in Finland (~1 SED/session),
    BSA = 0.25 (arms/face typical spring clothing), 56 day duration,
    source = 'solar_broadband'.
    """
    from vitd_model import ModelParams, simulate, build_weekly_schedule, get_E_source
    
    params = ModelParams(feedback_mode='biphasic')
    schedule = build_weekly_schedule(56, 1.0, [0, 2, 4])  # Mon/Wed/Fri
    
    print("\n--- Ala-Houhala Finland Reversal Check ---")
    results = {}
    for D0, label in [(37.0, 'low_baseline'), (116.0, 'high_baseline')]:
        r = simulate(fst=2, schedule=schedule, duration_days=56,
                     params=params, D0=D0, exposed_bsa_override=0.25,
                     E_source_override=get_E_source('solar_broadband'))
        delta = r['D'][-1] - D0
        expected_sign = '+' if D0 < 90 else '-'
        actual_sign = '+' if delta > 0 else '-'
        status = 'PASS' if expected_sign == actual_sign else 'FAIL'
        results[label] = {
            "D0": D0,
            "delta": delta,
            "actual_sign": actual_sign,
            "expected_sign": expected_sign,
            "status": status
        }
        print(f"  {label}: D0={D0:.0f}, delta D={delta:.1f}, sign={actual_sign}, expected={expected_sign} [{status}]")
    return results


def deboer2025_vitd_check():
    """Cross-validate synthesis rate A against De Boer 2025 vitamin D response.

    De Boer 2025 protocol:
      31 subjects, 0.8 SED per session, 3x/week for 5 weeks (15 sessions, 8.8 SED cumulative).
      Lower back exposure only (~8% BSA, exposed_bsa_override=0.08).
      Solar simulator (E_source=1.0). Baseline D0 ≈ 64.0 nmol/L.
      Observed response FP I-III: +6.0 nmol/L (64 -> 70 nmol/L, p=0.023).
    """
    from vitd_model import ModelParams, simulate, build_weekly_schedule, get_E_source
    
    params = ModelParams(feedback_mode='linear')
    schedule = build_weekly_schedule(35, 0.8, [0, 2, 4])  # 5 weeks
    
    r = simulate(
        fst=2,
        schedule=schedule,
        duration_days=35,
        params=params,
        D0=64.0,
        exposed_bsa_override=0.08,
        E_source_override=get_E_source('solar_broadband')
    )
    pred_delta = r['D'][-1] - 64.0
    obs_delta = 6.0
    ratio = obs_delta / pred_delta if pred_delta > 0 else float('inf')
    
    print("\n--- De Boer 2025 Vitamin D Check ---")
    print(f"  Observed delta D = {obs_delta:.1f} nmol/L, Predicted delta D = {pred_delta:.2f} nmol/L")
    if ratio > 2.0:
        print(f"  WARNING: Model synthesis rate A underpredicts observed response at small BSA by {ratio:.1f}x. (A=2.353 kept constant per constraints)")
    return {
        "obs_delta": obs_delta,
        "pred_delta": pred_delta,
        "underpredict_ratio": ratio
    }


if __name__ == "__main__":
    run(verbose=True)
    young2020_calibration_check()
    alahouhala_reversal_check()
    deboer2025_vitd_check()

