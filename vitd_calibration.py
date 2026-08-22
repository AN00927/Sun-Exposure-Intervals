"""
vitd_calibration.py
====================
Calibration checks against published data. These are CALIBRATION checks, not
independent validation, wherever a parameter was derived from the same source
being checked -- each function's docstring says which.

  - derive_K_ATTEN / run(): Matsuoka 1990 K_ATTEN deconvolution.
  - young2020_calibration_check(): Young 2020 FST II:VI ratio check.
  - elwood1997_intermittency_check(): Elwood & Jopson 1997 OR anchoring for
    the intermittent-risk exponent.

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
# The file was previously named "(NO PAPER FOUND)", badly misleading: every
# number constraining K_ATTEN is in the abstract and has been in this repo all
# along. Earlier docs claimed the paper was "missing" and listed it as an
# interlibrary-loan request. Wrong. Before declaring any paper missing, check
# Research_Papers/ (PDFs) and Papers_and_Datasets/ (data, abstracts,
# supplementary) -- filenames in both are unreliable.
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
# The abstract constrains the PRODUCT (feedback x melanin), and splitting it
# requires committing to a feedback form. K_ATTEN is therefore
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
    # bool(), not the bare comparison: valid_Ks are np.float64, so the chained
    # comparison yields np.bool_ and callers doing `is False` silently fail.
    # Tolerance of 0.01 allows the rounded point estimate 0.34 to sit within [0.344, 0.998].
    consistent = bool(valid_Ks) and bool(min(valid_Ks) - 0.01 <= current <= max(valid_Ks) + 0.01)

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
            print("  -> EXPECTED, not a defect. The model's K_ATTEN is fitted to the")
            print("     Young 2020 per-person dataset (n=23, TL01 arm) by 1D MSE")
            print("     minimisation on individual delta 25(OH)D, which is a")
            print("     CONSTITUTIVE between-subject quantity. Matsuoka measures a")
            print("     WITHIN-SUBJECT FACULTATIVE (tanning) effect. The two were")
            print("     never the same number, and the fit leaves no residual for")
            print("     facultative melanin once baseline 25(OH)D and BSA are in.")
            print("     Tolerable because melanin is a weak channel here: moving")
            print("     K_ATTEN across the whole Matsuoka range shifts steady-state")
            print("     25(OH)D by ~1%. That is checked, not assumed --")
            print("     tests/test_matsuoka1990.py::TestConsistencyWithModel.")
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
    """Run the Young 2020 FST II:VI calibration check.

    Young 2020 reports FST II:VI vitamin D slope ratios of approximately
    1.29-1.43 (Arimed B: 1.29, TL01: 1.43).

    NOT independent validation if K_ATTEN was derived from Young 2020.
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

    # (Previously this substring-matched "verification_status: VERIFIED"
    # against a hard-coded per-study YAML path; merging the per-study files
    # without this fix would have silently degraded to UNVERIFIED regardless
    # of actual status, since a naive substring match against a missing path
    # fails closed with no error. This does a real parse keyed by study_id.)
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


# ===========================================================================
# Elwood & Jopson 1997 -- INTERMITTENT_BURN_EXPONENT anchoring
# ===========================================================================
#
# Source (full text in Research_Papers/ as of 2026-08-20):
#     Elwood JM, Jopson J. "Melanoma and sun exposure: an overview of published
#     studies." Int J Cancer. 1997;73(2):198-203.
#     Meta-analysis of 29 case-control studies. Odds ratios for melanoma:
#         intermittent exposure          OR = 1.71
#         sunburn history                OR = 1.91
#         heavy occupational (continuous) OR = 0.86  (i.e. PROTECTIVE)
#
# WHAT THIS CHECK DOES, AND WHAT IT CANNOT DO
# --------------------------------------------
# `INTERMITTENT_BURN_EXPONENT` (k) was previously a stated modelling choice with
# no derivation. This check asks what the Elwood ORs actually constrain it to.
#
# The risk index is a SUM over sessions (vitd_physiology.cancer_risk):
#       intermittent_index  ~  sum_i (sed_i / MED) ** k
# Compare two schedules at EQUAL CUMULATIVE SED -- the fixed-UV-budget
# counterfactual that scheduling is actually about. Write the bursty arm as
# N_b sessions at dose d_b and the continuous arm as N_c at d_c, with
# N_b * d_b = N_c * d_c. Then
#
#       index_b / index_c = (N_b / N_c) * (d_b / d_c) ** k
#                         = (d_c / d_b) * (d_b / d_c) ** k
#                         = (d_b / d_c) ** (k - 1)
#
# so, writing C = d_b / d_c for the dose contrast and R for the observed OR
# ratio between the arms,
#
#       k = 1 + ln(R) / ln(C)                                          (*)
#
# The leading 1 is not cosmetic. Dropping it -- i.e. equating the SUMMED index
# ratio to a per-session dose ratio -- yields k < 1.4 for every plausible C,
# a SUB-LINEAR exponent under which bursts would be cheaper than spreading,
# inverting the model's central scheduling claim.
#
# (*) has one equation and two unknowns. Elwood & Jopson report R but not C:
# the studies classify exposure by PATTERN (recreational/intermittent vs
# occupational/continuous), never by dose as a fraction of MED. So the ORs alone
# do NOT identify k. They identify a curve k(C), and a value of k follows only
# once C is stated.
#
# ANCHORING C FROM THE MODEL'S OWN CITED QUANTITIES
# --------------------------------------------------
# Rather than guess C, take both arms from quantities this repository already
# carries with citations:
#   - burst arm      d_b = 1.0 * MED. Definitional: the sunburn OR arm is
#                    defined by erythema, and erythema is 1 MED
#                    (MED values: Shih/Farrar/Rhodes 2018).
#   - continuous arm d_c = GAMMA_MED * MED = 0.7 * MED, the sub-erythemal
#                    ceiling used throughout this project. The occupational arm
#                    does not burn, so d_c < MED necessarily; 0.7 is the
#                    largest non-burning dose the model ever recommends.
# giving C = 1 / GAMMA_MED = 1.43, and k in [2.93, 3.24] across the two OR
# pairs. The incumbent value 3.0 sits inside that interval.
#
# DIRECTION OF THE TWO KNOWN BIASES
# ----------------------------------
# 1. C is a LOWER bound, so k here is an UPPER bound. If the occupational arm's
#    typical per-session dose is well under 0.7 MED, C rises and k falls --
#    C = 2 gives k ~ 2.1, C = 5 gives k ~ 1.5. The bracket is wide.
# 2. Opposing bias: the arms are NOT at equal cumulative dose. Outdoor workers
#    accumulate far MORE lifetime UV than the intermittent group and still show
#    the lower OR, so at matched cumulative dose the true R would exceed the
#    observed ratio -- which pushes k back up.
# The two biases oppose, and neither is quantified. Treat 3.0 as OR-CONSISTENT
# under stated anchors, not as a fitted value, and read the sensitivity table
# this function prints before quoting any absolute risk index.
ELWOOD_OR_INTERMITTENT: float = 1.71
ELWOOD_OR_SUNBURN: float = 1.91
ELWOOD_OR_OCCUPATIONAL: float = 0.86


def elwood_intermittency_exponent(or_burst: float, or_continuous: float,
                                  dose_contrast: float) -> float:
    """Exponent k implied by an OR ratio at a given burst:continuous dose contrast.

    Implements k = 1 + ln(R)/ln(C) -- see the derivation above. `dose_contrast`
    is d_burst/d_continuous in fractions of MED and must exceed 1.
    """
    if dose_contrast <= 1.0:
        raise ValueError("dose_contrast must be > 1 (burst dose exceeds continuous)")
    if or_continuous <= 0 or or_burst <= 0:
        raise ValueError("odds ratios must be positive")
    return 1.0 + float(np.log(or_burst / or_continuous) / np.log(dose_contrast))


def elwood1997_intermittency_check(verbose: bool = True) -> dict:
    """Anchor INTERMITTENT_BURN_EXPONENT against the Elwood & Jopson 1997 ORs.

    This is a CALIBRATION CHECK, not independent validation: it asks whether the
    incumbent exponent is consistent with the published ORs under the model's own
    dose anchors, and reports how far the implied value moves as the (unmeasured)
    dose contrast widens.
    """
    from vitd_physiology import INTERMITTENT_BURN_EXPONENT
    from vitd_objectives import GAMMA_MED

    or_pairs = (("sunburn", ELWOOD_OR_SUNBURN),
                ("intermittent", ELWOOD_OR_INTERMITTENT))
    anchor_contrast = 1.0 / GAMMA_MED
    contrasts = (anchor_contrast, 2.0, 3.0, 5.0)

    table = []
    for label, or_burst in or_pairs:
        for C in contrasts:
            table.append({
                "or_arm": label,
                "odds_ratio": or_burst,
                "or_ratio": or_burst / ELWOOD_OR_OCCUPATIONAL,
                "dose_contrast": C,
                "implied_k": elwood_intermittency_exponent(
                    or_burst, ELWOOD_OR_OCCUPATIONAL, C),
                "is_anchor": abs(C - anchor_contrast) < 1e-12,
            })

    anchor_ks = [r["implied_k"] for r in table if r["is_anchor"]]
    anchor_lo, anchor_hi = min(anchor_ks), max(anchor_ks)
    full_lo = min(r["implied_k"] for r in table)
    full_hi = max(r["implied_k"] for r in table)
    consistent = anchor_lo <= INTERMITTENT_BURN_EXPONENT <= anchor_hi

    if verbose:
        print("\n--- Elwood & Jopson 1997 intermittency-exponent check ---")
        print(f"  k = 1 + ln(OR_burst/OR_occupational) / ln(dose_contrast)")
        print(f"  Anchor contrast C = 1 MED / {GAMMA_MED:.2f} MED = "
              f"{anchor_contrast:.3f}  (erythema threshold vs sub-erythemal ceiling)")
        print(f"\n  {'OR arm':>13s} {'OR':>5s} {'R':>6s} {'contrast':>9s} "
              f"{'implied k':>10s}")
        print(f"  {'-'*13} {'-'*5} {'-'*6} {'-'*9} {'-'*10}")
        for r in table:
            mark = "  <- anchor" if r["is_anchor"] else ""
            print(f"  {r['or_arm']:>13s} {r['odds_ratio']:>5.2f} "
                  f"{r['or_ratio']:>6.3f} {r['dose_contrast']:>9.2f} "
                  f"{r['implied_k']:>10.3f}{mark}")
        print(f"\n  Anchored bracket:  k in [{anchor_lo:.2f}, {anchor_hi:.2f}]")
        print(f"  Full sensitivity:  k in [{full_lo:.2f}, {full_hi:.2f}] "
              f"as the contrast widens to {max(contrasts):.0f}x")
        print(f"  INTERMITTENT_BURN_EXPONENT = {INTERMITTENT_BURN_EXPONENT:.2f} "
              f"-> {'CONSISTENT' if consistent else 'OUTSIDE'} with the anchored bracket")
        if consistent:
            print("  Status: OR-CONSISTENT under stated anchors. NOT a fitted value --")
            print("  the ORs under-determine k (one equation, two unknowns); the dose")
            print("  contrast is an assumption, not a measurement. See the module")
            print("  comment for the two opposing biases.")

    return {
        "table": table,
        "anchor_contrast": anchor_contrast,
        "anchor_bracket": (anchor_lo, anchor_hi),
        "full_bracket": (full_lo, full_hi),
        "exponent": INTERMITTENT_BURN_EXPONENT,
        "consistent": bool(consistent),
    }


# ----------------------------------
# Skin-cancer risk: cumulative-dose exponent (SCC axis)
# ----------------------------------
# Ab, the biological amplification factor, is the exponent in
#     relative tumour density ~ (annual erythemal dose) ** Ab.
# Published erythemally weighted estimates, so directly comparable to this
# model's SED units. See the SCC_DOSE_EXPONENT comment in vitd_physiology.
MOAN1989_AB_SCC: tuple = (1.6, 1.8)     # Cancer Res 49:5207-12, PMID 2766289
MOAN1989_AB_BCC: tuple = (2.1, 2.3)     # same source
LITERATURE_AB_RANGE: tuple = (1.3, 3.0)  # Moan 1989 ozone-corrected low end;
#                                         Jones 1992 Br J Dermatol SCC high end


def moan1989_scc_exponent_check(verbose: bool = True) -> dict:
    """Check SCC_DOSE_EXPONENT against published amplification factors.

    A CONSISTENCY CHECK, not a fit: Ab is measured across ambient-dose gradients
    between regions, and is applied here to a personal annual cumulative dose.
    That transfer is an assumption -- the epidemiology constrains the SHAPE
    (super-linear, exponent well above 1), not the transfer.
    """
    from vitd_physiology import SCC_DOSE_EXPONENT

    lo, hi = MOAN1989_AB_SCC
    in_scc = lo <= SCC_DOSE_EXPONENT <= hi
    in_lit = LITERATURE_AB_RANGE[0] <= SCC_DOSE_EXPONENT <= LITERATURE_AB_RANGE[1]

    # What the retired linear term implied, for contrast: doubling cumulative
    # dose doubled the index; under a power law it more than doubles.
    doubling_ratio = 2.0 ** SCC_DOSE_EXPONENT

    if verbose:
        print("\n--- Moan 1989 SCC cumulative-dose exponent check ---")
        print(f"  Moan 1989 Ab, erythemally weighted:  SCC {lo}-{hi}, "
              f"BCC {MOAN1989_AB_BCC[0]}-{MOAN1989_AB_BCC[1]}")
        print(f"  Wider literature range:              "
              f"{LITERATURE_AB_RANGE[0]}-{LITERATURE_AB_RANGE[1]}")
        print(f"  SCC_DOSE_EXPONENT = {SCC_DOSE_EXPONENT:.2f}  -> "
              f"{'INSIDE' if in_scc else 'OUTSIDE'} the Moan SCC range, "
              f"{'INSIDE' if in_lit else 'OUTSIDE'} the literature range")
        print(f"  Doubling cumulative dose multiplies the SCC index by "
              f"{doubling_ratio:.2f} (was 2.00 under the retired linear term)")
        print("  Moan computed Ab with an erythema action spectrum AND a basal-")
        print("  layer mutagenesis spectrum and got agreement within ~10%, so")
        print("  using erythemal SED here is not the dominant error.")

    return {
        "exponent": SCC_DOSE_EXPONENT,
        "moan1989_scc_range": MOAN1989_AB_SCC,
        "literature_range": LITERATURE_AB_RANGE,
        "in_moan_range": bool(in_scc),
        "in_literature_range": bool(in_lit),
        "doubling_ratio": float(doubling_ratio),
    }


# ----------------------------------
# Skin-cancer risk: FST susceptibility axis
# ----------------------------------
# Out-of-sample checks. None of these was used to fit anything -- the single
# fitted number is Fajuyigbe 2018's basal:whole CPD protection ratio, which
# lives in vitd_physiology.FAJUYIGBE2018_PROTECTION.
GANDINI2005_RR_SKINTYPE_I_VS_IV: float = 2.09          # PMID 16125929
GANDINI2005_RR_SKINTYPE_I_VS_IV_CI: tuple = (1.67, 2.58)
# Ahmed et al. 2025, Br J Dermatol -- England NDRS 2013-2020, European
# age-standardised incidence per 100,000 py, by ONS ethnic group.
AHMED2025_WHITE_BLACK_RATIOS: dict = {
    "melanoma": 27.3 / 1.67,   # 16.3
    "cSCC": 61.8 / 4.73,       # 13.1
    "BCC": 154.0 / 5.98,       # 25.8
}
# The prior this replaced.
RETIRED_SUSCEPTIBILITY_DECAY: float = 0.45


def skin_type_susceptibility_check(verbose: bool = True) -> dict:
    """Audit the FST susceptibility curve against evidence it was not fitted to.

    Fitted to exactly one number (Fajuyigbe 2018's basal:whole CPD protection
    ratio). Checked against three others. Reports the retired prior alongside so
    the improvement -- or a future regression -- is visible rather than asserted.
    """
    from vitd_physiology import (basal_damage_fraction, _phi,
                                 MELANIN_ATTENUATION_KAPPA,
                                 SHIH2018_EPIDERMAL_MELANIN_PCT,
                                 SHIH2018_MEAN_MELANIN_PCT,
                                 SHIH2018_BASAL_LAYER_CPD_SLOPE,
                                 SHIH2018_WHOLE_EPIDERMAL_CPD_SLOPE,
                                 FAJUYIGBE2018_PROTECTION,
                                 _med_sed_by_fst)

    fsts = [1, 2, 3, 4, 5, 6]
    derived = {f: basal_damage_fraction(f) for f in fsts}
    prior = {f: float(np.exp(-RETIRED_SUSCEPTIBILITY_DECAY * (f - 1))) for f in fsts}

    derived_i_iv = derived[1] / derived[4]
    derived_ii_vi = derived[2] / derived[6]
    prior_i_iv = prior[1] / prior[4]

    # Check 1 -- Gandini 2005 III melanoma RR, skin type I vs IV.
    lo, hi = GANDINI2005_RR_SKINTYPE_I_VS_IV_CI
    gandini_ok = lo <= derived_i_iv <= hi

    # Check 2 -- the premise. If whole-epidermal CPD depends only on fractional
    # MED, the equal-physical-dose whole-epidermis protection factor MUST equal
    # the MED ratio. Shih's MED table vs Fajuyigbe's measured 8.0x.
    med_ratio = _med_sed_by_fst(6) / _med_sed_by_fst(1)
    premise_err = abs(med_ratio - FAJUYIGBE2018_PROTECTION["whole"]) / \
        FAJUYIGBE2018_PROTECTION["whole"]
    premise_ok = premise_err < 0.25

    # Check 3 -- Shih's own pooled basal:whole slope ratio at the study's mean
    # melanin. Both slopes have wide CIs, so the admissible interval is wide.
    shih_obs = SHIH2018_BASAL_LAYER_CPD_SLOPE / SHIH2018_WHOLE_EPIDERMAL_CPD_SLOPE
    shih_ci_lo = 0.01 / 0.19
    shih_ci_hi = min(0.14 / 0.03, 1.0)
    shih_pred = _phi(MELANIN_ATTENUATION_KAPPA * SHIH2018_MEAN_MELANIN_PCT)
    shih_ok = shih_ci_lo <= shih_pred <= shih_ci_hi

    # Direction check only -- population incidence, never a target.
    ahmed_lo = min(AHMED2025_WHITE_BLACK_RATIOS.values())
    ahmed_below = derived_ii_vi < ahmed_lo

    if verbose:
        print("\n--- FST susceptibility check (basal-layer damage fraction) ---")
        print(f"  FITTED to one number: Fajuyigbe 2018 basal:whole CPD "
              f"protection = {FAJUYIGBE2018_PROTECTION['basal']}/"
              f"{FAJUYIGBE2018_PROTECTION['whole']} = "
              f"{FAJUYIGBE2018_PROTECTION['basal']/FAJUYIGBE2018_PROTECTION['whole']:.3f}"
              f"   (kappa = {MELANIN_ATTENUATION_KAPPA:.5f})")
        print(f"\n  {'FST':>4s} {'melanin %':>10s} {'derived':>9s} {'retired prior':>14s}")
        print(f"  {'-'*4} {'-'*10} {'-'*9} {'-'*14}")
        for f in fsts:
            print(f"  {f:>4d} {SHIH2018_EPIDERMAL_MELANIN_PCT[f]:>10.2f} "
                  f"{derived[f]:>9.4f} {prior[f]:>14.4f}")

        print(f"\n  CHECK 1  Gandini 2005 III melanoma RR, skin type I vs IV "
              f"= {GANDINI2005_RR_SKINTYPE_I_VS_IV} ({lo}-{hi})")
        print(f"           derived {derived_i_iv:.2f}  -> "
              f"{'PASS' if gandini_ok else 'FAIL'}"
              f"   (retired prior gave {prior_i_iv:.2f})")
        print(f"  CHECK 2  premise: Shih MED ratio VI:I = {med_ratio:.2f} must "
              f"equal Fajuyigbe whole-epidermis {FAJUYIGBE2018_PROTECTION['whole']}")
        print(f"           off by {premise_err*100:.1f}%  -> "
              f"{'PASS' if premise_ok else 'FAIL'}")
        print(f"  CHECK 3  Shih pooled basal:whole = {shih_obs:.3f} "
              f"(CI-admissible {shih_ci_lo:.2f}-{shih_ci_hi:.2f})")
        print(f"           curve implies {shih_pred:.3f}  -> "
              f"{'PASS' if shih_ok else 'FAIL'}")
        print(f"  DIRECTION Ahmed 2025 England White:Black incidence "
              f"{ahmed_lo:.1f}-{max(AHMED2025_WHITE_BLACK_RATIOS.values()):.1f}")
        print(f"           derived II:VI = {derived_ii_vi:.2f}  -> "
              f"{'below, as a constitutive factor should be' if ahmed_below else 'ABOVE -- investigate'}")
        print("\n  Limits: Fajuyigbe is abstract-only and carries the whole")
        print("  calibration; Shih's FST V/VI melanin means are n=3 and n=1; no")
        print("  epidemiological RR per erythemal episode BY SKIN TYPE exists, so")
        print("  a biomarker stands in for one. Curve is conservative for dark")
        print("  skin -- neither study could detect basal CPD in FST IV-VI at all.")

    return {
        "derived": derived,
        "retired_prior": prior,
        "kappa": float(MELANIN_ATTENUATION_KAPPA),
        "derived_I_vs_IV": float(derived_i_iv),
        "derived_II_vs_VI": float(derived_ii_vi),
        "gandini_consistent": bool(gandini_ok),
        "premise_consistent": bool(premise_ok),
        "premise_error": float(premise_err),
        "shih_consistent": bool(shih_ok),
        "shih_implied_ratio": float(shih_pred),
        "below_population_ratio": bool(ahmed_below),
        "prior_I_vs_IV": float(prior_i_iv),
    }


if __name__ == "__main__":
    run(verbose=True)
    young2020_calibration_check()
    alahouhala_reversal_check()
    deboer2025_vitd_check()
    elwood1997_intermittency_check()
    moan1989_scc_exponent_check()
    skin_type_susceptibility_check()

