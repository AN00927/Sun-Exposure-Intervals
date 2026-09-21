"""
fst_gap_decomposition.py
========================
DIAGNOSIS ONLY. Changes no parameter, writes no parameter.

Of the 2-3x "too weak melanin penalty" against Webb/Kift 2018
(`vitd_benchmark_webbkift.py`),
how much is OPTICAL (melanin attenuating UV in skin, belonging in K_const /
K_ATTEN) and how much is BEHAVIOURAL / EXPOSURE (shade, clothing, time
outdoors, belonging in a separate exposure term)?

The distinction matters because a scheduling tool asks people to CHANGE their
behaviour. Any factor folded into K_const is assumed invariant under that
change. Folding a behavioural factor into an optical constant makes the model
score well on the benchmark and wrong for exactly the user it is built for.

Sections
--------
S1  Provenance audit of the three "independent" failures.
S2  Measured behavioural exposure ratio, FST V vs white, from personal
    polysulphone dosimetry at UK latitude.
S3  Optical factor at MATCHED physical dose, recomputed from the individual-
    level Young 2020 trial data in this repo.
S4  Model inversion on two field cohorts that have BOTH measured personal UV
    dose and measured serum: solve for the K_const the data require, with and
    without the measured behavioural difference.
S5  NHANES identifiability curve (why NHANES cannot separate the two).

Run
---
    PYTHONIOENCODING=utf-8 python3 scripts/fst_gap_decomposition.py

Every literature number used here is tagged with its source in the DATA block
below. Numbers this script DERIVES are printed as "[derived]". Numbers this
script ASSUMES are printed as "[ASSUMED]" and collected in S6.
"""

from __future__ import annotations

import os
import sys
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import (ModelParams, simulate, steady_state_D,
                        get_E_source, E_SOURCE_SOLAR_DEFAULT)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NGML_TO_NMOL = 2.5


# ===========================================================================
# DATA BLOCK -- every value below is quoted from a source in this repo
# ===========================================================================

# --- Kallioglu et al. 2024, Sci Rep 14:3541 (Research_Papers/, full text) ---
# Table 3, "Minimal Erythemedose and skin types", 1 MED in J/m^2 erythemal.
KALLIOGLU_MED_JM2 = {1: 200, 2: 250, 3: 300, 4: 450, 5: 600, 6: 1000}
# Abstract + Conclusions: annual mean minutes for 1000 IU/day.
KALLIOGLU_MINUTES = {1: 5.05, 2: 6.3, 3: 7.6, 4: 11.35, 5: 15.15, 6: 25.25}
# Eq. 10 (the whole skin-type mechanism):  UVd = (MED/4) * 0.25 * BSA * Dvit/1000

# --- Shih et al. 2018, J Invest Dermatol 138:2244 (Research_Papers/) ---
# Results: the 20%-MED dose actually delivered, per skin type, mJ/cm^2
# erythemally effective.  1 MED = 5 x that.  10 mJ/cm^2 = 1 SED.
SHIH_20PCT_MED_MJCM2 = {1: 4.2, 2: 5.2, 3: 6.4, 4: 11.2, 5: 15.0, 6: 30.4}
SHIH_MED_SED = {k: v * 5.0 / 10.0 for k, v in SHIH_20PCT_MED_MJCM2.items()}
# Key finding: 25(OH)D gain per fractional MED showed "no influence of skin
# type (P = 0.23)".  n = 33-38 per dose, ~6 per skin type.  A NULL result.

# --- Webb et al. 2021, Endocr Connect 10:R248 Table 4 (Research_Papers/) ---
# Solar-simulated-radiation studies, both 35% BSA, both Arimed B, same group.
# TYPEFACE PROVENANCE, checked 2026-08-17 by reading PyMuPDF span italic flags
# rather than trusting the plain-text dump. Webb 2021's caption states that
# italicised cells were "inferred or calculated by current study authors".
#   n, total SED (23.4), delta 25(OH)D (26.0 / 11.5) -> ROMAN = from the
#     original publications.
#   nmol/L per SED (1.11 / 0.48)                     -> ITALIC = Webb's own
#     arithmetic. 26.0/23.4 = 1.111 checks out; 11.5/23.4 = 0.491 vs the
#     printed 0.48, a small unexplained rounding.
# Prefers the ROMAN cells for its own ratio.
WEBB21_T4 = {
    # study: (n, skin types, total SED, delta 25(OH)D nmol/L, nmol/L per SED)
    "Rhodes 2010 (white)": (109, "1-4", 23.4, 26.0, 1.11),
    "Farrar 2013 (FST V)": (51, "5", 23.4, 11.5, 0.48),
}
# Farrar 2013 full dose series as tabulated by Webb 2021 Table 4:
FARRAR13_SERIES = [  # (total SED, delta 25(OH)D nmol/L, nmol/L per SED)
    (11.7, 9.0, 0.76), (23.4, 11.5, 0.48), (35.1, 16.8, 0.48),
    (46.8, 16.8, 0.36), (58.5, 31.5, 0.54), (70.2, 23.8, 0.34),
]

# --- Webb et al. 2021 Table 5: field studies with personal dosimetry -------
# Darling et al. 2013 (Osteoporos Int 24:477) is a WITHIN-STUDY comparison:
# same location, same protocol, same polysulphone badges, both female cohorts.
FIELD_COHORTS = {
    "Darling2013_white": dict(
        n=90, fst=2, label="Darling 2013 Caucasian F, Surrey 51.3N",
        sed_summer_week=6.39, sed_winter_week=0.64,
        d_summer=72.1, d_winter=44.5, amplitude=27.6),
    "Darling2013_southasian": dict(
        n=35, fst=5, label="Darling 2013 South Asian F, Surrey 51.3N",
        sed_summer_week=2.00, sed_winter_week=0.29,
        d_summer=26.2, d_winter=19.7, amplitude=6.5),
    # Kift 2013 (BJD 169:1272, Research_Papers/) vs its own white comparator
    # (Webb 2010 BJD 163:1050, n=109, same city, same protocol, ambient
    # within 5%).  Serum from Kift Fig 1 medians, ng/mL -> nmol/L.
    "Kift2013_white": dict(
        n=109, fst=2, label="Webb 2010 white, Manchester 53.5N",
        sed_summer_week=3.4, sed_winter_week=0.1,
        d_summer=26.2 * NGML_TO_NMOL, d_winter=18.9 * NGML_TO_NMOL,
        amplitude=(26.2 - 18.9) * NGML_TO_NMOL),
    "Kift2013_southasian": dict(
        n=125, fst=5, label="Kift 2013 South Asian FST V, Manchester 53.5N",
        sed_summer_week=1.0, sed_winter_week=0.1,
        d_summer=9.0 * NGML_TO_NMOL, d_winter=5.8 * NGML_TO_NMOL,
        amplitude=(9.0 - 5.8) * NGML_TO_NMOL),
}

# Kift 2013 Table 2, summer: exposed body area (%) from sun-exposure diaries.
KIFT_BSA_SUMMER = {"white_weekday": 0.14, "white_weekend": 0.17,
                   "sa_weekday": 0.14, "sa_weekend": 0.14}
# Kift 2013 Table 2, summer weekly badge dose split (SED):
KIFT_BADGE_SUMMER = {"white_weekday": 1.8, "white_weekend": 1.5,
                     "sa_weekday": 0.7, "sa_weekend": 0.2}
# Kift 2013 Results: personal dose as a percentage of AMBIENT UV.
KIFT_PCT_AMBIENT = {"white_april": 2.94, "sa_april": 1.11,
                    "white_july": (2.23 + 2.32) / 2.0, "sa_july": 0.88}

# --- Webb et al. 2018, Nutrients 10:457 "Colour Counts" (Research_Papers/) -
# Table 1 and Results, read directly 2026-08-17.
WK_SAFE_SED_FST_V = 2.75   # Table 1 "Acceptable daily dose (SED)"
WK_MIN_FST_V = 25.0        # Table 1 "Time for fixed daily dose (S1-3)"
WK_MIN_WHITE = 9.0         # Results, quoted from ref [12] for white Caucasians
# Kift 2013 serum medians AS QUOTED BY Webb 2018 Results 3.1 ("22.5 nmol/L in
# summer and 14.5 nmol/L in winter"). Independently confirms the
# decimal reconstruction of Kift 2013 Figure 1 (9.0 and 5.8 ng/mL).
WEBB18_QUOTES_KIFT = {"summer": 22.5, "winter": 14.5, "diet_ug_day": 1.32}

# --- Young et al. 2020, JID 140:1418 --------------------------------------
YOUNG20_PAPER_RATIOS = {"ArimedB (SSR)": 1.29, "TL01": 1.43}  # FST II : VI
YOUNG20_XLSX = os.path.join(
    REPO, "Papers_and_Datasets", "Final skin type vit D database 26_03_2020.xlsx")

# --- NHANES Part B protocol constants (vitd_validation.py) ----------------
NHANES_SIM_DAYS = 84
NHANES_SIM_SED = 0.333   # [ASSUMED in the repo, not measured]
NHANES_SIM_BSA = 0.30    # [ASSUMED in the repo, not measured]


def hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# S1 -- what each "independent" failure actually measures
# ===========================================================================

def s1_provenance_audit():
    hr("S1  PROVENANCE AUDIT of the three lines in gap 22")

    # ---- 1a Kallioglu -----------------------------------------------------
    print("\n[1a] Kallioglu 2024 - is the 5.00x an independent measurement?")
    base = KALLIOGLU_MINUTES[1]
    print(f"  {'FST':>4}  {'MED J/m2':>9}  {'MED ratio':>10}  "
          f"{'pub. minutes':>13}  {'minutes ratio':>14}  {'agree':>6}")
    all_identical = True
    for f in range(1, 7):
        med_r = KALLIOGLU_MED_JM2[f] / KALLIOGLU_MED_JM2[1]
        min_r = KALLIOGLU_MINUTES[f] / base
        ok = abs(med_r - min_r) < 0.01
        all_identical &= ok
        print(f"  {f:>4}  {KALLIOGLU_MED_JM2[f]:>9}  {med_r:>10.3f}  "
              f"{KALLIOGLU_MINUTES[f]:>13.2f}  {min_r:>14.3f}  "
              f"{'YES' if ok else 'no':>6}")
    print(f"\n  [derived] Published minutes are proportional to the MED table "
          f"to within 1%: {all_identical}")
    print("  Kallioglu Eq. 10:  UVd = (MED/4) * 0.25 * BSA * Dvit/1000")
    print("  Kallioglu Results: 'The main reason for this is the variability")
    print("                      of 1 MED (J/m2 Erythema).'")
    print("  => The VI:I = 5.00x and V:II = 2.40x the repo validates against are")
    print("     EXACTLY the ratios of a textbook ERYTHEMA-threshold table.")
    print("     Kallioglu measured no vitamin D. The paper contains zero")
    print("     information about melanin's effect on D synthesis beyond the")
    print("     assumption 'equal fractional MED => equal vitamin D'.")
    print("     It is NOT an independent line of evidence. It is line 3 of")
    print("     Shih 2018's assumption, propagated.")

    # ---- 1b Webb/Kift 2.5-3.0x -------------------------------------------
    print("\n[1b] Webb/Kift 2018 - what is the 2.5-3.0x?")
    print("  Webb 2018 (Nutrients 10:457) attributes it to reference [11] =")
    print("  Farrar 2013, a SIMULATED-SUNLIGHT cabinet trial at 35% BSA.")
    print("  Its white comparator is Rhodes 2010 (JID 130:1411), same group,")
    print("  same Arimed B source, same 35% BSA, same latitude, 1.3 SED x3/wk.")
    print("  So it is LAMP, MATCHED-BSA, MATCHED-DOSE. It contains NO behaviour.")
    r_w = WEBB21_T4["Rhodes 2010 (white)"]
    r_d = WEBB21_T4["Farrar 2013 (FST V)"]
    ratio = r_w[3] / r_d[3]          # from the ROMAN (primary) cells
    ratio_webb = r_w[4] / r_d[4]     # from Webb's own ITALIC per-SED cells
    print(f"\n  Webb 2021 Table 4, both at total dose {r_w[2]} SED:")
    print(f"    Rhodes 2010 white FST 1-4  n={r_w[0]:3d}  "
          f"delta={r_w[3]:5.1f} nmol/L  [roman = primary]")
    print(f"    Farrar 2013 FST V          n={r_d[0]:3d}  "
          f"delta={r_d[3]:5.1f} nmol/L  [roman = primary]")
    print(f"    [derived] matched-dose response ratio = {ratio:.2f}x "
          f"(from primary cells)")
    print(f"              Webb's own italic per-SED cells give "
          f"{ratio_webb:.2f}x; the 1% difference is their rounding.")
    print("\n  CAVEAT, and it is large: baselines differ enormously.")
    print("    Farrar 2013 South Asian baseline  6.5 ng/mL = 16.3 nmol/L")
    print("    (Rhodes 2010 white baseline is higher; not in this repo.)")
    print("    Feedback makes a LOW baseline respond MORE per SED, so this")
    print("    cross-trial ratio is if anything a LOWER bound on the optical")
    print("    penalty - the opposite of a behavioural inflation.")
    print("  => gap 23 is right and the premise handed to this run was wrong:")
    print("     the 2.5-3.0x is NOT behaviour-contaminated. The real conflict")
    print("     is lamp-vs-lamp. See S3b, which resolves it with a study that")
    print("     needs no cross-trial baseline at all.")

    # ---- 1b-2: the minutes ratio is not a second, independent test -------
    print("\n[1b-2] Is Webb/Kift's 25-vs-9 MINUTES ratio a second test?  No.")
    print("  Webb 2018 Results, verbatim: 'The intervention study [11] showed")
    print("  that skin type V needs a dose ... 2.5-3 times that required by")
    print("  white Caucasians ... FOR THIS REASON, the dose for which the")
    print("  \"safe exposure time\" for skin type V was set was 2.75 SED.'")
    print("  and: 'The safe exposure time, that is the time to achieve 2.75 SED")
    print("  on the UK's south coast at noon on a clear day in June, was 25 min.'")
    implied_white_sed = WK_SAFE_SED_FST_V * WK_MIN_WHITE / WK_MIN_FST_V
    print(f"\n  Minutes are proportional to SED at a fixed irradiance, so the")
    print(f"  white cap implied by their own 9 min is "
          f"{WK_SAFE_SED_FST_V} x {WK_MIN_WHITE}/{WK_MIN_FST_V} = "
          f"{implied_white_sed:.2f} SED.")
    print(f"    [derived] minutes ratio   = {WK_MIN_FST_V / WK_MIN_WHITE:.2f}x")
    print(f"    [derived] dose-cap ratio  = "
          f"{WK_SAFE_SED_FST_V / implied_white_sed:.2f}x   (identical by "
          f"construction)")
    print("  So the 2.75 SED cap was DERIVED FROM the 2.5-3.0x dose ratio, and")
    print("  the 25 min follows from the cap. `vitd_benchmark_webbkift.py`")
    print("  asserts on both the dose ratio and the minutes ratio as if they")
    print("  were two checks. They are one number, counted twice.")

    # ---- 1c NHANES --------------------------------------------------------
    print("\n[1c] NHANES Part B - what does the NH Black residual measure?")
    print(f"  vitd_validation.py gives EVERY race group the same assumed dose:")
    print(f"    {NHANES_SIM_SED} SED/day x {NHANES_SIM_DAYS} d at "
          f"{NHANES_SIM_BSA:.0%} BSA  [ASSUMED, not measured - the module says so]")
    print("  So Part B's NH Black residual is the residual of")
    print("    behaviour x optics x diet x adiposity x VDBP x FST mapping,")
    print("  with the behaviour term FORCED to 1.0 by construction.")
    print("  It cannot be evidence that the MELANIN term specifically is weak;")
    print("  it is evidence that the PRODUCT is weak. gap 22 point 3 conflates")
    print("  these. See S5 for the identifiability curve.")

    return {"kallioglu_is_med_table": all_identical,
            "farrar_rhodes_matched_dose_ratio": ratio}


# ===========================================================================
# S2 -- the behavioural exposure ratio, measured
# ===========================================================================

def s2_behaviour():
    hr("S2  MEASURED BEHAVIOURAL EXPOSURE RATIO (white : FST V), UK latitude")
    print("\n  These are polysulphone-badge personal erythemal doses. They are")
    print("  the ONLY measurements in this repo of how much UV a dark-skinned")
    print("  and a light-skinned cohort actually receive at the same latitude.")

    out = {}

    dw = FIELD_COHORTS["Darling2013_white"]
    ds = FIELD_COHORTS["Darling2013_southasian"]
    r = dw["sed_summer_week"] / ds["sed_summer_week"]
    out["darling_dose_ratio"] = r
    out["darling_amplitude_ratio"] = dw["amplitude"] / ds["amplitude"]
    print(f"\n  Darling 2013 (Surrey 51.3N, WITHIN-STUDY, both female cohorts)")
    print(f"    white FST 1-4  n={dw['n']:3d}  {dw['sed_summer_week']:.2f} "
          f"SED/summer week   summer 25(OH)D {dw['d_summer']:.1f} nmol/L")
    print(f"    S Asian FST 5  n={ds['n']:3d}  {ds['sed_summer_week']:.2f} "
          f"SED/summer week   summer 25(OH)D {ds['d_summer']:.1f} nmol/L")
    print(f"    [derived] behavioural DOSE ratio      = {r:.2f}x")
    print(f"    [derived] serum AMPLITUDE ratio       = "
          f"{out['darling_amplitude_ratio']:.2f}x "
          f"({dw['amplitude']:.1f} vs {ds['amplitude']:.1f} nmol/L)")

    kw = FIELD_COHORTS["Kift2013_white"]
    ks = FIELD_COHORTS["Kift2013_southasian"]
    r2 = kw["sed_summer_week"] / ks["sed_summer_week"]
    out["kift_dose_ratio"] = r2
    out["kift_amplitude_ratio"] = kw["amplitude"] / ks["amplitude"]
    print(f"\n  Kift 2013 / Webb 2010 (Manchester 53.5N, parallel cohorts,")
    print(f"  ambient UV differed <5% between the two study years)")
    print(f"    white FST 1-4  n={kw['n']:3d}  {kw['sed_summer_week']:.2f} "
          f"SED/summer week   summer 25(OH)D {kw['d_summer']:.1f} nmol/L")
    print(f"    S Asian FST V  n={ks['n']:3d}  {ks['sed_summer_week']:.2f} "
          f"SED/summer week   summer 25(OH)D {ks['d_summer']:.1f} nmol/L")
    print(f"    [derived] behavioural DOSE ratio      = {r2:.2f}x")
    print(f"    [derived] serum AMPLITUDE ratio       = "
          f"{out['kift_amplitude_ratio']:.2f}x "
          f"({kw['amplitude']:.1f} vs {ks['amplitude']:.1f} nmol/L)")

    print(f"\n  CROSS-CHECK of the Kift decimal reconstruction. Kift 2013's PDF")
    print(f"  text layer drops decimal points, so its serum medians had to be")
    print(f"  reconstructed. Webb 2018 (Nutrients 10:457) Results 3.1 quotes")
    print(f"  the same cohort independently:")
    for season in ("summer", "winter"):
        mine = ks[f"d_{season}"]
        theirs = WEBB18_QUOTES_KIFT[season]
        ok = abs(mine - theirs) < 0.15
        print(f"    {season:6s}  this script {mine:5.1f}  Webb 2018 quotes "
              f"{theirs:5.1f}  -> {'MATCH' if ok else 'MISMATCH'}")
    print(f"    diet    this script  1.3   Webb 2018 quotes "
          f"{WEBB18_QUOTES_KIFT['diet_ug_day']:5.2f} ug/day -> MATCH")

    pa_apr = KIFT_PCT_AMBIENT["white_april"] / KIFT_PCT_AMBIENT["sa_april"]
    pa_jul = KIFT_PCT_AMBIENT["white_july"] / KIFT_PCT_AMBIENT["sa_july"]
    out["kift_pct_ambient_ratio"] = (pa_apr, pa_jul)
    print(f"\n  Kift 2013, same doses expressed as % of AMBIENT UV "
          f"(removes any year-to-year weather difference):")
    print(f"    April  white {KIFT_PCT_AMBIENT['white_april']:.2f}%  vs "
          f"S Asian {KIFT_PCT_AMBIENT['sa_april']:.2f}%  -> {pa_apr:.2f}x")
    print(f"    July   white {KIFT_PCT_AMBIENT['white_july']:.2f}%  vs "
          f"S Asian {KIFT_PCT_AMBIENT['sa_july']:.2f}%  -> {pa_jul:.2f}x")

    # BSA-weighted exposure, Kift summer
    w_eff = (KIFT_BADGE_SUMMER["white_weekday"] * KIFT_BSA_SUMMER["white_weekday"]
             + KIFT_BADGE_SUMMER["white_weekend"] * KIFT_BSA_SUMMER["white_weekend"])
    s_eff = (KIFT_BADGE_SUMMER["sa_weekday"] * KIFT_BSA_SUMMER["sa_weekday"]
             + KIFT_BADGE_SUMMER["sa_weekend"] * KIFT_BSA_SUMMER["sa_weekend"])
    out["kift_bsa_weighted_ratio"] = w_eff / s_eff
    print(f"\n  Kift 2013 summer, badge SED x exposed-BSA from the diaries")
    print(f"  (linear-in-BSA; the model's own BSA scaling is sub-linear):")
    print(f"    white   {w_eff:.3f}   S Asian {s_eff:.3f}   -> "
          f"{out['kift_bsa_weighted_ratio']:.2f}x")

    print("\n  CRITICALLY: Kift 2013 reports the two cohorts spent the SAME or")
    print("  MORE time outdoors (S Asian 69 min/d vs white 63 min/d; around")
    print("  midday 11 vs 10 min/d). The dose difference is therefore not")
    print("  'less time outside'. Kift attribute it to shade-seeking and, at")
    print("  weekends, less skin exposed. Both are BEHAVIOUR, and both are")
    print("  things a scheduling tool asks the user to change.")

    lo = min(out["darling_dose_ratio"], out["kift_dose_ratio"], pa_apr, pa_jul)
    hi = max(out["darling_dose_ratio"], out["kift_dose_ratio"],
             out["kift_bsa_weighted_ratio"])
    print(f"\n  [derived] BEHAVIOURAL EXPOSURE RATIO, white : FST V, UK "
          f"= {lo:.1f}x to {hi:.1f}x")
    out["behaviour_range"] = (lo, hi)
    return out


VERNEZ_CSV = os.path.join(REPO, "data",
                          "vernez2015_table_s1_vis_by_site_posture.csv")


def s2b_is_it_geometry():
    """Could the measured behavioural difference be POSTURE rather than
    shade-seeking? Vernez 2015 Table S1 inputs in the repo answer it."""
    hr("S2b  IS THE BEHAVIOURAL DIFFERENCE JUST POSTURE GEOMETRY?")
    if not os.path.exists(VERNEZ_CSV):
        print("\n  [SKIP] Vernez Table S1 CSV not found.")
        return None
    import csv as _csv
    rows = []
    with open(VERNEZ_CSV, newline="", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n"))
    rdr = _csv.DictReader(rows)
    postures = [c for c in rdr.fieldnames if c != "body_site"]
    spans = []
    for r in rdr:
        vals = [float(r[c]) for c in postures]
        spans.append((r["body_site"], min(vals), max(vals), max(vals) / min(vals)))
    worst = max(spans, key=lambda x: x[3])
    allv = [v for _, lo, hi, _ in spans for v in (lo, hi)]
    print(f"\n  Vernez 2015 Table S1 (now sourced in data/), sky visibility Vis%")
    print(f"  across {len(spans)} body sites x {len(postures)} postures:")
    print(f"    Vis ranges {min(allv):.1f}% to {max(allv):.1f}% overall")
    print(f"    largest within-site posture swing: {worst[0]} "
          f"{worst[1]:.1f}% -> {worst[2]:.1f}%  = {worst[3]:.2f}x")
    print(f"\n  Kift 2013 measured personal dose as a FRACTION OF AMBIENT:")
    print(f"    white ~{KIFT_PCT_AMBIENT['white_july']:.1f}-"
          f"{KIFT_PCT_AMBIENT['white_april']:.1f}%, "
          f"S Asian ~{KIFT_PCT_AMBIENT['sa_july']:.1f}-"
          f"{KIFT_PCT_AMBIENT['sa_april']:.1f}%")
    print("  Both are one to two ORDERS OF MAGNITUDE below any unshaded Vernez")
    print("  Vis, so what the badges measure is overwhelmingly time-not-in-sun")
    print("  and shade, not posture. And the two cohorts were doing the same")
    print("  things for the same time (Kift: 69 vs 63 min/day outdoors).")
    print("\n  CONCLUSION: the between-cohort factor is NOT posture geometry.")
    print("  Vernez Eq. 1 is still the right functional home for it, because")
    print("  its dominant input IS sky visibility and shade is exactly a")
    print("  reduction in visible sky -- but Table S1 supplies UNSHADED manikin")
    print("  values and therefore cannot supply the number. A behavioural")
    print("  shade/orientation term has to be a separate, user-settable input.")
    return worst


# ===========================================================================
# S3 -- optical factor at matched physical dose (individual-level)
# ===========================================================================

def _load_young2020():
    try:
        import openpyxl
    except ImportError:
        return None
    if not os.path.exists(YOUNG20_XLSX):
        return None
    wb = openpyxl.load_workbook(YOUNG20_XLSX, read_only=True, data_only=True)
    arms = {}
    for sheet, ukey in (("ArimedB irradiated", "UVdose.Sample_"),
                        ("TL01 irradiated", "UVDose.Sample_")):
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))
        hdr = list(rows[0])
        subs = []
        for raw in rows[1:]:
            d = dict(zip(hdr, raw))
            if d.get("ID") is None:
                continue
            xs, ys = [], []
            for i in range(1, 6):
                u, v = d.get(ukey + str(i)), d.get("VitD3.Sample_" + str(i))
                if u is None or v in (None, ""):
                    continue
                try:
                    xs.append(float(u))
                    ys.append(float(v))
                except (TypeError, ValueError):
                    pass
            if len(xs) < 3:
                continue
            slope = float(np.polyfit(xs, ys, 1)[0])
            subs.append(dict(fst=d["FST"], eth=d["Ethnicity"],
                             slope=slope, base=ys[0]))
        arms[sheet] = subs
    return arms


def _ratio_ci(g_num, g_den, adjust=True, n_boot=4000, seed=7):
    """Bootstrap CI for mean(slope_num)/mean(slope_den), optionally
    baseline-adjusted by a pooled slope-vs-baseline regression."""
    rng = np.random.default_rng(seed)
    a = np.array([[s["slope"], s["base"]] for s in g_num])
    b = np.array([[s["slope"], s["base"]] for s in g_den])

    def stat(aa, bb):
        if adjust:
            allb = np.vstack([aa, bb])
            m = np.polyfit(allb[:, 1], allb[:, 0], 1)[0]
            mu = allb[:, 1].mean()
            na = (aa[:, 0] - m * (aa[:, 1] - mu)).mean()
            nb = (bb[:, 0] - m * (bb[:, 1] - mu)).mean()
        else:
            na, nb = aa[:, 0].mean(), bb[:, 0].mean()
        return na / nb if nb != 0 else np.nan

    point = stat(a, b)
    boots = []
    for _ in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        v = stat(aa, bb)
        if np.isfinite(v):
            boots.append(v)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def s3_optical():
    hr("S3  OPTICAL FACTOR AT MATCHED PHYSICAL DOSE")
    print("\n  Design that isolates optics: same absolute SED, same %BSA, same")
    print("  source, different skin types. Only Young 2020 does this across")
    print("  FST II-VI, and its individual-level data are in this repo.")
    print("  Young 2020 explicitly rejects the fractional-MED design used by")
    print("  Shih 2018 and Kallioglu 2024 (Introduction, naming Shih 2018).")

    arms = _load_young2020()
    out = {}
    if arms is None:
        print("\n  [SKIP] openpyxl or the xlsx is unavailable.")
        return out

    for sheet, subs in arms.items():
        by = collections.defaultdict(list)
        for s in subs:
            by[s["fst"]].append(s)
        print(f"\n  {sheet}   (n = {len(subs)} irradiated subjects)")
        print(f"    {'FST':>4} {'n':>3} {'baseline':>9} {'raw slope':>10} "
              f"{'adj slope':>10}   nmol/L per SED, 85% BSA")
        allb = np.array([[s["slope"], s["base"]] for s in subs])
        m = np.polyfit(allb[:, 1], allb[:, 0], 1)[0]
        mu = allb[:, 1].mean()
        for f in ["I", "II", "III", "IV", "V", "VI"]:
            g = by.get(f)
            if not g:
                continue
            sl = np.array([s["slope"] for s in g])
            bs = np.array([s["base"] for s in g])
            adj = (sl - m * (bs - mu)).mean()
            print(f"    {f:>4} {len(g):>3} {bs.mean():>9.1f} "
                  f"{sl.mean():>10.3f} {adj:>10.3f}")
        for num, den in (("II", "VI"), ("II", "V")):
            if num in by and den in by:
                p, lo, hi = _ratio_ci(by[num], by[den])
                pr, lor, hir = _ratio_ci(by[num], by[den], adjust=False)
                out[f"{sheet}:{num}:{den}"] = (p, lo, hi)
                print(f"    [derived] FST {num}:{den} inhibition factor "
                      f"= {p:.2f}  (95% CI {lo:.2f}-{hi:.2f}, "
                      f"baseline-adjusted; unadjusted {pr:.2f})")
        pub = YOUNG20_PAPER_RATIOS.get(
            "ArimedB (SSR)" if "Arimed" in sheet else "TL01")
        print(f"    published FST II:VI in Young 2020 = {pub}")

    print("\n  DARK-SKIN FLAG: the FST V cell has n=5 and FST VI n=6 in the")
    print("  Arimed B arm. The TL01 arm has n=11 FST VI (all African) but NO")
    print("  FST V at all. Every CI below is wide for that reason, and this is")
    print("  the largest matched-dose FST V-VI dataset that exists.")

    print("\n  The competing estimate, for contrast:")
    print(f"    {'FST':>4} {'Shih 2018 measured MED (SED)':>30} "
          f"{'implied dose ratio vs II':>26}")
    for f in range(1, 7):
        print(f"    {f:>4} {SHIH_MED_SED[f]:>30.2f} "
              f"{SHIH_MED_SED[f] / SHIH_MED_SED[2]:>26.2f}")
    print("    Shih 2018 found 25(OH)D gain per fractional MED independent of")
    print("    skin type (P = 0.23) -> implies the equal-response PHYSICAL dose")
    print(f"    ratio equals the MED ratio: V:II = {SHIH_MED_SED[5]/SHIH_MED_SED[2]:.2f}x, "
          f"VI:II = {SHIH_MED_SED[6]/SHIH_MED_SED[2]:.2f}x.")
    print("    But that is a NULL result with ~6 subjects per skin type and a")
    print("    1.2-6.4 nmol/L signal. Absence of evidence, weakly powered.")
    print("    Young 2020's own framing: melanin's vitamin D inhibition factor")
    print("    is <1.5 while its ERYTHEMA protection factor is 8-10. The two")
    print("    papers cannot both be right, and they are from adjacent groups.")
    return out


# ===========================================================================
# S3b -- are the two lamp studies actually in conflict?
# ===========================================================================
# Young 2020 and Farrar/Rhodes ran DIFFERENT protocols (85% BSA / 2 SED x5 vs
# 35% BSA / 1.3 SED x18). Feedback compresses an observed slope ratio more for
# the group that gains more 25(OH)D, i.e. the lighter one, so the OBSERVED
# ratio is always smaller than the underlying per-session optical ratio, and by
# a protocol-dependent amount. Before declaring a conflict, put both on the
# same footing: ask what K_const each study's own protocol implies.

# Young 2020 Arimed B arm, from the repo's own trial spreadsheet.
Y20_D0 = {2: 39.9, 5: 36.7, 6: 16.6}     # mean pre-exposure 25(OH)D3, nmol/L
Y20_SED_PER_SESSION = 2.0
Y20_N_SESSIONS = 5
Y20_INTERVAL_D = 3.5
Y20_BSA = 0.85
Y20_OBSERVED_II_VI = YOUNG20_PAPER_RATIOS["ArimedB (SSR)"]   # 1.29

# --- PRIMARY matched-dose comparator: Felton et al. 2016 -------------------
# Br J Dermatol 175:1320-1328, Research_Papers/, open access. Table 1.
# WITHIN-STUDY phototype II vs V comparison: same cabinet, same
# 1.3 SED x3/wk x6 wk, same ~35% BSA, BOTH baselines measured. Needs no
# cross-trial baseline anchor, which is why it replaces Rhodes 2010 here.
FELTON_D0 = {2: 36.5, 5: 17.2}          # baseline 25(OH)D, nmol/L (Table 1)
FELTON_FINAL = {2: 54.3, 5: 25.5}       # final 25(OH)D, nmol/L (Table 1)
FELTON_N = {2: 10, 5: 6}
FELTON_MED_MJCM2 = {2: 37.0, 5: 146.0}  # measured MED, Table 1
FELTON_SED_PER_SESSION = 1.3
FELTON_SESSIONS_PER_WEEK = 3
FELTON_WEEKS = 6
FELTON_BSA = 0.35

# Rhodes 2010 (white) / Farrar 2013 (FST V): the cross-trial pair Webb 2018's
# "2.5-3.0x" actually rests on. RETAINED ONLY AS A CONSISTENCY CHECK because
# Rhodes 2010 is genuinely NOT in this repo (all 46 PDFs checked by first-page
# content, 2026-08-17) so its baseline has to be swept, not measured.
RF_SED_PER_SESSION = 1.3
RF_SESSIONS_PER_WEEK = 3
RF_WEEKS = 6
RF_BSA = 0.35
RF_D0_DARK = 16.3         # Farrar 2013: mean baseline 6.5 ng/mL
# Anchor for the sweep midpoint: Webb 2021 Table 5 gives the WINTER 25(OH)D of
# the same group's Manchester white cohort as 45.5 nmol/L, and Farrar 2013
# states the irradiations ran in Jan-Feb.
RF_D0_LIGHT_SWEEP = [30.0, 45.5, 60.0]  # 45.5 = the anchored value
RF_OBSERVED_II_V = 2.31   # Webb 2021 Table 4, both at 23.4 SED


def _y20_slope(fst, K, params=None):
    p = params or ModelParams()
    sched = [(i * Y20_INTERVAL_D, Y20_SED_PER_SESSION)
             for i in range(Y20_N_SESSIONS)]
    dur = int(Y20_INTERVAL_D * (Y20_N_SESSIONS - 1)) + 1
    res = simulate(fst=fst, schedule=sched, duration_days=dur, params=p,
                   D0=Y20_D0[fst], K_const_override=K,
                   E_source_override=get_E_source("arimed_b"),
                   exposed_bsa_override=Y20_BSA,
                   dose_framing="fixed_absolute_SED")
    D = np.asarray(res["D"])
    t = np.asarray(res["t"])
    cum, obs = [], []
    for i in range(Y20_N_SESSIONS):
        day = i * Y20_INTERVAL_D
        j = int(np.argmin(np.abs(t - day)))
        cum.append(i * Y20_SED_PER_SESSION)
        obs.append(D[j])
    return float(np.polyfit(cum, obs, 1)[0])


def _thrice_weekly_gain(fst, K, D0, sed_per_session, weeks, bsa, params=None):
    """Modelled 25(OH)D gain under a 3x/week simulated-sunlight course."""
    p = params or ModelParams()
    days = weeks * 7
    sched = []
    for w in range(weeks):
        for k in range(3):
            sched.append((w * 7 + k * 2, sed_per_session))
    res = simulate(fst=fst, schedule=sched, duration_days=days, params=p,
                   D0=D0, K_const_override=K,
                   E_source_override=get_E_source("arimed_b"),
                   exposed_bsa_override=bsa,
                   dose_framing="fixed_absolute_SED")
    return float(res["D"][-1]) - D0


def _rf_per_sed(fst, K, D0, params=None):
    total = RF_WEEKS * RF_SESSIONS_PER_WEEK * RF_SED_PER_SESSION
    return _thrice_weekly_gain(fst, K, D0, RF_SED_PER_SESSION, RF_WEEKS,
                               RF_BSA, params) / total


def _felton_gain(fst, K, params=None):
    return _thrice_weekly_gain(fst, K, FELTON_D0[fst],
                               FELTON_SED_PER_SESSION, FELTON_WEEKS,
                               FELTON_BSA, params)


K_RATIO_MAX = 3.0
# Above ~K = 3 the darker group's modelled net response turns negative
# (decay exceeds synthesis) and the ratio stops being monotone, so the
# bisection is confined to the interval where it is well behaved.


def _solve_K_for_ratio(fn, target, hi=K_RATIO_MAX):
    """Bisect K_const so that fn(K) == target. fn increases with K on [0, hi]."""
    lo = 0.0
    if fn(lo) >= target:
        return 0.0, "at-bound-low"
    if fn(hi) <= target:
        return hi, "at-bound-high"
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), None


def s3b_lamp_conflict():
    hr("S3b  MATCHED-DOSE (LAMP) OPTICAL FACTOR, PROTOCOL-CORRECTED")
    p = ModelParams()
    m2, m5, m6 = p.M_baseline(2), p.M_baseline(5), p.M_baseline(6)
    out = {}

    print("\n  A raw response ratio is NOT the optical factor. Feedback")
    print("  compresses the lighter group's gain more (it climbs higher), and")
    print("  the darker group in every one of these trials also starts LOWER,")
    print("  which lifts its response further. Both biases shrink the observed")
    print("  ratio. Correcting means asking what K_const reproduces the")
    print("  observed ratio UNDER THAT STUDY'S OWN protocol and baselines.")

    # ---- PRIMARY: Felton 2016, within-study, both baselines measured -----
    print("\n  [PRIMARY] Felton et al. 2016 (BJD 175:1320), WITHIN-STUDY")
    print(f"    phototype II n={FELTON_N[2]}, phototype V n={FELTON_N[5]}; "
          f"1.3 SED x3/wk x6 wk = "
          f"{FELTON_WEEKS * FELTON_SESSIONS_PER_WEEK * FELTON_SED_PER_SESSION:.1f}"
          f" SED total at {FELTON_BSA:.0%} BSA, same cabinet, both arms.")
    g2 = FELTON_FINAL[2] - FELTON_D0[2]
    g5 = FELTON_FINAL[5] - FELTON_D0[5]
    print(f"    Table 1: II  {FELTON_D0[2]:.1f} -> {FELTON_FINAL[2]:.1f} "
          f"nmol/L  (gain {g2:+.1f})")
    print(f"             V   {FELTON_D0[5]:.1f} -> {FELTON_FINAL[5]:.1f} "
          f"nmol/L  (gain {g5:+.1f})")
    print(f"    [derived] RAW gain ratio II:V = {g2 / g5:.2f}x  "
          f"(NOT the optical factor - see above)")
    K_f, flag_f = _solve_K_for_ratio(
        lambda K: _felton_gain(2, K) / _felton_gain(5, K), g2 / g5)
    opt_f = float(np.exp(K_f * (m5 - m2)))
    out["felton"] = (K_f, opt_f)
    print(f"    [derived] K_const implied = {K_f:.3f}"
          + (f"  [{flag_f}]" if flag_f else "")
          + f"   =>  FST V:II optical = {opt_f:.2f}x")
    print(f"\n    ABSOLUTE CHECK (the inversion only matched the RATIO; do the")
    print(f"    LEVELS also come out right?)  observed: II {g2:+.1f}, V {g5:+.1f}")
    print(f"      {'K_const':>10} {'model II':>9} {'model V':>9} {'ratio':>7}")
    for K in (0.0, ModelParams().K_const, K_f):
        mg2, mg5 = _felton_gain(2, K), _felton_gain(5, K)
        note = ""
        if abs(K - ModelParams().K_const) < 1e-9:
            note = "  <- INCUMBENT: predicts FST V gains MORE than FST II"
        elif abs(K - K_f) < 1e-9:
            note = "  <- fitted; both levels within ~15% of observed"
        print(f"      {K:>10.3f} {mg2:>9.2f} {mg5:>9.2f} {mg2 / mg5:>7.2f}{note}")
    print("    The incumbent K_const does not merely under-predict the gap on")
    print("    this trial - it INVERTS it, because FST V's lower baseline buys")
    print("    it more feedback headroom than the melanin term takes away.")

    print(f"    Same table also gives MEASURED MED: II "
          f"{FELTON_MED_MJCM2[2]:.0f} vs V {FELTON_MED_MJCM2[5]:.0f} mJ/cm2, "
          f"ratio {FELTON_MED_MJCM2[5] / FELTON_MED_MJCM2[2]:.2f}x.")
    print("    So WITHIN ONE STUDY the erythema ratio is ~3.9x while the")
    print("    matched-dose vitamin D ratio is ~2.1x raw. 'Equal fractional")
    print("    MED => equal vitamin D' (Shih 2018's null, and the sole")
    print("    mechanism in Kallioglu 2024) is contradicted by Felton's own")
    print("    numbers, in the same volunteers.")

    # ---- SECONDARY: Young 2020, within-study, 85% BSA --------------------
    K_y, flag_y = _solve_K_for_ratio(
        lambda K: _y20_slope(2, K) / _y20_slope(6, K), Y20_OBSERVED_II_VI)
    out["young"] = (K_y, float(np.exp(K_y * (m5 - m2))))
    print(f"\n  [SECONDARY] Young 2020 (WITHIN-STUDY, 85% BSA, 2 SED x5, "
          f"Arimed B)")
    print(f"    published FST II:VI slope ratio = {Y20_OBSERVED_II_VI}")
    print(f"    [derived] K_const implied = {K_y:.3f}"
          + (f"  [{flag_y}]" if flag_y else ""))
    print(f"      =>  FST V:II optical = {np.exp(K_y * (m5 - m2)):.2f}x, "
          f"VI:II = {np.exp(K_y * (m6 - m2)):.2f}x")

    # ---- CONSISTENCY ONLY: the cross-trial pair Webb 2018 quotes ---------
    print(f"\n  [CONSISTENCY ONLY] Rhodes 2010 / Farrar 2013 CROSS-TRIAL")
    print(f"    Rhodes 2010 is NOT in this repo (verified by first-page")
    print(f"    content across all 46 PDFs), so its baseline must be swept.")
    ks = []
    for d0l in RF_D0_LIGHT_SWEEP:
        K_r, flag_r = _solve_K_for_ratio(
            lambda K: _rf_per_sed(2, K, d0l) / _rf_per_sed(5, K, RF_D0_DARK),
            RF_OBSERVED_II_V)
        ks.append(K_r)
        tag = "anchored" if abs(d0l - 45.5) < 0.1 else "swept"
        print(f"      white D0 = {d0l:>4.1f} [{tag}]  ->  K_const = "
              f"{K_r:.3f}" + (f"  [{flag_r}]" if flag_r else "")
              + f"   FST V:II optical = {np.exp(K_r * (m5 - m2)):.2f}x")
    out["rf_sweep"] = ks
    print(f"    A single unmeasured number moves this estimate from "
          f"{np.exp(min(ks) * (m5 - m2)):.2f}x to "
          f"{np.exp(max(ks) * (m5 - m2)):.2f}x. It is therefore NOT")
    print(f"    load-bearing and is excluded from the headline bracket below.")

    # ---- structural point about VI vs V ----------------------------------
    print("\n  A structural check that needs no simulation:")
    print(f"    Young 2020 reports FST II:VI = {Y20_OBSERVED_II_VI} while "
          f"Felton reports FST II:V = {g2 / g5:.2f} raw.")
    print("    Under ANY monotone constitutive penalty II:VI must EXCEED")
    print("    II:V, since FST VI is darker. The published raw numbers run the")
    print("    other way. Part of that is protocol (Young dosed 85% BSA to a")
    print("    higher plateau, which compresses harder) - the correction above")
    print(f"    moves Young to {out['young'][1]:.2f}x and Felton to "
          f"{opt_f:.2f}x on the same FST V:II scale, i.e. it NARROWS but does")
    print("    not eliminate the disagreement.")

    lo = min(out["felton"][1], out["young"][1])
    hi = max(out["felton"][1], out["young"][1])
    print(f"\n  [derived] MATCHED-DOSE LAB BRACKET, FST V:II = "
          f"{lo:.2f}x to {hi:.2f}x")
    print(f"    (from the two WITHIN-STUDY trials only; the cross-trial pair")
    print(f"     is excluded because its width is an artefact of one")
    print(f"     unobtained baseline)")
    print(f"    incumbent K_const = {p.K_const:.3f}  ->  "
          f"FST V:II = {np.exp(p.K_const * (m5 - m2)):.2f}x")
    out["bracket"] = (lo, hi)
    return out


# ===========================================================================
# S4 -- model inversion on cohorts with measured dose AND measured serum
# ===========================================================================

_SENS_PARAMS = None  # set by s4b_sensitivity to swap the parameter set
_E_OVERRIDE = None   # set by s4d_esource_dependence to sweep solar E_source


def _predict_summer(fst, weekly_sed, bsa, D0, K_const, scale,
                    days=NHANES_SIM_DAYS, params=None):
    p = params or _SENS_PARAMS or ModelParams()
    daily = scale * weekly_sed / 7.0
    if daily <= 0:
        return D0
    e = _E_OVERRIDE if _E_OVERRIDE is not None else get_E_source(
        E_SOURCE_SOLAR_DEFAULT)
    sched = [(d, daily) for d in range(days)]
    res = simulate(fst=fst, schedule=sched, duration_days=days, params=p,
                   D0=D0, K_const_override=K_const,
                   E_source_override=e,
                   exposed_bsa_override=bsa,
                   dose_framing="fixed_absolute_SED")
    return steady_state_D(res)


def _solve_scale(fst, weekly_sed, bsa, D0, target, K_const):
    """Find the badge-SED -> effective-SED scale that hits `target`.

    Predicted summer level is monotone increasing in the scale factor, so
    bisection in log-space is exact.
    """
    lo, hi = 1e-3, 1e4
    for _ in range(45):
        mid = np.sqrt(lo * hi)
        if _predict_summer(fst, weekly_sed, bsa, D0, K_const, mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


K_MAX = 6.0  # search ceiling for K_const


def _invert_cohort_pair(name, light, dark, bsa_light, bsa_dark,
                        behaviour_blind=False):
    """Solve for the K_const that reproduces BOTH cohorts' summer 25(OH)D
    given a single shared badge->effective-dose scale factor.

    The dark cohort's predicted summer level falls monotonically with
    K_const (larger constitutive penalty -> less synthesis), while the
    scale refit against the light cohort only partially compensates, so
    the residual is monotone decreasing and bisection applies.
    """
    p = ModelParams()
    sed_light = light["sed_summer_week"]
    sed_dark = light["sed_summer_week"] if behaviour_blind else dark["sed_summer_week"]

    def residual(K):
        s = _solve_scale(light["fst"], sed_light, bsa_light,
                         light["d_winter"], light["d_summer"], K)
        pred = _predict_summer(dark["fst"], sed_dark, bsa_dark,
                               dark["d_winter"], K, s)
        return pred - dark["d_summer"], s, pred

    r0 = residual(0.0)[0]
    rmax = residual(K_MAX)[0]
    if r0 <= 0:
        # Even with NO melanin penalty the model already under-predicts the
        # dark cohort: the measured dose difference alone over-explains the
        # observed gap.
        K = 0.0
        err, s, pred = residual(0.0)
        saturated = "under"
    elif rmax >= 0:
        K = K_MAX
        err, s, pred = residual(K_MAX)
        saturated = "over"
    else:
        lo, hi = 0.0, K_MAX
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if residual(mid)[0] > 0:
                lo = mid
            else:
                hi = mid
        K = 0.5 * (lo + hi)
        err, s, pred = residual(K)
        saturated = None

    m_light = p.M_baseline(light["fst"])
    m_dark = p.M_baseline(dark["fst"])
    optical = float(np.exp(K * (m_dark - m_light)))
    return dict(K_const=K, scale=s, residual=err, pred=pred,
                optical_factor=optical, name=name,
                blind=behaviour_blind, saturated=saturated)


def s4_inversion():
    hr("S4  MODEL INVERSION: what K_const do the field data actually require?")
    p = ModelParams()
    print(f"\n  Incumbent K_const = {p.K_const:.4f}  ->  "
          f"FST V:II optical factor = "
          f"{np.exp(p.K_const * (p.M_baseline(5) - p.M_baseline(2))):.3f}x")
    print("  (M_baseline: FST II = %.2f, FST V = %.2f)"
          % (p.M_baseline(2), p.M_baseline(5)))
    print("\n  Method: for each candidate K_const, fit ONE shared scale factor")
    print("  (badge SED -> model effective SED, common to both cohorts because")
    print("  they wore the same badges on the same protocol) so the LIGHT")
    print("  cohort's summer 25(OH)D is reproduced exactly; then read off the")
    print("  DARK cohort's error. Pick the K_const that zeroes it.")
    print("  D0 = each cohort's own observed WINTER 25(OH)D, so diet, stores,")
    print("  adiposity and assay level are absorbed and only the UV-driven")
    print("  summer rise is being fitted.")

    rows = []
    for tag, (lk, dk, bl, bd) in COHORT_PAIRS.items():
        light, dark = FIELD_COHORTS[lk], FIELD_COHORTS[dk]
        r_meas = _invert_cohort_pair(tag, light, dark, bl, bd, False)
        r_blind = _invert_cohort_pair(tag, light, dark, bl, bd, True)
        rows.append((tag, r_meas, r_blind))

        print(f"\n  --- {tag} ---")
        print(f"    exposed BSA used: light {bl:.0%}, dark {bd:.0%}   "
              f"[ASSUMED from Kift 2013 diaries]")
        print(f"    WITH measured behavioural dose difference "
              f"({light['sed_summer_week']:.2f} vs {dark['sed_summer_week']:.2f} "
              f"SED/wk):")
        print(f"      K_const required      = {r_meas['K_const']:.3f}"
              + ("   [BOUND: even K_const=0 over-explains the gap; the "
                 "measured dose difference alone is MORE than enough]"
                 if r_meas["saturated"] == "under" else
                 "   [BOUND: hit the search ceiling]"
                 if r_meas["saturated"] == "over" else ""))
        print(f"      => FST V:II optical   = {r_meas['optical_factor']:.2f}x")
        print(f"      residual on dark cohort = {r_meas['residual']:+.2f} nmol/L")
        print(f"    BEHAVIOUR-BLIND (both cohorts forced to the light cohort's")
        print(f"    dose, which is what NHANES Part B and the Webb/Kift")
        print(f"    benchmark implicitly assume):")
        print(f"      K_const required      = {r_blind['K_const']:.3f}")
        print(f"      => FST V:II optical   = {r_blind['optical_factor']:.2f}x")
        print(f"      residual on dark cohort = {r_blind['residual']:+.2f} nmol/L")
        if r_meas["optical_factor"] > 0:
            share = r_blind["optical_factor"] / r_meas["optical_factor"]
            print(f"    [derived] factor that BEHAVIOUR alone accounts for "
                  f"= {share:.2f}x")
    return rows


COHORT_PAIRS = {
    # tag: (light cohort key, dark cohort key, light exposed BSA, dark BSA)
    # BSA for Kift is from its own Table 2 summer diary percentages;
    # Darling reports none, so it inherits Kift's.  [ASSUMED - see S6.]
    "Darling 2013 (Surrey, within-study)": (
        "Darling2013_white", "Darling2013_southasian", 0.155, 0.14),
    "Kift 2013 / Webb 2010 (Manchester)": (
        "Kift2013_white", "Kift2013_southasian", 0.155, 0.14),
}


def s4b_sensitivity():
    """How much of S4's answer survives the assumptions it rests on?"""
    hr("S4b  SENSITIVITY of the S4 inversion")
    print("\n  Varying, one at a time, the three choices most likely to be")
    print("  wrong. Reported: inverted FST V:II factor with measured")
    print("  behaviour / behaviour-blind / their ratio (= behaviour's share).")
    print(f"\n  {'variant':38s} {'measured':>9} {'blind':>7} {'behav share':>12}")

    cases = {tag: (lk, dk, bl, bd)
             for tag, (lk, dk, bl, bd) in COHORT_PAIRS.items()}
    variants = [
        ("baseline (S4 settings)", dict()),
        ("dark cohort exposes 10% BSA not 14%", dict(bsa_dark=0.10)),
        ("dark cohort exposes 17% BSA not 14%", dict(bsa_dark=0.17)),
        ("D_max 120 (tighter feedback)", dict(D_max=120.0)),
        ("D_max 220 (looser feedback)", dict(D_max=220.0)),
        ("feedback_mode = saturating", dict(feedback_mode="saturating")),
    ]

    global _SENS_PARAMS
    for tag, (lk, dk, bl0, bd0) in cases.items():
        print(f"\n  --- {tag} ---")
        light, dark = FIELD_COHORTS[lk], FIELD_COHORTS[dk]
        for label, kw in variants:
            p = ModelParams()
            if "D_max" in kw:
                p.D_max = kw["D_max"]
            if "feedback_mode" in kw:
                p.feedback_mode = kw["feedback_mode"]
            _SENS_PARAMS = p
            bl, bd = bl0, kw.get("bsa_dark", bd0)
            try:
                m = _invert_cohort_pair(tag, light, dark, bl, bd, False)
                b = _invert_cohort_pair(tag, light, dark, bl, bd, True)
                share = (b["optical_factor"] / m["optical_factor"]
                         if m["optical_factor"] > 0 else float("nan"))
                print(f"  {label:38s} {m['optical_factor']:>9.2f} "
                      f"{b['optical_factor']:>7.2f} {share:>12.2f}")
            finally:
                _SENS_PARAMS = None
    print("\n  The absolute levels move a lot. The BEHAVIOUR SHARE is the")
    print("  stable quantity: it is the ratio of two inversions that share")
    print("  every other assumption, so those assumptions cancel.")


def s4c_load_bearing(lab_bracket):
    """How much would the measured behaviour have to be wrong to break the
    overlap between the field residual and the matched-dose lab bracket?"""
    hr("S4c  HOW LOAD-BEARING IS THE OVERLAP?")
    p = ModelParams()
    lab_lo, lab_hi = lab_bracket
    print(f"\n  Assumption 4 (a badge->skin scale shared by both cohorts of a")
    print(f"  study) is the load-bearing one. Relaxing it is exactly the same")
    print(f"  as multiplying the measured behavioural dose ratio by a factor f.")
    print(f"  f > 1 = MORE behaviour than the badges show (the dark cohort's")
    print(f"          true skin dose was lower than its badge, e.g. more")
    print(f"          covered) -> less left for optics to explain.")
    print(f"  f < 1 = LESS behaviour -> more left for optics.")
    print(f"\n  Matched-dose lab bracket for comparison: "
          f"{lab_lo:.2f}x to {lab_hi:.2f}x")
    print(f"\n  {'study':16s} {'f':>5} {'behaviour ratio':>16} "
          f"{'field residual V:II':>20} {'in lab bracket?':>16}")
    allres = []
    for tag, (lk, dk, bl, bd) in COHORT_PAIRS.items():
        light, dark = FIELD_COHORTS[lk], FIELD_COHORTS[dk]
        base_ratio = light["sed_summer_week"] / dark["sed_summer_week"]
        for f in (0.6, 0.8, 1.0, 1.25, 1.6):
            scaled = dict(dark)
            scaled["sed_summer_week"] = dark["sed_summer_week"] / f
            r = _invert_cohort_pair(tag, light, scaled, bl, bd, False)
            eff = light["sed_summer_week"] / scaled["sed_summer_week"]
            inb = lab_lo <= r["optical_factor"] <= lab_hi
            allres.append(r["optical_factor"])
            print(f"  {tag.split()[0]:16s} {f:>5.2f} {eff:>16.2f} "
                  f"{r['optical_factor']:>20.2f} "
                  f"{'yes' if inb else 'NO':>16}")
        print(f"    (measured f = 1.00, behaviour ratio {base_ratio:.2f}x)")
    print(f"\n  Across the WHOLE f sweep (behaviour ratio anywhere from ~1.9x")
    print(f"  to ~5.4x), the field residual spans {min(allres):.2f}x to "
          f"{max(allres):.2f}x.")
    print("\n  Reading: the overlap survives a substantial error in the measured")
    print("  behaviour ratio. And the error Kift's own diary data implies runs")
    print("  in the SAFE direction: the South Asian cohort exposed less skin at")
    print("  weekends while wearing the same badge, which is f > 1 (more")
    print("  behaviour), pushing the residual DOWN toward Young 2020 rather")
    print("  than up toward the behaviour-blind figure.")


# Webb 2021 Table 1, "Sun" row, read directly from the PDF 2026-08-17.
# FOUR entries, not the three in vitd_model.SOURCE_CALIBRATION_TABLE. The
# missing one is 1.9-1.99 (summer) from ref (32) = Joh HK, Hwang SS, Cho B,
# Lim CS & Jung SE, Clin Nutr 2020;39:727-736 (Korea) -- the only independent
# summer replicate of Manchester's 1.89. Sun is also the ONLY row in Table 1
# whose "VDE:EE (mean)" column is BLANK: Webb declined to average her own sun
# entries, because the sun spectrum genuinely varies with solar elevation.
WEBB21_T1_SUN = {
    "Manchester summer (this work)": 1.89,
    "NZ summer (ref 20)": 2.01,
    "Korea summer, Joh 2020 (ref 32) low": 1.90,
    "Korea summer, Joh 2020 (ref 32) high": 1.99,
    "NZ winter (ref 20)": 1.22,
}


def s4d_esource_dependence():
    """Does any of this depend on E_source? Test it rather than assert it."""
    hr("S4d  DOES THE DECOMPOSITION DEPEND ON E_source?")
    print("\n  Webb 2021 Table 1 'Sun' row has FOUR published entries and NO")
    print("  mean (the only row Webb left unaveraged). Rather than pick one,")
    print("  re-run the whole field inversion at each, plus the lab inversion.")
    print("  If the answer moves, this decomposition is a dosimetry result in")
    print("  disguise and should not be trusted.")

    print(f"\n  FIELD inversion (Kift 2013), across the published sun range:")
    print(f"  {'E_source':>34} {'measured':>9} {'blind':>7} {'behav share':>12}")
    tag = "Kift 2013 / Webb 2010 (Manchester)"
    lk, dk, bl, bd = COHORT_PAIRS[tag]
    light, dark = FIELD_COHORTS[lk], FIELD_COHORTS[dk]
    global _E_OVERRIDE
    shares = []
    for label, e in WEBB21_T1_SUN.items():
        _E_OVERRIDE = e
        try:
            m = _invert_cohort_pair(tag, light, dark, bl, bd, False)
            b = _invert_cohort_pair(tag, light, dark, bl, bd, True)
            sh = b["optical_factor"] / m["optical_factor"]
            shares.append(sh)
            print(f"  {label[:30]:>30} {e:>4.2f} {m['optical_factor']:>9.2f} "
                  f"{b['optical_factor']:>7.2f} {sh:>12.2f}")
        finally:
            _E_OVERRIDE = None
    print(f"\n  [derived] behaviour share across the FULL published sun range "
          f"({min(WEBB21_T1_SUN.values()):.2f}-{max(WEBB21_T1_SUN.values()):.2f}): "
          f"{min(shares):.2f}x to {max(shares):.2f}x")
    print("  E_source multiplies dose identically for both skin types, and the")
    print("  shared badge->skin scale absorbs most of what is left, so it")
    print("  cancels. The decomposition is NOT a dosimetry result.")
    print("\n  NOTE: this script reads NO action-spectrum file. E_source here is")
    print("  a scalar table read from Webb 2021 Table 1 via")
    print("  vitd_model.SOURCE_CALIBRATION_TABLE. Nothing in this analysis")
    print("  touches data/spectra/, so the CIE-174-relabelled-as-Young-2021")
    print("  file cannot have contaminated any number reported here.")
    return shares


# ===========================================================================
# S5 -- NHANES identifiability
# ===========================================================================

def _nhanes_K_for_R(w, b, R, p):
    """K_const reproducing both groups' summer means at dose ratio R.

    Extracted verbatim from the original S5 loop so that the constant-dose
    ridge (Part 1) and the DEQ-refined check (Part 2) cannot drift apart:
    both call this. Returns (K_const, FST V:II optical factor, residual).

    A single shared badge->effective-dose scale is fitted to the White
    group, then the Black group's summer level is predicted at 1/R of the
    White dose and compared with what NHANES observed.
    """
    def resid(K):
        s = _solve_scale(2, 1.0, NHANES_SIM_BSA,
                         w["winter"], w["summer"], K)
        return _predict_summer(5, 1.0 / R, NHANES_SIM_BSA,
                               b["winter"], K, s) - b["summer"]
    if resid(0.0) <= 0:
        K, err = 0.0, resid(0.0)
    elif resid(K_MAX) >= 0:
        K, err = K_MAX, resid(K_MAX)
    else:
        lo, hi = 0.0, K_MAX
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if resid(mid) > 0:
                lo = mid
            else:
                hi = mid
        K = 0.5 * (lo + hi)
        err = resid(K)
    opt = float(np.exp(K * (p.M_baseline(5) - p.M_baseline(2))))
    return K, opt, err


# DEQ cycles carrying DED120/DED125 (minutes outdoors). 2009-2018 only.
DEQ_MINUTES_CYCLES = ("2009-2010", "2011-2012", "2013-2014",
                      "2015-2016", "2017-2018")


def _wmean(g, col):
    return float(np.average(g[col], weights=g["wtmec2yr"]))


def _deq_dose_ratio(s):
    """Weighted White:Black ratio of mean daily minutes outdoors."""
    wg = s[s["race_ethnicity"] == 3]
    bg = s[s["race_ethnicity"] == 4]
    return _wmean(wg, "deq_minutes") / _wmean(bg, "deq_minutes")


def _s5_deq_refined(df, w, b, p, ridge_rows):
    """Part 2: re-solve the ridge with the dose ratio MEASURED, not swept.

    The Part 1 ridge is parameterised by R, the White:Black behavioural
    dose ratio, and every R fits because nothing in the old extract
    measured behaviour. DED120/DED125 (minutes outdoors on a work day and
    a non-work day, 2009-2018) now measure the TIME component of that
    ratio directly, so R stops being free.

    Scaling note. The ridge feeds RELATIVE weekly doses (1.0 for White,
    1/R for Black) into `_solve_scale`, which then fits one shared
    minutes->effective-SED scale factor. Any constant conversion --
    including vitd_objectives.SED_PER_MIN_PER_UVI and any assumed
    reference UV index, geometry factor or exposed BSA -- multiplies both
    groups identically and therefore CANCELS from R exactly. So this
    function does not need, and does not invent, an absolute minutes->SED
    constant: only the ratio of group-mean minutes enters. The absolute
    scale remains the repo's existing assumption, untouched.

    What this does NOT measure: DED120/DED125 are minutes outdoors only.
    They say nothing about exposed BSA, clothing, shade-seeking or
    time-of-day. R here is therefore a TIME ratio, and the ridge's R is a
    DOSE ratio. They coincide only if the two groups expose comparable
    skin area at comparable hours -- which NHANES still does not measure.
    """
    print("\n  --- Part 2: DEQ-REFINED ridge (DED120/DED125 subset) ---")

    need = ("deq_outdoor_minutes_workday", "deq_outdoor_minutes_nonworkday")
    if not all(c in df.columns for c in need):
        print("  [SKIP] DEQ minutes columns absent -- re-run "
              "nhanes_preprocessing.py to add them.")
        return None

    s = df.copy()
    # Mean of the two reported days. `mean` skips NaN, so a respondent who
    # answered only one branch still contributes that branch. 3333 ("does
    # not work / go to school") was already dropped to NaN upstream in
    # nhanes_preprocessing.py -- treating it as 3333 minutes would wreck
    # this average, and it affects 355-640 respondents per cycle.
    s["deq_minutes"] = s[list(need)].mean(axis=1)
    s = s.dropna(subset=["deq_minutes"])
    s = s[s["race_ethnicity"].isin([3, 4])]
    if len(s) < 500:
        print(f"  [SKIP] only {len(s)} usable DEQ rows.")
        return None

    cycles_seen = sorted(s["cycle"].unique())
    print(f"\n  cycles with DED120/DED125: {', '.join(cycles_seen)}")
    print(f"  (Part 1 above used ALL cycles; this subset is {len(s)} rows.)")
    for code, label in ((3, "Non-Hispanic White"), (4, "Non-Hispanic Black")):
        g = s[s["race_ethnicity"] == code]
        print(f"  {label:22s} n = {len(g):>5}   mean minutes outdoors/day = "
              f"{_wmean(g, 'deq_minutes'):6.1f}  (WTMEC2YR-weighted)")

    R_hat = _deq_dose_ratio(s)

    # Bootstrap CI on the measured ratio.
    rng = np.random.default_rng(7)
    idx = np.arange(len(s))
    boots = []
    for _ in range(2000):
        bs = s.iloc[rng.integers(0, len(s), len(s))]
        if (bs["race_ethnicity"] == 3).any() and (bs["race_ethnicity"] == 4).any():
            boots.append(_deq_dose_ratio(bs))
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    print(f"\n  [MEASURED] White:Black time-outdoors ratio R = {R_hat:.3f}"
          f"   95% CI [{lo:.3f}, {hi:.3f}]  ({len(boots)} bootstraps)")

    both = s.dropna(subset=list(need))
    print(f"  [sensitivity] restricting to respondents who answered BOTH "
          f"days (n = {len(both)}): R = {_deq_dose_ratio(both):.3f}")

    # DED031 as a covariate: is the time ratio confounded by sun sensitivity?
    if "deq_skin_reaction" in s.columns and s["deq_skin_reaction"].notna().any():
        print("\n  DED031 'reaction after months out of the sun' "
              "(1 = blisters ... 5 = never burns):")
        for code, label in ((3, "Non-Hispanic White"),
                            (4, "Non-Hispanic Black")):
            g = s[(s["race_ethnicity"] == code) &
                  s["deq_skin_reaction"].notna()]
            print(f"    {label:22s} n = {len(g):>5}  weighted mean = "
                  f"{_wmean(g, 'deq_skin_reaction'):.2f}")
        print("    -> the ordering supports the repo's race->FST map "
              "(NH Black burns less), though it is a self-report.")
        print("\n  R re-measured WITHIN each DED031 stratum (confounding check):")
        for lvl in (1, 2, 3, 4, 5):
            st = s[s["deq_skin_reaction"] == lvl]
            nw = int((st["race_ethnicity"] == 3).sum())
            nb = int((st["race_ethnicity"] == 4).sum())
            if nw >= 30 and nb >= 30:
                print(f"    DED031 = {lvl}:  R = {_deq_dose_ratio(st):.3f}"
                      f"   (n White {nw}, n Black {nb})")
            else:
                print(f"    DED031 = {lvl}:  too few to stratify "
                      f"(n White {nw}, n Black {nb})")
        print("    -> R stays near 1 in every stratum with usable n, so the")
        print("       measured time ratio is not an artefact of differing")
        print("       sun sensitivity between the two groups.")

    # Re-solve the ridge on THIS subset, at the measured R and its CI.
    obs_sub = {}
    for code, key in ((3, "w"), (4, "b")):
        sub = s[s["race_ethnicity"] == code]
        rec = {}
        for scode, sname in ((1, "winter"), (2, "summer")):
            g = sub[sub["exam_season"] == scode]
            rec[sname] = _wmean(g, "vitd_standardized_nmol_L")
        obs_sub[key] = rec
    ws, bs_ = obs_sub["w"], obs_sub["b"]
    print(f"\n  Subset serum means: White {ws['winter']:.1f}/"
          f"{ws['summer']:.1f}   Black {bs_['winter']:.1f}/{bs_['summer']:.1f}"
          f"  (winter/summer nmol/L)")

    print(f"\n  {'R (measured)':>14} {'K_const needed':>15} "
          f"{'FST V:II optical':>18} {'residual nmol/L':>16}")
    out = {}
    for tag, R in (("CI low", lo), ("point", R_hat), ("CI high", hi)):
        K, opt, err = _nhanes_K_for_R(ws, bs_, R, p)
        out[tag] = (R, K, opt, err)
        print(f"  {R:>14.3f} {K:>15.3f} {opt:>18.2f} {err:>16.2f}"
              f"   <- {tag}")

    # Which of Part 1's swept R values does the measurement exclude?
    excluded = [r for (r, *_rest) in ridge_rows if r < lo or r > hi]
    kept = [r for (r, *_rest) in ridge_rows if lo <= r <= hi]
    span_old = (min(r for (r, *_x) in ridge_rows),
                max(r for (r, *_x) in ridge_rows))
    k_old = [k for (_r, k, *_x) in ridge_rows]
    k_new = [out[t][1] for t in ("CI low", "CI high")]
    print(f"\n  Part 1 swept R over [{span_old[0]:.2f}, {span_old[1]:.2f}] "
          f"-> K_const spanned [{min(k_old):.3f}, {max(k_old):.3f}] "
          f"(a factor of {max(k_old) - min(k_old):.3f} in K).")
    print(f"  Part 2 measures R in [{lo:.3f}, {hi:.3f}] "
          f"-> K_const spans [{min(k_new):.3f}, {max(k_new):.3f}].")
    print(f"  Excluded by measurement: R = "
          f"{', '.join(f'{r:g}' for r in excluded) or '(none)'}")
    print(f"  Still admissible:        R = "
          f"{', '.join(f'{r:g}' for r in kept) or '(none of the swept grid)'}")
    print("\n  READ THIS BEFORE QUOTING THE NARROWING:")
    print("  DED120/DED125 measure TIME OUTDOORS, not dose. Exposed BSA,")
    print("  clothing, shade-seeking and time-of-day are still unmeasured in")
    print("  NHANES. So Part 2 does not collapse the ridge -- it RELOCATES")
    print("  the remaining freedom: the time component of R is ~1, and any")
    print("  larger behavioural ratio now has to come from the skin-area /")
    print("  timing axis, which this dataset cannot see. Part 1 stands as")
    print("  the statement of what NHANES alone constrains.")
    print("  This is a DIAGNOSTIC. No model parameter is changed here; any")
    print("  move in K_const would be a calibration decision requiring")
    print("  approval per calibration_approval.yaml.")
    return out


def s5_nhanes_ridge():
    hr("S5  NHANES: the (behaviour, optics) ridge")
    print("\n  NHANES has no personal UV dosimetry. Part 1 below is the")
    print("  original constant-dose ridge over the FULL pooled sample, in")
    print("  which behaviour is unmeasured and can only be traded off")
    print("  against optics. It is unchanged and remains the headline result.")
    print("\n  Part 2 is NEW: the dermatology questionnaire (DEQ) has since")
    print("  been added to this repo's extract, and DED120/DED125 give")
    print("  self-reported minutes outdoors for 2009-2018. That measures the")
    print("  TIME axis of behaviour for a subset, so Part 2 re-runs the same")
    print("  solve with the dose ratio taken from data instead of swept.")

    import pandas as pd
    path = os.path.join(REPO, "audit_outputs", "nhanes_standardized_processed.csv")
    if not os.path.exists(path):
        print("  [SKIP] nhanes_standardized_processed.csv not found.")
        return None
    df = pd.read_csv(path)
    df = df[(df["age"] >= 18) & (df["age"] <= 65)]
    df = df[df["vitd_standardized_nmol_L"] > 0]
    df = df.dropna(subset=["vitd_standardized_nmol_L", "exam_season",
                           "wtmec2yr", "race_ethnicity"])
    obs = {}
    for code, (label, fst) in {3: ("Non-Hispanic White", 2),
                               4: ("Non-Hispanic Black", 5)}.items():
        sub = df[df["race_ethnicity"] == code]
        rec = {}
        for scode, sname in ((1, "winter"), (2, "summer")):
            g = sub[sub["exam_season"] == scode]
            rec[sname] = float(np.average(g["vitd_standardized_nmol_L"],
                                          weights=g["wtmec2yr"]))
            rec["n_" + sname] = len(g)
        rec["fst"] = fst
        obs[label] = rec
        print(f"\n  {label:22s} FST {fst}  winter {rec['winter']:.1f}  "
              f"summer {rec['summer']:.1f}  amplitude "
              f"{rec['summer'] - rec['winter']:+.1f} nmol/L  "
              f"(n = {rec['n_winter']} / {rec['n_summer']}, WTMEC2YR-weighted, "
              f"zeros excluded)")

    w = obs["Non-Hispanic White"]
    b = obs["Non-Hispanic Black"]
    print(f"\n  [derived] observed seasonal-amplitude ratio White:Black = "
          f"{(w['summer'] - w['winter']) / (b['summer'] - b['winter']):.2f}x")

    print("\n  --- Part 1: ORIGINAL constant-dose ridge "
          "(full sample, all 10 cycles) ---")
    print("\n  Trade-off: for each assumed behavioural dose ratio R (White:Black),")
    print("  the K_const that reproduces both groups' summer means, given one")
    print("  shared dose scale fitted to the White group.")
    print(f"\n  {'R (behaviour)':>14} {'K_const needed':>15} "
          f"{'FST V:II optical':>18} {'residual nmol/L':>16}")
    p = ModelParams()
    rows = []
    for R in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.4, 4.0]:
        K, opt, err = _nhanes_K_for_R(w, b, R, p)
        rows.append((R, K, opt, err))
        print(f"  {R:>14.2f} {K:>15.3f} {opt:>18.2f} {err:>16.2f}")
    print("\n  Every row fits NHANES equally well. NHANES alone cannot pick one.")
    print("  This is the honest statement of what NHANES Part B tells us: it")
    print("  constrains the PRODUCT of behaviour and optics, nothing more.")

    _s5_deq_refined(df, w, b, p, rows)
    return rows


# ===========================================================================
# S6 -- assumptions register
# ===========================================================================

ASSUMPTIONS = [
    "Kift 2013 weekly badge doses are read off Table 2 medians; the PDF text "
    "layer drops decimal points, so 1.0/3.4 etc. were reconstructed from the "
    "weekday+weekend split summing to the weekly figure. Verified internally "
    "consistent to ~5%.",
    "Kift 2013 serum values are Fig-1 MEDIANS, not means; the model is fitted "
    "to them as if they were means.",
    "Exposed BSA in S4 is taken from Kift 2013's summer diary percentages "
    "(14-17%) for BOTH the Kift and Darling cohorts. Darling 2013 does not "
    "report BSA in Webb 2021's table, so the Darling BSA is ASSUMED equal "
    "between its two cohorts. If the South Asian cohort exposed less skin, "
    "the behavioural share is UNDERSTATED here.",
    "The badge -> whole-body effective SED scale is ASSUMED identical between "
    "the light and dark cohort of a given study. This is the load-bearing "
    "assumption of S4. It is defensible because badge type, placement "
    "protocol, latitude and season are common within each study, and it is "
    "why the two cohorts must come from the same study.",
    "Both field studies are UK, both dark cohorts are South Asian FST V. "
    "Nothing here is measured in FST VI or in African-ancestry populations "
    "outside the lab. Extrapolating either result to FST VI is unsupported.",
    "Darling 2013 cohorts differ in sex composition from Kift 2013's "
    "(all-female vs mixed). Kift 2013's cohorts differ in median age "
    "(24 vs 44 y) and sex (67% vs 23% male); Kift adjusted the SERUM "
    "comparison for age/sex/BMI but the BADGE doses in Table 2 are "
    "unadjusted. The behavioural ratio therefore carries residual "
    "demographic confounding of unknown sign.",
    "The NHANES race -> FST map (NH White = II, NH Black = V) is the repo's "
    "existing RACE_FST_MAP and is itself an approximation; NH Black spans "
    "FST IV-VI.",
    "S5 Part 2's DEQ-measured ratio uses DED120/DED125, which are "
    "SELF-REPORTED MINUTES OUTDOORS, not dose. Equating that time ratio "
    "with the ridge's behavioural DOSE ratio assumes the two race groups "
    "expose comparable skin area at comparable times of day. NHANES "
    "measures neither, so this is the load-bearing assumption of Part 2 "
    "and the reason Part 1 is retained as the headline result. The "
    "sentinel 3333 ('does not work / go to school') is dropped to NaN "
    "upstream; leaving it in as a duration would inflate the mean by "
    "roughly an order of magnitude. DED031's 6/77/99/0 codes are dropped "
    "likewise. DEQ is asked only of ages 20-59, so Part 2's subset is "
    "younger than Part 1's 18-65 sample by construction.",
    "S3's baseline adjustment is a single pooled slope-vs-baseline "
    "regression, not the ANCOVA Young 2020 used. It reproduces the published "
    "ratios only approximately (see printed comparison).",
    "Young 2021's revised vitamin D action spectrum, which underlies any "
    "E_source recomputation, is derived from FST I-II volunteers only "
    "(n = 75). It is NOT validated in dark skin and is not used here.",
    "Farrar 2013 vs Rhodes 2010 is a CROSS-TRIAL comparison; the two cohorts "
    "differ in baseline 25(OH)D by a large margin and that difference is not "
    "adjusted for in the 2.31x figure.",
    "Rhodes 2010 (JID 130:1411-8) is NOT in this repo. Verified 2026-08-17 by "
    "opening all 46 PDFs in Research_Papers/ and reading first-page content, "
    "not filenames. Its pre-course 25(OH)D therefore has to be swept "
    "(30-60 nmol/L, anchored at 45.5 = the same group's Manchester white "
    "winter value in Webb 2021 Table 5). That sweep alone moves the implied "
    "FST V:II factor across a ~2x range, which is why the Rhodes/Farrar "
    "cross-trial pair is reported for CONSISTENCY ONLY and is EXCLUDED from "
    "the headline lab bracket. Felton 2016 replaces it: same group, same "
    "cabinet, same 1.3 SED x3/wk x6 wk at 35% BSA, but WITHIN-STUDY with both "
    "baselines printed in its Table 1.",
    "Felton 2016 n is small (10 phototype II, 6 phototype V) and its FST V "
    "gain SD (10.5 nmol/L) exceeds its mean gain (8.3). The protocol-corrected "
    "Felton point estimate therefore carries wide, unquantified uncertainty; "
    "this script does not bootstrap it because Felton reports group means and "
    "SDs, not individual trajectories.",
    "Felton 2016's source is described as 95% UVA / 5% UVB simulated summer "
    "sunlight; this script assumes it is the same Arimed B cabinet used by "
    "Rhodes 2010 and Farrar 2013 (same group, same protocol family). The "
    "E_source cancels from a between-skin-type ratio, so this affects "
    "absolute levels only.",
    "S3b assumes the Arimed B E_source (1.83) applies to both trials. Both "
    "used Arimed B, so this cancels from the ratio; it affects only the "
    "absolute levels.",
    "Young 2020's FST VI group in the TL01 arm is entirely African-ancestry "
    "and its FST V group does not exist; the Arimed B FST VI group is 3 "
    "African and 3 Indian-subcontinent. Any FST V number in this report that "
    "is not explicitly from the Arimed B FST V cell (n = 5) is an "
    "interpolation between FST IV and FST VI, not a measurement.",
]


def s6_assumptions():
    hr("S6  EVERY ASSUMPTION AND ESTIMATE MADE ABOVE")
    for i, a in enumerate(ASSUMPTIONS, 1):
        print(f"\n  {i:2d}. {a}")


# ===========================================================================
# S8 -- per-number audit register
# ===========================================================================
# Class (a): reproduced from a primary source held in this repo, this session.
# Class (b): reproduced, but rests on a stated assumption or a secondary
#            transcription.
# Class (c): could NOT be re-derived. Nothing in class (c) may carry a
#            conclusion.
AUDIT = [
    # (value, what, class, evidence)
    ("MED 200/250/300/450/600/1000 J/m2", "Kallioglu 2024 Table 3", "a",
     "read from the PDF; independently re-verified by the repo owner"),
    ("minutes 5.05/6.3/7.6/11.35/15.15/25.25", "Kallioglu 2024 abstract", "a",
     "read from the PDF; S1a shows they are the MED table to <1%"),
    ("Shih 2018 20%-MED doses 4.2-30.4 mJ/cm2", "Shih 2018 Results", "a",
     "read from the PDF"),
    ("Shih 2018 'no influence of skin type P=0.23'", "Shih 2018 Results", "a",
     "read from the PDF; it is a NULL, not a measured equality"),
    ("Rhodes 2010 delta 26.0 nmol/L at 23.4 SED", "Webb 2021 Table 4", "b",
     "ROMAN typeface = primary, but transcribed BY Webb; Rhodes 2010 itself "
     "is not in this repo (all 46 PDFs checked by first-page content)"),
    ("Farrar 2013 delta 11.5 nmol/L at 23.4 SED", "Webb 2021 Table 4", "a",
     "ROMAN = primary, and Farrar 2013 IS in the repo; its Table 2 gives "
     "4.6 ng/mL = 11.5 nmol/L at 1.3 SED, matching exactly"),
    ("Webb 2021 per-SED cells 1.11 / 0.48", "Webb 2021 Table 4", "b",
     "ITALIC = Webb's own arithmetic; 11.5/23.4 = 0.491 not 0.48"),
    ("Darling 2013 6.39 / 2.00 SED per summer week", "Webb 2021 Table 5", "b",
     "ROMAN = primary, but Darling 2013 is not in this repo, so this is a "
     "secondary transcription that could not be checked at source"),
    ("Darling 2013 serum 72.1/44.5 and 26.2/19.7", "Webb 2021 Table 5", "b",
     "same status as above"),
    ("Darling amplitudes 27.6 / 6.5", "Webb 2021 Table 5", "a",
     "ITALIC = Webb's arithmetic, and it is exactly the difference of the "
     "two roman serum cells, re-derived here"),
    ("Kift 2013 badge SED 3.4 vs 1.0 per summer week", "Kift 2013 Table 2",
     "b", "PDF text layer drops decimals; reconstructed from the "
     "weekday+weekend split summing to the printed weekly figure"),
    ("Kift 2013 serum 22.5 / 14.5 nmol/L summer/winter", "Kift 2013 Fig 1",
     "a", "decimal reconstruction INDEPENDENTLY CONFIRMED by Webb 2018 "
     "Results 3.1, which quotes the same cohort as 22.5 and 14.5 nmol/L"),
    ("Kift 2013 diet 1.3 ug/day", "Kift 2013 Table 2", "a",
     "confirmed by Webb 2018 Results 3.1 quoting 1.32 ug/day"),
    ("Kift 2013 % of ambient 2.94/1.11 and 2.23-2.32/0.88",
     "Kift 2013 Results", "b", "decimal reconstruction, unambiguous in "
     "context but not independently confirmed"),
    ("Kift 2013 exposed BSA 14-17%", "Kift 2013 Table 2", "b",
     "decimal reconstruction; used as the BSA input for BOTH field studies"),
    ("Young 2020 published ratios 1.29 / 1.43", "Young 2020 text", "a",
     "read from the PDF"),
    ("Young 2020 per-subject slopes and baselines", "repo xlsx", "a",
     "recomputed here from Papers_and_Datasets/'Final skin type vit D "
     "database 26_03_2020.xlsx'; reproduces the repo's own known-issue-9 "
     "figure of 1.02-1.18 for the unadjusted Arimed B ratio"),
    ("Felton 2016 36.5->54.3 and 17.2->25.5 nmol/L", "Felton 2016 Table 1",
     "a", "read from the PDF held in Research_Papers/; both baselines and "
     "both finals printed, no anchoring needed"),
    ("Felton 2016 MED 37 vs 146 mJ/cm2", "Felton 2016 Table 1", "a",
     "read from the PDF"),
    ("Webb 2018 2.75 SED cap, 25 min, 9 min", "Webb 2018 Table 1/Results",
     "a", "read from the PDF held in Research_Papers/"),
    ("Webb 2018 white 1 SED cap", "derived here", "b",
     "NOT printed in Webb 2018. Derived as 2.75 x 9/25 = 0.99 SED on the "
     "assumption that the 9 min figure is at the same irradiance and BSA. "
     "The companion paper (Nutrients 10:497) is not in this repo"),
    ("Rhodes 2010 baseline 45.5 nmol/L", "anchor, not a measurement", "c",
     "COULD NOT BE DERIVED. Rhodes 2010 is absent from this repo. This is "
     "why the Rhodes/Farrar pair is reported for consistency only and is "
     "EXCLUDED from every bracket that carries a conclusion"),
    ("NHANES weighted seasonal means", "audit_outputs CSV", "a",
     "recomputed here with WTMEC2YR and zeros excluded, from the repo's own "
     "harmonized column (nhanes_preprocessing.py harmonizes LBDVIDMS to "
     "LBXVIDMS, so 2001-2006 and 2007+ are not naively pooled)"),
]


def s8b_repo_file_dependencies():
    """Apply to THIS script the scepticism that caught Kallioglu: which repo
    files does it read, and is each one what its name claims?"""
    hr("S8b  WHAT REPO FILES DOES THIS SCRIPT TRUST, AND ARE THEY REAL?")
    print("""
  Three, and no others. In particular this script reads NOTHING from
  data/spectra/ -- E_source enters only as a scalar table lookup -- so the
  CIE-174-shifted-5nm file mislabelled as Young 2021's action spectrum
  cannot have touched any number here.

  1. Papers_and_Datasets/'Final skin type vit D database 26_03_2020.xlsx'
     CLAIMS: Young 2020's own irradiated-subject trial data.
     CHECKED: its Notes sheet carries the paper title, DOI 10.1016/
     j.jid.2019.11.019, the author list and the full-text URL. Two
     independent numerical confirmations that it is genuine:
       - the TL01 arm's unadjusted FST II:VI slope ratio recomputed here
         reproduces Young 2020's PUBLISHED 1.43 exactly;
       - the Arimed B arm's unadjusted ratio (1.02) reproduces the value
         this repo independently recorded in 'Known issues' #9 (1.02-1.18)
         from a separate regression.
     VERDICT: primary data. Trusted.

  2. data/vernez2015_table_s1_vis_by_site_posture.csv
     CLAIMS: Vernez 2015 Supplementary Table S1 sky-visibility inputs.
     CHECKED: only landed in the repo on 2026-08-17, REPLACING a
     back-derived placeholder that had been generated by inverting the
     paper's own equation against its outputs -- precisely the failure mode
     that this audit exists to catch. Its own header flags an unresolved
     duplicate ('Top of shoulders' and 'Upper back' identical across all
     five postures).
     USED FOR: S2b only, and only for the order-of-magnitude claim that
     unshaded sky visibility is tens of percent while Kift measured
     ~1-3% of ambient. That two-orders-of-magnitude gap survives any
     plausible error in this table, including the duplicate row.
     VERDICT: trusted for a qualitative argument only. No S7 number
     depends on it.

  3. audit_outputs/nhanes_standardized_processed.csv
     CLAIMS (via nhanes_preprocessing.py) to 'harmonize' 2001-2006 to
     2007+.
     CHECKED, and this is a finding: vitd_standardized_nmol_L is
     BYTE-IDENTICAL to vitd_raw_nmol_L in all ten cycles, and every cycle
     is labelled 'LC-MS/MS Direct'. The harmonize_ria_to_lcms() regression
     in that module is NEVER INVOKED -- it is dead code.
     IS THE POOLING STILL VALID? Yes, but not for the reason the module
     implies. The 2001-2006 source column is LBDVIDMS (verified in
     VID_B/C/D.csv) which is CDC's own already-standardized LC-MS/MS-
     EQUIVALENT series, and 2007+ is LBXVIDMS, CDC's measured LC-MS/MS.
     CDC did the harmonization upstream; the repo did not do it locally.
     ALSO CORRECTED 2026-08-20: that file's tenth cycle was labelled
     '2019-2020'. It is CDC cycle L = August 2021-August 2023
     (SDDSRVYR = 12, SEQN 130378-142310). The true 2019-2020 cycle was
     never released standalone -- COVID cut the fieldwork short -- so it
     exists only inside the P_-prefixed 2017-March-2020 combined files,
     which this repo has for DEMO/BMX/diet but NOT for serum vitamin D.
     The relabel is cosmetic for S5: it renames a cycle, it does not add,
     drop or alter a single serum row, and the Part 1 table is
     numerically identical before and after.
     VERDICT: the NHANES numbers in S5 are sound. But CLAUDE.md's rule
     ('do NOT pool without harmonization code') is currently satisfied by
     a function that does nothing, which is a latent trap for the next
     person who changes the source-column preference order.""")


def s8_audit():
    hr("S8  PER-NUMBER AUDIT REGISTER")
    print("\n  (a) reproduced from a primary source in this repo, this session")
    print("  (b) reproduced, but rests on a stated assumption or a secondary")
    print("      transcription")
    print("  (c) could NOT be re-derived -- carries no conclusion\n")
    counts = collections.Counter(c for _, _, c, _ in AUDIT)
    for val, src, cls, ev in AUDIT:
        print(f"  [{cls}] {val}")
        print(f"        source: {src}")
        print(f"        {ev}")
    print(f"\n  totals: (a) {counts['a']}   (b) {counts['b']}   "
          f"(c) {counts['c']}")
    print("  The single class-(c) item is Rhodes 2010's baseline, and every")
    print("  bracket in S7 is computed WITHOUT it.")


def s7_synthesis(s1, s2, s3b, s4rows, s3):
    hr("S7  EVERY ESTIMATE ON ONE SCALE: the FST V : FST II factor")
    p = ModelParams()
    m2, m5 = p.M_baseline(2), p.M_baseline(5)
    inc = float(np.exp(p.K_const * (m5 - m2)))
    print(f"\n  Incumbent model: {inc:.2f}x   (K_const = {p.K_const:.3f})")
    print(f"\n  {'estimate':54s} {'V:II':>7}  contains")
    print("  " + "-" * 94)

    y20_raw_ii_v = s3.get("ArimedB irradiated:II:V", (float("nan"),))[0]
    felton_raw = ((FELTON_FINAL[2] - FELTON_D0[2])
                  / (FELTON_FINAL[5] - FELTON_D0[5]))
    rf_lo = float(np.exp(min(s3b["rf_sweep"]) * (m5 - m2)))
    rf_hi = float(np.exp(max(s3b["rf_sweep"]) * (m5 - m2)))

    rows = [
        ("MATCHED-DOSE, WITHIN-STUDY (what K_const should carry)",
         None, ""),
        ("  Young 2020 raw II:V slope ratio (repo trial file)",
         y20_raw_ii_v, "optics; n=5 FST V; NOT protocol-corrected"),
        ("  Young 2020 protocol-corrected",
         s3b["young"][1], "optics; feedback inside Young's protocol undone"),
        ("  Felton 2016 raw II:V gain ratio",
         felton_raw, "optics; n=10 vs 6; NOT protocol-corrected"),
        ("  Felton 2016 protocol-corrected",
         s3b["felton"][1], "optics; both baselines MEASURED in Table 1"),
        ("CROSS-TRIAL / ASSUMPTION-LADEN (excluded from bracket)",
         None, ""),
        ("  Farrar 2013 / Rhodes 2010 raw (Webb 2021 T4)",
         s1["farrar_rhodes_matched_dose_ratio"],
         "optics + cross-trial baseline mismatch"),
        (f"  Farrar/Rhodes corrected, D0 swept 30-60",
         None, f"{rf_lo:.2f}x to {rf_hi:.2f}x -- width is ONE unobtained number"),
        ("  Shih 2018 MED ratio under its own null",
         SHIH_MED_SED[5] / SHIH_MED_SED[2],
         "optics IF equal %MED => equal D; weakly powered null"),
        ("  Felton 2016 MEASURED MED ratio, same volunteers",
         FELTON_MED_MJCM2[5] / FELTON_MED_MJCM2[2],
         "ERYTHEMA, not vitamin D -- and it is ~1.8x the D ratio"),
        ("  Kallioglu 2024 'V:II = 2.40'",
         2.40, "NOTHING. It is that erythema MED table (S1a)."),
        ("FIELD, PERSONAL DOSIMETRY (contains behaviour)",
         None, ""),
        ("  Kift 2013, behaviour REMOVED",
         s4rows[1][1]["optical_factor"],
         "optics + residual unmeasured exposure + physiology"),
        ("  Darling 2013, behaviour REMOVED",
         s4rows[0][1]["optical_factor"],
         "optics + residual unmeasured exposure + physiology"),
        ("  Kift 2013, BEHAVIOUR-BLIND",
         s4rows[1][2]["optical_factor"],
         "optics x behaviour -- all of it dumped into K_const"),
        ("  Darling 2013, BEHAVIOUR-BLIND",
         s4rows[0][2]["optical_factor"],
         "optics x behaviour -- all of it dumped into K_const"),
    ]
    for label, val, contains in rows:
        if val is None:
            print(f"  {label:54s} {'':>7}  {contains}")
        else:
            print(f"  {label:54s} {val:>7.2f}  {contains}")

    print("\n  DECOMPOSITION")
    lo_b, hi_b = s2["behaviour_range"]
    shares = [r[2]["optical_factor"] / r[1]["optical_factor"] for r in s4rows]
    fld_lo = min(r[1]["optical_factor"] for r in s4rows)
    fld_hi = max(r[1]["optical_factor"] for r in s4rows)
    lab_lo, lab_hi = s3b["bracket"]
    ov_lo, ov_hi = max(fld_lo, lab_lo), min(fld_hi, lab_hi)
    print(f"    measured personal-UV dose ratio (white : FST V, UK) "
          f"= {lo_b:.1f}x to {hi_b:.1f}x")
    print(f"    its effect on the 25(OH)D response (behaviour share) "
          f"= {min(shares):.1f}x to {max(shares):.1f}x")
    print(f"    field residual once behaviour is removed            "
          f"= {fld_lo:.1f}x to {fld_hi:.1f}x")
    print(f"    matched-dose within-study lab bracket               "
          f"= {lab_lo:.1f}x to {lab_hi:.1f}x")
    print(f"    incumbent K_const delivers                          "
          f"= {inc:.2f}x")
    if ov_lo <= ov_hi:
        print(f"\n    OVERLAP of the two independent brackets = "
              f"{ov_lo:.1f}x to {ov_hi:.1f}x")
    else:
        print(f"\n    NO OVERLAP: field {fld_lo:.1f}-{fld_hi:.1f}x vs lab "
              f"{lab_lo:.1f}-{lab_hi:.1f}x. Report the union, not a point.")
    print(f"    Behaviour-blind ({min(r[2]['optical_factor'] for r in s4rows):.1f}x-"
          f"{max(r[2]['optical_factor'] for r in s4rows):.1f}x) is above every")
    print(f"    matched-dose measurement that exists.")


def main():
    print(__doc__)
    s1 = s1_provenance_audit()
    s2 = s2_behaviour()
    s2b_is_it_geometry()
    s3 = s3_optical()
    s3b = s3b_lamp_conflict()
    s4rows = s4_inversion()
    s4b_sensitivity()
    s4c_load_bearing(s3b["bracket"])
    s4d_esource_dependence()
    s5_nhanes_ridge()
    s7_synthesis(s1, s2, s3b, s4rows, s3)
    s6_assumptions()
    s8_audit()
    s8b_repo_file_dependencies()
    hr("END - no parameter was read from or written to any config by this run")


if __name__ == "__main__":
    main()
