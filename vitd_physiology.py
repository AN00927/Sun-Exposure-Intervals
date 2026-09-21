"""
vitd_physiology.py
==================
Mechanistic sub-models added July 2026 after an external-literature pass.

Each function replaces something the core model previously handled with a
phenomenological stand-in. Opt-in via ModelParams flags so every change is
separately ablatable — a mechanism that cannot be switched off cannot be
shown to matter.

Contents
--------
1. body_weight_dilution_factor  -- volumetric dilution by body size
2. photodegradation_factor      -- previtamin-D3 photostationary plateau
2b. personal_geometry_factor    -- ambient horizontal dose -> personal anatomical
                                   dose, via Vernez 2015 (simulated, sky
                                   visibility) or a measured per-activity ERTA
                                   table (Schmalwieser 2025, jogging)
3. hydroxylation_efficiency     -- biphasic 25-hydroxylation (replaces D_max)
4. cancer_risk                  -- SCC (cumulative) + BCC/melanoma (intermittent)

References are given inline with the numbers actually used.
"""

from __future__ import annotations

import csv
import os
import re
import warnings
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Volumetric dilution by body size
# ---------------------------------------------------------------------------
# Drincic 2012 (Obesity 20:1444-8, N=686) tested linear vs hyperbolic fits of
# 25(OH)D against body size. HYPERBOLIC fit on TOTAL BODY WEIGHT removed the
# entire obesity-related inter-individual variance, beating every linear model
# AND every BMI-based model. So:
#   - right covariate is body weight, not BMI
#   - right functional form is 1/weight (dilution volume), not a linear offset
# Mechanism is dilution of fixed vitamin D into a larger distribution volume,
# not sequestration (Drincic 2012; Walsh 2017 review;
# Pannu 2016 meta-regression shows weight loss raises 25(OH)D).
#
# Reference weight 70 kg. A 100 kg adult gets 0.70x the 25(OH)D rise per unit
# of cutaneous synthesis; a 55 kg adult gets 1.27x.
#
# Cross-check against oral dosing (effect should be the same regardless of
# D3 route): Ekwaru 2014 (N=17,614) found obese subjects need 2-3x and
# overweight 1.5x the dose of normal-weight subjects for the same 25(OH)D.
# Weight ratio 100/70 = 1.43x dilution here; BMI 35 vs 22 at fixed height
# implies ~1.6x. Same order, slightly conservative.
BODY_WEIGHT_REF_KG: float = 70.0


def body_weight_dilution_factor(weight_kg: float | None,
                                ref_kg: float = BODY_WEIGHT_REF_KG) -> float:
    """Multiplicative factor on 25(OH)D gain per unit of vitamin D input.

    Hyperbolic in total body weight (Drincic 2012). Returns 1.0 when weight
    is unknown or equals the reference, so existing scenarios are unchanged.
    """
    if weight_kg is None or weight_kg <= 0:
        return 1.0
    return ref_kg / float(weight_kg)


# ---------------------------------------------------------------------------
# 2. Previtamin-D3 photodegradation plateau
# ---------------------------------------------------------------------------
# Holick 1981 (Science 211:590-3): during prolonged simulated-solar exposure,
# previtamin-D3 rises to a PLATEAU at 10-15% of original 7-DHC content,
# after which further UV photoisomerises it to inert lumisterol3 and
# tachysterol3. Pigmentation and latitude change TIME to plateau but NOT
# plateau height — a photostationary state, and the reason sun exposure
# cannot be banked in one long session.
#
# van Dijk 2016 (Photochem Photobiol Sci) quantifies the consequence: net
# production is "largely cancelled" by ~7.5 SED on previously unexposed skin.
# Their modified-QUT spectrum turns negative for longer UVB wavelengths after
# as little as 1.25 SED.
#
# Model: per-session saturating factor on synthesis,
#     f(sed) = (1 - exp(-sed / D_SAT)) / (sed / D_SAT)
# average of exp(-s/D_SAT) over the session — instantaneous yield decays
# exponentially as the substrate pool approaches its photostationary state.
# f -> 1 as sed -> 0 (no change at low dose) and f ~ D_SAT/sed at high dose
# (yield plateaus in ABSOLUTE terms).
#
# D_SAT = 7.5 SED anchors to van Dijk's "net production largely cancelled at
# 7.5 SED" on unexposed skin. [PRIOR] Single-source; van Dijk is a modelling
# paper, not a serum-response measurement.
PREVITAMIN_D_SAT_SED: float = 7.5


def photodegradation_factor(sed: float, d_sat: float = PREVITAMIN_D_SAT_SED) -> float:
    """Session-averaged yield reduction from previtamin-D3 photodegradation.

    Returns 1.0 for vanishing dose and falls as ~d_sat/sed for large dose,
    so total per-session yield saturates instead of growing without bound.

    This mechanism makes splitting a UV budget across sessions genuinely
    better than taking it in one hit — a real interior optimum, unlike the
    feedback-driven artifact that
    `vitd_optimization.gross_synth_per_SED_diagnostic` recomputes on every
    pipeline run.
    """
    if sed <= 0.0:
        return 1.0
    x = sed / d_sat
    if x < 1e-8:
        return 1.0
    return float((1.0 - np.exp(-x)) / x)


# ---------------------------------------------------------------------------
# 2b. Personal anatomical dose geometry: the Vernez 2015 exposure-ratio model
# ---------------------------------------------------------------------------
# REPLACED 2026-08-16. The previous form here was `0.15 + 0.30*cos(theta)`,
# attributed to Schmalwieser & Siani 2018 and Parisi 2000/2003. A full-text read
# (gap 26) established that no source supports THAT SPECIFIC FORM. It was
# carried as a PLACEHOLDER only because nothing better was obtainable.
# [CORRECTED 2026-08-20] Schmalwieser & Siani 2018 (full text now in
# Research_Papers/) is narrower than the previous comment here claimed: they do
# NOT conclude the elevation-dependence is undeliverable in general -- their
# Fig. 2 plots exactly that (face ERTA vs solar elevation) for activities they
# had data for. Their actual conclusion is coverage, not derivability: "for
# most activities, we still lack the possibility to predict the UV exposure of
# exposed body sites under any condition because ERTAs are not fully
# available" -- i.e. per-activity, per-site curves exist piecemeal, not a
# general-purpose one.
#
# [PARTIALLY CLOSED 2026-08-20] One of the two papers named here as worth
# acquiring is now IN THE REPO and IMPLEMENTED: Schmalwieser et al. 2025
# (Photochem Photobiol, jogging, solar elevation 20-60 deg) -- see the
# ACTIVITY-SPECIFIC MEASURED ERTA section further down this file, exposed as
# activity_erta() and as personal_geometry_factor(activity='jogging', ...).
# So elevation dependence is no longer unsourced: for ONE activity, at seven
# body sites, it is measured.
#
# WHAT IS STILL OPEN, and why this is "partially", not "closed":
#   - ONE activity out of the many people actually do, from ONE subject. The
#     other paper named here, Weihs et al. 2024 (Atmosphere, cycling, SZA up to
#     65 deg), is still NOT in the repo and would be the obvious next
#     acquisition.
#   - NOT TRANSFERABLE ACROSS ACTIVITIES. Weihs et al. 2013 (walking/sitting/
#     lying, same city, overlapping authors, now in the repo) reports a
#     shoulders:forehead contrast of 1.7x while walking, against 2.5x for
#     jogging over the same solar elevations -- and a different site ordering
#     once the posture changes, the lower legs taking the lead while lying and
#     sitting up. Schmalwieser & Siani's coverage objection therefore stands in
#     full: what exists is still piecemeal per-activity curves, not a general
#     model. [CORRECTED 2026-08-20] this bullet previously claimed a "5x"
#     disagreement, taken from a Figure 6 digitization since withdrawn -- see
#     the WEIHS et al. 2013 block further down.
#   - THE FLAT FALLBACK IS UNCHANGED. personal_geometry_factor() with no
#     activity= still returns the 0.35 Cockell scalar; nothing about the default
#     model output moved.
#
# Vernez et al. 2015 (J Expo Sci Environ Epidemiol 25:113-118, doi
# 10.1038/jes.2014.6) is now IN THE REPO, full text with printed coefficients:
#   Research_Papers/A general model to predict individual exposure to
#   solar UV by.pdf
# It closes data request #13, which this repo twice recorded as "unobtainable".
#
# WHAT THE MODEL IS. A multivariable fractional polynomial regression fitted to
# the SimUVEx 3D-manikin simulation (8 body sites x 3 postures, n=8515),
# R^2 = 0.988; held out against 2 unseen postures (n=5672), R^2 = 0.972.
# Backward selection reduced 6 candidate predictors to just two: the fraction of
# sky visible from the body site, and the cosine of maximal daily solar zenith
# angle. Surface area, body-site zenith angle, vertical angle and shading
# scoring all dropped out.
#
# TWO PROPERTIES THAT MATTER MORE THAN THE FIT QUALITY:
#
#   1. Vis DOMINATES; solar geometry is a second-order trim. Across the full
#      SZA range at fixed Vis, ER moves only ~1.18x; across the Vis range at
#      fixed SZA it moves ~11.6x. The replaced form had elevation dependence and
#      NO posture argument, i.e. it modelled the small term and omitted the
#      large one. Vernez state the same thing: "The Vis parameter is largely
#      predominant in the regression model."
#
#   2. THESE ER VALUES ARE AN UNSHADED UPPER BOUND, and are much higher than the
#      measured personal-dosimetry range this repo cites elsewhere (Parisi
#      0.14-0.39, Cockell 0.35 +/- 0.12). Vernez say why: the model "predicts ER
#      assuming an unprotected skin, with no shading and a continuous exposure",
#      whereas real people are "exposed intermittently and be shaded or
#      partially shaded from surrounding elements". Their own comparison to
#      full-time farmers has the model at 15-55% for the face against 11-15%
#      measured -- they call this "overestimates exposure of farmers" and say ER
#      "should therefore be weighted taking into account the time spent in the
#      shade". So Vernez ER is the ceiling for deliberate unshaded exposure --
#      which IS this project's scenario -- and must NOT be read as a
#      like-for-like replacement for a behaviourally-averaged dosimeter number.
#
# Supplementary Table S1 (typical Vis per posture and body site, data request
# #13b) IS NOW IN THE REPO -- see VERNEZ_TABLE_S1_VIS / vis_table_s1() below.
# It is not in the main PDF; it was extracted 2026-08-17 from a supplementary
# .doc supplied separately. Vis is still never guessed by this module: callers
# must pass a vis_percent from vis_table_s1() (sourced) or their own estimate.
# The unresolved geometry x BSA coupling this table does not fix has no
# surviving write-up [reference purged 2026-09-19]; see
# vitd_scenario.LIMITATION_STATEMENT for how the pipeline freezes around it.

# Vernez et al. 2015, Equation 1. ER in PERCENT.
#   ER = -3.396*lnVis_cent + 10.714*Vis_cent - 9.199*cosSZA3_cent + 56.991
# with the centred variables defined in the paper as:
#   lnVis_cent   = ln(Vis/10) - 1.758
#   Vis_cent     = Vis/10 - 5.800
#   cosSZA3_cent = cos(SZA)**3 - 0.315
VERNEZ_B_LNVIS: float = -3.396
VERNEZ_B_VIS: float = 10.714
VERNEZ_B_COSSZA3: float = -9.199
VERNEZ_INTERCEPT: float = 56.991
VERNEZ_C_LNVIS: float = 1.758
VERNEZ_C_VIS: float = 5.800
VERNEZ_C_COSSZA3: float = 0.315

# Mean ER (%) reported in Vernez 2015 Results for named sites, standing erect,
# annual mean at Payerne (46.815 N). Anchors for validation, NOT model inputs.
VERNEZ_REPORTED_SITE_ER: Dict[str, float] = {
    "top_of_shoulders": 89.0,   # "unshaded, horizontally oriented surface"
    "back_of_neck": 65.0,       # "intermediate ... partially shaded by the head"
    "face_standing": 50.0,
    "face_annual_mean": 43.0,   # "a vertically oriented surface"
    "face_bowing": 20.0,
}

# ER (%) RANGES that Vernez state Eq. 1 produces "depending on the body posture
# and time of year" (Discussion, 'Consistency between the Model and Field-Based
# ER'). Added 2026-08-17 from a second copy of the paper.
#
# These are more useful than the point values above for this project, because a
# recommendation should carry an interval rather than a single number -- the
# site you expose and the posture you hold move the answer more than anything
# else in the dosimetry chain. Still the UNSHADED ceiling; see the caveat above.
#
# The paper's own field comparison, quoted so the overprediction stays visible:
# modelled face 15-55% against 11-15% measured in full-time farmers, which they
# describe as "overestimates exposure of farmers".
VERNEZ_REPORTED_SITE_ER_RANGE: Dict[str, Tuple[float, float]] = {
    "upper_back": (30.0, 87.0),
    "forearm_external": (44.0, 64.0),
    "face": (15.0, 55.0),
    # Stated as classes rather than named sites:
    "upper_horizontal_sites": (80.0, 100.0),   # e.g. shoulders, skull
    "vertical_sites": (20.0, 50.0),            # e.g. face, chest
}

# The 8 body sites Vernez actually modelled (Table 1). Recorded so that any
# future whole-body area weighting starts from the paper's own site set rather
# than an invented one.
VERNEZ_MODELLED_SITES: Tuple[str, ...] = (
    "face", "skull", "forearm_external", "upper_arm_external",
    "back_of_neck", "top_of_shoulders", "upper_back", "belly",
)


def vernez_er_interval(site: str) -> Tuple[float, float]:
    """Published ER interval (as FRACTIONS) for a named site or site class.

    Use this instead of a point geometry factor when reporting minutes to a
    person: the interval is the honest object. RAISES on unknown sites rather
    than guessing.
    """
    if site not in VERNEZ_REPORTED_SITE_ER_RANGE:
        raise KeyError(
            f"No published Vernez ER range for '{site}'. Known: "
            f"{sorted(VERNEZ_REPORTED_SITE_ER_RANGE)}. The full per-site table "
            f"is Supplementary Table S1 (data request #13b), not yet obtained."
        )
    lo, hi = VERNEZ_REPORTED_SITE_ER_RANGE[site]
    return lo / 100.0, hi / 100.0

# Reference SZA at which Vernez's centring makes the cosSZA3 term vanish, so
# ER == intercept == 56.991% at Vis = 58%. cos(SZA)**3 = 0.315.
VERNEZ_REF_SZA_DEG: float = float(np.degrees(np.arccos(VERNEZ_C_COSSZA3 ** (1.0 / 3.0))))


def vernez_exposure_ratio(vis_percent: float,
                          sza_deg: float | None = None) -> float:
    """Vernez 2015 Eq. 1 exposure ratio, returned as a FRACTION in [0, 1].

    vis_percent : fraction of sky visible from the body site, in PERCENT.
                  Dominant term. 58% is the model's centring point.
    sza_deg     : maximal daily solar zenith angle, degrees. If None, uses
                  VERNEZ_REF_SZA_DEG, where the SZA term is exactly zero.

    Returns the UNSHADED, continuous-exposure ratio -- an upper bound. See the
    module comment above before comparing it to dosimeter measurements.
    """
    vis = float(vis_percent)
    if not 0.0 < vis <= 100.0:
        raise ValueError(f"vis_percent must be in (0, 100], got {vis}")
    sza = VERNEZ_REF_SZA_DEG if sza_deg is None else float(sza_deg)
    sza = max(0.0, min(90.0, sza))

    ln_vis_cent = np.log(vis / 10.0) - VERNEZ_C_LNVIS
    vis_cent = vis / 10.0 - VERNEZ_C_VIS
    cos_sza3_cent = np.cos(np.radians(sza)) ** 3 - VERNEZ_C_COSSZA3

    er_pct = (VERNEZ_B_LNVIS * ln_vis_cent
              + VERNEZ_B_VIS * vis_cent
              + VERNEZ_B_COSSZA3 * cos_sza3_cent
              + VERNEZ_INTERCEPT)
    # The regression is unbounded; clip to physical range. ER > 100% is possible
    # in reality for a surface normal to the beam, but Eq. 1 is fitted to
    # down-welling-only simulation and Vernez do not report ER above ~100%.
    return float(min(1.0, max(0.0, er_pct / 100.0)))


def personal_geometry_factor(solar_elevation_deg: float | None = None,
                             vis_percent: float | None = None,
                             activity: str | None = None,
                             body_site: str | None = None) -> float:
    """Ratio of personal anatomical UV dose to ambient horizontal UV.

    Three mutually exclusive routes, in order of how much the caller knows:

      vis_percent supplied -> Vernez 2015 Eq. 1 (MODELLED, R^2 = 0.988; a
                            3D-manikin simulation, unshaded ceiling).
      activity supplied    -> the MEASURED ERTA table for that activity
                            (Schmalwieser 2025 for 'jogging'), interpolated at
                            solar_elevation_deg, for `body_site` or the
                            unweighted mean over the paper's measured sites.
                            Requires solar_elevation_deg.
      neither              -> 0.35, the Cockell 2001 measured central value for
                            upright activity. [PRIOR] Scalar, behaviourally
                            averaged, and it directly scales every minutes
                            recommendation, so sweep it first.

    Passing BOTH vis_percent and activity RAISES. They are two different
    estimates of the same ratio -- one simulated from sky visibility, one
    measured on a jogger -- and silently preferring either would hide which
    one produced the answer.

    [NOT A CALIBRATION OF THE DEFAULT] 2026-08-20: the activity route was added
    without moving the 0.35 fallback, even though every jogging ERTA sits above
    it. Jogging ERTAs (n = 1, forward lean, continuous direct sun) and Cockell's
    behaviourally-averaged scalar measure different regimes, and Weihs 2013
    reports ERs from 0.05 to 1.0 across six sites and four postures in direct
    sun at 46-64 deg -- a spread that straddles 0.35 in both directions. The
    measured spread across activities and sites is wider than the gap to 0.35,
    so no single replacement is defensible. See calibration_approval.yaml,
    parameter_approvals.PERSONAL_DOSE_GEOMETRY_FACTOR.

    solar_elevation_deg is converted to zenith angle for the Vernez path and
    used directly for the activity path. On its own -- no vis_percent, no
    activity -- it still does nothing: the elevation-only form previously
    returned here was withdrawn on 2026-08-16 as unsourced (gap 26), and Vernez
    shows solar geometry is the ~1.18x term while posture is the ~11.6x term.
    """
    if vis_percent is not None and activity is not None:
        raise ValueError(
            "personal_geometry_factor: pass vis_percent OR activity, not both. "
            "Vernez 2015 (simulated, sky-visibility driven) and the measured "
            "activity ERTA tables are competing estimates of the same ratio; "
            "choose one explicitly."
        )
    if activity is not None:
        return activity_erta(activity,
                             _MEAN_SITE_KEY if body_site is None else body_site,
                             solar_elevation_deg)
    if body_site is not None:
        raise ValueError(
            "personal_geometry_factor: body_site is only meaningful with "
            "activity=. For the Vernez route, select the site by passing its "
            "vis_percent from vis_table_s1(body_site, posture)."
        )
    if vis_percent is None:
        if solar_elevation_deg is not None:
            warnings.warn(
                "personal_geometry_factor: solar_elevation_deg alone no longer "
                "changes the result. The `0.15 + 0.30*cos(theta)` form was "
                "withdrawn 2026-08-16 as unsourced (gap 26). Pass vis_percent "
                "for the Vernez 2015 model, or activity= for a measured ERTA "
                "table; returning the 0.35 Cockell scalar.",
                UserWarning, stacklevel=2,
            )
        return 0.35
    sza = None if solar_elevation_deg is None else 90.0 - float(solar_elevation_deg)
    return vernez_exposure_ratio(vis_percent, sza)


# ======================= VERNEZ 2015 TABLE S1 (SOURCED) ====================
# Vernez, Milon, Vuilleumier, Bulliard, Koechlin, Boniol & Dore 2015, J Expo
# Sci Environ Epidemiol 25:113-118, doi 10.1038/jes.2014.6, Supplementary
# Information Table S1: "Value of Vis [%] according to body part and body
# posture" (13 body sites x 5 postures, SimUVEx 3D-manikin simulation).
#
# EXTRACTION PROVENANCE: extracted 2026-08-17 from the supplementary .doc
# supplied by the repository owner (OLE2 Word document, metadata author
# 'dvernez', CHUV, created 2013-11-11) to
# data/vernez2015_table_s1_vis_by_site_posture.csv. That file is the single
# source of these numbers; read the CSV header before changing anything here.
# This CLOSES data request #13b (see calibration_approval.yaml,
# data_requests_closed_2026_08_17).
#
# This REPLACES the VIS_PRESETS_PLACEHOLDER dict that lived here from
# 2026-08-16 to 2026-08-17. That dict was back-derived by inverting Eq. 1 at
# VERNEZ_REF_SZA_DEG against the mean ERs Vernez print in their Results TEXT
# (VERNEZ_REPORTED_SITE_ER) -- consistent with the paper's OUTPUTS but never
# its INPUTS. The values below are the paper's actual inputs, transcribed
# (not inverted, not fitted).
#
# [CLOSED 2026-08-20, was VERIFY] 'Top of shoulders' and 'Upper back' carry
# IDENTICAL values across ALL FIVE postures in the source table (35.9 / 50.3 /
# 48.6 / 45.7 / 85.3). Confirmed against Vernez 2015's own supplementary .doc
# (Research_Papers/supplementary materials.doc, transcribed and diffed
# cell-by-cell against VERNEZ_TABLE_S1_VIS below -- all 65 values match). The
# duplication is IN THE SOURCE DOCUMENT, not a transcription error here.
# Whether it reflects a genuine shared manikin mesh region or an upstream
# copy-paste in Vernez's own table remains unknown -- still do not silently
# de-duplicate, since the source itself is ambiguous on this point.
#
# Still a ceiling, not expected real-world exposure: Vernez ER (and therefore
# the Vis that feeds it) assumes NO SHADE and continuous exposure -- see the
# module comment on vernez_exposure_ratio above.
VERNEZ_TABLE_S1_POSTURES: Tuple[str, ...] = (
    "seated", "kneeling", "standing_erect_arms_down",
    "standing_erect_arms_up", "standing_bowing",
)

_VERNEZ_TABLE_S1_CSV: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "vernez2015_table_s1_vis_by_site_posture.csv",
)


def _normalize_body_site(raw: str) -> str:
    """'Forearm (external)' -> 'forearm_external'; 'Upper back' -> 'upper_back'."""
    return re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")


def _load_vernez_table_s1(path: str = _VERNEZ_TABLE_S1_CSV) -> Dict[str, Dict[str, float]]:
    """Parse Vernez 2015 Supplementary Table S1 from the extracted CSV.

    The CSV, not this function, is the source of truth for the numbers --
    this only parses it, so the values are never duplicated in two places.
    Returns {normalized_body_site: {posture: Vis_percent}}. Raises on a
    missing/malformed file rather than silently falling back to anything.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        data_lines = [ln for ln in fh if not ln.lstrip().startswith("#") and ln.strip()]
    reader = csv.DictReader(data_lines)
    table: Dict[str, Dict[str, float]] = {}
    for row in reader:
        site = _normalize_body_site(row["body_site"])
        table[site] = {posture: float(row[posture]) for posture in VERNEZ_TABLE_S1_POSTURES}
    return table


# body_site -> {posture -> Vis percent}. Loaded once at import time from
# data/vernez2015_table_s1_vis_by_site_posture.csv (see provenance above).
VERNEZ_TABLE_S1_VIS: Dict[str, Dict[str, float]] = _load_vernez_table_s1()


def vis_table_s1(body_site: str, posture: str) -> float:
    """Sourced sky-visibility percentage (Vernez 2015 Table S1) for a body
    site and posture.

    body_site : e.g. 'face', 'forearm_external', 'top_of_shoulders' (any
                casing/spacing normalises the same way as the CSV header).
    posture   : one of VERNEZ_TABLE_S1_POSTURES.

    RAISES on an unknown site or posture rather than guessing. Feed the
    result into vernez_exposure_ratio() as `vis_percent` for a sourced,
    site-specific exposure ratio.

    [CLOSED 2026-08-20] 'top_of_shoulders' and 'upper_back' return identical
    numbers for every posture -- see the module comment above; this is not a
    lookup bug, confirmed against Vernez's own supplementary source document.
    """
    site = _normalize_body_site(body_site)
    if site not in VERNEZ_TABLE_S1_VIS:
        raise KeyError(
            f"No Vernez Table S1 entry for body site '{body_site}'. Known: "
            f"{sorted(VERNEZ_TABLE_S1_VIS)}."
        )
    if posture not in VERNEZ_TABLE_S1_POSTURES:
        raise KeyError(
            f"Unknown posture '{posture}'. Known: {VERNEZ_TABLE_S1_POSTURES}."
        )
    return VERNEZ_TABLE_S1_VIS[site][posture]
# ===================== END VERNEZ 2015 TABLE S1 (SOURCED) ==================


# ============ ACTIVITY-SPECIFIC MEASURED ERTA (SOURCED, 2026-08-20) ========
# Schmalwieser, Danhel, Schmalwieser, Reimoser, Rummel, Loy, Heydenreich &
# Weihs 2025, "Erythemal UV radiation exposure during jogging", Photochem
# Photobiol, doi 10.1111/php.14114. Full text in the repo:
#   Research_Papers/Schmalwieser_2025_Photo.pdf
#
# WHAT ERTA IS, AND WHY IT IS NOT THE SAME OBJECT AS VERNEZ ER.
# ERTA (exposure ratio to ambient) = E_i(h) / E_ambient(h): erythemally
# weighted irradiance MEASURED on a body site, over erythemally weighted
# irradiance measured simultaneously on a horizontal reference. Both numerator
# and denominator come from CIE-erythema-matched detectors, so the erythema
# action spectrum CANCELS and what survives is pure anatomical/angular
# geometry. That is the same quantity Vernez Eq. 1 PREDICTS, but here it is
# MEASURED on a moving human rather than simulated on a static 3D manikin, and
# it is resolved by activity and solar elevation rather than by sky visibility.
#
# WHAT THE STUDY IS. ONE volunteer wearing 12 SunSaver dosimeters at 7 body
# sites, jogging on asphalt on the Danube Island, Vienna (48.247 N, 165 m),
# cloud-free, 18-19 June 2025 (319/325 DU ozone), at solar elevations 60, 50,
# 40, 30 and 20 deg. Four cardinal-direction runs of ~1 min each per elevation
# were averaged to represent a "random pathway"; left/right sites were pooled
# except forehead and nape. Posture was the textbook jogging lean (upper body
# slightly forward, forearms at 90 deg, arms not crossing the midline).
#
# N = 1. ONE SUBJECT, ONE POSTURE, ONE CITY, ONE GROUND ALBEDO (asphalt/grass,
# 5-11%). This is not a population estimate and must never be presented as one.
#
# UNCERTAINTY THE AUTHORS THEMSELVES STATE:
#   +/-15% about the mean from dosimeter MOUNTING (left/right differences on
#     runs directly toward/away from the sun, where both sides should read
#     alike, ran 0.5-30%; largest at the shoulders, 30%).
#   +/-10% field accuracy of the SunSaver itself (+/-2% under lab control).
#   +/-25% swing in clear-sky UV index at fixed solar elevation from total
#     ozone, relative to the 320 DU during their measurements. This one moves
#     the AMBIENT term, not ERTA, so it does not enter here.
#
# NUMBERS BELOW ARE DIGITIZED FROM FIGURE 5, not transcribed from a table --
# the paper prints no ERTA table. Read off the published raster at 4x zoom to
# a precision of about +/-0.02, then rounded to 2 dp, which is already finer
# than the +/-15% the authors claim. Every series was cross-checked against the
# values the paper states in PROSE, and all seven agree:
#   Abstract: "the top of the shoulder is the most exposed body site by
#     receiving 80% of ambient UVR on average and up to 110%"; "All other body
#     parts receive up to 55% on average".
#   Results: shoulders "highest on average at a level of 0.8 with no dependence
#     on solar elevation"; forearms and forehead "no clear tendency ... at a
#     level of 0.5 and 0.3"; upper arms, calves, thighs and nape "decreases with
#     increasing solar elevation, starting with a value of 0.5 at 20 deg down to
#     around 0.4 and 0.3 at 60 deg".
# If these constants are ever disputed, re-digitize Figure 5 -- do NOT re-fit.
SCHMALWIESER_2025_ERTA_ELEVATIONS_DEG: Tuple[float, ...] = (20.0, 30.0, 40.0, 50.0, 60.0)

# body_site -> ERTA at each elevation in SCHMALWIESER_2025_ERTA_ELEVATIONS_DEG.
# "Random pathway": mean over the four cardinal running directions and both
# hemibodies. For a path that holds ONE orientation to the sun (a riverside
# route, a track), these means understate the sun-facing side -- see
# SCHMALWIESER_2025_JOGGING_ERTA_MAX below.
SCHMALWIESER_2025_JOGGING_ERTA: Dict[str, Tuple[float, ...]] = {
    "shoulders":  (0.80, 0.82, 0.82, 0.80, 0.80),   # flat; most exposed site at every elevation
    "forearms":   (0.52, 0.45, 0.44, 0.49, 0.52),   # U-shaped, no monotone trend
    "forehead":   (0.32, 0.29, 0.34, 0.29, 0.36),   # lowest; face is tilted slightly down
    "upper_arms": (0.55, 0.56, 0.56, 0.45, 0.35),   # falls above 40 deg
    "calves":     (0.50, 0.50, 0.48, 0.49, 0.42),   # falls above 50 deg
    "nape":       (0.50, 0.46, 0.45, 0.38, 0.35),   # falls monotonically
    "thighs":     (0.49, 0.42, 0.45, 0.37, 0.34),   # falls monotonically
}

# DIRECTIONAL MAXIMUM, from the paper's Discussion (Figure 6 shows the full
# per-direction spread). Jogging TOWARD the sun at 40-60 deg elevation puts the
# shoulders at ERTA 1.17 -- ABOVE ambient horizontal irradiance, because the
# torso leans forward and presents a surface closer to normal to the beam. Not
# used by any function here; recorded because a monotonous route is a real
# scenario and a ratio > 1 is the fact that makes it one.
SCHMALWIESER_2025_JOGGING_ERTA_MAX: Dict[str, float] = {"shoulders": 1.17}

# Relative measurement uncertainty the authors state (mounting), applied
# symmetrically about the mean by activity_erta_interval().
SCHMALWIESER_2025_ERTA_REL_UNCERTAINTY: float = 0.15

# The elevation band the measurements cover. The authors are explicit that the
# results "may not be necessarily applicable at higher solar elevations as
# found at subtropical and tropical latitudes", which is exactly where several
# of these series are still falling when the data runs out.
SCHMALWIESER_2025_ERTA_VALID_ELEVATION_DEG: Tuple[float, float] = (20.0, 60.0)

# Unweighted mean over a paper's measured sites, COMPUTED from its own table so
# it cannot drift from it. Exposed as body_site='mean_measured_sites'.
#
# READ THE CAVEAT BEFORE USING IT. This is NOT a body-surface-area-weighted
# whole-body exposure ratio. The 7 sites are the ones sportswear leaves bare,
# each counted once, so the shoulders -- ~2% of BSA and by far the most exposed
# site -- carry 1/7 of the weight. Nothing represents the trunk at all: the
# sternum was tested and DROPPED as a mounting position. A genuine whole-body
# number needs the BSA weighting in vitd_exposure.py, applied to whichever
# sites the person actually leaves uncovered.
_MEAN_SITE_KEY: str = "mean_measured_sites"


def _with_site_mean(table: Dict[str, Tuple[float, ...]]) -> Dict[str, Tuple[float, ...]]:
    """Copy of `table` with the unweighted across-site mean added. The measured
    table itself is left untouched, so the sourced constant keeps exactly the
    sites the paper measured and nothing else."""
    n_elev = len(SCHMALWIESER_2025_ERTA_ELEVATIONS_DEG)
    out = dict(table)
    out[_MEAN_SITE_KEY] = tuple(
        sum(vals[i] for vals in table.values()) / len(table) for i in range(n_elev)
    )
    return out


# Registry so a second activity (e.g. Weihs et al. 2024, cycling, not yet in
# the repo) slots in without touching the lookup function.
ACTIVITY_ERTA_TABLES: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "jogging": _with_site_mean(SCHMALWIESER_2025_JOGGING_ERTA),
}

# ---------------------------------------------------------------------------
# DOUBLE-COUNTING WITH E_source (VDE:EE): WHY THESE TWO TERMS MULTIPLY CLEANLY
# ---------------------------------------------------------------------------
# The concern is real and worth stating in full, because both ERTA and the
# vitamin-D-to-erythema effectiveness ratio (vitd_model.SOURCE_CALIBRATION_TABLE,
# Webb 2021) depend on solar zenith angle, and multiplying two SZA-dependent
# factors looks like applying the same physics twice.
#
# It is not. The dose chain factorises as
#
#   D3-effective dose = SED_ambient_horizontal          [ambient, has cos(SZA)]
#                     x ERTA(site, activity, elevation) [anatomical GEOMETRY]
#                     x E_source (VDE:EE)               [SPECTRAL weighting]
#
# and each factor is a ratio of two DIFFERENT things:
#   ERTA     = erythemal irradiance on the body / erythemal irradiance on a
#              horizontal plane. Same spectral weighting top and bottom, so the
#              action spectrum cancels; what remains is the angle between the
#              skin and the sky. Its elevation dependence is the changing
#              geometric relationship between a tilted body surface and the
#              horizontal reference plane, NOT a change in spectrum.
#   VDE:EE   = vitamin-D-weighted / erythema-weighted integral of the SAME
#              irradiance. A pure unit conversion between action spectra. Its
#              elevation dependence is the changing UVB:UVA composition of the
#              beam, NOT a change in geometry.
# So the two are orthogonal by construction and both may vary with elevation
# without either being counted twice.
#
# THE RESIDUAL THAT IS NOT RESOLVED, and is left as a documented caveat:
# E_source is measured for the spectrum on a HORIZONTAL plane, but ERTA moves
# the dose onto a TILTED anatomical surface which sees a different direct:diffuse
# mix -- more direct beam on a vertical surface at low sun, more diffuse sky at
# high sun. Direct and diffuse differ in UVB:UVA, so the true VDE:EE at the skin
# is not exactly the horizontal VDE:EE. The correction is the ratio
# VDE:EE(site) / VDE:EE(horizontal), which NO source in this repo measures --
# Webb 2021 reports only horizontal-plane values, and Schmalwieser 2025 measured
# with erythemal detectors only, never a spectroradiometer at the skin. It is
# second-order (the whole horizontal VDE:EE range across Webb's sun entries is
# 1.22-2.01, and the direct/diffuse split moves a fraction of that) and it is
# NOT corrected here. Any absolute D prediction using activity ERTA carries it.
#
# The practical rule: use ERTA for geometry and E_source for spectrum, never one
# to stand in for the other, and do NOT "compensate" a falling ERTA at high sun
# with a rising VDE:EE. They are not offsetting corrections to one quantity;
# they are two independent quantities that happen to share an argument.


def activity_erta(activity: str,
                  body_site: str = _MEAN_SITE_KEY,
                  solar_elevation_deg: float | None = None) -> float:
    """Measured exposure-ratio-to-ambient for a named ACTIVITY and body site.

    activity            : key of ACTIVITY_ERTA_TABLES ('jogging').
    body_site           : one of the measured sites, or 'mean_measured_sites'
                          for the unweighted mean over them (read the caveat
                          above ACTIVITY_ERTA_TABLES -- it is NOT BSA-weighted).
    solar_elevation_deg : required. Linearly interpolated between the measured
                          elevations; CLAMPED at 20 and 60 deg with a warning,
                          because the source refuses to extrapolate above 60.

    RAISES on an unknown activity or site rather than guessing. Returns the
    erythemally-weighted GEOMETRIC ratio only -- combine it with E_source for
    the spectral step, and see the double-counting note above before doing so.
    """
    if activity not in ACTIVITY_ERTA_TABLES:
        raise KeyError(
            f"No measured ERTA table for activity '{activity}'. Known: "
            f"{sorted(ACTIVITY_ERTA_TABLES)}. Activity ERTAs are measured per "
            "activity and are not transferable between them -- see the "
            "Weihs 2013 comparison in the module comment."
        )
    table = ACTIVITY_ERTA_TABLES[activity]
    site = _normalize_body_site(body_site)
    if site not in table:
        raise KeyError(
            f"No {activity} ERTA for body site '{body_site}'. Measured sites: "
            f"{sorted(k for k in table if k != _MEAN_SITE_KEY)}, or "
            f"'{_MEAN_SITE_KEY}'."
        )
    if solar_elevation_deg is None:
        raise ValueError(
            "activity_erta requires solar_elevation_deg: the table is "
            "elevation-resolved and four of its seven sites vary with it."
        )
    lo_deg, hi_deg = SCHMALWIESER_2025_ERTA_VALID_ELEVATION_DEG
    elev = float(solar_elevation_deg)
    if not lo_deg <= elev <= hi_deg:
        warnings.warn(
            f"activity_erta: solar elevation {elev:.1f} deg is outside the "
            f"measured band {lo_deg:.0f}-{hi_deg:.0f} deg (Schmalwieser 2025); "
            "clamping. The authors state the results 'may not be necessarily "
            "applicable at higher solar elevations as found at subtropical and "
            "tropical latitudes'.",
            UserWarning, stacklevel=2,
        )
        elev = min(hi_deg, max(lo_deg, elev))
    return float(np.interp(elev, SCHMALWIESER_2025_ERTA_ELEVATIONS_DEG, table[site]))


def activity_erta_interval(activity: str,
                           body_site: str = _MEAN_SITE_KEY,
                           solar_elevation_deg: float | None = None,
                           ) -> Tuple[float, float]:
    """(low, high) ERTA at the source's own stated measurement uncertainty.

    +/-15% about the mean, which is the mounting uncertainty Schmalwieser 2025
    derives from left/right disagreement on runs straight toward and away from
    the sun. Like vernez_er_interval(), this is the honest object to report to
    a person; the point value is for arithmetic. It does NOT include the n = 1
    between-subject variance, which is unmeasured and larger.
    """
    mid = activity_erta(activity, body_site, solar_elevation_deg)
    u = SCHMALWIESER_2025_ERTA_REL_UNCERTAINTY
    return mid * (1.0 - u), mid * (1.0 + u)


# ---------------------------------------------------------------------------
# WEIHS et al. 2013 -- PROSE-SOURCED SITE RATIOS. NO ABSOLUTE ERTA TABLE.
# ---------------------------------------------------------------------------
# Weihs, Schmalwieser, Reinisch, Meraner, Walisch & Harald 2013, "Measurements
# of Personal UV Exposure on Different Parts of the Body During Various
# Activities", Photochem Photobiol 89:1004-1007 (Research Note), full text in
#   Research_Papers/Measurements of Personal UV Exposure on Different Parts
#   of the Body During Various Activities.pdf
# Ten X2000-10 optoelectronic dosimeters mounted on a suit, reading six body
# parts (forehead, shoulders, arms, chest, thighs, lower legs) across four
# activities (walking, sitting, lying, sitting up). Vienna, campus of the
# Veterinary Medicine University (48.25 N, 150 m), May and July 2010; seven
# sequences of 30-40 min, six of them cloud-free and those six are what the
# means below average. Study subject always in direct sun, low albedo (brick,
# asphalt, grass), solar elevations 46-64 deg. Dosimeter inclination adjustable
# only to "roughly +/-10%". ERs reported across all sites and activities: 0.05
# to 1.0. Same city and an overlapping author group as Schmalwieser 2025 above
# (Schmalwieser and Weihs are coauthors on both).
#
# [PROVENANCE 2026-08-20] TWO READS OF THIS PAPER EXIST AND ONLY ONE IS GOOD.
#
#   PROSE -- USED, HIGH CONFIDENCE. The Results section "Average UV exposure"
#     states the between-site ratios directly, as sentences, and the Discussion
#     independently repeats the ordering. Extracted as text (not from any
#     image), transcribed into WEIHS_2013_PROSE_SITE_RATIOS below, and pinned
#     by tests/test_physiology.py::TestWeihs2013ProseRatios.
#
#   FIGURE 6 -- ABANDONED, DO NOT CITE. An earlier pass digitized Figure 6 into
#     per-site ER values and this file quoted them (a walking site mean of
#     "~0.26", walking lower leg "~0.10", sitting-up lower leg equal to the
#     forehead at "~0.6"). Figure 6 is a grouped range-bar chart -- six sites x
#     four activities, min-max bars with a black dot for the mean, series
#     distinguished only by grayscale hatching, with shoulders and lower legs
#     hatched the same way in opposite directions. At the resolution of the copy
#     held here it cannot be read to that precision. Those digitized numbers are
#     NOT REPRODUCIBLE and must not be quoted anywhere; they have been withdrawn
#     from vitd_objectives.py and calibration_approval.yaml as well.
#     The paper does NOT contradict itself -- the figure read was wrong. Its
#     Discussion confirms its own Results text: "The highest ERs occurred on the
#     lower legs while lying and sitting up. In these positions the lower legs
#     are oriented horizontally like the device for ambient UV measurements."
#     That is the same claim the abandoned digitization appeared to refute.
#
# [TODO 2026-08-20 :: WEIHS-2013-VOR] REVISIT WHEN THE VERSION OF RECORD LANDS.
# The PDF in the repo is the "Accepted Article" early-view proof and prints its
# own warning about "differences between this version and the Version of
# Record". If the VoR renders Figure 6 in colour, at higher resolution, or ships
# a supplementary data table, redo the digitization and attempt a full per-site
# quantitative entry in ACTIVITY_ERTA_TABLES for 'walking' / 'sitting' /
# 'lying' / 'sitting_up', cross-validating every series against the prose ratios
# below exactly the way Schmalwieser 2025's Figure 5 was cross-validated against
# its Results text. A new predictive table IS a calibration step and needs
# approval -- the same marker WEIHS-2013-VOR is filed in
# calibration_approval.yaml under parameter_approvals.PERSONAL_DOSE_GEOMETRY_
# FACTOR.open_items. Grep WEIHS-2013-VOR to find both ends.
#
# WHY THERE IS STILL NO TABLE, EVEN WITH GOOD PROSE. The text gives RATIOS
# BETWEEN SITES WITHIN ONE ACTIVITY and nothing else: no absolute ER for any
# single site, and no elevation resolution (all six sequences sit inside one
# 46-64 deg band and are reported as single "random orientation" means). An
# ACTIVITY_ERTA_TABLES entry needs absolute, elevation-resolved values. Scaling
# these ratios against an assumed forehead ER would invent the one number the
# paper never states -- a calibration decision, not a transcription. Do not.

# The measured band. Narrow, and entirely inside the upper half of the
# Schmalwieser 2025 jogging band (20-60 deg); the overlap is 46-60 deg.
WEIHS_2013_SOLAR_ELEVATION_BAND_DEG: Tuple[float, float] = (46.0, 64.0)

# Discussion: "ER values obtained within the scope of the present study are
# between 0.05 and 1." Across six sites and four activities, in direct sun.
WEIHS_2013_REPORTED_ER_RANGE: Tuple[float, float] = (0.05, 1.0)

# Discussion: dosimeter inclinations "may therefore only be adjusted with an
# accuracy of roughly +/-10%".
WEIHS_2013_MOUNTING_ACCURACY: float = 0.10

# activity -> (elevated_site, reference_site) -> ratio of mean ERs.
#
# RATIOS ONLY. These are NOT exposure ratios to ambient and carry no absolute
# magnitude; each is one site's mean ER over another site's mean ER, in the same
# activity, over the same six clear-sky sequences. Transcribed verbatim from the
# sentences quoted beside each block. Never feed one of these into a dose
# calculation as if it were an ERTA.
WEIHS_2013_PROSE_SITE_RATIOS: Dict[str, Dict[Tuple[str, str], float]] = {
    # "During walking, the study subject receives the highest exposures on the
    #  shoulders which is 70% higher than the exposure of the chest and the
    #  forehead."
    "walking": {
        ("shoulders", "chest"): 1.70,
        ("shoulders", "forehead"): 1.70,
    },
    # "During sitting the highest UV exposures are received by the thighs and
    #  the shoulders. Exposures are 60% higher than on the forehead and on the
    #  chest."
    "sitting": {
        ("thighs", "forehead"): 1.60,
        ("thighs", "chest"): 1.60,
        ("shoulders", "forehead"): 1.60,
        ("shoulders", "chest"): 1.60,
    },
    # "During sitting up and lying the exposures are highest on the lower legs
    #  (respectively 115% and 55% higher than on forehead)." Corroborated by the
    #  Discussion: "The highest ERs occurred on the lower legs while lying and
    #  sitting up."
    "sitting_up": {("lower_legs", "forehead"): 2.15},
    "lying": {("lower_legs", "forehead"): 1.55},
}


def weihs2013_prose_ratio(activity: str, site: str, reference_site: str) -> float:
    """Weihs 2013 between-site mean-ER ratio, as stated in the paper's text.

    A RATIO, not an exposure ratio to ambient: it says how much more UV `site`
    received than `reference_site` during `activity`, and says nothing about
    either one's absolute level. There is no absolute anchor anywhere in the
    paper's prose, which is why Weihs has no ACTIVITY_ERTA_TABLES entry -- see
    the comment block above, and the WEIHS-2013-VOR marker for the condition
    under which that could change.

    RAISES on any combination the paper does not state, rather than inverting
    or chaining the ones it does.
    """
    if activity not in WEIHS_2013_PROSE_SITE_RATIOS:
        raise KeyError(
            f"No Weihs 2013 prose ratios for activity '{activity}'. Known: "
            f"{sorted(WEIHS_2013_PROSE_SITE_RATIOS)}."
        )
    pair = (_normalize_body_site(site), _normalize_body_site(reference_site))
    table = WEIHS_2013_PROSE_SITE_RATIOS[activity]
    if pair not in table:
        raise KeyError(
            f"Weihs 2013 states no {activity} ratio for {pair[0]} vs "
            f"{pair[1]}. Stated pairs: {sorted(table)}. Do not invert or chain "
            "the stated pairs -- the paper prints only these."
        )
    return table[pair]


# CROSS-STUDY SANITY CHECK, computed 2026-08-20 from
# SCHMALWIESER_2025_JOGGING_ERTA above. Nothing in either source was adjusted to
# make this line up; the numbers are recomputed by
# tests/test_physiology.py::TestWeihs2013ProseRatios so this comment cannot go
# stale.
#
# The two studies share one comparable statement -- shoulders against forehead,
# upright locomotion, overlapping solar elevation (Weihs 46-64, Schmalwieser
# 20-60, overlap 46-60 deg):
#   Weihs 2013,        WALKING: shoulders / forehead = 1.70 (prose)
#   Schmalwieser 2025, JOGGING: shoulders / forehead = 2.61 at 46 deg,
#                               2.76 at 50 deg, 2.22 at 60 deg; 2.52 mean
#                               across the overlap band (digitized Fig. 5)
#
# VERDICT: ORDERING CORROBORATED, MAGNITUDE NOT. Both find the shoulders the
# most exposed site and the forehead among the least, by a wide margin, with
# different dosimeters 15 years apart -- so the shoulder result does not rest on
# Schmalwieser's single subject alone. But the jogging contrast is ~1.5x the
# walking contrast (2.52 vs 1.70), a gap far outside either paper's stated
# measurement uncertainty (+/-15% mounting for Schmalwieser, +/-10% for Weihs).
# The mechanism is the posture and is the point of the whole activity-resolved
# approach: the jogging lean tilts the shoulders toward the beam and the face
# further down, while walking is upright. It is evidence AGAINST, not for,
# borrowing one activity's table for another -- which is why activity_erta()
# raises on an unknown activity instead of substituting jogging.
#
# WHAT IT IS USED FOR, AND IT IS LOAD-BEARING FOR TWO CONCLUSIONS:
#  1. INDEPENDENT CORROBORATION OF THE SHOULDER RESULT, as just above.
#  2. THE REASON PERSONAL_DOSE_GEOMETRY_FACTOR WAS NOT RAISED. Weihs reports ERs
#     "between 0.05 and 1" across his six sites and four activities, every one
#     of them in continuous direct sun at 46-64 deg. That 20-fold spread
#     straddles the 0.35 Cockell scalar in both directions, while every jogging
#     ERTA (0.29-0.82) sits above it. WHICH SITE IS BARE AND WHICH POSTURE IS
#     HELD move the ratio further than the distance from 0.35 to either study's
#     central value, so no single measured number replaces a fallback whose
#     whole job is to answer when the activity is unknown -- which is why the
#     jogging table above is opt-in behind activity= and the flat fallback was
#     left where it was.
#     [CORRECTED 2026-08-20] This argument previously rested on a digitized
#     walking site mean of "~0.26" and a claimed 5x jogging-vs-walking
#     disagreement at the calves. Both came from the abandoned Figure 6 read and
#     are withdrawn (see PROVENANCE above). The conclusion is unchanged, the
#     0.35 default did not move, and the restatement above uses only numbers the
#     paper prints as text.
# ========== END ACTIVITY-SPECIFIC MEASURED ERTA (SOURCED, 2026-08-20) ======


# ---------------------------------------------------------------------------
# 3. Biphasic 25-hydroxylation (mechanistic replacement for linear D_max)
# ---------------------------------------------------------------------------
# The core model damps synthesis with (1 - D/D_max), D_max = 150 nmol/L of
# unconfirmed origin (KNOWN_ISSUES #3). That term drives the entire scheduling
# result, so its provenance matters more than any pigmentation parameter.
#
# Better-grounded alternative: 25-hydroxylation efficiency is BIPHASIC in
# 25(OH)D status. A 2025 Communications Medicine study (Nature) with bimonthly
# sampling found the conversion of cutaneous D3 to plasma 25(OH)D3 is efficient
# at low 25(OH)D3 and "decreases markedly" above an inflection at
#   25(OH)D3 = 56 nmol/L  (corresponding morning D3 ~2 nmol/L).
# It also reports 24-hydroxylase (CYP24A1) activity falling as 25(OH)D3 falls,
# linear in 25(OH)D3 (r^2 = 0.79) -- catabolism is substrate-driven, which the
# state-dependent k_decay_D already captures.
#
# This matters physiologically: regulation acts on hydroxylation and catabolism,
# NOT as a hard ceiling on cutaneous synthesis. A hard ceiling wrongly predicts
# that a very-high-D individual synthesises no D3 in skin at all.
# [C3-biphasic-anchor] Form: smooth Hill-type efficiency, ~1 below the inflection and decaying above.
#     eff(D) = 1 / (1 + (D / D_inflect) ** n)
# n = 2 gives a gentle knee (eff = 0.5 at the inflection, 0.8 at 0.5x, 0.2 at 2x).
# Empirical Anchor: Finland vernal UV study (Ala-Houhala et al.) shows net 25(OH)D
# reversal at high baseline (~90 nmol/L, where low baseline D0~37 nmol/L yields +6 nmol/L
# while high baseline D0~116 nmol/L yields -6.7 nmol/L net decrease).
# [EVIDENCE NOTE] Heaney et al. 2008 (Am J Clin Nutr 87:1738-1742) independently
# reports a biphasic (fast-then-slow) 25-hydroxylation relationship, aggregated
# across 6 human studies (n=206; oral D3 doses 0-11000 IU/d and cutaneous/UVB
# input): Y = 0.433X + 87.81[1 - exp(-0.468X)], R^2 = 0.4482 (Fig. 4, studies
# B/C/E/F pooled; text body states 0.443 for the same fit -- inconsistency in
# the source paper, both reported here), where X = serum vitamin D3 (nmol/L)
# and Y = serum 25(OH)D (nmol/L). This CORROBORATES the qualitative SHAPE of a
# biphasic feedback (rapid conversion at low precursor, much slower above a
# threshold they place at ~15 nmol/L serum D3) but does NOT set n or
# D_inflect quantitatively: Heaney's curve is a linear+exponential mix, not a
# Hill function, and its X-axis is serum vitamin D3 (the precursor pool), not
# serum 25(OH)D feeding back on its own synthesis the way g(D) does here.
# Possible follow-up (NOT done): refit Heaney's own aggregate curve to a Hill
# form to get a non-arbitrary n, rather than the "gentlest integer knee" below.
HYDROXYLATION_INFLECTION_NMOL_L: float = 56.0
HYDROXYLATION_HILL_N: float = 2.0


def hydroxylation_efficiency(D: float,
                             inflection: float = HYDROXYLATION_INFLECTION_NMOL_L,
                             n: float = HYDROXYLATION_HILL_N) -> float:
    """Relative efficiency of converting cutaneous D3 into plasma 25(OH)D.

    Never reaches exactly zero: unlike (1 - D/D_max), this does not claim that
    synthesis stops outright at some ceiling, which is the physiologically
    indefensible part of the linear form.
    """
    if D <= 0.0:
        return 1.0
    return float(1.0 / (1.0 + (D / inflection) ** n))


# ---------------------------------------------------------------------------
# 4. Skin-cancer risk
# ---------------------------------------------------------------------------
# The single most important epidemiological fact for scheduling, and the one the
# old qualitative sidecar missed entirely:
#
#   SCC  tracks CUMULATIVE lifetime UV dose.
#   BCC and MELANOMA track INTERMITTENT, intense exposure and sunburns.
#
# Sources: Elwood & Jopson 1997 (Int J Cancer 73:198-203), overview of 29
# case-control studies of sun exposure and 21 of sunburn, 9,121 incident
# melanomas. Full text now in Research_Papers/ and all four ORs verified
# against it: intermittent 1.71 (1.54-1.90), heavy OCCUPATIONAL 0.86
# (0.77-0.96, i.e. PROTECTIVE), sunburn at any age 1.91 (1.69-2.17), and --
# the number that most directly constrains this module -- TOTAL exposure only
# 1.18 (1.02-1.38). Melanoma is therefore almost entirely a pattern effect and
# barely a cumulative-dose effect, which is why the melanoma/BCC axis below is
# built on per-session intensity and the cumulative axis is labelled SCC.
# Also Leiter 2009; Calzavara-Pinton
# 2015 (BCC ~ intermittent, especially childhood; SCC ~ chronic/cumulative);
# Armstrong 1988. O'Sullivan 2026 (2.4M Ontario workers) finds low-intermittent
# HR = 1.26 vs high-continuous HR = 1.13, preserving the ordering.
#
# The scheduling implication is the opposite of naive dose-minimisation: for a
# FIXED total UV budget, spreading it into many small sub-erythemal sessions
# lowers melanoma/BCC risk (fewer intense events, no sunburn) even though
# cumulative dose is unchanged. Frequent-and-small beats rare-and-large.
#
# This is a RELATIVE risk index for ranking schedules, NOT an absolute
# incidence prediction. It is deliberately dimensionless and normalised so that
# 1.0 = a reference schedule of 1 SED/week for a year. Do not present it as a
# lifetime cancer probability.
#
# Erythema is the key non-linearity: risk per event rises steeply once a session
# exceeds ~1 MED. Sub-erythemal sessions still accrue cumulative (SCC) risk.
SCC_CUMULATIVE_WEIGHT: float = 1.0

# Exponent on CUMULATIVE dose in the SCC term -- the "biological amplification
# factor" Ab of the skin-cancer dosimetry literature.
#
# [LITERATURE-ANCHORED] This term used to be LINEAR in cumulative dose (exponent
# 1.0), which is not what the epidemiology says. Incidence follows a POWER LAW in
# annual UV dose,
#     ln(relative tumour density) = Ab * ln(annual erythemal dose),
# and Ab > 1 for every keratinocyte cancer measured (Moan 2015, Photodermatol
# Photoimmunol Photomed 31:26-35, which fits this form across SCC, BCC and
# melanoma).
#
# Values of Ab, erythemally weighted -- which is the weighting this model's SED
# already carries:
#   Moan, Dahlback, Henriksen & Magnus 1989 (Cancer Res 49:5207-12, PMID
#     2766289; 6 Norwegian regions, 1976-1985):  SCC 1.6-1.8, BCC 2.1-2.3.
#     Critically for the unit choice here, they computed Ab with an erythema
#     action spectrum AND with a basal-layer mutagenesis action spectrum and got
#     answers agreeing "to within about 10%" -- so using erythemal SED rather
#     than a carcinogenic action spectrum is not the dominant error.
#   Green et al. 1976 (Photochem Photobiol):     ~1.8 for NMSC (US).
#   Jones 1992 (Br J Dermatol 127:519-22):        3.0 SCC, 1.7 BCC.
# The literature therefore brackets Ab at roughly 1.3-3.0. 1.7 is the midpoint
# of the SCC range in the single source that (a) reports SCC and BCC separately,
# (b) uses erythemal weighting, and (c) cross-checks the action spectrum.
#
# Consequence for scheduling: cumulative dose now costs MORE than proportionally,
# so "many tiny sessions" no longer buys an unlimited free pass on the SCC axis
# the way a linear term allowed. See vitd_calibration.moan1989_scc_exponent_check.
SCC_DOSE_EXPONENT: float = 1.7

# Exponent on per-session intensity in the intermittent (BCC/melanoma) term.
#
# [OR-ANCHORED, NOT FITTED] Previously an unjustified modelling choice. It is now
# anchored to the Elwood & Jopson 1997 odds ratios, but the anchoring identifies
# an INTERVAL, not a point. Derivation, both bias directions, and the sensitivity
# of the implied value to the (unmeasured) burst:continuous dose contrast are in
# vitd_calibration.elwood1997_intermittency_check -- read it before quoting any
# absolute risk index. In brief:
#
#   at equal cumulative SED, index_burst / index_continuous = C ** (k - 1)
#   so  k = 1 + ln(OR_burst / OR_occupational) / ln(C)
#
# where C is the burst:continuous per-session dose contrast in fractions of MED.
# Anchoring C to this model's own cited doses -- the erythema threshold (1 MED,
# which defines the sunburn OR arm) against the sub-erythemal ceiling
# (vitd_objectives.GAMMA_MED, the largest non-burning dose the model recommends)
# -- puts k in a narrow interval that contains the value below. Widening C
# lowers the implied k substantially; the non-matched cumulative dose between
# Elwood's arms biases it back up. Neither bias is quantified.
#
# The value is retained rather than moved: it sits inside the anchored interval,
# and the ORs do not resolve it more finely than that interval.
INTERMITTENT_BURN_EXPONENT: float = 3.0
# The reference schedule both indices are normalised to: 1 SED/week for a year.
# Both score exactly 1.0 for FST I on it.
REFERENCE_ANNUAL_SED: float = 52.0
REFERENCE_SESSION_SED: float = 1.0


def _med_sed_by_fst(fst: int) -> float:
    """Minimal erythema dose in SED, Shih/Farrar/Rhodes 2018 (P24)."""
    return {1: 2.1, 2: 2.6, 3: 3.2, 4: 5.6, 5: 7.5, 6: 15.2}.get(fst, 3.5)


# ---------------------------------------------------------------------------
# 4a. FST susceptibility, derived from measured basal-layer DNA damage
# ---------------------------------------------------------------------------
# This replaces an admitted order-of-magnitude prior, exp(-0.45*(fst-1)), which
# was wrong in BOTH directions: too steep across FST I-IV and too shallow across
# I-VI (see the anchors below).
#
# THE MEASUREMENT IT IS BUILT FROM, and why it is the right one.
# Shih, Farrar, ... de Gruijl & Rhodes 2018 (J Invest Dermatol 138:2244-2252;
# PDF in Research_Papers/, and already this file's MED source) dosed FST I-VI
# at 20-80% of each person's own MED -- i.e. exactly the sub-erythemal,
# fractional-MED regime this model schedules -- and measured cyclobutane
# pyrimidine dimers (CPD) by epidermal depth. Two findings drive everything here:
#
#   1. WHOLE-epidermal CPD per fractional MED does NOT depend on skin type
#      (mixed-effects regression, P = 0.63): +0.13 AU per 20% MED increment
#      (95% CI 0.03-0.19), pooled over all skin types.
#   2. BASAL-layer CPD does: skin type is significant (P = 0.03). Pooled slope
#      +0.08 AU per 20% MED (95% CI 0.01-0.14), but basal-layer CPD was
#      UNDETECTABLE throughout 20-80% MED in skin types IV-VI, while being
#      dose-responsive in I-III. The CPD depth gradient correlated with skin
#      darkness at r = 0.74 (P < 0.0001) and did not vary with dose (P = 0.29).
#      Repair was complete by 48 h with no skin-type difference (P = 0.38).
#
# The germinative basal layer is where skin cancers initiate (Shih 2018's own
# framing: "actively dividing, long-residing cells"). So the carcinogenically
# relevant quantity is not damage per unit dose, and not even damage per
# fractional MED -- it is the FRACTION of that damage reaching the basal layer.
#
# DERIVATION (one fitted parameter; three independent out-of-sample checks).
# Melanin attenuates UVR through the epidermis, so take CPD(z) = C0*exp(-mu*z)
# (Beer-Lambert). With epidermal depth h and x = mu*h:
#     whole-epidermal mean = C0 * (1 - exp(-x)) / x
#     basal layer          = C0 * exp(-x)
#     basal : whole        = phi(x) = x*exp(-x) / (1 - exp(-x))
# phi(0) = 1 (no pigment: damage uniform with depth, which is what Shih observed
# in FST I) and phi -> 0 as pigment rises. Because finding 1 says the
# whole-epidermal level is FST-independent, phi(x_FST) IS the relative basal
# damage per fractional MED, normalised to unpigmented skin.
#
# x is taken proportional to Shih's own Table 1 melanin measurement -- percentage
# of epidermal area staining for melanin, by skin type -- so the FST dependence
# comes from measured pigment, not from an assumed decay rate.
#
# THE ONE FITTED NUMBER. kappa is fixed by Fajuyigbe et al. 2018 (FASEB J
# 32:3700-3706, PMID 29394104; full text in Research_Papers/ as of 2026-08-20),
# which measured CPD protection factors in FST VI versus fair Europeans by epidermal
# depth: whole epidermis 8.0x, upper 5.0x, middle 16.5x, BASAL LAYER 59.0x.
# The quantity this model needs is the basal:whole ratio, 59.0/8.0 = 7.375 --
# and because it is a ratio of two protection factors measured on the SAME
# subjects at the SAME exposures, it is independent of whether Fajuyigbe's arms
# were matched on physical dose or on fractional MED. That ambiguity, which
# would otherwise be disqualifying, cancels. kappa solves phi(kappa * 59.70) =
# 8.0/59.0. Nothing else is fitted.
#
# THREE CHECKS IT WAS NOT FITTED TO, AND PASSES:
#   1. Gandini et al. 2005 III (Eur J Cancer 41:2040-59, PMID 16125929), pooled
#      over 60 studies: melanoma RR for skin type I vs IV = 2.09 (1.67-2.58).
#      Derived here: 2.12. A CPD biomarker measured in two extreme phenotypes
#      predicts a melanoma meta-analysis to within 1.5%.
#   2. The premise itself. Shih's MED ratio FST VI:I is 15.2/2.1 = 7.24. If
#      whole-epidermal CPD really depends only on fractional MED, then at equal
#      PHYSICAL dose the whole-epidermis protection factor must be that same
#      7.24. Fajuyigbe measured 8.0. Two labs, two assays, ~10% apart.
#      Now DIRECTLY supported, not just cross-checked: Fajuyigbe's own full text
#      reports the whole-epidermal DeltaCPD-vs-MED slopes for SPT I/II and SPT VI
#      are statistically indistinguishable (P=0.68) -- i.e. the premise was
#      independently tested and passed within that single study, not just
#      inferred by comparing it against Shih's separate cohort.
#   3. Shih's own pooled basal:whole slope ratio, 0.08/0.13 = 0.615 at the
#      study's mean melanin of 12.56%. This curve implies 0.70 -- comfortably
#      inside the interval those two wide CIs permit (0.05 to 1.0).
#   For contrast the retired prior gave I:IV = 3.9 against Gandini's 2.09.
#
# HONEST LIMITS -- read before quoting a cross-skin-type absolute index:
#   - Fajuyigbe is a small extreme-phenotype study (n=3 SPT I, n=3 SPT II,
#     n=6 SPT VI, all Nigerian; black West Africans vs fair Europeans). Its 59x
#     basal figure carries the whole calibration. Full text confirms the design
#     and numbers transcribed here; it does not add FST III/IV/V data, so the
#     mid-phototype gradient still comes only from Shih 2018's melanin percentages.
#   - Corroboration available in the full text but not yet wired into
#     skin_type_susceptibility_check(): the paper's own discussion independently
#     ties basal DNA-PF (59x) to South African BCC white:black incidence (60x)
#     and middle-layer DNA-PF (16.5x) to SCC incidence (20x). Same-authors, not
#     independent of Fajuyigbe itself, but a second concrete number beyond
#     Gandini 2005.
#   - Shih's melanin percentages rest on n = 5, 6, 6, 5, 3 and 1 volunteers for
#     FST I-VI. FST VI is a single person.
#   - Shih found basal CPD UNDETECTABLE in FST IV-VI, whereas this curve assigns
#     them 0.47, 0.26 and 0.14. It is therefore CONSERVATIVE -- it charges dark
#     skin more residual basal damage than either study could detect, which is
#     the safe direction for a risk model.
#   - Nothing here is an epidemiological RR per erythemal episode BY SKIN TYPE.
#     No such source exists (searched 2026-08-09; the per-sunburn meta-analyses
#     either pool skin types or restrict to FST I-IV). Substituting a biomarker
#     for that missing RR is the central assumption of this axis.
#   - Sanity, not target: Ahmed et al. 2025 (Br J Dermatol; England NDRS
#     2013-2020) gives White:Black age-standardised incidence ratios of 16.3
#     (melanoma), 13.1 (cSCC) and 25.8 (BCC). This curve gives 7.3 for FST
#     II:VI -- SMALLER, as a constitutive factor should be, because population
#     ratios also contain behaviour, exposure, ancestry latitude and
#     ascertainment. It is a direction check, never a calibration target.
# See vitd_calibration.skin_type_susceptibility_check().

# Percentage of epidermal area stained for melanin by skin type, Shih 2018
# Table 1 (modified Warthin-Starry). n = 5, 6, 6, 5, 3, 1 for FST I-VI.
SHIH2018_EPIDERMAL_MELANIN_PCT: dict = {
    1: 0.0, 2: 0.28, 3: 2.21, 4: 25.14, 5: 42.04, 6: 59.70,
}
# "All" row of the same table -- the mean the pooled CPD slopes correspond to.
SHIH2018_MEAN_MELANIN_PCT: float = 12.56
# Pooled CPD dose-response slopes, AU per 20% MED increment (Shih 2018 Results).
# Used as a CHECK on the calibration below, not as its target.
SHIH2018_WHOLE_EPIDERMAL_CPD_SLOPE: float = 0.13   # 95% CI 0.03-0.19
SHIH2018_BASAL_LAYER_CPD_SLOPE: float = 0.08       # 95% CI 0.01-0.14

# Fajuyigbe et al. 2018 CPD protection factors, FST VI vs fair European, by
# epidermal depth (FASEB J 32:3700-3706, PMID 29394104; full text in
# Research_Papers/).
FAJUYIGBE2018_PROTECTION: dict = {
    "whole": 8.0, "upper": 5.0, "middle": 16.5, "basal": 59.0,
}
# Attenuation per unit of melanin-stained epidermal area. Solves
# phi(kappa * 59.70) = 8.0 / 59.0 -- see basal_damage_fraction below.
MELANIN_ATTENUATION_KAPPA: float = 0.0536715


def _phi(x: float) -> float:
    """basal:whole CPD ratio for Beer-Lambert attenuation of optical depth x."""
    if x <= 1e-9:
        return 1.0
    return float(x * np.exp(-x) / (1.0 - np.exp(-x)))


def basal_damage_fraction(fst: int,
                          kappa: float = MELANIN_ATTENUATION_KAPPA) -> float:
    """Relative basal-layer DNA damage per fractional-MED dose, by skin type.

    1.0 for unpigmented skin (damage uniform with epidermal depth, as observed
    in FST I) and falling with measured epidermal melanin. This is the model's
    skin-cancer susceptibility factor; the derivation and its limits are in the
    module comment above.

    Returns the FST III value for out-of-range inputs, matching the defensive
    behaviour of _med_sed_by_fst.
    """
    melanin = SHIH2018_EPIDERMAL_MELANIN_PCT.get(
        int(fst), SHIH2018_EPIDERMAL_MELANIN_PCT[3])
    return _phi(kappa * float(melanin))


def cancer_risk(session_seds, fst: int) -> dict:
    """Relative skin-cancer risk indices for a UV schedule.

    Parameters
    ----------
    session_seds : sequence of float
        Per-session erythemal SED doses for the whole schedule.
    fst : int
        Fitzpatrick skin type, 1-6. Sets the erythema threshold and a
        constitutive susceptibility factor.

    Returns
    -------
    dict with keys:
        scc_index          cumulative-dose-driven risk (SCC)
        intermittent_index burst-driven risk (BCC, melanoma)
        burn_fraction      fraction of sessions at or above 1 MED
        total_sed          cumulative erythemal SED
        combined_index     equal-weight summary for single-objective ranking

    Both indices scale with FST susceptibility, which is now the measured
    fraction of DNA damage reaching the germinative basal layer
    (`basal_damage_fraction`, derived from Shih 2018) rather than an
    order-of-magnitude prior.
    """
    seds = np.asarray([s for s in session_seds if s > 0], dtype=float)
    med = _med_sed_by_fst(fst)

    # Constitutive susceptibility: relative basal-layer DNA damage per
    # fractional-MED dose. FST I = 1.0 down to FST VI ~ 0.06.
    #
    # [DERIVED, two out-of-sample checks] Full derivation, anchors and limits in
    # section 4a above. Superseded the prior exp(-0.45*(fst-1)), which was too
    # steep across FST I-IV (3.9 vs Gandini's 2.09) and too shallow across I-VI
    # (9.5 vs England's observed 13-26).
    #
    # It cancels out of within-FST comparisons: every index here is proportional
    # to it, so schedule RANKINGS for one person are unaffected. It matters only
    # for cross-FST comparisons of absolute index values -- which is precisely
    # what the old prior could not support and this can, within stated limits.
    susceptibility = basal_damage_fraction(fst)

    if seds.size == 0:
        return {"scc_index": 0.0, "intermittent_index": 0.0,
                "burn_fraction": 0.0, "total_sed": 0.0,
                "combined_index": 0.0, "susceptibility": susceptibility,
                "med_sed": med}

    total_sed = float(seds.sum())

    # SCC: power law in cumulative dose, exponent Ab = SCC_DOSE_EXPONENT
    # (Moan 1989 / Moan 2015). Susceptibility multiplies the index rather than
    # entering the base: Ab is a WITHIN-population dose-response slope, measured
    # across ambient-dose gradients in a phenotypically homogeneous (Norwegian)
    # population, so it describes how risk scales with dose, not how it scales
    # between skin types. Raising the susceptibility to Ab as well would
    # double-count the phenotype effect -- and would predict a FST I:VI ratio of
    # ~120x against England's observed 13x for cSCC (Ahmed 2025).
    scc = (SCC_CUMULATIVE_WEIGHT * susceptibility
           * (total_sed / REFERENCE_ANNUAL_SED) ** SCC_DOSE_EXPONENT)

    # BCC / melanoma: super-linear in per-session intensity relative to MED, so
    # that one big session costs far more than the same dose split up.
    # (sed/med)^3 summed over sessions. At fixed total dose this is minimised by
    # many small sessions -- the Elwood/Leiter intermittency finding.
    frac_med = seds / med
    intermittent = float(susceptibility * np.sum(frac_med ** INTERMITTENT_BURN_EXPONENT))
    # Normalise by a FIXED reference schedule: 1 SED/week for a year delivered
    # to FST I skin, i.e. 52 sessions at 1.0/2.1 = 0.48 MED.
    #
    # [BUG FIX 2026-08-09] The reference used to be per-FST AND to include the
    # susceptibility factor: `susceptibility * 52 * (1/med)**k`. Both of those
    # then cancelled out of the ratio, leaving the raw absolute sum of sed**k.
    # Two consequences, both contradicting this module's own documented model:
    #   - the docstring's claim that "both indices scale with FST
    #     susceptibility" was false for this index -- it was exactly 1.0 for
    #     every skin type on the reference schedule;
    #   - because the per-FST reference is far gentler in fractional-MED terms
    #     for dark skin, the index ranked a FST VI person at 0.5 MED as ~380x
    #     riskier than a FST I person at 0.5 MED. Shih 2018 -- already this
    #     module's MED source -- measured EQUAL whole-epidermal damage and LESS
    #     basal damage at equal fractional MED in darker skin. The ordering was
    #     backwards.
    # A fixed reference keeps the index in fractional-MED currency, so it is
    # comparable across skin types, and lets the susceptibility survive. FST I
    # on the reference schedule still scores exactly 1.0, as before.
    ref_intermittent = (52.0 * (REFERENCE_SESSION_SED / _med_sed_by_fst(1))
                        ** INTERMITTENT_BURN_EXPONENT)
    if ref_intermittent > 0:
        intermittent /= ref_intermittent

    burn_fraction = float(np.mean(frac_med >= 1.0))

    # Combined index for single-objective ranking.
    #
    # A plain 0.5*scc + 0.5*intermittent is wrong: the two indices live on very
    # different scales (over a realistic schedule grid, SCC spans ~0.3-5 while
    # the intermittent index spans ~0.1-450, because it is cubic in dose). An
    # arithmetic mean is therefore ~100% intermittent risk and silently discards
    # the SCC objective.
    #
    # Geometric mean instead: scale-free, penalises being bad on EITHER pathway,
    # and cannot be gamed by making one term enormous. Still a convenience for
    # ranking only -- the honest output is the Pareto frontier over the two
    # indices separately (vitd_objectives.deficiency_vs_risk_frontier).
    combined = float(np.sqrt(max(scc, 0.0) * max(intermittent, 0.0)))

    return {
        "scc_index": scc,
        "intermittent_index": intermittent,
        "burn_fraction": burn_fraction,
        "total_sed": total_sed,
        "combined_index": combined,
        "susceptibility": susceptibility,
        "med_sed": med,
    }
