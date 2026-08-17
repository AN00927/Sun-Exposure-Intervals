"""
vitd_physiology.py
==================
Mechanistic sub-models added July 2026 after an external-literature pass.

Each function here replaces something the core model previously handled with a
phenomenological stand-in. They are opt-in via ModelParams flags so that every
change is separately ablatable -- the earlier lesson of this project is that a
mechanism which cannot be switched off cannot be shown to matter.

Contents
--------
1. body_weight_dilution_factor  -- volumetric dilution by body size
2. photodegradation_factor      -- previtamin-D3 photostationary plateau
3. hydroxylation_efficiency     -- biphasic 25-hydroxylation (replaces D_max)
4. cancer_risk                  -- SCC (cumulative) + BCC/melanoma (intermittent)

References are given inline with the numbers actually used.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Volumetric dilution by body size
# ---------------------------------------------------------------------------
# Drincic 2012 (Obesity 20:1444-8, N=686) tested linear vs hyperbolic fits of
# 25(OH)D against body size. A HYPERBOLIC fit on TOTAL BODY WEIGHT removed the
# entire obesity-related component of inter-individual variance, and beat every
# linear model AND every BMI-based model. So:
#   - the right covariate is body weight, not BMI
#   - the right functional form is 1/weight (a dilution volume), not a linear
#     offset
# Mechanism is dilution of a fixed amount of vitamin D into a larger
# distribution volume, not sequestration (Drincic 2012; Walsh 2017 review;
# Pannu 2016 meta-regression shows weight loss raises 25(OH)D).
#
# Reference weight 70 kg. A 100 kg adult gets 0.70x the 25(OH)D rise per unit
# of cutaneous synthesis; a 55 kg adult gets 1.27x.
#
# Cross-check against oral dosing (the effect should be the same regardless of
# whether D3 arrives from skin or gut): Ekwaru 2014 (N=17,614) found obese
# subjects need 2-3x and overweight 1.5x the dose of normal-weight subjects for
# the same 25(OH)D. Weight ratio 100/70 = 1.43x dilution here; BMI 35 vs 22 at
# fixed height implies ~1.6x. Same order, slightly conservative.
BODY_WEIGHT_REF_KG: float = 70.0


def body_weight_dilution_factor(weight_kg: float | None,
                                ref_kg: float = BODY_WEIGHT_REF_KG) -> float:
    """Multiplicative factor on 25(OH)D gain per unit of vitamin D input.

    Hyperbolic in total body weight (Drincic 2012). Returns 1.0 when weight is
    unknown or equals the reference, so existing scenarios are unchanged.
    """
    if weight_kg is None or weight_kg <= 0:
        return 1.0
    return ref_kg / float(weight_kg)


# ---------------------------------------------------------------------------
# 2. Previtamin-D3 photodegradation plateau
# ---------------------------------------------------------------------------
# Holick 1981 (Science 211:590-3): during prolonged simulated-solar exposure,
# previtamin-D3 rises to a PLATEAU at 10-15% of the original 7-DHC content,
# after which further UV photoisomerises it to inert lumisterol3 and
# tachysterol3. Pigmentation and latitude change the TIME to reach the plateau
# but NOT the plateau height -- i.e. this is a photostationary state, and it is
# the reason sun exposure cannot be banked in one long session.
#
# van Dijk 2016 (Photochem Photobiol Sci) quantifies the consequence: the
# effective action spectrum changes shape with accumulated dose, and net
# production is "largely cancelled" by ~7.5 SED on previously unexposed skin.
# Their modified-QUT spectrum turns negative for longer UVB wavelengths after
# as little as 1.25 SED.
#
# Model: a per-session saturating factor on synthesis,
#     f(sed) = (1 - exp(-sed / D_SAT)) / (sed / D_SAT)
# i.e. the average of exp(-s/D_SAT) over the session, which is what you get if
# instantaneous yield decays exponentially as the substrate pool is driven
# toward its photostationary state. f -> 1 as sed -> 0 (no change at low dose)
# and f ~ D_SAT/sed at high dose (yield plateaus in ABSOLUTE terms).
#
# D_SAT = 7.5 SED anchors to van Dijk's "net production largely cancelled at
# 7.5 SED" on unexposed skin. [PRIOR] Single-source; van Dijk is a modelling
# paper, not a serum-response measurement.
PREVITAMIN_D_SAT_SED: float = 7.5


def photodegradation_factor(sed: float, d_sat: float = PREVITAMIN_D_SAT_SED) -> float:
    """Session-averaged yield reduction from previtamin-D3 photodegradation.

    Returns 1.0 for vanishing dose and falls as ~d_sat/sed for large dose, so
    total per-session yield saturates instead of growing without bound.

    This is the mechanism that makes splitting a UV budget across sessions
    genuinely better than taking it in one hit -- a real interior optimum,
    unlike the feedback-driven artifact documented in MODEL_AND_EVIDENCE.md,
    'Remaining gaps', gap 19.
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
# (gap 26) established that no source supports it: Schmalwieser & Siani's own
# measured curve is non-monotonic, and their conclusion is that the ratio as a
# function of solar elevation is not currently derivable. It was carried as a
# PLACEHOLDER only because nothing better was obtainable.
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
# NOT IMPORTED, AND STILL MISSING: Supplementary Table S1 gives typical Vis per
# posture and body site. It is not in the main PDF (data request #13b). Until it
# lands, Vis must be supplied by the caller; this module does not guess it.
# See MODEL_AND_EVIDENCE.md gap 26 and 'Open decisions' item 12.

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
                             vis_percent: float | None = None) -> float:
    """Ratio of personal anatomical UV dose to ambient horizontal UV.

    vis_percent supplied -> Vernez 2015 Eq. 1 (sourced, R^2 = 0.988).
    vis_percent omitted  -> 0.35, the Cockell 2001 measured central value for
                            upright activity. [PRIOR] Scalar, behaviourally
                            averaged, and it directly scales every minutes
                            recommendation, so sweep it first.

    solar_elevation_deg is converted to zenith angle for the Vernez path. On its
    own it no longer does anything: the elevation-only form previously returned
    here was withdrawn on 2026-08-16 as unsourced (gap 26), and Vernez shows
    solar geometry is the ~1.18x term while posture is the ~11.6x term.
    """
    if vis_percent is None:
        if solar_elevation_deg is not None:
            warnings.warn(
                "personal_geometry_factor: solar_elevation_deg alone no longer "
                "changes the result. The `0.15 + 0.30*cos(theta)` form was "
                "withdrawn 2026-08-16 as unsourced (MODEL_AND_EVIDENCE.md gap "
                "26). Pass vis_percent to use the Vernez 2015 model; returning "
                "the 0.35 Cockell scalar.",
                UserWarning, stacklevel=2,
            )
        return 0.35
    sza = None if solar_elevation_deg is None else 90.0 - float(solar_elevation_deg)
    return vernez_exposure_ratio(vis_percent, sza)


# =========================== PLACEHOLDER ===================================
# [PLACEHOLDER -- NOT FROM THE PAPER. Recorded 2026-08-16, data request #13b.]
#
# Vernez's Supplementary Table S1 gives typical Vis by posture and body site.
# It is NOT in the main PDF and has not been obtained. These values are
# BACK-DERIVED by inverting Eq. 1 at VERNEZ_REF_SZA_DEG against the mean ERs
# Vernez print in their Results text (see VERNEZ_REPORTED_SITE_ER). That makes
# them consistent with the paper's own reported outputs, but they are NOT the
# paper's Vis inputs and must not be cited as such.
#
# Two further reasons to treat any number derived from these as provisional:
#   - Vernez ER assumes NO SHADE and continuous exposure (see above), so these
#     are ceilings, not expected real-world exposure.
#   - Whole-body presets additionally assume an area weighting across sites
#     that this repo has not sourced at all. That assumption is cruder than
#     the Vis inversion itself.
#
# Anything reported from these MUST be labelled a placeholder. Replace as soon
# as Table S1 lands.
def _vis_for_target_er(er_percent: float) -> float:
    """Invert Vernez Eq. 1 for Vis at the reference SZA. PLACEHOLDER helper."""
    lo, hi = 0.01, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if vernez_exposure_ratio(mid, VERNEZ_REF_SZA_DEG) * 100.0 < er_percent:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


VIS_PRESETS_PLACEHOLDER: Dict[str, float] = {}


def vis_preset(name: str) -> float:
    """Placeholder sky-visibility percentage for a named exposure scenario.

    RAISES if `name` is unknown rather than guessing. Every value returned is a
    PLACEHOLDER back-derived from Vernez's reported ERs -- see the block comment
    above. Callers that surface a number to a user must say so.
    """
    if not VIS_PRESETS_PLACEHOLDER:
        VIS_PRESETS_PLACEHOLDER.update({
            k: _vis_for_target_er(v) for k, v in VERNEZ_REPORTED_SITE_ER.items()
        })
    if name not in VIS_PRESETS_PLACEHOLDER:
        raise KeyError(
            f"No Vis preset '{name}'. Known placeholders: "
            f"{sorted(VERNEZ_REPORTED_SITE_ER)}. Vernez Supplementary Table S1 "
            f"(data request #13b) would replace these with sourced values."
        )
    return VIS_PRESETS_PLACEHOLDER[name]
# ========================= END PLACEHOLDER =================================


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
# 32:3700-3706, PMID 29394104; abstract only, full text not obtained), which
# measured CPD protection factors in FST VI versus fair Europeans by epidermal
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
#   3. Shih's own pooled basal:whole slope ratio, 0.08/0.13 = 0.615 at the
#      study's mean melanin of 12.56%. This curve implies 0.70 -- comfortably
#      inside the interval those two wide CIs permit (0.05 to 1.0).
#   For contrast the retired prior gave I:IV = 3.9 against Gandini's 2.09.
#
# HONEST LIMITS -- read before quoting a cross-skin-type absolute index:
#   - Fajuyigbe is ABSTRACT-ONLY here, and is a small extreme-phenotype study
#     (black West Africans vs fair Europeans). Its 59x basal figure carries the
#     whole calibration. Getting the full text is data request #12.
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
# epidermal depth (FASEB J 32:3700-3706, PMID 29394104; abstract only).
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
