"""
kconst_recalibration.py
=======================
DERIVATION for the K_const recalibration approved on 2026-08-17
(`calibration_approval.yaml`, `a1_diagnosis_result_2026_08_17`).

One question, no file written:

    What value of `ModelParams.K_const` reproduces the between-skin-type
    25(OH)D response measured in the WITHIN-STUDY, MATCHED-PHYSICAL-DOSE,
    MATCHED-BSA, SAME-SOURCE trials this repo holds?

The diagnosis it rests on is `scripts/fst_gap_decomposition.py` (~12 min).
That script established WHERE the target is -- FST V:II optical 1.7-2.7x from
two independent routes. This script picks the number and shows its working.

Design
------
CALIBRATION SET (fitted). Only trials satisfying ALL of:
  * within-study     -- both skin types in the same paper, so no cross-trial
                        baseline anchoring is needed;
  * matched PHYSICAL dose (SED), not fractional MED;
  * matched %BSA between the arms;
  * same UV source both arms, so E_source cancels from the ratio;
  * both groups' pre-exposure 25(OH)D measured and printed.
Exactly two studies in this repo qualify: Felton 2016 and Young 2020.

HELD OUT (never fitted; used only to score).
  * Kift 2013 and Darling 2013 field personal-dosimetry inversions -- a
    different method (field badges, real behaviour) on a different population.
  * Farrar 2013 / Rhodes 2010 cross-trial pair (one unobtained baseline).
  * Shih 2018's fractional-MED null and Kallioglu 2024 (withdrawn: it is an
    erythema MED table -- see fst_gap_decomposition S1a).
  * The whole existing validation suite (`vitd_validation.run_all`).

Everything the script CHOOSES is tagged `# [CHOICE]`; everything it
ESTIMATES rather than reads from a source is tagged `# [ESTIMATE]`.

DARK-SKIN PROVENANCE, stated once and true of every number below:
  * every FST V datapoint is Indian-subcontinent / South Asian ancestry;
  * every FST VI datapoint is lamp-based (Arimed B or TL01 cabinet) -- no
    field dosimetry in FST VI exists anywhere in this repo;
  * the only African-ancestry matched-dose cell of any size is Young 2020's
    TL01 FST VI arm (n = 11).
Nothing here is measured in FST VI outdoors, and nothing in FST V outside UK
latitudes. Every FST VI number this calibration produces is an EXTRAPOLATION
of one coefficient onto M_baseline(VI) (0.7324793388429751 on the 2026-09-22
ITA -> M scale; was a clipped 1.00 before it), itself a group-level prior.

Weighting
---------
[CHANGED 2026-09-22, TWICE] The default is RANDOM EFFECTS (DerSimonian-Laird,
`random_effects_dl`), see DEFAULT_WEIGHTING_SCHEME. The paragraph below
describes the `inverse_variance` scheme that was the default for part of that
day (W-W) and is retained as a selectable scheme:
routes
are inverse-variance weighted WHERE A MEASURED VARIANCE EXISTS, and equally
weighted where none does, inside the lab/field method tiers the incumbent
estimator already implied. `one_vote_per_study` is RETAINED and computed in
the same run so every figure published under it stays reproducible. See
`WEIGHTING_SCHEMES`, the `VARIANCE_PROVENANCE` register, section S3c
(variance-availability audit) and S3d (all schemes side by side).

NO VARIANCE IS IMPUTED ANYWHERE. Two of the four routes publish none, and
the estimator says so in its own output rather than assuming one.

Run
---
    PYTHONIOENCODING=utf-8 python3 scripts/kconst_recalibration.py
    PYTHONIOENCODING=utf-8 python3 scripts/kconst_recalibration.py         --scheme one_vote_per_study
"""

from __future__ import annotations

import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams, simulate, get_E_source

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FST_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
ROMAN_OF = {v: k for k, v in FST_ROMAN.items()}


def hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# DATA BLOCK -- every value quoted from a source held in this repo
# ===========================================================================

# --- Felton et al. 2016, Br J Dermatol 175:1320-1328 (Research_Papers/) ----
# Within-study phototype II vs V, same cabinet, same protocol, both arms.
# Table 1. NOTE the PDF text layer drops decimal points ("365" = 36.5).
FELTON_D0 = {2: 36.5, 5: 17.2}          # pre-course 25(OH)D, nmol/L
FELTON_FINAL = {2: 54.3, 5: 25.5}       # post-course 25(OH)D, nmol/L
FELTON_N = {2: 10, 5: 6}
FELTON_SED_PER_SESSION = 1.3
FELTON_SESSIONS_PER_WEEK = 3
FELTON_WEEKS = 6
FELTON_BSA = 0.35
# [CONFIRMED 2026-09-22, was ESTIMATE] Felton names the cabinet directly:
# "A Philips HB588 Sunstudio irradiation cabinet ... after fitting with
# alternating Arimed B (Cosmedico GmbH, Stuttgart, Germany) and Cleo Natural
# (Philips) fluorescent tubes, providing UVR emission close to U.K. summer
# sunlight (95% UVA: 320-400 nm, 5% UVB: 290-320 nm)" (Br J Dermatol
# 175:1320, p.1321-1322). This is a statement in the paper, not an
# assumption, so the [ESTIMATE] tag it carried was factually wrong -- the
# tag is resolved to CONFIRMED rather than retained "to be safe", because a
# tag that marks a printed fact as an estimate makes every other tag in this
# file less informative. Independently corroborated by Rhodes 2010 (same
# institution, same dose regime, same spectral description), which also gives
# Arimed B a spectroradiometric measurement outside Webb 2021's own table.
# The SENSITIVITY OF THIS CHOICE IS EXACTLY ZERO for K_const either way:
# E_source cancels from a between-skin-type ratio within one study, and the
# estimator fits ratios only. Nothing numeric moves; only the epistemic
# status of the choice does. Felton 2016 directly names its cabinet
# ("Arimed B ... and Cleo Natural" tubes, Br J Dermatol 175:1321-1322) --
# see calibration_approval.yaml's felton2016_source_estimate_resolved_2026_09_22.
FELTON_SOURCE = "arimed_b"

# --- Young et al. 2020, JID 140:1418 --------------------------------------
# Individual-level trial data, in this repo. Protocol, from the paper:
# "the same suberythemal UVR dose, to 85% body surface area ... repeated five
# times with intervals of 3-4 days".
YOUNG_XLSX = os.path.join(
    REPO, "Papers_and_Datasets", "Final skin type vit D database 26_03_2020.xlsx")
YOUNG_BSA = 0.85
YOUNG_INTERVAL_D = 3.5   # midpoint of the paper's stated 3-4 days  # [ESTIMATE]
YOUNG_ARMS = {
    # sheet: (dose column prefix, E_source key, SED per exposure)
    "ArimedB irradiated": ("UVdose.Sample_", "arimed_b", 2.0),
    "TL01 irradiated":    ("UVDose.Sample_", "tl01_nb_uvb", 1.5),
}
# Young 2020 Table 3 slope ratios. Table 3's own note: the regression model is
# "adjusted for pre-exposure 25(OH)D3, slope, and intercept". Both published
# ratios are therefore ANCOVA-ADJUSTED, not raw. (Read from the PDF, 2026-08-17.)
YOUNG_PUBLISHED_ADJUSTED = {"ArimedB irradiated": ("II", "VI", 1.29),
                            "TL01 irradiated":    ("II", "VI", 1.43)}

# --- Held-out targets, all reproduced from fst_gap_decomposition.py -------
# Run 2026-08-17; see that script for their derivation and their caveats.
HELD_OUT_FIELD = {          # label: (FST V:II optical factor, implied K_const)
    "Kift 2013 / Webb 2010, behaviour removed": (1.75, 0.788),
    "Darling 2013, behaviour removed":          (2.68, 1.387),
}
HELD_OUT_FIELD_BLIND = {    # what dumping behaviour into K_const would need
    "Kift 2013 / Webb 2010, BEHAVIOUR-BLIND":   (4.28, 2.047),
    "Darling 2013, BEHAVIOUR-BLIND":            (5.71, 2.453),
}
DIAGNOSED_BAND = (1.7, 2.7)   # FST V:II optical: the approved target band

# [BUG FIX 2026-09-22, W-AC] HELD_OUT_FIELD above is a CACHE of the field
# inversions as they stood on 2026-08-17 (K_const 0.33, A 2.353, the pre-Narea
# M_baseline_by_FST table, BSA_EXP 0.50, linear feedback). It was carried
# through the 2026-09-06 and 2026-09-21 refits on the stated belief that the
# field routes are "computed independently of M_baseline_by_FST" (vitd_model.py
# K_const entry). THAT BELIEF IS FALSE:
# fst_gap_decomposition._invert_cohort_pair -> _predict_summer calls
# simulate(fst=..., K_const_override=K, params=ModelParams()), so the implied
# K_const depends on M_baseline, BSA_EXP, the feedback form and (weakly) A.
# Re-run on the current tree the two field routes invert to materially
# different values (see field_route_values below and W-AC sec 0.5). The cache
# is RETAINED, unedited, and selectable with --field-routes carried, so every
# figure published from it stays reproducible; the default re-derives.
FIELD_ROUTE_MODES = ("rederived", "carried")
DEFAULT_FIELD_ROUTES = "rederived"
#: Set from --field-routes in main(); module-level so every section reads the
#: same mode without threading it through six signatures.
FIELD_ROUTES_MODE = DEFAULT_FIELD_ROUTES

RE_PREAMBLE = """
  WHY RANDOM EFFECTS IS THE DEFAULT, and it is not a preference. A
  fixed-effect inverse-variance pool asserts that every route estimates ONE
  common K_const and differs from the others only by sampling error. S3 of
  this script MEASURES that assertion and rejects it: the two lab routes miss
  in opposite directions by roughly 40% each, and Young's published FST II:VI
  ratio (1.29-1.43) is SMALLER than Felton's FST II:V (2.14) in the same class
  of cabinet, which is impossible under any monotone pigment penalty. Pooling
  that under fixed effects reports a precision the evidence does not have.
  Random effects estimates the between-route variance tau^2 instead and
  carries it into the interval.

  Kift 2013 has no usable variance ON THE QUANTITY THIS ROUTE INVERTS -- it
  publishes a CI on the RAW serum contrast, but this route inverts the
  contrast AFTER behaviour removal, whose uncertainty is unquantified
  anywhere in this repository. It is therefore EXCLUDED from the pool.
  Nothing is imputed to keep it in; the probe below bounds what its exclusion
  could be hiding instead."""

HKSJ_NOTE = """  READING: the HKSJ interval is printed for completeness and is
  UNINFORMATIVE at k = 3 -- the t(2) multiplier alone is 4.303, so the
  interval spans more than the whole physically admissible range of K_const.
  Do not quote it as this parameter's uncertainty. The tau^2-estimator range
  above is the usable statement."""

# --- Darling et al. 2013, Osteoporos Int 24(2):477-488 -- PRIMARY SOURCE ----
# DOI 10.1007/s00198-012-1973-2, PMID 22525977. Held since 2026-09-22 as the
# University of Surrey AUTHOR'S ACCEPTED MANUSCRIPT (not the version of
# record): Research_Papers/Vitamin_D_deficiency_Osteoporosis_International_2012.pdf
# Table 3, "Serum 25(OH)D ... and UV exposure by group and season", printed
# as mean, SD and the per-cell n (values are means +/- SD per its footnote).
# These cells are ALL AVAILABLE PARTICIPANTS PER SEASON, not the n = 90 / 35
# four-visit completers Webb 2021 Table 5 labels them with (Methods,
# "Numbers of participants recruited"). Every mean below matches the value
# this repo previously carried via Webb 2021 Table 5 exactly; what is NEW is
# the SD and the per-cell n, which give this route a sampling variance.
# Only the six cells the S4 field inversion reads are recorded here.
DARLING_T3 = {
    # cohort: {field: (mean, SD, n)}   units: SED/week or nmol/L
    "white": {"sed_summer_week": (6.39, 6.87, 113),
              "d_summer": (72.1, 26.1, 128),
              "d_winter": (44.5, 18.0, 80)},
    "southasian": {"sed_summer_week": (2.00, 2.13, 18),
                   "d_summer": (26.2, 9.9, 43),
                   "d_winter": (19.7, 10.6, 26)},
}
#: Darling's own multilevel model (Table 4) splits unexplained variance
#: 268.35 between individuals / 296.36 between seasons within individuals,
#: i.e. an intra-individual share of 0.475. Used ONLY as a sensitivity on the
#: summer-winter covariance; the primary SE treats the season cells as
#: independent (no correlation imputed), which is the wider of the two.
DARLING_MLM_ICC = 268.35 / (268.35 + 296.36)

#: [READ FROM SOURCE 2026-09-22] Darling 2012 Table 1 body weights, kg
#: (mean +/- SD): Caucasian 69.20 +/- 12.43 (n=135), Asian 65.80 +/- 11.33
#: (n=46). These are the RECRUITMENT n's. The inversion consumes them via
#: fst_gap_decomposition.FIELD_COHORTS[...]["weight_kg"].
DARLING_WEIGHT_KG = {"white": 69.20, "southasian": 65.80}

#: THE n IN THIS STUDY IS INTERNALLY INCONSISTENT, and the repo has carried
#: one of the three figures without saying which. Recorded once, here:
#:   * Abstract/Methods summary: "South Asian (n=35) and Caucasian (n=105)".
#:     The 105 appears NOWHERE ELSE in the paper.
#:   * Methods, "Numbers of participants recruited": 181 recruited
#:     (135 Caucasian + 46 Asian); 90 Caucasian and 35 Asian "had attended
#:     ... on all four occasions". Table 1 is headed with the 135/46 figures.
#:   * Study Power: "total of 39 Asian premenopausal women were recruited".
#: So the South Asian arm is variously 46 (recruited / Table 1), 39 (power
#: section) and 35 (four-visit completers), and the Caucasian arm 135, 105
#: and 90. Webb 2021 Table 5 labels the route with 90/35, i.e. the completer
#: counts. NONE of those is the right n for a sampling variance on the cells
#: this route inverts: Table 3 prints its OWN per-season n (113/128/80 and
#: 18/43/26), which are all available participants in that season, and those
#: are what DARLING_T3 above carries and what the SE below is built from.
DARLING_N_INCONSISTENCY = {
    "abstract": {"white": 105, "southasian": 35},
    "recruited_table1": {"white": 135, "southasian": 46},
    "power_section": {"southasian": 39},
    "four_visit_completers": {"white": 90, "southasian": 35},
}

#: [DARK-SKIN FLAG, Darling 2012 Discussion, read from source 2026-09-22]
#: "Seventy-five percent of the D-FINES Asian women were of Pakistani origin
#: and were partly veiled (showing hands and face only), so had minimal skin
#: exposure to the sun." Darling reports NO exposed-BSA figure of its own;
#: the repo's field inversion equates its cohorts' exposed BSA to Kift 2013's
#: diary percentages (0.155 light / 0.14 dark). If three quarters of the
#: South Asian arm were showing hands and face only, that 0.14 is too high,
#: the badge SED is being spread over more skin than was actually exposed,
#: and the inversion charges the shortfall to pigment -- i.e. Darling's
#: implied K_const is BIASED UPWARD. Darling stays in the pool (it is one of
#: only two field routes and the only non-Manchester one), with this flag
#: carried at every point of use and a SENSITIVITY VARIANT reported beside
#: the headline.
#:
#: THE VARIANT'S NUMBER, and where it comes from. "Hands and face only" is
#: converted by the Wallace rule of nines: anterior head ~4% BSA and two
#: hands ~5% (2 x 2.5%, palmar plus dorsal), i.e. 0.09. Applied to the 75%
#: who were partly veiled, with the remaining 25% left at Kift's 0.14:
#:     0.75 * 0.09 + 0.25 * 0.14 = 0.1025
#: [CHOICE] The rule-of-nines conversion is THIS REPOSITORY'S geometric
#: assumption, not Darling's -- the paper prints no BSA at all -- so the
#: variant bounds the veiling bias rather than measuring it.
DARLING_VEIL_FRACTION = 0.75
DARLING_VEIL_EXPOSED_BSA = 0.09
DARLING_BSA_KIFT_INHERITED = 0.14
DARLING_VEIL_ADJUSTED_BSA = (
    DARLING_VEIL_FRACTION * DARLING_VEIL_EXPOSED_BSA
    + (1.0 - DARLING_VEIL_FRACTION) * DARLING_BSA_KIFT_INHERITED)

#: [READ FROM SOURCE 2026-09-22] Darling 2012 never uses the word
#: "Fitzpatrick", "phototype" or "skin type" anywhere in its text (verified
#: by full-text search of the accepted manuscript). Its groups are
#: ETHNICITY-defined -- "Caucasian" and "Asian" (predominantly Pakistani
#: origin). The FST II and FST V labels this repository attaches to those two
#: cohorts are INFERRED FROM ETHNICITY by this repository, not reported by
#: Darling, and that inference is doing real work: it is what lets an
#: ethnicity contrast be read as a pigment contrast.
DARLING_USES_FITZPATRICK = False

#: The value K_const held BEFORE this recalibration. Pinned as a literal so
#: every "before" column stays a before column once the new value lands in
#: ModelParams -- reading it from ModelParams() would make the comparison
#: collapse to identity the moment the change is applied.
OLD_K_CONST = 0.33

# Sources deliberately EXCLUDED from the objective, kept to score the result.
# All values reproduced from fst_gap_decomposition.py S7, run 2026-08-17.
HELD_OUT_SCORING = [
    ("Farrar 2013 / Rhodes 2010 raw (Webb 2021 T4)", 2.26,
     "cross-trial; baselines unmatched"),
    ("Farrar/Rhodes corrected, Rhodes D0 swept 30-60", (2.76, 6.06),
     "width is ONE unobtained number"),
    ("Shih 2018 MED ratio under its own null", 2.88,
     "only if equal %MED => equal D; weak null"),
    ("Kallioglu 2024 'V:II = 2.40'", 2.40,
     "WITHDRAWN: it is an erythema MED table"),
    ("Felton 2016 MEASURED erythema MED ratio", 3.95,
     "erythema, not vitamin D"),
]


# ===========================================================================
# ROUTE VARIANCE REGISTER -- the input to inverse-variance weighting
# ===========================================================================
# Added 2026-09-22 (W-W). Every entry records WHAT THE SOURCE PUBLISHES, not
# what would be convenient. `None` means the source publishes no usable
# variance on the quantity this route inverts, and NO VARIANCE IS EVER
# IMPUTED to fill such a gap -- a route with no variance simply cannot enter
# an inverse-variance pool, and the estimator says so out loud rather than
# quietly assuming one. See VARIANCE_PROVENANCE below for the per-route
# evidence, and calibration_approval.yaml's K_const_random_effects_2026_09_22
# entry for the audit behind it.

#: Felton 2016, Br J Dermatol 175:1320-1328, Results, "Serum
#: 25-hydroxyvitamin D gain": 178 +/- 48 nmol/L (phototype II, n = 10) vs
#: 83 +/- 105 nmol/L (phototype V, n = 6) -- the BJD text layer drops the
#: decimal point, so these are gains of +17.8 +/- 4.8 and +8.3 +/- 10.5.
#: These are the SDs on the GAIN, which is exactly the quantity the II:V
#: ratio is built from. MEASURED, not assumed.
FELTON_GAIN_SD = {2: 4.8, 5: 10.5}

#: Young 2020: no variance is hardcoded because the individual-level
#: workbook is in the repository -- the per-subject slope SD is COMPUTED from
#: it in young_group_table() and read back here. Measured, not assumed.
#: CAVEAT carried at point of use: the route's observed value is Young's
#: ANCOVA-ADJUSTED published ratio (Table 3), while the SD available to us is
#: on the RAW per-subject slopes. ANCOVA removes between-subject baseline
#: variance, so the raw SD is an OVER-estimate of the adjusted ratio's
#: uncertainty. The bias therefore runs AGAINST Young getting more weight --
#: this route is handed less precision than it actually has, not more.
YOUNG_VARIANCE_SOURCE = "individual-level slopes, Papers_and_Datasets workbook"

#: Kift 2013 / Webb 2010 and Darling 2013: NO usable variance on the
#: INVERTED quantity. Recorded in full so the gap is auditable and is not
#: re-litigated from memory.
VARIANCE_PROVENANCE = {
    "Felton2016": (
        "PUBLISHED. Gain SDs +17.8 +/- 4.8 (n=10, phototype II) and "
        "+8.3 +/- 10.5 (n=6, phototype V), BJD 175:1320 Results. Delta "
        "method on the log gain ratio."),
    "Young2020": (
        "MEASURED FROM INDIVIDUAL DATA in this repo. Per-subject slope SD "
        "by FST by arm, computed in young_group_table(). Delta method on "
        "the log group-mean slope ratio. Applied to the ANCOVA-adjusted "
        "point estimate, which makes it CONSERVATIVE (too wide)."),
    "Kift2013": (
        "NONE ON THE INVERTED QUANTITY. Kift 2013 (BJD 169:1272) DOES "
        "publish dispersion -- IQRs on 25(OH)D by month and cohort, and a "
        "95% CI of 37-53% on the diet-adjusted South Asian:white serum "
        "ratio (point 45%, i.e. white:SA = 2.2222, CI 1.8868-2.7027, "
        "SE(ln) = 0.091679). VERIFIED FROM THE PDF 2026-09-22, correcting "
        "the earlier repo claim that routes 2-4 publish nothing. But that "
        "CI is on the RAW serum contrast; this route inverts the contrast "
        "AFTER the measured behavioural dose difference has been removed, "
        "and the behaviour-removal step carries no quantified uncertainty "
        "anywhere in this repo (fst_gap_decomposition S6's own assumption "
        "register). Using the serum CI as if it were the route's variance "
        "would UNDERSTATE the route's uncertainty and so OVER-weight it -- "
        "a partial variance used as a full one is worse than none, because "
        "it is confidently wrong. Not used."),
    "Darling2013": (
        "PUBLISHED, AS OF 2026-09-22. This entry previously read 'NONE ... "
        "NOT HELD IN THIS REPOSITORY'. That is now FALSE and the correction "
        "is the reason this route can enter a random-effects pool at all. "
        "The paper is held as the University of Surrey author's accepted "
        "manuscript (Research_Papers/Vitamin_D_deficiency_Osteoporosis_"
        "International_2012.pdf) and its Table 3 prints mean, SD and a "
        "per-season n for every one of the six cells the field inversion "
        "consumes. Delta-method SE by numerical Jacobian through the "
        "inversion, cells treated as independent (the conservative, wider "
        "choice -- see darling_se_K). TWO CAVEATS CARRIED AT EVERY POINT OF "
        "USE: (1) 75% of the South Asian arm were partly veiled (hands and "
        "face only), while the inversion equates their exposed BSA to "
        "Kift's diaries, so this route's K is BIASED UPWARD -- a sensitivity "
        "variant at a reduced exposed BSA is reported beside it; (2) Darling "
        "never uses Fitzpatrick type anywhere in the paper, so the FST V/II "
        "labels are INFERRED FROM ETHNICITY by this repository."),
}


# ===========================================================================
# LIVE FIELD-ROUTE RE-INVERSION  (replaces the 2026-08-17 HELD_OUT_FIELD cache)
# ===========================================================================
# [BUG FIX 2026-09-22] The cache above was carried through two refits on the
# stated belief that the field routes are "computed independently of
# M_baseline_by_FST". That belief is FALSE and the functions below prove it by
# re-running the inversion on the live tree. This is a BUG FIX, not a
# calibration: the code was reading a stale number, and reading the live one
# needs no approval. What it MOVES is approved separately.
#
# Note what is and is not scale-dependent, because it is the key to reading
# every number below. `_invert_cohort_pair` solves for the FST V:II OPTICAL
# FACTOR that reproduces both cohorts' summer 25(OH)D; K_const is only that
# factor's parameterisation on the M scale,
#     optical = exp(K_const * (M_V - M_II)).
# So a change to the ITA -> M mapping leaves each route's OPTICAL factor
# EXACTLY unchanged and moves its implied K_const by 1 / (M_V - M_II).
# Changes to BSA_EXP, SC_max, the feedback form, A and the body-weight term
# move the optical factor itself.

_FST_GAP = None


def _fst_gap():
    """Import scripts/fst_gap_decomposition lazily (it is the diagnosis)."""
    global _FST_GAP
    if _FST_GAP is None:
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        import fst_gap_decomposition as fg
        _FST_GAP = fg
    return _FST_GAP


def invert_field_route(tag, bsa_dark_override=None, cells=None, params=None):
    """Re-invert one field cohort pair on the CURRENT tree.

    `bsa_dark_override` replaces the dark cohort's exposed BSA (used for the
    Darling veiling sensitivity). `cells` optionally replaces the dark/light
    cohort dicts wholesale (used to perturb Darling's Table 3 cells for the
    delta-method SE). Returns the dict `_invert_cohort_pair` returns.
    """
    fg = _fst_gap()
    lk, dk, bl, bd = fg.COHORT_PAIRS[tag]
    light = dict(fg.FIELD_COHORTS[lk])
    dark = dict(fg.FIELD_COHORTS[dk])
    if cells:
        light.update(cells.get("light", {}))
        dark.update(cells.get("dark", {}))
    if bsa_dark_override is not None:
        bd = bsa_dark_override
    return fg._invert_cohort_pair(tag, light, dark, bl, bd, False,
                                  params=params)


DARLING_TAG = "Darling 2013 (Surrey, within-study)"
KIFT_TAG = "Kift 2013 / Webb 2010 (Manchester)"

_FIELD_CACHE = {}


def field_route_values(mode=None):
    """The two field routes, either re-inverted live or read from the cache.

    Memoised because each inversion is ~5 s and several sections need it.
    Returns {study: dict(K, optical, se, veiled_K, veiled_optical)}.
    """
    mode = mode or DEFAULT_FIELD_ROUTES
    if mode in _FIELD_CACHE:
        return _FIELD_CACHE[mode]
    if mode == "carried":
        out = {
            "Kift2013": dict(
                K=HELD_OUT_FIELD["Kift 2013 / Webb 2010, behaviour removed"][1],
                optical=HELD_OUT_FIELD["Kift 2013 / Webb 2010, behaviour removed"][0],
                se=None, se_parts={}, veiled_K=None, veiled_optical=None),
            "Darling2013": dict(
                K=HELD_OUT_FIELD["Darling 2013, behaviour removed"][1],
                optical=HELD_OUT_FIELD["Darling 2013, behaviour removed"][0],
                se=None, se_parts={}, veiled_K=None, veiled_optical=None),
        }
    else:
        r_k = invert_field_route(KIFT_TAG)
        r_d = invert_field_route(DARLING_TAG)
        se_d, parts_d = darling_se_K(r_d["K_const"])
        r_dv = invert_field_route(DARLING_TAG,
                                  bsa_dark_override=DARLING_VEIL_ADJUSTED_BSA)
        out = {
            "Kift2013": dict(K=r_k["K_const"], optical=r_k["optical_factor"],
                             se=None, se_parts={}, veiled_K=None,
                             veiled_optical=None),
            "Darling2013": dict(K=r_d["K_const"], optical=r_d["optical_factor"],
                                se=se_d, se_parts=parts_d,
                                veiled_K=r_dv["K_const"],
                                veiled_optical=r_dv["optical_factor"]),
        }
    _FIELD_CACHE[mode] = out
    return out


def darling_se_K(point_K, params=None):
    """Delta-method SE on Darling's implied K_const, from Table 3's own cells.

    THE POINT OF THIS FUNCTION. Before today this route had NO variance, so
    it could not enter an inverse-variance or random-effects pool at all --
    the repository's own VARIANCE_PROVENANCE register said Darling "is NOT
    HELD IN THIS REPOSITORY" and that "no dispersion of any kind reaches the
    repo for this route". The paper is now held, and its Table 3 prints mean,
    SD and n for every one of the six cells the inversion consumes. That is
    a sampling variance, measured, not imputed.

    THE n USED IS TABLE 3's OWN PER-CELL n (113/128/80 white,
    18/43/26 South Asian) -- NOT the 90/35 four-visit completer counts Webb
    2021 Table 5 labels this route with, and not the abstract's 105/35 or
    Table 1's 135/46. Each Table 3 cell is a different season with a
    different number of attendees; using one study-level n for all six would
    over-state the precision of the sparse cells (the South Asian summer
    dosimetry cell is n=18, not 35) and under-state it for the dense ones.
    See DARLING_N_INCONSISTENCY.

    METHOD: numerical Jacobian. Each cell is perturbed by +/- its own
    SE = SD / sqrt(n), the inversion is re-run, and the six partial
    derivatives are summed in quadrature. The cells are treated as
    INDEPENDENT. They are not -- summer and winter are repeated measures on
    overlapping women, and Darling's own Table 4 puts the between-individual
    share of unexplained variance at DARLING_MLM_ICC -- but a positive
    correlation between the summer and winter cells would SHRINK the
    variance of their contrast, so independence is the WIDER, conservative
    choice and is the one taken.

    Returns (se_K, {cell: partial derivative dK/dcell * SE(cell)}).
    """
    parts = {}
    for side, cohort in (("light", "white"), ("dark", "southasian")):
        for fieldname, (mean, sd, n) in DARLING_T3[cohort].items():
            se_cell = sd / np.sqrt(n)
            hi = invert_field_route(
                DARLING_TAG, cells={side: {fieldname: mean + se_cell}},
                params=params)["K_const"]
            lo = invert_field_route(
                DARLING_TAG, cells={side: {fieldname: mean - se_cell}},
                params=params)["K_const"]
            parts[f"{cohort}.{fieldname}"] = (hi - lo) / 2.0
    se = float(np.sqrt(sum(v ** 2 for v in parts.values())))
    return se, parts


def max_admissible_weight_probe(pooled_fn, routes, missing_study, grid=None):
    """How much could a variance-less route move the pool if it had one?

    A route with no published variance cannot be weighted, so it is dropped
    from a random-effects pool. Dropping it is only safe if its ABSENCE is
    not load-bearing. This probe re-admits the route at a sweep of assumed
    weights -- from negligible up to the largest weight any route in the pool
    carries -- and reports the largest movement in the pooled estimate. It
    IMPUTES NOTHING: the sweep is a bound, not an estimate, and its output is
    a sensitivity figure, never a pooled value.
    """
    base = pooled_fn([r for r in routes if r["study"] != missing_study])["K"]
    present = [r for r in routes
               if r["study"] != missing_study and r["se"] is not None]
    if not present:
        return base, 0.0, []
    se_min = min(r["se"] for r in present)      # the most precise route
    if grid is None:
        grid = [se_min * f for f in (8.0, 4.0, 2.0, 1.0)]
    rows = []
    worst = 0.0
    for se_assumed in grid:
        alt = [dict(r, se=(se_assumed if r["study"] == missing_study
                           else r["se"])) for r in routes]
        k = pooled_fn(alt)["K"]
        rows.append((se_assumed, k, k - base))
        worst = max(worst, abs(k - base))
    return base, worst, rows


def se_ln_ratio_from_group_sds(num, den):
    """Delta-method SE on ln(mean_num / mean_den) from two group summaries.

    Each argument is (mean, sd, n). Exact for the first-order expansion; the
    two groups are independent (different subjects), so the covariance term
    is zero.
    """
    (mn, sn, nn), (md, sd, nd) = num, den
    return float(np.hypot((sn / np.sqrt(nn)) / mn, (sd / np.sqrt(nd)) / md))


def se_K_from_se_ln_ratio(obs, fn, se_ln, h=0.05):
    """Propagate an SE on ln(observed ratio) onto the implied K_const.

    The map from observed ratio to implied K is the model inversion
    `invert_one`, which is not linear, so the derivative dK/d(ln r) is taken
    NUMERICALLY at the point estimate by a symmetric difference of +/- `h`
    in log-ratio space. Returns (se_K, dK_dlnr).

    [LIMIT] This is a local, first-order propagation. Felton's interval is
    wide enough that its upper end is unreachable at any K_const this model
    structure admits (the ratio stops being monotone above K_HI = 3.0), so
    the true interval on Felton's K is ASYMMETRIC and truncated above. The
    symmetric SE below is therefore an UNDER-statement of Felton's
    uncertainty, which biases the weighting AGAINST the change being made
    here. Recorded rather than corrected: every correction to it makes
    Felton's weight smaller still.
    """
    k_up, flag_up = invert_one(obs * float(np.exp(h)), fn)
    k_dn, flag_dn = invert_one(obs * float(np.exp(-h)), fn)
    d = (k_up - k_dn) / (2.0 * h)
    if d == 0.0:
        # Both perturbations returned the same value, which in practice means
        # the route is pinned at a search bound. The SE is NOT zero -- it is
        # UNKNOWN, and returning 0.0 would hand this route infinite weight in
        # an inverse-variance pool, or (if the pool refuses zero SEs) drop it
        # without saying so. Return None and shout. See the K_TIMES_M_HI note.
        print(f"  [WARNING] se_K_from_se_ln_ratio: dK/d(ln r) is exactly 0 "
              f"(bounds {flag_dn!r}/{flag_up!r} at K={k_dn:.6f}/{k_up:.6f}). "
              f"This route is pinned at a search bound; its SE is UNKNOWN, "
              f"not zero. Returning None.")
        return None, 0.0
    return float(abs(d) * se_ln), float(d)


# ---------------------------------------------------------------------------
# WEIGHTING SCHEMES -- explicit and selectable
# ---------------------------------------------------------------------------
# [CHOICE 2026-09-22, W-W] The estimator's weighting is now a named, selectable
# scheme rather than an implicit arithmetic mean. The incumbent scheme is
# RETAINED and stays computable, so every published figure derived under it
# remains reproducible and the change is auditable in one run.
#
#   one_vote_per_study
#       The incumbent. Arithmetic mean of the four study-level implied
#       K_const values, which (because ln(V:II) is linear in K_const) is
#       exactly the equal-weight least-squares solution in log-ratio space.
#
#   inverse_variance                                            <-- DEFAULT
#       Inverse-variance weighting WHERE A VARIANCE EXISTS, equal weighting
#       where none does, with the lab/field balance of the incumbent
#       estimator left exactly as it was. Concretely: routes are grouped into
#       the two METHOD TIERS the incumbent already implies (2 lab studies,
#       2 field studies, hence 50/50); inside a tier, routes with a measured
#       variance are combined by inverse variance; a tier in which no route
#       has a variance stays equal-weighted because there is nothing to
#       weight by. The tier split is INHERITED from the incumbent, not
#       invented here: under one-vote-per-study the lab pair and the field
#       pair already carry 50% each, and this scheme changes only the split
#       WITHIN the lab tier, which is the one place a measured variance
#       exists. Nothing is imputed.
#
#   inverse_variance_complete_case
#       Textbook complete-case inverse-variance meta-analysis: routes with no
#       published variance are DROPPED. Reported as a diagnostic, not used as
#       the default, because here it drops Kift AND Darling -- i.e. both field
#       routes, the only sun-exposure evidence in the estimator and the only
#       routes measured outside a cabinet. CLAUDE.md prefers sun exposure for
#       clinical conclusions; discarding it to satisfy an estimator would be
#       the tail wagging the dog. Its value is the honest strict-IV bound.
#
#   inverse_variance_within_study
#       The minimal change: one vote per study as before, but Young's two
#       source arms are combined by inverse variance instead of by a plain
#       mean. Isolates how much of the movement is the Felton down-weighting
#       (nearly all of it) rather than anything inside Young.
#
#   random_effects_dl                                           <-- DEFAULT
#       DerSimonian-Laird random effects over the routes that HAVE a
#       measured variance. Chosen as the default because the fixed-effect
#       assumption these routes would need -- that all four estimate one
#       common K_const with only sampling error between them -- is the very
#       thing S3 measures and REJECTS: the lab pair misses in opposite
#       directions by ~40% and cannot be a pair of noisy reads of one
#       number. Under fixed effects that disagreement is silently reported
#       as precision; under random effects it is estimated as tau^2 and
#       carried into the interval, which is what an honest summary of four
#       mutually inconsistent routes looks like. I^2 is reported beside it.
#       Kift 2013 has no usable variance and is EXCLUDED from the pool (see
#       VARIANCE_PROVENANCE); the max-admissible-weight probe bounds what
#       that exclusion could be hiding.
#
#   random_effects_pm / random_effects_reml
#       The same estimator with Paule-Mandel and REML tau^2 in place of
#       DerSimonian-Laird. Reported as the tau^2-estimator RANGE, not as
#       alternatives to choose between: DL is known to under-estimate tau^2
#       when k is small, and k is 3 here.
WEIGHTING_SCHEMES = ("one_vote_per_study",
                     "inverse_variance",
                     "inverse_variance_complete_case",
                     "inverse_variance_within_study",
                     "random_effects_dl",
                     "random_effects_pm",
                     "random_effects_reml")

RANDOM_EFFECTS_SCHEMES = ("random_effects_dl",
                          "random_effects_pm",
                          "random_effects_reml")
TAU2_METHOD_OF = {"random_effects_dl": "DL",
                  "random_effects_pm": "PM",
                  "random_effects_reml": "REML"}


def tau2_dersimonian_laird(y, s2):
    """DerSimonian-Laird moment estimator of tau^2."""
    w = 1.0 / np.asarray(s2, dtype=float)
    y = np.asarray(y, dtype=float)
    k = len(y)
    if k < 2:
        return 0.0, 0.0, 0.0
    mu_fe = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - mu_fe) ** 2))
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    return tau2, Q, float(k - 1)


def tau2_paule_mandel(y, s2, tol=1e-12, itmax=200):
    """Paule-Mandel iterative moment estimator of tau^2."""
    y = np.asarray(y, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    k = len(y)
    if k < 2:
        return 0.0
    tau2 = 0.0
    for _ in range(itmax):
        w = 1.0 / (s2 + tau2)
        mu = np.sum(w * y) / np.sum(w)
        F = np.sum(w * (y - mu) ** 2) - (k - 1)
        if abs(F) < tol:
            break
        dF = -np.sum(w ** 2 * (y - mu) ** 2)
        if dF == 0:
            break
        step = F / dF
        new = tau2 - step
        tau2 = max(0.0, float(new))
        if tau2 == 0.0 and F < 0:
            break
    return float(max(0.0, tau2))


def tau2_reml(y, s2, tol=1e-12, itmax=500):
    """REML (Fisher-scoring / fixed-point) estimator of tau^2."""
    y = np.asarray(y, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    k = len(y)
    if k < 2:
        return 0.0
    tau2 = max(0.0, float(np.var(y, ddof=1) - np.mean(s2)))
    for _ in range(itmax):
        w = 1.0 / (s2 + tau2)
        mu = np.sum(w * y) / np.sum(w)
        num = np.sum(w ** 2 * ((y - mu) ** 2 - s2)) + 1.0 / np.sum(w) * np.sum(w ** 2)
        den = np.sum(w ** 2)
        new = max(0.0, float(num / den))
        if abs(new - tau2) < tol:
            tau2 = new
            break
        tau2 = new
    return float(max(0.0, tau2))


TAU2_ESTIMATORS = {"DL": lambda y, s2: tau2_dersimonian_laird(y, s2)[0],
                   "PM": tau2_paule_mandel,
                   "REML": tau2_reml}


def pool_random_effects(routes, tau2_method="DL"):
    """Random-effects pool over the routes that carry a measured variance.

    Returns the pooled K, its SE, tau^2, I^2, Q, the 95% interval, and the
    per-route weights. Routes with `se is None` are DROPPED and listed --
    nothing is imputed to keep them in.

    The interval is the ordinary random-effects normal interval. A
    Hartung-Knapp-Sidik-Jonkman interval is ALSO computed and printed, but
    it is not the headline: at k = 3 the HKSJ t-multiplier is t(2) = 4.303
    and the resulting interval is so wide as to be uninformative. Reported
    so the reader can see that, not so anyone quotes it.
    """
    used = [r for r in routes if r["se"] is not None and r["se"] > 0]
    dropped = [r["study"] for r in routes if r["se"] is None or r["se"] <= 0]
    if not used:
        return {"scheme": "random_effects_" + tau2_method.lower(),
                "K": None, "se": None, "weights": {},
                "routes_used": [], "dropped": dropped}
    y = np.array([r["K"] for r in used], dtype=float)
    s2 = np.array([r["se"] ** 2 for r in used], dtype=float)
    tau2 = TAU2_ESTIMATORS[tau2_method](y, s2)
    _dl_tau2, Q, dfree = tau2_dersimonian_laird(y, s2)
    I2 = max(0.0, (Q - dfree) / Q) if Q > 0 else 0.0
    w = 1.0 / (s2 + tau2)
    K = float(np.sum(w * y) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    # HKSJ variance, reported and deliberately not adopted at k=3.
    k = len(used)
    q_hk = float(np.sum(w * (y - K) ** 2) / (k - 1)) if k > 1 else float("nan")
    se_hksj = float(np.sqrt(q_hk / np.sum(w))) if k > 1 else float("nan")
    t_mult = {2: 4.302652729911275, 3: 3.182446305284263,
              4: 2.7764451051977987}.get(k - 1, 1.959963984540054)
    return {"scheme": "random_effects_" + tau2_method.lower(),
            "K": K, "se": se, "tau2": float(tau2), "tau": float(np.sqrt(tau2)),
            "I2": float(I2), "Q": float(Q), "df": float(dfree), "k": k,
            "ci": (K - 1.959963984540054 * se, K + 1.959963984540054 * se),
            "ci_hksj": (K - t_mult * se_hksj, K + t_mult * se_hksj),
            "se_hksj": se_hksj, "t_mult": t_mult,
            "weights": {r["study"]: float(wi / np.sum(w))
                        for r, wi in zip(used, w)},
            "routes_used": [r["study"] for r in used], "dropped": dropped}

#: [CHOICE 2026-09-22, SUPERSEDING the same day's inverse_variance default]
#: RANDOM EFFECTS (DerSimonian-Laird) is the default. The fixed-effect
#: inverse-variance default that W-W landed earlier today assumes the routes
#: differ only by sampling error; S3 of this very script measures that they
#: do not, and the I^2 printed in S3e quantifies it. A fixed-effect pool of
#: routes this heterogeneous reports a precision that does not exist.
#: `inverse_variance` and `one_vote_per_study` remain computable in the same
#: run so every previously published figure stays reproducible. See
#: calibration_approval.yaml, K_const_random_effects_2026_09_22.
DEFAULT_WEIGHTING_SCHEME = "random_effects_dl"


def combine(values, ses):
    """Combine estimates: inverse-variance if every SE is present, else mean.

    Returns (estimate, pooled_se_or_None, weights_as_fractions).
    """
    values = list(values)
    if not values:
        return None, None, []
    if len(values) == 1:
        return float(values[0]), (float(ses[0]) if ses[0] is not None else None), [1.0]
    if all(s is not None and s > 0 for s in ses):
        w = [1.0 / s ** 2 for s in ses]
        tot = sum(w)
        est = sum(wi * v for wi, v in zip(w, values)) / tot
        return float(est), float(np.sqrt(1.0 / tot)), [wi / tot for wi in w]
    n = len(values)
    return float(np.mean(values)), None, [1.0 / n] * n


def pool_routes(routes, scheme=None):
    """Pool study-level routes under a named weighting scheme.

    `routes` is a list of dicts with keys: study, label, K, se (may be None),
    tier ('lab' or 'field'). Returns a dict with the pooled estimate, the
    per-route effective weight, and the pooled SE where one is defined.
    """
    scheme = scheme or DEFAULT_WEIGHTING_SCHEME
    if scheme not in WEIGHTING_SCHEMES:
        raise ValueError(f"unknown weighting scheme {scheme!r}; "
                         f"expected one of {WEIGHTING_SCHEMES}")
    used = list(routes)
    if scheme in RANDOM_EFFECTS_SCHEMES:
        return pool_random_effects(routes, TAU2_METHOD_OF[scheme])
    if scheme in ("one_vote_per_study", "inverse_variance_within_study"):
        K = float(np.mean([r["K"] for r in used]))
        w = {r["study"]: 1.0 / len(used) for r in used}
        return {"scheme": scheme, "K": K, "se": None, "weights": w,
                "routes_used": [r["study"] for r in used], "dropped": []}
    if scheme == "inverse_variance_complete_case":
        used = [r for r in routes if r["se"] is not None]
        dropped = [r["study"] for r in routes if r["se"] is None]
        K, se, fr = combine([r["K"] for r in used], [r["se"] for r in used])
        return {"scheme": scheme, "K": K, "se": se,
                "weights": {r["study"]: f for r, f in zip(used, fr)},
                "routes_used": [r["study"] for r in used], "dropped": dropped}
    # inverse_variance: IV within tier, tiers equal-weighted
    tiers, weights = [], {}
    for tier in ("lab", "field"):
        members = [r for r in routes if r["tier"] == tier]
        if not members:
            continue
        est, _se, fr = combine([r["K"] for r in members],
                               [r["se"] for r in members])
        tiers.append(est)
        for r, f in zip(members, fr):
            weights[r["study"]] = f      # scaled by the tier share below
    share = 1.0 / len(tiers)
    weights = {k: v * share for k, v in weights.items()}
    return {"scheme": scheme, "K": float(np.mean(tiers)), "se": None,
            "weights": weights, "routes_used": [r["study"] for r in routes],
            "dropped": []}


# ===========================================================================
# Model wrappers -- each reproduces ONE trial's protocol
# ===========================================================================

def felton_gain(fst: int, K: float, params: ModelParams = None) -> float:
    """Modelled 25(OH)D gain over Felton 2016's 6-week 3x/week course."""
    p = params or ModelParams()
    sched = [(w * 7 + k * 2, FELTON_SED_PER_SESSION)
             for w in range(FELTON_WEEKS)
             for k in range(FELTON_SESSIONS_PER_WEEK)]
    res = simulate(fst=fst, schedule=sched, duration_days=FELTON_WEEKS * 7,
                   params=p, D0=FELTON_D0[fst], K_const_override=K,
                   E_source_override=get_E_source(FELTON_SOURCE),
                   exposed_bsa_override=FELTON_BSA,
                   dose_framing="fixed_absolute_SED")
    return float(res["D"][-1]) - FELTON_D0[fst]


def _young_sample_days(cum_doses, sed_per_session):
    """Map Young's printed CUMULATIVE dose grid onto exposure-day times.

    The spreadsheet's grid is 0, 2, 4, 8, 10 SED (Arimed B) and
    0, 1.5, 4.5, 6, 7.5 SED (TL01) for five 2.0 / 1.5 SED exposures at 3-4 day
    intervals, i.e. blood was drawn before exposures 1, 2, 3 and 5 and after
    exposure 5, skipping one mid-course draw. Sampling the MODEL on the same
    grid matters: fst_gap_decomposition.py used an evenly spaced 0,2,4,6,8
    grid, which is not the grid the data sit on.
    """
    times = []
    for c in cum_doses:
        n_done = int(round(c / sed_per_session))
        if n_done == 0:
            times.append(0.0)                          # pre-exposure draw
        elif n_done < 5:
            times.append(n_done * YOUNG_INTERVAL_D - 1e-6)  # just before next
        else:
            times.append(5 * YOUNG_INTERVAL_D)         # final draw
    return times


def young_slope(fst: int, D0: float, K: float, sed_per_session: float,
                e_source: float, cum_doses,
                params: ModelParams = None) -> float:
    """Modelled d[25(OH)D3]/d[cumulative SED] on Young 2020's own dose grid."""
    p = params or ModelParams()
    sched = [(i * YOUNG_INTERVAL_D, sed_per_session) for i in range(5)]
    times = _young_sample_days(cum_doses, sed_per_session)
    res = simulate(fst=fst, schedule=sched,
                   duration_days=int(np.ceil(max(times))) + 1, params=p,
                   D0=D0, K_const_override=K, E_source_override=e_source,
                   exposed_bsa_override=YOUNG_BSA,
                   dose_framing="fixed_absolute_SED")
    t = np.asarray(res["t"], dtype=float)
    D = np.asarray(res["D"], dtype=float)
    obs = [D[int(np.argmin(np.abs(t - tt)))] for tt in times]
    return float(np.polyfit(np.asarray(cum_doses, dtype=float), obs, 1)[0])


# ===========================================================================
# Young 2020 individual-level loader
# ===========================================================================

def load_young():
    """Per-subject observed slope and baseline, plus the arm's protocol grid."""
    try:
        import openpyxl
    except ImportError:
        return None
    if not os.path.exists(YOUNG_XLSX):
        return None
    wb = openpyxl.load_workbook(YOUNG_XLSX, read_only=True, data_only=True)
    arms = {}
    for sheet, (ukey, esrc, sed) in YOUNG_ARMS.items():
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
            full = [d.get(ukey + str(i)) for i in range(1, 6)]
            subs.append(dict(fst=FST_ROMAN[d["FST"]], eth=d["Ethnicity"],
                             slope=float(np.polyfit(xs, ys, 1)[0]),
                             base=ys[0],
                             full_doses=tuple(float(v) for v in full
                                              if v is not None)))
        # The canonical grid is the PROTOCOL grid = the modal complete UVdose
        # row. Individual subjects miss blood draws; taking the first
        # subject's surviving points as "the grid" silently shortens it.
        modal = collections.Counter(s["full_doses"] for s in subs
                                    if len(s["full_doses"]) == 5).most_common(1)
        arms[sheet] = dict(subjects=subs, e_source=get_E_source(esrc), sed=sed,
                           source_key=esrc,
                           doses=list(modal[0][0]) if modal else None,
                           n_on_grid=modal[0][1] if modal else 0)
    return arms


def young_group_table(arms):
    """Group means per FST per arm."""
    out = {}
    for sheet, arm in arms.items():
        by = collections.defaultdict(list)
        for s in arm["subjects"]:
            by[s["fst"]].append(s)
        groups = {}
        for f, g in by.items():
            groups[f] = dict(
                n=len(g),
                base=float(np.mean([s["base"] for s in g])),
                slope=float(np.mean([s["slope"] for s in g])),
                slope_sd=(float(np.std([s["slope"] for s in g], ddof=1))
                          if len(g) > 1 else float("nan")),
                eth=dict(collections.Counter(s["eth"] for s in g)))
        out[sheet] = groups
    return out


# ===========================================================================
# S1 -- is the model's feedback the right strength to justify a RAW target?
# ===========================================================================

def s1_feedback_check(arms):
    hr("S1  DOES THE MODEL'S FEEDBACK MATCH YOUNG 2020's OWN BASELINE EFFECT?")
    print("""
  Young 2020's published ratios (1.29 Arimed B, 1.43 TL01) come from Table 3,
  whose own note says the model is "adjusted for pre-exposure 25(OH)D3, slope,
  and intercept". This model is instead handed each group's MEASURED
  pre-exposure value and generates its own baseline correction through the
  (1 - D/D_max) feedback term. A model prediction is therefore a RAW slope and
  belongs against a RAW observed slope -- unless the model's feedback is the
  wrong strength, in which case neither pairing is clean.

  Test, WITHIN FST II ONLY, so skin type is held fixed and the gradient is a
  pure baseline effect rather than a baseline-x-FST confound: regress observed
  individual slope on observed individual baseline, then compare with the
  model's own gradient over the same baseline range.""")
    p = ModelParams()
    rows = []
    for sheet, arm in arms.items():
        subs = [s for s in arm["subjects"] if s["fst"] == 2]
        b = np.array([s["base"] for s in subs])
        sl = np.array([s["slope"] for s in subs])
        m_obs = float(np.polyfit(b, sl, 1)[0])
        r = float(np.corrcoef(b, sl)[0, 1])
        lo, hi = float(b.min()), float(b.max())
        d0s = np.linspace(lo, hi, 7)
        mod = [young_slope(2, d0, OLD_K_CONST, arm["sed"], arm["e_source"],
                           arm["doses"]) for d0 in d0s]
        m_mod = float(np.polyfit(d0s, mod, 1)[0])
        ratio = m_mod / m_obs if m_obs != 0 else float("nan")
        rows.append((sheet, m_obs, m_mod, r, len(subs), ratio))
        print(f"\n  {sheet}  (FST II only, n = {len(subs)}, "
              f"baseline {lo:.1f}-{hi:.1f} nmol/L)")
        print(f"    observed d(slope)/d(baseline) = {m_obs:+.4f} "
              f"[nmol/L per SED] per [nmol/L]   (Pearson r = {r:+.2f})")
        print(f"    model    d(slope)/d(baseline) = {m_mod:+.4f}   "
              f"[K_const = {OLD_K_CONST:.3f}, D_max = {p.D_max:.0f}]")
        print(f"    model / observed = {ratio:.2f}x")

    rr = [x[5] for x in rows if np.isfinite(x[5])]
    agree = all(0.5 <= x <= 2.0 for x in rr)
    print(f"\n  [derived] model:observed baseline gradient spans "
          f"{min(rr):.2f}x to {max(rr):.2f}x across the two arms.")
    if agree:
        print("""
  READING (derived from the numbers above, not asserted). Both gradients are
  negative in the data and in the model, and they agree to within a factor of
  two. The model's feedback is then a usable stand-in for Young's ANCOVA and
  the RAW and ADJUSTED pairings should give similar answers; S2 checks that
  they do rather than assuming it.""")
    else:
        print("""
  READING (derived from the numbers above, not asserted). Both gradients are
  negative -- the sign the feedback term claims -- but the model's is only
  """ + f"{min(rr):.2f}-{max(rr):.2f}" + """x of the measured one. The model
  UNDER-corrects for baseline. It therefore cannot be used as a substitute for
  Young's own ANCOVA, and the RAW pairing is unsafe: Young's dark groups start
  much lower (Arimed B FST VI 16.6 vs FST II 39.9 nmol/L), so a raw target
  would push K_const down to cover a shortfall in the FEEDBACK term that has
  nothing to do with pigment.

  [CHOICE] The primary Young pairing is therefore 'adjusted@pooled': Young's
  ANCOVA removes the baseline difference from the DATA, and running both model
  arms from the same baseline removes it from the MODEL. That pairing is the
  only one of the three whose answer does not depend on the feedback term
  being the right strength. The other two are computed anyway, in S2, so the
  cost of the choice is visible.""")
    return rows, agree


# ===========================================================================
# S2 -- per-contrast inversion, then the joint fit
# ===========================================================================

# [CHOICE] Weight one per STUDY, not one per contrast and not one per subject.
# Young's two arms share one recruitment, one group and one protocol design,
# so they are not two independent studies. Weighting by subject count would
# let Young's n = 67 erase Felton's n = 16, even though Felton is the only
# within-study FST V comparison with both baselines AND both endpoints
# printed. Every alternative weighting is reported in S4.
STUDY_WEIGHTS = {"Felton2016": 1.0, "Young2020": 1.0}

# The Young target and the baseline it is simulated at must be PAIRED: an
# ANCOVA-adjusted ratio belongs with a common baseline, a raw ratio with each
# group's own. "adjusted@own" is the mismatched pairing that
# fst_gap_decomposition.py used; it is carried here so the difference it makes
# is visible rather than silent.
YOUNG_TREATMENTS = ["adjusted@pooled", "raw@own", "adjusted@own"]


def build_targets(arms, groups, young_treatment="adjusted@pooled"):
    """Return [(study, label, observed ratio, callable(K) -> model ratio)]."""
    t = []
    obs_f = ((FELTON_FINAL[2] - FELTON_D0[2])
             / (FELTON_FINAL[5] - FELTON_D0[5]))
    t.append(("Felton2016", "Felton 2016 FST II:V gain, 23.4 SED, 35% BSA",
              obs_f, lambda K: felton_gain(2, K) / felton_gain(5, K)))

    for sheet, (a, b, adj) in YOUNG_PUBLISHED_ADJUSTED.items():
        if sheet not in groups:
            continue
        fa, fb = FST_ROMAN[a], FST_ROMAN[b]
        g, arm = groups[sheet], arms[sheet]
        pool = float(np.mean([s["base"] for s in arm["subjects"]]))
        if young_treatment == "adjusted@pooled":
            obs, d0a, d0b = adj, pool, pool
        elif young_treatment == "adjusted@own":
            obs, d0a, d0b = adj, g[fa]["base"], g[fb]["base"]
        elif young_treatment == "raw@own":
            obs = g[fa]["slope"] / g[fb]["slope"]
            d0a, d0b = g[fa]["base"], g[fb]["base"]
        else:
            raise ValueError(young_treatment)
        label = (f"Young 2020 {sheet.split()[0]:8s} FST {a}:{b} "
                 f"[{young_treatment}]")

        def mk(fa=fa, fb=fb, d0a=d0a, d0b=d0b, arm=arm):
            return lambda K: (
                young_slope(fa, d0a, K, arm["sed"], arm["e_source"],
                            arm["doses"])
                / young_slope(fb, d0b, K, arm["sed"], arm["e_source"],
                              arm["doses"]))
        t.append(("Young2020", label, obs, mk()))
    return t


def objective(K, targets, weights=None):
    """Weighted sum of squared LOG ratio errors.

    Log space  # [CHOICE]: a ratio and its reciprocal must cost the same, and
    the calibrated quantity exp(K * dM) is log-linear in K, so a log residual
    is linear in the parameter being fitted.
    """
    w = weights or STUDY_WEIGHTS
    per_study = collections.defaultdict(list)
    for study, _label, obs, fn in targets:
        mod = fn(K)
        per_study[study].append(1e6 if (not np.isfinite(mod) or mod <= 0)
                                else (np.log(mod) - np.log(obs)) ** 2)
    return sum(w.get(s, 1.0) * float(np.mean(v)) for s, v in per_study.items())


# [BUG FIX 2026-09-22] The search ceiling is PHYSICAL IN K_const * M_baseline,
# not in K_const. Above a certain optical penalty the darker arm's modelled
# net response turns negative (decay exceeds synthesis) and the ratio stops
# being monotone -- but where that happens depends on the product, so a
# ceiling written as a bare number in K units silently moves whenever the
# ITA -> M mapping moves.
#
# K_HI was hardcoded at 3.0, which was that bound expressed on the M scale in
# force before 2026-09-22, where M_baseline(V) was 0.7539. Releasing the FST
# VI clip (decision A3, option 1C) compressed the scale by 1/1.4237, which
# pushed Felton 2016's required K_const from 2.138 to just above the old 3.0
# bound. THE FAILURE MODE THIS CAUSED IS WORTH RECORDING: `invert_one`
# returned the bound for the point estimate AND for both +/-h perturbations,
# so the numerical derivative dK/d(ln r) came out EXACTLY ZERO, Felton's
# SE(K) came out exactly zero, and a pool that (correctly) refuses a
# zero-or-missing SE dropped the only within-study FST V route without
# saying so. A stale bound expressed in the wrong units removed a study from
# a meta-analysis silently.
#
# 2.2617 = 3.0 * 0.7539, i.e. the old bound restated as a ceiling on the
# exponent argument, which is the quantity that is actually scale-free.
K_TIMES_M_HI = 2.2617


def k_hi(params=None):
    """Search ceiling for K_const on the CURRENT M scale."""
    p = params or ModelParams()
    return float(K_TIMES_M_HI / p.M_baseline(5))


K_HI = k_hi()


def minimise(targets, weights=None, lo=0.0, hi=K_HI, n=61):
    grid = np.linspace(lo, hi, n)
    vals = [objective(k, targets, weights) for k in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(50):
        if objective(c, targets, weights) < objective(d, targets, weights):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    K = 0.5 * (a + b)
    return float(K), float(objective(K, targets, weights)), grid, vals


def invert_one(obs, fn, lo=0.0, hi=K_HI, n=121):
    """K such that fn(K) == obs. Returns (K, bound_flag).

    Plain bisection is NOT safe here. fn rises with K only while the darker
    arm's modelled net response stays positive; past that point the ratio
    turns over and goes negative, so a bisection that only tests the two
    endpoints reads fn(hi) < obs and wrongly reports 'at-bound-high'. This
    scans a grid for the FIRST upward crossing and bisects inside it.
    """
    grid = np.linspace(lo, hi, n)
    vals = [fn(k) for k in grid]
    if vals[0] >= obs:
        return lo, "at-bound-low"
    for i in range(1, n):
        if np.isfinite(vals[i]) and vals[i] > 0 and vals[i] >= obs:
            a, b = grid[i - 1], grid[i]
            for _ in range(45):
                mid = 0.5 * (a + b)
                if fn(mid) < obs:
                    a = mid
                else:
                    b = mid
            return 0.5 * (a + b), None
        if np.isfinite(vals[i]) and vals[i] <= 0:
            # ratio has turned over before reaching the target
            return grid[i - 1], "no-crossing-below-turnover"
    return hi, "at-bound-high"


# ===========================================================================
# reporting helpers
# ===========================================================================

def optical(K, f_dark, f_light=2, p=None):
    p = p or ModelParams()
    return float(np.exp(K * (p.M_baseline(f_dark) - p.M_baseline(f_light))))


def s3c_variance_audit(arms, groups, per_contrast, sources,
                       field_routes=None):
    """Print the per-route variance-availability audit and return the register.

    Returns {study: dict(K, se, se_ln, dK, tier, K_iv_arms)} where `se` is
    None for any route whose source publishes no usable variance on the
    quantity this script inverts. Nothing is imputed.
    """
    hr("S3c  VARIANCE-AVAILABILITY AUDIT -- what each source actually publishes")
    print("""
  Inverse-variance weighting needs a variance PER ROUTE, on the quantity the
  route inverts. This section establishes, route by route, whether one
  exists. Where it does not, none is invented: the route keeps a plain vote
  inside its method tier and the gap is printed, because a study with no
  reported variance is a real limitation of this method and must stay
  visible rather than be smoothed over.""")

    field_routes = field_routes or DEFAULT_FIELD_ROUTES
    reg = {}

    # --- Felton 2016 -------------------------------------------------------
    obs_f = ((FELTON_FINAL[2] - FELTON_D0[2])
             / (FELTON_FINAL[5] - FELTON_D0[5]))
    fn_f = lambda K: felton_gain(2, K) / felton_gain(5, K)  # noqa: E731
    se_ln_f = se_ln_ratio_from_group_sds(
        (FELTON_FINAL[2] - FELTON_D0[2], FELTON_GAIN_SD[2], FELTON_N[2]),
        (FELTON_FINAL[5] - FELTON_D0[5], FELTON_GAIN_SD[5], FELTON_N[5]))
    K_f = [K for (study, _t, _l, _o, K, _f) in per_contrast
           if study == "Felton2016"][0]
    se_K_f, dK_f = se_K_from_se_ln_ratio(obs_f, fn_f, se_ln_f)
    reg["Felton2016"] = dict(K=K_f, se=se_K_f, se_ln=se_ln_f, dK=dK_f,
                             tier="lab", K_iv_arms=K_f)

    # --- Young 2020 --------------------------------------------------------
    primary = YOUNG_TREATMENTS[0]
    arm_K, arm_se = [], []
    for sheet, (a, b, adj) in YOUNG_PUBLISHED_ADJUSTED.items():
        if sheet not in groups:
            continue
        fa, fb = FST_ROMAN[a], FST_ROMAN[b]
        g, arm = groups[sheet], arms[sheet]
        pool = float(np.mean([s["base"] for s in arm["subjects"]]))

        def fn(K, fa=fa, fb=fb, pool=pool, arm=arm):
            return (young_slope(fa, pool, K, arm["sed"], arm["e_source"],
                                arm["doses"])
                    / young_slope(fb, pool, K, arm["sed"], arm["e_source"],
                                  arm["doses"]))

        K_arm = [K for (study, tr, lab, _o, K, _f) in per_contrast
                 if study == "Young2020" and tr == primary
                 and sheet.split()[0] in lab][0]
        se_ln = se_ln_ratio_from_group_sds(
            (g[fa]["slope"], g[fa]["slope_sd"], g[fa]["n"]),
            (g[fb]["slope"], g[fb]["slope_sd"], g[fb]["n"]))
        se_K, dK = se_K_from_se_ln_ratio(adj, fn, se_ln)
        arm_K.append(K_arm)
        arm_se.append(se_K)
        print(f"\n  Young 2020 {sheet:22s} FST {a}:{b}  n = {g[fa]['n']} vs "
              f"{g[fb]['n']}")
        print(f"    per-subject slope SD  {g[fa]['slope_sd']:.5f} / "
              f"{g[fb]['slope_sd']:.5f} nmol/L per SED")
        print(f"    SE(ln observed ratio) {se_ln:.6f}   dK/d(ln r) "
              f"{dK:.6f}   SE(K) {se_K:.6f}")
    K_y_equal = float(np.mean(arm_K))
    K_y_iv, se_y_iv, _fr = combine(arm_K, arm_se)
    se_y_equal = (None if any(s is None for s in arm_se) else
                  float(np.sqrt(sum(s ** 2 for s in arm_se)) / len(arm_se)))
    reg["Young2020"] = dict(K=K_y_equal, se=se_y_equal, se_ln=None, dK=None,
                            tier="lab", K_iv_arms=K_y_iv, se_iv_arms=se_y_iv)

    print(f"\n  Young 2020 study level: arms averaged {K_y_equal:.6f} "
          f"(SE {se_y_equal:.6f}); arms inverse-variance combined "
          f"{K_y_iv:.6f} (SE {se_y_iv:.6f})")
    print(f"\n  Felton 2016: SE(ln observed ratio) {se_ln_f:.6f}   "
          f"dK/d(ln r) {dK_f:.6f}   SE(K) {se_K_f:.6f}")

    # --- field routes ------------------------------------------------------
    # [BUG FIX 2026-09-22] RE-INVERTED LIVE. These were a 2026-08-17 cache
    # (K 0.788 / 1.387) carried through two refits on the false belief that
    # the field routes do not depend on M_baseline_by_FST. They do: the
    # inversion calls simulate(). See invert_field_route above.
    fr = field_route_values(field_routes)
    k_kift, opt_kift = fr["Kift2013"]["K"], fr["Kift2013"]["optical"]
    k_darl, opt_darl = fr["Darling2013"]["K"], fr["Darling2013"]["optical"]
    se_darl, darl_parts = fr["Darling2013"]["se"], fr["Darling2013"]["se_parts"]
    darl_veiled = (None if fr["Darling2013"]["veiled_K"] is None else
                   {"K_const": fr["Darling2013"]["veiled_K"],
                    "optical_factor": fr["Darling2013"]["veiled_optical"]})
    if field_routes != "carried":
        print(f"\n  RE-INVERTED LIVE on the current tree (was a 2026-08-17 "
              f"cache):")
        print(f"    {'route':22s} {'cached K':>9} {'live K':>9} "
              f"{'cached V:II':>12} {'live V:II':>10}")
        print(f"    {'Kift 2013 / Webb 2010':22s} "
              f"{HELD_OUT_FIELD['Kift 2013 / Webb 2010, behaviour removed'][1]:>9.3f} "
              f"{k_kift:>9.6f} "
              f"{HELD_OUT_FIELD['Kift 2013 / Webb 2010, behaviour removed'][0]:>12.2f} "
              f"{opt_kift:>10.4f}")
        print(f"    {'Darling 2013':22s} "
              f"{HELD_OUT_FIELD['Darling 2013, behaviour removed'][1]:>9.3f} "
              f"{k_darl:>9.6f} "
              f"{HELD_OUT_FIELD['Darling 2013, behaviour removed'][0]:>12.2f} "
              f"{opt_darl:>10.4f}")
        print("""
    The OPTICAL factor is what these routes actually estimate; K_const is
    only its parameterisation on the M scale, so an ITA -> M re-scaling
    moves K and leaves the optical factor untouched. Both moved here
    because BSA_EXP, SC_max, the feedback form, A and the body-weight term
    all moved between 2026-08-17 and today.""")

        print(f"\n  DARLING 2013 NOW CARRIES A MEASURED VARIANCE.")
        print(f"    Delta-method SE on the implied K_const, from Table 3's "
              f"own per-cell means, SDs and n's:")
        for cell, part in sorted(darl_parts.items(),
                                 key=lambda kv: -abs(kv[1])):
            print(f"      {cell:32s} dK contribution {part:+.6f}")
        print(f"      {'quadrature total SE(K)':32s} {se_darl:.6f}")
        print(f"    n USED = Table 3's own per-season cell n, NOT the 90/35 "
              f"four-visit")
        print(f"    completer counts Webb 2021 Table 5 labels this route "
              f"with. Darling's")
        print(f"    own n is inconsistent across the paper: "
              f"{DARLING_N_INCONSISTENCY}")

        print(f"\n  DARK-SKIN FLAG -- DARLING'S VEILING, and its sensitivity.")
        print(f"    Darling Discussion: 75% of the South Asian arm were of "
              f"Pakistani origin")
        print(f"    and 'partly veiled (showing hands and face only)'. The "
              f"inversion equates")
        print(f"    their exposed BSA to Kift's diary figure "
              f"({DARLING_BSA_KIFT_INHERITED:.0%}), which is then too high, "
              f"so the")
        print(f"    unexplained shortfall is charged to PIGMENT and this "
              f"route's K is")
        print(f"    BIASED UPWARD. Variant at "
              f"{DARLING_VEIL_ADJUSTED_BSA:.4f} exposed BSA "
              f"({DARLING_VEIL_FRACTION:.0%} at")
        print(f"    hands-and-face {DARLING_VEIL_EXPOSED_BSA:.2f}, rest at "
              f"{DARLING_BSA_KIFT_INHERITED:.2f}):")
        print(f"      headline (BSA {DARLING_BSA_KIFT_INHERITED:.2f})   "
              f"K = {k_darl:.6f}   FST V:II = {opt_darl:.4f}x")
        print(f"      veiling variant             "
              f"K = {darl_veiled['K_const']:.6f}   FST V:II = "
              f"{darl_veiled['optical_factor']:.4f}x")
        print(f"      movement                    "
              f"K {darl_veiled['K_const'] - k_darl:+.6f}   V:II "
              f"{darl_veiled['optical_factor'] - opt_darl:+.4f}x")
        print(f"    Darling is KEPT in the pool -- it is one of only two "
              f"field routes and")
        print(f"    the only one outside Manchester -- with this flag "
              f"carried at every")
        print(f"    point of use. Darling never uses Fitzpatrick type at "
              f"all; the FST V/II")
        print(f"    labels on its two cohorts are INFERRED FROM ETHNICITY "
              f"by this repo.")

    reg["Kift2013"] = dict(K=k_kift, se=None, se_ln=None, dK=None,
                           tier="field", K_iv_arms=k_kift,
                           optical=opt_kift)
    reg["Darling2013"] = dict(K=k_darl, se=se_darl, se_ln=None, dK=None,
                              tier="field", K_iv_arms=k_darl,
                              optical=opt_darl,
                              veiled=(darl_veiled["K_const"]
                                      if darl_veiled else None))

    print(f"\n  {'route':14s} {'K_const':>9} {'SE(K)':>10}  provenance")
    for study in ("Felton2016", "Young2020", "Kift2013", "Darling2013"):
        r = reg[study]
        se_txt = ("%.6f" % r["se"]) if r["se"] is not None else "NONE"
        print(f"  {study:14s} {r['K']:>9.3f} {se_txt:>10}")
        for line in _wrap(VARIANCE_PROVENANCE[study], 66):
            print(f"  {'':14s} {'':>9} {'':>10}  {line}")

    print(f"""
  [derived] {sum(1 for r in reg.values() if r['se'] is not None)} of 4 routes
  carry a usable variance, as of 2026-09-22 -- two lab (Felton's published
  gain SDs, Young's individual-level slopes) and ONE FIELD route, Darling,
  whose Table 3 cells were read from the paper today. The remaining gap is
  Kift 2013, and it is now the ONLY route with no variance rather than one of
  three. That matters for the estimator's balance: the field tier is no
  longer variance-blind, so the tier-splitting workaround the superseded
  inverse-variance default needed is no longer the only way to keep field
  evidence in the pool.""")
    return reg


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def build_routes(per_route, scheme=None):
    """Turn the S3c register into the list `pool_routes` consumes."""
    scheme = scheme or DEFAULT_WEIGHTING_SCHEME
    out = []
    for study in ("Felton2016", "Young2020", "Kift2013", "Darling2013"):
        r = per_route[study]
        K = r["K"]
        se = r["se"]
        if scheme == "inverse_variance_within_study" and study == "Young2020":
            K = r["K_iv_arms"]
            se = r.get("se_iv_arms", se)
        out.append(dict(study=study, label=study, K=K, se=se, tier=r["tier"]))
    return out


def ratio_table(K, p=None):
    p = p or ModelParams()
    pairs = [(5, 2, "V:II"), (6, 1, "VI:I"), (6, 2, "VI:II"),
             (6, 5, "VI:V"), (4, 2, "IV:II"), (3, 2, "III:II")]
    return [(lab, float(np.exp(K * (p.M_baseline(d) - p.M_baseline(l)))))
            for d, l, lab in pairs]


# ===========================================================================

def main(argv=None):
    global DEFAULT_WEIGHTING_SCHEME, FIELD_ROUTES_MODE
    import argparse
    ap = argparse.ArgumentParser(
        description="K_const cross-trial derivation.")
    ap.add_argument("--scheme", choices=list(WEIGHTING_SCHEMES),
                    default=DEFAULT_WEIGHTING_SCHEME,
                    help="weighting scheme for the headline estimate "
                         "(default: %(default)s). Every scheme is reported "
                         "in S3d regardless; this selects which one S5-S8 "
                         "read as the derived value.")
    ap.add_argument("--field-routes", choices=list(FIELD_ROUTE_MODES),
                    default=DEFAULT_FIELD_ROUTES,
                    help="'rederived' (default) re-inverts the Kift and "
                         "Darling field routes on the CURRENT tree; "
                         "'carried' reads the 2026-08-17 HELD_OUT_FIELD "
                         "cache, retained only so figures published from it "
                         "stay reproducible. The cache is STALE -- the "
                         "routes are not independent of M_baseline_by_FST, "
                         "contrary to what this repo asserted until "
                         "2026-09-22.")
    args = ap.parse_args(argv)
    DEFAULT_WEIGHTING_SCHEME = args.scheme
    FIELD_ROUTES_MODE = args.field_routes
    print(__doc__)
    p = ModelParams()
    arms = load_young()
    if arms is None:
        print("\n[ABORT] openpyxl or the Young 2020 xlsx is unavailable.")
        return
    groups = young_group_table(arms)
    obs_f = (FELTON_FINAL[2] - FELTON_D0[2]) / (FELTON_FINAL[5] - FELTON_D0[5])

    # ---------------------------------------------------------------- S0 ---
    hr("S0  THE CALIBRATION SET, AS READ FROM SOURCE")
    print("\n  Felton et al. 2016, Br J Dermatol 175:1320  (Research_Papers/)")
    print(f"    {FELTON_SED_PER_SESSION} SED x {FELTON_SESSIONS_PER_WEEK}/wk "
          f"x {FELTON_WEEKS} wk = "
          f"{FELTON_SED_PER_SESSION*FELTON_SESSIONS_PER_WEEK*FELTON_WEEKS:.1f}"
          f" SED at {FELTON_BSA:.0%} BSA, same cabinet both arms")
    for f in (2, 5):
        print(f"      phototype {ROMAN_OF[f]:<3} n={FELTON_N[f]:<3} "
              f"{FELTON_D0[f]:.1f} -> {FELTON_FINAL[f]:.1f} nmol/L   "
              f"gain {FELTON_FINAL[f]-FELTON_D0[f]:+.1f}")
    print(f"      observed II:V gain ratio = {obs_f:.3f}    "
          f"[FST V arm: Indian-subcontinent ancestry, UK]")

    print("\n  Young et al. 2020, JID 140:1418  (individual-level xlsx in repo)")
    for sheet, g in groups.items():
        a = arms[sheet]
        print(f"    {sheet}: source {a['source_key']} "
              f"(E_source {a['e_source']:.2f}), {a['sed']} SED x5 at "
              f"{YOUNG_BSA:.0%} BSA")
        print(f"      protocol dose grid {a['doses']} SED "
              f"(modal complete row, {a['n_on_grid']}/{len(a['subjects'])} "
              f"subjects)")
        for f in sorted(g):
            print(f"      FST {ROMAN_OF[f]:<3} n={g[f]['n']:<3} "
                  f"baseline {g[f]['base']:5.1f}  raw slope "
                  f"{g[f]['slope']:.3f} nmol/L per SED   {g[f]['eth']}")
        ra, rb, adj = YOUNG_PUBLISHED_ADJUSTED[sheet]
        print(f"      raw       FST {ra}:{rb} slope ratio = "
              f"{g[FST_ROMAN[ra]]['slope']/g[FST_ROMAN[rb]]['slope']:.3f}")
        print(f"      published FST {ra}:{rb} slope ratio = {adj:.2f}   "
              f"(Table 3, ANCOVA-adjusted)")
    print("\n  Ethnicity keys: WE = white European, SEEA = South East/East")
    print("  Asian, ISC = Indian subcontinent, Af = African.")

    # ---------------------------------------------------------------- S1 ---
    _fb, _agree = s1_feedback_check(arms)

    # ---------------------------------------------------------------- S2 ---
    hr("S2  WHAT EACH CONTRAST INDEPENDENTLY IMPLIES FOR K_const")
    print("\n  Each row inverts ONE contrast on its own, under its own study's")
    print("  protocol, source and baselines. No fitting across rows yet.")
    print(f"\n  {'contrast':52s} {'observed':>9} {'K_const':>9} {'V:II':>7}")
    per_contrast = []
    for tr in YOUNG_TREATMENTS:
        for study, label, obs, fn in build_targets(arms, groups, tr):
            if study == "Felton2016" and tr != YOUNG_TREATMENTS[0]:
                continue        # Felton has no treatment variants
            K, flag = invert_one(obs, fn)
            per_contrast.append((study, tr, label, obs, K, flag))
            print(f"  {label:52s} {obs:>9.3f} {K:>9.3f} {optical(K,5):>7.2f}"
                  + (f"  [{flag}]" if flag else ""))

    # ---------------------------------------------------------------- S3 ---
    hr("S3  THE LAB-ONLY JOINT FIT, AND WHY IT IS NOT THE ANSWER")
    fits = {}
    for tr in YOUNG_TREATMENTS:
        tg = build_targets(arms, groups, tr)
        K, J, grid, vals = minimise(tg)
        fits[tr] = (K, J, tg)
        print(f"\n  Young treatment '{tr}'  ->  K_const = {K:.3f}  "
              f"(FST V:II = {optical(K,5):.2f}x, objective {J:.5f})")
        for study, label, obs, fn in tg:
            mod = fn(K)
            print(f"    {label:52s} obs {obs:>6.3f}  model {mod:>6.3f}  "
                  f"{100*(mod/obs-1):>+6.1f}%")
    print("""
  FINDING, and it is the reason this script does not stop here. Under EVERY
  Young pairing the joint lab fit misses BOTH studies badly and in OPPOSITE
  directions -- roughly -40% on Felton and +40% on Young. The two matched-dose
  lab trials are not noisy estimates of one number; within this model
  structure they are incompatible, and their least-squares compromise
  describes neither. Reporting that compromise as "the calibrated value"
  would hide a structural disagreement behind an average.

  The disagreement is also visible without any model: Young reports FST II:VI
  = 1.29-1.43 while Felton reports FST II:V = 2.14 in the same kind of
  cabinet, and under ANY monotone pigment penalty II:VI must EXCEED II:V.""")

    # ---------------------------------------------------------------- S3b --
    hr("S3b  THE ESTIMATOR ACTUALLY USED: four independent within-study routes")
    print("""
  What the approved diagnosis established is that FOUR independent
  within-study routes estimate the same physical quantity -- the FST V:II
  optical factor -- by two different methods:

      LAB, matched physical dose, behaviour absent by construction
        Felton 2016   phototype II vs V, 35% BSA, 23.4 SED, same cabinet
        Young 2020    FST II vs VI, 85% BSA, both source arms
      FIELD, personal dosimetry, behaviour MEASURED and then removed
        Kift 2013 / Webb 2010   Manchester 53.5N
        Darling 2013            Surrey 51.3N

  K_const is fitted to all four in LOG RATIO space. Because
  ln(V:II factor) = K_const * (M_V - M_II) is linear in K_const, the
  least-squares solution under any set of study weights is exactly the
  WEIGHTED MEAN of the four study-level implied K_const values -- no
  iteration, no hidden weighting.

  [CHOICE, SUPERSEDED 2026-09-22] Equal weight per STUDY was the incumbent
  scheme. Its stated objection to method-weighting still holds verbatim ("it
  would require a precision for the field inversions that their own
  assumption register says is not quantified") and is exactly why the field
  tier below stays equal-weighted. What changed is that the LAB tier's two
  routes DO have measured precisions -- Felton 2016 publishes the gain with
  its SD, and Young 2020's individual-level slopes are in this repository --
  so within that tier the precision is a MEASUREMENT, not an author's choice
  of sharpness. S3c audits every route's variance before any of it is used.

  HELD OUT of this objective, and used only to score it in S6: the Farrar 2013
  / Rhodes 2010 cross-trial pair, Shih 2018's fractional-MED null, Kallioglu
  2024, NHANES Part B, and every check in vitd_validation.py.""")

    # Young's study-level value: primary pairing, both arms.
    primary = YOUNG_TREATMENTS[0]
    young_rows = [(lab, K) for (study, tr, lab, _o, K, _f) in per_contrast
                  if study == "Young2020" and tr == primary]
    young_ks = [K for _lab, K in young_rows]
    K_young = float(np.mean(young_ks))
    K_felton = [K for (study, tr, _l, _o, K, _f) in per_contrast
                if study == "Felton2016"][0]
    _fr = field_route_values(FIELD_ROUTES_MODE)
    sources = [
        ("Felton 2016 (lab, FST V, within-study)", K_felton),
        (f"Young 2020 (lab, FST VI, both arms, {primary})", K_young),
        ("Kift 2013 / Webb 2010 (field, behaviour removed)",
         _fr["Kift2013"]["K"]),
        ("Darling 2013 (field, behaviour removed)",
         _fr["Darling2013"]["K"]),
    ]
    print(f"\n  {'source':52s} {'K_const':>9} {'V:II':>7}")
    for lab, K in sources:
        print(f"  {lab:52s} {K:>9.3f} {optical(K,5):>7.2f}")
    K_raw = float(np.mean([K for _, K in sources]))
    sd = float(np.std([K for _, K in sources], ddof=1))
    print(f"\n  {'mean (one vote per study, the incumbent)':52s} {K_raw:>9.3f} "
          f"{optical(K_raw,5):>7.2f}")
    print(f"  {'SD across the four routes':52s} {sd:>9.3f}")
    print(f"  {'range':52s} "
          f"{min(K for _, K in sources):.3f}-{max(K for _, K in sources):.3f}")

    # --------------------------------------------------------------- S3c ---
    per_route = s3c_variance_audit(arms, groups, per_contrast, sources,
                                   field_routes=FIELD_ROUTES_MODE)
    schemes = {}
    for name in WEIGHTING_SCHEMES:
        routes = build_routes(per_route, scheme=name)
        schemes[name] = pool_routes(routes, scheme=name)

    hr("S3d  THE SAME FOUR ROUTES UNDER EVERY WEIGHTING SCHEME")
    print(f"\n  {'scheme':34s} {'K_const':>9} {'V:II':>7} {'pooled SE':>10}  "
          f"effective weights")
    for name in WEIGHTING_SCHEMES:
        r = schemes[name]
        wtxt = ", ".join(f"{s.replace('2016','').replace('2013','').replace('2020','')}"
                         f" {100*r['weights'][s]:.1f}%"
                         for s in ("Felton2016", "Young2020",
                                   "Kift2013", "Darling2013")
                         if s in r["weights"])
        tag = "  <- DEFAULT" if name == DEFAULT_WEIGHTING_SCHEME else ""
        se_txt = ("%.3f" % r["se"]) if r["se"] else "--"
        print(f"  {name:34s} {r['K']:>9.3f} {optical(r['K'],5):>7.2f} "
              f"{se_txt:>10}  {wtxt}{tag}")
        if r["dropped"]:
            print(f"  {'':34s} {'':>9} {'':>7} {'':>10}  "
                  f"DROPPED (no variance): {', '.join(r['dropped'])}")

    K_default = schemes[DEFAULT_WEIGHTING_SCHEME]["K"]
    # [PRECISION 2026-09-23] Full precision, no 2 dp rounding: K_const is a
    # calibration constant the model reads back (CLAUDE.md). See S9.
    K_new = K_default
    print(f"""
  [derived] ESTIMATOR OUTPUT under '{DEFAULT_WEIGHTING_SCHEME}' =
  {K_default!r} (shipped at full precision; FST V:II optical = {optical(K_new,5):.6f}x)
  [derived] under the retained incumbent 'one_vote_per_study' =
  {K_raw:.6f} -> {round(K_raw,2):.2f}   (FST V:II optical = {optical(round(K_raw,2),5):.2f}x)
  [derived] movement = {K_default - K_raw:+.6f} in K_const,
  {optical(K_default,5) - optical(K_raw,5):+.6f} in the FST V:II optical factor.
  [derived] pre-2026-08-17 value = {OLD_K_CONST:.2f}
  (FST V:II optical = {optical(OLD_K_CONST,5):.2f}x)""")

    # -------------------------------------------------------------- S3e ---
    hr("S3e  RANDOM EFFECTS: tau^2, I^2, the pooled interval, and the Kift exclusion")
    print(RE_PREAMBLE)
    re_routes = build_routes(per_route, scheme=DEFAULT_WEIGHTING_SCHEME)
    print(f"\n  {'tau^2 est':10s} {'tau^2':>12} {'tau':>10} {'I^2':>8} "
          f"{'Q':>10} {'pooled K':>10} {'SE':>9} {'95% CI':>24} {'V:II':>8}")
    re_results = {}
    for meth in ("DL", "PM", "REML"):
        r = pool_random_effects(re_routes, meth)
        re_results[meth] = r
        ci = f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
        print(f"  {meth:10s} {r['tau2']:>12.6f} {r['tau']:>10.6f} "
              f"{100*r['I2']:>7.2f}% {r['Q']:>10.4f} {r['K']:>10.6f} "
              f"{r['se']:>9.6f} {ci:>24} {optical(r['K'],5):>8.4f}")
    ks = [re_results[m]["K"] for m in ("DL", "PM", "REML")]
    rd = re_results["DL"]
    print(f"\n  [derived] routes in the pool (k) = {rd['k']}; dropped for "
          f"having no variance: {', '.join(rd['dropped']) or 'none'}")
    print("  [derived] per-route random-effects weights (DL): "
          + ", ".join(f"{k} {100*v:.2f}%" for k, v in rd['weights'].items()))
    print(f"  [derived] tau^2-ESTIMATOR RANGE on pooled K_const: "
          f"{min(ks):.6f} to {max(ks):.6f}  (width {max(ks)-min(ks):.6f})")
    print(f"  [derived] HKSJ interval at k={rd['k']} (t multiplier "
          f"{rd['t_mult']:.6f}): [{rd['ci_hksj'][0]:.4f}, {rd['ci_hksj'][1]:.4f}]")
    print(HKSJ_NOTE)

    base_k, worst, rows = max_admissible_weight_probe(
        lambda rs: pool_random_effects(rs, "DL"), re_routes, "Kift2013")
    print("\n  MAX-ADMISSIBLE-WEIGHT PROBE on the excluded Kift route.")
    print(f"    Kift's own implied K_const is "
          f"{per_route['Kift2013']['K']:.6f}. If it HAD a variance, how far")
    print("    could it move the pool? Sweeping an assumed SE from 8x down "
          "to 1x the most")
    print("    precise route's SE -- an upper bound on any weight it could "
          "earn:")
    print(f"      {'assumed SE(K)':>14} {'pooled K':>12} {'movement':>12}")
    for se_a, k_a, dlt in rows:
        print(f"      {se_a:>14.6f} {k_a:>12.6f} {dlt:>+12.6f}")
    print(f"    [derived] largest movement across the whole sweep: "
          f"{worst:.6f} in K_const")
    print(f"    [derived] pooled K without Kift (the shipped pool) = "
          f"{base_k:.6f}")
    print("    NOTHING IS IMPUTED. The sweep BOUNDS what the exclusion "
          "hides; it is not")
    print("    an estimate of Kift's precision and its output is never "
          "pooled.")

    v_default = optical(K_default, 5)
    in_band = DIAGNOSED_BAND[0] <= v_default <= DIAGNOSED_BAND[1]
    print(f"\n  APPROVED-BAND CHECK (a1_diagnosis_result_2026_08_17, FST V:II "
          f"optical {DIAGNOSED_BAND[0]:.1f}-{DIAGNOSED_BAND[1]:.1f}x):")
    print(f"    pooled FST V:II = {v_default:.6f}x -> "
          f"{'INSIDE' if in_band else 'OUTSIDE'} the band.")
    if not in_band:
        print("    REPORTED, NOT SILENTLY RE-OPENED. Moving the band is a "
              "separate approval;")
        print("    this script states the disagreement and leaves it to the "
              "user.")

    # ---------------------------------------------------------------- S4 ---
    hr("S4  SENSITIVITY: leave-one-out and method split, under BOTH schemes")
    print("""
  The leave-one-out / method-split envelope is the honest uncertainty
  statement W-J recommended carrying in place of a scalar. It is reported
  here under the incumbent one-vote scheme AND under the random-effects
  default, because the two are materially different objects.

  DECISION A4 IS SETTLED HERE, ON THE NUMBERS. The user's instruction was to
  carry the envelope ONLY IF it is still needed once the weighting question
  is resolved. Random effects already produces an interval of its own -- the
  tau^2-estimator range in S3e -- so the envelope is needed only if it is
  MATERIALLY WIDER than that. The comparison is printed at the end of this
  section, together with whether the headline FST V/VI feasibility
  conclusion flips across the envelope.""")
    keys = ["Felton2016", "Young2020", "Kift2013", "Darling2013"]
    subsets = [("all four routes (primary)", keys)]
    for i, k in enumerate(keys):
        subsets.append((f"drop {sources[i][0][:48]}",
                        [x for x in keys if x != k]))
    subsets.append(("LAB routes only (Felton + Young)", keys[:2]))
    subsets.append(("FIELD routes only (Kift + Darling)", keys[2:]))

    envelopes = {}
    for name in ("one_vote_per_study", DEFAULT_WEIGHTING_SCHEME):
        print(f"\n  --- scheme: {name} ---")
        print(f"  {'variant':54s} {'K_const':>8} {'V:II':>7}")
        vals = []
        for lab, subset in subsets:
            routes = [r for r in build_routes(per_route, scheme=name)
                      if r["study"] in subset]
            got = pool_routes(routes, scheme=name)
            if got["K"] is None:
                print(f"  {lab:54s} {'n/a':>8} {'':>7}")
                continue
            vals.append(got["K"])
            print(f"  {lab:54s} {got['K']:>8.3f} {optical(got['K'],5):>7.2f}")
        # Young-pairing variants, for the incumbent-comparable set
        for tr in YOUNG_TREATMENTS[1:]:
            yk = float(np.mean([K for (study, t, _l, _o, K, _f) in per_contrast
                                if study == "Young2020" and t == tr]))
            alt = [dict(r, K=(yk if r["study"] == "Young2020" else r["K"]))
                   for r in build_routes(per_route, scheme=name)]
            got = pool_routes(alt, scheme=name)
            vals.append(got["K"])
            vlab = "Young pairing '%s' instead" % tr
            print(f"  {vlab:54s} {got['K']:>8.3f} "
                  f"{optical(got['K'],5):>7.2f}")
        envelopes[name] = (min(vals), max(vals))
        lo, hi = envelopes[name]
        print(f"\n  [derived] ENVELOPE under {name}: K_const {lo:.6f} to "
              f"{hi:.6f}")
        print(f"  [derived] width {hi-lo:.6f} in K_const; FST V:II "
              f"{optical(lo,5):.4f}x to {optical(hi,5):.4f}x")
        inband = [v for v in vals
                  if DIAGNOSED_BAND[0] <= optical(v, 5) <= DIAGNOSED_BAND[1]]
        print(f"  [derived] {len(inband)} of {len(vals)} variants land inside "
              f"the approved {DIAGNOSED_BAND[0]:.1f}-{DIAGNOSED_BAND[1]:.1f}x "
              f"band.")

    lo0, hi0 = envelopes["one_vote_per_study"]
    lo1, hi1 = envelopes[DEFAULT_WEIGHTING_SCHEME]
    tau_lo, tau_hi = min(ks), max(ks)
    print(f"""
  [derived] envelope width under one_vote_per_study  {hi0-lo0:.6f}
  [derived] envelope width under {DEFAULT_WEIGHTING_SCHEME:22s} {hi1-lo1:.6f}
  [derived] ratio {(hi1-lo1)/(hi0-lo0):.6f}x  (above 1.0 = the random-effects
  envelope is the WIDER of the two)

  [derived] A4 TEST -- leave-one-out envelope vs the tau^2-estimator range:
    LOO envelope under the default   {lo1:.6f} to {hi1:.6f}  (width {hi1-lo1:.6f})
    tau^2-estimator range (S3e)      {tau_lo:.6f} to {tau_hi:.6f}  (width {tau_hi-tau_lo:.6f})
    ratio of widths                  {(hi1-lo1)/(tau_hi-tau_lo) if tau_hi > tau_lo else float("inf"):.4f}x""")
    if tau_hi > tau_lo and (hi1 - lo1) <= 2.0 * (tau_hi - tau_lo):
        print("""
    VERDICT: the leave-one-out envelope is NOT materially wider than the
    tau^2-estimator range the random-effects pool already carries, so the
    envelope adds no uncertainty the reported interval does not already
    express. A4's separate envelope is NOT carried.""")
    else:
        print("""
    VERDICT: the leave-one-out envelope IS materially wider than the
    tau^2-estimator range, i.e. WHICH routes are in the pool moves the answer
    more than HOW tau^2 is estimated does. A4's envelope must still be
    carried alongside the random-effects interval.""")
    print(f"""
  [derived] FST V:II optical factor across the LOO envelope:
    {optical(lo1,5):.4f}x to {optical(hi1,5):.4f}x
    against the approved band {DIAGNOSED_BAND[0]:.1f}-{DIAGNOSED_BAND[1]:.1f}x""")

    # ---------------------------------------------------------------- S5 ---
    hr("S5  SIGN CHECK -- direction on EVERY matched-dose contrast available")
    print("\n  A K_const that matches one ratio while still predicting the")
    print("  DARKER group to gain MORE is not a fix. Requirement: model ratio")
    print("  > 1 wherever the observed ratio is > 1.")
    print(f"\n  {'contrast':52s} {'obs':>6} {'K=%.3f' % OLD_K_CONST:>8} "
          f"{'K=%.3f' % K_new:>8} {'before':>9} {'after':>9}")
    sign_rows = [("Felton 2016 FST II:V gain", obs_f,
                  lambda K: felton_gain(2, K) / felton_gain(5, K))]
    for sheet, g in groups.items():
        arm = arms[sheet]
        pool = float(np.mean([s["base"] for s in arm["subjects"]]))
        ra, rb, adj = YOUNG_PUBLISHED_ADJUSTED[sheet]
        for dark in [f for f in sorted(g) if f >= 5]:
            # raw pairing, this arm's own baselines
            obs_raw = g[2]["slope"] / g[dark]["slope"]
            sign_rows.append((
                f"Young {sheet.split()[0]:8s} II:{ROMAN_OF[dark]:<2} "
                f"[raw@own]", obs_raw,
                (lambda K, dk=dark, ar=arm, gg=g: (
                    young_slope(2, gg[2]["base"], K, ar["sed"], ar["e_source"],
                                ar["doses"])
                    / young_slope(dk, gg[dk]["base"], K, ar["sed"],
                                  ar["e_source"], ar["doses"])))))
            if ROMAN_OF[dark] == rb:
                sign_rows.append((
                    f"Young {sheet.split()[0]:8s} II:{ROMAN_OF[dark]:<2} "
                    f"[adjusted@pooled]", adj,
                    (lambda K, dk=dark, ar=arm, pl=pool: (
                        young_slope(2, pl, K, ar["sed"], ar["e_source"],
                                    ar["doses"])
                        / young_slope(dk, pl, K, ar["sed"], ar["e_source"],
                                      ar["doses"])))))
    n_bad_old = n_bad_new = 0
    for label, obs, fn in sign_rows:
        old, new = fn(OLD_K_CONST), fn(K_new)
        bad_old = obs > 1.0 and old <= 1.0
        bad_new = obs > 1.0 and new <= 1.0
        n_bad_old += bad_old
        n_bad_new += bad_new
        print(f"  {label:52s} {obs:>6.2f} {old:>8.2f} {new:>8.2f} "
              f"{('INVERTED' if bad_old else 'ok'):>9} "
              f"{('INVERTED' if bad_new else 'ok'):>9}")
    print(f"\n  [derived] contrasts INVERTED at K_const = {OLD_K_CONST:.3f}: "
          f"{n_bad_old} of {len(sign_rows)}")
    print(f"  [derived] contrasts INVERTED at K_const = {K_new:.3f}: "
          f"{n_bad_new} of {len(sign_rows)}")

    print("\n  ABSOLUTE LEVELS on Felton 2016 (the fit matched RATIOS only --")
    print("  do the two arms' gains come out right in nmol/L?)")
    print(f"    observed: phototype II {FELTON_FINAL[2]-FELTON_D0[2]:+.1f}, "
          f"phototype V {FELTON_FINAL[5]-FELTON_D0[5]:+.1f} nmol/L")
    print(f"    {'K_const':>9} {'model II':>9} {'err':>7} {'model V':>9} "
          f"{'err':>7} {'ratio':>7}")
    for K in sorted({0.0, OLD_K_CONST, K_new, 1.473}):
        g2, g5 = felton_gain(2, K), felton_gain(5, K)
        e2 = 100 * (g2 / (FELTON_FINAL[2] - FELTON_D0[2]) - 1)
        e5 = 100 * (g5 / (FELTON_FINAL[5] - FELTON_D0[5]) - 1)
        tag = ""
        if abs(K - OLD_K_CONST) < 1e-9:
            tag = "  <- PRE-RECALIBRATION"
        elif abs(K - K_new) < 1e-9:
            tag = "  <- FITTED"
        elif abs(K - 1.473) < 1e-9:
            tag = "  <- Felton-only (fst_gap_decomposition)"
        print(f"    {K:>9.3f} {g2:>9.2f} {e2:>+6.1f}% {g5:>9.2f} {e5:>+6.1f}% "
              f"{g2/g5:>7.2f}{tag}")

    # ---------------------------------------------------------------- S6 ---
    hr("S6  HELD-OUT SCORING (none of this entered the objective)")
    v_new = optical(K_new, 5)
    print(f"\n  New K_const = {K_new:.2f}  ->  FST V:II optical = {v_new:.2f}x\n")
    print(f"  {'held-out estimate of FST V:II':52s} {'V:II':>10} {'verdict':>26}")
    for lab, v, note in HELD_OUT_SCORING:
        if isinstance(v, tuple):
            shown = f"{v[0]:.2f}-{v[1]:.2f}"
            ok = v[0] <= v_new <= v[1]
        else:
            shown = f"{v:.2f}"
            ok = abs(np.log(v_new / v)) < np.log(1.35)
        print(f"  {lab:52s} {shown:>10} "
              f"{('CONSISTENT' if ok else 'differs'):>12}  {note}")
    print()
    for lab, (v, k) in HELD_OUT_FIELD_BLIND.items():
        print(f"  {lab:52s} {v:>10.2f} "
              f"{'REJECTED':>12}  above every matched-dose measurement")
    print(f"\n  Approved target band (calibration_approval.yaml, "
          f"a1_diagnosis_result_2026_08_17): "
          f"{DIAGNOSED_BAND[0]:.1f}x to {DIAGNOSED_BAND[1]:.1f}x")
    print(f"  New value {v_new:.2f}x is "
          f"{'INSIDE' if DIAGNOSED_BAND[0] <= v_new <= DIAGNOSED_BAND[1] else 'OUTSIDE'}"
          f" that band.")

    # ---------------------------------------------------------------- S7 ---
    hr("S7  RESULTING FST RATIOS, BEFORE AND AFTER")
    print(f"\n  {'pair':8s} {'K=%.2f before' % OLD_K_CONST:>19} "
          f"{'K=%.2f after' % K_new:>19} {'change':>9}")
    for (lab, old), (_, new) in zip(ratio_table(OLD_K_CONST), ratio_table(K_new)):
        print(f"  {lab:8s} {old:>19.2f} {new:>19.2f} {new/old:>8.2f}x")
    print("""
  DARK-SKIN FLAG, at the point of use. The V:II row is anchored by FST V data
  that is entirely Indian-subcontinent ancestry at UK latitude (Felton 2016
  n = 6; Young 2020 Arimed B n = 5). Every VI:* row is an EXTRAPOLATION of one
  coefficient onto M_baseline(VI) = 1.00, a group-level prior with no
  melanin-index validation (see the M_baseline note in vitd_model.py), scored
  against FST VI data that is entirely lamp-based. Use the V:II row for
  clinical statements about South Asian users at temperate latitudes; treat
  every VI figure as an extrapolation until African-ancestry field dosimetry
  exists.""")

    # ---------------------------------------------------------------- S8 ---
    hr("S8  KNOCK-ON: what else in the model this moves")
    a_new = 2.353 * np.exp((K_new - OLD_K_CONST) * 0.21)
    print(f"""
  K_const multiplies M_baseline(FST) inside exp(-K_const * M_baseline), so it
  changes the ABSOLUTE synthesis level of EVERY skin type, not only the
  between-type ratios. FST III, the anchor A = 2.353 was calibrated against,
  moves by

    exp(-{K_new:.3f} * 0.21) / exp(-{OLD_K_CONST:.3f} * 0.21) = {np.exp(-(K_new - OLD_K_CONST) * 0.21):.4f}

  i.e. FST III synthesis per SED falls {100*(1-np.exp(-(K_new - OLD_K_CONST) * 0.21)):.1f}%. To hold that anchor exactly,
  A would have to move to {a_new:.4f}.

  THIS SCRIPT DOES NOT CHANGE A. A is a separate approval
  (calibration_approval.yaml, approved_2026_08_17_pending_implementation, id
  A4) and moving two coupled parameters in one step would make neither
  attributable afterwards. The consequence is recorded here, not silently
  absorbed. `vitd_validation.sanity_single_bolus_FST_III` hardcodes
  `exp(-0.33 * 0.21)` in its expected value and will fail until its expected
  value is made a function of params.K_const -- that file belongs to another
  owner and was deliberately not touched.""")

    # ---------------------------------------------------------------- S9 ---
    hr("S9  EVERY [CHOICE] AND [ESTIMATE] IN THIS DERIVATION")
    for i, s in enumerate(CHOICES, 1):
        print(f"\n  {i:2d}. {s}")


CHOICES = [
    "[CHOICE] Calibration set = within-study, matched-physical-dose, "
    "matched-BSA, same-source trials only (Felton 2016; Young 2020). Field "
    "cohorts, cross-trial pairs and fractional-MED designs are excluded from "
    "the objective and used only to score.",
    "[CHOICE] Objective = weighted sum of squared LOG errors on response "
    "RATIOS, not on absolute levels. Absolute levels depend on A, which is "
    "separately uncalibrated (Young 2021 puts fair-skin slopes at 2.0-5.0 "
    "nmol/L per SED against the model's ~1.85 at 85% BSA), so fitting K_const "
    "to absolute levels would launder A's error into the pigment term.",
    "[CHOICE] Weight one per STUDY, not per contrast or per subject. Young's "
    "two arms are one recruitment and one protocol design. S4 reports what "
    "every other weighting gives. [SUPERSEDED 2026-09-22 as the DEFAULT, "
    "RETAINED as a selectable scheme: see the next item.]",
    "[CHOICE 2026-09-22, W-W, user-approved in decision A4] "
    "The default weighting is now 'inverse_variance': routes are "
    "inverse-variance weighted WHERE A MEASURED VARIANCE EXISTS and equally "
    "weighted where none does, inside the lab/field method tiers the "
    "incumbent estimator already implied. The tier split is INHERITED (the "
    "incumbent already gave the lab pair and the field pair 50% each); the "
    "only thing that moves is the split WITHIN the lab tier, which is the "
    "one place a variance is published. Felton 2016 prints the gain with its "
    "SD and Young 2020's individual-level slopes are in this repository, so "
    "both lab precisions are MEASURED inputs rather than an author's choice "
    "of sharpness -- which is what the superseded [CHOICE] above rejected, "
    "and rightly. 'one_vote_per_study' remains computable in the same run so "
    "every previously published figure stays reproducible.",
    "[LIMIT 2026-09-22] NO VARIANCE IS EVER IMPUTED. Kift 2013 and Darling "
    "2013 publish nothing usable on the quantity this script inverts, so "
    "they cannot be inverse-variance weighted and are not pretended to be. "
    "Kift 2013 DOES publish a 95% CI on its raw diet-adjusted serum ratio "
    "(SE(ln) = 0.091679) -- verified from the PDF 2026-09-22, correcting an "
    "earlier repo claim that it publishes nothing -- but that is the RAW "
    "contrast, while the route inverts the BEHAVIOUR-REMOVED contrast, and "
    "the behaviour-removal step's uncertainty is unquantified anywhere in "
    "this repo. Using the serum CI as the route's variance would understate "
    "the route's uncertainty and over-weight it. [SUPERSEDED 2026-09-22 for "
    "Darling: the accepted manuscript is held and its Table 3 gives the "
    "route a delta-method SE -- see darling_se_K. Kift remains variance-less.]",
    "[CHOICE] The headline estimator combines FOUR within-study routes (two "
    "lab, two field-with-behaviour-removed) rather than the lab pair alone, "
    "because S3 shows the lab pair is internally incompatible and its "
    "least-squares compromise describes neither trial.",
    "[CHOICE] Young's study-level value uses the 'adjusted@pooled' pairing, "
    "the only one of the three whose answer does not depend on the model's "
    "feedback term being the right strength -- which S1 measures and finds it "
    "is not (0.4-0.8x of the observed baseline gradient).",
    "[CHOICE, REVERSED 2026-09-23] The result was rounded to 2 decimal places. "
    "It is now shipped at FULL PRECISION: K_const is a calibration constant "
    "other calculations read back, and rounding it also broke the joint "
    "(K_const, A) fixed point. The uncertainty is carried as the tau^2 range "
    "and the leave-one-out envelope, not by discarding digits.",
    "[CHOICE, BUG-FIXED 2026-09-22] Search bounded at K_const * M_baseline(V) "
    "<= K_TIMES_M_HI (was a bare K_const <= 3.0, which silently dropped Felton "
    "once the M scale moved). Above it the darker arm's modelled net response "
    "goes negative and the ratio stops being monotone.",
    "[CONFIRMED 2026-09-22, was ESTIMATE] Felton 2016's cabinet is NOT "
    "assumed. The paper names it: a Philips HB588 Sunstudio cabinet fitted "
    "with alternating Arimed B and Cleo Natural tubes (p.1321-1322). The old "
    "tag described the paper as silent on this, which it is not. Sensitivity "
    "is exactly zero regardless -- E_source cancels from a between-skin-type "
    "ratio and this estimator fits ratios only -- so resolving the tag "
    "changes no number, only whether the file tells the truth about its "
    "own inputs.",
    "[ESTIMATE] Young 2020's exposure interval is taken as 3.5 d, the "
    "midpoint of its stated '3-4 days'.",
    "[ESTIMATE] Young's blood-draw times are reconstructed from the printed "
    "cumulative-dose grid on the assumption that each intermediate draw "
    "immediately preceded the next exposure and the last followed exposure 5. "
    "The recorded 17-18 day study duration is consistent with that.",
    "[ESTIMATE/PROVENANCE] The held-out field numbers (Kift, Darling) are "
    "reproduced from fst_gap_decomposition.py, which rests on Kift 2013 "
    "decimal reconstructions and on an exposed-BSA figure ASSUMED equal "
    "between Darling's two cohorts. They score the fit; they do not set it.",
    "[LIMIT, REVISED 2026-09-22] Felton 2016 reports group means and SDs, "
    "not individual trajectories, and its FST V gain SD (10.5) exceeds its "
    "mean gain (8.3). This item previously said no confidence interval on "
    "the Felton-implied K_const was computable; that was WRONG, and S3c now "
    "computes one -- the published SDs are on the GAIN, which is exactly the "
    "quantity the ratio is built from, so a delta-method interval needs "
    "nothing further. What IS still true, and is the real limit: with n = 6 "
    "and SD > mean in the darker arm, a log-ratio normal approximation is "
    "optimistic in the tails, and the upper end of Felton's interval is "
    "unreachable at any K_const this model structure admits (the ratio stops "
    "being monotone above K_HI). Felton's true interval is therefore "
    "ASYMMETRIC and WIDER than the symmetric SE used here, so the SE in S3c "
    "UNDER-states Felton's imprecision and the weighting change is "
    "conservative in the direction that matters.",
    "[LIMIT] Young 2020's Arimed B FST V cell is n = 5 and its FST VI cell "
    "n = 6 (3 African, 3 Indian-subcontinent). The TL01 arm has n = 11 FST VI "
    "(all African) and NO FST V at all.",
]


if __name__ == "__main__":
    main()
