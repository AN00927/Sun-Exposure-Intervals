"""
mckenzie2012_check.py
======================
CHECK, not calibration. Exploits McKenzie et al. 2012 (Photochem. Photobiol.
Sci. 11:1174-1185, "Serum 25-hydroxyvitamin-D responses to multiple UV
exposures from solaria: inferences for exposure to sunlight") -- the paper
flagged in vitd_model.py's Gap-3 note as "not yet used quantitatively".

SOURCE LOCATION. Full text found by content search (not filename) at:
    Research_Papers/Serum 25-hydroxyvitamin-D responses to multiple UV
    exposures from.pdf
Filename gives no hint this is the McKenzie paper -- confirmed only by
opening it and matching "Photochem. Photobiol. Sci., 2012, 11, 1174" and
author "Richard McKenzie" on page 1. This is exactly the CLAUDE.md warning
about unreliable filenames in Research_Papers/, once more.

LEGIBILITY. The PDF is a native (non-scanned) RSC typeset article; every
number used below was extracted from selectable text, not an image. Figures
2, 3 and 5 (individual-participant scatter plots of 25(OH)D vs. UV dose) were
NOT digitized -- there is no tabulated individual-level data anywhere in the
paper, only the group-mean statistics in Tables 1, 3, 4, 5 and the two
sentences of prose describing Fig. 4's fit. Any check that needs
individual-level dose-response (e.g. a real per-subject fit of `m`) is not
possible from this PDF; only group-mean checks are. This is flagged per
CLAUDE.md's instruction to say so rather than silently write the source off
-- the underlying scatter data in Fig. 2/3/5 could in principle be
re-digitized from the images if a per-subject fit is ever wanted.

===============================================================================
EXTRACTED NUMBERS (every one with page / table / figure)
===============================================================================

STUDY DESIGN (p.1174 abstract; p.1175 Methods; p.1176 Table 1)
  Two-arm RCT, Auckland NZ (36.8 S), winters 2006-2007. Twice-weekly
  full-body UV exposure for 12 weeks (24 sessions), blood drawn at week 0,
  6, 12. Booth A = UV-A dominant (Daavlin Spectra 305/350, "F72 T12 BL-HO"
  tubes). Booth B = UV-B dominant, narrowband (Daavlin Spectra 726-SP-2X,
  "Philips Narrowband TL 100W/01 UV-B Medical" tubes -- i.e. TL-01 lamps,
  the same lamp family as this repo's `tl01_nb_uvb` SOURCE_CALIBRATION_TABLE
  entry).

  n = 119 analysed (123 recruited, 4 excluded). Table 1 (p.1176):
    Fitzpatrick 1-3 (white): 45  |  4-5 (brown): 41  |  6 (black): 33
    Booth A (n=61): 24 / 18 / 19      Booth B (n=58): 21 / 23 / 14
    Baseline 25(OH)D, nmol/L: full set 44.6 +/- 9.5; booth A 45.4 +/- 9.2;
      booth B 43.7 +/- 9.7
    Age 58.0 +/- 16.2 y; BMI 33.6 +/- 7.6

  ###########################################################################
  # CRITICAL INTERPRETIVE CORRECTION (added 2026-08-21 after re-reading
  # p.1177 directly). DO NOT USE McKENZIE'S NULL SKIN-TYPE RESULT AS
  # EVIDENCE THAT K_const SHOULD BE ZERO. It is not that kind of evidence.
  #
  # THIS IS A FRACTIONAL-MED DESIGN, NOT A MATCHED-DOSE DESIGN. From
  # p.1177, "UV exposures", verbatim:
  #
  #   "In the first year of the study, the skin MED doses (measured in SED)
  #    were estimated from skin type, but in the second year these were
  #    established for each individual at the outset of their participation
  #    period. The skin MED dose varied greatly from person-to-person,
  #    ranging from less than 2 SED to greater than 10 SED. The initial
  #    doses for the first session from booth B started at approximately
  #    0.6 MED per session and typically increased in steps of 0.05 MED up
  #    to a maximum close to 1 MED per session. The cumulative UV doses
  #    over the 12 week period were THUS VARIED ACCORDING TO SKIN TYPE."
  #                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  # Darker participants were GIVEN MORE SED, in proportion to their own
  # measured MED. Finding no residual skin-type effect on delta-D is
  # therefore the EXPECTED result, not a surprising one -- the dose was
  # already compensated for pigment before the response was measured.
  #
  # This repository already documents exactly this mechanism, from a
  # different source: see the "Dose framing modes" block in vitd_model.py's
  # module docstring, citing P24 (Shih/Farrar/Rhodes 2018) -- "fractional-MED
  # dosing removes the apparent skin-type effect for acute 25(OH)D gain".
  # McKenzie 2012 is an independent, much larger CORROBORATION of that
  # framing result. It is NOT a fifth within-study route for K_const, and it
  # must not be added to the four in scripts/kconst_recalibration.py, all of
  # which are matched-dose or behaviour-corrected.
  #
  # BONUS FINDING, same page, and this one IS direct evidence: despite booth
  # B delivering up to 25 SED per session, "there were no significant changes
  # in skin reflectance over the course of the study" (measured 3x by EEL
  # spectrophotometer: inner arm 66 +/- 2%, forehead 54 +/- 1%). A large
  # cohort, 62% dark-skinned, 24 sessions, and NO measurable tanning. That
  # is independent support for the 2026-08-18 retirement of K_ATTEN (see the
  # K_ATTEN entry in vitd_model.py). The authors flag it as "surprising" and
  # offer that the monitored sites may have been shielded -- so treat it as
  # supportive, not decisive.
  ###########################################################################

  IMPORTANT CAVEAT (p.1183, Appendix A1; p.1181 Discussion): skin type
  (skin reflectance, X1) was included as a covariate in BOTH regression
  models and was found NON-SIGNIFICANT / weakly correlated in every
  breakdown tried: "There does not seem to be a strong correlation with age
  or skin type (even for the individual data sets)" (p.1183). No delta-D
  value, sensitivity, or fitted `m` is reported SEPARATELY by Fitzpatrick
  group anywhere in the paper -- every number below is a POPULATION-POOLED
  statistic across all three skin bands, not a per-skin-type fit. The large
  dark-skin n (41+33=74/119, 62%) means dark skin dominates the pooled
  numbers, but it does not give three independent m's.

DOSE PROTOCOL, Table 3 (p.1177). Per-session SED, [min, mean, max] across
participants (mean is the "least sensitive"-to-"most sensitive" skin range
midpoint the paper reports, not a distributional mean/SD):
    Booth A: session 1 (wk1) = [0.10, 0.20, 0.25]
             session 12 (wk6) = [0.25, 1.00, 1.50]
             session 24 (wk12) = [0.25, 0.64, 1.25]
    Booth B: session 1 (wk1) = [1.5, 4.1, 7.6]
             session 12 (wk6) = [1.9, 12.9, 25.2]
             session 24 (wk12) = [2.9, 14.1, 25.2]
  Cumulative SED between blood draws, Table 3 mean column (matches Table 5):
    Booth A: period1 (wk1-6) = 7.7, period2 (wk7-12) = 9.3, total = 17.1
    Booth B: period1 (wk1-6) = 101.3, period2 (wk7-12) = 166.3, total = 268

SPECTRAL CHARACTERISATION, Table 2 (p.1176):
    RDE = UVVitD/UVEry (CIE-174 action-spectrum weighted / CIE erythema
    weighted), i.e. EXACTLY this repo's E_source concept, measured from the
    booths' own spectrographs (Ocean Optics USB4000, +/-15% booth
    uncertainty):
        Booth A: RDE = 1.16      Booth B: RDE = 2.27
        Winter sun (45S): RDE = 1.22   Summer sun (45S): RDE = 2.01
  NOTE (already in vitd_model.py SOURCE_CALIBRATION_TABLE comments, restated
  here for completeness): the two solar RDE values are numerically identical
  to the pre-existing solar_winter_nz/solar_summer_nz entries and are almost
  certainly the same Webb-2021 measurement chain, not an independent
  replicate. Booth B's RDE = 2.27 DOES sit inside (near the floor of) the
  existing tl01_nb_uvb range (2.24-2.63, midpoint 2.44) and was already
  logged there as corroboration, not as the source of that number.

RESULTS, Table 4 (p.1178, raw) and Table 5 (p.1180, seasonally adjusted --
used throughout below):
    Delta 25(OH)D, nmol/L, mean +/- StdErr:
        Booth A: wk0-6 = 18.7 +/- 1.2   wk6-12 = 13.0 +/- 1.3   wk0-12 = 31.7 +/- 1.5
        Booth B: wk0-6 = 40.7 +/- 1.9   wk6-12 = 22.1 +/- 1.9   wk0-12 = 62.8 +/- 2.3
    Sensitivity, nmol/L per SED:
        Booth A: 2.9 +/- 0.3 (wk0-6), 1.8 +/- 0.2 (wk6-12), 2.2 +/- 0.2 (wk0-12)
        Booth B: 0.6 +/- 0.1 (wk0-6), 0.3 +/- 0.04 (wk6-12), 0.4 +/- 0.1 (wk0-12)
    Sensitivity, nmol/L per SDD:
        Booth A: 2.5 +/- 0.2 (wk0-6), 1.5 +/- 0.2 (wk6-12), 1.9 +/- 0.1 (wk0-12)
        Booth B: 0.3 +/- 0.03 (wk0-6), 0.1 +/- 0.02 (wk6-12), 0.2 +/- 0.02 (wk0-12)

FIG. 4 GROUP-MEAN LOG FIT (p.1179 text + caption): a log fit to the four
group means (A06, A12, B06, B12) plus one un-tabulated near-zero anchor
point ("a fifth point representing the small change in 25(OH)D expected for
the small minimum dose shown" -- no number given for this point, so it is
NOT reproduced below; only the 4 tabulated points are used):
    "the mean changes in 25(OH)D is proportional to the logarithm of the UV
    dose (e.g., 10 ln(SED), or 8.6 ln(SDD))"
    Restricted-to-4-points intercepts: SED fit intercept = -3 (implausible,
    since dose=0 should give delta=0); SDD fit intercept = +1.1 (judged
    "more plausible" by the authors, p.1179).

APPENDIX A2 ALTERNATE REGRESSION MODEL (p.1183), the model referenced by the
vitd_model.py Gap-3 note:
    Delta_VitD = m * log(1 + UV_VitD) + F(X)
  where UV_VitD is in SDD and F(X) is a linear combination of 4 of the 6
  basis vectors from the linear model (skin reflectance, tanning history,
  age, BMI). THE FITTED NUMERIC VALUES OF m AND THE F(X) COEFFICIENTS ARE
  NOT PRINTED ANYWHERE IN THE PAPER -- no table gives them. What IS given is
  a set of DESCRIBED CONSEQUENCES of the fit (p.1179):
    - "a small offset in the model results, with an implied increase of
      5 nmol/L for zero UV exposure" (this is F(X) at mean covariates, not m)
    - "modelled increase in 25(OH)D from 0 to 1 SDD is approximately
      5 nmol/L" (read here as the log term's own contribution m*ln(2))
    - "over the first 10 SDD, the mean increase is reduced to 2.2 nmol/L
      per SDD" (average, not instantaneous, slope)
    - "asymptotes to an increase of approximately 60 nmol/L after cumulative
      UV exposures of 500 to 800 SDD"
    - model R^2 = 0.59 (combined-booth fit, p.1179); the SEPARATE linear
      model (not the log model) gets R^2 = 0.45/0.64/0.25 (period 1) and
      0.45/0.63/0.28 (period 2) for booth A / booth B / combined
      (Table A.2, p.1183)
  ABSTRACT gives one further number, presented as the headline result of
  this alternate model: "the deduced initial rate of increase of 25(OH)D
  was approximately 5 nmol/L per SED" (p.1174) -- note this says per-SED,
  the appendix text says per-SDD for the same "~5" figure; the paper does
  not reconcile the two unit labels for this one number. Treated below as
  an order-of-magnitude anchor only, not a precise m.
  CONCLUSION: no single numeric `m` can be read off this paper. The three
  or four ~5-nmol/L statements above are mutually loose paraphrases of one
  underlying fit whose actual coefficients were never tabulated.
  [ESTIMATE] used in analysis (b) below in place of a nonexistent tabulated
  m: the Fig. 4 group-mean slope (10 nmol/L per ln(SED) / 8.6 nmol/L per
  ln(SDD)) is the only NUMERICALLY STATED slope in the paper and is used as
  the m-equivalent, clearly flagged as a mean-only, non-individual-level fit.

SECOND-6-WEEK ATTENUATION (p.1181 Discussion, quoting the exact sentence):
    "Increases in 25(OH)D during the first 6 week period were greater than
    for the second 6 week period, particularly for booth B, where doses
    were much larger, with increase in 25(OH)D in the second period being
    less than half that in the first period."
  Checked against Table 5's precise numbers (both seasonally adjusted):
    Booth A: 13.0 / 18.7 = 0.695  (NOT less than half)
    Booth B: 22.1 / 40.7 = 0.543  (close to, but NOT strictly less than half)
  FLAG: the prose's "less than half" is not exactly reproduced by the
  precise Table 5 ratio for either booth on RAW delta-D; it IS closer to
  true for the SENSITIVITY ratios (nmol/L per dose unit), particularly
  booth B per-SDD: 0.1/0.3 = 0.33. This is a paper-internal looseness
  between prose and table, not an extraction error on this script's part
  -- both readings are reported here rather than silently picking the one
  that "sounds right".

===============================================================================
WHAT THIS SCRIPT DOES
===============================================================================
(a) log_form_check()     -- fits ln(dose), linear, sqrt, and power-law to the
                             4 Table-5 group-mean points (both SED and SDD
                             framings) and reports which best matches the
                             observed strong concavity. Explicitly a 4-point,
                             2-parameter-per-model test: weak, but it is what
                             the paper's own tabulated numbers support, and it
                             is what the paper itself did (Fig. 4).
(b) convert_m_and_implied_A() -- uses Table 2's OWN booth RDE (1.16 / 2.27,
                             not the generic SOURCE_CALIBRATION_TABLE entry)
                             to convert dose, then runs vitd_model.simulate()
                             with each booth's actual 24-session protocol
                             (reconstructed from Table 3's 3 anchor points,
                             rescaled to match Table 3/5's exact period sums)
                             and solves by bisection for the A that would
                             make the CURRENT model reproduce the observed
                             12-week delta-D, separately for representative
                             FST 2 (white band), FST 4 & 5 (brown band), and
                             FST 6 (black band). NOT a per-skin-type fit
                             FROM McKenzie (the paper does not offer one --
                             see caveat above); it is what the CURRENT
                             model's own melanin/BSA/feedback machinery
                             implies A would need to be, if this population's
                             pooled response is assigned to each FST in turn.
(c) second_half_attenuation_check() -- simulates the same reconstructed
                             protocols at the model's CURRENT A = 3.114 (no
                             fitting) and compares the model's own
                             second-6-week : first-6-week delta-D ratio to
                             the observed Table 5 ratio -- a genuine
                             out-of-sample check on the biphasic feedback
                             term, whose only prior anchor is Karppinen.

HARD RULE: this script does not change vitd_model.py, does not touch
calibration_approval.yaml, and does not assert that A should move. Any
number here that looks like "a better A" is reported as a MODEL-IMPLIED
figure under stated assumptions, for the user to weigh -- not adopted.

Run:  PYTHONIOENCODING=utf-8 python3 scripts/mckenzie2012_check.py
Writes: tables/mckenzie2012_check.csv
"""

from __future__ import annotations

import os
import sys
import textwrap

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vitd_model import ModelParams, simulate, get_E_source  # noqa: E402


# =============================================================================
# EXTRACTED DATA (see module docstring for page/table citations)
# =============================================================================

# Table 5 (p.1180), seasonally-adjusted cumulative dose & response.
TABLE5 = {
    "A": {
        "SED": {"wk6": 7.7, "wk12": 17.1},
        "SDD": {"wk6": 9.0, "wk12": 19.8},
        "delta_D": {"wk0_6": 18.7, "wk6_12": 13.0, "wk0_12": 31.7},
    },
    "B": {
        "SED": {"wk6": 101.3, "wk12": 268.0},
        "SDD": {"wk6": 230.0, "wk12": 608.0},
        "delta_D": {"wk0_6": 40.7, "wk6_12": 22.1, "wk0_12": 62.8},
    },
}

# Table 3 (p.1177), mean-column per-session SED anchors, and period sums.
TABLE3 = {
    "A": {"s1": 0.20, "s12": 1.00, "s24": 0.64, "sum1": 7.7, "sum2": 9.3},
    "B": {"s1": 4.1, "s12": 12.9, "s24": 14.1, "sum1": 101.3, "sum2": 166.3},
}

# Table 2 (p.1176), booth-specific measured spectral RDE = UVVitD/UVEry.
RDE = {"A": 1.16, "B": 2.27}

# Table 1 (p.1176), baseline 25(OH)D and skin-type counts.
BASELINE_D0 = {"A": 45.4, "B": 43.7}
FST_COUNTS = {"white_1_3": 45, "brown_4_5": 41, "black_6": 33}

# Fig. 4 (p.1179), group-mean log fit to the 4 Table-5 points (5th point --
# a near-zero anchor -- is described but not numerically given, so omitted).
FIG4_REPORTED_SLOPE = {"SED": 10.0, "SDD": 8.6}
FIG4_REPORTED_INTERCEPT = {"SED": -3.0, "SDD": 1.1}

# [CHOICE] representative FST per Fitzpatrick band used in McKenzie's
# grouping (they report 1-3 / 4-5 / 6, not individual FST). No sub-band
# distribution is given, so a single representative FST per band is used
# per band, with brown reported at both 4 and 5 rather than averaged, so
# the reader sees the spread rather than a hidden average.
REPRESENTATIVE_FST = {"white(1-3)": 2, "brown(4)": 4, "brown(5)": 5, "black(6)": 6}

# [ESTIMATE] exposed_bsa for booth exposure. Both booths are described as
# "walk-in chambers ... to receive measured full-body UV exposures"
# (p.1176). This repo's own convention for "full body" (Young 2021 FB arm,
# used to anchor A) is exposed_bsa = 0.85; used here for consistency, not
# because McKenzie states a number -- they do not.
EXPOSED_BSA_FULLBODY = 0.85


# =============================================================================
# (a) Does the log form hold up against alternatives, on the 4 tabulated
#     group-mean points?
# =============================================================================

def _fit_and_sse(x, y, form):
    """Least-squares 2-parameter fit of `form` to (x, y). Returns (params, sse, r2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if form == "log":
        basis = np.log(x)
    elif form == "linear":
        basis = x
    elif form == "sqrt":
        basis = np.sqrt(x)
    elif form == "power":
        # y = a * x^p + b is nonlinear in p; grid-search p, linear LSQ for a,b.
        best = None
        for p in np.linspace(0.05, 1.5, 146):
            basis_p = x ** p
            A_mat = np.vstack([basis_p, np.ones_like(basis_p)]).T
            coef, *_ = np.linalg.lstsq(A_mat, y, rcond=None)
            pred = A_mat @ coef
            sse = float(np.sum((y - pred) ** 2))
            if best is None or sse < best[0]:
                best = (sse, p, coef)
        sse, p, coef = best
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
        return {"a": float(coef[0]), "b": float(coef[1]), "p": float(p)}, sse, r2
    else:
        raise ValueError(form)
    A_mat = np.vstack([basis, np.ones_like(basis)]).T
    coef, *_ = np.linalg.lstsq(A_mat, y, rcond=None)
    pred = A_mat @ coef
    sse = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
    return {"a": float(coef[0]), "b": float(coef[1])}, sse, r2


def log_form_check():
    """(a) Fit ln(dose), linear, sqrt, power-law to the 4 Table-5 mean points."""
    print("=" * 78)
    print("(a) LOG-FORM CHECK -- 4 tabulated group-mean points (Table 5 / Fig 4)")
    print("=" * 78)

    results = {}
    for unit in ("SED", "SDD"):
        x = [TABLE5["A"][unit]["wk6"], TABLE5["A"][unit]["wk12"],
             TABLE5["B"][unit]["wk6"], TABLE5["B"][unit]["wk12"]]
        y = [TABLE5["A"]["delta_D"]["wk0_6"], TABLE5["A"]["delta_D"]["wk0_12"],
             TABLE5["B"]["delta_D"]["wk0_6"], TABLE5["B"]["delta_D"]["wk0_12"]]
        print(f"\n--- dose in {unit} ---")
        print(f"  x = {x}")
        print(f"  y (delta 25(OH)D, nmol/L) = {y}")
        for form in ("linear", "sqrt", "log", "power"):
            params, sse, r2 = _fit_and_sse(x, y, form)
            results[(unit, form)] = {"params": params, "sse": sse, "r2": r2}
            extra = f", p={params['p']:.2f}" if "p" in params else ""
            print(f"  {form:8s}: a={params['a']:8.3f} b={params['b']:8.3f}{extra}"
                  f"   SSE={sse:8.3f}  R^2={r2:6.3f}")

        log_p = results[(unit, "log")]["params"]
        print(f"  This script's own log fit: {log_p['a']:.2f}*ln({unit}) + {log_p['b']:.2f}"
              f"   [paper reports {FIG4_REPORTED_SLOPE[unit]:.1f}*ln({unit})"
              f" + {FIG4_REPORTED_INTERCEPT[unit]:.1f} for the same 4 points]")

    print("\n  INTERPRETATION (n=4 points, 2-parameter models -> 2 residual d.f.:")
    print("  this is a WEAK statistical test; every form gets a high R^2 with only")
    print("  4 points). The informative fact is not R^2 but the SHAPE: booth B's")
    print("  dose is ~15-30x booth A's (SED) or ~25-60x (SDD), yet booth B's")
    print("  delta-D is only ~2x booth A's. A LINEAR model cannot produce that")
    print("  compression without a near-zero slope that then undershoots booth A's")
    print("  low-dose response -- exactly the qualitative failure mode ln(1+x) is")
    print("  chosen to avoid. That directional fact is robust even with 4 points;")
    print("  which of {log, sqrt, power p<1} fits best numerically is NOT robust")
    print("  with only 4 points and is not treated as decisive here.")
    print("  This independently reproduces (not just cites) the docstring's claim")
    print("  that McKenzie corroborates the log FORM: a saturating/concave curve is")
    print("  required by their own numbers, and ln(1+x) is one of several concave")
    print("  forms consistent with it -- McKenzie's own Discussion (p.1181) makes")
    print("  the same point, offering an exponentially-decaying-marginal-rate form")
    print("  as an alternative they consider 'qualitatively consistent' with Fig 4.")
    print("  So: FORM corroborated (concave, strong early saturation) -- SPECIFIC")
    print("  functional family (ln(1+x) vs. alternatives) NOT uniquely discriminated.")
    return results


# =============================================================================
# (b) Reconstruct each booth's protocol, run vitd_model.simulate(), and solve
#     for the A the CURRENT model would need to reproduce observed delta-D.
# =============================================================================

def build_booth_schedule(booth: str):
    """24 twice-weekly sessions over 84 days, dose linearly interpolated
    between Table 3's 3 anchor sessions (1, 12, 24) within each 6-week
    period, then RESCALED so each period's summed SED matches Table 3/5's
    exact reported period sums. [CHOICE]: rescaling trades exact per-session
    fidelity (not tabulated) for exact period-cumulative fidelity (tabulated,
    and what drives the log-dose response)."""
    t3 = TABLE3[booth]
    days = [3.5 * i for i in range(24)]  # session index 0..23 -> day
    raw = []
    for s in range(1, 25):
        if s <= 12:
            frac = (s - 1) / 11.0
            dose = t3["s1"] + (t3["s12"] - t3["s1"]) * frac
        else:
            frac = (s - 12) / 12.0
            dose = t3["s12"] + (t3["s24"] - t3["s12"]) * frac
        raw.append(dose)
    raw1, raw2 = raw[:12], raw[12:]
    scale1 = t3["sum1"] / sum(raw1)
    scale2 = t3["sum2"] / sum(raw2)
    doses = [d * scale1 for d in raw1] + [d * scale2 for d in raw2]
    schedule = list(zip(days, doses))
    day_wk6 = days[11]
    day_wk12 = days[23]
    return schedule, day_wk6, day_wk12


def _D_at(result, day, tol=1e-6):
    t = result["t"]
    idx = np.where(np.abs(t - day) < tol)[0]
    if len(idx) == 0:
        idx = [int(np.argmin(np.abs(t - day)))]
    return float(result["D"][idx[-1]])


def _sim_total_delta(A, fst, schedule, D0, e_source, exposed_bsa, duration):
    params = ModelParams(A=A)
    res = simulate(fst=fst, schedule=schedule, duration_days=duration,
                    params=params, D0=D0, E_source_override=e_source,
                    exposed_bsa_override=exposed_bsa)
    return res["D"][-1] - D0


def solve_implied_A(target_delta, fst, schedule, D0, e_source, exposed_bsa,
                     duration, lo=0.05, hi=200.0, iters=60):
    def f(A):
        return _sim_total_delta(A, fst, schedule, D0, e_source, exposed_bsa, duration) - target_delta

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0:
        return float("nan"), "target below model floor even at A=%.2f" % lo
    if f_hi < 0:
        return float("nan"), "target above model ceiling even at A=%.1f (feedback-saturated)" % hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), "ok"


def convert_m_and_implied_A():
    """(b) Convert McKenzie's SDD dose using the booth's OWN Table-2 RDE
    (not the generic SOURCE_CALIBRATION_TABLE entry), simulate the
    reconstructed protocol, and solve for the A the current model would
    need to reproduce the observed 12-week delta-D, per representative FST.
    """
    print("\n" + "=" * 78)
    print("(b) CONVERTING m -> implied A via booth's own Table-2 RDE + simulate()")
    print("=" * 78)
    print(f"\n  Booth A measured RDE (Table 2) = {RDE['A']}  |  SOURCE_CALIBRATION_TABLE"
          f" tl01_nb_uvb (booth B's lamp family) midpoint = {get_E_source('tl01_nb_uvb'):.2f}"
          f" (range 2.24-2.63)")
    print(f"  Booth B measured RDE (Table 2) = {RDE['B']}  -- sits inside that range,")
    print("  near the floor, and was already logged there as corroboration.")
    print("\n  FINDING re: the docstring's claim ('not directly convertible without")
    print("  their booth's own spectral RDE') -- that RDE IS printed, in Table 2")
    print("  (1.16 / 2.27), measured from the booths' own spectrographs. The")
    print("  conversion the docstring says is missing is therefore AVAILABLE; what")
    print("  is genuinely missing is not the RDE but a TABULATED numeric m (see")
    print("  module docstring 'APPENDIX A2' section) -- the paper describes the")
    print("  fit's consequences in prose but never prints m or F(X)'s coefficients.")
    print("  So (b) proceeds using vitd_model.simulate() directly on reconstructed")
    print("  per-session doses (Table 3) converted through RDE via E_source_override,")
    print("  rather than trying to algebraically invert an m that was never printed.")

    rows = []
    for booth in ("A", "B"):
        schedule, day_wk6, day_wk12 = build_booth_schedule(booth)
        D0 = BASELINE_D0[booth]
        e_source = RDE[booth]
        target = TABLE5[booth]["delta_D"]["wk0_12"]
        print(f"\n  --- Booth {booth} (RDE={e_source}, D0={D0}, target 12-wk"
              f" delta-D={target}) ---")
        for band, fst in REPRESENTATIVE_FST.items():
            A_implied, status = solve_implied_A(
                target, fst, schedule, D0, e_source, EXPOSED_BSA_FULLBODY,
                duration=84.5)
            print(f"    FST {fst} ({band:12s}): implied A = "
                  f"{A_implied:6.2f}" if status == "ok" else
                  f"    FST {fst} ({band:12s}): {status}")
            rows.append({
                "booth": booth, "band": band, "fst": fst,
                "RDE": e_source, "D0": D0, "target_delta_D_12wk": target,
                "implied_A": A_implied, "status": status,
                "current_model_A": ModelParams().A,
            })

    current_A = ModelParams().A
    print(f"\n  Current model A = {current_A}. Compare implied values above.")
    print("  READ WITH CAUTION: these numbers answer 'what would A need to be for")
    print("  the CURRENT model's melanin/BSA/feedback machinery to reproduce this")
    print("  population's POOLED response, if that whole pooled response were")
    print("  assigned to one FST at a time' -- NOT 'what McKenzie's data says A is")
    print("  for that FST', because McKenzie never stratifies by FST (see caveat in")
    print("  module docstring). The spread across FST 2/4/5/6 below is therefore a")
    print("  property of this model's own melanin attenuation curve, applied to one")
    print("  pooled target, not a measured FST gradient from McKenzie's data.")
    return rows


# =============================================================================
# (c) Second-6-week attenuation: does the CURRENT model reproduce it?
# =============================================================================

def second_half_attenuation_check():
    print("\n" + "=" * 78)
    print("(c) SECOND-6-WEEK ATTENUATION -- out-of-sample check at CURRENT A")
    print("=" * 78)
    print(f"\n  Current model: A = {ModelParams().A}, K_const = {ModelParams().K_const},"
          f" feedback_mode = {ModelParams().feedback_mode}")

    rows = []
    for booth in ("A", "B"):
        schedule, day_wk6, day_wk12 = build_booth_schedule(booth)
        D0 = BASELINE_D0[booth]
        e_source = RDE[booth]
        obs_1 = TABLE5[booth]["delta_D"]["wk0_6"]
        obs_2 = TABLE5[booth]["delta_D"]["wk6_12"]
        obs_ratio = obs_2 / obs_1
        print(f"\n  --- Booth {booth}: observed delta-D wk0-6={obs_1}, wk6-12={obs_2},"
              f" ratio(2nd:1st)={obs_ratio:.3f} ---")
        for band, fst in REPRESENTATIVE_FST.items():
            params = ModelParams()
            res = simulate(fst=fst, schedule=schedule, duration_days=84.5,
                            params=params, D0=D0, E_source_override=e_source,
                            exposed_bsa_override=EXPOSED_BSA_FULLBODY)
            D_wk6 = _D_at(res, day_wk6)
            D_wk12 = _D_at(res, day_wk12)
            model_1 = D_wk6 - D0
            model_2 = D_wk12 - D_wk6
            model_ratio = model_2 / model_1 if model_1 != 0 else float("nan")
            verdict = ("OVER-predicts persistence (ratio too high)"
                       if model_ratio > obs_ratio else
                       "UNDER-predicts persistence (ratio too low, i.e. model"
                       " attenuates MORE than observed)")
            print(f"    FST {fst} ({band:12s}): model delta wk0-6={model_1:6.2f},"
                  f" wk6-12={model_2:6.2f}, ratio={model_ratio:.3f}  -> {verdict}")
            rows.append({
                "booth": booth, "band": band, "fst": fst,
                "obs_delta_wk0_6": obs_1, "obs_delta_wk6_12": obs_2,
                "obs_ratio_2nd_to_1st": obs_ratio,
                "model_delta_wk0_6": model_1, "model_delta_wk6_12": model_2,
                "model_ratio_2nd_to_1st": model_ratio,
            })

    print("\n  INTERPRETATION: the model's only prior anchor for response falling")
    print("  over time is Karppinen (per Gap-3 note). This is a genuine")
    print("  out-of-sample comparison against a second, independent dataset.")
    return rows


# =============================================================================
# main
# =============================================================================

def main():
    a_results = log_form_check()
    b_rows = convert_m_and_implied_A()
    c_rows = second_half_attenuation_check()

    out_path = os.path.join(BASE_DIR, "tables", "mckenzie2012_check.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "booth", "band", "fst", "key", "value"])
        for r in b_rows:
            for k in ("RDE", "D0", "target_delta_D_12wk", "implied_A",
                      "status", "current_model_A"):
                writer.writerow(["b_implied_A", r["booth"], r["band"], r["fst"], k, r[k]])
        for r in c_rows:
            for k in ("obs_delta_wk0_6", "obs_delta_wk6_12",
                      "obs_ratio_2nd_to_1st", "model_delta_wk0_6",
                      "model_delta_wk6_12", "model_ratio_2nd_to_1st"):
                writer.writerow(["c_attenuation", r["booth"], r["band"], r["fst"], k, r[k]])
        for (unit, form), res in a_results.items():
            writer.writerow(["a_log_form", "-", unit, "-", f"{form}_sse", res["sse"]])
            writer.writerow(["a_log_form", "-", unit, "-", f"{form}_r2", res["r2"]])
    print(f"\nWrote {out_path}")

    # -------------------------------------------------------------------
    # Live values for the docstring-update text below (C7, 2026-09-21):
    # computed from this run's own b_rows / c_rows, not typed in, so the
    # printed block stays accurate across future recalibrations.
    # -------------------------------------------------------------------
    current_A = ModelParams().A
    implied_A_ok = [r["implied_A"] for r in b_rows if r["status"] == "ok"]
    if implied_A_ok:
        implied_A_range_str = f"{min(implied_A_ok):.3f}-{max(implied_A_ok):.3f}"
    else:
        implied_A_range_str = "no booth/FST combination solved (all out of bracket [0.05, 200])"

    ratio_lines = []
    for booth in ("A", "B"):
        booth_rows = [r for r in c_rows if r["booth"] == booth]
        obs_ratio = booth_rows[0]["obs_ratio_2nd_to_1st"]
        model_ratios = [r["model_ratio_2nd_to_1st"] for r in booth_rows]
        ratio_lines.append(
            f"booth {booth} observed {obs_ratio:.3f} vs. model "
            f"{min(model_ratios):.3f}-{max(model_ratios):.3f} (FST 2/4/5/6)"
        )
    ratio_str = "; ".join(ratio_lines)
    over_under = [
        "OVER-predicts" if r["model_ratio_2nd_to_1st"] > r["obs_ratio_2nd_to_1st"]
        else "UNDER-predicts"
        for r in c_rows
    ]
    attenuation_verdict = (
        "the model OVER-predicts persistence (attenuates less than observed)"
        if all(v == "OVER-predicts" for v in over_under) else
        "the model UNDER-predicts persistence (attenuates more than observed)"
        if all(v == "UNDER-predicts" for v in over_under) else
        "the model's direction relative to observed is MIXED across booth/FST"
        " (see per-row verdicts printed in section (c) above)"
    )

    print("\n" + "=" * 78)
    print("PROPOSED vitd_model.py DOCSTRING UPDATE (text only -- not applied;")
    print("vitd_model.py is off-limits to this script per task instructions)")
    print("=" * 78)
    print(f"""
  See this run's stdout / tables/mckenzie2012_check.csv for numbers. Suggested
  replacement for the Gap-3 McKenzie note (2026-08-20 -> superseding date):

  [NOTE <date>] McKenzie et al. 2012 follow-up (scripts/mckenzie2012_check.py).
  Re-verified: skin type (X1, skin reflectance) was tested as a covariate and
  found NON-SIGNIFICANT in McKenzie's own regression (p.1183) -- there is no
  per-FST m in this paper, only a pooled fit across n=119 (45 white/41 brown/
  33 black). The log FORM is corroborated only in the weak sense that their
  own 4 group-mean points require strong concavity (booth B ~15-30x the dose
  of booth A yields only ~2x the response) -- ln(1+x) is one of several
  concave forms consistent with that, not uniquely selected (McKenzie's own
  Discussion offers an alternative exponentially-decaying-rate form as
  equally consistent). The booth's spectral RDE (Table 2: 1.16 booth A,
  2.27 booth B) IS printed, so the earlier "not directly convertible without
  the booth's own spectral RDE" is now WRONG on that specific point; what is
  actually missing is not the RDE but a tabulated m (the paper describes the
  Appendix-A2 fit's consequences in prose only, never prints m or F(X)'s
  coefficients numerically). Simulating each booth's reconstructed 24-session
  protocol through vitd_model.simulate() at representative FST 2/4/5/6 and
  solving for the A that reproduces observed 12-week delta-D gives implied A
  in {implied_A_range_str} (this run) against the current model A={current_A}
  -- reported, not adopted (see calibration_approval.yaml gate).
{textwrap.fill(
      f"Second-6-week:first-6-week attenuation ratio, this run: {ratio_str}. "
      f"Net verdict: {attenuation_verdict} -- a genuine out-of-sample check "
      "against the model's biphasic feedback term, whose only other anchor "
      "is Karppinen.",
      width=78, initial_indent="  ", subsequent_indent="  ")}
""")


if __name__ == "__main__":
    main()
