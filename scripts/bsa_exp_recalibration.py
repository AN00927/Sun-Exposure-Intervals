"""
bsa_exp_recalibration.py
=========================
DERIVATION for the BSA_EXP recalibration approved 2026-09-06
(`calibration_approval.yaml`, `parameter_approvals.BSA_EXP`).

Bucket: FEW-TIME (calibration_approval.yaml script-lifecycle rule). Writes no
file; prints only. Not imported by run_all.py or any module -- confirmed by
grepping run_all.py for `_import_script_module` and this script's own name
before it was added to .gitignore.

READ THIS BEFORE COMPARING THE OUTPUT TO THE APPROVED VALUE
-----------------------------------------------------------
[DRIFT AUDITED 2026-09-22, NOT A BUG] This script's output is NOT expected to
equal the BSA_EXP recorded in `calibration_approval.yaml`, and the difference
is not a defect. The derivation is a DECONVOLUTION AGAINST THE MODEL: it asks
what BSA_EXP makes THIS MODEL, at ITS CURRENT PARAMETER SET, reproduce each
source's observed response ratio. Its answer therefore moves whenever A,
K_const, the feedback form, k_decay_D, M_baseline_by_FST or (since 2026-09-22)
the body-weight term moves -- and all of those have.

Measured trajectory of the same method on three different tree states:
    2026-09-06 (the approval)                     0.3281
    2026-09-22, W-W, before the A3/K_const change 0.3577
    2026-09-22, after it                          0.3537
The script reproduces its METHOD exactly. What it cannot reproduce is an
output frozen against a superseded tree state, and no script that inverts a
model can. The approved value is correctly read as "this derivation's output
AT THE 2026-09-06 TREE STATE", and that is how it is now labelled here, in
vitd_model.py's BSA_EXP entry and in calibration_approval.yaml.

The current output is NOT adopted -- see the [DRIFT AUDITED] paragraph in
vitd_model.py's BSA_EXP entry for why (it would need a third parameter in
the coupled (A, K_const) fixed-point iteration, for a shift smaller than this
script's own Step 2c sensitivity sweep over two unsourced protocol
assumptions).

One question, four parts, no rounding down of an inconvenient answer:

    Two WITHIN-STUDY sources put the APPARENT BSA scaling exponent at ~0.38
    (Young et al. 2021 PNAS 85% vs 3.7% BSA; the Vahavihu NB-UVB triple in
    Jager et al. 2018 Table III). But that exponent is measured on serum
    25(OH)D response RATIOS that already contain g(D) FEEDBACK damping: the
    larger-BSA arm reaches a higher serum D and is therefore MORE suppressed
    by g(D), which by itself would flatten an observed response ratio even if
    the true GEOMETRIC scaling were linear (BSA_EXP = 1). Setting
    BSA_EXP = 0.38 directly would double-count that damping. This script asks
    instead: what BSA_EXP, run THROUGH the model's own g(D), reproduces each
    source's observed ratio?

The approval's derivation clause anticipated that g(D) damping the
larger-BSA arm's response would push the deconvolved exponent ABOVE 0.38 --
possibly at or above 0.50, meaning the apparent plateau would turn out to be
mostly a feedback artefact. Run against this model's actual state equations,
that expectation does NOT hold: see Step 2 below, which finds the opposite
sign and explains why (k_decay_D acts on the whole state D, baseline
included, not only the increment, and that is a bigger effect here than
g(D) alone). Whatever direction the derivation returns is reported as run,
not adjusted toward the pre-registered expectation in either direction.

Design
------
STEP 1 -- reproduce the two apparent (feedback-contaminated) exponents from
printed source numbers, arithmetic only, no model.

STEP 2 -- deconvolve g(D). For each source, build the model scenario that
reproduces its protocol (same total dose, same baseline, varying only
exposed BSA), run it at the source's two (or three) %BSA points under the
model's DEFAULT feedback_mode='biphasic', and solve for the BSA_EXP at which
the model's own predicted response ratio equals the source's OBSERVED ratio.
BSA_EXP is a bare module-level constant (`vitd_model.BSA_EXP`), read by
`ModelParams.bsa_factor()` at call time -- there is no per-call override
plumbed through `simulate()`, so this script monkeypatches the module
attribute for the duration of each trial evaluation and restores it after.
Same deconvolution logic as `vitd_calibration.derive_K_ATTEN` (residual
after removing what feedback already explains), adapted from a closed-form
inversion (K_ATTEN, K_const) to a numerical one (BSA_EXP enters `bsa_factor`
through both arms of the ratio, non-linearly, so no closed form exists).

STEP 2d -- [ADDED 2026-09-22, W-W] variance-availability audit, mirroring
kconst_recalibration.py's S3c. Its conclusion is NEGATIVE: Young 2021
publishes SDs on the quantity its route inverts (Table 1) and Jager 2018
Table III publishes none at all, so a two-route inverse-variance pool is
undefined and NO VARIANCE IS IMPUTED to make it defined.

STEP 3 -- report the deconvolved value per source and an adopted combination
(equal-weight mean in log space). This mirrored kconst_recalibration.py's
[CHOICE] to weight per STUDY; that script moved its DEFAULT to
inverse-variance weighting on 2026-09-22 and THIS SITE DID NOT FOLLOW, for
the reason STEP 2d gives. BSA_EXP does not move under that change.

STEP 4 -- sensitivity to feedback form (linear / power / saturating /
biphasic / none), as `vitd_calibration.derive_K_ATTEN` reports for K_ATTEN.

Everything the script CHOOSES is tagged `# [CHOICE]`; everything it
ESTIMATES rather than reads from a printed source is tagged `# [ESTIMATE]`.

DARK-SKIN PROVENANCE: neither source supplies FST V-VI data. Young 2021 is
FST I-II only (P20); the Vahavihu triple's FST composition is not
established here (Jager 2018 could not stratify its pool by skin type at
all). bsa_factor multiplies A directly, so any BSA_EXP derived here COMPOUNDS
the light-skin extrapolation already flagged on A, rather than mitigating it.

Run
---
    PYTHONIOENCODING=utf-8 python3 scripts/bsa_exp_recalibration.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vitd_model
from vitd_model import ModelParams, simulate, get_E_source

OLD_BSA_EXP = 0.50  # [CHOICE] pinned literal so "before" stays "before"


def hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# STEP 1 -- apparent exponents, arithmetic only
# ===========================================================================

def step1_apparent_exponents():
    hr("STEP 1  APPARENT EXPONENTS FROM PRINTED SOURCE NUMBERS")

    print("""
  Young et al. 2021 PNAS 118:e2015867118 -- within-study 85% vs 3.7% BSA,
  Full-Body vs Partial-Body arms, same source, same protocol family.""")
    area_ratio_young = 0.85 / 0.037
    obs_ratio_young = 3.28
    apparent_young = np.log(obs_ratio_young) / np.log(area_ratio_young)
    print(f"    area ratio  = 0.85 / 0.037 = {area_ratio_young:.15f}")
    print(f"    response ratio (printed)   = {obs_ratio_young:.2f}")
    print(f"    apparent exponent = ln({obs_ratio_young:.2f}) / "
          f"ln({area_ratio_young:.6f}) = {apparent_young:.4f}")

    print("""
  Vahavihu NB-UVB triple, Jager et al. 2018 (Anticancer Res 38:1165-1171)
  Table III -- 13 SED cumulative, 6% / 29% / 100% BSA, 4 / 11 / 11.4 nmol/L.
  ONLY this within-study triple is used from Jager 2018 -- see the
  [DOUBLE-COUNT WARNING] in vitd_model.py's BSA_EXP entry: Jager's 15-study
  pool re-aggregates this repo's own anchors and its pooled meta-regression
  (0.05/%BSA) is non-significant with no SE/CI/p and must not be used for
  anything.""")
    pts = [(6.0, 4.0), (29.0, 11.0), (100.0, 11.4)]
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    slope_ols = float(np.polyfit(x, y, 1)[0])
    slope_endpoints = float(np.log(pts[2][1] / pts[0][1])
                             / np.log(pts[2][0] / pts[0][0]))
    slope_low = float(np.log(pts[1][1] / pts[0][1])
                       / np.log(pts[1][0] / pts[0][0]))
    slope_high = float(np.log(pts[2][1] / pts[1][1])
                        / np.log(pts[2][0] / pts[1][0]))
    print(f"    log-log OLS across all 3 points   = {slope_ols:.4f}")
    print(f"    endpoints only (6% -> 100%)        = {slope_endpoints:.4f}")
    print(f"    segment 6% -> 29%  (genuine rise)   = {slope_low:.4f}")
    print(f"    segment 29% -> 100% (near-plateau)  = {slope_high:.4f}")
    print("""
  Two independent studies, sources, doses and BSA ranges reproducing each
  other to two decimals (0.3790, 0.3843) is the most striking convergence in
  the evidence base for this parameter -- and BOTH are on the same
  APPARENT, feedback-contaminated basis.""")
    return dict(young=apparent_young, vahavihu_ols=slope_ols,
                vahavihu_endpoints=slope_endpoints)


# ===========================================================================
# STEP 2 -- deconvolution
# ===========================================================================

class bsa_exp_override:
    """Temporarily monkeypatch the module-level BSA_EXP constant.

    `ModelParams.bsa_factor` reads the bare name `BSA_EXP`, resolved at call
    time from `vitd_model`'s module namespace -- there is no per-call
    override parameter threaded through `synthesis_per_session`/`simulate`
    for it (unlike K_const, K_ATTEN, K_SC, ...). This context manager is the
    least invasive way to sweep it without changing the model's calling
    convention for this derivation script alone.
    """

    def __init__(self, value):
        self.value = value
        self.old = None

    def __enter__(self):
        self.old = vitd_model.BSA_EXP
        vitd_model.BSA_EXP = self.value
        return self

    def __exit__(self, *exc):
        vitd_model.BSA_EXP = self.old
        return False


# --- Young 2021 scenario ---------------------------------------------------
# [ESTIMATE] Young 2021's PNAS full text does not print the Full-Body arm's
# session-by-session schedule in a form transcribed elsewhere in this repo;
# what IS printed and already used for the A anchor (vitd_model.py, A entry)
# is: Arimed B FB arm, D0 = 34.6 nmol/L, cumulative 10 SED, rising to
# observed delta = +38.6 nmol/L over the actual (not single-session) course.
# The PB arm is assumed to share the SAME total dose and session structure,
# differing only in exposed BSA -- the natural within-study design for
# isolating a BSA effect, and consistent with "within-study 85% vs 3.7%"
# framing. Session count/spacing (5 x 2 SED, 3.5 d apart) is carried over
# from Young 2020's protocol convention (same trial group, same cabinet
# family) since Young 2021's own spacing is not transcribed here -- FLAGGED,
# not asserted as read from the 2021 paper itself.
YOUNG_D0 = 34.6
YOUNG_SESSIONS = 5
YOUNG_SED_PER_SESSION = 2.0
YOUNG_INTERVAL_D = 3.5
YOUNG_BSA_HI = 0.85
YOUNG_BSA_LO = 0.037
YOUNG_OBS_RATIO = 3.28


def young_response(bsa: float, feedback_mode: str = "biphasic") -> float:
    p = ModelParams(feedback_mode=feedback_mode)
    sched = [(i * YOUNG_INTERVAL_D, YOUNG_SED_PER_SESSION)
             for i in range(YOUNG_SESSIONS)]
    dur = int(np.ceil((YOUNG_SESSIONS - 1) * YOUNG_INTERVAL_D)) + 2
    res = simulate(fst=2, schedule=sched, duration_days=dur, params=p,
                   D0=YOUNG_D0, E_source_override=get_E_source("arimed_b"),
                   exposed_bsa_override=bsa, dose_framing="fixed_absolute_SED")
    return float(res["D"][-1]) - YOUNG_D0


# --- Vahavihu (Jager 2018 Table III) scenario ------------------------------
# [ESTIMATE] Jager 2018 Table III prints only the endpoint doses, %BSA and
# response for the Vahavihu triple, not the underlying session schedule or
# each arm's baseline 25(OH)D. Total dose (13 SED) and number of sessions are
# assumed delivered as an NB-UVB course of 6 sessions ~3.5 d apart
# (a typical twice-to-thrice weekly NB-UVB course length; NOT read from the
# paper). D0 = 40.0 nmol/L is a placeholder winter-deficient baseline
# [ESTIMATE], sensitivity to both choices is reported below (S2c) precisely
# because neither is sourced. FST assumed II (Finnish/Nordic cohort,
# consistent with Ala-Houhala, a co-anchor of this repo's Vahavihu-adjacent
# literature) -- Jager 2018 could not stratify by skin type at all, so this
# is this script's own assumption, not the paper's.
VAHAVIHU_D0 = 40.0
VAHAVIHU_SESSIONS = 6
VAHAVIHU_TOTAL_SED = 13.0
VAHAVIHU_INTERVAL_D = 3.5
VAHAVIHU_BSA_LO = 0.06
VAHAVIHU_BSA_HI = 1.00
VAHAVIHU_OBS_RATIO_ENDPOINTS = 11.4 / 4.0


def vahavihu_response(bsa: float, feedback_mode: str = "biphasic",
                       sessions: int = VAHAVIHU_SESSIONS,
                       D0: float = VAHAVIHU_D0) -> float:
    p = ModelParams(feedback_mode=feedback_mode)
    sed_per = VAHAVIHU_TOTAL_SED / sessions
    sched = [(i * VAHAVIHU_INTERVAL_D, sed_per) for i in range(sessions)]
    dur = int(np.ceil((sessions - 1) * VAHAVIHU_INTERVAL_D)) + 2
    res = simulate(fst=2, schedule=sched, duration_days=dur, params=p,
                   D0=D0, E_source_override=get_E_source("tl01_nb_uvb"),
                   exposed_bsa_override=bsa, dose_framing="fixed_absolute_SED")
    return float(res["D"][-1]) - D0


def invert_bsa_exp(response_fn, bsa_lo, bsa_hi, obs_ratio,
                    lo=0.0, hi=0.8, n=161, **kwargs):
    """Find BSA_EXP such that response_fn(bsa_hi)/response_fn(bsa_lo) == obs.

    Grid + bisection, mirroring kconst_recalibration.invert_one, INCLUDING
    its turnover guard: the low-BSA arm's net response (gross synthesis minus
    decay of the WHOLE state, baseline included) can turn negative once
    BSA_EXP is large enough to starve it, at which point the ratio stops
    being a meaningful monotone function of BSA_EXP. hi=0.8 (not 2.0) keeps
    the search inside the region where both arms' net response stays
    positive for both sources tested here -- verified by inspection, not
    asserted a priori.

    NOTE ON DIRECTION -- READ BEFORE CHANGING THIS FUNCTION. The approval
    entry for BSA_EXP anticipated the deconvolved value would come out ABOVE
    the apparent one, on the reasoning that g(D) suppresses the higher-BSA
    (higher-D) arm's synthesis more, which would need MORE geometric scaling
    to reproduce the same observed ratio. That reasoning is INCOMPLETE for
    this model: k_decay_D(D) applies to the WHOLE state D, including the
    pre-exposure baseline, not only the incremental synthesis. Over a
    multi-week course, baseline decay is a near-fixed absolute drag that eats
    a much larger FRACTION of the low-BSA arm's small gross synthesis than of
    the high-BSA arm's large gross synthesis -- this widens the modelled
    ratio beyond the pure bsa_factor ratio, in the OPPOSITE direction from
    the g(D)-only argument, and dominates it for both sources tested here.
    Net effect: the deconvolved BSA_EXP for both sources comes out BELOW
    their apparent exponents, not above -- see Step 2's printed comparison
    and the SUMMARY at the end of this script for the actual sign and
    magnitude of that gap.
    """
    grid = np.linspace(lo, hi, n)
    vals = []
    for e in grid:
        with bsa_exp_override(e):
            hi_r, lo_r = response_fn(bsa_hi, **kwargs), response_fn(bsa_lo, **kwargs)
        if lo_r > 0 and np.isfinite(hi_r) and np.isfinite(lo_r):
            vals.append(hi_r / lo_r)
        else:
            vals.append(float("nan"))
    if not np.isfinite(vals[0]) or vals[0] >= obs_ratio:
        return grid[0], "at-bound-low"
    for i in range(1, n):
        if not np.isfinite(vals[i]):
            return grid[i - 1], "no-crossing-below-turnover"
        if vals[i] >= obs_ratio:
            a, b = grid[i - 1], grid[i]
            for _ in range(50):
                mid = 0.5 * (a + b)
                with bsa_exp_override(mid):
                    hi_r, lo_r = (response_fn(bsa_hi, **kwargs),
                                  response_fn(bsa_lo, **kwargs))
                r = hi_r / lo_r if lo_r > 0 else float("inf")
                if r < obs_ratio:
                    a = mid
                else:
                    b = mid
            return 0.5 * (a + b), None
    return grid[-1], "at-bound-high"


def step2_deconvolution():
    hr("STEP 2  DECONVOLVING g(D): BSA_EXP THAT REPRODUCES EACH OBSERVED "
       "RATIO UNDER THE MODEL'S OWN FEEDBACK")
    print(f"""
  For each source, sweep BSA_EXP (with everything else -- A, K_const,
  feedback_mode='biphasic' default, D0, dose -- held at the source's own
  protocol) until the model's predicted response ratio across the source's
  BSA span equals the OBSERVED ratio. This is the exponent net of whatever
  the model's own state equations (feedback AND decay) already contribute to
  the ratio at a given BSA_EXP -- see the note below invert_bsa_exp's
  docstring for why that turns out to matter more than g(D) alone here.
""")
    with bsa_exp_override(OLD_BSA_EXP):
        young_at_old = young_response(YOUNG_BSA_HI) / young_response(YOUNG_BSA_LO)
        vaha_at_old = (vahavihu_response(VAHAVIHU_BSA_HI)
                       / vahavihu_response(VAHAVIHU_BSA_LO))
    print(f"  At the SHIPPED BSA_EXP = {OLD_BSA_EXP:.2f}:")
    print(f"    Young 2021 model ratio (85%/3.7%)      = {young_at_old:.4f}"
          f"   (observed {YOUNG_OBS_RATIO:.2f})")
    print(f"    Vahavihu model ratio (100%/6%)          = {vaha_at_old:.4f}"
          f"   (observed {VAHAVIHU_OBS_RATIO_ENDPOINTS:.4f})")

    e_young, flag_young = invert_bsa_exp(young_response, YOUNG_BSA_LO,
                                          YOUNG_BSA_HI, YOUNG_OBS_RATIO)
    e_vaha, flag_vaha = invert_bsa_exp(vahavihu_response, VAHAVIHU_BSA_LO,
                                        VAHAVIHU_BSA_HI,
                                        VAHAVIHU_OBS_RATIO_ENDPOINTS)
    print(f"\n  DECONVOLVED BSA_EXP (feedback_mode = 'biphasic', default):")
    print(f"    Young 2021 (85% vs 3.7% BSA)      -> {e_young:.4f}"
          f"{'  [' + flag_young + ']' if flag_young else ''}")
    print(f"    Vahavihu triple (100% vs 6% BSA)  -> {e_vaha:.4f}"
          f"{'  [' + flag_vaha + ']' if flag_vaha else ''}")
    print(f"""
  *** UNEXPECTED DIRECTION, REPORTED PLAINLY, NOT FORCED -- READ THIS ***
  Both deconvolved values sit BELOW their apparent (Step 1) counterparts:
  {e_young:.4f} vs apparent 0.3790 for Young ({e_young/0.3790:.3f}x); {e_vaha:.4f}
  vs apparent 0.3723 (endpoints) for Vahavihu ({e_vaha/0.3723:.3f}x). This is
  the OPPOSITE of the direction calibration_approval.yaml's derivation clause
  anticipated (g(D) alone would need MORE exponent to reproduce the same
  ratio, since it suppresses the higher-BSA arm more).
  The reason, diagnosed directly from this model's own state equations: at
  0.85 vs 0.037 (or 1.00 vs 0.06) BSA, the LOW-BSA arm's gross synthesis over
  a multi-week course is small, and k_decay_D(D) acts on the WHOLE state D
  (pre-exposure baseline included), not only the incremental synthesis. That
  decay is a near-fixed absolute drag which eats a much larger FRACTION of
  the low-BSA arm's small gross gain than of the high-BSA arm's large one --
  this widens the modelled ratio in the SAME direction as raising BSA_EXP,
  so LESS geometric exponent is needed to match the same observed ratio, not
  more. This effect is compounded by a pre-existing, already-documented
  model limitation (vitd_model.py, A entry, 'NOT COVERED'): at these extreme
  BSA fractions the model under-predicts Young's absolute rises by 2.7x (FB)
  to 6.2x (PB) at any A consistent with the rest of the validation suite, so
  the low-BSA arm's modelled gross response here is smaller, relative to
  baseline decay drag, than the true physiology -- amplifying this artefact
  further. See the SUMMARY at the end of this script for the numbers this
  produces, and vitd_model.py's BSA_EXP docstring entry for how this is
  written up.""")
    return e_young, e_vaha


def step2c_vahavihu_sensitivity():
    hr("STEP 2c  VAHAVIHU: SENSITIVITY TO THE TWO UNSOURCED PROTOCOL "
       "ASSUMPTIONS (session count, baseline D0)")
    print("""
  Neither the Vahavihu session schedule nor its baseline 25(OH)D is printed
  in Jager 2018 Table III (only cumulative SED, %BSA and response are). The
  deconvolved exponent's sensitivity to both assumed inputs is reported here
  so the Step 2 headline number is not read as more sourced than it is.
""")
    print(f"  {'sessions':>9} {'D0':>6} {'deconvolved BSA_EXP':>20}")
    for sessions in (3, 6, 9, 12):
        for D0 in (25.0, 40.0, 55.0, 70.0):
            e, flag = invert_bsa_exp(
                vahavihu_response, VAHAVIHU_BSA_LO, VAHAVIHU_BSA_HI,
                VAHAVIHU_OBS_RATIO_ENDPOINTS, sessions=sessions, D0=D0)
            print(f"  {sessions:>9} {D0:>6.0f} {e:>20.4f}"
                  + (f"  [{flag}]" if flag else ""))


# ===========================================================================
# STEP 3 -- adopted combination
# ===========================================================================

#: Young et al. 2021 PNAS Table 1, "Demographics of PB and FB studies with
#: baseline, final, and delta 25(OH)D3 values" -- PUBLISHED SDs on the
#: quantity this route's observed ratio is built from:
#:     All FB  n = 40, delta 25(OH)D3 = 36.1 +/- 19.8 nmol/L
#:     All PB  n = 35, delta 25(OH)D3 = 11.0 +/- 7.8 nmol/L
#: ratio 36.1 / 11.0 = 3.281818..., i.e. the "3.28" STEP 1 reads off.
#: Verified from the PDF 2026-09-22 (W-W); this repo previously carried the
#: 3.28 as a bare point estimate with no dispersion attached.
YOUNG2021_BSA_DELTA = {"FB": (36.1, 19.8, 40), "PB": (11.0, 7.8, 35)}


def young2021_se_ln_ratio():
    """Delta-method SE on ln(FB delta / PB delta) from Young 2021 Table 1."""
    (mf, sf, nf), (mp, sp, npb) = (YOUNG2021_BSA_DELTA["FB"],
                                   YOUNG2021_BSA_DELTA["PB"])
    return float(np.hypot((sf / np.sqrt(nf)) / mf, (sp / np.sqrt(npb)) / mp))


def step2d_variance_audit():
    """Variance-availability audit for the two BSA_EXP routes.

    Added 2026-09-22 (W-W), mirroring kconst_recalibration.py's S3c. The
    conclusion here is NEGATIVE and is stated as such: inverse-variance
    weighting is NOT computable for this quantity.
    """
    hr("STEP 2d  VARIANCE-AVAILABILITY AUDIT (inverse-variance weighting)")
    se_ln = young2021_se_ln_ratio()
    r = (YOUNG2021_BSA_DELTA["FB"][0] / YOUNG2021_BSA_DELTA["PB"][0])
    print(f"""
  Young et al. 2021 PNAS, Table 1 -- VARIANCE PUBLISHED.
    All FB  n = 40   delta 25(OH)D3 = 36.1 +/- 19.8 nmol/L (SD)
    All PB  n = 35   delta 25(OH)D3 = 11.0 +/-  7.8 nmol/L (SD)
    observed response ratio = {r:.6f}   (printed in the paper as 3.28)
    delta-method SE(ln ratio) = {se_ln:.6f}
    implied 95% CI on the ratio = {r*np.exp(-1.959963984540054*se_ln):.6f} to
    {r*np.exp(1.959963984540054*se_ln):.6f}

  Vahavihu / Jager et al. 2018, Anticancer Res 38:1165-1171, Table III --
  NO VARIANCE OF ANY KIND. Read from the PDF 2026-09-22: Table III prints
  six rows of (study ref, UV source, total dose SED, BSA %, delta 25(OH)D,
  delta/SED, delta/SED/%BSA) as BARE POINT VALUES. There is no SD, SE, CI
  or n column anywhere in it, and the Vahavihu triple this script uses
  (rows tagged ref. 25, 13 SED at 6 / 29 / 100% BSA -> 4 / 11 / 11.4 nmol/L)
  is no exception.

  [CONSEQUENCE, NEGATIVE AND STATED AS SUCH] Inverse-variance weighting
  needs a variance for EVERY route in the pool; relative weights are not
  defined otherwise. This pool has exactly two routes and only one of them
  has a variance, so there is nothing for Young's precision to be weighted
  AGAINST. Inverse-variance weighting is therefore NOT COMPUTABLE for
  BSA_EXP, and NO VARIANCE IS IMPUTED for Vahavihu to make it computable.
  The equal-weight geometric mean below is RETAINED unchanged, and BSA_EXP
  does not move under the 2026-09-22 weighting change. This is a real
  limitation of the method at this site, not a null result to be glossed:
  it means half this parameter's evidence base carries no stated precision
  at all.

  WHAT WOULD UNBLOCK IT: a dispersion figure on the Vahavihu NB-UVB triple
  from its own primary source (Vahavihu et al., the study Jager's ref. 25
  points at), which is NOT held in this repository -- Jager 2018, the
  secondary source the numbers arrive through, IS held and does not carry
  one. Also still absent: Jager's Table III gives no per-row n, so even a
  reported SD could not be turned into an SE without it.""")
    return se_ln


# ---------------------------------------------------------------------------
# Weighting schemes -- named for parity with kconst_recalibration.py, even
# though only one of them is computable here. See STEP 2d.
# ---------------------------------------------------------------------------
WEIGHTING_SCHEMES = ("one_vote_per_study", "inverse_variance")

#: [CHOICE 2026-09-22, W-W] Stays `one_vote_per_study` -- NOT because the
#: incumbent was preferred, but because `inverse_variance` is not computable
#: at this site (STEP 2d). Recorded explicitly so a future reader does not
#: mistake the absence of a change for the absence of an audit.
DEFAULT_WEIGHTING_SCHEME = "one_vote_per_study"
BSA_EXP_WEIGHTING_NOT_APPLICABLE_REASON = (
    "Vahavihu/Jager 2018 Table III publishes no SD, SE, CI or per-row n; "
    "a two-route inverse-variance pool with one variance is undefined, and "
    "no variance is imputed.")


def step3_combination(e_young, e_vaha):
    hr("STEP 3  ADOPTED COMBINATION")
    # [CHOICE] Equal weight per STUDY, mean in LOG space -- the same
    # convention kconst_recalibration.py's four-route estimator used before
    # 2026-09-22. Note that kconst_recalibration.py has SINCE moved its
    # default to inverse-variance weighting (W-W); this site did NOT follow,
    # because STEP 2d establishes that inverse-variance weighting is not
    # computable here -- only one of the two routes publishes a variance.
    combined = float(np.exp(np.mean(np.log([e_young, e_vaha]))))
    print(f"""
  [CHOICE] Equal-weight mean in log space across the two independent
  within-study routes. RETAINED 2026-09-22 under the repo-wide
  inverse-variance weighting change, because that change is NOT COMPUTABLE
  at this site: {BSA_EXP_WEIGHTING_NOT_APPLICABLE_REASON}
  See STEP 2d for the full audit, including the Young 2021 SDs that ARE
  published and are now recorded even though they cannot be used to weight.
    Young 2021          {e_young:.4f}   (SE(ln observed ratio) {young2021_se_ln_ratio():.6f})
    Vahavihu (Jager18)  {e_vaha:.4f}   (no published variance)
    geometric mean       {combined:.4f}
""")
    if combined >= 0.50:
        print(f"""  *** RESULT, NOT FORCED DOWNWARD ***
  The deconvolved combination ({combined:.4f}) is AT OR ABOVE the withdrawn
  incumbent 0.50. Read plainly: once g(D) feedback is accounted for, the
  apparent ~0.38 plateau in serum 25(OH)D response vs %BSA is substantially
  or wholly a FEEDBACK artefact, not a geometric saturation of cutaneous
  synthesis with exposed area. This is consistent with Libon 2017's
  near-linear cholecalciferol (D3, upstream of feedback) result and the
  [DOUBLE-COUNT WARNING] already carried in vitd_model.py's BSA_EXP entry.
  Do NOT force this number down to 0.38 or to the withdrawn 0.50 -- both
  would re-introduce the double-count this derivation exists to remove.""")
    else:
        print(f"""  *** RESULT, NOT FORCED UPWARD ***
  The deconvolved combination ({combined:.4f}) is BELOW BOTH the apparent
  exponents (0.3790, 0.3723-0.3843) AND the withdrawn incumbent 0.50 -- the
  opposite of the direction the approval's derivation clause anticipated.
  Step 2's per-source printout diagnoses why: this model's state-dependent
  decay k_decay_D(D) acts on the WHOLE serum D level (baseline included),
  not only the incremental synthesis, and that decay is a near-fixed
  absolute drag that widens the modelled response ratio for a GIVEN BSA_EXP
  beyond what pure geometric scaling plus g(D) alone would produce -- so
  LESS exponent, not more, is needed to reproduce the same observed ratio.
  This is reported as the derivation's actual output, not adjusted toward
  either the apparent exponents or the pre-registered expectation of >=0.50.
  It should be read as evidence that mechanisms OTHER than the pure
  BSA-scaling exponent (specifically, the model's own decay structure)
  already explain more of the apparent plateau than g(D) feedback alone,
  which the approval's derivation clause did not anticipate.""")
    return combined


# ===========================================================================
# STEP 4 -- sensitivity to feedback form
# ===========================================================================

def step4_feedback_sensitivity():
    hr("STEP 4  SENSITIVITY TO FEEDBACK FORM")
    print(f"""
  {'feedback_mode':14} {'Young BSA_EXP':>14} {'Vahavihu BSA_EXP':>18} {'combined':>10}""")
    rows = []
    for mode in ("biphasic", "linear", "power", "saturating", "none"):
        try:
            e_y, _ = invert_bsa_exp(
                lambda bsa, m=mode: young_response(bsa, feedback_mode=m),
                YOUNG_BSA_LO, YOUNG_BSA_HI, YOUNG_OBS_RATIO)
            e_v, _ = invert_bsa_exp(
                lambda bsa, m=mode: vahavihu_response(bsa, feedback_mode=m),
                VAHAVIHU_BSA_LO, VAHAVIHU_BSA_HI, VAHAVIHU_OBS_RATIO_ENDPOINTS)
            comb = float(np.exp(np.mean(np.log([e_y, e_v]))))
            rows.append((mode, e_y, e_v, comb))
            print(f"  {mode:14} {e_y:>14.4f} {e_v:>18.4f} {comb:>10.4f}")
        except Exception as exc:  # feedback_mode='none' is not a real mode;
            print(f"  {mode:14}  (not a supported feedback_mode: {exc})")
    print("""
  'none' above (if reachable) would show the exponent needed with feedback
  entirely disabled -- the purely geometric reading -- as an upper-bound
  sanity check. If 'none' is not a supported feedback_mode string, that
  bound is instead read off feedback_mode='saturating' at a very large
  D_half_feedback, which is not attempted here to keep this script within
  the model's public interface.""")
    return rows


CHOICES = [
    "[CHOICE] Deconvolution target = model's own predicted response ratio "
    "across each source's BSA span, solved under feedback_mode='biphasic' "
    "(the model default), matching the K_ATTEN derivation's logic of "
    "removing what feedback already explains before reading a residual.",
    "[CHOICE] Combination = equal-weight geometric mean across exactly two "
    "independent within-study routes, same convention as "
    "kconst_recalibration.py's four-route estimator.",
    "[CHOICE] Vahavihu ratio target = endpoints (6% vs 100%), not the "
    "3-point OLS slope, so the deconvolution objective is a single number "
    "(the model is inverted on a scalar ratio, not fit to 3 points) and "
    "is directly comparable to Young's two-point contrast.",
    "[ESTIMATE] Young 2021 PB arm assumed to share the FB arm's total dose "
    "and session schedule (10 SED cumulative, 5 x 2 SED Arimed B, D0 = 34.6 "
    "nmol/L), differing only in exposed BSA. Session spacing (3.5 d) is "
    "carried over from Young 2020's protocol convention, not read from the "
    "2021 paper.",
    "[ESTIMATE] Vahavihu session count (6, ~3.5 d apart) and baseline D0 "
    "(40.0 nmol/L) are not printed in Jager 2018 Table III and are assumed; "
    "Step 2c reports the deconvolved exponent's sensitivity to both.",
    "[ESTIMATE] Vahavihu FST assumed II (Nordic/Finnish cohort); Jager 2018 "
    "could not stratify its pool by skin type at all, so this is this "
    "script's own assumption.",
    "[LIMIT] Neither source supplies FST V-VI data. bsa_factor multiplies A "
    "directly, so any BSA_EXP derived here COMPOUNDS the light-skin "
    "extrapolation already flagged on A, rather than mitigating it.",
]


def main():
    print(__doc__)
    step1 = step1_apparent_exponents()
    e_young, e_vaha = step2_deconvolution()
    step2c_vahavihu_sensitivity()
    step2d_variance_audit()
    combined = step3_combination(e_young, e_vaha)
    step4_feedback_sensitivity()

    hr("EVERY [CHOICE], [ESTIMATE] AND [LIMIT] IN THIS DERIVATION")
    for i, s in enumerate(CHOICES, 1):
        print(f"\n  {i}. {s}")

    hr("SUMMARY")
    print(f"""
  Apparent exponents (feedback-contaminated):
    Young 2021                         {step1['young']:.4f}
    Vahavihu (log-log OLS, 3 pts)      {step1['vahavihu_ols']:.4f}
    Vahavihu (endpoints)               {step1['vahavihu_endpoints']:.4f}

  Deconvolved exponents (feedback removed, biphasic default):
    Young 2021                         {e_young:.4f}
    Vahavihu (endpoints)               {e_vaha:.4f}
    ADOPTED (geometric mean)           {combined:.4f}

  Shipped BSA_EXP before this change: {OLD_BSA_EXP:.2f}
""")


if __name__ == "__main__":
    main()
