"""Dedicated test for the concavity assumption behind the spacing result.

WHY THIS FILE EXISTS
--------------------
``spacing_study/run_spacing_study.py`` states that the whole spacing
conclusion rests on the per-session synthesis kernel being **concave**, and
that a **convex** low-dose kernel "would invert [the ranking] immediately and
completely ... it is not a parameter, it is a functional form". Every other
sensitivity in that study is a parameter sweep that leaves the answer
standing. This one does not. So it gets a test of its own.

WHAT IT DOES -- and what it deliberately does not do
----------------------------------------------------
It does NOT assert that the kernel is concave. That would just restate the
assumption. It does three separable things:

1. **Bounds the curvature the ranking can absorb.** A one-parameter warp
   indexed by ``gamma`` is wrapped around the shipped kernel so that the
   low-dose response behaves as ``dose ** gamma``: ``gamma == 1`` is the
   shipped law, ``gamma < 1`` is more concave, ``gamma > 1`` is convex at low
   dose. The warp touches the SYNTHESIS DOSE ONLY -- melanin, stratum
   corneum, D-feedback, serum decay and the schedule are untouched, so what
   moves is the functional form and nothing else. Bisection finds
   ``gamma_flip``: the curvature at which daily dosing stops winning at a
   matched total-SED budget.

2. **Bounds the curvature the literature actually supports.** Four published
   datasets that varied PER-SESSION dose with the rest of the regimen matched
   are held in ``PER_SESSION_DOSE_SERIES`` below, with FST composition,
   source type and dose range recorded. Each yields an empirical log-log
   (power-law) exponent, which is exactly the ``gamma`` of the warp family.

3. **Tests the named mechanism directly.** The stated failure mode is
   specifically "a threshold below which no useful previtamin-D3 is made".
   A hard-threshold family is swept to find the threshold dose at which the
   ranking inverts, and that is compared with the lowest per-session dose at
   which a positive 25(OH)D response has actually been measured in vivo.

WHAT IT FOUND -- read this before citing the file
--------------------------------------------------
The measured exponents STRADDLE linearity (0.27 to 1.25 on point estimates).
The evidence does **not** settle the sign of the low-dose curvature, and this
file must not be cited as having established concavity.

``gamma_flip`` is strongly budget-dependent, and that is the whole result:

  * At the spacing study's headline operating point -- a 1 SED/day matched
    budget -- the flip needs roughly triple-power convexity, far outside
    anything measured. The headline claim is safe by a wide margin.
  * At LOW budgets (<= 0.25 SED/day) the flip falls to near-linear curvature,
    and the point estimate from Shih 2018 -- the only acute, all-FST,
    sub-erythemal series, and the one least confounded by photoadaptation --
    sits AT OR ABOVE it. At those budgets the daily-is-best ranking is not
    protected by the data.

``test_low_budget_flip_is_reachable_by_measured_exponents`` pins that
exposure so it cannot be quietly forgotten. This test suite is **bounding,
not resolving**.

PARAMETER VALUES: nothing here restates a scientific parameter. Canonical
sources are the ``vitd_model.py`` module docstring and ``vitd_scenario.py``.
Nothing here is a calibration; no shipped value is changed.

Run as a script for the full diagnostic table:

    PYTHONIOENCODING=utf-8 python3 tests/test_concavity_assumption.py
"""

from __future__ import annotations

import functools
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_SPACING = os.path.join(_REPO, "spacing_study")
if _SPACING not in sys.path:
    sys.path.insert(0, _SPACING)

import vitd_model as vm  # noqa: E402
import spacing_model as sm  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The empirical evidence, digitised from the published tables.
# ---------------------------------------------------------------------------
#
# INCLUSION RULE: a study qualifies only if it varied the *per-session* dose
# while holding the number of sessions, the interval, the exposed BSA and the
# source fixed. That is the only design that measures the shape of the
# per-session kernel. Studies that varied only the *number* of equal sessions
# (Young 2021) measure accumulation, not kernel shape, and are recorded in
# ACCUMULATION_EVIDENCE instead.
#
# Doses are per-session; responses are published group means of the change in
# serum 25(OH)D, nmol/L.

PER_SESSION_DOSE_SERIES: Dict[str, dict] = {
    "bogh2010": dict(
        citation="Bogh MKB, Schmedes AV, Philipsen PA, Thieden E, Wulf HC. "
                 "Exp Dermatol 2011;20:14-18 (online 2010), "
                 "doi:10.1111/j.1600-0625.2010.01201.x",
        fst="I-IV (5 / 18 / 21 / 11 by self-reported Fitzpatrick type); all "
            "fair-skinned. No FST V-VI.",
        source="lamp",
        source_detail="broadband UVB, Philips TL-12 bank",
        design="4 sessions, 2-3 d apart, 24% BSA, baseline-adjusted",
        # Only the two extreme group means appear numerically in the Results
        # text; the intermediate groups are shown in Fig. 1 only.
        doses_sed=[0.375, 3.0],
        delta_nmol_l=[14.2, 24.8],
        curvature="strongly concave on the secant; LINEAR on the authors' "
                  "own fit",
        note="The authors' own fit is linear in total dose with a large "
             "dose-independent intercept (Y = 3.5X + 14.3, R^2 = 0.176, X in "
             "total SED). The secant exponent is therefore as much an "
             "artefact of that intercept as a measure of curvature: the "
             "fitted MARGINAL response is constant, i.e. locally gamma = 1. "
             "Both readings matter, and they disagree.",
        legibility="Fig. 1 is a scatter with a fitted line; the four group "
                   "means are not tabulated, so only the two extremes quoted "
                   "in the text are used. Digitising Fig. 1 would add two "
                   "interior points -- FLAGGED, not attempted.",
    ),
    "farrar2013": dict(
        citation="Farrar MD, Webb AR, Kift R, Durkin MT, Allan D, Herbert A, "
                 "Berry JL, Rhodes LE. Am J Clin Nutr 2013;97:1210-1216, "
                 "doi:10.3945/ajcn.112.052639",
        fst="V only (n = 51 completing; UK South Asian adults). The largest "
            "dark-skin per-session dose-response series available here.",
        source="lamp",
        source_detail="solar-simulated (Arimed B + Cleo Natural)",
        design="18 sessions (3x/wk x 6 wk), 35% BSA",
        doses_sed=[0.65, 1.3, 1.95, 2.6, 3.25, 3.9],
        # Table 2 'Rise' column, ng/mL -> nmol/L at the paper's own x2.5.
        delta_nmol_l=[3.6 * 2.5, 4.6 * 2.5, 6.7 * 2.5,
                      6.7 * 2.5, 12.6 * 2.5, 9.5 * 2.5],
        curvature="concave",
        note="Non-monotone at the top: the 3.9 SED group rose LESS than the "
             "3.25 SED group. The authors attribute that to treatment-induced "
             "pigmentation and photodegradation -- mechanisms this model "
             "carries separately. So this series measures kernel shape "
             "CONFOUNDED with photoadaptation and D-feedback, and will read "
             "more concave than the bare kernel does.",
    ),
    "shih2018": dict(
        citation="Shih BB, Farrar MD, Cooke MS, ... Rhodes LE. J Invest "
                 "Dermatol 2018;138:2244-2252, doi:10.1016/j.jid.2018.04.015",
        fst="I-VI, all six types (n = 33-38 per dose level). The only "
            "all-FST sub-erythemal dose-response here, and the only one that "
            "reports FST V-VI alongside I-II under matched dosing.",
        source="lamp",
        source_detail="solar-simulated",
        design="ACUTE single exposure per dose level, 20/40/60/80% of each "
               "volunteer's own MED; 25(OH)D returned to baseline between "
               "doses, so no photoadaptation or plateau confound.",
        # Dose axis is fraction of personal MED, not absolute SED. A
        # power-law exponent is invariant under a per-subject rescaling of
        # dose, so gamma remains comparable; absolute-SED equivalents span
        # 0.42-1.68 (FST I) up to 3.04-12.16 (FST VI).
        doses_sed=[0.2, 0.4, 0.6, 0.8],
        dose_unit="fraction of personal MED",
        delta_nmol_l=[1.2, 3.3, 5.6, 6.4],
        sem_nmol_l=[0.5, 0.6, 0.5, 0.7],
        curvature="CONVEX on the raw group means; linear on the authors' fit",
        note="The cleanest available measurement of the bare kernel, and it "
             "reads convex. The authors' own mixed-effects fit is linear "
             "(1.6 nmol/L per 20% MED increment, 95% CI 1.1-2.0) and they "
             "state explicitly: 'There was no evidence of a minimum threshold "
             "dose for 25(OH)D gain or CPD induction.' The SEMs are large "
             "relative to the lowest mean (1.2 +/- 0.5), so the exponent is "
             "poorly determined -- see the +/-1 SEM envelope, which spans "
             "concave to strongly convex.",
    ),
    "jager2018": dict(
        citation="Jager N, Schoepe J, Wagenpfeil S, Bocionek P, Saternus R, "
                 "Vogt T, Reichrath J. Anticancer Res 2018;38:1165-1171, "
                 "Table II (meta-analysis)",
        fst="NOT REPORTED -- the authors name skin type as an unaddressed "
            "confounder. The underlying studies are Danish (Bogh), so FST "
            "I-IV is the likely composition. NOT dark-skin evidence.",
        source="lamp",
        source_detail="broadband UVB",
        design="pooled across studies: 4 sessions, 6% BSA",
        doses_sed=[0.75, 1.5, 3.0],
        delta_nmol_l=[6.63, 16.24, 29.4],
        curvature="convex below 1.5 SED, concave above",
        note="Per-SED efficiency is non-monotone (2.21, 2.70, 2.45 "
             "nmol/L/SED), peaking at 1.5 SED. That is locally CONVEX below "
             "1.5 SED. Pooled across studies, so between-study heterogeneity "
             "is not separable from curvature; treat as weak.",
    ),
}

#: Studies that varied the NUMBER of equal sessions, not the session size.
#: They bound accumulation and the presence of a plateau; they do not measure
#: per-session kernel curvature and are not fitted here.
ACCUMULATION_EVIDENCE: Dict[str, dict] = {
    "young2021": dict(
        citation="Young AR, Morgan KA, Harrison GI, Lawrence KP, Petersen B, "
                 "Wulf HC, Philipsen PA. PNAS 2021;118(40):e2015867118, "
                 "doi:10.1073/pnas.2015867118",
        fst="I-II only (n = 75).",
        source="lamp",
        source_detail="five polychromatic spectra (UV6, Arimed B, PUVA, "
                      "Xe-arc solar simulator with two filters)",
        design="5 serial exposures of 2 SED, 3-4 d apart (10 SED "
               "cumulative); 3.7% BSA and 85% BSA arms",
        finding="Per-individual model selection over {linear, logarithm, "
                "inverse, power, exponential}: 28 linear, 18 EXPONENTIAL, 10 "
                "power, 1 inverse, 18 non-significant. Linear was the best "
                "pooled fit; all intercepts were not significantly different "
                "from zero (P > 0.25); 'Linear regression was chosen because "
                "there is no evidence of a plateau over the dose range "
                "studied.' 18 of the 57 individuals with a significant fit "
                "ACCELERATED with cumulative dose -- the strongest in-vivo "
                "hint of convexity in this evidence base. But an accelerating "
                "ACCUMULATION curve is also what a hepatic 25-hydroxylation "
                "transit lag produces, and the model carries that separately "
                "(use_transit_delay), so this is not direct kernel evidence.",
    ),
    "webb2021": dict(
        citation="Webb AR, Alghamdi R, Kift R, Rhodes LE. Endocr Connect "
                 "2021;10(10):R248-R266, doi:10.1530/EC-21-0308",
        fst="Predominantly I-IV. The review states its dose-response 'is "
            "only applicable to white-skinned individuals'; studies of FST "
            "V-VI were 'rare'.",
        source="mixed",
        source_detail="21 quality-assured artificial-source studies plus 13 "
                      "solar studies",
        design="systematic review; dose-response fitted on CUMULATIVE "
               "standard vitamin D dose for whole-body exposures",
        finding="The consolidated relation is logarithmic in CUMULATIVE "
                "dose. Consistent message #1 of the review: 'Low dose, "
                "sub-erythemal doses are more effective for vitamin D "
                "synthesis than doses close to a personal minimum erythema "
                "dose. Short, frequent (e.g. daily or several times per week) "
                "exposures maximise vitamin D'. Reports of Bogh 2010 that "
                "'increasing dose by a factor of 8 less than doubled the "
                "vitamin D response'. This is the spacing conclusion stated "
                "as a review finding -- but it is a synthesis of the same "
                "primary studies fitted above, NOT independent evidence.",
    ),
}

#: Lowest per-session dose at which a significant positive 25(OH)D response
#: has been measured in vivo, absolute SED. Bogh 2010 reached 0.375 SED;
#: Shih 2018 reached 0.42 SED (FST I at 20% MED). The smaller binds.
LOWEST_DOSE_WITH_MEASURED_RESPONSE_SED: float = 0.375

#: The spacing study's headline operating point: experiment B reports its
#: budget-framing conclusion at this matched total-SED budget.
HEADLINE_BUDGET_SED_PER_DAY: float = 1.0


def loglog_exponent(doses: Sequence[float],
                    responses: Sequence[float]) -> float:
    """OLS slope of log(response) on log(dose): the power-law exponent.

    This is exactly the ``gamma`` of the warp family below. A response
    behaving as ``dose ** gamma`` returns ``gamma``. Concave => < 1,
    linear => 1, convex => > 1.
    """
    x = np.log(np.asarray(doses, dtype=float))
    y = np.log(np.asarray(responses, dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def secant_exponent(doses: Sequence[float],
                    responses: Sequence[float]) -> float:
    """Power-law exponent through the two endpoint doses only."""
    return float(math.log(responses[-1] / responses[0])
                 / math.log(doses[-1] / doses[0]))


def empirical_exponents() -> Dict[str, Dict[str, float]]:
    """Log-log and secant exponents for every per-session dose series."""
    out: Dict[str, Dict[str, float]] = {}
    for key, s in PER_SESSION_DOSE_SERIES.items():
        d, r = s["doses_sed"], s["delta_nmol_l"]
        rec = {"loglog": loglog_exponent(d, r),
               "secant": secant_exponent(d, r)}
        sem = s.get("sem_nmol_l")
        if sem:
            # Extremal readings within +/-1 SEM: push the lowest-dose mean
            # down and the highest-dose mean up for the most convex reading,
            # and the reverse for the most concave.
            conv = list(r)
            conc = list(r)
            conv[0], conv[-1] = r[0] - sem[0], r[-1] + sem[-1]
            conc[0], conc[-1] = r[0] + sem[0], r[-1] - sem[-1]
            rec["secant_most_convex"] = secant_exponent(d, conv)
            rec["secant_most_concave"] = secant_exponent(d, conc)
        out[key] = rec
    return out


def most_convex_measured() -> float:
    """Largest exponent obtainable from any series on any reading."""
    return max(max(rec.values()) for rec in empirical_exponents().values())


def most_convex_point_estimate() -> float:
    """Largest exponent from POINT estimates only (no SEM envelope)."""
    return max(max(rec["loglog"], rec["secant"])
               for rec in empirical_exponents().values())


# ---------------------------------------------------------------------------
# 2. The warp family: one knob on the SHAPE of the kernel, nothing else.
# ---------------------------------------------------------------------------

_ORIG_SYNTHESIS = vm.synthesis_per_session


def _effective_E(params, kwargs) -> float:
    E = kwargs.get("E_source_override")
    return params.E_source if E is None else E


def _power_warp(gamma: float):
    """Replace effective dose ``u = sed * E`` by ``u ** gamma`` in the kernel.

    ``gamma == 1`` reproduces the shipped kernel exactly. The warp touches
    ONLY the dose entering the synthesis term: the raw session SED still
    drives melanin, stratum corneum and the safety sidecar inside
    ``vm.simulate``, which read the schedule directly.
    """
    def warped(sed, M, M_baseline, D, SC, params, **kw):
        if sed <= 0.0:
            return 0.0
        E = _effective_E(params, kw)
        return _ORIG_SYNTHESIS(((sed * E) ** gamma) / E,
                               M, M_baseline, D, SC, params, **kw)
    return warped


def _threshold_warp(s0: float):
    """Hard low-dose threshold: no synthesis at all below ``s0`` SED.

    This is the exact mechanism ``run_spacing_study.py`` names as the one
    that would invert the ranking -- "a threshold below which no useful
    previtamin-D3 is made".
    """
    def warped(sed, M, M_baseline, D, SC, params, **kw):
        if sed < s0:
            return 0.0
        return _ORIG_SYNTHESIS(sed, M, M_baseline, D, SC, params, **kw)
    return warped


#: Intervals compared. 1 d is the incumbent answer under every framing.
FLIP_INTERVALS: Tuple[int, ...] = (1, 2, 3, 4, 7)
#: Reduced set used inside the bisection loop, for speed. The value returned
#: is re-checked against the full set by ``flip_is_confirmed``.
BISECT_INTERVALS: Tuple[int, ...] = (1, 2, 3, 7)

#: Total-SED budget levels, SED per day of horizon. Mirrors the spacing
#: study's own experiment-B budget grid.
BUDGETS_SED_PER_DAY: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)

#: FSTs carried. V and VI first -- dark-skin priority. II is the light-skin
#: comparator and is labelled as such wherever it is reported.
FSTS: Tuple[int, ...] = (5, 6, 2)

#: Bisection tolerance on gamma. The reported value is stable to the printed
#: digits at this tolerance; it is not a measurement precision.
GAMMA_TOL = 1.0e-4
#: Cap on the gamma search. Beyond this the "kernel" resembles no published
#: dose-response and the question is moot.
GAMMA_CAP = 16.0


def _ss_mean_by_interval(fst: int, budget_per_day: float,
                         warp_factory, knob: float,
                         intervals: Sequence[int] = FLIP_INTERVALS,
                         horizon: int = sm.HORIZON_SS) -> Dict[int, float]:
    """Steady-state mean 25(OH)D per interval at a matched total-SED budget."""
    vm.synthesis_per_session = warp_factory(knob)
    try:
        p = sm.make_params()
        return {iv: sm.run_schedule(
                    fst,
                    sm.budget_schedule(horizon, budget_per_day * horizon, iv),
                    horizon, p,
                    dose_framing="fixed_cumulative_SED",
                    period_days=iv)["ss_mean_D"]
                for iv in intervals}
    finally:
        vm.synthesis_per_session = _ORIG_SYNTHESIS


def daily_beaten(fst: int, budget_per_day: float, warp_factory, knob: float,
                 intervals: Sequence[int] = BISECT_INTERVALS) -> bool:
    """True when some interval longer than 1 d beats daily on ss_mean_D."""
    vals = _ss_mean_by_interval(fst, budget_per_day, warp_factory, knob,
                                intervals)
    daily = vals[1]
    return any(v > daily for iv, v in vals.items() if iv != 1)


@functools.lru_cache(maxsize=None)
def find_flip(fst: int, budget_per_day: float,
              cap: float = GAMMA_CAP, tol: float = GAMMA_TOL) -> float:
    """Smallest ``gamma`` at which daily stops winning. NaN if none <= cap.

    Bisection on a predicate that is monotone in practice across this grid
    (verified: daily's margin shrinks monotonically in gamma at every cell).
    """
    lo = 1.0
    if daily_beaten(fst, budget_per_day, _power_warp, lo):
        return lo
    hi, step = lo, 0.5
    while hi < cap:
        hi = min(cap, hi + step)
        if daily_beaten(fst, budget_per_day, _power_warp, hi):
            break
        step *= 2.0
    if not daily_beaten(fst, budget_per_day, _power_warp, hi):
        return float("nan")
    a, b = lo, hi
    while b - a > tol:
        m = 0.5 * (a + b)
        if daily_beaten(fst, budget_per_day, _power_warp, m):
            b = m
        else:
            a = m
    return b


def flip_is_confirmed(fst: int, budget_per_day: float, gamma: float) -> bool:
    """Re-check a bisected flip against the FULL interval set."""
    return daily_beaten(fst, budget_per_day, _power_warp, gamma,
                        FLIP_INTERVALS)


@functools.lru_cache(maxsize=None)
def find_threshold_flip(fst: int, budget_per_day: float,
                        tol: float = 1.0e-5) -> float:
    """Smallest hard-threshold dose (SED) at which daily stops winning."""
    lo, hi = budget_per_day * 0.25, budget_per_day * 8.0
    if not daily_beaten(fst, budget_per_day, _threshold_warp, hi):
        return float("nan")
    if daily_beaten(fst, budget_per_day, _threshold_warp, lo):
        return lo
    while hi - lo > tol:
        m = 0.5 * (lo + hi)
        if daily_beaten(fst, budget_per_day, _threshold_warp, m):
            hi = m
        else:
            lo = m
    return hi


# ---------------------------------------------------------------------------
# 3. Analytic anchor -- exact, no simulation.
# ---------------------------------------------------------------------------

def analytic_flip_gamma(c: float) -> float:
    """Closed-form flip exponent for the bare kernel, no other mechanism.

    With per-day budget ``c`` and effective dose ``u = c * E`` (E = 1 at the
    solar reference), total synthesis per two days is ``2*ln(1 + u**g)``
    daily and ``ln(1 + (2u)**g)`` every-2-days. Equal when

        2**g == 2 + u**g

    which is the exact curvature at which the bare kernel is indifferent to
    splitting. At ``c == 1`` this is ``log2(3)``.
    """
    def h(g: float) -> float:
        return 2.0 ** g - 2.0 - c ** g
    a, b = 1.0, 64.0
    if h(a) >= 0.0:
        return a
    if h(b) < 0.0:
        return float("nan")
    while b - a > 1e-12:
        m = 0.5 * (a + b)
        if h(m) >= 0.0:
            b = m
        else:
            a = m
    return b


# ---------------------------------------------------------------------------
# 4. Tests
# ---------------------------------------------------------------------------

def test_warp_is_identity_at_gamma_one():
    """gamma = 1 must reproduce the shipped kernel exactly."""
    p = sm.make_params()
    Mb = p.M_baseline(5)
    w = _power_warp(1.0)
    for sed in (0.1, 0.375, 1.0, 3.9, 12.0):
        a = _ORIG_SYNTHESIS(sed, Mb, Mb, 40.0, 0.0, p)
        b = w(sed, Mb, Mb, 40.0, 0.0, p)
        assert a == pytest.approx(b, rel=1e-12, abs=1e-15)


def test_shipped_law_ranks_daily_first_everywhere():
    """The incumbent claim, re-derived here rather than assumed."""
    for fst in FSTS:
        for c in BUDGETS_SED_PER_DAY:
            vals = _ss_mean_by_interval(fst, c, _power_warp, 1.0)
            assert max(vals, key=vals.get) == 1, (
                f"daily is not optimal at FST {fst}, budget {c} SED/d: {vals}")


def test_analytic_flip_matches_known_closed_form():
    """At a 1 SED/d budget the bare-kernel flip is exactly log2(3)."""
    assert analytic_flip_gamma(1.0) == pytest.approx(
        math.log(3.0) / math.log(2.0), abs=1e-9)


def test_flip_exponents_exceed_one_everywhere_they_exist():
    """The shipped law must never sit at or past its own inversion point."""
    found_any = False
    for fst in FSTS:
        for c in BUDGETS_SED_PER_DAY:
            g = find_flip(fst, c)
            if math.isnan(g):
                continue
            found_any = True
            assert g > 1.0, (
                f"FST {fst}, budget {c}: flip at gamma = {g}, i.e. the "
                f"shipped law is already at or past the inversion point")
            assert flip_is_confirmed(fst, c, g), (
                f"FST {fst}, budget {c}: bisected flip {g} not reproduced "
                f"against the full interval set {FLIP_INTERVALS}")
    assert found_any, "no flip found anywhere in the grid -- bisection broken?"


def test_headline_budget_is_protected_by_a_wide_margin():
    """At the spacing study's headline 1 SED/day budget the claim is safe.

    The flip there needs curvature roughly triple the linear exponent, which
    is far outside every measured series including the most convex +/-1 SEM
    reading. This is the assertion that actually defends the paper's headline.
    """
    worst = most_convex_measured()
    for fst in FSTS:
        g = find_flip(fst, HEADLINE_BUDGET_SED_PER_DAY)
        assert not math.isnan(g), f"no flip at FST {fst}, headline budget"
        assert g > 3.0, (
            f"FST {fst}: headline-budget flip has fallen to gamma = {g}")
        assert g - worst > 1.5, (
            f"FST {fst}: margin over the most convex measured reading "
            f"({worst}) has narrowed to {g - worst}")


def test_concave_side_series_never_flip_anywhere_in_the_grid():
    """Bogh 2010 and Farrar 2013 -- the two concave readings -- are safe.

    Farrar 2013 is the FST V series, so this is the dark-skin-priority check.
    """
    exps = empirical_exponents()
    flips = [find_flip(fst, c) for fst in FSTS for c in BUDGETS_SED_PER_DAY]
    lowest = min(g for g in flips if not math.isnan(g))
    for key in ("bogh2010", "farrar2013"):
        worst = max(exps[key].values())
        assert worst < lowest, (
            f"{key}: exponent {worst} now reaches the lowest flip {lowest}")


def test_low_budget_flip_is_reachable_by_measured_exponents():
    """THE EXPOSURE, pinned so it cannot be quietly forgotten.

    At budgets of 0.25 SED/day and below, the flip exponent falls close
    enough to linearity that the point estimate from Shih 2018 -- the only
    acute, all-FST, sub-erythemal series, and the one least confounded by
    photoadaptation -- reaches or exceeds it. The daily-is-best ranking is
    NOT protected by the data at those budgets.

    This test asserts that the exposure is real. If a future change makes it
    fail, that means the exposure has closed and the wording in
    ``docs/workstreams/W-H.md`` and in the spacing study should be revisited.
    """
    shih = empirical_exponents()["shih2018"]
    point = max(shih["loglog"], shih["secant"])
    low_budget_flips = [find_flip(fst, c)
                        for fst in FSTS for c in (0.1, 0.25)]
    assert all(not math.isnan(g) for g in low_budget_flips)
    assert min(low_budget_flips) < point, (
        "the low-budget flip no longer sits below the Shih 2018 point "
        "estimate; the documented exposure has changed and the memo is stale")


def test_threshold_only_inverts_below_the_lowest_measured_dose():
    """The named failure mode, bounded against the data that constrain it.

    A hard threshold has to exceed the daily session size before it can
    invert the ranking, and at a matched budget the daily session size IS the
    budget. So the threshold mechanism is empirically excluded for every
    budget at or above the lowest dose with a measured positive response
    (0.375 SED: Bogh 2010; Shih 2018 reaches 0.42 SED across FST I-VI and
    reports no threshold). Below that budget the data are silent.
    """
    for c in (0.5, 1.0, 2.0):
        s0 = find_threshold_flip(5, c)
        assert not math.isnan(s0), f"no threshold flip found at budget {c}"
        assert s0 > LOWEST_DOSE_WITH_MEASURED_RESPONSE_SED, (
            f"budget {c}: ranking inverts at a threshold of {s0} SED, at or "
            f"below a dose with a measured positive response")
        # And the threshold that does it is essentially the session size.
        assert s0 == pytest.approx(c, rel=0.05), (
            f"budget {c}: threshold flip at {s0} SED is not the session size")


def test_empirical_exponents_straddle_linearity():
    """The honest finding, pinned so it cannot be misread.

    The evidence does NOT agree on the sign of the low-dose curvature: at
    least one series reads concave and at least one reads convex. This test
    exists so no future reader can cite this file as having established
    concavity. It has not.
    """
    loglog = {k: v["loglog"] for k, v in empirical_exponents().items()}
    assert min(loglog.values()) < 1.0, loglog
    assert max(loglog.values()) > 1.0, loglog


def test_every_series_records_its_fst_and_source():
    """Provenance guard: no series may be used without FST and source."""
    for key, s in list(PER_SESSION_DOSE_SERIES.items()) + \
            list(ACCUMULATION_EVIDENCE.items()):
        for field in ("citation", "fst", "source", "design"):
            assert s.get(field), f"{key} is missing {field}"
        assert s["source"] in ("lamp", "sunlight", "mixed"), key


# ---------------------------------------------------------------------------
# 5. Diagnostic report (script mode)
# ---------------------------------------------------------------------------

def _report() -> str:
    lines: List[str] = []
    w = lines.append
    exps = empirical_exponents()

    w("=" * 78)
    w("CONCAVITY ASSUMPTION -- bounding test")
    w("=" * 78)
    w("")
    w("A. Empirical exponents from per-session dose series")
    w("   gamma < 1 concave | gamma = 1 linear | gamma > 1 convex at low dose")
    w("")
    w(f"   {'series':<11}{'log-log':>9}{'secant':>9}  {'FST':<10}{'src':<9}dose range")
    for key, rec in exps.items():
        s = PER_SESSION_DOSE_SERIES[key]
        unit = s.get("dose_unit", "SED")
        rng = f"{min(s['doses_sed']):g}-{max(s['doses_sed']):g} {unit}"
        fst_short = s["fst"].split("(")[0].split("--")[0].strip().rstrip(".,;")
        w(f"   {key:<11}{rec['loglog']:>9.4f}{rec['secant']:>9.4f}  "
          f"{fst_short:<10}{s['source']:<9}{rng}")
        if "secant_most_convex" in rec:
            w(f"   {'':<11}+/-1 SEM envelope on the secant: "
              f"{rec['secant_most_concave']:.4f} (most concave) .. "
              f"{rec['secant_most_convex']:.4f} (most convex)")
    w("")
    w("B. Analytic flip -- bare kernel alone, every other mechanism removed")
    w("")
    w(f"   {'budget SED/d':>13}{'gamma_flip':>13}")
    for c in BUDGETS_SED_PER_DAY:
        g = analytic_flip_gamma(c)
        w(f"   {c:>13g}{g:>13.6f}" if not math.isnan(g)
          else f"   {c:>13g}{'-':>13}")
    w("")
    w("C. Simulated flip -- full model, warp on the synthesis dose only")
    w(f"   bisection tolerance {GAMMA_TOL:g}; cap {GAMMA_CAP:g}; "
      f"'-' = no flip at or below the cap")
    w("")
    w("   " + f"{'budget SED/d':>13}"
      + "".join(f"{'FST ' + str(f):>12}" for f in FSTS))
    for c in BUDGETS_SED_PER_DAY:
        row = f"   {c:>13g}"
        for fst in FSTS:
            g = find_flip(fst, c)
            row += f"{'-':>12}" if math.isnan(g) else f"{g:>12.4f}"
        w(row)
    w("")
    w("D. Hard-threshold family -- the mechanism the spacing study names")
    w(f"   Lowest per-session dose with a measured positive response: "
      f"{LOWEST_DOSE_WITH_MEASURED_RESPONSE_SED:g} SED (Bogh 2010)")
    w("")
    w(f"   {'budget SED/d':>13}{'threshold SED at flip':>24}"
      f"{'vs lowest measured':>21}")
    for c in BUDGETS_SED_PER_DAY:
        s0 = find_threshold_flip(5, c)
        if math.isnan(s0):
            w(f"   {c:>13g}{'-':>24}{'-':>21}")
        else:
            verdict = ("EXCLUDED by data"
                       if s0 > LOWEST_DOSE_WITH_MEASURED_RESPONSE_SED
                       else "NOT excluded")
            w(f"   {c:>13g}{s0:>24.4f}{verdict:>21}")
    w("")
    w("E. Verdict")
    w("")
    point = most_convex_point_estimate()
    envelope = most_convex_measured()
    concave = min(min(r.values()) for r in exps.values())
    flips = {(fst, c): find_flip(fst, c)
             for fst in FSTS for c in BUDGETS_SED_PER_DAY}
    live = [g for g in flips.values() if not math.isnan(g)]
    head = min(find_flip(f, HEADLINE_BUDGET_SED_PER_DAY) for f in FSTS)
    w(f"   measured exponents, point estimates   "
      f"{concave:.4f} .. {point:.4f}")
    w(f"   most convex reading incl. +/-1 SEM    {envelope:.4f}")
    w(f"   flip at the headline 1 SED/d budget   {head:.4f}"
      f"   -> margin {head - envelope:+.4f}")
    w(f"   lowest flip anywhere in the grid      {min(live):.4f}"
      f"   -> margin {min(live) - point:+.4f}")
    w("")
    w("   The measured exponents STRADDLE linearity, so the evidence does")
    w("   not settle the sign of the low-dose curvature. Two things it does")
    w("   settle:")
    w("     (i)  no threshold exists above 0.375 SED, which removes the")
    w("          specific mechanism the spacing study names as fatal for")
    w("          every budget at or above that dose;")
    w("     (ii) at the headline 1 SED/day budget the flip is far outside")
    w("          anything measured, so the headline claim is safe.")
    w("   What it does NOT settle: at budgets of 0.25 SED/day and below the")
    w("   flip falls to near-linear curvature and the Shih 2018 point")
    w("   estimate reaches it. This test BOUNDS the risk; it does not")
    w("   RESOLVE the functional form.")
    w("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_report())
