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
    unlike the feedback-driven artifact documented in OPEN_QUESTIONS gap 19.
    """
    if sed <= 0.0:
        return 1.0
    x = sed / d_sat
    if x < 1e-8:
        return 1.0
    return float((1.0 - np.exp(-x)) / x)


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
#
# Form: smooth Hill-type efficiency, ~1 below the inflection and decaying above.
#     eff(D) = 1 / (1 + (D / D_inflect) ** n)
# n = 2 gives a gentle knee (eff = 0.5 at the inflection, 0.8 at 0.5x, 0.2 at 2x).
# [PRIOR] n is not reported by the source; only the inflection location is.
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
# Sources: Elwood & Jopson 1997 meta-analysis of 29 case-control studies
# (intermittent exposure OR = 1.71 for melanoma; heavy OCCUPATIONAL exposure
# OR = 0.86, i.e. PROTECTIVE; sunburn OR = 1.91); Leiter 2009; Calzavara-Pinton
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
INTERMITTENT_BURN_EXPONENT: float = 3.0
REFERENCE_ANNUAL_SED: float = 52.0


def _med_sed_by_fst(fst: int) -> float:
    """Minimal erythema dose in SED, Shih/Farrar/Rhodes 2018 (P24)."""
    return {1: 2.1, 2: 2.6, 3: 3.2, 4: 5.6, 5: 7.5, 6: 15.2}.get(fst, 3.5)


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

    Both indices scale with FST susceptibility: melanoma and NMSC incidence are
    roughly an order of magnitude lower in FST V-VI than FST I-II, and basal
    keratinocytes in darker skin are substantially protected from CPD
    (Shih/Rhodes 2018). Modelled as a factor falling with FST.
    """
    seds = np.asarray([s for s in session_seds if s > 0], dtype=float)
    med = _med_sed_by_fst(fst)

    # Constitutive susceptibility. FST I = 1.0 down to FST VI ~ 0.1.
    # Order-of-magnitude only; matches the ~10x incidence gap between
    # very fair and deeply pigmented populations.
    susceptibility = float(np.exp(-0.45 * (fst - 1)))

    if seds.size == 0:
        return {"scc_index": 0.0, "intermittent_index": 0.0,
                "burn_fraction": 0.0, "total_sed": 0.0,
                "combined_index": 0.0, "susceptibility": susceptibility,
                "med_sed": med}

    total_sed = float(seds.sum())

    # SCC: linear in cumulative dose.
    scc = SCC_CUMULATIVE_WEIGHT * susceptibility * total_sed / REFERENCE_ANNUAL_SED

    # BCC / melanoma: super-linear in per-session intensity relative to MED, so
    # that one big session costs far more than the same dose split up.
    # (sed/med)^3 summed over sessions. At fixed total dose this is minimised by
    # many small sessions -- the Elwood/Leiter intermittency finding.
    frac_med = seds / med
    intermittent = float(susceptibility * np.sum(frac_med ** INTERMITTENT_BURN_EXPONENT))
    # Normalise by the reference schedule (1 SED/wk for a year) for that FST.
    ref_intermittent = susceptibility * 52.0 * (1.0 / med) ** INTERMITTENT_BURN_EXPONENT
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
