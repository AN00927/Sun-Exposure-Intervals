"""Fit the melanin sub-model's kinetic parameters to published pigmentation data.

WHAT THIS SCRIPT IS FOR
-----------------------
Until now `alpha`, `alpha_by_FST`, `alpha_max`, `UV_halfmax_by_FST` and `beta`
had no derivation recorded anywhere in this repository. `docs/MELANIN_MODEL.md`
sec.2.1 states plainly that the arithmetic from measurement to parameter was
"not recorded anywhere, there is no derivation script for alpha, no fit, and no
stated mapping"; sec.2.4 records `beta` as a *choice* ("a mid-range estimate
compatible with Ravnbak 2008 if complete fading requires ~3 e-folding times"),
not a fit. This script supplies the missing derivation.

THE THREE SOURCES
-----------------
R08  Ravnbak MH, Philipsen PA, Wulf HC. "Skin pigmentation kinetics after UVB
     exposure." Acta Derm Venereol 2008; 88:223-228. PMID 18480919.
     24 volunteers (12 Scandinavian, 12 Indian/Pakistani), FST II-V.
     Solar simulator + narrowband UVB. 6 or 12 exposures at 0.8/0.6/0.4/0.2 of
     the individually predetermined single-exposure MMD.

R09  Ravnbak MH, Philipsen PA, Wulf HC. "Skin pigmentation kinetics after
     exposure to ultraviolet A." Acta Derm Venereol 2009; 89:357-363.
     The SAME 24-volunteer cohort, broadband UVA and UVA1 arms.

M11  Miyamura Y, Coelho SG, Schlenz K, Batzer J, Smuda C, Choi W, Brenner M,
     Passeron T, Zhang G, Kolbe L, Wolber R, Hearing VJ. "The deceptive nature
     of UVA tanning versus the modest protective effects of UVB tanning on
     human skin." Pigment Cell Melanoma Res 2011; 24(1):136-147.
     doi:10.1111/j.1755-148X.2010.00764.x  PMID 20979596  PMC3021652.
     7 volunteers, FST II-III, 10 suberythemal exposures over 2 weeks.

WHY R09 (UVA) IS NOT POOLED INTO THE alpha FIT -- the central synthesis point
----------------------------------------------------------------------------
R08 and R09 report closely comparable *pigmentation-%* increases for UVB and
UVA at matched MMD fractions. It is tempting to pool them: same cohort, same
instrument, same endpoint, twice the data.

That would be wrong, and M11 is the reason. M11 exposed skin to suberythemal
UVA and/or UVB for two weeks and then measured epidermal melanin content
directly by Fontana-Masson densitometry (M11 Table 1). UVB raised melanin
content from 9.03 to 12.70 a.u. (P<0.01). UVA did not raise it at all
(8.62 vs 9.03 control, not significant) -- yet UVA produced a visually and
instrumentally comparable tan, because the UVA doses were chosen precisely to
match the visible tanning response. M11's conclusion is that UVA tanning is
photooxidation and redistribution of melanin that already exists, and that it
confers essentially no photoprotection against DNA damage.

The model's M(t) is not "how dark the skin looks". It is the quantity that
attenuates UVB reaching 7-dehydrocholesterol. Fitting it to UVA-induced
reflectance change would credit the model with photoprotection that M11 shows
is not there. Therefore:

  * alpha is fitted to R08 (UVB / solar simulator) ONLY.
  * R09 (UVA) is a NEGATIVE CONTROL bounding how much of a reflectance-measured
    pigmentation-% increase is non-melanin optical change -- uva_negative_control().
  * M11 is an out-of-sample check on the fitted alpha -- check_miyamura().

Run:  PYTHONIOENCODING=utf-8 python3 scripts/fit_melanin_kinetics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import least_squares, brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vitd_model as vm  # noqa: E402


# ---------------------------------------------------------------------------
# 1. TRANSCRIBED DATA
# ---------------------------------------------------------------------------

# R08 Table I / R09 Table I -- pre-exposure pigmentation-% and single-exposure
# MMD in SED, per Fitzpatrick skin type. Same 24 volunteers in both papers,
# measured independently, so the two pigmentation-% columns double as a
# repeatability check on the instrument.
#   FST: (n, R08 pig%, R09 pig%, MMD_solar, MMD_nUVB, MMD_bUVA, MMD_UVA1)
BASELINE_BY_FST = {
    2: (5, 22.9, 23.5, 5.3, 5.2, 1.7, 0.8),
    3: (6, 26.7, 27.2, 7.5, 6.6, 1.8, 0.9),
    4: (4, 31.2, 30.7, 9.6, 8.3, 2.2, 0.9),
    5: (9, 46.3, 47.0, 10.5, 8.9, 2.4, 0.9),
}

# R08 Table II -- UVB arm (solar simulator + nUVB pooled, all skin types).
# (n_exposures, MMD fraction, n, mean pre-exposure pig%, absolute pig% increase,
#  percent increase)
R08_TABLE_II = [
    (6,  0.8, 26, 37.2, 7.6, 23.5),
    (6,  0.6, 36, 31.7, 6.9, 27.9),
    (6,  0.4, 27, 32.3, 6.9, 27.9),
    (12, 0.8, 21, 37.1, 9.1, 25.9),
    (12, 0.6, 33, 35.4, 9.0, 28.2),
    (12, 0.4, 41, 33.3, 7.2, 24.9),
]

# R09 Table III -- UVA arm (bUVA + UVA1 pooled), plus days to steady state.
R09_TABLE_III = [
    (6,  0.8, 41, 32.4, 7.7, 27.1, 15.0),
    (6,  0.6, 40, 32.6, 7.1, 25.6, 15.9),
    (6,  0.4, 34, 33.7, 6.8, 25.2, 17.3),
    (12, 0.8, 46, 34.2, 10.5, 33.9, 16.4),
    (12, 0.6, 47, 34.2, 9.2, 29.7, 17.9),
    (12, 0.4, 44, 34.9, 7.4, 24.6, 19.2),
]

# R08 exposure schedule: "The right side of the back was exposed on 2
# consecutive days per week and the left side was exposed on 4 consecutive days
# per week for 3 weeks; a total of 6 and 12 exposures, respectively."
R08_SCHEDULE = {6: "2 consecutive days/week x 3 weeks",
                12: "4 consecutive days/week x 3 weeks"}

# R08 / R09 fading. Days to TOTAL fading, i.e. the day pigmentation returns to
# pre-exposure baseline. CRITICAL METHODOLOGICAL DETAIL (R08 Methods): "Time to
# total fading was determined as the day when the pigmentation value was equal
# to pre-exposure pigmentation. This was determined by a linear regression
# model based on 2 or 3 points (pigmentation 1 week, 2 months and 4 months
# after the last UV exposure)." They fitted a STRAIGHT LINE to decaying data
# and extrapolated it to zero. invert_linear_extrapolated_fade() undoes that.
FADE_SAMPLE_DAYS = (7.0, 60.0, 120.0)
FADE_OBSERVED_UVB = {
    "nUVB, 6 exposures": 132.0,
    "solar, 6 exposures": 134.0,
    "nUVB, 12 exposures": 137.0,
    "solar, 12 exposures": 175.0,
}
FADE_OBSERVED_UVA = {  # reported for completeness; NOT used to fit beta
    "UVA, 6 exposures": 173.0,
    "UVA, 12 exposures": 185.0,
}

# M11 Table 1 -- epidermal melanin content (Fontana-Masson mean density, a.u.)
# and stratum corneum thickness (um), n=6, "without UV challenge" column.
M11_MELANIN = {"control": 9.03, "UVA": 8.62, "UVB": 12.70, "UVA+UVB": 12.50}
M11_MELANIN_SEM = {"control": 1.62, "UVA": 1.59, "UVB": 2.10, "UVA+UVB": 1.43}
M11_SC_UM = {"control": 16.6, "UVA": 19.5, "UVB": 20.7, "UVA+UVB": 20.0}
M11_SC_SEM = {"control": 2.1, "UVA": 2.9, "UVB": 3.6, "UVA+UVB": 2.3}
# M11 protocol: FST II-III, exposed 5x/week at 0.4 MED in week 1 and 5x/week at
# 0.5 MED in week 2 => 10 exposures, 4.5 MED cumulative, daily on weekdays.
M11_MED_PER_SESSION = [0.4] * 5 + [0.5] * 5
M11_INTERVALS = [1, 1, 1, 1, 3, 1, 1, 1, 1]  # Mon-Fri weeks, 3 d over weekend
# M11 does not report absolute doses, so MED->SED is carried as a range.
# 1 SED = 100 J/m^2 erythemally weighted (CIE/ISO 17166); unacclimatised FST II
# MED ~200-300 J/m^2, FST III ~250-350 J/m^2.
M11_SED_PER_MED_RANGE = (2.0, 3.5)


# ---------------------------------------------------------------------------
# 2. PIGMENTATION-% -> MODEL MELANIN INDEX MAP
# ---------------------------------------------------------------------------
# Ravnbak's endpoint is "pigmentation-%" from reflectance spectroscopy at
# 555/660 nm. The model's M is a 0-1 index anchored to the ITA of unexposed
# skin (vitd_model.py: M = 1 - (ITA_U + 30)/85). These are different nonlinear
# functions of the same underlying pigment, so a map has to be fitted. The four
# anchors are per-FST means: Ravnbak pre-exposure pigmentation-% against the
# model's own M_baseline_by_FST. Four points is few, so three candidate
# monotone forms are fitted and the spread carried as uncertainty.

PREF = 46.65  # the FST V anchor, held as the power-form scale reference


def anchor_points():
    params = vm.ModelParams()
    P, M = [], []
    for fst, row in BASELINE_BY_FST.items():
        P.append(0.5 * (row[1] + row[2]))  # same cohort, two readings
        M.append(params.M_baseline(fst))
    return np.array(P), np.array(M)


def fit_pigment_maps():
    """Return ({name: (forward, derivative)}, P_anchors, M_anchors)."""
    P, M = anchor_points()
    maps = {}

    a, b = np.polyfit(P, M, 1)
    maps["linear"] = (lambda p, a=a, b=b: a * np.asarray(p, float) + b,
                      lambda p, a=a: np.full(np.shape(p), a, float))

    c2, c1, c0 = np.polyfit(P, M, 2)
    maps["quadratic"] = (
        lambda p, c2=c2, c1=c1, c0=c0: c2 * np.asarray(p, float) ** 2 + c1 * np.asarray(p, float) + c0,
        lambda p, c2=c2, c1=c1: 2 * c2 * np.asarray(p, float) + c1)

    M_ref = M[-1]

    def resid(theta):
        P0, g = theta
        u = np.clip((P - P0) / (PREF - P0), 1e-12, None)
        return u ** g * M_ref - M

    sol = least_squares(resid, x0=[20.0, 1.0], bounds=([0.0, 0.2], [22.5, 6.0]))
    P0, g = sol.x

    def fwd(p, P0=P0, g=g, M_ref=M_ref):
        u = np.clip((np.asarray(p, float) - P0) / (PREF - P0), 0.0, None)
        return u ** g * M_ref

    def der(p, P0=P0, g=g, M_ref=M_ref):
        u = np.clip((np.asarray(p, float) - P0) / (PREF - P0), 1e-9, None)
        return M_ref * g * u ** (g - 1.0) / (PREF - P0)

    maps["power"] = (fwd, der)
    return maps, P, M


def mmd_sed_from_pigmentation(P_cell, source="uvb"):
    """Single-exposure MMD in SED for a cell of mean pre-exposure
    pigmentation-% P_cell. R08 Table II pools skin types, so the per-FST
    MMD-vs-pigmentation relation (R08/R09 Table I) is inverted to give an
    effective dose for the cell's actual composition. R09 reports this
    correlation as significant (p = 0.003)."""
    P = np.array([0.5 * (r[1] + r[2]) for r in BASELINE_BY_FST.values()])
    idx = (3, 4) if source == "uvb" else (5, 6)
    S = np.array([0.5 * (r[idx[0]] + r[idx[1]]) for r in BASELINE_BY_FST.values()])
    a, b = np.polyfit(P, S, 1)
    return a * np.asarray(P_cell, float) + b


# ---------------------------------------------------------------------------
# 3. THE MODEL'S OWN MELANIN PULSE UPDATE
# ---------------------------------------------------------------------------

def simulate_pulses(n_exposures, sed_per_session, M_base, M_ceil, alpha,
                    alpha_max, UV_halfmax, beta, intervals):
    """Reproduce vitd_model.simulate()'s melanin channel exactly: Hill-modulated
    alpha, saturating jump at each pulse, exponential relaxation between pulses.
    See vitd_model.py:1134-1141 (pulse) and :1101 (relaxation)."""
    M = M_base
    for i in range(n_exposures):
        sed = sed_per_session
        hill = sed ** 2 / (sed ** 2 + UV_halfmax ** 2)
        a_eff = alpha + (alpha_max - alpha) * hill
        M = M_ceil - (M_ceil - M) * np.exp(-a_eff * sed)
        if i < n_exposures - 1:
            gap = intervals[i] if i < len(intervals) else 1.0
            M = M_base + (M - M_base) * np.exp(-beta * gap)
    return M


def r08_intervals(n_exposures):
    """R08 dosing calendar. 6 exposures = 2 consecutive days/week x 3 weeks
    (gaps 1,6,1,6,1). 12 exposures = 4 consecutive days/week x 3 weeks
    (gaps 1,1,1,4,1,1,1,4,1,1,1)."""
    if n_exposures == 6:
        return [1, 6, 1, 6, 1]
    return [1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1]


# ---------------------------------------------------------------------------
# 4. FIT alpha / alpha_max / UV_halfmax TO R08 (UVB ONLY)
# ---------------------------------------------------------------------------

def fit_alpha(map_name="power", fit_headroom=False, beta=None):
    """Least-squares fit of the induction parameters to the six R08 Table II
    cells. Returns a dict of fitted values and diagnostics."""
    maps, _, _ = fit_pigment_maps()
    fwd, der = maps[map_name]
    params = vm.ModelParams()
    beta = params.beta if beta is None else beta

    cells = []
    for n_exp, frac, n_meas, P_pre, dP, _pct in R08_TABLE_II:
        mmd_sed = float(mmd_sed_from_pigmentation(P_pre, "uvb"))
        sed = frac * mmd_sed
        M_base = float(fwd(P_pre))
        # Convert the observed pigmentation-% increment through the local
        # derivative of the map, which is the correct first-order conversion
        # for an increment measured on the same instrument as the baseline.
        dM_obs = float(der(P_pre)) * dP
        cells.append(dict(n_exp=n_exp, frac=frac, n=n_meas, P_pre=P_pre,
                          dP=dP, sed=sed, mmd_sed=mmd_sed,
                          M_base=M_base, dM_obs=dM_obs))

    weights = np.sqrt([c["n"] for c in cells])

    def predict(theta, headroom):
        alpha, alpha_max, uv_half = theta[:3]
        out = []
        for c in cells:
            M_ceil = c["M_base"] + headroom
            M_end = simulate_pulses(c["n_exp"], c["sed"], c["M_base"], M_ceil,
                                    alpha, alpha_max, uv_half, beta,
                                    r08_intervals(c["n_exp"]))
            out.append(M_end - c["M_base"])
        return np.array(out)

    def resid(theta):
        headroom = theta[3] if fit_headroom else params.M_TAN_HEADROOM
        return weights * (predict(theta, headroom) - np.array([c["dM_obs"] for c in cells]))

    x0 = [params.alpha, params.alpha_max, params.UV_halfmax]
    lo = [1e-5, 1e-4, 0.2]
    hi = [1.0, 2.0, 40.0]
    if fit_headroom:
        x0.append(params.M_TAN_HEADROOM)
        lo.append(0.02)
        hi.append(0.80)
    sol = least_squares(resid, x0=x0, bounds=(lo, hi))

    headroom = sol.x[3] if fit_headroom else params.M_TAN_HEADROOM
    pred = predict(sol.x, headroom)
    obs = np.array([c["dM_obs"] for c in cells])
    return dict(alpha=sol.x[0], alpha_max=sol.x[1], UV_halfmax=sol.x[2],
                headroom=headroom, cells=cells, pred=pred, obs=obs,
                rms=float(np.sqrt(np.mean((pred - obs) ** 2))),
                map_name=map_name, cost=float(sol.cost))


def profile_alpha(alpha_fixed, map_name="power", beta=None):
    """Hold alpha at a given value and refit everything else. If the residual
    barely moves across decades of alpha, alpha is not identified by this data."""
    maps, _, _ = fit_pigment_maps()
    fwd, der = maps[map_name]
    params = vm.ModelParams()
    beta = params.beta if beta is None else beta

    cells = []
    for n_exp, frac, n_meas, P_pre, dP, _pct in R08_TABLE_II:
        sed = frac * float(mmd_sed_from_pigmentation(P_pre, "uvb"))
        M_base = float(fwd(P_pre))
        cells.append((n_exp, sed, M_base, float(der(P_pre)) * dP, n_meas))
    w = np.sqrt([c[4] for c in cells])
    obs = np.array([c[3] for c in cells])

    def resid(theta):
        alpha_max, uv_half, headroom = theta
        pred = []
        for n_exp, sed, M_base, _o, _n in cells:
            M_end = simulate_pulses(n_exp, sed, M_base, M_base + headroom,
                                    alpha_fixed, alpha_max, uv_half, beta,
                                    r08_intervals(n_exp))
            pred.append(M_end - M_base)
        return w * (np.array(pred) - obs)

    sol = least_squares(resid, x0=[max(alpha_fixed, 0.02), 2.0, 0.3],
                        bounds=([1e-4, 0.2, 0.02], [2.0, 40.0, 0.9]))
    r = resid(sol.x) / w
    return dict(alpha=alpha_fixed, alpha_max=sol.x[0], UV_halfmax=sol.x[1],
                headroom=sol.x[2], rms=float(np.sqrt(np.mean(r ** 2))))


# ---------------------------------------------------------------------------
# 5. beta FROM THE LINEAR-EXTRAPOLATION INVERSION
# ---------------------------------------------------------------------------

def linear_extrapolated_zero(beta, sample_days=FADE_SAMPLE_DAYS):
    """What Ravnbak's own procedure would report as 'days to total fading' if
    the tan actually decayed exponentially at rate beta. Fits an OLS line to
    (t, exp(-beta t)) at the three sampled days and returns its x-intercept."""
    t = np.asarray(sample_days, float)
    y = np.exp(-beta * t)
    slope, intercept = np.polyfit(t, y, 1)
    return -intercept / slope


def invert_linear_extrapolated_fade(observed_days, sample_days=FADE_SAMPLE_DAYS):
    """Recover the true exponential beta that Ravnbak's linear extrapolation
    would have reported as `observed_days`."""
    f = lambda b: linear_extrapolated_zero(b, sample_days) - observed_days
    return brentq(f, 1e-5, 0.5)


# ---------------------------------------------------------------------------
# 6. NEGATIVE CONTROL AND OUT-OF-SAMPLE CHECK
# ---------------------------------------------------------------------------

def uva_negative_control():
    """M11 shows UVA raises no melanin (8.62 vs 9.03 a.u., NS) while producing
    a comparable tan. R09 nonetheless measures large UVA pigmentation-%
    increases on the same instrument Ravnbak used for UVB. The ratio of the two
    bounds the non-melanin (photooxidative / redistribution) share of a
    reflectance-measured tan."""
    uvb = {(r[0], r[1]): r[4] for r in R08_TABLE_II}
    rows = []
    for n_exp, frac, _n, _P, dP_uva, _pct, _d in R09_TABLE_III:
        dP_uvb = uvb.get((n_exp, frac))
        if dP_uvb:
            rows.append((n_exp, frac, dP_uvb, dP_uva, dP_uva / dP_uvb))
    return rows


def check_miyamura(alpha, alpha_max, UV_halfmax, beta, headroom):
    """Out-of-sample check against M11, a study that plays no part in the fit.

    SCALE CAVEAT, stated up front: M11 measures epidermal melanin as
    Fontana-Masson mean optical density in arbitrary units; the model's M is an
    ITA-anchored 0-1 index. These are different scales and neither paper
    provides a conversion. The only quantity that transfers is the FRACTIONAL
    change, and even that transfers exactly only if both scales are locally
    linear in melanin amount. This is therefore an order-of-magnitude check on
    whether the fitted alpha lands in the right decade, not an equality test."""
    params = vm.ModelParams()
    obs_ratio = M11_MELANIN["UVB"] / M11_MELANIN["control"]
    rows = []
    for fst in (2, 3):
        M_base = params.M_baseline(fst)
        M_ceil = M_base + headroom
        for sed_per_med in M11_SED_PER_MED_RANGE:
            M = M_base
            for i, med in enumerate(M11_MED_PER_SESSION):
                sed = med * sed_per_med
                hill = sed ** 2 / (sed ** 2 + UV_halfmax ** 2)
                a_eff = alpha + (alpha_max - alpha) * hill
                M = M_ceil - (M_ceil - M) * np.exp(-a_eff * sed)
                if i < len(M11_INTERVALS):
                    M = M_base + (M - M_base) * np.exp(-beta * M11_INTERVALS[i])
            rows.append(dict(fst=fst, sed_per_med=sed_per_med,
                             cum_sed=4.5 * sed_per_med, M_base=M_base, M_end=M,
                             ratio=M / M_base if M_base > 0 else float("nan")))
    return obs_ratio, rows


def check_miyamura_stratum_corneum():
    """M11 Table 1 also reports stratum corneum thickness, which is an
    out-of-sample check on SC_max -- a parameter CLAUDE.md requires be flagged
    as an uncalibrated prior. M11 is FST II-III, so the comparison is against
    SC_max_by_FST for FST I-III (calibrated from de Boer 2025, a different
    cohort, different protocol and different measurement technique)."""
    params = vm.ModelParams()
    ctrl = M11_SC_UM["control"]
    out = {}
    for arm in ("UVA", "UVB", "UVA+UVB"):
        out[arm] = M11_SC_UM[arm] / ctrl - 1.0
    return out, params.SC_max_by_FST[2], ctrl


# ---------------------------------------------------------------------------
# 7. REPORT
# ---------------------------------------------------------------------------

def main():
    params = vm.ModelParams()
    W = 78
    def rule(c="-"):
        print(c * W)

    print("=" * W)
    print("MELANIN KINETICS FIT -- Ravnbak 2008 / Ravnbak 2009 / Miyamura 2011")
    print("=" * W)

    # -- Section 1: the scale map -------------------------------------------
    maps, P, M = fit_pigment_maps()
    print("\n[1] pigmentation-% -> model melanin index map")
    rule()
    print("  Anchors (Ravnbak pre-exposure pigmentation-% vs model M_baseline):")
    for fst, p, m in zip(BASELINE_BY_FST, P, M):
        print(f"    FST {fst}:  pig% = {p:5.2f}   M_baseline = {m:.3f}")
    print("\n  Candidate maps, fitted M at each anchor:")
    hdr = "    anchor  " + "".join(f"{k:>12}" for k in maps)
    print(hdr)
    for p, m in zip(P, M):
        row = f"    {p:5.2f}   " + "".join(f"{maps[k][0](p):12.3f}" for k in maps)
        print(row + f"   (obs {m:.3f})")
    print("\n  Map residual RMS on the four anchors:")
    for k, (fwd, _) in maps.items():
        r = np.sqrt(np.mean((fwd(P) - M) ** 2))
        print(f"    {k:>10}: {r:.4f}")

    # -- Section 2: alpha ----------------------------------------------------
    print("\n[2] Induction parameters fitted to Ravnbak 2008 (UVB/solar ONLY)")
    rule()

    # 2a. The ceiling problem, stated before any fit is reported.
    fixed = {name: fit_alpha(map_name=name) for name in maps}
    dM_obs = fixed["power"]["obs"]
    print("  FIRST, a structural finding that has to come before the numbers.")
    print(f"  Converted through the map, Ravnbak's measured tans are")
    print(f"  dM = {dM_obs.min():.6f} to {dM_obs.max():.6f} on the model's melanin index.")
    print(f"  The model's ENTIRE facultative headroom is "
          f"M_TAN_HEADROOM = {params.M_TAN_HEADROOM:.6g}.")
    n_over = int((dM_obs > params.M_TAN_HEADROOM).sum())
    if n_over:
        print(f"  {n_over} of the {len(dM_obs)} observed tans EXCEED that ceiling, so the")
        print("  model cannot represent them at any alpha: the ceiling binds, not")
        print("  the induction rate.")
    else:
        print(f"  All {len(dM_obs)} observed tans fit under that ceiling, with")
        print(f"  {params.M_TAN_HEADROOM - dM_obs.max():.6f} to spare against the largest.")
        print("  This was NOT true before the 2026-08-25 recalibration: at the")
        print("  previous headroom of 0.20, four of the six exceeded it, and the")
        print("  optimiser could only rail alpha to its bound trying to saturate")
        print("  a ceiling that was too low. That is what motivated the change.")
    print("  For scale: one whole Fitzpatrick step")
    print(f"  from FST III to IV is {P[2]-P[1]:.1f} pigmentation-%, while Ravnbak's")
    print(f"  12-exposure tans are {min(r[4] for r in R08_TABLE_II if r[0]==12):.1f}-"
          f"{max(r[4] for r in R08_TABLE_II if r[0]==12):.1f} pigmentation-%.")
    print("\n  Fit with headroom PINNED at the shipped value:")
    print(f"    {'map':>10} {'alpha':>10} {'alpha_max':>11} {'UV_halfmax':>11} {'RMS(dM)':>10}")
    for name, f in fixed.items():
        print(f"    {name:>10} {f['alpha']:10.5f} {f['alpha_max']:11.5f} "
              f"{f['UV_halfmax']:11.3f} {f['rms']:10.5f}")
    print("  Those are bound artefacts, not estimates. Do not quote them.")

    # 2b. Refit with the headroom free.
    print("\n  Refitting with M_TAN_HEADROOM free (it is flagged [PRIOR] in")
    print("  vitd_model.py and has no independent source):")
    fits = {name: fit_alpha(map_name=name, fit_headroom=True) for name in maps}
    print(f"    {'map':>10} {'alpha':>10} {'alpha_max':>11} {'UV_halfmax':>11} "
          f"{'headroom':>10} {'RMS(dM)':>10}")
    for name, f in fits.items():
        print(f"    {name:>10} {f['alpha']:10.5f} {f['alpha_max']:11.5f} "
              f"{f['UV_halfmax']:11.3f} {f['headroom']:10.3f} {f['rms']:10.5f}")
    print(f"\n  Currently shipped: alpha={params.alpha}, alpha_max={params.alpha_max}, "
          f"UV_halfmax={params.UV_halfmax}, headroom={params.M_TAN_HEADROOM}")

    best = fits["power"]
    print("\n  Per-cell fit under the 'power' map, headroom free:")
    print(f"    {'n_exp':>5} {'MMD':>5} {'n':>4} {'pig%':>6} {'SED/sess':>9} "
          f"{'dM_obs':>9} {'dM_pred':>9}")
    for c, pr in zip(best["cells"], best["pred"]):
        print(f"    {c['n_exp']:5d} {c['frac']:5.1f} {c['n']:4d} {c['P_pre']:6.1f} "
              f"{c['sed']:9.2f} {c['dM_obs']:9.4f} {pr:9.4f}")

    # 2c. Identifiability profile.
    print("\n  Identifiability: profiling alpha (refitting the rest at each value)")
    print(f"    {'alpha':>9} {'best RMS(dM)':>14} {'headroom':>10} {'UV_halfmax':>11}")
    for a_fix in (0.0025, 0.01, 0.03, 0.1, 0.3, 1.0):
        r = profile_alpha(a_fix, map_name="power")
        print(f"    {a_fix:9.4f} {r['rms']:14.5f} {r['headroom']:10.3f} "
              f"{r['UV_halfmax']:11.3f}")
    print("  Ravnbak's 6- and 12-exposure arms differ by only "
          f"{np.mean([r[4] for r in R08_TABLE_II if r[0]==12]) - np.mean([r[4] for r in R08_TABLE_II if r[0]==6]):.1f} pigmentation-%,")
    print("  i.e. the tan is already near plateau by 6 exposures. Two exposure")
    print("  counts three weeks apart cannot separate a fast rise to a low")
    print("  ceiling from a slow rise to a high one: alpha and the ceiling trade")
    print("  off directly. Ravnbak 2009 measured days-to-steady-state directly")
    print(f"  ({min(r[6] for r in R09_TABLE_III):.0f}-{max(r[6] for r in R09_TABLE_III):.0f} d), "
          "but only in the UVA arm, which section [4] shows")
    print("  is not the model's melanin. This is the binding data gap.")

    # -- Section 3: beta -----------------------------------------------------
    print("\n[3] beta from inverting Ravnbak's own linear-extrapolation method")
    rule()
    print("  Ravnbak fitted a STRAIGHT LINE through pigmentation measured at")
    print(f"  {FADE_SAMPLE_DAYS[0]:.0f} d, {FADE_SAMPLE_DAYS[1]:.0f} d and "
          f"{FADE_SAMPLE_DAYS[2]:.0f} d, then read off its x-intercept as")
    print("  'days to total fading'. If the tan really decays exponentially,")
    print("  that intercept is NOT the decay timescale. Inverting the procedure:")
    print(f"\n    {'arm':>22} {'reported fade (d)':>18} {'implied beta':>13} {'implied t1/2 (d)':>17}")
    betas = []
    for k, v in FADE_OBSERVED_UVB.items():
        b = invert_linear_extrapolated_fade(v)
        betas.append(b)
        print(f"    {k:>22} {v:18.0f} {b:13.5f} {np.log(2)/b:17.1f}")
    b_mean = float(np.mean(betas))
    print(f"\n    UVB-arm mean:  beta = {b_mean:.5f} /d   "
          f"half-life = {np.log(2)/b_mean:.1f} d")
    print(f"    Currently shipped: beta = {params.beta:.5f} /d   "
          f"half-life = {np.log(2)/params.beta:.1f} d")
    print("\n  UVA arm (NOT used to fit beta -- see module docstring):")
    for k, v in FADE_OBSERVED_UVA.items():
        b = invert_linear_extrapolated_fade(v)
        print(f"    {k:>22} {v:18.0f} {b:13.5f} {np.log(2)/b:17.1f}")
    # How many timepoints did the regression actually use? R08 Methods: "When
    # fading was completed at the 4 months control, the linear regression was
    # based only on the 1 week and 2 month measurements." So the sample is a
    # MIXTURE of 2-point and 3-point fits and the paper does not say in what
    # proportion. That ambiguity dominates the answer:
    print("\n  Sensitivity to how many timepoints Ravnbak's regression used.")
    print("  R08 Methods: 3 points normally, but only 1 wk + 2 mo when fading")
    print("  was already complete at 4 months. The mixture is not reported.")
    print(f"    {'timepoints used':>24} {'mean beta':>10} {'mean t1/2 (d)':>14}")
    for label, days in (("3 points (7, 60, 120 d)", (7.0, 60.0, 120.0)),
                        ("2 points (7, 60 d)", (7.0, 60.0))):
        bs = [invert_linear_extrapolated_fade(v, days)
              for v in FADE_OBSERVED_UVB.values()]
        print(f"    {label:>24} {np.mean(bs):10.5f} {np.log(2)/np.mean(bs):14.1f}")

    print("\n  The cleanest way to read this is forward: for a given true")
    print("  half-life, what would Ravnbak's procedure have reported?")
    print(f"    {'true t1/2 (d)':>14} {'reported total fading (d)':>26}")
    for hl in (30, 40, 48, 56, 70, 90):
        print(f"    {hl:14d} {linear_extrapolated_zero(np.log(2)/hl):26.0f}")
    print(f"\n  Ravnbak reported {min(FADE_OBSERVED_UVB.values()):.0f}-"
          f"{max(FADE_OBSERVED_UVB.values()):.0f} d for the UVB arms, which is "
          "matched by any true")
    print(f"  half-life in roughly 40-70 d. The shipped {np.log(2)/params.beta:.0f} d sits "
          "near the middle of")
    print("  that band, so the VALUE needs no change -- but its PROVENANCE does:")
    print("  it is no longer a guess about how many e-foldings 'total fading'")
    print("  means, it is the inversion of the procedure Ravnbak published.")

    # -- Section 4: UVA negative control ------------------------------------
    print("\n[4] UVA negative control (Ravnbak 2009 vs Miyamura 2011)")
    rule()
    print("  M11 Table 1, epidermal melanin content (a.u., mean +/- SEM, n=6):")
    for k in ("control", "UVA", "UVB", "UVA+UVB"):
        print(f"    {k:>9}: {M11_MELANIN[k]:6.2f} +/- {M11_MELANIN_SEM[k]:.2f}")
    print(f"    UVB raises melanin content by "
          f"{100*(M11_MELANIN['UVB']/M11_MELANIN['control']-1):.1f}% (P<0.01);")
    print(f"    UVA changes it by "
          f"{100*(M11_MELANIN['UVA']/M11_MELANIN['control']-1):+.1f}% (not significant).")
    print("\n  Yet Ravnbak measures comparable reflectance tans from both:")
    print(f"    {'n_exp':>5} {'MMD':>5} {'dPig% UVB':>11} {'dPig% UVA':>11} {'UVA/UVB':>9}")
    for n_exp, frac, dP_uvb, dP_uva, ratio in uva_negative_control():
        print(f"    {n_exp:5d} {frac:5.1f} {dP_uvb:11.1f} {dP_uva:11.1f} {ratio:9.2f}")
    print("\n  => a reflectance-measured tan of a given size carries a variable and")
    print("     source-dependent amount of actual melanin. Fitting the model's")
    print("     ATTENUATING melanin to pooled UVA+UVB reflectance data would")
    print("     overstate photoprotection. This is why the fit uses R08 only.")

    # -- Section 5: Miyamura out-of-sample ----------------------------------
    print("\n[5] Out-of-sample check: Miyamura 2011 (not used in the fit)")
    rule()
    m11_gain = M11_MELANIN["UVB"] / M11_MELANIN["control"] - 1.0
    r08_gain = [r[5] for r in R08_TABLE_II]
    print("  The cleanest comparison is between the two studies' own RELATIVE")
    print("  gains, since neither reports the other's units:")
    print(f"    M11, 10 UVB exposures (4.5 MED cumulative), FST II-III:")
    print(f"      epidermal melanin content  {m11_gain:+.1%}  (Fontana-Masson a.u.)")
    print(f"    R08, 6-12 UVB/solar exposures, FST II-V:")
    print(f"      reflectance pigmentation   +{min(r08_gain):.1f}% to +{max(r08_gain):.1f}%")
    print("\n  M11's melanin gain is LARGER than R08's reflectance gain, which is")
    print("  the direction section [4] predicts: M11 used pure UVB and measured")
    print("  melanin directly, while R08's reflectance endpoint includes")
    print("  non-melanin optical change that dilutes the per-unit melanin signal.")
    print("  The two are consistent. Neither pins alpha, because neither reports")
    print("  enough intermediate timepoints to separate rate from ceiling.")
    print("\n  Model prediction under the SHIPPED parameters (alpha="
          f"{params.alpha}, headroom={params.M_TAN_HEADROOM}),")
    print("  as a check that the shipped values are not grossly wrong:")
    _obs, rows = check_miyamura(params.alpha, params.alpha_max,
                                params.UV_halfmax, params.beta,
                                params.M_TAN_HEADROOM)
    print(f"    {'FST':>4} {'SED/MED':>8} {'cum SED':>8} {'M_base':>8} "
          f"{'M_end':>8} {'dM/M_base':>10}")
    for r in rows:
        gain = (r["M_end"] - r["M_base"]) / r["M_base"] if r["M_base"] > 0 else float("nan")
        print(f"    {r['fst']:4d} {r['sed_per_med']:8.1f} {r['cum_sed']:8.2f} "
              f"{r['M_base']:8.3f} {r['M_end']:8.3f} {gain:9.1%}")
    print("\n  SCALE CAVEAT: Fontana-Masson a.u. and the model's ITA-anchored")
    print("  index are different scales; only the fractional change transfers,")
    print("  and only to first order. Read this as a decade check, not a test.")

    # -- Section 6: SC bonus -------------------------------------------------
    print("\n[6] BONUS -- Miyamura 2011 is also an out-of-sample check on SC_max")
    rule()
    sc_obs, sc_model, ctrl = check_miyamura_stratum_corneum()
    print(f"  M11 Table 1 stratum corneum thickness, control = {ctrl:.1f} um:")
    for arm, frac in sc_obs.items():
        print(f"    {arm:>9}: {M11_SC_UM[arm]:5.1f} um  ({frac:+.1%} vs control)")
    print(f"\n  Model SC_max_by_FST for FST I-III = {sc_model:+.2f} "
          f"(+{100*sc_model:.0f}%), calibrated from de Boer 2025 --")
    print(f"  a different cohort, protocol and measurement method. M11's UVB arm")
    print(f"  gives {sc_obs['UVB']:+.1%}. These agree to within "
          f"{abs(sc_obs['UVB']-sc_model)*100:.1f} percentage points.")
    print("  NOTE M11 reports SEM not SD and n=6, so this is corroboration of")
    print("  magnitude, not a tight constraint. It is nonetheless the first")
    print("  independent support for SC_max in this repository.")

    print("\n" + "=" * W)
    print("CONCLUSIONS -- what this data does and does not fix")
    print("=" * W)
    h_lo = min(f["headroom"] for f in fits.values())
    h_hi = max(f["headroom"] for f in fits.values())
    print("""
  NOT IDENTIFIED -- leave at the shipped values, do not 'fit' them:

    alpha, alpha_by_FST, alpha_max, UV_halfmax
      The profile in section [2] moves the weighted residual by less than
      25% while alpha sweeps a 400-fold range. Ravnbak dosed at only
      two exposure counts (6 and 12) and the tan is already near plateau at
      6, so the data constrain WHERE the tan saturates, not HOW FAST it gets
      there. Any alpha in that range fits equally. The shipped alpha=0.0025
      is not contradicted by this data -- but it is not supported by it
      either, and the honest description remains PRIOR, not FITTED.
      Fixing this needs a study with intermediate timepoints during the
      build-up phase in a UVB arm.""")
    print(f"""
  CONSTRAINED -- the data does speak to these:

    M_TAN_HEADROOM   live value {params.M_TAN_HEADROOM:.6g}   free-fit range {h_lo:.6f} - {h_hi:.6f}
      ADOPTED 2026-08-25 (user-approved): 0.20 -> 0.238, where 0.237724 is
      the largest tan Ravnbak actually measured on this index. Stable across
      all three scale maps and across every alpha in the profile, because it
      is set by the plateau -- which is exactly the thing Ravnbak measured
      well. The previous 0.20 could not represent four of the six observed
      tans at any alpha.

    beta             shipped {params.beta:.5f} /d (t1/2 {np.log(2)/params.beta:.0f} d)  -- KEEP THE VALUE
      Supported range from the inversion: t1/2 roughly 40-70 d. The shipped
      56 d sits near the middle, so no change is warranted; a point estimate
      of {np.log(2)/b_mean:.0f} d would over-read the data, because R08 mixed 2-point and
      3-point regressions in unreported proportion and that choice alone
      moves the answer from ~48 d to ~74 d.
      What DOES change is the provenance. The shipped value was a stated
      guess ("a mid-range estimate compatible with Ravnbak if complete
      fading requires ~3 e-folding times"). Section [3] derives it instead
      by inverting the procedure Ravnbak actually published -- a straight
      line through the sampled timepoints, extrapolated to zero. Same
      number, but now with a derivation and a range behind it.""")
    print("""
  CORROBORATED and RECALIBRATED:

    SC_max (FST I-III)   0.23 -> 0.24, ADOPTED 2026-08-25 (user-approved)
      de Boer 2025 gives +23%. Miyamura 2011 Table 1 independently gives
      stratum corneum 16.6 -> 20.7 um in FST II-III, i.e. +24.6988%, from a
      different cohort, protocol and measurement method. 0.24 sits between
      them (their midpoint is 0.238494). This is the first out-of-sample
      support SC_max has had in this repository.
      It corroborates MAGNITUDE only: n=6, SEM reported, and Miyamura measured
      no vitamin D. CLAUDE.md still requires SC_max be flagged as an
      uncalibrated prior wherever absolute D is presented, and that flag
      stays. FST IV-VI is UNCHANGED at 0.36 and still rests on de Boer alone
      -- Miyamura enrolled FST II-III only. Under the dark-skin priority
      directive that is the gap that matters most, and it remains open.

  STATUS: both calibrations above are approved and applied to vitd_model.py.
  calibration_approval.yaml carries the approval records. run_all.py must be
  re-run before any downstream number is quoted.""")


if __name__ == "__main__":
    main()
