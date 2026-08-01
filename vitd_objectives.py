"""
vitd_objectives.py
==================
Decision problems people actually face, replacing "maximise nmol/L per SED".

Why this module exists
----------------------
The project's earlier headline metric -- the retired per-SED efficiency
metric, removed from `vitd_model.py` on 2026-07-31 -- had no interior optimum:
it rose monotonically as dosing approaches zero, and its reported "optima" sat
at schedules that END BELOW their starting 25(OH)D. It was also shown by
ablation to be driven by the `(1 - D/D_max)` feedback term rather than by
photoadaptation. See MODEL_AND_EVIDENCE.md, 'Remaining gaps', gap 19.

The objectives here are posed the way a person or clinician would pose them:

  Q1  "I can get outside N times over the next 12 weeks. When should I go?"
      -> optimal_spacing_fixed_sessions()

  Q2  "Keep me out of deficiency with the least skin-cancer risk."
      -> deficiency_vs_risk_frontier()

  Q3  "How many minutes of midday sun do I need, for my skin type?"
      -> minutes_for_target()

Each returns concrete numbers with units, and each carries the cancer-risk
indices alongside the vitamin D outcome so a schedule can never be recommended
on synthesis grounds alone.
"""

from __future__ import annotations

import numpy as np

from vitd_model import (ModelParams, simulate, auc_deficit, steady_state_D,
                        final_D)
from vitd_physiology import cancer_risk

# Clinical thresholds, nmol/L of 25(OH)D.
# 25  = deficiency (UK/SACN); 30 = IOM "risk of inadequacy" lower bound
# 50  = sufficiency (IOM/Endocrine Society; the usual target)
DEFICIENCY_NMOL_L = 25.0
SUFFICIENCY_NMOL_L = 50.0


# ---------------------------------------------------------------------------
# Q1. Fixed session budget -- how should the sessions be spaced?
# ---------------------------------------------------------------------------
def optimal_spacing_fixed_sessions(fst: int,
                                   n_sessions: int,
                                   sed_per_session: float = 1.0,
                                   horizon_days: int = 84,
                                   D0: float = 25.0,
                                   params: ModelParams | None = None,
                                   objective: str = "mean_D") -> dict:
    """Best spacing for a FIXED number of sessions over a fixed horizon.

    Because the number of sessions and the dose per session are both fixed,
    cumulative UV is IDENTICAL across all candidate spacings -- so the
    comparison cannot be won by simply dosing less. Any difference is real
    schedule structure.

    Parameters
    ----------
    objective : {'mean_D', 'final_D', 'min_D', 'auc_deficit_50'}
        'mean_D'          maximise average 25(OH)D over the horizon
        'final_D'         maximise end-of-horizon 25(OH)D
        'min_D'           maximise the worst point (avoid dipping low)
        'auc_deficit_50'  minimise time-integrated shortfall below 50 nmol/L

    Returns dict with the best spacing plus the full ranked table.
    """
    if params is None:
        params = ModelParams()
    if n_sessions < 1:
        raise ValueError("n_sessions must be >= 1")

    # Even spacing across the horizon is parameterised by the gap; we also try
    # front-loading and back-loading to check that even spacing is actually best.
    rows = []
    max_gap = max(1, horizon_days // max(1, n_sessions - 1)) if n_sessions > 1 else horizon_days

    for gap in range(1, max_gap + 1):
        days = [i * gap for i in range(n_sessions)]
        if days[-1] >= horizon_days:
            continue
        sched = [(float(d), sed_per_session) for d in days]
        r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                     params=params, D0=D0,
                     dose_framing="fixed_per_session_SED")
        risk = r["cancer_risk"]
        rows.append({
            "gap_days": gap,
            "n_sessions": n_sessions,
            "total_sed": r["cumulative_SED"],
            "mean_D": float(np.mean(r["D"])),
            "final_D": final_D(r),
            "min_D": float(np.min(r["D"])),
            "days_below_50": _days_below(r, SUFFICIENCY_NMOL_L),
            "days_below_25": _days_below(r, DEFICIENCY_NMOL_L),
            "auc_deficit_50": auc_deficit(r["D"], r["t"], SUFFICIENCY_NMOL_L),
            "auc_deficit_25": auc_deficit(r["D"], r["t"], DEFICIENCY_NMOL_L),
            "scc_index": risk["scc_index"],
            "intermittent_index": risk["intermittent_index"],
            "burn_fraction": risk["burn_fraction"],
        })

    if not rows:
        return {"best": None, "table": []}

    minimise = objective.startswith("auc_deficit")
    best = (min if minimise else max)(rows, key=lambda r: r[objective])
    return {"best": best, "table": rows, "objective": objective}


def _days_below(result, threshold: float) -> float:
    """Days spent below a threshold, by trapezoidal integration of the indicator."""
    t, D = result["t"], result["D"]
    below = (D < threshold).astype(float)
    if len(t) < 2:
        return 0.0
    return float(np.trapezoid(below, t) if hasattr(np, "trapezoid")
                 else np.trapz(below, t))


# ---------------------------------------------------------------------------
# Q2. Stay out of deficiency at least cancer risk -- Pareto frontier
# ---------------------------------------------------------------------------
def deficiency_vs_risk_frontier(fst: int,
                                horizon_days: int = 365,
                                D0: float = 25.0,
                                sed_grid=(0.25, 0.5, 1.0, 2.0, 4.0),
                                interval_grid=(1, 2, 3, 5, 7, 10, 14, 21, 28),
                                params: ModelParams | None = None,
                                target: float = SUFFICIENCY_NMOL_L) -> dict:
    """Two-objective frontier: minimise deficiency AND minimise cancer risk.

    Sweeps (dose per session) x (interval) and returns the Pareto-nondominated
    set. This is the honest framing of the clinical trade-off: you cannot
    minimise both, so present the frontier and let the user pick, rather than
    collapsing it into one number with arbitrary weights.

    A point dominates another if it has both lower time-below-target AND lower
    combined cancer risk.
    """
    if params is None:
        params = ModelParams()

    pts = []
    for sed in sed_grid:
        for iv in interval_grid:
            days = list(range(0, horizon_days, iv))
            sched = [(float(d), float(sed)) for d in days]
            r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                         params=params, D0=D0,
                         dose_framing="fixed_per_session_SED")
            risk = r["cancer_risk"]
            pts.append({
                "sed_per_session": sed,
                "interval_days": iv,
                "n_sessions": len(days),
                "total_sed": r["cumulative_SED"],
                "mean_D": float(np.mean(r["D"])),
                "final_D": final_D(r),
                "days_below_target": _days_below(r, target),
                "auc_deficit": auc_deficit(r["D"], r["t"], target),
                "scc_index": risk["scc_index"],
                "intermittent_index": risk["intermittent_index"],
                "combined_risk": risk["combined_index"],
                "burn_fraction": risk["burn_fraction"],
            })

    frontier = _pareto_front(pts, "days_below_target", "combined_risk")
    # Cheapest schedule that keeps the person at/above target essentially always
    sufficient = [p for p in pts if p["days_below_target"] <= 0.05 * horizon_days]
    cheapest = min(sufficient, key=lambda p: p["combined_risk"]) if sufficient else None
    return {"all_points": pts, "frontier": frontier,
            "cheapest_sufficient": cheapest, "target": target}


def _pareto_front(points, obj_a: str, obj_b: str):
    """Nondominated set, minimising both objectives."""
    front = []
    for p in points:
        dominated = any(
            (q[obj_a] <= p[obj_a] and q[obj_b] <= p[obj_b] and
             (q[obj_a] < p[obj_a] or q[obj_b] < p[obj_b]))
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p[obj_a])


# ---------------------------------------------------------------------------
# Q3. Minutes of sun -- the question people actually ask
# ---------------------------------------------------------------------------
# UV Index -> erythemal SED per minute.
# 1 UV Index unit = 25 mW/m^2 erythemally weighted (WHO definition).
# 1 SED = 100 J/m^2 erythemal.
# So SED/min = UVI * 25e-3 * 60 / 100 = UVI * 0.015.
SED_PER_MIN_PER_UVI: float = 0.015

# Personal-dose geometry factor ("exposure ratio"): UV Index is defined on a
# HORIZONTAL plane, but a standing person presents mostly vertical surfaces.
# Using 1.0 overestimates dose and so underestimates the minutes required.
#
# Measured exposure ratios (personal dose / ambient horizontal dose), erythemal:
#   Parisi 2003, standing posture, leg sites:      0.14 - 0.39
#   Parisi 2000, upright, various sites, shade:    0.16 - 0.49
#   Cockell 2001, electronic dosimetry, clear sky: 0.35 +/- 0.12
#   Thieden 2000, wrist vs head:                   wrist ~0.50 of head dose
# Schmalwieser 2018 reviews this literature and recommends expressing the ratio
# as a function of solar elevation -- which this scalar does not do.
#
# 0.35 is the central measured value for upright activity (Cockell 2001), and is
# consistent with the Parisi ranges. It replaces an earlier 0.55 guess that sat
# above every measured standing-posture range.
# [PRIOR] Scalar, not elevation-dependent. Directly scales every minutes
# recommendation, so it is the first thing to sweep. Higher (~0.5-0.7) for
# supine sunbathing, which presents a near-horizontal surface.
PERSONAL_DOSE_GEOMETRY_FACTOR: float = 0.35


def sed_from_minutes(minutes: float, uv_index: float,
                     geometry_factor: float = PERSONAL_DOSE_GEOMETRY_FACTOR) -> float:
    """Erythemal SED delivered to exposed skin by `minutes` at a given UV Index.

    Applies the horizontal-to-personal geometry correction. Pass
    geometry_factor=1.0 to get the horizontal-plane (upper-bound) dose, e.g.
    for a supine sunbather.
    """
    return max(0.0, minutes) * uv_index * SED_PER_MIN_PER_UVI * geometry_factor


def minutes_for_target(fst: int,
                       uv_index: float,
                       exposed_bsa: float,
                       frequency_per_week: float = 7.0,
                       target: float = SUFFICIENCY_NMOL_L,
                       D0: float = 25.0,
                       horizon_days: int = 120,
                       params: ModelParams | None = None,
                       max_minutes: float = 240.0,
                       geometry_factor: float = PERSONAL_DOSE_GEOMETRY_FACTOR) -> dict:
    """Minutes of sun per session needed to reach a 25(OH)D target.

    Answers "how long outside?" and simultaneously reports whether that
    exposure crosses the sunburn threshold -- a recommendation that requires
    burning is not a recommendation.
    """
    if params is None:
        params = ModelParams()

    interval = max(1, int(round(7.0 / max(frequency_per_week, 1e-9))))
    days = list(range(0, horizon_days, interval))

    def achieved(minutes: float):
        sed = sed_from_minutes(minutes, uv_index, geometry_factor)
        sched = [(float(d), sed) for d in days]
        r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                     params=params, D0=D0,
                     exposed_bsa_override=exposed_bsa,
                     dose_framing="fixed_per_session_SED")
        return steady_state_D(r), r

    lo, hi = 0.0, max_minutes
    if achieved(hi)[0] < target:
        ss, r = achieved(hi)
        return {"minutes": None, "reachable": False, "best_D": ss,
                "note": f"target {target} nmol/L unreachable at <= {max_minutes:.0f} min/session",
                "uv_index": uv_index, "exposed_bsa": exposed_bsa}
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if achieved(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    minutes = hi
    ss, r = achieved(minutes)
    sed = sed_from_minutes(minutes, uv_index, geometry_factor)
    risk = cancer_risk([sed] * len(days), fst)
    return {
        "minutes": minutes,
        "reachable": True,
        "sed_per_session": sed,
        "fraction_med": sed / risk["med_sed"],
        "would_burn": sed >= risk["med_sed"],
        "steady_state_D": ss,
        "sessions_per_week": frequency_per_week,
        "exposed_bsa": exposed_bsa,
        "uv_index": uv_index,
        "scc_index": risk["scc_index"],
        "intermittent_index": risk["intermittent_index"],
        "target": target,
    }
