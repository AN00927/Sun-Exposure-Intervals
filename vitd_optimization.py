"""
vitd_optimization.py
====================
Schedule generators and analysis for the vitamin D scheduling model.

Schedule generators create lists of (day, SED) tuples with different
spacing patterns for the constant-sessions analysis.

Analysis functions evaluate schedules on efficacy and safety metrics,
producing annotated rankings.
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Iterator, Optional, Sequence
from vitd_model import (
    simulate, build_pulse_schedule, steady_state_D,
    auc_deficit, net_final_D_change_per_SED, final_D,
    ModelParams, SOURCE_CALIBRATION_TABLE,
)
from vitd_objectives import (SUFFICIENCY_NMOL_L, GAMMA_MED, RAMP_IN_DAYS,
                             PRACTICAL_MAX_MINUTES,
                             PERSONAL_DOSE_GEOMETRY_FACTOR,
                             sed_from_minutes, _days_below, time_to_sufficiency)
from vitd_physiology import _med_sed_by_fst


def schedule_uniform(N: int, T_days: int, sed: float) -> List[Tuple[float, float]]:
    if N <= 0:
        return []
    interval = T_days / N
    return [(i * interval, sed) for i in range(N)]


def _geom_space(N: int, T_days: int, skew: float, reverse: bool = False) -> np.ndarray:
    if N <= 1:
        return np.array([0.0])
    if abs(skew - 1.0) < 1e-12:
        return np.linspace(0, T_days * (1 - 1e-12), N)
    raw = np.geomspace(1.0, skew, N) - 1.0
    raw = raw / raw[-1] * T_days
    if reverse:
        raw = T_days - raw[::-1]
    return raw


def schedule_frontloaded(N: int, T_days: int, sed: float,
                         skew: float = 2.0) -> List[Tuple[float, float]]:
    days = _geom_space(N, T_days, skew, reverse=False)
    return [(float(d), sed) for d in days]


def schedule_backloaded(N: int, T_days: int, sed: float,
                        skew: float = 2.0) -> List[Tuple[float, float]]:
    days = _geom_space(N, T_days, skew, reverse=True)
    return [(float(d), sed) for d in days]


def schedule_random(N: int, T_days: int, sed: float,
                    n_samples: int = 100) -> Iterator[List[Tuple[float, float]]]:
    for _ in range(n_samples):
        days = np.random.uniform(0, T_days, N)
        days.sort()
        yield [(float(d), sed) for d in days]


# ---------------------------------------------------------------------------
# Variable-dose profiles
# ---------------------------------------------------------------------------
# The generators above vary WHEN sessions happen at a fixed dose. These vary
# HOW MUCH each session delivers on a fixed set of session days -- the other
# half of the schedule, and the half no optimiser in this repo searched until
# now (see the C3/C7 note on optimal_interval_risk_constrained, whose dose
# bisection is exact only within the constant-dose family).
#
# Each returns an SED array aligned to `days`. They are shape functions only:
# clipping to the erythema ceiling is the caller's job, via
# build_variable_schedule.
#
# At a FIXED total budget and session count the intermittency term sum
# (d_i/MED)^BURN_EXPONENT is convex in the doses, so by Jensen equal doses
# strictly minimise it -- variable dose can never win on the risk axis alone.
# The gain comes from reducing the budget: one constant dose has to both climb
# to the target and hold there, so it necessarily overshoots in steady state.
# Splitting those two jobs is what pays.
def session_days(N: int, T_days: int) -> np.ndarray:
    """Uniformly spaced session days -- the spacing that minimises the largest gap."""
    if N <= 0:
        return np.zeros(0)
    return np.linspace(0.0, T_days, N, endpoint=False)


def dose_uniform(days: np.ndarray, d: float) -> np.ndarray:
    """Constant dose. The current baseline; 1 free parameter."""
    return np.full(len(days), float(d))


def dose_two_phase(days: np.ndarray, d_load: float, d_maint: float,
                   t_switch: float) -> np.ndarray:
    """Step: `d_load` before `t_switch`, `d_maint` after. 3 parameters.

    The most clinically communicable family -- "N weeks at X, then Y" -- and
    empirically it captures most of the available gain over constant dosing.
    """
    days = np.asarray(days, dtype=float)
    return np.where(days < float(t_switch), float(d_load), float(d_maint))


def dose_exponential(days: np.ndarray, d_inf: float, ratio: float,
                     tau: float) -> np.ndarray:
    """Smooth decay from `d_inf * (1 + ratio)` to `d_inf`, e-folding time `tau`.

    3 parameters. The continuous analogue of dose_two_phase; preferred when a
    hard step is implausible as behaviour.
    """
    days = np.asarray(days, dtype=float)
    return float(d_inf) * (1.0 + float(ratio) * np.exp(-days / max(float(tau), 1e-6)))


def dose_seasonal(days: np.ndarray, d0: float, amp: float,
                  phase_day: float = 80.0) -> np.ndarray:
    """Sinusoidal annual modulation. 2 free parameters (+ phase).

    Treated as a free *choice* this family loses to constant dosing, because
    the model's ambient UV does not vary: modulating dose then only adds
    intermittency for nothing. Its real use is as a *constraint* -- outside
    the tropics the achievable dose is seasonal whether or not the schedule
    wants it to be, so a schedule that is flat in behaviour is already
    sinusoidal in delivered SED. Reading a loss here as "seasonality does not
    matter" inverts the finding. `phase_day` defaults to a spring ascending
    zero-crossing, putting the peak near midsummer.
    """
    days = np.asarray(days, dtype=float)
    w = 1.0 + float(amp) * np.sin(2.0 * np.pi * (days - float(phase_day)) / 365.0)
    return float(d0) * np.clip(w, 0.0, None)


def dose_gap_coupled(days: np.ndarray, d_ref: float, p: float) -> np.ndarray:
    """Dose scaled by the preceding gap: d_i ~ d_ref * (gap_i / mean_gap) ** p.

    1 parameter beyond the constant-dose baseline, which it recovers at p = 0.
    Generalises schedule_frontloaded / schedule_backloaded, which vary spacing
    at fixed dose: here spacing and dose co-vary, so an unavoidably long gap
    gets compensated rather than simply tolerated.

    Degenerate under uniform spacing. When `days` is evenly spaced, every
    `gap_i` equals `mean_gap`, so `(gap_i / mean_gap) ** p == 1` for every p
    and this returns the constant-dose profile `dose_uniform(days, d_ref)`
    regardless of p (verified for p in {0, 0.5, 1, 1.5, 3}). The family only
    carries information when day spacing is non-uniform. In the current
    pipeline, `optimise_variable_dose` is the only caller and it always
    builds `days` from `session_days()`, which is uniformly spaced -- so as
    called today, this family is always degenerate and cannot outperform (or
    underperform) `dose_uniform`. `optimise_joint_schedule`, which does warp
    the day spacing non-uniformly, restricts its dose search to
    `JOINT_DOSE_FAMILIES = ("two_phase", "exponential")` and never calls this
    function, so the non-degenerate case is not currently exercised anywhere
    in this repository.
    """
    days = np.asarray(days, dtype=float)
    if days.size == 0:
        return np.zeros(0)
    gaps = np.diff(days, prepend=days[0] - (days[1] - days[0] if days.size > 1 else 1.0))
    mean_gap = float(np.mean(gaps)) if np.mean(gaps) > 0 else 1.0
    return float(d_ref) * (gaps / mean_gap) ** float(p)


def dose_knots(days: np.ndarray, values: Sequence[float],
               T_days: Optional[float] = None) -> np.ndarray:
    """Shape-preserving interpolation through evenly spaced knots.

    len(values) parameters -- the flexible middle ground between a 3-parameter
    family and a free per-session vector. Uses PCHIP, which does not overshoot
    between knots, so the profile cannot go negative or spike between control
    points. Smoothness is a hard requirement here rather than an aesthetic one:
    a dose profile that oscillates session to session is not a schedule a person
    can execute.
    """
    days = np.asarray(days, dtype=float)
    values = np.asarray(values, dtype=float)
    if days.size == 0:
        return np.zeros(0)
    if values.size == 1:
        return np.full(days.size, float(values[0]))
    span = float(T_days) if T_days is not None else float(days[-1] if days.size else 1.0)
    knot_x = np.linspace(0.0, max(span, 1e-9), values.size)
    try:
        from scipy.interpolate import PchipInterpolator
        return np.asarray(PchipInterpolator(knot_x, values)(days), dtype=float)
    except Exception:
        return np.interp(days, knot_x, values)


def build_variable_schedule(days: np.ndarray, seds: np.ndarray,
                            fst: Optional[int] = None,
                            cap_fraction: float = GAMMA_MED,
                            drop_below: float = 1e-6
                            ) -> List[Tuple[float, float]]:
    """Zip days and doses into a schedule, clipped to the erythema ceiling.

    Doses are clipped to `cap_fraction * MED(fst)` (constraint C1) rather than
    rejected, so an optimiser that walks into the ceiling gets a feasible point
    with a flat gradient instead of an error. Sessions at effectively zero dose
    are dropped, which is what makes the session count itself optimisable
    without any integer variables: `d_i = 0` simply means "no session that day".
    """
    days = np.asarray(days, dtype=float)
    seds = np.asarray(seds, dtype=float)
    seds = np.nan_to_num(seds, nan=0.0, posinf=0.0, neginf=0.0)
    seds = np.clip(seds, 0.0, None)
    if fst is not None:
        seds = np.clip(seds, 0.0, cap_fraction * _med_sed_by_fst(fst))
    return [(float(d), float(s)) for d, s in zip(days, seds) if s > drop_below]


def evaluate_schedule(
    schedule: List[Tuple[float, float]],
    fst: int,
    duration_days: int,
    params: Optional[ModelParams] = None,
    D0: float = 25.0,
) -> dict:
    result = simulate(
        fst=fst, schedule=schedule, duration_days=duration_days,
        params=params, D0=D0,
        dose_framing="fixed_absolute_SED",
    )

    t = result["t"]
    D = result["D"]
    safety_log = result.get("safety_log", [])

    max_frac = 0.0
    cpd_risk = "LOW"
    chronic_risk = "LOW"
    for _t, s in safety_log:
        frac = s.get("fraction_med", 0)
        if frac > max_frac:
            max_frac = frac
            cpd_risk = s.get("cpd_risk_level", "LOW")
            chronic_risk = s.get("chronic_risk_level", "LOW")

    # Rolling 7d SED: at each session, sum SED from sessions in prior 7 days
    max_rolling_7d = 0.0
    for i, (day_i, sed_i) in enumerate(schedule):
        rolling_7d = sed_i + sum(
            sed_j for day_j, sed_j in schedule[:i]
            if day_i - 7.0 <= day_j < day_i
        )
        if rolling_7d > max_rolling_7d:
            max_rolling_7d = rolling_7d

    return {
        "auc_deficit_30": auc_deficit(D, t, 30.0),
        "auc_deficit_50": auc_deficit(D, t, 50.0),
        "steady_state_D": steady_state_D(result),
        # Scored because the C3 floor starts at RAMP_IN_DAYS and so cannot see
        # how fast a schedule climbs -- see vitd_objectives.time_to_sufficiency.
        "time_to_sufficiency": time_to_sufficiency(result, SUFFICIENCY_NMOL_L),
        "min_D_after_ramp": (float(np.min(D[t >= RAMP_IN_DAYS]))
                             if np.any(t >= RAMP_IN_DAYS) else float("nan")),
        "max_D": float(np.max(D)),
        "cumulative_SED": result["cumulative_SED"],
        "n_sessions": len(schedule),
        "net_final_D_change_per_SED": net_final_D_change_per_SED(result, D0),
        "max_pct_MED": max_frac * 100,
        "max_rolling_7d_SED": max_rolling_7d,
        "cpd_risk": cpd_risk,
        "chronic_risk": chronic_risk,
    }


# ---------------------------------------------------------------------------
# Risk-constrained interval recommendation
# ---------------------------------------------------------------------------
# The Pareto frontier (vitd_objectives.deficiency_vs_risk_frontier) is the honest
# presentation of the trade-off, but a frontier is not advice. This collapses it
# to one answer under an explicit constraint set, which is what
# scripts/skin_type_breakdown.py already does for the deficiency-only problem.
#
# Constraints. The C-numbers are the repository's standing constraint
# labels; they are declared alongside the objectives they bind in
# vitd_objectives.OBJECTIVE_HIERARCHY (the narrative write-up that used to
# number them was purged on 2026-08-18):
#   C1  per-session dose <= GAMMA_MED * MED(F)          (never burns)
#   C3  D(t) >= target for all t >= ramp_in_days        (sufficiency floor)
#   C7  per-session dose <= what max_minutes can deliver at this UV index
#   C9  combined cancer-risk index <= risk_cap          (NEW)
#
# Objective: minimise the combined risk index subject to all of the above.
#
# At a fixed interval both risk indices are strictly increasing in per-session
# dose, so the lowest-risk feasible dose is always the smallest one that still
# satisfies C3. One bisection per interval replaces a dose grid without loss.
def optimal_interval_risk_constrained(
        fst: int,
        target: float = SUFFICIENCY_NMOL_L,
        risk_cap: Optional[float] = None,
        horizon_days: int = 365,
        D0: float = 25.0,
        params: Optional[ModelParams] = None,
        interval_grid: Sequence[int] = (1, 2, 3, 5, 7, 10, 14, 21, 28),
        max_minutes: Optional[float] = None,
        uv_index: float = 6.0,
        exposed_bsa: Optional[float] = None,
        geometry_factor: Optional[float] = None,
        ramp_in_days: int = RAMP_IN_DAYS,
) -> dict:
    """Lowest-risk exposure interval that still holds 25(OH)D at/above target.

    Answers "go outside every N days, for M minutes" with a single schedule,
    rather than a frontier the reader has to pick from.

    Parameters
    ----------
    fst : int
        Fitzpatrick skin type, 1-6.
    target : float
        Sufficiency floor in nmol/L that must hold for every day at or after
        `ramp_in_days` (constraint C3).
    risk_cap : float, optional
        Upper bound on the combined cancer-risk index (constraint C9). When
        None, every sufficiency-feasible interval is admissible and the
        recommendation is simply the lowest-risk one.
    max_minutes : float, optional
        Session-length ceiling (constraint C7). Converted to a dose ceiling at
        `uv_index` via vitd_objectives.sed_from_minutes, i.e. with the
        horizontal-to-personal geometry correction applied.
    exposed_bsa : float, optional
        Exposed body-surface fraction; None uses the model default.
    geometry_factor : float, optional
        Horizontal-to-personal geometry correction passed straight through to
        `sed_from_minutes`. None falls back to the flat 0.35 Cockell scalar --
        pass the location/clothing-specific value from
        `vitd_exposure.ExposureResult.geometry_factor` to use the real
        Vernez 2015 geometry for a given site set and solar elevation
        instead of the flat scalar. Independent of `exposed_bsa`: the two
        scale different things (dose per minute vs synthesis by skin area).

    Returns
    -------
    dict with keys
        recommended   the chosen schedule (None if no interval is admissible)
        evaluated     every interval on the grid, feasible or not
        risk_cap      echoed back
        target        echoed back
        note          set when nothing is admissible, explaining which
                      constraint eliminated the grid

    Notes
    -----
    The dose bisection assumes minimum-D-after-ramp is monotone increasing in
    per-session dose at a fixed interval. That holds for both feedback modes and
    with photodegradation on, since per-session yield
    d_sat * (1 - exp(-sed/d_sat)) is itself increasing in dose.

    The risk indices carry different evidence status on their two axes -- see
    vitd_physiology.cancer_risk. Rankings within one skin type are unaffected by
    the weaker (susceptibility) axis; cross-skin-type comparisons of the index
    VALUES are not.
    """
    if params is None:
        params = ModelParams()
    if fst not in range(1, 7):
        raise ValueError("fst must be 1-6")
    if not interval_grid:
        raise ValueError("interval_grid must be non-empty")

    med = _med_sed_by_fst(fst)
    sed_per_min = sed_from_minutes(1.0, uv_index, geometry_factor)

    # Dose ceiling: erythema cap (C1), optionally tightened by session length (C7).
    sed_cap_burn = GAMMA_MED * med
    if max_minutes is not None:
        sed_cap_time = max_minutes * sed_per_min
        sed_cap = min(sed_cap_burn, sed_cap_time)
        binding = "MED" if sed_cap_burn <= sed_cap_time else "minutes"
    else:
        sed_cap = sed_cap_burn
        binding = "MED"

    def run(interval: int, sed: float):
        days = list(range(0, horizon_days, interval))
        sched = [(float(d), float(sed)) for d in days]
        return simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                        params=params, D0=D0,
                        exposed_bsa_override=exposed_bsa,
                        dose_framing="fixed_per_session_SED")

    def min_D_after_ramp(r) -> float:
        t = np.asarray(r["t"])
        D = np.asarray(r["D"])
        mask = t >= ramp_in_days
        if not mask.any():
            return float("-inf")
        return float(np.min(D[mask]))

    evaluated = []
    for interval in interval_grid:
        interval = int(interval)
        if interval < 1:
            raise ValueError("interval_grid entries must be >= 1 day")

        # Feasibility at the dose ceiling. If the most this interval is allowed
        # to deliver still misses the floor, no dose on it can.
        r_cap = run(interval, sed_cap)
        if min_D_after_ramp(r_cap) < target:
            evaluated.append({
                "interval_days": interval,
                "n_sessions": len(range(0, horizon_days, interval)),
                "sed_per_session": sed_cap,
                "minutes_per_session": sed_cap / sed_per_min if sed_per_min > 0 else float("nan"),
                "binding_constraint": binding,
                "feasible": False,
                "within_risk_cap": False,
                "min_D_after_ramp": min_D_after_ramp(r_cap),
                "steady_state_D": steady_state_D(r_cap),
                "days_below_target": _days_below(r_cap, target),
                "total_sed": r_cap["cumulative_SED"],
                "scc_index": r_cap["cancer_risk"]["scc_index"],
                "intermittent_index": r_cap["cancer_risk"]["intermittent_index"],
                "combined_risk": r_cap["cancer_risk"]["combined_index"],
                "burn_fraction": r_cap["cancer_risk"]["burn_fraction"],
            })
            continue

        # Smallest dose that still clears the floor. Monotone, so bisect.
        lo, hi = 0.0, sed_cap
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if min_D_after_ramp(run(interval, mid)) < target:
                lo = mid
            else:
                hi = mid
        sed = hi
        r = run(interval, sed)
        risk = r["cancer_risk"]
        combined = risk["combined_index"]
        evaluated.append({
            "interval_days": interval,
            "n_sessions": len(range(0, horizon_days, interval)),
            "sed_per_session": sed,
            "minutes_per_session": sed / sed_per_min if sed_per_min > 0 else float("nan"),
            "binding_constraint": binding,
            "feasible": True,
            "within_risk_cap": bool(risk_cap is None or combined <= risk_cap),
            "min_D_after_ramp": min_D_after_ramp(r),
            "steady_state_D": steady_state_D(r),
            "days_below_target": _days_below(r, target),
            "total_sed": r["cumulative_SED"],
            "scc_index": risk["scc_index"],
            "intermittent_index": risk["intermittent_index"],
            "combined_risk": combined,
            "burn_fraction": risk["burn_fraction"],
        })

    admissible = [e for e in evaluated if e["feasible"] and e["within_risk_cap"]]
    note = None
    if not admissible:
        if any(e["feasible"] for e in evaluated):
            note = (f"sufficiency is reachable but every feasible interval exceeds "
                    f"the risk cap {risk_cap}; relax C9 or the session-length cap")
        else:
            note = (f"target {target:.0f} nmol/L is unreachable on this grid at "
                    f"<= {GAMMA_MED:.0%} MED"
                    + (f" and <= {max_minutes:.0f} min/session" if max_minutes else "")
                    + "; the constraint set, not the schedule, is what fails here")

    # Lowest risk wins; ties broken toward fewer sessions per year.
    recommended = (min(admissible, key=lambda e: (e["combined_risk"], e["n_sessions"]))
                   if admissible else None)

    return {
        "recommended": recommended,
        "evaluated": evaluated,
        "risk_cap": risk_cap,
        "target": target,
        "fst": fst,
        "sed_cap": sed_cap,
        "binding_constraint": binding,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Variable-dose optimisation
# ---------------------------------------------------------------------------
# Problem: minimise the combined cancer-risk index subject to the sufficiency
# floor after ramp-in (C3) and the erythema ceiling (C1), with the dose of every
# session free rather than shared.
#
# This is an optimal-control problem -- state (D, M, SC), control d(t) -- and it
# does NOT require mixed-integer machinery: d_i = 0 already lives in the
# continuous domain, so "no session that day" is just zero dose. Integrality
# would only be needed to impose a session-count cap or a fixed per-session
# cost, and even those are better handled by an L1 penalty on the dose vector
# than by branch and bound.
#
# Method: low-dimensional shape families searched globally, rather than a free
# per-session vector searched locally. This buys a schedule that can actually
# be described to a person, but it is NOT free: at the one operating point
# where it has been checked against a free per-day dose vector (FST III,
# N=104, horizon 365d -- the exact cell RESULTS.md 13c publishes at risk
# 1.3706), a 365-variable SLSQP solve warm-started from the parameterised
# answer reaches risk 1.1943, i.e. the parameterised family gives up 12.9%
# relative to it (80,067 forward evaluations; see
# scripts/joint_schedule_freesolve.py, which reproduces this number and is not
# part of run_all.py). Because that solve is local and warm-started rather
# than global, 12.9% is a LOWER bound on the true family-restriction gap, not
# an estimate of it. This has been measured at one (FST, N) cell only and
# should not be generalised to other cells without rerunning the script.
def _risk_of(fst: int, days: np.ndarray, seds: np.ndarray, horizon_days: int,
             D0: float, target: float, ramp_in_days: int,
             params: Optional[ModelParams],
             exposed_bsa: Optional[float] = None
             ) -> Tuple[float, float, float, float, dict]:
    """(combined risk, trough after ramp, total SED, time to sufficiency,
    full cancer_risk dict).

    The 5th element carries `scc_index` and `intermittent_index` separately.
    `combined_index` (element 0) is a geometric-mean convenience for
    single-objective ranking, not itself a literature-anchored quantity --
    see the docstring of `vitd_physiology.cancer_risk`. Callers that report
    risk changes to a reader should prefer the two components over the
    combined figure where both are available.
    """
    sched = build_variable_schedule(days, seds, fst=fst)
    if not sched:
        return (float("inf"), float("-inf"), 0.0, float("nan"),
                {"scc_index": float("nan"), "intermittent_index": float("nan")})
    r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                 params=params, D0=D0, exposed_bsa_override=exposed_bsa,
                 dose_framing="fixed_per_session_SED")
    t = np.asarray(r["t"]); D = np.asarray(r["D"])
    mask = t >= ramp_in_days
    trough = float(np.min(D[mask])) if mask.any() else float("-inf")
    return (r["cancer_risk"]["combined_index"], trough,
            float(r["cumulative_SED"]), time_to_sufficiency(r, target),
            r["cancer_risk"])


def _min_feasible(fst, days, make_seds, horizon_days, D0, target,
                  ramp_in_days, params, lo=0.0, hi=1.0, iters=28,
                  exposed_bsa: Optional[float] = None):
    """Smallest scale factor on a dose shape that still clears the floor.

    Bisection is valid because trough-after-ramp is monotone increasing in a
    uniform scaling of the whole profile: per-session yield is increasing in
    dose under both feedback modes and with photodegradation on.
    """
    seds_hi = make_seds(hi)
    _, trough_hi, _, _, _ = _risk_of(fst, days, seds_hi, horizon_days, D0, target,
                                     ramp_in_days, params, exposed_bsa)
    if trough_hi < target:
        return None                      # infeasible even at full scale
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, trough, _, _, _ = _risk_of(fst, days, make_seds(mid), horizon_days, D0,
                                      target, ramp_in_days, params, exposed_bsa)
        if trough < target:
            lo = mid
        else:
            hi = mid
    return hi


def optimise_variable_dose(fst: int,
                           n_sessions: int,
                           family: str = "two_phase",
                           horizon_days: int = 365,
                           D0: float = 25.0,
                           target: float = SUFFICIENCY_NMOL_L,
                           ramp_in_days: int = RAMP_IN_DAYS,
                           params: Optional[ModelParams] = None) -> Optional[dict]:
    """Lowest-risk dose profile in `family` that holds the sufficiency floor.

    Parameters
    ----------
    family : {'uniform', 'two_phase', 'exponential', 'seasonal', 'gap_coupled'}
        Shape family to search. 'uniform' is the constant-dose baseline every
        other analysis in this repo assumes.

    Returns None when no profile in the family clears the floor at the erythema
    ceiling -- reported rather than silently returning the closest miss, since
    "unreachable" is a real answer for dark skin at low session counts.

    Note
    ----
    Absolute 25(OH)D values here inherit the uncalibrated SC priors (`SC_max`,
    `alpha_SC`, `K_SC`); the *ranking* of profiles within one skin type is far
    more robust than the levels. Doses are absolute SED, not fractional MED --
    the two framings give materially different skin-type patterns.
    """
    days = session_days(n_sessions, horizon_days)
    cap = GAMMA_MED * _med_sed_by_fst(fst)
    if days.size == 0:
        return None

    def evaluate(seds, meta):
        risk, trough, total, ttf, _ = _risk_of(fst, days, seds, horizon_days, D0,
                                               target, ramp_in_days, params)
        if trough < target:
            return None
        # Report the doses actually DELIVERED, not the raw shape: the shape may
        # exceed the erythema ceiling and be clipped by build_variable_schedule
        # before it reaches the simulation. Reporting the unclipped shape would
        # claim a burning session that never happens, and would break any
        # downstream C1 check against these fields.
        delivered = np.array([s for _, s in
                              build_variable_schedule(days, seds, fst=fst)])
        return {"fst": fst, "n_sessions": n_sessions, "family": family,
                "combined_risk": risk, "trough_after_ramp": trough,
                "total_sed": total, "time_to_sufficiency": ttf,
                "n_sessions_delivered": int(delivered.size),
                "max_sed": float(np.max(delivered)) if delivered.size else 0.0,
                "min_sed": float(np.min(delivered)) if delivered.size else 0.0,
                **meta}

    best = None
    def keep(cand):
        nonlocal best
        if cand is not None and (best is None
                                 or cand["combined_risk"] < best["combined_risk"]):
            best = cand

    if family == "uniform":
        s = _min_feasible(fst, days, lambda x: dose_uniform(days, x * cap),
                          horizon_days, D0, target, ramp_in_days, params)
        if s is not None:
            keep(evaluate(dose_uniform(days, s * cap), {"dose_sed": s * cap}))

    elif family == "two_phase":
        # All three parameters are searched. Pinning the loading dose to the
        # ceiling looks harmless -- the fastest climb costs the fewest loading
        # sessions -- but it is not: a loading dose at the cap maximises the
        # per-session intensity that the intermittency term punishes cubically,
        # so the lowest-RISK loading dose is often below the ceiling even though
        # the fastest one is at it. Fixing d_load would hand the comparison to
        # the exponential family by construction rather than on the merits.
        for k in range(0, min(n_sessions, 61), 6):
            t_switch = days[k] if k < days.size else horizon_days + 1.0
            for load_frac in (0.5, 0.75, 1.0):
                def make(x, ts=t_switch, lf=load_frac):
                    return dose_two_phase(days, lf * cap, x * cap, ts)
                s = _min_feasible(fst, days, make, horizon_days, D0, target,
                                  ramp_in_days, params, iters=23)
                if s is None:
                    continue
                keep(evaluate(make(s),
                              {"n_loading": k, "t_switch": float(t_switch),
                               "d_load_sed": load_frac * cap,
                               "d_maint_sed": s * cap}))

    elif family == "exponential":
        for ratio in (0.5, 1.0, 2.0, 4.0, 8.0):
            for tau in (7.0, 14.0, 30.0, 60.0, 120.0):
                def make(x, r=ratio, tv=tau):
                    return dose_exponential(days, x * cap, r, tv)
                s = _min_feasible(fst, days, make, horizon_days, D0, target,
                                  ramp_in_days, params)
                if s is None:
                    continue
                keep(evaluate(make(s), {"ratio": ratio, "tau": tau,
                                        "d_inf_sed": s * cap}))

    elif family == "seasonal":
        for amp in (0.0, 0.2, 0.4, 0.6):
            def make(x, a=amp):
                return dose_seasonal(days, x * cap, a)
            s = _min_feasible(fst, days, make, horizon_days, D0, target,
                              ramp_in_days, params)
            if s is None:
                continue
            keep(evaluate(make(s), {"amp": amp, "d0_sed": s * cap}))

    elif family == "gap_coupled":
        for p in (0.0, 0.5, 1.0, 1.5):
            def make(x, pv=p):
                return dose_gap_coupled(days, x * cap, pv)
            s = _min_feasible(fst, days, make, horizon_days, D0, target,
                              ramp_in_days, params)
            if s is None:
                continue
            keep(evaluate(make(s), {"p": p, "d_ref_sed": s * cap}))

    else:
        raise ValueError(f"unknown family {family!r}")

    return best


VARIABLE_DOSE_FAMILIES = ("uniform", "two_phase", "exponential",
                          "seasonal", "gap_coupled")


# ---------------------------------------------------------------------------
# Joint timing + dose optimisation
# ---------------------------------------------------------------------------
# Everything above fixes the session DAYS (uniformly spaced) and searches only
# the doses. This searches both at once.
#
# WHY A SESSION BUDGET IS PART OF THE PROBLEM, NOT AN ADD-ON. Without one the
# joint optimum is degenerate. The SCC term depends only on cumulative dose, and
# at fixed cumulative dose the intermittency term sum (d_i/MED)^3 is minimised
# by spreading over as many days as possible -- so the risk objective always
# prefers more, smaller sessions, and the unconstrained optimum is "dose every
# single day". That is exactly what optimal_interval_risk_constrained already
# reports (every skin type lands on the 1-day interval), and it is reproduced
# here: risk falls monotonically in session count all the way to 365. The
# interesting problem is therefore the FRONTIER over a session budget, which is
# also the constraint people actually face -- adherence, not photobiology.
#
# WHY NOT MIXED-INTEGER. d_i = 0 already means "no session", so a sparse dose
# vector encodes timing and dose together with no integer variables. This is the
# maximum-hands-off control idea: minimising the number of active control
# instants is an L0 problem, and Nagahara, Quevedo & Nesic (2016, IEEE TAC
# 61:735-747) show the L1-optimal control coincides with the L0-optimal one
# under a normality condition, with reweighting (Candes, Wakin & Boyd 2008,
# J Fourier Anal Appl 14:877-905) tightening the approximation. That equivalence
# is proved for linear dynamics and is NOT established for this nonlinear
# system, so it motivates the formulation but does not guarantee it --
# scripts/joint_schedule_freesolve.py checks the low-dimensional answer against
# a direct sparse solve rather than assuming it.
#
# SPACING FAMILY. Session days are warped by a single monotone exponent:
#     t_i = T * (i / N) ** gamma
# gamma = 1 is uniform, gamma > 1 front-loads (sessions bunch early), gamma < 1
# back-loads. One parameter, and it nests the uniform baseline exactly, so the
# search cannot lose to uniform spacing by being unable to express it.
def warp_days(N: int, T_days: int, gamma: float = 1.0) -> np.ndarray:
    """Monotone power warp of session days. gamma = 1 recovers uniform spacing."""
    if N <= 0:
        return np.zeros(0)
    i = np.arange(N, dtype=float)
    return float(T_days) * (i / float(N)) ** float(gamma)


JOINT_DOSE_FAMILIES = ("two_phase", "exponential")


def _joint_seds(days: np.ndarray, cap: float, family: str,
                scale: float, a: float, b: float,
                horizon_days: int) -> np.ndarray:
    """Dose profile for the joint search, in absolute SED."""
    if family == "exponential":
        return dose_exponential(days, scale * cap, a, b)
    if family == "two_phase":
        return dose_two_phase(days, a * cap, scale * cap, b * horizon_days)
    raise ValueError(f"unknown joint dose family {family!r}")


def optimise_joint_schedule(fst: int,
                            n_sessions: int,
                            horizon_days: int = 365,
                            D0: float = 25.0,
                            target: float = SUFFICIENCY_NMOL_L,
                            ramp_in_days: int = RAMP_IN_DAYS,
                            free_spacing: bool = True,
                            families: Sequence[str] = JOINT_DOSE_FAMILIES,
                            params: Optional[ModelParams] = None,
                            seed: int = 0,
                            maxiter: int = 18,
                            popsize: int = 10,
                            n_seeds: int = 1) -> Optional[dict]:
    """Jointly optimise session timing and per-session dose at a session budget.

    Minimises the combined cancer-risk index subject to the sufficiency floor
    after ramp-in (C3) and the erythema ceiling (C1), searching the spacing
    exponent and the dose profile together by differential evolution.

    Parameters
    ----------
    free_spacing : bool
        When False the spacing exponent is pinned to 1.0 (uniform spacing) and
        only the dose profile is searched. This is the CONTROL arm: it isolates
        what free spacing buys from what a better dose search buys, by holding
        the solver, its budget and the dose families identical. Comparing a
        joint search against a coarse grid search instead would confound the
        two.
    n_seeds : int
        Number of DE seeds to replicate the search over, starting at `seed`
        and incrementing by 1 (`seed`, `seed + 1`, ..., `seed + n_seeds - 1`).
        Default 1 reproduces the previous single-seed behaviour exactly, so no
        existing caller changes. `differential_evolution` here terminates on
        `maxiter` every time rather than on `tol` (measured: `nit == maxiter`
        in every run checked), and direct replication at FST III / N=146 spans
        6.3% in `combined_risk` across 5 seeds with the winning dose family
        itself changing by seed [reference purged 2026-09-19] --
        so a single seed-0 run is a draw from a distribution several percent
        wide, not a repeatable point estimate. When `n_seeds > 1` the returned
        dict gains `risk_min`, `risk_max`, `risk_spread_frac` (the range as a
        fraction of `risk_min`) and `n_seeds_run`, computed over whichever
        seeds produced a feasible schedule at all; `combined_risk` and the
        rest of the fields describe the single best-of-n_seeds result, exactly
        as they would for a single successful seed. Paired free-vs-uniform
        gains replicate in SIGN (5/5 positive across the seeds checked in the
        review) but not reliably in MAGNITUDE (range +3.0% to +12.1% at one
        cell, mean +6.8%, sd 3.4 pp) -- so treat any single-cell percentage
        gain from this function as order-of-magnitude only, and do not compare
        magnitudes below roughly 10 percentage points across different (FST,
        N) cells.

    Returns None if no schedule in the family clears the floor in any seed.

    Notes
    -----
    The floor is enforced by penalty during the global search and then by an
    exact bisection on the overall dose level of the winning shape, so every
    returned schedule sits ON the floor rather than near it and risk figures are
    compared at matched efficacy.

    Doses are absolute SED. Absolute 25(OH)D inherits the uncalibrated SC priors
    (`SC_max`, `alpha_SC`, `K_SC`).
    """
    from scipy.optimize import differential_evolution

    cap = GAMMA_MED * _med_sed_by_fst(fst)
    if n_sessions <= 0:
        return None
    PEN = 10.0

    def build(family, x):
        scale, a, b = x[0], x[1], x[2]
        gamma = x[3] if free_spacing else 1.0
        days = warp_days(n_sessions, horizon_days, gamma)
        return days, _joint_seds(days, cap, family, scale, a, b, horizon_days), gamma

    def score(family, x):
        days, seds, _ = build(family, x)
        risk, trough, _, _, _ = _risk_of(fst, days, seds, horizon_days, D0,
                                         target, ramp_in_days, params)
        if not np.isfinite(risk):
            return 1e6
        return risk + PEN * max(0.0, target - trough) ** 2

    def _solve_once(seed_val: int) -> Optional[dict]:
        """One full DE solve (all `families`) at a single seed."""
        best = None
        for family in families:
            if family == "exponential":
                bounds = [(0.02, 1.0), (0.0, 8.0), (5.0, 180.0)]
            elif family == "two_phase":
                bounds = [(0.02, 1.0), (0.1, 1.0), (0.0, 0.6)]
            else:
                raise ValueError(f"unknown joint dose family {family!r}")
            # The spacing dimension is present in BOTH arms and merely inert in
            # the control (build() pins gamma to 1.0). differential_evolution
            # sizes its population as popsize * n_parameters, so giving the
            # free arm an extra dimension would also hand it a ~33% larger
            # search budget -- the gain would then be partly attributable to
            # search effort rather than to spacing, which is the one thing
            # this comparison exists to isolate.
            bounds = bounds + [(0.4, 2.5)]

            res = differential_evolution(lambda x, f=family: score(f, x),
                                         bounds=bounds, seed=seed_val,
                                         maxiter=maxiter, popsize=popsize,
                                         tol=1e-6, polish=True)

            # Land exactly on the floor: bisect the overall dose LEVEL of the
            # winning shape. Monotone, because per-session yield is increasing
            # in dose.
            _, _, gamma = build(family, res.x)
            days = warp_days(n_sessions, horizon_days, gamma)
            shape = _joint_seds(days, cap, family, res.x[0], res.x[1], res.x[2],
                                horizon_days)
            base = float(np.max(shape))
            if base <= 0:
                continue

            def make(mult, sh=shape, bs=base):
                return sh * (mult * cap / bs)

            s = _min_feasible(fst, days, make, horizon_days, D0, target,
                              ramp_in_days, params, iters=30)
            if s is None:
                continue
            seds = make(s)
            risk, trough, total, ttf, cr = _risk_of(fst, days, seds, horizon_days,
                                                    D0, target, ramp_in_days, params)
            if trough < target - 1e-6:
                continue
            delivered = np.array([v for _, v in
                                  build_variable_schedule(days, seds, fst=fst)])
            cand = {
                "fst": fst, "n_sessions": n_sessions, "family": family,
                "free_spacing": free_spacing, "gamma": float(gamma),
                "combined_risk": risk, "trough_after_ramp": trough,
                "scc_index": cr["scc_index"],
                "intermittent_index": cr["intermittent_index"],
                "total_sed": total, "time_to_sufficiency": ttf,
                "n_sessions_delivered": int(delivered.size),
                "max_sed": float(np.max(delivered)) if delivered.size else 0.0,
                "min_sed": float(np.min(delivered)) if delivered.size else 0.0,
                "first_gap_days": float(days[1] - days[0]) if days.size > 1 else float("nan"),
                "last_gap_days": float(days[-1] - days[-2]) if days.size > 1 else float("nan"),
                # Winning parameters, stored as scalars so the exact schedule
                # can be rebuilt by rebuild_joint_schedule() without
                # re-running the search.
                "p_scale": float(res.x[0]), "p_a": float(res.x[1]),
                "p_b": float(res.x[2]), "p_level": float(s),
                "seed": int(seed_val),
            }
            if best is None or cand["combined_risk"] < best["combined_risk"]:
                best = cand
        return best

    seeds = [seed + i for i in range(max(1, int(n_seeds)))]
    results = [r for r in (_solve_once(sv) for sv in seeds) if r is not None]
    if not results:
        return None

    best = min(results, key=lambda r: r["combined_risk"])
    if n_seeds > 1:
        risks = [r["combined_risk"] for r in results]
        risk_min, risk_max = min(risks), max(risks)
        best = dict(best)
        best["risk_min"] = risk_min
        best["risk_max"] = risk_max
        best["risk_spread_frac"] = ((risk_max - risk_min) / risk_min
                                     if risk_min > 0 else float("nan"))
        best["n_seeds_run"] = len(results)
    return best


def rebuild_joint_schedule(res: dict, horizon_days: int = 365
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Exact (days, seds) for a result from optimise_joint_schedule.

    Reconstructs from the stored winning parameters rather than re-running the
    stochastic search, so downstream analysis sees the schedule that was
    actually scored.
    """
    fst = int(res["fst"])
    cap = GAMMA_MED * _med_sed_by_fst(fst)
    days = warp_days(int(res["n_sessions"]), horizon_days, float(res["gamma"]))
    shape = _joint_seds(days, cap, res["family"], float(res["p_scale"]),
                        float(res["p_a"]), float(res["p_b"]), horizon_days)
    peak = float(np.max(shape))
    if peak <= 0:
        return days, np.zeros_like(days)
    return days, shape * (float(res["p_level"]) * cap / peak)




# ===========================================================================
# TIER 1 (PRIMARY). Minimum dose / minimum sessions to REACH AND HOLD
# sufficiency.
# ===========================================================================
# This is the headline objective declared in
# vitd_objectives.OBJECTIVE_HIERARCHY, and it is the question a person actually
# asks: not "which schedule is most efficient per SED" (a ratio with no floor,
# maximised by dosing nothing) but "what is the least I have to do to get to
# 50 nmol/L and then stay there".
#
# Two minima, because they are two different questions and they need not land
# on the same schedule:
#   * FEWEST SESSIONS  -- the adherence question. Smallest N on the budget grid
#                         at which any admissible schedule exists.
#   * LEAST TOTAL DOSE -- the UV-exposure question. Smallest cumulative SED
#                         across all admissible budgets.
# Fewer sessions forces a larger per-session dose, and per-session yield
# saturates (d_sat * (1 - exp(-sed/d_sat))), so the fewest-sessions answer
# generally costs MORE total UV than the least-dose answer. Reporting only one
# of them hides that trade.
#
# Constraints are the standing ones, nothing relaxed:
#   C3  min D(t >= ramp_in_days) >= target
#   C1  per-session dose <= GAMMA_MED * MED(FST)
#   C7  per-session dose <= what max_minutes delivers at uv_index
# An INFEASIBLE (FST, budget) cell is a result, not a hole in the table: it
# says this skin type cannot hold sufficiency on sun alone at that budget
# without burning or without a session longer than anyone will sit through.
def minimum_schedule_for_sufficiency(
        fst: int,
        session_budgets: Sequence[int] = (52, 104, 146, 183, 260, 365),
        horizon_days: int = 365,
        D0: float = 25.0,
        target: float = SUFFICIENCY_NMOL_L,
        ramp_in_days: int = RAMP_IN_DAYS,
        uv_index: float = 6.0,
        max_minutes: Optional[float] = PRACTICAL_MAX_MINUTES,
        gammas: Sequence[float] = (1.0,),
        params: Optional[ModelParams] = None,
        exposed_bsa: Optional[float] = None,
        geometry_factor: Optional[float] = None) -> dict:
    """Least dose and fewest sessions that reach and hold `target`.

    Parameters
    ----------
    uv_index, max_minutes
        Constraint C7. `max_minutes` None drops C7 and leaves only the
        erythema ceiling; the default is the practical session ceiling from
        vitd_objectives, which is what makes FST V-VI infeasibility bind.
    exposed_bsa : float, optional
        Exposed body-surface fraction passed to `simulate(exposed_bsa_override=...)`.
        None uses the model default (BSA_REF).
    geometry_factor : float, optional
        Horizontal-to-personal geometry correction passed to
        `sed_from_minutes`. None falls back to the flat 0.35 Cockell scalar.
        See `optimal_interval_risk_constrained`'s docstring for why this is
        not redundant with `exposed_bsa`.
    gammas
        Spacing exponents to try at each budget (`warp_days`; 1.0 is uniform).
        The default is uniform ONLY, which is the plain "go outside every N
        days" schedule and the condition under which
        tests/test_vitd_optimization.py::TestRecalibratedFeasibility and
        scripts/feasibility_map_kconst.py state their minimum budgets.
        Passing a wider grid lowers the minimum budget at some skin types, and
        it lowers it for a specific reason worth reporting rather than hiding:
        the C3 floor binds at the RAMP-IN BOUNDARY, not at the end of the year,
        so a schedule that reaches target faster can hold a budget that uniform
        spacing misses by a fraction of a nmol/L.

    Returns
    -------
    dict with keys
        rows                every budget on the grid, feasible or not, each
                            carrying `binding_constraint` when infeasible
        fewest_sessions     smallest feasible budget (None if none is)
        least_total_dose    feasible budget with the smallest cumulative SED
        feasible            whether ANY budget on the grid works
        note                which constraint eliminated the grid, when empty

    Notes
    -----
    Doses are absolute SED, not fractional MED. Absolute 25(OH)D inherits the
    uncalibrated SC priors (`SC_max`, `alpha_SC`, `K_SC`); the feasibility
    ORDERING across skin types is far more robust than the levels.
    """
    if fst not in range(1, 7):
        raise ValueError("fst must be 1-6")
    if not session_budgets:
        raise ValueError("session_budgets must be non-empty")

    cap_burn = GAMMA_MED * _med_sed_by_fst(fst)
    sed_per_min = sed_from_minutes(1.0, uv_index, geometry_factor)
    cap_time = (max_minutes * sed_per_min if max_minutes is not None
                else float("inf"))
    cap = min(cap_burn, cap_time)

    rows = []
    for n in sorted(int(n) for n in session_budgets):
        if n < 1:
            raise ValueError("session_budgets entries must be >= 1")

        best = None
        for gamma in gammas:
            days = warp_days(n, horizon_days, float(gamma))

            def make(x, dd=days, c=cap):
                return dose_uniform(dd, x * c)

            s = _min_feasible(fst, days, make, horizon_days, D0, target,
                              ramp_in_days, params, exposed_bsa=exposed_bsa)
            if s is None:
                continue
            seds = make(s)
            risk, trough, total, ttf, _ = _risk_of(fst, days, seds, horizon_days,
                                                   D0, target, ramp_in_days, params,
                                                   exposed_bsa)
            # Smallest total UV wins the budget; ties to the tighter spacing.
            key = (total, max_gap_days(days, horizon_days))
            if best is None or key < best[0]:
                best = (key, gamma, days, s, seds, risk, trough, total, ttf)

        if best is None:
            # Attribute the failure. If the erythema ceiling alone would have
            # allowed it, the session-length ceiling is what actually bound.
            binding = f"C1 erythema ({GAMMA_MED:.0%} MED)"
            if cap_time < cap_burn:
                for gamma in gammas:
                    dd = warp_days(n, horizon_days, float(gamma))
                    if _min_feasible(fst, dd,
                                     lambda x, d2=dd: dose_uniform(d2, x * cap_burn),
                                     horizon_days, D0, target, ramp_in_days,
                                     params, exposed_bsa=exposed_bsa) is not None:
                        binding = f"C7 session length (>{max_minutes:.0f} min)"
                        break
            rows.append({
                "fst": fst, "n_sessions": n, "feasible": False,
                "binding_constraint": binding, "gamma": float("nan"),
                "max_gap_days": float("nan"),
                "sed_per_session": float("nan"),
                "minutes_per_session": float("nan"),
                "total_sed": float("nan"), "trough_after_ramp": float("nan"),
                "time_to_sufficiency": float("nan"),
                "combined_risk": float("nan"),
                "scc_index": float("nan"),
                "intermittent_index": float("nan"),
            })
            continue

        _, gamma, days, s, seds, risk, trough, total, ttf = best
        sched = build_variable_schedule(days, seds, fst=fst)
        r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                     params=params, D0=D0, exposed_bsa_override=exposed_bsa,
                     dose_framing="fixed_per_session_SED")
        cr = r["cancer_risk"]
        rows.append({
            "fst": fst, "n_sessions": n, "feasible": True,
            "binding_constraint": "", "gamma": float(gamma),
            "max_gap_days": max_gap_days(days, horizon_days),
            "sed_per_session": float(s * cap),
            "minutes_per_session": (float(s * cap / sed_per_min)
                                    if sed_per_min > 0 else float("nan")),
            "total_sed": total, "trough_after_ramp": trough,
            "time_to_sufficiency": ttf,
            "combined_risk": risk,
            "scc_index": cr["scc_index"],
            "intermittent_index": cr["intermittent_index"],
        })

    ok = [r for r in rows if r["feasible"]]
    note = None
    if not ok:
        binding = {r["binding_constraint"] for r in rows}
        note = (f"FST {fst}: no session budget on the grid holds {target:.0f} "
                f"nmol/L within the standing constraints ({'; '.join(sorted(binding))}). "
                f"This is a result: the constraint set, not the schedule, is "
                f"what fails.")
    return {
        "fst": fst,
        "rows": rows,
        "feasible": bool(ok),
        "fewest_sessions": min(ok, key=lambda r: r["n_sessions"]) if ok else None,
        "least_total_dose": min(ok, key=lambda r: r["total_sed"]) if ok else None,
        "sed_cap": cap,
        "cap_binding": "MED" if cap_burn <= cap_time else "minutes",
        "target": target,
        "note": note,
    }


# ===========================================================================
# TIER 2 (SECONDARY). The tri-objective: best SED at least spacing, fewest
# sessions, accounting for skin cancer.
# ===========================================================================
# Three axes, all minimised, over the schedules that already satisfy the
# primary feasibility test (so "best SED" is a CONSTRAINT here -- every point
# on this frontier reaches and holds sufficiency -- and the reported
# `total_sed` is the dose that constraint actually cost):
#
#   A1  n_sessions      how often you have to go outside          (adherence)
#   A2  max_gap_days    the longest you ever go without a session (spacing)
#   A3  combined_risk   the skin-cancer index                     (harm)
#
# WHY THESE THREE CANNOT BE COLLAPSED. Over a fixed horizon the three axes are
# in genuine three-way conflict, and each pair conflicts for a different
# reason:
#   A1 vs A2  uniform spacing minimises the largest gap (see session_days), so
#             at a fixed budget the shortest-gap schedule is gamma = 1 exactly;
#             cutting sessions lengthens every gap proportionally.
#   A1 vs A3  fewer sessions must carry a larger per-session dose, and the
#             intermittency term is CUBIC in per-session dose.
#   A2 vs A3  the risk-minimising schedule front-loads (gamma > 1, section
#             13c), which shortens the early gaps by lengthening the late ones
#             -- so buying risk with spacing works in one direction only.
# A weighted sum would hide all three of those; the frontier shows them.
#
# [CHOICE] `max_gap_days` includes the tail gap from the last session to the
# end of the horizon. A front-loaded schedule that stops in October has a real
# four-month gap, and excluding it would let gamma > 1 look free on the spacing
# axis when it is not.
# [CHOICE] The spacing family is the same one-parameter monotone warp used in
# section 13c (`warp_days`), so the tri-objective and the 13c comparison are
# searching the same space and their answers are directly comparable.
TRI_GAMMA_GRID = (0.7, 1.0, 1.4, 2.0)
TRI_SESSION_BUDGETS = (52, 104, 146, 183, 260, 365)


def max_gap_days(days: np.ndarray, horizon_days: int) -> float:
    """Longest stretch with no session, including the tail to the horizon."""
    days = np.asarray(days, dtype=float)
    if days.size == 0:
        return float(horizon_days)
    gaps = np.diff(days) if days.size > 1 else np.zeros(0)
    tail = float(horizon_days) - float(days[-1])
    return float(max(gaps.max() if gaps.size else 0.0, tail, float(days[0])))


def _min_risk_on_fixed_days(fst: int, days: np.ndarray, cap: float,
                            horizon_days: int, D0: float, target: float,
                            ramp_in_days: int,
                            params: Optional[ModelParams]) -> Optional[dict]:
    """Lowest-risk feasible dose profile on a FIXED set of session days.

    A small deterministic shape grid rather than differential evolution: the
    days are already fixed here, so the remaining search is 1-2 dimensional and
    a grid plus the exact level bisection lands on the floor without the
    stochastic solver's run-to-run spread. That matters because this frontier
    is read as a comparison ACROSS cells, and DE noise (bounded at ~4% by
    section 13c) would swamp the differences between adjacent budgets.
    """
    n = int(np.asarray(days).size)
    if n == 0 or cap <= 0:
        return None

    cands = [("uniform", lambda x, dd=days, c=cap: dose_uniform(dd, x * c), {})]
    for k in (max(1, n // 12), max(1, n // 6)):
        if k >= n:
            continue
        t_switch = float(days[k])
        for load_frac in (0.6, 1.0):
            cands.append((
                "two_phase",
                lambda x, dd=days, c=cap, ts=t_switch, lf=load_frac:
                    dose_two_phase(dd, lf * c, x * c, ts),
                {"n_loading": k, "t_switch": t_switch, "load_frac": load_frac},
            ))

    best = None
    for family, make, meta in cands:
        s = _min_feasible(fst, days, make, horizon_days, D0, target,
                          ramp_in_days, params, iters=24)
        if s is None:
            continue
        seds = make(s)
        risk, trough, total, ttf, _ = _risk_of(fst, days, seds, horizon_days, D0,
                                               target, ramp_in_days, params)
        if trough < target - 1e-6:
            continue
        delivered = np.array([v for _, v in
                              build_variable_schedule(days, seds, fst=fst)])
        cand = {"family": family, "combined_risk": risk,
                "trough_after_ramp": trough, "total_sed": total,
                "time_to_sufficiency": ttf,
                "peak_sed": float(np.max(delivered)) if delivered.size else 0.0,
                "n_sessions_delivered": int(delivered.size), **meta}
        if best is None or cand["combined_risk"] < best["combined_risk"]:
            best = cand
    return best


# ---------------------------------------------------------------------------
# A3 axis: the cross-skin-type guard.
# ---------------------------------------------------------------------------
# WHY THIS TYPE EXISTS. `calibration_approval.yaml` (entry "A3 primary
# scheduling metric", field `not_covered`) records the blocker this workstream
# had to carry into the output: the FST susceptibility factor inside
# `vitd_physiology.cancer_risk` is a derived constitutive prior, and
# cross-skin-type comparisons of risk index VALUES are not anchored by any
# epidemiological source -- Elwood & Jopson 1997's odds ratios are ADJUSTED FOR
# skin pigmentation and so structurally cannot supply the missing contrast.
#
# Documenting that in a docstring is not enough: a float in a dict gets copied
# into a DataFrame, a CSV and then a sentence comparing FST II with FST VI.
# So on this objective the risk axis is not a float. It is `WithinFSTRisk`,
# which refuses `float()`, refuses numeric formatting, refuses arithmetic, and
# refuses ordering against an instance carrying a different skin type. Every
# within-FST use the frontier actually needs -- Pareto domination, ranking,
# "is this cell better than that one for the same person" -- still works,
# because all of those compare cells inside one skin type. The number that
# reaches the CSV is `risk_within_fst_ratio`, normalised to that skin type's
# own best feasible cell, so it is 1.0 at the minimum FOR EVERY SKIN TYPE and
# therefore carries no cross-FST magnitude to misread in the first place.
class CrossSkinTypeRiskError(TypeError):
    """Raised when a sealed risk index is used in a cross-skin-type way."""


class WithinFSTRisk:
    """A skin-cancer risk index sealed to one Fitzpatrick skin type.

    Comparable and rankable against other indices for the SAME skin type.
    Not convertible to a number, not formattable as one, and not comparable
    across skin types. The escape hatch is `absolute_value_for_fst(fst)`,
    which requires the caller to name the skin type it already believes it is
    holding, so a cross-FST read cannot happen by accident.
    """

    __slots__ = ("_fst", "_raw", "_reference")

    def __init__(self, fst: int, raw: float, reference: float):
        self._fst = int(fst)
        self._raw = float(raw)
        self._reference = float(reference)

    # -- the safe, self-normalised view -------------------------------------
    @property
    def fst(self) -> int:
        return self._fst

    @property
    def within_fst_ratio(self) -> float:
        """This cell's risk divided by the best feasible cell for this FST.

        1.0 at that skin type's own minimum, by construction, for every skin
        type. Comparing this number across skin types is meaningless rather
        than misleading -- there is no magnitude in it to misread.
        """
        if not np.isfinite(self._raw):
            return float("nan")
        if not np.isfinite(self._reference) or self._reference <= 0:
            return float("nan")
        return self._raw / self._reference

    def absolute_value_for_fst(self, fst: int) -> float:
        """Unsealed absolute index. UNANCHORED ACROSS SKIN TYPES.

        Provided for within-FST diagnostics and tests. `fst` must match the
        skin type this index was sealed with.
        """
        if int(fst) != self._fst:
            raise CrossSkinTypeRiskError(
                f"risk index is sealed to FST {self._fst}; asked for FST "
                f"{int(fst)}. Cross-skin-type risk VALUES are not anchored "
                f"(Elwood & Jopson 1997's ORs are pigmentation-adjusted; no "
                f"cross-FST comparison is licensed).")
        return self._raw

    # -- refusals ------------------------------------------------------------
    def _peer(self, other, op: str) -> float:
        if not isinstance(other, WithinFSTRisk):
            raise CrossSkinTypeRiskError(
                f"cannot {op} a sealed risk index against "
                f"{type(other).__name__}; use .within_fst_ratio for a plain "
                f"number, which is normalised within one skin type.")
        if other._fst != self._fst:
            raise CrossSkinTypeRiskError(
                f"cannot {op} risk across skin types (FST {self._fst} vs FST "
                f"{other._fst}). The FST susceptibility factor is a "
                f"constitutive prior and cross-skin-type risk VALUES are "
                f"unanchored; only within-skin-type comparisons are usable.")
        return other._raw

    def __lt__(self, other): return self._raw < self._peer(other, "order")
    def __le__(self, other): return self._raw <= self._peer(other, "order")
    def __gt__(self, other): return self._raw > self._peer(other, "order")
    def __ge__(self, other): return self._raw >= self._peer(other, "order")

    def __eq__(self, other):
        if not isinstance(other, WithinFSTRisk):
            return NotImplemented
        return self._raw == self._peer(other, "compare")

    __hash__ = object.__hash__

    def __float__(self):
        raise CrossSkinTypeRiskError(
            "refusing to hand out a bare float for a skin-cancer risk index: "
            "a loose float is exactly how a cross-skin-type value gets into a "
            "table. Use .within_fst_ratio (normalised inside one skin type) "
            "or .absolute_value_for_fst(fst).")

    __int__ = __float__
    __index__ = __float__

    def _no_arith(self, *_a, **_k):
        raise CrossSkinTypeRiskError(
            "arithmetic on a sealed risk index is refused; it would produce a "
            "number with no skin type attached.")

    __add__ = __radd__ = __sub__ = __rsub__ = _no_arith
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _no_arith

    def __format__(self, spec: str) -> str:
        if spec in ("", "s"):
            return repr(self)
        raise CrossSkinTypeRiskError(
            f"numeric format spec {spec!r} refused on a sealed risk index; "
            f"format .within_fst_ratio instead.")

    def __repr__(self) -> str:
        r = self.within_fst_ratio
        shown = "nan" if not np.isfinite(r) else f"{r:.4f}"
        return f"<risk FST {self._fst}: {shown}x that FST's own minimum>"


# WHY A SESSION BUDGET IS AN ASSUMPTION AND NOT A FINDING. Without a budget the
# tri-objective is degenerate: the SCC term depends only on cumulative dose and
# the intermittency term is cubic in per-session dose, so at fixed cumulative
# dose risk falls monotonically in session count all the way to 365 (the same
# degeneracy documented at vitd_optimization.py section 13c and reproduced by
# optimal_interval_risk_constrained, which lands every skin type on the 1-day
# interval). The budget that makes the problem well-posed is therefore a
# statement about how often a person will actually go outside. It is not
# photobiology and no measurement in this repo supports any particular value.
SESSION_BUDGET_BASIS: str = (
    "[ASSUMPTION, NOT PHOTOBIOLOGY] The session budget is an ADHERENCE "
    "constraint. Without it this objective is degenerate: risk falls "
    "monotonically in session count to 365 (dose every day). No source in "
    "this repo fixes a budget; the grid is swept so the assumption is visible "
    "as an axis rather than buried as a default."
)


def pareto_front_nd(points: Sequence[dict], keys: Sequence[str]) -> List[dict]:
    """Nondominated set over any number of MINIMISED objectives.

    Objective values need only be mutually orderable, not numeric, so that the
    sealed `WithinFSTRisk` axis can be a Pareto objective without ever being
    coerced to a float.
    """
    keys = tuple(keys)
    front = []
    for p in points:
        pv = tuple(p[k] for k in keys)
        dominated = False
        for q in points:
            if q is p:
                continue
            qv = tuple(q[k] for k in keys)
            if all(a <= b for a, b in zip(qv, pv)) and any(a < b for a, b in zip(qv, pv)):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def tri_objective_frontier(
        fst: int,
        session_budgets: Sequence[int] = TRI_SESSION_BUDGETS,
        gammas: Sequence[float] = TRI_GAMMA_GRID,
        horizon_days: int = 365,
        D0: float = 25.0,
        target: float = SUFFICIENCY_NMOL_L,
        ramp_in_days: int = RAMP_IN_DAYS,
        uv_index: float = 6.0,
        max_minutes: Optional[float] = PRACTICAL_MAX_MINUTES,
        params: Optional[ModelParams] = None) -> dict:
    """Sessions x spacing x cancer-risk frontier, at held sufficiency.

    Every point returned holds the sufficiency floor after ramp-in (C3) at a
    non-burning dose (C1) deliverable inside `max_minutes` (C7). Infeasible
    (budget, gamma) cells are RETURNED, flagged, not dropped.
    """
    if fst not in range(1, 7):
        raise ValueError("fst must be 1-6")

    cap_burn = GAMMA_MED * _med_sed_by_fst(fst)
    sed_per_min = sed_from_minutes(1.0, uv_index)
    cap = min(cap_burn, max_minutes * sed_per_min
              if max_minutes is not None else float("inf"))

    rows = []
    for n in sorted(int(v) for v in session_budgets):
        for gamma in gammas:
            days = warp_days(n, horizon_days, float(gamma))
            res = _min_risk_on_fixed_days(fst, days, cap, horizon_days, D0,
                                          target, ramp_in_days, params)
            base = {"fst": fst, "n_sessions": n, "gamma": float(gamma),
                    "max_gap_days": max_gap_days(days, horizon_days)}
            if res is None:
                rows.append({**base, "feasible": False,
                             "family": "", "_raw_risk": float("nan"),
                             "total_sed": float("nan"),
                             "peak_sed": float("nan"),
                             "minutes_per_session": float("nan"),
                             "trough_after_ramp": float("nan"),
                             "time_to_sufficiency": float("nan"),
                             "on_frontier": False})
                continue
            rows.append({**base, "feasible": True,
                         "family": res["family"],
                         "_raw_risk": res["combined_risk"],
                         "total_sed": res["total_sed"],
                         "peak_sed": res["peak_sed"],
                         "minutes_per_session": (res["peak_sed"] / sed_per_min
                                                 if sed_per_min > 0 else float("nan")),
                         "trough_after_ramp": res["trough_after_ramp"],
                         "time_to_sufficiency": res["time_to_sufficiency"],
                         "on_frontier": False})

    feas = [r for r in rows if r["feasible"]]

    # Seal the risk axis to this skin type before anything can read it. The
    # normalising reference is THIS skin type's own best feasible cell, so the
    # only number that leaves here on the risk axis is 1.0-at-the-minimum for
    # every FST -- see WithinFSTRisk above.
    raw = [r["_raw_risk"] for r in feas
           if np.isfinite(r["_raw_risk"])]
    reference = min(raw) if raw else float("nan")
    order = sorted(range(len(feas)), key=lambda i: feas[i]["_raw_risk"])
    rank_of = {id(feas[i]): k + 1 for k, i in enumerate(order)}
    for r in rows:
        sealed = WithinFSTRisk(fst, r.pop("_raw_risk"), reference)
        r["combined_risk"] = sealed
        r["risk_within_fst_ratio"] = sealed.within_fst_ratio
        r["risk_rank_within_fst"] = rank_of.get(id(r), 0)

    front = pareto_front_nd(feas, ("n_sessions", "max_gap_days", "combined_risk"))
    for r in front:
        r["on_frontier"] = True
    front = sorted(front, key=lambda r: (r["n_sessions"], r["max_gap_days"]))
    return {"fst": fst, "rows": rows, "frontier": front,
            "feasible": bool(feas), "sed_cap": cap, "target": target,
            "axes": ("n_sessions", "max_gap_days", "combined_risk"),
            "risk_axis_is_within_fst_only": True,
            "session_budget_basis": SESSION_BUDGET_BASIS}


# ===========================================================================
# TIER 3 (DIAGNOSTIC). gross_synth_per_SED -- why it is not the headline.
# ===========================================================================
def gross_synth_per_SED_diagnostic(
        fst: int = 3,
        intervals: Sequence[int] = tuple(range(1, 91)),
        sed_per_session: float = 1.0,
        horizon_days: int = 365,
        D0: float = 25.0,
        search_bounds: Sequence[int] = (30, 60, 90),
        params: Optional[ModelParams] = None) -> dict:
    """Recompute the two failures that demote `gross_synth_per_SED`.

    The metric was deleted from `vitd_model.py` on 2026-07-31 and survives only
    as a local copy in `scripts/metric_redundancy.py`. It is reproduced here --
    not quoted -- so that the reason it is DIAGNOSTIC-tier is checked against
    the CURRENT parameters on every pipeline run, and cannot silently become
    untrue (or stay true for the wrong reason) as the model is recalibrated.

    Two failures, both machine-checked:

    1. NO INTERIOR OPTIMUM. The argmax is re-evaluated under several search
       bounds. A real peak does not move when the bound moves; an argmax that
       tracks the bound is the signature of a monotone objective.
    2. NET LOSS AT THE OPTIMUM. Final 25(OH)D at the selected interval is
       compared against the STARTING value. A schedule that ends below where it
       began has not helped anyone, however efficient the ratio looks.
    """
    from vitd_objectives import objective_tier, TIER_DIAGNOSTIC
    assert objective_tier("gross_synth_per_SED") == TIER_DIAGNOSTIC

    def gross(result) -> float:
        log = result.get("synthesis_log") or []
        if not log:
            return 0.0
        total_s = sum(item[1] for item in log)
        total_d = sum(item[2] for item in log)
        return total_d / total_s if total_s > 0 else 0.0

    rows = []
    for iv in intervals:
        iv = int(iv)
        sched = [(float(d), float(sed_per_session))
                 for d in range(0, horizon_days, iv)]
        r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                     params=params, D0=D0,
                     dose_framing="fixed_per_session_SED")
        rows.append({"interval_days": iv,
                     "gross_synth_per_SED": gross(r),
                     "final_D": final_D(r),
                     "mean_D": float(np.mean(r["D"])),
                     "net_D_change": final_D(r) - D0,
                     "total_sed": float(r["cumulative_SED"])})

    by_iv = {r["interval_days"]: r for r in rows}
    argmax_by_bound = {}
    for b in search_bounds:
        within = [r for r in rows if r["interval_days"] <= b]
        if within:
            argmax_by_bound[int(b)] = int(
                max(within, key=lambda r: r["gross_synth_per_SED"])["interval_days"])

    best = max(rows, key=lambda r: r["gross_synth_per_SED"])
    tracks_bound = len(set(argmax_by_bound.values())) > 1

    # Trend, not strict monotonicity. The metric is NOT exactly monotone on an
    # integer interval grid: 365 / interval is not an integer, so stepping the
    # interval by one day sometimes drops a whole session and shifts where the
    # last one lands. Those are quantisation wobbles of order 1e-3, not a peak.
    # Spearman rho against the interval is the honest summary, with the
    # violation count and the largest violation reported beside it so the
    # wobble is visible rather than smoothed away.
    order = sorted(by_iv)
    series = [by_iv[k]["gross_synth_per_SED"] for k in order]
    drops = [(a, b, by_iv[a]["gross_synth_per_SED"] - by_iv[b]["gross_synth_per_SED"])
             for a, b in zip(order, order[1:])
             if by_iv[b]["gross_synth_per_SED"] < by_iv[a]["gross_synth_per_SED"] - 1e-9]
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(order, series).statistic)
    except Exception:                                       # pragma: no cover
        rho = float("nan")

    return {
        "fst": fst, "D0": D0, "rows": rows,
        "argmax_by_search_bound": argmax_by_bound,
        "argmax_tracks_search_bound": bool(tracks_bound),
        "selected_interval_days": int(best["interval_days"]),
        "final_D_at_selected": best["final_D"],
        "net_D_change_at_selected": best["net_D_change"],
        "net_loss_at_selected": bool(best["net_D_change"] < 0.0),
        "spearman_rho_interval_vs_metric": rho,
        "n_monotonicity_violations": len(drops),
        "largest_violation": max((d for _, _, d in drops), default=0.0),
        "metric_range": (float(min(series)), float(max(series))),
        "tier": TIER_DIAGNOSTIC,
    }




# ===========================================================================
# REPORTING & PRESENTATION (extracted to vitd_optimization_reports.py)
# Re-exported dynamically for backward compatibility without circular imports.
# ===========================================================================
_REEXPORTED_REPORT_SYMBOLS = {
    "_joint_grid_cell",
    "_run_joint_grid",
    "_plot_variable_dose",
    "analysis_joint_schedule",
    "analysis_variable_dose",
    "ROMAN_FST",
    "analysis_objective_hierarchy",
    "analysis_tri_objective",
    "analysis_constant_sessions",
    "analysis_spacing_fixed_dose",
}


def __getattr__(name: str):
    if name in _REEXPORTED_REPORT_SYMBOLS:
        import vitd_optimization_reports
        return getattr(vitd_optimization_reports, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

