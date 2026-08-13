"""
vitd_optimization.py
====================
Schedule generators and analysis for the vitamin D scheduling model.

Schedule generators create lists of (day, SED) tuples with different
spacing patterns for the constant-sessions analysis.

Analysis functions evaluate schedules on efficacy and safety metrics,
producing annotated rankings.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Optional, Sequence
from vitd_model import (
    simulate, build_pulse_schedule, steady_state_D,
    auc_deficit, net_final_D_change_per_SED,
    ModelParams, SOURCE_CALIBRATION_TABLE,
)
from vitd_objectives import (SUFFICIENCY_NMOL_L, GAMMA_MED, RAMP_IN_DAYS,
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
# Why this is worth searching at all: at a FIXED total budget and session count
# the intermittency term sum (d_i/MED)^BURN_EXPONENT is convex in the doses, so
# by Jensen equal doses strictly minimise it -- variable dose can never win on
# the risk axis alone. The gain comes from *reducing the budget*: one constant
# dose has to both climb to the target and hold there, so it necessarily
# overshoots in steady state. Splitting those two jobs is what pays.
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

    NOTE ON INTERPRETATION. Treated as a free *choice* this family loses to
    constant dosing, because the model's ambient UV does not vary: modulating
    dose then only adds intermittency for nothing. Its real use is as a
    *constraint* -- outside the tropics the achievable dose is seasonal whether
    or not the schedule wants it to be, so a schedule that is flat in behaviour
    is already sinusoidal in delivered SED. Reading a loss here as "seasonality
    does not matter" inverts the finding. `phase_day` defaults to a spring
    ascending zero-crossing, putting the peak near midsummer.
    """
    days = np.asarray(days, dtype=float)
    w = 1.0 + float(amp) * np.sin(2.0 * np.pi * (days - float(phase_day)) / 365.0)
    return float(d0) * np.clip(w, 0.0, None)


def dose_gap_coupled(days: np.ndarray, d_ref: float, p: float) -> np.ndarray:
    """Dose scaled by the preceding gap: d_i ~ d_ref * (gap_i / mean_gap) ** p.

    1 parameter beyond the constant-dose baseline, which it recovers at p = 0.
    Generalises schedule_frontloaded / schedule_backloaded, which vary spacing
    at fixed dose: here spacing and dose co-vary, so an unavoidably long gap can
    be compensated rather than simply tolerated.
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
# Constraints (MODEL_AND_EVIDENCE.md section 4):
#   C1  per-session dose <= GAMMA_MED * MED(F)          (never burns)
#   C3  D(t) >= target for all t >= ramp_in_days        (sufficiency floor)
#   C7  per-session dose <= what max_minutes can deliver at this UV index
#   C9  combined cancer-risk index <= risk_cap          (NEW)
#
# Objective: minimise the combined risk index subject to all of the above.
#
# Why minimum dose per interval, rather than sweeping dose freely: both risk
# indices are strictly increasing in per-session dose at a fixed interval, so the
# lowest-risk feasible dose is always the smallest one that still satisfies C3.
# One bisection per interval therefore replaces a dose grid without loss.
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

    The risk indices this ranks on carry different evidence status on their two
    axes -- see vitd_physiology.cancer_risk. Rankings within one skin type are
    unaffected by the weaker (susceptibility) axis; cross-skin-type comparisons
    of the index VALUES are not.
    """
    if params is None:
        params = ModelParams()
    if fst not in range(1, 7):
        raise ValueError("fst must be 1-6")
    if not interval_grid:
        raise ValueError("interval_grid must be non-empty")

    med = _med_sed_by_fst(fst)
    sed_per_min = sed_from_minutes(1.0, uv_index)

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
# per-session vector searched locally. A 3-parameter family recovers
# essentially all of the gain that a free 74-dimensional SLSQP solve finds, and
# it returns a schedule that can actually be described to a person.
def _risk_of(fst: int, days: np.ndarray, seds: np.ndarray, horizon_days: int,
             D0: float, target: float, ramp_in_days: int,
             params: Optional[ModelParams]) -> Tuple[float, float, float, float]:
    """(combined risk, trough after ramp, total SED, time to sufficiency)."""
    sched = build_variable_schedule(days, seds, fst=fst)
    if not sched:
        return float("inf"), float("-inf"), 0.0, float("nan")
    r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                 params=params, D0=D0, dose_framing="fixed_per_session_SED")
    t = np.asarray(r["t"]); D = np.asarray(r["D"])
    mask = t >= ramp_in_days
    trough = float(np.min(D[mask])) if mask.any() else float("-inf")
    return (r["cancer_risk"]["combined_index"], trough,
            float(r["cumulative_SED"]), time_to_sufficiency(r, target))


def _min_feasible(fst, days, make_seds, horizon_days, D0, target,
                  ramp_in_days, params, lo=0.0, hi=1.0, iters=28):
    """Smallest scale factor on a dose shape that still clears the floor.

    Bisection is valid because trough-after-ramp is monotone increasing in a
    uniform scaling of the whole profile: per-session yield is increasing in
    dose under both feedback modes and with photodegradation on.
    """
    seds_hi = make_seds(hi)
    _, trough_hi, _, _ = _risk_of(fst, days, seds_hi, horizon_days, D0, target,
                                  ramp_in_days, params)
    if trough_hi < target:
        return None                      # infeasible even at full scale
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, trough, _, _ = _risk_of(fst, days, make_seds(mid), horizon_days, D0,
                                   target, ramp_in_days, params)
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
        risk, trough, total, ttf = _risk_of(fst, days, seds, horizon_days, D0,
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
# system, so it is used here as motivation for the formulation, not as a
# guarantee -- scripts/joint_schedule_freesolve.py checks the low-dimensional
# answer against a direct sparse solve rather than assuming it.
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
                            popsize: int = 10) -> Optional[dict]:
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

    Returns None if no schedule in the family clears the floor.

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
        risk, trough, _, _ = _risk_of(fst, days, seds, horizon_days, D0,
                                      target, ramp_in_days, params)
        if not np.isfinite(risk):
            return 1e6
        return risk + PEN * max(0.0, target - trough) ** 2

    best = None
    for family in families:
        if family == "exponential":
            bounds = [(0.02, 1.0), (0.0, 8.0), (5.0, 180.0)]
        elif family == "two_phase":
            bounds = [(0.02, 1.0), (0.1, 1.0), (0.0, 0.6)]
        else:
            raise ValueError(f"unknown joint dose family {family!r}")
        # The spacing dimension is present in BOTH arms and merely inert in the
        # control (build() pins gamma to 1.0). differential_evolution sizes its
        # population as popsize * n_parameters, so giving the free arm an extra
        # dimension would also hand it a ~33% larger search budget -- the gain
        # would then be partly attributable to search effort rather than to
        # spacing, which is the one thing this comparison exists to isolate.
        bounds = bounds + [(0.4, 2.5)]

        res = differential_evolution(lambda x, f=family: score(f, x),
                                     bounds=bounds, seed=seed, maxiter=maxiter,
                                     popsize=popsize, tol=1e-6, polish=True)

        # Land exactly on the floor: bisect the overall dose LEVEL of the winning
        # shape. Monotone, because per-session yield is increasing in dose.
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
        risk, trough, total, ttf = _risk_of(fst, days, seds, horizon_days, D0,
                                            target, ramp_in_days, params)
        if trough < target - 1e-6:
            continue
        delivered = np.array([v for _, v in
                              build_variable_schedule(days, seds, fst=fst)])
        cand = {
            "fst": fst, "n_sessions": n_sessions, "family": family,
            "free_spacing": free_spacing, "gamma": float(gamma),
            "combined_risk": risk, "trough_after_ramp": trough,
            "total_sed": total, "time_to_sufficiency": ttf,
            "n_sessions_delivered": int(delivered.size),
            "max_sed": float(np.max(delivered)) if delivered.size else 0.0,
            "min_sed": float(np.min(delivered)) if delivered.size else 0.0,
            "first_gap_days": float(days[1] - days[0]) if days.size > 1 else float("nan"),
            "last_gap_days": float(days[-1] - days[-2]) if days.size > 1 else float("nan"),
            # Winning parameters, stored as scalars so the exact schedule can be
            # rebuilt by rebuild_joint_schedule() without re-running the search.
            "p_scale": float(res.x[0]), "p_a": float(res.x[1]),
            "p_b": float(res.x[2]), "p_level": float(s),
        }
        if best is None or cand["combined_risk"] < best["combined_risk"]:
            best = cand
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


def analysis_joint_schedule(params: Optional[ModelParams] = None,
                            fsts: Sequence[int] = (2, 3, 5),
                            session_counts: Sequence[int] = (104, 146, 183, 260)):
    """Does free spacing buy anything once the dose is already free?

    Both arms use the SAME solver, budget and dose families; the only difference
    is whether the spacing exponent is searched or pinned to uniform. Any
    difference is therefore attributable to spacing rather than to search
    effort.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS N: joint timing + dose optimisation")
    print("=" * 70)
    print("\n  min combined cancer risk  s.t.  min D(t >= ramp) >= 50 nmol/L,")
    print("  per-session dose <= 0.7 MED, session budget fixed per row.")
    print("  Control arm pins spacing to uniform; treatment arm frees the")
    print("  spacing exponent. Same solver and budget in both.")
    print("\n  A session budget is part of the problem: without one the optimum")
    print("  is degenerate (risk falls monotonically to a daily schedule),")
    print("  because at fixed cumulative dose the intermittency term is")
    print("  minimised by spreading over as many days as possible.")

    rows = []
    for fst in fsts:
        for N in session_counts:
            fixed = optimise_joint_schedule(fst, N, free_spacing=False,
                                            params=params)
            joint = optimise_joint_schedule(fst, N, free_spacing=True,
                                            params=params)
            for r in (fixed, joint):
                if r is None:
                    continue
                if fixed is not None and fixed["combined_risk"] > 0:
                    r["risk_vs_uniform_spacing_pct"] = (
                        100.0 * (fixed["combined_risk"] - r["combined_risk"])
                        / fixed["combined_risk"])
                else:
                    r["risk_vs_uniform_spacing_pct"] = float("nan")
                rows.append(r)

    if not rows:
        print("\n  No feasible schedule at any tested (FST, session budget).")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv("tables/joint_schedule_optimization.csv", index=False)

    print(f"\n  {'FST':>3} {'N':>4} {'spacing':>9} {'family':>12} {'gamma':>6} "
          f"{'risk':>9} {'vs unif%':>9} {'totSED':>8} {'t->suff':>8}")
    print("  " + "-" * 82)
    for fst in fsts:
        for N in session_counts:
            sub = df[(df["fst"] == fst) & (df["n_sessions"] == N)]
            for _, r in sub.iterrows():
                ttf = ("n/a" if not np.isfinite(r["time_to_sufficiency"])
                       else f"{r['time_to_sufficiency']:.0f}d")
                print(f"  {int(r['fst']):>3} {int(r['n_sessions']):>4} "
                      f"{('free' if r['free_spacing'] else 'uniform'):>9} "
                      f"{r['family']:>12} {r['gamma']:>6.2f} "
                      f"{r['combined_risk']:>9.4f} "
                      f"{r['risk_vs_uniform_spacing_pct']:>9.1f} "
                      f"{r['total_sed']:>8.1f} {ttf:>8}")
            if len(sub):
                print()

    free = df[df["free_spacing"]]
    if len(free):
        gains = free["risk_vs_uniform_spacing_pct"].to_numpy(dtype=float)
        gains = gains[np.isfinite(gains)]
        print(f"  Free spacing vs uniform spacing, {len(gains)} operating points:")
        print(f"    median gain {np.median(gains):+.2f}%   "
              f"best {np.max(gains):+.2f}%   worst {np.min(gains):+.2f}%")
        print(f"    gamma chosen: median {np.median(free['gamma']):.2f} "
              f"(1.00 = uniform), range "
              f"{np.min(free['gamma']):.2f}-{np.max(free['gamma']):.2f}")

        # A negative gain is impossible in principle: gamma = 1 lies inside the
        # bounds, so the free arm can always express the control exactly. Any
        # negative entry is therefore pure differential-evolution noise, and its
        # magnitude is a free calibration of the solver's run-to-run spread --
        # which is worth more than hiding it would be.
        worst = float(np.min(gains))
        if worst < 0:
            print(f"\n    A negative entry is not a real loss: gamma = 1 is")
            print(f"    inside the search bounds, so the free arm can reproduce")
            print(f"    the control exactly and cannot truly do worse. The worst")
            print(f"    entry ({worst:+.1f}%) therefore bounds solver noise, and")
            print(f"    gains below roughly {abs(worst):.0f}% should not be read as real.")

    print("\n  Mechanism. Free spacing does NOT win by lowering cumulative dose")
    print("  -- total SED is usually slightly HIGHER. It wins by lowering the")
    print("  PEAK per-session dose: bunching sessions more densely early lets")
    print("  the loading phase use smaller individual doses. The intermittency")
    print("  term is cubic in per-session dose while the SCC term is a 1.7 power")
    print("  of the cumulative, so trading a small cumulative increase for a")
    print("  large peak reduction is strongly favourable.")
    print("\n  This REVERSES the fixed-dose result: at fixed dose, spacing only")
    print("  redistributes an unchangeable per-session intensity, and uniform")
    print("  spacing wins by minimising the largest gap. Once dose is free,")
    print("  spacing acquires a second job -- setting how much dose each session")
    print("  needs to carry -- and front-loading takes over.")
    if len(free):
        n_won = int(np.sum(gains > 0))
        n_front = int(np.sum(free["gamma"].to_numpy(dtype=float) > 1.0))
        print(f"\n  Precisely: the search selected gamma > 1 at {n_front} of "
              f"{len(free)} points,")
        print(f"  and beat the uniform control at {n_won} of {len(gains)}.")
        if n_won < len(gains):
            print(f"  The {len(gains) - n_won} remaining point(s) sit inside the "
                  f"solver-noise bound")
            print(f"  above, so they are not evidence against front-loading -- but")
            print(f"  they are not evidence for it either. The claim is "
                  f"{n_won}/{len(gains)}, not unanimous.")
    print(f"\n  Saved tables/joint_schedule_optimization.csv")
    return df


def analysis_variable_dose(params: Optional[ModelParams] = None,
                           fsts: Sequence[int] = (2, 3, 5),
                           session_counts: Sequence[int] = (104, 122, 146, 183)):
    """Variable dose vs the constant-dose baseline, at matched efficacy.

    Every profile compared here holds the SAME sufficiency floor with the SAME
    number of sessions on the SAME days, so the comparison cannot be won by
    dosing less or going out more -- only by distributing dose better.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS M: variable-dose optimisation")
    print("=" * 70)
    print("\n  min combined cancer risk  s.t.  min D(t >= ramp) >= 50 nmol/L,")
    print("  per-session dose <= 0.7 MED, sessions uniformly spaced.")
    print("  Dose framing: absolute SED per session (not fractional MED).")

    rows = []
    for fst in fsts:
        for N in session_counts:
            base = optimise_variable_dose(fst, N, "uniform", params=params)
            for fam in VARIABLE_DOSE_FAMILIES:
                res = (base if fam == "uniform"
                       else optimise_variable_dose(fst, N, fam, params=params))
                if res is None:
                    continue
                if base is not None and base["combined_risk"] > 0:
                    res["risk_vs_uniform_pct"] = (
                        100.0 * (base["combined_risk"] - res["combined_risk"])
                        / base["combined_risk"])
                    res["sed_vs_uniform_pct"] = (
                        100.0 * (base["total_sed"] - res["total_sed"])
                        / base["total_sed"]) if base["total_sed"] > 0 else 0.0
                else:
                    res["risk_vs_uniform_pct"] = float("nan")
                    res["sed_vs_uniform_pct"] = float("nan")
                rows.append(res)

    if not rows:
        print("\n  No feasible profile at any tested (FST, session count).")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv("tables/variable_dose_optimization.csv", index=False)

    print(f"\n  {'FST':>3} {'N':>4} {'family':>12} {'risk':>9} {'vs unif%':>9} "
          f"{'totSED':>8} {'SED sav%':>9} {'trough':>7} {'t->suff':>8}")
    print("  " + "-" * 82)
    for fst in fsts:
        for N in session_counts:
            sub = df[(df["fst"] == fst) & (df["n_sessions"] == N)]
            for _, r in sub.iterrows():
                ttf = ("n/a" if not np.isfinite(r["time_to_sufficiency"])
                       else f"{r['time_to_sufficiency']:.0f}d")
                print(f"  {int(r['fst']):>3} {int(r['n_sessions']):>4} "
                      f"{r['family']:>12} {r['combined_risk']:>9.4f} "
                      f"{r['risk_vs_uniform_pct']:>9.1f} {r['total_sed']:>8.1f} "
                      f"{r['sed_vs_uniform_pct']:>9.1f} "
                      f"{r['trough_after_ramp']:>7.2f} {ttf:>8}")
            if len(sub):
                print()

    best = df.loc[df.groupby(["fst", "n_sessions"])["combined_risk"].idxmin()]
    print("  Lowest-risk family at each operating point:")
    for _, r in best.iterrows():
        print(f"    FST {int(r['fst'])}, {int(r['n_sessions'])} sessions/yr: "
              f"{r['family']} ({r['risk_vs_uniform_pct']:.1f}% less risk, "
              f"{r['sed_vs_uniform_pct']:.1f}% less total UV)")

    print("\n  Interpretation. At fixed budget AND session count the")
    print("  intermittency term is convex in the doses, so equal doses minimise")
    print("  it (Jensen) -- variable dose cannot win on the risk axis alone.")
    print("  Every gain below therefore comes from needing LESS total UV:")
    print("  a constant dose must both climb to target and hold there, so it")
    print("  overshoots in steady state. The gain shrinks as sessions rise,")
    print("  because a frequent schedule already runs a small per-session dose.")
    print("\n  Absolute D inherits the uncalibrated SC priors (SC_max, alpha_SC,")
    print("  K_SC); rankings within a skin type are more robust than levels.")
    print(f"\n  Saved tables/variable_dose_optimization.csv")

    _plot_variable_dose(df, params=params)
    return df


def _plot_variable_dose(df: pd.DataFrame, params: Optional[ModelParams] = None,
                        fst: int = 3, n_sessions: int = 122,
                        horizon_days: int = 365, D0: float = 25.0):
    """Dose profiles and the 25(OH)D trajectories they produce."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    sub = df[(df["fst"] == fst) & (df["n_sessions"] == n_sessions)]
    if sub.empty:
        return
    days = session_days(n_sessions, horizon_days)
    cap = GAMMA_MED * _med_sed_by_fst(fst)

    profiles = {}
    for _, r in sub.iterrows():
        fam = r["family"]
        if fam == "uniform":
            seds = dose_uniform(days, r["dose_sed"])
        elif fam == "two_phase":
            seds = dose_two_phase(days, r["d_load_sed"], r["d_maint_sed"],
                                  r["t_switch"])
        elif fam == "exponential":
            seds = dose_exponential(days, r["d_inf_sed"], r["ratio"], r["tau"])
        else:
            continue
        profiles[fam] = seds

    if len(profiles) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fam, seds in profiles.items():
        sched = build_variable_schedule(days, seds, fst=fst)
        axes[0].step(days, np.clip(seds, 0, cap), where="post", label=fam, lw=1.8)
        r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                     params=params, D0=D0, dose_framing="fixed_per_session_SED")
        axes[1].plot(r["t"], r["D"], label=fam, lw=1.8)

    axes[0].axhline(cap, color="crimson", ls="--", alpha=0.6,
                    label=f"0.7 MED cap ({cap:.2f} SED)")
    axes[0].set_xlabel("day"); axes[0].set_ylabel("SED per session")
    axes[0].set_title(f"Dose profile (FST {fst}, {n_sessions} sessions/yr)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].axhline(SUFFICIENCY_NMOL_L, color="gray", ls="--", alpha=0.6,
                    label="sufficiency 50")
    axes[1].axvline(RAMP_IN_DAYS, color="gray", ls=":", alpha=0.6,
                    label=f"ramp-in ends ({RAMP_IN_DAYS}d)")
    axes[1].set_xlabel("day"); axes[1].set_ylabel("25(OH)D (nmol/L)")
    axes[1].set_title("All profiles hold the same floor, at different cost")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figs/variable_dose_profiles.png", dpi=130)
    plt.close(fig)
    print("  Saved figs/variable_dose_profiles.png")


def analysis_constant_sessions(params: Optional[ModelParams] = None):
    """Sweep session counts 1-52 over 365 days with fixed total SED, uniform spacing.

    For each session count, evaluates max-D, AUC deficiency, and net D change
    per SED. Flags schedules with rolling 7d SED > 5 as requiring UV safety
    review.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS L: constant-sessions optimization sweep")
    print("=" * 70)

    total_SED = 52.0
    duration_days = 365
    fst = 5
    sed_per_session_default = 1.0
    if params is None:
        params = ModelParams()
        from vitd_model import get_E_source
        params.E_source = get_E_source("solar_summer")

    print(f"\n  Using E_source = {params.E_source:.2f} for constant-sessions sweep.")

    rows = []
    for N_sessions in range(1, 53):
        sed = total_SED / N_sessions
        sched = schedule_uniform(N_sessions, duration_days, sed)
        m = evaluate_schedule(sched, fst, duration_days, params=params)
        m["N_sessions"] = N_sessions
        m["sed_per_session"] = sed
        m["interval_days"] = duration_days / N_sessions

        # UV safety flag
        rolling_7d = m["max_rolling_7d_SED"]
        if rolling_7d > 5.0:
            m["uv_safety_flag"] = "UV safety: review"
        else:
            m["uv_safety_flag"] = ""

        rows.append(m)

    df = pd.DataFrame(rows)

    # Save
    df.to_csv("tables/constant_sessions_analysis.csv", index=False)
    print(f"\n  Saved tables/constant_sessions_analysis.csv")

    # Find optimal (lowest AUC deficit at 50)
    best = df.loc[df["auc_deficit_50"].idxmin()]
    print(f"\n  Optimal schedule: {int(best['N_sessions'])} sessions")
    print(f"    SED/session:     {best['sed_per_session']:.3f}")
    print(f"    Interval:        {best['interval_days']:.1f} days")
    print(f"    Max D:           {best['max_D']:.1f} nmol/L")
    print(f"    AUC deficit @50: {best['auc_deficit_50']:.1f} nmol/L·days")
    print(f"    Net ΔD/SED:      {best['net_final_D_change_per_SED']:.3f} nmol/L/SED")

    # Print table
    print(f"\n  {'N':>3s} {'SED/sesh':>9s} {'Interval':>9s} {'Max D':>7s} "
          f"{'Deficit@50':>11s} {'NetΔD/SED':>10s} "
          f"{'Roll7d':>7s} {'Flag':>20s}")
    print(f"  {'-'*3} {'-'*9} {'-'*9} {'-'*7} {'-'*11} {'-'*10} "
          f"{'-'*7} {'-'*20}")
    for _, row in df.iterrows():
        print(f"  {int(row['N_sessions']):3d} {row['sed_per_session']:>8.3f} "
              f"{row['interval_days']:>8.1f}  {row['max_D']:>6.1f}  "
              f"{row['auc_deficit_50']:>10.1f}  "
              f"{row['net_final_D_change_per_SED']:>9.3f}  "
              f"{row['max_rolling_7d_SED']:>6.1f}  "
              f"{row['uv_safety_flag']:>20s}")

    return df


if __name__ == "__main__":
    analysis_constant_sessions()
