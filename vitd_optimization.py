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
                             sed_from_minutes, _days_below)
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
