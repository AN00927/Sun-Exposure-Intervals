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
from typing import List, Tuple, Iterator, Optional
from vitd_model import (
    simulate, build_pulse_schedule, steady_state_D,
    auc_deficit, gross_synth_per_SED, net_final_D_change_per_SED,
    ModelParams,
)


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
        "gross_synth_per_SED": gross_synth_per_SED(result),
        "net_final_D_change_per_SED": net_final_D_change_per_SED(result, D0),
        "max_pct_MED": max_frac * 100,
        "max_rolling_7d_SED": max_rolling_7d,
        "cpd_risk": cpd_risk,
        "chronic_risk": chronic_risk,
    }


def analysis_constant_sessions():
    """Sweep session counts 1-52 over 365 days with fixed total SED, uniform spacing.

    For each session count, evaluates max-D, AUC deficiency, gross synthesis
    efficiency, and net D change per SED. Flags schedules with rolling 7d SED > 5
    as requiring UV safety review.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS L: constant-sessions optimization sweep")
    print("=" * 70)

    total_SED = 52.0
    duration_days = 365
    fst = 5
    sed_per_session_default = 1.0
    params = ModelParams()

    # Placeholder annotation per AGENTS.md
    E_source_note = "E_source=2.5 (NB-UVB 311 nm placeholder)"
    print(f"\n  {E_source_note}")
    print(f"  WARNING: E_source=2.5 is a placeholder. Per-study calibration")
    print(f"  from SOURCE_CALIBRATION_TABLE should be used for absolute predictions.")
    params.E_source = 2.5

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
    print(f"    Gross eff:       {best['gross_synth_per_SED']:.3f} nmol/L/SED")
    print(f"    Net ΔD/SED:      {best['net_final_D_change_per_SED']:.3f} nmol/L/SED")

    # Print table
    print(f"\n  {'N':>3s} {'SED/sesh':>9s} {'Interval':>9s} {'Max D':>7s} "
          f"{'Deficit@50':>11s} {'GrossEff':>9s} {'NetΔD/SED':>10s} "
          f"{'Roll7d':>7s} {'Flag':>20s}")
    print(f"  {'-'*3} {'-'*9} {'-'*9} {'-'*7} {'-'*11} {'-'*9} {'-'*10} "
          f"{'-'*7} {'-'*20}")
    for _, row in df.iterrows():
        print(f"  {int(row['N_sessions']):3d} {row['sed_per_session']:>8.3f} "
              f"{row['interval_days']:>8.1f}  {row['max_D']:>6.1f}  "
              f"{row['auc_deficit_50']:>10.1f}  "
              f"{row['gross_synth_per_SED']:>8.3f}  "
              f"{row['net_final_D_change_per_SED']:>9.3f}  "
              f"{row['max_rolling_7d_SED']:>6.1f}  "
              f"{row['uv_safety_flag']:>20s}")

    return df


if __name__ == "__main__":
    analysis_constant_sessions()
