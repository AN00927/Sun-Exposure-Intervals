"""
nhanes_macro_validation.py
==========================
Population Macro-Validation of Annual Equilibrium (52-Week Steady-State).
Evaluates whether simulated annual 25(OH)D dynamics across FST I-VI under
realistic free-living sun exposure schedules align with empirical,
survey-weighted NHANES unsupplemented population distributions.

Lifecycle Bucket: RECURRING (sensitivity / diagnostic tool)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vitd_model import ModelParams, simulate, build_pulse_schedule, get_E_source
import vitd_scenario

PRIORS_CSV = os.path.join(REPO_ROOT, "tables", "nhanes_baseline_priors.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "tables", "nhanes_macro_validation.csv")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
OUTPUT_PNG = os.path.join(FIGURES_DIR, "nhanes_macro_validation.png")


def build_annual_lifestyle_schedule(
    horizon_days: int = 365,
    summer_sed_per_session: float = 1.0,
    winter_sed_per_session: float = 0.25,
) -> tuple[tuple[float, float], ...]:
    """
    Build a representative 52-week solar lifestyle exposure schedule:
    - Summer/Autumn (Days 120-300, ~May-Oct): 2 sessions/week (e.g. Wednesday & Saturday) at 1.0 SED
    - Winter/Spring (Days 0-119 & 301-365, ~Nov-Apr): 1 session/week at 0.25 SED
    """
    events = []
    for day in range(horizon_days):
        day_of_week = day % 7
        is_summer = (120 <= day <= 300)
        
        if is_summer:
            if day_of_week in (2, 5):  # 2x per week in summer
                events.append((float(day), float(summer_sed_per_session)))
        else:
            if day_of_week == 5:  # 1x per week in winter (lower ambient UV)
                events.append((float(day), float(winter_sed_per_session)))
                
    return tuple(events)


def run_52week_simulation(fst: int, baseline_D0: float) -> dict:
    """Simulate 52-week ODE trajectory for given FST and baseline D0."""
    params = ModelParams()
    
    schedule = build_annual_lifestyle_schedule(365, summer_sed_per_session=1.0, winter_sed_per_session=0.25)
    t_eval = np.linspace(0, 365, 366)
    
    # 2 years simulation to ensure asymptotic annual steady state, analyzing the second year
    schedule_2yr = tuple(
        list(schedule) + [(float(t + 365), sed) for t, sed in schedule]
    )
    
    res = simulate(
        fst=fst,
        schedule=list(schedule_2yr),
        duration_days=730,
        params=params,
        D0=baseline_D0,
        exposed_bsa_override=vitd_scenario.EXPOSED_BSA,
        E_source_override=get_E_source("solar_summer"),
    )
    
    # Extract second year (days >= 365.0)
    t_arr = res["t"]
    d_arr = res["D"]
    
    mask_y2 = (t_arr >= 365.0) & (t_arr <= 730.0)
    t_year2 = t_arr[mask_y2] - 365.0
    d_year2 = d_arr[mask_y2]
    
    # Summer days (120 to 300) vs Winter days (0 to 119, 301 to 365)
    summer_mask = (t_year2 >= 120) & (t_year2 <= 300)
    winter_mask = ~summer_mask
    
    return {
        "annual_mean": float(np.mean(d_year2)),
        "annual_min": float(np.min(d_year2)),
        "annual_max": float(np.max(d_year2)),
        "summer_peak": float(np.max(d_year2[summer_mask])),
        "winter_nadir": float(np.min(d_year2[winter_mask])),
        "d_trajectory": d_year2,
        "t_trajectory": t_year2,
    }


def load_nhanes_priors() -> pd.DataFrame:
    """Load precomputed NHANES baseline priors table."""
    if not os.path.exists(PRIORS_CSV):
        raise FileNotFoundError(f"Missing priors table: {PRIORS_CSV}. Run scripts/generate_nhanes_priors_table.py first.")
    return pd.read_csv(PRIORS_CSV)


def evaluate_macro_validation():
    print("=" * 80)
    print("NHANES POPULATION MACRO-VALIDATION (52-WEEK EQUILIBRIUM)")
    print("=" * 80)
    
    priors_df = load_nhanes_priors()
    
    # Filter priors to unsupplemented cohorts
    unsupp_priors = priors_df[
        (priors_df["strata_type"] == "fst_proxy") &
        (priors_df["dietary_status"] == "all_unsupplemented")
    ]
    
    results = []
    
    for fst in range(1, 7):
        # Look up empirical unsupplemented priors for this FST
        fst_label = f"FST_{fst}"
        w_prior = unsupp_priors[(unsupp_priors["strata_value"] == fst_label) & (unsupp_priors["season"] == "winter_spring")]
        s_prior = unsupp_priors[(unsupp_priors["strata_value"] == fst_label) & (unsupp_priors["season"] == "summer_autumn")]
        
        if w_prior.empty or s_prior.empty:
            continue
            
        w_row = w_prior.iloc[0]
        s_row = s_prior.iloc[0]
        
        # Initial starting D0 is empirical winter unsupplemented median
        d0_init = float(w_row["p50"])
        
        sim_res = run_52week_simulation(fst, baseline_D0=d0_init)
        
        model_peak = sim_res["summer_peak"]
        model_nadir = sim_res["winter_nadir"]
        model_mean = sim_res["annual_mean"]
        
        # Benchmark against NHANES empirical distribution
        # Summer check
        s_p05, s_p25, s_p50, s_p75, s_p95 = s_row["p05"], s_row["p25"], s_row["p50"], s_row["p75"], s_row["p95"]
        # Winter check
        w_p05, w_p25, w_p50, w_p75, w_p95 = w_row["p05"], w_row["p25"], w_row["p50"], w_row["p75"], w_row["p95"]
        
        # Status determination:
        # PASS: within 5th-95th percentile in both seasons
        # WARN: outside IQR (25th-75th) in at least one season
        # FAIL: outside 5th-95th in at least one season
        summer_in_90 = (s_p05 <= model_peak <= s_p95)
        winter_in_90 = (w_p05 <= model_nadir <= w_p95)
        summer_in_iqr = (s_p25 <= model_peak <= s_p75)
        winter_in_iqr = (w_p25 <= model_nadir <= w_p75)
        
        if summer_in_90 and winter_in_90:
            status = "PASS (IQR)" if (summer_in_iqr and winter_in_iqr) else "PASS (90% CI)"
        else:
            status = "FAIL (Out of 90% band)"
            
        results.append({
            "FST": fst,
            "Model Summer Peak": round(model_peak, 1),
            "NHANES Summer Median [IQR]": f"{s_p50:.1f} [{s_p25:.1f}-{s_p75:.1f}]",
            "Model Winter Nadir": round(model_nadir, 1),
            "NHANES Winter Median [IQR]": f"{w_p50:.1f} [{w_p25:.1f}-{w_p75:.1f}]",
            "Model Annual Mean": round(model_mean, 1),
            "Macro Validation Status": status,
        })
        
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved macro validation table to: {OUTPUT_CSV}\n")
    print(res_df.to_string(index=False))
    print("\nMacro validation check complete.")
    return res_df


if __name__ == "__main__":
    evaluate_macro_validation()
