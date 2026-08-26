"""
nhanes_dietary_partitioning.py
==============================
Empirical Dietary Baseline Floor & Solar vs Dietary Contribution Partitioning.
Analyzes serum 25(OH)D concentrations across dietary strata (UV-only, food-only,
supplemented) in NHANES to quantify the non-solar baseline floor and seasonal
solar gain.

Lifecycle Bucket: RECURRING (diagnostic tool)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "audit_outputs", "nhanes_standardized_processed.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "tables", "nhanes_dietary_partitioning.csv")


def weighted_median_iqr(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """Compute survey-weighted median, 25th percentile, and 75th percentile."""
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    
    cum_weights = np.cumsum(weights) - 0.5 * weights
    cum_weights /= np.sum(weights)
    
    p25, p50, p75 = np.interp([0.25, 0.50, 0.75], cum_weights, values)
    return float(p50), float(p25), float(p75)


def analyze_dietary_partitioning():
    print("=" * 80)
    print("NHANES DIETARY BASELINE FLOOR & SOURCE PARTITIONING ANALYSIS")
    print("=" * 80)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing processed NHANES data: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    
    valid = df[
        (df["age"] >= 18) &
        (df["wtmec2yr"] > 0) &
        (df["vitd_standardized_nmol_L"].notna()) &
        (df["vitd_standardized_nmol_L"] > 0)
    ].copy()
    
    race_names = {
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        1: "Mexican American",
        6: "Non-Hispanic Asian",
    }
    
    results = []
    
    for r_code, r_name in race_names.items():
        sub = valid[valid["race_ethnicity"] == r_code]
        
        # Subsets
        uv_only_w = sub[(sub["dietary_source_flag"] == "UV-only") & (sub["exam_season"] == 1)]
        uv_only_s = sub[(sub["dietary_source_flag"] == "UV-only") & (sub["exam_season"] == 2)]
        
        food_w = sub[(sub["dietary_source_flag"] == "food-only") & (sub["exam_season"] == 1)]
        food_s = sub[(sub["dietary_source_flag"] == "food-only") & (sub["exam_season"] == 2)]
        
        unsupp_w = sub[(sub["dietary_source_flag"].isin(["UV-only", "food-only"])) & (sub["exam_season"] == 1)]
        unsupp_s = sub[(sub["dietary_source_flag"].isin(["UV-only", "food-only"])) & (sub["exam_season"] == 2)]
        
        supp_w = sub[(sub["dietary_source_flag"] == "supplemented") & (sub["exam_season"] == 1)]
        supp_s = sub[(sub["dietary_source_flag"] == "supplemented") & (sub["exam_season"] == 2)]
        
        def get_stat(df_group):
            if len(df_group) < 10:
                return np.nan, np.nan, np.nan, len(df_group)
            med, p25, p75 = weighted_median_iqr(
                df_group["vitd_standardized_nmol_L"].to_numpy(dtype=float),
                df_group["wtmec2yr"].to_numpy(dtype=float)
            )
            return med, p25, p75, len(df_group)
            
        uv_w_med, uv_w_p25, uv_w_p75, uv_w_n = get_stat(uv_only_w)
        uv_s_med, uv_s_p25, uv_s_p75, uv_s_n = get_stat(uv_only_s)
        
        food_w_med, _, _, _ = get_stat(food_w)
        food_s_med, _, _, _ = get_stat(food_s)
        
        unsupp_w_med, unsupp_w_p25, unsupp_w_p75, _ = get_stat(unsupp_w)
        unsupp_s_med, unsupp_s_p25, unsupp_s_p75, _ = get_stat(unsupp_s)
        
        supp_w_med, supp_w_p25, supp_w_p75, _ = get_stat(supp_w)
        supp_s_med, supp_s_p25, supp_s_p75, _ = get_stat(supp_s)
        
        # Computations
        # Baseline Floor: Winter median of UV-only (or unsupplemented)
        floor_d = uv_w_med if pd.notna(uv_w_med) else unsupp_w_med
        # Solar Swing: Summer median - Winter median (unsupplemented)
        solar_delta = unsupp_s_med - unsupp_w_med
        # Dietary Food Lift: food-only winter - UV-only winter
        food_lift = (food_w_med - uv_w_med) if (pd.notna(food_w_med) and pd.notna(uv_w_med)) else np.nan
        # Supplement Lift: supplemented winter - unsupplemented winter
        supp_lift = supp_w_med - unsupp_w_med
        
        results.append({
            "Race/Ethnicity": r_name,
            "Baseline Floor D_floor (nmol/L)": round(floor_d, 1),
            "Unsupplemented Winter Median": round(unsupp_w_med, 1),
            "Unsupplemented Summer Median": round(unsupp_s_med, 1),
            "Solar Seasonal Delta (nmol/L)": round(solar_delta, 1),
            "Dietary Food Lift (nmol/L)": round(food_lift, 1) if pd.notna(food_lift) else "N/A",
            "Supplement Lift (nmol/L)": round(supp_lift, 1),
            "Supplemented Winter Median": round(supp_w_med, 1),
            "N (Unsupp Winter)": len(unsupp_w),
            "N (Supp Winter)": len(supp_w),
        })
        
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dietary partitioning table to: {OUTPUT_CSV}\n")
    print(res_df.to_string(index=False))
    return res_df


if __name__ == "__main__":
    analyze_dietary_partitioning()
