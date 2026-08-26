"""
generate_nhanes_priors_table.py
===============================
Extracts survey-weighted empirical baseline 25(OH)D priors from standardized
NHANES data (2001-2023) across skin phototypes (FST proxies), race/ethnicity,
exam seasons, and dietary supplementation statuses.

Generates:
    tables/nhanes_baseline_priors.csv
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "audit_outputs", "nhanes_standardized_processed.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "tables", "nhanes_baseline_priors.csv")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Calculate survey-weighted quantiles for 1D arrays."""
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    
    cum_weights = np.cumsum(weights) - 0.5 * weights
    cum_weights /= np.sum(weights)
    return np.interp(quantiles, cum_weights, values)


def compute_group_stats(df_sub: pd.DataFrame, quantiles=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)) -> dict:
    """Compute survey-weighted mean, std, quantiles, and unweighted N."""
    vals = df_sub["vitd_standardized_nmol_L"].to_numpy(dtype=float)
    w = df_sub["wtmec2yr"].to_numpy(dtype=float)
    
    total_w = np.sum(w)
    if total_w <= 0 or len(vals) == 0:
        return None
        
    w_mean = np.sum(w * vals) / total_w
    w_var = np.sum(w * (vals - w_mean) ** 2) / total_w
    w_std = np.sqrt(w_var)
    
    q_vals = weighted_quantile(vals, w, list(quantiles))
    
    return {
        "weighted_mean": round(float(w_mean), 2),
        "weighted_std": round(float(w_std), 2),
        "p05": round(float(q_vals[0]), 2),
        "p10": round(float(q_vals[1]), 2),
        "p25": round(float(q_vals[2]), 2),
        "p50": round(float(q_vals[3]), 2),
        "p75": round(float(q_vals[4]), 2),
        "p90": round(float(q_vals[5]), 2),
        "p95": round(float(q_vals[6]), 2),
        "n_unweighted": int(len(vals)),
    }


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing processed NHANES data: {DATA_PATH}")

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Filter to adults with valid weights and positive 25(OH)D
    valid = df[
        (df["age"] >= 18) &
        (df["wtmec2yr"] > 0) &
        (df["vitd_standardized_nmol_L"].notna()) &
        (df["vitd_standardized_nmol_L"] > 0)
    ].copy()
    print(f"Filtered to {len(valid)} valid adult records.")

    # Define FST Proxy:
    # 1. Direct DED031 if present (1->FST I, 2->FST II, 3->FST III, 4->FST IV, 5->FST V)
    # 2. Race-based fallback if DED031 is missing or for FST VI:
    #    - NH White (3) -> FST II/III proxy
    #    - NH Black (4) -> FST V/VI proxy
    #    - Mexican American (1) / Other Hispanic (2) -> FST III/IV
    #    - NH Asian (6) -> FST III/IV
    
    def map_fst(row):
        skin_rx = row.get("deq_skin_reaction")
        race = row.get("race_ethnicity")
        
        if pd.notna(skin_rx) and skin_rx in (1, 2, 3, 4, 5):
            if skin_rx == 5 and race == 4:
                return 6  # Deeply pigmented Black individuals with no burning
            return int(skin_rx)
        
        # Demographic fallback
        if race == 4:
            return 5  # NH Black general proxy
        elif race == 3:
            return 2  # NH White general proxy
        elif race in (1, 2, 6):
            return 4  # Hispanic / Asian general proxy
        return np.nan

    valid["fst_proxy"] = valid.apply(map_fst, axis=1)

    # Dietary classifications
    dietary_categories = {
        "all": lambda d: d,
        "all_unsupplemented": lambda d: d[d["dietary_source_flag"].isin(["UV-only", "food-only"])],
        "UV-only": lambda d: d[d["dietary_source_flag"] == "UV-only"],
        "food-only": lambda d: d[d["dietary_source_flag"] == "food-only"],
        "supplemented": lambda d: d[d["dietary_source_flag"] == "supplemented"],
    }

    # Season classifications
    season_filters = {
        "all_year": lambda d: d,
        "winter_spring": lambda d: d[d["exam_season"] == 1],
        "summer_autumn": lambda d: d[d["exam_season"] == 2],
    }

    records = []

    # 1. Stratify by FST proxy (1..6)
    for fst in range(1, 7):
        df_fst = valid[valid["fst_proxy"] == fst]
        for s_name, s_fn in season_filters.items():
            df_season = s_fn(df_fst)
            for d_name, d_fn in dietary_categories.items():
                df_grp = d_fn(df_season)
                if len(df_grp) < 10:
                    continue
                stats = compute_group_stats(df_grp)
                if stats:
                    rec = {
                        "strata_type": "fst_proxy",
                        "strata_value": f"FST_{fst}",
                        "season": s_name,
                        "dietary_status": d_name,
                    }
                    rec.update(stats)
                    records.append(rec)

    # 2. Stratify by Race/Ethnicity
    race_map = {
        1: "Mexican_American",
        2: "Other_Hispanic",
        3: "Non_Hispanic_White",
        4: "Non_Hispanic_Black",
        6: "Non_Hispanic_Asian",
    }
    for r_code, r_name in race_map.items():
        df_race = valid[valid["race_ethnicity"] == r_code]
        for s_name, s_fn in season_filters.items():
            df_season = s_fn(df_race)
            for d_name, d_fn in dietary_categories.items():
                df_grp = d_fn(df_season)
                if len(df_grp) < 10:
                    continue
                stats = compute_group_stats(df_grp)
                if stats:
                    rec = {
                        "strata_type": "race_ethnicity",
                        "strata_value": r_name,
                        "season": s_name,
                        "dietary_status": d_name,
                    }
                    rec.update(stats)
                    records.append(rec)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} prior rows to: {OUTPUT_CSV}")
    print("\n--- Sample Prior Rows (Unsupplemented Winter/Spring) ---")
    sample = out_df[(out_df["season"] == "winter_spring") & (out_df["dietary_status"] == "all_unsupplemented")]
    print(sample[["strata_type", "strata_value", "season", "dietary_status", "weighted_mean", "p25", "p50", "p75", "n_unweighted"]].to_string(index=False))


if __name__ == "__main__":
    main()
