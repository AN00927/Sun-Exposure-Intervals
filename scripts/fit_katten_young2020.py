"""
fit_katten_young2020.py
=======================
Fit K_ATTEN (melanin attenuation coefficient for facultative tanning)
using per-person data from Young 2020 (P9, xlsx sheet 'TL01 irradiated', n=23 subjects).

Optimizes K_ATTEN by minimizing MSE between model-predicted and observed final 25(OH)D.
Computes 95% confidence intervals via subject-level bootstrap (B=1000).
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vitd_model import ModelParams, simulate, build_weekly_schedule, get_E_source

def load_young2020_tl01_data(xlsx_path: str) -> pd.DataFrame:
    """Load and preprocess TL01 irradiated arm dataset."""
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel file not found at: {xlsx_path}")
    
    df = pd.read_excel(xlsx_path, sheet_name="TL01 irradiated")
    df = df.dropna(subset=["VitD3.Sample_1", "VitD3.Sample_5"]).copy()
    
    fst_map = {"II": 2, "VI": 6}
    df["fst_num"] = df["FST"].map(fst_map)
    df = df.dropna(subset=["fst_num"]).copy()
    df["fst_num"] = df["fst_num"].astype(int)
    
    df["D0"] = df["VitD3.Sample_1"].astype(float)
    df["D5"] = df["VitD3.Sample_5"].astype(float)
    df["obs_delta"] = df["D5"] - df["D0"]
    return df

def run_fit(df: pd.DataFrame, n_bootstrap: int = 1000):
    """Precompute predictions and run vectorized grid search + bootstrap."""
    params = ModelParams()
    E_tl01 = get_E_source("tl01_nb_uvb")
    
    schedule = build_weekly_schedule(duration_days=42, sed_per_session=0.4167, days_of_week=[0, 2, 4])
    duration = 42
    
    k_grid = np.linspace(0.0, 2.0, 201)
    n_subjects = len(df)
    n_k = len(k_grid)
    
    # Precalculate predictions matrix (n_subjects x n_k)
    pred_matrix = np.zeros((n_subjects, n_k))
    obs_d5_array = df["D5"].values
    
    for i, (_, row) in enumerate(df.iterrows()):
        fst = int(row["fst_num"])
        d0 = float(row["D0"])
        for j, k_val in enumerate(k_grid):
            res = simulate(
                fst=fst,
                schedule=schedule,
                duration_days=duration,
                params=params,
                D0=d0,
                K_ATTEN_override=k_val,
                E_source_override=E_tl01,
                exposed_bsa_override=0.35
            )
            pred_matrix[i, j] = res["D"][-1]
            
    # MSE for whole sample
    errors_matrix = pred_matrix - obs_d5_array[:, None]
    mses = np.mean(np.square(errors_matrix), axis=0)
    best_idx = np.argmin(mses)
    optimal_k = float(k_grid[best_idx])
    min_mse = float(mses[best_idx])
    
    # Fast bootstrap
    rng = np.random.default_rng(42)
    boot_best_ks = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_subjects, size=n_subjects, replace=True)
        boot_errors = errors_matrix[boot_idx, :]
        boot_mses = np.mean(np.square(boot_errors), axis=0)
        boot_best_ks.append(float(k_grid[np.argmin(boot_mses)]))
        
    ci_low, ci_high = np.percentile(boot_best_ks, [2.5, 97.5])
    
    # Group statistics
    df_results = df.copy()
    df_results["pred_D5"] = pred_matrix[:, best_idx]
    df_results["pred_delta"] = df_results["pred_D5"] - df_results["D0"]
    
    fst_ii_obs = df_results[df_results["fst_num"] == 2]["obs_delta"].mean()
    fst_vi_obs = df_results[df_results["fst_num"] == 6]["obs_delta"].mean()
    obs_ratio = fst_ii_obs / fst_vi_obs if fst_vi_obs > 0 else np.nan
    
    fst_ii_pred = df_results[df_results["fst_num"] == 2]["pred_delta"].mean()
    fst_vi_pred = df_results[df_results["fst_num"] == 6]["pred_delta"].mean()
    pred_ratio = fst_ii_pred / fst_vi_pred if fst_vi_pred > 0 else np.nan
    
    # Calculate baseline-adjusted slopes using Bogh 2010 coefficient 0.223
    df_results["adj_obs_slope"] = (df_results["obs_delta"] + 0.223 * df_results["D0"]) / 7.5
    df_results["adj_pred_slope"] = (df_results["pred_delta"] + 0.223 * df_results["D0"]) / 7.5
    
    adj_obs_ii = df_results[df_results["fst_num"] == 2]["adj_obs_slope"].mean()
    adj_obs_vi = df_results[df_results["fst_num"] == 6]["adj_obs_slope"].mean()
    adj_obs_ratio = adj_obs_ii / adj_obs_vi
    
    return {
        "optimal_k": optimal_k,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min_mse": min_mse,
        "obs_fst_ii_delta": fst_ii_obs,
        "obs_fst_vi_delta": fst_vi_obs,
        "obs_ratio": obs_ratio,
        "adj_obs_ratio": adj_obs_ratio,
        "pred_fst_ii_delta": fst_ii_pred,
        "pred_fst_vi_delta": fst_vi_pred,
        "pred_ratio": pred_ratio,
        "n_subjects": len(df)
    }

def main():
    xlsx_path = os.path.join(BASE_DIR, "Papers_and_Datasets", "Final skin type vit D database 26_03_2020.xlsx")
    print(f"Loading data from: {xlsx_path}")
    df = load_young2020_tl01_data(xlsx_path)
    print(f"Loaded {len(df)} subjects (FST II: {(df['fst_num']==2).sum()}, FST VI: {(df['fst_num']==6).sum()})")
    
    print("Fitting K_ATTEN via precomputed 1D MSE minimization with B=1000 bootstrap...")
    res = run_fit(df, n_bootstrap=1000)
    
    print("\n--- FIT RESULTS ---")
    print(f"Optimal K_ATTEN: {res['optimal_k']:.4f} (95% CI: [{res['ci_low']:.4f}, {res['ci_high']:.4f}])")
    print(f"Minimum MSE: {res['min_mse']:.4f}")
    print(f"Observed FST II mean delta 25(OH)D: {res['obs_fst_ii_delta']:.2f} nmol/L")
    print(f"Observed FST VI mean delta 25(OH)D: {res['obs_fst_vi_delta']:.2f} nmol/L")
    print(f"Raw Observed FST II:VI delta ratio: {res['obs_ratio']:.4f}")
    print(f"Baseline-Adjusted FST II:VI slope ratio: {res['adj_obs_ratio']:.4f}")
    print(f"Predicted FST II:VI ratio at optimal K_ATTEN: {res['pred_ratio']:.4f}")

if __name__ == "__main__":
    main()
