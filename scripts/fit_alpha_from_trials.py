import sys, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vitd_model import ModelParams, simulate

FST_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
E_SOURCE_MAP = {"ArimedB": 1.78, "TL01": 2.35}
PULSE_DAYS = [0, 3, 7, 10, 14]
DURATION_DAYS = 17
EXPOSED_BSA = 0.85
ALPHA_DEFAULTS = {1: 0.0025, 2: 0.0025, 3: 0.0025, 4: 0.002, 5: 0.002, 6: 0.002}


def extract_subject_data(row, source):
    fst = FST_MAP[row["FST"].strip()]
    D0 = row["VitD3.Sample_1"]

    uv_col_prefix = "UVdose" if source == "ArimedB" else "UVDose"
    uv_doses = [row[f"{uv_col_prefix}.Sample_{i}"] for i in range(1, 6)]

    schedule = list(zip(PULSE_DAYS, uv_doses))
    observed_d3 = [row[f"VitD3.Sample_{i}"] for i in range(1, 6)]

    return {"ID": row["ID"], "FST": fst, "D0": D0, "schedule": schedule,
            "observed_d3": observed_d3, "E_source": E_SOURCE_MAP[source]}


def compute_rmse(predicted, observed):
    pred_arr = np.array(predicted, dtype=float)
    obs_arr = np.array(observed, dtype=float)
    mask = ~np.isnan(obs_arr)
    if np.sum(mask) < 1:
        return np.nan
    return float(np.sqrt(np.mean((pred_arr[mask] - obs_arr[mask]) ** 2)))


def extract_predicted_d3(result):
    t, D = result["t"], result["D"]
    vals = []
    for day in PULSE_DAYS:
        idx = np.where(np.abs(t - day) < 1e-9)[0]
        if len(idx) > 0:
            vals.append(float(D[idx[-1]]))
        else:
            vals.append(np.nan)
    return vals


def fit_alpha_for_subject(subject):
    fst = subject["FST"]
    schedule = subject["schedule"]
    D0 = subject["D0"]
    E_source = subject["E_source"]
    observed_d3 = subject["observed_d3"]

    obs_arr = np.array(observed_d3, dtype=float)
    mask = ~np.isnan(obs_arr)
    if np.sum(mask) < 2:
        return None, None, None

    def objective(alpha):
        params = ModelParams()
        M0 = params.M_baseline(fst)
        result = simulate(fst=fst, schedule=schedule, duration_days=DURATION_DAYS,
                          params=params, D0=D0, M0=M0, SC0=0.0,
                          E_source_override=E_source,
                          exposed_bsa_override=EXPOSED_BSA,
                          alpha_override=alpha)
        pred = extract_predicted_d3(result)
        return compute_rmse(pred, observed_d3)

    res = minimize_scalar(objective, bounds=(0.0005, 0.02), method="bounded")
    if not res.success:
        return None, None, None

    fitted_alpha = res.x
    final_rmse = objective(fitted_alpha)
    return fitted_alpha, final_rmse, extract_predicted_d3(
        simulate(fst=fst, schedule=schedule, duration_days=DURATION_DAYS,
                 params=ModelParams(), D0=D0,
                 M0=ModelParams().M_baseline(fst), SC0=0.0,
                 E_source_override=E_source,
                 exposed_bsa_override=EXPOSED_BSA,
                 alpha_override=fitted_alpha))


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "Papers_and_Datasets", "other")
    output_dir = os.path.join(base_dir, "audit_outputs")
    os.makedirs(output_dir, exist_ok=True)

    df_arimedb = pd.read_csv(os.path.join(data_dir, "Trial_SkinType_VitaminD_2020_ArimedB_irradiated.csv"))
    df_tl01 = pd.read_csv(os.path.join(data_dir, "Trial_SkinType_VitaminD_2020_TL01_irradiated.csv"))

    all_results = []

    for source, df in [("ArimedB", df_arimedb), ("TL01", df_tl01)]:
        print("=" * 80)
        print(f"{source} subjects")
        print("=" * 80)
        for _, row in df.iterrows():
            subject = extract_subject_data(row, source)
            fitted_alpha, rmse, _ = fit_alpha_for_subject(subject)
            if fitted_alpha is not None:
                print(f"  {subject['ID']}: FST={subject['FST']}, alpha={fitted_alpha:.6f}, "
                      f"RMSE={rmse:.4f}, D0={subject['D0']:.1f}")
                all_results.append({"ID": subject["ID"], "FST": subject["FST"],
                                    "source": source, "fitted_alpha": fitted_alpha,
                                    "rmse": rmse, "D0": subject["D0"]})
            else:
                print(f"  {subject['ID']}: FST={subject['FST']}, fit FAILED, D0={subject['D0']:.1f}")

    results_df = pd.DataFrame(all_results)
    out_file = os.path.join(output_dir, "alpha_fitted_results.csv")
    results_df.to_csv(out_file, index=False)
    print(f"\nResults saved to {out_file}")

    print("\n" + "=" * 80)
    print("FST-stratified alpha summary")
    print("=" * 80)
    for source_name in ["ArimedB", "TL01", "combined"]:
        subset = results_df if source_name == "combined" else results_df[results_df["source"] == source_name]
        if len(subset) == 0:
            continue
        print(f"\n{source_name}:")
        grouped = subset.groupby("FST")["fitted_alpha"]
        for fst, group in grouped:
            print(f"  FST {fst}: mean_alpha={group.mean():.6f}, std_alpha={group.std():.6f}, n={len(group)}")

    print("\n" + "=" * 80)
    print("Comparison to current defaults")
    print("=" * 80)
    print("  alpha_by_FST defaults: {1: 0.0025, 2: 0.0025, 3: 0.0025, 4: 0.002, 5: 0.002, 6: 0.002}\n")
    combined_grouped = results_df.groupby("FST")["fitted_alpha"]
    for fst in sorted(combined_grouped.groups.keys()):
        mean_alpha = combined_grouped.mean()[fst]
        default = ALPHA_DEFAULTS.get(fst, 0.0025)
        print(f"  FST {fst}: fitted_mean={mean_alpha:.6f}, default={default}, ratio={mean_alpha / default:.3f}")


if __name__ == "__main__":
    main()
