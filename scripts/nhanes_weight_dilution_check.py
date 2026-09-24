"""
nhanes_weight_dilution_check.py
===============================

Diagnostic: empirical scaling of serum 25(OH)D against body size in NHANES.

Fits survey-weighted log-log regressions on the standardized NHANES extract
(cycles 2001-2023, LC-MS/MS-harmonized 25(OH)D):

    ln(25(OH)D_nmol_L) = b0 + g * ln(body_size) + b_age * age + b_female * female

with body_size = weight_kg (primary) and body_size = bmi (comparison), and
tests the weight exponent g against three candidate scalings:

    g = -1.0   hyperbolic volumetric dilution (fixed input spread over body mass)
    g = -0.5   body-surface-area scaling (w ~ h^2, dose ~ area)
    g =  0.0   flat (no body-size association)

Standard errors are design-based: Taylor linearization of the weighted
estimating equations over SDMVPSU within SDMVSTRA, with t inference on
(PSU - stratum) design degrees of freedom. AIC/BIC use a Gaussian
pseudo-likelihood with weights rescaled to unit mean; both competing models
carry the same parameter count, so delta-AIC is invariant to the convention.

Analytic sample: adults (age >= 18) with vitd_standardized_nmol_L > 0,
weight_kg > 0, bmi > 0, wtmec2yr > 0. Primary analyses exclude the
'supplemented' dietary_source_flag ('UV-only' + 'food-only' only) to isolate
endogenous/dietary dynamics; an all-adults row is reported alongside.

Because cycles are pooled, weighted N is divided by the number of distinct
cycles to give an average-cycle-equivalent population (point estimates and
design-based SEs are invariant to any constant rescaling of weights).

Input : audit_outputs/nhanes_standardized_processed.csv
Output: tables/nhanes_weight_dilution_check.csv (+ formatted stdout tables)

Run: python3 scripts/nhanes_weight_dilution_check.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats as sps

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "audit_outputs", "nhanes_standardized_processed.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "tables", "nhanes_weight_dilution_check.csv")
FST_WEIGHT_CSV = os.path.join(BASE_DIR, "tables", "nhanes_fst_body_weight.csv")

RACE_ETHNICITY_MAP = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "NH White",
    4: "NH Black",
    6: "NH Asian",
}

SEASON_MAP = {
    1: "Nov-Apr (Winter/Spring)",
    2: "May-Oct (Summer/Autumn)",
}

HYPOTHESES = [
    (-1.0, "Supports 1/weight (volumetric dilution)", "1/weight"),
    (-0.5, "Consistent with BSA scaling (~w^-0.5)", "BSA-like"),
    (0.0, "Flat - no weight association", "flat"),
]

MIN_N_FIT = 100


def load_valid_adults():
    """Load the standardized extract, apply validity filters, report funnel."""
    df = pd.read_csv(INPUT_CSV)
    n_total = len(df)

    steps = []
    m = df["vitd_standardized_nmol_L"] > 0
    steps.append(("vitd_standardized_nmol_L > 0", int(m.sum())))
    m &= df["weight_kg"] > 0
    steps.append(("+ weight_kg > 0", int(m.sum())))
    m &= df["bmi"] > 0
    steps.append(("+ bmi > 0", int(m.sum())))
    m &= df["wtmec2yr"] > 0
    steps.append(("+ wtmec2yr > 0", int(m.sum())))
    m &= df["age"] >= 18
    steps.append(("+ age >= 18 (all valid adults)", int(m.sum())))

    valid = df.loc[m].copy()
    unsupp = valid.loc[
        valid["dietary_source_flag"].isin(["UV-only", "food-only"])
    ].copy()
    steps.append(
        ("+ dietary_source_flag in {UV-only, food-only} (unsupplemented)",
         len(unsupp))
    )

    for frame in (valid, unsupp):
        frame["ln_vitd"] = np.log(frame["vitd_standardized_nmol_L"].to_numpy())
        frame["ln_weight"] = np.log(frame["weight_kg"].to_numpy())
        frame["ln_bmi"] = np.log(frame["bmi"].to_numpy())
        frame["female"] = (frame["sex"].to_numpy() == 2).astype(float)

    return valid, unsupp, steps, sorted(valid["cycle"].unique())


def wls_design_fit(sub, size_col):
    """Survey-weighted log-log fit with Taylor-linearization design variance.

    Model: ln(vitd) ~ 1 + ln(size) + age + female.
    Returns beta, design-based covariance, design dof, weighted R^2,
    pseudo-likelihood AIC/BIC.
    """
    y = sub["ln_vitd"].to_numpy(dtype=float)
    w = sub["wtmec2yr"].to_numpy(dtype=float)
    X = np.column_stack(
        [
            np.ones(len(sub)),
            sub[size_col].to_numpy(dtype=float),
            sub["age"].to_numpy(dtype=float),
            sub["female"].to_numpy(dtype=float),
        ]
    )
    n, p = X.shape

    XtWX = X.T @ (X * w[:, None])
    XtWX_inv = np.linalg.pinv(XtWX)
    if np.linalg.matrix_rank(X) < p:
        return None
    beta = XtWX_inv @ (X.T @ (w * y))
    resid = y - X @ beta

    score = X * (w * resid)[:, None]
    sc = pd.DataFrame(score, columns=[f"x{j}" for j in range(p)])
    sc["strat"] = sub["sdmvstra"].to_numpy()
    sc["psu"] = sub["sdmvpsu"].to_numpy()
    agg = sc.groupby(["strat", "psu"], sort=False).sum()

    meat = np.zeros((p, p))
    n_psu_used = 0
    n_strata_used = 0
    for _, blk in agg.groupby(level=0, sort=False):
        M = blk.to_numpy(dtype=float)
        nh = M.shape[0]
        if nh < 2:
            continue
        C = M - M.mean(axis=0, keepdims=True)
        meat += (nh / (nh - 1.0)) * (C.T @ C)
        n_psu_used += nh
        n_strata_used += 1
    V = XtWX_inv @ meat @ XtWX_inv
    design_df = n_psu_used - n_strata_used

    w_sum = w.sum()
    ybar_w = float(np.sum(w * y) / w_sum)
    r2 = 1.0 - float(np.sum(w * resid**2)) / float(np.sum(w * (y - ybar_w) ** 2))

    wnorm = w / w.mean()
    sigma2 = float(np.sum(wnorm * resid**2)) / n
    llf = -0.5 * n * (np.log(2.0 * np.pi) + np.log(sigma2) + 1.0)
    k = p + 1
    aic = -2.0 * llf + 2.0 * k
    bic = -2.0 * llf + np.log(n) * k

    return {
        "beta": beta,
        "cov": V,
        "se": np.sqrt(np.diag(V)),
        "design_df": int(design_df),
        "r2": r2,
        "aic": aic,
        "bic": bic,
        "llf": llf,
        "n": n,
        "w_sum": float(w_sum),
    }


def t_test_vs(fit, value, param_idx=1):
    """Two-sided Wald t test of beta[param_idx] == value under design df."""
    est = fit["beta"][param_idx]
    se = fit["se"][param_idx]
    ddf = fit["design_df"]
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan
    tstat = (est - value) / se
    if ddf > 0:
        p = 2.0 * sps.t.sf(abs(tstat), ddf)
    else:
        p = 2.0 * sps.norm.sf(abs(tstat))
    return float(p), float(ddf)


def classify_gamma(g_hat, ci_lo, ci_hi):
    inside = [(v, long, short) for v, long, short in HYPOTHESES if ci_lo <= v <= ci_hi]
    if not inside:
        return f"Between hypotheses (excludes all of -1/-0.5/0)", "between"
    nearest = min(inside, key=lambda t: abs(g_hat - t[0]))
    ambiguous = len(inside) > 1
    if ambiguous:
        others = ", ".join(f"{t[0]:g}" for t in inside if t is not nearest)
        long = f"{nearest[1]} (closest; CI also contains {others})"
        short = f"{nearest[2]}*"
    else:
        long = nearest[1]
        short = nearest[2]
    return long, short


def fit_group(group_set, group_label, sub, n_cycles):
    row = {
        "group_set": group_set,
        "group": group_label,
        "n_unweighted": len(sub),
        "n_weighted_avg_cycle": np.nan,
    }
    if len(sub) < MIN_N_FIT:
        row["conclusion_long"] = f"insufficient n (< {MIN_N_FIT})"
        row["conclusion_short"] = "n too small"
        return row

    fit_w = wls_design_fit(sub, "ln_weight")
    fit_b = wls_design_fit(sub, "ln_bmi")
    row["n_weighted_avg_cycle"] = fit_w["w_sum"] / n_cycles

    g_hat = float(fit_w["beta"][1])
    se = float(fit_w["se"][1])
    ddf = max(int(fit_w["design_df"]), 0)
    tcrit = sps.t.ppf(0.975, ddf) if ddf > 0 else sps.norm.ppf(0.975)
    ci_lo, ci_hi = g_hat - tcrit * se, g_hat + tcrit * se

    p_m1, _ = t_test_vs(fit_w, -1.0)
    p_m05, _ = t_test_vs(fit_w, -0.5)
    p_0, _ = t_test_vs(fit_w, 0.0)
    conclusion_long, conclusion_short = classify_gamma(g_hat, ci_lo, ci_hi)

    row.update(
        {
            "gamma_ln_weight_kg": g_hat,
            "se_gamma": se,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "p_gamma_eq_minus1": p_m1,
            "p_gamma_eq_minus05": p_m05,
            "p_gamma_eq_0": p_0,
            "r2_weight_model": fit_w["r2"],
            "aic_weight_model": fit_w["aic"],
            "bic_weight_model": fit_w["bic"],
            "design_df": ddf,
            "gamma_ln_bmi": float(fit_b["beta"][1]) if fit_b else np.nan,
            "se_gamma_bmi": float(fit_b["se"][1]) if fit_b else np.nan,
            "r2_bmi_model": fit_b["r2"] if fit_b else np.nan,
            "aic_bmi_model": fit_b["aic"] if fit_b else np.nan,
            "bic_bmi_model": fit_b["bic"] if fit_b else np.nan,
            "delta_aic_bmi_minus_weight": (fit_b["aic"] - fit_w["aic"])
            if fit_b
            else np.nan,
            "conclusion_long": conclusion_long,
            "conclusion_short": conclusion_short,
        }
    )
    if fit_b is None:
        row["conclusion_long"] += "; BMI model rank-deficient"
    return row


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def build_groups(valid, unsupp, n_cycles):
    groups = [
        ("overall", "Adults, unsupplemented (UV+food)", unsupp),
        ("overall", "All adults (incl. supplemented)", valid),
    ]
    for code, label in RACE_ETHNICITY_MAP.items():
        groups.append((f"race={code}", label, unsupp.loc[unsupp["race_ethnicity"] == code]))
    for code, label in SEASON_MAP.items():
        groups.append((f"season={code}", label, unsupp.loc[unsupp["exam_season"] == code]))
    return groups


def print_main_table(results):
    hdr = (
        f"{'Group':<40} {'N unwt':>9} {'N wt(M)':>9} {'gamma':>8} "
        f"{'95% CI':>19} {'p(g=-1)':>8} {'p(g=-.5)':>8} {'p(g=0)':>8} "
        f"{'R2':>6}  Conclusion"
    )
    print("\n--- Survey-weighted log-log scaling of serum 25(OH)D vs body weight ---")
    print("(ln VitD ~ gamma*ln(weight) + age + sex; Taylor-linearized SEs, t(df))\n")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if not np.isfinite(r.get("gamma_ln_weight_kg", np.nan)):
            print(
                f"{r['group']:<40} {r['n_unweighted']:>9d} {'--':>9} "
                f"-- {r.get('conclusion_long', '')}"
            )
            continue
        ci = f"[{r['ci95_lo']:7.3f},{r['ci95_hi']:7.3f}]"
        print(
            f"{r['group']:<40} {r['n_unweighted']:>9,d} "
            f"{r['n_weighted_avg_cycle']/1e6:>9.1f} "
            f"{r['gamma_ln_weight_kg']:>8.3f} {ci:>19} "
            f"{fmt_p(r['p_gamma_eq_minus1']):>8} {fmt_p(r['p_gamma_eq_minus05']):>8} "
            f"{fmt_p(r['p_gamma_eq_0']):>8} {r['r2_weight_model']:>6.3f}  "
            f"{r['conclusion_short']}"
        )


def print_model_comparison(results):
    print("\n--- Weight model vs BMI model (same covariates; lower AIC preferred) ---")
    hdr = (
        f"{'Group':<40} {'AIC w':>10} {'AIC bmi':>10} {'dAIC':>9} "
        f"{'BIC w':>10} {'BIC bmi':>10} {'R2 w':>6} {'R2 bmi':>7}  Preferred"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if not np.isfinite(r.get("aic_weight_model", np.nan)):
            continue
        daic = r["delta_aic_bmi_minus_weight"]
        pref = "weight" if daic > 0 else ("BMI" if daic < 0 else "tie")
        if abs(daic) < 2.0:
            pref += " (~indistinguishable)"
        print(
            f"{r['group']:<40} {r['aic_weight_model']:>10.1f} "
            f"{r['aic_bmi_model']:>10.1f} {daic:>9.1f} "
            f"{r['bic_weight_model']:>10.1f} {r['bic_bmi_model']:>10.1f} "
            f"{r['r2_weight_model']:>6.3f} {r['r2_bmi_model']:>7.3f}  {pref}"
        )



# ---------------------------------------------------------------------------
# Per-FST population body weight -- the source of vitd_physiology.BODY_WEIGHT_BY_FST
# ---------------------------------------------------------------------------
# [ADDED 2026-09-22, decision A5 wire-in] The exponent this script fits is
# applied to a body weight, and until today nothing supplied one: the
# dilution term was inert because no caller set a weight. This section emits
# the per-FST survey-weighted mean body weight that is now the model's
# population default, so the exponent and the weights it multiplies come from
# ONE script, ONE analytic sample and ONE FST definition.
#
# FST ASSIGNMENT is the repository's own NHANES FST proxy, reproduced from
# scripts/generate_nhanes_priors_table.py rather than re-invented: the DEQ
# self-reported sun-reaction item where present, with a race/ethnicity
# fallback. Using the same proxy for weight as for baseline 25(OH)D keeps the
# two population defaults describing the same strata.
#
# SAMPLE is the UNSUPPLEMENTED analytic sample, matching this script's own
# primary regression row -- so the exponent and the weights describe the same
# people. The all-adults figures are emitted alongside for comparison.


def _fst_proxy(row):
    """Repository NHANES FST proxy (see generate_nhanes_priors_table.py)."""
    skin_rx = row.get("deq_skin_reaction")
    race = row.get("race_ethnicity")
    if pd.notna(skin_rx) and skin_rx in (1, 2, 3, 4, 5):
        if skin_rx == 5 and race == 4:
            return 6
        return int(skin_rx)
    if race == 4:
        return 5
    if race == 3:
        return 2
    if race in (1, 2, 6):
        return 4
    return np.nan


def _taylor_se_of_weighted_mean(sub, col="weight_kg"):
    """Design-based SE of a survey-weighted mean.

    Taylor linearization over SDMVPSU within SDMVSTRA, the same design basis
    the regressions above use. Returns (mean, se, design_df).
    """
    w = sub["wtmec2yr"].to_numpy(dtype=float)
    y = sub[col].to_numpy(dtype=float)
    W = w.sum()
    mu = float((w * y).sum() / W)
    z = w * (y - mu) / W
    strata = sub["sdmvstra"].to_numpy()
    psus = sub["sdmvpsu"].to_numpy()
    var, n_psu, n_str = 0.0, 0, 0
    for st in np.unique(strata):
        m = strata == st
        ps = np.unique(psus[m])
        if len(ps) < 2:
            continue
        tot = np.array([z[m & (psus == p)].sum() for p in ps])
        nh = len(ps)
        n_psu += nh
        n_str += 1
        var += nh / (nh - 1) * ((tot - tot.mean()) ** 2).sum()
    return mu, float(np.sqrt(var)), int(n_psu - n_str)


def fst_body_weight_table(valid, unsupp):
    """Survey-weighted mean body weight by FST proxy, both samples."""
    rows = []
    for sample_label, frame in (("unsupplemented (primary)", unsupp),
                                ("all adults", valid)):
        f = frame.copy()
        f["fst_proxy"] = f.apply(_fst_proxy, axis=1)
        for fst in range(1, 7):
            g = f.loc[f["fst_proxy"] == fst]
            if len(g) < MIN_N_FIT:
                continue
            mu, se, ddf = _taylor_se_of_weighted_mean(g)
            w = g["wtmec2yr"].to_numpy(dtype=float)
            y = g["weight_kg"].to_numpy(dtype=float)
            sd_w = float(np.sqrt((w * (y - mu) ** 2).sum() / w.sum()))
            rows.append({"sample": sample_label, "fst": fst,
                         "n_unweighted": int(len(g)),
                         "mean_weight_kg": mu, "se_mean_weight_kg": se,
                         "weighted_sd_kg": sd_w, "design_df": ddf})
    return pd.DataFrame(rows)


def print_fst_body_weight(df):
    print("\nPer-FST population body weight (the source of "
          "vitd_physiology.BODY_WEIGHT_BY_FST)")
    print(f"  {'sample':24s} {'FST':>4} {'n':>7} {'mean kg':>12} "
          f"{'SE':>9} {'wtd SD':>9} {'df':>5}")
    for _, r in df.iterrows():
        print(f"  {r['sample']:24s} {int(r['fst']):>4d} "
              f"{int(r['n_unweighted']):>7,d} {r['mean_weight_kg']:>12.6f} "
              f"{r['se_mean_weight_kg']:>9.6f} {r['weighted_sd_kg']:>9.4f} "
              f"{int(r['design_df']):>5d}")
    print("  Design-based SEs are 0.37-2.35 kg, so about four significant")
    print("  figures are meaningful. Full precision is in the CSV and in")
    print("  vitd_physiology.BODY_WEIGHT_BY_FST, which code reads back.")



def main():
    valid, unsupp, steps, cycles = load_valid_adults()
    n_cycles = len(cycles)

    print("NHANES body-size vs 25(OH)D scaling diagnostic")
    print(f"Input: {os.path.relpath(INPUT_CSV, BASE_DIR)}")
    print(f"Cycles pooled ({n_cycles}): {', '.join(cycles)}")
    print("\nSample funnel (cumulative):")
    for label, count in steps:
        print(f"  {label:<58} {count:>7,d}")

    groups = build_groups(valid, unsupp, n_cycles)
    results = [fit_group(gs, gl, sub, n_cycles) for gs, gl, sub in groups]

    out_cols = [
        "group_set", "group", "n_unweighted", "n_weighted_avg_cycle",
        "gamma_ln_weight_kg", "se_gamma", "ci95_lo", "ci95_hi",
        "p_gamma_eq_minus1", "p_gamma_eq_minus05", "p_gamma_eq_0",
        "r2_weight_model", "aic_weight_model", "bic_weight_model",
        "gamma_ln_bmi", "se_gamma_bmi", "r2_bmi_model",
        "aic_bmi_model", "bic_bmi_model", "delta_aic_bmi_minus_weight",
        "design_df", "conclusion_long",
    ]
    results_df = pd.DataFrame(results)[out_cols]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print_main_table(results)
    print_model_comparison(results)

    fst_wt = fst_body_weight_table(valid, unsupp)
    fst_wt.to_csv(FST_WEIGHT_CSV, index=False)
    print_fst_body_weight(fst_wt)
    print(f"  written to: {os.path.relpath(FST_WEIGHT_CSV, BASE_DIR)}")

    print(f"\nNotes: '*' marks groups whose 95% CI contains multiple candidate")
    print(f"exponents (-1/-0.5/0); see CSV conclusion_long for detail. Weight model")
    print(f"ln(weight_kg); BMI comparison uses ln(BMI); both adjust for age and sex.")
    print(f"Weighted N is an average-cycle-equivalent population (weights divided by")
    print(f"{n_cycles} pooled cycles); full-precision values saved to:")
    print(f"  {os.path.relpath(OUTPUT_CSV, BASE_DIR)}")


run_weight_dilution_check = main


if __name__ == "__main__":
    main()
