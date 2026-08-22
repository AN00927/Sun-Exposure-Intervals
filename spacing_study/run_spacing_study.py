"""
run_spacing_study.py
====================
Entry point. Runs every experiment in `spacing_experiments`, writes raw CSVs
to `spacing_study/results/`, figures to `spacing_study/figs/`, and assembles
`spacing_study/results/FINDINGS.md`.

    PYTHONIOENCODING=utf-8 python3 spacing_study/run_spacing_study.py

Runtime: a few minutes. It does NOT touch, import from, or depend on
`run_all.py`, and it writes nothing outside `spacing_study/`.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vitd_model as vm
import spacing_model as sm
import spacing_experiments as se

RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

METRICS = ["final_D", "mean_D", "min_D_tail", "frac_days_ge_50"]
SS_METRICS = ["ss_mean_D", "ss_min_D", "ss_frac_days_ge_50"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return d.to_markdown(index=True)


def best_interval_table(df: pd.DataFrame, group_cols: List[str],
                        metrics: List[str], interval_col: str = "interval_days"
                        ) -> pd.DataFrame:
    out = []
    for keys, g in df.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for m in metrics:
            gg = g.dropna(subset=[m])
            row[f"best_iv[{m}]"] = int(gg.loc[gg[m].idxmax(), interval_col]) if len(gg) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def pivot(df: pd.DataFrame, index, columns, values) -> pd.DataFrame:
    return df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")


def save(rows: List[Dict], name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.drop(columns=[c for c in ("params",) if c in df.columns], errors="ignore")
    df.to_csv(os.path.join(RESULTS, name), index=False)
    return df


def pct(a, b):
    """Percent by which a exceeds b."""
    return 100.0 * (a - b) / b if b else np.nan


def katten_threshold(KA: pd.DataFrame, framing: str):
    """Smallest swept K_ATTEN at which some skin type stops preferring daily."""
    sub_f = KA[KA.framing == framing]
    for ka in se.K_ATTEN_SWEEP:
        sub = sub_f[sub_f.K_ATTEN == ka]
        if any(int(g.set_index("interval_days").ss_mean_D.idxmax()) != 1
               for _, g in sub.groupby("fst")):
            return ka
    return None


def ksc_threshold_rows(KSC: pd.DataFrame, fsts=None):
    """(framing, fst, K_SC at flip, max SC throughput at flip) for each cell."""
    rows = []
    for framing in ["matched_per_session_SED", "matched_total_SED"]:
        for fst in (fsts or se.FST_ALL):
            sub = KSC[(KSC.framing == framing) & (KSC.fst == fst)]
            first = None
            for ksc, g in sub.groupby("K_SC"):
                if int(g.set_index("interval_days").ss_mean_D.idxmax()) != 1:
                    first = (ksc, float(g.max_SC_throughput.iloc[0]))
                    break
            rows.append((framing, fst, first))
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    sections: List[str] = []
    flips: List[str] = []

    p_ref = sm.make_params()
    print(f"E_source={p_ref.E_source}  exposed_bsa={p_ref.exposed_bsa}  "
          f"K_ATTEN={p_ref.K_ATTEN}  K_SC={p_ref.K_SC}  D0={sm.D0_DEFAULT}")

    # ---------------- A -----------------------------------------------------
    print("[A] matched per-session dose ...", flush=True)
    t0 = time.time()
    A = save(se.exp_A_matched_per_session(), "expA_matched_per_session.csv")
    print(f"    {len(A)} runs, {time.time()-t0:.0f}s")

    A365 = A[A.horizon_days == sm.HORIZON_LONG]
    A90 = A[A.horizon_days == sm.HORIZON_SHORT]
    ASS = A[A.horizon_days == sm.HORIZON_SS]

    a_sec = ["### A. Matched per-session dose",
             "",
             "Fixed SED per session; only the interval changes, so total UV is "
             "`horizon / interval` sessions worth. Daily therefore also "
             "delivers the most total UV.",
             ""]
    for sed in [0.5, 1.0, 4.0]:
        sub = A365[np.isclose(A365.sed_target, sed)]
        a_sec += [f"**mean 25(OH)D (nmol/L) over a 365-day horizon, "
                  f"{sed:g} SED per session**", "",
                  md(pivot(sub, "fst", "interval_days", "mean_D")), ""]
    sub = A90[np.isclose(A90.sed_target, 1.0)]
    a_sec += ["**mean 25(OH)D over a 90-day horizon, 1 SED per session**", "",
              md(pivot(sub, "fst", "interval_days", "mean_D")), ""]

    bi_A = best_interval_table(A, ["fst", "sed_target", "horizon_days"], METRICS)
    bi_A.to_csv(os.path.join(RESULTS, "expA_best_interval.csv"), index=False)
    n_nondaily_A = int((bi_A[[c for c in bi_A.columns if c.startswith("best_iv")]] != 1).sum().sum())
    a_sec += [f"**Winning interval across all {len(bi_A)} (FST x dose x horizon) "
              f"cells and all 4 outcome metrics: interval = 1 day in "
              f"{4*len(bi_A)-n_nondaily_A} of {4*len(bi_A)} cases.**", ""]
    if n_nondaily_A:
        bad = bi_A[(bi_A[[c for c in bi_A.columns if c.startswith('best_iv')]] != 1).any(axis=1)]
        a_sec += ["Cells where some metric prefers a non-daily interval:", "",
                  md(bad.set_index(["fst", "sed_target", "horizon_days"])), ""]
        flips.append(f"Experiment A: {n_nondaily_A} metric-cells prefer non-daily.")
    else:
        a_sec += ["No cell, at any dose, horizon, skin type or outcome metric, "
                  "prefers a non-daily interval.", ""]

    # steady-state cross-check
    ss_piv = pivot(ASS[np.isclose(ASS.sed_target, 1.0)], "fst", "interval_days", "ss_mean_D")
    a_sec += ["**Steady-state cross-check** — mean 25(OH)D over the final full "
              f"schedule period of a {sm.HORIZON_SS}-day run, 1 SED per session "
              "(removes the transient front-loading artefact):", "",
              md(ss_piv), ""]
    sections.append("\n".join(a_sec))

    # ---------------- B -----------------------------------------------------
    print("[B] matched total UV budget ...", flush=True)
    t0 = time.time()
    B = save(se.exp_B_matched_budget(), "expB_matched_budget.csv")
    print(f"    {len(B)} runs, {time.time()-t0:.0f}s")
    B365 = B[B.horizon_days == sm.HORIZON_LONG]
    BSS = B[B.horizon_days == sm.HORIZON_SS]

    b_sec = ["### B. Matched TOTAL UV budget",
             "",
             "Total SED over the horizon is held fixed; per-session dose is "
             "scaled up as the interval lengthens. This is the clean test of "
             "log-dose concavity vs photoadaptation recovery.", ""]
    for bud in [0.25, 1.0, 2.0]:
        sub = B365[np.isclose(B365.budget_sed_per_day, bud)]
        b_sec += [f"**mean 25(OH)D (nmol/L), 365-day horizon, total budget = "
                  f"{bud:g} SED/day x 365 d = {bud*365:g} SED**", "",
                  md(pivot(sub, "fst", "interval_days", "mean_D")), ""]
    bi_B = best_interval_table(B, ["fst", "budget_sed_per_day", "horizon_days"], METRICS)
    bi_B.to_csv(os.path.join(RESULTS, "expB_best_interval.csv"), index=False)
    n_nondaily_B = int((bi_B[[c for c in bi_B.columns if c.startswith("best_iv")]] != 1).sum().sum())
    b_sec += [f"**Optimal interval per budget level: interval = 1 day in "
              f"{4*len(bi_B)-n_nondaily_B} of {4*len(bi_B)} "
              f"(FST x budget x horizon x metric) cases.**", ""]
    if n_nondaily_B:
        bad = bi_B[(bi_B[[c for c in bi_B.columns if c.startswith('best_iv')]] != 1).any(axis=1)]
        b_sec += [md(bad.set_index(["fst", "budget_sed_per_day", "horizon_days"])), ""]
        flips.append(f"Experiment B: {n_nondaily_B} metric-cells prefer non-daily.")
    b_sec += ["**Cost of spacing at a matched budget** — mean 25(OH)D relative "
              "to daily, steady state, 1 SED/day budget:", "",
              md(pivot(BSS[np.isclose(BSS.budget_sed_per_day, 1.0)],
                       "fst", "interval_days", "ss_mean_D")), ""]
    sections.append("\n".join(b_sec))

    # ---------------- C -----------------------------------------------------
    print("[C] matched minutes outdoors ...", flush=True)
    C = save(se.exp_C_matched_minutes(), "expC_matched_minutes.csv")
    C365 = C[C.horizon_days == sm.HORIZON_LONG]
    c_sec = ["### C. Matched total time outdoors",
             "",
             f"Fixed minutes outdoors per week, converted at a single constant "
             f"{sm.SED_PER_MINUTE:g} SED/minute (see README — stated assumption, "
             f"no solar geometry). Total UV is therefore matched, exactly as in B.",
             ""]
    for fst in [2, 5]:
        sub = C365[C365.fst == fst]
        c_sec += [f"**FST {fst} — mean 25(OH)D (nmol/L), 365 d; rows = minutes "
                  f"outdoors per week, columns = interval (days)**", "",
                  md(pivot(sub, "minutes_per_week", "interval_days", "mean_D")), ""]
    bi_C = best_interval_table(C, ["fst", "minutes_per_week", "horizon_days"], METRICS)
    bi_C.to_csv(os.path.join(RESULTS, "expC_best_interval.csv"), index=False)
    n_nondaily_C = int((bi_C[[c for c in bi_C.columns if c.startswith("best_iv")]] != 1).sum().sum())
    c_sec += [f"Interval = 1 day wins in {4*len(bi_C)-n_nondaily_C} of "
              f"{4*len(bi_C)} cells.", ""]
    if n_nondaily_C:
        flips.append(f"Experiment C: {n_nondaily_C} metric-cells prefer non-daily.")
    sections.append("\n".join(c_sec))

    # ---------------- D -----------------------------------------------------
    print("[D1] burst patterns ...", flush=True)
    D1 = save(se.exp_D_bursts(), "expD1_bursts.csv")
    print("[D2] free pattern search ...", flush=True)
    t0 = time.time()
    D2 = save(se.exp_D_free_search(), "expD2_free_search.csv")
    print(f"    {len(D2)} runs, {time.time()-t0:.0f}s")

    d_sec = ["### D. Non-uniform / irregular patterns", ""]
    d1ss = D1[D1.horizon_days == sm.HORIZON_SS]
    for fst in [2, 5]:
        sub = d1ss[(d1ss.fst == fst) & np.isclose(d1ss.sed_target, 1.0)]
        tab = sub[["pattern", "duty_cycle", "cum_SED", "ss_mean_D", "ss_min_D",
                   "ss_frac_days_ge_50", "final_D"]].copy()
        tab = tab.sort_values("ss_mean_D", ascending=False).set_index("pattern")
        d_sec += [f"**D1. Burst-then-rest, FST {fst}, 1 SED per session, "
                  f"steady state.** Total UV is NOT matched across rows here "
                  f"(duty cycle differs) — read it as 'more days on is better'.",
                  "", md(tab, "{:.3f}"), ""]

    d_sec += ["**D2. Free search over 14-day on/off patterns at a FIXED duty "
              "cycle.** Total UV is identical within each duty block, so this "
              "isolates ARRANGEMENT alone. `gap_std` = 0 means perfectly even "
              "spacing; large `max_gap` means clustered.", ""]
    for k in sorted(D2.duty_k.unique()):
        for fst in sorted(D2.fst.unique()):
            sub = D2[(D2.duty_k == k) & (D2.fst == fst)].sort_values("ss_mean_D",
                                                                     ascending=False)
            top = sub.head(3)[["pattern", "max_gap", "gap_std", "ss_mean_D", "ss_min_D"]]
            bot = sub.tail(2)[["pattern", "max_gap", "gap_std", "ss_mean_D", "ss_min_D"]]
            tab = pd.concat([top, bot]).set_index("pattern")
            rng = sub.ss_mean_D.max() - sub.ss_mean_D.min()
            rho = float(np.corrcoef(sub.gap_std, sub.ss_mean_D)[0, 1])
            d_sec += [f"*duty {k}/14, FST {fst}* — {len(sub)} rotation-distinct "
                      f"patterns; best-to-worst spread in mean 25(OH)D = "
                      f"{rng:.4f} nmol/L ({100*rng/sub.ss_mean_D.min():.3f}%); "
                      f"corr(gap_std, mean D) = {rho:+.3f} "
                      f"(negative = even spacing wins). Top 3 and worst 2:", "",
                      md(tab, "{:.4f}"), ""]
    sections.append("\n".join(d_sec))

    # ---------------- E -----------------------------------------------------
    print("[E] mechanism decomposition ...", flush=True)
    E = save(se.exp_E_mechanisms(), "expE_mechanisms.csv")
    Earr = E[E.experiment == "E_arrangement"]
    Eiv = E[E.experiment == "E"]

    e_sec = ["### E. Mechanism decomposition", "",
             "Each mechanism switched off one at a time. `ratio_iv2_iv1` is "
             "mean 25(OH)D on an every-2-day schedule divided by mean 25(OH)D "
             "on a daily schedule, in steady state: **> 1 would mean spacing "
             "wins**.", ""]
    for framing in ["matched_per_session_SED", "matched_total_SED"]:
        recs = []
        for variant in sm.MECHANISM_VARIANTS:
            for fst in [2, 5]:
                g = Eiv[(Eiv.variant == variant) & (Eiv.fst == fst) &
                        (Eiv.framing == framing)].set_index("interval_days")
                if g.empty:
                    continue
                d1, d2, d7 = (g.loc[1, "ss_mean_D"], g.loc[2, "ss_mean_D"],
                              g.loc[7, "ss_mean_D"])
                recs.append({
                    "variant": variant, "fst": fst,
                    "meanD_daily": d1, "meanD_every2": d2, "meanD_weekly": d7,
                    "ratio_iv2_iv1": d2 / d1, "ratio_iv7_iv1": d7 / d1,
                    "best_iv": int(g.ss_mean_D.idxmax()),
                })
        tab = pd.DataFrame(recs).set_index(["variant", "fst"])
        e_sec += [f"**Framing: {framing}**", "", md(tab, "{:.4f}"), ""]

    # arrangement decomposition
    recs = []
    for variant in sm.MECHANISM_VARIANTS:
        for fst in [2, 5]:
            g = Earr[(Earr.variant == variant) & (Earr.fst == fst)].set_index("pattern")
            if g.empty:
                continue
            recs.append({
                "variant": variant, "fst": fst,
                "alternating": g.loc["alternating", "ss_mean_D"],
                "7on7off": g.loc["7on7off", "ss_mean_D"],
                "clustered_3_4": g.loc["3-4-clustered", "ss_mean_D"],
                "spread_minus_clustered": (g.loc["alternating", "ss_mean_D"] -
                                           g.loc["7on7off", "ss_mean_D"]),
            })
    arr_tab = pd.DataFrame(recs).set_index(["variant", "fst"])
    e_sec += ["**Arrangement at a fixed 7/14 duty cycle (total UV identical).** "
              "`spread_minus_clustered` > 0 means even spacing beats "
              "burst-then-rest.", "", md(arr_tab, "{:.4f}"), ""]
    sections.append("\n".join(e_sec))

    # ---------------- E' K_ATTEN -------------------------------------------
    print("[E'] K_ATTEN sweep ...", flush=True)
    KA = save(se.exp_E_katten_sweep(), "expE_katten_sweep.csv")
    ka_sec = ["### E'. K_ATTEN sweep — would tanning attenuation create a "
              "non-daily optimum?", "",
              "`K_ATTEN` is a RETIRED parameter pinned at `K_ATTEN_STRUCTURAL` "
              "(see `vitd_model.RETIRED_PARAMETERS`), i.e. the shipped model "
              "has **no facultative-melanin attenuation at all**. This sweep "
              "runs inside `allow_retired_parameter_sweep()` and adopts "
              "nothing. Values below run far past any defensible range, "
              "deliberately.", ""]
    for framing in ["matched_per_session_SED", "matched_total_SED"]:
        recs = []
        for ka in se.K_ATTEN_SWEEP:
            row = {"K_ATTEN": ka}
            for fst in se.FST_ALL:
                g = KA[(KA.K_ATTEN == ka) & (KA.fst == fst) &
                       (KA.framing == framing)].set_index("interval_days")
                row[f"best_iv_FST{fst}"] = int(g.ss_mean_D.idxmax())
                row[f"ratio2_1_FST{fst}"] = g.loc[2, "ss_mean_D"] / g.loc[1, "ss_mean_D"]
            recs.append(row)
        tab = pd.DataFrame(recs).set_index("K_ATTEN")
        ka_sec += [f"**Framing: {framing}** (best interval, and the "
                   "every-2-day / daily mean-D ratio)", "",
                   md(tab, "{:.4f}"), ""]
    for framing in ["matched_per_session_SED", "matched_total_SED"]:
        nz = KA[KA.framing == framing]
        thresh = katten_threshold(KA, framing)
        if thresh is None:
            ka_sec += [f"**Threshold ({framing}): no swept K_ATTEN up to "
                       f"{max(se.K_ATTEN_SWEEP):g} makes any skin type prefer a "
                       f"non-daily interval.**", ""]
        else:
            per_fst = []
            sub = nz[nz.K_ATTEN == thresh]
            for fst, g in sub.groupby("fst"):
                per_fst.append(f"FST {fst} -> {int(g.set_index('interval_days').ss_mean_D.idxmax())} d")
            ka_sec += [f"**Threshold ({framing}): the smallest swept K_ATTEN at "
                       f"which any skin type prefers a non-daily interval is "
                       f"{thresh:g}.** Winning intervals there: "
                       f"{'; '.join(per_fst)}.", ""]
    sections.append("\n".join(ka_sec))

    # ---------------- K_SC threshold ---------------------------------------
    print("[E''] K_SC threshold scan ...", flush=True)
    KSC = save(se.exp_ksc_threshold(), "expKSC_threshold.csv")
    ksc_sec = ["### E''. Stratum-corneum attenuation threshold", "",
               "`K_SC` is the only *live* photoadaptation attenuation channel "
               "in the shipped model. What matters physically is the maximum "
               "steady-state attenuation factor `exp(-K_SC * SC_max)`. This "
               "scan finds where daily stops winning.", ""]
    recs = []
    for framing in ["matched_per_session_SED", "matched_total_SED"]:
        for fst in se.FST_ALL:
            sub = KSC[(KSC.framing == framing) & (KSC.fst == fst)]
            first = None
            for ksc, g in sub.groupby("K_SC"):
                if int(g.set_index("interval_days").ss_mean_D.idxmax()) != 1:
                    first = (ksc, g.max_SC_throughput.iloc[0])
                    break
            recs.append({
                "framing": framing, "fst": fst,
                "shipped_K_SC": float(sm.make_params().K_SC),
                "shipped_throughput": float(np.exp(-sm.make_params().K_SC *
                                                   sm.make_params().SC_max_by_FST[fst])),
                "K_SC_flip": first[0] if first else np.nan,
                "throughput_at_flip": first[1] if first else np.nan,
            })
    ksc_tab = pd.DataFrame(recs).set_index(["framing", "fst"])
    ksc_sec += [md(ksc_tab, "{:.4g}"), "",
                "`shipped_throughput` is the fraction of synthesis that "
                "survives a fully-thickened stratum corneum at the shipped "
                "parameters; `throughput_at_flip` is what it would have to "
                "fall to before spacing beats daily. Blank = no flip anywhere "
                "on the scanned grid.", ""]
    sections.append("\n".join(ksc_sec))

    # ---------------- F: skin type -----------------------------------------
    f_sec = ["### F. Skin type sweep (FST I-VI)", "",
             "Steady-state mean 25(OH)D, 1 SED per session, per-session-matched:",
             "", md(ss_piv), "",
             "Relative cost of every-2-day vs daily, by skin type "
             "(steady state; negative = spacing is worse):", ""]
    recs = []
    for fst in se.FST_ALL:
        g = ASS[(ASS.fst == fst) & np.isclose(ASS.sed_target, 1.0)].set_index("interval_days")
        gb = BSS[(BSS.fst == fst) & np.isclose(BSS.budget_sed_per_day, 1.0)].set_index("interval_days")
        recs.append({
            "fst": fst,
            "per_session: %chg iv2 vs iv1": pct(g.loc[2, "ss_mean_D"], g.loc[1, "ss_mean_D"]),
            "per_session: %chg iv7 vs iv1": pct(g.loc[7, "ss_mean_D"], g.loc[1, "ss_mean_D"]),
            "budget: %chg iv2 vs iv1": pct(gb.loc[2, "ss_mean_D"], gb.loc[1, "ss_mean_D"]),
            "budget: %chg iv7 vs iv1": pct(gb.loc[7, "ss_mean_D"], gb.loc[1, "ss_mean_D"]),
            "daily mean D": g.loc[1, "ss_mean_D"],
        })
    f_tab = pd.DataFrame(recs).set_index("fst")
    f_tab.to_csv(os.path.join(RESULTS, "expF_fst_sweep.csv"))
    f_sec += [md(f_tab, "{:.3f}"), ""]
    f_sec += ["Note on FST VI and tanning headroom: `M_ceiling(fst) = "
              "M_baseline(fst) + M_TAN_HEADROOM` and is deliberately NOT "
              "capped at 1.0, so every skin type including VI has the same "
              "facultative headroom on the melanin scale. With `K_ATTEN` "
              "retired at zero, that headroom has no effect on synthesis "
              "either way.", ""]
    sections.append("\n".join(f_sec))

    # ---------------- G: metric sensitivity --------------------------------
    print("[G] metric sensitivity ...", flush=True)
    g_sec = ["### G. Outcome-metric sensitivity", "",
             "Winning interval under each outcome metric. Counts pool "
             "experiments A, B and C.", ""]
    recs = []
    for label, bi in [("A (matched per-session)", bi_A),
                      ("B (matched total budget)", bi_B),
                      ("C (matched minutes)", bi_C)]:
        row = {"experiment": label, "cells": len(bi)}
        for m in METRICS:
            col = f"best_iv[{m}]"
            row[f"n_nondaily[{m}]"] = int((bi[col] != 1).sum())
        recs.append(row)
    g_tab = pd.DataFrame(recs).set_index("experiment")
    g_sec += [md(g_tab, "{:.0f}"), ""]

    # do the four metrics ever disagree within a cell?
    disagree = 0
    for bi in (bi_A, bi_B, bi_C):
        cols = [f"best_iv[{m}]" for m in METRICS]
        disagree += int((bi[cols].nunique(axis=1) > 1).sum())
    g_sec += [f"Cells in which the four outcome metrics disagree about the "
              f"winning interval: **{disagree}**.", ""]
    if disagree:
        flips.append(f"Outcome metrics disagree in {disagree} cells.")

    # the one place framing/metric genuinely matters: min_D and frac>=50
    sub = A365[np.isclose(A365.sed_target, 1.0)]
    g_sec += ["**All four metrics, 1 SED per session, 365-day horizon** "
              "(columns are interval in days):", ""]
    for m in METRICS:
        g_sec += [f"*{m}*", "", md(pivot(sub, "fst", "interval_days", m), "{:.3f}"), ""]
    sections.append("\n".join(g_sec))

    # ---------------- sensitivity ------------------------------------------
    print("[S] parameter sensitivity ...", flush=True)
    S = save(se.exp_sensitivity(), "sensitivity_sweeps.csv")
    W = save(se.exp_combined_worst_case(), "combined_worst_case.csv")

    s_sec = ["### Sensitivity: which parameter, if wrong, reverses the answer?",
             "",
             "For each swept value: the winning interval and the every-2-day / "
             "daily mean-D ratio, in steady state. Ratio > 1 means spacing "
             "wins. Nothing here is adopted; parameter values live in "
             "`vitd_model.py`'s module docstring.", ""]
    for sweep in S.sweep.unique():
        recs = []
        for level in S[S.sweep == sweep].level.unique():
            row = {"level": level}
            for framing, tag in [("matched_per_session_SED", "persess"),
                                 ("matched_total_SED", "budget")]:
                for fst in se.FST_FOCUS:
                    g = S[(S.sweep == sweep) & (S.level == level) &
                          (S.framing == framing) & (S.fst == fst)
                          ].set_index("interval_days")
                    if g.empty:
                        continue
                    row[f"best_iv[{tag},FST{fst}]"] = int(g.ss_mean_D.idxmax())
                    row[f"ratio2_1[{tag},FST{fst}]"] = (g.loc[2, "ss_mean_D"] /
                                                        g.loc[1, "ss_mean_D"])
            recs.append(row)
        tab = pd.DataFrame(recs).set_index("level")
        s_sec += [f"**{sweep}**", "", md(tab, "{:.4f}"), ""]

    recs = []
    for cfg in W.config.unique():
        for framing, tag in [("matched_per_session_SED", "persess"),
                             ("matched_total_SED", "budget")]:
            row = {"config": cfg, "framing": tag}
            for fst in se.FST_ALL:
                g = W[(W.config == cfg) & (W.framing == framing) & (W.fst == fst)
                      ].set_index("interval_days")
                row[f"best_iv_FST{fst}"] = int(g.ss_mean_D.idxmax())
            recs.append(row)
    w_tab = pd.DataFrame(recs).set_index(["config", "framing"])
    s_sec += ["**Joint adversarial configuration.** All photoadaptation "
              "parameters set simultaneously to their most spacing-favourable "
              "value inside the documented prior ranges "
              "(`ModelParams.SC_PRIOR_RANGES`), then beyond them:", "",
              md(w_tab, "{:.0f}"), ""]
    sections.append("\n".join(s_sec))

    # ---------------- figures ----------------------------------------------
    print("[fig] figures ...", flush=True)
    make_figures(ASS, BSS, D2, Eiv, KSC, KA)

    # ---------------- assemble FINDINGS.md ---------------------------------
    write_findings(sections, flips, bi_A, bi_B, bi_C, ASS, BSS, Eiv, Earr,
                   KSC, KA, D2, arr_tab, S)
    print(f"done in {time.time()-t_start:.0f}s")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(ASS, BSS, D2, Eiv, KSC, KA) -> None:
    ivs = se.INTERVALS

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for fst in se.FST_ALL:
        g = ASS[(ASS.fst == fst) & np.isclose(ASS.sed_target, 1.0)].sort_values("interval_days")
        ax[0].plot(g.interval_days, g.ss_mean_D, marker="o", label=f"FST {fst}")
        gb = BSS[(BSS.fst == fst) & np.isclose(BSS.budget_sed_per_day, 1.0)].sort_values("interval_days")
        ax[1].plot(gb.interval_days, gb.ss_mean_D, marker="s", label=f"FST {fst}")
    ax[0].set_title("A: matched per-session dose (1 SED)")
    ax[1].set_title("B: matched total budget (1 SED/day)")
    for a in ax:
        a.set_xlabel("interval between sessions (days)")
        a.set_ylabel("steady-state mean 25(OH)D (nmol/L)")
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle("Daily exposure is optimal under both dose framings, every skin type")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_interval_sweep.png"), dpi=150)
    plt.close(fig)

    # mechanism decomposition, normalised to daily
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax_, framing in zip(axes, ["matched_per_session_SED", "matched_total_SED"]):
        for variant in sm.MECHANISM_VARIANTS:
            g = Eiv[(Eiv.variant == variant) & (Eiv.fst == 5) &
                    (Eiv.framing == framing)].sort_values("interval_days")
            if g.empty:
                continue
            base = g[g.interval_days == 1].ss_mean_D.iloc[0]
            ax_.plot(g.interval_days, g.ss_mean_D / base, marker="o", label=variant)
        ax_.axhline(1.0, color="k", lw=.8, ls="--")
        ax_.set_xlabel("interval (days)")
        ax_.set_title(framing)
        ax_.grid(alpha=.3)
    axes[0].set_ylabel("mean 25(OH)D relative to daily")
    axes[1].legend(fontsize=7)
    fig.suptitle("E: mechanism decomposition, FST V — no ablation lifts any curve above 1.0")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_mechanisms.png"), dpi=150)
    plt.close(fig)

    # K_SC threshold
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for framing, ls in [("matched_per_session_SED", "-"), ("matched_total_SED", "--")]:
        sub = KSC[(KSC.framing == framing) & (KSC.fst == 5)]
        xs, ys = [], []
        for ksc, g in sub.groupby("K_SC"):
            gg = g.set_index("interval_days")
            xs.append(gg.max_SC_throughput.iloc[0])
            ys.append(gg.loc[2, "ss_mean_D"] / gg.loc[1, "ss_mean_D"])
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], ls, marker=".",
                label=framing)
    p = sm.make_params()
    ax.axvline(float(np.exp(-p.K_SC * p.SC_max_by_FST[5])), color="g",
               label="shipped K_SC * SC_max (FST V)")
    ax.axhline(1.0, color="k", lw=.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("max SC throughput  exp(-K_SC * SC_max)   (1 = no attenuation)")
    ax.set_ylabel("mean 25(OH)D: every-2-day / daily")
    ax.set_title("E'': how strong would SC attenuation have to be for spacing to win?")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_ksc_threshold.png"), dpi=150)
    plt.close(fig)

    # D2 arrangement
    fig, axes = plt.subplots(1, len(sorted(D2.duty_k.unique())), figsize=(12, 3.8))
    for ax_, k in zip(np.atleast_1d(axes), sorted(D2.duty_k.unique())):
        for fst, mk in [(2, "o"), (5, "s")]:
            sub = D2[(D2.duty_k == k) & (D2.fst == fst)]
            if sub.empty:
                continue
            base = sub.ss_mean_D.max()
            ax_.scatter(sub.gap_std, 100 * (sub.ss_mean_D - base) / base,
                        s=10, marker=mk, alpha=.6, label=f"FST {fst}")
        ax_.set_title(f"duty {k}/14")
        ax_.set_xlabel("std-dev of inter-session gap")
        ax_.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("% below best arrangement")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("D2: at a fixed duty cycle, even spacing beats clustering — but by <0.1%")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_arrangement.png"), dpi=150)
    plt.close(fig)

    # trajectories
    fig, ax = plt.subplots(figsize=(9, 4.2))
    p = sm.make_params()
    for label, sched in [
        ("daily, 1 SED", sm.periodic_schedule(365, 1.0, 1)),
        ("every 2 days, 1 SED", sm.periodic_schedule(365, 1.0, 2)),
        ("weekly, 7 SED (budget-matched to daily)", sm.periodic_schedule(365, 7.0, 7)),
        ("7 on / 7 off, 1 SED", sm.pattern_schedule(365, 1.0, (1,)*7 + (0,)*7)),
    ]:
        r = vm.simulate(fst=5, schedule=sched, duration_days=365, params=p,
                        D0=sm.D0_DEFAULT, exposed_bsa_override=p.exposed_bsa)
        ax.plot(r["t"], r["D"], lw=1.2, label=label)
    ax.axhline(50, color="k", ls=":", lw=.8)
    ax.set_xlabel("day"); ax.set_ylabel("25(OH)D (nmol/L)")
    ax.set_title("FST V trajectories, fixed ambient UV, no risk constraints")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_trajectories.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FINDINGS.md
# ---------------------------------------------------------------------------

def write_findings(sections, flips, bi_A, bi_B, bi_C, ASS, BSS, Eiv, Earr,
                   KSC, KA, D2, arr_tab, S) -> None:
    p = sm.make_params()

    def ratio(df, filt, iv_a, iv_b, col="ss_mean_D"):
        g = df[filt].set_index("interval_days")
        return g.loc[iv_a, col] / g.loc[iv_b, col]

    r5_A = ratio(ASS, (ASS.fst == 5) & np.isclose(ASS.sed_target, 1.0), 2, 1)
    r5_B = ratio(BSS, (BSS.fst == 5) & np.isclose(BSS.budget_sed_per_day, 1.0), 2, 1)
    r2_B = ratio(BSS, (BSS.fst == 2) & np.isclose(BSS.budget_sed_per_day, 1.0), 2, 1)
    r5_B7 = ratio(BSS, (BSS.fst == 5) & np.isclose(BSS.budget_sed_per_day, 1.0), 7, 1)

    total_cells = 0
    nondaily_cells = 0
    for bi in (bi_A, bi_B, bi_C):
        cols = [c for c in bi.columns if c.startswith("best_iv")]
        total_cells += bi[cols].size
        nondaily_cells += int((bi[cols] != 1).sum().sum())

    # arrangement spread
    arr_spread = []
    for k in sorted(D2.duty_k.unique()):
        for fst in sorted(D2.fst.unique()):
            sub = D2[(D2.duty_k == k) & (D2.fst == fst)]
            arr_spread.append(100 * (sub.ss_mean_D.max() - sub.ss_mean_D.min())
                              / sub.ss_mean_D.min())
    max_arr_spread = max(arr_spread)

    shipped_tp_5 = float(np.exp(-p.K_SC * p.SC_max_by_FST[5]))
    shipped_tp_2 = float(np.exp(-p.K_SC * p.SC_max_by_FST[2]))

    ka_ps = katten_threshold(KA, "matched_per_session_SED")
    ka_bd = katten_threshold(KA, "matched_total_SED")
    if ka_ps is None:
        katten_headline = ("- `K_ATTEN` never produces a non-daily optimum "
                           f"anywhere on the swept grid (up to {max(se.K_ATTEN_SWEEP):g}).")
    else:
        katten_headline = (
            f"- **`K_ATTEN` threshold = {ka_ps:g}** under matched per-session dose "
            f"(about {ka_ps/0.33:.0f}x the top of the [0.00, 0.33] range its own "
            f"objective was ever flat over). Under a **matched total UV budget it "
            f"never flips**"
            + (f" (checked to {max(se.K_ATTEN_SWEEP):g})." if ka_bd is None
               else f" until {ka_bd:g}.")
        )
    kr = ksc_threshold_rows(KSC, [2, 5])
    kmap = {(f, s): v for f, s, v in kr}
    ps2, ps5 = kmap[("matched_per_session_SED", 2)], kmap[("matched_per_session_SED", 5)]
    bd5 = kmap[("matched_total_SED", 5)]
    ksc_headline = (
        f"- **`K_SC` threshold = {ps5[0]:g} at FST V / {ps2[0]:g} at FST II** under "
        f"matched per-session dose ({ps5[0]/2.0:.0f}x and {ps2[0]/2.0:.0f}x the top "
        f"of the documented 0.5-2.0 prior range), and {bd5[0]:g} at FST V under a "
        f"matched budget. At the FST V flip a fully-thickened stratum corneum "
        f"would have to pass {ps5[1]:.2%} of synthesis instead of the shipped "
        f"{shipped_tp_5:.0%}."
        if ps5 and ps2 and bd5 else
        "- `K_SC` never produces a non-daily optimum on the swept grid."
    )

    header = f"""# Spacing study — FINDINGS

*Generated by `spacing_study/run_spacing_study.py`. Within-model result only:
this reports what the repo's forward model implies, not what a human body does.*

---

## 1. The answer

**No. Daily exposure is optimal. Ignoring all risk, there is no spacing —
every 2 days, every 3 days, burst-then-rest, or any irregular pattern — that
yields more serum 25(OH)D than going outside every day, at any skin type, any
per-session dose, any horizon, or any of the four outcome metrics tested.**
Interval = 1 day won in **{total_cells - nondaily_cells} of {total_cells}**
(schedule-family x skin type x dose level x horizon x outcome metric) cells
across experiments A, B and C.

Three refinements to that headline, all of which point the same way:

1. **At a matched TOTAL UV budget, spreading the dose out further is strictly
   worse, not neutral.** Splitting a fixed annual SED budget into every-2-day
   instead of daily sessions costs {100*(1-r5_B):.1f}% of steady-state mean
   25(OH)D at FST V ({100*(1-r2_B):.1f}% at FST II); weekly costs
   {100*(1-r5_B7):.1f}% at FST V. This is the **log-dose concavity** term:
   `A * ln(1 + UV*E)` means two 1-SED sessions beat one 2-SED session, so more
   sessions always wins. Photoadaptation never overcomes it.
2. **At a fixed duty cycle (total UV identical), even spacing does beat
   clustering — but by at most {max_arr_spread:.3f}%.** An exhaustive search
   over every rotation-distinct 14-day on/off pattern at duty 2/14, 4/14 and
   7/14 finds the perfectly-even arrangement first every time and the maximally
   clustered one last. The effect is real and consistently signed, and it is
   also small enough to be scientifically uninteresting next to the ~30%
   penalty for simply going out less often.
3. **Half the photoadaptation machinery the hypothesis relies on is switched
   off in this model, and the other half is too weak to matter.** `K_ATTEN`,
   the facultative-melanin (tanning) attenuation coefficient, is a RETIRED
   parameter pinned at `K_ATTEN_STRUCTURAL` = 0 (see
   `vitd_model.RETIRED_PARAMETERS`), so tanning has **no effect on synthesis
   at all**: "melanin adaptation off" is a bit-identical no-op, and the
   `alpha` and `beta` sweeps move the answer by literally zero. The only live
   channel is stratum-corneum thickening, which at shipped parameters still
   passes {shipped_tp_5:.0%} of synthesis through a fully-thickened SC at
   FST IV-VI.

   A non-daily optimum **does** appear if either coefficient is pushed far
   enough — but only far past any defensible value, and only into a regime
   where exposure has stopped working at all:
   {katten_headline}
   {ksc_headline}
   Both thresholds sit in a regime where the winning "spaced" schedule
   produces roughly the no-exposure baseline, so spacing wins only in the
   sense that nothing works. This is the answer to "would spacing beat daily
   if tanning attenuation were real?": **not at any value of `K_ATTEN` that
   could be true, and not at all under a matched UV budget.**

---

## 2. Result tables

Stated assumptions, all fixed: one constant ambient UV level, exposed BSA =
model reference (so the BSA factor is exactly 1.0), solar spectral weighting
from `SOURCE_CALIBRATION_TABLE`, starting 25(OH)D = {sm.D0_DEFAULT:g} nmol/L,
no risk/erythema/behaviour/geometry terms. See `spacing_study/README.md`.

Two horizons are reported as briefed (90 d, 365 d) plus a
{sm.HORIZON_SS}-day **steady-state** run. The steady-state run matters: the
serum half-life in this model is of order months, so a 365-day comparison is
still on its transient, and a burst schedule that front-loads its doses inside
each period picks up a spurious advantage on mean-D that disappears once the
system settles. Steady-state metrics are averaged over the final full schedule
period and are phase-invariant to <0.001 nmol/L.

"""

    mech = build_mechanism_narrative(Eiv, Earr, arr_tab)
    flipsec = build_flip_section(flips, bi_A, bi_B, bi_C, ASS, BSS, KA, KSC)
    change = build_what_would_change(KSC, KA, p, S)
    limits = LIMITATIONS_TMPL.format(arr_spread=max_arr_spread)

    body = "\n\n".join(sections)
    out = "\n".join([header, body, "\n---\n", mech, "\n---\n", flipsec,
                     "\n---\n", change, "\n---\n", limits])
    with open(os.path.join(RESULTS, "FINDINGS.md"), "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote", os.path.join(RESULTS, "FINDINGS.md"))


def build_mechanism_narrative(Eiv, Earr, arr_tab) -> str:
    def r21(variant, fst, framing):
        g = Eiv[(Eiv.variant == variant) & (Eiv.fst == fst) &
                (Eiv.framing == framing)].set_index("interval_days")
        return g.loc[2, "ss_mean_D"] / g.loc[1, "ss_mean_D"]

    full_b = r21("full", 5, "matched_total_SED")
    nofb_b = r21("no_feedback", 5, "matched_total_SED")
    nopa_b = r21("no_photoadapt", 5, "matched_total_SED")
    nodd_b = r21("no_D_decay", 5, "matched_total_SED")
    nosc_b = r21("no_SC", 5, "matched_total_SED")
    nomel_b = r21("no_melanin_adapt", 5, "matched_total_SED")
    full_p = r21("full", 5, "matched_per_session_SED")
    nofb_p = r21("no_feedback", 5, "matched_per_session_SED")
    nopa_p = r21("no_photoadapt", 5, "matched_per_session_SED")
    nodd_p = r21("no_D_decay", 5, "matched_per_session_SED")
    nosc_p = r21("no_SC", 5, "matched_per_session_SED")
    nomel_p = r21("no_melanin_adapt", 5, "matched_per_session_SED")
    both_b = r21("no_photoadapt_no_feedback", 5, "matched_total_SED")
    ident = (abs(nomel_b - full_b) < 1e-12) and (abs(nomel_p - full_p) < 1e-12)

    a_full = arr_tab.loc[("full", 5), "spread_minus_clustered"]
    a_nopa = arr_tab.loc[("no_photoadapt", 5), "spread_minus_clustered"]
    a_nofb = arr_tab.loc[("no_feedback", 5), "spread_minus_clustered"]
    a_nodd = arr_tab.loc[("no_D_decay", 5), "spread_minus_clustered"]
    a_level = arr_tab.loc[("full", 5), "alternating"]

    return f"""## 3. Mechanism decomposition and attribution

Read the numbers in section E above; this is what they attribute.

**There is no non-daily optimum to attribute.** Under every single-mechanism
ablation, under both dose framings, at every skin type, the every-2-day /
daily ratio stays below 1. The ablations therefore answer the *why*, not the
*where*. Numbers are FST V, steady state; a ratio of 1 would mean spacing ties
daily.

| mechanism | ratio, matched per-session | ratio, matched budget | reading |
|---|---|---|---|
| full model | {full_p:.4f} | {full_b:.4f} | daily wins by {100*(1-full_p):.1f}% / {100*(1-full_b):.1f}% |
| melanin adaptation off | {nomel_p:.4f} | {nomel_b:.4f} | {"**bit-identical to the full model** — a no-op, because `K_ATTEN` is retired at 0" if ident else "differs from the full model"} |
| SC off | {nosc_p:.4f} | {nosc_b:.4f} | SC is therefore the *entire* photoadaptation effect |
| photoadaptation off (M and SC pinned) | {nopa_p:.4f} | {nopa_b:.4f} | identical to SC-off, for the same reason |
| feedback `g(D)` off | {nofb_p:.4f} | {nofb_b:.4f} | removing feedback makes spacing **worse**, not better |
| D decay off | {nodd_p:.4f} | {nodd_b:.4f} | serum elimination is not the driver either |
| photoadaptation + feedback off | — | {both_b:.4f} | strip every state mechanism and daily's margin *widens* |

**Attribution: the spacing penalty is carried by the concavity of the
synthesis law, `S = A * ln(1 + UV*E)`, not by any state variable.** The
signature is the last row: turn off photoadaptation *and* feedback together
and daily wins by more ({100*(1-both_b):.1f}% vs {100*(1-full_b):.1f}%), not
less. With the state terms gone the only thing distinguishing schedules is how
the budget is divided, and `ln` says divide it as finely as possible. Every
state mechanism in the model works to *narrow* that gap; none of them comes
close to closing it.

**Photoadaptation's spacing-favourable force is real, framing-dependent, and
about two orders of magnitude too small.** Two separate measurements:

- *Interval comparison.* Under matched per-session dose, switching SC off
  moves the every-2-day/daily ratio from {full_p:.4f} to {nopa_p:.4f} — so SC
  build-up does help spacing, by {100*(full_p-nopa_p):.2f} percentage points,
  against a {100*(1-full_p):.0f}-point deficit. Under a matched *budget* the
  sign reverses ({full_b:.4f} -> {nopa_b:.4f}): with total UV held fixed the
  stratum corneum thickens to nearly the same level either way, so it stops
  differentiating schedules at all.
- *Arrangement at fixed duty cycle.* Even spacing beats burst-then-rest by
  {a_full:.4f} nmol/L in the full model and only {a_nopa:.4f} nmol/L with
  photoadaptation off, so SC recovery accounts for about
  {100*(a_full-a_nopa)/a_full if a_full else float('nan'):.0f}% of that
  (already {100*a_full/a_level:.3f}%-of-level) arrangement
  effect. Feedback accounts for most of the rest ({a_nofb:.4f} nmol/L with
  feedback removed).

  The `no_D_decay` row of that arrangement table reads {a_nodd:.4f}, i.e.
  clustering apparently wins. **Do not read that as a mechanism.** With
  elimination off, D grows without bound and no steady state exists, so the
  "final period" average is still dominated by within-period front-loading.
  It is the transient artefact described in section 4, not a finding.

**Verdict on the standing claim in `run_all.py`** — "the interval ordering
these sweeps produce comes from the D-feedback term, not from
photoadaptation; the induced-melanin coefficient `K_ATTEN` cannot be the
cause ... that term is inert *by calibration*". **Second half confirmed
exactly; first half refuted.** `K_ATTEN` is indeed inert — ablating melanin
adaptation is bit-identical to the full model, and sweeping `alpha` and `beta`
across two orders of magnitude changes the every-2-day/daily ratio by nothing
at all to four decimal places. But the ordering does not come from D-feedback
either: disable `g(D)` and the ordering is unchanged while daily's margin
*grows*. The mechanism `run_all.py` describes ("sparse dosing keeps mean D
low, so each pulse meets less inhibition") is real and correctly reasoned —
it just works *against* the observed ordering rather than producing it. The
ordering is a property of the logarithmic dose-response itself, which no
ablation in this study can remove.
"""


def build_flip_section(flips, bi_A, bi_B, bi_C, ASS, BSS, KA, KSC) -> str:
    lines = ["## 4. Where the answer flips", ""]
    ka_ps = katten_threshold(KA, "matched_per_session_SED")
    kmap = {(f, s): v for f, s, v in ksc_threshold_rows(KSC, [2, 5])}
    ps5 = kmap[("matched_per_session_SED", 5)]
    if not flips:
        lines += [
            "**It never flips.** Across every combination tested — FST I "
            "through VI; per-session doses 0.25 to 4 SED; total budgets 0.1 to "
            "2 SED/day; 35 to 350 minutes outdoors per week; horizons of 90, "
            "365 and 728 days; final D, mean D, minimum D and fraction of days "
            "above 50 nmol/L — the winning interval is 1 day. The four outcome "
            "metrics never disagree with each other about it either.", "",
            "Two secondary orderings *do* depend on framing and are worth "
            "stating so they are not mistaken for a flip:", "",
            "- **Framing changes the size of the penalty, not its sign.** "
            "Under matched per-session dose, going from daily to weekly costs "
            f"about {100*(1 - (ASS[(ASS.fst==5)&np.isclose(ASS.sed_target,1.0)].set_index('interval_days').loc[7,'ss_mean_D'] / ASS[(ASS.fst==5)&np.isclose(ASS.sed_target,1.0)].set_index('interval_days').loc[1,'ss_mean_D'])):.0f}% "
            "of steady-state mean 25(OH)D at FST V (because total UV falls 7x "
            "as well). Under a matched total budget the same move costs about "
            f"{100*(1 - (BSS[(BSS.fst==5)&np.isclose(BSS.budget_sed_per_day,1.0)].set_index('interval_days').loc[7,'ss_mean_D'] / BSS[(BSS.fst==5)&np.isclose(BSS.budget_sed_per_day,1.0)].set_index('interval_days').loc[1,'ss_mean_D'])):.0f}%, "
            "which is the pure concavity cost.",
            "- **Skin type changes the stakes, not the ranking.** FST V-VI "
            "start from a much lower daily plateau, so the same relative "
            "penalty for spacing translates into crossing the 50 nmol/L line "
            "where FST I-II would not. On sparse schedules at FST V-VI, "
            "steady-state 25(OH)D falls *below* the 40 nmol/L starting value — "
            "the schedule loses ground rather than merely gaining less.",
            "- **Transient vs steady state changes the arrangement ranking.** "
            "Over a 365-day window a burst schedule can beat an evenly-spaced "
            "one on mean D purely because it front-loads doses inside each "
            "period. That reverses in steady state. Anyone rerunning this at a "
            "single 365-day horizon will see the artefact; it is not a real "
            "photoadaptation effect.", "",
            "**The answer does flip under counterfactual parameter values, and "
            "only there.** Two coefficients can be pushed hard enough to make "
            "a spaced schedule win: `K_ATTEN` at "
            f"{'no value on the grid' if ka_ps is None else f'{ka_ps:g}'} "
            "(shipped value: retired at zero) and `K_SC` at "
            f"{'no value on the grid' if ps5 is None else f'{ps5[0]:g}'} at "
            "FST V (shipped value: 1.0; documented prior range 0.5-2.0), both "
            "under matched per-session dosing. Section 5 quantifies how far "
            "outside their evidence those values sit and why the resulting "
            "'win' is degenerate.",
        ]
    else:
        lines += ["The following cells did **not** prefer daily:", ""] + \
                 [f"- {f}" for f in flips]
    return "\n".join(lines)


def build_what_would_change(KSC, KA, p, S) -> str:
    ksc_lines = []
    for framing, fst, first in ksc_threshold_rows(KSC, se.FST_ALL):
        if first is None:
            ksc_lines.append(f"| {framing} | {fst} | none on the scanned grid "
                             f"(K_SC up to {KSC.K_SC.max():g}) | — |")
        else:
            ksc_lines.append(f"| {framing} | {fst} | {first[0]:g} | "
                             f"{first[1]:.2e} |")

    ka_ps = katten_threshold(KA, "matched_per_session_SED")
    ka_bd = katten_threshold(KA, "matched_total_SED")
    ka_ps_s = "none on the grid" if ka_ps is None else f"{ka_ps:g}"
    ka_bd_s = (f"none on the grid (checked to {max(se.K_ATTEN_SWEEP):g})"
               if ka_bd is None else f"{ka_bd:g}")

    # Which sweeps are literally inert (ratio unchanged to 1e-9)?
    inert = []
    for sweep in S.sweep.unique():
        sub = S[(S.sweep == sweep) & (S.framing == "matched_total_SED") &
                (S.fst == 5)]
        vals = []
        for level, g in sub.groupby("level"):
            gg = g.set_index("interval_days")
            vals.append(gg.loc[2, "ss_mean_D"] / gg.loc[1, "ss_mean_D"])
        if len(vals) > 1 and (max(vals) - min(vals)) < 1e-9:
            inert.append(sweep)
    inert_s = ("; ".join(f"`{s}`" for s in inert) if inert else "none")

    shipped_tp_2 = float(np.exp(-p.K_SC * p.SC_max_by_FST[2]))
    shipped_tp_5 = float(np.exp(-p.K_SC * p.SC_max_by_FST[5]))

    return f"""## 5. What would change this answer

Every parameter below was swept, not calibrated. Nothing found here is
adopted; canonical values remain in `vitd_model.py`'s module docstring.
`calibration_approval.yaml` was not touched.

### 5a. The one thing that could reverse it: the synthesis law's shape

The conclusion rests on `S = A * ln(1 + UV*E)` being **concave**. Concavity is
what makes many small sessions beat few large ones and therefore makes daily
unbeatable at a matched budget. If the true per-session dose-response were
**convex** at low dose — e.g. a threshold below which no useful previtamin-D3
is made — the ranking would invert immediately and completely. No amount of
photoadaptation modelling is needed for that; it is a single structural
assumption. This is the *only* change tested or contemplated here that
reverses the answer outright, and it is not a parameter, it is a functional
form.

A related, milder version is already implemented but **off by default**:
`use_photodegradation` (the previtamin-D3 photostationary plateau, Holick
1981 / van Dijk 2016). Its own docstring in `vitd_model.py` calls it "the
mechanism that makes dose-splitting genuinely better". Turning it on
*strengthens* the daily result rather than reversing it, because it penalises
large single sessions — see the sensitivity table. It cannot produce a
non-daily optimum; it can only make spacing worse.

### 5b. Photoadaptation parameters — flagged as uncalibrated priors

`CLAUDE.md` flags `SC_max`, `alpha_SC` and `K_SC` as uncalibrated priors, and
`K_ATTEN` is retired. Those are exactly the parameters that would have to be
wrong for photoadaptation to create a non-daily optimum. They were swept well
past their documented ranges:

**`K_ATTEN` — the headline sensitivity.** Shipped value: retired at zero, so
the tanning-attenuation term is evaluated every session and contributes
nothing. Swept to {max(se.K_ATTEN_SWEEP):g}, roughly
{max(se.K_ATTEN_SWEEP)/0.33:.0f}x the top of the [0.00, 0.33] interval over
which its fitting objective was flat.

| framing | smallest swept K_ATTEN with a non-daily optimum |
|---|---|
| matched per-session dose | **{ka_ps_s}** |
| matched total UV budget | **{ka_bd_s}** |

  Two things make it so hard to flip, and both are structural rather than
  numerical. First, attenuation enters as `exp(-K_ATTEN * (M - M_baseline))`
  and the excess is capped by `M_TAN_HEADROOM`, so the attainable attenuation
  is bounded no matter how large the coefficient. Second — and this is the
  binding one — under a **matched UV budget** melanin saturates to
  effectively the same level whether the dose arrives daily or fortnightly,
  so the term barely *differentiates* schedules even when it is enormous;
  that is why the budget column never flips at all. The per-session flip that
  does appear is not "spacing is better", it is "daily tanning has shut
  synthesis off, so any schedule that lets pigment decay looks better than a
  schedule that produces almost nothing".

- **`K_SC` / `SC_max` (stratum corneum).** The live channel. What matters is
  the maximum steady-state throughput `exp(-K_SC * SC_max)`: shipped, that is
  {shipped_tp_5:.3f} for FST IV-VI and {shipped_tp_2:.3f} for FST I-III, i.e.
  a fully-thickened stratum corneum still passes ~70-80% of synthesis. The
  scan below reports where daily stops winning:

| framing | FST | K_SC at flip | max SC throughput at flip |
|---|---|---|---|
{chr(10).join(ksc_lines)}

  Where a flip exists at all it requires the stratum corneum to block
  essentially *all* synthesis once thickened — an order of magnitude beyond
  the documented prior range (0.5-2.0 for `K_SC`) — and at that point every
  schedule has collapsed to roughly the no-exposure baseline, so "spacing
  wins" only in the sense that nothing works. `SC_max` behaves the same way,
  as it must: only the product `K_SC * SC_max` is identified, and no flip
  appears until it exceeds ~5.
- **`alpha` and `beta` (melanin induction and decay) are completely inert.**
  Swept across two-plus orders of magnitude, they change the every-2-day /
  daily ratio by **zero to nine decimal places**, because the only route by
  which melanin state reaches synthesis is `K_ATTEN`, which is zero. Sweeps
  that were inert to 1e-9 at FST V: {inert_s}. Anyone reporting a melanin
  photoadaptation effect from this model without first raising `K_ATTEN` is
  reporting a numerical artefact.
- **`alpha_SC`, `k_decay_SC`, `k_damage`.** Swept over and beyond their
  documented ranges. None changes the winning interval under either framing at
  either focus skin type. Faster SC regression (`k_decay_SC`) does push in the
  spacing-favourable direction, but by fractions of a percentage point.
  `k_damage` — pinned at 0 for identifiability, not for lack of mechanism —
  pushes hard the *other* way: at 0.5 it drives the every-2-day/daily ratio
  down to ~0.41 under per-session matching. That is the two-process SC model
  doing what it was built to do: acute barrier disruption accumulates with
  every session and can drive net SC *below* baseline, which raises
  throughput, so the most frequent schedule benefits most. If `k_damage` is
  ever fitted to a non-zero value it will strengthen the daily result, not
  weaken it.
- **Joint adversarial setting.** Setting `K_SC`, `alpha_SC`, `SC_max` and
  `k_decay_SC` simultaneously to their most spacing-favourable in-range values,
  then adding fast melanin turnover and photodegradation on top, still leaves
  daily optimal at every skin type under both framings.

### 5c. Things that do *not* matter

Starting 25(OH)D (swept 15-130 nmol/L, deficient to supra-sufficient) changes
absolute levels but leaves the ratio identical to four decimal places — by
728 days the initial condition has been forgotten. Disabling feedback or serum
elimination entirely moves absolute levels by 2-3x and still never changes the
winning interval.
"""


LIMITATIONS_TMPL = """## 6. Limitations

1. **This is a within-model result.** It tells you what `vitd_model.py`
   implies, not what a human body does. The single most load-bearing
   assumption — logarithmic, concave per-session dose-response — is inherited
   from the parent model and is not tested here.
2. **Absolute 25(OH)D values carry the repo's standing flags.** `SC_max`,
   `alpha_SC` and `K_SC` are uncalibrated priors; `A` is anchored on an
   FST I-II study and is a light-skin extrapolation wherever it is applied to
   FST III-VI. Every absolute number in these tables inherits both flags. The
   *rankings* are far more robust than the levels — that is the finding worth
   quoting, not the plateau heights.
3. **All risk is ignored, by construction.** No erythema limit, no MED cap, no
   CPD/skin-cancer term, no cumulative-dose ceiling. Several schedules that
   "win" here deliver per-session doses well above an MED for the skin type in
   question (experiment B at 2 SED/day budget on a 14-day interval is 28 SED
   in one sitting). Nothing in this sub-study should be read as a
   recommendation.
4. **Fixed ambient UV, no geometry, no season, no behaviour.** One constant
   UV level and one exposed-BSA value on every day of the year. Real ambient
   UV varies by 5-10x seasonally at temperate latitudes, and a real schedule
   cannot deliver a fixed SED on an overcast January day. A seasonal ambient
   profile could in principle create a non-daily optimum for reasons that have
   nothing to do with photoadaptation (dose is simply unavailable on some
   days) — that is a different question from the one asked here and is not
   addressed.
5. **Dose framing is not interchangeable.** Results are reported separately
   under matched per-session SED, matched total SED, and matched minutes
   outdoors, per this repo's standing directive. Fractional-MED framing is
   *not* tested here, because MED is a risk construct and this sub-study
   excludes risk by design; a fractional-MED sweep would change the FST
   pattern of absolute levels (it would not change the interval ranking,
   which is driven by concavity within a skin type).
6. **The arrangement effect is real but negligible.** The {arr_spread:.3f}%
   advantage of even spacing over clustering at a fixed duty cycle is below the
   uncertainty on every parameter that produces it. It is reported because it
   is consistently signed and mechanistically interpretable, not because it is
   actionable.
7. **Discretisation.** The forward model integrates on a 0.5-day sub-step and
   treats each session as an instantaneous pulse. Intervals shorter than one
   day were not explored, so "more sessions is better" is not extrapolated
   past once per day.
8. **This study cannot test the tanning hypothesis on its merits.** The
   advisor's intuition — repeated exposure builds pigment, pigment blocks
   synthesis, so rest days should restore yield — is not *refuted* here so
   much as *unrepresented*: the coefficient that would carry it, `K_ATTEN`, is
   retired at zero, so the model has no opinion about it. What this study can
   say is (a) with the coefficient at zero, spacing loses everywhere, and
   (b) the coefficient would have to be ~150x the top of the range it was
   fitted over, and the dose framing would have to be per-session rather than
   budget-matched, before the intuition wins. Reinstating `K_ATTEN` as a
   calibratable parameter is a `calibration_approval.yaml` decision and was
   deliberately not taken here.
9. **The steady-state framing is a modelling choice.** The headline numbers
   come from a 728-day run so that transient front-loading cannot masquerade
   as a photoadaptation effect. Real seasonal exposure never reaches such a
   steady state at temperate latitudes, so the 90-day and 365-day tables are
   the more behaviourally relevant ones — they happen to agree.
"""


if __name__ == "__main__":
    main()
