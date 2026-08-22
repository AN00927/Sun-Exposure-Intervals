"""
vitd_analyses.py
================
Analyses for the vitamin D scheduling model:

  A. alpha sensitivity sweep (photoadaptation rise rate, FST V & VI)
  B. K_ATTEN sensitivity (synthesis attenuation coefficient)
  C. Schedule comparison under TWO framings (matched SED vs matched dose)
  D. Two-framing summary plot
  E. beta sensitivity sweep (tanning decay rate)
  F. Fine-grid interval sweep (1-30 days, 1-day steps)
  G. exhaustive integer interval sweep (1-60 days; replaces a former
     scipy.optimize.minimize_scalar call that ran a continuous line search
     on an objective that rounds to the nearest integer day)
  H. Maximum interval for sufficiency (D_ss >= 50 nmol/L threshold)
  I. Loading dose optimization (distribute UV budget across first 14 days)
  J. Maintenance dose splitting (single vs split sessions)
  K. Sun exposure scheduling (minutes of midday sun, no equipment)

All outputs saved to ./figs/ as PNG and ./tables/ as CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vitd_scenario
from vitd_model import (simulate, build_pulse_schedule,
                         steady_state_D, mean_final_M,
                         final_D, net_final_D_change_per_SED,
                          auc_above_baseline_per_SED,
                          auc_above_sufficiency_per_SED,
                          auc_deficit,
                          get_E_source, E_SOURCE_SOLAR_DEFAULT)

def print_safety_summary(safety_log, label=""):
    """One-line safety summary from a simulate() safety_log."""
    if not safety_log:
        print(f"  Safety{label}: no safety data")
        return
    max_frac = 0.0
    max_sed = 0.0
    for _t, s in safety_log:
        frac = s.get("fraction_med", 0)
        med = s.get("individual_med_sed", 1.0)
        sed = frac * med
        if frac > max_frac:
            max_frac = frac
            max_sed = sed
    total_sed = sum(
        s.get("fraction_med", 0) * s.get("individual_med_sed", 1.0)
        for _t, s in safety_log
    )
    n_sessions = len(safety_log)
    marker = " *** OVERDOSE ***" if max_frac > 1.0 else ""
    print(f"  Safety{label}: max single dose = {max_frac:.1%} MED "
          f"({max_sed:.2f} SED), cumulative = {total_sed:.1f} SED "
          f"over {n_sessions} sessions{marker}")

FIG_DIR = "figs"
TAB_DIR = "tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

INTERVALS_DAYS = [1, 2, 3, 5, 7, 10, 14, 21]
PROTOCOL_DAYS = 84   # 12 weeks


# ===========================================================================
# Analysis A: alpha sensitivity sweep
# ===========================================================================
def analysis_alpha_sweep():
    """
    Vary alpha (photoadaptation rise rate) and observe how the OPTIMAL
    inter-session interval shifts. Highest priority because alpha was
    calibrated from light-skin (Miyamura 2011) and may be larger in darker
    skin.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS A: alpha sensitivity sweep")
    print("=" * 70)
    alphas = [0.01, 0.025, 0.05, 0.10, 0.20, 0.40]
    fsts = [5, 6]
    sed_per_session = 1.0

    rows = []
    for fst in fsts:
        for alpha in alphas:
            for interval in INTERVALS_DAYS:
                sched = build_pulse_schedule(PROTOCOL_DAYS, sed_per_session, interval)
                r = simulate(fst=fst, schedule=sched,
                             duration_days=PROTOCOL_DAYS,
                             alpha_override=alpha)
                rows.append({
                    "FST": fst,
                    "alpha": alpha,
                    "interval_days": interval,
                    "cumulative_SED": r["cumulative_SED"],
                    "steady_state_D": steady_state_D(r),
                    "final_D": final_D(r),
                    "final_M": mean_final_M(r),
                    "net_final_D_change_per_SED": net_final_D_change_per_SED(r),
                    "auc_above_baseline_per_SED": auc_above_baseline_per_SED(r),
                    "auc_above_sufficiency_per_SED": auc_above_sufficiency_per_SED(r),
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/alpha_sweep.csv", index=False)

    # Plot: heatmap of ABSOLUTE steady-state D by interval and alpha.
    # (This used to be preceded by a matching "efficiency" heatmap of the
    # retired per-SED efficiency metric, removed 2026-07-31 -- see
    # vitd_model.py -- and swapping its panel to steady_state_D would have
    # made it an exact duplicate of this one, so the panel was deleted
    # instead of duplicated.)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, fst in zip(axes, fsts):
        sub = df[df["FST"] == fst]
        pivot = sub.pivot(index="alpha", columns="interval_days",
                          values="steady_state_D")
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       cmap="magma")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{a:.3f}" for a in pivot.index])
        ax.set_xlabel("inter-session interval (days)")
        ax.set_ylabel(r"$\alpha$ (photoadaptation rise per SED)")
        ax.set_title(f"FST {['I','II','III','IV','V','VI'][fst-1]}: "
                     "steady-state D (nmol/L)")
        for ri, alpha_val in enumerate(pivot.index):
            argmax_col = np.argmax(pivot.iloc[ri].values)
            ax.plot(argmax_col, ri, "r*", markersize=15,
                    markeredgecolor="white")
        plt.colorbar(im, ax=ax)
    fig.suptitle(r"$\alpha$ sensitivity: absolute-D framing (fixed 1 SED/session)"
                 "\nRed star = interval that maximizes steady-state D",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/alpha_sweep_absolute_D.png", dpi=130)
    plt.close(fig)

    # Summary print: optimal interval by alpha
    print("\nOptimal interval by alpha (steady-state 25(OH)D framing, fixed 1 SED/session):")
    for fst in fsts:
        sub = df[df["FST"] == fst]
        print(f"\nFST {fst}:")
        for alpha in alphas:
            sub2 = sub[sub["alpha"] == alpha]
            best = sub2.loc[sub2["steady_state_D"].idxmax()]
            print(f"  alpha={alpha:.3f}:  optimal interval = "
                  f"{int(best['interval_days'])}d "
                  f"(ss D {best['steady_state_D']:.1f}, "
                  f"net_eff {best['net_final_D_change_per_SED']:.2f})")

    return df


# ===========================================================================
# Analysis B: K_ATTEN sensitivity sweep
# ===========================================================================
def analysis_K_ATTEN_sweep():
    """
    Vary K_ATTEN (melanin penalty coefficient). Young 2020 measured
    vitamin D inhibition of only 1.3-1.4x between FST II and VI, suggesting
    K_ATTEN may be lower than the default 2.0.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS B: K_ATTEN sensitivity sweep")
    print("=" * 70)
    K_vals = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    fsts = [5, 6]
    sed_per_session = 1.0

    rows = []
    for fst in fsts:
        for K in K_vals:
            for interval in INTERVALS_DAYS:
                sched = build_pulse_schedule(PROTOCOL_DAYS, sed_per_session, interval)
                r = simulate(fst=fst, schedule=sched,
                             duration_days=PROTOCOL_DAYS,
                             K_ATTEN_override=K)
                rows.append({
                    "FST": fst,
                    "K_ATTEN": K,
                    "interval_days": interval,
                    "cumulative_SED": r["cumulative_SED"],
                    "steady_state_D": steady_state_D(r),
                    "final_D": final_D(r),
                    "final_M": mean_final_M(r),
                    "net_final_D_change_per_SED": net_final_D_change_per_SED(r),
                    "auc_above_baseline_per_SED": auc_above_baseline_per_SED(r),
                    "auc_above_sufficiency_per_SED": auc_above_sufficiency_per_SED(r),
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/K_ATTEN_sweep.csv", index=False)

    # This used to be a 2x2 grid (top row: efficiency heatmap of the retired
    # per-SED efficiency metric, bottom row: steady-state D heatmap). The top
    # row was removed 2026-07-31 along with the metric -- swapping it to
    # steady_state_D would have made it an exact duplicate of the bottom row,
    # so the grid collapses to 1x2.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, fst in enumerate([5, 6]):
        sub = df[df["FST"] == fst]
        pd_ = sub.pivot(index="K_ATTEN", columns="interval_days",
                        values="steady_state_D")
        ax = axes[col]
        im = ax.imshow(pd_.values, aspect="auto", origin="lower",
                       cmap="magma")
        ax.set_xticks(range(len(pd_.columns))); ax.set_xticklabels(pd_.columns)
        ax.set_yticks(range(len(pd_.index)))
        ax.set_yticklabels([f"{k:.2f}" for k in pd_.index])
        ax.set_xlabel("interval (days)"); ax.set_ylabel("K_ATTEN")
        ax.set_title(f"FST {fst}: steady-state D")
        for ri in range(len(pd_.index)):
            ax.plot(np.argmax(pd_.iloc[ri].values), ri,
                    "r*", markersize=14, markeredgecolor="white")
        plt.colorbar(im, ax=ax)
    fig.suptitle("K_ATTEN sensitivity (fixed 1 SED/session): steady-state D",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/K_ATTEN_sweep.png", dpi=130)
    plt.close(fig)

    print("\nOptimal interval by K_ATTEN (steady-state 25(OH)D framing):")
    for fst in fsts:
        sub = df[df["FST"] == fst]
        print(f"\nFST {fst}:")
        for K in K_vals:
            sub2 = sub[sub["K_ATTEN"] == K]
            best_D = sub2.loc[sub2["steady_state_D"].idxmax()]
            print(f"  K={K:.1f}:  "
                  f"opt interval={int(best_D['interval_days'])}d "
                  f"(D {best_D['steady_state_D']:.1f}, "
                  f"net_eff {best_D['net_final_D_change_per_SED']:.2f})")
    return df


# ===========================================================================
# Analysis C: schedule comparison under two framings
# ===========================================================================
def analysis_two_framings():
    """
    Two framings side-by-side for FST V and VI:

    Framing 1: MATCHED CUMULATIVE SED (UV budget unconstrained, distributing
               a fixed total). Total = 42 SED over 84 days.
               Daily would deliver 0.5 SED/session; every-21-day would
               deliver 10.5 SED/session.

    Framing 2: MATCHED PER-SESSION DOSE (UV is the constrained resource).
               1 SED per session regardless of interval. Daily yields 84
               cumulative SED; every-21-day yields ~4 cumulative SED.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS C: two-framing schedule comparison")
    print("=" * 70)

    fsts = [5, 6]
    TOTAL_SED = 42.0   # for framing 1

    rows_f1 = []
    rows_f2 = []
    for fst in fsts:
        for interval in INTERVALS_DAYS:
            # Framing 1: matched cumulative SED
            n_sessions = (PROTOCOL_DAYS + interval - 1) // interval
            sed_per_f1 = TOTAL_SED / n_sessions
            sched1 = build_pulse_schedule(PROTOCOL_DAYS, sed_per_f1, interval)
            r1 = simulate(fst=fst, schedule=sched1, duration_days=PROTOCOL_DAYS, dose_framing="matched_cumulative_SED")
            rows_f1.append({
                "FST": fst,
                "interval_days": interval,
                "sed_per_session": sed_per_f1,
                "n_sessions": n_sessions,
                "cumulative_SED": r1["cumulative_SED"],
                "steady_state_D": steady_state_D(r1),
                "final_D": final_D(r1),
                "final_M": mean_final_M(r1),
                "net_final_D_change_per_SED": net_final_D_change_per_SED(r1),
                "auc_above_baseline_per_SED": auc_above_baseline_per_SED(r1),
                "auc_above_sufficiency_per_SED": auc_above_sufficiency_per_SED(r1),
            })
            # Framing 2: matched per-session dose
            sched2 = build_pulse_schedule(PROTOCOL_DAYS, 1.0, interval)
            r2 = simulate(fst=fst, schedule=sched2, duration_days=PROTOCOL_DAYS, dose_framing="fixed_per_session_SED")
            rows_f2.append({
                "FST": fst,
                "interval_days": interval,
                "sed_per_session": 1.0,
                "n_sessions": len(r2["synthesis_log"]),
                "cumulative_SED": r2["cumulative_SED"],
                "steady_state_D": steady_state_D(r2),
                "final_D": final_D(r2),
                "final_M": mean_final_M(r2),
                "net_final_D_change_per_SED": net_final_D_change_per_SED(r2),
                "auc_above_baseline_per_SED": auc_above_baseline_per_SED(r2),
                "auc_above_sufficiency_per_SED": auc_above_sufficiency_per_SED(r2),
            })

    df1 = pd.DataFrame(rows_f1)
    df2 = pd.DataFrame(rows_f2)
    df1.to_csv(f"{TAB_DIR}/framing1_matched_SED.csv", index=False)
    df2.to_csv(f"{TAB_DIR}/framing2_matched_per_session.csv", index=False)

    # Side-by-side plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, fst in enumerate([5, 6]):
        s1 = df1[df1["FST"] == fst]
        s2 = df2[df2["FST"] == fst]
        # Framing 1
        ax = axes[0, col]
        ax.plot(s1["interval_days"], s1["steady_state_D"],
                "o-", color="C0", label="steady-state D")
        ax.set_xlabel("interval (days)")
        ax.set_ylabel("steady-state D (nmol/L)", color="C0")
        ax.tick_params(axis='y', labelcolor="C0")
        ax.legend(loc="upper left")
        
        # The retired per-SED efficiency metric (removed 2026-07-31) used to
        # share this twin axis with net_final_D_change_per_SED; steady-state D
        # is already on the primary axis above, so it is not duplicated here.
        ax2 = ax.twinx()
        ax2.plot(s1["interval_days"], s1["net_final_D_change_per_SED"],
                 "d--", color="C1", label="net ΔD per SED")
        ax2.set_ylabel("net ΔD per SED\n(nmol/L per SED)", color="C1")
        ax2.tick_params(axis='y', labelcolor="C1")
        ax2.legend(loc="upper right")
        ax.set_title(f"FST {fst}: Framing 1 — matched cumulative SED ({TOTAL_SED:.0f} SED)")
        ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
        ax.grid(alpha=0.3)
        
        # Framing 2
        ax = axes[1, col]
        ax.plot(s2["interval_days"], s2["steady_state_D"],
                "o-", color="C0", label="steady-state D")
        ax.set_xlabel("interval (days)")
        ax.set_ylabel("steady-state D (nmol/L)", color="C0")
        ax.tick_params(axis='y', labelcolor="C0")
        ax.legend(loc="upper left")
        
        ax2 = ax.twinx()
        ax2.plot(s2["interval_days"], s2["net_final_D_change_per_SED"],
                 "d--", color="C1", label="net ΔD per SED")
        ax2.set_ylabel("net ΔD per SED\n(nmol/L per SED)", color="C1")
        ax2.tick_params(axis='y', labelcolor="C1")
        ax2.legend(loc="upper right")
        ax.set_title(f"FST {fst}: Framing 2 — matched 1 SED/session (cumulative varies)")
        ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
        ax.grid(alpha=0.3)

    fig.suptitle("Two-framing comparison: optimal interval depends on framing",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/two_framings.png", dpi=130)
    plt.close(fig)

    print("\nFraming 1 (matched cumulative SED = 42):")
    print(df1.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nFraming 2 (matched 1 SED per session):")
    print(df2.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    max_int = max(INTERVALS_DAYS)
    n_s = (PROTOCOL_DAYS + max_int - 1) // max_int
    sed_f1 = TOTAL_SED / n_s
    sched = build_pulse_schedule(PROTOCOL_DAYS, sed_f1, max_int)
    r1_safe = simulate(fst=fsts[0], schedule=sched, duration_days=PROTOCOL_DAYS)
    print_safety_summary(r1_safe.get("safety_log", []),
        f" [FST {fsts[0]}, framing 1, {max_int}d interval, {sed_f1:.1f} SED/session]")
    sched2 = build_pulse_schedule(PROTOCOL_DAYS, 1.0, max_int)
    r2_safe = simulate(fst=fsts[0], schedule=sched2, duration_days=PROTOCOL_DAYS)
    print_safety_summary(r2_safe.get("safety_log", []),
        f" [FST {fsts[0]}, framing 2, {max_int}d interval, 1.0 SED/session]")
    return df1, df2


# ===========================================================================
# Analysis E: beta sensitivity sweep (tanning decay rate)
# ===========================================================================
def analysis_beta_sweep():
    """
    Vary beta (photoadaptation decay rate) over the range compatible with
    Ravnbak 2008 (PMID 18480919), which measured UVB tanning fade of 4-5
    months in Scandinavians and Indians/Pakistanis.

    "Complete fading" at ~3 e-folding times (12.5% residual) implies half-life
    ~40-50 days; at ~4 e-folding times (6% residual), ~30-37 days. Plausible
    range ~30-75 days. Sweep extends beyond for robustness.

    Analogous to analysis_alpha_sweep and analysis_K_ATTEN_sweep.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS E: beta sensitivity sweep (tanning decay rate)")
    print("=" * 70)

    half_lives_days = [21, 30, 56, 80, 120]
    betas = [np.log(2) / t for t in half_lives_days]
    fsts = [5, 6]
    sed_per_session = 1.0

    rows = []
    for fst in fsts:
        for hl, beta_val in zip(half_lives_days, betas):
            for interval in INTERVALS_DAYS:
                sched = build_pulse_schedule(PROTOCOL_DAYS, sed_per_session, interval)
                r = simulate(fst=fst, schedule=sched,
                             duration_days=PROTOCOL_DAYS,
                             k_decay_M_override=beta_val)
                rows.append({
                    "FST": fst,
                    "beta_halflife_days": hl,
                    "beta": beta_val,
                    "interval_days": interval,
                    "cumulative_SED": r["cumulative_SED"],
                    "steady_state_D": steady_state_D(r),
                    "final_D": final_D(r),
                    "final_M": mean_final_M(r),
                    "net_final_D_change_per_SED": net_final_D_change_per_SED(r),
                    "auc_above_baseline_per_SED": auc_above_baseline_per_SED(r),
                    "auc_above_sufficiency_per_SED": auc_above_sufficiency_per_SED(r),
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/beta_sweep.csv", index=False)

    # This used to be a 2x2 grid (top row: efficiency heatmap of the retired
    # per-SED efficiency metric, bottom row: steady-state D heatmap), matching
    # Analysis B. The top row was removed 2026-07-31 along with the metric --
    # swapping it to steady_state_D would have made it an exact duplicate of
    # the bottom row, so the grid collapses to 1x2.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, fst in enumerate([5, 6]):
        sub = df[df["FST"] == fst]
        pd_ = sub.pivot(index="beta_halflife_days", columns="interval_days",
                        values="steady_state_D")
        ax = axes[col]
        im = ax.imshow(pd_.values, aspect="auto", origin="lower", cmap="magma")
        ax.set_xticks(range(len(pd_.columns))); ax.set_xticklabels(pd_.columns)
        ax.set_yticks(range(len(pd_.index)))
        ax.set_yticklabels([f"{t}d" for t in pd_.index])
        ax.set_xlabel("interval (days)"); ax.set_ylabel("beta half-life (days)")
        ax.set_title(f"FST {fst}: steady-state D")
        for ri in range(len(pd_.index)):
            ax.plot(np.argmax(pd_.iloc[ri].values), ri,
                    "r*", markersize=14, markeredgecolor="white")
        plt.colorbar(im, ax=ax)

    fig.suptitle(
        "beta sensitivity: tanning decay half-life (Ravnbak 2008 plausible range: ~30-75 d)\n"
        "steady-state D.  Red star = optimal interval.",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/beta_sweep.png", dpi=130)
    plt.close(fig)

    print("\nOptimal interval by beta half-life (steady-state 25(OH)D framing, fixed 1 SED/session):")
    for fst in fsts:
        sub = df[df["FST"] == fst]
        print(f"\nFST {fst}:")
        for hl in half_lives_days:
            sub2 = sub[sub["beta_halflife_days"] == hl]
            best_D = sub2.loc[sub2["steady_state_D"].idxmax()]
            print(f"  t1/2={hl:3d}d:  "
                  f"opt interval={int(best_D['interval_days'])}d "
                  f"(D {best_D['steady_state_D']:.1f}, "
                  f"net_eff {best_D['net_final_D_change_per_SED']:.2f})")

    return df


# ===========================================================================
# Analysis F: fine-grid interval sweep (1-30 days, every 1 day)
# ===========================================================================
def analysis_fine_grid():
    """
    Test every 1-day interval from 1 to 30 days. High-resolution efficiency
    curve for FST V and VI.

    Fixed 1 SED per session, 84-day protocol.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS F: fine-grid interval sweep (1-30 days)")
    print("=" * 70)
    intervals = list(range(1, 31))
    fsts = [5, 6]
    sed_per_session = 1.0

    rows = []
    for fst in fsts:
        for interval in intervals:
            sched = build_pulse_schedule(PROTOCOL_DAYS, sed_per_session, interval)
            r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL_DAYS)
            rows.append({
                "FST": fst,
                "interval_days": interval,
                "cumulative_SED": r["cumulative_SED"],
                "steady_state_D": steady_state_D(r),
                "final_D": final_D(r),
                "final_M": mean_final_M(r),
                "net_final_D_change_per_SED": net_final_D_change_per_SED(r),
                "auc_above_baseline_per_SED": auc_above_baseline_per_SED(r),
                "auc_deficit_30": auc_deficit(r["D"], r["t"], 30.0),
                "auc_deficit_50": auc_deficit(r["D"], r["t"], 50.0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/fine_grid_sweep.csv", index=False)

    # Plot: steady-state D vs interval for both FSTs. The retired per-SED
    # efficiency metric (removed 2026-07-31) used to share a single axis here
    # with net_final_D_change_per_SED, since both were per-SED ratios on a
    # common scale. steady_state_D is an absolute quantity, not a per-SED
    # ratio, so it goes on its own twin axis (the same pattern already used
    # in analysis_two_framings) rather than sharing a mismatched scale.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, fst in zip(axes, fsts):
        sub = df[df["FST"] == fst]
        ax.plot(sub["interval_days"], sub["steady_state_D"],
                "o-", color="C0", label="steady-state D")
        # Mark the argmax
        best_idx = sub["steady_state_D"].idxmax()
        best = sub.loc[best_idx]
        ax.plot(best["interval_days"], best["steady_state_D"],
                "r*", markersize=18, markeredgecolor="white",
                label=f"optimal: {int(best['interval_days'])}d")
        ax.set_xlabel("inter-session interval (days)")
        ax.set_ylabel("steady-state D (nmol/L)", color="C0")
        ax.tick_params(axis='y', labelcolor="C0")
        ax.set_title(f"FST {['I','II','III','IV','V','VI'][fst-1]}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(sub["interval_days"], sub["net_final_D_change_per_SED"],
                 "s--", color="C1", label="net ΔD per SED")
        ax2.set_ylabel("net ΔD per SED (nmol/L per SED)", color="C1")
        ax2.tick_params(axis='y', labelcolor="C1")
        ax2.legend(loc="upper right", fontsize=9)
    fig.suptitle("Fine-grid sweep: steady-state D vs interval (1 SED/session)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fine_grid_efficiency.png", dpi=130)
    plt.close(fig)

    # Summary print
    print("\nFine-grid optimal interval (steady-state 25(OH)D framing):")
    for fst in fsts:
        sub = df[df["FST"] == fst]
        best = sub.loc[sub["steady_state_D"].idxmax()]
        print(f"  FST {fst}: optimal interval = {int(best['interval_days'])}d "
              f"(ss_D {best['steady_state_D']:.1f}, "
              f"net_eff {best['net_final_D_change_per_SED']:.3f})")

    sub = df[df["FST"] == 5]
    if sub.shape[0] > 0:
        best_row = sub.loc[sub["steady_state_D"].idxmax()]
        best_sched = build_pulse_schedule(PROTOCOL_DAYS, sed_per_session, int(best_row["interval_days"]))
        r_best = simulate(fst=5, schedule=best_sched, duration_days=PROTOCOL_DAYS)
        print_safety_summary(r_best.get("safety_log", []),
                             f" [Best: FST 5, every {best_row['interval_days']}d, {sed_per_session} SED]")
    return df


# ===========================================================================
# Analysis G: continuous optimization via scipy.optimize
# ===========================================================================
def analysis_scipy_optimize():
    """
    Exhaustive integer sweep of inter-session interval over 1-60 days.

    This used to call scipy.optimize.minimize_scalar(method='bounded') (a
    Brent-type continuous line search) on `interval_float`, but the objective
    rounds that argument to the nearest integer before building a schedule
    (`interval = max(1, int(round(interval_float)))`) — i.e. the function
    Brent's method was searching is piecewise-constant on every unit
    interval. A parabolic-interpolation / golden-section method assumes a
    smooth univariate landscape; run on a staircase it converges to
    whichever step it lands in, not to the argmax, and the result is not
    reproducible across scipy versions or starting brackets. It also is not
    the right tool for this problem on principle: the interval variable is
    integer-valued (a schedule cannot have a session every 6.3 days), so a
    continuous optimizer was never the correct method here, independent of
    the rounding bug. `interval_float` was never fed to the simulator
    directly, so there was no continuous objective to optimize.

    Fixed (2026-08-21, methodology review R4 / OPTIMIZATION_METHODOLOGY.md
    section 9-10): replaced with the exhaustive integer sweep over the same
    (1, 60) bounds the continuous solver was given. This is honest instead of
    approximating a discontinuous function with a continuous method, cheaper
    (60 forward evaluations vs. an iterative line search) and exact rather
    than approximate on an integer domain, and it removes the discrepancy
    against section 6 (whose 1-30 day fine-grid sweep is the same kind of
    exhaustive search over a narrower range and additionally the *correct*
    method for this domain — dropping this section entirely was the other
    option, but keeping it as a direct, wider-range replication of section 6
    is more informative than deleting it). See
    `docs/OPTIMIZATION_METHODOLOGY.md` sections 8 ("7 scipy continuous") and
    10 (R4) for the full argument; option (a) of the three offered there
    ("drop the continuous optimiser and report the exhaustive integer
    sweep") is the one implemented.

    Optimizes two objectives:
    1. net_final_D_change_per_SED — per-SED framing with interior optimum
       (the retired per-SED efficiency metric was removed 2026-07-31 because
       it ranks schedules anti-correlated with mean 25(OH)D)
    2. steady_state_D (absolute) — maximum-D framing

    Replicates the fine-grid argmax (section 6) over a wider integer range
    (1-60d vs 1-30d) rather than "confirming" it via a different method —
    it is the same method (exhaustive integer sweep) at a different
    resolution, so agreement with section 6 where the ranges overlap is
    expected, not an independent cross-check.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS G: exhaustive integer interval sweep (1-60 days)")
    print("=" * 70)

    fsts = [5, 6]
    sed_per_session = 1.0
    PROTOCOL = 84
    intervals = list(range(1, 61))

    def net_change_per_sed(interval, fst):
        sched = build_pulse_schedule(PROTOCOL, sed_per_session, interval)
        r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL)
        return net_final_D_change_per_SED(r)

    def ss_D(interval, fst):
        sched = build_pulse_schedule(PROTOCOL, sed_per_session, interval)
        r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL)
        return steady_state_D(r)

    rows = []
    for fst in fsts:
        net_vals = [net_change_per_sed(iv, fst) for iv in intervals]
        opt_interval_net = intervals[int(np.argmax(net_vals))]
        opt_net_change = max(net_vals)

        D_vals = [ss_D(iv, fst) for iv in intervals]
        opt_interval_D = intervals[int(np.argmax(D_vals))]
        opt_D = max(D_vals)

        sched_net = build_pulse_schedule(PROTOCOL, sed_per_session, opt_interval_net)
        r_net = simulate(fst=fst, schedule=sched_net, duration_days=PROTOCOL)

        sched_D = build_pulse_schedule(PROTOCOL, sed_per_session, opt_interval_D)
        r_D = simulate(fst=fst, schedule=sched_D, duration_days=PROTOCOL)

        row = {
            "FST": fst,
            "opt_interval_net_change": opt_interval_net,
            "opt_net_change_nmol_per_SED": opt_net_change,
            "net_change_func_evals": len(intervals),
            "opt_interval_absolute_D": opt_interval_D,
            "opt_steady_state_D": opt_D,
            "D_func_evals": len(intervals),
            "ss_D_at_net_change_opt": steady_state_D(r_net),
            "net_change_at_D_opt": net_final_D_change_per_SED(r_D),
        }
        rows.append(row)

        print(f"\nFST {fst}:")
        print(f"  Net-change optimum: interval={opt_interval_net}d, "
              f"net_change={opt_net_change:.3f} nmol/L/SED "
              f"({len(intervals)} evals, exhaustive over 1-60d)")
        print(f"  Absolute D optimum: interval={opt_interval_D}d, "
              f"steady_state_D={opt_D:.1f} nmol/L "
              f"({len(intervals)} evals, exhaustive over 1-60d)")
        print(f"  Cross-check: ss_D at net-change optimum = {steady_state_D(r_net):.1f}, "
              f"net_change at D optimum = {net_final_D_change_per_SED(r_D):.3f}")
        print(f"  AUC deficit at 50 nmol/L for optimal schedule: {auc_deficit(r_net['D'], r_net['t'], 50.0):.1f}")
        print(f"  AUC deficit at 30 nmol/L for optimal schedule: {auc_deficit(r_net['D'], r_net['t'], 30.0):.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/scipy_optimize_results.csv", index=False)
    return df


# ===========================================================================
# Analysis H: maximum interval for sufficiency
# ===========================================================================
def analysis_max_interval_for_sufficiency():
    """
    Find the longest inter-session interval maintaining steady-state D >= 50
    nmol/L under fixed 1 SED/session dosing.

    Scans intervals 1-60 days for each FST (V, VI). Reports threshold
    interval and plots D_ss vs interval with a sufficiency line.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS H: maximum interval for sufficiency")
    print("=" * 70)
    SED = 1.0
    THRESHOLD = 50.0
    rows = []
    for fst in [5, 6]:
        for interval in range(1, 61):
            sched = build_pulse_schedule(PROTOCOL_DAYS, SED, interval)
            r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL_DAYS)
            d_ss = steady_state_D(r)
            rows.append({
                "FST": fst, "interval_days": interval,
                "steady_state_D": d_ss,
                "above_sufficiency": d_ss >= THRESHOLD,
                "cumulative_SED": r["cumulative_SED"],
                "auc_deficit_30": auc_deficit(r["D"], r["t"], 30.0),
                "auc_deficit_50": auc_deficit(r["D"], r["t"], 50.0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/max_interval_sufficiency.csv", index=False)

    for fst in [5, 6]:
        sub = df[df["FST"] == fst]
        passing = sub[sub["above_sufficiency"]]
        if len(passing) > 0:
            max_int = int(passing["interval_days"].max())
            d_at_max = sub[sub["interval_days"] == max_int]["steady_state_D"].values[0]
            print(f"  FST {fst}: max interval = {max_int}d "
                  f"(D_ss = {d_at_max:.1f} nmol/L)")
        else:
            print(f"  FST {fst}: no interval 1-60d maintains sufficiency")

    # Plot D_ss vs interval
    fig, ax = plt.subplots(figsize=(9, 5))
    for fst in [5, 6]:
        sub = df[df["FST"] == fst]
        ax.plot(sub["interval_days"], sub["steady_state_D"],
                marker=".", markersize=3, label=f"FST {fst}")
    ax.axhline(THRESHOLD, color="gray", linestyle="--", alpha=0.7,
               label=f"{THRESHOLD} nmol/L sufficiency")
    ax.set_xlabel("Inter-session interval (days)")
    ax.set_ylabel("Steady-state 25(OH)D (nmol/L)")
    ax.set_title("Maximum interval for sufficiency (1 SED/session)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/max_interval_sufficiency.png", dpi=130)
    plt.close(fig)
    print(f"  Saved {FIG_DIR}/max_interval_sufficiency.png")
    return df


# ===========================================================================
# Analysis I: loading dose optimization
# ===========================================================================
def analysis_loading_dose():
    """
    Optimize how to distribute a UV budget across the first 14 days
    (loading phase), with fixed maintenance (1 SED every 7 days, day 15-84).

    Tests budgets of 2-20 SED split across 1, 2, 3, 5, 7, or 14 sessions.
    Compares D at day 14, D at day 84, and AUC above 50 nmol/L.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS I: loading dose optimization")
    print("=" * 70)
    LOADING_DAYS = 14
    MAINT_START = 14
    MAINT_INTERVAL = 7
    MAINT_SED = 1.0
    BUDGETS = [2, 5, 10, 15, 20]
    DISTRIBUTIONS = [
        (1,  [0]),
        (2,  [0, 6]),
        (3,  [0, 4, 8]),
        (5,  [0, 1, 2, 3, 4]),
        (7,  list(range(7))),
        (14, list(range(14))),
    ]
    rows = []
    for fst in [5, 6]:
        for budget in BUDGETS:
            for n_sessions, load_days in DISTRIBUTIONS:
                sed_per = budget / n_sessions
                # Build combined schedule: loading pulses + maintenance
                sched = []
                for day in load_days:
                    sched.append((day, sed_per))
                for day in range(MAINT_START, PROTOCOL_DAYS, MAINT_INTERVAL):
                    sched.append((day, MAINT_SED))
                sched.sort(key=lambda x: x[0])
                r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL_DAYS)
                # D at day 14: index 28 at 0.5-day steps
                d14_idx = min(28, len(r["D"]) - 1)
                d14 = r["D"][d14_idx]
                auc_50 = auc_above_sufficiency_per_SED(r) * r["cumulative_SED"]
                rows.append({
                    "FST": fst, "budget_SED": budget,
                    "n_loading_sessions": n_sessions,
                    "sed_per_session": sed_per,
                    "D_at_day14": d14,
                    "final_D": final_D(r),
                    "steady_state_D": steady_state_D(r),
                    "AUC_above_50": auc_50,
                    "cumulative_SED": r["cumulative_SED"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/loading_dose_optimization.csv", index=False)

    # Summary: best distribution per budget
    print("\n  Best loading distribution by budget:")
    for budget in BUDGETS:
        for fst in [5, 6]:
            sub = df[(df["FST"] == fst) & (df["budget_SED"] == budget)]
            best = sub.loc[sub["D_at_day14"].idxmax()]
            print(f"    FST {fst}, {budget} SED: "
                  f"{int(best['n_loading_sessions'])} sessions × "
                  f"{best['sed_per_session']:.1f} SED → "
                  f"D(day14)={best['D_at_day14']:.1f}, "
                  f"D(day84)={best['final_D']:.1f}")

    # Plot: D at day 14 by distribution, grouped by budget
    fig, axes = plt.subplots(1, len(BUDGETS), figsize=(3 * len(BUDGETS), 5),
                             sharey=True)
    for i, budget in enumerate(BUDGETS):
        ax = axes[i]
        sub = df[df["budget_SED"] == budget]
        labels = [f"{int(n)}×" for n in sub["n_loading_sessions"].unique()]
        x = np.arange(len(labels))
        width = 0.35
        for j, fst in enumerate([5, 6]):
            fs = sub[sub["FST"] == fst].sort_values("n_loading_sessions")
            vals = fs["D_at_day14"].values
            ax.bar(x + j * width, vals, width, label=f"FST {fst}")
        ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title(f"{budget} SED")
        if i == 0:
            ax.set_ylabel("D at day 14 (nmol/L)")
        ax.grid(alpha=0.3, axis="y")
    axes[-1].legend()
    fig.suptitle("Loading dose: D at day 14 by session count", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/loading_dose_D_at_day14.png", dpi=130)
    plt.close(fig)
    print(f"\n  Saved {FIG_DIR}/loading_dose_D_at_day14.png")

    max_budget = max(BUDGETS)
    sched_max = [(0, max_budget)] + build_pulse_schedule(PROTOCOL_DAYS, 1.0, 7)
    r_safe = simulate(fst=5, schedule=sched_max, duration_days=PROTOCOL_DAYS)
    print_safety_summary(r_safe.get("safety_log", []),
        f" [FST 5, {max_budget} SED loading in 1 session]")

    r_safe = simulate(fst=5,
        schedule=build_pulse_schedule(PROTOCOL_DAYS, 2.0, 1),
        duration_days=PROTOCOL_DAYS)
    print_safety_summary(r_safe.get("safety_log", []),
        f" [FST 5, 2 SED daily — aggressive loading]")
    return df


# ===========================================================================
# Analysis J: maintenance dose splitting
# ===========================================================================
def analysis_maintenance_split():
    """
    Compare single vs split maintenance doses at the same total UV rate.

    Strategy A: 1 SED every T days (single session)
    Strategy B: 0.5 SED every T/2 days (split sessions)

    Tests T = 7, 14, 21 days for FST V and VI.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS J: maintenance dose splitting")
    print("=" * 70)
    INTERVALS = [7, 14, 21]
    rows = []
    for fst in [5, 6]:
        for T in INTERVALS:
            # Strategy A: 1 SED every T days
            sched_a = build_pulse_schedule(PROTOCOL_DAYS, 1.0, T)
            r_a = simulate(fst=fst, schedule=sched_a, duration_days=PROTOCOL_DAYS)
            # Strategy B: split dose, equal total SED rate
            split_interval = max(1, T // 2)
            sed_split = split_interval / T
            sched_b = build_pulse_schedule(PROTOCOL_DAYS, sed_split, split_interval)
            r_b = simulate(fst=fst, schedule=sched_b, duration_days=PROTOCOL_DAYS)
            for label, sed, interval, r in [
                ("single_1SED", 1.0, T, r_a),
                ("split_0.5SED", sed_split, split_interval, r_b),
            ]:
                rows.append({
                    "FST": fst, "interval_T": T,
                    "strategy": label,
                    "sed_per_session": sed,
                    "actual_interval": interval,
                    "steady_state_D": steady_state_D(r),
                    "AUC_above_50": auc_above_sufficiency_per_SED(r) * r["cumulative_SED"],
                    "cumulative_SED": r["cumulative_SED"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/maintenance_split_comparison.csv", index=False)

    # Print comparison
    for fst in [5, 6]:
        print(f"\n  FST {fst}:")
        for T in INTERVALS:
            sub = df[(df["FST"] == fst) & (df["interval_T"] == T)]
            s = sub[sub["strategy"] == "single_1SED"].iloc[0]
            sp = sub[sub["strategy"] == "split_0.5SED"].iloc[0]
            winner = "single" if s["steady_state_D"] > sp["steady_state_D"] else "split"
            print(f"    T={T:2d}d: single D_ss={s['steady_state_D']:.1f} | "
                  f"split D_ss={sp['steady_state_D']:.1f} → {winner}")

    # Plot: side-by-side bars
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(INTERVALS))
    width = 0.35
    for j, fst in enumerate([5, 6]):
        for k, strat in enumerate(["single_1SED", "split_0.5SED"]):
            vals = []
            for T in INTERVALS:
                sub = df[(df["FST"] == fst) & (df["interval_T"] == T) &
                         (df["strategy"] == strat)]
                vals.append(sub["steady_state_D"].values[0])
            offset = (j * 2 + k) * width / 2
            ax.bar(x + offset, vals, width / 2,
                   label=f"FST {fst} {'single' if strat == 'single_1SED' else 'split'}")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"Every {T}d" for T in INTERVALS])
    ax.set_ylabel("Steady-state 25(OH)D (nmol/L)")
    ax.set_title("Maintenance: single vs split doses")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/maintenance_split.png", dpi=130)
    plt.close(fig)
    print(f"\n  Saved {FIG_DIR}/maintenance_split.png")

    sched_a = build_pulse_schedule(PROTOCOL_DAYS, 1.0, 7)
    r_a_safe = simulate(fst=5, schedule=sched_a, duration_days=PROTOCOL_DAYS)
    print_safety_summary(r_a_safe.get("safety_log", []),
        f" [FST 5, single 1 SED every 7d]")
    return df


# ===========================================================================
# Analysis K: sun exposure scheduling (no equipment)
# ===========================================================================
def analysis_sun_exposure():
    """
    Practical sun exposure scheduling: how many minutes of midday sun,
    how often, to maintain sufficiency?

    Uses the FROZEN REFERENCE SCENARIO (`vitd_scenario.py`): fixed ambient UV
    index, fixed horizontal-to-personal geometry factor, fixed exposed BSA,
    held constant on every exposure day.

    [BUG FIX 2026-08-21 -- unit/geometry error, not a calibration change.]
    This function previously used a hardcoded `MIN_PER_SED = 6.7` alongside a
    docstring claiming "UV index ~5" and "erythemal irradiance ~0.25 W/m^2".
    Those three statements are mutually inconsistent and two of them are wrong:

      * 1 UV Index unit = 25 mW/m^2 erythemal (WHO/WMO). UV index 5 is
        therefore 125 mW/m^2, not the 250 mW/m^2 (= 0.25 W/m^2) claimed --
        the stated irradiance is UV index 10, a factor of 2 off its own
        stated UV index.
      * 6.7 min/SED = 0.149 SED/min, which is UV index ~10 on a HORIZONTAL
        surface. No horizontal-to-personal geometry factor was applied at
        all, while every other minutes figure in this repository applies
        `vitd_objectives.PERSONAL_DOSE_GEOMETRY_FACTOR`. An upright person
        does not intercept horizontal-surface irradiance.

    Net effect: this section was under-reporting the minutes needed by roughly
    4.7x relative to the rest of the repository. Section 11's minute figures
    are correspondingly LONGER after this fix. No calibrated scientific
    parameter moved; `calibration_approval.yaml` was not touched. See
    `CLAUDE.md` ("Bug fixes ... unit errors ... are allowed without approval").

    Two parts:
    1. Practical schedule comparison (6 predefined strategies)
    2. Daily minimum sweep (5-60 min/day in 5-min steps)
    """
    print("\n" + "=" * 70)
    print("ANALYSIS K: sun exposure scheduling (no equipment)")
    print("=" * 70)
    print(vitd_scenario.describe())
    MIN_PER_SED = 1.0 / vitd_scenario.SED_PER_MINUTE
    # [CALIBRATION 2026-08-16] Was a hardcoded 1.0. Sun now uses the single
    # unified solar value; see vitd_model.E_SOURCE_SOLAR_DEFAULT.
    E_SOURCE_SUN = get_E_source(E_SOURCE_SOLAR_DEFAULT)
    # Frozen scenario BSA (== vitd_model.BSA_REF, so bsa_factor is exactly
    # 1.0 and every existing calibration is untouched). Was a separate
    # hardcoded 0.30, inconsistent with the 0.35 used everywhere else.
    BSA_SUN = vitd_scenario.EXPOSED_BSA

    # Pre-defined practical schedules: (name, sessions_per_week, minutes_per_session)
    PRACTICAL = [
        ("daily_10min",       7, 10),
        ("3xwk_20min",        3, 20),
        ("2xwk_30min",        2, 30),
        ("weekly_45min",      1, 45),
        ("minimal_15min_3x",  3, 15),
        ("ultra_10min_2x",    2, 10),
    ]

    rows = []
    for fst in [5, 6]:
        # Part 1: practical schedules
        for name, sessions_per_week, minutes in PRACTICAL:
            sed = minutes / MIN_PER_SED
            interval = max(1, int(round(7 / sessions_per_week)))
            actual_sessions_per_week = 7 / interval
            sched = build_pulse_schedule(PROTOCOL_DAYS, sed, interval)
            r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL_DAYS,
                         E_source_override=E_SOURCE_SUN,
                         exposed_bsa_override=BSA_SUN)
            d_ss = steady_state_D(r)
            rows.append({
                "FST": fst, "schedule": name, "category": "practical",
                "sessions_per_week": sessions_per_week,
                "actual_sessions_per_week": actual_sessions_per_week,
                "sed_per_session": sed,
                "minutes_per_session": minutes,
                "minutes_per_week": minutes * sessions_per_week,
                "weekly_SED": sed * sessions_per_week,
                "steady_state_D": d_ss,
                "above_sufficiency": d_ss >= 50.0,
                "cumulative_SED": r["cumulative_SED"],
            })

        # Part 2: daily minimum sweep
        for minutes in range(5, 65, 5):
            sed = minutes / MIN_PER_SED
            sched = build_pulse_schedule(PROTOCOL_DAYS, sed, 1)  # daily
            r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL_DAYS,
                         E_source_override=E_SOURCE_SUN,
                         exposed_bsa_override=BSA_SUN)
            d_ss = steady_state_D(r)
            rows.append({
                "FST": fst, "schedule": f"daily_{minutes}min", "category": "sweep",
                "sessions_per_week": 7,
                "sed_per_session": sed,
                "minutes_per_session": minutes,
                "minutes_per_week": minutes * 7,
                "weekly_SED": sed * 7,
                "steady_state_D": d_ss,
                "above_sufficiency": d_ss >= 50.0,
                "cumulative_SED": r["cumulative_SED"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/sun_exposure_schedules.csv", index=False)

    # Print practical schedule summary
    print("\n  Practical sun exposure schedules (spring/fall, arms+face):")
    practical = df[df["category"] == "practical"]
    for _, row in practical.iterrows():
        status = "OK" if row["above_sufficiency"] else "BELOW"
        print(f"    FST {row['FST']}: {row['schedule']:20s} "
              f"{row['sessions_per_week']}/wk (actual {row['actual_sessions_per_week']:.2f}/wk) "
              f"{row['minutes_per_week']:4.0f} min/wk → D_ss={row['steady_state_D']:.1f} [{status}]")

    # Find minimum daily minutes for sufficiency
    print("\n  Minimum daily exposure for sufficiency:")
    for fst in [5, 6]:
        sub = df[(df["FST"] == fst) & (df["category"] == "sweep")]
        passing = sub[sub["above_sufficiency"]]
        if len(passing) > 0:
            min_min = passing["minutes_per_session"].min()
            d_at_min = passing[passing["minutes_per_session"] == min_min]["steady_state_D"].values[0]
            print(f"    FST {fst}: {min_min:.0f} min/day "
                  f"({min_min * 7:.0f} min/week, {min_min / 60:.1f} hr/day) → "
                  f"D_ss={d_at_min:.1f} nmol/L")
        else:
            print(f"    FST {fst}: no duration 5-60 min/day maintains sufficiency")

    # Plot: D_ss vs daily minutes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col, fst in enumerate([5, 6]):
        ax = axes[col]
        sub = df[(df["FST"] == fst) & (df["category"] == "sweep")]
        ax.plot(sub["minutes_per_session"], sub["steady_state_D"],
                marker="o", markersize=4, label="Daily exposure")
        ax.axhline(50, color="gray", linestyle="--", alpha=0.7,
                   label="50 nmol/L sufficiency")
        # Mark practical schedules
        prac = df[(df["FST"] == fst) & (df["category"] == "practical")]
        for _, row in prac.iterrows():
            ax.plot(row["minutes_per_session"], row["steady_state_D"],
                    marker="x", markersize=10, color="red")
            ax.annotate(row["schedule"].replace("_", " "),
                        (row["minutes_per_session"], row["steady_state_D"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
        ax.set_xlabel("Minutes per session")
        ax.set_ylabel("Steady-state 25(OH)D (nmol/L)")
        ax.set_title(f"FST {fst}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Sun exposure: minutes per session vs D_ss (spring/fall, arms+face)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/sun_exposure_minutes_vs_D.png", dpi=130)
    plt.close(fig)
    print(f"\n  Saved {FIG_DIR}/sun_exposure_minutes_vs_D.png")

    # Plot: weekly minutes comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    prac = df[df["category"] == "practical"]
    x = np.arange(len(PRACTICAL))
    width = 0.35
    for j, fst in enumerate([5, 6]):
        fs = prac[prac["FST"] == fst]
        ax.bar(x + j * width, fs["minutes_per_week"].values, width,
               label=f"FST {fst}", alpha=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([p[0].replace("_", "\n") for p in PRACTICAL], fontsize=8)
    ax.set_ylabel("Minutes per week")
    ax.set_title("Weekly time commitment by schedule")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/sun_exposure_weekly_minutes.png", dpi=130)
    plt.close(fig)
    print(f"  Saved {FIG_DIR}/sun_exposure_weekly_minutes.png")

    r_safe = simulate(fst=5,
        schedule=build_pulse_schedule(PROTOCOL_DAYS, 45/MIN_PER_SED, 7),
        duration_days=PROTOCOL_DAYS,
        E_source_override=E_SOURCE_SUN,
        exposed_bsa_override=BSA_SUN)
    print_safety_summary(r_safe.get("safety_log", []),
        f" [FST 5, weekly 45 min sun]")

    print("\n  NOTE: Assumes spring/fall midday sun (UV index ~5). "
          "Summer sun ~2x more efficient; winter ~0.5x.")
    print("  Arms + face exposure (~30% BSA). More skin = less time needed.")
    return df


# ===========================================================================
# ===========================================================================
# Analysis D: trajectory plots for the best schedule under each framing
# ===========================================================================
def analysis_trajectories():
    """
    Show D(t), M(t), SC(t) trajectories for selected schedules at FST V and VI.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS D: example trajectories")
    print("=" * 70)
    intervals_to_plot = [2, 7, 14]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for col, fst in enumerate([5, 6]):
        ax_D = axes[0, col]
        ax_M = axes[1, col]
        ax_SC = axes[2, col]
        for iv in intervals_to_plot:
            sched = build_pulse_schedule(PROTOCOL_DAYS, 1.0, iv)
            r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL_DAYS)
            ax_D.plot(r["t"], r["D"], label=f"every {iv} days")
            ax_M.plot(r["t"], r["M"], label=f"every {iv} days")
            ax_SC.plot(r["t"], r["SC"], label=f"every {iv} days")
        ax_D.axhline(50, color="gray", linestyle=":", alpha=0.5,
                     label="50 nmol/L sufficiency")
        ax_D.set_ylabel("25(OH)D (nmol/L)")
        ax_D.set_title(f"FST {['I','II','III','IV','V','VI'][fst-1]}")
        ax_D.legend(loc="best", fontsize=9)
        ax_D.grid(alpha=0.3)
        
        ax_M.set_ylabel("effective melanin index M")
        ax_M.legend(loc="best", fontsize=9)
        ax_M.grid(alpha=0.3)
        
        ax_SC.set_ylabel("stratum corneum SC")
        ax_SC.set_xlabel("day")
        ax_SC.legend(loc="best", fontsize=9)
        ax_SC.grid(alpha=0.3)
    fig.suptitle("Trajectories at fixed 1 SED/session, three schedule densities",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/trajectories.png", dpi=130)
    plt.close(fig)
    print(f"  Saved {FIG_DIR}/trajectories.png")

    r_safe = simulate(fst=5,
        schedule=build_pulse_schedule(PROTOCOL_DAYS, 1.0, 7),
        duration_days=PROTOCOL_DAYS)
    print_safety_summary(r_safe.get("safety_log", []),
        f" [FST 5, 1 SED every 7d — trajectory reference]")


# ===========================================================================
# Master runner
# ===========================================================================
def main():
    print("Vitamin D scheduling model: full analysis run")
    print(f"Output: {FIG_DIR}/ and {TAB_DIR}/")
    analysis_alpha_sweep()
    analysis_K_ATTEN_sweep()
    analysis_two_framings()
    analysis_fine_grid()
    analysis_scipy_optimize()
    analysis_max_interval_for_sufficiency()
    analysis_loading_dose()
    analysis_maintenance_split()
    analysis_sun_exposure()
    analysis_trajectories()
    print("\nDONE.")


if __name__ == "__main__":
    main()
