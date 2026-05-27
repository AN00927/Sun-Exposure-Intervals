"""
vitd_analyses.py
================
Priority analyses for the vitamin D scheduling model, per HANDOFF_md.docx
"What To Do Next" section:

  A. alpha sensitivity sweep (photoadaptation rise rate, FST V & VI)
     - This is the #1 priority per the handoff.
  B. K_ATTEN sensitivity sweep (synthesis attenuation coefficient)
     - #2 priority. Young 2020 implies K_ATTEN may be much lower than 2.0;
       sweep tests how much this matters for scheduling.
  C. Schedule comparison under TWO framings:
     - Framing 1: Match cumulative SED, vary interval (current handoff approach)
     - Framing 2: Match per-session dose, vary interval (clinically meaningful;
                  treats UV as the constrained resource)
  D. Two-framing summary plot - shows that the optimal interval flips
     between framings (the central handoff finding).

All outputs saved to ./figs/ as PNG and ./tables/ as CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vitd_model import (ModelParams, simulate, build_pulse_schedule,
                        steady_state_D, efficiency_D_per_SED, mean_final_M,
                        final_D)

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
    inter-session interval shifts. Per the handoff this is the highest
    priority because alpha was calibrated from light-skin (Miyamura 2011)
    and may be larger in darker skin.

    For each alpha value and each FST (V, VI), we run all the candidate
    intervals at FIXED per-session dose (1.0 SED) and find which interval
    maximizes D per SED (the efficiency framing).
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
                    "efficiency_D_per_SED": efficiency_D_per_SED(r),
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/alpha_sweep.csv", index=False)

    # Plot: heatmap of efficiency by interval and alpha
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, fst in zip(axes, fsts):
        sub = df[df["FST"] == fst]
        pivot = sub.pivot(index="alpha", columns="interval_days",
                          values="efficiency_D_per_SED")
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{a:.3f}" for a in pivot.index])
        ax.set_xlabel("inter-session interval (days)")
        ax.set_ylabel(r"$\alpha$ (photoadaptation rise per SED)")
        ax.set_title(f"FST {['I','II','III','IV','V','VI'][fst-1]}: "
                     "efficiency (nmol/L per SED)")
        # mark the argmax per alpha row
        for ri, alpha_val in enumerate(pivot.index):
            argmax_col = np.argmax(pivot.iloc[ri].values)
            ax.plot(argmax_col, ri, "r*", markersize=15,
                    markeredgecolor="white")
        plt.colorbar(im, ax=ax)
    fig.suptitle(r"$\alpha$ sensitivity: efficiency framing (fixed 1 SED/session)"
                 "\nRed star = interval that maximizes efficiency at that "
                 r"$\alpha$",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/alpha_sweep_efficiency.png", dpi=130)
    plt.close(fig)

    # Same heatmap but for ABSOLUTE steady-state D
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
    print("\nOptimal interval by alpha (efficiency framing, fixed 1 SED/session):")
    for fst in fsts:
        sub = df[df["FST"] == fst]
        print(f"\nFST {fst}:")
        for alpha in alphas:
            sub2 = sub[sub["alpha"] == alpha]
            best = sub2.loc[sub2["efficiency_D_per_SED"].idxmax()]
            print(f"  alpha={alpha:.3f}:  optimal interval = "
                  f"{int(best['interval_days'])}d "
                  f"(efficiency {best['efficiency_D_per_SED']:.2f}, "
                  f"ss D {best['steady_state_D']:.1f})")

    return df


# ===========================================================================
# Analysis B: K_ATTEN sensitivity sweep
# ===========================================================================
def analysis_K_ATTEN_sweep():
    """
    Vary K_ATTEN (melanin penalty coefficient). Young 2020 measured a
    vitamin D inhibition factor of only 1.3-1.4x between FST II and VI,
    suggesting K_ATTEN may be substantially lower than the default 2.0.
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
                    "efficiency_D_per_SED": efficiency_D_per_SED(r),
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/K_ATTEN_sweep.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, fst in enumerate([5, 6]):
        sub = df[df["FST"] == fst]
        # Efficiency
        pe = sub.pivot(index="K_ATTEN", columns="interval_days",
                       values="efficiency_D_per_SED")
        ax = axes[0, col]
        im = ax.imshow(pe.values, aspect="auto", origin="lower",
                       cmap="viridis")
        ax.set_xticks(range(len(pe.columns))); ax.set_xticklabels(pe.columns)
        ax.set_yticks(range(len(pe.index)))
        ax.set_yticklabels([f"{k:.2f}" for k in pe.index])
        ax.set_xlabel("interval (days)"); ax.set_ylabel("K_ATTEN")
        ax.set_title(f"FST {fst}: efficiency D/SED")
        for ri in range(len(pe.index)):
            ax.plot(np.argmax(pe.iloc[ri].values), ri,
                    "r*", markersize=14, markeredgecolor="white")
        plt.colorbar(im, ax=ax)
        # Steady-state D
        pd_ = sub.pivot(index="K_ATTEN", columns="interval_days",
                        values="steady_state_D")
        ax = axes[1, col]
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
    fig.suptitle("K_ATTEN sensitivity (fixed 1 SED/session)"
                 "\nTop: efficiency.  Bottom: absolute steady-state D",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/K_ATTEN_sweep.png", dpi=130)
    plt.close(fig)

    print("\nOptimal interval by K_ATTEN (efficiency framing):")
    for fst in fsts:
        sub = df[df["FST"] == fst]
        print(f"\nFST {fst}:")
        for K in K_vals:
            sub2 = sub[sub["K_ATTEN"] == K]
            best_eff = sub2.loc[sub2["efficiency_D_per_SED"].idxmax()]
            best_D = sub2.loc[sub2["steady_state_D"].idxmax()]
            print(f"  K={K:.1f}:  "
                  f"opt-eff interval={int(best_eff['interval_days'])}d "
                  f"(eff {best_eff['efficiency_D_per_SED']:.2f}); "
                  f"opt-D interval={int(best_D['interval_days'])}d "
                  f"(D {best_D['steady_state_D']:.1f})")
    return df


# ===========================================================================
# Analysis C: schedule comparison under two framings
# ===========================================================================
def analysis_two_framings():
    """
    Show the two framings side-by-side for FST V and VI:

    Framing 1: MATCHED CUMULATIVE SED (UV budget unconstrained, just
               distributing a fixed total). Total = 42 SED over 84 days.
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
            r1 = simulate(fst=fst, schedule=sched1, duration_days=PROTOCOL_DAYS)
            rows_f1.append({
                "FST": fst,
                "interval_days": interval,
                "sed_per_session": sed_per_f1,
                "n_sessions": n_sessions,
                "cumulative_SED": r1["cumulative_SED"],
                "steady_state_D": steady_state_D(r1),
                "final_D": final_D(r1),
                "final_M": mean_final_M(r1),
                "efficiency": efficiency_D_per_SED(r1),
            })
            # Framing 2: matched per-session dose
            sched2 = build_pulse_schedule(PROTOCOL_DAYS, 1.0, interval)
            r2 = simulate(fst=fst, schedule=sched2, duration_days=PROTOCOL_DAYS)
            rows_f2.append({
                "FST": fst,
                "interval_days": interval,
                "sed_per_session": 1.0,
                "n_sessions": len(r2["synthesis_log"]),
                "cumulative_SED": r2["cumulative_SED"],
                "steady_state_D": steady_state_D(r2),
                "final_D": final_D(r2),
                "final_M": mean_final_M(r2),
                "efficiency": efficiency_D_per_SED(r2),
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
        ax2 = ax.twinx()
        ax2.plot(s1["interval_days"], s1["efficiency"],
                 "s-", color="C3", label="efficiency D/SED")
        ax2.set_ylabel("efficiency (nmol/L per SED)", color="C3")
        ax2.tick_params(axis='y', labelcolor="C3")
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
        ax2 = ax.twinx()
        ax2.plot(s2["interval_days"], s2["efficiency"],
                 "s-", color="C3", label="efficiency D/SED")
        ax2.set_ylabel("efficiency (nmol/L per SED)", color="C3")
        ax2.tick_params(axis='y', labelcolor="C3")
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
    return df1, df2


# ===========================================================================
# Analysis D: trajectory plots for the best schedule under each framing
# ===========================================================================
def analysis_trajectories():
    """
    Show D(t), M(t) trajectories for selected schedules at FST V and VI,
    so the temporal dynamics are visually obvious.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS D: example trajectories")
    print("=" * 70)
    intervals_to_plot = [2, 7, 14]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for col, fst in enumerate([5, 6]):
        ax_D = axes[0, col]
        ax_M = axes[1, col]
        for iv in intervals_to_plot:
            sched = build_pulse_schedule(PROTOCOL_DAYS, 1.0, iv)
            r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL_DAYS)
            ax_D.plot(r["t"], r["D"], label=f"every {iv} days")
            ax_M.plot(r["t"], r["M"], label=f"every {iv} days")
        ax_D.axhline(50, color="gray", linestyle=":", alpha=0.5,
                     label="50 nmol/L sufficiency")
        ax_D.set_ylabel("25(OH)D (nmol/L)")
        ax_D.set_title(f"FST {fst}")
        ax_D.legend(loc="best", fontsize=9)
        ax_D.grid(alpha=0.3)
        ax_M.set_ylabel("effective melanin index M")
        ax_M.set_xlabel("day")
        ax_M.legend(loc="best", fontsize=9)
        ax_M.grid(alpha=0.3)
    fig.suptitle("Trajectories at fixed 1 SED/session, three schedule densities",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/trajectories.png", dpi=130)
    plt.close(fig)
    print(f"  Saved {FIG_DIR}/trajectories.png")


# ===========================================================================
# Master runner
# ===========================================================================
def main():
    print("Vitamin D scheduling model: full analysis run")
    print(f"Output: {FIG_DIR}/ and {TAB_DIR}/")
    analysis_alpha_sweep()
    analysis_K_ATTEN_sweep()
    analysis_two_framings()
    analysis_trajectories()
    print("\nDONE.")


if __name__ == "__main__":
    main()
