"""
plot_sc_deboer2025.py
======================
Visualizes the stratum-corneum thinning-then-overshoot finding from
de Boer 2025 Supp Table S1 against the model's SC(t) trajectory.

Companion to fit_sc_deboer2025.py, which establishes (numerically) that no
parameter choice fits both the dip and the overshoot -- best achievable fit
misses by 71-75% of signal amplitude. This script makes that mismatch visible.

Saves figs/sc_deboer2025_fit.png.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams, simulate, build_weekly_schedule
from scripts.fit_sc_deboer2025 import (
    DEBOER_SC_UM, VISIT_DAYS, PROTOCOL_SED, PROTOCOL_DAYS,
    PROTOCOL_DURATION, STRATUM_FST, best_achievable_fit,
)

FIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")

COLORS = {"all": "#4C72B0", "I-III": "#DD8452", "IV-VI": "#55A868"}


def sc_trajectory_um(fit: dict, t0_um: float):
    """Run simulate() with the best-achievable fit params, return (t, SC_um)."""
    p = ModelParams()
    p.k_damage = fit["k_damage"]
    p.alpha_SC = fit["alpha_SC"]
    p.k_repair_SC = fit["k_repair_SC"]
    sched = build_weekly_schedule(PROTOCOL_DURATION, PROTOCOL_SED, PROTOCOL_DAYS)
    r = simulate(fst=fit["fst"], schedule=sched, duration_days=PROTOCOL_DURATION,
                 params=p, D0=25.0, SC_max_override=fit["SC_max"])
    t, sc_frac = r["t"], r["SC"]
    # SC state is a fractional change from baseline; convert back to um using T0.
    sc_um = t0_um * (1.0 + np.array(sc_frac))
    return np.array(t), sc_um


def envelope(t: np.ndarray, y: np.ndarray, window_days: float = 6.0):
    """Rolling midpoint of the sawtooth (dose-and-decay cycle), for a clean trend line."""
    t_grid = np.linspace(t.min(), t.max(), 300)
    mid = np.empty_like(t_grid)
    for i, tc in enumerate(t_grid):
        mask = np.abs(t - tc) <= window_days / 2
        if not mask.any():
            mask = np.argsort(np.abs(t - tc))[:5]
        mid[i] = 0.5 * (y[mask].max() + y[mask].min())
    return t_grid, mid


def main():
    fits = {s: best_achievable_fit(s, verbose=False) for s in DEBOER_SC_UM}
    stratum = "all"
    t0_um, t1_um, t2_um = DEBOER_SC_UM[stratum]
    fit = fits[stratum]

    t_model, sc_model_um = sc_trajectory_um(fit, t0_um)
    t_env, sc_env = envelope(t_model, sc_model_um)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Raw sawtooth, faint.
    ax.plot(t_model, sc_model_um, color=COLORS[stratum], lw=0.6, alpha=0.35,
            label="model, raw (each dose bumps SC, then decays)")
    # Trend line -- the thing to actually compare against the data.
    ax.plot(t_env, sc_env, color=COLORS[stratum], lw=2.5,
            label="model trend (best achievable fit)")

    ax.scatter(VISIT_DAYS, [t0_um, t1_um, t2_um], color="black", zorder=5, s=90,
               label="de Boer 2025 observed medians (n=31)")
    ax.plot(VISIT_DAYS, [t0_um, t1_um, t2_um], color="black", lw=1.3,
            linestyle="--", alpha=0.6)

    # Highlight the miss at T1 and T2: vertical connector + gap label.
    for day, obs_um in zip(VISIT_DAYS[1:], [t1_um, t2_um]):
        model_um = float(np.interp(day, t_env, sc_env))
        ax.plot([day, day], [obs_um, model_um], color="crimson", lw=1.5,
                linestyle=":", zorder=4)
        gap_pct = (model_um - obs_um) / t0_um * 100
        y_mid = (obs_um + model_um) / 2
        ax.annotate(f"{gap_pct:+.0f}% of baseline\nmissed", xy=(day, y_mid),
                    xytext=(day + 4, y_mid), fontsize=9, color="crimson",
                    va="center")

    ax.axhline(t0_um, color="gray", lw=0.8, linestyle=":", alpha=0.7)
    ax.text(1, t0_um, " baseline", fontsize=8, color="gray", va="bottom")

    ax.set_xlabel("Day  (protocol: 0.8 SED, 3x/week, 9 weeks)")
    ax.set_ylabel("Stratum corneum thickness (um)")
    ax.set_title(
        "Skin barrier (stratum corneum) thins first, then overshoots --\n"
        "the model's best possible fit still misses both moves",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.01,
             f"Relative RMSE = {fit['relative_rmse']:.0%} of signal amplitude "
             f"-> structural failure, not a tuning problem. k_damage stays 0.0 by default.",
             ha="center", fontsize=9, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    os.makedirs(FIGS_DIR, exist_ok=True)
    out_path = os.path.join(FIGS_DIR, "sc_deboer2025_fit.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
