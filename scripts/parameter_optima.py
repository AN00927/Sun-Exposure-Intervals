"""
parameter_optima.py
===================
How does the OPTIMAL EXPOSURE INTERVAL move as each limiting parameter moves?

Every parameter swept here is one that *penalises* synthesis or the body:
pigmentation (constitutive and facultative), stratum-corneum barrier change,
homeostatic feedback, and exposure geometry. For each one, the script sweeps
the parameter across its documented prior range, and at every value re-solves
for the best inter-session interval.

Two framings are reported side by side, because they do not agree, and the
disagreement is the point:

  FIXED BUDGET   total cumulative UV is held EXACTLY constant and split across
                 sessions. No schedule can win by dosing less. Objective:
                 maximise mean 25(OH)D over the horizon.

  PER-SED        one fixed dose every `interval` days, scored as net change in
                 25(OH)D per SED delivered. Cumulative UV VARIES between
                 candidates, so sparse schedules are flattered by the
                 denominator. This is the framing that produces the widely
                 quoted "every few days" answer.

Outputs
-------
  figs/parameter_optima_grid.png   one panel per parameter, both framings
  figs/parameter_optima_joint.png  all parameters varied together (Monte Carlo)
  tables/parameter_optima.csv      the underlying numbers

Run:  PYTHONIOENCODING=utf-8 python3 scripts/parameter_optima.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import (ModelParams, simulate, net_final_D_change_per_SED,
                        allow_retired_parameter_sweep)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS_DIR = os.path.join(ROOT, "figs")
TABLES_DIR = os.path.join(ROOT, "tables")

HORIZON = 365
D0 = 25.0
TOTAL_SED_BUDGET = 52.0      # fixed-budget framing: SED per year
SED_PER_SESSION = 1.0        # per-SED framing: dose held per session
INTERVALS = list(range(1, 46))
N_PARAM_VALUES = 9

# ---------------------------------------------------------------------------
# Parameters that limit synthesis or burden the body.
# Ranges are the documented prior ranges where the repo states one
# (ModelParams.SC_PRIOR_RANGES, MODEL_AND_EVIDENCE.md, 'Remaining gaps', gap 29
# for K_ATTEN); otherwise a
# plausible bracket around the default, marked as such in `note`.
# ---------------------------------------------------------------------------
def _p(name, attr, lo, hi, label, note, log=False):
    return dict(name=name, attr=attr, lo=lo, hi=hi, label=label, note=note, log=log)


PARAMETERS = [
    _p("K_ATTEN", "K_ATTEN", 0.0, 1.0,
       "melanin attenuation $K_{ATTEN}$",
       "0.288972-0.838389 implied by Matsuoka 1990 after deconvolving baseline feedback "
        "(MODEL_AND_EVIDENCE.md, 'Remaining gaps', gap 29); 0 included as "
        "ablation"),
    _p("K_const", "K_const", 0.15, 0.50,
       "constitutive pigment $K_{const}$",
       "bracket around Young 2020-derived value; PLAUSIBLE BRACKET, not a "
       "published range"),
    _p("K_SC", "K_SC", 0.5, 2.0,
       "stratum-corneum attenuation $K_{SC}$",
       "ModelParams.SC_PRIOR_RANGES"),
    _p("SC_max", "SC_max", 0.20, 0.40,
       "SC thickening ceiling $SC_{max}$",
       "de Boer 2025: +23% (FP I-III) to +36% (FP IV-VI)"),
    _p("alpha_SC", "alpha_SC", 0.08, 0.20,
       r"SC thickening rate $\alpha_{SC}$",
       "fitted to de Boer 2025, 3 timepoints"),
    _p("k_damage", "k_damage", 0.0, 0.15,
       "UV barrier damage $k_{damage}$",
       "SC_PRIOR_RANGES; default 0.0 because NOT IDENTIFIABLE from group medians"),
    _p("D_max", "D_max", 100.0, 250.0,
       "feedback ceiling $D_{max}$",
       "ORIGIN UNCONFIRMED (MODEL_AND_EVIDENCE.md, 'Known issues', #3); bracket "
       "is illustrative only"),
    _p("alpha", "alpha", 0.001, 0.005,
       r"melanin induction rate $\alpha$",
       "bracket around Miyamura 2011-calibrated default; light-skin derived"),
    _p("beta", "beta", np.log(2) / 120, np.log(2) / 28,
       r"melanin decay $\beta$",
       "Ravnbak 2008 measured 4-5 month fade; bracket spans 28-120 d half-life"),
    _p("M_TAN_HEADROOM", "M_TAN_HEADROOM", 0.05, 0.40,
       "facultative tanning headroom",
       "PRIOR; lower-bounded by Ravnbak 2008's largest measured tan "
       "(dM=0.237724) since 2026-08-25, shipped value 0.238"),
    _p("k_decay_SC", "k_decay_SC", np.log(2) / 28, np.log(2) / 21,
       "SC regression rate $k_{decay,SC}$",
       "epidermal turnover 21-28 d"),
    _p("k_repair_SC", "k_repair_SC", np.log(2) / 14, np.log(2) / 3,
       "barrier repair rate $k_{repair,SC}$",
       "PRIOR, not separately measured by de Boer"),
    _p("exposed_bsa", "exposed_bsa", 0.10, 0.90,
       "exposed body surface area",
       "Kimlin 2014: clothing cover is the largest naturalistic predictor"),
    _p("A", "A", 1.5, 3.5,
       "synthesis coefficient $A$",
       "Young 2021 fair-skin slopes imply 2.0-5.0 nmol/L/SED at 85% BSA"),
]


# ---------------------------------------------------------------------------
# Optimum finders
# ---------------------------------------------------------------------------
def optimum_fixed_budget(fst: int, params: ModelParams) -> tuple[int, float]:
    """Best interval when total cumulative UV is held constant. Maximise mean D."""
    best_iv, best_val = None, -np.inf
    for interval in INTERVALS:
        days = list(range(0, HORIZON, interval))
        sed = TOTAL_SED_BUDGET / len(days)
        sched = [(float(d), sed) for d in days]
        r = simulate(fst=fst, schedule=sched, duration_days=HORIZON,
                     params=params, D0=D0, dose_framing="fixed_cumulative_SED")
        val = float(np.mean(r["D"]))
        if val > best_val:
            best_iv, best_val = interval, val
    return best_iv, best_val


def optimum_per_sed(fst: int, params: ModelParams) -> tuple[int, float]:
    """Best interval scored as net D change per SED. Cumulative UV varies."""
    best_iv, best_val = None, -np.inf
    for interval in INTERVALS:
        days = list(range(0, HORIZON, interval))
        sched = [(float(d), SED_PER_SESSION) for d in days]
        r = simulate(fst=fst, schedule=sched, duration_days=HORIZON,
                     params=params, D0=D0, dose_framing="fixed_per_session_SED")
        val = net_final_D_change_per_SED(r, D0)
        if val > best_val:
            best_iv, best_val = interval, val
    return best_iv, best_val


def sweep_parameter(spec: dict, fst: int) -> dict:
    """Sweep one parameter; return optimal interval AND achieved value."""
    values = np.linspace(spec["lo"], spec["hi"], N_PARAM_VALUES)
    out = {"values": values, "fixed_budget": [], "per_sed": [],
           "mean_D_at_opt": []}
    for v in values:
        p = ModelParams()
        # PARAMETERS may list RETIRED parameters; sweeping one here MEASURES
        # its influence, and does not calibrate it. As of 2026-08-21 K_ATTEN
        # is reinstated and not retired (vitd_model.RETIRED_PARAMETERS is
        # empty), so allow_retired_parameter_sweep() below is currently a
        # no-op guard for K_ATTEN specifically -- kept in place for whatever
        # parameter is retired next.
        with allow_retired_parameter_sweep():
            setattr(p, spec["attr"], float(v))
        # SC_max_by_FST overrides the scalar SC_max inside simulate(), so keep
        # them consistent when SC_max is the swept parameter.
        if spec["attr"] == "SC_max":
            p.SC_max_by_FST = {k: float(v) for k in range(1, 7)}
        iv_fb, val_fb = optimum_fixed_budget(fst, p)
        iv_ps, _ = optimum_per_sed(fst, p)
        out["fixed_budget"].append(iv_fb)
        out["per_sed"].append(iv_ps)
        out["mean_D_at_opt"].append(val_fb)
    return out


# ---------------------------------------------------------------------------
# Joint variation: all parameters at once
# ---------------------------------------------------------------------------
def joint_monte_carlo(fst: int, n_draws: int = 200, seed: int = 0) -> dict:
    """Draw every parameter uniformly from its range simultaneously."""
    rng = np.random.default_rng(seed)
    fb, ps = [], []
    for _ in range(n_draws):
        p = ModelParams()
        for spec in PARAMETERS:
            v = float(rng.uniform(spec["lo"], spec["hi"]))
            with allow_retired_parameter_sweep():       # see note above
                setattr(p, spec["attr"], v)
            if spec["attr"] == "SC_max":
                p.SC_max_by_FST = {k: v for k in range(1, 7)}
        fb.append(optimum_fixed_budget(fst, p)[0])
        ps.append(optimum_per_sed(fst, p)[0])
    return {"fixed_budget": np.array(fb), "per_sed": np.array(ps)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
C_FB = "#2C6E9B"    # fixed budget
C_PS = "#C1673D"    # per-SED
C_LV = "#4F8A5B"    # achieved level


def plot_grid(results: dict, fst: int, out_path: str):
    n = len(PARAMETERS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    # Shared scale for the achieved-level axis across panels.
    all_lv = np.concatenate([np.asarray(results[s["name"]]["mean_D_at_opt"])
                             for s in PARAMETERS])
    lv_lo, lv_hi = float(all_lv.min()), float(all_lv.max())
    pad = 0.05 * (lv_hi - lv_lo) if lv_hi > lv_lo else 1.0

    for ax, spec in zip(axes, PARAMETERS):
        r = results[spec["name"]]
        ax.plot(r["values"], r["fixed_budget"], "o-", color=C_FB, lw=2,
                ms=4, label="optimal interval, fixed UV budget")
        ax.plot(r["values"], r["per_sed"], "s-", color=C_PS, lw=2,
                ms=4, label="optimal interval, per-SED")
        ax.set_ylabel("optimal interval (d)", fontsize=8, color="#333333")
        ax.set_ylim(0, max(INTERVALS) + 1)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

        # Right axis: what the person actually ends up with at that optimum.
        ax2 = ax.twinx()
        ax2.plot(r["values"], r["mean_D_at_opt"], "^--", color=C_LV, lw=1.8,
                 ms=4, label="mean 25(OH)D achieved")
        ax2.set_ylim(lv_lo - pad, lv_hi + pad)
        ax2.set_ylabel("mean 25(OH)D (nmol/L)", fontsize=8, color=C_LV)
        ax2.tick_params(labelsize=7, colors=C_LV)

        span = float(np.ptp(r["mean_D_at_opt"]))
        ax.set_title(f"{spec['label']}\n"
                     f"interval fixed; 25(OH)D range {span:.1f} nmol/L",
                     fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    h1, l1 = axes[0].get_legend_handles_labels()
    h2 = [plt.Line2D([], [], color=C_LV, ls="--", marker="^", lw=1.8)]
    fig.legend(h1 + h2, l1 + ["mean 25(OH)D achieved (right axis)"],
               loc="lower center", ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        f"Optimal exposure interval vs. each limiting parameter  --  FST {fst}\n"
        "Under a fixed UV budget the best interval is daily for every parameter "
        "at every value. What the parameters change is the 25(OH)D you reach "
        "(right axis), not when to go out.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.945))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_joint(mc: dict, fst: int, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    # Trim the axis to where the draws actually land; the full 45-day sweep
    # range would compress both histograms into an unreadable spike.
    hi = int(max(mc["fixed_budget"].max(), mc["per_sed"].max())) + 2
    bins = np.arange(0.5, hi + 1.5, 1)
    for ax, key, color, title in (
        (axes[0], "fixed_budget", C_FB, "Fixed UV budget\n(maximise mean 25(OH)D)"),
        (axes[1], "per_sed", C_PS, "Per-SED framing\n(net $\\Delta$D per SED)"),
    ):
        v = mc[key]
        ax.hist(v, bins=bins, color=color, alpha=0.85)
        med = float(np.median(v))
        ax.axvline(med, color="black", ls="--", lw=1.5,
                   label=f"median {med:.0f} d")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("optimal interval (days)")
        ax.set_xlim(0, hi)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("draws")
    fig.suptitle(
        f"All limiting parameters varied simultaneously  --  FST {fst}, "
        f"{len(mc['fixed_budget'])} Monte Carlo draws over documented prior ranges",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    fst = 5
    print(f"Sweeping {len(PARAMETERS)} parameters x {N_PARAM_VALUES} values "
          f"x {len(INTERVALS)} intervals x 2 framings, FST {fst} ...")

    results = {}
    csv_rows = ["parameter,value,optimal_interval_fixed_budget,"
                "optimal_interval_per_sed,range_note"]
    for spec in PARAMETERS:
        r = sweep_parameter(spec, fst)
        results[spec["name"]] = r
        for v, a, b in zip(r["values"], r["fixed_budget"], r["per_sed"]):
            note = spec["note"].replace(",", ";")
            csv_rows.append(f"{spec['name']},{v:.6g},{a},{b},{note}")
        print(f"  {spec['name']:<16s} fixed-budget {min(r['fixed_budget'])}"
              f"-{max(r['fixed_budget'])} d   "
              f"per-SED {min(r['per_sed'])}-{max(r['per_sed'])} d")

    csv_path = os.path.join(TABLES_DIR, "parameter_optima.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(csv_rows) + "\n")
    print(f"Saved {csv_path}")

    plot_grid(results, fst, os.path.join(FIGS_DIR, "parameter_optima_grid.png"))

    print("\nJoint Monte Carlo (all parameters varied together) ...")
    mc = joint_monte_carlo(fst, n_draws=200)
    for key, label in (("fixed_budget", "fixed UV budget"), ("per_sed", "per-SED")):
        v = mc[key]
        lo, hi = np.percentile(v, [5, 95])
        print(f"  {label:<18s} median {np.median(v):.0f} d, "
              f"90% interval [{lo:.0f}, {hi:.0f}] d")
    plot_joint(mc, fst, os.path.join(FIGS_DIR, "parameter_optima_joint.png"))


if __name__ == "__main__":
    main()
