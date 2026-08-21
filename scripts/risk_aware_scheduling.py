"""
risk_aware_scheduling.py
========================
One actionable schedule per skin type from the deficiency-vs-cancer-risk
trade-off, plus the frontier figure the schedule was picked from.

  A. "How often should I go out, and for how long, to stay sufficient at the
      least skin-cancer risk?"
     -> vitd_optimization.optimal_interval_risk_constrained(), constraints
        C1 (never burn), C3 (sufficiency floor), C7 (session length),
        C9 (risk cap) -- MODEL_AND_EVIDENCE.md section 4.

  B. The Pareto frontier those recommendations sit on
     -> vitd_objectives.deficiency_vs_risk_frontier(), plotted per skin type
        with the recommended point marked.

Dose framing: neither fixed absolute SED nor a fixed fraction of MED. The dose
is SOLVED per skin type as the smallest one satisfying the sufficiency floor
within the erythema and session-length caps, so the per-session dose differs
across skin types by construction. Do not read the resulting doses as a
recommendation to give everyone the same fraction of their MED.

Outputs
-------
  tables/risk_aware_schedule_by_fst.csv
  figs/deficiency_risk_frontier.png

Run:  PYTHONIOENCODING=utf-8 python3 scripts/risk_aware_scheduling.py
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams
from vitd_physiology import _med_sed_by_fst
from vitd_objectives import (SUFFICIENCY_NMOL_L, deficiency_vs_risk_frontier,
                             GAMMA_MED, RAMP_IN_DAYS, PRACTICAL_MAX_MINUTES,
                             PERSONAL_DOSE_GEOMETRY_FACTOR)
from vitd_optimization import optimal_interval_risk_constrained

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS_DIR = os.path.join(ROOT, "figs")
TABLES_DIR = os.path.join(ROOT, "tables")

HORIZON = 365
D0 = 25.0
FSTS = [1, 2, 3, 4, 5, 6]

# Reference conditions, matching scripts/skin_type_breakdown.py so the two
# tables can be read side by side.
UV_INDEX_REF = 6.0
EXPOSED_BSA_REF = 0.35

# Denser at the short end, where the sufficiency floor and
# risk objective both concentrate.
INTERVAL_GRID = (1, 2, 3, 4, 5, 7, 10, 14, 21, 28)

# Report-only frontier grid. NOT the default in
# deficiency_vs_risk_frontier(): denser grids cost one full 365-day simulate()
# per point and existing callers/tests do not need them.
FRONTIER_SED_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
FRONTIER_INTERVAL_GRID = (1, 2, 3, 4, 5, 7, 10, 14, 21, 28)


# ---------------------------------------------------------------------------
# A. One recommendation per skin type
# ---------------------------------------------------------------------------
def recommend_by_skin_type(risk_cap: float | None = None,
                           max_minutes: float | None = PRACTICAL_MAX_MINUTES):
    print("=" * 96)
    print("A. LOWEST-RISK SCHEDULE THAT HOLDS SUFFICIENCY, BY SKIN TYPE")
    print("=" * 96)
    print(f"  Objective:   minimise the combined skin-cancer risk index")
    print(f"  Constraint C1: per-session dose <= {GAMMA_MED:.0%} of MED "
          f"(sub-erythemal, never burns)")
    print(f"  Constraint C3: 25(OH)D >= {SUFFICIENCY_NMOL_L:.0f} nmol/L every day "
          f"after day {RAMP_IN_DAYS}")
    if max_minutes is not None:
        print(f"  Constraint C7: session length <= {max_minutes:.0f} min")
    print(f"  Constraint C9: combined risk index <= "
          f"{risk_cap if risk_cap is not None else 'unbounded'}")
    print(f"  Exposed BSA {EXPOSED_BSA_REF:.0%}; minutes at UV index "
          f"{UV_INDEX_REF:.0f}, geometry factor {PERSONAL_DOSE_GEOMETRY_FACTOR}")
    print()
    print(f"  {'FST':>3s} {'MED':>6s} {'binding':>8s} {'interval':>9s} "
          f"{'sess/yr':>8s} {'SED/sess':>9s} {'min/sess':>9s} {'%MED':>6s} "
          f"{'ss D':>7s} {'SCC':>7s} {'melanoma':>9s} {'combined':>9s}")
    print(f"  {'-'*3} {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*9} {'-'*6} "
          f"{'-'*7} {'-'*7} {'-'*9} {'-'*9}")

    results = {}
    rows = []
    for fst in FSTS:
        res = optimal_interval_risk_constrained(
            fst=fst, target=SUFFICIENCY_NMOL_L, risk_cap=risk_cap,
            horizon_days=HORIZON, D0=D0, params=ModelParams(),
            interval_grid=INTERVAL_GRID, max_minutes=max_minutes,
            uv_index=UV_INDEX_REF, exposed_bsa=EXPOSED_BSA_REF,
            ramp_in_days=RAMP_IN_DAYS)
        results[fst] = res
        med = _med_sed_by_fst(fst)
        rec = res["recommended"]

        if rec is None:
            print(f"  {fst:>3d} {med:>6.1f} {res['binding_constraint']:>8s} "
                  f"{'NONE':>9s} {'-':>8s} {'-':>9s} {'-':>9s} {'-':>6s} "
                  f"{'-':>7s} {'-':>7s} {'-':>9s} {'-':>9s}")
            rows.append(dict(fst=fst, med_sed=med, reachable=False,
                             binding_constraint=res["binding_constraint"],
                             interval_days=None, sessions_per_year=None,
                             sed_per_session=None, minutes_per_session=None,
                             pct_of_MED=None, steady_state_D=None,
                             days_below_target=None, total_sed=None,
                             scc_index=None, intermittent_index=None,
                             combined_risk=None, burn_fraction=None,
                             note=res["note"]))
            continue

        pct_med = rec["sed_per_session"] / med * 100
        print(f"  {fst:>3d} {med:>6.1f} {res['binding_constraint']:>8s} "
              f"{rec['interval_days']:>8d}d {rec['n_sessions']:>8d} "
              f"{rec['sed_per_session']:>9.2f} {rec['minutes_per_session']:>9.0f} "
              f"{pct_med:>6.0f} {rec['steady_state_D']:>7.1f} "
              f"{rec['scc_index']:>7.2f} {rec['intermittent_index']:>9.4f} "
              f"{rec['combined_risk']:>9.4f}")
        rows.append(dict(fst=fst, med_sed=med, reachable=True,
                         binding_constraint=res["binding_constraint"],
                         interval_days=rec["interval_days"],
                         sessions_per_year=rec["n_sessions"],
                         sed_per_session=rec["sed_per_session"],
                         minutes_per_session=rec["minutes_per_session"],
                         pct_of_MED=pct_med,
                         steady_state_D=rec["steady_state_D"],
                         days_below_target=rec["days_below_target"],
                         total_sed=rec["total_sed"],
                         scc_index=rec["scc_index"],
                         intermittent_index=rec["intermittent_index"],
                         combined_risk=rec["combined_risk"],
                         burn_fraction=rec["burn_fraction"],
                         note=""))

    path = os.path.join(TABLES_DIR, "risk_aware_schedule_by_fst.csv")
    _write_csv(path, rows)
    print(f"\n  Saved {path}")

    _interpret(results)
    return results, rows


def _interpret(results):
    """State what the constrained problem did, including degenerate cases.
    A recommender that always returns the grid endpoint should say so
    rather than let the reader infer an interior optimum."""
    chosen = [(fst, r["recommended"]["interval_days"])
              for fst, r in results.items() if r["recommended"]]
    if not chosen:
        return
    shortest = min(INTERVAL_GRID)
    if all(iv == shortest for _, iv in chosen):
        print(f"\n  All skin types land on the shortest interval on the grid "
              f"({shortest}d). Not a solver artefact:")
        print("  the sufficiency floor kills long intervals (they cannot reach the"
              " target within the")
        print("  erythema and session-length caps), and among surviving options, "
              "the intermittent term")
        print("  rises steeply with per-session dose, so the most frequent "
              "schedule is also the lowest-risk.")
        print("  Deficiency and cancer risk point the same way here; the "
              "Pareto frontier over the")
        print("  unconstrained grid still trades off, but the constrained "
              "problem does not.")
        print("  C9 therefore acts as a feasibility test, not a selector.")
    print("\n  Cross-skin-type comparisons of the risk index VALUES rest on the"
          " FST susceptibility")
    print("  prior in vitd_physiology.cancer_risk — an order-of-magnitude guess"
          " NOT anchored by")
    print("  Elwood & Jopson. Interval and minutes columns do not depend on it.")
    print("  Minutes carry at least +/-40% geometric uncertainty (gap 26) and "
          "assume the UV index")
    print(f"  holds at {UV_INDEX_REF:.0f} for the whole session.")


# ---------------------------------------------------------------------------
# B. The frontier the recommendation was picked from
# ---------------------------------------------------------------------------
def plot_frontiers(results, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, fst in zip(axes.ravel(), FSTS):
        fr = deficiency_vs_risk_frontier(
            fst=fst, horizon_days=HORIZON, D0=D0,
            sed_grid=FRONTIER_SED_GRID,
            interval_grid=FRONTIER_INTERVAL_GRID,
            target=SUFFICIENCY_NMOL_L)

        pts = fr["all_points"]
        ax.scatter([p["days_below_target"] for p in pts],
                   [p["combined_risk"] for p in pts],
                   s=14, c="#B9C6D2", alpha=0.75, zorder=1,
                   label="dose x interval grid")

        front = fr["frontier"]
        ax.plot([p["days_below_target"] for p in front],
                [p["combined_risk"] for p in front],
                "o-", color="#2C6E9B", lw=1.8, ms=5, zorder=2,
                label=f"Pareto frontier (n={len(front)})")

        # The recommendation only has to hold the floor AFTER the loading
        # period, so it carries ~RAMP_IN_DAYS on the x-axis. Without this
        # line the star reads as a badly deficient point rather than a
        # fully-sufficient one that started below target.
        ax.axvline(RAMP_IN_DAYS, color="grey", ls=":", lw=1.2, zorder=0)
        ax.text(RAMP_IN_DAYS + 4, ax.get_ylim()[1], "loading period", rotation=90,
                va="top", ha="left", fontsize=7, color="grey")

        rec = results[fst]["recommended"] if fst in results else None
        if rec is not None:
            ax.plot(rec["days_below_target"], rec["combined_risk"], "*",
                    color="crimson", ms=18, zorder=3,
                    label=(f"recommended: every {rec['interval_days']}d, "
                           f"{rec['minutes_per_session']:.0f} min"))

        ax.set_title(f"FST {'I II III IV V VI'.split()[fst-1]}  "
                     f"(MED {_med_sed_by_fst(fst):.1f} SED)", fontsize=11)
        ax.set_xlabel("days below 50 nmol/L (lower is better)")
        ax.set_ylabel("combined cancer-risk index (lower is better)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "Deficiency vs skin-cancer risk. Both axes are minimised, so the "
        "bottom-left corner is best.\n"
        "The recommended point is the constrained optimum (dose solved by "
        "bisection), so it need not sit exactly on the grid-derived frontier. "
        "It starts at 25 nmol/L and is\n"
        f"only required to hold the floor after day {RAMP_IN_DAYS}, so its "
        f"~{RAMP_IN_DAYS}-day x-position is the loading period, not a "
        "sufficiency failure.",
        fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    results, _ = recommend_by_skin_type()
    plot_frontiers(results, os.path.join(FIGS_DIR, "deficiency_risk_frontier.png"))


if __name__ == "__main__":
    main()
