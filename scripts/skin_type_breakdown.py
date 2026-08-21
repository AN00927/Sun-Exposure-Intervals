"""
skin_type_breakdown.py
======================
Per-skin-type answers to the two questions the model actually answers:

  A. "Every few days is optimal -- at what dose, and for which skin type?"
     Sweeps FST I-VI against per-session dose under both dose framings, and
     reports the optimal interval and what 25(OH)D it reaches.

  B. "I do not get much sun. How often do I need to go out?"
     Solves the minimum-sessions problem: fewest sessions per year that hold
     25(OH)D at or above sufficiency, subject to never exceeding a
     sub-erythemal fraction of the individual's MED.

Problem B is the formulation recommended in MODEL_AND_EVIDENCE.md, 'Problem
statement' section 3.3.
It has an interior answer, unlike maximising 25(OH)D under a free UV budget,
which always returns "go out daily".

Dose framing appears explicitly throughout. Absolute SED and fractional MED
give different skin-type patterns, and the two literatures disagree because
papers rarely state which they used.

Outputs
-------
  figs/skin_type_breakdown.png
  tables/skin_type_breakdown.csv
  tables/min_sessions_by_fst.csv

Run:  PYTHONIOENCODING=utf-8 python3 scripts/skin_type_breakdown.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams, simulate, steady_state_D
from vitd_physiology import _med_sed_by_fst
from vitd_objectives import (SUFFICIENCY_NMOL_L, sed_from_minutes,
                             PERSONAL_DOSE_GEOMETRY_FACTOR,
                             GAMMA_MED, RAMP_IN_DAYS, PRACTICAL_MAX_MINUTES)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS_DIR = os.path.join(ROOT, "figs")
TABLES_DIR = os.path.join(ROOT, "tables")

HORIZON = 365
D0 = 25.0
FSTS = [1, 2, 3, 4, 5, 6]
ABSOLUTE_SEDS = [0.5, 1.0, 2.0, 4.0]
MED_FRACTIONS = [0.1, 0.25, 0.5]
INTERVALS = list(range(1, 31))

# GAMMA_MED (sub-erythemal ceiling), RAMP_IN_DAYS and PRACTICAL_MAX_MINUTES are
# model-level constraints in vitd_objectives, shared with vitd_optimization.
# Reference conditions for the minutes translation.
UV_INDEX_REF = 6.0
EXPOSED_BSA_REF = 0.35


def _run(fst: int, sed: float, interval: int, params: ModelParams | None = None,
         bsa: float | None = None):
    days = list(range(0, HORIZON, interval))
    sched = [(float(d), sed) for d in days]
    return simulate(fst=fst, schedule=sched, duration_days=HORIZON,
                    params=params or ModelParams(), D0=D0,
                    exposed_bsa_override=bsa,
                    dose_framing="fixed_per_session_SED")


# ---------------------------------------------------------------------------
# A. Optimal interval by skin type and dose
# ---------------------------------------------------------------------------
def breakdown_by_skin_type():
    print("=" * 88)
    print("A. OPTIMAL INTERVAL BY SKIN TYPE AND DOSE")
    print("=" * 88)
    rows = []

    for framing, doses in (("absolute_SED", ABSOLUTE_SEDS),
                           ("fractional_MED", MED_FRACTIONS)):
        print(f"\n  Dose framing: {framing}")
        print(f"  {'FST':>3s} {'dose':>12s} {'SED/sess':>9s} {'best iv':>8s} "
              f"{'ss D':>8s} {'d<50':>6s} {'%MED':>6s} {'burn?':>6s}")
        print(f"  {'-'*3} {'-'*12} {'-'*9} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")

        for fst in FSTS:
            med = _med_sed_by_fst(fst)
            for dose in doses:
                sed = dose if framing == "absolute_SED" else dose * med
                # Best interval under the fixed-budget-equivalent objective:
                # maximise mean 25(OH)D. Dose per session is held.
                best_iv, best_mean, best_r = None, -np.inf, None
                for iv in INTERVALS:
                    r = _run(fst, sed, iv)
                    m = float(np.mean(r["D"]))
                    if m > best_mean:
                        best_iv, best_mean, best_r = iv, m, r
                ss = steady_state_D(best_r)
                below = float(np.sum(np.asarray(best_r["D"]) < SUFFICIENCY_NMOL_L)
                              / len(best_r["D"]) * HORIZON)
                pct_med = sed / med * 100
                burn = "YES" if sed >= med else ""
                label = (f"{dose:.2f} SED" if framing == "absolute_SED"
                         else f"{dose:.0%} MED")
                print(f"  {fst:>3d} {label:>12s} {sed:>9.2f} {best_iv:>7d}d "
                      f"{ss:>8.1f} {below:>6.0f} {pct_med:>6.0f} {burn:>6s}")
                rows.append(dict(framing=framing, fst=fst, dose_label=label,
                                 sed_per_session=sed, best_interval=best_iv,
                                 steady_state_D=ss, days_below_50=below,
                                 pct_of_MED=pct_med, would_burn=bool(sed >= med)))

    path = os.path.join(TABLES_DIR, "skin_type_breakdown.csv")
    _write_csv(path, rows)
    print(f"\n  Saved {path}")
    return rows


# ---------------------------------------------------------------------------
# B. Minimum sessions to hold sufficiency
# ---------------------------------------------------------------------------
def min_sessions_for_sufficiency(target: float = SUFFICIENCY_NMOL_L,
                                 max_minutes: float | None = None,
                                 label: str = "burn constraint only"):
    print("\n" + "=" * 88)
    print(f"B. FEWEST SESSIONS THAT HOLD SUFFICIENCY  --  {label}")
    print("=" * 88)
    print(f"  Constraint: per-session dose <= {GAMMA_MED:.0%} of MED "
          f"(sub-erythemal, never burns)")
    if max_minutes is not None:
        print(f"  Constraint: session length <= {max_minutes:.0f} min "
              f"(C7, what a person will actually do)")
    print(f"  Requirement: 25(OH)D >= {target:.0f} nmol/L for every day after "
          f"day {RAMP_IN_DAYS}")
    print(f"  Exposed BSA {EXPOSED_BSA_REF:.0%}; minutes computed at UV index "
          f"{UV_INDEX_REF:.0f}, geometry factor {PERSONAL_DOSE_GEOMETRY_FACTOR}")
    print()
    print(f"  {'FST':>3s} {'MED':>6s} {'dose':>7s} {'binding':>9s} "
          f"{'min sess/yr':>12s} "
          f"{'interval':>9s} {'per week':>9s} {'min/session':>12s} "
          f"{'ss D':>7s}")
    print(f"  {'-'*3} {'-'*6} {'-'*7} {'-'*9} {'-'*12} {'-'*9} {'-'*9} "
          f"{'-'*12} {'-'*7}")

    sed_per_min = sed_from_minutes(1.0, UV_INDEX_REF)
    rows = []
    for fst in FSTS:
        med = _med_sed_by_fst(fst)
        sed_burn = GAMMA_MED * med
        if max_minutes is None:
            sed = sed_burn
            binding = "MED"
        else:
            sed_time = max_minutes * sed_per_min
            sed = min(sed_burn, sed_time)
            binding = "MED" if sed_burn <= sed_time else "minutes"

        # min D after ramp-in is monotone non-decreasing in session count
        # for uniform spacing, so bisect on the number of sessions.
        def feasible(n_sessions: int):
            if n_sessions < 1:
                return False, None
            interval = max(1, HORIZON // n_sessions)
            r = _run(fst, sed, interval)
            t = np.asarray(r["t"])
            D = np.asarray(r["D"])
            mask = t >= RAMP_IN_DAYS
            return (float(np.min(D[mask])) >= target if mask.any() else False), r

        ok_max, _ = feasible(HORIZON)
        if not ok_max:
            _, r = feasible(HORIZON)
            ss = steady_state_D(r) if r else float("nan")
            print(f"  {fst:>3d} {med:>6.1f} {sed:>7.2f} {binding:>9s} "
                  f"{'UNREACHABLE':>12s} "
                  f"{'-':>9s} {'-':>9s} {'-':>12s} {ss:>7.1f}")
            rows.append(dict(fst=fst, med_sed=med, sed_per_session=sed,
                             binding_constraint=binding,
                             min_sessions_per_year=None, interval_days=None,
                             sessions_per_week=None, minutes_per_session=None,
                             steady_state_D=ss, reachable=False))
            continue

        lo, hi = 1, HORIZON
        while lo < hi:
            mid = (lo + hi) // 2
            ok, _ = feasible(mid)
            if ok:
                hi = mid
            else:
                lo = mid + 1
        n_star = lo
        _, r = feasible(n_star)
        interval = max(1, HORIZON // n_star)
        per_week = 7.0 / interval
        # Minutes needed to deliver `sed` at the reference UV index.
        minutes = sed / sed_per_min if sed_per_min > 0 else float("nan")
        ss = steady_state_D(r)
        print(f"  {fst:>3d} {med:>6.1f} {sed:>7.2f} {binding:>9s} "
              f"{n_star:>12d} "
              f"{interval:>8d}d {per_week:>9.2f} {minutes:>12.0f} {ss:>7.1f}")
        rows.append(dict(fst=fst, med_sed=med, sed_per_session=sed,
                         binding_constraint=binding,
                         min_sessions_per_year=n_star, interval_days=interval,
                         sessions_per_week=per_week, minutes_per_session=minutes,
                         steady_state_D=ss, reachable=True))

    suffix = "practical" if max_minutes is not None else "burnonly"
    path = os.path.join(TABLES_DIR, f"min_sessions_by_fst_{suffix}.csv")
    _write_csv(path, rows)
    print(f"\n  Saved {path}")

    # The MED cap scales with skin type, so darker skin gets a much larger
    # per-session dose — and the solver spends it. Check whether the resulting
    # session length is something a person could actually do.
    impractical = [r for r in rows
                   if r["reachable"] and r["minutes_per_session"] > PRACTICAL_MAX_MINUTES]
    if impractical:
        print(f"\n  WARNING: the {GAMMA_MED:.0%}-MED cap alone does not produce a"
              f" usable recommendation for every skin type.")
        for r in impractical:
            print(f"    FST {r['fst']}: {r['minutes_per_session']:.0f} min/session "
                  f"exceeds the {PRACTICAL_MAX_MINUTES:.0f} min practical ceiling.")
        print(f"  MED rises from {_med_sed_by_fst(1):.1f} SED (FST I) to "
              f"{_med_sed_by_fst(6):.1f} SED (FST VI), so a fixed MED fraction "
              f"licenses a {_med_sed_by_fst(6)/_med_sed_by_fst(1):.1f}x larger dose")
        print("  for FST VI, which then takes hours of continuous exposure to "
              "deliver.")
        print("  A session-length constraint (C7 in MODEL_AND_EVIDENCE.md, "
              "'Problem statement') has to bind here.")
        print("  Treat these rows as showing the constraint set is incomplete, "
              "not as advice.")

    print("\n  Minutes carry at least +/-40% geometric uncertainty: the")
    print("  personal-to-ambient exposure ratio is a scalar here, and")
    print("  Schmalwieser 2018 shows it varies with solar elevation.")
    print("  See MODEL_AND_EVIDENCE.md, 'Remaining gaps', gap 26.")
    print("  Minutes also assume UV index holds at "
          f"{UV_INDEX_REF:.0f} for the whole session, which fails for long ones.")
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("" if r[k] is None else str(r[k])
                              for k in keys) + "\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(breakdown_rows, minsess_rows, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # -- panel 1: steady-state D by FST, absolute SED framing ---------------
    ax = axes[0]
    for dose in ABSOLUTE_SEDS:
        xs, ys = [], []
        for r in breakdown_rows:
            if r["framing"] == "absolute_SED" and abs(r["sed_per_session"] - dose) < 1e-9:
                xs.append(r["fst"]); ys.append(r["steady_state_D"])
        ax.plot(xs, ys, "o-", lw=2, ms=5, label=f"{dose:g} SED")
    ax.axhline(SUFFICIENCY_NMOL_L, color="crimson", ls="--", lw=1.2)
    ax.text(4.4, SUFFICIENCY_NMOL_L + 1.5, "sufficiency 50", color="crimson",
            fontsize=8)
    ax.set_title("Absolute SED framing\nsame dose for everyone", fontsize=11)
    ax.set_xlabel("Fitzpatrick skin type"); ax.set_ylabel("steady-state 25(OH)D (nmol/L)")
    ax.legend(fontsize=8, title="per session", title_fontsize=8)
    ax.grid(alpha=0.3)

    # -- panel 2: same, fractional MED framing ------------------------------
    ax = axes[1]
    for frac in MED_FRACTIONS:
        xs, ys = [], []
        for r in breakdown_rows:
            if r["framing"] == "fractional_MED" and r["dose_label"] == f"{frac:.0%} MED":
                xs.append(r["fst"]); ys.append(r["steady_state_D"])
        ax.plot(xs, ys, "s-", lw=2, ms=5, label=f"{frac:.0%} MED")
    ax.axhline(SUFFICIENCY_NMOL_L, color="crimson", ls="--", lw=1.2)
    ax.set_title("Fractional MED framing\ndose scaled to each person's threshold",
                 fontsize=11)
    ax.set_xlabel("Fitzpatrick skin type")
    ax.legend(fontsize=8, title="per session", title_fontsize=8)
    ax.grid(alpha=0.3)

    # -- panel 3: minimum sessions ------------------------------------------
    ax = axes[2]
    xs = [r["fst"] for r in minsess_rows if r["reachable"]]
    ys = [r["min_sessions_per_year"] for r in minsess_rows if r["reachable"]]
    unreach = [r["fst"] for r in minsess_rows if not r["reachable"]]
    ax.bar(xs, ys, color="#2C6E9B", alpha=0.85)
    for x, y in zip(xs, ys):
        ax.text(x, y + 2, f"{y}", ha="center", fontsize=9)
    for x in unreach:
        ax.text(x, 5, "unreachable", ha="center", fontsize=8, rotation=90,
                color="crimson")
    ax.set_title(f"Fewest sessions holding 25(OH)D $\\geq$ 50\n"
                 f"at $\\leq${GAMMA_MED:.0%} MED per session", fontsize=11)
    ax.set_xlabel("Fitzpatrick skin type"); ax.set_ylabel("sessions per year")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Skin-type breakdown. The two dose framings disagree about who is "
        "disadvantaged, which is why every result must name its framing.",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    rows = breakdown_by_skin_type()
    # Burn constraint alone: shows the constraint set is incomplete.
    min_sessions_for_sufficiency(label="burn constraint only")
    # Burn AND session-length: the version anyone can act on.
    ms = min_sessions_for_sufficiency(
        max_minutes=PRACTICAL_MAX_MINUTES,
        label=f"burn + {PRACTICAL_MAX_MINUTES:.0f} min session cap")
    plot(rows, ms, os.path.join(FIGS_DIR, "skin_type_breakdown.png"))


if __name__ == "__main__":
    main()
