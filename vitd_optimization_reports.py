"""Report generation, visualization, and high-level analysis routines for vitamin D optimization.

Extracted from vitd_optimization.py to decouple presentation, plotting,
and tabular reports from core numerical optimization algorithms and cost functions.
"""
from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dataclasses

from vitd_model import (ModelParams, simulate, steady_state_D, get_E_source,
                        SC_MAX_IV_VI_RANGE)
from vitd_objectives import (
    GAMMA_MED,
    RAMP_IN_DAYS,
    SUFFICIENCY_NMOL_L,
    PRACTICAL_MAX_MINUTES,
    PERSONAL_DOSE_GEOMETRY_FACTOR,
)
from vitd_optimization import (
    TRI_GAMMA_GRID,
    TRI_SESSION_BUDGETS,
    VARIABLE_DOSE_FAMILIES,
    SESSION_BUDGET_BASIS,
    _med_sed_by_fst,
    _joint_seds,
    warp_days,
    build_variable_schedule,
    dose_exponential,
    dose_two_phase,
    dose_uniform,
    evaluate_schedule,
    gross_synth_per_SED_diagnostic,
    minimum_schedule_for_sufficiency,
    optimise_joint_schedule,
    optimise_variable_dose,
    schedule_uniform,
    session_days,
    tri_objective_frontier,
)

def _joint_grid_cell(task):
    """Solve one (FST, session-budget) cell of the joint-schedule grid.

    Module level, and taking a single picklable tuple, so it can be dispatched
    to a ProcessPoolExecutor under Windows' `spawn` start method (a closure or
    lambda would not pickle). Returns the two arms plus, when either arm is
    infeasible, the binding-constraint string from the cheap
    uniform-at-the-ceiling probe -- the probe is deterministic, so computing it
    once per cell here gives the same string the serial code produced per arm.
    """
    fst, n_sessions, params = task
    fixed = optimise_joint_schedule(fst, n_sessions, free_spacing=False,
                                    params=params)
    joint = optimise_joint_schedule(fst, n_sessions, free_spacing=True,
                                    params=params)
    binding = None
    if fixed is None or joint is None:
        at_ceiling = optimise_variable_dose(fst, n_sessions, "uniform",
                                            params=params)
        binding = (f"C1 erythema cap ({GAMMA_MED:.0%} MED) at this session budget"
                   if at_ceiling is None else
                   "no schedule in the searched dose families")
    return fst, n_sessions, fixed, joint, binding


def _run_joint_grid(tasks, n_workers: Optional[int] = None):
    """Run the grid cells, in a process pool when possible, serially otherwise.

    `ProcessPoolExecutor.map` yields results in submission order, so the
    returned list is in deterministic (fst, N) order regardless of which cell
    finishes first. Any failure to start or use the pool degrades to the serial
    path rather than taking the pipeline down.
    """
    n = (os.cpu_count() or 1) if n_workers is None else int(n_workers)
    n = max(1, min(n, len(tasks)))
    if n > 1:
        try:
            with ProcessPoolExecutor(max_workers=n) as ex:
                return list(ex.map(_joint_grid_cell, tasks))
        except Exception as exc:                                # pragma: no cover
            print(f"  [process pool unavailable ({type(exc).__name__}: {exc}); "
                  f"running the grid serially]")
    return [_joint_grid_cell(t) for t in tasks]


def analysis_joint_schedule(params: Optional[ModelParams] = None,
                            fsts: Sequence[int] = (1, 2, 3, 4, 5, 6),
                            session_counts: Sequence[int] = (104, 146, 183, 260),
                            n_workers: Optional[int] = None):
    """Does free spacing buy anything once the dose is already free?

    Both arms use the SAME solver, budget and dose families; the only difference
    is whether the spacing exponent is searched or pinned to uniform. Any
    difference is therefore attributable to spacing rather than to search
    effort.

    All six skin types are searched. FST VI is the priority population for this
    project and FST I is the other end of the range; running only II/III/V left
    the headline optimisation table silent about both. Several (FST, budget)
    cells are INFEASIBLE after the 2026-08-17 K_const recalibration -- they are
    emitted as rows with `feasible = False` and a binding constraint, never
    dropped, because "this skin type cannot do it at this budget" is the result
    most worth reporting.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS N: joint timing + dose optimisation")
    print("=" * 70)
    print(f"\n  min combined cancer risk  s.t.  min D(t >= ramp) >= "
          f"{SUFFICIENCY_NMOL_L:.0f} nmol/L,")
    print(f"  per-session dose <= {GAMMA_MED:.0%} MED, session budget fixed per row.")
    print("  Control arm pins spacing to uniform; treatment arm frees the")
    print("  spacing exponent. Same solver and budget in both.")
    print("\n  A session budget is part of the problem: without one the optimum")
    print("  is degenerate (risk falls monotonically to a daily schedule),")
    print("  because at fixed cumulative dose the intermittency term is")
    print("  minimised by spreading over as many days as possible.")

    rows = []
    # PARALLELISM IS AT THE CELL LEVEL, DELIBERATELY. Do NOT "optimise" this
    # into differential_evolution(workers=-1): scipy forces
    # updating='deferred' whenever workers != 1, which changes the DE search
    # trajectory and would move every seeded result and every pinned test
    # value. The grid cells, by contrast, are independent and each DE solve
    # inside them carries its own fixed seed, so dispatching whole cells
    # reproduces the serial CSV byte for byte.
    tasks = [(fst, N, params) for fst in fsts for N in session_counts]
    for fst, N, fixed, joint, binding in _run_joint_grid(tasks, n_workers):
        for arm_free, r in ((False, fixed), (True, joint)):
            if r is None:
                # Infeasible cells are RESULTS. Attribute the failure with
                # the same constraint set the search itself used (C1 only;
                # this section does not impose the session-length ceiling),
                # probed by the cheap uniform-at-the-ceiling solve
                # (computed once per cell in _joint_grid_cell).
                rows.append({
                    "fst": fst, "n_sessions": N,
                    "free_spacing": arm_free,
                    "feasible": False, "binding_constraint": binding,
                    "family": "", "gamma": float("nan"),
                    "combined_risk": float("nan"),
                    "scc_index": float("nan"),
                    "intermittent_index": float("nan"),
                    "trough_after_ramp": float("nan"),
                    "total_sed": float("nan"),
                    "time_to_sufficiency": float("nan"),
                    "risk_vs_uniform_spacing_pct": float("nan"),
                    "scc_vs_uniform_spacing_pct": float("nan"),
                    "intermittent_vs_uniform_spacing_pct": float("nan"),
                })
                continue
            r["feasible"] = True
            r["binding_constraint"] = ""
            if fixed is not None and fixed["combined_risk"] > 0:
                r["risk_vs_uniform_spacing_pct"] = (
                    100.0 * (fixed["combined_risk"] - r["combined_risk"])
                    / fixed["combined_risk"])
            else:
                r["risk_vs_uniform_spacing_pct"] = float("nan")
            # SCC and intermittent (BCC/melanoma) tracked SEPARATELY, not
            # just the geometric-mean combined_risk: they are the two
            # literature-anchored axes (Moan 1989 dose power law; Elwood/
            # Leiter intermittency), and the combination itself is not.
            if fixed is not None and fixed.get("scc_index", 0) > 0:
                r["scc_vs_uniform_spacing_pct"] = (
                    100.0 * (fixed["scc_index"] - r["scc_index"])
                    / fixed["scc_index"])
            else:
                r["scc_vs_uniform_spacing_pct"] = float("nan")
            if fixed is not None and fixed.get("intermittent_index", 0) > 0:
                r["intermittent_vs_uniform_spacing_pct"] = (
                    100.0 * (fixed["intermittent_index"] - r["intermittent_index"])
                    / fixed["intermittent_index"])
            else:
                r["intermittent_vs_uniform_spacing_pct"] = float("nan")
            rows.append(r)

    if not rows:
        print("\n  No feasible schedule at any tested (FST, session budget).")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv("tables/joint_schedule_optimization.csv", index=False)

    print("\n  SCC and intermittent (BCC/melanoma) risk are the two")
    print("  literature-anchored axes (Moan 1989 cumulative-dose power law;")
    print("  Elwood/Leiter intermittency) and are reported SEPARATELY below.")
    print("  'combined' (geometric mean) is a ranking-only convenience, not")
    print("  itself literature-anchored -- see vitd_physiology.cancer_risk.")
    if 6 in fsts:
        print()
        print(FST_VI_RESOLUTION_NOTE)
    print(f"\n  {'FST':>4} {'N':>4} {'spacing':>9} {'family':>12} {'gamma':>6} "
          f"{'SCC%':>7} {'interm%':>8} {'combined%':>10} {'totSED':>8} {'t->suff':>8}")
    print("  " + "-" * 96)
    for fst in fsts:
        for N in session_counts:
            sub = df[(df["fst"] == fst) & (df["n_sessions"] == N)]
            for _, r in sub.iterrows():
                arm = "free" if r["free_spacing"] else "uniform"
                if not r["feasible"]:
                    print(f"  {ROMAN_FST[int(r['fst'])]:>4} "
                          f"{int(r['n_sessions']):>4} {arm:>9} "
                          f"{'--':>12} {'--':>6} {'--':>7} {'--':>8} {'--':>10} "
                          f"{'--':>8} {'--':>8}   INFEASIBLE: "
                          f"{r['binding_constraint']}")
                    continue
                ttf = ("n/a" if not np.isfinite(r["time_to_sufficiency"])
                       else f"{r['time_to_sufficiency']:.0f}d")
                scc_pct = ("--" if not np.isfinite(r["scc_vs_uniform_spacing_pct"])
                           else f"{r['scc_vs_uniform_spacing_pct']:.1f}")
                interm_pct = ("--" if not np.isfinite(r["intermittent_vs_uniform_spacing_pct"])
                              else f"{r['intermittent_vs_uniform_spacing_pct']:.1f}")
                print(f"  {ROMAN_FST[int(r['fst'])]:>4} {int(r['n_sessions']):>4} "
                      f"{arm:>9} "
                      f"{r['family']:>12} {r['gamma']:>6.2f} "
                      f"{scc_pct:>7} {interm_pct:>8} "
                      f"{r['risk_vs_uniform_spacing_pct']:>10.1f} "
                      f"{r['total_sed']:>8.1f} {ttf:>8}")
            if len(sub):
                print()

    n_cells = len(fsts) * len(session_counts)
    infeas = df[~df["feasible"].astype(bool)]
    infeas_cells = sorted({(int(r["fst"]), int(r["n_sessions"]))
                           for _, r in infeas.iterrows()})
    print(f"  Feasibility: {n_cells - len(infeas_cells)} of {n_cells} "
          f"(skin type, session budget) cells admit a schedule.")
    if infeas_cells:
        print("  Infeasible cells, reported as results rather than dropped:")
        for f_, n_ in infeas_cells:
            b = infeas[(infeas["fst"] == f_) & (infeas["n_sessions"] == n_)]
            reason = b["binding_constraint"].iloc[0] if len(b) else "?"
            print(f"    FST {ROMAN_FST[f_]:<4} N = {n_:<5} {reason}")
        print("  Nothing was relaxed to fill these. They follow from the")
        print("  2026-08-17 K_const recalibration, are asserted in")
        print("  tests/test_vitd_optimization.py::TestRecalibratedFeasibility,")
        print("  and are mapped in scripts/feasibility_map_kconst.py.")

    free = df[df["free_spacing"].astype(bool) & df["feasible"].astype(bool)]
    if len(free):
        gains = free["risk_vs_uniform_spacing_pct"].to_numpy(dtype=float)
        gains = gains[np.isfinite(gains)]
        scc_gains = free["scc_vs_uniform_spacing_pct"].to_numpy(dtype=float)
        scc_gains = scc_gains[np.isfinite(scc_gains)]
        interm_gains = free["intermittent_vs_uniform_spacing_pct"].to_numpy(dtype=float)
        interm_gains = interm_gains[np.isfinite(interm_gains)]
        print(f"  Free spacing vs uniform spacing, {len(gains)} operating points:")
        print(f"    SCC (cumulative-dose axis):        median "
              f"{np.median(scc_gains):+.2f}%   best {np.max(scc_gains):+.2f}%   "
              f"worst {np.min(scc_gains):+.2f}%")
        print(f"    Intermittent (BCC/melanoma axis):  median "
              f"{np.median(interm_gains):+.2f}%   best {np.max(interm_gains):+.2f}%   "
              f"worst {np.min(interm_gains):+.2f}%")
        print(f"    Combined (ranking-only, NOT literature-anchored as a "
              f"combination): median {np.median(gains):+.2f}%   "
              f"best {np.max(gains):+.2f}%   worst {np.min(gains):+.2f}%")
        if np.median(scc_gains) < 0 < np.median(interm_gains):
            print(f"    Front-loading trades a small SCC (cumulative-dose) "
                  f"INCREASE for a large intermittent (BCC/melanoma) DECREASE --")
            print(f"    the two axes move in OPPOSITE directions. Report them "
                  f"separately; a single combined percentage hides this.")
        print(f"    gamma chosen: median {np.median(free['gamma']):.2f} "
              f"(1.00 = uniform), range "
              f"{np.min(free['gamma']):.2f}-{np.max(free['gamma']):.2f}")

        # A negative gain is impossible in principle: gamma = 1 lies inside the
        # bounds, so the free arm can always express the control exactly. Any
        # negative entry is therefore pure differential-evolution noise, and its
        # magnitude is a free calibration of the solver's run-to-run spread --
        # which is worth more than hiding it would be.
        #
        # CAVEAT [reference purged 2026-09-19]: this estimator only
        # lower-bounds solver noise. It takes the single most negative gain
        # across ~22 (FST, N) cells at seed=0, which is one draw per cell, not
        # a replication. Direct 5-seed replication at one cell (FST III,
        # N=146) spans 6.3% in combined_risk with the winning dose family
        # itself changing by seed, and paired free-vs-uniform gains at that
        # cell range +3.0% to +12.1% (mean +6.75%, sd 3.4 pp) -- roughly 2-3x
        # wider than the bound below suggests. Call optimise_joint_schedule
        # with n_seeds > 1 and use its risk_min/risk_max/risk_spread_frac for
        # a measured bound instead of this inferred one. The sign of the
        # free-vs-uniform gain replicates (5/5 positive across the seeds
        # checked); per-cell magnitudes below roughly 10 percentage points do
        # not, and should not be compared across cells.
        worst = float(np.min(gains))
        if worst < 0:
            print(f"\n    A negative entry is not a real loss: gamma = 1 is")
            print(f"    inside the search bounds, so the free arm can reproduce")
            print(f"    the control exactly and cannot truly do worse. The worst")
            print(f"    entry ({worst:+.1f}%) therefore LOWER-BOUNDS solver noise")
            print(f"    (direct seed replication at one cell spans ~9 percentage")
            print(f"    points, roughly 2-3x this figure -- see the CAVEAT comment")
            print(f"    in analysis_joint_schedule). Gains below roughly 10%")
            print(f"    should not be read as real, and cross-cell magnitude")
            print(f"    comparisons below that are not supported.")

    print("\n  Mechanism. Free spacing does NOT win by lowering cumulative dose")
    print("  -- total SED is usually slightly HIGHER. It wins by lowering the")
    print("  PEAK per-session dose: bunching sessions more densely early lets")
    print("  the loading phase use smaller individual doses. The intermittency")
    print("  term is cubic in per-session dose while the SCC term is a 1.7 power")
    print("  of the cumulative, so trading a small cumulative increase for a")
    print("  large peak reduction is strongly favourable.")
    print("\n  This REVERSES the fixed-dose result: at fixed dose, spacing only")
    print("  redistributes an unchangeable per-session intensity, and uniform")
    print("  spacing wins by minimising the largest gap. Once dose is free,")
    print("  spacing acquires a second job -- setting how much dose each session")
    print("  needs to carry -- and front-loading takes over.")
    if len(free):
        n_won = int(np.sum(gains > 0))
        n_front = int(np.sum(free["gamma"].to_numpy(dtype=float) > 1.0))
        print(f"\n  Precisely: the search selected gamma > 1 at {n_front} of "
              f"{len(free)} points,")
        print(f"  and beat the uniform control at {n_won} of {len(gains)}.")
        if n_won < len(gains):
            print(f"  The {len(gains) - n_won} remaining point(s) sit inside the "
                  f"solver-noise bound")
            print(f"  above, so they are not evidence against front-loading -- but")
            print(f"  they are not evidence for it either. The claim is "
                  f"{n_won}/{len(gains)}, not unanimous.")
    print(f"\n  Saved tables/joint_schedule_optimization.csv")
    return df


def analysis_variable_dose(params: Optional[ModelParams] = None,
                           fsts: Sequence[int] = (2, 3, 5),
                           session_counts: Sequence[int] = (104, 122, 146, 183)):
    """Variable dose vs the constant-dose baseline, at matched efficacy.

    Every profile compared here holds the SAME sufficiency floor with the SAME
    number of sessions on the SAME days, so the comparison cannot be won by
    dosing less or going out more -- only by distributing dose better.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS M: variable-dose optimisation")
    print("=" * 70)
    print(f"\n  min combined cancer risk  s.t.  min D(t >= ramp) >= "
          f"{SUFFICIENCY_NMOL_L:.0f} nmol/L,")
    print(f"  per-session dose <= {GAMMA_MED:.0%} MED, sessions uniformly spaced.")
    print("  Dose framing: absolute SED per session (not fractional MED).")

    rows = []
    for fst in fsts:
        for N in session_counts:
            base = optimise_variable_dose(fst, N, "uniform", params=params)
            for fam in VARIABLE_DOSE_FAMILIES:
                res = (base if fam == "uniform"
                       else optimise_variable_dose(fst, N, fam, params=params))
                if res is None:
                    continue
                if base is not None and base["combined_risk"] > 0:
                    res["risk_vs_uniform_pct"] = (
                        100.0 * (base["combined_risk"] - res["combined_risk"])
                        / base["combined_risk"])
                    res["sed_vs_uniform_pct"] = (
                        100.0 * (base["total_sed"] - res["total_sed"])
                        / base["total_sed"]) if base["total_sed"] > 0 else 0.0
                else:
                    res["risk_vs_uniform_pct"] = float("nan")
                    res["sed_vs_uniform_pct"] = float("nan")
                rows.append(res)

    if not rows:
        print("\n  No feasible profile at any tested (FST, session count).")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv("tables/variable_dose_optimization.csv", index=False)

    print(f"\n  {'FST':>3} {'N':>4} {'family':>12} {'risk':>9} {'vs unif%':>9} "
          f"{'totSED':>8} {'SED sav%':>9} {'trough':>7} {'t->suff':>8}")
    print("  " + "-" * 82)
    for fst in fsts:
        for N in session_counts:
            sub = df[(df["fst"] == fst) & (df["n_sessions"] == N)]
            for _, r in sub.iterrows():
                ttf = ("n/a" if not np.isfinite(r["time_to_sufficiency"])
                       else f"{r['time_to_sufficiency']:.0f}d")
                print(f"  {int(r['fst']):>3} {int(r['n_sessions']):>4} "
                      f"{r['family']:>12} {r['combined_risk']:>9.4f} "
                      f"{r['risk_vs_uniform_pct']:>9.1f} {r['total_sed']:>8.1f} "
                      f"{r['sed_vs_uniform_pct']:>9.1f} "
                      f"{r['trough_after_ramp']:>7.2f} {ttf:>8}")
            if len(sub):
                print()

    best = df.loc[df.groupby(["fst", "n_sessions"])["combined_risk"].idxmin()]
    print("  Lowest-risk family at each operating point:")
    for _, r in best.iterrows():
        print(f"    FST {int(r['fst'])}, {int(r['n_sessions'])} sessions/yr: "
              f"{r['family']} ({r['risk_vs_uniform_pct']:.1f}% less risk, "
              f"{r['sed_vs_uniform_pct']:.1f}% less total UV)")

    print("\n  Interpretation. At fixed budget AND session count the")
    print("  intermittency term is convex in the doses, so equal doses minimise")
    print("  it (Jensen) -- variable dose cannot win on the risk axis alone.")
    print("  Every gain below therefore comes from needing LESS total UV:")
    print("  a constant dose must both climb to target and hold there, so it")
    print("  overshoots in steady state. The gain shrinks as sessions rise,")
    print("  because a frequent schedule already runs a small per-session dose.")
    print("\n  Absolute D inherits the uncalibrated SC priors (SC_max, alpha_SC,")
    print("  K_SC); rankings within a skin type are more robust than levels.")
    print(f"\n  Saved tables/variable_dose_optimization.csv")

    _plot_variable_dose(df, params=params)
    return df


def _plot_variable_dose(df: pd.DataFrame, params: Optional[ModelParams] = None,
                        fst: int = 3, n_sessions: int = 122,
                        horizon_days: int = 365, D0: float = 25.0):
    """Dose profiles and the 25(OH)D trajectories they produce."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    sub = df[(df["fst"] == fst) & (df["n_sessions"] == n_sessions)]
    if sub.empty:
        return
    days = session_days(n_sessions, horizon_days)
    cap = GAMMA_MED * _med_sed_by_fst(fst)

    profiles = {}
    for _, r in sub.iterrows():
        fam = r["family"]
        if fam == "uniform":
            seds = dose_uniform(days, r["dose_sed"])
        elif fam == "two_phase":
            seds = dose_two_phase(days, r["d_load_sed"], r["d_maint_sed"],
                                  r["t_switch"])
        elif fam == "exponential":
            seds = dose_exponential(days, r["d_inf_sed"], r["ratio"], r["tau"])
        else:
            continue
        profiles[fam] = seds

    if len(profiles) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fam, seds in profiles.items():
        sched = build_variable_schedule(days, seds, fst=fst)
        axes[0].step(days, np.clip(seds, 0, cap), where="post", label=fam, lw=1.8)
        r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                     params=params, D0=D0, dose_framing="fixed_per_session_SED")
        axes[1].plot(r["t"], r["D"], label=fam, lw=1.8)

    axes[0].axhline(cap, color="crimson", ls="--", alpha=0.6,
                    label=f"{GAMMA_MED:.0%} MED cap ({cap:.2f} SED)")
    axes[0].set_xlabel("day"); axes[0].set_ylabel("SED per session")
    axes[0].set_title(f"Dose profile (FST {fst}, {n_sessions} sessions/yr)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].axhline(SUFFICIENCY_NMOL_L, color="gray", ls="--", alpha=0.6,
                    label=f"sufficiency {SUFFICIENCY_NMOL_L:.0f}")
    axes[1].axvline(RAMP_IN_DAYS, color="gray", ls=":", alpha=0.6,
                    label=f"ramp-in ends ({RAMP_IN_DAYS}d)")
    axes[1].set_xlabel("day"); axes[1].set_ylabel("25(OH)D (nmol/L)")
    axes[1].set_title("All profiles hold the same floor, at different cost")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figs/variable_dose_profiles.png", dpi=130)
    plt.close(fig)
    print("  Saved figs/variable_dose_profiles.png")



ROMAN_FST = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

# [decision B2, 2026-09-23] Point-of-use label for every FST VI output below.
# Superseded wording: decision A3 (2026-09-22) released the ITA -> M mapping's
# clip (see vitd_model.py's M_baseline_by_FST entry), so the pre-A3 label --
# "the melanin index is a saturated boundary with no within-FST-VI resolving
# power" -- is no longer true: FST VI's cell moved off the 1.00 ceiling and the
# mapping now DOES vary within FST VI. What replaces it is the honest
# post-release description: that resolving power rests on a single,
# digitised-from-a-box-plot lower anchor (Linde 2020; +/-0.5 deg ITA residual
# error) whose upper anchor and denominator remain unsourced, and that the
# calibration set behind every synthesis parameter still contains no
# non-lamp FST VI vitamin-D datapoint (vitd_model.py's own docstring: "every
# FST VI datapoint is lamp-based"). Read from vitd_model.py rather than
# restated here in numbers, per CLAUDE.md's "do not duplicate parameter
# values" rule -- this label states what the gap IS, not its numeric size.
FST_VI_RESOLUTION_NOTE = (
    "  [FST VI, point-of-use note] Since decision A3 (2026-09-22) this cell "
    "no longer sits on the saturated boundary of the ITA -> melanin-index "
    "mapping -- the clip was released, so the mapping now varies within "
    "FST VI rather than assigning every dark reading an identical penalty. "
    "What that resolving power rests on: a single, digitised-from-a-box-plot "
    "source for the mapping's new lower anchor (Linde 2020), with its upper "
    "anchor and denominator still unsourced, and a calibration set that "
    "still contains no non-lamp (real-sunlight) FST VI vitamin-D datapoint "
    "-- see vitd_model.py's M_baseline_by_FST and K_const docstring entries. "
    "Treat FST VI absolute numbers as resting on an extrapolated, "
    "sparsely-evidenced anchor, not as a model limitation that has closed."
)


# =============================================================================
# [decisions B1, A1, A2 -- 2026-09-23, "Phase 2b"] Band/threshold helpers.
#
# Every number these produce is computed against the LIVE ModelParams() at
# report time -- none of the K_const/A/SC_max/M_baseline values below are
# copied from a prior run. Only two things are held as data rather than
# computed: the Narea-Jimenez et al. 2026 ITA mean/SD-by-FST pair (a
# published SOURCE STATISTIC, not a model parameter -- see vitd_model.py
# docstring ~line 807-809), and the range endpoint read live from
# vitd_model.SC_MAX_IV_VI_RANGE (imported, not retyped).
# =============================================================================

# Narea-Jimenez et al. 2026, Biomed Phys Eng Express 12:025078: constitutive
# ITA mean +/- SD by FST (vitd_model.py docstring ~line 807-809).
NAREA_ITA_MEAN_BY_FST = {1: 51.76, 2: 40.42, 3: 27.54, 4: 10.45, 5: -9.08, 6: -33.63}
NAREA_ITA_SD_BY_FST = {1: 6.40, 2: 9.77, 3: 10.67, 4: 9.71, 5: 9.98, 6: 15.91}


def m_baseline_sd_band(params: Optional[ModelParams] = None):
    """+/-1 SD band on M_baseline_by_FST, from Narea's own ITA dispersion.

    The ITA -> M mapping's slope is fitted LIVE by linear regression against
    the CURRENT `params.M_baseline_by_FST` (never against a hardcoded
    formula or denominator), so this tracks vitd_model.py automatically if
    the mapping is ever revised again. Returns (band, r2) where band is
    {fst: {"mean", "lighter", "darker", "shift"}} and r2 is the linear fit's
    R^2 -- expected close to 1.0 since the mapping is affine by construction;
    printed so a future reader can see if that stops being true.
    """
    p = params or ModelParams()
    m0 = p.M_baseline_by_FST
    fsts_sorted = sorted(m0)
    itas = np.array([NAREA_ITA_MEAN_BY_FST[f] for f in fsts_sorted])
    ms = np.array([m0[f] for f in fsts_sorted])
    slope, intercept = np.polyfit(itas, ms, 1)
    fit = slope * itas + intercept
    ss_res = float(np.sum((ms - fit) ** 2))
    ss_tot = float(np.sum((ms - ms.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    band = {}
    for f in fsts_sorted:
        shift = abs(float(slope)) * NAREA_ITA_SD_BY_FST[f]
        band[f] = {"mean": m0[f], "lighter": m0[f] - shift,
                   "darker": m0[f] + shift, "shift": shift}
    return band, r2


def _minutes_row(fst, session_budgets, params, m_override=None, sc_override=None):
    """One `minimum_schedule_for_sufficiency` call, optionally overriding one
    FST's M_baseline or SC_max cell via `dataclasses.replace` (never mutating
    the live ModelParams default)."""
    trial = params
    if m_override is not None:
        overridden = dict(trial.M_baseline_by_FST)
        overridden[fst] = m_override
        trial = dataclasses.replace(trial, M_baseline_by_FST=overridden)
    if sc_override is not None:
        overridden = dict(trial.SC_max_by_FST)
        overridden[fst] = sc_override
        trial = dataclasses.replace(trial, SC_max_by_FST=overridden)
    return minimum_schedule_for_sufficiency(
        fst=fst, session_budgets=session_budgets, params=trial)


def sc_max_iv_vi_band(fsts=(4, 5, 6), session_budgets=TRI_SESSION_BUDGETS,
                      params: Optional[ModelParams] = None):
    """N_min / minutes at both ends of the carried SC_max_by_FST IV-VI range.

    Reads the range from `vitd_model.SC_MAX_IV_VI_RANGE` (live import, not
    retyped) rather than assuming which end is currently shipped, so this
    stays correct even if the shipped value moves inside that range.
    """
    p = params or ModelParams()
    lo, hi = SC_MAX_IV_VI_RANGE
    out = {}
    for fst in fsts:
        shipped = p.SC_max_by_FST[fst]
        row_lo = _minutes_row(fst, session_budgets, p, sc_override=lo)
        row_hi = _minutes_row(fst, session_budgets, p, sc_override=hi)
        out[fst] = {"shipped": shipped, "lo": lo, "hi": hi,
                   "row_lo": row_lo, "row_hi": row_hi}
    return out


def a_threshold_for_fst(fst, n_sessions, params: Optional[ModelParams] = None,
                        tol=1e-6, max_iter=80):
    """Smallest `A` at which `fst` clears the session-length ceiling at
    `n_sessions`/yr, every other shipped parameter held fixed.

    Bisects on the model's own `feasible` flag (monotone increasing in A
    over the relevant range -- verified directly, not assumed: see the
    Phase-2b report). Returns None if no bracket up to a generous multiple
    of the shipped A is found (i.e. even a large A does not rescue it).
    """
    p = params or ModelParams()

    def feasible_at(a_val):
        trial = dataclasses.replace(p, A=a_val)
        r = minimum_schedule_for_sufficiency(
            fst=fst, session_budgets=(n_sessions,), params=trial)
        return bool(r["rows"][0]["feasible"])

    lo = p.A
    if feasible_at(lo):
        return lo
    hi = lo * 1.5
    tries = 0
    while not feasible_at(hi) and tries < 30:
        hi *= 1.3
        tries += 1
    if not feasible_at(hi):
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if feasible_at(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def analysis_objective_hierarchy(params: Optional[ModelParams] = None,
                                 fsts: Sequence[int] = (1, 2, 3, 4, 5, 6),
                                 session_budgets: Sequence[int] = TRI_SESSION_BUDGETS):
    """PRIMARY objective, plus the diagnostic tier that must not be mistaken for it.

    Prints the hierarchy first so that every number in every later section has
    a declared tier before the reader meets it.
    """
    from vitd_objectives import (format_objective_hierarchy, primary_objective,
                                 assert_reportable_as_headline)

    print(format_objective_hierarchy())

    spec = primary_objective()
    assert_reportable_as_headline(spec.key)

    print("\n" + "=" * 74)
    print("PRIMARY RESULT: least you have to do to reach AND hold sufficiency")
    print("=" * 74)
    print(f"\n  {spec.question}")
    print("\n  Constraints, none relaxed:")
    for c in spec.subject_to:
        print(f"    {c}")
    print(f"\n  Reference conditions for the minutes column: UV index 6.0, "
          f"{PRACTICAL_MAX_MINUTES:.0f} min ceiling,")
    print(f"  geometry factor {PERSONAL_DOSE_GEOMETRY_FACTOR} "
          f"(vitd_objectives.PERSONAL_DOSE_GEOMETRY_FACTOR).")
    if 6 in fsts:
        print()
        print(FST_VI_RESOLUTION_NOTE)

    # Two spacing regimes, reported side by side. Uniform is the plain
    # "every N days" schedule and the condition under which
    # TestRecalibratedFeasibility and scripts/feasibility_map_kconst.py state
    # their minimum budgets; free spacing is the true minimum of the primary
    # objective, because nothing in the problem statement requires even gaps.
    rows = []
    summary = {}
    for label, gammas in (("uniform", (1.0,)), ("free", TRI_GAMMA_GRID)):
        summary[label] = []
        for fst in fsts:
            res = minimum_schedule_for_sufficiency(
                fst=fst, session_budgets=session_budgets, gammas=gammas,
                params=params)
            for r in res["rows"]:
                r["spacing"] = label
            rows.extend(res["rows"])
            summary[label].append(res)

    df = pd.DataFrame(rows)
    df.to_csv("tables/primary_minimum_schedule.csv", index=False)

    print(f"\n  Full grid, UNIFORM spacing. INFEASIBLE cells are results, not gaps.\n")
    print(f"  {'FST':>4} {'N/yr':>5} {'SED/sesh':>9} {'min/sesh':>9} "
          f"{'totSED':>8} {'t->suff':>8} {'risk':>9}  status")
    print("  " + "-" * 78)
    for res in summary["uniform"]:
        for r in res["rows"]:
            if r["feasible"]:
                ttf = ("n/a" if not np.isfinite(r["time_to_sufficiency"])
                       else f"{r['time_to_sufficiency']:.0f}d")
                print(f"  {ROMAN_FST[r['fst']]:>4} {r['n_sessions']:>5} "
                      f"{r['sed_per_session']:>9.3f} {r['minutes_per_session']:>9.1f} "
                      f"{r['total_sed']:>8.1f} {ttf:>8} "
                      f"{r['combined_risk']:>9.4f}  ok")
            else:
                print(f"  {ROMAN_FST[r['fst']]:>4} {r['n_sessions']:>5} "
                      f"{'--':>9} {'--':>9} {'--':>8} {'--':>8} {'--':>9}  "
                      f"INFEASIBLE: {r['binding_constraint']}")
        print()

    print("  HEADLINE. Minimum session budget that reaches and holds 50 nmol/L,")
    print("  under uniform spacing and then with the spacing free:")
    print(f"  {'FST':>4} {'uniform spacing':>26} {'free spacing':>30} {'gamma':>7}")
    print("  " + "-" * 72)
    rescued = []
    for u, f_ in zip(summary["uniform"], summary["free"]):
        fst = u["fst"]
        ua = ("INFEASIBLE on this grid" if not u["feasible"] else
              "{}/yr, {:.0f} min, {:.0f} SED".format(
                  u["fewest_sessions"]["n_sessions"],
                  u["fewest_sessions"]["minutes_per_session"],
                  u["fewest_sessions"]["total_sed"]))
        if not f_["feasible"]:
            fa, g = "INFEASIBLE on this grid", float("nan")
        else:
            fw = f_["fewest_sessions"]
            fa = "{}/yr, {:.0f} min, {:.0f} SED".format(
                fw["n_sessions"], fw["minutes_per_session"], fw["total_sed"])
            g = fw["gamma"]
            if u["feasible"] and fw["n_sessions"] < u["fewest_sessions"]["n_sessions"]:
                rescued.append((fst, u["fewest_sessions"]["n_sessions"],
                                fw["n_sessions"], g))
        gs = "n/a" if not np.isfinite(g) else f"{g:.2f}"
        print(f"  {ROMAN_FST[fst]:>4} {ua:>26} {fa:>30} {gs:>7}")

    print("\n  Least TOTAL UV (a different question from fewest sessions):")
    print(f"  {'FST':>4} {'uniform spacing':>26} {'free spacing':>30}")
    print("  " + "-" * 64)
    for u, f_ in zip(summary["uniform"], summary["free"]):
        ua = ("--" if not u["feasible"] else
              "{}/yr, {:.0f} SED total".format(u["least_total_dose"]["n_sessions"],
                                               u["least_total_dose"]["total_sed"]))
        fa = ("--" if not f_["feasible"] else
              "{}/yr, {:.0f} SED total".format(f_["least_total_dose"]["n_sessions"],
                                               f_["least_total_dose"]["total_sed"]))
        print(f"  {ROMAN_FST[u['fst']]:>4} {ua:>26} {fa:>30}")

    u_df = df[df["spacing"] == "uniform"]
    print(f"\n  {int((~u_df['feasible']).sum())} of {len(u_df)} (skin type, session "
          f"budget) cells are infeasible under uniform spacing;")
    f_df = df[df["spacing"] == "free"]
    print(f"  {int((~f_df['feasible']).sum())} of {len(f_df)} with the spacing free.")
    for res in summary["free"]:
        if res["note"]:
            print(f"    {res['note']}")

    if rescued:
        print("\n  SPACING CHANGES THE MINIMUM BUDGET -- this is a result, not a")
        print("  discrepancy with the uniform-spacing figures asserted in")
        print("  tests/test_vitd_optimization.py::TestRecalibratedFeasibility.")
        print("  Constraint C3 binds at the RAMP-IN BOUNDARY, not at the end of")
        print("  the year: under uniform spacing 25(OH)D is still climbing at")
        print("  day 90 and misses the floor by a fraction of a nmol/L, whereas")
        print("  a front-loaded schedule is already past it. Same budget, same")
        print("  ceiling, nothing relaxed -- only the order of the sessions.")
        for fst, nu, nf, g in rescued:
            print(f"    FST {ROMAN_FST[fst]:<4} {nu}/yr uniform -> {nf}/yr at "
                  f"gamma {g:.2f}")

    print("\n  Fewest-sessions and least-total-UV are DIFFERENT answers. Fewer")
    print("  sessions forces a larger per-session dose, and per-session yield")
    print("  saturates, so the fewest-sessions schedule generally costs more")
    print("  cumulative UV. Which one is 'the' minimum depends on whether the")
    print("  binding cost is getting outside or the UV itself.")
    print("\n  Saved tables/primary_minimum_schedule.csv")

    # ---- B1/A1/A2 (Phase 2b, 2026-09-23): FST V/VI as a BAND, not a point,
    # plus the carried SC_max_by_FST IV-VI range and the A-threshold for FST
    # V's twice-weekly exclusion. Everything below is computed live against
    # `params` (or ModelParams() if None) at report time -- nothing here is
    # copied from a prior run or a scratchpad script. Only FST V/VI are
    # affected (M_baseline_by_FST's dispersion, SC_max_by_FST's IV-VI cell),
    # so this is skipped entirely if neither is in the requested `fsts`.
    band_fsts = tuple(f for f in (5, 6) if f in fsts)
    if band_fsts:
        base_params = params or ModelParams()
        m_band, m_r2 = m_baseline_sd_band(base_params)
        sc_band = sc_max_iv_vi_band(fsts=band_fsts, session_budgets=session_budgets,
                                    params=base_params)
        print("\n" + "=" * 74)
        print("FST V / VI reported as a BAND, not a point (decisions B1, A1)")
        print("=" * 74)
        print("\n  B1: +/-1 SD on M_baseline_by_FST, from Narea-Jimenez et al.")
        print(f"  2026's own ITA dispersion, mapped through a LIVE linear fit "
              f"to the current mapping (R^2 = {m_r2:.6f} -- the mapping is "
              f"affine by construction, so this should stay ~1.0). "
              f"'lighter'/'darker' bracket the shipped mean:")
        for fst in band_fsts:
            b = m_band[fst]
            mean_rows = [r for r in summary["uniform"] if r["fst"] == fst][0]["rows"]
            row_lo = _minutes_row(fst, session_budgets, base_params,
                                  m_override=b["lighter"])["rows"]
            row_hi = _minutes_row(fst, session_budgets, base_params,
                                  m_override=b["darker"])["rows"]
            print(f"\n  FST {ROMAN_FST[fst]}   M = {b['lighter']:.6f} (lighter) / "
                  f"{b['mean']:.6f} (shipped) / {b['darker']:.6f} (darker)")
            print(f"  {'N/yr':>5} {'min@lighter':>12} {'min@shipped':>12} "
                  f"{'min@darker':>11}")

            def _fmt_min(r):
                return (f"{r['minutes_per_session']:.1f}" if r["feasible"]
                        else "infeasible")

            for rl, rm, rh in zip(row_lo, mean_rows, row_hi):
                print(f"  {rl['n_sessions']:>5} {_fmt_min(rl):>12} "
                      f"{_fmt_min(rm):>12} {_fmt_min(rh):>11}")
            n_min_lo = next((r["n_sessions"] for r in row_lo if r["feasible"]), None)
            n_min_mean = next((r["n_sessions"] for r in mean_rows if r["feasible"]), None)
            n_min_hi = next((r["n_sessions"] for r in row_hi if r["feasible"]), None)
            print(f"  N_min: {n_min_lo} (lighter) .. {n_min_mean} (shipped) .. "
                  f"{n_min_hi} (darker)")
        print(f"\n  A1: carried SC_max_by_FST IV-VI range "
              f"[{SC_MAX_IV_VI_RANGE[0]:g}, {SC_MAX_IV_VI_RANGE[1]:g}] "
              f"(vitd_model.SC_MAX_IV_VI_RANGE, read live; the shipped value "
              f"is the low end):")
        for fst in band_fsts:
            d = sc_band[fst]
            n_min_lo = next((r["n_sessions"] for r in d["row_lo"]["rows"]
                             if r["feasible"]), None)
            n_min_hi = next((r["n_sessions"] for r in d["row_hi"]["rows"]
                             if r["feasible"]), None)
            print(f"    FST {ROMAN_FST[fst]}   N_min at SC={d['lo']:g} (shipped): "
                  f"{n_min_lo}    N_min at SC={d['hi']:g}: {n_min_hi}")
        if 5 in band_fsts:
            n104 = 104 if 104 in session_budgets else session_budgets[0]
            thr = a_threshold_for_fst(5, n104, base_params)
            print(f"\n  A2: FST V's N={n104}/yr exclusion, A-threshold "
                  f"(point anchor kept; A_by_FST rejected as unidentifiable):")
            if thr is None:
                print(f"    no A up to a wide bracket rescues N={n104} for FST V "
                      f"at this session ceiling")
            else:
                headroom = thr / base_params.A
                print(f"    A >= {thr:.6f} clears the {PRACTICAL_MAX_MINUTES:.0f}-min "
                      f"ceiling (shipped A = {base_params.A:.6f}); shipped A is "
                      f"{100 * (1 - base_params.A / thr):.4f}% below that threshold, "
                      f"i.e. the threshold is {headroom:.6f}x shipped.")

    # ---- the diagnostic tier, stated as such --------------------------------
    diag = gross_synth_per_SED_diagnostic(fst=3, params=params)
    print("\n" + "=" * 74)
    print("BACKGROUND / DIAGNOSTIC TIER: gross_synth_per_SED")
    print("=" * 74)
    print("\n  Reported here so it is on the record, and labelled so it cannot")
    print("  be read as advice. Both failures are RECOMPUTED from the current")
    print("  parameters on this run, not quoted from a previous one.")
    print(f"\n  1. No interior optimum. argmax by search bound: "
          f"{diag['argmax_by_search_bound']}")
    print(f"     argmax tracks the search bound: "
          f"{diag['argmax_tracks_search_bound']}")
    print(f"     Spearman rho(interval, metric) = "
          f"{diag['spearman_rho_interval_vs_metric']:+.4f} over "
          f"{diag['metric_range'][0]:.3f}-{diag['metric_range'][1]:.3f}; "
          f"{diag['n_monotonicity_violations']} local dips, largest "
          f"{diag['largest_violation']:.4f}")
    print("     (the dips are integer-grid quantisation -- 365/interval is not")
    print("     a whole number of sessions -- not an interior peak)")
    print(f"\n  2. Net loss at its own optimum. At {diag['selected_interval_days']} d "
          f"(FST III, 1 SED/session):")
    print(f"     start 25(OH)D {diag['D0']:.1f} -> final "
          f"{diag['final_D_at_selected']:.1f} nmol/L "
          f"(net {diag['net_D_change_at_selected']:+.1f})")
    print(f"     ends BELOW baseline: {diag['net_loss_at_selected']}")
    print("\n  A metric maximised by dosing less, whose selected schedule leaves")
    print("  the person worse off than when they started, is a statement about")
    print("  the model, not a schedule. It stays DIAGNOSTIC-tier.")
    return df


def analysis_tri_objective(params: Optional[ModelParams] = None,
                           fsts: Sequence[int] = (1, 2, 3, 4, 5, 6)):
    """SECONDARY: fewest sessions x least spacing x least skin-cancer risk."""
    from vitd_objectives import OBJECTIVE_HIERARCHY

    spec = next(s for s in OBJECTIVE_HIERARCHY
                if s.key == "tri_objective_sessions_spacing_risk")

    print("\n" + "=" * 74)
    print("SECONDARY: tri-objective frontier -- sessions x spacing x cancer risk")
    print("=" * 74)
    print(f"\n  {spec.question}")
    print("\n  Three axes, all minimised, over schedules that ALREADY hold")
    print("  sufficiency (so efficacy is a constraint here, not an axis, and")
    print("  the total-SED column is what that constraint cost):")
    print("    A1  n_sessions     how often you must get outside")
    print("    A2  max_gap_days   longest stretch with no session (tail gap")
    print("                       to the end of the horizon included)")
    print("    A3  risk          the skin-cancer index, WITHIN skin type only")
    print("\n  Subject to: " + "; ".join(spec.subject_to))
    print("\n  A3 is reported as a ratio to THIS skin type's own lowest-risk")
    print("  feasible cell, so it reads 1.0000 at every skin type's minimum by")
    print("  construction. That is deliberate: the FST susceptibility factor is")
    print("  a constitutive prior, and cross-skin-type comparisons of risk")
    print("  VALUES are not anchored by any source in this repo (Elwood &")
    print("  Jopson 1997 adjusts its odds ratios FOR pigmentation, so it cannot")
    print("  supply the contrast). The axis is")
    print("  carried as a sealed vitd_optimization.WithinFSTRisk object that")
    print("  refuses float(), arithmetic and cross-FST ordering, so no absolute")
    print("  risk value can reach this table or its CSV at all.")
    print("\n  " + SESSION_BUDGET_BASIS)
    print(f"\n  Spacing is the same one-parameter monotone warp as section 13c")
    print(f"  (gamma = 1 is uniform); grid {TRI_GAMMA_GRID}.")

    all_rows = []
    for fst in fsts:
        res = tri_objective_frontier(fst=fst, params=params)
        all_rows.extend(res["rows"])

        print(f"\n  FST {ROMAN_FST[fst]}")
        if fst == 6:
            print(FST_VI_RESOLUTION_NOTE)
        print(f"  {'N/yr':>5} {'gamma':>6} {'maxgap':>7} {'risk/min':>9} "
              f"{'totSED':>8} {'peakSED':>8} {'min/sesh':>9} {'t->suff':>8}  front")
        print("  " + "-" * 78)
        for r in res["rows"]:
            if not r["feasible"]:
                print(f"  {r['n_sessions']:>5} {r['gamma']:>6.2f} "
                      f"{r['max_gap_days']:>7.1f} {'--':>9} {'--':>8} {'--':>8} "
                      f"{'--':>9} {'--':>8}  INFEASIBLE")
                continue
            ttf = ("n/a" if not np.isfinite(r["time_to_sufficiency"])
                   else f"{r['time_to_sufficiency']:.0f}d")
            print(f"  {r['n_sessions']:>5} {r['gamma']:>6.2f} "
                  f"{r['max_gap_days']:>7.1f} "
                  f"{r['risk_within_fst_ratio']:>9.4f} "
                  f"{r['total_sed']:>8.1f} {r['peak_sed']:>8.3f} "
                  f"{r['minutes_per_session']:>9.1f} {ttf:>8}  "
                  f"{'YES' if r['on_frontier'] else ''}")

        if not res["feasible"]:
            print(f"    No (budget, spacing) cell is feasible for FST "
                  f"{ROMAN_FST[fst]} on this grid. Reported, not dropped.")
            continue
        print(f"    Nondominated: {len(res['frontier'])} of "
              f"{sum(1 for r in res['rows'] if r['feasible'])} feasible cells.")

    # The sealed A3 axis never reaches the DataFrame. Dropping it here is
    # belt-and-braces -- WithinFSTRisk refuses float() and numeric formatting,
    # so pandas could not have written an absolute risk value as a number
    # anyway -- but it keeps the CSV's columns honest about what they are.
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "combined_risk"}
                       for r in all_rows])
    df["risk_axis"] = "within_fst_ratio_only"
    df["session_budget_basis"] = "adherence_assumption"
    df.to_csv("tables/tri_objective_frontier.csv", index=False)

    feas = df[df["feasible"]]
    print("\n  Reading the frontier. Uniform spacing (gamma = 1) minimises the")
    print("  largest gap at any budget, so it owns the A2 end of every")
    print("  frontier; front-loading (gamma > 1) buys risk by lengthening the")
    print("  late gaps, so it owns the A3 end. Nothing dominates on both, which")
    print("  is why this is reported as a frontier and not as a recommendation.")
    if len(feas):
        n_front = int(feas["on_frontier"].sum())
        print(f"\n  {n_front} nondominated points across {len(feas)} feasible")
        print(f"  cells and {int((~df['feasible']).sum())} infeasible ones.")

    # Spacing is not only an optimisation knob here -- at some budgets it
    # decides feasibility outright. Counted rather than asserted.
    rescues = []
    for fst in fsts:
        sub = df[df["fst"] == fst]
        for n in sorted(sub["n_sessions"].unique()):
            cell = sub[sub["n_sessions"] == n]
            uni = cell[np.isclose(cell["gamma"], 1.0)]
            if len(uni) and not bool(uni["feasible"].iloc[0]) \
                    and bool(cell["feasible"].any()):
                got = cell[cell["feasible"].astype(bool)]["gamma"].tolist()
                rescues.append((fst, int(n), got))
    if rescues:
        print("\n  Budgets where UNIFORM spacing is infeasible but some warped")
        print("  spacing is not. C3 binds at the ramp-in boundary, so reaching")
        print("  target faster can rescue a budget uniform spacing misses:")
        for fst, n, got in rescues:
            print(f"    FST {ROMAN_FST[fst]:<4} N = {n:<5} feasible at gamma "
                  f"{', '.join(f'{g:.2f}' for g in got)}")
    print("\n  Saved tables/tri_objective_frontier.csv")
    return df

def analysis_constant_sessions(params: Optional[ModelParams] = None):
    """Sweep session counts 1-52 over 365 days with fixed total SED, uniform spacing.

    For each session count, evaluates max-D, AUC deficiency, and net D change
    per SED. Flags schedules with rolling 7d SED > 5 as requiring UV safety
    review.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS L: constant-sessions optimization sweep")
    print("=" * 70)

    total_SED = 52.0
    duration_days = 365
    fst = 5
    sed_per_session_default = 1.0
    if params is None:
        params = ModelParams()
        from vitd_model import get_E_source
        params.E_source = get_E_source("solar_summer")

    print(f"\n  Using E_source = {params.E_source:.2f} for constant-sessions sweep.")

    rows = []
    for N_sessions in range(1, 53):
        sed = total_SED / N_sessions
        sched = schedule_uniform(N_sessions, duration_days, sed)
        m = evaluate_schedule(sched, fst, duration_days, params=params)
        m["N_sessions"] = N_sessions
        m["sed_per_session"] = sed
        m["interval_days"] = duration_days / N_sessions

        # UV safety flag
        rolling_7d = m["max_rolling_7d_SED"]
        if rolling_7d > 5.0:
            m["uv_safety_flag"] = "UV safety: review"
        else:
            m["uv_safety_flag"] = ""

        rows.append(m)

    df = pd.DataFrame(rows)

    df.to_csv("tables/constant_sessions_analysis.csv", index=False)
    print(f"\n  Saved tables/constant_sessions_analysis.csv")

    best = df.loc[df["auc_deficit_50"].idxmin()]
    print(f"\n  Optimal schedule: {int(best['N_sessions'])} sessions")
    print(f"    SED/session:     {best['sed_per_session']:.3f}")
    print(f"    Interval:        {best['interval_days']:.1f} days")
    print(f"    Max D:           {best['max_D']:.1f} nmol/L")
    print(f"    AUC deficit @50: {best['auc_deficit_50']:.1f} nmol/L·days")
    print(f"    Net ΔD/SED:      {best['net_final_D_change_per_SED']:.3f} nmol/L/SED")

    print(f"\n  {'N':>3s} {'SED/sesh':>9s} {'Interval':>9s} {'Max D':>7s} "
          f"{'Deficit@50':>11s} {'NetΔD/SED':>10s} "
          f"{'Roll7d':>7s} {'Flag':>20s}")
    print(f"  {'-'*3} {'-'*9} {'-'*9} {'-'*7} {'-'*11} {'-'*10} "
          f"{'-'*7} {'-'*20}")
    for _, row in df.iterrows():
        print(f"  {int(row['N_sessions']):3d} {row['sed_per_session']:>8.3f} "
              f"{row['interval_days']:>8.1f}  {row['max_D']:>6.1f}  "
              f"{row['auc_deficit_50']:>10.1f}  "
              f"{row['net_final_D_change_per_SED']:>9.3f}  "
              f"{row['max_rolling_7d_SED']:>6.1f}  "
              f"{row['uv_safety_flag']:>20s}")

    return df


def analysis_spacing_fixed_dose(params: Optional[ModelParams] = None,
                                fsts: Sequence[int] = (1, 2, 3, 4, 5, 6),
                                session_budgets: Sequence[int] = (52, 104, 146, 183, 260, 365),
                                sed_per_session: float = 1.0,
                                horizon_days: int = 365,
                                objective: str = "mean_D") -> pd.DataFrame:
    """Best spacing for a fixed dose per session, across skin types and budgets.

    Answers 'when should I go outside?' for a person who will use the same
    dose each time. Ranks by `mean_D` (maximise average 25(OH)D).
    """
    import os
    from vitd_objectives import optimal_spacing_fixed_sessions

    print("\n" + "=" * 70)
    print("ANALYSIS 13a: spacing optimisation at fixed dose")
    print("=" * 70)
    print(f"\n  Fixed dose: {sed_per_session:.2f} SED/session, "
          f"horizon {horizon_days}d, objective: {objective}")
    print("  Sweeping gap days for each (skin type, session budget) cell.\n")
    if 6 in fsts:
        print(FST_VI_RESOLUTION_NOTE)
        print()

    os.makedirs("tables", exist_ok=True)

    rows = []
    for fst in fsts:
        for n in session_budgets:
            res = optimal_spacing_fixed_sessions(
                fst=fst, n_sessions=n, sed_per_session=sed_per_session,
                horizon_days=horizon_days, params=params, objective=objective)
            best = res["best"]
            if best is None:
                rows.append({
                    "fst": fst, "n_sessions": n, "feasible": False,
                    "gap_days": float("nan"), "mean_D": float("nan"),
                    "final_D": float("nan"), "min_D": float("nan"),
                    "days_below_50": float("nan"),
                    "scc_index": float("nan"),
                    "intermittent_index": float("nan"),
                    "burn_fraction": float("nan"),
                })
                continue
            rows.append({
                "fst": fst, "n_sessions": n, "feasible": True,
                "gap_days": best["gap_days"],
                "mean_D": best["mean_D"],
                "final_D": best["final_D"],
                "min_D": best["min_D"],
                "days_below_50": best["days_below_50"],
                "scc_index": best["scc_index"],
                "intermittent_index": best["intermittent_index"],
                "burn_fraction": best["burn_fraction"],
            })

    df = pd.DataFrame(rows)
    df.to_csv("tables/spacing_fixed_dose.csv", index=False)
    print(f"  Saved tables/spacing_fixed_dose.csv")

    print(f"\n  {'FST':>4} {'N/yr':>5} {'gap':>5} {'meanD':>7} {'finalD':>7} "
          f"{'minD':>7} {'<50d':>5} {'SCC':>7} {'INT':>7} {'burn':>7}")
    print("  " + "-" * 67)
    for fst in fsts:
        sub = df[df["fst"] == fst]
        for _, r in sub.iterrows():
            if not r["feasible"]:
                print(f"  {ROMAN_FST[int(r['fst'])]:>4} {int(r['n_sessions']):>5} "
                      f"{'--':>5} {'--':>7} {'--':>7} {'--':>7} {'--':>5} "
                      f"{'--':>7} {'--':>7} {'--':>7}")
                continue
            print(f"  {ROMAN_FST[int(r['fst'])]:>4} {int(r['n_sessions']):>5} "
                  f"{r['gap_days']:>5.0f} {r['mean_D']:>7.1f} {r['final_D']:>7.1f} "
                  f"{r['min_D']:>7.1f} {r['days_below_50']:>5.0f} "
                  f"{r['scc_index']:>7.3f} {r['intermittent_index']:>7.3f} "
                  f"{r['burn_fraction']:>7.3f}")
        if len(sub):
            print()

    print("  Spacing is uniform (gap days). Front-loaded and back-loaded")
    print("  arrangements are checked by optimal_spacing_fixed_sessions but")
    print("  uniform spacing minimises the largest gap, so it dominates at")
    print("  fixed dose and session count.\n")

    return df


if __name__ == "__main__":
    analysis_constant_sessions()
