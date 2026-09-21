"""
feasibility_map_kconst.py
=========================
CONSEQUENCE MAP for a calibration change, expressed as scheduling feasibility.

Written for the 2026-08-17 K_const recalibration (0.33 -> 0.98). Since
2026-08-18 it sweeps an arbitrary SET of parameters, because the K_const result
turned out not to be separable: raising `A` (approved item A4) moves feasibility
in the OPPOSITE direction, so the honest map is the COMBINED one. The
single-parameter behaviour is unchanged and is still the default.

Question
--------
Under the standing scheduling constraints, which (skin type, session budget)
combinations can still hold the sufficiency floor for a whole year on sun
alone -- and where a combination CANNOT, which constraint stopped it?

The constraints are read from the code, not restated here:
    C3  floor      every day at/after `ramp_in_days` must be >= SUFFICIENCY_NMOL_L
    C1  erythema   per-session dose <= GAMMA_MED * MED(FST)
    C7  time       per-session dose <= what `max_minutes` delivers at `uv_index`
`GAMMA_MED`, `SUFFICIENCY_NMOL_L` and `RAMP_IN_DAYS` come from vitd_objectives;
`_med_sed_by_fst` from vitd_optimization. NOTHING in this script relaxes any
of them -- it measures where they bind.

Method
------
For each (FST, N):
  1. Ask the optimiser for the minimum uniform dose that holds the floor at
     the erythema ceiling (`optimise_variable_dose(family='uniform')`). None
     means the erythema cap C1 binds: even the largest non-burning dose,
     delivered N times, cannot hold the floor.
  2. Where it is feasible, convert the required per-session SED to minutes at
     a reference UV index and test it against the session-length ceiling C7.
  3. Report the same map at the BEFORE parameter set so the change is
     attributable to the recalibration and not to anything else in the model.

BEFORE sets
-----------
    kconst      K_const = 0.33                     (default; the 2026-08-17 map)
    combined    K_const = 0.33, A = 2.353          (K_const + A together)
    a_only      A = 2.353                          (isolates A)
    phase34     K_const = 0.33, A = 2.353, feedback_mode = 'linear'
                                                   (everything that has moved)

Run
---
    PYTHONIOENCODING=utf-8 python3 scripts/feasibility_map_kconst.py
    PYTHONIOENCODING=utf-8 python3 scripts/feasibility_map_kconst.py --before combined
    PYTHONIOENCODING=utf-8 python3 scripts/feasibility_map_kconst.py --before A=2.353
    PYTHONIOENCODING=utf-8 python3 scripts/feasibility_map_kconst.py --factorial
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams
from vitd_objectives import (SUFFICIENCY_NMOL_L, GAMMA_MED, RAMP_IN_DAYS,
                             sed_from_minutes)
from vitd_optimization import optimise_variable_dose, _med_sed_by_fst

OLD_K = 0.33
OLD_A = 2.353
OLD_FEEDBACK = "linear"
HORIZON = 365
# 52 = weekly, 104 = twice weekly, 146 = every 2.5 d, 183 = alternate days,
# 260 = five per week, 365 = daily. 52 was added 2026-08-18 so that a minimum
# of 104 can be pinned from BELOW as well as from above.
N_GRID = (52, 104, 146, 183, 260, 365)
# Reference conditions for the minutes conversion. UV index 6 and a
# 120-minute ceiling are the values tests/test_vitd_optimization.py and
# optimal_interval_risk_constrained already use; they are held fixed here so
# the map isolates the calibration change.
UV_INDEX = 6.0
MAX_MINUTES = 120.0
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

# Named BEFORE sets. Each is a dict of ModelParams attribute -> value applied
# on top of a default ModelParams().
BEFORE_SETS = {
    "kconst": {"K_const": OLD_K},
    "combined": {"K_const": OLD_K, "A": OLD_A},
    "a_only": {"A": OLD_A},
    "phase34": {"K_const": OLD_K, "A": OLD_A, "feedback_mode": OLD_FEEDBACK},
}


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def make_params(**overrides):
    """Default ModelParams with `overrides` applied.

    Casts values to the type the dataclass default already has, so
    `--before K_const=0.33,A=2.353` and `--before feedback_mode=linear` both
    work without the caller quoting types.
    """
    p = ModelParams()
    for name, value in overrides.items():
        if not hasattr(p, name):
            raise SystemExit(f"ModelParams has no attribute {name!r}")
        current = getattr(p, name)
        if isinstance(current, float) and not isinstance(value, str):
            value = float(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(p, name, value)
    return p


def describe(overrides):
    """'K_const 0.33 -> 0.98, A 2.353 -> 4.125' for a header line (values read live)."""
    now = ModelParams()
    bits = []
    for name, value in overrides.items():
        after = getattr(now, name)
        if isinstance(after, float):
            bits.append(f"{name} {float(value):g} -> {after:g}")
        else:
            bits.append(f"{name} {value} -> {after}")
    return ", ".join(bits) if bits else "(no overrides -- BEFORE == AFTER)"


def probe(fst, n, params):
    """Return (feasible_under_C1_and_C7, required_sed, required_minutes, binding)."""
    res = optimise_variable_dose(fst=fst, n_sessions=n, family="uniform",
                                 horizon_days=HORIZON, ramp_in_days=RAMP_IN_DAYS,
                                 params=params)
    if res is None:
        return False, None, None, "C1 erythema cap (70% MED)"
    sed = res["max_sed"]
    minutes = sed / sed_from_minutes(1.0, UV_INDEX)
    if minutes > MAX_MINUTES:
        return False, sed, minutes, f"C7 session length (>{MAX_MINUTES:.0f} min)"
    return True, sed, minutes, "none - feasible"


def table(params, label, n_grid=N_GRID):
    hr(f"FEASIBILITY MAP -- {label}")
    print(f"\n  Floor {SUFFICIENCY_NMOL_L:.0f} nmol/L held every day from day "
          f"{RAMP_IN_DAYS} to {HORIZON}, uniform dosing, D0 = 25.0 nmol/L.")
    print(f"  Caps: {GAMMA_MED:.0%} MED per session; {MAX_MINUTES:.0f} min per "
          f"session at UV index {UV_INDEX:.0f}.")
    print(f"\n  {'FST':>4} {'70%MED cap':>11} " +
          "".join(f"{'N=' + str(n):>16}" for n in n_grid))
    out = {}
    for fst in range(1, 7):
        cap = GAMMA_MED * _med_sed_by_fst(fst)
        cells = []
        for n in n_grid:
            ok, sed, minutes, binding = probe(fst, n, params)
            out[(fst, n)] = (ok, sed, minutes, binding)
            if ok:
                cells.append(f"{sed:.2f} SED/{minutes:.0f}m")
            elif sed is None:
                cells.append("INFEASIBLE C1")
            else:
                cells.append(f"INFEAS C7 {minutes:.0f}m")
        print(f"  {ROMAN[fst]:>4} {cap:>11.2f} " +
              "".join(f"{c:>16}" for c in cells))
    return out


def minimum_budgets(cell_map, n_grid=N_GRID):
    """Smallest N on the grid that is feasible, per FST, and the N below it.

    Returns {fst: (n_min, n_below)}; n_min is None when no N on the grid
    works, n_below is None when n_min is the smallest grid point.
    """
    mins = {}
    for fst in range(1, 7):
        n_min = None
        for n in n_grid:
            if cell_map[(fst, n)][0]:
                n_min = n
                break
        if n_min is None:
            mins[fst] = (None, None)
            continue
        i = n_grid.index(n_min)
        mins[fst] = (n_min, n_grid[i - 1] if i > 0 else None)
    return mins


def factorial(cells=((2, 104), (3, 104), (4, 104), (5, 104), (6, 104))):
    """Full 2x2x2 attribution over the three parameters that have moved.

    K_ATTEN is NOT swept: it was 0.00 before its 2026-08-18 retirement and is
    0.00 after (`K_ATTEN_STRUCTURAL`), so item A6 is numerically inert here.
    """
    hr("ATTRIBUTION -- K_const x A x feedback_mode, full factorial")
    print(f"\n  {'FST':>4} {'N':>5} {'K_const':>8} {'A':>7} {'feedback':>10}"
          f" {'SED/sess':>9} {'min@UVI6':>9}  verdict")
    for fst, n in cells:
        for k, a, fb in itertools.product((OLD_K, ModelParams().K_const),
                                          (OLD_A, ModelParams().A),
                                          (OLD_FEEDBACK,
                                           ModelParams().feedback_mode)):
            p = make_params(K_const=k, A=a, feedback_mode=fb)
            ok, sed, minutes, binding = probe(fst, n, p)
            s1 = f"{sed:9.3f}" if sed is not None else f"{'-':>9}"
            s2 = f"{minutes:9.1f}" if minutes is not None else f"{'-':>9}"
            verdict = "feasible" if ok else binding
            print(f"  {ROMAN[fst]:>4} {n:>5} {k:>8.2f} {a:>7.3f} {fb:>10}"
                  f" {s1} {s2}  {verdict}")
        print()


def parse_before(spec):
    if spec in BEFORE_SETS:
        return dict(BEFORE_SETS[spec])
    out = {}
    for item in spec.split(","):
        if "=" not in item:
            raise SystemExit(f"cannot parse --before item {item!r}; "
                             f"use name=value or one of {sorted(BEFORE_SETS)}")
        name, _, value = item.partition("=")
        out[name.strip()] = value.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="kconst",
                    help="named BEFORE set (%s) or comma-separated "
                         "name=value overrides. Default: kconst, i.e. the "
                         "original single-parameter map."
                         % ", ".join(sorted(BEFORE_SETS)))
    ap.add_argument("--factorial", action="store_true",
                    help="also print the K_const x A x feedback_mode "
                         "attribution factorial at N=104")
    args = ap.parse_args(argv)

    overrides = parse_before(args.before)
    print(__doc__)

    new = ModelParams()
    old = make_params(**overrides)

    print(f"\n  BEFORE -> AFTER: {describe(overrides)}")
    print(f"  FST V:II optical factor: "
          f"{np.exp(old.K_const*(new.M_baseline(5)-new.M_baseline(2))):.2f}x  ->  "
          f"{np.exp(new.K_const*(new.M_baseline(5)-new.M_baseline(2))):.2f}x")

    before = table(old, f"BEFORE -- {describe(overrides)}")
    after = table(new, "AFTER -- current defaults")

    hr("WHAT MOVED")
    print("\n  Cells that were feasible before and are not now:")
    lost = [(f, n) for (f, n) in after
            if before[(f, n)][0] and not after[(f, n)][0]]
    if not lost:
        print("    none")
    for f, n in sorted(lost):
        print(f"    FST {ROMAN[f]:<3} N={n:<4} now blocked by "
              f"{after[(f, n)][3]}")
    gained = [(f, n) for (f, n) in after
              if after[(f, n)][0] and not before[(f, n)][0]]
    print("\n  Cells that were infeasible before and are feasible now:")
    print("    " + ("none" if not gained else
                    ", ".join(f"FST {ROMAN[f]} N={n}" for f, n in sorted(gained))))

    hr("MINIMUM FEASIBLE SESSION BUDGET, on the grid " + str(N_GRID))
    b_min, a_min = minimum_budgets(before), minimum_budgets(after)
    print(f"\n  {'FST':>4} {'before':>8} {'after':>8}   binding at the budget "
          f"below the minimum")
    for f in range(1, 7):
        nb, _ = b_min[f]
        na, below = a_min[f]
        why = "" if below is None else after[(f, below)][3]
        print(f"  {ROMAN[f]:>4} {str(nb):>8} {str(na):>8}   "
              f"{'N=' + str(below) + ': ' + why if below else 'lowest grid point'}")

    hr("REQUIRED MINUTES PER DAY AT UV INDEX %.0f, N=365 (daily)" % UV_INDEX)
    print(f"\n  {'FST':>4} {'before':>10} {'after':>10}")
    for f in range(1, 7):
        b, a = before[(f, 365)], after[(f, 365)]
        fb = f"{b[2]:.0f} min" if b[2] is not None else "INFEAS C1"
        fa = f"{a[2]:.0f} min" if a[2] is not None else "INFEAS C1"
        print(f"  {ROMAN[f]:>4} {fb:>10} {fa:>10}")

    print("\n  Dose inflation on the cells that survive, required SED per")
    print("  session, before -> after:")
    print(f"    {'FST':>4} {'N':>5} {'before':>8} {'after':>8} {'x':>6}")
    for f in range(1, 7):
        for n in N_GRID:
            b, a = before[(f, n)], after[(f, n)]
            if b[1] is not None and a[1] is not None:
                print(f"    {ROMAN[f]:>4} {n:>5} {b[1]:>8.3f} {a[1]:>8.3f} "
                      f"{a[1]/b[1]:>6.2f}")

    if args.factorial:
        factorial()

    hr("READING")
    print("""
  This is a RESULT, not a regression. Raising the constitutive melanin penalty
  to the value four independent within-study routes support removes schedules
  that the previous, too-weak penalty had reported as achievable. The floor,
  the erythema cap and the session-length ceiling are all unchanged; what
  changed is how much 25(OH)D a given dose produces in pigmented skin.

  Read the infeasible cells as "sun alone, at this latitude proxy and this
  session budget, does not hold 50 nmol/L year-round without exceeding a
  non-burning dose" -- which is the honest answer, and is the same conclusion
  Webb 2018 and Kift 2013 reach for FST V in the UK from measured data.

  K_const IS NOT THE ONLY MOVER. Run with --before combined (or --factorial)
  before quoting any cell as a K_const consequence: A = %.6g -> %.6g pushes
  feasibility the other way and is large enough to reverse some cells that the
  2026-08-17 single-parameter map reported as lost.""" % (OLD_A, ModelParams().A))


if __name__ == "__main__":
    main()
