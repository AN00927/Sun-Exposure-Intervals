"""Two more robustness checks on the Table 1 answer: the target and g(D).
=======================================================================

WHY THIS EXISTS
---------------
Two suggested items from the 2026-09-29 mock review (roadmap S1 and S5):

* **S1 -- the sufficiency target.** Table 1 and every feasibility statement
  for FST V-VI use one 50 nmol/L floor. Whether darker skin types are
  "infeasible" at a given session budget depends on that number, and 50 is
  one of several thresholds in use. This reruns the Table 1 problem
  unchanged except for the target.
* **S5 -- the feedback form g(D).** The shipped biphasic form's exponent
  ``n`` is a choice the source cannot fit (``vitd_model.py`` docstring,
  ``feedback_mode`` entry), and two other forms are kept in the code "for
  sensitivity work". This reports the headline interval contrasts under each.

WHAT IT DOES
------------
S1. ``optimal_interval_risk_constrained`` with exactly the constants of
    ``scripts/risk_aware_scheduling.py``, at each target in ``TARGETS``, for
    every FST. Reports the recommended interval, dose, minutes and binding
    constraint, or infeasibility.
S5. Under each feedback variant, the daily-to-weekly and daily-to-2-day
    change in steady-state 25(OH)D at a matched budget (same harness as
    ``scripts/fixed_budget_robustness.py``), at FST V and VI, plus which
    interval wins. Variants: shipped biphasic; biphasic with ``n`` = 1 and 4
    (``n`` is unfittable -- see docstring); the rejected ``saturating`` form at
    its unattributed default scale and at the Karppinen-fitted scale; and the
    withdrawn ``linear`` ceiling.

STATUS -- read before quoting
-----------------------------
No variant is re-calibrated. Changing g(D) without re-fitting ``A`` shifts
absolute 25(OH)D by an amount nobody has characterised, so under S5 only the
interval CONTRASTS and the winning interval carry meaning, never the levels.
The targets in S1 are conventional reporting thresholds, not model
parameters; the manuscript must cite a source for any it quotes. Nothing here
changes a shipped value.

Run:  PYTHONIOENCODING=utf-8 python3 scripts/target_and_feedback_sensitivity.py
"""
from __future__ import annotations

import functools
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import vitd_model as vm                                           # noqa: E402
from vitd_model import ModelParams                                # noqa: E402
from vitd_optimization import optimal_interval_risk_constrained   # noqa: E402
from scripts import fixed_budget_robustness as fbr                 # noqa: E402
from scripts import risk_aware_scheduling as ras                   # noqa: E402

TABLES_DIR = os.path.join(ROOT, "tables")

#: nmol/L. 50 is the shipped SUFFICIENCY_NMOL_L; the others bracket it.
TARGETS = (30.0, 40.0, 50.0, 75.0)
S5_FSTS = (5, 6)
S5_INTERVALS = (1, 2, 3, 7)

_ORIG_HYDROX = vm.hydroxylation_efficiency


def _variants():
    """(label, params_kwargs, hill_n_or_None)."""
    return [
        ("biphasic n=2 (shipped)", {}, None),
        ("biphasic n=1", {}, 1.0),
        ("biphasic n=4", {}, 4.0),
        ("saturating D_half=50 (unattributed default)",
         {"feedback_mode": "saturating", "D_half_feedback": 50.0}, None),
        ("saturating D_half=90 (Karppinen-fitted)",
         {"feedback_mode": "saturating", "D_half_feedback": 90.0}, None),
        ("linear D_max ceiling (withdrawn)", {"feedback_mode": "linear"}, None),
    ]


def target_sweep(verbose=True):
    rows = []
    for target in TARGETS:
        for fst in ras.FSTS:
            res = optimal_interval_risk_constrained(
                fst=fst, target=target, risk_cap=None,
                horizon_days=ras.HORIZON, D0=ras.D0, params=ModelParams(),
                interval_grid=ras.INTERVAL_GRID,
                max_minutes=ras.PRACTICAL_MAX_MINUTES,
                uv_index=ras.UV_INDEX_REF, exposed_bsa=ras.EXPOSED_BSA_REF,
                ramp_in_days=ras.RAMP_IN_DAYS)
            rec = res["recommended"]
            feas = [e["interval_days"] for e in res["evaluated"] if e["feasible"]]
            rows.append(dict(
                target_nmol_L=target, fst=fst,
                reachable=rec is not None,
                interval_days=rec["interval_days"] if rec else None,
                sed_per_session=rec["sed_per_session"] if rec else None,
                minutes_per_session=rec["minutes_per_session"] if rec else None,
                steady_state_D=rec["steady_state_D"] if rec else None,
                feasible_intervals=" ".join(map(str, feas)),
                binding_constraint=res["binding_constraint"],
                note=res["note"] or ""))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "target_sensitivity.csv"), index=False)
    if verbose:
        print("=" * 100)
        print("S1. SUFFICIENCY TARGET -- the Table 1 problem at other 25(OH)D "
              "floors (all other constraints unchanged)")
        print("=" * 100)
        print(f"  {'target':>7s} {'FST':>3s} {'interval':>8s} {'SED/sess':>9s} "
              f"{'min/sess':>8s} {'ss D':>7s}  feasible intervals")
        for r in df.itertuples():
            if r.reachable:
                print(f"  {r.target_nmol_L:7.0f} {r.fst:>3d} {int(r.interval_days):>6d} d "
                      f"{r.sed_per_session:9.4f} {r.minutes_per_session:8.1f} "
                      f"{r.steady_state_D:7.2f}  {r.feasible_intervals}")
            else:
                print(f"  {r.target_nmol_L:7.0f} {r.fst:>3d}  UNREACHABLE  {r.note}")
        print()
    return df


def feedback_variants(verbose=True):
    rows = []
    for label, kw, hill_n in _variants():
        if hill_n is not None:
            vm.hydroxylation_efficiency = functools.partial(_ORIG_HYDROX, n=hill_n)
        try:
            p = ModelParams(**kw)
            for fst in S5_FSTS:
                for c in fbr.BUDGETS_PER_DAY:
                    ss = {iv: fbr._ss(fst, c, iv, params=p) for iv in S5_INTERVALS}
                    rows.append(dict(
                        variant=label, fst=fst, budget_sed_per_day=c,
                        winner_days=max(S5_INTERVALS, key=lambda iv: ss[iv]),
                        pct_2d_vs_daily=100.0 * (ss[2] / ss[1] - 1.0),
                        pct_weekly_vs_daily=100.0 * (ss[7] / ss[1] - 1.0)))
        finally:
            vm.hydroxylation_efficiency = _ORIG_HYDROX
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "feedback_form_sensitivity.csv"), index=False)
    if verbose:
        print("=" * 100)
        print("S5. FEEDBACK FORM g(D) -- interval contrasts under alternative "
              "forms (no variant re-calibrated; levels not meaningful)")
        print("=" * 100)
        for fst in S5_FSTS:
            print(f"  FST {fst}")
            for label, _, _ in _variants():
                g = df[(df.variant == label) & (df.fst == fst)]
                cells = "  ".join(
                    f"{r.budget_sed_per_day:4.2f}/d: 2d {r.pct_2d_vs_daily:+6.2f}% "
                    f"wk {r.pct_weekly_vs_daily:+7.2f}% best {r.winner_days}d"
                    for r in g.itertuples())
                print(f"    {label:44s} {cells}")
            print()
        n_nondaily = int((df.winner_days != 1).sum())
        print(f"  Cells where an interval other than daily wins: {n_nondaily} "
              f"of {len(df)}.\n")
    return df


def run(verbose=True):
    os.makedirs(TABLES_DIR, exist_ok=True)
    return {"target_sweep": target_sweep(verbose),
            "feedback_variants": feedback_variants(verbose)}


def main():
    return run(verbose=True)


if __name__ == "__main__":
    main()
