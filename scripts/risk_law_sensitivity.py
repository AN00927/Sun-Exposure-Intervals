"""Does the vitamin D / skin-cancer alignment survive a different risk law?
=========================================================================

WHY THIS EXISTS
---------------
The manuscript's headline says the schedule that maximises 25(OH)D is also the
one that minimises modelled melanoma/BCC risk. Every seat of the 2026-09-29
mock review traced that alignment to two functional forms chosen as inputs:
a concave per-session synthesis kernel, and an intermittent risk term that is
convex in per-session dose (``sum((sed/MED) ** INTERMITTENT_BURN_EXPONENT)`` in
``vitd_physiology.cancer_risk``). With both forms fixed, splitting a dose
lowers the intermittent index by construction, so the alignment cannot fail.
The review asked for the risk law to be varied (roadmap item R2).

WHAT IT DOES
------------
It solves exactly the Table 1 problem -- ``optimal_interval_risk_constrained``
with the constants of ``scripts/risk_aware_scheduling.py`` -- once per skin
type. The sufficiency floor, erythema cap and session-length cap do not depend
on the risk law, so the set of feasible intervals, and the smallest dose that
holds sufficiency on each, are computed once. Each feasible uniform schedule
is then RE-SCORED under alternative intermittent-axis laws, and the argmin is
recomputed. Nothing is re-simulated per law and no shipped value changes.

Law families on the intermittent axis (the SCC axis is left as shipped):

* ``power k``: ``sum((sed/MED) ** k)``. ``k = INTERMITTENT_BURN_EXPONENT`` is
  the shipped law; ``k = 1`` removes all per-session super-linearity (the
  intermittent index becomes proportional to total dose).
* ``threshold theta``: ``sum(max(0, sed/MED - theta))``. Only the part of a
  session above ``theta`` MED carries risk -- a burn-threshold law.
* ``carryover tau``: shipped exponent, but each session's effective dose adds
  the previous effective dose decayed by ``exp(-gap/tau)``: damage from one
  session is not fully repaired before the next. This is the one mechanism
  that can make spacing SAFER, which is why it is included.

PARAMETER STATUS -- read before quoting
--------------------------------------
None of the alternative-law knobs (``k`` other than the shipped value,
``theta``, ``tau``) is sourced or calibrated. They are sensitivity sweeps
chosen to bracket plausible behaviour, not estimates. In particular the
carry-over timescales are NOT literature repair half-lives; a reader who wants
to use this against a measured CPD-repair time must supply one. Every law is
normalised by the same fixed reference schedule the shipped index uses
(52 sessions of ``REFERENCE_SESSION_SED`` at FST I), so index values are
comparable across skin types within a law but NOT across laws. Only the
argmin within one skin type is meaningful across laws.

Run:  PYTHONIOENCODING=utf-8 python3 scripts/risk_law_sensitivity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams                                # noqa: E402
from vitd_physiology import (INTERMITTENT_BURN_EXPONENT,          # noqa: E402
                             REFERENCE_SESSION_SED, _med_sed_by_fst,
                             basal_damage_fraction)
from vitd_objectives import SUFFICIENCY_NMOL_L                    # noqa: E402
from vitd_optimization import optimal_interval_risk_constrained   # noqa: E402
from scripts import risk_aware_scheduling as ras                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(ROOT, "tables")
OUT_CSV = os.path.join(TABLES_DIR, "risk_law_sensitivity.csv")

#: Sweep grids. Sensitivity choices, not sourced values -- see module docstring.
POWER_EXPONENTS = (1.0, 1.5, 2.0, INTERMITTENT_BURN_EXPONENT, 4.0)
THRESHOLDS_MED = (0.05, 0.1, 0.3)
CARRYOVER_TAU_DAYS = (0.5, 1.0, 2.0, 4.0, 8.0)


def _laws():
    laws = []
    for k in POWER_EXPONENTS:
        tag = " (shipped)" if k == INTERMITTENT_BURN_EXPONENT else ""
        laws.append(("power k=%g%s" % (k, tag), "power", k))
    for th in THRESHOLDS_MED:
        laws.append(("threshold theta=%g MED" % th, "threshold", th))
    for tau in CARRYOVER_TAU_DAYS:
        laws.append(("carryover tau=%g d, k=%g" % (tau, INTERMITTENT_BURN_EXPONENT),
                     "carryover", tau))
    return laws


def _raw_intermittent(days, seds, med, family, knob):
    frac = np.asarray(seds, dtype=float) / med
    if family == "power":
        return float(np.sum(frac ** knob))
    if family == "threshold":
        return float(np.sum(np.maximum(0.0, frac - knob)))
    if family == "carryover":
        k = INTERMITTENT_BURN_EXPONENT
        eff, prev_day, total = 0.0, None, 0.0
        for d, f in zip(days, frac):
            carry = 0.0 if prev_day is None else eff * np.exp(-(d - prev_day) / knob)
            eff = f + carry
            total += eff ** k
            prev_day = d
        return float(total)
    raise ValueError(family)


def intermittent_index(days, seds, fst, family, knob):
    """Alternative intermittent index, normalised like the shipped one."""
    ref_days = np.arange(52) * 7.0
    ref_seds = np.full(52, REFERENCE_SESSION_SED)
    ref = _raw_intermittent(ref_days, ref_seds, _med_sed_by_fst(1), family, knob)
    raw = _raw_intermittent(days, seds, _med_sed_by_fst(fst), family, knob)
    return basal_damage_fraction(fst) * raw / ref if ref > 0 else float("nan")


def run(verbose: bool = True) -> pd.DataFrame:
    rows = []
    for fst in ras.FSTS:
        res = optimal_interval_risk_constrained(
            fst=fst, target=SUFFICIENCY_NMOL_L, risk_cap=None,
            horizon_days=ras.HORIZON, D0=ras.D0, params=ModelParams(),
            interval_grid=ras.INTERVAL_GRID, max_minutes=ras.PRACTICAL_MAX_MINUTES,
            uv_index=ras.UV_INDEX_REF, exposed_bsa=ras.EXPOSED_BSA_REF,
            ramp_in_days=ras.RAMP_IN_DAYS)
        feasible = [e for e in res["evaluated"] if e["feasible"]]
        for law, family, knob in _laws():
            scored = []
            for e in feasible:
                days = np.arange(0, ras.HORIZON, e["interval_days"], dtype=float)
                seds = np.full(days.size, e["sed_per_session"])
                inter = intermittent_index(days, seds, fst, family, knob)
                scc = e["scc_index"]
                scored.append(dict(interval=e["interval_days"],
                                   n=e["n_sessions"], inter=inter, scc=scc,
                                   combined=float(np.sqrt(max(scc, 0.0)
                                                          * max(inter, 0.0)))))
            if not scored:
                rows.append(dict(fst=fst, law=law, family=family, knob=knob,
                                 feasible_intervals="", combined_argmin=None,
                                 intermittent_argmin=None, scc_argmin=None,
                                 combined_all_zero=False,
                                 daily_to_2d_intermittent_ratio=np.nan,
                                 note=res["note"]))
                continue
            # Shipped tie rule: lowest combined, then fewer sessions.
            best = min(scored, key=lambda s: (s["combined"], s["n"]))
            best_i = min(scored, key=lambda s: (s["inter"], s["n"]))
            best_s = min(scored, key=lambda s: (s["scc"], s["n"]))
            by_iv = {s["interval"]: s for s in scored}
            ratio = (by_iv[2]["inter"] / by_iv[1]["inter"]
                     if 1 in by_iv and 2 in by_iv and by_iv[1]["inter"] > 0
                     else np.nan)
            rows.append(dict(
                fst=fst, law=law, family=family, knob=knob,
                feasible_intervals=" ".join(str(s["interval"]) for s in scored),
                combined_argmin=best["interval"],
                intermittent_argmin=best_i["interval"],
                scc_argmin=best_s["interval"],
                combined_all_zero=all(s["combined"] == 0.0 for s in scored),
                daily_to_2d_intermittent_ratio=ratio,
                note=""))
    df = pd.DataFrame(rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    if verbose:
        _print(df)
    return df


def _print(df: pd.DataFrame) -> None:
    print("=" * 100)
    print("RISK-LAW SENSITIVITY -- does the Table 1 optimum stay daily under "
          "other intermittent-risk laws?")
    print("=" * 100)
    print("  Same constraints as Table 1; feasible intervals and their minimum "
          "doses computed once, then re-scored.")
    print("  Columns: argmin interval (days) by combined index / intermittent "
          "axis alone / SCC axis alone;")
    print("  r2 = intermittent index of every-2-day over daily at each "
          "interval's own minimum sufficient dose.")
    print("  Alternative-law knobs are UNSOURCED sensitivity choices.\n")
    for fst, g in df.groupby("fst"):
        feas = g["feasible_intervals"].iloc[0] or "none"
        print(f"  FST {fst}  feasible intervals: {feas}")
        for _, r in g.iterrows():
            if r["combined_argmin"] is None or pd.isna(r["combined_argmin"]):
                print(f"    {r['law']:34s}  no feasible interval")
                continue
            flag = "  [combined index 0 everywhere: tie rule decides]" \
                if r["combined_all_zero"] else ""
            r2 = r["daily_to_2d_intermittent_ratio"]
            print(f"    {r['law']:34s}  combined {int(r['combined_argmin']):>2d} d"
                  f"   intermittent {int(r['intermittent_argmin']):>2d} d"
                  f"   SCC {int(r['scc_argmin']):>2d} d"
                  f"   r2 {'n/a' if pd.isna(r2) else '%.3f' % r2}{flag}")
        print()


def main():
    return run(verbose=True)


if __name__ == "__main__":
    main()
