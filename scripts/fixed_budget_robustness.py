"""Is the fixed-budget interval ranking robust where the usable schedules sit?
===========================================================================

WHY THIS EXISTS
---------------
The 2026-09-29 mock review (roadmap items R6 and R7) found three gaps in the
manuscript's claim that "the interval ranking survives because it depends on
the shape of the dose-response rather than its scale":

1. **Low-dose regime (R6).** Table 1's risk-aware schedules deliver
   0.19-1.36 SED per daily session (about 8-10 % MED). That is below the
   >= 0.5 SED/day budgets at which ``tests/test_concavity_assumption.py``
   finds the ordering protected by the measured per-session exponents. The
   concavity margin was never reported at those operating points.
2. **Scale inside the logarithm (R7a).** ``A`` and ``K_const`` multiply the
   kernel and cannot move the ranking, but the scale of the log ARGUMENT
   (``sed * E_source``) sets where the knee of ``ln(1 + x)`` falls, and the
   one-at-a-time analysis never varies it.
3. **Joint propagation at a fixed budget (R7b).** The only whole-model Monte
   Carlo (``vitd_sensitivity.uncertainty_propagation``) fixes the dose PER
   SESSION at 1.0 SED and ranks on a per-SED metric the paper itself bars from
   advice. The fixed-BUDGET ranking was never propagated jointly.

WHAT IT DOES
------------
A. At each FST's Table 1 operating point (per-session dose recomputed live
   with the same solver and constants as ``scripts/risk_aware_scheduling.py``)
   it holds that daily budget fixed and compares intervals 1, 2, 3, 4, 7 d on
   steady-state 25(OH)D (``vitd_model.steady_state_D``, final 28 days, 365-d
   horizon, D0 as in Table 1). It then bisects the curvature exponent at which
   daily stops winning (``gamma_flip``) using the concavity test's own warp and
   harness, and reports it beside the measured per-session exponents.
   Because the warp puts ``u ** gamma`` INSIDE ``ln(1 + .)``, the local
   log-log slope of the warped kernel equals ``gamma`` only as dose -> 0; the
   local slope at the operating dose is reported too, so the comparison with
   measured log-log exponents is made on one scale.
B. Sweeps a multiplicative scale ``lambda`` on the log argument (via
   ``E_source_override``) at FST V, VI and the FST II light-skin contrast, and
   reports the daily-to-weekly change and interval monotonicity at three
   budgets. ``lambda`` is a sensitivity knob, NOT a sourced prior.
C. Draws ``N_DRAWS`` parameter sets uniformly over
   ``vitd_sensitivity.PARAM_RANGES`` (the documented prior ranges, the same
   ones the existing Monte Carlo uses) and, for each draw, compares intervals
   at a FIXED budget. Reports P(daily is best) and 2.5/50/97.5 percentiles of
   each interval's change against daily.

Nothing here changes a shipped value. Parameter values are not restated: the
canonical sources are the ``vitd_model.py`` module docstring and
``vitd_scenario.py``.

Run:  PYTHONIOENCODING=utf-8 python3 scripts/fixed_budget_robustness.py
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vitd_model import (ModelParams, simulate, steady_state_D,    # noqa: E402
                        allow_retired_parameter_sweep)
from vitd_physiology import _med_sed_by_fst                       # noqa: E402
from vitd_objectives import SUFFICIENCY_NMOL_L                    # noqa: E402
from vitd_optimization import optimal_interval_risk_constrained   # noqa: E402
from vitd_sensitivity import PARAM_RANGES                         # noqa: E402
from scripts import risk_aware_scheduling as ras                   # noqa: E402

TABLES_DIR = os.path.join(ROOT, "tables")
FIGS_DIR = os.path.join(ROOT, "figs")

INTERVALS = (1, 2, 3, 4, 7)
MC_INTERVALS = (1, 2, 3, 7)
#: Matched budgets, SED per day of horizon: arm (a)'s three weekly budgets.
BUDGETS_PER_DAY = (0.25, 0.5, 1.0)
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
SWEEP_FSTS = (5, 6, 2)
MC_FSTS = (5, 6)
N_DRAWS = 400
SEED = 0


def _concavity_module():
    """The concavity test's warp and flip harness, loaded by path.

    Reused rather than copied so this script and the test can never disagree
    about what ``gamma_flip`` means. ``tests/`` is not a package, hence the
    path import.
    """
    path = os.path.join(ROOT, "tests", "test_concavity_assumption.py")
    spec = importlib.util.spec_from_file_location("_concavity_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _budget_schedule(horizon, per_day, interval):
    days = list(range(0, horizon, interval))
    per = per_day * horizon / len(days)
    return [(float(d), per) for d in days]


def _ss(fst, per_day, interval, params=None, E=None):
    kw = {"exposed_bsa_override": ras.EXPOSED_BSA_REF}
    if E is not None:
        kw["E_source_override"] = E
    r = simulate(fst=fst, schedule=_budget_schedule(ras.HORIZON, per_day, interval),
                 duration_days=ras.HORIZON, params=params or ModelParams(),
                 D0=ras.D0, dose_framing="fixed_per_session_SED", **kw)
    return steady_state_D(r)


def _local_loglog_slope(u, gamma):
    """d ln f / d ln u for f(u) = ln(1 + u**gamma)."""
    x = u ** gamma
    return gamma * x / ((1.0 + x) * math.log1p(x))


# ---------------------------------------------------------------------------
# A. Table 1 operating points
# ---------------------------------------------------------------------------
def operating_points(verbose=True):
    tca = _concavity_module()
    emp = tca.empirical_exponents()
    shih = emp["shih2018"]["loglog"]
    rows = []
    for fst in ras.FSTS:
        res = optimal_interval_risk_constrained(
            fst=fst, target=SUFFICIENCY_NMOL_L, risk_cap=None,
            horizon_days=ras.HORIZON, D0=ras.D0, params=ModelParams(),
            interval_grid=ras.INTERVAL_GRID, max_minutes=ras.PRACTICAL_MAX_MINUTES,
            uv_index=ras.UV_INDEX_REF, exposed_bsa=ras.EXPOSED_BSA_REF,
            ramp_in_days=ras.RAMP_IN_DAYS)
        rec = res["recommended"]
        if rec is None:
            rows.append(dict(fst=fst, note=res["note"]))
            continue
        c = rec["sed_per_session"] / rec["interval_days"]  # SED per day
        ss = {iv: _ss(fst, c, iv) for iv in INTERVALS}
        row = dict(fst=fst, table1_interval_days=rec["interval_days"],
                   budget_sed_per_day=c,
                   pct_MED_daily=100.0 * c / _med_sed_by_fst(fst),
                   ss_D_daily=ss[1])
        for iv in INTERVALS[1:]:
            row["pct_change_%dd_vs_daily" % iv] = 100.0 * (ss[iv] / ss[1] - 1.0)
        # NaN from find_flip means daily still wins at GAMMA_CAP: no flip
        # within the searched range. That is protection by the full model's
        # state dynamics (feedback), not by kernel curvature, so the bare
        # kernel's closed-form flip is reported beside it.
        g = tca.find_flip(fst, c)
        worst = tca.most_convex_point_estimate()
        row["gamma_flip"] = g
        row["no_flip_up_to_gamma_cap"] = not np.isfinite(g)
        row["gamma_cap"] = tca.GAMMA_CAP
        row["bare_kernel_gamma_flip"] = tca.analytic_flip_gamma(c)
        row["gamma_flip_local_slope_at_dose"] = (
            _local_loglog_slope(c, g) if np.isfinite(g) else float("nan"))
        row["shipped_local_slope_at_dose"] = _local_loglog_slope(c, 1.0)
        row["shih2018_loglog"] = shih
        row["most_convex_measured_loglog"] = worst
        # Two scales, because the warp exponent and a measured log-log slope
        # coincide only as dose -> 0 (see module docstring).
        row["protected_gamma_scale"] = bool(not np.isfinite(g) or g > worst)
        row["protected_local_slope_scale"] = bool(
            not np.isfinite(g) or row["gamma_flip_local_slope_at_dose"] > worst)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "low_dose_regime.csv"), index=False)
    if verbose:
        print("=" * 100)
        print("A. LOW-DOSE REGIME -- the Table 1 operating points, budget held "
              "fixed, interval varied")
        print("=" * 100)
        print("  Steady state = mean of final 28 d, 365-d horizon, same D0, BSA "
              "and solver constants as Table 1.")
        print("  gamma_flip = curvature exponent at which daily stops winning "
              "(concavity-test warp + harness).")
        print("  slope@dose = local log-log slope of ln(1+u^gamma) at the "
              "operating dose: the scale on which a measured")
        print("  per-session exponent should be compared. Measured exponents: "
              + ", ".join("%s %.4f" % (k, v["loglog"]) for k, v in emp.items()))
        print()
        print(f"  {'FST':>3s} {'SED/d':>7s} {'%MED':>6s} {'ss D 1d':>8s} "
              + " ".join(f"{'%dd' % iv:>8s}" for iv in INTERVALS[1:])
              + f" {'g_flip':>8s} {'bare':>7s} {'slope@d':>8s} {'shipped':>8s}"
              "  protected (gamma / slope scale)")
        for _, r in df.iterrows():
            if "budget_sed_per_day" not in r or pd.isna(r.get("budget_sed_per_day")):
                print(f"  {int(r['fst']):>3d}  no recommendation: {r.get('note')}")
                continue
            gf = (f"{r['gamma_flip']:8.4f}" if np.isfinite(r["gamma_flip"])
                  else f"{'>' + format(r['gamma_cap'], 'g'):>8s}")
            sl = (f"{r['gamma_flip_local_slope_at_dose']:8.4f}"
                  if np.isfinite(r["gamma_flip_local_slope_at_dose"]) else f"{'-':>8s}")
            print(f"  {int(r['fst']):>3d} {r['budget_sed_per_day']:7.4f} "
                  f"{r['pct_MED_daily']:6.2f} {r['ss_D_daily']:8.2f} "
                  + " ".join(f"{r['pct_change_%dd_vs_daily' % iv]:+7.2f}%"
                             for iv in INTERVALS[1:])
                  + f" {gf} {r['bare_kernel_gamma_flip']:7.4f} {sl} "
                  f"{r['shipped_local_slope_at_dose']:8.4f}  "
                  f"{'yes' if r['protected_gamma_scale'] else 'NO'} / "
                  f"{'yes' if r['protected_local_slope_scale'] else 'NO'}")
        print(f"  Protected = flip exponent above the most convex measured "
              f"point estimate ({df['most_convex_measured_loglog'].dropna().iloc[0]:.4f}, "
              "Shih 2018). '>cap' = daily still wins at the search cap: protection by")
        print("  the full model's state dynamics, not by kernel curvature; the "
              "'bare' column is the kernel-only closed-form flip.")
        print()
    return df


# ---------------------------------------------------------------------------
# B. Scale of the log argument
# ---------------------------------------------------------------------------
def log_scale_sweep(verbose=True):
    E0 = ModelParams().E_source
    rows = []
    for lam in LAMBDAS:
        for fst in SWEEP_FSTS:
            for c in BUDGETS_PER_DAY:
                ss = {iv: _ss(fst, c, iv, E=lam * E0) for iv in MC_INTERVALS}
                vals = [ss[iv] for iv in MC_INTERVALS]
                rows.append(dict(
                    lam=lam, fst=fst, budget_sed_per_day=c,
                    ss_D_daily=ss[1],
                    pct_change_weekly_vs_daily=100.0 * (ss[7] / ss[1] - 1.0),
                    pct_change_2d_vs_daily=100.0 * (ss[2] / ss[1] - 1.0),
                    monotone_decreasing_in_interval=bool(
                        all(a >= b for a, b in zip(vals, vals[1:])))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "log_scale_sweep.csv"), index=False)
    if verbose:
        print("=" * 100)
        print("B. SCALE OF THE LOG ARGUMENT -- lambda multiplies sed*E_source "
              "inside ln(1 + .); lambda = 1 is shipped")
        print("=" * 100)
        print("  lambda is a sensitivity knob, not a sourced prior. Change is "
              "steady-state 25(OH)D vs daily at a matched budget.\n")
        for fst in SWEEP_FSTS:
            tag = "  [light-skin contrast]" if fst == 2 else ""
            print(f"  FST {fst}{tag}")
            for c in BUDGETS_PER_DAY:
                g = df[(df.fst == fst) & (df.budget_sed_per_day == c)]
                cells = "  ".join(
                    f"l={r.lam:g}: wk {r.pct_change_weekly_vs_daily:+7.2f}% "
                    f"2d {r.pct_change_2d_vs_daily:+6.2f}%"
                    f"{'' if r.monotone_decreasing_in_interval else ' NONMONO'}"
                    for r in g.itertuples())
                print(f"    {c:4.2f} SED/d  {cells}")
            print()
        n_bad = int((~df.monotone_decreasing_in_interval).sum())
        print(f"  Non-monotone cells: {n_bad} of {len(df)}.\n")
    return df


# ---------------------------------------------------------------------------
# C. Joint prior propagation at a fixed budget
# ---------------------------------------------------------------------------
def fixed_budget_mc(verbose=True):
    rng = np.random.default_rng(SEED)
    keys = list(PARAM_RANGES)
    samples = {k: rng.uniform(PARAM_RANGES[k][0], PARAM_RANGES[k][1], N_DRAWS)
               for k in keys}
    raw = []
    for i in range(N_DRAWS):
        p = ModelParams()
        with allow_retired_parameter_sweep():
            for k in keys:
                setattr(p, k, float(samples[k][i]))
        for fst in MC_FSTS:
            for c in BUDGETS_PER_DAY:
                ss = {iv: _ss(fst, c, iv, params=p) for iv in MC_INTERVALS}
                best = max(MC_INTERVALS, key=lambda iv: ss[iv])
                rec = dict(draw=i, fst=fst, budget_sed_per_day=c, best=best,
                           ss_D_daily=ss[1])
                for iv in MC_INTERVALS[1:]:
                    rec["pct_%dd" % iv] = 100.0 * (ss[iv] / ss[1] - 1.0)
                raw.append(rec)
    raw = pd.DataFrame(raw)
    raw.to_csv(os.path.join(TABLES_DIR, "fixed_budget_mc_draws.csv"), index=False)
    summ = []
    for (fst, c), g in raw.groupby(["fst", "budget_sed_per_day"]):
        row = dict(fst=fst, budget_sed_per_day=c, n_draws=len(g),
                   p_daily_best=float((g["best"] == 1).mean()))
        for iv in MC_INTERVALS[1:]:
            v = g["pct_%dd" % iv].to_numpy()
            row["pct_%dd_p2.5" % iv] = float(np.percentile(v, 2.5))
            row["pct_%dd_median" % iv] = float(np.median(v))
            row["pct_%dd_p97.5" % iv] = float(np.percentile(v, 97.5))
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(TABLES_DIR, "fixed_budget_mc.csv"), index=False)
    if verbose:
        print("=" * 100)
        print(f"C. JOINT PRIOR PROPAGATION AT A FIXED BUDGET ({N_DRAWS} draws, "
              f"uniform over vitd_sensitivity.PARAM_RANGES, seed {SEED})")
        print("=" * 100)
        print("  Change in steady-state 25(OH)D vs daily at the same total SED; "
              "median [2.5, 97.5] percentiles.\n")
        for _, r in summ.iterrows():
            cells = "  ".join(
                f"{iv}d {r['pct_%dd_median' % iv]:+6.2f}% "
                f"[{r['pct_%dd_p2.5' % iv]:+6.2f}, {r['pct_%dd_p97.5' % iv]:+6.2f}]"
                for iv in MC_INTERVALS[1:])
            print(f"  FST {int(r['fst'])}  {r['budget_sed_per_day']:4.2f} SED/d  "
                  f"P(daily best) {100 * r['p_daily_best']:5.1f}%   {cells}")
        print()
    return summ, raw


def _figure(op, mc_raw):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ok = op.dropna(subset=["budget_sed_per_day"])
    for iv, mk in zip(INTERVALS[1:], ("o", "s", "^", "D")):
        ax.plot(ok["fst"], ok["pct_change_%dd_vs_daily" % iv], marker=mk,
                lw=1.4, label="every %d d" % iv)
    ax.axhline(0.0, color="0.4", lw=1)
    ax.set_xticks(list(ok["fst"]))
    ax.set_xticklabels(["FST %s" % "I II III IV V VI".split()[int(f) - 1]
                        for f in ok["fst"]])
    ax.set_ylabel("steady-state 25(OH)D vs daily (%)")
    ax.set_title("A. Table 1 operating points, budget held fixed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    data, labels = [], []
    for fst in MC_FSTS:
        for c in BUDGETS_PER_DAY:
            g = mc_raw[(mc_raw.fst == fst) & (mc_raw.budget_sed_per_day == c)]
            data.append(g["pct_2d"].to_numpy())
            labels.append("FST %s\n%g SED/d" % ("V" if fst == 5 else "VI", c))
    ax.boxplot(data, whis=(2.5, 97.5), showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.0, color="0.4", lw=1)
    ax.set_ylabel("every 2 d vs daily, same total SED (%)")
    ax.set_title("C. Joint prior propagation (%d draws; whiskers 2.5-97.5%%)"
                 % N_DRAWS)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    os.makedirs(FIGS_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGS_DIR, "fixed_budget_robustness.png"), dpi=150)
    plt.close(fig)


def run(verbose=True):
    os.makedirs(TABLES_DIR, exist_ok=True)
    op = operating_points(verbose)
    sweep = log_scale_sweep(verbose)
    summ, raw = fixed_budget_mc(verbose)
    _figure(op, raw)
    return {"operating_points": op, "log_scale_sweep": sweep,
            "mc_summary": summ}


def main():
    return run(verbose=True)


if __name__ == "__main__":
    main()
