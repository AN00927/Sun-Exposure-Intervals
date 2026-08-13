"""
joint_schedule_freesolve.py
===========================
Validity check for the low-dimensional joint timing+dose optimiser.

`vitd_optimization.optimise_joint_schedule` searches a 4-parameter family
(spacing exponent + a 3-parameter dose shape). That is fast and interpretable,
but it can only find what it can express. This solves the SAME problem without
a shape family -- one free dose per day over the whole horizon -- and reports
how much the parameterisation gives up.

Formulation. With one decision variable per day and `d_t = 0` meaning "no
session that day", timing and dose are encoded together and no integer
variables are needed. Session count is then the L0 norm of the dose vector,
which is relaxed to a reweighted L1 penalty:

    minimise   combined_risk(d) + lambda * sum_t w_t d_t
    subject to min D(t >= ramp) >= target,  0 <= d_t <= 0.7 * MED
    with       w_t <- 1 / (d_t + eps)   iterated

This is the maximum-hands-off control formulation (Nagahara, Quevedo & Nesic
2016, IEEE TAC 61:735-747), with the reweighting of Candes, Wakin & Boyd (2008,
J Fourier Anal Appl 14:877-905). Their L1 = L0 equivalence is proved for LINEAR
dynamics; this system is nonlinear, so sparsity is checked empirically here
rather than assumed.

Runtime is a few minutes -- it is deliberately NOT part of run_all.py.

Usage:  PYTHONIOENCODING=utf-8 python3 scripts/joint_schedule_freesolve.py
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import simulate                              # noqa: E402
from vitd_objectives import (SUFFICIENCY_NMOL_L, GAMMA_MED,  # noqa: E402
                             RAMP_IN_DAYS, time_to_sufficiency)
from vitd_physiology import cancer_risk, _med_sed_by_fst     # noqa: E402
from vitd_optimization import (optimise_joint_schedule,      # noqa: E402
                               rebuild_joint_schedule)

FST = 3
HORIZON = 365
D0 = 25.0
TARGET = SUFFICIENCY_NMOL_L
RAMP = RAMP_IN_DAYS
CAP = GAMMA_MED * _med_sed_by_fst(FST)
DAYS = np.arange(HORIZON, dtype=float)
PEN = 20.0
DROP = 1e-3          # below this a session is not a session
_nev = [0]


def _simulate(d):
    _nev[0] += 1
    sched = [(float(t), float(s)) for t, s in zip(DAYS, d) if s > DROP]
    if not sched:
        return None
    return simulate(fst=FST, schedule=sched, duration_days=HORIZON, D0=D0,
                    dose_framing="fixed_per_session_SED")


def metrics(d):
    d = np.clip(np.nan_to_num(d), 0.0, CAP)
    r = _simulate(d)
    if r is None:
        return dict(risk=np.inf, trough=-np.inf, total=0.0, n=0, ttf=np.nan,
                    peak=0.0)
    t = np.asarray(r["t"]); D = np.asarray(r["D"])
    mask = t >= RAMP
    active = d[d > DROP]
    return dict(risk=r["cancer_risk"]["combined_index"],
                trough=float(np.min(D[mask])) if mask.any() else -np.inf,
                total=float(r["cumulative_SED"]), n=int(active.size),
                ttf=time_to_sufficiency(r, TARGET),
                peak=float(np.max(active)) if active.size else 0.0)


def objective(d, lam, w):
    m = metrics(d)
    if not np.isfinite(m["risk"]):
        return 1e6
    return (m["risk"] + lam * float(np.sum(w * np.clip(d, 0.0, CAP)))
            + PEN * max(0.0, TARGET - m["trough"]) ** 2)


def solve(lam, d_init, reweights=2, maxiter=35):
    """Reweighted-L1 solve. Returns the dose vector."""
    d = np.clip(d_init, 0.0, CAP)
    w = np.ones_like(d)
    for _ in range(reweights):
        res = minimize(objective, d, args=(lam, w), method="SLSQP",
                       bounds=[(0.0, CAP)] * len(d),
                       options={"maxiter": maxiter, "ftol": 1e-7})
        d = np.clip(res.x, 0.0, CAP)
        w = 1.0 / (d + 0.05 * CAP)      # Candes-Wakin-Boyd reweighting
        w = w / np.max(w)
    return d


def parameterised_reference(n_sessions):
    """The low-dim answer, projected onto the daily grid as a warm start.

    Session days are real-valued and are snapped to the integer day grid, so
    the projection is not exactly the parameterised schedule -- it is only the
    starting point for the free solve. The comparison itself uses the
    optimiser's own reported figures, not this projection.
    """
    best = optimise_joint_schedule(FST, n_sessions, horizon_days=HORIZON,
                                   D0=D0, target=TARGET, ramp_in_days=RAMP,
                                   free_spacing=True)
    if best is None:
        return None, None
    days, seds = rebuild_joint_schedule(best, horizon_days=HORIZON)
    d = np.zeros(HORIZON)
    for day, sed in zip(days, seds):
        d[int(min(HORIZON - 1, round(day)))] += sed
    return best, np.clip(d, 0.0, CAP)


def main():
    print("=" * 74)
    print("Free-solve validity check: does the 4-parameter family give up much?")
    print("=" * 74)
    print(f"  FST {FST}, horizon {HORIZON}d, target {TARGET:.0f} nmol/L, "
          f"cap {CAP:.2f} SED")
    print("  One free dose per day (365 variables). Timing and dose are")
    print("  encoded together; d_t = 0 means no session, so there are no")
    print("  integer variables. Session count is controlled by a reweighted")
    print("  L1 penalty (maximum-hands-off control).\n")

    ref, d_init = parameterised_reference(104)
    if ref is None:
        print("  parameterised reference infeasible; aborting")
        return
    print(f"  Parameterised reference (N=104, gamma={ref['gamma']:.2f}, "
          f"{ref['family']}):")
    print(f"    risk {ref['combined_risk']:.4f}   total {ref['total_sed']:.1f} SED"
          f"   peak {ref['max_sed']:.3f} SED   sessions {ref['n_sessions_delivered']}")

    print(f"\n  {'lambda':>8} {'sessions':>9} {'risk':>9} {'totSED':>8} "
          f"{'peak':>7} {'trough':>7} {'t->suff':>8}")
    print("  " + "-" * 62)
    results = []
    for lam in (0.0, 0.002, 0.005, 0.02):
        d = solve(lam, d_init.copy())
        m = metrics(d)
        feasible = m["trough"] >= TARGET - 0.5
        ttf = "n/a" if not np.isfinite(m["ttf"]) else f"{m['ttf']:.0f}d"
        print(f"  {lam:>8.3f} {m['n']:>9} {m['risk']:>9.4f} {m['total']:>8.1f} "
              f"{m['peak']:>7.3f} {m['trough']:>7.2f} {ttf:>8}"
              f"{'' if feasible else '   INFEASIBLE'}")
        if feasible:
            results.append((lam, m))

    print(f"\n  function evaluations: {_nev[0]}")

    if not results:
        print("\n  No feasible free solve; the parameterised answer stands"
              " unchallenged but also unverified.")
        return

    counts = {m["n"] for _, m in results}
    if len(counts) == 1:
        print("\n  NOTE: the session count is identical at every lambda, so the")
        print("  L1 term is INERT here -- it never zeroes a session out. This is")
        print("  not a solver failure, it is the degeneracy again: risk is")
        print("  monotone decreasing in session count, so the objective never")
        print("  WANTS fewer sessions and sparsity is never rewarded. The")
        print("  session budget is exogenous (adherence), not something the")
        print("  optimiser should discover. Maximum-hands-off control motivates")
        print("  the zero-dose-means-no-session encoding, which is what removes")
        print("  the integer variables; its sparsity machinery is not needed.")

    best_lam, best_m = min(results, key=lambda kv: kv[1]["risk"])
    gap = 100.0 * (ref["combined_risk"] - best_m["risk"]) / ref["combined_risk"]
    print(f"\n  Best feasible free solve: risk {best_m['risk']:.4f} at "
          f"{best_m['n']} sessions (lambda={best_lam}).")
    print(f"  Parameterised family gives up {gap:+.1f}% relative to it.")
    if abs(gap) < 10.0:
        print("  => The 4-parameter family captures the available gain; the")
        print("     interpretable answer is not materially suboptimal.")
    else:
        print("  => The free solve finds materially more. The shape families")
        print("     are the binding limitation, not the physiology.")
    print("\n  Caveat: SLSQP on 365 variables with finite-difference gradients")
    print("  is a LOCAL method warm-started from the parameterised answer, so")
    print("  this bounds the gap from one side only -- it can confirm the")
    print("  family is adequate, but cannot prove global optimality.")


if __name__ == "__main__":
    main()
