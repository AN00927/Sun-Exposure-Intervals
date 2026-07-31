"""
melanin_sc_coupling.py
======================
Are pigmentation and stratum-corneum thickening the same channel?

Answers a specific question: does melanin covary with barrier thickening, and
can the model tell the two apart? Three tests:

  1. Trajectory coupling  -- correlation of M(t) and SC(t) under a common UV
     schedule. Both states are driven by the same input through the same
     saturating form.

  2. Joint identifiability -- SVD of the steady-state D response surface over
     (K_ATTEN, K_SC).

  3. Phenotype stratification -- SC_max is already FST-stratified in the model
     from de Boer 2025 (FP I-III vs FP IV-VI).

This is a diagnostic, not a calibration. It changes no parameter.

Run:  PYTHONIOENCODING=utf-8 python3 scripts/melanin_sc_coupling.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import (ModelParams, simulate, steady_state_D,
                        mean_final_M, mean_final_SC)

HORIZON = 180
D0 = 25.0


def _sched(sed: float, interval: int, horizon: int = HORIZON):
    return [(float(d), sed) for d in range(0, horizon, interval)]


# ---------------------------------------------------------------------------
# 1. Do M(t) and SC(t) move together?
# ---------------------------------------------------------------------------
def trajectory_coupling():
    print("=" * 76)
    print("1. TRAJECTORY COUPLING  --  does melanin track barrier thickening?")
    print("=" * 76)
    print(f"  {'FST':>3s} {'sed':>5s} {'interval':>9s} {'corr(M,SC)':>11s} "
          f"{'dM':>8s} {'dSC':>8s}")
    print(f"  {'-'*3} {'-'*5} {'-'*9} {'-'*11} {'-'*8} {'-'*8}")
    for fst in (2, 5):
        for sed in (1.0, 2.0):
            for interval in (3, 7):
                p = ModelParams()
                r = simulate(fst=fst, schedule=_sched(sed, interval),
                             duration_days=HORIZON, params=p, D0=D0,
                             dose_framing="fixed_per_session_SED")
                M = np.asarray(r["M"], dtype=float)
                SC = np.asarray(r["SC"], dtype=float)
                if M.std() > 1e-12 and SC.std() > 1e-12:
                    corr = float(np.corrcoef(M, SC)[0, 1])
                else:
                    corr = float("nan")
                print(f"  {fst:>3d} {sed:>5.1f} {interval:>8d}d {corr:>11.4f} "
                      f"{M[-1]-M[0]:>8.4f} {SC[-1]-SC[0]:>8.4f}")
    print("\n  Both states are driven by the same UV input through the same")
    print("  saturating form, so a correlation near 1.0 means the model cannot")
    print("  separate them from any single UV schedule.\n")


# ---------------------------------------------------------------------------
# 2. Can data tell K_ATTEN and K_SC apart?
# ---------------------------------------------------------------------------
def joint_identifiability(fst: int = 5, n_grid: int = 11):
    print("=" * 76)
    print("2. JOINT IDENTIFIABILITY  --  K_ATTEN vs K_SC on steady-state D")
    print("=" * 76)
    av = np.linspace(0.0, 1.0, n_grid)      # K_ATTEN plausible range
    bv = np.linspace(0.5, 2.0, n_grid)      # K_SC prior range
    Z = np.zeros((n_grid, n_grid))
    for i, x in enumerate(av):
        for j, y in enumerate(bv):
            p = ModelParams()
            p.K_ATTEN, p.K_SC = float(x), float(y)
            r = simulate(fst=fst, schedule=_sched(1.0, 3), duration_days=HORIZON,
                         params=p, D0=D0, dose_framing="fixed_per_session_SED")
            Z[i, j] = steady_state_D(r)
    Zc = Z - Z.mean()
    sv = np.linalg.svd(Zc, compute_uv=False)
    ridge = float(sv[0] ** 2 / np.sum(sv ** 2))
    print(f"  steady-state D spread across the whole grid: "
          f"{Z.max() - Z.min():.3f} nmol/L")
    print(f"  variance on the leading singular direction:  {ridge:.4%}")
    print(f"  marginal effect of K_ATTEN alone (K_SC held): "
          f"{np.ptp(Z[:, n_grid//2]):.3f} nmol/L")
    print(f"  marginal effect of K_SC alone (K_ATTEN held): "
          f"{np.ptp(Z[n_grid//2, :]):.3f} nmol/L")
    verdict = ("UNIDENTIFIABLE -- only the combination is constrained"
               if ridge > 0.99 else "separable in principle")
    print(f"  verdict: {verdict}\n")
    return ridge


# ---------------------------------------------------------------------------
# 3. How much of the FST effect is barrier, not pigment?
# ---------------------------------------------------------------------------
def phenotype_stratification():
    print("=" * 76)
    print("3. PHENOTYPE STRATIFICATION  --  decomposing the FST II vs V gap")
    print("=" * 76)
    print("  SC_max is already FST-stratified from de Boer 2025 "
          "(FP I-III 0.23, FP IV-VI 0.36).")
    print("  Ablate each channel to see what actually produces the FST gap.\n")

    configs = [
        ("full model",                 dict()),
        ("no facultative melanin",     dict(K_ATTEN=0.0)),
        ("no SC attenuation",          dict(K_SC=0.0)),
        ("no constitutive pigment",    dict(K_const=0.0)),
        ("no melanin at all",          dict(K_ATTEN=0.0, K_const=0.0)),
        ("no photoadaptation at all",  dict(K_ATTEN=0.0, K_SC=0.0)),
    ]
    print(f"  {'configuration':<28s} {'FST II':>8s} {'FST V':>8s} "
          f"{'gap':>7s} {'gap kept':>9s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*7} {'-'*9}")
    base_gap = None
    for label, overrides in configs:
        vals = {}
        for fst in (2, 5):
            p = ModelParams()
            for k, v in overrides.items():
                setattr(p, k, v)
            r = simulate(fst=fst, schedule=_sched(1.0, 3), duration_days=HORIZON,
                         params=p, D0=D0, dose_framing="fixed_per_session_SED")
            vals[fst] = steady_state_D(r)
        gap = vals[2] - vals[5]
        if base_gap is None:
            base_gap = gap
        kept = gap / base_gap if base_gap else float("nan")
        print(f"  {label:<28s} {vals[2]:>8.2f} {vals[5]:>8.2f} "
              f"{gap:>7.2f} {kept:>8.1%}")
    print()


def main():
    trajectory_coupling()
    joint_identifiability()
    phenotype_stratification()
    print("=" * 76)
    print("Interpretation notes")
    print("=" * 76)
    print("  - The model stratifies SC_max by Fitzpatrick phototype because")
    print("    de Boer 2025 measured a larger SC change in FP IV-VI (+36%) than")
    print("    in FP I-III (+23%). That is a PHENOTYPE association measured in")
    print("    31 people, not a genotype one.")
    print("  - No dataset in this repository links pigment genotype (ASIP,")
    print("    SLC24A4, SLC45A2, MC1R) to barrier or epidermal thickening.")
    print("    See OPEN_WORK.md gap 4 and data request 11.")
    print("  - de Boer 2025 attribute their SC change to EDEMA, not acanthosis;")
    print("    epidermal thickness was flat (108.6 -> 109.4 um, p = 0.718).")
    print("    Treat SC here as an empirical channel, not proven optics.")
    print("    See OPEN_WORK.md gap 28.")


if __name__ == "__main__":
    main()
