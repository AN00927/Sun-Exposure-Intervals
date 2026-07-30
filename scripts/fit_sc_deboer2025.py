"""
fit_sc_deboer2025.py
====================
Attempted fit of the stratum-corneum parameters to de Boer 2025 Supp Table S1,
and a demonstration of why it FAILS.

Run this before trusting any fitted SC parameter.

BACKGROUND
----------
Table S1 is printed inside
`Papers_and_Datasets/Effect of Repeated Low-Dose UVR Exposure - de boer et al..pdf`
(page 12) and is machine-extractable. It has been in the repository all along.

Protocol: 0.8 SED to the lower back, 3x/week, 9 weeks, n=31.
Visits: T0 = baseline, T1 = 5 weeks (day 35), T2 = 9 weeks (day 63).

SC thickness, median (um):
    all FPs   5.8 -> 4.3 -> 7.5     (-26% then +29% relative to T0)
    FP I-III  5.6 -> 4.3 -> 6.9
    FP IV-VI  6.6 -> 4.3 -> 9.0

TWO SEPARATE FINDINGS
---------------------
1. STRUCTURAL (fixed). The old model could not represent a below-baseline dip
   at all. SC was a non-negative "thickness above baseline" and k_damage entered
   multiplicatively (`SC *= exp(-k_damage*sed)`), which can only shrink SC
   toward zero. No value of k_damage could fit the dip. It was pinned at 0.0 for
   a structural reason that earlier docs misdiagnosed as missing data.
   `vitd_model` now carries two processes, SC = SC_build - SC_damage, so
   below-baseline excursion is representable.

2. IDENTIFIABILITY (not fixed, and not fixable with these data). Even with the
   structure corrected, the three group medians cannot identify k_damage. This
   script demonstrates that rather than papering over it.

WHY THE FIT FAILS
-----------------
The data require an ACCELERATING response under CONSTANT dosing: roughly zero
net change through day 35, then +55 percentage points over the next 28 days.
Every process driven by a constant input decelerates. Three candidate
structures were tried and all fail the same way, at RMSE ~0.21-0.23 on a
signal of amplitude ~0.29:

    structure                        pred T1   pred T2   (obs -0.259 / +0.293)
    two-process (build - damage)      -0.113    +0.038
    lagged / cascade build            -0.025    +0.064
    MED-coupled damage                -0.051    +0.039

The MED-coupling idea is directionally right -- constant absolute dose against
a rising MED (5.3 -> 6.9 -> 7.9 SED) means the fractional-MED insult fell ~33%
over the study -- but the fitted MED rise has a ~87-day half-life, far too slow
to drive a late surge.

WHAT THIS MEANS
---------------
With two informative timepoints (T0 is 0 by construction) and no error bars on
the medians, we cannot distinguish "the model is missing a mechanism" from
"the T1 median is noisy". n=31 split across phototype strata, with T1 IQR
spanning 3.2-5.5 um, means the true dip could be materially smaller than the
medians suggest.

Therefore k_damage REMAINS 0.0. Fitting 3-4 parameters to 2 points is exactly
the overfitting the identifiability analysis in `vitd_sensitivity.py` warns
about. Individual-level trajectories are required -- see PAPER_REQUESTS item 1.

CAVEAT ON THE MECHANISM ITSELF
------------------------------
de Boer attribute the SC change to *edema*, and in the same table epidermal
thickness barely moves (108.6 -> 109.4 um, p = 0.718). The model uses SC to
attenuate UV as `exp(-K_SC * SC)`, i.e. added optical path length through a
thicker barrier. If the change is fluid rather than added corneocyte layers,
that interpretation is unsupported even if a fitted number reproduced the data.
See OPEN_QUESTIONS gap 28.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams, simulate, build_weekly_schedule

# de Boer 2025 Supp Table S1: SC thickness medians in um at T0 / T1 / T2
DEBOER_SC_UM = {
    "all":   (5.8, 4.3, 7.5),
    "I-III": (5.6, 4.3, 6.9),
    "IV-VI": (6.6, 4.3, 9.0),
}
# Same table: MED in SED, showing the insult falls in fractional-MED terms
DEBOER_MED_SED = {"all": (5.3, 6.9, 7.9),
                  "I-III": (4.4, 6.2, 6.9),
                  "IV-VI": (6.1, 6.9, 7.9)}
VISIT_DAYS = (0.0, 35.0, 63.0)
PROTOCOL_SED = 0.8
PROTOCOL_DAYS = [0, 2, 4]      # 3x/week
PROTOCOL_DURATION = 63
STRATUM_FST = {"all": 3, "I-III": 3, "IV-VI": 5}

# Fit quality worse than this (RMSE relative to signal amplitude) is reported as
# a structural failure rather than a calibration.
ACCEPTABLE_RELATIVE_RMSE = 0.25


def observed_relative(stratum: str) -> tuple[float, float, float]:
    """Absolute um -> fractional change from T0, matching the model's SC state."""
    t0, t1, t2 = DEBOER_SC_UM[stratum]
    return (0.0, t1 / t0 - 1.0, t2 / t0 - 1.0)


def model_relative(k_damage: float, alpha_SC: float, SC_max: float,
                   k_repair: float, fst: int) -> tuple[float, float, float]:
    p = ModelParams()
    p.k_damage = k_damage
    p.alpha_SC = alpha_SC
    p.k_repair_SC = k_repair
    sched = build_weekly_schedule(PROTOCOL_DURATION, PROTOCOL_SED, PROTOCOL_DAYS)
    r = simulate(fst=fst, schedule=sched, duration_days=PROTOCOL_DURATION,
                 params=p, D0=25.0, SC_max_override=SC_max)
    t, S = r["t"], r["SC"]
    return tuple(float(np.interp(d, t, S)) for d in VISIT_DAYS)


def best_achievable_fit(stratum: str, verbose: bool = True) -> dict:
    """Grid search, reported as an upper bound on achievable agreement."""
    obs = observed_relative(stratum)
    fst = STRATUM_FST[stratum]
    sc_max = max(0.05, obs[2])
    amplitude = max(abs(obs[1]), abs(obs[2]))

    best = None
    for k_damage in np.linspace(0.0, 0.30, 31):
        for alpha_SC in np.linspace(0.02, 0.60, 30):
            for k_repair in (np.log(2) / 3, np.log(2) / 7, np.log(2) / 14):
                pred = model_relative(k_damage, alpha_SC, sc_max, k_repair, fst)
                sse = (pred[1] - obs[1]) ** 2 + (pred[2] - obs[2]) ** 2
                if best is None or sse < best["sse"]:
                    best = {"stratum": stratum, "fst": fst, "SC_max": sc_max,
                            "k_damage": k_damage, "alpha_SC": alpha_SC,
                            "k_repair_SC": k_repair, "sse": sse,
                            "pred": pred, "obs": obs}
    best["rmse"] = float(np.sqrt(best["sse"] / 2))
    best["relative_rmse"] = best["rmse"] / amplitude
    best["acceptable"] = best["relative_rmse"] <= ACCEPTABLE_RELATIVE_RMSE

    if verbose:
        b = best
        print(f"  {stratum:6} (FST {b['fst']}):")
        print(f"    obs   T1={b['obs'][1]:+.3f}  T2={b['obs'][2]:+.3f}")
        print(f"    best  T1={b['pred'][1]:+.3f}  T2={b['pred'][2]:+.3f}")
        print(f"    RMSE={b['rmse']:.4f} ({b['relative_rmse']:.0%} of signal "
              f"amplitude) -> {'ACCEPTABLE' if b['acceptable'] else 'STRUCTURAL FAILURE'}")
        if b["alpha_SC"] >= 0.59:
            print(f"    alpha_SC pinned at the search bound -- the optimiser is "
                  f"pushing against structure, not finding an interior optimum")
    return best


def fractional_med_trend(verbose: bool = True) -> dict:
    """Show that the fractional-MED insult FALLS during the study.

    Constant 0.8 SED against a rising MED means the relative insult declines.
    Directionally this is the right mechanism for late recovery, but the MED
    rise is far too slow (~87 d half-life) to produce the observed late surge.
    """
    out = {}
    for stratum, meds in DEBOER_MED_SED.items():
        fracs = tuple(PROTOCOL_SED / m for m in meds)
        out[stratum] = fracs
        if verbose:
            print(f"  {stratum:6}: MED {meds[0]}->{meds[1]}->{meds[2]} SED, "
                  f"so dose is {fracs[0]:.3f}->{fracs[1]:.3f}->{fracs[2]:.3f} "
                  f"of MED ({100*(fracs[2]/fracs[0]-1):+.0f}%)")
    return out


def run(verbose: bool = True) -> dict:
    if verbose:
        print("=" * 74)
        print("SC FIT ATTEMPT vs de Boer 2025 Supp Table S1 -- EXPECTED TO FAIL")
        print("=" * 74)
        print("Protocol: 0.8 SED 3x/wk, 9 wk. Visits T0=0, T1=35 d, T2=63 d.")
        print("SC dips ~26% below baseline at T1, then overshoots ~+29% at T2.\n")

    fits = {s: best_achievable_fit(s, verbose) for s in DEBOER_SC_UM}

    if verbose:
        print("\n  Fractional-MED insult over the study:")
    fracs = fractional_med_trend(verbose)

    any_ok = any(f["acceptable"] for f in fits.values())
    if verbose:
        print()
        print("  VERDICT: " + ("some strata fit acceptably" if any_ok else
                               "no stratum fits -- structural, not parametric"))
        print("  The data need an ACCELERATING response under CONSTANT dosing.")
        print("  Saturating and cascade processes both decelerate; MED-coupling")
        print("  is directionally right but ~87 d too slow.")
        print()
        print("  k_damage therefore REMAINS 0.0. Two informative timepoints and")
        print("  no error bars cannot distinguish a missing mechanism from a")
        print("  noisy T1 median (n=31 across strata; T1 IQR 3.2-5.5 um).")
        print("  Individual-level trajectories required -- PAPER_REQUESTS item 1.")

    return {"fits": fits, "fractional_med": fracs, "any_acceptable": any_ok}


if __name__ == "__main__":
    run(verbose=True)
