"""
basics/sc_only.py
=================
TIER-1 REFERENCE MODEL: stratum corneum thickening, and nothing else.

WHY THIS FILE EXISTS
--------------------
Companion to `basics/melanin_only.py`. Where that file isolates the mechanism
that WORKS (the skin-type gradient: direction unanimous, size disputed), this
file isolates the mechanism that is UNDERDETERMINED, and is built to
demonstrate that rather than to hide it.

The expected and intended conclusion of this file is a negative result: the
stratum-corneum subsystem cannot be identified from the data this repository
holds. `CLAUDE.md` already flags `SC_max`, `alpha_SC` and `K_SC` as
uncalibrated priors; `vitd_model.py` records that `alpha_SC ~ SC_max` and
`K_ATTEN ~ K_SC` are unidentifiable ridges. This file turns those docstring
caveats into something a reviewer can see in a table.

THE MODEL, IN FULL
------------------
    dSC/dt = alpha_SC * UV(t) * (SC_max - SC)  -  k_decay_SC * SC
    attenuation = exp(-K_SC * SC)

That is the entire model. No melanin, no serum compartment, no feedback, no
dose-response, no geometry, no risk. `k_damage` is omitted because it is
pinned at 0.0 by default (see the `vitd_model.py` SC entry: the week-5 dip in
de Boer 2025 is not identifiable from three group medians).

PARAMETER VALUES
----------------
Read from `vitd_model.ModelParams` at runtime -- never restated here. The
canonical source is the `vitd_model.py` module docstring (`CLAUDE.md`).

VALIDATION DATA -- de Boer 2025 (P2)
------------------------------------
Protocol: 0.8 SED, 3x/week, 9 weeks, lower back.
Reported endpoint: SC thickness +23% (FP I-III, 5.6 -> 6.9 um) and +36%
(FP IV-VI, 6.6 -> 9.0 um), measured at three visits only (T0 = day 0,
T1 = day 35, T2 = day 63) as GROUP MEDIANS.

Three group medians is the whole dataset. That is the root of every problem
below.

WHAT THIS FILE DEMONSTRATES
---------------------------
  1. The model reproduces the reported ENDPOINT. That is not impressive:
     `SC_max` was set FROM those endpoints, so this is a restatement.
  2. `alpha_SC` and `SC_max` trade off along a ridge -- many (alpha_SC,
     SC_max) pairs hit the same 9-week endpoint, and the data cannot choose
     between them. Section 2 below prints that ridge.
  3. `K_SC` is not constrained by de Boer 2025 AT ALL. The paper measured
     stratum corneum thickness, not vitamin D. Nothing in it says how much
     synthesis a thickened SC blocks. `K_SC = 1.0` is a bare prior.
  4. The observed SHAPE is not reproduced. de Boer's SC dips ~26% BELOW
     baseline at week 5 before overshooting. A single saturating-rise minus
     first-order-decay ODE cannot produce a dip below its own initial
     condition. This is a structural mismatch, reported not hidden.

USAGE
-----
    PYTHONIOENCODING=utf-8 python3 basics/sc_only.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams  # noqa: E402

# --- de Boer 2025 protocol, as published -----------------------------------
DEBOER_SED_PER_SESSION = 0.8
DEBOER_SESSION_WEEKDAYS = (0, 2, 4)      # 3x/week
DEBOER_DAYS = 63                          # 9 weeks
DEBOER_VISIT_DAYS = (0, 35, 63)           # T0, T1, T2 -- the ENTIRE dataset
#: Observed SC thickness in um at the three visits, all-phototype pooled
#: (vitd_validation.validate_sc_deBoer2025: 5.8 -> 4.3 -> 7.5).
DEBOER_OBSERVED_UM = (5.8, 4.3, 7.5)
#: Reported group endpoints: fractional SC increase at 9 weeks.
DEBOER_ENDPOINT_FRACTION = {"FP I-III": 0.23, "FP IV-VI": 0.36}


def deboer_schedule() -> List[Tuple[float, float]]:
    """(day, SED) pairs for the de Boer 2025 protocol."""
    return [(w * 7 + d, DEBOER_SED_PER_SESSION)
            for w in range(DEBOER_DAYS // 7)
            for d in DEBOER_SESSION_WEEKDAYS]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def simulate_sc(schedule, duration_days: int, alpha_SC: float,
                SC_max: float, k_decay_SC: float,
                dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the SC state alone. Returns (t, SC).

    UV enters as an instantaneous pulse on session days, applied as the exact
    solution of the saturating rise over that pulse (the same form
    `vitd_model` uses), so the result does not depend on `dt` through the
    dosing. Between sessions SC decays first-order.
    """
    n = int(round(duration_days / dt)) + 1
    t = np.linspace(0.0, duration_days, n)
    SC = np.zeros(n)
    pending = {round(day / dt): sed for day, sed in schedule}
    for i in range(1, n):
        sc = SC[i - 1]
        # first-order turnover
        sc = sc - k_decay_SC * sc * dt
        # UV pulse: saturating rise toward SC_max
        sed = pending.get(i)
        if sed:
            sc = SC_max - (SC_max - sc) * np.exp(-alpha_SC * sed)
        SC[i] = max(0.0, sc)
    return t, SC


def endpoint_sc(alpha_SC: float, SC_max: float, k_decay_SC: float) -> float:
    """SC at day 63 under the de Boer protocol."""
    _t, SC = simulate_sc(deboer_schedule(), DEBOER_DAYS,
                         alpha_SC, SC_max, k_decay_SC)
    return float(SC[-1])


def attenuation(SC: float, K_SC: float) -> float:
    """Fraction of synthesis surviving a stratum corneum of thickness SC."""
    return float(np.exp(-K_SC * SC))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> dict:
    p = ModelParams()
    out: dict = {}

    if verbose:
        print("=" * 74)
        print("TIER-1 REFERENCE MODEL: stratum corneum only")
        print("=" * 74)
        print("  dSC/dt = alpha_SC * UV * (SC_max - SC) - k_decay_SC * SC")
        print("  attenuation = exp(-K_SC * SC)")
        print()
        print(f"  alpha_SC   = {p.alpha_SC:.4f}   [PRIOR]")
        print(f"  SC_max     = {p.SC_max:.4f}   [PRIOR, evidence band 0.20-0.40]")
        print(f"  k_decay_SC = {p.k_decay_SC:.5f}  "
              f"(half-life {np.log(2) / p.k_decay_SC:.1f} d)")
        print(f"  K_SC       = {p.K_SC:.4f}   [PRIOR, no vitamin-D measurement]")
        print(f"  k_damage   = {p.k_damage:.4f}   (pinned; see vitd_model.py)")
        print()

    # --- 1. endpoint reproduction ------------------------------------------
    t, SC = simulate_sc(deboer_schedule(), DEBOER_DAYS,
                        p.alpha_SC, p.SC_max, p.k_decay_SC)
    sc_at = {d: float(np.interp(d, t, SC)) for d in DEBOER_VISIT_DAYS}
    out["sc_at_visits"] = sc_at
    out["sc_endpoint"] = sc_at[63]

    obs0, obs1, obs2 = DEBOER_OBSERVED_UM
    obs_rel = {0: 0.0, 35: obs1 / obs0 - 1.0, 63: obs2 / obs0 - 1.0}

    if verbose:
        print("-" * 74)
        print("1. Against de Boer 2025 -- the entire dataset is three medians")
        print("-" * 74)
        print(f"  {'visit':>7} {'observed':>10} {'model':>10}")
        for d in DEBOER_VISIT_DAYS:
            print(f"  day {d:>3} {obs_rel[d]:>+10.3f} {sc_at[d]:>+10.3f}")
        print()
        print("  ENDPOINT: reproduced, and that is NOT evidence. SC_max was set")
        print("  FROM these endpoints (+23% / +36%), so agreement here is a")
        print("  restatement of the input, not a test.")
        print()
        print("  SHAPE: NOT reproduced, and cannot be. The observed SC dips")
        print(f"  {obs_rel[35]:+.0%} BELOW baseline at week 5 before overshooting.")
        print("  A saturating rise minus first-order decay cannot go below its")
        print("  own initial condition. This is a STRUCTURAL mismatch, not a")
        print("  parameter one -- no value of alpha_SC, SC_max or k_decay_SC")
        print("  fixes it. k_damage was introduced for the dip and is pinned at")
        print("  0.0 because three group medians cannot identify it.")
        print()

    # --- 2. the alpha_SC ~ SC_max ridge -------------------------------------
    target = sc_at[63]
    ridge = []
    for sc_max in (0.20, 0.25, 0.30, 0.35, 0.40):
        # solve for the alpha_SC that hits the same 9-week endpoint
        lo, hi = 1e-4, 50.0
        if endpoint_sc(hi, sc_max, p.k_decay_SC) < target:
            ridge.append((sc_max, float("nan"), float("nan")))
            continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if endpoint_sc(mid, sc_max, p.k_decay_SC) < target:
                lo = mid
            else:
                hi = mid
        a = 0.5 * (lo + hi)
        ridge.append((sc_max, a, endpoint_sc(a, sc_max, p.k_decay_SC)))
    out["ridge"] = ridge

    if verbose:
        print("-" * 74)
        print("2. The alpha_SC ~ SC_max ridge, made explicit")
        print("-" * 74)
        print("  For each SC_max, the alpha_SC that reproduces the SAME 9-week")
        print("  endpoint. Every row below fits de Boer 2025 equally well.")
        print()
        print(f"  {'SC_max':>8} {'alpha_SC':>10} {'day-63 SC':>11} "
              f"{'max attenuation':>16}")
        for sc_max, a, e in ridge:
            if np.isnan(a):
                print(f"  {sc_max:>8.2f} {'unreachable':>10} {'-':>11} {'-':>16}")
                continue
            print(f"  {sc_max:>8.2f} {a:>10.3f} {e:>11.4f} "
                  f"{attenuation(sc_max, p.K_SC):>15.3f}x")
        print()
        valid = [(m, a) for m, a, _ in ridge if not np.isnan(a)]
        if len(valid) >= 2:
            spread = max(a for _, a in valid) / min(a for _, a in valid)
            print(f"  alpha_SC spans a factor of {spread:.1f} across the SC_max")
            print("  evidence band while fitting the data identically. The data")
            print("  constrain a COMBINATION, not either parameter. Reporting")
            print("  alpha_SC as calibrated overstates what de Boer 2025 says.")
        print()

    # --- 3. K_SC is unconstrained -------------------------------------------
    if verbose:
        print("-" * 74)
        print("3. K_SC is not constrained by this dataset at all")
        print("-" * 74)
        print("  de Boer 2025 measured stratum corneum THICKNESS. It did not")
        print("  measure vitamin D. Nothing in it says how much synthesis a")
        print("  thickened SC blocks, so K_SC cannot be fitted from it.")
        print()
        print(f"  {'K_SC':>8} {'throughput at SC_max':>22}")
        for k in (0.0, 0.5, 1.0, 2.0, 4.0):
            print(f"  {k:>8.1f} {attenuation(p.SC_max, k):>21.3f}x")
        print()
        print(f"  In force: K_SC = {p.K_SC:.2f}, i.e. a fully-adapted stratum")
        print(f"  corneum passes {attenuation(p.SC_max, p.K_SC):.0%} of synthesis.")
        print("  Pearse 1987 is cited for the SC thickening itself, not for a")
        print("  vitamin-D attenuation coefficient. K_SC is a bare prior, and")
        print("  it is jointly unidentifiable with K_ATTEN (retired at 0) --")
        print("  see the vitd_model.py K_ATTEN entry.")
        print()

    # --- verdict ------------------------------------------------------------
    if verbose:
        print("-" * 74)
        print("4. Verdict")
        print("-" * 74)
        print("  This subsystem is UNDERDETERMINED and should be presented as")
        print("  such. Specifically:")
        print("    SUPPORTED     : SC thickens under repeated sub-erythemal UV,")
        print("                    and thickens more in darker phototypes.")
        print("                    Direction and rough magnitude, from de Boer.")
        print("    NOT SUPPORTED : alpha_SC and SC_max as separate calibrated")
        print("                    values (section 2 ridge).")
        print("    NOT SUPPORTED : K_SC at any value (section 3 -- wrong endpoint")
        print("                    measured entirely).")
        print("    NOT REPRODUCED: the observed week-5 dip (section 1 -- a")
        print("                    structural mismatch, not a fitting failure).")
        print()
        print("  WHAT WOULD FIX IT: individual-level SC time courses from the")
        print("  de Boer 2025 supplementary, at finer than three visits. That")
        print("  single dataset would break the alpha_SC ~ SC_max ridge. A")
        print("  separate study measuring 25(OH)D response against measured SC")
        print("  thickness would be needed for K_SC; none is known to exist.")
        print()
        print("  CONSEQUENCE FOR THE MAIN MODEL: SC is the ONLY live")
        print("  photoadaptation channel (K_ATTEN is retired at 0), so the")
        print("  entire photoadaptation story rests on the priors above.")
        print("  `spacing_study/` shows this costs little in practice -- K_SC")
        print("  would need to be roughly an order of magnitude larger before")
        print("  photoadaptation changed any scheduling conclusion -- but that")
        print("  is a statement about insensitivity, not about evidence.")
        print()
    return out


if __name__ == "__main__":
    run(verbose=True)
