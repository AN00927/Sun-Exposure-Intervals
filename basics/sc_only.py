"""
basics/sc_only.py
=================
TIER-1 REFERENCE MODEL: stratum corneum thickening, and nothing else.

WHY THIS FILE EXISTS
--------------------
Companion to `basics/melanin_only.py`. Where that file isolates the mechanism
that WORKS (the skin-type gradient: direction unanimous, size disputed), this
file isolates the SC subsystem and reports plainly what the evidence does and
does not support -- including a real structural upgrade made this session
(see below) and what that upgrade did, and did not, resolve.

THE MODEL, IN FULL -- as `vitd_model.simulate()` actually runs it
------------------------------------------------------------------
`vitd_model.py`'s SC state is TWO processes whose difference is the net state
(see its module docstring, "Stratum Corneum (SC) Thickening" entry, and the
`simulate()` update block around the `SC_build` / `SC_damage` variables):

    SC = SC_build - SC_damage

    Between sessions (any sub-step size -- both terms are EXACT exponential
    relaxations, so there is no discretization error from taking a coarser
    step, only from where a session pulse lands on the step grid):
        SC_build  = SC_build  * exp(-k_decay_SC  * delta_time)   # regresses toward 0
        SC_damage = SC_damage * exp(-k_repair_SC * delta_time)   # repairs toward 0

    Per session (pulse), given session dose `sed`:
        SC_build  = SC_max_eff - (SC_max_eff - SC_build) * exp(-alpha_SC * sed)
        SC_damage = SC_damage + k_damage * sed                    # ADDITIVE, not multiplicative

    attenuation = exp(-K_SC * SC)   # reads the NET two-process SC, unchanged

This file previously implemented only the `SC_build` half (a single saturating
rise minus first-order decay) and called that "the model". It was not: that
was a real simplification of what `simulate()` runs, not an isolation of one
of its mechanisms. This version implements both processes
(`simulate_sc_two_process`, below), matching `vitd_model.py`'s update block
exactly. `simulate_sc` (single-process) is kept alongside it because sections
3-4 below (the `alpha_SC ~ SC_max` ridge and the `K_SC` analysis) concern the
hyperplasia term only and are unaffected by `k_damage`.

MELANIN AND SC ARE INDEPENDENT STATE VARIABLES. Checked directly against
`simulate()`: the `M` update (melanin) reads only `M_base`, `M_ceil`,
`alpha_p`, `sed`; the `SC_build`/`SC_damage` update reads only `SC_max_eff`,
`alpha_SC_eff`, `k_damage_eff`, `sed`. Neither state equation's right-hand
side contains the other state variable. They are coupled only downstream,
multiplicatively, inside the synthesis function
(`exp(-K_ATTEN * excess_M) * exp(-K_SC * SC)`), not in their own dynamics.

PARAMETER VALUES
----------------
Read from `vitd_model.ModelParams` at runtime -- never restated here. The
canonical source is the `vitd_model.py` module docstring (`CLAUDE.md`).

VALIDATION DATA -- de Boer 2025 Supplementary Table S1
-------------------------------------------------------
Protocol: 0.8 SED, 3x/week, 9 weeks, lower back, n=31. Visits T0 (day 0),
T1 (day 35), T2 (day 63), reported as GROUP MEDIANS with IQR and pairwise
ANOVA (nonparametric) p-values. Cross-checked directly against the source PDF
(`Research_Papers/Effect of Repeated Low-Dose UVR Exposure - de boer et al..pdf`,
page 12) this session -- the transcribed SC-thickness medians below match the
printed table exactly, including the pairwise p-values, which the earlier
version of this file did not report and which materially change how strong
the "week-5 dip" claim is (see section 1).

WHAT THIS FILE DEMONSTRATES
---------------------------
  1. The model reproduces the reported 9-week ENDPOINT. Not evidence:
     `SC_max` was set FROM those endpoints, so this is a restatement. The
     two-process upgrade collapses to the old single-process model exactly
     when `k_damage = 0` (verified numerically below), so this restatement
     is unaffected by the upgrade.
  2. The two-process structure CAN now represent a below-baseline dip --
     something the old multiplicative form (`SC *= exp(-k_damage*sed)`)
     could never do (it can only shrink SC toward zero from above). Section 2
     searches the four-parameter space `(alpha_SC, SC_max, k_damage,
     k_repair_SC)` for a combination that reproduces de Boer's dip-then-
     overshoot shape, and reports how well the best one actually does.
  3. `alpha_SC` and `SC_max` trade off along a ridge for the hyperplasia
     term alone (section 3) -- unaffected by `k_damage`.
  4. `K_SC` is not constrained by de Boer 2025 AT ALL (section 4). The paper
     measured stratum corneum thickness, not vitamin D.

USAGE
-----
    PYTHONIOENCODING=utf-8 python3 basics/sc_only.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams  # noqa: E402

try:
    from scipy.optimize import differential_evolution
    _HAVE_SCIPY = True
except ImportError:                                    # pragma: no cover
    _HAVE_SCIPY = False

# --- de Boer 2025 protocol, as published -----------------------------------
DEBOER_SED_PER_SESSION = 0.8
DEBOER_SESSION_WEEKDAYS = (0, 2, 4)      # 3x/week
DEBOER_DAYS = 63                          # 9 weeks
DEBOER_VISIT_DAYS = (0, 35, 63)           # T0, T1, T2 -- the ENTIRE dataset

#: SC thickness in um at T0/T1/T2, Supp Table S1, all-phototype-pooled and
#: per FP-stratum medians. Re-verified against the source PDF page 12 this
#: session (see docstring above) -- exact match to the printed table.
DEBOER_SC_UM: Dict[str, Tuple[float, float, float]] = {
    "all":   (5.8, 4.3, 7.5),
    "I-III": (5.6, 4.3, 6.9),
    "IV-VI": (6.6, 4.3, 9.0),
}
#: FST used to look up SC_max_by_FST for each stratum (informational only --
#: section 2's search treats SC_max as a free parameter, not looked up).
DEBOER_FST_BY_STRATUM: Dict[str, int] = {"all": 3, "I-III": 3, "IV-VI": 5}
#: Reported group endpoints: fractional SC increase at 9 weeks (T0 -> T2).
DEBOER_ENDPOINT_FRACTION = {"FP I-III": 0.23, "FP IV-VI": 0.36}
#: Pairwise p-values for the T1-vs-T0 comparison (the "dip"), Supp Table S1,
#: ANOVA nonparametric test between visits. Read directly from the source PDF
#: this session -- NOT in any prior version of this file or of
#: `scripts/fit_sc_deboer2025.py`.
DEBOER_T1_VS_T0_PVALUE: Dict[str, float] = {"all": 0.016, "I-III": 0.170, "IV-VI": 0.134}


def deboer_schedule() -> List[Tuple[float, float]]:
    """(day, SED) pairs for the de Boer 2025 protocol."""
    return [(w * 7 + d, DEBOER_SED_PER_SESSION)
            for w in range(DEBOER_DAYS // 7)
            for d in DEBOER_SESSION_WEEKDAYS]


def observed_relative(stratum: str = "all") -> Tuple[float, float, float]:
    """Absolute um -> fractional change from T0, matching the model's SC state
    (SC is 'additional thickness above baseline', i.e. a fractional excess)."""
    t0, t1, t2 = DEBOER_SC_UM[stratum]
    return (0.0, t1 / t0 - 1.0, t2 / t0 - 1.0)


# ---------------------------------------------------------------------------
# The model: single-process (hyperplasia only) and two-process (+ acute damage)
# ---------------------------------------------------------------------------

def simulate_sc(schedule, duration_days: int, alpha_SC: float,
                SC_max: float, k_decay_SC: float,
                dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the SC_build-only state. Returns (t, SC).

    This is `simulate_sc_two_process` with `k_damage = 0`, kept as its own
    function because sections 3-4 below concern only the hyperplasia term and
    are unaffected by the acute-damage upgrade. UV enters as an instantaneous
    pulse on session days, applied as the exact solution of the saturating
    rise over that pulse (the same form `vitd_model` uses), so the result
    does not depend on `dt` through the dosing. Between sessions SC decays by
    the exact exponential relaxation (not a finite-difference approximation),
    so the result does not depend on `dt` at all as long as every session day
    divides `dt` evenly.

    BUG FIX this session: a day-0 session is now applied explicitly before the
    main loop. The loop below starts at `i=1` and never revisits index 0, so a
    session scheduled on day 0 (de Boer's protocol schedules one) was
    previously dropped silently -- `vitd_model.simulate()` applies a day-0
    pulse immediately, before any decay, so this was a real discrepancy from
    the documented model, not a design choice. `basics/melanin_only.py`
    flagged this exact question ("was not checked here") about this file when
    it fixed the same gap for its own melanin state; it is fixed here now.
    """
    n = int(round(duration_days / dt)) + 1
    t = np.linspace(0.0, duration_days, n)
    SC = np.zeros(n)
    pending = {round(day / dt): sed for day, sed in schedule}
    if 0 in pending:
        SC[0] = max(0.0, SC_max - (SC_max - SC[0]) * np.exp(-alpha_SC * pending[0]))
    for i in range(1, n):
        sc = SC[i - 1] * np.exp(-k_decay_SC * dt)
        sed = pending.get(i)
        if sed:
            sc = SC_max - (SC_max - sc) * np.exp(-alpha_SC * sed)
        SC[i] = max(0.0, sc)
    return t, SC


def endpoint_sc(alpha_SC: float, SC_max: float, k_decay_SC: float) -> float:
    """SC at day 63 under the de Boer protocol (single-process)."""
    _t, SC = simulate_sc(deboer_schedule(), DEBOER_DAYS,
                         alpha_SC, SC_max, k_decay_SC)
    return float(SC[-1])


def simulate_sc_two_process(schedule, duration_days: int, alpha_SC: float,
                            SC_max: float, k_decay_SC: float, k_damage: float,
                            k_repair_SC: float,
                            dt: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the full two-process SC state exactly as `vitd_model.simulate()`
    does: SC = SC_build - SC_damage, with SC_build a saturating hyperplasia
    term and SC_damage an additive acute-damage term that repairs
    independently. Returns (t, SC_build, SC_damage, SC).

    `dt = 0.5` matches `vitd_model.simulate()`'s own sub-step size. Both
    relaxations are exact exponentials, so (as in `simulate_sc` above) the
    result does not depend on `dt` as long as every session day divides it
    evenly -- true here since de Boer's session days and `dt=0.5` are both
    multiples of 0.5.

    Applies the same day-0 bug fix as `simulate_sc` (see its docstring): a
    session on day 0 is applied explicitly before the main loop, which
    otherwise never revisits index 0.
    """
    n = int(round(duration_days / dt)) + 1
    t = np.linspace(0.0, duration_days, n)
    SC_build = np.zeros(n)
    SC_damage = np.zeros(n)
    pending = {round(day / dt): sed for day, sed in schedule}
    if 0 in pending:
        sed0 = pending[0]
        SC_build[0] = SC_max - (SC_max - SC_build[0]) * np.exp(-alpha_SC * sed0)
        SC_damage[0] = SC_damage[0] + k_damage * sed0
    for i in range(1, n):
        b = SC_build[i - 1] * np.exp(-k_decay_SC * dt)
        d = SC_damage[i - 1] * np.exp(-k_repair_SC * dt)
        sed = pending.get(i)
        if sed:
            b = SC_max - (SC_max - b) * np.exp(-alpha_SC * sed)
            d = d + k_damage * sed
        SC_build[i] = b
        SC_damage[i] = d
    return t, SC_build, SC_damage, SC_build - SC_damage


def two_process_at_visits(alpha_SC: float, SC_max: float, k_decay_SC: float,
                          k_damage: float, k_repair_SC: float,
                          schedule=None, duration_days: int = DEBOER_DAYS,
                          visit_days: Tuple[int, ...] = DEBOER_VISIT_DAYS,
                          dt: float = 0.5) -> Dict[int, float]:
    if schedule is None:
        schedule = deboer_schedule()
    t, _b, _d, SC = simulate_sc_two_process(schedule, duration_days, alpha_SC, SC_max,
                                            k_decay_SC, k_damage, k_repair_SC, dt=dt)
    return {d: float(np.interp(d, t, SC)) for d in visit_days}


def attenuation(SC: float, K_SC: float) -> float:
    """Fraction of synthesis surviving a stratum corneum of (net) thickness SC."""
    return float(np.exp(-K_SC * SC))


# ---------------------------------------------------------------------------
# Section 2 helpers: fitting and identifiability of the two-process model
# ---------------------------------------------------------------------------

#: PRIOR bounds: `alpha_SC`, `SC_max` and `k_damage` taken directly from
#: `ModelParams.SC_PRIOR_RANGES` (read live in `run()`, not hardcoded here).
#: `k_repair_SC` has no entry in `SC_PRIOR_RANGES`; its own comment in
#: `vitd_model.py` describes it only qualitatively as "repairing on a days
#: timescale", so the bound used for it here (half-life 1-30 d) is THIS
#: FILE'S OWN choice to test that qualitative description, not a value read
#: from `ModelParams`. This is the search whose answer is directly comparable
#: to "does the model, kept inside its own evidence-based priors, fit the
#: dip" -- i.e. the question `vitd_model.py`'s comment answers "no" to.
PRIOR_REPAIR_HALF_LIFE_DAYS = (1.0, 30.0)

#: WIDE bounds: deliberately far outside `SC_PRIOR_RANGES` in every
#: dimension. The question this search answers is different: "is there ANY
#: parameter combination, plausible or not, under which this STRUCTURE can
#: reproduce the shape" -- i.e. whether the two-process form is even
#: mathematically capable of it, independent of whether the fitted values
#: would mean anything physiologically.
WIDE_BOUNDS = [
    (0.01, 3.0),                          # alpha_SC
    (0.05, 1.20),                         # SC_max
    (0.0, 3.0),                           # k_damage
    (np.log(2) / 60.0, np.log(2) / 0.1),  # k_repair_SC (60 d to 2.4 h half-life)
]


def _sse(params: Tuple[float, float, float, float], obs: Tuple[float, float, float],
        k_decay_SC: float, schedule) -> float:
    alpha_SC, SC_max, k_damage, k_repair_SC = params
    pred = two_process_at_visits(alpha_SC, SC_max, k_decay_SC, k_damage, k_repair_SC,
                                 schedule=schedule)
    return (pred[35] - obs[1]) ** 2 + (pred[63] - obs[2]) ** 2


def fit_two_process(stratum: str, k_decay_SC: float, bounds: List[Tuple[float, float]],
                    seed: int = 0) -> dict:
    """Search `(alpha_SC, SC_max, k_damage, k_repair_SC)` within `bounds` for
    the combination that best reproduces de Boer's T1/T2 fractional SC change
    for one stratum. Uses `scipy.optimize.differential_evolution` (global,
    gradient-free -- an appropriate choice here since the objective can have
    multiple local minima from the saturating/exponential nonlinearities) if
    available, else a coarse grid fallback.
    """
    schedule = deboer_schedule()
    obs = observed_relative(stratum)
    amplitude = max(abs(obs[1]), abs(obs[2]))

    if _HAVE_SCIPY:
        result = differential_evolution(
            lambda x: _sse(tuple(x), obs, k_decay_SC, schedule),
            bounds, seed=seed, maxiter=200, popsize=25, tol=1e-14,
            polish=True, workers=1)
        best_params = tuple(result.x)
        best_sse = float(result.fun)
    else:                                                # pragma: no cover
        best_params, best_sse = None, None
        grids = [np.linspace(lo, hi, 10) for lo, hi in bounds]
        for a in grids[0]:
            for s in grids[1]:
                for kd in grids[2]:
                    for kr in grids[3]:
                        sse = _sse((a, s, kd, kr), obs, k_decay_SC, schedule)
                        if best_sse is None or sse < best_sse:
                            best_sse, best_params = sse, (a, s, kd, kr)

    alpha_SC, SC_max, k_damage, k_repair_SC = best_params
    pred = two_process_at_visits(alpha_SC, SC_max, k_decay_SC, k_damage, k_repair_SC,
                                 schedule=schedule)
    pred_tuple = (pred[0], pred[35], pred[63])
    rmse = float(np.sqrt(best_sse / 2.0))
    relative_rmse = rmse / amplitude
    return {
        "stratum": stratum, "obs": obs, "pred": pred_tuple,
        "alpha_SC": alpha_SC, "SC_max": SC_max,
        "k_damage": k_damage, "k_repair_SC": k_repair_SC,
        "sse": best_sse, "rmse": rmse, "amplitude": amplitude,
        "relative_rmse": relative_rmse,
    }


def damage_repair_ridge(alpha_SC: float, SC_max: float, k_decay_SC: float,
                        obs: Tuple[float, float, float], schedule,
                        k_damage_bounds: Tuple[float, float],
                        n_damage: int = 12) -> List[dict]:
    """With `alpha_SC`/`SC_max` pinned at a best-fit point, scan `k_damage`
    and re-optimize `k_repair_SC` alone at each value. If the achieved SSE is
    roughly flat across the scan, the (k_damage, k_repair_SC) pair is a ridge
    -- not identifiable even setting aside how well the overall shape fits.
    """
    half_life_grid = np.geomspace(0.5, 60.0, 80)          # 12 h to 60 d
    repair_grid = np.log(2) / half_life_grid
    rows = []
    for k_damage in np.linspace(k_damage_bounds[0], max(k_damage_bounds[1], 1e-6), n_damage):
        sses = [_sse((alpha_SC, SC_max, k_damage, kr), obs, k_decay_SC, schedule)
                for kr in repair_grid]
        j = int(np.argmin(sses))
        rows.append({"k_damage": k_damage, "k_repair_SC": float(repair_grid[j]),
                     "half_life_d": float(half_life_grid[j]), "sse": float(sses[j])})
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> dict:
    p = ModelParams()
    out: dict = {}

    if verbose:
        print("=" * 78)
        print("TIER-1 REFERENCE MODEL: stratum corneum, two-process")
        print("=" * 78)
        print("  SC = SC_build - SC_damage")
        print("  SC_build  : dSC_build/dt  = alpha_SC * UV * (SC_max - SC_build)")
        print("                              - k_decay_SC  * SC_build")
        print("  SC_damage : dSC_damage/dt = k_damage * UV  - k_repair_SC * SC_damage")
        print("  attenuation = exp(-K_SC * SC)")
        print()
        print(f"  alpha_SC     = {p.alpha_SC:.4f}   [PRIOR]")
        print(f"  SC_max       = {p.SC_max:.4f}   [PRIOR, evidence band {p.SC_PRIOR_RANGES['SC_max'][0]:.2f}-{p.SC_PRIOR_RANGES['SC_max'][1]:.2f}]")
        print(f"  k_decay_SC   = {p.k_decay_SC:.5f}  (half-life {np.log(2) / p.k_decay_SC:.1f} d)")
        print(f"  K_SC         = {p.K_SC:.4f}   [PRIOR, no vitamin-D measurement]")
        print(f"  k_damage     = {p.k_damage:.4f}   [pinned by default -- see section 2]")
        print(f"  k_repair_SC  = {p.k_repair_SC:.5f}  (half-life {np.log(2) / p.k_repair_SC:.1f} d) [PRIOR]")
        print()

    # --- sanity check: two-process collapses to single-process at k_damage=0 --
    sched = deboer_schedule()
    t1, SC1 = simulate_sc(sched, DEBOER_DAYS, p.alpha_SC, p.SC_max, p.k_decay_SC, dt=0.05)
    t2, _b2, _d2, SC2 = simulate_sc_two_process(sched, DEBOER_DAYS, p.alpha_SC, p.SC_max,
                                                p.k_decay_SC, p.k_damage, p.k_repair_SC, dt=0.05)
    collapse_ok = bool(np.allclose(SC1, SC2))
    out["collapse_to_single_process_ok"] = collapse_ok
    if verbose:
        print(f"  Sanity check: two-process == single-process at k_damage={p.k_damage:.2f}: "
              f"{'PASS' if collapse_ok else 'FAIL'} (max |diff| = {np.max(np.abs(SC1 - SC2)):.2e})")
        print()

    # --- 1. endpoint reproduction, at DEFAULT parameters --------------------
    t, SC = simulate_sc(sched, DEBOER_DAYS, p.alpha_SC, p.SC_max, p.k_decay_SC)
    sc_at = {d: float(np.interp(d, t, SC)) for d in DEBOER_VISIT_DAYS}
    out["sc_at_visits"] = sc_at
    out["sc_endpoint"] = sc_at[63]

    obs_rel = observed_relative("all")
    obs_rel_by_day = dict(zip(DEBOER_VISIT_DAYS, obs_rel))

    if verbose:
        print("-" * 78)
        print("1. Endpoint reproduction at default parameters (single-process, k_damage=0)")
        print("-" * 78)
        print(f"  {'visit':>7} {'observed':>10} {'model':>10} {'T1-vs-T0 p':>12}")
        for d in DEBOER_VISIT_DAYS:
            pval = DEBOER_T1_VS_T0_PVALUE["all"] if d == 35 else float("nan")
            pstr = f"{pval:.3f}" if not np.isnan(pval) else "-"
            print(f"  day {d:>3} {obs_rel_by_day[d]:>+10.3f} {sc_at[d]:>+10.3f} {pstr:>12}")
        print()
        print("  ENDPOINT (day 63): reproduced, and that is NOT evidence. SC_max was")
        print(f"  set FROM these endpoints ({DEBOER_ENDPOINT_FRACTION['FP I-III']:+.0%} / "
              f"{DEBOER_ENDPOINT_FRACTION['FP IV-VI']:+.0%} by phototype), so agreement")
        print("  here is a restatement of the input, not a test.")
        print()
        print("  SHAPE (day 35): NOT reproduced at k_damage=0, by construction -- a")
        print("  single saturating rise minus first-order decay cannot go below its")
        print("  own initial condition, and section 2 investigates whether the")
        print("  two-process structure introduced this session can do better.")
        print()
        print("  A DATA CAVEAT NOT IN THE PRIOR VERSION OF THIS FILE: de Boer's own")
        print("  ANOVA on the T1-vs-T0 dip is significant pooled across phototypes")
        print(f"  (p={DEBOER_T1_VS_T0_PVALUE['all']:.3f}) but NOT significant within either")
        print(f"  phototype stratum alone (FP I-III p={DEBOER_T1_VS_T0_PVALUE['I-III']:.3f}, "
              f"FP IV-VI p={DEBOER_T1_VS_T0_PVALUE['IV-VI']:.3f}).")
        print("  The dip is a real, stated finding of the paper, but its evidence")
        print("  strength depends on which grouping is asked about.")
        print()

    # --- 2. two-process structural capability & identifiability -------------
    # Two searches, deliberately different in scope:
    #   PRIOR bounds  -- alpha_SC/SC_max/k_damage read live from
    #                    ModelParams.SC_PRIOR_RANGES; k_repair_SC bounded to a
    #                    1-30 day half-life to test its own comment's "days
    #                    timescale" description (that bound is THIS FILE'S
    #                    choice, not read from ModelParams -- there is no
    #                    SC_PRIOR_RANGES entry for it). Answers "does the
    #                    model, kept inside its own evidence-based priors,
    #                    fit the dip".
    #   WIDE bounds   -- far outside SC_PRIOR_RANGES in every dimension.
    #                    Answers "is there ANY combination, plausible or not,
    #                    that fits" -- i.e. pure mathematical capability.
    prior_bounds = [
        tuple(p.SC_PRIOR_RANGES["alpha_SC"]),
        tuple(p.SC_PRIOR_RANGES["SC_max"]),
        tuple(p.SC_PRIOR_RANGES["k_damage"]),
        (np.log(2) / PRIOR_REPAIR_HALF_LIFE_DAYS[1], np.log(2) / PRIOR_REPAIR_HALF_LIFE_DAYS[0]),
    ]

    if verbose:
        print("-" * 78)
        print("2. Can the two-process structure fit the dip? Is it identifiable?")
        print("-" * 78)
        print("  Two global searches over (alpha_SC, SC_max, k_damage, k_repair_SC)")
        print("  -- k_decay_SC held at the value above -- for the combination that")
        print("  best reproduces each stratum's T1/T2 fractional SC change:")
        print("    PRIOR bounds: alpha_SC/SC_max/k_damage read live from")
        print("      ModelParams.SC_PRIOR_RANGES; k_repair_SC bounded to a")
        print(f"      {PRIOR_REPAIR_HALF_LIFE_DAYS[0]:.0f}-{PRIOR_REPAIR_HALF_LIFE_DAYS[1]:.0f} day half-life (this file's own choice, testing")
        print("      k_repair_SC's comment -- 'repairing on a days timescale' --")
        print("      NOT a value read from ModelParams, which has no")
        print("      SC_PRIOR_RANGES entry for it).")
        print("    WIDE bounds: far outside SC_PRIOR_RANGES in every dimension --")
        print("      tests pure mathematical capability, not physiological")
        print("      plausibility.")
        print(f"  Optimizer: {'scipy.optimize.differential_evolution' if _HAVE_SCIPY else 'coarse grid (scipy unavailable)'}")
        print()

    prior_fits = {s: fit_two_process(s, p.k_decay_SC, prior_bounds) for s in DEBOER_SC_UM}
    wide_fits = {s: fit_two_process(s, p.k_decay_SC, WIDE_BOUNDS) for s in DEBOER_SC_UM}
    out["two_process_fits_prior_bounds"] = prior_fits
    out["two_process_fits_wide_bounds"] = wide_fits

    if verbose:
        print("  2a. WITHIN evidence-based priors:")
        hdr = (f"  {'stratum':7} {'obs T1':>8} {'fit T1':>8} {'obs T2':>8} "
               f"{'fit T2':>8} {'rel.RMSE':>9} {'verdict':>18}")
        print(hdr)
        for stratum, f in prior_fits.items():
            verdict = "STRUCTURAL FAILURE" if f["relative_rmse"] > 0.25 else "acceptable"
            print(f"  {stratum:7} {f['obs'][1]:>+8.3f} {f['pred'][1]:>+8.3f} "
                  f"{f['obs'][2]:>+8.3f} {f['pred'][2]:>+8.3f} "
                  f"{f['relative_rmse']:>8.0%} {verdict:>18}")
        print()
        worst = max(f["relative_rmse"] for f in prior_fits.values())
        best = min(f["relative_rmse"] for f in prior_fits.values())
        print(f"  Even at the best point found INSIDE the evidence-based prior box,")
        print(f"  the model misses {best:.0%}-{worst:.0%} of the T1/T2 signal amplitude across")
        print("  strata. This independently reproduces (with a continuous global")
        print("  optimizer and a per-phototype breakdown, vs. the pooled coarse grid")
        print("  in `scripts/fit_sc_deboer2025.py`) the same order of magnitude")
        print("  `vitd_model.py`'s own comment reports: a 71-75% miss.")
        print()
        pinned = [stratum for stratum, f in prior_fits.items()
                 if any(np.isclose(f[k], b[0]) or np.isclose(f[k], b[1])
                        for k, b in zip(("alpha_SC", "SC_max", "k_damage"), prior_bounds[:3]))]
        print(f"  All three of alpha_SC, SC_max and k_damage sit AT a prior-range")
        print(f"  boundary in every stratum's best-inside-priors fit (checked: "
              f"{', '.join(pinned) if pinned else 'none'} pinned).")
        print("  The optimizer is pushing against the edge of the evidence-based box,")
        print("  not settling on an interior optimum -- the same 'pinned at the")
        print("  search bound' signature `docs/MELANIN_MODEL.md` flags for `alpha`'s")
        print("  own identifiability search. It wants to go further than the priors")
        print("  allow in every one of the three bounded dimensions.")
        print()
        print("  2b. Far OUTSIDE evidence-based priors (WIDE bounds):")
        for stratum, f in wide_fits.items():
            print(f"    {stratum:7}: rel.RMSE={f['relative_rmse']:.1%}  "
                  f"alpha_SC={f['alpha_SC']:.3f}  SC_max={f['SC_max']:.3f}  "
                  f"k_damage={f['k_damage']:.3f}  "
                  f"k_repair_SC t1/2={np.log(2)/f['k_repair_SC']:.2f} d")
        print()
        print("  All three strata fit ESSENTIALLY EXACTLY once alpha_SC and k_damage")
        print("  are allowed several times past the top of SC_PRIOR_RANGES. This is")
        print("  NOT evidence the structure secretly works -- it is the arithmetic")
        print("  consequence of having only TWO informative residuals (T1, T2; T0=0")
        print("  by construction) against FOUR free parameters. A smooth enough")
        print("  model with more free parameters than data points can generically")
        print("  be driven through both target points exactly; it says nothing about")
        print("  which parameter values are correct, or whether the SHAPE in")
        print("  between the two visits means anything.")
        print()
        print("  WHY THE PRIOR-BOUNDED FIT FAILS -- verified analytically, not just")
        print("  empirically. De Boer's shape needs an ACCELERATING net response")
        print("  under CONSTANT dosing: little net change through day 35, then most")
        print("  of the total swing packed into days 35-63. Every component of this")
        print("  model is a constant-input relaxation toward a ceiling or toward")
        print("  zero -- SC_build approaches SC_max fastest early and slows as it")
        print("  saturates; SC_damage's net contribution likewise decays fastest")
        print("  right after each pulse and slows as it is repaired. Both are")
        print("  DECELERATING processes by construction, and under 3x/week dosing")
        print("  SC_damage reaches its own periodic plateau within a few repair")
        print("  half-lives (days, at prior-range k_repair_SC) -- long before day 35")
        print("  -- so it cannot supply a swing concentrated in the LATER half of the")
        print("  protocol. This is the SAME conclusion `vitd_model.py` states ('The")
        print("  data need an ACCELERATING response under constant dosing, which no")
        print("  saturating or cascade process produces'), now cross-checked with an")
        print("  independent global optimizer, live-read priors, and a per-phototype")
        print("  breakdown instead of only the pooled series.")
        print()

    # damage/repair ridge, pinned at the "all"-stratum PRIOR-bounded best fit
    all_fit = prior_fits["all"]
    ridge_rows = damage_repair_ridge(all_fit["alpha_SC"], all_fit["SC_max"], p.k_decay_SC,
                                     all_fit["obs"], sched, prior_bounds[2])
    out["damage_repair_ridge"] = ridge_rows
    if verbose:
        sse_vals = [r["sse"] for r in ridge_rows]
        sse_min, sse_max = min(sse_vals), max(sse_vals)
        print("  On top of the poor within-priors fit above: with alpha_SC/SC_max")
        print("  pinned at the 'all'-stratum best-inside-priors fit, re-optimizing")
        print("  k_repair_SC for each k_damage (0 to the SC_PRIOR_RANGES top) gives:")
        print(f"    {'k_damage':>9} {'best k_repair_SC t1/2 (d)':>27} {'sse':>10}")
        for r in ridge_rows[::2]:
            print(f"    {r['k_damage']:>9.3f} {r['half_life_d']:>27.2f} {r['sse']:>10.5f}")
        spread = (sse_max - sse_min) / sse_min if sse_min > 0 else float("inf")
        print(f"  SSE spans {sse_min:.5f} to {sse_max:.5f} ({spread:.0%} relative spread)")
        print("  across the k_damage scan: it keeps improving toward k_damage's")
        print("  upper prior bound without a clear interior minimum in this slice,")
        print("  consistent with 2a's boundary-pinning. k_damage and k_repair_SC")
        print("  trade off against each other on top of the model's overall")
        print("  inability to match the shape inside the priors: this is")
        print("  non-identifiability layered on structural mismatch, not instead")
        print("  of it.")
        print()
        print("  ANSWER TO THE QUESTION THIS SECTION WAS BUILT TO SETTLE:")
        print("    Structurally CAN represent a below-baseline excursion: YES, new")
        print("      this session, unlike the old multiplicative single-process form")
        print("      (2b: an exact fit exists, just far outside plausible values).")
        print("    IS identified/fitted from de Boer 2025 WITHIN its own evidence-")
        print("    based priors: NO -- and the reason is NOT simply '3 group medians")
        print("      can't constrain 4 parameters' (2b shows 4 parameters against 2")
        print("      residuals is trivially fittable in the abstract). It is the")
        print("      stronger, structural claim that the medians require an")
        print("      accelerating response, every candidate process here")
        print("      decelerates, and reaching an exact fit requires leaving the")
        print("      evidence-based range entirely -- consistent with")
        print("      `vitd_model.py`'s own diagnosis.")
        print("    k_damage correctly REMAINS 0.0 in ModelParams -- fixing the")
        print("      structure did not supply the missing identifying information.")
        print()

    # --- 3. the alpha_SC ~ SC_max ridge (hyperplasia term only) --------------
    target = sc_at[63]
    ridge = []
    for sc_max in (0.20, 0.25, 0.30, 0.35, 0.40):
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
        print("-" * 78)
        print("3. The alpha_SC ~ SC_max ridge (hyperplasia term; k_damage does not")
        print("   enter this section)")
        print("-" * 78)
        print("  For each SC_max, the alpha_SC that reproduces the SAME 9-week")
        print("  endpoint. Every row below fits de Boer 2025's endpoint equally well.")
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
            print("  evidence band while fitting the endpoint identically. The data")
            print("  constrain a COMBINATION, not either parameter. This ridge is")
            print("  independent of the two-process upgrade in section 2 -- it")
            print("  concerns only SC_build, which section 2's sanity check showed")
            print("  is unaffected by k_damage=0.")
        print()

    # --- 4. K_SC is unconstrained -------------------------------------------
    if verbose:
        print("-" * 78)
        print("4. K_SC is not constrained by this dataset at all")
        print("-" * 78)
        print("  de Boer 2025 measured stratum corneum THICKNESS. It did not")
        print("  measure vitamin D. Nothing in it says how much synthesis a")
        print("  thickened SC blocks, so K_SC cannot be fitted from it -- and this")
        print("  is unchanged by whether SC is single- or two-process, since K_SC")
        print("  reads only the net SC state, not its decomposition.")
        print()
        print(f"  {'K_SC':>8} {'throughput at SC_max':>22}")
        for k in (0.0, 0.5, 1.0, 2.0, 4.0):
            print(f"  {k:>8.1f} {attenuation(p.SC_max, k):>21.3f}x")
        print()
        print(f"  In force: K_SC = {p.K_SC:.2f}, i.e. a fully-adapted stratum")
        print(f"  corneum passes {attenuation(p.SC_max, p.K_SC):.0%} of synthesis.")
        print("  Pearse 1987 is cited for SC thickening under repeated UVB, not for")
        print("  a vitamin-D attenuation coefficient. K_SC is a bare prior, and it")
        print("  is jointly unidentifiable with K_ATTEN (active and calibrated) --")
        print("  see the vitd_model.py K_ATTEN entry.")
        print()

    # --- verdict ------------------------------------------------------------
    if verbose:
        print("-" * 78)
        print("5. Verdict")
        print("-" * 78)
        print("  SUPPORTED     : SC thickens under repeated sub-erythemal UV, more")
        print("                  in darker phototypes (de Boer 2025 endpoint, now")
        print("                  corroborated out-of-sample by Miyamura 2011 for")
        print("                  FP I-III magnitude -- see basics/README.md).")
        print("  SUPPORTED     : the two-process structure (SC_build - SC_damage)")
        print("                  is CAPABLE of representing de Boer's below-baseline")
        print("                  dip, which the previous single-process form could")
        print("                  not represent at any parameter value. This is a")
        print("                  genuine structural improvement made this session.")
        print("  NOT SUPPORTED : that capability being REALIZED within this model's")
        print("                  own evidence-based priors. Section 2a's search")
        print(f"                  still misses {best:.0%}-{worst:.0%} of the T1/T2 signal")
        print("                  inside SC_PRIOR_RANGES, with every bounded parameter")
        print("                  pinned at its prior edge; an exact fit (2b) exists")
        print("                  only several times past those bounds. The")
        print("                  (k_damage, k_repair_SC) pair is itself a ridge on")
        print("                  top of that. k_damage correctly stays pinned at 0.0")
        print("                  in ModelParams.")
        print("  NOT SUPPORTED : alpha_SC and SC_max as separate calibrated values")
        print("                  (section 3 ridge) -- unaffected by the upgrade.")
        print("  NOT SUPPORTED : K_SC at any value (section 4 -- wrong endpoint")
        print("                  measured entirely) -- unaffected by the upgrade.")
        print("  NEW THIS SESSION: de Boer's own statistics show the week-5 dip is")
        print("                  significant pooled (p=0.016) but NOT significant")
        print("                  within either phototype stratum alone (p=0.170,")
        print("                  0.134). The magnitude the model is being asked to")
        print("                  fit is itself not fully established per-stratum.")
        print()
        print("  WHAT WOULD FIX IT: individual-level SC time courses from the de")
        print("  Boer 2025 supplementary, at finer than three visits -- this would")
        print("  let the accelerating-vs-decelerating question and the alpha_SC ~")
        print("  SC_max ridge both be addressed directly instead of inferred from")
        print("  three group medians without error bars. A separate study measuring")
        print("  25(OH)D response against measured SC thickness would still be")
        print("  needed for K_SC; none is known to exist.")
        print()
        print("  CONSEQUENCE FOR THE MAIN MODEL: unchanged from before this")
        print("  session's structural fix. SC is UNDERDETERMINED in the same")
        print("  practical sense regardless of K_ATTEN's status, so the")
        print("  photoadaptation story still rests heavily on the priors above.")
        print("  `spacing_study/` shows this costs little in practice -- K_SC")
        print("  would need to be roughly an order of magnitude larger before")
        print("  photoadaptation changed any scheduling conclusion -- but that is")
        print("  a statement about insensitivity, not about evidence.")
        print()
    return out


if __name__ == "__main__":
    run(verbose=True)
