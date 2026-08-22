"""
spacing_model.py
================
Thin, risk-free wrapper around the repo's forward model for the *spacing*
sub-study.

WHAT THIS IS
------------
Given a UV schedule (day indices + SED per session), an FST, a horizon and a
starting 25(OH)D, simulate and return outcome metrics. Nothing else.

WHAT THIS DELIBERATELY STRIPS OUT
---------------------------------
  * skin-cancer / erythema / CPD risk (``simulate`` still computes a
    ``cancer_risk`` sidecar internally; this module ignores it entirely)
  * location, latitude, solar geometry, season, cloud, time-of-day
  * behaviour adjustment (shade, clothing, time-outdoors multipliers)
  * MED-fractional dosing and any dose cap

Every session is assumed to deliver the same *absolute* SED under one fixed
ambient condition, on any day of the year.

PARAMETER VALUES: this module does not restate any scientific parameter.
The canonical source is the ``vitd_model.py`` module docstring. Everything
here either uses the shipped defaults or applies a clearly-labelled
*sensitivity override* that is never adopted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import vitd_model as vm

# ---------------------------------------------------------------------------
# STATED ASSUMPTIONS -- module-level constants, change these in one place.
# ---------------------------------------------------------------------------

#: Spectral weighting. The repo's solar default (see ``SOURCE_CALIBRATION_TABLE``
#: in vitd_model.py). Used for every session, every day.
E_SOURCE_NAME: str = vm.E_SOURCE_SOLAR_DEFAULT

#: Fraction of body surface exposed at every session. Set to the model's own
#: reference BSA so ``bsa_factor`` == 1.0 and no BSA scaling confounds the
#: spacing comparison.
EXPOSED_BSA: float = vm.BSA_REF

#: Starting serum 25(OH)D, nmol/L. Chosen as a mildly-insufficient adult
#: baseline so the biphasic feedback term is neither saturated nor idle at t=0.
D0_DEFAULT: float = 40.0

#: Sufficiency threshold for the "fraction of days above" metric, nmol/L.
SUFFICIENCY_THRESHOLD: float = 50.0

#: FIXED AMBIENT UV -> dose conversion for experiment C only.
#: One constant erythemal SED per minute of exposure, identical every day.
#: 0.09 SED/min corresponds to a horizontal ambient erythemal irradiance of
#: ~150 mW/m^2 (UV index 6), i.e. 1 SED in ~11 minutes. This is a STATED
#: ASSUMPTION, not a modelled quantity -- the solar-geometry chain is
#: deliberately not used.
SED_PER_MINUTE: float = 0.09

#: Default simulation horizons, days.
HORIZON_SHORT: int = 90
HORIZON_LONG: int = 365
#: Long horizon used for STEADY-STATE metrics. The state-dependent serum
#: half-life is of order months, so a 365-day run is still on its transient.
#: Metrics averaged over the final full schedule period of a 728-day run are
#: phase-invariant to <0.001 nmol/L (verified: rotating a 14-day on/off
#: pattern changes the last-period mean by 3e-4). Without this, "clustered
#: beats spread" appears purely because a burst schedule front-loads its
#: doses inside each period during the transient.
HORIZON_SS: int = 728


# ---------------------------------------------------------------------------
# Parameter construction, including mechanism ablations
# ---------------------------------------------------------------------------

@dataclass
class _NoDecayParams(vm.ModelParams):
    """ModelParams with serum 25(OH)D elimination switched off (ablation only)."""

    @staticmethod
    def k_decay_D(D: float) -> float:  # noqa: N802 - matches parent
        return 0.0


def make_params(no_D_decay: bool = False, **overrides) -> vm.ModelParams:
    """Build a ModelParams for this sub-study.

    Uses shipped defaults for every scientific parameter except the two
    STATED ASSUMPTIONS above (E_source, exposed_bsa). ``overrides`` are applied
    afterwards and are for SENSITIVITY/ABLATION ONLY -- nothing here is a
    calibration.
    """
    cls = _NoDecayParams if no_D_decay else vm.ModelParams
    p = cls()
    p.E_source = vm.get_E_source(E_SOURCE_NAME)
    p.exposed_bsa = EXPOSED_BSA
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


#: Mechanism ablation variants for experiment E.
#:
#: Each entry maps a variant name to (params_overrides, simulate_kwargs, note).
#: ``alpha_override=0`` pins melanin M at its FST baseline for the whole run;
#: ``alpha_SC_override=0`` pins the stratum-corneum build term at SC0 = 0.
#: Feedback is disabled by selecting the LEGACY 'linear' mode with an
#: effectively-infinite ceiling, which makes g(D) == 1 to machine precision.
MECHANISM_VARIANTS: Dict[str, Tuple[dict, dict, str]] = {
    "full": ({}, {}, "shipped defaults"),
    "no_photoadapt": (
        {}, {"alpha_override": 0.0, "alpha_SC_override": 0.0},
        "M pinned at FST baseline, SC pinned at 0",
    ),
    "no_SC": (
        {}, {"alpha_SC_override": 0.0},
        "stratum corneum pinned at 0; melanin dynamics intact",
    ),
    "no_melanin_adapt": (
        {}, {"alpha_override": 0.0},
        "M pinned at FST baseline; SC dynamics intact",
    ),
    "no_feedback": (
        {"feedback_mode": "linear"}, {"D_max_override": 1e12},
        "g(D) == 1",
    ),
    "no_D_decay": (
        {"__no_D_decay__": True}, {},
        "serum elimination k_decay_D(D) == 0",
    ),
    "no_photoadapt_no_feedback": (
        {"feedback_mode": "linear"},
        {"alpha_override": 0.0, "alpha_SC_override": 0.0, "D_max_override": 1e12},
        "both photoadaptation channels and feedback off",
    ),
}


def params_and_kwargs_for_variant(variant: str) -> Tuple[vm.ModelParams, dict]:
    """Return (params, simulate_kwargs) for a named mechanism variant."""
    if variant not in MECHANISM_VARIANTS:
        raise KeyError(f"unknown mechanism variant {variant!r}")
    p_over, sim_kwargs, _note = MECHANISM_VARIANTS[variant]
    p_over = dict(p_over)
    no_decay = p_over.pop("__no_D_decay__", False)
    return make_params(no_D_decay=no_decay, **p_over), dict(sim_kwargs)


# ---------------------------------------------------------------------------
# Schedule builders
# ---------------------------------------------------------------------------

def periodic_schedule(horizon: int, sed: float, interval: int,
                      start_day: int = 0) -> List[Tuple[float, float]]:
    """Every ``interval`` days, ``sed`` SED per session."""
    return [(float(d), float(sed))
            for d in range(start_day, horizon, interval)]


def burst_schedule(horizon: int, sed: float, days_on: int, days_off: int,
                   start_day: int = 0) -> List[Tuple[float, float]]:
    """``days_on`` consecutive exposure days, then ``days_off`` rest days."""
    period = days_on + days_off
    sched = []
    d = start_day
    while d < horizon:
        phase = (d - start_day) % period
        if phase < days_on:
            sched.append((float(d), float(sed)))
        d += 1
    return sched


def pattern_schedule(horizon: int, sed: float,
                     pattern: Sequence[int]) -> List[Tuple[float, float]]:
    """Tile a binary on/off ``pattern`` across the horizon."""
    P = len(pattern)
    return [(float(d), float(sed))
            for d in range(horizon) if pattern[d % P]]


def budget_schedule(horizon: int, total_sed: float,
                    interval: int) -> List[Tuple[float, float]]:
    """Split a fixed TOTAL SED budget evenly over every-``interval``-day sessions."""
    days = list(range(0, horizon, interval))
    if not days:
        return []
    per = total_sed / len(days)
    return [(float(d), per) for d in days]


def minutes_schedule(horizon: int, minutes_per_week: float,
                     interval: int) -> List[Tuple[float, float]]:
    """Fixed minutes-outdoors per week, split over every-``interval``-day sessions.

    Uses the single fixed SED_PER_MINUTE rate declared above.
    """
    sessions_per_week = 7.0 / interval
    minutes_per_session = minutes_per_week / sessions_per_week
    sed = minutes_per_session * SED_PER_MINUTE
    return periodic_schedule(horizon, sed, interval)


# ---------------------------------------------------------------------------
# Simulation + metrics
# ---------------------------------------------------------------------------

def _daily_series(t: np.ndarray, D: np.ndarray, horizon: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample D onto integer days, taking the POST-pulse value on pulse days."""
    days = np.arange(0.0, horizon + 1.0, 1.0)
    idx = np.searchsorted(t, days, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)
    return days, D[idx]


def _trapz(y, x):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def run_schedule(fst: int,
                 schedule: List[Tuple[float, float]],
                 horizon: int,
                 params: Optional[vm.ModelParams] = None,
                 D0: float = D0_DEFAULT,
                 sim_kwargs: Optional[dict] = None,
                 dose_framing: str = "fixed_per_session_SED",
                 tail_frac: float = 0.25,
                 period_days: Optional[int] = None) -> Dict[str, float]:
    """Simulate one schedule and return outcome metrics.

    Returns a flat dict. Risk outputs from the underlying model are discarded.
    """
    if params is None:
        params = make_params()
    sim_kwargs = dict(sim_kwargs or {})
    sim_kwargs.setdefault("exposed_bsa_override", params.exposed_bsa)

    res = vm.simulate(fst=fst, schedule=schedule, duration_days=horizon,
                      params=params, D0=D0, dose_framing=dose_framing,
                      **sim_kwargs)

    t, D = res["t"], res["D"]
    if not np.all(np.isfinite(D)):
        raise FloatingPointError("non-finite D in trajectory")
    if np.min(D) < 0.0:
        raise FloatingPointError(f"negative D ({np.min(D)}) -- model misuse")

    days, Dd = _daily_series(t, D, horizon)
    tail_start = horizon * (1.0 - tail_frac)
    tail_mask_t = t >= tail_start
    tail_mask_d = days >= tail_start

    auc = _trapz(D, t)
    n_sessions = len(schedule)
    cum_sed = res["cumulative_SED"]

    # Steady-state, phase-invariant metrics over the final full schedule period.
    ss = {}
    if period_days:
        P = float(period_days)
        m_t = t >= horizon - P
        m_d = days >= horizon - P
        ss = {
            "ss_mean_D": _trapz(D[m_t], t[m_t]) / P,
            "ss_min_D": float(np.min(D[m_t])),
            "ss_max_D": float(np.max(D[m_t])),
            "ss_frac_days_ge_50": float(np.mean(Dd[m_d] >= SUFFICIENCY_THRESHOLD)),
            "ss_period_days": float(P),
        }

    return {
        "fst": fst,
        "horizon_days": horizon,
        "n_sessions": n_sessions,
        "sed_per_session": (schedule[0][1] if schedule else 0.0),
        "cum_SED": cum_sed,
        "D0": D0,
        # --- outcome metrics (experiment G reports all of them) -------------
        "final_D": float(D[-1]),
        "mean_D": auc / horizon,
        "auc_D": auc,
        "min_D": float(np.min(D)),
        "min_D_tail": float(np.min(D[tail_mask_t])),
        "max_D": float(np.max(D)),
        "frac_days_ge_50": float(np.mean(Dd >= SUFFICIENCY_THRESHOLD)),
        "frac_days_ge_50_tail": float(np.mean(Dd[tail_mask_d] >= SUFFICIENCY_THRESHOLD)),
        "mean_D_tail": float(np.mean(D[tail_mask_t])),
        # --- state diagnostics ---------------------------------------------
        "final_M": float(res["M"][-1]),
        "final_SC": float(res["SC"][-1]),
        "mean_SC_tail": float(np.mean(res["SC"][tail_mask_t])),
        # --- efficiency -----------------------------------------------------
        "D_gain_per_SED": ((float(D[-1]) - D0) / cum_sed) if cum_sed > 0 else np.nan,
        **ss,
    }


#: The four headline outcome metrics required by experiment G, plus two extras.
OUTCOME_METRICS: List[str] = [
    "final_D", "mean_D", "min_D_tail", "frac_days_ge_50",
]
ALL_METRICS: List[str] = OUTCOME_METRICS + [
    "auc_D", "min_D", "frac_days_ge_50_tail", "mean_D_tail", "D_gain_per_SED",
]


def best_by(rows, key: str, metric: str):
    """Return the value of ``key`` in the row maximising ``metric``."""
    best = max(rows, key=lambda r: r[metric])
    return best[key]
