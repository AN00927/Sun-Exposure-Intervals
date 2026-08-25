"""
spacing_experiments.py
======================
The experiment grid for the spacing sub-study (A-G).

Every function returns a list of flat dicts, one per simulated schedule.
Nothing here calibrates anything: parameter sweeps exist to measure how the
conclusion depends on a parameter, never to adopt a new value.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

import vitd_model as vm
import spacing_model as sm

# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

FST_ALL: List[int] = [1, 2, 3, 4, 5, 6]
FST_FOCUS: List[int] = [2, 5]          # one light, one dark (dark-skin priority)
SED_LEVELS: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0]
INTERVALS: List[int] = [1, 2, 3, 4, 5, 7, 10, 14]
INTERVALS_COARSE: List[int] = [1, 2, 3, 4, 7, 14]
HORIZONS: List[int] = [sm.HORIZON_SHORT, sm.HORIZON_LONG, sm.HORIZON_SS]

#: Total-UV budget levels for experiment B, expressed as SED per day of horizon.
BUDGET_SED_PER_DAY: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0]

#: Minutes outdoors per week for experiment C.
MINUTES_PER_WEEK: List[float] = [35.0, 70.0, 140.0, 210.0, 350.0]

#: Burst (days_on, days_off) patterns for experiment D.
BURSTS: List[Tuple[int, int]] = [
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 6), (1, 13),
    (2, 2), (2, 5), (3, 4), (4, 3), (5, 2), (5, 9),
    (7, 7), (10, 10), (14, 14),
]

METRICS = sm.OUTCOME_METRICS
SS_METRICS = ["ss_mean_D", "ss_min_D", "ss_frac_days_ge_50"]


# ---------------------------------------------------------------------------
# A. Matched per-session dose
# ---------------------------------------------------------------------------

def exp_A_matched_per_session() -> List[Dict]:
    rows = []
    p = sm.make_params()
    for fst in FST_ALL:
        for sed in SED_LEVELS:
            for iv in INTERVALS:
                for H in HORIZONS:
                    r = sm.run_schedule(
                        fst, sm.periodic_schedule(H, sed, iv), H, p,
                        dose_framing="fixed_per_session_SED", period_days=iv)
                    r.update(experiment="A", interval_days=iv,
                             sed_target=sed, framing="matched_per_session_SED")
                    rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# B. Matched TOTAL UV budget
# ---------------------------------------------------------------------------

def exp_B_matched_budget() -> List[Dict]:
    rows = []
    p = sm.make_params()
    for fst in FST_ALL:
        for per_day in BUDGET_SED_PER_DAY:
            for iv in INTERVALS_COARSE:
                for H in HORIZONS:
                    total = per_day * H
                    sched = sm.budget_schedule(H, total, iv)
                    r = sm.run_schedule(
                        fst, sched, H, p,
                        dose_framing="fixed_cumulative_SED", period_days=iv)
                    r.update(experiment="B", interval_days=iv,
                             budget_sed_per_day=per_day, total_sed_budget=total,
                             framing="matched_total_SED")
                    rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# C. Matched total time outdoors
# ---------------------------------------------------------------------------

def exp_C_matched_minutes() -> List[Dict]:
    rows = []
    p = sm.make_params()
    for fst in FST_ALL:
        for mpw in MINUTES_PER_WEEK:
            for iv in [1, 2, 3, 4, 7]:
                for H in [sm.HORIZON_LONG, sm.HORIZON_SS]:
                    sched = sm.minutes_schedule(H, mpw, iv)
                    r = sm.run_schedule(
                        fst, sched, H, p,
                        dose_framing="fixed_cumulative_SED", period_days=iv)
                    r.update(experiment="C", interval_days=iv,
                             minutes_per_week=mpw,
                             minutes_per_session=mpw / (7.0 / iv),
                             sed_per_minute=sm.SED_PER_MINUTE,
                             framing="matched_minutes_outdoors")
                    rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# D. Non-uniform / irregular patterns
# ---------------------------------------------------------------------------

def exp_D_bursts() -> List[Dict]:
    rows = []
    p = sm.make_params()
    for fst in FST_ALL:
        for sed in [0.5, 1.0, 2.0]:
            for on, off in BURSTS:
                P = on + off
                for H in [sm.HORIZON_LONG, sm.HORIZON_SS]:
                    sched = sm.burst_schedule(H, sed, on, off)
                    r = sm.run_schedule(
                        fst, sched, H, p,
                        dose_framing="fixed_per_session_SED", period_days=P)
                    r.update(experiment="D1", days_on=on, days_off=off,
                             duty_cycle=on / P, period_days_pattern=P,
                             sed_target=sed, pattern=f"{on}on{off}off",
                             framing="matched_per_session_SED")
                    rows.append(r)
    return rows


def _canonical_rotation(pattern: Tuple[int, ...]) -> Tuple[int, ...]:
    P = len(pattern)
    return min(tuple(pattern[i:] + pattern[:i]) for i in range(P))


def exp_D_free_search(period: int = 14,
                      duties: Optional[List[int]] = None,
                      sed: float = 1.0,
                      fsts: Optional[List[int]] = None) -> List[Dict]:
    """Exhaustive search over binary on/off patterns at fixed duty cycle.

    Total UV is identical across all patterns at a given duty cycle, so this
    isolates ARRANGEMENT. Evaluated at HORIZON_SS with last-period metrics,
    which are phase-invariant, so rotations of the same pattern are equivalent
    and only one representative of each rotation class is simulated.
    """
    duties = duties or [2, 4, 7]
    fsts = fsts or FST_FOCUS
    H = sm.HORIZON_SS
    p = sm.make_params()
    rows = []
    for k in duties:
        seen = set()
        reps = []
        for combo in itertools.combinations(range(period), k):
            pat = tuple(1 if i in combo else 0 for i in range(period))
            can = _canonical_rotation(pat)
            if can in seen:
                continue
            seen.add(can)
            reps.append(can)
        for fst in fsts:
            for pat in reps:
                sched = sm.pattern_schedule(H, sed, pat)
                r = sm.run_schedule(
                    fst, sched, H, p,
                    dose_framing="fixed_per_session_SED", period_days=period)
                gaps = _gap_stats(pat)
                r.update(experiment="D2", duty_k=k, period_days_pattern=period,
                         pattern="".join(str(b) for b in pat),
                         sed_target=sed, max_gap=gaps[0], gap_std=gaps[1],
                         framing="matched_per_session_SED")
                rows.append(r)
    return rows


def _gap_stats(pattern: Tuple[int, ...]) -> Tuple[int, float]:
    """(longest run of off-days wrapping around, std-dev of inter-session gaps)."""
    P = len(pattern)
    on_days = [i for i, b in enumerate(pattern) if b]
    if len(on_days) < 2:
        return P - len(on_days), 0.0
    gaps = [(on_days[(i + 1) % len(on_days)] - on_days[i]) % P
            for i in range(len(on_days))]
    gaps = [g if g > 0 else P for g in gaps]
    return int(max(gaps)) - 1, float(np.std(gaps))


# ---------------------------------------------------------------------------
# E. Mechanism decomposition
# ---------------------------------------------------------------------------

def exp_E_mechanisms() -> List[Dict]:
    rows = []
    H = sm.HORIZON_SS
    for variant in sm.MECHANISM_VARIANTS:
        p, kw = sm.params_and_kwargs_for_variant(variant)
        note = sm.MECHANISM_VARIANTS[variant][2]
        for fst in FST_ALL:
            for iv in INTERVALS_COARSE:
                # (i) matched per-session dose
                r = sm.run_schedule(fst, sm.periodic_schedule(H, 1.0, iv), H, p,
                                    sim_kwargs=kw, period_days=iv,
                                    dose_framing="fixed_per_session_SED")
                r.update(experiment="E", variant=variant, variant_note=note,
                         interval_days=iv, framing="matched_per_session_SED")
                rows.append(r)
                # (ii) matched total budget, 1 SED/day equivalent
                r = sm.run_schedule(fst, sm.budget_schedule(H, 1.0 * H, iv), H, p,
                                    sim_kwargs=kw, period_days=iv,
                                    dose_framing="fixed_cumulative_SED")
                r.update(experiment="E", variant=variant, variant_note=note,
                         interval_days=iv, framing="matched_total_SED")
                rows.append(r)
        # (iii) arrangement at fixed duty cycle 7/14
        for fst in FST_FOCUS:
            for name, pat in [("alternating", (1, 0) * 7),
                              ("7on7off", (1,) * 7 + (0,) * 7),
                              ("3-4-clustered", (1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0))]:
                r = sm.run_schedule(fst, sm.pattern_schedule(H, 1.0, pat), H, p,
                                    sim_kwargs=kw, period_days=14,
                                    dose_framing="fixed_per_session_SED")
                r.update(experiment="E_arrangement", variant=variant,
                         variant_note=note, pattern=name,
                         framing="matched_per_session_SED")
                rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# E'. K_ATTEN sweep -- the facultative-melanin attenuation coefficient
# ---------------------------------------------------------------------------
# K_ATTEN was a RETIRED parameter (pinned at 0) when this sweep was written.
# It was reinstated as an active, calibrated parameter on 2026-08-21
# (user-approved; see the vitd_model.py module docstring for the live value),
# so vitd_model.RETIRED_PARAMETERS may or may not contain it depending on
# when this module is imported relative to that change -- check it at
# runtime rather than trusting this comment.

K_ATTEN_SWEEP: List[float] = [0.0, 0.1, 0.33, 0.5, 1.0, 2.0, 3.0, 5.0,
                              10.0, 20.0, 50.0, 100.0]


def exp_E_katten_sweep() -> List[Dict]:
    """Does a non-daily optimum appear across a range of tanning attenuation?

    If K_ATTEN is currently RETIRED (see vitd_model.RETIRED_PARAMETERS), the
    shipped model has no facultative-melanin attenuation at all and this sweep
    explores what would happen if it did. If K_ATTEN is currently ACTIVE
    (reinstated 2026-08-21), the shipped model already has a nonzero value
    (see the vitd_model.py module docstring) and this sweep explores the
    surrounding range, including the shipped value and points far outside it.
    Either way this sweep runs inside `allow_retired_parameter_sweep()`,
    which is explicitly for sensitivity/ablation (a no-op guard whenever
    K_ATTEN is not currently retired). No value found here is adopted.
    """
    rows = []
    H = sm.HORIZON_SS
    for ka in K_ATTEN_SWEEP:
        with vm.allow_retired_parameter_sweep():
            p = sm.make_params()
            p.K_ATTEN = ka
            for fst in FST_ALL:
                for iv in INTERVALS_COARSE:
                    r = sm.run_schedule(fst, sm.periodic_schedule(H, 1.0, iv), H, p,
                                        period_days=iv,
                                        dose_framing="fixed_per_session_SED")
                    r.update(experiment="E_katten", K_ATTEN=ka, interval_days=iv,
                             framing="matched_per_session_SED")
                    rows.append(r)
                    r = sm.run_schedule(fst, sm.budget_schedule(H, 1.0 * H, iv), H, p,
                                        period_days=iv,
                                        dose_framing="fixed_cumulative_SED")
                    r.update(experiment="E_katten", K_ATTEN=ka, interval_days=iv,
                             framing="matched_total_SED")
                    rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Sensitivity: photoadaptation parameters and optional mechanisms
# ---------------------------------------------------------------------------

_LN2 = float(np.log(2.0))

#: name -> (list of (label, params_overrides, sim_kwargs), description)
SENSITIVITY_SWEEPS: Dict[str, List[Tuple[str, dict, dict]]] = {
    "alpha (melanin induction rate)": [
        (f"alpha={v}", {}, {"alpha_override": v})
        for v in [0.0, 0.001, 0.002, 0.005, 0.02, 0.1, 0.5]
    ],
    "beta (melanin decay rate)": [
        (f"beta=ln2/{d}", {}, {"k_decay_M_override": _LN2 / d})
        for d in [1e6, 224, 56, 28, 14, 3]
    ],
    "alpha_SC (SC build rate)": [
        (f"alpha_SC={v}", {"alpha_SC": v}, {})
        for v in [0.0, 0.04, 0.08, 0.12, 0.20, 0.5, 2.0]
    ],
    "k_decay_SC (SC regression rate)": [
        (f"k_decay_SC=ln2/{d}", {"k_decay_SC": _LN2 / d}, {})
        for d in [1e6, 56, 28, 21, 7, 2]
    ],
    "K_SC (SC attenuation strength)": [
        (f"K_SC={v}", {"K_SC": v}, {})
        for v in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]
    ],
    "SC_max (SC ceiling)": [
        (f"SC_max={v}", {"SC_max_by_FST": {i: v for i in range(1, 7)}}, {})
        for v in [0.0, 0.23, 0.36, 0.6, 1.0, 3.0, 10.0]
    ],
    "k_damage (acute SC disruption)": [
        (f"k_damage={v}", {"k_damage": v}, {})
        for v in [0.0, 0.05, 0.15, 0.5]
    ],
    "D0 (starting 25(OH)D)": [],   # filled below, handled specially
    "optional mechanisms (default OFF)": [
        ("photodegradation ON", {"use_photodegradation": True}, {}),
        ("transit delay ON", {"use_transit_delay": True}, {}),
        ("both ON", {"use_photodegradation": True, "use_transit_delay": True}, {}),
    ],
}


def exp_sensitivity() -> List[Dict]:
    rows = []
    H = sm.HORIZON_SS
    for sweep_name, entries in SENSITIVITY_SWEEPS.items():
        if not entries:
            continue
        for label, p_over, kw in entries:
            p = sm.make_params(**p_over)
            for fst in FST_FOCUS:
                for iv in INTERVALS_COARSE:
                    r = sm.run_schedule(fst, sm.periodic_schedule(H, 1.0, iv), H, p,
                                        sim_kwargs=kw, period_days=iv,
                                        dose_framing="fixed_per_session_SED")
                    r.update(experiment="SENS", sweep=sweep_name, level=label,
                             interval_days=iv, framing="matched_per_session_SED")
                    rows.append(r)
                    r = sm.run_schedule(fst, sm.budget_schedule(H, 1.0 * H, iv), H, p,
                                        sim_kwargs=kw, period_days=iv,
                                        dose_framing="fixed_cumulative_SED")
                    r.update(experiment="SENS", sweep=sweep_name, level=label,
                             interval_days=iv, framing="matched_total_SED")
                    rows.append(r)
    # D0 sweep is a run_schedule argument, not a params field.
    for D0 in [15.0, 25.0, 40.0, 60.0, 90.0, 130.0]:
        p = sm.make_params()
        for fst in FST_FOCUS:
            for iv in INTERVALS_COARSE:
                for framing, sched, df in [
                    ("matched_per_session_SED", sm.periodic_schedule(H, 1.0, iv),
                     "fixed_per_session_SED"),
                    ("matched_total_SED", sm.budget_schedule(H, 1.0 * H, iv),
                     "fixed_cumulative_SED"),
                ]:
                    r = sm.run_schedule(fst, sched, H, p, D0=D0, period_days=iv,
                                        dose_framing=df)
                    r.update(experiment="SENS", sweep="D0 (starting 25(OH)D)",
                             level=f"D0={D0:g}", interval_days=iv, framing=framing)
                    rows.append(r)
    return rows


def exp_ksc_threshold(fsts: Optional[List[int]] = None) -> List[Dict]:
    """Locate the SC-attenuation strength at which daily stops being optimal.

    What matters physically is the maximum steady-state SC attenuation factor
    exp(-K_SC * SC_max). This scan varies K_SC on a fine grid at the shipped
    SC_max and records that factor alongside the winning interval.
    """
    rows = []
    H = sm.HORIZON_SS
    grid = sorted(set([0.0, 0.5, 1.0, 2.0] +
                      [round(x, 2) for x in np.arange(3.0, 60.1, 1.0)]))
    fsts = fsts or FST_ALL
    for ksc in grid:
        p = sm.make_params(K_SC=ksc)
        for fst in fsts:
            sc_max = p.SC_max_by_FST[fst]
            for framing, df, builder in [
                ("matched_per_session_SED", "fixed_per_session_SED",
                 lambda iv: sm.periodic_schedule(H, 1.0, iv)),
                ("matched_total_SED", "fixed_cumulative_SED",
                 lambda iv: sm.budget_schedule(H, 1.0 * H, iv)),
            ]:
                for iv in INTERVALS_COARSE:
                    r = sm.run_schedule(fst, builder(iv), H, p, period_days=iv,
                                        dose_framing=df)
                    r.update(experiment="KSC_THRESHOLD", K_SC=ksc,
                             SC_max=sc_max,
                             max_SC_throughput=float(np.exp(-ksc * sc_max)),
                             interval_days=iv, framing=framing)
                    rows.append(r)
    return rows


def exp_combined_worst_case() -> List[Dict]:
    """All photoadaptation parameters simultaneously at their most
    spacing-favourable value inside the documented prior ranges
    (vitd_model.ModelParams.SC_PRIOR_RANGES), to test whether daily survives
    a joint adversarial choice rather than a one-at-a-time one."""
    rows = []
    H = sm.HORIZON_SS
    configs = [
        ("shipped defaults", {}, {}),
        ("SC priors, spacing-favourable (in-range)",
         {"K_SC": 2.0, "alpha_SC": 0.20,
          "SC_max_by_FST": {i: 0.40 for i in range(1, 7)},
          "k_decay_SC": _LN2 / 21.0, "k_damage": 0.0}, {}),
        ("SC priors in-range + fast melanin turnover",
         {"K_SC": 2.0, "alpha_SC": 0.20,
          "SC_max_by_FST": {i: 0.40 for i in range(1, 7)},
          "k_decay_SC": _LN2 / 21.0},
         {"alpha_override": 0.02, "k_decay_M_override": _LN2 / 14.0}),
        ("above + photodegradation ON (non-default mechanism)",
         {"K_SC": 2.0, "alpha_SC": 0.20,
          "SC_max_by_FST": {i: 0.40 for i in range(1, 7)},
          "k_decay_SC": _LN2 / 21.0, "use_photodegradation": True},
         {"alpha_override": 0.02, "k_decay_M_override": _LN2 / 14.0}),
    ]
    for label, p_over, kw in configs:
        p = sm.make_params(**p_over)
        for fst in FST_ALL:
            for framing, df, builder in [
                ("matched_per_session_SED", "fixed_per_session_SED",
                 lambda iv: sm.periodic_schedule(H, 1.0, iv)),
                ("matched_total_SED", "fixed_cumulative_SED",
                 lambda iv: sm.budget_schedule(H, 1.0 * H, iv)),
            ]:
                for iv in INTERVALS_COARSE:
                    r = sm.run_schedule(fst, builder(iv), H, p, sim_kwargs=kw,
                                        period_days=iv, dose_framing=df)
                    r.update(experiment="WORST_CASE", config=label,
                             interval_days=iv, framing=framing)
                    rows.append(r)
    return rows
