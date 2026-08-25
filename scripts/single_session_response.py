"""
single_session_response.py
==========================
Single-exposure-session dose-response and decay.

Answers the advisor's question verbatim:

  "Can you plot how vitamin D concentration changes with exposure duration
   with a single exposure session, as it is made longer (including how it
   decays once the session is over). A single plot with different curves
   (time on x-axis, vitamin D concentration on the y-axis, and one curve for
   each exposure duration)."

One session at t = 0, then a 180-day observation window so the full return to
baseline is visible. One curve per session DURATION in minutes; minutes are
converted to erythemal SED with ``vitd_scenario.sed_for_minutes()`` under the
frozen reference scenario (see ``vitd_scenario.describe()`` /
``vitd_scenario.LIMITATION_STATEMENT`` -- the numbers live there, not here).

Two things this figure is actually about
----------------------------------------
1. DIMINISHING RETURNS. Per-session synthesis is logarithmic in dose, so
   doubling the minutes does not double the 25(OH)D rise. The CSV quantifies
   this as marginal nmol/L per additional minute.

2. NON-EXPONENTIAL, AND SLOW, DECAY. ``ModelParams.k_decay_D`` is
   state-dependent (Datta 2017 power law), so the half-life LENGTHENS as D
   falls and the washout is not a single exponential. This is a whole-body
   apparent decline rate, not a plasma clearance rate, so one session's
   effect persists for many months. See the limitation below before reading
   anything off the decay limb.

PRINCIPAL LIMITATION -- the decay limb is an extrapolation
----------------------------------------------------------
``k_decay_D`` was fitted to Datta 2017's three 25(OH)D groups, whose means
span ``DATTA_FITTED_D_RANGE`` below (the fitted half-lives themselves are in
``vitd_model.py``'s module docstring -- the canonical source; they are not
restated here). This figure starts at the model's default baseline and decays
downward from there, so essentially the ENTIRE trajectory sits BELOW the
lowest 25(OH)D that Datta observed. The rate governing every decay curve in
this deliverable is therefore an extrapolation past the low end of the data
that set it, and the half-life the power law returns there is much longer
than any half-life Datta actually measured.

Nothing in the script hardcodes those half-lives: ``_thalf()`` calls
``ModelParams.k_decay_D`` directly, the runtime log prints what it got, and
``_extrapolation_banner()`` renders the same computed numbers onto the figure
itself. If the parameterisation changes, all three move together. Peak
heights and the marginal-yield ranking do NOT depend on this -- they are set
by the synthesis term at t=0. The washout TIMES do.

Modelling caveat -- read this before interpreting the rise
----------------------------------------------------------
In ``vitd_model.simulate()`` the DEFAULT behaviour is that a session's
synthesis is credited to serum 25(OH)D as an INSTANTANEOUS impulse at the
session time. Duration enters only through the SED it buys; it does not
stretch the rise out in time. So in the default model the rise is a vertical
step, not a ramp -- that step is a modelling artefact of the impulse
formulation, not a physiological claim.

``ModelParams.use_transit_delay`` (default ``False``) routes synthesis through
a cutaneous/serum D3 pool that drains into 25(OH)D with
``transit_half_life_days`` (default 1.0 d, flagged [PRIOR] in
``vitd_model.py`` -- no study reports it as a fitted rate constant). With it
on, the same session produces a physiologically shaped rise peaking about a
day later. Both are plotted side by side; the LEFT column is the default
model that every other headline number in this repo uses.

No-exposure control -- why the curves fall THROUGH baseline
------------------------------------------------------------
The model has no dietary/endogenous source term (the oral pathway was
deliberately removed -- see the block near "DIETARY_UG_TO_NMOL_PER_L" in
``vitd_model.py``). Baseline 25(OH)D is therefore an initial condition, not a
fixed point: with one session and nothing after it, D decays past baseline and
keeps going. Absolute levels late in the window are NOT a physiological
prediction -- they are what the model says about UV input alone.

So every panel also carries a dashed grey NO-EXPOSURE CONTROL (identical run,
empty schedule), and the session's effect is measured as the gap between the
curve and that control. Time-to-return is reported both ways in the CSV; the
control-referenced columns are the meaningful ones.

Outputs
-------
  figs/single_session_response.png
  tables/single_session_response.csv

Integrator note
---------------
``simulate()`` steps the decay with a fixed 0.5-day exponential-Euler
sub-step. Over a 3-year window that is worth checking, so ``run()`` audits it
against the CLOSED-FORM solution of the decay ODE. With no pulse after t=0,
dD/dt = -c*D^1.72 is separable and solves exactly to
``D(t) = (D_peak^-0.72 + 0.72*c*t)^(-1/0.72)``. The audit result is printed
at runtime and returned on the DataFrame's ``.attrs``.

Run:  PYTHONIOENCODING=utf-8 python3 scripts/single_session_response.py
"""

from __future__ import annotations

import dataclasses
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vitd_scenario
from vitd_model import ModelParams, simulate
from vitd_objectives import PRACTICAL_MAX_MINUTES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS_DIR = os.path.join(ROOT, "figs")
TABLES_DIR = os.path.join(ROOT, "tables")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZON_DAYS = 1095          # ~3 years, so the washout visibly completes
INSET_DAYS = 180             # main inset: rise + early decay
DETAIL_DAYS = 10             # small inset: impulse step vs transit-delay ramp

# 25(OH)D range spanned by the group means Datta 2017 fitted k_decay_D over.
# Descriptive of the SOURCE DATA, not a model parameter -- the parameters and
# their citation live in vitd_model.py's module docstring.
DATTA_FITTED_D_RANGE = (43.0, 132.0)
DURATIONS_MIN = [5.0, 10.0, 20.0, 40.0, 80.0, 120.0]

# FST V and VI are this project's priority population (standing dark-skin
# directive). FST II is carried only as a light-skin contrast -- read it as
# the extrapolation, not the target.
FSTS = [5, 6, 2]
FST_LABELS = {
    2: "FST II (light-skin contrast)",
    5: "FST V (priority population)",
    6: "FST VI (priority population)",
}

# "Returned to baseline" tolerance, nmol/L. Small but not zero: the
# state-dependent decay approaches baseline asymptotically.
RETURN_TOL_NMOL_L = 1.0


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _simulate_one_schedule(fst: int, schedule, params: ModelParams, D0: float):
    """Run *schedule* over HORIZON_DAYS from baseline *D0*."""
    return simulate(
        fst=fst,
        schedule=schedule,
        duration_days=HORIZON_DAYS,
        params=params,
        D0=D0,
        dose_framing="fixed_absolute_SED",
    )


def _simulate_one(fst: int, sed: float, params: ModelParams, D0: float):
    """One session of *sed* at t=0, then decay over HORIZON_DAYS."""
    return _simulate_one_schedule(fst, [(0.0, float(sed))], params, D0)


def _thalf(params: ModelParams, D: float) -> float:
    """25(OH)D half-life in days that the model is using at level *D*.

    Derived from ``ModelParams.k_decay_D`` so it can never drift out of sync
    with the parameterisation.
    """
    return float(np.log(2.0) / params.k_decay_D(D))


def _exact_decay(D_peak: float, t: np.ndarray) -> np.ndarray:
    """Closed-form solution of the pure-decay limb, for integrator auditing.

    With no pulse after t=0 the ODE is dD/dt = -c * D^(1+q) with
    q = 0.72, and c = ln2/2953 read back out of ``k_decay_D`` itself. It
    separates to D(t) = (D_peak^-q + q*c*t)^(-1/q).
    """
    q = 0.72
    # Recover c without restating it: k = c * D^q  =>  c = k(D) / D^q.
    c = ModelParams.k_decay_D(1.0)  # D=1 => D^q = 1 => k(1) == c
    return (D_peak ** -q + q * c * t) ** (-1.0 / q)


def _audit_integrator(t: np.ndarray, D: np.ndarray) -> float:
    """Max relative error of the simulated decay limb vs the closed form."""
    i0 = int(np.argmax(D))
    tt, DD = t[i0:], D[i0:]
    exact = _exact_decay(float(DD[0]), tt - tt[0])
    return float(np.max(np.abs(DD - exact) / exact))


def _first_time_below(t: np.ndarray, y: np.ndarray, thresh: float) -> float:
    """First time at which *y* has fallen below *thresh* and stayed there.

    0.0 if it was never above; NaN if still above at the end of the window.
    """
    above = np.where(y > thresh)[0]
    if len(above) == 0:
        return 0.0
    last = above[-1]
    if last >= len(t) - 1:
        return float("nan")
    return float(t[last + 1])


def run(params: ModelParams = None) -> pd.DataFrame:
    """Build the single-session dose-response figure and table.

    Returns the per-(FST, duration, model-variant) DataFrame that is also
    written to ``tables/single_session_response.csv``.
    """
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    base = ModelParams() if params is None else params
    D0 = base.D0_default

    # Two model variants. 'impulse' is the repo default.
    variants = [
        ("impulse", "Impulse synthesis (DEFAULT model)", False),
        ("transit_delay", "With D3 transit delay (use_transit_delay=True)", True),
    ]

    print(vitd_scenario.describe())
    print()
    print(vitd_scenario.LIMITATION_STATEMENT)
    print()
    print(f"Baseline 25(OH)D (ModelParams.D0_default) : {D0:g} nmol/L")
    print(f"Observation window                        : {HORIZON_DAYS} days")
    lo, hi = DATTA_FITTED_D_RANGE
    print(f"25(OH)D half-life at baseline             : "
          f"{_thalf(base, D0):.0f} d   (from ModelParams.k_decay_D)")
    print(f"  half-life at the Datta fit's low end    : {_thalf(base, lo):.0f} d "
          f"(D = {lo:g} nmol/L)")
    print(f"  half-life at the Datta fit's high end   : {_thalf(base, hi):.0f} d "
          f"(D = {hi:g} nmol/L)")
    if D0 < lo:
        print(f"  *** EXTRAPOLATION: baseline {D0:g} nmol/L is BELOW the "
              f"{lo:g}-{hi:g} nmol/L range k_decay_D was fitted over, and the "
              f"decay only goes lower. Every washout time below is off the "
              f"low end of the source data. See this module's docstring. ***")
    print(f"Baseline-return tolerance                 : {RETURN_TOL_NMOL_L:g} nmol/L")
    print(f"PRACTICAL_MAX_MINUTES (vitd_objectives)   : {PRACTICAL_MAX_MINUTES:g} min")
    over = [m for m in DURATIONS_MIN if m > PRACTICAL_MAX_MINUTES]
    if over:
        print(f"  NOTE: durations above the practical cap and therefore "
              f"outside the schedule search space: {over}")
    else:
        print("  All plotted durations are at or below the practical cap.")
    print()

    rows = []
    curves = {}    # (variant_key, fst) -> list of (minutes, t, D)
    controls = {}  # (variant_key, fst) -> (t, D) with no exposure at all

    for vkey, vlabel, use_delay in variants:
        # dataclasses.replace keeps every calibrated field on `base`;
        # only the transit switch moves.
        p = dataclasses.replace(base, use_transit_delay=use_delay)

        for fst in FSTS:
            # No-exposure control: same initial condition, empty schedule.
            # The model has no dietary source term, so this decays below
            # baseline; the session's effect is the gap against it.
            ctrl = _simulate_one_schedule(fst, [], p, D0)
            t_ctrl, D_ctrl = ctrl["t"], ctrl["D"]
            series = []
            prev_peak = None
            prev_gap = None
            prev_min = 0.0
            for minutes in DURATIONS_MIN:
                sed = vitd_scenario.sed_for_minutes(minutes)
                res = _simulate_one(fst, sed, p, D0)
                t, D = res["t"], res["D"]
                peak = float(np.max(D))
                peak_t = float(t[int(np.argmax(D))])
                delta = peak - D0
                # Gap against the no-exposure control, interpolated onto t.
                gap = D - np.interp(t, t_ctrl, D_ctrl)
                gap_peak = float(np.max(gap))
                # 90% washout of the session's effect: scale-free, so it stays
                # informative for the small doses too.
                t_90 = _first_time_below(t, gap, 0.10 * gap_peak)
                t_50 = _first_time_below(t, gap, 0.50 * gap_peak)
                gap_end = float(gap[-1])
                # Marginal yield relative to the previous (shorter) duration on
                # the ladder; for the first rung, relative to no exposure.
                if prev_peak is None:
                    marginal = delta / minutes if minutes > 0 else float("nan")
                    marginal_gap = gap_peak / minutes
                else:
                    marginal = (peak - prev_peak) / (minutes - prev_min)
                    marginal_gap = (gap_peak - prev_gap) / (minutes - prev_min)
                rows.append({
                    "variant": vkey,
                    "fst": fst,
                    "minutes": minutes,
                    "sed_delivered": sed,
                    "peak_D_nmol_L": peak,
                    "peak_day": peak_t,
                    "delta_from_baseline_nmol_L": delta,
                    "delta_per_minute_nmol_L": delta / minutes,
                    # Peak-referenced. NOTE: with use_transit_delay on, the
                    # absolute peak is depressed by decay running during the
                    # 1-day rise, so this column is non-monotone at the short
                    # end for the transit variant. The gap-referenced pair
                    # below is the clean diminishing-returns measure.
                    "marginal_nmol_L_per_extra_minute": marginal,
                    "gap_per_minute_nmol_L": gap_peak / minutes,
                    "marginal_gap_nmol_L_per_extra_minute": marginal_gap,
                    # Control-referenced -- the meaningful columns.
                    "peak_gap_vs_no_exposure_nmol_L": gap_peak,
                    "days_to_50pct_washout_vs_control": t_50,
                    "days_to_90pct_washout_vs_control": t_90,
                    "gap_at_end_of_window_nmol_L": gap_end,
                    # Per-run audit of the decay rate actually in force, so
                    # the headline half-life is checkable rather than quoted.
                    "D_at_end_of_window_nmol_L": float(D[-1]),
                    "t_half_days_at_peak_D": _thalf(p, peak),
                    "t_half_days_at_end_D": _thalf(p, float(D[-1])),
                    "decay_extrapolated_below_datta": bool(
                        peak < DATTA_FITTED_D_RANGE[0]),
                    "frac_of_peak_gap_remaining_at_end": (
                        gap_end / gap_peak if gap_peak > 0 else float("nan")),
                    "days_gap_vs_control_below_tol": _first_time_below(
                        t, gap, RETURN_TOL_NMOL_L),
                    # Absolute-baseline crossing. The model has no dietary
                    # source term, so D keeps falling past D0 after this:
                    # read it as a crossing time, not a return to steady
                    # state. 0.0 => never rose more than tol above baseline.
                    "days_until_D_crosses_baseline": _first_time_below(
                        t, D - D0, RETURN_TOL_NMOL_L),
                    "peak_exceeded_return_tol": bool(delta > RETURN_TOL_NMOL_L),
                    "baseline_D0_nmol_L": D0,
                    "above_practical_max_minutes": minutes > PRACTICAL_MAX_MINUTES,
                })
                series.append((minutes, t.copy(), D.copy()))
                prev_peak, prev_gap, prev_min = peak, gap_peak, minutes
            curves[(vkey, fst)] = series
            controls[(vkey, fst)] = (t_ctrl.copy(), D_ctrl.copy())

    # Integrator audit: the closed form is valid for the pure-decay limb, so
    # it applies to the impulse variant (transit-delay keeps feeding the serum
    # pool after t=0 and is checked structurally instead).
    audit = {}
    for fst in FSTS:
        res = _simulate_one(fst, vitd_scenario.sed_for_minutes(max(DURATIONS_MIN)),
                            dataclasses.replace(base, use_transit_delay=False), D0)
        audit[fst] = _audit_integrator(res["t"], res["D"])
    worst = max(audit.values())
    print(f"Integrator audit over {HORIZON_DAYS} d (0.5-d exponential-Euler "
          f"vs closed form):")
    for fst, e in audit.items():
        print(f"  FST {fst:<2d} max relative error {e:.2e}")
    verdict = ("OK -- no meaningful drift" if worst < 1e-3
               else "DEGRADED -- do not trust the decay limb")
    print(f"  worst {worst:.2e}  ->  {verdict}")

    # Structural checks that must hold for both variants over the long window.
    for (vkey, fst), series in curves.items():
        for minutes, t, D in series:
            assert (D > 0).all(), f"non-physical D<=0: {vkey} FST{fst} {minutes}min"
            i0 = int(np.argmax(D))
            assert (np.diff(D[i0:]) <= 1e-9).all(), (
                f"non-monotone decay after peak: {vkey} FST{fst} {minutes}min")
    print("  D > 0 and monotone decreasing after the peak for all "
          f"{len(curves) * len(DURATIONS_MIN)} runs")

    df = pd.DataFrame(rows)
    df.attrs["integrator_max_rel_err"] = worst
    df.attrs["integrator_verdict"] = verdict
    csv_path = os.path.join(TABLES_DIR, "single_session_response.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # ---------------------------------------------------------------- figure
    n_rows, n_cols = len(FSTS), len(variants)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.3 * n_rows),
                             sharex=True)
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.05, 0.9, len(DURATIONS_MIN))]
    handles, labels = None, None

    for r, fst in enumerate(FSTS):
        for c, (vkey, vlabel, _use) in enumerate(variants):
            ax = axes[r, c]
            t_c, D_c = controls[(vkey, fst)]

            def _draw(target, xmax, lw):
                """Draw every duration curve plus the control, clipped to xmax."""
                for (minutes, t, D), col in zip(curves[(vkey, fst)], colors):
                    m = t <= xmax
                    sed = vitd_scenario.sed_for_minutes(minutes)
                    cap = " *" if minutes > PRACTICAL_MAX_MINUTES else ""
                    target.plot(t[m], D[m], color=col, lw=lw,
                                label=f"{minutes:g} min ({sed:.2f} SED){cap}")
                mc = t_c <= xmax
                target.plot(t_c[mc], D_c[mc], color="0.30", ls="--", lw=lw,
                            label="no exposure (control)")
                target.axhline(D0, color="0.45", ls=":", lw=0.9)

            _draw(ax, HORIZON_DAYS, 1.6)
            ax.set_xlim(0, HORIZON_DAYS)
            ax.grid(alpha=0.3)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            if r == 0:
                ax.set_title(vlabel, fontsize=10)
            if c == 0:
                ax.set_ylabel("25(OH)D (nmol/L)")
            if r == n_rows - 1:
                ax.set_xlabel("Days since the single session")
            ax.text(0.985, 0.96, FST_LABELS[fst], transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="0.7", alpha=0.9))

            # Main inset: rise + early decay, the window the schedule work
            # actually optimises over. Sits in the empty upper-right region
            # left by the long decay.
            ins = ax.inset_axes([0.42, 0.34, 0.55, 0.45])
            _draw(ins, INSET_DAYS, 1.2)
            ins.set_xlim(0, INSET_DAYS)
            ins.set_title(f"first {INSET_DAYS} d", fontsize=7.5)
            ins.tick_params(labelsize=6)
            ins.grid(alpha=0.25)

            # Detail inset: the first few days, where the impulse step and the
            # transit-delay ramp visibly differ. Bottom-left stays empty
            # because the curves are still high there.
            det = ax.inset_axes([0.05, 0.08, 0.26, 0.28])
            _draw(det, DETAIL_DAYS, 1.0)
            det.set_xlim(0, DETAIL_DAYS)
            det.set_title(f"first {DETAIL_DAYS} d", fontsize=7)
            det.tick_params(labelsize=5.5)
            det.grid(alpha=0.25)

    star = ("   * = above vitd_objectives.PRACTICAL_MAX_MINUTES "
            f"({PRACTICAL_MAX_MINUTES:g} min)" if over else "")
    fig.suptitle(
        "Single exposure session at t=0: 25(OH)D rise and decay by session "
        "duration\n"
        f"Frozen reference scenario (vitd_scenario.py); baseline {D0:g} nmol/L "
        "(dotted). The LEFT column's rise is a step because default synthesis is "
        "an instantaneous\nimpulse -- duration buys SED, it does not stretch the "
        "rise. Dashed grey = no-exposure control: the model carries no dietary "
        "source term, so\n25(OH)D decays THROUGH baseline and a session's effect "
        f"is the GAP to that control.{star}",
        fontsize=10)

    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.055))

    # Extrapolation banner -- every number computed here from k_decay_D, never
    # typed in, so it cannot drift from the parameterisation.
    lo, hi = DATTA_FITTED_D_RANGE
    D_min_plotted = min(float(np.min(D_c)) for (D_c) in
                        [controls[k][1] for k in controls])
    banner = (
        "READ THE DECAY LIMB WITH THIS CAVEAT: the rate governing it is "
        "EXTRAPOLATED BELOW THE DATA THAT SET IT.\n"
        f"k_decay_D (vitd_model.py; Datta 2017 power law) was fitted over "
        f"25(OH)D {lo:g}-{hi:g} nmol/L. This figure runs from "
        f"{D0:g} down to {D_min_plotted:.1f} nmol/L -- entirely below that "
        f"range.\nModel half-life: {_thalf(base, hi):.0f} d at {hi:g} nmol/L "
        f"and {_thalf(base, lo):.0f} d at {lo:g} nmol/L (both inside the fit), "
        f"but {_thalf(base, D0):.0f} d at the {D0:g} nmol/L baseline and "
        f"{_thalf(base, D_min_plotted):.0f} d at {D_min_plotted:.1f} nmol/L "
        "(both outside it).\n"
        "Peak heights and the marginal-yield ranking do NOT depend on this; the "
        "washout TIMES do. Treat every washout time here as unvalidated.")
    fig.text(0.5, 0.005, banner, ha="center", va="bottom", fontsize=8.5,
             color="#7a1300",
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff4e6", ec="#c0392b",
                       lw=1.4, alpha=0.97))

    fig.tight_layout(rect=[0, 0.115, 1, 0.935])
    fig_path = os.path.join(FIGS_DIR, "single_session_response.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    # ------------------------------------------------------------- console
    for vkey, vlabel, _u in variants:
        print(f"\n--- {vlabel} ---")
        for fst in FSTS:
            sub = df[(df.variant == vkey) & (df.fst == fst)]
            print(f"  {FST_LABELS[fst]}")
            for _, row in sub.iterrows():
                w50 = row["days_to_50pct_washout_vs_control"]
                ret_s = (f">{HORIZON_DAYS} d" if np.isnan(w50)
                         else f"{w50:.0f} d")
                rem = row["frac_of_peak_gap_remaining_at_end"]
                print(f"    {row['minutes']:5.0f} min  "
                      f"{row['sed_delivered']:5.2f} SED  "
                      f"peak {row['peak_D_nmol_L']:6.2f}  "
                      f"delta {row['delta_from_baseline_nmol_L']:5.2f}  "
                      f"gap {row['peak_gap_vs_no_exposure_nmol_L']:5.2f}  "
                      f"marginal {row['marginal_gap_nmol_L_per_extra_minute']:6.4f} "
                      f"nmol/L/extra-min  t_half(effect) {ret_s:>7}  "
                      f"remaining at d{HORIZON_DAYS} {rem:5.1%}")

    return df


if __name__ == "__main__":
    run()
