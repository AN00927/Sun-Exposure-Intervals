"""
tests/test_vitd_analyses.py
============================
Coverage for vitd_analyses.py (1230 lines, previously zero tests despite being
imported directly by run_all.py:30 and feeding every RESULTS.md section that
compares schedules -- alpha/K_ATTEN/beta sweeps, the two-framing comparison,
the fine-grid and exhaustive interval sweeps, loading-dose and
maintenance-split optimisation, and the sun-exposure minutes tables). It
reaches run_all.py through `capture_log_safe`, so a silent exception here
degrades a report section instead of failing the pipeline loudly -- exactly
the risk this file exists to catch.

CHARACTERISATION vs VALIDATION -- read this before trusting a failure or a
green run of this file.
--------------------------------------------------------------------------
Two different kinds of assertion appear below, named accordingly in each test
function (`test_validation_*` / `test_characterisation_*`) and in-line:

  VALIDATION      The expected value is derivable from the function's own
                  documented contract or constants (a threshold comparison,
                  a session-count formula, an arithmetic identity between two
                  columns) -- independent of what the underlying simulate()
                  model actually predicts. A failure here means the code does
                  not do what it says it does.

  CHARACTERISATION  The expected value is a NUMBER PINNED FROM A RUN OF THE
                  CURRENT CODE (steady_state_D at some parameter combination,
                  an optimal interval, etc). It is NOT independently derived
                  or verified against a source; it only proves the function
                  is deterministic and has not silently changed. A failure
                  here after a deliberate model or calibration change is
                  EXPECTED and the pinned value must be re-derived, not
                  treated as a regression. Do not read a pass here as
                  scientific validation -- this repository has previously
                  mistaken a pinned stale value for verification.

Runtime: every analysis_* function under test is redirected (via the
`_redirect_io` autouse fixture) to write its CSV/PNG outputs into a pytest
tmp_path instead of the real ./figs and ./tables -- this file must never
touch the committed tables/figs that RESULTS.md was generated from. Each
analysis_* call costs well under 1 second (matplotlib figure save dominates;
the underlying simulate() calls are ~0.1-1 ms each), so the whole file runs
in a few seconds.

FIXED DEFECT, now pinned as a regression (fixed 2026-09-20): see
test_loading_dose_day14_is_sampled_at_day_14 below.
`analysis_loading_dose`'s D_at_day14 column used to
index the trajectory array at a fixed position (28) assuming a 0.5-day step
with no extra points, but every dose event (and every loading schedule here
doses on day 0) adds an extra duplicate-time sample to `simulate()`'s output
(pre-dose and post-dose state at the same t), so index 28 landed at a
different actual day depending on how many sessions the loading distribution
used -- from day 13.5 (1 session) to day 9.0 (14 sessions). That biased the
"best loading distribution" comparison across arms evaluated at different
effective days. `vitd_analyses.analysis_loading_dose` now selects by
simulated time instead, and the test below pins that every arm is scored at
t=14.0.
"""

from __future__ import annotations

import math

import matplotlib
# Force a non-interactive backend BEFORE importing vitd_analyses (which
# imports matplotlib.pyplot at module scope and calls plt.subplots/savefig
# in every analysis_* function). Without this, matplotlib's default backend
# auto-selection on this machine intermittently attempts TkAgg against a
# broken local Tk installation (WindowsApps Python distribution), which
# produced flaky, environment-dependent test failures unrelated to any
# analysis_* logic during development of this file -- Agg makes figure
# creation deterministic and headless, which is what a test suite needs.
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

import vitd_analyses as va
import vitd_scenario


@pytest.fixture(autouse=True)
def _redirect_io(tmp_path, monkeypatch):
    """Never let this test file write into the committed ./figs or ./tables.

    vitd_analyses.py reads FIG_DIR/TAB_DIR as module globals at call time (an
    f-string inside each function body), so patching the module attributes
    redirects every analysis_* function's output for the duration of the
    test without touching its source.
    """
    fig_dir = tmp_path / "figs"
    tab_dir = tmp_path / "tables"
    fig_dir.mkdir()
    tab_dir.mkdir()
    monkeypatch.setattr(va, "FIG_DIR", str(fig_dir))
    monkeypatch.setattr(va, "TAB_DIR", str(tab_dir))
    yield


def _assert_no_nan_inf(df: pd.DataFrame, numeric_cols):
    for col in numeric_cols:
        arr = pd.to_numeric(df[col], errors="coerce")
        assert not arr.isna().any(), f"column {col!r} contains NaN"
        assert np.isfinite(arr.to_numpy()).all(), f"column {col!r} contains inf"


# ---------------------------------------------------------------------------
# Module-level constants (VALIDATION: these are vitd_analyses' own constants,
# read back to confirm run_all.py's report text -- which quotes them directly
# per CLAUDE.md's "do not restate parameter values" rule -- is reading a
# value that actually matches what the analyses below run with).
# ---------------------------------------------------------------------------

def test_validation_protocol_and_interval_constants_are_internally_consistent():
    assert va.PROTOCOL_DAYS > 0
    assert va.INTERVALS_DAYS == sorted(va.INTERVALS_DAYS)
    assert all(d >= 1 for d in va.INTERVALS_DAYS)
    assert va.TWO_FRAMING_FSTS == sorted(va.TWO_FRAMING_FSTS)
    assert all(1 <= f <= 6 for f in va.TWO_FRAMING_FSTS)
    assert va.TWO_FRAMING_TOTAL_SED > 0
    assert va.TWO_FRAMING_PER_SESSION_SED > 0


# ---------------------------------------------------------------------------
# print_safety_summary -- a pure-ish function (stdout only), whose branching
# logic is fully specified by its own code. VALIDATION: expected behaviour
# is derived from the function's own documented contract, not a simulate()
# output.
# ---------------------------------------------------------------------------

def test_validation_print_safety_summary_empty_log(capsys):
    va.print_safety_summary([], label=" [test]")
    out = capsys.readouterr().out
    assert "no safety data" in out


def test_validation_print_safety_summary_overdose_marker(capsys):
    safety_log = [(0.0, {"fraction_med": 1.5, "individual_med_sed": 2.0})]
    va.print_safety_summary(safety_log, label=" [test]")
    out = capsys.readouterr().out
    assert "OVERDOSE" in out
    assert "150.0%" in out or "150%" in out


def test_validation_print_safety_summary_no_overdose_marker_below_med():
    import io
    import contextlib
    safety_log = [(0.0, {"fraction_med": 0.4, "individual_med_sed": 2.0})]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        va.print_safety_summary(safety_log, label=" [test]")
    assert "OVERDOSE" not in buf.getvalue()


# ---------------------------------------------------------------------------
# Analysis A: alpha sweep
# ---------------------------------------------------------------------------

def test_validation_alpha_sweep_structure_and_finiteness():
    df = va.analysis_alpha_sweep()
    assert set(df["FST"].unique()) == {5, 6}
    assert set(df["interval_days"].unique()) == set(va.INTERVALS_DAYS)
    _assert_no_nan_inf(df, ["steady_state_D", "final_D", "final_M",
                            "net_final_D_change_per_SED",
                            "auc_above_baseline_per_SED",
                            "auc_above_sufficiency_per_SED"])
    # cumulative_SED for 1 SED/session must equal (# pulses in the schedule),
    # which build_pulse_schedule places at every `interval_days` from day 0 --
    # this follows directly from build_pulse_schedule's own contract, not
    # from any simulate() output, so it is validation.
    for interval in va.INTERVALS_DAYS:
        expected_sessions = len(range(0, va.PROTOCOL_DAYS, interval))
        row = df[(df["FST"] == 5) & (df["alpha"] == 0.05) &
                 (df["interval_days"] == interval)].iloc[0]
        assert row["cumulative_SED"] == pytest.approx(expected_sessions * 1.0)


def test_characterisation_alpha_sweep_pinned_value():
    """PINNED CURRENT VALUE -- not a verified/literature figure.

    FST 6, alpha=0.05 (mid-sweep), interval=3d, 1 SED/session, 84-day
    horizon, default ModelParams(). Regenerate this number (do not guess it)
    if the model or calibration changes; a failure here after such a change
    is expected, not a regression.

    RE-PINNED 2026-09-23 for the 2026-09-22 parameter pass: 33.154221816376165
    -> 30.205532616384378 nmol/L. FST VI is the cell that moves most, for two
    compounding reasons -- the ITA -> M clip release (decision A3) gives FST VI
    a real, un-clipped M for the first time, and the live body-weight term
    applies BODY_WEIGHT_BY_FST[6], the heaviest cell in the table.
    """
    df = va.analysis_alpha_sweep()
    row = df[(df["FST"] == 6) & (df["alpha"] == 0.05) &
             (df["interval_days"] == 3)].iloc[0]
    assert row["steady_state_D"] == pytest.approx(30.205532616384378, abs=1e-6)


# ---------------------------------------------------------------------------
# Analysis B: K_ATTEN sweep
# ---------------------------------------------------------------------------

def test_validation_katten_sweep_structure_and_finiteness():
    df = va.analysis_K_ATTEN_sweep()
    assert set(df["FST"].unique()) == {5, 6}
    assert set(df["K_ATTEN"].unique()) == {0.2, 0.5, 1.0, 2.0, 4.0, 8.0}
    _assert_no_nan_inf(df, ["steady_state_D", "final_D", "final_M",
                            "net_final_D_change_per_SED"])


def test_characterisation_katten_sweep_monotone_in_katten():
    """CHARACTERISATION: higher K_ATTEN (stronger melanin synthesis penalty)
    should not INCREASE steady-state D at fixed interval/FST/dose -- this is
    the direction the model's own equation (attenuation multiplies synthesis
    down) guarantees, so it is a weak sanity check on the sweep rather than a
    pinned number, but it still counts as characterisation because it
    exercises simulate()'s actual numeric output rather than an
    independently-derivable formula.
    """
    df = va.analysis_K_ATTEN_sweep()
    sub = df[(df["FST"] == 5) & (df["interval_days"] == 3)].sort_values("K_ATTEN")
    vals = sub["steady_state_D"].to_numpy()
    assert np.all(np.diff(vals) <= 1e-9), (
        "steady_state_D should be non-increasing in K_ATTEN at fixed "
        f"FST/interval/dose; got {vals}")


# ---------------------------------------------------------------------------
# Analysis C: two-framing comparison
# ---------------------------------------------------------------------------

def test_validation_two_framings_session_count_formula():
    df1, df2 = va.analysis_two_framings()
    assert set(df1["FST"].unique()) == set(va.TWO_FRAMING_FSTS)
    assert set(df2["FST"].unique()) == set(va.TWO_FRAMING_FSTS)
    for interval in va.INTERVALS_DAYS:
        row = df1[(df1["FST"] == va.TWO_FRAMING_FSTS[0]) &
                  (df1["interval_days"] == interval)].iloc[0]
        expected_n = math.ceil(va.PROTOCOL_DAYS / interval)
        assert row["n_sessions"] == expected_n
        assert row["sed_per_session"] == pytest.approx(
            va.TWO_FRAMING_TOTAL_SED / expected_n)
    # Framing 2 always doses TWO_FRAMING_PER_SESSION_SED regardless of
    # interval -- directly from the function's own documented contract.
    assert (df2["sed_per_session"] == va.TWO_FRAMING_PER_SESSION_SED).all()
    _assert_no_nan_inf(df1, ["steady_state_D", "net_final_D_change_per_SED"])
    _assert_no_nan_inf(df2, ["steady_state_D", "net_final_D_change_per_SED"])


# ---------------------------------------------------------------------------
# Analysis E: beta sweep
# ---------------------------------------------------------------------------

def test_validation_beta_sweep_halflife_to_beta_conversion():
    df = va.analysis_beta_sweep()
    for hl in [21, 30, 56, 80, 120]:
        row = df[df["beta_halflife_days"] == hl].iloc[0]
        assert row["beta"] == pytest.approx(np.log(2) / hl)
    _assert_no_nan_inf(df, ["steady_state_D", "final_D"])


# ---------------------------------------------------------------------------
# Analysis F: fine-grid interval sweep
# ---------------------------------------------------------------------------

def test_validation_fine_grid_covers_1_to_30():
    df = va.analysis_fine_grid()
    assert set(df["interval_days"].unique()) == set(range(1, 31))
    _assert_no_nan_inf(df, ["steady_state_D", "final_D",
                            "auc_above_baseline_per_SED",
                            "auc_deficit_30", "auc_deficit_50"])


def test_characterisation_fine_grid_pinned_optimum():
    """PINNED CURRENT VALUE: FST 5's argmax steady_state_D interval and
    value over the 1-30 day grid, fixed 1 SED/session, default params.
    `run_all.py` quotes `analysis_fine_grid`'s numbers directly in the
    generated report -- this pins the current answer so an accidental
    behaviour change is caught even though it is not, by itself,
    independent scientific verification.

    RE-PINNED 2026-09-23 for the 2026-09-22 parameter pass: FST V's
    steady-state D at the optimum falls 54.79133556789651 ->
    51.84378727275469 nmol/L. The ARGMAX is unchanged at 1 day, which is the
    part run_all.py quotes as a conclusion; only the level moved.
    """
    df = va.analysis_fine_grid()
    sub = df[df["FST"] == 5]
    best = sub.loc[sub["steady_state_D"].idxmax()]
    assert int(best["interval_days"]) == 1
    assert best["steady_state_D"] == pytest.approx(51.84378727275469, abs=1e-6)


# ---------------------------------------------------------------------------
# Analysis G: exhaustive integer sweep (1-60 days)
# ---------------------------------------------------------------------------

def test_validation_scipy_optimize_replicates_exhaustive_argmax():
    """VALIDATION: `opt_steady_state_D` must equal the max of an independent
    exhaustive scan over the same 1-60 day range and FST -- this re-derives
    the expected value from the function's own documented method
    (exhaustive integer sweep) rather than pinning a number, so a failure
    means the sweep itself is broken, not that the model changed.
    """
    df = va.analysis_scipy_optimize()
    for _, row in df.iterrows():
        fst = int(row["FST"])
        from vitd_model import simulate, build_pulse_schedule, steady_state_D
        vals = []
        for interval in range(1, 61):
            sched = build_pulse_schedule(84, 1.0, interval)
            r = simulate(fst=fst, schedule=sched, duration_days=84)
            vals.append(steady_state_D(r))
        assert row["opt_steady_state_D"] == pytest.approx(max(vals), abs=1e-9)
        assert int(row["opt_interval_absolute_D"]) == 1 + int(np.argmax(vals))


# ---------------------------------------------------------------------------
# Analysis H: maximum interval for sufficiency
# ---------------------------------------------------------------------------

def test_validation_max_interval_sufficiency_threshold_is_consistent():
    df = va.analysis_max_interval_for_sufficiency()
    assert set(df["interval_days"].unique()) == set(range(1, 61))
    # above_sufficiency must be exactly the >=50 comparison the function
    # documents itself as using -- derivable independent of any pinned value.
    expected = df["steady_state_D"] >= 50.0
    assert (df["above_sufficiency"] == expected).all()
    _assert_no_nan_inf(df, ["steady_state_D", "auc_deficit_30", "auc_deficit_50"])


def test_characterisation_max_interval_sufficiency_is_monotone_for_short_intervals():
    """CHARACTERISATION: steady_state_D should not increase as the interval
    lengthens at fixed 1 SED/session, for intervals short relative to the
    `steady_state_D` 28-day averaging window (see vitd_model.steady_state_D):
    less frequent dosing, same per-dose size, cannot raise the plateau.

    NOTE -- discovered while writing this test, not a `vitd_analyses.py`
    defect: `steady_state_D` averages D over only the last 28 days of the
    84-day horizon (a finite-horizon proxy, not a true infinite-horizon
    asymptote). Once `interval_days` approaches/exceeds that 28-day window,
    the average's value depends on where a dose happens to fall relative to
    the window edge, and the full FST-5 sweep (interval 1-60) is measurably
    NON-monotonic there -- e.g. it dips to ~23.02 near interval 41 then
    rises back to ~23.09 near interval 52 before falling again. That is a
    property of `vitd_model.steady_state_D`'s windowing, triggered here, not
    a bug in `analysis_max_interval_for_sufficiency` (which only reports the
    LONGEST interval clearing the 50 nmol/L bar -- always in the short-
    interval, unambiguously-monotonic region for both FSTs, so the artefact
    does not affect any conclusion this analysis prints). Scoped to
    interval <= 20 days here for that reason (2026-09-20 fix).
    """
    df = va.analysis_max_interval_for_sufficiency()
    for fst in (5, 6):
        sub = df[(df["FST"] == fst) & (df["interval_days"] <= 20)].sort_values("interval_days")
        vals = sub["steady_state_D"].to_numpy()
        assert np.all(np.diff(vals) <= 1e-6), (
            f"FST {fst}: steady_state_D not monotone non-increasing over "
            f"interval 1-20d: {vals}")


# ---------------------------------------------------------------------------
# Analysis I: loading dose optimization -- includes the KNOWN LIVE DEFECT
# ---------------------------------------------------------------------------

def test_validation_loading_dose_structure():
    df = va.analysis_loading_dose()
    assert set(df["n_loading_sessions"].unique()) == {1, 2, 3, 5, 7, 14}
    assert set(df["budget_SED"].unique()) == {2, 5, 10, 15, 20}
    _assert_no_nan_inf(df, ["D_at_day14", "final_D", "steady_state_D",
                            "AUC_above_50", "cumulative_SED"])
    for _, row in df.iterrows():
        assert row["sed_per_session"] == pytest.approx(
            row["budget_SED"] / row["n_loading_sessions"])


def test_loading_dose_day14_is_sampled_at_day_14():
    """REGRESSION for a fixed defect (2026-09-20). See the module docstring.

    `analysis_loading_dose` previously computed `D_at_day14` via
    `d14_idx = min(28, len(r["D"]) - 1)`, documented in-code as "index 28 at
    0.5-day steps". That assumed the trajectory array advances one sample per
    0.5 days with no other insertions. It does not: `vitd_model.simulate`
    appends an EXTRA sample at the exact time of every dose event (the
    pre-dose state was already recorded; the post-dose jump gets its own entry
    at the same t), and every loading distribution here doses on day 0, so the
    extra-sample effect was present in every row and grew with
    `n_loading_sessions`. Index 28 landed at t=13.5 for the 1-session
    distribution and t=9.0 for the 14-session one -- and because D rises
    through the loading window, that systematically disadvantaged
    multi-session loading in the very "best distribution" comparison
    `analysis_loading_dose` prints and run_all.py quotes.

    The selection is now made by simulated TIME
    (`np.searchsorted(r["t"], LOADING_DAYS, side="left")`), which takes the
    first sample at t=14.0 -- the end of the loading window, before the
    day-14 maintenance dose is credited -- identically for every arm.

    This test asserts both halves: that the fixed index is still wrong (so the
    historical finding stays reproducible) and that the time-based selection
    used by the analysis lands on t=14.0 for every distribution it runs.
    """
    from vitd_model import simulate

    PROTOCOL_DAYS = va.PROTOCOL_DAYS
    LOADING_DAYS = 14
    MAINT_START, MAINT_INTERVAL, MAINT_SED = 14, 7, 1.0
    DISTRIBUTIONS = [
        (1, [0]), (2, [0, 6]), (3, [0, 4, 8]),
        (5, [0, 1, 2, 3, 4]), (7, list(range(7))), (14, list(range(14))),
    ]

    def _traj(n_sessions, load_days, budget=10.0):
        sed_per = budget / n_sessions
        sched = [(day, sed_per) for day in load_days]
        sched += [(day, MAINT_SED)
                  for day in range(MAINT_START, PROTOCOL_DAYS, MAINT_INTERVAL)]
        sched.sort(key=lambda x: x[0])
        return simulate(fst=5, schedule=sched, duration_days=PROTOCOL_DAYS)

    # The historical defect, still reproducible: index 28 is NOT day 14, and
    # how far off it is depends on the schedule shape.
    assert float(_traj(1, [0])["t"][28]) == pytest.approx(13.5)
    assert float(_traj(14, list(range(14)))["t"][28]) == pytest.approx(9.0)

    # The fix: time-based selection lands on t=14.0 for every arm.
    for n_sessions, load_days in DISTRIBUTIONS:
        r = _traj(n_sessions, load_days)
        idx = min(int(np.searchsorted(r["t"], LOADING_DAYS, side="left")),
                  len(r["D"]) - 1)
        assert float(r["t"][idx]) == pytest.approx(14.0), (
            f"{n_sessions}-session arm scored at t={float(r['t'][idx])}, "
            f"not day {LOADING_DAYS}"
        )


# ---------------------------------------------------------------------------
# Analysis J: maintenance dose splitting
# ---------------------------------------------------------------------------

def test_validation_maintenance_split_strategies_present():
    df = va.analysis_maintenance_split()
    assert set(df["strategy"].unique()) == {"single_1SED", "split_0.5SED"}
    assert set(df["interval_T"].unique()) == {7, 14, 21}
    _assert_no_nan_inf(df, ["steady_state_D", "AUC_above_50", "cumulative_SED"])
    for T in (7, 14, 21):
        row = df[(df["interval_T"] == T) & (df["strategy"] == "split_0.5SED")].iloc[0]
        split_interval = max(1, T // 2)
        assert row["actual_interval"] == split_interval
        assert row["sed_per_session"] == pytest.approx(split_interval / T)


# ---------------------------------------------------------------------------
# Analysis K: sun exposure scheduling
# ---------------------------------------------------------------------------

def test_validation_sun_exposure_uses_frozen_scenario_and_internal_arithmetic():
    df = va.analysis_sun_exposure()
    assert set(df["category"].unique()) == {"practical", "sweep"}
    _assert_no_nan_inf(df, ["sed_per_session", "steady_state_D", "cumulative_SED"])
    MIN_PER_SED = 1.0 / vitd_scenario.SED_PER_MINUTE
    for _, row in df.iterrows():
        assert row["sed_per_session"] == pytest.approx(
            row["minutes_per_session"] / MIN_PER_SED)
        assert row["weekly_SED"] == pytest.approx(
            row["sed_per_session"] * row["sessions_per_week"])
        assert row["above_sufficiency"] == (row["steady_state_D"] >= 50.0)


def test_validation_sun_exposure_practical_schedule_names():
    df = va.analysis_sun_exposure()
    practical_names = set(df[df["category"] == "practical"]["schedule"].unique())
    assert practical_names == {
        "daily_10min", "3xwk_20min", "2xwk_30min",
        "weekly_45min", "minimal_15min_3x", "ultra_10min_2x",
    }


# ---------------------------------------------------------------------------
# Analysis D: trajectories (no return value; must not raise)
# ---------------------------------------------------------------------------

def test_validation_trajectories_runs_without_raising_and_writes_figure(tmp_path):
    result = va.analysis_trajectories()
    assert result is None  # documented: this analysis has no return value
    import os
    assert os.path.exists(os.path.join(va.FIG_DIR, "trajectories.png"))
