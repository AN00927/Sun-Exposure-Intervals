"""
run_all.py
==========
End-to-end run: validation + all priority analyses + summary report.

Usage:  python3 run_all.py
"""

import argparse
import datetime as _dt
import os
import platform
import subprocess
import sys
import time
import traceback
from io import StringIO
from contextlib import redirect_stdout

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

import vitd_scenario
import vitd_validation
import vitd_analyses
import vitd_calibration
from vitd_optimization import (analysis_constant_sessions, analysis_spacing_fixed_dose,
                               analysis_variable_dose,
                               analysis_joint_schedule,
                               analysis_objective_hierarchy,
                               analysis_tri_objective)


REPORT_PATH = "RESULTS.md"


def capture_log(fn, *args, **kwargs):
    buf = StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def _ensure_scripts_on_path():
    """Put `scripts/` on sys.path once. Idempotent."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def capture_log_safe(label, fn, *args, **kwargs):
    """`capture_log` that cannot abort the pipeline.

    The two `scripts/` sections wired in below (1a, 5a) are heavier than a
    typical analysis and live outside the core modules, so a failure in one of
    them must not cost the other seventeen sections a ten-minute run. Whatever
    the analysis printed before it died is kept, a failure banner is appended
    so the failure is *visible in the report* rather than silently missing,
    and the return value is `None` so the prose code can fall back.

    Returns ``(log_text, result, error_or_None)``.
    """
    buf = StringIO()
    result = None
    err = None
    try:
        with redirect_stdout(buf):
            result = fn(*args, **kwargs)
    except Exception as exc:                       # not BaseException: Ctrl-C
        err = "%s: %s" % (type(exc).__name__, exc)
        buf.write(
            "\n\n*** SECTION FAILED: %s ***\n%s\n\n%s\n"
            "The rest of the pipeline continued. Output above (if any) is\n"
            "partial and the figures/tables for this section may be stale or\n"
            "absent.\n" % (label, err, traceback.format_exc()))
    return buf.getvalue(), result, err


def _import_script_module(name):
    """Import a module from `scripts/` defensively.

    Used for the two Phase-F sections. Import is deferred to call time and
    wrapped, because an import-time failure (a mid-edit file, a missing
    optional dependency) should degrade one section, not the pipeline.
    """
    _ensure_scripts_on_path()
    return __import__(name)


def _report_decision_problems():
    """Print the three decision problems people actually face."""
    from vitd_objectives import (optimal_spacing_fixed_sessions,
                                 deficiency_vs_risk_frontier,
                                 minutes_for_target)

    print("=" * 74)
    print("Q1. FIXED SESSION BUDGET: 'I can get outside N times in 12 weeks'")
    print("=" * 74)
    print("Cumulative UV is IDENTICAL across every candidate spacing, so no")
    print("spacing can win by simply dosing less.\n")
    for fst, sed in ((5, 1.0), (5, 3.0), (6, 3.0)):
        print(f"  FST {fst}, {sed} SED/session, 84-day horizon, baseline 25 nmol/L:")
        print(f"    {'sessions':>8} {'best gap':>9} {'mean D':>8} {'final D':>8} "
              f"{'min D':>7} {'d<50':>6} {'melanoma':>9}")
        for n in (6, 12, 18, 24, 36):
            r = optimal_spacing_fixed_sessions(fst=fst, n_sessions=n,
                                               sed_per_session=sed,
                                               objective="auc_deficit_50")
            b = r["best"]
            if b is None:
                continue
            # Spacing spread shows whether the objective discriminates or all
            # spacings tie (unreachable target at this dose).
            spread = (max(row["mean_D"] for row in r["table"])
                      - min(row["mean_D"] for row in r["table"]))
            print(f"    {n:>8} {b['gap_days']:>8}d {b['mean_D']:>8.1f} "
                  f"{b['final_D']:>8.1f} {b['min_D']:>7.1f} "
                  f"{b['days_below_50']:>6.0f} {b['intermittent_index']:>9.3f}"
                  f"   (mean D spread across spacings: {spread:.1f})")
        print()

    print("=" * 74)
    print("Q2. STAY SUFFICIENT AT LEAST CANCER RISK (Pareto frontier)")
    print("=" * 74)
    print("SCC tracks cumulative dose; melanoma/BCC track intermittent bursts")
    print("(Elwood 1997, Leiter 2009). Both are reported.\n")
    for fst in (2, 5):
        fr = deficiency_vs_risk_frontier(fst=fst, horizon_days=365)
        print(f"  FST {fst}, 1-year horizon, target 50 nmol/L:")
        cs = fr["cheapest_sufficient"]
        if cs:
            print(f"    Lowest-risk schedule that stays sufficient: "
                  f"{cs['sed_per_session']} SED every {cs['interval_days']}d "
                  f"({cs['n_sessions']} sessions, {cs['total_sed']:.0f} SED total)")
            print(f"      mean D {cs['mean_D']:.1f}, days below 50: "
                  f"{cs['days_below_target']:.0f}, SCC idx {cs['scc_index']:.2f}, "
                  f"melanoma idx {cs['intermittent_index']:.3f}, "
                  f"burn fraction {cs['burn_fraction']:.0%}")
        else:
            print("    No schedule on the grid keeps this skin type sufficient.")
        print(f"    Pareto frontier ({len(fr['frontier'])} nondominated points):")
        print(f"      {'SED':>5} {'interval':>9} {'d<50':>7} {'SCC':>7} {'melanoma':>9}")
        for p in fr["frontier"][:8]:
            print(f"      {p['sed_per_session']:>5.2f} {p['interval_days']:>8}d "
                  f"{p['days_below_target']:>7.0f} {p['scc_index']:>7.2f} "
                  f"{p['intermittent_index']:>9.3f}")
        print()

    print("=" * 74)
    print("Q3. MINUTES OF SUN PER DAY")
    print("=" * 74)
    print("UV Index -> personal erythemal dose, with the horizontal-to-body")
    print("geometry correction applied. 'would burn' flags unusable advice.\n")
    print(f"  {'FST':>4} {'UVI':>5} {'BSA':>6} {'target':>7} {'minutes':>9} "
          f"{'%MED':>6} {'burn':>6}")
    for uvi, bsa, label in ((3.5, 0.35, "UK Mar-Sep avg"),
                            (7.0, 0.35, "UK midsummer"),
                            (7.0, 0.10, "hands+face only")):
        for fst in (2, 5, 6):
            for target in (50.0, 85.8):
                r = minutes_for_target(fst=fst, uv_index=uvi, exposed_bsa=bsa,
                                       frequency_per_week=7.0, target=target,
                                       horizon_days=150, max_minutes=600.0)
                mins = f"{r['minutes']:.0f}" if r["reachable"] else "n/a"
                pmed = f"{100*r['fraction_med']:.0f}%" if r["reachable"] else "-"
                burn = ("YES" if r["would_burn"] else "no") if r["reachable"] else "-"
                print(f"  {fst:>4} {uvi:>5.1f} {bsa:>6.2f} {target:>7.1f} "
                      f"{mins:>9} {pmed:>6} {burn:>6}   {label}")
    print()
    print("  'n/a' means the target is unreachable at <=600 min/day -- reported")
    print("  rather than silently returning the search bound.")


def _single_session_headline(df):
    """One sentence of headline numbers for section 1a, from the returned df.

    Every number is computed here from the analysis's own output; nothing is
    typed in (CLAUDE.md: parameter values live in the code, not in prose).
    Returns "" if the frame is missing or does not carry the columns, so a
    schema change in the script degrades the sentence rather than the run.
    """
    try:
        d = df[df["variant"] == sorted(df["variant"].unique())[0]]
        d = d[d["fst"] == 6] if (d["fst"] == 6).any() else d
        d = d.sort_values("minutes")
        lo, hi = d.iloc[0], d.iloc[-1]
        min_ratio = hi["minutes"] / lo["minutes"]
        gap_ratio = (hi["peak_gap_vs_no_exposure_nmol_L"]
                     / lo["peak_gap_vs_no_exposure_nmol_L"])
        marg_lo = lo["marginal_gap_nmol_L_per_extra_minute"]
        marg_hi = hi["marginal_gap_nmol_L_per_extra_minute"]
        rem = hi["frac_of_peak_gap_remaining_at_end"]
        return (
            "\n**Headline, generated from this run's own output table.** For "
            "the darkest skin type carried here, under the default "
            "(impulse-synthesis) model, a %g-fold increase in session length "
            "(%g -> %g minutes) buys only a %.1f-fold increase in the peak "
            "25(OH)D gap over the no-exposure control: the marginal yield "
            "falls from %.4f to %.4f nmol/L per extra minute across that "
            "range. The washout is correspondingly slow — %.0f%% of the peak "
            "gap from the longest session is still present at the end of the "
            "observation window. Longer sessions are subject to diminishing "
            "returns because per-session synthesis is logarithmic in dose; "
            "this is the single-session statement of the same concavity that "
            "makes frequent-and-small win in sections 5, 5a and 17.\n"
            % (min_ratio, lo["minutes"], hi["minutes"], gap_ratio,
               marg_lo, marg_hi, 100.0 * rem))
    except Exception as exc:                       # pragma: no cover - guard
        return ("\n*(Headline numbers unavailable: %s: %s. Read them from the "
                "log and CSV below.)*\n" % (type(exc).__name__, exc))


def _frontload_headline(df):
    """Headline numbers for section 5a, computed from the returned frame.

    Nothing here is typed in: every figure is derived from the analysis's own
    output at report time (CLAUDE.md). Column names are read defensively, so
    a schema change in the script degrades this paragraph rather than the run.
    """
    try:
        import numpy as _np
        out = []
        RUNG = "melanin_only"
        DEF = "AUC_deficit_below_sufficiency"

        # ---------------------------------------------------------- arm (a)
        a = df[(df["arm"] == "within_week") & (df["rung"] == RUNG)]
        pcts, mono = [], 0
        for _, g in a.groupby(["FST", "weekly_budget_SED"]):
            g = g.sort_values("sessions_per_week")
            hi = g[g["sessions_per_week"] == g["sessions_per_week"].max()]
            lo = g[g["sessions_per_week"] == g["sessions_per_week"].min()]
            if hi.empty or lo.empty:
                continue
            pcts.append(100.0 * (float(lo["steady_state_D"].iloc[0])
                                 / float(hi["steady_state_D"].iloc[0]) - 1.0))
            mono += int(g["steady_state_D"].is_monotonic_increasing)
        if pcts:
            worst = a.loc[a["max_frac_MED"].idxmax()]
            out.append(
                "**Arm (a) - within-week concentration. Splitting beats "
                "concentrating, monotonically.** Holding the weekly SED "
                "budget fixed, going from a session every day to one session "
                "a week changes steady-state 25(OH)D by between %+.1f%% and "
                "%+.1f%% across the %d skin-type x budget cells, and "
                "steady-state 25(OH)D rises monotonically with sessions per "
                "week in %d of those %d cells. This is the advisor's question "
                "read literally, and the sign is unambiguous: at fixed weekly "
                "UV, front-loading into fewer days is worse. It is also "
                "unsafe at the concentrated end - the worst arm on the grid "
                "(FST %d, %g SED/week, %d session(s) per week) reaches "
                "%.0f%% of one MED in a single session, i.e. a frank burn."
                % (max(pcts), min(pcts), len(pcts), mono, len(pcts),
                   int(worst["FST"]), float(worst["weekly_budget_SED"]),
                   int(worst["sessions_per_week"]),
                   100.0 * float(worst["max_frac_MED"])))

        # G4: name the rung the headline above was read from, and repeat the
        # arm-(a) monotonicity test at every rung so the reader can see
        # whether the SIGN (the only thing `melanin_only`, an uncalibrated
        # ablation, is licensed to speak to) survives. Absolute levels are
        # never compared across rungs here -- only the pass count.
        inv = []
        for rung in sorted(df["rung"].unique()):
            ar = df[(df["arm"] == "within_week") & (df["rung"] == rung)]
            m = n = 0
            for _, g in ar.groupby(["FST", "weekly_budget_SED"]):
                g = g.sort_values("sessions_per_week")
                n += 1
                m += int(g["steady_state_D"].is_monotonic_increasing)
            if n:
                inv.append("`%s` %d/%d" % (rung, m, n))
        if inv:
            out.append(
                "**Rung provenance and invariance.** Every generated figure "
                "and sentence in this section is read off the "
                "**`%s`** rung, which `vitd_ablation.py` flags as an "
                "uncalibrated mechanism-isolation ablation: its absolute "
                "25(OH)D levels must not be quoted and only within-rung "
                "schedule rankings are valid. What transfers is the SIGN, "
                "and it does: arm (a)'s monotonicity holds in %s cells "
                "respectively, across all %d rungs. `melanin_sc_risk` is "
                "dynamically identical to `melanin_plus_sc` by construction "
                "(`vitd_ablation` docstring), so those two are not two "
                "independent confirmations of the third."
                % (RUNG, ", ".join(inv), len(inv)))

        # ---------------------------------------------------------- arm (b)
        b = df[(df["arm"] == "horizon_taper") & (df["rung"] == RUNG)]
        plateau, transient, artefact, conflict = [], [], [], []
        for horizon in sorted(b["horizon_days"].unique()):
            bh = b[b["horizon_days"] == horizon]
            pg, dg, dg_f = [], [], []
            g_def, g_ss = set(), set()
            dts_def, dts_ss, art, spans_t, spans_i = [], [], [], [], []
            n_front = 0
            for _, g in bh.groupby(["FST", "weekly_budget_SED",
                                    "n_sessions_planned"]):
                ref = g[(g["gamma"] == 1.0) & (g["family"] == "uniform")]
                if ref.empty:
                    continue
                r_ss = float(ref["steady_state_D"].iloc[0])
                r_def = float(ref[DEF].iloc[0])
                front = g[g["gamma"] > 1.0]
                if not front.empty and r_ss > 0:
                    pg.append(100.0 * (float(front["steady_state_D"].max())
                                       / r_ss - 1.0))
                    n_front += len(front)
                if r_def > 0:
                    dg.append(100.0 * (1.0 - float(g[DEF].min()) / r_def))
                    fe = g[g["sub_erythemal"].astype(bool)]
                    if not fe.empty:
                        dg_f.append(100.0 * (1.0 - float(fe[DEF].min()) / r_def))
                best_def = g.loc[g[DEF].idxmin()]
                best_ss = g.loc[g["steady_state_D"].idxmax()]
                g_def.add(float(best_def["gamma"]))
                g_ss.add(float(best_ss["gamma"]))
                for row, sink in ((best_def, dts_def), (best_ss, dts_ss)):
                    v = float(row["days_to_sufficiency"])
                    if _np.isfinite(v):
                        sink.append(v)
                if r_def > 0 and r_ss > 0:
                    art.append((100.0 * (float(best_ss["steady_state_D"])
                                         / r_ss - 1.0),
                                float(best_ss[DEF]) / r_def,
                                float(best_ss["gamma"]),
                                float(best_ss.get("first_gap_days", _np.nan))))
                flat = g[g["family"] == "uniform"]
                if len(flat) > 1 and r_ss > 0 and r_def > 0:
                    spans_t.append(100.0 * (float(flat["steady_state_D"].max())
                                            - float(flat["steady_state_D"].min()))
                                   / r_ss)
                    spans_i.append(100.0 * (float(flat[DEF].max())
                                            - float(flat[DEF].min())) / r_def)
            if pg:
                plateau.append(
                    "- **%d-day horizon**: over %d matched-N cells (%d "
                    "front-loaded arms), the best front-loaded profile "
                    "differs from flat-and-uniform on steady-state 25(OH)D "
                    "by a mean of %+.2f%% and at best %+.2f%%. The largest "
                    "POSITIVE plateau gain available anywhere on the "
                    "front-loaded half of the grid is **%+.2f%%**."
                    % (horizon, len(pg), n_front, sum(pg) / len(pg), max(pg),
                       max(0.0, max(pg))))
            if dg:
                transient.append(
                    "- **%d-day horizon**: mean **%.1f%% reduction** in "
                    "accumulated deficiency below sufficiency versus "
                    "flat-and-uniform at the same session count (max %.1f%%); "
                    "%s when the search is restricted to sub-erythemal arms "
                    "only. The deficit-optimal spacing exponent is %s - "
                    "front-loaded - and reaches sufficiency in a median of "
                    "%s days, against %s days at the spacing that maximises "
                    "the terminal plateau instead."
                    % (horizon, sum(dg) / len(dg), max(dg),
                       ("mean %.1f%%" % (sum(dg_f) / len(dg_f))) if dg_f
                       else "no sub-erythemal arm exists",
                       ", ".join("%g" % x for x in sorted(g_def)),
                       ("%.0f" % _np.median(dts_def)) if dts_def else "n/a",
                       ("%.0f" % _np.median(dts_ss)) if dts_ss else "n/a"))
            if art:
                worst_art = max(art)
                gain_pct, def_ratio, gam, gap = worst_art
                if gam < 1.0 and def_ratio > 1.0:
                    artefact.append(
                        "- **%d-day horizon**: the arm that maximises the "
                        "terminal plateau wins it by %+.1f%% - at gamma = %g, "
                        "i.e. **back**-loading. It opens with a %s-day "
                        "no-exposure gap and stacks dose underneath the "
                        "final-28-day averaging window, and pays for it with "
                        "**%.1fx worse** accumulated deficiency than the flat "
                        "reference. That is an artefact of the measurement "
                        "window, not a schedule anyone should follow."
                        % (horizon, gain_pct, gam,
                           ("%.0f" % gap) if _np.isfinite(gap) else "long",
                           def_ratio))
                else:
                    artefact.append(
                        "- **%d-day horizon**: no such artefact. The terminal "
                        "winner here is gamma = %g (%s), it wins by only "
                        "%+.1f%%, and its accumulated deficiency is %.2fx the "
                        "flat reference - so the terminal and integrated "
                        "measures are not being pulled apart at this horizon."
                        % (horizon, gam,
                           "back-loaded" if gam < 1.0 else
                           "front-loaded" if gam > 1.0 else "uniform",
                           gain_pct, def_ratio))
            if spans_t:
                conflict.append(
                    "- **%d-day horizon**: across the spacing exponent the "
                    "terminal measure moves by ~%.0f percentage points while "
                    "the horizon-integrated measure moves by ~%.0f. The "
                    "terminal argmax is gamma = %s and the integrated argmax "
                    "is gamma = %s."
                    % (horizon, sum(spans_t) / len(spans_t),
                       sum(spans_i) / len(spans_i),
                       ", ".join("%g" % x for x in sorted(g_ss)),
                       ", ".join("%g" % x for x in sorted(g_def))))

        if plateau:
            out.append(
                "**Arm (b) - the plateau is a null, and the longer horizon "
                "makes it cleaner, not weaker.** Every arm is referenced to "
                "the flat, uniformly spaced arm at the *same* session count, "
                "so the session-count effect is divided out and only dose "
                "shape and spacing remain:\n\n" + "\n".join(plateau) + "\n\n"
                "Giving the three-parameter shape families four times the "
                "horizon to express themselves did not produce a plateau "
                "gain; it removed what little there was. That forecloses "
                "the obvious "
                "objection that the shapes simply had no room to work. **The "
                "steady-state level is set by the budget and by how finely "
                "that budget is split (arm a). Arranging the same budget "
                "differently over the horizon does not move it.**")
        if transient:
            out.append(
                "**But front-loading has large, real leverage on the "
                "TRANSIENT.** Ranked on accumulated deficiency below "
                "sufficiency - a horizon-integrated measure rather than a "
                "terminal one - the picture inverts from a null to a "
                "substantial effect:\n\n" + "\n".join(transient) + "\n\n"
                "**This is the section's headline.** What front-loading buys "
                "is not a higher plateau but a faster climb to one: reaching "
                "sufficiency in weeks rather than months, and spending far "
                "less time deficient on the way. The effect survives the "
                "erythema cap, so it is not bought with a burn.")
        if artefact:
            out.append(
                "**The large terminal 'gain' at the long horizon is an "
                "artefact and must not be quoted bare.**\n\n"
                + "\n".join(artefact) + "\n\n"
                "Steady-state 25(OH)D is a mean over the final 28 days, so "
                "at a 365-day horizon the averaging window is under 8% of the "
                "run, and any schedule that piles dose into the endpoint wins "
                "it. That is why arm (b) is ranked on deficit AUC with the "
                "terminal measures retained as reported secondaries.")
        if conflict:
            out.append(
                "**Metric-conflict caveat - no taper sign may be stated "
                "without naming the metric it was read off, and the conflict "
                "WIDENS with horizon.**\n\n" + "\n".join(conflict) + "\n\n"
                "Note what this rules out. `steady_state_D` and `final_D` are "
                "**both terminal measures**, and as the horizon grows the "
                "averaging window shrinks toward the endpoint, so the two "
                "converge on each other. Their agreement at the long horizon "
                "is co-movement between two views of the same instant, **not "
                "corroboration**, and must never be presented as two metrics "
                "independently confirming a result.")
        return "\n" + "\n\n".join(out) + "\n" if out else ""
    except Exception as exc:                       # pragma: no cover - guard
        return ("\n*(Headline numbers unavailable: %s: %s. Read them from the "
                "log and CSVs below.)*\n" % (type(exc).__name__, exc))

def main():
    _wall_start = time.time()
    parser = argparse.ArgumentParser(
        description="End-to-end run: validation + all priority analyses + summary report."
    )
    parser.add_argument(
        "--baseline-source",
        choices=["preset", "nhanes"],
        default="preset",
        help="Baseline 25(OH)D source: 'preset' (default scenario constants) or 'nhanes' (empirical survey-weighted priors)",
    )
    parser.add_argument(
        "--run-macro-scripts",
        action="store_true",
        help="Run NHANES population macro-validation, dietary partitioning, and weight dilution diagnostic scripts",
    )
    args = parser.parse_args()

    # Configure baseline source in vitd_scenario
    vitd_scenario.set_default_baseline_source(args.baseline_source)

    print("=" * 70)
    print("Vitamin D scheduling model: full pipeline")
    print(f"Baseline Configuration: {args.baseline_source.upper()}")
    print("=" * 70)
    print(
        "NOTE: Running NHANES population macro-validation scripts is RECOMMENDED\n"
        "whenever baseline assumptions, scenario definitions, or calibrations change:\n"
        "  - scripts/nhanes_macro_validation.py (52-week population steady-state)\n"
        "  - scripts/nhanes_dietary_partitioning.py (baseline floor & dietary lift)\n"
        "  - scripts/nhanes_weight_dilution_check.py (body weight scaling validation)\n"
        "Pass --run-macro-scripts to execute them automatically.\n"
    )

    # Optional: Run NHANES macro diagnostic suite if requested
    # macro_logs is populated only when --run-macro-scripts is passed; a
    # None here (checked at emission time, ~section 16b) is what makes the
    # suite's absence VISIBLE in RESULTS.md instead of silently missing (G11).
    macro_logs = None
    if args.run_macro_scripts:
        print("=" * 70)
        print("RUNNING NHANES MACRO DIAGNOSTIC SUITE")
        print("=" * 70)
        _macro_val = _import_script_module("nhanes_macro_validation")
        _diet_part = _import_script_module("nhanes_dietary_partitioning")
        _wt_dil = _import_script_module("nhanes_weight_dilution_check")

        log_mv, res_mv, err_mv = capture_log_safe("nhanes_macro_validation", _macro_val.evaluate_macro_validation)
        if log_mv:
            print(log_mv.strip())
        log_dp, res_dp, err_dp = capture_log_safe("nhanes_dietary_partitioning", _diet_part.analyze_dietary_partitioning)
        if log_dp:
            print(log_dp.strip())
        log_wd, res_wd, err_wd = capture_log_safe("nhanes_weight_dilution_check", _wt_dil.run_weight_dilution_check)
        if log_wd:
            print(log_wd.strip())
        print("=" * 70)
        print("NHANES Macro Suite Execution Complete.\n")

        macro_logs = [
            ("nhanes_macro_validation.py (52-week population steady-state)",
             log_mv, err_mv),
            ("nhanes_dietary_partitioning.py (baseline floor & dietary lift)",
             log_dp, err_dp),
            ("nhanes_weight_dilution_check.py (body weight scaling)",
             log_wd, err_wd),
        ]

    # 1. Validation
    buf = StringIO()
    with redirect_stdout(buf):
        val_results = vitd_validation.run_all()
    val_log = buf.getvalue()

    # 1b. NHANES preprocessing RETIRED 2026-08-21 to exploratory/ --
    # its own seasonal validation check was already EXPLORATORY_ONLY and
    # this block was already an optional try/except, so nothing that fed
    # a headline number is lost. Population-level NHANES validation now
    # lives in scripts/nhanes_macro_validation.py (run via
    # --run-macro-scripts). See exploratory/README.md.

    # 2. Analyses
    analyses = [
        vitd_analyses.analysis_alpha_sweep,
        vitd_analyses.analysis_K_ATTEN_sweep,
        vitd_analyses.analysis_beta_sweep,
        vitd_analyses.analysis_two_framings,
        vitd_analyses.analysis_fine_grid,
        vitd_analyses.analysis_scipy_optimize,
        vitd_analyses.analysis_max_interval_for_sufficiency,
        vitd_analyses.analysis_loading_dose,
        vitd_analyses.analysis_maintenance_split,
        vitd_analyses.analysis_sun_exposure,
        vitd_analyses.analysis_trajectories,
        analysis_constant_sessions,
        analysis_spacing_fixed_dose,
        analysis_variable_dose,
        analysis_joint_schedule,
        # Objective hierarchy runs last but reports first (section 0) -- reader
        # meets the ranking before any number.
        analysis_objective_hierarchy,
        analysis_tri_objective,
    ]
    alpha_log, k_log, beta_log, frame_log, fine_log, scipy_log, \
        maxint_log, load_log, split_log, sun_log, traj_log, const_log, \
        spacing_log, vardose_log, joint_log, hierarchy_log, tri_log = [
            capture_log(fn) for fn in analyses
        ]

    # Echo to console
    print(hierarchy_log)
    print(tri_log)
    print(val_log)
    print(alpha_log)
    print(k_log)
    print(beta_log)
    print(frame_log)
    print(fine_log)
    print(scipy_log)
    print(maxint_log)
    print(load_log)
    print(split_log)
    print(sun_log)
    print(traj_log)
    print(const_log)
    print(spacing_log)
    print(vardose_log)
    print(joint_log)

    # 2b. Advisor-question deliverables (scripts/). Both are heavier than the
    # analyses above and both are wired defensively via capture_log_safe, so a
    # failure in either one costs its own section and nothing else.
    def _run_single_session():
        return _import_script_module("single_session_response").run()

    def _run_frontload():
        return _import_script_module("frontload_fixed_budget").run()

    single_log, single_df, single_err = capture_log_safe(
        "1a. Single-session dose-response and washout", _run_single_session)
    frontload_log, frontload_df, frontload_err = capture_log_safe(
        "5a. Front-load vs split at a fixed budget", _run_frontload)
    print(single_log)
    print(frontload_log)
    if single_err:
        print("WARNING: section 1a failed (%s) -- pipeline continued."
              % single_err)
    if frontload_err:
        print("WARNING: section 5a failed (%s) -- pipeline continued."
              % frontload_err)

    # 3. Markdown report.
    # Every constant below is interpolated from code, never typed out --
    # prevents the documentation drift CLAUDE.md warns about.
    from vitd_objectives import (PERSONAL_DOSE_GEOMETRY_FACTOR, GAMMA_MED,
                                 SUFFICIENCY_NMOL_L, RAMP_IN_DAYS,
                                 PRACTICAL_MAX_MINUTES)
    from vitd_model import (ModelParams as _MP, E_SOURCE_SOLAR_DEFAULT,
                            get_E_source)
    _p = _MP()
    K_ATTEN_NOW = _p.K_ATTEN
    E_SOLAR_NOW = get_E_source(E_SOURCE_SOLAR_DEFAULT)
    CONST_SESSIONS_FST = 5
    CONST_SESSIONS_TOTAL_SED = 52.0
    CONST_SESSIONS_DAYS = 365

    n_pass = sum(val_results.values())
    n_total = len(val_results)

    # G3(i): decompose the pooled pass count into required vs warning-only,
    # and into evidence tiers, both generated from vitd_validation's own
    # CHECK_REGISTRY rather than typed in -- a pooled "n_pass / n_total" was
    # reported here until this fix and conflated the two buckets.
    _REG = vitd_validation.CHECK_REGISTRY
    _req_specs = [c for c in _REG if not c.warning_only]
    _warn_specs = [c for c in _REG if c.warning_only]
    _n_req_pass = sum(1 for c in _req_specs if val_results.get(c.name))
    _n_warn_pass = sum(1 for c in _warn_specs if val_results.get(c.name))
    _tier_counts = {}
    for c in _REG:
        t = _tier_counts.setdefault(c.tier, [0, 0])
        t[1] += 1
        if val_results.get(c.name):
            t[0] += 1
    report = []
    report.append("# Vitamin D Scheduling Model — Results\n")
    report.append(
        "\n> **AUTO-GENERATED FILE — DO NOT HAND-EDIT.** Rewritten in full on "
        "every `run_all.py` run. Any manual edits here will be silently "
        "overwritten on the next run.\n"
    )
    report.append("\nThis report is generated by `run_all.py`. It captures the validation "
                  "suite output, parameter sweeps, and a brief interpretation.\n")

    # --- run provenance ---------------------------------------------------
    # Nothing here is typed in: every value is read from the environment or
    # from the parsed args at report time (G1). Two runs that produce
    # different numbers must not be textually indistinguishable.
    def _git_desc():
        try:
            h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=10)
            d = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
            if h.returncode:
                return "unavailable"
            return h.stdout.strip() + (" (working tree dirty)" if d.stdout.strip() else "")
        except Exception as exc:
            return "unavailable (%s)" % type(exc).__name__

    report.append("\n## Run provenance\n")
    report.append(
        "\n```\n"
        "generated          : %s\n"
        "git commit         : %s\n"
        "wall-clock to here : %s\n"
        "host logical cores : %s\n"
        "python             : %s\n"
        "pandas / numpy     : %s / %s\n"
        "--baseline-source  : %s%s\n"
        "--run-macro-scripts: %s\n"
        "feedback_mode      : %s\n"
        "```\n"
        % (_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           _git_desc(),
           # Elapsed when this block is BUILT, not at process exit:
           # the report is assembled before the final sections finish
           # writing, so this is a LOWER BOUND on total runtime and is
           # labelled as such. CLAUDE.md carries the end-to-end figure.
           "%d min %02d s (at report assembly; see CLAUDE.md for "
           "end-to-end)" % divmod(int(time.time() - _wall_start), 60),
           os.cpu_count(),
           platform.python_version(),
           pd.__version__, __import__("numpy").__version__,
           args.baseline_source,
           "" if args.baseline_source == "preset"
              else "  <-- NOT the default; every baseline 25(OH)D below is the "
                   "NHANES empirical median, not the preset",
           args.run_macro_scripts, _p.feedback_mode))

    report.append("\n## Scope and reference conditions — read with section 0\n")
    report.append(
        "\nOn 2026-08-21 this repository was cut back to the model it can "
        "actually defend. The location-based dosimetry chain (solar position "
        "→ ambient UV index → posture/site geometry → exposed BSA), the "
        "genotype presets, the behavioural multiplier and the NHANES "
        "preprocessing were **retired to `exploratory/`**. They still exist "
        "and their tests still pass; nothing they produce reaches any number "
        "in this report. What was retired, why, and the un-retirement "
        "procedure are in "
        "[`exploratory/README.md`](exploratory/README.md).\n\n"
        "Every schedule below is evaluated under one frozen scenario, "
        "identical on every exposure day:\n"
    )
    report.append("\n```\n" + vitd_scenario.describe() + "\n```\n")
    report.append("\n**Limitation.** " + vitd_scenario.LIMITATION_STATEMENT + "\n")
    report.append(
        "\nThese values were chosen so that **no recalibration was required**: "
        "each was already the reference value somewhere in the codebase, and "
        "the exposed-BSA figure is `vitd_model.BSA_REF`, at which the BSA "
        "factor is exactly 1.0 and every existing calibration is untouched. "
        "The clothing description is a description of that existing value, "
        "not a new choice — the midpoint of the retired module's "
        "`short_sleeves` (0.165) and `shorts_and_tshirt` (0.495) presets is "
        "0.33, within 6% of it.\n"
    )

    report.append("\n## 0. Which number is the answer — the objective hierarchy\n")
    report.append(
        "\n**Read this section before any other.** This repository carries "
        "eight schedule-scoring functions, and has twice published a headline "
        "\"optimum\" from a metric with no interior optimum. The ranking below "
        "is not prose: it is generated from "
        "`vitd_objectives.OBJECTIVE_HIERARCHY`, and report code that tries to "
        "present a `DIAGNOSTIC`-tier metric as advice raises rather than "
        "prints (`assert_reportable_as_headline`).\n\n"
        "- **PRIMARY** — minimum dose and fewest sessions to *reach and hold* "
        "sufficiency. One objective. Everything reported as advice comes from "
        "here.\n"
        "- **SECONDARY** — the tri-objective frontier (section 13d) and the "
        "fixed-budget risk minimisation (sections 13b/13c, 14b). These refine "
        "a recommendation; they never replace it.\n"
        "- **DIAGNOSTIC** — `gross_synth_per_SED` and "
        "`net_final_D_change_per_SED`. Informative about the *model*, barred "
        "from being reported as advice.\n\n"
        "The `gross_synth_per_SED` failure is **recomputed on every run** "
        "rather than quoted: the log below reports its argmax under three "
        "different search bounds (a real peak does not move when the bound "
        "moves) and the final 25(OH)D at that argmax against the starting "
        "value. It selects a schedule that ends *below* baseline — a net loss "
        "of vitamin D scored as maximal efficiency — which is why it is "
        "background only.\n\n"
        "Infeasible cells in the primary table are **results**. Nothing was "
        "relaxed to fill them; they say that this skin type cannot hold "
        "sufficiency on sun alone at that session budget without burning "
        "(C1) or without a session longer than anyone will sit through "
        f"(C7, {PRACTICAL_MAX_MINUTES:.0f} min). FST VI is the priority "
        "population for this project and is included throughout.\n"
    )
    report.append("\n```\n" + hierarchy_log + "\n```\n")

    report.append("\n## 1. Validation\n")
    report.append(
        f"\n**Required: {_n_req_pass} / {len(_req_specs)}. "
        f"Warning-only: {_n_warn_pass} / {len(_warn_specs)}.** A single "
        f"pooled \"{n_pass} / {n_total}\" was reported here until this fix "
        "and conflated the two buckets. The evidence-tier split, orthogonal "
        "to both and the number that actually bounds what this suite "
        "licenses:\n\n"
    )
    report.append("| tier | passed |\n|---|---|\n" + "".join(
        "| %s | %d / %d |\n" % (t, v[0], v[1])
        for t, v in _tier_counts.items()))
    report.append(
        "\nOnly the QUANTITATIVE row is an external numeric comparison; see "
        "the per-check `tier_note` in `vitd_validation.CHECK_REGISTRY` for "
        "why each check carries the tier it does, and **section 15**, which "
        "carries the only out-of-sample test and is not counted in any of "
        "the numbers above.\n")
    report.append("\n```\n" + val_log + "\n```\n")

    # ---- 1a. Single-session dose-response and washout -------------------
    # Numbered 1a rather than inserted as a new integer section: several
    # sections below cross-reference each other by number ("see section 13c",
    # "section 6's exhaustive sweep"), and renumbering would silently break
    # every one of them. 1a places the foundational single-session result
    # before the parameter sweeps that assume it.
    report.append("\n## 1a. Single session: dose-response and washout\n")
    report.append(
        "\n**The question this answers**, put by the project's advisor: *plot "
        "how vitamin D concentration changes with exposure duration for a "
        "single session, including how it decays once the session is over.* "
        "Everything above and below this section scores a *schedule*; this "
        "one scores a single exposure, which is the unit those schedules are "
        "built from. It is the dose-response and washout kernel behind the "
        "trajectories in section 12 and the interval orderings in sections 5 "
        "and 5a.\n\n"
        "**What is held fixed.** One session at t = 0 and nothing afterwards, "
        "then an observation window long enough for the washout to complete. "
        "Session length in minutes is converted to erythemal SED under the "
        "single frozen reference scenario quoted at the top of this report "
        "(`vitd_scenario`) — no latitude, date or posture enters. Only the "
        "session DURATION varies between curves. Each panel also carries a "
        "no-exposure control run from the same baseline, because the model "
        "has no dietary or endogenous source term, so baseline 25(OH)D is an "
        "initial condition rather than a fixed point and the curves decay "
        "*through* it. The session's effect is therefore read as the GAP to "
        "that control, not as an absolute level.\n")
    report.append(_single_session_headline(single_df)
                  if single_df is not None else "")
    report.append(
        "\n**Limitation — the decay limb is an extrapolation, and the "
        "washout times are unvalidated.** The state-dependent decay rate "
        "`ModelParams.k_decay_D` (Datta 2017 power law) was fitted over a "
        "25(OH)D range that this figure sits almost entirely *below*, so the "
        "half-life the power law returns here is longer than any half-life "
        "Datta measured. The script computes the fitted range and the model "
        "half-lives inside and outside it at run time and renders them both "
        "into the log below and onto the figure itself, so they cannot drift "
        "from the parameterisation:\n")
    _ss_caveat = ""
    try:
        _ssm = _import_script_module("single_session_response")
        from vitd_model import ModelParams as _MP2
        _b = _MP2()
        _lo, _hi = _ssm.DATTA_FITTED_D_RANGE
        _ss_caveat = (
            "\n- `k_decay_D` was fitted over 25(OH)D **%g-%g nmol/L**; model "
            "half-life is **%.0f d** at %g nmol/L and **%.0f d** at %g "
            "nmol/L (both inside the fitted range), against **%.0f d** at "
            "the %g nmol/L default baseline this figure starts from — "
            "outside it.\n"
            % (_lo, _hi, _ssm._thalf(_b, _hi), _hi, _ssm._thalf(_b, _lo),
               _lo, _ssm._thalf(_b, _b.D0_default), _b.D0_default))
    except Exception as _exc:                      # pragma: no cover - guard
        _ss_caveat = ("\n- *(computed caveat unavailable: %s: %s — read the "
                      "banner on the figure)*\n" % (type(_exc).__name__, _exc))
    report.append(_ss_caveat)
    report.append(
        "\nPeak heights and the marginal-yield ranking do **not** depend on "
        "that rate — they are set by the synthesis term at t = 0 — but every "
        "washout time in this section does, and should be treated as "
        "unvalidated. A second caveat is structural: in the default model a "
        "session's synthesis is credited to serum as an instantaneous "
        "impulse, so the rise is a vertical step rather than a ramp. That is "
        "a property of the impulse formulation, not a physiological claim; "
        "the transit-delay variant (`ModelParams.use_transit_delay`, whose "
        "half-life is flagged `[PRIOR]` in `vitd_model.py`) is plotted "
        "alongside it, and the default column is the one every other headline "
        "number in this report uses.\n")
    if single_err:
        report.append(
            "\n> **This section failed on this run** (`%s`). The figure and "
            "table referenced below may be stale or absent; the traceback is "
            "in the log block.\n" % single_err)
    report.append("\n```\n" + single_log + "\n```\n")
    report.append("\n![single session dose-response and washout]"
                  "(figs/single_session_response.png)\n")
    report.append("\nPer-duration table: "
                  "[`tables/single_session_response.csv`]"
                  "(tables/single_session_response.csv).\n")

    report.append("\n## 2. Alpha sensitivity sweep (Priority 1 completed in Session 2)\n")
    report.append(
        "\nVaries the photoadaptation rise rate (alpha) and checks whether the "
        "optimal inter-session interval shifts. Alpha was calibrated from "
        "light-skin Miyamura 2011 data; whether it is larger in FST V-VI is "
        "untested, which is the reason for the sweep.\n"
    )
    report.append("\n```\n" + alpha_log + "\n```\n")
    report.append("\n![alpha sweep, absolute D](figs/alpha_sweep_absolute_D.png)\n")

    report.append("\n## 3. beta sensitivity sweep (tanning decay rate, NEW)\n")
    report.append(
        "\nVaries beta (tanning decay rate). Ravnbak et al. 2008 (PMID 18480919) "
        "measured UVB tanning fade of 4-5 months in Scandinavians and "
        "Indians/Pakistanis. The prior citation (Tadokoro 2005, PMID 15955111) "
        "only measured 0-7 days post-exposure and does not report a decay "
        "timescale. The 56-day half-life (default beta) is a mid-range estimate "
        "compatible with Ravnbak 2008 if complete fading requires ~3 e-folding "
        "times. The sweep covers half-lives of 21-120 days.\n"
    )
    report.append("\n```\n" + beta_log + "\n```\n")
    report.append("\n![beta sweep](figs/beta_sweep.png)\n")

    report.append("\n## 4. K_ATTEN sensitivity sweep\n")
    report.append(
        "\nVaries the melanin attenuation coefficient `K_ATTEN`, the "
        "*induced*-pigment term. Its calibrated value and provenance live in "
        "the `vitd_model.py` module docstring and are not restated here; the "
        f"value this run used is **{K_ATTEN_NOW:g}**, read from "
        "`ModelParams()` at report time so this line cannot drift from the "
        "code. The sweep explores sensitivity around it.\n\n"
        "Note that `K_ATTEN` is the coefficient on *acquired* melanin "
        "`exp(-K_ATTEN * (M - M_baseline))`; the constitutive skin-type "
        "penalty is `K_const`, a separate parameter, and the two are jointly "
        "unidentifiable from the present data (section 16).\n"
    )
    report.append("\n```\n" + k_log + "\n```\n")
    report.append("\n![K_ATTEN sweep](figs/K_ATTEN_sweep.png)\n")

    # Section 5 assumption block. Every quantity is read from vitd_analyses'
    # module constants and ModelParams() at report time, so the assumptions
    # stated in the prose cannot drift from the ones the analysis ran under.
    _TF_DAYS = vitd_analyses.PROTOCOL_DAYS
    _TF_TOTAL_SED = vitd_analyses.TWO_FRAMING_TOTAL_SED
    _TF_PER_SESSION = vitd_analyses.TWO_FRAMING_PER_SESSION_SED
    _TF_FSTS = ", ".join(str(f) for f in vitd_analyses.TWO_FRAMING_FSTS)
    _TF_INTERVALS = ", ".join(str(d) for d in vitd_analyses.INTERVALS_DAYS)
    report.append("\n## 5. Two-framing schedule comparison\n")
    report.append(
        "\nCentral handoff finding: the optimal interval depends on what's being "
        "optimized. A schedule claim only means something relative to what is "
        "held fixed, so the conditions come first and the claim is stated "
        "conditioned on them.\n\n"
        "### What is held fixed\n\n"
        "Identical across **every row of both framings**. Each value is read "
        "from `vitd_analyses`' module constants or `vitd_model.ModelParams()` "
        "at report time, so this list cannot drift from the code that ran:\n\n"
        f"- **Horizon**: {_TF_DAYS:g} days ({_TF_DAYS / 7:g} weeks), "
        "`vitd_analyses.PROTOCOL_DAYS`. Every schedule is scored over the same "
        "window, and steady-state D is read from the end of it.\n"
        "- **Baseline 25(OH)D**: every arm starts at `ModelParams.D0_default` "
        f"= {_p.D0_default:g} nmol/L. No arm is given a head start.\n"
        f"- **Exposed BSA**: `ModelParams.exposed_bsa` = {_p.exposed_bsa:g} of "
        "body surface, the same in every arm. Posture, clothing and body-site "
        "geometry are not varied, so a longer interval buys no extra skin.\n"
        f"- **Skin types**: FST {_TF_FSTS} only "
        "(`vitd_analyses.TWO_FRAMING_FSTS`) - the dark-skin target population. "
        "FST I-IV are not in this table and nothing here should be read as a "
        "light-skin result.\n"
        "- **Feedback mode**: `ModelParams.feedback_mode` = "
        f"'{_p.feedback_mode}'. Substrate feedback is active, so a large single "
        "dose is penalised for landing on an already-higher 25(OH)D.\n"
        f"- **UV source**: `ModelParams.E_source` = {_p.E_source:g}, the "
        "broadband solar reference. This section does **not** use the "
        "`solar_summer` calibration that section 11 uses, so absolute D values "
        "here are not comparable with section 11's.\n"
        "- **Spacing is uniform only**: sessions come from "
        "`vitd_model.build_pulse_schedule`, which emits equally spaced pulses "
        "from day 0. Irregular, clustered and burst schedules are never tested "
        f"here. The intervals compared are the fixed grid {_TF_INTERVALS} days "
        "(`vitd_analyses.INTERVALS_DAYS`).\n"
        "- **Both photoadaptation channels are active**: induced melanin and "
        "stratum-corneum thickening both run at their `ModelParams()` defaults, "
        "starting from `M = M_baseline(FST)` and `SC = 0`. Neither is ablated, "
        "so the penalty for concentrating UV into few large sessions carries "
        "both mechanisms.\n"
        "- **Doses are absolute SED, not a fraction of MED.** Both framings "
        "specify UV in SED and neither rescales by the individual MED. Under "
        "fractional-MED framing the skin-type response pattern differs "
        "materially (section 0), so these rows do not transfer to it.\n\n"
        "### What each framing varies\n\n"
        f"- **Framing 1**: matched cumulative SED - a fixed {_TF_TOTAL_SED:g} "
        f"SED budget spread over the {_TF_DAYS:g}-day horizon. Total UV is the "
        "constraint and the only free choice is how to distribute it, so a "
        "longer interval necessarily means a larger per-session dose.\n"
        f"- **Framing 2**: matched per-session dose - {_TF_PER_SESSION:g} SED "
        "every session regardless of interval. Per-session dose is the "
        "constraint; cumulative UV falls as the interval lengthens, so these "
        "arms are *not* matched on total exposure.\n\n"
        "### The claim, conditioned\n\n"
        "**Under Framing 1 - at fixed total UV over the horizon, with "
        "everything in the list above held fixed - daily dosing beats every "
        "longer spacing tested.** It wins on steady-state D, final D, net "
        "delta-D per SED and AUC above baseline per SED, for both skin types. "
        "The relationship is **monotone in interval across the whole tested "
        "grid**: every step from daily toward wider spacing is worse than the "
        "step before it, and there is no interior optimum. The finding is "
        "therefore 'as often as you can manage', not 'an optimal gap of N "
        "days'.\n\n"
        "Two qualifications, both visible in the log below. First, part of what "
        "rules out the widest spacing is **dose safety, not vitamin D "
        "efficiency**: concentrating the whole budget into the "
        "longest-interval schedule pushes a single session above one MED, "
        "flagged as an overdose on the safety line. Second, the monotonicity "
        "is a Framing-1 property. Under Framing 2 daily still reaches the "
        "highest 25(OH)D, but only by delivering far more total UV than the "
        "wide-interval arms (compare the `cumulative_SED` column), and on the "
        "per-SED efficiency columns Framing 2 has an **interior** optimum at an "
        "intermediate interval where daily is not the most UV-efficient "
        "choice. 'Daily is best' and 'daily is most efficient per unit UV' are "
        "different claims, and only the first survives both framings.\n\n"
        "### Out of scope for this section\n\n"
        "Both framings vary the **interval**. Neither holds frequency fixed "
        "while varying the per-session dose, so nothing here addresses **daily "
        "small doses versus daily large doses** - that comparison is not in the "
        "table and this section cannot answer it. The dose axis is treated "
        "elsewhere: section 10 compares one session against a split one at a "
        "matched UV rate, and section 13b frees the per-session dose while "
        "holding the session count fixed.\n"
    )
    report.append("\n```\n" + frame_log + "\n```\n")
    report.append("\n![two framings](figs/two_framings.png)\n")

    # ---- 5a. Front-load vs split at a fixed budget ------------------------
    # 5a, not a new integer: this sits directly against section 5's
    # two-framing comparison, and the report's later sections cross-reference
    # each other by number, so nothing may be renumbered.
    report.append("\n## 5a. Front-load or split? A fixed UV budget, three "
                  "mechanism rungs\n")
    report.append(
        "\n**The question this answers**, put by the project's advisor: *for "
        "a fixed total exposure time per week, is it better to front load "
        "rather than split time across days?* — and, explicitly, *answer it "
        "with just melanin dynamics first, then add skin thickening, then "
        "add skin cancer risk.* That staging is why this section is run at "
        "three rungs of `vitd_ablation` rather than once: `melanin_only`, "
        "`melanin_plus_sc`, and `melanin_sc_risk`.\n\n"
        "Section 5 varies the **interval** at a fixed budget. This section "
        "separates the two distinct things \"front-loading\" can mean and "
        "reports them as separate arms:\n\n"
        "- **Arm (a), within-week concentration.** The weekly SED budget is "
        "held fixed and only the number of days it is spread over changes, "
        "from a session every day down to one session a week. The pattern "
        "repeats every week, so nothing is front-loaded across the horizon — "
        "only within each week. This is the advisor's question read "
        "literally.\n"
        "- **Arm (b), across-horizon taper.** The total SED over the whole "
        "horizon is held fixed and the dose *profile* is varied from flat to "
        "front-loaded-and-tapering (two-phase and exponential families), "
        "crossed with a monotone spacing warp (`gamma = 1` is uniform, "
        "`> 1` front-loaded). Every profile is rescaled to deliver exactly "
        "the same total SED, so only the shape is swept.\n\n"
        "**What is held fixed.** The same frozen reference scenario as every "
        "other section; the same baseline; FST V and VI as the priority "
        "population with FST II carried only as an explicitly-labelled "
        "light-skin contrast; and, within each comparison, the SED budget. "
        "The method is a **deterministic grid sweep with no constraint "
        "applied** — deliberately not the risk-constrained optimisers of "
        "sections 13b/13c, whose sufficiency floor and erythema cap would "
        "silently delete exactly the concentrated arms the question is "
        "about. Because no cap is applied, every row instead carries the "
        "largest single session as a fraction of MED, checked against the "
        f"sub-erythemal ceiling ({GAMMA_MED:.0%} MED) and against a frank "
        "burn at 1.0 MED. An arm that wins while requiring a burn has not "
        "won, and is flagged as such in the CSV, the console summary and the "
        "plots.\n\n"
        "**The answer in one line, because the two arms answer different "
        "halves of it.** The steady-state *plateau* is set by the budget "
        "and by how finely it is split across days (arm a); rearranging "
        "the same budget over the horizon does not move it (arm b, a "
        "null that gets cleaner at the longer horizon). What "
        "front-loading actually buys is the *transient*: reaching "
        "sufficiency fast and minimising the deficiency accumulated on "
        "the way. Arm (b) is therefore ranked on accumulated deficiency "
        "below sufficiency - a horizon-integrated measure - with "
        "steady-state and final 25(OH)D retained as reported secondary "
        "terminal measures. Limitation 1 below says why that choice is "
        "not cosmetic.\n")
    report.append(_frontload_headline(frontload_df)
                  if frontload_df is not None else "")
    report.append(
        "\n**Limitations — these three constrain what may be read off this "
        "section.**\n\n"
        "1. **Session count must be matched before any taper claim is made, "
        "and the outcome measure must be named.** An unmatched comparison - "
        "one that lets the tapered arm also use a different number of "
        "sessions - shows a gain of several percent that is an artefact of "
        "the session-count difference, not of the dose shape. Referenced to "
        "the flat, uniformly spaced arm at the *same* N, the plateau effect "
        "is sub-percent at the protocol horizon and vanishes entirely at the "
        "longer one. Any \"front-loading helps\" claim taken from this "
        "material must therefore say (i) whether N was matched - if it was "
        "not, the claim is about how often you go outside, not about how the "
        "dose is shaped - and (ii) which outcome it was read off, because "
        "the sign of the taper depends on that choice and the dependence "
        "**widens with the horizon**. The terminal measures "
        "(`steady_state_D`, `final_D`) reward whatever sits nearest the "
        "endpoint; the horizon-integrated measure (deficit AUC below "
        "sufficiency) rewards getting there early. Arm (b) is ranked on the "
        "integrated measure for that reason, with the terminal measures "
        "retained as reported secondaries, and the disagreement is plotted "
        "rather than resolved by picking a favourite.\n"
        "2. **The back-loaded winner on the terminal measure is an artefact, "
        "not a schedule.** At the long horizon the arm that maximises "
        "steady-state 25(OH)D does so by maximum back-loading: it opens with "
        "a long no-exposure gap, leaves the subject deficient for most of "
        "the year, and stacks dose underneath the final-28-day averaging "
        "window. The numbers that expose it - its deficit AUC relative to "
        "the flat reference, and its days-to-sufficiency - are generated "
        "above so the \"gain\" cannot be quoted bare.\n"
        "3. **`melanin_only` is an ABLATION, not a re-fit.** Turning the "
        "stratum-corneum channel off does not re-estimate anything, so "
        "absolute 25(OH)D levels under that rung are a mechanism-isolation "
        "diagnostic and are **not comparable across rungs**. Only *within*-"
        "rung schedule rankings are defensible. This is why the headline "
        "figure plots each arm relative to its own rung's reference schedule "
        "and the absolute-level figure keeps the rungs in separate panels. "
        "Relatedly, `melanin_sc_risk` is dynamically **identical** to "
        "`melanin_plus_sc` — it is a *reporting* rung that obliges the "
        "analysis to show the skin-cancer indices `simulate()` computes "
        "anyway, not \"cancer risk added to the model\", and its trace sits "
        "exactly on top of `melanin_plus_sc` on every plot.\n")
    if frontload_err:
        report.append(
            "\n> **This section failed on this run** (`%s`). The figures and "
            "tables referenced below may be stale or absent; the traceback is "
            "in the log block.\n" % frontload_err)
    report.append("\n```\n" + frontload_log + "\n```\n")
    report.append("\n![front-load vs split, within week]"
                  "(figs/frontload_within_week.png)\n")
    report.append("\n![front-load vs split, absolute levels by rung]"
                  "(figs/frontload_within_week_absolute.png)\n")
    report.append("\n![across-horizon taper, protocol horizon]"
                  "(figs/frontload_horizon_taper.png)\n")
    report.append("\n![across-horizon taper, 365-day horizon]"
                  "(figs/frontload_horizon_taper_365d.png)\n")
    report.append("\n![taper metric conflict: terminal vs horizon-integrated "
                  "outcomes](figs/frontload_taper_metric_conflict.png)\n")
    report.append("\n![front-load safety sidecar](figs/frontload_safety.png)\n")
    report.append(
        "\nTables: [`tables/frontload_within_week.csv`]"
        "(tables/frontload_within_week.csv) (arm a), "
        "[`tables/frontload_horizon_taper.csv`]"
        "(tables/frontload_horizon_taper.csv) (arm b), and "
        "[`tables/frontload_taper_horizon_comparison.csv`]"
        "(tables/frontload_taper_horizon_comparison.csv) (the matched-N "
        "horizon check, one row per rung x skin type x budget x session "
        "count, carrying `best_gain_pct`, `spacing_only_gain_pct` and the "
        "three argmax-gamma columns the paragraph above is computed from). "
        "The 365-day re-run of arm (b) writes "
        "[`tables/frontload_horizon_taper_365d.csv`]"
        "(tables/frontload_horizon_taper_365d.csv) and its own figure. If "
        "the script is run over a different set of horizons it writes one "
        "`tables/frontload_horizon_taper_<N>d.csv` and one matching figure "
        "per extra horizon; the exact filenames written on this run are "
        "listed at the end of the log above.\n")

    report.append("\n## 6. Fine-grid interval sweep\n")
    report.append(
        "\nSweeps inter-session intervals in 1-day steps to find the exact "
        "optimal interval under each framing, complementing the coarse "
        "integer-interval analysis in the two-framing comparison. The interval "
        "range and skin types are set in "
        "`vitd_analyses.analysis_fine_grid` and appear in the log below; this "
        "line used to name a 1-35 day range the code does not sweep.\n"
    )
    report.append("\n```\n" + fine_log + "\n```\n")
    report.append("\n![fine grid efficiency](figs/fine_grid_efficiency.png)\n")

    report.append("\n## 7. Exhaustive integer interval sweep (1-60 days)\n")
    report.append(
        "\nSweeps every integer interval from 1 to 60 days to find the interval "
        "that maximizes net ΔD per SED and, separately, steady-state D.\n\n"
        "**This section was wrong until 2026-08-21 and its answer has changed.** "
        "It used to call `scipy.optimize.minimize_scalar` (bounded Brent) on a "
        "continuous-valued interval, but the objective *rounded that value to "
        "the nearest integer day* before building a schedule — so Brent's "
        "method was searching a piecewise-constant staircase. That is not a "
        "valid use of a continuous line search, and it is not reproducible. It "
        "reported an optimum of **2 days** while section 6's exhaustive sweep "
        "reported **1 day**, and the prose here claimed the two agreed. They "
        "did not. Replaced with the exhaustive integer sweep the interval "
        "variable actually warrants; the optimum is now **1 day** for both FST "
        "V and FST VI, matching section 6 exactly.\n\n"
        "Because this is now the same method as section 6 at a wider range "
        "(1-60d vs 1-30d), agreement between the two where their ranges "
        "overlap is **expected by construction and is not an independent "
        "cross-check**. It is reported as a range extension, nothing more.\n"
    )
    report.append("\n```\n" + scipy_log + "\n```\n")

    report.append("\n## 8. Maximum interval for sufficiency\n")
    report.append(
        "\nFinds the longest inter-session interval where steady-state D stays "
        "above 50 nmol/L under 1 SED/session dosing.\n"
    )
    report.append("\n```\n" + maxint_log + "\n```\n")
    report.append("\n![max interval](figs/max_interval_sufficiency.png)\n")

    report.append("\n## 9. Loading dose optimization\n")
    report.append(
        "\nFor UV budgets of 2-20 SED in the first 14 days, tests splitting "
        "that budget across a set of session counts between 1 and 14 (the "
        "exact splits are `DISTRIBUTIONS` in "
        "`vitd_analyses.analysis_loading_dose` and are listed in the log "
        "below, not every integer in the range). After loading, all schedules "
        "switch to 1 SED every 7 days maintenance.\n"
    )
    report.append("\n```\n" + load_log + "\n```\n")
    report.append("\n![loading dose](figs/loading_dose_D_at_day14.png)\n")

    report.append("\n## 10. Maintenance dose splitting\n")
    report.append(
        "\nCompares 1 SED every T days vs 0.5 SED every T/2 days at the same "
        "total UV rate. Tests T = 7, 14, 21 days.\n"
    )
    report.append("\n```\n" + split_log + "\n```\n")
    report.append("\n![maintenance split](figs/maintenance_split.png)\n")

    report.append("\n## 11. Sun exposure scheduling (no equipment)\n")
    report.append(
        "\nPractical sun exposure scheduling: how many minutes of midday sun, "
        "how often, to maintain sufficiency? Runs entirely on the frozen "
        "reference scenario above — no latitude, no date, no posture. "
        f"`E_source` is the solar value keyed `{E_SOURCE_SOLAR_DEFAULT}` "
        f"(**{E_SOLAR_NOW:g}** on this run), not the neutral 1.0 this line "
        "used to claim — `analysis_sun_exposure` switched to the solar "
        "calibration on 2026-08-16 and the prose here did not follow.\n\n"
        "**The minute figures in this section changed on 2026-08-21.** This "
        "analysis previously used a hardcoded `MIN_PER_SED = 6.7` next to a "
        "docstring claiming UV index ~5 and 0.25 W/m² erythemal — three "
        "mutually inconsistent statements. 1 UV Index unit is 25 mW/m² "
        "(WHO/WMO), so 0.25 W/m² is UV index 10, not 5; and 6.7 min/SED "
        "applies **no horizontal-to-personal geometry factor at all**, while "
        "every other minutes figure in this report applies "
        "`PERSONAL_DOSE_GEOMETRY_FACTOR`. This section was therefore "
        "under-reporting required minutes by roughly 4.7× relative to the "
        "rest of the repository. It now reads its conversion from "
        "`vitd_scenario.py`. This is a **unit/geometry bug fix, not a "
        "calibration change** — no scientific parameter moved and "
        "`calibration_approval.yaml` was not touched.\n"
    )
    report.append("\n```\n" + sun_log + "\n```\n")
    report.append("\n![sun exposure](figs/sun_exposure_minutes_vs_D.png)\n")

    report.append("\n## 12. Example trajectories\n")
    report.append(
        "\nD(t) and M(t) under three representative schedules at fixed 1 SED per "
        "session. Shows the temporal dynamics directly.\n"
    )
    report.append("\n![trajectories](figs/trajectories.png)\n")

    report.append("\n## 13. Constant-sessions analysis\n")
    report.append(
        f"\nSweeps session counts 1-52 over {CONST_SESSIONS_DAYS} days at a "
        f"FIXED total budget of {CONST_SESSIONS_TOTAL_SED:g} SED (so the "
        f"per-session dose is {CONST_SESSIONS_TOTAL_SED:g}/N), uniform "
        f"spacing, FST {CONST_SESSIONS_FST}, ranked by AUC deficiency below "
        f"{SUFFICIENCY_NMOL_L:.0f} nmol/L, with a rolling-7-day UV safety "
        "sidecar. (This paragraph previously described a 15-session, 84-day "
        "protocol that `analysis_constant_sessions` has not run for some "
        "time.)\n"
    )
    report.append("\n```\n" + const_log + "\n```\n")

    report.append("\n## 13a. Spacing optimisation at fixed dose\n")
    report.append(
        "\nThis section fixes the dose per session (1.0 SED) and searches only "
        "the spacing between sessions. It answers 'when should I go outside?' "
        "for a person who will use the same dose each time. The best uniform "
        "gap for each (skin type, session budget) cell is reported under the "
        "mean_D objective. The table also includes the skin-cancer-convergence "
        "index (SCC) and the intermittency index for the full cancer-safety "
        "profile.\n"
    )
    report.append("\n```\n" + spacing_log + "\n```\n")

    report.append("\n## 13b. Variable-dose optimisation\n")
    report.append(
        "\nEvery other analysis in this repository fixes one dose and searches "
        "only *when* to dose. This frees the dose of each session and asks what "
        "that buys, holding the sufficiency floor and the session count fixed so "
        "the comparison cannot be won by dosing less or going outside more.\n\n"
        "At a fixed budget **and** session count, variable dose cannot help: the "
        "intermittency term is convex in the doses, so equal doses minimise it by "
        "Jensen's inequality. The gain instead comes from needing a smaller "
        "budget — a single constant dose has to both climb to the target and hold "
        "there, so it necessarily overshoots in steady state. Splitting those two "
        "jobs is the whole effect, which is also why it shrinks as the session "
        "count rises and the per-session dose is already small.\n\n"
        "`seasonal` returning the uniform optimum is the correct result, not a "
        "search failure: its grid contains the constant-dose special case "
        "(amplitude 0) and selects it. **`gap_coupled` is a different matter, "
        "and this report used to describe it wrongly.** On the uniformly-spaced "
        "days this analysis always uses, every gap equals the mean gap, so "
        "`dose_gap_coupled` returns the constant-dose profile for *every* value "
        "of its exponent `p` (verified for p in {0, 0.5, 1, 1.5, 3}). It is "
        "**degenerate under uniform spacing** — it did not select the "
        "constant-dose case from among alternatives, it could not have returned "
        "anything else. The family carries information only when day spacing "
        "varies, which no current caller does. "
        "Seasonality matters here as a *constraint* on achievable dose, not as a "
        "free choice — see `dose_seasonal`.\n\n"
        "Time-to-sufficiency is now reported because constraint C3 starts at "
        "`RAMP_IN_DAYS` and so cannot see how fast a schedule climbs; two "
        "schedules that both hold the floor from day 90 are scored identically by "
        "C3 even when one reaches sufficiency a month sooner.\n"
    )
    report.append("\n```\n" + vardose_log + "\n```\n")
    report.append("\n![variable dose profiles](figs/variable_dose_profiles.png)\n")

    report.append("\n## 13c. Joint timing + dose optimisation\n")
    report.append(
        "\nSection 13b frees the dose but keeps sessions uniformly spaced. This "
        "searches spacing and dose together, using a one-parameter monotone warp "
        "of the session days (`gamma = 1` is uniform) on top of the dose "
        "families.\n\n"
        "**A session budget is part of the problem, not an add-on.** Without one "
        "the joint optimum is degenerate: at fixed cumulative dose the "
        "intermittency term is minimised by spreading over as many days as "
        "possible, so risk falls monotonically all the way to a daily schedule — "
        "which is what section 14b already reports from a different direction. "
        "The meaningful object is therefore the frontier over a session budget, "
        "and that budget is set by adherence rather than photobiology.\n\n"
        "**This does not need to be mixed-integer.** `d_t = 0` already means "
        "'no session that day', so a dose vector encodes timing and dose together "
        "with no integer variables — the encoding used in maximum-hands-off "
        "control (Nagahara, Quevedo & Nešić 2016, *IEEE TAC* 61:735-747). Their "
        "L1 = L0 equivalence is proved for linear dynamics and is not established "
        "for this nonlinear system, so it motivates the formulation rather than "
        "justifying it; `scripts/joint_schedule_freesolve.py` checks the answer "
        "against a direct 365-variable solve instead of assuming it.\n\n"
        "**The comparison is against uniform spacing searched by the same "
        "solver at the same budget**, not against the coarse grid of 13b. The "
        "spacing dimension is present in both arms and merely inert in the "
        "control, because `differential_evolution` sizes its population as "
        "`popsize × n_parameters` — giving the free arm an extra dimension would "
        "also give it ~33% more evaluations and confound spacing with search "
        "effort.\n"
    )
    report.append("\n```\n" + joint_log + "\n```\n")

    # G5: measure solver noise at the one cell the in-code CAVEAT already
    # names (FST III, N=146) rather than only inferring a lower bound from
    # the most negative gain across the ~22-cell grid above. n_seeds=5 means
    # this cell is solved five times instead of once -- real but bounded
    # added wall-clock, guarded with capture_log_safe so a failure here costs
    # only this paragraph.
    def _replicate_solver_noise():
        from vitd_optimization import optimise_joint_schedule as _ojs
        r = _ojs(fst=3, n_sessions=146, n_seeds=5)
        if r is None or "risk_spread_frac" not in r:
            print("  No feasible schedule / no spread reported at this cell.")
            return None
        print(f"  FST III, N=146, {r['n_seeds_run']} seeds: combined_risk "
              f"spread {100.0 * r['risk_spread_frac']:.1f}% of the best "
              f"value ({r['risk_min']:.4f} to {r['risk_max']:.4f}).")
        return r

    _seed_log, _seed_res, _seed_err = capture_log_safe(
        "13c solver-noise replication", _replicate_solver_noise)
    report.append(
        "\n**Measured solver noise, not only inferred.** The bound quoted "
        "in the log above is a lower bound taken from the most negative "
        "gain across cells at a single seed (see the CAVEAT in "
        "`vitd_optimization.analysis_joint_schedule`). Re-running the one "
        "representative cell the CAVEAT names (FST III, N = 146) at "
        "multiple seeds gives a directly measured spread instead:\n")
    if _seed_err:
        report.append(
            "\n> **This replication failed on this run** (`%s`). No "
            "measured bound available; fall back to the inferred one "
            "above.\n" % _seed_err)
    elif _seed_res is None:
        report.append(
            "\n*(No feasible schedule or no spread reported at this cell "
            "on this run.)*\n")
    else:
        report.append(
            "\n**%d seeds: combined-risk spread %.1f%% of the best value "
            "(%.4f to %.4f).** Per-cell gains in the grid above of roughly "
            "that magnitude or smaller are solver noise and must not be "
            "compared across cells; the SIGN of the free-vs-uniform gain is "
            "what replicates, not any single-cell percentage.\n"
            % (_seed_res["n_seeds_run"], 100.0 * _seed_res["risk_spread_frac"],
               _seed_res["risk_min"], _seed_res["risk_max"]))
    report.append("\n```\n" + _seed_log + "\n```\n")

    report.append("\n## 13d. Tri-objective: sessions × spacing × skin-cancer risk\n")
    report.append(
        "\nSections 13b and 13c minimise **one** objective (the combined "
        "cancer-risk index) at a fixed session budget. This makes the "
        "three-way trade-off explicit instead, over the schedules that "
        "already work:\n\n"
        "| axis | quantity | why it is separate |\n"
        "|---|---|---|\n"
        "| A1 | `n_sessions` per year | adherence — how often you must get outside |\n"
        "| A2 | `max_gap_days` | the longest stretch with no session, tail gap to the end of the horizon included |\n"
        "| A3 | `risk_within_fst_ratio` | the skin-cancer index, **within skin type only** |\n\n"
        "Efficacy is a **constraint**, not an axis: every point on this "
        f"frontier holds {SUFFICIENCY_NMOL_L:.0f} nmol/L for every day after "
        f"ramp-in (C3) at a dose at or below {GAMMA_MED:.0%} MED (C1) that "
        f"fits in {PRACTICAL_MAX_MINUTES:.0f} minutes (C7). The `totSED` "
        "column is what that constraint cost — which is the \"best SED\" part "
        "of the question, reported rather than optimised away.\n\n"
        "**The three axes genuinely conflict, each pair for a different "
        "reason.** A1 against A2: uniform spacing minimises the largest gap at "
        "any budget, so cutting sessions lengthens every gap proportionally. "
        "A1 against A3: fewer sessions must each carry a larger dose, and the "
        "intermittency term is cubic in per-session dose. A2 against A3: the "
        "risk-minimising schedule front-loads, which shortens the early gaps "
        "by lengthening the late ones. A weighted sum would hide all three; "
        "the nondominated set shows them.\n\n"
        "Spacing uses the same one-parameter monotone warp as section 13c "
        "(`gamma = 1` is uniform), so the two sections search the same space "
        "and their answers are directly comparable. The inner dose search is a "
        "deterministic shape grid rather than differential evolution: the "
        "session days are already fixed at each cell, and DE run-to-run spread "
        "(bounded at roughly 4% by 13c) would otherwise swamp the differences "
        "between adjacent budgets.\n\n"
        "Infeasible `(budget, spacing)` cells are printed, not dropped. Some "
        "budgets are feasible under a warped spacing and infeasible under "
        "uniform spacing at the same budget and the same ceiling: constraint "
        f"C3 binds at the **ramp-in boundary** (day {RAMP_IN_DAYS}), not at "
        "the end of the year, so a schedule that reaches target faster can "
        "hold a budget that uniform spacing misses by a fraction of a "
        "nmol/L.\n"
    )
    report.append("\n```\n" + tri_log + "\n```\n")

    # Decision problems, held-out benchmark, and uncertainty. Read first.
    # Sweeps now select by steady_state_D; everything below is posed as a
    # real decision.
    import vitd_objectives
    import vitd_benchmark_webbkift
    import vitd_sensitivity

    # Collapses the Q2 frontier to one actionable schedule per skin type and
    # saves the frontier figure. Routed through _import_script_module (not a
    # bare `import risk_aware_scheduling`) so this scripts/ dependency is
    # visible to CLAUDE.md's documented detection rule ("grep run_all.py for
    # _import_script_module and the script's own name before ignoring
    # anything under scripts/") and so an import-time failure (a mid-edit
    # file, a missing optional dependency) degrades this one section instead
    # of raising out of an un-wrapped `import` statement at module-exec time.
    def _run_risk_aware_scheduling():
        return _import_script_module("risk_aware_scheduling").main()

    obj_log = capture_log(_report_decision_problems)
    # G2: keep the return value so section 15 can headline PASS/FAIL instead
    # of leaving the only out-of-sample verdict buried inside a fenced block.
    bench_log, bench_res, bench_err = capture_log_safe(
        "15. Webb/Kift 2018 held-out benchmark", vitd_benchmark_webbkift.run)
    # G3(ii)/G7: keep the return value for the identifiability paragraph in
    # section 17, and run at both FST 5 (historical default) and FST 6 (the
    # project's stated priority population, previously never covered here).
    sens_log, sens_res, sens_err = capture_log_safe(
        "16. Sensitivity / identifiability / uncertainty", vitd_sensitivity.run)
    # Was `capture_log(risk_aware_scheduling.main)` -- a bare capture_log lets
    # a failure here abort the rest of the pipeline (validation, sweeps,
    # everything after this point in the function) instead of degrading only
    # this section, unlike every other scripts/-backed analysis in this file.
    risk_log, risk_res, risk_err = capture_log_safe(
        "14b. Risk-constrained schedule recommendation",
        _run_risk_aware_scheduling)
    if risk_err:
        print("WARNING: risk-aware scheduling section failed (%s) -- "
              "pipeline continued." % risk_err)
    risk_anchor_log = "\n".join(
        capture_log(fn) for fn in (
            vitd_calibration.elwood1997_intermittency_check,
            vitd_calibration.moan1989_scc_exponent_check,
            vitd_calibration.skin_type_susceptibility_check,
        ))
    # G6: the one place a shipped parameter is checked against the source
    # most often cited for it (Matsuoka 1990 vs K_ATTEN). Cheap -- no
    # optimisation -- so run unconditionally alongside the other calibration
    # checks above.
    katten_log, katten_res, katten_err = capture_log_safe(
        "14d. K_ATTEN vs Matsuoka 1990", vitd_calibration.run)
    print(obj_log)
    print(bench_log)
    print(sens_log)
    print(risk_log)
    print(katten_log)

    report.append("\n## 14. Decision problems (recommended objectives)\n")
    report.append(
        "\nThe sweeps above used to maximise the retired per-SED efficiency "
        "metric (removed from the code on 2026-07-31), which rose "
        "monotonically as dosing approaches zero and whose reported optima "
        "ended *below* their starting 25(OH)D. These are the same questions "
        "posed so that cumulative UV is held fixed, or so that cancer risk "
        "is scored alongside the vitamin D outcome.\n"
    )
    report.append("\n```\n" + obj_log + "\n```\n")

    report.append("\n### 14b. Risk-constrained schedule recommendation (constraint C9)\n")
    report.append(
        "\nQ2 above returns a frontier, and a frontier is not advice. This "
        "collapses it to one schedule per skin type: minimise the combined "
        "cancer-risk index subject to never burning (C1), holding the "
        "sufficiency floor after the loading period (C3), a session-length "
        "ceiling (C7), and an optional risk cap (C9).\n\n"
        "All three axes of the risk index are now literature-anchored, with "
        "different evidence status each. Interval and minutes do not depend on "
        "the skin-type axis; cross-skin-type comparisons of the index values "
        "do.\n"
    )
    report.append("\n```\n" + risk_log + "\n```\n")
    report.append("\n![deficiency vs risk frontier](figs/deficiency_risk_frontier.png)\n")
    report.append("\nPer-skin-type table: [`tables/risk_aware_schedule_by_fst.csv`]"
                  "(tables/risk_aware_schedule_by_fst.csv).\n")

    report.append("\n### 14c. What anchors the three axes of the risk index\n")
    report.append(
        "\n- **Intensity** (`INTERMITTENT_BURN_EXPONENT`): anchored to the "
        "Elwood & Jopson 1997 odds ratios, to an *interval* rather than a "
        "point — the ORs give one equation in two unknowns.\n"
        "- **Cumulative dose** (`SCC_DOSE_EXPONENT`): the biological "
        "amplification factor. Super-linear, Moan 1989 Ab = 1.6-1.8 for SCC, "
        "erythemally weighted. This term used to be linear.\n"
        "- **Skin type** (`basal_damage_fraction`): derived from measured "
        "basal-layer CPD, fitted to one number and checked against three it "
        "was not fitted to. It replaced an order-of-magnitude prior.\n"
    )
    report.append("\n```\n" + risk_anchor_log + "\n```\n")

    # ---- 14d. Parameter-vs-source consistency (K_ATTEN) -------------------
    # New subsection rather than a new integer section, so no existing
    # cross-reference by number breaks (G6).
    report.append("\n### 14d. Parameter-vs-source consistency (K_ATTEN)\n")
    report.append(
        "\nThe one place in this repository where a shipped parameter is "
        "checked against the source most often cited for it. `K_ATTEN` is "
        "fitted to the Young 2020 per-person dataset — a between-subject "
        "constitutive quantity — while Matsuoka 1990 measures a "
        "within-subject facultative (tanning) effect. The two were never "
        "the same number. Whether the shipped value falls inside Matsuoka's "
        "implied range is recomputed below on every run rather than "
        "asserted%s\n"
        % ("." if katten_res is None else
           ", and on this run it is **%s**." %
           ("CONSISTENT" if katten_res["consistent"] else "INCONSISTENT")))
    if katten_err:
        report.append(
            "\n> **This section failed on this run** (`%s`).\n" % katten_err)
    report.append("\n```\n" + katten_log + "\n```\n")

    report.append("\n## 15. Held-out external benchmark (Webb/Kift 2018)\n")
    report.append(
        "\nNothing in this repository is calibrated to this study. Every other "
        "check in the validation suite is a calibration or debug target, so this "
        "is the only out-of-sample evidence available.\n"
    )
    if bench_res is not None:
        report.append(
            "\n**%s — model FST V:II dose ratio %.2fx against the reported "
            "%s band.** This is the only un-fitted external comparison in "
            "the pipeline, headlined here rather than left inside the log "
            "below, because a reader who skims section 1's pass counts would "
            "otherwise never meet it. The minutes ratio (%s) is a diagnostic "
            "restatement of the same underlying Webb/Kift measurement, not a "
            "second independent check (see the log and "
            "`vitd_benchmark_webbkift.py`'s module docstring).\n"
            % ("PASS" if bench_res["webbkift_dose_ratio"] else "**FAIL**",
               bench_res["model_dose_ratio"],
               "%.1f-%.1fx" % (vitd_benchmark_webbkift.WK_DOSE_RATIO_LOW,
                           vitd_benchmark_webbkift.WK_DOSE_RATIO_HIGH),
               ("%.2fx" % bench_res["model_minutes_ratio"]
                if bench_res["model_minutes_ratio"] is not None
                else "unreachable for at least one skin type")))
    elif bench_err:
        report.append(
            "\n> **This section failed on this run** (`%s`). No verdict "
            "available.\n" % bench_err)
    report.append("\n```\n" + bench_log + "\n```\n")

    report.append("\n## 16. Sensitivity, identifiability, and uncertainty\n")
    report.append(
        "\nWhich parameters actually drive results, which pairs cannot be "
        "separated by the current data, and what the headline numbers look like "
        "as intervals rather than point estimates. Run at **FST 5** (the "
        "historical default) and **FST 6** (the project's stated priority "
        "population, not previously covered by this section) — both appear "
        "in the log below, one after the other.\n"
    )
    if sens_err:
        report.append(
            "\n> **This section failed on this run** (`%s`). Output below "
            "may be partial; section 17's identifiability paragraph falls "
            "back to a static notice.\n" % sens_err)
    report.append("\n```\n" + sens_log + "\n```\n")

    # ---- 16b. NHANES population macro-diagnostics -------------------------
    # New subsection, not a new top-level integer, so cross-references by
    # section number elsewhere are unaffected (G11).
    report.append("\n## 16b. NHANES population macro-diagnostics\n")
    if not macro_logs:
        report.append(
            "\n**Not run on this run.** These three diagnostics check that "
            "the model's population-level predictions are plausible against "
            "unsupplemented NHANES cohorts. CLAUDE.md recommends running "
            "them after any calibration, baseline, or scenario change. "
            "Re-run with `--run-macro-scripts` to populate this section.\n"
        )
    else:
        for label, log, err in macro_logs:
            report.append("\n### %s%s\n" % (label, " — FAILED" if err else ""))
            report.append("\n```\n" + (log or "").strip() + "\n```\n")
        report.append(
            "\nTables: [`tables/nhanes_macro_validation.csv`]"
            "(tables/nhanes_macro_validation.csv), "
            "[`tables/nhanes_dietary_partitioning.csv`]"
            "(tables/nhanes_dietary_partitioning.csv), "
            "[`tables/nhanes_weight_dilution_check.csv`]"
            "(tables/nhanes_weight_dilution_check.csv).\n")

    # G3(ii): generate the identifiability paragraph from section 16's own
    # output instead of typing it in. An earlier hand-written version named
    # three pairs and silently omitted `A~D_max` at ridge 1.0000 -- the pair
    # with the largest steady-state-D spread of the four, involving `A`, the
    # top driver in the sensitivity table. Read off FST 5 (the historical
    # default this paragraph always described); FST 6 is computed alongside
    # it in section 16 above but is not re-typed here.
    _IDENT_PARA = ("*(identifiability output unavailable on this run -- see "
                   "section 16's log for whatever printed before the "
                   "failure.)*" if sens_err or not sens_res else None)
    if _IDENT_PARA is None:
        _ident5 = sens_res[5]["identifiability"]
        _bad = sorted(
            ((k, v["ridge_fraction"], v["output_spread"])
             for k, v in _ident5.items() if v["ridge_fraction"] > 0.99),
            key=lambda r: -r[2])
        _IDENT_PARA = (
            "%d parameter pair(s) are jointly unidentifiable from the "
            "current data on THIS run, at FST 5 (>99%% of the "
            "steady-state-D response surface explained by one linear "
            "combination): %s. Only their combinations are constrained. "
            "This list is generated from section 16's own output rather "
            "than typed here: an earlier hand-written version of this "
            "paragraph named three pairs and silently omitted the pair "
            "involving `A`, the top driver in the sensitivity table."
            % (len(_bad),
               ", ".join("`%s` (ridge %.4f, ss_D spread %.2f)" % b
                         for b in _bad)))

    # G-heading: derive section 17's out-of-sample heading from the same
    # bench_res used to headline section 15, so the two can never disagree
    # the way a hand-typed heading once did (section 15 said PASS while this
    # heading still asserted failure).
    _WEBBKIFT_HEADING = (
        "### The one out-of-sample test's current status is unavailable"
        if bench_res is None else
        "### The one out-of-sample test now passes"
        if bench_res["webbkift_dose_ratio"] else
        "### The one out-of-sample test currently fails")

    report.append("\n## 17. Interpretation\n")
    report.append(f"""
### Read section 0 first, then the decision-problem sections

Section 0 states the objective hierarchy the rest of this report is ranked
by, and it is generated from `vitd_objectives.OBJECTIVE_HIERARCHY` rather than
written here, so the two cannot disagree. **The headline number is the PRIMARY
result in section 0**: the fewest sessions and the least total UV that reach
and then hold sufficiency, per skin type, with infeasible cells reported as
results.

Sections 2-13 used to sweep the retired per-SED efficiency metric (removed
from the code on 2026-07-31; see `vitd_model.py`). That metric had **no
interior optimum**: it rose monotonically as dosing approaches zero, so its
apparent optimum drifted upward with the search bound -- the signature of a
monotone objective rather than a peak -- and the schedule it selected ended
*below* its starting 25(OH)D, a net loss of vitamin D scored as maximal
efficiency. Both failures are now **recomputed from the current parameters on
every run** and printed in section 0
(`vitd_optimization.gross_synth_per_SED_diagnostic`), so the demotion is
re-earned rather than remembered. The sweeps now select by `steady_state_D`
instead.

The interval ordering these sweeps produce comes from **neither** photoadaptation
**nor** the D-feedback term. It is a property of the logarithmic dose-response
itself.

This paragraph previously claimed the ordering came from D-feedback ("sparse
dosing keeps mean D low, so each pulse meets less inhibition"). That claim was
tested directly in `spacing_study/` (2026-08-21) and **does not hold**:
disabling `g(D)` entirely leaves the ordering unchanged and *widens* daily
dosing's margin rather than narrowing it. Turning off every state mechanism at
once leaves daily winning by more still. What remains, and what actually carries
the result, is that `S = A * ln(1 + UV * E)` is **concave** -- so at a fixed
total UV budget, dividing that budget into more and smaller sessions always
yields more 25(OH)D. D-feedback and photoadaptation both modulate the *size* of
the gap; neither creates it.

The induced-melanin coefficient `K_ATTEN` was never a candidate cause: its value
is {K_ATTEN_NOW:g} (see the `vitd_model.py` docstring), so that term is inert
*by calibration*, not merely by ablation. `spacing_study/` swept it as far as
100 (against a literature-implied 0.288972-0.838389) inside
`allow_retired_parameter_sweep()` without ever producing a non-daily optimum
under a matched total budget.

The practical consequence is worth stating plainly, because it is
counter-intuitive and it is this report's cleanest result: **the schedule that
is best for vitamin D is also the one that is safest.** Frequent small sessions
maximise synthesis (log concavity, above) and simultaneously minimise
melanoma/BCC risk (which tracks intermittent intensity, not cumulative dose --
see "Skin cancer is now scored" below). The two objectives agree rather than
trade off.

`net_final_D_change_per_SED` is **DIAGNOSTIC-tier** in the section 0 hierarchy,
not a recommendation: it repairs the retired metric's net-loss failure by being
signed, but it is still a ratio with no sufficiency floor, so it can prefer a
schedule that never reaches target. The interval it selects, and the Monte
Carlo spread of that interval over the documented prior ranges, are in
section 16 and should be quoted from there rather than from this paragraph.

### What the sweeps do legitimately show

The interval ranking is robust to `alpha` (0.01-0.40), `beta` (21-120 d
half-life), and `K_ATTEN`. This is a real robustness result, but the reason is
not reassuring: those three parameters barely participate in the model at all.
At 1 SED/session the melanin state barely moves over an 84-day protocol.
Section 16 quantifies the comparison -- `A` swings steady-state D across its
plausible range by far more than `K_ATTEN` does, despite `K_ATTEN` carrying the
longer citation trail. The magnitudes are reported there, from that run, and
are deliberately not copied into this paragraph.

### Identifiability

{_IDENT_PARA}

Reporting either member of such a pair as a separately calibrated value
overstates what the data support -- melanin and stratum corneum are both
`exp(-k * saturating state)` terms driven by the same UV input, so they trade
off along a ridge.

{_WEBBKIFT_HEADING}

Webb/Kift 2018 report that skin type V needs {vitd_benchmark_webbkift.WK_DOSE_RATIO_LOW:.1f}-{vitd_benchmark_webbkift.WK_DOSE_RATIO_HIGH:.1f}x the dose of white
Caucasians for an equivalent 25(OH)D rise. What the model gives, and whether it
now clears that band, is in **section 15** -- read the ratio from there rather
than from this paragraph, which has previously carried a figure from an earlier
calibration. `K_const` was recalibrated on 2026-08-17 and is no longer the
single-ratio inversion of Young 2020 that this paragraph used to describe; its
current derivation, the four within-study estimates behind it, and the spread
across those routes are in the `vitd_model.py` module docstring, which is the
only canonical source for parameter values in this repository.

The distinction that made the old gap look worse than it is still holds:
matched-dose lamp ratios and whole-body naturalistic ratios measure different
quantities, and behaviour (shade, clothing, time outdoors) is a separate factor
that lives in `exploratory/vitd_behaviour.py` rather than inside `K_const`.
That module was retired from this pipeline on 2026-08-21 -- it was already
inert (its scheduling-mode multiplier is 1.0 for every skin type), so nothing
in this report moved. The separation argument stands; it is simply no longer
reported here. See `exploratory/README.md`.

### Skin cancer is now scored, not just flagged

The risk model separates the two epidemiologically distinct pathways: SCC tracks
**cumulative** dose, while melanoma and BCC track **intermittent** intensity
(Elwood 1997: intermittent OR 1.71, heavy occupational OR 0.86 -- i.e.
*protective*; Leiter 2009; Calzavara-Pinton 2015). The scheduling consequence is
the opposite of naive dose-minimisation: at a fixed total UV budget, many small
sub-erythemal sessions carry far less melanoma/BCC risk than a few large ones.
Frequent-and-small beats rare-and-large, and this now agrees with the vitamin D
objective rather than opposing it.

All three axes are literature-anchored as of 2026-08-09. The cumulative axis is
a power law, not linear -- incidence follows `dose ** Ab` with the biological
amplification factor `Ab` = 1.6-1.8 for SCC (Moan 1989, erythemally weighted),
so doubling cumulative dose costs 3.25x rather than 2x. The skin-type axis is no
longer a prior: it is the measured fraction of DNA damage reaching the
germinative basal layer, derived from Shih 2018 and Fajuyigbe 2018, and it
reproduces Gandini 2005's melanoma RR for skin type I vs IV (2.09 measured,
2.12 derived) without having been fitted to it. Recommended schedules did not
move; the index values did. See section 14c.

### What still needs better data

- **The skin-type penalty**: the real discrepancy is between two *lamp* studies
  (Young 2020's 1.3-1.4x against Farrar 2013's 2.5-3x at equal 25(OH)D gain),
  not between lamp and naturalistic exposure as previously recorded.
- **`D_max`** drives the headline result and has unconfirmed origin. The
  mechanistic alternative is `feedback_mode='biphasic'`; the mode actually in
  force on this run is **'{_p.feedback_mode}'** (read from `ModelParams()` at
  report time). Its inflection is an *output* of a breakpoint fitted on the
  vitamin D3 axis (Scholze 2026, Commun Med 6:105), not a regressor -- the Hill
  exponent cannot be fitted from that paper at all.
- **Fajuyigbe 2018 (FASEB J 32:3700)** -- full text now in Research_Papers/
  (previously miscited as abstract-only) -- and its basal:whole CPD protection
  ratio remains the single fitted number behind the whole skin-type risk axis.
  It covers only extreme phototypes (SPT I/II vs VI); FST III/IV/V still rest
  on Shih 2018's melanin percentages alone.
- **de Boer 2025 individual-level SC/MI trajectories** would break the
  `alpha_SC ~ SC_max` and `K_ATTEN ~ K_SC` ridges.
- **Ambient-to-personal UV dosimetry** for upright activity:
  `vitd_objectives.PERSONAL_DOSE_GEOMETRY_FACTOR` (**{PERSONAL_DOSE_GEOMETRY_FACTOR}**
  on this run) is a scalar prior, not elevation-dependent, and it directly
  scales every minutes recommendation in this report. This line previously
  stated a hardcoded 0.55, a value the code had already replaced because it
  sat above every measured standing-posture exposure ratio in the literature
  the parameter cites. It is interpolated from the code now so it cannot drift
  again.
""")

    # Encoding is explicit: the report contains non-ASCII; Windows defaults
    # to cp1252 and crashes on write.
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("".join(report))

    print(f"\nReport written to {REPORT_PATH}")
    print(f"Figures in figs/")
    print(f"Tables in tables/")


if __name__ == "__main__":
    main()
