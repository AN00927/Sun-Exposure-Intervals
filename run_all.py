"""
run_all.py
==========
End-to-end run: validation + all priority analyses + summary report.

Usage:  python3 run_all.py
"""

import os
from io import StringIO
from contextlib import redirect_stdout

import vitd_validation
import vitd_analyses
from vitd_optimization import analysis_constant_sessions


REPORT_PATH = "RESULTS.md"


def capture_log(fn, *args, **kwargs):
    buf = StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


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
            # Show the spread across spacings so it is visible whether the
            # objective can actually discriminate, or whether every spacing
            # ties (which happens when the target is unreachable at this dose).
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


def main():
    print("=" * 70)
    print("Vitamin D scheduling model: full pipeline")
    print("=" * 70)

    # 1. Validation
    buf = StringIO()
    with redirect_stdout(buf):
        val_results = vitd_validation.run_all()
    val_log = buf.getvalue()

    # 1b. NHANES data preprocessing (if available)
    nhanes_available = False
    try:
        import nhanes_preprocessing
        data_dir = os.path.join(os.path.dirname(__file__), "Papers_and_Datasets")
        output_dir = os.path.join(os.path.dirname(__file__), "audit_outputs")
        nhanes_preprocessing.preprocess_nhanes_data(data_dir, output_dir)
        nhanes_available = True
    except Exception:
        pass

    # 1c. NHANES seasonal validation
    from vitd_validation import validate_nhanes_seasonal
    buf_nhanes = StringIO()
    with redirect_stdout(buf_nhanes):
        nhanes_results = validate_nhanes_seasonal(nhanes_available=nhanes_available)
    val_log += buf_nhanes.getvalue()
    val_results.update(nhanes_results)

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
    ]
    alpha_log, k_log, beta_log, frame_log, fine_log, scipy_log, \
        maxint_log, load_log, split_log, sun_log, traj_log, const_log = [
            capture_log(fn) for fn in analyses
        ]

    # Echo to console
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

    # 3. Markdown report
    n_pass = sum(val_results.values())
    n_total = len(val_results)
    report = []
    report.append("# Vitamin D Scheduling Model — Results\n")
    report.append("\nThis report is generated by `run_all.py`. It captures the validation "
                  "suite output, parameter sweeps, and a brief interpretation.\n")

    report.append("\n## 1. Validation\n")
    report.append(f"\n**{n_pass} / {n_total} checks passed.**\n")
    report.append("\n```\n" + val_log + "\n```\n")

    report.append("\n## 2. Alpha sensitivity sweep (Priority 1 completed in Session 2)\n")
    report.append(
        "\nVaries the photoadaptation rise rate (alpha) and checks whether the "
        "optimal inter-session interval shifts. Alpha was calibrated from light-skin "
        "Miyamura 2011 data and the handoff flagged that it may be larger in FST V-VI.\n"
    )
    report.append("\n```\n" + alpha_log + "\n```\n")
    report.append("\n![alpha sweep, efficiency](figs/alpha_sweep_efficiency.png)\n")
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
        "\nVaries the melanin attenuation coefficient (K_ATTEN). The default value "
        "is now 0.37 (paper-backed from Young 2020 FST II:VI ratio). The sweep "
        "explores sensitivity around this value.\n"
    )
    report.append("\n```\n" + k_log + "\n```\n")
    report.append("\n![K_ATTEN sweep](figs/K_ATTEN_sweep.png)\n")

    report.append("\n## 5. Two-framing schedule comparison\n")
    report.append(
        "\nCentral handoff finding: the optimal interval depends on what's being "
        "optimized.\n\n"
        "- **Framing 1**: matched cumulative SED (42 SED over 12 weeks). UV budget "
        "is the constraint; we just choose how to distribute it.\n"
        "- **Framing 2**: matched per-session dose (1 SED per session). Per-session "
        "dose is fixed; cumulative UV exposure varies with frequency.\n"
    )
    report.append("\n```\n" + frame_log + "\n```\n")
    report.append("\n![two framings](figs/two_framings.png)\n")

    report.append("\n## 6. Fine-grid interval sweep\n")
    report.append(
        "\nSweeps intervals from 1–35 days in 1-day steps to find the exact optimal "
        "interval under each framing. Complements the coarse integer-interval analysis "
        "in the two-framing comparison.\n"
    )
    report.append("\n```\n" + fine_log + "\n```\n")

    report.append("\n## 7. Continuous optimization (scipy.optimize)\n")
    report.append(
        "\nUses scipy.optimize.minimize (Nelder-Mead) to find the continuous-valued "
        "interval that maximizes per-SED efficiency, confirming the discrete sweep.\n"
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
        "that budget across 1-14 sessions. After loading, all schedules switch "
        "to 1 SED every 7 days maintenance.\n"
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
        "how often, to maintain sufficiency? Uses spring/fall moderate conditions "
        "(UV index ~5), arms + face exposure (~30% BSA), E_source = 1.0.\n"
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
        "\n15 sessions over 84 days, varying spacing patterns ranked by AUC deficiency "
        "below 50 nmol/L + safety sidecar.\n"
    )
    report.append("\n```\n" + const_log + "\n```\n")

    # ---- Decision problems, held-out benchmark, and uncertainty -------------
    # These are the sections to read first. The sweeps above optimise
    # gross_synth_per_SED, which has no interior optimum (see the warning at the
    # top of this report); everything below is posed as a real decision.
    import vitd_objectives
    import vitd_benchmark_webbkift
    import vitd_sensitivity

    obj_log = capture_log(_report_decision_problems)
    bench_log = capture_log(vitd_benchmark_webbkift.run)
    sens_log = capture_log(vitd_sensitivity.run)
    print(obj_log)
    print(bench_log)
    print(sens_log)

    report.append("\n## 14. Decision problems (recommended objectives)\n")
    report.append(
        "\nThe sweeps above maximise `gross_synth_per_SED`, which rises "
        "monotonically as dosing approaches zero and whose reported optima end "
        "*below* their starting 25(OH)D. These are the same questions posed so "
        "that cumulative UV is held fixed, or so that cancer risk is scored "
        "alongside the vitamin D outcome.\n"
    )
    report.append("\n```\n" + obj_log + "\n```\n")

    report.append("\n## 15. Held-out external benchmark (Webb/Kift 2018)\n")
    report.append(
        "\nNothing in this repository is calibrated to this study. Every other "
        "check in the validation suite is a calibration or debug target, so this "
        "is the only out-of-sample evidence available.\n"
    )
    report.append("\n```\n" + bench_log + "\n```\n")

    report.append("\n## 16. Sensitivity, identifiability, and uncertainty\n")
    report.append(
        "\nWhich parameters actually drive results, which pairs cannot be "
        "separated by the current data, and what the headline numbers look like "
        "as intervals rather than point estimates.\n"
    )
    report.append("\n```\n" + sens_log + "\n```\n")

    report.append("\n## 17. Interpretation\n")
    report.append("""
### Read the decision-problem sections first

Sections 2-13 sweep `gross_synth_per_SED`. That metric has **no interior
optimum**: it rises monotonically as dosing approaches zero, and its reported
"optima" (21-30 d, and ~59 d under continuous optimisation) end *below* the
starting 25(OH)D -- a net loss of vitamin D scored as maximal efficiency. The
apparent optimum drifts upward with the search bound, which is the signature of
a monotone objective rather than a peak.

Ablation shows why: setting `K_ATTEN = 0` changes the interval ranking by ~0.3%,
and setting `K_ATTEN = 0` *and* `K_SC = 0` leaves it intact. The ordering comes
from the `(1 - D/D_max)` feedback term, not from photoadaptation. Sparse dosing
keeps mean D low, so each pulse meets less inhibition.

Under `net_final_D_change_per_SED` the optimum is **2-4 days**, and Monte Carlo
over the documented prior ranges puts it at 2-4 days in 86% of draws
(section 16). That is the number to quote.

### What the sweeps do legitimately show

The interval ranking is robust to `alpha` (0.01-0.40), `beta` (21-120 d
half-life), and `K_ATTEN`. This is a real robustness result, but the reason is
not reassuring: those three parameters barely participate in the model at all.
At 1 SED/session the melanin state moves only +0.019 over 84 days. Section 16
quantifies this -- `A` swings steady-state D by 36% across its plausible range
while `K_ATTEN` swings it by 0.3%, despite `K_ATTEN` carrying the longer
citation trail.

### Identifiability

Three parameter pairs are jointly unidentifiable from the current data
(>99% of the response surface explained by a single linear combination):
`alpha_SC ~ SC_max`, `K_const ~ K_ATTEN`, and `K_ATTEN ~ K_SC`. Only their
combinations are constrained. Reporting either member of such a pair as a
separately calibrated value overstates what the data support -- melanin and
stratum corneum are both `exp(-k * saturating state)` terms driven by the same
UV input, so they trade off along a ridge.

### The one out-of-sample test currently fails

Webb/Kift 2018 report that skin type V needs 2.5-3.0x the dose of white
Caucasians for an equivalent 25(OH)D rise. The model gives 1.40x (section 15).
`K_const` is calibrated to Young 2020's FST II:VI ratio of 1.3-1.4, which is a
per-unit-dose ratio at *matched BSA under lamps*; Webb/Kift's 2.5-3.0x is a
whole-body naturalistic response. These measure different quantities, and the
model currently only reproduces the former. The required-minutes ratio (1.77 vs
2.78) passes a looser +/-40% band. This is the clearest evidence that the
skin-type penalty is under-modelled for real-world use.

### Skin cancer is now scored, not just flagged

The risk model separates the two epidemiologically distinct pathways: SCC tracks
**cumulative** dose, while melanoma and BCC track **intermittent** intensity
(Elwood 1997: intermittent OR 1.71, heavy occupational OR 0.86 -- i.e.
*protective*; Leiter 2009; Calzavara-Pinton 2015). The scheduling consequence is
the opposite of naive dose-minimisation: at a fixed total UV budget, many small
sub-erythemal sessions carry far less melanoma/BCC risk than a few large ones.
Frequent-and-small beats rare-and-large, and this now agrees with the vitamin D
objective rather than opposing it.

### What still needs better data

- **The skin-type penalty for whole-body naturalistic exposure** (the Webb/Kift
  gap above) is the highest-value calibration target.
- **`D_max`** drives the headline result and has unconfirmed origin. The
  mechanistic alternative (`feedback_mode='biphasic'`, inflection ~56 nmol/L
  from Comms Med 2025) is implemented but not yet the default.
- **Matsuoka 1990 (J Lab Clin Med, PMID 2165512)** is the sole source for
  `K_ATTEN` and is absent from the repository.
- **de Boer 2025 individual-level SC/MI trajectories** would break the
  `alpha_SC ~ SC_max` and `K_ATTEN ~ K_SC` ridges.
- **Ambient-to-personal UV dosimetry** for upright activity: the 0.55 geometry
  factor is a mid-range prior and directly scales every minutes recommendation.
""")

    # encoding is explicit: the report contains non-ASCII (arrows, mu, +/-) and
    # Windows would otherwise default to cp1252 and crash on write.
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("".join(report))

    print(f"\nReport written to {REPORT_PATH}")
    print(f"Figures in figs/")
    print(f"Tables in tables/")


if __name__ == "__main__":
    main()
