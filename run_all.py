"""
run_all.py
==========
End-to-end run: validation + all priority analyses + summary report.

Usage:  python3 run_all.py
"""

import datetime as _dt
import os
import sys
from io import StringIO
from contextlib import redirect_stdout

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

import vitd_validation
import vitd_analyses
import vitd_calibration
import vitd_exposure
from vitd_optimization import (analysis_constant_sessions, analysis_variable_dose,
                               analysis_joint_schedule,
                               analysis_objective_hierarchy,
                               analysis_tri_objective)


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


# Fixed illustrative UV Index used by _report_location_exposure(). It is the
# SAME number at every location so the table isolates what changes with
# latitude and clothing (solar elevation, sky-visibility, exposed BSA) from
# what would change with real, location-specific cloud/ozone -- which is
# exactly what vitd_ambient's live Open-Meteo path is for, and exactly what
# this offline report does not attempt to reproduce. It is NOT a forecast or
# a measurement for any of the cities below; treat it as a dial, not data.
_LOCATION_UV_INDEX_FIXED = 8.0
# 1.0 SED/session matches the dose used in Q1 above (_report_decision_problems),
# so the "minutes" column here is directly comparable to that section.
_LOCATION_TARGET_SED = 1.0
_LOCATION_REF_MINUTES = 60.0
# Equinox local noon: solar declination ~ 0 deg on this date, so the solar
# elevation differences across the location table below reflect latitude
# alone, not a seasonal tilt that would happen to favour whichever
# hemisphere is in its own summer.
_LOCATION_WHEN_LOCAL = _dt.datetime(2026, 3, 20, 12, 0)
_LOCATION_PRESETS = ("face_and_hands", "shorts_and_tshirt")
# Dark-skin-priority location set (CLAUDE.md): FST V-VI populations are
# concentrated at low and equatorial latitudes, so most of this table is
# drawn from there instead of defaulting to the UK/Europe-heavy locations
# common in the underlying photobiology literature. One high-latitude city
# (Oslo) is included ONLY as contrast, not as the reference case. UTC
# offsets are each location's STANDARD time (no daylight saving applied,
# for reproducibility -- a real March 20 clock reading could differ by an
# hour where DST is in effect).
_LOCATIONS = (
    # name,             lat,      lon,      utc_offset_hours, why chosen
    ("Lagos, Nigeria",     6.5244,   3.3792,  1.0,
     "West Africa, FST VI population, low latitude"),
    ("Nairobi, Kenya",    -1.2921,  36.8219,  3.0,
     "equatorial, East Africa, FST V-VI population"),
    ("Mumbai, India",     19.0760,  72.8777,  5.5,
     "South Asia, FST IV-V population, low latitude"),
    ("Atlanta, USA",      33.7490, -84.3880, -5.0,
     "mid-latitude, sizeable FST V-VI population; contrasts with the "
     "low-latitude cities above at comparable UV input"),
    ("Oslo, Norway",      59.9139,  10.7522,  1.0,
     "high latitude -- CONTRAST case only, not the reference"),
)


def _report_location_exposure():
    """Exercise the PHASE 1 chain end to end: (latitude, longitude, date/time,
    clothing) -> solar geometry (vitd_solar) -> personal SED
    (vitd_exposure.exposure_sed). This is the non-network path -- see the
    printed caveats below for exactly what that trades away.
    """
    print("=" * 74)
    print("PHASE 1 CHAIN: location + date/time -> solar geometry -> personal SED")
    print("=" * 74)
    print(
        "vitd_exposure.exposure_sed_live() can fetch a live ambient UV Index\n"
        "for any of these locations from Open-Meteo. This report does NOT call\n"
        "it: RESULTS.md must reproduce byte-for-byte offline, and a network\n"
        "fetch cannot guarantee that. The UV Index below "
        f"({_LOCATION_UV_INDEX_FIXED:g}) is a FIXED INPUT the caller supplies\n"
        "(exposure_sed's non-live path), not a forecast or a measurement for\n"
        "any city named here -- it is deliberately the SAME number everywhere\n"
        "so the table isolates what changes with latitude and clothing from\n"
        "what would change with real, location-specific weather.\n"
    )
    print(
        "Sky-visibility (vis_percent) comes from Vernez et al. 2015 "
        "Supplementary Table S1\n"
        "(data/vernez2015_table_s1_vis_by_site_posture.csv), area-weighted "
        "over each\n"
        "preset's bare sites by vitd_exposure.BSA_SITE_FRACTION (Lund & "
        "Browder 1944).\n"
        "Vernez's exposure ratio assumes UNPROTECTED skin, NO shade, and "
        "CONTINUOUS\n"
        "exposure -- every SED figure below is that unshaded ceiling for the "
        "stated\n"
        "minutes, not a forecast of what a person doing normal activity "
        "would absorb.\n"
        "The UV Index itself is erythemally weighted on the CIE action "
        "spectrum, which\n"
        "was anchored to predominantly light-skin (FST I-III) sunburn data "
        "(McKinlay &\n"
        "Diffey 1987) -- that inheritance is not corrected anywhere in this "
        "chain.\n"
    )

    # Preset geometry (vis_percent, exposed_bsa) depends only on which sites
    # are bare and the posture, not on location -- shown once so the reader
    # sees what each preset resolves to before the per-location table mixes
    # it with solar geometry.
    print("  Preset -> resolved geometry (posture = standing_erect_arms_down,")
    print("            independent of location and date/time):")
    print(f"    {'preset':<20} {'exposed BSA':>12} {'vis_percent':>12}   description")
    _preset_geom = {}
    for preset in _LOCATION_PRESETS:
        g = vitd_exposure.geometry_for_preset(preset, "standing_erect_arms_down")
        _preset_geom[preset] = g
        desc = vitd_exposure.CLOTHING_PRESET_DESCRIPTIONS[preset]
        print(f"    {preset:<20} {g.exposed_bsa:>11.1%} {g.vis_percent:>11.1f}%   {desc}")
    print()

    rows = []
    for name, lat, lon, utc_off, why in _LOCATIONS:
        for preset in _LOCATION_PRESETS:
            r = vitd_exposure.exposure_sed(
                lat, lon, _LOCATION_WHEN_LOCAL, _LOCATION_REF_MINUTES,
                _LOCATION_UV_INDEX_FIXED, preset=preset,
                utc_offset_hours=utc_off)
            per_min_sed = (r.sed / _LOCATION_REF_MINUTES
                           if _LOCATION_REF_MINUTES > 0 else 0.0)
            minutes_needed = (_LOCATION_TARGET_SED / per_min_sed
                              if per_min_sed > 0 else float("inf"))
            rows.append({
                "location": name,
                "why_chosen": why,
                "latitude": lat,
                "longitude": lon,
                "date_time_local": _LOCATION_WHEN_LOCAL.isoformat(),
                "utc_offset_hours": utc_off,
                "preset": preset,
                "solar_elevation_deg": r.solar.elevation_deg,
                "solar_zenith_deg": r.solar.zenith_deg,
                "vis_percent": r.geometry.vis_percent,
                "exposed_bsa": r.geometry.exposed_bsa,
                "geometry_factor": r.geometry_factor,
                "ambient_uv_index_fixed_input": _LOCATION_UV_INDEX_FIXED,
                "erythemal_sed_at_ref_minutes": r.sed,
                "ref_minutes": _LOCATION_REF_MINUTES,
                "minutes_for_target_sed": minutes_needed,
                "target_sed": _LOCATION_TARGET_SED,
            })

    df = pd.DataFrame(rows)
    csv_path = "tables/location_exposure_chain.csv"
    df.to_csv(csv_path, index=False)

    print(f"  Erythemal SED chain by location and preset ({_LOCATION_WHEN_LOCAL.date()}, "
          "local noon, UVI="
          f"{_LOCATION_UV_INDEX_FIXED:g} fixed input):")
    print(f"    {'location':<18} {'preset':<20} {'lat':>6} {'sun elev':>9} "
          f"{'geom f':>7} {'SED@60m':>8} {'min/1 SED':>10}")
    for row in rows:
        print(f"    {row['location']:<18} {row['preset']:<20} "
              f"{row['latitude']:>6.1f} {row['solar_elevation_deg']:>8.1f}d "
              f"{row['geometry_factor']:>7.3f} "
              f"{row['erythemal_sed_at_ref_minutes']:>8.2f} "
              f"{row['minutes_for_target_sed']:>10.1f}")
    print()
    print("  'geom f' is the horizontal-to-personal geometry factor actually "
          "applied to")
    print("  minutes (Vernez vis_percent trimmed by instantaneous solar "
          "elevation -- see")
    print("  vitd_exposure module docstring caveat 5 on why instantaneous, "
          "not the daily-")
    print("  max SZA Vernez's own regression was fit to). 'SED@60m' and "
          "'min/1 SED' are")
    print(f"  ERYTHEMAL SED ({_LOCATION_TARGET_SED:g} SED = a standard "
          "session dose used above, e.g.")
    print("  Q1); no per-source E_source vitamin-D weighting has been "
          "applied here -- that")
    print("  conversion belongs to vitd_model.synthesis_per_session(), one "
          "layer downstream")
    print("  of what this module (PHASE 1) computes.")
    print(f"\n  Full table: {csv_path}")


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
        analysis_variable_dose,
        analysis_joint_schedule,
        # The objective hierarchy runs LAST but is REPORTED FIRST (section 0),
        # so that a reader meets the ranking before meeting any number.
        analysis_objective_hierarchy,
        analysis_tri_objective,
    ]
    alpha_log, k_log, beta_log, frame_log, fine_log, scipy_log, \
        maxint_log, load_log, split_log, sun_log, traj_log, const_log, \
        vardose_log, joint_log, hierarchy_log, tri_log = [
            capture_log(fn) for fn in analyses
        ]

    # 2b. Location-based exposure chain (PHASE 1: lat/lon/date/time -> SED),
    # offline/deterministic -- see _report_location_exposure() docstring.
    location_log = capture_log(_report_location_exposure)

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
    print(location_log)
    print(traj_log)
    print(const_log)
    print(vardose_log)
    print(joint_log)

    # 3. Markdown report.
    # Every constant quoted in the prose below is INTERPOLATED FROM THE CODE,
    # never typed out. This report previously stated a personal-dose geometry
    # factor, a K_ATTEN and an E_source that the code had all moved past, and
    # described a constant-sessions protocol the code no longer runs -- the
    # documentation drift CLAUDE.md warns about, in the one file that generates
    # the public report. Interpolation makes that class of bug impossible here.
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
    report = []
    report.append("# Vitamin D Scheduling Model — Results\n")
    report.append(
        "\n> **AUTO-GENERATED FILE — DO NOT HAND-EDIT.** Rewritten in full on "
        "every `run_all.py` run. Any manual edits here will be silently "
        "overwritten on the next run.\n"
    )
    report.append("\nThis report is generated by `run_all.py`. It captures the validation "
                  "suite output, parameter sweeps, and a brief interpretation.\n")

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
    report.append(f"\n**{n_pass} / {n_total} checks passed.**\n")
    report.append("\n```\n" + val_log + "\n```\n")

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
        "\nSweeps inter-session intervals in 1-day steps to find the exact "
        "optimal interval under each framing, complementing the coarse "
        "integer-interval analysis in the two-framing comparison. The interval "
        "range and skin types are set in "
        "`vitd_analyses.analysis_fine_grid` and appear in the log below; this "
        "line used to name a 1-35 day range the code does not sweep.\n"
    )
    report.append("\n```\n" + fine_log + "\n```\n")

    report.append("\n## 7. Continuous optimization (scipy.optimize)\n")
    report.append(
        "\nUses scipy.optimize.minimize_scalar (bounded Brent) to find the "
        "continuous-valued interval that maximizes net ΔD per SED and, "
        "separately, steady-state D, confirming the discrete sweep.\n"
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
        "how often, to maintain sufficiency? Uses spring/fall moderate "
        "conditions (UV index ~5) and arms + face exposure (~30% BSA). "
        f"`E_source` is the solar value keyed `{E_SOURCE_SOLAR_DEFAULT}` "
        f"(**{E_SOLAR_NOW:g}** on this run), not the neutral 1.0 this line "
        "used to claim — `analysis_sun_exposure` switched to the solar "
        "calibration on 2026-08-16 and the prose here did not follow.\n"
    )
    report.append("\n```\n" + sun_log + "\n```\n")
    report.append("\n![sun exposure](figs/sun_exposure_minutes_vs_D.png)\n")

    report.append("\n## 11b. Location-based UV exposure — the PHASE 1 chain "
                  "(lat/lon/date/time → SED)\n")
    report.append(
        "\nSection 11 above uses a single generic UV index (~5) and a fixed "
        "30% BSA for every schedule. This section instead runs the actual "
        "location → SED chain built for this purpose "
        "(`vitd_solar.py` for solar position, `vitd_ambient.py` for ambient "
        "UV Index, `vitd_exposure.py` joining both to the BSA/sky-visibility "
        "geometry) and reports what it produces for a small, named set of "
        "real places.\n\n"
        "**This report never calls the live path.** `vitd_exposure."
        "exposure_sed_live()` fetches ambient UV Index from Open-Meteo and "
        "exists and is tested (offline, against recorded fixtures — see "
        "`tests/test_exposure.py`); it is not used here because `RESULTS.md` "
        "has to reproduce identically on every run, and a network fetch "
        "cannot guarantee that. Every number below instead comes from "
        "`exposure_sed()`, the non-live path, called with a "
        f"**fixed, illustrative UV Index of {_LOCATION_UV_INDEX_FIXED:g}** "
        "(the same number at every location, deliberately — this isolates "
        "what latitude and clothing change from what real, "
        "location-specific cloud and ozone would change). That number is "
        "not a forecast or a measurement for any city named below.\n\n"
        "**Locations were chosen for this project's dark-skin-priority "
        "stance (CLAUDE.md), not for latitude spread alone.** FST V-VI "
        "populations are concentrated at low and equatorial latitudes, so "
        "three of the five cities (Lagos, Nairobi, Mumbai) sit there; "
        "Atlanta represents a mid-latitude location with a sizeable FST V-VI "
        "population; Oslo is included only as a high-latitude *contrast* "
        "case, not as the reference the way a UK-centric table would treat "
        "it. Date/time is fixed at local noon on the March equinox (solar "
        "declination ≈ 0°), so the elevation differences in the "
        "table are attributable to latitude alone rather than one "
        "hemisphere's summer. UTC offsets are each location's standard "
        "time, without daylight saving, for reproducibility.\n\n"
        "The 'minutes for target SED' column inverts the same linear "
        "`sed_from_minutes()` relationship section 14 (Q3) uses, at a "
        "1.0 SED target to match the per-session dose used there — so it is "
        "directly comparable to Q1/Q3 above, not a new dose convention. All "
        "SED values are **erythemal**, not vitamin-D-weighted: the "
        "per-study `E_source` multiplication happens one layer downstream, "
        "in `vitd_model.synthesis_per_session()`, and is deliberately not "
        "applied here (see `vitd_exposure.py` module docstring caveat 1).\n\n"
        "Sky-visibility (`vis_percent`) is Vernez et al. 2015 Supplementary "
        "Table S1 (`data/vernez2015_table_s1_vis_by_site_posture.csv`), "
        "area-weighted over each preset's bare sites by "
        "`vitd_exposure.BSA_SITE_FRACTION` (Lund & Browder 1944). It "
        "assumes unprotected skin, no shade, and continuous exposure — a "
        "ceiling, not a behavioural forecast — and the UV Index it "
        "multiplies is itself erythemally weighted on the CIE action "
        "spectrum, anchored to predominantly light-skin (FST I-III) "
        "sunburn data (McKinlay & Diffey 1987). Neither limitation is "
        "corrected anywhere in this chain; both are inherited as-is.\n"
    )
    report.append("\n```\n" + location_log + "\n```\n")
    report.append(
        "\nFull table (all locations × presets, every intermediate "
        "value): `tables/location_exposure_chain.csv`.\n"
    )

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
        "`seasonal` and `gap_coupled` returning the uniform optimum is the "
        "correct result, not a search failure: their grids contain the "
        "constant-dose special case (amplitude 0, exponent 0) and select it. "
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
        "| A3 | `combined_risk` | the skin-cancer index |\n\n"
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

    # ---- Decision problems, held-out benchmark, and uncertainty -------------
    # These are the sections to read first. The sweeps above used to optimise
    # the retired per-SED efficiency metric (removed from the code on
    # 2026-07-31; see vitd_model.py), which had no interior optimum -- they
    # now select by steady_state_D instead; everything below is posed as a
    # real decision.
    import vitd_objectives
    import vitd_benchmark_webbkift
    import vitd_sensitivity

    # The Q2 frontier above is printed as a text table. This turns it into one
    # actionable schedule per skin type and saves the frontier figure.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "scripts"))
    import risk_aware_scheduling

    obj_log = capture_log(_report_decision_problems)
    bench_log = capture_log(vitd_benchmark_webbkift.run)
    sens_log = capture_log(vitd_sensitivity.run)
    risk_log = capture_log(risk_aware_scheduling.main)
    risk_anchor_log = "\n".join(
        capture_log(fn) for fn in (
            vitd_calibration.elwood1997_intermittency_check,
            vitd_calibration.moan1989_scc_exponent_check,
            vitd_calibration.skin_type_susceptibility_check,
        ))
    print(obj_log)
    print(bench_log)
    print(sens_log)
    print(risk_log)

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

The interval ordering these sweeps produce comes from the D-feedback term, not
from photoadaptation: sparse dosing keeps mean D low, so each pulse meets less
inhibition. The induced-melanin coefficient `K_ATTEN` cannot be the cause --
its fitted value is {K_ATTEN_NOW:g} (see the `vitd_model.py` docstring), so on
this run that term is inert *by calibration*, not merely by ablation.

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

Three parameter pairs are jointly unidentifiable from the current data
(>99% of the response surface explained by a single linear combination):
`alpha_SC ~ SC_max`, `K_const ~ K_ATTEN`, and `K_ATTEN ~ K_SC`. Only their
combinations are constrained. Reporting either member of such a pair as a
separately calibrated value overstates what the data support -- melanin and
stratum corneum are both `exp(-k * saturating state)` terms driven by the same
UV input, so they trade off along a ridge.

### The one out-of-sample test currently fails

Webb/Kift 2018 report that skin type V needs 2.5-3.0x the dose of white
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
that lives in `vitd_behaviour.py` rather than inside `K_const`.

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
- **Fajuyigbe 2018 (FASEB J 32:3700)** is abstract-only and its basal:whole CPD
  protection ratio is the single fitted number behind the whole skin-type risk
  axis.
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

    # encoding is explicit: the report contains non-ASCII (arrows, mu, +/-) and
    # Windows would otherwise default to cp1252 and crash on write.
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("".join(report))

    print(f"\nReport written to {REPORT_PATH}")
    print(f"Figures in figs/")
    print(f"Tables in tables/")


if __name__ == "__main__":
    main()
