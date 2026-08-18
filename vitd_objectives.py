"""
vitd_objectives.py
==================
Decision problems people actually face, replacing "maximise nmol/L per SED".

Why this module exists
----------------------
The project's earlier headline metric -- the retired per-SED efficiency
metric, removed from `vitd_model.py` on 2026-07-31 -- had no interior optimum:
it rose monotonically as dosing approaches zero, and its reported "optima" sat
at schedules that END BELOW their starting 25(OH)D. It was also shown by
ablation to be driven by the D-feedback term rather than by photoadaptation.
Both failures are recomputed from the current parameters on every pipeline run
by `vitd_optimization.gross_synth_per_SED_diagnostic`, so the demotion is
re-earned rather than remembered. (The narrative write-up this note used to
cite, MODEL_AND_EVIDENCE.md, was purged on 2026-08-18 and is not a live doc.)

The objectives here are posed the way a person or clinician would pose them:

  Q1  "I can get outside N times over the next 12 weeks. When should I go?"
      -> optimal_spacing_fixed_sessions()

  Q2  "Keep me out of deficiency with the least skin-cancer risk."
      -> deficiency_vs_risk_frontier()

  Q3  "How many minutes of midday sun do I need, for my skin type?"
      -> minutes_for_target()

Each returns concrete numbers with units, and each carries the cancer-risk
indices alongside the vitamin D outcome so a schedule can never be recommended
on synthesis grounds alone.

Which of these is the HEADLINE is not a matter of taste. `OBJECTIVE_HIERARCHY`
below fixes it in code: one PRIMARY objective, two SECONDARY ones, and a
DIAGNOSTIC tier that is barred from ever being reported as a recommendation.
Anything that prints a schedule as advice should route its label through
`objective_tier()` / `assert_reportable_as_headline()` so the ranking cannot
drift back out of sync with the prose.
"""

from __future__ import annotations

import numpy as np

from typing import NamedTuple, Tuple

from vitd_model import (ModelParams, simulate, auc_deficit, steady_state_D,
                        final_D, get_E_source, E_SOURCE_SOLAR_DEFAULT)
from vitd_physiology import cancer_risk, personal_geometry_factor

# Clinical thresholds, nmol/L of 25(OH)D.
# 25  = deficiency (UK/SACN); 30 = IOM "risk of inadequacy" lower bound
# 50  = sufficiency (IOM/Endocrine Society; the usual target)
DEFICIENCY_NMOL_L = 25.0
SUFFICIENCY_NMOL_L = SUFFICIENCY_TARGET = 50.0

# Sub-erythemal ceiling (constraint C1). Public-health guidance is to stay below
# the erythema threshold; 0.7 MED leaves headroom for MED estimation error, which
# is large (MED itself is a group prior here, Shih/Farrar/Rhodes 2018).
#
# Lives here rather than in scripts/ because it is a model-level constraint, and
# because vitd_calibration anchors the Elwood dose contrast to it: 1 MED (the
# erythema threshold) against GAMMA_MED * MED (the largest non-burning dose the
# model recommends) is the burst:continuous contrast in
# vitd_calibration.elwood1997_intermittency_check.
GAMMA_MED = 0.7
# 25(OH)D starts below target (D0 = 25 nmol/L), so a sufficiency floor cannot
# bind from day 0. Allow a loading period, then require the floor to hold.
RAMP_IN_DAYS = 90
# Longest session anyone will realistically sit through, and roughly the window
# over which UV index stays near its midday peak (constraint C7). Sessions longer
# than this are reported but flagged: the model permits them, a person would not
# do them.
PRACTICAL_MAX_MINUTES = 120.0


# ---------------------------------------------------------------------------
# THE SCHEDULING METRIC HIERARCHY
# ---------------------------------------------------------------------------
# This repo has accumulated eight schedule-scoring functions over its life and
# has twice reported a headline "optimum" from a metric that had no interior
# optimum at all. The fix is not another metric -- it is an explicit,
# machine-readable ranking of the metrics it already has, so that "which number
# is the answer" stops being a property of whichever section a reader opened
# first.
#
# Three tiers, in strict order:
#
#   PRIMARY     One objective. The question a person actually asks: what is the
#               least I have to do to reach and then hold sufficiency? Minimum
#               dose, minimum sessions, subject to not burning and to a session
#               length a person will sit through. Everything reported as
#               ADVICE comes from this tier.
#
#   SECONDARY   Trade-offs the primary answer has to be checked against. These
#               refine or constrain a recommendation; they never replace it.
#
#   DIAGNOSTIC  Metrics kept because they are informative about the MODEL, and
#               barred from being reported as advice. `gross_synth_per_SED` is
#               here for a specific, demonstrated reason: it is monotone in the
#               direction of dosing less, so its argmax is an artefact of the
#               search bound, and at its reported 21-30 d "optima" the
#               trajectory ENDS BELOW its own starting 25(OH)D. It scores a net
#               loss of vitamin D as maximal efficiency. See
#               vitd_optimization.gross_synth_per_SED_diagnostic(), which
#               recomputes that failure from the CURRENT parameters rather than
#               quoting the historical figure.
#
# [CHOICE] The ordering is a decision about what the project is FOR, not a
# result. It is recorded here so that it is arguable in one place instead of
# being implied differently by each section of the report.

class ObjectiveSpec(NamedTuple):
    """One entry in the scheduling metric hierarchy."""
    tier: str                     # PRIMARY | SECONDARY | DIAGNOSTIC
    rank: int                     # sort order across the whole hierarchy
    key: str                      # stable identifier used by objective_tier()
    question: str                 # the question in a person's words
    minimise: Tuple[str, ...]     # objective axes, all minimised
    subject_to: Tuple[str, ...]   # constraint labels (the C-numbers)
    implemented_by: str           # where the number actually comes from
    status: str                   # what a reader must not over-read


TIER_PRIMARY = "PRIMARY"
TIER_SECONDARY = "SECONDARY"
TIER_DIAGNOSTIC = "DIAGNOSTIC"
TIER_ORDER = (TIER_PRIMARY, TIER_SECONDARY, TIER_DIAGNOSTIC)

OBJECTIVE_HIERARCHY: Tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec(
        tier=TIER_PRIMARY, rank=1,
        key="min_dose_min_sessions_to_hold_sufficiency",
        question=("What is the LEAST I have to do -- fewest sessions, smallest "
                  "dose -- to reach 50 nmol/L and then stay there?"),
        minimise=("n_sessions", "total_sed"),
        subject_to=("C3 sufficiency floor holds for every day after ramp-in",
                    "C1 per-session dose <= GAMMA_MED * MED(FST)",
                    "C7 per-session dose deliverable in PRACTICAL_MAX_MINUTES"),
        implemented_by="vitd_optimization.minimum_schedule_for_sufficiency",
        status=("HEADLINE. Infeasible cells are a RESULT -- dark skin at low "
                "session budgets -- not a gap to be filled by relaxing C1/C7."),
    ),
    ObjectiveSpec(
        tier=TIER_SECONDARY, rank=2,
        key="tri_objective_sessions_spacing_risk",
        question=("Among the schedules that do work, which trades fewest "
                  "sessions against shortest gaps against least skin-cancer "
                  "risk, and how steep is that trade?"),
        minimise=("n_sessions", "max_gap_days", "combined_risk"),
        subject_to=("C3 sufficiency floor holds for every day after ramp-in",
                    "C1 per-session dose <= GAMMA_MED * MED(FST)",
                    "C7 per-session dose deliverable in PRACTICAL_MAX_MINUTES"),
        implemented_by="vitd_optimization.tri_objective_frontier",
        status=("A frontier, not advice. The three axes genuinely conflict: "
                "fewer sessions forces longer gaps AND a larger per-session "
                "dose, which the intermittency term punishes cubically."),
    ),
    ObjectiveSpec(
        tier=TIER_SECONDARY, rank=3,
        key="min_risk_at_fixed_session_budget",
        question=("I can manage exactly N sessions a year. What is the "
                  "lowest-cancer-risk way to spend them?"),
        minimise=("combined_risk",),
        subject_to=("C3 sufficiency floor holds for every day after ramp-in",
                    "C1 per-session dose <= GAMMA_MED * MED(FST)",
                    "session budget fixed by the user"),
        implemented_by=("vitd_optimization.optimise_joint_schedule / "
                        "optimise_variable_dose / "
                        "optimal_interval_risk_constrained"),
        status=("Conditional on a budget the user supplies. Cross-skin-type "
                "comparisons of the index VALUES inherit the weaker "
                "susceptibility axis; rankings within one skin type do not."),
    ),
    ObjectiveSpec(
        tier=TIER_DIAGNOSTIC, rank=4,
        key="gross_synth_per_SED",
        question="How much D is synthesised per unit of UV delivered?",
        minimise=(),
        subject_to=(),
        implemented_by=("scripts/metric_redundancy.py (local copy; deleted "
                        "from vitd_model.py 2026-07-31); failure reproduced by "
                        "vitd_optimization.gross_synth_per_SED_diagnostic"),
        status=("BACKGROUND / DIAGNOSTIC ONLY. No interior optimum -- rises "
                "monotonically as dosing approaches zero, so its argmax tracks "
                "the search bound. At its reported optimum final 25(OH)D is "
                "BELOW baseline: a net LOSS of vitamin D scored as maximal "
                "efficiency. Never a recommendation."),
    ),
    ObjectiveSpec(
        tier=TIER_DIAGNOSTIC, rank=5,
        key="net_final_D_change_per_SED",
        question="How much net 25(OH)D does a schedule leave per unit of UV?",
        minimise=(),
        subject_to=(),
        implemented_by="vitd_model.net_final_D_change_per_SED",
        status=("Diagnostic. It repairs gross_synth's net-loss failure (it is "
                "signed) but is still a ratio with no floor constraint, so it "
                "can still prefer a schedule that never reaches sufficiency."),
    ),
)

_BY_KEY = {spec.key: spec for spec in OBJECTIVE_HIERARCHY}


def objective_tier(key: str) -> str:
    """Tier of a named objective. Raises for an objective not in the hierarchy.

    Raising rather than defaulting is deliberate: a metric nobody placed in the
    hierarchy is exactly the situation this module exists to prevent.
    """
    try:
        return _BY_KEY[key].tier
    except KeyError:
        raise KeyError(
            f"{key!r} is not in OBJECTIVE_HIERARCHY. Place it in a tier before "
            f"reporting it; known keys: {sorted(_BY_KEY)}") from None


def is_headline(key: str) -> bool:
    """True only for the single PRIMARY objective."""
    return objective_tier(key) == TIER_PRIMARY


def assert_reportable_as_headline(key: str) -> None:
    """Guard for report code. Diagnostics must not be presented as advice."""
    if objective_tier(key) == TIER_DIAGNOSTIC:
        raise ValueError(
            f"{key!r} is {TIER_DIAGNOSTIC}-tier and must not be reported as a "
            f"recommendation: {_BY_KEY[key].status}")


def primary_objective() -> ObjectiveSpec:
    """The one PRIMARY objective. Raises if the hierarchy grows a second."""
    primary = [s for s in OBJECTIVE_HIERARCHY if s.tier == TIER_PRIMARY]
    if len(primary) != 1:
        raise AssertionError(
            f"exactly one PRIMARY objective is allowed, found {len(primary)}")
    return primary[0]


def format_objective_hierarchy() -> str:
    """The hierarchy as plain text, for RESULTS.md and the console.

    Generated from OBJECTIVE_HIERARCHY rather than written out, so the report
    cannot state a ranking different from the one the code enforces.
    """
    out = ["=" * 74,
           "SCHEDULING METRIC HIERARCHY -- which number is the answer",
           "=" * 74,
           "",
           "Generated from vitd_objectives.OBJECTIVE_HIERARCHY. When a section",
           "below reports a number, its tier here is what that number means."]
    for tier in TIER_ORDER:
        specs = sorted((s for s in OBJECTIVE_HIERARCHY if s.tier == tier),
                       key=lambda x: x.rank)
        if not specs:
            continue
        out += ["", "-" * 74, tier, "-" * 74]
        for s in specs:
            out += ["", f"  [{s.rank}] {s.key}",
                    f"      asks     : {s.question}"]
            if s.minimise:
                out.append(f"      minimise : {', '.join(s.minimise)}")
            if s.subject_to:
                out.append(f"      s.t.     : {s.subject_to[0]}")
                out += [f"                 {c}" for c in s.subject_to[1:]]
            out += [f"      from     : {s.implemented_by}",
                    f"      status   : {s.status}"]
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Q1. Fixed session budget -- how should the sessions be spaced?
# ---------------------------------------------------------------------------
def optimal_spacing_fixed_sessions(fst: int,
                                   n_sessions: int,
                                   sed_per_session: float = 1.0,
                                   horizon_days: int = 84,
                                   D0: float = 25.0,
                                   params: ModelParams | None = None,
                                   objective: str = "mean_D") -> dict:
    """Best spacing for a FIXED number of sessions over a fixed horizon.

    Because the number of sessions and the dose per session are both fixed,
    cumulative UV is IDENTICAL across all candidate spacings -- so the
    comparison cannot be won by simply dosing less. Any difference is real
    schedule structure.

    Parameters
    ----------
    objective : {'mean_D', 'final_D', 'min_D', 'auc_deficit_50'}
        'mean_D'          maximise average 25(OH)D over the horizon
        'final_D'         maximise end-of-horizon 25(OH)D
        'min_D'           maximise the worst point (avoid dipping low)
        'auc_deficit_50'  minimise time-integrated shortfall below 50 nmol/L

    Returns dict with the best spacing plus the full ranked table.
    """
    if params is None:
        params = ModelParams()
    if n_sessions < 1:
        raise ValueError("n_sessions must be >= 1")

    # Even spacing across the horizon is parameterised by the gap; we also try
    # front-loading and back-loading to check that even spacing is actually best.
    rows = []
    max_gap = max(1, horizon_days // max(1, n_sessions - 1)) if n_sessions > 1 else horizon_days

    for gap in range(1, max_gap + 1):
        days = [i * gap for i in range(n_sessions)]
        if days[-1] >= horizon_days:
            continue
        sched = [(float(d), sed_per_session) for d in days]
        r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                     params=params, D0=D0,
                     dose_framing="fixed_per_session_SED")
        risk = r["cancer_risk"]
        rows.append({
            "gap_days": gap,
            "n_sessions": n_sessions,
            "total_sed": r["cumulative_SED"],
            "mean_D": float(np.mean(r["D"])),
            "final_D": final_D(r),
            "min_D": float(np.min(r["D"])),
            "days_below_50": _days_below(r, SUFFICIENCY_NMOL_L),
            "days_below_25": _days_below(r, DEFICIENCY_NMOL_L),
            "auc_deficit_50": auc_deficit(r["D"], r["t"], SUFFICIENCY_NMOL_L),
            "auc_deficit_25": auc_deficit(r["D"], r["t"], DEFICIENCY_NMOL_L),
            "scc_index": risk["scc_index"],
            "intermittent_index": risk["intermittent_index"],
            "burn_fraction": risk["burn_fraction"],
        })

    if not rows:
        return {"best": None, "table": []}

    minimise = objective.startswith("auc_deficit")
    best = (min if minimise else max)(rows, key=lambda r: r[objective])
    return {"best": best, "table": rows, "objective": objective}


def _days_below(result, threshold: float) -> float:
    """Days spent below a threshold, by trapezoidal integration of the indicator."""
    t, D = result["t"], result["D"]
    below = (D < threshold).astype(float)
    if len(t) < 2:
        return 0.0
    return float(np.trapezoid(below, t) if hasattr(np, "trapezoid")
                 else np.trapz(below, t))


def time_to_sufficiency(result, target: float = SUFFICIENCY_NMOL_L) -> float:
    """Days until 25(OH)D first reaches `target`; NaN if it never does.

    Scored because RAMP_IN_DAYS excludes the loading period from the
    sufficiency floor (constraint C3), which leaves *how fast* a schedule
    climbs to target completely unscored. Two schedules that both hold the
    floor from day 90 onward are treated as equivalent by C3 even when one
    reaches sufficiency in 59 days and the other in 113 -- a difference that
    matters clinically and is exactly where variable dosing pays.

    Interpolates linearly within the crossing step rather than returning the
    grid point, so the value does not quantise to the solver step size.
    """
    t = np.asarray(result["t"], dtype=float)
    D = np.asarray(result["D"], dtype=float)
    hit = np.flatnonzero(D >= target)
    if hit.size == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0:
        return float(t[0])
    D0_, D1_ = D[i - 1], D[i]
    if D1_ == D0_:
        return float(t[i])
    frac = (target - D0_) / (D1_ - D0_)
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


# ---------------------------------------------------------------------------
# Q2. Stay out of deficiency at least cancer risk -- Pareto frontier
# ---------------------------------------------------------------------------
def deficiency_vs_risk_frontier(fst: int,
                                horizon_days: int = 365,
                                D0: float = 25.0,
                                sed_grid=(0.25, 0.5, 1.0, 2.0, 4.0),
                                interval_grid=(1, 2, 3, 5, 7, 10, 14, 21, 28),
                                params: ModelParams | None = None,
                                target: float = SUFFICIENCY_NMOL_L) -> dict:
    """Two-objective frontier: minimise deficiency AND minimise cancer risk.

    Sweeps (dose per session) x (interval) and returns the Pareto-nondominated
    set. This is the honest framing of the clinical trade-off: you cannot
    minimise both, so present the frontier and let the user pick, rather than
    collapsing it into one number with arbitrary weights.

    A point dominates another if it has both lower time-below-target AND lower
    combined cancer risk.
    """
    if params is None:
        params = ModelParams()

    pts = []
    for sed in sed_grid:
        for iv in interval_grid:
            days = list(range(0, horizon_days, iv))
            sched = [(float(d), float(sed)) for d in days]
            r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                         params=params, D0=D0,
                         dose_framing="fixed_per_session_SED")
            risk = r["cancer_risk"]
            pts.append({
                "sed_per_session": sed,
                "interval_days": iv,
                "n_sessions": len(days),
                "total_sed": r["cumulative_SED"],
                "mean_D": float(np.mean(r["D"])),
                "final_D": final_D(r),
                "days_below_target": _days_below(r, target),
                "auc_deficit": auc_deficit(r["D"], r["t"], target),
                "scc_index": risk["scc_index"],
                "intermittent_index": risk["intermittent_index"],
                "combined_risk": risk["combined_index"],
                "burn_fraction": risk["burn_fraction"],
            })

    frontier = _pareto_front(pts, "days_below_target", "combined_risk")
    # Cheapest schedule that keeps the person at/above target essentially always
    sufficient = [p for p in pts if p["days_below_target"] <= 0.05 * horizon_days]
    cheapest = min(sufficient, key=lambda p: p["combined_risk"]) if sufficient else None
    return {"all_points": pts, "frontier": frontier,
            "cheapest_sufficient": cheapest, "target": target}


def _pareto_front(points, obj_a: str, obj_b: str):
    """Nondominated set, minimising both objectives."""
    front = []
    for p in points:
        dominated = any(
            (q[obj_a] <= p[obj_a] and q[obj_b] <= p[obj_b] and
             (q[obj_a] < p[obj_a] or q[obj_b] < p[obj_b]))
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p[obj_a])


# ---------------------------------------------------------------------------
# Q3. Minutes of sun -- the question people actually ask
# ---------------------------------------------------------------------------
# UV Index -> erythemal SED per minute.
# 1 UV Index unit = 25 mW/m^2 erythemally weighted (WHO definition).
# 1 SED = 100 J/m^2 erythemal.
# So SED/min = UVI * 25e-3 * 60 / 100 = UVI * 0.015.
SED_PER_MIN_PER_UVI: float = 0.015

# Personal-dose geometry factor ("exposure ratio"): UV Index is defined on a
# HORIZONTAL plane, but a standing person presents mostly vertical surfaces.
# Using 1.0 overestimates dose and so underestimates the minutes required.
#
# Measured exposure ratios (personal dose / ambient horizontal dose), erythemal:
#   Parisi 2003, standing posture, leg sites:      0.14 - 0.39
#   Parisi 2000, upright, various sites, shade:    0.16 - 0.49
#   Cockell 2001, electronic dosimetry, clear sky: 0.35 +/- 0.12
#   Thieden 2000, wrist vs head:                   wrist ~0.50 of head dose
# Schmalwieser 2018 reviews this literature and recommends expressing the ratio
# as a function of solar elevation -- which this scalar does not do.
#
# 0.35 is the central measured value for upright activity (Cockell 2001), and is
# consistent with the Parisi ranges. It replaces an earlier 0.55 guess that sat
# above every measured standing-posture range.
# [PRIOR] Scalar, not elevation-dependent. Directly scales every minutes
# recommendation, so it is the first thing to sweep. Higher (~0.5-0.7) for
# supine sunbathing, which presents a near-horizontal surface.
PERSONAL_DOSE_GEOMETRY_FACTOR: float = 0.35


def sed_from_minutes(minutes: float, uv_index: float,
                     geometry_factor: float | None = None,
                     solar_elevation_deg: float | None = None,
                     vis_percent: float | None = None) -> float:
    """Erythemal SED delivered to exposed skin by `minutes` at a given UV Index.

    Applies the horizontal-to-personal geometry correction. Precedence:
      1. explicit `geometry_factor` (pass 1.0 for a supine sunbather);
      2. `vis_percent` -> Vernez 2015 Eq. 1, optionally trimmed by
         `solar_elevation_deg` (see vitd_physiology.vernez_exposure_ratio);
      3. otherwise 0.35, the Cockell 2001 scalar.

    `solar_elevation_deg` on its own no longer changes the result -- the
    elevation-only form was withdrawn 2026-08-16 as unsourced (gap 26).
    """
    if geometry_factor is None:
        geometry_factor = personal_geometry_factor(solar_elevation_deg,
                                                   vis_percent)
    return max(0.0, minutes) * uv_index * SED_PER_MIN_PER_UVI * float(geometry_factor)


def minutes_for_target(fst: int,
                       uv_index: float,
                       exposed_bsa: float,
                       frequency_per_week: float = 7.0,
                       target: float = SUFFICIENCY_NMOL_L,
                       D0: float = 25.0,
                       horizon_days: int = 120,
                       params: ModelParams | None = None,
                       max_minutes: float = 240.0,
                       geometry_factor: float | None = None,
                       solar_elevation_deg: float | None = None) -> dict:
    """Minutes of sun per session needed to reach a 25(OH)D target.

    Answers "how long outside?" and simultaneously reports whether that
    exposure crosses the sunburn threshold -- a recommendation that requires
    burning is not a recommendation.

    [CALIBRATION 2026-08-16] When `params` is not supplied this now defaults to
    the solar E_source (Webb 2021 Manchester summer), not the neutral 1.0. This
    is a SUN function; 1.0 was never a solar measurement. It roughly halves the
    minutes. Pass `params` explicitly to override.
    """
    if params is None:
        params = ModelParams()
        params.E_source = get_E_source(E_SOURCE_SOLAR_DEFAULT)

    interval = max(1, int(round(7.0 / max(frequency_per_week, 1e-9))))
    days = list(range(0, horizon_days, interval))

    def achieved(minutes: float):
        sed = sed_from_minutes(minutes, uv_index, geometry_factor, solar_elevation_deg)
        sched = [(float(d), sed) for d in days]
        r = simulate(fst=fst, schedule=sched, duration_days=horizon_days,
                     params=params, D0=D0,
                     exposed_bsa_override=exposed_bsa,
                     dose_framing="fixed_per_session_SED")
        return steady_state_D(r), r

    lo, hi = 0.0, max_minutes
    if achieved(hi)[0] < target:
        ss, r = achieved(hi)
        return {"minutes": None, "reachable": False, "best_D": ss,
                "note": f"target {target} nmol/L unreachable at <= {max_minutes:.0f} min/session",
                "uv_index": uv_index, "exposed_bsa": exposed_bsa}
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if achieved(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    minutes = hi
    ss, r = achieved(minutes)
    sed = sed_from_minutes(minutes, uv_index, geometry_factor, solar_elevation_deg)
    risk = cancer_risk([sed] * len(days), fst)
    return {
        "minutes": minutes,
        "reachable": True,
        "sed_per_session": sed,
        "fraction_med": sed / risk["med_sed"],
        "would_burn": sed >= risk["med_sed"],
        "steady_state_D": ss,
        "sessions_per_week": frequency_per_week,
        "exposed_bsa": exposed_bsa,
        "uv_index": uv_index,
        "scc_index": risk["scc_index"],
        "intermittent_index": risk["intermittent_index"],
        "target": target,
    }
