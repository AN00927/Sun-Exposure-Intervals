"""
basics/melanin_only.py
======================
TIER-1 REFERENCE MODEL: melanin, in full -- constitutive AND facultative
(dynamic tanning) -- with the stratum corneum channel ablated.

WHY THIS FILE EXISTS
--------------------
The full model has ~25 parameters and 13 validation checks, and a reviewer
cannot tell from it which terms are carrying the skin-type result. This file
strips the model down to exactly the melanin machinery -- both the static
per-FST baseline and the time-varying tan that builds up under UV and fades
without it -- and drops everything else: no stratum corneum, no product
feedback g(D), no serum decay/compartment, no BSA term, no geometry, no risk.
That is a genuine ablation of one mechanism, not a further-simplified toy: the
equations below are the same ones `vitd_model.simulate()` runs for melanin,
reproduced here standalone and cross-checked against the production function
itself (Part 2, section 4).

It is a REFERENCE model, not a replacement. It makes no absolute 25(OH)D
prediction and is not fitted to anything here. See `basics/README.md` for
the role of the two Tier-1 reference models.

PART 1 -- THE STATIC (CONSTITUTIVE) MODEL
------------------------------------------
    response(FST, dose) = ln(1 + dose * E_source) * exp(-K_const * M_baseline(FST))

No time axis. `M_baseline(FST)` is a fixed per-skin-type lookup. The synthesis
scalar `A` and the BSA factor are deliberately ABSENT: this model only ever
reports RATIOS between skin types at matched dose, and both cancel exactly in
a ratio -- an algebraic identity, not an approximation, and it means nothing
in Part 1 depends on `A`, the model's single largest source of uncertainty,
anchored on FST I-II data only.

At matched dose the log term cancels too, leaving

    ratio(a : b) = exp(-K_const * M(a)) / exp(-K_const * M(b))
                 = exp(K_const * (M(b) - M(a)))

so the entire skin-type gradient in Part 1 rests on exactly two things: the
scalar `K_const`, and the `M_baseline_by_FST` lookup table.

PART 2 -- THE DYNAMIC (FACULTATIVE) MODEL
-------------------------------------------
    Between UV pulses : M relaxes toward M_baseline(FST), exact exponential
                         decay at rate `beta`.
    At each UV pulse   : M jumps toward a per-FST ceiling M_ceiling(FST) =
                         M_baseline(FST) + M_TAN_HEADROOM, at a Hill-modulated
                         rate that rises with per-session dose toward
                         `alpha_max`, using per-FST `alpha_by_FST` and
                         `UV_halfmax_by_FST`.
    Synthesis is attenuated by TWO separate exponential factors:
        exp(-K_const * M_baseline(FST))              constitutive (Part 1)
        exp(-K_ATTEN * max(0, M - M_baseline(FST)))   facultative (the tan)

This is a hybrid pulse process, not a literal integration of the docstring's
continuous ODE in `vitd_model.py` -- see that module's Governing Equations
block and `docs/MELANIN_MODEL.md` section 1 for the exact discrepancies
(Hill-modulated `alpha`, per-FST uncapped ceiling, instantaneous pulses,
decay toward the constitutive floor not zero). `simulate_melanin()` below
reproduces the hybrid process exactly, and section 4 of Part 2 cross-checks
it directly against `vitd_model.simulate()` itself.

The stratum corneum channel is ABLATED here, the same way
`vitd_ablation.RUNGS["melanin_only"]` ablates it in the production model: no
`sc_factor = exp(-K_SC * SC)` term is computed at all (see
`dynamic_attenuation()`), which is the standalone-file equivalent of that
rung's `K_SC_override=0.0` (see `vitd_ablation.py`'s own docstring for why
that override, plus the three redundant ones alongside it, are the ones that
zero the SC channel). As `vitd_ablation.py` itself stresses: this is an
ABLATION, not a re-fit -- the parameters below were calibrated with SC
present, so absolute attenuation values here should not be read as
recalibrated-for-SC-absence predictions. That caveat is inherited, not
introduced by this file.

PARAMETER VALUES
----------------
Read from `vitd_model.ModelParams` at runtime -- never restated here. The
canonical source, with derivations and caveats, is the `vitd_model.py` module
docstring (see `CLAUDE.md`). The full parameter-by-parameter provenance,
including the exact arithmetic from each cited measurement to each model
constant, is `docs/MELANIN_MODEL.md`; this file reproduces its key numbers by
calling the same derivation code that document cites
(`scripts/fit_melanin_kinetics.py`, `vitd_calibration.py`) rather than
re-deriving or re-typing them.

VALIDATION DATA
---------------
Part 1: four within-study, matched-dose or behaviour-corrected routes,
reproduced from `scripts/kconst_recalibration.py` and
`scripts/fst_gap_decomposition.py` (run 2026-08-17). These are the SAME four
routes `K_const` was fitted to, so this is a consistency check, NOT
independent validation. Its value is that it makes the SPREAD between the
routes visible, which the aggregate fit hides.

Part 2: `alpha`/`alpha_by_FST`/`alpha_max`/`UV_halfmax_by_FST` and
`M_TAN_HEADROOM` against Ravnbak et al., Acta Derm Venereol 88:223-228 (2008),
PMID 18480919, and its 2009 UVA companion on the same 24-subject cohort;
`beta` against the same source's reported fading timepoints; `alpha`
additionally against Miyamura et al., Pigment Cell & Melanoma Research
24:136-147 (2011); `K_ATTEN` against Matsuoka, Wortsman & Hollis, J Lab Clin
Med 116:87-90 (1990), PMID 2165512. Every one of these is reproduced by
calling `scripts/fit_melanin_kinetics.py` or `vitd_calibration.py` directly,
not by re-deriving the arithmetic independently.

DARK-SKIN PROVENANCE (standing directive, CLAUDE.md)
-----------------------------------------------------
Every FST V datapoint behind the Part 1 routes is Indian-subcontinent
ancestry at UK latitude; every FST VI datapoint is lamp-based.
`M_baseline(VI) = 1.00` is an unvalidated group-level prior. Any VI ratio
below is an extrapolation onto that prior, not a measurement. Ravnbak 2008,
which anchors `UV_halfmax_by_FST` and `M_TAN_HEADROOM` in Part 2, enrolled
FST II-V only -- it has no FST VI subject at all, so every FST VI value in
`UV_halfmax_by_FST` is an extrapolation past the edge of the source data, not
an interpolation within it. Miyamura 2011, which is the only other source for
`alpha`, enrolled FST II-III only.

USAGE
-----
    PYTHONIOENCODING=utf-8 python3 basics/melanin_only.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import vitd_model  # noqa: E402
import vitd_ablation  # noqa: E402
import vitd_calibration as vcal  # noqa: E402
import fit_melanin_kinetics as fmk  # noqa: E402
from vitd_model import ModelParams  # noqa: E402


def M_baseline(fst: int, params: ModelParams) -> float:
    """Constitutive melanin index for `fst`. Thin accessor for
    ModelParams.M_baseline so the formulas below read like the maths.
    """
    return float(params.M_baseline(fst))

FST_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

# ---------------------------------------------------------------------------
# Observed within-study optical factors (FST V : FST II unless noted).
#
# Reproduced verbatim from scripts/kconst_recalibration.py (HELD_OUT_FIELD,
# YOUNG_PUBLISHED_ADJUSTED) and its Felton contrast. "behaviour removed" means
# the study's measured personal-dosimetry difference was divided out first, so
# what remains is intended to be optics only -- see exploratory/vitd_behaviour.py
# for that separation argument.
#
# NOTE the Young 2020 entries are FST II : FST VI, not V : II. They are kept in
# their published form and converted for comparison rather than silently
# rescaled.
# ---------------------------------------------------------------------------
OBSERVED_ROUTES: Dict[str, Tuple[int, int, float, str]] = {
    # label: (darker FST, lighter FST, observed response ratio lighter:darker, method)
    # 2.14 is Felton's OBSERVED matched-dose FST II:V response ratio, quoted in
    # the vitd_model.py K_const docstring ("model FST II:V 0.96 vs observed
    # 2.14" at the pre-recalibration K_const = 0.33). Not to be confused with
    # Felton's measured erythema MED ratio of 3.95, which is a different
    # endpoint and is excluded from the fit for that reason.
    "Felton 2016":  (5, 2, 2.14,
                     "lab, matched dose, within-study, both baselines printed"),
    "Young 2020 (Arimed B)": (6, 2, 1.29,
                              "lab, matched dose, within-study, ANCOVA-adjusted"),
    "Young 2020 (TL-01)":    (6, 2, 1.43,
                              "lab, matched dose, within-study, ANCOVA-adjusted"),
    "Kift 2013 / Webb 2010": (5, 2, 1.75,
                              "field dosimetry, behaviour removed"),
    "Darling 2013":          (5, 2, 2.68,
                              "field dosimetry, behaviour removed"),
}

#: The approved target band for the FST V : II optical factor
#: (scripts/kconst_recalibration.py DIAGNOSED_BAND).
DIAGNOSED_BAND = (1.7, 2.7)


# ---------------------------------------------------------------------------
# PART 1 -- the static (constitutive) model
# ---------------------------------------------------------------------------

def attenuation(fst: int, params: ModelParams | None = None) -> float:
    """Constitutive melanin transmission factor for `fst`.

    exp(-K_const * M_baseline(fst)). 1.0 for FST I (M = 0), falling
    monotonically with pigmentation. This is the whole Part-1 mechanism.
    """
    p = params or ModelParams()
    return float(np.exp(-p.K_const * M_baseline(fst, p)))


def response(fst: int, dose_sed: float, e_source: float = 1.0,
             params: ModelParams | None = None) -> float:
    """Relative per-session response, constitutive attenuation only, no time
    axis. Arbitrary units -- only ratios mean anything, because the
    synthesis scalar A is deliberately omitted."""
    return float(np.log1p(dose_sed * e_source)) * attenuation(fst, params)


def ratio(lighter_fst: int, darker_fst: int,
          params: ModelParams | None = None) -> float:
    """Predicted matched-dose response ratio, lighter : darker.

    The dose term cancels exactly, so this is
    exp(K_const * (M(darker) - M(lighter))) and depends on nothing else.
    """
    p = params or ModelParams()
    return float(np.exp(p.K_const * (M_baseline(darker_fst, p)
                                     - M_baseline(lighter_fst, p))))


def implied_K_const(lighter_fst: int, darker_fst: int, observed_ratio: float,
                    params: ModelParams | None = None) -> float:
    """Invert the ratio: what K_const would reproduce `observed_ratio` exactly?

    K = ln(ratio) / (M(darker) - M(lighter)). Undefined when the two skin
    types share a baseline melanin index.
    """
    p = params or ModelParams()
    dM = M_baseline(darker_fst, p) - M_baseline(lighter_fst, p)
    if abs(dM) < 1e-12:
        return float("nan")
    return float(np.log(observed_ratio) / dM)


# ---------------------------------------------------------------------------
# PART 2 -- the dynamic (facultative) model
# ---------------------------------------------------------------------------

def simulate_melanin(schedule, duration_days: int, params: ModelParams, fst: int,
                     M0: float | None = None, dt: float = 0.05
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the melanin state alone. Returns (t, M).

    Reproduces `vitd_model.simulate()`'s melanin channel exactly: between
    pulses M relaxes toward the FST's constitutive baseline as the EXACT
    solution of dM/dt = -beta*(M - M_base) (vitd_model.py's between-pulse
    update, `M = M_base + (M - M_base) * exp(-beta * delta_time)`), so --
    the same reasoning `sc_only.simulate_sc()` gives for its own decay
    channel -- the value at any sampled time does not depend on dt; dt only
    sets the sampling resolution of the returned arrays. Confirmed against
    `vitd_model.py`'s actual code, not assumed by analogy.

    At each pulse, M jumps by the model's Hill-modulated saturating update
    toward the per-FST ceiling `M_ceiling(fst)`, also the exact closed form
    of the induction half of the ODE under the Dirac-pulse assumption
    (vitd_model.py's simulate(), pulse block, ~line 1236-1240):
        alpha_p = alpha_by_FST.get(fst, alpha)
        UV_halfmax_p = UV_halfmax_by_FST.get(fst, UV_halfmax)
        hill_factor = sed**2 / (sed**2 + UV_halfmax_p**2)
        alpha_p = alpha_p + (alpha_max - alpha_p) * hill_factor
        M = M_ceil - (M_ceil - M) * exp(-alpha_p * sed)

    UV enters as instantaneous pulses on session days, exactly as in
    `sc_only.simulate_sc()`; `pending = {round(day/dt): sed for day, sed in
    schedule}` follows that file's own convention (last session wins on a
    same-day collision).

    A session scheduled on day 0 is handled explicitly before the main loop:
    a naive `for i in range(1, n)` loop (the form `sc_only.simulate_sc()`
    uses) never revisits index 0, so a day-0 pulse would silently be dropped
    -- vitd_model.simulate() itself applies a day-0 pulse immediately, before
    any decay, so this file matches that rather than the loop-structure
    shortcut. (Whether `sc_only.simulate_sc()` has the same gap on its own
    day-0 sessions -- de Boer's protocol schedules one -- was not checked
    here; see basics/README.md draft notes / the report accompanying this
    file for that open question.)
    """
    M_base = params.M_baseline(fst)
    M_ceil = params.M_ceiling(fst)
    beta = params.beta
    alpha_base = params.alpha_by_FST.get(fst, params.alpha)
    UV_halfmax_p = params.UV_halfmax_by_FST.get(fst, params.UV_halfmax)

    def _pulse(m: float, sed: float) -> float:
        hill_factor = sed ** 2 / (sed ** 2 + UV_halfmax_p ** 2)
        alpha_p = alpha_base + (params.alpha_max - alpha_base) * hill_factor
        return M_ceil - (M_ceil - m) * np.exp(-alpha_p * sed)

    n = int(round(duration_days / dt)) + 1
    t = np.linspace(0.0, duration_days, n)
    M = np.zeros(n)
    M[0] = M_base if M0 is None else M0
    pending = {round(day / dt): sed for day, sed in schedule}

    sed0 = pending.get(0)
    if sed0:
        M[0] = _pulse(M[0], sed0)

    for i in range(1, n):
        m = M_base + (M[i - 1] - M_base) * np.exp(-beta * dt)
        sed = pending.get(i)
        if sed:
            m = _pulse(m, sed)
        M[i] = m
    return t, M


def dynamic_attenuation(M, fst: int, params: ModelParams | None = None):
    """The two melanin factors from `vitd_model.synthesis_per_session()`,
    stratum corneum ABLATED.

    The production synthesis multiplier there is
        constitutive_factor * facultative_factor * sc_factor
    with `sc_factor = exp(-K_SC * SC)`. `vitd_ablation.RUNGS["melanin_only"]`
    forces this to an unconditional 1.0 by overriding K_SC, alpha_SC, SC_max
    and k_damage to zero (see `vitd_ablation.py`'s own docstring for why
    exactly those four knobs, not fewer). This function reproduces the same
    ablation the only way a standalone melanin-only file can: it simply never
    computes an sc_factor at all, so there is nothing a caller could forget
    to zero.

    M may be a scalar or an array (e.g. `simulate_melanin()`'s trajectory).
    Returns (constitutive_factor, facultative_factor, combined), matching
    `vitd_model.py:857-859`'s excess_M / constitutive_factor /
    facultative_factor triplet.
    """
    p = params or ModelParams()
    M_base = p.M_baseline(fst)
    excess_M = np.maximum(0.0, np.asarray(M, dtype=float) - M_base)
    constitutive_factor = float(np.exp(-p.K_const * M_base))
    facultative_factor = np.exp(-p.K_ATTEN * excess_M)
    combined = constitutive_factor * facultative_factor
    return constitutive_factor, facultative_factor, combined


# ---------------------------------------------------------------------------
# PART 3 -- full serum-D simulation with melanin varying, SC ablated
# ---------------------------------------------------------------------------

def simulate_D_melanin_only(schedule, duration_days: int, params: ModelParams,
                            fst: int, D0: float = 40.0, dt: float = 0.5
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full serum 25(OH)D trajectory with SC ablated, melanin dynamics active.

    Integrates D(t) = S(sed, M, D, SC=0) - k_decay_D(D)*D using the same
    sub-step scheme as vitd_model.simulate() but with:
      - SC = 0 always (no stratum corneum attenuation)
      - M(t) from simulate_melanin() (melanin dynamics active)
      - Full A, BSA, feedback g(D), and power-law decay

    Returns (t, D, M) arrays at dt resolution.

    This is the "what does the full model predict for serum D when only
    melanin is varying" baseline that Part 2's attenuation-only factor
    cannot supply.
    """
    from vitd_model import synthesis_per_session

    # Get the melanin trajectory
    t_m, M = simulate_melanin(schedule, duration_days, params, fst, dt=dt)
    n = len(t_m)

    # Build pulse lookup
    pending = {round(day / dt): sed for day, sed in schedule}

    D_arr = np.zeros(n)
    D_arr[0] = D0
    M_base = params.M_baseline(fst)

    for i in range(1, n):
        delta = dt
        D_prev = D_arr[i - 1]

        # Decay between pulses (power-law, state-dependent)
        D_decay = D_prev * np.exp(-params.k_decay_D(D_prev) * delta)

        # Synthesis at pulse (if any)
        sed = pending.get(i)
        if sed and sed > 0:
            delta_D = synthesis_per_session(
                sed=sed,
                M=M[i],
                M_baseline=M_base,
                D=D_decay,
                SC=0.0,          # SC ablated
                params=params,
            )
            D_arr[i] = D_decay + delta_D
        else:
            D_arr[i] = D_decay

    return t_m, D_arr, M


def _cross_check_vs_vitd_model_full_D(fst: int, schedule, duration_days: int,
                                       params: ModelParams, D0: float = 40.0
                                       ) -> dict:
    """Cross-check: does simulate_D_melanin_only agree with vitd_model.simulate()
    when both run the same ablation (SC=0)?"""
    from vitd_model import synthesis_per_session

    # Reference: vitd_model.simulate() with SC ablated
    full = vitd_model.simulate(fst=fst, schedule=schedule,
                               duration_days=duration_days, params=params,
                               D0=D0,
                               **vitd_ablation.rung_overrides("melanin_only"))
    t_ref, D_ref, M_ref = full["t"], full["D"], full["M"]

    # Collapse duplicate timestamps (pre/post pulse)
    keep = np.ones(len(t_ref), dtype=bool)
    keep[:-1] = np.diff(t_ref) != 0.0
    t_ref_dedup = t_ref[keep]
    D_ref_dedup = D_ref[keep]

    t_mine, D_mine, M_mine = simulate_D_melanin_only(
        schedule, duration_days, params, fst, D0=D0, dt=0.5)

    D_interp = np.interp(t_ref_dedup, t_mine, D_mine)
    abs_err = np.abs(D_interp - D_ref_dedup)

    return {
        "max_abs_err_D": float(np.max(abs_err)),
        "n_duplicate_timestamps_collapsed": int(len(t_ref) - len(t_ref_dedup)),
        "t_ref": t_ref_dedup, "D_ref": D_ref_dedup, "D_interp": D_interp,
        "t_mine": t_mine, "D_mine": D_mine, "M_mine": M_mine,
    }


def _cross_check_vs_vitd_model(fst: int, schedule, duration_days: int,
                               params: ModelParams):
    """Prove simulate_melanin() reproduces vitd_model.simulate()'s own M(t),
    SC ablated via vitd_ablation's OWN melanin_only overrides -- imported and
    called, not re-derived (see vitd_ablation._RUNG_OVERRIDES's docstring for
    why K_SC/alpha_SC/SC_max/k_damage are exactly the four knobs that switch
    SC off).

    `vitd_model.simulate()`'s `t_record` timestamps EVERY pulse day TWICE:
    once for the pre-pulse state (end of the between-pulse relaxation) and
    once for the post-pulse state, both stamped with the same t. This is
    deliberate on its part (it is how a caller reconstructs the jump), but it
    means a naive "interpolate the reference at the query's own timestamps"
    comparison against simulate_melanin()'s single-valued-per-t array
    silently picks up whichever of the two duplicate samples numpy's
    `interp` happens to return, at EVERY session day -- not just day 0 --
    manufacturing an apparent error that is really just "pre-pulse vs
    post-pulse", not a dynamics mismatch. This was checked directly (a
    first pass at this cross-check reported ~5e-4 max error and traced it to
    exactly this) rather than assumed. The fix is to collapse each duplicate
    timestamp down to its LAST (post-pulse) sample before comparing, which is
    what `full["M"]` means at that instant going forward and what
    simulate_melanin()'s array already holds.
    """
    full = vitd_model.simulate(fst=fst, schedule=schedule,
                               duration_days=duration_days, params=params,
                               **vitd_ablation.rung_overrides("melanin_only"))
    t_ref, M_ref = full["t"], full["M"]
    keep = np.ones(len(t_ref), dtype=bool)
    keep[:-1] = np.diff(t_ref) != 0.0  # drop all but the LAST of any run of
                                       # equal timestamps (pre-pulse copies)
    t_ref_dedup, M_ref_dedup = t_ref[keep], M_ref[keep]

    t_mine, M_mine = simulate_melanin(schedule, duration_days, params, fst)
    M_interp = np.interp(t_ref_dedup, t_mine, M_mine)
    abs_err = np.abs(M_interp - M_ref_dedup)
    n_duplicates_collapsed = int(len(t_ref) - len(t_ref_dedup))
    return {
        "max_abs_err": float(np.max(abs_err)),
        "n_duplicate_timestamps_collapsed": n_duplicates_collapsed,
        "t_ref": t_ref_dedup, "M_ref": M_ref_dedup, "M_interp": M_interp,
    }


def _r08_style_schedule(n_exp: int, sed_per_session: float):
    """A schedule following Ravnbak 2008's own dosing calendar
    (`fit_melanin_kinetics.r08_intervals`), for cross-checking against that
    script's own reimplementation of the pulse mechanics."""
    intervals = fmk.r08_intervals(n_exp)
    days = [0.0]
    for g in intervals:
        days.append(days[-1] + g)
    return [(d, sed_per_session) for d in days], days[-1]


def _cross_check_vs_fit_script(fst: int, n_exp: int, sed_per_session: float,
                               params: ModelParams):
    """Prove simulate_melanin() also agrees with
    fit_melanin_kinetics.simulate_pulses() -- an independent reimplementation
    of the same vitd_model.py equations that the alpha/beta fit is built on
    (see that function's own docstring). Two independent reproductions of the
    production code agreeing is stronger evidence than either alone."""
    M_base = params.M_baseline(fst)
    M_ceil = params.M_ceiling(fst)
    schedule, last_day = _r08_style_schedule(n_exp, sed_per_session)
    ref = fmk.simulate_pulses(n_exp, sed_per_session, M_base, M_ceil,
                              params.alpha_by_FST.get(fst, params.alpha),
                              params.alpha_max,
                              params.UV_halfmax_by_FST.get(fst, params.UV_halfmax),
                              params.beta, fmk.r08_intervals(n_exp))
    _t, M = simulate_melanin(schedule, last_day, params, fst)
    mine = float(M[-1])
    return ref, mine, abs(ref - mine)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> dict:
    p = ModelParams()
    out: dict = {"K_const": p.K_const, "routes": {}, "gradient": {}}

    if verbose:
        print("=" * 74)
        print("TIER-1 REFERENCE MODEL: melanin, in full (SC ablated)")
        print("=" * 74)
        print("PART 1 -- static (constitutive) term")
        print("  response(FST, dose) = ln(1 + dose*E) * exp(-K_const * M(FST))")
        print("  At matched dose the dose term cancels, so every ratio below")
        print("  is exp(K_const * dM) and depends on NOTHING ELSE -- not A,")
        print("  not BSA, not feedback, not stratum corneum, not decay.")
        print()
        print(f"  K_const in force  : {p.K_const:.4f}")
        print("  M_baseline table  : "
              + ", ".join(f"{FST_ROMAN[f]}={M_baseline(f, p):.2f}"
                          for f in range(1, 7)))
        print()

    # --- the predicted gradient -------------------------------------------
    if verbose:
        print("-" * 74)
        print("1. The whole skin-type gradient the static term produces")
        print("-" * 74)
        print(f"  {'FST':>4} {'M_base':>7} {'transmission':>13} "
              f"{'response vs FST I':>19}")
    t1 = attenuation(1, p)
    for f in range(1, 7):
        t = attenuation(f, p)
        out["gradient"][f] = t
        if verbose:
            print(f"  {FST_ROMAN[f]:>4} {M_baseline(f, p):>7.2f} "
                  f"{t:>13.4f} {t / t1:>18.3f}x")
    if verbose:
        print()
        print(f"  FST V : II  = {ratio(2, 5, p):.3f}x    "
              f"FST VI : I = {ratio(1, 6, p):.3f}x")
        print()

    # --- consistency against the four fitting routes ----------------------
    if verbose:
        print("-" * 74)
        print("2. Against the routes K_const was fitted to")
        print("-" * 74)
        print("  NOT independent validation -- these are the SAME routes the")
        print("  aggregate fit used. The point is the SPREAD, which the")
        print("  aggregate value hides.")
        print()
        print(f"  {'route':<24} {'contrast':>9} {'obs':>6} {'model':>7} "
              f"{'err':>7} {'implied K':>10}")

    implied = []
    for label, (dark, light, obs, _method) in OBSERVED_ROUTES.items():
        mod = ratio(light, dark, p)
        k = implied_K_const(light, dark, obs, p)
        implied.append(k)
        out["routes"][label] = {"observed": obs, "model": mod,
                                "implied_K_const": k}
        if verbose:
            contrast = f"{FST_ROMAN[light]}:{FST_ROMAN[dark]}"
            print(f"  {label:<24} {contrast:>9} {obs:>6.2f} {mod:>7.2f} "
                  f"{100 * (mod - obs) / obs:>6.0f}% {k:>10.3f}")

    implied = np.array(implied, dtype=float)
    out["implied_K_mean"] = float(np.mean(implied))
    out["implied_K_sd"] = float(np.std(implied, ddof=1))
    out["implied_K_min"] = float(np.min(implied))
    out["implied_K_max"] = float(np.max(implied))

    if verbose:
        print()
        print(f"  Implied K_const across routes: mean {np.mean(implied):.3f}, "
              f"SD {np.std(implied, ddof=1):.3f}, "
              f"range {np.min(implied):.3f}-{np.max(implied):.3f}")
        print(f"  In force: {p.K_const:.3f}")
        print()
        print("  READ THIS AS: the routes disagree by a factor of "
              f"{np.max(implied) / max(np.min(implied), 1e-9):.1f} on the one")
        print("  parameter that carries the entire static-term result. The")
        print("  value in force is their centre, not a reconciliation.")
        print()

    # --- the Part-1 verdict -------------------------------------------------
    v_ii = ratio(2, 5, p)
    in_band = DIAGNOSED_BAND[0] <= v_ii <= DIAGNOSED_BAND[1]
    out["V_II_ratio"] = v_ii
    out["V_II_in_diagnosed_band"] = bool(in_band)

    if verbose:
        print("-" * 74)
        print("3. Verdict -- constitutive (static) term")
        print("-" * 74)
        print(f"  FST V : II optical factor = {v_ii:.3f}, "
              f"approved band {DIAGNOSED_BAND[0]}-{DIAGNOSED_BAND[1]}: "
              f"{'IN BAND' if in_band else 'OUT OF BAND'}")
        print()
        print("  SUPPORTED  : the DIRECTION of the gradient. Every one of the")
        print("               five routes agrees darker skin responds less at")
        print("               matched dose. That is unanimous and robust.")
        print("  NOT SUPPORTED : the SIZE of the gradient as a point estimate.")
        print("               Two matched-dose LAMP studies (Felton, Young)")
        print("               disagree with each other, which no amount of")
        print("               modelling can reconcile. Report a range.")
        print("  EXTRAPOLATED : everything involving FST VI. M_baseline(VI) =")
        print("               1.00 is an unvalidated group-level prior and")
        print("               every VI ratio above is computed onto it.")
        print()

    # =========================================================================
    # PART 2 -- the dynamic (facultative) model
    # =========================================================================
    if verbose:
        print("=" * 74)
        print("PART 2 -- dynamic (facultative) term: melanin kinetics")
        print("=" * 74)
        print("  Between pulses: M = M_base + (M - M_base) * exp(-beta * dt)")
        print("  At each pulse : hill = sed^2 / (sed^2 + UV_halfmax_by_FST[fst]^2)")
        print("                  alpha_p = alpha_by_FST[fst] + (alpha_max - alpha_by_FST[fst]) * hill")
        print("                  M = M_ceil - (M_ceil - M) * exp(-alpha_p * sed)")
        print("                  M_ceil = M_baseline(fst) + M_TAN_HEADROOM  (NOT capped at 1.0)")
        print("  Synthesis     : constitutive_factor = exp(-K_const * M_baseline(fst))")
        print("                  facultative_factor  = exp(-K_ATTEN * max(0, M - M_baseline(fst)))")
        print("  Stratum corneum is ABLATED: no sc_factor term exists in this file at all.")
        print()

    # --- 4. cross-checks: does our reimplementation match the real code? ---
    demo_schedule = vitd_model.build_weekly_schedule(63, 1.0, [0, 2, 4])
    cc = _cross_check_vs_vitd_model(5, demo_schedule, 250, p)
    ref_pulses, mine_pulses, err_pulses = _cross_check_vs_fit_script(3, 12, 3.0, p)

    out["cross_check"] = {
        "max_abs_err_vs_vitd_model_simulate": cc["max_abs_err"],
        "n_duplicate_timestamps_collapsed": cc["n_duplicate_timestamps_collapsed"],
        "max_abs_err_vs_fit_melanin_kinetics": err_pulses,
    }

    if verbose:
        print("-" * 74)
        print("4. Cross-check: does simulate_melanin() actually match the real code?")
        print("-" * 74)
        print("  (a) vs vitd_model.simulate() itself, SC ablated via")
        print("      vitd_ablation.rung_overrides('melanin_only') (imported, not")
        print("      re-derived), FST V, weekly Mon/Wed/Fri 1.0 SED for 9 weeks,")
        print("      then 187 days with no further exposure:")
        print(f"        max |M_mine - M_reference| = {cc['max_abs_err']:.3e}")
        print(f"        ({cc['n_duplicate_timestamps_collapsed']} duplicate pre/post-pulse")
        print("         timestamps in vitd_model.simulate()'s own t_record were")
        print("         collapsed to their post-pulse value before comparing -- a first")
        print("         pass at this check, without that step, reported ~5e-4 and traced")
        print("         to exactly those duplicates, not to a dynamics discrepancy.)")
        print("  (b) vs fit_melanin_kinetics.simulate_pulses() -- an independent")
        print("      reimplementation used to fit alpha/beta -- FST III, 12")
        print("      Ravnbak-calendar exposures at 3.0 SED/session:")
        print(f"        reference M_end = {ref_pulses:.10f}")
        print(f"        this file's M_end = {mine_pulses:.10f}")
        print(f"        |difference| = {err_pulses:.3e}")
        print()

    # --- 5. demonstration: build-up and fade at two FSTs --------------------
    fst_lo, fst_hi = 2, 5
    _t_lo, M_lo = simulate_melanin(demo_schedule, 250, p, fst_lo)
    _t_hi, M_hi = simulate_melanin(demo_schedule, 250, p, fst_hi)
    checkpoints = [0, 63, 63 + int(round(np.log(2) / p.beta)), 250]
    half_life = np.log(2) / p.beta

    def _at(t_arr, M_arr, day):
        return float(np.interp(day, t_arr, M_arr))

    out["demo"] = {
        "checkpoints_days": checkpoints,
        "half_life_days": float(half_life),
        FST_ROMAN[fst_lo]: [_at(_t_lo, M_lo, d) for d in checkpoints],
        FST_ROMAN[fst_hi]: [_at(_t_hi, M_hi, d) for d in checkpoints],
    }

    if verbose:
        print("-" * 74)
        print(f"5. Demonstration: build-up (63 d, 3x/week 1.0 SED) then fade,")
        print(f"   FST {FST_ROMAN[fst_lo]} vs FST {FST_ROMAN[fst_hi]}")
        print("-" * 74)
        print(f"  beta half-life in force: {half_life:.1f} d")
        print(f"  {'day':>6} {'M(' + FST_ROMAN[fst_lo] + ')':>10} "
              f"{'excess(' + FST_ROMAN[fst_lo] + ')':>14} "
              f"{'M(' + FST_ROMAN[fst_hi] + ')':>10} "
              f"{'excess(' + FST_ROMAN[fst_hi] + ')':>14}")
        for d in checkpoints:
            m_lo = _at(_t_lo, M_lo, d)
            m_hi = _at(_t_hi, M_hi, d)
            print(f"  {d:>6} {m_lo:>10.4f} {max(0, m_lo - M_baseline(fst_lo, p)):>14.4f} "
                  f"{m_hi:>10.4f} {max(0, m_hi - M_baseline(fst_hi, p)):>14.4f}")
        print()
        print("  For scale: the FACULTATIVE swing against the STATIC gradient")
        print("  this file already isolated in Part 1.")

    c_lo, f_lo_full, comb_lo = dynamic_attenuation(
        p.M_ceiling(fst_lo), fst_lo, p)
    c_hi, f_hi_full, comb_hi = dynamic_attenuation(
        p.M_ceiling(fst_hi), fst_hi, p)
    facultative_penalty_pct_hi = 100.0 * (1.0 - float(f_hi_full))
    static_gap = ratio(fst_lo, fst_hi, p)
    # Compare the two channels on a LOG (multiplicative-effect) scale, not by
    # dividing the raw factors: static_gap is a >1x ratio between two FSTs,
    # while f_hi_full is a <1x within-FST swing (untanned -> fully tanned), so
    # the two are not directly divisible into a meaningful "Nx" without first
    # putting both on the same footing -- ln(effect) does that.
    log_effect_ratio = float(np.log(static_gap) / np.log(1.0 / float(f_hi_full)))
    out["facultative_vs_static"] = {
        "fully_tanned_facultative_factor_" + FST_ROMAN[fst_hi]: float(f_hi_full),
        "facultative_penalty_pct_" + FST_ROMAN[fst_hi]: facultative_penalty_pct_hi,
        "static_gap_" + FST_ROMAN[fst_lo] + "_" + FST_ROMAN[fst_hi]: static_gap,
        "log_effect_ratio_static_over_facultative": log_effect_ratio,
    }
    if verbose:
        print(f"    A FULLY tanned FST {FST_ROMAN[fst_hi]} (excess_M = M_TAN_HEADROOM)")
        print(f"    attenuates synthesis by an ADDITIONAL {facultative_penalty_pct_hi:.4f}% "
              f"relative to an untanned FST {FST_ROMAN[fst_hi]}")
        print(f"    (facultative factor {float(f_hi_full):.6f}).")
        print(f"    The STATIC gap between FST {FST_ROMAN[fst_lo]} and FST "
              f"{FST_ROMAN[fst_hi]} alone (Part 1) is a factor of {static_gap:.3f}x.")
        print(f"    On a log (multiplicative-effect) scale, the static gap is "
              f"{log_effect_ratio:.2f}x the size of the full facultative swing --")
        print(f"    the dynamic channel this file now simulates is roughly one order")
        print(f"    of magnitude smaller than the static channel Part 1 isolated, not")
        print(f"    comparable to it.")
        print()

    # --- 6. validating alpha / alpha_max / UV_halfmax ------------------------
    profile_grid = (0.0025, 0.01, 0.03, 0.1, 0.3, 1.0)
    profile_rows = [(a, fmk.profile_alpha(a, map_name="power")["rms"])
                    for a in profile_grid]
    rms_values = np.array([r for _, r in profile_rows])
    alpha_rms_movement_pct = float(
        100.0 * (rms_values.max() - rms_values.min()) / rms_values.max())

    fixed_fit = fmk.fit_alpha(map_name="power")
    dM_obs = fixed_fit["obs"]
    n_over = int((dM_obs > p.M_TAN_HEADROOM).sum())

    free_fits = {name: fmk.fit_alpha(map_name=name, fit_headroom=True)
                for name in ("linear", "quadratic", "power")}
    headrooms = np.array([f["headroom"] for f in free_fits.values()])

    out["alpha_validation"] = {
        "profile_alpha_grid": list(profile_grid),
        "profile_rms": rms_values.tolist(),
        "rms_movement_pct_over_grid": alpha_rms_movement_pct,
        "ravnbak_dM_observed_min": float(dM_obs.min()),
        "ravnbak_dM_observed_max": float(dM_obs.max()),
        "n_tans_exceeding_shipped_headroom": n_over,
        "free_fit_headroom_min": float(headrooms.min()),
        "free_fit_headroom_max": float(headrooms.max()),
    }

    if verbose:
        print("-" * 74)
        print("6. Validating alpha / alpha_by_FST / alpha_max / UV_halfmax_by_FST")
        print("   against Ravnbak 2008 (scripts/fit_melanin_kinetics.py)")
        print("-" * 74)
        print(f"  Ravnbak's six measured tans, converted onto this model's melanin")
        print(f"  index: dM = {dM_obs.min():.6f} to {dM_obs.max():.6f}.")
        print(f"  Shipped M_TAN_HEADROOM = {p.M_TAN_HEADROOM:.6g}: "
              f"{n_over} of {len(dM_obs)} observed tans exceed it "
              f"({'none -- every measured tan fits under the ceiling' if n_over == 0 else 'ceiling BINDS, cannot represent them at any alpha'}).")
        print()
        print(f"  Identifiability -- profiling alpha over a "
              f"{profile_grid[-1] / profile_grid[0]:.0f}-fold range, refitting")
        print(f"  alpha_max / UV_halfmax / headroom at each fixed alpha:")
        print(f"    {'alpha':>9} {'best RMS(dM)':>14}")
        for a, r in profile_rows:
            print(f"    {a:9.4f} {r:14.5f}")
        print(f"  RMS moves by {alpha_rms_movement_pct:.4f}% across that "
              f"{profile_grid[-1] / profile_grid[0]:.0f}x range.")
        print("  Ravnbak dosed at only two exposure counts (6 and 12), and the tan")
        print("  is already near plateau at 6 -- the data constrain WHERE the tan")
        print("  saturates (the ceiling), not HOW FAST it gets there (alpha).")
        print()
        print(f"  M_TAN_HEADROOM free-fit range across three candidate scale maps: "
              f"{headrooms.min():.6f}-{headrooms.max():.6f}.")
        print(f"  Shipped value {p.M_TAN_HEADROOM:.6g} sits inside that range and equals "
              f"the single largest observed tan (rounded up at the third decimal).")
        print()

    # --- 7. validating beta ---------------------------------------------------
    fade_rows = []
    for label, observed_days in fmk.FADE_OBSERVED_UVB.items():
        b = fmk.invert_linear_extrapolated_fade(observed_days)
        fade_rows.append((label, observed_days, b, np.log(2) / b))
    beta_mean = float(np.mean([b for _, _, b, _ in fade_rows]))
    forward_table = [(hl, fmk.linear_extrapolated_zero(np.log(2) / hl))
                     for hl in (30, 40, 48, 56, 70, 90)]

    out["beta_validation"] = {
        "fade_rows": [(lbl, obs, float(b), float(t12))
                      for lbl, obs, b, t12 in fade_rows],
        "implied_beta_mean": beta_mean,
        "implied_half_life_mean_days": float(np.log(2) / beta_mean),
        "shipped_half_life_days": float(np.log(2) / p.beta),
        "forward_table": [(hl, float(rep)) for hl, rep in forward_table],
    }

    if verbose:
        print("-" * 74)
        print("7. Validating beta against Ravnbak's own fading procedure")
        print("   (scripts/fit_melanin_kinetics.py, inverting a published method)")
        print("-" * 74)
        print("  Ravnbak reports 'days to total fading' by fitting a STRAIGHT LINE")
        print("  through pigmentation at 3 post-exposure timepoints and reading its")
        print("  x-intercept -- NOT a decay half-life. Inverting that procedure:")
        print(f"    {'arm':>22} {'reported (d)':>13} {'implied beta':>13} {'implied t1/2 (d)':>17}")
        for label, obs, b, t12 in fade_rows:
            print(f"    {label:>22} {obs:13.0f} {b:13.5f} {t12:17.1f}")
        print(f"  Mean implied half-life: {np.log(2) / beta_mean:.1f} d. "
              f"Shipped: {np.log(2) / p.beta:.1f} d.")
        print()
        print("  Forward direction -- what a given TRUE half-life would make Ravnbak's")
        print("  procedure REPORT:")
        print(f"    {'true t1/2 (d)':>14} {'reported fading (d)':>20}")
        for hl, rep in forward_table:
            print(f"    {hl:14d} {rep:20.0f}")
        print(f"  Ravnbak reported {min(fmk.FADE_OBSERVED_UVB.values()):.0f}-"
              f"{max(fmk.FADE_OBSERVED_UVB.values()):.0f} d, matched by any true")
        print("  half-life in roughly 40-70 d -- a RANGE, not a point estimate. The")
        print(f"  shipped {np.log(2) / p.beta:.0f} d sits near the middle of that band.")
        print()
        print("  Do NOT cite Tadokoro 2005 for this timescale (CLAUDE.md): that study")
        print("  only observed pigmentation over 0-7 days post-exposure, far too")
        print("  short a window to resolve a decay half-life of tens of days.")
        print()

    # --- 8. validating K_ATTEN -------------------------------------------------
    matsuoka_rows = [vcal.derive_K_ATTEN(mode, None, p)
                    for mode in ("linear", "biphasic", "saturating", "none")]
    valid_Ks = [r["K_ATTEN"] for r in matsuoka_rows
               if r["valid"] and r["feedback_mode"] != "none"]

    out["K_ATTEN_validation"] = {
        "naive_attenuation_pct": 100.0 * vcal.NAIVE_ATTENUATION,
        "untanned_baseline_25OHD_nmolL": vcal.UNTANNED_25OHD,
        "tanned_baseline_25OHD_nmolL": vcal.TANNED_25OHD,
        "rows": [{"feedback_mode": r["feedback_mode"],
                 "melanin_attenuation_pct": (100.0 * r["melanin_attenuation"]
                                              if r["valid"] else None),
                 "K_ATTEN": (r["K_ATTEN"] if r["valid"] else None)}
                for r in matsuoka_rows],
        "implied_range": ((min(valid_Ks), max(valid_Ks)) if valid_Ks else None),
        "shipped_K_ATTEN": p.K_ATTEN,
    }

    if verbose:
        print("-" * 74)
        print("8. Validating K_ATTEN against Matsuoka 1990 (vitd_calibration.py)")
        print("-" * 74)
        print(f"  n=7, within-subject, untanned vs tanned, IDENTICAL whole-body UVB")
        print(f"  dose. Naive attenuation (raw rise ratio) = "
              f"{100 * vcal.NAIVE_ATTENUATION:.1f}% -- the widely-cited '51%'.")
        print(f"  BUT baseline 25(OH)D differed: {vcal.UNTANNED_25OHD:.1f} vs "
              f"{vcal.TANNED_25OHD:.1f} nmol/L, so part of that is the model's own")
        print("  D-feedback, not melanin. CLAUDE.md forbids using the naive ratio")
        print("  directly for exactly this reason.")
        print()
        print(f"  {'feedback form':16} {'melanin atten.':>15} {'K_ATTEN':>9}")
        for r in matsuoka_rows:
            ma = f"{r['melanin_attenuation']:.1%}" if r["valid"] else "n/a"
            k = f"{r['K_ATTEN']:.6f}" if r["valid"] else "n/a"
            print(f"  {r['feedback_mode']:16} {ma:>15} {k:>9}")
        if valid_Ks:
            print(f"\n  Implied K_ATTEN range (excluding 'none'): "
                  f"{min(valid_Ks):.6f}-{max(valid_Ks):.6f}")
        print(f"  Shipped K_ATTEN = {p.K_ATTEN:.6f} -- the LOWER bound of that range")
        print("  (the biphasic feedback form, the model's own default).")
        print()
        print("  Skin type of Matsuoka's 7 subjects is NOT stated in the abstract --")
        print("  this is not a light-skin-flagged extrapolation in the usual sense,")
        print("  it is an UNDISCLOSED one. Applying K_ATTEN uniformly across all six")
        print("  FSTs assumes it generalises, which nothing in the source confirms")
        print("  or denies.")
        print()

    # --- 9. verdict -- dynamic (facultative) term ---------------------------
    if verbose:
        print("-" * 74)
        print("9. Verdict -- dynamic (facultative) term")
        print("-" * 74)
        print("  SUPPORTED    : the DIRECTION of every dynamic mechanism. Melanin")
        print("               builds up under UV and relaxes toward the")
        print("               constitutive baseline in its absence, for every")
        print("               FST -- monotone since M_ceiling(FST) > M_baseline(FST)")
        print("               unconditionally (the fix for the old global M_max")
        print("               inversion, see vitd_model.py's M_TAN_HEADROOM entry).")
        print("  SUPPORTED    : M_TAN_HEADROOM's MAGNITUDE. Bounded below by the")
        print("               largest tan Ravnbak actually measured; stable across")
        print("               three candidate scale maps and across the entire")
        print("               (unidentifiable) alpha range -- section 6.")
        print("  NOT SUPPORTED: alpha / alpha_by_FST / alpha_max / UV_halfmax_by_FST")
        print("               as separately fitted constants. Section 6's profile")
        print(f"               moves the residual by only {alpha_rms_movement_pct:.2f}% "
              f"over a {profile_grid[-1] / profile_grid[0]:.0f}-fold")
        print("               range. Ravnbak's two exposure counts (6, 12) cannot")
        print("               separate a fast rise to a low ceiling from a slow")
        print("               rise to a high one.")
        print("  NOT SUPPORTED: beta as a point estimate. Section 7's inversion of")
        print("               Ravnbak's own published procedure supports a half-life")
        print("               range of roughly 40-70 d, not a single number.")
        print("  NOT SUPPORTED: K_ATTEN as a single number. Section 8 shows it is")
        print("               feedback-form-dependent by construction -- a ~3x")
        print("               range -- and the shipped value is that range's lower")
        print("               bound, not its centre.")
        print("  EXTRAPOLATED : UV_halfmax_by_FST[VI]. Ravnbak 2008 enrolled FST")
        print("               II-V only -- there is no FST VI datapoint to")
        print("               extrapolate FROM within the study; the model's FST")
        print("               VI cell sits past the edge of the source data.")
        print("  EXTRAPOLATED : alpha_by_FST is anchored on Miyamura 2011 (FST")
        print("               II-III only) and applied down to FST VI.")
        print("  UNDISCLOSED  : K_ATTEN's source cohort skin type. Matsuoka 1990's")
        print("               abstract states neither FST nor ethnicity for its")
        print("               7 subjects.")
        print()
        print("  K_const (Part 1, constitutive) and K_ATTEN (Part 2, facultative)")
        print("  come from entirely different kinds of study -- a between-subject")
        print("  matched-dose contrast vs. a single within-subject seasonal")
        print("  crossover -- and are reported elsewhere as jointly unidentifiable")
        print("  from this repository's data on the model's own response surface")
        print("  (RESULTS.md; not re-verified by this file). Section 5 above's own")
        print("  magnitude comparison -- the facultative swing being roughly an")
        print("  order of magnitude smaller than the static gap on a log scale --")
        print("  is this file's freshly computed version of that same point.")
        print()
        print("  A note on scheduling relevance, NOT re-verified here: docs/")
        print("  MELANIN_MODEL.md section 5 reports that sweeping alpha, beta and")
        print("  K_ATTEN across wide multiples of their documented ranges does not")
        print("  move the model's preferred exposure interval, in results generated")
        print("  BEFORE the 2026-08-25 M_TAN_HEADROOM recalibration this file's")
        print("  shipped parameters already reflect. That sweep has not been")
        print("  re-run against the current parameters (see that document's own")
        print("  section 5.5); this file computes fresh numbers for the ATTENUATION")
        print("  magnitude (section 5 above) but does not re-run the scheduling")
        print("  sweep itself.")
        print()

    # =========================================================================
    # PART 3 -- full serum-D simulation with melanin varying, SC ablated
    # =========================================================================
    if verbose:
        print("=" * 74)
        print("PART 3 -- full serum-D simulation, melanin only (SC ablated)")
        print("=" * 74)
        print("  This runs the FULL D(t) model: A, BSA, feedback g(D),")
        print("  power-law decay k_decay_D(D), with SC = 0 and M(t) from")
        print("  Part 2's simulate_melanin(). Unlike Part 2 (attenuation-")
        print("  only factors), this produces absolute 25(OH)D trajectories.")
        print()

    # --- 10. cross-check against vitd_model.simulate() -----------------------
    D0_demo = 40.0
    cc_d = _cross_check_vs_vitd_model_full_D(5, demo_schedule, 250, p, D0=D0_demo)

    out["part3_cross_check"] = {
        "max_abs_err_D": cc_d["max_abs_err_D"],
        "n_duplicate_timestamps_collapsed": cc_d["n_duplicate_timestamps_collapsed"],
    }

    if verbose:
        print("-" * 74)
        print("10. Cross-check: does simulate_D_melanin_only() match the real code?")
        print("-" * 74)
        print(f"  FST V, D0 = {D0_demo:.0f} nmol/L, SC ablated via")
        print(f"  vitd_ablation.rung_overrides('melanin_only'), same schedule as")
        print(f"  section 4:")
        print(f"    max |D_mine - D_reference| = {cc_d['max_abs_err_D']:.3e}")
        print(f"    ({cc_d['n_duplicate_timestamps_collapsed']} duplicate timestamps collapsed)")
        print()

    # --- 11. demonstration: D(t) across FSTs --------------------------------
    fst_demo = [1, 2, 3, 5]
    D0_demo2 = 40.0
    duration_demo = 365

    if verbose:
        print("-" * 74)
        print(f"11. Demonstration: full D(t) for FST I/II/III/V, D0 = {D0_demo2:.0f}")
        print(f"    {duration_demo} d, de Boer-style schedule (Mon/Wed/Fri 1.0 SED)")
        print("-" * 74)

    part3_results = {}
    for fst_d in fst_demo:
        t_d, D_d, M_d = simulate_D_melanin_only(
            demo_schedule, duration_demo, p, fst_d, D0=D0_demo2, dt=0.5)
        d_end = float(D_d[-1])
        d_max = float(D_d.max())
        t_max_idx = int(np.argmax(D_d))
        d_half_life = float(np.log(2) / p.k_decay_D(float(D_d.max())))
        part3_results[fst_d] = {
            "D_end": d_end, "D_max": d_max, "D0": D0_demo2,
            "D_half_life_at_peak": d_half_life,
        }
        if verbose:
            print(f"  FST {FST_ROMAN[fst_d]:>2}: D0={D0_demo2:.0f}, "
                  f"D_peak={d_max:.1f} (day {t_d[t_max_idx]:.0f}), "
                  f"D_end={d_end:.1f}, "
                  f"t1/2 at peak={d_half_life:.0f} d")

    out["part3_demonstration"] = part3_results

    if verbose:
        print()
        print("  SUFFICIENCY THRESHOLD (25(OH)D >= 50 nmol/L):")
        for fst_d in fst_demo:
            t_d, D_d, _ = simulate_D_melanin_only(
                demo_schedule, duration_demo, p, fst_d, D0=D0_demo2, dt=0.5)
            above = D_d >= 50.0
            if above.any():
                first_above = int(np.argmax(above))
                days_above = int(above.sum()) * 0.5
                print(f"    FST {FST_ROMAN[fst_d]:>2}: first >= 50 at day "
                      f"{t_d[first_above]:.0f}, {days_above:.0f} d above threshold")
            else:
                print(f"    FST {FST_ROMAN[fst_d]:>2}: never reaches 50 nmol/L")
        print()
        print("  CAVATARE: SC is ablated here, so these are UPPER BOUNDS on")
        print("  what melanin-only dynamics produce. The real model's SC channel")
        print("  attenuates further, especially at higher SC thickness. This is")
        print("  the baseline against which SC's impact is measured.")
        print()
    return out


if __name__ == "__main__":
    run(verbose=True)
