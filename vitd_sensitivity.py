"""
vitd_sensitivity.py
===================
Which parameters actually matter, and which the data cannot constrain.

Motivation
----------
This project accumulated ~12 free parameters, most documented as "coarse prior"
or "single-source prior". Ablation had already shown that `K_ATTEN` -- the
parameter with the most documentation and the longest citation trail -- moves
scheduling results by ~0.3%, while `D_max`, whose origin is unconfirmed, sets
the headline result. That asymmetry is invisible without a systematic scan.

Three analyses:
  1. one_at_a_time_sensitivity  -- elasticity of outputs to each parameter
  2. identifiability_pairs      -- which parameter PAIRS trade off (ridges)
  3. uncertainty_propagation    -- Monte Carlo over documented prior ranges,
                                   turning point estimates into intervals
"""

from __future__ import annotations

import numpy as np

from vitd_model import (ModelParams, simulate, build_pulse_schedule,
                        steady_state_D, net_final_D_change_per_SED,
                        allow_retired_parameter_sweep, K_CONST_LOO_ENVELOPE)
from vitd_validation import young2021_A_anchor

# BUG FIX 2026-09-23 (SWEEP-RANGES workstream): A and K_const were
# recalibrated 2026-09-22 (K_const 2.096070737357932, A 4.359560972608079,
# vitd_model.py) but this module's ranges were never updated, so the
# one-at-a-time swing for both was being measured over a bracket that
# EXCLUDES the value actually shipped -- not "vary the parameter around its
# calibrated value" but "vary it around a stale one". Fixed by deriving both
# ranges from the same calibration evidence the shipped values themselves
# come from, instead of a second, independently-hardcoded literal.

# K_const range = the leave-one-out envelope over calibration source studies,
# imported from vitd_model.K_CONST_LOO_ENVELOPE (single source).

# A range = the Young 2021 source-matched cross-check bracket (Arimed B vs.
# UV6 E_source), read LIVE from vitd_validation.young2021_A_anchor() rather
# than pinned, because the call is cheap (closed-form arithmetic, no
# simulate()) and this way the range tracks the anchor automatically if
# K_const (which the anchor is computed against) changes again.
_A_CROSS_CHECKS = young2021_A_anchor()["cross_checks"]
A_YOUNG2021_BRACKET = (min(_A_CROSS_CHECKS.values()), max(_A_CROSS_CHECKS.values()))

# Documented plausible ranges. Sources are in the vitd_model.py docstring and
# ModelParams.SC_PRIOR_RANGES; ranges here are deliberately WIDE where the
# evidence is thin, so the propagated interval reflects real ignorance.
# SC_max, alpha_SC, K_SC, k_damage are read from ModelParams.SC_PRIOR_RANGES
# -- the single canonical source (CLAUDE.md) -- instead of being retyped
# here, so they cannot drift out of sync with it the way A/K_const did.
PARAM_RANGES = {
    # (low, high, evidence note)
    "A":          (*A_YOUNG2021_BRACKET,
                   "Young 2021 source-matched cross-check bracket (Arimed B "
                   ".. UV6 E_source), read live from "
                   "vitd_validation.young2021_A_anchor()['cross_checks']; "
                   "replaces a hardcoded (1.6, 3.8) that excluded the "
                   "shipped A = 4.359560972608079 (2026-09-23 bug fix)"),
    "K_const":    (*K_CONST_LOO_ENVELOPE,
                   "leave-one-out envelope over calibration source studies, "
                   "calibration_approval.yaml "
                   "K_const_leave_one_out_envelope_2026_09_22; replaces a "
                   "hardcoded (0.20, 0.90) that excluded the shipped "
                   "K_const = 2.096070737357932 (2026-09-23 bug fix)"),
    "K_ATTEN":    (0.0, 2.0, "Matsuoka 1990 abstract (in repo) implies 0.288972-0.838389 once its baseline-25(OH)D confound is deconvolved; 0 = no tanning penalty"),
    "K_SC":       (*ModelParams.SC_PRIOR_RANGES["K_SC"],
                   "ModelParams.SC_PRIOR_RANGES (single source); no direct "
                   "vitamin-D attenuation measurement exists"),
    "SC_max":     (*ModelParams.SC_PRIOR_RANGES["SC_max"],
                   "ModelParams.SC_PRIOR_RANGES (single source); de Boer 2025 "
                   "+23% (FP I-III) to +36% (FP IV-VI), ceiling widened to "
                   "0.54 2026-09-22"),
    "alpha_SC":   (*ModelParams.SC_PRIOR_RANGES["alpha_SC"],
                   "ModelParams.SC_PRIOR_RANGES (single source); fitted to "
                   "3 group-mean timepoints only"),
    "D_max":      (116.0, 250.0, "ORIGIN UNCONFIRMED; 116 = Bogh 2010 observed max"),
    "beta":       (np.log(2) / 120, np.log(2) / 21, "Ravnbak 2008 4-5 month fade"),
    "k_damage":   (*ModelParams.SC_PRIOR_RANGES["k_damage"],
                   "ModelParams.SC_PRIOR_RANGES (single source); NOT "
                   "identifiable from group medians per SC_PRIOR_RANGES "
                   "docstring, default 0.0 sits at this range's lower bound "
                   "-- added to PARAM_RANGES 2026-09-23, was previously "
                   "swept in scripts/parameter_optima.py only"),
}


def _default_scenario(params: ModelParams, fst: int = 5,
                      sed: float = 1.0, interval: int = 3,
                      horizon: int = 84, D0: float = 25.0):
    sched = build_pulse_schedule(horizon, sed, interval)
    return simulate(fst=fst, schedule=sched, duration_days=horizon,
                    params=params, D0=D0)


def _outputs(result, D0: float = 25.0) -> dict:
    return {
        "steady_state_D": steady_state_D(result),
        "final_D": float(result["D"][-1]),
        "net_per_SED": net_final_D_change_per_SED(result, D0),
    }


def one_at_a_time_sensitivity(fst: int = 5, verbose: bool = True) -> list[dict]:
    """Elasticity of each output to each parameter, over its documented range.

    Reports the RELATIVE output change across the full plausible range, which is
    the decision-relevant quantity: a parameter with a huge range but a flat
    response does not matter, and vice versa.
    """
    base = _outputs(_default_scenario(ModelParams(), fst=fst))
    rows = []
    for name, (lo, hi, note) in PARAM_RANGES.items():
        p_lo, p_hi = ModelParams(), ModelParams()
        # PARAM_RANGES may contain RETIRED parameters. Sweeping them here is
        # the point -- this function MEASURES how little they matter, which
        # would be the evidence any future retirement rests on. It is not a
        # calibration, and the named context manager keeps that distinction
        # visible at the call site. As of 2026-08-21, K_ATTEN is reinstated
        # and not retired (vitd_model.RETIRED_PARAMETERS is empty), so the
        # context manager below is currently a no-op guard for K_ATTEN.
        with allow_retired_parameter_sweep():
            setattr(p_lo, name, lo)
            setattr(p_hi, name, hi)
        o_lo = _outputs(_default_scenario(p_lo, fst=fst))
        o_hi = _outputs(_default_scenario(p_hi, fst=fst))
        row = {"parameter": name, "low": lo, "high": hi, "note": note}
        for key in base:
            b = base[key]
            spread = abs(o_hi[key] - o_lo[key])
            row[f"{key}_range"] = spread
            row[f"{key}_rel"] = (spread / abs(b)) if abs(b) > 1e-12 else float("nan")
        rows.append(row)

    rows.sort(key=lambda r: -(r["steady_state_D_rel"]))
    if verbose:
        print("=" * 74)
        print("ONE-AT-A-TIME SENSITIVITY (FST %d, 1 SED every 3 d, 84 d)" % fst)
        print("=" * 74)
        # G8: final_D and net_per_SED relative swings were computed but never
        # printed -- whether a parameter's influence agrees on the plateau
        # and on the terminal/efficiency measures is exactly the
        # metric-conflict question RESULTS.md section 5a raises separately.
        print(f"  {'parameter':10} {'range':>22} {'ss_D swing':>12} {'ss rel':>8} "
              f"{'finalD rel':>11} {'perSED rel':>11}")
        for r in rows:
            rng = f"{r['low']:.4g} - {r['high']:.4g}"
            print(f"  {r['parameter']:10} {rng:>22} "
                  f"{r['steady_state_D_range']:12.2f} {r['steady_state_D_rel']:7.1%} "
                  f"{r['final_D_rel']:10.1%} {r['net_per_SED_rel']:10.1%}")
        print("\n  Interpretation: parameters at the top drive results; those at the")
        print("  bottom cannot be blamed for prediction error no matter how")
        print("  uncertain their citation trail is.")
    return rows


def identifiability_pairs(fst: int = 5, n_grid: int = 9,
                          verbose: bool = True) -> dict:
    """Detect trade-off ridges between parameter pairs.

    If two parameters can compensate for each other, no amount of the current
    data will pin them down separately, and reporting either as a calibrated
    value is misleading. Measured as the correlation between the two
    parameters' contours of equal steady-state D.

    DEGENERATE vs UNIDENTIFIABLE (fixed 2026-09-20, reporting defect, not a
    calibration change -- see the DEGENERATE-verdict code below).
    -------------------------------------------------------------------------
    A ridge_fraction near 1.0 (the 2-D response surface collapses onto a
    single line) has TWO possible causes that this function used to conflate
    under one "UNIDENTIFIABLE" verdict:

      1. Genuine confounding: both parameters move steady-state D, but only
         some combination of them is constrained by the data (the informative
         case -- e.g. K_const~K_ATTEN ~0.9997, K_ATTEN~K_SC ~0.9986).
      2. Rank-1 degeneracy: ONE of the two parameters is inert (D_max, SC_max
         -- see vitd_model's D_max override being bit-identical across
         116/150/250/1e9, tests/test_vitd_model.py:121, and this module's own
         `one_at_a_time_sensitivity`, which reports steady_state_D_range =
         0.0 for both). A 2-D grid over one live and one dead axis is
         mechanically a perfect rank-1 surface (ridge_fraction = 1.0000
         exactly) regardless of any relationship between the two parameters
         -- there is no tradeoff, because the inert one never moves the
         output. The reported "ss_D spread" in this case is entirely the
         live parameter's own one-at-a-time swing, misattributed to a pair.

    Case 2 is not a milder version of case 1; it is a different claim
    ("this parameter cannot be estimated from D at all" vs "these two
    parameters can only be estimated jointly"), and it must not be printed
    under the same "UNIDENTIFIABLE" verdict a reader would use to decide which
    ridges reflect a genuine, data-driven tradeoff.

    Detection reuses `one_at_a_time_sensitivity`'s own steady_state_D_range
    for each parameter at this fst (not a fresh marginal sweep here): if
    either member of the pair has a range at/below INERT_TOL, the pair is
    labelled DEGENERATE and the inert parameter is named, independent of how
    close ridge_fraction is to 1. Sort order also changes: DEGENERATE pairs
    sort after every UNIDENTIFIABLE/separable pair (their ridge_fraction
    number is not a meaningful "how confounded" measure), so the genuinely
    informative ridges below 1.0000 stay visible at the top instead of being
    buried under two exact-1.0000 degeneracies.
    """
    # INERT_TOL: D_max/SC_max's one-at-a-time steady_state_D_range is exactly
    # 0.0 (bit-identical simulate() output across their documented range --
    # see docstring above). 1e-9 is a float-noise guard, not a calibrated
    # threshold: it is far below the smallest genuine effect this project has
    # measured (K_ATTEN's OAT range is order 0.1, three orders of magnitude
    # above this tolerance).
    INERT_TOL = 1e-9
    oat_rows = one_at_a_time_sensitivity(fst=fst, verbose=False)
    oat_range = {row["parameter"]: row["steady_state_D_range"] for row in oat_rows}

    pairs = [("K_ATTEN", "K_SC"), ("A", "D_max"), ("A", "K_const"),
             ("alpha_SC", "SC_max"), ("K_const", "K_ATTEN")]
    out = {}
    for a, b in pairs:
        alo, ahi, _ = PARAM_RANGES[a]
        blo, bhi, _ = PARAM_RANGES[b]
        av = np.linspace(alo, ahi, n_grid)
        bv = np.linspace(blo, bhi, n_grid)
        Z = np.zeros((n_grid, n_grid))
        for i, x in enumerate(av):
            for j, y in enumerate(bv):
                p = ModelParams()
                with allow_retired_parameter_sweep():   # see note above
                    setattr(p, a, float(x))
                    setattr(p, b, float(y))
                Z[i, j] = steady_state_D(_default_scenario(p, fst=fst))
        # Ridge strength: how much of the output variation is explained by a
        # single linear combination of the two parameters. A near-perfect
        # 1-D structure (first singular value dominating) means the pair is
        # jointly unidentifiable -- only their combination is determined --
        # UNLESS it is caused by one parameter being inert (see docstring).
        Zc = Z - Z.mean()
        sv = np.linalg.svd(Zc, compute_uv=False)
        ridge = float(sv[0] ** 2 / np.sum(sv ** 2))

        inert = [name for name in (a, b)
                 if oat_range.get(name, float("nan")) <= INERT_TOL]
        degenerate = len(inert) > 0
        if degenerate:
            verdict = ("DEGENERATE: %s inert (zero standalone sensitivity) -- "
                       "ridge is a rank-1 artefact of that inertness, not a "
                       "data-driven tradeoff" % " and ".join(inert))
        elif ridge > 0.99:
            verdict = "UNIDENTIFIABLE (only the combination is constrained)"
        else:
            verdict = "separable"

        out[f"{a}~{b}"] = {"ridge_fraction": ridge,
                           "output_spread": float(Z.max() - Z.min()),
                           "degenerate": degenerate,
                           "inert_parameters": inert,
                           "verdict": verdict}
    if verbose:
        print("\n" + "=" * 74)
        print("PAIRWISE IDENTIFIABILITY (steady-state D response surface)")
        print("=" * 74)
        print(f"  {'pair':24} {'1-D ridge':>11} {'ss_D spread':>12}  verdict")
        # Non-degenerate pairs first (genuinely informative -- sorted by
        # ridge_fraction descending within that group), degenerate pairs last
        # regardless of their ridge_fraction, since 1.0000 there is a rank-1
        # artefact, not a stronger tradeoff signal. See docstring.
        for k, v in sorted(out.items(),
                           key=lambda kv: (kv[1]["degenerate"], -kv[1]["ridge_fraction"])):
            print(f"  {k:24} {v['ridge_fraction']:10.4f} "
                  f"{v['output_spread']:12.2f}  {v['verdict']}")
        if any(v["degenerate"] for v in out.values()):
            print("\n  DEGENERATE pairs above involve a parameter with zero "
                  "one-at-a-time sensitivity (see one_at_a_time_sensitivity);")
            print("  their ridge_fraction = 1.0000 is mechanical, not evidence "
                  "of confounding, and their 'ss_D spread' is the OTHER")
            print("  parameter's own one-at-a-time swing, not a joint effect.")
    return out


def uncertainty_propagation(fst: int = 5, n_draws: int = 400,
                            seed: int = 0, verbose: bool = True) -> dict:
    """Monte Carlo over documented prior ranges -> intervals, not point values.

    Every headline number in this project has been reported as a single value
    derived from single-source priors. This converts them into ranges that
    reflect the actual state of the evidence.

    Uniform priors over the documented ranges: with this little data, a uniform
    prior is more honest than a peaked one, which would imply we know a central
    value we do not.
    """
    rng = np.random.default_rng(seed)
    keys = list(PARAM_RANGES)
    samples = {k: rng.uniform(PARAM_RANGES[k][0], PARAM_RANGES[k][1], n_draws)
               for k in keys}

    recs = {"steady_state_D": [], "net_per_SED": [], "best_interval": []}
    for i in range(n_draws):
        p = ModelParams()
        with allow_retired_parameter_sweep():           # see note above
            for k in keys:
                setattr(p, k, float(samples[k][i]))
        r = _default_scenario(p, fst=fst)
        o = _outputs(r)
        recs["steady_state_D"].append(o["steady_state_D"])
        recs["net_per_SED"].append(o["net_per_SED"])
        # Best interval under the honest metric, per draw
        best, best_v = None, -np.inf
        for iv in (1, 2, 3, 4, 5, 7, 10, 14, 21):
            rr = simulate(fst=fst, schedule=build_pulse_schedule(84, 1.0, iv),
                          duration_days=84, params=p, D0=25.0)
            v = net_final_D_change_per_SED(rr, 25.0)
            if v > best_v:
                best, best_v = iv, v
        recs["best_interval"].append(best)

    summary = {}
    for k, v in recs.items():
        arr = np.asarray(v, dtype=float)
        summary[k] = {
            "median": float(np.median(arr)),
            "p2.5": float(np.percentile(arr, 2.5)),
            "p97.5": float(np.percentile(arr, 97.5)),
        }
    intervals, counts = np.unique(np.asarray(recs["best_interval"]), return_counts=True)
    summary["best_interval_distribution"] = {
        int(i): int(c) for i, c in zip(intervals, counts)
    }

    if verbose:
        print("\n" + "=" * 74)
        print(f"UNCERTAINTY PROPAGATION ({n_draws} draws, uniform over prior ranges)")
        print("=" * 74)
        for k in ("steady_state_D", "net_per_SED"):
            s = summary[k]
            print(f"  {k:16} median {s['median']:8.3f}   "
                  f"95% range [{s['p2.5']:.3f}, {s['p97.5']:.3f}]")
        print("  best interval under net-gain-per-SED, frequency across draws:")
        for iv, c in sorted(summary["best_interval_distribution"].items()):
            bar = "#" * int(round(40 * c / n_draws))
            print(f"    {iv:3} d : {c:4} ({100*c/n_draws:4.1f}%) {bar}")
        print("\n  Report these as intervals. A single headline number implies")
        print("  precision the evidence does not support.")
    return summary


def run(verbose: bool = True, fsts: tuple[int, ...] = (5, 6)) -> dict:
    """Run all three analyses at each FST in `fsts`.

    G7: previously hardcoded to FST 5 only (every function's own default).
    FST 6 is this project's stated priority population and had no
    identifiability or uncertainty statement at all before this change.
    Note: `uncertainty_propagation` is 400 draws x 9 intervals x a full
    84-day `simulate` per draw, so adding a second FST roughly doubles this
    function's wall-clock cost -- the other two analyses are comparatively
    cheap.

    Returns ``{fst: {"sensitivity", "identifiability", "uncertainty"}}`` for
    each fst in `fsts`.
    """
    out = {}
    for fst in fsts:
        out[fst] = {
            "sensitivity": one_at_a_time_sensitivity(fst=fst, verbose=verbose),
            "identifiability": identifiability_pairs(fst=fst, verbose=verbose),
            "uncertainty": uncertainty_propagation(fst=fst, verbose=verbose),
        }
    return out


if __name__ == "__main__":
    run(verbose=True)
