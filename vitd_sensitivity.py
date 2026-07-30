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
                        gross_synth_per_SED)

# Documented plausible ranges. Sources are in the vitd_model.py docstring and
# ModelParams.SC_PRIOR_RANGES; ranges here are deliberately WIDE where the
# evidence is thin, so the propagated interval reflects real ignorance.
PARAM_RANGES = {
    # (low, high, evidence note)
    "A":          (1.6, 3.8, "Young 2021 fair-skin slopes 2.0-5.0 nmol/L/SED imply A up to ~3.8"),
    "K_const":    (0.20, 0.90, "Young 2020 FST II:VI 1.3-1.4; Webb/Kift naturalistic 2.5-3.0x implies higher"),
    "K_ATTEN":    (0.0, 2.0, "single-source prior (Matsuoka 1990, absent from repo); 0 = no tanning penalty"),
    "K_SC":       (0.5, 2.0, "no direct vitamin-D attenuation measurement exists"),
    "SC_max":     (0.20, 0.40, "de Boer 2025 +23% (FP I-III) to +36% (FP IV-VI)"),
    "alpha_SC":   (0.08, 0.20, "fitted to 3 group-mean timepoints only"),
    "D_max":      (116.0, 250.0, "ORIGIN UNCONFIRMED; 116 = Bogh 2010 observed max"),
    "beta":       (np.log(2) / 120, np.log(2) / 21, "Ravnbak 2008 4-5 month fade"),
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
        "gross_per_SED": gross_synth_per_SED(result),
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
        print(f"  {'parameter':10} {'range':>22} {'ss_D swing':>12} {'rel':>8}")
        for r in rows:
            rng = f"{r['low']:.4g} - {r['high']:.4g}"
            print(f"  {r['parameter']:10} {rng:>22} "
                  f"{r['steady_state_D_range']:12.2f} {r['steady_state_D_rel']:7.1%}")
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
    """
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
                setattr(p, a, float(x))
                setattr(p, b, float(y))
                Z[i, j] = steady_state_D(_default_scenario(p, fst=fst))
        # Ridge strength: how much of the output variation is explained by a
        # single linear combination of the two parameters. A near-perfect
        # 1-D structure (first singular value dominating) means the pair is
        # jointly unidentifiable -- only their combination is determined.
        Zc = Z - Z.mean()
        sv = np.linalg.svd(Zc, compute_uv=False)
        ridge = float(sv[0] ** 2 / np.sum(sv ** 2))
        out[f"{a}~{b}"] = {"ridge_fraction": ridge,
                           "output_spread": float(Z.max() - Z.min())}
    if verbose:
        print("\n" + "=" * 74)
        print("PAIRWISE IDENTIFIABILITY (steady-state D response surface)")
        print("=" * 74)
        print(f"  {'pair':24} {'1-D ridge':>11} {'ss_D spread':>12}  verdict")
        for k, v in sorted(out.items(), key=lambda kv: -kv[1]["ridge_fraction"]):
            verdict = ("UNIDENTIFIABLE (only the combination is constrained)"
                       if v["ridge_fraction"] > 0.99 else "separable")
            print(f"  {k:24} {v['ridge_fraction']:10.4f} "
                  f"{v['output_spread']:12.2f}  {verdict}")
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


def run(verbose: bool = True) -> dict:
    s = one_at_a_time_sensitivity(verbose=verbose)
    ident = identifiability_pairs(verbose=verbose)
    unc = uncertainty_propagation(verbose=verbose)
    return {"sensitivity": s, "identifiability": ident, "uncertainty": unc}


if __name__ == "__main__":
    run(verbose=True)
