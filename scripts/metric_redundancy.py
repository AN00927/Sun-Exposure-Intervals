"""
metric_redundancy.py
====================
Which scheduling objectives in this repo are actually distinct?

The codebase carries eight schedule-scoring functions. Several were added at
different times for different questions and have never been compared against
each other.

This script sweeps a schedule family, scores it under every metric, and reports:

  1. the Spearman rank-correlation matrix between metrics
  2. the argmax (optimal interval) each metric selects
  3. which metrics are rank-identical (|rho| > 0.999) and therefore redundant

Nothing is deleted here. This produces the evidence for a deletion decision.

Run:  PYTHONIOENCODING=utf-8 python3 scripts/metric_redundancy.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import (
    ModelParams, simulate, build_pulse_schedule,
    steady_state_D, net_final_D_change_per_SED,
    auc_above_baseline_per_SED, auc_above_sufficiency_per_SED,
    auc_deficit, final_D, _trapz,
)


def gross_synth_per_SED(result):
    """Local copy of a metric DELETED from vitd_model.py on 2026-07-31.

    Removed because it ranks schedules at Spearman rho = -0.999 against mean
    25(OH)D and -0.999 against time-integrated deficiency below 50 nmol/L,
    i.e. anti-correlated with the clinical outcome it claimed to optimise.
    """
    log = result["synthesis_log"]
    if not log: return 0.0
    total_d = sum(item[2] for item in log)
    total_s = sum(item[1] for item in log)
    return total_d / total_s if total_s > 0 else 0.0


def auc_D_per_SED(result):
    """Local copy of a metric DELETED from vitd_model.py on 2026-07-31.

    Removed because it ranks schedules at Spearman rho = -1.000 against
    final_D, i.e. that metric exactly inverted.
    """
    t, D = result["t"], result["D"]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0 or len(t) < 2:
        return 0.0
    return float(_trapz(D, t)) / total_sed

# Schedule family: one session every `interval` days, fixed dose per session,
# over a fixed horizon. Cumulative UV therefore varies across candidates.
INTERVALS = list(range(1, 61))
HORIZON_DAYS = 365
SED_PER_SESSION = 1.0
D0 = 25.0

# "higher is better" for all but the deficit metrics.
METRICS = {
    "gross_synth_per_SED":          (gross_synth_per_SED,                    "max"),
    "net_final_D_change_per_SED":   (net_final_D_change_per_SED,             "max"),
    "auc_D_per_SED":                (auc_D_per_SED,                          "max"),
    "auc_above_baseline_per_SED":   (auc_above_baseline_per_SED,             "max"),
    "auc_above_sufficiency_per_SED": (auc_above_sufficiency_per_SED,         "max"),
    "steady_state_D":               (steady_state_D,                         "max"),
    "final_D":                      (final_D,                                "max"),
    "mean_D":                       (lambda r: float(np.mean(r["D"])),       "max"),
    "auc_deficit_50":               (lambda r: auc_deficit(r["D"], r["t"], 50.0), "min"),
    "auc_deficit_25":               (lambda r: auc_deficit(r["D"], r["t"], 25.0), "min"),
}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho without a scipy dependency (ranks + Pearson)."""
    def rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        # average ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        if counts.max() > 1:
            sums = np.zeros(len(counts))
            np.add.at(sums, inv, r)
            r = (sums / counts)[inv]
        return r
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


def score_sweep(fst: int, params: ModelParams | None = None) -> dict:
    """Score every interval under every metric. Returns {metric: array}."""
    if params is None:
        params = ModelParams()
    scores = {name: [] for name in METRICS}
    for interval in INTERVALS:
        sched = build_pulse_schedule(HORIZON_DAYS, SED_PER_SESSION, interval)
        r = simulate(fst=fst, schedule=sched, duration_days=HORIZON_DAYS,
                     params=params, D0=D0,
                     dose_framing="fixed_per_session_SED")
        for name, (fn, _sense) in METRICS.items():
            scores[name].append(fn(r))
    return {k: np.asarray(v, dtype=float) for k, v in scores.items()}


def optimal_interval(name: str, vals: np.ndarray) -> tuple[int, float]:
    sense = METRICS[name][1]
    idx = int(np.argmax(vals)) if sense == "max" else int(np.argmin(vals))
    return INTERVALS[idx], float(vals[idx])


def main():
    for fst in (3, 5):
        print("=" * 78)
        print(f"METRIC REDUNDANCY  --  FST {fst}, {SED_PER_SESSION} SED/session, "
              f"{HORIZON_DAYS} d horizon, D0={D0}")
        print("=" * 78)

        scores = score_sweep(fst)
        names = list(METRICS)

        # ---- 1. Which interval does each metric call optimal? ---------------
        print("\n  Optimal interval selected by each metric")
        print(f"  {'metric':<32s} {'sense':>5s} {'best interval':>14s} {'value':>12s}")
        print(f"  {'-'*32} {'-'*5} {'-'*14} {'-'*12}")
        for name in names:
            iv, val = optimal_interval(name, scores[name])
            edge = ""
            if iv == INTERVALS[0] or iv == INTERVALS[-1]:
                edge = "  <-- AT SWEEP BOUND (no interior optimum)"
            print(f"  {name:<32s} {METRICS[name][1]:>5s} {iv:>11d} d "
                  f"{val:>12.3f}{edge}")

        # ---- 2. Rank-correlation matrix -------------------------------------
        # Flip 'min' metrics so a positive rho always means "same preference".
        oriented = {}
        for name in names:
            v = scores[name]
            oriented[name] = -v if METRICS[name][1] == "min" else v

        print("\n  Spearman rank correlation between metrics")
        print("  (oriented so +1 = ranks schedules identically, -1 = exactly opposite)")
        hdr = "".join(f"{i:>7d}" for i in range(len(names)))
        print(f"  {'':<32s}{hdr}")
        rho = np.eye(len(names))
        for i, a in enumerate(names):
            row = ""
            for j, b in enumerate(names):
                if i == j:
                    rho[i, j] = 1.0
                else:
                    rho[i, j] = _spearman(oriented[a], oriented[b])
                row += f"{rho[i, j]:>7.3f}"
            print(f"  {i:>2d} {a:<29s}{row}")

        # ---- 3. Redundant pairs ---------------------------------------------
        print("\n  Rank-identical pairs (|rho| > 0.999) -- one of each pair is "
              "redundant:")
        found = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(rho[i, j]) > 0.999:
                    sign = "same" if rho[i, j] > 0 else "EXACTLY INVERTED"
                    print(f"    {names[i]:<32s} ~ {names[j]:<32s} ({sign})")
                    found = True
        if not found:
            print("    (none)")

        # ---- 4. Monotone metrics = no interior optimum -----------------------
        print("\n  Monotone-in-interval metrics (no interior optimum -> cannot "
              "define an 'optimal interval'):")
        found = False
        for name in names:
            v = oriented[name]
            d = np.diff(v)
            if np.all(d >= -1e-12) or np.all(d <= 1e-12):
                direction = "increasing" if d[0] > 0 else "decreasing"
                print(f"    {name:<32s} monotonically {direction} in interval")
                found = True
        if not found:
            print("    (none)")
        print()


if __name__ == "__main__":
    main()
