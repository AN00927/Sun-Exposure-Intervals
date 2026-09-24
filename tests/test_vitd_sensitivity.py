"""
tests/test_vitd_sensitivity.py
================================
Coverage for vitd_sensitivity.py (previously zero tests), and the primary
regression test for the 2026-09-20 `identifiability_pairs` reporting fix
(see the "DEGENERATE vs UNIDENTIFIABLE" section
of `identifiability_pairs`'s own docstring in vitd_sensitivity.py).

Background this file protects against recurring: Wave 2 (W-F) established,
and this workstream independently reproduced, that `D_max` is INERT in the
shipped model (final D at FST VI is bit-identical across D_max = 116, 150,
250, 1e9 -- delta exactly 0.0; also pinned at tests/test_vitd_model.py:121)
and that `SC_max` is likewise inert. Before the fix, `identifiability_pairs`
reported both `A~D_max` and `alpha_SC~SC_max` at "ridge_fraction 1.0000,
UNIDENTIFIABLE" -- mechanically true (a 2-D grid over one live and one dead
axis IS a perfect rank-1 surface) but misleading, because it promoted two
inert parameters to the top of a list headed "UNIDENTIFIABLE" ahead of the
genuinely informative, data-driven ridges (K_const~K_ATTEN ~0.9997,
K_ATTEN~K_SC ~0.9986), which is exactly what RESULTS.md repeated to the
reader with no caveat (RESULTS.md:2897 at the time of the Wave 2 audit).

CHARACTERISATION vs VALIDATION
-------------------------------------------------------------------------
  VALIDATION        Expected value follows from the function's own
                    documented contract (sort order, dict-key structure,
                    percentile-bracketing-median, the fixed relationship
                    between a pair's `degenerate` flag and its members'
                    one-at-a-time sensitivity) -- independent of exactly
                    what number simulate() produces.

  CHARACTERISATION  Expected value is a NUMBER PINNED FROM RUNNING THE
                    CURRENT MODEL (a specific ridge_fraction, a specific
                    steady_state_D swing). Proves determinism, not
                    correctness. A failure after a deliberate model or
                    calibration change means the pin needs updating, not
                    that something regressed.

Runtime: `one_at_a_time_sensitivity` and `identifiability_pairs` are cheap
(tens to a few hundred simulate() calls, well under a second). The full
`run()` at default settings (400 Monte Carlo draws x 2 FSTs) is the one
expensive entry point in this module and is exercised here only once, with
`fsts=(5,)` to halve the cost (~5.5s observed) and `verbose=False`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import vitd_sensitivity as vs
from vitd_model import ModelParams


def _import_parameter_optima():
    """Import scripts/parameter_optima.py the way run_all.py does (defensive
    sys.path insert; scripts/ is not a package). Import-only -- this never
    calls its `main()`, so nothing under figs/ or tables/ is touched.
    """
    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import parameter_optima
    return parameter_optima


# ---------------------------------------------------------------------------
# PARAM_RANGES -- module-level documented prior ranges
# ---------------------------------------------------------------------------

def test_validation_param_ranges_low_below_high():
    for name, (lo, hi, note) in vs.PARAM_RANGES.items():
        assert lo < hi, f"{name}: low {lo} is not < high {hi}"
        assert isinstance(note, str) and len(note) > 0


def test_validation_shipped_defaults_lie_inside_sweep_range_vitd_sensitivity():
    """Regression test for the 2026-09-23 SWEEP-RANGES bug fix: A's and
    K_const's hardcoded ranges here ((1.6, 3.8) and (0.20, 0.90)) excluded
    the shipped, recalibrated defaults (A = 4.359560972608079,
    K_const = 2.096070737357932) after the 2026-09-22 parameter pass, so a
    "vary each parameter around its calibrated value" sweep was silently
    measuring a swing around a stale value instead. Every PARAM_RANGES entry
    must bracket (inclusive) the value ModelParams() actually ships with.
    """
    p = ModelParams()
    for name, (lo, hi, note) in vs.PARAM_RANGES.items():
        default = getattr(p, name)
        assert lo <= default <= hi, (
            f"{name}: shipped default {default!r} is outside sweep range "
            f"[{lo!r}, {hi!r}]")


def test_validation_shipped_defaults_lie_inside_sweep_range_parameter_optima():
    """Same regression, same cause, in scripts/parameter_optima.py's
    PARAMETERS list: K_const was (0.15, 0.50) and A was (1.5, 3.5), both
    excluding the shipped defaults after the 2026-09-22 parameter pass.
    """
    parameter_optima = _import_parameter_optima()
    p = ModelParams()
    for spec in parameter_optima.PARAMETERS:
        default = getattr(p, spec["attr"])
        assert spec["lo"] <= default <= spec["hi"], (
            f"{spec['name']}: shipped default {default!r} is outside sweep "
            f"range [{spec['lo']!r}, {spec['hi']!r}]")


# ---------------------------------------------------------------------------
# one_at_a_time_sensitivity
# ---------------------------------------------------------------------------

def test_validation_one_at_a_time_sensitivity_sorted_descending_by_ss_D_rel():
    rows = vs.one_at_a_time_sensitivity(fst=5, verbose=False)
    rels = [r["steady_state_D_rel"] for r in rows]
    assert rels == sorted(rels, reverse=True)
    assert {r["parameter"] for r in rows} == set(vs.PARAM_RANGES)


@pytest.mark.parametrize("fst", [5, 6])
def test_validation_D_max_and_SC_max_are_exactly_inert(fst):
    """VALIDATION, not characterisation: D_max/SC_max being exactly inert
    (steady_state_D_range == 0.0, not just small) is the ALREADY-ESTABLISHED
    fact this whole fix is built on (Wave 2 W-F, tests/test_vitd_model.py:121).
    A nonzero value here would mean either that finding no longer holds or
    that `identifiability_pairs`'s INERT_TOL-based detection (see below) is
    checking the wrong thing -- this test exists to catch exactly that
    silently invalidating the fix.
    """
    rows = vs.one_at_a_time_sensitivity(fst=fst, verbose=False)
    by_name = {r["parameter"]: r for r in rows}
    assert by_name["D_max"]["steady_state_D_range"] == 0.0
    assert by_name["D_max"]["final_D_range"] == 0.0
    assert by_name["SC_max"]["steady_state_D_range"] == 0.0
    assert by_name["SC_max"]["final_D_range"] == 0.0


def test_characterisation_one_at_a_time_top_parameter_pinned():
    """PINNED CURRENT VALUE (FST 5): the top-ranked parameter by relative
    steady-state-D swing and its numeric range. Not independently verified;
    only proves determinism.

    RE-PINNED 2026-09-23 for the SWEEP-RANGES bug fix (see the comment
    above `PARAM_RANGES` in vitd_sensitivity.py). The PREVIOUS pin of
    7.468854649469776 (itself a 2026-09-23 re-pin "for the 2026-09-22
    parameter pass") was the swing of a hardcoded K_const range
    (0.20, 0.90) that EXCLUDED the shipped K_const = 2.096070737357932 --
    a bug, not a calibration change: the sweep was not measuring "vary
    K_const around its calibrated value" at all. K_const's range is now the
    calibration leave-one-out envelope (1.5884400707455726,
    3.1839988650360818), which DOES bracket the shipped value, and the
    swing over that corrected range is 9.55394246902243 -> 8.975769062450752
    nmol/L. K_const stays the top-ranked parameter (the ordering this test
    exists to pin) both before and after the fix.
    """
    rows = vs.one_at_a_time_sensitivity(fst=5, verbose=False)
    top = rows[0]
    assert top["parameter"] == "K_const"
    assert top["steady_state_D_range"] == pytest.approx(8.975769062450752, abs=1e-9)


# ---------------------------------------------------------------------------
# identifiability_pairs -- the fix under test
# ---------------------------------------------------------------------------

def test_validation_degenerate_flag_matches_zero_oat_sensitivity():
    """VALIDATION: a pair's `degenerate` flag must be True if and only if at
    least one of its two members has steady_state_D_range <= INERT_TOL in
    `one_at_a_time_sensitivity` at the same fst -- this is the documented
    detection rule (see identifiability_pairs' docstring), re-derived here
    independent of identifiability_pairs' own internals.
    """
    fst = 5
    oat = {r["parameter"]: r["steady_state_D_range"]
           for r in vs.one_at_a_time_sensitivity(fst=fst, verbose=False)}
    out = vs.identifiability_pairs(fst=fst, verbose=False)
    for key, v in out.items():
        a, b = key.split("~")
        expected_inert = [p for p in (a, b) if oat[p] <= 1e-9]
        assert v["degenerate"] == (len(expected_inert) > 0), key
        assert sorted(v["inert_parameters"]) == sorted(expected_inert), key


def test_validation_known_inert_pairs_are_labelled_degenerate():
    out = vs.identifiability_pairs(fst=5, verbose=False)
    assert out["A~D_max"]["degenerate"] is True
    assert out["A~D_max"]["inert_parameters"] == ["D_max"]
    assert "DEGENERATE" in out["A~D_max"]["verdict"]

    assert out["alpha_SC~SC_max"]["degenerate"] is True
    assert out["alpha_SC~SC_max"]["inert_parameters"] == ["SC_max"]
    assert "DEGENERATE" in out["alpha_SC~SC_max"]["verdict"]


def test_validation_informative_ridges_are_not_labelled_degenerate():
    """The pair this fix must NOT demote or relabel: K_const~K_ATTEN and
    K_ATTEN~K_SC are genuine (both-parameters-live) ridges below 1.0000 and
    must keep the UNIDENTIFIABLE verdict, not DEGENERATE.
    """
    out = vs.identifiability_pairs(fst=5, verbose=False)
    for key in ("K_const~K_ATTEN", "K_ATTEN~K_SC"):
        assert out[key]["degenerate"] is False, key
        assert out[key]["inert_parameters"] == [], key
        assert 0.99 < out[key]["ridge_fraction"] < 1.0, (
            f"{key}: expected a genuine ridge strictly below 1.0000, "
            f"got {out[key]['ridge_fraction']}")
        assert "UNIDENTIFIABLE" in out[key]["verdict"]


def test_validation_degenerate_pairs_sort_after_informative_ones_when_printed(capsys):
    """The demotion itself: printed table order must put every non-degenerate
    pair before every degenerate one, regardless of ridge_fraction (the two
    exact-1.0000 rank-1 degeneracies must not sit above K_const~K_ATTEN's
    0.9997 the way they did before this fix).
    """
    vs.identifiability_pairs(fst=5, verbose=True)
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "~" in l and "separable" not in l
             and ("UNIDENTIFIABLE" in l or "DEGENERATE" in l)]
    seen_degenerate = False
    for line in lines:
        if "DEGENERATE" in line:
            seen_degenerate = True
        elif "UNIDENTIFIABLE" in line:
            assert not seen_degenerate, (
                f"an UNIDENTIFIABLE (informative) pair printed after a "
                f"DEGENERATE one: {line!r} in\n{out}")


def test_characterisation_ridge_fraction_pinned_values():
    """PINNED CURRENT VALUES (FST 5, n_grid=9 default) for the two
    genuinely informative ridges. Not independently verified against a
    source -- these are the model's own current output, re-derive if the
    model or calibration changes.
    """
    out = vs.identifiability_pairs(fst=5, verbose=False)
    # RE-PINNED 2026-09-23 for the SWEEP-RANGES bug fix (K_const's sweep
    # range moved from a hardcoded (0.20, 0.90), which excluded the shipped
    # K_const, to the calibration leave-one-out envelope (1.5884400707455726,
    # 3.1839988650360818), which does not). K_const~K_ATTEN's grid is built
    # over PARAM_RANGES["K_const"] x PARAM_RANGES["K_ATTEN"], so it moves:
    # 0.9993128192805887 -> 0.999871349336661. K_ATTEN~K_SC's grid uses only
    # K_ATTEN and K_SC ranges, neither of which this fix touched, so it is
    # UNCHANGED at 0.9967387580224814 (still asserted below as a check that
    # the fix did not have a spillover effect on an unrelated pair).
    assert out["K_const~K_ATTEN"]["ridge_fraction"] == pytest.approx(
        0.999871349336661, abs=1e-9)
    assert out["K_ATTEN~K_SC"]["ridge_fraction"] == pytest.approx(
        0.9967387580224814, abs=1e-9)
    # The two degenerate pairs are EXACTLY 1.0 (not merely close), because a
    # grid over one dead axis is mechanically rank-1 -- this exactness is
    # itself part of what distinguishes "degenerate" from "informative ridge"
    # and is asserted precisely, not with a tolerance.
    assert out["A~D_max"]["ridge_fraction"] == 1.0
    assert out["alpha_SC~SC_max"]["ridge_fraction"] == 1.0


# ---------------------------------------------------------------------------
# uncertainty_propagation
# ---------------------------------------------------------------------------

def test_validation_uncertainty_propagation_percentiles_bracket_median():
    summary = vs.uncertainty_propagation(fst=5, n_draws=30, seed=1, verbose=False)
    for key in ("steady_state_D", "net_per_SED"):
        s = summary[key]
        assert s["p2.5"] <= s["median"] <= s["p97.5"]


def test_validation_uncertainty_propagation_interval_distribution_sums_to_n_draws():
    n_draws = 30
    summary = vs.uncertainty_propagation(fst=5, n_draws=n_draws, seed=2, verbose=False)
    dist = summary["best_interval_distribution"]
    assert sum(dist.values()) == n_draws
    assert set(dist.keys()) <= {1, 2, 3, 4, 5, 7, 10, 14, 21}


def test_validation_uncertainty_propagation_is_seed_deterministic():
    s1 = vs.uncertainty_propagation(fst=5, n_draws=15, seed=7, verbose=False)
    s2 = vs.uncertainty_propagation(fst=5, n_draws=15, seed=7, verbose=False)
    assert s1["steady_state_D"] == s2["steady_state_D"]
    assert s1["best_interval_distribution"] == s2["best_interval_distribution"]


# ---------------------------------------------------------------------------
# run() -- top-level integration
# ---------------------------------------------------------------------------

def test_validation_run_returns_all_three_analyses_per_fst():
    out = vs.run(verbose=False, fsts=(5,))
    assert set(out.keys()) == {5}
    assert set(out[5].keys()) == {"sensitivity", "identifiability", "uncertainty"}
    assert isinstance(out[5]["sensitivity"], list)
    assert isinstance(out[5]["identifiability"], dict)
    assert isinstance(out[5]["uncertainty"], dict)
    # The fix's fields must be present all the way through run()'s return
    # value, not just when identifiability_pairs is called directly.
    for v in out[5]["identifiability"].values():
        assert "degenerate" in v and "verdict" in v
