"""
tests/test_vitd_sensitivity.py
================================
Coverage for vitd_sensitivity.py (previously zero tests), and the primary
regression test for the 2026-09-20 `identifiability_pairs` reporting fix
(see docs/workstreams/W-O.md and the "DEGENERATE vs UNIDENTIFIABLE" section
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

import numpy as np
import pytest

import vitd_sensitivity as vs


# ---------------------------------------------------------------------------
# PARAM_RANGES -- module-level documented prior ranges
# ---------------------------------------------------------------------------

def test_validation_param_ranges_low_below_high():
    for name, (lo, hi, note) in vs.PARAM_RANGES.items():
        assert lo < hi, f"{name}: low {lo} is not < high {hi}"
        assert isinstance(note, str) and len(note) > 0


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
    """
    rows = vs.one_at_a_time_sensitivity(fst=5, verbose=False)
    top = rows[0]
    assert top["parameter"] == "K_const"
    assert top["steady_state_D_range"] == pytest.approx(9.534915652060903, abs=1e-9)


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
    assert out["K_const~K_ATTEN"]["ridge_fraction"] == pytest.approx(
        0.99966957901959, abs=1e-9)
    assert out["K_ATTEN~K_SC"]["ridge_fraction"] == pytest.approx(
        0.9985529218046425, abs=1e-9)
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
