"""Tests for the decision-problem objectives.

These guard the property that made the old headline metric unusable: an
objective must not be winnable by simply dosing less.
"""

import numpy as np
import pytest

from vitd_model import ModelParams
from vitd_objectives import (optimal_spacing_fixed_sessions,
                             deficiency_vs_risk_frontier,
                             minutes_for_target,
                             sed_from_minutes,
                             SED_PER_MIN_PER_UVI,
                             PERSONAL_DOSE_GEOMETRY_FACTOR,
                             SUFFICIENCY_NMOL_L)


class TestFixedSessionSpacing:

    def test_total_dose_is_constant_across_candidates(self):
        """Cumulative UV is held fixed, so a spacing cannot win by reducing
        exposure. Contrast the retired per-SED efficiency metric (removed
        from vitd_model.py 2026-07-31), whose optimum was achieved by
        dosing toward zero."""
        r = optimal_spacing_fixed_sessions(fst=5, n_sessions=10)
        totals = {round(row["total_sed"], 9) for row in r["table"]}
        assert len(totals) == 1, f"cumulative dose varied: {totals}"

    def test_returns_a_best_and_a_table(self):
        r = optimal_spacing_fixed_sessions(fst=5, n_sessions=8)
        assert r["best"] is not None
        assert len(r["table"]) > 1
        assert r["best"] in r["table"]

    def test_objective_actually_selects(self):
        r_mean = optimal_spacing_fixed_sessions(fst=5, n_sessions=12, objective="mean_D")
        r_final = optimal_spacing_fixed_sessions(fst=5, n_sessions=12, objective="final_D")
        assert r_mean["best"]["mean_D"] >= r_final["best"]["mean_D"]
        assert r_final["best"]["final_D"] >= r_mean["best"]["final_D"]

    def test_deficit_objective_is_minimised(self):
        r = optimal_spacing_fixed_sessions(fst=5, n_sessions=12,
                                           objective="auc_deficit_50")
        best = r["best"]["auc_deficit_50"]
        assert all(best <= row["auc_deficit_50"] for row in r["table"])

    def test_more_sessions_is_never_worse_for_mean_D(self):
        a = optimal_spacing_fixed_sessions(fst=5, n_sessions=6)["best"]["mean_D"]
        b = optimal_spacing_fixed_sessions(fst=5, n_sessions=18)["best"]["mean_D"]
        assert b > a

    def test_rejects_zero_sessions(self):
        with pytest.raises(ValueError):
            optimal_spacing_fixed_sessions(fst=5, n_sessions=0)

    def test_risk_reported_alongside(self):
        """A schedule must never be recommendable on synthesis grounds alone."""
        r = optimal_spacing_fixed_sessions(fst=5, n_sessions=10)
        for key in ("scc_index", "intermittent_index", "burn_fraction"):
            assert key in r["best"]


class TestParetoFrontier:

    def test_frontier_is_nondominated(self):
        r = deficiency_vs_risk_frontier(fst=5, horizon_days=120,
                                        sed_grid=(0.5, 1.0, 2.0),
                                        interval_grid=(1, 3, 7, 14))
        front = r["frontier"]
        assert len(front) >= 1
        for p in front:
            for q in r["all_points"]:
                strictly_better = (
                    q["days_below_target"] <= p["days_below_target"]
                    and q["combined_risk"] <= p["combined_risk"]
                    and (q["days_below_target"] < p["days_below_target"]
                         or q["combined_risk"] < p["combined_risk"]))
                assert not strictly_better

    def test_frontier_sorted_and_trades_off(self):
        r = deficiency_vs_risk_frontier(fst=5, horizon_days=120,
                                        sed_grid=(0.5, 1.0, 2.0, 4.0),
                                        interval_grid=(1, 3, 7, 14, 28))
        front = r["frontier"]
        defs = [p["days_below_target"] for p in front]
        assert defs == sorted(defs)
        if len(front) > 1:
            risks = [p["combined_risk"] for p in front]
            assert risks[0] >= risks[-1], "frontier must trade off, not dominate"

    def test_cheapest_sufficient_is_actually_sufficient(self):
        r = deficiency_vs_risk_frontier(fst=3, horizon_days=120,
                                        sed_grid=(0.5, 1.0, 2.0),
                                        interval_grid=(1, 2, 3, 7))
        cs = r["cheapest_sufficient"]
        if cs is not None:
            assert cs["days_below_target"] <= 0.05 * 120


class TestMinutesConversion:

    def test_uvi_definition(self):
        """1 UVI = 25 mW/m^2 erythemal; 1 SED = 100 J/m^2.
        => SED/min = UVI * 25e-3 * 60 / 100 = UVI * 0.015"""
        assert SED_PER_MIN_PER_UVI == pytest.approx(25e-3 * 60 / 100)

    def test_horizontal_vs_personal(self):
        horiz = sed_from_minutes(30, 7.0, geometry_factor=1.0)
        personal = sed_from_minutes(30, 7.0)
        assert personal < horiz
        assert personal == pytest.approx(horiz * PERSONAL_DOSE_GEOMETRY_FACTOR)

    def test_linear_in_minutes_and_uvi(self):
        assert sed_from_minutes(20, 5.0) == pytest.approx(2 * sed_from_minutes(10, 5.0))
        assert sed_from_minutes(10, 10.0) == pytest.approx(2 * sed_from_minutes(10, 5.0))

    def test_negative_minutes_clamped(self):
        assert sed_from_minutes(-5, 7.0) == 0.0


class TestMinutesForTarget:

    def test_darker_skin_needs_longer(self):
        m = {}
        for fst in (2, 5, 6):
            r = minutes_for_target(fst=fst, uv_index=7.0, exposed_bsa=0.35,
                                   frequency_per_week=7, horizon_days=120)
            assert r["reachable"], f"FST {fst} should reach 50 nmol/L at UVI 7"
            m[fst] = r["minutes"]
        assert m[2] < m[5] <= m[6]

    def test_more_exposed_skin_needs_less_time(self):
        a = minutes_for_target(fst=5, uv_index=7.0, exposed_bsa=0.10,
                               frequency_per_week=7, horizon_days=120)
        b = minutes_for_target(fst=5, uv_index=7.0, exposed_bsa=0.35,
                               frequency_per_week=7, horizon_days=120)
        assert a["minutes"] > b["minutes"]

    def test_reports_burn_status(self):
        r = minutes_for_target(fst=5, uv_index=7.0, exposed_bsa=0.35,
                               frequency_per_week=7, horizon_days=120)
        assert "would_burn" in r and "fraction_med" in r
        assert r["fraction_med"] >= 0.0

    def test_unreachable_is_reported_not_faked(self):
        """A recommendation that cannot work must say so rather than return a
        number at the search bound."""
        r = minutes_for_target(fst=6, uv_index=0.3, exposed_bsa=0.05,
                               frequency_per_week=1, horizon_days=60,
                               target=150.0, max_minutes=60.0)
        assert r["reachable"] is False
        assert r["minutes"] is None
        assert "note" in r

    def test_achieves_requested_target(self):
        r = minutes_for_target(fst=5, uv_index=7.0, exposed_bsa=0.35,
                               frequency_per_week=7, horizon_days=150,
                               target=SUFFICIENCY_NMOL_L)
        assert r["reachable"]
        assert r["steady_state_D"] >= SUFFICIENCY_NMOL_L - 1.0


# ---------------------------------------------------------------------------
# The scheduling metric hierarchy (Phase 4b, item A3)
# ---------------------------------------------------------------------------
class TestObjectiveHierarchy:
    """Structural guards preventing the ranking from quietly growing a second
    headline or losing the barrier around the diagnostic tier."""

    def test_exactly_one_primary_objective(self):
        """A hierarchy with two headlines has no headline."""
        from vitd_objectives import (OBJECTIVE_HIERARCHY, TIER_PRIMARY,
                                     primary_objective)
        primaries = [s for s in OBJECTIVE_HIERARCHY if s.tier == TIER_PRIMARY]
        assert len(primaries) == 1
        assert primary_objective() is primaries[0]

    def test_primary_is_reach_and_hold_not_a_ratio(self):
        """The PRIMARY objective must be a constrained minimisation with the
        sufficiency floor among its constraints, the property every retired
        headline metric lacked."""
        from vitd_objectives import primary_objective
        spec = primary_objective()
        assert spec.minimise, "the primary objective must minimise something"
        assert any("C3" in c for c in spec.subject_to), spec.subject_to
        assert any("C1" in c for c in spec.subject_to), spec.subject_to

    def test_keys_and_ranks_are_unique(self):
        from vitd_objectives import OBJECTIVE_HIERARCHY
        keys = [s.key for s in OBJECTIVE_HIERARCHY]
        ranks = [s.rank for s in OBJECTIVE_HIERARCHY]
        assert len(set(keys)) == len(keys)
        assert len(set(ranks)) == len(ranks)

    def test_ranks_respect_tier_order(self):
        """PRIMARY outranks SECONDARY outranks DIAGNOSTIC, numerically."""
        from vitd_objectives import OBJECTIVE_HIERARCHY, TIER_ORDER
        pos = {t: i for i, t in enumerate(TIER_ORDER)}
        ordered = sorted(OBJECTIVE_HIERARCHY, key=lambda s: s.rank)
        tiers = [pos[s.tier] for s in ordered]
        assert tiers == sorted(tiers), [(s.rank, s.tier) for s in ordered]

    def test_gross_synth_is_diagnostic_only(self):
        """The specific demotion item A3 asks for."""
        from vitd_objectives import (objective_tier, TIER_DIAGNOSTIC,
                                     is_headline)
        assert objective_tier("gross_synth_per_SED") == TIER_DIAGNOSTIC
        assert not is_headline("gross_synth_per_SED")

    def test_diagnostics_cannot_be_reported_as_headline(self):
        """The guard is the point: report code that tries to present a
        diagnostic as advice must fail loudly rather than print."""
        from vitd_objectives import (assert_reportable_as_headline,
                                     OBJECTIVE_HIERARCHY, TIER_DIAGNOSTIC)
        for spec in OBJECTIVE_HIERARCHY:
            if spec.tier == TIER_DIAGNOSTIC:
                with pytest.raises(ValueError):
                    assert_reportable_as_headline(spec.key)
            else:
                assert_reportable_as_headline(spec.key)

    def test_unknown_objective_raises_rather_than_defaults(self):
        """Defaulting an unplaced metric into some tier is how the ranking
        drifts; refusing to guess is the whole mechanism."""
        from vitd_objectives import objective_tier
        with pytest.raises(KeyError):
            objective_tier("some_metric_nobody_ranked")

    def test_every_entry_points_at_something_that_exists(self):
        """`implemented_by` is a promise. Check the promise: each entry names
        either an importable attribute or a file that is present."""
        import importlib
        import os
        from vitd_objectives import OBJECTIVE_HIERARCHY

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for spec in OBJECTIVE_HIERARCHY:
            found = False
            for token in spec.implemented_by.replace("/", " ").split():
                token = token.strip("(),;")
                if token.endswith(".py") and os.path.exists(
                        os.path.join(root, "scripts", token)):
                    found = True
                    break
                if "." not in token:
                    continue
                mod, _, attr = token.rpartition(".")
                try:
                    if hasattr(importlib.import_module(mod), attr):
                        found = True
                        break
                except Exception:
                    continue
            assert found, f"{spec.key}: nothing found for {spec.implemented_by!r}"

    def test_formatted_hierarchy_names_every_objective_in_tier_order(self):
        from vitd_objectives import (format_objective_hierarchy,
                                     OBJECTIVE_HIERARCHY, TIER_ORDER)
        text = format_objective_hierarchy()
        for spec in OBJECTIVE_HIERARCHY:
            assert spec.key in text
        positions = [text.index(t) for t in TIER_ORDER if t in text]
        assert positions == sorted(positions), "tiers printed out of order"

    def test_report_generator_does_not_hardcode_the_geometry_factor(self):
        """Drift guard for the bug this phase fixed. run_all.py stated a
        personal-dose geometry factor of 0.55 while the code held a different
        value; the fix was to interpolate it. Re-hardcoding the value,
        right or wrong, puts the drift back.

        REWRITTEN: previously split on the literal heading
        "## 17. Interpretation" with no check that the split actually found
        it. `str.split` on an absent substring is a no-op (returns the whole
        string as a single-element list), so a renumbered/renamed heading
        made `report = src.split(...)[-1]` silently become the WHOLE file
        rather than just the Interpretation section -- the two assertions
        below would then still run, just scoped wrong, so a genuine
        reintroduction of the bug inside the (mis-scoped) section would
        still have been caught by luck, but the test's own stated scope
        ("the generated report") would have quietly become "everything",
        which is not what it claims to check and would also start flagging
        unrelated, legitimate uses of "0.55" elsewhere in the file. Now
        asserts the anchor is found exactly once before scoping the scan,
        so a renumber fails loudly instead of silently drifting the scope.
        """
        import io
        import os
        from vitd_objectives import PERSONAL_DOSE_GEOMETRY_FACTOR

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "run_all.py")
        src = io.open(path, encoding="utf-8").read()
        assert "PERSONAL_DOSE_GEOMETRY_FACTOR" in src, (
            "run_all.py no longer reads the geometry factor from the code")

        heading = "## 17. Interpretation"
        occurrences = src.count(heading)
        assert occurrences == 1, (
            f"expected exactly one {heading!r} anchor in run_all.py to "
            f"scope this check to the Interpretation section, found "
            f"{occurrences}; the section was likely renumbered or "
            "duplicated -- update this test's anchor string to match "
            "rather than let the scan silently fall back to the whole file"
        )
        report = src.split(heading, 1)[1]
        assert repr(PERSONAL_DOSE_GEOMETRY_FACTOR) not in report, (
            "the geometry factor is hardcoded in the generated report again")
        assert "0.55" not in report.replace(
            "stated a hardcoded 0.55", ""), (
            "a stale geometry factor is back in the generated report")
