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
