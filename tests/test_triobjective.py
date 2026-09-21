"""W-R: the A3 tri-objective (dose x spacing x cancer risk).

These tests exist to pin the two things the workstream was asked to make
STRUCTURAL rather than documented:

  1. Cross-skin-type risk VALUES must be unreadable off this objective. The
     FST susceptibility factor in `vitd_physiology.cancer_risk` is a
     constitutive prior and is not anchored by any epidemiological source in
     the repo (Elwood & Jopson 1997 adjusts its odds ratios FOR pigmentation
     -- see docs/workstreams/W-R.md). Within-skin-type comparisons stay usable.

  2. The session budget is an ADHERENCE assumption, not photobiology, and the
     objective's own output has to say so.

They are behavioural tests of the API contract, not validation of any
scientific value. No parameter value is asserted here.
"""
import pytest

from vitd_optimization import (SESSION_BUDGET_BASIS, CrossSkinTypeRiskError,
                               WithinFSTRisk, pareto_front_nd,
                               tri_objective_frontier)

BUDGETS = (183, 365)
GAMMAS = (1.0, 1.4)


@pytest.fixture(scope="module")
def front3():
    return tri_objective_frontier(fst=3, session_budgets=BUDGETS,
                                  gammas=GAMMAS)


@pytest.fixture(scope="module")
def front5():
    return tri_objective_frontier(fst=5, session_budgets=BUDGETS,
                                  gammas=GAMMAS)


class TestRiskAxisIsSealedToOneSkinType:

    def test_risk_axis_is_not_a_bare_number(self, front3):
        for r in front3["rows"]:
            assert isinstance(r["combined_risk"], WithinFSTRisk)

    def test_float_is_refused(self, front3):
        risk = front3["rows"][0]["combined_risk"]
        with pytest.raises(CrossSkinTypeRiskError):
            float(risk)

    def test_numeric_formatting_is_refused(self, front3):
        """The failure mode this guards: an f-string with a numeric spec is
        how an unanchored value gets into a printed table."""
        risk = front3["rows"][0]["combined_risk"]
        with pytest.raises(CrossSkinTypeRiskError):
            format(risk, ".4f")
        assert "FST 3" in f"{risk}"

    def test_arithmetic_is_refused(self, front3):
        risk = front3["rows"][0]["combined_risk"]
        with pytest.raises(CrossSkinTypeRiskError):
            risk * 2.0

    def test_cross_fst_ordering_is_refused(self, front3, front5):
        a = front3["rows"][0]["combined_risk"]
        b = front5["rows"][0]["combined_risk"]
        with pytest.raises(CrossSkinTypeRiskError):
            a < b
        with pytest.raises(CrossSkinTypeRiskError):
            a <= b

    def test_escape_hatch_requires_the_matching_skin_type(self, front3):
        risk = front3["rows"][0]["combined_risk"]
        assert risk.absolute_value_for_fst(3) > 0.0
        with pytest.raises(CrossSkinTypeRiskError):
            risk.absolute_value_for_fst(5)

    def test_within_fst_ordering_still_works(self, front3):
        feas = [r["combined_risk"] for r in front3["rows"] if r["feasible"]]
        assert len(feas) >= 2
        ordered = sorted(feas)
        assert ordered[0] <= ordered[-1]

    def test_published_ratio_is_self_normalised_per_skin_type(self, front3,
                                                              front5):
        """Every skin type's best feasible cell reads exactly 1.0, so the
        published number carries no cross-FST magnitude to misread."""
        for res in (front3, front5):
            feas = [r["risk_within_fst_ratio"] for r in res["rows"]
                    if r["feasible"]]
            if not feas:
                continue
            assert min(feas) == pytest.approx(1.0)
            assert all(v >= 1.0 - 1e-12 for v in feas)

    def test_result_declares_the_axis_restriction(self, front3):
        assert front3["risk_axis_is_within_fst_only"] is True


class TestParetoStillWorksOnASealedAxis:

    def test_frontier_is_nonempty_and_nondominated(self, front3):
        axes = front3["axes"]
        assert front3["frontier"]
        for p in front3["frontier"]:
            for q in front3["frontier"]:
                if p is q:
                    continue
                assert not (all(q[k] <= p[k] for k in axes)
                            and any(q[k] < p[k] for k in axes))

    def test_pareto_front_nd_still_accepts_plain_floats(self):
        pts = [{"a": 1.0, "b": 5.0}, {"a": 3.0, "b": 9.0}]
        assert pareto_front_nd(pts, ("a", "b")) == [pts[0]]

    def test_ranks_agree_with_the_sealed_ordering(self, front3):
        feas = [r for r in front3["rows"] if r["feasible"]]
        by_rank = sorted(feas, key=lambda r: r["risk_rank_within_fst"])
        for lo, hi in zip(by_rank, by_rank[1:]):
            assert lo["combined_risk"] <= hi["combined_risk"]


class TestSessionBudgetIsAnAdherenceAssumption:

    def test_basis_string_is_carried_in_the_result(self, front3):
        assert front3["session_budget_basis"] == SESSION_BUDGET_BASIS

    def test_basis_string_says_what_it_is(self):
        text = SESSION_BUDGET_BASIS.lower()
        assert "adherence" in text
        assert "assumption" in text
        assert "365" in SESSION_BUDGET_BASIS

    def test_budget_is_an_axis_not_a_default(self, front3):
        """Swept, so the assumption is visible rather than buried."""
        assert len({r["n_sessions"] for r in front3["rows"]}) == len(BUDGETS)
