"""Tests for the Matsuoka 1990 K_ATTEN derivation.

The data are in the repo (abstract, PMID 2165512) and always were. These tests
lock in the arithmetic and, more importantly, the finding that the widely cited
"51% tanning attenuation" is confounded by a baseline 25(OH)D difference.
"""

import numpy as np
import pytest

from vitd_model import (ModelParams, simulate, steady_state_D,
                        allow_retired_parameter_sweep)
from vitd_calibration import (
    derive_K_ATTEN, run,
    UNTANNED_RISE, TANNED_RISE, NAIVE_ATTENUATION,
    UNTANNED_25OHD, TANNED_25OHD,
)


class TestAbstractArithmetic:

    def test_rises_match_abstract(self):
        assert UNTANNED_RISE == pytest.approx(9.3)
        assert TANNED_RISE == pytest.approx(4.7)

    def test_naive_attenuation_reproduces_cited_51_percent(self):
        assert NAIVE_ATTENUATION == pytest.approx(0.495, abs=0.01)

    def test_tanned_subjects_started_higher(self):
        """The confound: tanned baseline 25(OH)D was 64% higher, so the naive
        ratio conflates melanin attenuation with D-feedback."""
        assert TANNED_25OHD > UNTANNED_25OHD
        assert TANNED_25OHD / UNTANNED_25OHD == pytest.approx(1.64, abs=0.02)


class TestDeconvolution:

    def test_feedback_absorbs_part_of_the_blunting(self):
        """Under any real feedback form, melanin-only attenuation must be
        SMALLER than the naive 49.5%."""
        for mode in ("linear", "biphasic", "saturating"):
            r = derive_K_ATTEN(mode)
            assert r["valid"], f"{mode} produced an invalid residual"
            assert r["melanin_attenuation"] < NAIVE_ATTENUATION

    def test_no_feedback_recovers_the_naive_number(self):
        r = derive_K_ATTEN("none")
        assert r["melanin_attenuation"] == pytest.approx(NAIVE_ATTENUATION)

    def test_result_is_feedback_model_dependent(self):
        """The abstract constrains the PRODUCT of feedback and melanin, so the
        split depends on the assumed feedback form. Reporting one number
        without stating the form would be overclaiming."""
        ks = [derive_K_ATTEN(m)["K_ATTEN"]
              for m in ("linear", "biphasic", "saturating")]
        assert max(ks) / min(ks) > 2.0, "forms should disagree materially"

    def test_smaller_assumed_tan_implies_larger_K(self):
        """K_ATTEN and excess_M trade off: the same observed attenuation needs a
        bigger coefficient if the tan is assumed smaller."""
        big = derive_K_ATTEN("linear", excess_M=0.20)["K_ATTEN"]
        small = derive_K_ATTEN("linear", excess_M=0.10)["K_ATTEN"]
        assert small > big

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            derive_K_ATTEN("nonsense")


class TestConsistencyWithModel:
    """The model does NOT use a Matsuoka-consistent K_ATTEN, and that is a
    deliberate, documented divergence rather than a defect.

    History: K_ATTEN was 0.37, taken from Young 2020's FST II:VI ratio, and it
    happened to sit inside the range Matsuoka implies. It was later refitted to
    the Young 2020 per-person dataset (n=23, TL01 arm) by 1D MSE minimisation on
    individual delta 25(OH)D, which drove it to 0.0 -- baseline 25(OH)D and BSA
    absorb the whole FST II:VI difference, leaving no residual for facultative
    melanin. See vitd_model.ModelParams for the canonical value and citation.

    These two quantities were never the same thing:
      - Matsuoka measures a WITHIN-SUBJECT FACULTATIVE effect (same person,
        tanned vs untanned).
      - The Young 2020 fit measures a CONSTITUTIVE between-subject effect.
    So the divergence is expected. It is locked in here so that changing
    K_ATTEN, or changing the deconvolution, has to argue with this test rather
    than silently flip the story.
    """

    def test_current_K_ATTEN_sits_below_the_matsuoka_facultative_range(self):
        """The headline check, stated in the direction that is actually true.

        The model assigns ZERO facultative-melanin penalty, which is strictly
        below anything Matsuoka's residual can support under any feedback form.
        Asserting mere inequality would also pass if K_ATTEN went far ABOVE the
        range, which would be a different and much more serious disagreement.
        """
        out = run(verbose=False)
        lo, hi = out["implied_range"]
        assert out["consistent"] is False
        assert out["current_K_ATTEN"] < lo, (
            f"K_ATTEN={out['current_K_ATTEN']} is no longer below the Matsuoka "
            f"facultative range [{lo:.2f}, {hi:.2f}]. If this was intentional, "
            f"the divergence documented in this class needs rewriting.")

    def test_the_divergence_does_not_change_conclusions(self):
        """What makes the divergence tolerable, checked rather than asserted.

        MODEL_AND_EVIDENCE.md's claim that melanin is a weak channel is the only
        reason a disagreement with the sole within-subject study in existence is
        acceptable. If that stopped holding, the divergence would become
        load-bearing and would have to be resolved rather than documented.
        """
        out = run(verbose=False)
        lo, hi = out["implied_range"]
        schedule = [(float(d), 1.0) for d in range(0, 365, 3)]

        for fst in (2, 5, 6):
            ss = []
            for k in (out["current_K_ATTEN"], lo, hi):
                p = ModelParams()
                # K_ATTEN was RETIRED 2026-08-18 (approved item A6). This loop
                # is exactly the sensitivity sweep the retirement rests on --
                # it MEASURES how little the parameter matters -- so it goes
                # through the explicit escape hatch rather than around it.
                with allow_retired_parameter_sweep():
                    p.K_ATTEN = float(k)
                r = simulate(fst=fst, schedule=schedule, duration_days=365,
                             params=p, D0=25.0,
                             dose_framing="fixed_per_session_SED")
                ss.append(steady_state_D(r))
            swing = (max(ss) - min(ss)) / max(ss)
            assert swing < 0.02, (
                f"FST {fst}: steady-state 25(OH)D moves {swing:.1%} across the "
                f"Matsuoka K_ATTEN range. Melanin is no longer a weak channel, "
                f"so the K_ATTEN divergence can no longer be waved through.")

    def test_run_reports_the_divergence_rather_than_hiding_it(self):
        out = run(verbose=False)
        assert out["consistent"] is False
        assert out["implied_range"] is not None

    def test_run_returns_all_forms(self):
        out = run(verbose=False)
        modes = {r["feedback_mode"] for r in out["results"]}
        assert modes == {"linear", "biphasic", "saturating", "none"}
