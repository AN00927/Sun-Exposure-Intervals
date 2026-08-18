"""Tests for vitd_behaviour -- the behavioural exposure term.

The contract this module has to keep is narrow and load-bearing:

  1. it is NEUTRAL by default, so turning it on changes nothing until asked;
  2. it reproduces the measured field cohorts when explicitly enabled;
  3. it REFUSES to invent a cohort that was never measured.

(3) matters as much as (1). Every behavioural measurement in this repo is UK,
FST II vs FST V, South Asian ancestry for the dark arm. Silently interpolating
an FST VI factor would put a fabricated number in front of exactly the users
the project prioritises.
"""

import numpy as np
import pytest

import vitd_behaviour as vb
from vitd_model import ModelParams, simulate


# ---------------------------------------------------------------------------
# 1. Neutral by default
# ---------------------------------------------------------------------------

def test_default_mode_is_scheduling():
    assert vb.MODE_SCHEDULING == "scheduling"
    assert vb.behaviour_factor() == 1.0
    assert vb.behaviour_factor(vb.MODE_SCHEDULING) == vb.NEUTRAL_FACTOR


def test_neutral_factor_is_exactly_one():
    """Not 0.999. Exactly 1.0, so scaling is an identity on floats."""
    assert vb.NEUTRAL_FACTOR == 1.0
    assert isinstance(vb.NEUTRAL_FACTOR, float)


def test_apply_to_schedule_is_identity_at_default():
    sched = [(0.0, 1.3), (2.0, 0.7), (4.5, 2.0)]
    out = vb.apply_to_schedule(sched, vb.behaviour_factor())
    assert out == [(0.0, 1.3), (2.0, 0.7), (4.5, 2.0)]


def test_simulation_is_bit_identical_under_default_factor():
    """The whole point of defaulting to 1.0: model output must not move."""
    sched = [(d * 2.0, 1.3) for d in range(9)]
    base = simulate(fst=5, schedule=sched, duration_days=20,
                    params=ModelParams(), D0=25.0,
                    dose_framing="fixed_absolute_SED")
    via = simulate(fst=5,
                   schedule=vb.apply_to_schedule(sched, vb.behaviour_factor()),
                   duration_days=20, params=ModelParams(), D0=25.0,
                   dose_framing="fixed_absolute_SED")
    assert np.array_equal(np.asarray(base["D"]), np.asarray(via["D"]))


def test_scheduling_mode_rejects_a_cohort():
    """Passing a cohort in scheduling mode is a mistake, not a silent upgrade."""
    with pytest.raises(ValueError, match="scheduling mode"):
        vb.behaviour_factor(vb.MODE_SCHEDULING, cohort="kift2013_south_asian")


# ---------------------------------------------------------------------------
# 2. Reproduces the measured cohorts when explicitly enabled
# ---------------------------------------------------------------------------

def test_cohort_mode_reproduces_kift_dose_ratio():
    # Kift 2013 Table 2 summer: white 3.4, South Asian 1.0 SED per week.
    f = vb.behaviour_factor(vb.MODE_COHORT, cohort="kift2013_south_asian")
    assert f == pytest.approx(1.0 / 3.4, rel=1e-9)
    assert vb.measured_dose_ratio("kift2013") == pytest.approx(3.4, rel=1e-9)


def test_cohort_mode_reproduces_darling_dose_ratio():
    # Webb 2021 Table 5: white 6.39, South Asian 2.00 SED per summer week.
    f = vb.behaviour_factor(vb.MODE_COHORT, cohort="darling2013_south_asian")
    assert f == pytest.approx(2.00 / 6.39, rel=1e-9)
    assert vb.measured_dose_ratio("darling2013") == pytest.approx(3.195, rel=1e-3)


def test_light_cohort_is_its_own_reference():
    for study, (light, _dark) in vb.COHORT_PAIRS.items():
        assert vb.behaviour_factor(vb.MODE_COHORT, cohort=light) == 1.0


def test_cohort_factor_by_fst():
    assert vb.cohort_factor(2, "kift2013") == 1.0
    assert vb.cohort_factor(5, "kift2013") == pytest.approx(1.0 / 3.4, rel=1e-9)


def test_measured_dose_ratios_bracket_the_reported_range():
    """fst_gap_decomposition S2 reports 2.6x-4.0x for the dose ratio; the two
    within-study pairs must sit inside that."""
    for study in vb.COHORT_PAIRS:
        assert 2.6 <= vb.measured_dose_ratio(study) <= 4.0


def test_behaviour_share_matches_the_approved_diagnosis():
    """calibration_approval.yaml records the behavioural share as 2.1-2.4x."""
    shares = list(vb.BEHAVIOUR_SHARE_ON_RESPONSE.values())
    assert min(shares) >= 2.1
    assert max(shares) <= 2.45


def test_behaviour_is_not_time_outdoors():
    """The dark cohort spent MORE time outdoors and still got less UV. If this
    ever flips, the module's whole framing (shade and clothing, not hours)
    would need rewriting."""
    w = vb.COHORTS["kift2013_white"]
    s = vb.COHORTS["kift2013_south_asian"]
    assert s.minutes_outdoors_per_day >= w.minutes_outdoors_per_day
    assert s.sed_summer_week < w.sed_summer_week


def test_personal_dose_is_orders_of_magnitude_below_unshaded_sky_visibility():
    """Kift's cohorts received 0.9-2.9% of ambient. Vernez 2015's unshaded
    sky-visibility values are tens of percent. That gap is why the term is
    shade, not posture geometry."""
    for key in ("kift2013_white", "kift2013_south_asian"):
        lo, hi = vb.COHORTS[key].pct_of_ambient
        assert 0.0 < lo <= hi < 5.0


def test_applying_a_cohort_factor_lowers_the_modelled_response():
    sched = [(float(d), 1.0) for d in range(60)]
    p = ModelParams()
    full = simulate(fst=5, schedule=sched, duration_days=60, params=p,
                    D0=25.0, exposed_bsa_override=0.14,
                    dose_framing="fixed_absolute_SED")
    f = vb.cohort_factor(5, "kift2013")
    damped = simulate(fst=5,
                      schedule=vb.apply_to_schedule(sched, f),
                      duration_days=60, params=p, D0=25.0,
                      exposed_bsa_override=0.14,
                      dose_framing="fixed_absolute_SED")
    assert damped["D"][-1] < full["D"][-1]


# ---------------------------------------------------------------------------
# 3. Refuses to invent what was never measured
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fst", [1, 3, 4, 6])
def test_no_cohort_factor_for_unmeasured_skin_types(fst):
    with pytest.raises(ValueError, match="measured no FST"):
        vb.cohort_factor(fst, "kift2013")


def test_fst_vi_is_explicitly_unavailable():
    """The dark-skin priority case: FST VI has no field dosimetry anywhere."""
    with pytest.raises(ValueError) as exc:
        vb.cohort_factor(6, "darling2013")
    assert "no FST VI measurement" in str(exc.value)


def test_cross_study_reference_is_rejected():
    with pytest.raises(ValueError, match="different studies"):
        vb.behaviour_factor(vb.MODE_COHORT,
                            cohort="kift2013_south_asian",
                            reference="darling2013_white")


def test_cohort_mode_requires_a_cohort():
    with pytest.raises(ValueError, match="needs an explicit cohort"):
        vb.behaviour_factor(vb.MODE_COHORT)


def test_unknown_mode_and_cohort_raise():
    with pytest.raises(ValueError, match="unknown mode"):
        vb.behaviour_factor("population_average")
    with pytest.raises(ValueError, match="unknown cohort"):
        vb.behaviour_factor(vb.MODE_COHORT, cohort="nhanes_black")


def test_negative_factor_rejected():
    with pytest.raises(ValueError):
        vb.apply_to_schedule([(0.0, 1.0)], -0.5)


# ---------------------------------------------------------------------------
# 4. Provenance is carried, not assumed
# ---------------------------------------------------------------------------

def test_every_cohort_carries_a_source():
    for key, c in vb.COHORTS.items():
        assert c.source and len(c.source) > 10, key
        assert c.n > 0, key
        assert c.sed_summer_week > 0, key


def test_describe_states_the_dark_skin_caveat():
    for study in vb.COHORT_PAIRS:
        text = vb.describe(study)
        assert "South Asian" in text
        assert "No FST VI cohort exists." in text
        assert "SHADE AND CLOTHING" in text


def test_module_does_not_import_the_optical_model():
    """Behaviour must not reach into the pigment parameters. If this module
    ever needs ModelParams, the split has leaked."""
    import inspect
    src = inspect.getsource(vb)
    assert "from vitd_model" not in src
    assert "import vitd_model" not in src
