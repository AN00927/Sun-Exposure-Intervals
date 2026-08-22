"""The behaviour/optics split, guarded against the RETIRED behaviour module.

MOVED here 2026-08-21 from tests/test_kconst_calibration.py, when
`vitd_behaviour` was retired to `exploratory/` (see exploratory/README.md).
The tests are unchanged in substance: retirement means "does not influence
headline outputs", not "unmaintained".

Why they still matter. `K_const` (optics) and the behavioural multiplier
(shade, clothing, time outdoors) are MULTIPLICATIVE and must not be merged.
Dumping behaviour into K_const would require an optical factor above every
matched-dose measurement in existence -- these tests are what stops that
happening by accident. The argument is documented in
`exploratory/vitd_behaviour.py` and `calibration_approval.yaml`
(a1_diagnosis_result_2026_08_17).

The half of the argument that does not need the retired module -- that
K_const alone stays clear of the behaviour-blind floor -- remains in the core
suite as test_does_not_absorb_the_behavioural_share.
"""

import numpy as np

from vitd_model import ModelParams

# What dumping the behavioural share into K_const would have required.
# Kept in sync deliberately with tests/test_kconst_calibration.py; both read
# it from calibration_approval.yaml a1_diagnosis_result_2026_08_17.
BEHAVIOUR_BLIND_FLOOR = 4.28


def optical(K, dark, light=2):
    p = ModelParams()
    return float(np.exp(K * (p.M_baseline(dark) - p.M_baseline(light))))


def test_behaviour_module_is_neutral_by_default():
    """Retirement did not change any number, because the module was already
    inert: the scheduling-mode factor is 1.0 for every skin type."""
    import vitd_behaviour as vb
    assert vb.behaviour_factor() == 1.0


def test_optical_and_behavioural_factors_are_separable():
    """Their product is what the behaviour-blind field inversion measured
    (4.3-5.7x). Each on its own must be well below that."""
    import vitd_behaviour as vb
    opt = optical(ModelParams().K_const, 5)
    beh = vb.measured_dose_ratio("kift2013")
    assert opt < BEHAVIOUR_BLIND_FLOOR
    assert opt * beh > BEHAVIOUR_BLIND_FLOOR   # together they over-explain it
