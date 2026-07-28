"""Tests for Bogh 2012 dose schedule."""

import pytest


def test_number_of_sessions():
    from vitd_protocols.bogh2012 import bogh2012_session_doses
    doses = bogh2012_session_doses()
    assert len(doses) == 18


def test_first_two_sessions_equal_start():
    from vitd_protocols.bogh2012 import bogh2012_session_doses
    doses = bogh2012_session_doses()
    assert doses[0] == 0.36
    assert doses[1] == 0.36


def test_no_dose_exceeds_cap():
    from vitd_protocols.bogh2012 import bogh2012_session_doses
    doses = bogh2012_session_doses()
    assert all(d <= 1.20 for d in doses)


def test_escalation_every_two_sessions():
    from vitd_protocols.bogh2012 import bogh2012_session_doses
    doses = bogh2012_session_doses()
    assert doses[2] > doses[0]
    assert doses[3] == doses[2]
