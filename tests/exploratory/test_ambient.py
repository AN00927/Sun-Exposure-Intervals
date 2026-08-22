"""
tests/test_ambient.py
======================
Tests for vitd_ambient.py. No test makes a real network call. fetch_ambient_uvi()
always gets a stub `http_get` returning either a recorded fixture
(tests/fixtures/open_meteo_*.json) or an inline literal for a specific failure
path.

FIXTURE PROVENANCE
-------------------
tests/fixtures/open_meteo_forecast_la_20260817.json and
tests/fixtures/open_meteo_forecast_seattle_20260817.json are real responses
captured live via:

    curl -s "https://api.open-meteo.com/v1/forecast?latitude=<LAT>&longitude=<LON>&hourly=uv_index,uv_index_clear_sky[,cloud_cover]&timezone=UTC&forecast_days=1"

on 2026-08-17 (LA: latitude=34.05, longitude=-118.25; Seattle: latitude=47.6,
longitude=-122.33, with cloud_cover added so the fixture includes an observed
cloud_cover% alongside the two UVI series. The module itself never reads
cloud_cover; the field is only there so the fixture is independently checkable
by eye). The one edit after capture was re-requesting with timezone=UTC (instead
of a local IANA zone) so hourly.time strings line up with this module's UTC-only
design; see vitd_ambient.py's FIELDS section for why UTC was chosen.

tests/fixtures/open_meteo_error_bad_latitude.json is the real error body
Open-Meteo returned live on 2026-08-17 for latitude=999.

Real values used below:
  LA,  2026-08-17T20:00Z: uv_index=8.10, uv_index_clear_sky=8.15 (near-clear day)
  LA,  2026-08-17T04:00Z: uv_index=0.00, uv_index_clear_sky=0.00 (night)
  SEA, 2026-08-17T15:00Z: uv_index=0.75, uv_index_clear_sky=0.80, cloud_cover=100%
       (a measured sub-1.0 cloud modifier from real data, not constructed)
"""

import datetime as dt
import datetime as _dt
import json as _json
import json
import math
from pathlib import Path

import pytest

from vitd_ambient import (
    AmbientUVI,
    AmbientUVIUnavailable,
    cloud_modifier,
    fetch_ambient_uvi,
    uvi_from_user_input,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_HISTORICAL_URL,
    HOUR_MATCH_TOLERANCE_MINUTES,
)

# Fixtures stayed in tests/fixtures/ when this test moved to
# tests/exploratory/ (retirement, 2026-08-21) -- they are shared.
FIXTURES = Path(__file__).parent.parent / "fixtures"
UTC = dt.timezone.utc


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


LA_FIXTURE = _load("open_meteo_forecast_la_20260817.json")
SEATTLE_FIXTURE = _load("open_meteo_forecast_seattle_20260817.json")
ERROR_FIXTURE = _load("open_meteo_error_bad_latitude.json")


def _stub(text: str):
    """Return an http_get(url, timeout) stub that ignores its arguments."""
    def _get(url, timeout):
        return text
    return _get


def _recording_stub(text: str):
    """Like _stub but remembers the last URL it was called with and refuses
    to be called twice without a reset. Asserts local validation short-circuits
    before any network call."""
    calls = []

    def _get(url, timeout):
        calls.append(url)
        return text
    _get.calls = calls
    return _get


# ---------------------------------------------------------------------------
# fetch_ambient_uvi: happy path, real recorded data
# ---------------------------------------------------------------------------

class TestFetchHappyPath:
    def test_matches_exact_hour_la_daytime(self):
        when = dt.datetime(2026, 8, 17, 20, 0, tzinfo=UTC)
        result = fetch_ambient_uvi(34.05, -118.25, when, http_get=_stub(LA_FIXTURE))
        assert isinstance(result, AmbientUVI)
        assert result.timestamp == "2026-08-17T20:00"
        assert result.uv_index_live == pytest.approx(8.10)
        assert result.uv_index_clear_sky == pytest.approx(8.15)
        assert result.effective_uv_index == pytest.approx(8.10)
        assert result.is_fallback is False
        assert result.source == "open-meteo-forecast"
        assert result.cloud_modifier == pytest.approx(8.10 / 8.15)
        assert result.latitude == 34.05
        assert result.longitude == -118.25

    def test_nearest_hour_matching_within_tolerance(self):
        # 20 minutes off the hour, should snap to the 20:00 entry.
        when = dt.datetime(2026, 8, 17, 20, 20, tzinfo=UTC)
        result = fetch_ambient_uvi(34.05, -118.25, when, http_get=_stub(LA_FIXTURE))
        assert result.timestamp == "2026-08-17T20:00"
        assert HOUR_MATCH_TOLERANCE_MINUTES == pytest.approx(90.0)

    def test_seattle_cloudy_hour_real_submodifier(self):
        # Real recorded partly-cloudy hour: 0.75 / 0.80, a genuine <1.0
        # cloud modifier from Open-Meteo, not constructed by us.
        when = dt.datetime(2026, 8, 17, 15, 0, tzinfo=UTC)
        result = fetch_ambient_uvi(47.6, -122.33, when, http_get=_stub(SEATTLE_FIXTURE))
        assert result.uv_index_live == pytest.approx(0.75)
        assert result.uv_index_clear_sky == pytest.approx(0.80)
        assert result.cloud_modifier == pytest.approx(0.75 / 0.80)
        assert result.cloud_modifier < 1.0
        assert result.is_fallback is False

    def test_default_when_is_now(self):
        # No `when` given -> defaults to current UTC time.
        #
        # FIXED 2026-08-18. The old version stubbed the STATIC LA fixture, whose
        # hourly window is 2026-08-17T00:00..23:00. That made the test a time
        # bomb: it passed only while the wall clock sat inside that one calendar
        # day, and began failing with AmbientUVIUnavailable once real time moved
        # past it.
        #
        # The fix synthesises a response centred on the CURRENT hour so it
        # holds forever.
        now = _dt.datetime.now(_dt.timezone.utc).replace(
            minute=0, second=0, microsecond=0, tzinfo=None)
        hours = [now + _dt.timedelta(hours=h) for h in range(-6, 7)]
        payload = {
            "latitude": 34.05, "longitude": -118.25,
            "hourly_units": {"uv_index": "", "uv_index_clear_sky": ""},
            "hourly": {
                "time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
                "uv_index": [3.0] * len(hours),
                "uv_index_clear_sky": [4.0] * len(hours),
            },
        }
        result = fetch_ambient_uvi(
            34.05, -118.25, http_get=_stub(_json.dumps(payload)))
        assert isinstance(result, AmbientUVI)
        # the matched hour must be the current one, not an arbitrary fixture hour
        assert result.uv_index_live == 3.0
        assert result.cloud_modifier == 0.75


# ---------------------------------------------------------------------------
# Night-time / zero UVI: legitimate, not a failure
# ---------------------------------------------------------------------------

class TestNightTime:
    def test_night_zero_is_not_an_error(self):
        when = dt.datetime(2026, 8, 17, 4, 0, tzinfo=UTC)
        result = fetch_ambient_uvi(34.05, -118.25, when, http_get=_stub(LA_FIXTURE))
        assert result.uv_index_live == 0.0
        assert result.uv_index_clear_sky == 0.0
        assert result.effective_uv_index == 0.0
        assert result.is_fallback is False  # live WAS present, just zero

    def test_night_cloud_modifier_is_none_not_fabricated(self):
        when = dt.datetime(2026, 8, 17, 4, 0, tzinfo=UTC)
        result = fetch_ambient_uvi(34.05, -118.25, when, http_get=_stub(LA_FIXTURE))
        assert result.cloud_modifier is None
        assert result.note is not None
        assert "undefined" in result.note


# ---------------------------------------------------------------------------
# Fallback to clear-sky when live field is missing
# ---------------------------------------------------------------------------

class TestFallback:
    def _synthetic_missing_live(self):
        return json.dumps({
            "latitude": 10.0, "longitude": 20.0,
            "hourly": {
                "time": ["2026-08-17T12:00"],
                "uv_index": [None],
                "uv_index_clear_sky": [7.5],
            },
        })

    def test_falls_back_to_clear_sky_and_flags_it(self):
        when = dt.datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        result = fetch_ambient_uvi(10.0, 20.0, when,
                                   http_get=_stub(self._synthetic_missing_live()))
        assert result.uv_index_live is None
        assert result.uv_index_clear_sky == 7.5
        assert result.effective_uv_index == 7.5
        assert result.is_fallback is True
        assert result.note is not None
        assert "fell back" in result.note

    def test_both_missing_is_unavailable_not_fallback(self):
        payload = json.dumps({
            "hourly": {
                "time": ["2026-08-17T12:00"],
                "uv_index": [None],
                "uv_index_clear_sky": [None],
            },
        })
        when = dt.datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        with pytest.raises(AmbientUVIUnavailable):
            fetch_ambient_uvi(10.0, 20.0, when, http_get=_stub(payload))


# ---------------------------------------------------------------------------
# Failure handling: never substitute a guessed value
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_bad_latitude_rejected_locally_without_network(self):
        stub = _recording_stub(LA_FIXTURE)
        with pytest.raises(AmbientUVIUnavailable, match="latitude"):
            fetch_ambient_uvi(999.0, -118.25, http_get=stub)
        assert stub.calls == []  # never even tried the network

    def test_bad_longitude_rejected_locally(self):
        with pytest.raises(AmbientUVIUnavailable, match="longitude"):
            fetch_ambient_uvi(34.05, 999.0, http_get=_stub(LA_FIXTURE))

    def test_api_error_body_is_surfaced(self):
        # Simulates Open-Meteo itself rejecting an otherwise-locally-valid
        # request. ERROR_FIXTURE is the exact real response for latitude=999
        # (see module docstring), replayed here to test the response-parsing
        # path independent of local checks.
        when = dt.datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        with pytest.raises(AmbientUVIUnavailable, match="Latitude must be in range"):
            fetch_ambient_uvi(34.05, -118.25, when, http_get=_stub(ERROR_FIXTURE))

    def test_network_error_from_getter_propagates(self):
        def _broken_get(url, timeout):
            raise AmbientUVIUnavailable("simulated: network unreachable")
        with pytest.raises(AmbientUVIUnavailable, match="unreachable"):
            fetch_ambient_uvi(34.05, -118.25, http_get=_broken_get)

    def test_invalid_json_body(self):
        with pytest.raises(AmbientUVIUnavailable, match="not valid JSON"):
            fetch_ambient_uvi(34.05, -118.25, http_get=_stub("<html>not json</html>"))

    def test_missing_hourly_block(self):
        with pytest.raises(AmbientUVIUnavailable, match="hourly"):
            fetch_ambient_uvi(34.05, -118.25, http_get=_stub(json.dumps({"latitude": 1})))

    def test_missing_uv_index_field(self):
        payload = json.dumps({
            "hourly": {"time": ["2026-08-17T12:00"], "uv_index_clear_sky": [5.0]}
        })
        with pytest.raises(AmbientUVIUnavailable):
            fetch_ambient_uvi(34.05, -118.25, http_get=_stub(payload))

    def test_mismatched_array_lengths(self):
        payload = json.dumps({
            "hourly": {
                "time": ["2026-08-17T12:00", "2026-08-17T13:00"],
                "uv_index": [5.0],
                "uv_index_clear_sky": [5.0, 5.0],
            }
        })
        with pytest.raises(AmbientUVIUnavailable, match="mismatched"):
            fetch_ambient_uvi(34.05, -118.25, http_get=_stub(payload))

    def test_requested_time_far_outside_returned_window_refuses_to_guess(self):
        # LA_FIXTURE only covers 2026-08-17. Two days later, the nearest match
        # is off by ~48h, far past HOUR_MATCH_TOLERANCE_MINUTES.
        when = dt.datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
        with pytest.raises(AmbientUVIUnavailable, match="tolerance"):
            fetch_ambient_uvi(34.05, -118.25, when, http_get=_stub(LA_FIXTURE))

    def test_when_too_far_in_future_raises_value_error(self):
        when = dt.datetime.now(UTC) + dt.timedelta(days=30)
        with pytest.raises(ValueError):
            fetch_ambient_uvi(34.05, -118.25, when, http_get=_stub(LA_FIXTURE))

    def test_when_too_far_in_past_raises_value_error(self):
        when = dt.datetime.now(UTC) - dt.timedelta(days=200)
        with pytest.raises(ValueError):
            fetch_ambient_uvi(34.05, -118.25, when, http_get=_stub(LA_FIXTURE))

    def test_naive_when_is_treated_as_utc_not_rejected(self):
        # naive datetimes are assumed UTC [CHOICE], see module docstring.
        when = dt.datetime(2026, 8, 17, 20, 0)  # no tzinfo
        result = fetch_ambient_uvi(34.05, -118.25, when, http_get=_stub(LA_FIXTURE))
        assert result.timestamp == "2026-08-17T20:00"


# ---------------------------------------------------------------------------
# uvi_from_user_input: fully offline path
# ---------------------------------------------------------------------------

class TestUserInput:
    def test_basic_wrap(self):
        result = uvi_from_user_input(6.0)
        assert result.effective_uv_index == 6.0
        assert result.uv_index_live == 6.0
        assert result.uv_index_clear_sky is None
        assert result.cloud_modifier is None
        assert result.is_fallback is False
        assert result.source == "user-reported"
        assert result.timestamp == "user-supplied"

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            uvi_from_user_input(-1.0)

    def test_none_rejected(self):
        with pytest.raises(ValueError):
            uvi_from_user_input(None)

    def test_zero_is_valid(self):
        # Zero UVI (e.g. indoors, night) is a legitimate user report.
        result = uvi_from_user_input(0.0)
        assert result.effective_uv_index == 0.0

    def test_implausibly_high_is_accepted_but_flagged(self):
        result = uvi_from_user_input(30.0)
        assert result.effective_uv_index == 30.0  # not clamped
        assert result.note is not None
        assert "unusually high" in result.note

    def test_optional_location_echoed(self):
        result = uvi_from_user_input(5.0, latitude=1.0, longitude=2.0, note="from app X")
        assert result.latitude == 1.0
        assert result.longitude == 2.0
        assert "from app X" in result.note


# ---------------------------------------------------------------------------
# cloud_modifier: pure function
# ---------------------------------------------------------------------------

class TestCloudModifier:
    def test_measured_ratio_from_real_data(self):
        # Same real Seattle hour as TestFetchHappyPath.
        assert cloud_modifier(0.75, 0.80) == pytest.approx(0.75 / 0.80)

    def test_clear_day_ratio_near_one(self):
        assert cloud_modifier(8.10, 8.15) == pytest.approx(0.9939, abs=1e-3)

    def test_zero_clear_sky_raises(self):
        with pytest.raises(ValueError, match="undefined"):
            cloud_modifier(0.0, 0.0)

    def test_negative_clear_sky_raises(self):
        with pytest.raises(ValueError):
            cloud_modifier(1.0, -1.0)

    def test_none_clear_sky_raises(self):
        with pytest.raises(ValueError):
            cloud_modifier(1.0, None)

    def test_negative_uv_index_raises(self):
        with pytest.raises(ValueError):
            cloud_modifier(-1.0, 5.0)

    def test_enhancement_above_one_not_clipped(self):
        # Cloud-edge enhancement: live can exceed clear-sky.
        assert cloud_modifier(9.0, 8.0) == pytest.approx(1.125)


# ---------------------------------------------------------------------------
# Module-level constants stay what they claim to be
# ---------------------------------------------------------------------------

class TestConstants:
    def test_forecast_url(self):
        assert OPEN_METEO_FORECAST_URL == "https://api.open-meteo.com/v1/forecast"

    def test_historical_url_is_documented_and_distinct(self):
        assert OPEN_METEO_HISTORICAL_URL == (
            "https://historical-forecast-api.open-meteo.com/v1/forecast"
        )
        assert OPEN_METEO_HISTORICAL_URL != OPEN_METEO_FORECAST_URL


# ---------------------------------------------------------------------------
# AmbientUVIUnavailable carries a usable .reason
# ---------------------------------------------------------------------------

class TestExceptionShape:
    def test_reason_attribute(self):
        try:
            fetch_ambient_uvi(999.0, 0.0, http_get=_stub(LA_FIXTURE))
        except AmbientUVIUnavailable as e:
            assert isinstance(e.reason, str)
            assert len(e.reason) > 0
        else:
            pytest.fail("expected AmbientUVIUnavailable")

    def test_is_a_runtime_error(self):
        assert issubclass(AmbientUVIUnavailable, RuntimeError)
