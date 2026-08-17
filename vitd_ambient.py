"""
vitd_ambient.py
================
Ambient UV Index input path: get an erythemal UV Index for a place and time,
either by fetching a live/forecast value from Open-Meteo or by accepting a
value the caller already has (e.g. typed in from a weather app or a UV
sensor). This is the module that removes the "the caller must already know
today's UV Index" precondition that every other function in this repo
(sed_from_minutes, minutes_for_target, etc. in vitd_objectives.py) has always
had.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It does not compute a vitamin-D dose, and it does not apply any vitamin-D
weighting. It hands back the ambient erythemal UV Index and, when available,
how much of it survived the day's cloud cover. Converting that erythemal
quantity into a vitamin-D-effective one is the job of this repo's E_source
(VDE:EE ratio) machinery -- see the SOURCE_CALIBRATION_TABLE dosimetry chain
in vitd_model.py, and the solar-zenith-angle-based E_source work in
vitd_esource.py (as of this writing, being built by another agent working
from vitd_solar.py's solar-position output). Nothing here duplicates or
guesses at that weighting.

CRITICAL SCIENTIFIC CAVEAT -- erythemal, not vitamin-D, and anchored to light
skin
-----------------------------------------------------------------------------
The UV Index (UVI) is defined by the WHO/WMO/ICNIRP/UNEP "Global Solar UV
Index: A Practical Guide" (WHO, 2002) as erythemally effective irradiance
weighted by the CIE 1998 erythema reference action spectrum (McKinlay &
Diffey 1987), NOT by any vitamin-D action spectrum (e.g. CIE 2006 D-vitamin
weighting). Erythema and previtamin-D3 photosynthesis peak at different
wavelengths and have different spectral shapes, so "high UVI" and "high
vitamin-D dose rate" are correlated but NOT proportional across all
sun/sky/altitude/ozone conditions -- the ratio between them (E_source in this
repo's terms) genuinely varies by spectral composition, which is exactly why
vitd_esource.py exists as a separate module rather than this one silently
multiplying UVI by a constant.

This has already cost this repo one bogus validation benchmark: the
Kallioglu 2024 "held-out skin-type" table that a previous audit treated as a
vitamin-D result was actually its Table 3 erythemal MED values (Fitzpatrick
I=200, ..., VI=1000 J/m^2 -- an erythemal MED progression, not a vitamin-D
one) run through Hollick's rule, and the "5.00x"/"2.40x" ratios quoted from
it are literally 1000/200 and 600/250 of THAT table -- i.e. purely erythemal
arithmetic that was mistaken for a vitamin-D-weighted result. See
MODEL_AND_EVIDENCE.md for the full writeup. This module is deliberately
structured so that mixing up the two is hard to do by accident: every value
this module returns is named uv_index* (erythemal), nothing here is named or
described as a "vitamin D index", and no conversion factor is applied.

Further: the CIE erythema action spectrum that DEFINES the UV Index was
derived from, and is validated against, predominantly light (Fitzpatrick
I-III) skin erythema response data (McKinlay & Diffey 1987; the CIE
standard's own reference is unweighted by skin type but its empirical
grounding is fair-skin sunburn data). This repo is a DARK-SKIN-PRIORITY
model. The UV Index this module returns therefore inherits a light-skin
measurement lineage before it ever reaches this repo's darker-FST-focused
dose logic downstream -- the same kind of inherited-bias caveat this repo
already documents for other imported constants (see MODEL_AND_EVIDENCE.md,
"assumptions"). Nothing in THIS module corrects for that; it is flagged here
so nothing downstream forgets it exists.

UNITS
-----
The UV Index is dimensionless. Open-Meteo's own hourly_units field for both
uv_index and uv_index_clear_sky is the empty string "" (confirmed live,
2026-08-17 -- see fixtures below), consistent with the WHO UVI being a
unitless index, not a physical irradiance. It is standardly defined such
that 1 UVI unit = 25 mW/m^2 of erythemally-weighted irradiance (WHO 2002)
-- this is the SAME conversion this repo already uses in
vitd_objectives.SED_PER_MIN_PER_UVI (0.015 SED per minute per UVI unit; see
that module for the full derivation: 1 UVI = 25 mW/m^2 erythemal, 1 SED =
100 J/m^2 erythemal, so SED/min = UVI * 25e-3 * 60 / 100). This module does
NOT redefine, re-derive, or duplicate that constant -- it only produces the
UVI number that constant is meant to consume. Do not change
SED_PER_MIN_PER_UVI here; it is owned by vitd_objectives.py.

The UV Index is also, by convention, a solar-noon-referenced daily-style
quantity in its classic (satellite/TEMIS/ozone-model) form, but Open-Meteo
reports it as an hourly-resolved forecast (see FIELDS below) -- i.e. what this
module fetches is the instantaneous erythemal UVI for the requested clock
hour, not a single daily maximum. That is actually what a minutes-of-sun
calculation needs (the UVI at the time someone is actually outside), not the
daily peak, so this is a feature of using Open-Meteo's hourly product rather
than a mismatch to work around.

DESIGN: LIVE/FORECAST IS THE TRUNK, CLEAR-SKY IS ONLY A FALLBACK
------------------------------------------------------------------
Open-Meteo's forecast endpoint returns BOTH `uv_index` (its actual,
cloud/aerosol-aware forecast -- "all-sky") and `uv_index_clear_sky` (what the
same model would predict with no clouds) in the SAME response. That means:
  - the primary, returned value (`effective_uv_index` on AmbientUVI, sourced
    from `uv_index_live`) is the all-sky forecast -- already better than any
    clear-sky astronomical model for someone standing outside today, and it
    costs nothing extra to get since it is the same API call;
  - `uv_index_clear_sky` is used ONLY as a documented fallback, when the live
    field is missing from a particular hour (which does happen -- Open-Meteo
    is a live forecast product and fields can be sparse/absent at the edges
    of its update cycle) -- fetch_ambient_uvi() falls back to it and marks
    `is_fallback=True` with a `note` explaining why, rather than silently
    returning a number indistinguishable from a normal live reading;
  - the ratio of the two, computed by cloud_modifier(), is the MEASURED cloud
    attenuation for that hour -- not an assumed constant. This module never
    invents its own clear-sky irradiance model (e.g. from solar zenith angle
    via vitd_solar.py) to use as the fallback; Open-Meteo already supplies
    one for free in the same call, and building a second, independent
    clear-sky model here would just be two unvalidated guesses instead of
    one measured value plus one clearly-labelled fallback field.

SOURCE
------
Open-Meteo Forecast API, https://api.open-meteo.com/v1/forecast -- chosen
because it needs no API key, is global, and returns both `uv_index` and
`uv_index_clear_sky` per hour in one response (confirmed against
https://open-meteo.com/en/docs, fetched 2026-08-17). It also offers a
separate Historical Forecast API at
https://historical-forecast-api.open-meteo.com/v1/forecast ("The API is
identical to the Forecast API"; per https://open-meteo.com/en/docs/historical-forecast-api,
fetched 2026-08-17, coverage starts around 2022) for backtesting model
output against recorded ambient UV -- this module exposes its base URL as
OPEN_METEO_HISTORICAL_URL for that future use, but does not implement a
dedicated backtesting function here; that is out of this module's scope
(input path only).

FIELDS AND UNITS USED (from https://open-meteo.com/en/docs, fetched
2026-08-17, and cross-checked against a live call on the same date -- see
tests/fixtures/):
  hourly=uv_index,uv_index_clear_sky
    uv_index            -- forecast UV Index, all-sky (cloud/aerosol-aware).
                            hourly_units.uv_index == "" (dimensionless).
    uv_index_clear_sky  -- forecast UV Index under a clear-sky assumption.
                            hourly_units.uv_index_clear_sky == "" (dimensionless).
  hourly.time           -- ISO 8601 timestamps, one per hour. Per Open-Meteo
                            docs: "Time always starts at 0:00 today and
                            contains 168 hours" (i.e. anchored to the
                            requested timezone's midnight), confirmed live.
  timezone=UTC           -- [CHOICE] this module always requests UTC
                            explicitly rather than "auto" or a local IANA
                            name, so that hourly.time strings are directly
                            comparable to a caller-supplied UTC datetime with
                            no separate DST/offset bookkeeping. The cost is
                            that this module does not need, and does not
                            attempt, local-clock alignment (e.g. "solar
                            noon" in local time) -- pass an explicit UTC
                            `when` if that alignment matters to the caller.
  past_days, forecast_days -- integers Open-Meteo uses to widen the returned
                            window around "today" so that a requested `when`
                            in the near past or future is actually inside
                            the returned hourly.time array. Documented
                            defaults/maxima (fetched 2026-08-17):
                            past_days: default 0, max 92.
                            forecast_days: default 7, max 16.
  Error responses          -- on invalid input (e.g. out-of-range latitude),
                            Open-Meteo returns HTTP 400 with a JSON body
                            {"error": true, "reason": "<message>"} (confirmed
                            live 2026-08-17 with latitude=999 ->
                            {"reason":"Latitude must be in range of -90 to
                            90°. Given: 999.0.","error":true}; see
                            tests/fixtures/open_meteo_error_bad_latitude.json).

No rate-limit or attribution requirement for this use pattern is stated on
the fetched docs page; none is assumed or enforced here.

TESTING WITHOUT NETWORK
------------------------
fetch_ambient_uvi() takes an injectable `http_get(url, timeout) -> str`
callable. Production code omits it and gets the real
urllib.request-backed implementation (_default_http_get); tests/test_ambient.py
supplies a stub that returns the text of a recorded fixture, so the entire
test suite runs with zero network access. Fixtures in tests/fixtures/ were
captured live via `curl` against the real API on 2026-08-17 (see each
fixture's paired comment in test_ambient.py for the exact request URL used
to capture it) -- they are real recorded responses, not hand-written.

FAILURE HANDLING
------------------
Every failure mode raises AmbientUVIUnavailable with a human-readable
`.reason` rather than returning a plausible-looking-but-wrong number:
  - network unreachable / timeout          -> AmbientUVIUnavailable
  - Open-Meteo's own {"error": true, ...}  -> AmbientUVIUnavailable(reason=...)
  - malformed/missing fields in a 200 body -> AmbientUVIUnavailable
  - requested `when` outside the returned hourly window (by more than
    HOUR_MATCH_TOLERANCE) -> AmbientUVIUnavailable
  - requested `when` beyond Open-Meteo's documented past_days/forecast_days
    limits -> ValueError (a caller programming error, not a runtime failure)
Night-time / zero UVI is explicitly NOT a failure: uv_index == 0.0 at night
is a correct, legitimate reading and is returned normally. What IS refused
is treating a 0/0 clear-sky ratio (night, when uv_index_clear_sky is also
0) as a cloud_modifier -- that ratio is mathematically undefined, and
cloud_modifier() raises ValueError rather than returning a fabricated 1.0 or
0.0; fetch_ambient_uvi() catches that internally and records
cloud_modifier=None with an explanatory note instead of propagating an
exception for what is, from the caller's point of view, an unremarkable
night reading.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
# [CHOICE] Forecast endpoint is the trunk (live/forecast UVI). See module
# docstring "DESIGN" section for why this, not a clear-sky model, is primary.
OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"

# Historical archive endpoint, same field names/semantics ("The API is
# identical to the Forecast API" -- https://open-meteo.com/en/docs/historical-forecast-api,
# fetched 2026-08-17). Exposed for future backtesting work; not called by
# any function in this module.
OPEN_METEO_HISTORICAL_URL: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Open-Meteo documented limits (https://open-meteo.com/en/docs, fetched
# 2026-08-17). Used only to fail fast with a clear ValueError instead of
# sending a request Open-Meteo would reject anyway.
_PAST_DAYS_MAX: int = 92
_FORECAST_DAYS_MAX: int = 16

# [CHOICE] How far the nearest available hourly timestamp is allowed to sit
# from the caller's requested `when` before we refuse to answer rather than
# silently substituting a different hour's UVI. Open-Meteo's hourly grid is
# exactly on the hour, so a well-formed request should match within half an
# hour; 90 minutes gives slack for the boundary hours of the requested
# past_days/forecast_days window without being loose enough to, say, hand
# back yesterday's midday UVI for a request nominally "in range" but at the
# window's edge.
HOUR_MATCH_TOLERANCE_MINUTES: float = 90.0


class AmbientUVIUnavailable(RuntimeError):
    """Raised whenever an ambient UVI cannot be honestly produced.

    Deliberately a single exception type covering network failure, API-side
    errors, malformed responses, and out-of-window timestamps: callers of a
    minutes-of-sun recommendation need exactly one thing from a failure --
    "do not use this number" -- not a taxonomy to pattern-match on. The
    `.reason` attribute carries the specific cause for logging/debugging.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class AmbientUVI:
    """A single ambient-UVI observation, from either source in this module.

    `effective_uv_index` is what a caller should feed into UVI-consuming
    code (e.g. vitd_objectives.sed_from_minutes' `uv_index` argument) --
    NOT a vitamin-D dose; see module docstring. It is erythemally weighted,
    never vitamin-D weighted, regardless of source.
    """

    latitude: Optional[float]
    longitude: Optional[float]
    # ISO 8601 UTC timestamp this observation applies to, or a fixed literal
    # "user-supplied" string when there is no associated clock time.
    timestamp: str
    # Open-Meteo's own all-sky forecast for this hour, if this observation
    # came from a fetch and the field was present. None for user-supplied
    # readings (we were not told a clear-sky counterpart either).
    uv_index_live: Optional[float]
    # Open-Meteo's clear-sky forecast for this hour, if available.
    uv_index_clear_sky: Optional[float]
    # uv_index_live / uv_index_clear_sky when both are present and the
    # clear-sky value is nonzero; None when undefined (night, or
    # user-supplied readings where no clear-sky counterpart exists).
    cloud_modifier: Optional[float]
    # The value downstream code should use.
    effective_uv_index: float
    # True when effective_uv_index came from uv_index_clear_sky because the
    # live field was missing for this hour -- i.e. the fallback path fired.
    is_fallback: bool
    # "open-meteo-forecast" or "user-reported".
    source: str
    # Human-readable explanation, populated whenever something non-default
    # happened (fallback used, cloud_modifier undefined, etc.); None when
    # this is a plain, unremarkable live reading.
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# 1. Network fetch (the trunk path)
# ---------------------------------------------------------------------------

def _default_http_get(url: str, timeout: float) -> str:
    """Real HTTP GET, used when fetch_ambient_uvi() is not given a stub.

    Not exercised by tests/test_ambient.py (that suite injects a fake
    `http_get` so it never touches the network); this function exists so
    production callers get real data without having to know about the
    injection seam.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        # Open-Meteo returns a JSON error BODY with a 4xx status (confirmed
        # live for bad latitude -- see module docstring), so read it rather
        # than discarding it: the caller-facing reason comes from that body.
        try:
            return e.read().decode("utf-8")
        except Exception:
            raise AmbientUVIUnavailable(
                f"Open-Meteo request failed (HTTP {e.code}) with no readable body"
            )
    except urllib.error.URLError as e:
        raise AmbientUVIUnavailable(f"network error reaching Open-Meteo: {e.reason}")
    except TimeoutError:
        raise AmbientUVIUnavailable(f"Open-Meteo request timed out after {timeout}s")


def _build_forecast_url(latitude: float, longitude: float,
                        past_days: int, forecast_days: int,
                        base_url: str) -> str:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "uv_index,uv_index_clear_sky",
        "timezone": "UTC",  # [CHOICE] see module docstring FIELDS section
    }
    if past_days:
        params["past_days"] = past_days
    if forecast_days:
        params["forecast_days"] = forecast_days
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _nearest_hour(times: list[str], target: dt.datetime) -> tuple[int, float]:
    """Index into `times` nearest `target`, and the gap in minutes.

    `times` are Open-Meteo's ISO 8601 UTC hourly timestamps (no offset
    suffix, since we always request timezone=UTC -- see module docstring).
    """
    best_idx = -1
    best_gap = math.inf
    for i, t in enumerate(times):
        candidate = dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc)
        gap = abs((candidate - target).total_seconds()) / 60.0
        if gap < best_gap:
            best_gap = gap
            best_idx = i
    return best_idx, best_gap


def fetch_ambient_uvi(latitude: float, longitude: float,
                       when: Optional[dt.datetime] = None, *,
                       timeout: float = 10.0,
                       http_get: Optional[Callable[[str, float], str]] = None,
                       base_url: str = OPEN_METEO_FORECAST_URL) -> AmbientUVI:
    """Fetch the ambient (erythemal) UV Index for a place and time.

    Network call (unless `http_get` is injected for testing -- see module
    docstring "TESTING WITHOUT NETWORK"). Returns the all-sky forecast UVI as
    `effective_uv_index`, falling back to the clear-sky forecast (with
    `is_fallback=True`) only if the live field is absent for the matched
    hour. Never returns a guessed number: any failure raises
    AmbientUVIUnavailable (see module docstring "FAILURE HANDLING").

    `when` defaults to now (UTC). Must be timezone-aware if given, or naive
    and already-UTC -- naive datetimes are assumed UTC [CHOICE], since this
    module has no way to know a caller's intended zone and silently
    assuming local-machine time would be a worse guess.
    """
    if latitude is None or longitude is None:
        raise AmbientUVIUnavailable("latitude and longitude are required")
    if not (-90.0 <= latitude <= 90.0):
        raise AmbientUVIUnavailable(
            f"latitude {latitude} out of range [-90, 90] (checked locally; "
            f"Open-Meteo would also reject this)"
        )
    if not (-180.0 <= longitude <= 180.0):
        raise AmbientUVIUnavailable(
            f"longitude {longitude} out of range [-180, 180] (checked locally; "
            f"Open-Meteo would also reject this)"
        )

    now = dt.datetime.now(dt.timezone.utc)
    if when is None:
        when = now
    elif when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)  # [CHOICE] see docstring above
    else:
        when = when.astimezone(dt.timezone.utc)

    delta_hours = (when - now).total_seconds() / 3600.0
    if delta_hours >= 0:
        forecast_days = max(1, math.ceil(delta_hours / 24.0) + 1)
        past_days = 0
        if forecast_days > _FORECAST_DAYS_MAX:
            raise ValueError(
                f"`when` is {delta_hours/24:.1f} days in the future; Open-Meteo's "
                f"forecast API supports at most {_FORECAST_DAYS_MAX} forecast_days"
            )
    else:
        past_days = max(1, math.ceil(-delta_hours / 24.0) + 1)
        forecast_days = 1
        if past_days > _PAST_DAYS_MAX:
            raise ValueError(
                f"`when` is {-delta_hours/24:.1f} days in the past; Open-Meteo's "
                f"forecast API supports at most {_PAST_DAYS_MAX} past_days"
            )

    url = _build_forecast_url(latitude, longitude, past_days, forecast_days, base_url)
    getter = http_get or _default_http_get
    raw = getter(url, timeout)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise AmbientUVIUnavailable(f"Open-Meteo response was not valid JSON: {raw[:200]!r}")

    if isinstance(payload, dict) and payload.get("error"):
        raise AmbientUVIUnavailable(
            f"Open-Meteo reported an error: {payload.get('reason', '(no reason given)')}"
        )

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise AmbientUVIUnavailable("Open-Meteo response missing 'hourly' block")

    times = hourly.get("time")
    live_series = hourly.get("uv_index")
    clear_series = hourly.get("uv_index_clear_sky")
    if not times or live_series is None or clear_series is None:
        raise AmbientUVIUnavailable(
            "Open-Meteo response missing 'time', 'uv_index', or 'uv_index_clear_sky' "
            "in the hourly block"
        )
    if not (len(times) == len(live_series) == len(clear_series)):
        raise AmbientUVIUnavailable(
            "Open-Meteo hourly arrays have mismatched lengths -- malformed response"
        )

    idx, gap_minutes = _nearest_hour(times, when)
    if idx < 0:
        raise AmbientUVIUnavailable("Open-Meteo response had an empty hourly.time array")
    if gap_minutes > HOUR_MATCH_TOLERANCE_MINUTES:
        raise AmbientUVIUnavailable(
            f"nearest available hour ({times[idx]}) is {gap_minutes:.0f} min from "
            f"the requested time -- outside the {HOUR_MATCH_TOLERANCE_MINUTES:.0f} min "
            f"tolerance; refusing to substitute a different hour's UVI"
        )

    live = live_series[idx]
    clear = clear_series[idx]
    live = float(live) if live is not None else None
    clear = float(clear) if clear is not None else None

    note = None
    is_fallback = False
    if live is not None:
        effective = live
    elif clear is not None:
        effective = clear
        is_fallback = True
        note = (
            f"live uv_index missing for {times[idx]}; fell back to "
            f"uv_index_clear_sky ({clear}) -- this OVERSTATES actual UVI under "
            f"any cloud cover, since the clear-sky value has no cloud attenuation"
        )
    else:
        raise AmbientUVIUnavailable(
            f"both uv_index and uv_index_clear_sky are null for {times[idx]}"
        )

    modifier = None
    if live is not None and clear is not None:
        try:
            modifier = cloud_modifier(live, clear)
        except ValueError as e:
            note = str(e) if note is None else f"{note}; {e}"

    return AmbientUVI(
        latitude=latitude,
        longitude=longitude,
        timestamp=times[idx],
        uv_index_live=live,
        uv_index_clear_sky=clear,
        cloud_modifier=modifier,
        effective_uv_index=effective,
        is_fallback=is_fallback,
        source="open-meteo-forecast",
        note=note,
    )


# ---------------------------------------------------------------------------
# 2. No-network, user-supplied UVI
# ---------------------------------------------------------------------------

def uvi_from_user_input(uv_index: float, *,
                        latitude: Optional[float] = None,
                        longitude: Optional[float] = None,
                        note: Optional[str] = None) -> AmbientUVI:
    """Wrap a caller-supplied UV Index into the same AmbientUVI shape.

    No network access. This is the path for "I looked at my phone's weather
    app / a UV sensor and the UVI is X" -- the model should be able to run
    entirely offline from a manually entered number, which this function
    exists to guarantee (it does nothing but validate and wrap).

    Raises ValueError -- NOT AmbientUVIUnavailable -- for invalid input,
    since this is a caller programming/data-entry error, not a runtime
    service failure; the two exception types are kept distinct so callers
    can tell "you gave me a bad number" apart from "I couldn't get a
    number".
    """
    if uv_index is None:
        raise ValueError("uv_index is required")
    uv_index = float(uv_index)
    if uv_index < 0:
        raise ValueError(f"uv_index cannot be negative, got {uv_index}")
    # [ESTIMATE] 25 is used only as a sanity-check ceiling, not a hard limit:
    # the highest UV Index values ever measured/reported (high-altitude
    # tropics, e.g. Bolivian Altiplano studies) are in the low-to-mid 20s.
    # We do not reject values above this -- a real sensor reading should
    # never be second-guessed and silently discarded -- we only attach a
    # note so a downstream consumer/human can sanity-check an unusual input.
    if uv_index > 25:
        note = (
            f"uv_index {uv_index} is unusually high (highest ever recorded UVI "
            f"values are in the low-to-mid 20s); accepted as given, not clamped"
            + (f"; {note}" if note else "")
        )

    return AmbientUVI(
        latitude=latitude,
        longitude=longitude,
        timestamp="user-supplied",
        uv_index_live=uv_index,
        uv_index_clear_sky=None,
        cloud_modifier=None,
        effective_uv_index=uv_index,
        is_fallback=False,
        source="user-reported",
        note=note,
    )


# ---------------------------------------------------------------------------
# 3. Cloud modifier -- measured, not assumed
# ---------------------------------------------------------------------------

def cloud_modifier(uv_index: float, uv_index_clear_sky: float) -> float:
    """uv_index / uv_index_clear_sky -- the MEASURED cloud/aerosol attenuation.

    This is the entire reason Open-Meteo was chosen over a source that only
    reports one UVI number: with both the all-sky and clear-sky forecast for
    the same hour, this ratio is read off real model output instead of being
    assumed from a generic "cloud cover %" heuristic.

    Raises ValueError if uv_index_clear_sky <= 0 (undefined -- typically
    night, when both values are legitimately 0 and there is nothing to take
    a ratio of). fetch_ambient_uvi() catches this internally and records
    cloud_modifier=None with a note rather than treating it as a fetch
    failure; call this function directly and you get the exception.

    Values above 1.0 are NOT clipped. [CHOICE] Cloud-edge/broken-cloud UV
    enhancement above the clear-sky value is a real, observed phenomenon in
    ground-based UV monitoring (reflection/scattering off nearby cloud can
    briefly exceed the clear-sky irradiance); silently capping the ratio at
    1.0 would discard a real measured signal in favour of an assumption this
    module has no basis for.
    """
    if uv_index_clear_sky is None or uv_index_clear_sky <= 0:
        raise ValueError(
            f"cloud_modifier undefined: uv_index_clear_sky={uv_index_clear_sky!r} "
            f"is zero, negative, or missing (typically night)"
        )
    if uv_index is None or uv_index < 0:
        raise ValueError(f"uv_index must be non-negative, got {uv_index!r}")
    return uv_index / uv_index_clear_sky
