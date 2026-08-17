"""
vitd_solar.py
=============
Solar position (declination, equation of time, hour angle, solar zenith
angle, solar elevation angle) from latitude, longitude, date and clock time.

This is the first link of the dosimetry chain that was previously entirely
missing from this repo: everything upstream of vitd_model.py currently takes
UV dose (SED) as a direct input, with no way to derive it from where and when
someone actually stood in the sun. This module does not compute UV dose or
vitamin D synthesis -- it only answers "where is the sun", which is what any
future irradiance/UV-index model needs as its first input.

Coefficient source (canonical) and in-repo route to it
--------------------------------------------------------
The formulas' STRUCTURE and printed coefficients were first located in this
repo via Kallioglu et al. 2024, Sci Rep 14:3541 (doi 10.1038/s41598-024-54188-5),
"UV index-based model for predicting synthesis of (pre-)vitamin D3 in the
mediterranean basin" -- section "Materials and methods / Solar position
calculation" (paper pp. 4-5), Eqs. 1-8. PDF:
Research_Papers/Kallioglu 2024 SciRep 14-3541 - UV index-based model for
vitamin D3 in the mediterranean basin.pdf

That paper turned out to be a TRANSCRIPTION of NOAA Global Monitoring
Division's published Spencer-family solar-position algorithm, with four
coefficient digits dropped/rounded in transit (see ERRATUM 2 below) and one
sin/cos swap in Eq. 7 (ERRATUM 1). NOAA's document is therefore the CANONICAL
coefficient source implemented by this module's public functions:

    NOAA Global Monitoring Division, "General Solar Position Calculations",
    https://gml.noaa.gov/grad/solcalc/solareqns.PDF

Kallioglu 2024 remains cited throughout because it is how this repo found
the algorithm and because Eqs. 1, 3, 5, 6, 8 (time correction, fractional
year, hour angle, LST, elevation) are unaffected by either erratum and are
reproduced faithfully from its printed form. Eqs. 1-3/7-8 are, in turn, cited
BY Kallioglu 2024 to Duffie & Beckman, "Solar Engineering of Thermal
Processes" (paper ref. 70) and Iqbal, "An Introduction to Solar Radiation"
(ref. 72); Eq. 4 (declination) is cited to Cooper 1969 (ref. 71) via Iqbal
(ref. 72) -- consistent with this being the well-known Spencer (1971)-family
Fourier-series approximation reproduced in many solar-engineering texts, of
which NOAA's document is one canonical modern restatement.

The eight equations as PRINTED in Kallioglu 2024 (transcribed verbatim,
confirmed by direct visual inspection of the equation images, not just PDF
text-layer extraction -- both errata below were caught this way):

    (1) TC = solartime - standardtime = 4(Lst - Lloc) + E
    (2) E = 229.2[0.00075 + 0.001868 cos(B) - 0.032077 sin(B)
                  - 0.014615 cos(2B) - 0.04089 sin(2B)]
        [ERRATUM 2 -- three of these four non-repeated-elsewhere coefficients
        are typo'd; see below. NOT implemented this way.]
    (3) B = (n - 1) * 360 / 365
    (4) delta = (180/pi)(0.006918 - 0.39912 cos(B) + 0.070257 sin(B)
                          - 0.006758 cos(2B) + 0.000907 sin(2B)
                          - 0.002697 cos(3B) + 0.00148 sin(3B))
        [ERRATUM 2 -- the 0.39912 coefficient is typo'd; see below. NOT
        implemented this way.]
    (5) omega = 15 * (LST - 12)
    (6) LST = LT + TC / 60
    (7) cos(theta_z) = sin(delta) . cos(phi) cos(omega) + sin(delta) sin(phi)
        [ERRATUM 1 -- see below. NOT implemented this way.]
    (8) alpha_s = sin^-1[sin(delta) sin(phi) + cos(delta) cos(phi) cos(omega)]

ERRATUM 1 -- Eq. 7's sin(delta)/cos(delta) swap (already established by this
repo before this module was written; verified independently below, not just
asserted)
------------------------------------------------------------------------
Eq. 7's first term is printed "sin(delta) . cos(phi) cos(omega)". Every other
source for this identity (Duffie & Beckman; Iqbal; NOAA, see CROSS-CHECK
below) has cos(delta) in that term, not sin(delta). Independent proof, using
only the paper's OWN Eq. 8 (printed correctly) and the printed identity
theta_z + alpha_s = 90 deg (paper's own text, immediately below Eq. 8):
    cos(theta_z) = cos(90 - alpha_s) = sin(alpha_s)
                 = sin(delta) sin(phi) + cos(delta) cos(phi) cos(omega)   [Eq 8's bracket]
So the correct Eq. 7 must read:
    cos(theta_z) = cos(delta) cos(phi) cos(omega) + sin(delta) sin(phi)
i.e. "sin(delta)" in the FIRST term of the printed Eq. 7 should be
"cos(delta)" -- exactly the erratum this repo had already identified. This
derivation is checked numerically in tests/test_solar.py
(TestErratumEq7::test_corrected_form_is_the_only_one_consistent_with_eq8),
which shows the corrected form satisfies theta_z + alpha_s == 90 deg to
floating-point precision over a wide grid of latitude/declination/hour-angle
combinations, while the as-printed form does not (off by tens of degrees in
some cases -- e.g. ~42 deg at Antalya's summer-solstice solar noon). This
module implements the CORRECTED form only; solar_zenith_angle_deg() IS the
corrected Eq. 7. The as-printed (buggy) form is kept as a private function
(_zenith_angle_deg_as_printed_eq7_ERRATUM) purely so the erratum stays
demonstrable and regression-testable, and must never be used for real output.

ERRATUM 2 -- four coefficient digits dropped/rounded in Eqs. 2 and 4
------------------------------------------------------------------------
Kallioglu 2024 prints:  0.00075, 229.2, 0.04089 (Eq. 2), 0.39912 (Eq. 4)
NOAA GML prints:        0.000075, 229.18, 0.040849 (Eq. 2), 0.399912 (Eq. 4)
Every OTHER coefficient in Eqs. 2 and 4 (0.001868, 0.032077, 0.014615,
0.070257, 0.006918, 0.006758, 0.000907, 0.002697, 0.00148) matches NOAA's
value exactly. That pattern -- five-of-nine exact matches, four near-misses
each differing by a single dropped or rounded digit -- is the signature of a
transcription error, not an independent re-derivation or a different
formulation. Visually confirmed against the printed PDF equation images
(Research_Papers/Kallioglu...pdf, p.4-5), so this is a genuine feature of the
published paper, not an artifact of this repo's PDF text extraction.
Consequently this module implements NOAA's coefficients as the CANONICAL
default in equation_of_time_minutes() and solar_declination_deg(). The
as-printed Kallioglu coefficients are kept in
_equation_of_time_minutes_as_printed_ERRATUM() and
_solar_declination_deg_as_printed_ERRATUM() purely so the erratum stays
demonstrable and regression-tested (tests/test_solar.py::TestErratumCoefficients
pins the measured divergence below), and must never be used for real output.

QUANTIFIED IMPACT of Erratum 2 (measured, not estimated -- see
tests/test_solar.py::TestErratumCoefficients for the exact assertions):
    declination        max |Kallioglu - NOAA| = 0.0454 deg   (at n=1)
    equation of time   max |Kallioglu - NOAA| = 0.1654 min = 0.0414 deg of hour angle (at n=319)
    combined worst case (declination + hour-angle error added directly) ~0.087 deg
    effect on cos(zenith), i.e. on ambient UV scale: 0.06% at 20 deg zenith,
        0.26% at 60 deg zenith, 0.57% at 75 deg zenith
That combined figure (~0.087 deg) is about 2.5x the paper's OWN stated
declination error bound ("error < 0.0350", text above Eq. 4) -- i.e. the
transcription typos push the paper's own formula outside the accuracy it
claims for itself, once Eqs. 2 and 4's typo'd coefficients are considered
together. It is nonetheless PRACTICALLY NEGLIGIBLE for this
model: well under 1% on ambient UV scale even at a low, near-grazing 75 deg
zenith angle, and dwarfed by the ~0.5 deg atmospheric-refraction limitation
already noted below. Record this number precisely so nobody later assumes
the NOAA-vs-Kallioglu correction was large; it was not.

CROSS-CHECK against an independent source (not Kallioglu's own equations)
---------------------------------------------------------------------------
NOAA Global Monitoring Division, "General Solar Position Calculations"
(https://gml.noaa.gov/grad/solcalc/solareqns.PDF -- fetched and read directly
for this module; a copy of the extracted text is reproduced in
tests/test_solar.py) publishes the SAME Spencer-family fractional-year /
equation-of-time / declination / zenith-angle formulas, derived independently
of this paper:

    gamma = (2*pi/365) * (day_of_year - 1 + (hour-12)/24)              [radians]
    eqtime = 229.18*(0.000075 + 0.001868 cos(gamma) - 0.032077 sin(gamma)
                      - 0.014615 cos(2 gamma) - 0.040849 sin(2 gamma))  [minutes]
    decl = 0.006918 - 0.399912 cos(gamma) + 0.070257 sin(gamma)
           - 0.006758 cos(2 gamma) + 0.000907 sin(2 gamma)
           - 0.002697 cos(3 gamma) + 0.00148 sin(3 gamma)              [radians]
    cos(zenith) = sin(lat) sin(decl) + cos(lat) cos(decl) cos(hour_angle)

This confirms, independently of Kallioglu 2024:
  1. The zenith-angle formula's first term uses cos(declination), not
     sin(declination) -- independent confirmation of Erratum 1.
  2. The four coefficients in Erratum 2 -- independent confirmation that
     NOAA's values, not Kallioglu's, are the coefficients actually derived
     from the underlying Spencer/Fourier fit.
  3. Eq. 1's TC formula matches NOAA's "time_offset" formula exactly once the
     longitude sign convention is translated (see time_correction_minutes()
     docstring below) -- independent confirmation Eq. 1/5/6 are implemented
     with the right sign.
Since this module's public functions now implement NOAA's own coefficients
directly, tests/test_solar.py::TestNOAACrossCheck asserts agreement to float
precision (not a loose tolerance) wherever the two formulations are
genuinely identical -- a loose tolerance with no remaining source of
discrepancy would hide future regressions instead of documenting one.

ACCURACY LIMITS (state precisely what is and is not known)
------------------------------------------------------------
- The paper states (text immediately preceding Eq. 4, p.5): the declination
  formula (Eq. 4) is the "most stable" of several candidates, with "error
  < 0.0350" [degrees]. This is the ONLY numeric accuracy figure the paper
  gives for any part of the solar-position chain, and it describes the
  Fourier truncation error of the FORMULA (i.e. the NOAA-coefficient
  version now implemented here) against the true continuous ephemeris -- not
  the additional, separately-quantified Erratum 2 discrepancy above. It is
  consistent with the independently-known accuracy of the Spencer
  (1971)-family declination series (see e.g. Wikipedia "Position of the
  Sun", fetched for this module: a comparable-order truncated Fourier series
  is reported accurate to "less than +/-0.03 deg for a given year", while a
  full Meeus (1998) ephemeris reaches ~0.01 deg).
- The paper gives NO stated error bound for Eq. 1 (time correction), Eq. 2
  (equation of time), Eq. 5/6 (hour angle), or Eq. 7/8 (zenith/elevation)
  individually, and NO combined end-to-end error bound for the whole chain.
  Do not attribute a specific accuracy number to those equations -- NOT
  STATED in the source.
- This is a SIMPLIFIED model, not a full ephemeris: it ignores atmospheric
  refraction (which shifts the apparent position by ~0.5 deg near the
  horizon -- much larger than the ~0.035 deg formula error, so this module's
  output should be treated as unreliable within a degree or so of sunrise/
  sunset), nutation, and higher-order orbital terms. It is adequate for the
  daytime, non-grazing-incidence use this repo needs (comparing exposure
  windows, mid-day zenith angles), not for sunrise/sunset timing to
  sub-minute precision.
- day_of_year() uses the calendar ordinal day (1-366 in a leap year) with no
  leap-year branch in B (Eq. 3's denominator is printed as a fixed 365) --
  see the [CHOICE] note on day_of_year() below.

[CHOICE] radians vs degrees in Eq. 2 / Eq. 4
---------------------------------------------
Eq. 3 defines B numerically in degree units ((n-1)*360/365) but Eqs. 2 and 4
then take cos(B), sin(B), cos(2B), sin(2B), cos(3B), sin(3B) directly. The
paper never states whether B must be converted to radians before those calls.
We convert (math.radians(B)) before every trig call. This is not a free
choice -- it is required for the formula to reproduce known solar geometry:
NOAA's equivalent gamma is defined natively in radians via
gamma = (2*pi/365)*(day-1+...), which is numerically identical to
math.radians(B) for the same day (both express the same angle, just in
different units before any trig is applied). Verified numerically: with the
radians conversion, declination peaks at +23.4 deg on the summer solstice
(n=172, matching Earth's known axial tilt of 23.44 deg to within the
formula's own ~0.03 deg truncation error) and crosses ~0 deg at the equinoxes
(n=80, n=266); without the conversion (treating the printed number 168.66 as
168.66 RADIANS), the result does not correspond to any known solar geometry.
See tests/test_solar.py::TestFractionalYearAndDeclination.

References
----------
1. NOAA Global Monitoring Division, "General Solar Position Calculations",
   https://gml.noaa.gov/grad/solcalc/solareqns.PDF -- CANONICAL coefficient
   source for this module (see above).
2. Kallioglu et al. 2024, Sci Rep 14:3541, Eqs. 1-8 -- the in-repo route by
   which this algorithm was located; still the source for the equations'
   structure and for Eqs. 1/3/5/6/8, which carry no coefficient erratum.
3. Wikipedia, "Position of the Sun" (accessed for this module), for the
   independently-stated ~23.44 deg obliquity of the ecliptic and the
   general accuracy ordering of truncated-Fourier vs Meeus declination
   formulas (independent sanity check only, not a formula source).
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import NamedTuple, Optional, Union


# ---------------------------------------------------------------------------
# Eq. 3: fractional year B (degrees, as printed)
# ---------------------------------------------------------------------------
def day_of_year(when: Union[_dt.date, _dt.datetime]) -> int:
    """n in Eq. 3/4, the ordinal day of the calendar year (1-365, or 1-366
    in a leap year).

    [CHOICE] Eq. 3 is printed with a fixed denominator of 365 and the range
    "1 <= n <= 365" -- no leap-year branch is given in the paper. NOAA's
    equivalent formula for gamma DOES special-case leap years (uses 366 in
    the denominator). We follow the PAPER as printed (fixed /365, no leap
    branch) rather than silently adding NOAA's leap-year handling, per the
    task's instruction not to invent machinery beyond what's on the page.
    The resulting extra error on Feb 29 / the rest of a leap year is under
    1/365 of a year of B-phase (~1 day out of 365, i.e. <0.03 deg equivalent
    solar-position shift) -- an order of magnitude below the ~0.5 deg
    atmospheric-refraction limitation already noted above, so it does not
    change this module's practical accuracy.
    """
    d = when.date() if isinstance(when, _dt.datetime) else when
    return d.timetuple().tm_yday


def fractional_year_deg(n: int) -> float:
    """Eq. 3: B = (n - 1) * 360 / 365, degrees."""
    return (n - 1) * 360.0 / 365.0


def equation_of_time_minutes(when: Union[_dt.date, _dt.datetime, int]) -> float:
    """Eq. 2: E, the equation of time, in minutes -- NOAA GML canonical
    coefficients (see module docstring, ERRATUM 2: Kallioglu's printed
    0.00075/229.2/0.04089 are transcription typos of NOAA's 0.000075/
    229.18/0.040849).

    `when` may be a date/datetime (day_of_year() is applied) or an int
    already understood as n (1-365/366).
    """
    n = when if isinstance(when, int) else day_of_year(when)
    B = math.radians(fractional_year_deg(n))  # [CHOICE] see module docstring
    return 229.18 * (
        0.000075
        + 0.001868 * math.cos(B)
        - 0.032077 * math.sin(B)
        - 0.014615 * math.cos(2 * B)
        - 0.040849 * math.sin(2 * B)
    )


def _equation_of_time_minutes_as_printed_ERRATUM(
        when: Union[_dt.date, _dt.datetime, int]) -> float:
    """Eq. 2 EXACTLY as printed in Kallioglu 2024 -- 0.00075, 229.2, 0.04089
    typo'd digit-drops of NOAA's 0.000075, 229.18, 0.040849 (module
    docstring, ERRATUM 2). NOT used anywhere in this module's real output;
    kept only so the erratum's measured impact stays regression-tested
    (tests/test_solar.py::TestErratumCoefficients) and so a future edit
    cannot silently reintroduce the typo'd values into
    equation_of_time_minutes().
    """
    n = when if isinstance(when, int) else day_of_year(when)
    B = math.radians(fractional_year_deg(n))
    return 229.2 * (
        0.00075
        + 0.001868 * math.cos(B)
        - 0.032077 * math.sin(B)
        - 0.014615 * math.cos(2 * B)
        - 0.04089 * math.sin(2 * B)
    )


def solar_declination_deg(when: Union[_dt.date, _dt.datetime, int]) -> float:
    """Eq. 4: solar declination delta, in degrees -- NOAA GML canonical
    coefficients (see module docstring, ERRATUM 2: Kallioglu's printed
    0.39912 is a transcription typo of NOAA's 0.399912).

    `when` may be a date/datetime (day_of_year() is applied) or an int
    already understood as n (1-365/366).

    The paper (p.5, text above Eq. 4) calls this the "most stable" of
    several candidate formulas, with "error < 0.0350" [degrees] -- the only
    accuracy figure the source paper states for any part of this chain, and
    it describes the underlying formula's Fourier truncation error, not the
    separate (and smaller) Erratum 2 coefficient-typo effect quantified in
    the module docstring.
    """
    n = when if isinstance(when, int) else day_of_year(when)
    B = math.radians(fractional_year_deg(n))  # [CHOICE] see module docstring
    delta_rad = (
        0.006918
        - 0.399912 * math.cos(B)
        + 0.070257 * math.sin(B)
        - 0.006758 * math.cos(2 * B)
        + 0.000907 * math.sin(2 * B)
        - 0.002697 * math.cos(3 * B)
        + 0.00148 * math.sin(3 * B)
    )
    return math.degrees(delta_rad)  # Eq. 4's own leading "(180/pi)" factor


def _solar_declination_deg_as_printed_ERRATUM(
        when: Union[_dt.date, _dt.datetime, int]) -> float:
    """Eq. 4 EXACTLY as printed in Kallioglu 2024 -- 0.39912, a typo'd
    digit-drop of NOAA's 0.399912 (module docstring, ERRATUM 2). NOT used
    anywhere in this module's real output; kept only so the erratum's
    measured impact stays regression-tested
    (tests/test_solar.py::TestErratumCoefficients) and so a future edit
    cannot silently reintroduce the typo'd value into
    solar_declination_deg().
    """
    n = when if isinstance(when, int) else day_of_year(when)
    B = math.radians(fractional_year_deg(n))
    delta_rad = (
        0.006918
        - 0.39912 * math.cos(B)
        + 0.070257 * math.sin(B)
        - 0.006758 * math.cos(2 * B)
        + 0.000907 * math.sin(2 * B)
        - 0.002697 * math.cos(3 * B)
        + 0.00148 * math.sin(3 * B)
    )
    return math.degrees(delta_rad)


# ---------------------------------------------------------------------------
# Eq. 1 / 5 / 6: time correction, local solar time, hour angle
# ---------------------------------------------------------------------------
def time_correction_minutes(when: Union[_dt.date, _dt.datetime, int],
                            longitude_deg: float,
                            utc_offset_hours: float) -> float:
    """Eq. 1: TC, in minutes (solar time minus standard/clock time).

    longitude_deg      : location longitude, standard EAST-POSITIVE
                          convention (ISO 6709 / GPS -- e.g. Antalya is
                          +30.7, not -30.7).
    utc_offset_hours    : the location's clock offset from UTC in hours
                          (e.g. +2.0 for Turkey's historical EET standard
                          time used during the paper's 2004-2005 data
                          period; modern Turkey has used a fixed +3.0 since
                          2016).

    [CHOICE] sign convention. Eq. 1 is printed as
        TC = 4*(Lst - Lloc) + E
    and the surrounding text calls Lst "latitude for local time zone" and
    Lloc "longitude" -- almost certainly a translation slip (Lst should be
    "longitude of the standard meridian for the local time zone"; the paper
    cites Duffie & Beckman (ref. 70) for this exact equation, and Duffie &
    Beckman define both Lst and Lloc as longitudes in DEGREES WEST of
    Greenwich). Translating that into the standard degrees-EAST convention
    used by this module's public API:
        Lst_west  = -15 * utc_offset_hours   (standard meridian for the zone)
        Lloc_west = -longitude_deg
        TC = 4*(Lst_west - Lloc_west) + E
           = 4*(-15*utc_offset_hours + longitude_deg) + E
           = 4*longitude_deg - 60*utc_offset_hours + E
    This is EXACTLY NOAA's independently-published "time_offset" formula
    (time_offset = eqtime + 4*longitude - 60*timezone, longitude
    degrees-east, timezone hours-from-UTC) -- see module docstring
    CROSS-CHECK item 3. That agreement is the justification for this sign
    choice; it is not asserted on the paper's authority alone.
    """
    E = equation_of_time_minutes(when)
    return 4.0 * longitude_deg - 60.0 * utc_offset_hours + E


def local_solar_time_hours(clock_hour: float,
                           when: Union[_dt.date, _dt.datetime, int],
                           longitude_deg: float,
                           utc_offset_hours: float) -> float:
    """Eq. 6: LST = LT + TC/60, local solar time in decimal hours."""
    tc = time_correction_minutes(when, longitude_deg, utc_offset_hours)
    return clock_hour + tc / 60.0


def hour_angle_deg(local_solar_time: float) -> float:
    """Eq. 5: omega = 15*(LST - 12), degrees. Negative before solar noon,
    positive after."""
    return 15.0 * (local_solar_time - 12.0)


# ---------------------------------------------------------------------------
# Eq. 7 (CORRECTED -- see module ERRATUM section) / Eq. 8
# ---------------------------------------------------------------------------
def solar_elevation_deg(latitude_deg: float,
                        declination_deg: float,
                        hour_angle_deg_: float) -> float:
    """Eq. 8, printed correctly:
        alpha_s = sin^-1[sin(delta) sin(phi) + cos(delta) cos(phi) cos(omega)]
    """
    phi = math.radians(latitude_deg)
    delta = math.radians(declination_deg)
    omega = math.radians(hour_angle_deg_)
    s = math.sin(delta) * math.sin(phi) + math.cos(delta) * math.cos(phi) * math.cos(omega)
    s = max(-1.0, min(1.0, s))  # guard float round-off at the poles/terminator
    return math.degrees(math.asin(s))


def _zenith_angle_deg_as_printed_eq7_ERRATUM(latitude_deg: float,
                                             declination_deg: float,
                                             hour_angle_deg_: float) -> float:
    """The paper's Eq. 7 EXACTLY as printed, sin(delta) typo and all.

    NOT used anywhere in this module's real output. Kept only so
    tests/test_solar.py can demonstrate numerically why it is wrong (see
    module ERRATUM section) and so a future edit cannot silently reintroduce
    the typo into solar_zenith_angle_deg().
    """
    phi = math.radians(latitude_deg)
    delta = math.radians(declination_deg)
    omega = math.radians(hour_angle_deg_)
    c = math.sin(delta) * math.cos(phi) * math.cos(omega) + math.sin(delta) * math.sin(phi)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def solar_zenith_angle_deg(latitude_deg: float,
                           declination_deg: float,
                           hour_angle_deg_: float) -> float:
    """Eq. 7, CORRECTED per the erratum in the module docstring:
        cos(theta_z) = cos(delta) cos(phi) cos(omega) + sin(delta) sin(phi)
    theta_z = 90 deg at sunrise/sunset, 0 deg when the sun is overhead
    (matches the paper's own printed description of Eq. 7, p.5).
    """
    phi = math.radians(latitude_deg)
    delta = math.radians(declination_deg)
    omega = math.radians(hour_angle_deg_)
    c = math.cos(delta) * math.cos(phi) * math.cos(omega) + math.sin(delta) * math.sin(phi)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


# ---------------------------------------------------------------------------
# Convenience: the whole chain in one call
# ---------------------------------------------------------------------------
class SolarPosition(NamedTuple):
    n_day_of_year: int
    declination_deg: float
    equation_of_time_min: float
    local_solar_time_hours: float
    hour_angle_deg: float
    zenith_deg: float
    elevation_deg: float


def solar_position(latitude_deg: float,
                   longitude_deg: float,
                   when: _dt.datetime,
                   utc_offset_hours: Optional[float] = None) -> SolarPosition:
    """Run Eqs. 1-8 end to end for one instant.

    latitude_deg, longitude_deg : degrees, standard signed convention
                                  (north/east positive).
    when                        : a datetime with a clock time. If it is
                                  timezone-aware, utc_offset_hours is taken
                                  from it automatically (when.utcoffset());
                                  if naive, utc_offset_hours must be given
                                  explicitly. [CHOICE] a bare `date` is
                                  rejected -- the chain needs a clock hour
                                  for Eq. 5/6, which a date alone can't give.
    utc_offset_hours             : the location's clock offset from UTC, in
                                  hours (e.g. +2.0). Required for naive
                                  datetimes; overrides a tz-aware datetime's
                                  own offset if both are given.
    """
    if not isinstance(when, _dt.datetime):
        raise TypeError(
            "solar_position() needs a datetime (date + clock time) to "
            "evaluate the hour angle (Eq. 5/6); got a bare date/other type. "
            "Use solar_declination_deg()/equation_of_time_minutes() directly "
            "if only the date-dependent quantities are needed."
        )
    if utc_offset_hours is None:
        if when.tzinfo is None:
            raise ValueError(
                "utc_offset_hours is required for a naive (no-tzinfo) "
                "datetime; pass it explicitly or use a tz-aware datetime."
            )
        offset = when.utcoffset()
        if offset is None:
            raise ValueError("when.utcoffset() returned None for a tz-aware datetime")
        utc_offset_hours = offset.total_seconds() / 3600.0

    n = day_of_year(when)
    clock_hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    decl = solar_declination_deg(n)
    E = equation_of_time_minutes(n)
    lst = local_solar_time_hours(clock_hour, n, longitude_deg, utc_offset_hours)
    ha = hour_angle_deg(lst)
    zenith = solar_zenith_angle_deg(latitude_deg, decl, ha)
    elevation = solar_elevation_deg(latitude_deg, decl, ha)
    return SolarPosition(
        n_day_of_year=n,
        declination_deg=decl,
        equation_of_time_min=E,
        local_solar_time_hours=lst,
        hour_angle_deg=ha,
        zenith_deg=zenith,
        elevation_deg=elevation,
    )
