"""
tests/test_solar.py
====================
Pins vitd_solar.py against:
  (a) NOAA Global Monitoring Division's "General Solar Position
      Calculations" (https://gml.noaa.gov/grad/solcalc/solareqns.PDF), the
      CANONICAL coefficient source for this module (see vitd_solar.py module
      docstring) -- transcribed directly into TestNOAACrossCheck below, with
      the fetched text reproduced there for anyone auditing this file
      without net access;
  (b) values and identities PRINTED in Kallioglu et al. 2024 (Sci Rep
      14:3541, Eqs. 1-8), the in-repo route by which this algorithm was
      found -- the same style as
      test_physiology.py::TestPersonalGeometryFactor pins Vernez 2015 --
      including two TWO measured ERRATA: Eq. 7's sin/cos swap
      (TestErratumEq7) and four typo'd coefficients in Eqs. 2/4
      (TestErratumCoefficients);
  (c) plain orbital-mechanics facts that depend on neither source (Earth's
      axial tilt ~23.44 deg; zenith angle == latitude at solar noon on an
      equinox) -- TestGeometricSanityChecks.

All dates below use a NON-leap year (2023) so the calendar ordinal day lines
up cleanly with Eq. 3/4's printed range "1 <= n <= 365" (day_of_year()'s
[CHOICE] note in vitd_solar.py explains why leap years are not special-cased).
"""

import datetime as dt
import math

import pytest

from vitd_solar import (
    day_of_year,
    fractional_year_deg,
    equation_of_time_minutes,
    _equation_of_time_minutes_as_printed_ERRATUM,
    solar_declination_deg,
    _solar_declination_deg_as_printed_ERRATUM,
    time_correction_minutes,
    local_solar_time_hours,
    hour_angle_deg,
    solar_elevation_deg,
    solar_zenith_angle_deg,
    _zenith_angle_deg_as_printed_eq7_ERRATUM,
    solar_position,
)

MARCH_21 = dt.date(2023, 3, 21)   # n = 80 -- printed equinox date, Eq. 4 text
SEPT_23 = dt.date(2023, 9, 23)    # n = 266 -- printed equinox date, Eq. 4 text
JUNE_21 = dt.date(2023, 6, 21)    # n = 172 -- printed solstice date, Eq. 4 text
DEC_21 = dt.date(2023, 12, 21)    # n = 355 -- printed solstice date, Eq. 4 text

ANTALYA_LAT, ANTALYA_LON = 37.0, 30.7   # paper: "37 N 00, 30 E 42" (p.6)
ISTANBUL_LAT, ISTANBUL_LON = 41.0167, 28.9667  # paper: "41 N 01, 28 E 58"
TURKEY_UTC_OFFSET_2004_2005 = 2.0  # EET standard time in force during the
# paper's HelioClim-3 data period (2004-2005); modern Turkey has used a fixed
# UTC+3 since 2016. [CHOICE] the paper does not state which was used for its
# own clock-time axis, so this is our best-supported reading, not a printed
# value -- see TestAntalyaJuneWindow's docstring.


# ---------------------------------------------------------------------------
# Eq. 3/4: fractional year and declination, pinned against the coefficients
# transcribed from the printed equation images (visually re-verified against
# Research_Papers/Kallioglu...pdf p.4-5, not just PDF-text-layer extraction).
# ---------------------------------------------------------------------------
class TestFractionalYearAndDeclination:
    def test_eq3_fractional_year_deg(self):
        """Eq. 3: B = (n-1)*360/365."""
        assert fractional_year_deg(1) == pytest.approx(0.0)
        assert fractional_year_deg(366) == pytest.approx(365 * 360.0 / 365.0)

    def test_eq4_pinned_at_specific_days(self):
        """Eq. 4 hand-evaluated independently of vitd_solar.py, using NOAA
        GML's canonical coefficients (0.006918, -0.399912, 0.070257,
        -0.006758, 0.000907, -0.002697, 0.00148 -- see module docstring
        ERRATUM 2), to catch implementation regressions -- not merely
        re-running the same code."""
        def reference_decl_deg(n):
            B = math.radians((n - 1) * 360.0 / 365.0)
            rad = (0.006918 - 0.399912 * math.cos(B) + 0.070257 * math.sin(B)
                   - 0.006758 * math.cos(2 * B) + 0.000907 * math.sin(2 * B)
                   - 0.002697 * math.cos(3 * B) + 0.00148 * math.sin(3 * B))
            return math.degrees(rad)

        for n in (1, 80, 172, 266, 355, 365):
            assert solar_declination_deg(n) == pytest.approx(
                reference_decl_deg(n), abs=1e-9)

    def test_eq4_as_printed_erratum_pinned_at_specific_days(self):
        """The PRIVATE as-printed function must stay pinned to Kallioglu's
        literal typo'd coefficient (0.39912, not 0.399912) forever -- this
        is the fixed erratum reference point TestErratumCoefficients diffs
        against."""
        def reference_decl_deg_as_printed(n):
            B = math.radians((n - 1) * 360.0 / 365.0)
            rad = (0.006918 - 0.39912 * math.cos(B) + 0.070257 * math.sin(B)
                   - 0.006758 * math.cos(2 * B) + 0.000907 * math.sin(2 * B)
                   - 0.002697 * math.cos(3 * B) + 0.00148 * math.sin(3 * B))
            return math.degrees(rad)

        for n in (1, 80, 172, 266, 355, 365):
            assert _solar_declination_deg_as_printed_ERRATUM(n) == pytest.approx(
                reference_decl_deg_as_printed(n), abs=1e-9)

    def test_declination_near_zero_at_printed_equinox_dates(self):
        """Paper (p.4-5): "This angle is zero degrees at the Equinoxes
        (March 21 and September 23)".

        Tolerance is 0.3 deg, not the formula's own stated "<0.0350" --
        that figure bounds the smooth-curve Fourier truncation error against
        the TRUE continuous ephemeris, not the extra discretization error
        from using an integer calendar day n for one specific instant
        (the true equinox can fall up to ~36h from the nominal date in a
        given year, and the September equinox in particular lands ~0.24 deg
        off zero under this formula -- measured directly here, not assumed).
        See module docstring accuracy-limits section.
        """
        assert solar_declination_deg(day_of_year(MARCH_21)) == pytest.approx(0.0, abs=0.3)
        assert solar_declination_deg(day_of_year(SEPT_23)) == pytest.approx(0.0, abs=0.3)

    def test_declination_extremes_at_printed_solstice_dates(self):
        """Paper (p.4-5): declination "reaches its maximum absolute value at
        the Solstices (June 21 and December 21)".

        Sign and rough symmetry are printed facts; the exact magnitude
        (~23.44 deg, Earth's axial tilt) is cross-checked independently in
        TestGeometricSanityChecks, not asserted here from the paper alone.
        """
        june = solar_declination_deg(day_of_year(JUNE_21))
        dec = solar_declination_deg(day_of_year(DEC_21))
        assert june > 20.0        # positive max, roughly the solstice value
        assert dec < -20.0        # negative max
        assert june == pytest.approx(-dec, abs=0.2)  # printed near-symmetry


# ---------------------------------------------------------------------------
# Eq. 2: equation of time
# ---------------------------------------------------------------------------
class TestEquationOfTime:
    def test_eq2_pinned_at_specific_days(self):
        """Eq. 2, hand-evaluated independently with NOAA GML's canonical
        coefficients (229.18, 0.000075, 0.001868, -0.032077, -0.014615,
        -0.040849 -- see module docstring ERRATUM 2)."""
        def reference_E_min(n):
            B = math.radians((n - 1) * 360.0 / 365.0)
            return 229.18 * (0.000075 + 0.001868 * math.cos(B)
                             - 0.032077 * math.sin(B)
                             - 0.014615 * math.cos(2 * B)
                             - 0.040849 * math.sin(2 * B))

        for n in (1, 45, 80, 172, 266, 320, 365):
            assert equation_of_time_minutes(n) == pytest.approx(
                reference_E_min(n), abs=1e-9)

    def test_eq2_as_printed_erratum_pinned_at_specific_days(self):
        """The PRIVATE as-printed function must stay pinned to Kallioglu's
        literal typo'd coefficients (229.2, 0.00075, 0.04089) forever -- this
        is the fixed erratum reference point TestErratumCoefficients diffs
        against."""
        def reference_E_min_as_printed(n):
            B = math.radians((n - 1) * 360.0 / 365.0)
            return 229.2 * (0.00075 + 0.001868 * math.cos(B)
                             - 0.032077 * math.sin(B)
                             - 0.014615 * math.cos(2 * B)
                             - 0.04089 * math.sin(2 * B))

        for n in (1, 45, 80, 172, 266, 320, 365):
            assert _equation_of_time_minutes_as_printed_ERRATUM(n) == pytest.approx(
                reference_E_min_as_printed(n), abs=1e-9)

    def test_equation_of_time_within_known_physical_range(self):
        """The equation of time is bounded by about +/-16.5 min / -14.2 min
        over a real year (well-known astronomical fact, independent of
        either source). A badly-unit-converted implementation blows this
        range open by orders of magnitude, so this is a cheap but effective
        regression guard."""
        for n in range(1, 366, 5):
            E = equation_of_time_minutes(n)
            assert -20.0 <= E <= 20.0


# ---------------------------------------------------------------------------
# Eq. 2/4 coefficient ERRATUM (module docstring, ERRATUM 2). Kallioglu 2024
# prints 0.00075/229.2/0.04089 (Eq. 2) and 0.39912 (Eq. 4) where NOAA GML's
# canonical source -- now this module's implemented default -- prints
# 0.000075/229.18/0.040849 and 0.399912. Every other coefficient in both
# equations matches NOAA exactly, which is the signature of a transcription
# typo, not a different formulation. This class pins the MEASURED divergence
# so it can never silently drift, and documents that it is real but small.
# ---------------------------------------------------------------------------
class TestErratumCoefficients:
    def test_max_declination_divergence_matches_measured_value(self):
        """Measured max |Kallioglu(as-printed) - NOAA(canonical)| declination
        divergence over a full year, n=1..365, is 0.0454 deg (at n=1)."""
        worst = max(
            abs(_solar_declination_deg_as_printed_ERRATUM(n) - solar_declination_deg(n))
            for n in range(1, 366)
        )
        assert worst == pytest.approx(0.0454, abs=0.001)

    def test_max_equation_of_time_divergence_matches_measured_value(self):
        """Measured max |Kallioglu(as-printed) - NOAA(canonical)| equation-of
        -time divergence over a full year, n=1..365, is 0.1654 min (at
        n=319), equivalent to 0.0414 deg of hour angle (E is in minutes;
        Eq. 5's 15 deg/hour = 0.25 deg/min converts it)."""
        worst_min = max(
            abs(_equation_of_time_minutes_as_printed_ERRATUM(n) - equation_of_time_minutes(n))
            for n in range(1, 366)
        )
        assert worst_min == pytest.approx(0.1654, abs=0.001)
        worst_deg_hour_angle = worst_min * 0.25  # Eq. 5: omega = 15*(LST-12)
        assert worst_deg_hour_angle == pytest.approx(0.0414, abs=0.001)

    def test_combined_worst_case_exceeds_papers_own_error_bound(self):
        """Combined worst case (declination error + hour-angle error added
        directly, a conservative upper bound since they don't generally
        co-occur at the same day of year) is ~0.087 deg -- about 2.5x the
        paper's own stated Eq. 4 error bound ("error < 0.0350", p.5): the
        transcription typo pushes the paper's own formula outside the
        accuracy it claims for itself, once both equations' typo'd
        coefficients are considered together. Recorded here so this fact
        stays checkable, not just asserted in prose."""
        max_decl_diff = max(
            abs(_solar_declination_deg_as_printed_ERRATUM(n) - solar_declination_deg(n))
            for n in range(1, 366)
        )
        max_E_diff_deg = max(
            abs(_equation_of_time_minutes_as_printed_ERRATUM(n) - equation_of_time_minutes(n))
            for n in range(1, 366)
        ) * 0.25
        combined = max_decl_diff + max_E_diff_deg
        assert combined == pytest.approx(0.087, abs=0.002)
        assert combined > 2.0 * 0.0350  # combined exceeds the paper's own stated Eq. 4 bound

    def test_effect_on_cos_zenith_is_practically_negligible(self):
        """Even though the declination divergence formally exceeds the
        paper's own error bound, its effect on cos(zenith) -- the quantity
        that scales ambient UV -- is small: measured here at three zenith
        angles using the worst-case combined angular error (~0.087 deg)
        applied directly to the hour angle at that zenith, giving an upper
        bound on the fractional change in cos(zenith).
        """
        combined_deg = 0.087
        for zenith_deg, max_pct in [(20.0, 0.06), (60.0, 0.26), (75.0, 0.57)]:
            base = math.cos(math.radians(zenith_deg))
            perturbed = math.cos(math.radians(zenith_deg + combined_deg))
            pct_change = abs(perturbed - base) / base * 100.0
            assert pct_change < max_pct + 0.05, (
                f"zenith={zenith_deg}: pct_change={pct_change}, expected <~{max_pct}%")
            assert pct_change < 1.0  # in all cases, well under 1% -- negligible


# ---------------------------------------------------------------------------
# Eq. 7 ERRATUM verification -- the core rigour requirement of this task.
# ---------------------------------------------------------------------------
class TestErratumEq7:
    """Kallioglu 2024's printed Eq. 7 has "sin(delta)" in its first term where
    every other source (Duffie & Beckman; NOAA; the paper's OWN Eq. 8) has
    "cos(delta)". This class proves that numerically, using nothing but the
    paper's own printed Eq. 8 and its own printed identity
    theta_z + alpha_s = 90 deg (text immediately below Eq. 8, p.5) -- i.e.
    the paper is used to catch its own typo, exactly as the task required
    ("Verify this yourself against Eq. 8... record the verification").
    """

    def test_corrected_form_is_the_only_one_consistent_with_eq8(self):
        """cos(theta_z) = cos(90 - alpha_s) = sin(alpha_s), and the paper's
        Eq. 8 gives sin(alpha_s) = sin(delta)sin(phi) + cos(delta)cos(phi)cos(omega)
        -- i.e. the correct Eq. 7 bracket is the corrected one, not the
        printed one. Swept over a wide grid; must hold to float precision.
        """
        worst_corrected = 0.0
        worst_buggy = 0.0
        for lat in range(-80, 81, 10):
            for decl in (-23.44, -11.75, 0.0, 11.75, 23.44):
                for ha in range(-180, 181, 15):
                    elevation = solar_elevation_deg(lat, decl, ha)
                    zenith_corrected = solar_zenith_angle_deg(lat, decl, ha)
                    zenith_buggy = _zenith_angle_deg_as_printed_eq7_ERRATUM(lat, decl, ha)
                    worst_corrected = max(worst_corrected,
                                          abs((zenith_corrected + elevation) - 90.0))
                    worst_buggy = max(worst_buggy,
                                      abs((zenith_buggy + elevation) - 90.0))
        assert worst_corrected < 1e-9, (
            "corrected Eq. 7 must satisfy theta_z + alpha_s == 90 deg "
            f"(paper's own printed identity); worst violation {worst_corrected} deg")
        assert worst_buggy > 1.0, (
            "the as-printed Eq. 7 should NOT generally satisfy the "
            f"identity (that's the erratum); worst violation was only "
            f"{worst_buggy} deg, erratum may not be reproduced correctly")

    def test_erratum_magnitude_at_antalya_solstice_noon(self):
        """Concrete, reproducible illustration of how large the erratum is:
        at Antalya's latitude on the summer solstice at solar noon, the
        as-printed Eq. 7 is off by tens of degrees, not a rounding-sized
        amount -- this is a typo, not a precision issue."""
        decl = 23.44
        zenith_corrected = solar_zenith_angle_deg(ANTALYA_LAT, decl, 0.0)
        zenith_buggy = _zenith_angle_deg_as_printed_eq7_ERRATUM(ANTALYA_LAT, decl, 0.0)
        assert zenith_corrected == pytest.approx(13.56, abs=0.05)
        assert abs(zenith_buggy - zenith_corrected) > 30.0

    def test_solar_zenith_angle_deg_uses_the_corrected_form(self):
        """Regression guard: the public function must stay wired to the
        corrected form, not silently revert to the printed typo."""
        for lat, decl, ha in [(37.0, 23.44, 0.0), (41.0, -10.0, 30.0), (0.0, 0.0, 0.0)]:
            elevation = solar_elevation_deg(lat, decl, ha)
            zenith = solar_zenith_angle_deg(lat, decl, ha)
            assert zenith == pytest.approx(90.0 - elevation, abs=1e-9)


# ---------------------------------------------------------------------------
# Independent cross-check: NOAA Global Monitoring Division.
# Formulas transcribed directly from the fetched PDF text (reproduced here
# so this file is self-documenting even without network access):
#
#   gamma = (2*pi/365) * (day_of_year - 1 + (hour-12)/24)              [rad]
#   eqtime = 229.18*(0.000075 + 0.001868 cos(gamma) - 0.032077 sin(gamma)
#             - 0.014615 cos(2 gamma) - 0.040849 sin(2 gamma))     [minutes]
#   decl = 0.006918 - 0.399912 cos(gamma) + 0.070257 sin(gamma)
#          - 0.006758 cos(2 gamma) + 0.000907 sin(2 gamma)
#          - 0.002697 cos(3 gamma) + 0.00148 sin(3 gamma)             [rad]
#   time_offset = eqtime + 4*longitude - 60*timezone     (longitude deg-east,
#                                                          timezone hrs-from-UTC)
#   tst = hr*60 + mn + sc/60 + time_offset
#   ha = (tst/4) - 180                                             [degrees]
#   cos(zenith) = sin(lat)sin(decl) + cos(lat)cos(decl)cos(ha)
#
# Source: NOAA Global Monitoring Division, "General Solar Position
# Calculations", https://gml.noaa.gov/grad/solcalc/solareqns.PDF
# ---------------------------------------------------------------------------
def _noaa_gamma_rad(n, hour):
    return (2 * math.pi / 365.0) * (n - 1 + (hour - 12) / 24.0)


def _noaa_eqtime_min(n, hour):
    g = _noaa_gamma_rad(n, hour)
    return 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                      - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))


def _noaa_decl_deg(n, hour):
    g = _noaa_gamma_rad(n, hour)
    rad = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
           - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
           - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    return math.degrees(rad)


def _noaa_zenith_deg(lat, lon, n, hour, minute, timezone_hours):
    eqtime = _noaa_eqtime_min(n, hour + minute / 60.0)
    decl = math.radians(_noaa_decl_deg(n, hour + minute / 60.0))
    time_offset = eqtime + 4 * lon - 60 * timezone_hours
    tst = hour * 60 + minute + time_offset
    ha = math.radians((tst / 4.0) - 180.0)
    lat_r = math.radians(lat)
    c = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


class TestNOAACrossCheck:
    """Kallioglu's own equations are not the only check used here: this
    class implements NOAA's independently-published equivalent algorithm
    from scratch (see the module-level comment above) and compares outputs.

    This module's public functions now implement NOAA's coefficients
    DIRECTLY as their canonical default (module docstring, ERRATUM 2) -- the
    earlier ~0.1 deg tolerance here existed only to absorb the
    Kallioglu-vs-NOAA coefficient rounding, which no longer applies to the
    public functions (it is still tracked, against the PRIVATE as-printed
    functions, in TestErratumCoefficients). With that source of discrepancy
    gone, agreement is expected to float precision (measured residual
    ~1e-14 deg, pure floating-point noise from evaluating the same formula
    two different ways -- see B/n=day-1 here vs NOAA's gamma with an
    hour=12 term that cancels to the same value). A loose tolerance with no
    remaining reason to be loose would hide a future regression instead of
    catching one, so it is tightened here.
    """

    def test_declination_agrees_with_noaa_to_float_precision(self):
        for n in (1, 80, 172, 266, 355):
            mine = solar_declination_deg(n)
            noaa = _noaa_decl_deg(n, hour=12.0)  # NOAA gamma includes a
            # (hour-12)/24 term; hour=12 zeroes it out so both formulas are
            # evaluated at the same nominal instant (local solar noon).
            assert mine == pytest.approx(noaa, abs=1e-9), (
                f"n={n}: mine={mine}, NOAA={noaa}, diff={abs(mine - noaa)}")

    def test_equation_of_time_agrees_with_noaa_to_float_precision(self):
        for n in (1, 80, 172, 266, 355):
            mine = equation_of_time_minutes(n)
            noaa = _noaa_eqtime_min(n, hour=12.0)
            assert mine == pytest.approx(noaa, abs=1e-9), (
                f"n={n}: mine={mine}, NOAA={noaa}, diff={abs(mine - noaa)}")

    def test_zenith_angle_agrees_with_noaa_for_antalya_solstice(self):
        """Full end-to-end chain (Eqs. 1/2/3/4/5/6/7 together), compared
        against NOAA's independently-implemented algorithm for the same
        location/date/time, confirming both the corrected Eq. 7 AND the
        time-correction sign convention derived in
        time_correction_minutes()'s docstring -- to float precision, now
        that the coefficients are identical too."""
        n = day_of_year(JUNE_21)
        clock_hour = 12
        clock_minute = 0
        mine = solar_position(
            ANTALYA_LAT, ANTALYA_LON,
            dt.datetime(2023, 6, 21, clock_hour, clock_minute),
            utc_offset_hours=TURKEY_UTC_OFFSET_2004_2005,
        ).zenith_deg
        noaa = _noaa_zenith_deg(
            ANTALYA_LAT, ANTALYA_LON, n, clock_hour, clock_minute,
            TURKEY_UTC_OFFSET_2004_2005,
        )
        assert mine == pytest.approx(noaa, abs=1e-9), (
            f"mine={mine}, NOAA={noaa}, diff={abs(mine - noaa)}")

    def test_time_correction_matches_noaa_time_offset_formula_by_construction(self):
        """Documents (as an executable check, not just a docstring claim)
        that time_correction_minutes() is algebraically identical to NOAA's
        time_offset once the sign convention is applied -- see
        time_correction_minutes()'s docstring for the derivation."""
        for lon in (-120.0, -30.7, 0.0, 28.9667, 139.7):
            for utc_offset in (-8.0, 0.0, 2.0, 3.0, 9.0):
                n = 172
                E = equation_of_time_minutes(n)
                mine = time_correction_minutes(n, lon, utc_offset)
                noaa_time_offset = E + 4 * lon - 60 * utc_offset
                assert mine == pytest.approx(noaa_time_offset, abs=1e-9)


# ---------------------------------------------------------------------------
# Independent of BOTH sources: plain orbital mechanics.
# ---------------------------------------------------------------------------
class TestGeometricSanityChecks:
    def test_zenith_equals_latitude_at_solar_noon_on_equinox(self):
        """At delta=0 (equinox) and omega=0 (solar noon), Eq. 7's corrected
        bracket reduces to cos(theta_z) = cos(phi), i.e. theta_z = |phi|.
        This is a pure spherical-trigonometry identity, independent of
        either Kallioglu or NOAA's specific coefficients -- it only assumes
        the (uncontested) functional FORM of Eq. 7/8, not their Fourier
        coefficients."""
        decl_at_equinox = solar_declination_deg(day_of_year(MARCH_21))  # ~0
        for lat in (0.0, 10.0, 23.44, 37.0, 41.0, 60.0):
            zenith = solar_zenith_angle_deg(lat, decl_at_equinox, 0.0)
            assert zenith == pytest.approx(lat, abs=0.1)

    def test_declination_matches_known_obliquity_of_the_ecliptic_at_solstice(self):
        """Earth's axial tilt is ~23.44 deg (23 deg 26') -- a physical
        constant independent of this paper, this repo, or NOAA's specific
        Fourier fit (also stated explicitly in the paper's own prose, p.4:
        "derived from the 23 deg 27' value of the Earth's axis of
        rotation"). The declination formula must reproduce it at the
        solstice to within its own stated tolerance."""
        june_decl = solar_declination_deg(day_of_year(JUNE_21))
        dec_decl = solar_declination_deg(day_of_year(DEC_21))
        assert june_decl == pytest.approx(23.44, abs=0.1)
        assert dec_decl == pytest.approx(-23.44, abs=0.1)

    def test_subsolar_point_at_tropic_of_cancer_on_solstice(self):
        """On the June solstice at solar noon, the sun sits (almost)
        directly overhead at the Tropic of Cancer (lat ~23.44 deg N) --
        zenith angle should be close to 0."""
        decl = solar_declination_deg(day_of_year(JUNE_21))
        zenith = solar_zenith_angle_deg(23.44, decl, 0.0)
        assert zenith < 0.5

    def test_elevation_bounded(self):
        for lat in range(-90, 91, 15):
            for decl in (-23.44, 0.0, 23.44):
                for ha in range(-180, 181, 30):
                    el = solar_elevation_deg(lat, decl, ha)
                    assert -90.0 <= el <= 90.0

    def test_elevation_peaks_at_solar_noon(self):
        """Paper (p.5): "This angle reaches its maximum value at noon in all
        seasons." """
        decl = solar_declination_deg(day_of_year(JUNE_21))
        noon_elevation = solar_elevation_deg(ANTALYA_LAT, decl, 0.0)
        for ha in (-90, -45, -15, 15, 45, 90):
            assert solar_elevation_deg(ANTALYA_LAT, decl, ha) < noon_elevation

    def test_hour_angle_sign_matches_printed_convention(self):
        """Paper (p.5): "The hour angle is negative before noon and positive
        in the afternoon." """
        assert hour_angle_deg(9.0) < 0.0
        assert hour_angle_deg(12.0) == pytest.approx(0.0)
        assert hour_angle_deg(15.0) > 0.0


# ---------------------------------------------------------------------------
# Approximate cross-check against the paper's own reported OUTPUT (not just
# its equations): the June daylight-window figures quoted in the Results
# text (p.9): "According to the average values of June (162 n)... between
# 08:42 and 15:18... for Antalya" and "the longest daytime interval of D3
# synthesis during the year is 396 min... for Antalya".
#
# This is NOT treated as an exact pin (see caveats below), unlike the Eq. 4
# coefficient pins above:
#   - the paper does not print the longitude/timezone convention used for
#     its own clock-time axis (Antalya's coordinates come from Fig. 3's
#     caption, not the equations section);
#   - "08:42-15:18" is attributed in-text to "n=162" (~June 11), yet in the
#     same sentence called "the longest... interval... during the year",
#     which should be the solstice (n=172, June 21) elsewhere in the paper
#     -- 15:18-08:42 = 396 min matches the explicitly-stated peak value
#     exactly, so we read "162" as a probable single-digit transcription of
#     "172" (2 vs 7) and compare against n=172;
#   - the 08:42-15:18 window itself is read off a plotted figure (Fig. 5),
#     not restated as exact equation output anywhere in the text.
# The tolerance below (20 minutes on each edge) reflects those three
# uncertainties, not a formula-precision claim.
# ---------------------------------------------------------------------------
class TestAntalyaJuneWindowApproximateCrossCheck:
    def test_45deg_elevation_window_near_printed_antalya_solstice_window(self):
        decl = solar_declination_deg(day_of_year(JUNE_21))
        crossings = []
        prev_above = None
        clock_hour = 4.0
        while clock_hour < 20.0:
            lst = local_solar_time_hours(clock_hour, day_of_year(JUNE_21),
                                          ANTALYA_LON, TURKEY_UTC_OFFSET_2004_2005)
            ha = hour_angle_deg(lst)
            el = solar_elevation_deg(ANTALYA_LAT, decl, ha)
            above = el > 45.0
            if prev_above is False and above:
                crossings.append(clock_hour)
            if prev_above is True and not above:
                crossings.append(clock_hour)
            prev_above = above
            clock_hour += 1.0 / 120.0  # 30-second steps

        assert len(crossings) == 2, f"expected one rise and one fall crossing, got {crossings}"
        start, end = crossings
        window_min = (end - start) * 60.0
        # Paper: window 08:42-15:18, duration 396 min (see class docstring
        # for why these are treated as approximate, not exact).
        assert start == pytest.approx(8 + 42 / 60.0, abs=20 / 60.0)
        assert end == pytest.approx(15 + 18 / 60.0, abs=20 / 60.0)
        assert window_min == pytest.approx(396.0, abs=20.0)


# ---------------------------------------------------------------------------
# API-level checks (input handling, not physics).
# ---------------------------------------------------------------------------
class TestSolarPositionAPI:
    def test_requires_datetime_not_bare_date(self):
        with pytest.raises(TypeError):
            solar_position(37.0, 30.7, JUNE_21, utc_offset_hours=2.0)

    def test_requires_utc_offset_for_naive_datetime(self):
        with pytest.raises(ValueError):
            solar_position(37.0, 30.7, dt.datetime(2023, 6, 21, 12, 0))

    def test_tz_aware_datetime_infers_utc_offset(self):
        tz = dt.timezone(dt.timedelta(hours=2))
        aware = dt.datetime(2023, 6, 21, 12, 0, tzinfo=tz)
        naive_equivalent = dt.datetime(2023, 6, 21, 12, 0)
        a = solar_position(ANTALYA_LAT, ANTALYA_LON, aware)
        b = solar_position(ANTALYA_LAT, ANTALYA_LON, naive_equivalent, utc_offset_hours=2.0)
        assert a.zenith_deg == pytest.approx(b.zenith_deg, abs=1e-9)

    def test_explicit_utc_offset_overrides_tzinfo(self):
        tz = dt.timezone(dt.timedelta(hours=2))
        aware = dt.datetime(2023, 6, 21, 12, 0, tzinfo=tz)
        overridden = solar_position(ANTALYA_LAT, ANTALYA_LON, aware, utc_offset_hours=3.0)
        default = solar_position(ANTALYA_LAT, ANTALYA_LON, aware)
        assert overridden.zenith_deg != pytest.approx(default.zenith_deg, abs=1e-6)
