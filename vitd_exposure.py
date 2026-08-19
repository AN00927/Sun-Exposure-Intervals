"""
vitd_exposure.py
=================
PHASE 1 entry point: turns real-world inputs into a UV dose.

    (latitude, longitude, datetime, exposed body sites, posture) -> SED

Before this module, nothing in the repo connected the two links that already
existed -- vitd_solar.py (where is the sun) and vitd_ambient.py (what is the
ambient UV Index) -- to the personal-dose machinery in vitd_physiology.py and
vitd_objectives.py. A caller had to already know a UV Index, a geometry
factor, and separately keep an `exposed_bsa` fraction consistent with
whatever body sites that geometry factor implied. This module is that join.
It computes nothing new scientifically: every number below is produced by an
existing, sourced function in vitd_solar.py, vitd_ambient.py,
vitd_physiology.py or vitd_objectives.py. This module's only job is asking
the two questions a real person can actually answer -- "what date/time and
where" and "what's uncovered" -- and wiring the answers through consistently.

THE CRUX THIS MODULE EXISTS TO FIX (repo open item 5b)
--------------------------------------------------------
Which body sites are bare determines BOTH:
  (a) the mean personal exposure ratio (via `vis_percent`, the area-weighted
      sky-visibility feeding vernez_exposure_ratio()), and
  (b) the exposed body-surface-area fraction (`exposed_bsa`), which scales
      vitd_model.synthesis_per_session() OUTSIDE the log dose-response term
      (`bsa_f` in that function), while vis_percent's effect enters INSIDE
      it (`sed` -> `ln(1 + sed*E_source)` via personal_geometry_factor()
      changing the per-minute SED). These are not arithmetically the same
      operation, so getting one from a site set and the other from a
      DIFFERENT, inconsistent site set does not cancel out -- it silently
      biases the combined answer, in either direction, with no error raised
      anywhere in the model.

Nothing upstream enforced that both numbers come from the same declaration
of what is bare. This module makes that inconsistency structurally
impossible: `ExposureGeometry` (built exclusively by
`geometry_for_site_set()` / `geometry_for_preset()`) is the ONLY place either
number is produced, from ONE `sites` tuple, and both `vis_percent` and
`exposed_bsa` are read-only fields on the same object. There is no code path
in this module that lets a caller set one without the other flowing from the
same site set.

BSA SOURCE -- Lund & Browder (1944), adult column
--------------------------------------------------
Lund CC, Browder NC. "The estimation of areas of burns." Surgical, Gynecology
and Obstetrics. 1944;79:352-358. The per-region adult percentages below are
the "16 years and over" column of that chart, as it is universally
reproduced in clinical burns literature (cross-checked here 2026-08-17 via
web search against multiple independent reproductions, including the
squarespace-hosted scanned chart
https://static1.squarespace.com/static/51827056e4b001b882273584/t/6000344d9d313c12bb06af3f/1610626129728/Burns+Lund+and+Browder+Chart+1.pdf
and a summary aggregation citing the same figures; not re-derived or
adjusted here). This repo does not hold the 1944 original as a PDF (checked
Research_Papers/ and Papers_and_Datasets/ -- absent from both); the table
below is the standard secondary reproduction, not an independent
transcription of the primary source. If page-exact provenance is needed
later, get the 1944 original or a scan of it.

    LUND_BROWDER_ADULT_BSA_PERCENT (16 regions, sums to 100.0):
        Head 7, Neck 2, Anterior trunk 13, Posterior trunk 13,
        R+L Buttock 2.5 each (5 total), Genitalia 1,
        R+L Upper arm 4 each (8 total), R+L Forearm 3 each (6 total),
        R+L Hand 2.5 each (5 total), R+L Thigh 9.5 each (19 total),
        R+L Leg (lower) 7 each (14 total), R+L Foot 3.5 each (7 total).

Bilateral regions are stored here already SUMMED (both sides) because every
Vernez site below is likewise a single bilateral entity ("Forearm
(external)" is not handed out per-arm) -- see BSA_SITE_FRACTION.

MAPPING VERNEZ'S 13 SITES ONTO THE 16 LUND-BROWDER REGIONS
-------------------------------------------------------------
Vernez 2015 Table S1 (vitd_physiology.VERNEZ_TABLE_S1_VIS) has 13 body-site
labels that do not line up 1:1 with Lund-Browder's regions. Every mapping
decision is marked # [CHOICE] at its definition below; summarized here:

  - Head (7%) has no face/scalp split in the standard chart. Vernez has both
    Face and Skull. # [CHOICE] equal 50/50 split (3.5% each) -- no finer
    authoritative subdivision exists to weight this any other way, so equal
    split is used rather than inventing an uneven number that would imply
    false precision.
  - Neck (2%) -> Vernez only has "Neck back" (no neck-front site).
    # [CHOICE] the full 2% is assigned to neck_back; this uses back-of-neck
    Vis as a proxy for the whole neck's exposure, which the source data
    cannot avoid.
  - Anterior trunk (13%) -> Vernez only has "Belly". # [CHOICE] the full 13%
    is assigned to belly; belly Vis stands in for the whole anterior trunk
    (chest included), which the source data cannot avoid.
  - Posterior trunk (13%) -> Vernez has "Upper back", "Low back", AND "Top
    of shoulders" (a horizontal, dorsal, trapezius-line surface, distinct
    from both). # [CHOICE] equal 3-way split (13/3 = 4.333% each) for the
    same false-precision reason as the head split.
  - Buttock (5%) and Genitalia (1%) -> no Vernez site exists for either.
    # [CHOICE] EXCLUDED from BSA_SITE_FRACTION entirely -- never offered as
    bare in any preset or site set here, not even "swimwear" (see preset
    docstring). Not modelled rather than guessed.
  - Upper arm (8%) -> Vernez has "Upper arm (external)" (the lateral/
    posterior arm shaft) AND "Shoulder" (the deltoid cap, anatomically the
    proximal end of the same limb segment, not the trunk). # [CHOICE] equal
    split (4% each) between shoulder and upper_arm_external -- both are
    genuinely distinct SimUVEx sites with different measured Vis, so this is
    not a duplicate; it is Lund-Browder's coarser single region being spent
    across Vernez's finer partition of it.
  - Forearm (6%) -> Vernez "Forearm (external)" only. # [CHOICE] full 6%;
    external forearm Vis stands in for the whole forearm circumference.
  - Hand (5%) -> Vernez "Hand (back)" only (dorsum, not palm). # [CHOICE]
    full 5% assigned; dorsum Vis (facing mostly upward/outward) likely
    OVERSTATES mean hand Vis relative to a true palm+dorsum average, since a
    downward-facing palm sees less sky. Flagged, not corrected -- correcting
    it would mean inventing a palm Vis value with no source.
  - Thigh (19%) -> Vernez "Upper Leg (front)" only (no posterior-thigh
    site). # [CHOICE] full 19%; front-of-thigh Vis likely OVERSTATES mean
    thigh Vis (posterior thigh is more self-shaded when standing/walking).
  - Leg/lower leg (14%) -> Vernez "Lower leg (back)" only (calf, no shin
    site). # [CHOICE] full 14%; back-of-calf Vis likely UNDERSTATES mean
    lower-leg Vis (the shin typically faces more sky than the calf).
  - Foot (7%) -> no Vernez site exists at all. # [CHOICE] EXCLUDED, same as
    buttock/genitalia -- feet are never modelled as bare here (sandals,
    barefoot) because there is no Vis measurement to use.

Net effect: BSA_SITE_FRACTION's 13 entries sum to 87.0% (=100% minus
buttock 5% + genitalia 1% + foot 7%). That 87.0% is therefore the highest
`exposed_bsa` this module can ever construct from real Vernez sites -- see
the "swimwear" preset, which is deliberately named and documented as hitting
that ceiling rather than 100%, because covering the excluded 13% honestly
would require inventing numbers this module refuses to invent.

CRITICAL CAVEATS (do not lose these downstream)
--------------------------------------------------------------------------
1. UV Index is ERYTHEMALLY weighted, not vitamin-D weighted (see
   vitd_ambient.py module docstring). This module never applies a
   vitamin-D weighting -- that is E_source (VDE:EE), owned by
   vitd_model.SOURCE_CALIBRATION_TABLE / vitd_esource.py. Every SED value
   this module returns is an ERYTHEMAL SED; multiply by the appropriate
   per-study E_source before treating it as a vitamin-D-effective dose (see
   vitd_model.synthesis_per_session, which does that multiplication itself
   -- do not do it twice).
2. Vernez's exposure ratio (which vis_percent ultimately drives, via
   vitd_objectives.sed_from_minutes -> personal_geometry_factor ->
   vernez_exposure_ratio) assumes UNPROTECTED skin, NO shade, and
   CONTINUOUS exposure. It is a ceiling, not a behavioural average.
   Real personal doses measured by dosimeter badge are ~0.9-2.9% of ambient
   (Parisi/Cockell ranges cited in vitd_physiology.py), one to two orders of
   magnitude below the unshaded Vernez values this module produces. A
   separate behavioural term (time-in-shade, intermittency) is being built
   independently in vitd_behaviour.py by another agent; this module does
   NOT attempt to model or approximate that -- treat every SED returned here
   as the deliberate-unshaded-exposure ceiling for the stated minutes, not a
   forecast of what a person will actually absorb doing normal activity.
3. [VERIFY] Vernez Table S1 gives IDENTICAL Vis values for 'Top of
   shoulders' and 'Upper back' across all five postures (reproduced verbatim
   upstream in vitd_physiology.py / the source CSV). This module does not
   de-duplicate that; both sites are used independently below as documented.
4. DARK-SKIN PRIORITY: the CIE erythemal action spectrum defining the UV
   Index (and therefore every ambient UVI this module consumes) is anchored
   to predominantly light (Fitzpatrick I-III) skin sunburn data (McKinlay &
   Diffey 1987; see vitd_ambient.py). That inheritance is not corrected
   anywhere in this module -- it is flagged here so it is not forgotten
   downstream, consistent with this repo's dark-skin-priority stance.
5. Solar zenith angle timing. vitd_physiology.vernez_exposure_ratio()'s
   `sza_deg` parameter is documented (vitd_physiology.py) as "maximal daily
   solar zenith angle, degrees" -- i.e. Vernez's own regression was fit
   against a single per-day SZA summary, not an instant. # [CHOICE] this
   module instead passes the INSTANTANEOUS solar elevation at the caller's
   requested `when` (via vitd_solar.solar_position), because (a) that is the
   one quantity actually available from a "what date/time" question, and
   (b) Vernez's own text states solar geometry is only a ~1.18x term against
   Vis's ~11.6x, so this substitution is a second-order approximation on an
   already second-order term. It is not a re-derivation of what "maximal
   daily" meant in the original paper -- that ambiguity predates this
   module (vitd_physiology.py already carries it) and is not resolved here.

WHAT THIS MODULE DOES NOT DO
-----------------------------
- Does not compute 25(OH)D synthesis or run vitd_model.simulate(). It stops
  at SED, the documented boundary of "PHASE 1". `exposed_bsa` is returned on
  ExposureGeometry/ExposureResult so a caller can pass it straight to
  vitd_model.simulate(exposed_bsa_override=...) or
  vitd_objectives.minutes_for_target(exposed_bsa=...) themselves.
- Does not model clothing physically (UPF, fabric transmission) -- a site is
  binary bare/covered in this module. A real garment is not perfectly
  opaque; this module cannot tell you that.
- Does not model hair coverage of the scalp. `skull` has a BSA_SITE_FRACTION
  entry (for a caller who wants it -- e.g. a bald or shaved head) but is
  deliberately excluded from every named preset below, since assuming a bare
  scalp by default would silently overstate a typical person's exposure.
- Does not touch vitd_model.py, vitd_physiology.py, vitd_ambient.py,
  vitd_solar.py, vitd_objectives.py, or any of their tests. Purely additive:
  imports only.

TESTING WITHOUT NETWORK
------------------------
`exposure_sed()` (the deterministic core) takes an already-known ambient UV
Index and makes no network call. `exposure_sed_live()` (the convenience
wrapper that also fetches ambient UVI) accepts the same injectable
`http_get` seam as vitd_ambient.fetch_ambient_uvi(), so tests/test_exposure.py
can exercise it with the SAME recorded fixtures vitd_ambient's own tests use,
with zero real network access.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

from vitd_ambient import AmbientUVI, AmbientUVIUnavailable, fetch_ambient_uvi
from vitd_objectives import SED_PER_MIN_PER_UVI, sed_from_minutes
from vitd_physiology import (VERNEZ_TABLE_S1_POSTURES, _normalize_body_site,
                             vis_table_s1)
from vitd_solar import SolarPosition, solar_position

# ---------------------------------------------------------------------------
# 1. Lund & Browder (1944) adult BSA regions -- SOURCED, see module docstring.
# ---------------------------------------------------------------------------
# Bilateral regions summed (both sides) here, since every Vernez site below
# is likewise a single bilateral entity. Sums to 100.0.
LUND_BROWDER_ADULT_BSA_PERCENT: Dict[str, float] = {
    "head": 7.0,
    "neck": 2.0,
    "anterior_trunk": 13.0,
    "posterior_trunk": 13.0,
    "buttock": 2.5 + 2.5,       # excluded below -- no Vernez site
    "genitalia": 1.0,           # excluded below -- no Vernez site
    "upper_arm": 4.0 + 4.0,
    "forearm": 3.0 + 3.0,
    "hand": 2.5 + 2.5,
    "thigh": 9.5 + 9.5,
    "leg": 7.0 + 7.0,           # lower leg (shin/calf)
    "foot": 3.5 + 3.5,          # excluded below -- no Vernez site
}
assert abs(sum(LUND_BROWDER_ADULT_BSA_PERCENT.values()) - 100.0) < 1e-9

# Lund-Browder regions with NO corresponding Vernez Table S1 site. Never
# includable in any site set below -- see module docstring mapping section.
BSA_EXCLUDED_LUND_BROWDER_REGIONS: Tuple[str, ...] = ("buttock", "genitalia", "foot")
BSA_EXCLUDED_PERCENT_TOTAL: float = sum(
    LUND_BROWDER_ADULT_BSA_PERCENT[r] for r in BSA_EXCLUDED_LUND_BROWDER_REGIONS
)  # 5.0 + 1.0 + 7.0 = 13.0

# ---------------------------------------------------------------------------
# 2. Vernez site -> BSA fraction (of TOTAL body, as a fraction 0-1)
# ---------------------------------------------------------------------------
# Keys are normalized exactly as vitd_physiology._normalize_body_site() would
# produce from the Table S1 CSV headers, so they line up with vis_table_s1().
# Every split is documented as # [CHOICE] at the point of use; see module
# docstring "MAPPING" section for the reasoning behind each one.
BSA_SITE_FRACTION: Dict[str, float] = {
    # Head (7%) split evenly face/skull -- [CHOICE] no finer source.
    "face": 7.0 / 2 / 100.0,
    "skull": 7.0 / 2 / 100.0,
    # Neck (2%) -> neck_back is the only neck site Vernez measured. [CHOICE]
    "neck_back": 2.0 / 100.0,
    # Anterior trunk (13%) -> belly is the only anterior-trunk site. [CHOICE]
    "belly": 13.0 / 100.0,
    # Posterior trunk (13%) split 3 ways -- [CHOICE] equal split, no finer
    # source; top_of_shoulders included here as a dorsal/trunk surface, not
    # arm (see next block for the distinct "shoulder" deltoid site).
    "upper_back": 13.0 / 3 / 100.0,
    "low_back": 13.0 / 3 / 100.0,
    "top_of_shoulders": 13.0 / 3 / 100.0,
    # Upper arm (8%) split evenly between the lateral shaft (upper_arm_
    # external) and the deltoid cap (shoulder) -- [CHOICE] equal split,
    # both are distinct measured Vernez sites, not a duplicate.
    "upper_arm_external": 8.0 / 2 / 100.0,
    "shoulder": 8.0 / 2 / 100.0,
    # Forearm (6%) -> forearm_external is the only forearm site. [CHOICE]
    "forearm_external": 6.0 / 100.0,
    # Hand (5%) -> hand_back (dorsum) is the only hand site. [CHOICE]
    "hand_back": 5.0 / 100.0,
    # Thigh (19%) -> upper_leg_front is the only thigh site. [CHOICE]
    "upper_leg_front": 19.0 / 100.0,
    # Lower leg (14%) -> lower_leg_back (calf) is the only lower-leg site. [CHOICE]
    "lower_leg_back": 14.0 / 100.0,
}
# Sums to 0.87 (=100% minus the three excluded regions, 13%). See module
# docstring "Net effect" paragraph.
BSA_SITE_FRACTION_TOTAL: float = sum(BSA_SITE_FRACTION.values())
assert abs(BSA_SITE_FRACTION_TOTAL - (100.0 - BSA_EXCLUDED_PERCENT_TOTAL) / 100.0) < 1e-9

# All 13 Vernez Table S1 sites should have a BSA entry; confirms the mapping
# above accounted for every site the CSV actually provides (module import
# time -- fails loudly rather than silently omitting a site from every
# preset built below).
_VERNEZ_SITES_NORMALIZED = {
    _normalize_body_site(s) for s in
    ("Face", "Skull", "Forearm (external)", "Upper arm (external)",
     "Neck back", "Top of shoulders", "Belly", "Upper back", "Hand (back)",
     "Shoulder", "Upper Leg (front)", "Lower leg (back)", "Low back")
}
assert _VERNEZ_SITES_NORMALIZED == set(BSA_SITE_FRACTION), (
    "BSA_SITE_FRACTION keys do not match the 13 Vernez Table S1 sites -- "
    f"missing {_VERNEZ_SITES_NORMALIZED - set(BSA_SITE_FRACTION)}, "
    f"extra {set(BSA_SITE_FRACTION) - _VERNEZ_SITES_NORMALIZED}"
)


# ---------------------------------------------------------------------------
# 3. The single declaration of "what is bare" -- vis_percent AND exposed_bsa
#    both derive from the SAME `sites` tuple, so they cannot silently drift
#    apart (repo open item 5b, see module docstring).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExposureGeometry:
    """One declaration of which body sites are bare, and everything derived
    from it. Build this ONLY via geometry_for_site_set() / geometry_for_preset()
    -- never construct vis_percent and exposed_bsa independently elsewhere.
    """
    sites: Tuple[str, ...]
    posture: str
    vis_percent: float           # area-weighted mean sky visibility, 0-100
    exposed_bsa: float           # fraction of total BSA, 0-1
    per_site_vis: Dict[str, float] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce the whole point of this class: vis_percent and exposed_bsa
        must both be derivable from `sites`, not asserted independently.

        ADDED 2026-08-18 after the original guarantee was found to be false.
        `frozen=True` blocks attribute mutation but blocks NOTHING else, and
        two paths defeated it outright:

            g = geometry_for_preset("short_sleeves", posture)   # bsa 0.165
            dataclasses.replace(g, vis_percent=10.0)            # -> accepted
            ExposureGeometry(sites=g.sites, posture=posture,
                             vis_percent=10.0, exposed_bsa=0.60, ...)  # accepted

        Both produced exactly the pairing this module exists to make
        impossible -- a low single-site Vis carried on a BSA that belongs to a
        different, higher-Vis site set (repo open item 5b). A guarantee that
        holds only if callers use the blessed constructor is a convention, not
        a guarantee, so it is now checked here and applies to EVERY path.
        """
        recomputed_vis = vis_percent_for_site_set(self.sites, self.posture)
        recomputed_bsa = exposed_bsa_for_site_set(self.sites)
        # Tolerance is float-noise only, deliberately not a fudge factor.
        if abs(recomputed_vis - self.vis_percent) > 1e-6:
            raise ValueError(
                f"ExposureGeometry: vis_percent={self.vis_percent!r} is not "
                f"derivable from sites={self.sites!r} at posture "
                f"{self.posture!r} (recomputed {recomputed_vis:.6f}). "
                "Both vis_percent and exposed_bsa must come from one "
                "declaration of which sites are bare -- see open item 5b. "
                "Build via geometry_for_site_set()/geometry_for_preset()."
            )
        if abs(recomputed_bsa - self.exposed_bsa) > 1e-9:
            raise ValueError(
                f"ExposureGeometry: exposed_bsa={self.exposed_bsa!r} is not "
                f"derivable from sites={self.sites!r} (recomputed "
                f"{recomputed_bsa:.9f}). See open item 5b."
            )


def vis_percent_for_site_set(sites: Iterable[str], posture: str) -> float:
    """Area-weighted mean Vis[%] across `sites` for `posture`.

    Weights are BSA_SITE_FRACTION -- i.e. a large-area site (e.g. thigh,
    19% of body) pulls the mean toward its own Vis more than a small-area
    site (e.g. neck, 2%) would. This is what makes the mean the RIGHT
    average to combine with exposed_bsa: both are computed from the same
    per-site BSA weights, over the same site set.

    RAISES on an unknown site (delegates to vitd_physiology.vis_table_s1)
    or an empty site set, rather than returning a meaningless 0/nan.
    """
    sites = tuple(_normalize_body_site(s) for s in sites)
    if not sites:
        raise ValueError("vis_percent_for_site_set: sites must be non-empty")
    total_weight = 0.0
    weighted_vis = 0.0
    for site in sites:
        if site not in BSA_SITE_FRACTION:
            raise KeyError(
                f"No BSA_SITE_FRACTION entry for '{site}'. Known: "
                f"{sorted(BSA_SITE_FRACTION)}. (vis_table_s1 may still know "
                f"this site name -- BSA_SITE_FRACTION is the narrower list "
                f"this module can area-weight.)"
            )
        w = BSA_SITE_FRACTION[site]
        weighted_vis += w * vis_table_s1(site, posture)
        total_weight += w
    return weighted_vis / total_weight


def exposed_bsa_for_site_set(sites: Iterable[str]) -> float:
    """Sum of BSA_SITE_FRACTION over `sites` -- fraction of total body (0-1).

    Uses the SAME BSA_SITE_FRACTION table as vis_percent_for_site_set(), so
    calling both on the same `sites` iterable can never produce an
    inconsistent pair -- that is the entire point of this module (see
    docstring). Duplicate sites in `sites` are only counted once (a set is
    taken internally) -- a body site cannot be "more bare" by being named
    twice.
    """
    sites = {_normalize_body_site(s) for s in sites}
    if not sites:
        raise ValueError("exposed_bsa_for_site_set: sites must be non-empty")
    missing = sites - set(BSA_SITE_FRACTION)
    if missing:
        raise KeyError(f"No BSA_SITE_FRACTION entry for: {sorted(missing)}")
    return sum(BSA_SITE_FRACTION[s] for s in sites)


def geometry_for_site_set(sites: Iterable[str], posture: str) -> ExposureGeometry:
    """Build the ONE ExposureGeometry for a custom (non-preset) site set.

    This is the structural fix for repo open item 5b: call this once with a
    `sites` tuple, and use the SAME returned object's .vis_percent and
    .exposed_bsa everywhere downstream -- never source them from two
    different site-set declarations.
    """
    sites_norm = tuple(dict.fromkeys(_normalize_body_site(s) for s in sites))  # de-dup, keep order
    if posture not in VERNEZ_TABLE_S1_POSTURES:
        raise KeyError(f"Unknown posture '{posture}'. Known: {VERNEZ_TABLE_S1_POSTURES}.")
    per_site = {s: vis_table_s1(s, posture) for s in sites_norm}
    vis_pct = vis_percent_for_site_set(sites_norm, posture)
    bsa = exposed_bsa_for_site_set(sites_norm)
    notes = ()
    if "skull" in sites_norm:
        notes = notes + (
            "'skull' is included in this site set -- BSA_SITE_FRACTION and "
            "vis_table_s1 treat it as bare scalp with no hair coverage; no "
            "preset in this module includes it by default (see module "
            "docstring).",
        )
    return ExposureGeometry(sites=sites_norm, posture=posture,
                            vis_percent=vis_pct, exposed_bsa=bsa,
                            per_site_vis=per_site, notes=notes)


# ---------------------------------------------------------------------------
# 4. Named presets -- what a real person can actually answer ("what's
#    uncovered?"), built from the same site-set machinery above.
# ---------------------------------------------------------------------------
# skull is deliberately never included by default (hair coverage -- see
# module docstring "WHAT THIS MODULE DOES NOT DO"). A caller modelling a
# bald/shaved head can build a custom site set including "skull" via
# geometry_for_site_set() directly.
CLOTHING_PRESETS: Dict[str, Tuple[str, ...]] = {
    # Long sleeves + long trousers, only face/neck/hands bare.
    "face_and_hands": ("face", "neck_back", "hand_back"),
    # Short-sleeve shirt + long trousers: forearms bare below the sleeve
    # hem, but [CHOICE] upper arm treated as fully covered by the sleeve
    # (a real short sleeve ends mid-bicep and bares a sliver of upper arm;
    # simplified to "covered" here rather than guessing what fraction).
    "short_sleeves": ("face", "neck_back", "forearm_external", "hand_back"),
    # Short-sleeve shirt + shorts: adds both leg sites to short_sleeves.
    "shorts_and_tshirt": ("face", "neck_back", "forearm_external", "hand_back",
                          "upper_leg_front", "lower_leg_back"),
    # [CHOICE] All 12 non-scalp Vernez sites bare -- the highest exposed_bsa
    # (0.835) this module's BSA table can construct. This is NOT "fully
    # nude" (buttock/genitalia/foot are excluded everywhere in this module,
    # see docstring) and is not scaled down for a swimsuit actually covering
    # a small area of trunk -- it is deliberately the ceiling case, named
    # for the closest everyday scenario to it.
    "swimwear": ("face", "neck_back", "forearm_external", "upper_arm_external",
                "shoulder", "top_of_shoulders", "belly", "upper_back",
                "low_back", "hand_back", "upper_leg_front", "lower_leg_back"),
}

CLOTHING_PRESET_DESCRIPTIONS: Dict[str, str] = {
    "face_and_hands": "Long sleeves and long trousers; only face, neck and "
                      "hands bare.",
    "short_sleeves": "Short-sleeve shirt and long trousers; face, neck, "
                     "forearms and hands bare.",
    "shorts_and_tshirt": "Short-sleeve shirt and shorts; face, neck, "
                         "forearms, hands and both legs bare.",
    "swimwear": "Minimal coverage; every non-scalp Vernez site bare (the "
               "ceiling this module's BSA table supports -- see module "
               "docstring for why buttock/genitalia/feet are not included).",
}


def geometry_for_preset(preset: str, posture: str) -> ExposureGeometry:
    """ExposureGeometry for one of CLOTHING_PRESETS. RAISES on an unknown
    preset name rather than guessing a site set."""
    if preset not in CLOTHING_PRESETS:
        raise KeyError(
            f"Unknown preset '{preset}'. Known: {sorted(CLOTHING_PRESETS)}."
        )
    return geometry_for_site_set(CLOTHING_PRESETS[preset], posture)


# ---------------------------------------------------------------------------
# 5. Solar geometry -- thin, direct use of vitd_solar.solar_position(). No
#    new solar-position logic lives here; see vitd_solar.py for the sourced
#    NOAA-coefficient implementation.
# ---------------------------------------------------------------------------
def solar_geometry_at(latitude_deg: float, longitude_deg: float,
                      when: dt.datetime,
                      utc_offset_hours: Optional[float] = None) -> SolarPosition:
    """Solar position at (lat, lon, when). Thin pass-through to
    vitd_solar.solar_position() -- see that module for the sourced
    algorithm and the naive/aware datetime rules (utc_offset_hours is
    required for a naive `when`, optional/overriding for a tz-aware one).
    """
    return solar_position(latitude_deg, longitude_deg, when, utc_offset_hours)


# ---------------------------------------------------------------------------
# 6. The chain, end to end: geometry + solar position + ambient UVI -> SED
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExposureResult:
    """Output of exposure_sed() / exposure_sed_live() -- the SED delivered to
    EXPOSED skin (not scaled by exposed_bsa; see module docstring "WHAT THIS
    MODULE DOES NOT DO"), plus every intermediate value that produced it.

    `sed` is ERYTHEMAL SED (see module docstring caveat 1) -- apply the
    appropriate per-study E_source (vitd_model.get_E_source /
    SOURCE_CALIBRATION_TABLE) before treating it as vitamin-D-effective, or
    pass it straight into vitd_model.synthesis_per_session(), which applies
    E_source itself.
    """
    latitude: float
    longitude: float
    timestamp_utc: str
    minutes: float
    geometry: ExposureGeometry
    solar: SolarPosition
    ambient_uv_index: float
    geometry_factor: float   # the ER actually applied to `minutes` (0-1)
    sed: float

    @property
    def exposed_bsa(self) -> float:
        """Convenience passthrough -- see geometry.exposed_bsa. Feed this to
        vitd_model.simulate(exposed_bsa_override=...) or
        vitd_objectives.minutes_for_target(exposed_bsa=...) downstream; this
        module does not call either itself (see module docstring)."""
        return self.geometry.exposed_bsa


def _resolve_geometry(sites: Optional[Iterable[str]], preset: Optional[str],
                      posture: str) -> ExposureGeometry:
    if (sites is None) == (preset is None):
        raise ValueError(
            "exactly one of `sites` or `preset` must be given "
            f"(sites={sites!r}, preset={preset!r})"
        )
    if preset is not None:
        return geometry_for_preset(preset, posture)
    return geometry_for_site_set(sites, posture)


def exposure_sed(latitude: float, longitude: float, when: dt.datetime,
                 minutes: float, ambient_uv_index: float, *,
                 sites: Optional[Iterable[str]] = None,
                 preset: Optional[str] = None,
                 posture: str = "standing_erect_arms_down",
                 utc_offset_hours: Optional[float] = None) -> ExposureResult:
    """The deterministic core of PHASE 1: (place, time, minutes, ambient UVI,
    what's bare) -> SED. No network access.

    Exactly one of `sites` (a custom iterable of Vernez site names) or
    `preset` (a CLOTHING_PRESETS key) must be given -- pass neither / both
    and this raises ValueError rather than guessing which was intended.

    `ambient_uv_index` is caller-supplied (e.g. from
    vitd_ambient.uvi_from_user_input() or a live vitd_ambient.fetch_ambient_uvi()
    call made by the caller) -- see exposure_sed_live() for a convenience
    wrapper that fetches it for you.
    """
    if minutes < 0:
        raise ValueError(f"minutes must be non-negative, got {minutes}")
    if ambient_uv_index < 0:
        raise ValueError(f"ambient_uv_index must be non-negative, got {ambient_uv_index}")

    geometry = _resolve_geometry(sites, preset, posture)
    solar = solar_geometry_at(latitude, longitude, when, utc_offset_hours)

    # sed_from_minutes() internally converts elevation -> SZA and calls
    # vernez_exposure_ratio(vis_percent, sza) -- see vitd_objectives.py /
    # vitd_physiology.py. No geometry math is duplicated here.
    sed = sed_from_minutes(minutes, ambient_uv_index,
                           solar_elevation_deg=solar.elevation_deg,
                           vis_percent=geometry.vis_percent)
    # Recover the exact geometry_factor actually applied, for transparency
    # (sed_from_minutes does not hand this back on its own).
    geometry_factor = (sed / (minutes * ambient_uv_index * SED_PER_MIN_PER_UVI)
                       if minutes > 0 and ambient_uv_index > 0 else 0.0)

    when_utc = when if when.tzinfo is not None else when.replace(
        tzinfo=dt.timezone(dt.timedelta(hours=utc_offset_hours or 0.0)))
    timestamp_utc = when_utc.astimezone(dt.timezone.utc).isoformat()

    return ExposureResult(
        latitude=latitude, longitude=longitude, timestamp_utc=timestamp_utc,
        minutes=minutes, geometry=geometry, solar=solar,
        ambient_uv_index=ambient_uv_index, geometry_factor=geometry_factor,
        sed=sed,
    )


def exposure_sed_live(latitude: float, longitude: float, when: dt.datetime,
                      minutes: float, *,
                      sites: Optional[Iterable[str]] = None,
                      preset: Optional[str] = None,
                      posture: str = "standing_erect_arms_down",
                      timeout: float = 10.0,
                      http_get: Optional[Callable[[str, float], str]] = None
                      ) -> ExposureResult:
    """Convenience wrapper: fetches ambient UVI via vitd_ambient.fetch_ambient_uvi()
    (live network call unless `http_get` is injected -- see that module's
    "TESTING WITHOUT NETWORK" section, and tests/test_exposure.py, which
    reuses the SAME recorded fixtures as tests/test_ambient.py) and then
    calls exposure_sed(). Raises AmbientUVIUnavailable on any fetch failure
    (see vitd_ambient.py) -- never substitutes a guessed UVI.
    """
    ambient: AmbientUVI = fetch_ambient_uvi(latitude, longitude, when,
                                            timeout=timeout, http_get=http_get)
    return exposure_sed(latitude, longitude, when, minutes,
                        ambient.effective_uv_index,
                        sites=sites, preset=preset, posture=posture)


# ---------------------------------------------------------------------------
# 7. Seasonal variation -- OPTIONAL, caller-supplied only. See module
#    docstring point 2 below for why this repo never invents this table.
# ---------------------------------------------------------------------------
def exposure_across_months(latitude: float, longitude: float, year: int,
                           monthly_uv_index: Dict[int, float], minutes: float,
                           *, sites: Optional[Iterable[str]] = None,
                           preset: Optional[str] = None,
                           posture: str = "standing_erect_arms_down",
                           utc_offset_hours: Optional[float] = None,
                           day_of_month: int = 15
                           ) -> Dict[int, ExposureResult]:
    """One ExposureResult per month in `monthly_uv_index` (keys 1-12), all at
    `day_of_month` of `year` -- the caller-supplied seasonal extension of
    exposure_sed(). Requires a real monthly UV Index PER MONTH from the
    caller (e.g. a live vitd_ambient.fetch_ambient_uvi()/OPEN_METEO_HISTORICAL_URL
    fetch for each month, or a transcribed printed climatology). This module
    NEVER invents that table itself -- see module docstring "WHAT THIS MODULE
    DOES NOT DO" -- and no caller in this repo currently supplies one:
    run_all.py's own report skips this function entirely rather than pass a
    fabricated profile (see run_all.py section 11d).

    The SAME `sites`/`preset`/`posture` -- hence the SAME `exposed_bsa` -- is
    used for every month (a person's chosen clothing doesn't vary by this
    function's own logic; a caller modelling seasonal clothing changes would
    call exposure_sed() directly per month with a different preset).
    `geometry_factor` DOES vary month to month even though clothing does not,
    because solar elevation genuinely changes with `when` and Vernez's
    regression has a (small) solar-zenith-angle term -- see exposure_sed().

    NOT MODELLED: this only varies ambient UV Index INTENSITY by month. The
    solar E_source (vitamin-D spectral effectiveness) this repo uses is a
    single Manchester-summer value applied year-round -- see
    calibration_approval.yaml parameter_approvals.E_SOURCE_SOLAR_DEFAULT.not_covered.
    A caller combining this function's output with vitd_model.simulate()
    inherits that limitation; this function does not fix it.

    Each month is computed INDEPENDENTLY (same as calling exposure_sed() once
    per month) -- there is no cross-month interaction here or in any
    downstream schedule function that consumes this output.

    Raises
    ------
    ValueError if `monthly_uv_index` is empty.
    KeyError if any key of `monthly_uv_index` is outside 1-12.
    """
    if not monthly_uv_index:
        raise ValueError("monthly_uv_index must be non-empty")
    bad_months = sorted(m for m in monthly_uv_index if not (1 <= m <= 12))
    if bad_months:
        raise KeyError(f"monthly_uv_index has invalid month keys: {bad_months}; "
                       "expected integers 1-12")

    out: Dict[int, ExposureResult] = {}
    for month, uvi in sorted(monthly_uv_index.items()):
        when = dt.datetime(year, month, day_of_month, 12, 0)
        out[month] = exposure_sed(latitude, longitude, when, minutes, uvi,
                                  sites=sites, preset=preset, posture=posture,
                                  utc_offset_hours=utc_offset_hours)
    return out
