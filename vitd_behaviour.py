"""
vitd_behaviour.py
=================
BEHAVIOURAL exposure, separate from the optical model.

A multiplier on delivered UV dose: how much of the available sun a person
actually takes via shade-seeking, clothing cover, and time outdoors. NOT
optics. Nothing here describes melanin, skin, or photochemistry.

Why separate, and why default 1.0
---------------------------------
This repository tells people when to go outside. When the model says
"25 minutes at 12:30", the user must change behaviour: go out, stand in sun
rather than shade, uncover skin. Any behavioural factor folded into a
pigment constant would apply to someone who has, by construction, just
stopped behaving that way -- and apply strongest to dark-skinned users, who
are exactly who this project serves.

  * SCHEDULING mode (the default): factor is 1.0 for every skin type. Model
    output is bit-identical to a build without this module.
  * COHORT mode: switched on to reproduce measured behaviour of a named
    field cohort, explaining observed population data.

What the field evidence says
-----------------------------
The behaviour/optics split comes from `scripts/fst_gap_decomposition.py`,
recorded in `calibration_approval.yaml` (`a1_diagnosis_result_2026_08_17`):

  * Behaviour accounts for 2.1-2.4x of the observed white : FST V 25(OH)D
    response at UK latitudes -- Kift 2013 2.44x, Darling 2013 2.13x.
  * SHADE AND CLOTHING, not less time outdoors. Kift's South Asian cohort
    spent MORE time outdoors than its white comparator (69 vs 63 min/day;
    11 vs 10 min/day around midday) and still received 0.9-1.1% of ambient
    UV against the white cohort's 2.2-2.9%.
  * Not posture geometry. Every unshaded sky-visibility value in Vernez 2015
    Table S1 is tens of percent; both Kift cohorts sit one to two ORDERS OF
    MAGNITUDE below that. The badges record shade and cover, not body
    orientation.
  * The remaining optical part is carried by `ModelParams.K_const`
    (recalibrated 2026-08-17 to 0.98). The two are multiplicative and must
    not merge: dumping behaviour into K_const would require an optical factor
    of 4.3-5.7x, above every matched-dose measurement in existence.

DARK-SKIN PROVENANCE -- true of every number in this module
-----------------------------------------------------------
Both field studies are UK, both dark cohorts are FST V of Indian-subcontinent
ancestry, both use polysulphone badges. No personal-dosimetry measurement of
FST VI, African-ancestry populations, or any population outside northern
Europe exists in this repository. Do not extrapolate to FST VI or other
latitudes; `cohort_factor()` refuses to invent one.

Cross-references
----------------
  * derivation of the split      scripts/fst_gap_decomposition.py  (S2, S4)
  * optical half of the split    ModelParams.K_const in vitd_model.py
  * approval                     calibration_approval.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

#: SCHEDULING mode: factor is 1.0 for every skin type. The model prescribes
#: exposure, so it cannot assume the user behaves like a population average.
MODE_SCHEDULING = "scheduling"

#: COHORT mode: reproduce the measured behaviour of a named field cohort.
#: Requires an explicit `cohort` -- no "average person" fallback.
MODE_COHORT = "cohort"

MODES = (MODE_SCHEDULING, MODE_COHORT)

#: The value returned in scheduling mode, for every skin type, always.
NEUTRAL_FACTOR: float = 1.0


# ---------------------------------------------------------------------------
# Measured cohorts
# ---------------------------------------------------------------------------
# Every field below is a measurement transcribed from a source held in this
# repo, or marked otherwise. Reproduced by scripts/fst_gap_decomposition.py
# section S2; see that script's S6 assumption register and S8 per-number
# audit for transcription caveats (Kift 2013's PDF text layer drops decimal
# points, and Darling 2013 itself is not in the repo -- its numbers come via
# Webb 2021 Table 5).

@dataclass(frozen=True)
class Cohort:
    """One measured population's UV-taking behaviour."""
    key: str
    label: str
    fst: int
    n: int
    #: Personal erythemal dose from polysulphone badges, SED per summer week.
    sed_summer_week: float
    #: Personal dose as a percentage of AMBIENT UV over the same period.
    #: None where the source does not report it.
    pct_of_ambient: Optional[Tuple[float, float]]
    #: Minutes per day spent outdoors, from sun-exposure diaries. None where
    #: the source does not report it.
    minutes_outdoors_per_day: Optional[float]
    #: Fraction of body surface left uncovered, from the same diaries.
    exposed_bsa: Optional[float]
    source: str


#: Each pair's entries come from one study, one protocol, one badge type.
#: That makes the within-pair ratio a behavioural measurement, not a
#: comparison of two dosimetry methods.
COHORTS: Dict[str, Cohort] = {
    "kift2013_white": Cohort(
        key="kift2013_white", label="Webb 2010 white, Manchester 53.5N",
        fst=2, n=109, sed_summer_week=3.4, pct_of_ambient=(2.23, 2.94),
        minutes_outdoors_per_day=63.0, exposed_bsa=0.155,
        source="Webb 2010 BJD 163:1050 via Webb 2021 Table 5; Kift 2013 Table 2"),
    "kift2013_south_asian": Cohort(
        key="kift2013_south_asian",
        label="Kift 2013 South Asian FST V, Manchester 53.5N",
        fst=5, n=125, sed_summer_week=1.0, pct_of_ambient=(0.88, 1.11),
        minutes_outdoors_per_day=69.0, exposed_bsa=0.14,
        source="Kift 2013 BJD 169:1272, Table 2 and Results"),
    "darling2013_white": Cohort(
        key="darling2013_white", label="Darling 2013 Caucasian F, Surrey 51.3N",
        fst=2, n=90, sed_summer_week=6.39, pct_of_ambient=None,
        minutes_outdoors_per_day=None, exposed_bsa=None,
        source="Darling 2013 Osteoporos Int 24:477 via Webb 2021 Table 5"),
    "darling2013_south_asian": Cohort(
        key="darling2013_south_asian",
        label="Darling 2013 South Asian F, Surrey 51.3N",
        fst=5, n=35, sed_summer_week=2.00, pct_of_ambient=None,
        minutes_outdoors_per_day=None, exposed_bsa=None,
        source="Darling 2013 Osteoporos Int 24:477 via Webb 2021 Table 5"),
}

#: Light-skinned comparator for each study pair. Behaviour is only measured
#: relative to a comparator on the same protocol.
COHORT_PAIRS: Dict[str, Tuple[str, str]] = {
    "kift2013": ("kift2013_white", "kift2013_south_asian"),
    "darling2013": ("darling2013_white", "darling2013_south_asian"),
}

#: Each pair's measured dose difference as a share of the modelled 25(OH)D
#: response ratio -- after the model's own feedback and BSA scaling. Derived,
#: not transcribed: the ratio of behaviour-blind to behaviour-aware K_const
#: inversion in scripts/fst_gap_decomposition.py S4, run 2026-08-17.
BEHAVIOUR_SHARE_ON_RESPONSE: Dict[str, float] = {
    "kift2013": 2.44,
    "darling2013": 2.13,
}


# ---------------------------------------------------------------------------
# The factor
# ---------------------------------------------------------------------------

def _study_of(cohort_key: str) -> str:
    """Which study pair a cohort belongs to."""
    for study, members in COHORT_PAIRS.items():
        if cohort_key in members:
            return study
    raise ValueError(f"cohort {cohort_key!r} belongs to no study pair")


def _default_reference(cohort_key: str) -> str:
    """The light-skinned comparator measured on the same protocol."""
    return COHORT_PAIRS[_study_of(cohort_key)][0]


def behaviour_factor(mode: str = MODE_SCHEDULING,
                     cohort: Optional[str] = None,
                     reference: Optional[str] = None) -> float:
    """Multiplier on delivered UV dose. 1.0 unless explicitly asked otherwise.

    Parameters
    ----------
    mode : str
        ``MODE_SCHEDULING`` (default) returns ``NEUTRAL_FACTOR`` = 1.0.
        Scheduling mode tells the user to go outside, so it cannot assume
        they behave like the population average that stayed in.
        ``MODE_COHORT`` returns a measured cohort's badge dose relative to a
        reference cohort on the same protocol.
    cohort : str, optional
        Key into ``COHORTS``. Required in cohort mode.
    reference : str, optional
        Key into ``COHORTS`` for the comparator. Defaults to the light-skinned
        member of the same study pair. Must come from the same study: badge
        type, placement protocol, latitude and season must match or the ratio
        measures dosimetry, not behaviour.

    Returns
    -------
    float
        Dose multiplier. 1.0 means "takes as much sun as the reference".

    Raises
    ------
    ValueError
        On an unknown mode or cohort, a missing cohort in cohort mode, or a
        cross-study reference pairing.
    """
    if mode == MODE_SCHEDULING:
        if cohort is not None or reference is not None:
            raise ValueError(
                "behaviour_factor(): a cohort was supplied in scheduling mode. "
                "Scheduling mode is deliberately behaviour-free -- pass "
                f"mode={MODE_COHORT!r} if you mean to model observed "
                "population behaviour.")
        return NEUTRAL_FACTOR

    if mode != MODE_COHORT:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if cohort is None:
        raise ValueError(
            f"mode={MODE_COHORT!r} needs an explicit cohort. There is no "
            "'average person' here: the only behavioural measurements in this "
            f"repo are the named cohorts {sorted(COHORTS)}.")
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}; known: {sorted(COHORTS)}")

    if reference is None:
        reference = _default_reference(cohort)
    if reference not in COHORTS:
        raise ValueError(f"unknown reference {reference!r}")
    if _study_of(cohort) != _study_of(reference):
        raise ValueError(
            f"cohort {cohort!r} and reference {reference!r} come from "
            "different studies. A behavioural ratio is only meaningful within "
            "one study: badge type, placement protocol, latitude and season "
            "must be shared, or the ratio measures dosimetry, not behaviour.")
    return COHORTS[cohort].sed_summer_week / COHORTS[reference].sed_summer_week


def cohort_factor(fst: int, study: str) -> float:
    """Behavioural factor for the cohort of skin type `fst` in `study`.

    Convenience wrapper over :func:`behaviour_factor`. Raises rather than
    interpolates: no measured FST VI, FST I, III or IV cohort exists in this
    repo, and inventing one is the failure this module/parameter split
    exists to prevent.
    """
    if study not in COHORT_PAIRS:
        raise ValueError(f"unknown study {study!r}; known: {sorted(COHORT_PAIRS)}")
    light, dark = COHORT_PAIRS[study]
    for key in (light, dark):
        if COHORTS[key].fst == fst:
            return behaviour_factor(MODE_COHORT, cohort=key, reference=light)
    raise ValueError(
        f"{study} measured no FST {fst} cohort. Measured skin types are "
        f"{sorted({COHORTS[k].fst for k in COHORT_PAIRS[study]})}. Every field "
        "dosimetry cohort in this repo is FST II or FST V, UK, South Asian "
        "ancestry for the FST V arm -- there is no FST VI measurement and no "
        "non-UK measurement to interpolate from.")


def measured_dose_ratio(study: str) -> float:
    """Measured light : dark personal-UV DOSE ratio for one study pair.

    Raw badge-dose ratio. NOT the effect on serum 25(OH)D -- see
    :data:`BEHAVIOUR_SHARE_ON_RESPONSE` for that, which is smaller because
    feedback and sub-linear BSA scaling compress it.
    """
    if study not in COHORT_PAIRS:
        raise ValueError(f"unknown study {study!r}; known: {sorted(COHORT_PAIRS)}")
    light, dark = COHORT_PAIRS[study]
    return COHORTS[light].sed_summer_week / COHORTS[dark].sed_summer_week


def apply_to_schedule(schedule: Iterable[Tuple[float, float]],
                      factor: float) -> List[Tuple[float, float]]:
    """Scale every session's SED by `factor`, leaving the timing untouched.

    Behaviour changes the DOSE TAKEN, not when the sun is up, so it multiplies
    the SED of each pulse and never moves a pulse. With the default
    ``NEUTRAL_FACTOR`` this returns the schedule unchanged (x * 1.0 == x).
    """
    if factor < 0.0:
        raise ValueError(f"factor must be >= 0, got {factor}")
    return [(float(t), float(sed) * factor) for t, sed in schedule]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def describe(study: str) -> str:
    """Human-readable provenance for one study pair. For report footers."""
    if study not in COHORT_PAIRS:
        raise ValueError(f"unknown study {study!r}; known: {sorted(COHORT_PAIRS)}")
    light, dark = (COHORTS[k] for k in COHORT_PAIRS[study])
    out = [f"{study}: measured behavioural exposure, personal polysulphone "
           f"dosimetry"]
    for c in (light, dark):
        bits = [f"    FST {c.fst}  n={c.n:<4} {c.sed_summer_week:.2f} "
                f"SED/summer week"]
        if c.pct_of_ambient is not None:
            bits.append(f"{c.pct_of_ambient[0]:.2f}-{c.pct_of_ambient[1]:.2f}% "
                        f"of ambient")
        if c.minutes_outdoors_per_day is not None:
            bits.append(f"{c.minutes_outdoors_per_day:.0f} min/day outdoors")
        out.append("   ".join(bits))
        out.append(f"      {c.label}  [{c.source}]")
    out.append(f"    light:dark dose ratio       "
               f"{measured_dose_ratio(study):.2f}x")
    if study in BEHAVIOUR_SHARE_ON_RESPONSE:
        out.append(f"    share of the 25(OH)D gap    "
                   f"{BEHAVIOUR_SHARE_ON_RESPONSE[study]:.2f}x")
    out.append("    This is SHADE AND CLOTHING, not less time outdoors and "
               "not posture.")
    out.append("    FST V arm is South Asian ancestry at UK latitude. No FST "
               "VI cohort exists.")
    return "\n".join(out)


if __name__ == "__main__":   # pragma: no cover - manual inspection aid
    print(__doc__)
    print(f"scheduling-mode factor (the default): "
          f"{behaviour_factor():.3f}\n")
    for study in COHORT_PAIRS:
        print(describe(study))
        print()
