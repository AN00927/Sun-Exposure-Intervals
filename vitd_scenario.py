"""
vitd_scenario.py
================
Scenario architecture for UV exposure protocols.

Provides a frozen ``ExposureScenario`` dataclass encapsulating all exposure
parameters and clinical study metadata, a ``SCENARIO_REGISTRY`` of named
presets, helper functions for retrieval and simulation, and backward-
compatible aliases for code that still imports the old frozen-reference
constants.
"""

from __future__ import annotations

import os
import functools
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from vitd_model import BSA_REF, build_pulse_schedule, get_E_source
from vitd_model import ModelParams, simulate
from vitd_physiology import cancer_risk
from vitd_objectives import PERSONAL_DOSE_GEOMETRY_FACTOR, SED_PER_MIN_PER_UVI


# ---------------------------------------------------------------------------
# ExposureScenario dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExposureScenario:
    """A frozen description of one UV exposure environment and protocol.

    Immutable and hashable so instances can be dict keys and set members.
    """

    name: str
    description: str
    study_citation: str
    pmid_or_doi: str
    uvi: float = 6.0
    exposed_bsa: float = 0.35
    geometry_factor: float = 0.35
    e_source_name: str = "solar_summer"
    target_fst: int = 3
    baseline_D0: float = 25.0
    protocol_duration_days: int = 84
    protocol_schedule: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    observed_delta_D: Optional[Union[float, Tuple[float, float, float]]] = None

    @property
    def sed_per_minute(self) -> float:
        """Erythemal SED delivered per minute under this scenario's UVI and geometry."""
        return SED_PER_MIN_PER_UVI * self.uvi * self.geometry_factor

    def minutes_for_sed(self, sed: float) -> float:
        """Minutes needed to deliver *sed* under this scenario."""
        rate = self.sed_per_minute
        return max(0.0, float(sed)) / rate if rate > 0 else float("inf")

    def sed_for_minutes(self, minutes: float) -> float:
        """Erythemal SED delivered by *minutes* of exposure under this scenario."""
        return max(0.0, float(minutes)) * self.sed_per_minute


# ---------------------------------------------------------------------------
# Schedule builder helper
# ---------------------------------------------------------------------------

def _build_weekly_protocol(
    duration_days: int,
    sed_per_session: float,
    days_of_week: Tuple[int, ...] = (0, 2, 4),
) -> Tuple[Tuple[float, float], ...]:
    """Build a repeating weekly protocol as a tuple of (day, SED) pairs."""
    return tuple(
        (float(w * 7 + d), float(sed_per_session))
        for w in range(duration_days // 7)
        for d in days_of_week
        if w * 7 + d < duration_days
    )


# ---------------------------------------------------------------------------
# SCENARIO_REGISTRY: 7 named presets
# ---------------------------------------------------------------------------

SCENARIO_REGISTRY: Dict[str, ExposureScenario] = {
    "summer_midday_reference": ExposureScenario(
        name="summer_midday_reference",
        description="Standard summer sun reference scenario",
        study_citation="Reference conditions for the minutes column",
        pmid_or_doi="N/A (reference)",
        uvi=6.0,
        exposed_bsa=0.35,
        geometry_factor=0.35,
        e_source_name="solar_summer",
        target_fst=3,
        baseline_D0=25.0,
        protocol_duration_days=84,
        protocol_schedule=_build_weekly_protocol(84, 1.0),
        observed_delta_D=None,
    ),
    "farrar2013_fst5_uk": ExposureScenario(
        name="farrar2013_fst5_uk",
        description="South Asian UK trial, FST V, 3x/wk NB-UVB",
        study_citation="Farrar et al. 2013, J Clin Endocrinol Metab 98:E526",
        pmid_or_doi="10.1210/jc.2012-3864",
        uvi=6.0,
        exposed_bsa=0.35,
        geometry_factor=0.35,
        e_source_name="solar_summer",
        target_fst=5,
        baseline_D0=17.2,
        protocol_duration_days=42,
        protocol_schedule=_build_weekly_protocol(42, 1.3),
        observed_delta_D=21.7,
    ),
    "young2021_fst2_solar": ExposureScenario(
        name="young2021_fst2_solar",
        description="Solar simulator full body, FST II, Arimed B",
        study_citation="Young et al. 2021, PNAS 118:e2015867118",
        pmid_or_doi="10.1073/pnas.2015867118",
        uvi=6.0,
        exposed_bsa=0.85,
        geometry_factor=0.35,
        e_source_name="arimed_b",
        target_fst=2,
        baseline_D0=34.6,
        protocol_duration_days=10,
        protocol_schedule=(
            (0.0, 2.0), (2.0, 2.0), (4.0, 2.0), (6.0, 2.0), (8.0, 2.0),
        ),
        observed_delta_D=36.1,
    ),
    "felton2016_fst2_matched": ExposureScenario(
        name="felton2016_fst2_matched",
        description="Matched-dose fair skin, FST II",
        study_citation="Felton et al. 2016, Br J Dermatol 175:1030-1040",
        pmid_or_doi="10.1111/bjd.14832",
        uvi=6.0,
        exposed_bsa=0.35,
        geometry_factor=0.35,
        e_source_name="solar_summer",
        target_fst=2,
        baseline_D0=36.5,
        protocol_duration_days=42,
        protocol_schedule=_build_weekly_protocol(42, 1.3),
        observed_delta_D=17.8,
    ),
    "felton2016_fst5_matched": ExposureScenario(
        name="felton2016_fst5_matched",
        description="Matched-dose dark skin, FST V",
        study_citation="Felton et al. 2016, Br J Dermatol 175:1030-1040",
        pmid_or_doi="10.1111/bjd.14832",
        uvi=6.0,
        exposed_bsa=0.35,
        geometry_factor=0.35,
        e_source_name="solar_summer",
        target_fst=5,
        baseline_D0=17.2,
        protocol_duration_days=42,
        protocol_schedule=_build_weekly_protocol(42, 1.3),
        observed_delta_D=8.3,
    ),
    "deboer2025_barrier": ExposureScenario(
        name="deboer2025_barrier",
        description="Barrier thickening trial, lower back, FST I-III",
        study_citation="de Boer et al. 2025",
        pmid_or_doi="TBD",
        uvi=6.0,
        exposed_bsa=0.10,
        geometry_factor=0.35,
        e_source_name="solar_summer",
        target_fst=3,
        baseline_D0=25.0,
        protocol_duration_days=63,
        protocol_schedule=_build_weekly_protocol(63, 0.8),
        observed_delta_D=None,
    ),
    "webb2018_uk_guidance": ExposureScenario(
        name="webb2018_uk_guidance",
        description="UK public health lunch-hour guidance reference",
        study_citation="Webb et al. 2018",
        pmid_or_doi="TBD",
        uvi=6.0,
        exposed_bsa=0.35,
        geometry_factor=0.35,
        e_source_name="solar_summer",
        target_fst=2,
        baseline_D0=25.0,
        protocol_duration_days=84,
        protocol_schedule=tuple((float(d), 0.5) for d in range(0, 84, 1)),
        observed_delta_D=None,
    ),
}


# ---------------------------------------------------------------------------
# Backward-compatible aliases (delegates to summer_midday_reference)
# ---------------------------------------------------------------------------

_DEFAULT = SCENARIO_REGISTRY["summer_midday_reference"]
UV_INDEX: float = _DEFAULT.uvi
GEOMETRY_FACTOR: float = _DEFAULT.geometry_factor
EXPOSED_BSA: float = _DEFAULT.exposed_bsa
SED_PER_MINUTE: float = _DEFAULT.sed_per_minute

CLOTHING_DESCRIPTION: str = (
    "t-shirt and shorts with partial coverage (face, neck, forearms, hands "
    "and part of the legs bare)"
)

LIMITATION_STATEMENT: str = (
    "All results assume a fixed ambient UV index of {uvi:g}, a "
    "horizontal-to-personal geometry factor of {geom:g}, and {bsa:.0%} exposed "
    "body surface area ({clothing}), held constant on every exposure day. Real "
    "ambient UV varies with latitude, season, time of day, altitude, cloud and "
    "ozone, and real personal dose varies with posture and clothing. We do not "
    "model that variation. Every duration reported here should therefore be "
    "read as a duration under these reference conditions, not as a calendar "
    "recommendation for a specific place or date."
).format(uvi=UV_INDEX, geom=GEOMETRY_FACTOR, bsa=EXPOSED_BSA,
         clothing=CLOTHING_DESCRIPTION)


def sed_for_minutes(minutes: float) -> float:
    """Erythemal SED delivered by *minutes* of exposure under the default scenario."""
    return _DEFAULT.sed_for_minutes(minutes)


def minutes_for_sed(sed: float) -> float:
    """Minutes needed to deliver *sed* erythemal SED under the default scenario."""
    return _DEFAULT.minutes_for_sed(sed)


def describe() -> str:
    """One-block human-readable summary, for report headers."""
    return (
        "FROZEN REFERENCE SCENARIO\n"
        f"  ambient UV index      : {UV_INDEX:g}\n"
        f"  geometry factor       : {GEOMETRY_FACTOR:g} "
        "(horizontal ambient -> personal, Cockell 2001)\n"
        f"  exposed BSA           : {EXPOSED_BSA:.0%} "
        f"({CLOTHING_DESCRIPTION})\n"
        f"  erythemal SED/minute  : {SED_PER_MINUTE:.4f} (derived)\n"
        "  held constant on every exposure day; see vitd_scenario.py"
    )


# ---------------------------------------------------------------------------
# NHANES Empirical Baseline Priors & Source Configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PRIORS_FILE = os.path.join(_REPO_ROOT, "tables", "nhanes_baseline_priors.csv")

BASELINE_SOURCES: Tuple[str, ...] = ("preset", "nhanes")
DEFAULT_BASELINE_SOURCE: str = "preset"

# Empirical fallback priors (unsupplemented winter/spring medians, nmol/L)
# from NHANES 2001-2023 standardized LC-MS/MS data:
NHANES_FALLBACK_BASELINE_D0: Dict[int, float] = {
    1: 56.8,
    2: 65.9,
    3: 59.6,
    4: 54.5,
    5: 50.2,
    6: 35.1,
}


@functools.lru_cache(maxsize=1)
def _load_priors_df() -> Optional[pd.DataFrame]:
    if os.path.exists(_PRIORS_FILE):
        try:
            return pd.read_csv(_PRIORS_FILE)
        except Exception:
            return None
    return None


def get_nhanes_baseline_D0(
    fst: int,
    season: str = "winter_spring",
    dietary_status: str = "all_unsupplemented",
    percentile: str = "p50",
) -> float:
    """Look up empirical survey-weighted baseline 25(OH)D from NHANES priors.
    
    Parameters
    ----------
    fst : int
        Fitzpatrick skin type (1 to 6).
    season : str
        'winter_spring' (default), 'summer_autumn', or 'all_year'.
    dietary_status : str
        'all_unsupplemented' (default), 'UV-only', 'food-only', 'supplemented', 'all'.
    percentile : str
        'p50' (median, default), 'p05', 'p10', 'p25', 'p75', 'p90', 'p95', or 'weighted_mean'.
    """
    df = _load_priors_df()
    if df is not None:
        sub = df[
            (df["strata_type"] == "fst_proxy") &
            (df["strata_value"] == f"FST_{fst}") &
            (df["season"] == season) &
            (df["dietary_status"] == dietary_status)
        ]
        if not sub.empty and percentile in sub.columns:
            val = sub.iloc[0][percentile]
            if pd.notna(val):
                return float(val)
    return NHANES_FALLBACK_BASELINE_D0.get(fst, 50.0)


def get_baseline_D0(
    fst: int,
    baseline_source: Optional[str] = None,
    default_preset: float = 25.0,
    season: str = "winter_spring",
) -> float:
    """Return baseline 25(OH)D according to the requested baseline source."""
    src = baseline_source or DEFAULT_BASELINE_SOURCE
    if src == "nhanes":
        return get_nhanes_baseline_D0(fst, season=season)
    return default_preset


def set_default_baseline_source(source: str) -> None:
    """Set global default baseline source ('preset' or 'nhanes')."""
    global DEFAULT_BASELINE_SOURCE
    if source not in BASELINE_SOURCES:
        raise ValueError(
            f"Invalid baseline source '{source}'. Must be one of {BASELINE_SOURCES}"
        )
    DEFAULT_BASELINE_SOURCE = source


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_scenario(name: str, baseline_source: Optional[str] = None) -> ExposureScenario:
    """Retrieve a scenario by name.
    
    If baseline_source is 'nhanes', the scenario's baseline_D0 is overridden
    with the empirical NHANES median for the target FST.
    """
    if name not in SCENARIO_REGISTRY:
        raise KeyError(
            f"Unknown scenario '{name}'. Available: {sorted(SCENARIO_REGISTRY)}"
        )
    sc = SCENARIO_REGISTRY[name]
    src = baseline_source or DEFAULT_BASELINE_SOURCE
    if src == "nhanes":
        nhanes_d0 = get_nhanes_baseline_D0(sc.target_fst)
        return replace(sc, baseline_D0=nhanes_d0)
    return sc


def list_scenarios() -> List[str]:
    """Return sorted list of all registered scenario names."""
    return sorted(SCENARIO_REGISTRY.keys())


def simulate_scenario(name: str) -> dict:
    """Run vitd_model.simulate with the scenario\'s parameters.

    Returns the full simulate() dict enriched with:
      - scenario_name: str
      - scenario_metadata: dict (citation, doi, observed_delta_D)
      - cancer risk keys promoted to top level for convenience
    """
    scenario = get_scenario(name)
    params = ModelParams(
        exposed_bsa=scenario.exposed_bsa,
        E_source=get_E_source(scenario.e_source_name),
    )
    schedule = list(scenario.protocol_schedule)
    result = simulate(
        fst=scenario.target_fst,
        schedule=schedule,
        duration_days=scenario.protocol_duration_days,
        params=params,
        D0=scenario.baseline_D0,
    )
    # Promote cancer risk keys to top level
    risk = result.get("cancer_risk", {})
    result["scc_index"] = risk.get("scc_index", 0.0)
    result["intermittent_index"] = risk.get("intermittent_index", 0.0)
    result["burn_fraction"] = risk.get("burn_fraction", 0.0)
    result["final_D"] = float(result["D"][-1])
    result["delta_D"] = float(result["D"][-1] - scenario.baseline_D0)
    # Add scenario metadata
    result["scenario_name"] = name
    result["scenario_metadata"] = {
        "citation": scenario.study_citation,
        "doi": scenario.pmid_or_doi,
        "observed_delta_D": scenario.observed_delta_D,
    }
    return result


if __name__ == "__main__":
    print(describe())
    print()
    print(LIMITATION_STATEMENT)
