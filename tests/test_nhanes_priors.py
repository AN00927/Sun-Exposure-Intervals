"""
test_nhanes_priors.py
=====================
Tests for NHANES empirical baseline priors and scenario switchable configuration.
"""

import os
import pytest
import pandas as pd
import vitd_scenario


def test_nhanes_priors_table_exists():
    """Verify that tables/nhanes_baseline_priors.csv exists and is well-formed."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    priors_csv = os.path.join(repo_root, "tables", "nhanes_baseline_priors.csv")
    assert os.path.exists(priors_csv), "Missing tables/nhanes_baseline_priors.csv"
    
    df = pd.read_csv(priors_csv)
    assert not df.empty
    expected_cols = {"strata_type", "strata_value", "season", "dietary_status", "p50", "p25", "p75"}
    assert expected_cols.issubset(set(df.columns))


def test_get_nhanes_baseline_d0():
    """Verify that empirical baseline D0 values match expected ranges."""
    # Fair skin (FST I/II) in winter unsupplemented should be ~55-75 nmol/L
    d0_fst2 = vitd_scenario.get_nhanes_baseline_D0(2, season="winter_spring")
    assert 50.0 <= d0_fst2 <= 80.0
    
    # Dark skin (FST VI) in winter unsupplemented should be lower (~30-45 nmol/L)
    d0_fst6 = vitd_scenario.get_nhanes_baseline_D0(6, season="winter_spring")
    assert 25.0 <= d0_fst6 <= 45.0
    assert d0_fst2 > d0_fst6


def test_switchable_scenario_baseline():
    """Test switching baseline_source between preset and nhanes."""
    # Preset default for summer_midday_reference is 25.0
    sc_preset = vitd_scenario.get_scenario("summer_midday_reference", baseline_source="preset")
    assert sc_preset.baseline_D0 == 25.0
    
    # NHANES baseline for FST 3
    sc_nhanes = vitd_scenario.get_scenario("summer_midday_reference", baseline_source="nhanes")
    expected_nhanes_d0 = vitd_scenario.get_nhanes_baseline_D0(sc_preset.target_fst)
    assert sc_nhanes.baseline_D0 == expected_nhanes_d0
    assert sc_nhanes.baseline_D0 != 25.0


def test_set_default_baseline_source():
    """Test global default baseline source setting."""
    try:
        vitd_scenario.set_default_baseline_source("nhanes")
        sc = vitd_scenario.get_scenario("summer_midday_reference")
        assert sc.baseline_D0 == vitd_scenario.get_nhanes_baseline_D0(sc.target_fst)
        
        vitd_scenario.set_default_baseline_source("preset")
        sc = vitd_scenario.get_scenario("summer_midday_reference")
        assert sc.baseline_D0 == 25.0
    finally:
        vitd_scenario.set_default_baseline_source("preset")
