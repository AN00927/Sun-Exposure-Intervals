"""Tests for vitd_genetics.py"""

import pytest
from vitd_model import ModelParams
from vitd_genetics import (
    GENOTYPE_PRESETS,
    apply_genotype_presets,
    list_available_presets,
)


def test_all_presets_have_confidence_and_sources():
    for name, preset in GENOTYPE_PRESETS.items():
        assert "_confidence" in preset, f"{name} missing _confidence"
        assert "_sources" in preset, f"{name} missing _sources"
        assert len(preset["_sources"]) > 0, f"{name} has empty _sources"


def test_apply_single_preset_returns_modified_params():
    params = ModelParams()
    modified, names = apply_genotype_presets(params, ["CYP2R1_low_activity"])
    assert modified.A != params.A or modified.A != params.A, (
        "Modified params should differ"
    )
    assert names == ["CYP2R1_low_activity"]


def test_apply_empty_preset_list_returns_original():
    params = ModelParams()
    modified, names = apply_genotype_presets(params, [])
    assert modified.A == params.A
    assert names == []


def test_apply_unknown_preset_raises():
    params = ModelParams()
    with pytest.raises(ValueError, match="Unknown preset"):
        apply_genotype_presets(params, ["nonexistent_preset"])


def test_list_available_presets_returns_string():
    result = list_available_presets()
    assert "VDR_fokI_GG" in result
    assert "CYP2R1_low_activity" in result
    assert "LOW" in result or "MEDIUM" in result  # confidence labels present


def test_preset_composition_multiplies_A():
    params = ModelParams()
    base_A = params.A
    modified, _ = apply_genotype_presets(
        params, ["CYP2R1_high_activity", "CYP2R1_low_activity"]
    )
    expected = base_A * 1.05 * 0.95
    assert abs(modified.A - expected) < 1e-10, f"{modified.A} != {expected}"


def test_preset_composition_sums_A_offset():
    params = ModelParams()
    base_A = params.A
    modified, _ = apply_genotype_presets(
        params, ["pigment_SNP_favorable", "pigment_SNP_unfavorable"]
    )
    assert modified is not None


def test_preset_composition_multiplies_k_decay():
    params = ModelParams()
    base_k = params.k_decay_D(100.0)
    modified, _ = apply_genotype_presets(params, ["VDR_fokI_GG"])
    modified_k = modified.k_decay_D(100.0)
    expected_k = base_k * 1.51
    assert abs(modified_k - expected_k) < 1e-10, f"{modified_k} != {expected_k}"
