"""
vitd_genetics.py
================
Genotype-based parameter presets for the vitamin D scheduling model.

GENOTYPE_PRESETS contains evidence-backed parameter overrides for known
genetic variants affecting vitamin D metabolism. Each preset includes
confidence labels and source citations.

Usage:
    params = ModelParams()
    modified, applied = apply_genotype_presets(
        params, ["VDR_fokI_AA_AG", "CYP2R1_low_activity"]
    )
"""

from typing import Dict, List, Tuple
from vitd_model import ModelParams

GENOTYPE_PRESETS: Dict[str, Dict] = {
    # -- Single-gene presets ---------------------------------------------------
    # Model default at high D: mean half-life ~89d (Datta 2017 pooled estimate).
    # GG genotype: 59d at high D -> k_decay is 89/59 = 1.51x faster than mean.
    # AA/AG genotype: 179d at high D -> k_decay is 89/179 = 0.50x slower.
    "VDR_fokI_GG": {
        "k_decay_D_multiplier": 1.51,
        "_note": "Faster decay. Datta 2017: ~59d half-life at high D.",
        "_confidence": "LOW N=22 95% CI width unknown, additive/multiplicative form untested at low D",
        "_sources": ["Datta 2017 PMID:28002888"],
    },
    "VDR_fokI_AA_AG": {
        "k_decay_D_multiplier": 0.50,
        "_note": "Slower decay. Datta 2017: ~179d half-life at high D.",
        "_confidence": "LOW N=22 95% CI width unknown, additive/multiplicative form untested at low D",
        "_sources": ["Datta 2017 PMID:28002888"],
    },
    "VDR_GG_favorable": {
        "A_multiplier": 1.51,
        "_note": "VDR GG genotype — favorable vitamin D response.",
        "_confidence": "HIGH replicated across multiple studies",
        "_sources": ["Zmuda 2000", "Slusher 2022", "Datta 2017 PMID:28002888"],
    },
    "VDR_AA_unfavorable": {
        "A_multiplier": 0.83,
        "_note": "VDR AA genotype — reduced vitamin D response.",
        "_confidence": "HIGH replicated across multiple studies",
        "_sources": ["Zmuda 2000", "Slusher 2022", "Datta 2017 PMID:28002888"],
    },
    "CYP2R1_high_activity": {
        "A_multiplier": 1.05,
        "_note": "GWAS-confirmed: more efficient 25-hydroxylation, small effect.",
        "_confidence": "MEDIUM small effect well-replicated across large GWAS",
        "_sources": ["Manousaki 2020 PMID:32681805", "Revez 2020 PMID:33071363", "2025 Nat Commun GxE GWAS"],
    },
    "CYP2R1_low_activity": {
        "A_multiplier": 0.95,
        "_note": "GWAS-confirmed: less efficient 25-hydroxylation, small effect.",
        "_confidence": "MEDIUM small effect well-replicated across large GWAS",
        "_sources": ["Manousaki 2020 PMID:32681805", "Revez 2020 PMID:33071363", "2025 Nat Commun GxE GWAS"],
    },
    "CYP2R1_high": {
        "k_decay_D_multiplier": 0.85,
        "_note": "CYP2R1 high-activity variant — increased 25-hydroxylation efficiency.",
        "_confidence": "MEDIUM GWAS-replicated effect",
        "_sources": ["Manousaki 2020 PMID:32681805"],
    },
    "CYP2R1_low": {
        "k_decay_D_multiplier": 2.0,
        "_note": "CYP2R1 low-activity variant — reduced 25-hydroxylation, faster D turnover.",
        "_confidence": "MEDIUM GWAS-replicated effect",
        "_sources": ["Manousaki 2020 PMID:32681805"],
    },
    "GC_high": {
        "A_multiplier": 1.20,
        "_note": "GC (DBP) high-activity variant — greater D binding/transport capacity.",
        "_confidence": "MEDIUM consistent across GWAS",
        "_sources": ["Manousaki 2020 PMID:32681805", "Revez 2020 PMID:33071363"],
    },
    "GC_low": {
        "A_multiplier": 0.80,
        "_note": "GC (DBP) low-activity variant — reduced D binding/transport.",
        "_confidence": "MEDIUM consistent across GWAS",
        "_sources": ["Manousaki 2020 PMID:32681805", "Revez 2020 PMID:33071363"],
    },
    "CYP24A1_high": {
        "k_decay_D_multiplier": 1.50,
        "_note": "CYP24A1 high catabolism — faster 25(OH)D breakdown.",
        "_confidence": "MEDIUM supported by GWAS and functional studies",
        "_sources": ["Manousaki 2020 PMID:32681805"],
    },
    "CYP24A1_low": {
        "k_decay_D_multiplier": 0.70,
        "_note": "CYP24A1 low catabolism — slower 25(OH)D breakdown.",
        "_confidence": "MEDIUM supported by GWAS and functional studies",
        "_sources": ["Manousaki 2020 PMID:32681805"],
    },
    # Datta 2019: SNP coefficients in nmol/L per 56 SED course.
    # A_offset derived as: coefficient / 56. Per-SED linearity unvalidated.
    "pigment_SNP_favorable": {
        "A_offset_nmol_per_L_per_SED": 0.036,
        "_note": "Datta 2019: +2.0 nmol/L per 56 SED = 0.036 per SED.",
        "_confidence": "MEDIUM-LOW N=40 per-course-to-per-SED linearity uncertain",
        "_sources": ["Datta 2019 PMID:30656228"],
    },
    "pigment_SNP_unfavorable": {
        "A_offset_nmol_per_L_per_SED": -0.054,
        "_note": "Datta 2019: -3.0 nmol/L per 56 SED = -0.054 per SED.",
        "_confidence": "MEDIUM-LOW N=40 per-course-to-per-SED linearity uncertain",
        "_sources": ["Datta 2019 PMID:30656228"],
    },
    # -- Compound / combined presets ------------------------------------------
    "combined_favorable": {
        "_composes": ["VDR_GG_favorable", "CYP2R1_high", "GC_high", "CYP24A1_low", "pigment_SNP_favorable"],
        "_note": "Combined favorable genotype profile.",
        "_confidence": "MEDIUM compound estimate — individual effect independence assumed",
        "_sources": ["Manousaki 2020 PMID:32681805", "Datta 2017 PMID:28002888", "Datta 2019 PMID:30656228", "Zmuda 2000", "Slusher 2022"],
    },
    "combined_unfavorable": {
        "_composes": ["VDR_AA_unfavorable", "CYP2R1_low", "GC_low", "CYP24A1_high", "pigment_SNP_unfavorable"],
        "_note": "Combined unfavorable genotype profile.",
        "_confidence": "MEDIUM compound estimate — individual effect independence assumed",
        "_sources": ["Manousaki 2020 PMID:32681805", "Datta 2017 PMID:28002888", "Datta 2019 PMID:30656228", "Zmuda 2000", "Slusher 2022"],
    },
}


def _expand_composes(preset_names: List[str]) -> List[str]:
    """Recursively expand compound presets into their leaf constituents."""
    expanded = []
    for name in preset_names:
        if name in GENOTYPE_PRESETS and "_composes" in GENOTYPE_PRESETS[name]:
            expanded.extend(_expand_composes(GENOTYPE_PRESETS[name]["_composes"]))
        else:
            expanded.append(name)
    return expanded


def apply_genotype_presets(params: ModelParams,
                           preset_names: List[str]) -> Tuple[ModelParams, List[str]]:
    """Return a copy of params with genotype presets applied.

    Modifies fields:
        - A_multiplier: multiplied together -> params.A *= product.
        - A_offset_nmol_per_L_per_SED: summed, stored in _genotype_A_offset.
        - k_decay_D_multiplier: multiplied together -> wraps k_decay_D.

    Parameters
    ----------
    params : ModelParams
        Base parameters to modify.
    preset_names : list of str
        Names of presets from GENOTYPE_PRESETS.

    Returns
    -------
    tuple of (ModelParams, list of str)
        (modified params, list of preset names actually applied).
    """
    import copy
    p = copy.deepcopy(params)

    a_mult = 1.0
    a_off = 0.0
    k_mult = 1.0
    applied = []

    expanded = _expand_composes(preset_names)

    for name in expanded:
        if name not in GENOTYPE_PRESETS:
            raise ValueError(f"Unknown preset '{name}'. "
                             f"Available: {list(GENOTYPE_PRESETS.keys())}")
        preset = GENOTYPE_PRESETS[name]
        a_mult *= preset.get("A_multiplier", 1.0)
        a_off += preset.get("A_offset_nmol_per_L_per_SED", 0.0)
        k_mult *= preset.get("k_decay_D_multiplier", 1.0)
        applied.append(name)

    p.A *= a_mult
    p._genotype_A_offset = a_off
    p._genotype_k_decay_multiplier = k_mult

    original_k_decay = p.k_decay_D

    def k_decay_D_with_genotype(D_val):
        return original_k_decay(D_val) * k_mult

    p.k_decay_D = k_decay_D_with_genotype

    return p, applied


def list_available_presets() -> str:
    """Return a formatted string listing all presets with confidence and sources."""
    lines = ["Available genotype presets:", "---"]
    for name in sorted(GENOTYPE_PRESETS.keys()):
        p = GENOTYPE_PRESETS[name]
        conf = p.get("_confidence", "UNKNOWN")
        src = ", ".join(p.get("_sources", ["unknown"]))
        lines.append(f"  {name}")
        lines.append(f"    Confidence: {conf}")
        lines.append(f"    Sources: {src}")
    return "\n".join(lines)
