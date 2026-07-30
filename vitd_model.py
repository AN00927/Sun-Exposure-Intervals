"""
vitd_model.py
=============
Three-state ODE model of serum 25(OH)D dynamics under UV exposure schedules,
accounting for photoadaptation (melanin upregulation and stratum corneum thickening)
in darker skin types.

State variables
---------------
D(t)  : effective serum 25(OH)D, nmol/L
        Single compartment representing the time-integrated plasma + adipose
        reservoir relevant for week-to-month scheduling dynamics.
M(t)  : effective melanin index, dimensionless 0-1
        (Datta 2019 reflectance scale; 0.0 = FST I, 1.0 = FST VI)
SC(t) : additional stratum corneum thickness, dimensionless (above baseline 0)
        (representing epidermal thickening/hyperplasia photoadaptation channel;
        [C8] this state is ACTIVE in the model -- SC is NOT unmodeled;
        remaining task is FST/source-specific calibration, not implementation)

Governing equations
-------------------
dD/dt  = S(UV(t), M(t), SC(t), D(t), F, bsa)  -  k_decay_D(D) * D
dM/dt  = alpha * UV(t) * (M_max - M)  -  beta * (M - M_baseline(F))
dSC/dt = alpha_SC * UV(t) * (SC_max - SC)  -  k_decay_SC * SC  -  k_damage * UV(t) * SC

Synthesis function (logarithmic dose-response in effective D3-weighted dose SED * E_source,
                    Webb 2021 form Δ = c * ln(SED * VDE) + b, melanin and SC-attenuated,
                    BSA-scaled, feedback-inhibited):
    S(UV, M, SC, D, F, bsa) = A * ln(1 + UV * E_source) *
                              bsa_factor(bsa) *
                              exp(-K_const * M_baseline(F)) *
                              exp(-K_ATTEN * max(0, M - M_baseline(F))) *
                              exp(-K_SC * SC) * max(0, 1 - D / D_max)

    bsa_factor(bsa) = (bsa / BSA_REF) ** BSA_EXP
        where BSA_REF = 0.35 (Farrar 2013 (35% BSA); note Bogh 2011 used 6-24% BSA),
        BSA_EXP = 0.50 (sub-linear scaling; P5 ln-BSA coefficient supports
        diminishing returns above ~35% BSA; Matsuoka 1990 P8 shows plateau above 33%).
        Default bsa = BSA_REF so existing scenarios are unchanged (factor = 1.0).

Parameter & Source Sourcing
---------------------------
- k_decay_D(D) = ln(2) / (2953 * D^(-0.72)):
    State-dependent D decay rate from Datta 2017 power-law fit (DOI: 10.1039/c6pp00258g).
    # Power-law fitted to Datta 2017 three-group data (89d, 148d, 199d)
    Three-group validation: t_half(132) ≈ 89d, t_half(65) ≈ 148d, t_half(43) ≈ 199d,
    all within 1.4% of Datta 2017 reported values. Replaces prior fixed 80-day constant.
- A = 2.353:
    Synthesis scaling factor calibrated to preserve a 1.27 nmol/L rise for a 1 SED
    dose in untanned FST III at reference BSA (35%), baseline D=25.0, SC=0.0.
    NOTE: Young 2021 PNAS reports fair-skin slopes of 2.0-5.0 nmol/L per SED
    at 85% BSA. Model at 85% BSA predicts ~1.85 — low end but order-of-magnitude
    consistent. If A were recalibrated to midpoint (~3.5), absolute D predictions
    would shift ~50% upward; scheduling rankings would be unchanged.
- K_const = 0.33:
    Baseline (constitutive) inhibition coefficient. Recalculated to preserve Young 2020
    FST II:VI synthesis ratio of ~1.36: K_const = ln(1.36) / (M_VI - M_II) = 0.33.
    See K_ATTEN note below.
- D_max = 150.0 nmol/L:
    Homeostatic feedback regulation saturation ceiling (WARNING: ORIGIN UNCONFIRMED — not found in Bogh 2010; PMID 19812604 max was 116 nmol/L). Corrected PMID 19798069 -> 19812604.
- UV_halfmax_by_FST = {I-III: 2.0, IV: 5.0, V: 8.0, VI: 10.0}:
    Hill saturation constant for melanin induction kinetics (Ravnbak 2008 Table IV post-adaptation multi-exposure MMD, PMID: 18480919).
- Stratum Corneum (SC) Thickening:
    [C8] SC(t) IS modeled as an active state variable. Documents or comments
    saying 'SC not modeled' are incorrect and have been corrected.
    Decay k_decay_SC (28-day half-life), rise alpha_SC = 0.005.
    [C7-SC] SC_max = 0.30 is a PRIOR, not a calibrated constant. De Boer 2025
    (P2) supports an SC thickening range of +23% (FP I-III) to +36% (FP IV-VI);
    evidence-based prior is 0.20-0.40. Do NOT tighten alpha_SC, k_decay_SC, or
    K_SC without individual-level time-course data from de Boer 2025 supplementary.
- E_source (SOURCE CALIBRATION TABLE):
    [C3] E_source must be a per-study calibration input, NOT a fixed multiplier.
    The SOURCE_CALIBRATION_TABLE below provides evidence-derived starting priors
    from Webb 2021 (P21) and Grigalavicius 2015 (P35). These are priors only;
    Young 2021 PNAS (P20) shows the CIE previtamin-D3 action spectrum needs a ~5 nm
    blue shift, meaning exact values require recomputation from measured spectra.
    Osmancevic 2009 (P18) shows broadband UVB can outperform NB-UVB per SED in
    psoriasis, confirming these cannot be universal constants.
- K_ATTEN = 0.37:
    [C5] Melanin attenuation coefficient from Young 2020 (P9) evidence.
    Young 2020 shows FST II:VI vitamin D slope ratio is only ~1.3-1.4
    (Arimed B: 1.29; TL01: 1.43), implying a small melanin penalty.
    K_ATTEN=2.0 was a coarse prior; 0.37 is the paper-backed value giving
    an FST II:VI ratio consistent with Young 2020 when combined with new
    M_baseline_by_FST and K_const values.
    July 2026 Consensus search confirmed: Matsuoka 1990 remains the ONLY direct
    within-subject tanning-attenuation study. Note local PDF is JAAD 1990 (sunscreen/BSA);
    the 51% tanning attenuation figure is from J Lab Clin Med 1990 (PMID 2165512),
    whose ABSTRACT is in Papers_and_Datasets/ and contains the full group-mean data.
    That 51% is confounded by a baseline 25(OH)D difference; see
    vitd_calibration/matsuoka1990_check.py for the deconvolution.
    Neville 2020 explicitly states the field still lacks this data. No independent replication exists.
- M_baseline_by_FST:
    [C7] Anchored to ITA_Unexposed means from trial CSVs (ArimedB trial data,
    N=44). Mapping: M = 1 - (ITA_U + 30) / 85, clipped to [0, 1].
    FST I is extrapolated (ITA_U ≈ 58°). Values:
    {I: 0.0, II: 0.07, III: 0.21, IV: 0.44, V: 0.78, VI: 1.00}.
    [C10-PRIOR] GROUP-LEVEL PRIOR — Datta 2019 (P13, PMID: 30656228) shows
    measured pigmentation explains only 1.9-2.6% of individual response variance
    (note Datta 2019 studied pigment SNPs: ASIP, SLC24A4, SLC45A2, not VDR/CYP2R1/CYP24A1).
    Do NOT use as a precise individual predictor or to justify large FST penalties.
    Genetics (VDR, CYP2R1, CYP24A1), supplementation, baseline D, and BSA dominate.
    WARNING: FST V (0.78) and VI (1.00) are group-level priors with NO direct
    melanin-index validation from published data. Do not use as individual predictors.
- exposed_bsa (Bogh 2011 / P5, Bogh 2010 / P36, Kimlin 2014 / P32):
    [C1] Fraction of body surface area exposed to UV (0.0-1.0). Kimlin 2014 (P32)
    shows clothing cover is the LARGEST single contributor (~27%) to explained
    25(OH)D variance in naturalistic settings. Bogh 2011 (P5) derived:
    delta = 7.1 - 0.223*baseline + 11.0*ln(UVB_dose) + 5.5*ln(%BSA), R2=0.47.
    Matsuoka 1990 (P8) shows a plateau above ~33% BSA. All validation scenarios
    must specify exposed_bsa explicitly; use BSA_REF=0.35 as the reference.
    Clothing cover = 1 - exposed_bsa.

    # -------------------------------------------------------------------------
    # PARAMETER EVIDENCE STATUS (updated from Consensus literature search, July 2026)
    # -------------------------------------------------------------------------
    # Gap 1 — M_baseline FST V/VI:
    #   Current values: {V: 0.78, VI: 1.00} (ITA_Unexposed means, ArimedB N=44)
    #   Literature support: WILKES 2015 gives MI>750->FST VI on Mexameter (partial).
    #   ALALUF 2001 gives quantitative melanin in FST V/VI but not as a baseline MI table.
    #   No paper provides separate untanned FST V and FST VI baseline MI means.
    #   Status: COARSE PRIOR — awaiting de Boer 2025 individual-level data.
    #
    # Gap 2 — K_ATTEN:
    #   Current value: 0.37
    #   Literature support: MATSUOKA 1990 (J Lab Clin Med, PMID 2165512; abstract IS
    #   in the repo -- local PDF is a DIFFERENT paper, JAAD 1990 sunscreen/BSA).
    #   Within-subject n=7. Naive 51% attenuation is confounded by a 64% higher
    #   tanned baseline 25(OH)D; deconvolved melanin-only effect implies
    #   K_ATTEN 0.34-1.00 by feedback form. Only source; no replication.
    #   NEVILLE 2020 confirms field still lacks tanning-effect data.
    #   MARKIEWICZ 2020: tanning may be redistribution, not pure synthesis.
    #   Status: SINGLE-SOURCE PRIOR — no independent replication.
    #
    # Gap 3 — A (synthesis rate):
    #   Current value: 2.353 (calibrated to 1.17 nmol/L per 1 SED at FST III, D=25)
    #   Literature support: YOUNG 2021 gives fair-skin slopes ~2.0-5.0 nmol/L/SED.
    #   WEBB 2021 confirms logarithmic form; BORECKA 2024 and CHALCRAFT 2020 provide
    #   partial low-dose response data in light skin.
    #   Status: ORDER-OF-MAGNITUDE CONSISTENT — exact value is BSA/feedback-normalized.
    #
    # Gap 4 — K_const:
    #   Current value: 0.33 (from Young 2020 FST II:VI ratio ~1.36)
    #   Literature support: YOUNG 2020 (direct, inhibition factor 1.3-1.4).
    #   LIBON 2013 (directional: FST VI no significant D rise vs FST II-III significant).
    #   ARMAS 2007 (80% variation from dose + skin tone).
    #   Status: ONE DIRECT + TWO PARTIAL CHECKS — directionally confirmed.
    # -------------------------------------------------------------------------

Future covariates (not yet in core model; add when matching dataset available):
    [C9] Total cholesterol / 7-DHC precursor status: Bogh 2010 (P36) found total
    cholesterol explained 26.5% of delta variance in a controlled UVB subgroup,
    second only to baseline 25(OH)D. Do NOT add to core model without a local
    dataset linking serum cholesterol, UV exposure, and 25(OH)D response.

Dose framing modes (label all scenario outputs):
    [C4] Fixed absolute SED: each session delivers the same SED regardless of MED.
    [C4] Fixed cumulative SED: total UV budget is split across sessions.
    [C4] Fixed per-session SED: 1 SED/session; cumulative varies with frequency.
    [C4] Fixed fractional MED: each session is N% of the individual MED.
    P24 (Shih/Farrar/Rhodes 2018) shows fractional-MED dosing removes the apparent
    skin-type effect for acute 25(OH)D gain. P22/P3 show fixed absolute SED can
    leave South Asian/FST V participants below sufficiency. Do NOT treat these
    framings as interchangeable in output reports.

Audit change tags (July 2026 Scientific Audit):
  [C1]  exposed_bsa / clothing_cover support added to model and validation
  [C2]  SOURCE_CALIBRATION_TABLE replaces ad hoc E_source scalar
  [C3]  E_source must be per-study calibration input
  [C4]  Dose framing labels added to simulate() output
  [C5]  K_ATTEN small-penalty evidence caveat
  [C7]  SC_max reparameterized as 0.20-0.40 prior (default 0.30)
  [C7-SC] SC defaults documented as priors, not calibrated constants
  [C8]  SC state active, calibration pending
  [C9]  Cholesterol/7-DHC added to future covariates
  [C10-PRIOR] K_ATTEN and M_baseline_by_FST explicitly tagged as coarse priors

Author: Ahaan
Last updated: 2026-07 (Scientific Audit Rev 2)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vitd_physiology import (body_weight_dilution_factor,
                             photodegradation_factor,
                             hydroxylation_efficiency,
                             cancer_risk)


# ---------------------------------------------------------------------------
# [C2] Source/Spectrum Calibration Table
# ---------------------------------------------------------------------------
# Evidence-derived D3-effective dose weighting factors per UV source.
# Sources: Webb 2021 (P21), Grigalavicius 2015 (P35).
# IMPORTANT: Treat these as priors until recalculated from measured spectra.
# Young 2021 PNAS (P20) shows the CIE previtamin-D3 action spectrum needs a ~5 nm
# blue shift, which will change all of these values. Osmancevic 2009 (P18)
# shows broadband UVB can outperform NB-UVB per SED in psoriasis patients,
# confirming source-specific variation cannot be collapsed to a single constant.
#
# Keys: source name (str). Values: (E_source_mean, E_source_range_low, E_source_range_high).
# Use E_source_mean for point estimates; report range in sensitivity analyses.
SOURCE_CALIBRATION_TABLE: Dict[str, Tuple[float, float, float]] = {
    # (mean, range_low, range_high)
    "solar_summer":   (1.89, 1.79, 2.00),  # P21/P35: summer sun
    "solar_broadband": (1.00, 0.90, 1.10),  # baseline reference
    "tl01_nb_uvb":    (2.44, 2.24, 2.63),  # P21: 2.24; P35: 2.63; use midpoint
    "tl12_bb_uvb":    (1.24, 1.23, 1.25),  # P21/P35 agree
    "uv6_broadband":  (1.54, 1.54, 1.55),  # P21/P35 agree
    "arimed_b":       (1.83, 1.83, 1.83),  # P21 only
    "sunbed_uvb":     (1.18, 1.18, 1.18),  # P35 (Grigalavicius 2015 Table 2); P21 Wolff Life Sun = 1.85
}

# Dietary intake conversion: nmol/L of 25(OH)D added per ug/day of vitamin D3,
# per day of intake. This is an INPUT-RATE coefficient, not a steady-state
# response -- the steady state emerges from this source balanced against
# k_decay_D(D). Cashman 2008/2011 dose-response trials give roughly
# +1.5-2.0 nmol/L steady-state per ug/day of D3; with the Datta 2017
# state-dependent clearance in this model, 0.07 reproduces that band
# (25 ug/day -> ~140 nmol/L over a year with no UV, no baseline intake).
# [PRIOR] Not independently fitted; revisit if a dietary-only cohort is added.
DIETARY_UG_TO_NMOL_PER_L: float = 0.07

# Reference exposed BSA fraction for synthesis normalization.
# Farrar 2013 (35% BSA); note Bogh 2011 used 6-24% BSA.
BSA_REF: float = 0.35
# Sub-linear BSA scaling exponent (P5 ln-BSA term; plateau evidence from P8)
BSA_EXP: float = 0.50


def get_E_source(source_name: str) -> float:
    """Return E_source mean for a named source from SOURCE_CALIBRATION_TABLE.

    [C2] Use this instead of hardcoding an E_source float wherever possible.
    Falls back to 1.0 (broadband reference) with a warning if source unknown.
    """
    if source_name in SOURCE_CALIBRATION_TABLE:
        return SOURCE_CALIBRATION_TABLE[source_name][0]
    import warnings
    warnings.warn(
        f"Unknown source '{source_name}' — falling back to E_source=1.0 (broadband reference). "
        f"Add to SOURCE_CALIBRATION_TABLE or pass E_source_override directly.",
        UserWarning, stacklevel=2,
    )
    return 1.0


@dataclass
class ModelParams:
    A: float = 2.353
    alpha: float = 0.0025
    alpha_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0025, 2: 0.0025, 3: 0.0025,
        4: 0.002,  5: 0.002,  6: 0.002,
    })
    alpha_max: float = 0.02
    # Hill saturation constant for melanin induction kinetics (Ravnbak 2008 Table IV post-adaptation multi-exposure MMD)
    UV_halfmax: float = 2.0
    UV_halfmax_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 2.0, 2: 2.0, 3: 2.0,
        4: 5.0, 5: 8.0, 6: 10.0,
    })
    beta: float = np.log(2) / 56.0
    K_const: float = 0.33
    # Facultative (tanning) headroom above constitutive pigmentation, on the
    # normalized 0-1 melanin scale. The absolute ceiling for a given FST is
    # M_baseline(FST) + M_TAN_HEADROOM, capped at 1.0 -- see M_ceiling().
    #
    # A single global M_max was wrong: M_max=0.95 sat BELOW
    # M_baseline_by_FST[6]=1.00, so the saturating pulse update
    #     M <- M_max - (M_max - M) * exp(-alpha * sed)
    # DECREASED melanin for FST VI (UV made the skin lighter). FST V/VI is the
    # target population for this project, so that inversion mattered.
    # Headroom 0.20 keeps FST I-III behaviour close to the old global 0.95
    # ceiling while making tanning monotone for every FST.
    # [PRIOR] Magnitude is not separately identified; Ravnbak 2008 reports
    # facultative MI gains but on instrument-specific units.
    M_TAN_HEADROOM: float = 0.20
    M_max: float = 0.95
    # [C5][C10-PRIOR] K_ATTEN is a COARSE PRIOR. Evidence (Young 2020 P9, Datta 2019 P13,
    # Bogh 2010 P36) supports only a ~1.3-1.4x FST II-VI synthesis ratio, not a large
    # multi-fold melanin penalty. Calibrate baseline 25(OH)D + BSA + source before
    # attributing residuals to pigmentation.
    K_ATTEN: float = 0.37
    D0_default: float = 25.0
    # WARNING: ORIGIN UNCONFIRMED — not found in Bogh 2010 (PMID 19812604 max was 116 nmol/L)
    D_max: float = 150.0
    # Daily dietary Vitamin D intake in micrograms (ug/day)
    dietary_D_daily_ug: float = 0.0
    # [C2] E_source default corresponds to broadband solar reference (1.0).
    # Use SOURCE_CALIBRATION_TABLE and get_E_source() instead of hardcoding.
    E_source: float = 1.0
    # [C1] Reference and default exposed BSA fraction (Bogh 2011 P5 / Farrar 2013).
    # Kimlin 2014 (P32) shows clothing cover is the largest naturalistic predictor.
    # Default is BSA_REF so all existing scenarios are unchanged (BSA factor = 1.0).
    exposed_bsa: float = BSA_REF
    # [C3-feedback] Baseline feedback parameters
    # feedback_mode can be 'linear', 'power', 'saturating', or 'biphasic'.
    # 'biphasic' is the mechanistically preferred option (see vitd_physiology):
    # 25-hydroxylation efficiency falls above a ~56 nmol/L inflection rather
    # than synthesis stopping at a hard D_max ceiling. Default stays 'linear'
    # so historical results remain reproducible; flip it deliberately.
    feedback_mode: str = "linear"
    feedback_power: float = 1.0
    D_half_feedback: float = 50.0

    # ---- Mechanistic extensions (July 2026 literature pass) -----------------
    # All default OFF/neutral so every one is separately ablatable and existing
    # scenarios are bit-identical unless explicitly enabled.
    #
    # Body weight for volumetric dilution (Drincic 2012: hyperbolic in total
    # body WEIGHT, not BMI, and not sequestration). None disables the term.
    body_weight_kg: Optional[float] = None
    # Previtamin-D3 photodegradation plateau (Holick 1981, van Dijk 2016).
    # This is the mechanism that makes dose-splitting genuinely better, as
    # opposed to the feedback artifact in OPEN_QUESTIONS gap 19.
    use_photodegradation: bool = False
    # Cutaneous D3 -> serum 25(OH)D transit delay. Without this, UV synthesis
    # appears in serum 25(OH)D instantaneously, which is wrong on exactly the
    # timescale this project optimises over (1-3 day intervals).
    # Two serial steps, lumped into one first-order transfer:
    #   previtamin-D3 -> D3   thermal isomerisation, ~80% complete in 8 h
    #   D3 -> 25(OH)D         hepatic CYP2R1; serum D3 half-life ~4 h, but
    #                         appearance of 25(OH)D peaks over ~1-2 days
    # t_half = 1.0 day is the lumped transfer half-life. [PRIOR] No study
    # reports this as a fitted rate constant; it is inferred from the 8 h
    # isomerisation plus the ~4 h D3 half-life and the observation that serum
    # 25(OH)D is typically sampled 24 h post-exposure.
    use_transit_delay: bool = False
    transit_half_life_days: float = 1.0
    # Stratum corneum thickening state parameters
    # [C7-SC] Recalibrated from de Boer 2025 Supplementary Table S1:
    #   FP I-III: SC 5.6->6.9 um (+23% -> SC_max=0.23)
    #   FP IV-VI: SC 6.6->9.0 um (+36% -> SC_max=0.36)
    # alpha_SC=0.12 calibrated so SC reaches ~90% of SC_max within
    # 27 sessions (0.8 SED 3x/week, de Boer protocol).
    # k_decay_SC remains at 28d half-life (epidermal turnover literature).
    # k_damage (UV-driven desquamation) fits the week-5 SC dip only
    # weakly from 3 group-mean timepoints; kept at 0.0 until individual-
    # level data is available for joint fitting.
    k_decay_SC: float = np.log(2) / 28.0
    alpha_SC: float = 0.12
    SC_max: float = 0.30
    SC_max_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.23, 2: 0.23, 3: 0.23,
        4: 0.36, 5: 0.36, 6: 0.36,
    })
    K_SC: float = 1.0
    k_damage: float = 0.0
    # [C7-SC-FITTED] Evidence-based ranges for SC parameters.
    # SC_max: de Boer 2025 (P2) reports +23% (FP I-III) and +36% (FP IV-VI).
    #   Default 0.30 is the midpoint; use SC_max_by_FST for per-FST values.
    # alpha_SC: Fitted to de Boer 2025. SC reaches ~90% of SC_max within
    #   9 weeks (0.8 SED 3x/week). Prior range 0.003-0.008 was too low.
    #   Experimental range: 0.08-0.20 (fitting uncertainty from 3 timepoints).
    #   Default 0.12 is the mid-range fitted value.
    # k_decay_SC: 21-28 day half-life. Pearse 1987 (P23) measured SC after
    #   6 weeks of 3x/week UVB. Epidermal turnover literature: 26-28 days.
    #   de Boer 2025 shows SC still rising at 9 weeks -> decay slower than rise.
    # K_SC: Attenuation strength. Pearse 1987 did not measure vitamin D
    #   attenuation directly. K_SC=1.0 gives exp(-0.30)=74% at SC_max.
    #   Range 0.5-2.0 reflects uncertainty.
    # k_damage: de Boer 2025 shows SC dip at week 5 (~4.3 um, -35% from
    #   baseline 6.6 um in FP IV-VI). Single k_damage cannot fit both dip
    #   and recovery — likely requires time-varying or UV-penetration-dependent
    #   formulation. Default 0.0 preserves monotonic SC; dip is a known
    #   model limitation.
    SC_PRIOR_RANGES = {
        "SC_max": (0.20, 0.40),
        "alpha_SC": (0.08, 0.20),
        "k_decay_SC": (np.log(2) / 28, np.log(2) / 21),
        "K_SC": (0.5, 2.0),
        "k_damage": (0.0, 0.15),
    }
    # [C7][C10-PRIOR] M_baseline_by_FST anchored to ITA_Unexposed means from
    # trial CSVs (ArimedB trial data, N=44). Mapping: M = 1 - (ITA_U + 30) / 85.
    # K_const recalculated to preserve Young 2020 FST II:VI ratio of ~1.36.
    # Documented as rounded group priors.
    M_baseline_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0, 2: 0.07, 3: 0.21, 4: 0.44, 5: 0.78, 6: 1.00,
    })

    def M_baseline(self, fst: int) -> float:
        return self.M_baseline_by_FST[fst]

    def M_ceiling(self, fst: int) -> float:
        """Maximum attainable melanin index for this FST (fully tanned).

        Always M_baseline(fst) + M_TAN_HEADROOM, so UV exposure increases
        melanin for EVERY FST. Deliberately NOT capped at 1.0: 1.0 is the
        *constitutive* FST VI reference on the Datta 2019 reflectance scale,
        not a physical ceiling. FST VI does undergo facultative melanogenesis,
        so a fully tanned FST VI sits above untanned FST VI (M = 1.20 here).
        Capping at 1.0 would silently make tanning a no-op for FST VI --
        different from the old inversion bug, but equally wrong.

        Synthesis attenuation stays well-behaved above 1.0: only the
        difference (M - M_baseline) enters exp(-K_ATTEN * excess_M).
        """
        return self.M_baseline(fst) + self.M_TAN_HEADROOM

    def bsa_factor(self, bsa: Optional[float] = None) -> float:
        """[C1] Sub-linear BSA scaling factor relative to BSA_REF.

        S ∝ (bsa / BSA_REF) ** BSA_EXP.  Default bsa = self.exposed_bsa.
        Returns 1.0 when bsa == BSA_REF (backward-compatible default).
        P5 (Bogh 2011) ln-BSA coefficient; P8 (Matsuoka 1990) BSA plateau above 33%.
        """
        b = self.exposed_bsa if bsa is None else bsa
        b = max(0.0, min(1.0, b))
        if b <= 0.0:
            return 0.0
        return (b / BSA_REF) ** BSA_EXP

    @staticmethod
    def k_decay_D(D: float) -> float:
        """State-dependent D decay rate from Datta 2017 power-law fit.

        # Power-law fitted to Datta 2017 three-group data (89d, 148d, 199d)
        t_half(D) = 2953 * D^(-0.72)
        Returns k = ln(2) / t_half(D) in units of 1/days.
        """
        t_half = 2953.0 * D ** (-0.72)
        return np.log(2.0) / t_half


def synthesis_per_session(sed: float, M: float, M_baseline: float,
                          D: float,
                          SC: float,
                          params: ModelParams,
                          A_override: float = None,
                          K_const_override: float = None,
                          K_ATTEN_override: float = None,
                          E_source_override: float = None,
                          D_max_override: float = None,
                          K_SC_override: float = None,
                          exposed_bsa_override: float = None) -> float:
    """Compute per-session vitamin D synthesis (nmol/L).

    Aligns logarithmic dose-response with Webb 2021 formulation:
    delta = c * ln(SED * VDE) + b, using effective D3-weighted dose (sed * E_source).

    [C1] exposed_bsa_override: override exposed BSA fraction for this session.
    [C2] E_source_override: per-study spectral weighting (use SOURCE_CALIBRATION_TABLE).
    """
    if sed <= 0:
        return 0.0
    A_use = params.A if A_override is None else A_override
    Kc_use = params.K_const if K_const_override is None else K_const_override
    K_use = params.K_ATTEN if K_ATTEN_override is None else K_ATTEN_override
    E_use = params.E_source if E_source_override is None else E_source_override
    D_max_use = params.D_max if D_max_override is None else D_max_override
    K_SC_use = params.K_SC if K_SC_override is None else K_SC_override
    bsa_f = params.bsa_factor(exposed_bsa_override)  # [C1]

    excess_M = max(0.0, M - M_baseline)
    constitutive_factor = np.exp(-Kc_use * M_baseline)
    facultative_factor = np.exp(-K_use * excess_M)
    sc_factor = np.exp(-K_SC_use * SC)
    # [C3-feedback] Upgrade baseline response to a calibration-supported feedback factor.
    # Linear feedback (default): max(0.0, 1.0 - D / D_max_use)
    # Power feedback: max(0.0, 1.0 - (D / D_max_use) ** feedback_power)
    # Saturating/exponential feedback (from P5/P33/P35): exp(-decay * D)
    f_mode = getattr(params, 'feedback_mode', 'linear')
    if f_mode == 'linear':
        feedback_factor = max(0.0, 1.0 - D / D_max_use)
    elif f_mode == 'power':
        feedback_factor = max(0.0, 1.0 - (D / D_max_use) ** getattr(params, 'feedback_power', 1.0))
    elif f_mode == 'saturating':
        # saturating feedback: exp(-decay * D) where decay = ln(2) / D_half_feedback
        decay = np.log(2.0) / getattr(params, 'D_half_feedback', 50.0)
        feedback_factor = np.exp(-decay * D)
    elif f_mode == 'biphasic':
        # Mechanistic: 25-hydroxylation efficiency, inflection ~56 nmol/L
        # (Comms Med 2025). Never exactly zero -- cutaneous synthesis does not
        # stop outright at any serum level.
        feedback_factor = hydroxylation_efficiency(D)
    else:
        feedback_factor = max(0.0, 1.0 - D / D_max_use)

    # Previtamin-D3 photostationary plateau: per-session yield saturates in
    # absolute terms, so one large session cannot substitute for several small
    # ones (Holick 1981 plateau at 10-15% of 7-DHC; van Dijk 2016).
    photo_f = (photodegradation_factor(sed * E_use)
               if getattr(params, 'use_photodegradation', False) else 1.0)

    # Volumetric dilution: the same cutaneous input raises serum 25(OH)D less
    # in a larger body (Drincic 2012, hyperbolic in total body weight).
    weight_f = body_weight_dilution_factor(getattr(params, 'body_weight_kg', None))

    return (A_use * np.log(1.0 + sed * E_use) * bsa_f *
            constitutive_factor * facultative_factor * sc_factor *
            feedback_factor * photo_f * weight_f)


def compute_safety_sidecar(sed: float, fst: int, accumulated_sed_7d: float = 0.0) -> Dict:
    """[C6-safety] Safety sidecar to assess DNA damage / CPD risk.

    P22 (Rhodes 2017) and P24 (Shih/Farrar/Rhodes 2018) show that cyclobutane pyrimidine
    dimers (CPD) and DNA damage occur concurrently with vitamin D synthesis even
    at suberythemal doses, and that basal-layer CPD clearance and risk differ
    strongly by FST.

    This function returns qualitative risk flags and recommendations based on
    session SED, FST, and recent 7-day cumulative UV dose.
    """
    # 1. Estimate individual MED in SED
    # FST MED values from Shih/Farrar/Rhodes 2018 (P24): FST I-II ~2-3 SED, FST V ~10 SED, FST VI ~15 SED
    med_by_fst = {1: 2.1, 2: 2.6, 3: 3.2, 4: 5.6, 5: 7.5, 6: 15.2}
    med = med_by_fst.get(fst, 3.5)

    fraction_med = sed / med

    # CPD Risk Level
    # P24: even at 20-40% MED, basal CPD is detectable in FST I-II, but protected in FST V-VI
    if fst <= 2:
        if fraction_med >= 0.8:
            cpd_risk = "CRITICAL (High basal CPD, high erythema risk)"
        elif fraction_med >= 0.4:
            cpd_risk = "HIGH (Significant basal CPD, suberythemal damage)"
        elif fraction_med >= 0.2:
            cpd_risk = "MODERATE (basal CPD detectable, low erythema)"
        else:
            cpd_risk = "LOW"
    elif fst <= 4:
        if fraction_med >= 0.8:
            cpd_risk = "HIGH (Erythema risk, basal CPD likely)"
        elif fraction_med >= 0.4:
            cpd_risk = "MODERATE (Suberythemal DNA damage)"
        else:
            cpd_risk = "LOW"
    else:  # FST V-VI
        if fraction_med >= 0.8:
            cpd_risk = "MODERATE (Erythema risk, basal layers relatively protected by melanin)"
        else:
            cpd_risk = "LOW (Basal layer highly protected, minimal CPD risk)"

    # Chronic exposure risk (7-day accumulation)
    # Repeated low dose (e.g. de Boer 2025 0.8 SED 3x/wk = 2.4 SED/wk)
    # If 7d dose exceeds 2x MED, high risk of photoaging/damage accumulation
    if accumulated_sed_7d > 2.0 * med:
        chronic_risk = "HIGH (Risk of chronic photoaging, stratum corneum thickening barrier upregulation)"
    elif accumulated_sed_7d > 1.0 * med:
        chronic_risk = "MODERATE (Accumulating sub-erythemal damage)"
    else:
        chronic_risk = "LOW"

    return {
        "individual_med_sed": med,
        "fraction_med": fraction_med,
        "cpd_risk_level": cpd_risk,
        "chronic_risk_level": chronic_risk,
        "action_spectrum_warning": (
            "Warning: Action spectra for DNA damage (erythema) and vitamin D synthesis "
            "differ. Suberythemal exposures still cause cumulative DNA damage."
        )
    }


def build_pulse_schedule(duration_days: int, sed_per_session: float,
                         interval_days: int, start_day: int = 0
                         ) -> List[Tuple[float, float]]:
    """Periodic UV pulses. Returns list of (day, SED) tuples."""
    return [(d, sed_per_session)
            for d in range(start_day, duration_days, interval_days)]


def build_weekly_schedule(duration_days: int, sed_per_session: float,
                          days_of_week: List[int]) -> List[Tuple[float, float]]:
    """Schedule on specific weekdays (0=Mon ... 6=Sun)."""
    sched = []
    for w in range(int(np.ceil(duration_days / 7)) + 1):
        for d in days_of_week:
            day = w * 7 + d
            if day < duration_days:
                sched.append((day, sed_per_session))
    return sched


def simulate(fst: int,
             schedule: List[Tuple[float, float]],
             duration_days: int,
             params: ModelParams = None,
             D0: float = None,
             M0: float = None,
             SC0: float = 0.0,
             k_decay_M_override: float = None,
             alpha_override: float = None,
             K_const_override: float = None,
             K_ATTEN_override: float = None,
             E_source_override: float = None,
             D_max_override: float = None,
             k_decay_SC_override: float = None,
             alpha_SC_override: float = None,
              K_SC_override: float = None,
              k_damage_override: float = None,
              SC_max_override: float = None,
              M_max_override: float = None,
              exposed_bsa_override: float = None,
               dose_framing: str = "unspecified") -> Dict:
    """Simulate vitamin D dynamics under a UV pulse schedule.

    Parameters
    ----------
    exposed_bsa_override : float, optional
        [C1] Override exposed BSA fraction (0-1) for all sessions in this run.
        If None, uses params.exposed_bsa (default BSA_REF = 0.35).
    dose_framing : str
        [C4] Descriptor for how UV doses are specified in this scenario.
        Must be one of:
          'fixed_absolute_SED'   - each session delivers a fixed SED regardless of MED
          'fixed_cumulative_SED' - total UV budget is split across sessions
          'fixed_per_session_SED'- per-session SED is constant; cumulative varies (default)
          'fixed_fractional_MED' - each session is a fraction of the individual MED
        This label is stored in the result dict to document scenario provenance.
        P24 (Shih/Farrar/Rhodes 2018): fractional-MED dosing removes apparent skin-type
        effect; P22/P3: fixed absolute SED can leave FST V deficient. Never treat
        framings as interchangeable in output reports.
    """
    if params is None:
        params = ModelParams()
    M_base = params.M_baseline(fst)

    Kc_eff = params.K_const if K_const_override is None else K_const_override
    beta_eff = params.beta if k_decay_M_override is None else k_decay_M_override
    K_eff = params.K_ATTEN if K_ATTEN_override is None else K_ATTEN_override
    E_eff = params.E_source if E_source_override is None else E_source_override
    D_max_eff = params.D_max if D_max_override is None else D_max_override
    beta_SC_eff = params.k_decay_SC if k_decay_SC_override is None else k_decay_SC_override
    alpha_SC_eff = params.alpha_SC if alpha_SC_override is None else alpha_SC_override
    K_SC_eff = params.K_SC if K_SC_override is None else K_SC_override
    k_damage_eff = params.k_damage if k_damage_override is None else k_damage_override
    SC_max_eff = (params.SC_max_by_FST.get(fst, params.SC_max)
                  if SC_max_override is None else SC_max_override)
    # Per-FST melanin ceiling; always above M_base so tanning is monotone.
    M_ceil = params.M_ceiling(fst) if M_max_override is None else M_max_override

    D = params.D0_default if D0 is None else D0
    M = M_base if M0 is None else M0
    SC = SC0
    # Cutaneous/circulating D3 awaiting 25-hydroxylation, in nmol/L of
    # eventual 25(OH)D equivalent. Zero unless use_transit_delay is on.
    D3_pool = 0.0
    k_transit = (np.log(2.0) / params.transit_half_life_days
                 if getattr(params, 'use_transit_delay', False) else 0.0)

    schedule = sorted(schedule, key=lambda x: x[0])
    pulse_days = [p[0] for p in schedule]
    pulse_seds = [p[1] for p in schedule]

    t_record, D_record, M_record, SC_record = [0.0], [D], [M], [SC]
    synthesis_log = []
    safety_log = []  # [C6-safety] safety sidecar logs
    cumulative_SED = 0.0
    current_t = 0.0
    pulse_idx = 0

    while current_t < duration_days:
        next_t = (pulse_days[pulse_idx]
                  if pulse_idx < len(pulse_days)
                  else duration_days + 1)
        end_t = min(next_t, duration_days)

        if end_t > current_t:
            # March the state forward in dt-sized sub-steps, advancing current_t
            # each iteration. The recorded samples ARE the state trajectory --
            # there is no separate "big step" to the interval end, which would
            # double-count the dietary input and decay from a stale origin.
            dt = 0.5
            t_steps = np.arange(current_t + dt, end_t, dt)
            if len(t_steps) == 0 or t_steps[-1] < end_t:
                t_steps = np.append(t_steps, end_t)
            for ti in t_steps:
                delta_time = ti - current_t
                # Dietary input is a zero-order source over the sub-step, applied
                # before decay so it is subject to clearance like synthesised D.
                if params.dietary_D_daily_ug > 0:
                    # Volumetric dilution applies to oral D3 exactly as it does
                    # to cutaneous D3 -- it is a distribution-volume effect, not
                    # a skin effect (Drincic 2012; Ekwaru 2014 dose-response).
                    D += (params.dietary_D_daily_ug * DIETARY_UG_TO_NMOL_PER_L
                          * body_weight_dilution_factor(
                              getattr(params, 'body_weight_kg', None))
                          * delta_time)
                # Transit: D3 pool drains into the 25(OH)D compartment.
                # Applied before decay so newly arrived D3 is subject to
                # clearance over the remainder of the sub-step.
                if k_transit > 0.0 and D3_pool > 0.0:
                    transferred = D3_pool * (1.0 - np.exp(-k_transit * delta_time))
                    D += transferred
                    D3_pool -= transferred
                D = D * np.exp(-params.k_decay_D(D) * delta_time)
                M = M_base + (M - M_base) * np.exp(-beta_eff * delta_time)
                SC = SC * np.exp(-beta_SC_eff * delta_time)
                current_t = ti
                t_record.append(ti)
                D_record.append(D)
                M_record.append(M)
                SC_record.append(SC)
            current_t = end_t

        if (pulse_idx < len(pulse_days)
                and abs(current_t - pulse_days[pulse_idx]) < 1e-9):
            sed = pulse_seds[pulse_idx]
            delta_D = synthesis_per_session(sed, M, M_base, D, SC, params,
                                             K_const_override=Kc_eff,
                                             K_ATTEN_override=K_eff,
                                             E_source_override=E_eff,
                                             D_max_override=D_max_eff,
                                             K_SC_override=K_SC_eff,
                                             exposed_bsa_override=exposed_bsa_override)
            if k_transit > 0.0:
                # Route synthesis through the skin/serum D3 reservoir instead of
                # crediting serum 25(OH)D instantaneously.
                D3_pool += delta_D
            else:
                D = D + delta_D
            # Apply genotype A_offset (from vitd_genetics.py presets) — additive offset
            # to per-session synthesis, independent of D feedback.
            if hasattr(params, '_genotype_A_offset'):
                D = D + params._genotype_A_offset
            if alpha_override is not None:
                alpha_p = alpha_override
            else:
                alpha_p = params.alpha_by_FST.get(fst, params.alpha)
                UV_halfmax_p = params.UV_halfmax_by_FST.get(fst, params.UV_halfmax)
                hill_factor = sed**2 / (sed**2 + UV_halfmax_p**2)
                alpha_p = alpha_p + (params.alpha_max - alpha_p) * hill_factor
            M = M_ceil - (M_ceil - M) * np.exp(-alpha_p * sed)
            SC = SC * np.exp(-k_damage_eff * sed)
            SC = SC_max_eff - (SC_max_eff - SC) * np.exp(-alpha_SC_eff * sed)
            
            # [C6-safety] Calculate rolling 7-day cumulative SED (including current pulse)
            rolling_7d_sed = sed + sum(
                p_sed for p_day, p_sed in schedule[:pulse_idx]
                if current_t - 7.0 <= p_day < current_t
            )
            safety = compute_safety_sidecar(sed, fst, rolling_7d_sed)
            safety_log.append((current_t, safety))
            
            cumulative_SED += sed
            synthesis_log.append((current_t, sed, delta_D, M, SC))

            t_record.append(current_t)
            D_record.append(D)
            M_record.append(M)
            SC_record.append(SC)
            pulse_idx += 1

    # Quantitative skin-cancer risk for the whole schedule. Unlike the
    # per-session qualitative sidecar, this separates the two epidemiologically
    # distinct pathways: SCC tracks CUMULATIVE dose, while BCC and melanoma
    # track INTERMITTENT intensity (Elwood 1997: intermittent OR 1.71, heavy
    # occupational OR 0.86 i.e. protective; Leiter 2009; Calzavara-Pinton 2015).
    risk = cancer_risk([s for _, s in schedule], fst)

    _VALID_FRAMINGS = {
        "fixed_absolute_SED", "fixed_cumulative_SED",
        "fixed_per_session_SED", "fixed_fractional_MED",
        "unspecified",
    }
    if dose_framing not in _VALID_FRAMINGS:
        import warnings
        warnings.warn(
            f"dose_framing='{dose_framing}' is not a recognized value. "
            f"Use one of: {sorted(_VALID_FRAMINGS)}",
            UserWarning, stacklevel=2,
        )
    return {
        "t": np.array(t_record),
        "D": np.array(D_record),
        "M": np.array(M_record),
        "SC": np.array(SC_record),
        "synthesis_log": synthesis_log,
        "safety_log": safety_log,            # [C6-safety] per-session qualitative flags
        "cancer_risk": risk,                 # schedule-level quantitative indices
        "D3_pool_final": D3_pool,            # un-hydroxylated D3 still in transit
        "cumulative_SED": cumulative_SED,
        "fst": fst,
        "params": params,
        "dose_framing": dose_framing,        # [C4] scenario provenance label
        "exposed_bsa": (exposed_bsa_override
                        if exposed_bsa_override is not None
                        else params.exposed_bsa),  # [C1]
    }


def steady_state_D(result, window_days=28):
    t = result["t"]; D = result["D"]
    mask = t >= t[-1] - window_days
    return float(np.mean(D[mask]))


def efficiency_D_per_SED(result):
    log = result["synthesis_log"]
    if not log: return 0.0
    total_d = sum(item[2] for item in log)
    total_s = sum(item[1] for item in log)
    return total_d / total_s if total_s > 0 else 0.0


gross_synth_per_SED = efficiency_D_per_SED


def net_final_D_change_per_SED(result, D0=None):
    if D0 is None:
        D0 = result["D"][0]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0:
        return 0.0
    return (result["D"][-1] - D0) / total_sed


def _trapz(y, x):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def auc_D_per_SED(result):
    t = result["t"]
    D = result["D"]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0 or len(t) < 2:
        return 0.0
    return float(_trapz(D, t)) / total_sed


def auc_above_baseline_per_SED(result, D0=None):
    if D0 is None:
        D0 = result["D"][0]
    t = result["t"]
    D = result["D"]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0 or len(t) < 2:
        return 0.0
    y = np.maximum(0.0, D - D0)
    return float(_trapz(y, t)) / total_sed


def auc_above_sufficiency_per_SED(result, threshold_nmol_L=50.0):
    t = result["t"]
    D = result["D"]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0 or len(t) < 2:
        return 0.0
    if np.max(D) <= threshold_nmol_L:
        return 0.0
    y = np.maximum(0.0, D - threshold_nmol_L)
    return float(_trapz(y, t)) / total_sed


def auc_deficit(D, t, threshold):
    """Time-integrated deficiency below threshold.

    deficit = ∫ max(0, threshold - D(t)) dt  over the full time course.
    Units: nmol/L · days.

    Parameters
    ----------
    D : np.ndarray
        Serum 25(OH)D time series.
    t : np.ndarray
        Time points in days (same length as D).
    threshold : float
        Deficiency threshold in nmol/L (e.g., 30 or 50).

    Returns
    -------
    float
        Area under the curve where D < threshold (nmol/L·days).
    """
    y = np.maximum(0.0, threshold - D)
    return float(_trapz(y, t))


def mean_final_value(result, key, window_days=28):
    t = result["t"]; vals = result[key]
    mask = t >= t[-1] - window_days
    return float(np.mean(vals[mask]))

mean_final_M = lambda r, w=28: mean_final_value(r, "M", w)
mean_final_SC = lambda r, w=28: mean_final_value(r, "SC", w)


def final_D(result):
    return float(result["D"][-1])


if __name__ == "__main__":
    p = ModelParams()
    sched = build_pulse_schedule(84, 1.0, 2)
    r = simulate(fst=5, schedule=sched, duration_days=84, params=p)
    print(f"FST V, every-2-day, 1 SED, 12 weeks:")
    print(f"  cum SED: {r['cumulative_SED']:.1f}")
    print(f"  ss D:    {steady_state_D(r):.2f} nmol/L")
    print(f"  final M: {mean_final_M(r):.3f}")
    print(f"  final SC:{mean_final_SC(r):.3f}")
    print(f"  gross_synth_per_SED:           {gross_synth_per_SED(r):.3f}")
    print(f"  net_final_D_change_per_SED:    {net_final_D_change_per_SED(r):.3f}")
    print(f"  auc_D_per_SED:                 {auc_D_per_SED(r):.3f}")
    print(f"  auc_above_baseline_per_SED:    {auc_above_baseline_per_SED(r):.3f}")
    print(f"  auc_above_sufficiency_per_SED: {auc_above_sufficiency_per_SED(r):.3f}")
