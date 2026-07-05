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
        (Datta 2019 reflectance scale; 0.10 = FST I, 0.80 = FST VI)
SC(t) : additional stratum corneum thickness, dimensionless (above baseline 0)
        (representing epidermal thickening/hyperplasia photoadaptation channel;
        [C8] this state is ACTIVE in the model -- SC is NOT unmodeled;
        remaining task is FST/source-specific calibration, not implementation)

Governing equations
-------------------
dD/dt  = S(UV(t), M(t), SC(t), D(t), F, bsa)  -  k_decay_D * D
dM/dt  = alpha * UV(t) * (M_max - M)  -  beta * (M - M_baseline(F))
dSC/dt = alpha_SC * UV(t) * (SC_max - SC)  -  k_decay_SC * SC

Synthesis function (logarithmic dose-response, melanin and SC-attenuated,
                    BSA-scaled, feedback-inhibited):
    S(UV, M, SC, D, F, bsa) = A * ln(1 + UV) *
                              bsa_factor(bsa) *
                              exp(-K_const * M_baseline(F)) *
                              exp(-K_ATTEN * max(0, M - M_baseline(F))) *
                              exp(-K_SC * SC) * E_source * max(0, 1 - D / D_max)

    bsa_factor(bsa) = (bsa / BSA_REF) ** BSA_EXP
        where BSA_REF = 0.35 (reference exposed fraction, Bogh 2011 P5 baseline),
        BSA_EXP = 0.50 (sub-linear scaling; P5 ln-BSA coefficient supports
        diminishing returns above ~35% BSA; Matsuoka 1990 P8 shows plateau above 33%).
        Default bsa = BSA_REF so existing scenarios are unchanged (factor = 1.0).

Parameter & Source Sourcing
---------------------------
- k_decay_D = ln(2) / 80 per day:
    Effective systemic decay rate including adipose buffering. Supported by
    Prentice 2017 (PMID: 28738961) and post-UVB cycles in Datta 2017 (DOI: 10.1039/c6pp00258g).
    Datta 2017 (P4) directly measures an exponential half-life of 89 days in
    high-start subjects, with longer half-lives (148-199 days) at medium/low start
    levels. An 80-day fixed constant is approximately supported but should become
    state-dependent in more mechanistic implementations.
- A = 2.353:
    Synthesis scaling factor calibrated to preserve a 1.17 nmol/L rise for a 1 SED
    dose in untanned FST III at reference BSA (35%), baseline D=25.0, SC=0.0.
- K_const = 0.50:
    Baseline (constitutive) inhibition coefficient. Calibrated so that the FST II:VI
    synthesis ratio from the constitutive factor alone is consistent with the
    1.3-1.4x range reported in Young 2020 (PMID: 31883961). See K_ATTEN note below.
- D_max = 150.0 nmol/L:
    Homeostatic feedback regulation saturation ceiling, based on Bogh 2010 (PMID: 19798069).
- UV_threshold_by_FST = {I-III: 2.0, IV: 5.0, V: 8.0, VI: 10.0}:
    Fitzpatrick-stratified minimal melanogenesis thresholds, based on Ravnbak 2008 (PMID: 18480919).
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
    Young 2020 (P20) shows the CIE action spectrum needs a ~5 nm blue shift,
    meaning exact values require recomputation from measured spectra.
    Osmancevic 2009 (P18) shows broadband UVB can outperform NB-UVB per SED in
    psoriasis, confirming these cannot be universal constants.
- K_ATTEN = 2.0:
    [C5][C10-PRIOR] K_ATTEN is a COARSE PRIOR, not a measured parameter.
    Evidence from Young 2020 (P9) shows the FST II:VI vitamin D slope ratio is
    only ~1.3-1.4 (Arimed B: 1.29; TL01: 1.43), implying a SMALL melanin penalty.
    Datta 2019 (P13) further shows measured skin pigmentation explains only
    1.9-2.6% of individual response variation. Future calibration should fit
    baseline 25(OH)D, BSA, and source FIRST before attributing residuals to
    pigmentation. Do NOT present K_ATTEN=2.0 as a directly measured parameter.
- M_baseline_by_FST:
    [C7][C10-PRIOR] COARSE PRIOR / INDIVIDUAL-VARIABILITY WARNING:
    Datta 2019 (P13, PMID: 30656228) reports a 25(OH)D response range of
    2.9 to 139 nmol/L under IDENTICAL UVB dose and body area, with measured
    skin pigmentation (PPF) explaining only 1.9-2.6% of individual variation.
    M_baseline_by_FST provides a group-level lookup ONLY. Do NOT use as a precise
    individual predictor or to justify large FST-based synthesis penalties.
    Genetics (VDR, CYP2R1, CYP24A1), supplementation, baseline serum D, and
    BSA dominate individual outcomes. Bogh 2010 (P36) found no significant
    delta difference between dark-skinned FST V-VI and fair FST I-IV participants
    when matched on baseline 25(OH)D.
- exposed_bsa (Bogh 2011 / P5, Bogh 2010 / P36, Kimlin 2014 / P32):
    [C1] Fraction of body surface area exposed to UV (0.0-1.0). Kimlin 2014 (P32)
    shows clothing cover is the LARGEST single contributor (~27%) to explained
    25(OH)D variance in naturalistic settings. Bogh 2011 (P5) derived:
    delta = 7.1 - 0.223*baseline + 11.0*ln(UVB_dose) + 5.5*ln(%BSA), R2=0.47.
    Matsuoka 1990 (P8) shows a plateau above ~33% BSA. All validation scenarios
    must specify exposed_bsa explicitly; use BSA_REF=0.35 as the reference.
    Clothing cover = 1 - exposed_bsa.

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
    P24 (Felton/Rhodes 2018) shows fractional-MED dosing removes the apparent
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


# ---------------------------------------------------------------------------
# [C2] Source/Spectrum Calibration Table
# ---------------------------------------------------------------------------
# Evidence-derived D3-effective dose weighting factors per UV source.
# Sources: Webb 2021 (P21), Grigalavicius 2015 (P35).
# IMPORTANT: Treat these as priors until recalculated from measured spectra.
# Young 2020 (P20) shows the CIE previtamin-D3 action spectrum needs a ~5 nm
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
    "sunbed_uvb":     (1.18, 1.18, 1.18),  # P21 only (low-UVB tanning lamp)
}

# Reference exposed BSA fraction for synthesis normalization.
# Bogh 2011 (P5) range was 6-24%; Farrar 2013 used 35%; this is our reference.
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
    k_decay_D: float = np.log(2) / 80.0
    A: float = 2.353
    alpha: float = 0.0025
    alpha_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0025, 2: 0.0025, 3: 0.0025,
        4: 0.002,  5: 0.002,  6: 0.002,
    })
    alpha_max: float = 0.02
    UV_threshold: float = 2.0
    UV_threshold_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 2.0, 2: 2.0, 3: 2.0,
        4: 5.0, 5: 8.0, 6: 10.0,
    })
    beta: float = np.log(2) / 56.0
    K_const: float = 0.50
    M_max: float = 0.95
    # [C5][C10-PRIOR] K_ATTEN is a COARSE PRIOR. Evidence (Young 2020 P9, Datta 2019 P13,
    # Bogh 2010 P36) supports only a ~1.3-1.4x FST II-VI synthesis ratio, not a large
    # multi-fold melanin penalty. Calibrate baseline 25(OH)D + BSA + source before
    # attributing residuals to pigmentation.
    K_ATTEN: float = 2.0
    D0_default: float = 25.0
    D_max: float = 150.0
    # [C2] E_source default corresponds to broadband solar reference (1.0).
    # Use SOURCE_CALIBRATION_TABLE and get_E_source() instead of hardcoding.
    E_source: float = 1.0
    # [C1] Reference and default exposed BSA fraction (Bogh 2011 P5 / Farrar 2013).
    # Kimlin 2014 (P32) shows clothing cover is the largest naturalistic predictor.
    # Default is BSA_REF so all existing scenarios are unchanged (BSA factor = 1.0).
    exposed_bsa: float = BSA_REF
    # [C3-feedback] Baseline feedback parameters
    # feedback_mode can be 'linear', 'power', or 'saturating'
    feedback_mode: str = "linear"
    feedback_power: float = 1.0
    D_half_feedback: float = 50.0
    # Stratum corneum thickening state parameters
    # [C7-SC] SC_max is a PRIOR (0.20-0.40); de Boer 2025 (P2) reports +23% (FP I-III)
    # and +36% (FP IV-VI). Default set to midpoint 0.30. Do not tighten alpha_SC,
    # k_decay_SC, or K_SC without individual-level supplementary data.
    k_decay_SC: float = np.log(2) / 28.0
    alpha_SC: float = 0.005
    SC_max: float = 0.30   # revised from 0.50; prior range 0.20-0.40 (P2)
    K_SC: float = 1.0
    # [C7][C10-PRIOR] M_baseline_by_FST is a GROUP-LEVEL PRIOR only.
    # Datta 2019 (P13): pigmentation explains <3% of individual UV-D3 response.
    # Bogh 2010 (P36): matched-baseline dark/fair pairs show no significant delta diff.
    # Do NOT use as a precise individual predictor or justify large FST penalties.
    M_baseline_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.10, 2: 0.20, 3: 0.30, 4: 0.38, 5: 0.65, 6: 0.80,
    })

    def M_baseline(self, fst: int) -> float:
        return self.M_baseline_by_FST[fst]

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
    else:
        feedback_factor = max(0.0, 1.0 - D / D_max_use)

    return (A_use * np.log(1.0 + sed) * bsa_f *
            constitutive_factor * facultative_factor * sc_factor *
            E_use * feedback_factor)


def compute_safety_sidecar(sed: float, fst: int, accumulated_sed_7d: float = 0.0) -> Dict:
    """[C6-safety] Safety sidecar to assess DNA damage / CPD risk.

    P22 (Rhodes 2017) and P24 (Felton/Rhodes 2018) show that cyclobutane pyrimidine
    dimers (CPD) and DNA damage occur concurrently with vitamin D synthesis even
    at suberythemal doses, and that basal-layer CPD clearance and risk differ
    strongly by FST.

    This function returns qualitative risk flags and recommendations based on
    session SED, FST, and recent 7-day cumulative UV dose.
    """
    # 1. Estimate individual MED in SED
    # FST MED values from Felton 2018 (P24): FST I-II ~2-3 SED, FST V ~10 SED, FST VI ~15 SED
    med_by_fst = {1: 2.1, 2: 2.5, 3: 3.5, 4: 5.5, 5: 10.5, 6: 15.2}
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
             exposed_bsa_override: float = None,
             dose_framing: str = "fixed_per_session_SED") -> Dict:
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
        P24 (Felton/Rhodes 2018): fractional-MED dosing removes apparent skin-type
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

    D = params.D0_default if D0 is None else D0
    M = M_base if M0 is None else M0
    SC = SC0

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
            dt = 0.5
            t_steps = np.arange(current_t + dt, end_t, dt)
            if len(t_steps) == 0 or t_steps[-1] < end_t:
                t_steps = np.append(t_steps, end_t)
            for ti in t_steps:
                delta_time = ti - current_t
                Di = D * np.exp(-params.k_decay_D * delta_time)
                Mi = M_base + (M - M_base) * np.exp(-beta_eff * delta_time)
                SCi = SC * np.exp(-beta_SC_eff * delta_time)
                t_record.append(ti)
                D_record.append(Di)
                M_record.append(Mi)
                SC_record.append(SCi)
            D = D * np.exp(-params.k_decay_D * (end_t - current_t))
            M = M_base + (M - M_base) * np.exp(-beta_eff * (end_t - current_t))
            SC = SC * np.exp(-beta_SC_eff * (end_t - current_t))
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
            D = D + delta_D
            if alpha_override is not None:
                alpha_p = alpha_override
            else:
                alpha_p = params.alpha_by_FST.get(fst, params.alpha)
                UV_thresh_p = params.UV_threshold_by_FST.get(fst, params.UV_threshold)
                hill_factor = sed**2 / (sed**2 + UV_thresh_p**2)
                alpha_p = alpha_p + (params.alpha_max - alpha_p) * hill_factor
            M = params.M_max - (params.M_max - M) * np.exp(-alpha_p * sed)
            SC = params.SC_max - (params.SC_max - SC) * np.exp(-alpha_SC_eff * sed)
            
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

    _VALID_FRAMINGS = {
        "fixed_absolute_SED", "fixed_cumulative_SED",
        "fixed_per_session_SED", "fixed_fractional_MED",
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
        "safety_log": safety_log,            # [C6-safety]
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


def auc_D_per_SED(result):
    t = result["t"]
    D = result["D"]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0:
        return 0.0
    if len(t) < 2:
        return 0.0
    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(D, t)
    else:
        auc = np.trapz(D, t)
    return float(auc) / total_sed


def auc_above_baseline_per_SED(result, D0=None):
    if D0 is None:
        D0 = result["D"][0]
    t = result["t"]
    D = result["D"]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0:
        return 0.0
    if len(t) < 2:
        return 0.0
    y = np.maximum(0.0, D - D0)
    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(y, t)
    else:
        auc = np.trapz(y, t)
    return float(auc) / total_sed


def auc_above_sufficiency_per_SED(result, threshold_nmol_L=50.0):
    t = result["t"]
    D = result["D"]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0:
        return 0.0
    if len(t) < 2:
        return 0.0
    if np.max(D) <= threshold_nmol_L:
        return 0.0
    y = np.maximum(0.0, D - threshold_nmol_L)
    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(y, t)
    else:
        auc = np.trapz(y, t)
    return float(auc) / total_sed


def mean_final_M(result, window_days=28):
    t = result["t"]; M = result["M"]
    mask = t >= t[-1] - window_days
    return float(np.mean(M[mask]))


def mean_final_SC(result, window_days=28):
    t = result["t"]; SC = result["SC"]
    mask = t >= t[-1] - window_days
    return float(np.mean(SC[mask]))


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
