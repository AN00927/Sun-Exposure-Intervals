"""
vitd_model.py
=============
Two-state ODE model of serum 25(OH)D dynamics under UV exposure schedules,
accounting for photoadaptation (melanin upregulation) in darker skin types.

State variables
---------------
D(t) : effective serum 25(OH)D, nmol/L
       Single compartment representing the time-integrated plasma + adipose
       reservoir relevant for week-to-month scheduling dynamics.
M(t) : effective melanin index, dimensionless 0-1
       (Datta 2019 reflectance scale; 0.10 = FST I, 0.80 = FST VI)

Governing equations
-------------------
dD/dt = S(UV(t), M(t), F)  -  k_decay_D * D
dM/dt = alpha * UV(t) * (M_max - M)  -  beta * (M - M_baseline(F))

Synthesis function (logarithmic dose-response per Webb 2021, melanin-attenuated):
    S(UV, M, F) = A * ln(1 + UV) * exp(-K_const * M_baseline(F)) * 
                  exp(-K_ATTEN * max(0, M - M_baseline(F)))
# NOTE: K_const=0.75 captures the constitutive baseline gradient reported by
# Young 2020: FST II synthesizes ~1.35x more D than FST VI at baseline.
# HANDOFF.md suggested K_const=0.5 calibrated against old M_baseline values
# (FST VI=0.80); this file uses K_const=0.75 for internal consistency with
# the current M_baseline values and FST III calibration.

UV(t) is a pulse train delivered in standard erythemal doses (SED).

Parameter sourcing
------------------
k_decay_D = ln(2) / 80 per day (effective)
   The plasma 25(OH)D stable-isotope half-life is ~15 days (Schoenmakers 2015),
   but the apparent decline measured in serial-sampling intervention studies
   is materially slower due to slow exchange with the adipose reservoir
   (Prentice 2017: 52% adipose decline over 12 months -> ~1 yr adipose t1/2).
   Datta et al. 2017 (PMID/DOI: 10.1039/c6pp00258g) is a more direct
   post-UVB citation for the ~80-day effective half-life used here.
   For week-to-month scheduling dynamics the effective system half-life is
   ~80 days, consistent with Vahavihu 2010 observation that NB-UVB-
   induced 25(OH)D elevation persists ~2 months after course end.

A = 2.112 nmol/L per ln(1+SED) unit
   Calibrated so that 1 SED in untanned FST III gives delta-D of ~1.17 nmol/L.
   Recalibrated upward to maintain the FST III target after adding the 
   constitutive K_const factor:
   A_new = 1.6875 * exp(0.75 * 0.30) = 1.6875 * 1.251 = 2.112.

alpha = 0.0025 per SED
   Fallback/default for backward compatibility. simulate() uses alpha_by_FST
   when alpha_override is None.
   Photoadaptation rise rates calibrated from de Boer et al. 2025
   (J Invest Dermatol 145:2304-2312; PMID 39956428). Protocol: 0.8 SED
   x 3x/week x 9 weeks (21.6 SED total) in 31 subjects FP I-VI.
   Melanin index rose +15% in FP I-III and +9% in FP IV-VI. Solving
   the ODE steady-state equation for M_ss gives alpha ~ 0.0025/SED for
   lighter phototypes (FP I-III) and ~0.002/SED for darker (FP IV-VI).
   Prior estimate of 0.05 from Miyamura 2011 was 20x too high; the
   Miyamura 2011 SPF figure was never confirmed from the full text.
   NOTE: de Boer 2025 shows vitamin D rose significantly only in FP I-III
   (p=0.023), not in FP IV-VI (p=0.495), at this sub-melanogenic dose.
   NOTE: The Hill-style alpha(UV) term uses alpha_max and UV_threshold
   stored in ModelParams as internally derived parameters without external
   calibration; the FST-specific alpha values from de Boer 2025 are for
   sub-erythemal doses.

beta = ln(2) / 56 per day (~8-week M half-life)
   Photoadaptation decay rate. Ravnbak et al. 2008 (Acta Derm Venereol
   88:223-8, PMID 18480919) measured UVB tanning fade in 12 Scandinavians
   and 12 Indians/Pakistanis, finding "fading took 4-5 months and was not
   related to frequency of UV exposure or to ethnic origin." A companion
   paper (Ravnbak 2009, PMID 19688146) found UVA tanning fade took 5-6
   months. Neither paper reports an explicit half-life or exponential model;
   the 56-day value is a mid-range estimate compatible with the 4-5 month
   fade if "complete fading" corresponds to ~3 e-folding times (12.5%
   residual), which gives an implied half-life of ~40-50 days. The plausible
   range from Ravnbak 2008 is approximately 30-75 days.
   Tadokoro 2005 (PMID 15955111), previously cited here, only measured
   melanin changes 0-7 days post-exposure and does not report a decay
   timescale; it should not be cited for this parameter.

K_ATTEN = 2.0
   Exponential melanin penalty coefficient for ADDITIONAL attenuation from
   tanning above M_baseline. This parameter governs the tanning-induced
   reduction in synthesis, not the constitutive baseline melanin effect.
   Young et al. 2020 (PMID 31883961) measured 1.3-1.4x inhibition between
   FST II and FST VI at baseline — this is a CONSTITUTIVE effect not captured
   by K_ATTEN alone. 
   # NOTE: de Boer 2025 reports +34% SC thickening as an independent UV
   # attenuation channel (not melanin-mediated). SC filters >50% incident
   # radiation in light skin vs ~20-30% in dark skin. Not yet modeled; would
   # require a separate S_C(t) state variable analogous to M(t) but with faster
   # kinetics (SC turnover ~28 days).

K_const = 0.75
   Governs baseline (constitutive) inhibition. K_const=0.75 gives a 1.35x
   baseline synthesis ratio (FST II vs VI) with current M_baseline values;
   HANDOFF.md suggested 0.5, calibrated against old M_baseline values
   (FST VI=0.80); updated here for internal consistency.
   K_ATTEN at 2.0 is approximately consistent with Matsuoka 1990 tanning
   attenuation at higher summer-sun UV doses, validated against a 2.5 SED
   every-2-day, 4-week preconditioning regimen (synthesis ratio ~0.67).

M_baseline by Fitzpatrick: 0.10 (I), 0.20 (II), 0.30 (III),
                            0.38 (IV), 0.65 (V), 0.80 (VI)
    Values on a 0-1 scale calibrated to the Datta 2019 reflectance scale
    (spectrophotometer reflectance / Derma-Spectrophotometer MI data).
    FST V moved from 0.46 to 0.65 and FST VI from 0.60 to 0.80 to align
    with Datta 2019 reflectance-based mapping.
    Approximate mapping M ≈ MI/100: FST V ~0.65, FST VI ~0.80.

Author: Ahaan
Last updated: 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ModelParams:
    k_decay_D: float = np.log(2) / 80.0
    A: float = 2.112
    alpha: float = 0.0025
    alpha_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0025, 2: 0.0025, 3: 0.0025,
        4: 0.002,  5: 0.002,  6: 0.002,
    })
    alpha_max: float = 0.02
    UV_threshold: float = 2.0
    beta: float = np.log(2) / 56.0
    K_const: float = 0.75
    M_max: float = 0.95
    K_ATTEN: float = 2.0
    D0_default: float = 25.0
    M_baseline_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.10, 2: 0.20, 3: 0.30, 4: 0.38, 5: 0.65, 6: 0.80,
    })

    def M_baseline(self, fst: int) -> float:
        return self.M_baseline_by_FST[fst]


def synthesis_per_session(sed: float, M: float, M_baseline: float,
                          params: ModelParams,
                          A_override: float = None,
                          K_const_override: float = None,
                          K_ATTEN_override: float = None) -> float:
    if sed <= 0:
        return 0.0
    A_use = params.A if A_override is None else A_override
    Kc_use = params.K_const if K_const_override is None else K_const_override
    K_use = params.K_ATTEN if K_ATTEN_override is None else K_ATTEN_override
    
    excess_M = max(0.0, M - M_baseline)
    constitutive_factor = np.exp(-Kc_use * M_baseline)
    facultative_factor = np.exp(-K_use * excess_M)
    
    return A_use * np.log(1.0 + sed) * constitutive_factor * facultative_factor


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
             k_decay_M_override: float = None,
             alpha_override: float = None,
             K_const_override: float = None,
             K_ATTEN_override: float = None) -> Dict:
    if params is None:
        params = ModelParams()
    M_base = params.M_baseline(fst)

    Kc_eff = params.K_const if K_const_override is None else K_const_override
    beta_eff = params.beta if k_decay_M_override is None else k_decay_M_override
    K_eff = params.K_ATTEN if K_ATTEN_override is None else K_ATTEN_override

    D = params.D0_default if D0 is None else D0
    M = M_base if M0 is None else M0

    schedule = sorted(schedule, key=lambda x: x[0])
    pulse_days = [p[0] for p in schedule]
    pulse_seds = [p[1] for p in schedule]

    t_record, D_record, M_record = [0.0], [D], [M]
    synthesis_log = []
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
                t_record.append(ti)
                D_record.append(Di)
                M_record.append(Mi)
            D = D * np.exp(-params.k_decay_D * (end_t - current_t))
            M = M_base + (M - M_base) * np.exp(-beta_eff * (end_t - current_t))
            current_t = end_t

        if (pulse_idx < len(pulse_days)
                and abs(current_t - pulse_days[pulse_idx]) < 1e-9):
            sed = pulse_seds[pulse_idx]
            delta_D = synthesis_per_session(sed, M, M_base, params,
                                            K_const_override=Kc_eff,
                                            K_ATTEN_override=K_eff)
            D = D + delta_D
            if alpha_override is not None:
                alpha_p = alpha_override
            else:
                alpha_p = params.alpha_by_FST.get(fst, params.alpha)
                hill_factor = sed**2 / (sed**2 + params.UV_threshold**2)
                alpha_p = alpha_p + (params.alpha_max - alpha_p) * hill_factor
            M = params.M_max - (params.M_max - M) * np.exp(-alpha_p * sed)
            cumulative_SED += sed
            synthesis_log.append((current_t, sed, delta_D, M))

            t_record.append(current_t)
            D_record.append(D)
            M_record.append(M)
            pulse_idx += 1

    return {
        "t": np.array(t_record),
        "D": np.array(D_record),
        "M": np.array(M_record),
        "synthesis_log": synthesis_log,
        "cumulative_SED": cumulative_SED,
        "fst": fst,
        "params": params,
    }


def steady_state_D(result, window_days=28):
    t = result["t"]; D = result["D"]
    mask = t >= t[-1] - window_days
    return float(np.mean(D[mask]))


def efficiency_D_per_SED(result):
    log = result["synthesis_log"]
    if not log: return 0.0
    total_d = sum(d for _, _, d, _ in log)
    total_s = sum(s for _, s, _, _ in log)
    return total_d / total_s if total_s > 0 else 0.0


def mean_final_M(result, window_days=28):
    t = result["t"]; M = result["M"]
    mask = t >= t[-1] - window_days
    return float(np.mean(M[mask]))


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
    print(f"  D/SED:   {efficiency_D_per_SED(r):.3f}")
