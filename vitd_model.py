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
        (representing epidermal thickening/hyperplasia photoadaptation channel)

Governing equations
-------------------
dD/dt  = S(UV(t), M(t), SC(t), D(t), F)  -  k_decay_D * D
dM/dt  = alpha * UV(t) * (M_max - M)  -  beta * (M - M_baseline(F))
dSC/dt = alpha_SC * UV(t) * (SC_max - SC)  -  k_decay_SC * SC

Synthesis function (logarithmic dose-response, melanin and SC-attenuated, feedback-inhibited):
    S(UV, M, SC, D, F) = A * ln(1 + UV) * exp(-K_const * M_baseline(F)) * 
                         exp(-K_ATTEN * max(0, M - M_baseline(F))) *
                         exp(-K_SC * SC) * E_source * max(0, 1 - D / D_max)

Parameter & Source Sourcing
---------------------------
- k_decay_D = ln(2) / 80 per day:
    Effective systemic decay rate including adipose buffering. Supported by
    Prentice 2017 (PMID: 28738961) and post-UVB cycles in Datta 2017 (DOI: 10.1039/c6pp00258g).
- A = 2.353:
    Synthesis scaling factor calibrated to preserve a 1.17 nmol/L rise for a 1 SED
    dose in untanned FST III at baseline D=25.0, SC=0.0.
- K_const = 0.50:
    Baseline (constitutive) inhibition coefficient. Calibrated to match the 1.3-1.4x
    synthesis ratio between FST II and VI reported by Young 2020 (PMID: 31883961).
- D_max = 150.0 nmol/L:
    Homeostatic feedback regulation saturation ceiling, based on Bogh 2010 (PMID: 19798069).
- UV_threshold_by_FST = {I-III: 2.0, IV: 5.0, V: 8.0, VI: 10.0}:
    Fitzpatrick-stratified minimal melanogenesis thresholds, based on Ravnbak 2008 (PMID: 18480919).
- Stratum Corneum (SC) Thickening:
    Decay k_decay_SC (28-day half-life), rise alpha_SC = 0.005, and ceiling SC_max = 0.50.
    Calibrated to represent the +34% thickness rise observed in de Boer 2025 (PMID: 39956428).
- E_source:
    Spectral efficiency coefficient (1.0 default for broadband/solar simulator;
    2.5 for Narrowband UVB 311 nm clinical phototherapy per Bogh 2012, PMID: 22524297).

Author: Ahaan
Last updated: 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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
    K_ATTEN: float = 2.0
    D0_default: float = 25.0
    D_max: float = 150.0
    E_source: float = 1.0
    # Stratum corneum thickening state parameters
    k_decay_SC: float = np.log(2) / 28.0
    alpha_SC: float = 0.005
    SC_max: float = 0.50
    K_SC: float = 1.0
    M_baseline_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.10, 2: 0.20, 3: 0.30, 4: 0.38, 5: 0.65, 6: 0.80,
    })

    def M_baseline(self, fst: int) -> float:
        return self.M_baseline_by_FST[fst]


def synthesis_per_session(sed: float, M: float, M_baseline: float,
                          D: float,
                          SC: float,
                          params: ModelParams,
                          A_override: float = None,
                          K_const_override: float = None,
                          K_ATTEN_override: float = None,
                          E_source_override: float = None,
                          D_max_override: float = None,
                          K_SC_override: float = None) -> float:
    if sed <= 0:
        return 0.0
    A_use = params.A if A_override is None else A_override
    Kc_use = params.K_const if K_const_override is None else K_const_override
    K_use = params.K_ATTEN if K_ATTEN_override is None else K_ATTEN_override
    E_use = params.E_source if E_source_override is None else E_source_override
    D_max_use = params.D_max if D_max_override is None else D_max_override
    K_SC_use = params.K_SC if K_SC_override is None else K_SC_override
    
    excess_M = max(0.0, M - M_baseline)
    constitutive_factor = np.exp(-Kc_use * M_baseline)
    facultative_factor = np.exp(-K_use * excess_M)
    sc_factor = np.exp(-K_SC_use * SC)
    feedback_factor = max(0.0, 1.0 - D / D_max_use)
    
    return A_use * np.log(1.0 + sed) * constitutive_factor * facultative_factor * sc_factor * E_use * feedback_factor


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
             K_SC_override: float = None) -> Dict:
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
                                             K_SC_override=K_SC_eff)
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
            cumulative_SED += sed
            synthesis_log.append((current_t, sed, delta_D, M, SC))

            t_record.append(current_t)
            D_record.append(D)
            M_record.append(M)
            SC_record.append(SC)
            pulse_idx += 1

    return {
        "t": np.array(t_record),
        "D": np.array(D_record),
        "M": np.array(M_record),
        "SC": np.array(SC_record),
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
    total_d = sum(item[2] for item in log)
    total_s = sum(item[1] for item in log)
    return total_d / total_s if total_s > 0 else 0.0


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
    print(f"  D/SED:   {efficiency_D_per_SED(r):.3f}")
