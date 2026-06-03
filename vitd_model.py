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

Synthesis function (Webb 2018 logarithmic dose-response, melanin-attenuated):
    S(UV, M, F) = A * ln(1 + UV) * exp(-K_ATTEN * max(0, M - M_baseline(F)))

UV(t) is a pulse train delivered in standard erythemal doses (SED).

Parameter sourcing
------------------
k_decay_D = ln(2) / 30 per day (effective)
   The plasma 25(OH)D stable-isotope half-life is ~15 days (Schoenmakers 2015),
   but the apparent decline measured in serial-sampling intervention studies
   is materially slower due to slow exchange with the adipose reservoir
   (Prentice 2017: 52% adipose decline over 12 months -> ~1 yr adipose t1/2).
   For week-to-month scheduling dynamics the effective system half-life is
   ~25-35 days, consistent with Vahavihu 2010 observation that NB-UVB-
   induced 25(OH)D elevation persists ~2 months after course end.

A = 4.50 nmol/L per ln(1+SED) unit
   Calibrated so that 1 SED in untanned FST III gives delta-D of 3.12 nmol/L,
   the lower end of Webb 2018's 3-5 nmol/L light-skin range.

alpha = 0.05 per SED
   Photoadaptation rise rate estimated from Miyamura 2011 (PMID 20979596).
   That study found UVB-induced tanning provides modest photoprotection
   while UVA-induced tanning provides essentially none. A specific SPF-3
   figure cited in earlier notes could not be confirmed from any accessible
   source (full text paywalled; open-access follow-up Coelho 2015 describes
   only "very modest photoprotective benefit" without a numerical SPF value).
   NEEDS CALIBRATION FOR FST V-VI; sensitivity sweep is the priority test.

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
   Exponential melanin penalty coefficient. Estimated; not measured in FST V-VI.
   Young 2020 measured overall vitamin D inhibition factor of only 1.3-1.4x
   between FST II and FST VI, implying K_ATTEN may be substantially LOWER
   than 2.0. Sensitivity sweep is the priority test.

M_baseline by Fitzpatrick: 0.10 (I), 0.20 (II), 0.30 (III),
                           0.45 (IV), 0.65 (V), 0.80 (VI)
   Datta 2019-style reflectance baseline.

Author: Ahaan
Last updated: 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ModelParams:
    k_decay_D: float = np.log(2) / 30.0
    A: float = 4.50
    alpha: float = 0.05
    beta: float = np.log(2) / 56.0
    M_max: float = 0.95
    K_ATTEN: float = 2.0
    D0_default: float = 25.0
    M_baseline_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.10, 2: 0.20, 3: 0.30, 4: 0.45, 5: 0.65, 6: 0.80,
    })

    def M_baseline(self, fst: int) -> float:
        return self.M_baseline_by_FST[fst]


def synthesis_per_session(sed: float, M: float, M_baseline: float,
                          params: ModelParams,
                          A_override: float = None,
                          K_ATTEN_override: float = None) -> float:
    if sed <= 0:
        return 0.0
    A_use = params.A if A_override is None else A_override
    K_use = params.K_ATTEN if K_ATTEN_override is None else K_ATTEN_override
    excess_M = max(0.0, M - M_baseline)
    return A_use * np.log(1.0 + sed) * np.exp(-K_use * excess_M)


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
             K_ATTEN_override: float = None) -> Dict:
    if params is None:
        params = ModelParams()
    M_base = params.M_baseline(fst)

    alpha_eff = params.alpha if alpha_override is None else alpha_override
    beta_eff = params.beta if k_decay_M_override is None else k_decay_M_override
    K_eff = params.K_ATTEN if K_ATTEN_override is None else K_ATTEN_override

    D = params.D0_default if D0 is None else D0
    M = M_base if M0 is None else M0

    schedule = sorted(schedule, key=lambda x: x[0])
    pulse_days = [p[0] for p in schedule]
    pulse_seds = [p[1] for p in schedule]

    def deriv(t, y):
        D, M = y
        return [-params.k_decay_D * D, -beta_eff * (M - M_base)]

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
            sol = solve_ivp(deriv, (current_t, end_t), [D, M],
                            rtol=1e-7, atol=1e-10, max_step=0.5)
            for ti, Di, Mi in zip(sol.t[1:], sol.y[0][1:], sol.y[1][1:]):
                t_record.append(ti)
                D_record.append(Di)
                M_record.append(Mi)
            D, M = sol.y[0][-1], sol.y[1][-1]
            current_t = end_t

        if (pulse_idx < len(pulse_days)
                and abs(current_t - pulse_days[pulse_idx]) < 1e-9):
            sed = pulse_seds[pulse_idx]
            delta_D = synthesis_per_session(sed, M, M_base, params,
                                            K_ATTEN_override=K_eff)
            D = D + delta_D
            M = M + alpha_eff * sed * (params.M_max - M)
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
