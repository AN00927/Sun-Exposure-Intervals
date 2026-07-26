"""
Pragmatic SC recalibration from de Boer 2025.
Goal: update alpha_SC and SC_max_by_FST to match observed T2 values.
k_damage=0 default maintained (dip at T1 unresolved, needs individual-level data).

From de Boer Supplementary Table S1:
  FP I-III: SC 5.6 -> 6.9 um (+23%, so SC_max=0.23)
  FP IV-VI: SC 6.6 -> 9.0 um (+36%, so SC_max=0.36)

Calibrate alpha_SC: SC reaches ~90% of SC_max within 27 sessions (0.8 SED each)
  0.90 = 1 - exp(-alpha * 27 * 0.8)
  alpha = -ln(0.1) / 21.6 = 0.107

Use alpha_SC = 0.12 (slightly faster to reach plateau)
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Ahaan/sunresearch/Sun-Exposure-Intervals')
import numpy as np
from vitd_model import ModelParams, simulate, build_pulse_schedule

# Test new alpha_SC=0.12, SC_max_by_FST with de Boer protocol
params = ModelParams()
params.alpha_SC = 0.12
params.SC_max = 0.36

# Build schedule: 0.8 SED 3x/week for 9 weeks
schedule = [(i * 7/3, 0.8) for i in range(27)]

for fst, label, expected_sc_max in [(3, "FP I-III", 0.23), (5, "FP IV-VI", 0.36)]:
    params.SC_max = expected_sc_max
    r = simulate(fst=fst, schedule=schedule, duration_days=63, params=params)
    SC = r["SC"]
    t = r["t"]
    sc_T1 = float(np.interp(35, t, SC))
    sc_T2 = float(np.interp(63, t, SC))
    print(f"{label} (SC_max={expected_sc_max}):")
    print(f"  SC_T1 (week 5): {sc_T1:.4f}  (obs: {'N/A' if fst==3 else '-0.35'})")
    print(f"  SC_T2 (week 9): {sc_T2:.4f}  (obs: {'+0.23' if fst==3 else '+0.36'})")
    print(f"  SC_max reached: {sc_T2/expected_sc_max*100:.1f}%")

# Also check: how does this affect Bogh 2012 validation?
print("\n\nBogh 2012 validation with new SC parameters:")
params.SC_max = 0.23  # FST II
bogh_schedule = build_pulse_schedule(84, 1.0, 2)
r_bogh = simulate(fst=2, schedule=bogh_schedule, duration_days=84, params=params)
D_final = r_bogh["D"][-1]
D0 = r_bogh["D"][0]
print(f"  FST II: D0={D0:.1f}, D_final={D_final:.1f}, delta={D_final-D0:.1f}")
print(f"  Bogh 2012 observed: +55.8 nmol/L")
print(f"  Current model (old SC params): +41.6 nmol/L")

# Check SC values for the validation range
print("\n\nSC trajectories for validation check update:")
for fst, label in [(3, "FST III"), (5, "FST V")]:
    params.SC_max = 0.23 if fst == 3 else 0.36
    r = simulate(fst=fst, schedule=schedule, duration_days=63, params=params)
    sc_mean = float(np.mean(r["SC"][-28:]))
    print(f"  {label} final SC (28d mean): {sc_mean:.4f}")
