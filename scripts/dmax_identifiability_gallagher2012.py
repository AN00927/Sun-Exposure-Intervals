"""Is D_max identifiable from Gallagher 2012, and does the oral route need feedback?

Reproduces the numbers quoted in MODEL_AND_EVIDENCE.md, 'Open decisions' item 3
(2026-08-15 update). Run:

    PYTHONIOENCODING=utf-8 python3 scripts/dmax_identifiability_gallagher2012.py

Background. 'Open decisions' item 3 established (2026-08-09) that `D_max` is not
identifiable from acute UV trial data *in principle*: a 6-week sub-erythemal
protocol never drives 25(OH)D high enough for `max(0, 1 - D/D_max)` to engage,
so MSE falls monotonically to whatever upper search bound is imposed. That proof
turns on the protocol, not on the input being UV. Gallagher, Sai, Templin & Smith
2012 (Ann Intern Med 156:425-37, PMID 22431675) is a 1-year, 8-arm ORAL
dose-ranging RCT that does reach the saturating regime:

    baseline mean 39 nmol/L; arms placebo + 400..4800 IU/d
    "the dose response was curvilinear and tended to plateau at approximately
     112 nmol/L in patients receiving more than 3200 IU/d"

Corroborated by the same trial's companion paper (Yalamanchili & Gallagher 2017,
PMID 29104103, open access PMC5854154): baseline 15.6 ng/mL = 39.0 nmol/L, top
arm 46 ng/mL = 115 nmol/L, "a quadratic curve function". DiaSorin RIA, not
LC-MS/MS -- same method offset this repo documents for NHANES.

SELF-CONTAINED BY DESIGN (since 2026-08-16). The model's oral/dietary pathway
was removed on that date -- see the "ORAL / DIETARY VITAMIN D: REMOVED" note in
`vitd_model.py`. This script therefore imports nothing from the model and
integrates the documented dynamics directly, so the argument below survives
that removal and stays runnable.

WHY THE EVIDENCE STILL COUNTS WITHOUT AN ORAL PATHWAY. The oral route was the
experimental *probe*, not the thing being measured. What Gallagher constrains is
the saturating step, and Scholze et al. 2026 locate that step in the LIVER --
downstream of both the cutaneous and the oral route. So this bounds the systemic
feedback ceiling `D_max`, which is very much still a live parameter of the
sun-exposure model, regardless of how the D3 that exercised it got there.

Read the D_max numbers below as evidence about `D_max`. Do NOT read them as an
argument for reinstating oral dosing.
"""
import numpy as np

IU_TO_UG = 0.025

# Was vitd_model.DIETARY_UG_TO_NMOL_PER_L before the oral pathway was removed.
# Retained here ONLY as the historical value this analysis was run at, so the
# published numbers stay reproducible. It is not a model parameter any more,
# and it was measured to miss its own documented Cashman target by 3-9x.
HISTORICAL_DIET_COEFF: float = 0.07
ARMS_IU = [400, 800, 1600, 2400, 3200, 4000, 4800]
D0 = 39.0
DAYS = 365
OBS_PLATEAU = 112.0        # Gallagher 2012 abstract, >3200 IU/d
OBS_TOP_ARM = 115.0        # Yalamanchili 2017, 4800 IU/d, white women


def clearance(D):
    """ln(2) * D**1.72 / 2953 -- from the Datta 2017 power-law half-life.

    t_half(D) = 2953 * D**-0.72, k = ln2/t_half, so k*D = ln2 * D**1.72 / 2953.
    """
    return np.log(2.0) * D ** 1.72 / 2953.0


def integrate(ug_per_day, D_max=None, diet=HISTORICAL_DIET_COEFF,
              days=DAYS, D_start=D0, dt=0.05):
    """Forward-Euler the oral-only dynamics. D_max=None reproduces the live model."""
    D = D_start
    for _ in range(int(days / dt)):
        inp = ug_per_day * diet
        if D_max is not None:
            inp *= max(0.0, 1.0 - D / D_max)
        D = max(D + (inp - clearance(D)) * dt, 1e-9)
    return D


def main():
    print("=" * 74)
    print("D_max identifiability from Gallagher 2012 (oral dose-ranging)")
    print("=" * 74)

    print("\n1. Reference values (model as it stood 2026-08-15, oral route live)")
    print("   Recorded before the oral pathway was removed so this stays checkable.")
    print("   simulate() agreed with this integration to 0.67% at the time.")
    print(f"   {'IU/d':>6} {'simulate() then':>16} {'this integration':>17}")
    for iu, live in ((3200, 348.9), (4800, 441.1)):
        mine = integrate(iu * IU_TO_UG)
        print(f"   {iu:>6} {live:>16.1f} {mine:>17.1f}"
              f"   ({abs(mine-live)/live*100:.2f}% apart)")


    print("\n2. The pre-removal model, with no feedback on the oral route:")
    print(f"   {'IU/d':>6} {'D(365)':>9}")
    for iu in ARMS_IU:
        print(f"   {iu:>6} {integrate(iu * IU_TO_UG):>9.1f}")
    print(f"   observed plateau above 3200 IU/d: ~{OBS_PLATEAU:.0f} nmol/L")


    print("\n3. Can rescaling the input coefficient fix the SHAPE?")
    print("   Steady state solves input = ln2*D**1.72/2953, so D ~ input**(1/1.72)")
    print("   -- a power law with NO asymptote. Rescaling slides the level only.")
    print(f"\n   {'scale':>6} {'D(3200)':>9} {'D(4800)':>9} {'ratio':>7}")
    for s in (1.0, 0.5, 0.25, 0.1):
        a = integrate(3200 * IU_TO_UG, diet=HISTORICAL_DIET_COEFF * s)
        b = integrate(4800 * IU_TO_UG, diet=HISTORICAL_DIET_COEFF * s)
        print(f"   {s:>6.2f} {a:>9.1f} {b:>9.1f} {b/a:>7.3f}")
    print(f"   analytic prediction (1.5)**(1/1.72) = {(1.5)**(1/1.72):>.3f}")
    print(f"   OBSERVED ratio                      = ~1.000  <-- a true plateau")
    print("   => the ratio is invariant under rescaling. The defect is")
    print("      STRUCTURAL, not parametric. Same class as gap 27's SC dip.")

    print("\n4. Counterfactual: apply the EXISTING feedback factor to the oral")
    print("   route. Does D_max acquire an interior optimum?")
    targets = {3200: OBS_PLATEAU, 4800: OBS_PLATEAU}
    grid = np.arange(90.0, 300.0, 0.5)
    print(f"\n   {'diet coeff':>11} {'best D_max':>11} {'SSE':>8} {'D(400IU)':>10}")
    for diet in (0.020, 0.035, 0.050, HISTORICAL_DIET_COEFF, 0.100, 0.140):
        sse = [sum((integrate(iu * IU_TO_UG, D_max=dm, diet=diet) - t) ** 2
                   for iu, t in targets.items()) for dm in grid]
        i = int(np.argmin(sse))
        interior = 0 < i < len(grid) - 1
        lo = integrate(400 * IU_TO_UG, D_max=grid[i], diet=diet)
        flag = "" if interior else "  <-- PINNED AT BOUND"
        star = " *" if abs(diet - HISTORICAL_DIET_COEFF) < 1e-9 else "  "
        print(f"   {diet:>11.3f}{star}{grid[i]:>9.1f} {sse[i]:>8.2f} {lo:>10.1f}{flag}")
    print("   (* = the coefficient in use before oral removal)")

    print("\n   Interior minima throughout -- the first dataset in this repo to")
    print("   produce one for D_max. It is a BOUNDED RIDGE, not a point: D_max")
    print("   is pinned from below by the plateau itself (D approaches D_max")
    print("   from below, so D_max > 112 necessarily) and spans ~119-185 across")
    print("   a 7x range of the dietary coefficient. The repo's incumbent 150")
    print("   sits inside that interval.")
    print("\n   The D(400 IU) column is how to BREAK the ridge: 42.9 vs 80.8")
    print("   nmol/L across the coefficient range. Gallagher's per-arm means")
    print("   would separate the two parameters -- data request #15.")

    print("\n" + "=" * 74)
    print("Caveats. Oral D3 bypasses skin entirely, so this constrains the")
    print("hepatic/clearance limb ONLY -- it says nothing about K_const,")
    print("K_ATTEN, or any pigmentation channel. Assay is DiaSorin RIA. And")
    print("all of section 4 presupposes a structural change that has NOT been")
    print("made and needs calibration_approval.yaml sign-off.")
    print("=" * 74)


if __name__ == "__main__":
    main()
