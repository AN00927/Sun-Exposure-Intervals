"""
melanin_parameters_report.py
============================
COMPANION SCRIPT for `docs/MELANIN_MODEL.md`. Prints nothing new: it reads the
LIVE melanin-related values out of `vitd_model.ModelParams()` and shows the
arithmetic they imply, so the prose document never has to restate a parameter
value that could then drift out of sync with the code.

Writes no file, changes no parameter, imports nothing from `exploratory/`.

Run
---
    PYTHONIOENCODING=utf-8 python3 scripts/melanin_parameters_report.py
"""

from __future__ import annotations

import os
import sys
import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams, synthesis_per_session  # noqa: E402
import vitd_calibration  # noqa: E402

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}


def hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    p = ModelParams()
    print(f"Live melanin parameters read from vitd_model.ModelParams() on "
          f"{datetime.date.today().isoformat()}")
    print("Canonical source for values and citations: the vitd_model.py module "
          "docstring.")

    # ---------------------------------------------------------------- 1 ----
    hr("1  MELANIN STATE: baselines, ceilings, tan headroom")
    print(f"\n  M_TAN_HEADROOM = {p.M_TAN_HEADROOM}")
    print(f"\n  {'FST':>4} {'M_baseline':>11} {'M_ceiling':>10} "
          f"{'alpha_base (1/SED)':>19} {'UV_halfmax (SED)':>17}")
    for f in range(1, 7):
        print(f"  {ROMAN[f]:>4} {p.M_baseline(f):>11.3f} {p.M_ceiling(f):>10.3f} "
              f"{p.alpha_by_FST.get(f, p.alpha):>19.4f} "
              f"{p.UV_halfmax_by_FST.get(f, p.UV_halfmax):>17.1f}")
    print(f"\n  alpha_max (Hill ceiling on the induction rate) = {p.alpha_max}")
    print("  NOTE: M_ceiling is M_baseline + M_TAN_HEADROOM and is NOT capped "
          "at 1.0\n        (see ModelParams.M_ceiling).")

    # ---------------------------------------------------------------- 2 ----
    hr("2  MELANIN DECAY: beta")
    t_half = float(np.log(2.0) / p.beta)
    print(f"\n  beta = {p.beta:.6f} / day   ->  half-life = {t_half:.1f} d")
    print(f"  {'fraction of the tan remaining':38} {'days':>8}")
    for frac in (0.50, 0.25, 0.10, 0.05, 0.02):
        print(f"  {frac:38.2f} {float(-np.log(frac) / p.beta):>8.0f}")
    print("\n  Ravnbak 2008 reports TOTAL fading in 4-5 months (~122-152 d);")
    print("  Ravnbak 2009 (UVA) reports 5-6 months. Compare the rows above to")
    print("  decide what 'total' has to mean for the current beta to be "
          "compatible.")

    # ---------------------------------------------------------------- 3 ----
    hr("3  INDUCTION: the Hill-modulated per-pulse rate and the pulse update")
    print("\n  alpha_eff(sed, FST) = alpha_base(FST)"
          " + (alpha_max - alpha_base) * sed^2/(sed^2 + UV_halfmax^2)")
    print("  M_after = M_ceiling - (M_ceiling - M_before) * exp(-alpha_eff * sed)")
    seds = [0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"\n  alpha_eff by FST and session dose:")
    print("  " + f"{'FST':>4}" + "".join(f"{s:>10.1f} SED" for s in seds))
    for f in range(1, 7):
        a0 = p.alpha_by_FST.get(f, p.alpha)
        uh = p.UV_halfmax_by_FST.get(f, p.UV_halfmax)
        row = []
        for s in seds:
            hill = s ** 2 / (s ** 2 + uh ** 2)
            row.append(a0 + (p.alpha_max - a0) * hill)
        print("  " + f"{ROMAN[f]:>4}" + "".join(f"{v:>14.5f}" for v in row))

    print("\n  Melanin gained from a SINGLE pulse starting from M_baseline "
          "(excess_M):")
    print("  " + f"{'FST':>4}" + "".join(f"{s:>10.1f} SED" for s in seds))
    for f in range(1, 7):
        a0 = p.alpha_by_FST.get(f, p.alpha)
        uh = p.UV_halfmax_by_FST.get(f, p.UV_halfmax)
        Mb, Mc = p.M_baseline(f), p.M_ceiling(f)
        row = []
        for s in seds:
            hill = s ** 2 / (s ** 2 + uh ** 2)
            a = a0 + (p.alpha_max - a0) * hill
            row.append((Mc - (Mc - Mb) * np.exp(-a * s)) - Mb)
        print("  " + f"{ROMAN[f]:>4}" + "".join(f"{v:>14.5f}" for v in row))

    # ---------------------------------------------------------------- 4 ----
    hr("4  ATTENUATION: the two separate exponential terms")
    print(f"\n  constitutive  exp(-K_const * M_baseline(FST)),   "
          f"K_const = {p.K_const}")
    print(f"  facultative   exp(-K_ATTEN * max(0, M - M_baseline)), "
          f"K_ATTEN = {p.K_ATTEN}")
    print(f"\n  {'FST':>4} {'constitutive factor':>20} {'penalty vs FST I':>18} "
          f"{'facultative factor at full tan':>32}")
    full = p.M_TAN_HEADROOM
    for f in range(1, 7):
        cf = float(np.exp(-p.K_const * p.M_baseline(f)))
        ff = float(np.exp(-p.K_ATTEN * full))
        print(f"  {ROMAN[f]:>4} {cf:>20.4f} {1.0 / cf:>18.2f}x {ff:>32.4f}")
    print(f"\n  A full tan (excess_M = M_TAN_HEADROOM = {full}) therefore costs "
          f"{100 * (1 - np.exp(-p.K_ATTEN * full)):.1f}% of synthesis,")
    print(f"  against a constitutive FST VI:I penalty of "
          f"{np.exp(p.K_const * p.M_baseline(6)):.2f}x.")

    # ---------------------------------------------------------------- 5 ----
    hr("5  SIZE OF THE MELANIN CHANNEL IN ONE SESSION")
    print("\n  Per-session synthesis (nmol/L) at 1 SED, D = 25 nmol/L, SC = 0,")
    print("  default E_source and BSA, untanned vs fully tanned:")
    print(f"\n  {'FST':>4} {'untanned':>10} {'full tan':>10} {'ratio':>8}")
    for f in range(1, 7):
        Mb = p.M_baseline(f)
        s_un = synthesis_per_session(1.0, Mb, Mb, 25.0, 0.0, p)
        s_tan = synthesis_per_session(1.0, Mb + full, Mb, 25.0, 0.0, p)
        print(f"  {ROMAN[f]:>4} {s_un:>10.4f} {s_tan:>10.4f} "
              f"{s_tan / s_un:>8.3f}")

    # ---------------------------------------------------------------- 6 ----
    hr("6  K_ATTEN PROVENANCE (Matsuoka 1990 deconvolution, live)")
    vitd_calibration.run(verbose=True)


if __name__ == "__main__":
    main()
