"""Write the manuscript's parameter table from the live code values.
===================================================================

WHY THIS EXISTS
---------------
Roadmap item R13 of the 2026-09-29 mock review asked for a parameter table in
the paper. CLAUDE.md forbids restating parameter values in prose, because
hand-typed copies are exactly what drifted out of sync before. So the table
is GENERATED: every value below is read from ``vitd_model``,
``vitd_physiology``, ``vitd_objectives``, ``vitd_scenario`` or
``scripts/risk_aware_scheduling`` at run time, and ``paper/manuscript.tex``
``\\input``s the result. Re-run this after any calibration change.

WHAT IS HAND-WRITTEN HERE, AND WHY THAT IS ALLOWED
-------------------------------------------------
Only the symbol, a short description, the unit, the bib keys and the evidence
STATUS. None of these is a number. The status of each row comes from the tag
in its ``vitd_model.py`` docstring entry or field comment (CALIBRATION /
CALIBRATED -> "Calibrated"; PRIOR -> "Prior"; a value read straight off a
source table -> "Sourced"; an unfittable modelling decision -> "Choice"). If a
tag there changes, change the status here too. Every bib key is checked
against ``paper/references.bib`` and the script fails if one is missing.

Rates are shown as half-lives, ``ln 2 / k``, computed from the live rate. The
Datta 2017 decay power law's two coefficients are recovered from two live
evaluations of ``ModelParams.k_decay_D`` rather than copied.

Parameters that do not act in the headline runs are left out and named in the
caption: ``M_max`` (read by nothing; the ceiling in force is
``M_ceiling(F)``), ``D_max`` (linear feedback mode only), ``k_damage`` (0),
body-weight dilution (``body_weight_kg`` is None), photodegradation and
transit delay (both off).

Run:  PYTHONIOENCODING=utf-8 python3 scripts/make_parameter_table.py
"""
from __future__ import annotations

import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import vitd_model as vm                                           # noqa: E402
import vitd_physiology as vp                                      # noqa: E402
import vitd_scenario as vs                                        # noqa: E402
from vitd_objectives import GAMMA_MED, SUFFICIENCY_NMOL_L         # noqa: E402
from scripts import risk_aware_scheduling as ras                   # noqa: E402

OUT_TEX = os.path.join(ROOT, "paper", "parameters.tex")
BIB = os.path.join(ROOT, "paper", "references.bib")

FSTS = (1, 2, 3, 4, 5, 6)
UNCAL_MARK = r"$^{\dagger}$"


def _num(x) -> str:
    """A stored value at full precision (CLAUDE.md): repr, integers bare."""
    if isinstance(x, (bool, int)):
        return str(x)
    x = float(x)
    return repr(int(x)) if x.is_integer() else repr(x)


def _derived(x: float) -> str:
    """A value computed here, minus float-arithmetic noise only.

    ln 2 / (ln 2 / 56.0) comes back as 56.00000000000001; the trailing digit
    is rounding error in this script, not information in the code.
    """
    return _num(float(f"{x:.12g}"))


def _by_fst(d) -> str:
    vals = [d[f] for f in FSTS]
    if len(set(vals)) == 1:
        return _num(vals[0]) + " (all)"
    return "; ".join(_num(v) for v in vals)


def _half_life(k: float) -> str:
    return _derived(math.log(2.0) / k)


def _decay_law() -> str:
    """Recover t_half(D) = c * D**b from two live evaluations."""
    d1, d2 = 25.0, 100.0
    t1 = math.log(2.0) / vm.ModelParams.k_decay_D(d1)
    t2 = math.log(2.0) / vm.ModelParams.k_decay_D(d2)
    b = math.log(t2 / t1) / math.log(d2 / d1)
    c = t1 / d1 ** b
    return rf"$t_{{1/2}} = {_derived(c)}\,D^{{{_derived(b)}}}$"


def _rows():
    p = vm.ModelParams()
    med = {f: vp._med_sed_by_fst(f) for f in FSTS}
    # (group, symbol, description, value, unit, bib keys, status, uncalibrated)
    return [
        ("Synthesis", r"$A$", r"Synthesis scale (Young 2021 anchor, joint fixed point with $K_{\mathrm{const}}$)",
         _num(p.A), r"\nmol{} per session", ["young2021pnas"], "Calibrated", False),
        ("Synthesis", r"$E_{\mathrm{source}}$", "Source spectral efficiency; solar broadband reference",
         _num(p.E_source), "--", ["webb2021", "young2021pnas", "grigalavicius2015", "osmancevic2009"], "Sourced", False),
        ("Synthesis", r"$b_{\mathrm{ref}}$", "Reference exposed body-surface fraction",
         _num(vm.BSA_REF), "fraction", ["bogh2011bjd", "bogh2010jid"], "Sourced", False),
        ("Synthesis", r"$\phi(b)$ exponent", r"Exponent of $(b/b_{\mathrm{ref}})$",
         _num(vm.BSA_EXP), "--", ["bogh2010jid", "bogh2011bjd", "bogh2012bjd"], "Calibrated", False),
        ("Pigment", r"$K_{\mathrm{const}}$", "Constitutive-pigment attenuation (DerSimonian--Laird pool)",
         _num(p.K_const), "--", ["kift2013", "felton2016", "bogh2010jid"], "Calibrated", False),
        ("Pigment", r"$M_{\mathrm{base}}(F)$", "Constitutive melanin index, FST I; II; III; IV; V; VI",
         _by_fst(p.M_baseline_by_FST), "--", ["datta2019"], "Calibrated", False),
        ("Pigment", r"$K_{\mathrm{ATTEN}}$", "Facultative (tan) attenuation",
         _num(p.K_ATTEN), "--", ["young2020"], "Calibrated", False),
        ("Pigment", r"$\Delta M_{\mathrm{tan}}$", r"Tanning headroom, $M_{\mathrm{ceil}}(F)=M_{\mathrm{base}}(F)+\Delta M_{\mathrm{tan}}$",
         _num(p.M_TAN_HEADROOM), "--", ["ravnbak2008"], "Calibrated", False),
        ("Pigment", r"$\alpha(F)$", "Melanin induction rate, FST I; II; III; IV; V; VI (not identified by data)",
         _by_fst(p.alpha_by_FST), r"SED$^{-1}$", ["ravnbak2008"], "Prior", True),
        ("Pigment", r"$UV_{1/2}(F)$", "Induction half-saturation dose, FST I; II; III; IV; V; VI",
         _by_fst(p.UV_halfmax_by_FST), "SED", ["ravnbak2008"], "Sourced", False),
        ("Pigment", r"$\ln 2/\beta$", "Tan fading half-life",
         _half_life(p.beta), "d", [], "Prior", True),
        ("Stratum corneum", r"$\alpha_{\mathrm{SC}}$", "SC growth rate",
         _num(p.alpha_SC), r"SED$^{-1}$", ["deboer2025"], "Prior", True),
        ("Stratum corneum", r"$\mathrm{SC}_{\max}(F)$", "SC thickening ceiling, FST I; II; III; IV; V; VI",
         _by_fst(p.SC_max_by_FST), "fraction", ["deboer2025", "miyamura2011"], "Prior", True),
        ("Stratum corneum", r"$\mathrm{SC}_{\max}$ range, IV--VI", "Sensitivity range carried for FST IV--VI",
         "[" + ", ".join(_num(v) for v in vm.SC_MAX_IV_VI_RANGE) + "]", "fraction",
         ["deboer2025", "miyamura2011"], "Prior", True),
        ("Stratum corneum", r"$K_{\mathrm{SC}}$", "SC attenuation coefficient",
         _num(p.K_SC), "--", [], "Prior", True),
        ("Stratum corneum", r"$\ln 2/k_{\mathrm{decay,SC}}$", "SC thinning half-life",
         _half_life(p.k_decay_SC), "d", [], "Prior", True),
        ("Feedback and decay", r"$D_{\mathrm{infl}}$", r"Inflection of $g(D)=1/(1+(D/D_{\mathrm{infl}})^n)$",
         _num(vp.HYDROXYLATION_INFLECTION_NMOL_L), r"\nmol{}", ["scholze2026"], "Sourced", False),
        ("Feedback and decay", r"$n$", r"Hill exponent of $g(D)$ (not identifiable; see S5)",
         _num(vp.HYDROXYLATION_HILL_N), "--", [], "Choice", False),
        ("Feedback and decay", r"$t_{1/2}(D)$", r"25(OH)D half-life power law, $D$ in \nmol{}, $t_{1/2}$ in d",
         _decay_law(), "d", ["datta2017"], "Sourced", False),
        ("Risk", r"$\mathrm{MED}(F)$", "Minimal erythema dose, FST I; II; III; IV; V; VI",
         _by_fst(med), "SED", ["shih2018"], "Sourced", False),
        ("Risk", r"$k_{\mathrm{int}}$", "Intermittent (BCC/melanoma) per-session exponent",
         _num(vp.INTERMITTENT_BURN_EXPONENT), "--", ["elwood1997"], "Choice", False),
        ("Risk", r"$k_{\mathrm{SCC}}$", "Cumulative-dose exponent, SCC axis",
         _num(vp.SCC_DOSE_EXPONENT), "--", [], "Sourced", False),
        ("Risk", r"$\kappa$", "Melanin attenuation of basal-layer DNA damage",
         _num(vp.MELANIN_ATTENUATION_KAPPA), "--", ["fajuyigbe2018"], "Calibrated", False),
        ("Scenario", "UV index", "Ambient UV index, every exposure day",
         _num(vs.UV_INDEX), "--", [], "Choice", False),
        ("Scenario", "Geometry factor", "Horizontal ambient to personal dose",
         _num(vs.GEOMETRY_FACTOR), "--", ["cockell2001"], "Sourced", False),
        ("Scenario", "Exposed BSA", "Exposed body-surface fraction",
         _num(vs.EXPOSED_BSA), "fraction", [], "Choice", False),
        ("Scenario", "SED per minute", "Erythemal dose rate (derived from the three above)",
         _num(vs.SED_PER_MINUTE), r"SED min$^{-1}$", [], "Derived", False),
        ("Optimisation", "Target", "25(OH)D sufficiency floor",
         _num(SUFFICIENCY_NMOL_L), r"\nmol{}", [], "Choice", False),
        ("Optimisation", r"$\gamma$", "Per-session erythema cap",
         _num(GAMMA_MED), "MED", [], "Choice", False),
        ("Optimisation", "Session cap", "Maximum session length",
         _num(ras.PRACTICAL_MAX_MINUTES), "min", [], "Choice", False),
        ("Optimisation", r"$D_0$", "Initial 25(OH)D",
         _num(ras.D0), r"\nmol{}", [], "Choice", False),
        ("Optimisation", "Horizon; ramp-in", "Simulated days; days before the floor applies",
         f"{_num(ras.HORIZON)}; {_num(ras.RAMP_IN_DAYS)}", "d", [], "Choice", False),
        ("Optimisation", "Interval grid", "Uniform intervals searched",
         ", ".join(_num(i) for i in ras.INTERVAL_GRID), "d", [], "Choice", False),
    ]


def _bib_keys():
    with open(BIB, encoding="utf-8") as fh:
        return set(re.findall(r"^\s*@\w+\{([^,\s]+),", fh.read(), flags=re.M))


def build() -> str:
    known = _bib_keys()
    rows = _rows()
    missing = sorted({k for r in rows for k in r[5] if k not in known})
    if missing:
        raise SystemExit(f"bib keys not in {BIB}: {missing}")
    out = [
        "% GENERATED by scripts/make_parameter_table.py from the live code values.",
        "% Do not edit by hand; re-run the script after any parameter change.",
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Model parameters and scenario constants, as shipped. Values are read from",
        r"the code at build time; full derivations are in the \texttt{vitd\_model.py} module",
        r"docstring. Status: \emph{Calibrated}, fitted to data with the fit recorded;",
        r"\emph{Sourced}, taken from the cited study; \emph{Prior}, not identified by any",
        r"data set; \emph{Choice}, a modelling or scenario decision; \emph{Derived},",
        r"computed from other rows. " + UNCAL_MARK + r"~Uncalibrated prior: absolute 25(OH)D",
        r"predictions depend on it. Omitted because inactive in every reported run:",
        r"$M_{\max}$, $D_{\max}$ (linear feedback only), $k_{\mathrm{damage}}$ (zero),",
        r"body-weight dilution, photodegradation and transit delay.}",
        r"\label{tab:params}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}p{0.13\linewidth}p{0.28\linewidth}p{0.21\linewidth}p{0.08\linewidth}lp{0.1\linewidth}@{}}",
        r"\toprule",
        r"Symbol & Description & Value & Unit & Status & Source \\",
        r"\midrule",
    ]
    group = None
    for g, sym, desc, val, unit, keys, status, uncal in rows:
        if g != group:
            if group is not None:
                out.append(r"\addlinespace")
            out.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{g}}}}} \\")
            group = g
        cite = rf"\citep{{{','.join(keys)}}}" if keys else "--"
        mark = UNCAL_MARK if uncal else ""
        out.append(f"{sym}{mark} & {desc} & {val} & {unit} & {status} & {cite} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def main():
    tex = build()
    with open(OUT_TEX, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tex)
    print(f"wrote {os.path.relpath(OUT_TEX, ROOT)}")
    return tex


if __name__ == "__main__":
    main()
