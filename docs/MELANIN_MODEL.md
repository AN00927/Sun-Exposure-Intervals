# The melanin sub-model: equations, provenance, and what it can and cannot support

**Scope.** This document isolates the melanin dynamics of the vitamin D
scheduling model: the state equation as *implemented*, every melanin-related
parameter, the publication each one comes from, and the arithmetic that turns
a published measurement into a model constant. It also states, in a mandatory
limitations section, what each parameter does *not* rest on.

**On numbers in this document.** This repository has a documented history of
parameter values being copied into prose and then drifting out of sync with the
code (`CLAUDE.md:34-41`). The canonical source for every value is the
`vitd_model.py` module docstring and `vitd_model.ModelParams`; nothing here
supersedes them. To avoid creating a second table that can drift, the live
values and the derived quantities that depend on them are printed by a
companion script:

```bash
PYTHONIOENCODING=utf-8 python3 scripts/melanin_parameters_report.py
```

That script reads `vitd_model.ModelParams()` at run time and prints the
baselines, ceilings, induction rates, decay timescale, both attenuation terms
and the Matsuoka deconvolution. Where a number appears inline below because a
derivation is unreadable without it, it is marked **[read from code
2026-08-23]** and the canonical location is given. Anything marked that way
should be re-checked against the script rather than trusted from this page.

Everything is cited `file:line` into this repository.

---

## 1. The melanin state equation as implemented

### 1.1 What the docstring says

The module docstring states a continuous ODE (`vitd_model.py:21`):

```
dM/dt = alpha * UV(t) * (M_max - M)  -  beta * (M - M_baseline(F))
```

with `M(t)` an "effective melanin index, dimensionless 0-1, Datta 2019
reflectance scale; 0.0 = FST I, 1.0 = FST VI" (`vitd_model.py:12-13`). The
manuscript states the same equation (`paper/manuscript.tex:141`).

### 1.2 What the code actually does

`simulate()` never integrates that ODE. It runs a **hybrid pulse process**: an
exact analytic relaxation between sessions, and an instantaneous saturating
jump at each session.

**Between pulses** — sub-stepped at `dt = 0.5 d` (`vitd_model.py:2462-2482`):

```python
M = M_base + (M - M_base) * np.exp(-beta_eff * delta_time)   # vitd_model.py:2475
```

This is the *exact* solution of the decay half of the ODE with `UV = 0`, not a
finite-difference approximation of it, so there is no integration error in the
decay channel.

**At each pulse** (`vitd_model.py:2510-2516`):

```python
alpha_p = params.alpha_by_FST.get(fst, params.alpha)
UV_halfmax_p = params.UV_halfmax_by_FST.get(fst, params.UV_halfmax)
hill_factor = sed**2 / (sed**2 + UV_halfmax_p**2)
alpha_p = alpha_p + (params.alpha_max - alpha_p) * hill_factor
M = M_ceil - (M_ceil - M) * np.exp(-alpha_p * sed)
```

This is the exact solution of the *induction* half of the ODE over a session,
under the assumption that a session is a Dirac pulse: UV is treated as
infinitely intense over zero elapsed time, with fluence `∫UV dt = sed`, and the
decay term is switched off for the duration. Session dose enters twice — once
through `exp(-alpha_p * sed)` and once through `alpha_p` itself.

### 1.3 Four discrepancies between the stated ODE and the implementation

These are worth stating explicitly, because a reader who reads only the
docstring or only the manuscript will have a different model in mind than the
one that produced the results.

1. **The induction rate is not a constant.** `alpha` is Hill-modulated by
   per-session dose (`vitd_model.py:2510-2515`), rising toward `alpha_max` as
   `sed` exceeds a per-FST half-max dose. The docstring ODE
   (`vitd_model.py:21`) omits this entirely. The manuscript *does* state it
   (`paper/manuscript.tex:153`), so the discrepancy is docstring-vs-code, not
   paper-vs-code. Consequence, stated in the manuscript: "a single large
   session induces more tanning per unit dose than several small ones."

2. **The ceiling is per-FST, not the global `M_max`.** The code uses
   `M_ceil = params.M_ceiling(fst)` (`vitd_model.py:2420`), defined as
   `M_baseline(fst) + M_TAN_HEADROOM` and deliberately **not** capped at 1.0
   (`vitd_model.py:2123-2136`). The docstring's `M_max` symbol survives as
   `ModelParams.M_max` (`vitd_model.py:1720`), and **nothing in the core
   pipeline reads it** — a grep for `.M_max` finds only the
   `M_max_override` plumbing at `vitd_model.py:2362,2420`. It is a dead
   parameter that still appears in the published equation. The reason for the
   change is recorded at `vitd_model.py:1706-1719`: a single global `M_max=0.95`
   sat *below* `M_baseline_by_FST[6] = 1.00`, so the saturating update made UV
   exposure *lighten* FST VI skin — an inversion that mattered because FST V-VI
   is this project's target population.

3. **Pulses are instantaneous.** No decay occurs during a session, and
   synthesis is evaluated at the *pre-pulse* melanin state
   (`vitd_model.py:2491-2499` calls `synthesis_per_session(sed, M, M_base, ...)`
   before `M` is updated at `:2129`). Within-session self-shielding by melanin
   induced during that same session is therefore not represented. At realistic
   per-session doses the induced increment is ~1e-3 on the melanin scale
   (companion script, section 3), so this is a small approximation, but it is
   an approximation.

4. **Decay is toward the per-FST constitutive baseline, not toward zero.**
   `M_base = params.M_baseline(fst)` (`vitd_model.py:2398`) is the floor. Only
   the *facultative* excess fades. This matches the ODE and is noted here only
   because it is what makes the constitutive/facultative split in §3 coherent.

---

## 2. Parameter-by-parameter provenance

Values: run the companion script. Citations: `vitd_model.py` docstring lines
given per row. Status flags in the right-hand column are the manuscript's own
(`paper/manuscript.tex:187-201`).

### 2.1 `alpha` / `alpha_by_FST` — melanin induction base rate (1/SED)

- **Code:** `vitd_model.py:1648-1651`. Lower for FST IV-VI than for I-III.
- **Publication:** Miyamura Y, Coelho SG, Schlenz K, Batzer J, Smuda C, Choi W,
  Brenner M, Passeron T, Zhang G, Kolbe L, Wolber R, Hearing VJ. "The deceptive
  nature of UVA tanning versus the modest protective effects of UVB tanning on
  human skin." *Pigment Cell & Melanoma Research* 2011; 24(1):136-147.
  doi:10.1111/j.1755-148X.2010.00764.x, PMID 20979596, PMC3021652.
  **Citation verified 2026-08-25** (previously a `TODO verify` stub). The PDF
  is now on disk as `Research_Papers/The deceptive nature of UVA tanning versus
  the modest protective effects of UVB tanning on human skin.pdf`.
  *Historical note:* an earlier revision of this document recorded that the
  paper "does not appear to be on disk" after a filename search. That was a
  false negative of exactly the kind `CLAUDE.md` warns about — the search was
  by filename, and the file was obtained within minutes once searched by
  metadata instead.
- **What it measured:** 7 volunteers, FST II-III, three 2x2 cm sites on the
  back exposed 5x/week at 0.4 MED in week 1 and 5x/week at 0.5 MED in week 2
  (10 exposures, 4.5 MED cumulative), to UVA, UVB, or UVA+UVB. Endpoint:
  epidermal melanin content by Fontana-Masson densitometry, plus stratum
  corneum thickness and CPD damage (Table 1).
- **Arithmetic from measurement to parameter:** `scripts/fit_melanin_kinetics.py`
  (added 2026-08-25) now performs the fit that was previously missing, pooling
  Miyamura 2011 with Ravnbak 2008 (UVB) and Ravnbak 2009 (UVA).
  **Its conclusion is that `alpha` is NOT identifiable from this evidence.**
  Profiling `alpha` over a 400-fold range (0.0025 to 1.0) moves the
  weighted residual by less than 25%, because Ravnbak dosed at only two
  exposure counts (6 and 12) and the tan is already near plateau at 6. The data
  constrain *where* tanning saturates, not *how fast* it gets there; `alpha`
  and the ceiling trade off directly. The shipped 0.0025 is **not contradicted**
  by the data but is **not supported** by it either. Status stays PRIOR.
  Closing this needs a UVB study with intermediate timepoints during build-up.
- **Corroboration, not derivation:** de Boer 2025 supports the *direction*
  (FST-stratified melanin-index gains: MI rose 15% in phototypes I-III versus
  9% in IV-VI over a nine-week protocol — `paper/manuscript.tex:112,191`),
  which is consistent with `alpha_by_FST` being smaller for darker skin. Note
  this is a *relative* MI gain, while Ravnbak 2008's own conclusion is that
  *absolute* pigment increase is independent of skin type and pre-exposure
  pigmentation. The two are not in conflict, but only the relative reading
  supports the model's FST gradient.
- **Manuscript status flag:** PRIOR.

### 2.2 `alpha_max` — induction rate ceiling at high per-session dose

- **Code:** `vitd_model.py:1652`.
- **Publication:** none. `paper/manuscript.tex:193`: "Hill-form ceiling; not
  independently sourced."
- **Manuscript status flag:** PRIOR.

### 2.3 `UV_halfmax_by_FST` — Hill half-max dose for induction (SED)

- **Code:** `vitd_model.py:1654-1658`; docstring entry `vitd_model.py:463-464`.
- **Publication:** Ravnbak et al. 2008, *Acta Derm Venereol* 88:223-228, PMID
  18480919. Full text is in the repo as `Research_Papers/Skin pigmentation
  kinetics after UVB exposure.pdf`.
- **What it measured:** minimal melanogenesis dose (MMD) — "the UV dose to
  elicit just perceptible tanning 7 days after a single UV exposure" — in 12
  Scandinavians and 12 Indians/Pakistanis (FST II-V), for a solar simulator and
  for narrowband UVB, after 1, 6 and 12 exposures. Table IV, per-exposure dose
  in SED (cumulative in parentheses), read from the PDF text layer
  **[read from paper 2026-08-23]**:

  | FST | Solar, 1 exp | Solar, 6 exp | Solar, 12 exp | nUVB, 1 | nUVB, 6 | nUVB, 12 |
  |---|---|---|---|---|---|---|
  | II | 5.3 | 2.8 | 1.7 | 5.2 | 2.7 | 1.6 |
  | III | 7.5 | 3.9 | 2.2 | 6.6 | 3.1 | 2.0 |
  | IV | 9.6 | 4.6 | 3.4 | 8.3 | 4.2 | 2.9 |
  | V | 10.5 | 5.7 | 4.0 | 8.9 | 4.7 | 3.6 |

- **Arithmetic from measurement to parameter:** the docstring says the values
  are aligned to the "post-adaptation multi-exposure MMD"
  (`vitd_model.py:463-464`); the manuscript adds that it is Table IV and
  explicitly *not* the single-exposure MMD of Table I
  (`paper/manuscript.tex:195`). **I could not reproduce the model values from
  any single column of Table IV by any stated arithmetic.** The model's FST
  I-III value matches the 12-exposure columns for FST II-III (1.6-2.2) well;
  its FST IV value is roughly the 6-exposure FST IV figure; its FST V value
  exceeds every multi-exposure entry in the table and sits between the
  single-exposure III and IV figures. The mapping is therefore best described
  as **ordinal alignment** — monotone in FST, inside the range Ravnbak spans —
  rather than a transcription. FST VI has no Ravnbak datapoint at all (the
  study stops at FST V) and is an extrapolation.
- **Manuscript status flag:** PRIOR.

### 2.4 `beta` — facultative melanin decay rate (1/day)

- **Code:** `vitd_model.py:1659`, written as `np.log(2) / 56.0`, i.e. a 56-day
  half-life is hard-coded into the expression.
- **Publication:** Ravnbak 2008 (`Research_Papers/Skin pigmentation kinetics
  after UVB exposure.pdf`) and Ravnbak 2009, the UVA companion on the same
  24-volunteer cohort (`Research_Papers/Skin Pigmentation Kinetics after
  Exposure to Ultraviolet A - amandersson,+3001.pdf`).
- **What it measured:** *time to total fading of tan*, measured at only three
  timepoints — 1 week, 2 months and 4 months after the last exposure. Group
  means: 132 d (nUVB, 6 exposures), 134 d (solar, 6), 137 d (nUVB, 12), 175 d
  (solar, 12). No half-life, no rate constant, and no exponential fit is
  reported anywhere in the paper.
- **The methodological detail that makes this fittable after all.** Ravnbak's
  Methods state exactly how "days to total fading" was produced: *"Time to
  total fading was determined as the day when the pigmentation value was equal
  to pre-exposure pigmentation. This was determined by a linear regression
  model based on 2 or 3 points (pigmentation 1 week, 2 months and 4 months
  after the last UV exposure)."* That is a **straight line fitted to decaying
  data and extrapolated to zero**. The reported number is therefore not a decay
  timescale, and should never be read as one — but the procedure is fully
  specified, so it can be inverted.
- **Arithmetic from measurement to parameter:**
  `scripts/fit_melanin_kinetics.py` §3 (added 2026-08-25) inverts it: assume
  exponential decay at rate `beta`, sample it on Ravnbak's own timepoints, fit
  an OLS line, take the x-intercept, and solve for the `beta` that reproduces
  the published figure. Read forward, this says:

  | true half-life (d) | what Ravnbak's procedure would report (d) |
  |---|---|
  | 30 | 119 |
  | 40 | 131 |
  | 48 | 141 |
  | 56 | 151 |
  | 70 | 170 |
  | 90 | 198 |

  Ravnbak reported 132-175 d for the UVB arms, which is matched by any true
  half-life in roughly **40-70 d**. The shipped 56 d sits near the middle of
  that band. A point estimate would over-read the data: Ravnbak mixed 2-point
  and 3-point regressions in unreported proportion (the 2-point variant was
  used "when fading was completed at the 4 months control"), and that choice
  alone moves the answer from ~48 d to ~74 d.
- **What changed, therefore, is provenance and not value.** The previous
  account was explicit that the number was a *choice*: "a mid-range estimate
  compatible with Ravnbak 2008 if complete fading requires ~3 e-folding times"
  (`RESULTS.md:559`). It is now the inversion of a published procedure, with a
  stated supported range (40-70 d) rather than an assumption about how many
  e-foldings "total fading" means. **No value change is proposed.**
- **Do not use the UVA arm for this.** Ravnbak 2009 reports longer fading (173
  and 185 d, implying t½ ~72-81 d), but Miyamura 2011 Table 1 shows UVA raises
  epidermal melanin content not at all (8.62 vs 9.03 a.u., NS) while producing
  a comparable visible tan. UVA "fading" is the decay of photooxidised existing
  pigment, not of the melanin the model's `M` represents. See §2.1 and
  `scripts/fit_melanin_kinetics.py` §4.
- **Manuscript status flag:** VERIFIED (`paper/manuscript.tex:197`). That flag
  is now better supported than when §4.2 was written, but "VERIFIED" still
  overstates a parameter whose supported range spans 40-70 d. DERIVED, with the
  range quoted, would be the accurate label.
- **Superseded citation:** the value was previously attributed to Tadokoro
  et al. 2005. That citation is now forbidden (`CLAUDE.md:111`); see §4.2.

### 2.5 `M_TAN_HEADROOM` — facultative headroom above constitutive pigment

- **Code:** `vitd_model.py:1705`; used by `ModelParams.M_ceiling`
  (`vitd_model.py:2123-2136`). **Changed 2026-08-25: 0.20 -> 0.238**
  (`calibration_approval.yaml`, entry "M_TAN_HEADROOM 0.20 -> 0.238 (melanin
  ceiling)", APPROVED). Do not restate the old value elsewhere in this
  repository as current.
- **Publication:** Ravnbak MH, Philipsen PA, Wulf HC. "Skin pigmentation
  kinetics after UVB exposure." *Acta Derm Venereol* 2008; 88:223-228. PMID
  18480919. Table II (absolute pigmentation-% increase, 6 and 12 exposures at
  0.8/0.6/0.4 MMD) and Table I (per-FST pre-exposure pigmentation-% and
  single-exposure MMD in SED), plus Miyamura 2011 (PMID 20979596) as an
  out-of-sample check and Ravnbak 2009 (*Acta Derm Venereol* 89:357-363, the
  UVA companion on the same 24-volunteer cohort) as a **negative control
  only** — see below.
- **Why the old value was wrong, not just unsourced.** Before 2026-08-25 the
  stated rationale for 0.20 was structural, not empirical: headroom "keeps
  FST I-III behaviour close to the old global 0.95 ceiling while making
  tanning monotone for every FST" (**[POINTER NO LONGER RESOLVES,
  checked 2026-09-21]** originally cited as `vitd_model.py:632-633`; that
  exact phrase does not appear anywhere in `vitd_model.py` today — I grepped
  for both "tanning monotone for every FST" and "old global 0.95 ceiling" and
  found no match. This is deliberately quoted *historical* text describing the
  pre-2026-08-25 rationale, which the M_TAN_HEADROOM calibration this
  paragraph is about has since superseded; the file location that once held it
  has since been overwritten by unrelated docstring growth elsewhere in the
  same module (line numbers in this large, actively-edited docstring have
  drifted repeatedly). The quote itself is not in doubt — it is reproduced
  faithfully from an earlier revision of this document — only the file:line
  pointer is stale and I could not find a current location to repoint it to).
  `scripts/fit_melanin_kinetics.py` (added 2026-08-25) mapped Ravnbak
  2008 Table II's six measured tans (6/12 exposures x 0.8/0.6/0.4 MMD) onto
  the model's 0-1 melanin index, anchored on the four per-FST pre-exposure
  pigmentation-% values from Table I. Those six measured tans span **dM =
  0.192059 to 0.237724**. The old headroom of 0.20 sat *below* four of the
  six, so the model could not represent a Ravnbak tan at *any* value of
  `alpha` — the ceiling, not the induction rate, was the binding constraint.
- **Arithmetic from measurement to parameter.** Refitting with the headroom
  free (rather than fixed at 0.20) returns different point estimates
  depending on which of three candidate scale maps is used to convert
  Ravnbak's instrument units onto the model index: 0.230880 (power map),
  0.268013 (linear map), and an intermediate quadratic-map value — the free-fit
  range is **[0.230880, 0.268013]**. `alpha` cannot resolve which map is
  correct: the headroom value is stable at 0.221-0.229 across a four-decade
  profile of `alpha` (0.0025-1.0), confirming the ceiling estimate does not
  depend on the (unidentifiable) induction-rate assumption.
- **Value adopted: 0.238.** This is the largest tan Ravnbak actually measured
  (dM = 0.237724 exactly, rounded up at the third decimal), chosen by the
  repository owner in preference to the power-map point estimate 0.230880,
  which would have left the two largest Ravnbak tans unrepresentable. 0.238
  sits inside the free-fit range [0.230880, 0.268013].
- **Why Ravnbak 2009 (UVA) is a negative control, not extra data.** Miyamura
  2011 Table 1 shows UVA raises epidermal melanin content not at all (8.62 vs
  9.03 a.u. control, not significant) while producing a visually and
  instrumentally comparable tan — yet Ravnbak 2009 measures UVA reflectance
  tans within 1-15% of the UVB ones on the same instrument. A
  reflectance-measured tan therefore carries a variable, source-dependent
  amount of actual melanin, and fitting the model's *attenuating* melanin
  index to UVA data would credit it with photoprotection Miyamura shows does
  not exist. Ravnbak 2009 is used only to bound how much of a reflectance
  pigmentation-% change is non-melanin optical change
  (`scripts/fit_melanin_kinetics.py`, `uva_negative_control()`); it does not
  contribute to the headroom estimate.
- **What this does NOT approve.** `calibration_approval.yaml`'s `not_covered`
  field is explicit: this entry does not approve any change to `alpha`,
  `alpha_by_FST`, `alpha_max` or `UV_halfmax_by_FST`. The same fit shows those
  are **not identifiable** from this evidence — profiling `alpha` from 0.0025
  to 1.0 (a factor of 400) moves the weighted residual by under 25%,
  because Ravnbak used only two exposure counts (6 and 12) and the tan is
  already near plateau at 6. They stay PRIOR at their shipped values; do not
  describe them as fitted or calibrated anywhere in this repository.
- **Consequence, not yet quantified.** Raising the ceiling increases
  facultative self-shielding reachable at high cumulative dose and therefore
  slightly reduces synthesis late in long protocols. It interacts with
  `K_ATTEN` (facultative attenuation, active at 0.288972). The downstream effect
  on headline numbers has not been re-quantified in this document — see §5.5.
- **Manuscript status flag:** was PRIOR; still not a fitted value (the
  repository owner chose 0.238 as the *maximum* observed tan, not a
  least-squares fit), so PRIOR remains the accurate label, but it is no
  longer a bare structural choice — it is now bounded below by measured
  Ravnbak data rather than unconstrained.

### 2.6 `M_baseline_by_FST` — constitutive melanin index

**[RE-ANCHOR 2026-09-22, decision A3 option 1C — READ THIS FIRST. The
"Current shipped values" table just below (2026-09-21 vintage) is now ALSO
superseded, on both the VALUE and the MAPPING.]** The ITA -> M mapping
itself was re-scaled from `M = 1 - (ITA_U + 30) / 85` to
`M = (55 - ITA_U) / 121`, and the clip at M = 1.0 discussed below (the
FST VI clip finding) was RELEASED -- FST VI can now exceed 1.0. Current
shipped values are **[read from code: `ModelParams.M_baseline_by_FST`]**,
per this repository's "do not restate parameter values" rule; do not quote
the 2026-09-21 table below as current. Derivation: `vitd_model.py`'s
`M_baseline_by_FST` docstring entry, the decision-A3 block
(`vitd_model.py:736-744`); approval: `calibration_approval.yaml`, decision
A3 (see also §2.7's matching RE-ANCHOR note, since the same change moved
`K_const`). The table immediately below is the 2026-09-06-to-2026-09-22
shipped table, kept for history only.

**[UPDATED 2026-09-21] Superseded 2026-09-06 — read this before the rest of
the section.** The table this subsection originally documented (the
Young-2020-anchored one derived below) is **no longer what ships**. In the
same user-approved, coupled change that refitted `K_const` 0.98 -> 1.15
(§2.7), `M_baseline_by_FST` was re-anchored to Narea-Jimenez et al. 2026
(*Biomed Phys Eng Express* 12:025078, DOI 10.1088/2057-1976/ae6458; n=1005
diffuse-reflectance spectra, Venezuelan mestizo dermatology outpatients,
clinician-assessed FST, constitutive site = inner left forearm 8 cm proximal
to the elbow). Values as of 2026-09-21 (now themselves superseded again by
the RE-ANCHOR note above) **[read from code 2026-09-21]**; preserved in the
code-comment provenance at `vitd_model.py:2046-2057` (the field default
itself has since been overwritten by the decision-A3 rescale, see above):

| FST | shipped `M_baseline` |
|---|---|
| I | 0.0381 |
| II | 0.1715 |
| III | 0.3231 |
| IV | 0.5241 |
| V | 0.7539 |
| VI | 1.00 (raw 1.0427, clipped) |

Same mapping as the Young-2020-anchored table below, `M = 1 - (ITA_U + 30) /
85` clipped to [0, 1], applied to Narea's constitutive ITA means instead of
Young 2020's. **Recomputed independently here [recomputed from the source ITA
means 2026-09-21]** from the per-FST ITA means quoted in the docstring
(`vitd_model.py:807-810`) — I 51.76 (n=25), II 40.42 (n=95), III 27.54
(n=369), IV 10.45 (n=400), V −9.08 (n=84), VI −33.63 (n=32) — this reproduces
the shipped table to 4 decimal places (e.g. FST V: `1 - (-9.08 + 30)/85 =
0.7538823...` → 0.7539; FST VI: `1 - (-33.63 + 30)/85 = 1.0427058...` →
clipped to 1.00), confirming the shipped numbers are the unrounded mapping
output, not a further-rounded transcription.

**Why this matters for dark-skin priority.** The two cells this project's
standing directive cares about most moved from n = 4 each (Young 2020, below)
to n = 84 (FST V) and n = 32 (FST VI) — a 21× and 8× increase respectively —
and the two source cohorts *agree*: Narea's FST V and VI constitutive ITA
means, run through this module's mapping, sit within 0.026 and 0.043 (pre-clip)
of the old Young-2020-derived values. See §4.4 for the fuller picture,
including a problem this larger dataset *surfaces* that the old n=4 framing
could not have shown: the clip at M=1.0 is not a rare edge case for FST VI —
it is where the group *mean* sits — and an estimated 59.02% of Narea's own
FST VI individuals fall beyond it (`vitd_model.py:855-856`, [EVIDENCE NOTE
2026-09-06]).

**Caveat, unresolved.** Narea's cohort is Venezuelan mestizo, not
African-ancestry, and its instrument reports ITA and melanosome fraction, not
a Mexameter melanin index. Per the dark-skin-priority directive, this is real
corroboration of *magnitude*, not closure of the FST V-VI evidence gap — there
is still no outdoor, African-ancestry-cohort validation of either cell
anywhere in this repository.

**What follows is the PRE-2026-09-06, Young-2020-anchored derivation.** It is
retained in full because (a) it is the arithmetic `K_const` was refitted
alongside — the M-gap this derivation computes is exactly what changed and
forced the `K_const` refit (§2.7) — and (b) the reproducibility check below
is itself still-valid methodology, even though its output table is
superseded.

- **Code (historical table, superseded):** docstring `vitd_model.py:704-725`.
- **Publication / data:** the individual-level Young 2020 trial dataset held in
  this repo as `Papers_and_Datasets/Final skin type vit D database
  26_03_2020.xlsx`, column `ITA_Unexposed` (individual typology angle of
  unexposed skin, degrees).
- **Arithmetic from measurement to parameter:** stated in code as
  `M = 1 - (ITA_U + 30) / 85`, clipped to [0, 1] (`vitd_model.py:707`;
  implemented as `ModelParams.M_baseline()`, `vitd_model.py:2120-2121`
  **[repointed 2026-09-21]** — the historical `:1741-1745` pointer has drifted
  with docstring growth). That is a linear map sending ITA = +55° to M = 0 and ITA = −30°
  to M = 1. **This one I could reproduce.** Pooling the `ArimedB irradiated`
  and `ARIMED Controls` sheets (n = 48 rows with a non-null `ITA_Unexposed`)
  and taking per-FST means of `ITA_Unexposed` gives
  **[recomputed from the repo dataset 2026-08-23]**:

  | FST | n | mean ITA_U (deg) | M from the mapping | model value |
  |---|---|---|---|---|
  | II | 13 | 49.7 | 0.062 | 0.07 |
  | III | 15 | 37.5 | 0.206 | 0.21 |
  | IV | 12 | 16.7 | 0.450 | 0.44 |
  | V | 4 | −11.3 | 0.780 | 0.78 |
  | VI | 4 | −35.9 | 1.077 → clipped to 1.000 | 1.00 |

  FST I is not in the dataset and is extrapolated from an assumed ITA_U ≈ 58°
  (`vitd_model.py:723`), which the mapping sends to −0.035, clipped to 0.
- **Caveat on the stated sample size — [REPOINTED 2026-09-21].** This document
  previously cited the docstring phrase "ArimedB trial data, N=44" at
  `vitd_model.py:284-285`. That exact phrase is **gone**: the docstring was
  rewritten 2026-08-23, independently of this document, to describe N=44 as a
  corrected historical error, and the content that now occupies
  `vitd_model.py:284-285` is unrelated (interval-optimum / feedback-form
  material, not `M_baseline`). The corrected text now lives inside the
  `M_baseline_by_FST` docstring entry itself, `vitd_model.py:704-722`, and
  says essentially what this document found independently: the `ArimedB
  irradiated` sheet does have exactly 44 rows, but only 28 of them carry an
  `ITA_Unexposed` value, and those 28 alone reproduce (docstring's own
  rounding) {II 0.07, III 0.19, IV 0.48, V 0.77, VI 0.96} — which does **not**
  reproduce the shipped (pre-2026-09-06) table. My own recomputation below
  gives slightly different third-decimal figures ({II 0.074, III 0.186, IV
  0.481, V 0.772, VI 0.956}) from that same 28-row subset; the two are
  consistent to the precision either was reported at. The pooled irradiated +
  controls set (n = 48) reproduces the shipped table. The published N was
  attached to the wrong subset; the values themselves are sound and
  reproducible. This was a documentation defect, already fixed in the code's
  own docstring independently of this pass — I have not further edited
  `vitd_model.py`.
- **Manuscript status flag:** PRIOR, with FST V and VI explicitly flagged as
  having "no direct published melanin-index validation"
  (`paper/manuscript.tex:199`). That flag predates the Narea corroboration
  above and should be read alongside it. See §4.4.

### 2.7 `K_const` — constitutive-pigment attenuation coefficient

**[RE-ANCHOR 2026-09-22/23 — READ THIS FIRST. Everything below it,
including the 2026-09-21 block, is superseded on the VALUE and on the
ESTIMATOR, though not on the structure.]** `K_const` now ships at
**[read from code: `ModelParams.K_const`]**, which is roughly 1.8x the
1.17 that shipped on 2026-09-21. The full methodology is §2.7b below; the
derivation is `scripts/kconst_recalibration.py`; the approval is
`calibration_approval.yaml`, `K_const_random_effects_2026_09_22` and
`K_const_A_full_precision_2026_09_23`. **The 2026-09-21 [FLAG] further down
this section — "I could not reproduce this derivation by re-running the
script" — is now RESOLVED, and the resolution is that the flag was right:
the script and the docstring disagreed because two of the four routes were
a stale cache.** See §2.7b, step 1.

**[SUPERSEDED 2026-09-22] [UPDATED 2026-09-21] Superseded 2026-09-06.**
`K_const` shipped as **1.15** [historical], not the 0.98 the rest of this
subsection was written against. This was a **coupled refit**, done in the
same user-approved change as §2.6's `M_baseline_by_FST` re-anchoring: because
the constitutive term is `exp(-K_const * M_baseline(F))`, changing the M-gap
between skin types forces a re-fit of `K_const` to hold the same
between-study response ratios. Do not read 0.98 anywhere below as current.

- **Code:** field default `vitd_model.py:1678`; docstring entry
  `vitd_model.py:168-305`; derivation `scripts/kconst_recalibration.py`;
  diagnosis `scripts/fst_gap_decomposition.py`; approval
  `calibration_approval.yaml`.
- **Publications:** four independent within-study routes, two lab and two
  field:
  - **Felton 2016**, *Br J Dermatol* 175:1320-1328 — phototype II vs V, same
    cabinet, same protocol, both baselines printed
    (`scripts/kconst_recalibration.py:81-95`). Full text in
    `Research_Papers/`.
  - **Young 2020** — FST II vs VI, matched dose, both source arms,
    individual-level data in the repo (the xlsx above).
  - **Kift 2013 / Webb 2010** — field personal polysulphone dosimetry,
    Manchester 53.5°N, with the measured behavioural dose difference removed.
  - **Darling 2013** — field personal dosimetry, Surrey 51.3°N, same treatment.
- **What they measured:** each yields a within-study *ratio* of 25(OH)D
  response between a lighter and a darker arm at matched physical dose (lab) or
  at measured personal dose with behaviour subtracted (field).
- **Arithmetic from measurement to parameter:** the model's constitutive term
  is `exp(-K_const * M_baseline(F))`, so
  `ln(V:II optical factor) = K_const * (M_V − M_II)` is *linear* in `K_const`.
  Equal-weight least squares in log-ratio space therefore reduces to the plain
  arithmetic mean of the four study-level implied `K_const` values.
  **[FALSE, CORRECTED 2026-09-22 — this is the bug §2.7b step 1 describes.
  The sentence that follows was the repository's stated belief and it was
  wrong.]** ~~Kift and Darling are hardcoded literals in the script (field
  inversions computed independently of `M_baseline_by_FST`), so they are
  unaffected by the 2026-09-06 table change~~; Felton and Young are
  recomputed from `M_baseline_by_FST` live and therefore shift. **[quoted from the docstring
  2026-09-21, `vitd_model.py:163-170`]**, the docstring's own worked
  derivation under the current (Narea) table is: Felton 2016 2.097, Young 2020
  0.347 (adjusted@pooled), Kift 2013/Webb 2010 0.788, Darling 2013 1.387;
  mean 1.155 → 1.15; SD across the four routes 0.759; range 0.347-2.097.
  Weighting is one vote per *study*, not per subject or per contrast, an
  explicitly flagged `[CHOICE]` (`scripts/kconst_recalibration.py:605-612`).
  - **[FLAG 2026-09-21] I could not reproduce this derivation by re-running
    the script.** `PYTHONIOENCODING=utf-8 python3 scripts/kconst_recalibration.py`,
    run against the current working tree, gives Felton 2.138 (not 2.097) and
    Young 0.358 (not 0.347) for the two live-computed routes — Kift (0.788)
    and Darling (1.387) reproduced exactly, as expected for hardcoded values.
    That moves the mean to 1.168 and the SD to 0.772 (the script's own
    rounding convention would then suggest K_const = 1.17, not 1.15). I
    checked for an explanation and could not find one: `scripts/kconst_recalibration.py`
    is byte-for-byte unchanged since the commit that produced this derivation
    (`37f9a94`, 2026-09-06); `K_const`, `BSA_EXP` and `M_baseline_by_FST` in
    `vitd_model.py` are unchanged since that same commit (only docstring/comment
    text has moved since, including an edit in progress by another agent at the
    time of this check); and a live `ModelParams()` read confirms
    `K_const=1.15`, `BSA_EXP=0.3281`, and the current (Narea) `M_baseline_by_FST`
    table are exactly what the script sees. The shipped `K_const=1.15` is
    unaffected either way — it is a stored default, not recomputed at
    import — but the docstring's own worked derivation for *why* 1.15 is not
    reproducible from this repository as I read it today. Flagging rather than
    silently picking one number set; whoever owns the docstring should re-run
    and reconcile before the per-study figures are quoted elsewhere.
- **Status in the docstring's own words:** "Treat 1.15 as a central estimate of
  a wide range (SD 0.759, i.e. roughly 0.4-1.9), not a measured constant --
  WIDER than the 0.3-1.5 range this parameter carried at 0.98"
  (`vitd_model.py:184-186`). See §4.5.

### 2.7b `K_const` — the 2026-09-22/23 re-estimation, in full

This subsection is the methodology and the result for the move recorded in
`calibration_approval.yaml` as `K_const_random_effects_2026_09_22` (value and
estimator) and `K_const_A_full_precision_2026_09_23` (precision and the joint
fixed point). Every number below was **recomputed on 2026-09-23**, not copied
from the 2026-09-22 session: the random-effects pooling was re-implemented
independently (REML by direct maximisation of the restricted log-likelihood
rather than by the fixed-point iteration the repo's script uses), Darling's
Table 3 was re-read from the PDF, and the (K_const, A) fixed point was
re-derived from `vitd_validation.young2021_A_anchor` rather than from the
derivation script's own copy of the anchor arithmetic. Where a figure differs
in the last digits from the 2026-09-22 record, the 2026-09-23 figure is the
one printed here, because it is measured at the shipped unrounded pair.

**The shipped value is a pooled estimate with an I² of 83%, a τ² larger than
the estimate's own square, and a leave-one-out envelope twenty times wider
than its estimator uncertainty. It is a central estimate of a wide range, and
§4.5 is the section that says what that means for anything built on it.**

#### 1. The bug that caused most of the move: the field routes were a cache

Two of the four routes — Kift 2013 / Webb 2010 and Darling 2013 — were
carried as literal numbers taken on 2026-08-17 (`HELD_OUT_FIELD` in
`scripts/kconst_recalibration.py`, K 0.788 and 1.387). They were carried
through the 2026-09-06 and 2026-09-21 refits on the stated belief, written
into `vitd_model.py`'s own `K_const` entry and quoted verbatim in §2.7 above,
that the field inversions are "computed independently of
`M_baseline_by_FST`".

That belief is false. `fst_gap_decomposition._invert_cohort_pair` calls
`simulate()`, so each field route's implied `K_const` depends on
`M_baseline_by_FST`, `BSA_EXP`, `SC_max`, the feedback form, the body-weight
term and (weakly) `A`. Re-inverted live on the current tree the two routes
are **2.0009453682405365** (Kift) and **3.1776093028993273** (Darling) — a
2.5x and 2.3x move from the cache. The cache is retained, unedited, and
selectable with `--field-routes carried`, so every previously published
figure stays reproducible; the default re-derives.

This is a **bug fix**, not a calibration: the code was reading a stale
number and reading the live one needs no approval under CLAUDE.md. What it
*moves* was approved separately. It is also the resolution of the 2026-09-21
[FLAG] in §2.7 above: the docstring's worked derivation and a fresh script
run disagreed because the script's live routes had drifted from the
docstring's cached transcription of them.

There is a second, subtler bug in the same file, and it is worth recording as
a failure mode rather than as a number. `invert_one`'s search ceiling was the
literal `K_HI = 3.0` — a bound in `K_const` units, while the quantity that is
actually physical is the **product** `K_const * M_baseline(V)`, above which
the darker arm's modelled net response turns negative and the ratio stops
being monotone. When the §2.6 ITA→M re-scale compressed the M scale, Felton's
required `K_const` crossed that stale bound; `invert_one` then returned the
bound for the point estimate *and* for both ±h perturbations, so the
numerical derivative dK/d(ln r) came out **exactly zero**, Felton's SE came
out exactly zero, and a pool that correctly refuses a zero SE **dropped the
only within-study FST V route without saying so**. The fix is a scale-free
ceiling, `K_TIMES_M_HI = 2.2617` (= 3.0 × 0.7539, the old bound restated on
the old M scale), with `k_hi(params)` dividing by the current
`M_baseline(V)`. Verified 2026-09-23: `k_hi()` is 4.27068820224719, Felton
inverts to 3.1945434824918686 well inside it, and Felton carries 27.49% of
the random-effects weight — it is in the pool, not silently absent.

#### 2. The A3 re-scale: a reparameterisation, not new evidence

The §2.6 clip release re-scales ITA→M from `M = 1 − (ITA_U + 30)/85` to
`M = (55 − ITA_U)/121`. That compresses the FST V:II M-gap from 0.582400 to
0.409091, and since the routes estimate the **optical factor**
`exp(K_const · (M_V − M_II))` rather than `K_const` itself, every route's
implied `K_const` rises by exactly 1/0.7024… = **1.4237x** for an *exactly
unchanged* optical factor. None of that 1.4237x is new evidence about
pigment; it is the same measurement expressed on a shorter ruler. Any
comparison of `K_const` across 2026-09-22 must divide by it first.

#### 3. Random effects, and why not fixed effect

The four routes are not noisy reads of one number. §4.5 has the full
argument; the measurable form of it is that at the shipped point
**Q = 11.9465 on 2 df, I² = 83.26%**. A fixed-effect inverse-variance pool
asserts the routes differ only by sampling error and reports a precision that
this heterogeneity says does not exist. DerSimonian–Laird random effects
estimates the between-route variance τ² instead and carries it into the
interval.

| τ² estimator | τ² | pooled `K_const` | SE | 95% CI |
|---|---|---|---|---|
| DerSimonian–Laird (**shipped**) | 2.911181 | **[read from code: `ModelParams.K_const`]** | 1.097680 | [−0.0553, 4.2475] |
| Paule–Mandel | 1.960891 | 2.017210 | 0.935913 | [0.1829, 3.8516] |
| REML | 2.260534 | 2.048069 | 0.990230 | [0.1073, 3.9889] |

The τ²-estimator **range** on the pooled value is 2.017210 to 2.096071
(width 0.078861) and is carried rather than DL alone, because DL is known to
under-estimate τ² at small k and k is 3 here. A Hartung–Knapp–Sidik–Jonkman
interval is computed — [−1.8938, 6.0859], t(2) multiplier 4.302653 — and is
deliberately **not adopted**: at k = 3 it spans more than the whole physically
admissible range of `K_const` and carries no information. Do not quote it.

Note what the DL interval's lower bound is: **below zero**. The evidence in
this repository does not exclude "no constitutive pigment penalty at all" at
95%, which is a statement about how thin four within-study routes are, not
about photobiology.

Per-route weights under DL: Felton 27.49%, Young 40.83%, Darling 31.68%.
The retained `one_vote_per_study` scheme gives 2.222642 and remains
computable in the same run, so every figure published under it stays
reproducible.

#### 4. Darling 2013 now carries a measured variance — and two flags

Darling could not enter any variance-weighted pool before 2026-09-22, because
the repository's own `VARIANCE_PROVENANCE` register said the paper was "NOT
HELD IN THIS REPOSITORY". It is held: the University of Surrey author's
accepted manuscript,
`Research_Papers/Vitamin_D_deficiency_Osteoporosis_International_2012.pdf`.
Its **Table 3** (page 29 of that PDF) prints mean, SD *and a per-season n* for
every one of the six cells the field inversion consumes. Re-read visually on
2026-09-23: the table is cleanly legible, and all six cells match what the
code carries.

SE(K) = **0.9445251347765502** by delta method — each cell perturbed by
±SD/√n, the inversion re-run, the six partial derivatives summed in
quadrature, cells treated as **independent**. Independence is the wider and
therefore conservative choice: summer and winter are repeated measures on
overlapping women and Darling's own Table 4 puts the between-individual share
of unexplained variance at 268.35/(268.35+296.36), and a positive correlation
would *shrink* the variance of their contrast. The dominant contributions are
the South Asian cells (`sed_summer_week` +0.552145, `d_summer` −0.486099,
`d_winter` +0.449926), not the dense Caucasian ones — which is the sparse
n = 18 summer-dosimetry cell doing most of the work. Recomputed at a
quarter-step the SE is 0.9313042843755703, i.e. the local linearisation
under-states the curvature by about 1.4%; small enough not to change the
pooling, recorded rather than absorbed.

**The n in Darling is internally inconsistent**, and the repository had been
carrying one figure without saying which: abstract 105 Caucasian / 35 South
Asian; Methods and Table 1, 135 / 46 recruited; Study Power section, "39 Asian
premenopausal women were recruited"; four-visit completers 90 / 35 (the pair
Webb 2021 Table 5 labels the route with). **None of those is the right n for
this route's variance.** Table 3 prints its own per-season cell n — white
113/128/80, South Asian 18/43/26 — and those are what the SE is built from.

**DARK-SKIN FLAG (veiling).** Darling's Discussion states that 75% of the
South Asian arm "were of Pakistani origin and were partly veiled (showing
hands and face only)". Darling publishes no exposed-BSA figure at all, so the
inversion equates its cohorts' exposed BSA to Kift 2013's diary percentages;
for a veiled arm that is too high, the badge dose is spread over more skin
than was exposed, and the shortfall is charged to **pigment**. This route's
`K_const` is therefore **biased upward**. Sensitivity, reported beside the
headline rather than folded into it: at a veiling-adjusted exposed BSA of
0.1025 (75% at a rule-of-nines hands-and-face 0.09, the rest at Kift's 0.14)
Darling falls from K 3.1776093028993273 to 2.906856 and its FST V:II from
3.6690x to 3.2843x. The rule-of-nines conversion is **this repository's
geometric assumption, not Darling's** — the paper prints no BSA — so the
variant *bounds* the veiling bias rather than measuring it. Darling stays in
the pool: it is one of only two field routes and the only one outside
Manchester.

**DARLING NEVER USES FITZPATRICK TYPE.** The words "Fitzpatrick",
"phototype" and "skin type" do not appear anywhere in the paper. Its groups
are "Caucasian" and "Asian" (predominantly Pakistani origin). The FST II and
FST V labels this repository attaches to them are **inferred from ethnicity
here**, and that inference is what lets an ethnicity contrast be read as a
pigment contrast. It is load-bearing for one of the three pooled routes.

#### 5. Kift 2013 is excluded, and the exclusion is bounded

Kift publishes a 95% CI on its **raw** diet-adjusted serum ratio (point 45%,
i.e. white:SA = 2.2222, CI 1.8868–2.7027, SE(ln) = 0.091679) — verified from
the PDF, correcting an earlier repository claim that it publishes nothing.
But this route inverts the contrast **after** the measured behavioural dose
difference has been removed, and the behaviour-removal step's uncertainty is
unquantified anywhere in this repository. Using the raw CI as the route's
variance would understate its uncertainty and so over-weight it: a partial
variance used as a full one is worse than none, because it is confidently
wrong. **No variance is imputed** and the route is dropped from the pool.

Dropping it is only safe if its absence is not load-bearing, so the exclusion
is bounded rather than assumed harmless. A max-admissible-weight probe
re-admits Kift at assumed SEs from 8x down to 1x the most precise route's SE:

| assumed SE(K) | pooled `K_const` | movement |
|---|---|---|
| 1.593575 | 2.044342 | −0.051729 |
| 0.796787 | 2.003107 | −0.092964 |
| 0.398394 | 1.947427 | −0.148644 |
| 0.199197 | 1.911152 | −0.184918 |

The largest movement across the whole sweep is **0.184918** in `K_const`,
about 8.8% of the pooled value and well inside the τ²-driven interval. The
probe imputes nothing: its output is a sensitivity figure and is never pooled.

#### 6. The leave-one-out envelope, and the decision to keep carrying it

Decision A4 asked for the envelope to be carried **only if still needed**
once the weighting question was settled. Random effects produces an interval
of its own — the τ²-estimator range, width 0.078861 — so the test applied was
whether the leave-one-out envelope is materially wider than that.

| variant | `K_const` | FST V:II |
|---|---|---|
| all routes (primary, = the shipped pool) | 2.096071 | 2.36x |
| drop Felton 2016 | 1.687313 | 1.99x |
| drop Young 2020 | 3.183999 | 3.68x |
| drop Kift 2013 (already excluded — no change) | 2.096071 | 2.36x |
| drop Darling 2013 | 1.588440 | 1.92x |
| LAB routes only (Felton + Young) | 1.588440 | 1.92x |
| FIELD routes only (= Darling alone, in the RE pool) | 3.177609 | 3.67x |
| Young pairing `raw@own` instead | 2.185 | 2.44x |
| Young pairing `adjusted@own` instead | 2.228 | 2.49x |

**Envelope 1.588440 to 3.183999, width 1.595559 — 20.2326x wider than the
τ²-estimator range.** WHICH routes are in the pool moves the answer twenty
times more than HOW τ² is estimated does, so the envelope is not redundant
and **is carried**. Note also what its two ends *are*: the low end is exactly
the lab pair and the high end is exactly "drop Young", so the envelope is
measuring which of two mutually incompatible route families is believed, not
sampling noise.

It is load-bearing in the only sense that matters — the headline dark-skin
scheduling conclusion flips across it, in both directions. With `A`
re-anchored at each end (the two are coupled; moving `K_const` alone would
mis-state the level), at UV index 6 over a 365-day horizon with a 90-day
ramp and `SC_max` IV–VI at the shipped low end:

| `K_const` | re-anchored `A` | FST V N_min | FST VI N_min |
|---|---|---|---|
| 1.5884400707455726 | 4.100889741363239 | 104 | 183 |
| **shipped** | **shipped** | **146** | **260** |
| 3.1839988650360818 | 4.9702105418146125 | 260 | none up to daily |

At the low end FST V's twice-weekly exclusion is **false**. At the high end
FST VI cannot be scheduled at any session budget up to daily. No FST V/VI
feasibility claim may be stated as a bare fact.

#### 7. The approved-band check

The diagnosis approved on 2026-08-17 fixed the target as an FST V:II
**optical factor** of 1.7–2.7x. At the shipped value the optical factor is
**2.3572212372489663x — inside the band.** The band was not re-opened.
Across the leave-one-out envelope the factor runs 1.9152x to 3.6786x, so the
envelope's high end sits **outside** the band; reported, not absorbed.

#### 8. The joint (K_const, A) fixed point

`A` enters every one of `K_const`'s routes (each calls `simulate()`) and
`K_const` enters `A`'s Young 2021 anchor kernel through `exp(−K_const·M_II)`,
so the pair has a fixed point and neither can be moved alone. Iterating
`A ← young2021_A_anchor(K_const)` and `K_const ← DL pool at that A` from the
2026-09-22 pair converges in 6 iterations to |ΔK| < 1e−10. The coupling is
weak: `A` moves 0.05% between iterations 2 and 6.

**[PRECISION 2026-09-23]** The 2026-09-22 pair rounded `K_const` to two
decimals *and then anchored `A` against the rounded value*, so neither half
was the fixed point it was described as. Both now ship unrounded at the fixed
point [read from code: `ModelParams.K_const`, `ModelParams.A`], where
`young2021_A_anchor(ModelParams())["A_implied"]` returns the shipped `A`
exactly rather than within a 5e−4 tolerance. This is not a calibration: no
evidence, estimator or anchor changed, only the digits that had been
discarded.

#### 9. Where the 1.17 → ~2.10 move actually came from

Three causes compound in one number and must not be read as one
recalibration:

| cause | class | effect |
|---|---|---|
| field routes re-inverted live instead of read from a 2026-08-17 cache | **bug fix** | the two field routes move 0.788 → 2.0009 and 1.387 → 3.1776 |
| ITA→M re-scale (§2.6, decision A3) | **reparameterisation** | ×1.4237 on `K_const` for an *unchanged* optical factor |
| pooling one-vote → fixed-effect IV → random-effects DL | **estimator choice** | one-vote would give 2.222642; DL gives the shipped value |

The cleanest way to see that the A3 re-scale is not evidence: the shipped
optical factor, 2.3572212372489663x, sits inside the same 1.7–2.7x band the
2026-08-17 diagnosis approved, and that band is stated in optical units
precisely because they are the scale-free ones.

#### 10. What did *not* change, and what this does not license

- The **calibration set** is unchanged: within-study, matched-physical-dose,
  matched-BSA, same-source trials only.
- **Behaviour** stays out of `K_const`. The behaviour-blind field inversions
  need FST V:II of 4.28x and 5.71x, above every matched-dose measurement in
  existence; shade, clothing and time outdoors live in
  `exploratory/vitd_behaviour.py`, neutral by default.
- **Dark-skin provenance is unchanged and still thin.** Every FST V datapoint
  behind this number is Indian-subcontinent ancestry at UK latitude; every
  FST VI datapoint is lamp-based; no African-ancestry field dosimetry exists
  anywhere in this repository. Every VI:* ratio remains an extrapolation of
  one coefficient onto an `M_baseline(VI)` that no measurement of response
  constrains.
- **Fajuyigbe et al. 2018's 8.4x** (recomputed from the paper's own printed
  slopes: 7.59x — both recorded, the discrepancy flagged not resolved) is an
  **erythema** ratio and does **not** enter `K_const`. The repository already
  holds the cautionary case: Kallioglu 2024's "V:II = 2.40" was withdrawn
  from the held-out scoring set precisely because it turned out to be an
  erythema MED table, and Felton 2016's own measured erythema MED ratio
  (3.95x) sits far above every matched-dose vitamin D estimate. An erythema
  ratio scores a vitamin D calibration at best; it never sets one.

#### 11. Reproducing this section

```bash
PYTHONIOENCODING=utf-8 python3 scripts/kconst_recalibration.py
```
Sections S3b (the four routes), S3c (variance-availability audit, Darling's
delta-method SE, the veiling variant), S3d (all weighting schemes side by
side), S3e (τ², I², the intervals, the Kift probe) and S4 (the leave-one-out
envelope and the A4 test) print everything above. The joint fixed point is
one line:

```bash
python3 -c "from vitd_model import ModelParams; \
from vitd_validation import young2021_A_anchor; p = ModelParams(); \
print(repr(p.K_const), repr(p.A), repr(young2021_A_anchor(p)['A_implied']))"
```

### 2.7a `SC_max_by_FST` — stratum corneum thickening ceiling (I-III cell)

This parameter is not part of the melanin state `M`, but its 2026-08-25
change shares the same evidence package (Miyamura 2011) as §2.5, so it is
recorded here rather than left undocumented.

- **Code:** `vitd_model.py:1828` (`ModelParams.SC_max_by_FST`), consumed at
  `vitd_model.py:2524`. **Changed 2026-08-25: FST I-III 0.23 -> 0.24**
  (`calibration_approval.yaml`, entry "SC_max_by_FST I-III 0.23 -> 0.24
  (stratum corneum ceiling)", APPROVED).
  **[SUPERSEDED 2026-09-22]** ~~FST IV-VI is unchanged at 0.36.~~ The
  FST IV-VI cell moved to the corroborated FST I-III value and is now carried
  as an explicit RANGE, `vitd_model.SC_MAX_IV_VI_RANGE`, whose low end is the
  shipped point value [read from code: `ModelParams.SC_max_by_FST`,
  `vitd_model.SC_MAX_IV_VI_RANGE`]. It was the only cell in the melanin/SC
  cluster whose sole source is a study that did not enrol the skin types the
  cell is applied to, and the error is one-sided — no source in this
  repository argues for a HIGHER value — so the old cell sat at the
  most-burdensome extreme of its own evidence while two headline dark-skin
  exclusions rested on it. See `calibration_approval.yaml`,
  `SC_max_by_FST_IV_VI_2026_09_22`, which also records that under the settled
  `K_const` this cell no longer flips either FST V's or FST VI's minimum
  session budget at either end of the range — the `K_const` move swamped it.
  The scalar `SC_max` default (0.30) is unchanged.
- **Publications, now two independent routes:**
  - de Boer 2025 Supplementary Table S1: FP I-III stratum corneum 5.6 -> 6.9
    um, i.e. **+23.2143%** ((6.9-5.6)/5.6). This was the sole basis for
    `SC_max_by_FST` until now.
  - Miyamura 2011 (PMID 20979596) Table 1: stratum corneum 16.6 -> 20.7 um in
    FST II-III after 10 suberythemal UVB exposures over 2 weeks, i.e.
    **+24.6988%**. Independent cohort, protocol, and measurement method
    (histology of formalin-fixed biopsies vs. de Boer's technique).
- **Arithmetic:** two independent studies now bracket the FST I-III
  stratum-corneum-thickening ceiling at +23.2143% and +24.6988%. 0.24 sits
  between them (midpoint of the two unrounded measurements, 0.239565); 0.24
  was chosen by the repository owner as the round value inside the bracket.
- **What this is — first out-of-sample support `SC_max` has ever had.** Before
  2026-08-25, `SC_max_by_FST` I-III was fitted to de Boer 2025 alone, so
  agreement with de Boer was circular (see `basics/README.md`, "the endpoint
  is reproduced — and that is not evidence"). Miyamura 2011 is a *different*
  study, cohort, and measurement technique landing within 1.7 percentage
  points of de Boer's figure. That corroborates the **magnitude** of the
  ceiling.
- **What it does NOT establish.** Miyamura reports SEM with n=6 and did not
  measure vitamin D — the corroboration is of a stratum-corneum-thickness
  endpoint, not of any D-synthesis prediction. Per `CLAUDE.md`, `SC_max` must
  still be flagged as an **uncalibrated prior** whenever absolute D
  predictions are presented. Miyamura's own Table 1 flags the SC cell NS
  ("20.7 +/- 3.6 NS"; text: SC thickness "did not change significantly in
  any of the specimens", "increasing trend") — NS is a power statement, not
  evidence of absence; corroborates trend and magnitude, not an independently
  significant replication. It does not extend to FST IV-VI: Miyamura
  enrolled FST II-III only, so the 0.36 cell still rests on de Boer alone —
  per the dark-skin-priority directive (`CLAUDE.md`), this is the gap that
  matters most and it remains open. It does not touch `alpha_SC`,
  `k_decay_SC`, `K_SC`, or `k_damage`.
- **Manuscript status flag:** unchanged — still an uncalibrated prior, now
  with a second corroborating (not calibrating) source for FST I-III only.

### 2.8 `K_ATTEN` — facultative (tanning) attenuation coefficient

- **Code:** `K_ATTEN_DEFAULT` at `vitd_model.py:1561`, consumed by
  `ModelParams.K_ATTEN` at `vitd_model.py:1727`; docstring entry
  `vitd_model.py:526-548`; derivation `vitd_calibration.py:28-145`.
- **Publication:** Matsuoka, Wortsman & Hollis 1990, *J Lab Clin Med*
  116:87-90, PMID 2165512. **Abstract only** — held in the repo as
  `Papers_and_Datasets/Matsuoka 1990 JLabClinMed PMID2165512 - ABSTRACT ONLY
  (has the tanning data).txt`. (This file was formerly named "(NO PAPER
  FOUND)" and the paper was wrongly listed as missing; every number needed is
  in the abstract — `vitd_calibration.py:45-49`.)
- **What it measured:** n = 7 healthy subjects, **within-subject**. Whole-body
  UVB in a phototherapy unit, identical dose, once in midwinter (untanned) and
  once in midsummer (tanned); blood immediately before and 24 h after. Serum
  vitamin D3 (the parent compound, not 25(OH)D) rose 1.7 → 11.0 ng/mL untanned
  and 9.6 → 14.3 ng/mL tanned; baseline 25(OH)D was 22.5 vs 36.9 ng/mL.
- **Arithmetic from measurement to parameter** (`vitd_calibration.py:108-144`),
  four steps:
  1. Observed response ratio = tanned rise / untanned rise = 4.7 / 9.3 = 0.505.
     One minus that is the widely quoted "51% attenuation".
  2. The two arms had different baseline 25(OH)D (56.2 vs 92.1 nmol/L after
     the ng/mL → nmol/L conversion at `vitd_calibration.py:79`), so part of the
     blunting is the model's own feedback factor `g(D)`. Compute
     `feedback_ratio = g(92.1) / g(56.2)` under a chosen feedback form.
  3. `melanin_ratio = observed_ratio / feedback_ratio` — the residual the
     feedback form cannot explain.
  4. `K_ATTEN = −ln(melanin_ratio) / excess_M`, with `excess_M` assumed equal
     to `M_TAN_HEADROOM`, i.e. a fully developed tan.
- **Result:** the answer is *feedback-form-dependent by construction*, which is
  why the module reports the whole set rather than one number
  (`vitd_calibration.py:160-198`). Under the model's default biphasic feedback
  the residual melanin attenuation is 6.6%; under the legacy linear form it is
  18.1%; with feedback switched off entirely the naive 49.5% is recovered. The
  adopted value is the biphasic one, i.e. the **lower bound** of the range the
  study can support (`vitd_model.py:537-538`). Run the companion script,
  section 6, for the live table.
- **FIXED 2026-08-25.** `vitd_calibration.derive_K_ATTEN`'s `excess_M`
  parameter now defaults to `params.M_TAN_HEADROOM` read live instead of a
  hardcoded `0.20`, so it tracks the current headroom (0.238) automatically.
  Recomputing at 0.238 moves the `K_ATTEN` figures below (§4.3) from
  [0.34, 0.92, 1.00] to [0.288972, 0.773884, 0.838389] — the melanin-ratio
  *percentages* (6.6%/16.8%/18.1%) are unaffected since they don't depend on
  `excess_M`, but every `K_ATTEN` point estimate does. `K_ATTEN_DEFAULT`
  itself moved from 0.34 to 0.288972 in the same pass. The
  "0.34-1.00" figure that was previously quoted in `run_all.py`,
  `vitd_sensitivity.py`, `scripts/parameter_optima.py` and the
  `vitd_model.py` docstring has been updated in all of those places to
  "0.288972-0.838389". (A sixth site, a reconstructed evidence document,
  was purged on 2026-09-19 without ever having been committed.)
- **Manuscript status flag:** SINGLE-SOURCE PRIOR (`paper/manuscript.tex:187`).
  See §4.3.

### 2.9 Cross-references that are stale

Two places in the repository still describe an older state of this parameter,
and a reader following either of them will be misled:

- `paper/manuscript.tex:187` prints a different value from the code.
  `paper/manuscript.tex:153` also states the linear feedback term is the
  model's default, which `vitd_model.py:1473` contradicts.
- `spacing_study/results/FINDINGS.md:35-41,370-372` was generated while
  `K_ATTEN` was still pinned at zero and says so (a hand-added staleness
  banner at the top of that file flags this as of 2026-08-23).

[Update 2026-08-24, repointed 2026-09-23: the third item previously listed
here -- `vitd_model.py`'s own "Gap 2" block allegedly still saying `K_ATTEN`
is "CLOSED BY RETIREMENT" -- is no longer accurate. The `K_ATTEN` docstring
entry (`vitd_model.py:681-686`) now reads "CALIBRATED 2026-08-21
(user-approved) ... Reinstated as an active parameter", and
`RETIRED_PARAMETERS` (`vitd_model.py:1564`) is an empty dict, consistent
with the live `K_ATTEN` entry above it. The exact wording quoted here in
2026-08-24 has since been superseded by further docstring edits; only the
substance (reinstated and calibrated, not retired) is checked, not a
literal string match. `vitd_model.py` is internally consistent on this
point; only the two files above still lag it.]

I have not edited `paper/manuscript.tex` or `spacing_study/results/FINDINGS.md`;
other agents are working in them.

---

## 3. Where melanin enters synthesis — two separate exponential terms

The synthesis function is `synthesis_per_session()`
(`vitd_model.py:1817-1891`). Pigment enters through **two distinct exponential
factors with separately derived coefficients**, computed at
`vitd_model.py:1846-1848` and multiplied into the return value at
`vitd_model.py:1890`:

```python
excess_M = max(0.0, M - M_baseline)                    # :857
constitutive_factor = np.exp(-Kc_use * M_baseline)     # :858
facultative_factor  = np.exp(-K_use  * excess_M)       # :859
```

so that

```
S = A * ln(1 + sed * E_source)
      * bsa_factor
      * exp(-K_const * M_baseline(F))        <- constitutive pigment
      * exp(-K_ATTEN * (M - M_baseline(F)))  <- facultative pigment (tan)
      * exp(-K_SC * SC)
      * g(D) * photodegradation * weight
```

Three points about this structure that matter for interpretation:

1. **The split is not cosmetic.** The two coefficients come from different
   study designs measuring different quantities. `K_const` is a
   *between-subject* quantity from matched-dose contrasts across skin types
   (§2.7). `K_ATTEN` is a *within-subject* quantity from one seasonal
   tanned-vs-untanned crossover (§2.8). `vitd_calibration.py:204-205` makes
   this explicit: "The two were never the same number."

2. **They are jointly unidentifiable from the current data.**
   `RESULTS.md:2729` reports a `K_const ~ K_ATTEN` pairwise ridge of 0.9998 —
   over 99.9% of the steady-state response surface is explained by a single
   linear combination of the two. `K_ATTEN ~ K_SC` is 0.9990. Reporting either
   member of such a pair as a separately calibrated value overstates what the
   data support (`RESULTS.md:2831-2836`).

3. **The magnitudes are very unequal.** The companion script (section 4)
   prints both: the constitutive factor spans the full FST range while a fully
   developed tan costs a single-digit percentage of synthesis. The
   manuscript's own summary is that constitutive pigment and the stratum
   corneum barrier "account for over 94% of the total skin-type synthesis gap"
   (`paper/manuscript.tex:988`).

Note also that `M` enters synthesis at its **pre-pulse** value
(`vitd_model.py:2105`, before the update at `:2129`), and that `excess_M` is
floored at zero, so a caller who initialises `M0` below `M_baseline` gets no
facultative *bonus* — the term is one-sided by construction
(`vitd_model.py:1846`).

---

## 4. Limitations (mandatory section)

Each item below was checked against the sources; where I could not verify a
claim, that is stated.

### 4.1 The induction rate is reconstructed and light-skin-anchored — VERIFIED

The provenance record for this gap was itself marked
**Confidence: RECONSTRUCTED** — rebuilt from secondary citation sites rather
than read from a primary record — and that record has since been purged
[reference purged 2026-09-19]. Its substantive claims, quoted here while it
still existed, are the only surviving copy:

- `alpha` "was originally calibrated from Miyamura 2011, a light-skin
  photoadaptation kinetics study" (`:96-98`), corroborated at
  `paper/README.md:183`, `scripts/parameter_optima.py:99` ("light-skin
  derived"), and `vitd_analyses.py:83-84`.
- "A dedicated re-fit from Young 2020 individual-level trial data was attempted
  and found alpha structurally unidentifiable from that dataset" (`:98-101`).
  The manuscript quantifies this as "92.5% of fits hit search bounds"
  (`paper/manuscript.tex:191`).
- **Status: open.** `alpha` remains light-skin-anchored.

**What I could not verify:** the "92.5% of fits hit search bounds" figure has
no reproducing script in the repository. The only Young-2020 fitting script,
`scripts/fit_katten_young2020.py`, fits `K_ATTEN` (not `alpha`) on the n=23
TL01 arm. The alpha re-fit is reported but not reproducible from this tree, so
the 92.5% figure should be cited as a recorded historical result, not as
something a reader can re-run.

The 2026-08-21 scope audit listed `alpha`, `beta` and `K_ATTEN` among
parameters that "barely participate in the model at all", which lowers the
practical stakes of the gap without closing it. That audit has since been
purged [reference purged 2026-09-19]; §5 below reproduces the point from the
live sweeps, which is the reason to believe it.

### 4.2 The decay rate is a choice, not a fitted half-life — VERIFIED

`RESULTS.md:559` states it directly: "The 56-day half-life (default beta) is a
**mid-range estimate** compatible with Ravnbak 2008 if complete fading requires
~3 e-folding times." Ravnbak 2008 reports *time to total fading* (4-5 months),
sampled at three timepoints only, and reports no rate constant — I confirmed
this by reading the paper's Results and Discussion directly
(`Research_Papers/Skin pigmentation kinetics after UVB exposure.pdf`; the
Discussion says "Fading was investigated by measuring only 3 times… This is
obviously a very approximate scale"). There is no exponential fit in the paper
to inherit.

The implied fractions are in §2.4. A 56-day half-life leaves 15-22% of the tan
present at Ravnbak's 4-5 month mark. That is *compatible* if "total fading"
means "faded to the instrument's detection floor for a small tan", and *too
slow* if it means near-complete decay. The evidence does not distinguish these.

**Why Tadokoro 2005 must not be cited for this.** `CLAUDE.md:111` forbids it;
`RESULTS.md:559` and `paper/manuscript.tex:110,197` give the reason: Tadokoro
2005 measured pigmentation only over the **0-7 day** window after exposure and
reports no decay timescale at all. A decay half-life of tens of days cannot be
estimated from a seven-day observation window — over that window a 56-day
half-life and a 30-day half-life differ by a few percent of the signal, well
inside measurement noise, so the data are consistent with essentially any
timescale in the plausible range. Citing it would present an unidentifiable
quantity as a measured one. The citation was corrected to Ravnbak
(`paper/manuscript.tex:110`).

**Dissent from the manuscript's flag.** `paper/manuscript.tex:197` marks `beta`
**VERIFIED**. On the definition used elsewhere in that table — a value read or
computed directly from a published measurement — a mid-range estimate chosen to
be *compatible with* a qualitative fade time is closer to PRIOR. This is a
disagreement about a status label, not about a number, and I have not edited
the manuscript.

### 4.3 `K_ATTEN` rests on one n=7 study, and its famous number is not a melanin effect — VERIFIED

Design facts confirmed against the abstract text held in the repo: n = 7,
within-subject, group means only, no individual data, no measured melanin
index, and the measured outcome is serum **vitamin D3**, not 25(OH)D
(`vitd_calibration.py:70-77`). Neville 2020 is cited as confirming the field
still lacks any replication (`paper/manuscript.tex:1007`,
`vitd_model.py:976-978`).

**Why the "51%" is not a melanin effect.** `CLAUDE.md:112-116` and
`paper/manuscript.tex:1007` both warn against it, and the numbers support the
warning: the tanned arm also started with 64% higher baseline 25(OH)D (92.1 vs
56.2 nmol/L, a ratio of 1.64 — asserted in
`tests/test_matsuoka1990.py:33`). Matsuoka's own conclusion sentence calls the
finding a "resetting of the vitamin D3 synthetic mechanism", which is language
about feedback, not pigment. Since synthesis in this model — and in the
underlying physiology — is baseline-dependent, an unknown share of the 49.5%
blunting is feedback that would have occurred with no tan at all.

**The deconvolution and the resulting range.** Steps in §2.8. Because the
abstract constrains only the *product* (feedback × melanin), splitting it
requires committing to a feedback form, and the answer moves with that choice:
melanin-only attenuation of 6.6% (biphasic), 16.8% (saturating), 18.1%
(linear), implying `K_ATTEN` of 0.288972, 0.773884 and 0.838389 respectively —
a range of roughly 3× (companion script section 6; `vitd_calibration.py:160-196`;
`tests/test_matsuoka1990.py:50-56` asserts the forms must disagree by more than
2×). The shipped value is the **lower bound** of that range. The plausible
range to quote is therefore **0.288972-0.838389**, and the honest framing is
that Matsuoka 1990 bounds this parameter to within a factor of three, not that
it measures it.

**These `K_ATTEN` figures (0.288972, 0.773884, 0.838389) are computed at
`excess_M = M_TAN_HEADROOM = 0.238`, read live by `derive_K_ATTEN` as of
2026-08-25 (previously hardcoded at the pre-2026-08-25 headroom of 0.20, which
gave 0.34, 0.92 and 1.00 — those older figures are superseded).** The
melanin-ratio percentages above (6.6%, 16.8%, 18.1%) are unaffected by this
change — they don't depend on `excess_M`.

Residual assumptions that are not in the data: that a "midsummer tan" in fair
skin equals `excess_M = M_TAN_HEADROOM`; that serum D3 response is proportional
to 25(OH)D response; and that no seasonal confounder (diet, other sun exposure,
weight) contributes (`vitd_calibration.py:70-77`).

### 4.4 The darkest-skin baselines are group-level priors — VERIFIED, and the picture changed materially on 2026-09-06

**[UPDATED 2026-09-21].** The claim quoted below — "FST V (0.78) and VI (1.00)
are group-level priors with NO direct melanin-index validation from published
data" — described the pre-2026-09-06 (Young-2020-anchored) table. It is
**partly superseded**. The shipped values are now different (V 0.7539, VI 1.00
from raw 1.0427; §2.6) and, per the docstring's own updated warning
(`vitd_model.py:584-597`, `[UPDATED 2026-09-06]`), they are "NO LONGER without
direct melanin-index validation": Narea-Jimenez et al. 2026 supplies separate
constitutive ITA means at FST V (n=84) and FST VI (n=32) that agree with the
shipped table within 0.026 (V) and 0.043 (VI, pre-clip). `paper/manuscript.tex:199`
and the original docstring Gap 1 block have not been updated to reflect this
and should be read as describing the superseded state (`vitd_model.py:975-981`
now explicitly flags Gap 1 as "PARTLY FALSE" for exactly this reason).

What my own recomputation (§2.6) adds, now split into what improved and what
did not:

- **Sample size improved sharply.** The FST V and VI cells no longer rest on
  n = 4 each. Narea's constitutive site gives n = 84 (FST V) and n = 32 (FST VI)
  — a 21× and 8× increase. That is real, independently-collected corroboration
  from a second cohort, instrument, and population, not a re-analysis of the
  same n=4 data.
- **But a new, arguably more serious problem surfaced precisely because the
  larger dataset makes it visible.** The measurement is still **ITA from
  colorimetry, not a melanin index** (Narea reports ITA and melanosome
  fraction; neither is a Mexameter MI), and the FST VI cell is still
  **clipped** at M=1.0 (raw 1.0427). The docstring's [EVIDENCE NOTE 2026-09-06]
  point (3) (`vitd_model.py:659-676`) establishes that the clip is **not a
  rare edge case**: FST VI's group *mean* clips independently in both cohorts
  (Young 2020 raw 1.077, Narea raw 1.0427), and on Narea's own mean and SD an
  estimated **59.02% of FST VI individuals** fall beyond the clip (z = 0.2282),
  versus 1.80% at FST V. Within FST VI the mapping therefore has "essentially
  no resolving power" between individuals — more n will not fix a mapping-shape
  problem. The same note also finds the FST label itself is **population-anchored**:
  applying this module's mapping to a South African farmworker cohort (Linde
  2020) shows the UK Young-2020 cohort reads 7-10° lighter at FST II-IV than
  the Venezuelan cohort at the same nominal phototype, while V and VI agree —
  i.e. the label transfers least well exactly where most of this repo's FST
  II-IV data live, and (coincidentally or not) transfers best at the two
  cells this project prioritises.

This compounds through `K_const`, which multiplies these values: any change to
`M_baseline_by_FST` rescales the entire skin-type penalty, which is exactly
why `K_const` was refitted in the same 2026-09-06 change (§2.7). Every FST VI
conclusion in the model remains an extrapolation of one coefficient onto a
clipped boundary value — that structural fact is unchanged by the larger
dataset, even though the dataset independently corroborates the boundary's
*magnitude*.

**Net read for the dark-skin-priority directive.** Treat this as a real but
partial improvement: the FST V/VI baselines are better corroborated in
magnitude than they were, from a second, larger, independently-collected
source, but they remain (a) ITA/melanosome-fraction rather than a validated
melanin index, (b) a saturated clip rather than a fitted mean for FST VI
specifically, and (c) unvalidated against any African-ancestry outdoor cohort.
Flag light-skin (here, non-African-ancestry) extrapolation at the point of use
as before; the evidence base is wider, not closed.

### 4.5 `K_const` is the centre of four mutually inconsistent routes — VERIFIED

**[UPDATED 2026-09-21] Superseded 2026-09-06.** This section was written
against `K_const = 0.98` and the pre-2026-09-06 `M_baseline_by_FST` table.
`K_const` now ships as **1.15** (§2.7). Per the docstring's own re-derivation
under the current table **[quoted from the docstring 2026-09-21,
`vitd_model.py:163-170`]**, the four implied values now span 0.347 to 2.097 —
roughly a **6.04× spread** (2.097/0.347) — with a sample SD of 0.759 against a
mean of 1.155 → 1.15. The docstring's own instruction is now to "treat 1.15 as
a central estimate of a wide range (SD 0.759, i.e. roughly 0.4-1.9), not a
measured constant -- WIDER than the 0.3-1.5 range this parameter carried at
0.98" (`vitd_model.py:184-186`). **See the [FLAG 2026-09-21] note in §2.7**: I
could not reproduce the docstring's 2.097/0.347/1.155/0.759 figures by
re-running `scripts/kconst_recalibration.py` today — a live run gives
2.138/0.358/1.168/0.772 instead — so treat the exact per-study numbers in this
paragraph as the docstring's claim, not as independently re-verified by this
pass. Either way the qualitative conclusion below (wide, mutually inconsistent
routes) holds, and the range is "roughly 0.4-1.9" under both figure sets.

The disagreement is not noise, and the derivation script says so without a
model in the loop: "Young reports FST II:VI = 1.29-1.43 while Felton reports
FST II:V = 2.14 in the same kind of cabinet, and under ANY monotone pigment
penalty II:VI must EXCEED II:V"
(`scripts/kconst_recalibration.py:582-584`). The two lab studies are therefore
*mutually incompatible within this model structure*, and 1.15 "is their and the
field's centre, not a reconciliation -- same status as before, at a different
value" (`vitd_model.py:1054-1055`). Averaging them hides a structural conflict
behind a mean.

Provenance narrowness compounds this: every FST V datapoint behind the number
is Indian-subcontinent ancestry at UK latitude, every FST VI datapoint is
lamp-based, and there is no field dosimetry in FST VI anywhere in this repo
(`scripts/kconst_recalibration.py:40-48`, `vitd_model.py:197-200`).

**A live datapoint on that inconsistency.** `vitd_calibration.young2020_check`
(`vitd_calibration.py:229-329`) compares the model's FST II:VI synthesis ratio
against Young 2020's own 1.29-1.50 band. Running it on 2026-08-23 returns
**2.92** — roughly double the top of the band, i.e. the model now over-predicts
the melanin gap on the one within-study contrast it can be checked against
directly. [CORRECTED 2026-09-20: this paragraph previously said the model
*under-predicted* the Webb/Kift ratio "by about half", making the two
discrepancies opposite-signed. That is no longer true and the opposite-sign
framing below should be read with this correction in mind -- the held-out
Webb/Kift benchmark now returns 2.87x against a reported 2.5-3.0x and PASSES.
The same sentence also called Webb/Kift "naturalistic"; the module's own
2026-08-17 correction establishes that its 2.5-3.0x band derives from Farrar
2013, a matched-dose simulated-sunlight trial. The committed artefact
`tables/debug/young2020_check.csv` has likewise been reconciled and now records
2.9071, not the 1.4868 this paragraph described.] The check is self-labelled UNVERIFIED / "calibration check
only — not independent validation", so it is not a failing benchmark; but the
direction of the discrepancy is opposite to the naturalistic one, which is
exactly the Felton-vs-Young structural conflict above showing up as an output.
I did not rewrite the committed CSV.

**[RE-VERIFIED 2026-09-21]** `tables/debug/young2020_check.csv` has moved
again since the 2026-09-20 correction above: the currently committed row
(`git_hash 6a1585f`, `timestamp 2026-09-21T14:20:28Z`) records **predicted
FST_II_VI_ratio = 2.9288** against target 1.29-1.50, `source_status PARTIAL`,
`result UNVERIFIED`. This is close to but not identical to the 2.9071 quoted
above — consistent with a re-run rather than a hand edit — and does not change
the qualitative reading: still roughly double the top of Young's own band. I
read this CSV; I did not regenerate it.

Finally, `K_const` excludes behaviour by design — shade, clothing and time
outdoors are a separate 2.1-2.4× factor that deliberately does not live here
(`vitd_model.py:193-196`) — so `K_const` is not the real-world skin-type
penalty and should never be quoted as one.

**[CORRECTED 2026-09-21] The sentence that closed this section — "the model
still fails its one held-out benchmark (Webb/Kift 2018's 2.5-3.0× FST V dose
penalty), under-predicting it by roughly half" — was itself stale and
self-contradictory: it restated exactly the claim the `[CORRECTED 2026-09-20]`
note earlier in this section had already retracted.** Per `RESULTS.md` §15
("Held-out external benchmark (Webb/Kift 2018)") and `vitd_benchmark_webbkift.py`,
the current, correct statement is: **the model PASSES its one held-out
benchmark — FST V:II dose ratio 2.87× against a reported 2.5-3.0× band**
(`RESULTS.md:2638-2650`). The citation to `paper/manuscript.tex:65` no longer
supports this passage either way — that line now contains unrelated abstract
text about front-loading budgets, not a Webb/Kift figure; I have not
investigated `paper/manuscript.tex` further since it is owned by another agent
(§2.9). The correct, current citation for the Webb/Kift result is
`RESULTS.md:2638-2650` and `vitd_benchmark_webbkift.py`'s module docstring,
both quoted above.

### 4.6 The induction ceiling still has no derivation; the headroom now does

`alpha_max` is not derived from any measurement (§2.2), and `UV_halfmax_by_FST`'s
claimed derivation from Ravnbak Table IV could not be reproduced arithmetically
(§2.3). **Updated 2026-08-25:** `M_TAN_HEADROOM` is no longer in this category
— §2.5 records that it is now bounded below by the largest tan Ravnbak 2008
actually measured (dM = 0.237724), via `scripts/fit_melanin_kinetics.py`. It
is still not a least-squares fit (the repository owner chose the *maximum*
observed value, not a fitted point estimate), so it stays PRIOR, but it is no
longer a bare structural choice. A reader should understand that two of the
five induction-side parameters (`alpha_max`, and `UV_halfmax_by_FST`'s exact
mapping) remain structural choices; `M_TAN_HEADROOM` no longer is.

### 4.7 Source-availability flags

Per `CLAUDE.md:47-51` and `:98-106`:

- **Miyamura 2011 — resolved 2026-08-25, was a false negative.** An earlier
  revision of this document recorded it as genuinely absent after a
  filename-only search of `Research_Papers/`. It was obtained within minutes
  once searched by metadata instead, and is now on disk as
  `Research_Papers/The deceptive nature of UVA tanning versus the modest
  protective effects of UVB tanning on human skin.pdf`. `paper/references.bib`
  no longer carries a "TODO verify" stub (see its `miyamura2011` entry). It
  is now used by `scripts/fit_melanin_kinetics.py` as the out-of-sample check
  for `M_TAN_HEADROOM` (§2.5) and, jointly with de Boer 2025, as corroborating
  evidence for `SC_max_by_FST` I-III (§2.7a). `alpha`'s own provenance is
  unchanged by this: the same script finds `alpha` not identifiable from
  Miyamura pooled with Ravnbak 2008/2009 (§2.1).
- **Matsuoka 1990 is abstract-only.** The full text is not in the repo. The
  abstract contains every number the derivation uses, but the individual-level
  data that would let the feedback/melanin split be estimated rather than
  assumed is not available at all.
- **Ravnbak 2008's PDF is machine-readable and I read Table IV from the text
  layer directly.** One legibility caveat: the extracted Table IV is
  column-scrambled (the 12-exposure column for the solar simulator appears
  after the FST V row rather than aligned with it), so I reconstructed the
  row/column assignment from the paper's prose, which independently states the
  reduction factors (≈2 at 6 exposures, ≈3 at 12) that the reconstruction
  reproduces. I am confident in the table as transcribed in §2.3, but a reader
  checking it against the original PDF should verify the column alignment
  rather than trust my transcription blind.

---

## 5. Scheduling leverage: what melanin dynamics actually do to the optimal interval

This is the section most likely to matter to a reader deciding how much weight
to give the melanin sub-model.

### 5.1 What the current generated results show — VERIFIED against `RESULTS.md`

**Melanin decay rate (`beta`), `RESULTS.md:557-585`.** The sweep covers
half-lives of 21, 30, 56, 80 and 120 days. At FST V the optimal interval is
1 day at every one of them, and steady-state 25(OH)D moves by at most 0.1
nmol/L across the whole range. At FST VI the optimum is 1 day at every value
and steady-state 25(OH)D does not move at all to the reported precision.

**Melanin induction rate (`alpha`), `RESULTS.md:525-555`.** Swept over
0.010-0.400, a 40× range and far wider than any defensible prior. The optimal
interval is 1 day at every value, for both FST V and FST VI. Steady-state
25(OH)D falls by 1.0 nmol/L at FST V and 0.9 at FST VI across that entire
range — i.e. the effect has the physiologically expected sign, and is tiny.

**Facultative attenuation (`K_ATTEN`), `RESULTS.md` §4 ("K_ATTEN sensitivity
sweep").** Swept 0.2 to 8.0 — up to **27.7×** the calibrated value
(0.288972) and **9.54×** the top of the Matsuoka-implied range (0.838389).
(The calibrated and range endpoints are quoted in full because they are
load-bearing constants other calculations read back; the multipliers are
derived ratios and are quoted to three significant figures.) The optimal interval is 1 day at every value. Steady-state
25(OH)D falls by 2.4 nmol/L at FST V (51.1 -> 48.7) and 2.0 nmol/L at FST VI
(46.6 -> 44.6) across that 40× sweep.

> **[Re-derived 2026-08-25]** `K_ATTEN` was recalibrated from 0.34 to
> 0.288972 (tracking the `M_TAN_HEADROOM` 0.20 -> 0.238 and FST I-III
> `SC_max` 0.23 -> 0.24 recalibration), and the Matsuoka-implied range across
> feedback forms is now [0.288972, 0.838389]. The multipliers and deltas
> above were recomputed from the current `RESULTS.md` K_ATTEN sweep section
> and the current calibrated/Matsuoka-range values; the sweep grid itself
> (0.2-8.0) is unchanged.

**One-at-a-time sensitivity, `RESULTS.md:2663-2677`.** Over documented prior
ranges, at FST V, 1 SED every 3 days, 84 days: `A` swings steady-state D by
8.61 nmol/L (24.8%) and `K_const` by 8.19 (23.6%), while `K_ATTEN` swings it by
0.09 (0.3%) and `beta` by 0.01 (0.0%). Melanin *dynamics* sit at the bottom of
that table; constitutive pigment sits near the top.

**Independent confirmation.** `tests/test_matsuoka1990.py:82-100` asserts as a
regression test that moving `K_ATTEN` across the whole Matsuoka-implied range
shifts steady-state 25(OH)D by less than 2%. The dedicated spacing study swept
`K_ATTEN` as far as 100 — roughly 300× the calibrated value — and reports "the
smallest swept K_ATTEN at which any skin type prefers a non-daily interval =
none up to 100.0" (`spacing_study/results/FINDINGS.md:408`). The manuscript
puts the same result in physical units: at 1 SED every 3 days the melanin index
changes by 0.011 over a full year, altering mean serum 25(OH)D by 0.16 nmol/L
(`paper/manuscript.tex:988`).

**Conclusion, stated plainly:** *within this model, melanin dynamics have no
leverage over the optimal schedule.* Neither the rate at which a tan builds,
nor the rate at which it fades, nor the strength of its optical penalty moves
the optimal interval anywhere in or well beyond its documented range. The
skin-type dependence the model does exhibit comes from **constitutive**
pigment, which is a static per-FST multiplier and not a dynamic quantity at
all.

### 5.2 An important honesty caveat about the sweeps

In sections 2-4 of `RESULTS.md` the optimum sits at **1 day, the finest
interval tested**. That is a *corner solution*. "Nothing moves the optimum" is
therefore a weaker statement than it looks: a parameter would have to be strong
enough to push the optimum off the grid boundary before any movement could be
observed at all.

Three things stop this from hollowing out the conclusion. First, the reported
steady-state 25(OH)D values — the objective, not just its argmax — barely move
either, and those are not boundary-constrained. Second, the one-at-a-time
sensitivity table (`RESULTS.md:2663-2677`) is a magnitude comparison, not an
argmax comparison, and it gives the same ranking. Third, the spacing study
pushed `K_ATTEN` two orders of magnitude past its calibrated value and still
never produced a non-daily optimum. The conclusion is well supported; the
single line "the optimum does not move" is not by itself sufficient evidence
for it.

### 5.3 Where the interval structure actually comes from — the brief's premise does not survive checking

The task that produced this document asked me to cross-check the melanin result
against the claim that "the interior structure that does exist comes from the
25(OH)D feedback term", citing the `feedback_mode` entry in the `vitd_model.py`
docstring. **That claim is stale, and the repository has already retracted it.**

The docstring does say something close to it: "Ordering comes from HAVING a
decreasing feedback factor: at incumbent A the argmax is 3 d under linear,
biphasic AND both saturating variants alike" (`vitd_model.py:293-299`, dated
2026-08-18). That entry is itself a correction of an earlier, more specific
claim naming `(1 - D/D_max)`.

`RESULTS.md:2783-2796` (generated later, from the 2026-08-21 spacing study)
retracts it further and is unambiguous:

> The interval ordering these sweeps produce comes from **neither**
> photoadaptation **nor** the D-feedback term. It is a property of the
> logarithmic dose-response itself. …That claim was tested directly in
> `spacing_study/` (2026-08-21) and **does not hold**: disabling `g(D)`
> entirely leaves the ordering unchanged and *widens* daily dosing's margin
> rather than narrowing it.

The mechanism the spacing study identifies instead is **concavity of the
dose-response**: because `S = A * ln(1 + UV * E)` is concave, at a fixed total
UV budget dividing that budget into more and smaller sessions always yields
more 25(OH)D. Quantitatively, splitting a fixed annual SED budget into
every-2-day rather than daily sessions costs 9.6% of steady-state mean 25(OH)D
at FST V; weekly costs 29.4% (`spacing_study/results/FINDINGS.md:19-27`).
"D-feedback and photoadaptation both modulate the *size* of the gap; neither
creates it" (`RESULTS.md:2795-2796`).

**So the correct statement is a three-part one:**

1. Melanin dynamics (`alpha`, `beta`, `K_ATTEN`, `M_TAN_HEADROOM`) have
   essentially no leverage over the optimal schedule in this model. This is
   robustly supported.
2. The 25(OH)D feedback term does **not** carry the interval ordering either.
   That hypothesis was tested by direct ablation and rejected.
3. What carries it is the concavity of the logarithmic dose-response — a
   structural property of the synthesis functional form, not of any
   photoadaptation or homeostatic parameter.

There is a live inconsistency here between `vitd_model.py:293-299` (feedback
carries ordering) and `RESULTS.md:2783-2796` (it does not). The `RESULTS.md`
statement is the later of the two and is backed by a dedicated ablation study;
the docstring line predates it. I have not edited either file. A reader
comparing them should prefer `RESULTS.md` and note the docstring needs
updating.

### 5.4 What this means for the melanin sub-model's role

The melanin sub-model is not load-bearing for scheduling conclusions. It is
load-bearing for **absolute** 25(OH)D predictions by skin type, and there
almost all the weight sits on one static parameter, `K_const`, whose four
source routes disagree by roughly 6× (§4.5, **[UPDATED 2026-09-21]** — was
5.3× under the pre-2026-09-06 table), multiplying a baseline melanin table
whose two darkest cells now rest on n = 84 (FST V) and n = 32 (FST VI) from
an independent corroborating cohort, but still on a clipped boundary value
for FST VI (§4.4, **[UPDATED 2026-09-21]** — the darkest cells no longer rest
on n = 4 apiece; that was the pre-2026-09-06 table).

That is an uncomfortable place for the parameters to be, but it is a *different*
discomfort from the one a reader might expect. The dynamic photoadaptation
machinery — the part with the most equations — is the part that matters least.
The reason its citation trail has been audited so heavily
(`vitd_model.py:526-548`, `vitd_calibration.py`, `tests/test_matsuoka1990.py`)
is historical: it was once believed to drive the scheduling result. It does
not. Effort spent tightening `alpha` or `beta` will not move any headline
number in this repository; effort spent on `K_const`, on the FST V-VI baseline
melanin values, and on `A` will.

### 5.5 The §5.1 sweeps predate the 2026-08-25 `M_TAN_HEADROOM`/`SC_max_by_FST` change

`calibration_approval.yaml`'s own `consequence` field for the `M_TAN_HEADROOM`
change says it plainly: "Downstream effect on headline numbers is NOT yet
quantified — `run_all.py` has not been re-run against this value." Everything
quoted in §5.1 above — the `beta`, `alpha` and `K_ATTEN` sweeps, and the
one-at-a-time sensitivity table — was generated against the *old* ceiling
(0.20) and the *old* FST I-III stratum-corneum cap (0.23). Given §5.1's own
finding that melanin dynamics have essentially no scheduling leverage, and
that raising a ceiling by 0.038 (M_TAN_HEADROOM) or 0.01 (SC_max_by_FST
I-III) is a small change relative to the ranges already swept (`beta` over
21-120 d half-lives, `K_ATTEN` over a 40x range), the §5.1 conclusion is
unlikely to flip — but this is an expectation, not a re-verified result. Quote
§5.1's figures as pre-2026-08-25, and re-run
`PYTHONIOENCODING=utf-8 python3 run_all.py` before citing any absolute
25(OH)D number that depends on `M_TAN_HEADROOM` or `SC_max_by_FST` I-III.

---

## 6. Reproducing everything in this document

```bash
# Live melanin parameters + derived quantities + Matsuoka deconvolution
PYTHONIOENCODING=utf-8 python3 scripts/melanin_parameters_report.py

# M_TAN_HEADROOM (Ravnbak 2008) and SC_max_by_FST I-III (de Boer 2025 +
# Miyamura 2011) derivations, plus the alpha/beta identifiability analysis
# and the Ravnbak 2009 (UVA) negative control -- added 2026-08-25
PYTHONIOENCODING=utf-8 python3 scripts/fit_melanin_kinetics.py

# Matsuoka K_ATTEN derivation on its own (plus other calibration checks)
PYTHONIOENCODING=utf-8 python3 -m vitd_calibration

# K_const derivation, four within-study routes, all [CHOICE]/[ESTIMATE] tags
PYTHONIOENCODING=utf-8 python3 scripts/kconst_recalibration.py

# Assertions behind §4.3 and §5.1
python3 -m pytest tests/test_matsuoka1990.py -q
```

The sweeps in §5.1 are regenerated by `run_all.py`, which takes about ten
minutes and was **not** run for this document — §5.1 quotes the committed
`RESULTS.md` as generated, and every figure quoted there was checked against
its line in that file.
