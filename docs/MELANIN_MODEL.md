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

**Between pulses** — sub-stepped at `dt = 0.5 d` (`vitd_model.py:1082-1105`):

```python
M = M_base + (M - M_base) * np.exp(-beta_eff * delta_time)   # vitd_model.py:1101
```

This is the *exact* solution of the decay half of the ODE with `UV = 0`, not a
finite-difference approximation of it, so there is no integration error in the
decay channel.

**At each pulse** (`vitd_model.py:1134-1141`):

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
   per-session dose (`vitd_model.py:1137-1140`), rising toward `alpha_max` as
   `sed` exceeds a per-FST half-max dose. The docstring ODE
   (`vitd_model.py:21`) omits this entirely. The manuscript *does* state it
   (`paper/manuscript.tex:153`), so the discrepancy is docstring-vs-code, not
   paper-vs-code. Consequence, stated in the manuscript: "a single large
   session induces more tanning per unit dose than several small ones."

2. **The ceiling is per-FST, not the global `M_max`.** The code uses
   `M_ceil = params.M_ceiling(fst)` (`vitd_model.py:1046`), defined as
   `M_baseline(fst) + M_TAN_HEADROOM` and deliberately **not** capped at 1.0
   (`vitd_model.py:788-801`). The docstring's `M_max` symbol survives as
   `ModelParams.M_max` (`vitd_model.py:637`), and **nothing in the core
   pipeline reads it** — a grep for `.M_max` finds only the
   `M_max_override` plumbing at `vitd_model.py:1008,1046`. It is a dead
   parameter that still appears in the published equation. The reason for the
   change is recorded at `vitd_model.py:626-635`: a single global `M_max=0.95`
   sat *below* `M_baseline_by_FST[6] = 1.00`, so the saturating update made UV
   exposure *lighten* FST VI skin — an inversion that mattered because FST V-VI
   is this project's target population.

3. **Pulses are instantaneous.** No decay occurs during a session, and
   synthesis is evaluated at the *pre-pulse* melanin state
   (`vitd_model.py:1120-1124` calls `synthesis_per_session(sed, M, M_base, ...)`
   before `M` is updated at `:1141`). Within-session self-shielding by melanin
   induced during that same session is therefore not represented. At realistic
   per-session doses the induced increment is ~1e-3 on the melanin scale
   (companion script, section 3), so this is a small approximation, but it is
   an approximation.

4. **Decay is toward the per-FST constitutive baseline, not toward zero.**
   `M_base = params.M_baseline(fst)` (`vitd_model.py:1032`) is the floor. Only
   the *facultative* excess fades. This matches the ODE and is noted here only
   because it is what makes the constitutive/facultative split in §3 coherent.

---

## 2. Parameter-by-parameter provenance

Values: run the companion script. Citations: `vitd_model.py` docstring lines
given per row. Status flags in the right-hand column are the manuscript's own
(`paper/manuscript.tex:187-201`).

### 2.1 `alpha` / `alpha_by_FST` — melanin induction base rate (1/SED)

- **Code:** `vitd_model.py:604-608`. Lower for FST IV-VI than for I-III.
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

- **Code:** `vitd_model.py:609`.
- **Publication:** none. `paper/manuscript.tex:193`: "Hill-form ceiling; not
  independently sourced."
- **Manuscript status flag:** PRIOR.

### 2.3 `UV_halfmax_by_FST` — Hill half-max dose for induction (SED)

- **Code:** `vitd_model.py:610-615`; docstring entry `vitd_model.py:228-229`.
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
  (`vitd_model.py:228-229`); the manuscript adds that it is Table IV and
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

- **Code:** `vitd_model.py:616`, written as `np.log(2) / 56.0`, i.e. a 56-day
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
  (`RESULTS.md:445`). It is now the inversion of a published procedure, with a
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

- **Code:** `vitd_model.py:636`; used by `ModelParams.M_ceiling`
  (`vitd_model.py:788-801`). **Changed 2026-08-25: 0.20 -> 0.238**
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
  tanning monotone for every FST" (`vitd_model.py:632-633`, pre-2026-08-25
  text). `scripts/fit_melanin_kinetics.py` (added 2026-08-25) mapped Ravnbak
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

- **Code:** `vitd_model.py:759-761`; docstring `vitd_model.py:283-295`.
- **Publication / data:** the individual-level Young 2020 trial dataset held in
  this repo as `Papers_and_Datasets/Final skin type vit D database
  26_03_2020.xlsx`, column `ITA_Unexposed` (individual typology angle of
  unexposed skin, degrees).
- **Arithmetic from measurement to parameter:** stated in code as
  `M = 1 - (ITA_U + 30) / 85`, clipped to [0, 1] (`vitd_model.py:285`,
  `:754-758`). That is a linear map sending ITA = +55° to M = 0 and ITA = −30°
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
  (`vitd_model.py:286`), which the mapping sends to −0.035, clipped to 0.
- **Caveat on the stated sample size.** The docstring says "ArimedB trial data,
  N=44" (`vitd_model.py:284-285`). The `ArimedB irradiated` sheet does have
  exactly 44 rows, but only 28 of them carry an `ITA_Unexposed` value, and
  those 28 alone reproduce {II 0.074, III 0.186, IV 0.481, V 0.772, VI 0.956},
  which does **not** round to the shipped table. The pooled irradiated +
  controls set (n = 48) does. The published N is therefore attached to the
  wrong subset; the values themselves are sound and reproducible. This is a
  documentation defect, not a numerical one, and I have not edited
  `vitd_model.py` to fix it.
- **Manuscript status flag:** PRIOR, with FST V and VI explicitly flagged as
  having "no direct published melanin-index validation"
  (`paper/manuscript.tex:199`). See §4.4.

### 2.7 `K_const` — constitutive-pigment attenuation coefficient

- **Code:** `vitd_model.py:622`; docstring entry `vitd_model.py:154-181`;
  derivation `scripts/kconst_recalibration.py`; diagnosis
  `scripts/fst_gap_decomposition.py`; approval `calibration_approval.yaml`.
- **Publications:** four independent within-study routes, two lab and two
  field:
  - **Felton 2016**, *Br J Dermatol* 175:1320-1328 — phototype II vs V, same
    cabinet, same protocol, both baselines printed
    (`scripts/kconst_recalibration.py:84-100`). Full text in
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
  arithmetic mean of the four study-level implied `K_const` values
  (`scripts/kconst_recalibration.py:598-604`, objective at `:410-421`, weighting
  choice at `:605-612`). The four implied values recorded in the docstring
  (`vitd_model.py:167-169`) are Felton 1.473, Young 0.278, Kift 0.788, Darling
  1.387 **[read from the docstring 2026-08-23]**; their mean is 0.9815, which
  rounds to the shipped value, and their sample SD is 0.559, matching the
  docstring's stated 0.56. Weighting is one vote per *study*, not per subject
  or per contrast, an explicitly flagged `[CHOICE]`
  (`scripts/kconst_recalibration.py:605-612`).
- **Status in the docstring's own words:** "treat 0.98 as a central estimate of
  a 0.3-1.5 range, not a measured constant" (`vitd_model.py:170-171`). See
  §4.5.

### 2.7a `SC_max_by_FST` — stratum corneum thickening ceiling (I-III cell)

This parameter is not part of the melanin state `M`, but its 2026-08-25
change shares the same evidence package (Miyamura 2011) as §2.5, so it is
recorded here rather than left undocumented.

- **Code:** `vitd_model.py:785` (`ModelParams.SC_max_by_FST`), consumed at
  `vitd_model.py:1126`. **Changed 2026-08-25: FST I-III 0.23 -> 0.24**
  (`calibration_approval.yaml`, entry "SC_max_by_FST I-III 0.23 -> 0.24
  (stratum corneum ceiling)", APPROVED). FST IV-VI is **unchanged at 0.36**.
  The scalar `SC_max` default (0.30) is unchanged.
- **Publications, now two independent routes:**
  - de Boer 2025 Supplementary Table S1: FP I-III stratum corneum 5.6 -> 6.9
    um, i.e. **+23%**. This was the sole basis for `SC_max_by_FST` until now.
  - Miyamura 2011 (PMID 20979596) Table 1: stratum corneum 16.6 -> 20.7 um in
    FST II-III after 10 suberythemal UVB exposures over 2 weeks, i.e.
    **+24.6988%**. Independent cohort, protocol, and measurement method
    (histology of formalin-fixed biopsies vs. de Boer's technique).
- **Arithmetic:** two independent studies now bracket the FST I-III
  stratum-corneum-thickening ceiling at +23% and +24.6988%. 0.24 sits between
  them (midpoint 0.238494); 0.24 was chosen by the repository owner as the
  round value inside the bracket.
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

- **Code:** `K_ATTEN_DEFAULT` at `vitd_model.py:528`, consumed by
  `ModelParams.K_ATTEN` at `vitd_model.py:642`; docstring entry
  `vitd_model.py:272-282`; derivation `vitd_calibration.py:28-145`.
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
- **Arithmetic from measurement to parameter** (`vitd_calibration.py:105-142`),
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
  (`vitd_calibration.py:148-196`). Under the model's default biphasic feedback
  the residual melanin attenuation is 6.6%; under the legacy linear form it is
  18.1%; with feedback switched off entirely the naive 49.5% is recovered. The
  adopted value is the biphasic one, i.e. the **lower bound** of the range the
  study can support (`vitd_model.py:280-282`). Run the companion script,
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
  `vitd_sensitivity.py`, `scripts/parameter_optima.py`,
  `docs/MODEL_AND_EVIDENCE_RECONSTRUCTED.md`, and the `vitd_model.py`
  docstring has been updated in all of those places to "0.288972-0.838389".
- **Manuscript status flag:** SINGLE-SOURCE PRIOR (`paper/manuscript.tex:187`).
  See §4.3.

### 2.9 Cross-references that are stale

Two places in the repository still describe an older state of this parameter,
and a reader following either of them will be misled:

- `paper/manuscript.tex:187` prints a different value from the code.
  `paper/manuscript.tex:153` also states the linear feedback term is the
  model's default, which `vitd_model.py:665` contradicts.
- `spacing_study/results/FINDINGS.md:35-41,370-372` was generated while
  `K_ATTEN` was still pinned at zero and says so (a hand-added staleness
  banner at the top of that file flags this as of 2026-08-23).

[Update 2026-08-24: the third item previously listed here -- `vitd_model.py`'s
own "Gap 2" block allegedly still saying `K_ATTEN` is "CLOSED BY RETIREMENT" --
is no longer accurate. That block (`vitd_model.py:348-353`) now reads
"REINSTATED and calibrated 2026-08-21 ... RETIRED_PARAMETERS is now empty",
consistent with the live `K_ATTEN` entry above it. `vitd_model.py` is
internally consistent on this point; only the two files above still lag it.]

I have not edited `paper/manuscript.tex` or `spacing_study/results/FINDINGS.md`;
other agents are working in them.

---

## 3. Where melanin enters synthesis — two separate exponential terms

The synthesis function is `synthesis_per_session()`
(`vitd_model.py:828-902`). Pigment enters through **two distinct exponential
factors with separately derived coefficients**, computed at
`vitd_model.py:857-859` and multiplied into the return value at
`vitd_model.py:901`:

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
   tanned-vs-untanned crossover (§2.8). `vitd_calibration.py:185-190` makes
   this explicit: "The two were never the same number."

2. **They are jointly unidentifiable from the current data.**
   `RESULTS.md:1657` reports a `K_const ~ K_ATTEN` pairwise ridge of 0.9998 —
   over 99.9% of the steady-state response surface is explained by a single
   linear combination of the two. `K_ATTEN ~ K_SC` is 0.9990. Reporting either
   member of such a pair as a separately calibrated value overstates what the
   data support (`RESULTS.md:1752-1758`).

3. **The magnitudes are very unequal.** The companion script (section 4)
   prints both: the constitutive factor spans the full FST range while a fully
   developed tan costs a single-digit percentage of synthesis. The
   manuscript's own summary is that constitutive pigment and the stratum
   corneum barrier "account for over 94% of the total skin-type synthesis gap"
   (`paper/manuscript.tex:988`).

Note also that `M` enters synthesis at its **pre-pulse** value
(`vitd_model.py:1120`, before the update at `:1141`), and that `excess_M` is
floored at zero, so a caller who initialises `M0` below `M_baseline` gets no
facultative *bonus* — the term is one-sided by construction
(`vitd_model.py:857`).

---

## 4. Limitations (mandatory section)

Each item below was checked against the sources; where I could not verify a
claim, that is stated.

### 4.1 The induction rate is reconstructed and light-skin-anchored — VERIFIED

`docs/MODEL_AND_EVIDENCE_RECONSTRUCTED.md:95-110` records Gap 7 with
**Confidence: RECONSTRUCTED**, meaning the provenance itself was rebuilt from
secondary citation sites rather than read from a primary record. The
substantive claims:

- `alpha` "was originally calibrated from Miyamura 2011, a light-skin
  photoadaptation kinetics study" (`:96-98`), corroborated at
  `paper/README.md:183`, `scripts/parameter_optima.py:99` ("light-skin
  derived"), and `vitd_analyses.py:83-84`.
- "A dedicated re-fit from Young 2020 individual-level trial data was attempted
  and found alpha structurally unidentifiable from that dataset" (`:98-101`).
  The manuscript quantifies this as "92.5% of fits hit search bounds"
  (`paper/manuscript.tex:191`).
- **Status: open** (`:107`). `alpha` remains light-skin-anchored.

**What I could not verify:** the "92.5% of fits hit search bounds" figure has
no reproducing script in the repository. The only Young-2020 fitting script,
`scripts/fit_katten_young2020.py`, fits `K_ATTEN` (not `alpha`) on the n=23
TL01 arm. The alpha re-fit is reported but not reproducible from this tree, so
the 92.5% figure should be cited as a recorded historical result, not as
something a reader can re-run.

`docs/REPO_SCOPE_AUDIT.md:244` lists `alpha`, `beta` and `K_ATTEN` among
parameters that "barely participate in the model at all", which lowers the practical stakes of the gap
without closing it (`docs/MODEL_AND_EVIDENCE_RECONSTRUCTED.md:107-110`). §5
below shows why.

### 4.2 The decay rate is a choice, not a fitted half-life — VERIFIED

`RESULTS.md:445` states it directly: "The 56-day half-life (default beta) is a
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
`RESULTS.md:445` and `paper/manuscript.tex:110,197` give the reason: Tadokoro
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
`vitd_model.py:315-319`).

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
a range of roughly 3× (companion script section 6; `vitd_calibration.py:148-176`;
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

### 4.4 The darkest-skin baselines are group-level priors — VERIFIED, and worse than the flag suggests

`vitd_model.py:293-295`: "WARNING: FST V (0.78) and VI (1.00) are group-level
priors with NO direct melanin-index validation from published data. Do not use
as individual predictors." `paper/manuscript.tex:199` and the docstring's Gap 1
block (`vitd_model.py:305-312`) agree: Wilkes 2015 gives only a partial
Mexameter threshold, Alaluf 2001 gives quantitative melanin in FST V/VI but not
as a baseline melanin-index table, and "no paper provides separate untanned FST
V and FST VI baseline MI means."

Two things my own recomputation (§2.6) adds:

- The FST V and VI cells rest on **n = 4 each** in the underlying trial
  dataset. That is a small-sample group mean, not a population prior.
- The measurement is **ITA from colorimetry, not a melanin index**, mapped onto
  the model's 0-1 scale by an affine transform whose two anchor points
  (ITA = +55° → 0, ITA = −30° → 1) are not themselves sourced anywhere I could
  find. The FST VI cell is additionally **clipped**: the raw mapping returns
  1.077 and the code caps it at 1.0 (`vitd_model.py:285`). So the darkest cell
  is not a mean at all; it is a saturated boundary.

This compounds through `K_const`, which multiplies these values: any change to
`M_baseline_by_FST` rescales the entire skin-type penalty
(`vitd_model.py:756-758`). Every FST VI conclusion in the model is an
extrapolation of one coefficient onto an unvalidated prior
(`scripts/kconst_recalibration.py:44-48`).

### 4.5 `K_const` is the centre of four mutually inconsistent routes — VERIFIED

The four implied values (§2.7) span 0.278 to 1.473 — a **5.3× spread** — with a
sample SD of 0.56 against a mean of 0.98. The docstring's own instruction is to
"treat 0.98 as a central estimate of a 0.3-1.5 range, not a measured constant"
(`vitd_model.py:170-171`).

The disagreement is not noise, and the derivation script says so without a
model in the loop: "Young reports FST II:VI = 1.29-1.43 while Felton reports
FST II:V = 2.14 in the same kind of cabinet, and under ANY monotone pigment
penalty II:VI must EXCEED II:V"
(`scripts/kconst_recalibration.py:581-585`). The two lab studies are therefore
*mutually incompatible within this model structure*, and 0.98 "is their and the
field's centre, not a reconciliation" (`vitd_model.py:352`). Averaging them
hides a structural conflict behind a mean.

Provenance narrowness compounds this: every FST V datapoint behind the number
is Indian-subcontinent ancestry at UK latitude, every FST VI datapoint is
lamp-based, and there is no field dosimetry in FST VI anywhere in this repo
(`scripts/kconst_recalibration.py:40-48`, `vitd_model.py:176-181`).

**A live datapoint on that inconsistency.** `vitd_calibration.young2020_check`
(`vitd_calibration.py:230-303`) compares the model's FST II:VI synthesis ratio
against Young 2020's own 1.29-1.50 band. Running it on 2026-08-23 returns
**2.92** — roughly double the top of the band, i.e. the model now over-predicts
the melanin gap on the one within-study contrast it can be checked against
directly, while under-predicting the naturalistic Webb/Kift ratio by about half
(`RESULTS.md:1762-1764`). The committed artefact `tables/debug/young2020_check.csv`
still records 1.4868 from a much earlier commit, so this drift is not visible in
the tracked outputs. The check is self-labelled UNVERIFIED / "calibration check
only — not independent validation", so it is not a failing benchmark; but the
direction of the discrepancy is opposite to the naturalistic one, which is
exactly the Felton-vs-Young structural conflict above showing up as an output.
I did not rewrite the committed CSV.

Finally, `K_const` excludes behaviour by design — shade, clothing and time
outdoors are a separate 2.1-2.4× factor that deliberately does not live here
(`vitd_model.py:172-176`) — so `K_const` is not the real-world skin-type
penalty and should never be quoted as one. The model still fails its one
held-out benchmark (Webb/Kift 2018's 2.5-3.0× FST V dose penalty), under-
predicting it by roughly half (`paper/manuscript.tex:65`, `RESULTS.md:1762-1764`).

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

**Melanin decay rate (`beta`), `RESULTS.md:443-471`.** The sweep covers
half-lives of 21, 30, 56, 80 and 120 days. At FST V the optimal interval is
1 day at every one of them, and steady-state 25(OH)D moves by at most 0.1
nmol/L across the whole range. At FST VI the optimum is 1 day at every value
and steady-state 25(OH)D does not move at all to the reported precision.

**Melanin induction rate (`alpha`), `RESULTS.md:411-441`.** Swept over
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

**One-at-a-time sensitivity, `RESULTS.md:1635-1644`.** Over documented prior
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
sensitivity table (`RESULTS.md:1635-1644`) is a magnitude comparison, not an
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
biphasic AND both saturating variants alike" (`vitd_model.py:218-222`, dated
2026-08-18). That entry is itself a correction of an earlier, more specific
claim naming `(1 - D/D_max)`.

`RESULTS.md:1702-1716` (generated later, from the 2026-08-21 spacing study)
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
creates it" (`RESULTS.md:1715-1716`).

**So the correct statement is a three-part one:**

1. Melanin dynamics (`alpha`, `beta`, `K_ATTEN`, `M_TAN_HEADROOM`) have
   essentially no leverage over the optimal schedule in this model. This is
   robustly supported.
2. The 25(OH)D feedback term does **not** carry the interval ordering either.
   That hypothesis was tested by direct ablation and rejected.
3. What carries it is the concavity of the logarithmic dose-response — a
   structural property of the synthesis functional form, not of any
   photoadaptation or homeostatic parameter.

There is a live inconsistency here between `vitd_model.py:218-222` (feedback
carries ordering) and `RESULTS.md:1702-1716` (it does not). The `RESULTS.md`
statement is the later of the two and is backed by a dedicated ablation study;
the docstring line predates it. I have not edited either file. A reader
comparing them should prefer `RESULTS.md` and note the docstring needs
updating.

### 5.4 What this means for the melanin sub-model's role

The melanin sub-model is not load-bearing for scheduling conclusions. It is
load-bearing for **absolute** 25(OH)D predictions by skin type, and there
almost all the weight sits on one static parameter, `K_const`, whose four
source routes disagree by 5.3× (§4.5), multiplying a baseline melanin table
whose two darkest cells rest on n = 4 apiece and a clipped boundary value
(§4.4).

That is an uncomfortable place for the parameters to be, but it is a *different*
discomfort from the one a reader might expect. The dynamic photoadaptation
machinery — the part with the most equations — is the part that matters least.
The reason its citation trail has been audited so heavily
(`vitd_model.py:272-321`, `vitd_calibration.py`, `tests/test_matsuoka1990.py`)
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
