# `basics/` — Tier-1 single-mechanism reference models

**Added 2026-08-21.** Two deliberately tiny models, each isolating **one**
mechanism, each checked against **one** dataset, each short enough to read in
one sitting.

They exist because the full model has ~25 parameters and 13 validation checks,
and a reviewer cannot tell from it which term is carrying which result. These
files let you check one thing at a time — and let you say "this piece is right"
without having to defend the whole stack.

They are **reference models, not replacements**. Neither is fitted to anything.
Neither makes an absolute 25(OH)D prediction. Nothing in the main pipeline
imports them.

Rationale: [`../docs/REPO_SCOPE_AUDIT.md`](../docs/REPO_SCOPE_AUDIT.md) §6.

## The two models

### `melanin_only.py` — the mechanism that works

```
response(FST, dose) = ln(1 + dose * E_source) * exp(-K_const * M(FST))
```

At matched dose the dose term cancels, so every skin-type ratio collapses to
`exp(K_const * ΔM)`. **The entire skin-type gradient of this repository rests on
exactly two things: the scalar `K_const` and the `M_baseline_by_FST` table.**
The synthesis scalar `A` and the BSA factor cancel out algebraically, which is
the point — nothing in the comparison depends on `A`, the model's single largest
source of uncertainty and the one anchored on FST I–II data only.

What it shows, run against the five within-study routes `K_const` was fitted to:

| | |
|---|---|
| Direction of the gradient | **Unanimous** across all five routes — darker skin responds less at matched dose |
| Implied `K_const` per route | 0.274 (Young 2020 Arimed B) to 1.388 (Darling 2013) |
| Spread | **5.1×** disagreement, mean 0.781, SD 0.466, against 0.98 in force |
| FST V : II optical factor | 2.005, inside the approved 1.7–2.7 band |

The honest reading: the **direction** is robust and publishable; the **size** is
not a point estimate and should be reported as a range. Two matched-dose *lamp*
studies (Felton, Young) disagree with each other, which no amount of modelling
reconciles.

This is a **consistency check, not independent validation** — these are the same
routes the aggregate fit used. Its value is that it makes the spread visible,
which the aggregate value hides.

### `sc_only.py` — the mechanism that is underdetermined

```
dSC/dt      = alpha_SC * UV * (SC_max - SC) - k_decay_SC * SC
attenuation = exp(-K_SC * SC)
```

Built to demonstrate a **negative result**, not to hide one. Against de Boer
2025 (0.8 SED 3×/week × 9 weeks; the entire dataset is three group medians):

1. **The endpoint is reproduced — and that is not evidence.** `SC_max` was set
   *from* those endpoints, so agreement restates the input.
2. **`alpha_SC` and `SC_max` trade off along a ridge.** Across the 0.20–0.40
   evidence band for `SC_max`, `alpha_SC` spans a factor of **13** while fitting
   the data identically. The data constrain a combination, not either parameter.
3. **`K_SC` is not constrained by this dataset at all.** de Boer measured
   stratum corneum *thickness*. It did not measure vitamin D. `K_SC = 1.0` is a
   bare prior.
4. **The observed shape is not reproduced, and structurally cannot be.** The
   measured SC dips ~26% *below* baseline at week 5 before overshooting; a
   saturating rise minus first-order decay cannot go below its own initial
   condition. No parameter value fixes this.

What would fix it: individual-level SC time courses from the de Boer 2025
supplementary, at finer resolution than three visits. That one dataset breaks
the `alpha_SC ~ SC_max` ridge. `K_SC` would need a study measuring 25(OH)D
response against *measured* SC thickness; none is known to exist.

## Why these two, and not others

Because between them they cover the two mechanisms that could carry the
scheduling result, and they reach opposite verdicts — which is exactly what a
reviewer needs to see:

- Melanin (`K_const`) carries the **skin-type** result, and survives scrutiny in
  direction.
- Stratum corneum (`K_SC`, `alpha_SC`, `SC_max`) does not survive scrutiny as
  calibrated parameters. [Note, 2026-08-24: at the time this was written
  `K_ATTEN` was a retired structural constant pinned at 0, making SC the only
  live photoadaptation channel; `K_ATTEN` was reinstated as an active,
  calibrated parameter on 2026-08-21 (`vitd_model.py`, `RETIRED_PARAMETERS` is
  now empty), so facultative melanin is a second live photoadaptation channel
  again. This file's own identifiability analysis above is unaffected — it
  already treats `K_ATTEN` as jointly unidentifiable with `K_SC`.]

The third candidate — that photoadaptation drives the **interval** result — was
tested separately in [`../spacing_study/`](../spacing_study/) and refuted: the
interval ordering comes from the concavity of `ln(1 + UV·E)`, not from any state
variable.

## Running them

```
PYTHONIOENCODING=utf-8 python3 basics/melanin_only.py
PYTHONIOENCODING=utf-8 python3 basics/sc_only.py
```

Both read their parameter values from `vitd_model.ModelParams` at runtime and
never restate them (per `CLAUDE.md`). Both print a verdict section that
separates SUPPORTED from NOT SUPPORTED from EXTRAPOLATED.
