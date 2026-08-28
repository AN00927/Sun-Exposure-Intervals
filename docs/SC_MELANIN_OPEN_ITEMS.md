# Handoff: open items from the `basics/melanin_only.py` + `basics/sc_only.py` rewrite

Read `CLAUDE.md` first, in full, before touching anything below — it governs
precision, citation, calibration-approval, and documentation rules for this
repo, and every item here has to respect it.

## Context (read this before doing anything)

Two subagents rewrote `basics/melanin_only.py` and `basics/sc_only.py` from
simplified toy reference models into genuine isolations of the full
mechanisms `vitd_model.py` actually runs (melanin: constitutive + facultative
dynamics, SC ablated; SC: the real two-process `SC_build - SC_damage`
structure). `basics/README.md` was rewritten to match, source-first, with a
full derivation trail for every parameter. All of that is done, committed to
the working tree (not yet git-committed), and verified: both scripts run
clean, and `pytest tests/ -q --ignore=tests/exploratory` passes 463/463.

While doing that work, both subagents surfaced findings they were told to
report rather than silently resolve. This document is the handoff for those
findings. None of them were acted on beyond the flagging described here —
each needs either a source-tracking effort, a judgment call from Ahaan, or
(in one case) a possible `calibration_approval.yaml` item, none of which is
this handoff's job to decide unilaterally.

**Standing constraint for whoever picks this up:** none of these items
license changing a calibrated parameter value in `vitd_model.py` without
running that change through `calibration_approval.yaml` first (see
`CLAUDE.md`'s bug-fix-vs-calibration distinction) — most of the items below
are citation/documentation gaps, not calibration errors, and should stay that
way unless investigation turns up an actual bug (unit error, inverted sign,
state-update error), which *would* be in-scope to fix directly.

---

## 1. `k_repair_SC`'s 7-day half-life appears to be an unsourced structural convenience

**Where:** `vitd_model.py:818` (`k_repair_SC: float = np.log(2) / 7.0`) and
the comment block at `vitd_model.py:836-839`.

**Finding:** The comment says the value was "set so the T1 dip recovers by T2
under continued dosing" — i.e. chosen to make the *qualitative shape* plausible,
not measured. The subagent searched de Boer 2025, Miyamura 2011, and Pearse
1987 (the three papers cited anywhere near the SC subsystem) and found no
repair-rate measurement in any of them.

**Action:**
- Search the repo (`Research_Papers/`, gitignored but present on disk, and
  `Papers_and_Datasets/`) for any other stratum-corneum-barrier-repair
  citation that might already be sitting on disk under a misleading filename
  — `CLAUDE.md` flags this as a repeat failure mode ("Verify before claiming
  unobtainable").
- If nothing turns up, the honest fix is a comment-and-docstring change (not
  a value change) in `vitd_model.py` and `basics/README.md`/`docs/` stating
  plainly that `k_repair_SC` is a structural-plausibility prior with no
  independent citation, rather than implying "days-scale" is itself sourced.
  This is a documentation fix, not a calibration change.

## 2. `k_decay_SC`'s 21-28 day range is attributed to "epidermal turnover literature," not to Pearse 1987 — but Pearse is cited in the same breath

**Where:** `vitd_model.py:826-828`:
```
# k_decay_SC: 21-28 day half-life. Pearse 1987 (P23) measured SC after
#   6 weeks of 3x/week UVB. Epidermal turnover literature: 26-28 days.
```

**Finding:** The subagent read Pearse 1987 in full. It is a single-timepoint,
between-arm dose-threshold study (SC thickness after 6 weeks at 0.5x/1.0x/2.0x
MED vs. control) — it contains no decay/turnover measurement at all, so it
cannot be the source of the specific "21-28 day" or "26-28 day" figures. The
comment's own phrasing ("epidermal turnover literature") suggests the real
source is a separate, uncited paper — a very standard dermatology figure,
often traced to Halprin 1972 (*Arch Dermatol* — epidermal turnover time), but
that paper is not in this repo and is not named anywhere in it.

**Action:**
- Track down the actual source for the 21-28 (or 26-28) day figure — Halprin
  1972 is a reasonable first guess to check for and try to obtain, but do not
  assume it's the source without confirming the number actually appears there.
- If found: add the citation to `vitd_model.py`'s comment and to
  `basics/README.md`, and stop citing Pearse 1987 for a number it does not
  contain.
- If not found: relabel the range in both places as an unsourced/textbook
  figure rather than implying it traces to a specific paper this repo holds.
- Either way, correct the misleading juxtaposition in `vitd_model.py:826-828`
  that currently reads as if Pearse supports the half-life number.

## 3. Miyamura 2011's SC-thickness corroboration is, by the paper's own text, not statistically significant — this caveat is missing from `vitd_model.py` and `docs/MELANIN_MODEL.md`

**Where:**
- `vitd_model.py:789-799` (the `SC_max_by_FST` calibration comment,
  "first out-of-sample support SC_max has had").
- `docs/MELANIN_MODEL.md:418-457`, section "2.7a `SC_max_by_FST`" — in
  particular the "What this is" bullet (line 440) which calls the +24.6988%
  figure corroboration of magnitude without flagging significance.

**Finding:** Miyamura 2011's own Table 1 / text reports the stratum-corneum
thickness change (16.6 -> 20.7 um, the +24.6988% figure this repo cites) as
**not statistically significant**: "the thickness of the stratum corneum ...
did not change significantly in any of the specimens," and "it seems
unlikely that skin thickness was a significant factor ... although it cannot
be entirely excluded." `basics/README.md` (this session's rewrite) now states
this caveat explicitly. `vitd_model.py` and `docs/MELANIN_MODEL.md` do not.

**Action:** Add the same caveat to both:
- `vitd_model.py:789-799`: one added line noting Miyamura's own NS finding on
  this specific endpoint, so the "first out-of-sample support" framing isn't
  overstated as independently *significant* corroboration.
- `docs/MELANIN_MODEL.md` section 2.7a: extend the "What it does NOT
  establish" bullet (line 447) to state the NS finding explicitly, matching
  `basics/README.md`'s current wording.
This is a documentation-accuracy fix (the underlying value doesn't change),
not a calibration change.

## 4. De Boer 2025's week-5 "dip" is significant pooled (p=0.016) but not within either phototype stratum alone (p=0.170 / 0.134)

**Where:** Newly transcribed by the SC subagent into `basics/sc_only.py`
(`DEBOER_T1_VS_T0_PVALUE`) and `basics/README.md`, from
`Research_Papers/Effect of Repeated Low-Dose UVR Exposure - de boer et al..pdf`,
Supplementary Table S1, page 12. This p-value breakdown was **not** present
in any prior version of `sc_only.py`, `scripts/fit_sc_deboer2025.py`, or
`vitd_model.py`'s comments.

**Action:**
- Decide whether this caveat belongs in `vitd_model.py`'s own SC comment
  block (`vitd_model.py:832-835`, the `k_damage` entry that describes the dip)
  since that file is the canonical source other docs point to. Currently only
  `basics/README.md` and `basics/sc_only.py` carry it.
- If added there, cross-check the transcribed p-values (0.016 / 0.170 / 0.134)
  against the source PDF once more independently before propagating further —
  they were read once, by one agent, from one page.

## 5. A dedicated `docs/SC_MODEL.md` may be warranted, mirroring `docs/MELANIN_MODEL.md`

**Finding:** Both subagents independently suggested this. The SC subsystem
now has three source papers (de Boer 2025, Miyamura 2011, Pearse 1987), a
structural fix made this session, and a numerical identifiability
investigation behind it — comparable in depth to what `docs/MELANIN_MODEL.md`
already documents for melanin, but with no equivalent dedicated document.
`basics/README.md`'s SC section is the closest thing that exists today.

**Action:** Judgment call for Ahaan — not acted on by either subagent, and not
urgent. If wanted, `docs/MELANIN_MODEL.md`'s structure (per-parameter
sections, a dedicated limitations section, exact arithmetic) is the template
to follow.

## 6. `M_ceiling()` is not actually clipped at 1.0

**Where:** `vitd_model.py`'s `ModelParams.M_ceiling()` method (returns
`M_baseline(fst) + M_TAN_HEADROOM`, explicitly documented in that file as
"NOT capped at 1.0").

**Finding:** The melanin subagent was briefed (by the orchestrating session,
not by any repo source) to expect a clip on `M_ceiling()` and was asked to
verify it. It found no such clip anywhere in the actual code. `basics/
melanin_only.py`'s new dynamic model was built around the real, uncapped
behavior, which is correct — this item is just flagging that the original
briefing's premise was wrong, in case a clip was intended to exist somewhere
else in the pipeline and is missing.

**Action:** Grep the repo for any other place a melanin ceiling clip might be
expected (e.g. `M_baseline_by_FST`'s own construction is separately clipped
to [0,1] — that is a different clip, on a different quantity, and is not in
question). If no such clip is missing anywhere, this item can simply be
closed as "premise was wrong, no bug" — do not add a clip that doesn't
already exist without confirming it's actually needed; that would be a
model-behavior change requiring `calibration_approval.yaml` sign-off.

## 7. Scope call: `basics/melanin_only.py` is attenuation-only, not a full serum D(t) simulation

**Where:** `basics/melanin_only.py`'s `simulate_melanin()` +
`dynamic_attenuation()` — they produce the melanin trajectory and the two
synthesis-multiplier factors (constitutive x facultative), but do not run a
full serum-D simulation (no `A`, no BSA, no feedback `g(D)`, no decay).

**Finding:** This was a deliberate scope decision made during this session,
consistent with the file's "isolate melanin only" design philosophy (Part 1
of the file already omits `A`/BSA for the same reason). It is flagged here
because the advisor's original question ("isolate just melanin dynamics
model") and the instruction to "take that only the melanin part and put that
in here" could also be read as wanting a full D(t) run with only melanin
varying.

**Action:** Judgment call for Ahaan. If a full D(t) run restricted to the
melanin channel is wanted, it can be added as an additional function/section
in `basics/melanin_only.py` without touching the existing attenuation-only
content — this is additive, not a rewrite.

---

## Suggested order of attack

Items 1-4 are pure research/documentation-accuracy tasks (find or fail to
find a citation, then write down what was found) and are independent of each
other — safe to parallelize across subagents again if desired. Item 5 is a
larger optional writing task that should wait until 1-4 settle, since it
would otherwise need a second pass to incorporate their findings. Items 6-7
are quick judgment calls for Ahaan, not research tasks — worth resolving by
just asking him directly rather than spawning an agent for either.
