# Open items from the `basics/melanin_only.py` + `basics/sc_only.py` rewrite

Parameter values below follow the code at full precision; historical values
are marked superseded inline.

**Status as of 2026-09-13: All 7 items resolved.** This document was
re-audited and fully closed on 2026-09-13; each entry below states what
resolved it and where, so nobody re-researches a settled question.

## Context

`basics/melanin_only.py` and `basics/sc_only.py` isolate the mechanisms
`vitd_model.py` runs (melanin: constitutive + facultative dynamics, SC
ablated; SC: the two-process `SC_build - SC_damage` structure). `basics/README.md` was rewritten to match, source-first, with a
full derivation trail for every parameter. That work surfaced seven findings,
recorded below, that were flagged rather than resolved at the time.

**Standing constraint:** none of these items licenses changing a calibrated
parameter value in `vitd_model.py` on its own; treat such a change as a
calibration decision with its source recorded. This still applies to the
one remaining open item (5) if it is ever acted on.

---

## 1. RESOLVED — `k_repair_SC`'s 7-day half-life

**Where:** `vitd_model.py:1578-1591` (`k_repair_SC: float = np.log(2) / 7.0`).

**What resolved it:** The comment now carries an explicit `[PRIOR, UNCITED]`
tag and documents the search that failed to find a source: no measured
barrier-repair rate constant exists in any source held in this repo (checked
2026-08-25) nor in a 100-paper broad literature search (Consensus,
2026-08-26). The closest anchors are named and explicitly rejected as fits:
Abe & Mayuzumi 1979 (TEWL returns to control by 3 weeks post-UVB, but no
intermediate timepoints to fit an exponential) and mouse-UVB data (Haratake
1997; Jiang 2007; Holleran 1997, a biphasic insult-repair process, not a
first-order decay). The 7-day half-life is labeled structural (chosen so the
T1 dip recovers by T2), and the file notes `(k_damage, k_repair_SC)` sit on a
ridge, not independently identifiable from the T0/T1/T2 group medians
(`basics/sc_only.py` sec. 2; `scripts/fit_sc_deboer2025.py`).

This is exactly the documentation fix item 1 asked for (label it as an
unsourced structural prior rather than implying it is measured) — it was not
resolved by *finding* a source. Do not confuse this with item 2's citation:
Weinstein & Van Scott 1965 sources `k_decay_SC` (a different parameter), not
`k_repair_SC`.

## 2. RESOLVED — `k_decay_SC`'s 21-28 day range vs. Pearse 1987

**Where:** `vitd_model.py:1599-1714`.

**What resolved it:** The comment block now names Weinstein & Van Scott 1965
(*J Invest Dermatol* 45:257-262, PMID 5837896, via Rothberg 1961) as the
actual source for the SC-residence leg (13-14 d) of whole-epidermis transit,
and explicitly separates Pearse 1987 out: "Pearse 1987 (P23) measured SC
thickness after 6 weeks of 3x/week UVB — a level, not a rate — and is cited
for the thickening phenomenon only, never for this number" (`vitd_model.py`
around line 1611). Post-1972 turnover literature is folded in by name —
Halprin 1972, Bergstresser & Taylor 1977, Iizuka 1994, Datta 2019 (line
1615-1616) — and an `[EVIDENCE NOTE 2026-09-06]` (lines 1622-1714) documents a
full read of two additional dansyl-chloride SC-turnover papers, Jansen,
Hojyo-Tomoko & Kligman 1974 and Grove & Kligman 1983, with their numbers,
disagreements with each other (~47%), and caveats (all-Caucasian, no FST
typing — a light-skin-extrapolation flag per the dark-skin-priority
directive) recorded in full. 21-28 d is retained as an interior, uncalibrated
choice — no source directly measures SC-hyperplasia regression — not
presented as a measured half-life.

**Loose end resolved 2026-09-13:**
`scripts/parameter_optima.py:109-114` was verified to carry the correct
SC-residence leg justification ("Weinstein & Van Scott 1965 via Rothberg 1961
SC-RESIDENCE leg alone, 13-14 d (ln(2)/T conversion) -- NOT the 26-27 d
COMBINED whole-epidermis figure") and Jansen 1974 corroboration. The tracking
tag at `vitd_model.py:1708` has been updated to reflect this resolution.

## 3. RESOLVED (both locations) — Miyamura 2011 non-significance caveat

**Where:**
- `vitd_model.py:1526-1531` (caveat dated 2026-08-25, "Verified against the
  PDF twice independently").
- `docs/MELANIN_MODEL.md:447-459`, section 2.7a, "What it does NOT
  establish" bullet.
- `basics/README.md:207-216, 279` (the original source of the wording).

**What resolved it:** All three now state Miyamura's own finding plainly —
"20.7 +/- 3.6 NS"; text: SC thickness "did not change significantly in any of
the specimens", "increasing trend" — and all three use the same framing: NS
is a power statement, not evidence of absence, so the result corroborates
trend and magnitude, not an independently significant replication.

This closes item 3 completely. The original handoff asked specifically
whether `docs/MELANIN_MODEL.md` still lacked this caveat (it was written
before the 2026-08-25 pass); on inspection it does not — the caveat is
present there too, worded consistently with `basics/README.md`. No action
needed.

## 4. RESOLVED — de Boer 2025 week-5 dip p-values

**Where:** `vitd_model.py:1721-1723` (the `k_damage` comment entry):
"Significance (Supp S1, double-verified 2026-08-25): T1-vs-T0 dip p=.016
pooled, NS within stratum (I-III .170, IV-VI .134); T2-vs-T0 overshoot
p<=.002 in ALL groups. Dip is pooled-real; stratum-level evidence weaker."

**What resolved it:** The values were verified for a third time, directly
from the primary source PDF (`Research_Papers/Effect of Repeated Low-Dose UVR Exposure - de boer et al..pdf`,
Supplementary Table S1, page 12) on 2026-09-13:
- T1 vs T0 (dip): all FPs p = .016; FP I–III p = .170; FP IV–VI p = .134.
- T2 vs T0 (overshoot): all FPs p < .001; FP I–III p = .002; FP IV–VI p < .001.
- T2 vs T1: all FPs p < .0001; FP I–III p < .0001; FP IV–VI p < .0001.

The values in `vitd_model.py:1721-1723`, `basics/sc_only.py`, and `basics/README.md`
match the printed source table byte-for-byte. Item 4 is fully closed.

## 5. RESOLVED — a dedicated `docs/SC_MODEL.md`

**Where:** `docs/SC_MODEL.md` (authored 2026-09-13).

**What resolved it:** Written to mirror `docs/MELANIN_MODEL.md`'s structure
(stated ODE vs pulse implementation, detailed parameter-by-parameter
provenance, exact arithmetic mapping published data to constants, mandatory
limitations register, and scheduling sensitivity context). Covers de Boer 2025,
Miyamura 2011, Pearse 1987, Weinstein & Van Scott 1965, Jansen 1974, Grove &
Kligman 1983, and Bruls 1984. Item 5 is fully closed.

## 6. RESOLVED — `M_ceiling()` not clipped at 1.0: premise was wrong, no bug

**Where:** `vitd_model.py:1780-1793` (`ModelParams.M_ceiling()`).

**What resolved it:** Confirmed by inspection — `M_ceiling()` returns
`M_baseline(fst) + M_TAN_HEADROOM` with no clip, and the docstring says so
explicitly: "NOT capped at 1.0: 1.0 is the *constitutive* FST VI reference on
the Datta 2019 reflectance scale, not a physical ceiling... Capping at 1.0
silently makes tanning a no-op for FST VI." A grep for other clips near
melanin found exactly one, and it is a different quantity: `M_baseline_by_FST`
itself is clipped to [0, 1] at *table-construction* time (e.g.
`vitd_model.py:562,618,1739`, "VI 1.0427 -> clipped 1.00" for the
*constitutive* baseline), which is unrelated to `M_ceiling()`'s *facultative*
headroom. No other part of the pipeline expects `M_ceiling()` itself to be
capped. Closed as "premise was wrong, no bug" — no clip was added.

## 7. RESOLVED — `basics/melanin_only.py` now has a full serum-D run

**Where:** `basics/melanin_only.py:336-503` and `:949-953`, "PART 3 — full
serum-D simulation with melanin varying, SC ablated"
(`simulate_D_melanin_only()`, plus `_cross_check_vs_vitd_model_full_D()` for
consistency against `vitd_model.simulate()`).

**What resolved it:** Commit `1d9898847ff24ea4f20d49c8b5c601971b9ac3fd`
("Add full serum-D simulation with melanin dynamics and SC ablated (Part
3)", 2026-08-26) added this section — 178 lines to
`basics/melanin_only.py` plus a supporting `calibration_approval.yaml` entry
and small edits to `vitd_model.py` and `scripts/parameter_optima.py`. The
original attenuation-only Parts 1-2 are unchanged; Part 3 was added
alongside them, as the handoff's suggested resolution described.

---

## Final status

All 7 items are resolved and fully documented as of 2026-09-13. No open audit
items remain from the `melanin_only.py` / `sc_only.py` rewrite. Both the
melanin and stratum corneum subsystems now have matching, dedicated
first-principles documentation files (`docs/MELANIN_MODEL.md` and
`docs/SC_MODEL.md`).
