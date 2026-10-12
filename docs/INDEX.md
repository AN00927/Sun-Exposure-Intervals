# Documentation index

Navigation root for every tracked Markdown document in this repository. One row
per document: the question it answers, and its status.

**This page links; it does not summarise and it restates no value.** Parameter
values live in the `vitd_model.py` module docstring, exposure-scenario
constants in `vitd_scenario.py`, and current outputs in `RESULTS.md`.

Status vocabulary:

- **canonical** — the authoritative statement of its subject; edit here, link elsewhere.
- **generated** — written by a script; never hand-edit.
- **archive** — a durable historical record; read, do not rewrite.
- **working-notes** — scoped to a past task; superseded in part, kept for provenance.

## Start here

| Document | Question it answers | Status |
|---|---|---|
| `README.md` | What is this project, how is it laid out, and how do I run it? | canonical |
| `RESULTS.md` | What does the model currently output? | generated (`run_all.py`) |
| `docs/SCRIPTS.md` | What is each script under `scripts/` for, and which lifecycle bucket is it in? | canonical |

## The model

| Document | Question it answers | Status |
|---|---|---|
| `docs/MELANIN_MODEL.md` | What are the melanin dynamics as implemented, where does each melanin parameter come from, and what does it not rest on? | canonical |
| `docs/SC_MODEL.md` | The same treatment for the stratum corneum sub-model. | canonical |
| `docs/SC_MELANIN_OPEN_ITEMS.md` | Which audit items came out of the `basics/` rewrite, and how was each resolved? | working-notes (all items closed 2026-09-13) |
| `basics/README.md` | What do the two Tier-1 single-mechanism reference models do? | canonical |
| `exploratory/README.md` | What was retired from the core pipeline on 2026-08-21, why, and how would it be un-retired? | canonical (sole record of the rationale) |

## Sub-studies, data and outputs

| Document | Question it answers | Status |
|---|---|---|
| `spacing_study/README.md` | Is daily exposure actually optimal for vitamin D synthesis, and how is that sub-study built? | canonical |
| `spacing_study/results/FINDINGS.md` | What did the spacing study find? | generated (`spacing_study/run_spacing_study.py`) |
| `data/spectra/README.md` | What is in `data/spectra/`, and what is the provenance of the action-spectrum file? | canonical |
| `data/spectra/tl01_conversion_factor.md` | Where does the TL-01 conversion factor come from? | canonical |
| `audit_outputs/provenance_reverification_2026-08-21.md` | Were the `data_provenance.yaml` records whose verification pointed at a never-committed file re-verified against the primary sources? | archive |
| `paper/README.md` | How is the manuscript skeleton built and what does it depend on? | canonical |
| `docs/SOURCES.md` | Which papers has this repo chased, and what did each one turn out to contain or not contain? | canonical |

## Retired documents (substance mapped to live locations below)

Six documents were deleted on 2026-09-19 with zero commits and are
**unrecoverable**: `MODEL_AND_EVIDENCE.md`, `REPO_SCOPE_AUDIT.md`,
`OPTIMIZATION_METHODOLOGY.md`, `MODEL_AND_EVIDENCE_RECONSTRUCTED.md`,
`ABSTRACT_RECONCILIATION.md`, `REPO_HYGIENE.md` (all formerly under `docs/`).
Live documents no longer point at them. Do not attempt to reconstruct
their contents.

Where the recoverable substance of those six documents actually went (recorded
here because the post-manuscript workstream programme's own tracking memos —
its plan, its handoffs, its decision register and the per-workstream memos —
were deleted on 2026-09-23 and are no longer on disk to trace it): the C1/C3/C7/C9 constraint numbering ->
`vitd_objectives.OBJECTIVE_HIERARCHY`; evidence-tier rationale ->
`vitd_validation.py`'s `CHECK_REGISTRY` `tier_note` fields; the k_damage
identifiability gap -> `docs/SC_MODEL.md` §2.4; the SC attenuation sign/edema
gap -> `docs/SC_MODEL.md` §2.6 and `vitd_model.py`'s `K_SC` entry; the
K_ATTEN Matsuoka-implied-range gap -> `docs/MELANIN_MODEL.md` §2.8; the
D-feedback interior-optimum artifact -> a live
`vitd_optimization.gross_synth_per_SED_diagnostic` check, recomputed every
run; out-of-sample-skill gaps -> per-check `tier_note` fields plus
`vitd_benchmark_webbkift.py`; the Farrar-2013-is-itself-a-matched-dose-trial
point -> `scripts/fst_gap_decomposition.py` S1b; geometry/BSA uncertainty ->
`vitd_scenario.LIMITATION_STATEMENT`; the D_max identifiability argument ->
`scripts/dmax_identifiability_gallagher2012.py`, now the primary record; what
was retired from the core pipeline and why -> `exploratory/README.md`.
Everything not listed above was either already duplicated at its point of
use, or is recorded as unrecoverable with the claim left standing and tagged
`[reference purged 2026-09-19]` at that site.

## Decision log 2026-09-21/23

The post-manuscript programme's decision register was deleted on 2026-09-23.
Its decisions live at their point of use: parameter and calibration decisions
(A-series) in `calibration_approval.yaml`, paper-acquisition decisions
(D-series) in `docs/SOURCES.md`, script buckets in `docs/SCRIPTS.md`, and
reporting changes in the code that emits them. The calls below have no other
natural home, or were weighed and declined, and are recorded so they are not
silently re-proposed.

- **B3 — structurally block FST VI absolute predictions from headline
  reporting?** Declined (2026-09-21). The model keeps answering its headline
  question for FST VI, with the FST VI point-of-use label
  (`vitd_optimization_reports.FST_VI_RESOLUTION_NOTE`) instead of a refusal.
- **B5 — README absolute-minutes band.** Recomputed against the current
  `vitd_scenario.py` constants (2026-09-23); regenerated every run in
  `RESULTS.md`'s "Scope and reference conditions" section. `vitd_scenario.py`
  holds UV index and geometry fixed, so the band covers exposed BSA only and
  is a floor. Adding season/geometry ranges to `vitd_scenario.py` would be a
  separate decision.
- **C3 — `tables/debug/young2020_check.csv` owner.** Wired into `run_all.py`
  (section 14e) 2026-09-23.
- **C5(i)** corrected in place in `calibration_approval.yaml` (2026-09-23).
  **C5(ii)** done 2026-09-23: the four FST V-VI field-dosimetry sources are
  in `paper/references.bib` (three new in section (g), plus the existing
  `fajuyigbe2018photochem`), read but not cited. `calibration_approval.yaml`'s
  "no field dosimetry" claims are annotated in place to say "in this
  repository".
- **C8 — `tables/debug/abstract_recon_*.csv`.** Not regenerated: the producing
  scripts (`abstract_reconciliation_check.py`, `abstract_reconciliation_seeds.py`)
  were never tracked and no copy exists on disk (content search, 2026-09-23).
  Nothing in the repo reads these four tables; they are stale against the
  2026-09-22/23 recalibration. The Bogh 2012 ablation table was regenerated.
- **C10** — `docs/SC_MODEL.md` / `docs/MELANIN_MODEL.md` carry marked
  historical values for provenance; current values live in `vitd_model.py`.
- **C11** — `scripts/abstract_risk_decomposition.py` deleted as ONE-TIME
  (recoverable: `git show b98cd02:scripts/abstract_risk_decomposition.py`).
- **E1 — chase Nkogatse 2018's 24% geometry factor.** Yes, but blocked: full
  text unobtainable; fallback lead in `docs/SOURCES.md`.
- **E2 — natural-sunlight per-session dose-response series.** Yes; none found
  yet. The four curvature sources behind the spacing result remain lamp-only.
- **E4** — the long parallel-grid determinism test stays in the default suite.
- **F — known, deliberately unfixed items** accepted as they stand
  (2026-09-23): the unrecognised `dose_framing="matched_cumulative_SED"` label
  warning in `vitd_analyses.py` (cosmetic, feeds no computation); the
  non-monotonic `steady_state_D` near its 28-day averaging window (not a
  defect; touches no printed number); nothing in `exploratory/` un-retired;
  the action-spectrum refinement deferred until `E_source` is zenith-angle
  dependent.
- **A5 BMI-as-predictor** — prepared, not applied;
  `BMI_predictor_proposal_2026_09_22` in `calibration_approval.yaml` awaits a
  decision.
