# CLAUDE.md

Project instructions. These override any default behaviour.

## Git — commit only when asked, never push

**Do not run `git commit` unless the user asks for that commit. Never run
`git push` unless the user asks for that push.**

When the user does ask, commit. Otherwise leave the working tree as it is and
report what changed.

A past authorization does not carry forward. Each commit needs its own fresh,
explicit instruction — "you committed last time" is not permission to commit
again, and neither is work being verified, tested, and obviously correct. When
in doubt, hand back the message as text instead of running it.

History rewrites (`filter-repo`, `filter-branch`, BFG) and `push --force` are a
separate class: ask every time, and take a full **filesystem copy** first, not a
`git clone` — much of what makes this repo recoverable is gitignored and a clone
skips it. A rewrite here has already destroyed uncommitted work once.

Read-only git is fine: `git status`, `git diff`, `git log`, `git show`.

## Precision — carry the full value, never a tidied one

**When a number comes out of a fit, a derivation, or a source table, carry it
at full precision. Do not round it for readability, and never present a rounded
figure as though it were the computed one.**

Round only where rounding is the actual decision, and when you do, say so and
give the unrounded value next to it. `M_TAN_HEADROOM = 0.238` is a deliberate
choice recorded against its exact origin (max observed dM = 0.237724); writing
"~0.23" in prose and then setting 0.23 in code would silently discard a real
digit and make the parameter untraceable to its source.

This applies to reporting as well as to code. Quoting "about 24%" when the
measurement is 24.6988% throws away information the reader needs to check you.
Report `24.7%` and keep the full value in the derivation. If a value is
genuinely uncertain, give the range — `40-70 d` — rather than rounding to a
single tidy midpoint, which fakes precision in the opposite direction.

**This is not a licence to print float64 repr.** Precision means *don't discard
digits that carry information*; it does not mean pasting
`+0.05323821715712285` into a table. A simulation output on a model whose
absolute predictions rest on uncalibrated priors carries maybe three
meaningful digits, and seventeen of them is noise wearing a lab coat — it
misleads the reader about how much the model actually knows. The test is
whether a digit is *load-bearing*: a calibration constant, a source
measurement, or anything another calculation reads back gets its full value;
a reported result gets the emitting script's own print format, with the full
value left in the CSV or the code. When in doubt, ask what a reader would do
with the digit.

Sibling rule to the one below: values live in the code at full precision, and
prose links to them rather than restating them.

## Where facts live — do not duplicate them

| You want | Read |
|---|---|
| Parameter values + citations | `vitd_model.py` module docstring — **the only canonical source** |
| Exposure-scenario constants (ambient UV index, geometry factor, exposed BSA) | `vitd_scenario.py` — **the only canonical source** |
| Latest model output & results | `RESULTS.md` (generated) |
| Project overview & usage | `README.md` |

**Do not restate parameter values in prose here or in any other .md file.**
`K_ATTEN` was previously written out in nine places and `A` in five, including
two mutually contradictory copies inside `vitd_model.py` itself. That
duplication is exactly why the docs drifted out of sync with the code. Link
instead. The same rule applies to `vitd_scenario.py`'s exposure-scenario
constants (`UV_INDEX`, `GEOMETRY_FACTOR`, `EXPOSED_BSA`, `SED_PER_MINUTE`) —
quote `vitd_scenario.LIMITATION_STATEMENT` or `vitd_scenario.describe()`
instead of retyping the numbers.

**EXEMPTION:** `docs/SC_MODEL.md` and `docs/MELANIN_MODEL.md` carry historical
parameter values (marked with `[SUPERSEDED ...]` or `[RE-ANCHOR ...]` inline)
for calibration-change audit and provenance tracking. These files document the
derivation context when a parameter changed, so the reader can understand why
the current value differs from its predecessor. Use `[read from code ...]`
markers and explicit section tags to distinguish current values (which link to
`vitd_model.py`) from historical ones (which are labelled as superseded). This
exemption applies to these two files only; all other `.md` files must follow
the "do not restate" rule.

- `calibration_approval.yaml` gates **calibration** (changing a scientific
  parameter to fit data) — that needs explicit user approval. **Bug fixes**
  (code that does not implement the documented model: unit errors, state-update
  errors, inverted signs) are allowed without approval.
- Before listing a paper as "missing from the repo", check `Research_Papers/`
  (PDFs, **gitignored** — present on the user's disk, absent from a fresh
  clone) and `Papers_and_Datasets/` (datasets, abstracts, supplementary).
  Filenames in both are unreliable — one file named "(NO PAPER FOUND)"
  contained the complete dataset it claimed was absent.
- Run the pipeline with `PYTHONIOENCODING=utf-8 python3 run_all.py`. The local
  `.venv` lacks numpy; `python3` is the interpreter with the dependencies.
  **`run_all.py` takes 11 min 29 s** (measured 10:07:50-10:19:19 EDT
  2026-09-21 on a **16-logical-core** machine, at HEAD `6a1585f` + working
  tree, `--baseline-source preset`, `--run-macro-scripts`. Historical, both
  **pre-parallelisation** and not to be quoted as current: 21 min 55 s
  (16:03:12-16:25:07 EDT 2026-09-06, on the re-anchored A = 4.125 parameter
  set) and 25 min 07 s (23:02:17-23:27:24 EDT 2026-08-23).)
  **There is no longer a single bottleneck.** The 24-cell (FST x
  session-budget) grid -- `analysis_joint_schedule` / `_run_joint_grid` /
  `_joint_grid_cell`, `vitd_optimization.py:947-1033` (there is no function
  named `optimise_joint_schedule_grid`; that name does not appear anywhere
  else in the repo) -- **is already parallelised** via a
  `ProcessPoolExecutor` over the independent, per-cell-seeded grid cells,
  landed in commit `37f9a94` (2026-09-06 18:09). That took it from ~13 min
  of the 2026-09-06 run to **2 min 04 s**, and the remaining time is now
  spread across five comparable serial blocks rather than concentrated in
  one. Measured from artifact write times on the 2026-09-21 run:
  365-day front-load taper 2 min 12 s; `analysis_variable_dose` 2 min 07 s;
  the joint grid 2 min 04 s; `analysis_tri_objective` 1 min 54 s;
  risk-aware scheduling 1 min 41 s -- together ~92% of wall-clock, with no
  one of them above 20%. Parallelising any single one of them now buys at
  most ~2 min. `differential_evolution(workers=-1)` is a SEPARATE thing
  and is still NOT the speed-up, because scipy forces `updating='deferred'`
  when `workers != 1`, which changes the search trajectory and so moves every
  seeded result and pinned test value. Do **not** run it just to "update
  everything", refresh results, or check mid-task — it is expensive and should
  only be run when genuinely needed (e.g. before quoting final numbers, after a
  calibration change, or when the user explicitly asks).

## Script lifecycle — four buckets, decide by evidence

Every script gets exactly one bucket, decided by checking imports/tests/docs,
never by filename or gut feel:

- **PIPELINE** — imported by `run_all.py`, a module it imports, or `tests/`.
  Never gitignore, never delete. Cautionary example: `.gitignore` once
  gitignored two scripts `run_all.py` imports via `_import_script_module` — a
  fresh clone could not run the pipeline. Grep for a real `import` statement
  (`_import_script_module("name")`, `import name`, or `from scripts.name
  import ...`) before ignoring anything under `scripts/` — not for the
  script's own name as a bare string. A 2026-09-20 audit found the name-grep
  unreliable: five scripts turned up in `tests/` only inside comments, with
  `fit_sc_deboer2025` the sole genuine test import at the time.
  `tests/test_structural_invariants.py` enforces this by parsing real imports
  with `ast` rather than grepping for names.
- **RECURRING** — a sensitivity/diagnostic tool meant to be re-run
  occasionally (archetype: `scripts/bogh2012_ablation.py`, "retained as a
  sensitivity tool"). Keep tracked, not gitignored.
- **FEW-TIME** — a fitting/extraction/checking script whose committed
  artefact it might regenerate. Gitignore it (don't delete — the artefact's
  provenance depends on the script existing on disk).
- **ONE-TIME** — answered its question; the answer already lives in a doc or
  table. Delete — but only after confirming the file is recoverable
  (`git show HEAD:<path>` on a tracked, unmodified file) or is genuinely
  worthless. Deletion is irreversible here; a rewrite has already destroyed
  uncommitted work once.

Can't tell which bucket from the evidence? Ask — don't guess.

## Retired modules — `exploratory/`

On 2026-08-21 the location-based dosimetry chain (solar position -> ambient UV
index -> posture/site geometry -> exposed BSA) and several other modules were
moved to `exploratory/`: `vitd_genetics.py`, `vitd_ambient.py`,
`vitd_behaviour.py`, `nhanes_preprocessing.py`, `vitd_solar.py`,
`vitd_exposure.py`. They still import and their own tests
(`tests/exploratory/`) still pass. What changed is that **nothing they produce
reaches any headline number** — `run_all.py` no longer imports them. What was
retired, why, and the un-retirement procedure: `exploratory/README.md`, which
is now the only record of the rationale (the separate scope audit was purged
2026-09-19 with zero commits and is unrecoverable).

**Rule: nothing in the core pipeline may import from `exploratory/`.** The core
pipeline's frozen replacement for the location chain is `vitd_scenario.py` (see
the table above). `tests/exploratory/conftest.py` puts `exploratory/` on
`sys.path` for that directory's own tests only — a `sys.path` leak between
conftest files has previously masked a broken import, so this is a real
invariant, not a style preference. Check it with:

```bash
python3 -m pytest tests/ -q --ignore=tests/exploratory
```

That command must pass with `exploratory/` absent from `sys.path` entirely.

## Agent guidelines

- **DO** use the per-study `E_source` values from `SOURCE_CALIBRATION_TABLE`
  (`vitd_model.py`) for cross-source comparisons — never treat one `E_source`
  value as universal.
- **DO** specify dose framing (absolute SED vs fractional MED) in any schedule
  comparison — the two framings give materially different FST response
  patterns.
- **DO** flag parameter priors (`SC_max`, `alpha_SC`, `K_SC`) as uncalibrated
  when presenting absolute D predictions.
- **DO** recognize that `E_source=1.0` is the solar broadband reference — the
  model works for sun exposure out of the box. TL-01/sunbed sources are
  volatile; sun exposure is preferred for clinical conclusions.
- **DO** flag it to the user when a source figure/image is hard to read (low
  resolution, small text, ambiguous gridlines) instead of silently digitizing
  a best guess or writing the source off as unusable. A paper judged "not
  useful" because its evidence sits in an image you struggled to read is a
  false negative — say so explicitly so the user can decide whether to get a
  cleaner copy or transcribe it by hand. This has already cost real signal
  once (Weihs et al. 2013, Fig. 6). The bar for flagging is low: even mild
  uncertainty about legibility is worth a one-line note — flag it and keep
  going, don't stop and wait for a reply.
- **Do NOT** pool NHANES 2001-2006 (`LBDVIDMS`) with 2007+ (`LBXVIDMS`) without
  harmonization code.
- **DO** leverage the NHANES macro diagnostic suite in `scripts/`:
  * `scripts/nhanes_macro_validation.py` (`RECURRING`): 52-week population
    steady-state macro-validation against unsupplemented NHANES cohorts.
  * `scripts/nhanes_dietary_partitioning.py` (`RECURRING`): Unsupplemented
    baseline floor ($D_{\text{floor}}$) and solar vs dietary contribution analysis.
  * `scripts/nhanes_weight_dilution_check.py` (`RECURRING`): Cross-sectional
    body-weight dilution scaling check ($70/\text{weight\_kg}$ vs BMI).
  * `tables/nhanes_baseline_priors.csv`: Survey-weighted baseline $D_0$ quantiles
    by FST, season, and diet (switchable via `--baseline-source nhanes`).
- **DO** note: When model calibration, baseline assumptions, or scenario constants
  are updated, running the NHANES macro scripts is recommended to verify
  population-level plausibility. **Always ask the user if they would like to run
  these macro scripts right after completing core model modifications.**
- **Do NOT** invent validation counts. Re-run
  `PYTHONIOENCODING=utf-8 python3 run_all.py` before quoting any count.
- **Do NOT** cite Tadokoro 2005 for tanning decay timescale.
- **Do NOT** cite Matsuoka 1990's "51% attenuation" as a melanin effect.
  Baseline 25(OH)D differed between the tanned and untanned arms (92.1 vs
  56.2 nmol/L), so most of that blunting is D-feedback, not pigment. See
  `vitd_calibration.py` (`matsuoka1990_check`). Also do not confuse the tanning
  data with the local JAAD 1990 PDF (a different paper, on sunscreen/BSA).

Ablation summary (Bogh 2012): the 21-config ablation search (A00–A20) was
looking for a spectral weighting to close a gap that was actually a
**dose-unit error**. Any per-config error figures quoted from that old run
(e.g. "A03, error 1.86") are **not reproducible and must not be cited**.
`scripts/bogh2012_ablation.py` is retained as a sensitivity tool.
