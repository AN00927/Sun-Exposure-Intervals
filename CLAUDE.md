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
  **`run_all.py` takes ~10 minutes** (dominated by the differential-evolution
  optimisations in sections 13b/13c). Do **not** run it just to "update
  everything", refresh results, or check mid-task — it is expensive and should
  only be run when genuinely needed (e.g. before quoting final numbers, after a
  calibration change, or when the user explicitly asks).

## Retired modules — `exploratory/`

On 2026-08-21 the location-based dosimetry chain (solar position -> ambient UV
index -> posture/site geometry -> exposed BSA) and several other modules were
moved to `exploratory/`: `vitd_genetics.py`, `vitd_ambient.py`,
`vitd_behaviour.py`, `nhanes_preprocessing.py`, `vitd_solar.py`,
`vitd_exposure.py`. They still import and their own tests
(`tests/exploratory/`) still pass. What changed is that **nothing they produce
reaches any headline number** — `run_all.py` no longer imports them. Rationale:
`docs/REPO_SCOPE_AUDIT.md`. What was retired, why, and the un-retirement
procedure: `exploratory/README.md`.

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
