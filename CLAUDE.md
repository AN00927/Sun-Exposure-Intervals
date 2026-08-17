# CLAUDE.md

Project instructions. These override any default behaviour.

## Git — never commit, never push

**Do not run `git commit`. Do not run `git push`. Ever. Not even when asked.**

The user handles every commit and push in this repository, without exception.

If work is finished, stop after staging is *not* required either — leave the
working tree as it is and report what changed. Say what you did and let the user
commit it. If the user asks for a commit message, write one as text for them to
use; do not execute it.

This rule has no exceptions for:
- "just this once" / a one-off request in a previous turn
- follow-up work in the same session after an earlier commit was requested
- work that is verified, tested, and obviously correct
- "finish up and commit" style instructions

A single past authorization does not carry forward. Treat every commit as
requiring a fresh, explicit instruction — and even then, prefer to hand back the
message rather than run the command.

Read-only git is fine: `git status`, `git diff`, `git log`, `git show`.

## Where facts live — do not duplicate them

| You want | Read |
|---|---|
| Parameter values + citations | `vitd_model.py` module docstring — **the only canonical source** |
| Model description, evidence, problem statement, open work, assumptions, parameter history | `MODEL_AND_EVIDENCE.md` |
| Latest model output | `RESULTS.md` (generated) |

**Do not restate parameter values in prose here or in any other .md file.**
`K_ATTEN` was previously written out in nine places and `A` in five, including
two mutually contradictory copies inside `vitd_model.py` itself. That
duplication is exactly why the docs drifted out of sync with the code. Link
instead.

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

## Agent guidelines

- **DO** use the per-study `E_source` values from `SOURCE_CALIBRATION_TABLE`
  (`vitd_model.py`) for cross-source comparisons — never treat one `E_source`
  value as universal.
- **DO** specify dose framing (absolute SED vs fractional MED) in any schedule
  comparison — the two framings give materially different FST response
  patterns.
- **DO** reference `MODEL_AND_EVIDENCE.md` for evidence on
  any audit-related claim.
- **DO** flag parameter priors (`SC_max`, `alpha_SC`, `K_SC`) as uncalibrated
  when presenting absolute D predictions.
- **DO** recognize that `E_source=1.0` is the solar broadband reference — the
  model works for sun exposure out of the box. TL-01/sunbed sources are
  volatile; sun exposure is preferred for clinical conclusions.
- **Do NOT** pool NHANES 2001-2006 (`LBDVIDMS`) with 2007+ (`LBXVIDMS`) without
  harmonization code.
- **Do NOT** invent validation counts. Re-run
  `PYTHONIOENCODING=utf-8 python3 run_all.py` before quoting any count.
- **Do NOT** cite Tadokoro 2005 for tanning decay timescale — see `MODEL_AND_EVIDENCE.md`.
- **Do NOT** cite Matsuoka 1990's "51% attenuation" as a melanin effect.
  Baseline 25(OH)D differed between the tanned and untanned arms (92.1 vs
  56.2 nmol/L), so most of that blunting is D-feedback, not pigment. See
  `vitd_calibration.py` (`matsuoka1990_check`). Also do not confuse the tanning
  data with the local JAAD 1990 PDF (a different paper, on sunscreen/BSA).

Ablation summary (Bogh 2012): the 21-config ablation search (A00–A20) was
looking for a spectral weighting to close a gap that was actually a
**dose-unit error** — see `MODEL_AND_EVIDENCE.md`, 'Known issues', #1. Any per-config error
figures quoted from that old run (e.g. "A03, error 1.86") are **not
reproducible and must not be cited**. `scripts/bogh2012_ablation.py` is
retained as a sensitivity tool.
