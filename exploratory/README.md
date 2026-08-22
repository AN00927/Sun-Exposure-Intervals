# `exploratory/` — retired from the core pipeline

**Retired 2026-08-21.** These modules still exist, still import, and their tests
still run and pass. What changed is that **nothing here influences any headline
number.** `run_all.py` no longer imports them, and no value they produce reaches
`RESULTS.md`.

Retirement is not a judgement that the work is wrong. Read the module docstrings
— several contain results that are worth keeping and would be a real loss to
delete (the Kallioglu 2024 identity proof in `vitd_ambient.py`, the
behaviour-versus-optics split argument in `vitd_behaviour.py`, the two NOAA
coefficient errata caught in `vitd_solar.py`). They are retired because the
*claims they support* are not proportionate to the evidence behind them, and
because the repository was carrying far more scope than it could defend.

Rationale and the full audit: [`../docs/REPO_SCOPE_AUDIT.md`](../docs/REPO_SCOPE_AUDIT.md).

## What is here and why it left

| Module | Why retired |
|---|---|
| `vitd_solar.py` | Correct — the best-sourced file in the repo (NOAA solar position, two source errata caught and documented). Retired not for quality but because the *chain* it starts is out of scope. |
| `vitd_ambient.py` | Makes **live outbound HTTP calls** to Open-Meteo at run time. A results pipeline with a network dependency is not reproducible: rerun it in six months and the numbers change with no record of why; rerun it offline and it fails. |
| `vitd_exposure.py` | The site-set → visibility-percent → exposed-BSA join. Rests on a secondary reproduction of the Lund & Browder (1944) chart (the primary is not held in this repo) and on roughly six stacked `[CHOICE]` mapping decisions. No *product* of this chain has ever been validated against measured personal dosimetry. |
| `vitd_behaviour.py` | Well argued, but **already inert**: the scheduling-mode multiplier is 1.0 for every skin type, so removing it from the pipeline changes nothing numerically. |
| `vitd_genetics.py` | Self-reports `_confidence: "LOW N=22 95% CI width unknown, additive/multiplicative form untested at low D"`. Applying a 1.51× multiplier to `A` on that basis adds a factor of two of spread to every prediction it touches in exchange for nothing testable. |
| `nhanes_preprocessing.py` | Its own validation check is marked `EXPLORATORY_ONLY`, and it was already wrapped in a bare `try/except` in `run_all.py` — i.e. optional in practice before it was optional by design. |

## What replaced the location chain

A single frozen reference scenario: see `vitd_scenario.py` in the repository
root. Fixed ambient UV index, fixed geometry factor, fixed exposed BSA, held
constant on every exposure day. The values were chosen so that **no
recalibration was required** — they are the reference conditions the model was
already calibrated at.

That is a real loss of capability, stated as a limitation rather than hidden:
the model can no longer answer "how long should I stand outside in Manchester on
15 March". It answers "how long under reference conditions", which is what its
evidence base actually supports.

## Running these modules

They are not on the default import path. Their tests carry a `conftest.py` that
adds this directory to `sys.path`:

```
PYTHONIOENCODING=utf-8 python3 -m pytest tests/exploratory/ -q
```

To use one interactively, add this directory to `sys.path` first. Nothing in the
core pipeline should ever do that — if you find yourself wanting to, the module
should be un-retired deliberately, with the evidence that justifies it.

## Un-retiring something

Move the file back to the repository root, restore its import in `run_all.py`,
move its tests back to `tests/`, and record in `calibration_approval.yaml` what
new evidence justifies the promotion. Do not un-retire a module because a
result would look better with it.
