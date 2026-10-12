# `scripts/` manifest — lifecycle bucket per script

Buckets are defined in `CLAUDE.md` ("Script lifecycle — four buckets, decide by
evidence"): **PIPELINE** / **RECURRING** / **FEW-TIME** / **ONE-TIME**.

Classification here is by **evidence only**, never by filename. PIPELINE
membership is established with `ast`-parsed `Import`/`ImportFrom` nodes and
`_import_script_module("name")` calls with a string-literal argument — a bare
name-grep is unreliable (a 2026-09-20 audit found five scripts appearing in
`tests/` only inside comments). Non-PIPELINE buckets cite the script's own
docstring declaration, a `README.md`/`docs/` usage entry, or the tracked
artefact the script regenerates.

`tests/test_structural_invariants.py` enforces two invariants against this
file: no PIPELINE-listed script may be matched by `.gitignore`, and this table
must list every file in `scripts/`.

Restates no parameter values — see `CLAUDE.md` "Where facts live".

| script | bucket | evidence | artefact regenerated |
|---|---|---|---|
| `__init__.py` | PIPELINE | package marker; without it `from scripts.X import ...` in `tests/test_physiology.py:614` cannot resolve | — |
| `bogh2012_ablation.py` | RECURRING | no importer found; docstring line 9 "Retained as a sensitivity tool"; named as the RECURRING archetype in `CLAUDE.md` | `tables/debug/bogh2012_ablation.csv` (tracked) — **regenerated 2026-09-23** under the 2026-09-22/23 recalibration (decision C8); structure unchanged (21 configs A00-A20, same 20 OK / 1 BLOCKED_UNVERIFIED_BSA_EQUATION pattern). Per `CLAUDE.md`, per-config error figures are not the historically-cited ones (e.g. "A03, error 1.86") and must not be quoted as fixed reference values; they move with every calibration change |
| `bsa_exp_recalibration.py` | FEW-TIME | no importer found; docstring line 7 self-declares "Bucket: FEW-TIME"; currently matched by `.gitignore` | `BSA_EXP` derivation recorded in `calibration_approval.yaml` |
| `dmax_identifiability_gallagher2012.py` | RECURRING | no importer found; docstring says the script "is now the repository's primary record of the argument" and it remains the live D_max diagnostic | none tracked |
| `extract_young2020.py` | FEW-TIME | no importer found; extraction script writing a tracked dataset | `Papers_and_Datasets/other/Trial_Young2020_PNAS_SerumD_UVR.csv` (tracked) |
| `feasibility_map_kconst.py` | RECURRING | no importer found; parameterized consequence map referenced in `calibration_approval.yaml` and multiple workstreams, intended for reruns after model or calibration changes | none tracked |
| `fit_alpha_from_trials.py` | FEW-TIME | no importer found; fitting script; no module docstring | `alpha_fitted_results.csv` (**not present on disk**) |
| `fit_katten_young2020.py` | FEW-TIME | no importer found; fitting script, documented as a fitting run at `README.md:232`; result lives as a parameter in `vitd_model.py` | parameter value in `vitd_model.py` |
| `fit_melanin_kinetics.py` | FEW-TIME | imported at `basics/melanin_only.py:138` — but `basics/` is not `run_all.py`, a module it imports, or `tests/`, so this is **not** PIPELINE under `CLAUDE.md`'s definition; fitting script cited in `calibration_approval.yaml` and `docs/MELANIN_MODEL.md` | melanin kinetic parameters in `vitd_model.py` |
| `fit_sc_deboer2025.py` | PIPELINE | `tests/test_physiology.py:614`, `tests/test_physiology.py:623`; also `scripts/plot_sc_deboer2025.py:27` | — |
| `fixed_budget_robustness.py` | PIPELINE | `run_all.py:1677` (`_import_script_module`). Added 2026-09-30 for mock-review roadmap items R6/R7 | `tables/low_dose_regime.csv`, `tables/log_scale_sweep.csv`, `tables/fixed_budget_mc.csv`, `tables/fixed_budget_mc_draws.csv`, `figs/fixed_budget_robustness.png` |
| `frontload_fixed_budget.py` | PIPELINE | `run_all.py:717` (`_import_script_module`) | — |
| `fst_gap_decomposition.py` | RECURRING | no importer found; cited in `calibration_approval.yaml` and six `docs/` files; docstring says "DIAGNOSIS ONLY. Changes no parameter, writes no parameter" and it remains the live decomposition diagnostic | none tracked |
| `generate_nhanes_priors_table.py` | FEW-TIME | no importer found; generator of a tracked table named in `CLAUDE.md` | `tables/nhanes_baseline_priors.csv` (tracked) |
| `joint_schedule_freesolve.py` | RECURRING | no importer found; validity check for the joint optimiser; user approved 2026-09-21 as RECURRING to be rerun on optimizer/model changes | none tracked |
| `kconst_recalibration.py` | FEW-TIME | no importer found; docstring: "DERIVATION for the K_const recalibration approved on 2026-08-17 (`calibration_approval.yaml`)" — same shape as the self-declared FEW-TIME `bsa_exp_recalibration.py` | `K_const` derivation recorded in `calibration_approval.yaml` |
| `mckenzie2012_check.py` | FEW-TIME | no importer found; docstring: "CHECK, not calibration"; regenerates a tracked table | `tables/mckenzie2012_check.csv` (tracked) |
| `melanin_parameters_report.py` | RECURRING | no importer found; docstring: "COMPANION SCRIPT for `docs/MELANIN_MODEL.md` ... reads the LIVE melanin-related values out of `vitd_model.ModelParams()`" — re-run whenever those values change; cited in `basics/README.md` and `docs/MELANIN_MODEL.md` | none tracked (prints) |
| `melanin_sc_coupling.py` | RECURRING | no importer found; documented as a re-runnable analysis at `README.md:238` | none tracked |
| `metric_redundancy.py` | RECURRING | no importer found; documented as a re-runnable analysis at `README.md:237` | none tracked |
| `nhanes_dietary_partitioning.py` | PIPELINE | `run_all.py:602` (`_import_script_module`). Its own docstring line 9 says "Lifecycle Bucket: RECURRING"; the real import makes PIPELINE the binding bucket — it is also a diagnostic meant to be re-run | `tables/nhanes_dietary_partitioning.csv` (tracked) |
| `nhanes_macro_validation.py` | PIPELINE | `run_all.py:601` (`_import_script_module`). Docstring line 9 says "Lifecycle Bucket: RECURRING"; the real import makes PIPELINE binding | `tables/nhanes_macro_validation.csv`, `figs/nhanes_macro_validation.png` |
| `nhanes_weight_dilution_check.py` | PIPELINE | `run_all.py:603` (`_import_script_module`); `CLAUDE.md` also lists it as a RECURRING diagnostic — the real import makes PIPELINE binding | `tables/nhanes_weight_dilution_check.csv` (tracked) |
| `nhanes_weighted_check.py` | RECURRING | no importer found; documented as a re-runnable analysis at `README.md:233` | `tables/nhanes_weighted_check.csv` (tracked) |
| `parameter_optima.py` | RECURRING | no importer found; documented as a re-runnable analysis at `README.md:236`; cited in ten `docs/`/`paper/` files | `tables/parameter_optima.csv`, `figs/parameter_optima_grid.png`, `figs/parameter_optima_joint.png` (all tracked) |
| `plot_sc_deboer2025.py` | FEW-TIME | no importer found; imports `scripts.fit_sc_deboer2025` at line 27; plotting script for a tracked figure | `figs/sc_deboer2025_fit.png` (tracked) |
| `risk_aware_scheduling.py` | PIPELINE | `run_all.py:1648` (`_import_script_module`) | `tables/risk_aware_schedule_by_fst.csv`, `figs/deficiency_risk_frontier.png` |
| `risk_law_sensitivity.py` | PIPELINE | `run_all.py:1674` (`_import_script_module`). Added 2026-09-30 for mock-review roadmap item R2 | `tables/risk_law_sensitivity.csv` |
| `single_session_response.py` | PIPELINE | `run_all.py:714`, `run_all.py:1050` (`_import_script_module`) | `tables/single_session_response.csv`, `figs/single_session_response.png` |
| `target_and_feedback_sensitivity.py` | PIPELINE | `run_all.py:1685` (`_import_script_module`). Added 2026-09-30 for mock-review roadmap items S1/S5 | `tables/target_sensitivity.csv`, `tables/feedback_form_sensitivity.csv` |
| `make_parameter_table.py` | PIPELINE | `run_all.py` (`_import_script_module`, after 14b.3). Added 2026-09-30 for mock-review roadmap item R13; values read live from code, never hand-typed | `paper/parameters.tex` (`\input` by the manuscript) |
| `skin_type_breakdown.py` | FEW-TIME | no importer found; regenerates tracked reporting artefacts, but is not listed as a recurring analysis and has no recurring-use declaration; untracked + gitignored 2026-09-22 (file kept on disk) | `tables/skin_type_breakdown.csv`, `figs/skin_type_breakdown.png` (both tracked) |
