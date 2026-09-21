"""
scripts/ -- one-off and recurring analysis scripts, see CLAUDE.md's four-bucket
script lifecycle (PIPELINE / RECURRING / FEW-TIME / ONE-TIME) for what belongs
here and why.

This file exists to make `scripts` a regular package rather than an implicit
namespace package. `tests/test_physiology.py` and
`scripts/plot_sc_deboer2025.py` import `scripts.fit_sc_deboer2025` in
package-qualified form; `run_all.py`'s `_import_script_module()` instead puts
`scripts/` itself on `sys.path` and imports each script by its bare top-level
name (e.g. `fit_sc_deboer2025`), which does not go through this package.
"""
