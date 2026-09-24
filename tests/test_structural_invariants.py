"""Structural invariants that were previously enforced by convention only.

Four unrelated invariants live in this one file because each is small,
static (AST/text based, no model computation), and fast -- none belongs next
to the parallel-grid determinism test (`test_parallel_grid_determinism.py`),
which is slow for the opposite reason. See
a 2026-09-20 audit that found each
gap.

Deliberately does NOT modify any existing test file (W-O is concurrently
adding coverage tests under `tests/` in this same wave; this file's name is
chosen to avoid any collision).
"""

import ast
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modules retired to exploratory/ on 2026-08-21 (CLAUDE.md "Retired modules"
# section / exploratory/README.md). Nothing in the core pipeline may import
# these.
BANNED_EXPLORATORY_MODULES = {
    "vitd_genetics", "vitd_ambient", "vitd_behaviour",
    "nhanes_preprocessing", "vitd_solar", "vitd_exposure",
}

# Directories excluded from the "core pipeline" walk: exploratory/ and
# tests/exploratory/ are explicitly allowed to import the retired modules;
# everything else (build/venv/cache/vendored output) is just noise.
EXCLUDED_DIR_NAMES = {
    "exploratory", "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "audit_outputs",
}


def _iter_core_py_files():
    """Yield absolute paths of every core (non-exploratory) .py file."""
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        rel = os.path.relpath(dirpath, REPO_ROOT)
        parts = [] if rel == "." else rel.split(os.sep)
        # Prune excluded directories in-place (os.walk honours mutation).
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES]
        if any(p in EXCLUDED_DIR_NAMES for p in parts):
            continue
        # tests/exploratory is nested under tests/, which is not itself
        # excluded (tests/*.py must be scanned for item 4's imports), so
        # skip it explicitly here.
        if parts[:2] == ["tests", "exploratory"]:
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _imported_module_roots(tree):
    """Top-level module names named in Import/ImportFrom nodes of `tree`."""
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


# ---------------------------------------------------------------------------
# Item 1: pytest.ini pythonpath
# ---------------------------------------------------------------------------

def test_pytest_ini_declares_pythonpath():
    """`pytest.ini` must set `pythonpath = .`.

    Without it, `python -m pytest tests/` only collects core modules because
    `-m` happens to put CWD on sys.path as a side effect of how Python runs
    `-m` modules; a bare `pytest tests/` invocation (the console-script entry
    point, no `-m`) does not get that side effect and fails collection with
    `ModuleNotFoundError: No module named 'vitd_scenario'` (etc.) for every
    core test module. That makes CLAUDE.md's documented verification command
    silently depend on exactly how it is typed. Verified empirically
    (2026-09-20): the bare console-script `pytest` fails core collection
    without this line and passes with it.
    """
    ini_path = os.path.join(REPO_ROOT, "pytest.ini")
    with open(ini_path, encoding="utf-8") as f:
        text = f.read()

    import re
    m = re.search(r"(?m)^\s*pythonpath\s*=\s*(.+?)\s*$", text)
    assert m is not None, (
        "pytest.ini has no `pythonpath` setting; bare `pytest tests/` "
        "(without `python -m`) will fail to collect core modules."
    )
    tokens = m.group(1).split()
    assert "." in tokens, (
        f"pytest.ini's pythonpath is {tokens!r}, which does not include the "
        "repo root ('.'); core modules would still fail to import under a "
        "bare `pytest` invocation."
    )


# ---------------------------------------------------------------------------
# Item 3: exploratory/ isolation
# ---------------------------------------------------------------------------

def test_core_pipeline_does_not_import_exploratory_modules():
    """No core .py file may import a retired exploratory/ module.

    Regex-on-source-text was the audit's original check; this uses `ast`
    instead so a match is a real Import/ImportFrom node (not a string that
    merely contains one of these names in a comment or docstring), matching
    the same "grep for a real import, not a name" standard applied to the
    script-bucket check below (item 4) and to CLAUDE.md's own script-lifecycle
    rule after this wave's edit.
    """
    violations = []
    for path in _iter_core_py_files():
        with open(path, encoding="utf-8") as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        hit = _imported_module_roots(tree) & BANNED_EXPLORATORY_MODULES
        if hit:
            violations.append((os.path.relpath(path, REPO_ROOT), sorted(hit)))

    assert not violations, (
        "Core pipeline files import retired exploratory/ modules (see "
        "CLAUDE.md 'Retired modules' / exploratory/README.md); the frozen "
        "replacement is vitd_scenario.py:\n" +
        "\n".join(f"  {p}: {mods}" for p, mods in violations)
    )


def test_exploratory_absent_from_sys_path_during_core_collection():
    """`exploratory/` must not be on sys.path when core tests run.

    `tests/exploratory/conftest.py` inserts `exploratory/` onto sys.path for
    that directory's own tests only (CLAUDE.md: "a sys.path leak between
    conftest files has previously masked a broken import"). This test asserts
    the leak has not happened in the current session -- it is meaningful for
    the mandated core command,
    `pytest tests/ --ignore=tests/exploratory`, under which
    `tests/exploratory/conftest.py` is never collected and so never runs. If
    this test is run together with `tests/exploratory` in the SAME pytest
    session (no `--ignore`), it will (correctly) fail once collection reaches
    `tests/exploratory/conftest.py` -- that failure is the invariant working
    as intended, not a false positive; combining the two directories in one
    session defeats the isolation the `--ignore` flag exists to provide.
    """
    exploratory_abs = os.path.join(REPO_ROOT, "exploratory")
    leaked = [
        p for p in sys.path
        if os.path.normcase(os.path.abspath(p)) == os.path.normcase(exploratory_abs)
    ]
    assert not leaked, (
        f"exploratory/ is on sys.path during core test collection: {leaked}. "
        "This means tests/exploratory/conftest.py ran in this session -- "
        "check how pytest was invoked (the core command must pass "
        "--ignore=tests/exploratory)."
    )


# ---------------------------------------------------------------------------
# Item 4: script bucket consistency (PIPELINE scripts must not be gitignored
# or untracked)
# ---------------------------------------------------------------------------

def _scripts_dir_names():
    scripts_dir = os.path.join(REPO_ROOT, "scripts")
    return {
        fn[:-3] for fn in os.listdir(scripts_dir)
        if fn.endswith(".py")
    }


def _pipeline_scripts_imported_by_run_all():
    """Scripts real-imported by run_all.py: bare `import X` or
    `_import_script_module("X")` (both by-string and by-AST-literal)."""
    path = os.path.join(REPO_ROOT, "run_all.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    script_names = _scripts_dir_names()
    found = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in script_names:
                    found.add(root)
        elif isinstance(node, ast.Call):
            func = node.func
            is_import_script_module = (
                isinstance(func, ast.Name) and func.id == "_import_script_module"
            )
            if is_import_script_module and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    if arg0.value in script_names:
                        found.add(arg0.value)
    return found


def _pipeline_scripts_imported_by_tests():
    """Scripts real-imported by tests/*.py (excluding tests/exploratory):
    `from scripts.X import ...` or `import scripts.X`."""
    script_names = _scripts_dir_names()
    found = set()
    tests_dir = os.path.join(REPO_ROOT, "tests")
    for fn in os.listdir(tests_dir):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(tests_dir, fn)
        with open(path, encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=path)
            except SyntaxError:
                continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "scripts":
                    parts = node.module.split(".")
                    if len(parts) > 1 and parts[1] in script_names:
                        found.add(parts[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[0] == "scripts" and len(parts) > 1 and parts[1] in script_names:
                        found.add(parts[1])
    return found


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_pipeline_scripts_are_not_gitignored_or_untracked():
    """A script classified PIPELINE by a real `import` must be tracked.

    CLAUDE.md's script-lifecycle rule: "PIPELINE -- imported by run_all.py, a
    module it imports, or tests/. Never gitignore, never delete." The
    cautionary example on record is `.gitignore` once gitignoring two scripts
    `run_all.py` imports via `_import_script_module`, which would silently
    break a fresh clone. Classification here is by real `import`
    statements/`_import_script_module("name")` calls found via `ast`, not by
    grepping for the script's bare name -- the audit found the name-grep
    unreliable (five scripts appear in tests/ only inside comments).
    """
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not inside a git work tree; cannot check tracked/ignored state")

    pipeline = _pipeline_scripts_imported_by_run_all() | _pipeline_scripts_imported_by_tests()
    assert pipeline, "expected at least one PIPELINE script (sanity check on the detector itself)"

    problems = []
    for name in sorted(pipeline):
        rel_path = f"scripts/{name}.py"
        abs_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(abs_path):
            problems.append(f"{rel_path}: imported but does not exist on disk")
            continue

        # --no-index: match the pattern regardless of tracked state. Plain
        # `git check-ignore` silently reports "not ignored" for a path that
        # is already tracked, even when a .gitignore pattern matches it
        # (verified empirically 2026-09-20) -- which would make this check a
        # no-op for exactly the currently-tracked PIPELINE scripts it exists
        # to guard, since a newly added but-not-yet-`git rm --cached`d
        # ignore pattern is the actual smell to catch before it silently
        # drops the file on some future re-add.
        ignored = _git("check-ignore", "--no-index", "-q", rel_path).returncode == 0
        tracked = _git("ls-files", "--error-unmatch", rel_path).returncode == 0

        if ignored:
            problems.append(f"{rel_path}: PIPELINE script is gitignored")
        if not tracked:
            problems.append(f"{rel_path}: PIPELINE script is untracked")

    assert not problems, (
        "PIPELINE-bucket scripts (real-imported by run_all.py or tests/) "
        "must be tracked and not gitignored:\n" +
        "\n".join(f"  {p}" for p in problems)
    )


# ---------------------------------------------------------------------------
# Item 5: docs/SCRIPTS.md manifest consistency
#
# The manifest exists because the absence of one caused two mis-bucketing
# incidents. These two tests guard it from the opposite direction to item 4:
# item 4 derives PIPELINE membership from real imports, whereas these check
# that the *written-down* classification cannot drift away from `.gitignore`
# or from what is actually on disk.
# ---------------------------------------------------------------------------

SCRIPTS_MANIFEST = os.path.join(REPO_ROOT, "docs", "SCRIPTS.md")


def _parse_scripts_manifest():
    """Rows of docs/SCRIPTS.md as {script filename: bucket}.

    A row is a Markdown table line whose first cell is a backticked `*.py`
    filename. The bucket cell may be bold-wrapped (`**UNDECIDED**`), so
    asterisks are stripped before comparison.
    """
    import re

    rows = {}
    with open(SCRIPTS_MANIFEST, encoding="utf-8") as f:
        for line in f:
            if not line.lstrip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            m = re.fullmatch(r"`([\w./-]+\.py)`", cells[0])
            if not m:
                continue
            rows[m.group(1)] = cells[1].replace("*", "").strip()
    return rows


def test_scripts_manifest_lists_every_script_on_disk():
    """docs/SCRIPTS.md must have exactly one row per file in scripts/.

    Drift in either direction is a failure: a script added to disk but not to
    the manifest is unbucketed (the state that preceded both recorded
    mis-bucketing incidents), and a row for a file that no longer exists makes
    the manifest a stale pointer.
    """
    manifest = set(_parse_scripts_manifest())
    on_disk = {
        fn for fn in os.listdir(os.path.join(REPO_ROOT, "scripts"))
        if fn.endswith(".py")
    }

    missing = sorted(on_disk - manifest)
    extra = sorted(manifest - on_disk)
    assert not missing and not extra, (
        "docs/SCRIPTS.md has drifted from scripts/ on disk:\n"
        + "".join(f"  on disk but not in the manifest: {n}\n" for n in missing)
        + "".join(f"  in the manifest but not on disk: {n}\n" for n in extra)
    )


def test_manifest_pipeline_scripts_are_not_gitignored():
    """No script the manifest classifies PIPELINE may match .gitignore.

    The recorded failure (CLAUDE.md, script-lifecycle rule): `.gitignore` once
    ignored two scripts `run_all.py` imports via `_import_script_module`, so a
    fresh clone could not run the pipeline. This complements
    `test_pipeline_scripts_are_not_gitignored_or_untracked` -- that one asks
    git what is imported, this one asks the written manifest, so a row
    mislabelled PIPELINE while being ignored is caught too.

    `--no-index` for the same reason documented on the item-4 check: plain
    `git check-ignore` reports "not ignored" for an already-tracked path even
    when a pattern matches it, which would make this a no-op on exactly the
    scripts it guards.
    """
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not inside a git work tree; cannot check ignored state")

    pipeline = [
        name for name, bucket in _parse_scripts_manifest().items()
        if bucket == "PIPELINE"
    ]
    assert pipeline, (
        "docs/SCRIPTS.md lists no PIPELINE script; either the manifest is "
        "empty/unparseable or the bucket column has changed shape"
    )

    ignored = []
    for name in sorted(pipeline):
        rel_path = f"scripts/{name}"
        if _git("check-ignore", "--no-index", "-q", rel_path).returncode == 0:
            ignored.append(rel_path)

    assert not ignored, (
        "scripts classified PIPELINE in docs/SCRIPTS.md are matched by "
        ".gitignore; a fresh clone could not run the pipeline:\n"
        + "\n".join(f"  {p}" for p in ignored)
    )


def test_manifest_run_all_pointers_still_name_the_script_import():
    """Live run_all.py pointers in docs/SCRIPTS.md must retain their value.

    Checking only that a cited line exists is insufficient: a line shift can
    land on a different valid statement. The manifest's cited line must still
    be the dynamic script import and must contain the row's script name.
    """
    import re

    with open(SCRIPTS_MANIFEST, encoding="utf-8") as f:
        manifest_text = f.read()

    problems = []
    for script_name, line_number in re.findall(
        r"`([\w./-]+\.py)`[^\n]*?`run_all\.py:(\d+)`", manifest_text
    ):
        source_path = os.path.join(REPO_ROOT, "run_all.py")
        with open(source_path, encoding="utf-8") as f:
            source_lines = f.readlines()
        index = int(line_number) - 1
        if not 0 <= index < len(source_lines):
            problems.append(f"{script_name}: run_all.py:{line_number} is out of range")
            continue
        target = source_lines[index]
        if "_import_script_module" not in target or script_name[:-3] not in target:
            problems.append(
                f"{script_name}: run_all.py:{line_number} points to {target.strip()!r}"
            )

    assert not problems, "Stale docs/SCRIPTS.md run_all.py pointers:\n" + "\n".join(
        f"  {problem}" for problem in problems
    )


# ---------------------------------------------------------------------------
# Stage 1: Pointer & Value Integrity Guard Tests
# ---------------------------------------------------------------------------

def test_stratum_corneum_parameter_invariants():
    """Verify stratum corneum parameters in ModelParams match documented specifications."""
    import numpy as np
    from vitd_model import ModelParams

    params = ModelParams()
    assert params.SC_max == 0.30
    # [REBASED 2026-09-23 for decision A1, 2026-09-22] The FST IV-VI cell
    # moved 0.36 -> 0.24, i.e. to the corroborated FST I-III value, and is
    # carried as a RANGE, SC_MAX_IV_VI_RANGE. The old 0.36 rested solely on
    # de Boer 2025's FP IV-VI arm, which was never checked for FST V-VI
    # enrolment; the error is one-sided (no source argues for HIGHER), so
    # 0.36 sat at the most-burdensome extreme of its own evidence. The
    # RANGE, not the point, is what an FST IV-VI result must be reported
    # across -- so the range constant is asserted here too.
    from vitd_model import SC_MAX_IV_VI_RANGE
    assert params.SC_max_by_FST == {
        1: 0.24, 2: 0.24, 3: 0.24,
        4: 0.24, 5: 0.24, 6: 0.24,
    }
    for fst in range(1, 7):
        assert params.SC_max_by_FST[fst] == 0.24
    assert SC_MAX_IV_VI_RANGE == (0.24, 0.36)
    assert params.alpha_SC == 0.12
    assert np.isclose(params.k_decay_SC, np.log(2) / 28.0)
    assert np.isclose(params.k_repair_SC, np.log(2) / 7.0)
    assert params.k_damage == 0.0

    # [REBASED 2026-09-23] Prior ceiling widened 0.40 -> 0.54 on 2026-09-22:
    # two measured thickenings sit above the old ceiling (Bruls 1984
    # 0.5384615384615384 ex vivo, and Felton 2020 FST II n=10 +41.6% in
    # vivo), and a prior interval that excludes its own measurements is a
    # constraint, not a prior. 0.54 is a recorded rounding decision: the
    # smallest two-decimal bound containing 0.5384615384615384.
    assert ModelParams.SC_PRIOR_RANGES["SC_max"] == (0.20, 0.54)
    assert ModelParams.SC_PRIOR_RANGES["alpha_SC"] == (0.08, 0.20)
    assert ModelParams.SC_PRIOR_RANGES["K_SC"] == (0.5, 2.0)


def test_melanin_parameter_invariants():
    """Verify constitutive and facultative melanin parameters match documented specifications."""
    from vitd_model import ModelParams

    params = ModelParams()
    assert params.alpha == 0.0025
    assert params.alpha_by_FST == {
        1: 0.0025, 2: 0.0025, 3: 0.0025,
        4: 0.002, 5: 0.002, 6: 0.002,
    }
    # [REBASED 2026-09-23] K_const 1.17 -> 2.096070737357932 and the whole
    # M_baseline_by_FST table recomputed through the re-scaled ITA -> M
    # mapping M = (55 - ITA_U) / 121 (decision A3, 2026-09-22), whose lower
    # anchor is Linde 2020's printed constitutive minimum ITA -66 instead of
    # Del Bino's -30. NO CELL CLIPS now; the old table's FST VI cell was a
    # clip at 1.00 with no resolving power inside FST VI. K_const and the
    # table moved TOGETHER and must be read together: the re-scale alone
    # raises K_const 1.4237x for an exactly unchanged optical factor.
    assert params.K_const == 2.096070737357932
    assert params.M_baseline_by_FST == {
        1: 0.026776859504132246,
        2: 0.12049586776859503,
        3: 0.2269421487603306,
        4: 0.36818181818181817,
        5: 0.5295867768595042,
        6: 0.7324793388429751,
    }
    assert max(params.M_baseline_by_FST.values()) < 1.0, (
        "no cell may clip at 1.0 under the 2026-09-22 mapping")
    assert params.M_TAN_HEADROOM == 0.238


def test_m_ceiling_logic_and_m_max_dead():
    """Verify M_ceiling logic allows tanning above 1.0 and M_max is not read in simulate/M_ceiling."""
    import numpy as np
    from vitd_model import ModelParams

    params = ModelParams()
    ceiling_6 = params.M_ceiling(6)
    # [REBASED 2026-09-23] The invariant under test is that the ceiling is
    # M_baseline(F) + M_TAN_HEADROOM with NO cap -- it was written as
    # "> 1.0" only because M_baseline(VI) was itself a clip at exactly 1.0.
    # After the 2026-09-22 ITA -> M re-scale M_baseline(VI) is
    # 0.7324793388429751, so the FST VI ceiling is 0.9704793388429751 and a
    # literal "> 1.0" would now assert the clip this repo just removed. What
    # must hold, and is asserted instead, is that the ceiling is strictly
    # above the baseline by the full headroom and is not truncated anywhere.
    expected_ceiling = params.M_baseline(6) + params.M_TAN_HEADROOM
    assert np.isclose(ceiling_6, expected_ceiling)
    assert np.isclose(ceiling_6, 0.9704793388429751)
    assert ceiling_6 > params.M_baseline(6)
    for fst in range(1, 7):
        assert np.isclose(params.M_ceiling(fst),
                          params.M_baseline(fst) + params.M_TAN_HEADROOM), fst

    # AST check: in vitd_model.py, M_max must NOT be read or used in simulate() or M_ceiling()
    vitd_path = os.path.join(REPO_ROOT, "vitd_model.py")
    with open(vitd_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=vitd_path)

    target_funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ("simulate", "M_ceiling"):
                target_funcs[node.name] = node

    assert "simulate" in target_funcs, "simulate function not found in vitd_model.py AST"
    assert "M_ceiling" in target_funcs, "M_ceiling function not found in vitd_model.py AST"

    for func_name, func_node in target_funcs.items():
        for subnode in ast.walk(func_node):
            if isinstance(subnode, ast.Attribute) and subnode.attr == "M_max":
                pytest.fail(
                    f"vitd_model.{func_name}() accesses attribute 'M_max' at line {subnode.lineno}; "
                    "M_max is dead code and should not be read."
                )
            if isinstance(subnode, ast.Name) and subnode.id == "M_max":
                pytest.fail(
                    f"vitd_model.{func_name}() references symbol 'M_max' at line {subnode.lineno}."
                )


def test_e_source_deprecation_warning():
    """Calling get_E_source('solar_broadband') must emit a UserWarning."""
    from vitd_model import get_E_source

    with pytest.warns(UserWarning, match=r"DEPRECATED"):
        val = get_E_source("solar_broadband")
    assert val == 1.0


def test_doc_citation_line_pointers_valid():
    """Verify vitd_model.py line citations in SC_MODEL.md and MELANIN_MODEL.md are valid."""
    import re

    vitd_path = os.path.join(REPO_ROOT, "vitd_model.py")
    with open(vitd_path, encoding="utf-8") as f:
        vitd_text = f.read()
        vitd_lines = vitd_text.splitlines()

    total_lines = len(vitd_lines)
    tree = ast.parse(vitd_text, filename=vitd_path)

    mp_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelParams":
            mp_node = node
            break
    assert mp_node is not None, "ModelParams class not found in vitd_model.py"
    mp_start = mp_node.lineno
    mp_end = mp_node.end_lineno

    # Map field/method symbol -> line definition inside ModelParams
    mp_symbols = {}
    for item in mp_node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            mp_symbols[item.target.id] = item.lineno
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name):
                    mp_symbols[t.id] = item.lineno
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mp_symbols[item.name] = item.lineno

    # Load-bearing symbols to check
    load_bearing_symbols = {
        "SC_max", "SC_max_by_FST", "alpha_SC", "k_decay_SC", "k_damage",
        "k_repair_SC", "K_SC", "K_const", "M_TAN_HEADROOM", "M_ceiling", "M_baseline"
    }

    docs = ["docs/SC_MODEL.md", "docs/MELANIN_MODEL.md"]
    all_citations = []
    load_bearing_checked = []

    for doc_rel in docs:
        doc_path = os.path.join(REPO_ROOT, doc_rel)
        assert os.path.exists(doc_path), f"{doc_rel} does not exist"
        with open(doc_path, encoding="utf-8") as f:
            doc_lines = f.readlines()

        current_section_symbol = None
        for doc_idx, line in enumerate(doc_lines, 1):
            if line.startswith("### "):
                bt = re.findall(r"`([A-Za-z_]\w*)`", line)
                current_section_symbol = bt[0] if bt else None

            citations = [int(x) for x in re.findall(r"vitd_model\.py:(\d+)", line)]
            for c in citations:
                all_citations.append((doc_rel, doc_idx, c))
                assert 1 <= c <= total_lines, (
                    f"{doc_rel}:{doc_idx} cites line {c} outside vitd_model.py bounds (1..{total_lines})"
                )

                # Identify symbols associated with this line/section
                syms = set(re.findall(r"ModelParams\.([A-Za-z_]\w*)", line))
                if current_section_symbol in load_bearing_symbols and any(
                    kw in line for kw in ("Code:", "Symbols:", "Symbol:")
                ):
                    syms.add(current_section_symbol)

                for sym in syms:
                    if sym in load_bearing_symbols and sym in mp_symbols:
                        load_bearing_checked.append((sym, doc_rel, doc_idx, c))
                        sym_def = mp_symbols[sym]
                        directly_contains = sym in vitd_lines[c - 1]
                        within_50 = abs(c - sym_def) <= 50
                        within_mp = mp_start <= c <= mp_end
                        assert directly_contains or within_50 or within_mp, (
                            f"{doc_rel}:{doc_idx} cites line {c} for {sym}, "
                            f"which is neither directly named on that line, nor within 50 lines "
                            f"of definition (line {sym_def}), nor inside ModelParams class "
                            f"({mp_start}..{mp_end}). Citation is completely stale or misplaced."
                        )

    # Sanity checks
    assert len(all_citations) >= 10, f"Expected >= 10 citations across docs, found {len(all_citations)}"
    for req_sym in ("SC_max", "K_const", "M_TAN_HEADROOM"):
        assert any(item[0] == req_sym for item in load_bearing_checked), (
            f"Expected {req_sym} to be verified among load-bearing citations, checked: {load_bearing_checked}"
        )


# ---------------------------------------------------------------------------
# Item 6: pointer-integrity guard against the retired post-manuscript
# workstream programme -- proposed but never built during that programme
# (see docs/SOURCES.md for what it tracked and calibration_approval.yaml's
# pointer-repair history for why this exists: a frozen file:line or
# docs/workstreams/W-*.md citation rots the moment the cited file moves,
# is rewritten, or -- for these four docs plus docs/workstreams/ -- is
# deleted outright). Two independent checks:
#   1. no tracked file may still name the four doomed top-level docs or the
#      docs/workstreams/ directory by path;
#   2. every docs/*.md path named anywhere in a tracked file must exist.
# Neither check is scoped to files this fold pass owns -- a failure here
# from a file owned by a different in-flight agent is left failing on
# purpose (not xfailed) so it stays visible until that agent fixes it.
# ---------------------------------------------------------------------------

DOOMED_DOC_PATTERNS = (
    "docs/workstreams/",
    "DECISIONS_PENDING",
    "HANDOFF_2026",
    "ORCHESTRATOR_",
)

DOOMED_DOC_PATHS = {
    "docs/HANDOFF_2026-09-22.md",
    "docs/ORCHESTRATOR_HANDOFF.md",
    "docs/ORCHESTRATOR_PLAN.md",
    "docs/DECISIONS_PENDING.md",
}


def _tracked_text_files(*globs):
    """(rel_path, text) for every git-tracked file matching `globs`.

    Excludes .venv/ (defensive -- normally gitignored already, so this is
    a no-op today) and the doomed docs themselves (defensive -- they are
    currently untracked, so `git ls-files` would not list them anyway,
    but this keeps the exclusion correct if that ever changes before they
    are deleted).
    """
    result = _git("ls-files", *globs)
    if result.returncode != 0:
        pytest.skip("not inside a git work tree; cannot enumerate tracked files")
    files = []
    for rel in result.stdout.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        if rel in DOOMED_DOC_PATHS or rel.startswith("docs/workstreams/"):
            continue
        if rel.startswith(".venv/") or "/.venv/" in rel:
            continue
        if rel == "tests/test_structural_invariants.py":
            # This file itself: it necessarily names the doomed patterns
            # and paths as literal string constants in order to check for
            # them, which would otherwise make it self-matching.
            continue
        abs_path = os.path.join(REPO_ROOT, rel)
        try:
            with open(abs_path, encoding="utf-8") as f:
                text = f.read()
        except (FileNotFoundError, UnicodeDecodeError, IsADirectoryError):
            continue
        files.append((rel, text))
    return files


def test_no_tracked_file_references_the_doomed_docs():
    """No tracked .py/.md/.yaml file may reference the retired programme.

    `docs/HANDOFF_2026-09-22.md`, `docs/ORCHESTRATOR_HANDOFF.md`,
    `docs/ORCHESTRATOR_PLAN.md`, `docs/DECISIONS_PENDING.md` and
    `docs/workstreams/*.md` are slated for deletion. A tracked file that
    still names one of them is a dangling reference that will silently
    stop resolving to anything the moment the deletion happens -- exactly
    the failure mode the fold-and-delete pass this test comes from exists
    to prevent. If this fails on a file this pass does not own, that is
    intentional: it stays failing, with the file and line named below,
    until that file's owner fixes it -- it is not xfailed away.
    """
    problems = []
    for rel, text in _tracked_text_files("*.py", "*.md", "*.yaml", "*.yml"):
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(pat in line for pat in DOOMED_DOC_PATTERNS):
                problems.append(f"{rel}:{lineno}: {line.strip()[:160]}")

    assert not problems, (
        "The following tracked files still reference a doomed "
        "docs/workstreams/, DECISIONS_PENDING, HANDOFF_2026, or "
        "ORCHESTRATOR_ path:\n" + "\n".join(f"  {p}" for p in problems)
    )


def test_every_referenced_docs_md_path_exists():
    """Every `docs/*.md` path named in a tracked file must exist on disk.

    The general case of the check above: any live file citing a
    docs/ Markdown path that is not actually present is pointing a reader
    at nothing, doomed-programme or not.
    """
    import re

    pattern = re.compile(r"docs/[A-Za-z0-9_./-]+\.md")
    problems = []
    for rel, text in _tracked_text_files("*.py", "*.md", "*.yaml", "*.yml"):
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in pattern.finditer(line):
                doc_path = m.group(0).rstrip("'\"`).,:;")
                if not os.path.exists(os.path.join(REPO_ROOT, doc_path)):
                    problems.append(
                        f"{rel}:{lineno}: cites {doc_path!r}, which does "
                        f"not exist ({line.strip()[:120]})"
                    )

    assert not problems, (
        "The following docs/*.md paths are cited but do not exist on "
        "disk:\n" + "\n".join(f"  {p}" for p in problems)
    )

