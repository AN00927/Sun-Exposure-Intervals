"""Structural invariants that were previously enforced by convention only.

Four unrelated invariants live in this one file because each is small,
static (AST/text based, no model computation), and fast -- none belongs next
to the parallel-grid determinism test (`test_parallel_grid_determinism.py`),
which is slow for the opposite reason. See
`docs/ORCHESTRATOR_PLAN.md` W-M items 1, 3, 4 for the audit that found each
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
