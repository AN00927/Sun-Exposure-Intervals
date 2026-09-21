"""Determinism test for the joint-schedule grid's process-pool dispatch.

`vitd_optimization.py` (`analysis_joint_schedule`, just above the `tasks = [...]`
line) claims:

    "The grid cells, by contrast, are independent and each DE solve inside
    them carries its own fixed seed, so dispatching whole cells reproduces
    the serial CSV byte for byte."

Before this test, nothing checked that claim: a grep for `_run_joint_grid`,
`_joint_grid_cell`, `n_workers`, or `ProcessPool` across `tests/` returned
zero hits (2026-09-20 audit; see docs/ORCHESTRATOR_PLAN.md W-M item 2). This
test runs the SAME two grid cells through `_run_joint_grid` once serially
(`n_workers=1`) and once through the process pool (`n_workers=2`), and
asserts the two result sets are identical field-for-field -- the closest
thing to the "byte for byte" claim that a Python object comparison can make
(the CSV serialisation step itself is untouched by n_workers and is not
re-tested here).

COST, MEASURED (2026-09-20, this machine): a single grid cell -- two dose
families (`exponential`, `two_phase`) x two spacing arms (fixed, free), each
a `differential_evolution(maxiter=18, popsize=10)` solve -- takes ~35 s.
`_joint_grid_cell` does not accept a `maxiter`/`popsize` override, so this
cannot be made cheaper without editing production code (out of scope for a
test-only change). Two cells is the SMALLEST configuration that exercises
the pool at all: `_run_joint_grid` clamps `n = min(n_workers, len(tasks))`,
so a single-task list never spins up a second worker regardless of
`n_workers`. Measured end-to-end runtime of this one test, this machine,
2026-09-20: 94.82 s (one serial pass over 2 cells plus one 2-worker pool
pass over the same 2 cells). That is a substantial addition against a
~3-minute core-suite baseline -- roughly +50% wall time from a single test.
This is called out here, as the plan instructs, rather than silently
accepted. It is accepted
because this is the newest, least-tested code path in the optimiser and a
violation (e.g. a BLAS/thread-count difference between a spawned worker and
the parent process changing floating-point rounding) would silently corrupt
`tables/joint_schedule_optimization.csv` in whichever direction the CI
machine's core count happens to push `n_workers`.

Non-vacuousness was checked by running the SAME comparison against two
DIFFERENT task lists (e.g. session_counts (104,) vs (146,) at fixed FST) and
confirming the equality assertion actually fails -- i.e. the comparison
logic distinguishes real differences rather than trivially matching (see
docs/workstreams/W-M.md for the transcript). That check is not re-run here
on every test session (it would double the already-substantial cost); it
was a one-time verification during test authoring.
"""
import math

from vitd_optimization import _run_joint_grid


# Smallest configuration that exercises the pool: 2 cells at the SAME FST
# (so both cells are equally likely to be feasible/infeasible in the same
# way) but different session budgets, both drawn from the FST VI x
# session-budget grid actually used by analysis_joint_schedule.
_TASKS = [(6, 104, None), (6, 146, None)]


def _floats_equal(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return a == b


def _dict_rows_equal(d1, d2):
    if d1 is None or d2 is None:
        return d1 is d2
    if set(d1.keys()) != set(d2.keys()):
        return False
    return all(_floats_equal(d1[k], d2[k]) for k in d1)


def _cell_results_equal(cell_a, cell_b):
    fst_a, n_a, fixed_a, joint_a, binding_a = cell_a
    fst_b, n_b, fixed_b, joint_b, binding_b = cell_b
    return (
        fst_a == fst_b
        and n_a == n_b
        and binding_a == binding_b
        and _dict_rows_equal(fixed_a, fixed_b)
        and _dict_rows_equal(joint_a, joint_b)
    )


def test_joint_grid_pool_matches_serial_byte_for_byte():
    """2-worker ProcessPoolExecutor path reproduces the serial results exactly.

    Guards the claim at the top of this file (source: the comment above
    `tasks = [...]` in `vitd_optimization.analysis_joint_schedule`).
    """
    serial = _run_joint_grid(list(_TASKS), n_workers=1)
    pooled = _run_joint_grid(list(_TASKS), n_workers=2)

    assert len(serial) == len(pooled) == len(_TASKS)

    # ProcessPoolExecutor.map preserves submission order (documented), so
    # order itself is part of the invariant -- zip, don't sort-then-compare.
    mismatches = [
        (s, p) for s, p in zip(serial, pooled) if not _cell_results_equal(s, p)
    ]
    assert not mismatches, (
        "n_workers=2 (process pool) produced different results than "
        "n_workers=1 (serial) for the same tasks -- the "
        "'reproduces the serial CSV byte for byte' claim in "
        "vitd_optimization.analysis_joint_schedule does not hold:\n"
        + "\n".join(f"  serial={s!r}\n  pooled={p!r}" for s, p in mismatches)
    )
