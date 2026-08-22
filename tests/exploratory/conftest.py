"""Put `exploratory/` on sys.path for the retired modules' tests.

These modules were moved out of the core pipeline on 2026-08-21 (see
`exploratory/README.md`). Their tests still run and still pass -- retirement
means "does not influence headline outputs", not "unmaintained". Adding the
directory to sys.path here, rather than rewriting every import inside the
moved modules, keeps the retired code byte-identical to its reviewed state.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXPLORATORY = os.path.join(_ROOT, "exploratory")
for _p in (_ROOT, _EXPLORATORY):
    if _p not in sys.path:
        sys.path.insert(0, _p)
