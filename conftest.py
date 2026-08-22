"""Ensure the repository root is importable for every test run.

Previously this worked only by accident, via `python -m pytest` putting the
current working directory on sys.path. Made explicit when the retired modules
moved into `exploratory/` (2026-08-21) and the import surface stopped being
flat.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
