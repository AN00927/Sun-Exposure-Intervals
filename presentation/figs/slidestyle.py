"""Shared style for the presentation figures in this directory.

Every f*.py script here imports use_slide_style() before drawing and writes
its PNG into this same directory. Nothing outside presentation/ is read for
style or written to.

Palette: navy ink on white, one teal accent, a light-to-dark FST ramp for
skin type. The FST ramp is verified below at import time -- WCAG relative
luminance must decrease monotonically I -> VI with every adjacent gap at
least 0.06, so the six colours read as skin type and stay separable even in
greyscale. See README.md for the measured luminance table.
"""
import matplotlib.pyplot as plt

ACCENT = "#0584a4"        # teal
TEXT_COLOR = "#0e2a47"    # navy

GREY_INFEASIBLE = "#bdbdbd"
GREYS = ["#595959", "#767676", "#969696", "#b8b8b8"]  # strongest evidence first

FST_LABELS = ["I", "II", "III", "IV", "V", "VI"]
FST_COLORS = [
    "#f2d8c3",  # I
    "#e2b78e",  # II
    "#ce945f",  # III
    "#b47239",  # IV
    "#90582c",  # V
    "#67401f",  # VI
]


def relative_luminance(hex_color):
    """WCAG relative luminance of a '#rrggbb' colour, in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    channels = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def fst_luminances():
    return [relative_luminance(c) for c in FST_COLORS]


def _verify_fst_ramp():
    lums = fst_luminances()
    for i in range(len(lums) - 1):
        gap = lums[i] - lums[i + 1]
        assert gap >= 0.06, (
            f"FST ramp gap too small between FST {FST_LABELS[i]} and "
            f"FST {FST_LABELS[i + 1]}: {gap:.4f} (need >= 0.06)"
        )


_verify_fst_ramp()


def pct(x, decimals=0, show_plus=False):
    """Format a percentage the one agreed way: rounded, with a proper
    minus sign (U+2212) rather than a hyphen. Pass decimals=1 only where a
    whole number would collapse two distinct values onto the same label.
    """
    rounded = round(x, decimals)
    if rounded < 0:
        sign = "−"
    elif show_plus and rounded > 0:
        sign = "+"
    else:
        sign = ""
    return f"{sign}{abs(rounded):.{decimals}f}%"


def use_slide_style():
    plt.rcParams.update({
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Calibri", "DejaVu Sans", "Arial"],
        "font.size": 13,
        "text.color": TEXT_COLOR,
        "axes.labelsize": 14,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlesize": 15,
        "axes.titlecolor": TEXT_COLOR,
        "axes.edgecolor": "#4d4d4d",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.5,
        "legend.frameon": False,
    })
