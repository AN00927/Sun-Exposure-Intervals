"""F5: what the 14 validation checks actually test.

No CSV -- the tier counts are given, and are hard-coded here with the tier
gloss.  QUANTITATIVE is the only tier that is a genuine external numeric test,
so it carries the accent colour; the other four are graded greys.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidestyle import use_slide_style, ACCENT, GREYS, TEXT_COLOR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "f5_evidence_tiers.png")

# (tier, count, gloss, colour)  -- strongest evidence first
# Not used on any slide (cut from both decks); retained here for the
# write-up and Q&A. Tier labels are plain English, not the internal
# validation-tier constants (NOT_EVIDENCE, CIRCULAR, etc.).
TIERS = [
    ("external, numeric",       2, "external data, real numeric comparison", ACCENT),
    ("direction only",          3, "external data, direction only",          GREYS[0]),
    ("self-consistency",        4, "self-consistency, not biology",          GREYS[1]),
    ("fitted to this data",     3, "data the model was fitted to",           GREYS[2]),
    ("not vitamin-D evidence",  2, "its own docstring says it is not "
                                    "vitamin-D evidence",                    GREYS[3]),
]

TOTAL = 14


def main():
    use_slide_style()

    counts = [t[1] for t in TIERS]
    assert sum(counts) == TOTAL, (
        f"F5 assertion FAILED: tier counts sum to {sum(counts)}, expected {TOTAL}"
    )
    print(f"F5: {len(TIERS)} tiers summing to {sum(counts)} checks - matches.")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.grid(False)

    ypos = np.arange(len(TIERS))[::-1]
    for y, (tier, n, gloss, color) in zip(ypos, TIERS):
        ax.barh(y, n, height=0.62, color=color, edgecolor="none", zorder=3)
        ax.text(n - 0.13, y, str(n), ha="right", va="center",
                color="white", fontsize=15, fontweight="bold", zorder=4)
        ax.text(-0.22, y, tier, ha="right", va="center",
                fontsize=13.5, fontweight="bold",
                color=ACCENT if color == ACCENT else TEXT_COLOR)
        ax.text(n + 0.18, y, gloss, ha="left", va="center",
                fontsize=13, color="#4d4d4d")

    # the one tier that is a real external numeric test
    ax.annotate("the only tier that is a genuine external numeric test",
                xy=(2.0, ypos[0] + 0.34), xycoords="data",
                xytext=(3.5, ypos[0] + 0.78), textcoords="data",
                fontsize=13, fontweight="bold", color=ACCENT,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.4,
                                connectionstyle="angle,angleA=0,angleB=90,rad=6"))

    ax.set_xlim(0, 9.2)
    ax.set_ylim(-0.55, len(TIERS) - 0.05)
    ax.set_yticks([])
    ax.set_xticks(range(0, 5))
    ax.set_xlabel(f"Checks in the validation suite (n = {TOTAL})", labelpad=2)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticklabels([])

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
