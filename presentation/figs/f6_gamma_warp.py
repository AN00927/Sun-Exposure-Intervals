"""F6: what the spacing exponent gamma actually does to the calendar.

No CSV.  Session days come straight from vitd_optimization.warp_days.
N = 52 (not 365) so individual ticks are resolvable on a slide.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from slidestyle import use_slide_style, ACCENT, TEXT_COLOR  # noqa: E402
from vitd_optimization import warp_days  # noqa: E402

OUT = os.path.join(HERE, "f6_gamma_warp.png")

N = 52
T = 365


def main():
    use_slide_style()

    uni = warp_days(N=N, T_days=T, gamma=1.0)
    fro = warp_days(N=N, T_days=T, gamma=1.4)

    assert len(uni) == len(fro) == N, (
        f"F6 assertion FAILED: tick counts differ - uniform {len(uni)}, "
        f"front-loaded {len(fro)}, expected {N} each"
    )
    print(f"F6: {len(uni)} sessions on both rows; "
          f"uniform first gap {uni[1] - uni[0]:.4f} d, "
          f"front-loaded first gap {fro[1] - fro[0]:.4f} d, "
          f"front-loaded last gap {fro[-1] - fro[-2]:.4f} d.")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.grid(False)

    rows = [
        (1.0, uni, "Evenly spaced", "#6f6f6f"),
        (0.0, fro, "Front-loaded", ACCENT),
    ]
    half = 0.20
    for y, days, label, color in rows:
        ax.hlines(y, 0, T, color="#d9d9d9", lw=1.4, zorder=1)
        ax.vlines(days, y - half, y + half, color=color, lw=1.8, zorder=3)
        ax.text(-14, y, label, ha="right", va="center", fontsize=13.5,
                color=color, fontweight="bold")

    ax.annotate("52 sessions on both rows",
                xy=(T * 0.5, 0.5), ha="center", va="center",
                fontsize=13, color=TEXT_COLOR,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#bdbdbd", linewidth=0.9))

    ax.set_xlim(-4, T + 4)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.set_xticks([0, 60, 120, 180, 240, 300, 365])
    ax.set_xlabel("Day of year")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
