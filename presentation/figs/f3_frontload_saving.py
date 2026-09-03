"""F3 (hero): UV saving from front-loading, by Fitzpatrick type.

Source: tables/primary_minimum_schedule.csv, n_sessions == 365,
spacing 'uniform' vs 'free'.  Percent change in total_sed, uniform -> free.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidestyle import use_slide_style, FST_COLORS, FST_LABELS, TEXT_COLOR, pct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(REPO, "tables", "primary_minimum_schedule.csv")
OUT = os.path.join(HERE, "f3_frontload_saving.png")

EXPECTED = {
    1: -29.5604, 2: -29.7758, 3: -30.2468,
    4: -32.0548, 5: -33.9544, 6: -35.5031,
}


def compute():
    df = pd.read_csv(CSV)
    sel = df[df["n_sessions"] == 365]
    uni = sel[sel["spacing"] == "uniform"].set_index("fst")["total_sed"]
    free = sel[sel["spacing"] == "free"].set_index("fst")["total_sed"]
    pct_change = (free - uni) / uni * 100.0
    return pct_change.sort_index()


def main():
    use_slide_style()
    pct_change = compute()

    for fst, want in EXPECTED.items():
        got = float(pct_change.loc[fst])
        assert round(got, 4) == want, (
            f"F3 assertion FAILED for FST {fst}: expected {want:.4f}%, "
            f"got {got:.10f}%"
        )
    print("F3: all six FST percent changes match the verified values.")

    fig, ax = plt.subplots(figsize=(9, 5))
    order = [1, 2, 3, 4, 5, 6]
    ypos = np.arange(len(order))[::-1]  # FST I at top

    vals = [float(pct_change.loc[f]) for f in order]
    ax.barh(ypos, vals, height=0.68,
            color=[FST_COLORS[f - 1] for f in order],
            edgecolor="#4d4d4d", linewidth=0.8, zorder=3)

    for y, v in zip(ypos, vals):
        # decimals=1: at whole-number precision, FST I/II/III all round to
        # -30% and collapse onto the same label (-29.5604, -29.7758,
        # -30.2468) -- one decimal is the minimum that keeps all six bars
        # distinct, per the "collapse" clause in the rounding rule.
        ax.text(v - 0.7, y, pct(v, decimals=1), va="center", ha="right",
                fontsize=14, fontweight="bold", color=TEXT_COLOR, zorder=4)

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"FST {FST_LABELS[f - 1]}" for f in order])
    ax.set_xlabel("Reduction in annual UV needed to stay sufficient (%)")
    ax.set_xlim(-42.0, 0.6)
    ax.set_ylim(-0.65, len(order) - 0.35)

    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.axvline(0.0, color="#4d4d4d", linewidth=1.0, zorder=2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
