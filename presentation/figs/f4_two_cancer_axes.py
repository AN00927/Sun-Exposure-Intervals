"""F4: the two cancer axes move in opposite directions under free spacing.

Source: tables/joint_schedule_optimization.csv.  Uniform (free_spacing False)
rows merged against free (free_spacing True) rows on (fst, n_sessions), keeping
only pairs feasible on both sides -- 22 such pairs.  For each pair, percent
change uniform -> free in scc_index and in intermittent_index.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidestyle import (use_slide_style, ACCENT, FST_COLORS, FST_LABELS,  # noqa: E402
                        TEXT_COLOR)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(REPO, "tables", "joint_schedule_optimization.csv")
OUT = os.path.join(HERE, "f4_two_cancer_axes.png")

EXPECT_PAIRS = 22
EXPECT_SCC_MEDIAN = 7.65        # % change, uniform -> free (worse)
EXPECT_INT_MEDIAN = -21.36      # % change, uniform -> free (better)


def pairs():
    df = pd.read_csv(CSV)
    uni = df[~df["free_spacing"].astype(bool)]
    fre = df[df["free_spacing"].astype(bool)]
    m = uni.merge(fre, on=["fst", "n_sessions"], suffixes=("_u", "_f"))
    m = m[m["feasible_u"].astype(bool) & m["feasible_f"].astype(bool)].copy()
    m["scc_pct"] = (m["scc_index_f"] - m["scc_index_u"]) / m["scc_index_u"] * 100.0
    m["int_pct"] = ((m["intermittent_index_f"] - m["intermittent_index_u"])
                    / m["intermittent_index_u"] * 100.0)
    return m.sort_values(["fst", "n_sessions"]).reset_index(drop=True)


def main():
    use_slide_style()
    m = pairs()

    assert len(m) == EXPECT_PAIRS, (
        f"F4 assertion FAILED: expected {EXPECT_PAIRS} pairs feasible on both "
        f"sides, got {len(m)}"
    )
    scc_med = float(m["scc_pct"].median())
    int_med = float(m["int_pct"].median())
    assert round(scc_med, 2) == EXPECT_SCC_MEDIAN, (
        f"F4 assertion FAILED: SCC median {scc_med!r} does not round to "
        f"{EXPECT_SCC_MEDIAN}"
    )
    assert round(int_med, 2) == EXPECT_INT_MEDIAN, (
        f"F4 assertion FAILED: intermittent median {int_med!r} does not round "
        f"to {EXPECT_INT_MEDIAN}"
    )
    print(f"F4: {len(m)} pairs; median SCC {scc_med:.4f}%, "
          f"median intermittent {int_med:.4f}% - match.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x0, x1 = 0.0, 1.0

    ax.axhline(0.0, color="#4d4d4d", lw=1.4, zorder=2)

    for _, r in m.iterrows():
        c = FST_COLORS[int(r["fst"]) - 1]
        ax.plot([x0, x1], [r["scc_pct"], r["int_pct"]],
                color=c, lw=1.9, alpha=0.85, zorder=3,
                solid_capstyle="round")
        ax.plot([x0], [r["scc_pct"]], "o", ms=8.5, color=c,
                markeredgecolor="#4d4d4d", markeredgewidth=0.7, zorder=4)
        ax.plot([x1], [r["int_pct"]], "o", ms=8.5, color=c,
                markeredgecolor="#4d4d4d", markeredgewidth=0.7, zorder=4)

    # median reference lines
    for x, med, lab, va in (
        (x0, scc_med, f"median  +{scc_med:.2f}%   SCC worsens", "bottom"),
        (x1, int_med, f"median  {int_med:.2f}%   intermittent improves", "top"),
    ):
        ax.plot([x - 0.13, x + 0.13], [med, med], color=ACCENT, lw=3.0,
                zorder=6, solid_capstyle="butt")
    ax.text(x0 - 0.16, scc_med, f"median +{scc_med:.2f}%", ha="right",
            va="center", color=ACCENT, fontsize=13.5, fontweight="bold",
            zorder=7)
    ax.text(x1 + 0.16, int_med, f"median {int_med:.2f}%", ha="left",
            va="center", color=ACCENT, fontsize=13.5, fontweight="bold",
            zorder=7)

    ax.set_xlim(-0.62, 1.62)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["SCC index\n(cumulative-dose pathway)",
                        "Intermittent index\n(BCC / melanoma pathway)"])
    ax.set_ylabel("Change from uniform to free spacing (%)")
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)

    lo, hi = ax.get_ylim()
    ax.text(-0.55, hi * 0.90, "worse", color="#4d4d4d", fontsize=13,
            style="italic", ha="left", va="center")
    ax.text(-0.55, lo * 0.90, "better", color="#4d4d4d", fontsize=13,
            style="italic", ha="left", va="center")

    handles = [Line2D([], [], marker="o", ls="", ms=9,
                      color=FST_COLORS[i], markeredgecolor="#4d4d4d",
                      markeredgewidth=0.7, label=f"FST {FST_LABELS[i]}")
               for i in range(6)]
    ax.legend(handles=handles, loc="lower center", ncol=6,
              bbox_to_anchor=(0.5, -0.34), handletextpad=0.3,
              columnspacing=1.1, fontsize=12.5)

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
