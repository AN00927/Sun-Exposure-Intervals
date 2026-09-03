"""F4: where each schedule family falls on the two cancer-risk axes.

Source: tables/joint_schedule_optimization.csv. The table carries two
non-uniform schedule families -- two_phase (loading dose then a lower
maintenance dose: "front-loaded") and exponential (continuous taper) --
each compared against its own (fst, n_sessions)-matched two_phase,
free_spacing=False row ("evenly spaced"). Evenly spaced is the baseline
against which every percent change is computed, so it sits at the origin
by construction.

x = percent change in scc_index (long-term cumulative-dose risk)
y = percent change in intermittent_index (burn-driven risk)
both relative to the evenly-spaced schedule for the same (fst, n_sessions).
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidestyle import use_slide_style, ACCENT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(REPO, "tables", "joint_schedule_optimization.csv")
OUT = os.path.join(HERE, "f4_two_cancer_axes.png")

EXPONENTIAL_COLOR = "#b47239"  # FST IV -- reused as the exponential-family colour

EXPECT_UNIFORM_N = 22
EXPECT_FRONTLOAD_N = 18
EXPECT_EXPONENTIAL_N = 4
EXPECT_FRONTLOAD_SCC_MEDIAN = 7.65
EXPECT_FRONTLOAD_INT_MEDIAN = -23.50
EXPECT_EXPONENTIAL_SCC_MEDIAN = 7.27
EXPECT_EXPONENTIAL_INT_MEDIAN = -19.71


def vs_uniform(df, family):
    """Rows of `family` (always free_spacing=True) merged against the
    evenly-spaced (two_phase, free_spacing=False) row for the same
    (fst, n_sessions), keeping only pairs feasible on both sides."""
    uni = df[(df["family"] == "two_phase") & (~df["free_spacing"].astype(bool))]
    sel = df[(df["family"] == family) & (df["free_spacing"].astype(bool))]
    m = uni.merge(sel, on=["fst", "n_sessions"], suffixes=("_u", "_f"))
    m = m[m["feasible_u"].astype(bool) & m["feasible_f"].astype(bool)].copy()
    m["scc_pct"] = (m["scc_index_f"] - m["scc_index_u"]) / m["scc_index_u"] * 100.0
    m["int_pct"] = ((m["intermittent_index_f"] - m["intermittent_index_u"])
                    / m["intermittent_index_u"] * 100.0)
    return m


def main():
    use_slide_style()
    df = pd.read_csv(CSV)

    uniform_n = int(((df["family"] == "two_phase") & (~df["free_spacing"].astype(bool))
                     & df["feasible"].astype(bool)).sum())
    front = vs_uniform(df, "two_phase")
    expo = vs_uniform(df, "exponential")

    assert uniform_n == EXPECT_UNIFORM_N, (
        f"F4 assertion FAILED: {uniform_n} feasible evenly-spaced rows, "
        f"expected {EXPECT_UNIFORM_N}"
    )
    assert len(front) == EXPECT_FRONTLOAD_N, (
        f"F4 assertion FAILED: {len(front)} front-loaded pairs, expected "
        f"{EXPECT_FRONTLOAD_N}"
    )
    assert len(expo) == EXPECT_EXPONENTIAL_N, (
        f"F4 assertion FAILED: {len(expo)} exponential pairs, expected "
        f"{EXPECT_EXPONENTIAL_N}"
    )
    fl_scc_med, fl_int_med = float(front["scc_pct"].median()), float(front["int_pct"].median())
    ex_scc_med, ex_int_med = float(expo["scc_pct"].median()), float(expo["int_pct"].median())
    assert round(fl_scc_med, 2) == EXPECT_FRONTLOAD_SCC_MEDIAN
    assert round(fl_int_med, 2) == EXPECT_FRONTLOAD_INT_MEDIAN
    assert round(ex_scc_med, 2) == EXPECT_EXPONENTIAL_SCC_MEDIAN
    assert round(ex_int_med, 2) == EXPECT_EXPONENTIAL_INT_MEDIAN
    print(f"F4: evenly-spaced n={uniform_n}; front-loaded n={len(front)} "
          f"(median {fl_scc_med:.4f}%, {fl_int_med:.4f}%); exponential "
          f"n={len(expo)} (median {ex_scc_med:.4f}%, {ex_int_med:.4f}%).")

    fig, ax = plt.subplots(figsize=(8.2, 7.2))

    all_x = pd.concat([front["scc_pct"], expo["scc_pct"], pd.Series([0.0])])
    all_y = pd.concat([front["int_pct"], expo["int_pct"], pd.Series([0.0])])
    xpad = (all_x.max() - all_x.min()) * 0.18
    ypad = (all_y.max() - all_y.min()) * 0.12
    ax.set_xlim(all_x.min() - xpad, all_x.max() + xpad)
    ax.set_ylim(all_y.min() - ypad, max(all_y.max() + ypad, 6.0))

    ax.axhline(0.0, color="#9a9a9a", lw=1.0, zorder=1)
    ax.axvline(0.0, color="#9a9a9a", lw=1.0, zorder=1)
    ax.grid(True, color="#e6e6e6", linewidth=0.7, zorder=0)

    ax.plot(front["scc_pct"], front["int_pct"], "o", ms=10.5, color=ACCENT,
            markeredgecolor="#4d4d4d", markeredgewidth=0.8, alpha=0.9, zorder=4)
    ax.plot(expo["scc_pct"], expo["int_pct"], "^", ms=12, color=EXPONENTIAL_COLOR,
            markeredgecolor="#4d4d4d", markeredgewidth=0.8, alpha=0.9, zorder=4)
    ax.plot([0.0], [0.0], "s", ms=13, color="#8a8a8a",
            markeredgecolor="#4d4d4d", markeredgewidth=0.8, zorder=5)

    ax.set_xlabel("Change in long-term cumulative-dose risk (%)")
    ax.set_ylabel("Change in burn-driven risk (%)")

    handles = [
        Line2D([], [], marker="s", ls="", ms=11, color="#8a8a8a",
              markeredgecolor="#4d4d4d", markeredgewidth=0.8, label="Evenly spaced"),
        Line2D([], [], marker="o", ls="", ms=10, color=ACCENT,
              markeredgecolor="#4d4d4d", markeredgewidth=0.8, label="Front-loaded"),
        Line2D([], [], marker="^", ls="", ms=11, color=EXPONENTIAL_COLOR,
              markeredgecolor="#4d4d4d", markeredgewidth=0.8, label="Exponential taper"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.22), handletextpad=0.4,
              columnspacing=1.3, fontsize=12.5)

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
