"""F2: feasibility / cumulative-UV heatmap over FST x session budget.

Source: tables/primary_minimum_schedule.csv.
Two outputs, one per spacing mode:
  f2_feasibility_heatmap.png       spacing == 'uniform'  (9 / 36 infeasible)
  f2_feasibility_heatmap_free.png  spacing == 'free'     (7 / 36 infeasible)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidestyle import use_slide_style, FST_LABELS, GREY_INFEASIBLE, TEXT_COLOR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(REPO, "tables", "primary_minimum_schedule.csv")

N_SESSIONS = [52, 104, 146, 183, 260, 365]
FSTS = [1, 2, 3, 4, 5, 6]

CASES = [
    ("uniform", "f2_feasibility_heatmap.png", 9),
    ("free", "f2_feasibility_heatmap_free.png", 7),
]


def grid_for(df, spacing):
    sel = df[df["spacing"] == spacing]
    vals = np.full((len(FSTS), len(N_SESSIONS)), np.nan)
    feas = np.zeros((len(FSTS), len(N_SESSIONS)), dtype=bool)
    for i, fst in enumerate(FSTS):          # row 0 = FST I  (origin='lower')
        for j, n in enumerate(N_SESSIONS):
            row = sel[(sel["fst"] == fst) & (sel["n_sessions"] == n)]
            assert len(row) == 1, f"expected 1 row for fst={fst} n={n} {spacing}, got {len(row)}"
            ok = bool(row["feasible"].iloc[0])
            feas[i, j] = ok
            if ok:
                vals[i, j] = float(row["total_sed"].iloc[0])
    return vals, feas


def draw(spacing, outname, expect_infeasible):
    df = pd.read_csv(CSV)
    vals, feas = grid_for(df, spacing)

    n_infeasible = int((~feas).sum())
    assert n_infeasible == expect_infeasible, (
        f"F2 assertion FAILED ({spacing}): expected {expect_infeasible} "
        f"infeasible cells of {feas.size}, got {n_infeasible}"
    )
    print(f"F2 [{spacing}]: {n_infeasible} of {feas.size} cells infeasible - matches.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.grid(False)

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    im = ax.imshow(vals, origin="lower", aspect="auto", cmap="YlOrRd",
                   vmin=vmin, vmax=vmax,
                   extent=(-0.5, len(N_SESSIONS) - 0.5, -0.5, len(FSTS) - 0.5))

    for i in range(len(FSTS)):
        for j in range(len(N_SESSIONS)):
            if feas[i, j]:
                v = vals[i, j]
                # dark text on the pale end, white on the deep-red end
                frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if frac > 0.62 else TEXT_COLOR)
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=GREY_INFEASIBLE,
                                       edgecolor="white", linewidth=1.0,
                                       zorder=2))
                ax.plot([j - 0.26, j + 0.26], [i - 0.26, i + 0.26],
                        color="black", lw=2.0, zorder=3, solid_capstyle="round")
                ax.plot([j - 0.26, j + 0.26], [i + 0.26, i - 0.26],
                        color="black", lw=2.0, zorder=3, solid_capstyle="round")

    # thin white cell separators
    for j in range(len(N_SESSIONS) + 1):
        ax.axvline(j - 0.5, color="white", lw=1.0, zorder=4)
    for i in range(len(FSTS) + 1):
        ax.axhline(i - 0.5, color="white", lw=1.0, zorder=4)

    ax.set_xticks(range(len(N_SESSIONS)))
    ax.set_xticklabels([str(n) for n in N_SESSIONS])
    ax.set_yticks(range(len(FSTS)))
    ax.set_yticklabels([f"FST {FST_LABELS[f - 1]}" for f in FSTS])
    ax.set_xlabel("Sessions per year")
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Cumulative annual UV (SED)")
    cbar.outline.set_visible(False)

    # legend for the infeasible marker
    ax.add_patch(Rectangle((len(N_SESSIONS) - 0.5 + 0.12, -1.55), 0.28, 0.42,
                           facecolor=GREY_INFEASIBLE, edgecolor="none",
                           clip_on=False, zorder=5))
    ax.text(len(N_SESSIONS) - 0.5 + 0.26, -1.34, "x", ha="center", va="center",
            fontsize=11, fontweight="bold", color="black",
            clip_on=False, zorder=6)
    ax.text(len(N_SESSIONS) - 0.5 + 0.5, -1.34,
            "no feasible schedule", ha="left", va="center", fontsize=12.5,
            color=TEXT_COLOR, clip_on=False, zorder=6)

    out = os.path.join(HERE, outname)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    use_slide_style()
    for spacing, outname, expect in CASES:
        draw(spacing, outname, expect)


if __name__ == "__main__":
    main()
