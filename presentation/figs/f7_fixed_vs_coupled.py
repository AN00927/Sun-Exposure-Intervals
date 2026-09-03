"""F7: what the adaptive skin states cost you over 180 days.

No CSV -- this is a live call into vitd_model.simulate for FST VI on a fixed
schedule (one session every 3 days, 4.0 SED each, 60 sessions, 240 SED total).
4.0 SED is 26.3% of the FST VI MED of 15.2 SED used by
vitd_model.compute_safety_sidecar, i.e. firmly sub-erythemal.

The two runs differ ONLY in whether the adaptive skin states are allowed to
move.  Both use documented arguments of simulate(); nothing in the model is
patched:

    coupled : default -- melanin M tans toward the per-FST ceiling and the
              stratum corneum thickens, both attenuating later sessions.
    pinned  : alpha_override=0.0, alpha_SC_override=0.0, k_damage_override=0.0
              -- the per-session melanin and SC update terms are zeroed, so M
              stays at M_baseline(6) and SC stays at 0 for the whole run.

Writes the figure and a caption text file alongside it.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from slidestyle import use_slide_style, ACCENT, FST_COLORS, TEXT_COLOR, pct  # noqa: E402
from vitd_model import simulate, ModelParams  # noqa: E402

OUT = os.path.join(HERE, "f7_fixed_vs_coupled.png")
CAPTION = os.path.join(HERE, "f7_fixed_vs_coupled_caption.txt")

FST = 6
DURATION = 180
INTERVAL_DAYS = 3
SED_PER_SESSION = 4.0
MED_FST_VI = 15.2   # SED, vitd_model.compute_safety_sidecar med_by_fst[6]
D0 = 25.0
SUFFICIENCY_FLOOR = 50.0


def crossing_day(t, D, threshold=SUFFICIENCY_FLOOR):
    """First day D(t) reaches threshold, linearly interpolated between the
    two straddling simulation samples."""
    t = np.asarray(t)
    D = np.asarray(D)
    idx = int(np.argmax(D >= threshold))
    assert D[idx] >= threshold, (
        f"F7: series never reaches {threshold} nmol/L over the simulated range"
    )
    if idx == 0:
        return float(t[0])
    t0, t1 = t[idx - 1], t[idx]
    D0_, D1_ = D[idx - 1], D[idx]
    frac = (threshold - D0_) / (D1_ - D0_)
    return float(t0 + frac * (t1 - t0))


def main():
    use_slide_style()
    params = ModelParams()
    schedule = [(float(d), SED_PER_SESSION)
                for d in range(0, DURATION, INTERVAL_DAYS)]

    coupled = simulate(FST, schedule, DURATION, params=params, D0=D0,
                       dose_framing="fixed_per_session_SED")
    pinned = simulate(FST, schedule, DURATION, params=params, D0=D0,
                      alpha_override=0.0,
                      alpha_SC_override=0.0,
                      k_damage_override=0.0,
                      dose_framing="fixed_per_session_SED")

    m_base = params.M_baseline(FST)
    assert np.allclose(pinned["M"], m_base), (
        "F7 assertion FAILED: melanin did not stay pinned at baseline "
        f"(range {pinned['M'].min()} .. {pinned['M'].max()}, "
        f"baseline {m_base})"
    )
    assert np.allclose(pinned["SC"], 0.0), (
        "F7 assertion FAILED: stratum corneum did not stay pinned at 0 "
        f"(range {pinned['SC'].min()} .. {pinned['SC'].max()})"
    )
    assert coupled["M"].max() > m_base, (
        "F7 assertion FAILED: coupled run showed no tanning at all"
    )
    assert len(schedule) == 60 and coupled["cumulative_SED"] == 60 * SED_PER_SESSION

    d_c = float(coupled["D"][-1])
    d_p = float(pinned["D"][-1])
    print(f"F7: FST VI, {len(schedule)} sessions x {SED_PER_SESSION} SED "
          f"({coupled['cumulative_SED']:.1f} SED total).")
    print(f"F7: day-{DURATION} 25(OH)D  coupled {d_c:.4f} nmol/L, "
          f"pinned {d_p:.4f} nmol/L, shortfall {d_p - d_c:.4f} nmol/L "
          f"({(d_p - d_c) / d_p * 100:.2f}% of the pinned value).")
    print(f"F7: melanin rose {coupled['M'].min():.4f} -> "
          f"{coupled['M'].max():.4f}; SC rose 0 -> {coupled['SC'].max():.4f}.")

    fig, ax = plt.subplots(figsize=(9, 5))

    t_p, D_p = pinned["t"], pinned["D"]
    t_c, D_c = coupled["t"], coupled["D"]
    D_c_on_p = np.interp(t_p, t_c, D_c)

    ax.fill_between(t_p, D_c_on_p, D_p, color=ACCENT, alpha=0.12,
                    linewidth=0, zorder=2)
    ax.plot(t_p, D_p, color="#4d4d4d", lw=2.6, zorder=4,
            label="If skin never adapted")
    ax.plot(t_c, D_c, color=FST_COLORS[FST - 1], lw=2.6, zorder=5,
            label="Real skin (tans and thickens)")

    ax.axhline(SUFFICIENCY_FLOOR, color="#9a9a9a", lw=1.2, ls=(0, (5, 4)), zorder=3)
    ax.text(2, SUFFICIENCY_FLOOR + 0.9, "50 nmol/L sufficiency floor",
            fontsize=12.5, color="#6f6f6f", va="bottom", ha="left")

    day_pinned = crossing_day(t_p, D_p)
    day_coupled = crossing_day(t_c, D_c)
    print(f"F7: crosses {SUFFICIENCY_FLOOR:.0f} nmol/L on day "
          f"{day_pinned:.4f} (pinned), day {day_coupled:.4f} (coupled).")
    for day, color, dy in ((day_pinned, "#4d4d4d", 3.0), (day_coupled, FST_COLORS[FST - 1], -3.6)):
        ax.plot([day], [SUFFICIENCY_FLOOR], "o", ms=7, color=color,
                markeredgecolor="white", markeredgewidth=1.0, zorder=6)
        ax.annotate(f"day {round(day):.0f}", xy=(day, SUFFICIENCY_FLOOR),
                    xytext=(day, SUFFICIENCY_FLOOR + dy),
                    ha="center", va="center", fontsize=12.5, fontweight="bold",
                    color=color, zorder=6)

    # session rug
    ylo = 20.0
    ax.vlines([d for d, _ in schedule], ylo, ylo + 1.3,
              color="#bdbdbd", lw=1.2, zorder=3)
    ax.text(2, ylo - 1.4,
            f"one {SED_PER_SESSION:.0f} SED session every {INTERVAL_DAYS} days",
            fontsize=12, color="#6f6f6f", ha="left", va="top")

    # the gap
    gap = d_p - d_c
    ax.annotate("", xy=(DURATION, d_c), xytext=(DURATION, d_p),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.8))
    ax.text(DURATION - 5, (d_c + d_p) / 2 - 0.9,
            f"{gap:.0f} nmol/L\nlost to adaptation",
            ha="right", va="center", fontsize=13, fontweight="bold",
            color=ACCENT)

    ax.set_xlim(0, DURATION + 3)
    ax.set_ylim(ylo - 4.5, 64)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_xlabel("Day")
    ax.set_ylabel("Serum 25(OH)D (nmol/L)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.13))

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")

    with open(CAPTION, "w", encoding="utf-8") as fh:
        fh.write(
            "F7 - f7_fixed_vs_coupled.png\n"
            "\n"
            f"FST VI, 180 days, one session every {INTERVAL_DAYS} days at a "
            f"fixed {SED_PER_SESSION:.1f} SED per session ({len(schedule)} "
            f"sessions, {coupled['cumulative_SED']:.1f} SED cumulative). "
            f"{SED_PER_SESSION:.1f} SED is "
            f"{SED_PER_SESSION / MED_FST_VI * 100:.1f}% of the FST VI MED of "
            f"{MED_FST_VI} SED used in vitd_model.compute_safety_sidecar, so "
            "every session is sub-erythemal.\n"
            "\n"
            "Both curves are vitd_model.simulate on the identical schedule "
            f"from D0 = {D0} nmol/L. They differ only in whether the adaptive "
            "skin states move: the grey curve pins them with the documented "
            "overrides alpha_override=0.0, alpha_SC_override=0.0, "
            "k_damage_override=0.0, holding melanin at M_baseline(6) = "
            f"{m_base:g} and stratum corneum at 0. The brown curve is the "
            "default, in which melanin tans "
            f"{coupled['M'].min():.4f} -> {coupled['M'].max():.4f} and the "
            f"stratum corneum thickens 0 -> {coupled['SC'].max():.4f}, both "
            "attenuating later sessions.\n"
            "\n"
            f"Day-{DURATION} 25(OH)D: {d_c:.4f} nmol/L coupled vs "
            f"{d_p:.4f} nmol/L pinned, a shortfall of {gap:.4f} nmol/L "
            f"({gap / d_p * 100:.2f}% of the pinned value). The same UV "
            "budget buys less as the skin adapts.\n"
            "\n"
            "Our estimate of how fast the outer skin layer thickens is still "
            "rough, so read the size of the gap as the mechanism, not a "
            "calibrated quantity.\n"
        )
    print(f"wrote {CAPTION}")


if __name__ == "__main__":
    main()
