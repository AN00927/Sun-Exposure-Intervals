import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT="#0584a4"; TEXT="#0e2a47"; GREYC="#9aa7b4"
# days to first reach 50 nmol/L, FST III, minimum-dose schedule
# source: tables/primary_minimum_schedule.csv (time_to_sufficiency)
N   = [104, 146, 183, 260, 365]
uni = [84.23, 85.00, 87.76, 88.44, 88.00]
fre = [73.18, 76.90, 78.40, 80.10, 81.42]

plt.rcParams.update({"figure.dpi":220,"savefig.dpi":220,"savefig.bbox":"tight",
                     "text.color":TEXT,"axes.labelcolor":TEXT,
                     "xtick.color":TEXT,"ytick.color":TEXT,"font.size":11})
fig, ax = plt.subplots(figsize=(5.0,3.0))
y = range(len(N))
for i,(u,f) in enumerate(zip(uni,fre)):
    ax.plot([f,u],[i,i], color=GREYC, lw=2.4, zorder=1, solid_capstyle="round")
    ax.scatter([u],[i], s=70, color=GREYC, zorder=3)
    ax.scatter([f],[i], s=70, color=ACCENT, zorder=3)
    ax.annotate(f"−{u-f:.0f} d", ((u+f)/2, i+0.30), ha="center", va="top",
                fontsize=9.5, color=ACCENT, fontweight="bold")
ax.scatter([],[], s=70, color=GREYC, label="Evenly spaced")
ax.scatter([],[], s=70, color=ACCENT, label="Front-loaded")
ax.set_yticks(list(y)); ax.set_yticklabels([str(n) for n in N])
ax.set_ylabel("Sessions per year")
ax.set_xlabel("Days to first reach 50 nmol/L")
ax.set_xlim(68, 94); ax.invert_yaxis()
ax.legend(frameon=False, loc="lower left", fontsize=9.5, ncols=2,
          bbox_to_anchor=(0.0, 1.01), handletextpad=0.3, columnspacing=1.4)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GREYC)
ax.tick_params(length=0)
ax.grid(axis="x", color="#e6ebef", lw=0.8, zorder=0)
ax.set_axisbelow(True)
fig.savefig("f8_time_to_sufficiency.png")
print("saved")
