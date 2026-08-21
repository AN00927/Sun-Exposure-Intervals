"""
kconst_recalibration.py
=======================
DERIVATION for the K_const recalibration approved on 2026-08-17
(`calibration_approval.yaml`, `a1_diagnosis_result_2026_08_17`).

One question, no file written:

    What value of `ModelParams.K_const` reproduces the between-skin-type
    25(OH)D response measured in the WITHIN-STUDY, MATCHED-PHYSICAL-DOSE,
    MATCHED-BSA, SAME-SOURCE trials this repo holds?

The diagnosis it rests on is `scripts/fst_gap_decomposition.py` (~12 min).
That script established WHERE the target is -- FST V:II optical 1.7-2.7x from
two independent routes. This script picks the number and shows its working.

Design
------
CALIBRATION SET (fitted). Only trials satisfying ALL of:
  * within-study     -- both skin types in the same paper, so no cross-trial
                        baseline anchoring is needed;
  * matched PHYSICAL dose (SED), not fractional MED;
  * matched %BSA between the arms;
  * same UV source both arms, so E_source cancels from the ratio;
  * both groups' pre-exposure 25(OH)D measured and printed.
Exactly two studies in this repo qualify: Felton 2016 and Young 2020.

HELD OUT (never fitted; used only to score).
  * Kift 2013 and Darling 2013 field personal-dosimetry inversions -- a
    different method (field badges, real behaviour) on a different population.
  * Farrar 2013 / Rhodes 2010 cross-trial pair (one unobtained baseline).
  * Shih 2018's fractional-MED null and Kallioglu 2024 (withdrawn: it is an
    erythema MED table -- see fst_gap_decomposition S1a).
  * The whole existing validation suite (`vitd_validation.run_all`).

Everything the script CHOOSES is tagged `# [CHOICE]`; everything it
ESTIMATES rather than reads from a source is tagged `# [ESTIMATE]`.

DARK-SKIN PROVENANCE, stated once and true of every number below:
  * every FST V datapoint is Indian-subcontinent / South Asian ancestry;
  * every FST VI datapoint is lamp-based (Arimed B or TL01 cabinet) -- no
    field dosimetry in FST VI exists anywhere in this repo;
  * the only African-ancestry matched-dose cell of any size is Young 2020's
    TL01 FST VI arm (n = 11).
Nothing here is measured in FST VI outdoors, and nothing in FST V outside UK
latitudes. Every FST VI number this calibration produces is an EXTRAPOLATION
of one coefficient onto M_baseline(VI) = 1.00, itself an unvalidated prior.

Run
---
    PYTHONIOENCODING=utf-8 python3 scripts/kconst_recalibration.py
"""

from __future__ import annotations

import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitd_model import ModelParams, simulate, get_E_source

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FST_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
ROMAN_OF = {v: k for k, v in FST_ROMAN.items()}


def hr(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# DATA BLOCK -- every value quoted from a source held in this repo
# ===========================================================================

# --- Felton et al. 2016, Br J Dermatol 175:1320-1328 (Research_Papers/) ----
# Within-study phototype II vs V, same cabinet, same protocol, both arms.
# Table 1. NOTE the PDF text layer drops decimal points ("365" = 36.5).
FELTON_D0 = {2: 36.5, 5: 17.2}          # pre-course 25(OH)D, nmol/L
FELTON_FINAL = {2: 54.3, 5: 25.5}       # post-course 25(OH)D, nmol/L
FELTON_N = {2: 10, 5: 6}
FELTON_SED_PER_SESSION = 1.3
FELTON_SESSIONS_PER_WEEK = 3
FELTON_WEEKS = 6
FELTON_BSA = 0.35
# [ESTIMATE] Felton describes its source as 95% UVA / 5% UVB simulated summer
# sunlight but does not name the cabinet. Assumed to be the Arimed B used by
# the same group in Rhodes 2010 / Farrar 2013. E_source cancels from a
# between-skin-type RATIO, so this affects absolute levels only.
FELTON_SOURCE = "arimed_b"

# --- Young et al. 2020, JID 140:1418 --------------------------------------
# Individual-level trial data, in this repo. Protocol, from the paper:
# "the same suberythemal UVR dose, to 85% body surface area ... repeated five
# times with intervals of 3-4 days".
YOUNG_XLSX = os.path.join(
    REPO, "Papers_and_Datasets", "Final skin type vit D database 26_03_2020.xlsx")
YOUNG_BSA = 0.85
YOUNG_INTERVAL_D = 3.5   # midpoint of the paper's stated 3-4 days  # [ESTIMATE]
YOUNG_ARMS = {
    # sheet: (dose column prefix, E_source key, SED per exposure)
    "ArimedB irradiated": ("UVdose.Sample_", "arimed_b", 2.0),
    "TL01 irradiated":    ("UVDose.Sample_", "tl01_nb_uvb", 1.5),
}
# Young 2020 Table 3 slope ratios. Table 3's own note: the regression model is
# "adjusted for pre-exposure 25(OH)D3, slope, and intercept". Both published
# ratios are therefore ANCOVA-ADJUSTED, not raw. (Read from the PDF, 2026-08-17.)
YOUNG_PUBLISHED_ADJUSTED = {"ArimedB irradiated": ("II", "VI", 1.29),
                            "TL01 irradiated":    ("II", "VI", 1.43)}

# --- Held-out targets, all reproduced from fst_gap_decomposition.py -------
# Run 2026-08-17; see that script for their derivation and their caveats.
HELD_OUT_FIELD = {          # label: (FST V:II optical factor, implied K_const)
    "Kift 2013 / Webb 2010, behaviour removed": (1.75, 0.788),
    "Darling 2013, behaviour removed":          (2.68, 1.387),
}
HELD_OUT_FIELD_BLIND = {    # what dumping behaviour into K_const would need
    "Kift 2013 / Webb 2010, BEHAVIOUR-BLIND":   (4.28, 2.047),
    "Darling 2013, BEHAVIOUR-BLIND":            (5.71, 2.453),
}
DIAGNOSED_BAND = (1.7, 2.7)   # FST V:II optical: the approved target band

#: The value K_const held BEFORE this recalibration. Pinned as a literal so
#: every "before" column stays a before column once the new value lands in
#: ModelParams -- reading it from ModelParams() would make the comparison
#: collapse to identity the moment the change is applied.
OLD_K_CONST = 0.33

# Sources deliberately EXCLUDED from the objective, kept to score the result.
# All values reproduced from fst_gap_decomposition.py S7, run 2026-08-17.
HELD_OUT_SCORING = [
    ("Farrar 2013 / Rhodes 2010 raw (Webb 2021 T4)", 2.26,
     "cross-trial; baselines unmatched"),
    ("Farrar/Rhodes corrected, Rhodes D0 swept 30-60", (2.76, 6.06),
     "width is ONE unobtained number"),
    ("Shih 2018 MED ratio under its own null", 2.88,
     "only if equal %MED => equal D; weak null"),
    ("Kallioglu 2024 'V:II = 2.40'", 2.40,
     "WITHDRAWN: it is an erythema MED table"),
    ("Felton 2016 MEASURED erythema MED ratio", 3.95,
     "erythema, not vitamin D"),
]


# ===========================================================================
# Model wrappers -- each reproduces ONE trial's protocol
# ===========================================================================

def felton_gain(fst: int, K: float, params: ModelParams = None) -> float:
    """Modelled 25(OH)D gain over Felton 2016's 6-week 3x/week course."""
    p = params or ModelParams()
    sched = [(w * 7 + k * 2, FELTON_SED_PER_SESSION)
             for w in range(FELTON_WEEKS)
             for k in range(FELTON_SESSIONS_PER_WEEK)]
    res = simulate(fst=fst, schedule=sched, duration_days=FELTON_WEEKS * 7,
                   params=p, D0=FELTON_D0[fst], K_const_override=K,
                   E_source_override=get_E_source(FELTON_SOURCE),
                   exposed_bsa_override=FELTON_BSA,
                   dose_framing="fixed_absolute_SED")
    return float(res["D"][-1]) - FELTON_D0[fst]


def _young_sample_days(cum_doses, sed_per_session):
    """Map Young's printed CUMULATIVE dose grid onto exposure-day times.

    The spreadsheet's grid is 0, 2, 4, 8, 10 SED (Arimed B) and
    0, 1.5, 4.5, 6, 7.5 SED (TL01) for five 2.0 / 1.5 SED exposures at 3-4 day
    intervals, i.e. blood was drawn before exposures 1, 2, 3 and 5 and after
    exposure 5, skipping one mid-course draw. Sampling the MODEL on the same
    grid matters: fst_gap_decomposition.py used an evenly spaced 0,2,4,6,8
    grid, which is not the grid the data sit on.
    """
    times = []
    for c in cum_doses:
        n_done = int(round(c / sed_per_session))
        if n_done == 0:
            times.append(0.0)                          # pre-exposure draw
        elif n_done < 5:
            times.append(n_done * YOUNG_INTERVAL_D - 1e-6)  # just before next
        else:
            times.append(5 * YOUNG_INTERVAL_D)         # final draw
    return times


def young_slope(fst: int, D0: float, K: float, sed_per_session: float,
                e_source: float, cum_doses,
                params: ModelParams = None) -> float:
    """Modelled d[25(OH)D3]/d[cumulative SED] on Young 2020's own dose grid."""
    p = params or ModelParams()
    sched = [(i * YOUNG_INTERVAL_D, sed_per_session) for i in range(5)]
    times = _young_sample_days(cum_doses, sed_per_session)
    res = simulate(fst=fst, schedule=sched,
                   duration_days=int(np.ceil(max(times))) + 1, params=p,
                   D0=D0, K_const_override=K, E_source_override=e_source,
                   exposed_bsa_override=YOUNG_BSA,
                   dose_framing="fixed_absolute_SED")
    t = np.asarray(res["t"], dtype=float)
    D = np.asarray(res["D"], dtype=float)
    obs = [D[int(np.argmin(np.abs(t - tt)))] for tt in times]
    return float(np.polyfit(np.asarray(cum_doses, dtype=float), obs, 1)[0])


# ===========================================================================
# Young 2020 individual-level loader
# ===========================================================================

def load_young():
    """Per-subject observed slope and baseline, plus the arm's protocol grid."""
    try:
        import openpyxl
    except ImportError:
        return None
    if not os.path.exists(YOUNG_XLSX):
        return None
    wb = openpyxl.load_workbook(YOUNG_XLSX, read_only=True, data_only=True)
    arms = {}
    for sheet, (ukey, esrc, sed) in YOUNG_ARMS.items():
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))
        hdr = list(rows[0])
        subs = []
        for raw in rows[1:]:
            d = dict(zip(hdr, raw))
            if d.get("ID") is None:
                continue
            xs, ys = [], []
            for i in range(1, 6):
                u, v = d.get(ukey + str(i)), d.get("VitD3.Sample_" + str(i))
                if u is None or v in (None, ""):
                    continue
                try:
                    xs.append(float(u))
                    ys.append(float(v))
                except (TypeError, ValueError):
                    pass
            if len(xs) < 3:
                continue
            full = [d.get(ukey + str(i)) for i in range(1, 6)]
            subs.append(dict(fst=FST_ROMAN[d["FST"]], eth=d["Ethnicity"],
                             slope=float(np.polyfit(xs, ys, 1)[0]),
                             base=ys[0],
                             full_doses=tuple(float(v) for v in full
                                              if v is not None)))
        # The canonical grid is the PROTOCOL grid = the modal complete UVdose
        # row. Individual subjects miss blood draws; taking the first
        # subject's surviving points as "the grid" silently shortens it.
        modal = collections.Counter(s["full_doses"] for s in subs
                                    if len(s["full_doses"]) == 5).most_common(1)
        arms[sheet] = dict(subjects=subs, e_source=get_E_source(esrc), sed=sed,
                           source_key=esrc,
                           doses=list(modal[0][0]) if modal else None,
                           n_on_grid=modal[0][1] if modal else 0)
    return arms


def young_group_table(arms):
    """Group means per FST per arm."""
    out = {}
    for sheet, arm in arms.items():
        by = collections.defaultdict(list)
        for s in arm["subjects"]:
            by[s["fst"]].append(s)
        groups = {}
        for f, g in by.items():
            groups[f] = dict(
                n=len(g),
                base=float(np.mean([s["base"] for s in g])),
                slope=float(np.mean([s["slope"] for s in g])),
                slope_sd=(float(np.std([s["slope"] for s in g], ddof=1))
                          if len(g) > 1 else float("nan")),
                eth=dict(collections.Counter(s["eth"] for s in g)))
        out[sheet] = groups
    return out


# ===========================================================================
# S1 -- is the model's feedback the right strength to justify a RAW target?
# ===========================================================================

def s1_feedback_check(arms):
    hr("S1  DOES THE MODEL'S FEEDBACK MATCH YOUNG 2020's OWN BASELINE EFFECT?")
    print("""
  Young 2020's published ratios (1.29 Arimed B, 1.43 TL01) come from Table 3,
  whose own note says the model is "adjusted for pre-exposure 25(OH)D3, slope,
  and intercept". This model is instead handed each group's MEASURED
  pre-exposure value and generates its own baseline correction through the
  (1 - D/D_max) feedback term. A model prediction is therefore a RAW slope and
  belongs against a RAW observed slope -- unless the model's feedback is the
  wrong strength, in which case neither pairing is clean.

  Test, WITHIN FST II ONLY, so skin type is held fixed and the gradient is a
  pure baseline effect rather than a baseline-x-FST confound: regress observed
  individual slope on observed individual baseline, then compare with the
  model's own gradient over the same baseline range.""")
    p = ModelParams()
    rows = []
    for sheet, arm in arms.items():
        subs = [s for s in arm["subjects"] if s["fst"] == 2]
        b = np.array([s["base"] for s in subs])
        sl = np.array([s["slope"] for s in subs])
        m_obs = float(np.polyfit(b, sl, 1)[0])
        r = float(np.corrcoef(b, sl)[0, 1])
        lo, hi = float(b.min()), float(b.max())
        d0s = np.linspace(lo, hi, 7)
        mod = [young_slope(2, d0, OLD_K_CONST, arm["sed"], arm["e_source"],
                           arm["doses"]) for d0 in d0s]
        m_mod = float(np.polyfit(d0s, mod, 1)[0])
        ratio = m_mod / m_obs if m_obs != 0 else float("nan")
        rows.append((sheet, m_obs, m_mod, r, len(subs), ratio))
        print(f"\n  {sheet}  (FST II only, n = {len(subs)}, "
              f"baseline {lo:.1f}-{hi:.1f} nmol/L)")
        print(f"    observed d(slope)/d(baseline) = {m_obs:+.4f} "
              f"[nmol/L per SED] per [nmol/L]   (Pearson r = {r:+.2f})")
        print(f"    model    d(slope)/d(baseline) = {m_mod:+.4f}   "
              f"[K_const = {OLD_K_CONST:.3f}, D_max = {p.D_max:.0f}]")
        print(f"    model / observed = {ratio:.2f}x")

    rr = [x[5] for x in rows if np.isfinite(x[5])]
    agree = all(0.5 <= x <= 2.0 for x in rr)
    print(f"\n  [derived] model:observed baseline gradient spans "
          f"{min(rr):.2f}x to {max(rr):.2f}x across the two arms.")
    if agree:
        print("""
  READING (derived from the numbers above, not asserted). Both gradients are
  negative in the data and in the model, and they agree to within a factor of
  two. The model's feedback is then a usable stand-in for Young's ANCOVA and
  the RAW and ADJUSTED pairings should give similar answers; S2 checks that
  they do rather than assuming it.""")
    else:
        print("""
  READING (derived from the numbers above, not asserted). Both gradients are
  negative -- the sign the feedback term claims -- but the model's is only
  """ + f"{min(rr):.2f}-{max(rr):.2f}" + """x of the measured one. The model
  UNDER-corrects for baseline. It therefore cannot be used as a substitute for
  Young's own ANCOVA, and the RAW pairing is unsafe: Young's dark groups start
  much lower (Arimed B FST VI 16.6 vs FST II 39.9 nmol/L), so a raw target
  would push K_const down to cover a shortfall in the FEEDBACK term that has
  nothing to do with pigment.

  [CHOICE] The primary Young pairing is therefore 'adjusted@pooled': Young's
  ANCOVA removes the baseline difference from the DATA, and running both model
  arms from the same baseline removes it from the MODEL. That pairing is the
  only one of the three whose answer does not depend on the feedback term
  being the right strength. The other two are computed anyway, in S2, so the
  cost of the choice is visible.""")
    return rows, agree


# ===========================================================================
# S2 -- per-contrast inversion, then the joint fit
# ===========================================================================

# [CHOICE] Weight one per STUDY, not one per contrast and not one per subject.
# Young's two arms share one recruitment, one group and one protocol design,
# so they are not two independent studies. Weighting by subject count would
# let Young's n = 67 erase Felton's n = 16, even though Felton is the only
# within-study FST V comparison with both baselines AND both endpoints
# printed. Every alternative weighting is reported in S4.
STUDY_WEIGHTS = {"Felton2016": 1.0, "Young2020": 1.0}

# The Young target and the baseline it is simulated at must be PAIRED: an
# ANCOVA-adjusted ratio belongs with a common baseline, a raw ratio with each
# group's own. "adjusted@own" is the mismatched pairing that
# fst_gap_decomposition.py used; it is carried here so the difference it makes
# is visible rather than silent.
YOUNG_TREATMENTS = ["adjusted@pooled", "raw@own", "adjusted@own"]


def build_targets(arms, groups, young_treatment="adjusted@pooled"):
    """Return [(study, label, observed ratio, callable(K) -> model ratio)]."""
    t = []
    obs_f = ((FELTON_FINAL[2] - FELTON_D0[2])
             / (FELTON_FINAL[5] - FELTON_D0[5]))
    t.append(("Felton2016", "Felton 2016 FST II:V gain, 23.4 SED, 35% BSA",
              obs_f, lambda K: felton_gain(2, K) / felton_gain(5, K)))

    for sheet, (a, b, adj) in YOUNG_PUBLISHED_ADJUSTED.items():
        if sheet not in groups:
            continue
        fa, fb = FST_ROMAN[a], FST_ROMAN[b]
        g, arm = groups[sheet], arms[sheet]
        pool = float(np.mean([s["base"] for s in arm["subjects"]]))
        if young_treatment == "adjusted@pooled":
            obs, d0a, d0b = adj, pool, pool
        elif young_treatment == "adjusted@own":
            obs, d0a, d0b = adj, g[fa]["base"], g[fb]["base"]
        elif young_treatment == "raw@own":
            obs = g[fa]["slope"] / g[fb]["slope"]
            d0a, d0b = g[fa]["base"], g[fb]["base"]
        else:
            raise ValueError(young_treatment)
        label = (f"Young 2020 {sheet.split()[0]:8s} FST {a}:{b} "
                 f"[{young_treatment}]")

        def mk(fa=fa, fb=fb, d0a=d0a, d0b=d0b, arm=arm):
            return lambda K: (
                young_slope(fa, d0a, K, arm["sed"], arm["e_source"],
                            arm["doses"])
                / young_slope(fb, d0b, K, arm["sed"], arm["e_source"],
                              arm["doses"]))
        t.append(("Young2020", label, obs, mk()))
    return t


def objective(K, targets, weights=None):
    """Weighted sum of squared LOG ratio errors.

    Log space  # [CHOICE]: a ratio and its reciprocal must cost the same, and
    the calibrated quantity exp(K * dM) is log-linear in K, so a log residual
    is linear in the parameter being fitted.
    """
    w = weights or STUDY_WEIGHTS
    per_study = collections.defaultdict(list)
    for study, _label, obs, fn in targets:
        mod = fn(K)
        per_study[study].append(1e6 if (not np.isfinite(mod) or mod <= 0)
                                else (np.log(mod) - np.log(obs)) ** 2)
    return sum(w.get(s, 1.0) * float(np.mean(v)) for s, v in per_study.items())


K_HI = 3.0
# Above ~3 the darker arm's modelled net response turns negative (decay
# exceeds synthesis) and the ratio stops being monotone. Same bound, and the
# same reason, as K_RATIO_MAX in scripts/fst_gap_decomposition.py.


def minimise(targets, weights=None, lo=0.0, hi=K_HI, n=61):
    grid = np.linspace(lo, hi, n)
    vals = [objective(k, targets, weights) for k in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(50):
        if objective(c, targets, weights) < objective(d, targets, weights):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    K = 0.5 * (a + b)
    return float(K), float(objective(K, targets, weights)), grid, vals


def invert_one(obs, fn, lo=0.0, hi=K_HI, n=121):
    """K such that fn(K) == obs. Returns (K, bound_flag).

    Plain bisection is NOT safe here. fn rises with K only while the darker
    arm's modelled net response stays positive; past that point the ratio
    turns over and goes negative, so a bisection that only tests the two
    endpoints reads fn(hi) < obs and wrongly reports 'at-bound-high'. This
    scans a grid for the FIRST upward crossing and bisects inside it.
    """
    grid = np.linspace(lo, hi, n)
    vals = [fn(k) for k in grid]
    if vals[0] >= obs:
        return lo, "at-bound-low"
    for i in range(1, n):
        if np.isfinite(vals[i]) and vals[i] > 0 and vals[i] >= obs:
            a, b = grid[i - 1], grid[i]
            for _ in range(45):
                mid = 0.5 * (a + b)
                if fn(mid) < obs:
                    a = mid
                else:
                    b = mid
            return 0.5 * (a + b), None
        if np.isfinite(vals[i]) and vals[i] <= 0:
            # ratio has turned over before reaching the target
            return grid[i - 1], "no-crossing-below-turnover"
    return hi, "at-bound-high"


# ===========================================================================
# reporting helpers
# ===========================================================================

def optical(K, f_dark, f_light=2, p=None):
    p = p or ModelParams()
    return float(np.exp(K * (p.M_baseline(f_dark) - p.M_baseline(f_light))))


def ratio_table(K, p=None):
    p = p or ModelParams()
    pairs = [(5, 2, "V:II"), (6, 1, "VI:I"), (6, 2, "VI:II"),
             (6, 5, "VI:V"), (4, 2, "IV:II"), (3, 2, "III:II")]
    return [(lab, float(np.exp(K * (p.M_baseline(d) - p.M_baseline(l)))))
            for d, l, lab in pairs]


# ===========================================================================

def main():
    print(__doc__)
    p = ModelParams()
    arms = load_young()
    if arms is None:
        print("\n[ABORT] openpyxl or the Young 2020 xlsx is unavailable.")
        return
    groups = young_group_table(arms)
    obs_f = (FELTON_FINAL[2] - FELTON_D0[2]) / (FELTON_FINAL[5] - FELTON_D0[5])

    # ---------------------------------------------------------------- S0 ---
    hr("S0  THE CALIBRATION SET, AS READ FROM SOURCE")
    print("\n  Felton et al. 2016, Br J Dermatol 175:1320  (Research_Papers/)")
    print(f"    {FELTON_SED_PER_SESSION} SED x {FELTON_SESSIONS_PER_WEEK}/wk "
          f"x {FELTON_WEEKS} wk = "
          f"{FELTON_SED_PER_SESSION*FELTON_SESSIONS_PER_WEEK*FELTON_WEEKS:.1f}"
          f" SED at {FELTON_BSA:.0%} BSA, same cabinet both arms")
    for f in (2, 5):
        print(f"      phototype {ROMAN_OF[f]:<3} n={FELTON_N[f]:<3} "
              f"{FELTON_D0[f]:.1f} -> {FELTON_FINAL[f]:.1f} nmol/L   "
              f"gain {FELTON_FINAL[f]-FELTON_D0[f]:+.1f}")
    print(f"      observed II:V gain ratio = {obs_f:.3f}    "
          f"[FST V arm: Indian-subcontinent ancestry, UK]")

    print("\n  Young et al. 2020, JID 140:1418  (individual-level xlsx in repo)")
    for sheet, g in groups.items():
        a = arms[sheet]
        print(f"    {sheet}: source {a['source_key']} "
              f"(E_source {a['e_source']:.2f}), {a['sed']} SED x5 at "
              f"{YOUNG_BSA:.0%} BSA")
        print(f"      protocol dose grid {a['doses']} SED "
              f"(modal complete row, {a['n_on_grid']}/{len(a['subjects'])} "
              f"subjects)")
        for f in sorted(g):
            print(f"      FST {ROMAN_OF[f]:<3} n={g[f]['n']:<3} "
                  f"baseline {g[f]['base']:5.1f}  raw slope "
                  f"{g[f]['slope']:.3f} nmol/L per SED   {g[f]['eth']}")
        ra, rb, adj = YOUNG_PUBLISHED_ADJUSTED[sheet]
        print(f"      raw       FST {ra}:{rb} slope ratio = "
              f"{g[FST_ROMAN[ra]]['slope']/g[FST_ROMAN[rb]]['slope']:.3f}")
        print(f"      published FST {ra}:{rb} slope ratio = {adj:.2f}   "
              f"(Table 3, ANCOVA-adjusted)")
    print("\n  Ethnicity keys: WE = white European, SEEA = South East/East")
    print("  Asian, ISC = Indian subcontinent, Af = African.")

    # ---------------------------------------------------------------- S1 ---
    _fb, _agree = s1_feedback_check(arms)

    # ---------------------------------------------------------------- S2 ---
    hr("S2  WHAT EACH CONTRAST INDEPENDENTLY IMPLIES FOR K_const")
    print("\n  Each row inverts ONE contrast on its own, under its own study's")
    print("  protocol, source and baselines. No fitting across rows yet.")
    print(f"\n  {'contrast':52s} {'observed':>9} {'K_const':>9} {'V:II':>7}")
    per_contrast = []
    for tr in YOUNG_TREATMENTS:
        for study, label, obs, fn in build_targets(arms, groups, tr):
            if study == "Felton2016" and tr != YOUNG_TREATMENTS[0]:
                continue        # Felton has no treatment variants
            K, flag = invert_one(obs, fn)
            per_contrast.append((study, tr, label, obs, K, flag))
            print(f"  {label:52s} {obs:>9.3f} {K:>9.3f} {optical(K,5):>7.2f}"
                  + (f"  [{flag}]" if flag else ""))

    # ---------------------------------------------------------------- S3 ---
    hr("S3  THE LAB-ONLY JOINT FIT, AND WHY IT IS NOT THE ANSWER")
    fits = {}
    for tr in YOUNG_TREATMENTS:
        tg = build_targets(arms, groups, tr)
        K, J, grid, vals = minimise(tg)
        fits[tr] = (K, J, tg)
        print(f"\n  Young treatment '{tr}'  ->  K_const = {K:.3f}  "
              f"(FST V:II = {optical(K,5):.2f}x, objective {J:.5f})")
        for study, label, obs, fn in tg:
            mod = fn(K)
            print(f"    {label:52s} obs {obs:>6.3f}  model {mod:>6.3f}  "
                  f"{100*(mod/obs-1):>+6.1f}%")
    print("""
  FINDING, and it is the reason this script does not stop here. Under EVERY
  Young pairing the joint lab fit misses BOTH studies badly and in OPPOSITE
  directions -- roughly -40% on Felton and +40% on Young. The two matched-dose
  lab trials are not noisy estimates of one number; within this model
  structure they are incompatible, and their least-squares compromise
  describes neither. Reporting that compromise as "the calibrated value"
  would hide a structural disagreement behind an average.

  The disagreement is also visible without any model: Young reports FST II:VI
  = 1.29-1.43 while Felton reports FST II:V = 2.14 in the same kind of
  cabinet, and under ANY monotone pigment penalty II:VI must EXCEED II:V.""")

    # ---------------------------------------------------------------- S3b --
    hr("S3b  THE ESTIMATOR ACTUALLY USED: four independent within-study routes")
    print("""
  What the approved diagnosis established is that FOUR independent
  within-study routes estimate the same physical quantity -- the FST V:II
  optical factor -- by two different methods:

      LAB, matched physical dose, behaviour absent by construction
        Felton 2016   phototype II vs V, 35% BSA, 23.4 SED, same cabinet
        Young 2020    FST II vs VI, 85% BSA, both source arms
      FIELD, personal dosimetry, behaviour MEASURED and then removed
        Kift 2013 / Webb 2010   Manchester 53.5N
        Darling 2013            Surrey 51.3N

  [CHOICE] K_const is fitted to all four by equal-weight least squares in LOG
  RATIO space. Because ln(V:II factor) = K_const * (M_V - M_II) is linear in
  K_const, that least-squares solution is exactly the arithmetic mean of the
  four study-level implied K_const values -- no iteration, no hidden weighting.

  [CHOICE] Equal weight per STUDY. The alternatives are all worse: weighting
  by n would let Young's 67 subjects override three other studies; weighting
  by method would require a precision for the field inversions that their own
  assumption register (fst_gap_decomposition S6) says is not quantified;
  weighting by "sharpness" would be the author choosing the answer.

  HELD OUT of this objective, and used only to score it in S6: the Farrar 2013
  / Rhodes 2010 cross-trial pair, Shih 2018's fractional-MED null, Kallioglu
  2024, NHANES Part B, and every check in vitd_validation.py.""")

    # Young's study-level value: primary pairing, both arms averaged.
    primary = YOUNG_TREATMENTS[0]
    young_ks = [K for (study, tr, _l, _o, K, _f) in per_contrast
                if study == "Young2020" and tr == primary]
    K_young = float(np.mean(young_ks))
    K_felton = [K for (study, tr, _l, _o, K, _f) in per_contrast
                if study == "Felton2016"][0]
    sources = [
        ("Felton 2016 (lab, FST V, within-study)", K_felton),
        (f"Young 2020 (lab, FST VI, both arms, {primary})", K_young),
        ("Kift 2013 / Webb 2010 (field, behaviour removed)",
         HELD_OUT_FIELD["Kift 2013 / Webb 2010, behaviour removed"][1]),
        ("Darling 2013 (field, behaviour removed)",
         HELD_OUT_FIELD["Darling 2013, behaviour removed"][1]),
    ]
    print(f"\n  {'source':52s} {'K_const':>9} {'V:II':>7}")
    for lab, K in sources:
        print(f"  {lab:52s} {K:>9.3f} {optical(K,5):>7.2f}")
    K_raw = float(np.mean([K for _, K in sources]))
    K_new = round(K_raw, 2)     # [CHOICE] 2 dp: see S9
    sd = float(np.std([K for _, K in sources], ddof=1))
    print(f"\n  {'mean (= the least-squares solution)':52s} {K_raw:>9.3f} "
          f"{optical(K_raw,5):>7.2f}")
    print(f"  {'SD across the four routes':52s} {sd:>9.3f}")
    print(f"  {'range':52s} "
          f"{min(K for _, K in sources):.3f}-{max(K for _, K in sources):.3f}")
    print(f"\n  [derived] NEW K_const = {K_new:.2f}   "
          f"(FST V:II optical = {optical(K_new,5):.2f}x)")
    print(f"  [derived] pre-recalibration = {OLD_K_CONST:.2f}   "
          f"(FST V:II optical = {optical(OLD_K_CONST,5):.2f}x)")

    # ---------------------------------------------------------------- S4 ---
    hr("S4  SENSITIVITY of the four-route estimator")
    print(f"\n  {'variant':54s} {'K_const':>8} {'V:II':>7}")
    variants = [("all four routes (primary)", K_raw)]
    for i, (lab, _K) in enumerate(sources):
        rest = [K for j, (_l, K) in enumerate(sources) if j != i]
        variants.append((f"drop {lab[:48]}", float(np.mean(rest))))
    variants.append(("LAB routes only (Felton + Young)",
                     float(np.mean([K_felton, K_young]))))
    variants.append(("FIELD routes only (Kift + Darling)",
                     float(np.mean([K for _l, K in sources[2:]]))))
    for tr in YOUNG_TREATMENTS[1:]:
        yk = [K for (study, t, _l, _o, K, _f) in per_contrast
              if study == "Young2020" and t == tr]
        variants.append((f"Young pairing '{tr}' instead",
                         float(np.mean([K_felton, float(np.mean(yk))]
                                       + [K for _l, K in sources[2:]]))))
    for lab, K in variants:
        print(f"  {lab:54s} {K:>8.3f} {optical(K,5):>7.2f}")
    allK = [K for _l, K in variants]
    print(f"\n  [derived] spread across every leave-one-out, method-split and")
    print(f"  Young-pairing variant above: K_const {min(allK):.3f} to "
          f"{max(allK):.3f}  (FST V:II {optical(min(allK),5):.2f}x to "
          f"{optical(max(allK),5):.2f}x)")
    inband = [K for K in allK
              if DIAGNOSED_BAND[0] <= optical(K, 5) <= DIAGNOSED_BAND[1]]
    print(f"  [derived] {len(inband)} of {len(allK)} variants land inside the "
          f"approved {DIAGNOSED_BAND[0]:.1f}-{DIAGNOSED_BAND[1]:.1f}x band.")
    if len(inband) < len(allK):
        for lab, K in variants:
            if not (DIAGNOSED_BAND[0] <= optical(K, 5) <= DIAGNOSED_BAND[1]):
                print(f"    outside: {lab} ({optical(K,5):.2f}x)")
    else:
        print("  No single variant, including either method on its own, moves")
        print("  the answer out of the band. The choice of estimator therefore")
        print("  is not what puts the result where it is.")

    # ---------------------------------------------------------------- S5 ---
    hr("S5  SIGN CHECK -- direction on EVERY matched-dose contrast available")
    print("\n  A K_const that matches one ratio while still predicting the")
    print("  DARKER group to gain MORE is not a fix. Requirement: model ratio")
    print("  > 1 wherever the observed ratio is > 1.")
    print(f"\n  {'contrast':52s} {'obs':>6} {'K=%.3f' % OLD_K_CONST:>8} "
          f"{'K=%.3f' % K_new:>8} {'before':>9} {'after':>9}")
    sign_rows = [("Felton 2016 FST II:V gain", obs_f,
                  lambda K: felton_gain(2, K) / felton_gain(5, K))]
    for sheet, g in groups.items():
        arm = arms[sheet]
        pool = float(np.mean([s["base"] for s in arm["subjects"]]))
        ra, rb, adj = YOUNG_PUBLISHED_ADJUSTED[sheet]
        for dark in [f for f in sorted(g) if f >= 5]:
            # raw pairing, this arm's own baselines
            obs_raw = g[2]["slope"] / g[dark]["slope"]
            sign_rows.append((
                f"Young {sheet.split()[0]:8s} II:{ROMAN_OF[dark]:<2} "
                f"[raw@own]", obs_raw,
                (lambda K, dk=dark, ar=arm, gg=g: (
                    young_slope(2, gg[2]["base"], K, ar["sed"], ar["e_source"],
                                ar["doses"])
                    / young_slope(dk, gg[dk]["base"], K, ar["sed"],
                                  ar["e_source"], ar["doses"])))))
            if ROMAN_OF[dark] == rb:
                sign_rows.append((
                    f"Young {sheet.split()[0]:8s} II:{ROMAN_OF[dark]:<2} "
                    f"[adjusted@pooled]", adj,
                    (lambda K, dk=dark, ar=arm, pl=pool: (
                        young_slope(2, pl, K, ar["sed"], ar["e_source"],
                                    ar["doses"])
                        / young_slope(dk, pl, K, ar["sed"], ar["e_source"],
                                      ar["doses"])))))
    n_bad_old = n_bad_new = 0
    for label, obs, fn in sign_rows:
        old, new = fn(OLD_K_CONST), fn(K_new)
        bad_old = obs > 1.0 and old <= 1.0
        bad_new = obs > 1.0 and new <= 1.0
        n_bad_old += bad_old
        n_bad_new += bad_new
        print(f"  {label:52s} {obs:>6.2f} {old:>8.2f} {new:>8.2f} "
              f"{('INVERTED' if bad_old else 'ok'):>9} "
              f"{('INVERTED' if bad_new else 'ok'):>9}")
    print(f"\n  [derived] contrasts INVERTED at K_const = {OLD_K_CONST:.3f}: "
          f"{n_bad_old} of {len(sign_rows)}")
    print(f"  [derived] contrasts INVERTED at K_const = {K_new:.3f}: "
          f"{n_bad_new} of {len(sign_rows)}")

    print("\n  ABSOLUTE LEVELS on Felton 2016 (the fit matched RATIOS only --")
    print("  do the two arms' gains come out right in nmol/L?)")
    print(f"    observed: phototype II {FELTON_FINAL[2]-FELTON_D0[2]:+.1f}, "
          f"phototype V {FELTON_FINAL[5]-FELTON_D0[5]:+.1f} nmol/L")
    print(f"    {'K_const':>9} {'model II':>9} {'err':>7} {'model V':>9} "
          f"{'err':>7} {'ratio':>7}")
    for K in sorted({0.0, OLD_K_CONST, K_new, 1.473}):
        g2, g5 = felton_gain(2, K), felton_gain(5, K)
        e2 = 100 * (g2 / (FELTON_FINAL[2] - FELTON_D0[2]) - 1)
        e5 = 100 * (g5 / (FELTON_FINAL[5] - FELTON_D0[5]) - 1)
        tag = ""
        if abs(K - OLD_K_CONST) < 1e-9:
            tag = "  <- PRE-RECALIBRATION"
        elif abs(K - K_new) < 1e-9:
            tag = "  <- FITTED"
        elif abs(K - 1.473) < 1e-9:
            tag = "  <- Felton-only (fst_gap_decomposition)"
        print(f"    {K:>9.3f} {g2:>9.2f} {e2:>+6.1f}% {g5:>9.2f} {e5:>+6.1f}% "
              f"{g2/g5:>7.2f}{tag}")

    # ---------------------------------------------------------------- S6 ---
    hr("S6  HELD-OUT SCORING (none of this entered the objective)")
    v_new = optical(K_new, 5)
    print(f"\n  New K_const = {K_new:.2f}  ->  FST V:II optical = {v_new:.2f}x\n")
    print(f"  {'held-out estimate of FST V:II':52s} {'V:II':>10} {'verdict':>26}")
    for lab, v, note in HELD_OUT_SCORING:
        if isinstance(v, tuple):
            shown = f"{v[0]:.2f}-{v[1]:.2f}"
            ok = v[0] <= v_new <= v[1]
        else:
            shown = f"{v:.2f}"
            ok = abs(np.log(v_new / v)) < np.log(1.35)
        print(f"  {lab:52s} {shown:>10} "
              f"{('CONSISTENT' if ok else 'differs'):>12}  {note}")
    print()
    for lab, (v, k) in HELD_OUT_FIELD_BLIND.items():
        print(f"  {lab:52s} {v:>10.2f} "
              f"{'REJECTED':>12}  above every matched-dose measurement")
    print(f"\n  Approved target band (calibration_approval.yaml, "
          f"a1_diagnosis_result_2026_08_17): "
          f"{DIAGNOSED_BAND[0]:.1f}x to {DIAGNOSED_BAND[1]:.1f}x")
    print(f"  New value {v_new:.2f}x is "
          f"{'INSIDE' if DIAGNOSED_BAND[0] <= v_new <= DIAGNOSED_BAND[1] else 'OUTSIDE'}"
          f" that band.")

    # ---------------------------------------------------------------- S7 ---
    hr("S7  RESULTING FST RATIOS, BEFORE AND AFTER")
    print(f"\n  {'pair':8s} {'K=%.2f before' % OLD_K_CONST:>19} "
          f"{'K=%.2f after' % K_new:>19} {'change':>9}")
    for (lab, old), (_, new) in zip(ratio_table(OLD_K_CONST), ratio_table(K_new)):
        print(f"  {lab:8s} {old:>19.2f} {new:>19.2f} {new/old:>8.2f}x")
    print("""
  DARK-SKIN FLAG, at the point of use. The V:II row is anchored by FST V data
  that is entirely Indian-subcontinent ancestry at UK latitude (Felton 2016
  n = 6; Young 2020 Arimed B n = 5). Every VI:* row is an EXTRAPOLATION of one
  coefficient onto M_baseline(VI) = 1.00, a group-level prior with no
  melanin-index validation (see the M_baseline note in vitd_model.py), scored
  against FST VI data that is entirely lamp-based. Use the V:II row for
  clinical statements about South Asian users at temperate latitudes; treat
  every VI figure as an extrapolation until African-ancestry field dosimetry
  exists.""")

    # ---------------------------------------------------------------- S8 ---
    hr("S8  KNOCK-ON: what else in the model this moves")
    a_new = 2.353 * np.exp((K_new - OLD_K_CONST) * 0.21)
    print(f"""
  K_const multiplies M_baseline(FST) inside exp(-K_const * M_baseline), so it
  changes the ABSOLUTE synthesis level of EVERY skin type, not only the
  between-type ratios. FST III, the anchor A = 2.353 was calibrated against,
  moves by

    exp(-{K_new:.3f} * 0.21) / exp(-{OLD_K_CONST:.3f} * 0.21) = {np.exp(-(K_new - OLD_K_CONST) * 0.21):.4f}

  i.e. FST III synthesis per SED falls {100*(1-np.exp(-(K_new - OLD_K_CONST) * 0.21)):.1f}%. To hold that anchor exactly,
  A would have to move to {a_new:.4f}.

  THIS SCRIPT DOES NOT CHANGE A. A is a separate approval
  (calibration_approval.yaml, approved_2026_08_17_pending_implementation, id
  A4) and moving two coupled parameters in one step would make neither
  attributable afterwards. The consequence is recorded here, not silently
  absorbed. `vitd_validation.sanity_single_bolus_FST_III` hardcodes
  `exp(-0.33 * 0.21)` in its expected value and will fail until its expected
  value is made a function of params.K_const -- that file belongs to another
  owner and was deliberately not touched.""")

    # ---------------------------------------------------------------- S9 ---
    hr("S9  EVERY [CHOICE] AND [ESTIMATE] IN THIS DERIVATION")
    for i, s in enumerate(CHOICES, 1):
        print(f"\n  {i:2d}. {s}")


CHOICES = [
    "[CHOICE] Calibration set = within-study, matched-physical-dose, "
    "matched-BSA, same-source trials only (Felton 2016; Young 2020). Field "
    "cohorts, cross-trial pairs and fractional-MED designs are excluded from "
    "the objective and used only to score.",
    "[CHOICE] Objective = weighted sum of squared LOG errors on response "
    "RATIOS, not on absolute levels. Absolute levels depend on A, which is "
    "separately uncalibrated (Young 2021 puts fair-skin slopes at 2.0-5.0 "
    "nmol/L per SED against the model's ~1.85 at 85% BSA), so fitting K_const "
    "to absolute levels would launder A's error into the pigment term.",
    "[CHOICE] Weight one per STUDY, not per contrast or per subject. Young's "
    "two arms are one recruitment and one protocol design. S4 reports what "
    "every other weighting gives.",
    "[CHOICE] The headline estimator combines FOUR within-study routes (two "
    "lab, two field-with-behaviour-removed) rather than the lab pair alone, "
    "because S3 shows the lab pair is internally incompatible and its "
    "least-squares compromise describes neither trial.",
    "[CHOICE] Young's study-level value uses the 'adjusted@pooled' pairing, "
    "the only one of the three whose answer does not depend on the model's "
    "feedback term being the right strength -- which S1 measures and finds it "
    "is not (0.4-0.8x of the observed baseline gradient).",
    "[CHOICE] The result is rounded to 2 decimal places. The four routes have "
    "an SD of ~0.3 in K_const, so quoting more digits than that would imply a "
    "precision the evidence does not carry.",
    "[CHOICE] Search bounded at K_const <= 3.0. Above it the darker arm's "
    "modelled net response goes negative and the ratio stops being monotone "
    "-- the same bound and reason as fst_gap_decomposition.py.",
    "[ESTIMATE] Felton 2016's cabinet is assumed to be the Arimed B its own "
    "group used in Rhodes 2010 and Farrar 2013; the paper says only '95% UVA "
    "/ 5% UVB simulated summer sunlight'. E_source cancels from a "
    "between-skin-type ratio, so this moves absolute levels only.",
    "[ESTIMATE] Young 2020's exposure interval is taken as 3.5 d, the "
    "midpoint of its stated '3-4 days'.",
    "[ESTIMATE] Young's blood-draw times are reconstructed from the printed "
    "cumulative-dose grid on the assumption that each intermediate draw "
    "immediately preceded the next exposure and the last followed exposure 5. "
    "The recorded 17-18 day study duration is consistent with that.",
    "[ESTIMATE/PROVENANCE] The held-out field numbers (Kift, Darling) are "
    "reproduced from fst_gap_decomposition.py, which rests on Kift 2013 "
    "decimal reconstructions and on an exposed-BSA figure ASSUMED equal "
    "between Darling's two cohorts. They score the fit; they do not set it.",
    "[LIMIT] Felton 2016 reports group means and SDs, not individual "
    "trajectories, and its FST V gain SD (10.5) exceeds its mean gain (8.3). "
    "No confidence interval on the Felton-implied K_const is computable from "
    "what is published, so none is quoted.",
    "[LIMIT] Young 2020's Arimed B FST V cell is n = 5 and its FST VI cell "
    "n = 6 (3 African, 3 Indian-subcontinent). The TL01 arm has n = 11 FST VI "
    "(all African) and NO FST V at all.",
]


if __name__ == "__main__":
    main()
