"""One-off: decompose the abstract's cancer-risk trade-off (claim iv) into its
two components, and test whether facultative melanin accumulation mediates it.

Not wired into run_all.py. Written to answer a specific question raised while
reconciling paper/abstract.tex against the model (see
docs/ABSTRACT_RECONCILIATION.md): the abstract says front-loaded schedules cut
the intermittent-exposure (BCC/melanoma) risk index by a median of ~22% while
raising the cumulative-dose (SCC) index by a median of ~8%, at a matched
sufficiency floor. Does that two-component trade-off actually hold in the
model, and is it driven by the tan (facultative melanin, M(t)) built up early?

Two checks:

1. DECOMPOSITION. Re-derive the per-cell scc_index / intermittent_index /
   total_sed / mean and max per-session dose for every free-spacing vs
   uniform-spacing paired cell already computed in
   tables/joint_schedule_optimization.csv (the source of "20 of 22", "~22%",
   "~8%"). No re-optimisation -- this is a re-read of existing output plus
   arithmetic.

2. MELANIN MEDIATION. vitd_physiology.cancer_risk() takes only per-session SED
   and a STATIC per-FST susceptibility (Shih 2018 constitutive melanin %); it
   never reads the dynamic tanning state M(t). So melanin dynamics cannot move
   the risk indices directly for a FIXED schedule -- confirmed here by running
   the same winning schedules through vitd_ablation's melanin_only vs
   melanin_plus_sc rungs and checking cancer_risk is bit-identical. The
   channel through which melanin COULD matter is indirect: K_ATTEN attenuates
   synthesis as the tan builds (exp(-K_ATTEN * excess_M)), which could change
   how much total dose a schedule needs to clear the sufficiency floor -- and
   a front-loaded schedule builds more excess melanin earlier than a uniform
   one. This is tested directly: rebuild each winning schedule exactly
   (rebuild_joint_schedule) and re-simulate it at K_ATTEN=0 (melanin present,
   dynamically inert) vs the calibrated default (K_ATTEN=0.288972), and compare
   the trough-after-ramp gain between the uniform and free arms.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

# Repo root must be on sys.path BEFORE the vitd_* imports: running this as
# `python3 scripts/abstract_risk_decomposition.py` puts scripts/ on sys.path,
# not the repo root, so `import vitd_ablation` raised ModuleNotFoundError.
# REPO_ROOT was defined below the imports, which is too late to help.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import vitd_ablation  # noqa: E402
import vitd_model  # noqa: E402
import vitd_optimization as vo  # noqa: E402
import vitd_physiology as vp  # noqa: E402

SRC_CSV = os.path.join(REPO_ROOT, "tables", "joint_schedule_optimization.csv")
OUT_DIR = os.path.join(REPO_ROOT, "tables", "debug")
OUT_CELLS = os.path.join(OUT_DIR, "abstract_risk_decomposition_cells.csv")
OUT_MEDIATION = os.path.join(OUT_DIR, "abstract_risk_decomposition_mediation.csv")


def load_rows():
    with open(SRC_CSV, newline="") as f:
        return list(csv.DictReader(f))


def part1_decomposition(rows):
    """Pair free-spacing vs uniform-spacing rows per (fst, n_sessions) and
    decompose the risk delta into its two components plus the dose-shape
    statistics that mechanically drive each one."""
    by_key = {}
    for r in rows:
        if r["feasible"] != "True":
            continue
        key = (int(r["fst"]), int(r["n_sessions"]))
        slot = "free" if r["free_spacing"] == "True" else "uniform"
        # Keep the best (lowest combined_risk) row per slot per key -- CSV has
        # one row per (fst, n_sessions, family); analysis_joint_schedule's own
        # selection picks the best family, replicate that here.
        cur = by_key.setdefault(key, {})
        prev = cur.get(slot)
        if prev is None or float(r["combined_risk"]) < float(prev["combined_risk"]):
            cur[slot] = r

    out = []
    for (fst, n), slot in sorted(by_key.items()):
        if "free" not in slot or "uniform" not in slot:
            continue
        u, f = slot["uniform"], slot["free"]
        nu = int(float(u["n_sessions_delivered"])) or 1
        nf = int(float(f["n_sessions_delivered"])) or 1
        row = {
            "fst": fst, "n_sessions": n,
            "scc_uniform": float(u["scc_index"]), "scc_free": float(f["scc_index"]),
            "scc_pct_change": 100.0 * (float(f["scc_index"]) - float(u["scc_index"])) / float(u["scc_index"]),
            "interm_uniform": float(u["intermittent_index"]), "interm_free": float(f["intermittent_index"]),
            "interm_pct_change": 100.0 * (float(f["intermittent_index"]) - float(u["intermittent_index"])) / float(u["intermittent_index"]),
            "total_sed_uniform": float(u["total_sed"]), "total_sed_free": float(f["total_sed"]),
            "total_sed_pct_change": 100.0 * (float(f["total_sed"]) - float(u["total_sed"])) / float(u["total_sed"]),
            "mean_sed_per_session_uniform": float(u["total_sed"]) / nu,
            "mean_sed_per_session_free": float(f["total_sed"]) / nf,
            "max_sed_uniform": float(u["max_sed"]), "max_sed_free": float(f["max_sed"]),
            "max_sed_pct_change": 100.0 * (float(f["max_sed"]) - float(u["max_sed"])) / float(u["max_sed"]),
            "pattern": None,
        }
        row["pattern"] = (
            ("scc_up" if row["scc_pct_change"] > 0 else "scc_down") + "/" +
            ("interm_down" if row["interm_pct_change"] < 0 else "interm_up")
        )
        out.append(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CELLS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    scc_changes = np.array([r["scc_pct_change"] for r in out])
    interm_changes = np.array([r["interm_pct_change"] for r in out])
    total_sed_changes = np.array([r["total_sed_pct_change"] for r in out])
    max_sed_changes = np.array([r["max_sed_pct_change"] for r in out])
    n_classic = sum(1 for r in out if r["pattern"] == "scc_up/interm_down")

    print(f"=== Part 1: decomposition over {len(out)} paired cells ===")
    print(f"median scc_pct_change     = {np.median(scc_changes):+.2f}%  "
          f"(abstract: ~+8%)")
    print(f"median interm_pct_change  = {np.median(interm_changes):+.2f}%  "
          f"(abstract: ~-22%)")
    print(f"median total_sed_pct_change = {np.median(total_sed_changes):+.2f}%")
    print(f"median max_sed_pct_change   = {np.median(max_sed_changes):+.2f}%")
    print(f"cells matching classic pattern (scc up AND interm down): "
          f"{n_classic}/{len(out)}")
    print("correlation(total_sed_pct_change, scc_pct_change) = "
          f"{np.corrcoef(total_sed_changes, scc_changes)[0,1]:.3f}  "
          "(expect strongly positive: SCC is a power law in total SED)")
    print("correlation(max_sed_pct_change, interm_pct_change) = "
          f"{np.corrcoef(max_sed_changes, interm_changes)[0,1]:.3f}  "
          "(expect strongly positive: intermittent index is cubic in per-session dose)")
    print(f"wrote {OUT_CELLS}")
    return out


def part2_mediation(rows, sample_keys):
    """For a handful of representative cells: (a) confirm cancer_risk is
    ablation-rung-independent for a fixed schedule, (b) test whether the tan's
    attenuation of synthesis (K_ATTEN) is a net cost or benefit to the
    front-loaded arm's ability to clear the sufficiency floor cheaply."""
    by_key = {}
    for r in rows:
        if r["feasible"] != "True":
            continue
        key = (int(r["fst"]), int(r["n_sessions"]))
        slot = "free" if r["free_spacing"] == "True" else "uniform"
        cur = by_key.setdefault(key, {})
        prev = cur.get(slot)
        if prev is None or float(r["combined_risk"]) < float(prev["combined_risk"]):
            cur[slot] = r

    out = []
    for key in sample_keys:
        if key not in by_key or "free" not in by_key[key] or "uniform" not in by_key[key]:
            continue
        fst, n = key
        for arm_name, r in by_key[key].items():
            res = {
                "fst": int(r["fst"]), "n_sessions": int(r["n_sessions"]),
                "gamma": float(r["gamma"]), "family": r["family"],
                "p_scale": float(r["p_scale"]), "p_a": float(r["p_a"]),
                "p_b": float(r["p_b"]), "p_level": float(r["p_level"]),
            }
            days, seds = vo.rebuild_joint_schedule(res, horizon_days=365)
            sched = vo.build_variable_schedule(days, seds, fst=fst)

            # (a) cancer_risk vs ablation rung, fixed schedule.
            session_seds = [v for _, v in sched]
            risk_direct = vp.cancer_risk(session_seds, fst)
            rung_risks = {}
            for rung in ("melanin_only", "melanin_plus_sc", "melanin_sc_risk"):
                sim = vitd_ablation.simulate_rung(rung, fst, sched, 365)
                rung_risks[rung] = sim["cancer_risk"]

            identical_across_rungs = all(
                abs(rung_risks[rg]["scc_index"] - risk_direct["scc_index"]) < 1e-9 and
                abs(rung_risks[rg]["intermittent_index"] - risk_direct["intermittent_index"]) < 1e-9
                for rg in rung_risks
            )

            # (b) K_ATTEN on vs off, same schedule -- does the tan's synthesis
            # attenuation help or hurt this arm's ability to hold the floor?
            base_params = vitd_model.ModelParams()
            sim_katten_on = vitd_model.simulate(
                fst=fst, schedule=sched, duration_days=365,
                params=base_params, D0=25.0,
                dose_framing="fixed_per_session_SED")
            sim_katten_off = vitd_model.simulate(
                fst=fst, schedule=sched, duration_days=365,
                params=base_params, D0=25.0, K_ATTEN_override=0.0,
                dose_framing="fixed_per_session_SED")

            def trough_after_ramp(sim, ramp=vo.RAMP_IN_DAYS):
                t = np.asarray(sim["t"]); D = np.asarray(sim["D"])
                mask = t >= ramp
                return float(np.min(D[mask])) if mask.any() else float("nan")

            trough_on = trough_after_ramp(sim_katten_on)
            trough_off = trough_after_ramp(sim_katten_off)

            out.append({
                "fst": fst, "n_sessions": n, "arm": arm_name,
                "gamma": res["gamma"],
                "cancer_risk_identical_across_rungs": identical_across_rungs,
                "scc_index": risk_direct["scc_index"],
                "intermittent_index": risk_direct["intermittent_index"],
                "trough_after_ramp_K_ATTEN_on": trough_on,
                "trough_after_ramp_K_ATTEN_off": trough_off,
                "trough_gain_from_removing_K_ATTEN": trough_off - trough_on,
            })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_MEDIATION, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print("\n=== Part 2: melanin mediation ===")
    print("cancer_risk identical across melanin_only/melanin_plus_sc/"
          f"melanin_sc_risk for a fixed schedule: "
          f"{all(o['cancer_risk_identical_across_rungs'] for o in out)}  "
          "(expected True -- cancer_risk() never reads M(t))")
    for key in sample_keys:
        u = next((o for o in out if (o["fst"], o["n_sessions"]) == key and o["arm"] == "uniform"), None)
        f_ = next((o for o in out if (o["fst"], o["n_sessions"]) == key and o["arm"] == "free"), None)
        if u is None or f_ is None:
            continue
        print(f"FST {key[0]} N={key[1]}: trough gain from removing K_ATTEN -- "
              f"uniform arm {u['trough_gain_from_removing_K_ATTEN']:+.4f} nmol/L, "
              f"free (front-loaded) arm {f_['trough_gain_from_removing_K_ATTEN']:+.4f} nmol/L")
    print(f"wrote {OUT_MEDIATION}")
    return out


if __name__ == "__main__":
    rows = load_rows()
    cells = part1_decomposition(rows)
    # Representative sample: the doc's own seed-stability sample (I/183,
    # II/260, III/146) plus the largest-gain cell (VI/146).
    sample = [(1, 183), (2, 260), (3, 146), (6, 146)]
    part2_mediation(rows, sample)
