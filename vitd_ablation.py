"""Model-ablation ladder: answer scheduling questions one mechanism at a time.

Motivation
----------
The advisor's instruction was: "Let's answer these questions carefully with
just melanin dynamics. Then, later, we can add skin thickening. And then add
skin cancer risk."

Every analysis in this repo currently runs the full model, with melanin
dynamics AND stratum-corneum (SC) photoadaptation always active, and there was
no vocabulary for the intermediate rungs. This module supplies that vocabulary
so a downstream analysis can take a rung name as an argument and be explicit
about which mechanisms were in play when it produced a number.

The ladder (in order, see ``list_rungs()``)
-------------------------------------------
``melanin_only``
    Melanin dynamics active; SC photoadaptation fully disabled. The SC state
    stays identically zero and contributes no attenuation to synthesis.

``melanin_plus_sc``
    The current default model: melanin dynamics plus SC photoadaptation. No
    overrides are injected, so this rung reproduces
    ``vitd_model.simulate(...)`` exactly.

``melanin_sc_risk``
    Dynamically IDENTICAL to ``melanin_plus_sc``. ``vitd_model.simulate()``
    already computes the schedule-level skin-cancer indices unconditionally
    and returns them under ``"cancer_risk"`` on every call, at every rung.
    This rung therefore changes nothing about the trajectory -- it is a
    REPORTING distinction only: it declares that the analysis is obliged to
    present the risk indices alongside the 25(OH)D answer rather than quietly
    discarding them. Use ``reports_risk(rung)`` to branch on that obligation.
    Do not describe this rung as "adding" cancer risk to the model.

CAVEAT -- ``melanin_only`` is an ABLATION, not a re-calibrated model
-------------------------------------------------------------------
The model parameters were fitted with the SC channel PRESENT. Nothing was
re-fitted after removing it. Deleting an attenuating channel without
re-fitting the terms it was collinear with (notably ``K_ATTEN`` vs ``K_SC``,
flagged as jointly unidentifiable in the ``vitd_model`` module docstring)
inflates predicted synthesis by an amount that has not been characterised.

So: absolute 25(OH)D values produced under ``melanin_only`` are NOT calibrated
predictions and must not be quoted as such, nor compared on an absolute scale
against validation data or against the other rungs' absolute levels. They are
a mechanism-isolation diagnostic -- useful for "which way does melanin push
this schedule, and how strongly, on its own?", and for within-rung schedule
RANKINGS, not for "what serum level will this person reach?".

The SC priors being ablated here (``SC_max``, ``alpha_SC``, ``K_SC``) are
themselves flagged as uncalibrated in the ``vitd_model`` module docstring,
which is the only canonical source for their values -- read them there.

This module is purely additive. It injects keyword overrides into
``vitd_model.simulate()`` and changes no other module's behaviour.
"""

from typing import Dict, List, Tuple

import vitd_model


# Per-rung keyword overrides handed to ``vitd_model.simulate()``.
#
# melanin_only -- why these four overrides:
#   K_SC_override=0.0
#       The load-bearing one for the DYNAMICS. Synthesis is attenuated by
#       ``exp(-K_SC * SC)`` (see ``vitd_model.synthesis_per_session``); with
#       K_SC = 0 that factor is identically 1 whatever SC does, so the SC
#       channel cannot influence 25(OH)D at all.
#   SC_max_override=0.0 and alpha_SC_override=0.0
#       These make the reported SC trajectory read as a clean zero rather than
#       tracking a state that is present but inert. Note they are REDUNDANT
#       WITH EACH OTHER given the default SC0 = 0.0: the build update is
#       ``SC_build = SC_max - (SC_max - SC_build) * exp(-alpha_SC * sed)``, so
#       alpha_SC = 0 freezes SC_build at its initial value, and SC_max = 0
#       drives it to 0 from an initial 0. Both are set so that the ablation
#       still holds if a caller passes a non-zero SC0 or a params object with
#       non-default SC settings.
#   k_damage_override=0.0
#       NOT in the original three, and necessary. The SC state is two-process:
#       ``SC = SC_build - SC_damage``, and the damage term accumulates as
#       ``SC_damage += k_damage * sed`` INDEPENDENTLY of SC_max and alpha_SC.
#       ``ModelParams.k_damage`` defaults to 0.0, so the first three overrides
#       happen to suffice today -- but any caller passing a params object with
#       k_damage > 0 (its documented prior range is non-degenerate) would get a
#       NEGATIVE SC trajectory under a rung that claims SC is disabled. Pinning
#       it here makes the no-op unconditional rather than accidental.
#   (k_decay_SC / k_repair_SC are not overridden: they are relaxation rates on
#    states that are pinned at zero, so they have nothing to act on.)
_RUNG_OVERRIDES: Dict[str, Dict[str, float]] = {
    "melanin_only": {
        "K_SC_override": 0.0,
        "alpha_SC_override": 0.0,
        "SC_max_override": 0.0,
        "k_damage_override": 0.0,
    },
    "melanin_plus_sc": {},
    "melanin_sc_risk": {},
}

_RUNG_LABELS: Dict[str, str] = {
    "melanin_only": "Melanin only",
    "melanin_plus_sc": "Melanin + SC",
    "melanin_sc_risk": "Melanin + SC + risk",
}

# Whether the rung obliges the analysis to REPORT the skin-cancer indices.
# Every rung computes them (see module docstring); only this one must show them.
_RUNG_REPORTS_RISK: Dict[str, bool] = {
    "melanin_only": False,
    "melanin_plus_sc": False,
    "melanin_sc_risk": True,
}

# Ladder order, rung -> immutable override mapping. Insertion-ordered.
RUNGS: Dict[str, Dict[str, float]] = {
    name: dict(_RUNG_OVERRIDES[name])
    for name in ("melanin_only", "melanin_plus_sc", "melanin_sc_risk")
}


def list_rungs() -> List[str]:
    """Rung names in ladder order (least to most mechanism)."""
    return list(RUNGS)


def _check_rung(rung: str) -> None:
    if rung not in RUNGS:
        raise ValueError(
            f"unknown ablation rung {rung!r}; expected one of {list_rungs()}")


def rung_label(rung: str) -> str:
    """Short human-readable label for plot legends and table columns."""
    _check_rung(rung)
    return _RUNG_LABELS[rung]


def reports_risk(rung: str) -> bool:
    """Whether this rung requires skin-cancer indices to be REPORTED.

    False does not mean the indices are absent from the result dict -- they
    are computed on every rung. It means this rung's answer is not obliged to
    present them.
    """
    _check_rung(rung)
    return _RUNG_REPORTS_RISK[rung]


def rung_overrides(rung: str) -> Dict[str, float]:
    """A copy of the ``simulate()`` keyword overrides this rung injects."""
    _check_rung(rung)
    return dict(RUNGS[rung])


def simulate_rung(rung: str,
                  fst: int,
                  schedule: List[Tuple[float, float]],
                  duration_days: int,
                  **kwargs) -> Dict:
    """Run ``vitd_model.simulate()`` at one rung of the ablation ladder.

    Thin wrapper: the rung's overrides are injected as keyword arguments and
    everything else is passed straight through. The returned dict is
    ``vitd_model.simulate()``'s, with the rung name stamped under
    ``"ablation_rung"``.

    Raises
    ------
    ValueError
        If ``rung`` is unknown, or if ``kwargs`` contains an override keyword
        that the rung also sets. Silently letting a caller's override win (or
        lose) is the failure mode this guard exists to prevent: it would
        produce a result labelled with a rung whose mechanism set was not
        actually the one simulated.
    """
    _check_rung(rung)
    overrides = RUNGS[rung]

    clashes = sorted(set(overrides) & set(kwargs))
    if clashes:
        raise ValueError(
            f"rung {rung!r} already sets {clashes}; passing "
            f"{'them' if len(clashes) > 1 else 'it'} again would silently "
            f"conflict with the ablation. Drop "
            f"{'those keywords' if len(clashes) > 1 else 'that keyword'}, or "
            f"call vitd_model.simulate() directly if you want a custom "
            f"combination that is not one of the named rungs.")

    result = vitd_model.simulate(fst=fst,
                                 schedule=schedule,
                                 duration_days=duration_days,
                                 **overrides,
                                 **kwargs)
    result["ablation_rung"] = rung
    return result
