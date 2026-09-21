"""
vitd_model.py
=============
Three-state ODE model of serum 25(OH)D dynamics under UV exposure schedules.
Models photoadaptation (melanin upregulation and stratum corneum thickening)
for darker skin types.

State variables
---------------
    D(t)  : effective serum 25(OH)D, nmol/L
        Time-integrated plasma + adipose reservoir for week-to-month scheduling.
    M(t)  : effective melanin index, dimensionless 0-1
        Datta 2019 reflectance scale; 0.0 = FST I, 1.0 = FST VI.
    SC(t) : additional stratum corneum thickness, dimensionless (above baseline 0)
        [C8] ACTIVE state variable. Remaining task: FST/source-specific
        calibration, not implementation.

Governing equations
-------------------
dD/dt  = S(UV(t), M(t), SC(t), D(t), F, bsa)  -  k_decay_D(D) * D
dM/dt  = alpha * UV(t) * (M_max - M)  -  beta * (M - M_baseline(F))
dSC/dt = alpha_SC * UV(t) * (SC_max - SC)  -  k_decay_SC * SC  -  k_damage * UV(t) * SC

Synthesis function (logarithmic dose-response in effective D3-weighted dose SED * E_source,
                    Webb 2021 form Δ = c * ln(SED * VDE) + b, melanin and SC-attenuated,
                    BSA-scaled, feedback-inhibited):
    S(UV, M, SC, D, F, bsa) = A * ln(1 + UV * E_source) *
                              bsa_factor(bsa) *
                              exp(-K_const * M_baseline(F)) *
                              exp(-K_ATTEN * max(0, M - M_baseline(F))) *
                              exp(-K_SC * SC) * g(D)

    g(D) is the FEEDBACK FACTOR, selected by `feedback_mode`. DEFAULT
    is 'biphasic' (see the feedback-form entry below):
        g(D) = 1 / (1 + (D / 56.0) ** 2)          # 'biphasic'  DEFAULT
        g(D) = max(0, 1 - D / D_max)              # 'linear'    LEGACY
        g(D) = max(0, 1 - (D/D_max)**p)           # 'power'     LEGACY
        g(D) = exp(-ln2 * D / D_half_feedback)    # 'saturating' investigated,
                                                  #   not adopted; see below

    bsa_factor(bsa) = (bsa / BSA_REF) ** BSA_EXP
        where BSA_REF = 0.35 (Farrar 2013, 35% BSA; Bogh 2011 used 6-24% BSA),
        BSA_EXP = 0.3281 [CALIBRATION 2026-09-06, user-approved; was 0.50].
        Feedback-deconvolved from two within-study 25(OH)D BSA contrasts
        (Young 2021, Vahavihu/Jager 2018) -- see the BSA_EXP entry below and
        scripts/bsa_exp_recalibration.py for the full derivation, including
        an UNEXPECTED result: the deconvolution moved the exponent DOWN, not
        up, because this model's own decay term dominates over feedback in
        this regime. Read that entry before using this value.
        Default bsa = BSA_REF, existing scenarios unchanged (factor = 1.0).

Parameters & Sources
---------------------------
- k_decay_D(D) = ln(2) / (2953 * D^(-0.72)):
    State-dependent D decay rate from Datta 2017 power-law fit (DOI: 10.1039/c6pp00258g).
    # Power-law fitted to Datta 2017 three-group data (89d, 148d, 199d)
    Three-group validation: t_half(132) ≈ 89d, t_half(65) ≈ 148d, t_half(43) ≈ 199d,
    all within 1.4% of Datta 2017 reported values. Replaces prior fixed 80-day constant.
- A = 3.114:
    Synthesis scaling factor.
    [CALIBRATION 2026-08-18, user-approved item A4] Raised from 2.353.
    Reproducible in one line of arithmetic; asserted by
    `vitd_validation.young2021_A_anchor()` and
    `tests/test_vitd_model.py::TestYoung2021AAnchor`.

    ANCHOR. Young et al. 2021, PNAS 118:e2015867118, Fig. 4B — the FULL-BODY
    (FB) study, linear regressions of delta-25(OH)D3 against ERYTHEMAL SED, not
    forced through zero, at 85% BSA:
        UV6       y = 4.97x + 0.86   (n = 10)
        Arimed B  y = 3.18x + 2.26   (n = 19)
        PUVA      y = 1.99x + 1.12   (n = 11)
    Midpoint of that printed range = (1.99 + 4.97) / 2 = 3.48 nmol/L per SED.
    Cross-check: the n-weighted pooled FB arm (Table 1, n = 40) rose
    34.6 -> 70.7 on a cumulative 10 SED, i.e. 3.61 nmol/L per SED — 3.7% above
    the range midpoint, so the two readings of "midpoint" agree.

    A is then set so that the model's marginal yield for ONE 1 SED exposure
    equals 3.48 nmol/L under the study's own conditions:
        FST II (the FB arm was 38/40 FST II, 2/40 FST I),
        exposed_bsa = 0.85 (the paper's stated FB exposure),
        D = 34.6 nmol/L (the printed FB baseline mean),
        SC = 0, untanned, E_source = E_SOURCE_SOLAR_DEFAULT = 1.89.
        A = 3.48 / [ ln(1+1.89) * bsa_factor(0.85) * exp(-K_const*M_II)
                     * g_biphasic(34.6) ]
          = 3.48 / [ 1.061257 * 1.558387 * 0.933700 * 0.723721 ]
          = 3.48 / 1.117569 = 3.1139  ->  3.114

    WHAT 2.353 WAS: preserved a 1.27 nmol/L rise for 1 SED in untanned
    FST III at 35% BSA, D = 25, computed at K_const = 0.33. That reference
    stopped being reproducible when K_const moved to 0.98 on 2026-08-17
    (the same expression now returns 1.11), so 2.353 was already orphaned
    before this change.

    THE "~1.85 vs 2.0-5.0" GAP WAS PARTLY A SPECTRAL ARTEFACT, NOT A BIOLOGICAL
    ONE. The 1.85 figure is computed at E_source = 1.00 — the
    "solar_broadband" entry DEPRECATED on 2026-08-16 because no solar spectrum
    has VDE:EE = 1.00. At the current solar default (1.89) the incumbent
    A = 2.353 already yields 2.63 nmol/L per SED, already INSIDE Young's
    2.0-5.0 band. This calibration moves A by x1.32, not the x1.5-1.9 that
    the pre-2026-08-16 framing implied.

    STATUS: CENTRE OF A WIDE, SOURCE-DEPENDENT RANGE, NOT A MEASURED CONSTANT.
    Young's three FB slopes span 2.5x (1.99-4.97); the spread is the paper's
    headline finding: SED is a poor spectral predictor of vitamin D synthesis.
    Pairing each arm with its own E_source from SOURCE_CALIBRATION_TABLE gives
    A = 2.90 (Arimed B, E = 1.83) and A = 5.06 (UV6, E = 1.54) — a 1.74x
    disagreement, and in the WRONG ORDER: those E_source values (Webb 2021,
    unshifted CIE-174) predict UV6 slightly weaker than Arimed B, while Young
    measures it 56% stronger. The adopted value sits between them. Read 3.114
    as the centre of roughly 2.9-5.1.

    DARK-SKIN FLAG (standing_directives.dark_skin_priority). Young 2021 is
    FST I-II ONLY (n = 75 overall, 6x I + 69x II, zero FST III-VI); its
    authors disclaim generalisation: "The results from this study only apply
    to FST I/II". A is a WHOLE-MODEL synthesis scalar applied at every FST,
    so calibrating it on this study and using it for FST V-VI is a
    LIGHT-SKIN EXTRAPOLATION. Mitigated but not removed by the fact that
    K_const is anchored separately on matched-dose FST V data (Felton 2016,
    Young 2020, Kift 2013, Darling 2013), so what is extrapolated is the
    overall level, not the skin-type gradient. Any absolute 25(OH)D
    prediction for FST V-VI must carry this flag.

    MEASURED CONSEQUENCES (2026-08-18, both changes A2 + A4 in force):
      Farrar 2013 (FST V, the only dark-skin quantitative check in the suite)
        moves from FAIL to PASS: delta +8.7 -> +11.2 nmol/L against observed
        +21.7, acceptance band 10-35.
      Bogh 2012 (light-skin NB-UVB, DEBUGGING_TARGET) stays PASS:
        +48.0 -> +51.6 against observed +55.8, band 45.8-65.8.
      Karppinen 2017 now passes in DEFAULT feedback mode; the
        experimental-mode override was removed from that check.
      Required validation score 8/10 -> 10/10.
    NOT COVERED: A cannot reproduce Young's FB protocol END-TO-END. The
    actual 5 x 2 SED protocol needs A ~ 8.3 (Arimed B arm) to reach
    observed +38.6, and PB (3.7% BSA) arms need more. The model
    under-predicts Young's absolute rises by 2.7x (FB) to 6.2x (PB) at
    any A that keeps Bogh 2012 and Farrar 2013 in band. Young 2021 and
    Bogh 2012 are MUTUALLY INCOMPATIBLE: Bogh delivered ~6.7x Young's
    D3-effective dose from a LOWER baseline and produced only 1.45x the
    rise. This calibration matches Young's per-SED SLOPE (what the
    approval asked for); it does not resolve that conflict, and no value
    of A can. Young's own within-study BSA contrast (85% vs 3.7% BSA,
    3.28x response for a 23x area ratio) implies an APPARENT BSA_EXP ~ 0.38.
    [CALIBRATION 2026-09-06, user-approved] THIS WAS ACTED ON, but not by
    setting 0.38 directly (that would double-count feedback): BSA_EXP moved
    0.50 -> 0.3281 by deconvolving g(D) -- see the BSA_EXP entry below.
    A itself is UNCHANGED here and its own anchor arithmetic (the derivation
    above, which uses bsa_factor(0.85) at the OLD BSA_EXP = 0.50) is now
    STALE: recomputing it with the new BSA_EXP no longer reproduces
    A = 3.114 (tests/test_vitd_model.py::TestYoung2021AAnchor, left FAILING
    deliberately as of 2026-09-06 rather than silently patched -- A was
    explicitly out of scope for this calibration pass). A future pass must
    either re-anchor A jointly with BSA_EXP or accept this inconsistency
    with a recorded rationale.
- K_const = 1.15:
    Baseline (constitutive) inhibition coefficient. FST V:II optical factor
    of 1.9538x, FST VI:I of 3.0228x (was 2.0053x and 2.6645x at K_const=0.98).
    [CALIBRATION 2026-08-17, user-approved] Raised from 0.33 to 0.98.
    [CALIBRATION 2026-09-06, user-approved, COUPLED REFIT] 0.98 -> 1.15
    (script output 1.15490, rounded to 2 dp per this script's own convention).
    Refitted in the SAME change as M_baseline_by_FST's move to the Narea 2026
    table, on which this fit runs (M DIFFERENCES enter the four ratios below
    directly). Re-running `scripts/kconst_recalibration.py` with the new
    M_baseline_by_FST table (no other change) gives:
        Felton 2016 (lab, FST V, within-study)                   2.097
        Young 2020 (lab, FST VI, both arms, adjusted@pooled)     0.347
        Kift 2013 / Webb 2010 (field, behaviour removed)         0.788
        Darling 2013 (field, behaviour removed)                  1.387
        mean (= the least-squares solution)                      1.155 -> 1.15
        SD across the four routes                                0.759
        range                                                0.347-2.097
    Kift and Darling are unchanged from the pre-refit run (they are field
    inversions computed independently of M_baseline_by_FST in
    fst_gap_decomposition.py); Felton and Young shift because the M gap they
    are inverted against changed (FST V:II gap 0.71 -> 0.5824, FST II:VI gap
    widens from the old table's 0.93 to 0.8285). The SD across routes widened
    (0.56 -> 0.759) because the narrower FST V:II M-gap makes the Felton route
    more sensitive to its own observed ratio -- Felton's implied K_const alone
    moved 1.473 -> 2.097, a bigger swing than the other three routes combined.
    Derivation: `scripts/kconst_recalibration.py`; diagnosis:
    `scripts/fst_gap_decomposition.py`; approval: `calibration_approval.yaml`.
    Fitted by equal-weight least squares in log ratio space to the same four
    independent WITHIN-STUDY estimates as before, two by lab matched-dose and
    two by field personal dosimetry with behavioural dose difference removed.
    Treat 1.15 as a central estimate of a wide range (SD 0.759, i.e. roughly
    0.4-1.9), not a measured constant -- WIDER than the 0.3-1.5 range this
    parameter carried at 0.98.
    WHAT 0.33 WAS: ln(1.36) / (M_VI - M_II), a static-optics inversion of
    one Young 2020 ratio that ignored feedback and protocol entirely. On
    Felton 2016's matched-dose protocol it INVERTED the skin-type gap (model
    FST II:V 0.96 vs observed 2.14), because FST V's lower baseline buys
    more headroom under (1 - D/D_max) than the melanin term took away. That
    inversion is gone at 0.98 (model 1.49), on every matched-dose contrast.
    NOT COVERED: behaviour. Shade, clothing and time outdoors are a SEPARATE
    2.1-2.4x factor and live in `exploratory/vitd_behaviour.py`, defaulting to 1.0.
    Folding them in here would need 4.3-5.7x, above every matched-dose
    measurement in existence.
    DARK-SKIN PROVENANCE: every FST V datapoint behind this number is Indian-
    subcontinent ancestry at UK latitude; every FST VI datapoint is lamp-based.
    All VI:* ratios are extrapolations onto M_baseline(VI) = 1.00, itself an
    unvalidated group-level prior.
    [EVIDENCE NOTE 2026-09-05, CORRECTED 2026-09-06] "Young et al. 2019" (J
    Invest Dermatol 140:1418-1426, DOI 10.1016/j.jid.2019.11.019, e-pub 2019 /
    print 2020) is NOT a fifth K_const route. It IS "Young 2020" already
    counted above (the 0.278 low outlier) — the same trial behind
    M_baseline_by_FST's ITA means and `YOUNG_PUBLISHED_ADJUSTED` in
    scripts/kconst_recalibration.py (ArimedB 1.29, TL01 1.43), which are the
    exact "1.3:1 / 1.4:1" figures a 2026-09-05 documentation pass here mistook
    for new, independent, larger evidence via a literature-search citation
    that carried the paper's e-pub year instead of its already-used "Young
    2020" label. Retracting that framing: no new route, no open reconciliation
    gap. The 2.66x FST VI:I comparison that pass drew was also invalid on its
    own terms — Young's ratio is FST II:VI, not I:VI.
    CAVEAT worth keeping: Young 2020's FST/ITA grouping is fixed at baseline,
    but its slope is fit across all 5 exposures over ~3 weeks, so some
    facultative tanning during the study may be blended into the existing
    0.278 route itself — a note on reading that route, not a new gap.
    [EVIDENCE NOTE 2026-09-05] Bogh 2010 (P36, already cited below for its
    cholesterol finding) found NO significant correlation between deltaD25(OH)D
    and pigmentation in its own regression, measured by reflectance-meter PPF
    at the buttock (constitutive, P=0.5) and upper body (facultative, P=0.4) in
    n=28 non-sun-worshippers (FST I-VI, Danish, wintertime, 3 SED x 4 to 24%
    BSA); nor any difference (P=0.7) between 9 dark-skinned (FST V-VI) and 9
    fair-skinned (FST I-IV) participants matched on baseline 25(OH)D, despite a
    highly significant PPF difference between them (P=0.008, both measures).
    This is a null finding the four fitted routes above don't engage with.
    Differing statistical power (n=28 vs the four routes' larger matched-dose
    or dosimetry samples), population (Danish, low ambient UV, wintertime),
    and dose framing (fixed low SED, not MED-scaled) plausibly explain the
    discrepancy; noted as an open caveat, not treated as invalidating K_const.
    See K_ATTEN note below.
- feedback_mode = 'biphasic'  (was 'linear' with D_max = 150.0):
    [CALIBRATION 2026-08-18, user-approved item A2] Hard synthesis ceiling
    WITHDRAWN as default, replaced by saturating form.

    WHY THE LINEAR FORM WENT. max(0, 1 - D/D_max) asserts cutaneous synthesis
    stops OUTRIGHT at D = D_max. No such switch exists: 25(OH)D homeostasis
    acts downstream on 1,25(OH)2D via CYP24A1 induction and substrate-driven
    catabolism — the latter already carried here by k_decay_D. D_max = 150.0
    had no source: absent from Bogh 2010, and PMID 19812604's observed
    maximum was 116 nmol/L.

    THE ADOPTED FORM is `vitd_physiology.hydroxylation_efficiency`:
        g(D) = 1 / (1 + (D / D_inflect) ** n),  D_inflect = 56.0, n = 2.0
    Scholze et al. 2026, Commun Med 6:105, bimonthly sampling, reports
    conversion of cutaneous D3 to plasma 25(OH)D3 is efficient at low
    25(OH)D3 and "decreases markedly" above inflection at 25(OH)D3 = 56 nmol/L.
    [ESTIMATE] 56.0 is a fitted OUTPUT (breakpoint on vitamin D3 axis), not
    a regressor, so it carries the breakpoint's own (unpublished) width.
    [CHOICE] n = 2.0 is NOT fittable from that paper — single reported
    breakpoint cannot estimate it. Chosen as the gentlest integer knee:
    g = 0.5 at inflection, 0.80 at half, 0.20 at twice. g(D) is never
    exactly zero — the whole point.
    [EVIDENCE NOTE 2026-09-05] Heaney et al. 2008 (AJCN 87:1738-1742, n=206
    across 6 aggregated studies) independently reports a biphasic serum
    25(OH)D-vs-D3 relationship (Y = 0.433X + 87.81[1-exp(-0.468X)], R^2=0.4482)
    — corroborates the qualitative SHAPE (fast-then-slow) but not n or
    D_inflect: their fit is linear+exponential, not Hill, and X is serum
    vitamin D3 (the precursor pool), not 25(OH)D feeding back on itself. Full
    note: `vitd_physiology.py` above `HYDROXYLATION_INFLECTION_NMOL_L`. Not
    used to change n = 2.0.

    WHAT WAS INVESTIGATED AND NOT ADOPTED. `feedback_mode='saturating'`,
    g(D) = exp(-ln2 * D / D_half_feedback), existed for Karppinen check only.
    Rejected on two grounds. (1) Scale parameter has no source: default
    D_half_feedback = 50.0 is unattributed, and the 90.0 used in Karppinen
    was FITTED to Karppinen — adopting it makes that check circular. (2) An
    exponential is steepest at D -> 0, contradicting Scholze's finding that
    hydroxylation is EFFICIENT below the inflection and falls above it.
    Cost of adopting anyway (A = 3.641 from Young 2021 anchor): Karppinen
    FAILS (low-baseline rise overshoots its 10 nmol/L bound). Retained
    unchanged for sensitivity work.

    WHAT HAPPENS TO D_max. NOT deleted: 'linear' and 'power' still read it,
    and vitd_sensitivity/scripts still sweep it. But OUT OF FORCE in the
    default model, so `D_max_override=` is INERT unless feedback_mode is
    explicitly 'linear' or 'power'. Any sensitivity result reporting D_max as
    influential describes a non-default configuration.

    MEASURED EFFECT ON THE HEADLINE, REPORTED IN FULL BECAUSE IT MOVES IT.
    The interval optimum under net_final_D_change_per_SED (FST III, 1 SED,
    84 d, D0 = 25) goes 3 d -> 5 d, and the band within 5% of peak goes
    [2,3,4] -> [3,4,5,6]. DECOMPOSED: feedback form ALONE (biphasic, old A)
    leaves FST III optimum at 3 d, band [2,3,4,5]; A recalibration ALONE
    (linear, A = 2.930) also leaves it at 3 d, band [2,3,4,5]. The shift to
    5 d is their INTERACTION — more synthesis per session raises steady-state
    D, which sits further up the feedback curve. Optimum remains graded by
    pigmentation and gets SHORTER with darker skin: FST I-II 5 d, III 5 d,
    IV 4 d, V-VI 2 d (incumbent: 3/3/3/2/2/1 d). Interval ORDERING is
    near-invariant (Spearman rho 0.964-0.999 against incumbent across
    FST II/III/V/VI); the ARGMAX moves. The claim that recalibrating A
    leaves "scheduling rankings unchanged" is TRUE for ordering and FALSE
    for reported optimum.
    HISTORICAL NOTE ON WHAT CARRIES THE INTERVAL ORDERING — TWICE CORRECTED.
    [2026-08-18] The claim in run_all.py section 17 that ordering "comes from
    the (1 - D/D_max) feedback term" was found too specific: at incumbent A
    the argmax is 3 d under linear, biphasic AND both saturating variants
    alike, so no particular feedback FORM carries it. That observation stands.
    The inference drawn from it at the time — that ordering comes from HAVING
    a decreasing feedback factor — does NOT stand, and is RETRACTED here.
    [2026-08-21] `spacing_study/` tested it directly by ablation: disabling
    g(D) entirely leaves the interval ordering unchanged and WIDENS daily
    dosing's margin rather than narrowing it, and turning off every state
    mechanism at once widens it further still. The ordering is instead a
    property of the CONCAVITY of the synthesis law A * ln(1 + sed * E_source)
    — at a fixed total UV budget, dividing it into more and smaller sessions
    always yields more 25(OH)D. D-feedback and photoadaptation modulate the
    SIZE of the gap; neither creates it. Sources for this retraction:
    spacing_study/results/FINDINGS.md (section 3, mechanism decomposition)
    and the generated RESULTS.md paragraph beginning "The interval ordering
    these sweeps produce comes from neither photoadaptation nor the
    D-feedback term".
    [FLAGGED 2026-09-06, NOT INVESTIGATED FURTHER] The "argmax is 3 d under
    linear, biphasic AND both saturating variants alike" claim from
    [2026-08-18] above NO LONGER HOLDS for 'saturating' under the current
    M_baseline_by_FST/K_const table (A held at the historical 2.353 for this
    specific check, FST III, 84 d): saturating's argmax moved from day 3 to
    day 2. `tests/test_vitd_model.py::TestFeedbackFormA2::
    test_interval_optimum_is_not_an_artefact_of_the_linear_form` is left
    FAILING deliberately rather than silently patched. The [2026-08-21]
    retraction (ordering comes from the concavity of the synthesis law, not
    from any particular feedback term) is UNAFFECTED by this -- it is
    ordering-invariance across feedback FORMS at one fixed parameter set that
    has weakened, a narrower claim than the retraction addressed, and this is
    a one-cell (FST III, saturating) shift, not a re-run of the ablation.
- D_max = 150.0 nmol/L:
    LEGACY. Ceiling of the withdrawn 'linear'/'power' feedback modes; see the
    feedback_mode entry above. Not in force in the default model.
    (WARNING: ORIGIN UNCONFIRMED — not found in Bogh 2010; PMID 19812604 max
    was 116 nmol/L. Corrected PMID 19798069 -> 19812604.)
- UV_halfmax_by_FST = {I-III: 2.0, IV: 5.0, V: 8.0, VI: 10.0}:
    Hill saturation constant for melanin induction kinetics (Ravnbak 2008 Table IV post-adaptation multi-exposure MMD, PMID: 18480919).
- Stratum Corneum (SC) Thickening:
    SC(t) is an active state variable, implemented as TWO processes whose
    difference is the net state:
        SC = SC_build - SC_damage
    SC_build  : slow hyperplasia toward SC_max, regressing at epidermal
                turnover rate (k_decay_SC, ~28 d half-life [PRIOR, see :826])
    SC_damage : fast UV-driven barrier disruption, additive per SED
                (k_damage), repairing on a days timescale (k_repair_SC)
    The difference may be NEGATIVE. de Boer 2025 Supp Table S1 shows SC
    thickness 5.8 -> 4.3 -> 7.5 um at T0 / 5 wk / 9 wk: it falls ~26% BELOW
    baseline before overshooting. The previous single multiplicative form
    (`SC *= exp(-k_damage*sed)`) could only shrink SC toward zero, so no
    k_damage could reproduce that. k_damage was pinned at 0.0 for this
    structural reason, which earlier docs misdiagnosed as missing data.

    k_damage REMAINS 0.0 by default. Fixing the structure did not make the
    parameter identifiable: with two informative timepoints and no error bars,
    the best achievable fit still misses by 71-75% of signal amplitude
    (`scripts/fit_sc_deboer2025.py`). The data need an ACCELERATING response
    under constant dosing, which no saturating or cascade process produces.
    Do not fit k_damage until individual-level trajectories are available.

    [PREMISE WARNING, STRENGTHENED 2026-09-06 -- see the K_SC calibration
    below; DO NOT WEAKEN THIS] K_SC attenuates synthesis as exp(-K_SC * SC),
    i.e. as added optical path length through a thicker barrier. de Boer
    attribute the SC change to EDEMA, not acanthosis, and in the same table
    epidermal thickness is flat (108.6 -> 109.4 um, p = 0.718). Fluid is not
    extra corneocyte layers. Bruls et al. 1984 make this warning WORSE, not
    just present: they measure that hydrating the SC by +56.6% (essentially
    all water, since water does not absorb UV) does NOT raise optical depth
    proportionally -- UV-B transmission RISES on hydration (their Fig. 4b).
    If de Boer's edema attribution is correct, the mechanism this parameter
    encodes may have THE WRONG SIGN, not merely the wrong magnitude, and
    K_SC = 1.509 below is then an upper bound on a quantity that could be
    near zero or negative. This is not resolved and is not expected to be
    resolved by picking a baseline thickness or a wavelength -- both of those
    choices only fix the MAGNITUDE conditional on the mechanism being right.
    Treat SC as an EMPIRICAL photoadaptation channel, not a validated optical
    mechanism -- and note K_ATTEN ~ K_SC is jointly unidentifiable anyway
    (vitd_sensitivity.py). See docs/SC_MODEL.md section 2.6.
    [EVIDENCE NOTE 2026-09-06] Bruls et al. 1984 (Photochem Photobiol
    40:485-494, DOI 10.1111/j.1751-1097.1984.tb04622.x) is the first DIRECT
    optical measurement this repo holds for the SC term. Not previously cited
    anywhere (checked against vitd_model.py, vitd_physiology.py,
    paper/references.bib, scripts/, docs; the Slaper hits in references.bib
    are van Dijk et al., a different paper).
    (1) SUPPORTS THE FUNCTIONAL FORM. Their Fig. 2 finds log-transmission
    linear in thickness across 254-546 nm, i.e. Lambert-Beer holds well enough
    that a single half-value layer d_1/2 characterises the SC. exp(-K_SC * SC)
    is the right SHAPE. Their own hedge, worth carrying: "the spread in the
    measurements is so large that a small deviation is easily obscured."
    (2) MAGNITUDE, AND WHY IT DOES NOT SETTLE K_SC. Their Table 3 gives in
    vivo SC d_1/2 = 9.9 um at 297 nm (5.5 at 280, 7.2 at 290, 12.3 at 302,
    17.1 at 313 -- a factor 1.8 across 290-302, so wavelength choice is not a
    detail). Because the model's SC is FRACTIONAL thickening and Bruls measure
    ABSOLUTE um, converting needs a baseline SC thickness d0 that the model
    does not have:
        K_SC = ln(2) * d0 / d_1/2
    At Bruls' own adopted d0 = 20 um (cryostatic 19.8 +/- 4.3) and 297 nm this
    gives K_SC = 1.4003, i.e. ~40% above the shipped 1.0 but inside the
    existing (0.5, 2.0) prior range. Inputs carry 2-3 s.f.; do not read more
    than ~4 into that figure. The wavelength spread alone (290-302 nm) moves
    it over 1.1271-1.9254, and Young 2021's ~5 nm blue shift of the action
    spectrum pushes the effective wavelength toward the HIGH end.
    (3) THE DOMINANT UNCERTAINTY IS INTERNAL TO THIS REPO, NOT TO BRULS. K_SC
    scales LINEARLY with d0, and the two studies behind SC_max disagree about
    d0 by 2.86x: de Boer 2025 Supp S1 baselines at 5.8 um, Miyamura 2011
    Table 1 at 16.6 um. Those alone give K_SC = 0.4061 and 1.1622. Across the
    full plausible d0 range (5.8-31 um) K_SC spans 0.4061-2.1705 -- wider than
    the whole current prior interval. A K_SC anchored to optics is only
    meaningful once the model commits to a baseline SC thickness. It does not
    currently.
    (4) CUTS AGAINST ITS OWN DIRECTION ON THE EDEMA QUESTION. The PREMISE
    WARNING above is, if anything, understated. Bruls state water does not
    absorb UV, and measure that hydrating the SC from ~19.8 to ~31 um (+56.6%,
    essentially all water) does NOT raise optical depth proportionally --
    UV-B transmission RISES on hydration (their Fig. 4b: drying DECREASES
    UV-B transmission). If de Boer's edema attribution is right, the SC term's
    sign is wrong, not just its magnitude, and the optical value above is an
    upper bound on a quantity that could be near zero. Bruls does not resolve
    the premise warning; it sharpens it into a testable disagreement.
    (5) DISAGREES WITH SC_max MORE SHARPLY THAN WITH K_SC. Bruls Table 1
    (n=6, lower back, twice weekly x 8 wk, escalating UVB) measures SC
    26 +/- 3 um after a rise of 14 +/- 4 um, i.e. +53.85% thickening --
    1.50x the SC_max default of 0.30 and 2.24x the FST I-III 0.24, and well
    outside the 0.20-0.40 evidence-based prior. Caveats: n=6, isolated
    saline-hydrated samples, escalating rather than fixed suberythemal dose,
    all-Caucasian, no FST recorded. Both prior sources (de Boer, Miyamura)
    sit low and this sits high; the upper bound of that prior is not as
    settled as two concordant studies made it look.
    (6) CONTRADICTS de BOER ON EPIDERMAL THICKNESS. Bruls Table 1 has the
    epidermis thickening +53.6% (69 um, +37 um) "practically the same
    proportion" as the SC, so the viable layer thickens too. de Boer has
    epidermal thickness FLAT (108.6 -> 109.4 um, p = 0.718) -- which is the
    PREMISE WARNING's own argument for edema. Dose regime is the obvious
    reconciliation (Bruls escalating and erythemally relevant, de Boer 0.8 SED
    suberythemal), and if that is the explanation then the photoadaptation
    channel is DOSE-REGIME DEPENDENT in a way neither SC_max nor K_SC
    currently expresses.
    CAVEATS ON ALL OF THE ABOVE: n=8 Caucasian volunteers aged 20-30 (11 with
    the UV-exposed arm), in vitro isolated epidermis/SC, lower back and upper
    leg, collimated perpendicular incidence, scattering and absorption LUMPED
    (not separated), discrete Hg lines only (no line between 302 and 313), and
    Table 3 is Bruls' own reconstruction from wet/dry measurements rather than
    a direct in vivo reading -- they say so. NO DARK SKIN: applying any Bruls-
    derived K_SC to FST IV-VI is a light-skin extrapolation, and FST IV-VI
    carries the LARGER SC_max (0.36). Nothing here bears on K_const or
    K_ATTEN; melanin is mentioned once, speculatively, and the protocol
    produced "no intense tanning" by design.
    [CALIBRATION 2026-09-06, user-approved] SC_BASELINE_THICKNESS_UM
    introduced (NEW constant, 20.0 um, module-level, documented above near
    BSA_EXP; consumed ONLY by this derivation, not wired into any state
    equation) and K_SC: 1.0 -> 1.509.
    BASELINE CHOICE. d0 = 20 um, Bruls' own adopted in vivo estimate. Decided
    on MEASUREMENT-METHOD grounds, not by picking a convenient number:
      - de Boer 2025 (5.8 um): NOT a direct thickness reading. It is the
        fitted plateau of a Boltzmann sigmoid on cumulative tape-strip
        protein mass (Squame Scan 850 densitometry), in vivo, unfixed, lower
        back. de Boer's own OCT channel explicitly could NOT resolve SC
        thickness ("covers 1-2 pixels ... not reliably segmented", p.2310).
        An indirect biochemical proxy, not a ruler measurement.
      - Miyamura 2011 (16.6 um): UNVERIFIABLE in this pass. The PDF is not in
        Research_Papers/ or Papers_and_Datasets/ (checked by filename and a
        full-repo grep for "miyamura" and PMID 20979596, 2026-09-06) despite
        being cited elsewhere in this file as SC_max's out-of-sample
        corroboration -- FLAGGED per project rule rather than silently
        assumed comparable; obtain the PDF before leaning on its method.
      - Bruls 1984 (20 um): their own cryostatic in vivo measurement
        (19.8 +/- 4.3 um), cross-checked against two OTHER independent in
        vivo readings for the same body region (Klaschka 1977; Hulsbergen
        Henning et al. 1977, both ~18 um). Bruls' decisive argument for
        preferring a corrected in vivo estimate over raw ex vivo/histological
        readings: histological sections read systematically LOW from
        dehydration shrinkage -- they note the frequently-cited Holbrook &
        Odland 10 um back SC sits below even Bruls' own DRIED ex vivo sample
        average (13.5 um). Bruls measure ex vivo samples dried and on saline
        of varying concentration, then back-calculate an "approximate in
        vivo" value from estimated in vivo SC water content -- a correction
        neither de Boer's nor (so far as verifiable) Miyamura's method
        applies. This is a method argument, not a preference for the largest
        number: Bruls' own dried ex vivo readings sit BELOW 20 um too.
    WAVELENGTH CHOICE. Rather than pick a single line from Table 3, Bruls'
    d_1/2(lambda) (converted to absorption coefficient mu = ln(2)/d_1/2 and
    linearly interpolated across their five points: 280 nm 5.5 um, 290 nm
    7.2 um, 297 nm 9.9 um, 302 nm 12.3 um, 313 nm 17.1 um) was convolved
    against data/spectra/vitd_action_spectrum_young2021.csv -- this repo's
    preferred blue-shifted vitamin-D action spectrum -- over their common
    range, 284-313 nm (weighted mean wavelength 296.11 nm):
        mu_eff = sum(w(lambda) * mu(lambda)) / sum(w(lambda)) = 0.075446 /um
        d_half_eff = ln(2) / mu_eff = 9.1873 um
        K_SC = ln(2) * 20.0 / 9.1873 = 1.5089  ->  1.509
    LIMITATION, flagged rather than silently assumed away: no ground-level
    solar spectral-irradiance file exists in this repo, so the convolution
    weight is the ACTION SPECTRUM ALONE, not action-spectrum x source SPD as
    the approval ideally specifies. A true solar-SPD weighting would further
    suppress the shortest wavelengths (steep ozone absorption below ~305 nm),
    pushing the effective wavelength LONGER and K_SC somewhat LOWER than
    1.509. Discarded alternatives, for comparison: CIE (unshifted) action
    spectrum convolution, weighted mean 300.85 nm, K_SC = 1.231; single line
    at Bruls' own measured 297 nm, K_SC = 1.4003; single line at their
    interpolated 300 nm, K_SC = 1.336; Diffey (secondhand via Bruls p.492,
    8.5 um at 300 nm), K_SC = 1.6309. All sit inside the pre-existing
    (0.5, 2.0) prior range. Carries 4 significant figures at most, per the
    2-3 s.f. inputs.
    THE PREMISE WARNING ABOVE IS UNCHANGED BY THIS CALIBRATION and governs
    how to read 1.509: it fixes the MAGNITUDE of K_SC conditional on
    exp(-K_SC*SC) being the right mechanism; it says nothing about whether
    that mechanism (versus an edema artefact with the wrong sign) is correct.
    [SOURCE NOT AVAILABLE 2026-09-06] Diffey 1983 (Phys Med Biol 28:647-657,
    DOI 10.1088/0031-9155/28/6/005, PMID 6878426) was fetched for this pass
    but the file in Research_Papers/ is a MIS-DOWNLOAD: IOPscience served
    0031-9155/28/1/005 (Maughan & Roper, neutron and gamma-ray dosimetry for
    the d-Be reaction) instead of 28/6/005 -- wrong issue number in the URL.
    Verified independently against the PDF's own cover page. The filename says
    otherwise; do not trust it. Diffey is the more informative of the two for
    this parameter because he separates forward from backward scattering,
    which Bruls does not. All that is known of it here is secondhand, via
    Bruls p.492: d_1/2 = 8.5 um at 300 nm and 14 um at 350 nm, read by Bruls
    off Diffey's Fig. 3. At d0 = 20 um that 8.5 um implies K_SC = 1.6309 --
    33% more attenuating than Bruls' own interpolated 11.34 um at 300 nm, an
    agreement Bruls call "reasonable" and which is a third of this parameter's
    value. Treat as provisional until the correct PDF is obtained.
    [C7-SC] SC_max = 0.30 is a PRIOR, not a calibrated constant. De Boer 2025
    (P2) supports an SC thickening range of +23% (FP I-III) to +36% (FP IV-VI);
    evidence-based prior is 0.20-0.40. Do NOT tighten alpha_SC, k_decay_SC, or
    K_SC without individual-level time-course data from de Boer 2025 supplementary.
- E_source (SOURCE CALIBRATION TABLE):
    [C3] E_source must be a per-study calibration input, NOT a fixed multiplier.
    The SOURCE_CALIBRATION_TABLE below provides evidence-derived starting priors
    from Webb 2021 (P21) and Grigalavicius 2015 (P35). These are priors only;
    Young 2021 PNAS (P20) shows the CIE previtamin-D3 action spectrum needs a ~5 nm
    blue shift, meaning exact values require recomputation. Osmancevic 2009 (P18)
    shows broadband UVB can outperform NB-UVB per SED in psoriasis, confirming
    these cannot be universal.
- K_ATTEN = 0.288972  --  CALIBRATED 2026-08-21 (user-approved), excess_M updated 2026-08-25:
    Facultative (tanning) melanin attenuation coefficient.
    Reinstated as an active parameter anchored to Matsuoka 1990 (PMID 2165512, n=7
    within-subject tanned-vs-untanned arm synthesis) with biphasic feedback g(D)
    deconvolved (residual melanin attenuation 6.6%, K_ATTEN = 0.288972; plausible
    range across feedback forms: [0.288972, 0.838389]). derive_K_ATTEN's excess_M
    now reads M_TAN_HEADROOM live rather than a hardcoded value, so this figure
    tracks M_TAN_HEADROOM automatically.
    MECHANISM: exp(-K_ATTEN * max(0, M - M_baseline)) is evaluated on every session.
    A full tan (excess_M = M_TAN_HEADROOM = 0.238) gives 6.6% attenuation at
    K_ATTEN=0.288972 and 18.1% at K_ATTEN=0.838389.
    Caveats: Matsuoka 1990 is n=7 and collinear with K_SC; 0.288972 provides the
    evidence-based lower bound from the only within-subject trial in existence.
    [EVIDENCE NOTE 2026-09-05, CORRECTED 2026-09-06] "Young 2019" (see the
    CORRECTION in the K_const entry above — it is Young 2020, already the
    source of one of the four K_const routes, not new evidence) reports FST
    II:VI melanin inhibition factors of 1.29-1.43x under REPEATED
    suberythemal dosing (5 exposures over ~3 weeks). Their design cannot
    cleanly separate constitutive from facultative pigmentation (FST/ITA
    fixed at baseline, response fit across the whole repeated-exposure
    course), so it does not isolate a facultative-only estimate to compare
    K_ATTEN against directly. K_ATTEN unchanged; this is not independent
    corroboration or contradiction of it either way.
- M_baseline_by_FST:
    [C7] Anchored to ITA_Unexposed means from the Young 2020 trial workbook
    `Papers_and_Datasets/Final skin type vit D database 26_03_2020.xlsx`.
    Mapping: M = 1 - (ITA_U + 30) / 85, clipped to [0, 1].
    PROVENANCE CORRECTED 2026-08-23 (documentation only; no value changed).
    The previously stated "N=44" is the ROW COUNT of the `ArimedB irradiated`
    sheet, not the sample behind these numbers: only 28 of those 44 rows carry
    an ITA_Unexposed value, and those 28 alone give {II 0.07, III 0.19,
    IV 0.48, V 0.77, VI 0.96} — which is NOT the shipped table. The subset that
    reproduces it is `ArimedB irradiated` POOLED WITH `ARIMED Controls`,
    n = 48 rows with a non-null ITA_Unexposed (the two TL01 sheets contribute
    none), taking the per-FST mean of ITA_Unexposed and then applying the
    mapping. That route returns {II 0.06, III 0.21, IV 0.45, V 0.78,
    VI 1.077 -> clipped 1.00}: exact for III, V and VI, and within 0.01 for
    II and IV. No subset or aggregation order reproduces all five cells
    exactly, so the shipped values are best described as ROUNDED group means
    from the pooled n=48 set, not a transcription. Per-FST n in that set is
    II 13, III 15, IV 12, V 4, VI 4 — the two darkest cells rest on n=4 each,
    and the FST VI cell is a clipped boundary, not a mean.
    FST I is extrapolated (ITA_U ≈ 58°). Values:
    {I: 0.0, II: 0.07, III: 0.21, IV: 0.44, V: 0.78, VI: 1.00}.
    [C10-PRIOR] GROUP-LEVEL PRIOR — Datta 2019 (P13, PMID: 30656228) shows
    measured pigmentation explains only 1.9-2.6% of individual response variance
    (note Datta 2019 studied pigment SNPs: ASIP, SLC24A4, SLC45A2, not VDR/CYP2R1/CYP24A1).
    Do NOT use as a precise individual predictor or to justify large FST penalties.
    Genetics (VDR, CYP2R1, CYP24A1), supplementation, baseline D, and BSA dominate.
    WARNING: FST V (0.78) and VI (1.00) are group-level priors with NO direct
    melanin-index validation from published data. Do not use as individual predictors.
    [EVIDENCE NOTE 2026-09-06] Two non-UK constitutive-pigmentation papers read
    in full. Neither previously cited (checked vitd_model.py, vitd_physiology.py,
    paper/references.bib, scripts/, docs, calibration_approval.yaml, tables/).
    NOT-INDEPENDENT WARNINGS, both real: (i) Linde 2020 shares a lab (Wright +
    du Plessis) with `wilkes2015`, already in references.bib and the source of
    the "MI>750 -> FST VI" claim in the Gap 1 block below — Linde's Mexameter
    thresholds ARE the Wilkes scale, so the two are not independent on
    classification. (ii) Narea's total-melanin calibration derives from Alaluf
    et al. 2002 (Pigment Cell Res 15:119-126), which is NOT the `alaluf2001`
    (Pigment Cell Res 14) already in references.bib — same first author,
    adjacent years and volumes, exactly the shape of the Young 2019/2020 error.
    Distinct papers; do not merge.
      Narea-Jimenez et al. 2026 (Biomed Phys Eng Express 12:025078, DOI
    10.1088/2057-1976/ae6458; note this journal has no page ranges — a citation
    without the article number does not resolve). n=1005 diffuse-reflectance
    spectra, Venezuelan mestizo dermatology outpatients (Carabobo), inner left
    forearm 8 cm proximal to elbow, a site chosen for minimal chronic actinic
    exposure. FST CLINICIAN-ASSESSED, and spectral acquisition was run by
    separate optical specialists, so the colour reading did not inform the
    phototype label. Constitutive ITA by FST (mean +/- SD): I 51.76 +/- 6.40
    (n=25), II 40.42 +/- 9.77 (n=95), III 27.54 +/- 10.67 (n=369), IV
    10.45 +/- 9.71 (n=400), V -9.08 +/- 9.98 (n=84), VI -33.63 +/- 15.91
    (n=32).
      Linde, Wright & du Plessis 2020 (Skin Res Technol, DOI 10.1111/srt.12895).
    n=71 Limpopo farmworkers (61 Black African, 10 White), summer, inner upper
    arm as the constitutive site. Constitutive ITA group means: Black African
    -28.07, White 48.42.
    (1) GAP 1 BELOW IS NOW PARTLY FALSE AND SHOULD BE READ WITH THIS NOTE.
    It states "No paper provides separate untanned FST V and FST VI baseline MI
    means." Narea provides separate untanned FST V (n=84) and FST VI (n=32)
    constitutive means with dispersion, from a sun-protected site, with
    clinician-assigned FST. It reports ITA and melanosome fraction, not
    Mexameter MI — but the shipped table is ITA-derived, not MI-derived, so ITA
    is the relevant currency. n rises 21x at FST V and 8x at FST VI over the
    n=4 cells.
    (2) APPLYING THIS MODULE'S OWN MAPPING, M = 1 - (ITA_U + 30)/85, TO NAREA:
        I   0.0381    (shipped 0.00,  diff +0.0381)
        II  0.1715    (shipped 0.07,  diff +0.1015)
        III 0.3231    (shipped 0.21,  diff +0.1131)
        IV  0.5241    (shipped 0.44,  diff +0.0841)
        V   0.7539    (shipped 0.78,  diff -0.0261)
        VI  1.0427 -> clipped 1.00 (shipped 1.00, raw diff +0.0427)
    THE PATTERN IS THE OPPOSITE OF WHAT GAP 1 ANTICIPATES. The two darkest
    cells — the n=4 ones flagged as coarse priors — are the two the external
    n=84/n=32 cohort AGREES with (V within 0.026, VI within 0.043 pre-clip).
    The mid-scale cells built on n=12-15 are where the disagreement is largest.
    Read as implied ITA_U, the UK Young 2020 cohort is 7-10 degrees LIGHTER at
    FST II, III and IV than the Venezuelan cohort at the same nominal
    phototype, and agrees at V (2.22 deg) and VI (3.63 deg). The natural
    reading is not that V/VI are right and II-IV wrong, but that FST is
    POPULATION-ANCHORED: the same label denotes different constitutive pigment
    in a UK trial cohort than in a Venezuelan one, most strongly mid-scale
    where the label is least determined by ancestry. Both papers say this
    independently — see (5).
    Linde's only clean cell-level anchor is FST II via its White group
    (M = 0.0774 vs shipped 0.07, within 0.0074, from a South African cohort on
    a different instrument). Its Black African group is 100% objectively FST V
    or VI but is NOT broken out by phototype, so it gives a large-n POOLED V+VI
    constitutive mean of M = 0.9773 (n approx 61) and nothing that resolves the
    V-vs-VI contrast the shipped table asserts.
    (3) THE CLIP IS LOAD-BEARING, NOT A RARE EDGE CASE. This is the most
    consequential finding in this pass. M = 1.0 falls exactly at ITA = -30,
    which is also Del Bino's Brown/Dark boundary — i.e. the MIDDLE of the
    observed dark-skin range, not its edge. Consequences, from two unrelated
    populations and two unrelated instruments:
      - FST VI's GROUP MEAN clips in both cohorts independently: Young 2020 raw
        1.077, Narea raw 1.0427. The ceiling is hit by the mean, not by a tail.
      - On Narea's mean and SD, an estimated 59.02% of FST VI individuals clip
        (z = 0.2282). FST V is essentially unaffected (1.80%).
      - 48% of Linde's Black African CONSTITUTIVE readings sit beyond it; its
        25th percentile already clips (raw 1.0401) and its constitutive minimum
        reaches raw 1.1987. At the facultative site only the single lightest
        hand in the cohort escapes.
      - Observed dark extremes run well past the ceiling: Linde constitutive
        min ITA -46.89 (all-site -66), Narea FST VI -1SD -49.54.
    So within FST VI the mapping has essentially no resolving power: an
    individual at ITA_U = -66 and one at -30 receive an identical melanin
    penalty. This is a MAPPING-SHAPE problem, not a sampling problem, and more
    n will not fix it. Note also that the denominator (85), the offset (30) and
    the clip are NOT separable knobs — moving the ceiling to the observed dark
    extreme would need a denominator near 121, which rescales every lighter
    cell too, and K_const multiplies the whole result. Recorded, NOT acted on.
    (4) INTRA-PHOTOTYPE SPREAD SWAMPS INTER-PHOTOTYPE SPACING. Narea's per-FST
    SDs in M units are I +/-0.0753, II +/-0.1149, III +/-0.1255, IV +/-0.1142,
    V +/-0.1174, VI +/-0.1872 — every one larger than the shipped II->III step
    (0.14) or comparable to III->IV (0.23). FST V's +1SD (0.636) and FST IV's
    -1SD (0.638) are contiguous, consistent with Narea's own SSIM(IV,V) = 0.94.
    This corroborates the [C10-PRIOR] warning above from a completely different
    direction than Datta 2019: not only does pigment explain little RESPONSE
    variance, the FST label does not cleanly index PIGMENT.
    (5) FST IS A POOR PIGMENTATION PROXY IN BOTH NON-UK COHORTS, BY TWO
    DIFFERENT MECHANISMS. Narea: spectral overlap — fme interval overlap up to
    52.05% between neighbours, SSIM 0.42-0.94, and their own clinician-assigned
    FST I has mean ITA 51.76, below Chardon's Very-Light threshold of 55, i.e.
    not even in the ITA category FST I is meant to occupy. Linde: self-report
    bias — 62% of Black African farmworkers self-reported FST I, III or IV
    while 100% measured objectively as FST V or VI; only 5% self-reported VI
    where 48% measured VI. The authors rule out simple confusion for the 46-47%
    who chose FST IV and attribute part of it to a social preference for
    lighter skin. DIRECT CONSEQUENCE FOR THIS TABLE: a model that looks up
    M_baseline from a SELF-REPORTED FST would, on Linde's cohort, assign
    M = 0.44 (FST IV) to a modal worker whose measured constitutive M is 0.977
    — an error of 0.537 in M, i.e. a factor of exp(0.98 * 0.537) = 1.694 in the
    melanin term, larger than any of the four within-study ratios K_const was
    fitted to. Both papers independently call for a category beyond FST VI.
    (6) STRUCTURAL SUPPORT FOR THE EXPONENTIAL FORM, WHICH THE REPO DID NOT
    HAVE. Narea's melanosome volume fraction fme enters radiative transport as
    mu_a,epi = fme * 6.6e11 * lambda^-3.3, so Beer-Lambert gives transmission
    exponential and LINEAR in fme. Regressing this module's M(ITA) against fme
    across Narea's six phototypes gives M = 0.0969*fme + 0.0052 over FST III-VI
    (residuals within +/-0.021, intercept indistinguishable from zero;
    r^2 = 0.9835 over all six). So exp(-K_const * M) is approximately
    Beer-Lambert-consistent over III-VI, with K_const absorbing path length,
    cross-section, scattering and physiology. This supports the FORM only.
    It says NOTHING about K_const's magnitude: extrapolating Narea's eq. 9 to
    300 nm (far outside its 400-700 nm fitting range, single-pass, no
    scattering, assumed 100 um epidermis) gives an I:VI transmission ratio of
    52.8x against the model's 2.676x. That 20-fold gap is EXPECTED — K_const
    was fitted to serum 25(OH)D response ratios that already fold in feedback,
    baseline-D headroom and protocol — and must NOT be cited as an optical
    estimate of K_const.
    (7) INCIDENTAL, BEARS ON K_ATTEN NOT ON THIS TABLE. Linde gives a
    within-subject constitutive->facultative field contrast in the same cohort,
    season and instrument: Black African dITA -13.90 (dM +0.1635), White dITA
    -28.92 (dM +0.3402). The dark-skinned group tans about half as much as the
    light-skinned group on this scale (ratio 0.4806), and the authors argue the
    textbook claim that FST V-VI "tan profusely" is wrong because V and VI were
    bolted onto a scale originally defined by UV response. This is a
    cross-sectional site contrast, not a controlled dose contrast — the two
    sites differ in delivered dose as well as response — so it bounds nothing
    on its own, but the light/dark asymmetry is within-cohort and
    same-protocol. Narea has no exposure contrast at all and bears on K_ATTEN
    not at all.
    CAVEATS. Narea: 76.5% of the cohort is FST III+IV, so the extremes are thin
    in absolute terms even at n=1005; subjects are dermatology OUTPATIENTS and
    the paper does not report excluding pigmentary disorders, an unaddressed
    confound; season not reported; no inter-rater reliability statistic for the
    FST assignment (three residents each saw different patients); individual-
    level data not public, so the 59.02% clip fraction is model-based, not
    counted; the fme retrieval rests on constants Z = 0.2796, k = -7.174 from
    the authors' own 2015 Spanish-language optics paper with no independent
    validation here, and its headline rho = 0.996 against "biochemical" total
    melanin is NOT external validation — both quantities are smooth monotone
    functions of the same L*, so they correlate by construction. Linde: n=10 in
    the White group; summer only, so the constitutive site is sun-protected but
    not never-exposed; FST is self-reported through an interpreter using a
    BESPOKE six-item instrument with the burning and tanning questions removed
    as culturally inapplicable — not the standard Fitzpatrick instrument, so
    FST-indexed comparison to a trial using the standard one is
    instrument-mismatched; outdoor vs indoor job type made no difference
    (P = .17), so there is no unexposed control in the cohort.
    INSTRUMENTS ARE NOT INTERCHANGEABLE. Narea used a HunterLab MiniScan XE
    Plus reflectance spectrophotometer (400-700 nm) -> CIELAB -> ITA; Linde used
    a Courage+Khazaka CL 400 colorimeter for ITA and a Mexameter MX18 for MI.
    Mexameter MI is a proprietary narrowband index on an arbitrary scale and is
    NOT L*a*b*, NOT ITA, and NOT this module's M — only the ITA arms of the two
    papers are comparable to each other and to this mapping. Do not carry MI
    numbers into anything ITA-calibrated. Note also that the two papers use
    DIFFERENT ITA classification schemes (Linde uses Del Bino with a -30 Dark
    cut; Narea uses Chardon, which has no Dark class and lumps V with VI), so
    their category names must not be conflated either.
    PROVENANCE FLAGS ON THE SOURCES THEMSELVES. Linde's PDF on disk is the
    EARLY-VIEW version, paginated "2020;00:1-9" with no volume, issue or page
    range; the 26(6):923-931 / PMID 32621354 identifiers are print-only and
    cannot be verified from the file — if this is ever entered in
    references.bib, take volume/issue from PubMed and record both. Linde's own
    ABSTRACT inverts its constitutive/facultative labels relative to its
    Methods section 2.3 (which is unambiguous: inner upper arm = constitutive);
    all numbers above follow 2.3. Linde's Discussion also says "approximately
    40%" FST VI where its own Figure 3 and Results say 48%.
    LEGIBILITY FLAG (project rule). Linde 2020 contains NO numeric results
    table — every ITA and MI value above is DIGITISED FROM BOX PLOTS
    (Figures 5 and 6), not printed. The digitisation was done programmatically
    (10x render, axis-tick centroid fit) and validates against the paper's own
    printed extremes: extracted MI max 1188.8 vs printed 1188, ITA max -5.00
    and min -65.72 vs printed -5 and -66. Residual error approx +/-0.5 deg ITA,
    which propagates to +/-0.0059 in M. Note the box centre line is the MEAN,
    not the median (stated in the captions), which is why it is usable here.
    Any Linde figure ever written into this file must be labelled
    digitised-from-Figure-5 with that uncertainty attached. Two Linde
    sub-values could not be read reliably and were NOT guessed: the White
    group's inner-upper-arm MI quartiles, and the volar-forearm ITA quartiles.
    The ITA<->MI correlation coefficients in its Figures 7-8 are also not
    printed and not legible.
- BSA_EXP = 0.3281  --  CALIBRATED 2026-09-06 (user-approved), was 0.50:
    Body-surface-area scaling exponent, bsa_factor = (bsa / BSA_REF) ** BSA_EXP.
    [EVIDENCE NOTE 2026-09-06] Two papers read in full for this parameter.
    Neither was previously cited. IMPORTANT NAME COLLISION: the repo already
    cites `libon2013` (Dermatology 227:250-254, skin colour) at :438 -- that is
    a DIFFERENT study by an overlapping author group, not the 2017 BSA letter.
    Do not merge them.
    THE HEADLINE IS THAT 0.50 IS NOT THE VALUE ANY SINGLE SOURCE GIVES. The
    available estimates fall into two clusters with 0.50 sitting in the gap
    between them, not at the centre of one distribution:
      PLATEAU CLUSTER, ~0.38 -- Young 2021's within-study 85% vs 3.7% BSA
        contrast implies 0.3790 (already recorded in the A entry above), and
        Jager 2018 Table III's within-study NB-UVB triple (Vahavihu, 13 SED,
        6/29/100% BSA) independently implies 0.3843 by log-log OLS (0.3723 on
        endpoints). Two different studies, sources, doses and BSA ranges
        reproducing each other to two decimals is the most striking convergence
        in this pass. That triple's own shape is a genuine plateau: exponent
        0.6421 over 6->29% BSA, then 0.0289 over 29->100%, i.e. essentially
        flat past ~30% -- close to the ~33% Matsuoka plateau cited below.
      NEAR-LINEAR CLUSTER, ~0.8-0.9 -- Libon 2017 (JID 137:1391-1393, n=72,
        FST III, single 0.8 MED NB-UVB, five between-subject groups at 9/23/50/
        96% BSA) measures CHOLECALCIFEROL AUC, whose control arm is ~0
        (0.73 +/- 2.14), giving the cleanest direct read of cutaneous synthesis
        vs area in either paper: control-corrected log-log OLS exponent 0.8892,
        R^2 = 0.9862, with NO plateau to 96% BSA. A straight line fits it
        marginally better still (through-origin R^2 = 0.9964), matching the
        authors' own "constant increase according to the body surface exposed".
        Jager's own matched-dose BB-UVB pair (Bogh, 3 SED, 6% vs 24%) implies
        0.8011. Over Libon's 9->96% span the model UNDER-predicts the observed
        response ratio by 2.4951x.
    AND THE INCUMBENT JUSTIFICATION DOES NOT GIVE 0.50 EITHER. Bogh 2011 / P5's
    `5.5*ln(%BSA)` is LOGARITHMIC, so its local exponent is 5.5/delta and falls
    as the response grows -- it is level-dependent by construction and no fixed
    BSA_EXP can represent it. Anchored at Bogh's own observation (delta = 24.7
    at 24% BSA / 12 SED) it implies 0.2663 over 6->24% BSA and 0.2509 over the
    model's own 3.7->85% scenario span. The docstring line above is right that
    P5 supports DIMINISHING RETURNS; it is the number 0.50 that P5 does not
    supply -- 0.50 is roughly what P5 gives only at delta ~ 11 nmol/L, the low
    end of observed responses.
    THE TWO CLUSTERS MAY BE MEASURING DIFFERENT THINGS, WHICH MATTERS
    STRUCTURALLY. Libon's near-linear result is on the D3 channel; the plateau
    signal lives in the 25(OH)D channel. That is mechanistically coherent --
    synthesis scaling with area while the serum response saturates downstream --
    but it raises the question of whether the plateau is being DOUBLE-COUNTED
    here, once in BSA_EXP and again in the biphasic g(D). Recorded as a
    structural observation, NOT acted on.
    [DOUBLE-COUNT WARNING] Jager 2018 (Anticancer Res 38:1165-1171, PRISMA/
    MOOSE, 1099 screened -> 15 studies, n=829) is NOT independent evidence for
    this model. Its pool re-aggregates Bogh 2011 (P5 itself -- the source of the
    ln-BSA coefficient above), Bogh 2010 (P36), Bogh 2012 (the DEBUGGING_TARGET
    anchor), Datta 2012, Rhodes 2010, McKenzie 2012, Karppinen and Ala-Houhala.
    Treating its BSA finding as corroboration would count P5 against itself.
    Its only genuinely new contributions here are the cross-study aggregation
    FORM and its finding of non-significance. (Farrar, Kift, Felton, Young and
    Matsuoka are NOT in its pool. Vahavihu, supplying the 0.3843 triple above,
    is not otherwise used by this repo.)
    Its meta-regression, E(dD) = 30.18 + 0.19*SED - 0.39*baseline + 0.05*BSA%,
    enters BSA linearly and is the WEAKEST number in the set: explicitly NOT
    significant, printed with no SE, no CI and no p; no I^2, tau^2, Q or forest
    plot appears anywhere in the paper. Forced into power-law units it implies
    0.1448 (10->100% BSA at baseline 50, 1 SED) or 0.0999 over the model's
    3.7->85% span -- but the fit is structurally wrong, not merely low: the BSA
    term contributes at most 5.0 nmol/L against a 30.18 intercept, and at
    baseline 100 nmol/L the formula returns NEGATIVE predictions, which no
    power law can represent. The paper concedes the point ("results obtained
    from meta-regression should be interpreted with caution"). Do not read
    0.05/%BSA as an estimate of anything. Equally, do NOT read its pooled
    0.19 nmol/L per SED as a competing estimate of A: it comes out of a
    multivariable fit whose intercept absorbs most of the level, whereas A's
    Young 2021 anchor is a raw slope not forced through zero.
    CAVEATS ON BOTH: neither paper contains a single dark-skinned subject.
    Jager could not consider skin type at all ("information was missing,
    incomplete or not comparable") and flags it as the leading confounder in
    its own abstract; its constituent studies are overwhelmingly N-European.
    Libon is FST III only, Belgian medical students, ages 19-29, BMI 18-25.
    Applying either paper's BSA scaling to FST V-VI is a LIGHT-SKIN
    EXTRAPOLATION, and because bsa_factor multiplies A directly it COMPOUNDS
    the extrapolation flag already carried on A above rather than mitigating
    it. Dose framing differs between them and from each other: Jager is
    absolute SED throughout, Libon is fractional MED (0.8 MED, individually
    titrated), so Libon's gradient is measured under per-subject-normalised
    erythemal response, not matched absorbed dose. Jager pools BB-UVB and
    NB-UVB rows with no spectral weighting -- and those two spectra give
    OPPOSITE exponents within its own Table III (BB 0.8011, NB 0.3843), which
    given Young 2021's finding that SED is a poor spectral predictor is a
    serious limitation of the pooling, not a detail.
    ARITHMETIC IN THE SOURCES THAT DOES NOT REPRODUCE (both verified, both
    minor): Jager's "8-fold" efficiency claim is not recoverable from its
    printed Table III (the candidate row pairs give 9.25 and 5.56). Libon's
    headline "7.84x more vitamin D per percentage" recomputes to 7.7792 -- and
    more importantly that whole gradient is computed on UN-baseline-corrected
    25(OH)D3 AUC against a ~249 nmol/L control, so it is a baseline artefact:
    on the clean D3 channel the per-%BSA yields are essentially flat across
    sites (1.0522 / 0.7596 / 0.7594 / 0.8039), leaving a residual head-and-hands
    premium of ~1.3x, not 8x. The model has no body-site term; on this evidence
    that omission is a real but SECOND-ORDER structural gap, small against the
    spread already carried by A (2.9-5.1) and K_const (0.3-1.5).
    ALSO UNCAPTURED: Libon reports time-to-peak in D3 shifting with area
    (day 1 at 9% and 23% BSA, day 2 at 50% and 96%). No BSA-dependent kinetic
    term exists in this model.
    SECONDHAND, NOT VERIFIED AGAINST THE PRIMARY SOURCE: Libon restates
    Matsuoka 1990 (P8) as concluding "at least 19% of BSA should be exposed to
    induce an increase in cholecalciferol" -- a lower and more specific threshold
    than the ~33% plateau recorded elsewhere in this docstring. Read only as
    quoted by Libon; re-read the Matsuoka PDF directly before relying on it.
    [CALIBRATION 2026-09-06, user-approved] 0.50 -> 0.3281. Derivation:
    `scripts/bsa_exp_recalibration.py` (bucket FEW-TIME, gitignored;
    confirmed not imported by run_all.py before being ignored).
    THE APPROVAL EXPLICITLY FORBADE SETTING 0.38 DIRECTLY: both apparent
    exponents above (0.3790 Young, 0.3843/0.3723 Vahavihu) are measured on
    response RATIOS that already contain g(D) feedback damping, so using
    them as BSA_EXP would double-count that damping. The approved method:
    find the BSA_EXP at which THIS MODEL's own predicted response ratio
    across each source's BSA span (feedback_mode='biphasic', the default)
    equals that source's observed ratio.
    RESULT, IN THE OPPOSITE DIRECTION FROM WHAT WAS ANTICIPATED. The
    approval's derivation clause expected the deconvolved value to come out
    ABOVE 0.38 (possibly >= 0.50), on the reasoning that g(D) suppresses the
    higher-BSA arm's synthesis more and so needs MORE geometric scaling to
    reproduce the same ratio. That is not what this model's own equations
    return:
        Young 2021 (85% vs 3.7% BSA)         -> 0.3310
        Vahavihu triple (100% vs 6% BSA)     -> 0.3254
        adopted (geometric mean of the two)  -> 0.3281
    Both deconvolved values sit BELOW their apparent counterparts (0.87x),
    not above.
    WHY: k_decay_D(D) in this model acts on the WHOLE serum D state,
    including the pre-exposure baseline, not only the incremental
    synthesis. Over a multi-week course this decay is a near-fixed absolute
    drag that removes a much larger FRACTION of the low-BSA arm's small
    gross synthesis than of the high-BSA arm's large gross synthesis. That
    widens the modelled response ratio at a GIVEN BSA_EXP in the SAME
    direction as raising the exponent would -- the opposite sign from the
    g(D)-suppresses-the-high-arm argument the approval anticipated, and it
    dominates that argument for both sources tested. The effect is amplified
    by a pre-existing, already-documented limitation on A (this module's A
    entry, 'NOT COVERED'): the model under-predicts Young's absolute rises by
    2.7x (FB) to 6.2x (PB) at any A consistent with the rest of the
    validation suite, so the low-BSA arm's modelled gross gain here is
    smaller relative to the baseline decay drag than the true physiology,
    sharpening the artefact. Full numeric diagnosis, including the
    sensitivity check that isolates this mechanism, is in
    scripts/bsa_exp_recalibration.py (invert_bsa_exp's docstring and Step 2's
    printed explanation).
    THIS IS REPORTED AS THE DERIVATION'S ACTUAL OUTPUT, NOT ADJUSTED TOWARD
    EITHER THE APPARENT EXPONENTS OR THE PRE-REGISTERED >= 0.50 EXPECTATION,
    per the same "write whatever the derivation gives" instruction that
    would have applied had it come out high instead of low. It should be
    read as evidence that a mechanism OTHER than the BSA exponent itself
    (this model's decay structure interacting with A's known
    under-prediction at extreme BSA) already explains more of the apparent
    plateau than g(D) feedback alone -- a structural finding about this
    model's OWN equations, not a new claim about cutaneous physiology.
    SENSITIVITY (scripts/bsa_exp_recalibration.py Step 4): 0.3105 (saturating
    feedback) to 0.3281 (biphasic, adopted) across every feedback_mode this
    model supports -- narrow relative to the two-source spread, so feedback
    FORM is not what is driving the surprising direction; the decay
    mechanism above is.
    NOT COVERED / UNSOURCED INPUTS (both flagged in the derivation script,
    not silently assumed): (1) Young 2021's Partial-Body arm protocol
    (session count, spacing, baseline D0) is assumed identical to its
    Full-Body arm (10 SED cumulative, 5 x 2 SED Arimed B, D0 = 34.6 nmol/L,
    3.5 d spacing carried over from Young 2020's protocol convention) --
    the 2021 paper's own PB schedule is not transcribed elsewhere in this
    repo. (2) The Vahavihu triple's session count, spacing and baseline D0
    are NOT printed in Jager 2018 Table III (only cumulative SED, %BSA and
    response are); assumed 6 sessions ~3.5 d apart, D0 = 40.0 nmol/L, FST II.
    Step 2c of the derivation script sweeps sessions in {3,6,9,12} and D0 in
    {25,40,55,70} nmol/L: the deconvolved Vahavihu exponent ranges from 0.00
    (at-bound, D0=70, 12 sessions) to 0.39 (D0=25, 9-12 sessions) -- WIDER
    than the two-source spread this entry otherwise reports, so 0.3281
    should be read as sensitive to these two unsourced assumptions, not as
    tightly pinned as the headline four-significant-figure combination
    implies. (3) NO DARK SKIN. bsa_factor multiplies A directly, so this
    COMPOUNDS the light-skin extrapolation already flagged on A, exactly as
    the pre-derivation entry above already noted for the withdrawn 0.50.
- exposed_bsa (Bogh 2011 / P5, Bogh 2010 / P36, Kimlin 2014 / P32):
    [C1] Fraction of body surface area exposed to UV (0.0-1.0). Kimlin 2014 (P32)
    shows clothing cover is the LARGEST single contributor (~27%) to explained
    25(OH)D variance in naturalistic settings. Bogh 2011 (P5) derived:
    delta = 7.1 - 0.223*baseline + 11.0*ln(UVB_dose) + 5.5*ln(%BSA), R2=0.47.
    Matsuoka 1990 (P8) shows a plateau above ~33% BSA. All validation scenarios
    must specify exposed_bsa explicitly; use BSA_REF=0.35 as the reference.
    Clothing cover = 1 - exposed_bsa.

    # -------------------------------------------------------------------------
    # PARAMETER EVIDENCE STATUS (updated from Consensus literature search, July 2026)
    # -------------------------------------------------------------------------
    # Gap 1 — M_baseline FST V/VI:
    #   Current values: {V: 0.78, VI: 1.00} (ITA_Unexposed means; pooled
    #   ArimedB irradiated + ARIMED Controls, n=48 with ITA -- NOT the "N=44"
    #   sheet row count once stated here; see the M_baseline_by_FST entry
    #   above. n = 4 at FST V and 4 at FST VI.)
    #   Literature support: WILKES 2015 gives MI>750->FST VI on Mexameter (partial).
    #   ALALUF 2001 gives quantitative melanin in FST V/VI but not as a baseline MI table.
    #   No paper provides separate untanned FST V and FST VI baseline MI means.
    #   Status: COARSE PRIOR — awaiting de Boer 2025 individual-level data.
    #   [SUPERSEDED IN PART 2026-09-06] The sentence above is no longer accurate
    #   as written. Narea-Jimenez et al. 2026 (n=1005, Venezuela) DOES provide
    #   separate untanned FST V (n=84) and FST VI (n=32) constitutive means with
    #   dispersion, from a sun-protected site with clinician-assigned FST — as
    #   ITA and melanosome fraction rather than Mexameter MI, but the shipped
    #   table is ITA-derived, so ITA is the relevant currency. It agrees with
    #   the shipped V cell within 0.026 and the VI cell within 0.043 pre-clip.
    #   The live constraint is no longer the absence of V/VI data; it is that
    #   the ITA->M mapping CLIPS at ITA = -30, which FST VI group means from two
    #   unrelated cohorts already exceed. Full note, with the Linde 2020 field
    #   cohort and all caveats: the M_baseline_by_FST entry in this module docstring.
    #
    # Gap 2 — K_ATTEN:  OPEN. Retired 2026-08-18, REINSTATED and calibrated
    #   2026-08-21 (user-approved) — see the K_ATTEN entry above for the live
    #   value and its Matsuoka 1990 derivation. RETIRED_PARAMETERS is now empty:
    #   K_ATTEN is an ACTIVE parameter again, not a structural constant pinned
    #   at 0.00, and the 2026-08-18 "CLOSED BY RETIREMENT" status recorded here
    #   previously is superseded.
    #   Matsuoka 1990 implies 0.288972-0.838389 depending on feedback form; only source,
    #   no replication. NEVILLE 2020 confirms the field still lacks tanning-effect
    #   data. MARKIEWICZ 2020: tanning may be redistribution, not pure synthesis.
    #   Status: SINGLE-SOURCE CALIBRATION at the lower bound of that range. The
    #   evidence gap is not resolved; the parameter is simply no longer fenced.
    #   NOTE: any result generated before 2026-08-21 (including
    #   spacing_study/results/FINDINGS.md) describes K_ATTEN = 0.00.
    #
    # Gap 3 — A (synthesis rate):
    #   Current value: 3.114 (2026-08-18 recalibration; see the A entry above)
    #   Anchored to the midpoint of YOUNG 2021 Fig. 4B's three printed full-body
    #   SED slopes (1.99 / 3.18 / 4.97 nmol/L per SED at 85% BSA).
    #   WEBB 2021 confirms logarithmic form; BORECKA 2024 and CHALCRAFT 2020 provide
    #   partial low-dose response data in light skin.
    #   Status: CENTRE OF A 2.9-5.1 RANGE. The spread is source-dependent and is
    #   Young's own finding that SED is a poor spectral predictor. LIGHT-SKIN
    #   EXTRAPOLATION when applied to FST III-VI (Young 2021 is FST I-II only).
    #   [NOTE 2026-08-20] McKenzie et al. 2012 (Photochem Photobiol Sci 11:1174;
    #   full text in Research_Papers/) fits delta-D = m*ln(1+dose), the same
    #   functional FORM as A*ln(1+UV*E_source) here, on n=119 including FST IV-VI
    #   (n=41+33 -- the largest dark-skin dose-response cohort this repo has
    #   seen). CORROBORATES the log form, does NOT independently set A: their m is
    #   in nmol/L per solaria SDD, not per-SED, and is not directly convertible
    #   without their booth's own spectral RDE. Also reports second-6-week
    #   response falling to LESS THAN HALF the first 6 weeks at equal or larger
    #   dose -- a real empirical target for k_decay_D's feedback interaction,
    #   currently anchored only to Karppinen. Not yet used quantitatively.
    #
    # Gap 4 — K_const:
    #   Current value: 0.98 (2026-08-17 recalibration; see the K_const entry above)
    #   Fitted to four within-study routes: Felton 2016, Young 2020, Kift 2013,
    #   Darling 2013. SD across routes 0.56.
    #   Literature support: FELTON 2016 (within-study, matched dose, FST V, both
    #   baselines printed). YOUNG 2020 (within-study, matched dose, FST VI,
    #   individual-level data in repo — the LOW outlier at 0.28).
    #   LIBON 2013 (directional: FST VI no significant D rise vs FST II-III significant).
    #   ARMAS 2007 (80% variation from dose + skin tone).
    #   Status: FOUR ROUTES, TWO METHODS, WIDE. Felton and Young are mutually
    #   incompatible within this model structure (S3 of the derivation script);
    #   0.98 is their and the field's centre, not a reconciliation.
    # -------------------------------------------------------------------------

Future covariates (not yet in core model; add when matching dataset available):
    [C9] Total cholesterol / 7-DHC precursor status: Bogh 2010 (P36) found total
    cholesterol explained 26.5% of delta variance in a controlled UVB subgroup,
    second only to baseline 25(OH)D. Do NOT add to core model without a local
    dataset linking serum cholesterol, UV exposure, and 25(OH)D response.

Dose framing modes (label all scenario outputs):
    [C4] Fixed absolute SED: each session delivers the same SED regardless of MED.
    [C4] Fixed cumulative SED: total UV budget is split across sessions.
    [C4] Fixed per-session SED: 1 SED/session; cumulative varies with frequency.
    [C4] Fixed fractional MED: each session is N% of the individual MED.
    P24 (Shih/Farrar/Rhodes 2018) shows fractional-MED dosing removes the apparent
    skin-type effect for acute 25(OH)D gain. P22/P3 show fixed absolute SED can
    leave South Asian/FST V participants below sufficiency. Do NOT treat these
    framings as interchangeable in output reports.

Audit change tags (July 2026 Scientific Audit):
  [C1]  exposed_bsa / clothing_cover support added to model and validation
  [C2]  SOURCE_CALIBRATION_TABLE replaces ad hoc E_source scalar
  [C3]  E_source must be per-study calibration input
  [C4]  Dose framing labels added to simulate() output
  [C5]  K_ATTEN small-penalty evidence caveat
  [C7]  SC_max reparameterized as 0.20-0.40 prior (default 0.30)
  [C7-SC] SC defaults documented as priors, not calibrated constants
  [C8]  SC state active, calibration pending
  [C9]  Cholesterol/7-DHC added to future covariates
  [C10-PRIOR] K_ATTEN and M_baseline_by_FST explicitly tagged as coarse priors

Author: Ahaan
Last updated: 2026-07 (Scientific Audit Rev 2)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vitd_physiology import (body_weight_dilution_factor,
                             photodegradation_factor,
                             hydroxylation_efficiency,
                             cancer_risk)


# ---------------------------------------------------------------------------
# [C2] Source/Spectrum Calibration Table
# ---------------------------------------------------------------------------
# Evidence-derived D3-effective dose weighting factors per UV source.
# Sources: Webb 2021 (P21), Grigalavicius 2015 (P35).
# TREAT AS PRIORS until recalculated from measured spectra. Young 2021 PNAS
# (P20) shows the CIE previtamin-D3 action spectrum needs a ~5 nm blue shift,
# which will change all values. Osmancevic 2009 (P18) shows broadband UVB can
# outperform NB-UVB per SED in psoriasis, confirming source-specific variation
# cannot be collapsed to a single constant.
#
# Keys: source name (str). Values: (E_source_mean, E_source_range_low, E_source_range_high).
# Use E_source_mean for point estimates; report range in sensitivity analyses.
SOURCE_CALIBRATION_TABLE: Dict[str, Tuple[float, float, float]] = {
    # (mean, range_low, range_high)
    # --- SUN ---------------------------------------------------------------
    # Webb 2021 Table 1 reports VDE:EE for sun. They are not competing estimates
    # of one number: sun VDE:EE genuinely varies with solar zenith angle and,
    # secondarily, ozone -- Webb says so directly ("The VDE:EE ratio therefore
    # changes with solar elevation and ranges from about 2 at low latitudes/
    # mid-summer/noon to around 1 or below at high latitudes/mid-winter").
    #
    # CORRECTED 2026-08-17. This comment previously said ~33% of erythemal
    # weighting comes from 315-400 nm "where the CIE-174 vitamin D spectrum is
    # exactly zero". The mechanism is right; "exactly zero" is WRONG. This
    # repo's own data/spectra/vitd_action_spectrum_cie.csv has CIE-174 support
    # running to 332 nm, with 18 nonzero entries at/above 315 nm (0.045 at 315,
    # 0.033 at 316, ... ). The overlap contribution is small and the argument
    # survives, but the stated cut-off is not a fact. Do not requote it.
    #
    # INCOMPLETE 2026-08-17. Table 1 carries FOUR sun entries; this table has
    # three. The missing one is "1.9-1.99 (summer)" from Webb's ref 32 = Joh et
    # al. 2020, Clin Nutr 39:727-736 (Korea) -- the only INDEPENDENT summer
    # replicate of Manchester's 1.89. Also note Sun is the only row in Table 1
    # with a BLANK mean column: Webb deliberately declined to average her own
    # sun entries. Do not "fix" that by averaging them here.
    #
    # These values are pure spectral-integral unit conversions (CIE 174:2006
    # numerator, CIE erythema reference denominator). NO vitamin D was measured
    # to produce any of them, so none of them validates anything about human
    # response. Webb uses them honestly as conversions; so must we.
    "solar_summer":    (1.89, 1.79, 2.00),  # P21: Manchester summer. DEFAULT for sun.
    "solar_summer_nz": (2.01, 2.01, 2.01),  # P21: NZ summer
    "solar_winter_nz": (1.22, 1.22, 1.22),  # P21: NZ winter -- lowest sun value published
    # [NOTE 2026-08-20] McKenzie et al. 2012 (Photochem Photobiol Sci 11:1174,
    # "Serum 25-hydroxyvitamin-D responses to multiple UV exposures from
    # solaria"; full text in Research_Papers/) Table 2 reports 45S solar
    # RDE(=VDE:EE) of 1.22 winter / 2.01 summer -- numerically identical to the
    # two NZ entries above. Almost certainly the SAME underlying measurement
    # chain via Webb 2021, not an independent replicate -- do not double-count
    # as corroboration for solar_summer_nz/solar_winter_nz specifically.
    # DEPRECATED 2026-08-16, retained so old scripts fail loudly rather than
    # silently. 1.00 was never a measurement of anything: it was the backward
    # compatibility default when E_source was introduced (audit tag [C2]) and was
    # retro-labelled "solar broadband reference". No solar spectrum has
    # VDE:EE = 1.00 -- Webb's lowest sun entry is 1.22, in NEW ZEALAND WINTER.
    # Using it for sun under-weights vitamin D effectiveness by ~1.9x.
    "solar_broadband": (1.00, 0.90, 1.10),
    # --- LAMPS -------------------------------------------------------------
    # [NOTE 2026-08-20] McKenzie et al. 2012 (see solar_summer_nz note above)
    # Table 2 gives a THIRD, genuinely independent TL-01 NB-UVB (booth B)
    # RDE estimate of 2.27 -- inside this range, near the floor. Corroborating,
    # not fitted to; the range/midpoint below are unchanged.
    "tl01_nb_uvb":    (2.44, 2.24, 2.63),  # P21: 2.24; P35: 2.63; use midpoint
    "tl12_bb_uvb":    (1.24, 1.23, 1.25),  # P21/P35 agree
    "uv6_broadband":  (1.54, 1.54, 1.55),  # P21/P35 agree
    "arimed_b":       (1.83, 1.83, 1.83),  # P21 only
    "sunbed_uvb":     (1.18, 1.18, 1.18),  # P35 (Grigalavicius 2015 Table 2); P21 Wolff Life Sun = 1.85
}

# Default sun-facing E_source, used unless overridden.
# [CALIBRATION 2026-08-16, user-approved] Changed from implicit 1.00 to
# Webb's Manchester summer measurement. See calibration_approval.yaml for
# the approval record and for the cost: NHANES
# Part B drops 6/6 -> 4/6 because those checks feed an ASSUMED daily SED
# nobody measured, making solar E_source unidentifiable from them. That
# assumed dose was deliberately NOT retuned to keep the check green.
E_SOURCE_SOLAR_DEFAULT: str = "solar_summer"

# ORAL / DIETARY VITAMIN D: REMOVED 2026-08-16.
#
# `DIETARY_UG_TO_NMOL_PER_L` and `ModelParams.dietary_D_daily_ug` fed a
# zero-order oral source term in `simulate()`. Both gone.
#
# Why. This model schedules SUN EXPOSURE. The oral pathway was scope creep
# (entered via audit directive 6f for "naturalistic cohort simulations"),
# never used by any scheduling analysis, and measurement on 2026-08-15 showed
# its coefficient missed its own documented Cashman target by 3-9x at every
# dose while its stated worked example was not reproducible. An uncalibrated
# pathway nobody exercises is a liability, not a feature.
#
# What was NOT lost. The oral route was the experimental probe behind the only
# evidence that gives `D_max` an interior optimum (Gallagher 2012). That
# evidence SURVIVES: the
# saturating step it constrains is hepatic -- downstream of both cutaneous and
# oral routes -- so it bounds systemic feedback ceiling `D_max` regardless of
# route. The argument is preserved, self-contained and runnable, in
# `scripts/dmax_identifiability_gallagher2012.py`, which no longer depends on
# any oral capability in the model.
#
# Do not reintroduce an oral term without recalibrating the coefficient first
# -- see that script for what it takes.

# Reference exposed BSA fraction for synthesis normalization.
# Farrar 2013 (35% BSA); note Bogh 2011 used 6-24% BSA.
BSA_REF: float = 0.35
# Sub-linear BSA scaling exponent.
# [CALIBRATION 2026-09-06, user-approved] 0.50 -> 0.3281. Feedback-deconvolved
# from two within-study 25(OH)D BSA contrasts (Young 2021; Vahavihu/Jager
# 2018) -- full derivation in this module's docstring (BSA_EXP entry) and
# scripts/bsa_exp_recalibration.py. UNEXPECTED DIRECTION: the deconvolution
# moved the exponent DOWN from the ~0.38 apparent estimates, not up, because
# this model's decay term (acting on the whole D state, not just the
# increment) dominates over g(D) feedback in the BSA ranges tested. Read the
# docstring entry before treating this as settled.
BSA_EXP: float = 0.3281

# [CALIBRATION 2026-09-06, user-approved] NEW CONSTANT. Baseline (unstimulated)
# stratum-corneum thickness, micrometres. Introduced SOLELY so that K_SC below
# can be derived from an optical measurement (Bruls 1984) instead of being an
# arbitrary number -- see the K_SC entry in this module's docstring for the
# full derivation and the SC_BASELINE_THICKNESS_UM entry in
# calibration_approval.yaml for the measurement-method comparison.
# CONSUMED ONLY BY THE K_SC DERIVATION BELOW (a documentation-time
# calculation). It is NOT read by ModelParams, simulate(), or any state
# equation: SC(t) remains a purely fractional state, exactly as before this
# change. Wiring an absolute baseline into the SC ODEs is a separate,
# unapproved change (calibration_approval.yaml SC_BASELINE_THICKNESS_UM
# entry, not_covered clause).
#
# CHOICE: 20.0 um, Bruls et al. 1984's own adopted in vivo estimate (their
# cryostatic measurement 19.8 +/- 4.3 um, cross-checked against two
# independent in vivo readings for the same body region: Klaschka 1977 and
# Hulsbergen Henning et al. 1977, both ~18 um for lower arm/back). Chosen over:
#   - de Boer 2025 Supp Table S1, 5.8 um (all FP; 5.6 FP I-III, 6.6 FP IV-VI).
#     NOT a directly measured thickness at all: it is the fitted plateau of a
#     Boltzmann sigmoid on CUMULATIVE PROTEIN MASS from sequential adhesive
#     tape-strips (Squame Scan 850 densitometry), in vivo, lower back,
#     unfixed. de Boer's own OCT channel could not resolve SC thickness
#     directly ("covers 1-2 pixels ... not reliably segmented", p.2310) --
#     the 5.8 um figure is an indirect biochemical proxy, not a length
#     measurement, and its absolute calibration against a ruler is unclear.
#   - Miyamura 2011 (PMID 20979596), 16.6 um. UNVERIFIED IN THIS PASS: the
#     paper is not held in Research_Papers/ or Papers_and_Datasets/ (checked
#     by filename and full-repo grep for "miyamura" and the PMID, 2026-09-06)
#     despite being cited elsewhere in this file as SC_max's out-of-sample
#     corroboration -- that earlier citation predates this pass and its
#     source PDF could not be re-located. FLAGGED, not silently dropped: if
#     the PDF is obtained, its measurement method must be checked before
#     using 16.6 um for anything beyond the existing SC_max corroboration.
# Bruls' own decisive argument for preferring their corrected estimate over
# raw histological or ex vivo readings: histological SECTIONS read
# systematically LOW from dehydration shrinkage during fixation/processing --
# they note Holbrook & Odland's frequently-cited 10 um back SC sits below
# even Bruls' own DRIED ex vivo sample average (13.5 um), let alone the
# hydration-corrected in vivo estimate. Bruls measure ex vivo samples dried
# AND on saline of varying concentration, then back-calculate an "approximate
# in vivo" value from the estimated in vivo SC water content -- a correction
# neither de Boer's tape-strip proxy nor (so far as verifiable here)
# Miyamura's method appears to apply. This is a measurement-method argument,
# not a preference for the largest number: Bruls' own ex vivo dried
# measurements sit BELOW 20 um too, and their in vivo correction is what
# raises the adopted value.
# NOT RESOLVED BY THIS CHOICE: the K_SC entry's [PREMISE WARNING] on whether
# the model's own SC state change is real hyperplasia or edema is
# INDEPENDENT of which d0 is chosen here, and de Boer's SC values (whichever
# d0 is used to convert them) are the ones whose physiological meaning is in
# question -- see that entry.
SC_BASELINE_THICKNESS_UM: float = 20.0


# ---------------------------------------------------------------------------
# Retired parameters
# ---------------------------------------------------------------------------
# A RETIRED parameter keeps its mechanism and value but loses calibratable
# status — it can no longer be moved to improve agreement with data. The
# machinery below makes "retired" enforceable in code, not just asserted in a
# comment.
#
# Rules:
#   - value lives in a module-level constant, named here;
#   - ModelParams refuses to hold any other value for it;
#   - ablation and sensitivity work goes through
#     `allow_retired_parameter_sweep()`, deliberately verbose at the call
#     site so a sweep can never be mistaken for a calibration;
#   - reinstating a parameter means deleting its RETIRED_PARAMETERS entry,
#     a code change reviewable in a diff, gated by
#     calibration_approval.yaml like any other calibration.

# [CALIBRATION 2026-08-25, user-approved] 0.34 -> 0.288972. Facultative-melanin
# attenuation coefficient, anchored to Matsuoka 1990 (PMID 2165512)
# deconvolution under biphasic feedback. Range across feedback forms
# [0.288972 biphasic, 0.838389 linear].
#
# WHY IT MOVED. K_ATTEN is not independently measured. It is derived from the
# Matsuoka residual attenuation and the size of a fully developed tan:
#     vitd_calibration.derive_K_ATTEN(feedback_mode, excess_M=M_TAN_HEADROOM)
# so it is a FUNCTION of M_TAN_HEADROOM, which was recalibrated 0.20 -> 0.238
# on the same day (Ravnbak 2008; see M_TAN_HEADROOM below). Holding K_ATTEN at
# the old 0.34 while the headroom grew would have raised the facultative
# attenuation actually applied from 6.5740% to 7.7733%, overshooting the
# Matsuoka residual the parameter exists to encode. The measured quantity is
# that residual; K_ATTEN is only its parameterisation, so it tracks.
# derive_K_ATTEN now reads M_TAN_HEADROOM live, so the two cannot drift again.
K_ATTEN_DEFAULT: float = 0.288972
K_ATTEN_STRUCTURAL: float = K_ATTEN_DEFAULT  # backward-compatibility alias

RETIRED_PARAMETERS: Dict[str, str] = {}

# Module-level constant mapping for retired parameters (none currently retired).
_RETIRED_VALUES: Dict[str, float] = {}

# Depth counter, so nested sweeps behave.
_retired_sweep_depth: int = 0


class RetiredParameterError(RuntimeError):
    """Raised when a retired parameter is assigned a value it may not take."""


class allow_retired_parameter_sweep:
    """Context manager permitting temporary assignment to retired parameters.

    For ABLATION AND SENSITIVITY ANALYSIS ONLY. Entering this block does not
    make a retired parameter calibratable: values discovered inside it have
    no standing until removed from RETIRED_PARAMETERS and given a
    `parameter_approvals` entry in calibration_approval.yaml.

        with allow_retired_parameter_sweep():
            p.K_ATTEN = 0.4          # measuring, not calibrating
    """

    def __enter__(self):
        global _retired_sweep_depth
        _retired_sweep_depth += 1
        return self

    def __exit__(self, *exc):
        global _retired_sweep_depth
        _retired_sweep_depth -= 1
        return False


def get_E_source(source_name: str) -> float:
    """Return E_source mean for a named source from SOURCE_CALIBRATION_TABLE.

    [C2] Use this instead of hardcoding an E_source float wherever possible.
    Falls back to 1.0 (broadband reference) with a warning if source unknown.
    """
    if source_name == "solar_broadband":
        import warnings
        warnings.warn(
            "E_source 'solar_broadband' (1.00) is DEPRECATED for sun as of "
            "2026-08-16. It was the backward-compatibility default when E_source "
            "was introduced, not a measurement; no solar spectrum has "
            "VDE:EE = 1.00 (Webb 2021 Table 1's lowest sun entry is 1.22, NZ "
            f"winter). Use '{E_SOURCE_SOLAR_DEFAULT}' for sun, or a named "
            "seasonal entry. Returning 1.00 unchanged.",
            UserWarning, stacklevel=2,
        )
        return SOURCE_CALIBRATION_TABLE[source_name][0]
    if source_name in SOURCE_CALIBRATION_TABLE:
        return SOURCE_CALIBRATION_TABLE[source_name][0]
    import warnings
    warnings.warn(
        f"Unknown source '{source_name}' — falling back to E_source=1.0 (broadband reference). "
        f"Add to SOURCE_CALIBRATION_TABLE or pass E_source_override directly.",
        UserWarning, stacklevel=2,
    )
    return 1.0


@dataclass
class ModelParams:
    # [CALIBRATION 2026-08-18, user-approved item A4] 2.353 -> 3.114. Anchored
    # to the midpoint of Young 2021 Fig. 4B's three printed full-body SED
    # slopes. Full derivation in this module's docstring; reproduced and
    # asserted by vitd_validation.young2021_A_anchor(). LIGHT-SKIN
    # EXTRAPOLATION when applied to FST III-VI -- Young 2021 is FST I-II only.
    # [RE-ANCHOR 2026-09-06, user-approved] 3.114 -> 4.125. NOT a new
    # calibration and NOT a new anchor: the SAME Young 2021 target slope of
    # 3.48 nmol/L per SED, recomputed because two of its own denominator terms
    # moved in the same commit. bsa_factor(0.85) fell 1.558387 -> 1.337949
    # (BSA_EXP 0.50 -> 0.3281) and exp(-K_const*M_II) fell 0.933700 ->
    # 0.821006 (K_const 0.98 -> 1.15 with M_baseline(II) 0.07 -> 0.1715);
    # ln(1+E) and g_biphasic(34.6) are unchanged. Kernel 1.117569 -> 0.843668,
    # so A = 3.48 / 0.843668 = 4.1248 -> 4.125.
    # WHY THIS WAS NECESSARY: A had gone STALE. At A = 3.114 the model's own
    # anchor produced 2.6272 nmol/L per SED against its 3.48 target -- the
    # anchor the parameter is defined by no longer held. The K_const approval
    # entry anticipated exactly this ("A is NOT yet recalibrated ... the
    # feasibility consequence is PROVISIONAL until the combined K_const + A
    # map is re-run"). Leaving A alone would have silently rebased the model's
    # absolute level onto no anchor at all.
    # MEASURED CONSEQUENCE, AND IT IS NOT A CLEAN WIN. All four changed
    # parameters push synthesis DOWN and A carries the absolute level, so the
    # four-parameter move knocked Bogh 2012 and Farrar 2013 from PASS to FAIL
    # (validation 10/10 -> 8/10). Re-anchoring restores BOTH -- Bogh delta
    # +38.8 -> +50.1 (band 45.8-65.8), Farrar +8.7 -> +13.6 (band 10-35) --
    # but OVERSHOOTS a third check that had been passing:
    # felton2016_fst5_matched now predicts +21.9 nmol/L against an observed
    # +8.3, just outside the reference-scenario band of 2.5x observed
    # (<= 20.75). NET: 8/10 -> 9/10, with a different check failing than
    # before. This is recorded, not hidden.
    # WHY A WAS NOT TUNED TO MAKE ALL THREE PASS. A window does exist --
    # A in roughly [3.75, 3.87] passes Bogh, Farrar and Felton FST V
    # simultaneously (measured: A = 3.80 -> Bogh 47.5, Felton5 20.4, all
    # scenarios PASS; A = 3.85 -> 47.9 / 20.6, PASS; A = 3.90 -> 48.3 / 20.8,
    # FAIL). Picking a value from that window was REJECTED: it would fit A to
    # the validation set, which is exactly the separation this repo maintains
    # (Felton 2016 is in the CALIBRATION set behind K_const, and Bogh 2012 is
    # self-labelled DEBUGGING_TARGET / TIER: CIRCULAR). An A tuned to 3.85
    # would no longer reproduce its own anchor -- it would yield 3.25 nmol/L
    # per SED against Young's 3.48 target. 4.125 is what the anchor says;
    # the failing check is reported as a cost, not engineered away.
    # The LIGHT-SKIN EXTRAPOLATION flag above is UNCHANGED and if anything
    # more load-bearing: A is now larger, so more of the model's absolute
    # level rests on an FST I-II anchor.
    A: float = 4.125
    alpha: float = 0.0025
    alpha_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0025, 2: 0.0025, 3: 0.0025,
        4: 0.002,  5: 0.002,  6: 0.002,
    })
    alpha_max: float = 0.02
    # Hill saturation constant for melanin induction kinetics (Ravnbak 2008 Table IV post-adaptation multi-exposure MMD)
    UV_halfmax: float = 2.0
    UV_halfmax_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 2.0, 2: 2.0, 3: 2.0,
        4: 5.0, 5: 8.0, 6: 10.0,
    })
    beta: float = np.log(2) / 56.0
    # [CALIBRATION 2026-08-17, user-approved] 0.33 -> 0.98.
    # [CALIBRATION 2026-09-06, user-approved] 0.98 -> 1.15, refitted in the
    # SAME change as M_baseline_by_FST's move to the Narea 2026 table (this
    # fit runs on M differences, so the two are coupled -- see both entries).
    # Full derivation and provenance in this module's docstring and
    # scripts/kconst_recalibration.py. This is CONSTITUTIVE PIGMENT ONLY.
    # Behaviour (shade, clothing, time outdoors) is a separate 2.1-2.4x factor
    # and lives in vitd_behaviour.py, neutral by default -- do NOT absorb it
    # here.
    K_const: float = 1.15
    # Facultative (tanning) headroom above constitutive pigmentation, on the
    # normalized 0-1 melanin scale. The absolute ceiling for a given FST is
    # M_baseline(FST) + M_TAN_HEADROOM -- see M_ceiling(). NOT capped at 1.0
    # (FST VI: 1.00 + 0.238 = 1.238); 1.0 caps only the constitutive baseline.
    # (An earlier revision said "capped at 1.0"; corrected 2026-08-25.)
    #
    # A single global M_max was wrong: M_max=0.95 sat BELOW
    # M_baseline_by_FST[6]=1.00, so the saturating pulse update
    #     M <- M_max - (M_max - M) * exp(-alpha * sed)
    # DECREASED melanin for FST VI (UV made the skin lighter). FST V/VI is the
    # target population for this project, so that inversion mattered.
    # [CALIBRATION 2026-08-25, user-approved] 0.20 -> 0.238. Anchored to
    # Ravnbak 2008 (PMID 18480919) Table II via scripts/fit_melanin_kinetics.py.
    # Converted onto this model's melanin index, Ravnbak's six measured tans
    # span dM = 0.192059 to 0.237724. The previous headroom of 0.20 sat BELOW
    # four of those six, so the model could not represent a Ravnbak tan at any
    # value of alpha -- the ceiling, not the induction rate, was binding.
    # 0.238 is the largest observed tan (0.237724, rounded up at the third
    # decimal), so the ceiling now accommodates every measured tan while
    # remaining inside the free-fit range [0.230880, 0.268013] returned by the
    # three candidate scale maps (power / quadratic / linear).
    # NOTE this parameter, not alpha, is what Ravnbak actually constrains:
    # profiling alpha over 0.0025-1.0 moves the weighted residual <25%, because
    # the tan is already near plateau at the earlier of Ravnbak's only two
    # exposure counts. See scripts/fit_melanin_kinetics.py sec.2 and
    # docs/MELANIN_MODEL.md sec.2.1.
    M_TAN_HEADROOM: float = 0.238
    # [UNUSED BY THE CORE PIPELINE -- flagged 2026-08-23, NOT removed.]
    # Nothing reads ModelParams.M_max: simulate() takes its ceiling from
    # M_ceiling(fst) (or from the separate `M_max_override` argument, which IS
    # used and is a different thing). A repo-wide grep for `.M_max` outside
    # exploratory/ finds only this definition, the historical comment above,
    # and the `M_max_override` plumbing.
    # DISCREPANCY LEFT FOR THE AUTHOR TO RESOLVE: the symbol M_max still
    # appears in the published governing equation for dM/dt (this module's
    # docstring, README.md, paper/manuscript.tex), where it reads as the
    # melanin ceiling actually in force. It is not. Either the equation should
    # be restated in terms of M_ceiling(F) = M_baseline(F) + M_TAN_HEADROOM, or
    # this field should be removed -- both are behaviour/text changes beyond a
    # documentation pass. The field is retained for backward compatibility with
    # callers that construct ModelParams(M_max=...).
    M_max: float = 0.95
    # [CALIBRATION 2026-08-21, user-approved] 0.00 -> 0.34; [CALIBRATION
    # 2026-08-25, user-approved] 0.34 -> 0.288972 tracking M_TAN_HEADROOM.
    # Active parameter
    # anchored to Matsuoka 1990 (PMID 2165512, n=7 within-subject tanned vs untanned)
    # with biphasic feedback g(D) deconvolved (residual melanin attenuation 6.6%,
    # K_ATTEN = 0.288972; range [0.288972, 0.838389]).
    K_ATTEN: float = K_ATTEN_DEFAULT
    D0_default: float = 25.0
    # LEGACY. Ceiling of the withdrawn 'linear'/'power' feedback modes only; the
    # default feedback_mode is 'biphasic' and does not read this. See the
    # feedback_mode entry in this module's docstring.
    # WARNING: ORIGIN UNCONFIRMED — not found in Bogh 2010 (PMID 19812604 max was 116 nmol/L)
    D_max: float = 150.0
    # [C2] E_source default corresponds to broadband solar reference (1.0).
    # Use SOURCE_CALIBRATION_TABLE and get_E_source() instead of hardcoding.
    E_source: float = 1.0
    # [C1] Reference and default exposed BSA fraction (Bogh 2011 P5 / Farrar 2013).
    # Kimlin 2014 (P32) shows clothing cover is the largest naturalistic predictor.
    # Default is BSA_REF so all existing scenarios are unchanged (BSA factor = 1.0).
    exposed_bsa: float = BSA_REF
    # [C3-feedback] Baseline feedback parameters
    # feedback_mode can be 'linear', 'power', 'saturating', or 'biphasic'.
    # [CALIBRATION 2026-08-18, user-approved item A2] Default moved
    # 'linear' -> 'biphasic'. 25-hydroxylation efficiency falls above a
    # ~56 nmol/L inflection (Scholze 2026, Commun Med 6:105) rather than
    # synthesis stopping at a hard D_max ceiling of unconfirmed origin.
    # 'linear' and 'power' are retained, unchanged, so pre-2026-08-18 results
    # stay reproducible by setting them explicitly. Measured effect on the
    # interval headline is in this module's docstring, feedback_mode entry.
    feedback_mode: str = "biphasic"
    feedback_power: float = 1.0
    D_half_feedback: float = 50.0

    # ---- Mechanistic extensions (July 2026 literature pass) -----------------
    # All default OFF/neutral so every one is separately ablatable and existing
    # scenarios are bit-identical unless explicitly enabled.
    #
    # Body weight for volumetric dilution (Drincic 2012: hyperbolic in total
    # body WEIGHT, not BMI, and not sequestration). None disables the term.
    body_weight_kg: Optional[float] = None
    # Previtamin-D3 photodegradation plateau (Holick 1981, van Dijk 2016).
    # This is the mechanism that makes dose-splitting genuinely better, as
    # opposed to the feedback artifact that
    # vitd_optimization.gross_synth_per_SED_diagnostic recomputes on every
    # pipeline run (see the vitd_objectives.py module docstring).
    use_photodegradation: bool = False
    # Cutaneous D3 -> serum 25(OH)D transit delay. Without this, UV synthesis
    # appears in serum 25(OH)D instantaneously, which is wrong on exactly the
    # timescale this project optimises over (1-3 day intervals).
    # Two serial steps, lumped into one first-order transfer:
    #   previtamin-D3 -> D3   thermal isomerisation, ~80% complete in 8 h
    #   D3 -> 25(OH)D         hepatic CYP2R1; serum D3 half-life ~4 h, but
    #                         appearance of 25(OH)D peaks over ~1-2 days
    # t_half = 1.0 day is the lumped transfer half-life. [PRIOR] No study
    # reports this as a fitted rate constant; it is inferred from the 8 h
    # isomerisation plus the ~4 h D3 half-life and the observation that serum
    # 25(OH)D is typically sampled 24 h post-exposure.
    use_transit_delay: bool = False
    transit_half_life_days: float = 1.0
    # Stratum corneum thickening state parameters
    # [C7-SC] Recalibrated from de Boer 2025 Supplementary Table S1:
    #   FP I-III: SC 5.6->6.9 um (+23% -> SC_max=0.23)
    #   FP IV-VI: SC 6.6->9.0 um (+36% -> SC_max=0.36)
    # alpha_SC=0.12 calibrated so SC reaches ~90% of SC_max within
    # 27 sessions (0.8 SED 3x/week, de Boer protocol).
    # k_decay_SC: ~28 d half-life [PRIOR, Weinstein & Van Scott 1965 via
    #   Rothberg 1961 -- transit time, not measured half-life; see :826].
    # k_damage (UV-driven desquamation) fits the week-5 SC dip only
    # weakly from 3 group-mean timepoints; kept at 0.0 until individual-
    # level data is available for joint fitting.
    k_decay_SC: float = np.log(2) / 28.0
    alpha_SC: float = 0.12
    SC_max: float = 0.30
    # [CALIBRATION 2026-08-25, user-approved] FST I-III 0.23 -> 0.24.
    # de Boer 2025 Supp Table S1 gives +23% for FP I-III. Miyamura 2011
    # (PMID 20979596) Table 1 independently gives stratum corneum 16.6 -> 20.7
    # um in FST II-III after 10 suberythemal UVB exposures, i.e. +24.6988% --
    # a different cohort, protocol and measurement method. 0.24 sits between
    # the two. This is the first out-of-sample support SC_max has had; before
    # it, the parameter rested on de Boer alone. Miyamura reports SEM with
    # n=6, so this corroborates magnitude and does not tighten the prior --
    # SC_max stays flagged as uncalibrated when absolute D is presented
    # (CLAUDE.md). Derivation: scripts/fit_melanin_kinetics.py sec.6.
    # Caveat (added 2026-08-25): Miyamura's own Table 1 flags the SC cell NS
    # ("20.7 +/- 3.6 NS" vs control; text: SC thickness "did not change
    # significantly in any of the specimens", "increasing trend"). With
    # n=6 SEM and suberythemal dose, NS is a power statement, not evidence
    # of absence -- corroborates TREND and MAGNITUDE, not an independently
    # significant replication. Verified against the PDF twice independently.
    # FST IV-VI is unchanged at 0.36: Miyamura enrolled FST II-III only and
    # provides no constraint on darker skin.
    SC_max_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.24, 2: 0.24, 3: 0.24,
        4: 0.36, 5: 0.36, 6: 0.36,
    })
    # [CALIBRATION 2026-09-06, user-approved; REVISED SAME DAY] Value stays at
    # 1.0. What changed is its PROVENANCE, which is the real result here.
    # Bruls et al. 1984 Table 3 (in vivo SC half-value layer d_half(lambda)),
    # convolved against the vitamin-D action spectrum
    # (data/spectra/vitd_action_spectrum_young2021.csv, the repo's preferred
    # blue-shifted spectrum) over their common range 284-313 nm (weighted mean
    # wavelength 296.11 nm, mu_eff = 0.075446/um, d_half_eff = 9.1873 um),
    # gives K_SC = ln(2) * SC_BASELINE_THICKNESS_UM / d_half_eff = 1.509 at
    # SC_BASELINE_THICKNESS_UM = 20.0.
    # THAT 1.509 IS AN UPPER BOUND, NOT THE OPERATING VALUE, and it was
    # briefly shipped as the latter before being backed off the same day.
    # Why it is a bound: the derivation assumes every added micron of SC is
    # CORNEOCYTE with the same absorption coefficient as baseline SC. The
    # SC state variable it multiplies is not calibrated on corneocyte
    # hyperplasia -- SC_max comes from de Boer 2025, who attribute their SC
    # change to EDEMA, with epidermal thickness flat (108.6 -> 109.4 um,
    # p = 0.718). Bruls themselves state water does not absorb UV, and measure
    # that hydrating SC from ~19.8 to ~31 um (+56.6%, essentially all water)
    # RAISES UV-B transmission rather than lowering it (their Fig. 4b: drying
    # DECREASES UV-B transmission). So for an edema-driven thickening the
    # correct coefficient is far below 1.509 and could be ~0 or negative.
    # If a fraction f of the modelled thickening is corneocyte and the rest is
    # fluid, the effective coefficient is ~1.509 * f. NO SOURCE MEASURES f.
    # NET EFFECT OF THIS PASS: K_SC moves from an ADMITTEDLY ARBITRARY number
    # (its cited source, Pearse 1987, never measured this attenuation) to an
    # empirical value sitting inside a DERIVED BOUND of [~0, 1.509], at 66.3%
    # of that bound. The value did not move; its status did. Do not read 1.0
    # as calibrated -- read it as bounded and uncontradicted.
    # Full working, the discarded single-wavelength and CIE-weighted
    # alternatives, and every caveat (no dark skin, Bruls-vs-de Boer epidermal
    # thickness conflict, Diffey unavailable) are in the K_SC entry of this
    # module's docstring.
    K_SC: float = 1.0
    # UV-driven acute barrier disruption, per SED. Now ADDITIVE (subtracted from
    # the hyperplasia term) so net SC can fall BELOW baseline, which de Boer 2025
    # Supp Table S1 shows it does: SC thickness 5.8 -> 4.3 -> 7.5 um at
    # T0 / 5 wk / 9 wk. The old multiplicative form could only shrink SC toward
    # zero, so no value of k_damage could fit the dip -- it was pinned at 0.0 for
    # a structural reason that was misdiagnosed as missing data.
    k_damage: float = 0.0
    # Repair rate of acute damage, 1/day. [PRIOR, UNCITED] No measured barrier-
    # repair rate constant exists in any source this repo holds (checked
    # 2026-08-25) nor in a 100-paper broad literature search (Consensus,
    # 2026-08-26): human in vivo UV-specific SC repair kinetics are a known
    # gap. Closest anchor: Abe & Mayuzumi 1979 -- TEWL returns to control by
    # 3 wk after single 2 MED UVB, but no intermediate timepoints to fit an
    # exponential. Mouse UVB shows delayed peak at ~4 d, recovery by ~5 d
    # (Haratake 1997; Jiang 2007; Holleran 1997), but that is a biphasic
    # insult-repair process, not a first-order decay from time zero.
    # 7-day half-life is structural: days-scale recovery, dip visible at T1,
    # recovered by T2 under continued dosing. Not identifiable:
    # (k_damage, k_repair_SC) is a ridge on T0/T1/T2 group medians
    # (basics/sc_only.py sec.2; scripts/fit_sc_deboer2025.py scanned t1/2 3/7/14 d).
    k_repair_SC: float = np.log(2) / 7.0
    # [C7-SC-FITTED] Evidence-based ranges for SC parameters.
    # SC_max: de Boer 2025 (P2) reports +23% (FP I-III) and +36% (FP IV-VI).
    #   Default 0.30 is the midpoint; use SC_max_by_FST for per-FST values.
    # alpha_SC: Fitted to de Boer 2025. SC reaches ~90% of SC_max within
    #   9 weeks (0.8 SED 3x/week). Prior range 0.003-0.008 was too low.
    #   Experimental range: 0.08-0.20 (fitting uncertainty from 3 timepoints).
    #   Default 0.12 is the mid-range fitted value.
    # k_decay_SC: 21-28 day half-life. SOURCE: Weinstein & Van Scott 1965
    #   (J Invest Dermatol 45:257-262, PMID 5837896) via Rothberg 1961, which
    #   breaks whole-epidermis transit into two legs: stratum malpighii
    #   13 d (basal cell to granular layer) + stratum corneum 13-14 d
    #   (cornification to desquamation) = 26-27 d combined. k_decay_SC models
    #   clearance of EXCESS SC thickness once hyperplasia stops -- i.e.
    #   desquamation of already-cornified cells -- which is the SC-residence
    #   leg (13-14 d) alone, not the combined figure: a cell already in the
    #   SC when UV stops does not need to re-transit the malpighian layer to
    #   shed. Read this way the source's own numbers sit BELOW the adopted
    #   21-28 d range, not above it. This is a TRANSIT TIME either way, not
    #   an exponential half-life; converting via ln(2)/T is a modeling
    #   choice, not a measured rate constant. Pearse 1987 (P23) measured SC
    #   thickness after 6 weeks of 3x/week UVB -- a level, not a rate -- and
    #   is cited for the thickening phenomenon only, never for this number.
    #   de Boer 2025 shows SC still rising at 9 weeks -> decay slower than
    #   rise. Post-1972 literature (Halprin 1972; Bergstresser & Taylor 1977;
    #   Iizuka 1994; Datta 2019) revises whole-epidermis turnover upward to
    #   40-56 d, which is the wrong leg for the same reason the 26-27 d
    #   combined figure is. 21-28 d is retained as an interior, uncalibrated
    #   choice: no source measures SC-hyperplasia regression directly, and
    #   the closest conceptual match (13-14 d SC residence) and the various
    #   whole-epidermis figures (26-56 d) bracket it from opposite sides.
    #   [EVIDENCE NOTE 2026-09-06] Two dansyl-chloride SC-turnover papers read
    #   in full. Neither was previously cited anywhere in this repo (checked
    #   vitd_model.py, vitd_physiology.py, paper/references.bib, scripts/,
    #   docs). Caution on grepping: "Grove" also matches Pulkit Grover, a
    #   co-author, in README.md and presentation/ -- a false hit, not a prior
    #   citation.
    #     Jansen, Hojyo-Tomoko & Kligman 1974 (Br J Dermatol 90:9-12): n=12
    #   adults, 15 body sites, SC ALONE (dansyl chloride binds only nonviable
    #   corneocytes). Site means 8.5-17.2 d; unweighted 15-site mean 13.49 d;
    #   the paper's own prose figure is "about 2 weeks", forehead fastest at
    #   8.5 d.
    #     Grove & Kligman 1983 (J Gerontol 38:137-142): n=100, ages 18-80,
    #   mid-upper INNER ARM (deliberately non-sun-exposed), same method,
    #   compartment confirmed SC-only by shave biopsy. Transit time by age
    #   band (mean +/- SD, n): 18-29 19.8 +/- 2.4 (27); 30-39 20.5 +/- 4.1
    #   (11); 40-49 20.2 +/- 5.7 (14); 50-59 24.7 +/- 4.5 (12); 60-69
    #   31.0 +/- 12.6 (18); 70-80 36.8 +/- 16.3 (18). Transit vs age r = .62,
    #   linear-spline breakpoint at age 48. Horny cell LAYERS are age-invariant
    #   (16.6-17.0), so the lengthening is not a layer-count artefact.
    #   (1) THE 13-14 d SC LEG IS NOW REPLICATED BY AN INDEPENDENT METHOD.
    #   The repo's SC-residence anchor comes from Weinstein & Van Scott 1965
    #   (autoradiography) via Rothberg 1961 (C-14 glycine + tape stripping).
    #   Jansen gets 13.49 d with fluorescence staining, no isotopes, no
    #   stripping. Different method, different cohort, same answer. The
    #   docstring's identification of the CORRECT COMPARTMENT is well
    #   supported.
    #   (2) THE HALPRIN ARGUMENT ABOVE IS A PRIMARY-SOURCE POSITION, NOT ONLY
    #   THIS REPO'S READING. Jansen 1974 p.12, on Halprin's upward revision of
    #   whole-epidermis turnover: "His calculations, however, do not apply to
    #   the value for horny layer replacement. The figure of 2 weeks has been
    #   confirmed once more in this study." That is the same wrong-leg argument
    #   made here, in Halprin's own decade, by one of the authors whose data
    #   underpin the alternative.
    #   (3) BUT THE TWO PAPERS DISAGREE WITH EACH OTHER BY ~47%, AND NEITHER
    #   RECONCILES IT. Grove 19.8 d (young adults, inner arm) vs Jansen 13.49 d
    #   (15-site mean) for the same compartment. Grove & Kligman argue the
    #   RADIOACTIVE methods run falsely short because repeated tape stripping is
    #   itself a proliferative stimulus -- which, if right, biases the repo's
    #   13-14 d anchor SHORT -- but that argument does not explain the gap with
    #   Jansen, who used Grove's own method without stripping. Site and cohort
    #   differ. Treat 13.5-19.8 d as the honest width of the SC-residence leg,
    #   not 13-14 d.
    #   (4) UNDER THE REPO'S OWN ln(2)/T CONVENTION BOTH PAPERS SIT FAST OF THE
    #   MODEL for every adult under 50. ln(2)/13.49 = 0.0514/d, ln(2)/19.8 =
    #   0.0350/d, against the shipped ln(2)/28 = 0.0248/d. Grove's three
    #   under-50 bands (19.8, 20.5, 20.2 d) all fall OUTSIDE SC_PRIOR_RANGES on
    #   the fast side (floor 21 d); 28 d lands between Grove's 50-59 and 60-69
    #   bands. So if k_decay_SC were read as a measured SC transit, the shipped
    #   value describes a ~60-year-old, not a general adult.
    #   (5) THE TRANSIT-TIME-IS-NOT-A-HALF-LIFE WARNING ABOVE GETS STRONGER,
    #   NOT WEAKER. Grove & Kligman restate Weinstein & Van Scott's finding that
    #   corneocytes "move in unison as a layer", i.e. PLUG FLOW, and Jansen's
    #   endpoint is time to COMPLETE disappearance of marker -- a plug-flow
    #   endpoint by construction. A plug-flow compartment has no half-life at
    #   all: at a constant layers/day removal the excess clears LINEARLY and
    #   then stops. ln(2)/T is therefore a convention, and the choice of
    #   convention moves the implied rate by ~2x (k = ln2/T gives 0.0350/d at
    #   Grove's 19.8 d; k = 1/T gives 0.0505/d; matching only the 50%-cleared
    #   point gives 0.0700/d).
    #   (6) NEITHER PAPER MEASURES WHAT THIS PARAMETER MODELS. Grove & Kligman
    #   deliberately chose a site "where photoaging can play no role"; Jansen
    #   applied no UV. Both measure BASELINE turnover in unstimulated skin, not
    #   regression of UV-INDUCED EXCESS thickness. The [PRIOR] tag and the
    #   statement above that no source measures SC-hyperplasia regression
    #   directly both SURVIVE this pass unchanged. What improved is the quality
    #   of the bracketing, not the existence of a measurement.
    #   (7) AGE IS A BIGGER MODERATOR THAN THE WHOLE PRIOR WIDTH, AND THE MODEL
    #   HAS NO AGE TERM. Grove's transit nearly doubles across adulthood
    #   (19.8 -> 36.8 d, 1.86x) against a prior interval spanning 1.33x, the
    #   effect is non-linear (breakpoint at 48), and the 70-80 SD alone
    #   (16.3 d) exceeds the entire 21-28 d range. Recorded, NOT acted on.
    #   CAVEATS: both papers are all-Caucasian with no FST typing and no
    #   FST IV-VI data -- against this project's dark-skin priority, any
    #   k_decay_SC anchored to them is a light-skin extrapolation. Endpoint in
    #   both is subjective (complete loss of fluorescence by eye under a Wood's
    #   lamp) at daily resolution, so >= 1 d quantisation, no inter-observer
    #   data. Grove: single site; no sex-stratified results despite both sexes
    #   enrolled; n=11-18 per band away from the youngest; the 5-6 oldest
    #   subjects who could not be blistered are dropped from the layer/rate rows
    #   and were the SLOWEST (M=40.2 vs 32.9, p<.05), so the age effect is if
    #   anything understated. Jansen: n=12, no ages given, dispersion in its
    #   table is unlabelled (read as SEM on magnitude grounds), and its site-4
    #   row did not reconcile against the printed mean on the scan available --
    #   flagged as a legibility problem, not silently resolved; a cleaner copy
    #   would move the 15-site mean by at most 13.49 -> 13.16 d, changing
    #   nothing above.
    #   [INCONSISTENCY RESOLVED 2026-09-13] scripts/parameter_optima.py lines 109-114
    #   were updated to justify this prior by the SC-residence leg alone (13-14 d),
    #   resolving the combined whole-epidermis discrepancy, and citing Jansen 1974.
    # K_SC: Attenuation strength. Pearse 1987 did not measure vitamin D
    #   attenuation directly. K_SC=1.0 gives exp(-0.30)=74% at SC_max.
    #   Range 0.5-2.0 reflects uncertainty.
    # k_damage: de Boer 2025 Supp Table S1 shows SC dipping BELOW baseline at
    #   week 5 (T1: 5.8 -> 4.3 um ALL, FP I-III: 5.6 -> 4.3 um, FP IV-VI: 6.6 -> 4.3 um)
    #   then overshooting at week 9 (T2: 7.5 um ALL, FP I-III: 6.9 um, FP IV-VI: 9.0 um).
    #   Significance (Supp S1, double-verified 2026-08-25): T1-vs-T0 dip p=.016
    #   pooled, NS within stratum (I-III .170, IV-VI .134); T2-vs-T0 overshoot
    #   p<=.002 in ALL groups. Dip is pooled-real; stratum-level evidence weaker.
    #   At T1 (+8.8 SED), net SC = 4.3/5.8 - 1 = -25.9% below baseline.
    #   Identifiability of k_damage requires individual-level time-course data at
    #   sub-session resolution, NOT just T0/T1/T2 group medians. k_damage is an acute
    #   per-session effect that largely repairs between sessions (k_repair_SC half-life ~7 days),
    #   so steady-state damage under 3x/week dosing is k_damage * 0.8 / (1 - exp(-k_repair * 2.33)) = k_damage * 3.65.
    #   Default remains 0.0 until sub-session resolution data is available.
    SC_PRIOR_RANGES = {
        "SC_max": (0.20, 0.40),
        "alpha_SC": (0.08, 0.20),
        "k_decay_SC": (np.log(2) / 28, np.log(2) / 21),
        "K_SC": (0.5, 2.0),
        "k_damage": (0.0, 0.15),
    }
    # [CALIBRATION 2026-09-06, user-approved] {I 0.0, II 0.07, III 0.21,
    # IV 0.44, V 0.78, VI 1.00} -> {I 0.0381, II 0.1715, III 0.3231,
    # IV 0.5241, V 0.7539, VI 1.0427 -> clipped 1.00}. Replaces the n=48 UK
    # Young 2020 trial anchor (per-FST n: II 13, III 15, IV 12, V 4, VI 4)
    # with Narea-Jimenez et al. 2026 (Biomed Phys Eng Express 12:025078,
    # n=1005, Venezuelan mestizo, constitutive inner-forearm site,
    # clinician-assessed FST), n=21x higher at FST V and 8x higher at FST VI.
    # SAME MAPPING, new inputs: M = 1 - (ITA_U + 30) / 85, applied to Narea's
    # constitutive ITA means (I 51.76, II 40.42, III 27.54, IV 10.45,
    # V -9.08, VI -33.63). Full derivation, the coupled K_const refit it
    # requires, and every caveat: see the M_baseline_by_FST entry in this
    # module's docstring and the [EVIDENCE NOTE 2026-09-06] block below it.
    # K_const multiplies these values, so this rescales the whole skin-type
    # penalty and K_const WAS refitted in the same change (see K_const entry).
    M_baseline_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0381, 2: 0.1715, 3: 0.3231, 4: 0.5241, 5: 0.7539, 6: 1.00,
    })

    def __setattr__(self, name, value):
        """Refuse to hold a non-structural value for a RETIRED parameter.

        [A6, 2026-08-18] Retirement is enforceable, not advisory. Assigning
        the pinned value is always allowed (the dataclass __init__ does that).
        Anything else raises unless the caller is running a sweep.
        """
        if name in RETIRED_PARAMETERS and _retired_sweep_depth == 0:
            pinned = _RETIRED_VALUES[name]
            if value != pinned:
                raise RetiredParameterError(
                    f"{name} is a RETIRED parameter, fixed at {pinned!r} "
                    f"(vitd_model.RETIRED_PARAMETERS). Refusing to set it to "
                    f"{value!r}.\n"
                    f"  Reason: {RETIRED_PARAMETERS[name]}\n"
                    f"  For ablation/sensitivity, wrap the assignment in "
                    f"`with vitd_model.allow_retired_parameter_sweep():`.\n"
                    f"  To make it calibratable again, delete its "
                    f"RETIRED_PARAMETERS entry and add a parameter_approvals "
                    f"entry in calibration_approval.yaml.")
        object.__setattr__(self, name, value)

    def M_baseline(self, fst: int) -> float:
        return self.M_baseline_by_FST[fst]

    def M_ceiling(self, fst: int) -> float:
        """Maximum attainable melanin index for this FST (fully tanned).

        Always M_baseline(fst) + M_TAN_HEADROOM, so UV exposure increases
        melanin for EVERY FST. NOT capped at 1.0: 1.0 is the *constitutive*
        FST VI reference on the Datta 2019 reflectance scale, not a physical
        ceiling. FST VI undergoes facultative melanogenesis, so a fully tanned
        FST VI sits above untanned FST VI (M = 1.20 here). Capping at 1.0
        silently makes tanning a no-op for FST VI.

        Synthesis attenuation stays well-behaved above 1.0: only the
        difference (M - M_baseline) enters exp(-K_ATTEN * excess_M).
        """
        return self.M_baseline(fst) + self.M_TAN_HEADROOM

    def bsa_factor(self, bsa: Optional[float] = None) -> float:
        """[C1] Sub-linear BSA scaling factor relative to BSA_REF.

        S ∝ (bsa / BSA_REF) ** BSA_EXP.  Default bsa = self.exposed_bsa.
        Returns 1.0 when bsa == BSA_REF (backward-compatible default).
        P5 (Bogh 2011) ln-BSA coefficient; P8 (Matsuoka 1990) BSA plateau above 33%.
        """
        b = self.exposed_bsa if bsa is None else bsa
        b = max(0.0, min(1.0, b))
        if b <= 0.0:
            return 0.0
        return (b / BSA_REF) ** BSA_EXP

    @staticmethod
    def k_decay_D(D: float) -> float:
        """State-dependent D decay rate from Datta 2017 power-law fit.

        # Power-law fitted to Datta 2017 three-group data (89d, 148d, 199d)
        t_half(D) = 2953 * D^(-0.72)
        Returns k = ln(2) / t_half(D) in units of 1/days.
        """
        t_half = 2953.0 * D ** (-0.72)
        return np.log(2.0) / t_half


def synthesis_per_session(sed: float, M: float, M_baseline: float,
                          D: float,
                          SC: float,
                          params: ModelParams,
                          A_override: float = None,
                          K_const_override: float = None,
                          K_ATTEN_override: float = None,
                          E_source_override: float = None,
                          D_max_override: float = None,
                          K_SC_override: float = None,
                          exposed_bsa_override: float = None) -> float:
    """Compute per-session vitamin D synthesis (nmol/L).

    Aligns logarithmic dose-response with Webb 2021 formulation:
    delta = c * ln(SED * VDE) + b, using effective D3-weighted dose (sed * E_source).

    [C1] exposed_bsa_override: override exposed BSA fraction for this session.
    [C2] E_source_override: per-study spectral weighting (use SOURCE_CALIBRATION_TABLE).
    """
    if sed <= 0:
        return 0.0
    A_use = params.A if A_override is None else A_override
    Kc_use = params.K_const if K_const_override is None else K_const_override
    K_use = params.K_ATTEN if K_ATTEN_override is None else K_ATTEN_override
    E_use = params.E_source if E_source_override is None else E_source_override
    D_max_use = params.D_max if D_max_override is None else D_max_override
    K_SC_use = params.K_SC if K_SC_override is None else K_SC_override
    bsa_f = params.bsa_factor(exposed_bsa_override)  # [C1]

    excess_M = max(0.0, M - M_baseline)
    constitutive_factor = np.exp(-Kc_use * M_baseline)
    facultative_factor = np.exp(-K_use * excess_M)
    sc_factor = np.exp(-K_SC_use * SC)
    # Feedback factor g(D). DEFAULT is 'biphasic' since 2026-08-18 (approved
    # item A2); 'linear' and 'power' are LEGACY and are the only modes that read
    # D_max, so D_max_override is inert unless one of them is selected.
    #   'biphasic'   1 / (1 + (D/56)**2)              Scholze 2026 inflection
    #   'saturating' exp(-ln2 * D / D_half_feedback)  scale param unsourced
    #   'linear'     max(0, 1 - D/D_max)              LEGACY, hard ceiling
    #   'power'      max(0, 1 - (D/D_max)**p)         LEGACY, hard ceiling
    f_mode = getattr(params, 'feedback_mode', 'biphasic')
    if f_mode == 'linear':
        feedback_factor = max(0.0, 1.0 - D / D_max_use)
    elif f_mode == 'power':
        feedback_factor = max(0.0, 1.0 - (D / D_max_use) ** getattr(params, 'feedback_power', 1.0))
    elif f_mode == 'saturating':
        # saturating feedback: exp(-decay * D) where decay = ln(2) / D_half_feedback
        decay = np.log(2.0) / getattr(params, 'D_half_feedback', 50.0)
        feedback_factor = np.exp(-decay * D)
    elif f_mode == 'biphasic':
        # DEFAULT. Mechanistic: 25-hydroxylation efficiency, inflection
        # ~56 nmol/L (Scholze et al. 2026, Commun Med 6:105). Never exactly
        # zero -- cutaneous synthesis does not stop outright at any serum level.
        feedback_factor = hydroxylation_efficiency(D)
    else:
        # Was a silent fall-through to 'linear'. That was harmless while
        # 'linear' was the default and is a trap now that it is not: a typo'd
        # mode would silently reinstate the withdrawn hard ceiling.
        raise ValueError(
            f"unknown feedback_mode {f_mode!r}; expected one of "
            f"'biphasic' (default), 'saturating', 'linear', 'power'")

    # Previtamin-D3 photostationary plateau: per-session yield saturates in
    # absolute terms, so one large session cannot substitute for several small
    # ones (Holick 1981 plateau at 10-15% of 7-DHC; van Dijk 2016).
    photo_f = (photodegradation_factor(sed * E_use)
               if getattr(params, 'use_photodegradation', False) else 1.0)

    # Volumetric dilution: the same cutaneous input raises serum 25(OH)D less
    # in a larger body (Drincic 2012, hyperbolic in total body weight).
    weight_f = body_weight_dilution_factor(getattr(params, 'body_weight_kg', None))

    return (A_use * np.log(1.0 + sed * E_use) * bsa_f *
            constitutive_factor * facultative_factor * sc_factor *
            feedback_factor * photo_f * weight_f)


def compute_safety_sidecar(sed: float, fst: int, accumulated_sed_7d: float = 0.0) -> Dict:
    """[C6-safety] Safety sidecar to assess DNA damage / CPD risk.

    P22 (Rhodes 2017) and P24 (Shih/Farrar/Rhodes 2018) show that cyclobutane pyrimidine
    dimers (CPD) and DNA damage occur concurrently with vitamin D synthesis even
    at suberythemal doses, and that basal-layer CPD clearance and risk differ
    strongly by FST.

    This function returns qualitative risk flags and recommendations based on
    session SED, FST, and recent 7-day cumulative UV dose.
    """
    # FST MED values from Shih/Farrar/Rhodes 2018 (P24): FST I-II ~2-3 SED, FST V ~10 SED, FST VI ~15 SED
    med_by_fst = {1: 2.1, 2: 2.6, 3: 3.2, 4: 5.6, 5: 7.5, 6: 15.2}
    med = med_by_fst.get(fst, 3.5)

    fraction_med = sed / med

    # CPD Risk Level
    # P24: even at 20-40% MED, basal CPD is detectable in FST I-II, but protected in FST V-VI
    if fst <= 2:
        if fraction_med >= 0.8:
            cpd_risk = "CRITICAL (High basal CPD, high erythema risk)"
        elif fraction_med >= 0.4:
            cpd_risk = "HIGH (Significant basal CPD, suberythemal damage)"
        elif fraction_med >= 0.2:
            cpd_risk = "MODERATE (basal CPD detectable, low erythema)"
        else:
            cpd_risk = "LOW"
    elif fst <= 4:
        if fraction_med >= 0.8:
            cpd_risk = "HIGH (Erythema risk, basal CPD likely)"
        elif fraction_med >= 0.4:
            cpd_risk = "MODERATE (Suberythemal DNA damage)"
        else:
            cpd_risk = "LOW"
    else:  # FST V-VI
        if fraction_med >= 0.8:
            cpd_risk = "MODERATE (Erythema risk, basal layers relatively protected by melanin)"
        else:
            cpd_risk = "LOW (Basal layer highly protected, minimal CPD risk)"

    # Chronic exposure risk (7-day accumulation)
    # Repeated low dose (e.g. de Boer 2025 0.8 SED 3x/wk = 2.4 SED/wk)
    # If 7d dose exceeds 2x MED, high risk of photoaging/damage accumulation
    if accumulated_sed_7d > 2.0 * med:
        chronic_risk = "HIGH (Risk of chronic photoaging, stratum corneum thickening barrier upregulation)"
    elif accumulated_sed_7d > 1.0 * med:
        chronic_risk = "MODERATE (Accumulating sub-erythemal damage)"
    else:
        chronic_risk = "LOW"

    return {
        "individual_med_sed": med,
        "fraction_med": fraction_med,
        "cpd_risk_level": cpd_risk,
        "chronic_risk_level": chronic_risk,
        "action_spectrum_warning": (
            "Warning: Action spectra for DNA damage (erythema) and vitamin D synthesis "
            "differ. Suberythemal exposures still cause cumulative DNA damage."
        )
    }


def build_pulse_schedule(duration_days: int, sed_per_session: float,
                         interval_days: int, start_day: int = 0
                         ) -> List[Tuple[float, float]]:
    """Periodic UV pulses. Returns list of (day, SED) tuples."""
    return [(d, sed_per_session)
            for d in range(start_day, duration_days, interval_days)]


def build_weekly_schedule(duration_days: int, sed_per_session: float,
                          days_of_week: List[int]) -> List[Tuple[float, float]]:
    """Schedule on specific weekdays (0=Mon ... 6=Sun)."""
    sched = []
    for w in range(int(np.ceil(duration_days / 7)) + 1):
        for d in days_of_week:
            day = w * 7 + d
            if day < duration_days:
                sched.append((day, sed_per_session))
    return sched


def simulate(fst: int,
             schedule: List[Tuple[float, float]],
             duration_days: int,
             params: ModelParams = None,
             D0: float = None,
             M0: float = None,
             SC0: float = 0.0,
             k_decay_M_override: float = None,
             alpha_override: float = None,
             K_const_override: float = None,
             K_ATTEN_override: float = None,
             E_source_override: float = None,
             D_max_override: float = None,
             k_decay_SC_override: float = None,
             alpha_SC_override: float = None,
              K_SC_override: float = None,
              k_damage_override: float = None,
              k_repair_SC_override: float = None,
              SC_max_override: float = None,
              M_max_override: float = None,
              exposed_bsa_override: float = None,
               dose_framing: str = "unspecified") -> Dict:
    """Simulate vitamin D dynamics under a UV pulse schedule.

    Parameters
    ----------
    exposed_bsa_override : float, optional
        [C1] Override exposed BSA fraction (0-1) for all sessions in this run.
        If None, uses params.exposed_bsa (default BSA_REF = 0.35).
    dose_framing : str
        [C4] Descriptor for how UV doses are specified in this scenario.
        Must be one of:
          'fixed_absolute_SED'   - each session delivers a fixed SED regardless of MED
          'fixed_cumulative_SED' - total UV budget is split across sessions
          'fixed_per_session_SED'- per-session SED is constant; cumulative varies (default)
          'fixed_fractional_MED' - each session is a fraction of the individual MED
        This label is stored in the result dict to document scenario provenance.
        P24 (Shih/Farrar/Rhodes 2018): fractional-MED dosing removes apparent skin-type
        effect; P22/P3: fixed absolute SED can leave FST V deficient. Never treat
        framings as interchangeable in output reports.
    """
    if params is None:
        params = ModelParams()
    M_base = params.M_baseline(fst)

    Kc_eff = params.K_const if K_const_override is None else K_const_override
    beta_eff = params.beta if k_decay_M_override is None else k_decay_M_override
    K_eff = params.K_ATTEN if K_ATTEN_override is None else K_ATTEN_override
    E_eff = params.E_source if E_source_override is None else E_source_override
    D_max_eff = params.D_max if D_max_override is None else D_max_override
    beta_SC_eff = params.k_decay_SC if k_decay_SC_override is None else k_decay_SC_override
    alpha_SC_eff = params.alpha_SC if alpha_SC_override is None else alpha_SC_override
    K_SC_eff = params.K_SC if K_SC_override is None else K_SC_override
    k_damage_eff = params.k_damage if k_damage_override is None else k_damage_override
    SC_max_eff = (params.SC_max_by_FST.get(fst, params.SC_max)
                  if SC_max_override is None else SC_max_override)
    # Per-FST melanin ceiling; always above M_base so tanning is monotone.
    M_ceil = params.M_ceiling(fst) if M_max_override is None else M_max_override

    D = params.D0_default if D0 is None else D0
    M = M_base if M0 is None else M0
    SC = SC0
    # Cutaneous/circulating D3 awaiting 25-hydroxylation, in nmol/L of
    # eventual 25(OH)D equivalent. Zero unless use_transit_delay is on.
    D3_pool = 0.0
    k_transit = (np.log(2.0) / params.transit_half_life_days
                 if getattr(params, 'use_transit_delay', False) else 0.0)
    # Two-process SC: net SC = SC_build - SC_damage. SC0 seeds the build term so
    # that a caller passing SC0 gets the old behaviour when k_damage = 0.
    SC_build = SC0
    SC_damage = 0.0
    k_repair_eff = (params.k_repair_SC if k_repair_SC_override is None
                    else k_repair_SC_override)

    schedule = sorted(schedule, key=lambda x: x[0])
    pulse_days = [p[0] for p in schedule]
    pulse_seds = [p[1] for p in schedule]

    t_record, D_record, M_record, SC_record = [0.0], [D], [M], [SC]
    synthesis_log = []
    safety_log = []  # [C6-safety] safety sidecar logs
    cumulative_SED = 0.0
    current_t = 0.0
    pulse_idx = 0

    while current_t < duration_days:
        next_t = (pulse_days[pulse_idx]
                  if pulse_idx < len(pulse_days)
                  else duration_days + 1)
        end_t = min(next_t, duration_days)

        if end_t > current_t:
            # March the state forward in dt-sized sub-steps, advancing current_t
            # each iteration. The recorded samples ARE the state trajectory --
            # there is no separate "big step" to the interval end, which would
            # decay from a stale origin. (This structure was originally forced
            # by a source-term bug in the since-removed oral pathway; it is
            # still correct and still required for the D3 transit term.)
            dt = 0.5
            t_steps = np.arange(current_t + dt, end_t, dt)
            if len(t_steps) == 0 or t_steps[-1] < end_t:
                t_steps = np.append(t_steps, end_t)
            for ti in t_steps:
                delta_time = ti - current_t
                # Transit: D3 pool drains into the 25(OH)D compartment.
                # Applied before decay so newly arrived D3 is subject to
                # clearance over the remainder of the sub-step.
                if k_transit > 0.0 and D3_pool > 0.0:
                    transferred = D3_pool * (1.0 - np.exp(-k_transit * delta_time))
                    D += transferred
                    D3_pool -= transferred
                D = D * np.exp(-params.k_decay_D(D) * delta_time)
                M = M_base + (M - M_base) * np.exp(-beta_eff * delta_time)
                # Hyperplasia regresses at epidermal-turnover rate; acute damage
                # repairs faster (barrier recovery is days, not weeks).
                SC_build = SC_build * np.exp(-beta_SC_eff * delta_time)
                SC_damage = SC_damage * np.exp(-k_repair_eff * delta_time)
                SC = SC_build - SC_damage
                current_t = ti
                t_record.append(ti)
                D_record.append(D)
                M_record.append(M)
                SC_record.append(SC)
            current_t = end_t

        if (pulse_idx < len(pulse_days)
                and abs(current_t - pulse_days[pulse_idx]) < 1e-9):
            sed = pulse_seds[pulse_idx]
            delta_D = synthesis_per_session(sed, M, M_base, D, SC, params,
                                             K_const_override=Kc_eff,
                                             K_ATTEN_override=K_eff,
                                             E_source_override=E_eff,
                                             D_max_override=D_max_eff,
                                             K_SC_override=K_SC_eff,
                                             exposed_bsa_override=exposed_bsa_override)
            if k_transit > 0.0:
                # Route synthesis through the skin/serum D3 reservoir instead of
                # crediting serum 25(OH)D instantaneously.
                D3_pool += delta_D
            else:
                D = D + delta_D
            # Apply genotype A_offset (from vitd_genetics.py presets) — additive offset
            # to per-session synthesis, independent of D feedback.
            if hasattr(params, '_genotype_A_offset'):
                D = D + params._genotype_A_offset
            if alpha_override is not None:
                alpha_p = alpha_override
            else:
                alpha_p = params.alpha_by_FST.get(fst, params.alpha)
                UV_halfmax_p = params.UV_halfmax_by_FST.get(fst, params.UV_halfmax)
                hill_factor = sed**2 / (sed**2 + UV_halfmax_p**2)
                alpha_p = alpha_p + (params.alpha_max - alpha_p) * hill_factor
            M = M_ceil - (M_ceil - M) * np.exp(-alpha_p * sed)
            # Two-process stratum corneum (de Boer 2025 Supp Table S1).
            # SC_build : slow hyperplasia toward SC_max (saturating, as before)
            # SC_damage: fast UV-driven barrier disruption, ADDITIVE and
            #            subtracted, so net SC can go BELOW baseline.
            # The old single multiplicative `SC *= exp(-k_damage*sed)` could only
            # shrink SC toward 0 and so could never reproduce the observed dip;
            # k_damage was pinned at 0.0 for that reason, not for lack of data.
            SC_build = SC_max_eff - (SC_max_eff - SC_build) * np.exp(-alpha_SC_eff * sed)
            SC_damage = SC_damage + k_damage_eff * sed
            SC = SC_build - SC_damage
            
            # [C6-safety] Calculate rolling 7-day cumulative SED (including current pulse)
            rolling_7d_sed = sed + sum(
                p_sed for p_day, p_sed in schedule[:pulse_idx]
                if current_t - 7.0 <= p_day < current_t
            )
            safety = compute_safety_sidecar(sed, fst, rolling_7d_sed)
            safety_log.append((current_t, safety))
            
            cumulative_SED += sed
            synthesis_log.append((current_t, sed, delta_D, M, SC))

            t_record.append(current_t)
            D_record.append(D)
            M_record.append(M)
            SC_record.append(SC)
            pulse_idx += 1

    # Quantitative skin-cancer risk for the whole schedule. Unlike the
    # per-session qualitative sidecar, this separates the two epidemiologically
    # distinct pathways: SCC tracks CUMULATIVE dose, while BCC and melanoma
    # track INTERMITTENT intensity (Elwood 1997: intermittent OR 1.71, heavy
    # occupational OR 0.86 i.e. protective; Leiter 2009; Calzavara-Pinton 2015).
    risk = cancer_risk([s for _, s in schedule], fst)

    _VALID_FRAMINGS = {
        "fixed_absolute_SED", "fixed_cumulative_SED",
        "fixed_per_session_SED", "fixed_fractional_MED",
        "unspecified",
    }
    if dose_framing not in _VALID_FRAMINGS:
        import warnings
        warnings.warn(
            f"dose_framing='{dose_framing}' is not a recognized value. "
            f"Use one of: {sorted(_VALID_FRAMINGS)}",
            UserWarning, stacklevel=2,
        )
    return {
        "t": np.array(t_record),
        "D": np.array(D_record),
        "M": np.array(M_record),
        "SC": np.array(SC_record),
        "synthesis_log": synthesis_log,
        "safety_log": safety_log,            # [C6-safety] per-session qualitative flags
        "cancer_risk": risk,                 # schedule-level quantitative indices
        "D3_pool_final": D3_pool,            # un-hydroxylated D3 still in transit
        "SC_build_final": SC_build,          # hyperplasia component
        "SC_damage_final": SC_damage,        # acute barrier-disruption component
        "cumulative_SED": cumulative_SED,
        "fst": fst,
        "params": params,
        "dose_framing": dose_framing,        # [C4] scenario provenance label
        "exposed_bsa": (exposed_bsa_override
                        if exposed_bsa_override is not None
                        else params.exposed_bsa),  # [C1]
    }


def steady_state_D(result, window_days=28):
    t = result["t"]; D = result["D"]
    mask = t >= t[-1] - window_days
    return float(np.mean(D[mask]))


# REMOVED 2026-07-31: efficiency_D_per_SED / gross_synth_per_SED (alias).
# It ranked schedules at Spearman rho = -0.999 against mean 25(OH)D and
# -0.999 against time-integrated deficiency below 50 nmol/L, i.e. it was
# anti-correlated with the clinical outcome it claimed to optimise.
# The finding is reproducible from scripts/metric_redundancy.py, which keeps a
# local copy of the function for that purpose.


def net_final_D_change_per_SED(result, D0=None):
    if D0 is None:
        D0 = result["D"][0]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0:
        return 0.0
    return (result["D"][-1] - D0) / total_sed


def _trapz(y, x):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# REMOVED 2026-07-31: auc_D_per_SED.
# It ranked schedules at Spearman rho = -1.000 against final_D, i.e. it was
# that metric exactly inverted, carrying no independent information. Reporting
# both invited treating one measurement as two agreeing ones.
# The finding is reproducible from scripts/metric_redundancy.py, which keeps a
# local copy of the function for that purpose.


def auc_above_baseline_per_SED(result, D0=None):
    if D0 is None:
        D0 = result["D"][0]
    t = result["t"]
    D = result["D"]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0 or len(t) < 2:
        return 0.0
    y = np.maximum(0.0, D - D0)
    return float(_trapz(y, t)) / total_sed


def auc_above_sufficiency_per_SED(result, threshold_nmol_L=50.0):
    t = result["t"]
    D = result["D"]
    total_sed = result["cumulative_SED"]
    if total_sed <= 0 or len(t) < 2:
        return 0.0
    if np.max(D) <= threshold_nmol_L:
        return 0.0
    y = np.maximum(0.0, D - threshold_nmol_L)
    return float(_trapz(y, t)) / total_sed


def auc_deficit(D, t, threshold):
    """Time-integrated deficiency below threshold.

    deficit = ∫ max(0, threshold - D(t)) dt  over the full time course.
    Units: nmol/L · days.

    Parameters
    ----------
    D : np.ndarray
        Serum 25(OH)D time series.
    t : np.ndarray
        Time points in days (same length as D).
    threshold : float
        Deficiency threshold in nmol/L (e.g., 30 or 50).

    Returns
    -------
    float
        Area under the curve where D < threshold (nmol/L·days).
    """
    y = np.maximum(0.0, threshold - D)
    return float(_trapz(y, t))


def mean_final_value(result, key, window_days=28):
    t = result["t"]; vals = result[key]
    mask = t >= t[-1] - window_days
    return float(np.mean(vals[mask]))

mean_final_M = lambda r, w=28: mean_final_value(r, "M", w)
mean_final_SC = lambda r, w=28: mean_final_value(r, "SC", w)


def final_D(result):
    return float(result["D"][-1])


if __name__ == "__main__":
    p = ModelParams()
    sched = build_pulse_schedule(84, 1.0, 2)
    r = simulate(fst=5, schedule=sched, duration_days=84, params=p)
    print(f"FST V, every-2-day, 1 SED, 12 weeks:")
    print(f"  cum SED: {r['cumulative_SED']:.1f}")
    print(f"  ss D:    {steady_state_D(r):.2f} nmol/L")
    print(f"  final M: {mean_final_M(r):.3f}")
    print(f"  final SC:{mean_final_SC(r):.3f}")
    print(f"  net_final_D_change_per_SED:    {net_final_D_change_per_SED(r):.3f}")
    print(f"  auc_above_baseline_per_SED:    {auc_above_baseline_per_SED(r):.3f}")
    print(f"  auc_above_sufficiency_per_SED: {auc_above_sufficiency_per_SED(r):.3f}")
