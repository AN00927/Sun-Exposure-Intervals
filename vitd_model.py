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
        BSA_EXP = 0.50 (sub-linear; P5 ln-BSA supports diminishing returns
        above ~35% BSA; Matsuoka 1990 P8 shows plateau above 33%).
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
    3.28x response for a 23x area ratio) implies BSA_EXP ~ 0.38 rather
    than the model's 0.50 — recorded here, NOT acted on.
- K_const = 0.98:
    Baseline (constitutive) inhibition coefficient. FST V:II optical factor
    of 2.01x, FST VI:I of 2.66x.
    [CALIBRATION 2026-08-17, user-approved] Raised from 0.33. Derivation:
    `scripts/kconst_recalibration.py`; diagnosis: `scripts/fst_gap_decomposition.py`;
    approval: `calibration_approval.yaml`. Fitted by equal-weight least squares
    in log ratio space to four independent WITHIN-STUDY estimates, two by lab
    matched-dose and two by field personal dosimetry with behavioural dose
    difference removed: Felton 2016 (1.473), Young 2020 both arms (0.278),
    Kift 2013 (0.788), Darling 2013 (1.387). SD across routes 0.56; treat
    0.98 as a central estimate of a 0.3-1.5 range, not a measured constant.
    WHAT 0.33 WAS: ln(1.36) / (M_VI - M_II), a static-optics inversion of
    one Young 2020 ratio that ignored feedback and protocol entirely. On
    Felton 2016's matched-dose protocol it INVERTED the skin-type gap (model
    FST II:V 0.96 vs observed 2.14), because FST V's lower baseline buys
    more headroom under (1 - D/D_max) than the melanin term took away. That
    inversion is gone at 0.98 (model 1.49), on every matched-dose contrast.
    NOT COVERED: behaviour. Shade, clothing and time outdoors are a SEPARATE
    2.1-2.4x factor and live in `vitd_behaviour.py`, defaulting to 1.0.
    Folding them in here would need 4.3-5.7x, above every matched-dose
    measurement in existence.
    DARK-SKIN PROVENANCE: every FST V datapoint behind this number is Indian-
    subcontinent ancestry at UK latitude; every FST VI datapoint is lamp-based.
    All VI:* ratios are extrapolations onto M_baseline(VI) = 1.00, itself an
    unvalidated group-level prior.
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
    IMPORTANT: the claim in run_all.py section 17 that ordering "comes from
    the (1 - D/D_max) feedback term" is too specific. Ordering comes from
    HAVING a decreasing feedback factor: at incumbent A the argmax is 3 d
    under linear, biphasic AND both saturating variants alike.
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
                turnover rate (k_decay_SC, ~28 d half-life)
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

    [PREMISE WARNING] K_SC attenuates synthesis as exp(-K_SC * SC), i.e. as
    added optical path length through a thicker barrier. de Boer attribute the
    SC change to EDEMA, not acanthosis, and in the same table epidermal
    thickness is flat (108.6 -> 109.4 um, p = 0.718). Fluid is not extra
    corneocyte layers. Treat SC as an EMPIRICAL photoadaptation channel, not a
    validated optical mechanism -- and note K_ATTEN ~ K_SC is jointly
    unidentifiable anyway (vitd_sensitivity.py). See MODEL_AND_EVIDENCE.md,
    'Remaining gaps', gap 28.
    [C7-SC] SC_max = 0.30 is a PRIOR, not a calibrated constant. De Boer 2025
    (P2) supports an SC thickening range of +23% (FP I-III) to +36% (FP IV-VI);
    evidence-based prior is 0.20-0.40. Do NOT tighten alpha_SC, k_decay_SC, or
    K_SC without individual-level time-course data from de Boer 2025 supplementary.
- E_source (SOURCE CALIBRATION TABLE):
    [C3] E_source must be a per-study calibration input, NOT a fixed multiplier.
    The SOURCE_CALIBRATION_TABLE below provides evidence-derived starting priors
    from Webb 2021 (P21) and Grigalavicius 2015 (P35). These are priors only;
    Young 2021 PNAS (P20) shows the CIE previtamin-D3 action spectrum needs a ~5 nm
    blue shift, meaning exact values require recomputation from measured spectra.
    Osmancevic 2009 (P18) shows broadband UVB can outperform NB-UVB per SED in
    psoriasis, confirming these cannot be universal
- K_ATTEN = 0.00  --  RETIRED AS A FREE PARAMETER, 2026-08-18 (approved item A6):
    Facultative (tanning) melanin attenuation coefficient. Now a FIXED
    STRUCTURAL CONSTANT (`K_ATTEN_STRUCTURAL`, in `RETIRED_PARAMETERS`).
    MECHANISM NOT REMOVED — exp(-K_ATTEN * excess_M) is still evaluated on
    every session — but the coefficient can no longer be assigned a different
    value without deliberate code change: ModelParams raises
    RetiredParameterError; the only path is `allow_retired_parameter_sweep()`,
    which exists for ablation and sensitivity work and is not a calibration
    route.

    WHY RETIRED, in the order the evidence lands.
    (1) NOT IDENTIFIABLE. Jointly unidentifiable with K_SC and K_const
        (vitd_sensitivity.py: >99% of each pair's response surface is one
        linear combination). Only combinations are constrained; reporting
        K_ATTEN as separately calibrated overstates the evidence.
    (2) IMMATERIAL. Swings steady-state D by 0.3% across its full plausible
        range, against 36% for A. Best-cited, least consequential number in
        the model.
    (3) THE FIT THAT PRODUCED 0.00 IS NOT AN ESTIMATE. 1D MSE minimisation on
        Young 2020 per-person data (n = 23, TL-01 arm) returns 0.00 with
        CI [0.00, 0.00], but the objective is flat to 0.08% across
        K_ATTEN in [0.00, 0.33] on that acute 7.5 SED protocol. The CI is
        a bound artefact of flat-surface fitting, not tight estimation.
        Baseline 25(OH)D and BSA explain that trial's FST II:VI response
        without any facultative term.
    RETIREMENT IS NOT A CLAIM THAT TANNING DOES NOTHING. It claims this
    repository's data cannot measure it. Matsuoka 1990 (n = 7), deconvolved,
    implies 0.34-1.00 depending on assumed feedback form
    (vitd_calibration.derive_K_ATTEN); divergence documented, immaterial at
    present, guarded by `vitd_validation.sanity_photoadaptation_reduction`
    and tests/test_matsuoka1990.py. To reinstate as a free parameter, remove
    from RETIRED_PARAMETERS and supply a dataset separating it from K_SC.
- M_baseline_by_FST:
    [C7] Anchored to ITA_Unexposed means from trial CSVs (ArimedB trial data,
    N=44). Mapping: M = 1 - (ITA_U + 30) / 85, clipped to [0, 1].
    FST I is extrapolated (ITA_U ≈ 58°). Values:
    {I: 0.0, II: 0.07, III: 0.21, IV: 0.44, V: 0.78, VI: 1.00}.
    [C10-PRIOR] GROUP-LEVEL PRIOR — Datta 2019 (P13, PMID: 30656228) shows
    measured pigmentation explains only 1.9-2.6% of individual response variance
    (note Datta 2019 studied pigment SNPs: ASIP, SLC24A4, SLC45A2, not VDR/CYP2R1/CYP24A1).
    Do NOT use as a precise individual predictor or to justify large FST penalties.
    Genetics (VDR, CYP2R1, CYP24A1), supplementation, baseline D, and BSA dominate.
    WARNING: FST V (0.78) and VI (1.00) are group-level priors with NO direct
    melanin-index validation from published data. Do not use as individual predictors.
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
    #   Current values: {V: 0.78, VI: 1.00} (ITA_Unexposed means, ArimedB N=44)
    #   Literature support: WILKES 2015 gives MI>750->FST VI on Mexameter (partial).
    #   ALALUF 2001 gives quantitative melanin in FST V/VI but not as a baseline MI table.
    #   No paper provides separate untanned FST V and FST VI baseline MI means.
    #   Status: COARSE PRIOR — awaiting de Boer 2025 individual-level data.
    #
    # Gap 2 — K_ATTEN:  CLOSED BY RETIREMENT 2026-08-18, not by measurement.
    #   Matsuoka 1990 implies 0.34-1.00 depending on feedback form; only source,
    #   no replication. NEVILLE 2020 confirms the field still lacks tanning-effect
    #   data. MARKIEWICZ 2020: tanning may be redistribution, not pure synthesis.
    #   Status: RETIRED STRUCTURAL CONSTANT at 0.00 — see the K_ATTEN entry above
    #   and RETIRED_PARAMETERS. The gap is not resolved; it is fenced.
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
# Webb's Manchester summer measurement. See calibration_approval.yaml and
# MODEL_AND_EVIDENCE.md 'Open decisions' item 12(a) for the cost: NHANES
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
# evidence that gives `D_max` an interior optimum (Gallagher 2012;
# MODEL_AND_EVIDENCE.md 'Open decisions' item 3). That evidence SURVIVES: the
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
# Sub-linear BSA scaling exponent (P5 ln-BSA term; plateau evidence from P8)
BSA_EXP: float = 0.50


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

# [RETIRED 2026-08-18, approved item A6] Facultative-melanin attenuation
# coefficient. Full reasoning in this module's docstring, K_ATTEN entry.
K_ATTEN_STRUCTURAL: float = 0.00

RETIRED_PARAMETERS: Dict[str, str] = {
    "K_ATTEN": (
        "Retired 2026-08-18 (approved item A6). Jointly unidentifiable with "
        "both K_SC and K_const; swings steady-state D by 0.3% across its full "
        "plausible range; the fitted 0.00 comes from an objective flat to "
        "0.08% over [0.00, 0.33], so its CI is a bound artefact. Mechanism "
        "retained, value fixed at K_ATTEN_STRUCTURAL."
    ),
}

# Module-level constant that each retired parameter is pinned to.
_RETIRED_VALUES: Dict[str, float] = {"K_ATTEN": K_ATTEN_STRUCTURAL}

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
    A: float = 3.114
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
    # [CALIBRATION 2026-08-17, user-approved] 0.33 -> 0.98. Full derivation and
    # provenance in this module's docstring and scripts/kconst_recalibration.py.
    # This is CONSTITUTIVE PIGMENT ONLY. Behaviour (shade, clothing, time
    # outdoors) is a separate 2.1-2.4x factor and lives in vitd_behaviour.py,
    # neutral by default -- do NOT absorb it here.
    K_const: float = 0.98
    # Facultative (tanning) headroom above constitutive pigmentation, on the
    # normalized 0-1 melanin scale. The absolute ceiling for a given FST is
    # M_baseline(FST) + M_TAN_HEADROOM, capped at 1.0 -- see M_ceiling().
    #
    # A single global M_max was wrong: M_max=0.95 sat BELOW
    # M_baseline_by_FST[6]=1.00, so the saturating pulse update
    #     M <- M_max - (M_max - M) * exp(-alpha * sed)
    # DECREASED melanin for FST VI (UV made the skin lighter). FST V/VI is the
    # target population for this project, so that inversion mattered.
    # Headroom 0.20 keeps FST I-III behaviour close to the old global 0.95
    # ceiling while making tanning monotone for every FST.
    # [PRIOR] Magnitude is not separately identified; Ravnbak 2008 reports
    # facultative MI gains but on instrument-specific units.
    M_TAN_HEADROOM: float = 0.20
    M_max: float = 0.95
    # [RETIRED 2026-08-18, approved item A6] NOT A FREE PARAMETER. Pinned to
    # K_ATTEN_STRUCTURAL; assignment of any other value raises
    # RetiredParameterError unless inside allow_retired_parameter_sweep().
    # See RETIRED_PARAMETERS and this module's docstring, K_ATTEN entry.
    K_ATTEN: float = K_ATTEN_STRUCTURAL
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
    # opposed to the feedback artifact in MODEL_AND_EVIDENCE.md, 'Remaining
    # gaps', gap 19.
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
    # k_decay_SC remains at 28d half-life (epidermal turnover literature).
    # k_damage (UV-driven desquamation) fits the week-5 SC dip only
    # weakly from 3 group-mean timepoints; kept at 0.0 until individual-
    # level data is available for joint fitting.
    k_decay_SC: float = np.log(2) / 28.0
    alpha_SC: float = 0.12
    SC_max: float = 0.30
    SC_max_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.23, 2: 0.23, 3: 0.23,
        4: 0.36, 5: 0.36, 6: 0.36,
    })
    K_SC: float = 1.0
    # UV-driven acute barrier disruption, per SED. Now ADDITIVE (subtracted from
    # the hyperplasia term) so net SC can fall BELOW baseline, which de Boer 2025
    # Supp Table S1 shows it does: SC thickness 5.8 -> 4.3 -> 7.5 um at
    # T0 / 5 wk / 9 wk. The old multiplicative form could only shrink SC toward
    # zero, so no value of k_damage could fit the dip -- it was pinned at 0.0 for
    # a structural reason that was misdiagnosed as missing data.
    k_damage: float = 0.0
    # Repair rate of acute damage, 1/day. Barrier recovery after UV is a
    # days-scale process, much faster than the ~28 d epidermal turnover that
    # governs regression of hyperplasia. 7-day half-life.
    # [PRIOR] Not separately measured by de Boer; set so the T1 dip recovers by
    # T2 under continued dosing.
    k_repair_SC: float = np.log(2) / 7.0
    # [C7-SC-FITTED] Evidence-based ranges for SC parameters.
    # SC_max: de Boer 2025 (P2) reports +23% (FP I-III) and +36% (FP IV-VI).
    #   Default 0.30 is the midpoint; use SC_max_by_FST for per-FST values.
    # alpha_SC: Fitted to de Boer 2025. SC reaches ~90% of SC_max within
    #   9 weeks (0.8 SED 3x/week). Prior range 0.003-0.008 was too low.
    #   Experimental range: 0.08-0.20 (fitting uncertainty from 3 timepoints).
    #   Default 0.12 is the mid-range fitted value.
    # k_decay_SC: 21-28 day half-life. Pearse 1987 (P23) measured SC after
    #   6 weeks of 3x/week UVB. Epidermal turnover literature: 26-28 days.
    #   de Boer 2025 shows SC still rising at 9 weeks -> decay slower than rise.
    # K_SC: Attenuation strength. Pearse 1987 did not measure vitamin D
    #   attenuation directly. K_SC=1.0 gives exp(-0.30)=74% at SC_max.
    #   Range 0.5-2.0 reflects uncertainty.
    # k_damage: de Boer 2025 Supp Table S1 shows SC dipping BELOW baseline at
    #   week 5 (T1: 5.8 -> 4.3 um ALL, FP I-III: 5.6 -> 4.3 um, FP IV-VI: 6.6 -> 4.3 um)
    #   then overshooting at week 9 (T2: 7.5 um ALL, FP I-III: 6.9 um, FP IV-VI: 9.0 um).
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
    # [C7][C10-PRIOR] M_baseline_by_FST anchored to ITA_Unexposed means from
    # trial CSVs (ArimedB trial data, N=44). Mapping: M = 1 - (ITA_U + 30) / 85.
    # Documented as rounded group priors. K_const multiplies these values, so
    # any change to them rescales the whole skin-type penalty -- see the
    # K_const entry in this module's docstring.
    M_baseline_by_FST: Dict[int, float] = field(default_factory=lambda: {
        1: 0.0, 2: 0.07, 3: 0.21, 4: 0.44, 5: 0.78, 6: 1.00,
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
    # 1. Estimate individual MED in SED
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
