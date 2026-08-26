### `melanin_only.py` — melanin: constitutive pigment and facultative tanning

**Constitutive (baseline) pigment**

    response(FST, dose) = ln(1 + dose * E_source) * exp(-K_const * M(FST))

At matched dose the dose term cancels, so every skin-type ratio reduces to
`exp(K_const * ΔM)`. The skin-type gradient rests on two quantities:
`K_const` and the per-FST baseline melanin index `M_baseline_by_FST`.

`K_const` is fitted to four independent within-study routes — two lab
(matched physical dose) and two field (personal dosimetry with the measured
behavioural difference subtracted out): **Felton 2016** (*Br J Dermatol*
175:1320-1328, phototype II vs V, same cabinet, both baselines printed),
**Young 2020** (FST II vs VI, matched dose, two source arms), **Kift 2013 /
Webb 2010** (field polysulphone dosimetry, Manchester), and **Darling 2013**
(the same field-dosimetry treatment, Surrey). Because the constitutive term
is `exp(-K_const * M_baseline(F))`, each study's response ratio is
log-linear in `K_const`, so each inverts to a single implied value; the four
are averaged in log-ratio space. `M_baseline_by_FST` comes from a different
source: individual typology angle (ITA) of unexposed skin from the Young
2020 cohort, mapped to a 0-1 melanin index by `M = 1 - (ITA + 30) / 85`
(clipped to [0, 1]) and averaged per FST. FST I has no data and is
extrapolated from an assumed ITA of about 58°; FST V and VI rest on four
subjects each, and the FST VI cell is a clipped boundary, not a sample mean
(Young et al. 2020).

**Facultative (tanning) dynamics**

    Between UV exposures: melanin relaxes toward the baseline, at rate `beta`.
    At each exposure:     melanin rises toward a ceiling
                           (baseline + M_TAN_HEADROOM), at a rate that grows
                           with dose toward `alpha_max`.
    Synthesis attenuation: exp(-K_const * baseline) * exp(-K_ATTEN * tan)

Four parameters carry this behaviour, drawn from three studies:

- **`alpha`, the tanning-induction rate, and the dose-response curve
  (`alpha_max`, `UV_halfmax`).** Ravnbak, Philipsen & Wulf, *Acta Derm
  Venereol* 88:223-228 (2008) — 24 volunteers (12 Scandinavian, 12
  Indian/Pakistani), FST II-V, given 6 or 12 UVB exposures at fractions of
  each subject's own minimal melanogenesis dose, pigmentation read by
  reflectance spectroscopy. Reflectance is a different unit from the 0-1
  melanin index used here, so converting between them is itself an
  assumption, anchored on each FST's pre-exposure pigmentation. Corroborating
  source for `alpha`: Miyamura et al., *Pigment Cell & Melanoma Research*
  24:136-147 (2011) — 7 volunteers, FST II-III, 10 UVB/UVA exposures over two
  weeks, melanin read directly by densitometry. `alpha` itself cannot be
  pinned down from either study: Ravnbak dosed at only two exposure counts,
  and the tan is already near its plateau at the lower one, so the data
  constrain where the tan saturates, not how fast it gets there. FST VI has
  no datapoint in Ravnbak at all, so that cell is an extrapolation past the
  edge of the source data; Miyamura's FST II-III-only cohort is extrapolated
  down to FST V-VI for the same reason.
- **`beta`, the fade rate.** Same Ravnbak cohort, plus its 2009 UVA
  companion study on the same 24 subjects. Ravnbak reports only a "time to
  total fading" — a straight line fitted through three post-exposure
  timepoints (1 week, 2 and 4 months) and read at its x-intercept, not an
  exponential half-life. Inverting that procedure — assuming an exponential
  decay, sampling it at Ravnbak's own timepoints, and solving for the rate
  that reproduces the reported fading time — supports a half-life range of
  roughly 40-70 days, not a single value. The UVA arm is excluded from this
  fit: Miyamura shows UVA tans raise melanin content by an amount
  indistinguishable from placebo despite a visually comparable tan, so a UVA
  "fading" timescale would describe decay of photo-oxidised pigment, not the
  melanin this model represents.
- **`M_TAN_HEADROOM`, the ceiling above baseline pigment.** Ravnbak's six
  measured tans (Table II: 6 and 12 exposures, three dose fractions),
  converted through the same reflectance-to-index mapping. The adopted value
  is the single largest tan Ravnbak measured, chosen so every measured tan
  in that dataset remains representable, rather than a least-squares
  midpoint.
- **`K_ATTEN`, how strongly a developed tan blocks further synthesis.**
  Matsuoka, Wortsman & Hollis, *J Lab Clin Med* 116:87-90 (1990) — 7
  subjects, given an identical whole-body UVB dose once untanned (midwinter)
  and once tanned (midsummer). The raw difference between the two arms is
  confounded: the tanned arm also started at roughly 1.6× the untanned arm's
  baseline serum 25(OH)D, which alone would blunt further synthesis through
  the body's own feedback, independent of pigment. Separating the two
  requires assuming a feedback form, and the answer moves by roughly
  threefold depending on that choice; the adopted value is the low end of
  that range. Matsuoka's abstract states neither the skin type nor ethnicity
  of its subjects, so applying `K_ATTEN` across all skin types assumes a
  generalisation the source does not confirm.

`K_const` and `K_ATTEN` come from entirely different kinds of study — a
between-subject matched-dose contrast versus a single within-subject
seasonal crossover — and cannot be separated from each other given the data
available (this repository's own identifiability analysis on the model's
response surface, not a finding from either source paper; see
[`../docs/MELANIN_MODEL.md`](../docs/MELANIN_MODEL.md) §2.8). Comparing the
two mechanisms directly: a fully developed tan blocks synthesis by a low
single-digit percentage (Matsuoka et al. 1990), while the constitutive gap
between the lightest and darkest skin types is roughly a twofold
multiplicative effect (Felton et al. 2016; Young et al. 2020; Kift et al.
2013; Darling et al. 2013) — on a multiplicative scale, the constitutive
effect is roughly an order of magnitude larger than the full tanning effect.

**What the evidence supports.** The direction of every mechanism is
supported: melanin builds under UV and fades toward baseline without it, for
every skin type (Ravnbak et al. 2008, 2009). `M_TAN_HEADROOM`'s magnitude is
supported, bounded below by the largest tan actually measured (Ravnbak et
al. 2008, Table II). `alpha` and the dose-response curve are not
identifiable as separate constants from the available data (Ravnbak et al.
2008; Miyamura et al. 2011). `beta` is a range, not a point (Ravnbak et al.
2008, 2009). `K_ATTEN` depends on an assumed feedback form and the skin type
of its source cohort is undisclosed (Matsuoka et al. 1990). FST VI is
extrapolated past the edge of the data for every tanning parameter (Ravnbak
et al. 2008; Miyamura et al. 2011).

Full derivation: [`../docs/MELANIN_MODEL.md`](../docs/MELANIN_MODEL.md).
Reproduce the numbers with:

```
PYTHONIOENCODING=utf-8 python3 scripts/melanin_parameters_report.py
PYTHONIOENCODING=utf-8 python3 scripts/fit_melanin_kinetics.py
PYTHONIOENCODING=utf-8 python3 -m vitd_calibration
```

**Sources**

- Felton SJ, Cooke MS, Kift R, Berry JL, Webb AR, Lam PMW, de Gruijl FR,
  Vail A, Rhodes LE. Concurrent beneficial (vitamin D production) and
  hazardous (cutaneous DNA damage) impact of repeated low-level summer
  sunlight exposures. *Br J Dermatol* 2016;175:1320-1328.
  doi:10.1111/bjd.14863. — Primary route for `K_const` (matched-dose lab
  contrast, FST II vs V).
- Young AR, Morgan KA, Ho T-W, Ojimba N, Harrison GI, Lawrence KP,
  Jakharia-Shah N, Wulf HC, Cruickshank JK, Philipsen PA. Melanin has a
  small inhibitory effect on cutaneous vitamin D synthesis: a comparison of
  extreme phenotypes. *J Invest Dermatol* 2020;140:1418-1426.e1.
  doi:10.1016/j.jid.2019.11.019. — Primary route for `K_const` (matched-dose,
  FST II vs VI, two source arms) and sole source for `M_baseline_by_FST`
  (the `ITA_Unexposed` dataset).
- Kift R, Berry JL, Vail A, Durkin MT, Rhodes LE, Webb AR. Lifestyle
  factors including less cutaneous sun exposure contribute to starkly lower
  vitamin D levels in UK South Asians compared with the white population.
  *Br J Dermatol* 2013;169:1272-1278. doi:10.1111/bjd.12518. — Field-dosimetry
  route for `K_const`, paired with Webb 2010 below.
- Webb AR et al. *Br J Dermatol* 2010;163:1050. Full citation not held in
  this repository's bibliography; the route's figures were taken from Webb
  AR, Alghamdi R, Kift R, Rhodes LE. Dose-response for change in
  25-hydroxyvitamin D after UV exposure: outcome of a systematic review.
  *Endocrine Connections* 2021;10:R248-R266. doi:10.1530/EC-21-0308 (Table
  5), not from the primary paper directly. — Field-dosimetry route for
  `K_const`, Manchester cohort.
- Darling AL et al. *Osteoporos Int* 2013;24:477. Full citation not held in
  this repository's bibliography; used via the same Webb 2021 review Table
  5 above, not the primary paper directly. — Field-dosimetry route for
  `K_const`, Surrey cohort.
- Ravnbak MH, Philipsen PA, Wiegell SR, Wulf HC. Skin pigmentation
  kinetics after UVB exposure. *Acta Derm Venereol* 2008;88:223-228.
  doi:10.2340/00015555-0449. PMID:18480919. — Primary source for `alpha`,
  `UV_halfmax_by_FST`, and `M_TAN_HEADROOM`, and (with its 2009 companion
  below) for `beta`.
- Ravnbak MH, Philipsen PA, Wiegell SR, Wulf HC. Skin pigmentation
  kinetics after exposure to ultraviolet A. *Acta Derm Venereol*
  2009;89:357-363. doi:10.2340/00015555-0640. — UVA companion cohort to the
  above; used only as a negative control excluded from the `beta` fit, not
  as supporting data.
- Miyamura Y, Coelho SG, Schlenz K, Batzer J, Smuda C, Choi W, Brenner M,
  Passeron T, Zhang G, Kolbe L, Wolber R, Hearing VJ. The deceptive nature
  of UVA tanning versus the modest protective effects of UVB tanning on
  human skin. *Pigment Cell Melanoma Res* 2011;24(1). — Corroborating
  (not primary) source for `alpha`; sole basis for the UVA negative-control
  argument used against Ravnbak 2009 above.
- Matsuoka LY, Wortsman J, Hollis BW. Suntanning and cutaneous synthesis
  of vitamin D3. *J Lab Clin Med* 1990;116(1):87-90. PMID:2165512. — Sole
  source for `K_ATTEN`; skin type and ethnicity of its 7 subjects are not
  stated in the paper.

The `K_const`/`K_ATTEN` non-separability claim above is not from any of
these papers — it is this repository's own finding, not a published result.

### `sc_only.py` — stratum corneum thickening

    SC = hyperplasia - acute damage
    hyperplasia:  saturating rise under repeated UV, regressing slowly
    acute damage: rises per exposure, repairs quickly
    attenuation = exp(-K_SC * SC)

Stratum corneum thickness is the *difference* of two processes: slow
hyperplasia that builds and regresses over weeks, and fast acute barrier
disruption that rises and repairs within days.

**De Boer 2025 — the primary dataset.** n=31 volunteers, lower back exposed
to 0.8 SED three times weekly for nine weeks, stratum corneum biopsied at
baseline, week 5, and week 9:

| | baseline | week 5 | week 9 |
|---|---|---|---|
| All phototypes (µm) | 5.8 | 4.3 | 7.5 |
| FP I-III (µm) | 5.6 | 4.3 | 6.9 |
| FP IV-VI (µm) | 6.6 | 4.3 | 9.0 |

The nine-week endpoints, **+23%** (FP I-III) and **+36%** (FP IV-VI), set
the hyperplasia ceiling directly; the build-up rate is fitted so the ceiling
is nearly reached within the same nine-week, 27-session protocol — it is not
an independent kinetic measurement. The data also show something a single
saturating rise cannot produce: thickness falls **~26% below baseline** at
week 5 before overshooting to +29% at week 9. This dip-then-overshoot is why
an acute-damage term exists alongside hyperplasia. It is a real finding, but
a qualified one: the week-5 drop is statistically significant pooled across
phototypes (p=0.016) but not within either phototype group on its own
(p=0.170, p=0.134).

**Miyamura 2011 — independent support for the lighter-skin ceiling.** 7
volunteers, FST II-III, ten UVB exposures over two weeks. Stratum corneum
rose 16.6 to 20.7 µm (**+24.7%**) — a different cohort, protocol, and
measurement method from de Boer, landing within two points of de Boer's own
+23% figure for the same skin-type range, and the strongest independent
support any parameter in this subsystem has. Miyamura's own text, however,
reports this specific change as **not statistically significant** ("did not
change significantly in any of the specimens"), so it corroborates the
magnitude without standing as a second significant measurement. It says
nothing about darker skin: Miyamura enrolled FST II-III only.

**Pearse 1987 — thickening under threshold dose.** 12 women, FST I-II,
exposed three times weekly for six weeks at 0.5x, 1.0x, or 2.0x their own
minimal erythema dose. Thickness rose with dose, but only the 2.0x arm
differed significantly from control; 0.5x and 1.0x did not. De Boer's own
protocol sits well below 1x MED for its cohort, inside the range Pearse
found ambiguous, even though de Boer's much larger sample does report a
significant endpoint. Pearse is a single-timepoint endpoint comparison, not
a time-course, and supplies no information about how fast thickening
regresses once exposure stops.

**The dip, and what a two-process model can and cannot do with it.** A
model combining saturating growth with a separate, fast-repairing damage
term can in principle fall below its own starting value — something growth
alone cannot do. But fitting that combination to the observed dip
(de Boer et al. 2025), while
keeping every parameter inside its physiologically plausible range, still
misses most of the effect; matching the dip closely requires damage and
repair rates well outside that range. The reason is structural: the
observed change is small through week 5 and large by week 9 — an
accelerating response — while both growth and damage-repair are processes
that decelerate, approaching a plateau soon after each exposure. Under
three-times-weekly dosing, the damage term reaches its own steady rhythm
well before week 5 and so cannot supply a swing concentrated in the second
half of the protocol. This is why the acute-damage parameter is left at
zero rather than fitted: the shape of the data, not merely a shortage of
data points, is what resists it.

**`K_SC`, the attenuation strength, has no source.** None of de Boer et al.
(2025), Miyamura et al. (2011), or Pearse et al. (1987) measured vitamin D
synthesis alongside stratum corneum thickness. `K_SC` is a bare prior, and
cannot be separated from `K_ATTEN` (melanin's own attenuation term, Matsuoka
et al. 1990) given the data available.

**What remains uncertain.** The hyperplasia build-up rate and ceiling trade
off against each other — many combinations fit the nine-week endpoint
equally well (de Boer et al. 2025), so neither is a calibrated value on its
own. The repair rate for acute damage has no citation among de Boer et al.
(2025), Miyamura et al. (2011), or Pearse et al. (1987); it was chosen so
the modelled dip recovers by week 9 under continued dosing, which is a
structural choice, not a measurement. `K_SC` is unconstrained by anything
cited here. And the dip these findings are built around, while real, is
established pooled across phototypes but not within either phototype group
individually (de Boer et al. 2025). What would resolve most of this:
individual-level thickness measurements between the three visits already
taken (de Boer et al. 2025), and — separately — a study measuring vitamin D
response against measured stratum corneum thickness directly, of which none
is known to exist.

**Sources**

- de Boer F, Kezic S, van der Lelie C, Motazedi E, Rustemeyer T, van
  Dijk A, Almasian A, Jakasa I, van der Molen H. Effect of repeated
  low-dose UVR exposure on human skin. *J Invest Dermatol*
  2025;145:2304-2312. doi:10.1016/j.jid.2025.01.024. — Primary and only
  source for `SC_max` (both FST bands) and `alpha_SC`, and the sole source
  for the week-5 dip that motivates the acute-damage term.
- Miyamura Y, Coelho SG, Schlenz K, Batzer J, Smuda C, Choi W, Brenner M,
  Passeron T, Zhang G, Kolbe L, Wolber R, Hearing VJ. The deceptive nature
  of UVA tanning versus the modest protective effects of UVB tanning on
  human skin. *Pigment Cell Melanoma Res* 2011;24(1). — Independent
  corroborating (not primary) source for the FP I-III `SC_max` cell only;
  its own reported change is not statistically significant.
- Pearse AD, Gaskell SA, Marks R. Epidermal changes in human skin
  following irradiation with either UVB or UVA. *J Invest Dermatol*
  1987;88:83-87. doi:10.1111/1523-1747.ep12465094. — Source for the
  qualitative claim that stratum corneum thickens under repeated UVB;
  not a source for `k_decay_SC`'s specific half-life, for `k_repair_SC`,
  or for `K_SC` — it contains no decay measurement and no vitamin D
  measurement.

No source among the three above, or found elsewhere, supports `k_decay_SC`'s
specific 21-28 day half-life, `k_repair_SC`'s value, or `K_SC` at any value —
these remain uncited priors (see "What remains uncertain" above).

## Running them

```
PYTHONIOENCODING=utf-8 python3 basics/melanin_only.py
PYTHONIOENCODING=utf-8 python3 basics/sc_only.py
```

Both read parameter values from `vitd_model.ModelParams` at runtime rather
than restating them here.
