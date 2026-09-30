# The stratum corneum sub-model: equations, provenance, and what it can and cannot support

**Scope.** This document isolates the stratum corneum (SC) dynamics of the vitamin D
scheduling model: the state equations as *implemented*, every SC-related parameter,
the publication each one comes from, and the arithmetic that maps a published
measurement into a model constant. It also records, in a mandatory limitations section,
what each parameter does *not* rest on.

**On numbers in this document.** The canonical source for every value is the
`vitd_model.py` module docstring and `vitd_model.ModelParams`; nothing here
supersedes them (`CLAUDE.md`). To avoid creating a second table that can drift,
live values and standalone validations are run directly by:

```bash
PYTHONIOENCODING=utf-8 python3 basics/sc_only.py
PYTHONIOENCODING=utf-8 python3 scripts/fit_sc_deboer2025.py
```

Where a number appears inline below to make a derivation readable, it is marked
**[read from code 2026-09-13]** and the canonical location in this repository is given.
Everything is cited `file:line` into this repository.

---

## 1. The stratum corneum state equations as implemented

### 1.1 What the docstring and manuscript say

The module docstring (`vitd_model.py:27-31`) and manuscript describe stratum corneum
photoadaptation as a two-process system whose difference forms the net fractional excess
stratum corneum thickness above baseline:

$$\text{SC}(t) = \text{SC}_{\text{build}}(t) - \text{SC}_{\text{damage}}(t)$$

where:
- $\text{SC}_{\text{build}}$ represents slow, cumulative epidermal hyperplasia (thickening)
  induced by repeated UV exposure and regressing over weeks.
- $\text{SC}_{\text{damage}}$ represents acute barrier disruption and cellular damage
  induced per session and repairing over days.
- $\text{SC}(t)$ is dimensionless fractional excess thickness ($0.0 = \text{baseline}$,
  $+0.20 = 20\%$ thickening, $-0.10 = 10\%$ thinning/desquamation).

### 1.2 What the code actually does

`simulate()` in `vitd_model.py` (`vitd_model.py:2343-2582`) never integrates continuous
differential equations numerically with Euler steps. It evaluates a **hybrid pulse process**:
exact analytic exponential relaxations between sessions, and instantaneous pulse updates at
each session.

**Between pulses** — sub-stepped at $\Delta t = 0.5\text{ d}$ (`vitd_model.py:2474-2479`):

```python
SC_build = SC_build * np.exp(-k_decay_SC * delta_time)
SC_damage = SC_damage * np.exp(-k_repair_SC * delta_time)
```

Both state channels are solved analytically over the interval $\Delta t$ with zero ambient UV.
There is zero discretization error in the decay/repair channels.

**At each pulse** (`vitd_model.py:2516-2524`):

```python
SC_max_p = params.SC_max_by_FST.get(fst, params.SC_max)
SC_build = SC_max_p - (SC_max_p - SC_build) * np.exp(-params.alpha_SC * sed)
SC_damage = SC_damage + params.k_damage * sed
```

Net stratum corneum fractional excess thickness is then evaluated downstream:

```python
SC = SC_build - SC_damage
```

### 1.3 Structural features and upgrades

1. **Additive vs. Multiplicative Damage Formulation.**
   In earlier iterations of the model, acute damage was implemented multiplicatively:
   $\text{SC} \leftarrow \text{SC} \times \exp(-k_{\text{damage}} \cdot \text{sed})$.
   That formulation could only shrink excess $\text{SC}$ asymptotically toward zero from above;
   it was mathematically impossible for net $\text{SC}$ to ever drop below baseline ($\text{SC} < 0$).
   When de Boer et al. (2025) reported that stratum corneum thickness dips $-26\%$ *below*
   baseline at week 5 before overshooting at week 9, the old multiplicative model could not fit
   the dip at any parameter value (forcing $k_{\text{damage}} = 0$).
   The two-process additive formulation ($\text{SC}_{\text{build}} - \text{SC}_{\text{damage}}$)
   upgraded the architecture so that net $\text{SC}$ can represent sub-baseline excursions.

2. **Independence from Melanin State.**
   $\text{M}$ (melanin) and $\text{SC}$ are completely independent dynamical state variables.
   Neither state equation includes the other on its right-hand side. Melanin does not moderate
   stratum corneum hyperplasia rates or acute barrier repair, and stratum corneum does not
   moderate melanogenesis. They couple strictly downstream, multiplicatively, in the cutaneous
   synthesis attenuation function:

   $$\text{attenuation} = \exp(-K_{\text{const}} \cdot M_{\text{baseline}}) \cdot \exp(-K_{\text{ATTEN}} \cdot \Delta M) \cdot \exp(-K_{\text{SC}} \cdot \text{SC})$$

3. **Session-0 Initial Pulse Execution.**
   In both `vitd_model.simulate()` and `basics/sc_only.py`, if a session is scheduled at $t = 0$,
   the pulse update is applied immediately at time zero before any inter-session relaxation occurs.

---

## 2. Parameter-by-parameter provenance

### 2.1 `SC_max` / `SC_max_by_FST` — hyperplasia thickening ceiling

- **Symbols:** `ModelParams.SC_max: float = 0.30` (`vitd_model.py:1790`),
  `ModelParams.SC_max_by_FST: Dict[int, float]` (`vitd_model.py:1828-1831`).
- **Prior range:** `SC_PRIOR_RANGES["SC_max"]` (`vitd_model.py:2039`), ceiling
  widened 2026-09-22 (`calibration_approval.yaml`,
  `SC_max_prior_ceiling_2026_09_22`) — read the interval from the code.
- **Default value:** `0.30` (midpoint across phototypes).
- **Per-FST breakdown:**
  - FST I–III: `0.23` (+23% excess thickness)
  - FST IV–VI: `0.36` (+36% excess thickness)
- **Primary source:**
  de Boer F, Kezic S, van der Lelie C, Motazedi E, Rustemeyer T, van Dijk A, Almasian A,
  Jakasa I, van der Molen H. Effect of repeated low-dose UVR exposure on human skin.
  *J Invest Dermatol* 2025;145:2304-2312. doi:10.1016/j.jid.2025.01.024.
- **Derivation from source:**
  In de Boer et al. (2025), $n=31$ healthy adult volunteers received $0.8\text{ SED}$ of solar-simulated
  radiation 3 times weekly for 9 weeks (27 sessions total, cumulative dose $21.6\text{ SED}$) on the
  unprotected lower back. Stratum corneum thickness was measured by in vivo confocal Raman
  spectroscopy at baseline (T0, day 0), week 5 (T1, day 35), and week 9 (T2, day 63).
  From Supplementary Table S1 (page 12):
  - **FP I–III:** Baseline median $5.6\ \mu\text{m}$ (IQR 4.4–6.8) $\to$ Week 9 median $6.9\ \mu\text{m}$ (IQR 6.3–8.1),
    representing an endpoint fractional increase of:
    $$\frac{6.9 - 5.6}{5.6} = +23.2\% \implies \text{SC}_{\max}(\text{FP I–III}) = 0.23$$
  - **FP IV–VI:** Baseline median $6.6\ \mu\text{m}$ (IQR 4.3–9.2) $\to$ Week 9 median $9.0\ \mu\text{m}$ (IQR 7.2–9.7),
    representing an endpoint fractional increase of:
    $$\frac{9.0 - 6.6}{6.6} = +36.4\% \implies \text{SC}_{\max}(\text{FP IV–VI}) = 0.36$$
  - **Pooled (All FPs):** Baseline median $5.8\ \mu\text{m} \to$ Week 9 median $7.5\ \mu\text{m}$ (+29.3%).
- **Independent corroboration (FP I–III):**
  Miyamura Y, Coelho SG, Schlenz K, et al. The deceptive nature of UVA tanning versus the modest
  protective effects of UVB tanning on human skin. *Pigment Cell Melanoma Res* 2011;24(1):136-147.
  $n=7$ volunteers, FST II–III, exposed to 10 daily UVB exposures over 2 weeks. Mean SC thickness
  rose from $16.6\ \mu\text{m}$ to $20.7\ \mu\text{m}$ (+24.7%). This independent cohort and protocol
  land within 1.5 percentage points of de Boer's $+23\%$ figure for fair skin.
  *Caveat:* Miyamura's text states this change was **not statistically significant** ("did not change
  significantly in any of the specimens"), so it corroborates magnitude, not a second independent
  rejection of the null hypothesis. It enrolled no dark-skin participants (FST IV–VI).
- **Further corroboration and an above-ceiling data point [ADDED 2026-09-23]:**
  Lavker RM, Gerberick GF, Veres D, Irwin CJ, Kaidbey KH. Cumulative effects from repeated exposures
  to suberythemal doses of UVB and UVA in human skin. *J Am Acad Dermatol* 1995;32:53-62.
  DOI 10.1016/0190-9622(95)90184-1, PMID 7822517. Obtained and read in full 2026-09-21 (see
  `docs/SOURCES.md`). FST I–III, $n=23$; three arms at matched biological dose (~0.5×MED):
  solar-simulated radiation (SSR) SC thickness $+22.2\%$ (viable epidermis $-2.1\%$, n.s.) —
  within one point of de Boer's own $+23.2\%$ FST I–III figure, a third independent corroboration
  at matched dose; UVA-only arm $+85.2\%$; UVA-dominant SSRSS arm $+35.8\%$ — the same biological
  dose producing a nearly 4x spread in SC thickening depending on source spectrum, an axis
  `SC_max` does not currently carry (see limitation below).
  Felton SJ, Shih BB, Watson REB, Kift R, Webb AR, Rhodes LE. Photoprotection conferred by low level
  summer sunlight exposures against pro-inflammatory UVR insult. *Photochem Photobiol Sci* 2020;19:810-818.
  DOI 10.1039/c9pp00452a, PMID 33856672. FST II, $n=10$: SC thickness $+41.6\%$ ($29.3 \to 41.5\ \mu\text{m}$)
  over a 6-week, 1.3 SED x 18-session course — a fourth data point, but one that sits ABOVE the
  pre-2026-09-22 prior ceiling (0.40) -- read the current interval from `SC_PRIOR_RANGES['SC_max']`
  above; see `calibration_approval.yaml`'s `SC_max_prior_ceiling_2026_09_22`, which widened the
  ceiling citing this source alongside Bruls 1984. Not adopted as a fifth per-FST cell here;
  flagged for a future ceiling review.
- **Pearse 1987 comparison:**
  Pearse AD, Gaskell SA, Marks R. Epidermal changes in human skin following irradiation with either
  UVB or UVA. *J Invest Dermatol* 1987;88:83-87.
  12 female volunteers, FST I–II, irradiated 3x/week for 6 weeks at $0.5\times$, $1.0\times$, or $2.0\times$
  individual MED. SC thickness increased significantly only in the $2.0\times\text{ MED}$ arm; changes
  at $0.5\times$ and $1.0\times\text{ MED}$ did not achieve significance. De Boer's protocol doses at
  $0.8\text{ SED}$ (roughly $0.2\text{–}0.3\times\text{ MED}$ for FST II), demonstrating that with a
  larger cohort ($n=31$) and 9 weeks of exposure, sub-erythemal dosing does induce measurable hyperplasia.

---

### 2.2 `alpha_SC` — hyperplasia induction rate per SED

- **Symbol:** `ModelParams.alpha_SC: float = 0.12` (`vitd_model.py:1789`).
- **Prior range:** $[0.08, 0.20]$ (`SC_PRIOR_RANGES["alpha_SC"]`, `vitd_model.py:2039`).
- **Default value:** `0.12`.
- **Primary source:** Fitted to de Boer et al. (2025) protocol trajectory.
- **Derivation from source:**
  Under the de Boer protocol (0.8 SED per session, 3 sessions/week for 9 weeks = 27 sessions),
  thickness approaches saturation by week 9.
  For a given $\text{SC}_{\max} = 0.30$ and regression rate $k_{\text{decay,SC}} = \ln(2)/28\text{ d}^{-1}$,
  $\alpha_{\text{SC}} = 0.12$ yields an excess thickness of $+0.177$ at day 63 (~90% of steady-state
  saturation under continuous dosing).
- **The $\alpha_{\text{SC}} \sim \text{SC}_{\max}$ Ridge:**
  Because de Boer 2025 measured only three timepoints (days 0, 35, 63), $\alpha_{\text{SC}}$ and
  $\text{SC}_{\max}$ are not independently identifiable from the endpoint alone. As demonstrated in
  `basics/sc_only.py` (Section 3):
  
  | $\text{SC}_{\max}$ | Implied $\alpha_{\text{SC}}$ | Day-63 Excess $\text{SC}$ | Attenuation ($K_{\text{SC}}=1.0$) |
  |:---:|:---:|:---:|:---:|
  | 0.20 | 0.924 | 0.1774 | $0.819\times$ |
  | 0.25 | 0.205 | 0.1774 | $0.779\times$ |
  | 0.30 | 0.120 | 0.1774 | $0.741\times$ |
  | 0.35 | 0.086 | 0.1774 | $0.705\times$ |
  | 0.40 | 0.068 | 0.1774 | $0.670\times$ |

  $\alpha_{\text{SC}}$ spans a 13.6-fold range ($0.068\text{–}0.924$) across the $\text{SC}_{\max}$
  evidence band while fitting the day-63 endpoint identically. The data constrain the pair's product,
  not either parameter independently.

---

### 2.3 `k_decay_SC` — hyperplasia regression rate (1/day)

- **Symbol:** `ModelParams.k_decay_SC: float = np.log(2) / 28.0` ($\approx 0.024755\text{ d}^{-1}$, `vitd_model.py:1788`).
- **Prior range:** $[\ln(2)/28, \ln(2)/21]$ ($\approx 0.02476\text{–}0.03301\text{ d}^{-1}$, half-life 21–28 days).
- **Status:** **[STRUCTURAL PRIOR, UNCALIBRATED]**. No source in the scientific literature directly
  measures the post-UV regression rate of human stratum corneum hyperplasia.
- **Primary bibliographic anchors:**
  1. Weinstein GD, Van Scott EJ. Turnover times of normal and psoriatic epidermis. *J Invest Dermatol*
     1965;45(4):257-262. PMID:5837896. (via Rothberg S, Crounse RG, Lee JL. *J Invest Dermatol* 1961;37:497-505).
  2. Jansen LH, Hojyo-Tomoko MT, Kligman AM. Improved fluorescence staining technique for estimating
     turnover of the human stratum corneum. *Br J Dermatol* 1974;90(1):9-12. doi:10.1111/j.1365-2133.1974.tb06355.x.
  3. Grove GL, Kligman AM. Age-associated changes in human epidermal cell renewal. *J Gerontol*
     1983;38(2):137-142. doi:10.1093/geronj/38.2.137.
- **Derivation and Compartment Analysis:**
  - *The Compartment Question:* Epidermal renewal involves two distinct anatomical stages:
    1. Stratum malpighii (living nucleated layers, basal to granular layer): ~13–14 days.
    2. Stratum corneum (cornification to surface shedding/desquamation): ~13–14 days.
    Together, these sum to the classic 26–27 day whole-epidermis transit time.
  - *Why Whole-Epidermis Transit is the Wrong Leg:*
    $k_{\text{decay,SC}}$ governs the shedding of **excess, already-formed stratum corneum** once UV
    stops. A corneocyte already residing in the stratum corneum does not re-transit the malpighian layer;
    it clears strictly through corneum desquamation. Therefore, the relevant anatomical compartment is
    the **stratum corneum residence leg alone (13–14 days)**, not whole-epidermis transit (26–56 days).
  - *Independent Replication of the SC Residence Leg:*
    - Weinstein & Van Scott (1965) used radioactive autoradiography ($^3\text{H}$-thymidine and $^{14}\text{C}$-glycine)
      with tape stripping, estimating SC residence at 13–14 days.
    - Jansen, Hojyo-Tomoko & Kligman (1974) measured non-invasive dansyl chloride fluorescence (which binds
      only non-viable corneocytes) across 15 body sites ($n=12$). The unweighted 15-site mean transit time
      was **13.49 days** ("about 2 weeks", forehead fastest at 8.5 d).
    - Grove & Kligman (1983) measured dansyl chloride turnover in $n=100$ healthy volunteers on the mid-upper
      inner arm (unexposed site), confirming shave-biopsy SC localization. Young adult transit times were
      $19.8 \pm 2.4\text{ d}$ (ages 18–29, $n=27$), $20.5 \pm 4.1\text{ d}$ (ages 30–39, $n=11$), and
      $20.2 \pm 5.7\text{ d}$ (ages 40–49, $n=14$).
  - *Why Transit Time is Not an Exponential Half-Life:*
    Both Weinstein & Van Scott (1965) and Grove & Kligman (1983) observe that corneocytes migrate in unison
    as a coherent layer (**plug flow**). In pure plug flow, layers are shed linearly from the surface:
    there is no exponential half-life. Converting a transit time $T$ to a first-order rate constant via
    $k = \ln(2)/T$ is an applied modeling convention. The choice of convention introduces a ~2-fold spread
    ($k = \ln(2)/T \implies 0.035\text{ d}^{-1}$ vs $k = 1/T \implies 0.051\text{ d}^{-1}$ at $T = 19.8\text{ d}$).
  - *Why 21–28 Days is an Interior Structural Choice:*
    The 21–28 day prior half-life ($k = 0.0248\text{–}0.0330\text{ d}^{-1}$) is slower than young-adult
    unstimulated SC transit ($13.5\text{–}19.8\text{ d}$). This is deliberate: under repeated UV exposure,
    hyperplasia involves increased corneocyte adhesion and multiple extra layers, which regress more slowly
    than single-layer turnover. However, because no study has measured post-UV hyperplasia decay kinetics,
    21–28 days remains an uncalibrated prior.

---

### 2.4 `k_damage` — acute barrier disruption per SED

- **Symbol:** `ModelParams.k_damage: float = 0.0` (`vitd_model.py:1871`).
- **Prior range:** $[0.0, 0.15]$ (`SC_PRIOR_RANGES["k_damage"]`, `vitd_model.py:2039`).
- **Default value:** `0.0`.
- **Primary source:** de Boer et al. (2025), Supplementary Table S1.
- **The Empirical Week-5 Dip:**
  In de Boer et al. (2025), median SC thickness across all participants fell from $5.8\ \mu\text{m}$ at T0
  to $4.3\ \mu\text{m}$ at week 5 (T1, after $8.8\text{ SED}$ cumulative dose) before rising to $7.5\ \mu\text{m}$
  at week 9 (T2, after $21.6\text{ SED}$):
  - Pooled (all FPs): $5.8 \to 4.3 \to 7.5\ \mu\text{m}$ (dip: $-25.9\%$, $p = 0.016$)
  - FP I–III: $5.6 \to 4.3 \to 6.9\ \mu\text{m}$ (dip: $-23.2\%$, $p = 0.170$, **not significant**)
  - FP IV–VI: $6.6 \to 4.3 \to 9.0\ \mu\text{m}$ (dip: $-34.8\%$, $p = 0.134$, **not significant**)
- **Structural Mismatch Analysis:**
  `basics/sc_only.py` (Section 2) searches the 4-parameter space $(\alpha_{\text{SC}}, \text{SC}_{\max}, k_{\text{damage}}, k_{\text{repair,SC}})$
  using differential evolution to test whether the two-process model can reproduce this dip:
  1. **Inside physiological priors (`SC_PRIOR_RANGES`):**
     The best fit leaves $72\%\text{–}80\%$ of the variance unexplained ($\text{SSE} \approx 0.089$).
     The optimizer pins every parameter to its extreme boundary ($\alpha_{\text{SC}} = 0.08$,
     $\text{SC}_{\max} = 0.40$, $k_{\text{damage}} = 0.15$).
  2. **The Mechanism Failure:**
     De Boer's trajectory requires an **accelerating** net response: near-zero net change through day 35,
     followed by a massive upward swing between days 35 and 63. However, both $\text{SC}_{\text{build}}$ and
     $\text{SC}_{\text{damage}}$ are **decelerating** first-order processes that approach steady state early.
     Under 3x/week dosing, $\text{SC}_{\text{damage}}$ reaches its dynamic steady state within ~10 days
     (a few half-lives of $k_{\text{repair,SC}}$) and cannot produce a delayed divergence at week 5.
  3. **Exact fit requires implausible parameters:**
     An unconstrained fit can match the three medians exactly ($\text{SSE} < 10^{-6}$), but only with
     extreme values: $\alpha_{\text{SC}} \approx 0.0035$, $\text{SC}_{\max} \approx 2.45$ (+245% hyperplasia),
     $k_{\text{damage}} \approx 0.021$, $k_{\text{repair,SC}} \approx 0.008\text{ d}^{-1}$ (repair half-life 87 days).
     These parameters contradict all biological literature.
  4. **Conclusion:**
     $k_{\text{damage}}$ is retained at **`0.0`** in `ModelParams`. The model maintains the additive
     mathematical capability to support acute barrier damage, but refuses to activate an unidentifiable,
     poorly fitting parameter.

---

### 2.5 `k_repair_SC` — barrier repair rate constant (1/day)

- **Symbol:** `ModelParams.k_repair_SC: float = np.log(2) / 7.0` ($\approx 0.099021\text{ d}^{-1}$, `vitd_model.py:1885`).
- **Prior range:** $[\ln(2)/14, \ln(2)/3]$ (half-life 3–14 days).
- **Status:** **[STRUCTURAL PRIOR, UNCITED]**.
- **Evidence Search & Biological Context:**
  An exhaustive literature search (Consensus 100-paper query, 2026-08-26) confirmed that human in vivo
  kinetic rate constants for UV-induced stratum corneum repair do not exist in the published literature:
  - *Abe & Mayuzumi (1979)* (*J Dermatol* 6:39-44): Transepidermal water loss (TEWL) after a single
    2 MED UVB dose returned to control levels by 3 weeks, but intermediate timepoints were insufficient
    to fit an exponential decay constant.
  - *Animal studies (Haratake 1997, Jiang 2007, Holleran 1997):* Mouse UVB barrier disruption shows
    delayed disruption peaking around day 4 and resolving by day 5–6, which is a complex biphasic
    cascade, not first-order exponential repair.
- **Role in the Model:**
  The 7-day half-life ($\ln(2)/7\text{ d}^{-1}$) is a structural choice ensuring that acute barrier damage
  recovers over a days-scale interval between weekly exposures. As shown in `basics/sc_only.py`,
  $(k_{\text{damage}}, k_{\text{repair,SC}})$ form an unidentifiable ridge. Because $k_{\text{damage}} = 0$,
  $k_{\text{repair,SC}}$ is dormant in the default model.

---

### 2.6 `K_SC` — stratum corneum synthesis attenuation coefficient

- **Symbol:** `ModelParams.K_SC: float = 1.0` (`vitd_model.py:1864`).
- **Prior range:** $[0.5, 2.0]$ (`SC_PRIOR_RANGES["K_SC"]`, `vitd_model.py:2039`).
- **Default value:** `1.0`.
- **Status:** **[DERIVED BOUND, UNCALIBRATED PRIOR]**.
- **Theoretical Derivation & Optical Bound:**
  Neither de Boer et al. (2025), Miyamura et al. (2011), nor Pearse et al. (1987) measured vitamin D
  photosynthesis. They measured epidermal thickness.
  To establish whether $K_{\text{SC}} = 1.0$ is physically plausible, an optical transmission bound was
  derived from published spectrophotometric transmission data of human stratum corneum (Bruls et al. 1984,
  *Photochem Photobiol* 40:485-494):
  - Bruls measured direct and forward-scattered UVB transmission through excised human stratum corneum
    (mean thickness ~12.5 $\mu$m).
  - Integrating the previtamin D action spectrum against solar UVB through thickened corneum yields an
    effective attenuation upper bound of $K_{\text{SC}} \le 1.509$ (in the limit of zero forward scatter
    redirection).
  - Shipped value $K_{\text{SC}} = 1.0$ sits at 66.3% of this theoretical bound, providing a fractional
    transmission of:
    $$\exp(-K_{\text{SC}} \cdot \text{SC}_{\max}) = \exp(-1.0 \times 0.30) = 0.741\ (74.1\%\text{ throughput})$$
  - Prior range $[0.5, 2.0]$ spans throughputs of $86.1\%$ down to $54.9\%$.
- **Non-separability from `K_ATTEN`:**
  In vivo, tanning ($\Delta M$) and corneum thickening ($\text{SC}$) develop simultaneously under
  repeated UV exposure. Both appear as exponential dampening factors on synthesis:
  $$\exp(-K_{\text{ATTEN}} \cdot \Delta M - K_{\text{SC}} \cdot \text{SC})$$
  Because no published human trial has measured 25(OH)D synthesis while independently controlling or
  measuring melanin and stratum corneum thickness simultaneously, $K_{\text{SC}}$ and $K_{\text{ATTEN}}$
  cannot be separated from in vivo data.

---

## 3. Downstream impact on cutaneous synthesis

Stratum corneum enters the primary synthesis equation purely as an exponential attenuation factor:

$$\Delta D_3 \propto \ln(1 + \text{sed} \cdot E_{\text{source}}) \cdot \exp(-K_{\text{const}} \cdot M_{\text{baseline}}) \cdot \exp(-K_{\text{ATTEN}} \cdot \Delta M) \cdot \exp(-K_{\text{SC}} \cdot \text{SC})$$

Comparing the three optical filtering terms at maximum steady-state adaptation:
1. **Constitutive melanin ($K_{\text{const}} \cdot M_{\text{baseline}}$):**
   Across FST I ($M=0.0381$) to FST VI ($M=1.00$), attenuation varies by $\exp(-1.06 \times (1.00 - 0.0381)) = 0.360$
   (a **$2.78\times$ reduction** in synthesis).
2. **Facultative tanning ($K_{\text{ATTEN}} \cdot \Delta M$):**
   At maximum tan headroom ($M_{\text{TAN\_HEADROOM}} = 0.238$, $K_{\text{ATTEN}} = 0.134$), attenuation is
   $\exp(-0.134 \times 0.238) = 0.969$ (a **$3.1\%$ reduction**).
3. **Stratum corneum thickening ($K_{\text{SC}} \cdot \text{SC}$):**
   At default maximum thickening ($\text{SC} = 0.30$, $K_{\text{SC}} = 1.0$), attenuation is
   $\exp(-1.0 \times 0.30) = 0.741$ (a **$25.9\%$ reduction**).

Stratum corneum thickening is an order of magnitude stronger than facultative tanning in its optical
impact on vitamin D synthesis, but both are secondary to constitutive melanin.

---

## 4. Mandatory limitations register

1. **The week-5 dip is statistically unverified within phototype strata:**
   The dip-and-overshoot phenomenon that motivates the two-process damage/repair structure is significant
   pooled ($p = 0.016$), but fails to achieve significance in either FP I–III ($p = 0.170$) or FP IV–VI ($p = 0.134$)
   individually (de Boer et al. 2025, Supp Table S1).
2. **Decelerating ODEs cannot fit accelerating data within physiological bounds:**
   Constant dosing produces decelerating relaxation in both $\text{SC}_{\text{build}}$ and $\text{SC}_{\text{damage}}$.
   Matching de Boer's trajectory inside evidence-based priors fails to fit $72\%\text{–}80\%$ of the response.
3. **$\alpha_{\text{SC}}$ and $\text{SC}_{\max}$ trade off along an unconstrained ridge:**
   Three group-median timepoints cannot resolve both the asymptote and the rate constant.
4. **$k_{\text{decay,SC}}$ is drawn from unstimulated transit times, not post-UV regression:**
   The 21–28 day half-life is an uncalibrated prior. Actual SC residence transit is 13.5–19.8 days under
   linear plug-flow desquamation in un-irradiated skin.
5. **$k_{\text{repair,SC}}$ has no human in vivo kinetic measurement:**
   The 7-day half-life is a structural prior chosen to recover barrier function between weekly sessions.
6. **$K_{\text{SC}}$ is uncalibrated against vitamin D data:**
   No study in the literature has co-measured 25(OH)D response and stratum corneum thickness.
7. **All turnover literature is all-Caucasian:**
   Weinstein & Van Scott (1965), Jansen et al. (1974), and Grove & Kligman (1983) enrolled light-skinned
   cohorts with no FST typing. Extrapolating their transit rates to FST IV–VI is unvalidated.
8. **SC-thickening magnitude may be source-spectrum dependent, not just FST/study dependent
   [ADDED 2026-09-23]:** Lavker et al. 1995, at matched biological dose (~0.5×MED), found SC
   thickening ranging from $+22.2\%$ (solar-spectrum SSR) to $+85.2\%$ (UVA-only) to $+35.8\%$
   (UVA-dominant SSRSS) within the same study, same subjects. `SC_max` is currently a single value
   per FST with no spectral axis, so this source-spectrum dependence is entirely unrepresented.

---

## 5. Scheduling leverage: what SC dynamics do to schedule ranking

A key question for this project is whether stratum corneum photoadaptation changes the optimal
inter-session interval.

In `scripts/melanin_sc_coupling.py` and the spacing sensitivity studies (`spacing_study/`):
- Ablating stratum corneum thickening entirely ($\text{SC} \equiv 0$) shifts steady-state 25(OH)D upward
  by a uniform multiplicative factor across all schedules.
- Because $\text{SC}$ accumulates slowly over 9 weeks, it attenuates all schedules proportionally once
  steady state is reached.
- The schedule ranking—dividing a UV budget into more frequent, smaller doses yields higher steady-state
  25(OH)D—is driven by the **mathematical concavity of the logarithmic synthesis function**, $\ln(1 + \text{sed})$,
  not by photoadaptation.
- Stratum corneum thickening dampens the absolute plateau height without altering schedule optimality.

---

## 6. Reproducing this document

To execute the underlying models, grid searches, and verification routines:

```bash
# Standalone SC model & de Boer 2025 two-process optimization:
PYTHONIOENCODING=utf-8 python3 basics/sc_only.py

# Hyperplasia fitting & parameter ridge inspection:
PYTHONIOENCODING=utf-8 python3 scripts/fit_sc_deboer2025.py

# Melanin and SC downstream coupling analysis:
PYTHONIOENCODING=utf-8 python3 scripts/melanin_sc_coupling.py
```
