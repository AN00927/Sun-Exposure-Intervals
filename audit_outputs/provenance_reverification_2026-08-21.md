# Provenance re-verification — 2026-08-21

Re-verifies the four `data_provenance.yaml` records whose `verified_by`
pointed at `PAPER_SUMMARIES.md` (gitignored, never committed, absent from
disk) [reference purged 2026-09-19]. All four primary sources were
located and read directly, by content, in `Research_Papers/`. Filenames were
misleading in two of four cases (see below) — confirming `CLAUDE.md`'s
standing warning not to trust filenames.

**Scope note.** Per the task's hard rules, this report records what the
primary sources say. It does **not** change any substantive value in
`data_provenance.yaml` — only the verification metadata (`verified_by`,
`verification_date`, `verification_status`) was updated in the YAML. Every
mismatch below is a finding for the user to act on, not something this pass
corrected.

---

## File identification

| Record | Filename found | Match confirmed by |
|---|---|---|
| Bogh2011 | `Research_Papers/Interdependence between body surface area and ultraviolet - bogh 2010.pdf` | Title, DOI (10.1111/j.1365-2133.2010.10082.x), BJD 164:163-169 all match exactly. Filename says "2010" (the DOI-registration year); the journal issue is 2011 — not an error, just an early-online-vs-print artifact. |
| Bogh2012 | `Research_Papers/Narrowband ultraviolet B three times per week is more effective in treating vitamin D deficiency than 1600 IU oral vitamin D3 per day, a randomized clinical trial..pdf` | DOI 10.1111/j.1365-2133.2012.11069.x, BJD 167:625-630 — exact match. **Filename gives no hint this is "Bogh2012"** — two other Bogh-adjacent-sounding filenames in the same folder ("Narrowband ultraviolet B course improves vitamin D.pdf", "...treatment improves vitamin D.pdf") are by a different author (Vähävihu) and were checked and ruled out. |
| Datta2017 | `Research_Papers/Half-life in 25(OH)D after UVB exposure depends on - datta 2017.pdf` | DOI 10.1039/C6PP00258G — exact match. **This is the RSC "Accepted Manuscript" version**, self-paginated 1–34, not the final typeset journal pagination (985–995). Content matches; exact final-print page numbers could not be independently confirmed from this file (flagged below, not a legibility problem — the text is perfectly readable, just differently paginated). |
| Young2020 | `Research_Papers/Melanin has a Small Inhibitory Effect on Cutaneous Vitamin D Synthesis, A Comparison of Extreme Phenotypes.pdf` | Title, DOI (10.1016/j.jid.2019.11.019), JID 140:1418-1426 — exact match. |

No legibility problems — all four PDFs are clean, searchable, born-digital text. No scans, no low-resolution figures.

---

## Bogh 2011 (BJD 164:163-169)

| Field | Status | Detail |
|---|---|---|
| `source_pdf`, `doi_or_pmid`, `sample_size` (92) | CONFIRMED | p.163–164 |
| `bsa_equation`, `intercept` (7.1), `baseline_coefficient` (−0.223), `dose_term_coefficient` (11.0), `bsa_term_coefficient` (5.5), `full_model_r_squared` (0.47) | CONFIRMED | Verbatim in-text equation, p.167: "Δ25(OH)D = 7.1−0.223×[baseline 25(OH)D]+11.0×ln(UVB dose)+5.5×ln(% body area), (P<0.0001; R²=0.47)" |
| `response_variable` | CONFIRMED | Δ25(OH)D in nmol/L throughout |
| `table_or_figure` = "Table 2 (regression model)" | **MISMATCH** | The combined regression equation above is **not** in a table — it is prose, under the heading "Estimation of 25-hydroxyvitamin-D₃ increase after ultraviolet B in a combined model," p.167. **Table 2** (p.166) is a *different* object: the raw mean Δ25(OH)D grid by UVB dose × BSA (0.75/1.5/3.0 SED × 6/12/24%), used elsewhere in the model for validation checks, not for the regression coefficients. |
| `page: 166` | **MISMATCH** | The regression equation is on **p.167**, not p.166. (Table 2, which *is* on p.166, is the unrelated dose×BSA grid above.) |
| `baseline_coefficient_p: 0.000024` and `baseline_r_squared: 0.313` | **MISMATCH (mislabeled provenance)** | These two numbers are correctly transcribed from the paper (p.165) but belong to a **different regression, from a different, earlier paper** — Bogh et al. 2010, *J Invest Dermatol* 130:546-53 ("our previous study"), cited as reference 16: "Δ25(OH)D = 31.4−0.223×[baseline 25(OH)D] (P=0.000024; R²=0.313)." The current (2011, n=92) paper reused that same −0.223 coefficient rather than refitting it, and gives **its own** univariate baseline correlation elsewhere (p.165, Results): "negatively correlated with baseline 25(OH)D level (P<0.0001; **R²=0.18**)" — a different R² from 0.313. Neither the current paper's own univariate stat (R²=0.18) nor the combined model's overall stat (P<0.0001, R²=0.47) is P=0.000024/R²=0.313 — that pairing is specific to the 2010 paper's univariate fit. **Net effect: the −0.223 coefficient value itself is right, but the p-value/R² attached to it in the record belong to a different study's regression, not this one's.** |

**Verdict: PARTIAL.** Coefficients and R² of the combined model are all correct; the citation (table/page) and the baseline-term statistics need correction.

---

## Bogh 2012 (BJD 167:625-630)

| Field | Status | Detail |
|---|---|---|
| `source_pdf`, `doi_or_pmid` | CONFIRMED | p.625 |
| `sample_size`: "34 NB-UVB, 39 oral D3 randomized" | CONFIRMED | Results, p.627: "originally 34 and 39 individuals randomized to the NB-UVB and oral vitamin D3 treated groups, respectively" |
| `baseline_mean` (19.2), `baseline_sd` (6), `final_mean` (75), `final_sd` (16.8), `delta_mean` (55.8, derived) | CONFIRMED | Table 1, p.628: NB-UVB "19.2 ± 6 → 75 ± 16.8"; also stated in Results text, p.627 |
| `dose_start` (0.36), `dose_cap` (1.20), `dose_escalation_rule`, `dose_cumulative_mean` (9.0) | CONFIRMED | p.627 |
| `assay_method` (LC-MS/MS), `bsa_exposed_fraction`/`definition` (full body), `lamp` (TL-01) | CONFIRMED | p.626 (Methods) |
| `latitude`: "56°N (Sweden)" | CONFIRMED | p.626: "Lund University, Malmö Hospital, Sweden (56° N)" |
| `table_or_figure`: "Table 1 (main results)", `page: 627` | **MISMATCH (page)** | Table 1 itself is printed on **p.628**, not 627. The supporting Results narrative (same numbers, in prose) is on p.627. Minor, but the page pointer as written sends a reader to the wrong page for the table. |
| `season`: "Feb-May 2011 (winter-spring, avoiding summer)" | **MISMATCH** | Methods, p.626: "This clinical, open-label, randomized trial was conducted at Lund University, Malmö Hospital, Sweden (56° N), **from February 2010 to May 2011**, avoiding the summer period (June–September)." The record's "Feb-May 2011" describes only the tail end of a **15-month enrollment window spanning two winters** (Feb 2010–May 2011), not a single Feb–May span. This matters for the "avoiding summer" framing — the study spanned two non-summer seasons, not one. |
| `delta_sd`: NEEDS_VERIFICATION | Confirmed still NOT_FOUND | Paper reports baseline SD and final SD separately (Table 1) but never a paired-difference SD. Not derivable from the published summary statistics alone. Correctly flagged as unverified in the record. |
| `fst_distribution`: all NEEDS_VERIFICATION, note "exact FST counts not reported" | **MISMATCH — data exists** | Table 1, p.628, row "Skin type (I/II/III/IV/V/VI)" **does** give exact counts per arm at baseline: NB-UVB arm 0/5/2/7/0/2 (n=16), oral-D3 arm 0/4/5/3/2/2 (n=16). The record's note that exact FST counts "are not reported" is incorrect — they are in Table 1, just not yet transcribed into the record. (Per this task's scope, I have not written these counts into the YAML — flagging here so the user can decide whether to add them.) |

**Verdict: PARTIAL**, trending toward FULL on the numbers that matter most (baseline/final/delta, dose schedule) but with a real season-window mismatch and an incorrectly-labeled "not reported" claim on FST counts that are, in fact, available.

---

## Datta 2017 (Photochem. Photobiol. Sci. 16:985-995; verified from the RSC Accepted Manuscript, DOI 10.1039/C6PP00258G)

| Field | Status | Detail |
|---|---|---|
| `sample_size` (22 high-level; +14 medium, +92 low "from prior studies") | CONFIRMED | Abstract; Experimental p.14–16 of the manuscript PDF. Medium/low groups are explicitly *external* — "two other accessible participant groups, collected and investigated elsewhere" (refs 18, 19, 27) — the record's framing is accurate. |
| `half_life_parameterization`, `baseline_dependence` | CONFIRMED | Results p.4–5, Statistics p.18 (formula reproduced verbatim) |
| `decay_constant_k: -0.0078` | CONFIRMED (exact, not approximate) | **Table 4** (manuscript p.29): "Common decrease with personal start levels" row lists k = **−0.0078** directly — this is a printed value, not one I had to derive from the "0.78%/day" text. |
| `reported_half_lives.high_start_level`: mean 89, range "59–179 days", start_level_mean 132, start_level_sd 31, end_level_7wk 92, mean_decrease_per_day_pct 0.78, decrease_range_pct "0.2–2.4" | CONFIRMED, all fields | mean 89 days: Results p.5, Table 4/5, Fig.3. start level 132±31 (68–216): **Table 1**, manuscript p.26 ("25(OH)D start level – nmol l⁻¹: 132 ± 31 (68-216)"). end level 92: p.4, Table 2. decrease 0.78%/day, range 0.2–2.4%: p.5. The "range 59–179 days" is the range across the four gender×genotype subgroups in Table 4 (not a raw range over all 22 individuals) — worth the record being explicit about that, but the numbers themselves are right. |
| `medium_start_level`: n=14, mean_half_life=148, start_level_mean=65 | CONFIRMED | n=14, start level 64.8≈65 nmol/L: p.15 (Experimental). Half-life 148 days: main text p.9, **Table 5** and **Fig.3** caption (manuscript p.30, p.33) all say 148. |
| `low_start_level`: n=92, mean_half_life=199, start_level_mean=43 | CONFIRMED | n=92, start level 42.7≈43: p.16. Half-life 199: p.9, Table 5, Fig.3. |
| `gender_genotype_effects.model_r_squared` (0.878), `model_p` ("2.0×10⁻⁴⁸") | CONFIRMED | p.7: "R²=0.878, P=2.0×10⁻⁴⁸" |
| `shortest_half_life` (Male GG, 59d), `longest_half_life` (Female AA/AG, 179d) | CONFIRMED | Table 4 (manuscript p.29): Male/GG k=−0.0120→59d; Female/AA-AG k=−0.0039→179d |
| `max_interpersonal_difference`: "114 nmol/L (7 vs 121 nmol/L)" | CONFIRMED, exact | p.5: "The maximal inter-personal difference of the decrease in 25(OH)D was 114 nmol l⁻¹ (the difference between 7 nmol l⁻¹ and 121 nmol l⁻¹)." |
| `table_or_figure`: "Abstract + Table 2 (half-life by subgroup)" | **MISMATCH** | The paper's actual **Table 2** is "Time course of 25(OH)D and PTH" — a 5-timepoint (day 0/7/21/34/49) descriptive table, not a half-life-by-subgroup table. The subgroup half-life data the record actually draws on live in **Table 4** (gender × rs2228570 combinations) and **Table 5 / Fig. 3** (high/medium/low group comparison). The record's table citation points at the wrong table. |
| `page: 987-989` | **UNVERIFIABLE from this file, not a mismatch** | This PDF is the RSC Accepted-Manuscript pre-print (self-paginated 1–34), not the final typeset article (985–995 in the printed journal). The content is complete and matches, but I cannot confirm final print-page correspondence from this file alone. |
| `unverified_fields`: "Full experimental protocol details (abstract only available)" | **MISMATCH — full text was available** | The file in `Research_Papers/` is the complete 34-page accepted manuscript (Introduction, full Results, full Experimental/Methods, Tables 1–5, Fig. 1–3, References), not an abstract. Every field previously marked "NEEDS_VERIFICATION" or "unverified" for lack of full text is resolved above. This record's original claim that only the abstract was available is out of date — the primary source was already sitting in `Research_Papers/`. |
| Abstract vs. main text: "149 days" vs "148 days" | **Flag — internal inconsistency in the paper itself, not in the record** | The Abstract (manuscript p.2) reads "...longer half-lives of 149 days and 199 days, respectively." Every other mention of the medium-group half-life in the paper — main text (p.9), Table 5 (p.30), Fig. 3 caption (p.33) — says **148 days**. This looks like a typo in the Abstract of the accepted manuscript (single occurrence, contradicted by four consistent occurrences elsewhere). The provenance record's value (148) matches the dominant, repeated figure, so this is not a record error — flagging it because a reader who checks only the abstract will see a different number. |

**Verdict: Effectively FULL on substance** — every quantitative field checked against the primary source and confirmed. Two citation-hygiene issues remain (wrong table number cited; page range unconfirmed against final typeset pagination), so status is recorded as PARTIAL rather than FULL, per the instruction to only mark FULL if every field, including citation fields, checks out.

---

## Young 2020 (JID 140:1418-1426)

| Field | Status | Detail |
|---|---|---|
| `fst_ii_to_vi_ratio_low` (1.29), `_low_source` ("Arimed B, 2 SED/exposure, 10 SED cumulative") | CONFIRMED | Ratio: p.1419 ("The slope ratio of FST II versus VI is 1.29:1"). Dose: p.1424 Methods, "The Arimed B and TL01 exposures were 2 and 1.5 SED, respectively, based on the highest suberythemal doses for FST II, producing cumulative doses of 10 and 7.5 SED" — matches verbatim. |
| `fst_ii_to_vi_ratio_high` (1.43), `_high_source` ("TL01, 1.5 SED/exposure, 7.5 SED cumulative") | CONFIRMED | Ratio and P=8.33×10⁻⁹: p.1421 text and Table 3 (p.1423). Dose: same Methods sentence as above. |
| `melanin_inhibition_factor`: "1.3-1.4" | CONFIRMED | Abstract |
| `bsa_exposed`: "85% BSA" | CONFIRMED | Abstract, Methods |
| `exposures`: "5 UVR exposures at 3-4 day intervals" | CONFIRMED | Abstract |
| `ethnicity_ratios`: white_european_vs_ISC 1.33, white_european_vs_African 1.18 | CONFIRMED, exact | p.1420: "The ratio between the slopes for white European and ISC (the least steep) was 1.33:1 and for Af 1.18:1." |
| `sample_size`: "102 (44 Arimed B irradiated + 20 control; **38** TL01 irradiated + **10** control)" | **MISMATCH, clear and countable** | **Table 1** (p.1420) gives exact per-arm, per-FST counts. Arimed B irradiated: FST II 19 + III 9 + IV 5 + V 5 + VI 6 = **44** (matches record). Arimed B control: 4+6+7+1+2 = **20** (matches record). **TL01 irradiated: FST II 12 + FST VI 11 = 23, not 38.** **TL01 control: FST II 8 + FST VI 7 = 15, not 10.** Check: 44+20+23+15 = 102, matching the paper's stated total (n=102) exactly — so the corrected TL01 figures (23 irradiated, 15 control) are the ones consistent with the paper's own total; the record's 38/10 figures are not (44+20+38+10 = 112 ≠ 102). |
| `fst_ii_slope`: "Reference (not numerically reported; used as comparator)"; `fst_vi_slope`: "...(P values); absolute slopes not reported" (per `unverified_fields`) | **MISMATCH — absolute slopes are reported** | **Table 3** (p.1423), "Slopes of Linear Regressions (SED versus 25(OH)D₃)," gives absolute slope values with 95% CI and p-values for every FST group under both spectra — e.g. Arimed B: FST II = 4.990 (95% CI 4.185–5.794), FST VI = 3.856 (95% CI 3.026–4.685); TL01: FST II(WE) = 7.879, FST VI(Af) = 5.527. These are in nmol/L per SED, exactly the unit the record's `unverified_fields` note says is unavailable. (4.990/3.856 = 1.29 and 7.879/5.527 = 1.43, which is where the ratios in the record actually come from — so the ratios were correctly computed *from* these slopes, but the record then describes the underlying slopes themselves as unreported, which is incorrect.) |
| `ita_correlation`: "r = -0.903 (protected), -0.902 (exposed), P=4.47×10⁻²²" | **PARTIAL — r-values right, one p-value missing/misleading** | p.1419: "Spearman Rank correlation ... exposed r=−0.902, P=4.47×10⁻²² and protected r=−0.903, P=3.42×10⁻²²." Both r-values match exactly. But the paper reports **two distinct p-values**, one per correlation (exposed: 4.47×10⁻²²; protected: 3.42×10⁻²²) — the record states only one p-value without attribution, which as written reads as if it applies to both, or to whichever r is listed first (protected). It is actually the exposed correlation's p-value. Note there is also a *second*, unrelated correlation in this paper (Fig. 3: FST-vs-ITA, not 25(OH)D-vs-ITA) with yet another pair of values (exposed R=−0.886, P=5.53×10⁻¹⁷; unexposed R=−0.881, P=1.44×10⁻¹⁶) — the record correctly picked the FST-independent, dose-response-relevant correlation and not this one, which is good, but worth noting the paper contains two similarly-worded correlations that could be conflated. |

**Verdict: PARTIAL.** The two headline ratios (1.29, 1.43) that feed `SOURCE_CALIBRATION_TABLE`-adjacent reasoning are solidly confirmed, exact-figure and exact-p-value. But the sample-size breakdown for the TL01 arm is numerically wrong by a wide margin (38/10 claimed vs. 23/15 actual, confirmed two independent ways), and the record's characterization of absolute slopes as "not reported" is incorrect — Table 3 reports them directly.

---

## Summary — mismatch count by study

| Study | Fields confirmed | Fields mismatched | Fields not found |
|---|---:|---:|---:|
| Bogh 2011 | 7 | 3 (table/page citation; baseline p/R² mislabeled) | 0 |
| Bogh 2012 | 12 | 3 (page citation; season window; FST "not reported" claim) | 1 (delta_sd, genuinely absent from paper) |
| Datta 2017 | 15 | 2 (table citation; "abstract only" claim) | 0 (page range unconfirmable due to pre-print pagination, not absent) |
| Young 2020 | 6 | 3 (sample size TL01 arm; "slopes not reported" claim; ITA p-value attribution) | 0 |

**No legibility problems.** All four PDFs are clean, digitally-typeset, fully readable — no scans, no illegible tables, no ambiguous gridlines worth flagging under the `CLAUDE.md` legibility rule.

**Most consequential single finding:** Young 2020's TL01 arm sample size (23 irradiated + 15 control, not the recorded 38 + 10) is wrong by a factor large enough that it should be checked against wherever `n` from this study is used downstream (e.g., any power/confidence-weighting logic keyed to per-arm sample size).

**Second most consequential:** the Bogh 2011 combined-model equation is correctly transcribed, but two of its accompanying statistics (P=0.000024, R²=0.313) actually belong to a citation of a *different, earlier* Bogh paper — anyone reading the record as "this is what the 92-participant BJD-2011 study found for its baseline term" would be describing a number from a different dataset.
