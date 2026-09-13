# paper/

Elsevier `elsarticle` manuscript skeleton for the vitamin D scheduling model in
this repository. This directory is self-contained: nothing here modifies any
file outside `paper/`.

## Build

Requires a TeX distribution with the `elsarticle` class (part of TeX Live and
MiKTeX by default).

```bash
cd paper
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Three `pdflatex` passes are needed: the first writes the citation keys,
`bibtex` resolves them against `references.bib`, and the second and third
passes settle cross-references (`\ref{}` to sections and tables) and the
bibliography. `manuscript.pdf` and the compile logs are build products; do not
commit them without checking with the author first.

The `longtable` package spans the parameter table across as many pages as it
needs. If a LaTeX engine other than `pdflatex` is used (`xelatex`,
`lualatex`), the same four-pass sequence applies.

**Build status: not compiled.** No `pdflatex` or `bibtex` binary was found on
the machine this was drafted on, so the sequence above has not actually been
run. What was checked instead, by static inspection of the source: brace
balance, `$...$` math-mode delimiter parity, `\begin`/`\end` environment
pairing, that every `\cite*{}` key resolves to an entry in `references.bib`
and vice versa, and that no literal `%` character sits where it would silently
truncate a line (one such bug was found and fixed, in the `\address{}` field).
That is not a substitute for actually compiling it. Run the four-pass sequence
above and fix whatever it reports before submission; `longtable` column widths
in particular (Table 1 uses six columns at fixed `p{}` widths) may need
adjusting once you can see actual page breaks.

## What is drafted vs. stubbed

Fully drafted, from the repository's own documentation and code:

- **Introduction** (`\section{Introduction}`): deficiency burden, why
  scheduling matters independent of total dose, and what single-exposure
  dose-response models miss.
- **Background** (`\section{Background}`, six subsections): cutaneous
  photosynthesis and dose-response form; melanin as an optical filter;
  facultative photoadaptation (tanning and stratum corneum change);
  homeostatic feedback and catabolism; dose framing (absolute SED vs.
  fractional MED); and UV exposure vs. skin cancer risk as the competing,
  deliberately deprioritized objective.
- **Model structure and parameters** (`\section{Model structure and
  parameters}`): the governing equations, the three decision-framed
  objectives, and two full parameter tables (Table 1: model parameters;
  Table 2: per-source dose-weighting priors), both transcribed directly from
  `vitd_model.py`, `vitd_physiology.py`, and `vitd_objectives.py` on
  **2026-07-31**. Re-transcribe from the code, not from this table, if the
  parameters change; do not hand-edit values here.
- **Limitations and assumptions** (`\section{Limitations and assumptions}`):
  the full register of provisional results, cross-referenced to
  `MODEL_AND_EVIDENCE.md` in the repository root.

### Objective-choice correction (read this before quoting an "optimal interval")

A first draft of this manuscript repeated this project's own earlier claim
that `net_final_D_change_per_SED` gives a genuine interior optimum near three
days. Three new diagnostic scripts in `scripts/` (not present when
`MODEL_AND_EVIDENCE.md` was last written) show that claim needs correcting, and the
manuscript now reflects the correction in the abstract, Introduction,
Section 3.2 (Objectives), and Section 7 (Limitations):

- `scripts/metric_redundancy.py` shows `gross_synth_per_SED` ranks schedules
  in near-exact opposition to mean serum 25(OH)D and to avoiding
  time-integrated deficiency (Spearman rho about -0.999 against both), and
  that several of the project's eight scoring functions are exactly
  rank-identical or rank-inverted to each other. Both `gross_synth_per_SED`
  and `net_final_D_change_per_SED` are now formally **DIAGNOSTIC tier** in
  `vitd_objectives.OBJECTIVE_HIERARCHY` — background metrics, barred in code
  from being reported as a recommendation.
- `scripts/parameter_optima.py` shows that `net_final_D_change_per_SED`'s
  three-day optimum is itself a per-SED ratio artifact: under a fixed
  cumulative UV budget (the framing immune to that artifact), the optimal
  interval is one day, the finest spacing tested, for all fourteen limiting
  parameters checked across their full documented prior ranges, and in a
  joint 200-draw Monte Carlo.
- `scripts/melanin_sc_coupling.py` shows melanin attenuation changes mean
  25(OH)D by about 0.16 nmol/L at a realistic dose (1 SED every three days,
  skin type V), and that constitutive pigmentation and stratum corneum, not
  facultative tanning, account for essentially all of the model's skin-type
  gap.

**The manuscript does not recommend a specific inter-session interval
anywhere**, and states plainly that no objective checked so far supports one.
If a future revision reintroduces an interval recommendation, it should cite
which framing produced it and note that the fixed-budget framing (the one
immune to ratio artifacts) has so far always returned "as frequent as
possible."

Stubbed, marked with a visible red `[TODO verify: ...]` marker in the compiled
PDF rather than left silently blank:

- **Simulation setup and validation status**: a placeholder paragraph states
  the validation count and the held-out benchmark result, but the full
  numerical method, schedule-generator, and reproducibility write-up is not
  yet written. **Do not hardcode the count when writing this up** — regenerate
  with `run_all.py`. It changed on 2026-08-15 when a second held-out benchmark
  (Gallagher 2012 oral plateau, warning-only) was added, which is exactly how
  the previously written "13/14" went stale.
- **Results**: no tables or figures yet. Quantitative output exists in the
  repository's generated `RESULTS.md`; selecting and writing up specific
  results for the paper was out of scope for this task.
- **Discussion**: not written.
- **Conclusion**: a one-paragraph placeholder only.
- **End matter**: CRediT authorship statement, declaration of competing
  interest, data availability (repository URL/DOI), and acknowledgments are
  all placeholders for the author to fill in before submission.
- **Author affiliation**: `Ahaan`, no institution assigned yet.

## Every claim that still needs a source

This list matches the `\todo{...}` markers embedded in `manuscript.tex` (and a
handful of `references.bib` entries with `%TODO verify` comments that support
them). Each line names the claim and what is missing.

1. **$D_{\max} = 150$ nmol/L origin.** No primary source in this repository
   supports this value. An earlier project revision attributed it to Bogh
   et al. 2010, whose own reported ceiling was 116 nmol/L, and a PMID once
   associated with it belongs to an unrelated obesity paper (Doumatey et al.,
   *Obesity* 2010). This is the model's single highest-value open item: it
   drives the interval-ranking result more than any pigmentation parameter.
2. ~~Communications Medicine 2025 (PMC12894941)~~ **Resolved 2026-09-13.** Full publisher record verified: Scholze et al., *Communications Medicine* 6(1):105, 2026 (doi:10.1038/s43856-025-01368-7).
3. ~~Elwood & Jopson 1997~~ **Resolved 2026-09-13.** Full publisher record verified: *International Journal of Cancer* 73(2):198--203, 1997 (doi:10.1002/(sici)1097-0215(19971009)73:2<198::aid-ijc6>3.0.co;2-r).
4. ~~Leiter 2009 and Calzavara-Pinton 2015~~ **Resolved.** Full records verified: Leiter & Garbe 2008 (*Adv Exp Med Biol* 624:89-103) and Calzavara-Pinton et al. 2015 (*G Ital Dermatol Venereol* 150(4):369-378).
5. **Eadie 2021, Armstrong & Kricker 2001, and a 2023 *PLOS ONE* melanoma
   study.** Armstrong & Kricker verified (*J Photochem Photobiol B* 63:8-18, 2001). Eadie 2021 and PLOS ONE 2023 remain named as future quantitative risk model requests.
6. **Matsuoka et al. 1990 (*J Lab Clin Med* 116:87-90, PMID 2165512), full
   text.** Only the abstract is in the repository. It is sufficient to derive
   $K_{\mathrm{ATTEN}}$ (see `vitd_calibration.py`), but individual-level
   data, exact UVB dose, and lamp spectrum are still absent, and no
   independent replication of this within-subject result exists anywhere in
   the published literature located so far. Note: Matsuoka 1990 JAAD (sunscreen/body-region) is verified at *JAAD* 22(5 Pt 1):772-775.
7. ~~van Dijk et al. 2016~~ **Resolved 2026-09-13.** Full publisher record verified: *Photochem. Photobiol. Sci.* 15(7):896--909, 2016 (doi:10.1039/c6pp00034g).
8. ~~Holick et al. 1981~~ **Resolved 2026-09-13.** Full publisher record verified: *Science* 211(4482):590--593, 1981 (doi:10.1126/science.6256855).
9. **Cashman 2008/2011**, the dietary vitamin D dose-response source. Cashman 2008 verified (*Am J Clin Nutr* 87(3):673-681).
10. ~~Drincic 2012~~ **Resolved 2026-09-13.** Full publisher record verified: *Obesity* 20(7):1444--1448, 2012 (doi:10.1038/oby.2011.404).
11. ~~Ekwaru 2014, Walsh 2017, Pannu 2016~~ **Resolved 2026-09-13.** Full publisher records verified: Ekwaru et al. 2014 (*PLOS ONE* 9(11):e111265), Walsh et al. 2017 (*Curr Opin Endocrinol Diabetes Obes* 24(6):389-394), and Pannu et al. 2016 (*Nutr Res* 36(3):201-213).
12. ~~Cockell 2001, Parisi 2000, Parisi 2003, Thieden 2000~~ **Resolved 2026-09-13.** Full publisher records verified: Cockell et al. 2001 (*J Photochem Photobiol B* 60:85-92), Parisi et al. 2000 (*Phys Med Biol* 45(8):2253-2262), Lester/Parisi et al. 2003 (*Phys Med Biol* 48(22):3685-3698), and Thieden et al. 2000 (*Photodermatol Photoimmunol Photomed* 16(2):57-61).
13. ~~Schmalwieser et al. 2018~~ **Resolved 2026-09-13.** Full publisher record verified: *Photochem Photobiol* 94(5):900-915 (doi:10.1111/php.12946).
14. ~~Rodríguez-Granados 2017~~ **Resolved 2026-09-13.** Full publisher record verified from PDF in `Research_Papers/`: *Actas Dermosifiliogr* 108(9):852-858 (doi:10.1016/j.ad.2017.05.009).
15. ~~Miyamura 2011~~ **Resolved 2026-08-25.** Full bibliographic detail is now
    in `paper/references.bib` (`miyamura2011`; doi:10.1111/j.1755-148X.2010.00764.x,
    PMID 20979596, PMC3021652), and the PDF is on disk in `Research_Papers/`.
16. **Rhodes 2017** ("P22" in `vitd_model.py`), the CPD/DNA-damage
    co-occurrence source distinct from Shih et al. 2018 ("P24," which has a
    complete citation). Kept as separate entry per repository numbering.
17. ~~Wilkes 2015, Alaluf 2001, Neville 2020, Markiewicz & Idowu 2020,
    Borecka 2024, Chalcraft 2020, Libon 2013, Armas 2007~~ **Resolved 2026-09-13.** All eight independently cross-checked against publisher records (DOIs, page ranges, volume/issue, and full author lists added to `references.bib`).
18. ~~O'Sullivan 2026~~ **Resolved 2026-09-13.** Full publisher record verified: *Scientific Reports* 16(1):48477, 2026 (doi:10.1038/s41598-026-48477-4, PMID 41963456).
19. ~~Armstrong 1988~~ **Resolved 2026-09-13.** Full publisher record verified: *J Dermatol Surg Oncol* 14(8):835-849, 1988 (doi:10.1111/j.1524-4725.1988.tb03588.x, PMID 3294982).

If a primary source is found for any of the above, update the corresponding
`references.bib` entry directly (fill in the missing fields, delete the
`%TODO verify` comment above it) and remove the matching `\todo{...}` marker
in `manuscript.tex`. Do not fill in a volume or page number from memory or
inference; an incomplete citation with a visible TODO is better than a wrong
one that looks complete.

## Bibliography entries not currently cited in the text

`references.bib` holds 71 entries; 64 are cited somewhere in `manuscript.tex`.
Seven are verified from the same evidence cards in `MODEL_AND_EVIDENCE.md` as
everything else here, but ended up not woven into this drafted subset of
sections: `alahouhala2014`, `assi2022`, `keurentjes2021`, `lehoang2016`,
`rhodes2017`, `thieden2000`, and `vahavihu2010b`. None is fabricated; all seven
were left out only because Section 4 (Results) and Section 6 (Discussion) are
stubs that would have been their natural home. Cite them from there in the
next revision, or delete the entries if they turn out not to be needed.

## A note on the parameter table

Every value in Table 1 was read out of `vitd_model.py` on 2026-07-31, not
retyped from `MODEL_AND_EVIDENCE.md` or this README. If those
values drift apart in the future, `vitd_model.py` is correct by the
repository's own rule (see `CLAUDE.md`): parameter values live in exactly one
place in the source code, and every document that quotes one is a copy, not a
source.
