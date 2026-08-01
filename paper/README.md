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
  rank-identical or rank-inverted to each other.
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
  the current validation count (13/14 required checks, regenerable with
  `run_all.py`) and the held-out benchmark result, but the full numerical
  method, schedule-generator, and reproducibility write-up is not yet
  written.
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
2. **Communications Medicine 2025 (PMC12894941), the biphasic
   25-hydroxylation source.** Full author list, article title, volume, and
   pages are not recorded anywhere in the repository; only the PMC identifier
   is available.
3. **Elwood & Jopson 1997.** Cited only by author, year, and the specific
   odds ratios attributed to it (intermittent OR 1.71, occupational OR 0.86,
   sunburn OR 1.91). Journal, volume, and pages are not recorded.
4. **Leiter 2009** and **Calzavara-Pinton 2015.** Both cited only as
   author-year fragments throughout the repository. No further bibliographic
   detail is available.
5. **Eadie 2021, Armstrong & Kricker 2001, and a 2023 *PLOS ONE* melanoma
   study.** All three are named in `MODEL_AND_EVIDENCE.md` as papers the project wants
   for a quantitative cancer-risk model and does **not** have. None is in
   `Papers_and_Datasets/`.
6. **Matsuoka et al. 1990 (*J Lab Clin Med* 116:87-90, PMID 2165512), full
   text.** Only the abstract is in the repository. It is sufficient to derive
   $K_{\mathrm{ATTEN}}$ (see `vitd_calibration.py`), but individual-level
   data, exact UVB dose, and lamp spectrum are still absent, and no
   independent replication of this within-subject result exists anywhere in
   the published literature located so far.
7. **van Dijk et al. 2016**, the previtamin-D3 photodegradation source. Full
   author list, volume, and pages are not recorded; the repository has only
   the paper's title and journal name. Its abstract reports a sixfold spread
   (7.5 SED vs. 1.25 SED for two different measured spectra) that has never
   been reconciled, and the model uses the larger value without justifying
   the choice over the smaller one.
8. **Holick et al. 1981** (*Science* 211:590-593), the original previtamin-D3
   plateau measurement. Author list beyond "Holick" is not recorded.
9. **Cashman 2008/2011**, the dietary vitamin D dose-response source. Cited
   only as a year range, which suggests two different papers, not one; no
   journal, volume, or pages recorded for either.
10. **Drincic 2012** (*Obesity* 20:1444-1448), the body-weight dilution
    source. Full author list beyond "Drincic" is not recorded.
11. **Ekwaru 2014, Walsh 2017, Pannu 2016.** Cited only as author-year
    fragments supporting the body-weight dilution mechanism. No further
    detail recorded.
12. **Cockell 2001, Parisi 2000, Parisi 2003, Thieden 2000.** All four
    support the personal-dose geometry factor (ambient-to-personal UV
    exposure ratio). Cited only as author-year-value fragments; no journal,
    volume, or page data recorded for any of them. Note that "Thieden 2000"
    is a different paper from Thieden et al. 2008 (`thieden2008` in
    `references.bib`), which does have a full citation; do not merge them.
13. **Schmalwieser et al. 2018** (*Photochemistry and Photobiology*), the
    source recommending an elevation-dependent exposure-geometry function.
    Volume and pages are not recorded.
14. **Rodríguez-Granados 2017**, the source of the TL-01 J/cm² to erythemal
    SED conversion factor (5.5, cabinet-specific to a Waldmann UV7002).
    Journal, volume, pages, and full author list are not recorded anywhere in
    the repository, only the measured irradiance values themselves.
15. **Miyamura 2011**, named in `MODEL_AND_EVIDENCE.md` gap 7 as an early source for
    the melanin-induction rate parameter. No further bibliographic detail
    recorded.
16. **Rhodes 2017** ("P22" in `vitd_model.py`), the CPD/DNA-damage
    co-occurrence source distinct from Shih et al. 2018 ("P24," which has a
    complete citation). Not confirmed whether P22 and P24 are the same
    author group's separate papers or how they relate; full detail not
    recorded.
17. **Wilkes 2015, Alaluf 2001, Neville 2020, Markiewicz & Idowu 2020,
    Borecka 2024, Chalcraft 2020, Libon 2013, Armas 2007.** Full citations for
    all eight appear in `references.bib`, transcribed from
    `MODEL_AND_EVIDENCE.md`'s "[New]" evidence cards, but none has been
    independently cross-checked against the publisher's own record (DOI,
    exact page range). Treat the page ranges and volume numbers for these
    eight as provisional pending that check.
18. **O'Sullivan 2026**, the Ontario occupational-cohort melanoma study cited
    in `vitd_physiology.py`. Journal, volume, and pages not recorded.
19. **Armstrong 1988**, cited alongside Elwood & Jopson 1997 and Leiter 2009
    as supporting evidence for the intermittent-vs-cumulative cancer-risk
    structure. Full bibliographic detail not recorded.

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
