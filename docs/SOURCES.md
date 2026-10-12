# Source acquisition status

Canonical record of which papers this repo has chased, obtained, misfiled,
or ruled out -- so nobody re-searches for something already found (or
re-trusts a filename already proven wrong). Supersedes the acquisition
content of the retired post-manuscript workstream memos (the citation
hunts previously tracked under workstream IDs W-D, W-L, W-T, W-U, W-AB,
now off disk).

## Obtained and read in full

| Paper | Identity | Obtained | Key finding |
|---|---|---|---|
| Felton et al. 2020 | Photochem Photobiol Sci 19:810-818, DOI 10.1039/c9pp00452a, PMID 33856672 | 2026-09-21 | SC sign question: edema/spongiosis applies to viable epidermis under ACUTE challenge only; chronic SC thickening (+41.6%, FST II) attributed to proliferation. See `vitd_model.py`'s `K_SC` entry / `calibration_approval.yaml`. |
| Lavker et al. 1995 | J Am Acad Dermatol 32:53-62, DOI 10.1016/0190-9622(95)90184-1, PMID 7822517 | 2026-09-21 | Solar-spectrum (SSR) arm: SC +22.2%, viable epidermis n.s. -- corroborates Felton/de Boer. UVA-only arm: SC +85.2% -- spectral dependence, not previously documented. FST I-III only. |
| Rhodes et al. 2010 | J Invest Dermatol 130(5):1411-1418, DOI 10.1038/jid.2009.417, PMID 20072137 | 2026-09-21 | 109 subjects FST I-IV, 0% FST V/VI, skin type not a covariate in its own regression -- CANNOT supply a fifth K_const route (corrects a standing repo expectation that it would). |
| Fajuyigbe et al. 2018 (Photochem Photobiol) | DOI 10.1111/php.12947, PMID 29869793 -- NOT the same as the existing `fajuyigbe2018` FASEB J entry (PMID 29394104); this repo uses the distinct bib key `fajuyigbe2018photochem` for this one (`paper/references.bib`) | 2026-09-21 | 8.4x erythema-response ratio (white FST I/II, n=9, vs black FST VI, n=11), no stated CI; dividing the paper's own printed slopes (k(I/II)=-0.0258, corrected k(VI)~=0.0034 /SED) gives 7.588..., not 8.4 -- both figures carried, discrepancy flagged as rounding of the displayed 3-sig-fig slopes, neither "corrected". Erythema dose-response, not vitamin-D synthesis -- must never enter K_const. No b* published, so no ITA can be computed from it -- no FST VI resolving power. |
| Stamatas et al. 2008 | Br J Dermatol 159:683-690, DOI 10.1111/j.1365-2133.2008.08642.x | 2026-09-21 | Method-validation paper, discontinued instrument (Minolta CM-2002), no FST classification, no vitamin-D endpoint. Not usable for any model parameter. |

## Confirmed genuinely absent (full-text content scan, not filename)

| Paper | Identity | Status |
|---|---|---|
| Nkogatse et al. 2018 | Photochem Photobiol 95(1):439-445, DOI 10.1111/php.13008, PMID 30183090 | Absent from `Research_Papers/` + `Papers_and_Datasets/` (68+ PDF full-text scan, twice, independently). Abstract obtained via PubMed: confirms the headline 24%-of-ambient personal-dose figure directly from source. Per-body-site table not in the abstract; full text not retrievable (no PMC deposit, not open access). Blocks the E1 body-site personal-dosimetry question. |
| Darling et al. 2013 | Osteoporosis International 24(2):477-488, PMID 22525977, DOI 10.1007/s00198-012-1973-2 | Absent from the repo (confirmed by full-text scan). All six route values the repo uses arrive via Webb 2021 Table 5, not this paper directly. Abstract-only: confirms the paper has its own personal-dosimetry data (which could test the "Darling's exposed BSA == Kift's" assumption) but gives no BSA figures itself. n-discrepancy flagged: abstract states Caucasian n=105 enrolled vs Webb 2021's n=90 (likely enrolled-vs-completing, unconfirmed). Full text: not open access, no PMC deposit. |
| Diffey 1983 | Phys Med Biol 28(6):647-657, DOI 10.1088/0031-9155/28/6/005, PMID 6878426 | Two DISTINCT wrong downloads under this exact citation slot: (1) a 1983 neutron/gamma dosimetry paper (wrong IOPscience issue, 28/1 not 28/6); (2) a 2-page excerpt of a 1986 Springer book chapter by the same author ("Mathematical Models for the Ultraviolet Optics of Human Epidermis," in Marks et al. eds., *Skin Models*). Neither contains d_1/2 = 8.5 um. The d_1/2 figure remains known only secondhand via Bruls 1984 p.492. Try a different acquisition route (ILL, coauthor, ResearchGate) rather than a third same-search re-download. |
| Matsuoka 1990 pre-tan paired data | J Lab Clin Med 116(1):87-90, PMID 2165512, NO DOI EXISTS (journal ceased 2005, never retrospectively assigned) | Full text blocked: ScienceDirect fetch returns HTTP 403 (bot protection). Needs institutional/librarian access, or contact co-author Bruce W. Hollis (Medical University of South Carolina), an active researcher, for a reprint. |
| Haddad & Chyu 1971 | JCEM 33(6):992-995, PMID 4332615 | Low priority (Luxwolda 2012 already gives a better-assayed version of the same "~160 nmol/L ceiling" claim). Not in PMC; pre-1971 JCEM. |

## New, previously unidentified leads

- **Linde, Wright & du Plessis 2022**, Ann Work Expo Health 66(1):41-51, DOI
  10.1093/annweh/wxab049, PMID 34302333. Same research group as Nkogatse
  2018 (co-author Wright), same population class (deeply pigmented South
  African outdoor workers, farmworkers not car guards). Fallback lead for
  the E1 body-site personal-dosimetry question if Nkogatse itself stays
  unobtainable. Abstract does not itself state a personal:ambient ratio;
  not open access, no PMC deposit.
- **Bib entries, abstract-level only (2026-09-23, decision C5(ii)).**
  Nkogatse 2018 (keyed `nkogatse2019`, the print-issue year), Blesić 2016
  (`blesic2016`) and Blesić 2019 (keyed `blesic2020`) are now in
  `paper/references.bib` section (g) as read-but-not-cited, with fields
  verified against PubMed. Listing them records that FST V-VI field
  dosimetry exists in the literature; none is obtained in full text here
  and none feeds a parameter.

## Ruled out as substitutes (checked, not just assumed)

- **Linde, Wright & du Plessis 2020** (already in the repo, DOI
  10.1111/srt.12895) does NOT contain personal dosimetry data (0 hits for
  "personal" or "dosimet" on full-text search) and cannot substitute for
  Nkogatse 2018.
- **Kaidbey & Kligman 1979**, J Am Acad Dermatol 1:249-260 -- a concrete
  lead on FST-dependent K_SC (>50% SC filtering in FST I-III vs 20-30% in
  "black skin," per Felton 2020's citation of it) but not independently
  verified here; primary source not chased.

## Dark-skin priority note

Per the repo's governing dark-skin-priority constraint: none of the five
papers obtained above (Felton 2020, Lavker 1995, Rhodes 2010, Fajuyigbe
2018, Stamatas 2008) include FST V-VI subjects. Nkogatse 2018 is the one
lead in this file targeting a deeply-pigmented population and it remains
unobtainable in full text; Linde 2022 is its fallback. Any parameter drawn
from the "obtained" table above is a light-skin extrapolation until FST
V-VI evidence is found -- flag at point of use, per the governing
constraint.
