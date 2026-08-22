## 1. Research Papers & Their Role in Code

| # | Paper & Authors | What It Provides & Why It Matters | Status |
|---|---|---|---|
| **1** | **Webb et al. (2021)**<br>*Endocrine Connections* | **Sunlight power boost (1.89x):** Standard UV meters track sunburn, not vitamin D. This review gives the conversion factor showing natural summer sunlight makes ~89% more vitamin D per sunburn unit than standard UV meters assume. | Core constant |
| **2** | **Heaney et al. (2008)**<br>*Am J Clin Nutr* | **Liver conversion slowdown:** Proves that when you take in a lot of vitamin D, your liver converts a smaller percentage into active blood levels, explaining why blood gains level off. | Supporting science |
| **3** | **Vernez et al. (2015)**<br>*J Expo Sci Environ Epidemiol* | **Body posture & sun angles:** Formula calculating how much sunlight reaches specific body parts (face, shoulders, legs) when standing, sitting, or kneeling. *(Duplicate of #49)* | Core formula |
| **4** | **Carpenter (2017)**<br>*J Steroid Biochem Mol Biol* | **Genetic breakdown defects:** Explains how mutations in the CYP24A1 gene stop the body from breaking down vitamin D, causing unusually slow clearance. | Genetics module |
| **5** | **Webb & Kift et al. (2018)**<br>*Nutrients* | **Real-world UK sun benchmark:** Field data showing fair skin needs ~9 min/day in UK summer, while darker skin cannot safely reach sufficiency from UK sun alone. Used to test the model against reality. | Validation benchmark |
| **6** | **Felton et al. (2016)**<br>*Br J Dermatol* | **Skin tone baseline (`K_const = 0.98`):** Gave identical low UV doses to fair and South Asian adults. Anchors how much natural melanin slows down synthesis without sunburn. | Core calibration |
| **7** | **Jahan (2018)**<br>*BSMMU Thesis* | **Tropical deficiency context:** Observational study in Bangladesh showing that people in sunny climates often stay deficient because of indoor work and clothing habits. | Supporting context |
| **8** | **de Boer et al. (2025)**<br>*J Invest Dermatol* | **Skin thickening without sunburn:** Showed that gentle, repeated UV exposure thickens the dead outer skin layer by ~34%, helping shield against sunburn over time. | Active mechanism |
| **9** | **Farrar et al. (2013)**<br>*Am J Clin Nutr* | **Dark skin validation target:** Clinical trial testing multiple UV doses on South Asian adults. Acts as the primary accuracy test for Skin Type V predictions. | Required test |
| **10** | **Elwood & Jopson (1997)**<br>*Int J Cancer* | **Sunburn penalty power (3.0):** Meta-analysis showing sudden burning bursts drive melanoma risk while steady gentle exposure does not. Anchors the heavy penalty on burning sessions. | Optimizer objective |
| **11** | **Pearse et al. (1987)**<br>*J Invest Dermatol* | **Skin recovery speed:** Biopsy data showing skin thickening peaks and takes about 3 weeks (21 days) to shed and return to normal. | Core parameter |
| **12** | **Bogh et al. (2011)**<br>*Exp Dermatol* | **Total dose rule:** Proves that vitamin D synthesis depends on total UV dose received, not whether it is delivered fast or slow. | Foundation rule |
| **13** | **Shih et al. (2018)**<br>*J Invest Dermatol* | **Melanin levels by skin type:** Measures exact melanin amounts across Skin Types I–VI and maps how deeply UV penetrates each skin type to cause DNA damage. | Calibration & risk |
| **14** | **Datta et al. (2017)**<br>*Photochem Photobiol Sci* | **Vitamin D clearance equation:** Proved half-life is not fixed; high levels clear in ~89 days, but low/deficient levels hang on for ~199 days. Governs daily decay. | Core state equation |
| **15** | **ISO / CIE (1999)**<br>*ISO 17166 / CIE S 007* | **Definition of 1 SED:** The international metrology standard defining 1 Standard Erythema Dose (100 J/m² of sunburn-weighted UV). | Standard definition |
| **16** | **Uçar & Holick (2025)**<br>*Nutrients* | **Sun benefits vs risks review:** Overview of the balance between making vitamin D and avoiding skin damage. | Supporting review |
| **17** | **Bogh et al. (2011)**<br>*Br J Dermatol* | **Skin area square-root rule:** Tested uncovering 6%, 12%, and 24% body area. Proved doubling exposed skin does not double vitamin D; it scales with the square root. | Core parameter |
| **18** | **Kallioglu et al. (2024)**<br>*Sci Rep* | **Mediterranean sunlight test:** Independent physics model of sun times in Turkey and the Mediterranean, used to check our seasonal estimates. | Validation test |
| **19** | **Webb et al. (2011)**<br>*Photochem Photobiol* | **Action spectrum math check:** Technical note standardizing the piecewise math formulas used for sunburn wavelength weighting. | Technical check |
| **20** | **Kift et al. (2013)**<br>*Br J Dermatol* | **Behavior vs biology split:** Used body badges to prove UK South Asians get ~2.1x less actual UV on their skin due to clothing and shade habits, keeping lifestyle separate from skin optics. | Core calibration |
| **21** | **Weihs et al. (2013)**<br>*Photochem Photobiol* | **Body part ratios:** Measures sun reaching different body parts (e.g. shoulders get 1.7x more than forehead while walking), justifying the 0.35 general posture default. | Geometry fallback |
| **22** | **Fajuyigbe et al. (2018)**<br>*FASEB J* | **Deep skin cancer protection:** Discovered dark skin gives 59-fold protection to deep basal stem cells where cancers start. Sets the cancer risk curve across skin types. | Core calibration |
| **23** | **Young et al. (2020)**<br>*J Invest Dermatol* | **Melanin's small effect:** Landmark trial proving dark skin only slows vitamin D by ~1.3–1.4x compared to fair skin under gentle UV, debunking the myth that dark skin needs 10x more sun. | Core calibration |
| **24** | **Krzyścin et al. (2015)**<br>*Przegl Dermatol* | **UV machine meter accuracy:** Engineering review on how clinical phototherapy sensors can drift. | Supporting context |
| **25** | **CDC / NCHS (2015)**<br>*CDC Analytical Note* | **NHANES lab test converter:** CDC equations converting older RIA blood test numbers into modern LC-MS/MS equivalents so years can be compared fairly. | Data pipeline |
| **26** | **Vähävihu et al. (2010a)**<br>*Br J Dermatol* | **Winter phototherapy test:** Clinical trial testing medical UVB lamps in Finnish women during winter. | Validation test |
| **27** | **Bogh et al. (2012)**<br>*Br J Dermatol* | **Phototherapy course benchmark:** High-precision trial (+55.8 nmol/L rise after 18 lamp sessions). One of our primary pass/fail tests. | Required test |
| **28** | **Vähävihu et al. (2010b)**<br>*Br J Dermatol* | **Skin immune response:** Tested UVB phototherapy in eczema and psoriasis patients, showing it boosts both vitamin D and skin immune defenses. | Supporting science |
| **29** | **Schmalwieser & Siani (2018)**<br>*Photochem Photobiol* | **Personal UV review:** Comprehensive review of body-badge studies across sports, showing why posture and angle matter. | Literature review |
| **30** | **Datta et al. (2019)**<br>*Photochem Photobiol Sci* | **Genes vs skin tone:** Shows that specific inherited genes (like VDR and CYP2R1) explain individual response differences better than visual skin tone alone. | Genetics module |
| **31** | **Shraim et al. (2025)**<br>*J Steroid Biochem Mol Biol* | **UK Biobank severe deficiency:** 500,000-person study showing extreme deficiency (<10 nmol/L) is concentrated in darker skin types during winter and spring. | Population context |
| **32** | **Schmalwieser et al. (2025)**<br>*Photochem Photobiol* | **Jogging sun exposure table:** Measured data for 7 body parts across sun angles from 20° to 60° while running. | Direct dataset |
| **33** | **Le et al. (2016)**<br>*Australas J Dermatol* | **Australian phototherapy trial:** Hospital clinic data from Perth, Australia tracking vitamin D gains across skin types I–IV. | Validation test |
| **34** | **McKenzie et al. (2012)**<br>*Photochem Photobiol Sci* | **Sunbed vs sun modeling:** New Zealand study comparing sunbed response to natural midday sun. | Validation test |
| **35** | **Ravnbak et al. (2009)**<br>*Acta Derm Venereol* | **UVA tanning speed:** Tracks how fast immediate tanning appears and fades after UVA exposure. | Mechanism |
| **36** | **Ravnbak et al. (2008)**<br>*Acta Derm Venereol* | **UVB tanning fade time:** Measures the multi-week fade of UVB tanning, setting the 28-day tanning decay constant. | Core parameter |
| **37** | **Keurentjes et al. (2021)**<br>*Photodermatol Photoimmunol* | **Gentle sun barrier changes:** Molecular evidence showing outer skin proteins adapt after gentle UV without sunburn. | Active mechanism |
| **38** | **Assi et al. (2022)**<br>*Eur J Dermatol* | **Laser skin analysis:** Confocal Raman laser scans showing structural barrier adaptation in outer skin after sub-sunburn UV. | Supporting science |
| **39** | **Liu et al. (2010)**<br>*Skin Pharmacol Physiol* | **Asian skin photoadaptation:** Confirmed skin barrier thickening happens in Chinese subjects after sun exposure. | Population evidence |
| **40** | **Kimlin et al. (2014)**<br>*Am J Epidemiol* | **Australian UV study (AusD):** Nationwide study connecting sunlight levels across 4 Australian cities to citizen blood levels. | Supporting survey |
| **41** | **Karppinen et al. (2017)**<br>*Int J Circumpolar Health* | **Spring sun validation target:** Real spring sun exposure in Finland (+14.9 nmol/L rise), used as one of our 10 required pass/fail tests. | Required test |
| **42** | **Lloyd (2004)**<br>*Br J Dermatol* | **Hospital UV meter error:** Found hospital meters vary by up to ±60%, warning that hospital lamp studies have built-in measurement fuzziness. | Measurement check |
| **43** | **Bogh et al. (2010)**<br>*J Invest Dermatol* | **Baseline level controls response:** Found that people starting with low vitamin D gain far more from the same sun dose than people starting high. | Foundation discovery |
| **44** | **Osmancevic et al. (2009)**<br>*Photodermatol Photoimmunol* | **Broadband vs Narrowband lamps:** Compared lamp types, proving broadband UVB makes more vitamin D per unit of redness than narrowband 311nm tubes. | Spectral check |
| **45** | **Osmancevic et al. (2010)**<br>*J Photochem Photobiol B* | **Long-term phototherapy:** Followed patients across repeated winter courses to track long-term maintenance. | Supporting test |
| **46** | `extract_pdfs.py` | Python script for extracting text from PDFs during development. Not a research paper. | Utility tool |
| **47** | **Pérez-Ferriols et al. (2014)**<br>*Actas Dermosifiliogr* | **Sunburn testing in Spain:** Clinical sunburn threshold testing protocols across skin phototypes. | Supporting context |
| **48** | **Rodríguez-Granados (2017)**<br>*Actas Dermosifiliogr* | **Lamp conversion factor (`1 J/cm² = 5.5 SED`):** Exact measurements of a Waldmann TL-01 cabinet, letting us turn hospital doses (J/cm²) into standard SED units. | Direct conversion |
| **49** | **Vernez et al. (2015)**<br>*Reprint* | **Exact duplicate of #3:** Publisher reprint of the Vernez 3D body exposure geometry paper. | Redundant duplicate |
| **50** | **Maghiar et al. (2026)**<br>*Life* | **Skin phototype barrier reactivity:** Recent study looking at skin blood flow and barrier response across phototypes. | Supporting context |
| **51** | **Thieden et al. (2008)**<br>*Photochem Photobiol* | **Gentle sunbed trial:** Randomized trial proving low-dose sunbed exposure raises vitamin D safely without sunburn. | Validation test |
| **52** | `supplementary materials.doc` | **3D Manikin visibility table:** Supplementary table from Vernez 2015 giving sky visibility percentages for 13 body parts across 5 postures. | Direct dataset |
| **53** | `var_demo_h NHANES dict.pdf` | **CDC variable dictionary:** Printed codebook defining variable names (SEQN, age, sex, race) for 2013–2014 NHANES data. | Data dictionary |
| **54** | **Grigalavicius et al. (2015)**<br>*J Photochem Photobiol B* | **TL-01 lamp efficiency (2.63x):** Showed medical 311nm lamps produce 2.63x more vitamin D per sunburn unit than standard erythema weighting. Averaged with Webb 2021 to 2.44x. | Core multiplier |
| **55** | **Young et al. (2021)**<br>*PNAS* | **Master synthesis scale & action spectrum:** Measured 3.48 nmol/L rise per SED in human subjects, anchoring the core model synthesis parameter $A$. | Master anchor |

---

## 2. Additional Research Papers Cited in Code

- **Matsuoka et al. (1990, JLCM)**: The only human trial testing the same people untanned vs tanned. Analysis revealed its 51% lower gain in summer was almost entirely due to higher starting vitamin D levels blunting new gains.
- **Scholze et al. (2026)** (*Commun Med*): Discovered the liver conversion knee at 56 nmol/L, replacing the old artificial hard cutoff with our realistic biphasic feedback mode.
- **Moan et al. (1989)** (*Cancer Res*): Found squamous cell skin cancer scales with lifetime UV dose raised to power 1.7, setting `SCC_DOSE_EXPONENT = 1.7`.
- **Cockell et al. (2001)** (*J Photochem Photobiol B*): Established that moving, free-living people receive ~35% of horizontal sunlight on exposed skin (`PERSONAL_DOSE_GEOMETRY_FACTOR = 0.35`).
- **Drincic et al. (2012)** (*Obesity*): Proved that lower vitamin D in heavier individuals is caused by dilution across bigger body volume, setting our body weight scaling formula.
- **Gandini et al. (2005)** (*Eur J Cancer*): Meta-analysis showing fair skin has 2.09x the melanoma risk of olive skin, used to check our cancer risk formula (model predicts 2.12x).

---

## 3.  Datasets

NHANES runs in 2-year survey cycles, indicated by single-letter suffixes:

| Suffix | Survey Years | Practical Details |
|:---:|---|---|
| `_B` to `_D` | 2001–2006 | Older cycles; lab assays were pre-standardized by CDC to modern LC-MS/MS scale (`LBDVIDMS`). |
| `_E` to `_J` | 2007–2018 | Modern direct LC-MS/MS lab tests (`LBXVIDMS`) and supplement tracking (`DS1TOT`). |
| ~~`_K`~~ | 2019–2020 | **Never released.** COVID halted fieldwork in March 2020. |
| `_L` | **2021–2023** | Latest post-pandemic release (August 2021–August 2023). *(Legacy files sometimes mislabel this as 2019–2020).* |
| `P_*` | 2017–March 2020 | Pre-pandemic combined file (15,560 rows) with special weights to bridge the COVID gap. |

### Module Overview
- **`BMX_*` (Body Measures):** Height, weight, BMI. Feeds body volume dilution so heavier individuals get appropriately scaled doses.
- **`DEMO_*` (Demographics):** Age, gender, race/ethnicity, exam season, and survey sample weights. Merges all records together.
- **`DEQ_*` (Sun Habits):** Self-reported sunburn reactions and daily outdoor minutes. *(Note: `DEQ_E` is restricted at CDC; `DEQ_L` omits the reaction question).*
- **`DR1TOT_*` / `DRXTOT_*` (Dietary Totals):** Daily food vitamin D intake in micrograms to set baseline dietary intake.
- **`DS1TOT_*` (Supplements):** Daily vitamin D pill intake from 2007 onward to separate supplement users from sun-dependent individuals.
- **`VID_*` / `L25_*` (Blood Tests):** Measured serum 25(OH)D concentrations in nmol/L.
- **`DR1IFF_*` (Food Items):** Massive files (30–57 MB) listing individual foods eaten. Unused by the main pipeline since daily totals are already in `DR1TOT`.

---

## . Clinical Spreadsheets & Optical Data

- **`Final skin type vit D database 26_03_2020.xlsx` (Young 2020):** Raw subject-level trial data (102 participants, Skin Types I–VI) proving melanin has only a modest 1.3–1.4x shielding effect on vitamin D.
- **`PNAS 202015867.xlsx` (Young 2021):** Raw trial slopes anchoring whole-body synthesis rate.
- **`de boer 2025 supplementary`:** Tab-delimited tables validating ~34% skin barrier thickening after repeated gentle UV.
- **`data/vernez2015_table_s1_vis_by_site_posture.csv`:** Sky visibility percentages for 13 body parts across 5 postures.
- **`data/spectra/` (Action Spectra):** Standard wavelength lookup files for vitamin D synthesis (CIE-174), sunburn (CIE-98), and TL-01 medical lamps.
l info.txt`:** Audit notes recording evidence gap resolutions.