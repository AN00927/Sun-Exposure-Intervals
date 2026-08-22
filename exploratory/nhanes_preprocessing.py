"""
nhanes_preprocessing.py
=======================
NHANES serum 25(OH)D and demographic preprocessing and harmonization.
Implements Concrete Recommended Change #8 from the July 2026 Scientific Audit.

Loads 2001-2006 data on the CDC-standardized LC-MS/MS-equivalent scale
(LBDVIDMS) and 2007+ data on the direct LC-MS/MS scale (LBXVIDMS).
See "Harmonization status" below -- this module does NOT apply a
RIA->LC-MS/MS transform itself.

Maps RIDEXMON (exam season) per P34/NCHS guidelines:
1 = November 1 through April 30 (Winter/Spring)
2 = May 1 through October 31 (Summer/Autumn)

Maps race/ethnicity codes (RIDRETH1, RIDRETH3) to consistent categories.
Merges laboratory, demographic, body measures (BMI) and dermatology
questionnaire (DEQ) files.

NHANES cycle letters (corrected 2026-08-20)
-------------------------------------------
The `cycles` dict below previously labelled cycle letter "L" as
"2019-2020". That was WRONG. Cycle "L" is CDC's **August 2021 - August
2023** release. Three independent checks agree:

  * CDC publishes DEMO_L under its 2021 folder
    (wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm).
  * DEMO_L.csv/VID_L.csv/DEQ_L.csv in Papers_and_Datasets/ carry SEQN
    130378-142310. The 2017-2018 cycle (J) ends at SEQN 102956; a genuine
    2019-2020 file would continue from there, not skip ~27k sequence
    numbers.
  * DEMO_L.csv has SDDSRVYR = 12 uniformly. NHANES codes 10 = 2017-2018
    and 12 = Aug 2021-Aug 2023. There is no SDDSRVYR = 11, because...

...cycle letter "K" (the true 2019-2020) was NEVER released as a
standalone public file. COVID interrupted fieldwork in March 2020, so
2019-2020 data exists ONLY folded into the "P_"-prefixed 2017-March-2020
pre-pandemic combined files (P_DEMO, P_VID, ...). Those use a different
naming convention and different survey weights (WTMECPRP, not WTMEC2YR),
so they cannot be dropped into the `cycles` loop as another two-year
cycle without weight-harmonization code that does not exist here.

Do NOT "restore" a 2019-2020 row to `cycles`. There is no such file.

Two copies of the same data (investigated 2026-08-20)
------------------------------------------------------
Papers_and_Datasets/ holds the flat CDC convention (DEMO_B.csv ...
DEMO_L.csv) that THIS module reads -- `preprocess_nhanes_data()` is called
with data_dir = Papers_and_Datasets (run_all.py), NOT the nhanes/ subdir.

Papers_and_Datasets/nhanes/ holds descriptively-named files
(NHANES_2001-2002_Serum_VitaminD.csv ...). These are NOT a second,
independent data pull. Papers_and_Datasets/rename_map.csv is the explicit
derivation: VID_L.csv -> NHANES_2019-2020_Serum_VitaminD.csv, and so on.
Verified by content: the two files have identical row counts, identical
SEQN ranges, and compare equal on their shared columns.

So `NHANES_2019-2020_Serum_VitaminD.csv` IS `VID_L.csv` under another
name, and it inherits the same mislabel -- it is 2021-2023 data. The
descriptive copy is unused by the pipeline; only the audit inventory in
audit_outputs/ references it. Renaming that copy is left as a follow-up
(it would invalidate the checked-in inventory CSVs), but nothing
scientific depends on it.

Harmonization status (verified 2026-08-17)
-------------------------------------------
CLAUDE.md requires: "Do NOT pool NHANES 2001-2006 (LBDVIDMS) with 2007+
(LBXVIDMS) without harmonization code." That rule was previously satisfied
only on paper: `harmonize_ria_to_lcms()` below implements the CDC/NCHS P27
regression equations, but `preprocess_nhanes_data()` never actually calls
it for any of the ten cycles distributed in Papers_and_Datasets/ --
confirmed by diffing `vitd_raw_nmol_L` against `vitd_standardized_nmol_L`
in the merged output, which are byte-identical in every cycle (2001-2002
through 2021-2023). The function looked wired in but was not.

The pooling is valid for a different reason: VID_B.csv/VID_C.csv/VID_D.csv
(2001-2006) contain only the `LBDVIDMS` column -- CDC's own
already-standardized series -- never a raw `LBDVIDUS`/`LBXVID` RIA column.
CDC did the RIA->LC-MS/MS harmonization upstream, once, before distributing
the public files this repo uses. Both 2001-2006 (`LBDVIDMS`) and 2007+
(`LBXVIDMS`) are already on the LC-MS/MS-equivalent scale by construction.

`harmonize_ria_to_lcms()` and its regression equations are kept as a
documented, guarded fallback for a hypothetical future data pull that
supplies a genuinely raw, un-standardized RIA column -- see its docstring.
`verify_no_harmonization_needed()` makes the current no-op state an
explicit, checked assertion: it raises if any cycle's data ever takes the
harmonize path, or if a row labelled "already standardized" is not
numerically identical to its raw value. Call it after
`preprocess_nhanes_data()` (the `__main__` block does) before treating the
pooled dataset as valid.
"""

import os
import pandas as pd
import numpy as np


# --- Dermatology questionnaire (DEQ) coding ------------------------------
# DED031: "reaction to the sun after several months of not being in the
# sun" -- a Fitzpatrick-reaction-type proxy. Valid ordinal responses are
# 1..5, increasing = less burn / more tolerance:
#   1 = severe sunburn with blisters       4 = mild sunburn, turns tan
#   2 = severe sunburn, peeling            5 = nothing happens (no burn)
#   3 = mild sunburn, some tan
# Everything else is a non-response and MUST become NaN, not a number:
#   6  = "never goes out in the sun"  (not a reaction severity at all)
#   77 = refused,  99 = don't know
#   0  = undocumented; appears 1-4 times per cycle in the raw files and is
#        not in the codebook's 1-5 range, so it is dropped as invalid.
DED031_VALID = (1, 2, 3, 4, 5)

# DED120 / DED125: self-reported minutes spent outside on a work/school day
# and on a non-work day (renamed from DEQ120/DEQ125 on public release).
# 3333 is a SENTINEL, not a duration -- it encodes "does not go outside" /
# "does not work or go to school" (and in the DED125 branch, "at work
# 9-to-5 seven days a week"). Treating it as 3333 minutes (55 hours) would
# corrupt any average. It is common: 355-640 respondents per cycle on
# DED120. 7777 = refused, 9999 = don't know.
# NOTE: 480 is a legitimate TOP-CODE (8 h), not a sentinel -- it is kept.
DEQ_MINUTES_SENTINELS = (3333, 7777, 9999)


def _recode_deq_minutes(series: pd.Series) -> pd.Series:
    """DED120/DED125 raw codes -> minutes, with sentinels dropped to NaN.

    See DEQ_MINUTES_SENTINELS above: 3333/7777/9999 are non-durations and
    become NaN. Negative values (none observed, but cheap to guard) are
    also dropped.
    """
    out = pd.to_numeric(series, errors="coerce")
    out = out.where(~out.isin(DEQ_MINUTES_SENTINELS))
    return out.where(out >= 0)


def _recode_ded031(series: pd.Series) -> pd.Series:
    """DED031 raw codes -> 1..5 ordinal, non-responses dropped to NaN."""
    out = pd.to_numeric(series, errors="coerce")
    return out.where(out.isin(DED031_VALID))


def harmonize_ria_to_lcms(ria_val: float, cycle: str) -> float:
    """
    CDC/NCHS regression equations (P27 Analytical Note) converting original
    DiaSorin RIA 25(OH)D values (nmol/L) to LC-MS/MS-equivalent concentrations.

    STATUS (verified 2026-08-17): NOT invoked by `preprocess_nhanes_data()`
    for any of the ten NHANES cycles distributed in Papers_and_Datasets/.
    The 2001-2006 files (VID_B.csv, VID_C.csv, VID_D.csv) contain only the
    `LBDVIDMS` column -- CDC's own pre-standardized series -- never a raw
    `LBDVIDUS`/`LBXVID` RIA column, so `preprocess_nhanes_data()` always
    takes the "already standardized" branch and this function is never
    called. Confirmed: `vitd_standardized_nmol_L` is byte-identical to
    `vitd_raw_nmol_L` in all ten pooled cycles.

    Pooling 2001-2006 with 2007+ is valid WITHOUT this function: both halves
    are already on the LC-MS/MS-equivalent scale (LBDVIDMS for 2001-2006,
    LBXVIDMS for 2007+) -- CDC harmonized upstream, once, before
    distributing the public files. Kept as a documented fallback (see the
    `else` branch of the vitd_raw_col dispatch in
    `preprocess_nhanes_data()`) for a hypothetical future data pull that
    supplies a genuinely raw, un-standardized RIA column. If that branch is
    ever taken, `verify_no_harmonization_needed()` below raises rather than
    silently pooling a locally-regressed series against CDC's own.
    """
    if pd.isna(ria_val):
        return np.nan
    
    # original equations from CDC:
    # 2001-2002: LC_MSMS = 6.43435 + 0.95212 * RIA
    # 2003-2004: LC_MSMS = 1.72786 + 0.98284 * RIA
    # 2005-2006: LC_MSMS = 8.36753 + 0.97012 * RIA
    if cycle == "2001-2002":
        return 6.43435 + 0.95212 * ria_val
    elif cycle == "2003-2004":
        return 1.72786 + 0.98284 * ria_val
    elif cycle == "2005-2006":
        return 8.36753 + 0.97012 * ria_val
    else:
        return ria_val


def preprocess_nhanes_data(data_dir: str, output_dir: str):
    """
    Loads, standardizes, and merges NHANES files across cycles.
    """
    # Cycle-letter -> survey years. See "NHANES cycle letters" in the module
    # docstring before editing: cycle "L" is August 2021 - August 2023, NOT
    # 2019-2020, and there is no standalone 2019-2020 cycle to add here.
    cycles = {
        "2001-2002": "B",
        "2003-2004": "C",
        "2005-2006": "D",
        "2007-2008": "E",
        "2009-2010": "F",
        "2011-2012": "G",
        "2013-2014": "H",
        "2015-2016": "I",
        "2017-2018": "J",
        "2021-2023": "L",  # Aug 2021-Aug 2023; SDDSRVYR=12. NOT 2019-2020.
    }
    
    merged_dfs = []
    
    print("Beginning NHANES data harmonization and preprocessing...")
    
    for cycle_years, letter in cycles.items():
        demo_file = os.path.join(data_dir, f"DEMO_{letter}.csv")
        vitd_file = os.path.join(data_dir, f"VID_{letter}.csv")
        bmx_file = os.path.join(data_dir, f"BMX_{letter}.csv")
        
        if not (os.path.exists(demo_file) and os.path.exists(vitd_file)):
            print(f"Skipping cycle {cycle_years} (files not found)")
            continue
            
        print(f"Processing cycle {cycle_years} (Cycle {letter})...")
        
        # 1. Load data
        df_demo = pd.read_csv(demo_file)
        df_vitd = pd.read_csv(vitd_file)
        
        df_bmx = None
        if os.path.exists(bmx_file):
            df_bmx = pd.read_csv(bmx_file)
            
        # Ensure SEQN is index or join key
        df_demo = df_demo.set_index("SEQN")
        df_vitd = df_vitd.set_index("SEQN")
        if df_bmx is not None:
            df_bmx = df_bmx.set_index("SEQN")
            
        # 2. Extract key demographic variables
        # RIDEXMON: 1 = Nov-Apr, 2 = May-Oct (P34)
        season_col = "RIDEXMON" if "RIDEXMON" in df_demo.columns else None
        
        # Race/Ethnicity proxy variables
        # RIDRETH1 (all cycles) or RIDRETH3 (introduced in 2011-2012 to add Asian category)
        race_col = "RIDRETH3" if "RIDRETH3" in df_demo.columns else "RIDRETH1"
        
        df_cycle = pd.DataFrame(index=df_demo.index)
        df_cycle["cycle"] = cycle_years
        df_cycle["age"] = df_demo.get("RIDAGEYR", np.nan)
        df_cycle["sex"] = df_demo.get("RIAGENDR", np.nan) # 1 = Male, 2 = Female
        df_cycle["wtmec2yr"] = df_demo.get("WTMEC2YR", np.nan)
        df_cycle["wtint2yr"] = df_demo.get("WTINT2YR", np.nan)
        df_cycle["sdmvpsu"] = df_demo.get("SDMVPSU", np.nan)
        df_cycle["sdmvstra"] = df_demo.get("SDMVSTRA", np.nan)
        
        if season_col:
            df_cycle["exam_season"] = df_demo[season_col]
            # Map season for clarity
            df_cycle["exam_season_label"] = df_cycle["exam_season"].map({
                1: "November-April (Winter/Spring)",
                2: "May-October (Summer/Autumn)"
            })
        else:
            df_cycle["exam_season"] = np.nan
            df_cycle["exam_season_label"] = "Unknown"
            
        df_cycle["race_ethnicity"] = df_demo.get(race_col, np.nan)
        
        # 3. Extract and Harmonize Vitamin D (25(OH)D)
        # P27: use LC-MS/MS equivalent values.
        # Original RIA variable in 2001-2006 cycles is LBXVID (or LBDVIDUS in nmol/L).
        vitd_raw_col = None
        for col in ["LBDVIDUS", "LBDVIDMS", "LBXVIDMS", "LBXVID"]:
            if col in df_vitd.columns:
                vitd_raw_col = col
                break
                
        if vitd_raw_col:
            raw_series = df_vitd[vitd_raw_col]
            
            # Apply CDC/NCHS regression equations for RIA cycles
            if cycle_years in ["2001-2002", "2003-2004", "2005-2006"]:
                if vitd_raw_col in ("LBDVIDMS", "LBXVIDMS"):
                    # Already CDC-standardized, use directly
                    df_cycle["vitd_raw_nmol_L"] = raw_series
                    df_cycle["vitd_standardized_nmol_L"] = raw_series
                    df_cycle["harmonized_method"] = "LC-MS/MS Direct (already standardized)"
                else:
                    # Raw RIA value, apply CDC regression
                    df_cycle["vitd_raw_nmol_L"] = raw_series
                    df_cycle["vitd_standardized_nmol_L"] = raw_series.apply(
                        lambda x: harmonize_ria_to_lcms(x, cycle_years)
                    )
                    df_cycle["harmonized_method"] = "DiaSorin RIA (CDC Standardized)"
            else:
                # 2007+ used LC-MS/MS directly
                df_cycle["vitd_raw_nmol_L"] = raw_series
                df_cycle["vitd_standardized_nmol_L"] = raw_series
                df_cycle["harmonized_method"] = "LC-MS/MS Direct"
        else:
            df_cycle["vitd_raw_nmol_L"] = np.nan
            df_cycle["vitd_standardized_nmol_L"] = np.nan
            df_cycle["harmonized_method"] = "Missing"
            
        # 4. Extract Body Measures (BMI)
        if df_bmx is not None:
            df_cycle["bmi"] = df_bmx.get("BMXBMI", np.nan)
            df_cycle["weight_kg"] = df_bmx.get("BMXWT", np.nan)
            df_cycle["height_cm"] = df_bmx.get("BMXHT", np.nan)
        else:
            df_cycle["bmi"] = np.nan
            df_cycle["weight_kg"] = np.nan
            df_cycle["height_cm"] = np.nan
            
        # 5. Dietary vitamin D data (DR1TOT, DS1TOT)
        # DR1TOT contains DR1TVD (food vitamin D, mcg) for 2007+ cycles (E-L)
        # DS1TOT contains DS1TVD (supplement vitamin D, mcg) for 2007+ cycles (E-L)
        df_cycle["dietary_vitd_food_mcg"] = np.nan
        diet_file = os.path.join(data_dir, f"DR1TOT_{letter}.csv")
        if os.path.exists(diet_file):
            df_diet = pd.read_csv(diet_file).set_index("SEQN")
            if "DR1TVD" in df_diet.columns:
                df_cycle["dietary_vitd_food_mcg"] = df_diet["DR1TVD"]
        
        df_cycle["dietary_vitd_supplement_mcg"] = np.nan
        supp_file = os.path.join(data_dir, f"DS1TOT_{letter}.csv")
        if os.path.exists(supp_file):
            df_supp = pd.read_csv(supp_file).set_index("SEQN")
            if "DS1TVD" in df_supp.columns:
                df_cycle["dietary_vitd_supplement_mcg"] = df_supp["DS1TVD"]
        
        # Total dietary vitamin D (sum, NaN if both missing)
        both_missing = df_cycle["dietary_vitd_food_mcg"].isna() & df_cycle["dietary_vitd_supplement_mcg"].isna()
        df_cycle["total_dietary_vitD_mcg"] = df_cycle["dietary_vitd_food_mcg"].fillna(0) + df_cycle["dietary_vitd_supplement_mcg"].fillna(0)
        df_cycle.loc[both_missing, "total_dietary_vitD_mcg"] = np.nan
        
        # Dietary source flag
        def _classify_dietary(row):
            food = row["dietary_vitd_food_mcg"]
            supp = row["dietary_vitd_supplement_mcg"]
            if pd.isna(food) and pd.isna(supp):
                return "missing"
            f = food if pd.notna(food) else 0.0
            s = supp if pd.notna(supp) else 0.0
            if s > 0:
                return "supplemented"
            if f < 1:
                return "UV-only"
            return "food-only"
        
        df_cycle["dietary_source_flag"] = df_cycle.apply(_classify_dietary, axis=1)

        # 6. Dermatology questionnaire (DEQ) -- sun-reaction and outdoor time.
        # Column availability is NOT uniform across cycles, so every field is
        # pre-filled with NaN and only overwritten where the column exists
        # (same pattern as DR1TOT/DS1TOT above). Verified against the raw
        # files 2026-08-20:
        #   DED031          present in B, C, D, F, G, H, I, J  (absent: L)
        #   DED120/DED125   present in F, G, H, I, J only      (2009-2018)
        # DEQ_E (2007-2008) does not exist publicly -- it is RDC
        # limited-access only -- so cycle E simply has no DEQ file. That is
        # expected, not a missing download.
        # DEQ is administered to ages 20-59 only, so even where the file
        # exists these columns are NaN for most children and older adults.
        df_cycle["deq_skin_reaction"] = np.nan
        df_cycle["deq_outdoor_minutes_workday"] = np.nan
        df_cycle["deq_outdoor_minutes_nonworkday"] = np.nan

        deq_file = os.path.join(data_dir, f"DEQ_{letter}.csv")
        if os.path.exists(deq_file):
            df_deq = pd.read_csv(deq_file).set_index("SEQN")
            if "DED031" in df_deq.columns:
                df_cycle["deq_skin_reaction"] = _recode_ded031(
                    df_deq["DED031"])
            if "DED120" in df_deq.columns:
                df_cycle["deq_outdoor_minutes_workday"] = (
                    _recode_deq_minutes(df_deq["DED120"]))
            if "DED125" in df_deq.columns:
                df_cycle["deq_outdoor_minutes_nonworkday"] = (
                    _recode_deq_minutes(df_deq["DED125"]))

        df_cycle = df_cycle.reset_index()
        merged_dfs.append(df_cycle)
        
    if merged_dfs:
        final_df = pd.concat(merged_dfs, ignore_index=True)
        # Drop rows missing both vitamin D and demographics
        final_df = final_df.dropna(subset=["vitd_standardized_nmol_L", "age"], how="all")
        
        # Explicit, checked no-op verification (see module docstring
        # "Harmonization status" and harmonize_ria_to_lcms() docstring).
        verify_no_harmonization_needed(final_df)
        print("Harmonization check: PASS -- all 2001-2006 rows already "
              "CDC-standardized (LBDVIDMS); harmonize_ria_to_lcms() was not "
              "invoked. Pooling with 2007+ (LBXVIDMS) is valid.")

        output_file = os.path.join(output_dir, "nhanes_standardized_processed.csv")
        final_df.to_csv(output_file, index=False)
        print(f"Preprocessing completed. Saved standardized dataset to: {output_file}")
        print(f"Total processed records: {len(final_df)}")
        print(final_df.groupby("cycle")["vitd_standardized_nmol_L"].describe())
        return final_df
    else:
        print("No files were merged.")
        return None


def verify_no_harmonization_needed(df: pd.DataFrame) -> None:
    """Fail loudly if RIA harmonization was silently needed for any cycle.

    Checks the explicit, verified no-op described in the module docstring
    and in `harmonize_ria_to_lcms()`'s docstring. Pooling 2001-2006
    (LBDVIDMS) with 2007+ (LBXVIDMS) is valid only if every 2001-2006 row
    came from an already-standardized column, rather than from
    `harmonize_ria_to_lcms()`'s locally-fitted regression. Call after
    `preprocess_nhanes_data()` (the `__main__` block does) before treating
    the pooled dataset as valid.

    Raises
    ------
    AssertionError
        If any 2001-2006 row took the `harmonize_ria_to_lcms()` path (label
        mismatch), or if a row labelled "already standardized" is not
        numerically identical between raw and standardized columns.
    """
    ria_cycles = {"2001-2002", "2003-2004", "2005-2006"}
    cycle_rows = df[df["cycle"].isin(ria_cycles)]

    expected_labels = {"LC-MS/MS Direct (already standardized)", "Missing"}
    bad_method = cycle_rows[~cycle_rows["harmonized_method"].isin(expected_labels)]
    if not bad_method.empty:
        raise AssertionError(
            f"harmonize_ria_to_lcms() was applied to {len(bad_method)} rows "
            f"(cycles: {sorted(bad_method['cycle'].unique())}, methods seen: "
            f"{sorted(bad_method['harmonized_method'].unique())}). Pooling "
            "with 2007+ LBXVIDMS values now mixes a CDC-standardized series "
            "with a locally regression-standardized one -- this needs "
            "manual review (check harmonize_ria_to_lcms()'s equations "
            "against the current CDC P27 Analytical Note) before the pooled "
            "dataset is used for any cross-cycle comparison.")

    standardized = cycle_rows[
        cycle_rows["harmonized_method"] == "LC-MS/MS Direct (already standardized)"]
    both_present = (standardized["vitd_raw_nmol_L"].notna() &
                    standardized["vitd_standardized_nmol_L"].notna())
    mismatched = both_present & (standardized["vitd_raw_nmol_L"] !=
                                 standardized["vitd_standardized_nmol_L"])
    if mismatched.any():
        raise AssertionError(
            f"{int(mismatched.sum())} rows labelled 'already standardized' "
            "have vitd_raw_nmol_L != vitd_standardized_nmol_L -- the no-op "
            "assumption is numerically violated; investigate before using "
            "this data.")


if __name__ == "__main__":
    # Local path to datasets
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, "Papers_and_Datasets")
    output_path = os.path.join(base_dir, "audit_outputs")

    preprocess_nhanes_data(raw_data_path, output_path)
