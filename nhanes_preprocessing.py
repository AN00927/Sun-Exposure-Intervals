"""
nhanes_preprocessing.py
=======================
NHANES serum 25(OH)D and demographic preprocessing and harmonization script.
Implements the requirements of Concrete Recommended Change #8 from the July 2026 Scientific Audit.

Specifically:
- Uses CDC/NCHS Analytical Note (P27) regression equations to standardize
  2001-2006 original DiaSorin RIA assays to NIST-traceable LC-MS/MS-equivalent values (LBDVIDMS).
- Standardizes 2007+ LC-MS/MS total 25(OH)D (LBXVIDMS).
- Explicitly maps RIDEXMON (exam season) per P34/NCHS guidelines:
  1 = November 1 through April 30 (Winter/Spring)
  2 = May 1 through October 31 (Summer/Autumn)
- Maps race/ethnicity codes (RIDRETH1, RIDRETH3) to consistent categories.
- Merges laboratory, demographic, and body measures (BMI) files.
"""

import os
import pandas as pd
import numpy as np


def harmonize_ria_to_lcms(ria_val: float, cycle: str) -> float:
    """
    Apply CDC/NCHS regression equations (from P27 Analytical Note) to convert
    original DiaSorin RIA values (nmol/L) to LC-MS/MS-equivalent concentrations.
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
        "2019-2020": "L",  # 2019-2020 pre-pandemic cycle L
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
        # We need to look for the appropriate column and apply regression if 2001-2006.
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
        
        df_cycle = df_cycle.reset_index()
        merged_dfs.append(df_cycle)
        
    if merged_dfs:
        final_df = pd.concat(merged_dfs, ignore_index=True)
        # Drop rows missing both vitamin D and demographics
        final_df = final_df.dropna(subset=["vitd_standardized_nmol_L", "age"], how="all")
        
        output_file = os.path.join(output_dir, "nhanes_standardized_processed.csv")
        final_df.to_csv(output_file, index=False)
        print(f"Preprocessing completed. Saved standardized dataset to: {output_file}")
        print(f"Total processed records: {len(final_df)}")
        print(final_df.groupby("cycle")["vitd_standardized_nmol_L"].describe())
        return final_df
    else:
        print("No files were merged.")
        return None


if __name__ == "__main__":
    # Local path to datasets
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, "Papers_and_Datasets")
    output_path = os.path.join(base_dir, "audit_outputs")
    
    preprocess_nhanes_data(raw_data_path, output_path)
