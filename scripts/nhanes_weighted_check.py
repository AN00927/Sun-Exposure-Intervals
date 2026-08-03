"""
nhanes_weighted_check.py
========================
Survey-weighted analysis of NHANES serum 25(OH)D concentrations (nmol/L).
Calculates weighted mean 25(OH)D and prevalence of deficiency (<30 nmol/L and <50 nmol/L)
stratified by exam season (RIDEXMON) and race/ethnicity (RIDRETH3).

Note: zero 25(OH)D values retained per CDC Analytical Note guidelines.
"""

import os
import sys
import requests
import io
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Papers_and_Datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "tables")

CDC_URL_MAP = {
    "2001-2002": (2001, "L25_B", "DEMO_B", "BMX_B"),
    "2003-2004": (2003, "L25_C", "DEMO_C", "BMX_C"),
    "2005-2006": (2005, "VID_D", "DEMO_D", "BMX_D"),
    "2007-2008": (2007, "VID_E", "DEMO_E", "BMX_E"),
    "2009-2010": (2009, "VID_F", "DEMO_F", "BMX_F"),
    "2011-2012": (2011, "VID_G", "DEMO_G", "BMX_G"),
    "2013-2014": (2013, "VID_H", "DEMO_H", "BMX_H"),
    "2015-2016": (2015, "VID_I", "DEMO_I", "BMX_I"),
    "2017-2018": (2017, "VID_J", "DEMO_J", "BMX_J"),
}

RACE_ETHNICITY_MAP = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other / Multi-Racial",
}

SEASON_MAP = {
    1: "Nov-Apr (Winter/Spring)",
    2: "May-Oct (Summer/Autumn)",
}

def download_and_convert_xpt(year: int, file_prefix: str, target_csv: str) -> pd.DataFrame:
    """Download .xpt file directly from CDC Public DataFiles if CSV does not exist."""
    if os.path.exists(target_csv):
        return pd.read_csv(target_csv)
    
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file_prefix}.xpt"
    print(f"Downloading {file_prefix} from CDC: {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    
    df = pd.read_sas(io.BytesIO(resp.content), format="xport")
    df.to_csv(target_csv, index=False)
    print(f"Saved {target_csv} ({len(df)} rows)")
    return df

def load_nhanes_dataset() -> pd.DataFrame:
    """Load and merge NHANES datasets across cycles."""
    all_dfs = []
    
    for cycle, (year, vid_prefix, demo_prefix, bmx_prefix) in CDC_URL_MAP.items():
        vid_csv = os.path.join(DATA_DIR, f"{vid_prefix}.csv")
        demo_csv = os.path.join(DATA_DIR, f"{demo_prefix}.csv")
        bmx_csv = os.path.join(DATA_DIR, f"{bmx_prefix}.csv")
        
        try:
            df_vid = download_and_convert_xpt(year, vid_prefix, vid_csv)
            df_demo = download_and_convert_xpt(year, demo_prefix, demo_csv)
        except Exception as e:
            print(f"Error loading cycle {cycle}: {e}")
            continue
            
        df_vid = df_vid.set_index("SEQN")
        df_demo = df_demo.set_index("SEQN")
        
        df_cycle = pd.DataFrame(index=df_demo.index)
        df_cycle["cycle"] = cycle
        df_cycle["age"] = df_demo.get("RIDAGEYR", np.nan)
        df_cycle["sex"] = df_demo.get("RIAGENDR", np.nan)
        df_cycle["wtmec2yr"] = df_demo.get("WTMEC2YR", np.nan)
        df_cycle["exam_season"] = df_demo.get("RIDEXMON", np.nan)
        
        # Race/Ethnicity (prefer RIDRETH3 if available, else RIDRETH1)
        if "RIDRETH3" in df_demo.columns:
            df_cycle["race_code"] = df_demo["RIDRETH3"]
        else:
            df_cycle["race_code"] = df_demo.get("RIDRETH1", np.nan)
            
        # Serum 25(OH)D column extraction
        vitd_col = None
        for col in ["LBDVIDMS", "LBXVIDMS", "LBDVIDUS", "LBXVID"]:
            if col in df_vid.columns:
                vitd_col = col
                break
                
        if vitd_col:
            df_cycle["vitd_nmol_L"] = df_vid[vitd_col]
        else:
            df_cycle["vitd_nmol_L"] = np.nan
            
        df_cycle = df_cycle.reset_index()
        all_dfs.append(df_cycle)
        
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined

def compute_weighted_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute survey-weighted mean 25(OH)D and prevalence below 30 and 50 nmol/L."""
    # Retain 0 nmol/L values per CDC Analytical Note guidelines (do NOT filter vitd_nmol_L > 0)
    # zero 25(OH)D values retained per CDC Analytical Note guidelines
    valid = df.dropna(subset=["vitd_nmol_L", "wtmec2yr", "exam_season", "race_code"]).copy()
    valid = valid[valid["wtmec2yr"] > 0].copy()
    
    valid["race_label"] = valid["race_code"].map(RACE_ETHNICITY_MAP).fillna("Other")
    valid["season_label"] = valid["exam_season"].map(SEASON_MAP).fillna("Unknown")
    
    records = []
    
    groups = valid.groupby(["season_label", "race_label"])
    
    for (season, race), grp in groups:
        weights = grp["wtmec2yr"].values
        vals = grp["vitd_nmol_L"].values
        
        w_sum = np.sum(weights)
        if w_sum <= 0:
            continue
            
        w_mean = np.sum(weights * vals) / w_sum
        prev_30 = np.sum(weights[vals < 30.0]) / w_sum * 100.0
        prev_50 = np.sum(weights[vals < 50.0]) / w_sum * 100.0
        unweighted_n = len(grp)
        
        records.append({
            "Season": season,
            "Race/Ethnicity": race,
            "Wtd Mean 25(OH)D (nmol/L)": round(w_mean, 2),
            "Prev <30 nmol/L (%)": round(prev_30, 2),
            "Prev <50 nmol/L (%)": round(prev_50, 2),
            "N (unweighted)": unweighted_n,
        })
        
    res_df = pd.DataFrame(records)
    res_df = res_df.sort_values(by=["Season", "Race/Ethnicity"]).reset_index(drop=True)
    return res_df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading NHANES datasets (with CDC auto-download if missing)...")
    df = load_nhanes_dataset()
    print(f"Total merged records across cycles: {len(df)}")
    
    print("Computing survey-weighted statistics (zero 25(OH)D retained per CDC guidelines)...")
    stats_df = compute_weighted_statistics(df)
    
    output_path = os.path.join(OUTPUT_DIR, "nhanes_weighted_check.csv")
    stats_df.to_csv(output_path, index=False)
    print(f"\nSaved weighted analysis table to: {output_path}")
    print("\n--- NHANES Weighted Summary Table ---")
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    main()
