#!/usr/bin/env python3
import openpyxl
import csv
from pathlib import Path

xlsx_path = Path("Papers_and_Datasets/PNAS 202015867.xlsx")
csv_path = Path("Papers_and_Datasets/other/Trial_Young2020_PNAS_SerumD_UVR.csv")

wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
ws = wb["PNAS 202015867"]

rows = list(ws.iter_rows(values_only=True))

# Header is row 3 (0-indexed)
header = rows[3]
# Data is rows 4 onwards
data_rows = rows[4:]

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in data_rows:
        if row[0] is not None:  # skip empty rows
            writer.writerow(row)

print(f"Extracted {len(data_rows)} rows to {csv_path}")
wb.close()
