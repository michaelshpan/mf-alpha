#!/usr/bin/env python3
"""
Convert scaled_annual_data_JFE.Rdata (matrix) -> Parquet with fundno + Date
"""

from pathlib import Path
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter, default_converter
import pandas as pd
import numpy as np

RAW = Path("data/raw") / "scaled_annual_data_JFE.Rdata"
OUT = Path("data/processed/characteristics")
OUT.mkdir(parents=True, exist_ok=True)

# ---- load into R globalenv ----
ro.r(f'load("{RAW.as_posix()}")')
obj_name = next(iter(ro.globalenv.keys()))
robj = ro.globalenv[obj_name]
print(f"Loaded R object '{obj_name}' of type {type(robj)}")

# ---- convert to pandas ----
with localconverter(default_converter + pandas2ri.converter):
    df = ro.conversion.rpy2py(robj)

# df is likely 29 rows (vars) x ~52k columns (fund-year combos) => transpose
if df.shape[0] < df.shape[1]:
    df = df.T

# Row index now carries fund+date; split them
df = df.reset_index()
if "index" in df.columns:
    # the replication files encode something like "fundno.Date" or "fundno"
    parts = df["index"].astype(str).str.split("[._]", expand=True)
    if parts.shape[1] == 2:
        df["fundno"], df["Date"] = parts[0], parts[1]
    else:
        df["fundno"] = parts[0]
        # there should be a separate Date column already
    df = df.drop(columns=["index"])

# Clean types
df["fundno"] = df["fundno"].astype(str)
df["Date"]   = df["Date"].astype(str).str[:6]
df["year"]   = df["Date"].str[:4].astype(int)

# Optional rename map to match your Python vars_list
rename = {
    "realized alpha": "alpha6_1mo_12m",
    # add the rest as needed
}
df = df.rename(columns=rename)

# Basic sanity
expected_cols = ["fundno", "Date"] + list(rename.values())
missing = set(expected_cols) - set(df.columns)
assert not missing, f"Still missing columns: {missing}"

# ---- save ----
df.to_parquet(OUT / "scaled_annual_data_JFE.parquet", index=False)
print("✅  Saved", OUT / "scaled_annual_data_JFE.parquet", "with shape", df.shape)
