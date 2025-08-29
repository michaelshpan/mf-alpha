import rpy2.robjects as ro, pandas as pd, numpy as np
from pathlib import Path, PurePosixPath

RAW = Path("data/raw") / "scaled_annual_data_JFE.Rdata"
OUT = Path("data/processed/characteristics")
OUT.mkdir(parents=True, exist_ok=True)

# ---- load ----
ro.r(f'load("{PurePosixPath(RAW)}")')
obj_name = next(iter(ro.globalenv.keys()))
robj = ro.globalenv[obj_name]

# ---- convert ----
row_names = list(robj.rownames)
col_names = list(robj.colnames)
df = (pd.DataFrame(np.array(robj), index=row_names, columns=col_names)
        .reset_index()
        .rename(columns={'index': 'fundno'}))

# ---- clean ----
df["fundno"] = df["fundno"].astype(str)
df["Date"]   = df["Date"].astype(str).str[:6]
df["year"]   = df["Date"].str[:4].astype(int)

df.to_parquet(OUT / "scaled_annual_data_JFE.parquet", index=False)
print("✅  Saved Parquet with", len(df), "rows and", len(df.columns), "columns")
