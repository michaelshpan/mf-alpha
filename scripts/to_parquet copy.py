#!/usr/bin/env python3
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path
import sys

RAW = Path("data/raw")
OUT = Path("data/processed")

def save_parquet(df: pd.DataFrame, path: Path, partition_cols=None, max_open_files=None):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if partition_cols:
        kwargs = {
            "root_path": str(path),
            "format": "parquet",
            "partitioning": partition_cols,
        }
        if max_open_files is not None:
            kwargs["max_open_files"] = max_open_files
        ds.write_dataset(table, **kwargs)
    else:
        pq.write_table(table, str(path))

def main():
    # === Characteristics ===
    # Prefer CSV you already have. If it's an RData, uncomment pyreadr block below.
    char_path = RAW / "scaled_annual_data_JFE.csv"
    if char_path.exists():
        char = pd.read_csv(char_path, dtype={"fundno": str, "Date": str})
    else:
        try:
            import pyreadr
            rdata = pyreadr.read_r(RAW / "scaled_annual_data_JFE.Rdata")
            # pick the object by inspecting keys()
            char = rdata[list(rdata.keys())[0]]
            char["fundno"] = char["fundno"].astype(str)
            char["Date"]   = char["Date"].astype(str)
        except Exception as e:
            print("Cannot find characteristics file in CSV or RData:", e)
            sys.exit(1)

    # ensure YYYYMM
    char["Date"] = char["Date"].str.slice(0, 6)
    char["year"] = char["Date"].str.slice(0, 4).astype(int)

    OUT_CHAR = OUT / "characteristics"
    OUT_CHAR.mkdir(parents=True, exist_ok=True)
    save_parquet(char, OUT_CHAR / "scaled_annual_data_JFE.parquet")

    # === Monthly Returns ===
    rets_path = RAW / "masked_fund_returns.csv"
    if rets_path.exists():
        rets = pd.read_csv(rets_path, dtype={"fundno": str, "Date": str})
    else:
        try:
            import pyreadr
            rdata = pyreadr.read_r(RAW / "masked_fund_returns.Rdata")
            rets = rdata[list(rdata.keys())[0]]
            rets["fundno"] = rets["fundno"].astype(str)
            rets["Date"]   = rets["Date"].astype(str)
        except Exception as e:
            print("Cannot find returns file in CSV or RData:", e)
            sys.exit(1)

    # normalize to first-of-month date
    rets["Date"]  = pd.to_datetime(rets["Date"].str[:6] + "01", format="%Y%m%d")
    rets["year"]  = rets["Date"].dt.year
    rets["month"] = rets["Date"].dt.month

    OUT_RETS = OUT / "returns"
    save_parquet(rets, OUT_RETS, partition_cols=["year", "month"], max_open_files=32)

    # === (Optional) Factor files ===
    # Example placeholder: skip if you don't have yet
    fac_path = RAW / "ff_factors.csv"
    if fac_path.exists():
        fac = pd.read_csv(fac_path)
        fac["Date"] = pd.to_datetime(fac["Date"])  # depends on file format
        save_parquet(fac, OUT / "factors" / "ff_factors.parquet")

    print("All done. Parquet written to data/processed/")


if __name__ == "__main__":
    main()
 