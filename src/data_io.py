import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def read_characteristics(s3_uri: str, usecols=None):
    logger.info("Reading characteristics from %s", s3_uri)
    df = pd.read_parquet(s3_uri, columns=usecols)
    return df


#def read_returns(s3_prefix: str):
#    """Read all monthly returns parquet partitions under prefix.
#    s3_prefix example: s3://bucket/processed/returns
#    """
#    logger.info("Reading returns from %s", s3_prefix)
#    dataset = ds.dataset(s3_prefix, format="parquet")
#    tbl = dataset.to_table()
#    df = tbl.to_pandas()
#    return df

def read_returns(root_path: str) -> pd.DataFrame:
    """
    Load the entire returns dataset partitioned as
    root_path/year=YYYY/month=MM/*.parquet
    Returns columns: fundno (str), Date (datetime64[M]), mret (float64)
    """
    logger.info("Reading returns from %s", root_path)

    # Let PyArrow read the whole partitioned dataset (hive-style year=/month=/)
    dataset = ds.dataset(root_path, format="parquet", partitioning="hive")

    # Read only the needed columns if present
    have = set(dataset.schema.names)
    cols = [c for c in ("fundno", "Date", "mret") if c in have]
    table = dataset.to_table(columns=cols)

    # NOTE: don't pass types_mapper here
    df = table.to_pandas()

    # --- Normalize dtypes ---
    # fundno: avoid "1234.0" from float -> string
    df["fundno"] = (
        df["fundno"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    # Date: coerce to first-of-month datetime
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = df["Date"].values.astype("datetime64[M]")
    else:
        s = df["Date"].astype(str).str.replace("-", "", regex=False)
        # Try YYYYMMDD first, then fallback to YYYYMM
        dt = pd.to_datetime(s.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        dt = dt.fillna(pd.to_datetime(s.str.slice(0, 6), format="%Y%m", errors="coerce"))
        df["Date"] = dt.values.astype("datetime64[M]")

    # mret: ensure float
    df["mret"] = pd.to_numeric(df["mret"], errors="coerce").astype("float64")

    return df

def write_parquet(df: pd.DataFrame, s3_uri: str):
    logger.info("Writing parquet to %s", s3_uri)
    df.to_parquet(s3_uri, index=False)
