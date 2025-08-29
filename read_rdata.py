#!/usr/bin/env python
import sys
from pathlib import Path

import pandas as pd

try:
    import pyreadr  # pip install pyreadr
except ImportError as e:
    raise SystemExit("pyreadr is not installed. Run: pip install pyreadr") from e


def load_r_file(path: str | Path) -> dict:
    """Return a dict of objects from an .RData or .rds file."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        return pyreadr.read_r(str(path))
    except pyreadr.PyreadrError as e:
        if "matrix, array or table object with more than one vector" in str(e):
            # Try to see what objects are available first
            try:
                objects = pyreadr.list_objects(str(path))
                print(f"\nFound objects in file: {objects}")
                print("The file contains complex R objects that can't be auto-converted.")
                print("Try using R to convert these objects to data.frames first, or")
                print("use a different approach to read specific objects.")
                raise SystemExit(f"Complex R objects detected: {e}")
            except Exception:
                pass
        raise SystemExit(f"Error reading R file: {e}") from e


def pick_dataframe(objs: dict) -> pd.DataFrame:
    """Pick a pandas DataFrame out of the loaded R objects."""
    # Filter only DataFrames
    df_objs = {k: v for k, v in objs.items() if isinstance(v, pd.DataFrame)}
    if not df_objs:
        raise TypeError("No pandas DataFrame objects found in the file.")
    if len(df_objs) == 1:
        return next(iter(df_objs.values()))
    # If multiple, print choices and ask user (or pick the largest)
    print("Multiple DataFrames found; names and shapes:")
    for name, df in df_objs.items():
        print(f"  - {name}: {df.shape}")
    choice = input("Enter the object name to use (blank = largest): ").strip()
    if choice and choice in df_objs:
        return df_objs[choice]
    # default to largest by rows
    return max(df_objs.items(), key=lambda kv: kv[1].shape[0])[1]


def main(path: str):
    objs = load_r_file(path)

    print("\nObjects loaded:")
    for name, obj in objs.items():
        shape = getattr(obj, "shape", "")
        print(f"  - {name}: {type(obj).__name__}{' ' + str(shape) if shape else ''}")

    try:
        df = pick_dataframe(objs)
    except Exception as e:
        raise SystemExit(f"\nCould not select a DataFrame: {e}") from e

    print("\nFirst 20 rows:")
    # Use to_string to avoid truncation in some environments
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python read_rdata.py /path/to/file.RData")
    main(sys.argv[1])
