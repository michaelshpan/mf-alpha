import os
import re
import io
import time
import json
import argparse
import zipfile
import pathlib
from typing import List, Tuple, Dict, Optional
from urllib.parse import urljoin

import requests
import pandas as pd
from lxml import html
from dateutil.relativedelta import relativedelta
from datetime import datetime

# ---------------------------
# Config
# ---------------------------

DATASET_RR_PAGE = "https://www.sec.gov/data-research/sec-markets-data/mutual-fund-prospectus-riskreturn-summary-data-sets"

# These pages/paths tend to be stable; we scrape primary pages to avoid hard-coding quarterly URLs.
SERIES_CLASS_LANDING = "https://www.sec.gov/data-research/sec-markets-data"  # we’ll search for Series/Class dataset link from here
# Fallback JSON mapping (Tier 2)
MF_TICKERS_JSON = "https://www.sec.gov/files/company_tickers_mf.json"

OUTDIR_DEFAULT = "sec_outputs"

# ---------------------------
# Utilities
# ---------------------------

def sec_headers(host="www.sec.gov") -> dict:
    ua = os.getenv("SEC_USER_AGENT", "")
    if not ua or "@" not in ua:
        raise SystemExit(
            "SEC_USER_AGENT not set or missing an email. Example:\n"
            '  export SEC_USER_AGENT="Michael Pan michael@spacetimelogics.com"'
        )
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
        "Connection": "keep-alive",
    }

def polite_get(url: str, host="www.sec.gov", **kw) -> requests.Response:
    time.sleep(0.5)  # polite default
    r = requests.get(url, headers=sec_headers(host=host), timeout=60, **kw)
    if r.status_code == 403:
        raise RuntimeError(f"403 Forbidden for {url}. Check SEC_USER_AGENT and avoid parallel/rapid requests.")
    r.raise_for_status()
    return r

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[\s\-/]+", "_", c.strip().lower()) for c in df.columns]
    return df

def coalesce(*vals):
    for v in vals:
        if v is not None and pd.notna(v) and str(v).strip() != "":
            return v
    return None

def to_str_or_none(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return s if s else None

# ---------------------------
# Discover & Fetch: Series/Class bulk mapping (Tier 1) with fallback to JSON (Tier 2)
# ---------------------------

def discover_series_class_csv_links() -> List[str]:
    """
    Try to locate the 'Investment Company Series and Class Information' dataset and extract CSV links.
    We scan SEC markets data landing and follow links to find CSV/ZIP resources whose names indicate
    'series' & 'class' mapping.
    """
    links = []
    # Crawl one level: the markets-data landing page, then any page linked that includes 'series' & 'class'
    resp = polite_get(SERIES_CLASS_LANDING)
    root = html.fromstring(resp.text)
    hrefs = [a.get("href") for a in root.xpath("//a[@href]")]
    hrefs = [h for h in hrefs if h]

    # Consider candidate pages likely to host the dataset
    candidates = []
    for h in hrefs:
        if re.search(r"series.*class", h, re.I):
            candidates.append(urljoin("https://www.sec.gov/", h))

    # Visit candidates; look for CSV or ZIP links that look like data files
    resources = []
    for page in candidates[:10]:  # limit
        try:
            r = polite_get(page)
            t = r.text
            tree = html.fromstring(t)
            for a in tree.xpath("//a[@href]"):
                href = a.get("href") or ""
                if href.lower().endswith((".csv", ".zip")) and re.search(r"(series).*?(class)", href, re.I):
                    resources.append(urljoin("https://www.sec.gov/", href))
        except Exception:
            continue

    # Deduplicate, keep order
    seen = set()
    for u in resources:
        if u not in seen:
            seen.add(u)
            links.append(u)
    return links

def fetch_series_class_mapping(outdir: pathlib.Path) -> pd.DataFrame:
    """
    Try Tier 1 (bulk CSV/ZIP). If not found, fallback to Tier 2 (company_tickers_mf.json).
    Normalize to columns: cik, series_id, class_id, series_name, class_name, ticker (optional)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Attempt Tier 1
    csv_links = discover_series_class_csv_links()
    mapping = pd.DataFrame()
    for url in csv_links:
        try:
            print(f"[SERIES/CLASS] trying bulk resource: {url}")
            r = polite_get(url)
            content = r.content

            if url.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".csv"):
                            df = pd.read_csv(io.BytesIO(zf.read(name)), dtype=str, low_memory=False)
                            df = normalize_cols(df)
                            mapping = pd.concat([mapping, df], ignore_index=True)
            else:
                df = pd.read_csv(io.BytesIO(content), dtype=str, low_memory=False)
                df = normalize_cols(df)
                mapping = pd.concat([mapping, df], ignore_index=True)
        except Exception as e:
            print(f"[SERIES/CLASS] skip {url}: {e}")

    # If Tier 1 succeeded, try to standardize columns
    if not mapping.empty:
        # Try to locate canonical columns by name variations
        mapping["cik"] = mapping.get("cik", mapping.get("registrant_cik"))
        mapping["series_id"] = mapping.get("series_id", mapping.get("seriesid", mapping.get("series_id_")))
        mapping["class_id"] = mapping.get("class_contract_id", mapping.get("class_id", mapping.get("classcontractid")))
        mapping["series_name"] = mapping.get("series_name", mapping.get("seriesname"))
        mapping["class_name"] = mapping.get("class_contract_name", mapping.get("classname", mapping.get("class_contract")))
        mapping["ticker"] = mapping.get("ticker", mapping.get("class_contract_ticker_symbol", mapping.get("symbol")))

        keep = ["cik", "series_id", "class_id", "series_name", "class_name", "ticker"]
        mapping = mapping[keep].drop_duplicates().reset_index(drop=True)
        # Clean strings
        for c in keep:
            if c in mapping.columns:
                mapping[c] = mapping[c].map(to_str_or_none)
        mapping = mapping.dropna(subset=["cik", "series_id", "class_id"])
        return mapping

    # Tier 2 fallback: company_tickers_mf.json
    print("[SERIES/CLASS] falling back to company_tickers_mf.json …")
    r = polite_get(MF_TICKERS_JSON)
    data = r.json()
    rows = []
    
    # The JSON is usually a dict with numeric keys
    if isinstance(data, dict):
        for key, item in data.items():
            if not isinstance(item, dict):
                continue
            cik = to_str_or_none(item.get("cik"))
            ticker = to_str_or_none(item.get("ticker"))
            title = to_str_or_none(item.get("title"))
            
            # Try to extract series/class from title if present
            import re
            series_match = re.search(r"S\d{9}", str(title) if title else "")
            class_match = re.search(r"C\d{9}", str(title) if title else "")
            
            if series_match or class_match:
                rows.append({
                    "cik": cik,
                    "series_id": series_match.group() if series_match else None,
                    "class_id": class_match.group() if class_match else None,
                    "series_name": None,
                    "class_name": None,
                    "ticker": ticker
                })
    elif isinstance(data, list):
        # Handle list format (older style)
        for item in data:
            if not isinstance(item, dict):
                continue
            cik = to_str_or_none(item.get("cik"))
            ticker = to_str_or_none(item.get("ticker"))
            # Nested series/classes?
            series_list = item.get("series", []) or []
            if series_list:
                for ser in series_list:
                    sid = to_str_or_none(ser.get("id") or ser.get("seriesId") or ser.get("series_id"))
                    sname = to_str_or_none(ser.get("name"))
                    classes = ser.get("classes", []) or []
                    if classes:
                        for cl in classes:
                            cid = to_str_or_none(cl.get("id") or cl.get("classId") or cl.get("class_id"))
                            cname = to_str_or_none(cl.get("name"))
                            cticker = to_str_or_none(cl.get("ticker")) or ticker
                            rows.append({
                                "cik": cik, "series_id": sid, "class_id": cid,
                                "series_name": sname, "class_name": cname, "ticker": cticker
                            })
                    else:
                        rows.append({"cik": cik, "series_id": sid, "class_id": None,
                                     "series_name": sname, "class_name": None, "ticker": ticker})
            else:
                # If no nested structure, just record what we have
                rows.append({"cik": cik, "series_id": None, "class_id": None,
                             "series_name": None, "class_name": None, "ticker": ticker})
    mapping = pd.DataFrame(rows)
    mapping = mapping.dropna(subset=["cik", "series_id", "class_id"]).drop_duplicates().reset_index(drop=True)
    return mapping

# ---------------------------
# Discover & Fetch: Risk/Return ZIPs (last N quarters)
# ---------------------------

LABEL_PAT = re.compile(r"(\d{4})\s*Q\s*([1-4])", re.I)

def discover_rr_zip_links() -> List[Tuple[int, int, str]]:
    r = polite_get(DATASET_RR_PAGE)
    tree = html.fromstring(r.text)
    links = []
    for a in tree.xpath("//a[@href]"):
        href = a.get("href") or ""
        if not href.lower().endswith(".zip"):
            continue
        # Quarter can be in text or in href
        text = "".join(a.itertext()).strip()
        m = LABEL_PAT.search(text)
        if not m:
            m = LABEL_PAT.search(href)
        if m:
            yyyy, q = int(m.group(1)), int(m.group(2))
            links.append((yyyy, q, urljoin("https://www.sec.gov/", href)))
    # Dedup by (yyyy, q), keep the last occurrence (often the canonical one)
    uniq = {}
    for y, q, u in links:
        uniq[(y, q)] = u
    # Sort newest first
    out = sorted(((y, q, u) for (y, q), u in uniq.items()), key=lambda t: (t[0], t[1]), reverse=True)
    return out

def fetch_last_n_quarter_rr_zips(n_quarters: int, outdir: pathlib.Path) -> List[pathlib.Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    discovered = discover_rr_zip_links()
    if not discovered:
        raise SystemExit("No MFRR ZIP links discovered on the SEC page—page structure may have changed.")
    targets = discovered[:n_quarters]
    paths = []
    for y, q, url in targets:
        fname = f"{y}q{q}_mfrr.zip"
        p = outdir / fname
        if p.exists() and p.stat().st_size > 0:
            print(f"[RR] Skip existing {p.name}")
            paths.append(p)
            continue
        print(f"[RR] Downloading {p.name}  ←  {url}")
        resp = polite_get(url)
        with open(p, "wb") as f:
            f.write(resp.content)
        print(f"[RR] Wrote {p} ({p.stat().st_size:,} bytes)")
        paths.append(p)
    return paths

# ---------------------------
# Parse MFRR ZIPs: pick relevant tables and normalize IDs
# ---------------------------

ID_CANDIDATES = [
    "series_id", "seriesid", "series", "dei_seriesid", "dei_series_id",
    "class_id", "classcontractid", "class_contract_id", "class", "dei_classcontractid", "dei_class_contract_id",
    "cik", "registrant_cik", "registrantcik"
]

# Common fee/turnover columns we’d like to keep if present
VALUE_CANDIDATES = [
    # ER (net & gross)
    "total_annual_fund_operating_expenses_net",
    "total_annual_fund_operating_expenses_gross",
    "net_expense_ratio",
    "gross_expense_ratio",
    # Fee table breakdowns
    "management_fees", "management_fee",
    "distribution_and_service_12b_1_fees", "12b_1_fees", "distribution_and_service_12b1_fees",
    "other_expenses",
    # Turnover
    "portfolio_turnover_rate", "turnover", "portfolio_turnover"
]

def normalize_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    # map various possibilities to canonical
    df["cik"] = df.get("cik", df.get("registrant_cik", df.get("registrantcik")))
    df["series_id"] = coalesce(df.get("series_id"), df.get("seriesid"), df.get("dei_seriesid"), df.get("dei_series_id"))
    df["class_id"] = coalesce(df.get("class_id"), df.get("classcontractid"), df.get("class_contract_id"), df.get("dei_classcontractid"), df.get("dei_class_contract_id"))
    # strip/upper IDs
    for col in ["cik", "series_id", "class_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def read_rr_zip_tables(zippath: pathlib.Path) -> List[pd.DataFrame]:
    out = []
    with zipfile.ZipFile(zippath, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            try:
                df = pd.read_csv(zf.open(name), dtype=str, low_memory=False)
                df = normalize_id_cols(df)
                # keep only tables that contain at least some ID columns
                if not (("series_id" in df.columns) or ("class_id" in df.columns)):
                    continue
                # keep a narrow set of columns: IDs + likely values + as-of/period if present
                keep = [c for c in ["cik", "series_id", "class_id"] if c in df.columns]
                vals = [c for c in df.columns if c in VALUE_CANDIDATES]
                # some datasets use generic names; keep everything numeric-ish as fallback
                if not vals:
                    numeric_like = [c for c in df.columns if re.search(r"(fee|expense|ratio|turnover)", c)]
                    vals = list(dict.fromkeys(numeric_like))  # unique
                # also keep period/as-of if present
                date_like = [c for c in df.columns if re.search(r"(period|as_of|date|fye|fy_end)", c)]
                cols = list(dict.fromkeys(keep + vals + date_like))
                if cols:
                    out.append(df[cols].copy())
            except Exception as e:
                print(f"[RR] skip {zippath.name}:{name}: {e}")
    return out

def build_rr_panel(zip_paths: List[pathlib.Path]) -> pd.DataFrame:
    frames = []
    for zp in zip_paths:
        tables = read_rr_zip_tables(zp)
        if not tables:
            continue
        # tag quarter label for recency ranking
        m = re.search(r"(\d{4})q([1-4])", zp.name, re.I)
        if m:
            y, q = int(m.group(1)), int(m.group(2))
            qdate = datetime(y, 3*q, 1)  # rough
        else:
            qdate = None
        for df in tables:
            if qdate is not None:
                df = df.copy()
                df["quarter"] = f"{y}q{q}"
                df["quarter_dt"] = pd.Timestamp(qdate)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    rr = pd.concat(frames, ignore_index=True)
    # Clean IDs: SEC identifiers are uppercase (S#########, C#########)
    if "series_id" in rr.columns:
        rr["series_id"] = rr["series_id"].str.upper()
    if "class_id" in rr.columns:
        rr["class_id"] = rr["class_id"].str.upper()
    # Drop fully empty rows on IDs
    rr = rr.dropna(how="all", subset=[c for c in ["series_id", "class_id"] if c in rr.columns])
    return rr

# ---------------------------
# Join logic
# ---------------------------

def join_mapping_rr(mapping: pd.DataFrame, rr: pd.DataFrame) -> pd.DataFrame:
    # Normalize mapping IDs
    mapping = normalize_cols(mapping)
    if "series_id" in mapping.columns:
        mapping["series_id"] = mapping["series_id"].astype(str).str.upper()
    if "class_id" in mapping.columns:
        mapping["class_id"] = mapping["class_id"].astype(str).str.upper()
    if "cik" in mapping.columns:
        mapping["cik"] = mapping["cik"].astype(str).str.lstrip("0")

    if "cik" in rr.columns:
        rr["cik"] = rr["cik"].astype(str).str.lstrip("0")

    # Prefer to join on class_id if present, else series_id
    if "class_id" in rr.columns and "class_id" in mapping.columns:
        merged = rr.merge(mapping.drop_duplicates(subset=["class_id"]), on="class_id", how="left", suffixes=("", "_map"))
    else:
        merged = rr.merge(mapping.drop_duplicates(subset=["series_id"]), on="series_id", how="left", suffixes=("", "_map"))

    # If both class_id and series_id exist in both, ensure they don't conflict (optional sanity check)
    if all(c in merged.columns for c in ["series_id", "series_id_map"]):
        mismatch = merged["series_id"].notna() & merged["series_id_map"].notna() & (merged["series_id"] != merged["series_id_map"])
        if mismatch.any():
            print(f"[WARN] {mismatch.sum()} rows have series mismatch between RR and mapping; keeping RR value.")
            merged["series_id"] = merged["series_id"]
            merged.drop(columns=["series_id_map"], inplace=True, errors="ignore")

    # Coalesce identifiers (keep mapping's cik/series/class where RR lacks them)
    for col in ["cik", "series_id", "class_id"]:
        mcol = f"{col}_map"
        if mcol in merged.columns:
            merged[col] = merged[col].where(merged[col].notna() & (merged[col] != ""), merged[mcol])
            merged.drop(columns=[mcol], inplace=True, errors="ignore")

    return merged

# ---------------------------
# Main
# ---------------------------

def main(years: int, outdir: str):
    outpath = pathlib.Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    # 1) Fetch mapping (Series/Class)
    mapping = fetch_series_class_mapping(outpath)
    if mapping.empty:
        raise SystemExit("Failed to build Series/Class mapping from both bulk and JSON sources.")

    # Persist mapping for inspection
    mapping_csv = outpath / "mapping_series_class.csv"
    mapping.to_csv(mapping_csv, index=False)
    print(f"[OUT] wrote {mapping_csv} ({mapping.shape[0]:,} rows)")

    # 2) Fetch RR ZIPs for last N quarters (years*4)
    n_quarters = max(1, int(years) * 4)
    rr_zips = fetch_last_n_quarter_rr_zips(n_quarters, outpath / "mfrr_zips")

    # 3) Build RR panel from the ZIPs
    rr_panel = build_rr_panel(rr_zips)
    if rr_panel.empty:
        raise SystemExit("No RR tables parsed from the ZIPs. The dataset format may have changed.")

    # 4) Join
    joined = join_mapping_rr(mapping, rr_panel)

    # Optional: keep only a core set of value columns if too wide
    # (we keep IDs, ticker if present, quarter labels, and any value-like columns we found)
    id_cols = [c for c in ["cik", "series_id", "class_id", "ticker", "series_name", "class_name", "quarter", "quarter_dt"] if c in joined.columns]
    val_cols = [c for c in joined.columns if c not in id_cols and re.search(r"(fee|expense|ratio|turnover)", c)]
    final_cols = list(dict.fromkeys(id_cols + val_cols))
    if final_cols:
        joined = joined[final_cols]

    out_csv = outpath / "rr_joined_last3y.csv"
    joined.to_csv(out_csv, index=False)
    print(f"[OUT] wrote {out_csv} ({joined.shape[0]:,} rows, {joined.shape[1]} cols)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch SEC Series/Class mapping and Risk/Return datasets, then join.")
    ap.add_argument("--years", type=int, default=3, help="How many years (quarters*4) of RR data to fetch. Default 3.")
    ap.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT, help="Output directory.")
    args = ap.parse_args()
    main(args.years, args.outdir)