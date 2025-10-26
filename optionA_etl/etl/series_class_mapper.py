#!/usr/bin/env python3
"""
Series/Class ID mapping module for SEC data.
Maps series IDs to class IDs to enable proper joining of turnover data (series-level) 
with expense data (class-level).
"""

import os
import re
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from lxml import html

log = logging.getLogger(__name__)

# SEC data endpoints
SERIES_CLASS_LANDING = "https://www.sec.gov/data-research/sec-markets-data"
MF_TICKERS_JSON = "https://www.sec.gov/files/company_tickers_mf.json"

def sec_headers() -> dict:
    """Get SEC API headers"""
    ua = os.getenv("SEC_USER_AGENT", "Research Contact research@example.com")
    if "@" not in ua:
        log.warning("SEC_USER_AGENT should contain email address")
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
        "Connection": "keep-alive",
    }

def discover_series_class_bulk_files() -> List[str]:
    """
    Discover bulk CSV/ZIP files containing series/class mapping data from SEC.
    Prioritizes the official Investment Company Series and Class Information files.
    
    Returns:
        List of URLs to CSV/ZIP files (prioritized)
    """
    log.info("Discovering series/class bulk files from SEC")
    
    # Priority 1: Known Investment Company Series/Class CSV files (most reliable)
    priority_files = []
    base_url = "https://www.sec.gov/files/investment/data/other/investment-company-series-class-information/"
    
    # Try recent years first
    for year in [2025, 2024, 2023, 2022, 2021]:
        urls_to_try = [
            f"{base_url}investment-company-series-class-{year}.csv",
            f"{base_url}investment_company_series_class_{year}.csv",
        ]
        
        for url in urls_to_try:
            try:
                resp = requests.head(url, headers=sec_headers(), timeout=10)
                if resp.status_code == 200:
                    priority_files.append(url)
                    log.info(f"Found priority series/class file: {url}")
                    break  # Found this year's file, move to next year
            except:
                continue
    
    # If we found priority files, return them (most recent first)
    if priority_files:
        return priority_files
    
    # Fallback: Discovery method (as backup)
    log.info("Priority files not found, attempting discovery")
    
    try:
        # Direct check of investment company data page
        investment_page = "https://www.sec.gov/about/opendatasetsshtmlinvestment_company"
        resp = requests.get(investment_page, headers=sec_headers(), timeout=30)
        resp.raise_for_status()
        
        root = html.fromstring(resp.text)
        discovered_files = []
        
        # Look specifically for series-class CSV files
        for a in root.xpath("//a[@href]"):
            href = a.get("href", "")
            if "series" in href.lower() and "class" in href.lower() and href.endswith(".csv"):
                full_url = href if href.startswith("http") else f"https://www.sec.gov{href}"
                discovered_files.append(full_url)
                log.info(f"Discovered series/class file: {full_url}")
        
        if discovered_files:
            # Sort by year (newest first)
            def extract_year(url):
                import re
                match = re.search(r'(\d{4})', url)
                return int(match.group(1)) if match else 0
            
            discovered_files.sort(key=extract_year, reverse=True)
            return discovered_files[:5]  # Top 5 most recent
        
    except Exception as e:
        log.warning(f"Discovery method failed: {e}")
    
    return []


def fetch_series_class_mapping() -> pd.DataFrame:
    """
    Fetch series/class mapping from SEC, trying multiple sources:
    1. Bulk CSV/ZIP files from SEC markets data (PRIORITY)
    2. company_tickers_mf.json as fallback
    
    Returns DataFrame with columns: cik, series_id, class_id, series_name, class_name, ticker
    """
    
    # Try bulk CSV/ZIP files first (more reliable and comprehensive)
    log.info("Attempting to fetch series/class mapping from bulk CSV/ZIP files")
    bulk_files = discover_series_class_bulk_files()
    
    all_mappings = []
    for file_url in bulk_files[:5]:  # Limit attempts
        try:
            log.info(f"Downloading and parsing: {file_url}")
            resp = requests.get(file_url, headers=sec_headers(), timeout=120)
            resp.raise_for_status()
            
            if file_url.endswith(".zip"):
                # Handle ZIP files
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".csv"):
                            log.info(f"Processing CSV from ZIP: {name}")
                            try:
                                df = pd.read_csv(io.BytesIO(zf.read(name)), dtype=str, low_memory=False)
                                if len(df) > 0:
                                    all_mappings.append(df)
                            except Exception as e:
                                log.warning(f"Could not parse CSV {name}: {e}")
            else:
                # Handle direct CSV files
                try:
                    df = pd.read_csv(io.BytesIO(resp.content), dtype=str, low_memory=False)
                    if len(df) > 0:
                        all_mappings.append(df)
                except Exception as e:
                    log.warning(f"Could not parse CSV {file_url}: {e}")
                    
        except Exception as e:
            log.warning(f"Could not download {file_url}: {e}")
            continue
    
    # Process and standardize CSV data
    if all_mappings:
        log.info(f"Found {len(all_mappings)} CSV files, processing...")
        combined_df = pd.concat(all_mappings, ignore_index=True)
        
        # Normalize column names
        combined_df.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in combined_df.columns]
        
        # Map to standard columns - try common column name variations
        standard_mapping = pd.DataFrame()
        
        # CIK mapping - matches SEC CSV format
        cik_cols = ["cik_number", "cik", "registrant_cik", "central_index_key", "company_cik"]
        for col in cik_cols:
            if col in combined_df.columns:
                standard_mapping["cik"] = combined_df[col].astype(str).str.strip().str.lstrip("0")
                break
        
        # Series ID mapping - matches SEC CSV format
        series_cols = ["series_id", "seriesid", "series_class_id", "dei_series_id", "series_identifier"]
        for col in series_cols:
            if col in combined_df.columns:
                standard_mapping["series_id"] = combined_df[col].astype(str).str.strip().str.upper()
                break
        
        # Class ID mapping - matches SEC CSV format
        class_cols = ["class_id", "class_contract_id", "classcontractid", "dei_class_contract_id", "class_identifier"]
        for col in class_cols:
            if col in combined_df.columns:
                standard_mapping["class_id"] = combined_df[col].astype(str).str.strip().str.upper()
                break
        
        # Optional fields - updated for SEC CSV format
        name_cols = ["series_name", "seriesname", "series_class_name"]
        for col in name_cols:
            if col in combined_df.columns:
                standard_mapping["series_name"] = combined_df[col].astype(str)
                break
                
        class_name_cols = ["class_name", "class_contract_name", "classname"]
        for col in class_name_cols:
            if col in combined_df.columns:
                standard_mapping["class_name"] = combined_df[col].astype(str)
                break
        
        ticker_cols = ["class_ticker", "ticker", "class_contract_ticker_symbol", "symbol"]
        for col in ticker_cols:
            if col in combined_df.columns:
                standard_mapping["ticker"] = combined_df[col].astype(str)
                break
        
        # Clean and validate
        if "cik" in standard_mapping.columns and "series_id" in standard_mapping.columns and "class_id" in standard_mapping.columns:
            # Remove rows with missing essential data
            standard_mapping = standard_mapping.dropna(subset=["cik", "series_id", "class_id"])
            # Remove empty strings
            standard_mapping = standard_mapping[
                (standard_mapping["cik"] != "") & 
                (standard_mapping["series_id"] != "") & 
                (standard_mapping["class_id"] != "")
            ]
            
            if len(standard_mapping) > 0:
                log.info(f"Successfully extracted {len(standard_mapping)} series/class mappings from CSV files")
                return standard_mapping.drop_duplicates().reset_index(drop=True)
        else:
            log.warning("Could not find required columns (cik, series_id, class_id) in CSV data")
    
    # Fallback to JSON (original logic)
    log.info("Falling back to JSON endpoint")
    try:
        log.info("Fetching series/class mapping from company_tickers_mf.json")
        r = requests.get(MF_TICKERS_JSON, headers=sec_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        
        rows = []
        
        # Parse the JSON structure - handle both dict and list formats
        if isinstance(data, dict):
            # Newer format: dict with numeric or string keys
            for key, item in data.items():
                if not isinstance(item, dict):
                    continue
                    
                cik = str(item.get("cik", "")).lstrip("0")
                ticker = item.get("ticker", "")
                title = item.get("title", "")
                
                # Try to extract series/class from title if structured
                # Some entries have series/class in title like "Series S000012345 Class C000067890"
                series_match = re.search(r"S\d{9}", str(title))
                class_match = re.search(r"C\d{9}", str(title))
                
                if series_match or class_match:
                    rows.append({
                        "cik": cik,
                        "series_id": series_match.group() if series_match else None,
                        "class_id": class_match.group() if class_match else None,
                        "ticker": ticker,
                        "fund_name": title
                    })
        elif isinstance(data, list):
            # List format
            for item in data:
                if not isinstance(item, dict):
                    continue
                    
                cik = str(item.get("cik", "")).lstrip("0")
                ticker = item.get("ticker", "")
                title = item.get("title", "")
                
                series_match = re.search(r"S\d{9}", str(title))
                class_match = re.search(r"C\d{9}", str(title))
                
                if series_match or class_match:
                    rows.append({
                        "cik": cik,
                        "series_id": series_match.group() if series_match else None,
                        "class_id": class_match.group() if class_match else None,
                        "ticker": ticker,
                        "fund_name": title
                    })
        
        if rows:
            mapping = pd.DataFrame(rows)
            log.info(f"Found {len(mapping)} series/class mappings from JSON")
            return mapping
            
    except Exception as e:
        log.warning(f"Failed to fetch from company_tickers_mf.json: {e}")
    
    # Fallback: try discovering bulk CSV/ZIP files
    try:
        log.info("Attempting to discover bulk series/class CSV files")
        resp = requests.get(SERIES_CLASS_LANDING, headers=sec_headers(), timeout=30)
        resp.raise_for_status()
        
        root = html.fromstring(resp.text)
        csv_links = []
        
        # Look for links containing "series" and "class"
        for a in root.xpath("//a[@href]"):
            href = a.get("href", "")
            text = "".join(a.itertext()).strip()
            if re.search(r"series.*class", href + text, re.I):
                if href.endswith((".csv", ".zip")):
                    full_url = href if href.startswith("http") else f"https://www.sec.gov{href}"
                    csv_links.append(full_url)
        
        # Try each CSV/ZIP file
        all_mappings = []
        for url in csv_links[:5]:  # Limit attempts
            try:
                log.info(f"Trying {url}")
                r = requests.get(url, headers=sec_headers(), timeout=60)
                r.raise_for_status()
                
                if url.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                        for name in zf.namelist():
                            if name.endswith(".csv"):
                                df = pd.read_csv(io.BytesIO(zf.read(name)), dtype=str)
                                all_mappings.append(df)
                else:
                    df = pd.read_csv(io.BytesIO(r.content), dtype=str)
                    all_mappings.append(df)
                    
            except Exception as e:
                log.debug(f"Failed to process {url}: {e}")
                continue
        
        if all_mappings:
            mapping = pd.concat(all_mappings, ignore_index=True)
            # Normalize column names
            mapping.columns = [c.lower().replace(" ", "_") for c in mapping.columns]
            
            # Map to standard columns
            standard_mapping = pd.DataFrame()
            standard_mapping["cik"] = mapping.get("cik", mapping.get("registrant_cik", "")).str.lstrip("0")
            standard_mapping["series_id"] = mapping.get("series_id", mapping.get("seriesid", ""))
            standard_mapping["class_id"] = mapping.get("class_contract_id", mapping.get("class_id", ""))
            standard_mapping["ticker"] = mapping.get("ticker", mapping.get("class_contract_ticker_symbol", ""))
            
            # Clean and return
            standard_mapping = standard_mapping.dropna(subset=["series_id", "class_id"])
            log.info(f"Found {len(standard_mapping)} series/class mappings from CSV")
            return standard_mapping
            
    except Exception as e:
        log.warning(f"Failed to fetch bulk CSV files: {e}")
    
    # Return empty DataFrame if all methods fail
    log.warning("Could not fetch series/class mapping from any source")
    return pd.DataFrame()


class SeriesClassMapper:
    """Maps series IDs to class IDs using SEC mapping data"""
    
    def __init__(self, cache_path: Optional[str] = None):
        """
        Initialize mapper with optional cache path for mapping data.
        
        Args:
            cache_path: Path to cache the mapping CSV (avoids repeated downloads)
        """
        self.cache_path = Path(cache_path) if cache_path else None
        self.mapping_df = None
        self._load_mapping()
    
    def _load_mapping(self):
        """Load or fetch the series/class mapping"""
        
        # Try loading from cache first
        if self.cache_path and self.cache_path.exists():
            log.info(f"Loading series/class mapping from cache: {self.cache_path}")
            self.mapping_df = pd.read_csv(self.cache_path, dtype=str)
            return
        
        # Fetch fresh mapping
        self.mapping_df = fetch_series_class_mapping()
        
        # Cache if path provided
        if self.cache_path and not self.mapping_df.empty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.mapping_df.to_csv(self.cache_path, index=False)
            log.info(f"Cached mapping to {self.cache_path}")
    
    def get_class_for_series(self, series_id: str) -> Optional[List[str]]:
        """
        Get class IDs associated with a series ID.
        
        Args:
            series_id: Series ID (e.g., "S000006037")
            
        Returns:
            List of class IDs or None if not found
        """
        if self.mapping_df is None or self.mapping_df.empty:
            return None
        
        # Ensure series_id format matches
        series_id = str(series_id).upper().strip()
        
        matches = self.mapping_df[self.mapping_df["series_id"] == series_id]
        if matches.empty:
            return None
        
        class_ids = matches["class_id"].dropna().unique().tolist()
        return class_ids if class_ids else None
    
    def get_series_for_class(self, class_id: str) -> Optional[str]:
        """
        Get series ID for a class ID.
        
        Args:
            class_id: Class ID (e.g., "C000016596")
            
        Returns:
            Series ID or None if not found
        """
        if self.mapping_df is None or self.mapping_df.empty:
            return None
        
        # Ensure class_id format matches
        class_id = str(class_id).upper().strip()
        
        matches = self.mapping_df[self.mapping_df["class_id"] == class_id]
        if matches.empty:
            return None
        
        series_id = matches["series_id"].iloc[0]
        return series_id if pd.notna(series_id) else None
    
    def map_series_to_classes(self, df: pd.DataFrame, series_col: str = "series") -> pd.DataFrame:
        """
        Add class_id column(s) to a DataFrame based on series_id.
        
        Args:
            df: DataFrame with series column
            series_col: Name of the series column
            
        Returns:
            DataFrame with added class_id information
        """
        if self.mapping_df is None or self.mapping_df.empty:
            log.warning("No mapping data available")
            return df
        
        # Ensure series format matches
        df = df.copy()
        df[series_col] = df[series_col].astype(str).str.upper().str.strip()
        
        # Merge with mapping
        merged = df.merge(
            self.mapping_df[["series_id", "class_id"]].drop_duplicates(),
            left_on=series_col,
            right_on="series_id",
            how="left"
        )
        
        # If multiple classes per series, we'll get duplicate rows
        # For now, take the first class per series (can be enhanced based on requirements)
        if "class_id" in merged.columns:
            merged = merged.groupby(merged.columns.difference(["class_id"]).tolist()).first().reset_index()
        
        return merged
    
    def build_series_class_lookup(self) -> Dict[str, List[str]]:
        """
        Build a dictionary lookup of series -> [class_ids].
        
        Returns:
            Dictionary mapping series IDs to lists of class IDs
        """
        if self.mapping_df is None or self.mapping_df.empty:
            return {}
        
        lookup = {}
        for series_id, group in self.mapping_df.groupby("series_id"):
            class_ids = group["class_id"].dropna().unique().tolist()
            if class_ids:
                lookup[str(series_id)] = class_ids
        
        return lookup
    
    def get_mapping_stats(self) -> dict:
        """Get statistics about the mapping data"""
        if self.mapping_df is None or self.mapping_df.empty:
            return {"status": "no_data"}
        
        return {
            "total_mappings": len(self.mapping_df),
            "unique_ciks": self.mapping_df["cik"].nunique(),
            "unique_series": self.mapping_df["series_id"].nunique(),
            "unique_classes": self.mapping_df["class_id"].nunique(),
            "series_with_multiple_classes": (
                self.mapping_df.groupby("series_id")["class_id"]
                .nunique()
                .gt(1)
                .sum()
            ),
        }


def integrate_turnover_with_series_mapping(
    num_df: pd.DataFrame,
    mapper: SeriesClassMapper
) -> pd.DataFrame:
    """
    Map turnover data from series level to class level using series/class mapping.
    
    Args:
        num_df: DataFrame with turnover data at series level
        mapper: SeriesClassMapper instance
        
    Returns:
        DataFrame with turnover mapped to class level
    """
    # Filter to turnover data
    turnover_df = num_df[num_df["tag"] == "PortfolioTurnoverRate"].copy()
    
    if turnover_df.empty:
        log.warning("No turnover data found")
        return pd.DataFrame()
    
    # Build series to class lookup
    series_class_lookup = mapper.build_series_class_lookup()
    
    # Map each series to its classes
    mapped_rows = []
    for _, row in turnover_df.iterrows():
        series_id = row.get("series")
        if pd.isna(series_id):
            continue
            
        series_id = str(series_id).upper().strip()
        class_ids = series_class_lookup.get(series_id, [])
        
        if class_ids:
            # Create a row for each class
            for class_id in class_ids:
                new_row = row.copy()
                new_row["class"] = class_id
                new_row["mapped_from_series"] = series_id
                mapped_rows.append(new_row)
        else:
            # Keep original row if no mapping found
            row["mapped_from_series"] = None
            mapped_rows.append(row)
    
    if mapped_rows:
        result = pd.DataFrame(mapped_rows)
        log.info(f"Mapped {len(turnover_df)} series-level turnover records to {len(result)} class-level records")
        return result
    else:
        log.warning("No series could be mapped to classes")
        return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the mapper
    mapper = SeriesClassMapper(cache_path="data/series_class_mapping.csv")
    
    # Print stats
    stats = mapper.get_mapping_stats()
    print("\nMapping Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test lookups
    test_series = "S000006037"
    classes = mapper.get_class_for_series(test_series)
    print(f"\nClasses for series {test_series}: {classes}")
    
    if classes:
        test_class = classes[0]
        series = mapper.get_series_for_class(test_class)
        print(f"Series for class {test_class}: {series}")