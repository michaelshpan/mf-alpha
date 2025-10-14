"""
SEC Bulk Data Downloader for N-PORT Monthly Returns
Downloads and processes monthly return data from SEC's bulk data service
"""

import logging
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import json

log = logging.getLogger(__name__)

class SECBulkDataDownloader:
    """
    Downloads and processes N-PORT bulk data from SEC's free data service
    """
    
    # SEC bulk data URLs
    BASE_URL = "https://www.sec.gov/data-research/sec-markets-data"
    NPORT_DATA_URL = "https://www.sec.gov/files/structureddata/data/form-n-port-data-sets"
    
    # Rate limiting
    RATE_LIMIT_DELAY = 0.15  # 150ms between requests (well under 10 req/sec limit)
    
    def __init__(self, cache_dir: str = "data/sec_bulk_cache", 
                 user_agent: str = "Research User research@example.com"):
        """
        Initialize downloader with cache directory and user agent
        
        Args:
            cache_dir: Directory to cache downloaded files
            user_agent: User agent string (required by SEC)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        # Cache for processed data
        self.returns_cache_file = self.cache_dir / "monthly_returns_cache.parquet"
        self.metadata_cache_file = self.cache_dir / "download_metadata.json"
        
    def get_available_quarters(self) -> List[str]:
        """
        Get list of available N-PORT data quarters from SEC website
        
        Returns:
            List of quarter strings (e.g., ['2023q1', '2023q2', ...])
        """
        # SEC provides quarterly N-PORT datasets
        # Pattern: https://www.sec.gov/files/structureddata/data/form-n-port-data-sets/2023q4_form_nport.zip
        
        current_year = datetime.now().year
        current_quarter = (datetime.now().month - 1) // 3 + 1
        
        quarters = []
        
        # Start from 2019 Q3 (when N-PORT reporting began)
        for year in range(2019, current_year + 1):
            for quarter in range(1, 5):
                # Skip future quarters
                if year == current_year and quarter > current_quarter:
                    break
                    
                # N-PORT started in 2019 Q3
                if year == 2019 and quarter < 3:
                    continue
                    
                quarters.append(f"{year}q{quarter}")
        
        return quarters
    
    def download_quarter_data(self, quarter: str, force_download: bool = False) -> Optional[Path]:
        """
        Download N-PORT bulk data for a specific quarter
        
        Args:
            quarter: Quarter string (e.g., '2023q4')
            force_download: Force re-download even if cached
            
        Returns:
            Path to downloaded zip file or None if failed
        """
        zip_filename = f"{quarter}_form_nport.zip"
        zip_path = self.cache_dir / zip_filename
        
        # Check if already downloaded
        if zip_path.exists() and not force_download:
            log.info(f"Using cached file: {zip_path}")
            return zip_path
        
        # Construct download URL
        url = f"{self.NPORT_DATA_URL}/{zip_filename}"
        
        log.info(f"Downloading N-PORT data for {quarter} from {url}")
        
        try:
            # Rate limiting
            time.sleep(self.RATE_LIMIT_DELAY)
            
            response = requests.get(url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            # Save to file
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            log.info(f"Downloaded {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
            return zip_path
            
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to download {quarter}: {e}")
            return None
    
    def extract_monthly_returns(self, zip_path: Path, 
                               target_ciks: Optional[List[str]] = None,
                               target_series: Optional[List[str]] = None,
                               target_classes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract monthly return data from N-PORT zip file
        
        Args:
            zip_path: Path to N-PORT zip file
            target_ciks: Optional list of CIKs to filter
            target_series: Optional list of series IDs to filter
            target_classes: Optional list of class IDs to filter
            
        Returns:
            DataFrame with monthly return data
        """
        log.info(f"Extracting monthly returns from {zip_path}")
        
        returns_data = []
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            # N-PORT bulk data typically contains these files:
            # - sub.txt: Submission metadata (CIK, filing date, etc.)
            # - num.txt: Numeric data including returns
            # - tag.txt: Tag definitions
            
            # Read submission metadata
            if 'sub.txt' in z.namelist():
                with z.open('sub.txt') as f:
                    sub_df = pd.read_csv(f, sep='\t', low_memory=False)
                    log.info(f"Found {len(sub_df)} submissions")
            
            # Read numeric data (contains returns)
            if 'num.txt' in z.namelist():
                with z.open('num.txt') as f:
                    num_df = pd.read_csv(f, sep='\t', low_memory=False)
                    
                    # Filter for monthly return tags
                    # N-PORT uses tags like:
                    # - MonthlyReturn1, MonthlyReturn2, MonthlyReturn3
                    # - Or specific class return tags
                    
                    return_tags = num_df[num_df['tag'].str.contains('MonthlyReturn|monthlyTotReturn', 
                                                                    case=False, na=False)]
                    
                    if not return_tags.empty:
                        log.info(f"Found {len(return_tags)} monthly return records")
                        
                        # Merge with submission data for CIK info
                        if 'sub_df' in locals():
                            return_tags = return_tags.merge(sub_df[['adsh', 'cik', 'period', 'filed']], 
                                                           on='adsh', how='left')
                        
                        # Process each return record
                        for _, row in return_tags.iterrows():
                            # Apply filters if specified
                            if target_ciks and str(row.get('cik')) not in target_ciks:
                                continue
                            
                            # Extract return data
                            return_record = {
                                'cik': str(row.get('cik', '')).zfill(10),
                                'filing_date': pd.to_datetime(row.get('filed')),
                                'period': pd.to_datetime(row.get('period')),
                                'tag': row.get('tag'),
                                'value': float(row.get('value', 0)) / 100.0,  # Convert percentage
                                'class_id': row.get('instance', ''),  # May need adjustment
                                'series_id': row.get('series', ''),  # May need extraction
                            }
                            
                            # Determine which month this return represents
                            if 'MonthlyReturn1' in str(row.get('tag')):
                                return_record['month_offset'] = 0  # Current month
                            elif 'MonthlyReturn2' in str(row.get('tag')):
                                return_record['month_offset'] = -1  # Previous month
                            elif 'MonthlyReturn3' in str(row.get('tag')):
                                return_record['month_offset'] = -2  # Two months ago
                            
                            returns_data.append(return_record)
        
        if returns_data:
            df = pd.DataFrame(returns_data)
            
            # Calculate actual month_end for each return
            if 'month_offset' in df.columns:
                df['month_end'] = df.apply(
                    lambda x: (x['period'] + pd.DateOffset(months=x['month_offset'])) + pd.offsets.MonthEnd(0),
                    axis=1
                )
            else:
                df['month_end'] = df['period'] + pd.offsets.MonthEnd(0)
            
            log.info(f"Extracted {len(df)} monthly return records")
            return df
        
        return pd.DataFrame()
    
    def build_returns_database(self, quarters: Optional[List[str]] = None,
                             target_ciks: Optional[List[str]] = None,
                             target_series: Optional[List[str]] = None,
                             target_classes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Build comprehensive monthly returns database from SEC bulk data
        
        Args:
            quarters: List of quarters to download (None for all available)
            target_ciks: Optional list of CIKs to filter
            target_series: Optional list of series IDs to filter
            target_classes: Optional list of class IDs to filter
            
        Returns:
            DataFrame with all monthly returns
        """
        if quarters is None:
            quarters = self.get_available_quarters()
        
        log.info(f"Building returns database for {len(quarters)} quarters")
        
        all_returns = []
        
        for quarter in quarters:
            log.info(f"Processing {quarter}")
            
            # Download quarter data
            zip_path = self.download_quarter_data(quarter)
            if not zip_path:
                log.warning(f"Skipping {quarter} - download failed")
                continue
            
            # Extract returns
            quarter_returns = self.extract_monthly_returns(
                zip_path, 
                target_ciks=target_ciks,
                target_series=target_series,
                target_classes=target_classes
            )
            
            if not quarter_returns.empty:
                all_returns.append(quarter_returns)
                log.info(f"Added {len(quarter_returns)} returns from {quarter}")
        
        if all_returns:
            # Combine all quarters
            combined_df = pd.concat(all_returns, ignore_index=True)
            
            # Deduplicate (keep most recent filing for each fund-month)
            combined_df = combined_df.sort_values(['cik', 'class_id', 'month_end', 'filing_date'])
            combined_df = combined_df.drop_duplicates(
                subset=['cik', 'class_id', 'month_end'], 
                keep='last'
            )
            
            # Save to parquet database
            combined_df.to_parquet(self.returns_cache_file, index=False)
            log.info(f"Saved {len(combined_df)} unique monthly returns to {self.returns_cache_file}")
            
            # Save metadata
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'quarters_processed': quarters,
                'total_records': len(combined_df),
                'unique_ciks': combined_df['cik'].nunique(),
                'unique_classes': combined_df['class_id'].nunique(),
                'date_range': {
                    'min': combined_df['month_end'].min().isoformat(),
                    'max': combined_df['month_end'].max().isoformat()
                }
            }
            
            with open(self.metadata_cache_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return combined_df
        
        return pd.DataFrame()
    
    def load_cached_returns(self) -> Optional[pd.DataFrame]:
        """
        Load cached monthly returns from parquet database
        
        Returns:
            DataFrame with monthly returns or None if not cached
        """
        if self.returns_cache_file.exists():
            log.info(f"Loading cached returns from {self.returns_cache_file}")
            df = pd.read_parquet(self.returns_cache_file)
            
            # Load metadata
            if self.metadata_cache_file.exists():
                with open(self.metadata_cache_file, 'r') as f:
                    metadata = json.load(f)
                log.info(f"Cache info: {metadata.get('total_records')} records, "
                        f"last updated {metadata.get('last_updated')}")
            
            return df
        
        return None
    
    def get_fund_returns(self, class_id: str = None, 
                        cik: str = None,
                        start_date: str = None, 
                        end_date: str = None) -> pd.DataFrame:
        """
        Get monthly returns for specific fund(s)
        
        Args:
            class_id: Fund class ID
            cik: Fund CIK
            start_date: Start date for returns
            end_date: End date for returns
            
        Returns:
            DataFrame with monthly returns
        """
        # Load cached data
        df = self.load_cached_returns()
        if df is None:
            log.warning("No cached returns found. Run build_returns_database() first.")
            return pd.DataFrame()
        
        # Apply filters
        if class_id:
            df = df[df['class_id'] == class_id]
        if cik:
            df = df[df['cik'] == str(cik).zfill(10)]
        if start_date:
            df = df[df['month_end'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['month_end'] <= pd.to_datetime(end_date)]
        
        return df.sort_values(['month_end']).reset_index(drop=True)
    
    def update_database(self, force_full_update: bool = False) -> pd.DataFrame:
        """
        Update the returns database with latest data
        
        Args:
            force_full_update: Force download of all quarters
            
        Returns:
            Updated DataFrame
        """
        if force_full_update:
            # Full rebuild
            return self.build_returns_database()
        
        # Incremental update - get quarters not in cache
        if self.metadata_cache_file.exists():
            with open(self.metadata_cache_file, 'r') as f:
                metadata = json.load(f)
            processed_quarters = set(metadata.get('quarters_processed', []))
        else:
            processed_quarters = set()
        
        all_quarters = set(self.get_available_quarters())
        new_quarters = list(all_quarters - processed_quarters)
        
        if new_quarters:
            log.info(f"Updating with {len(new_quarters)} new quarters: {new_quarters}")
            
            # Load existing data
            existing_df = self.load_cached_returns()
            
            # Get new data
            new_df = self.build_returns_database(quarters=new_quarters)
            
            if existing_df is not None and not new_df.empty:
                # Combine and deduplicate
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.sort_values(['cik', 'class_id', 'month_end', 'filing_date'])
                combined_df = combined_df.drop_duplicates(
                    subset=['cik', 'class_id', 'month_end'], 
                    keep='last'
                )
                
                # Save updated database
                combined_df.to_parquet(self.returns_cache_file, index=False)
                log.info(f"Updated database to {len(combined_df)} records")
                
                return combined_df
        else:
            log.info("Database is up to date")
            return self.load_cached_returns()


def main():
    """
    Example usage and testing
    """
    logging.basicConfig(level=logging.INFO)
    
    # Initialize downloader
    downloader = SECBulkDataDownloader(
        cache_dir="data/sec_bulk_cache",
        user_agent="Research User research@university.edu"
    )
    
    # Get available quarters
    quarters = downloader.get_available_quarters()
    print(f"Available quarters: {quarters}")
    
    # Build database for recent quarters (for testing)
    recent_quarters = quarters[-4:]  # Last 4 quarters
    print(f"Building database for: {recent_quarters}")
    
    # Example CIKs (you would get these from your pilot config)
    target_ciks = [
        "0000793769",  # Example CIK
        "0001234567",  # Another example
    ]
    
    # Build database
    df = downloader.build_returns_database(
        quarters=recent_quarters,
        target_ciks=target_ciks
    )
    
    if not df.empty:
        print(f"\nDatabase built successfully:")
        print(f"Total records: {len(df)}")
        print(f"Unique CIKs: {df['cik'].nunique()}")
        print(f"Date range: {df['month_end'].min()} to {df['month_end'].max()}")
        print(f"\nSample data:")
        print(df.head())
    
    # Test retrieval
    fund_returns = downloader.get_fund_returns(
        cik=target_ciks[0],
        start_date="2023-01-01"
    )
    
    if not fund_returns.empty:
        print(f"\nReturns for CIK {target_ciks[0]}:")
        print(fund_returns[['month_end', 'value']].head(12))


if __name__ == "__main__":
    main()