"""
Monthly Returns Database Manager
Manages local parquet database of monthly fund returns from SEC bulk data
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

from .sec_bulk_downloader import SECBulkDataDownloader

log = logging.getLogger(__name__)

class MonthlyReturnsDatabase:
    """
    Manages a local parquet database of monthly fund returns
    Integrates SEC bulk data with pilot pipeline data
    """
    
    def __init__(self, db_path: str = "data/monthly_returns_db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to database directory
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Database files
        self.returns_file = self.db_path / "monthly_returns.parquet"
        self.metadata_file = self.db_path / "metadata.json"
        self.index_file = self.db_path / "fund_index.parquet"
        
        # SEC bulk downloader
        self.sec_downloader = SECBulkDataDownloader(
            cache_dir=self.db_path / "sec_cache",
            user_agent="ETL Pipeline research@example.com"
        )
        
        # Load metadata if exists
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load database metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'created': datetime.now().isoformat(),
            'last_updated': None,
            'total_records': 0,
            'funds': {},
            'date_range': {}
        }
    
    def _save_metadata(self) -> None:
        """Save database metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def initialize_from_pilot_config(self, pilot_config_path: str) -> None:
        """
        Initialize database using fund list from pilot configuration
        
        Args:
            pilot_config_path: Path to pilot YAML configuration
        """
        import yaml
        
        log.info(f"Initializing database from pilot config: {pilot_config_path}")
        
        # Load pilot configuration
        with open(pilot_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        registrants = config.get('registrants', [])
        
        # Extract CIKs, series, and class IDs
        target_ciks = []
        target_series = []
        target_classes = []
        
        for reg in registrants:
            if 'cik' in reg:
                # Ensure CIK is 10 digits, zero-padded
                cik = str(reg['cik']).zfill(10)
                target_ciks.append(cik)
                
                # Store fund info in metadata
                self.metadata['funds'][cik] = {
                    'name': reg.get('name', 'Unknown'),
                    'series_ids': reg.get('series_ids', []),
                    'class_ids': reg.get('class_ids', [])
                }
            
            if 'series_ids' in reg:
                target_series.extend(reg['series_ids'])
            
            if 'class_ids' in reg:
                target_classes.extend(reg['class_ids'])
        
        log.info(f"Found {len(target_ciks)} CIKs, {len(target_series)} series, {len(target_classes)} classes")
        
        # Build or update database for these funds
        self.build_database(
            target_ciks=target_ciks,
            target_series=target_series,
            target_classes=target_classes
        )
    
    def build_database(self, 
                      target_ciks: Optional[List[str]] = None,
                      target_series: Optional[List[str]] = None,
                      target_classes: Optional[List[str]] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Build or update the monthly returns database
        
        Args:
            target_ciks: List of CIKs to include
            target_series: List of series IDs to include
            target_classes: List of class IDs to include
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with monthly returns
        """
        log.info("Building monthly returns database from SEC bulk data")
        
        # Determine which quarters to download
        quarters = self.sec_downloader.get_available_quarters()
        
        # Filter quarters based on date range if specified
        if start_date or end_date:
            filtered_quarters = []
            for q in quarters:
                year = int(q[:4])
                quarter = int(q[-1])
                
                # Approximate quarter dates
                quarter_start = datetime(year, (quarter - 1) * 3 + 1, 1)
                quarter_end = datetime(year, quarter * 3, 1) if quarter < 4 else datetime(year, 12, 31)
                
                if start_date and quarter_end < pd.to_datetime(start_date):
                    continue
                if end_date and quarter_start > pd.to_datetime(end_date):
                    continue
                    
                filtered_quarters.append(q)
            
            quarters = filtered_quarters
        
        log.info(f"Processing {len(quarters)} quarters of data")
        
        # Download and process SEC bulk data
        df = self.sec_downloader.build_returns_database(
            quarters=quarters,
            target_ciks=target_ciks,
            target_series=target_series,
            target_classes=target_classes
        )
        
        if df.empty:
            log.warning("No data retrieved from SEC bulk downloads")
            return pd.DataFrame()
        
        # Standardize column names to match pilot pipeline
        df = df.rename(columns={
            'value': 'return',
            'period': 'report_period'
        })
        
        # Ensure proper data types
        df['month_end'] = pd.to_datetime(df['month_end'])
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        df['return'] = pd.to_numeric(df['return'], errors='coerce')
        
        # Remove duplicates (keep most recent filing)
        df = df.sort_values(['cik', 'class_id', 'month_end', 'filing_date'])
        df = df.drop_duplicates(subset=['cik', 'class_id', 'month_end'], keep='last')
        
        # Save to parquet database
        df.to_parquet(self.returns_file, index=False)
        log.info(f"Saved {len(df)} monthly returns to {self.returns_file}")
        
        # Update metadata
        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata['total_records'] = len(df)
        self.metadata['date_range'] = {
            'min': df['month_end'].min().isoformat() if not df.empty else None,
            'max': df['month_end'].max().isoformat() if not df.empty else None
        }
        self._save_metadata()
        
        # Create fund index for quick lookups
        self._create_fund_index(df)
        
        return df
    
    def _create_fund_index(self, df: pd.DataFrame) -> None:
        """Create index of funds for quick lookups"""
        index_data = []
        
        for (cik, class_id), group in df.groupby(['cik', 'class_id']):
            index_data.append({
                'cik': cik,
                'class_id': class_id,
                'series_id': group['series_id'].iloc[0] if 'series_id' in group.columns else None,
                'first_month': group['month_end'].min(),
                'last_month': group['month_end'].max(),
                'n_months': len(group),
                'avg_return': group['return'].mean()
            })
        
        if index_data:
            index_df = pd.DataFrame(index_data)
            index_df.to_parquet(self.index_file, index=False)
            log.info(f"Created fund index with {len(index_df)} fund-classes")
    
    def get_fund_returns(self, 
                        class_id: Optional[str] = None,
                        cik: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get monthly returns for specific fund(s)
        
        Args:
            class_id: Fund class ID
            cik: Fund CIK
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with monthly returns
        """
        if not self.returns_file.exists():
            log.warning("Database not initialized. Run build_database() first.")
            return pd.DataFrame()
        
        # Load returns data
        df = pd.read_parquet(self.returns_file)
        
        # Apply filters
        if class_id:
            df = df[df['class_id'] == class_id]
        if cik:
            df = df[df['cik'] == str(cik).zfill(10)]
        if start_date:
            df = df[df['month_end'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['month_end'] <= pd.to_datetime(end_date)]
        
        return df.sort_values('month_end').reset_index(drop=True)
    
    def get_complete_monthly_series(self, 
                                  class_id: str,
                                  start_date: str,
                                  end_date: str,
                                  fill_method: str = 'none') -> pd.DataFrame:
        """
        Get complete monthly return series for a fund, with optional gap filling
        
        Args:
            class_id: Fund class ID
            start_date: Start date
            end_date: End date
            fill_method: Method to fill gaps ('none', 'forward', 'interpolate', 'zero')
            
        Returns:
            DataFrame with complete monthly series
        """
        # Get available returns
        df = self.get_fund_returns(
            class_id=class_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return df
        
        # Create complete monthly index
        date_range = pd.date_range(
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date),
            freq='ME'
        )
        
        # Reindex to complete monthly series
        df = df.set_index('month_end')
        df = df.reindex(date_range)
        
        # Fill gaps if requested
        if fill_method == 'forward':
            df['return'] = df['return'].fillna(method='ffill')
        elif fill_method == 'interpolate':
            df['return'] = df['return'].interpolate(method='linear')
        elif fill_method == 'zero':
            df['return'] = df['return'].fillna(0.0)
        
        # Reset index
        df = df.reset_index()
        df = df.rename(columns={'index': 'month_end'})
        
        # Fill other columns
        df['class_id'] = class_id
        
        return df
    
    def merge_with_pilot_data(self, pilot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge database returns with pilot pipeline data
        Replaces sparse N-PORT data with complete monthly series
        
        Args:
            pilot_df: DataFrame from pilot pipeline
            
        Returns:
            Enhanced DataFrame with complete monthly returns
        """
        log.info("Merging database returns with pilot data")
        
        # Load database returns
        if not self.returns_file.exists():
            log.warning("Database not initialized. Returning original data.")
            return pilot_df
        
        db_returns = pd.read_parquet(self.returns_file)
        
        # Standardize columns for merge
        db_returns = db_returns[['cik', 'class_id', 'month_end', 'return']].copy()
        db_returns = db_returns.rename(columns={'return': 'db_return'})
        
        # Merge on class_id and month_end
        merged = pilot_df.merge(
            db_returns,
            on=['class_id', 'month_end'],
            how='left',
            suffixes=('', '_db')
        )
        
        # Use database return if available, otherwise keep original
        if 'db_return' in merged.columns:
            merged['return_source'] = 'pilot'
            merged.loc[merged['db_return'].notna(), 'return_source'] = 'database'
            
            # Replace return with database value where available
            merged.loc[merged['db_return'].notna(), 'return'] = merged.loc[merged['db_return'].notna(), 'db_return']
            
            # Log statistics
            n_replaced = (merged['return_source'] == 'database').sum()
            n_filled = merged['return'].notna().sum() - pilot_df['return'].notna().sum()
            
            log.info(f"Replaced {n_replaced} returns with database values")
            log.info(f"Filled {n_filled} previously missing returns")
            
            # Drop temporary columns
            merged = merged.drop(columns=['db_return', 'cik_db'])
        
        return merged
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {
            'initialized': self.returns_file.exists(),
            'metadata': self.metadata
        }
        
        if self.returns_file.exists():
            df = pd.read_parquet(self.returns_file)
            stats.update({
                'total_records': len(df),
                'unique_ciks': df['cik'].nunique(),
                'unique_classes': df['class_id'].nunique(),
                'date_range': {
                    'min': df['month_end'].min().isoformat(),
                    'max': df['month_end'].max().isoformat()
                },
                'coverage': {
                    'mean_months_per_fund': df.groupby('class_id')['month_end'].nunique().mean(),
                    'funds_with_gaps': (df.groupby('class_id')['month_end'].nunique() < 
                                      df['month_end'].nunique()).sum()
                }
            })
        
        return stats


def test_database():
    """Test the database functionality"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    db = MonthlyReturnsDatabase(db_path="data/test_returns_db")
    
    # Initialize from pilot config
    db.initialize_from_pilot_config("config/funds_pilot.yaml")
    
    # Get stats
    stats = db.get_database_stats()
    print(f"Database stats: {json.dumps(stats, indent=2, default=str)}")
    
    # Test retrieval
    returns = db.get_fund_returns(start_date="2023-01-01")
    if not returns.empty:
        print(f"\nSample returns:")
        print(returns.head())
    
    # Test complete series
    if not returns.empty:
        class_id = returns['class_id'].iloc[0]
        complete_series = db.get_complete_monthly_series(
            class_id=class_id,
            start_date="2023-01-01",
            end_date="2023-12-31",
            fill_method='none'
        )
        print(f"\nComplete series for {class_id}:")
        print(complete_series)


if __name__ == "__main__":
    test_database()