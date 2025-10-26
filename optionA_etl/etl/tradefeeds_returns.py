#!/usr/bin/env python3
"""
Tradefeeds API integration for fetching monthly returns.
Replaces SEC N-PORT-P filing downloads with direct API access.
"""

import pandas as pd
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import time

from .tradefeeds_client import TradeFeedsClient
# SeriesClassMapper is optional - only used for series/class mapping
try:
    from .series_class_mapper import SeriesClassMapper
except ImportError:
    SeriesClassMapper = None

log = logging.getLogger(__name__)


class TradeFeedsReturnsFetcher:
    """Fetch monthly returns from Tradefeeds API."""
    
    def __init__(self, api_client: TradeFeedsClient = None):
        """Initialize with API client."""
        self.client = api_client or TradeFeedsClient()
        self.mapper = SeriesClassMapper() if SeriesClassMapper else None
        
        # Create ticker lookup for series IDs
        self._ticker_lookup = self._build_ticker_lookup()
    
    def _build_ticker_lookup(self) -> dict:
        """
        Build lookup dictionary from series_id to ticker symbols.
        Structure: series_id -> class_id -> ticker
        """
        if not self.mapper:
            log.warning("No series/class mapper available for ticker lookup")
            return {}
        
        # Get mapper data (SeriesClassMapper uses mapping_df attribute)
        mapper_data = None
        if hasattr(self.mapper, 'mapping_df') and self.mapper.mapping_df is not None:
            mapper_data = self.mapper.mapping_df
        elif hasattr(self.mapper, 'data') and self.mapper.data is not None:
            mapper_data = self.mapper.data
        
        if mapper_data is None or mapper_data.empty:
            log.warning("No mapping data available in SeriesClassMapper")
            return {}
        
        try:
            if 'ticker' not in mapper_data.columns:
                log.warning("No ticker column found in mapper data")
                return {}
            
            # Create lookup of series_id to list of (class_id, ticker) pairs
            series_to_tickers = {}
            
            for series_id in mapper_data['series_id'].unique():
                # Find all class_ids for this series_id
                series_data = mapper_data[mapper_data['series_id'] == series_id]
                
                tickers_for_series = []
                for _, row in series_data.iterrows():
                    class_id = row['class_id']
                    ticker = row['ticker']
                    
                    # Only include valid tickers
                    if pd.notna(ticker) and str(ticker) not in ['nan', 'None', '']:
                        tickers_for_series.append({
                            'class_id': class_id,
                            'ticker': str(ticker)
                        })
                
                if tickers_for_series:
                    series_to_tickers[series_id] = tickers_for_series
            
            log.info(f"Built ticker lookup for {len(series_to_tickers)} series")
            return series_to_tickers
            
        except Exception as e:
            log.error(f"Error building ticker lookup: {e}")
            return {}
    
    def fetch_returns_for_series(
        self,
        series_id: str,
        date_from: str,
        date_to: str,
        cik: str = None
    ) -> pd.DataFrame:
        """
        Fetch monthly returns for a single series using OHLCV data.
        
        Args:
            series_id: SEC series ID (e.g., 'S000002594')
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            cik: Optional CIK for metadata
        
        Returns:
            DataFrame with monthly returns for all classes in the series
        """
        log.info(f"Fetching returns for series {series_id} from {date_from} to {date_to}")
        
        # Get ticker information for this series
        if series_id not in self._ticker_lookup:
            log.warning(f"No ticker mapping found for series {series_id}")
            return pd.DataFrame()
        
        ticker_info = self._ticker_lookup[series_id]
        log.info(f"Found {len(ticker_info)} class/ticker pairs for series {series_id}")
        
        all_returns = []
        
        # Fetch data for each class/ticker in this series
        for class_ticker in ticker_info:
            class_id = class_ticker['class_id']
            ticker = class_ticker['ticker']
            
            try:
                # Fetch OHLCV data for this ticker
                df = self.client.get_mutual_fund_returns(
                    series_id=series_id,
                    ticker=ticker,
                    date_from=date_from,
                    date_to=date_to
                )
                
                if not df.empty:
                    # Add class_id to distinguish between share classes
                    df['class_id'] = class_id
                    df['ticker'] = ticker
                    all_returns.append(df)
                    log.info(f"Retrieved {len(df)} monthly returns for {ticker} (class {class_id})")
                else:
                    log.warning(f"No data returned for ticker {ticker} (class {class_id})")
                    
            except Exception as e:
                log.error(f"Error fetching data for ticker {ticker} (class {class_id}): {e}")
                continue
        
        if not all_returns:
            log.warning(f"No data retrieved for any ticker in series {series_id}")
            return pd.DataFrame()
        
        # Combine all returns
        combined_df = pd.concat(all_returns, ignore_index=True)
        
        # Add CIK if provided
        if cik:
            combined_df['cik'] = str(int(cik))  # Remove leading zeros
        
        # Add filing_date as report_date for compatibility
        if 'report_date' in combined_df.columns:
            combined_df['filing_date'] = combined_df['report_date']
        
        log.info(f"Retrieved {len(combined_df)} total monthly returns for series {series_id}")
        return combined_df
    
    def fetch_flow_data_hybrid(
        self,
        pilot_config: Dict,
        date_from: str,
        date_to: str,
        sec_flow_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Fetch flow data using hybrid approach: SEC N-PORT primary, Tradefeeds fallback.
        
        Args:
            pilot_config: Configuration dict with registrants
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            sec_flow_data: Existing SEC N-PORT flow data (if available)
        
        Returns:
            Combined DataFrame with flow data from both sources
        """
        log.info("Fetching flow data using hybrid approach (SEC primary, Tradefeeds fallback)")
        
        all_flow_data = []
        
        # Use SEC data if provided
        if sec_flow_data is not None and not sec_flow_data.empty:
            log.info(f"Using {len(sec_flow_data)} SEC N-PORT flow records as primary source")
            all_flow_data.append(sec_flow_data)
        
        # Identify gaps and fetch from Tradefeeds as fallback
        for registrant in pilot_config['registrants']:
            name = registrant['name']
            series_ids = registrant['series_ids']
            
            for series_id in series_ids:
                try:
                    # Check if we have SEC data for this series in the date range
                    has_sec_data = False
                    if sec_flow_data is not None and not sec_flow_data.empty:
                        sec_series_data = sec_flow_data[
                            (sec_flow_data['series_id'] == series_id) &
                            (sec_flow_data['month_end'] >= pd.to_datetime(date_from)) &
                            (sec_flow_data['month_end'] <= pd.to_datetime(date_to))
                        ]
                        has_sec_data = not sec_series_data.empty
                    
                    if not has_sec_data:
                        log.info(f"No SEC flow data for {series_id}, fetching from Tradefeeds as fallback")
                        
                        # Fetch flow data from Tradefeeds mutual fund API
                        tradefeeds_flows = self.client.get_mutual_fund_flow_data(
                            series_id=series_id,
                            date_from=date_from,
                            date_to=date_to
                        )
                        
                        if not tradefeeds_flows.empty:
                            # Add source metadata
                            tradefeeds_flows['data_source'] = 'tradefeeds'
                            all_flow_data.append(tradefeeds_flows)
                            log.info(f"Retrieved {len(tradefeeds_flows)} flow records from Tradefeeds for {series_id}")
                        else:
                            log.warning(f"No flow data available from Tradefeeds for {series_id}")
                    else:
                        log.info(f"Using SEC N-PORT flow data for {series_id}")
                        
                except Exception as e:
                    log.error(f"Error fetching flow data for {series_id}: {e}")
                    continue
        
        if all_flow_data:
            combined_df = pd.concat(all_flow_data, ignore_index=True)
            
            # Ensure proper data types
            if 'month_end' in combined_df.columns:
                combined_df['month_end'] = pd.to_datetime(combined_df['month_end'])
            if 'date' in combined_df.columns:
                combined_df['date'] = pd.to_datetime(combined_df['date'])
            
            # Ensure numeric columns
            flow_cols = ['sales', 'redemptions', 'reinvest']
            for col in flow_cols:
                if col in combined_df.columns:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
            
            # Remove duplicates (prefer SEC data over Tradefeeds)
            if 'data_source' in combined_df.columns:
                # Sort so SEC data comes first, then drop duplicates
                combined_df['source_priority'] = combined_df['data_source'].map({'sec': 0, 'tradefeeds': 1}).fillna(1)
                combined_df = combined_df.sort_values(['class_id', 'month_end', 'source_priority'])
                combined_df = combined_df.drop_duplicates(subset=['class_id', 'month_end'], keep='first')
                combined_df = combined_df.drop(columns=['source_priority'])
            
            log.info(f"Total flow data records: {len(combined_df)}")
            return combined_df
        else:
            log.warning("No flow data retrieved from any source")
            return pd.DataFrame()
    
    def fetch_returns_for_pilot_funds(
        self,
        pilot_config: Dict,
        date_from: str,
        date_to: str
    ) -> pd.DataFrame:
        """
        Fetch monthly returns for all pilot funds.
        
        Args:
            pilot_config: Configuration dict with registrants
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
        
        Returns:
            Combined DataFrame with all fund returns
        """
        all_returns = []
        
        for registrant in pilot_config['registrants']:
            name = registrant['name']
            cik = registrant['cik']
            series_ids = registrant['series_ids']
            
            log.info(f"Processing {name} (CIK: {cik})")
            
            for series_id in series_ids:
                try:
                    # Fetch returns for this series
                    df = self.fetch_returns_for_series(
                        series_id=series_id,
                        date_from=date_from,
                        date_to=date_to,
                        cik=cik
                    )
                    
                    if not df.empty:
                        df['registrant_name'] = name
                        all_returns.append(df)
                    
                    # Small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    log.error(f"Failed to fetch returns for {series_id}: {e}")
                    continue
        
        if all_returns:
            combined_df = pd.concat(all_returns, ignore_index=True)
            log.info(f"Total monthly returns fetched: {len(combined_df)}")
            
            # Ensure proper data types
            if 'month_end' in combined_df.columns:
                combined_df['month_end'] = pd.to_datetime(combined_df['month_end'])
            if 'date' in combined_df.columns:
                combined_df['date'] = pd.to_datetime(combined_df['date'])
            if 'return' in combined_df.columns:
                combined_df['return'] = pd.to_numeric(combined_df['return'], errors='coerce')
            
            return combined_df
        else:
            log.warning("No returns data fetched from Tradefeeds")
            return pd.DataFrame()
    
    def fetch_with_chunking(
        self,
        pilot_config: Dict,
        date_from: str,
        date_to: str,
        chunk_months: int = 12
    ) -> pd.DataFrame:
        """
        Fetch returns in chunks to handle large date ranges.
        
        Args:
            pilot_config: Configuration dict with registrants
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            chunk_months: Number of months per chunk
        
        Returns:
            Combined DataFrame with all fund returns
        """
        all_chunks = []
        
        start_date = pd.to_datetime(date_from)
        end_date = pd.to_datetime(date_to)
        
        # Create date chunks
        current_start = start_date
        while current_start < end_date:
            current_end = min(
                current_start + pd.DateOffset(months=chunk_months),
                end_date
            )
            
            log.info(f"Fetching chunk: {current_start.date()} to {current_end.date()}")
            
            chunk_df = self.fetch_returns_for_pilot_funds(
                pilot_config=pilot_config,
                date_from=str(current_start.date()),
                date_to=str(current_end.date())
            )
            
            if not chunk_df.empty:
                all_chunks.append(chunk_df)
            
            current_start = current_end + pd.DateOffset(days=1)
        
        if all_chunks:
            combined_df = pd.concat(all_chunks, ignore_index=True)
            
            # Remove duplicates (keep most recent fetch)
            if 'class_id' in combined_df.columns and 'month_end' in combined_df.columns:
                # Sort by available columns for consistent deduplication
                sort_cols = ['class_id', 'month_end']
                if 'filing_date' in combined_df.columns:
                    sort_cols.append('filing_date')
                elif 'report_date' in combined_df.columns:
                    sort_cols.append('report_date')
                
                combined_df = combined_df.sort_values(sort_cols)
                combined_df = combined_df.drop_duplicates(
                    subset=['class_id', 'month_end'],
                    keep='last'
                )
            
            log.info(f"Total unique monthly returns: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the returns data.
        
        Args:
            df: DataFrame with monthly returns
        
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        initial_count = len(df)
        
        # Remove invalid returns
        if 'return' in df.columns:
            # Flag extreme returns
            extreme_returns = df[df['return'].abs() > 0.5]
            if len(extreme_returns) > 0:
                log.warning(f"Found {len(extreme_returns)} extreme returns (>50%)")
                for _, row in extreme_returns.iterrows():
                    log.warning(f"  {row['class_id']} {row['month_end']}: {row['return']:.1%}")
            
            # Remove null returns
            df = df[df['return'].notna()]
        
        # Ensure required columns
        required_columns = ['class_id', 'month_end', 'return']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            log.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        final_count = len(df)
        if final_count < initial_count:
            log.info(f"Data validation: {initial_count} → {final_count} records")
        
        return df