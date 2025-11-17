#!/usr/bin/env python3
"""
Tradefeeds.com API Client for fetching historical mutual fund returns.
Attempts to use CIK and series_ID directly before falling back to LEI.
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging
import time
from pathlib import Path
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

log = logging.getLogger(__name__)


class TradeFeedsClient:
    """Client for Tradefeeds.com API"""
    
    BASE_URL = "https://data.tradefeeds.com/api/v1"
    
    def __init__(self, api_key: str = None):
        """Initialize client with API key from environment or parameter."""
        self.api_key = api_key or os.getenv('TRADEFEEDS_API_KEY')
        if not self.api_key:
            raise ValueError("TRADEFEEDS_API_KEY not found in environment or parameters")
        
        self.session = requests.Session()
        # Note: Tradefeeds uses key parameter in URL, not Authorization header
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 1 request per 2 seconds to avoid quota limits
        
        # Setup API response logging
        self.debug_log_dir = Path("logs/tradefeeds_api_debug")
        self.debug_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug flag to disable date range extension
        self.disable_date_extension = os.getenv('TRADEFEEDS_DISABLE_DATE_EXTENSION', 'false').lower() == 'true'
        if self.disable_date_extension:
            log.info("TRADEFEEDS DEBUG MODE: Date range extension DISABLED - using original user dates")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _params_to_query(self, params: dict) -> str:
        """Convert params dict to query string for debugging."""
        from urllib.parse import urlencode
        return urlencode(params)
    
    def _log_api_response(self, endpoint: str, params: dict, response: requests.Response, ticker: str):
        """Log detailed API response for debugging date range issues."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'endpoint': endpoint,
            'ticker': ticker,
            'request_params': {
                'date_from': params.get('date_from'),
                'date_to': params.get('date_to'),
                'ticker': params.get('ticker'),
                'has_api_key': bool(params.get('key'))
            },
            'response_status': response.status_code,
            'response_headers': dict(response.headers),
            'response_size_bytes': len(response.content) if response.content else 0
        }
        
        # Parse response data if successful
        if response.status_code == 200:
            try:
                response_data = response.json()
                log_entry['response_data'] = response_data
                
                # Analyze data structure for debugging
                if isinstance(response_data, dict) and 'result' in response_data:
                    result = response_data['result']
                    if isinstance(result, dict) and 'output' in result:
                        output = result['output']
                        if isinstance(output, dict) and 'daily_stock_data' in output:
                            daily_data = output['daily_stock_data']
                            if daily_data and isinstance(daily_data, list):
                                log_entry['data_analysis'] = {
                                    'total_daily_records': len(daily_data),
                                    'date_range_actual': {
                                        'first_date': daily_data[0].get('date') if daily_data else None,
                                        'last_date': daily_data[-1].get('date') if daily_data else None
                                    },
                                    'sample_records': daily_data[:3] if len(daily_data) >= 3 else daily_data
                                }
            except Exception as e:
                log_entry['response_parse_error'] = str(e)
                log_entry['response_text'] = response.text[:1000]  # First 1000 chars for debugging
        else:
            log_entry['response_error'] = response.text
        
        # Save to timestamped log file
        log_file = self.debug_log_dir / f"api_response_{ticker}_{timestamp}.json"
        try:
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2, default=str)
            log.info(f"API response logged to {log_file}")
        except Exception as e:
            log.error(f"Failed to write API response log: {e}")
        
        # Also log key info to main logger
        if response.status_code == 200 and 'data_analysis' in log_entry:
            analysis = log_entry['data_analysis']
            log.info(f"TRADEFEEDS API DEBUG - {ticker}: {analysis['total_daily_records']} daily records "
                    f"from {analysis['date_range_actual']['first_date']} to {analysis['date_range_actual']['last_date']}")
            log.info(f"TRADEFEEDS API DEBUG - Requested: {params.get('date_from')} to {params.get('date_to')}")
        
        return log_entry
    
    def _chunk_date_range(self, start_date: str, end_date: str, chunk_months: int = 16) -> list:
        """
        Split a date range into chunks to respect API limitations.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_months: Maximum months per chunk (default: 16 = 1 year 4 months)
            
        Returns:
            List of (start_date, end_date) tuples for each chunk
        """
        import pandas as pd
        from dateutil.relativedelta import relativedelta
        
        chunks = []
        current_start = pd.to_datetime(start_date)
        target_end = pd.to_datetime(end_date)
        
        while current_start <= target_end:
            # Calculate chunk end date
            chunk_end = current_start + relativedelta(months=chunk_months) - pd.Timedelta(days=1)
            
            # Don't exceed target end date
            if chunk_end > target_end:
                chunk_end = target_end
            
            chunks.append((
                current_start.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            ))
            
            # Move to next chunk
            current_start = chunk_end + pd.Timedelta(days=1)
        
        return chunks
    
    def _get_mutual_fund_returns_direct(
        self,
        series_id: str = None,
        series_lei: str = None,
        date_from: str = None,
        date_to: str = None,
        ticker: str = None
    ) -> pd.DataFrame:
        """
        Direct API call for mutual fund returns WITHOUT auto-chunking detection.
        Used internally by chunking method to avoid infinite recursion.
        
        Args:
            series_id: SEC series ID (e.g., 'S000002594') - used for output metadata
            series_lei: LEI of the fund series - not used in OHLCV API
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            ticker: Mutual fund ticker symbol (e.g., 'VPMCX')
        
        Returns:
            DataFrame with monthly returns calculated from OHLCV data
        """
        if not ticker:
            raise ValueError("Ticker symbol must be provided for OHLCV API")
        
        # API endpoint for historical OHLCV
        endpoint = f"{self.BASE_URL}/historical_ohlcv"
        
        # Build params with API key and ticker
        params = {
            'key': self.api_key,
            'ticker': ticker
        }
        
        # Add date range - extend to quarterly boundaries for reliable API response (unless disabled)
        original_date_from = date_from
        
        if self.disable_date_extension:
            # Use original dates exactly as requested for debugging
            if date_from:
                params['date_from'] = date_from
                log.info(f"DEBUG MODE: Using original start date: {date_from}")
            if date_to:
                params['date_to'] = date_to 
                log.info(f"DEBUG MODE: Using original end date: {date_to}")
        else:
            # Standard date extension logic
            if date_from:
                from dateutil.relativedelta import relativedelta
                start_dt = pd.to_datetime(date_from)
                
                # Extend to start of quarter and back one additional quarter for return calculation
                # This ensures we have enough data for reliable monthly return calculation
                quarter_start = start_dt.replace(month=((start_dt.month-1)//3)*3+1, day=1)
                extended_start = quarter_start - relativedelta(months=3)
                extended_start_str = extended_start.strftime('%Y-%m-%d')
                
                params['date_from'] = extended_start_str
                log.debug(f"Extended start date from {date_from} to {extended_start_str} (quarterly boundaries for reliable API response)")
            if date_to:
                # Extend end date by 1-2 months to ensure target month is included
                # API appears to exclude last month(s) of requested range
                from dateutil.relativedelta import relativedelta
                end_dt = pd.to_datetime(date_to)
                extended_end = end_dt + relativedelta(months=2)
                extended_end_str = extended_end.strftime('%Y-%m-%d')
                params['date_to'] = extended_end_str
                log.debug(f"Extended end date from {date_to} to {extended_end_str} to ensure target month inclusion")

        # Make API call
        try:
            log.info(f"Fetching OHLCV data for ticker={ticker}, seriesId={series_id or 'N/A'}")
            log.debug(f"URL: {endpoint}?{self._params_to_query(params)}")
            
            self._rate_limit()
            
            # Add accept header
            headers = {'accept': 'application/json'}
            response = self.session.get(endpoint, params=params, headers=headers)
            
            # Log API response for debugging date range issues
            self._log_api_response(endpoint, params, response, ticker)
            
            if response.status_code == 200:
                data = response.json()
                
                # Debug: Log the actual response structure
                log.debug(f"OHLCV API response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                if isinstance(data, dict) and 'result' in data:
                    result = data['result']
                    if result is not None and isinstance(result, dict):
                        log.debug(f"Result keys: {list(result.keys())}")
                        if 'output' in result:
                            output = result['output']
                            log.debug(f"Output type: {type(output)}, keys: {list(output.keys()) if isinstance(output, dict) else 'Not dict'}")
                    else:
                        log.warning(f"Result is None or not dict: {type(result)}")
                
                # Parse OHLCV response structure
                if isinstance(data, dict) and 'result' in data and data['result'] is not None:
                    result = data['result']
                    
                    # Check for the expected structure: result.output.daily_stock_data
                    if isinstance(result, dict) and 'output' in result:
                        output = result['output']
                        if isinstance(output, dict) and 'daily_stock_data' in output:
                            daily_data = output['daily_stock_data']
                            
                            if daily_data and len(daily_data) > 0:
                                # Convert to DataFrame and calculate monthly returns
                                df = self._calculate_monthly_returns_from_ohlcv(
                                    daily_data, ticker, series_id
                                )
                                
                                if not df.empty:
                                    # Enhanced logging for debugging date range issues
                                    log.info(f"TRADEFEEDS CALCULATION: Generated {len(df)} monthly returns from OHLCV data")
                                    if not df.empty:
                                        df_start = df['month_end'].min()
                                        df_end = df['month_end'].max()
                                        log.info(f"TRADEFEEDS CALCULATION: Data range {df_start} to {df_end}")
                                    
                                    # Filter to original requested date range (unless disabled)
                                    if not self.disable_date_extension and original_date_from and date_to:
                                        start_dt = pd.to_datetime(original_date_from)
                                        end_dt = pd.to_datetime(date_to)
                                        df['month_end'] = pd.to_datetime(df['month_end'])
                                        filtered_df = df[
                                            (df['month_end'] >= start_dt) & 
                                            (df['month_end'] <= end_dt)
                                        ]
                                        log.info(f"TRADEFEEDS FILTERING: Requested range {original_date_from} to {date_to}")
                                        log.info(f"TRADEFEEDS FILTERING: Filtered {len(df)} → {len(filtered_df)} records for requested range")
                                        if len(filtered_df) != len(df):
                                            log.warning(f"TRADEFEEDS FILTERING: Lost {len(df) - len(filtered_df)} records in filtering!")
                                            # Log what was filtered out
                                            excluded = df[~((df['month_end'] >= start_dt) & (df['month_end'] <= end_dt))]
                                            if not excluded.empty:
                                                log.warning(f"TRADEFEEDS FILTERING: Excluded dates: {excluded['month_end'].tolist()}")
                                        return filtered_df
                                    else:
                                        if self.disable_date_extension:
                                            log.info(f"TRADEFEEDS DEBUG: No filtering applied - returning all {len(df)} records")
                                        return df
                                else:
                                    log.warning("No monthly returns could be calculated from OHLCV data")
                                    return pd.DataFrame()
                            else:
                                log.warning("No daily_stock_data found in response")
                                return pd.DataFrame()
                        else:
                            log.warning("Unexpected OHLCV response structure: missing output.daily_stock_data")
                            return pd.DataFrame()
                    else:
                        log.warning(f"Unexpected result structure: {type(result)}")
                        return pd.DataFrame()
                else:
                    log.warning("No 'result' key or empty result in OHLCV response")
                    log.debug(f"Full response for debugging: {data}")
                    return pd.DataFrame()
            
            elif response.status_code == 404:
                log.error(f"Ticker not found: {ticker}")
                return pd.DataFrame()
            
            else:
                log.error(f"OHLCV API returned {response.status_code}: {response.text}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            log.error(f"OHLCV API request failed: {e}")
            return pd.DataFrame()

    def get_mutual_fund_returns_chunked(
        self,
        series_id: str = None,
        series_lei: str = None,
        date_from: str = None,
        date_to: str = None,
        ticker: str = None,
        chunk_pause_seconds: float = 2.0
    ) -> pd.DataFrame:
        """
        Fetch returns using chunked requests to handle API limitations.
        
        Args:
            series_id: SEC series ID (e.g., 'S000002594') - used for output metadata
            series_lei: LEI of the fund series - not used in OHLCV API
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            ticker: Mutual fund ticker symbol (e.g., 'VPMCX')
            chunk_pause_seconds: Pause between chunks (default: 10.0)
        
        Returns:
            DataFrame with monthly returns calculated from OHLCV data
        """
        if not ticker:
            raise ValueError("Ticker symbol must be provided for OHLCV API")
        
        if not date_from or not date_to:
            raise ValueError("Both date_from and date_to must be provided")
        
        # Calculate chunks
        chunks = self._chunk_date_range(date_from, date_to, chunk_months=16)
        log.info(f"Chunking request for {ticker}: {len(chunks)} chunks from {date_from} to {date_to}")
        
        all_data = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            log.info(f"Fetching chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
            
            try:
                # FIXED: Use direct API method to avoid infinite recursion
                chunk_data = self._get_mutual_fund_returns_direct(
                    series_id=series_id,
                    series_lei=series_lei,
                    date_from=chunk_start,
                    date_to=chunk_end,
                    ticker=ticker
                )
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                    log.info(f"✓ Chunk {i+1}: Got {len(chunk_data)} monthly returns")
                else:
                    log.warning(f"⚠ Chunk {i+1}: No data returned")
                
                # Pause between chunks (except for last chunk)
                if i < len(chunks) - 1:
                    log.info(f"Pausing {chunk_pause_seconds} seconds before next chunk...")
                    time.sleep(chunk_pause_seconds)
                    
            except Exception as e:
                log.error(f"✗ Chunk {i+1} failed: {e}")
                # Continue with remaining chunks
                continue
        
        # Combine all chunks
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates that might occur at chunk boundaries
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['month_end'], keep='first')
            final_count = len(combined_df)
            
            if initial_count != final_count:
                log.info(f"Removed {initial_count - final_count} duplicate records at chunk boundaries")
            
            # Sort by date
            combined_df = combined_df.sort_values('month_end').reset_index(drop=True)
            
            log.info(f"✓ Combined chunked data: {len(combined_df)} total monthly returns")
            log.info(f"  Date range: {combined_df['month_end'].min()} to {combined_df['month_end'].max()}")
            
            return combined_df
        else:
            log.warning("No data retrieved from any chunks")
            return pd.DataFrame()
    
    def get_mutual_fund_returns(
        self,
        series_id: str = None,
        series_lei: str = None,
        date_from: str = None,
        date_to: str = None,
        ticker: str = None,
        use_chunking: bool = None
    ) -> pd.DataFrame:
        """
        Fetch returns using historical OHLCV data and calculate monthly returns.
        
        Args:
            series_id: SEC series ID (e.g., 'S000002594') - used for output metadata
            series_lei: LEI of the fund series - not used in OHLCV API
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            ticker: Mutual fund ticker symbol (e.g., 'VPMCX')
            use_chunking: Force chunking on/off. If None, auto-detect based on date range.
        
        Returns:
            DataFrame with monthly returns calculated from OHLCV data
        """
        if not ticker:
            raise ValueError("Ticker symbol must be provided for OHLCV API")
        
        # Auto-detect if chunking is needed for long date ranges
        if use_chunking is None and date_from and date_to:
            start_dt = pd.to_datetime(date_from)
            end_dt = pd.to_datetime(date_to)
            months_requested = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
            
            # Use chunking if more than 15 months requested (API limit is ~16 months)
            use_chunking = months_requested > 15
            
            if use_chunking:
                log.info(f"Auto-enabling chunking: {months_requested} months requested (>15 month API limit)")
        
        # Delegate to chunked method if chunking is enabled
        if use_chunking:
            return self.get_mutual_fund_returns_chunked(
                series_id=series_id,
                series_lei=series_lei,
                date_from=date_from,
                date_to=date_to,
                ticker=ticker
            )
        
        # API endpoint for historical OHLCV
        endpoint = f"{self.BASE_URL}/historical_ohlcv"
        
        # Build params with API key and ticker
        params = {
            'key': self.api_key,
            'ticker': ticker
        }
        
        # Add date range - extend to quarterly boundaries for reliable API response (unless disabled)
        original_date_from = date_from
        
        if self.disable_date_extension:
            # Use original dates exactly as requested for debugging
            if date_from:
                params['date_from'] = date_from
                log.info(f"DEBUG MODE: Using original start date: {date_from}")
            if date_to:
                params['date_to'] = date_to 
                log.info(f"DEBUG MODE: Using original end date: {date_to}")
        else:
            # Standard date extension logic
            if date_from:
                from dateutil.relativedelta import relativedelta
                start_dt = pd.to_datetime(date_from)
                
                # Extend to start of quarter and back one additional quarter for return calculation
                # This ensures we have enough data for reliable monthly return calculation
                quarter_start = start_dt.replace(month=((start_dt.month-1)//3)*3+1, day=1)
                extended_start = quarter_start - relativedelta(months=3)
                extended_start_str = extended_start.strftime('%Y-%m-%d')
                
                params['date_from'] = extended_start_str
                log.debug(f"Extended start date from {date_from} to {extended_start_str} (quarterly boundaries for reliable API response)")
            if date_to:
                # Extend end date by 1-2 months to ensure target month is included
                # API appears to exclude last month(s) of requested range
                from dateutil.relativedelta import relativedelta
                end_dt = pd.to_datetime(date_to)
                extended_end = end_dt + relativedelta(months=2)
                extended_end_str = extended_end.strftime('%Y-%m-%d')
                params['date_to'] = extended_end_str
                log.debug(f"Extended end date from {date_to} to {extended_end_str} to ensure target month inclusion")

        # Make API call
        try:
            log.info(f"Fetching OHLCV data for ticker={ticker}, seriesId={series_id or 'N/A'}")
            log.debug(f"URL: {endpoint}?{self._params_to_query(params)}")
            
            self._rate_limit()
            
            # Add accept header
            headers = {'accept': 'application/json'}
            response = self.session.get(endpoint, params=params, headers=headers)
            
            # Log API response for debugging date range issues
            self._log_api_response(endpoint, params, response, ticker)
            
            if response.status_code == 200:
                data = response.json()
                
                # Debug: Log the actual response structure
                log.debug(f"OHLCV API response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                if isinstance(data, dict) and 'result' in data:
                    result = data['result']
                    if result is not None and isinstance(result, dict):
                        log.debug(f"Result keys: {list(result.keys())}")
                        if 'output' in result:
                            output = result['output']
                            log.debug(f"Output type: {type(output)}, keys: {list(output.keys()) if isinstance(output, dict) else 'Not dict'}")
                    else:
                        log.warning(f"Result is None or not dict: {type(result)}")
                
                # Parse OHLCV response structure
                if isinstance(data, dict) and 'result' in data and data['result'] is not None:
                    result = data['result']
                    
                    # Check for the expected structure: result.output.daily_stock_data
                    if isinstance(result, dict) and 'output' in result:
                        output = result['output']
                        if isinstance(output, dict) and 'daily_stock_data' in output:
                            daily_data = output['daily_stock_data']
                            
                            if daily_data and len(daily_data) > 0:
                                # Convert to DataFrame and calculate monthly returns
                                df = self._calculate_monthly_returns_from_ohlcv(
                                    daily_data, ticker, series_id
                                )
                                
                                if not df.empty:
                                    # Enhanced logging for debugging date range issues
                                    log.info(f"TRADEFEEDS CALCULATION: Generated {len(df)} monthly returns from OHLCV data")
                                    if not df.empty:
                                        df_start = df['month_end'].min()
                                        df_end = df['month_end'].max()
                                        log.info(f"TRADEFEEDS CALCULATION: Data range {df_start} to {df_end}")
                                    
                                    # Filter to original requested date range (unless disabled)
                                    if not self.disable_date_extension and original_date_from and date_to:
                                        start_dt = pd.to_datetime(original_date_from)
                                        end_dt = pd.to_datetime(date_to)
                                        df['month_end'] = pd.to_datetime(df['month_end'])
                                        filtered_df = df[
                                            (df['month_end'] >= start_dt) & 
                                            (df['month_end'] <= end_dt)
                                        ]
                                        log.info(f"TRADEFEEDS FILTERING: Requested range {original_date_from} to {date_to}")
                                        log.info(f"TRADEFEEDS FILTERING: Filtered {len(df)} → {len(filtered_df)} records for requested range")
                                        if len(filtered_df) != len(df):
                                            log.warning(f"TRADEFEEDS FILTERING: Lost {len(df) - len(filtered_df)} records in filtering!")
                                            # Log what was filtered out
                                            excluded = df[~((df['month_end'] >= start_dt) & (df['month_end'] <= end_dt))]
                                            if not excluded.empty:
                                                log.warning(f"TRADEFEEDS FILTERING: Excluded dates: {excluded['month_end'].tolist()}")
                                        return filtered_df
                                    else:
                                        if self.disable_date_extension:
                                            log.info(f"TRADEFEEDS DEBUG: No filtering applied - returning all {len(df)} records")
                                        return df
                                else:
                                    log.warning("No monthly returns could be calculated from OHLCV data")
                                    return pd.DataFrame()
                            else:
                                log.warning("No daily_stock_data found in response")
                                return pd.DataFrame()
                        else:
                            log.warning("Unexpected OHLCV response structure: missing output.daily_stock_data")
                            return pd.DataFrame()
                    else:
                        log.warning(f"Unexpected result structure: {type(result)}")
                        return pd.DataFrame()
                else:
                    log.warning("No 'result' key or empty result in OHLCV response")
                    log.debug(f"Full response for debugging: {data}")
                    return pd.DataFrame()
            
            elif response.status_code == 404:
                log.error(f"Ticker not found: {ticker}")
                return pd.DataFrame()
            
            else:
                log.error(f"OHLCV API returned {response.status_code}: {response.text}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            log.error(f"OHLCV API request failed: {e}")
            return pd.DataFrame()
    
    def _calculate_monthly_returns_from_ohlcv(
        self, 
        daily_data: list, 
        ticker: str, 
        series_id: str = None
    ) -> pd.DataFrame:
        """
        Calculate monthly returns from daily OHLCV data using adjustclose.
        
        Args:
            daily_data: List of daily OHLCV records from API
            ticker: Fund ticker symbol
            series_id: SEC series ID for metadata
        
        Returns:
            DataFrame with monthly returns
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(daily_data)
            if df.empty:
                return pd.DataFrame()
            
            # Convert to proper dtypes (following JSONvisualizer.py pattern)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            numeric_cols = ["open", "high", "low", "close", "adjustclose"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            
            # Sort by date
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            # Group by calendar month and get last trading day each month
            df["period"] = df["date"].dt.to_period("M")
            
            if df["period"].isna().any():
                log.warning("Encountered invalid monthly period while computing returns")
                return pd.DataFrame()
            
            # Aggregate to monthly data (last trading day of each month)
            monthly = (
                df.sort_values("date")
                .groupby("period")
                .agg(
                    actual_last_trading_day=("date", "max"),
                    adjustclose=("adjustclose", "last"),
                )
            )
            
            # Calculate monthly returns using adjustclose (dividend-adjusted)
            monthly["return"] = monthly["adjustclose"].pct_change()
            
            # Convert period to standard format and normalize month_end to calendar month-end
            monthly.reset_index(inplace=True)
            
            # Normalize month_end to calendar month-end for consistent merging with other data sources
            # Use clean datetime (midnight) to match factor data format, not end-of-day nanoseconds
            monthly["month_end"] = monthly["period"].dt.to_timestamp(how="end").dt.normalize()
            monthly["actual_trading_day"] = pd.to_datetime(monthly["actual_last_trading_day"])
            
            # Add metadata columns for compatibility with existing pipeline
            monthly["date"] = monthly["month_end"]
            monthly["series_id"] = series_id
            monthly["series_name"] = f"Fund for {ticker}"
            monthly["ticker"] = ticker
            monthly["report_date"] = monthly["month_end"]
            
            # Remove rows where return calculation is not possible (first month or NaN)
            monthly = monthly[monthly["return"].notna()]
            
            # Select and order columns for output
            output_columns = [
                "date", "month_end", "series_id", "series_name", "ticker", 
                "return", "report_date", "adjustclose", "actual_trading_day"
            ]
            
            result = monthly[output_columns].copy()
            result = result.sort_values("date")
            
            return result
            
        except Exception as e:
            log.error(f"Error calculating monthly returns from OHLCV data: {e}")
            return pd.DataFrame()
    
    def get_mutual_fund_returns_by_series(self, series_id: str, date_from: str = None, date_to: str = None, ticker: str = None) -> pd.DataFrame:
        """Convenience method to fetch by series_id."""
        return self.get_mutual_fund_returns(series_id=series_id, date_from=date_from, date_to=date_to, ticker=ticker)
    
    def test_api_connectivity(self) -> bool:
        """
        Test basic API connectivity and authentication.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Test with mutual_fund_information endpoint using example CIK
            endpoint = f"{self.BASE_URL}/mutual_fund_information"
            params = {
                'key': self.api_key,
                'regCik': '1137095'  # Example CIK from documentation
            }
            
            self._rate_limit()
            response = self.session.get(endpoint, params=params)
            
            if response.status_code == 200:
                log.info("API connectivity test successful")
                return True
            elif response.status_code == 401 or response.status_code == 403:
                log.error("API authentication failed - check API key")
                return False
            else:
                log.warning(f"API test returned status {response.status_code}")
                # Try a real endpoint with minimal data
                return self._test_with_known_fund()
                
        except requests.exceptions.RequestException as e:
            log.error(f"API connectivity test failed: {e}")
            return False
    
    def _test_with_known_fund(self) -> bool:
        """
        Test OHLCV API with known fund ticker.
        """
        try:
            # Try with known tickers from pilot funds
            test_tickers = ['VPMCX', 'HACAX']  # Vanguard PRIMECAP and Harbor Capital Appreciation
            
            for ticker in test_tickers:
                try:
                    # Test with a small date range
                    df = self.get_mutual_fund_returns(
                        ticker=ticker,
                        date_from='2024-01-01',
                        date_to='2024-02-29'
                    )
                    
                    if not df.empty:
                        log.info(f"OHLCV API test successful with ticker: {ticker}")
                        return True
                        
                except Exception as e:
                    log.warning(f"Test failed for ticker {ticker}: {e}")
                    continue
            
            log.warning("Could not validate OHLCV API with known fund tickers")
            return False
            
        except Exception as e:
            log.error(f"OHLCV API test with known fund failed: {e}")
            return False
    
    def get_mutual_fund_flow_data(
        self,
        series_id: str,
        date_from: str = None,
        date_to: str = None,
    ) -> pd.DataFrame:
        """
        Fetch flow data using the mutual fund returns API.
        This is used as fallback when SEC N-PORT data is not available.
        
        Args:
            series_id: SEC series ID (e.g., 'S000002594')
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with flow data (sales, redemptions, reinvest)
        """
        # API endpoint for mutual fund returns (which includes flow data)
        endpoint = f"{self.BASE_URL}/mutual_fund_periodical_returns"
        
        # Build params with API key
        params = {
            'key': self.api_key,
            'seriesId': series_id
        }
        
        # Add date range if provided
        if date_from:
            params['date_from'] = date_from
        if date_to:
            params['date_to'] = date_to

        try:
            log.info(f"Fetching flow data for seriesId={series_id}")
            log.debug(f"URL: {endpoint}?{self._params_to_query(params)}")
            
            self._rate_limit()
            
            headers = {'accept': 'application/json'}
            response = self.session.get(endpoint, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse mutual fund API response structure
                if isinstance(data, dict) and 'result' in data and data['result'] is not None:
                    result = data['result']
                    
                    if isinstance(result, dict) and 'output' in result:
                        output = result['output']
                        if isinstance(output, list) and len(output) > 0:
                            # Parse flow data from the complex nested structure
                            all_flows = []
                            
                            for fund_data in output:
                                attributes = fund_data.get('attributes', {})
                                series_id_resp = attributes.get('seriesId')
                                rep_date = attributes.get('repPdDate')  # Report date
                                
                                return_info = fund_data.get('return_info', {})
                                monthly_returns = return_info.get('monthly_total_returns', [])
                                
                                # Extract flow data for each month
                                mon1_flow = return_info.get('mon1Flow', {})
                                mon2_flow = return_info.get('mon2Flow', {})
                                mon3_flow = return_info.get('mon3Flow', {})
                                
                                if rep_date:
                                    base_date = pd.to_datetime(rep_date)
                                    
                                    # Extract class IDs from monthly returns
                                    class_ids = [item.get('class_id') for item in monthly_returns if item.get('class_id')]
                                    
                                    # Create flow records for each class and each month
                                    # mon1Flow is most recent month, mon2Flow is -1 month, mon3Flow is -2 months
                                    flow_data = [mon3_flow, mon2_flow, mon1_flow]  # In chronological order
                                    
                                    for i, month_flow in enumerate(flow_data):
                                        if month_flow:
                                            month_offset = 2 - i  # 2, 1, 0 for chronological order
                                            flow_date = base_date - pd.DateOffset(months=month_offset)
                                            month_end = flow_date + pd.offsets.MonthEnd(0)
                                            
                                            sales = float(month_flow.get('sales', 0)) if month_flow.get('sales') else 0
                                            redemptions = float(month_flow.get('redemption', 0)) if month_flow.get('redemption') else 0
                                            reinvest = float(month_flow.get('reinvestment', 0)) if month_flow.get('reinvestment') else 0
                                            
                                            # Create flow record for each class in this series
                                            for class_id in class_ids:
                                                all_flows.append({
                                                    'date': flow_date,
                                                    'month_end': month_end,
                                                    'series_id': series_id_resp,
                                                    'class_id': class_id,
                                                    'sales': sales,
                                                    'redemptions': redemptions,
                                                    'reinvest': reinvest,
                                                    'report_date': base_date
                                                })
                            
                            if all_flows:
                                log.info(f"SUCCESS: Parsed {len(all_flows)} flow records")
                                df = pd.DataFrame(all_flows)
                                df = df.sort_values(['class_id', 'date'])
                                return df
                            else:
                                log.warning("No flow data found in response")
                                return pd.DataFrame()
                        else:
                            log.warning("No output data in mutual fund API result")
                            return pd.DataFrame()
                    else:
                        log.warning(f"Unexpected mutual fund API result structure: {type(result)}")
                        return pd.DataFrame()
                else:
                    log.warning("No 'result' key or empty result in mutual fund API response")
                    return pd.DataFrame()
            
            elif response.status_code == 404:
                log.error(f"Series not found in mutual fund API: {series_id}")
                return pd.DataFrame()
            
            else:
                log.error(f"Mutual fund API returned {response.status_code}: {response.text}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            log.error(f"Mutual fund API request failed: {e}")
            return pd.DataFrame()
    
    