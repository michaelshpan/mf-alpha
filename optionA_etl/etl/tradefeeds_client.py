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
    
    def get_mutual_fund_returns(
        self,
        series_id: str = None,
        series_lei: str = None,
        date_from: str = None,
        date_to: str = None,
        ticker: str = None,
    ) -> pd.DataFrame:
        """
        Fetch returns using historical OHLCV data and calculate monthly returns.
        
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
        
        # Add date range if provided
        if date_from:
            params['date_from'] = date_from
        if date_to:
            params['date_to'] = date_to

        # Make API call
        try:
            log.info(f"Fetching OHLCV data for ticker={ticker}, seriesId={series_id or 'N/A'}")
            log.debug(f"URL: {endpoint}?{self._params_to_query(params)}")
            
            self._rate_limit()
            
            # Add accept header
            headers = {'accept': 'application/json'}
            response = self.session.get(endpoint, params=params, headers=headers)
            
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
                                    log.info(f"SUCCESS: Calculated {len(df)} monthly returns from OHLCV data")
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
                    month_end=("date", "max"),
                    adjustclose=("adjustclose", "last"),
                )
            )
            
            # Calculate monthly returns using adjustclose (dividend-adjusted)
            monthly["return"] = monthly["adjustclose"].pct_change()
            
            # Convert period to standard format
            monthly.reset_index(inplace=True)
            monthly["month_end"] = monthly["month_end"].dt.date
            monthly["month_end"] = pd.to_datetime(monthly["month_end"])
            
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
                "return", "report_date", "adjustclose"
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
    
    