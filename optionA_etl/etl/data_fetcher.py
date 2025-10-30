#!/usr/bin/env python3
"""
Data Fetching Module for Mutual Fund ETL

Handles data collection from multiple sources with:
- 36-month lookback for factor regressions
- Duplicate detection and consultation
- Source prioritization (SEC > Tradefeeds > Override)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path

from .identifier_mapper import IdentifierMapper
from .tradefeeds_client import TradeFeedsClient
from .factors import get_monthly_ff5_mom
from .config import AppConfig
from .sec_edgar import get_submissions_for_cik
from .sec_client import SECClient
from .nport_parser import parse_nport_primary_xml
from .oef_rr_extractor_robust import get_er_turnover_for_entities
from .manager_tenure import get_manager_data_for_entities

log = logging.getLogger(__name__)


class DuplicateDetector:
    """Detects and reports duplicate month-class_id combinations."""
    
    def __init__(self):
        self.duplicates_found = {}
        self.resolution_strategies = {}
    
    def check_duplicates(self, df: pd.DataFrame, source_name: str) -> List[Tuple[str, str]]:
        """
        Check for duplicate class_id-month_end combinations.
        
        Returns:
            List of (class_id, month_end) tuples that are duplicated
        """
        if df.empty:
            return []
        
        # Check if required columns exist
        required_cols = ['class_id', 'month_end']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            log.debug(f"Cannot check duplicates for {source_name}: missing columns {missing_cols}")
            log.debug(f"Available columns: {list(df.columns)}")
            return []
        
        try:
            # Check for duplicates
            duplicate_mask = df.duplicated(subset=['class_id', 'month_end'], keep=False)
            
            if duplicate_mask.any():
                duplicates_df = df[duplicate_mask].copy()
                duplicates_df['source'] = source_name
                
                # Group duplicates
                duplicate_groups = duplicates_df.groupby(['class_id', 'month_end']).size()
                duplicate_list = [(cid, str(me)) for cid, me in duplicate_groups.index]
                
                # Store for reporting
                if source_name not in self.duplicates_found:
                    self.duplicates_found[source_name] = []
                self.duplicates_found[source_name].extend(duplicate_list)
                
                log.warning(f"Found {len(duplicate_list)} duplicate class_id-month_end pairs in {source_name}")
                
                return duplicate_list
            
            return []
            
        except Exception as e:
            log.error(f"Error checking duplicates for {source_name}: {e}")
            log.debug(f"DataFrame columns: {list(df.columns)}")
            log.debug(f"DataFrame shape: {df.shape}")
            return []
    
    def report_duplicates(self) -> str:
        """Generate a report of all duplicates found."""
        if not self.duplicates_found:
            return "No duplicates detected."
        
        report = ["=== DUPLICATE DETECTION REPORT ===\n"]
        
        for source, dups in self.duplicates_found.items():
            report.append(f"\n{source}:")
            for class_id, month_end in dups[:10]:  # Show first 10
                report.append(f"  - {class_id} on {month_end}")
            if len(dups) > 10:
                report.append(f"  ... and {len(dups) - 10} more")
        
        report.append("\n=== RESOLUTION REQUIRED ===")
        report.append("Please specify deduplication strategy:")
        report.append("  1. 'latest': Keep most recent filing/update")
        report.append("  2. 'earliest': Keep earliest filing/update")
        report.append("  3. 'average': Average numeric values")
        report.append("  4. 'sec_priority': Prefer SEC source over others")
        report.append("  5. 'manual': Review each case individually")
        
        return "\n".join(report)
    
    def set_resolution_strategy(self, source: str, strategy: str):
        """Set resolution strategy for a source."""
        self.resolution_strategies[source] = strategy
    
    def apply_deduplication(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Apply deduplication based on configured strategy."""
        if df.empty or source not in self.resolution_strategies:
            return df
        
        strategy = self.resolution_strategies[source]
        
        if strategy == 'latest':
            # Sort by filing_date if available, then keep last
            if 'filing_date' in df.columns:
                df = df.sort_values(['class_id', 'month_end', 'filing_date'])
            else:
                df = df.sort_values(['class_id', 'month_end'])
            return df.drop_duplicates(subset=['class_id', 'month_end'], keep='last')
        
        elif strategy == 'earliest':
            # Keep first occurrence
            if 'filing_date' in df.columns:
                df = df.sort_values(['class_id', 'month_end', 'filing_date'])
            else:
                df = df.sort_values(['class_id', 'month_end'])
            return df.drop_duplicates(subset=['class_id', 'month_end'], keep='first')
        
        elif strategy == 'average':
            # Average numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'class_id' in numeric_cols:
                numeric_cols.remove('class_id')
            
            # Group and average
            grouped = df.groupby(['class_id', 'month_end'])
            
            # Average numeric columns
            avg_df = grouped[numeric_cols].mean().reset_index()
            
            # Keep first value for non-numeric columns
            non_numeric = df.columns.difference(['class_id', 'month_end'] + numeric_cols)
            if len(non_numeric) > 0:
                first_df = grouped[non_numeric].first().reset_index()
                avg_df = avg_df.merge(first_df, on=['class_id', 'month_end'])
            
            return avg_df
        
        elif strategy == 'sec_priority':
            # Prefer SEC source
            if 'data_source' in df.columns:
                df['source_priority'] = df['data_source'].map({'sec': 0, 'tradefeeds': 1}).fillna(2)
                df = df.sort_values(['class_id', 'month_end', 'source_priority'])
                return df.drop_duplicates(subset=['class_id', 'month_end'], keep='first')
            else:
                return df.drop_duplicates(subset=['class_id', 'month_end'], keep='last')
        
        else:
            # Manual or unknown - don't deduplicate
            return df


class MutualFundDataFetcher:
    """
    Main data fetching orchestrator.
    
    Manages collection from all sources with proper lookback periods
    and duplicate handling.
    """
    
    def __init__(
        self,
        identifier_mapper: IdentifierMapper = None,
        tradefeeds_client: TradeFeedsClient = None,
        cache_dir: str = "./cache"
    ):
        """
        Initialize the data fetcher.
        
        Args:
            identifier_mapper: Identifier mapping instance
            tradefeeds_client: Tradefeeds API client
            cache_dir: Directory for caching data
        """
        self.config = AppConfig()
        self.mapper = identifier_mapper or IdentifierMapper(cache_dir=cache_dir)
        self.tradefeeds = tradefeeds_client or TradeFeedsClient()
        self.sec_client = SECClient(self.config)
        self.duplicate_detector = DuplicateDetector()
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track data collection
        self.collected_data = {}
        self.missing_data_report = []
    
    def calculate_date_range_with_lookback(
        self,
        start_date: str,
        end_date: str,
        lookback_months: int = 36
    ) -> Tuple[str, str, str, str]:
        """
        Calculate extended date range including lookback period.
        
        Args:
            start_date: Requested start date (YYYY-MM-DD)
            end_date: Requested end date (YYYY-MM-DD)
            lookback_months: Months of historical data needed
            
        Returns:
            Tuple of (extended_start, extended_end, original_start, original_end)
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Calculate extended start date (36 months before requested start)
        extended_start_dt = start_dt - relativedelta(months=lookback_months)
        
        # Ensure we don't go too far back
        min_date = pd.to_datetime('1990-01-01')
        if extended_start_dt < min_date:
            extended_start_dt = min_date
            log.warning(f"Extended start date capped at {min_date.date()}")
        
        return (
            str(extended_start_dt.date()),
            str(end_dt.date()),
            str(start_dt.date()),
            str(end_dt.date())
        )
    
    def fetch_returns_data(
        self,
        fund_selection: pd.DataFrame,
        extended_start: str,
        extended_end: str
    ) -> pd.DataFrame:
        """
        Fetch monthly returns from Tradefeeds.
        
        Args:
            fund_selection: DataFrame with fund identifiers
            extended_start: Start date including lookback
            extended_end: End date
            
        Returns:
            DataFrame with monthly returns
        """
        log.info(f"Fetching returns data from {extended_start} to {extended_end}")
        
        all_returns = []
        
        for _, fund in fund_selection.iterrows():
            class_id = fund['class_id']
            series_id = fund.get('series_id')
            ticker = fund.get('ticker')
            
            if not ticker:
                log.warning(f"No ticker for class_id {class_id}, skipping returns")
                continue
            
            try:
                # Fetch from Tradefeeds
                df = self.tradefeeds.get_mutual_fund_returns(
                    series_id=series_id,
                    ticker=ticker,
                    date_from=extended_start,
                    date_to=extended_end
                )
                
                if not df.empty:
                    # TradeFeedsClient already returns monthly returns
                    # Just add class_id identifier
                    df['class_id'] = class_id
                    
                    all_returns.append(df)
                    log.info(f"Fetched {len(df)} months of returns for {ticker}")
                
            except Exception as e:
                log.error(f"Failed to fetch returns for {ticker}: {e}")
                self.missing_data_report.append(f"Returns missing for {ticker}: {e}")
        
        if all_returns:
            returns_df = pd.concat(all_returns, ignore_index=True)
            
            # Check for duplicates
            duplicates = self.duplicate_detector.check_duplicates(returns_df, "returns_data")
            if duplicates:
                log.warning(f"Found {len(duplicates)} duplicates in returns data")
            
            return returns_df
        else:
            return pd.DataFrame()
    
    def fetch_flow_data(
        self,
        fund_selection: pd.DataFrame,
        extended_start: str,
        extended_end: str
    ) -> pd.DataFrame:
        """
        Fetch flow data (sales, redemptions, reinvestments).
        
        Uses hybrid approach: SEC N-PORT primary, Tradefeeds fallback.
        """
        log.info(f"Fetching flow data from {extended_start} to {extended_end}")
        
        all_flows = []
        
        # Group by series_id for efficient SEC fetching
        for series_id in fund_selection['series_id'].unique():
            if pd.isna(series_id):
                continue
            
            series_funds = fund_selection[fund_selection['series_id'] == series_id]
            cik = series_funds.iloc[0]['cik']
            
            try:
                # Try SEC N-PORT first
                sec_flows = self._fetch_sec_flows(
                    cik=cik,
                    series_id=series_id,
                    start_date=extended_start,
                    end_date=extended_end
                )
                
                if not sec_flows.empty:
                    sec_flows['data_source'] = 'sec'
                    all_flows.append(sec_flows)
                    log.info(f"Fetched {len(sec_flows)} SEC flow records for series {series_id}")
                else:
                    # Fallback to Tradefeeds
                    tf_flows = self._fetch_tradefeeds_flows(
                        series_id=series_id,
                        start_date=extended_start,
                        end_date=extended_end
                    )
                    
                    if not tf_flows.empty:
                        tf_flows['data_source'] = 'tradefeeds'
                        all_flows.append(tf_flows)
                        log.info(f"Fetched {len(tf_flows)} Tradefeeds flow records for series {series_id}")
                    else:
                        self.missing_data_report.append(f"No flow data for series {series_id}")
                        
            except Exception as e:
                log.error(f"Failed to fetch flows for series {series_id}: {e}")
                self.missing_data_report.append(f"Flow data error for {series_id}: {e}")
        
        if all_flows:
            flows_df = pd.concat(all_flows, ignore_index=True)
            
            # Check for duplicates
            duplicates = self.duplicate_detector.check_duplicates(flows_df, "flow_data")
            if duplicates:
                log.warning(f"Found {len(duplicates)} duplicates in flow data")
            
            return flows_df
        else:
            return pd.DataFrame()
    
    def _fetch_sec_flows(
        self,
        cik: str,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch flow data from SEC N-PORT filings."""
        try:
            # Get N-PORT filings for the period
            filings = self.sec_client.get_nport_filings(
                cik=cik,
                start_date=start_date,
                end_date=end_date
            )
            
            all_flows = []
            
            for filing in filings:
                # Download and parse XML
                xml_content = self.sec_client.download_filing(
                    cik=cik,
                    accession_number=filing['accession_number'],
                    primary_document=filing['primary_document']
                )
                
                # Parse flow data
                flow_data = parse_nport_primary_xml(xml_content)
                
                if not flow_data.empty:
                    # N-PORT parser returns DataFrame with flow data
                    flows_df = flow_data.copy()
                    flows_df['series_id'] = series_id
                    flows_df['filing_date'] = filing['filing_date']
                    all_flows.append(flows_df)
            
            if all_flows:
                return pd.concat(all_flows, ignore_index=True)
            
        except Exception as e:
            log.debug(f"SEC flow fetch failed for {cik}: {e}")
        
        return pd.DataFrame()
    
    def _fetch_tradefeeds_flows(
        self,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch flow data from Tradefeeds API."""
        try:
            return self.tradefeeds.get_mutual_fund_flow_data(
                series_id=series_id,
                date_from=start_date,
                date_to=end_date
            )
        except Exception as e:
            log.debug(f"Tradefeeds flow fetch failed for {series_id}: {e}")
            return pd.DataFrame()
    
    def fetch_factor_data(
        self,
        extended_start: str,
        extended_end: str
    ) -> pd.DataFrame:
        """
        Fetch Fama-French 5 factors + momentum.
        
        Args:
            extended_start: Start date including lookback
            extended_end: End date
            
        Returns:
            DataFrame with factor data
        """
        log.info(f"Fetching factor data from {extended_start} to {extended_end}")
        
        try:
            # Fetch combined FF5 + momentum factors
            factors = get_monthly_ff5_mom()
            
            # Filter to date range
            factors['month_end'] = pd.to_datetime(factors['month_end'])
            factors = factors[
                (factors['month_end'] >= extended_start) &
                (factors['month_end'] <= extended_end)
            ]
            
            log.info(f"Fetched {len(factors)} months of factor data")
            
            # No duplicates expected for factor data (one row per month)
            return factors
            
        except Exception as e:
            log.error(f"Failed to fetch factor data: {e}")
            self.missing_data_report.append(f"Factor data fetch failed: {e}")
            return pd.DataFrame()
    
    def fetch_expense_turnover_data(
        self,
        fund_selection: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fetch expense ratios and turnover from SEC Risk/Return datasets.
        
        Args:
            fund_selection: DataFrame with fund identifiers
            
        Returns:
            DataFrame with expense and turnover data
        """
        log.info("Fetching expense and turnover data")
        
        try:
            # Prepare entities for expense/turnover extraction
            entities = []
            for cik in fund_selection['cik'].unique():
                if pd.isna(cik):
                    continue
                
                cik_funds = fund_selection[fund_selection['cik'] == cik]
                entities.append({
                    'cik': cik,
                    'series_ids': cik_funds['series_id'].unique().tolist(),
                    'class_ids': cik_funds['class_id'].unique().tolist()
                })
            
            # Extract expense and turnover data
            expense_turnover = get_er_turnover_for_entities(entities)
            
            if not expense_turnover.empty:
                log.info(f"Fetched expense/turnover for {len(expense_turnover)} funds")
                
                # Check for duplicates (shouldn't happen for static data)
                duplicates = self.duplicate_detector.check_duplicates(
                    expense_turnover, "expense_turnover"
                )
                
            return expense_turnover
            
        except Exception as e:
            log.error(f"Failed to fetch expense/turnover data: {e}")
            self.missing_data_report.append(f"Expense/turnover fetch failed: {e}")
            return pd.DataFrame()
    
    def fetch_manager_tenure_data(
        self,
        fund_selection: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fetch manager tenure and fund age from N-1A filings.
        
        Args:
            fund_selection: DataFrame with fund identifiers
            
        Returns:
            DataFrame with manager tenure and fund age
        """
        log.info("Fetching manager tenure and fund age data")
        
        # Prepare entities for tenure extraction
        entities = []
        for cik in fund_selection['cik'].unique():
            if pd.isna(cik):
                continue
            
            cik_funds = fund_selection[fund_selection['cik'] == cik]
            
            entities.append({
                'cik': cik,
                'series_ids': cik_funds['series_id'].unique().tolist(),
                'class_ids': cik_funds['class_id'].unique().tolist()
            })
        
        try:
            tenure_data = get_manager_data_for_entities(entities)
            
            if not tenure_data.empty:
                log.info(f"Fetched manager data for {len(tenure_data)} fund-classes")
                
                # Check for duplicates
                duplicates = self.duplicate_detector.check_duplicates(
                    tenure_data, "manager_tenure"
                )
            
            return tenure_data
            
        except Exception as e:
            log.error(f"Failed to fetch manager tenure data: {e}")
            self.missing_data_report.append(f"Manager tenure fetch failed: {e}")
            return pd.DataFrame()
    
    def fetch_tna_data(
        self,
        fund_selection: pd.DataFrame,
        extended_start: str,
        extended_end: str
    ) -> pd.DataFrame:
        """
        Fetch Total Net Assets from SEC N-PORT filings.
        
        Args:
            fund_selection: DataFrame with fund identifiers
            extended_start: Start date including lookback
            extended_end: End date
            
        Returns:
            DataFrame with TNA data
        """
        log.info(f"Fetching TNA data from {extended_start} to {extended_end}")
        
        all_tna = []
        
        for cik in fund_selection['cik'].unique():
            if pd.isna(cik):
                continue
            
            try:
                # Get N-PORT filings
                filings = self.sec_client.get_nport_filings(
                    cik=cik,
                    start_date=extended_start,
                    end_date=extended_end
                )
                
                for filing in filings:
                    # Download and parse XML
                    xml_content = self.sec_client.download_filing(
                        cik=cik,
                        accession_number=filing['accession_number'],
                        primary_document=filing['primary_document']
                    )
                    
                    # Parse TNA data
                    tna_data = parse_nport_primary_xml(xml_content)
                    
                    if not tna_data.empty and 'total_investments' in tna_data.columns:
                        # N-PORT parser returns DataFrame with TNA data
                        # Add filing date and map to fund classes
                        tna_data = tna_data.copy()
                        tna_data['filing_date'] = filing['filing_date']
                        
                        # Rename columns for consistency
                        if 'total_investments' in tna_data.columns:
                            tna_data['tna'] = tna_data['total_investments']
                        
                        # Map to all classes in this CIK that match the data
                        cik_funds = fund_selection[fund_selection['cik'] == cik]
                        for _, fund in cik_funds.iterrows():
                            # Check if this class has data in the TNA results
                            class_data = tna_data[tna_data['class_id'] == fund['class_id']]
                            if not class_data.empty:
                                all_tna.append(class_data)
                
            except Exception as e:
                log.error(f"Failed to fetch TNA for CIK {cik}: {e}")
                self.missing_data_report.append(f"TNA fetch failed for {cik}: {e}")
        
        if all_tna:
            tna_df = pd.concat(all_tna, ignore_index=True)
            
            # Check for duplicates
            duplicates = self.duplicate_detector.check_duplicates(tna_df, "tna_data")
            if duplicates:
                log.warning(f"Found {len(duplicates)} duplicates in TNA data")
            
            return tna_df
        else:
            return pd.DataFrame()
    
    def fetch_all_data(
        self,
        start_date: str,
        end_date: str,
        fund_config: Dict
    ) -> Dict[str, pd.DataFrame]:
        """
        Main entry point to fetch all required data.
        
        Args:
            start_date: Requested start date (YYYY-MM-DD)
            end_date: Requested end date (YYYY-MM-DD)
            fund_config: Configuration from funds_pilot.yaml
            
        Returns:
            Dictionary of DataFrames by data type
        """
        log.info(f"Starting data fetch for {start_date} to {end_date}")
        
        # Calculate extended date range
        extended_start, extended_end, orig_start, orig_end = \
            self.calculate_date_range_with_lookback(start_date, end_date)
        
        log.info(f"Extended date range: {extended_start} to {extended_end} "
                f"(includes 36-month lookback)")
        
        # Select funds based on configuration
        fund_selection = self._select_funds_from_config(fund_config)
        
        if fund_selection.empty:
            log.error("No funds selected from configuration")
            return {}
        
        log.info(f"Selected {len(fund_selection)} fund-classes for data collection")
        
        # Fetch all data types
        results = {}
        
        # Tier 1 - Essential
        results['returns'] = self.fetch_returns_data(
            fund_selection, extended_start, extended_end
        )
        results['factors'] = self.fetch_factor_data(
            extended_start, extended_end
        )
        
        # Tier 2 - Core Characteristics
        results['flows'] = self.fetch_flow_data(
            fund_selection, extended_start, extended_end
        )
        results['tna'] = self.fetch_tna_data(
            fund_selection, extended_start, extended_end
        )
        results['expense_turnover'] = self.fetch_expense_turnover_data(
            fund_selection
        )
        
        # Tier 3 - Important Features
        results['manager_tenure'] = self.fetch_manager_tenure_data(
            fund_selection
        )
        
        # Store metadata
        results['_metadata'] = {
            'requested_start': orig_start,
            'requested_end': orig_end,
            'extended_start': extended_start,
            'extended_end': extended_end,
            'fund_count': len(fund_selection),
            'funds': fund_selection
        }
        
        # Check for duplicates and report
        if self.duplicate_detector.duplicates_found:
            duplicate_report = self.duplicate_detector.report_duplicates()
            log.warning("Duplicates detected - consultation required:\n" + duplicate_report)
            results['_duplicate_report'] = duplicate_report
        
        # Report missing data
        if self.missing_data_report:
            log.warning(f"Missing data for {len(self.missing_data_report)} items")
            results['_missing_data'] = self.missing_data_report
        
        return results
    
    def _select_funds_from_config(self, fund_config: Dict) -> pd.DataFrame:
        """
        Select funds based on configuration.
        
        IMPORTANT: Only select funds for the specific series_ids and class_ids 
        listed in the config, not all funds for the CIK.
        """
        all_selections = []
        
        for registrant in fund_config.get('registrants', []):
            cik = registrant.get('cik')
            series_ids = registrant.get('series_ids', [])
            class_ids = registrant.get('class_ids', [])
            
            log.info(f"Processing registrant '{registrant.get('name')}' with CIK {cik}")
            
            # Select by series (includes all classes for those series only)
            if series_ids:
                log.info(f"Selecting funds for specific series: {series_ids}")
                selection = self.mapper.select_funds(series_ids=series_ids)
                
                # Verify the selection matches expected CIK if provided
                if cik and not selection.empty:
                    # Filter to ensure we only get the expected CIK
                    cik_normalized = str(int(cik))  # Remove leading zeros
                    selection = selection[selection['cik'] == cik_normalized]
                    
                if not selection.empty:
                    log.info(f"Selected {len(selection)} fund-classes for series {series_ids}")
                    all_selections.append(selection)
                else:
                    log.warning(f"No funds found for series {series_ids}")
            
            # Select specific classes
            if class_ids:
                log.info(f"Selecting funds for specific classes: {class_ids}")
                selection = self.mapper.select_funds(class_ids=class_ids)
                
                # Verify the selection matches expected CIK if provided
                if cik and not selection.empty:
                    cik_normalized = str(int(cik))
                    selection = selection[selection['cik'] == cik_normalized]
                
                if not selection.empty:
                    log.info(f"Selected {len(selection)} fund-classes for classes {class_ids}")
                    all_selections.append(selection)
                else:
                    log.warning(f"No funds found for classes {class_ids}")
            
            # If no series or classes specified, warn and skip (don't select all for CIK)
            if not series_ids and not class_ids:
                log.warning(f"No series_ids or class_ids specified for registrant '{registrant.get('name')}' "
                           f"(CIK {cik}). Skipping to avoid selecting all funds for this CIK.")
        
        if all_selections:
            combined = pd.concat(all_selections, ignore_index=True).drop_duplicates()
            log.info(f"Total selected funds: {len(combined)} fund-classes across "
                    f"{combined['series_id'].nunique()} series")
            return combined
        else:
            log.warning("No funds selected from configuration")
            return pd.DataFrame()