#!/usr/bin/env python3
"""
ETL Orchestrator for Mutual Fund Data Pipeline

Coordinates data fetching, deduplication, and computation of derived metrics.
Handles the complete ETL workflow with proper error handling and reporting.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import yaml

from .data_fetcher import MutualFundDataFetcher, DuplicateDetector
from .identifier_mapper import IdentifierMapper
from .metrics import (
    compute_net_flow,
    compute_flow_volatility, 
    rolling_factor_regressions,
    value_added
)
from .data_overrides import DataOverrideProcessor

log = logging.getLogger(__name__)


class ETLOrchestrator:
    """
    Main ETL orchestrator that coordinates the entire data pipeline.
    
    Features:
    - Manages data fetching with 36-month lookback
    - Handles duplicate detection and resolution
    - Computes all derived metrics
    - Produces final output compatible with prediction model
    """
    
    def __init__(
        self,
        config_path: str = "config/funds_pilot.yaml",
        cache_dir: str = "./cache",
        output_dir: str = "./outputs"
    ):
        """
        Initialize the ETL orchestrator.
        
        Args:
            config_path: Path to fund configuration YAML
            cache_dir: Directory for caching
            output_dir: Directory for output files
        """
        self.config_path = Path(config_path)
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.fund_config = self._load_config()
        
        # Initialize components
        self.identifier_mapper = IdentifierMapper(cache_dir=str(self.cache_dir))
        self.data_fetcher = MutualFundDataFetcher(
            identifier_mapper=self.identifier_mapper,
            cache_dir=str(self.cache_dir)
        )
        
        # Track processing status
        self.processing_status = {
            'data_fetched': False,
            'duplicates_resolved': False,
            'metrics_computed': False,
            'output_generated': False
        }
        
        # Store data
        self.raw_data = {}
        self.processed_data = None
        self.duplicate_resolutions = {}
    
    def _load_config(self) -> Dict:
        """Load fund configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_etl(
        self,
        start_date: str,
        end_date: str,
        auto_dedupe: bool = False,
        dedupe_strategy: str = 'latest'
    ) -> pd.DataFrame:
        """
        Run the complete ETL pipeline.
        
        Args:
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            auto_dedupe: Automatically apply deduplication strategy
            dedupe_strategy: Strategy to use if auto_dedupe is True
            
        Returns:
            Final processed DataFrame ready for prediction model
        """
        log.info("="*60)
        log.info("Starting ETL Pipeline")
        log.info(f"Period: {start_date} to {end_date}")
        log.info("="*60)
        
        # Step 1: Fetch all data
        log.info("\n=== Step 1: Data Fetching ===")
        self.raw_data = self.data_fetcher.fetch_all_data(
            start_date=start_date,
            end_date=end_date,
            fund_config=self.fund_config
        )
        self.processing_status['data_fetched'] = True
        
        # Check for duplicates
        if '_duplicate_report' in self.raw_data:
            if not auto_dedupe:
                print("\n" + "="*60)
                print("DUPLICATE DETECTION - USER INPUT REQUIRED")
                print("="*60)
                print(self.raw_data['_duplicate_report'])
                print("\nPlease call resolve_duplicates() with your chosen strategy")
                print("Example: orchestrator.resolve_duplicates({'returns': 'latest', 'flows': 'sec_priority'})")
                return None
            else:
                # Auto-resolve with provided strategy
                log.info(f"Auto-resolving duplicates with strategy: {dedupe_strategy}")
                self._auto_resolve_duplicates(dedupe_strategy)
        
        # Step 2: Combine and align data
        log.info("\n=== Step 2: Data Combination ===")
        combined_data = self._combine_data_sources()
        
        # Step 3: Compute derived metrics
        log.info("\n=== Step 3: Computing Derived Metrics ===")
        final_data = self._compute_metrics(combined_data)
        self.processing_status['metrics_computed'] = True
        
        # Step 4: Final validation and filtering
        log.info("\n=== Step 4: Final Validation ===")
        final_data = self._validate_and_filter(final_data, start_date, end_date)
        
        # Step 5: Save output
        log.info("\n=== Step 5: Saving Output ===")
        self._save_output(final_data, start_date, end_date)
        self.processing_status['output_generated'] = True
        
        # Generate summary report
        self._generate_summary_report(final_data)
        
        self.processed_data = final_data
        return final_data
    
    def resolve_duplicates(self, resolution_strategies: Dict[str, str]):
        """
        Manually resolve duplicates with specified strategies.
        
        Args:
            resolution_strategies: Dict mapping data source to strategy
                e.g., {'returns': 'latest', 'flows': 'sec_priority'}
        """
        if not self.processing_status['data_fetched']:
            raise RuntimeError("Must fetch data before resolving duplicates")
        
        log.info("Applying duplicate resolution strategies")
        
        for source, strategy in resolution_strategies.items():
            if source in self.raw_data and not self.raw_data[source].empty:
                self.data_fetcher.duplicate_detector.set_resolution_strategy(source, strategy)
                self.raw_data[source] = self.data_fetcher.duplicate_detector.apply_deduplication(
                    self.raw_data[source], source
                )
                log.info(f"Applied '{strategy}' deduplication to {source}")
        
        self.duplicate_resolutions = resolution_strategies
        self.processing_status['duplicates_resolved'] = True
        
        # Continue processing
        return self.continue_after_deduplication()
    
    def _auto_resolve_duplicates(self, strategy: str):
        """Apply the same deduplication strategy to all data sources."""
        resolution_strategies = {}
        
        for source in ['returns', 'flows', 'tna', 'expense_turnover', 'manager_tenure']:
            if source in self.raw_data and not self.raw_data[source].empty:
                resolution_strategies[source] = strategy
        
        self.resolve_duplicates(resolution_strategies)
    
    def continue_after_deduplication(self) -> pd.DataFrame:
        """Continue ETL pipeline after duplicate resolution."""
        if not self.processing_status['duplicates_resolved']:
            raise RuntimeError("Duplicates must be resolved first")
        
        # Continue from Step 2
        log.info("\n=== Continuing ETL After Deduplication ===")
        
        # Combine data
        combined_data = self._combine_data_sources()
        
        # Compute metrics
        final_data = self._compute_metrics(combined_data)
        self.processing_status['metrics_computed'] = True
        
        # Validate and filter
        metadata = self.raw_data.get('_metadata', {})
        final_data = self._validate_and_filter(
            final_data,
            metadata.get('requested_start'),
            metadata.get('requested_end')
        )
        
        # Save output
        self._save_output(
            final_data,
            metadata.get('requested_start'),
            metadata.get('requested_end')
        )
        self.processing_status['output_generated'] = True
        
        # Generate summary
        self._generate_summary_report(final_data)
        
        self.processed_data = final_data
        return final_data
    
    def _combine_data_sources(self) -> pd.DataFrame:
        """Combine all data sources into a single DataFrame."""
        log.info("Combining data from all sources")
        
        # Start with returns as base
        if 'returns' not in self.raw_data or self.raw_data['returns'].empty:
            raise ValueError("No returns data available - cannot proceed")
        
        combined = self.raw_data['returns'].copy()
        combined['month_end'] = pd.to_datetime(combined['month_end'])
        
        # Merge factors (one row per month, broadcast to all funds)
        if 'factors' in self.raw_data and not self.raw_data['factors'].empty:
            factors = self.raw_data['factors'].copy()
            factors['month_end'] = pd.to_datetime(factors['month_end'])
            combined = combined.merge(factors, on='month_end', how='left')
            log.info(f"Merged factor data: {len(factors)} months")
        
        # Merge flows
        if 'flows' in self.raw_data and not self.raw_data['flows'].empty:
            flows = self.raw_data['flows'].copy()
            flows['month_end'] = pd.to_datetime(flows['month_end'])
            
            # Select relevant columns
            flow_cols = ['class_id', 'month_end', 'sales', 'redemptions', 'reinvest']
            flow_cols = [c for c in flow_cols if c in flows.columns]
            
            combined = combined.merge(
                flows[flow_cols],
                on=['class_id', 'month_end'],
                how='left'
            )
            log.info(f"Merged flow data: {len(flows)} records")
        
        # Merge TNA (handle multiple column names)
        if 'tna' in self.raw_data and not self.raw_data['tna'].empty:
            tna = self.raw_data['tna'].copy()
            tna['month_end'] = pd.to_datetime(tna['month_end'])
            
            # Determine TNA column name
            tna_column = None
            if 'tna' in tna.columns:
                tna_column = 'tna'
            elif 'total_investments' in tna.columns:
                tna_column = 'total_investments'
                # Standardize column name for downstream processing
                tna['tna'] = tna['total_investments']
            
            if tna_column:
                # Select merge columns
                merge_cols = ['class_id', 'month_end']
                value_cols = ['tna']  # Always use 'tna' as standardized name
                
                if tna_column in tna.columns:
                    combined = combined.merge(
                        tna[merge_cols + value_cols],
                        on=merge_cols,
                        how='left'
                    )
                    
                    tna_coverage = combined['tna'].notna().sum()
                    log.info(f"Merged TNA data: {len(tna)} records")
                    log.info(f"TNA coverage: {tna_coverage}/{len(combined)} ({tna_coverage/len(combined)*100:.1f}%)")
                    
                    # Also compute tna_lag for value_added calculation
                    combined = combined.sort_values(['class_id', 'month_end'])
                    combined['tna_lag'] = combined.groupby('class_id')['tna'].shift(1)
                    
                    tna_lag_coverage = combined['tna_lag'].notna().sum()
                    log.info(f"TNA lag coverage: {tna_lag_coverage}/{len(combined)} ({tna_lag_coverage/len(combined)*100:.1f}%)")
            else:
                log.warning("TNA data found but no recognized TNA column (tna, total_investments)")
        else:
            log.warning("No TNA data available for merging")
        
        # Merge expense/turnover (static data, forward-fill)
        if 'expense_turnover' in self.raw_data and not self.raw_data['expense_turnover'].empty:
            exp_turn = self.raw_data['expense_turnover'].copy()
            
            # Check if data already has SEC class IDs or needs mapping from tickers
            if 'class_id' in exp_turn.columns:
                # Check if class_id values are already SEC class IDs (format: C followed by digits)
                sample_class_ids = exp_turn['class_id'].dropna().head(5).astype(str)
                has_sec_class_ids = all(
                    class_id.startswith('C') and class_id[1:].isdigit() 
                    for class_id in sample_class_ids if class_id != 'None'
                )
                
                if has_sec_class_ids:
                    # Data already has SEC class IDs, merge directly
                    log.info(f"SEC RR data already has proper class IDs, merging directly ({len(exp_turn)} records)")
                    combined = combined.merge(
                        exp_turn[['class_id', 'net_expense_ratio', 'turnover_pct']],
                        on='class_id',
                        how='left'
                    )
                    log.info(f"Merged expense/turnover data directly using SEC class IDs")
                else:
                    # Data has ticker-based class IDs, need mapping
                    log.info(f"Data has ticker-based class IDs, applying mapping")
                    
                    # Create mapping from ticker to SEC class_id
                    ticker_to_sec_class = {}
                    if hasattr(self, 'identifier_mapper') and self.identifier_mapper:
                        mapping_df = self.identifier_mapper.mapping_df
                        for _, row in mapping_df.iterrows():
                            ticker = row.get('ticker')
                            sec_class_id = row.get('class_id')
                            if ticker and sec_class_id:
                                ticker_to_sec_class[ticker] = sec_class_id
                    
                    # Map OEF/RR ticker-based class_ids to SEC class_ids
                    exp_turn['sec_class_id'] = exp_turn['class_id'].map(ticker_to_sec_class)
                    
                    # Log mapping results
                    mapped_count = exp_turn['sec_class_id'].notna().sum()
                    log.info(f"Mapped {mapped_count}/{len(exp_turn)} OEF/RR records to SEC class IDs")
                    
                    # Keep records that successfully mapped
                    exp_turn_mapped = exp_turn[exp_turn['sec_class_id'].notna()].copy()
                    
                    if not exp_turn_mapped.empty:
                        # Use sec_class_id for merging
                        combined = combined.merge(
                            exp_turn_mapped[['sec_class_id', 'net_expense_ratio', 'turnover_pct']].rename(
                                columns={'sec_class_id': 'class_id'}
                            ),
                            on='class_id',
                            how='left'
                        )
                        log.info(f"Merged expense/turnover data using class ID mapping")
                    else:
                        log.warning("No OEF/RR records could be mapped to SEC class IDs")
        
        # Merge manager tenure (static data)
        if 'manager_tenure' in self.raw_data and not self.raw_data['manager_tenure'].empty:
            tenure = self.raw_data['manager_tenure'].copy()
            
            if 'class_id' in tenure.columns:
                combined = combined.merge(
                    tenure[['class_id', 'manager_tenure', 'fund_age']],
                    on='class_id',
                    how='left'
                )
                log.info(f"Merged manager tenure data")
        
        log.info(f"Combined dataset: {len(combined)} records, {len(combined.columns)} columns")
        
        # Apply manual overrides after all merges
        combined = self._apply_manual_overrides(combined)
        
        return combined
    
    def _apply_manual_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply manual override data from CSV files."""
        log.info("Applying manual override data")
        
        # Path to manual overrides file
        override_file = Path("etl/data/manual_overrides.csv")
        if not override_file.exists():
            # Try relative to current directory
            override_file = Path("optionA_etl/etl/data/manual_overrides.csv")
        
        if not override_file.exists():
            log.warning(f"Manual override file not found at {override_file}")
            return df
        
        try:
            # Initialize override processor
            processor = DataOverrideProcessor(str(override_file))
            
            # Apply overrides
            df_with_overrides = processor.apply_overrides(df)
            
            # Log coverage improvements
            coverage_before = {}
            coverage_after = {}
            override_fields = ['net_expense_ratio', 'manager_tenure', 'fund_age']
            
            for field in override_fields:
                if field in df.columns:
                    coverage_before[field] = (~df[field].isna()).sum()
                if field in df_with_overrides.columns:
                    coverage_after[field] = (~df_with_overrides[field].isna()).sum()
            
            log.info("Override application results:")
            for field in override_fields:
                if field in coverage_before and field in coverage_after:
                    before = coverage_before[field]
                    after = coverage_after[field]
                    improvement = after - before
                    log.info(f"  {field}: {before} → {after} records (+{improvement})")
            
            return df_with_overrides
            
        except Exception as e:
            log.error(f"Failed to apply manual overrides: {e}")
            return df
    
    def _compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all derived metrics."""
        log.info("Computing derived metrics")
        
        # Sort by class and date for proper calculations
        df = df.sort_values(['class_id', 'month_end'])
        
        # 1. Compute net flows
        if all(col in df.columns for col in ['sales', 'redemptions', 'reinvest']):
            df['net_flow'] = compute_net_flow(df)
            log.info("Computed net flows")
        
        # 2. Compute flow volatility (12-month rolling)
        if 'net_flow' in df.columns:
            df['vol_of_flows'] = compute_flow_volatility(df)
            log.info("Computed flow volatility")
        
        # 3. Compute factor regressions (36-month rolling)
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        if all(col in df.columns for col in factor_cols + ['return', 'RF']):
            log.info("Computing factor regressions (this may take a while)...")
            
            # Compute excess returns
            df['excess_return'] = df['return'] - df['RF']
            
            # Run rolling regressions
            regression_results = rolling_factor_regressions(df)
            
            if regression_results.empty:
                log.warning("No factor regression results - insufficient data for 36-month window")
                log.info("Skipping factor regression merge")
            else:
                log.info(f"Generated {len(regression_results)} factor regression results")
                log.debug(f"Regression result columns: {list(regression_results.columns)}")
                
                # Merge regression results
                df = df.merge(regression_results, on=['class_id', 'month_end'], how='left')
                log.info("Computed factor regressions and statistics")
        
        # 4. Compute value added  
        alpha_cols = ['realized_alpha', 'alpha_hat']  # alpha_hat comes from factor regression
        expense_cols = ['net_expense_ratio']
        tna_cols = ['tna', 'total_investments']
        
        has_alpha = any(col in df.columns for col in alpha_cols)
        has_expense = any(col in df.columns for col in expense_cols)
        has_tna = any(col in df.columns for col in tna_cols)
        
        if has_alpha and has_expense and has_tna:
            # Use alpha_hat if available (from factor regression), otherwise realized_alpha
            if 'alpha_hat' in df.columns:
                df['realized_alpha'] = df['alpha_hat']  # Standardize name for value_added function
            
            df['value_added'] = value_added(df)
            log.info("Computed value added")
        else:
            missing = []
            if not has_alpha: missing.append("alpha/alpha_hat")
            if not has_expense: missing.append("expense_ratio") 
            if not has_tna: missing.append("TNA")
            log.warning(f"Skipping value_added: missing {', '.join(missing)}")
        
        # 5. Calculate fund flows as percentage (if TNA available)
        if 'tna' in df.columns and 'net_flow' in df.columns:
            df['flows'] = df.groupby('class_id').apply(
                lambda g: g['net_flow'] / g['tna'].shift(1)
            ).reset_index(drop=True)
            log.info("Computed flows as percentage of TNA")
        
        return df
    
    def _validate_and_filter(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Validate data and filter to requested date range."""
        log.info("Validating and filtering data")
        
        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter to requested date range
        initial_count = len(df)
        df = df[
            (df['month_end'] >= start_dt) &
            (df['month_end'] <= end_dt)
        ]
        
        log.info(f"Filtered from {initial_count} to {len(df)} records "
                f"(requested period only)")
        
        # Check data completeness for key fields
        required_fields = [
            'class_id', 'month_end', 'return'
        ]
        
        for field in required_fields:
            if field in df.columns:
                missing = df[field].isna().sum()
                if missing > 0:
                    pct = 100 * missing / len(df)
                    log.warning(f"Field '{field}' has {missing} ({pct:.1f}%) missing values")
        
        # Final deduplication check
        duplicates = df.duplicated(subset=['class_id', 'month_end'], keep=False)
        if duplicates.any():
            log.error(f"CRITICAL: {duplicates.sum()} duplicates remain after processing!")
            # Show duplicate details
            dup_df = df[duplicates][['class_id', 'month_end']].drop_duplicates()
            log.error(f"Duplicate class-month pairs:\n{dup_df.head(10)}")
        
        return df
    
    def _save_output(self, df: pd.DataFrame, start_date: str, end_date: str):
        """Save processed data to output files."""
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main output as parquet
        output_file = self.output_dir / f"mutual_fund_data_{start_date}_{end_date}_{timestamp}.parquet"
        df.to_parquet(output_file, index=False)
        log.info(f"Saved output to {output_file}")
        df.to_csv(output_file.with_suffix('.csv'), index=False)
        log.info(f"Saved output to {output_file.with_suffix('.csv')}")
        
        # Save metadata
        metadata = {
            'generation_timestamp': timestamp,
            'requested_start': start_date,
            'requested_end': end_date,
            'extended_start': self.raw_data.get('_metadata', {}).get('extended_start'),
            'extended_end': self.raw_data.get('_metadata', {}).get('extended_end'),
            'record_count': len(df),
            'fund_count': df['class_id'].nunique(),
            'duplicate_resolutions': self.duplicate_resolutions,
            'missing_data': self.raw_data.get('_missing_data', [])
        }
        
        metadata_file = self.output_dir / f"metadata_{start_date}_{end_date}_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        log.info(f"Saved metadata to {metadata_file}")
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """Generate and display summary report."""
        print("\n" + "="*60)
        print("ETL PIPELINE SUMMARY REPORT")
        print("="*60)
        
        print(f"\nData Shape: {len(df)} records × {len(df.columns)} columns")
        print(f"Unique Funds: {df['class_id'].nunique()}")
        print(f"Date Range: {df['month_end'].min()} to {df['month_end'].max()}")
        
        # Field completeness
        print("\nField Completeness (17 required characteristics):")
        
        key_fields = [
            'realized_alpha', 'flows', 'value_added', 'vol_of_flows',
            'tna', 'tna_lag', 'net_expense_ratio', 'fund_age', 'manager_tenure',
            'turnover_pct', 'MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM',
            'R2', 'return', 'net_flow'
        ]
        
        for field in key_fields:
            if field in df.columns:
                complete = (~df[field].isna()).sum()
                pct = 100 * complete / len(df)
                status = "✓" if pct > 90 else "⚠" if pct > 50 else "✗"
                print(f"  {status} {field:20} {pct:6.1f}% complete")
            else:
                print(f"  ✗ {field:20} MISSING")
        
        # Processing status
        print("\nProcessing Status:")
        for step, completed in self.processing_status.items():
            status = "✓" if completed else "✗"
            print(f"  {status} {step.replace('_', ' ').title()}")
        
        print("\n" + "="*60)