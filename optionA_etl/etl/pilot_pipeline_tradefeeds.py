#!/usr/bin/env python3
"""
Option A pilot ETL pipeline using Tradefeeds API for monthly returns.
Replaces SEC N-PORT-P filing downloads with direct API access.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import existing modules
from .config import AppConfig
from .tradefeeds_client import TradeFeedsClient
from .tradefeeds_returns import TradeFeedsReturnsFetcher
from .factors import get_monthly_ff5_mom
from .sec_edgar import list_recent_nport_p_accessions, download_filing_xml
from .nport_parser_fixed import parse_nport_primary_xml
from .metrics import (
    compute_net_flow, 
    compute_flow_volatility, 
    compute_realized_alpha_lagged, 
    rolling_factor_regressions, 
    value_added
)
from .oef_rr_extractor_robust import get_er_turnover_for_entities
from .manager_tenure import get_manager_data_for_entities
from .series_class_mapper import SeriesClassMapper
from .sec_rr_integration import SECRRDataLoader
from .data_overrides import apply_data_overrides, generate_override_template

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("pilot_tradefeeds")


def load_pilot_list(path: Path) -> list[dict]:
    """Load pilot fund configuration."""
    cfg = yaml.safe_load(path.read_text())
    return cfg.get("registrants", [])


def fetch_tna_from_sec_nport(regs: list, date_from: str, date_to: str, appcfg, mapper) -> pd.DataFrame:
    """
    Fetch TNA (total_investments) data from SEC N-PORT filings.
    This supplements Tradefeeds returns data with asset size information.
    """
    log.info("Fetching TNA data from SEC N-PORT filings...")
    
    # Build lookup of series -> CIK from config
    series_to_cik = {}
    for reg in regs:
        cik = reg.get("cik")
        if cik:
            cik_str = str(int(cik))
            series_ids = reg.get("series_ids", [])
            for sid in series_ids:
                series_to_cik[sid] = cik_str
    
    # Get all valid class IDs for configured series
    valid_class_ids = set()
    for series_id in series_to_cik.keys():
        class_ids = mapper.get_class_for_series(series_id)
        if class_ids:
            valid_class_ids.update(class_ids)
    
    log.info(f"Looking for TNA data for {len(valid_class_ids)} class IDs")
    
    tna_rows = []
    for reg in regs:
        cik = reg.get("cik")
        if not cik:
            continue
        
        name = reg.get("name", "")
        log.info(f"Fetching SEC filings for {name} (CIK: {cik})")
        
        try:
            # Get N-PORT-P filings for this CIK
            filings = list_recent_nport_p_accessions(cik, appcfg, since_yyyymmdd=date_from.replace('-', ''))
            log.info(f"Found {len(filings)} N-PORT-P filings for {name}")
            
            for filing in filings:
                try:
                    # Download and parse the filing
                    xml = download_filing_xml(cik, filing["accession"], filing["primary_doc"], appcfg)
                    df = parse_nport_primary_xml(xml)
                    
                    if not df.empty:
                        df["cik"] = str(int(cik))
                        df["filing_date"] = pd.to_datetime(filing["filing_date"])
                        
                        # Filter to only target class IDs
                        df = df[df["class_id"].isin(valid_class_ids)]
                        
                        if not df.empty:
                            # Add series_id mapping
                            df["series_id"] = df["class_id"].apply(lambda cid: mapper.get_series_for_class(cid))
                            tna_rows.append(df)
                            
                except Exception as e:
                    log.warning(f"Failed to parse filing {filing['accession']}: {e}")
                    continue
                    
        except Exception as e:
            log.error(f"Failed to fetch filings for CIK {cik}: {e}")
            continue
    
    if tna_rows:
        tna_data = pd.concat(tna_rows, ignore_index=True)
        
        # Deduplicate - keep most recent filing for each class_id-month_end
        log.info(f"Deduplicating TNA data from {len(tna_data)} raw records...")
        tna_data = tna_data.sort_values(['class_id', 'month_end', 'filing_date'])
        tna_data = tna_data.drop_duplicates(subset=['class_id', 'month_end'], keep='last')
        
        log.info(f"Retrieved TNA data: {len(tna_data)} unique class-month records")
        return tna_data
    else:
        log.warning("No TNA data retrieved from SEC filings")
        return pd.DataFrame()


def main():
    """Main ETL pipeline using Tradefeeds API."""
    
    p = argparse.ArgumentParser(description="Option A pilot ETL with Tradefeeds")
    p.add_argument("--pilot", default="config/funds_pilot.yaml", help="Pilot fund configuration")
    p.add_argument("--since", default="2019-01-01", help="Start date for data collection")
    p.add_argument("--to", default=None, help="End date (default: today)")
    p.add_argument("--out", default="data/pilot_fact_class_month_tradefeeds.parquet", help="Output file")
    p.add_argument("--fees", default="data/fees_turnover_override.csv", help="Fee override file")
    p.add_argument("--extra-history", type=int, default=36, help="Extra months for alpha calculation")
    p.add_argument("--use-sec", action="store_true", help="Use SEC data instead of Tradefeeds (fallback)")
    
    args = p.parse_args()
    
    # Check for API key
    if not args.use_sec and not os.getenv('TRADEFEEDS_API_KEY'):
        log.error("TRADEFEEDS_API_KEY not found in environment")
        log.error("Please set: export TRADEFEEDS_API_KEY='your_key_here'")
        log.error("Or use --use-sec flag to fallback to SEC data")
        sys.exit(1)
    
    appcfg = AppConfig()
    
    # Calculate date range
    since_date = pd.to_datetime(args.since)
    to_date = pd.to_datetime(args.to) if args.to else pd.Timestamp.now()
    
    # Extend start date for rolling window calculations
    historical_start_date = since_date - pd.DateOffset(months=args.extra_history)
    effective_since = historical_start_date.strftime("%Y-%m-%d")
    effective_to = to_date.strftime("%Y-%m-%d")
    
    log.info("="*60)
    log.info("TRADEFEEDS ETL PIPELINE")
    log.info("="*60)
    log.info(f"Requested period: {args.since} to {effective_to}")
    log.info(f"Fetching data from {effective_since} ({args.extra_history} months earlier) for alpha calculation")
    
    # Load pilot configuration
    regs = load_pilot_list(Path(args.pilot))
    if not regs:
        log.error("No registrants found in configuration")
        sys.exit(2)
    
    log.info(f"Loaded {len(regs)} registrants from config")
    
    # Initialize series/class mapper
    log.info("Initializing series/class mapper...")
    mapper = SeriesClassMapper(cache_path="data/series_class_mapping_cache.csv")
    
    if args.use_sec:
        log.warning("Using SEC data source (fallback mode)")
        # Import and run original SEC pipeline
        from .pilot_pipeline import main as sec_main
        return sec_main()
    
    # Initialize Tradefeeds client
    log.info("Initializing Tradefeeds API client...")
    try:
        client = TradeFeedsClient()
        fetcher = TradeFeedsReturnsFetcher(api_client=client)
    except Exception as e:
        log.error(f"Failed to initialize Tradefeeds client: {e}")
        sys.exit(1)
    
    # Test API connectivity
    if not client.test_api_connectivity():
        log.error("Failed to connect to Tradefeeds API")
        log.error("Check API key and network connection")
        sys.exit(1)
    
    log.info("✓ Tradefeeds API connection successful")
    
    # Prepare pilot configuration for fetcher
    pilot_config = {'registrants': regs}
    
    # Fetch monthly returns from Tradefeeds OHLCV API
    log.info("Fetching monthly returns from Tradefeeds OHLCV API...")
    log.info("This may take several minutes for large date ranges...")
    
    # Use chunking for large date ranges to get returns
    facts = fetcher.fetch_with_chunking(
        pilot_config=pilot_config,
        date_from=effective_since,
        date_to=effective_to,
        chunk_months=12  # Fetch 1 year at a time
    )
    
    if facts.empty:
        log.error("No return data retrieved from Tradefeeds OHLCV API")
        sys.exit(3)
    
    log.info(f"Retrieved {len(facts)} monthly returns from Tradefeeds OHLCV")
    
    # Data quality validation for returns
    facts = fetcher.validate_data_quality(facts)
    
    # Fetch TNA and flow data from SEC N-PORT filings
    log.info("Fetching TNA and flow data from SEC N-PORT filings...")
    tna_data = fetch_tna_from_sec_nport(regs, effective_since, effective_to, appcfg, mapper)
    
    # Extract flow data from SEC N-PORT (primary source)
    sec_flow_data = None
    if not tna_data.empty and all(col in tna_data.columns for col in ['sales', 'redemptions', 'reinvest']):
        sec_flow_data = tna_data[['class_id', 'month_end', 'series_id', 'sales', 'redemptions', 'reinvest']].copy()
        sec_flow_data['data_source'] = 'sec'
        log.info(f"Extracted {len(sec_flow_data)} flow records from SEC N-PORT data")
    
    # Use hybrid flow data approach: SEC primary, Tradefeeds fallback
    log.info("Fetching comprehensive flow data using hybrid approach...")
    comprehensive_flows = fetcher.fetch_flow_data_hybrid(
        pilot_config=pilot_config,
        date_from=effective_since,
        date_to=effective_to,
        sec_flow_data=sec_flow_data
    )
    
    # Merge TNA data with returns
    if not tna_data.empty:
        log.info(f"Merging TNA data ({len(tna_data)} records) with returns")
        facts = facts.merge(
            tna_data[['class_id', 'month_end', 'total_investments']],
            on=['class_id', 'month_end'],
            how='left'
        )
        tna_coverage = facts['total_investments'].notna().sum()
        log.info(f"TNA data coverage: {tna_coverage}/{len(facts)} ({tna_coverage/len(facts)*100:.1f}%)")
    else:
        log.warning("No TNA data retrieved from SEC filings")
        facts['total_investments'] = None
    
    # Merge comprehensive flow data with returns
    if not comprehensive_flows.empty:
        log.info(f"Merging comprehensive flow data ({len(comprehensive_flows)} records) with returns")
        
        # Ensure flow data has required columns
        flow_merge_cols = ['class_id', 'month_end']
        flow_data_cols = ['sales', 'redemptions', 'reinvest']
        
        for col in flow_data_cols:
            if col not in comprehensive_flows.columns:
                comprehensive_flows[col] = 0
        
        # Remove any existing flow columns from facts to avoid duplicates
        for col in flow_data_cols:
            if col in facts.columns:
                facts = facts.drop(columns=[col])
        
        facts = facts.merge(
            comprehensive_flows[flow_merge_cols + flow_data_cols],
            on=flow_merge_cols,
            how='left'
        )
        
        flow_coverage = facts['sales'].notna().sum()
        log.info(f"Flow data coverage: {flow_coverage}/{len(facts)} ({flow_coverage/len(facts)*100:.1f}%)")
    else:
        log.warning("No flow data retrieved from any source")
        # Add placeholder flow columns
        for col in ['sales', 'redemptions', 'reinvest']:
            if col not in facts.columns:
                facts[col] = 0
    
    # Log data summary
    log.info("="*60)
    log.info("DATA SUMMARY")
    log.info("="*60)
    log.info(f"Total records: {len(facts)}")
    log.info(f"Date range: {facts['month_end'].min()} to {facts['month_end'].max()}")
    log.info(f"Unique series: {facts['series_id'].nunique()}")
    log.info(f"Unique classes: {facts['class_id'].nunique()}")
    
    # Check for sufficient historical data
    months_available = facts.groupby('class_id')['month_end'].nunique()
    sufficient_history = (months_available >= 36).sum()
    log.info(f"Classes with 36+ months: {sufficient_history}/{len(months_available)} ({sufficient_history/len(months_available)*100:.1f}%)")
    
    # Compute flow metrics if flow data is available
    if all(col in facts.columns for col in ['sales', 'redemptions', 'reinvest']):
        log.info("Computing flow metrics...")
        facts = compute_net_flow(facts)
        facts = compute_flow_volatility(facts)
    else:
        log.info("Flow data not available - skipping flow metrics")
        facts['net_flow'] = None
        facts['flow_volatility'] = None
    
    # Fetch and merge factor data
    log.info("Fetching Fama-French factor data...")
    fac = get_monthly_ff5_mom(appcfg)
    log.info(f"Factor data shape: {fac.shape}")
    log.info(f"Factor date range: {fac['month_end'].min()} to {fac['month_end'].max()}")
    
    # Merge with factors
    facts_with_factors = facts.merge(fac, on="month_end", how="left")
    log.info(f"Facts + factors shape: {facts_with_factors.shape}")
    
    # Critical fix: Deduplicate class_id + month_end combinations to prevent multicollinearity
    log.info("Deduplicating class-month combinations...")
    initial_count = len(facts_with_factors)
    
    # Sort by completeness and keep most complete record
    factor_cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"]
    facts_with_factors['factor_completeness'] = facts_with_factors[factor_cols].count(axis=1)
    facts_with_factors = facts_with_factors.sort_values(['class_id', 'month_end', 'factor_completeness'])
    facts_with_factors = facts_with_factors.drop_duplicates(subset=['class_id', 'month_end'], keep='last')
    facts_with_factors = facts_with_factors.drop(columns=['factor_completeness'])
    
    final_count = len(facts_with_factors)
    log.info(f"Deduplication: {initial_count} -> {final_count} records ({initial_count - final_count} duplicates removed)")
    
    # Check factor data availability
    factor_counts = {col: facts_with_factors[col].count() for col in factor_cols if col in facts_with_factors.columns}
    log.info(f"Factor data availability: {factor_counts}")
    
    # Compute rolling factor regressions (36-month window)
    log.info("Computing 36-month rolling factor regressions...")
    regstats = rolling_factor_regressions(facts_with_factors)
    
    if len(regstats) > 0:
        log.info(f"Factor regressions produced {len(regstats)} results")
        
        # Merge regression results back
        # Use the actual column names from the regression results
        reg_pivot = regstats.pivot_table(
            index=["class_id", "month_end"],
            values=["alpha_hat", "R2"],
            aggfunc="first"
        ).reset_index()
        
        reg_pivot.columns = ["class_id", "month_end", "realized alpha", "adj_r_squared"]
        
        facts_with_factors = facts_with_factors.merge(
            reg_pivot,
            on=["class_id", "month_end"],
            how="left"
        )
        
        # Report alpha coverage
        alpha_coverage = facts_with_factors["realized alpha"].notna().sum()
        total_records = len(facts_with_factors)
        log.info(f"Realized alpha coverage: {alpha_coverage}/{total_records} ({alpha_coverage/total_records*100:.1f}%)")
        
        # Check coverage for originally requested period
        original_start = pd.to_datetime(args.since)
        recent_data = facts_with_factors[facts_with_factors['month_end'] >= original_start]
        recent_alpha = recent_data["realized alpha"].notna().sum()
        log.info(f"Alpha coverage for {args.since} onwards: {recent_alpha}/{len(recent_data)} ({recent_alpha/len(recent_data)*100:.1f}%)")
    else:
        log.warning("No factor regression results produced")
        facts_with_factors["realized alpha"] = None
        facts_with_factors["adj_r_squared"] = None
    
    # Get expense ratios and turnover from SEC RR datasets
    log.info("Fetching expense ratios and turnover from SEC RR datasets...")
    
    try:
            sec_rr_loader = SECRRDataLoader(base_dir="sec_rr_datasets", use_series_mapping=True)
            
            # Extract unique CIKs
            unique_ciks = facts_with_factors['cik'].unique().tolist() if 'cik' in facts_with_factors.columns else []
            
            if unique_ciks:
                log.info(f"Extracting SEC RR data for {len(unique_ciks)} unique CIKs")
                sec_rr_df = sec_rr_loader.extract_expense_turnover(ciks=unique_ciks)
                
                if not sec_rr_df.empty:
                    log.info(f"Found SEC RR data for {len(sec_rr_df)} records")
                    sec_rr_df = sec_rr_df.rename(columns={
                        'expense_ratio': 'net_expense_ratio',
                        'turnover_rate': 'turnover_pct'
                    })
                    er_turnover_df = sec_rr_df
                else:
                    log.warning("No SEC RR data found")
                    er_turnover_df = pd.DataFrame()
            else:
                er_turnover_df = pd.DataFrame()
                
    except Exception as e:
        log.error(f"Failed to load SEC RR data: {e}")
        er_turnover_df = pd.DataFrame()
    
    # Get manager tenure data
    log.info("Fetching manager tenure and fund age...")
    manager_df = get_manager_data_for_entities(regs)
    
    # Merge expense ratio and turnover data
    if not er_turnover_df.empty:
        log.info(f"Merging expense ratio and turnover data...")
        
        # Merge strategy based on available identifiers
        if 'series_id' in er_turnover_df.columns and 'series_id' in facts_with_factors.columns:
            facts_with_factors = facts_with_factors.merge(
                er_turnover_df[['series_id', 'net_expense_ratio', 'turnover_pct']].drop_duplicates(),
                on='series_id',
                how='left'
            )
        elif 'class_id' in er_turnover_df.columns:
            facts_with_factors = facts_with_factors.merge(
                er_turnover_df[['class_id', 'net_expense_ratio', 'turnover_pct']].drop_duplicates(),
                on='class_id',
                how='left'
            )
        
        expense_coverage = facts_with_factors['net_expense_ratio'].notna().sum()
        turnover_coverage = facts_with_factors['turnover_pct'].notna().sum()
        log.info(f"Expense ratio coverage: {expense_coverage}/{len(facts_with_factors)} ({expense_coverage/len(facts_with_factors)*100:.1f}%)")
        log.info(f"Turnover coverage: {turnover_coverage}/{len(facts_with_factors)} ({turnover_coverage/len(facts_with_factors)*100:.1f}%)")
    
    # Merge manager tenure data
    if not manager_df.empty:
        log.info("Merging manager tenure data...")
        if 'series_id' in manager_df.columns and 'series_id' in facts_with_factors.columns:
            facts_with_factors = facts_with_factors.merge(
                manager_df,
                on='series_id',
                how='left'
            )
    
    # Apply data overrides if file exists
    if Path(args.fees).exists():
        log.info(f"Applying data overrides from {args.fees}")
        facts_with_factors = apply_data_overrides(facts_with_factors, args.fees)
    
    # Compute value added metric
    log.info("Computing value added metric...")
    
    # Debug: Check available columns
    log.info(f"Facts with factors columns: {list(facts_with_factors.columns)}")
    
    # Find the correct class_id column (handle merge duplicates)
    class_id_col = 'class_id'
    if 'class_id_x' in facts_with_factors.columns:
        class_id_col = 'class_id_x'
    elif 'class_id_y' in facts_with_factors.columns:
        class_id_col = 'class_id_y'
    
    # Create TNA proxy from total_investments for value added calculation
    if 'total_investments' in facts_with_factors.columns and class_id_col in facts_with_factors.columns:
        tna_proxy = facts_with_factors[[class_id_col, 'month_end', 'total_investments']].rename(
            columns={'total_investments': 'tna', class_id_col: 'class_id'}
        ).dropna()
    else:
        log.warning("total_investments or class_id column not found, skipping value added calculation")
        tna_proxy = pd.DataFrame()
    
    # Create expense ratio map
    er_map = pd.DataFrame()
    if 'net_expense_ratio' in facts_with_factors.columns and class_id_col in facts_with_factors.columns:
        er_map = facts_with_factors[[class_id_col, 'net_expense_ratio']].rename(
            columns={class_id_col: 'class_id'}
        ).drop_duplicates().dropna()
    
    # Clean up column names first (consolidate duplicate columns from merges)
    if 'class_id_x' in facts_with_factors.columns and 'class_id_y' in facts_with_factors.columns:
        # Use class_id_x as the primary class_id
        facts_with_factors['class_id'] = facts_with_factors['class_id_x']
        facts_with_factors = facts_with_factors.drop(columns=['class_id_x', 'class_id_y'])
    elif 'class_id_x' in facts_with_factors.columns:
        facts_with_factors['class_id'] = facts_with_factors['class_id_x']
        facts_with_factors = facts_with_factors.drop(columns=['class_id_x'])
    elif 'class_id_y' in facts_with_factors.columns:
        facts_with_factors['class_id'] = facts_with_factors['class_id_y']
        facts_with_factors = facts_with_factors.drop(columns=['class_id_y'])
    
    # Clean up CIK columns too if duplicated
    if 'cik_x' in facts_with_factors.columns and 'cik_y' in facts_with_factors.columns:
        facts_with_factors['cik'] = facts_with_factors['cik_x'].fillna(facts_with_factors['cik_y'])
        facts_with_factors = facts_with_factors.drop(columns=['cik_x', 'cik_y'])
    elif 'cik_x' in facts_with_factors.columns:
        facts_with_factors['cik'] = facts_with_factors['cik_x']
        facts_with_factors = facts_with_factors.drop(columns=['cik_x'])
    elif 'cik_y' in facts_with_factors.columns:
        facts_with_factors['cik'] = facts_with_factors['cik_y']
        facts_with_factors = facts_with_factors.drop(columns=['cik_y'])
    
    # Now compute value added with clean column names
    facts_with_factors = value_added(facts_with_factors, er_map, tna_proxy)
    
    # Final data summary
    log.info("="*60)
    log.info("FINAL OUTPUT SUMMARY")
    log.info("="*60)
    log.info(f"Output shape: {facts_with_factors.shape}")
    log.info(f"Date range: {facts_with_factors['month_end'].min()} to {facts_with_factors['month_end'].max()}")
    
    # Column availability summary
    key_columns = [
        'return', 'net_flow', 'flow_volatility', 'realized alpha',
        'net_expense_ratio', 'turnover_pct', 'manager_tenure', 'total_investments', 'value_added'
    ]
    
    for col in key_columns:
        if col in facts_with_factors.columns:
            coverage = facts_with_factors[col].notna().sum()
            pct = coverage / len(facts_with_factors) * 100
            log.info(f"{col:20s}: {coverage:6d}/{len(facts_with_factors)} ({pct:5.1f}%)")
    
    # Save output
    output_path = Path(args.out)
    output_path.parent.mkdir(exist_ok=True)
    facts_with_factors.to_parquet(output_path, index=False)
    log.info(f"✓ Saved {len(facts_with_factors)} records to {output_path}")
    
    # Generate override template if needed
    if not Path(args.fees).exists():
        log.info(f"Generating override template at {args.fees}")
        generate_override_template(facts_with_factors, args.fees)
    
    log.info("="*60)
    log.info("✓ TRADEFEEDS ETL PIPELINE COMPLETE")
    log.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())