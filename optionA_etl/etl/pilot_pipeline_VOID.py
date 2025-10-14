import argparse, logging, sys
from pathlib import Path
import pandas as pd, yaml
from tqdm import tqdm
from .config import AppConfig
from .sec_edgar import list_recent_nport_p_accessions, download_filing_xml
from .nport_parser_fixed import parse_nport_primary_xml
from .factors import get_monthly_ff5_mom
from .metrics import compute_net_flow, compute_flow_volatility, compute_realized_alpha_lagged, rolling_factor_regressions, value_added
from .oef_rr_extractor_robust import get_er_turnover_for_entities
from .manager_tenure import get_manager_data_for_entities
from .series_class_mapper import SeriesClassMapper
from .sec_rr_integration import SECRRDataLoader
from .data_overrides import apply_data_overrides, generate_override_template
from .returns_database import MonthlyReturnsDatabase
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log=logging.getLogger("pilot")
def load_pilot_list(path: Path)->list[dict]:
    cfg=yaml.safe_load(path.read_text()); return cfg.get("registrants",[])
def main():
    p=argparse.ArgumentParser(description="Option A pilot ETL")
    p.add_argument("--pilot", default="config/funds_pilot.yaml")
    p.add_argument("--since", default="2023-01-01")
    p.add_argument("--out", default="data/pilot_fact_class_month.parquet")
    p.add_argument("--fees", default="data/fees_turnover_override.csv")
    p.add_argument("--extra-history", type=int, default=36, help="Extra months of historical data to fetch for alpha calculation (default: 36)")
    p.add_argument("--use-returns-db", action="store_true", help="Use SEC bulk returns database for complete monthly data")
    p.add_argument("--update-returns-db", action="store_true", help="Update the returns database before processing")
    p.add_argument("--returns-db-path", default="data/monthly_returns_db", help="Path to returns database")
    a=p.parse_args()
    appcfg=AppConfig()
    
    # Adjust the start date to fetch extra historical data for rolling window calculations
    since_date = pd.to_datetime(a.since)
    historical_start_date = since_date - pd.DateOffset(months=a.extra_history)
    effective_since = historical_start_date.strftime("%Y-%m-%d")
    log.info("Original --since date: %s", a.since)
    log.info("Fetching filings from %s (%d months earlier) for regression window", effective_since, a.extra_history)
    regs=load_pilot_list(Path(a.pilot))
    if not regs: log.error("No registrants found"); sys.exit(2)
    
    # Initialize series/class mapper (automatically loads mapping in __init__)
    log.info("Initializing series/class mapper...")
    mapper = SeriesClassMapper(cache_path="data/series_class_mapping_cache.csv")
    
    # Build lookup of series -> CIK from config
    series_to_cik = {}
    cik_series_map = {}  # CIK -> list of series_ids
    for reg in regs:
        cik = reg.get("cik")
        if cik:
            cik_str = str(int(cik))
            series_ids = reg.get("series_ids", [])
            cik_series_map[cik_str] = series_ids
            for sid in series_ids:
                series_to_cik[sid] = cik_str
    
    log.info("Config specifies %d series IDs across %d CIKs", len(series_to_cik), len(cik_series_map))
    
    # Get all valid class IDs for configured series
    valid_class_ids = set()
    series_to_classes = {}
    for series_id in series_to_cik.keys():
        class_ids = mapper.get_class_for_series(series_id)
        if class_ids:
            valid_class_ids.update(class_ids)
            series_to_classes[series_id] = class_ids
            log.info("Series %s has %d class IDs: %s", series_id, len(class_ids), class_ids[:3])
        else:
            log.warning("No class IDs found for series %s", series_id)
    
    log.info("Total valid class IDs from configured series: %d", len(valid_class_ids))
    
    rows=[]
    for reg in regs:
        cik=reg.get("cik"); 
        if not cik: log.warning("Skip registrant without CIK: %s", reg); continue
        sids=reg.get("series_ids") or []; cids=reg.get("class_ids") or []
        log.info("Processing CIK %s (%s) with series %s", cik, reg.get("name",""), sids)
        filings=list_recent_nport_p_accessions(cik, appcfg, since_yyyymmdd=effective_since)
        if not filings: log.info("No NPORT-P filings since %s for CIK %s", effective_since, cik); continue
        for f in tqdm(filings, desc=f"CIK {cik} filings"):
            try:
                xml=download_filing_xml(cik, f["accession"], f["primary_doc"], appcfg)
                df=parse_nport_primary_xml(xml)
                if not df.empty:
                    df["cik"]=str(int(cik))
                    df["filing_date"]=pd.to_datetime(f["filing_date"])
                    
                    # Filter to only include class_ids that belong to configured series
                    if len(valid_class_ids) > 0:
                        orig_len = len(df)
                        df = df[df["class_id"].isin(valid_class_ids)].copy()
                        log.debug("Filtered CIK %s data from %d to %d rows (valid classes only)", cik, orig_len, len(df))
                    
                    # Add series_id column based on class_id mapping
                    if not df.empty:
                        df["series_id"] = df["class_id"].apply(lambda cid: mapper.get_series_for_class(cid))
                        rows.append(df)
            except Exception as e:
                log.exception("Parse failure for %s %s: %s", f["accession"], f["primary_doc"], e)
    
    if not rows: log.error("No class-month rows extracted."); sys.exit(3)
    facts=pd.concat(rows, ignore_index=True)
    
    # Deduplicate N-PORT data (keep most recent filing for each class-month)
    log.info("Deduplicating N-PORT data...")
    original_count = len(facts)
    facts = facts.sort_values(['class_id', 'month_end', 'filing_date'])
    facts = facts.drop_duplicates(subset=['class_id', 'month_end'], keep='last')
    log.info(f"Deduplicated: {original_count} → {len(facts)} records (removed {original_count - len(facts)} duplicates)")
    
    # Use SEC bulk returns database if requested
    if a.use_returns_db:
        log.info("Using SEC bulk returns database for complete monthly data")
        returns_db = MonthlyReturnsDatabase(db_path=a.returns_db_path)
        
        # Update database if requested
        if a.update_returns_db:
            log.info("Updating returns database with latest SEC data...")
            returns_db.initialize_from_pilot_config(a.pilot)
        
        # Get database statistics
        db_stats = returns_db.get_database_stats()
        if db_stats.get('initialized'):
            log.info(f"Returns database: {db_stats.get('total_records', 0)} records, "
                    f"{db_stats.get('unique_classes', 0)} funds")
            
            # Enhance facts with complete monthly returns
            facts_before = len(facts)
            returns_before = facts['return'].notna().sum()
            
            # Get complete monthly returns for each fund
            enhanced_facts = []
            for class_id in facts['class_id'].unique():
                # Get date range for this fund
                fund_data = facts[facts['class_id'] == class_id]
                min_date = fund_data['month_end'].min() - pd.DateOffset(months=a.extra_history)
                max_date = fund_data['month_end'].max()
                
                # Get complete monthly series from database
                db_returns = returns_db.get_complete_monthly_series(
                    class_id=class_id,
                    start_date=min_date.strftime('%Y-%m-%d'),
                    end_date=max_date.strftime('%Y-%m-%d'),
                    fill_method='none'
                )
                
                if not db_returns.empty:
                    # Merge with existing fund data
                    merged = fund_data.merge(
                        db_returns[['month_end', 'return']].rename(columns={'return': 'db_return'}),
                        on='month_end',
                        how='outer'
                    )
                    
                    # Use database return where available
                    merged['return'] = merged['db_return'].fillna(merged['return'])
                    
                    # Fill missing metadata
                    for col in ['class_id', 'cik', 'series_id']:
                        if col in merged.columns:
                            merged[col] = merged[col].fillna(method='ffill').fillna(method='bfill')
                    
                    enhanced_facts.append(merged)
                else:
                    enhanced_facts.append(fund_data)
            
            if enhanced_facts:
                facts = pd.concat(enhanced_facts, ignore_index=True)
                facts = facts.drop(columns=['db_return'], errors='ignore')
                
                facts_after = len(facts)
                returns_after = facts['return'].notna().sum()
                
                log.info(f"Enhanced with database: {facts_before} → {facts_after} records")
                log.info(f"Returns coverage: {returns_before} → {returns_after} "
                        f"({returns_after/facts_after*100:.1f}% coverage)")
        else:
            log.warning("Returns database not initialized. Run with --update-returns-db first.")
    
    # Log series_id population status
    log.info("Facts shape after concat: %s", facts.shape)
    log.info("Series ID population: %d/%d rows have series_id", 
             facts["series_id"].notna().sum(), len(facts))
    log.info("Unique series IDs in data: %s", sorted(facts["series_id"].dropna().unique()))
    
    facts=compute_net_flow(facts)
    facts=compute_flow_volatility(facts)
    
    # Fetch factor data - now we have historical fund data, so factors should align
    log.info("Facts shape before factor merge: %s", facts.shape)
    log.info("Facts date range: %s to %s", facts['month_end'].min(), facts['month_end'].max())
    
    # Get all available factor data to cover the full period
    fac=get_monthly_ff5_mom(appcfg)
    log.info("Factor data shape: %s", fac.shape)
    log.info("Factor data date range: %s to %s", fac['month_end'].min(), fac['month_end'].max())
    
    facts_with_factors=facts.merge(fac, on="month_end", how="left")
    log.info("Facts + factors shape: %s", facts_with_factors.shape)
    log.info("DEBUG: Unique month_end in facts: %d", facts["month_end"].nunique())
    log.info("DEBUG: Unique month_end in factors: %d", fac["month_end"].nunique())
    
    # Check factor data availability
    factor_counts = {col: facts_with_factors[col].count() for col in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"]}
    log.info("Factor data availability: %s", factor_counts)
    log.info("Return data: %d/%d", facts_with_factors["return"].count(), len(facts_with_factors))
    
    # Compute factor regressions on the full dataset with factors
    log.info("Computing rolling factor regressions on full dataset...")
    regstats=rolling_factor_regressions(facts_with_factors)
    log.info("Factor regressions produced %d results", len(regstats))
    
    # Report alpha coverage improvement
    if len(regstats) > 0:
        alpha_coverage = regstats.groupby('month_end').size()
        log.info("Alpha coverage by month: min=%d, max=%d, mean=%.1f", 
                 alpha_coverage.min(), alpha_coverage.max(), alpha_coverage.mean())
        
        # Check coverage for the originally requested period
        original_start = pd.to_datetime(a.since)
        recent_alpha = regstats[regstats['month_end'] >= original_start]
        log.info("Alpha coverage for requested period (%s onwards): %d/%d records", 
                 a.since, len(recent_alpha), 
                 len(facts_with_factors[facts_with_factors['month_end'] >= original_start]))
    
    # Get expense ratios and turnover from SEC RR datasets (primary source)
    log.info("Fetching expense ratios and turnover from SEC RR datasets...")
    
    # Initialize SEC RR data loader
    try:
        sec_rr_loader = SECRRDataLoader(base_dir="sec_rr_datasets", use_series_mapping=True)
        
        # Extract CIKs from the current data
        unique_ciks = facts_with_factors['cik'].unique().tolist()
        log.info(f"Extracting SEC RR data for {len(unique_ciks)} unique CIKs")
        
        # Get expense ratio and turnover data from SEC RR datasets
        sec_rr_df = sec_rr_loader.extract_expense_turnover(ciks=unique_ciks)
        
        if not sec_rr_df.empty:
            log.info(f"Found SEC RR data for {len(sec_rr_df)} records")
            # Rename columns to match expected format
            sec_rr_df = sec_rr_df.rename(columns={
                'expense_ratio': 'net_expense_ratio',
                'turnover_rate': 'turnover_pct'
            })
            er_turnover_df = sec_rr_df
        else:
            log.warning("No SEC RR data found")
            er_turnover_df = pd.DataFrame()
            
    except Exception as e:
        log.error(f"Failed to load SEC RR data: {e}")
        er_turnover_df = pd.DataFrame()
    
    # Fallback to N-1A parsing (DISABLED - will be activated later if requested)
    use_n1a_fallback = False  # Set to True when instructed
    
    if er_turnover_df.empty and use_n1a_fallback:
        log.info("Fallback: Fetching expense ratios and turnover from N-1A filings...")
        er_turnover_df = get_er_turnover_for_entities(regs)
    elif er_turnover_df.empty:
        log.info("SEC RR data not available, leaving expense/turnover fields blank")
    
    log.info("Fetching manager tenure and fund age from N-1A filings...")
    manager_df = get_manager_data_for_entities(regs)
    
    # Merge expense ratio and turnover data with proper class_id/series_id strategy
    log.info("DEBUG: Before expense/turnover merge shape: %s", facts_with_factors.shape)
    if not er_turnover_df.empty:
        log.info("DEBUG: SEC RR data shape: %s", er_turnover_df.shape)
        
        # Step 1: Merge expense ratios on class_id
        expense_records = er_turnover_df[er_turnover_df['net_expense_ratio'].notna() & er_turnover_df['class_id'].notna()]
        if not expense_records.empty:
            log.info(f"Merging {len(expense_records)} expense ratio records on class_id")
            facts_with_factors = facts_with_factors.merge(
                expense_records[['class_id', 'net_expense_ratio']].drop_duplicates(), 
                on='class_id', 
                how='left'
            )
        else:
            facts_with_factors['net_expense_ratio'] = None
            log.info("No expense ratio data available for class_id merge")
        
        # Step 2: Merge turnover rates on series_id  
        turnover_records = er_turnover_df[er_turnover_df['turnover_pct'].notna() & er_turnover_df['series_id'].notna()]
        if not turnover_records.empty:
            log.info(f"Merging {len(turnover_records)} turnover rate records on series_id")
            facts_with_factors = facts_with_factors.merge(
                turnover_records[['series_id', 'turnover_pct']].drop_duplicates(), 
                on='series_id', 
                how='left'
            )
        else:
            facts_with_factors['turnover_pct'] = None
            log.info("No turnover rate data available for series_id merge")
            
        log.info("DEBUG: After expense/turnover merge shape: %s", facts_with_factors.shape)
    else:
        log.info("No SEC RR data available, setting expense/turnover fields to None")
        facts_with_factors["net_expense_ratio"] = None
        facts_with_factors["turnover_pct"] = None
    
    # Merge manager and fund age data by CIK
    if not manager_df.empty:
        if manager_df['class_id'].notna().any():
            facts_with_factors = facts_with_factors.merge(manager_df[["class_id", "manager_tenure", "fund_age"]], 
                               on="class_id", how="left")
        else:
            # Merge on CIK and apply to all classes from that CIK
            facts_with_factors = facts_with_factors.merge(manager_df[["cik", "manager_tenure", "fund_age"]].drop_duplicates(subset=["cik"]), 
                               on="cik", how="left")
    else:
        facts_with_factors["manager_tenure"] = None
        facts_with_factors["fund_age"] = None
    
    # Apply manual data overrides for missing expense ratios, manager tenure, and fund age
    log.info("Applying manual data overrides for missing values...")
    
    # Generate template for missing data (optional - helps users identify what needs overriding)
    try:
        generate_override_template(facts_with_factors, "etl/data/missing_data_template.csv")
    except Exception as e:
        log.warning(f"Failed to generate override template: {e}")
    
    # Apply overrides from manual_overrides.csv file
    facts_with_factors = apply_data_overrides(facts_with_factors, "etl/data/manual_overrides.csv")
    
    if len(regstats) > 0:
        # Merge regression results into the facts_with_factors dataset (which includes external data)
        log.info("Merging regression results...")
        log.info("Main data classes: %s", sorted(facts_with_factors["class_id"].unique()))
        log.info("Regression data classes: %s", sorted(regstats["class_id"].unique()) if len(regstats) > 0 else [])
        
        # Ensure series_id is preserved during merge
        out=facts_with_factors.merge(regstats, on=["class_id","month_end"], how="left")
        
        # Debug: check how many got merged
        merged_results = out["alpha_hat"].notna().sum()
        log.info("Successfully merged %d rows with regression results", merged_results)
    else:
        log.warning("No regression results produced - adding empty columns")
        # No regression results - add empty columns
        out = facts_with_factors.copy()
        out["alpha_hat"] = None
        out["alpha_t"] = None
        out["market_beta_t"] = None
        out["size_beta_t"] = None
        out["value_beta_t"] = None
        out["profit_beta_t"] = None
        out["invest_beta_t"] = None
        out["momentum_beta_t"] = None
        out["R2"] = None
    # Add realized alpha lagged and compute value added before renaming
    log.info("DEBUG: Before realized_alpha_lagged shape: %s", out.shape)
    out = compute_realized_alpha_lagged(out)
    log.info("DEBUG: After realized_alpha_lagged shape: %s", out.shape)
    
    fees_path=Path(a.fees); fees_df=pd.read_csv(fees_path) if fees_path.exists() else pd.DataFrame()
    
    # Instead of using the problematic value_added function, compute value_added directly
    # to avoid the circular merge that causes data duplication
    out = out.copy()
    
    # Merge fee data if available
    if not fees_df.empty:
        out = out.merge(fees_df, on="class_id", how="left")
    
    # Compute lagged TNA for value added calculation
    out = out.sort_values(["class_id", "month_end"])
    out["tna_lag"] = out.groupby("class_id")["total_investments"].shift(1)
    
    # Calculate value added directly
    if "net_expense_ratio" in out.columns and "alpha_hat" in out.columns:
        out["value_added"] = (out["alpha_hat"] - out["net_expense_ratio"]) * out["tna_lag"]
    else:
        out["value_added"] = None
    
    log.info("DEBUG: After value_added computation shape: %s", out.shape)
    
    # Rename columns to match DeMiguel paper naming
    out = out.rename(columns={
        "alpha_hat": "realized alpha", 
        "realized_alpha_lagged": "realized alpha lagged",
        "alpha_t": "alpha (intercept t-stat)",
        "total_investments": "total net assets",
        "net_expense_ratio": "expense ratio",
        "fund_age": "age",
        "net_flow": "flows",
        "turnover_pct": "turnover ratio",
        "market_beta_t": "market beta t-stat",
        "profit_beta_t": "profit. beta t-stat", 
        "invest_beta_t": "invest. beta t-stat",
        "size_beta_t": "size beta t-stat",
        "value_beta_t": "value beta t-stat",
        "momentum_beta_t": "momentum beta t-stat"
    })
    # Final validation of series_id column
    log.info("Final output shape: %s", out.shape)
    log.info("Series ID in final output: %d/%d rows have series_id", 
             out["series_id"].notna().sum() if "series_id" in out.columns else 0, len(out))
    if "series_id" in out.columns:
        log.info("Unique series IDs in final output: %s", sorted(out["series_id"].dropna().unique()))
    else:
        log.error("WARNING: series_id column missing from final output!")
    
    out_path=Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(out_path, index=False)
    log.info("Wrote %s rows to %s with columns: %s", len(out), out_path, list(out.columns)[:10])
if __name__=="__main__": main()
