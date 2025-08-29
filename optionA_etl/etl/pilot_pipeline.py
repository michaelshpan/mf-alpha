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
    a=p.parse_args()
    appcfg=AppConfig()
    regs=load_pilot_list(Path(a.pilot))
    if not regs: log.error("No registrants found"); sys.exit(2)
    rows=[]
    for reg in regs:
        cik=reg.get("cik"); 
        if not cik: log.warning("Skip registrant without CIK: %s", reg); continue
        sids=reg.get("series_ids") or []; cids=reg.get("class_ids") or []
        log.info("Processing CIK %s (%s)", cik, reg.get("name",""))
        filings=list_recent_nport_p_accessions(cik, appcfg, since_yyyymmdd=a.since)
        if not filings: log.info("No NPORT-P filings since %s for CIK %s", a.since, cik); continue
        for f in tqdm(filings, desc=f"CIK {cik} filings"):
            try:
                xml=download_filing_xml(cik, f["accession"], f["primary_doc"], appcfg)
                df=parse_nport_primary_xml(xml)
                if not df.empty:
                    df["cik"]=str(int(cik)); df["filing_date"]=pd.to_datetime(f["filing_date"]); rows.append(df)
            except Exception as e:
                log.exception("Parse failure for %s %s: %s", f["accession"], f["primary_doc"], e)
    if not rows: log.error("No class-month rows extracted."); sys.exit(3)
    facts=pd.concat(rows, ignore_index=True); facts=compute_net_flow(facts)
    facts=compute_flow_volatility(facts)
    
    # Merge factor data FIRST to ensure full dataset for regressions
    log.info("Facts shape before factor merge: %s", facts.shape)
    fac=get_monthly_ff5_mom(appcfg)
    log.info("Factor data shape: %s", fac.shape)
    
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
    
    # Get expense ratios, turnover, and manager data
    log.info("Fetching expense ratios and turnover from OEF/RR filings...")
    er_turnover_df = get_er_turnover_for_entities(regs)
    
    log.info("Fetching manager tenure and fund age from N-1A filings...")
    manager_df = get_manager_data_for_entities(regs)
    
    # Merge expense ratio and turnover data by CIK (since class_id may be None in external data)
    log.info("DEBUG: Before OEF/RR merge shape: %s", facts_with_factors.shape)
    if not er_turnover_df.empty:
        log.info("DEBUG: OEF/RR data shape: %s", er_turnover_df.shape)
        
        # Check if any class_ids match between datasets
        facts_classes = set(facts_with_factors['class_id'].unique())
        oef_classes = set(er_turnover_df['class_id'].dropna().unique())
        matching_classes = facts_classes.intersection(oef_classes)
        log.info("DEBUG: Matching class_ids: %d", len(matching_classes))
        
        if len(matching_classes) > 0:
            # Direct class_id match - safe merge
            facts_with_factors = facts_with_factors.merge(er_turnover_df[["class_id", "net_expense_ratio", "turnover_pct"]], 
                               on="class_id", how="left")
        else:
            # No direct match - use fund-level data (take first record per CIK to avoid duplication)
            fund_level_data = er_turnover_df.groupby("cik")[["net_expense_ratio", "turnover_pct"]].first().reset_index()
            facts_with_factors = facts_with_factors.merge(fund_level_data, on="cik", how="left")
            
        log.info("DEBUG: After OEF/RR merge shape: %s", facts_with_factors.shape)
    else:
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
    
    if len(regstats) > 0:
        # Merge regression results into the facts_with_factors dataset (which includes external data)
        log.info("Merging regression results...")
        log.info("Main data classes: %s", sorted(facts_with_factors["class_id"].unique()))
        log.info("Regression data classes: %s", sorted(regstats["class_id"].unique()) if len(regstats) > 0 else [])
        
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
    out_path=Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(out_path, index=False)
    log.info("Wrote %s rows to %s", len(out), out_path)
if __name__=="__main__": main()
