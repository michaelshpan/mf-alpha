#!/usr/bin/env python3
"""
Debug early pipeline steps to find where we lose data from 783 to 213
"""
import argparse, logging, sys
from pathlib import Path
import pandas as pd, yaml
from tqdm import tqdm
from etl.config import AppConfig
from etl.sec_edgar import list_recent_nport_p_accessions, download_filing_xml
from etl.nport_parser_fixed import parse_nport_primary_xml
from etl.metrics import compute_net_flow, compute_flow_volatility

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("debug")

def load_pilot_list(path: Path) -> list[dict]:
    cfg = yaml.safe_load(path.read_text())
    return cfg.get("registrants", [])

def debug_pipeline():
    appcfg = AppConfig()
    regs = load_pilot_list(Path("config/funds_pilot.yaml"))
    
    if not regs:
        log.error("No registrants found")
        sys.exit(2)
    
    rows = []
    for reg in regs:
        cik = reg.get("cik")
        if not cik:
            log.warning("Skip registrant without CIK: %s", reg)
            continue
            
        log.info("Processing CIK %s (%s)", cik, reg.get("name", ""))
        filings = list_recent_nport_p_accessions(cik, appcfg, since_yyyymmdd="2023-01-01")
        if not filings:
            log.info("No NPORT-P filings since 2023-01-01 for CIK %s", cik)
            continue
            
        for f in tqdm(filings, desc=f"CIK {cik} filings"):
            try:
                xml = download_filing_xml(cik, f["accession"], f["primary_doc"], appcfg)
                df = parse_nport_primary_xml(xml)
                if not df.empty:
                    df["cik"] = str(int(cik))
                    df["filing_date"] = pd.to_datetime(f["filing_date"])
                    rows.append(df)
                    log.info("Added %d rows from %s", len(df), f["accession"])
            except Exception as e:
                log.exception("Parse failure for %s %s: %s", f["accession"], f["primary_doc"], e)
    
    if not rows:
        log.error("No class-month rows extracted.")
        return
    
    # Step 1: Concat raw data
    raw_facts = pd.concat(rows, ignore_index=True)
    log.info("✅ Step 1 - Raw concat: %d rows", len(raw_facts))
    
    # Step 2: Compute net flow
    facts_with_flow = compute_net_flow(raw_facts)
    log.info("✅ Step 2 - After net_flow: %d rows", len(facts_with_flow))
    
    # Step 3: Compute flow volatility
    facts_final = compute_flow_volatility(facts_with_flow)
    log.info("✅ Step 3 - After flow_volatility: %d rows", len(facts_final))
    
    # Check data consistency
    print(f"\n📊 Data summary:")
    print(f"  Classes: {facts_final['class_id'].nunique()}")
    print(f"  Date range: {facts_final['month_end'].min()} to {facts_final['month_end'].max()}")
    print(f"  Avg rows per class: {len(facts_final) / facts_final['class_id'].nunique():.1f}")
    
    # Save raw data to debug
    facts_final.to_parquet("debug_raw_facts.parquet", index=False)
    print(f"✅ Saved debug data to debug_raw_facts.parquet")

if __name__ == "__main__":
    debug_pipeline()