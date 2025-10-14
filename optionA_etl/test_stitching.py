#!/usr/bin/env python3
"""
Quick test of the enhanced monthly return stitching logic
"""

from etl.config import AppConfig
from etl.sec_edgar import list_recent_nport_p_accessions, download_filing_xml
from etl.nport_parser_fixed import parse_nport_primary_xml
from etl.series_class_mapper import SeriesClassMapper
import pandas as pd

def test_monthly_stitching():
    print("Testing enhanced monthly return stitching...")
    
    # Test with Vanguard (smaller dataset)
    cik = '0000932471'
    cfg = AppConfig()
    
    # Get filings since 2020 (should have multiple quarters)
    print(f"Fetching N-PORT-P filings for CIK {cik} since 2020...")
    filings = list_recent_nport_p_accessions(cik, cfg, since_yyyymmdd='20200101')
    print(f"Found {len(filings)} filings")
    
    # Initialize mapper
    mapper = SeriesClassMapper(cache_path="data/series_class_mapping_cache.csv")
    
    # Target series for this CIK
    target_series = ['S000002594']
    valid_class_ids = set()
    for series_id in target_series:
        class_ids = mapper.get_class_for_series(series_id)
        if class_ids:
            valid_class_ids.update(class_ids)
            print(f"Series {series_id}: {class_ids}")
    
    # Process more filings to get sufficient data
    rows = []
    for i, filing in enumerate(filings):  # Process all filings
        if i % 10 == 0 or i < 5:  # Show progress
            print(f"Processing filing {i+1}/{len(filings)}: {filing['filing_date']}")
        try:
            xml = download_filing_xml(cik, filing['accession'], filing['primary_doc'], cfg)
            df = parse_nport_primary_xml(xml)
            
            if not df.empty:
                df["cik"] = str(int(cik))
                df["filing_date"] = pd.to_datetime(filing["filing_date"])
                
                # Filter to target classes
                df = df[df["class_id"].isin(valid_class_ids)]
                if not df.empty:
                    rows.append(df)
                    print(f"  Extracted {len(df)} records")
                else:
                    print("  No records for target classes")
            else:
                print("  No data parsed")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    if not rows:
        print("No data extracted!")
        return
    
    # Combine and analyze
    facts = pd.concat(rows, ignore_index=True)
    print(f"\nRaw data: {len(facts)} records")
    
    # Deduplicate (keep most recent filing for each class_id-month_end)
    print("Deduplicating...")
    original_count = len(facts)
    facts = facts.sort_values(['class_id', 'month_end', 'filing_date'])
    facts = facts.drop_duplicates(subset=['class_id', 'month_end'], keep='last')
    print(f"Deduplication: {original_count} → {len(facts)} records (removed {original_count - len(facts)} duplicates)")
    
    # Analyze results
    print(f"\nAnalysis:")
    print(f"Unique class IDs: {facts['class_id'].nunique()}")
    print(f"Date range: {facts['month_end'].min()} to {facts['month_end'].max()}")
    
    # Check monthly coverage for one fund
    test_class = facts['class_id'].iloc[0]
    class_data = facts[facts['class_id'] == test_class].sort_values('month_end')
    
    print(f"\nMonthly coverage for {test_class}:")
    print(f"Total months: {len(class_data)}")
    
    # Check for consecutive months
    class_data['month_end'] = pd.to_datetime(class_data['month_end'])
    gaps = class_data['month_end'].diff().dt.days
    normal_gaps = (gaps >= 25) & (gaps <= 35)  # ~30 days = normal monthly gap
    large_gaps = gaps > 60
    
    print(f"Normal monthly gaps: {normal_gaps.sum()}")
    print(f"Large gaps (>60 days): {large_gaps.sum()}")
    
    # Show sample data
    print(f"\nFirst 10 months for {test_class}:")
    sample = class_data.head(10)[['month_end', 'return', 'filing_date']]
    for _, row in sample.iterrows():
        print(f"  {row['month_end'].strftime('%Y-%m')}: return={row['return']:.4f} (filed {row['filing_date'].strftime('%Y-%m-%d')})")
    
    # Check if we have 36+ months for any fund
    months_per_fund = facts.groupby('class_id')['month_end'].nunique()
    funds_36_plus = (months_per_fund >= 36).sum()
    print(f"\nFunds with 36+ months: {funds_36_plus}/{len(months_per_fund)}")
    if funds_36_plus > 0:
        print("SUCCESS: Sufficient data for alpha calculation!")
    else:
        max_months = months_per_fund.max()
        print(f"Maximum months per fund: {max_months}")
        print("Need more data for alpha calculation")

if __name__ == "__main__":
    test_monthly_stitching()