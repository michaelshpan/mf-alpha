#!/usr/bin/env python3
"""
Test script to verify that the enhanced pipeline provides full alpha coverage.
Compares old and new pipeline outputs.
"""

import pandas as pd
import sys
from pathlib import Path

def compare_alpha_coverage():
    """Compare alpha coverage between old and new pipeline outputs."""
    
    print("="*70)
    print("TESTING ENHANCED PIPELINE ALPHA COVERAGE")
    print("="*70)
    
    # Check if we have existing output to compare
    old_output = Path("optionA_etl/data/pilot_fact_class_month.parquet")
    
    if old_output.exists():
        print("\nLoading existing pipeline output for comparison...")
        old_df = pd.read_parquet(old_output)
        
        # Analyze old alpha coverage
        old_alpha_coverage = old_df['realized alpha'].notna().sum()
        old_total = len(old_df)
        old_pct = (old_alpha_coverage / old_total * 100) if old_total > 0 else 0
        
        print(f"\nOLD PIPELINE RESULTS:")
        print(f"  Total records: {old_total:,}")
        print(f"  Records with alpha: {old_alpha_coverage:,}")
        print(f"  Coverage: {old_pct:.1f}%")
        print(f"  Date range: {old_df['month_end'].min()} to {old_df['month_end'].max()}")
        
        # Analyze by year
        old_df['year'] = old_df['month_end'].dt.year
        old_yearly = old_df.groupby('year').agg({
            'class_id': 'count',
            'realized alpha': lambda x: x.notna().sum()
        }).rename(columns={'class_id': 'total', 'realized alpha': 'with_alpha'})
        old_yearly['coverage'] = (old_yearly['with_alpha'] / old_yearly['total'] * 100).round(1)
        
        print("\n  Coverage by year:")
        print(old_yearly.to_string())
    else:
        print(f"\nNo existing output found at {old_output}")
        print("Run the pipeline first to generate baseline data.")
    
    print("\n" + "-"*70)
    print("\nTo test the enhanced pipeline, run:")
    print("\n  cd optionA_etl")
    print("  python -m etl.pilot_pipeline --pilot config/funds_pilot.yaml \\")
    print("    --since 2023-01-01 --out data/pilot_enhanced.parquet")
    print("\nThis will fetch 36 extra months of historical data (back to 2020-01-01)")
    print("to enable alpha calculations from the first month of the requested period.")
    
    print("\n" + "-"*70)
    print("\nEXPECTED IMPROVEMENTS:")
    print("  1. Nearly 100% alpha coverage for months from 2023 onwards")
    print("  2. Alpha available from month 1 for all funds (no 36-month delay)")
    print("  3. More accurate backtesting with complete alpha data")
    
    # Check if enhanced output exists
    enhanced_output = Path("optionA_etl/data/pilot_enhanced.parquet")
    if enhanced_output.exists():
        print("\n" + "="*70)
        print("ENHANCED PIPELINE RESULTS FOUND!")
        print("="*70)
        
        new_df = pd.read_parquet(enhanced_output)
        
        # Analyze new alpha coverage
        new_alpha_coverage = new_df['realized alpha'].notna().sum()
        new_total = len(new_df)
        new_pct = (new_alpha_coverage / new_total * 100) if new_total > 0 else 0
        
        print(f"\nENHANCED PIPELINE RESULTS:")
        print(f"  Total records: {new_total:,}")
        print(f"  Records with alpha: {new_alpha_coverage:,}")
        print(f"  Coverage: {new_pct:.1f}%")
        print(f"  Date range: {new_df['month_end'].min()} to {new_df['month_end'].max()}")
        
        # Analyze by year
        new_df['year'] = new_df['month_end'].dt.year
        new_yearly = new_df.groupby('year').agg({
            'class_id': 'count',
            'realized alpha': lambda x: x.notna().sum()
        }).rename(columns={'class_id': 'total', 'realized alpha': 'with_alpha'})
        new_yearly['coverage'] = (new_yearly['with_alpha'] / new_yearly['total'] * 100).round(1)
        
        print("\n  Coverage by year:")
        print(new_yearly.to_string())
        
        if old_output.exists():
            print("\n" + "="*70)
            print("IMPROVEMENT SUMMARY")
            print("="*70)
            
            improvement = new_alpha_coverage - old_alpha_coverage
            pct_improvement = new_pct - old_pct
            
            print(f"\n  Additional records with alpha: {improvement:,}")
            print(f"  Coverage improvement: {pct_improvement:+.1f} percentage points")
            
            # Focus on 2023+ data
            old_2023 = old_df[old_df['month_end'] >= '2023-01-01']
            new_2023 = new_df[new_df['month_end'] >= '2023-01-01']
            
            old_2023_cov = (old_2023['realized alpha'].notna().sum() / len(old_2023) * 100) if len(old_2023) > 0 else 0
            new_2023_cov = (new_2023['realized alpha'].notna().sum() / len(new_2023) * 100) if len(new_2023) > 0 else 0
            
            print(f"\n  2023+ Coverage:")
            print(f"    Old: {old_2023_cov:.1f}%")
            print(f"    New: {new_2023_cov:.1f}%")
            print(f"    Improvement: {new_2023_cov - old_2023_cov:+.1f} percentage points")

if __name__ == "__main__":
    compare_alpha_coverage()