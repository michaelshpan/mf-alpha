#!/usr/bin/env python3
"""
Analyze the monthly return data coverage to understand gaps and filing patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_return_coverage():
    print("="*80)
    print("ANALYZING MONTHLY RETURN DATA COVERAGE")
    print("="*80)
    
    # Load the ETL output
    df = pd.read_parquet('optionA_etl/data/pilot_fact_class_month.parquet')
    
    print(f"\nTotal records: {len(df):,}")
    print(f"Unique funds (class_ids): {df['class_id'].nunique()}")
    print(f"Date range: {df['month_end'].min()} to {df['month_end'].max()}")
    
    # Analyze coverage by fund
    print("\n" + "="*80)
    print("FUND-LEVEL COVERAGE ANALYSIS")
    print("="*80)
    
    fund_coverage = []
    for fund_id in df['class_id'].unique():
        fund_data = df[df['class_id'] == fund_id].sort_values('month_end')
        
        # Get date range for this fund
        min_date = fund_data['month_end'].min()
        max_date = fund_data['month_end'].max()
        
        # Calculate expected months
        expected_months = pd.date_range(start=min_date, end=max_date, freq='ME')
        actual_months = set(fund_data['month_end'])
        expected_set = set(expected_months)
        
        # Find gaps
        missing_months = expected_set - actual_months
        
        # Calculate coverage
        coverage_pct = len(actual_months) / len(expected_months) * 100 if len(expected_months) > 0 else 0
        
        fund_coverage.append({
            'class_id': fund_id,
            'first_date': min_date,
            'last_date': max_date,
            'expected_months': len(expected_months),
            'actual_months': len(actual_months),
            'missing_months': len(missing_months),
            'coverage_pct': coverage_pct,
            'has_returns': fund_data['return'].notna().sum(),
            'has_alpha': fund_data['realized alpha'].notna().sum() if 'realized alpha' in fund_data.columns else 0
        })
    
    coverage_df = pd.DataFrame(fund_coverage)
    
    print(f"\nFund Coverage Summary:")
    print(f"  Average coverage: {coverage_df['coverage_pct'].mean():.1f}%")
    print(f"  Median coverage: {coverage_df['coverage_pct'].median():.1f}%")
    print(f"  Min coverage: {coverage_df['coverage_pct'].min():.1f}%")
    print(f"  Max coverage: {coverage_df['coverage_pct'].max():.1f}%")
    
    print(f"\nFunds with gaps:")
    funds_with_gaps = coverage_df[coverage_df['missing_months'] > 0]
    print(f"  {len(funds_with_gaps)} out of {len(coverage_df)} funds have missing months")
    print(f"  Average missing months per fund: {funds_with_gaps['missing_months'].mean():.1f}")
    
    # Show examples of funds with poor coverage
    print("\nFunds with worst coverage (< 50%):")
    poor_coverage = coverage_df[coverage_df['coverage_pct'] < 50].sort_values('coverage_pct')
    if len(poor_coverage) > 0:
        print(poor_coverage[['class_id', 'expected_months', 'actual_months', 'coverage_pct']].head(10))
    else:
        print("  No funds with < 50% coverage")
    
    # Analyze filing patterns
    print("\n" + "="*80)
    print("N-PORT FILING PATTERN ANALYSIS")
    print("="*80)
    
    # Group by month to see filing patterns
    monthly_counts = df.groupby('month_end').size().sort_index()
    
    print(f"\nMonthly record counts:")
    print(f"  Mean: {monthly_counts.mean():.1f}")
    print(f"  Std: {monthly_counts.std():.1f}")
    print(f"  Min: {monthly_counts.min()}")
    print(f"  Max: {monthly_counts.max()}")
    
    # Identify months with unusually low counts
    low_threshold = monthly_counts.mean() - 2 * monthly_counts.std()
    low_months = monthly_counts[monthly_counts < low_threshold]
    if len(low_months) > 0:
        print(f"\nMonths with unusually low filing counts (< {low_threshold:.0f}):")
        print(low_months.head(10))
    
    # Analyze quarterly patterns (N-PORT-P is filed quarterly)
    print("\n" + "="*80)
    print("QUARTERLY FILING PATTERN")
    print("="*80)
    print("\nN-PORT-P filings are QUARTERLY, containing 3 months of data each.")
    print("This means each filing provides returns for the current and 2 prior months.")
    
    # Check for quarterly pattern
    df['quarter'] = df['month_end'].dt.to_period('Q')
    quarterly_coverage = df.groupby('quarter')['class_id'].nunique()
    
    print(f"\nQuarterly coverage (unique funds per quarter):")
    print(quarterly_coverage.tail(8))
    
    # Analyze duplicate records (multiple filings for same fund-month)
    print("\n" + "="*80)
    print("DUPLICATE RECORDS ANALYSIS")
    print("="*80)
    
    duplicates = df.groupby(['class_id', 'month_end']).size()
    duplicates = duplicates[duplicates > 1]
    
    print(f"\nDuplicate fund-month combinations: {len(duplicates)}")
    if len(duplicates) > 0:
        print(f"  Average duplicates: {duplicates.mean():.1f}")
        print(f"  Max duplicates: {duplicates.max()}")
        print("\nExamples of duplicates:")
        print(duplicates.head(10))
    
    # Analyze data completeness for a sample fund
    print("\n" + "="*80)
    print("SAMPLE FUND DETAILED ANALYSIS")
    print("="*80)
    
    sample_fund = df['class_id'].value_counts().index[0]  # Most common fund
    sample_data = df[df['class_id'] == sample_fund].sort_values('month_end')
    
    print(f"\nFund {sample_fund}:")
    print(f"  Total records: {len(sample_data)}")
    print(f"  Date range: {sample_data['month_end'].min()} to {sample_data['month_end'].max()}")
    print(f"  Records with returns: {sample_data['return'].notna().sum()}")
    print(f"  Records with alpha: {sample_data['realized alpha'].notna().sum() if 'realized alpha' in sample_data.columns else 0}")
    
    # Show the actual months
    print(f"\n  First 12 months of data:")
    print(sample_data[['month_end', 'return']].head(12).to_string(index=False))
    
    # Check for consecutive months
    months = sample_data['month_end'].sort_values()
    month_diffs = months.diff()
    non_consecutive = month_diffs[month_diffs > pd.Timedelta(days=32)]
    
    if len(non_consecutive) > 0:
        print(f"\n  Gaps in monthly data (> 1 month):")
        for idx in non_consecutive.index[:5]:
            prev_month = months.loc[months.index[months.index.get_loc(idx) - 1]]
            curr_month = months.loc[idx]
            gap = (curr_month - prev_month).days // 30
            print(f"    {prev_month.date()} -> {curr_month.date()} ({gap} months gap)")
    
    return df, coverage_df

def explain_nport_structure():
    print("\n" + "="*80)
    print("KEY INSIGHTS ABOUT N-PORT DATA STRUCTURE")
    print("="*80)
    
    print("""
1. **N-PORT-P Filing Frequency**: 
   - Filed QUARTERLY (every 3 months)
   - Each filing contains 3 months of return data
   - Filing date is typically 60 days after quarter end
   
2. **Return Data in Each Filing**:
   - rtn1: Most recent month (month of quarter end)
   - rtn2: Previous month (1 month before quarter end)  
   - rtn3: Two months ago (2 months before quarter end)
   
3. **Why We Have Gaps**:
   - If we start fetching from 2023-01-01, we only get Q1 2023 filing onwards
   - Each filing gives us 3 months, but we need continuous monthly data
   - To get 36 months of history, we need 12 quarterly filings (3 years)
   
4. **The Problem**:
   - Fetching filings from 36 months ago gives us more filings
   - But each filing still only contains 3 months of returns
   - We're getting the SAME return data, just from different filings
   
5. **Solution Options**:
   a) **Fetch More Historical Filings**: Go back further (e.g., 48-60 months)
   b) **Use Expanding Window**: Start alpha calculations with less history
   c) **Alternative Data Sources**: Use other return sources for historical data
   d) **Accept the Limitation**: Understand alpha won't be available for first 36 months
""")

if __name__ == "__main__":
    df, coverage = analyze_return_coverage()
    explain_nport_structure()