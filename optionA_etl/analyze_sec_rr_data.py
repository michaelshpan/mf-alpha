#!/usr/bin/env python3
"""
Analyze SEC Risk/Return dataset to understand structure and extract expense/turnover data.
This shows how to use SEC's structured quarterly datasets as a more reliable data source.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_sec_rr_data(data_dir="sec_rr_datasets/2025q2_mfrr"):
    """Analyze SEC RR data structure and extract key metrics"""
    
    data_path = Path(data_dir)
    
    # Load data files
    print("Loading SEC RR dataset files...")
    sub_df = pd.read_csv(data_path / "sub.tsv", sep="\t", dtype=str)
    num_df = pd.read_csv(data_path / "num.tsv", sep="\t", dtype=str)
    
    print(f"\nDataset Overview:")
    print(f"- Submissions: {len(sub_df)} filings")
    print(f"- Numeric data points: {len(num_df)}")
    print(f"- Unique CIKs: {sub_df['cik'].nunique()}")
    
    # Analyze expense ratio tags
    expense_tags = [
        'NetExpensesOverAssets',
        'ExpensesOverAssets', 
        'NetExpenseRatio',
        'ExpenseRatio',
        'TotalAnnualFundOperatingExpensesRatio'
    ]
    
    print("\n=== Expense Ratio Analysis ===")
    for tag in expense_tags:
        count = len(num_df[num_df['tag'] == tag])
        if count > 0:
            print(f"{tag}: {count} records")
            sample = num_df[num_df['tag'] == tag].head(3)
            if 'value' in sample.columns:
                print(f"  Sample values: {sample['value'].tolist()}")
    
    # Analyze turnover tags
    turnover_tags = [
        'PortfolioTurnoverRate',
        'PortfolioTurnoverPercent',
        'AnnualPortfolioTurnoverRate'
    ]
    
    print("\n=== Portfolio Turnover Analysis ===")
    for tag in turnover_tags:
        count = len(num_df[num_df['tag'] == tag])
        if count > 0:
            print(f"{tag}: {count} records")
            sample = num_df[num_df['tag'] == tag].head(3)
            if 'value' in sample.columns:
                print(f"  Sample values: {sample['value'].tolist()}")
    
    # Check class/series identification
    print("\n=== Class/Series Identification ===")
    print(f"Records with series ID: {num_df['series'].notna().sum()}")
    print(f"Records with class ID: {num_df['class'].notna().sum()}")
    print(f"Unique series: {num_df['series'].nunique()}")
    print(f"Unique classes: {num_df['class'].nunique()}")
    
    # Sample data with class identifiers
    class_data = num_df[num_df['class'].notna()]
    if len(class_data) > 0:
        print("\nSample class identifiers:")
        print(class_data[['adsh', 'series', 'class', 'tag']].head(5).to_string())
    
    # Map CIKs to fund names
    print("\n=== Sample Fund Names ===")
    sample_ciks = sub_df.head(10)
    for _, row in sample_ciks.iterrows():
        print(f"CIK {row['cik']}: {row['name']}")
    
    # Extract a complete example
    print("\n=== Complete Example ===")
    # Find a submission with both expense and turnover data
    for adsh in sub_df['adsh'].head(20):
        expense = num_df[(num_df['adsh'] == adsh) & (num_df['tag'].str.contains('Expense', case=False, na=False))]
        turnover = num_df[(num_df['adsh'] == adsh) & (num_df['tag'].str.contains('Turnover', case=False, na=False))]
        
        if len(expense) > 0 and len(turnover) > 0:
            sub_info = sub_df[sub_df['adsh'] == adsh].iloc[0]
            print(f"\nFund: {sub_info['name']} (CIK: {sub_info['cik']})")
            print(f"Filing: {sub_info['form']} on {sub_info['filed']}")
            print(f"Expense data points: {len(expense)}")
            print(f"Turnover data points: {len(turnover)}")
            
            # Show specific values
            net_exp = num_df[(num_df['adsh'] == adsh) & (num_df['tag'] == 'NetExpensesOverAssets')]
            if len(net_exp) > 0:
                print(f"Net Expense Ratios found: {len(net_exp)}")
                for _, row in net_exp.head(3).iterrows():
                    print(f"  Class {row['class']}: {row.get('value', 'N/A')}")
            
            turn_rate = num_df[(num_df['adsh'] == adsh) & (num_df['tag'] == 'PortfolioTurnoverRate')]
            if len(turn_rate) > 0:
                print(f"Turnover Rates found: {len(turn_rate)}")
                for _, row in turn_rate.head(3).iterrows():
                    print(f"  Class {row['class']}: {row.get('value', 'N/A')}")
            break
    
    return sub_df, num_df

def extract_fund_metrics(cik, sub_df, num_df):
    """Extract expense and turnover for a specific CIK"""
    
    # Find submissions for this CIK
    cik_subs = sub_df[sub_df['cik'] == str(cik)]
    if len(cik_subs) == 0:
        return None
    
    # Get most recent submission
    latest = cik_subs.iloc[0]
    adsh = latest['adsh']
    
    # Extract expense ratios
    expenses = num_df[
        (num_df['adsh'] == adsh) & 
        (num_df['tag'].isin(['NetExpensesOverAssets', 'ExpensesOverAssets']))
    ]
    
    # Extract turnover
    turnover = num_df[
        (num_df['adsh'] == adsh) & 
        (num_df['tag'] == 'PortfolioTurnoverRate')
    ]
    
    result = {
        'cik': cik,
        'name': latest['name'],
        'filing_date': latest['filed'],
        'form': latest['form'],
        'expense_ratios': {},
        'turnover_rates': {}
    }
    
    # Group by class
    for _, row in expenses.iterrows():
        class_id = row['class'] if pd.notna(row['class']) else 'fund_level'
        if 'value' in row:
            result['expense_ratios'][class_id] = float(row['value'])
    
    for _, row in turnover.iterrows():
        class_id = row['class'] if pd.notna(row['class']) else 'fund_level'
        if 'value' in row:
            result['turnover_rates'][class_id] = float(row['value'])
    
    return result

if __name__ == "__main__":
    # Analyze the data structure
    sub_df, num_df = analyze_sec_rr_data()
    
    # Show how the structured data could help with class ID mapping
    print("\n" + "="*60)
    print("KEY INSIGHTS FOR PILOT PIPELINE:")
    print("="*60)
    print("""
1. SEC RR datasets provide STRUCTURED data with proper class IDs
2. Class IDs in these files match SEC's official identifiers (like C000001973)
3. Both expense ratios and turnover are available per class
4. Data is pre-parsed - no HTML/XBRL parsing needed
5. Updated quarterly with all recent filings

SOLUTION for Issue #3 (Class ID Mismatch):
- Use these SEC RR datasets as the primary source for expense/turnover
- Join on CIK + class_id to match with N-PORT data
- This eliminates HTML parsing errors and class ID extraction issues
    """)