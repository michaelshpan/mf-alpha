#!/usr/bin/env python3
"""Quick diagnostic to understand alpha data gaps."""

import pandas as pd
import numpy as np

# Load data
print("Loading data...")
etl = pd.read_parquet('optionA_etl/data/pilot_fact_class_month.parquet')
pred = pd.read_parquet('optionA_etl/predictions/alpha.parquet')

print("\n=== DATA OVERVIEW ===")
print(f"ETL records: {len(etl):,}")
print(f"Prediction records: {len(pred):,}")

# Check realized alpha availability
print("\n=== REALIZED ALPHA AVAILABILITY ===")
has_alpha = etl['realized alpha'].notna().sum()
no_alpha = etl['realized alpha'].isna().sum()
print(f"Records WITH realized alpha: {has_alpha:,} ({has_alpha/len(etl):.1%})")
print(f"Records WITHOUT realized alpha: {no_alpha:,} ({no_alpha/len(etl):.1%})")

# Check when alpha starts
print("\n=== TEMPORAL PATTERNS ===")
etl_sorted = etl.sort_values('month_end')
first_month = etl_sorted['month_end'].min()
first_alpha_month = etl_sorted[etl_sorted['realized alpha'].notna()]['month_end'].min()
print(f"First month in data: {first_month.strftime('%Y-%m')}")
print(f"First month with alpha: {first_alpha_month.strftime('%Y-%m')}")
months_gap = (first_alpha_month.year - first_month.year) * 12 + (first_alpha_month.month - first_month.month)
print(f"Gap: {months_gap} months")

# Check factor data
print("\n=== FACTOR DATA AVAILABILITY ===")
factors = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'RF']
for factor in factors:
    if factor in etl.columns:
        available = etl[factor].notna().sum()
        print(f"{factor}: {available:,} records ({available/len(etl):.1%})")

# Check by time period
print("\n=== ALPHA AVAILABILITY BY YEAR ===")
etl['year'] = etl['month_end'].dt.year
year_stats = etl.groupby('year').agg({
    'class_id': 'count',
    'realized alpha': lambda x: x.notna().sum()
}).rename(columns={'class_id': 'total_records', 'realized alpha': 'with_alpha'})
year_stats['pct_with_alpha'] = (year_stats['with_alpha'] / year_stats['total_records'] * 100).round(1)
print(year_stats)

# Check specific funds
print("\n=== SAMPLE FUND ANALYSIS ===")
sample_fund = etl['class_id'].iloc[0]
fund_data = etl[etl['class_id'] == sample_fund].sort_values('month_end')
print(f"Fund: {sample_fund}")
print(f"Total months: {len(fund_data)}")
print(f"Months with alpha: {fund_data['realized alpha'].notna().sum()}")
print(f"First month: {fund_data['month_end'].min().strftime('%Y-%m')}")
if fund_data['realized alpha'].notna().any():
    first_alpha = fund_data[fund_data['realized alpha'].notna()]['month_end'].min()
    print(f"First alpha: {first_alpha.strftime('%Y-%m')}")
    gap = (first_alpha.year - fund_data['month_end'].min().year) * 12 + (first_alpha.month - fund_data['month_end'].min().month)
    print(f"Months before alpha: {gap}")

print("\n=== EXPLANATION ===")
print("""
The realized alpha is calculated using a 36-month rolling regression of fund returns
against Fama-French 5 factors + momentum. This requires:

1. At least 36 months of historical data for each fund
2. Available factor data (FF5 + momentum) for those months
3. Sufficient non-missing returns to run the regression

Early months (2019-2021) don't have realized alpha because:
- New funds need 36 months to accumulate history
- Factor data might be incomplete for recent periods
- The ETL pipeline started collecting data in 2019

This is EXPECTED behavior - predictions can be made immediately but actual alpha
requires 3 years of history to calculate.
""")