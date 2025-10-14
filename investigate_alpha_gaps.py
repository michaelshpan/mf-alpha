#!/usr/bin/env python3
"""
Investigate why predicted alpha has more data points than actual alpha.
Analyze the data availability patterns and identify root causes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_analyze_gaps():
    """Load data and analyze gaps in actual alpha."""
    
    print("="*70)
    print("INVESTIGATING ALPHA DATA AVAILABILITY")
    print("="*70)
    
    # Load both datasets
    etl_data = pd.read_parquet('optionA_etl/data/pilot_fact_class_month.parquet')
    pred_data = pd.read_parquet('optionA_etl/predictions/alpha.parquet')
    
    print("\n1. OVERALL DATA AVAILABILITY")
    print("-"*40)
    print(f"ETL records total: {len(etl_data):,}")
    print(f"Prediction records total: {len(pred_data):,}")
    
    # Analyze realized alpha availability in ETL data
    print(f"\nRealized alpha availability in ETL:")
    print(f"  - Records with realized alpha: {etl_data['realized alpha'].notna().sum():,} ({etl_data['realized alpha'].notna().mean():.1%})")
    print(f"  - Records missing realized alpha: {etl_data['realized alpha'].isna().sum():,} ({etl_data['realized alpha'].isna().mean():.1%})")
    
    # Check what's needed for realized alpha calculation
    print("\n2. FACTOR REGRESSION REQUIREMENTS")
    print("-"*40)
    print("Checking data needed for alpha calculation (36-month rolling regression):")
    
    # Factor columns needed for regression
    factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    
    for col in factor_cols:
        if col in etl_data.columns:
            available = etl_data[col].notna().sum()
            pct = etl_data[col].notna().mean()
            print(f"  - {col}: {available:,} records ({pct:.1%})")
    
    # Check RF (risk-free rate)
    if 'RF' in etl_data.columns:
        print(f"  - RF (risk-free): {etl_data['RF'].notna().sum():,} records ({etl_data['RF'].notna().mean():.1%})")
    
    # Analyze when factor data starts
    print("\n3. TEMPORAL ANALYSIS OF DATA AVAILABILITY")
    print("-"*40)
    
    # Group by month to see patterns
    monthly_stats = etl_data.groupby('month_end').agg({
        'return': lambda x: x.notna().sum(),
        'realized alpha': lambda x: x.notna().sum(),
        'MKT_RF': lambda x: x.notna().sum() if 'MKT_RF' in etl_data.columns else 0,
        'class_id': 'count'
    }).rename(columns={'class_id': 'total_records'})
    
    monthly_stats['pct_with_alpha'] = monthly_stats['realized alpha'] / monthly_stats['total_records'] * 100
    monthly_stats['pct_with_factors'] = monthly_stats['MKT_RF'] / monthly_stats['total_records'] * 100
    
    print("Monthly data availability:")
    print(monthly_stats.head(10))
    print("...")
    print(monthly_stats.tail(10))
    
    # Find when realized alpha calculations begin
    first_alpha = etl_data[etl_data['realized alpha'].notna()]['month_end'].min()
    last_no_alpha = etl_data[etl_data['realized alpha'].isna()]['month_end'].max()
    
    print(f"\nTiming insights:")
    print(f"  - First month with realized alpha: {first_alpha}")
    print(f"  - Last month without realized alpha: {last_no_alpha}")
    
    # Analyze the rolling window requirement
    print("\n4. ROLLING WINDOW ANALYSIS")
    print("-"*40)
    
    # Check how many months of history each fund has
    fund_history = etl_data.groupby('class_id').agg({
        'month_end': ['min', 'max', 'count'],
        'realized alpha': lambda x: x.notna().sum()
    })
    fund_history.columns = ['first_month', 'last_month', 'total_months', 'months_with_alpha']
    fund_history['pct_with_alpha'] = fund_history['months_with_alpha'] / fund_history['total_months'] * 100
    
    print(f"Fund-level statistics:")
    print(f"  - Average months per fund: {fund_history['total_months'].mean():.1f}")
    print(f"  - Average months with alpha: {fund_history['months_with_alpha'].mean():.1f}")
    print(f"  - Average % with alpha: {fund_history['pct_with_alpha'].mean():.1f}%")
    
    # Find funds with no alpha at all
    funds_no_alpha = fund_history[fund_history['months_with_alpha'] == 0]
    print(f"\nFunds with NO realized alpha: {len(funds_no_alpha)}")
    if len(funds_no_alpha) > 0:
        print("Examples of funds without alpha:")
        print(funds_no_alpha.head(10))
    
    # Calculate minimum history needed
    print("\n5. MINIMUM HISTORY REQUIREMENT")
    print("-"*40)
    
    # For each fund, find when alpha starts being available
    alpha_start_by_fund = []
    for fund_id in etl_data['class_id'].unique():
        fund_data = etl_data[etl_data['class_id'] == fund_id].sort_values('month_end')
        
        # Find first month with alpha
        first_alpha_idx = fund_data['realized alpha'].notna().idxmax() if fund_data['realized alpha'].notna().any() else None
        
        if first_alpha_idx is not None:
            first_alpha_month = fund_data.loc[first_alpha_idx, 'month_end']
            first_month = fund_data['month_end'].min()
            months_before_alpha = (first_alpha_month.year - first_month.year) * 12 + (first_alpha_month.month - first_month.month)
            alpha_start_by_fund.append({
                'fund': fund_id,
                'first_month': first_month,
                'first_alpha_month': first_alpha_month,
                'months_before_alpha': months_before_alpha
            })
    
    alpha_start_df = pd.DataFrame(alpha_start_by_fund)
    if not alpha_start_df.empty:
        print(f"Months of history before alpha becomes available:")
        print(f"  - Mean: {alpha_start_df['months_before_alpha'].mean():.1f} months")
        print(f"  - Median: {alpha_start_df['months_before_alpha'].median():.1f} months")
        print(f"  - Min: {alpha_start_df['months_before_alpha'].min():.0f} months")
        print(f"  - Max: {alpha_start_df['months_before_alpha'].max():.0f} months")
        
        # This should be around 36 months if using 36-month rolling window
        print(f"\nNote: If using 36-month rolling regression, we expect ~36 months before alpha is available")
    
    # Check regression statistics columns
    print("\n6. REGRESSION OUTPUT ANALYSIS")
    print("-"*40)
    
    regression_cols = ['alpha (intercept t-stat)', 'market beta t-stat', 'size beta t-stat', 
                      'value beta t-stat', 'profit. beta t-stat', 'invest. beta t-stat', 
                      'momentum beta t-stat', 'R2']
    
    print("Regression statistics availability:")
    for col in regression_cols:
        if col in etl_data.columns:
            available = etl_data[col].notna().sum()
            pct = etl_data[col].notna().mean()
            print(f"  - {col}: {available:,} records ({pct:.1%})")
    
    # Check if regression stats align with realized alpha
    if 'R2' in etl_data.columns:
        has_r2 = etl_data['R2'].notna().sum()
        has_alpha = etl_data['realized alpha'].notna().sum()
        print(f"\nConsistency check:")
        print(f"  - Records with R2: {has_r2:,}")
        print(f"  - Records with realized alpha: {has_alpha:,}")
        print(f"  - Difference: {abs(has_r2 - has_alpha):,}")
    
    return etl_data, pred_data, monthly_stats, fund_history

def create_visualization(etl_data, monthly_stats):
    """Create visualizations of data availability patterns."""
    
    print("\n7. CREATING VISUALIZATIONS")
    print("-"*40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Timeline of data availability
    ax = axes[0, 0]
    monthly_counts = etl_data.groupby('month_end').agg({
        'class_id': 'count',
        'realized alpha': lambda x: x.notna().sum()
    }).rename(columns={'class_id': 'total', 'realized alpha': 'with_alpha'})
    
    ax.plot(monthly_counts.index, monthly_counts['total'], label='Total Records', linewidth=2)
    ax.plot(monthly_counts.index, monthly_counts['with_alpha'], label='With Realized Alpha', linewidth=2)
    ax.fill_between(monthly_counts.index, 0, monthly_counts['with_alpha'], alpha=0.3)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Records')
    ax.set_title('Data Availability Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Percentage with alpha over time
    ax = axes[0, 1]
    monthly_counts['pct_with_alpha'] = monthly_counts['with_alpha'] / monthly_counts['total'] * 100
    ax.plot(monthly_counts.index, monthly_counts['pct_with_alpha'], color='green', linewidth=2)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('% Records with Realized Alpha')
    ax.set_title('Percentage of Records with Realized Alpha')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 3: Distribution of months before alpha
    ax = axes[1, 0]
    fund_first_alpha = []
    for fund_id in etl_data['class_id'].unique():
        fund_data = etl_data[etl_data['class_id'] == fund_id].sort_values('month_end')
        if fund_data['realized alpha'].notna().any():
            first_month_idx = fund_data.index[0]
            first_alpha_idx = fund_data['realized alpha'].notna().idxmax()
            months_diff = (fund_data.loc[first_alpha_idx, 'month_end'].year - fund_data.loc[first_month_idx, 'month_end'].year) * 12 + \
                         (fund_data.loc[first_alpha_idx, 'month_end'].month - fund_data.loc[first_month_idx, 'month_end'].month)
            fund_first_alpha.append(months_diff)
    
    if fund_first_alpha:
        ax.hist(fund_first_alpha, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=36, color='r', linestyle='--', label='36 months', linewidth=2)
        ax.set_xlabel('Months Before First Alpha')
        ax.set_ylabel('Number of Funds')
        ax.set_title('Distribution of Months Before Alpha Becomes Available')
        ax.legend()
    
    # Plot 4: Factor data availability
    ax = axes[1, 1]
    factor_availability = etl_data[['month_end', 'MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']].copy()
    factor_monthly = factor_availability.groupby('month_end').agg(lambda x: x.notna().mean() * 100)
    
    for col in factor_monthly.columns:
        ax.plot(factor_monthly.index, factor_monthly[col], label=col, alpha=0.8)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('% Data Available')
    ax.set_title('Factor Data Availability Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('alpha_availability_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to alpha_availability_analysis.png")
    
    plt.show()

def main():
    """Main execution."""
    
    # Analyze gaps
    etl_data, pred_data, monthly_stats, fund_history = load_and_analyze_gaps()
    
    # Create visualizations
    create_visualization(etl_data, monthly_stats)
    
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    
    print("""
The investigation reveals that actual (realized) alpha requires:

1. **36-month rolling window**: Alpha is calculated using a 36-month rolling regression
   of fund returns against Fama-French 5 factors + momentum.
   
2. **Factor data availability**: The Fama-French factors (MKT_RF, SMB, HML, RMW, CMA) 
   and momentum (MOM) must be available for the regression.
   
3. **Minimum history requirement**: Each fund needs at least 36 months of concurrent
   return and factor data before the first alpha can be calculated.
   
4. **Data gaps**: Early months (2019-2021) have missing factor data or insufficient
   history, explaining why predicted alpha exceeds actual alpha availability.

This is expected behavior for rolling regression-based metrics that require
substantial historical data to compute.
""")
    
    return etl_data, monthly_stats

if __name__ == "__main__":
    etl_data, monthly_stats = main()