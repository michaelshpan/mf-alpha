import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List

def load_data(file_path: str) -> pd.DataFrame:
    """Load the scaled annual data."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Only CSV files supported currently")
    
    print(f"Loaded {len(df):,} observations from {file_path}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Unique funds: {df['fundno'].nunique():,}")
    
    return df

def get_table2_variables() -> Dict[str, str]:
    """Define Table 2 variables with their descriptions (based on DeMiguel et al.)"""
    return {
        'alpha6_1mo_12m': 'Alpha (12-month rolling)',
        'alpha6_1mo_12m_lag_1': 'Alpha (lagged)',
        'alpha6_tstat_lag_1': 'Alpha t-statistic (lagged)',
        'mtna_lag_1': 'Total net assets (lagged)',
        'exp_ratio_lag_1': 'Expense ratio (lagged)', 
        'age_lag_1': 'Fund age (lagged)',
        'flow_12m_lag_1': '12-month flows (lagged)',
        'mgr_tenure_lag_1': 'Manager tenure (lagged)',
        'turn_ratio_lag_1': 'Turnover ratio (lagged)',
        'flow_vol_12m_lag_1': 'Flow volatility (lagged)',
        'value_added_12m_lag_1': 'Value added (lagged)',
        'beta_market_tstat_lag_1': 'Market beta t-stat (lagged)',
        'beta_profit_tstat_lag_1': 'Profitability beta t-stat (lagged)',
        'beta_invest_tstat_lag_1': 'Investment beta t-stat (lagged)',
        'beta_size_tstat_lag_1': 'Size beta t-stat (lagged)',
        'beta_value_tstat_lag_1': 'Value beta t-stat (lagged)',
        'beta_mom_tstat_lag_1': 'Momentum beta t-stat (lagged)',
        'R2_lag_1': 'R-squared (lagged)'
    }

def compute_statistics(df: pd.DataFrame, variables: Dict[str, str]) -> pd.DataFrame:
    """Compute Table 2 statistics: mean, median, std dev, observations."""
    stats_list = []
    
    for var, description in variables.items():
        if var not in df.columns:
            print(f"Warning: Variable {var} not found in data")
            continue
            
        # Remove missing values for statistics
        data = df[var].dropna()
        
        stats = {
            'Variable': var,
            'Description': description,
            'Mean': data.mean(),
            'Median': data.median(),
            'Std Dev': data.std(),
            'Observations': len(data),
            'Missing': df[var].isnull().sum(),
            'Min': data.min(),
            'Max': data.max()
        }
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)

def format_table2(stats_df: pd.DataFrame, scaled_data: bool = True) -> pd.DataFrame:
    """Format the statistics table similar to Table 2 in the paper."""
    # Create formatted version
    table2 = stats_df.copy()
    
    # Round to appropriate decimal places
    table2['Mean'] = table2['Mean'].round(3)
    table2['Median'] = table2['Median'].round(3)
    table2['Std Dev'] = table2['Std Dev'].round(3)
    table2['Min'] = table2['Min'].round(3)
    table2['Max'] = table2['Max'].round(3)
    
    # Format observations with thousands separator
    table2['Observations'] = table2['Observations'].apply(lambda x: f"{x:,}")
    
    # Add note about scaling
    if scaled_data:
        print("\n" + "="*80)
        print("NOTE: This data appears to be standardized (scaled).")
        print("- Mean values close to 0 and Std Dev close to 1 indicate standardization")
        print("- These statistics still provide meaningful distributional information")
        print("- The 'Observations' count shows the sample size for each variable")
        print("="*80)
    
    return table2

def save_results(table2: pd.DataFrame, output_path: str):
    """Save results to CSV and print formatted table."""
    # Save to CSV
    table2.to_csv(output_path, index=False)
    print(f"\nTable 2 statistics saved to: {output_path}")
    
    # Print formatted table
    print("\n" + "="*120)
    print("TABLE 2: SUMMARY STATISTICS OF FUND CHARACTERISTICS")
    print("="*120)
    
    # Print with nice formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 30)
    
    # Select main columns for display (similar to paper format)
    display_cols = ['Description', 'Mean', 'Median', 'Std Dev', 'Observations']
    print(table2[display_cols].to_string(index=False))
    print("="*120)
    
    # Print summary
    total_obs = table2['Observations'].str.replace(',', '').astype(int)
    print(f"\nSUMMARY:")
    print(f"Variables analyzed: {len(table2)}")
    print(f"Average observations per variable: {total_obs.mean():,.0f}")
    print(f"Range of observations: {total_obs.min():,} to {total_obs.max():,}")

def analyze_data_scaling(df: pd.DataFrame, variables: List[str]) -> Dict[str, bool]:
    """Analyze if data appears to be scaled/standardized."""
    scaling_evidence = {}
    
    for var in variables:
        if var not in df.columns:
            continue
            
        data = df[var].dropna()
        mean_near_zero = abs(data.mean()) < 0.1
        std_near_one = abs(data.std() - 1.0) < 0.1
        
        scaling_evidence[var] = {
            'mean_near_zero': mean_near_zero,
            'std_near_one': std_near_one,
            'likely_scaled': mean_near_zero and std_near_one,
            'mean': data.mean(),
            'std': data.std()
        }
    
    return scaling_evidence

def main():
    parser = argparse.ArgumentParser(description="Generate Table 2 statistics from mutual fund data")
    parser.add_argument("--input", 
                       default="data/raw/scaled_annual_data_JFE_clean.csv",
                       help="Input CSV file path")
    parser.add_argument("--output", 
                       default="outputs/table2_statistics.csv",
                       help="Output CSV file path")
    parser.add_argument("--analyze_scaling", 
                       action="store_true",
                       help="Analyze if data appears to be scaled")
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input)
    
    # Get variables for Table 2
    variables = get_table2_variables()
    
    # Analyze scaling if requested
    if args.analyze_scaling:
        print("\n" + "="*80)
        print("SCALING ANALYSIS")
        print("="*80)
        scaling_evidence = analyze_data_scaling(df, list(variables.keys()))
        
        scaled_vars = sum(1 for v in scaling_evidence.values() if v['likely_scaled'])
        total_vars = len(scaling_evidence)
        
        print(f"Variables likely scaled: {scaled_vars}/{total_vars}")
        
        # Show a few examples
        print("\nExamples:")
        for var, evidence in list(scaling_evidence.items())[:5]:
            status = "SCALED" if evidence['likely_scaled'] else "NOT SCALED"
            print(f"{var:25s}: Mean={evidence['mean']:6.3f}, Std={evidence['std']:6.3f} [{status}]")
    
    # Compute statistics
    stats_df = compute_statistics(df, variables)
    
    # Check if data appears scaled
    scaled_data = abs(stats_df['Mean'].mean()) < 0.1 and abs(stats_df['Std Dev'].mean() - 1.0) < 0.2
    
    # Format table
    table2 = format_table2(stats_df, scaled_data)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_results(table2, args.output)
    
    # Also save detailed stats
    detailed_path = output_path.parent / f"{output_path.stem}_detailed.csv"
    stats_df.to_csv(detailed_path, index=False)
    print(f"Detailed statistics saved to: {detailed_path}")

if __name__ == "__main__":
    main()