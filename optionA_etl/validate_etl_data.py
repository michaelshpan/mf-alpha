#!/usr/bin/env python3
"""
ETL Data Validation Script
Extracts sample fund data from ETL pipeline output and creates human-reviewable reports
with data sources for cross-checking against alternative sources.
"""

import pandas as pd
import numpy as np
import openpyxl
from pathlib import Path
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data source mapping for each field
DATA_SOURCES = {
    # Fund identifiers
    'cik': {'source': 'SEC EDGAR', 'url': 'https://www.sec.gov/edgar/searchedgar/companysearch', 'file': 'N-PORT-P'},
    'series_id': {'source': 'SEC EDGAR', 'url': 'https://www.sec.gov/edgar/searchedgar/companysearch', 'file': 'Series & Classes'},
    'class_id': {'source': 'SEC EDGAR', 'url': 'https://www.sec.gov/edgar/searchedgar/companysearch', 'file': 'Series & Classes'},
    
    # Time identifiers
    'month_end': {'source': 'SEC N-PORT-P', 'url': 'SEC EDGAR filings', 'file': 'N-PORT-P XML'},
    
    # Fund characteristics from SEC filings
    'r12_2': {'source': 'SEC N-PORT-P', 'url': 'SEC EDGAR', 'file': 'N-PORT-P Part B Item 3', 'description': '12-month return lagged 2 months'},
    'r36_13': {'source': 'SEC N-PORT-P', 'url': 'SEC EDGAR', 'file': 'N-PORT-P Part B Item 3', 'description': '36-month return lagged 13 months'},
    'flow': {'source': 'SEC N-PORT-P', 'url': 'SEC EDGAR', 'file': 'N-PORT-P Part B Item 4', 'description': 'Net flows'},
    'TNA': {'source': 'SEC N-PORT-P', 'url': 'SEC EDGAR', 'file': 'N-PORT-P Part B Item 1', 'description': 'Total Net Assets'},
    'TO': {'source': 'SEC OEF/RR', 'url': 'SEC EDGAR', 'file': 'Form N-1A Risk/Return', 'description': 'Turnover ratio'},
    'exp_ratio': {'source': 'SEC OEF/RR', 'url': 'SEC EDGAR', 'file': 'Form N-1A Risk/Return', 'description': 'Expense ratio'},
    'expense_ratio': {'source': 'SEC RR Datasets', 'url': 'https://www.sec.gov/data-research/sec-markets-data/mutual-fund-prospectus-riskreturn-summary-data-sets', 'file': 'Quarterly RR TSV files', 'description': 'Net expense ratio from structured SEC data'},
    'turnover_rate': {'source': 'SEC RR Datasets + Series/Class Mapping', 'url': 'https://www.sec.gov/data-research/sec-markets-data/mutual-fund-prospectus-riskreturn-summary-data-sets', 'file': 'Quarterly RR TSV + Investment Company Series/Class CSV', 'description': 'Portfolio turnover rate mapped from series to class level'},
    'Family_TNA': {'source': 'SEC N-PORT-P (aggregated)', 'url': 'SEC EDGAR', 'file': 'Computed from all funds in family', 'description': 'Family total net assets'},
    'fund_age': {'source': 'Computed', 'url': 'N/A', 'file': 'Derived from fund inception date', 'description': 'Fund age in months'},
    'ST_vol': {'source': 'Computed', 'url': 'N/A', 'file': 'Calculated from returns', 'description': 'Short-term volatility'},
    
    # Factor loadings from regression
    'Mkt_RF': {'source': 'Ken French Data Library', 'url': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html', 'file': 'F-F_Research_Data_5_Factors_2x3.CSV', 'description': 'Market beta from FF5 model'},
    'SMB': {'source': 'Ken French Data Library', 'url': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html', 'file': 'F-F_Research_Data_5_Factors_2x3.CSV', 'description': 'Size factor loading'},
    'HML': {'source': 'Ken French Data Library', 'url': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html', 'file': 'F-F_Research_Data_5_Factors_2x3.CSV', 'description': 'Value factor loading'},
    'RMW': {'source': 'Ken French Data Library', 'url': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html', 'file': 'F-F_Research_Data_5_Factors_2x3.CSV', 'description': 'Profitability factor loading'},
    'CMA': {'source': 'Ken French Data Library', 'url': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html', 'file': 'F-F_Research_Data_5_Factors_2x3.CSV', 'description': 'Investment factor loading'},
    'MOM': {'source': 'Ken French Data Library', 'url': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html', 'file': 'F-F_Momentum_Factor.CSV', 'description': 'Momentum factor loading'},
    'alpha': {'source': 'Computed', 'url': 'N/A', 'file': 'Regression intercept from FF5+MOM model', 'description': 'Alpha from factor model'},
    'r2': {'source': 'Computed', 'url': 'N/A', 'file': 'Regression R-squared from FF5+MOM model', 'description': 'Model R-squared'},
    'vol': {'source': 'Computed', 'url': 'N/A', 'file': 'Regression residual volatility', 'description': 'Idiosyncratic volatility'},
    
    # Additional metrics
    'value_added': {'source': 'Computed', 'url': 'N/A', 'file': 'Calculated as fund return - benchmark', 'description': 'Value added over benchmark'},
    'net_flow': {'source': 'SEC N-PORT-P', 'url': 'SEC EDGAR', 'file': 'Computed from TNA changes and returns', 'description': 'Net fund flows'}
}

def load_etl_data(data_path: str = "data/pilot_fact_class_month.parquet"):
    """Load ETL output data"""
    try:
        df = pd.read_parquet(data_path)
        print(f"✅ Loaded {len(df):,} rows from {data_path}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
        print("Please run the ETL pipeline first: python -m etl.pilot_pipeline --pilot config/funds_pilot.yaml")
        return None

def get_comprehensive_fund_coverage(df: pd.DataFrame):
    """
    Get comprehensive coverage of all CIKs, series, and classes
    Returns:
    - all_funds_data: DataFrame with ALL unique combinations of CIK, series, and class for Fund Summary
    - sample_funds: One class per series for detailed analysis (Data Values sheet)
    - timeseries_funds: One class per series for time series worksheets
    """
    if 'class_id' not in df.columns:
        print("❌ No class_id column found in data")
        return None, None, None
    
    # Get latest month data for fund selection
    if 'month_end' in df.columns:
        latest_month = df['month_end'].max()
        df_latest = df[df['month_end'] == latest_month].copy()
    else:
        df_latest = df.copy()
    
    # Create comprehensive fund summary with ALL CIKs, series, and classes
    all_funds_data = []
    
    # Get unique combinations of CIK, series_id, and class_id
    if 'cik' in df_latest.columns and 'series_id' in df_latest.columns:
        # Group by CIK, series, and class to get all unique combinations
        grouped = df_latest.groupby(['cik', 'series_id', 'class_id'], dropna=False).size().reset_index(name='count')
        
        for _, row in grouped.iterrows():
            all_funds_data.append({
                'cik': row['cik'],
                'series_id': row['series_id'],
                'class_id': row['class_id']
            })
    else:
        # Fall back to just class_id if other columns not available
        for class_id in df_latest['class_id'].unique():
            all_funds_data.append({
                'cik': df_latest[df_latest['class_id'] == class_id]['cik'].iloc[0] if 'cik' in df_latest.columns else 'N/A',
                'series_id': df_latest[df_latest['class_id'] == class_id]['series_id'].iloc[0] if 'series_id' in df_latest.columns else 'N/A',
                'class_id': class_id
            })
    
    # Sample funds: RANDOMLY pick one class per series for Data Values sheet (all 24 fields)
    sample_funds = []
    series_to_class_map = {}
    
    if 'series_id' in df_latest.columns:
        # Group by series and pick one class per series randomly
        np.random.seed(42)  # For reproducibility
        for series_id in df_latest['series_id'].dropna().unique():
            series_classes = df_latest[df_latest['series_id'] == series_id]['class_id'].unique()
            if len(series_classes) > 0:
                selected_class = np.random.choice(series_classes)
                sample_funds.append(selected_class)
                series_to_class_map[series_id] = selected_class
        
        # Add any classes without series_id
        no_series = df_latest[df_latest['series_id'].isna()]['class_id'].unique()
        sample_funds.extend(no_series.tolist())
    else:
        # If no series data, use all unique class IDs
        sample_funds = df_latest['class_id'].unique().tolist()
    
    # For time series: RANDOMLY pick one class per series (different random selection)
    timeseries_funds = []
    if 'series_id' in df_latest.columns:
        np.random.seed(123)  # Different seed for potentially different selection
        for series_id in df_latest['series_id'].dropna().unique():
            series_classes = df_latest[df_latest['series_id'] == series_id]['class_id'].unique()
            if len(series_classes) > 0:
                selected_class = np.random.choice(series_classes)
                timeseries_funds.append(selected_class)
        
        # Add any classes without series_id (limit total to reasonable number for Excel)
        no_series = df_latest[df_latest['series_id'].isna()]['class_id'].unique()
        timeseries_funds.extend(no_series.tolist()[:5])  # Limit orphan classes
    else:
        timeseries_funds = sample_funds[:15]  # Limit for Excel readability
    
    # Limit timeseries to manageable number of worksheets
    timeseries_funds = timeseries_funds[:15]  # Max 15 timeseries worksheets
    
    return all_funds_data, sample_funds, timeseries_funds

def create_validation_report(df: pd.DataFrame, all_funds_data: list, sample_funds: list, timeseries_funds: list, output_dir: str = "validation"):
    """Create detailed validation report with comprehensive coverage of all funds, series, and classes"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare data for Excel export
    all_fund_data = []
    sample_fund_data = []
    source_documentation = []
    series_class_summary = []
    
    # 1. Fund Summary Sheet - ALL FUNDS with CIK, Series, and Class information
    print(f"   Processing {len(all_funds_data)} fund-class combinations for comprehensive Fund Summary...")
    for fund_info in all_funds_data:
        fund_id = fund_info['class_id']
        cik = fund_info['cik']
        series_id = fund_info['series_id']
        
        fund_data = df[df['class_id'] == fund_id].copy()
        
        if len(fund_data) == 0:
            # Still include the fund in summary even if no data
            fund_summary = {
                'CIK': cik,
                'Series ID': series_id,
                'Class ID': fund_id,
                'Latest Date': 'N/A',
                'Data Points': 0,
                'Has Expense Ratio': 'No',
                'Has Turnover Rate': 'No',
            }
            all_fund_data.append(fund_summary)
            continue
        
        # Sort by date if available
        if 'month_end' in fund_data.columns:
            fund_data = fund_data.sort_values('month_end')
        
        # Get latest observation for summary
        latest_row = fund_data.iloc[-1]
        
        # Create comprehensive fund summary with explicit CIK, Series, Class hierarchy
        fund_summary = {
            'CIK': cik if pd.notna(cik) else latest_row.get('cik', 'N/A'),
            'Series ID': series_id if pd.notna(series_id) else latest_row.get('series_id', 'N/A'),
            'Class ID': fund_id,
            'Latest Date': latest_row.get('month_end', 'N/A'),
            'Data Points': len(fund_data),
            'Has Expense Ratio': 'Yes' if pd.notna(latest_row.get('expense_ratio')) else 'No',
            'Has Turnover Rate': 'Yes' if pd.notna(latest_row.get('turnover_rate')) else 'No',
        }
        
        # Add key metrics for summary view
        key_metrics = ['return', 'sales', 'reinvest', 'redemptions', 'total net assets', 'cash', 
                      'filing_date', 'flows', 'vol_of_flows', 'turnover ratio', 'manager_tenure', 'age',
                      'realized alpha', 'alpha (intercept t-stat)', 'market beta t-stat', 'size beta t-stat',
                      'value beta t-stat', 'profit. beta t-stat', 'invest. beta t-stat', 'momentum beta t-stat',
                      'R2', 'realized alpha lagged', 'tna_lag', 'turnover_rate', 'filing_date_rr', 'expense_ratio']
        
        for metric in key_metrics:
            value = latest_row.get(metric, np.nan)
            if pd.notna(value):
                if isinstance(value, (int, float)):
                    fund_summary[metric] = round(value, 4)
                else:
                    fund_summary[metric] = value
        
        all_fund_data.append(fund_summary)
    
    # 2. Data Values by Fund - ONE CLASS PER SERIES (24 fields)
    print(f"   Processing {len(sample_funds)} representative funds (one per series) for detailed analysis...")
    
    # Define the 24 key fields to show
    key_fields = ['cik', 'series_id', 'class_id', 'month_end', 'r12_2', 'r36_13', 'flow', 'TNA', 
                  'TO', 'exp_ratio', 'expense_ratio', 'turnover_rate', 'Family_TNA', 'fund_age', 
                  'ST_vol', 'Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'alpha', 'r2', 'vol']
    
    for fund_id in sample_funds:
        fund_data = df[df['class_id'] == fund_id].copy()
        
        if len(fund_data) == 0:
            continue
        
        # Sort by date if available
        if 'month_end' in fund_data.columns:
            fund_data = fund_data.sort_values('month_end')
        
        # Get latest observation
        latest_row = fund_data.iloc[-1]
        
        # Track series/class coverage
        if 'series_id' in fund_data.columns:
            series_id = latest_row.get('series_id')
            if pd.notna(series_id):
                series_classes = df[df['series_id'] == series_id]['class_id'].nunique()
                series_class_summary.append({
                    'Series ID': series_id,
                    'Class ID (Representative)': fund_id,
                    'CIK': latest_row.get('cik', 'N/A'),
                    'Total Classes in Series': series_classes,
                    'Expense Ratio Coverage': fund_data['expense_ratio'].notna().sum() if 'expense_ratio' in fund_data.columns else 0,
                    'Turnover Rate Coverage': fund_data['turnover_rate'].notna().sum() if 'turnover_rate' in fund_data.columns else 0,
                    'Total Observations': len(fund_data)
                })
        
        # Create source documentation for ALL 24 key fields (not just those in DATA_SOURCES)
        for col in key_fields:
            if col in fund_data.columns:
                source_info = DATA_SOURCES.get(col, {
                    'source': 'Computed/Derived',
                    'url': 'N/A',
                    'file': 'Calculated field',
                    'description': f'Field: {col}'
                }).copy()
                source_info['Fund ID'] = fund_id
                source_info['Series ID'] = latest_row.get('series_id', 'N/A')
                source_info['Field'] = col
                source_info['Sample Value'] = latest_row.get(col, np.nan)
                source_documentation.append(source_info)
    
    # Convert to DataFrames
    df_summary = pd.DataFrame(all_fund_data)
    df_sources = pd.DataFrame(source_documentation)
    
    # Create Excel file with multiple sheets
    excel_file = output_path / f"etl_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Sheet 1: Fund Summary
        df_summary.to_excel(writer, sheet_name='Fund Summary', index=False)
        
        # Sheet 1.5: Series/Class Coverage
        if series_class_summary:
            df_series_class = pd.DataFrame(series_class_summary)
            df_series_class.to_excel(writer, sheet_name='Series Class Coverage', index=False)
        
        # Sheet 2: Data Sources
        if not df_sources.empty:
            df_sources_pivot = df_sources.pivot_table(
                index='Field',
                columns='Fund ID',
                values='Sample Value',
                aggfunc='first'
            )
            df_sources_pivot.to_excel(writer, sheet_name='Data Values by Fund')
        
        # Sheet 3: Source Documentation
        source_docs = []
        for field, info in DATA_SOURCES.items():
            source_docs.append({
                'Field': field,
                'Data Source': info['source'],
                'URL/Location': info['url'],
                'Source File': info['file'],
                'Description': info.get('description', '')
            })
        df_source_docs = pd.DataFrame(source_docs)
        df_source_docs.to_excel(writer, sheet_name='Source Documentation', index=False)
        
        # Sheet 4+: Time Series for representative funds (one per series)
        print(f"   Creating time series worksheets for {len(timeseries_funds)} representative series...")
        for i, fund_id in enumerate(timeseries_funds):
            fund_data = df[df['class_id'] == fund_id].copy()
            if len(fund_data) == 0:
                continue
                
            if 'month_end' in fund_data.columns:
                fund_data = fund_data.sort_values('month_end')
            
            # Get series and CIK info for sheet naming and identification
            latest_row = fund_data.iloc[-1] if len(fund_data) > 0 else None
            series_id = latest_row.get('series_id', 'NoSeries') if latest_row is not None else 'NoSeries'
            cik = latest_row.get('cik', 'NoCIK') if latest_row is not None else 'NoCIK'
            
            # Add identifying columns at the beginning
            fund_data_display = fund_data.copy()
            fund_data_display.insert(0, 'CIK', cik)
            fund_data_display.insert(1, 'Series_ID', series_id)
            fund_data_display.insert(2, 'Class_ID', fund_id)
            
            # Select key columns for time series view (all 24 fields if available)
            key_cols = ['month_end', 'r12_2', 'r36_13', 'flow', 'TNA', 'TO', 'exp_ratio', 
                       'expense_ratio', 'turnover_rate', 'Family_TNA', 'fund_age', 'ST_vol',
                       'Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'alpha', 'r2', 'vol',
                       'value_added', 'net_flow']
            available_cols = [col for col in key_cols if col in fund_data_display.columns]
            
            if available_cols:
                # Name sheet with series info (ensure unique names)
                if pd.notna(series_id) and series_id != 'NoSeries':
                    sheet_name = f"S_{str(series_id)[:20]}_{i}"[:31]  # Include index to ensure uniqueness
                else:
                    sheet_name = f"Fund_{i}_{str(fund_id)[:15]}"[:31]
                
                # Include CIK, Series_ID, Class_ID plus available columns
                cols_to_export = ['CIK', 'Series_ID', 'Class_ID'] + available_cols
                fund_data_display[cols_to_export].to_excel(
                    writer, 
                    sheet_name=sheet_name, 
                    index=False
                )
    
    print(f"✅ Excel validation report saved to: {excel_file}")
    
    # Also create a CSV for easy viewing
    csv_file = output_path / f"fund_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_summary.to_csv(csv_file, index=False)
    print(f"✅ CSV summary saved to: {csv_file}")
    
    # Create validation instructions
    instructions_file = output_path / "validation_instructions.md"
    with open(instructions_file, 'w') as f:
        f.write("# ETL Data Validation Instructions\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Selected Funds for Validation\n\n")
        
        for i, fund_id in enumerate(sample_funds[:10], 1):  # Show first 10 representative funds
            fund_info = df_summary[df_summary['Class ID'] == fund_id]
            if not fund_info.empty:
                cik = fund_info.iloc[0].get('CIK', 'N/A')
                series_id = fund_info.iloc[0].get('Series ID', 'N/A')
                f.write(f"{i}. **Class ID**: {fund_id}\n")
                f.write(f"   - CIK: {cik}\n")
                f.write(f"   - Series ID: {series_id}\n")
                f.write(f"   - SEC EDGAR URL: https://www.sec.gov/edgar/browse/?CIK={cik}\n\n")
        
        f.write("## How to Validate\n\n")
        f.write("1. **SEC Filings Validation**:\n")
        f.write("   - Go to SEC EDGAR using the URLs above\n")
        f.write("   - Look for N-PORT-P filings (monthly portfolio holdings)\n")
        f.write("   - Compare Total Net Assets (TNA) values\n")
        f.write("   - Check monthly returns in Part B Item 3\n")
        f.write("   - Verify fund flows in Part B Item 4\n\n")
        
        f.write("2. **Factor Data Validation**:\n")
        f.write("   - Visit Ken French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html\n")
        f.write("   - Download F-F Research Data 5 Factors 2x3\n")
        f.write("   - Download Momentum Factor\n")
        f.write("   - Verify factor values match for the corresponding months\n\n")
        
        f.write("3. **Expense Ratio & Turnover**:\n")
        f.write("   - Look for Form N-1A (Risk/Return Summary) in SEC filings\n")
        f.write("   - Or search for fund prospectus on fund company website\n")
        f.write("   - Compare expense ratio and turnover values\n\n")
        
        f.write("4. **Cross-Check with Financial Websites**:\n")
        f.write("   - Morningstar.com\n")
        f.write("   - Yahoo Finance\n")
        f.write("   - Fund company websites\n")
        f.write("   - Compare TNA, returns, expense ratios\n\n")
        
        f.write("## Data Fields and Sources\n\n")
        f.write("| Field | Source | Description |\n")
        f.write("|-------|--------|-------------|\n")
        for field, info in DATA_SOURCES.items():
            f.write(f"| {field} | {info['source']} | {info.get('description', '')} |\n")
    
    print(f"✅ Validation instructions saved to: {instructions_file}")
    
    return excel_file, csv_file, instructions_file

def main():
    """Main validation function"""
    print("=" * 60)
    print("ETL DATA VALIDATION TOOL")
    print("=" * 60)
    print("Validates actual data from ETL pipeline against original sources")
    print("=" * 60)
    
    # Check for actual ETL output
    data_file = "data/pilot_fact_class_month.parquet"
    
    # First check if file exists
    if not Path(data_file).exists():
        print(f"\n❌ No ETL output found at: {data_file}")
        print("\n📋 To generate real data for validation:")
        print("   1. Ensure you have .env file with SEC_EMAIL configured")
        print("   2. Run the ETL pipeline:")
        print("      python -m etl.pilot_pipeline --pilot config/funds_pilot.yaml --since 2023-01-01")
        print("\n   This will pull REAL data from:")
        print("   • SEC EDGAR (N-PORT filings)")
        print("   • Ken French Data Library (factor data)")
        print("\n   Then run this validation script again to verify the data.")
        return
    
    df = load_etl_data(data_file)
    
    if df is None:
        return
    
    # Display data overview
    print(f"\n📊 Data Overview:")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Date range: {df['month_end'].min() if 'month_end' in df.columns else 'N/A'} to {df['month_end'].max() if 'month_end' in df.columns else 'N/A'}")
    
    if 'class_id' in df.columns:
        print(f"  - Unique funds: {df['class_id'].nunique()}")
    
    # Get comprehensive fund coverage
    print(f"\n🎯 Analyzing comprehensive fund coverage...")
    all_funds_data, sample_funds, timeseries_funds = get_comprehensive_fund_coverage(df)
    
    if not all_funds_data:
        print("❌ Could not analyze fund coverage")
        return
    
    print(f"\n✅ Comprehensive Coverage Summary:")
    print(f"  - Total Fund-Class Combinations: {len(all_funds_data)}")
    print(f"  - Representative Funds (one per series): {len(sample_funds)}")
    print(f"  - Time Series Analysis Funds: {len(timeseries_funds)}")
    
    # Show series/class breakdown
    if 'series_id' in df.columns:
        unique_series = df['series_id'].dropna().nunique()
        unique_ciks = df['cik'].nunique() if 'cik' in df.columns else 'N/A'
        unique_classes = df['class_id'].nunique() if 'class_id' in df.columns else 'N/A'
        print(f"  - Unique CIKs: {unique_ciks}")
        print(f"  - Unique Series: {unique_series}")
        print(f"  - Unique Classes: {unique_classes}")
    
    # Create validation reports
    print(f"\n📝 Creating comprehensive validation reports...")
    excel_file, csv_file, instructions = create_validation_report(df, all_funds_data, sample_funds, timeseries_funds)
    
    print(f"\n✨ Validation reports created successfully!")
    print(f"\n📋 Next Steps:")
    print(f"  1. Open the Excel file: {excel_file}")
    print(f"  2. Review the validation instructions: {instructions}")
    print(f"  3. Cross-check values with SEC EDGAR and other sources")
    print(f"  4. Document any discrepancies found")
    
    # Print sample validation URLs
    print(f"\n🔗 Quick Validation Links:")
    unique_ciks = df['cik'].unique() if 'cik' in df.columns else []
    for cik in unique_ciks[:5]:  # Show first 5 CIKs
        if pd.notna(cik):
            print(f"  • CIK {cik}: https://www.sec.gov/edgar/browse/?CIK={cik}")

if __name__ == "__main__":
    main()