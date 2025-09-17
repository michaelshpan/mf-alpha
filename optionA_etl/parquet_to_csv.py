#!/usr/bin/env python3
"""
Simple script to convert pilot_fact_class_month.parquet to CSV
"""

import pandas as pd
from pathlib import Path
import sys

def parquet_to_csv(parquet_path="data/pilot_fact_class_month.parquet", 
                   csv_path="data/pilot_fact_class_month.csv"):
    """Convert parquet file to CSV"""
    
    # Check if parquet file exists
    if not Path(parquet_path).exists():
        print(f"❌ File not found: {parquet_path}")
        print("Please run the ETL pipeline first:")
        print("  python -m etl.pilot_pipeline --pilot config/funds_pilot.yaml")
        return False
    
    try:
        # Load parquet file
        print(f"📖 Reading parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        # Display info
        print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
        print(f"   Date range: {df['month_end'].min() if 'month_end' in df.columns else 'N/A'} to {df['month_end'].max() if 'month_end' in df.columns else 'N/A'}")
        
        if 'class_id' in df.columns:
            print(f"   Unique classes: {df['class_id'].nunique()}")
        if 'series_id' in df.columns:
            print(f"   Unique series: {df['series_id'].dropna().nunique()}")
        if 'cik' in df.columns:
            print(f"   Unique CIKs: {df['cik'].nunique()}")
        
        # Write to CSV
        print(f"\n📝 Writing CSV file: {csv_path}")
        df.to_csv(csv_path, index=False)
        
        # Check file size
        csv_size = Path(csv_path).stat().st_size / (1024 * 1024)  # Size in MB
        print(f"✅ CSV saved successfully ({csv_size:.1f} MB)")
        
        # Show sample of columns
        print(f"\n📊 Columns in dataset:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
            if i >= 20 and len(df.columns) > 20:
                print(f"   ... and {len(df.columns) - 20} more columns")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert parquet file to CSV')
    parser.add_argument('--input', '-i', 
                       default='data/pilot_fact_class_month.parquet',
                       help='Input parquet file path')
    parser.add_argument('--output', '-o',
                       default='data/pilot_fact_class_month.csv', 
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PARQUET TO CSV CONVERTER")
    print("=" * 60)
    
    success = parquet_to_csv(args.input, args.output)
    
    if success:
        print(f"\n✨ Conversion complete!")
        print(f"📋 You can now open the CSV file in Excel or any text editor")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()