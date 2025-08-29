#!/usr/bin/env python3
"""
User-friendly parquet file explorer
"""
import pandas as pd
import sys
from pathlib import Path

def explore_parquet(file_path: str, head_rows: int = 20, fund_id: str = None):
    """Explore parquet file in a user-friendly way"""
    
    print(f"📊 EXPLORING: {file_path}")
    print("=" * 60)
    
    try:
        # Load the data
        df = pd.read_parquet(file_path)
        
        # Basic info
        print(f"📈 Dataset Overview:")
        print(f"   • Total rows: {len(df):,}")
        print(f"   • Total columns: {len(df.columns)}")
        print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Column info
        print(f"\n📋 Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            null_pct = (len(df) - non_null) / len(df) * 100
            print(f"   {i:2d}. {col:<25} ({dtype:<12}) - {non_null:,} non-null ({null_pct:5.1f}% missing)")
        
        # Filter by fund if specified
        if fund_id and 'class_id' in df.columns:
            df_filtered = df[df['class_id'] == fund_id]
            print(f"\n🔍 Filtered to fund {fund_id}: {len(df_filtered)} rows")
            df_to_show = df_filtered
        else:
            df_to_show = df
        
        # Unique funds if class_id exists
        if 'class_id' in df.columns:
            unique_funds = df['class_id'].nunique()
            print(f"\n🏢 Unique funds: {unique_funds}")
            print("   Fund IDs:", df['class_id'].unique()[:10].tolist(), 
                  "..." if unique_funds > 10 else "")
        
        # Date range if month_end exists
        if 'month_end' in df.columns:
            date_col = pd.to_datetime(df['month_end'])
            print(f"\n📅 Date range:")
            print(f"   • Start: {date_col.min()}")
            print(f"   • End: {date_col.max()}")
            print(f"   • Unique dates: {date_col.nunique()}")
        
        # Prediction statistics if ensemble_prediction exists
        if 'ensemble_prediction' in df.columns:
            pred_col = df['ensemble_prediction']
            print(f"\n🎯 Prediction Statistics:")
            print(f"   • Mean: {pred_col.mean():.4f}")
            print(f"   • Median: {pred_col.median():.4f}")
            print(f"   • Std: {pred_col.std():.4f}")
            print(f"   • Min: {pred_col.min():.4f}")
            print(f"   • Max: {pred_col.max():.4f}")
            print(f"   • 25th percentile: {pred_col.quantile(0.25):.4f}")
            print(f"   • 75th percentile: {pred_col.quantile(0.75):.4f}")
        
        # Show first few rows
        print(f"\n📑 First {min(head_rows, len(df_to_show))} rows:")
        print("-" * 80)
        
        # Configure pandas display options for better readability
        with pd.option_context('display.max_columns', None, 
                               'display.width', 1000, 
                               'display.max_colwidth', 20):
            print(df_to_show.head(head_rows).to_string())
        
        # Value counts for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\n📊 Categorical Data Summary:")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                print(f"\n   {col} ({unique_count} unique values):")
                if unique_count <= 10:
                    print("   ", df[col].value_counts().head().to_dict())
                else:
                    print("   ", df[col].value_counts().head(5).to_dict(), "...")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_parquet.py <file_path> [head_rows] [fund_id]")
        print("\nExamples:")
        print("python explore_parquet.py predictions/alpha.parquet")
        print("python explore_parquet.py data/pilot_fact_class_month.parquet 10")
        print("python explore_parquet.py predictions/alpha.parquet 20 C000007112")
        sys.exit(1)
    
    file_path = sys.argv[1]
    head_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    fund_id = sys.argv[3] if len(sys.argv) > 3 else None
    
    df = explore_parquet(file_path, head_rows, fund_id)