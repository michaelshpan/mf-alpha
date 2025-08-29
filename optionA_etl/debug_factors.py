#!/usr/bin/env python3
"""
Debug factor regression issues
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def debug_factor_data():
    """Debug why factor regressions aren't working"""
    
    # Load the latest output to see what we have
    out_path = Path("data/pilot_fact_class_month.parquet")
    if not out_path.exists():
        print("❌ Output file not found - run pipeline first")
        return
        
    df = pd.read_parquet(out_path)
    print(f"📊 Data shape: {df.shape}")
    print(f"📅 Date range: {df['month_end'].min()} to {df['month_end'].max()}")
    print(f"🏢 Classes: {df['class_id'].nunique()} unique")
    
    # Check factor data availability
    factor_cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"]
    
    print(f"\n📈 Factor data availability:")
    for col in factor_cols:
        if col in df.columns:
            count = df[col].notna().sum()
            print(f"  {col}: {count}/{len(df)} ({100*count/len(df):.1f}%)")
        else:
            print(f"  {col}: MISSING")
    
    # Check return data
    if "return" in df.columns:
        return_count = df["return"].notna().sum()
        print(f"  return: {return_count}/{len(df)} ({100*return_count/len(df):.1f}%)")
    
    # Check how many observations per class
    print(f"\n🔍 Observations per class:")
    class_counts = df.groupby("class_id").size().describe()
    print(class_counts)
    
    # Check how many classes have both return and factor data
    required_cols = ["return", "RF", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    complete_data = df[required_cols].notna().all(axis=1)
    complete_count = complete_data.sum()
    print(f"\n✅ Rows with complete data: {complete_count}/{len(df)} ({100*complete_count/len(df):.1f}%)")
    
    # Group by class and check complete data per class
    complete_by_class = df.groupby("class_id")[required_cols].apply(lambda x: x.notna().all(axis=1).sum())
    print(f"\n📋 Complete observations per class:")
    print(complete_by_class.sort_values(ascending=False).head(10))
    
    # Classes with 30+ complete observations (minimum for regressions)
    eligible_classes = complete_by_class[complete_by_class >= 30]
    print(f"\n🎯 Classes eligible for regression (30+ obs): {len(eligible_classes)}")
    if len(eligible_classes) > 0:
        print(eligible_classes)
        
        # Calculate expected total regression windows
        expected_regressions = 0
        for cid, count in eligible_classes.items():
            # For each class, we can run regressions starting from observation 30
            # up to the last observation (rolling window approach)
            expected_regressions += max(0, count - 29)  # 30 minimum, so first regression at obs 30
        
        print(f"\n🔢 Expected regression results: ~{expected_regressions}")

if __name__ == "__main__":
    debug_factor_data()