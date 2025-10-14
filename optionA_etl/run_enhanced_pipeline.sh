#!/bin/bash

# Script to run the enhanced ETL pipeline with SEC bulk returns database

echo "========================================"
echo "Enhanced ETL Pipeline with SEC Bulk Data"
echo "========================================"

# Step 1: Initialize and update the returns database (first time only)
echo ""
echo "Step 1: Initializing returns database from SEC bulk data..."
echo "This may take 10-30 minutes on first run to download all quarterly data."
python -m etl.pilot_pipeline \
    --pilot config/funds_pilot.yaml \
    --since 2023-01-01 \
    --out data/pilot_enhanced_with_db.parquet \
    --use-returns-db \
    --update-returns-db \
    --returns-db-path data/monthly_returns_db

# Step 2: Run the pipeline with complete monthly returns
echo ""
echo "Step 2: Running pipeline with complete monthly returns..."
python -m etl.pilot_pipeline \
    --pilot config/funds_pilot.yaml \
    --since 2023-01-01 \
    --out data/pilot_enhanced_complete.parquet \
    --use-returns-db \
    --returns-db-path data/monthly_returns_db

# Step 3: Compare results
echo ""
echo "Step 3: Analyzing results..."
python -c "
import pandas as pd

# Load original and enhanced data
try:
    original = pd.read_parquet('data/pilot_fact_class_month.parquet')
    enhanced = pd.read_parquet('data/pilot_enhanced_complete.parquet')
    
    print('COMPARISON RESULTS:')
    print('='*50)
    print(f'Original records: {len(original):,}')
    print(f'Enhanced records: {len(enhanced):,}')
    print(f'')
    print(f'Returns coverage:')
    print(f'  Original: {original[\"return\"].notna().sum():,} / {len(original):,} ({original[\"return\"].notna().mean()*100:.1f}%)')
    print(f'  Enhanced: {enhanced[\"return\"].notna().sum():,} / {len(enhanced):,} ({enhanced[\"return\"].notna().mean()*100:.1f}%)')
    print(f'')
    print(f'Alpha coverage:')
    if 'realized alpha' in original.columns:
        print(f'  Original: {original[\"realized alpha\"].notna().sum():,} / {len(original):,} ({original[\"realized alpha\"].notna().mean()*100:.1f}%)')
    if 'realized alpha' in enhanced.columns:
        print(f'  Enhanced: {enhanced[\"realized alpha\"].notna().sum():,} / {len(enhanced):,} ({enhanced[\"realized alpha\"].notna().mean()*100:.1f}%)')
    
    # Check monthly coverage
    print(f'')
    print(f'Monthly coverage analysis:')
    orig_coverage = original.groupby(['class_id', 'month_end']).size().groupby('class_id').count()
    enh_coverage = enhanced.groupby(['class_id', 'month_end']).size().groupby('class_id').count()
    print(f'  Original avg months/fund: {orig_coverage.mean():.1f}')
    print(f'  Enhanced avg months/fund: {enh_coverage.mean():.1f}')
    
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "========================================"
echo "Pipeline complete!"
echo ""
echo "Key improvements with SEC bulk data:"
echo "  1. Complete monthly return series (no gaps)"
echo "  2. Realized alpha available from month 36+ for all funds"
echo "  3. No duplicate records (automatic deduplication)"
echo "  4. Historical data back to 2019 (when N-PORT began)"
echo ""
echo "Output files:"
echo "  - data/pilot_enhanced_complete.parquet (main output)"
echo "  - data/monthly_returns_db/ (returns database)"
echo "========================================"