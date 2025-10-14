#!/bin/bash
# Run ETL pipeline with sufficient history for alpha calculations

echo "Running ETL pipeline with 60 months of history for alpha calculations..."
echo "This will fetch data from 2019-01-01 to ensure 36+ months for regression"

cd "$(dirname "$0")"

python -m etl.pilot_pipeline \
    --pilot config/funds_pilot.yaml \
    --since 2019-01-01 \
    --extra-history 60 \
    --out data/pilot_with_alpha.parquet

echo ""
echo "Checking results..."
python -c "
import pandas as pd
df = pd.read_parquet('data/pilot_with_alpha.parquet')
print(f'Total records: {len(df)}')
print(f'Date range: {df[\"month_end\"].min()} to {df[\"month_end\"].max()}')
alpha_coverage = df['realized alpha'].notna().sum()
print(f'Realized alpha coverage: {alpha_coverage}/{len(df)} ({100*alpha_coverage/len(df):.1f}%)')
"