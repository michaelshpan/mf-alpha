# SEC Bulk Data Integration for Complete Monthly Returns

## Overview
This enhancement integrates SEC's free bulk data download service to provide complete monthly return data for all target funds, enabling accurate realized alpha calculations from month 1.

## Components

### 1. SEC Bulk Data Downloader (`etl/sec_bulk_downloader.py`)
- Downloads quarterly N-PORT bulk data from SEC's free service
- No API key required, respects SEC rate limits (10 req/sec)
- Extracts monthly returns from structured data files
- Caches downloaded data locally for efficiency

### 2. Monthly Returns Database (`etl/returns_database.py`)
- Manages local parquet database of monthly returns
- Integrates with pilot configuration for target funds
- Provides complete monthly series with gap detection
- Supports incremental updates

### 3. Enhanced Pilot Pipeline
- New flags: `--use-returns-db`, `--update-returns-db`
- Automatic deduplication of N-PORT overlapping data
- Seamless integration of bulk returns with existing pipeline
- Improved alpha calculation coverage

## Usage

### Initial Setup (One-time)
Build the returns database from SEC bulk data:

```bash
cd optionA_etl
python -m etl.pilot_pipeline \
    --pilot config/funds_pilot.yaml \
    --since 2023-01-01 \
    --out data/pilot_with_sec_bulk.parquet \
    --use-returns-db \
    --update-returns-db
```

This will:
1. Download all available N-PORT quarterly data from SEC
2. Extract monthly returns for target funds (from pilot config)
3. Build local parquet database
4. Process ETL with complete monthly returns

### Regular Usage
Once database is built, run pipeline with complete returns:

```bash
python -m etl.pilot_pipeline \
    --pilot config/funds_pilot.yaml \
    --since 2023-01-01 \
    --out data/pilot_enhanced.parquet \
    --use-returns-db
```

### Update Database
To get latest quarterly data:

```bash
python -m etl.pilot_pipeline \
    --pilot config/funds_pilot.yaml \
    --since 2023-01-01 \
    --out data/pilot_updated.parquet \
    --use-returns-db \
    --update-returns-db
```

## Data Sources

### SEC Bulk Data Service
- **URL**: https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets
- **Format**: Quarterly ZIP files containing TSV data
- **Coverage**: All registered mutual funds filing N-PORT
- **Frequency**: Quarterly (60-day lag)
- **Cost**: FREE

### Data Structure
Each quarterly file contains:
- `sub.txt`: Submission metadata (CIK, filing dates)
- `num.txt`: Numeric data including monthly returns
- `tag.txt`: Tag definitions
- Monthly returns reported as:
  - MonthlyReturn1: Current month (quarter end)
  - MonthlyReturn2: Previous month
  - MonthlyReturn3: Two months prior

## Benefits

### Before Integration
- **32.6% monthly coverage** (only quarters with N-PORT-P filings)
- **10-month gaps** between quarters
- **Duplicate records** from overlapping filings
- **No alpha for first 36 months** of fund data

### After Integration
- **~100% monthly coverage** (all available months)
- **No gaps** in monthly return series
- **Automatic deduplication** (one record per fund-month)
- **Alpha available from month 36+** for all periods

## Database Structure

### Location
```
data/monthly_returns_db/
├── monthly_returns.parquet      # Main returns data
├── fund_index.parquet           # Quick lookup index
├── metadata.json                # Database metadata
└── sec_cache/                   # Downloaded SEC files
    ├── 2023q1_form_nport.zip
    ├── 2023q2_form_nport.zip
    └── ...
```

### Schema
```python
monthly_returns.parquet:
- cik: str (10-digit, zero-padded)
- class_id: str
- series_id: str (optional)
- month_end: datetime
- return: float (decimal, not percentage)
- filing_date: datetime
- report_period: datetime
```

## Performance

### Download Times
- Initial setup: 10-30 minutes (downloading all quarters since 2019)
- Incremental update: 2-5 minutes per new quarter
- Database query: <1 second for fund lookups

### Storage Requirements
- SEC cache: ~500MB per quarter (compressed)
- Returns database: ~50-100MB (parquet compressed)
- Total: ~5-10GB for complete history

## Error Handling

### Common Issues and Solutions

1. **Rate Limiting**
   - Automatic 150ms delay between requests
   - Well under SEC's 10 req/sec limit

2. **Missing Data**
   - Some funds may not have complete history
   - Pipeline handles gracefully with warnings

3. **Duplicate Returns**
   - Automatic deduplication keeps most recent filing
   - Configurable to use mean/median if preferred

## Monitoring

Check database status:
```python
from etl.returns_database import MonthlyReturnsDatabase

db = MonthlyReturnsDatabase()
stats = db.get_database_stats()
print(f"Total records: {stats['total_records']}")
print(f"Date range: {stats['date_range']}")
print(f"Coverage: {stats['coverage']}")
```

## Future Enhancements

1. **Real-time Updates**
   - Monitor SEC RSS feeds for new filings
   - Automatic daily/weekly updates

2. **Alternative Sources**
   - Integration with CRSP database
   - Yahoo Finance / Bloomberg fallbacks

3. **Data Quality**
   - Anomaly detection for returns
   - Cross-validation with other sources

## Testing

Run tests:
```bash
# Test database functionality
python -m etl.returns_database

# Test bulk downloader
python -m etl.sec_bulk_downloader

# Run enhanced pipeline test
./run_enhanced_pipeline.sh
```

## Compliance Note
This implementation:
- Uses only publicly available SEC data
- Respects SEC rate limits and user agent requirements
- Does not require authentication or API keys
- Complies with SEC's fair access policy

## Support
For issues or questions:
1. Check SEC data availability: https://www.sec.gov/data-research
2. Verify fund CIKs in pilot configuration
3. Review logs in `data/monthly_returns_db/logs/`