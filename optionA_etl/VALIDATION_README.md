# ETL Data Validation Guide

## Purpose
This validation system verifies the accuracy of data collected by the ETL pipeline by comparing it against original sources (SEC EDGAR, Ken French Data Library, etc.).

## Important Note
**The validation script is designed to validate REAL data from the ETL pipeline, not sample/synthetic data.** The purpose is to ensure that the ETL pipeline correctly:
- Parses SEC N-PORT filings
- Calculates financial metrics
- Merges factor data accurately
- Computes derived variables correctly

## Prerequisites

1. **Configure SEC Access**
   ```bash
   cp .env.template .env
   # Edit .env and add your email for SEC_EMAIL
   ```

2. **Run the Actual ETL Pipeline**
   ```bash
   python -m etl.pilot_pipeline --pilot config/funds_pilot.yaml --since 2023-01-01
   ```
   This will download REAL data from:
   - SEC EDGAR N-PORT-P filings
   - Ken French Data Library
   - SEC Risk/Return summaries

## Running Validation

After the ETL pipeline completes:

```bash
python validate_etl_data.py
```

This will:
1. Load the actual ETL output from `data/pilot_fact_class_month.parquet`
2. Select 5 diverse funds for validation
3. Generate validation reports in `validation/` directory

## Validation Outputs

### 1. Excel Report (`etl_validation_[timestamp].xlsx`)
- **Fund Summary**: Latest values for all metrics
- **Data Values by Fund**: Side-by-side comparison
- **Source Documentation**: Complete field-to-source mapping  
- **Time Series**: Monthly data for each fund

### 2. CSV Summary (`fund_summary_[timestamp].csv`)
Quick reference file with key metrics for selected funds

### 3. Validation Instructions (`validation_instructions.md`)
Step-by-step guide for manual verification with direct links to:
- SEC EDGAR filings for each fund
- Ken French Data Library
- Verification steps for each data field

## How to Validate

### Step 1: Check SEC Filing Data
1. Open the Excel report
2. Note the CIK for each fund
3. Go to SEC EDGAR: `https://www.sec.gov/edgar/browse/?CIK=[CIK]`
4. Find recent N-PORT-P filings
5. Compare:
   - Total Net Assets (Part B, Item 1)
   - Monthly Returns (Part B, Item 3)
   - Fund Flows (Part B, Item 4)

### Step 2: Verify Factor Data
1. Visit [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
2. Download:
   - F-F Research Data 5 Factors 2x3
   - Momentum Factor
3. Compare factor values for corresponding months

### Step 3: Check Computed Metrics
Verify calculations:
- **Alpha**: Regression intercept from factor model
- **R-squared**: Should be between 0 and 1
- **Net flows**: (TNA_t - TNA_{t-1} * (1 + return)) / TNA_{t-1}
- **Value added**: Fund return - benchmark return

### Step 4: Document Findings
Record any discrepancies between:
- ETL output vs SEC filings
- Factor loadings vs Ken French data
- Computed metrics vs expected calculations

## Data Source Reference

| Field | Source | Verification Method |
|-------|--------|-------------------|
| TNA | SEC N-PORT-P Part B Item 1 | Compare with filing |
| Monthly Returns | SEC N-PORT-P Part B Item 3 | Compare with filing |
| Expense Ratio | SEC Form N-1A | Check prospectus |
| Turnover | SEC Form N-1A | Check prospectus |
| Factor Data | Ken French Library | Download and compare |
| Alpha | Computed from regression | Verify calculation |

## Troubleshooting

### No data found
Run the ETL pipeline first:
```bash
python -m etl.pilot_pipeline --pilot config/funds_pilot.yaml --since 2023-01-01
```

### SEC rate limiting
The ETL pipeline includes rate limiting. If you encounter issues:
- Check your SEC_EMAIL in .env
- Wait a few minutes and retry
- Use shorter date ranges

### Missing fund data
Some funds may not have complete data if:
- They're newly created
- N-PORT filings are delayed
- Data is not yet publicly available

## Why Real Data Matters

Validating with real ETL output ensures:
- **Parsing accuracy**: XML/HTML parsers extract correct values
- **Data integrity**: No data corruption during processing
- **Calculation correctness**: Derived metrics are computed properly
- **Source alignment**: Data matches authoritative sources

Using synthetic/sample data would only test the validation script itself, not the actual ETL pipeline's accuracy.