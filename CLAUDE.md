# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

### Training Pipeline
The main pipeline trains models and generates portfolio backtests:
```bash
source .venv/bin/activate
python src/main.py [--config src/config.yaml] [--local_out outputs] [--s3_out bucket/prefix]
```

### Alpha Prediction for New Funds
Generate alpha predictions for new mutual funds using trained models:
```bash
python src/predict.py --input sample_funds.csv [--models_dir outputs/models] [--output predictions.csv]
```

Required input: CSV with 17 fund characteristics (see sample_funds.csv for format)

### Real-Time Data Collection (New)
The `optionA_etl/` module provides automated data collection from public sources:
```bash
cd optionA_etl
python -m etl.pilot_pipeline --pilot config/funds_pilot.yaml --since 2023-01-01
```

This generates fresh mutual fund data compatible with the 17-variable prediction model.

## Code Architecture

This is a mutual fund machine learning pipeline that replicates research on fund selection using ML methods. The codebase implements a rolling prediction system with multiple ML models.

### Core Components

- **`src/main.py`**: Main pipeline orchestrating the entire ML experiment with expanding window training and model persistence
- **`src/predict.py`**: Standalone prediction script for generating alpha forecasts on new funds
- **`src/table2_stats.py`**: Generate Table 2 summary statistics from fund characteristics data
- **`src/data_io.py`**: Data loading utilities for parquet files (characteristics and returns)
- **`src/models.py`**: ML model specifications (OLS, Elastic Net, Random Forest, XGBoost) with hyperparameter grids
- **`src/portfolio.py`**: Portfolio construction logic - weight calculation and return computation with monthly rebalancing
- **`src/utils.py`**: Configuration loading and logging setup

### ETL Module (`optionA_etl/`)

Real-time data collection system that gathers the 17 required fund characteristics from public sources:

**Core Pipeline:**
- **`etl/pilot_pipeline.py`**: Main ETL orchestrator collecting data from SEC filings and factor libraries
- **`etl/config.py`**: Configuration management with SEC rate limiting and environment variables
- **`etl/utils.py`**: Rate limiting, environment variable helpers, and common utilities

**Data Collection:**
- **`etl/sec_edgar.py`**: SEC EDGAR interface for N-PORT-P filings (monthly returns and flows)
- **`etl/nport_parser.py`**: Parser for SEC N-PORT XML filings to extract fund-level data
- **`etl/factors.py`**: Ken French factor data collection (FF5+MOM, risk-free rate)

**Data Processing:**
- **`etl/metrics.py`**: Computation of derived metrics (net flows, rolling regressions, value added)
- **`etl/tna_reducer.py`**: TNA (Total Net Assets) proxy calculation using holdings data and flow reconciliation
- **`etl/oef_rr_extractor.py`**: OEF/RR (Open-End Fund Risk/Return) data extraction for expense ratios and turnover

**Testing & Validation:**
- **`tests/offline_smoketest.py`**: Offline testing with sample N-PORT data
- **`tests/fixtures/nport_sample.xml`**: Sample XML for testing parser functionality
- **`config/funds_pilot.yaml`**: Configuration specifying target fund families (CIKs and series/class IDs)

### Data Flow

1. Load fund characteristics (18 standardized variables) and monthly returns from partitioned parquet files
2. For each time period, train models on expanding window of historical data
3. Generate predictions for current period and construct top-decile portfolios
4. Compute 12-month portfolio returns with monthly rebalancing
5. Track turnover metrics across periods
6. Output portfolio returns and turnover to parquet files

### Key Data Structures

- **Characteristics**: Fund-level features with Date (YYYYMM string) and fundno identifiers
- **Returns**: Monthly fund returns partitioned by year/month in hive format
- **Portfolio matrix**: Time series of portfolio returns for each model strategy
- **Weight tracking**: Fund-level weights stored for turnover calculations

### Configuration

All experiment parameters are controlled via `src/config.yaml`:
- Model hyperparameter grids for cross-validation
- Panel length (10 years expanding window)
- Portfolio construction (top 10% funds)
- Data paths (local or S3)

### Data Handling Notes

- Returns data uses pandas Period arithmetic for robust date calculations
- Duplicate fund-date records are handled by taking mean values before pivoting
- Missing return windows are skipped with logging warnings
- Fund weights are rebalanced monthly within 12-month holding periods
- Trained models are automatically saved after the final training iteration for use in predictions

### Prediction Workflow

1. **Model Training**: Main pipeline saves trained models to `outputs/models/` directory
2. **Data Preparation**: New funds require 17 standardized characteristics (see `sample_funds.csv`)
3. **Alpha Generation**: Prediction script loads models and generates alpha forecasts
4. **Portfolio Construction**: Applies same top-decile selection logic as training pipeline
5. **Ensemble Results**: Combines predictions across all model types with consensus scoring

### Data Collection Workflow (ETL)

1. **SEC Filing Collection**: Automatically downloads N-PORT-P filings from EDGAR for specified fund families
2. **Data Parsing**: Extracts monthly returns, flows, and portfolio data from XML filings
3. **TNA Calculation**: Computes Total Net Assets using holdings data and flow reconciliation 
4. **Factor Integration**: Merges with Ken French factor data (FF5+MOM model)
5. **Metric Computation**: Calculates rolling 36-month factor regressions, net flows, and value added
6. **OEF/RR Enhancement**: Extracts expense ratios and turnover from risk/return filings
7. **Output Generation**: Produces fund-class-month level data in parquet format compatible with prediction pipeline

### Testing & Validation

The ETL module includes comprehensive testing:
- **Offline smoke testing** with sample N-PORT XML data
- **End-to-end validation** of the complete pipeline from parsing to output
- **Rate limiting verification** for SEC API compliance

### Data Sources

**Public Sources (ETL Module):**
- SEC N-PORT-P filings via EDGAR API (monthly returns, flows, holdings)
- Ken French Data Library (FF5 factors, momentum, risk-free rate)
- SEC Series & Class mapping data for fund identification

**Configuration:**
- Target fund families specified by CIK (Central Index Key)
- Configurable date ranges and filtering options
- Override files for expense ratios and turnover when needed