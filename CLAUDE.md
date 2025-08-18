# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

The main pipeline is run via:
```bash
source .venv/bin/activate
python src/main.py [--config src/config.yaml] [--local_out outputs] [--s3_out bucket/prefix]
```

## Code Architecture

This is a mutual fund machine learning pipeline that replicates research on fund selection using ML methods. The codebase implements a rolling prediction system with multiple ML models.

### Core Components

- **`src/main.py`**: Main pipeline orchestrating the entire ML experiment with expanding window training
- **`src/data_io.py`**: Data loading utilities for parquet files (characteristics and returns)
- **`src/models.py`**: ML model specifications (OLS, Elastic Net, Random Forest, XGBoost) with hyperparameter grids
- **`src/portfolio.py`**: Portfolio construction logic - weight calculation and return computation with monthly rebalancing
- **`src/utils.py`**: Configuration loading and logging setup

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