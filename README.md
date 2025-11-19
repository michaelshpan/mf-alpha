# MF-Alpha

## 1. Purpose

Building a machine learning mutual fund and investment manager alpha prediction tool for institutional investors 
based on "Machine learning and fund characteristics help to select mutual funds with positive alpha" 
by DeMiguel et al. 

## 2. High-Level Architecture

**Data Pipeline Flow**:

1. **Collection Layer** – Multi-source data gathering from SEC, Ken French, and market data APIs
2. **Processing Layer** – Data standardization, cleaning, metric computation
3. **Storage Layer** – Parquet files with Hive partitioning for efficient retrieval
4. **ML Training Layer** – Expanding window model training with cross-validation
5. **Prediction Layer** – Ensemble alpha forecasting (OLS/Elastic Net/Random Forest/XGBoost)
6. **Portfolio Construction** – Top-decile selection with turnover tracking

**Key Components**:

- **ETL Module** (`optionA_etl/`): Automated real-time data collection
- **Feature Engineering**: 17 standardized fund characteristics computation
- **Model Training**: Multi-model ensemble with equal-weighted top decile
- **Backtesting Engine**: Rolling out-of-sample validation
- **Prediction Service**: Generate forecasts for new funds

## 3. Data Sources

**Primary Sources**:

- **SEC EDGAR API**: N-PORT filings for monthly returns, flows, holdings
- **Ken French Data Library**: Fama-French 5-factor + Momentum + Risk-free rates
- **Tradefeeds API**: Market data and monthly return calculations

**Fund Characteristics** (17 variables):

- Returns-based: Alpha, beta t-statistics, R-squared (36-month rolling regressions)
- Flow metrics: Net flows, flow volatility (12-month rolling)
- Cost metrics: Expense ratio, turnover ratio
- Size metrics: Total Net Assets (TNA)
- Performance: Realized alpha, value-added
- Other: Manager tenure, fund age

**Data Quality Features**:

- Priority-based deduplication across sources
- Date normalization (calendar month-end standardization)
- Robust handling of missing data periods
- Cross-validation between SEC, Tradefeeds, and factor data

## 4. Technical Requirements

**System Requirements**:

- Python 3.8+
- AWS infrastructure (future production deployment)
- 16GB+ RAM for large dataset processing
- Storage: Local disk or S3 for Parquet files

**Core Libraries**:

- **ML/Data Science**: scikit-learn, XGBoost, pandas, numpy
- **Data Storage**: pyarrow (Parquet), AWS S3 SDK (boto3)
- **API Integration**: requests (SEC EDGAR, Tradefeeds)
- **Statistical Analysis**: statsmodels (rolling regressions)
- **Configuration**: PyYAML
- **Validation**: pytest

**Development Tools**:

- Version control: Git
- AI coding assistants: Cursor, Claude Code
- Environment: Virtual environments (.venv)

**Infrastructure**:

- Development: Local Python environment
- Production: AWS (S3 for storage, EC2/Lambda for compute)
- Data format: Parquet with Hive partitioning
- Configuration: YAML-based hyperparameter management

## 5. Model Specifications

**Training Strategy**:

- 10-year expanding window with rolling predictions
- Annual retraining cadence
- Cross-validation for hyperparameter tuning

**Model Ensemble**:

- Ordinary Least Squares (OLS)
- Elastic Net (regularized regression)
- Random Forest
- XGBoost
- Equal-weighted top decile portfolio construction

**Target Variable**: Next 12-month realized alpha vs. Fama-French factor models

**Validation Metrics**:

- Out-of-sample alpha (basis points)
- Portfolio turnover
- Transaction cost sensitivity
- Factor-neutral alpha verification

## 6. Success Metrics

**Data Completeness**:

- Primary metric: Zero missing fields and months for user-defined mutual funds in ETL pipeline
- Success criteria: 100% completeness for all 17 fund characteristics across entire time series
- Measurement: Automated checks during ETL execution reporting null counts and date gaps

**Prediction Accuracy**:

- Out-of-sample testing for user-defined mutual funds using post-2020 data
- Compare predicted alpha versus actual realized alpha for holdout periods
- Success threshold: Positive correlation between predictions and outcomes; top-decile portfolio generates positive net alpha
- Measurement: Rolling validation reports with prediction error distributions and portfolio performance metrics

## 7. Monitoring & Alerting

**Data Quality Validation**:
The system generates sample spot check reports for manual verification across key data points:

**Spot Check Categories**:

1. **Monthly Returns**: Random sample of fund-month pairs with computed returns for manual verification against fund sponsor data
2. **Expense Ratios**: Sample of fund expense ratios for cross-check against prospectus filings
3. **Factor Data**: Sample of (factor, month) pairs from Ken French library to verify data integrity
4. **Flow Calculations**: Sample of net flow computations with underlying sales/redemption components
5. **Alpha Metrics**: Sample of 36-month rolling regression outputs (alpha, t-stats, R-squared)

**Output Format**: 

- CSV reports with fund identifiers, computed values, data source references, and timestamps
- Generated automatically with each ETL run
- Designed for quick manual verification against original sources

**Quality Metrics Tracked**:

- Percentage of successfully processed funds per run
- Number of missing data points flagged
- Deduplication statistics (records merged vs. retained)
- Date normalization issues detected

## 8. Scope & Deliverables

**Version 1.0 (US Market)**:

- Coverage: US open-end mutual funds + ETFs
- Refresh frequency: Quarterly (T+15)
- Output format: CSV with fund_id, expected_alpha, rank, decile, notes

**Future Expansion (V2)**:

- China market: Mutual funds + private funds
- Timeline: Within Year 1, depending on demand
