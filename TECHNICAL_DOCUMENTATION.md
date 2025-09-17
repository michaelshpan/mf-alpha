# MF-Alpha Technical Documentation

## Project Overview

### Objective
MF-Alpha is a machine learning-based mutual fund selection system that predicts fund alpha (risk-adjusted excess returns) using fund characteristics. The system implements and extends the research methodology from DeMiguel et al.'s academic paper on ML-based fund selection, adapting it for practical deployment with real-time data collection capabilities.

### Key Innovation
- Replicates academic research using publicly available data sources instead of proprietary CRSP database
- Implements automated ETL pipeline for SEC EDGAR data collection
- Provides prediction service for external funds not in training dataset
- Uses 17 fund characteristics to predict 12-month forward alpha

## Research Foundation

### DeMiguel Paper Implementation
The project is based on DeMiguel et al.'s research on machine learning methods for mutual fund selection. Key aspects:

1. **Original Data Source**: Paper used CRSP mutual fund database (proprietary)
2. **Our Adaptation**: SEC EDGAR N-PORT filings and Risk/Return datasets (public)
3. **Methodology Preserved**:
   - Expanding window training (10-year minimum history)
   - Top decile portfolio construction
   - Monthly rebalancing with 12-month holding periods
   - Multiple ML models (OLS, Elastic Net, Random Forest, XGBoost)

4. **Key Differences**:
   - Cross-validation methodology: Different implementation for RF/XGB (not specified in original paper)
   - Data: Partially randomized in paper's provided dataset to preserve proprietary nature of source data
   - Source: Public SEC data vs proprietary CRSP database

### Model Target
- **Prediction Window**: 12-month forward alpha (`alpha6_1mo_12m`)
- **Features**: 17 lagged fund characteristics including past alpha, expense ratios, flows, factor loadings

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MF-Alpha System                          │
├───────────────────────────┬───────────────────────────────────┤
│     Training Pipeline     │        ETL Pipeline               │
├───────────────────────────┼───────────────────────────────────┤
│                           │                                   │
│  ┌──────────────┐        │   ┌─────────────────┐            │
│  │ Historical   │        │   │  SEC EDGAR API  │            │
│  │ Data (R/CSV) │        │   └────────┬────────┘            │
│  └──────┬───────┘        │            │                     │
│         │                │   ┌─────────▼────────┐            │
│  ┌──────▼───────┐        │   │  N-PORT Parser  │            │
│  │ Data Loader  │        │   └────────┬────────┘            │
│  └──────┬───────┘        │            │                     │
│         │                │   ┌─────────▼────────┐            │
│  ┌──────▼───────┐        │   │  Factor Merger  │            │
│  │ Model Train  │        │   │  (Ken French)   │            │
│  │ (OLS/EN/RF/  │        │   └────────┬────────┘            │
│  │    XGB)      │        │            │                     │
│  └──────┬───────┘        │   ┌─────────▼────────┐            │
│         │                │   │ Metric Computer │            │
│  ┌──────▼───────┐        │   │ (Regressions)  │            │
│  │ Portfolio    │        │   └────────┬────────┘            │
│  │ Constructor  │        │            │                     │
│  └──────┬───────┘        │   ┌─────────▼────────┐            │
│         │                │   │  SEC RR Data    │            │
│  ┌──────▼───────┐        │   │  Integration    │            │
│  │Model Persist │        │   └────────┬────────┘            │
│  │   (.pkl)     │        │            │                     │
│  └──────────────┘        │   ┌─────────▼────────┐            │
│                           │   │ Data Overrides  │            │
│                           │   └────────┬────────┘            │
│                           │            │                     │
│                           │   ┌─────────▼────────┐            │
│                           │   │  Output Parquet │            │
│                           │   └─────────────────┘            │
└───────────────────────────┴───────────────────────────────────┘
                                        │
                           ┌────────────▼────────────┐
                           │  Prediction Service    │
                           │  - Load Models         │
                           │  - Transform ETL Data  │
                           │  - Generate Alphas    │
                           └─────────────────────┘
```

## Module Dependencies

### Core Dependencies
```
Project Root (mf-alpha/)
│
├── src/                    [Training Pipeline]
│   ├── main.py            → models.py, portfolio.py, data_io.py
│   ├── predict.py         → portfolio.py, utils.py
│   ├── models.py          → sklearn, xgboost
│   ├── portfolio.py       → numpy
│   ├── data_io.py         → pandas, pyarrow, boto3
│   └── utils.py           → yaml, logging
│
├── optionA_etl/           [ETL Pipeline]
│   ├── etl/
│   │   ├── pilot_pipeline.py    → sec_edgar.py, factors.py, metrics.py
│   │   ├── sec_edgar.py         → requests, xml.etree
│   │   ├── nport_parser.py      → xml parsing libraries
│   │   ├── factors.py           → pandas_datareader
│   │   ├── metrics.py           → statsmodels
│   │   ├── sec_rr_integration.py → pandas
│   │   └── data_overrides.py    → pandas
│   │
│   └── prediction_service/      [Prediction Interface]
│       ├── predictor.py         → model_loader.py, data_preprocessor.py
│       ├── model_loader.py      → pickle, numpy
│       ├── data_preprocessor.py → pandas, sklearn.preprocessing
│       └── cli.py               → predictor.py
│
└── requirements.txt              [Python Dependencies]
```

### External Dependencies (requirements.txt)
- **Core ML**: scikit-learn, xgboost, statsmodels
- **Data Processing**: pandas, numpy, pyarrow
- **Cloud/Storage**: boto3 (AWS SDK)
- **Data Sources**: pandas-datareader (Ken French), requests (SEC API)
- **Utilities**: pyyaml, python-dotenv

## Module Descriptions

### Training Pipeline (`src/`)

#### `main.py`
- **Purpose**: Orchestrates full ML training pipeline
- **Key Functions**:
  - Loads historical fund data
  - Implements expanding window training
  - Generates portfolio backtests
  - Persists trained models
- **Output**: Portfolio returns, turnover metrics, saved models
- **Note**: Translated from original R code provided by DeMiguel et al

#### `predict.py`
- **Purpose**: Standalone prediction for new funds
- **Key Functions**:
  - Loads persisted models
  - Validates input data (17 features)
  - Generates alpha predictions
  - Creates top-decile selections
- **Input**: CSV with fund characteristics
- **Output**: Alpha predictions, portfolio weights
- **Note**: Translated from original R code provided by DeMiguel et al

#### `models.py`
- **Purpose**: ML model specifications and training logic
- **Models Implemented**:
  - OLS (baseline)
  - Elastic Net (with CV grid search)
  - Random Forest (with hyperparameter tuning)
  - XGBoost (gradient boosting with CV)
- **Key Functions**:
  - `get_model_specs()`: Returns model configurations
  - `fit_and_predict()`: Trains and predicts with optional CV

#### `portfolio.py`
- **Purpose**: Portfolio construction and return computation
- **Key Functions**:
  - `weights_in_top_funds()`: Top decile selection
  - `compute_portfolio_returns()`: Monthly rebalancing logic

#### `data_io.py`
- **Purpose**: Data loading utilities
- **Supports**: Local files, S3 buckets
- **Formats**: Parquet (primary), CSV

### ETL Pipeline (`optionA_etl/etl/`)

#### `pilot_pipeline.py`
- **Purpose**: Main ETL orchestrator
- **Flow**:
  1. Download N-PORT filings from SEC
  2. Parse XML for returns/flows
  3. Merge with factor data
  4. Compute rolling metrics
  5. Integrate SEC RR data
  6. Apply data overrides
  7. Output to parquet

#### `sec_edgar.py`
- **Purpose**: SEC EDGAR API interface
- **Features**:
  - Rate limiting (10 requests/second)
  - N-PORT-P filing downloads
  - Series/Class mapping

#### `nport_parser.py`
- **Purpose**: XML parsing for N-PORT filings
- **Extracts**:
  - Monthly returns
  - Net flows
  - Portfolio holdings
  - Total net assets

#### `factors.py`
- **Purpose**: Ken French factor data collection
- **Data**: FF5 factors + momentum, risk-free rate

#### `metrics.py`
- **Purpose**: Derived metric computation
- **Computes**:
  - 36-month rolling factor regressions
  - Net flow percentages
  - Value added (alpha - expense ratio) × TNA
  - Flow volatility

#### `sec_rr_integration.py`
- **Purpose**: Process SEC Risk/Return datasets
- **Data**: Quarterly XBRL filings
- **Extracts**: Expense ratios, turnover rates
- **Hierarchy**: Class-level expenses, Series-level turnover

#### `data_overrides.py`
- **Purpose**: Manual override system for missing data
- **Features**:
  - Template generation
  - Class/Series level targeting
  - Priority over SEC data

### Prediction Service (`optionA_etl/prediction_service/`)

#### `predictor.py`
- **Purpose**: Main prediction orchestrator
- **Functions**:
  - Model loading and ensemble creation
  - Batch prediction processing
  - Report generation

#### `model_loader.py`
- **Purpose**: Model management
- **Features**:
  - Pickle model loading
  - Ensemble methods (mean, weighted)
  - Model validation

#### `data_preprocessor.py`
- **Purpose**: ETL-to-DeMiguel feature transformation
- **Handles**:
  - Feature name mapping
  - Missing value imputation
  - Scaling/normalization

#### `cli.py`
- **Purpose**: Command-line interface
- **Commands**:
  - `predict`: Generate alpha predictions
  - `validate`: Check data compatibility
  - `info`: Show model information

## Data Architecture

### Data Sources

1. **Historical Training Data**
   - Source: DeMiguel paper's R dataset
   - Format: 11.8MB parquet file
   - Content: 18 fund characteristics, monthly returns
   - Period: Multi-decade historical data

2. **SEC EDGAR Data** (ETL Pipeline)
   - **N-PORT-P Filings**: Monthly portfolio data
     - Returns, flows, holdings
     - XML format, parsed programmatically
   - **Risk/Return Datasets**: Quarterly XBRL
     - Expense ratios (class-level)
     - Turnover rates (series-level)
     - Structured data format

3. **Ken French Data Library**
   - Fama-French 5 factors + momentum
   - Risk-free rate
   - Daily/monthly frequencies

4. **Manual Overrides**
   - CSV templates for missing data
   - Manager tenure, fund age
   - Expense ratios when SEC data unavailable
   - User collects data from public sources (e.g., Morningstar, fund sponsor website) and input into CSV template

### Data Flow

```
Historical R Data → Parquet Conversion → Training Pipeline → Models (.pkl)
                                                                ↓
SEC EDGAR → XML Parse → Factor Merge → Metrics → SEC RR → Override → Parquet
                                                                ↓
                                                     Prediction Service
                                                                ↓
                                                     Alpha Predictions
```

### Key Data Structures

#### Fund Identifiers
- **CIK**: Central Index Key (fund family level)
- **Series ID**: Investment company series
- **Class ID**: Share class (expense ratio level)
- **Ticker**: Market trading symbol
- See reference below for SEC securities/funds identifier data structure

#### Feature Set (17 DeMiguel Characteristics)
1. `alpha6_1mo_12m_lag_1`: Previous period alpha

2. `alpha6_tstat_lag_1`: Alpha statistical significance

3. `mtna_lag_1`: Log total net assets

4. `exp_ratio_lag_1`: Expense ratio

5. `age_lag_1`: Fund age in years

6. `flow_12m_lag_1`: 12-month net flows

7. `mgr_tenure_lag_1`: Manager tenure

8. `turn_ratio_lag_1`: Portfolio turnover

9. `flow_vol_12m_lag_1`: Flow volatility

10. `value_added_12m_lag_1`: Value added metric
    11-17. Factor betas and R-squared from 36-month regressions

    ![image-20250915145609095](/Users/michaelshpan/Library/Application Support/typora-user-images/image-20250915145609095.png)

## Development Environment

### Setup Requirements
- **Python**: 3.8+ (requirements.txt for packages)
- **OS**: Cross-platform (developed on macOS M1)
- **Memory**: 16GB+ recommended for full pipeline
- **Storage**: 10GB+ for data and models

### Local Development Workflow
```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Training
python src/main.py --config src/config.yaml --local_out outputs

# ETL Data Collection
cd optionA_etl
python -m etl.pilot_pipeline --pilot config/funds_pilot.yaml

# Prediction
python src/predict.py --input sample_funds.csv --output predictions.csv
```

## Current Limitations & Technical Debt

### Data Limitations
- Historical data partially randomized (paper authors' privacy)
- No CRSP database access (using public SEC data)
- Manual override required for some missing fields
- 36-month regression window reduces early data availability

### Technical Debt
- No automated testing framework
- Models not versioned/tracked
- No monitoring or alerting
- Single-machine deployment
- Hours-long training time
- Limited to 10-20 funds per ETL query (manual constraints)

### Known Issues
- Cross-validation methodology differs from paper
- Memory usage high during full pipeline runs
- No backup/disaster recovery procedures

## Future Development Roadmap

### Phase 1: Testing & Validation
- **Review current ETL structure for external funds -- enhance and validate data (Week 1 9/15-9/22)**
- Implement comprehensive test suite
- Add data quality checks
- Model performance validation framework
- Backtesting accuracy verification

### Phase 2: Prediction Service Expansion

- Toggle interface to select/upload funds for ETL/prediction
- Toggle time horizon for alpha prediction
- **Analytical functions**: 
  - Compare predicted alpha with actual alpha
  - Generate insights from predicted alpha 

### Phase 3: AWS Migration
- **Compute**: EC2/SageMaker for training
- **Storage**: S3 for data/models
- **Orchestration**: Step Functions/Airflow
- **Prediction**: Lambda/SageMaker endpoints
- **Monitoring**: CloudWatch metrics

### Optional: Service Expansion & Production Hardening

- Expand ETL to more fund families
- API development for predictions
- Model versioning with MLflow
- A/B testing framework
- Real-time prediction capabilities
- Automated retraining pipeline
- Performance optimization
- Horizontal scaling capability
- Disaster recovery procedures

## Collaboration Guidelines

### Development Process
- **Version Control**: Git with feature branches
- **Code Review**: PR-based review process
- **Communication**: Async-friendly (different time zones)
- **Documentation**: Update with code changes

### Code Standards
- Type hints for function signatures
- Docstrings for modules/functions
- Logging over print statements
- Error handling with specific exceptions
- Configuration via YAML/environment variables

### Testing Requirements
- Unit tests for new functions
- Integration tests for pipelines
- Data validation tests
- Performance benchmarks

## Quick Start for New Engineers

1. **Environment Setup**
   ```bash
   git clone [repository]
   cd mf-alpha
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Sample Prediction**
   ```bash
   python src/predict.py --input sample_funds.csv --output test_predictions.csv
   ```

3. **Explore ETL Pipeline**
   ```bash
   cd optionA_etl
   python -m etl.pilot_pipeline --pilot config/funds_pilot.yaml --dry-run
   ```

4. **Key Files to Review**
   - `src/main.py`: Training pipeline entry point
   - `src/models.py`: ML model specifications
   - `optionA_etl/etl/pilot_pipeline.py`: ETL orchestrator
   - `CLAUDE.md`: Project-specific instructions



### Reference: SEC Permanent Identifiers for Mutual Funds

For U.S.-domiciled mutual funds, the SEC assigns three levels of permanent identifiers, each reflecting a different level of the fund’s legal structure:

------

#### **1. CIK – Central Index Key (Registrant level)**

- **Format**: 10-digit numeric string (often shown without leading zeros).
- **Scope**: Identifies the **registrant entity** (typically the investment company trust).
- **Example**: Vanguard World Fund → CIK 0000052848.
- **Use**: All filings made by the registrant carry this CIK. Think of it as the “parent” identifier.

------

#### **2. Series ID (Fund level)**

- **Format**: Alphanumeric starting with **S** followed by 9 digits (e.g., S000004444).
- **Scope**: Identifies a **distinct series of the registrant**.
  - Each series corresponds to what investors think of as a “mutual fund” (e.g., Growth Fund of America, Vanguard 500 Index Fund).
- **Uniqueness**: Series IDs are unique across EDGAR and permanent once assigned.
- **Use**: Links filings, fees, and performance data at the **fund level**.

------

#### **3. Class/Contract ID (Share class level)**

- **Format**: Alphanumeric starting with **C** followed by 9 digits (e.g., C000025064).
- **Scope**: Identifies an individual **share class** within a series.
  - Classes differ by fee structure, distribution arrangements, or other terms (e.g., Class A, Class I, Admiral Shares).
- **Uniqueness**: Permanent identifiers for each class/contract across EDGAR.
- **Use**: Needed for precision when mapping expenses, NAVs, and flows, since **fees differ by class**.

------

#### **Hierarchy**

Registrant (CIK)
 └── Series (S#########)
       └── Class/Contract (C#########)

- **CIK** = the legal registrant (the umbrella trust or corporation).
- **Series ID** = each separate mutual fund under that umbrella.
- **Class ID** = each investor share class of that mutual fund.

------

✅ **Bottom line**:

- **CIK** is at the top (registrant/trust).

- **Series ID** identifies the fund.

- **Class ID** identifies the share class.

  They form a nested hierarchy that lets you uniquely reference any mutual fund share class registered with the SEC.



## Support & Resources

### Documentation
- `CLAUDE.md`: AI assistant instructions
- `README.md`: Basic project overview
- This document: Technical deep-dive

### Key Directories
- `src/`: Core ML pipeline
- `optionA_etl/`: Data collection system
- `outputs/`: Model artifacts and results
- `config/`: Configuration files

### Contact
- Project Owner: Michael Seohyun Pan
- Development Environment: 2021 MacBook Pro M1

---

*Document Version: 1.0*
*Last Updated: Sep 15, 2025*