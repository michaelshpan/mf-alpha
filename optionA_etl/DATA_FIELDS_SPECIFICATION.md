# Mutual Fund Data Fields Specification

## Overview
This document specifies all data fields required for the mutual fund alpha prediction model, based on the DeMiguel et al. academic paper (Section 2.2) and the existing ETL implementation. Each field includes primary and backup data sources.

## Core Identifiers (Required for All Records)

### 1. class_id
- **Description**: Unique identifier for fund share class
- **Primary Source**: SEC Series/Class mapping CSV files
- **Backup Source**: SEC company_tickers_mf.json
- **Update Method**: IdentifierMapper module with cached lookups

### 2. series_id  
- **Description**: Unique identifier for fund series (contains multiple classes)
- **Primary Source**: SEC Series/Class mapping CSV files
- **Backup Source**: SEC company_tickers_mf.json
- **Update Method**: IdentifierMapper module with cached lookups

### 3. cik
- **Description**: Central Index Key for registrant
- **Primary Source**: Configuration file (config/funds_pilot.yaml)
- **Backup Source**: SEC Series/Class mapping
- **Update Method**: IdentifierMapper module

### 4. ticker
- **Description**: Trading symbol for fund class
- **Primary Source**: SEC Series/Class mapping CSV files  
- **Backup Source**: SEC company_tickers_mf.json
- **Update Method**: IdentifierMapper module

### 5. month_end
- **Description**: Month-end date for observation
- **Format**: YYYY-MM-DD (last day of month)
- **Source**: Generated based on reporting period

## Fund Characteristics (17 Variables per Academic Paper)

### 6. realized_alpha (α̂ᵢ,ₜ)
- **Description**: Intercept from FF5+MOM factor model regression
- **Calculation**: 36-month rolling OLS regression (min 30 observations)
- **Required Inputs**: monthly returns, FF5+MOM factors
- **Primary Source**: Computed from returns and factor data
- **No backup source** (core computed metric)

### 7. flows (flowᵢ,ₜ)
- **Description**: Net fund flows as percentage of TNA
- **Formula**: (TNAₜ - TNAₜ₋₁(1 + Rₜ)) / TNAₜ₋₁
- **Primary Source**: SEC N-PORT-P filings (sales, redemptions, reinvestments)
- **Backup Source**: Tradefeeds mutual_fund_periodical_returns API
- **Processing**: Hybrid approach with SEC priority

### 8. value_added (vaᵢ,ₜ)
- **Description**: Dollar value added by fund
- **Formula**: (alpha - expense_ratio) × lagged_TNA
- **Primary Source**: Computed field
- **Required Inputs**: realized_alpha, net_expense_ratio, TNA
- **No backup source** (computed metric)

### 9. vol_of_flows (σ(flow)ᵢ,ₜ)
- **Description**: Volatility of fund flows
- **Calculation**: 12-month rolling standard deviation of flows
- **Primary Source**: Computed from monthly flow data
- **Required Window**: Minimum 3 months
- **No backup source** (computed metric)

### 10. tna (TNAᵢ,ₜ)
- **Description**: Total Net Assets
- **Primary Source**: SEC N-PORT-P filings (total_investments field)
- **Backup Source**: Computed from holdings data if direct TNA unavailable
- **Processing**: XML parsing of N-PORT filings

### 11. net_expense_ratio (expenseᵢ,ₜ)
- **Description**: Annual expense ratio
- **Primary Source**: SEC Risk/Return quarterly datasets (TSV files)
- **Backup Source**: Manual override file (fees_turnover_override.csv)
- **Fallback**: SEC N-1A filings text extraction

### 12. fund_age (ageᵢ,ₜ)
- **Description**: Years since fund inception
- **Primary Source**: SEC N-1A filings (text extraction)
- **Backup Source**: Manual override file
- **Regex Patterns**: "commenced operations", "inception", "established"
- **Calculation**: (filing_date - inception_date) / 365.25

### 13. manager_tenure (tenureᵢ,ₜ)
- **Description**: Years of portfolio manager tenure
- **Primary Source**: SEC N-1A filings (text extraction)
- **Backup Source**: Manual override file  
- **Regex Patterns**: "since YYYY", "Portfolio Manager since", "joined"
- **Calculation**: filing_date.year - start_year

### 14. turnover_pct (turnoverᵢ,ₜ)
- **Description**: Portfolio turnover rate
- **Primary Source**: SEC Risk/Return quarterly datasets
- **Backup Source**: Manual override file (fees_turnover_override.csv)
- **Fallback**: SEC N-1A filings

### 15-21. Factor Betas (β̂ᴹᴷᵀ, β̂ˢᴹᴮ, β̂ᴴᴹᴸ, β̂ᴿᴹᵂ, β̂ᶜᴹᴬ, β̂ᴹᴼᴹ)
- **Description**: Loadings on Fama-French 5 factors + momentum
- **Calculation**: Coefficients from 36-month rolling regression
- **Primary Source**: Computed from factor regression
- **Required Inputs**: monthly returns, FF5+MOM factors
- **No backup source** (computed metrics)

### 22-27. Factor T-Statistics (t(β̂ᴹᴷᵀ), t(β̂ˢᴹᴮ), etc.)
- **Description**: T-statistics for factor loadings
- **Calculation**: From 36-month rolling regression standard errors
- **Primary Source**: Computed from factor regression
- **No backup source** (computed metrics)

### 28. adj_r_squared (R²ᵢ,ₜ)
- **Description**: Adjusted R-squared from factor regression
- **Calculation**: From 36-month rolling regression
- **Primary Source**: Computed from factor regression
- **No backup source** (computed metric)

## Supporting Data (Required for Calculations)

### 29. return
- **Description**: Monthly total return
- **Primary Source**: Tradefeeds historical_ohlcv API
- **Backup Source**: SEC N-PORT-P filings (if available)
- **Calculation**: (close_price_t - close_price_t-1) / close_price_t-1

### 30-36. Fama-French Factors (MKT_RF, SMB, HML, RMW, CMA, RF, MOM)
- **Description**: Monthly factor returns
- **Primary Source**: Ken French Data Library (Dartmouth)
- **URLs**:
  - FF5: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip
  - MOM: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip
- **No backup source** (authoritative source)
- **Processing**: Convert from percentage to decimal

### 37. sales
- **Description**: Monthly gross sales/subscriptions
- **Primary Source**: SEC N-PORT-P filings
- **Backup Source**: Tradefeeds mutual_fund_periodical_returns API
- **Field Mapping**: mon1Flow.sales, mon2Flow.sales, mon3Flow.sales

### 38. redemptions  
- **Description**: Monthly gross redemptions
- **Primary Source**: SEC N-PORT-P filings
- **Backup Source**: Tradefeeds mutual_fund_periodical_returns API
- **Field Mapping**: mon1Flow.redemption, mon2Flow.redemption, mon3Flow.redemption

### 39. reinvestments
- **Description**: Monthly dividend reinvestments
- **Primary Source**: SEC N-PORT-P filings
- **Backup Source**: Tradefeeds mutual_fund_periodical_returns API
- **Field Mapping**: mon1Flow.reinvestment, mon2Flow.reinvestment, mon3Flow.reinvestment

### 40. net_flow
- **Description**: Net monthly flow
- **Formula**: sales + reinvestments - redemptions
- **Primary Source**: Computed from flow components
- **No backup source** (computed metric)

## Data Quality Requirements

### Minimum Data Requirements
- **Factor Regression**: 30 months minimum, 36 months for full window
- **Flow Volatility**: 3 months minimum, 12 months for full window
- **Deduplication**: By class_id + month_end, prefer SEC over third-party sources

### Missing Data Handling
- **Returns**: Skip months with missing returns (log warning)
- **Expenses/Turnover**: Use most recent available value (forward-fill)
- **Manager Tenure/Fund Age**: Use N-1A from multiple years if needed (2020, 2015, 2010, 2005, 1990)
- **Factor Regressions**: Require minimum observations or set to NaN

## Data Collection Priority

### Tier 1 - Essential (Must Have)
1. Monthly returns (return)
2. Fama-French factors (MKT_RF, SMB, HML, RMW, CMA, RF, MOM)
3. Identifiers (class_id, series_id, cik, month_end)

### Tier 2 - Core Characteristics (Required for Model)
4. Realized alpha (computed)
5. Fund flows (sales, redemptions, reinvestments)
6. Total Net Assets (TNA)
7. Expense ratio
8. All factor betas and t-stats (computed)
9. Adjusted R-squared (computed)

### Tier 3 - Important Features
10. Turnover rate
11. Manager tenure
12. Fund age
13. Flow volatility (computed)
14. Value added (computed)

## Implementation Notes

### Rate Limiting
- **SEC EDGAR API**: 10 requests per second max
- **Tradefeeds API**: 1 request per 2 seconds
- **Ken French Data**: No rate limit (static files)

### Caching Strategy
- **Identifier mappings**: 7-day TTL, updatable via separate script
- **Factor data**: Monthly refresh (data updates monthly)
- **N-1A filings**: Cache indefinitely (historical documents)
- **Returns/flows**: No caching (always fetch latest)

### Data Source Selection Logic
```python
if sec_data_available and sec_data_valid:
    use_sec_data()
elif tradefeeds_data_available:
    use_tradefeeds_data()
elif override_file_has_data:
    use_override_data()
else:
    log_missing_data()
```

## Configuration-Driven Fund Selection

Funds are selected based on `config/funds_pilot.yaml`:
- By CIK (includes all series and classes)
- By series_id (includes all classes)
- By specific class_ids
- By ticker symbols

Example configuration:
```yaml
registrants:
  - name: "VANGUARD HORIZON FUNDS"
    cik: "0000932471"
    series_ids: ["S000002594"]
    class_ids: []  # Empty means all classes in series
```

---

## Summary

**Total Required Fields**: 17 fund characteristics per the academic paper
**Total Data Points Collected**: ~40 fields (including supporting data)
**Primary Data Sources**: SEC filings, Tradefeeds API, Ken French Data Library
**Computation Requirements**: 36-month rolling windows for factor models
**Update Frequency**: Monthly for returns/flows, quarterly for expenses/turnover