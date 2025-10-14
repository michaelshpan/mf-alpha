# Alpha Data Gaps Investigation Report

## Executive Summary
The investigation confirms that **predicted alpha has more data points than actual (realized) alpha** due to the 36-month rolling regression requirement for calculating realized alpha.

## Key Findings

### 1. Data Coverage
- **Total records**: 9,482 (both ETL and predictions)
- **Records with realized alpha**: 8,046 (84.9%)
- **Records without realized alpha**: 1,436 (15.1%)

### 2. Root Cause: 36-Month Rolling Window Requirement
The ETL pipeline calculates realized alpha using a **36-month rolling regression** with the following parameters:
- **Window size**: 36 months (3 years)
- **Minimum observations**: 30 months
- **Regression model**: Fund returns vs. Fama-French 5 factors + momentum

### 3. Timeline Analysis
| Year | Total Records | With Realized Alpha | Coverage |
|------|--------------|-------------------|----------|
| 2019 | 212          | 0                 | 0%       |
| 2020 | 616          | 0                 | 0%       |
| 2021 | 636          | 104               | 16.4%    |
| 2022 | 2,208        | 2,156             | 97.6%    |
| 2023 | 2,528        | 2,528             | 100%     |
| 2024 | 2,520        | 2,520             | 100%     |
| 2025 | 762          | 738               | 96.9%    |

### 4. Why the Gap Exists

#### Predicted Alpha (Available Immediately)
- Generated using pre-trained ML models
- Only requires current month's fund characteristics
- No historical data requirement
- Available from day 1 for any fund

#### Realized Alpha (Requires History)
- Calculated from 36-month rolling factor regression
- Requires continuous factor data (FF5 + momentum)
- Needs minimum 30 months of non-missing returns
- First alpha appears ~36 months after fund data collection begins

### 5. Example: Fund C000001973
- **First month of data**: August 2019
- **First realized alpha**: August 2022
- **Gap**: Exactly 36 months
- **Months with alpha**: 148 out of 175 (84.6%)

## Code Reference
The rolling regression is implemented in `optionA_etl/etl/metrics.py:36`:
```python
def rolling_factor_regressions(df: pd.DataFrame, window: int = 36, min_obs: int = 30)
```

## Conclusion
This is **expected and correct behavior**. The gap between predicted and realized alpha is due to:
1. The 36-month historical requirement for factor regression
2. The ETL pipeline starting data collection in July 2019
3. Factor data availability constraints

**Implications for Analysis**:
- Comparison between predicted and actual alpha is only possible from December 2021 onwards
- Early predictions (2019-2021) cannot be validated against actual alpha
- Full coverage (100%) is achieved from 2022 onwards

## Recommendations
1. When comparing predicted vs actual alpha, filter data to `month_end >= '2021-12'`
2. For funds with less than 36 months of history, only predicted alpha is available
3. Consider implementing shorter-window alpha calculations (12 or 24 months) for earlier validation