# Data Dictionary

## characteristics/scaled_annual_data_JFE.parquet
- fundno (string): mutual fund ID
- Date (string, YYYYMM)
- [18 lagged, standardized vars...]
- year (int): derived

## returns/ (partitioned by year/month)
- fundno (string)
- Date (date, first day of month)
- mret (float): monthly return
- year (int)
- month (int)

## factors/ff_factors.parquet (optional)
- Date (date)
- Mkt_RF, SMB, HML, MOM, RMW, CMA, LIQ, RF, etc.
