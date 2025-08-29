# Option A — Mutual Fund ETL (Pilot)

Small-batch pilot for US mutual fund class-level ETL using public sources.
Pulls **N-PORT-P** (monthly returns + flows), **Ken French factors**, computes **net flows**,
rolling **FF5+MOM** regressions, and writes **class × month** parquet.

See Quick start below.
