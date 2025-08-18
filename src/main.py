import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from utils import setup_logging, load_config, ensure_dir
from data_io import read_characteristics, read_returns, write_parquet
from models import get_model_specs, fit_and_predict
from portfolio import weights_in_top_funds, compute_portfolio_returns

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="MF-Alpha ML Pipeline")
    p.add_argument("--config", default="src/config.yaml")
    p.add_argument("--local_out", default="outputs", help="local folder to also dump outputs")
    p.add_argument("--s3_out", default=None, help="s3 prefix to upload outputs (bucket/prefix)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    setup_logging(cfg["logging"]["level"])

# Build paths: local files if no bucket is set
    bucket = cfg["aws"].get("s3_bucket")
    if bucket:
        char_uri = f"s3://{bucket}/{cfg['aws']['characteristics_key']}"
        rets_prefix = f"s3://{bucket}/{cfg['aws']['returns_prefix']}"
        if not char_uri.startswith("s3://") and not Path(char_uri).exists():
            raise FileNotFoundError(f"Characteristics file not found: {char_uri}")
    else:
        char_uri = cfg["aws"]["characteristics_key"]
        rets_prefix = cfg["aws"]["returns_prefix"]

    # Vars & target
    vars_list = [
        "alpha6_1mo_12m",
        "alpha6_1mo_12m_lag_1","alpha6_tstat_lag_1","mtna_lag_1","exp_ratio_lag_1",
        "age_lag_1","flow_12m_lag_1","mgr_tenure_lag_1","turn_ratio_lag_1","flow_vol_12m_lag_1",
        "value_added_12m_lag_1","beta_market_tstat_lag_1","beta_profit_tstat_lag_1","beta_invest_tstat_lag_1",
        "beta_size_tstat_lag_1","beta_value_tstat_lag_1","beta_mom_tstat_lag_1","R2_lag_1"
    ]
    target_col = "alpha6_1mo_12m"  # following original R script

    # Load data
    df_char = read_characteristics(char_uri, usecols=["fundno", "Date"] + vars_list)
    df_rets = read_returns(rets_prefix)

    # Normalize characteristics dates as YYYYMM strings, then create Periods for robust date math
    df_char["Date"] = df_char["Date"].astype(str).str[:6]
    date_periods = pd.to_datetime(df_char["Date"], format="%Y%m").dt.to_period("M")
    date_vector = sorted(date_periods.unique())  # list/array of pandas Periods ('YYYY-MM')
    total_size = len(date_vector)
    panel_len = cfg["experiment"]["panel_years"]  # years
    top_cut = cfg["experiment"]["top_cut"]

    # Output containers
    strategies = cfg["experiment"]["models"]
    colnames = [f"{m}_top_{int((1-top_cut)*100)}" for m in strategies]
    portf_ret_matrix = []
    turnover = []
    w_big = {}

    # Pre-allocate w_big dict for turnover tracking
    fund_ids_all = df_char["fundno"].unique()
    for name in colnames:
        w_big[name] = pd.DataFrame(0.0, index=fund_ids_all, columns=range(total_size - panel_len))

    specs = get_model_specs(cfg)
    seed = cfg["experiment"]["seed"]

    portf_dates_rows = []
    
    logger.info("Earliest returns month: %s", df_rets["Date"].min().strftime("%Y-%m"))
    logger.info("Latest returns month: %s", df_rets["Date"].max().strftime("%Y-%m"))
    logger.info("Earliest formation date: %s", date_vector[0].strftime("%Y-%m"))
    logger.info("Latest formation date: %s", date_vector[-1].strftime("%Y-%m"))

    for t in range(total_size - panel_len):
        logger.info("Iteration %d / %d", t + 1, total_size - panel_len)
        formation_period = date_vector[t + panel_len]               # pandas Period('YYYY-MM', 'M')
        formation_date = formation_period.strftime("%Y%m")          # keep string for filtering df_char

        # TRAIN DATA (expanding window) — list of YYYYMM strings
        train_dates = [p.strftime("%Y%m") for p in date_vector[: t + panel_len]]
        train = df_char[df_char["Date"].isin(train_dates)].drop(columns=["Date", "fundno"])

        # TEST DATA (single date)
        test = df_char[df_char["Date"] == formation_date]
        X_test = test.drop(columns=["Date", "fundno", target_col], errors="ignore")
        fund_codes = test["fundno"].values

        # Build fund_returns matrix for 12‑month window using Period math
        year_start = pd.Period(f"{formation_period.year}01", "M").to_timestamp()
        year_end   = formation_period.to_timestamp()
        
        if year_start < df_rets["Date"].min():
            logger.warning("Skipping %s – incomplete return window", formation_date)
            continue

        fund_rets_subset = df_rets[
            (df_rets["Date"] >= year_start)
            & (df_rets["Date"] <= year_end)
            & (df_rets["fundno"].isin(fund_codes))
        ]
        
        # Handle duplicates by taking the mean
        fund_rets_subset = fund_rets_subset.groupby(["fundno", "Date"])["mret"].mean().reset_index()
        
        fund_rets_block = (
            fund_rets_subset
            .pivot(index="fundno", columns="Date", values="mret")
            .reindex(fund_codes)        # preserve order
            .to_numpy()
        )

        # ensure 12 columns (pad or trim if needed)
        if fund_rets_block.shape[1] != 12:
            # pad with NaNs or slice
            if fund_rets_block.shape[1] < 12:
                pad = np.full((fund_rets_block.shape[0], 12 - fund_rets_block.shape[1]), np.nan)
                fund_rets_block = np.concatenate([fund_rets_block, pad], axis=1)
            else:
                fund_rets_block = fund_rets_block[:, :12]

        this_iter_returns = []
        this_iter_turnover = []

        for model_name in strategies:
            spec = specs[model_name]
            X_train = train.drop(columns=[target_col], errors="ignore").values
            y_train = train[target_col].values

            y_hat = fit_and_predict(spec, X_train, y_train, X_test.values, seed)

            w = weights_in_top_funds(y_hat, top_cut)
            portf_ret, w_mat = compute_portfolio_returns(w, fund_rets_block)
            this_iter_returns.append(portf_ret)
            
            #check if weights are all 0
            if np.all(w == 0):
                print("all weights are 0")
                print("w", w)
                print("w_mat", w_mat)
                print("portf_ret", portf_ret)
                print("model_name", model_name)

            if t == 0:
                print("formation_date", formation_date)
                print("fund_rets_block shape", fund_rets_block.shape)
                print("fund_rets_block all_nan?", np.isnan(fund_rets_block).all())
                print("weights sum", w.sum(), "non-zero weights", (w > 0).sum())

            # turnover
            if t > 0:
                prev_w = w_big[f"{model_name}_top_{int((1-top_cut)*100)}"].iloc[:, t - 1].reindex(fund_codes).fillna(0).values
                curr = np.zeros_like(prev_w)
                curr[: len(w)] = w  # aligned by fund_codes order
                to = np.nansum(np.abs(curr - prev_w))
            else:
                to = np.nan
            this_iter_turnover.append(to)

            # store current weights for turnover next round
            w_df = pd.Series(0.0, index=fund_ids_all)
            w_df[fund_codes] = w
            w_big[f"{model_name}_top_{int((1-top_cut)*100)}"].iloc[:, t] = w_df.values

        portf_ret_matrix.append(np.column_stack(this_iter_returns))
        turnover.append(this_iter_turnover)

        # portfolio dates rows (12 months)
        mons = [f"{m:02d}" for m in range(1, 13)]
        dates_vec = [formation_date[:4] + m for m in mons]
        portf_dates_rows.extend(dates_vec)

    # Stack outputs
    portf_ret_matrix = np.vstack(portf_ret_matrix)
    turnover = np.vstack(turnover)

    # Build df.returns
    df_returns = pd.DataFrame(portf_ret_matrix, columns=colnames)
    df_returns.insert(0, "portf_dates", pd.to_datetime(pd.Series(portf_dates_rows) + "01", format="%Y%m%d"))
    df_returns["portf_dates"] = df_returns["portf_dates"].dt.strftime("%Y%m")

    # Turnover DF
    df_turnover = pd.DataFrame(turnover, columns=colnames)

    # Save locally
    local_out = Path(args.local_out)
    ensure_dir(local_out)
    df_returns.to_parquet(local_out / "portfolio_returns.parquet", index=False)
    df_turnover.to_parquet(local_out / "turnover.parquet", index=False)

    # Upload to S3 if requested
    if args.s3_out:
        write_parquet(df_returns, f"s3://{bucket}/{cfg['experiment']['output_prefix']}_returns.parquet")
        write_parquet(df_turnover, f"s3://{bucket}/{cfg['experiment']['output_prefix']}_turnover.parquet")

    logger.info("Done.")

if __name__ == "__main__":
    main()
