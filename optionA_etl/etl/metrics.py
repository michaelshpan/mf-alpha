import pandas as pd
import numpy as np
import statsmodels.api as sm

def compute_net_flow(df: pd.DataFrame) -> pd.Series:
    """Compute net flow as sales + reinvestments - redemptions."""
    return df["sales"] + df["reinvest"] - df["redemptions"]

def compute_flow_volatility(df: pd.DataFrame, window: int = 12) -> pd.Series:
    """
    Compute rolling volatility of flows (standard deviation) for each class.
    """
    # Ensure data is sorted by class and date
    df = df.sort_values(["class_id", "month_end"])
    
    # Calculate rolling standard deviation of net flows by class
    return df.groupby("class_id")["net_flow"].transform(
        lambda x: x.rolling(window=window, min_periods=3).std()
    )

def compute_realized_alpha_lagged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lagged realized alpha for each class.
    """
    df = df.copy()
    df = df.sort_values(["class_id", "month_end"])
    
    # Lag alpha_hat by 1 period within each class
    df["realized_alpha_lagged"] = df.groupby("class_id")["alpha_hat"].shift(1)
    
    return df

def rolling_factor_regressions(df: pd.DataFrame, window: int = 36, min_obs: int = 30) -> pd.DataFrame:
    results = []
    grouped = df.groupby("class_id")
    for cid, g in grouped:
        g = g.sort_values("month_end").reset_index(drop=True)
        for i in range(len(g)):
            window_df = g.iloc[max(0, i-window+1):i+1]
            if len(window_df) < min_obs:
                continue
            
            # Clean data: remove rows with NaN/inf in required columns
            required_cols = ["return", "RF", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
            clean_df = window_df[required_cols].dropna()
            
            if len(clean_df) < min_obs:
                continue
                
            # Check for infinite values
            if not np.isfinite(clean_df.values).all():
                continue
            
            y = clean_df["return"] - clean_df["RF"]
            X = clean_df[["MKT_RF","SMB","HML","RMW","CMA","MOM"]]
            X = sm.add_constant(X)
            
            try:
                model = sm.OLS(y, X).fit()
                row = {
                    "class_id": cid,
                    "month_end": g.loc[i,"month_end"],
                    "alpha_hat": model.params["const"],
                    "alpha_t": model.tvalues["const"],
                    "market_beta_t": model.tvalues["MKT_RF"],
                    "size_beta_t": model.tvalues["SMB"],
                    "value_beta_t": model.tvalues["HML"],
                    "profit_beta_t": model.tvalues["RMW"],
                    "invest_beta_t": model.tvalues["CMA"],
                    "momentum_beta_t": model.tvalues["MOM"],
                    "R2": model.rsquared
                }
                results.append(row)
            except Exception as e:
                # Skip this regression if it fails
                continue
    return pd.DataFrame(results)

def value_added(df: pd.DataFrame) -> pd.Series:
    """
    Compute value added as (alpha - expense_ratio) * lagged_TNA.
    Expects df to have columns: realized_alpha, net_expense_ratio, tna (or total_investments)
    """
    # Ensure data is sorted by class and date
    df = df.sort_values(["class_id", "month_end"])
    
    # Use TNA or total_investments
    tna_col = 'tna' if 'tna' in df.columns else 'total_investments'
    
    if tna_col not in df.columns:
        # No TNA data available
        return pd.Series(index=df.index, dtype=float)
    
    # Calculate lagged TNA
    tna_lag = df.groupby("class_id")[tna_col].shift(1)
    
    # Calculate value added, handling missing data
    if "net_expense_ratio" in df.columns and "realized_alpha" in df.columns:
        alpha = df["realized_alpha"].fillna(0)
        expense = df["net_expense_ratio"].fillna(0)
        return (alpha - expense) * tna_lag
    else:
        # Missing required columns
        return pd.Series(index=df.index, dtype=float)
