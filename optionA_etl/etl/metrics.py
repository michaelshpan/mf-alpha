import pandas as pd
import numpy as np
import statsmodels.api as sm

def compute_net_flow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["net_flow"] = df["sales"] + df["reinvest"] - df["redemptions"]
    return df

def compute_flow_volatility(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Compute rolling volatility of flows (standard deviation) for each class.
    """
    df = df.copy()
    df = df.sort_values(["class_id", "month_end"])
    
    # Calculate rolling standard deviation of net flows by class
    df["vol_of_flows"] = df.groupby("class_id")["net_flow"].transform(
        lambda x: x.rolling(window=window, min_periods=3).std()
    )
    
    return df

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

def value_added(df: pd.DataFrame, er_map: pd.DataFrame, tna_proxy: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    merged = df.merge(er_map, on="class_id", how="left")
    
    if len(tna_proxy) > 0:
        merged = merged.merge(tna_proxy.rename(columns={"tna":"tna_proxy"}), on=["class_id","month_end"], how="left")
    else:
        merged["tna_proxy"] = None
        
    merged = merged.sort_values(["class_id","month_end"])
    merged["tna_lag"] = merged.groupby("class_id")["tna_proxy"].shift(1)
    
    # Calculate value added, handling missing expense ratio
    if "net_expense_ratio" in merged.columns and "alpha_hat" in merged.columns:
        merged["value_added"] = (merged["alpha_hat"] - merged["net_expense_ratio"]) * merged["tna_lag"]
    else:
        merged["value_added"] = None
        
    return merged
