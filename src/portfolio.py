import numpy as np
import pandas as pd


def weights_in_top_funds(scores: np.ndarray, top_cut: float) -> np.ndarray:
    """Return normalized weights for funds above the given quantile.
    If quantile threshold < 0, rescale 0-1 first (paper/R logic).
    """
    q = np.quantile(scores, top_cut)
    if q < 0:
        # rescale 0-1
        smin, smax = scores.min(), scores.max()
        if smax == smin:
            rescaled = np.zeros_like(scores)
        else:
            rescaled = (scores - smin) / (smax - smin)
        mask = rescaled > np.quantile(rescaled, top_cut)
        w_raw = rescaled * mask
    else:
        mask = scores > q
        w_raw = scores * mask

    s = w_raw.sum()
    if s == 0:
        # fallback: equal weight across all positives or all funds
        if mask.sum() > 0:
            w = np.ones_like(scores) * mask / mask.sum()
        else:
            w = np.ones_like(scores) / len(scores)
    else:
        w = w_raw / s
    return w


def compute_portfolio_returns(w: np.ndarray, fund_returns: np.ndarray):
    """Replicates R computePortfolioReturns: 12x1 returns, with monthly compounding
    and weight rebalancing.
    fund_returns: shape (n_funds, 12) for the holding period.
    """
    portf_ret = np.full(12, np.nan)
    w_mat = []
    for r in range(12):
        portf_ret[r] = np.nansum(w * fund_returns[:, r])
        if r < 11:
            w = w * (1 + np.nan_to_num(fund_returns[:, r + 1], nan=0.0))
            s = w.sum()
            w = np.zeros_like(w) if s == 0 else w / s
            w_mat.append(w.copy())
    return portf_ret, np.column_stack(w_mat) if w_mat else np.empty((len(w), 0))

