from __future__ import annotations

import numpy as np
import pandas as pd

class VaRMetricsHelper:
    @staticmethod
    def sqrt_time_scale_returns(returns: pd.Series, holding_period: int) -> pd.Series:
        return returns * np.sqrt(holding_period)

    @staticmethod
    def quantile_return(returns: pd.Series, confidence_level: float) -> float:
        alpha = 1.0 - confidence_level
        return float(np.percentile(returns.dropna(), alpha * 100.0))

    @staticmethod
    def tail_mean_return(returns: pd.Series, confidence_level: float) -> float:
        cutoff = VaRMetricsHelper.quantile_return(returns, confidence_level)
        tail = returns[returns <= cutoff].dropna()

        if tail.empty:
            return cutoff

        return float(tail.mean())

    @staticmethod
    def return_to_loss_amount(return_value: float, position_value: float) -> float:
        loss_amount = -float(return_value) * float(position_value)
        return max(loss_amount, 0.0)

    @staticmethod
    def return_to_pnl_amount(return_value: float, position_value: float) -> float:
        return float(return_value) * float(position_value)

    @staticmethod
    def losses_from_returns(returns: pd.Series, position_value: float) -> pd.Series:
        return -returns * float(position_value)

    @staticmethod
    def basic_summary_stats(returns: pd.Series) -> dict[str, float]:
        clean = returns.dropna()
        return {
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)),
            "min": float(clean.min()),
            "max": float(clean.max()),
            "skew": float(clean.skew()),
            "kurtosis": float(clean.kurt()),
            "n_obs": int(clean.shape[0]),
        }