from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class VaRValidator(object):
    @staticmethod
    def validate_confidence_level(confidence_level: float) -> None:
        if not isinstance(confidence_level, (int, float, np.floating)):
            raise TypeError("confidence_level must be a numeric type.")
        if not (0.0 < float(confidence_level) < 1.0):
            raise ValueError("confidence_level must be between 0 and 1.")

    @staticmethod
    def validate_holding_period(holding_period: int) -> None:
        if not isinstance(holding_period, (int, np.integer)):
            raise TypeError("holding_period must be an integer.")
        if int(holding_period) < 1:
            raise ValueError("holding_period must be greater than or equal to 1.")

    @staticmethod
    def validate_position_value(position_value: float) -> None:
        if not isinstance(position_value, (int, float, np.floating)):
            raise TypeError("position_value must be a numeric type.")
        if float(position_value) <= 0:
            raise ValueError("position_value must be positive.")

    @staticmethod
    def validate_n_simulations(n_simulations: int) -> None:
        if not isinstance(n_simulations, (int, np.integer)):
            raise TypeError("n_simulations must be an integer.")
        if int(n_simulations) < 1:
            raise ValueError("n_simulations must be greater than or equal to 1.")

    @staticmethod
    def validate_non_empty_series(series: pd.Series, name: str = "series") -> None:
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")
        if series.empty:
            raise ValueError(f"{name} must not be empty.")
        if series.dropna().empty:
            raise ValueError(f"{name} must contain at least one non-null value.")

    @staticmethod
    def validate_non_empty_dataframe(df: pd.DataFrame, name: str = "dataframe") -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
        if df.empty:
            raise ValueError(f"{name} must not be empty.")
        if df.dropna(how="all").empty:
            raise ValueError(f"{name} must contain at least one non-null row.")

    @staticmethod
    def validate_returns(series: pd.Series, name: str = "returns") -> None:
        VaRValidator.validate_non_empty_series(series, name=name)

        if not np.issubdtype(series.dropna().dtype, np.number):
            raise TypeError(f"{name} must contain numeric values.")

        if np.isinf(series.dropna().to_numpy()).any():
            raise ValueError(f"{name} must not contain infinite values.")

    @staticmethod
    def validate_weights(weights: pd.Series, tolerance: float = 1e-6) -> None:
        VaRValidator.validate_non_empty_series(weights, name="weights")

        if not np.issubdtype(weights.dropna().dtype, np.number):
            raise TypeError("weights must contain numeric values.")

        total = float(weights.sum())
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"weights must sum to 1.0 within tolerance={tolerance}.")

    @staticmethod
    def validate_same_length(
        left: pd.Series | pd.DataFrame,
        right: pd.Series | pd.DataFrame,
        left_name: str = "left",
        right_name: str = "right",
    ) -> None:
        if len(left) != len(right):
            raise ValueError(f"{left_name} and {right_name} must have the same length.")

    @staticmethod
    def validate_required(value, name: str) -> None:
        if value is None:
            raise ValueError(f"{name} is required.")

    @staticmethod
    def validate_shock_values(shocks: Iterable[float]) -> None:
        shocks = list(shocks)
        if len(shocks) == 0:
            raise ValueError("shocks must contain at least one value.")
        for shock in shocks:
            if not isinstance(shock, (int, float, np.floating)):
                raise TypeError("each shock must be numeric.")
            if np.isnan(shock):
                raise ValueError("shock values must not be NaN.")