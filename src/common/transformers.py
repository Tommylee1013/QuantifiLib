from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

class InputTransformer(object):
    @staticmethod
    def to_series(
        data: Any,
        name: str = "series",
        dropna: bool = True,
        dtype: str | type | None = float,
    ) -> pd.Series:
        if isinstance(data, pd.Series):
            series = data.copy()
        elif isinstance(data, (list, tuple, np.ndarray)):
            series = pd.Series(data, name=name)
        else:
            raise TypeError(f"{name} must be convertible to pandas Series.")

        if dtype is not None:
            series = series.astype(dtype)

        if dropna:
            series = series.dropna()

        series.name = name
        return series

    @staticmethod
    def to_dataframe(
        data: Any,
        name: str = "dataframe",
        dropna: bool = False,
        dtype: str | type | None = float,
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            raise TypeError(f"{name} must be convertible to pandas DataFrame.")

        if dtype is not None:
            df = df.astype(dtype)

        if dropna:
            df = df.dropna()

        return df

    @staticmethod
    def ensure_datetime_index(
        obj: pd.Series | pd.DataFrame,
        sort_index: bool = True,
    ) -> pd.Series | pd.DataFrame:
        result = obj.copy()
        result.index = pd.to_datetime(result.index)

        if sort_index:
            result = result.sort_index()

        return result

    @staticmethod
    def prices_to_returns(
        prices: pd.Series,
        method: str = "simple",
        dropna: bool = True,
    ) -> pd.Series:
        prices = InputTransformer.to_series(prices, name="prices", dropna=True, dtype=float)

        if len(prices) < 2:
            raise ValueError("prices must contain at least two observations.")

        method = method.lower()

        if method == "simple":
            returns = prices.pct_change()
        elif method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("method must be either 'simple' or 'log'.")

        if dropna:
            returns = returns.dropna()

        returns.name = "returns"
        return returns

    @staticmethod
    def align_series(
        left: pd.Series,
        right: pd.Series,
        join: str = "inner",
    ) -> tuple[pd.Series, pd.Series]:
        left_aligned, right_aligned = left.align(right, join=join)
        return left_aligned, right_aligned

    @staticmethod
    def align_dataframe_and_series(
        df: pd.DataFrame,
        series: pd.Series,
        join: str = "inner",
    ) -> tuple[pd.DataFrame, pd.Series]:
        df_aligned, series_aligned = df.align(series, axis=0, join=join)
        return df_aligned, series_aligned

    @staticmethod
    def scalar_to_list(value: float | list[float] | np.ndarray | pd.Series) -> list[float]:
        if np.isscalar(value):
            return [float(value)]
        return [float(x) for x in pd.Series(value).tolist()]