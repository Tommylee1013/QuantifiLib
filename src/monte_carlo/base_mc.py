from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

class BaseMonteCarlo(ABC) :
    """
    Abstract base class for Monte Carlo simulation engines.

    This class is intertionally generic and should not contain
    domain-specific logic such as VaR, Expected Shortfall, pricing,
    or portfolio optimization.

    Child classes are responsible for implementing `simulated_returns`
    and returning simulated return samples as a pandas Series or
    pandas DataFrame, depending on the use case.
    """

    def __init__(
            self, n_simulations : int = 10000,
            random_seed : int | None = None,
            name : str | None = None,
        ) -> None:
        self._validate_n_simulations(n_simulations)

        self.n_simulations = int(n_simulations)
        self.random_seed = random_seed
        self.name = name or self.__class__.__name__

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_simulations={self.n_simulations}, "
            f"random_seed={self.random_seed}, "
            f"name='{self.name}')"
        )

    @staticmethod
    def _validate_n_simulations(n_simulations: int) -> None:
        if not isinstance(n_simulations, (int, np.integer)):
            raise TypeError("n_simulations must be an integer.")
        if int(n_simulations) < 1:
            raise ValueError("n_simulations must be greater than or equal to 1.")

    @staticmethod
    def _validate_holding_period(holding_period: int) -> None:
        if not isinstance(holding_period, (int, np.integer)):
            raise TypeError("holding_period must be an integer.")
        if int(holding_period) < 1:
            raise ValueError("holding_period must be greater than or equal to 1.")

    @staticmethod
    def _to_series(
            returns: pd.Series | list[float] | np.ndarray,
            name: str = "returns",
            dropna: bool = True,
    ) -> pd.Series:
        if isinstance(returns, pd.Series):
            series = returns.copy()
        elif isinstance(returns, (list, tuple, np.ndarray)):
            series = pd.Series(returns, name = name, dtype = float)
        else:
            raise TypeError(
                f"{name} must be a pandas Series, list, tuple, or numpy ndarray."
            )

        series = series.astype(float)

        if dropna:
            series = series.dropna()

        if series.empty:
            raise ValueError(f"{name} must contain at least one valid observation.")

        if np.isinf(series.to_numpy()).any():
            raise ValueError(f"{name} must not contain infinite values.")

        return series

    def _get_rng(self, random_seed: int | None = None) -> np.random.Generator:
        seed = self.random_seed if random_seed is None else random_seed
        return np.random.default_rng(seed)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "random_seed": self.random_seed,
            "name": self.name,
        }

    @abstractmethod
    def simulate_returns(
            self,
            returns: pd.Series | list[float] | np.ndarray,
            holding_period: int = 1,
            **kwargs: Any,
    ) -> pd.Series | pd.DataFrame:
        """
        Simulate return samples from the fitted or implied return distribution.

        Parameters
        ----------
        returns : pd.Series | list[float] | np.ndarray
            Historical return observations used to estimate simulation inputs.
        holding_period : int, default 1
            Forecast horizon used for simulation scaling.
        **kwargs : Any
            Engine-specific parameters.

        Returns
        -------
        pd.Series | pd.DataFrame
            Simulated return samples.
            - For single-asset one-period engines, pd.Series is recommended.
            - For multi-asset or path-based engines, pd.DataFrame may be used.
        """
        raise NotImplementedError("Child classes must implement simulate_returns().")