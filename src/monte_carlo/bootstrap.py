from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.monte_carlo.base_mc import BaseMonteCarlo

class BootstrapMonteCarlo(BaseMonteCarlo):
    """
    IID bootstrap Monte Carlo simulator.

    Simulates future returns by sampling historical returns with replacement.
    For holding_period > 1, it samples (n_simulations x holding_period) returns
    and sums them as an approximation of cumulative arithmetic returns.
    """

    def simulate_returns(
        self,
        returns: pd.Series | list[float] | np.ndarray,
        holding_period: int = 1,
        random_seed: int | None = None,
        replace: bool = True,
        **kwargs: Any,
    ) -> pd.Series:
        self._validate_holding_period(holding_period)
        returns = self._to_series(returns, name="returns", dropna=True)

        values = returns.to_numpy(dtype=float)
        rng = self._get_rng(random_seed=random_seed)

        sampled = rng.choice(
            values,
            size=(self.n_simulations, holding_period),
            replace=replace,
        )

        simulated = sampled.sum(axis=1)

        return pd.Series(
            simulated,
            name=f"{self.name}_simulated_returns",
            dtype=float,
        )

class BlockBootstrapMonteCarlo(BaseMonteCarlo):
    """
    Block bootstrap Monte Carlo simulator.

    Preserves some serial dependence by sampling contiguous return blocks.
    """

    @staticmethod
    def _validate_block_size(block_size: int) -> None:
        if not isinstance(block_size, (int, np.integer)):
            raise TypeError("block_size must be an integer.")
        if int(block_size) < 1:
            raise ValueError("block_size must be greater than or equal to 1.")

    def simulate_returns(
        self,
        returns: pd.Series | list[float] | np.ndarray,
        holding_period: int = 1,
        block_size: int = 5,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        self._validate_holding_period(holding_period)
        self._validate_block_size(block_size)

        returns = self._to_series(returns, name="returns", dropna=True)
        values = returns.to_numpy(dtype=float)
        n_obs = len(values)

        if n_obs < block_size:
            raise ValueError("Length of returns must be >= block_size.")

        rng = self._get_rng(random_seed=random_seed)

        simulated = np.empty(self.n_simulations, dtype=float)

        max_start = n_obs - block_size + 1

        for i in range(self.n_simulations):
            path = []
            while len(path) < holding_period:
                start = rng.integers(0, max_start)
                block = values[start : start + block_size]
                path.extend(block.tolist())

            simulated[i] = np.sum(path[:holding_period])

        return pd.Series(
            simulated,
            name=f"{self.name}_simulated_returns",
            dtype=float,
        )