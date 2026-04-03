from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.monte_carlo.base_mc import BaseMonteCarlo

class KernelDensityMonteCarlo(BaseMonteCarlo):
    """
    KDE-based Monte Carlo simulator.

    Estimates the empirical return distribution using gaussian_kde
    and resamples from it.
    """

    def simulate_returns(
        self,
        returns: pd.Series | list[float] | np.ndarray,
        holding_period: int = 1,
        bandwidth: float | None = None,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        self._validate_holding_period(holding_period)
        returns = self._to_series(returns, name="returns", dropna=True)

        values = returns.to_numpy(dtype=float)
        kde = gaussian_kde(values)

        if bandwidth is not None:
            if bandwidth <= 0:
                raise ValueError("bandwidth must be positive.")
            kde.set_bandwidth(bw_method=bandwidth)

        seed = self.random_seed if random_seed is None else random_seed
        np.random.seed(seed)

        horizon_samples = []
        for _ in range(holding_period):
            draw = kde.resample(self.n_simulations).ravel()
            horizon_samples.append(draw)

        simulated = np.sum(np.column_stack(horizon_samples), axis=1)

        return pd.Series(
            simulated,
            name=f"{self.name}_simulated_returns",
            dtype=float,
        )