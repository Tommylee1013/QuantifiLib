from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.monte_carlo.base_mc import BaseMonteCarlo

class RegimeSwitchingMonteCarlo(BaseMonteCarlo):
    """
    Simple regime-switching Monte Carlo simulator.

    Default design:
    - Split historical returns into low-vol and high-vol regimes using median abs(return)
    - Estimate mean/std within each regime
    - Estimate a 2x2 transition matrix from inferred regime sequence
    - Simulate future returns by evolving the latent regime

    This is a lightweight first implementation, not a full HMM.
    """

    @staticmethod
    def _infer_regimes(values: np.ndarray) -> np.ndarray:
        threshold = np.median(np.abs(values))
        return (np.abs(values) > threshold).astype(int)

    @staticmethod
    def _estimate_transition_matrix(regimes: np.ndarray) -> np.ndarray:
        counts = np.ones((2, 2), dtype=float)  # Laplace smoothing
        for i in range(len(regimes) - 1):
            counts[regimes[i], regimes[i + 1]] += 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
        return counts / row_sums

    def simulate_returns(
        self,
        returns: pd.Series | list[float] | np.ndarray,
        holding_period: int = 1,
        random_seed: int | None = None,
        regime_means: list[float] | np.ndarray | None = None,
        regime_stds: list[float] | np.ndarray | None = None,
        transition_matrix: np.ndarray | None = None,
        initial_regime: int | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        self._validate_holding_period(holding_period)
        returns = self._to_series(returns, name="returns", dropna=True)
        values = returns.to_numpy(dtype=float)

        rng = self._get_rng(random_seed=random_seed)

        if regime_means is None or regime_stds is None or transition_matrix is None:
            inferred_regimes = self._infer_regimes(values)

            low_values = values[inferred_regimes == 0]
            high_values = values[inferred_regimes == 1]

            if len(low_values) < 2 or len(high_values) < 2:
                raise ValueError("Not enough observations to estimate two regimes.")

            means = np.array(
                [low_values.mean(), high_values.mean()],
                dtype=float,
            )
            stds = np.array(
                [low_values.std(ddof=1), high_values.std(ddof=1)],
                dtype=float,
            )
            trans = self._estimate_transition_matrix(inferred_regimes)

            if initial_regime is None:
                current_regime = int(inferred_regimes[-1])
            else:
                current_regime = int(initial_regime)
        else:
            means = np.asarray(regime_means, dtype=float)
            stds = np.asarray(regime_stds, dtype=float)
            trans = np.asarray(transition_matrix, dtype=float)

            if means.shape != (2,) or stds.shape != (2,) or trans.shape != (2, 2):
                raise ValueError("This first implementation expects exactly 2 regimes.")
            current_regime = 0 if initial_regime is None else int(initial_regime)

        simulated = np.empty(self.n_simulations, dtype=float)

        for i in range(self.n_simulations):
            regime = current_regime
            cum_ret = 0.0

            for _ in range(holding_period):
                mu = means[regime]
                sigma = max(stds[regime], 1e-12)

                r_t = rng.normal(loc=mu, scale=sigma)
                cum_ret += r_t

                regime = rng.choice([0, 1], p=trans[regime])

            simulated[i] = cum_ret

        return pd.Series(
            simulated,
            name=f"{self.name}_simulated_returns",
            dtype=float,
        )