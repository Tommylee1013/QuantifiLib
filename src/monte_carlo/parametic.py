from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

try:
    from sklearn.mixture import GaussianMixture
except ImportError:  # pragma: no cover
    GaussianMixture = None

from src.monte_carlo.base_mc import BaseMonteCarlo

# gaussian monte carlo simulation
class GaussianMonteCarlo(BaseMonteCarlo) :
    """
    Gaussian Monte Carlo return simulator.

    This engine estimates mean and volatility from historical returns
    and generates normally distributed simulated returns.

    Notes
    -----
    For holding_period > 1, the model uses:
    - mean scaling: mu * T
    - volatility scaling: sigma * sqrt(T)
    """

    def simulate_returns(
            self, returns : pd.Series | list[float] | np.ndarray,
            holding_period : int = 1,
            use_mean : bool = True,
            random_seed : int | None = None,
            **kwargs : Any,
        ) -> pd.Series :
        self._validate_holding_period(holding_period)
        returns = self._to_series(returns, name = 'returns', dropna = True)

        mu = float(returns.mean()) if use_mean else 0.0
        sigma = float(returns.std(ddof = 1))

        if sigma < 0 : raise ValueError('Estimated sigma must be non-negative')

        rng = self._get_rng(random_seed = random_seed)

        simulated = rng.normal(
            loc = mu * holding_period,
            scale = sigma * np.sqrt(holding_period),
            size = self.n_simulations
        )

        return pd.Series(
            simulated,
            name = f'{self.name}_simulated_returns',
            dtype = float
        )

class StudentTMonteCarlo(BaseMonteCarlo):
    """
    Student-t Monte Carlo return simulator.

    This engine estimates mean and volatility from historical returns
    and generates fat-tailed simulated returns using the Student-t distribution.

    Parameters
    ----------
    df : int, default 5
        Degrees of freedom of the Student-t distribution.
        Must be greater than 2 for finite variance.

    Notes
    -----
    If X ~ t(df), then Var(X) = df / (df - 2), for df > 2.

    To match the empirical standard deviation sigma, we scale the draws by:

        sigma / sqrt(df / (df - 2))

    For holding_period > 1:
    - mean scaling: mu * T
    - volatility scaling: sigma * sqrt(T)
    """

    @staticmethod
    def _validate_df(df: int) -> None:
        if not isinstance(df, (int, np.integer)):
            raise TypeError("df must be an integer.")
        if int(df) <= 2:
            raise ValueError("df must be greater than 2 for finite variance.")

    def simulate_returns(
        self,
        returns: pd.Series | list[float] | np.ndarray,
        holding_period: int = 1,
        df: int = 5,
        use_mean: bool = True,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        self._validate_holding_period(holding_period)
        self._validate_df(df)

        returns = self._to_series(returns, name="returns", dropna=True)

        mu = float(returns.mean()) if use_mean else 0.0
        sigma = float(returns.std(ddof=1))

        if sigma < 0:
            raise ValueError("Estimated sigma must be non-negative.")

        rng = self._get_rng(random_seed=random_seed)

        raw_t = rng.standard_t(df=df, size=self.n_simulations)

        std_t = np.sqrt(df / (df - 2))
        scaled_t = raw_t / std_t

        simulated = (
            mu * holding_period
            + scaled_t * sigma * np.sqrt(holding_period)
        )

        return pd.Series(
            simulated,
            name=f"{self.name}_simulated_returns",
            dtype=float,
        )

class SkewNormalMonteCarlo(BaseMonteCarlo):
    """
    Skew-normal Monte Carlo simulator.

    If shape is not provided, it is estimated from sample skewness using
    a simple heuristic. This is not exact MLE, but is a reasonable first pass.
    """

    @staticmethod
    def _estimate_shape_from_skewness(sample_skew: float) -> float:
        # Simple bounded heuristic for first implementation
        if np.isnan(sample_skew):
            return 0.0
        return float(np.clip(sample_skew * 4.0, -20.0, 20.0))

    def simulate_returns(
        self,
        returns: pd.Series | list[float] | np.ndarray,
        holding_period: int = 1,
        shape: float | None = None,
        use_mean: bool = True,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        self._validate_holding_period(holding_period)
        returns = self._to_series(returns, name="returns", dropna=True)

        mu = float(returns.mean()) if use_mean else 0.0
        sigma = float(returns.std(ddof=1))
        sample_skew = float(returns.skew())

        a = self._estimate_shape_from_skewness(sample_skew) if shape is None else float(shape)

        rng = self._get_rng(random_seed=random_seed)

        # Standardized draws from skew-normal, then recenter/rescale
        raw = skewnorm.rvs(a=a, size=self.n_simulations, random_state=rng)
        raw_mean = float(np.mean(raw))
        raw_std = float(np.std(raw, ddof=1))

        if raw_std <= 0:
            raise ValueError("Generated skew-normal samples have zero variance.")

        standardized = (raw - raw_mean) / raw_std
        simulated = mu * holding_period + standardized * sigma * np.sqrt(holding_period)

        return pd.Series(
            simulated,
            name=f"{self.name}_simulated_returns",
            dtype=float,
        )

class MixtureGaussianMonteCarlo(BaseMonteCarlo):
    """
    Gaussian mixture Monte Carlo simulator.

    Uses sklearn.mixture.GaussianMixture to fit a mixture distribution
    to historical returns, then samples from the fitted mixture.
    """

    @staticmethod
    def _validate_n_components(n_components: int) -> None:
        if not isinstance(n_components, (int, np.integer)):
            raise TypeError("n_components must be an integer.")
        if int(n_components) < 1:
            raise ValueError("n_components must be >= 1.")

    def simulate_returns(
        self,
        returns: pd.Series | list[float] | np.ndarray,
        holding_period: int = 1,
        n_components: int = 2,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        self._validate_holding_period(holding_period)
        self._validate_n_components(n_components)

        if GaussianMixture is None:
            raise ImportError(
                "MixtureGaussianMonteCarlo requires scikit-learn. "
                "Please install it with `pip install scikit-learn`."
            )

        returns = self._to_series(returns, name="returns", dropna=True)
        values = returns.to_numpy(dtype=float).reshape(-1, 1)

        seed = self.random_seed if random_seed is None else random_seed

        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=seed,
        )
        model.fit(values)

        horizon_samples = []
        for _ in range(holding_period):
            draw, _ = model.sample(self.n_simulations)
            horizon_samples.append(draw.ravel())

        simulated = np.sum(np.column_stack(horizon_samples), axis=1)

        return pd.Series(
            simulated,
            name=f"{self.name}_simulated_returns",
            dtype=float,
        )