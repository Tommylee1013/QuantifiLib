from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.monte_carlo.base_mc import BaseMonteCarlo

try:
    from arch import arch_model
except ImportError:  # pragma: no cover
    arch_model = None

class GARCHMonteCarlo(BaseMonteCarlo):
    """
    GARCH-based Monte Carlo simulator.

    Fits a GARCH(p, q) model using the `arch` package, then simulates
    future returns recursively.
    """

    @staticmethod
    def _validate_order(p: int, q: int) -> None:
        if not isinstance(p, (int, np.integer)) or not isinstance(q, (int, np.integer)):
            raise TypeError("p and q must be integers.")
        if p < 1 or q < 1:
            raise ValueError("p and q must be >= 1.")

    def simulate_returns(
        self,
        returns: pd.Series | list[float] | np.ndarray,
        holding_period: int = 1,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "constant",
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> pd.Series:
        self._validate_holding_period(holding_period)
        self._validate_order(p, q)

        if arch_model is None:
            raise ImportError(
                "GARCHMonteCarlo requires the `arch` package. "
                "Please install it with `pip install arch`."
            )

        returns = self._to_series(returns, name="returns", dropna=True)
        values = returns.to_numpy(dtype=float)

        # arch package convention often works better with percentage returns
        scaled_values = values * 100.0

        model = arch_model(
            scaled_values,
            mean=mean,
            vol="GARCH",
            p=p,
            q=q,
            dist=dist,
        )
        fitted = model.fit(disp="off")

        params = fitted.params
        mu = float(params.get("mu", 0.0))

        omega = float(params["omega"])
        alpha_keys = [k for k in params.index if k.startswith("alpha[")]
        beta_keys = [k for k in params.index if k.startswith("beta[")]

        alpha = np.array([float(params[k]) for k in alpha_keys], dtype=float)
        beta = np.array([float(params[k]) for k in beta_keys], dtype=float)

        resids = np.asarray(fitted.resid, dtype=float)
        cond_vol = np.asarray(fitted.conditional_volatility, dtype=float)

        last_eps = np.square(resids[-p:][::-1]) if p > 0 else np.array([], dtype=float)
        last_sig2 = np.square(cond_vol[-q:][::-1]) if q > 0 else np.array([], dtype=float)

        rng = self._get_rng(random_seed=random_seed)
        simulated = np.empty(self.n_simulations, dtype=float)

        for i in range(self.n_simulations):
            eps_hist = last_eps.copy()
            sig2_hist = last_sig2.copy()

            cum_ret = 0.0
            for _ in range(holding_period):
                sig2_t = omega

                if p > 0:
                    sig2_t += np.dot(alpha, eps_hist[:p])

                if q > 0:
                    sig2_t += np.dot(beta, sig2_hist[:q])

                sig2_t = max(sig2_t, 1e-12)
                z_t = rng.normal()
                eps_t = np.sqrt(sig2_t) * z_t
                r_t = mu + eps_t
                cum_ret += r_t

                if p > 0:
                    eps_hist = np.insert(eps_hist, 0, eps_t**2)[:p]
                if q > 0:
                    sig2_hist = np.insert(sig2_hist, 0, sig2_t)[:q]

            simulated[i] = cum_ret / 100.0

        return pd.Series(
            simulated,
            name=f"{self.name}_simulated_returns",
            dtype=float,
        )