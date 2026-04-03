from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.risk.results.var_results import VaRResult
from src.common.decorators import validate_common_var_inputs
from src.risk.utils.metrics import VaRMetricsHelper
from src.common.transformers import InputTransformer
from src.common.validators import VaRValidator

class BaseVaR(ABC) :
    """
    Base class for Value-at-Risk style risk engines

    This class provides:
    - common validation flow
    - input standardization
    - historical / parametric / stressed VaR
    - generic stress test
    - expected shortfall
    - shared utility methods

    Child classes should implement domain-specific return/PnL preparation logic.
    """
    def __init__(
            self, position_value : float,
            returns : pd.Series | None = None,
            prices : pd.Series | None = None,
            return_method : str = 'simple',
            name : str | None = None,
        ) -> None:
        """
        Initializer for BaseVaR
        """
        self.validator = VaRValidator()
        self.transformer = InputTransformer()
        self.metrics = VaRMetricsHelper()

        self.validator.validate_position_value(position_value)
        self.position_value = float(position_value)
        self.name = name or self.__class__.__name__

        self.prices : pd.Series | None = None
        self.returns : pd.Series | None = None
        self.return_method = return_method

        if prices is not None:
            self.prices = self.transformer.to_series(prices, name="prices", dtype=float)
            if isinstance(prices, pd.Series) and isinstance(prices.index, pd.DatetimeIndex):
                self.prices = self.transformer.ensure_datetime_index(self.prices)

        if returns is not None:
            self.returns = self.transformer.to_series(returns, name="returns", dtype=float)
            if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
                self.returns = self.transformer.ensure_datetime_index(self.returns)

        if self.returns is None and self.prices is not None:
            self.returns = self.transformer.prices_to_returns(
                self.prices,
                method=self.return_method,
                dropna=True,
            )

        if self.returns is not None:
            self.validator.validate_returns(self.returns, name="returns")

    @abstractmethod
    def _prepare_returns(self) -> pd.Series :
        """
        Return a clean return series used for VaR or ES calculations
        Child classes may override this when raw returns need custom construction.
        :return:
        """
        raise NotImplementedError()

    def _get_returns(self) -> pd.Series :
        returns = self._prepare_returns()
        self.validator.validate_returns(returns, name = 'prepared_returns')
        return returns

    def _build_result(
            self, metric_name : str,
            method : str,
            value : float,
            confidence_level : float | None = None,
            holding_period : int | None = None,
            **metadata : Any,
        ) -> VaRResult :
        return VaRResult(
            metric_name = metric_name,
            method = method,
            value = float(value),
            confidence_level = confidence_level,
            holding_period = holding_period,
            position_value = self.position_value,
            metadata = metadata
        )

    def historical_var(
            self, confidence_level : float = 0.95,
            holding_period : int = 1,
            as_result : bool = False
        ) -> float | VaRResult :
        returns = self._get_returns()
        scaled_returns = self.metrics.sqrt_time_scale_returns(returns, holding_period)
        var_return = self.metrics.quantile_return(scaled_returns, confidence_level)
        var_amount = self.metrics.return_to_loss_amount(var_return, self.position_value)

        if as_result:
            return self._build_result(
                metric_name = "VaR",
                method = "historical",
                value = var_amount,
                confidence_level = confidence_level,
                holding_period = holding_period,
            )
        return var_amount

    @validate_common_var_inputs
    def parametric_var(
            self,
            confidence_level: float = 0.95,
            holding_period: int = 1,
            use_mean: bool = True,
            as_result: bool = False,
    ) -> float | VaRResult:
        returns = self._get_returns()
        mu = float(returns.mean()) if use_mean else 0.0
        sigma = float(returns.std(ddof=1))

        z = norm.ppf(1.0 - confidence_level)
        var_return = mu * holding_period + z * sigma * np.sqrt(holding_period)
        var_amount = self.metrics.return_to_loss_amount(var_return, self.position_value)

        if as_result:
            return self._build_result(
                metric_name="VaR",
                method="parametric",
                value=var_amount,
                confidence_level=confidence_level,
                holding_period=holding_period,
                use_mean=use_mean,
            )
        return var_amount

    @validate_common_var_inputs
    def stressed_var(
            self,
            stress_returns: pd.Series,
            confidence_level: float = 0.95,
            holding_period: int = 1,
            as_result: bool = False,
    ) -> float | VaRResult:
        stress_returns = self.transformer.to_series(
            stress_returns,
            name="stress_returns",
            dtype=float,
        )
        self.validator.validate_returns(stress_returns, name="stress_returns")

        scaled_returns = self.metrics.sqrt_time_scale_returns(stress_returns, holding_period)
        var_return = self.metrics.quantile_return(scaled_returns, confidence_level)
        var_amount = self.metrics.return_to_loss_amount(var_return, self.position_value)

        if as_result:
            return self._build_result(
                metric_name="VaR",
                method="stressed",
                value=var_amount,
                confidence_level=confidence_level,
                holding_period=holding_period,
            )
        return var_amount

    def stress_test(
            self,
            shocks: float | list[float] | np.ndarray | pd.Series,
            as_result: bool = False,
    ) -> pd.DataFrame | list[VaRResult]:
        shock_list = self.transformer.scalar_to_list(shocks)
        self.validator.validate_shock_values(shock_list)

        records: list[dict[str, float]] = []
        results: list[VaRResult] = []

        for shock in shock_list:
            pnl = self.metrics.return_to_pnl_amount(shock, self.position_value)
            loss = max(-pnl, 0.0)

            if as_result:
                results.append(
                    self._build_result(
                        metric_name="StressLoss",
                        method="stress_test",
                        value=loss,
                        shock=shock,
                        pnl=pnl,
                    )
                )
            else:
                records.append(
                    {
                        "shock": shock,
                        "pnl": pnl,
                        "loss": loss,
                    }
                )

        if as_result:
            return results

        return pd.DataFrame(records)

    @validate_common_var_inputs
    def expected_shortfall(
            self,
            confidence_level: float = 0.95,
            holding_period: int = 1,
            method: str = "historical",
            use_mean: bool = True,
            stress_returns: pd.Series | None = None,
            as_result: bool = False,
    ) -> float | VaRResult:
        method = method.lower()

        if method == "historical":
            returns = self._get_returns()
            scaled_returns = self.metrics.sqrt_time_scale_returns(returns, holding_period)
            es_return = self.metrics.tail_mean_return(scaled_returns, confidence_level)

        elif method == "parametric":
            returns = self._get_returns()
            mu = float(returns.mean()) if use_mean else 0.0
            sigma = float(returns.std(ddof=1))

            alpha = 1.0 - confidence_level
            z_alpha = norm.ppf(alpha)
            es_return = (
                    mu * holding_period
                    - sigma * np.sqrt(holding_period) * norm.pdf(z_alpha) / alpha
            )

        elif method == "stressed":
            self.validator.validate_required(stress_returns, "stress_returns")
            stress_returns = self.transformer.to_series(
                stress_returns,
                name = "stress_returns",
                dtype = float,
            )
            self.validator.validate_returns(stress_returns, name="stress_returns")
            scaled_returns = self.metrics.sqrt_time_scale_returns(stress_returns, holding_period)
            es_return = self.metrics.tail_mean_return(scaled_returns, confidence_level)

        else:
            raise ValueError(
                "method must be one of: 'historical', 'parametric', 'stressed'."
            )

        es_amount = self.metrics.return_to_loss_amount(es_return, self.position_value)

        if as_result:
            return self._build_result(
                metric_name = "ExpectedShortfall",
                method = method,
                value = es_amount,
                confidence_level = confidence_level,
                holding_period = holding_period,
                use_mean = use_mean,
            )
        return es_amount

    def summary_stats(self) -> dict[str, float]:
        returns = self._get_returns()
        return self.metrics.basic_summary_stats(returns)

    def summary(
            self,
            confidence_level: float = 0.95,
            holding_period: int = 1,
    ) -> dict :
        self.validator.validate_confidence_level(confidence_level)
        self.validator.validate_holding_period(holding_period)

        data = {
            "historical_var": self.historical_var(
                confidence_level = confidence_level,
                holding_period = holding_period,
            ),
            "parametric_var": self.parametric_var(
                confidence_level = confidence_level,
                holding_period = holding_period,
            ),
            "historical_es": self.expected_shortfall(
                confidence_level = confidence_level,
                holding_period = holding_period,
                method = "historical",
            ),
            "parametric_es": self.expected_shortfall(
                confidence_level = confidence_level,
                holding_period = holding_period,
                method = "parametric",
            ),
        }

        return data