from __future__ import annotations
from typing import Any
import pandas as pd

from .base_var import BaseVaR

class StockVaR(BaseVaR):
    """
    VaR engine for a single stock or ETF position.

    Parameters
    ----------
    prices : pd.Series | None
        Price series of the asset.
    returns : pd.Series | None
        Return series of the asset.
    position_value : float | None
        Current market value of the position.
    shares : float | None
        Number of shares held.
    current_price : float | None
        Current asset price. Used with shares when position_value is not provided.
    return_method : str, default "simple"
        Return calculation method when prices are provided.
        One of {"simple", "log"}.
    name : str | None
        Optional instance name.
    ticker : str | None
        Optional ticker symbol metadata.
    """
    def __init__(
            self, prices : pd.Series | None = None,
            returns : pd.Series | None = None,
            position_value : float | None = None,
            shares : float | None = None,
            current_price : float | None = None,
            return_method : str = "simple",
            name : str | None = None,
            ticker : str | None = None
        ) -> None :
        if prices is None and returns is None :
            raise ValueError("Either prices or returns must be provided")

        inferred_position_value = self._resolve_position_value(
            prices = prices,
            position_value = position_value,
            shares = shares,
            current_price = current_price,
        )

        self.ticker = ticker
        self.shares = float(shares) if shares is not None else None
        self.current_price = float(current_price) if current_price is not None else None

        super().__init__(
            position_value = inferred_position_value,
            returns = returns,
            prices = prices,
            return_method = return_method,
            name = name or (ticker if ticker is not None else self.__class__.__name__),
        )

    @staticmethod
    def _resolve_position_value(
            prices : pd.Series | None,
            position_value : float | None,
            shares : float | None,
            current_price : float | None,
        ) -> float :
        """
        Resolve position value using the following priority:

        1. explicit position_value
        2. shares * current_price
        3. shares * last_price_from_prices
        """
        if position_value is not None:
            return float(position_value)

        if shares is not None and current_price is not None:
            return float(shares) * float(current_price)

        if shares is not None and prices is not None:
            price_series = pd.Series(prices).dropna()
            if price_series.empty:
                raise ValueError("prices must contain at least one non-null observation.")
            last_price = float(price_series.iloc[-1])
            return float(shares) * last_price

        raise ValueError(
            "position_value is required unless it can be inferred from "
            "shares * current_price or shares * last price from prices."
        )

    def _prepare_returns(self) -> pd.Series:
        """
        Prepare clean return series for VaR calculations.
        """
        if self.returns is None:
            raise ValueError("Return series is not available.")

        returns = self.returns.dropna()

        if returns.empty:
            raise ValueError("Prepared returns are empty.")

        return returns.astype(float)

    @classmethod
    def from_prices(
            cls, prices : pd.Series,
            position_value : float | None = None,
            shares : float | None = None,
            current_price : float | None = None,
            return_method : str = "simple",
            name : str | None = None,
            ticker : str | None = None
        ) -> "StockVaR" :
        """
        Alternate constructor using price series
        """
        return cls(
            prices = prices,
            returns = None,
            position_value = position_value,
            shares = shares,
            current_price = current_price,
            return_method = return_method,
            name = name,
            ticker = ticker
        )

    @classmethod
    def from_returns(
            cls,
            returns: pd.Series,
            position_value: float | None = None,
            shares: float | None = None,
            current_price: float | None = None,
            name: str | None = None,
            ticker: str | None = None,
    ) -> "StockVaR":
        """
        Alternate constructor using return series.
        """
        return cls(
            prices=None,
            returns=returns,
            position_value=position_value,
            shares=shares,
            current_price=current_price,
            return_method="simple",
            name=name,
            ticker=ticker,
        )

    def get_position_info(self) -> dict[str, Any]:
        """
        Return simple metadata for the current stock position.
        """
        return {
            "name": self.name,
            "ticker": self.ticker,
            "shares": self.shares,
            "current_price": self.current_price,
            "position_value": self.position_value,
            "return_method": self.return_method,
            "n_returns": 0 if self.returns is None else int(self.returns.dropna().shape[0]),
            "n_prices": 0 if self.prices is None else int(self.prices.dropna().shape[0]),
        }