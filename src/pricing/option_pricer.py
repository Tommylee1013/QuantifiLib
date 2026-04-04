from __future__ import annotations

from abc import abstractmethod
from typing import Any

from src.pricing.base_pricer import BasePricer


class OptionPricer(BasePricer):
    """
    Abstract base class for option pricing engines.

    Expected inputs:
    - contract: option contract object
    - market: market input object
    """

    @abstractmethod
    def price(self, contract: Any, market_inputs: Any) -> float:
        raise NotImplementedError

    @abstractmethod
    def greeks(self, contract: Any, market_inputs: Any) -> dict[str, float]:
        raise NotImplementedError