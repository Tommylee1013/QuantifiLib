from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

class BasePricer(ABC) :
    """
    Abstract base class for all pricing engines
    """
    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    @abstractmethod
    def price(self, *args: Any, **kwargs: Any) -> float:
        """
        Return the model price.
        """
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
        }