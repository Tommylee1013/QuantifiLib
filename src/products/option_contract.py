from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(slots=True)
class OptionContract:
    """
    Plain option contract container.

    Parameters
    ----------
    underlying : str
        Underlying asset ticker or identifier.
    option_type : str
        "call" or "put".
    strike : float
        Strike price.
    maturity_date : pd.Timestamp | str
        Option maturity date.
    contracts : float, default 1.0
        Number of option contracts.
    multiplier : float, default 100.0
        Contract multiplier.
    style : str, default "european"
        Option exercise style.
    """

    underlying: str
    option_type: str
    strike: float
    maturity_date: pd.Timestamp | str
    contracts: float = 1.0
    multiplier: float = 100.0
    style: str = "european"

    def __post_init__(self) -> None:
        self.option_type = str(self.option_type).lower()
        self.style = str(self.style).lower()
        self.maturity_date = pd.Timestamp(self.maturity_date)

        self._validate_option_type()
        self._validate_style()
        self._validate_numeric_fields()

    def _validate_option_type(self) -> None:
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be either 'call' or 'put'.")

    def _validate_style(self) -> None:
        if self.style not in {"european", "american"}:
            raise ValueError("style must be either 'european' or 'american'.")

    def _validate_numeric_fields(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive.")
        if self.contracts <= 0:
            raise ValueError("contracts must be positive.")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive.")

    @property
    def notional_scale(self) -> float:
        """
        Position scale = number of contracts × contract multiplier.
        """
        return float(self.contracts) * float(self.multiplier)

    def time_to_maturity(
        self,
        valuation_date: pd.Timestamp | str | None,
        day_count: float = 365.25,
    ) -> float:
        """
        Return time to maturity in years.

        Parameters
        ----------
        valuation_date : pd.Timestamp | str | None
            Valuation date used to compute remaining maturity.
        day_count : float, default 365.25
            Day-count denominator.

        Returns
        -------
        float
            Non-negative year fraction to maturity.
        """
        if valuation_date is None:
            raise ValueError("valuation_date must be provided.")

        valuation_date = pd.Timestamp(valuation_date)
        delta_days = (self.maturity_date - valuation_date).days
        return max(float(delta_days) / float(day_count), 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "underlying": self.underlying,
            "option_type": self.option_type,
            "strike": self.strike,
            "maturity_date": self.maturity_date,
            "contracts": self.contracts,
            "multiplier": self.multiplier,
            "style": self.style,
        }