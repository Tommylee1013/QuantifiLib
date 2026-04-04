from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(slots=True)
class MarketInputs:
    """
    Market input container for pricing and risk calculations.

    Parameters
    ----------
    spot : float
        Current underlying spot price.
    volatility : float
        Annualized volatility as a decimal (e.g. 0.2 for 20%).
    rate : float
        Risk-free rate as a decimal.
    dividend_yield : float, default 0.0
        Continuous dividend yield as a decimal.
    valuation_date : pd.Timestamp | str | None, default None
        Valuation date.
    """

    spot: float
    volatility: float
    rate: float
    dividend_yield: float = 0.0
    valuation_date: pd.Timestamp | str | None = None

    def __post_init__(self) -> None:
        if self.valuation_date is not None:
            self.valuation_date = pd.Timestamp(self.valuation_date)

        self._validate_numeric_fields()

    def _validate_numeric_fields(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive.")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative.")

    def shocked(
        self,
        spot: float | None = None,
        volatility: float | None = None,
        rate: float | None = None,
        dividend_yield: float | None = None,
        valuation_date: pd.Timestamp | str | None = None,
    ) -> "MarketInputs":
        """
        Return a new MarketInputs instance with shocked/replaced values.
        """
        return MarketInputs(
            spot=self.spot if spot is None else float(spot),
            volatility=self.volatility if volatility is None else float(volatility),
            rate=self.rate if rate is None else float(rate),
            dividend_yield=self.dividend_yield if dividend_yield is None else float(dividend_yield),
            valuation_date=self.valuation_date if valuation_date is None else valuation_date,
        )

    def apply_relative_spot_shock(self, shock: float) -> "MarketInputs":
        """
        Apply relative shock to spot.
        Example: shock=-0.1 means spot becomes spot * 0.9
        """
        new_spot = self.spot * (1.0 + float(shock))
        if new_spot <= 0:
            raise ValueError("Shocked spot must remain positive.")
        return self.shocked(spot=new_spot)

    def apply_vol_shock(self, shock: float, relative: bool = True) -> "MarketInputs":
        """
        Apply shock to volatility.

        Parameters
        ----------
        shock : float
            If relative=True, new_vol = vol * (1 + shock)
            If relative=False, new_vol = vol + shock
        relative : bool, default True
            Whether the shock is proportional or absolute.
        """
        if relative:
            new_vol = self.volatility * (1.0 + float(shock))
        else:
            new_vol = self.volatility + float(shock)

        if new_vol < 0:
            raise ValueError("Shocked volatility must be non-negative.")

        return self.shocked(volatility=new_vol)

    def apply_rate_shock(self, shock: float, relative: bool = False) -> "MarketInputs":
        """
        Apply shock to rate.

        Parameters
        ----------
        shock : float
            If relative=True, new_rate = rate * (1 + shock)
            If relative=False, new_rate = rate + shock
        relative : bool, default False
            Rates are usually shocked additively, so default is False.
        """
        if relative:
            new_rate = self.rate * (1.0 + float(shock))
        else:
            new_rate = self.rate + float(shock)

        return self.shocked(rate=new_rate)

    def to_dict(self) -> dict[str, object]:
        return {
            "spot": self.spot,
            "volatility": self.volatility,
            "rate": self.rate,
            "dividend_yield": self.dividend_yield,
            "valuation_date": self.valuation_date,
        }