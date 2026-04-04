from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm

from src.pricing.option_pricer import OptionPricer


class BlackScholesPricer(OptionPricer):
    """
    Black-Scholes option pricer for European call/put options.

    The pricer expects:
    - contract.option_type: "call" or "put"
    - contract.strike
    - contract.time_to_maturity(market.valuation_date) or contract.maturity_date
    - contract.contracts
    - contract.multiplier

    and
    - market.spot
    - market.volatility
    - market.rate
    - market.dividend_yield
    - market.valuation_date
    """

    @staticmethod
    def _validate_inputs(
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
    ) -> None:
        if spot <= 0:
            raise ValueError("spot must be positive.")
        if strike <= 0:
            raise ValueError("strike must be positive.")
        if time_to_maturity < 0:
            raise ValueError("time_to_maturity must be non-negative.")
        if volatility < 0:
            raise ValueError("volatility must be non-negative.")

    @staticmethod
    def _extract_time_to_maturity(contract: Any, market_inputs: Any) -> float:
        if hasattr(contract, "time_to_maturity"):
            return float(contract.time_to_maturity(market_inputs.valuation_date))

        if hasattr(contract, "maturity_date") and hasattr(market_inputs, "valuation_date"):
            maturity_date = contract.maturity_date
            valuation_date = market_inputs.valuation_date

            if valuation_date is None:
                raise ValueError(
                    "market.valuation_date is required when contract.time_to_maturity is not implemented."
                )

            delta_days = (maturity_date - valuation_date).days
            return max(delta_days / 365.25, 0.0)

        raise ValueError(
            "Could not determine time to maturity. "
            "Provide contract.time_to_maturity(...) or both contract.maturity_date "
            "and market.valuation_date."
        )

    @staticmethod
    def _d1_d2(
        spot: float,
        strike: float,
        time_to_maturity: float,
        rate: float,
        dividend_yield: float,
        volatility: float,
    ) -> tuple[float, float]:
        if time_to_maturity == 0 or volatility == 0:
            # Avoid division by zero; values won't be used in standard way for expiry handling
            return float("inf"), float("inf")

        sqrt_t = math.sqrt(time_to_maturity)
        d1 = (
            math.log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity
        ) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        return d1, d2

    @staticmethod
    def _intrinsic_value(option_type: str, spot: float, strike: float) -> float:
        option_type = option_type.lower()
        if option_type == "call":
            return max(spot - strike, 0.0)
        if option_type == "put":
            return max(strike - spot, 0.0)
        raise ValueError("option_type must be either 'call' or 'put'.")

    @staticmethod
    def _position_scale(contract: Any) -> float:
        contracts = float(getattr(contract, "contracts", 1.0))
        multiplier = float(getattr(contract, "multiplier", 1.0))
        return contracts * multiplier

    def price(self, contract: Any, market_inputs: Any) -> float:
        option_type = str(contract.option_type).lower()
        spot = float(market_inputs.spot)
        strike = float(contract.strike)
        rate = float(market_inputs.rate)
        dividend_yield = float(getattr(market_inputs, "dividend_yield", 0.0))
        volatility = float(market_inputs.volatility)
        t = self._extract_time_to_maturity(contract, market_inputs)

        self._validate_inputs(
            spot=spot,
            strike=strike,
            time_to_maturity=t,
            volatility=volatility,
        )

        scale = self._position_scale(contract)

        if t == 0 or volatility == 0:
            return self._intrinsic_value(option_type, spot, strike) * scale

        d1, d2 = self._d1_d2(
            spot=spot,
            strike=strike,
            time_to_maturity=t,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )

        disc_q = math.exp(-dividend_yield * t)
        disc_r = math.exp(-rate * t)

        if option_type == "call":
            value = spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
        elif option_type == "put":
            value = strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be either 'call' or 'put'.")

        return value * scale

    def greeks(self, contract: Any, market_inputs: Any) -> dict[str, float]:
        option_type = str(contract.option_type).lower()
        spot = float(market_inputs.spot)
        strike = float(contract.strike)
        rate = float(market_inputs.rate)
        dividend_yield = float(getattr(market_inputs, "dividend_yield", 0.0))
        volatility = float(market_inputs.volatility)
        t = self._extract_time_to_maturity(contract, market_inputs)

        self._validate_inputs(
            spot=spot,
            strike=strike,
            time_to_maturity=t,
            volatility=volatility,
        )

        scale = self._position_scale(contract)

        if t == 0 or volatility == 0:
            intrinsic = self._intrinsic_value(option_type, spot, strike)
            delta = 1.0 if (option_type == "call" and spot > strike) else 0.0
            if option_type == "put":
                delta = -1.0 if spot < strike else 0.0

            return {
                "price": intrinsic * scale,
                "delta": delta * scale,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }

        d1, d2 = self._d1_d2(
            spot=spot,
            strike=strike,
            time_to_maturity=t,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )

        sqrt_t = math.sqrt(t)
        disc_q = math.exp(-dividend_yield * t)
        disc_r = math.exp(-rate * t)
        pdf_d1 = norm.pdf(d1)

        if option_type == "call":
            delta = disc_q * norm.cdf(d1)
            theta = (
                -spot * disc_q * pdf_d1 * volatility / (2 * sqrt_t)
                - rate * strike * disc_r * norm.cdf(d2)
                + dividend_yield * spot * disc_q * norm.cdf(d1)
            )
            rho = strike * t * disc_r * norm.cdf(d2)
        elif option_type == "put":
            delta = disc_q * (norm.cdf(d1) - 1.0)
            theta = (
                -spot * disc_q * pdf_d1 * volatility / (2 * sqrt_t)
                + rate * strike * disc_r * norm.cdf(-d2)
                - dividend_yield * spot * disc_q * norm.cdf(-d1)
            )
            rho = -strike * t * disc_r * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be either 'call' or 'put'.")

        gamma = disc_q * pdf_d1 / (spot * volatility * sqrt_t)
        vega = spot * disc_q * pdf_d1 * sqrt_t

        price_value = self.price(contract, market_inputs)

        return {
            "price": price_value,
            "delta": delta * scale,
            "gamma": gamma * scale,
            "vega": vega * scale,
            "theta": theta * scale,
            "rho": rho * scale,
        }