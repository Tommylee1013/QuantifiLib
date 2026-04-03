from __future__ import annotations

import inspect
from functools import wraps

def validate_common_var_inputs(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        signature = inspect.signature(func)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()

        arguments = bound.arguments

        confidence_level = arguments.get("confidence_level", None)
        holding_period = arguments.get("holding_period", None)
        position_value = arguments.get("position_value", None)
        n_simulations = arguments.get("n_simulations", None)

        if confidence_level is not None:
            self.validator.validate_confidence_level(confidence_level)

        if holding_period is not None:
            self.validator.validate_holding_period(holding_period)

        if position_value is not None:
            self.validator.validate_position_value(position_value)

        if n_simulations is not None:
            self.validator.validate_n_simulations(n_simulations)

        return func(*bound.args, **bound.kwargs)

    return wrapper