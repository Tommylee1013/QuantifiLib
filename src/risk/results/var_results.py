from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class VaRResult:
    metric_name: str
    method: str
    value: float
    confidence_level: float | None = None
    holding_period: int | None = None
    position_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "method": self.method,
            "value": self.value,
            "confidence_level": self.confidence_level,
            "holding_period": self.holding_period,
            "position_value": self.position_value,
            "metadata": self.metadata,
        }