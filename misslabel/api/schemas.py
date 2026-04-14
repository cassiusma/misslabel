"""
misslabel.api.schemas
---------------------
Pydantic models for POST /summary request and response.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

Label = Union[str, int]


class SummaryRequest(BaseModel):
    """Body for POST /summary."""

    y_true: List[Label]
    y_noisy: List[Label]
    pi: Optional[List[float]] = None
    t: Optional[float] = None

    @field_validator("y_true", "y_noisy")
    @classmethod
    def non_empty(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("Label list must not be empty.")
        return v

    @model_validator(mode="after")
    def same_length(self) -> "SummaryRequest":
        if len(self.y_true) != len(self.y_noisy):
            raise ValueError(
                f"y_true and y_noisy must have the same length "
                f"(got {len(self.y_true)} vs {len(self.y_noisy)})."
            )
        return self

    @field_validator("pi")
    @classmethod
    def pi_sums_to_one(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if any(x < 0 for x in v):
            raise ValueError("All pi values must be non-negative.")
        total = sum(v)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"pi must sum to 1.0 (got {total:.6f}).")
        return v

    @field_validator("t")
    @classmethod
    def t_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("t must be >= 0.")
        return v


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class PerClassStats(BaseModel):
    n: int
    n_flipped: int
    flip_rate: float
    flip_rate_theoretical: Optional[float]


class SummaryResponse(BaseModel):
    n: int
    k: int
    n_flipped: int
    flip_rate: float
    per_class: Dict[str, PerClassStats]   # keys are always strings for JSON
    T_hat: List[List[float]]
    P_t: Optional[List[List[float]]]
    max_T_error: Optional[float]
