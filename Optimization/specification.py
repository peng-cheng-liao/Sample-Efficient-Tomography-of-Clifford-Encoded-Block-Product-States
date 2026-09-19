"""Validated objective specifications for reusable CEBP optimization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Union


class OptimizationMode(str, Enum):
    """Supported constrained empirical optimization objectives."""

    FIXED_BUDGET_MIN_ERROR = "fixed_budget_min_error"
    FIXED_ERROR_MIN_COPIES = "fixed_error_min_copies"


@dataclass(frozen=True)
class OptimizationObjective:
    """One explicit constrained objective.

    ``copy_ceiling`` is the physical budget in fixed-budget mode.  It remains
    deprecated, inert compatibility metadata in modern fixed-error mode.
    ``scientific_copy_cap`` is a distinct fixed-error search-domain boundary;
    it is never interpreted as a computational or memory-safety limit.
    """

    mode: Union[OptimizationMode, str] = OptimizationMode.FIXED_BUDGET_MIN_ERROR
    copy_ceiling: Optional[int] = 500_000
    error_target: Optional[float] = None
    budget_utilization_target: float = 0.99
    error_target_margin: float = 0.0
    success_probability_threshold: Optional[float] = None
    scientific_copy_cap: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            normalized = OptimizationMode(self.mode)
        except ValueError as error:
            raise ValueError(f"Unsupported optimization mode: {self.mode!r}.") from error
        object.__setattr__(self, "mode", normalized)
        if normalized is OptimizationMode.FIXED_BUDGET_MIN_ERROR:
            if isinstance(self.copy_ceiling, bool) or not isinstance(
                self.copy_ceiling, int
            ) or self.copy_ceiling <= 0:
                raise ValueError(
                    "fixed_budget_min_error requires a positive copy_ceiling."
                )
        elif self.copy_ceiling is not None and (
            isinstance(self.copy_ceiling, bool)
            or not isinstance(self.copy_ceiling, int)
            or self.copy_ceiling <= 0
        ):
            raise ValueError(
                "Legacy fixed-error copy_ceiling metadata must be positive or None."
            )
        if not 0.0 < float(self.budget_utilization_target) <= 1.0:
            raise ValueError("budget_utilization_target must lie in (0,1].")
        if not math.isfinite(float(self.error_target_margin)) or self.error_target_margin < 0.0:
            raise ValueError("error_target_margin must be finite and nonnegative.")
        if normalized is OptimizationMode.FIXED_ERROR_MIN_COPIES:
            if self.error_target is None or not math.isfinite(float(self.error_target)):
                raise ValueError("fixed_error_min_copies requires a finite error_target.")
            if not 0.0 < float(self.error_target) <= 1.0:
                raise ValueError("error_target must lie in (0,1].")
            if self.error_target_margin >= float(self.error_target):
                raise ValueError("error_target_margin must be smaller than error_target.")
            probability = (
                0.85
                if self.success_probability_threshold is None
                else float(self.success_probability_threshold)
            )
            if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
                raise ValueError(
                    "success_probability_threshold must be finite and lie in (0,1]."
                )
            object.__setattr__(self, "success_probability_threshold", probability)
            if self.scientific_copy_cap is not None and (
                isinstance(self.scientific_copy_cap, bool)
                or not isinstance(self.scientific_copy_cap, int)
                or self.scientific_copy_cap <= 0
            ):
                raise ValueError(
                    "scientific_copy_cap must be a positive integer or None."
                )
        elif self.error_target is not None:
            raise ValueError("error_target is only valid for fixed_error_min_copies.")
        elif self.success_probability_threshold is not None:
            raise ValueError(
                "success_probability_threshold is only valid for "
                "fixed_error_min_copies."
            )
        elif self.scientific_copy_cap is not None:
            raise ValueError(
                "scientific_copy_cap is only valid for fixed_error_min_copies."
            )

    @property
    def effective_error_threshold(self) -> Optional[float]:
        if self.error_target is None:
            return None
        return float(self.error_target - self.error_target_margin)


def default_fixed_budget_objective(
    copy_ceiling: int, *, budget_utilization_target: float = 0.99
) -> OptimizationObjective:
    """Build the backward-compatible objective for legacy callers."""

    return OptimizationObjective(
        mode=OptimizationMode.FIXED_BUDGET_MIN_ERROR,
        copy_ceiling=int(copy_ceiling),
        budget_utilization_target=float(budget_utilization_target),
    )
