"""Privileged full-information planning and matched evaluation."""

from .comparison import ComparisonResult, compare_costs, trajectory_divergence
from .cost import CostBreakdown, SharedNavigationCost
from .planner import (
    PedestrianState, PlanResult, ReferenceWaypointController, TimeExpandedAStar,
)

__all__ = [
    "ComparisonResult", "CostBreakdown", "PedestrianState", "PlanResult",
    "ReferenceWaypointController", "SharedNavigationCost", "TimeExpandedAStar",
    "compare_costs", "trajectory_divergence",
]
