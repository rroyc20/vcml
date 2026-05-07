from __future__ import annotations

from typing import Any, Dict

from refactor_algorithm.core.master.aggregated_master import solve_with_aggregated_algorithm
from refactor_algorithm.core.master.arc_based_check import (
    solve_arc_based_pcarp,
    solve_arc_based_pcarp_optimal,
)
from refactor_algorithm.core.master.compare_arc_vs_bnp import solve_with_current_algorithm
from refactor_algorithm.core.master.compare_global_rmp_bnp import solve_with_global_rmp_algorithm


def solve_arc(instance: Dict[str, Any], time_limit: float) -> Dict[str, Any]:
    if time_limit > 0:
        return solve_arc_based_pcarp(
            instance=instance,
            time_limit=time_limit,
            require_optimal=False,
            mip_gap=0.0,
        )
    return solve_arc_based_pcarp_optimal(instance)


def solve_branch_and_price(instance: Dict[str, Any], variant: str, use_aggregation: bool) -> Dict[str, Any]:
    normalized = str(variant).lower()
    if use_aggregation:
        return solve_with_aggregated_algorithm(instance)
    if normalized == "global_rmp":
        return solve_with_global_rmp_algorithm(instance)
    return solve_with_current_algorithm(instance)

