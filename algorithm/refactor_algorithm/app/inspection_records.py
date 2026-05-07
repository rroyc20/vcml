from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from refactor_algorithm.core.pricing.node import (
    BINARY_LIKE_BRANCH_FAMILIES,
    dist_to_nearest_integer,
)


@dataclass
class CGIterRecord:
    cg_iter: int
    lp_obj: float
    cols_generated: int
    cols_added: int
    rmp_time_s: float
    cut_separation_time_s: float
    pricing_time_s: float
    phase1_active: bool
    art_sum: float


@dataclass
class NodeRecord:
    node_id: int
    depth: int
    parent_id: Optional[int]
    global_lb_at_entry: float
    global_ub_at_entry: float
    open_nodes_at_entry: int
    col_pool_size_at_entry: int
    global_lb_at_exit: Optional[float] = None
    global_ub_at_exit: Optional[float] = None
    active_constraints: List[Dict[str, Any]] = field(default_factory=list)
    active_cuts: List[Dict[str, Any]] = field(default_factory=list)
    cuts_added_this_node: List[Dict[str, Any]] = field(default_factory=list)
    cg_iters: List[CGIterRecord] = field(default_factory=list)
    total_cg_iters: int = 0
    total_cols_added: int = 0
    total_cols_generated: int = 0
    final_lp_obj: Optional[float] = None
    art_sum_at_converge: float = 0.0
    model_nvars: int = 0
    model_ncons: int = 0
    branch_candidates: List[Dict[str, Any]] = field(default_factory=list)
    chosen_candidate: Optional[Dict[str, Any]] = None
    outcome: str = "UNKNOWN"
    disagg_result: Optional[str] = None
    node_time_s: float = 0.0
    rmp_failure_msg: Optional[str] = None
    solve_rmp_time_s: float = 0.0
    solve_rmp_lp_time_s: float = 0.0
    solve_cut_separation_time_s: float = 0.0
    solve_pricing_time_s: float = 0.0
    solve_addcol_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "global_lb_at_entry": fmt_value(self.global_lb_at_entry),
            "global_ub_at_entry": fmt_value(self.global_ub_at_entry),
            "global_lb_at_exit": fmt_value(self.global_lb_at_exit),
            "global_ub_at_exit": fmt_value(self.global_ub_at_exit),
            "open_nodes_at_entry": self.open_nodes_at_entry,
            "col_pool_size_at_entry": self.col_pool_size_at_entry,
            "active_constraints": self.active_constraints,
            "active_cuts": self.active_cuts,
            "cuts_added_this_node": self.cuts_added_this_node,
            "total_cg_iters": self.total_cg_iters,
            "total_cols_added": self.total_cols_added,
            "total_cols_generated": self.total_cols_generated,
            "final_lp_obj": fmt_value(self.final_lp_obj),
            "art_sum_at_converge": self.art_sum_at_converge,
            "model_nvars": self.model_nvars,
            "model_ncons": self.model_ncons,
            "branch_candidates": self.branch_candidates,
            "chosen_candidate": self.chosen_candidate,
            "outcome": self.outcome,
            "disagg_result": self.disagg_result,
            "rmp_failure_msg": self.rmp_failure_msg,
            "node_time_s": round(self.node_time_s, 6),
            "solve_rmp_time_s": round(self.solve_rmp_time_s, 6),
            "solve_rmp_lp_time_s": round(self.solve_rmp_lp_time_s, 6),
            "solve_cut_separation_time_s": round(self.solve_cut_separation_time_s, 6),
            "solve_pricing_time_s": round(self.solve_pricing_time_s, 6),
            "solve_addcol_time_s": round(self.solve_addcol_time_s, 6),
            "cg_iters": [
                {
                    "iter": record.cg_iter,
                    "lp_obj": fmt_value(record.lp_obj),
                    "cols_gen": record.cols_generated,
                    "cols_added": record.cols_added,
                    "rmp_s": round(record.rmp_time_s, 4),
                    "cut_separation_s": round(record.cut_separation_time_s, 4),
                    "pricing_s": round(record.pricing_time_s, 4),
                    "phase1": record.phase1_active,
                    "art_sum": round(record.art_sum, 6),
                }
                for record in self.cg_iters
            ],
        }


def fmt_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return round(value, 6)
    return value


def branch_candidate_metrics(family: str, value: float) -> Dict[str, float]:
    raw_value = float(value)
    dist_nearest_int = round(dist_to_nearest_integer(raw_value), 6)
    dist_half = round(abs(raw_value - 0.5), 6)
    rank_dist = dist_half if family in BINARY_LIKE_BRANCH_FAMILIES else dist_nearest_int
    return {
        "dist_to_half": dist_half,
        "dist_nearest_int": dist_nearest_int,
        "rank_dist": rank_dist,
    }


def branch_candidates_equal(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> bool:
    if left is None or right is None:
        return False
    keys = ("family", "day", "driver", "value", "target")
    return all(left.get(key) == right.get(key) for key in keys)

