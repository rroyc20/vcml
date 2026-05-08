from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from refactor_algorithm.engine.instances import list_existing_instances, load_existing_instance
from refactor_algorithm.engine.solvers import solve_arc, solve_branch_and_price


@dataclass
class BatchCompareConfig:
    instance_paths: List[str]
    full_instance: bool
    required_limit: int
    node_limit: int
    days: int
    schedule_mode: str
    vehicles_override: int | None
    max_nodes: int
    max_cg_iterations_per_node: int
    gurobi_output: int
    arc_gurobi_output: int | None
    alg_gurobi_output: int | None
    pricing_max_columns: int
    require_proof_optimality: int
    arc_time_limit_s: float
    alg_time_limit_s: float
    pricing_method: str
    cut_pricing_mode: str = "legacy"
    cut_pricing_dual_tol: float = 1e-15
    use_coeff_dominance_filter: int = 1
    coeff_dom_obj_tol: float = 1e-9
    node_search_strategy: str = "dfs"
    eps_reduced_cost: float = 1e-4
    discount_theta: float = 0.0
    use_alns_initialization: bool = False
    alns_iterations: int = 300
    skip_arc: bool = False
    bnp_variant: str = "aggregated"
    phase1_col_cap: int = 3
    use_aggregation: bool = False
    yao_style_pricing: int = 1
    pricing_ng_size: int = 8
    use_capacity_cuts: int = 0
    use_sri_cuts: int = 0
    sri_cardinality: int = 3
    enable_sri: int = 1
    root_only_sri: int = 1
    max_sri_rounds: int = 3
    max_cuts_per_round: int = 20
    max_cuts_per_day: int = 5
    min_sri_violation: float = 1e-4
    enable_sri_similarity_filter: int = 1
    max_shared_edges_between_sri3: int = 1
    cut_root_only: int = 1
    cut_separation_max_depth: int | None = None


def select_instance_paths(instance_dir: str, instances_csv: str, num_instances: int) -> List[str]:
    candidates = list_existing_instances(instance_dir)
    if not candidates:
        raise RuntimeError(f"No .dat instances found in: {instance_dir}")

    if instances_csv.strip():
        requested = {item.strip() for item in instances_csv.split(",") if item.strip()}
        selected = [str(path) for path in candidates if path.name in requested]
        if len(selected) != len(requested):
            found = {Path(path).name for path in selected}
            missing = sorted(requested - found)
            raise RuntimeError(f"Requested instance file(s) not found: {missing}")
        return selected

    return [str(path) for path in candidates[: max(1, int(num_instances))]]


def run_existing_batch(config: BatchCompareConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for path in config.instance_paths:
        try:
            instance = load_existing_instance(
                dat_path=path,
                use_full_instance=bool(config.full_instance),
                required_limit=config.required_limit,
                node_limit=config.node_limit,
                num_days=config.days,
                schedule_mode=str(config.schedule_mode),
                vehicles_override=config.vehicles_override,
                max_nodes=config.max_nodes,
                max_cg_iterations_per_node=config.max_cg_iterations_per_node,
                gurobi_output=config.gurobi_output,
                arc_gurobi_output=config.arc_gurobi_output,
                alg_gurobi_output=config.alg_gurobi_output,
                pricing_max_columns=config.pricing_max_columns,
                require_proof_optimality=config.require_proof_optimality,
                algorithm_time_limit_s=config.alg_time_limit_s,
                pricing_method=config.pricing_method,
                pricing_ng_size=int(config.pricing_ng_size),
                cut_pricing_mode=config.cut_pricing_mode,
                cut_pricing_dual_tol=config.cut_pricing_dual_tol,
                use_coeff_dominance_filter=int(config.use_coeff_dominance_filter),
                coeff_dom_obj_tol=float(config.coeff_dom_obj_tol),
                node_search_strategy=config.node_search_strategy,
                eps_reduced_cost=config.eps_reduced_cost,
                use_dual_stabilization=False,
                dual_stab_alpha=0.5,
                discount_theta=config.discount_theta,
                use_alns_initialization=config.use_alns_initialization,
                alns_iterations=config.alns_iterations,
                alns_destroy_fraction=0.25,
                alns_seed=-1,
                alns_replicate_all_contexts=True,
                phase1_col_cap=config.phase1_col_cap,
                use_aggregation=int(config.use_aggregation),
                yao_style_pricing=int(config.yao_style_pricing),
                use_sri_cuts=int(config.use_sri_cuts),
                sri_cardinality=int(config.sri_cardinality),
                enable_sri=int(config.enable_sri),
                root_only_sri=int(config.root_only_sri),
                max_sri_rounds=int(config.max_sri_rounds),
                max_cuts_per_round=int(config.max_cuts_per_round),
                max_cuts_per_day=int(config.max_cuts_per_day),
                min_sri_violation=float(config.min_sri_violation),
                enable_sri_similarity_filter=int(config.enable_sri_similarity_filter),
                max_shared_edges_between_sri3=int(config.max_shared_edges_between_sri3),
            )
            instance["use_capacity_cuts"] = int(config.use_capacity_cuts)
            instance["use_sri_cuts"] = int(config.use_sri_cuts)
            instance["sri_cardinality"] = int(config.sri_cardinality)
            instance["enable_sri"] = int(config.enable_sri)
            instance["root_only_sri"] = int(config.root_only_sri)
            instance["max_sri_rounds"] = int(config.max_sri_rounds)
            instance["max_cuts_per_round"] = int(config.max_cuts_per_round)
            instance["max_cuts_per_day"] = int(config.max_cuts_per_day)
            instance["min_sri_violation"] = float(config.min_sri_violation)
            instance["enable_sri_similarity_filter"] = int(config.enable_sri_similarity_filter)
            instance["max_shared_edges_between_sri3"] = int(config.max_shared_edges_between_sri3)
            instance["cut_root_only"] = int(config.cut_root_only)
            if config.cut_separation_max_depth is not None:
                instance["cut_separation_max_depth"] = int(config.cut_separation_max_depth)
        except Exception as exc:
            rows.append(_construction_failed_row(path, exc, config.skip_arc))
            continue

        arc_result, arc_time_s = _solve_arc_if_needed(instance, config)
        alg_result, alg_time_s, alg_error = _solve_algorithm(instance, config)
        rows.append(
            _build_row(
                instance=instance,
                instance_path=path,
                arc_result=arc_result,
                arc_time_s=arc_time_s,
                alg_result=alg_result,
                alg_time_s=alg_time_s,
                alg_error=alg_error,
                skip_arc=config.skip_arc,
            )
        )

    return rows


def _construction_failed_row(path: str, exc: Exception, skip_arc: bool) -> Dict[str, Any]:
    return {
        "instance": Path(path).stem,
        "instance_file": path,
        "nodes_used": 0,
        "required_used": 0,
        "arc_obj": float("nan"),
        "alg_obj": float("nan"),
        "match": None,
        "arc_time": 0.0,
        "alg_time": 0.0,
        "arc_status": -1,
        "arc_hit_time_limit": False,
        "mode": "construction_failed",
        "nodes": None,
        "artificial_sum": float("nan"),
        "hit_node_limit": None,
        "hit_time_limit": None,
        "hit_cg_limit": None,
        "solver_gap_pct": None,
        "alg_gap_pct": None,
        "profile": {},
        "root_incumbent": None,
        "incumbent_obj": None,
        "incumbent_abs_diff": None,
        "incumbent_solver_gap_pct": None,
        "has_incumbent_solution": False,
        "alg_error": f"construction_failed: {exc}",
        "skip_arc": bool(skip_arc),
    }


def _solve_arc_if_needed(instance: Dict[str, Any], config: BatchCompareConfig) -> tuple[Dict[str, Any], float]:
    import time

    if config.skip_arc:
        return {
            "objective": float("nan"),
            "status": -1,
            "gap_pct": None,
        }, 0.0

    t0 = time.perf_counter()
    result = solve_arc(instance=instance, time_limit=float(config.arc_time_limit_s))
    t1 = time.perf_counter()
    return result, (t1 - t0)


def _solve_algorithm(
    instance: Dict[str, Any],
    config: BatchCompareConfig,
) -> tuple[Dict[str, Any], float, Optional[str]]:
    import time

    t0 = time.perf_counter()
    error: Optional[str] = None
    try:
        result = solve_branch_and_price(
            instance=instance,
            variant=config.bnp_variant,
            use_aggregation=bool(config.use_aggregation),
        )
    except Exception as exc:
        result = {
            "objective": float("nan"),
            "mode": "failed",
            "nodes_processed": None,
            "artificial_sum": float("nan"),
            "hit_node_limit": None,
            "hit_time_limit": None,
            "hit_cg_limit": None,
            "gap_pct": None,
            "profile": {},
            "root_incumbent": None,
            "incumbent_objective": None,
            "incumbent_solution": None,
        }
        error = str(exc)
    t1 = time.perf_counter()
    return result, (t1 - t0), error


def _build_row(
    *,
    instance: Dict[str, Any],
    instance_path: str,
    arc_result: Dict[str, Any],
    arc_time_s: float,
    alg_result: Dict[str, Any],
    alg_time_s: float,
    alg_error: Optional[str],
    skip_arc: bool,
) -> Dict[str, Any]:
    arc_obj = float(arc_result["objective"])
    alg_obj = float(alg_result["objective"])
    arc_status = int(arc_result.get("status", -1))
    arc_hit_time_limit = bool(arc_status == 9)
    solver_gap_pct = arc_result.get("gap_pct", None)
    alg_gap_pct = alg_result.get("gap_pct", None)
    incumbent_raw = alg_result.get("incumbent_objective", None)
    incumbent_obj = None if incumbent_raw is None else float(incumbent_raw)
    incumbent_abs_diff = None
    incumbent_solver_gap_pct = None

    if incumbent_obj is not None and not skip_arc:
        incumbent_abs_diff = abs(incumbent_obj - arc_obj)
        if abs(arc_obj) > 1e-12:
            incumbent_solver_gap_pct = (incumbent_obj - arc_obj) / abs(arc_obj) * 100.0

    return {
        "instance": instance["instance_name"],
        "instance_file": instance_path,
        "nodes_used": len(instance["nodes"]),
        "required_used": len(instance["required_edges"]),
        "arc_obj": arc_obj,
        "alg_obj": alg_obj,
        "match": abs(arc_obj - alg_obj) <= 1e-6 if not skip_arc else None,
        "arc_time": arc_time_s,
        "alg_time": alg_time_s,
        "arc_status": arc_status,
        "arc_hit_time_limit": arc_hit_time_limit,
        "mode": alg_result.get("mode", "unknown"),
        "nodes": alg_result.get("nodes_processed", None),
        "artificial_sum": float(alg_result.get("artificial_sum", 0.0)),
        "hit_node_limit": bool(alg_result.get("hit_node_limit", False)),
        "hit_time_limit": bool(alg_result.get("hit_time_limit", False)),
        "hit_cg_limit": bool(alg_result.get("hit_cg_limit", False)),
        "solver_gap_pct": solver_gap_pct,
        "alg_gap_pct": alg_gap_pct,
        "profile": dict(alg_result.get("profile", {})),
        "root_incumbent": alg_result.get("root_incumbent", None),
        "incumbent_obj": incumbent_obj,
        "incumbent_abs_diff": incumbent_abs_diff,
        "incumbent_solver_gap_pct": incumbent_solver_gap_pct,
        "has_incumbent_solution": bool(alg_result.get("incumbent_solution") is not None),
        "alg_error": alg_error,
        "skip_arc": bool(skip_arc),
    }
