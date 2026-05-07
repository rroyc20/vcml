from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from existing_instance import list_existing_instances, load_existing_instance
from src.master.arc_based_check import solve_arc_based_pcarp, solve_arc_based_pcarp_optimal
from src.master.compare_arc_vs_bnp import solve_with_current_algorithm
from src.master.compare_global_rmp_bnp import solve_with_global_rmp_algorithm
from src.master.aggregated_master import solve_with_aggregated_algorithm


def _fmt_optional_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _pricing_share_from_profile(profile: Dict[str, Any]) -> float:
    pricing_s = float(profile.get("pricing_time_s", 0.0))
    rmp_s = float(profile.get("rmp_time_s", 0.0))
    addcol_s = float(profile.get("addcol_time_s", 0.0))
    denom = pricing_s + rmp_s + addcol_s
    return (pricing_s / denom) if denom > 0 else 0.0


def _infer_mismatch_cause(row: Dict[str, Any]) -> str:
    if row["match"]:
        return "ok"
    if row.get("alg_error"):
        return "proof_failed"
    if row.get("arc_hit_time_limit", False):
        return "solver_time_limit_reached"
    if row.get("artificial_sum", 0.0) > 1e-8:
        return "artificial_positive"
    if row.get("hit_node_limit", False):
        return "max_nodes_reached"
    if row.get("hit_time_limit", False):
        return "time_limit_reached"
    if row.get("hit_cg_limit", False):
        return "cg_iter_limit_reached"
    gap = row.get("alg_gap_pct", None)
    if gap is not None and float(gap) > 1e-6:
        return "bnb_gap_remaining"
    return "modeling_or_pricing_mismatch"


def run_existing_batch(
    instance_paths: List[str],
    full_instance: bool,
    required_limit: int,
    node_limit: int,
    days: int,
    schedule_mode: str,
    vehicles_override: int | None,
    max_nodes: int,
    max_cg_iterations_per_node: int,
    gurobi_output: int,
    arc_gurobi_output: int | None,
    alg_gurobi_output: int | None,
    pricing_max_columns: int,
    require_proof_optimality: int,
    arc_time_limit_s: float,
    alg_time_limit_s: float,
    pricing_method: str,
    cut_pricing_mode: str = "legacy",
    cut_pricing_dual_tol: float = 1e-15,
    use_coeff_dominance_filter: int = 1,
    coeff_dom_obj_tol: float = 1e-9,
    node_search_strategy: str = "dfs",
    eps_reduced_cost: float = 1e-4,
    discount_theta: float = 0.0,
    use_alns_initialization: bool = False,
    alns_iterations: int = 300,
    skip_arc: bool = False,
    bnp_variant: str = "aggregated",
    phase1_col_cap: int = 3,
    use_aggregation: bool = False,
    yao_style_pricing: int = 1,
    pricing_ng_size: int = 8,
    use_capacity_cuts: int = 0,
    cut_root_only: int = 1,
    cut_separation_max_depth: int | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for p in instance_paths:
        try:
            inst = load_existing_instance(
                dat_path=p,
                use_full_instance=bool(full_instance),
                required_limit=required_limit,
                node_limit=node_limit,
                num_days=days,
                schedule_mode=str(schedule_mode),
                vehicles_override=vehicles_override,
                max_nodes=max_nodes,
                max_cg_iterations_per_node=max_cg_iterations_per_node,
                gurobi_output=gurobi_output,
                arc_gurobi_output=arc_gurobi_output,
                alg_gurobi_output=alg_gurobi_output,
                pricing_max_columns=pricing_max_columns,
                require_proof_optimality=require_proof_optimality,
                algorithm_time_limit_s=alg_time_limit_s,
                pricing_method=pricing_method,
                pricing_ng_size=int(pricing_ng_size),
                cut_pricing_mode=cut_pricing_mode,
                cut_pricing_dual_tol=cut_pricing_dual_tol,
                use_coeff_dominance_filter=int(use_coeff_dominance_filter),
                coeff_dom_obj_tol=float(coeff_dom_obj_tol),
                node_search_strategy=node_search_strategy,
                eps_reduced_cost=eps_reduced_cost,
                use_dual_stabilization=False,
                dual_stab_alpha=0.5,
                discount_theta=discount_theta,
                use_alns_initialization=use_alns_initialization,
                alns_iterations=alns_iterations,
                alns_destroy_fraction=0.25,
                alns_seed=-1,
                alns_replicate_all_contexts=True,
                phase1_col_cap=phase1_col_cap,
                use_aggregation=int(use_aggregation),
                yao_style_pricing=int(yao_style_pricing),
            )
            # Extra cut controls not exposed by load_existing_instance() signature.
            inst["use_capacity_cuts"] = int(use_capacity_cuts)
            inst["cut_root_only"] = int(cut_root_only)
            if cut_separation_max_depth is not None:
                inst["cut_separation_max_depth"] = int(cut_separation_max_depth)
        except Exception as exc:
            print(f"[SKIP] Instance construction failed for {p}: {exc}", flush=True)
            rows.append({
                "instance": Path(p).stem,
                "instance_file": p,
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
            })
            continue

        # --- Arc-based solver (skippable) ---
        if skip_arc:
            arc = {
                "objective": float("nan"),
                "status": -1,
                "gap_pct": None,
            }
            t0 = t1 = time.perf_counter()
        else:
            # Both models must use the same graph representation (metric closure)
            # to ensure objective comparability.  The sparse subgraph lacks
            # intermediate nodes, making the arc-based model solve a harder
            # (different) problem and yielding inflated optimal values.
            t0 = time.perf_counter()
            if arc_time_limit_s > 0:
                arc = solve_arc_based_pcarp(
                    instance=inst,
                    time_limit=arc_time_limit_s,
                    require_optimal=False,
                    mip_gap=0.0,
                )
            else:
                arc = solve_arc_based_pcarp_optimal(inst)
            t1 = time.perf_counter()

        # --- BP algorithm ---
        t2 = time.perf_counter()
        alg_error = None
        try:
            _variant = str(bnp_variant).lower()
            _use_agg = bool(use_aggregation)
            if _use_agg:
                alg = solve_with_aggregated_algorithm(inst)
            elif _variant == "global_rmp":
                alg = solve_with_global_rmp_algorithm(inst)
            else:
                alg = solve_with_current_algorithm(inst)
        except Exception as exc:
            alg = {
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
            alg_error = str(exc)
        t3 = time.perf_counter()

        arc_obj = float(arc["objective"])
        alg_obj = float(alg["objective"])
        arc_status = int(arc.get("status", -1))
        arc_hit_time_limit = bool(arc_status == 9)  # Gurobi TIME_LIMIT
        solver_gap_pct = arc.get("gap_pct", None)
        alg_gap_pct = alg.get("gap_pct", None)
        inc_obj_raw = alg.get("incumbent_objective", None)
        inc_obj = None if inc_obj_raw is None else float(inc_obj_raw)
        inc_abs_diff = None
        inc_rel_gap_pct = None
        if inc_obj is not None and not skip_arc:
            inc_abs_diff = abs(inc_obj - arc_obj)
            if abs(arc_obj) > 1e-12:
                inc_rel_gap_pct = (inc_obj - arc_obj) / abs(arc_obj) * 100.0
        rows.append(
            {
                "instance": inst["instance_name"],
                "instance_file": p,
                "nodes_used": len(inst["nodes"]),
                "required_used": len(inst["required_edges"]),
                "arc_obj": arc_obj,
                "alg_obj": alg_obj,
                "match": abs(arc_obj - alg_obj) <= 1e-6 if not skip_arc else None,
                "arc_time": t1 - t0,
                "alg_time": t3 - t2,
                "arc_status": arc_status,
                "arc_hit_time_limit": arc_hit_time_limit,
                "mode": alg.get("mode", "unknown"),
                "nodes": alg.get("nodes_processed", None),
                "artificial_sum": float(alg.get("artificial_sum", 0.0)),
                "hit_node_limit": bool(alg.get("hit_node_limit", False)),
                "hit_time_limit": bool(alg.get("hit_time_limit", False)),
                "hit_cg_limit": bool(alg.get("hit_cg_limit", False)),
                "solver_gap_pct": solver_gap_pct,
                "alg_gap_pct": alg_gap_pct,
                "profile": dict(alg.get("profile", {})),
                "root_incumbent": alg.get("root_incumbent", None),
                "incumbent_obj": inc_obj,
                "incumbent_abs_diff": inc_abs_diff,
                "incumbent_solver_gap_pct": inc_rel_gap_pct,
                "has_incumbent_solution": bool(alg.get("incumbent_solution") is not None),
                "alg_error": alg_error,
                "skip_arc": bool(skip_arc),
            }
        )
    return rows


def print_report(rows: List[Dict[str, Any]]) -> None:
    skipped_arc = any(r.get("skip_arc", False) for r in rows)

    if skipped_arc:
        # Compact report: algorithm-only columns.
        print(
            "instance,instance_file,nodes_used,required_used,alg_obj,alg_time_s,"
            "mode,nodes,artificial_sum,hit_node_limit,hit_time_limit,hit_cg_limit,alg_gap_pct,pricing_share,"
            "cols_generated,cols_added,cg_iters,phase1_iters,"
            "root_incumbent,incumbent_obj,has_incumbent_solution,alg_error"
        )
        for r in rows:
            prof = r.get("profile", {})
            pricing_share = _pricing_share_from_profile(prof)
            alg_gap_str = _fmt_optional_float(r.get("alg_gap_pct"))
            root_inc_str = _fmt_optional_float(r.get("root_incumbent"))
            inc_obj_str = _fmt_optional_float(r.get("incumbent_obj"))
            err = "" if not r.get("alg_error") else str(r["alg_error"]).replace(",", ";")
            cols_gen = int(prof.get("columns_generated", 0))
            cols_add = int(prof.get("columns_added", 0))
            cg_iters = int(prof.get("cg_iterations", 0))
            ph1_iters = int(prof.get("phase1_iters", 0))
            print(
                f"{r['instance']},{r['instance_file']},{r['nodes_used']},{r['required_used']},"
                f"{r['alg_obj']:.6f},{r['alg_time']:.6f},"
                f"{r['mode']},{r['nodes']},{r['artificial_sum']:.6f},"
                f"{r['hit_node_limit']},{r['hit_time_limit']},{r['hit_cg_limit']},{alg_gap_str},{pricing_share:.6f},"
                f"{cols_gen},{cols_add},{cg_iters},{ph1_iters},"
                f"{root_inc_str},{inc_obj_str},{r['has_incumbent_solution']},{err}"
            )
    else:
        # Full report: arc-based vs algorithm comparison.
        print(
            "instance,instance_file,nodes_used,required_used,arc_obj,alg_obj,match,arc_time_s,alg_time_s,arc_status,arc_hit_time_limit,"
            "mode,nodes,artificial_sum,hit_node_limit,hit_time_limit,hit_cg_limit,solver_gap_pct,alg_gap_pct,pricing_share,root_incumbent,"
            "incumbent_obj,incumbent_abs_diff,incumbent_solver_gap_pct,has_incumbent_solution,cause,alg_error"
        )
        for r in rows:
            pricing_share = _pricing_share_from_profile(r.get("profile", {}))
            solver_gap_str = _fmt_optional_float(r.get("solver_gap_pct"))
            alg_gap_str = _fmt_optional_float(r.get("alg_gap_pct"))
            root_inc_str = _fmt_optional_float(r.get("root_incumbent"))
            inc_obj_str = _fmt_optional_float(r.get("incumbent_obj"))
            inc_abs_str = _fmt_optional_float(r.get("incumbent_abs_diff"))
            inc_gap_str = _fmt_optional_float(r.get("incumbent_solver_gap_pct"))
            cause = _infer_mismatch_cause(r)
            err = "" if not r.get("alg_error") else str(r["alg_error"]).replace(",", ";")
            print(
                f"{r['instance']},{r['instance_file']},{r['nodes_used']},{r['required_used']},"
                f"{r['arc_obj']:.6f},{r['alg_obj']:.6f},{r['match']},{r['arc_time']:.6f},{r['alg_time']:.6f},"
                f"{r['arc_status']},{r['arc_hit_time_limit']},{r['mode']},{r['nodes']},{r['artificial_sum']:.6f},"
                f"{r['hit_node_limit']},{r['hit_time_limit']},{r['hit_cg_limit']},{solver_gap_str},{alg_gap_str},"
                f"{pricing_share:.6f},{root_inc_str},{inc_obj_str},{inc_abs_str},{inc_gap_str},"
                f"{r['has_incumbent_solution']},{cause},{err}"
            )

    alg_times = [r["alg_time"] for r in rows]
    print("--- summary ---")
    print(f"instances: {len(rows)}")
    if not skipped_arc:
        arc_times = [r["arc_time"] for r in rows]
        matches = sum(1 for r in rows if r["match"])
        print(f"matches: {matches}/{len(rows)}")
        print(f"arc_avg_s: {mean(arc_times):.6f}")
        print(f"arc_median_s: {median(arc_times):.6f}")
        print(f"arc_total_s: {sum(arc_times):.6f}")
    print(f"alg_avg_s: {mean(alg_times):.6f}")
    print(f"alg_median_s: {median(alg_times):.6f}")
    print(f"alg_total_s: {sum(alg_times):.6f}")


def main() -> None:
    import random
    random.seed(42)
    parser = argparse.ArgumentParser(
        description="Run arc-based check (optional) vs branch-and-price on EGL .dat instances.",
    )
    parser.add_argument("--instance-dir", type=str, default="data/existing/egl", help="Directory of .dat files.")
    parser.add_argument(
        "--instances",
        type=str,
        default="",
        help="Comma-separated basenames (default: first --num-instances files, sorted).",
    )
    parser.add_argument("--num-instances", type=int, default=9, help="If --instances is empty, run this many files.")
    parser.add_argument("--full-instance", type=int, choices=[0, 1], default=0, help="1=all required edges & nodes; 0=subsample.")
    parser.add_argument("--required-limit", type=int, default=10, help="Required edges when full-instance=0.")
    parser.add_argument("--node-limit", type=int, default=50, help="Node cap when full-instance=0.")
    parser.add_argument("--days", type=int, default=4, help="Number of periods |T|.")
    parser.add_argument(
        "--schedule-mode",
        type=str,
        choices=["all_days", "all_edges_daily", "regular"],
        default="regular",
        help="regular: mixed patterns; all_days / all_edges_daily: every required edge every period (same patterns).",
    )
    parser.add_argument("--vehicles", type=int, default=0, help="Fleet size (0 = from .dat).")
    parser.add_argument("--max-nodes", type=int, default=10000, help="BnB node limit.")
    parser.add_argument("--max-cg-iter", type=int, default=1000000, help="Max CG iterations per BnB node.")
    parser.add_argument("--gurobi-output", type=int, choices=[0, 1], default=0, help="Default Gurobi log for both solvers.")
    parser.add_argument("--arc-output", type=int, choices=[0, 1], default=1, help="Override arc-based solver log.")
    parser.add_argument("--alg-output", type=int, choices=[0, 1], default=0, help="Override B&P log.")
    parser.add_argument(
        "--pricing-max-cols",
        type=int,
        default=0,
        help="Pricing column cap per solve (>0 fixed cap, <=0 high auto cap).",
    )
    parser.add_argument("--proof-mode", type=int, choices=[0, 1], default=0, help="1 = require proven BnB optimality.")
    parser.add_argument("--arc-time-limit-s", type=float, default=3600.0, help="Arc MIP time limit (0 = none).")
    parser.add_argument("--alg-time-limit-s", type=float, default=3600.0, help="B&P wall time (0 = none).")
    parser.add_argument(
        "--pricing-method",
        type=str,
        choices=["labeling", "dp", "cpp_dp", "cpp_dp_lex", "cpp_ng"],
        default="cpp_dp_lex",
        help="Pricing subproblem backend (cpp_dp_lex: include vehicle-lex duals in pricing RC; cpp_ng: Yao-style ng-route relaxation in C++).",
    )
    parser.add_argument(
        "--pricing-ng-size",
        type=int,
        default=8,
        help="ng-route neighborhood size for cpp_ng pricing.",
    )
    parser.add_argument(
        "--cut-pricing-mode",
        type=str,
        choices=["legacy", "bitmask", "auto"],
        default="legacy",
        help="Compatibility flag for cut-dual pricing.",
    )
    parser.add_argument(
        "--cut-pricing-dual-tol",
        type=float,
        default=1e-15,
        help="Ignore cut duals with |pi| <= tol when building pricing extras.",
    )
    parser.add_argument(
        "--use-coeff-dominance-filter",
        type=int,
        choices=[0, 1],
        default=1,
        help="Filter coefficient-equivalent columns whose objective is not better than existing best.",
    )
    parser.add_argument(
        "--coeff-dom-obj-tol",
        type=float,
        default=1e-5,
        help="Objective tolerance for coefficient-equivalent dominance filter.",
    )
    parser.add_argument("--skip-arc", type=int, choices=[0, 1], default=1, help="1 = only run B&P (skip arc MIP).")
    parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["dfs", "best_bound"],
        default="best_bound",
        help="BnB node order.",
    )
    parser.add_argument("--eps-rc", type=float, default=1e-4, help="CG reduced-cost threshold.")
    parser.add_argument(
        "--bnp-variant",
        type=str,
        choices=["aggregated", "copy_rmp", "global_rmp"],
        default="global_rmp",
        help="aggregated=A-RMP (default); copy_rmp / global_rmp = other backends.",
    )
    parser.add_argument("--discount-theta", type=float, default=0.1, help="Non-required edge consistency discount θ.")
    parser.add_argument("--alns", type=int, choices=[0, 1], default=0, help="1 = ALNS initial columns.")
    parser.add_argument("--alns-iters", type=int, default=300, help="ALNS iterations (if --alns 1).")
    parser.add_argument("--phase1-col-cap", type=int, default=3, help="Phase-I columns per pricing call (0=off).")
    parser.add_argument(
        "--yao-pricing",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 = μ in sparse SP (Yao-style); 0 = legacy RC patch.",
    )
    parser.add_argument("--use-capacity-cuts", type=int, choices=[0, 1], default=0, help="Enable capacity-link cuts.")
    parser.add_argument("--cut-root-only", type=int, choices=[0, 1], default=0, help="Run separation only at root node.")
    parser.add_argument(
        "--cut-separation-max-depth",
        type=int,
        default=None,
        help="With --cut-root-only 0: only separate when B&B depth <= this (0=root only). Omit for no cap.",
    )
    args = parser.parse_args()

    candidates = list_existing_instances(args.instance_dir)
    if not candidates:
        raise RuntimeError(f"No .dat instances found in: {args.instance_dir}")

    if args.instances.strip():
        name_set = {s.strip() for s in args.instances.split(",") if s.strip()}
        selected = [str(p) for p in candidates if p.name in name_set]
        if len(selected) != len(name_set):
            found = {Path(s).name for s in selected}
            missing = sorted(name_set - found)
            raise RuntimeError(f"Requested instance file(s) not found: {missing}")
    else:
        selected = [str(p) for p in candidates[: max(1, int(args.num_instances))]]

    vehicles_override = None if int(args.vehicles) <= 0 else int(args.vehicles)
    use_aggregation = str(args.bnp_variant).lower() == "aggregated"

    rows = run_existing_batch(
        instance_paths=selected,
        full_instance=bool(args.full_instance),
        required_limit=int(args.required_limit),
        node_limit=int(args.node_limit),
        days=int(args.days),
        schedule_mode=str(args.schedule_mode),
        vehicles_override=vehicles_override,
        max_nodes=int(args.max_nodes),
        max_cg_iterations_per_node=int(args.max_cg_iter),
        gurobi_output=int(args.gurobi_output),
        arc_gurobi_output=args.arc_output,
        alg_gurobi_output=args.alg_output,
        pricing_max_columns=int(args.pricing_max_cols),
        require_proof_optimality=int(args.proof_mode),
        arc_time_limit_s=float(args.arc_time_limit_s),
        alg_time_limit_s=float(args.alg_time_limit_s),
        pricing_method=str(args.pricing_method),
        cut_pricing_mode=str(args.cut_pricing_mode),
        cut_pricing_dual_tol=float(args.cut_pricing_dual_tol),
        use_coeff_dominance_filter=int(args.use_coeff_dominance_filter),
        coeff_dom_obj_tol=float(args.coeff_dom_obj_tol),
        node_search_strategy=str(args.search_strategy),
        eps_reduced_cost=float(args.eps_rc),
        discount_theta=float(args.discount_theta),
        use_alns_initialization=bool(int(args.alns)),
        alns_iterations=int(args.alns_iters),
        skip_arc=bool(int(args.skip_arc)),
        bnp_variant=str(args.bnp_variant),
        phase1_col_cap=int(args.phase1_col_cap),
        use_aggregation=use_aggregation,
        yao_style_pricing=int(args.yao_pricing),
        pricing_ng_size=int(args.pricing_ng_size),
        use_capacity_cuts=int(args.use_capacity_cuts),
        cut_root_only=int(args.cut_root_only),
        cut_separation_max_depth=args.cut_separation_max_depth,
    )
    print_report(rows)


if __name__ == "__main__":
    main()
