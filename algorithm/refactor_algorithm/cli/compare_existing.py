from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from refactor_algorithm.app.batch_compare import (
    BatchCompareConfig,
    run_existing_batch,
    select_instance_paths,
)
from refactor_algorithm.app.reporting import print_compare_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run arc-based check (optional) vs branch-and-price on EGL .dat instances.",
    )
    parser.add_argument("--instance-dir", type=str, default="data/existing/egl", help="Directory of .dat files.")
    parser.add_argument("--instances", type=str, default="", help="Comma-separated basenames to run.")
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
        help="regular: mixed patterns; all_days / all_edges_daily: every required edge every period.",
    )
    parser.add_argument("--vehicles", type=int, default=0, help="Fleet size (0 = from .dat).")
    parser.add_argument("--max-nodes", type=int, default=10000, help="BnB node limit.")
    parser.add_argument("--max-cg-iter", type=int, default=1000000, help="Max CG iterations per BnB node.")
    parser.add_argument("--gurobi-output", type=int, choices=[0, 1], default=0, help="Default Gurobi log for both solvers.")
    parser.add_argument("--arc-output", type=int, choices=[0, 1], default=1, help="Override arc-based solver log.")
    parser.add_argument("--alg-output", type=int, choices=[0, 1], default=0, help="Override B&P log.")
    parser.add_argument("--pricing-max-cols", type=int, default=0, help="Pricing column cap per solve.")
    parser.add_argument("--proof-mode", type=int, choices=[0, 1], default=0, help="1 = require proven BnB optimality.")
    parser.add_argument("--arc-time-limit-s", type=float, default=3600.0, help="Arc MIP time limit (0 = none).")
    parser.add_argument("--alg-time-limit-s", type=float, default=3600.0, help="B&P wall time (0 = none).")
    parser.add_argument(
        "--pricing-method",
        type=str,
        choices=["labeling", "dp", "cpp_dp", "cpp_dp_lex", "cpp_ng"],
        default="cpp_dp_lex",
        help="Pricing subproblem backend.",
    )
    parser.add_argument("--pricing-ng-size", type=int, default=8, help="ng-route neighborhood size for cpp_ng pricing.")
    parser.add_argument("--cut-pricing-mode", type=str, choices=["legacy", "bitmask", "auto"], default="legacy")
    parser.add_argument("--cut-pricing-dual-tol", type=float, default=1e-15)
    parser.add_argument("--use-coeff-dominance-filter", type=int, choices=[0, 1], default=1)
    parser.add_argument("--coeff-dom-obj-tol", type=float, default=1e-5)
    parser.add_argument("--skip-arc", type=int, choices=[0, 1], default=1, help="1 = only run B&P.")
    parser.add_argument("--search-strategy", type=str, choices=["dfs", "best_bound"], default="best_bound")
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
    parser.add_argument("--yao-pricing", type=int, choices=[0, 1], default=1, help="1 = μ in sparse SP (Yao-style).")
    parser.add_argument("--use-capacity-cuts", type=int, choices=[0, 1], default=0, help="Enable capacity-link cuts.")
    parser.add_argument("--cut-root-only", type=int, choices=[0, 1], default=0, help="Run separation only at root node.")
    parser.add_argument("--cut-separation-max-depth", type=int, default=None)
    return parser


def main() -> None:
    random.seed(42)
    args = build_parser().parse_args()

    selected = select_instance_paths(
        instance_dir=args.instance_dir,
        instances_csv=args.instances,
        num_instances=args.num_instances,
    )
    vehicles_override = None if int(args.vehicles) <= 0 else int(args.vehicles)
    use_aggregation = str(args.bnp_variant).lower() == "aggregated"

    config = BatchCompareConfig(
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
    rows = run_existing_batch(config)
    print_compare_report(rows)


if __name__ == "__main__":
    main()
