from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, List


def fmt_optional_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def pricing_share_from_profile(profile: Dict[str, Any]) -> float:
    pricing_s = float(profile.get("pricing_time_s", 0.0))
    rmp_s = float(profile.get("rmp_time_s", 0.0))
    addcol_s = float(profile.get("addcol_time_s", 0.0))
    denom = pricing_s + rmp_s + addcol_s
    return (pricing_s / denom) if denom > 0 else 0.0


def infer_mismatch_cause(row: Dict[str, Any]) -> str:
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


def print_compare_report(rows: List[Dict[str, Any]]) -> None:
    skipped_arc = any(r.get("skip_arc", False) for r in rows)

    if skipped_arc:
        print(
            "instance,instance_file,nodes_used,required_used,alg_obj,alg_time_s,"
            "mode,nodes,artificial_sum,hit_node_limit,hit_time_limit,hit_cg_limit,alg_gap_pct,pricing_share,"
            "cols_generated,cols_added,cg_iters,phase1_iters,"
            "root_incumbent,incumbent_obj,has_incumbent_solution,alg_error"
        )
        for row in rows:
            profile = row.get("profile", {})
            pricing_share = pricing_share_from_profile(profile)
            alg_gap_str = fmt_optional_float(row.get("alg_gap_pct"))
            root_inc_str = fmt_optional_float(row.get("root_incumbent"))
            inc_obj_str = fmt_optional_float(row.get("incumbent_obj"))
            err = "" if not row.get("alg_error") else str(row["alg_error"]).replace(",", ";")
            cols_gen = int(profile.get("columns_generated", 0))
            cols_add = int(profile.get("columns_added", 0))
            cg_iters = int(profile.get("cg_iterations", 0))
            ph1_iters = int(profile.get("phase1_iters", 0))
            print(
                f"{row['instance']},{row['instance_file']},{row['nodes_used']},{row['required_used']},"
                f"{row['alg_obj']:.6f},{row['alg_time']:.6f},"
                f"{row['mode']},{row['nodes']},{row['artificial_sum']:.6f},"
                f"{row['hit_node_limit']},{row['hit_time_limit']},{row['hit_cg_limit']},{alg_gap_str},{pricing_share:.6f},"
                f"{cols_gen},{cols_add},{cg_iters},{ph1_iters},"
                f"{root_inc_str},{inc_obj_str},{row['has_incumbent_solution']},{err}"
            )
    else:
        print(
            "instance,instance_file,nodes_used,required_used,arc_obj,alg_obj,match,arc_time_s,alg_time_s,arc_status,arc_hit_time_limit,"
            "mode,nodes,artificial_sum,hit_node_limit,hit_time_limit,hit_cg_limit,solver_gap_pct,alg_gap_pct,pricing_share,root_incumbent,"
            "incumbent_obj,incumbent_abs_diff,incumbent_solver_gap_pct,has_incumbent_solution,cause,alg_error"
        )
        for row in rows:
            pricing_share = pricing_share_from_profile(row.get("profile", {}))
            solver_gap_str = fmt_optional_float(row.get("solver_gap_pct"))
            alg_gap_str = fmt_optional_float(row.get("alg_gap_pct"))
            root_inc_str = fmt_optional_float(row.get("root_incumbent"))
            inc_obj_str = fmt_optional_float(row.get("incumbent_obj"))
            inc_abs_str = fmt_optional_float(row.get("incumbent_abs_diff"))
            inc_gap_str = fmt_optional_float(row.get("incumbent_solver_gap_pct"))
            cause = infer_mismatch_cause(row)
            err = "" if not row.get("alg_error") else str(row["alg_error"]).replace(",", ";")
            print(
                f"{row['instance']},{row['instance_file']},{row['nodes_used']},{row['required_used']},"
                f"{row['arc_obj']:.6f},{row['alg_obj']:.6f},{row['match']},{row['arc_time']:.6f},{row['alg_time']:.6f},"
                f"{row['arc_status']},{row['arc_hit_time_limit']},{row['mode']},{row['nodes']},{row['artificial_sum']:.6f},"
                f"{row['hit_node_limit']},{row['hit_time_limit']},{row['hit_cg_limit']},{solver_gap_str},{alg_gap_str},"
                f"{pricing_share:.6f},{root_inc_str},{inc_obj_str},{inc_abs_str},{inc_gap_str},"
                f"{row['has_incumbent_solution']},{cause},{err}"
            )

    alg_times = [row["alg_time"] for row in rows]
    print("--- summary ---")
    print(f"instances: {len(rows)}")
    if not skipped_arc:
        arc_times = [row["arc_time"] for row in rows]
        matches = sum(1 for row in rows if row["match"])
        print(f"matches: {matches}/{len(rows)}")
        print(f"arc_avg_s: {mean(arc_times):.6f}")
        print(f"arc_median_s: {median(arc_times):.6f}")
        print(f"arc_total_s: {sum(arc_times):.6f}")
    print(f"alg_avg_s: {mean(alg_times):.6f}")
    print(f"alg_median_s: {median(alg_times):.6f}")
    print(f"alg_total_s: {sum(alg_times):.6f}")

