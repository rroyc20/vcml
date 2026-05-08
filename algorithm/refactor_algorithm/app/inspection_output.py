from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from refactor_algorithm.app.inspection_records import (
    NodeRecord,
    branch_candidates_equal,
    fmt_value,
)
from refactor_algorithm.core.master.compare_arc_vs_bnp import discount_objective_cost_per_edge
from refactor_algorithm.core.pricing.node import branch_selection_key_from_parts


def print_node_summary(record: NodeRecord) -> None:
    sep = "─" * 72
    print(sep)
    disagg_tag = f"  disagg={record.disagg_result}" if record.disagg_result else ""
    print(
        f"[Node {record.node_id:>4d}]  depth={record.depth}  parent={record.parent_id}  "
        f"outcome={record.outcome}{disagg_tag}  time={record.node_time_s:.3f}s"
    )
    print(
        f"  Global  LB={fmt_value(record.global_lb_at_entry)}  UB={fmt_value(record.global_ub_at_entry)}  "
        f"open={record.open_nodes_at_entry}  col_pool={record.col_pool_size_at_entry}"
    )
    print(
        f"  CG      iters={record.total_cg_iters}  "
        f"cols_gen={record.total_cols_generated}  cols_added={record.total_cols_added}  "
        f"final_lp={fmt_value(record.final_lp_obj)}  art_sum={record.art_sum_at_converge:.2e}"
    )
    if record.total_cg_iters > 0 or record.solve_addcol_time_s > 0.0:
        print(
            f"  Timings rmp={record.solve_rmp_lp_time_s:.3f}s  cut_sep={record.solve_cut_separation_time_s:.3f}s  "
            f"rmp_all={record.solve_rmp_time_s:.3f}s  prc={record.solve_pricing_time_s:.3f}s  "
            f"addcol={record.solve_addcol_time_s:.3f}s  (incl. outside per-iter table)"
        )
    if record.sri_rounds > 0 or record.sri_cuts_added > 0 or record.sri_separation_time_s > 0.0:
        print(
            f"  SRI     rounds={record.sri_rounds}  added={record.sri_cuts_added}  "
            f"sep={record.sri_separation_time_s:.3f}s"
        )
    if record.rmp_failure_msg:
        print(f"  RMP     failure: {record.rmp_failure_msg}")
    print(f"  Model   vars={record.model_nvars}  constrs={record.model_ncons}")

    if record.active_constraints:
        print(f"  Constraints ({len(record.active_constraints)}):")
        for constraint in record.active_constraints:
            print(
                f"    [{constraint['family']}]  {constraint['sense']} {constraint['rhs']}  "
                f"day={constraint['day']}  target={constraint['target']}"
            )

    if record.sri_round_logs:
        for sri_log in record.sri_round_logs:
            print(
                f"  [SRI] round={int(sri_log.get('round', 0))} found={int(sri_log.get('found', 0))} "
                f"selected={int(sri_log.get('selected', 0))} added={int(sri_log.get('added', 0))} "
                f"max_viol={float(sri_log.get('max_violation', 0.0)):.6g} "
                f"avg_viol={float(sri_log.get('avg_violation', 0.0)):.6g} "
                f"per_day={dict(sri_log.get('per_day', {}) or {})}"
            )

    if record.cg_iters:
        print("  CG iterations:  ph1=Y → cover artificials still >0 right after that RMP solve (not separate Simplex Phase I)")
        print(f"    {'iter':>4}  {'LP_obj':>12}  {'gen':>5}  {'add':>5}  {'rmp_s':>7}  {'cut_s':>7}  {'prc_s':>7}  {'ph1':>4}  {'art':>8}")
        for cg_iter in record.cg_iters:
            ph1_mark = "Y" if cg_iter.phase1_active else "."
            lp_cell = (
                f"{cg_iter.lp_obj:>12.4f}"
                if not (isinstance(cg_iter.lp_obj, float) and math.isnan(cg_iter.lp_obj))
                else f"{'RMP_FAIL':>12}"
            )
            print(
                f"    {cg_iter.cg_iter:>4}  {lp_cell}  {cg_iter.cols_generated:>5}  {cg_iter.cols_added:>5}"
                f"  {cg_iter.rmp_time_s:>7.4f}  {cg_iter.cut_separation_time_s:>7.4f}  {cg_iter.pricing_time_s:>7.4f}  {ph1_mark:>4}  {cg_iter.art_sum:>8.2e}"
            )

    if record.chosen_candidate:
        chosen = record.chosen_candidate
        print(
            f"  Branch  rule={chosen['family']}  value={chosen['value']}  "
            f"rank_dist={chosen['rank_dist']}  d_ni={chosen['dist_nearest_int']}  "
            f"|v-0.5|={chosen['dist_to_half']}  day={chosen['day']}  target={chosen['target']}"
        )
        if len(record.branch_candidates) > 1:
            sorted_candidates = sorted(
                record.branch_candidates,
                key=lambda item: branch_selection_key_from_parts(str(item["family"]), float(item["value"])),
            )
            print(f"  Candidates ({len(record.branch_candidates)} total, best-first 5):")
            for candidate in sorted_candidates[:5]:
                marker = "→" if branch_candidates_equal(candidate, record.chosen_candidate) else " "
                print(
                    f"    {marker} [{candidate['family']}]  val={candidate['value']}  "
                    f"rank={candidate['rank_dist']}  d_ni={candidate['dist_nearest_int']}  {candidate['target']}"
                )
    elif record.outcome == "INTEGRAL":
        print("  → Integer solution found (no branching needed)")


def print_incumbent_discount_report(
    inst: Dict[str, Any],
    rmp: Any,
    best: Optional[Dict[str, Any]],
    theta: float,
    global_lb: float,
    global_ub: float,
) -> None:
    print("\n" + "═" * 72)
    print("OPTIMAL / INCUMBENT DISCOUNT & SOLUTION DETAIL")
    print("═" * 72)

    n_days = len(inst.get("periods", []) or [])
    discount_weight = float(theta) * float(n_days)
    print(f"  theta           : {theta:.6g}")
    print(f"  |T| (days)      : {n_days}")
    print(f"  discount weight : θ·|T| = {discount_weight:.6g}  (objective coef is -weight·c·var)")
    print(f"  Global LB / UB  : {fmt_value(global_lb)} / {fmt_value(global_ub)}")
    if math.isfinite(global_lb) and math.isfinite(global_ub) and abs(global_ub) > 1e-12:
        gap = max(0.0, (global_ub - global_lb) / abs(global_ub) * 100.0)
        print(f"  Gap (UB-LB)/|UB|: {gap:.4f}%")

    if not best:
        print("  Incumbent       : (none — no integer solution was stored)")
        return

    objective = best.get("objective")
    node_id = best.get("node_id")
    source = best.get("source", "")
    vars_dict: Dict[str, Any] = best.get("variables") or {}
    if not isinstance(vars_dict, dict):
        vars_dict = {}
    num_nonzero = sum(1 for _, value in vars_dict.items() if abs(float(value)) > 1e-6)
    print(f"  Incumbent source : {source or '(unspecified)'}")
    print(f"  Incumbent node_id: {node_id if node_id is not None else 'n/a (seed / no B&B snapshot)'}")
    print(f"  Incumbent obj    : {fmt_value(objective)}")
    print(f"  Nonzero master vars (snapshot): {num_nonzero}")

    if float(theta) <= 0.0:
        print("  Discount edges   : θ=0 — no discount term in objective.")
        return

    if not vars_dict or num_nonzero == 0:
        active_routes = best.get("active_routes")
        if isinstance(active_routes, list) and active_routes:
            print(
                "  Note: incumbent has active_routes but no RMP variable snapshot — "
                "discount y/z listing requires an integral node capture (B&P) or root MIP seed with variables."
            )
        elif source in {"alns_initial", ""} and not vars_dict:
            print(
                "  Note: upper bound may come from ALNS objective only; run until an integral RMP node "
                "is found to list y/z from the master."
            )

    eps = 1e-6
    travel_cost = getattr(rmp, "travel_cost", None) or inst.get("travel_cost") or {}

    rows: List[Tuple[str, Any, Optional[int], float, float, float]] = []

    for edge, name in getattr(rmp, "agg_y_var_name", getattr(rmp, "y_var_name", {})).items():
        if name not in vars_dict:
            continue
        value = float(vars_dict[name])
        if abs(value) <= eps:
            continue
        disc_cost = float(discount_objective_cost_per_edge(inst, edge, travel_cost))
        savings = discount_weight * disc_cost * value
        rows.append(("y", edge, None, value, disc_cost, savings))

    fallback_rmp = getattr(rmp, "_fallback_rmp", None)
    discount_map = getattr(fallback_rmp, "discount_z_var_name", {}) if fallback_rmp is not None else {}
    for (edge, driver), name in discount_map.items():
        if name not in vars_dict:
            continue
        value = float(vars_dict[name])
        if abs(value) <= eps:
            continue
        disc_cost = float(discount_objective_cost_per_edge(inst, edge, travel_cost))
        savings = discount_weight * disc_cost * value
        rows.append(("z", edge, int(driver), value, disc_cost, savings))

    if not rows:
        print("  Discount-active edges: (none — no y/z above ε in incumbent snapshot)")
        print("  Hint: A-RMP uses y_e; SimpleSPMaster uses z_{e,k}. Empty can mean no incumbent")
        print("        or integrality snapshot omitted fractional discount vars.")
        return

    rows.sort(key=lambda row: (-row[5], str(row[1]), row[2] if row[2] is not None else -1))
    total_savings = sum(row[5] for row in rows)
    print(f"  Estimated total discount cost reduction (Σ θ·|T|·c·var): {total_savings:.6g}")
    print(f"  Active discount vars ({len(rows)}):")
    print(f"    {'type':>4}  {'edge':^14}  {'k':>4}  {'var':>10}  {'c_e':>12}  {'θ|T|c·var':>14}")
    for kind, edge, driver, value, disc_cost, savings in rows:
        edge_key = f"({edge[0]},{edge[1]})"
        driver_text = "" if driver is None else str(driver)
        print(f"    {kind:>4}  {edge_key:^14}  {driver_text:>4}  {value:>10.6g}  {disc_cost:>12.6g}  {savings:>14.6g}")


def profile_sum(profile: Dict[str, Any], keys: Sequence[str]) -> float:
    return sum(float(profile.get(key, 0.0)) for key in keys)


def print_pricing_breakdown(profile: Dict[str, Any], pricing_time_s: float) -> None:
    denom = pricing_time_s if pricing_time_s > 1e-12 else 1.0
    data_node = profile_sum(
        profile,
        (
            "pricing_prof_prep_s",
            "pricing_prof_dual_cut_s",
            "pricing_prof_day_graph_s",
            "pricing_prof_yao_apsp_s",
        ),
    )
    cpp_bind = profile_sum(
        profile,
        (
            "pricing_prof_cpp_inproc_ctx_s",
            "pricing_prof_cpp_inproc_cand_svc_s",
            "pricing_prof_cpp_inproc_ctypes_s",
        ),
    )
    cpp_so = float(profile.get("pricing_prof_cpp_inproc_native_s", 0.0))
    post = profile_sum(
        profile,
        ("pricing_prof_cpp_inproc_decode_s", "pricing_prof_cpp_post_s"),
    )
    py_algo = profile_sum(profile, ("pricing_prof_py_dp_s", "pricing_prof_python_label_s"))
    accounted = data_node + cpp_bind + cpp_so + post + py_algo
    other = max(0.0, pricing_time_s - accounted)
    inproc_sum = profile_sum(
        profile,
        (
            "pricing_prof_cpp_inproc_ctx_s",
            "pricing_prof_cpp_inproc_cand_svc_s",
            "pricing_prof_cpp_inproc_ctypes_s",
            "pricing_prof_cpp_inproc_native_s",
            "pricing_prof_cpp_inproc_decode_s",
        ),
    )
    core_try = float(profile.get("pricing_prof_cpp_core_s", 0.0))
    wrapper_gap = max(0.0, core_try - inproc_sum)

    print("\n  prc_s (= pricing_time_s) breakdown (rollup, % of pricing_time_s):")
    rows: List[Tuple[str, float]] = [
        ("데이터 정리 (prep + dual/cut + day_graph + yao)", data_node),
        ("C++ 바인딩 준비 (ctx + 후보·svc + ctypes·버퍼)", cpp_bind),
        ("C++ .so (cpp_price_dp / cpp_price_ng)", cpp_so),
        ("후처리 (경로 복원 + RC/신규 컬럼 필터)", post),
        ("Python 가격 (dp + labeling)", py_algo),
        ("기타·미분류 (stabilizer·import 등)", other),
    ]
    for label, value in rows:
        print(f"    {label:<46} {value:8.4f}s  ({100.0 * value / denom:5.1f}%)")
    if core_try > 1e-8 or inproc_sum > 1e-8:
        print(
            f"    (참고) node cpp try 전체 pricing_prof_cpp_core_s={core_try:.4f}s  "
            f"inproc 합={inproc_sum:.4f}s  Δ≈래퍼·import={wrapper_gap:.4f}s"
        )
