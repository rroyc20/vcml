"""Pricing subproblem micro-benchmark.

직접 pricing 함수를 호출해 SimpleSPMaster/Gurobi 초기화 오버헤드 없이
순수 pricing 속도를 측정한다.

Usage:
    python scripts/_bench_pricing.py [--runs N] [--required-limit K] [--days D]
"""
from __future__ import annotations

import argparse
import heapq
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.existing_instance import list_existing_instances, load_existing_instance


# ── pricing 함수 직접 import ─────────────────────────────────────────────────

def _make_inst(dat_path: str, required_limit: int, days: int) -> dict:
    return load_existing_instance(
        dat_path=dat_path,
        use_full_instance=False,
        required_limit=required_limit,
        node_limit=20,
        num_days=days,
        schedule_mode="regular",
        vehicles_override=None,
        max_nodes=1,
        max_cg_iterations_per_node=1,
        gurobi_output=0,
        arc_gurobi_output=0,
        alg_gurobi_output=0,
        pricing_max_columns=50,
        require_proof_optimality=0,
        algorithm_time_limit_s=0,
        pricing_method="cpp_dp",
        node_search_strategy="dfs",
        eps_reduced_cost=1e-4,
        use_dual_stabilization=False,
        dual_stab_alpha=0.5,
        discount_theta=0.0,
        use_alns_initialization=False,
        alns_iterations=0,
        alns_destroy_fraction=0.25,
        alns_seed=-1,
        alns_replicate_all_contexts=False,
        phase1_col_cap=0,
        use_cutting_plane_separation=False,
        use_aggregation=0,
    )


def _build_pricing_inputs(inst: dict) -> dict:
    """
    BnBNode._get_pricing_graph_precompute()와 동일한 전처리를 직접 수행.
    adjacency를 구성하고 APSP + req metadata를 반환한다.
    """
    edges: List = inst["edges"]
    required_edge_set = set(inst["required_edges"])
    travel_cost: Dict = inst["travel_cost"]
    service_extra: Dict = inst["service_extra"]
    demand: Dict = inst["demand"]
    depot: int = inst["depot"]
    nodes: List = inst["nodes"]

    # adjacency (양방향)
    adjacency: Dict[Any, List[Dict]] = {n: [] for n in nodes}
    for e in edges:
        i, j = e
        req = e in required_edge_set
        dem = float(demand.get(e, 0.0)) if req else 0.0
        serv = float(service_extra.get(e, 0.0)) if req else 0.0
        tc = float(travel_cost[e])
        adjacency[i].append({"id": (i, j), "to": j, "travel_cost": tc,
                              "required": req, "required_id": e,
                              "demand": dem, "service_cost": serv})
        adjacency[j].append({"id": (j, i), "to": i, "travel_cost": tc,
                              "required": req, "required_id": e,
                              "demand": dem, "service_cost": serv})

    # req metadata + service arcs
    req_service_meta: Dict = {}
    req_service_arcs: Dict = {}
    pricing_nodes = {depot}
    for u, arcs in adjacency.items():
        for arc in arcs:
            if not arc.get("required", False):
                continue
            req_id = arc["required_id"]
            v = arc["to"]
            pricing_nodes.add(u)
            pricing_nodes.add(v)
            tc = float(arc["travel_cost"])
            dem = float(arc.get("demand", 0.0))
            sc = float(arc.get("service_cost", 0.0))
            req_service_arcs.setdefault(req_id, []).append((u, v, arc["id"], tc, dem, sc))
            min_total = tc + sc
            if req_id not in req_service_meta:
                req_service_meta[req_id] = (dem, min_total)
            else:
                old_dem, old_min = req_service_meta[req_id]
                req_service_meta[req_id] = (old_dem, min(old_min, min_total))

    # APSP (Dijkstra) on pricing nodes
    pricing_nodes_list = list(pricing_nodes)
    sp_cost: Dict = {}
    sp_path: Dict = {}
    for src in pricing_nodes_list:
        dist = {src: 0.0}
        prev: Dict = {}
        pq = [(0.0, src)]
        remaining = set(pricing_nodes_list)
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf) + 1e-12:
                continue
            if u in remaining:
                remaining.remove(u)
                if not remaining:
                    break
            for arc in adjacency.get(u, []):
                v2 = arc["to"]
                nd = d + float(arc["travel_cost"])
                if nd + 1e-12 < dist.get(v2, math.inf):
                    dist[v2] = nd
                    prev[v2] = (u, arc["id"])
                    heapq.heappush(pq, (nd, v2))
        sp_cost[src] = {}
        sp_path[src] = {}
        for dst in pricing_nodes_list:
            if src == dst:
                sp_cost[src][dst] = 0.0
                sp_path[src][dst] = ()
                continue
            if dst not in dist:
                sp_cost[src][dst] = math.inf
                sp_path[src][dst] = ()
                continue
            rev = []
            c = dst
            while c != src:
                pu, parc = prev[c]
                rev.append(parc)
                c = pu
            sp_cost[src][dst] = float(dist[dst])
            sp_path[src][dst] = tuple(reversed(rev))

    return {
        "adjacency": adjacency,
        "depot": depot,
        "req_service_meta": req_service_meta,
        "req_service_arcs": req_service_arcs,
        "sp_cost": sp_cost,
        "sp_path": sp_path,
        "capacity": float(inst["capacity"]),
        "req_ids": list(req_service_meta.keys()),
    }


def _make_duals(req_ids: list, pre: dict) -> Tuple[Dict, float]:
    """
    dual을 충분히 높게 설정해 모든 엣지가 개별적으로 profitable하고
    C++ 내부의 상태 공간 탐색이 최대로 발생하도록 한다.
      - edge_dual = max_sp_cost + service_cost * 3.0
        (depot에서 가장 멀리 있어도 deadheading을 상쇄)
      - vehicle_dual = large → starting RC = 매우 음수
    이렇게 하면 capacity 한도 내 모든 엣지 조합이 negative RC를 가져
    실제 state-space 탐색이 fully exercised된다.
    """
    # max possible deadhead cost
    sp_cost = pre["sp_cost"]
    max_dh = 0.0
    for row in sp_cost.values():
        for v in row.values():
            if math.isfinite(v):
                max_dh = max(max_dh, v)

    edge_duals = {}
    for r in req_ids:
        arcs = pre["req_service_arcs"].get(r, [])
        if arcs:
            min_svc = min(tc + sc for (_, _, _, tc, _, sc) in arcs)
        else:
            min_svc = 1.0
        edge_duals[r] = max_dh + min_svc * 3.0
    vehicle_dual = max_dh * len(req_ids) * 2.0
    return edge_duals, vehicle_dual


# ── cpp_dp 직접 호출 ─────────────────────────────────────────────────────────

def bench_cpp_dp(pre: dict, inst: dict, runs: int, max_columns: int = 50) -> Tuple[List[float], List[int]]:
    from src.pricing.cpp_pricer import solve_day_cpp_dp
    days = list(inst["periods"])
    vehicles = list(inst["vehicles"])
    edge_duals, vehicle_dual = _make_duals(pre["req_ids"], pre)

    # warm-up
    solve_day_cpp_dp(
        day=days[0], driver=vehicles[0] if vehicles else None,
        depot=pre["depot"], capacity=pre["capacity"],
        edge_duals=edge_duals, vehicle_dual=vehicle_dual,
        req_service_meta=pre["req_service_meta"],
        req_service_arcs=pre["req_service_arcs"],
        sp_cost=pre["sp_cost"], sp_path=pre["sp_path"],
        max_columns=max_columns, eps_reduced_cost=1e-4,
    )

    times: List[float] = []
    cols_found: List[int] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        total_cols = 0
        for day in days:
            for driver in vehicles:
                cols = solve_day_cpp_dp(
                    day=day, driver=driver,
                    depot=pre["depot"], capacity=pre["capacity"],
                    edge_duals=edge_duals, vehicle_dual=vehicle_dual,
                    req_service_meta=pre["req_service_meta"],
                    req_service_arcs=pre["req_service_arcs"],
                    sp_cost=pre["sp_cost"], sp_path=pre["sp_path"],
                    max_columns=max_columns, eps_reduced_cost=1e-4,
                    forbidden_edges=set(),
                )
                total_cols += len(cols)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        cols_found.append(total_cols)
    return times, cols_found


# ── Python dp 직접 호출 ──────────────────────────────────────────────────────

def _py_dp_one_day(
    pre: dict,
    edge_duals: Dict,
    vehicle_dual: float,
    capacity: float,
    depot: Any,
    eps_rc: float = 1e-4,
    max_columns: int = 50,
) -> int:
    """Python dp pricer (node.py의 dp 방식을 독립 구현)."""
    req_service_arcs = pre["req_service_arcs"]
    sp_cost = pre["sp_cost"]
    req_ids = pre["req_ids"]
    req_to_bit = {r: (1 << i) for i, r in enumerate(req_ids)}
    dom_eps = eps_rc * 0.1

    start_key = (0, depot)
    best_rc: Dict = {start_key: -vehicle_dual}
    best_load: Dict = {start_key: 0.0}
    pq: List = []
    serial = 0
    heapq.heappush(pq, (-vehicle_dual, serial, 0, depot))
    serial += 1

    while pq:
        cur_rc, _, mask, cur_node = heapq.heappop(pq)
        key_cur = (mask, cur_node)
        if cur_rc > best_rc.get(key_cur, math.inf) + dom_eps:
            continue
        cur_load = best_load.get(key_cur, 0.0)

        for req_id, svc_arcs in req_service_arcs.items():
            req_bit = req_to_bit.get(req_id, 0)
            if req_bit == 0 or (mask & req_bit) != 0:
                continue
            for svc_from, svc_to, _, svc_travel, dem, sc in svc_arcs:
                new_load = cur_load + dem
                if new_load > capacity + 1e-9:
                    continue
                dead = sp_cost.get(cur_node, {}).get(svc_from, math.inf)
                if not math.isfinite(dead):
                    continue
                new_mask = mask | req_bit
                new_rc = cur_rc + dead + svc_travel + sc - edge_duals.get(req_id, 0.0)
                key_new = (new_mask, svc_to)
                if new_rc + dom_eps < best_rc.get(key_new, math.inf):
                    best_rc[key_new] = new_rc
                    best_load[key_new] = new_load
                    heapq.heappush(pq, (new_rc, serial, new_mask, svc_to))
                    serial += 1

    found = 0
    for (mask, node), rc_val in best_rc.items():
        if mask == 0:
            continue
        back = sp_cost.get(node, {}).get(depot, math.inf)
        if not math.isfinite(back):
            continue
        if rc_val + back < -eps_rc:
            found += 1
            if found >= max_columns:
                break
    return found


def bench_py_dp(pre: dict, inst: dict, runs: int, max_columns: int = 50) -> Tuple[List[float], List[int]]:
    days = list(inst["periods"])
    vehicles = list(inst["vehicles"])
    edge_duals, vehicle_dual = _make_duals(pre["req_ids"], pre)
    capacity = pre["capacity"]
    depot = pre["depot"]

    # warm-up
    _py_dp_one_day(pre, edge_duals, vehicle_dual, capacity, depot, max_columns=max_columns)

    times: List[float] = []
    cols_found: List[int] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        total_cols = 0
        for _ in days:
            for _ in vehicles:
                total_cols += _py_dp_one_day(
                    pre, edge_duals, vehicle_dual, capacity, depot,
                    max_columns=max_columns,
                )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        cols_found.append(total_cols)
    return times, cols_found


def fmt_row(method: str, inst_name: str, runs: int, times: List[float], cols: List[int]) -> str:
    m = mean(times) * 1000
    mn = min(times) * 1000
    mx = max(times) * 1000
    sd = stdev(times) * 1000 if len(times) > 1 else 0.0
    mc = mean(cols)
    return f"{method:<10}  {inst_name:<20}  {runs:>5}  {m:>9.3f}  {mn:>8.3f}  {mx:>8.3f}  {sd:>8.3f}  {mc:>6.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--required-limit", type=int, default=10)
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--num-instances", type=int, default=5)
    parser.add_argument("--instance-dir", type=str, default="data/existing/egl")
    parser.add_argument("--max-columns", type=int, default=50)
    args = parser.parse_args()

    candidates = list_existing_instances(args.instance_dir)
    selected = candidates[: args.num_instances]

    hdr = f"{'method':<10}  {'instance':<20}  {'runs':>5}  {'mean_ms':>9}  {'min_ms':>8}  {'max_ms':>8}  {'std_ms':>8}  {'cols':>6}"
    print(hdr)
    print("-" * len(hdr))

    all_cpp: List[float] = []
    all_py: List[float] = []

    for p in selected:
        inst_name = p.stem
        print(f"  loading {inst_name}...", flush=True)
        inst = _make_inst(str(p), args.required_limit, args.days)
        pre = _build_pricing_inputs(inst)
        n_req = len(pre["req_ids"])
        n_ctx = len(inst["periods"]) * len(inst["vehicles"])
        print(f"    req_edges={n_req}  contexts(days×veh)={n_ctx}  nodes={len(pre['sp_cost'])}", flush=True)

        cpp_times, cpp_cols = bench_cpp_dp(pre, inst, args.runs, args.max_columns)
        print(fmt_row("cpp_dp", inst_name, args.runs, cpp_times, cpp_cols))
        all_cpp.extend(cpp_times)

        py_times, py_cols = bench_py_dp(pre, inst, args.runs, args.max_columns)
        print(fmt_row("py_dp", inst_name, args.runs, py_times, py_cols))
        all_py.extend(py_times)

    print("-" * len(hdr))
    if all_cpp and all_py:
        cm = mean(all_cpp) * 1000
        pm = mean(all_py) * 1000
        speedup = pm / cm if cm > 0 else float("inf")
        print(f"\nTotal  cpp_dp: {cm:.3f} ms/call   py_dp: {pm:.3f} ms/call   cpp speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
