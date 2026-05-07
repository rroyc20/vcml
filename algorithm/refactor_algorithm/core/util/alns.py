from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from refactor_algorithm.core.util.initial_heuristic import build_route_path_for_edges, canon_edge


Edge = Tuple[int, int]


def _order_edges_for_route(
    depot: int,
    edges: Sequence[Edge],
    travel_cost: Dict[Edge, float],
) -> List[Edge]:
    if not edges:
        return []
    def _dist(i: int, j: int) -> float:
        if int(i) == int(j):
            return 0.0
        return float(travel_cost[canon_edge(int(i), int(j))])

    remaining = [canon_edge(e[0], e[1]) for e in edges]
    ordered: List[Edge] = []
    cur = int(depot)
    while remaining:
        best_idx = 0
        best_cost = float("inf")
        for idx, e in enumerate(remaining):
            u, v = e
            c = min(_dist(cur, u), _dist(cur, v))
            if c < best_cost:
                best_cost = c
                best_idx = idx
        chosen = remaining.pop(best_idx)
        ordered.append(chosen)
        u, v = chosen
        if _dist(cur, u) <= _dist(cur, v):
            cur = int(v)
        else:
            cur = int(u)
    return ordered


def _routes_from_assignment(
    assignment: Dict[Edge, Tuple[int, int]],
    required_edges: Sequence[Edge],
    schedule_patterns: Dict[Edge, List[frozenset[int]]],
    periods: Sequence[int],
    vehicles: Sequence[int],
) -> Dict[Tuple[int, int], List[Edge]]:
    out: Dict[Tuple[int, int], List[Edge]] = {(int(t), int(k)): [] for t in periods for k in vehicles}
    for e_raw in required_edges:
        e = canon_edge(e_raw[0], e_raw[1])
        sel = assignment.get(e)
        if sel is None:
            continue
        p_idx, k = int(sel[0]), int(sel[1])
        pat = schedule_patterns[e][p_idx]
        for t in pat:
            out[(int(t), int(k))].append(e)
    return out


def _assignment_feasible(
    assignment: Dict[Edge, Tuple[int, int]],
    required_edges: Sequence[Edge],
    schedule_patterns: Dict[Edge, List[frozenset[int]]],
    periods: Sequence[int],
    vehicles: Sequence[int],
    demand: Dict[Edge, float],
    capacity: float,
) -> bool:
    routes = _routes_from_assignment(
        assignment=assignment,
        required_edges=required_edges,
        schedule_patterns=schedule_patterns,
        periods=periods,
        vehicles=vehicles,
    )
    for key, edges in routes.items():
        _ = key
        load = sum(float(demand[canon_edge(e[0], e[1])]) for e in edges)
        if load > float(capacity) + 1e-9:
            return False
    return True


def _spm_column_signature(col: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """Dedup key aligned with SimpleSPMaster._add_route_var (sorted served edges)."""
    served_raw = col.get("serviced_required_edges") or []
    arcs_raw = col.get("path_arcs") or []
    if not served_raw or not arcs_raw:
        return None
    se = tuple(
        sorted(
            canon_edge(int(e[0]), int(e[1]))
            for e in served_raw
            if isinstance(e, (tuple, list)) and len(e) >= 2
        )
    )
    pa = tuple(
        (int(a[0]), int(a[1]))
        for a in arcs_raw
        if isinstance(a, (tuple, list)) and len(a) >= 2
    )
    if not se or not pa:
        return None
    return (int(col["day"]), int(col["driver"]), se, pa)


def _merge_spm_columns_into_pool(
    pool: Dict[Tuple[Any, ...], Dict[str, Any]],
    columns: Sequence[Dict[str, Any]],
) -> None:
    for c in columns:
        sig = _spm_column_signature(c)
        if sig is None:
            continue
        if sig in pool:
            continue
        pool[sig] = {
            "day": int(c["day"]),
            "driver": int(c["driver"]),
            "serviced_required_edges": [tuple(int(x) for x in e) for e in c.get("serviced_required_edges", [])],
            "path_arcs": [tuple(int(x) for x in a) for a in c.get("path_arcs", [])],
            "cost": float(c.get("cost", 0.0)),
        }


def verify_rmp_feasible_assignment(
    inst: Dict[str, Any],
    assignment: Dict[Any, Tuple[int, int]],
) -> Tuple[bool, str]:
    """
    Check whether an edge→(pattern_idx, vehicle) map is feasible for the SPM-style RMP:
    full coverage of required edges, valid pattern indices and vehicles, and capacity per (day, driver).
    """
    if not inst.get("required_edges"):
        return True, ""
    required_edges: List[Edge] = [canon_edge(e[0], e[1]) for e in inst["required_edges"]]
    required_set = set(required_edges)
    canon_assignment: Dict[Edge, Tuple[int, int]] = {}
    for e_raw, v in assignment.items():
        if not isinstance(e_raw, (tuple, list)) or len(e_raw) < 2:
            return False, "invalid assignment key"
        e = canon_edge(int(e_raw[0]), int(e_raw[1]))
        if not isinstance(v, (tuple, list)) or len(v) < 2:
            return False, f"invalid value for edge {e}"
        canon_assignment[e] = (int(v[0]), int(v[1]))
    if set(canon_assignment.keys()) != required_set:
        missing = sorted(required_set - set(canon_assignment.keys()))
        if missing:
            return False, f"missing required edges (showing up to 5): {missing[:5]}"
        return False, "extra edges in assignment"
    periods: List[int] = [int(t) for t in inst["periods"]]
    vehicles: List[int] = [int(k) for k in inst["vehicles"]]
    schedule_patterns: Dict[Edge, List[frozenset[int]]] = inst["schedule_patterns"]
    demand: Dict[Edge, float] = inst["demand"]
    capacity = float(inst["capacity"])
    veh_set = set(vehicles)
    for e in required_edges:
        p_idx, k = canon_assignment[e]
        pats = schedule_patterns[e]
        if p_idx < 0 or p_idx >= len(pats):
            return False, f"invalid pattern index {p_idx} for edge {e}"
        if int(k) not in veh_set:
            return False, f"invalid vehicle {k} for edge {e}"
    if not _assignment_feasible(
        assignment=canon_assignment,
        required_edges=required_edges,
        schedule_patterns=schedule_patterns,
        periods=periods,
        vehicles=vehicles,
        demand=demand,
        capacity=capacity,
    ):
        return False, "capacity infeasible for some day–vehicle route"
    return True, ""


def _objective_from_assignment(
    assignment: Dict[Edge, Tuple[int, int]],
    inst: Dict[str, Any],
) -> Tuple[float, List[Dict[str, Any]]]:
    # Match SimpleSPMaster: Yao-style sparse travel on paths; discount only on master's non-required edges.
    from refactor_algorithm.core.master.compare_arc_vs_bnp import discount_objective_cost_per_edge, path_arcs_travel_total

    required_edges: List[Edge] = [canon_edge(e[0], e[1]) for e in inst["required_edges"]]
    edges_all: List[Edge] = [canon_edge(e[0], e[1]) for e in inst["edges"]]
    required_set = set(required_edges)
    _sparse_edge_set: set = set(inst.get("arc_sparse_edges", []))
    if _sparse_edge_set:
        nonrequired_edges = [e for e in edges_all if e not in required_set and e in _sparse_edge_set]
    else:
        nonrequired_edges = [e for e in edges_all if e not in required_set]
    nonreq_discount_eligible = set(nonrequired_edges)
    periods: List[int] = [int(t) for t in inst["periods"]]
    vehicles: List[int] = [int(k) for k in inst["vehicles"]]
    depot = int(inst["depot"])
    travel_cost: Dict[Edge, float] = inst["travel_cost"]
    service_extra: Dict[Edge, float] = inst["service_extra"]
    schedule_patterns: Dict[Edge, List[frozenset[int]]] = inst["schedule_patterns"]
    theta = float(inst.get("discount_theta", 0.0))

    by_day_driver = _routes_from_assignment(
        assignment=assignment,
        required_edges=required_edges,
        schedule_patterns=schedule_patterns,
        periods=periods,
        vehicles=vehicles,
    )
    columns: List[Dict[str, Any]] = []
    route_cost_total = 0.0
    nonreq_traverse: Dict[Tuple[Edge, int, int], float] = {}

    for t in periods:
        for k in vehicles:
            served = by_day_driver[(int(t), int(k))]
            if not served:
                continue
            ordered = _order_edges_for_route(depot=depot, edges=served, travel_cost=travel_cost)
            path_arcs = build_route_path_for_edges(
                depot=depot,
                serviced_edges=ordered,
                edge_cost=travel_cost,
            )
            travel = path_arcs_travel_total(inst, path_arcs, travel_cost)
            serv = sum(float(service_extra[canon_edge(e[0], e[1])]) for e in ordered)
            rcost = float(travel + serv)
            route_cost_total += float(rcost)
            columns.append(
                {
                    "day": int(t),
                    "driver": int(k),
                    "serviced_required_edges": [tuple(e) for e in ordered],
                    "path_arcs": [tuple(a) for a in path_arcs],
                    "cost": float(rcost),
                }
            )
            for a in path_arcs:
                e = canon_edge(int(a[0]), int(a[1]))
                if e in required_set:
                    continue
                if e not in nonreq_discount_eligible:
                    continue
                key = (e, int(t), int(k))
                nonreq_traverse[key] = nonreq_traverse.get(key, 0.0) + 1.0

    discount_sum = 0.0
    if theta > 0.0 and periods and nonrequired_edges:
        for e in nonrequired_edges:
            for k in vehicles:
                all_days_used = True
                for t in periods:
                    if nonreq_traverse.get((e, int(t), int(k)), 0.0) <= 0.0:
                        all_days_used = False
                        break
                if all_days_used:
                    discount_sum += float(discount_objective_cost_per_edge(inst, e, travel_cost))

    total_obj = float(route_cost_total - theta * float(len(periods)) * discount_sum)
    return total_obj, columns


def run_alns_initial_solution(
    inst: Dict[str, Any],
    iterations: int = 300,
    destroy_fraction: float = 0.25,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    required_edges: List[Edge] = [canon_edge(e[0], e[1]) for e in inst["required_edges"]]
    periods: List[int] = [int(t) for t in inst["periods"]]
    vehicles: List[int] = [int(k) for k in inst["vehicles"]]
    schedule_patterns: Dict[Edge, List[frozenset[int]]] = inst["schedule_patterns"]
    demand: Dict[Edge, float] = inst["demand"]
    capacity = float(inst["capacity"])
    rng_seed = (
        int(seed)
        if seed is not None
        else int(abs(hash((inst.get("instance_name", "inst"), len(required_edges), len(periods), len(vehicles)))) % (2**31))
    )
    rng = random.Random(rng_seed)

    if not required_edges:
        return {
            "objective": 0.0,
            "columns": [],
            "column_pool": [],
            "assignment": {},
            "active_routes": [],
            "rmp_feasible": True,
            "rmp_feasible_detail": "",
            "seed": rng_seed,
        }

    def candidate_pairs(edge: Edge) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        pats = schedule_patterns[edge]
        for p_idx in range(len(pats)):
            for k in vehicles:
                pairs.append((int(p_idx), int(k)))
        rng.shuffle(pairs)
        return pairs

    # Greedy randomized feasible construction.
    edges_order = list(required_edges)
    edges_order.sort(key=lambda e: float(demand[e]), reverse=True)
    assignment: Dict[Edge, Tuple[int, int]] = {}
    for e in edges_order:
        choices = candidate_pairs(e)
        placed = False
        for p_idx, k in choices:
            trial = dict(assignment)
            trial[e] = (int(p_idx), int(k))
            if _assignment_feasible(
                assignment=trial,
                required_edges=required_edges,
                schedule_patterns=schedule_patterns,
                periods=periods,
                vehicles=vehicles,
                demand=demand,
                capacity=capacity,
            ):
                assignment = trial
                placed = True
                break
        if not placed:
            raise RuntimeError(f"ALNS init failed: cannot place required edge {e} within capacity/day-driver limits.")

    best_assignment = dict(assignment)
    best_obj, best_cols = _objective_from_assignment(best_assignment, inst)
    column_pool_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    _merge_spm_columns_into_pool(column_pool_map, best_cols)
    current_assignment = dict(best_assignment)
    current_obj = float(best_obj)

    iters = max(1, int(iterations))
    destroy_frac = max(0.05, min(0.8, float(destroy_fraction)))
    remove_count = max(1, int(math.ceil(destroy_frac * len(required_edges))))

    for iter_idx in range(iters):
        working = dict(current_assignment)
        remove_edges = rng.sample(required_edges, k=min(remove_count, len(required_edges)))
        for e in remove_edges:
            working.pop(e, None)

        # Repair with randomized greedy insertion.
        reinsert = list(remove_edges)
        rng.shuffle(reinsert)
        feasible = True
        for e in reinsert:
            placed = False
            for p_idx, k in candidate_pairs(e):
                trial = dict(working)
                trial[e] = (int(p_idx), int(k))
                if _assignment_feasible(
                    assignment=trial,
                    required_edges=required_edges,
                    schedule_patterns=schedule_patterns,
                    periods=periods,
                    vehicles=vehicles,
                    demand=demand,
                    capacity=capacity,
                ):
                    working = trial
                    placed = True
                    break
            if not placed:
                feasible = False
                break
        if not feasible:
            continue

        cand_obj, cols_iter = _objective_from_assignment(working, inst)
        _merge_spm_columns_into_pool(column_pool_map, cols_iter)
        # SA-style acceptance to escape local minima.
        temp = max(1e-6, 1.0 - (float(iter_idx) / float(iters)))
        accept = cand_obj <= current_obj + 1e-9
        if not accept:
            delta = cand_obj - current_obj
            accept_prob = math.exp(-delta / max(1e-6, temp * max(1.0, abs(current_obj))))
            accept = rng.random() < accept_prob
        if accept:
            current_assignment = working
            current_obj = float(cand_obj)
        if cand_obj < best_obj - 1e-9:
            best_obj = float(cand_obj)
            best_assignment = dict(working)

    best_obj, best_cols = _objective_from_assignment(best_assignment, inst)
    _merge_spm_columns_into_pool(column_pool_map, best_cols)
    rmp_ok, rmp_detail = verify_rmp_feasible_assignment(inst, best_assignment)
    pool_sigs = sorted(column_pool_map.keys(), key=lambda s: (s[0], s[1], s[2], s[3]))
    column_pool: List[Dict[str, Any]] = [column_pool_map[s] for s in pool_sigs]

    active_routes: List[Dict[str, Any]] = []
    for idx, col in enumerate(best_cols):
        active_routes.append(
            {
                "var": f"alns_route_{idx}",
                "value": 1.0,
                "day": int(col["day"]),
                "driver": int(col["driver"]),
                "serviced_required_edges": [tuple(e) for e in col.get("serviced_required_edges", [])],
                "path_arcs": [tuple(a) for a in col.get("path_arcs", [])],
                "cost": float(col.get("cost", 0.0)),
            }
        )

    return {
        "objective": float(best_obj),
        "columns": best_cols,
        "column_pool": column_pool,
        "assignment": {tuple(e): (int(v[0]), int(v[1])) for e, v in best_assignment.items()},
        "active_routes": active_routes,
        "rmp_feasible": bool(rmp_ok),
        "rmp_feasible_detail": str(rmp_detail),
        "seed": rng_seed,
    }
