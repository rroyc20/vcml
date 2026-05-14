from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB


Edge = Tuple[int, int]
Arc = Tuple[int, int]


def canon_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def shortest_direct_arc(i: int, j: int, edge_cost: Dict[Edge, float]) -> Tuple[List[Arc], float]:
    if i == j:
        return [], 0.0
    e = canon_edge(i, j)
    return [(i, j)], float(edge_cost[e])


def build_route_path_for_edges(
    depot: int,
    serviced_edges: Sequence[Edge],
    edge_cost: Dict[Edge, float],
) -> List[Arc]:
    """
    Build a depot-to-depot path that services the given required edges in order.
    Greedy orientation choice at each serviced edge.
    """
    cur = depot
    path: List[Arc] = []

    for e in serviced_edges:
        u, v = e
        # option 1: ... -> u, then service u->v
        p1, c1 = shortest_direct_arc(cur, u, edge_cost)
        p1_cost = c1
        # option 2: ... -> v, then service v->u
        p2, c2 = shortest_direct_arc(cur, v, edge_cost)
        p2_cost = c2

        if p1_cost <= p2_cost:
            path.extend(p1)
            path.append((u, v))
            cur = v
        else:
            path.extend(p2)
            path.append((v, u))
            cur = u

    p_back, _ = shortest_direct_arc(cur, depot, edge_cost)
    path.extend(p_back)
    return path


def greedy_initial_columns(
    days: Sequence[int],
    required_edges: Sequence[Edge],
    demand: Dict[Edge, float],
    capacity: float,
    depot: int,
    edge_cost: Dict[Edge, float],
) -> List[Dict]:
    """
    Build initial route columns by packing required edges into capacity-feasible groups per day.
    This does not enforce schedule choices; it only prepares a rich initial omega.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive.")

    cols: List[Dict] = []
    req_list = list(required_edges)

    for day in days:
        # First-fit decreasing by demand for slightly better packing.
        sorted_edges = sorted(req_list, key=lambda e: float(demand.get(e, 0.0)), reverse=True)
        groups: List[List[Edge]] = []
        loads: List[float] = []

        for e in sorted_edges:
            de = float(demand.get(e, 0.0))
            placed = False
            for gi, load in enumerate(loads):
                if load + de <= capacity + 1e-9:
                    groups[gi].append(e)
                    loads[gi] += de
                    placed = True
                    break
            if not placed:
                groups.append([e])
                loads.append(de)

        for group in groups:
            path = build_route_path_for_edges(
                depot=depot,
                serviced_edges=group,
                edge_cost=edge_cost,
            )
            cols.append(
                {
                    "day": day,
                    "serviced_required_edges": list(group),
                    "path_arcs": path,
                }
            )

        # ensure at least single-edge columns exist (robustness)
        for e in req_list:
            cols.append(
                {
                    "day": day,
                    "serviced_required_edges": [e],
                    "path_arcs": build_route_path_for_edges(depot, [e], edge_cost),
                }
            )

    return cols


def _pack_edges_by_capacity(
    edges: Sequence[Edge],
    demand: Dict[Edge, float],
    capacity: float,
) -> Tuple[List[List[Edge]], List[float]]:
    """Best-fit decreasing packing for one day's required edges."""
    groups: List[List[Edge]] = []
    loads: List[float] = []
    ordered = sorted(
        (canon_edge(int(e[0]), int(e[1])) for e in edges),
        key=lambda e: float(demand.get(e, 0.0)),
        reverse=True,
    )

    for e in ordered:
        de = float(demand.get(e, 0.0))
        best_idx = -1
        best_slack = float("inf")
        for idx, load in enumerate(loads):
            new_load = float(load) + de
            slack = float(capacity) - new_load
            if slack >= -1e-9 and slack < best_slack:
                best_idx = idx
                best_slack = slack
        if best_idx < 0:
            groups.append([e])
            loads.append(de)
            continue
        groups[best_idx].append(e)
        loads[best_idx] = float(loads[best_idx]) + de

    return groups, loads


def build_q_load_aggregated_initial_columns(
    days: Sequence[int],
    required_edges: Sequence[Edge],
    schedule_patterns: Dict[Edge, List[frozenset[int]]],
    demand: Dict[Edge, float],
    capacity: float,
    num_vehicles: int,
    depot: int,
    edge_cost: Dict[Edge, float],
) -> Dict[str, Any]:
    """
    Build A-RMP seed columns from a q-load balancing heuristic.

    Step 1: choose one pattern per required edge to balance day-wise aggregate load.
    Step 2: for each day, pack active edges into capacity-feasible route groups.

    The output gives a schedule-consistent root column scaffold for aggregated q_{e,p}.
    """
    if float(capacity) <= 0.0:
        raise ValueError("capacity must be positive.")

    day_ids = [int(t) for t in days]
    req_edges = [canon_edge(int(e[0]), int(e[1])) for e in required_edges]
    if not req_edges:
        return {
            "columns": [],
            "selected_pattern_idx": {},
            "day_loads": {int(t): 0.0 for t in day_ids},
            "day_route_counts": {int(t): 0 for t in day_ids},
            "packing_feasible": True,
        }

    vehicle_count = max(1, int(num_vehicles))
    day_capacity = float(capacity) * float(vehicle_count)
    q_load: Dict[int, float] = {int(t): 0.0 for t in day_ids}
    selected_pattern_idx: Dict[Edge, int] = {}
    active_edges_by_day: Dict[int, List[Edge]] = {int(t): [] for t in day_ids}

    ordered_edges = sorted(
        req_edges,
        key=lambda e: (-float(demand.get(e, 0.0)), len(schedule_patterns.get(e, ())), e),
    )

    for e in ordered_edges:
        pats = schedule_patterns.get(e) or [frozenset(day_ids)]
        de = float(demand.get(e, 0.0))
        best_idx = 0
        best_score = None
        for p_idx, pat in enumerate(pats):
            pat_days = {int(t) for t in pat}
            cand_loads = {
                int(t): float(q_load[int(t)]) + (de if int(t) in pat_days else 0.0)
                for t in day_ids
            }
            overflow = sum(max(0.0, load - day_capacity) for load in cand_loads.values())
            denom = max(1.0, day_capacity)
            max_ratio = max((load / denom) for load in cand_loads.values()) if cand_loads else 0.0
            sq_ratio = sum((load / denom) ** 2 for load in cand_loads.values())
            load_span = (
                max(cand_loads.values()) - min(cand_loads.values()) if cand_loads else 0.0
            )
            score = (
                float(overflow),
                float(max_ratio),
                float(sq_ratio),
                float(load_span),
                len(pat_days),
                int(p_idx),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_idx = int(p_idx)

        selected_pattern_idx[e] = int(best_idx)
        chosen_pat = pats[best_idx]
        for t in chosen_pat:
            t_int = int(t)
            q_load[t_int] = float(q_load[t_int]) + de
            active_edges_by_day.setdefault(t_int, []).append(e)

    columns: List[Dict[str, Any]] = []
    day_route_counts: Dict[int, int] = {int(t): 0 for t in day_ids}
    packing_feasible = True

    for t in day_ids:
        groups, _loads = _pack_edges_by_capacity(
            active_edges_by_day.get(int(t), ()),
            demand=demand,
            capacity=float(capacity),
        )
        day_route_counts[int(t)] = int(len(groups))
        if len(groups) > vehicle_count:
            packing_feasible = False
        for group in groups:
            if not group:
                continue
            columns.append(
                {
                    "day": int(t),
                    "serviced_required_edges": list(group),
                    "path_arcs": build_route_path_for_edges(
                        depot=int(depot),
                        serviced_edges=group,
                        edge_cost=edge_cost,
                    ),
                }
            )

    return {
        "columns": columns,
        "selected_pattern_idx": selected_pattern_idx,
        "day_loads": {int(t): float(q_load[int(t)]) for t in day_ids},
        "day_route_counts": day_route_counts,
        "packing_feasible": bool(packing_feasible),
    }


def add_cover_artificials(
    model: gp.Model,
    cover_constr_names: Iterable[str],
    penalty: float = 1e5,
) -> Dict[str, str]:
    """
    Add Phase-I artificial variables to equality cover constraints:
      (cover_expr) + a_cover = 0, a_cover >= 0.
    A large objective penalty keeps artificials only when necessary.
    Returns mapping: cover constraint name -> artificial var name.
    """
    mapping: Dict[str, str] = {}
    model.update()
    for cname in cover_constr_names:
        c = model.getConstrByName(cname)
        if c is None:
            continue

        aname = f"a_{cname}"
        # avoid duplicate creation
        if model.getVarByName(aname) is not None:
            mapping[cname] = aname
            continue

        a = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=float(penalty), name=aname)
        model.chgCoeff(c, a, 1.0)
        mapping[cname] = aname

    model.update()
    return mapping
