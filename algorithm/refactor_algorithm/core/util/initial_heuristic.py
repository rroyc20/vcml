from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

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
