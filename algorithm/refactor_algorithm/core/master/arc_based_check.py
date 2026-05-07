from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB


Edge = Tuple[int, int]


def _canon_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def _all_nonempty_subsets(nodes: Sequence[int], depot: int) -> Iterable[Tuple[int, ...]]:
    candidates = [i for i in nodes if i != depot]
    for r in range(1, len(candidates) + 1):
        for comb in combinations(candidates, r):
            yield comb


def _delta(edges: Sequence[Edge], subset: Sequence[int]) -> List[Edge]:
    s = set(subset)
    return [e for e in edges if (e[0] in s) ^ (e[1] in s)]


def _inside(edges: Sequence[Edge], subset: Sequence[int]) -> List[Edge]:
    s = set(subset)
    return [e for e in edges if e[0] in s and e[1] in s]


def build_tiny_instance() -> Dict[str, Any]:
    nodes = [0, 1, 2]
    depot = 0
    periods = [0, 1]
    vehicles = [0]

    edges = [_canon_edge(0, 1), _canon_edge(1, 2), _canon_edge(0, 2)]
    required_edges = [_canon_edge(0, 1), _canon_edge(1, 2)]

    travel_cost = {
        _canon_edge(0, 1): 2.0,
        _canon_edge(1, 2): 3.0,
        _canon_edge(0, 2): 2.0,
    }
    service_cost = {
        _canon_edge(0, 1): 1.0,
        _canon_edge(1, 2): 1.0,
    }
    demand = {
        _canon_edge(0, 1): 1.0,
        _canon_edge(1, 2): 1.0,
    }

    schedule_patterns: Dict[Edge, List[frozenset[int]]] = {
        _canon_edge(0, 1): [frozenset([0]), frozenset([1])],
        _canon_edge(1, 2): [frozenset([0]), frozenset([1])],
    }

    return {
        "nodes": nodes,
        "depot": depot,
        "periods": periods,
        "vehicles": vehicles,
        "edges": edges,
        "required_edges": required_edges,
        "travel_cost": travel_cost,
        "service_cost": service_cost,
        "demand": demand,
        "capacity": 2.0,
        "schedule_patterns": schedule_patterns,
    }


def _add_cutset_connectivity(
    m: gp.Model,
    V: Sequence[int],
    depot: int,
    T: Sequence[int],
    K: Sequence[int],
    E: Sequence[Edge],
    ER: Sequence[Edge],
    x: gp.tupledict,
    y: gp.tupledict,
) -> None:
    """Legacy cutset-based connectivity constraints."""
    for t in T:
        for k in K:
            for S in _all_nonempty_subsets(V, depot):
                delta_all = _delta(E, S)
                delta_req = [e for e in delta_all if e in ER]
                req_inside = _inside(ER, S)
                if not req_inside:
                    continue
                lhs = gp.quicksum(x[E.index(e), t, k] for e in delta_req) + gp.quicksum(
                    y[E.index(e), t, k] for e in delta_all
                )
                for f in req_inside:
                    m.addConstr(lhs >= 2.0 * x[E.index(f), t, k], name=f"conn_t{t}_k{k}_S{S}_f{f}")


def solve_arc_based_pcarp(
    instance: Dict[str, Any],
    time_limit: float | None = None,
    require_optimal: bool = False,
    mip_gap: float = 0.0,
) -> Dict[str, Any]:
    V = instance["nodes"]
    depot = instance["depot"]
    T = instance["periods"]
    K = instance["vehicles"]
    E = instance["edges"]
    ER = instance["required_edges"]
    r = instance["travel_cost"]
    c = instance["service_cost"]
    q = instance["demand"]
    Q = float(instance["capacity"])
    P = instance["schedule_patterns"]
    theta = float(instance.get("discount_theta", 0.0))

    e_idx = list(range(len(E)))
    er_idx = [E.index(e) for e in ER]
    idx_to_edge = {i: E[i] for i in e_idx}
    er_set = set(ER)
    sparse_edge_set = set(instance.get("arc_sparse_edges", []))
    if sparse_edge_set:
        nonreq_idx = [ei for ei in e_idx if idx_to_edge[ei] not in er_set and idx_to_edge[ei] in sparse_edge_set]
    else:
        nonreq_idx = [ei for ei in e_idx if idx_to_edge[ei] not in er_set]

    m = gp.Model("pcarp_arc_based_check")
    m.Params.OutputFlag = int(instance.get("arc_gurobi_output", instance.get("gurobi_output", 0)))
    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = float(mip_gap)

    x = m.addVars(er_idx, T, K, vtype=GRB.BINARY, name="x")
    y = m.addVars(e_idx, T, K, vtype=GRB.INTEGER, lb=0, ub=2 * len(E), name="y")
    z = m.addVars(nonreq_idx, K, vtype=GRB.BINARY, name="z")
    w = m.addVars(V, T, K, vtype=GRB.INTEGER, lb=0, ub=2 * len(E), name="w")

    s_keys = []
    for e in ER:
        for p_idx, _ in enumerate(P[e]):
            for k in K:
                s_keys.append((e, p_idx, k))
    s = m.addVars(s_keys, vtype=GRB.BINARY, name="s")

    obj_expr = (
        gp.quicksum(r[idx_to_edge[e]] * y[e, t, k] for e in e_idx for t in T for k in K)
        + gp.quicksum(c[idx_to_edge[e]] * x[e, t, k] for e in er_idx for t in T for k in K)
        # Discount on nonrequired edges traversed by same vehicle on all days.
        # z[e,k]=1 only if y[e,t,k] >= 1 for every t (linked below).
        # Objective uses -theta*|T|*r_e*z[e,k].
        - float(theta) * float(len(T)) * gp.quicksum(r[idx_to_edge[e]] * z[e, k] for e in nonreq_idx for k in K)
    )
    m.setObjective(obj_expr, GRB.MINIMIZE)

    # exactly one (schedule, vehicle) pair per required edge
    for e in ER:
        m.addConstr(
            gp.quicksum(s[e, p_idx, k] for p_idx in range(len(P[e])) for k in K) == 1,
            name=f"schedule_{e}",
        )

    # daily service induced by chosen schedule for each vehicle
    for e in ER:
        ei = E.index(e)
        for t in T:
            for k in K:
                rhs = gp.quicksum(s[e, p_idx, k] for p_idx, pat in enumerate(P[e]) if t in pat)
                m.addConstr(x[ei, t, k] == rhs, name=f"cover_{e}_t{t}_k{k}")

    # node incidence balance
    for t in T:
        for k in K:
            for i in V:
                delta_i = _delta(E, [i])
                delta_req_i = [e for e in delta_i if e in ER]
                lhs = gp.quicksum(x[E.index(e), t, k] for e in delta_req_i) + gp.quicksum(
                    y[E.index(e), t, k] for e in delta_i
                )
                m.addConstr(lhs == 2.0 * w[i, t, k], name=f"deg_i{i}_t{t}_k{k}")

    # one-route-per-vehicle-per-day alignment
    for t in T:
        for k in K:
            m.addConstr(w[depot, t, k] <= 1.0, name=f"single_trip_t{t}_k{k}")

    # capacity (per vehicle/day)
    for t in T:
        for k in K:
            m.addConstr(
                gp.quicksum(q[idx_to_edge[e]] * x[e, t, k] for e in er_idx) <= Q,
                name=f"cap_t{t}_k{k}",
            )

    # Discount-link constraints: if z[e,k]=1 then edge e must be traversed every day by k.
    for e in nonreq_idx:
        for t in T:
            for k in K:
                m.addConstr(y[e, t, k] >= z[e, k], name=f"disc_link_e{e}_t{t}_k{k}")

    # depot activation consistency
    for t in T:
        for k in K:
            for e in er_idx:
                m.addConstr(x[e, t, k] <= w[depot, t, k], name=f"depot_use_e{e}_t{t}_k{k}")

    # connectivity: cutset only
    _add_cutset_connectivity(m=m, V=V, depot=depot, T=T, K=K, E=E, ER=ER, x=x, y=y)

    m.optimize()

    if require_optimal and m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Arc-based check model did not reach OPTIMAL (status={m.Status})")

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Arc-based check model ended with status {m.Status}")

    has_solution = int(m.SolCount) > 0
    obj_val = float(m.ObjVal) if has_solution else float("inf")
    obj_bound = float(m.ObjBound) if m.ObjBound is not None else float("nan")
    gap_pct: float | None = None
    if has_solution and abs(obj_val) > 1e-12 and obj_bound == obj_bound:
        gap_pct = max(0.0, (obj_val - obj_bound) / abs(obj_val) * 100.0)
    elif m.Status == GRB.OPTIMAL and has_solution:
        gap_pct = 0.0

    chosen_schedules: Dict[Edge, int] = {}
    chosen_vehicle_by_edge: Dict[Edge, int] = {}
    serviced: Dict[Tuple[Edge, int, int], int] = {}
    deadhead: Dict[Tuple[Edge, int, int], int] = {}
    discount_active: Dict[Tuple[Edge, int], int] = {}

    for e in ER:
        for p_idx in range(len(P[e])):
            for k in K:
                if s[e, p_idx, k].X > 0.5:
                    chosen_schedules[e] = p_idx
                    chosen_vehicle_by_edge[e] = k

    for e in ER:
        ei = E.index(e)
        for t in T:
            for k in K:
                if x[ei, t, k].X > 0.5:
                    serviced[(e, t, k)] = int(round(x[ei, t, k].X))

    for ei, e in enumerate(E):
        for t in T:
            for k in K:
                val = int(round(y[ei, t, k].X))
                if val > 0:
                    deadhead[(e, t, k)] = val

    for ei in nonreq_idx:
        e = idx_to_edge[ei]
        for k in K:
            val = int(round(z[ei, k].X))
            if val > 0:
                discount_active[(e, k)] = val

    return {
        "status": int(m.Status),
        "objective": obj_val,
        "best_bound": obj_bound,
        "gap_pct": gap_pct,
        "connectivity_model": "cutset",
        "chosen_schedules": chosen_schedules,
        "chosen_vehicle_by_edge": chosen_vehicle_by_edge,
        "serviced": serviced,
        "deadhead": deadhead,
        "discount_active": discount_active,
    }


def solve_arc_based_pcarp_optimal(instance: Dict[str, Any]) -> Dict[str, Any]:
    return solve_arc_based_pcarp(
        instance=instance,
        time_limit=None,
        require_optimal=True,
        mip_gap=0.0,
    )


if __name__ == "__main__":
    inst = build_tiny_instance()
    out = solve_arc_based_pcarp(inst)
    print("status:", out["status"])
    print("objective:", out["objective"])
    print("connectivity_model:", out["connectivity_model"])
    print("chosen_schedules:", out["chosen_schedules"])
