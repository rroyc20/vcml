from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB


Edge = Tuple[int, int]


def _canon_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


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


def _support_components(
    nodes: Sequence[int],
    active_edges: Sequence[Edge],
) -> List[set[int]]:
    adj: Dict[int, List[int]] = {int(i): [] for i in nodes}
    for u, v in active_edges:
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))

    seen: set[int] = set()
    comps: List[set[int]] = []
    for start in nodes:
        if start in seen or not adj.get(int(start)):
            continue
        stack = [int(start)]
        comp: set[int] = set()
        seen.add(int(start))
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in adj.get(cur, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        comps.append(comp)
    return comps


def _separate_lazy_cutsets(model: gp.Model) -> None:
    x_vals = model.cbGetSolution(model._lazy_x)
    y_vals = model.cbGetSolution(model._lazy_y)

    for t in model._lazy_T:
        for k in model._lazy_K:
            active_edges: List[Edge] = []
            serviced_required_inside: List[Edge] = []

            for ei, edge in model._lazy_idx_to_edge.items():
                x_used = False
                if ei in model._lazy_required_idx_set:
                    x_used = x_vals[ei, t, k] > 0.5
                    if x_used:
                        serviced_required_inside.append(edge)
                if x_used or y_vals[ei, t, k] > 0.5:
                    active_edges.append(edge)

            if not active_edges:
                continue

            for comp in _support_components(model._lazy_V, active_edges):
                if model._lazy_depot in comp:
                    continue

                req_inside = [e for e in serviced_required_inside if e[0] in comp and e[1] in comp]
                if not req_inside:
                    continue

                delta_all = _delta(model._lazy_E, list(comp))
                delta_req = [e for e in delta_all if e in model._lazy_er_set]
                if not delta_all:
                    witness = req_inside[0]
                    witness_idx = model._lazy_edge_to_idx[witness]
                    model.cbLazy(model._lazy_x[witness_idx, t, k] <= 0.0)
                    model._lazy_cuts_added += 1
                    continue

                lhs = gp.quicksum(
                    model._lazy_x[model._lazy_edge_to_idx[e], t, k] for e in delta_req
                ) + gp.quicksum(
                    model._lazy_y[model._lazy_edge_to_idx[e], t, k] for e in delta_all
                )
                witness = req_inside[0]
                witness_idx = model._lazy_edge_to_idx[witness]
                model.cbLazy(lhs >= 2.0 * model._lazy_x[witness_idx, t, k])
                model._lazy_cuts_added += 1


def _lazy_cutset_callback(model: gp.Model, where: int) -> None:
    if where != GRB.Callback.MIPSOL:
        return
    _separate_lazy_cutsets(model)


def solve_arc_based_pcarp(
    instance: Dict[str, Any],
    time_limit: float | None = None,
    require_optimal: bool = False,
    mip_gap: float = 0.0,
    connectivity_model: str = "cutset",
) -> Dict[str, Any]:
    connectivity_model = str(connectivity_model).lower()
    if connectivity_model in {"lazy", "lazy_cutset"}:
        connectivity_model = "cutset"
    if connectivity_model not in {"cutset", "flow"}:
        raise ValueError(f"Unknown arc connectivity model: {connectivity_model!r}")

    V = instance["nodes"]
    depot = instance["depot"]
    T = instance["periods"]
    K = instance["vehicles"]
    # Prefer the original sparse road graph when provided. The metric closure is
    # useful for pricing-based methods, but it makes the arc MIP artificially dense.
    E = list(instance.get("arc_sparse_edges") or instance["edges"])
    ER = instance["required_edges"]
    r = dict(instance.get("arc_sparse_travel_cost") or instance["travel_cost"])
    c = instance["service_cost"]
    q = instance["demand"]
    Q = float(instance["capacity"])
    P = instance["schedule_patterns"]
    theta = float(instance.get("discount_theta", 0.0))

    e_idx = list(range(len(E)))
    edge_to_idx = {e: idx for idx, e in enumerate(E)}
    missing_required = [e for e in ER if e not in edge_to_idx]
    if missing_required:
        raise ValueError(
            "Arc graph is missing required edge(s): "
            + ", ".join(str(e) for e in missing_required[:5])
        )
    er_idx = [edge_to_idx[e] for e in ER]
    idx_to_edge = {i: E[i] for i in e_idx}
    er_set = set(ER)
    nonreq_idx = [ei for ei in e_idx if idx_to_edge[ei] not in er_set]

    m = gp.Model("pcarp_arc_based_check")
    m.Params.OutputFlag = int(instance.get("arc_gurobi_output", instance.get("gurobi_output", 0)))
    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = float(mip_gap)
    if connectivity_model == "cutset":
        m.Params.LazyConstraints = 1

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
        ei = edge_to_idx[e]
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
                lhs = gp.quicksum(x[edge_to_idx[e], t, k] for e in delta_req_i) + gp.quicksum(
                    y[edge_to_idx[e], t, k] for e in delta_i
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

    if connectivity_model == "flow":
        # Single-commodity flow from the depot to serviced required-edge endpoints.
        # This replaces lazy cutset separation with explicit connectivity constraints.
        directed_arcs: List[Tuple[int, int]] = []
        directed_arc_to_edge_idx: Dict[Tuple[int, int], int] = {}
        incoming: Dict[int, List[Tuple[int, int]]] = {int(i): [] for i in V}
        outgoing: Dict[int, List[Tuple[int, int]]] = {int(i): [] for i in V}
        for ei, (u, v) in enumerate(E):
            directed_arcs.append((u, v))
            directed_arcs.append((v, u))
            directed_arc_to_edge_idx[u, v] = ei
            directed_arc_to_edge_idx[v, u] = ei
            outgoing[int(u)].append((u, v))
            incoming[int(v)].append((u, v))
            outgoing[int(v)].append((v, u))
            incoming[int(u)].append((v, u))

        flow_M = float(max(1, len(ER)))
        f = m.addVars(directed_arcs, T, K, vtype=GRB.CONTINUOUS, lb=0.0, ub=flow_M, name="f")
        required_idx_set = set(er_idx)
        required_incident_by_node = {
            int(i): [edge_to_idx[e] for e in _delta(ER, [i])] for i in V
        }

        for t in T:
            for k in K:
                for u, v in directed_arcs:
                    ei = directed_arc_to_edge_idx[u, v]
                    edge_use = y[ei, t, k]
                    if ei in required_idx_set:
                        edge_use = edge_use + x[ei, t, k]
                    m.addConstr(
                        f[u, v, t, k] <= flow_M * edge_use,
                        name=f"flow_link_{u}_{v}_t{t}_k{k}",
                    )

                nondepot_demand = gp.quicksum(
                    0.5 * x[ei, t, k]
                    for i in V
                    if int(i) != int(depot)
                    for ei in required_incident_by_node[int(i)]
                )
                for i in V:
                    i_int = int(i)
                    net_in = gp.quicksum(f[u, v, t, k] for u, v in incoming[i_int]) - gp.quicksum(
                        f[u, v, t, k] for u, v in outgoing[i_int]
                    )
                    if i_int == int(depot):
                        m.addConstr(net_in == -nondepot_demand, name=f"flow_src_t{t}_k{k}")
                    else:
                        node_demand = gp.quicksum(
                            0.5 * x[ei, t, k] for ei in required_incident_by_node[i_int]
                        )
                        m.addConstr(net_in == node_demand, name=f"flow_bal_i{i}_t{t}_k{k}")

        m._lazy_cuts_added = 0
        m.optimize()
    else:
        # connectivity: cutset separation via lazy constraints
        m._lazy_V = list(V)
        m._lazy_depot = int(depot)
        m._lazy_T = list(T)
        m._lazy_K = list(K)
        m._lazy_E = list(E)
        m._lazy_er_set = set(ER)
        m._lazy_edge_to_idx = dict(edge_to_idx)
        m._lazy_idx_to_edge = dict(idx_to_edge)
        m._lazy_required_idx_set = set(er_idx)
        m._lazy_x = x
        m._lazy_y = y
        m._lazy_cuts_added = 0

        m.optimize(_lazy_cutset_callback)

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

    if not has_solution:
        return {
            "status": int(m.Status),
            "objective": obj_val,
            "best_bound": obj_bound,
            "gap_pct": gap_pct,
            "connectivity_model": "lazy_cutset" if connectivity_model == "cutset" else "flow",
            "lazy_cuts_added": int(getattr(m, "_lazy_cuts_added", 0)),
            "chosen_schedules": {},
            "chosen_vehicle_by_edge": {},
            "serviced": {},
            "deadhead": {},
            "discount_active": {},
        }

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
        ei = edge_to_idx[e]
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
        "connectivity_model": "lazy_cutset" if connectivity_model == "cutset" else "flow",
        "lazy_cuts_added": int(getattr(m, "_lazy_cuts_added", 0)),
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
        connectivity_model="cutset",
    )


def solve_arc_based_pcarp_flow(
    instance: Dict[str, Any],
    time_limit: float | None = None,
    require_optimal: bool = False,
    mip_gap: float = 0.0,
) -> Dict[str, Any]:
    return solve_arc_based_pcarp(
        instance=instance,
        time_limit=time_limit,
        require_optimal=require_optimal,
        mip_gap=mip_gap,
        connectivity_model="flow",
    )


def solve_arc_based_pcarp_flow_optimal(instance: Dict[str, Any]) -> Dict[str, Any]:
    return solve_arc_based_pcarp_flow(
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
