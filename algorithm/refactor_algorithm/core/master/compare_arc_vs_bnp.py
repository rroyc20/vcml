from __future__ import annotations

import random
import math
import copy
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from refactor_algorithm.core.master.arc_based_check import solve_arc_based_pcarp_optimal
from refactor_algorithm.core.master.separation import (
    CapacityLinkSeparator,
    SeparationRoundResult,
    SeparationManager,
    SRI3Separator,
)
from refactor_algorithm.core.pricing.node import BnBConfig, BnBNode, BnBTree, BestBoundSelector, DepthFirstSelector
from refactor_algorithm.core.util.alns import run_alns_initial_solution
from refactor_algorithm.core.util.initial_heuristic import (
    add_cover_artificials,
    canon_edge,
)


Edge = Tuple[int, int]


def _canon_edge(i: int, j: int) -> Edge:
    return canon_edge(i, j)


def discount_objective_cost_per_edge(
    inst: Optional[Dict[str, Any]],
    e: Edge,
    travel_cost: Dict[Edge, float],
) -> float:
    """Unit cost in -θ|T|·c·z (or y) for consistency discount on physical non-required edge e.

    Uses sparse/road link cost when present so the reward matches an actual road segment,
    not an inflated metric-closure chord distance (which can make the objective very negative).
    """
    if isinstance(inst, dict):
        for key in ("arc_sparse_travel_cost", "road_sparse_travel_cost"):
            d = inst.get(key)
            if isinstance(d, dict) and e in d:
                return float(d[e])
    return float(travel_cost.get(e, 0.0))


def path_arcs_travel_total(
    inst: Optional[Dict[str, Any]],
    path_arcs: Sequence[Any],
    travel_cost: Dict[Edge, float],
) -> float:
    """Sum travel cost along path_arcs.

    When ``inst["yao_style_pricing"]`` is set and ``arc_sparse_travel_cost`` is
    populated, use physical sparse arc costs for arcs present there (Yao-style
    road-layer paths); otherwise use ``travel_cost`` (metric closure).
    """
    sparse: Optional[Dict[Edge, float]] = None
    if isinstance(inst, dict) and int(inst.get("yao_style_pricing", 0)):
        raw = inst.get("road_sparse_travel_cost") or inst.get("arc_sparse_travel_cost")
        if isinstance(raw, dict) and raw:
            sparse = raw
    total = 0.0
    for a in path_arcs:
        if not isinstance(a, tuple) or len(a) < 2:
            continue
        e = _canon_edge(int(a[0]), int(a[1]))
        if sparse is not None and e in sparse:
            total += float(sparse[e])
        else:
            total += float(travel_cost.get(e, 0.0))
    return total


def _directed_arc_id(i: int, j: int) -> Tuple[int, int]:
    return (i, j)


def _edge_endpoints(e: Edge) -> Tuple[int, int]:
    return e[0], e[1]


def _build_complete_edges(nodes: Sequence[int]) -> List[Edge]:
    return [_canon_edge(i, j) for i, j in combinations(nodes, 2)]


def _shortest_path_direct(i: int, j: int, edge_cost: Dict[Edge, float]) -> Tuple[List[Tuple[int, int]], float]:
    if i == j:
        return [], 0.0
    e = _canon_edge(i, j)
    return [(_directed_arc_id(i, j))], float(edge_cost[e])


def _single_service_route(
    depot: int,
    required_edge: Edge,
    edge_cost: Dict[Edge, float],
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Build a cheap depot -> service(required edge) -> depot route.
    Cost returned here is deadheading-only travel cost.
    """
    i, j = _edge_endpoints(required_edge)

    p1_a, c1_a = _shortest_path_direct(depot, i, edge_cost)
    p1_b, c1_b = _shortest_path_direct(j, depot, edge_cost)
    route1 = p1_a + [(_directed_arc_id(i, j))] + p1_b
    cost1 = c1_a + c1_b

    p2_a, c2_a = _shortest_path_direct(depot, j, edge_cost)
    p2_b, c2_b = _shortest_path_direct(i, depot, edge_cost)
    route2 = p2_a + [(_directed_arc_id(j, i))] + p2_b
    cost2 = c2_a + c2_b

    if cost1 <= cost2:
        return route1, cost1
    return route2, cost2


def create_random_pcarp_instance(seed: int = 7) -> Dict[str, Any]:
    random.seed(seed)

    nodes = list(range(10))  # 6 nodes
    depot = 0
    periods = [0, 1, 2, 3]  # 4 days
    vehicles = [0, 1, 2, 3]  # m_t = 4

    edges = _build_complete_edges(nodes)

    # random symmetric travel costs
    travel_cost: Dict[Edge, float] = {e: float(random.randint(2, 10)) for e in edges}

    # choose required edges
    candidates = [e for e in edges if depot not in e]
    random.shuffle(candidates)
    required_edges = candidates[:6]

    # service cost in arc-based objective should equal (travel-on-service-edge + extra-service)
    # so that pricing objective travel + service_extra matches arc-based objective.
    service_extra: Dict[Edge, float] = {e: float(random.randint(1, 5)) for e in required_edges}
    service_cost: Dict[Edge, float] = {e: float(travel_cost[e] + service_extra[e]) for e in required_edges}
    demand: Dict[Edge, float] = {e: 1.0 for e in required_edges}

    # per required edge, random regular schedules:
    # either [[0,2],[1,3]] or [[0],[1],[2],[3]]
    schedule_patterns: Dict[Edge, List[frozenset[int]]] = {}
    for e in required_edges:
        if random.random() < 0.5:
            schedule_patterns[e] = [frozenset([0, 2]), frozenset([1, 3])]
        else:
            schedule_patterns[e] = [
                frozenset([0]),
                frozenset([1]),
                frozenset([2]),
                frozenset([3]),
            ]

    return {
        "nodes": nodes,
        "depot": depot,
        "periods": periods,
        "vehicles": vehicles,
        "edges": edges,
        "required_edges": required_edges,
        "travel_cost": travel_cost,
        "service_cost": service_cost,
        "service_extra": service_extra,
        "demand": demand,
        "capacity": 1.0,  # forces single serviced edge per route in this checker
        "schedule_patterns": schedule_patterns,
    }


def _apply_inspect_bnp_defaults(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill algorithmic defaults so compare runs follow inspect_bnp.py by default.

    Keep instance data intact and only backfill missing solver/pricing settings.
    """
    out = dict(inst)
    defaults: Dict[str, Any] = {
        "max_cg_iterations_per_node": 10000,
        "max_nodes": 999999,
        "algorithm_time_limit_s": 0.0,
        "pricing_method": "cpp_ng",
        "pricing_ng_size": 10,
        "cpp_ng_empty_fallback": "none",
        "cut_pricing_mode": "auto",
        "cut_pricing_dual_tol": 1e-15,
        "use_coeff_dominance_filter": 0,
        "coeff_dom_obj_tol": 1e-9,
        "node_search_strategy": "best_bound",
        "eps_reduced_cost": 1e-4,
        "use_dual_stabilization": 0,
        "dual_stab_alpha": 0.5,
        "dual_stab_alpha_decay": 0.9,
        "dual_stab_min_alpha": 0.0,
        "use_ub_zero_branching": 0,
        "partial_pricing_ratio": 1.0,
        "phase1_col_cap": 1000,
        "discount_theta": 0.0,
        "use_alns_initialization": 0,
        "yao_style_pricing": 1,
        "use_transformed_pricing_graph": 1,
        "use_aggregation": 0,
        "use_vehicle_lex_symmetry": 1,
        "use_capacity_cuts": 0,
        "use_sri_cuts": 0,
        "sri_cardinality": 3,
        "enable_sri": 1,
        "root_only_sri": 1,
        "max_sri_rounds": 3,
        "max_cuts_per_round": 20,
        "max_cuts_per_day": 5,
        "min_sri_violation": 1e-4,
        "enable_sri_similarity_filter": 1,
        "max_shared_edges_between_sri3": 1,
        "cut_root_only": 0,
        "cut_separation_max_depth": 0,
    }
    for key, value in defaults.items():
        out.setdefault(key, value)
    return out


@dataclass
class RouteColumn:
    day: int
    driver: int
    serviced_required_edges: Tuple[Edge, ...]
    path_arcs: Tuple[Tuple[int, int], ...]
    cost: float
    nonrequired_edges_used: Tuple[Edge, ...] = ()


class SimpleSPMaster:
    """
    Small set-partitioning master wrapper compatible with BnBNode.
    Designed for validation runs (not production).
    """

    def __init__(self, inst: Dict[str, Any]) -> None:
        self.inst = inst
        self.model = gp.Model("sp_master_checker")
        self.model.Params.OutputFlag = int(inst.get("alg_gurobi_output", inst.get("gurobi_output", 0)))

        self.days = list(inst["periods"])
        self.vehicles = list(inst["vehicles"])
        self.contexts: List[Tuple[int, int]] = [(int(t), int(k)) for t in self.days for k in self.vehicles]
        self.edges: List[Edge] = list(inst["edges"])
        self.required_edges: List[Edge] = list(inst["required_edges"])
        self.required_edge_set: set[Edge] = set(self.required_edges)
        # Discount is defined only on physical sparse edges, not on metric-closure
        # chords between selected nodes.
        _sparse_edge_set: set = set(inst.get("arc_sparse_edges", []))
        self.nonrequired_edges: List[Edge] = [
            e for e in self.edges
            if e not in self.required_edge_set and e in _sparse_edge_set
        ] if _sparse_edge_set else [e for e in self.edges if e not in self.required_edge_set]
        self.nonrequired_edge_set: set[Edge] = set(self.nonrequired_edges)
        self.capacity = float(inst["capacity"])
        self.depot = int(inst["depot"])
        self.travel_cost: Dict[Edge, float] = inst["travel_cost"]
        self.service_extra: Dict[Edge, float] = inst["service_extra"]
        self.schedule_patterns: Dict[Edge, List[frozenset[int]]] = inst["schedule_patterns"]
        self.discount_theta: float = float(inst.get("discount_theta", 0.0))
        self.m_t: Dict[int, int] = {t: len(self.vehicles) for t in self.days}

        self.schedule_var_name: Dict[Tuple[Edge, int, int], str] = {}
        self.cover_constr_name: Dict[Tuple[Edge, int, int], str] = {}
        self.veh_constr_name: Dict[Tuple[int, int], str] = {}
        # (day t, k_lo, k_hi): Σ_r λ^{t,k_hi}_r - Σ_r λ^{t,k_lo}_r <= 0  (sorted k_lo < k_hi)
        self.veh_lex_constr_name: Dict[Tuple[int, int, int], str] = {}
        self.use_vehicle_lex_symmetry: bool = bool(int(self.inst.get("use_vehicle_lex_symmetry", 1)))
        self._vehicles_sorted: List[int] = sorted(int(k) for k in self.vehicles)
        self.discount_z_var_name: Dict[Tuple[Edge, int], str] = {}
        self.discount_link_constr_name: Dict[Tuple[Edge, int, int], str] = {}
        self.lambda_var_names_by_day: Dict[Tuple[int, int], List[str]] = {
            (int(t), int(k)): [] for t in self.days for k in self.vehicles
        }
        self.lambda_var_name_to_index: Dict[str, int] = {}
        self.column_signatures: set = set()
        self.route_columns: List[RouteColumn] = []
        self.column_signature_meta: Dict[Tuple[Any, ...], Dict[str, float]] = {}
        # Coefficient-equivalent column dominance:
        # same master coefficients -> keep best (lowest objective) route only.
        self.column_coeff_best_obj: Dict[Tuple[Any, ...], float] = {}
        self.use_coeff_dominance_filter: bool = bool(int(self.inst.get("use_coeff_dominance_filter", 1)))
        self.coeff_dom_obj_tol: float = abs(float(self.inst.get("coeff_dom_obj_tol", 1e-9)))
        self.column_pool_dominated: int = 0
        self._last_add_route_status: str = "none"
        self.column_pool_hits: int = 0
        self.column_pool_misses: int = 0
        self._column_pool_tick: int = 0
        self.artificial_var_name_by_cover: Dict[str, str] = {}
        # Tracks aggregate branching constraints so newly generated columns participate.
        # Key: tuple identifying the constraint type and scope, e.g.:
        #   ('whole_route',)
        #   ('daily_route', t, k)
        #   ('visit_node', node_id, t, k)
        #   ('visit_arc', canon_edge, t, k)
        # Value: Gurobi constraint name string
        self.aggregate_branch_constrs: Dict[tuple, str] = {}
        self._aggregate_constr_handles: Dict[tuple, Any] = {}
        self.capacity_cuts_added: int = 0
        self.sri_cuts_added: int = 0
        self.separation_manager: Optional[SeparationManager] = None
        self.sri_separation_manager: Optional[SeparationManager] = None
        self.initial_incumbent: Optional[Dict[str, Any]] = None
        # Performance: avoid repeated name-based lookups during column add / pricing data prep.
        self._constr_by_name: Dict[str, Any] = {}
        self._var_by_name: Dict[str, Any] = {}
        # Branching caches are maintained incrementally so node-level branching no longer
        # rescans all schedule vars / route columns on every solve.
        self._branch_edge_driver_assign_expr: Dict[Tuple[Edge, int], Dict[str, float]] = {}
        self._branch_edge_day_driver_service_expr: Dict[Tuple[Edge, int, int], Dict[str, float]] = {}
        self._branch_arc_visit_expr: Dict[Tuple[Any, int, int], Dict[str, float]] = {}
        self._branch_node_visit_expr: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self._branch_ryan_foster_pair_expr: Dict[Tuple[Edge, Edge, int], Dict[str, float]] = {}
        self._branch_schedule_vars_by_edge: Dict[Edge, List[str]] = {}
        self._branch_schedule_vars_by_edge_pattern: Dict[Tuple[Edge, int], List[str]] = {}
        self._branch_schedule_pattern_sum_expr: Dict[Tuple[Edge, int], Dict[str, float]] = {}
        self._branch_schedule_vars_by_edge_driver: Dict[Tuple[Edge, int], List[str]] = {}
        self._branch_schedule_vars_by_edge_day_driver: Dict[Tuple[Edge, int, int], List[str]] = {}

        self._build_base_model()
        self._add_initial_columns()
        if bool(self.inst.get("use_cutting_plane_separation", 0)):
            separators = []
            sri_separators = []
            if bool(self.inst.get("use_capacity_cuts", 0)):
                separators.append(CapacityLinkSeparator())
            if bool(self.inst.get("use_sri_cuts", 0)):
                sri_separators.append(SRI3Separator(cardinality=int(self.inst.get("sri_cardinality", 3))))
            if separators:
                self.separation_manager = SeparationManager(
                    separators=separators,
                    tol=float(self.inst.get("cut_separation_tol", 1e-7)),
                    max_rounds=int(self.inst.get("cut_max_rounds_per_solve", 50)),
                )
            if sri_separators:
                self.sri_separation_manager = SeparationManager(
                    separators=sri_separators,
                    tol=float(self.inst.get("cut_separation_tol", 1e-7)),
                    max_rounds=1,
                )
        elif bool(self.inst.get("use_capacity_cuts", 0)):
            self._add_capacity_type_cuts()

    def _build_base_model(self) -> None:
        m = self.model
        discount_weight = float(self.discount_theta) * float(len(self.days))

        # schedule vars
        s_var: Dict[Tuple[Edge, int, int], Any] = {}
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            for p_idx, _ in enumerate(pats):
                for k in self.vehicles:
                    # RMP is solved as LP relaxation during branch-and-price.
                    v = m.addVar(
                        lb=0.0,
                        ub=1.0,
                        vtype=GRB.CONTINUOUS,
                        name=f"s_{e[0]}_{e[1]}_p{p_idx}_k{k}",
                    )
                    s_var[(e, p_idx, k)] = v
        m.update()
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            for p_idx, _ in enumerate(pats):
                for k in self.vehicles:
                    vname = s_var[(e, p_idx, k)].VarName
                    self.schedule_var_name[(e, p_idx, k)] = vname
                    self._register_branching_schedule_var(e, p_idx, int(k), vname)

        # exactly one (schedule, driver) assignment per required edge
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            m.addConstr(
                gp.quicksum(s_var[(e, p_idx, k)] for p_idx in range(len(pats)) for k in self.vehicles) == 1.0,
                name=f"sched_{e}",
            )

        # cover constraints by (edge, day, driver), lambda part added via columns
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            for t in self.days:
                for k in self.vehicles:
                    expr = -gp.quicksum(s_var[(e, p_idx, k)] for p_idx, pat in enumerate(pats) if t in pat)
                    cname = f"cover_{e}_t{t}_k{k}"
                    m.addConstr(expr == 0.0, name=cname)
                    self.cover_constr_name[(e, int(t), int(k))] = cname

        # one route per (day, driver)
        for t in self.days:
            for k in self.vehicles:
                cname = f"veh_t{t}_k{k}"
                m.addConstr(gp.LinExpr() <= 1.0, name=cname)
                self.veh_constr_name[(int(t), int(k))] = cname

        # 차량 인덱스 사전순 사용: 작은 k가 큰 k 이상으로 “사용량”을 가짐
        #   Σ_r λ^{t,k_lo}_r >= Σ_r λ^{t,k_hi}_r  ⇔  Σ λ_hi - Σ λ_lo <= 0
        if self.use_vehicle_lex_symmetry and len(self._vehicles_sorted) >= 2:
            vs = self._vehicles_sorted
            for t in self.days:
                for i in range(len(vs) - 1):
                    k_lo, k_hi = int(vs[i]), int(vs[i + 1])
                    cname = f"veh_lex_t{int(t)}_k{k_lo}_ge_k{k_hi}"
                    m.addConstr(gp.LinExpr() <= 0.0, name=cname)
                    self.veh_lex_constr_name[(int(t), k_lo, k_hi)] = cname

        # Discount activation variables for nonrequired edge day-consistency:
        # z[e,k] can turn on only if every day route set traverses edge e at least once.
        discount_z_var: Dict[Tuple[Edge, int], Any] = {}
        for e in self.nonrequired_edges:
            for k in self.vehicles:
                c_disc = discount_objective_cost_per_edge(self.inst, e, self.travel_cost)
                obj_coef = -discount_weight * float(c_disc)
                zv = m.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    obj=obj_coef,
                    name=f"z_{e[0]}_{e[1]}_k{int(k)}",
                )
                discount_z_var[(e, int(k))] = zv

        # For each nonrequired edge/day/driver: sum_r b_{e,r}^{t,k} * lambda_r^{t,k} >= z[e,k]
        # (equivalently z - sum lambda <= 0). Wrong way (lambda - z <= 0) lets z=1 with lambda=0.
        for e in self.nonrequired_edges:
            for t in self.days:
                for k in self.vehicles:
                    expr = gp.LinExpr()
                    expr += 1.0 * discount_z_var[(e, int(k))]
                    cname = f"disc_link_{e}_t{int(t)}_k{int(k)}"
                    m.addConstr(expr <= 0.0, name=cname)
                    self.discount_link_constr_name[(e, int(t), int(k))] = cname

        # Phase-I safety net: make root RMP always feasible.
        self.artificial_var_name_by_cover = add_cover_artificials(
            model=m,
            cover_constr_names=self.cover_constr_name.values(),
            penalty=1e5,
        )

        m.update()
        for key, var in discount_z_var.items():
            self.discount_z_var_name[key] = var.VarName

        m.ModelSense = GRB.MINIMIZE
        m.update()

    def _register_branching_schedule_var(self, e: Edge, p_idx: int, k: int, vname: str) -> None:
        e_can = _canon_edge(e[0], e[1])
        self._branch_schedule_vars_by_edge.setdefault(e_can, []).append(vname)
        self._branch_schedule_vars_by_edge_pattern.setdefault((e_can, int(p_idx)), []).append(vname)
        q_expr = self._branch_schedule_pattern_sum_expr.setdefault((e_can, int(p_idx)), {})
        q_expr[vname] = q_expr.get(vname, 0.0) + 1.0
        self._branch_schedule_vars_by_edge_driver.setdefault((e_can, int(k)), []).append(vname)
        z_expr = self._branch_edge_driver_assign_expr.setdefault((e_can, int(k)), {})
        z_expr[vname] = z_expr.get(vname, 0.0) + 1.0

        pat = self.schedule_patterns[e_can][int(p_idx)]
        for t in pat:
            key = (e_can, int(t), int(k))
            self._branch_schedule_vars_by_edge_day_driver.setdefault(key, []).append(vname)
            x_expr = self._branch_edge_day_driver_service_expr.setdefault(key, {})
            x_expr[vname] = x_expr.get(vname, 0.0) + 1.0

    def _register_branching_route_var(
        self,
        *,
        day: int,
        driver: int,
        vname: str,
        serviced_edges: Sequence[Edge],
        path_arcs: Sequence[Tuple[int, int]],
    ) -> None:
        t_int = int(day)
        k_int = int(driver)
        arc_cnt: Dict[Edge, float] = {}
        node_cnt: Dict[int, float] = {}
        served_unique = tuple(sorted({_canon_edge(e[0], e[1]) for e in serviced_edges}))

        for arc in path_arcs:
            if not (isinstance(arc, tuple) and len(arc) >= 2):
                continue
            e_can = _canon_edge(arc[0], arc[1])
            arc_cnt[e_can] = arc_cnt.get(e_can, 0.0) + 1.0
            node_id = int(arc[1])
            node_cnt[node_id] = node_cnt.get(node_id, 0.0) + 1.0

        for e_can, coeff in arc_cnt.items():
            expr = self._branch_arc_visit_expr.setdefault((e_can, t_int, k_int), {})
            expr[vname] = expr.get(vname, 0.0) + float(coeff)

        for node_id, coeff in node_cnt.items():
            expr = self._branch_node_visit_expr.setdefault((node_id, t_int, k_int), {})
            expr[vname] = expr.get(vname, 0.0) + float(coeff)

        for i in range(len(served_unique)):
            e_i = served_unique[i]
            for j in range(i + 1, len(served_unique)):
                e_j = served_unique[j]
                expr = self._branch_ryan_foster_pair_expr.setdefault((e_i, e_j, t_int), {})
                expr[vname] = expr.get(vname, 0.0) + 1.0

    def _column_cost(self, path_arcs: Sequence[Tuple[int, int]], serviced_edges: Sequence[Edge]) -> float:
        travel = path_arcs_travel_total(
            getattr(self, "inst", None), path_arcs, self.travel_cost
        )
        serv = sum(float(self.service_extra[e]) for e in serviced_edges)
        return travel + serv

    def _column_coeff_signature(
        self,
        day: int,
        driver: int,
        serviced_edges: Sequence[Edge],
        nonreq_arc_used: Sequence[Edge],
    ) -> Tuple[Any, ...]:
        return (
            int(day),
            int(driver),
            tuple(sorted(serviced_edges)),
            tuple(sorted(nonreq_arc_used)),
        )

    def _add_route_var(
        self,
        day: int,
        driver: int,
        serviced_edges: Sequence[Edge],
        path_arcs: Sequence[Tuple[int, int]],
        nonrequired_edges_used: Optional[Sequence[Edge]] = None,
        active_aggregate_constrs: Optional[Sequence[Tuple[tuple, Any]]] = None,
    ) -> Optional[Any]:
        self._last_add_route_status = "none"
        allowed_nonreq = self.nonrequired_edge_set
        nonreq_arc_used: set = set()
        if nonrequired_edges_used is not None:
            for e in nonrequired_edges_used:
                if not (isinstance(e, tuple) and len(e) >= 2):
                    continue
                e_can = _canon_edge(int(e[0]), int(e[1]))
                if e_can in self.required_edge_set or (allowed_nonreq and e_can not in allowed_nonreq):
                    continue
                nonreq_arc_used.add(e_can)
        else:
            for a in path_arcs:
                if not (isinstance(a, tuple) and len(a) >= 2):
                    continue
                e_can = _canon_edge(int(a[0]), int(a[1]))
                if e_can in self.required_edge_set or (allowed_nonreq and e_can not in allowed_nonreq):
                    continue
                nonreq_arc_used.add(e_can)

        coeff_sig = self._column_coeff_signature(day, driver, serviced_edges, tuple(nonreq_arc_used))
        cost = self._column_cost(path_arcs, serviced_edges)
        if self.use_coeff_dominance_filter:
            best_cost = self.column_coeff_best_obj.get(coeff_sig)
            if best_cost is not None and float(cost) + self.coeff_dom_obj_tol >= float(best_cost):
                self.column_pool_dominated += 1
                self._last_add_route_status = "dominated"
                return None

        sig = (int(day), int(driver), tuple(sorted(serviced_edges)), tuple(path_arcs))
        self._column_pool_tick += 1
        if sig in self.column_signatures:
            self.column_pool_hits += 1
            meta = self.column_signature_meta.setdefault(sig, {"seen": 0.0, "last_tick": 0.0})
            meta["seen"] = float(meta.get("seen", 0.0)) + 1.0
            meta["last_tick"] = float(self._column_pool_tick)
            self._last_add_route_status = "duplicate"
            return None
        self.column_pool_misses += 1
        self.column_signatures.add(sig)
        self.column_signature_meta[sig] = {"seen": 1.0, "last_tick": float(self._column_pool_tick)}

        m = self.model
        col = gp.Column()
        t_int, k_int = int(day), int(driver)

        for e in serviced_edges:
            cname = self.cover_constr_name[(e, t_int, k_int)]
            c = self._get_constr_cached(cname)
            if c is not None:
                col.addTerms(1.0, c)

        veh_c = self._get_constr_cached(self.veh_constr_name[(t_int, k_int)])
        if veh_c is not None:
            col.addTerms(1.0, veh_c)

        if self.use_vehicle_lex_symmetry and len(self._vehicles_sorted) >= 2:
            vs = self._vehicles_sorted
            for i in range(len(vs) - 1):
                k_lo, k_hi = int(vs[i]), int(vs[i + 1])
                cname = self.veh_lex_constr_name.get((t_int, k_lo, k_hi))
                if cname is None:
                    continue
                lex_c = self._get_constr_cached(cname)
                if lex_c is None:
                    continue
                if k_int == k_hi:
                    col.addTerms(1.0, lex_c)
                elif k_int == k_lo:
                    col.addTerms(-1.0, lex_c)

        # Discount link coefficients: binary (route uses edge or not).
        for e in nonreq_arc_used:
            cname = self.discount_link_constr_name.get((e, t_int, k_int))
            if cname is None:
                continue
            c = self._get_constr_cached(cname)
            if c is not None:
                col.addTerms(-1.0, c)

        if self.aggregate_branch_constrs:
            arc_counts: Dict[Edge, float] = {}
            node_counts: Dict[int, float] = {}
            served_set = set(serviced_edges)
            for arc in path_arcs:
                if isinstance(arc, tuple) and len(arc) >= 2:
                    e_c = _canon_edge(arc[0], arc[1])
                    arc_counts[e_c] = arc_counts.get(e_c, 0.0) + 1.0
                    node_counts[arc[1]] = node_counts.get(arc[1], 0.0) + 1.0

            cap_neg = -float(self.capacity)
            agg_items = active_aggregate_constrs
            if agg_items is None:
                agg_items = self._active_aggregate_constr_items()

            for agg_key, constr in agg_items:
                kind = agg_key[0]
                coeff = 0.0
                if kind == "whole_route":
                    coeff = 1.0
                elif kind == "daily_route":
                    if agg_key[1] == t_int and agg_key[2] == k_int:
                        coeff = 1.0
                elif kind == "visit_node":
                    if agg_key[2] == t_int and agg_key[3] == k_int:
                        coeff = float(node_counts.get(agg_key[1], 0.0))
                elif kind == "visit_arc":
                    if agg_key[2] == t_int and agg_key[3] == k_int:
                        coeff = float(arc_counts.get(agg_key[1], 0.0))
                elif kind == "capacity_link_tk":
                    if agg_key[1] == t_int and agg_key[2] == k_int:
                        coeff = cap_neg
                elif kind == "capacity_link_t":
                    if agg_key[1] == t_int:
                        coeff = cap_neg
                elif kind == "sri3_t":
                    if agg_key[1] == t_int:
                        overlap = sum(1 for e in agg_key[2] if e in served_set)
                        if overlap >= 2:
                            coeff = 1.0
                elif kind == "ryan_foster_pair":
                    if len(agg_key) >= 4 and agg_key[3] == t_int:
                        if agg_key[1] in served_set and agg_key[2] in served_set:
                            coeff = 1.0
                if coeff != 0.0:
                    col.addTerms(coeff, constr)

        vname = f"lam_t{int(day)}_k{int(driver)}_r{len(self.route_columns)}"
        # Keep lambda continuous in RMP LP; exact fallback converts to binary.
        lam = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, obj=cost, name=vname, column=col)
        self._var_by_name[vname] = lam
        self.lambda_var_name_to_index[vname] = len(self.route_columns)
        self.lambda_var_names_by_day[(int(day), int(driver))].append(vname)
        self._register_branching_route_var(
            day=int(day),
            driver=int(driver),
            vname=vname,
            serviced_edges=serviced_edges,
            path_arcs=path_arcs,
        )
        self.route_columns.append(
            RouteColumn(
                day=int(day),
                driver=int(driver),
                serviced_required_edges=tuple(serviced_edges),
                path_arcs=tuple(path_arcs),
                cost=float(cost),
                nonrequired_edges_used=tuple(sorted(nonreq_arc_used)),
            )
        )
        old_best = self.column_coeff_best_obj.get(coeff_sig, float("inf"))
        if float(cost) < float(old_best):
            self.column_coeff_best_obj[coeff_sig] = float(cost)
        self._last_add_route_status = "added"
        return lam

    def _active_aggregate_constr_items(self) -> List[Tuple[tuple, Any]]:
        items: List[Tuple[tuple, Any]] = []
        if not self.aggregate_branch_constrs:
            return items
        for agg_key, cname in self.aggregate_branch_constrs.items():
            cn = str(cname)
            constr = self._aggregate_constr_handles.get(agg_key)
            if constr is None:
                constr = self._get_constr_cached(cn)
            if constr is None:
                self._aggregate_constr_handles.pop(agg_key, None)
                self._constr_by_name.pop(cn, None)
                continue
            self._constr_by_name[cn] = constr
            self._aggregate_constr_handles[agg_key] = constr
            items.append((agg_key, constr))
        return items

    def _add_capacity_type_cuts(self) -> None:
        """
        Add capacity-type valid inequalities linking schedule demand and route usage.

        (1) day-driver cut:
            sum_e d_e * x_{e,t,k} <= Q * sum_r lambda_{r}^{t,k}
        (2) day aggregate cut:
            sum_{e,k} d_e * x_{e,t,k} <= Q * sum_{k,r} lambda_{r}^{t,k}
        where x_{e,t,k} is represented by schedule vars s_{e,p,k}.
        """
        m = self.model
        added = 0

        # (1) day-driver capacity-link cuts
        for t in self.days:
            for k in self.vehicles:
                cname = f"cap_link_t{int(t)}_k{int(k)}"
                if m.getConstrByName(cname) is not None:
                    continue
                expr = gp.LinExpr()
                for e in self.required_edges:
                    dem = float(self.inst["demand"].get(e, 0.0))
                    if dem <= 0.0:
                        continue
                    pats = self.schedule_patterns[e]
                    for p_idx, pat in enumerate(pats):
                        if int(t) not in pat:
                            continue
                        vname = self.schedule_var_name.get((e, p_idx, int(k)))
                        sv = m.getVarByName(vname) if vname is not None else None
                        if sv is not None:
                            expr += dem * sv
                for lname in self.lambda_var_names_by_day.get((int(t), int(k)), []):
                    lv = m.getVarByName(lname)
                    if lv is not None:
                        expr += -float(self.capacity) * lv
                m.addConstr(expr <= 0.0, name=cname)
                self.register_aggregate_constr(("capacity_link_tk", int(t), int(k)), cname)
                added += 1

        # (2) day aggregate capacity-link cuts
        for t in self.days:
            cname = f"cap_link_t{int(t)}"
            if m.getConstrByName(cname) is not None:
                continue
            expr = gp.LinExpr()
            for e in self.required_edges:
                dem = float(self.inst["demand"].get(e, 0.0))
                if dem <= 0.0:
                    continue
                pats = self.schedule_patterns[e]
                for k in self.vehicles:
                    for p_idx, pat in enumerate(pats):
                        if int(t) not in pat:
                            continue
                        vname = self.schedule_var_name.get((e, p_idx, int(k)))
                        sv = m.getVarByName(vname) if vname is not None else None
                        if sv is not None:
                            expr += dem * sv
            for k in self.vehicles:
                for lname in self.lambda_var_names_by_day.get((int(t), int(k)), []):
                    lv = m.getVarByName(lname)
                    if lv is not None:
                        expr += -float(self.capacity) * lv
            m.addConstr(expr <= 0.0, name=cname)
            self.register_aggregate_constr(("capacity_link_t", int(t)), cname)
            added += 1

        m.update()
        self.capacity_cuts_added += int(added)

    def _add_initial_columns(self) -> None:
        """Seed root columns from ALNS incumbent routes."""
        use_alns = bool(int(self.inst.get("use_alns_initialization", 1)))
        replicate_all_contexts = bool(int(self.inst.get("alns_replicate_all_contexts", 1)))
        if use_alns:
            try:
                alns_out = run_alns_initial_solution(
                    inst=self.inst,
                    iterations=int(self.inst.get("alns_iterations", 300)),
                    destroy_fraction=float(self.inst.get("alns_destroy_fraction", 0.25)),
                    seed=(None if self.inst.get("alns_seed", None) in (None, "", -1) else int(self.inst.get("alns_seed"))),
                )
                pool_cols: List[Dict[str, Any]] = list(alns_out.get("column_pool") or alns_out.get("columns", []))
                for col in pool_cols:
                    day = int(col["day"])
                    driver = int(col["driver"])
                    served = tuple(tuple(e) for e in col.get("serviced_required_edges", []))
                    path_arcs = tuple(tuple(a) for a in col.get("path_arcs", []))
                    nonreq_used = tuple(tuple(e) for e in col.get("nonrequired_edges_used", []))
                    if not served or not path_arcs:
                        continue
                    self._add_route_var(
                        day=day,
                        driver=driver,
                        serviced_edges=list(served),
                        path_arcs=list(path_arcs),
                        nonrequired_edges_used=list(nonreq_used) if nonreq_used else None,
                    )

                base_routes: List[Tuple[Tuple[Edge, ...], Tuple[Tuple[int, int], ...], int, int]] = []
                seen_base: set[Tuple[Tuple[Edge, ...], Tuple[Tuple[int, int], ...], int, int]] = set()
                for col in alns_out.get("columns", []):
                    day = int(col["day"])
                    driver = int(col["driver"])
                    served = tuple(tuple(e) for e in col.get("serviced_required_edges", []))
                    path_arcs = tuple(tuple(a) for a in col.get("path_arcs", []))
                    if not served or not path_arcs:
                        continue
                    base_key = (served, path_arcs, day, driver)
                    if base_key in seen_base:
                        continue
                    seen_base.add(base_key)
                    base_routes.append(base_key)

                # Expand by cloning each *best* ALNS route across all (day, driver) contexts.
                if replicate_all_contexts:
                    for served, path_arcs, _day0, _driver0 in base_routes:
                        for day in self.days:
                            for driver in self.vehicles:
                                self._add_route_var(
                                    day=int(day),
                                    driver=int(driver),
                                    serviced_edges=list(served),
                                    path_arcs=list(path_arcs),
                                )
                if isinstance(alns_out, dict) and math.isfinite(float(alns_out.get("objective", float("inf")))):
                    self.initial_incumbent = {
                        "objective": float(alns_out["objective"]),
                        "active_routes": list(alns_out.get("active_routes", [])),
                        "source": "alns_initial",
                        "rmp_feasible": bool(alns_out.get("rmp_feasible", True)),
                        "rmp_feasible_detail": str(alns_out.get("rmp_feasible_detail", "")),
                        "alns_column_pool_size": int(len(pool_cols)),
                    }
            except Exception:
                self.initial_incumbent = None

        # Fallback: at least one column to start root pricing.
        if not self.route_columns:
            for day in self.days:
                for driver in self.vehicles:
                    for e in self.required_edges:
                        path_arcs, _ = _single_service_route(
                            depot=self.depot,
                            required_edge=e,
                            edge_cost=self.travel_cost,
                        )
                        self._add_route_var(
                            day=day,
                            driver=driver,
                            serviced_edges=[e],
                            path_arcs=list(path_arcs),
                        )
        self.model.update()

    def add_pricing_columns(self, columns: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        attempted = 0
        added = 0
        skipped_empty = 0
        skipped_capacity = 0
        skipped_duplicate = 0
        skipped_dominated = 0
        active_aggregate_constrs = self._active_aggregate_constr_items()
        for c in columns:
            attempted += 1
            day = int(c["day"])
            driver = int(c.get("driver", self.vehicles[0]))
            served = [_canon_edge(*e) if isinstance(e, tuple) and len(e) == 2 else e for e in c["serviced_required_edges"]]
            path_arcs = [tuple(a) for a in c["path_arcs"]]
            nonreq_used = [tuple(e) for e in c.get("nonrequired_edges_used", [])]
            if not served:
                skipped_empty += 1
                continue
            # guard capacity-feasible route columns only
            load = sum(float(self.inst["demand"].get(e, 0.0)) for e in served)
            if load > self.capacity + 1e-9:
                skipped_capacity += 1
                continue
            v = self._add_route_var(
                day,
                driver,
                served,
                path_arcs,
                nonreq_used if nonreq_used else None,
                active_aggregate_constrs=active_aggregate_constrs,
            )
            if v is None:
                if self._last_add_route_status == "dominated":
                    skipped_dominated += 1
                else:
                    skipped_duplicate += 1
            else:
                added += 1
        self.model.update()
        self.last_add_stats = {
            "attempted": attempted,
            "added": added,
            "skipped_empty": skipped_empty,
            "skipped_capacity": skipped_capacity,
            "skipped_duplicate": skipped_duplicate,
            "skipped_dominated": skipped_dominated,
        }
        return dict(self.last_add_stats)

    def get_pricing_data(self) -> Dict[str, Any]:
        adjacency: Dict[int, List[Dict[str, Any]]] = {i: [] for i in self.inst["nodes"]}
        for e in self.edges:
            i, j = e
            req = e in self.required_edges
            sid = e
            dem = float(self.inst["demand"].get(e, 0.0)) if req else 0.0
            serv = float(self.service_extra[e]) if req else 0.0

            adjacency[i].append(
                {
                    "id": (i, j),
                    "to": j,
                    "travel_cost": float(self.travel_cost[e]),
                    "required": req,
                    "required_id": sid,
                    "demand": dem,
                    "service_cost": serv,
                }
            )
            adjacency[j].append(
                {
                    "id": (j, i),
                    "to": i,
                    "travel_cost": float(self.travel_cost[e]),
                    "required": req,
                    "required_id": sid,
                    "demand": dem,
                    "service_cost": serv,
                }
            )

        # Derive schedule-driven hard filters from current RMP variable domains.
        # For each required edge e, keep patterns still possible by bounds (UB > 0.5).
        # If all remaining patterns include a day t => mandatory service on t.
        # If no remaining pattern includes day t => forbidden service on t.
        mandatory_required_edges_by_day: Dict[Tuple[int, int], List[Edge]] = {
            (int(t), int(k)): [] for t in self.days for k in self.vehicles
        }
        forbidden_required_edges_by_day: Dict[Tuple[int, int], List[Edge]] = {
            (int(t), int(k)): [] for t in self.days for k in self.vehicles
        }
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            for k in self.vehicles:
                forced_idx: List[int] = []
                possible_idx: List[int] = []
                for p_idx, _ in enumerate(pats):
                    vname = self.schedule_var_name.get((e, p_idx, int(k)))
                    var = self._get_var_cached(vname) if vname is not None else None
                    if var is None:
                        continue
                    if float(var.LB) > 0.5:
                        forced_idx.append(p_idx)
                    if float(var.UB) > 0.5:
                        possible_idx.append(p_idx)
                # If one (or more) patterns are fixed by LB, use them as the only feasible set.
                if forced_idx:
                    possible_idx = sorted(set(forced_idx))
                # If all patterns are cut off by bounds unexpectedly, skip hard filtering for safety.
                if not possible_idx:
                    continue
                for t in self.days:
                    in_count = sum(1 for p_idx in possible_idx if int(t) in pats[p_idx])
                    key = (int(t), int(k))
                    if in_count == 0:
                        forbidden_required_edges_by_day[key].append(e)
                    elif in_count == len(possible_idx):
                        mandatory_required_edges_by_day[key].append(e)

        return {
            "master_mode": "simple_sp",
            "days": list(self.contexts),
            "capacity": self.capacity,
            "depot": self.depot,
            "adjacency": adjacency,
            "pricing_method": str(self.inst.get("pricing_method", "labeling")),
            "pricing_ng_size": int(self.inst.get("pricing_ng_size", 8)),
            "cpp_empty_fallback": str(self.inst.get("cpp_empty_fallback", "dp")),
            "cpp_ng_empty_fallback": str(
                self.inst.get(
                    "cpp_ng_empty_fallback",
                    self.inst.get("cpp_empty_fallback", "labeling"),
                )
            ),
            "cut_pricing_mode": str(self.inst.get("cut_pricing_mode", "legacy")),
            "cut_pricing_dual_tol": float(self.inst.get("cut_pricing_dual_tol", 1e-15)),
            "use_schedule_hard_filter": bool(self.inst.get("use_schedule_hard_filter", False)),
            "max_columns": int(self.inst.get("pricing_max_columns", 0)),
            "use_coeff_dominance_filter": bool(self.use_coeff_dominance_filter),
            "coeff_dom_obj_tol": float(self.coeff_dom_obj_tol),
            "mandatory_required_edges_by_day": mandatory_required_edges_by_day,
            "forbidden_required_edges_by_day": forbidden_required_edges_by_day,
            "cover_constr_name_by_edge_day": dict(self.cover_constr_name),
            "vehicle_limit_constr_name_by_day": dict(self.veh_constr_name),
            "vehicle_lex_constr_name_by_day": dict(self.veh_lex_constr_name),
            "discount_link_constr_name_by_edge_day": dict(self.discount_link_constr_name),
            # Pass references (not copies) to avoid per-CG large allocations.
            "existing_column_signatures": self.column_signatures,
            "existing_column_coeff_best_obj": self.column_coeff_best_obj,
            # Cutting planes (capacity-link): same registry as column coeff updates.
            "aggregate_branch_constrs": dict(self.aggregate_branch_constrs),
        }

    def _build_branching_schedule_exprs(
        self,
    ) -> Tuple[Dict[Any, str], Dict[Tuple[Edge, int], Dict[str, float]], Dict[Tuple[Edge, int, int], Dict[str, float]]]:
        return (
            self.schedule_var_name,
            self._branch_edge_driver_assign_expr,
            self._branch_edge_day_driver_service_expr,
        )

    def extend_branching_route_expressions(self, data: Dict[str, Any]) -> None:
        """Expose incrementally maintained route-lifting expressions."""
        if data.get("_route_lifting_done"):
            return
        data["arc_visit_expr"] = self._branch_arc_visit_expr
        data["node_visit_expr"] = self._branch_node_visit_expr
        data["ryan_foster_pair_expr"] = self._branch_ryan_foster_pair_expr
        data["_route_lifting_done"] = True

    def get_branching_data(self, *, include_route_lifting: bool = True) -> Dict[str, Any]:
        (
            schedule_vars,
            edge_driver_assign_expr,
            edge_day_driver_service_expr,
        ) = self._build_branching_schedule_exprs()
        arc_visit_expr: Dict[Tuple, Dict[str, float]] = {}
        node_visit_expr: Dict[Tuple, Dict[str, float]] = {}
        if include_route_lifting:
            tmp: Dict[str, Any] = {
                "arc_visit_expr": arc_visit_expr,
                "node_visit_expr": node_visit_expr,
            }
            self.extend_branching_route_expressions(tmp)
            arc_visit_expr = tmp["arc_visit_expr"]
            node_visit_expr = tmp["node_visit_expr"]

        return {
            "branching_mode": "rmp",
            "lambda_vars_by_day": self._lambda_vars_by_day_only(),
            "schedule_vars": schedule_vars,
            "schedule_vars_by_edge": self._branch_schedule_vars_by_edge,
            "schedule_vars_by_edge_pattern": self._branch_schedule_vars_by_edge_pattern,
            "schedule_vars_by_edge_driver": self._branch_schedule_vars_by_edge_driver,
            "schedule_vars_by_edge_day_driver": self._branch_schedule_vars_by_edge_day_driver,
            "schedule_pattern_sum_expr": self._branch_schedule_pattern_sum_expr,
            "edge_driver_assign_expr": edge_driver_assign_expr,
            "edge_day_driver_service_expr": edge_day_driver_service_expr,
            "node_visit_expr": node_visit_expr,
            "arc_visit_expr": arc_visit_expr,
            "ryan_foster_pair_expr": self._branch_ryan_foster_pair_expr,
            # Enable aggregate (whole/daily) and expression (node/arc) branching levels
            "enable_aggregate_lambda_branching": True,
            "enable_expression_branching": True,
        }

    def _lambda_vars_by_day_only(self) -> Dict[int, List[str]]:
        """Aggregate lambda variable names by day t only (sum over all vehicle types k).

        Used for daily_route branching: Σ_{k,r} λ^{tk}_r per day t.
        """
        by_day: Dict[int, List[str]] = {}
        for (day, _k), names in self.lambda_var_names_by_day.items():
            by_day.setdefault(int(day), []).extend(names)
        return by_day

    def register_aggregate_constr(self, key: tuple, cname: str) -> None:
        """Register an aggregate branching constraint so future columns participate in it."""
        self.aggregate_branch_constrs[key] = cname
        self._aggregate_constr_handles[key] = self._get_constr_cached(str(cname))

    def invalidate_constr_cache_after_cut_removal(self, removed_cnames: Sequence[str]) -> None:
        """Drop cached Gurobi constraint handles after separation rollback."""
        removed = {str(cn) for cn in removed_cnames}
        for cn in removed_cnames:
            self._constr_by_name.pop(str(cn), None)
        if removed:
            stale_keys = [key for key, cname in self.aggregate_branch_constrs.items() if str(cname) in removed]
            for key in stale_keys:
                self._aggregate_constr_handles.pop(key, None)

    def _get_constr_cached(self, cname: str) -> Optional[Any]:
        c = self._constr_by_name.get(cname)
        if c is not None:
            return c
        c = self.model.getConstrByName(cname)
        if c is None:
            self.model.update()
            c = self.model.getConstrByName(cname)
        if c is not None:
            self._constr_by_name[cname] = c
        return c

    def _get_var_cached(self, vname: str) -> Optional[Any]:
        v = self._var_by_name.get(vname)
        if v is not None:
            return v
        v = self.model.getVarByName(vname)
        if v is None:
            self.model.update()
            v = self.model.getVarByName(vname)
        if v is not None:
            self._var_by_name[vname] = v
        return v

    def separate_cuts(self, node_depth: Optional[int] = None, node_id: Optional[int] = None) -> int:
        """Run LP separation rounds on current RMP (if enabled)."""
        if bool(self.inst.get("cut_root_only", 1)):
            if node_depth is not None and int(node_depth) > 0:
                return 0
        max_depth = self.inst.get("cut_separation_max_depth", None)
        if max_depth is not None and node_depth is not None:
            if int(node_depth) > int(max_depth):
                return 0
        if self.separation_manager is None:
            return 0
        added = int(self.separation_manager.separate(self))
        self.capacity_cuts_added += int(added)
        return int(added)

    def separate_sri_cuts(
        self,
        node_depth: Optional[int] = None,
        node_id: Optional[int] = None,
        optimize_before: bool = False,
    ) -> SeparationRoundResult:
        del node_id
        if not bool(int(self.inst.get("enable_sri", self.inst.get("use_sri_cuts", 0)))):
            return SeparationRoundResult()
        if bool(int(self.inst.get("root_only_sri", 1))) and node_depth is not None and int(node_depth) > 0:
            return SeparationRoundResult()
        if self.sri_separation_manager is None:
            return SeparationRoundResult()
        result = self.sri_separation_manager.separate_once(self, optimize_before=optimize_before)
        self.sri_cuts_added += int(result.added_count)
        return result

    def build_executable_solution(self, values: Dict[str, float], source: str = "bnb") -> Dict[str, Any]:
        eps = 1e-6
        routes: List[Dict[str, Any]] = []
        for vname, ridx in self.lambda_var_name_to_index.items():
            x = float(values.get(vname, 0.0))
            if x <= eps:
                continue
            if ridx < 0 or ridx >= len(self.route_columns):
                continue
            col = self.route_columns[ridx]
            routes.append(
                {
                    "var": vname,
                    "value": x,
                    "day": int(col.day),
                    "driver": int(col.driver),
                    "serviced_required_edges": [tuple(e) for e in col.serviced_required_edges],
                    "path_arcs": [tuple(a) for a in col.path_arcs],
                    "nonrequired_edges_used": [tuple(e) for e in col.nonrequired_edges_used],
                    "cost": float(col.cost),
                }
            )

        schedules: List[Dict[str, Any]] = []
        for (e, p_idx, k), vname in self.schedule_var_name.items():
            x = float(values.get(vname, 0.0))
            if x <= eps:
                continue
            schedules.append(
                {
                    "var": vname,
                    "value": x,
                    "edge": tuple(e),
                    "pattern_index": int(p_idx),
                    "driver": int(k),
                    "pattern_days": sorted(int(d) for d in self.schedule_patterns[e][p_idx]),
                }
            )

        artificial_positive: List[Dict[str, Any]] = []
        for cname, aname in self.artificial_var_name_by_cover.items():
            x = float(values.get(aname, 0.0))
            if x > eps:
                artificial_positive.append({"cover_constraint": cname, "var": aname, "value": x})

        routes.sort(key=lambda r: (r["day"], r["driver"], r["var"]))
        schedules.sort(key=lambda s: (s["edge"], s["driver"], s["pattern_index"]))

        return {
            "source": str(source),
            "route_cost_total": float(sum(float(r["cost"]) * float(r["value"]) for r in routes)),
            "num_active_routes": len(routes),
            "num_selected_schedules": len(schedules),
            "routes": routes,
            "schedules": schedules,
            "artificial_positive": artificial_positive,
        }

    def copy_for_child(self) -> "SimpleSPMaster":
        new = object.__new__(SimpleSPMaster)
        new.inst = self.inst
        new.model = self.model.copy()

        new.days = list(self.days)
        new.vehicles = list(self.vehicles)
        new.contexts = list(self.contexts)
        new.required_edges = list(self.required_edges)
        new.capacity = self.capacity
        new.depot = self.depot
        new.edges = list(self.edges)
        new.travel_cost = dict(self.travel_cost)
        new.service_extra = dict(self.service_extra)
        new.schedule_patterns = self.schedule_patterns
        new.required_edge_set = set(self.required_edge_set)
        new.nonrequired_edges = list(self.nonrequired_edges)
        new.nonrequired_edge_set = set(self.nonrequired_edge_set)
        new.discount_theta = float(self.discount_theta)
        new.m_t = dict(self.m_t)

        new.schedule_var_name = dict(self.schedule_var_name)
        new.cover_constr_name = dict(self.cover_constr_name)
        new.veh_constr_name = dict(self.veh_constr_name)
        new.veh_lex_constr_name = dict(self.veh_lex_constr_name)
        new.use_vehicle_lex_symmetry = bool(self.use_vehicle_lex_symmetry)
        new._vehicles_sorted = list(self._vehicles_sorted)
        new.discount_z_var_name = dict(self.discount_z_var_name)
        new.discount_link_constr_name = dict(self.discount_link_constr_name)
        new.lambda_var_names_by_day = {ctx: list(v) for ctx, v in self.lambda_var_names_by_day.items()}
        new.lambda_var_name_to_index = dict(self.lambda_var_name_to_index)
        new.column_signatures = set(self.column_signatures)
        new.column_signature_meta = dict(self.column_signature_meta)
        new.column_coeff_best_obj = dict(self.column_coeff_best_obj)
        new.use_coeff_dominance_filter = bool(self.use_coeff_dominance_filter)
        new.coeff_dom_obj_tol = float(self.coeff_dom_obj_tol)
        new.column_pool_dominated = int(self.column_pool_dominated)
        new._last_add_route_status = "none"
        new.column_pool_hits = int(self.column_pool_hits)
        new.column_pool_misses = int(self.column_pool_misses)
        new._column_pool_tick = int(self._column_pool_tick)
        new.route_columns = list(self.route_columns)
        new.artificial_var_name_by_cover = dict(self.artificial_var_name_by_cover)
        new.capacity_cuts_added = int(self.capacity_cuts_added)
        new.sri_cuts_added = int(self.sri_cuts_added)
        # Aggregate branching constraints are in the copied Gurobi model already;
        # copy the registry so new columns participate correctly.
        new.aggregate_branch_constrs = dict(self.aggregate_branch_constrs)
        new._aggregate_constr_handles = {}
        new._constr_by_name = {}
        new._var_by_name = {}
        new._branch_edge_driver_assign_expr = {
            key: dict(expr) for key, expr in self._branch_edge_driver_assign_expr.items()
        }
        new._branch_edge_day_driver_service_expr = {
            key: dict(expr) for key, expr in self._branch_edge_day_driver_service_expr.items()
        }
        new._branch_arc_visit_expr = {
            key: dict(expr) for key, expr in self._branch_arc_visit_expr.items()
        }
        new._branch_node_visit_expr = {
            key: dict(expr) for key, expr in self._branch_node_visit_expr.items()
        }
        new._branch_ryan_foster_pair_expr = {
            key: dict(expr) for key, expr in self._branch_ryan_foster_pair_expr.items()
        }
        new._branch_schedule_vars_by_edge = {
            key: list(vals) for key, vals in self._branch_schedule_vars_by_edge.items()
        }
        new._branch_schedule_vars_by_edge_pattern = {
            key: list(vals) for key, vals in self._branch_schedule_vars_by_edge_pattern.items()
        }
        new._branch_schedule_pattern_sum_expr = {
            key: dict(expr) for key, expr in self._branch_schedule_pattern_sum_expr.items()
        }
        new._branch_schedule_vars_by_edge_driver = {
            key: list(vals) for key, vals in self._branch_schedule_vars_by_edge_driver.items()
        }
        new._branch_schedule_vars_by_edge_day_driver = {
            key: list(vals) for key, vals in self._branch_schedule_vars_by_edge_day_driver.items()
        }
        new.separation_manager = None
        new.sri_separation_manager = None
        new.initial_incumbent = copy.deepcopy(self.initial_incumbent)
        if self.separation_manager is not None:
            new.separation_manager = SeparationManager(
                separators=list(self.separation_manager.separators),
                tol=float(self.separation_manager.tol),
                max_rounds=int(self.separation_manager.max_rounds),
            )
            new.separation_manager.total_rounds = int(self.separation_manager.total_rounds)
            new.separation_manager.total_cuts_added = int(self.separation_manager.total_cuts_added)
        if self.sri_separation_manager is not None:
            new.sri_separation_manager = SeparationManager(
                separators=list(self.sri_separation_manager.separators),
                tol=float(self.sri_separation_manager.tol),
                max_rounds=int(self.sri_separation_manager.max_rounds),
            )
            new.sri_separation_manager.total_rounds = int(self.sri_separation_manager.total_rounds)
            new.sri_separation_manager.total_cuts_added = int(self.sri_separation_manager.total_cuts_added)
        return new


def _solve_sp_master_exact(rmp: SimpleSPMaster) -> Dict[str, Any]:
    m = rmp.model
    # solve exact master MIP on current column set
    for v in m.getVars():
        name = v.VarName.lower()
        if name.startswith("lam_") or name.startswith("lambda_") or name.startswith("s_") or name.startswith("z_"):
            v.VType = GRB.BINARY
    m.update()
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Exact SP master solve failed with status {m.Status}")
    return {"objective": float(m.ObjVal)}


def _try_root_incumbent(
    rmp: SimpleSPMaster,
    time_limit_s: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """
    Try to get an initial finite UB by solving the restricted master as a MIP.
    - Uses a copied model (does not mutate the live RMP).
    - Forces artificial vars to 0 to get a meaningful feasible incumbent.
    """
    m = rmp.model.copy()
    if time_limit_s > 0:
        m.Params.TimeLimit = float(time_limit_s)
    m.Params.OutputFlag = 0
    m.Params.MIPFocus = 1
    m.Params.Heuristics = 0.5

    for v in m.getVars():
        name = v.VarName.lower()
        if name.startswith("lam_") or name.startswith("lambda_") or name.startswith("s_") or name.startswith("z_"):
            v.VType = GRB.BINARY

    for aname in rmp.artificial_var_name_by_cover.values():
        a = m.getVarByName(aname)
        if a is not None:
            a.UB = 0.0

    m.update()
    m.optimize()
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        values: Dict[str, float] = {}
        active_routes: List[Dict[str, Any]] = []
        eps = 1e-6
        for v in m.getVars():
            xv = float(v.X)
            if xv <= eps:
                continue
            values[v.VarName] = xv
            lname = v.VarName.lower()
            if lname.startswith("lam_t") and "_r" in lname:
                try:
                    ridx = int(lname.split("_r")[-1])
                except ValueError:
                    ridx = -1
                if 0 <= ridx < len(rmp.route_columns):
                    col = rmp.route_columns[ridx]
                    active_routes.append(
                        {
                            "var": v.VarName,
                            "value": xv,
                            "day": int(col.day),
                            "driver": int(col.driver),
                            "serviced_required_edges": [tuple(e) for e in col.serviced_required_edges],
                            "path_arcs": [tuple(a) for a in col.path_arcs],
                            "cost": float(col.cost),
                        }
                    )
        return {
            "objective": float(m.ObjVal),
            "status": int(m.Status),
            "variables": values,
            "active_routes": active_routes,
            "source": "root_rmp_mip",
        }
    return None


def seed_bnb_tree_initial_ub(
    tree: Any,
    rmp: Any,
    *,
    time_limit_s: float = 3.0,
) -> None:
    """
    Set tree.global_upper_bound / best_solution from (1) ALNS incumbent if rmp_feasible,
    optionally improved by (2) a short root restricted-master MIP (SimpleSPMaster only).
    """
    root_incumbent: Optional[Dict[str, Any]] = None
    inc = getattr(rmp, "initial_incumbent", None)
    if isinstance(inc, dict) and bool(inc.get("rmp_feasible", True)):
        obj = float(inc.get("objective", float("inf")))
        if math.isfinite(obj) and obj < float("inf"):
            root_incumbent = copy.deepcopy(inc)

    mip_cand: Optional[Dict[str, Any]] = None
    if isinstance(rmp, SimpleSPMaster):
        mip_cand = _try_root_incumbent(rmp, time_limit_s=float(time_limit_s))
    if isinstance(mip_cand, dict):
        mip_obj = float(mip_cand.get("objective", float("inf")))
        cur_obj = (
            float(root_incumbent.get("objective", float("inf")))
            if isinstance(root_incumbent, dict)
            else float("inf")
        )
        if mip_obj < cur_obj:
            root_incumbent = mip_cand

    if isinstance(root_incumbent, dict):
        o = float(root_incumbent.get("objective", float("inf")))
        if math.isfinite(o) and o < float("inf"):
            tree.global_upper_bound = float(o)
            tree.best_solution = root_incumbent


def _build_executable_solution_payload(
    rmp: Any,
    solution_obj: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(solution_obj, dict):
        return None
    values = solution_obj.get("variables")
    if isinstance(values, dict):
        return rmp.build_executable_solution(values=values, source=str(solution_obj.get("source", "bnb")))
    active_routes = solution_obj.get("active_routes")
    if isinstance(active_routes, list):
        return {
            "source": str(solution_obj.get("source", "active_routes_only")),
            "route_cost_total": float(
                sum(float(r.get("cost", 0.0)) * float(r.get("value", 0.0)) for r in active_routes if isinstance(r, dict))
            ),
            "num_active_routes": len(active_routes),
            "num_selected_schedules": 0,
            "routes": active_routes,
            "schedules": [],
            "artificial_positive": [],
        }
    return None


def solve_with_current_algorithm(inst: Dict[str, Any]) -> Dict[str, Any]:
    from refactor_algorithm.core.master.aggregated_master import AggregatedMaster
    from refactor_algorithm.core.master.compare_global_rmp_bnp import GlobalRMPBnBTree

    inst = _apply_inspect_bnp_defaults(inst)

    rmp = AggregatedMaster(inst)
    if int(inst.get("use_aggregation", 0)) == 0:
        rmp.switch_to_rmp_mode()
    root = BnBNode(node_id=0, depth=0, master_problem=rmp)
    require_proof = bool(inst.get("require_proof_optimality", False))
    max_cg_iter = int(inst.get("max_cg_iterations_per_node", 10000))
    max_nodes = int(inst.get("max_nodes", 999999))
    max_time_s = float(inst.get("algorithm_time_limit_s", 0.0))
    strategy = str(inst.get("node_search_strategy", "best_bound")).lower()
    selector = BestBoundSelector() if strategy == "best_bound" else DepthFirstSelector()

    eps_rc = float(inst.get("eps_reduced_cost", 1e-4))
    use_stab = bool(int(inst.get("use_dual_stabilization", 0)))
    stab_alpha = float(inst.get("dual_stab_alpha", 0.5))
    stab_alpha_decay = float(inst.get("dual_stab_alpha_decay", 0.9))
    stab_min_alpha = float(inst.get("dual_stab_min_alpha", 0.0))
    use_ub_zero = bool(int(inst.get("use_ub_zero_branching", 0)))
    partial_pricing_ratio = float(inst.get("partial_pricing_ratio", 1.0))
    phase1_col_cap = int(inst.get("phase1_col_cap", 1000))
    enable_sri = bool(int(inst.get("enable_sri", inst.get("use_sri_cuts", 0))))
    root_only_sri = bool(int(inst.get("root_only_sri", 1)))
    max_sri_rounds = int(inst.get("max_sri_rounds", 3))

    tree = GlobalRMPBnBTree(
        root_node=root,
        config=BnBConfig(
            eps_integrality=1e-6,
            eps_reduced_cost=eps_rc,
            max_cg_iterations_per_node=max_cg_iter,
            max_nodes=max_nodes,
            max_time_s=(max_time_s if max_time_s > 0 else None),
            verbose=bool(inst.get("bnb_log", False)),
            use_dual_stabilization=use_stab,
            dual_stab_alpha=stab_alpha,
            dual_stab_alpha_decay=stab_alpha_decay,
            dual_stab_min_alpha=stab_min_alpha,
            use_ub_zero_branching=use_ub_zero,
            partial_pricing_ratio=partial_pricing_ratio,
            phase1_col_cap=phase1_col_cap,
            enable_sri=enable_sri,
            root_only_sri=root_only_sri,
            max_sri_rounds=max_sri_rounds,
        ),
        selector=selector,
    )
    sol = tree.solve()
    alns_root = getattr(rmp, "initial_incumbent", None)
    root_incumbent_obj: Optional[float] = None
    if isinstance(alns_root, dict):
        try:
            ro = float(alns_root.get("objective", float("inf")))
            if math.isfinite(ro) and ro < float("inf"):
                root_incumbent_obj = ro
        except (TypeError, ValueError):
            root_incumbent_obj = None

    m = rmp.model
    artificial_sum = 0.0
    if int(getattr(m, "SolCount", 0)) > 0:
        for aname in rmp.artificial_var_name_by_cover.values():
            a = m.getVarByName(aname)
            if a is not None:
                artificial_sum += float(a.X)
    hit_node_limit = tree.config.max_nodes is not None and tree.nodes_processed >= tree.config.max_nodes
    hit_node_limit = bool(hit_node_limit or tree.terminated_by_node_limit)
    hit_time_limit = bool(tree.terminated_by_time_limit)
    hit_cg_limit = bool(tree.profile.get("nodes_hit_cg_limit", 0.0) > 0.0)
    final_gap_pct = tree._gap_percent()
    profile = {
        "rmp_time_s": float(tree.profile.get("rmp_time_s", 0.0)),
        "pricing_time_s": float(tree.profile.get("pricing_time_s", 0.0)),
        "addcol_time_s": float(tree.profile.get("addcol_time_s", 0.0)),
        "labels_generated": int(tree.profile.get("labels_generated", 0.0)),
        "labels_expanded": int(tree.profile.get("labels_expanded", 0.0)),
        "backtrack_pruned": int(tree.profile.get("backtrack_pruned", 0.0)),
        "shortcut_returns": int(tree.profile.get("shortcut_returns", 0.0)),
        "existing_sig_filtered": int(tree.profile.get("existing_sig_filtered", 0.0)),
        "columns_generated": int(tree.profile.get("columns_generated", 0.0)),
        "columns_added": int(tree.profile.get("columns_added", 0.0)),
        "column_pool_hits": int(getattr(rmp, "column_pool_hits", 0)),
        "column_pool_misses": int(getattr(rmp, "column_pool_misses", 0)),
        "zero_add_iterations": int(tree.profile.get("zero_add_iterations", 0.0)),
        "nodes_hit_cg_limit": int(tree.profile.get("nodes_hit_cg_limit", 0.0)),
        "separation_rounds": int(getattr(getattr(rmp, "separation_manager", None), "total_rounds", 0)),
        "separation_cuts_added": int(getattr(getattr(rmp, "separation_manager", None), "total_cuts_added", 0)),
        "cg_iterations": int(tree.profile.get("cg_iterations", 0.0)),
        "phase1_iters": int(tree.profile.get("phase1_iters", 0.0)),
    }
    incumbent_obj = float(tree.global_upper_bound) if tree.global_upper_bound < float("inf") else None
    incumbent_solution = tree.best_solution
    if incumbent_solution is None and isinstance(alns_root, dict):
        incumbent_solution = copy.deepcopy(alns_root)
    executable_solution = _build_executable_solution_payload(rmp, incumbent_solution)
    capacity_cuts_added = int(getattr(rmp, "capacity_cuts_added", 0))

    if final_gap_pct is None:
        gap_display = "n/a"
    else:
        gap_display = f"{float(final_gap_pct):.6f}%"
    inc_display = "None" if incumbent_obj is None else f"{float(incumbent_obj):.6f}"
    if bool(inst.get("bnb_log", False)):
        print(
            f"[BnB-Final] mode={'proof' if require_proof else 'default'} "
            f"nodes={tree.nodes_processed} incumbent_obj={inc_display} gap={gap_display} "
            f"node_limit={hit_node_limit} time_limit={hit_time_limit} cg_limit={hit_cg_limit} "
            f"has_incumbent_solution={incumbent_solution is not None}"
        )

    if require_proof:
        if hit_node_limit:
            raise RuntimeError("Proof mode failed: reached BnB node limit before tree exhaustion.")
        if hit_time_limit:
            raise RuntimeError("Proof mode failed: reached BnB time limit before tree exhaustion.")
        if hit_cg_limit:
            raise RuntimeError("Proof mode failed: reached CG iteration limit at least one node.")
        if not tree.selector.is_empty():
            raise RuntimeError("Proof mode failed: open nodes remain in BnB queue.")
        if tree.best_solution is None or not (tree.global_upper_bound < float("inf")):
            raise RuntimeError("Proof mode failed: no proven finite incumbent.")
        return {
            "objective": float(tree.global_upper_bound),
            "solution": tree.best_solution,
            "incumbent_objective": incumbent_obj,
            "incumbent_solution": incumbent_solution,
            "executable_solution": executable_solution,
            "nodes_processed": tree.nodes_processed,
            "mode": "inspect_style_bnp_proven_optimal",
            "artificial_sum": artificial_sum,
            "hit_node_limit": False,
            "hit_time_limit": False,
            "hit_cg_limit": False,
            "gap_pct": final_gap_pct,
            "profile": profile,
            "root_incumbent": root_incumbent_obj,
            "capacity_cuts_added": capacity_cuts_added,
        }

    # Preferred: full BnB result
    if sol is not None and tree.global_upper_bound < float("inf"):
        return {
            "objective": float(tree.global_upper_bound),
            "solution": sol,
            "incumbent_objective": incumbent_obj,
            "incumbent_solution": incumbent_solution,
            "executable_solution": executable_solution,
            "nodes_processed": tree.nodes_processed,
            "mode": "inspect_style_bnp",
            "artificial_sum": artificial_sum,
            "hit_node_limit": bool(hit_node_limit),
            "hit_time_limit": bool(hit_time_limit),
            "hit_cg_limit": bool(hit_cg_limit),
            "gap_pct": final_gap_pct,
            "profile": profile,
            "root_incumbent": root_incumbent_obj,
            "capacity_cuts_added": capacity_cuts_added,
        }

    # Not proven optimal / no exact integral node solution object, but incumbent UB exists.
    if incumbent_obj is not None:
        return {
            "objective": float(incumbent_obj),
            "solution": incumbent_solution,
            "incumbent_objective": incumbent_obj,
            "incumbent_solution": incumbent_solution,
            "executable_solution": executable_solution,
            "nodes_processed": tree.nodes_processed,
            "mode": "inspect_style_bnp_incumbent_only",
            "artificial_sum": artificial_sum,
            "hit_node_limit": bool(hit_node_limit),
            "hit_time_limit": bool(hit_time_limit),
            "hit_cg_limit": bool(hit_cg_limit),
            "gap_pct": final_gap_pct,
            "profile": profile,
            "root_incumbent": root_incumbent_obj,
            "capacity_cuts_added": capacity_cuts_added,
        }

    # Fallback for validation experiments: exact solve of current SP master
    exact = _solve_sp_master_exact(rmp)
    exact_values: Dict[str, float] = {}
    for v in rmp.model.getVars():
        x = float(v.X)
        if x > 1e-6:
            exact_values[v.VarName] = x
    exact_solution = {"source": "sp_master_exact_fallback", "variables": exact_values}
    exact_exec = _build_executable_solution_payload(rmp, exact_solution)
    fallback_obj = float(exact["objective"])
    tree.global_upper_bound = min(float(tree.global_upper_bound), fallback_obj)
    tree._refresh_global_lower_bound()
    fallback_gap_pct = tree._gap_percent()
    return {
        "objective": fallback_obj,
        "solution": exact_solution,
        "incumbent_objective": fallback_obj,
        "incumbent_solution": exact_solution,
        "executable_solution": exact_exec,
        "nodes_processed": tree.nodes_processed,
        "mode": "sp_master_exact_fallback",
        "artificial_sum": artificial_sum,
        "hit_node_limit": bool(hit_node_limit),
        "hit_time_limit": bool(hit_time_limit),
        "hit_cg_limit": bool(hit_cg_limit),
        "gap_pct": fallback_gap_pct,
        "profile": profile,
        "root_incumbent": root_incumbent_obj,
        "capacity_cuts_added": capacity_cuts_added,
    }


def run_comparison(seed: int = 7) -> Dict[str, Any]:
    inst = create_random_pcarp_instance(seed=seed)
    arc = solve_arc_based_pcarp_optimal(inst)
    alg = solve_with_current_algorithm(inst)

    diff = abs(float(arc["objective"]) - float(alg["objective"]))
    return {
        "seed": seed,
        "arc_obj": float(arc["objective"]),
        "alg_obj": float(alg["objective"]),
        "abs_diff": diff,
        "match": diff <= 1e-5,
        "nodes_processed": alg["nodes_processed"],
        "algorithm_mode": alg["mode"],
        "required_edges": inst["required_edges"],
        "schedules": inst["schedule_patterns"],
    }


if __name__ == "__main__":
    out = run_comparison(seed=7)
    print("seed:", out["seed"])
    print("arc_obj:", out["arc_obj"])
    print("alg_obj:", out["alg_obj"])
    print("abs_diff:", out["abs_diff"])
    print("match:", out["match"])
    print("algorithm_mode:", out["algorithm_mode"])
    print("nodes_processed:", out["nodes_processed"])
    print("required_edges:", out["required_edges"])
    print("schedules:", out["schedules"])
