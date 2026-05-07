from __future__ import annotations

import argparse
import heapq
from itertools import combinations
import random as _random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Edge = Tuple[int, int]


def _canon_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def _parse_int_after_colon(line: str) -> int:
    m = re.search(r":\s*([0-9]+)", line)
    if not m:
        raise ValueError(f"Could not parse integer from header line: {line!r}")
    return int(m.group(1))


def parse_egl_dat(path: str | Path) -> Dict[str, Any]:
    """
    Parse EGL CARP .dat format (as distributed in common benchmarks).
    """
    p = Path(path)
    lines = [ln.rstrip("\n") for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
    by_prefix = {ln.split(":")[0].strip().upper(): ln for ln in lines if ":" in ln}

    num_vertices = _parse_int_after_colon(by_prefix["VERTICES"])
    num_req = _parse_int_after_colon(by_prefix["ARISTAS_REQ"])
    num_nonreq = _parse_int_after_colon(by_prefix["ARISTAS_NOREQ"])
    num_vehicles = _parse_int_after_colon(by_prefix["VEHICULOS"])
    capacity = float(_parse_int_after_colon(by_prefix["CAPACIDAD"]))
    depot_1based = _parse_int_after_colon(by_prefix["DEPOSITO"])
    depot = depot_1based - 1

    req_start = next(i for i, ln in enumerate(lines) if ln.strip().upper().startswith("LISTA_ARISTAS_REQ")) + 1
    nonreq_start = next(i for i, ln in enumerate(lines) if ln.strip().upper().startswith("LISTA_ARISTAS_NOREQ"))

    req_lines = lines[req_start:nonreq_start]
    nonreq_lines = lines[nonreq_start + 1 :]

    edge_pat = re.compile(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)\s+coste\s+([0-9]+)(?:\s+demanda\s+([0-9]+))?")

    required_edges: List[Edge] = []
    nonrequired_edges: List[Edge] = []
    required_cost: Dict[Edge, float] = {}
    required_demand: Dict[Edge, float] = {}
    all_cost: Dict[Edge, float] = {}

    for ln in req_lines:
        m = edge_pat.search(ln)
        if not m:
            continue
        i = int(m.group(1)) - 1
        j = int(m.group(2)) - 1
        c = float(int(m.group(3)))
        d = float(int(m.group(4))) if m.group(4) is not None else 0.0
        e = _canon_edge(i, j)
        required_edges.append(e)
        required_cost[e] = c
        required_demand[e] = d
        all_cost[e] = c

    for ln in nonreq_lines:
        if ln.strip().upper().startswith("DEPOSITO"):
            break
        m = edge_pat.search(ln)
        if not m:
            continue
        i = int(m.group(1)) - 1
        j = int(m.group(2)) - 1
        c = float(int(m.group(3)))
        e = _canon_edge(i, j)
        nonrequired_edges.append(e)
        all_cost[e] = c

    if len(required_edges) != num_req:
        raise ValueError(f"{p.name}: expected {num_req} required edges, parsed {len(required_edges)}")
    if len(nonrequired_edges) != num_nonreq:
        raise ValueError(f"{p.name}: expected {num_nonreq} non-required edges, parsed {len(nonrequired_edges)}")

    return {
        "name": p.stem,
        "path": str(p),
        "num_vertices": num_vertices,
        "depot": depot,
        "num_vehicles": num_vehicles,
        "capacity": capacity,
        "required_edges": required_edges,
        "nonrequired_edges": nonrequired_edges,
        "required_cost": required_cost,
        "required_demand": required_demand,
        "all_cost": all_cost,
    }


def _build_sparse_adj(edges: Iterable[Edge], edge_cost: Dict[Edge, float]) -> Dict[int, List[Tuple[int, float]]]:
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for i, j in edges:
        c = float(edge_cost[_canon_edge(i, j)])
        adj.setdefault(i, []).append((j, c))
        adj.setdefault(j, []).append((i, c))
    return adj


def _dijkstra_all_sources(nodes: Sequence[int], adj: Dict[int, List[Tuple[int, float]]]) -> Dict[int, Dict[int, float]]:
    dist_all: Dict[int, Dict[int, float]] = {}
    all_nodes = set(adj.keys())
    for u, nbrs in adj.items():
        all_nodes.add(u)
        for v, _ in nbrs:
            all_nodes.add(v)
    for src in nodes:
        dist: Dict[int, float] = {v: float("inf") for v in all_nodes}
        dist[src] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, src)]
        while pq:
            cur_d, u = heapq.heappop(pq)
            if cur_d > dist[u] + 1e-12:
                continue
            for v, w in adj.get(u, []):
                nd = cur_d + w
                if nd + 1e-12 < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        dist_all[src] = dist
    return dist_all


def _select_random_nodes_and_edges(
    all_nodes: Sequence[int],
    required_edges: Sequence[Edge],
    depot: int,
    required_limit: int,
    node_limit: int,
    rng: _random.Random,
) -> Tuple[List[int], List[Edge]]:
    """Random node-first sub-instance selection.

    Step 1 – Randomly pick node_limit nodes (depot always included).
    Step 2 – Collect all arcs (required + non-required) whose both endpoints
              are in selected_nodes; non-required ones become deadhead as-is.
    Step 3 – From the required edges between selected_nodes, randomly pick
              required_limit of them as service targets; the rest become
              deadhead edges.

    Returns
    -------
    selected_nodes : sorted list of chosen node ids
    chosen_req     : required_limit required edges (randomly sampled);
                     all other arcs between selected nodes are deadhead.
    """
    # Step 1: random node selection (depot always in)
    candidates = [n for n in all_nodes if n != depot]
    rng.shuffle(candidates)
    selected = sorted({depot} | set(candidates[: node_limit - 1]))

    # Step 2 (implicit): all arcs between selected nodes are available;
    # non-required ones stay deadhead automatically via metric closure.

    # Step 3: randomly sample required_limit service edges from covered req edges
    sel_set = set(selected)
    covered_req = [e for e in required_edges if e[0] in sel_set and e[1] in sel_set]
    rng.shuffle(covered_req)
    chosen_req = covered_req[:required_limit]

    return selected, chosen_req


def _build_regular_schedule_patterns(
    required_edges: Sequence[Edge],
    periods: Sequence[int],
    rng: _random.Random,
) -> Dict[Edge, List[frozenset[int]]]:
    """
    Custom rule:
    - fixed group : decision group = 50 : 70 (normalized shares)
    - fixed group: 항상 매일 방문 (frozenset(전체 기간)).
    - decision group:
      choose from two-day visit patterns only (no single-day options)
    """
    days = [int(d) for d in periods]
    if not days:
        raise ValueError("periods must be non-empty.")

    all_days_pat = frozenset(days)
    two_day_options = [frozenset(c) for c in combinations(days, 2)]

    fixed_weight = 50.0
    decision_weight = 70.0
    fixed_share = fixed_weight / (fixed_weight + decision_weight)

    edge_order = [_canon_edge(e[0], e[1]) for e in required_edges]
    rng.shuffle(edge_order)
    num_fixed = int(round(len(edge_order) * fixed_share))

    out: Dict[Edge, List[frozenset[int]]] = {}
    for idx, e in enumerate(edge_order):
        if idx < num_fixed:
            out[e] = [all_days_pat]
        else:
            if two_day_options:
                out[e] = list(two_day_options)
            else:
                out[e] = [all_days_pat]
    return out


def load_existing_instance(
    dat_path: str | Path,
    required_limit: int = 10,
    node_limit: int = 10,
    num_days: int = 4,
    use_full_instance: bool = True,
    schedule_mode: str = "all_days",
    vehicles_override: int | None = None,
    max_nodes: int = 10000,
    max_cg_iterations_per_node: int = 10000,
    gurobi_output: int = 0,
    arc_gurobi_output: int | None = None,
    alg_gurobi_output: int | None = None,
    pricing_max_columns: int = 0,
    require_proof_optimality: int = 0,
    algorithm_time_limit_s: float = 0.0,
    pricing_method: str = "labeling",
    pricing_ng_size: int = 8,
    cut_pricing_mode: str = "legacy",
    cut_pricing_dual_tol: float = 1e-15,
    use_coeff_dominance_filter: int = 1,
    coeff_dom_obj_tol: float = 1e-9,
    node_search_strategy: str = "dfs",
    eps_reduced_cost: float = 1e-4,
    use_dual_stabilization: bool = True,
    dual_stab_alpha: float = 0.5,
    discount_theta: float = 0.0,
    use_alns_initialization: bool = True,
    alns_iterations: int = 300,
    alns_destroy_fraction: float = 0.25,
    alns_seed: int = -1,
    alns_replicate_all_contexts: bool = True,
    phase1_col_cap: int = 3,
    use_cutting_plane_separation: bool = True,
    cut_separation_tol: float = 1e-7,
    cut_max_rounds_per_solve: int = 50,
    use_aggregation: int = 0,
    yao_style_pricing: int = 1,
    use_transformed_pricing_graph: int = 1,
    use_vehicle_lex_symmetry: int = 1,
) -> Dict[str, Any]:
    raw = parse_egl_dat(dat_path)

    if bool(use_full_instance):
        chosen_req = [_canon_edge(e[0], e[1]) for e in raw["required_edges"]]
        selected_nodes = list(range(int(raw["num_vertices"])))
        node_cap = int(raw["num_vertices"])
        req_limit = len(chosen_req)
    else:
        req_limit = max(1, int(required_limit))
        node_cap = max(3, int(node_limit))
        # Deterministic per-instance seed so results are reproducible.
        rng = _random.Random(raw["name"])
        all_nodes_list = list(range(int(raw["num_vertices"])))
        # Retry up to 200 times in case the random draw covers too few req edges.
        chosen_req = []
        selected_nodes = []
        for _ in range(200):
            cand_nodes, cand_req = _select_random_nodes_and_edges(
                all_nodes=all_nodes_list,
                required_edges=raw["required_edges"],
                depot=int(raw["depot"]),
                required_limit=req_limit,
                node_limit=node_cap,
                rng=rng,
            )
            if len(cand_req) >= req_limit:
                chosen_req = cand_req
                selected_nodes = cand_nodes
                break
        if len(chosen_req) < req_limit:
            raise ValueError(
                f"{Path(dat_path).name}: could not find {req_limit} required edges "
                f"within {node_cap} nodes after 200 random attempts."
            )

    sparse_edges = list(raw["required_edges"]) + list(raw["nonrequired_edges"])
    sparse_adj = _build_sparse_adj(sparse_edges, raw["all_cost"])
    dist_all = _dijkstra_all_sources(selected_nodes, sparse_adj)

    # Build metric closure for selected nodes (used by BP algorithm internals).
    edges: List[Edge] = []
    travel_cost: Dict[Edge, float] = {}
    for i_idx in range(len(selected_nodes)):
        i = selected_nodes[i_idx]
        for j_idx in range(i_idx + 1, len(selected_nodes)):
            j = selected_nodes[j_idx]
            d = float(dist_all[i][j])
            if d == float("inf"):
                raise ValueError(f"{Path(dat_path).name}: disconnected nodes in selected subgraph ({i}, {j}).")
            e = _canon_edge(i, j)
            edges.append(e)
            travel_cost[e] = d

    # Original sparse graph for arc-based solver (far fewer edges than metric closure).
    # Filter to edges whose both endpoints are in selected_nodes.
    selected_set = set(selected_nodes)
    arc_sparse_edges = sorted(set(
        _canon_edge(e[0], e[1]) for e in sparse_edges
        if e[0] in selected_set and e[1] in selected_set
    ))
    arc_sparse_travel_cost: Dict[Edge, float] = {
        e: float(raw["all_cost"][e]) for e in arc_sparse_edges
    }

    # Full EGL sparse road network (all vertices on file edges). Metric-closure distances
    # between selected nodes use Steiner nodes outside the selection; Yao pricing SP must
    # run on this graph so θ=0 matches closure-based pricing.
    road_sparse_edges: List[Edge] = sorted(
        {_canon_edge(e[0], e[1]) for e in sparse_edges}
    )
    road_sparse_travel_cost: Dict[Edge, float] = {
        e: float(raw["all_cost"][e]) for e in road_sparse_edges
    }

    required_edges = [_canon_edge(e[0], e[1]) for e in chosen_req]
    demand: Dict[Edge, float] = {e: float(raw["required_demand"][e]) for e in required_edges}
    service_cost: Dict[Edge, float] = {e: float(raw["required_cost"][e]) for e in required_edges}
    service_extra: Dict[Edge, float] = {e: float(service_cost[e] - travel_cost[e]) for e in required_edges}

    # Automatic day handling: periods are generated from num_days, then schedule patterns are built.
    periods = list(range(max(1, int(num_days))))
    # all_days / all_edges_daily: 각 필수 간 e에 허용 패턴이 전 기간 T 하나뿐 → 스케줄·커버상 매일(매 기간) 서비스.
    if schedule_mode in ("all_days", "all_edges_daily"):
        schedule_patterns: Dict[Edge, List[frozenset[int]]] = {
            e: [frozenset(periods)] for e in required_edges
        }
    elif schedule_mode == "regular":
        # Deterministic custom schedule generation (seeded by instance name).
        rng = _random.Random(raw["name"])  # deterministic per instance
        schedule_patterns = _build_regular_schedule_patterns(
            required_edges=required_edges,
            periods=periods,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported schedule_mode: {schedule_mode}")

    num_vehicles = int(raw["num_vehicles"] if vehicles_override is None else vehicles_override)
    if num_vehicles <= 0:
        raise ValueError("vehicles must be positive.")

    inst = {
        "instance_name": raw["name"],
        "instance_source_path": raw["path"],
        "use_full_instance": bool(use_full_instance),
        "schedule_mode": str(schedule_mode),
        "requested_node_limit": int(node_limit),
        "effective_node_limit": int(node_cap),
        "nodes": selected_nodes,
        "depot": int(raw["depot"]),
        "periods": periods,
        "vehicles": list(range(num_vehicles)),
        "edges": edges,
        "required_edges": required_edges,
        "travel_cost": travel_cost,
        "service_cost": service_cost,
        "service_extra": service_extra,
        "demand": demand,
        "capacity": float(raw["capacity"]),
        "schedule_patterns": schedule_patterns,
        # Sparse graph data for arc-based solver.
        "arc_sparse_edges": arc_sparse_edges,
        "arc_sparse_travel_cost": arc_sparse_travel_cost,
        "road_sparse_edges": road_sparse_edges,
        "road_sparse_travel_cost": road_sparse_travel_cost,
        "max_nodes": int(max_nodes),
        "max_cg_iterations_per_node": int(max_cg_iterations_per_node),
        "gurobi_output": int(gurobi_output),
        "arc_gurobi_output": int(gurobi_output if arc_gurobi_output is None else arc_gurobi_output),
        "alg_gurobi_output": int(gurobi_output if alg_gurobi_output is None else alg_gurobi_output),
        "pricing_max_columns": int(pricing_max_columns),
        "require_proof_optimality": int(require_proof_optimality),
        "algorithm_time_limit_s": float(algorithm_time_limit_s),
        "pricing_method": str(pricing_method),
        "pricing_ng_size": int(pricing_ng_size),
        "cut_pricing_mode": str(cut_pricing_mode),
        "cut_pricing_dual_tol": float(cut_pricing_dual_tol),
        "use_coeff_dominance_filter": int(use_coeff_dominance_filter),
        "coeff_dom_obj_tol": float(coeff_dom_obj_tol),
        "node_search_strategy": str(node_search_strategy),
        "eps_reduced_cost": float(eps_reduced_cost),
        "use_dual_stabilization": int(bool(use_dual_stabilization)),
        "dual_stab_alpha": float(dual_stab_alpha),
        "discount_theta": float(discount_theta),
        "use_alns_initialization": int(bool(use_alns_initialization)),
        "alns_iterations": int(alns_iterations),
        "alns_destroy_fraction": float(alns_destroy_fraction),
        "alns_seed": int(alns_seed),
        "alns_replicate_all_contexts": int(bool(alns_replicate_all_contexts)),
        "phase1_col_cap": int(phase1_col_cap),
        "use_cutting_plane_separation": int(bool(use_cutting_plane_separation)),
        "cut_separation_tol": float(cut_separation_tol),
        "cut_max_rounds_per_solve": int(cut_max_rounds_per_solve),
        "use_aggregation": int(use_aggregation),
        # Yao et al. (2021): discount-link duals in lower-layer SP (see src/pricing/node.py).
        "yao_style_pricing": int(yao_style_pricing),
        # Pricing on sparse -> less-sparse transformed graph:
        # required arcs stay explicit, pure deadhead subpaths become shortcut arcs.
        "use_transformed_pricing_graph": int(use_transformed_pricing_graph),
        # SimpleSPMaster: Σ λ_{t,0} >= Σ λ_{t,1} >= ... (sorted vehicle ids)
        "use_vehicle_lex_symmetry": int(use_vehicle_lex_symmetry),
    }
    return inst


def list_existing_instances(instance_dir: str | Path) -> List[Path]:
    base = Path(instance_dir)
    return sorted(base.glob("*.dat"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load/public CARP instance parser smoke utility.")
    parser.add_argument("--instance", type=str, required=True, help="Path to EGL .dat instance.")
    parser.add_argument("--full-instance", type=int, choices=[0, 1], default=1)
    parser.add_argument("--required-limit", type=int, default=10)
    parser.add_argument("--node-limit", type=int, default=10)
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--pricing-ng-size", type=int, default=8)
    parser.add_argument(
        "--schedule-mode",
        type=str,
        choices=["all_days", "all_edges_daily", "regular"],
        default="all_days",
        help="all_days / all_edges_daily: every required edge every period; regular: mixed patterns.",
    )
    args = parser.parse_args()
    inst = load_existing_instance(
        dat_path=args.instance,
        required_limit=args.required_limit,
        node_limit=args.node_limit,
        num_days=args.days,
        use_full_instance=bool(args.full_instance),
        schedule_mode=str(args.schedule_mode),
        pricing_ng_size=int(args.pricing_ng_size),
    )
    print(
        f"name={inst['instance_name']} nodes={len(inst['nodes'])} "
        f"required={len(inst['required_edges'])} edges={len(inst['edges'])} "
        f"vehicles={len(inst['vehicles'])} periods={len(inst['periods'])}"
    )


if __name__ == "__main__":
    main()
