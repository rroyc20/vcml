#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

from polygon_to_dat import _largest_component_graph, _load_polygon, _to_simple_undirected

Edge = Tuple[int, int]


def _demand_yao(rng: random.Random) -> int:
    value = int(round(rng.normalvariate(50.0, math.sqrt(10.0))))
    return max(1, value)


def _select_left_connected_nodes(G: nx.Graph, target_nodes: int) -> List[int]:
    if target_nodes <= 0:
        raise ValueError("target_nodes must be positive.")
    if G.number_of_nodes() < target_nodes:
        raise ValueError(f"Graph only has {G.number_of_nodes()} nodes, cannot pick {target_nodes}.")

    start = min(
        G.nodes(),
        key=lambda n: (float(G.nodes[n]["x"]), float(G.nodes[n]["y"]), int(n)),
    )
    selected: Set[int] = {int(start)}
    frontier: List[Tuple[float, float, int]] = []

    def push_neighbors(node: int) -> None:
        for nbr in G.neighbors(node):
            nbr = int(nbr)
            if nbr in selected:
                continue
            heapq.heappush(
                frontier,
                (float(G.nodes[nbr]["x"]), float(G.nodes[nbr]["y"]), nbr),
            )

    push_neighbors(int(start))
    while len(selected) < target_nodes:
        if not frontier:
            raise RuntimeError("Frontier exhausted before reaching target_nodes.")
        _, _, node = heapq.heappop(frontier)
        if node in selected:
            continue
        selected.add(node)
        push_neighbors(node)

    return sorted(
        selected,
        key=lambda n: (float(G.nodes[n]["x"]), float(G.nodes[n]["y"]), int(n)),
    )


def _expand_from_base_nodes(G: nx.Graph, base_nodes: List[int], target_nodes: int) -> List[int]:
    base_set = {int(n) for n in base_nodes}
    if not base_set:
        raise ValueError("base_nodes must be non-empty.")
    missing = sorted(n for n in base_set if n not in G.nodes)
    if missing:
        raise ValueError(f"Some base nodes are not in graph: {missing[:10]}")
    if len(base_set) > target_nodes:
        raise ValueError(f"base_nodes has {len(base_set)} nodes, larger than target_nodes={target_nodes}.")

    H = G.subgraph(base_set)
    if not nx.is_connected(H):
        raise ValueError("base_nodes must already induce a connected subgraph.")

    selected: Set[int] = set(base_set)
    frontier: List[Tuple[float, float, int]] = []
    frontier_seen: Set[int] = set()

    def push_neighbors(node: int) -> None:
        for nbr in G.neighbors(node):
            nbr = int(nbr)
            if nbr in selected or nbr in frontier_seen:
                continue
            heapq.heappush(frontier, (float(G.nodes[nbr]["x"]), float(G.nodes[nbr]["y"]), nbr))
            frontier_seen.add(nbr)

    for node in sorted(selected):
        push_neighbors(node)

    while len(selected) < target_nodes:
        if not frontier:
            raise RuntimeError("Frontier exhausted before reaching target_nodes.")
        _, _, node = heapq.heappop(frontier)
        if node in selected:
            continue
        selected.add(node)
        push_neighbors(node)

    return sorted(
        selected,
        key=lambda n: (float(G.nodes[n]["x"]), float(G.nodes[n]["y"]), int(n)),
    )


def _relabel_subgraph(G: nx.Graph, nodes: List[int]) -> Tuple[nx.Graph, Dict[int, int]]:
    H0 = G.subgraph(nodes).copy()
    ordered = sorted(
        H0.nodes(),
        key=lambda n: (float(H0.nodes[n]["x"]), float(H0.nodes[n]["y"]), int(n)),
    )
    mapping = {int(old): idx + 1 for idx, old in enumerate(ordered)}
    H = nx.Graph()
    for old in ordered:
        new = mapping[int(old)]
        H.add_node(
            new,
            original_id=int(old),
            x=float(H0.nodes[old]["x"]),
            y=float(H0.nodes[old]["y"]),
        )
    for u, v, data in H0.edges(data=True):
        H.add_edge(
            mapping[int(u)],
            mapping[int(v)],
            cost=int(data["cost"]),
            highways=tuple(data.get("highways", tuple())),
        )
    return H, mapping


def _choose_required_edges(G: nx.Graph, required_count: int, seed: int) -> Set[Edge]:
    edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in G.edges())
    if len(edges) < required_count:
        raise ValueError(f"Subgraph only has {len(edges)} edges, cannot pick {required_count} required edges.")
    rng = random.Random(seed)
    chosen = rng.sample(edges, required_count)
    return set(chosen)


def _nearest_subgraph_centroid_node(G: nx.Graph) -> int:
    cx = sum(float(data["x"]) for _, data in G.nodes(data=True)) / max(1, G.number_of_nodes())
    cy = sum(float(data["y"]) for _, data in G.nodes(data=True)) / max(1, G.number_of_nodes())
    best_node = None
    best_dist = float("inf")
    for node, data in G.nodes(data=True):
        dx = float(data["x"]) - cx
        dy = float(data["y"]) - cy
        dist2 = dx * dx + dy * dy
        if dist2 < best_dist:
            best_dist = dist2
            best_node = int(node)
    if best_node is None:
        raise ValueError("Could not determine depot node.")
    return best_node


def _write_dat(
    path: Path,
    name: str,
    G: nx.Graph,
    required_edges: Set[Edge],
    required_demands: Dict[Edge, int],
    depot_1based: int,
    num_vehicles: int,
    capacity: int,
    comment: str,
) -> Dict[str, object]:
    req_rows: List[Tuple[int, int, int, int]] = []
    nreq_rows: List[Tuple[int, int, int]] = []
    total_req_cost = 0

    for u, v, data in sorted(G.edges(data=True), key=lambda x: (min(x[0], x[1]), max(x[0], x[1]))):
        a, b = min(int(u), int(v)), max(int(u), int(v))
        cost = int(data["cost"])
        if (a, b) in required_edges:
            demand = int(required_demands[(a, b)])
            req_rows.append((a, b, cost, demand))
            total_req_cost += cost
        else:
            nreq_rows.append((a, b, cost))

    lines = [
        f" NOMBRE : {name}",
        f" COMENTARIO : {comment}",
        f" VERTICES : {G.number_of_nodes()}",
        f" ARISTAS_REQ : {len(req_rows)}",
        f" ARISTAS_NOREQ : {len(nreq_rows)}",
        f" VEHICULOS : {num_vehicles}",
        f" CAPACIDAD : {capacity}",
        " TIPO_COSTES_ARISTAS : EXPLICITOS ",
        f" COSTE_TOTAL_REQ : {total_req_cost}",
        " LISTA_ARISTAS_REQ :",
    ]
    for i, j, c, d in req_rows:
        lines.append(f" ( {i}, {j})   coste {c}   demanda {d}")
    lines.append(" LISTA_ARISTAS_NOREQ :")
    for i, j, c in nreq_rows:
        lines.append(f" ( {i}, {j})   coste {c}")
    lines.append(f" DEPOSITO :   {depot_1based}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "name": name,
        "path": str(path),
        "num_vertices": int(G.number_of_nodes()),
        "num_required_edges": int(len(req_rows)),
        "num_nonrequired_edges": int(len(nreq_rows)),
        "depot": int(depot_1based),
        "vehicles": int(num_vehicles),
        "capacity": int(capacity),
        "total_required_cost": int(total_req_cost),
        "total_demand": int(sum(d for _, _, _, d in req_rows)),
    }


def _plot_subset(
    path: Path,
    poly,
    full_graph: nx.Graph,
    subset_graph: nx.Graph,
    required_edges: Set[Edge],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    for u, v in full_graph.edges():
        ax.plot(
            [float(full_graph.nodes[u]["x"]), float(full_graph.nodes[v]["x"])],
            [float(full_graph.nodes[u]["y"]), float(full_graph.nodes[v]["y"])],
            color="#cbd5e1",
            linewidth=1.0,
            alpha=0.8,
            zorder=0,
        )

    for u, v, data in subset_graph.edges(data=True):
        a, b = min(int(u), int(v)), max(int(u), int(v))
        ax.plot(
            [float(subset_graph.nodes[u]["x"]), float(subset_graph.nodes[v]["x"])],
            [float(subset_graph.nodes[u]["y"]), float(subset_graph.nodes[v]["y"])],
            color="#0f766e" if (a, b) in required_edges else "#1d4ed8",
            linewidth=2.8 if (a, b) in required_edges else 2.0,
            alpha=0.95,
            zorder=2,
        )

    xs, ys = poly.exterior.xy
    ax.plot(xs, ys, color="#475569", linewidth=1.2, linestyle="--", alpha=0.9, zorder=1)
    ax.scatter(
        [float(data["x"]) for _, data in full_graph.nodes(data=True)],
        [float(data["y"]) for _, data in full_graph.nodes(data=True)],
        s=10,
        color="#94a3b8",
        alpha=0.9,
        zorder=1,
    )
    ax.scatter(
        [float(data["x"]) for _, data in subset_graph.nodes(data=True)],
        [float(data["y"]) for _, data in subset_graph.nodes(data=True)],
        s=32,
        color="#0f172a",
        alpha=0.95,
        zorder=3,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#cbd5e1", linewidth=2, label="full graph"),
            plt.Line2D([0], [0], color="#1d4ed8", linewidth=2, label="selected subset edge"),
            plt.Line2D([0], [0], color="#0f766e", linewidth=3, label="required edge"),
        ],
        loc="lower left",
    )
    ax.set_aspect("equal", adjustable="datalim")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a left-connected node subset .dat from a polygon road network.")
    parser.add_argument("--geojson", required=True, help="GeoJSON Polygon, Feature, or FeatureCollection string.")
    parser.add_argument("--name", default="left_subset", help="Instance name.")
    parser.add_argument("--out", required=True, help="Output .dat path.")
    parser.add_argument("--stats-out", default=None, help="Optional JSON path for stats.")
    parser.add_argument("--plot-out", default=None, help="Optional PNG path for visualization.")
    parser.add_argument("--network-type", default="drive", choices=["all", "all_public", "bike", "drive", "drive_service", "walk"])
    parser.add_argument("--node-count", type=int, default=30, help="Connected node count to keep.")
    parser.add_argument("--required-count", type=int, default=11, help="Random required edge count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for required edge selection.")
    parser.add_argument(
        "--base-node-ids-json",
        default=None,
        help="Optional JSON file containing original node ids to keep and expand from.",
    )
    parser.add_argument("--vehicles", type=int, default=5, help="Vehicle count to store in the .dat header.")
    parser.add_argument("--capacity-scale", type=float, default=1.25, help="Capacity = ceil(total_demand / vehicles * capacity_scale).")
    parser.add_argument("--demand-mode", default="cost", choices=["cost", "yao"], help="Required-edge demand generation rule.")
    args = parser.parse_args()

    if args.vehicles <= 0:
        raise ValueError("--vehicles must be positive.")
    if args.capacity_scale <= 0:
        raise ValueError("--capacity-scale must be positive.")

    poly = _load_polygon(args.geojson)
    ox.settings.log_console = False
    ox.settings.use_cache = True

    G_raw = ox.graph_from_polygon(poly, network_type=args.network_type, simplify=True)
    G_raw = _largest_component_graph(G_raw)
    G_full, _ = _to_simple_undirected(G_raw)

    if args.base_node_ids_json:
        base_payload = json.loads(Path(args.base_node_ids_json).read_text(encoding="utf-8"))
        if isinstance(base_payload, dict):
            base_nodes_raw = base_payload.get("selected_original_node_ids")
        else:
            base_nodes_raw = base_payload
        if not isinstance(base_nodes_raw, list):
            raise ValueError("base-node-ids-json must contain a JSON list or a dict with selected_original_node_ids.")
        selected_nodes = _expand_from_base_nodes(
            G_full,
            [int(n) for n in base_nodes_raw],
            args.node_count,
        )
    else:
        selected_nodes = _select_left_connected_nodes(G_full, args.node_count)
    subset_graph, mapping = _relabel_subgraph(G_full, selected_nodes)
    required_edges = _choose_required_edges(subset_graph, args.required_count, args.seed)
    rng = random.Random(args.seed)
    required_demands: Dict[Edge, int] = {}
    for edge in sorted(required_edges):
        if args.demand_mode == "yao":
            required_demands[edge] = _demand_yao(rng)
        else:
            required_demands[edge] = int(subset_graph.edges[edge]["cost"])
    depot = _nearest_subgraph_centroid_node(subset_graph)
    total_demand = sum(int(required_demands[e]) for e in required_edges)
    capacity = max(1, int(math.ceil((total_demand / args.vehicles) * args.capacity_scale)))
    comment = (
        f"Left-connected {args.node_count}-node subset with random {args.required_count} required edges"
    )

    stats = _write_dat(
        path=Path(args.out),
        name=args.name,
        G=subset_graph,
        required_edges=required_edges,
        required_demands=required_demands,
        depot_1based=depot,
        num_vehicles=int(args.vehicles),
        capacity=capacity,
        comment=comment,
    )
    stats["seed"] = int(args.seed)
    stats["node_count"] = int(args.node_count)
    stats["demand_mode"] = str(args.demand_mode)
    stats["base_node_ids_json"] = str(args.base_node_ids_json) if args.base_node_ids_json else None
    stats["selected_original_node_ids"] = [int(n) for n in selected_nodes]
    stats["selected_new_node_ids"] = {str(mapping[old]): int(old) for old in selected_nodes}
    stats["required_edges"] = [list(e) for e in sorted(required_edges)]
    stats["required_demands"] = {f"{a}-{b}": int(required_demands[(a, b)]) for a, b in sorted(required_edges)}

    if args.stats_out:
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.plot_out:
        _plot_subset(
            Path(args.plot_out),
            poly,
            G_full,
            subset_graph,
            required_edges,
            title=comment,
        )

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
