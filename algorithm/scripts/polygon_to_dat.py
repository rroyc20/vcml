#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon, shape

Edge = Tuple[int, int]

MAJOR_HIGHWAYS_DEFAULT: Set[str] = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
}


def _load_polygon(geojson_text: str) -> Polygon:
    payload = json.loads(geojson_text)
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if not features:
            raise ValueError("FeatureCollection has no features.")
        geom = features[0].get("geometry")
    elif payload.get("type") == "Feature":
        geom = payload.get("geometry")
    else:
        geom = payload

    poly = shape(geom)
    if not isinstance(poly, Polygon):
        raise ValueError(f"Expected Polygon geometry, got {poly.geom_type}.")
    return poly


def _round_cost(length_m: float) -> int:
    return max(1, int(round(float(length_m))))


def _normalize_highway_tags(highway: Any) -> Tuple[str, ...]:
    if highway is None:
        return tuple()
    if isinstance(highway, (list, tuple, set)):
        vals = [str(v).strip().lower() for v in highway if str(v).strip()]
    else:
        vals = [str(highway).strip().lower()]
    return tuple(sorted(set(vals)))


def _largest_component_graph(G: Any) -> Any:
    if G.number_of_nodes() == 0:
        raise ValueError("Fetched graph is empty.")
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    UG.add_edges_from((u, v) for u, v in G.edges())
    largest_nodes = max(nx.connected_components(UG), key=len)
    return G.subgraph(largest_nodes).copy()


def _to_simple_undirected(G: Any) -> Tuple[nx.Graph, Dict[int, int]]:
    H = nx.Graph()
    node_ids = sorted(int(n) for n in G.nodes())
    node_map = {node_id: idx + 1 for idx, node_id in enumerate(node_ids)}

    for node_id in node_ids:
        H.add_node(node_map[node_id], osmid=node_id, x=float(G.nodes[node_id]["x"]), y=float(G.nodes[node_id]["y"]))

    edge_meta: Dict[Edge, Dict[str, Any]] = {}
    for u, v, data in G.edges(data=True):
        if int(u) == int(v):
            continue
        a = node_map[int(u)]
        b = node_map[int(v)]
        e = (a, b) if a < b else (b, a)
        cost = _round_cost(data.get("length", 1.0))
        highways = set(_normalize_highway_tags(data.get("highway")))
        meta = edge_meta.setdefault(e, {"cost": cost, "highways": set()})
        meta["cost"] = min(int(meta["cost"]), cost)
        meta["highways"].update(highways)

    for (a, b), meta in edge_meta.items():
        H.add_edge(a, b, cost=int(meta["cost"]), highways=tuple(sorted(meta["highways"])))
    return H, node_map


def _nearest_depot_node(simple_graph: nx.Graph, poly: Polygon) -> int:
    cx = float(poly.centroid.x)
    cy = float(poly.centroid.y)
    best_node = None
    best_dist = float("inf")
    for node, data in simple_graph.nodes(data=True):
        dx = float(data["x"]) - cx
        dy = float(data["y"]) - cy
        dist2 = dx * dx + dy * dy
        if dist2 < best_dist:
            best_dist = dist2
            best_node = int(node)
    if best_node is None:
        raise ValueError("Could not determine depot node.")
    return best_node


def _is_required_edge(highways: Iterable[str], preset: str, major_highways: Set[str]) -> bool:
    tags = {str(tag).strip().lower() for tag in highways if str(tag).strip()}
    if preset == "all-required":
        return True
    if preset == "local-required-major-nonrequired":
        return not any(tag in major_highways for tag in tags)
    raise ValueError(f"Unsupported classification preset: {preset}")


def _write_dat(
    path: Path,
    name: str,
    G: nx.Graph,
    depot_1based: int,
    num_vehicles: int,
    capacity: int,
    classification_preset: str,
    major_highways: Set[str],
) -> Dict[str, Any]:
    required: List[Tuple[int, int, int, int]] = []
    nonrequired: List[Tuple[int, int, int]] = []
    total_req_cost = 0
    for u, v, data in sorted(G.edges(data=True), key=lambda x: (min(x[0], x[1]), max(x[0], x[1]))):
        cost = int(data["cost"])
        highways = tuple(data.get("highways", tuple()))
        if _is_required_edge(highways, classification_preset, major_highways):
            demand = cost
            total_req_cost += cost
            required.append((min(int(u), int(v)), max(int(u), int(v)), cost, demand))
        else:
            nonrequired.append((min(int(u), int(v)), max(int(u), int(v)), cost))

    lines = [
        f" NOMBRE : {name}",
        f" COMENTARIO : OSM road network converted from user polygon ({classification_preset})",
        f" VERTICES : {G.number_of_nodes()}",
        f" ARISTAS_REQ : {len(required)}",
        f" ARISTAS_NOREQ : {len(nonrequired)}",
        f" VEHICULOS : {num_vehicles}",
        f" CAPACIDAD : {capacity}",
        " TIPO_COSTES_ARISTAS : EXPLICITOS ",
        f" COSTE_TOTAL_REQ : {total_req_cost}",
        " LISTA_ARISTAS_REQ :",
    ]
    for i, j, c, d in required:
        lines.append(f" ( {i}, {j})   coste {c}   demanda {d}")
    lines.append(" LISTA_ARISTAS_NOREQ :")
    for i, j, c in nonrequired:
        lines.append(f" ( {i}, {j})   coste {c}")
    lines.append(f" DEPOSITO :   {depot_1based}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total_demand = sum(d for _, _, _, d in required)
    return {
        "name": name,
        "path": str(path),
        "num_vertices": int(G.number_of_nodes()),
        "num_required_edges": int(len(required)),
        "num_nonrequired_edges": int(len(nonrequired)),
        "depot": int(depot_1based),
        "vehicles": int(num_vehicles),
        "capacity": int(capacity),
        "total_required_cost": int(total_req_cost),
        "total_demand": int(total_demand),
        "classification_preset": classification_preset,
    }


def _plot_classified_graph(
    path: Path,
    G: nx.Graph,
    poly: Polygon,
    classification_preset: str,
    major_highways: Set[str],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    for u, v, data in G.edges(data=True):
        x1 = float(G.nodes[u]["x"])
        y1 = float(G.nodes[u]["y"])
        x2 = float(G.nodes[v]["x"])
        y2 = float(G.nodes[v]["y"])
        is_required = _is_required_edge(tuple(data.get("highways", tuple())), classification_preset, major_highways)
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#0f766e" if is_required else "#dc2626",
            linewidth=1.6 if is_required else 2.2,
            alpha=0.95,
        )

    xs, ys = poly.exterior.xy
    ax.plot(xs, ys, color="#334155", linewidth=1.3, linestyle="--", alpha=0.8)
    ax.scatter(
        [float(data["x"]) for _, data in G.nodes(data=True)],
        [float(data["y"]) for _, data in G.nodes(data=True)],
        s=8,
        color="#1e293b",
        alpha=0.75,
    )
    ax.set_title("Required vs Non-required Road Edges")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#0f766e", linewidth=2, label="required (local roads)"),
            plt.Line2D([0], [0], color="#dc2626", linewidth=2, label="non-required (major roads)"),
        ],
        loc="lower left",
    )
    ax.set_aspect("equal", adjustable="datalim")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an OSM road network in a GeoJSON polygon to EGL-style .dat.")
    parser.add_argument("--geojson", required=True, help="GeoJSON Polygon, Feature, or FeatureCollection string.")
    parser.add_argument("--name", default="polygon-road-network", help="Instance name to write in the .dat file.")
    parser.add_argument("--out", required=True, help="Output .dat path.")
    parser.add_argument("--stats-out", default=None, help="Optional JSON path for conversion metadata.")
    parser.add_argument("--plot-out", default=None, help="Optional PNG path for classified edge visualization.")
    parser.add_argument(
        "--network-type",
        default="drive",
        choices=["all", "all_public", "bike", "drive", "drive_service", "walk"],
        help="OSMnx network type.",
    )
    parser.add_argument("--vehicles", type=int, default=5, help="Vehicle count to store in the .dat header.")
    parser.add_argument(
        "--capacity-scale",
        type=float,
        default=1.25,
        help="Capacity = ceil(total_demand / vehicles * capacity_scale).",
    )
    parser.add_argument(
        "--classification-preset",
        default="local-required-major-nonrequired",
        choices=["all-required", "local-required-major-nonrequired"],
        help="How to split required vs non-required edges.",
    )
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
    G_simple, _ = _to_simple_undirected(G_raw)
    depot = _nearest_depot_node(G_simple, poly)

    total_demand = sum(
        int(data["cost"])
        for _, _, data in G_simple.edges(data=True)
        if _is_required_edge(tuple(data.get("highways", tuple())), args.classification_preset, MAJOR_HIGHWAYS_DEFAULT)
    )
    capacity = max(1, int(math.ceil((total_demand / args.vehicles) * args.capacity_scale)))

    out_path = Path(args.out)
    stats = _write_dat(
        out_path,
        name=args.name,
        G=G_simple,
        depot_1based=depot,
        num_vehicles=int(args.vehicles),
        capacity=capacity,
        classification_preset=args.classification_preset,
        major_highways=MAJOR_HIGHWAYS_DEFAULT,
    )
    stats["network_type"] = args.network_type
    stats["capacity_scale"] = float(args.capacity_scale)
    stats["major_highways"] = sorted(MAJOR_HIGHWAYS_DEFAULT)

    if args.stats_out:
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.plot_out:
        _plot_classified_graph(
            Path(args.plot_out),
            G_simple,
            poly,
            args.classification_preset,
            MAJOR_HIGHWAYS_DEFAULT,
        )

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
