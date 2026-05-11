#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import Polygon, shape


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


def _graph_stats(G: Any) -> Dict[str, Any]:
    edge_lengths_m = []
    for _, _, data in G.edges(data=True):
        length = data.get("length")
        if length is not None:
            edge_lengths_m.append(float(length))
    return {
        "num_nodes": int(G.number_of_nodes()),
        "num_edges": int(G.number_of_edges()),
        "total_edge_length_m": round(sum(edge_lengths_m), 2),
        "avg_edge_length_m": round(sum(edge_lengths_m) / len(edge_lengths_m), 2) if edge_lengths_m else 0.0,
    }


def _polygon_outline(poly: Polygon) -> Iterable[Tuple[float, float]]:
    return list(poly.exterior.coords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and visualize an OSM road network inside a GeoJSON polygon.")
    parser.add_argument("--geojson", required=True, help="GeoJSON Polygon, Feature, or FeatureCollection string.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument(
        "--network-type",
        default="drive",
        choices=["all", "all_public", "bike", "drive", "drive_service", "walk"],
        help="OSMnx network type.",
    )
    parser.add_argument("--stats-out", default=None, help="Optional JSON path for graph statistics.")
    args = parser.parse_args()

    poly = _load_polygon(args.geojson)
    ox.settings.log_console = False
    ox.settings.use_cache = True

    G = ox.graph_from_polygon(poly, network_type=args.network_type, simplify=True)
    stats = _graph_stats(G)

    fig, ax = ox.plot_graph(
        G,
        bgcolor="white",
        node_size=6,
        node_color="#164e63",
        edge_color="#0f172a",
        edge_linewidth=0.8,
        show=False,
        close=False,
    )

    xs, ys = zip(*_polygon_outline(poly))
    ax.plot(xs, ys, color="#ef4444", linewidth=1.6, linestyle="--", alpha=0.85, label="input polygon")
    ax.legend(loc="lower left")
    ax.set_title(
        f"Road Network in Polygon\n"
        f"nodes={stats['num_nodes']}, edges={stats['num_edges']}, total_length={stats['total_edge_length_m']} m"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if args.stats_out:
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"image": str(out_path), "stats": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
