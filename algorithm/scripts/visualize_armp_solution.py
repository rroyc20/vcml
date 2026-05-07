#!/usr/bin/env python3
"""A-RMP 최적(또는 현재) 해를 차량·일별 경로로 시각화.

- Disaggregation 성공 시: 차량 k, 일 t, 필수 엣지(서비스), 전체 path_arcs(데드헤드 포함)를 색으로 구분.
- 좌표가 없는 EGL 인스턴스는 road_sparse 서브그래프에 spring layout 적용.

Usage:
  python scripts/visualize_armp_solution.py --out /tmp/routes.png
  python scripts/visualize_armp_solution.py --instance data/existing/egl/egl-e1-A.dat \\
      --required-limit 10 --node-limit 55 --days 4 --discount-theta 0.2 --out routes_e1a.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from existing_instance import _canon_edge, load_existing_instance
from src.master.aggregated_master import AggregatedMaster
from src.master.compare_arc_vs_bnp import _build_executable_solution_payload
from src.master.compare_global_rmp_bnp import GlobalRMPBnBTree
from src.pricing.node import BnBConfig, BnBNode, BestBoundSelector, DepthFirstSelector

Edge = Tuple[int, int]


def _arc_uv(arc: Any) -> Optional[Tuple[int, int]]:
    if isinstance(arc, tuple) and len(arc) >= 2:
        return int(arc[0]), int(arc[1])
    return None


def _expand_vehicle_routes(
    executable: Dict[str, Any],
    rmp: AggregatedMaster,
) -> List[Dict[str, Any]]:
    """Return rows: day, vehicle, serviced_required_edges, path_arcs, cost, ridx."""
    dis = executable.get("disaggregated")
    if isinstance(dis, dict) and dis.get("assignments"):
        assignments = dis["assignments"]
        cols = dis.get("agg_route_columns")
        if cols is None:
            cols = rmp.agg_route_columns
        out: List[Dict[str, Any]] = []
        for (t, k, ridx), val in assignments.items():
            if float(val) <= 0.5:
                continue
            col = cols[int(ridx)]
            out.append(
                {
                    "day": int(t),
                    "vehicle": int(k),
                    "serviced_required_edges": [tuple(e) for e in col.serviced_required_edges],
                    "path_arcs": [tuple(a) for a in col.path_arcs],
                    "cost": float(col.cost),
                    "ridx": int(ridx),
                }
            )
        out.sort(key=lambda r: (r["day"], r["vehicle"], r["ridx"]))
        return out

    # 집계 λ만 있는 경우: 일별로 익명 경로(가상 차량 번호)로 표시
    routes = executable.get("routes") or []
    out = []
    synth = 0
    for r in routes:
        if float(r.get("value", 0.0)) <= 0.5:
            continue
        synth += 1
        out.append(
            {
                "day": int(r["day"]),
                "vehicle": int(r.get("driver", synth - 1)),
                "serviced_required_edges": [tuple(e) for e in r.get("serviced_required_edges", [])],
                "path_arcs": [tuple(a) for a in r.get("path_arcs", [])],
                "cost": float(r.get("cost", 0.0)),
                "ridx": synth - 1,
                "aggregated_only": True,
            }
        )
    out.sort(key=lambda x: (x["day"], x["vehicle"]))
    return out


def _build_road_graph(inst: Dict[str, Any]) -> "Any":
    import networkx as nx

    nodes = set(int(n) for n in inst["nodes"])
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for e in inst.get("road_sparse_edges", []):
        u, v = int(e[0]), int(e[1])
        if u in nodes and v in nodes:
            G.add_edge(u, v)
    if G.number_of_edges() == 0:
        for e in inst.get("edges", []):
            u, v = int(e[0]), int(e[1])
            if u in nodes and v in nodes:
                G.add_edge(u, v)
    return G


def _print_summary(rows: List[Dict[str, Any]], inst: Dict[str, Any]) -> None:
    depot = int(inst["depot"])
    print("=== 차량·일별 요약 (서비스 필수엣지 + 경로 아크 수) ===")
    for r in rows:
        tag = " [aggregated λ, 차량 미배정]" if r.get("aggregated_only") else ""
        se = r["serviced_required_edges"]
        pa = r["path_arcs"]
        print(
            f"  day={r['day']}  vehicle={r['vehicle']}{tag}  "
            f"service_edges={len(se)}  path_arcs={len(pa)}  cost={r['cost']:.4g}"
        )
        if se:
            print(f"    서비스(필수): {se}")
        if pa:
            print(f"    경로(방향 아크 순서): {pa[:12]}{' ...' if len(pa) > 12 else ''}")
    print(f"depot={depot}  |K|={len(inst['vehicles'])}")


def _draw_solution(
    inst: Dict[str, Any],
    rows: List[Dict[str, Any]],
    out_path: Path,
    figscale: float,
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib import cm
    from matplotlib.patches import FancyArrowPatch

    G = _build_road_graph(inst)
    if G.number_of_nodes() == 0:
        raise RuntimeError("No nodes for layout.")
    pos = nx.spring_layout(G, seed=42, k=0.45 / max(1, int(G.number_of_nodes()) ** 0.5))

    depot = int(inst["depot"])
    req_global: Set[Edge] = {_canon_edge(int(e[0]), int(e[1])) for e in inst["required_edges"]}

    days = sorted({r["day"] for r in rows}) if rows else list(inst["periods"])
    n_days = max(1, len(days))
    fig_w = 5.2 * figscale * min(n_days, 3)
    fig_h = 3.8 * figscale * max(1, (n_days + 2) // 3)
    fig, axes = plt.subplots(
        nrows=(n_days + 2) // 3,
        ncols=min(3, n_days),
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    try:
        cmap = plt.colormaps["tab10"]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap("tab10")
    vehicles = sorted({int(r["vehicle"]) for r in rows}) if rows else [0]

    legend_handles = None
    for di, day in enumerate(days):
        ax = axes.flat[di]
        ax.set_title(f"day {day}")
        ax.set_aspect("equal")
        ax.axis("off")

        # 배경: 무방향 도로
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#dddddd", linewidth=0.8, zorder=0)

        for n in G.nodes():
            x, y = pos[n]
            if n == depot:
                ax.scatter([x], [y], c="black", s=120, zorder=3, marker="*")
            else:
                ax.scatter([x], [y], c="#888888", s=28, zorder=2)
            ax.text(x, y + 0.03, str(n), fontsize=7, ha="center", va="bottom")

        day_rows = [r for r in rows if r["day"] == day]
        for ri, r in enumerate(day_rows):
            k = int(r["vehicle"])
            color = cmap(vehicles.index(k) % 10 / 10.0) if k in vehicles else cmap(ri % 10 / 10.0)
            serviced = {_canon_edge(int(e[0]), int(e[1])) for e in r["serviced_required_edges"]}

            for arc in r["path_arcs"]:
                uv = _arc_uv(arc)
                if uv is None:
                    continue
                u, v = uv
                if u not in pos or v not in pos:
                    continue
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                ec = _canon_edge(u, v)
                is_svc = ec in serviced and ec in req_global
                lw = 2.4 if is_svc else 1.0
                ls = "-" if is_svc else "--"
                arr = FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="-|>",
                    mutation_scale=10 if is_svc else 7,
                    color=color,
                    linewidth=lw,
                    linestyle=ls,
                    zorder=4 if is_svc else 3,
                    alpha=0.9,
                )
                ax.add_patch(arr)

        if legend_handles is None:
            legend_handles = [
                plt.Line2D([0], [0], color="black", linestyle="", marker="*", markersize=12, label="depot"),
                plt.Line2D([0], [0], color=cmap(0.0), linewidth=2.4, label="service arc"),
                plt.Line2D([0], [0], color=cmap(0.0), linewidth=1.0, linestyle="--", label="deadhead"),
            ]
            ax.legend(handles=legend_handles, loc="upper left", fontsize=7)

    # 빈 서브플롯 숨김
    for j in range(len(days), len(axes.flat)):
        axes.flat[j].axis("off")

    fig.suptitle(
        f"{inst.get('instance_name', 'instance')}  |  vehicle-colored routes (solid=service, dashed=deadhead)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize A-RMP solution (per vehicle / day)")
    p.add_argument("--instance", type=str, default="data/existing/egl/egl-e1-A.dat")
    p.add_argument("--full-instance", type=int, choices=[0, 1], default=1)
    p.add_argument("--required-limit", type=int, default=10)
    p.add_argument("--node-limit", type=int, default=55)
    p.add_argument("--days", type=int, default=4)
    p.add_argument(
        "--schedule-mode",
        type=str,
        default="regular",
        choices=["regular", "all_days", "all_edges_daily"],
    )
    p.add_argument("--vehicles", type=int, default=0)
    p.add_argument("--discount-theta", type=float, default=0.2)
    p.add_argument("--max-nodes", type=int, default=2000)
    p.add_argument("--max-cg-iter", type=int, default=1000000)
    p.add_argument("--search-strategy", type=str, default="best_bound", choices=["dfs", "best_bound"])
    p.add_argument("--eps-rc", type=float, default=1e-4)
    p.add_argument("--phase1-col-cap", type=int, default=100)
    p.add_argument("--alns", type=int, choices=[0, 1], default=0)
    p.add_argument("--yao-pricing", type=int, choices=[0, 1], default=1)
    p.add_argument("--stab", action="store_true", help="dual stabilization on")
    p.add_argument("--out", type=str, default="armp_routes.png")
    p.add_argument("--figscale", type=float, default=1.0)
    p.add_argument("--no-figure", action="store_true", help="print summary only")
    args = p.parse_args()

    vehicles_override = None if args.vehicles <= 0 else args.vehicles
    inst = load_existing_instance(
        dat_path=args.instance,
        use_full_instance=bool(int(args.full_instance)),
        required_limit=args.required_limit,
        node_limit=args.node_limit,
        num_days=args.days,
        schedule_mode=args.schedule_mode,
        vehicles_override=vehicles_override,
        gurobi_output=0,
        max_nodes=args.max_nodes if args.max_nodes > 0 else 100000,
        max_cg_iterations_per_node=args.max_cg_iter,
        use_alns_initialization=bool(int(args.alns)),
        pricing_method="cpp_dp",
        node_search_strategy=args.search_strategy,
        eps_reduced_cost=args.eps_rc,
        use_dual_stabilization=bool(args.stab),
        phase1_col_cap=args.phase1_col_cap,
        discount_theta=float(args.discount_theta),
        use_aggregation=1,
        yao_style_pricing=int(args.yao_pricing),
        algorithm_time_limit_s=3600.0,
    )

    rmp = AggregatedMaster(inst)
    root = BnBNode(node_id=0, depth=0, master_problem=rmp)
    selector = BestBoundSelector() if args.search_strategy == "best_bound" else DepthFirstSelector()
    cfg = BnBConfig(
        eps_integrality=1e-6,
        eps_reduced_cost=float(args.eps_rc),
        max_cg_iterations_per_node=int(args.max_cg_iter),
        max_nodes=int(args.max_nodes) if args.max_nodes > 0 else 999999,
        max_time_s=None,
        verbose=False,
        use_dual_stabilization=bool(args.stab),
        phase1_col_cap=int(args.phase1_col_cap),
    )
    tree = GlobalRMPBnBTree(root_node=root, config=cfg, selector=selector)
    tree.solve()

    incumbent = tree.best_solution
    executable = _build_executable_solution_payload(rmp, incumbent)
    if not executable:
        print("No incumbent / executable solution to visualize.")
        sys.exit(1)

    rows = _expand_vehicle_routes(executable, rmp)
    _print_summary(rows, inst)

    if not args.no_figure:
        try:
            _draw_solution(inst, rows, Path(args.out), figscale=float(args.figscale))
        except ImportError as e:
            print("Install visualization deps: pip install matplotlib networkx", file=sys.stderr)
            raise SystemExit(1) from e


if __name__ == "__main__":
    main()
