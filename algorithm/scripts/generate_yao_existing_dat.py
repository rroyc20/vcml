#!/usr/bin/env python3
"""
Generate EGL-style .dat road instances for Yao et al. (2021) TR-B experiments.

The paper uses ConVRP on a road network (West Jordan). Exact GIS data are not in
the PDF; this script builds:
  - yao-fig1.dat: digitized toy network from Fig. 1 (10 nodes, 11 undirected links).
  - yao-westjordan-{S,M,L}.dat: connected random geometric graphs matching Table 3
    counts (|N|, |A|, |C|) with demands ~ N(50,10) rounded positive, Q=300.

Output: data/existing/yao/*.dat (compatible with scripts/existing_instance.parse_egl_dat).
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

Edge = Tuple[int, int]


def _canon(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def _write_dat(
    path: Path,
    name: str,
    n: int,
    depot_1: int,
    required: List[Tuple[int, int, int, int]],
    nonreq: List[Tuple[int, int, int]],
    num_vehicles: int,
    capacity: int,
) -> None:
    """required: (i,j,coste,demanda) 1-based; nonreq: (i,j,coste) 1-based."""
    total_req_cost = sum(t[2] for t in required)
    lines: List[str] = [
        f" NOMBRE : {name}",
        " COMENTARIO : Yao et al. (2021) TR-B proxy (see scripts/generate_yao_existing_dat.py)",
        f" VERTICES : {n}",
        f" ARISTAS_REQ : {len(required)}",
        f" ARISTAS_NOREQ : {len(nonreq)}",
        f" VEHICULOS : {num_vehicles}",
        f" CAPACIDAD : {capacity}",
        " TIPO_COSTES_ARISTAS : EXPLICITOS ",
        f" COSTE_TOTAL_REQ : {total_req_cost}",
        " LISTA_ARISTAS_REQ :",
    ]
    for i, j, c, d in required:
        lines.append(f" ( {i}, {j})   coste {c}   demanda {d}")
    lines.append(" LISTA_ARISTAS_NOREQ :")
    for i, j, c in nonreq:
        lines.append(f" ( {i}, {j})   coste {c}")
    lines.append(f" DEPOSITO :   {depot_1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fig1_instance() -> Tuple[int, int, List[Tuple[int, int, int, int]], List[Tuple[int, int, int]]]:
    """
    Fig. 1 road network (panel a), 1-based node ids:
      1 depot, 2..7 intersections, 8..10 customers.
    Coordinates (arbitrary scale) match relative layout from the figure.
    """
    # 1-based id -> (x, y)
    xy: Dict[int, Tuple[float, float]] = {
        1: (0.0, 0.0),
        2: (0.0, 2.0),
        3: (0.0, 4.0),
        4: (6.0, 4.0),
        5: (6.0, 0.0),
        6: (4.0, 0.0),
        7: (4.0, 2.0),
        8: (0.0, 5.0),
        9: (5.0, 5.0),
        10: (4.0, -1.0),
    }

    def dist(a: int, b: int) -> int:
        xa, ya = xy[a]
        xb, yb = xy[b]
        return int(round(10.0 * math.hypot(xa - xb, ya - yb)))

    all_pairs = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 1),
        (2, 7),
        (7, 6),
        (3, 8),
        (7, 9),
        (6, 10),
    ]
    # Arterial = main block + chords (discount-eligible as non-required in our CARP encoding).
    arterial: Set[Edge] = {
        _canon(1, 2),
        _canon(2, 3),
        _canon(3, 4),
        _canon(4, 5),
        _canon(5, 6),
        _canon(6, 1),
        _canon(2, 7),
        _canon(7, 6),
    }
    # Customer spurs = required service edges (CARP analogue of visiting customers).
    req_pairs = [(3, 8), (7, 9), (6, 10)]
    rng_fig = random.Random(42)
    required: List[Tuple[int, int, int, int]] = []
    for i, j in req_pairs:
        required.append((i, j, dist(i, j), _demand_yao(rng_fig)))

    nonreq: List[Tuple[int, int, int]] = []
    for i, j in all_pairs:
        e = _canon(i, j)
        if e in {_canon(r[0], r[1]) for r in required}:
            continue
        nonreq.append((i, j, dist(i, j)))

    return 10, 1, required, nonreq


def _demand_yao(rng: random.Random) -> int:
    v = int(round(rng.normalvariate(50.0, math.sqrt(10.0))))
    return max(1, v)


def _build_table3_network(n: int, target_m: int, num_customers: int, seed: int) -> Tuple[
    List[Tuple[int, int, int, int]], List[Tuple[int, int, int]]
]:
    """
    Random geometric graph in [0,1000]^2 with exactly target_m undirected edges,
    then mark num_customers edges as required (service), rest non-required.
    """
    rng = random.Random(seed)
    pts = [(rng.random() * 1000.0, rng.random() * 1000.0) for _ in range(n)]

    def dist_1based(a: int, b: int) -> int:
        xa, ya = pts[a - 1]
        xb, yb = pts[b - 1]
        return max(1, int(round(math.hypot(xa - xb, ya - yb))))

    edges: Set[Edge] = set()
    # Random spanning tree
    nodes = list(range(1, n + 1))
    rng.shuffle(nodes)
    for v in nodes[1:]:
        idx = nodes.index(v)
        pool = nodes[:idx]  # strict prefix so u != v (no self-loops)
        u = rng.choice(pool)
        edges.add(_canon(u, v))

    cand: List[Edge] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            e = (i, j)
            if e not in edges:
                cand.append(e)
    rng.shuffle(cand)
    for e in cand:
        if len(edges) >= target_m:
            break
        edges.add(e)
    if len(edges) < target_m:
        raise RuntimeError(f"Could not reach {target_m} edges (got {len(edges)})")

    edge_list = sorted(edges)
    # Choose required edges among links incident to a diverse customer set.
    deg: Dict[int, int] = {i: 0 for i in range(1, n + 1)}
    for i, j in edge_list:
        deg[i] += 1
        deg[j] += 1
    scored = sorted(edge_list, key=lambda e: -(deg[e[0]] + deg[e[1]]))
    required_edges = scored[:num_customers]
    req_set = set(required_edges)

    required: List[Tuple[int, int, int, int]] = []
    for i, j in required_edges:
        required.append((i, j, dist_1based(i, j), _demand_yao(rng)))

    nonreq: List[Tuple[int, int, int]] = []
    for i, j in edge_list:
        if (i, j) in req_set:
            continue
        nonreq.append((i, j, dist_1based(i, j)))

    return required, nonreq


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "existing" / "yao"
    out_dir.mkdir(parents=True, exist_ok=True)

    n, depot, req, nreq = _fig1_instance()
    _write_dat(
        out_dir / "yao-fig1.dat",
        "yao-fig1",
        n,
        depot,
        req,
        nreq,
        num_vehicles=5,
        capacity=300,
    )

    specs = [
        ("yao-westjordan-S", 27, 68, 11, 31001, 2),
        ("yao-westjordan-M", 55, 142, 23, 31002, 4),
        ("yao-westjordan-L", 133, 310, 45, 31003, 7),
    ]
    for name, nv, ne, nc, seed, num_vehicles in specs:
        r, nr = _build_table3_network(nv, ne, nc, seed)
        _write_dat(
            out_dir / f"{name}.dat",
            name,
            nv,
            1,
            r,
            nr,
            num_vehicles=num_vehicles,
            capacity=300,
        )

    print(f"Wrote instances under {out_dir}")


if __name__ == "__main__":
    main()
