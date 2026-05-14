#!/usr/bin/env python3
"""
Generate photo-based proxy instances for Yao et al. (2021) West Jordan panels.

Unlike `generate_yao_existing_dat.py`, these instances are not random geometric
graphs. They are hand-digitized from the paper's small / medium / large road
network panels:

  - orange roads -> non-required deadhead roads
  - blue roads   -> required service roads
  - green star   -> depot

The output uses EGL CARP `.dat` format and also writes lightweight SVG
visualizations for quick inspection.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Edge = Tuple[int, int]


@dataclass(frozen=True)
class Template:
    name: str
    depot_label: str
    points: Dict[str, Tuple[float, float]]
    required_roads: List[List[str]]
    nonrequired_roads: List[List[str]]
    req_target: int
    nonreq_target: int
    num_vehicles: int
    scale: float


def _canon(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _road_edges(polyline: Sequence[str]) -> List[Tuple[str, str]]:
    return [(str(polyline[i]), str(polyline[i + 1])) for i in range(len(polyline) - 1)]


def _allocate_segments(lengths: Sequence[float], target: int) -> List[int]:
    if target < len(lengths):
        raise ValueError(f"target={target} is smaller than edge count={len(lengths)}")
    if not lengths:
        return []
    total = float(sum(max(1e-9, x) for x in lengths))
    extras = target - len(lengths)
    alloc = [1] * len(lengths)
    if extras <= 0:
        return alloc
    raw = [extras * max(1e-9, x) / total for x in lengths]
    floors = [int(math.floor(v)) for v in raw]
    for i, fl in enumerate(floors):
        alloc[i] += fl
    used = sum(floors)
    remain = extras - used
    order = sorted(range(len(lengths)), key=lambda i: (raw[i] - floors[i], lengths[i]), reverse=True)
    for idx in order[:remain]:
        alloc[idx] += 1
    return alloc


def _demand_yao(rng: random.Random) -> int:
    v = int(round(rng.normalvariate(50.0, math.sqrt(10.0))))
    return max(1, v)


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
    total_req_cost = sum(t[2] for t in required)
    lines: List[str] = [
        f" NOMBRE : {name}",
        " COMENTARIO : Photo-digitized proxy from Yao et al. (2021) West Jordan panel",
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


def _write_svg(
    path: Path,
    coords: Dict[int, Tuple[float, float]],
    depot_id: int,
    required_edges: Iterable[Edge],
    nonrequired_edges: Iterable[Edge],
) -> None:
    req_set = {_canon(int(i), int(j)) for i, j in required_edges}
    nreq_set = {_canon(int(i), int(j)) for i, j in nonrequired_edges}
    xs = [float(x) for x, _ in coords.values()]
    ys = [float(y) for _, y in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = 12.0
    width = (max_x - min_x) + 2 * pad
    height = (max_y - min_y) + 2 * pad

    def _pt(node_id: int) -> Tuple[float, float]:
        x, y = coords[node_id]
        sx = (x - min_x) + pad
        sy = height - ((y - min_y) + pad)
        return sx, sy

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(math.ceil(width))}" height="{int(math.ceil(height))}" viewBox="0 0 {width:.1f} {height:.1f}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
    ]
    for e in sorted(nreq_set):
        x1, y1 = _pt(e[0])
        x2, y2 = _pt(e[1])
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#ff6b4a" stroke-width="2.2" stroke-linecap="round"/>'
        )
    for e in sorted(req_set):
        x1, y1 = _pt(e[0])
        x2, y2 = _pt(e[1])
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#3f7cff" stroke-width="2.4" stroke-linecap="round"/>'
        )
    for node_id in sorted(coords):
        x, y = _pt(node_id)
        if int(node_id) == int(depot_id):
            parts.append(
                f'<text x="{x:.2f}" y="{y + 4.0:.2f}" text-anchor="middle" font-size="14" fill="#14866d">★</text>'
            )
        else:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.8" fill="#14866d"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _subdivide_template(tpl: Template) -> Tuple[Dict[int, Tuple[float, float]], int, List[Tuple[int, int, int, int]], List[Tuple[int, int, int]]]:
    road_req = [e for poly in tpl.required_roads for e in _road_edges(poly)]
    road_nreq = [e for poly in tpl.nonrequired_roads for e in _road_edges(poly)]
    req_lengths = [_distance(tpl.points[a], tpl.points[b]) for a, b in road_req]
    nreq_lengths = [_distance(tpl.points[a], tpl.points[b]) for a, b in road_nreq]
    req_seg = _allocate_segments(req_lengths, tpl.req_target)
    nreq_seg = _allocate_segments(nreq_lengths, tpl.nonreq_target)

    label_to_id: Dict[str, int] = {}
    coords: Dict[int, Tuple[float, float]] = {}
    next_id = 1

    def get_label_id(label: str, xy: Tuple[float, float]) -> int:
        nonlocal next_id
        if label in label_to_id:
            return label_to_id[label]
        label_to_id[label] = next_id
        coords[next_id] = (float(xy[0]), float(xy[1]))
        next_id += 1
        return label_to_id[label]

    for label, xy in tpl.points.items():
        get_label_id(label, xy)

    rng = random.Random(tpl.name)
    required: List[Tuple[int, int, int, int]] = []
    nonreq: List[Tuple[int, int, int]] = []

    def emit_edge(label_u: str, label_v: str, *, is_required: bool, segments: int, idx: int) -> None:
        nonlocal next_id
        p1 = tpl.points[label_u]
        p2 = tpl.points[label_v]
        prev_id = get_label_id(label_u, p1)
        prev_xy = p1
        for s in range(1, segments):
            alpha = float(s) / float(segments)
            xy = (
                (1.0 - alpha) * float(p1[0]) + alpha * float(p2[0]),
                (1.0 - alpha) * float(p1[1]) + alpha * float(p2[1]),
            )
            mid_label = f"__{label_u}_{label_v}_{idx}_{s}"
            cur_id = get_label_id(mid_label, xy)
            cost = max(1, int(round(tpl.scale * _distance(prev_xy, xy))))
            if is_required:
                required.append((prev_id, cur_id, cost, _demand_yao(rng)))
            else:
                nonreq.append((prev_id, cur_id, cost))
            prev_id = cur_id
            prev_xy = xy
        end_id = get_label_id(label_v, p2)
        cost = max(1, int(round(tpl.scale * _distance(prev_xy, p2))))
        if is_required:
            required.append((prev_id, end_id, cost, _demand_yao(rng)))
        else:
            nonreq.append((prev_id, end_id, cost))

    for idx, ((u, v), segs) in enumerate(zip(road_req, req_seg)):
        emit_edge(u, v, is_required=True, segments=segs, idx=idx)
    for idx, ((u, v), segs) in enumerate(zip(road_nreq, nreq_seg)):
        emit_edge(u, v, is_required=False, segments=segs, idx=idx)

    depot_id = label_to_id[tpl.depot_label]
    return coords, depot_id, required, nonreq


def _template_s() -> Template:
    pts = {
        "depot": (10.0, 18.0),
        "a0": (18.0, 18.0),
        "a1": (18.0, 33.0),
        "a2": (18.0, 50.0),
        "a3": (18.0, 69.0),
        "a4": (18.0, 88.0),
        "b0": (37.0, 18.0),
        "b1": (37.0, 50.0),
        "b2": (37.0, 78.0),
        "mspur": (47.0, 51.0),
        "c1": (49.0, 50.0),
        "p0": (61.0, 42.0),
        "r0": (74.0, 20.0),
        "r1": (74.0, 46.0),
        "r2": (72.0, 74.0),
        "rt": (90.0, 70.0),
        "rb": (87.0, 20.0),
        "rs": (99.0, 20.0),
    }
    return Template(
        name="new-yao-westjordan-S",
        depot_label="depot",
        points=pts,
        required_roads=[
            ["a2", "b1", "b0"],
            ["b1", "mspur"],
            ["p0", "r1", "rb"],
            ["p0", "r2"],
        ],
        nonrequired_roads=[
            ["depot", "a0", "a1", "a2", "a3", "a4"],
            ["b0", "b1", "b2"],
            ["r0", "r1", "r2"],
            ["a3", "b2", "r2", "rt"],
            ["a0", "b0", "r0", "rb", "rs"],
            ["b1", "c1", "p0"],
        ],
        req_target=11,
        nonreq_target=24,
        num_vehicles=2,
        scale=6.5,
    )


def _template_m() -> Template:
    pts = {
        "a0": (18.0, 18.0),
        "a1": (18.0, 30.0),
        "a2": (18.0, 42.0),
        "a3": (18.0, 55.0),
        "a4": (18.0, 69.0),
        "a5": (18.0, 84.0),
        "a6": (18.0, 98.0),
        "ls1": (8.0, 26.0),
        "ls2": (8.0, 39.0),
        "ls3": (8.0, 54.0),
        "b0": (37.0, 18.0),
        "b1": (37.0, 42.0),
        "b2": (37.0, 58.0),
        "b3": (37.0, 80.0),
        "h1": (49.0, 42.0),
        "h2": (51.0, 58.0),
        "p0": (60.0, 46.0),
        "c0": (74.0, 18.0),
        "depot": (74.0, 34.0),
        "c1": (74.0, 48.0),
        "c2": (73.0, 80.0),
        "rt": (90.0, 76.0),
        "rb": (88.0, 18.0),
        "rs": (96.0, 28.0),
    }
    return Template(
        name="new-yao-westjordan-M",
        depot_label="depot",
        points=pts,
        required_roads=[
            ["a3", "b2", "b1", "a2", "ls2"],
            ["a1", "ls1"],
            ["a4", "ls3"],
            ["b1", "h1", "p0"],
            ["p0", "c1", "rb"],
            ["p0", "c2"],
            ["b0", "b1"],
        ],
        nonrequired_roads=[
            ["a0", "a1", "a2", "a3", "a4", "a5", "a6"],
            ["b0", "b1", "b2", "b3"],
            ["c0", "depot", "c1", "c2"],
            ["a5", "b3", "c2", "rt"],
            ["a0", "b0", "c0", "rb"],
            ["rb", "rs"],
            ["a3", "b2", "h2", "c1"],
        ],
        req_target=23,
        nonreq_target=52,
        num_vehicles=4,
        scale=6.0,
    )


def _template_l() -> Template:
    pts = {
        "a0": (16.0, 12.0),
        "a1": (16.0, 24.0),
        "a2": (16.0, 36.0),
        "a3": (16.0, 48.0),
        "a4": (16.0, 60.0),
        "a5": (16.0, 72.0),
        "a6": (16.0, 84.0),
        "a7": (16.0, 96.0),
        "a8": (16.0, 108.0),
        "b0": (32.0, 12.0),
        "b1": (32.0, 32.0),
        "b2": (32.0, 48.0),
        "b3": (32.0, 64.0),
        "b4": (32.0, 84.0),
        "b5": (32.0, 104.0),
        "c0": (52.0, 44.0),
        "c1": (58.0, 30.0),
        "c2": (64.0, 26.0),
        "d0": (78.0, 16.0),
        "d1": (78.0, 34.0),
        "d2": (78.0, 56.0),
        "depot": (78.0, 76.0),
        "d3": (78.0, 90.0),
        "d4": (76.0, 108.0),
        "rt": (98.0, 104.0),
        "mr": (98.0, 74.0),
        "br": (98.0, 40.0),
        "e0": (90.0, 10.0),
        "e1": (88.0, 26.0),
        "e2": (90.0, 52.0),
        "f0": (64.0, 62.0),
        "f1": (62.0, 78.0),
        "g0": (8.0, 30.0),
        "g1": (8.0, 48.0),
        "g2": (8.0, 70.0),
        "g3": (8.0, 92.0),
        "h0": (44.0, 20.0),
        "h1": (46.0, 58.0),
        "h2": (58.0, 58.0),
    }
    return Template(
        name="new-yao-westjordan-L",
        depot_label="depot",
        points=pts,
        required_roads=[
            ["a6", "b4", "b3", "a4", "g2"],
            ["a4", "b2", "c0"],
            ["a2", "b1", "h0"],
            ["g1", "a3", "b2"],
            ["g0", "a1"],
            ["g3", "a7"],
            ["c0", "h1", "h2", "d2"],
            ["f0", "d2", "e2"],
            ["f1", "depot", "d2"],
            ["c2", "e1", "br"],
            ["c1", "d1", "br"],
            ["b0", "c1", "d1"],
            ["d3", "mr"],
            ["d4", "rt"],
        ],
        nonrequired_roads=[
            ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
            ["b0", "b1", "b2", "b3", "b4", "b5"],
            ["d0", "d1", "d2", "depot", "d3", "d4"],
            ["a7", "b5", "d4", "rt"],
            ["a5", "b4", "d3", "mr"],
            ["a3", "b2", "h1", "d2"],
            ["a1", "b1", "c1", "d1", "br"],
            ["a0", "b0", "d0", "e0"],
            ["c0", "f0", "depot"],
            ["b1", "c0", "h2"],
            ["e0", "e1", "br"],
        ],
        req_target=45,
        nonreq_target=110,
        num_vehicles=7,
        scale=5.0,
    )


def _build_templates() -> List[Template]:
    return [_template_s(), _template_m(), _template_l()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "existing" / "yao"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tpl in _build_templates():
        coords, depot_id, required, nonreq = _subdivide_template(tpl)
        dat_path = out_dir / f"{tpl.name}.dat"
        svg_path = out_dir / f"{tpl.name}-visualization.svg"
        _write_dat(
            dat_path,
            tpl.name,
            n=len(coords),
            depot_1=int(depot_id),
            required=required,
            nonreq=nonreq,
            num_vehicles=tpl.num_vehicles,
            capacity=300,
        )
        _write_svg(
            svg_path,
            coords=coords,
            depot_id=depot_id,
            required_edges=[(i, j) for i, j, _, _ in required],
            nonrequired_edges=[(i, j) for i, j, _ in nonreq],
        )
        print(
            f"Wrote {dat_path.name}: |V|={len(coords)} "
            f"|Req|={len(required)} |NonReq|={len(nonreq)} depot={depot_id}"
        )
        print(f"Wrote {svg_path.name}")


if __name__ == "__main__":
    main()
