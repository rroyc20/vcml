#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def parse_carpdata_file(path: Path) -> Dict[str, object]:
    values = [int(x) for x in path.read_text(encoding="utf-8").split()]
    n = values[0]
    m = values[1]
    i = 2
    edges: List[Tuple[int, int, int, int]] = []
    for _ in range(m):
        u = int(values[i])
        v = int(values[i + 1])
        cost = int(values[i + 2])
        demand = int(values[i + 3])
        edges.append((u, v, cost, demand))
        i += 4
    vehicles = int(values[i])
    capacity = int(values[i + 1])
    lb = int(values[i + 2])
    ub = int(values[i + 3])
    return {
        "n": n,
        "m": m,
        "edges": edges,
        "vehicles": vehicles,
        "capacity": capacity,
        "lb": lb,
        "ub": ub,
    }


def write_egl_dat(path: Path, name: str, data: Dict[str, object], source_group: str) -> None:
    edges = list(data["edges"])
    required = [(u + 1, v + 1, c, d) for (u, v, c, d) in edges if int(d) > 0]
    nonreq = [(u + 1, v + 1, c) for (u, v, c, d) in edges if int(d) <= 0]
    total_req_cost = sum(c for _, _, c, _ in required)
    lines: List[str] = [
        f" NOMBRE : {name}",
        (
            " COMENTARIO : Imported from rafaelmartinelli/CARPData.jl "
            f"({source_group}); original bounds lb={int(data['lb'])} ub={int(data['ub'])}"
        ),
        f" VERTICES : {int(data['n'])}",
        f" ARISTAS_REQ : {len(required)}",
        f" ARISTAS_NOREQ : {len(nonreq)}",
        f" VEHICULOS : {int(data['vehicles'])}",
        f" CAPACIDAD : {int(data['capacity'])}",
        " TIPO_COSTES_ARISTAS : EXPLICITOS ",
        f" COSTE_TOTAL_REQ : {total_req_cost}",
        " LISTA_ARISTAS_REQ :",
    ]
    for i, j, c, d in required:
        lines.append(f" ( {i}, {j})   coste {c}   demanda {d}")
    lines.append(" LISTA_ARISTAS_NOREQ :")
    for i, j, c in nonreq:
        lines.append(f" ( {i}, {j})   coste {c}")
    lines.append(" DEPOSITO :   1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify_instance(name: str) -> Tuple[str, bool]:
    low = name.lower()
    if low.startswith("egl-g"):
        return ("egl_large", False)
    if low.startswith("egl-"):
        return ("egl", True)
    if low.startswith("gdb"):
        return ("gdb", False)
    if low.startswith("kshs"):
        return ("kshs", False)
    if low.startswith("val"):
        return ("val", False)
    if low.startswith(("c", "d", "e", "f")) and len(name) == 3:
        return ("beullens", False)
    if low.startswith("a") and len(name) == 4:
        return ("unknown_ab", False)
    if low.startswith("b") and len(name) == 4:
        return ("unknown_ab", False)
    return ("misc", False)


def import_repo_data(src_dir: Path, dst_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for src in sorted(src_dir.glob("*.dat")):
        name = src.stem
        group, skip = classify_instance(name)
        if skip:
            continue
        dst_dir = dst_root / group
        dst_dir.mkdir(parents=True, exist_ok=True)
        data = parse_carpdata_file(src)
        write_egl_dat(dst_dir / src.name, name, data, group)
        counts[group] = counts.get(group, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Import CARPData.jl instances into EGL-style existing data folders.")
    parser.add_argument("--src", required=True, help="Path to CARPData.jl data directory")
    parser.add_argument("--dst", required=True, help="Path to algorithm/data/existing")
    args = parser.parse_args()

    src_dir = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    if not src_dir.is_dir():
        raise SystemExit(f"Source directory not found: {src_dir}")
    dst_root.mkdir(parents=True, exist_ok=True)
    counts = import_repo_data(src_dir, dst_root)
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
