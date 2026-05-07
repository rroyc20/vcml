"""Run ALNS-only on existing instances and compare against B&P results.

Usage examples
--------------
# Same defaults as compare_existing.py (9 instances, required=7, nodes=25, days=4)
python scripts/run_alns_only.py

# Specific instances
python scripts/run_alns_only.py --instances egl-e1-A.dat,egl-e1-C.dat

# More ALNS iterations
python scripts/run_alns_only.py --alns-iters 1000

# Compare against a saved B&P CSV
python scripts/run_alns_only.py --bnp-csv results_bnp.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from existing_instance import list_existing_instances, load_existing_instance
from src.util.alns import run_alns_initial_solution


def run_alns_batch(
    instance_paths: List[str],
    required_limit: int,
    node_limit: int,
    days: int,
    schedule_mode: str,
    vehicles_override: Optional[int],
    alns_iterations: int,
    alns_destroy_fraction: float,
    alns_seed: int,
    alns_replicate_all_contexts: bool,
    num_repeats: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for p in instance_paths:
        inst = load_existing_instance(
            dat_path=p,
            use_full_instance=False,
            required_limit=required_limit,
            node_limit=node_limit,
            num_days=days,
            schedule_mode=schedule_mode,
            vehicles_override=vehicles_override,
            gurobi_output=0,
            use_alns_initialization=True,
            alns_iterations=alns_iterations,
            alns_destroy_fraction=alns_destroy_fraction,
            alns_seed=alns_seed,
            alns_replicate_all_contexts=alns_replicate_all_contexts,
        )

        seed_base = int(alns_seed) if int(alns_seed) >= 0 else None

        best_obj = float("inf")
        worst_obj = float("-inf")
        obj_list: List[float] = []
        total_time = 0.0

        for rep in range(max(1, num_repeats)):
            seed_rep = None if seed_base is None else (seed_base + rep)
            t0 = time.perf_counter()
            result = run_alns_initial_solution(
                inst=inst,
                iterations=alns_iterations,
                destroy_fraction=alns_destroy_fraction,
                seed=seed_rep,
            )
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            obj = float(result["objective"])
            obj_list.append(obj)
            if obj < best_obj:
                best_obj = obj
            if obj > worst_obj:
                worst_obj = obj

        avg_obj = mean(obj_list)
        avg_time = total_time / max(1, num_repeats)

        rows.append(
            {
                "instance": inst["instance_name"],
                "nodes_used": len(inst["nodes"]),
                "required_used": len(inst["required_edges"]),
                "alns_best_obj": best_obj,
                "alns_avg_obj": avg_obj,
                "alns_worst_obj": worst_obj,
                "alns_avg_time_s": avg_time,
                "num_repeats": num_repeats,
            }
        )

    return rows


def load_bnp_csv(path: str) -> Dict[str, float]:
    """Load instance -> bnp_obj mapping from a compare_existing.py CSV output."""
    bnp_map: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("instance", "").strip()
            obj_str = row.get("alg_obj", row.get("arc_obj", "")).strip()
            if name and obj_str:
                try:
                    bnp_map[name] = float(obj_str)
                except ValueError:
                    pass
    return bnp_map


def print_report(rows: List[Dict[str, Any]], bnp_map: Dict[str, float]) -> None:
    has_bnp = bool(bnp_map)

    header = (
        "instance,nodes_used,required_used,"
        "alns_best_obj,alns_avg_obj,alns_worst_obj,alns_avg_time_s,num_repeats"
    )
    if has_bnp:
        header += ",bnp_obj,gap_pct"
    print(header)

    gaps: List[float] = []
    for r in rows:
        line = (
            f"{r['instance']},{r['nodes_used']},{r['required_used']},"
            f"{r['alns_best_obj']:.2f},{r['alns_avg_obj']:.2f},"
            f"{r['alns_worst_obj']:.2f},{r['alns_avg_time_s']:.4f},{r['num_repeats']}"
        )
        if has_bnp:
            bnp = bnp_map.get(r["instance"])
            if bnp is not None and bnp > 1e-12:
                gap = (r["alns_best_obj"] - bnp) / bnp * 100.0
                gaps.append(gap)
                line += f",{bnp:.2f},{gap:+.2f}%"
            else:
                line += ",n/a,n/a"
        print(line)

    print("--- summary ---")
    print(f"instances: {len(rows)}")
    alns_times = [r["alns_avg_time_s"] for r in rows]
    print(f"alns_avg_time_s: {mean(alns_times):.4f}")
    if gaps:
        print(f"gap_vs_bnp_avg: {mean(gaps):+.2f}%")
        print(f"gap_vs_bnp_best: {min(gaps):+.2f}%")
        print(f"gap_vs_bnp_worst: {max(gaps):+.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ALNS-only on existing EGL instances and compare against B&P."
    )
    parser.add_argument("--instance-dir", type=str, default="data/existing/egl")
    parser.add_argument("--instances", type=str, default="",
                        help="Comma-separated .dat basenames (default: first N sorted).")
    parser.add_argument("--num-instances", type=int, default=9)
    parser.add_argument("--required-limit", type=int, default=7)
    parser.add_argument("--node-limit", type=int, default=25)
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument(
        "--schedule-mode",
        type=str,
        default="regular",
        choices=["regular", "all_days", "all_edges_daily"],
    )
    parser.add_argument("--vehicles", type=int, default=0,
                        help="Vehicle override (0 = use instance value).")
    parser.add_argument("--alns-iters", type=int, default=300)
    parser.add_argument("--alns-destroy-frac", type=float, default=0.25)
    parser.add_argument("--alns-seed", type=int, default=-1,
                        help="Seed (-1 = instance-based deterministic).")
    parser.add_argument("--alns-replicate-all", type=int, choices=[0, 1], default=1)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of independent ALNS runs per instance (reports best/avg/worst).")
    parser.add_argument("--bnp-csv", type=str, default="",
                        help="Path to compare_existing.py CSV output for gap calculation.")
    args = parser.parse_args()

    candidates = list_existing_instances(args.instance_dir)
    if not candidates:
        raise RuntimeError(f"No .dat instances found in: {args.instance_dir}")

    if args.instances.strip():
        name_set = {s.strip() for s in args.instances.split(",") if s.strip()}
        selected = [str(p) for p in candidates if p.name in name_set]
    else:
        selected = [str(p) for p in candidates[: max(1, args.num_instances)]]

    vehicles_override = None if args.vehicles <= 0 else args.vehicles

    bnp_map: Dict[str, float] = {}
    if args.bnp_csv.strip():
        bnp_map = load_bnp_csv(args.bnp_csv.strip())

    rows = run_alns_batch(
        instance_paths=selected,
        required_limit=args.required_limit,
        node_limit=args.node_limit,
        days=args.days,
        schedule_mode=args.schedule_mode,
        vehicles_override=vehicles_override,
        alns_iterations=args.alns_iters,
        alns_destroy_fraction=args.alns_destroy_frac,
        alns_seed=args.alns_seed,
        alns_replicate_all_contexts=bool(args.alns_replicate_all),
        num_repeats=args.repeats,
    )
    print_report(rows, bnp_map)


if __name__ == "__main__":
    main()
