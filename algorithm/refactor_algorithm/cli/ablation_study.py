"""
Ablation Study: A-RMP Algorithm Component Analysis
===================================================
Runs experiments in a Table-6-style grid:

  Rows    : instance × days  (e.g., westjordan-S-d2, westjordan-S-d4, …)
  Columns : variant          (ALL_ON | NO_AGG | NO_SRI | NO_TRANSFORM)

Default instance set (full, no subsampling):
  1. yao-westjordan-S          (27 nodes, 11 req, 2 vehicles)
  2. seoul_left30_req11_yaoS   (30 nodes, 11 req, 3 vehicles)
  3. seoul_left50_req15_yaoS   (50 nodes, 15 req, 5 vehicles)

Default days : 2, 4, 6
Schedule mode: regular
  - days=2 : all-days only (no partial patterns)
  - days=4 : gap-2 patterns  {1,3} / {2,4}
  - days=6 : gap-3 patterns  {1,4}/{2,5}/{3,6}  +  gap-2 patterns  {1,3,5}/{2,4,6}

Usage examples
--------------
  # Full run (all 3 instances × 3 day configs × 4 variants, 3600s each)
  cd algorithm
  python refactor_algorithm/cli/ablation_study.py

  # Quick smoke test (S instance only, 2 days, 120s)
  python refactor_algorithm/cli/ablation_study.py \\
      --instances data/existing/yao/yao-westjordan-S.dat \\
      --days 2 --time-limit 120

  # Run only specific variants
  python refactor_algorithm/cli/ablation_study.py --variants ALL_ON,NO_AGG
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import multiprocessing
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from refactor_algorithm.app.batch_compare import (
    BatchCompareConfig,
    run_existing_batch,
)
from refactor_algorithm.engine.instances import load_existing_instance
from refactor_algorithm.engine.solvers import solve_arc


# ---------------------------------------------------------------------------
# Default instances
# ---------------------------------------------------------------------------
DEFAULT_INSTANCES = [
    "data/existing/yao/yao-westjordan-S.dat",
    "data/existing/custom/seoul_left30_req11_yaoS.dat",
    "data/existing/custom/seoul_left50_req15_yaoS.dat",
]

# Short labels used as row names in the summary table
_INSTANCE_LABELS: Dict[str, str] = {
    "yao-westjordan-S":        "westjordan-S",
    "seoul_left30_req11_yaoS": "seoul30-r11",
    "seoul_left50_req15_yaoS": "seoul50-r15",
}


def _inst_label(dat_path: str) -> str:
    stem = Path(dat_path).stem
    return _INSTANCE_LABELS.get(stem, stem)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "ALL_ON",
        "label": "All features ON",
        "use_aggregation": True,
        "enable_sri": 1,
        "use_transformed_pricing_graph": 1,
        "arc_flow_only": False,
    },
    {
        "name": "NO_AGG",
        "label": "No aggregation (A-RMP OFF)",
        "use_aggregation": False,
        "enable_sri": 1,
        "use_transformed_pricing_graph": 1,
        "arc_flow_only": False,
    },
    {
        "name": "NO_SRI",
        "label": "No SRI cuts",
        "use_aggregation": True,
        "enable_sri": 0,
        "use_transformed_pricing_graph": 1,
        "arc_flow_only": False,
    },
    {
        "name": "NO_TRANSFORM",
        "label": "No graph transformation",
        "use_aggregation": True,
        "enable_sri": 1,
        "use_transformed_pricing_graph": 0,
        "arc_flow_only": False,
    },
    {
        "name": "ARC_FLOW",
        "label": "Arc-flow MIP with flow connectivity (Gurobi direct, no BnP)",
        "use_aggregation": False,    # unused for this variant
        "enable_sri": 0,
        "use_transformed_pricing_graph": 0,
        "arc_flow_only": True,       # skip BnP entirely; call solve_arc directly
        "arc_model": "flow",
    },
    {
        "name": "ARC_CUTSET",
        "label": "Arc MIP with lazy cutset connectivity (former ARC_FLOW)",
        "use_aggregation": False,    # unused for this variant
        "enable_sri": 0,
        "use_transformed_pricing_graph": 0,
        "arc_flow_only": True,       # skip BnP entirely; call solve_arc directly
        "arc_model": "cutset",
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablation study (Table-6 style): instance × days × variant grid.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Instances
    p.add_argument(
        "--instances",
        default="",
        help=(
            "Comma-separated .dat file paths (relative to algorithm/ dir).\n"
            "If empty, falls back to --instance-dir / --num-instances."
        ),
    )
    p.add_argument(
        "--instance-dir",
        default="",
        help="Directory of .dat files. Used when --instances is empty.",
    )
    p.add_argument(
        "--num-instances", type=int, default=0,
        help="How many .dat files to pick from --instance-dir (0 = all).",
    )
    # Days
    p.add_argument("--days", default="2,4,6",
                   help="Comma-separated day counts to test (default: 2,4,6)")
    # Schedule
    p.add_argument("--schedule-mode", default="regular",
                   choices=["regular", "all_days", "all_edges_daily"],
                   help="Schedule generation rule (default: regular)")
    # Instance subsampling
    p.add_argument("--full-instance", type=int, choices=[0, 1], default=1,
                   help="1=full instance (default), 0=subsample by --required-limit / --node-limit")
    p.add_argument("--required-limit", type=int, default=14,
                   help="Required-edge cap when --full-instance 0")
    p.add_argument("--node-limit", type=int, default=50,
                   help="Graph node cap when --full-instance 0")
    # Solver
    p.add_argument("--vehicles", type=int, default=0,
                   help="0 = use instance default (recommended)")
    p.add_argument("--max-nodes", type=int, default=0,
                   help="BnB node limit per run (0 = unlimited = 999999)")
    p.add_argument("--max-cg-iter", type=int, default=1000000)
    p.add_argument("--time-limit", type=float, default=3600.0,
                   help="Wall-clock time limit per single run in seconds (default: 3600)")
    p.add_argument("--gurobi-output", type=int, choices=[0, 1], default=0,
                   help="Default Gurobi log for both solvers.")
    p.add_argument("--arc-output", type=int, choices=[0, 1], default=0,
                   help="Override arc-flow solver log.")
    p.add_argument("--alg-output", type=int, choices=[0, 1], default=0,
                   help="Override BnP solver log.")
    p.add_argument("--pricing-method", default="dp",           # inspect_bnp default
                   choices=["labeling", "dp", "cpp_dp", "cpp_dp_lex", "cpp_ng"])
    p.add_argument("--search-strategy", default="best_bound",
                   choices=["dfs", "best_bound"])
    # SRI params — defaults match inspect_bnp.py DEFAULT_INSPECT_ARGS
    p.add_argument("--root-only-sri", type=int, choices=[0, 1], default=0)
    p.add_argument("--max-sri-rounds", type=int, default=2)
    p.add_argument("--max-cuts-per-round", type=int, default=100)
    p.add_argument("--max-cuts-per-day", type=int, default=1000)
    p.add_argument("--min-sri-violation", type=float, default=1e-4)
    p.add_argument("--enable-sri-similarity-filter", type=int, choices=[0, 1], default=0)
    p.add_argument("--max-shared-edges-between-sri3", type=int, default=2)
    p.add_argument("--cut-separation-max-depth", type=int, default=100)
    p.add_argument("--phase1-col-cap", type=int, default=0,    # 0=off (inspect_bnp default)
                   help="Phase-I column cap per pricing call (0=off)")
    p.add_argument("--use-coeff-dominance-filter", type=int, choices=[0, 1], default=0)
    # Variants / output
    p.add_argument("--variants", default="ALL_ON,NO_AGG,NO_SRI,NO_TRANSFORM",
                   help="Comma-separated variant names to run (e.g. ARC_FLOW,ARC_CUTSET)")
    p.add_argument("--out-dir", default="algorithm/output/ablation",
                   help="Directory for JSON/CSV outputs")
    return p


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
def make_config(
    args: argparse.Namespace,
    dat_path: str,
    days: int,
    variant: Dict[str, Any],
) -> BatchCompareConfig:
    vehicles_override = None if int(args.vehicles) <= 0 else int(args.vehicles)
    _full = bool(int(args.full_instance))
    return BatchCompareConfig(
        instance_paths=[dat_path],
        full_instance=_full,
        required_limit=9999 if _full else int(args.required_limit),
        node_limit=9999 if _full else int(args.node_limit),
        days=days,
        schedule_mode=str(args.schedule_mode),
        vehicles_override=vehicles_override,
        max_nodes=int(args.max_nodes) if int(args.max_nodes) > 0 else 999999,
        max_cg_iterations_per_node=int(args.max_cg_iter),
        gurobi_output=int(args.gurobi_output),
        arc_gurobi_output=int(args.arc_output),
        alg_gurobi_output=int(args.alg_output),
        pricing_max_columns=0,
        require_proof_optimality=0,
        arc_time_limit_s=float(args.time_limit),
        alg_time_limit_s=float(args.time_limit),
        pricing_method=str(args.pricing_method),
        cut_pricing_mode="auto",
        use_coeff_dominance_filter=int(args.use_coeff_dominance_filter),
        node_search_strategy=str(args.search_strategy),
        eps_reduced_cost=1e-4,
        discount_theta=0.0,
        use_alns_initialization=False,
        alns_iterations=300,
        skip_arc=True,
        bnp_variant="aggregated" if variant["use_aggregation"] else "global_rmp",
        phase1_col_cap=int(args.phase1_col_cap),
        use_aggregation=bool(variant["use_aggregation"]),
        yao_style_pricing=1,
        pricing_ng_size=8,
        use_capacity_cuts=0,
        use_sri_cuts=0,
        sri_cardinality=3,
        enable_sri=int(variant["enable_sri"]),
        root_only_sri=int(args.root_only_sri),
        max_sri_rounds=int(args.max_sri_rounds),
        max_cuts_per_round=int(args.max_cuts_per_round),
        max_cuts_per_day=int(args.max_cuts_per_day),
        min_sri_violation=float(args.min_sri_violation),
        enable_sri_similarity_filter=int(args.enable_sri_similarity_filter),
        max_shared_edges_between_sri3=int(args.max_shared_edges_between_sri3),
        cut_root_only=0,
        cut_separation_max_depth=int(args.cut_separation_max_depth),
        use_transformed_pricing_graph=int(variant["use_transformed_pricing_graph"]),
    )


# ---------------------------------------------------------------------------
# Module-level worker functions (must be at top level for spawn pickling)
# ---------------------------------------------------------------------------
def _arc_worker(dat_path: str, days: int, args_ns, arc_model: str = "flow") -> dict:
    """Load instance and run an arc MIP. Called in a child process."""
    _full = bool(int(args_ns.full_instance))
    vehicles_override = None if int(args_ns.vehicles) <= 0 else int(args_ns.vehicles)
    inst = load_existing_instance(
        dat_path=dat_path,
        use_full_instance=_full,
        required_limit=9999 if _full else int(args_ns.required_limit),
        node_limit=9999 if _full else int(args_ns.node_limit),
        num_days=days,
        schedule_mode=str(args_ns.schedule_mode),
        vehicles_override=vehicles_override,
        gurobi_output=int(args_ns.gurobi_output),
        arc_gurobi_output=int(args_ns.arc_output),
        alg_gurobi_output=int(args_ns.alg_output),
        algorithm_time_limit_s=float(args_ns.time_limit),
    )
    return solve_arc(
        instance=inst,
        time_limit=float(args_ns.time_limit),
        connectivity_model=arc_model,
    )


def _bnp_worker(cfg) -> list:
    """Run BnP batch. Called in a child process."""
    return run_existing_batch(cfg)


# ---------------------------------------------------------------------------
# Hard wall-clock timeout via subprocess kill
# ---------------------------------------------------------------------------
def _run_in_worker(fn, args_tuple, result_queue: multiprocessing.Queue) -> None:
    """Target for the worker process. Puts (result_or_None, error_str) in the queue."""
    try:
        result = fn(*args_tuple)
        result_queue.put((result, None))
    except Exception as exc:
        result_queue.put((None, str(exc)))


def run_with_hard_timeout(
    fn,
    args_tuple: tuple,
    timeout_s: float,
) -> tuple[Any, str, bool]:
    """Run fn(*args_tuple) in a child process with a hard OS-level timeout.

    Returns (result, error_str, timed_out).
    If the process is still alive after timeout_s seconds, it is SIGKILL-ed
    and (None, 'HARD_TIMEOUT', True) is returned.
    """
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(target=_run_in_worker, args=(fn, args_tuple, q), daemon=True)
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        # Hard kill — terminates Gurobi presolve and anything else
        try:
            os.kill(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.join(timeout=5)
        return None, "HARD_TIMEOUT", True

    if not q.empty():
        result, error = q.get_nowait()
        return result, error or "", False

    return None, "worker_exited_no_result", False


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------
def _extract(row: Dict[str, Any]) -> Dict[str, Any]:
    profile = row.get("profile") or {}
    incumbent_obj = row.get("incumbent_obj")
    alg_obj = row.get("alg_obj")
    objective = incumbent_obj if incumbent_obj is not None else alg_obj
    alg_gap = row.get("alg_gap_pct")
    solver_gap = row.get("solver_gap_pct")
    gap_pct = alg_gap if alg_gap is not None else solver_gap
    return {
        "objective": objective,
        "arc_objective": None,
        "arc_best_bound": None,
        "arc_status": None,
        "root_lb":   profile.get("root_lb"),
        "gap_pct":   gap_pct,
        "nodes":     row.get("nodes"),
        "time_s":    round(row.get("alg_time") or 0.0, 2),
        "sri_cuts":  profile.get("sri_cuts_added", 0),
        "hit_tl":    bool(row.get("hit_time_limit")),
        "hit_nl":    bool(row.get("hit_node_limit")),
        "mode":      row.get("mode", ""),
        "error":     row.get("alg_error") or "",
    }


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Y" if v else "N"
    if isinstance(v, float):
        if v != v:
            return "nan"
        if abs(v) > 1e10:
            return "∞"
        return f"{v:.{digits}g}"
    return str(v)


# ---------------------------------------------------------------------------
# Table printer  (Table-6 style)
# ---------------------------------------------------------------------------
def print_table6_style(
    results: Dict[Tuple[str, str], Dict[str, Any]],
    row_labels: List[str],
    variant_names: List[str],
    time_limit: float,
) -> None:
    V = variant_names
    INST_W = 18
    COL_W = 11

    # Build header
    header_top = f"{'Inst.':<{INST_W}} | " + " | ".join(
        f"{v:^{2*COL_W+3}}" for v in V
    )
    header_bot = f"{'':<{INST_W}} | " + " | ".join(
        f"{'T.Time(s)':>{COL_W}}  {'Gap(%)':>{COL_W}}" for _ in V
    )
    sep = "-" * len(header_top)

    print("\n" + "=" * len(sep))
    print("  ABLATION STUDY RESULTS  (Table-6 style)")
    print(f"  Time limit per run: {time_limit:.0f}s   (* = hit limit)")
    print("=" * len(sep))
    print(header_top)
    print(header_bot)
    print(sep)

    prev_inst_base = None
    for row_label in row_labels:
        # e.g. "westjordan-S-d4" → base = "westjordan-S"
        base = "-".join(row_label.split("-")[:-1]) if "-d" in row_label else row_label
        if prev_inst_base and base != prev_inst_base:
            print(sep)
        prev_inst_base = base

        parts = [f"{row_label:<{INST_W}}"]
        for vname in V:
            r = results.get((row_label, vname))
            if r is None:
                parts.append(f"{'—':>{COL_W}}  {'—':>{COL_W}}")
            else:
                t_str = _fmt(r["time_s"], 2) + ("*" if r["hit_tl"] else "")
                g_str = _fmt(r["gap_pct"], 2) if r["gap_pct"] is not None else "—"
                parts.append(f"{t_str:>{COL_W}}  {g_str:>{COL_W}}")
        print(" | ".join(parts))

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    # Resolve instance paths: explicit list > instance-dir > default list
    instance_paths = [p.strip() for p in args.instances.split(",") if p.strip()]
    if not instance_paths and args.instance_dir:
        from refactor_algorithm.app.batch_compare import select_instance_paths
        n = int(args.num_instances) if int(args.num_instances) > 0 else 99999
        instance_paths = select_instance_paths(
            instance_dir=args.instance_dir,
            instances_csv="",
            num_instances=n,
        )
    if not instance_paths:
        instance_paths = list(DEFAULT_INSTANCES)

    days_list = [int(d) for d in args.days.split(",") if d.strip()]
    requested_variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    active_variants = [v for v in VARIANTS if v["name"] in requested_variants]

    if not instance_paths:
        print("No instances specified.")
        sys.exit(1)
    if not active_variants:
        print(f"No valid variants in: {args.variants}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_names = [v["name"] for v in active_variants]

    # Build ordered row labels  (instance × days)
    row_labels: List[str] = []
    row_meta: List[Tuple[str, str, int]] = []   # (label, dat_path, days)
    for dat in instance_paths:
        base_label = _inst_label(dat)
        for d in days_list:
            lbl = f"{base_label}-d{d}"
            row_labels.append(lbl)
            row_meta.append((lbl, dat, d))

    total_runs = len(row_meta) * len(active_variants)
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY — {timestamp}")
    print(f"  Instances : {[_inst_label(p) for p in instance_paths]}")
    print(f"  Days      : {days_list}")
    print(f"  Variants  : {variant_names}")
    print(f"  Rows      : {len(row_meta)}  ×  {len(active_variants)} variants = {total_runs} runs")
    print(f"  Time limit: {args.time_limit:.0f}s / run")
    print(f"  Schedule  : {args.schedule_mode}")
    print(f"  Output    : {out_dir}")
    print(f"{'='*70}\n")

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    all_raw: List[Dict[str, Any]] = []
    run_idx = 0

    for variant in active_variants:
        vname = variant["name"]
        print(f"\n{'─'*70}")
        print(f"  VARIANT: {vname}  —  {variant['label']}")
        print(f"    use_aggregation            = {variant['use_aggregation']}")
        print(f"    enable_sri                 = {variant['enable_sri']}")
        print(f"    use_transformed_graph      = {variant['use_transformed_pricing_graph']}")
        print(f"{'─'*70}")

        for lbl, dat_path, days in row_meta:
            run_idx += 1
            print(f"  [{run_idx:>3}/{total_runs}] {lbl} ...", end="  ", flush=True)

            t0 = time.perf_counter()
            ext: Dict[str, Any]

            if variant.get("arc_flow_only"):
                arc_model = str(variant.get("arc_model", "flow"))
                arc_mode = "arc_flow" if arc_model == "flow" else "arc_cutset"
                # ── Arc MIP: hard-kill wrapper ──────────────────────────────
                arc_res, err_str, timed_out = run_with_hard_timeout(
                    _arc_worker, (dat_path, days, args, arc_model), timeout_s=float(args.time_limit)
                )
                elapsed = time.perf_counter() - t0
                if timed_out or arc_res is None:
                    ext = {
                        "objective": None, "root_lb": None, "gap_pct": None,
                        "arc_objective": None, "arc_best_bound": None, "arc_status": None,
                        "nodes": None, "time_s": round(elapsed, 2),
                        "sri_cuts": 0, "hit_tl": True, "hit_nl": False,
                        "mode": arc_mode, "error": err_str or "HARD_TIMEOUT",
                    }
                else:
                    status_code = int(arc_res.get("status", -1))
                    arc_objective = arc_res.get("objective")
                    arc_best_bound = arc_res.get("best_bound")
                    ext = {
                        "objective": arc_objective,
                        "arc_objective": arc_objective,
                        "root_lb":   arc_best_bound,
                        "arc_best_bound": arc_best_bound,
                        "arc_status": status_code,
                        "gap_pct":   arc_res.get("gap_pct"),
                        "nodes":     None,
                        "time_s":    round(elapsed, 2),
                        "sri_cuts":  0,
                        "hit_tl":    status_code == 9 or timed_out,
                        "hit_nl":    False,
                        "mode":      arc_mode,
                        "error":     err_str or "",
                    }
            else:
                # ── BnP: hard-kill wrapper ───────────────────────────────────
                cfg = make_config(args, dat_path, days, variant)
                rows_or_none, err_str, timed_out = run_with_hard_timeout(
                    _bnp_worker, (cfg,), timeout_s=float(args.time_limit)
                )
                elapsed = time.perf_counter() - t0
                if timed_out or rows_or_none is None:
                    ext = {
                        "objective": None, "root_lb": None, "gap_pct": None,
                        "arc_objective": None, "arc_best_bound": None, "arc_status": None,
                        "nodes": None, "time_s": round(elapsed, 2),
                        "sri_cuts": 0, "hit_tl": True, "hit_nl": False,
                        "mode": "error", "error": err_str or "HARD_TIMEOUT",
                    }
                    results[(lbl, vname)] = ext
                    all_raw.append({
                        "variant": vname, "row_label": lbl,
                        "instance": _inst_label(dat_path), "days": days, **ext,
                    })
                    status = [f"{elapsed:.1f}s", "HARD_TIMEOUT"]
                    print(" | ".join(status))
                    continue
                row = rows_or_none[0] if rows_or_none else {}
                ext = _extract(row)
                ext["time_s"] = round(elapsed, 2)

            results[(lbl, vname)] = ext

            status = [f"{ext['time_s']:.1f}s"]
            if ext["gap_pct"] is not None:
                status.append(f"gap={ext['gap_pct']:.2f}%")
            if ext["objective"] is not None:
                status.append(f"obj={_fmt(ext['objective'], 2)}")
            if ext["hit_tl"]:
                status.append("HIT_TL")
            if ext["hit_nl"]:
                status.append("HIT_NL")
            if ext["error"]:
                status.append(f"ERR={ext['error'][:50]}")
            print(" | ".join(status))

            all_raw.append({
                "variant": vname, "row_label": lbl,
                "instance": _inst_label(dat_path), "days": days, **ext,
            })

    # Print summary table
    print_table6_style(results, row_labels, variant_names, args.time_limit)

    # Save CSV
    csv_keys = [
        "row_label", "instance", "days", "variant",
        "objective", "arc_objective", "arc_best_bound", "arc_status",
        "root_lb", "gap_pct", "nodes", "time_s",
        "sri_cuts", "hit_tl", "hit_nl", "mode", "error",
    ]
    csv_path = out_dir / f"ablation_summary_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_raw)
    print(f"\n  Summary CSV : {csv_path}")

    # Save master JSON
    master_path = out_dir / f"ablation_master_{timestamp}.json"
    with open(master_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "args": vars(args),
                "row_labels": row_labels,
                "variants": variant_names,
                "results": all_raw,
            },
            f, indent=2, default=str,
        )
    print(f"  Master JSON : {master_path}")
    print(f"\n{'='*70}\n  Done. ({run_idx} runs)\n{'='*70}\n")


if __name__ == "__main__":
    main()
