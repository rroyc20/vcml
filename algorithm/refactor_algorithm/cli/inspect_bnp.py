from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from refactor_algorithm.app import inspection


# Edit these defaults and run this file directly when you want a quick inspect run
# without typing a long command. Any CLI arguments you pass will override this block
# because defaults are only used when no extra argv is provided.
DEFAULT_INSPECT_ARGS = [
    ### 1. Instance / size / full-or-subsample
    "--instance", "data/existing/egl/egl-e1-A.dat",  # any .dat path, e.g. data/existing/egl/egl-e1-A.dat
    "--full-instance", "0",  # 1=full instance, 0=subsample
    "--required-limit", "14",  # any positive int, used when --full-instance 0
    "--node-limit", "50",  # any positive int, used when --full-instance 0
    "--schedule-mode", "regular",  # regular, all_days, all_edges_daily
    "--days", "4",  # e.g. 2, 4
    "--vehicles", "0",  # 0=use instance default, or any positive int override
    "--max-nodes", "0",  # 0=unlimited, or any positive int
    "--max-cg-iter", "10000",  # any positive int
    ### 2. Cuts / separation
    "--use-capacity-cuts", "0",  # 0, 1
    "--use-sri-cuts", "1",  # 0, 1
    "--sri-cardinality", "3",  # currently fixed to 3
    "--enable-sri", "1",  # 0, 1
    "--root-only-sri", "0",  # 0, 1
    "--max-sri-rounds", "2",  # any nonnegative int
    "--max-cuts-per-round", "100",  # any positive int
    "--max-cuts-per-day", "1000",  # any positive int
    "--min-sri-violation", "1e-4",  # any nonnegative float
    "--enable-sri-similarity-filter", "0",  # 0, 1
    "--max-shared-edges-between-sri3", "2",  # any nonnegative int
    "--cut-root-only", "0",  # 0, 1
    "--cut-separation-max-depth", "100",  # 0=root only, or any nonnegative int
    "--cut-pricing-mode", "auto",  # legacy, bitmask, auto
    "--cut-pricing-dual-tol", "1e-15",  # any nonnegative float

    ### 3. Pricing
    "--pricing-method", "dp",  # labeling, dp, cpp_dp, cpp_dp_lex, cpp_ng
    "--pricing-ng-size", "5",  # any positive int, mainly for cpp_ng
    "--cpp-ng-empty-fallback", "none",  # labeling, dp, none
    "--cpp-core-variant", "default",  # default, rc_load_dom
    "--yao-pricing", "1",  # 0, 1
    "--use-coeff-dominance-filter", "0",  # 0, 1
    "--coeff-dom-obj-tol", "1e-9",  # any nonnegative float
    "--eps-rc", "1e-4",  # any positive float
    "--phase1-col-cap", "0", # 0=off, or any positive int
    "--use-transformed-pricing-graph", "1", # 0, 1
    ### 4. Search / heuristics / output
    "--search-strategy", "best_bound",  # dfs, best_bound
    "--alns-iters", "300",  # any nonnegative int
    "--discount-theta", "0",  # e.g. 0, 0.1
    "--alns", "0",  # 0, 1
    "--use-aggregation", "1",  # 0=SimpleSP start, 1=AggregatedMaster start
    "--use-vehicle-lex-symmetry", "0",  # 0, 1
    "--stab-alpha", "0.5",  # any nonnegative float
    "--out", "__AUTO__",  # auto -> algorithm/output/inspect_<instance>_<timestamp>.json

    ### 5. Optional flag-only args
    # "--stabilization-on",  # add this flag to enable dual stabilization
    # add this flag to suppress per-node detail output
]


def _arg_value(args: list[str], flag: str) -> str | None:
    for i in range(len(args) - 1):
        if args[i] == flag:
            return args[i + 1]
    return None


def _set_arg_value(args: list[str], flag: str, value: str) -> list[str]:
    out = list(args)
    for i in range(len(out) - 1):
        if out[i] == flag:
            out[i + 1] = value
            return out
    out.extend([flag, value])
    return out


def _auto_out_path(args: list[str]) -> str:
    instance_arg = _arg_value(args, "--instance") or "instance"
    instance_stem = Path(instance_arg).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("algorithm/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"inspect_{instance_stem}_{timestamp}.json")


def _resolve_output_path(args: list[str], *, using_defaults: bool) -> list[str]:
    out_arg = _arg_value(args, "--out")
    if out_arg == "__AUTO__":
        return _set_arg_value(args, "--out", _auto_out_path(args))
    if using_defaults and out_arg in {None, "", "newest_test.json"}:
        return _set_arg_value(args, "--out", _auto_out_path(args))
    return args


def main() -> None:
    old_argv = list(sys.argv)
    try:
        cli_args = sys.argv[1:]
        using_defaults = not cli_args
        effective_args = cli_args if cli_args else list(DEFAULT_INSPECT_ARGS)
        effective_args = _resolve_output_path(effective_args, using_defaults=using_defaults)
        sys.argv = ["inspect_bnp.py", *effective_args]
        inspection.main(effective_args)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
