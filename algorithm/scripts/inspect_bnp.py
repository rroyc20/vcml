"""B&P Inspection Mode: per-node 상세 로그를 JSON Lines + 터미널 요약으로 출력.

각 노드마다 다음을 기록합니다:
  - node_id, depth, parent_id
  - 이번 노드에서 CG iteration 횟수
  - 각 CG iteration별: LP obj, 생성/추가 column 수, pricing time, rmp time
  - 수렴 시 최종 LP obj
  - branching 후보 목록 (family, value, target)
  - 선택된 branching candidate (어떤 rule이 적용됐는지)
  - 이번 노드에 걸린 branch constraints (부모로부터 상속된 것 포함)
  - 이번 노드에 활성화된 cut 목록
  - 이번 노드 처리 중 새로 추가된 cut 목록
  - 노드 종료 사유 (INTEGRAL / PRUNED_BOUND / PRUNED_INFEASIBLE / CG_LIMIT / TIME_LIMIT)
  - Phase-I 인공변수 합 (feasibility 확인용)
  - 현재 모델 크기: #vars, #constraints
  - 글로벌 LB/UB at node entry
  - open nodes count at node entry
  - 누적 column pool 크기
  - 노드별 CG 수렴 직후 최종 LP에서 (t, route)마다 Σ_k λ_r^{t,k} (stdout만)

Usage:
  python scripts/inspect_bnp.py [options]

  # 기본: full instance(전체 원본). 서브샘플은 --full-instance 0
  python scripts/inspect_bnp.py --instance data/existing/yao/yao-westjordan-S.dat

  # 처음 50노드만 출력
  python scripts/inspect_bnp.py --instance data/existing/yao/yao-westjordan-S.dat --max-nodes 50

  # JSON Lines 파일로 저장
  python scripts/inspect_bnp.py --instance data/existing/yao/yao-westjordan-S.dat --out inspect_yao_s.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from existing_instance import load_existing_instance
from src.master.aggregated_master import AggregatedMaster, solve_with_aggregated_algorithm
from src.master.compare_arc_vs_bnp import discount_objective_cost_per_edge
from src.pricing.node import (
    BnBConfig,
    BnBNode,
    BINARY_LIKE_BRANCH_FAMILIES,
    BranchCandidate,
    BranchConstraint,
    BestBoundSelector,
    DepthFirstSelector,
    NodeSolveResult,
    NodeStatus,
    branch_selection_key_from_parts,
    dist_to_nearest_integer,
)
from src.master.compare_global_rmp_bnp import GlobalRMPBnBTree


# ─────────────────────────────────────────────────────────────────────────────
# Per-iteration record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CGIterRecord:
    cg_iter: int
    lp_obj: float
    cols_generated: int
    cols_added: int
    rmp_time_s: float
    cut_separation_time_s: float
    pricing_time_s: float
    phase1_active: bool
    art_sum: float


# ─────────────────────────────────────────────────────────────────────────────
# Per-node record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeRecord:
    node_id: int
    depth: int
    parent_id: Optional[int]
    global_lb_at_entry: float
    global_ub_at_entry: float
    open_nodes_at_entry: int
    col_pool_size_at_entry: int
    global_lb_at_exit: Optional[float] = None
    global_ub_at_exit: Optional[float] = None

    # branch constraints active on this node
    active_constraints: List[Dict[str, Any]] = field(default_factory=list)
    active_cuts: List[Dict[str, Any]] = field(default_factory=list)
    cuts_added_this_node: List[Dict[str, Any]] = field(default_factory=list)

    # CG iteration detail
    cg_iters: List[CGIterRecord] = field(default_factory=list)
    total_cg_iters: int = 0
    total_cols_added: int = 0
    total_cols_generated: int = 0
    final_lp_obj: Optional[float] = None
    art_sum_at_converge: float = 0.0
    model_nvars: int = 0
    model_ncons: int = 0

    # branching decision
    branch_candidates: List[Dict[str, Any]] = field(default_factory=list)
    chosen_candidate: Optional[Dict[str, Any]] = None

    # outcome
    outcome: str = "UNKNOWN"   # INTEGRAL_* / PRUNED_* / RMP_INFEASIBLE / CG_LIMIT / TIME_LIMIT / SOLVED_LP
    disagg_result: Optional[str] = None   # success / failed_switched / skipped / not_applicable / None
    node_time_s: float = 0.0
    rmp_failure_msg: Optional[str] = None  # set when solve_rmp() raises RuntimeError (e.g. LP infeasible after cuts)
    # Full-node solve_stats totals (CG table omits addcol; stabilizer can add pricing not in a row).
    solve_rmp_time_s: float = 0.0            # total (lp + cut separation + bookkeeping)
    solve_rmp_lp_time_s: float = 0.0         # lp solve + dual extraction (cut separation excluded)
    solve_cut_separation_time_s: float = 0.0 # cut separation only
    solve_pricing_time_s: float = 0.0
    solve_addcol_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "global_lb_at_entry": _fmt(self.global_lb_at_entry),
            "global_ub_at_entry": _fmt(self.global_ub_at_entry),
            "global_lb_at_exit": _fmt(self.global_lb_at_exit),
            "global_ub_at_exit": _fmt(self.global_ub_at_exit),
            "open_nodes_at_entry": self.open_nodes_at_entry,
            "col_pool_size_at_entry": self.col_pool_size_at_entry,
            "active_constraints": self.active_constraints,
            "active_cuts": self.active_cuts,
            "cuts_added_this_node": self.cuts_added_this_node,
            "total_cg_iters": self.total_cg_iters,
            "total_cols_added": self.total_cols_added,
            "total_cols_generated": self.total_cols_generated,
            # LP objective at this node after CG (relaxation); not the global integer incumbent.
            "final_lp_obj": _fmt(self.final_lp_obj),
            "art_sum_at_converge": self.art_sum_at_converge,
            "model_nvars": self.model_nvars,
            "model_ncons": self.model_ncons,
            "branch_candidates": self.branch_candidates,
            "chosen_candidate": self.chosen_candidate,
            "outcome": self.outcome,
            "disagg_result": self.disagg_result,
            "rmp_failure_msg": self.rmp_failure_msg,
            "node_time_s": round(self.node_time_s, 6),
            "solve_rmp_time_s": round(self.solve_rmp_time_s, 6),
            "solve_rmp_lp_time_s": round(self.solve_rmp_lp_time_s, 6),
            "solve_cut_separation_time_s": round(self.solve_cut_separation_time_s, 6),
            "solve_pricing_time_s": round(self.solve_pricing_time_s, 6),
            "solve_addcol_time_s": round(self.solve_addcol_time_s, 6),
            "cg_iters": [
                {
                    "iter": r.cg_iter,
                    "lp_obj": _fmt(r.lp_obj),
                    "cols_gen": r.cols_generated,
                    "cols_added": r.cols_added,
                    "rmp_s": round(r.rmp_time_s, 4),
                    "cut_separation_s": round(r.cut_separation_time_s, 4),
                    "pricing_s": round(r.pricing_time_s, 4),
                    "phase1": r.phase1_active,
                    "art_sum": round(r.art_sum, 6),
                }
                for r in self.cg_iters
            ],
        }


def _fmt(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if math.isnan(v):
            return "nan"
        return round(v, 6)
    return v


def _branch_candidate_metrics(family: str, value: float) -> Dict[str, float]:
    """Inspect-only: metrics for JSON / terminal (matches BnBNode.choose_branch_candidate)."""
    v = float(value)
    dni = round(dist_to_nearest_integer(v), 6)
    d05 = round(abs(v - 0.5), 6)
    rank = d05 if family in BINARY_LIKE_BRANCH_FAMILIES else dni
    return {"dist_to_half": d05, "dist_nearest_int": dni, "rank_dist": rank}


def _branch_candidates_equal(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    if a is None or b is None:
        return False
    keys = ("family", "day", "driver", "value", "target")
    return all(a.get(k) == b.get(k) for k in keys)


def _inspect_print_sum_k_lambda_over_route(node: Any) -> None:
    """CG 수렴 직후 최종 LP에서 (t, route) 버킷별 질량을 stdout에만 출력.

    - SimpleSP(RMP): 동일 (t,route) 시그니처에 매달린 모든 lam 변수의 합 → Σ_k λ^{t,k}_r.
    - A-RMP: 일별 집계 변수 하나(agg_lam_t*_r*)이므로 합이 아니라 그 LP 값 λ^t_r (차량 차원 없음).
    """
    from gurobipy import GRB

    from src.util.initial_heuristic import canon_edge as _canon_edge

    rmp = getattr(node, "rmp", None)
    if rmp is None:
        return
    try:
        m = node._get_gurobi_model()
    except Exception:
        return
    if m is None or int(getattr(m, "Status", 0)) != int(GRB.OPTIMAL):
        return

    # AggregatedMaster가 switch_to_rmp_mode() 후에는 LP/λ가 _fallback_rmp(SimpleSP)에만 있음.
    def _rmp_for_lambda_snapshot(master: Any) -> Any:
        fb = getattr(master, "_fallback_rmp", None)
        if fb is not None and not bool(getattr(master, "_a_flag", True)):
            return fb
        return master

    rm = _rmp_for_lambda_snapshot(rmp)

    def _rk(col: Any) -> Tuple[int, Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]:
        se = tuple(
            sorted(_canon_edge(int(e[0]), int(e[1])) for e in getattr(col, "serviced_required_edges", ()))
        )
        pa = tuple(
            (int(a[0]), int(a[1]))
            for a in getattr(col, "path_arcs", ())
            if isinstance(a, tuple) and len(a) >= 2
        )
        return (int(getattr(col, "day", 0)), se, pa)

    def _accumulate(
        by_bucket: Dict[Tuple[int, Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]], float],
        names: List[str],
        n2i: Dict[str, int],
        cols: List[Any],
    ) -> None:
        for vname in names:
            ridx = int(n2i.get(vname, -1))
            if ridx < 0 or ridx >= len(cols):
                continue
            var = m.getVarByName(str(vname))
            if var is None:
                continue
            try:
                xv = float(var.X)
            except Exception:
                continue
            col = cols[ridx]
            key = _rk(col)
            by_bucket[key] = by_bucket.get(key, 0.0) + xv

    sums: Dict[Tuple[int, Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]], float] = {}
    used_simple_sp = False
    route_cols = getattr(rm, "route_columns", None)
    by_tk = getattr(rm, "lambda_var_names_by_day", None)
    n2i = getattr(rm, "lambda_var_name_to_index", None)

    if route_cols is not None and by_tk is not None and n2i is not None:
        sample = next(iter(by_tk.keys()), None)
        if sample is not None and isinstance(sample, tuple) and len(sample) == 2:
            for _names in by_tk.values():
                _accumulate(sums, _names, n2i, route_cols)
            used_simple_sp = True

    if not used_simple_sp:
        agg_cols = getattr(rm, "agg_route_columns", None)
        by_day = getattr(rm, "agg_lambda_var_names_by_day", None)
        n2i_agg = getattr(rm, "agg_lambda_name_to_index", None)
        if agg_cols is None or by_day is None or n2i_agg is None:
            print(
                f"[inspect] node {int(getattr(node, 'node_id', -1))}  Σ_k λ print skipped "
                f"(no route/lambda index on effective RMP: {type(rm).__name__})"
            )
            sys.stdout.flush()
            return
        for _names in by_day.values():
            _accumulate(sums, _names, n2i_agg, agg_cols)

    def _fmt_req_edges(es: Tuple[Tuple[int, int], ...], *, max_edges: int = 60) -> str:
        if len(es) <= max_edges:
            return str(list(es))
        head = list(es[:max_edges])
        return f"{head!s} ...(+{len(es) - max_edges} edges)"

    eps = 1e-8
    rows = [(s, rk) for rk, s in sums.items() if abs(s) > eps]
    rows.sort(key=lambda x: (x[1][0], -x[0]))
    nid = int(getattr(node, "node_id", -1))
    mode = "SimpleSP Σ_k" if used_simple_sp else "A-RMP λ^t_r (no k split)"
    print(f"[inspect] node {nid} final LP  Σ_k λ_r^(t,k)  by (t,route)  [{mode}]  nonzero={len(rows)}")
    val_lbl = "sum_k" if used_simple_sp else "lam_t_r"
    for s, rk in rows[:200]:
        t, se, pa = rk
        print(
            f"  t={t}  {val_lbl}={s:.8g}  |reqE|={len(se)}  "
            f"reqE={_fmt_req_edges(se)}  path_len={len(pa)}"
        )
    if len(rows) > 200:
        print(f"  ... ({len(rows) - 200} more rows omitted)")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Patched BnBNode: overrides solve_node to capture per-iteration data
# ─────────────────────────────────────────────────────────────────────────────

class InspectBnBNode(BnBNode):
    """BnBNode subclass that records per-CG-iteration data for inspection."""

    def __init__(self, *args: Any, inspector: "BnPInspector", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._inspector = inspector
        self._iter_log: List[CGIterRecord] = []
        # LP 솔루션이 살아 있는 solve_node 내부에서 직접 캡처
        self._captured_branch_candidates: List[Dict[str, Any]] = []
        self._captured_chosen_candidate: Optional[Dict[str, Any]] = None
        self._captured_model_nvars: int = 0
        self._captured_model_ncons: int = 0

    def _snapshot_model_size(self) -> None:
        """LP solve 직후 모델 크기 캡처 (update() 호출 없이)."""
        try:
            m = self._get_gurobi_model()
            # NumVars / NumConstrs 접근이 Gurobi lazy-update를 트리거할 수 있으므로
            # Gurobi 내부 update 없이 읽을 수 있는 방법으로 읽는다.
            self._captured_model_nvars = int(m.NumVars)
            self._captured_model_ncons = int(m.NumConstrs)
        except Exception:
            pass

    def _snapshot_branch_candidates(
        self,
        config: BnBConfig,
        candidates: Optional[List[BranchCandidate]] = None,
    ) -> None:
        """LP 솔루션이 살아 있는 상태에서 branching 후보를 캡처."""
        try:
            if candidates is None:
                candidates = self.extract_fractional_objects(config)
            self._captured_branch_candidates = [
                {
                    "family": c.family,
                    "value": round(float(c.value), 6),
                    "day": c.day,
                    "driver": c.driver,
                    "target": str(c.target),
                    **_branch_candidate_metrics(c.family, float(c.value)),
                }
                for c in candidates
            ]
            chosen = self.choose_branch_candidate(candidates)
            if chosen is not None:
                self._captured_chosen_candidate = {
                    "family": chosen.family,
                    "value": round(float(chosen.value), 6),
                    "day": chosen.day,
                    "driver": chosen.driver,
                    "target": str(chosen.target),
                    **_branch_candidate_metrics(chosen.family, float(chosen.value)),
                }
            else:
                self._captured_chosen_candidate = None
        except Exception as e:
            self._captured_branch_candidates = [{"error": str(e)}]
            self._captured_chosen_candidate = None

    def solve_node(self, config: BnBConfig, incumbent_ub: float) -> NodeSolveResult:
        """Wrap the original solve_node to intercept per-iteration data."""
        self._iter_log = []
        self._captured_branch_candidates = []
        self._captured_chosen_candidate = None
        self._active_config = config

        self.solve_stats = {
            "cg_iterations": 0,
            "farkas_rounds": 0,
            "rmp_time_s": 0.0,
            "rmp_lp_time_s": 0.0,
            "cut_separation_time_s": 0.0,
            "pricing_time_s": 0.0,
            "addcol_time_s": 0.0,
            "labels_generated": 0,
            "labels_expanded": 0,
            "backtrack_pruned": 0,
            "shortcut_returns": 0,
            "completion_bound_pruned": 0,
            "existing_sig_filtered": 0,
            "coeff_dominated_filtered": 0,
            "columns_generated": 0,
            "columns_added": 0,
            "columns_attempted_add": 0,
            "columns_skipped_duplicate": 0,
            "columns_skipped_dominated": 0,
            "zero_add_iterations": 0,
            "hit_cg_iteration_limit": False,
            "hit_time_limit": False,
            "pruned_before_cg": False,
        }

        from src.pricing.node import DualStabilizer
        stabilizer = None
        if bool(config.use_dual_stabilization):
            stabilizer = DualStabilizer(
                alpha=float(config.dual_stab_alpha),
                alpha_decay=float(config.dual_stab_alpha_decay),
                min_alpha=float(config.dual_stab_min_alpha),
            )

        def _is_pruned_by_bound(lp_value: Any) -> bool:
            try:
                ub = float(incumbent_ub)
                lb = float(lp_value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(ub) or not math.isfinite(lb):
                return False
            return lb >= ub - config.eps_integrality

        inherited_lb = getattr(self, "lower_bound", None)
        if _is_pruned_by_bound(inherited_lb):
            inherited_lb_f = float(inherited_lb)
            self.status = NodeStatus.PRUNED
            self.is_solved = True
            self.is_integral = False
            self.lp_obj_value = inherited_lb_f
            self.lower_bound = inherited_lb_f
            self.solve_stats["pruned_before_cg"] = True
            return NodeSolveResult(
                node_id=self.node_id,
                status=self.status,
                lower_bound=inherited_lb_f,
                is_integral=False,
            )

        self.apply_branch_constraints()
        self._restore_lp_basis_if_any()

        cg_iter = 0
        for _ in range(config.max_cg_iterations_per_node):
            if config.deadline_ts is not None and time.perf_counter() >= float(config.deadline_ts):
                self.solve_stats["hit_time_limit"] = True
                break
            cg_iter += 1
            self.solve_stats["cg_iterations"] += 1

            t0 = time.perf_counter()
            lp_obj, dual_values = self.solve_rmp(allow_farkas=bool(config.use_farkas_pricing))
            is_farkas = bool(dual_values.get("is_farkas", False))
            if not is_farkas:
                self._capture_lp_basis()
            rmp_elapsed = time.perf_counter() - t0
            cut_sep_elapsed = float(
                dual_values.get("cut_separation_time_s", getattr(self, "_last_cut_separation_time_s", 0.0))
            )
            if not math.isfinite(cut_sep_elapsed):
                cut_sep_elapsed = 0.0
            cut_sep_elapsed = max(0.0, min(cut_sep_elapsed, rmp_elapsed))
            rmp_lp_elapsed = max(0.0, rmp_elapsed - cut_sep_elapsed)
            self.solve_stats["rmp_time_s"] += rmp_elapsed
            self.solve_stats["rmp_lp_time_s"] += rmp_lp_elapsed
            self.solve_stats["cut_separation_time_s"] += cut_sep_elapsed

            if is_farkas:
                self.solve_stats["farkas_rounds"] += 1
                t1 = time.perf_counter()
                pricing_result = self.solve_subproblem(dual_values, config)
                pricing_elapsed = time.perf_counter() - t1
                self.solve_stats["pricing_time_s"] += pricing_elapsed
                meta = pricing_result.metadata if isinstance(pricing_result.metadata, dict) else {}
                self.solve_stats["labels_generated"] += int(meta.get("labels_generated", 0))
                self.solve_stats["labels_expanded"] += int(meta.get("labels_expanded", 0))
                self.solve_stats["backtrack_pruned"] += int(meta.get("backtrack_pruned", 0))
                self.solve_stats["shortcut_returns"] += int(meta.get("shortcut_returns", 0))
                self.solve_stats["completion_bound_pruned"] += int(meta.get("completion_bound_pruned", 0))
                self.solve_stats["existing_sig_filtered"] += int(meta.get("existing_sig_filtered", 0))
                self.solve_stats["coeff_dominated_filtered"] += int(meta.get("coeff_dominated_filtered", 0))
                cols_gen_this = int(meta.get("num_new_columns", len(pricing_result.new_columns)))
                self.solve_stats["columns_generated"] += cols_gen_this
                cols_added_this = 0
                if pricing_result.new_columns:
                    t2 = time.perf_counter()
                    cols_added_this = self.add_columns_to_rmp(pricing_result.new_columns)
                    self.solve_stats["addcol_time_s"] += time.perf_counter() - t2
                    self.solve_stats["columns_added"] += int(cols_added_this)
                    add_meta = getattr(self, "_last_addcol_stats", None)
                    if isinstance(add_meta, dict):
                        self.solve_stats["columns_attempted_add"] += int(add_meta.get("attempted", 0))
                        self.solve_stats["columns_skipped_duplicate"] += int(add_meta.get("skipped_duplicate", 0))
                        self.solve_stats["columns_skipped_dominated"] += int(add_meta.get("skipped_dominated", 0))
                self._iter_log.append(
                    CGIterRecord(
                        cg_iter=cg_iter,
                        lp_obj=float("nan"),
                        cols_generated=int(cols_gen_this),
                        cols_added=int(cols_added_this),
                        rmp_time_s=round(rmp_lp_elapsed, 5),
                        cut_separation_time_s=round(cut_sep_elapsed, 5),
                        pricing_time_s=round(pricing_elapsed, 5),
                        phase1_active=False,
                        art_sum=0.0,
                    )
                )
                if int(cols_added_this) > 0:
                    continue
                self.status = NodeStatus.INFEASIBLE
                self.is_solved = True
                self.is_integral = False
                return NodeSolveResult(
                    node_id=self.node_id,
                    status=NodeStatus.INFEASIBLE,
                    lower_bound=float("inf"),
                    is_integral=False,
                )

            phase1_active = self._artificial_sum() > 1e-8
            art_sum_now = self._artificial_sum()
            if phase1_active:
                self.solve_stats["phase1_iters"] = int(self.solve_stats.get("phase1_iters", 0)) + 1
            effective_config = config

            if stabilizer is not None:
                stabilizer.update(dual_values, lp_obj)

            if config.deadline_ts is not None and time.perf_counter() >= float(config.deadline_ts):
                self.solve_stats["hit_time_limit"] = True
                break

            pricing_duals = stabilizer.blend(dual_values) if stabilizer is not None else dual_values
            t1 = time.perf_counter()
            pricing_result = self.solve_subproblem(pricing_duals, effective_config)
            pricing_elapsed = time.perf_counter() - t1
            self.solve_stats["pricing_time_s"] += pricing_elapsed
            meta = pricing_result.metadata if isinstance(pricing_result.metadata, dict) else {}
            self.solve_stats["labels_generated"] += int(meta.get("labels_generated", 0))
            self.solve_stats["labels_expanded"] += int(meta.get("labels_expanded", 0))
            self.solve_stats["backtrack_pruned"] += int(meta.get("backtrack_pruned", 0))
            self.solve_stats["shortcut_returns"] += int(meta.get("shortcut_returns", 0))
            self.solve_stats["completion_bound_pruned"] += int(meta.get("completion_bound_pruned", 0))
            self.solve_stats["existing_sig_filtered"] += int(meta.get("existing_sig_filtered", 0))
            self.solve_stats["coeff_dominated_filtered"] += int(meta.get("coeff_dominated_filtered", 0))
            cols_gen_this = int(meta.get("num_new_columns", len(pricing_result.new_columns)))
            self.solve_stats["columns_generated"] += cols_gen_this
            for _pk, _pv in meta.items():
                if isinstance(_pk, str) and _pk.startswith("pricing_prof_"):
                    self.solve_stats[_pk] = self.solve_stats.get(_pk, 0.0) + float(_pv)

            cols_added_this = 0
            if pricing_result.new_columns:
                t2 = time.perf_counter()
                cols_added_this = self.add_columns_to_rmp(pricing_result.new_columns)
                self.solve_stats["addcol_time_s"] += time.perf_counter() - t2
                self.solve_stats["columns_added"] += int(cols_added_this)
                add_meta = getattr(self, "_last_addcol_stats", None)
                if isinstance(add_meta, dict):
                    self.solve_stats["columns_attempted_add"] += int(add_meta.get("attempted", 0))
                    self.solve_stats["columns_skipped_duplicate"] += int(add_meta.get("skipped_duplicate", 0))
                    self.solve_stats["columns_skipped_dominated"] += int(add_meta.get("skipped_dominated", 0))

            # Record this CG iteration
            self._iter_log.append(CGIterRecord(
                cg_iter=cg_iter,
                lp_obj=float(lp_obj),
                cols_generated=cols_gen_this,
                cols_added=int(cols_added_this),
                rmp_time_s=round(rmp_lp_elapsed, 5),
                cut_separation_time_s=round(cut_sep_elapsed, 5),
                pricing_time_s=round(pricing_elapsed, 5),
                phase1_active=phase1_active,
                art_sum=round(float(art_sum_now), 6),
            ))

            if pricing_result.new_columns and int(cols_added_this) > 0:
                if stabilizer is not None:
                    stabilizer.switch_to_in_step()
                continue
            if pricing_result.new_columns and int(cols_added_this) == 0:
                self.solve_stats["zero_add_iterations"] += 1

            if stabilizer is not None and stabilizer.is_in_phase:
                stabilizer.switch_to_out_step()
                t1b = time.perf_counter()
                pricing_result_out = self.solve_subproblem(dual_values, effective_config)
                self.solve_stats["pricing_time_s"] += time.perf_counter() - t1b
                meta_out = pricing_result_out.metadata if isinstance(pricing_result_out.metadata, dict) else {}
                self.solve_stats["coeff_dominated_filtered"] += int(meta_out.get("coeff_dominated_filtered", 0))
                self.solve_stats["columns_generated"] += int(meta_out.get("num_new_columns", len(pricing_result_out.new_columns)))
                for _pk, _pv in meta_out.items():
                    if isinstance(_pk, str) and _pk.startswith("pricing_prof_"):
                        self.solve_stats[_pk] = self.solve_stats.get(_pk, 0.0) + float(_pv)
                if pricing_result_out.new_columns:
                    t2b = time.perf_counter()
                    added_out = self.add_columns_to_rmp(pricing_result_out.new_columns)
                    self.solve_stats["addcol_time_s"] += time.perf_counter() - t2b
                    self.solve_stats["columns_added"] += int(added_out)
                    add_meta = getattr(self, "_last_addcol_stats", None)
                    if isinstance(add_meta, dict):
                        self.solve_stats["columns_attempted_add"] += int(add_meta.get("attempted", 0))
                        self.solve_stats["columns_skipped_duplicate"] += int(add_meta.get("skipped_duplicate", 0))
                        self.solve_stats["columns_skipped_dominated"] += int(add_meta.get("skipped_dominated", 0))
                    if int(added_out) > 0:
                        stabilizer.switch_to_in_step()
                        continue

            # CG converged
            if lp_obj >= incumbent_ub - config.eps_integrality:
                _inspect_print_sum_k_lambda_over_route(self)
                self.status = NodeStatus.PRUNED
                self.is_solved = True
                self.is_integral = False
                return NodeSolveResult(
                    node_id=self.node_id,
                    status=self.status,
                    lower_bound=lp_obj,
                    is_integral=False,
                )

            fractional = self.extract_fractional_objects(config)
            if not fractional:
                art_sum = self._artificial_sum()
                if art_sum > 1e-8:
                    _inspect_print_sum_k_lambda_over_route(self)
                    self.status = NodeStatus.PRUNED
                    self.is_integral = False
                    self.is_solved = True
                    self.solve_stats["artificial_sum_at_integral"] = float(art_sum)
                    return NodeSolveResult(
                        node_id=self.node_id,
                        status=self.status,
                        lower_bound=lp_obj,
                        is_integral=False,
                    )

                _rmp = getattr(self, "rmp", None)
                if (
                    _rmp is not None
                    and getattr(_rmp, "a_flag", False)
                    and not getattr(self, "_agg_mode_switched", False)
                    and hasattr(_rmp, "switch_to_rmp_mode")
                ):
                    _switch_on_integral = bool(int(getattr(_rmp, "inst", {}).get("armp_integral_switch_to_rmp", 1)))
                    _need_switch = _switch_on_integral
                    _armp_vals: Dict[str, float] = {}
                    try:
                        _armp_vals = {
                            v.VarName: float(v.X)
                            for v in self._get_gurobi_model().getVars()
                        }
                    except Exception:
                        _armp_vals = {}
                    if not _switch_on_integral and hasattr(_rmp, "try_disaggregate"):
                        _dis_fn = getattr(_rmp, "try_disaggregate_verified", None)
                        if callable(_dis_fn):
                            _disagg = _dis_fn(_armp_vals)
                        else:
                            _disagg = _rmp.try_disaggregate(_armp_vals)
                        if _disagg is not None:
                            setattr(self, "_disaggregated_solution", _disagg)
                            self.solve_stats["disagg_result"] = "success"
                        else:
                            _need_switch = True
                    if _need_switch:
                        self.solve_stats["disagg_result"] = "integral_switched"
                        _rmp.switch_to_rmp_mode(_armp_vals)
                        setattr(self, "_agg_mode_switched", True)
                        self._branch_constraints_applied = False
                        self.apply_branch_constraints()
                        if bool(config.use_dual_stabilization):
                            stabilizer = DualStabilizer(
                                alpha=float(config.dual_stab_alpha),
                                alpha_decay=float(config.dual_stab_alpha_decay),
                                min_alpha=float(config.dual_stab_min_alpha),
                            )
                        else:
                            stabilizer = None
                        continue
                    elif not _switch_on_integral:
                        self.solve_stats["disagg_result"] = "skipped"
                else:
                    # A-RMP 모드 아님 (SimpleSPMaster 직접 정수해)
                    self.solve_stats["disagg_result"] = "not_applicable"

                self.status = NodeStatus.INTEGRAL
                self.is_integral = True
                self.is_solved = True
                self.upper_bound = lp_obj
                _inspect_print_sum_k_lambda_over_route(self)
                return NodeSolveResult(
                    node_id=self.node_id,
                    status=self.status,
                    lower_bound=lp_obj,
                    is_integral=True,
                    best_integer_obj=lp_obj,
                )

            self.solve_stats["stab_alpha"] = float(stabilizer.alpha) if stabilizer else 0.0
            self.status = NodeStatus.SOLVED_LP
            self.is_integral = False
            self.is_solved = True
            _inspect_print_sum_k_lambda_over_route(self)
            # LP 솔루션이 살아 있는 지금 캡처 (process_node에서 var.X 접근 시 무효화될 수 있음)
            self._snapshot_model_size()
            self._snapshot_branch_candidates(config, candidates=fractional)
            return NodeSolveResult(
                node_id=self.node_id,
                status=self.status,
                lower_bound=lp_obj,
                is_integral=False,
            )

        # max iterations reached
        if bool(self.solve_stats.get("hit_time_limit", False)):
            lp_obj_final = self.lp_obj_value if self.lp_obj_value is not None else float("inf")
        else:
            self.solve_stats["hit_cg_iteration_limit"] = True
            lp_obj_final = self._iter_log[-1].lp_obj if self._iter_log else float("inf")

        self.status = NodeStatus.SOLVED_LP
        self.is_integral = False
        self.is_solved = True
        _inspect_print_sum_k_lambda_over_route(self)
        self._snapshot_model_size()
        self._snapshot_branch_candidates(config)
        return NodeSolveResult(
            node_id=self.node_id,
            status=self.status,
            lower_bound=lp_obj_final,
            is_integral=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Inspector: wraps GlobalRMPBnBTree, intercepts process_node + create_children
# ─────────────────────────────────────────────────────────────────────────────

class BnPInspector(GlobalRMPBnBTree):
    """GlobalRMPBnBTree subclass that records inspection data per node."""

    def __init__(
        self,
        *args: Any,
        out_path: Optional[Path] = None,
        max_inspect_nodes: int = 0,
        quiet_nodes: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.out_path = out_path
        self.max_inspect_nodes = max_inspect_nodes
        self.quiet_nodes = bool(quiet_nodes)
        self.node_records: List[NodeRecord] = []
        self._out_file = None
        if out_path is not None:
            self._out_file = open(out_path, "w", encoding="utf-8")

    def _col_pool_size(self, node: BnBNode) -> int:
        rmp = getattr(node, "rmp", None)
        if rmp is None:
            return 0
        # AggregatedMaster
        if hasattr(rmp, "agg_route_columns"):
            return len(rmp.agg_route_columns)
        # SimpleSPMaster
        if hasattr(rmp, "route_columns"):
            return len(rmp.route_columns)
        # fallback: count lambda vars in model
        try:
            m = rmp.model if hasattr(rmp, "model") else rmp
            return sum(
                1
                for v in m.getVars()
                if v.VarName.startswith(("lam_", "alm_", "agg_lam_"))
            )
        except Exception:
            return -1

    def _open_count(self) -> int:
        sel = self.selector
        if hasattr(sel, "_open"):
            return len(sel._open)
        return -1

    def _constraint_to_dict(self, bc: BranchConstraint) -> Dict[str, Any]:
        return {
            "family": bc.family,
            "sense": bc.sense,
            "rhs": bc.rhs,
            "day": bc.day,
            "driver": bc.driver,
            "target": str(bc.target),
        }

    def _candidate_to_dict(self, c: BranchCandidate) -> Dict[str, Any]:
        return {
            "family": c.family,
            "value": round(float(c.value), 6),
            "day": c.day,
            "driver": c.driver,
            "target": str(c.target),
            **_branch_candidate_metrics(c.family, float(c.value)),
        }

    def _cut_info_from_agg_key(self, key: Any, cname: str) -> Optional[Dict[str, Any]]:
        if not (isinstance(key, tuple) and len(key) >= 1):
            return None
        fam = str(key[0])
        if fam not in {"capacity_link_tk", "capacity_link_t"}:
            return None
        return {
            "family": fam,
            "cname": str(cname),
            "key": str(key),
        }

    def _current_cut_registry(self, node: BnBNode) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        rmp = getattr(node, "rmp", None)
        if rmp is None:
            return out
        registries: List[Dict[Any, Any]] = []
        main_reg = getattr(rmp, "aggregate_branch_constrs", None)
        if isinstance(main_reg, dict):
            registries.append(main_reg)
        fb = getattr(rmp, "_fallback_rmp", None)
        if fb is not None:
            fb_reg = getattr(fb, "aggregate_branch_constrs", None)
            if isinstance(fb_reg, dict) and fb_reg is not main_reg:
                registries.append(fb_reg)
        for reg in registries:
            for k, cname in reg.items():
                info = self._cut_info_from_agg_key(k, str(cname))
                if info is not None:
                    out[str(cname)] = info

        # fallback by constraint name prefix (in case registry is unavailable)
        if not out:
            try:
                m = rmp.model if hasattr(rmp, "model") else rmp
                for c in m.getConstrs():
                    cname = str(c.ConstrName)
                    fam: Optional[str] = None
                    if cname.startswith("cap_sep_") or cname.startswith("cap_link_"):
                        fam = "capacity_link"
                    if fam is not None:
                        out[cname] = {"family": fam, "cname": cname, "key": ""}
            except Exception:
                pass
        return out

    def process_node(self, node: BnBNode) -> NodeSolveResult:
        cuts_before = self._current_cut_registry(node)
        rec = NodeRecord(
            node_id=node.node_id,
            depth=node.depth,
            parent_id=node.parent_id,
            global_lb_at_entry=float(self.global_lower_bound),
            global_ub_at_entry=float(self.global_upper_bound),
            open_nodes_at_entry=self._open_count(),
            col_pool_size_at_entry=self._col_pool_size(node),
            active_constraints=[self._constraint_to_dict(bc) for bc in node.constraints],
            active_cuts=sorted(cuts_before.values(), key=lambda x: (x.get("family", ""), x.get("cname", ""))),
        )

        t_start = time.perf_counter()
        result = super().process_node(node)
        rec.node_time_s = time.perf_counter() - t_start
        cuts_after = self._current_cut_registry(node)
        rec.active_cuts = sorted(cuts_after.values(), key=lambda x: (x.get("family", ""), x.get("cname", "")))
        added_cut_names = sorted(set(cuts_after.keys()) - set(cuts_before.keys()))
        rec.cuts_added_this_node = [cuts_after[nm] for nm in added_cut_names]

        # Pull iter log from InspectBnBNode
        if hasattr(node, "_iter_log"):
            rec.cg_iters = node._iter_log

        stats = getattr(node, "solve_stats", {})
        _rerr = stats.get("rmp_runtime_error")
        if _rerr is not None:
            rec.rmp_failure_msg = str(_rerr)
        rec.total_cg_iters = int(stats.get("cg_iterations", 0))
        rec.total_cols_added = int(stats.get("columns_added", 0))
        rec.total_cols_generated = int(stats.get("columns_generated", 0))
        rec.solve_rmp_time_s = float(stats.get("rmp_time_s", 0.0))
        rec.solve_rmp_lp_time_s = float(stats.get("rmp_lp_time_s", max(0.0, rec.solve_rmp_time_s)))
        rec.solve_cut_separation_time_s = float(
            stats.get("cut_separation_time_s", max(0.0, rec.solve_rmp_time_s - rec.solve_rmp_lp_time_s))
        )
        rec.solve_pricing_time_s = float(stats.get("pricing_time_s", 0.0))
        rec.solve_addcol_time_s = float(stats.get("addcol_time_s", 0.0))
        _art_int = stats.get("artificial_sum_at_integral", None)
        if _art_int is not None:
            rec.art_sum_at_converge = float(_art_int)
        elif rec.cg_iters:
            # Most exits (PRUNED_BOUND, CG_LIMIT, …) never set artificial_sum_at_integral;
            # use last RMP snapshot so header matches the per-iter `art` column.
            rec.art_sum_at_converge = float(rec.cg_iters[-1].art_sum)
        else:
            rec.art_sum_at_converge = 0.0

        # Final LP obj
        if rec.cg_iters:
            rec.final_lp_obj = rec.cg_iters[-1].lp_obj
        else:
            lpv = getattr(node, "lp_obj_value", None)
            if lpv is not None:
                rec.final_lp_obj = float(lpv)

        # Model size: solve_node 내부에서 LP 솔루션이 유효할 때 캡처한 값 사용
        # (process_node에서 NumVars 접근 시 Gurobi lazy-update 트리거로 var.X 무효화됨)
        if hasattr(node, "_captured_model_nvars"):
            rec.model_nvars = int(node._captured_model_nvars)
            rec.model_ncons = int(node._captured_model_ncons)

        # disagg_result 기록
        disagg = stats.get("disagg_result", None)
        rec.disagg_result = disagg

        # Outcome (INTEGRAL을 세분화)
        if result.status == NodeStatus.INTEGRAL:
            if disagg == "success":
                rec.outcome = "INTEGRAL_DISAGG"
            elif disagg == "not_applicable":
                rec.outcome = "INTEGRAL_RMP"
            else:
                rec.outcome = "INTEGRAL_DIRECT"
        elif result.status == NodeStatus.PRUNED:
            if result.lower_bound >= self.global_upper_bound - self.config.eps_integrality:
                rec.outcome = "PRUNED_BOUND"
            else:
                rec.outcome = "PRUNED_INFEASIBLE"
        elif result.status == NodeStatus.INFEASIBLE:
            rec.outcome = "RMP_INFEASIBLE"
        elif bool(stats.get("hit_cg_iteration_limit", False)):
            rec.outcome = "CG_LIMIT"
        elif bool(stats.get("hit_time_limit", False)):
            rec.outcome = "TIME_LIMIT"
        else:
            rec.outcome = "SOLVED_LP"

        rec.global_lb_at_exit = float(self.global_lower_bound)
        rec.global_ub_at_exit = float(self.global_upper_bound)

        # Branching candidates: solve_node 내부에서 LP 솔루션이 살아 있을 때 캡처한 값 사용
        if result.status == NodeStatus.SOLVED_LP and not result.is_integral:
            if hasattr(node, "_captured_branch_candidates"):
                rec.branch_candidates = node._captured_branch_candidates
                rec.chosen_candidate = node._captured_chosen_candidate

        self.node_records.append(rec)
        self._emit(rec)

        if self.max_inspect_nodes > 0 and len(self.node_records) >= self.max_inspect_nodes:
            self.terminated_by_node_limit = True

        return result

    def _emit(self, rec: NodeRecord) -> None:
        d = rec.to_dict()
        line = json.dumps(d, ensure_ascii=False)
        if self._out_file is not None:
            self._out_file.write(line + "\n")
            self._out_file.flush()
        if not getattr(self, "quiet_nodes", False):
            _print_node_summary(rec)

    def close(self) -> None:
        if self._out_file is not None:
            self._out_file.close()
            self._out_file = None

    def create_children(self, parent: BnBNode, candidate: "BranchCandidate") -> Tuple[BnBNode, BnBNode]:
        left_bc, right_bc = parent.build_child_constraints(candidate)
        left_constraints = list(parent.constraints) + [left_bc]
        right_constraints = list(parent.constraints) + [right_bc]
        left_master = self._clone_master_problem(parent.rmp)
        right_master = self._clone_master_problem(parent.rmp)

        left_node = InspectBnBNode(
            node_id=self.nodes_created,
            depth=parent.depth + 1,
            master_problem=left_master,
            routes=list(parent.routes),
            constraints=left_constraints,
            parent_id=parent.node_id,
            inspector=self,
        )
        left_node.lower_bound = parent.lower_bound
        left_node._lp_basis_cache = getattr(parent, "_lp_basis_cache", None)
        self.nodes_created += 1

        right_node = InspectBnBNode(
            node_id=self.nodes_created,
            depth=parent.depth + 1,
            master_problem=right_master,
            routes=list(parent.routes),
            constraints=right_constraints,
            parent_id=parent.node_id,
            inspector=self,
        )
        right_node.lower_bound = parent.lower_bound
        right_node._lp_basis_cache = getattr(parent, "_lp_basis_cache", None)
        self.nodes_created += 1

        return left_node, right_node


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print per node
# ─────────────────────────────────────────────────────────────────────────────

def _print_node_summary(rec: NodeRecord) -> None:
    sep = "─" * 72
    print(sep)
    disagg_tag = f"  disagg={rec.disagg_result}" if rec.disagg_result else ""
    print(
        f"[Node {rec.node_id:>4d}]  depth={rec.depth}  parent={rec.parent_id}  "
        f"outcome={rec.outcome}{disagg_tag}  time={rec.node_time_s:.3f}s"
    )
    print(
        f"  Global  LB={_fmt(rec.global_lb_at_entry)}  UB={_fmt(rec.global_ub_at_entry)}  "
        f"open={rec.open_nodes_at_entry}  col_pool={rec.col_pool_size_at_entry}"
    )
    print(
        f"  CG      iters={rec.total_cg_iters}  "
        f"cols_gen={rec.total_cols_generated}  cols_added={rec.total_cols_added}  "
        f"final_lp={_fmt(rec.final_lp_obj)}  art_sum={rec.art_sum_at_converge:.2e}"
    )
    if rec.total_cg_iters > 0 or rec.solve_addcol_time_s > 0.0:
        print(
            f"  Timings rmp={rec.solve_rmp_lp_time_s:.3f}s  cut_sep={rec.solve_cut_separation_time_s:.3f}s  "
            f"rmp_all={rec.solve_rmp_time_s:.3f}s  prc={rec.solve_pricing_time_s:.3f}s  "
            f"addcol={rec.solve_addcol_time_s:.3f}s  (incl. outside per-iter table)"
        )
    if rec.rmp_failure_msg:
        print(f"  RMP     failure: {rec.rmp_failure_msg}")
    print(
        f"  Model   vars={rec.model_nvars}  constrs={rec.model_ncons}"
    )

    # Active branch constraints
    if rec.active_constraints:
        print(f"  Constraints ({len(rec.active_constraints)}):")
        for bc in rec.active_constraints:
            print(f"    [{bc['family']}]  {bc['sense']} {bc['rhs']}  day={bc['day']}  target={bc['target']}")

    # Active cuts + newly added cuts on this node
    if rec.active_cuts:
        print(f"  Active cuts ({len(rec.active_cuts)}):")
        for cut in rec.active_cuts:
            print(f"    [{cut.get('family')}]  {cut.get('cname')}  key={cut.get('key')}")
    if rec.cuts_added_this_node:
        print(f"  Cuts added in this node ({len(rec.cuts_added_this_node)}):")
        for cut in rec.cuts_added_this_node:
            print(f"    [+][{cut.get('family')}]  {cut.get('cname')}  key={cut.get('key')}")

    # Per-CG iteration table (compact)
    if rec.cg_iters:
        print(f"  CG iterations:  ph1=Y → cover artificials still >0 right after that RMP solve (not separate Simplex Phase I)")
        print(f"    {'iter':>4}  {'LP_obj':>12}  {'gen':>5}  {'add':>5}  {'rmp_s':>7}  {'cut_s':>7}  {'prc_s':>7}  {'ph1':>4}  {'art':>8}")
        for r in rec.cg_iters:
            ph1_mark = "Y" if r.phase1_active else "."
            lp_cell = (
                f"{r.lp_obj:>12.4f}"
                if not (isinstance(r.lp_obj, float) and math.isnan(r.lp_obj))
                else f"{'RMP_FAIL':>12}"
            )
            print(
                f"    {r.cg_iter:>4}  {lp_cell}  {r.cols_generated:>5}  {r.cols_added:>5}"
                f"  {r.rmp_time_s:>7.4f}  {r.cut_separation_time_s:>7.4f}  {r.pricing_time_s:>7.4f}  {ph1_mark:>4}  {r.art_sum:>8.2e}"
            )

    # Branching decision
    if rec.chosen_candidate:
        c = rec.chosen_candidate
        print(
            f"  Branch  rule={c['family']}  value={c['value']}  "
            f"rank_dist={c['rank_dist']}  d_ni={c['dist_nearest_int']}  "
            f"|v-0.5|={c['dist_to_half']}  day={c['day']}  target={c['target']}"
        )
        if len(rec.branch_candidates) > 1:
            sorted_c = sorted(
                rec.branch_candidates,
                key=lambda d: branch_selection_key_from_parts(str(d["family"]), float(d["value"])),
            )
            print(f"  Candidates ({len(rec.branch_candidates)} total, best-first 5):")
            for ci in sorted_c[:5]:
                marker = "→" if _branch_candidates_equal(ci, rec.chosen_candidate) else " "
                print(
                    f"    {marker} [{ci['family']}]  val={ci['value']}  "
                    f"rank={ci['rank_dist']}  d_ni={ci['dist_nearest_int']}  {ci['target']}"
                )
    elif rec.outcome == "INTEGRAL":
        print(f"  → Integer solution found (no branching needed)")


# ─────────────────────────────────────────────────────────────────────────────
# Post-run: incumbent solution — discount edges & diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _print_incumbent_discount_report(
    inst: Dict[str, Any],
    rmp: Any,
    best: Optional[Dict[str, Any]],
    theta: float,
    global_lb: float,
    global_ub: float,
) -> None:
    """List non-required edges where discount vars are active in the stored incumbent (best_solution).

    Uses ``GlobalRMPBnBTree.best_solution`` (variable snapshot at integral nodes), not the live model
    after rollback. A-RMP: ``y_e``; SimpleSPMaster: ``z_{e,k}``.
    """
    print("\n" + "═" * 72)
    print("OPTIMAL / INCUMBENT DISCOUNT & SOLUTION DETAIL")
    print("═" * 72)

    n_days = len(inst.get("periods", []) or [])
    discount_weight = float(theta) * float(n_days)
    print(f"  theta           : {theta:.6g}")
    print(f"  |T| (days)      : {n_days}")
    print(f"  discount weight : θ·|T| = {discount_weight:.6g}  (objective coef is -weight·c·var)")
    print(f"  Global LB / UB  : {_fmt(global_lb)} / {_fmt(global_ub)}")
    if math.isfinite(global_lb) and math.isfinite(global_ub) and abs(global_ub) > 1e-12:
        gap = max(0.0, (global_ub - global_lb) / abs(global_ub) * 100.0)
        print(f"  Gap (UB-LB)/|UB|: {gap:.4f}%")

    if not best:
        print("  Incumbent       : (none — no integer solution was stored)")
        return

    obj = best.get("objective")
    nid = best.get("node_id")
    src = best.get("source", "")
    vars_dict: Dict[str, Any] = best.get("variables") or {}
    if not isinstance(vars_dict, dict):
        vars_dict = {}
    n_pos = sum(1 for _k, vv in vars_dict.items() if abs(float(vv)) > 1e-6)
    print(f"  Incumbent source : {src or '(unspecified)'}")
    print(f"  Incumbent node_id: {nid if nid is not None else 'n/a (seed / no B&B snapshot)'}")
    print(f"  Incumbent obj    : {_fmt(obj)}")
    print(f"  Nonzero master vars (snapshot): {n_pos}")

    if float(theta) <= 0.0:
        print("  Discount edges   : θ=0 — no discount term in objective.")
        return

    if not vars_dict or n_pos == 0:
        ar = best.get("active_routes")
        if isinstance(ar, list) and ar:
            print(
                "  Note: incumbent has active_routes but no RMP variable snapshot — "
                "discount y/z listing requires an integral node capture (B&P) or root MIP seed with variables."
            )
        elif src in {"alns_initial", ""} and not vars_dict:
            print(
                "  Note: upper bound may come from ALNS objective only; run until an integral RMP node "
                "is found to list y/z from the master."
            )

    eps = 1e-6
    tc = getattr(rmp, "travel_cost", None) or inst.get("travel_cost") or {}

    rows: List[Tuple[str, Any, Optional[int], float, float, float]] = []

    for e, nm in getattr(rmp, "agg_y_var_name", getattr(rmp, "y_var_name", {})).items():
        if nm not in vars_dict:
            continue
        v = float(vars_dict[nm])
        if abs(v) <= eps:
            continue
        c_disc = float(discount_objective_cost_per_edge(inst, e, tc))
        savings = discount_weight * c_disc * v
        rows.append(("y", e, None, v, c_disc, savings))

    fb = getattr(rmp, "_fallback_rmp", None)
    dz_map = getattr(fb, "discount_z_var_name", {}) if fb is not None else {}
    for (e, k), nm in dz_map.items():
        if nm not in vars_dict:
            continue
        v = float(vars_dict[nm])
        if abs(v) <= eps:
            continue
        c_disc = float(discount_objective_cost_per_edge(inst, e, tc))
        savings = discount_weight * c_disc * v
        rows.append(("z", e, int(k), v, c_disc, savings))

    if not rows:
        print("  Discount-active edges: (none — no y/z above ε in incumbent snapshot)")
        print("  Hint: A-RMP uses y_e; SimpleSPMaster uses z_{e,k}. Empty can mean no incumbent")
        print("        or integrality snapshot omitted fractional discount vars.")
        return

    rows.sort(key=lambda r: (-r[5], str(r[1]), r[2] if r[2] is not None else -1))
    total_sav = sum(r[5] for r in rows)
    print(f"  Estimated total discount cost reduction (Σ θ·|T|·c·var): {total_sav:.6g}")
    print(f"  Active discount vars ({len(rows)}):")
    print(f"    {'type':>4}  {'edge':^14}  {'k':>4}  {'var':>10}  {'c_e':>12}  {'θ|T|c·var':>14}")
    for kind, e, k, v, c_disc, sav in rows:
        ek = f"({e[0]},{e[1]})"
        ks = "" if k is None else str(k)
        print(f"    {kind:>4}  {ek:^14}  {ks:>4}  {v:>10.6g}  {c_disc:>12.6g}  {sav:>14.6g}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _prof_sum(prof: Dict[str, Any], keys: Sequence[str]) -> float:
    return sum(float(prof.get(k, 0.0)) for k in keys)


def _print_prc_s_breakdown(prof: Dict[str, Any], pr_t: float) -> None:
    """
    Roll up pricing_time_s from pricing_prof_* into a small set of buckets:
    data prep, C++ binding prep, native .so, post-process, Python pricers, remainder.
    """
    denom = pr_t if pr_t > 1e-12 else 1.0
    data_node = _prof_sum(
        prof,
        (
            "pricing_prof_prep_s",
            "pricing_prof_dual_cut_s",
            "pricing_prof_day_graph_s",
            "pricing_prof_yao_apsp_s",
        ),
    )
    cpp_bind = _prof_sum(
        prof,
        (
            "pricing_prof_cpp_inproc_ctx_s",
            "pricing_prof_cpp_inproc_cand_svc_s",
            "pricing_prof_cpp_inproc_ctypes_s",
        ),
    )
    cpp_so = float(prof.get("pricing_prof_cpp_inproc_native_s", 0.0))
    post = _prof_sum(
        prof,
        ("pricing_prof_cpp_inproc_decode_s", "pricing_prof_cpp_post_s"),
    )
    py_algo = _prof_sum(prof, ("pricing_prof_py_dp_s", "pricing_prof_python_label_s"))
    accounted = data_node + cpp_bind + cpp_so + post + py_algo
    other = max(0.0, pr_t - accounted)
    inproc_sum = _prof_sum(
        prof,
        (
            "pricing_prof_cpp_inproc_ctx_s",
            "pricing_prof_cpp_inproc_cand_svc_s",
            "pricing_prof_cpp_inproc_ctypes_s",
            "pricing_prof_cpp_inproc_native_s",
            "pricing_prof_cpp_inproc_decode_s",
        ),
    )
    core_try = float(prof.get("pricing_prof_cpp_core_s", 0.0))
    gap_try = max(0.0, core_try - inproc_sum)

    print(f"\n  prc_s (= pricing_time_s) breakdown (rollup, % of pricing_time_s):")
    rows: List[Tuple[str, float]] = [
        ("데이터 정리 (prep + dual/cut + day_graph + yao)", data_node),
        ("C++ 바인딩 준비 (ctx + 후보·svc + ctypes·버퍼)", cpp_bind),
        ("C++ .so (cpp_price_dp / cpp_price_ng)", cpp_so),
        ("후처리 (경로 복원 + RC/신규 컬럼 필터)", post),
        ("Python 가격 (dp + labeling)", py_algo),
        ("기타·미분류 (stabilizer·import 등)", other),
    ]
    for label, v in rows:
        print(f"    {label:<46} {v:8.4f}s  ({100.0 * v / denom:5.1f}%)")
    if core_try > 1e-8 or inproc_sum > 1e-8:
        print(
            f"    (참고) node cpp try 전체 pricing_prof_cpp_core_s={core_try:.4f}s  "
            f"inproc 합={inproc_sum:.4f}s  Δ≈래퍼·import={gap_try:.4f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="B&P Inspection Mode: per-node 상세 분석")
    parser.add_argument("--instance", type=str, default="data/existing/yao/yao-westjordan-S.dat",
                        help=".dat 인스턴스 경로")
    parser.add_argument(
        "--full-instance",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=전체 원본 인스턴스(inspect 기본), 0=서브샘플(compare_existing 스타일).",
    )
    parser.add_argument("--required-limit", type=int, default=10)
    parser.add_argument("--node-limit", type=int, default=65)
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument(
        "--schedule-mode",
        type=str,
        default="regular",
        choices=["regular", "all_days", "all_edges_daily"],
    )
    parser.add_argument("--vehicles", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=0,
                        help="검사할 최대 B&B 노드 수 (0=제한없음)")
    parser.add_argument("--max-cg-iter", type=int, default=10000,
                        help="노드당 최대 CG iteration 수")
    parser.add_argument(
        "--pricing-method",
        type=str,
        choices=["labeling", "dp", "cpp_dp", "cpp_dp_lex", "cpp_ng"],
        default="labeling",
        help="Pricing backend (cpp_dp_lex: vehicle lex dual 반영, cpp_ng: Yao-style ng-route relaxation).",
    )
    parser.add_argument(
        "--pricing-ng-size",
        type=int,
        default=10,
        help="cpp_ng 에서 사용할 ng-route neighborhood size.",
    )
    parser.add_argument(
        "--cpp-ng-empty-fallback",
        type=str,
        choices=["labeling", "dp", "none"],
        default="none",
        help="cpp_ng 가 음수 컬럼을 못 찾았을 때 사용할 fallback backend. none 이면 fallback 없이 종료 판정.",
    )
    parser.add_argument(
        "--cpp-core-variant",
        type=str,
        choices=["default", "rc_load_dom"],
        default="default",
        help="C++ pricing core variant 선택. rc_load_dom 은 reduced-cost + load dominance 실험용.",
    )
    parser.add_argument(
        "--cut-pricing-mode",
        type=str,
        choices=["legacy", "bitmask", "auto"],
        default="auto",
        help="Cut dual pricing compatibility flag.",
    )
    parser.add_argument(
        "--cut-pricing-dual-tol",
        type=float,
        default=1e-15,
        help="|pi| <= tol 인 컷 듀얼은 pricing 반영에서 무시.",
    )
    parser.add_argument(
        "--use-coeff-dominance-filter",
        type=int,
        choices=[0, 1],
        default=0,
        help="동일 계수 컬럼(같은 제약계수 벡터) 중 비용이 나쁜 컬럼 추가를 차단.",
    )
    parser.add_argument(
        "--coeff-dom-obj-tol",
        type=float,
        default=1e-9,
        help="동일 계수 컬럼 지배 필터의 objective tolerance.",
    )
    parser.add_argument("--search-strategy", type=str, default="best_bound",
                        choices=["dfs", "best_bound"])
    parser.add_argument("--alns-iters", type=int, default=300)
    parser.add_argument("--out", type=str, default="newest_test.json",
                        help="JSON Lines 출력 파일 경로 (비어있으면 파일 미저장)")
    parser.add_argument(
        "--stabilization-on",
        action="store_true",
        default=False,
        help="듀얼 안정화 사용 (기본: 끔, compare_existing 과 동일).",
    )
    parser.add_argument("--stab-alpha", type=float, default=0.5)
    parser.add_argument("--eps-rc", type=float, default=1e-4)
    parser.add_argument("--phase1-col-cap", type=int, default=1000)
    parser.add_argument(
        "--discount-theta",
        type=float,
        default=0,
        help="비필수 링크 일관성 할인 θ (compare_existing --discount-theta 와 동일 의미).",
    )
    parser.add_argument(
        "--alns",
        type=int,
        choices=[0, 1],
        default=0,
        help="1=ALNS 초기 컬럼 (compare_existing 기본 0).",
    )
    parser.add_argument(
        "--yao-pricing",
        type=int,
        choices=[0, 1],
        default=0,
        help="1=Yao-style μ in SP (기본).",
    )
    parser.add_argument(
        "--quiet-nodes",
        action="store_true",
        default=False,
        help="노드별 상세 출력 생략, 요약만.",
    )
    parser.add_argument(
        "--use-aggregation",
        type=int,
        choices=[0, 1],
        default=0,
        help="1=A-RMP(기본), 0=표준 RMP(SimpleSPMaster)로 시작.",
    )
    parser.add_argument(
        "--use-vehicle-lex-symmetry",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=차량 인덱스 사전순 대칭 파괴(SimpleSPMaster): 각 일 t마다 Σ_r λ_{t,k_hi,r} − Σ_r λ_{t,k_lo,r} ≤ 0 (k_lo<k_hi). 0=끔.",
    )
    parser.add_argument("--use-capacity-cuts", type=int, choices=[0, 1], default=0)
    parser.add_argument("--cut-root-only", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--cut-separation-max-depth",
        type=int,
        default=0,
        help="With --cut-root-only 0: separate only if node depth <= this (0 = root only).",
    )
    args = parser.parse_args()
    os.environ["CPP_PRICER_VARIANT"] = str(args.cpp_core_variant)

    use_stab = bool(args.stabilization_on)

    print(f"Loading instance: {args.instance}")
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
        alns_iterations=args.alns_iters,
        use_alns_initialization=bool(int(args.alns)),
        pricing_method=str(args.pricing_method),
        pricing_ng_size=int(args.pricing_ng_size),
        cut_pricing_mode=str(args.cut_pricing_mode),
        cut_pricing_dual_tol=float(args.cut_pricing_dual_tol),
        use_coeff_dominance_filter=int(args.use_coeff_dominance_filter),
        coeff_dom_obj_tol=float(args.coeff_dom_obj_tol),
        node_search_strategy=args.search_strategy,
        eps_reduced_cost=args.eps_rc,
        use_dual_stabilization=use_stab,
        dual_stab_alpha=args.stab_alpha,
        phase1_col_cap=args.phase1_col_cap,
        discount_theta=float(args.discount_theta),
        use_aggregation=int(args.use_aggregation),
        yao_style_pricing=int(args.yao_pricing),
        use_vehicle_lex_symmetry=int(args.use_vehicle_lex_symmetry),
    )
    inst["cpp_ng_empty_fallback"] = str(args.cpp_ng_empty_fallback)
    inst["use_capacity_cuts"] = int(args.use_capacity_cuts)
    inst["cut_root_only"] = int(args.cut_root_only)
    if args.cut_separation_max_depth is not None:
        inst["cut_separation_max_depth"] = int(args.cut_separation_max_depth)

    print(
        f"Instance: {inst['instance_name']}  theta={float(args.discount_theta):.4g}  "
        f"nodes={len(inst['nodes'])}  required={len(inst['required_edges'])}  "
        f"days={len(inst['periods'])}  vehicles={len(inst['vehicles'])}  "
        f"veh_lex_sym={int(inst.get('use_vehicle_lex_symmetry', 1))}  "
        f"cpp_core={args.cpp_core_variant}"
    )
    print(f"Max B&B nodes: {args.max_nodes if args.max_nodes > 0 else 'unlimited'}")
    print()

    rmp = AggregatedMaster(inst)
    if int(args.use_aggregation) == 0:
        rmp.switch_to_rmp_mode()
    root_node = InspectBnBNode(
        node_id=0,
        depth=0,
        master_problem=rmp,
        inspector=None,  # will be set below
    )

    strategy = args.search_strategy.lower()
    selector = BestBoundSelector() if strategy == "best_bound" else DepthFirstSelector()

    config = BnBConfig(
        eps_integrality=1e-6,
        eps_reduced_cost=args.eps_rc,
        max_cg_iterations_per_node=args.max_cg_iter,
        max_nodes=args.max_nodes if args.max_nodes > 0 else 999999,
        max_time_s=None,
        verbose=False,
        use_dual_stabilization=use_stab,
        dual_stab_alpha=args.stab_alpha,
        phase1_col_cap=args.phase1_col_cap,
    )

    out_path = Path(args.out) if args.out.strip() else None
    inspector = BnPInspector(
        root_node=root_node,
        config=config,
        selector=selector,
        out_path=out_path,
        max_inspect_nodes=args.max_nodes,
        quiet_nodes=bool(args.quiet_nodes),
    )
    root_node._inspector = inspector

    t0 = time.perf_counter()
    best = inspector.solve()
    elapsed = time.perf_counter() - t0
    inspector.close()

    # ── Final summary ──────────────────────────────────────────────────────
    cg_limit_nodes = sum(1 for r in inspector.node_records if r.outcome == "CG_LIMIT")

    print("\n" + "═" * 72)
    print("INSPECTION SUMMARY")
    print("═" * 72)
    print(f"  Instance      : {inst['instance_name']}")
    print(f"  discount_theta: {float(args.discount_theta):.6g}")
    print(f"  CG_LIMIT nodes: {cg_limit_nodes}  (pricing stopped at max-cg-iter without RC convergence)")
    print(f"  Nodes processed: {inspector.nodes_processed}")
    print(f"  Nodes created  : {inspector.nodes_created}")
    print(f"  Global LB      : {_fmt(inspector.global_lower_bound)}")
    print(f"  Global UB      : {_fmt(inspector.global_upper_bound)}")
    if best:
        print(f"  Best obj       : {_fmt(best.get('objective'))}")
    print(f"  Total time     : {elapsed:.3f}s")

    prof = getattr(inspector, "profile", {}) or {}
    rmp_t = float(prof.get("rmp_time_s", 0.0))
    pr_t = float(prof.get("pricing_time_s", 0.0))
    add_t = float(prof.get("addcol_time_s", 0.0))
    print(f"\n  Wall-time profile (tree totals):")
    print(f"    rmp_time_s      : {rmp_t:8.4f}s")
    print(f"    pricing_time_s  : {pr_t:8.4f}s")
    print(f"    addcol_time_s   : {add_t:8.4f}s")
    prof_keys = [k for k in prof if isinstance(k, str) and k.startswith("pricing_prof_")]
    if prof_keys and pr_t > 1e-12:
        _print_prc_s_breakdown(prof, pr_t)
    elif prof_keys and pr_t <= 1e-12 and any(float(prof[k]) > 1e-12 for k in prof_keys):
        print("\n  prc_s breakdown: pricing_time_s is ~0 but pricing_prof_* > 0 (per-node 타이밍은 tree profile에 없을 수 있음)")
    if prof_keys:
        denom = pr_t if pr_t > 1e-12 else 1.0
        inproc_prefix = "pricing_prof_cpp_inproc_"
        detail_keys = sorted(k for k in prof_keys if not k.startswith(inproc_prefix))
        inproc_keys = sorted(k for k in prof_keys if k.startswith(inproc_prefix))
        if inproc_keys and sum(float(prof[k]) for k in inproc_keys) > 1e-10:
            print(f"\n  C++ pricer in-process (solve_day_cpp_* 내부, % of pricing_time_s):")
            for k in inproc_keys:
                v = float(prof[k])
                short = k.replace("pricing_prof_", "", 1)
                if short.endswith("_s"):
                    short = short[:-2]
                print(f"    {short:<26}: {v:8.4f}s  ({100.0 * v / denom:5.1f}%)")
        if detail_keys:
            print(f"\n  Pricing prof. detail (기타 counters, % of pricing_time_s):")
            for k in detail_keys:
                v = float(prof[k])
                short = k.replace("pricing_prof_", "", 1)
                if short.endswith("_s"):
                    short = short[:-2]
                print(f"    {short:<26}: {v:8.4f}s  ({100.0 * v / denom:5.1f}%)")

    # Outcome breakdown
    from collections import Counter
    outcome_counts = Counter(r.outcome for r in inspector.node_records)
    print(f"\n  Outcome breakdown:")
    for k, v in sorted(outcome_counts.items()):
        print(f"    {k:<30}: {v}")

    # Disaggregation stats
    disagg_counts = Counter(
        r.disagg_result for r in inspector.node_records if r.disagg_result
    )
    if disagg_counts:
        print(f"\n  Disaggregation results:")
        for k, v in disagg_counts.most_common():
            print(f"    {k:<30}: {v}")

    # Branching family breakdown
    branch_families = Counter(
        r.chosen_candidate["family"]
        for r in inspector.node_records
        if r.chosen_candidate
    )
    if branch_families:
        print(f"\n  Branching rule usage:")
        for k, v in branch_families.most_common():
            print(f"    {k:<35}: {v}")

    # Nodes with most columns added
    top_col = sorted(inspector.node_records, key=lambda r: r.total_cols_added, reverse=True)[:5]
    print(f"\n  Top-5 nodes by cols_added:")
    print(f"    {'node_id':>8}  {'depth':>5}  {'cg_iters':>8}  {'cols_added':>10}  {'outcome'}")
    for r in top_col:
        print(f"    {r.node_id:>8}  {r.depth:>5}  {r.total_cg_iters:>8}  {r.total_cols_added:>10}  {r.outcome}")

    # Deepest nodes
    top_deep = sorted(inspector.node_records, key=lambda r: r.depth, reverse=True)[:5]
    print(f"\n  Top-5 deepest nodes:")
    print(f"    {'node_id':>8}  {'depth':>5}  {'final_lp':>12}  {'outcome'}")
    for r in top_deep:
        lp_txt = str(_fmt(r.final_lp_obj))
        print(f"    {r.node_id:>8}  {r.depth:>5}  {lp_txt:>12}  {r.outcome}")

    _print_incumbent_discount_report(
        inst=inst,
        rmp=rmp,
        best=best,
        theta=float(args.discount_theta),
        global_lb=float(inspector.global_lower_bound),
        global_ub=float(inspector.global_upper_bound),
    )

    if out_path:
        print(f"\n  JSON Lines saved to: {out_path}")


if __name__ == "__main__":
    main()
