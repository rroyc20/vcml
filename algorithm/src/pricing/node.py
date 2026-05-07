from __future__ import annotations

import heapq
import json
import math
import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.pricing.cut_pricing import build_cut_pricing_state, normalize_cut_pricing_mode


class NodeStatus(str, Enum):
    """Branch-and-bound node status."""

    NEW = "new"
    PROCESSING = "processing"
    SOLVED_LP = "solved_lp"
    INTEGRAL = "integral"
    PRUNED = "pruned"
    INFEASIBLE = "infeasible"


@dataclass
class BnBConfig:
    """Global controls for tree exploration and node processing."""

    eps_integrality: float = 1e-6
    eps_reduced_cost: float = 1e-6
    max_cg_iterations_per_node: int = 200
    max_nodes: Optional[int] = None
    max_time_s: Optional[float] = None
    deadline_ts: Optional[float] = None
    verbose: bool = False

    # ── Dual Stabilization (in-out method, Wentges 1997) ─────────────────
    # LP 퇴화로 인한 CG cycling(LB 고정 + 컬럼 폭발)을 방지.
    # π_stab = α*π_center + (1-α)*π_LP 로 pricing dual을 smoothing.
    use_dual_stabilization: bool = True
    dual_stab_alpha: float = 0.5       # stabilization weight (0=pure LP, 1=pure center)
    dual_stab_alpha_decay: float = 0.9 # LB 개선 없을 때 α 감쇠 (0.9 = 10% 감소)
    dual_stab_min_alpha: float = 0.0   # α 하한 (0 = 완전 비활성화 가능)
    # Branching integration
    use_ub_zero_branching: bool = False
    # Partial pricing
    partial_pricing_ratio: float = 1.0
    # Phase-I column cap: while artificial variables are positive (infeasible Phase-I),
    # limit columns added per pricing call to this value (0 = no cap).
    # Prevents column explosion caused by inflated duals (~1e5) during Phase-I.
    phase1_col_cap: int = 3


class DualStabilizer:
    """In-out dual stabilization for Column Generation (Wentges 1997).

    알고리즘:
    - In-step : π_stab = α*π_center + (1-α)*π_LP 로 pricing 수행
                컬럼 발견 → 추가 후 in-step 유지
                컬럼 없음 → out-step으로 전환 (같은 LP, pure π_LP 재시도)
    - Out-step: pure π_LP 로 pricing 수행
                컬럼 발견 → 추가 후 in-step 복귀
                컬럼 없음 → CG 진짜 수렴 (termination)
    - LB 개선 시 π_center = π_LP 업데이트
    - LB 개선 없을 때 α를 decay하여 자동으로 비활성화에 수렴
    """

    def __init__(self, alpha: float = 0.5, alpha_decay: float = 0.9, min_alpha: float = 0.0):
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self._center: Optional[Dict[str, float]] = None   # π_center
        self._best_lb: float = float("-inf")
        self._in_phase: bool = True
        self._no_improve_count: int = 0

    # ── Public interface ─────────────────────────────────────────────────

    def blend(self, dual_values: Dict[str, Any]) -> Dict[str, Any]:
        """Return stabilized dual_values dict for pricing.

        In-step: π_stab = α*π_center + (1-α)*π_LP
        Out-step (or not initialized): π_LP 그대로 반환
        """
        if self._center is None or not self._in_phase or self.alpha <= 1e-9:
            return dual_values

        raw_pi: Dict[str, float] = dual_values.get("constr_pi_by_name", {})
        blended: Dict[str, float] = {}
        all_keys = set(raw_pi.keys()) | set(self._center.keys())
        for k in all_keys:
            v_lp     = raw_pi.get(k, 0.0)
            v_center = self._center.get(k, v_lp)
            blended[k] = self.alpha * v_center + (1.0 - self.alpha) * v_lp

        result = dict(dual_values)
        result["constr_pi_by_name"] = blended
        result["pi_cover"]      = {k: v for k, v in blended.items() if k.startswith("cover_")}
        result["sigma_vehicle"] = {k: v for k, v in blended.items() if k.startswith("veh_")}
        return result

    def update(self, dual_values: Dict[str, Any], lb: float) -> None:
        """CG iteration 후 LB 개선 여부에 따라 center와 α를 업데이트."""
        raw_pi: Dict[str, float] = dual_values.get("constr_pi_by_name", {})
        if self._center is None:
            self._center = dict(raw_pi)
            self._best_lb = lb
            return

        if lb > self._best_lb + 1e-8:
            # LB 개선 → center를 현재 LP dual로 이동 (in-step으로 복귀)
            self._best_lb = lb
            self._center = dict(raw_pi)
            self._in_phase = True
            self._no_improve_count = 0
        else:
            # LB 개선 없음 → α 감쇠
            self._no_improve_count += 1
            self.alpha = max(self.min_alpha,
                             self.alpha * self.alpha_decay ** self._no_improve_count)

    def switch_to_out_step(self) -> None:
        """In-step에서 컬럼을 못 찾았을 때 out-step으로 전환."""
        self._in_phase = False

    def switch_to_in_step(self) -> None:
        """Out-step에서 컬럼을 찾았을 때 in-step으로 복귀."""
        self._in_phase = True

    @property
    def is_in_phase(self) -> bool:
        return self._in_phase


@dataclass(frozen=True)
class BranchConstraint:
    """One branching restriction to be enforced by the RMP/pricing side."""

    # 예: "whole_route", "daily_route", "schedule", "schedule_fix", "visit_node", "visit_arc"
    family: str
    # 제약이 걸리는 대상 식별자(arc id, customer id, route pattern id 등)
    target: Any
    # 분기 방향: "<=", ">="
    sense: str
    # 우변값(일반적으로 0 또는 1)
    rhs: float
    # 주기 문제를 위해 day/driver를 선택적으로 보관
    day: Optional[int] = None
    driver: Optional[int] = None


@dataclass
class BranchCandidate:
    """A fractional object selected for branching."""

    family: str
    target: Any
    value: float
    day: Optional[int] = None
    driver: Optional[int] = None


BRANCH_MODE_ARMP = "armp"
BRANCH_MODE_RMP = "rmp"

ARMP_WHOLE_ROUTE_FAMILY = "armp_whole_route"
ARMP_DAILY_ROUTE_FAMILY = "armp_daily_route"
# A-RMP: 집계 스케줄 q_{e,p} = Σ_k s_{e,p,k} 에 대한 0/1 고정 (차량 인덱스 없음)
ARMP_SCHEDULE_FIX_FAMILY = "armp_schedule_fix"
# 이전 명칭 호환
ARMP_SCHEDULE_FAMILY = ARMP_SCHEDULE_FIX_FAMILY

RMP_WHOLE_ROUTE_FAMILY = "rmp_whole_route"
RMP_DAILY_ROUTE_FAMILY = "rmp_daily_route"
RMP_EDGE_DRIVER_ASSIGN_FAMILY = "rmp_edge_driver_assign"
RMP_VISIT_NODE_FAMILY = "rmp_visit_node"
RMP_VISIT_ARC_FAMILY = "rmp_visit_arc"
RMP_EDGE_DAY_DRIVER_SERVICE_FAMILY = "rmp_edge_day_driver_service"

WHOLE_ROUTE_BRANCH_FAMILIES = frozenset(
    {
        "whole_route",
        ARMP_WHOLE_ROUTE_FAMILY,
        RMP_WHOLE_ROUTE_FAMILY,
    }
)
DAILY_ROUTE_BRANCH_FAMILIES = frozenset(
    {
        "daily_route",
        ARMP_DAILY_ROUTE_FAMILY,
        RMP_DAILY_ROUTE_FAMILY,
    }
)
SCHEDULE_BRANCH_FAMILIES = frozenset(
    {
        "schedule",
        "schedule_fix",
        "armp_schedule",  # legacy alias → same handling as schedule_fix
        ARMP_SCHEDULE_FIX_FAMILY,
        ARMP_SCHEDULE_FAMILY,
    }
)
EDGE_DRIVER_ASSIGN_BRANCH_FAMILIES = frozenset(
    {
        "edge_driver_assign",
        RMP_EDGE_DRIVER_ASSIGN_FAMILY,
    }
)
EDGE_DAY_DRIVER_SERVICE_BRANCH_FAMILIES = frozenset(
    {
        "edge_day_driver_service",
        RMP_EDGE_DAY_DRIVER_SERVICE_FAMILY,
    }
)
VISIT_NODE_BRANCH_FAMILIES = frozenset(
    {
        "visit_node",
        RMP_VISIT_NODE_FAMILY,
    }
)
VISIT_ARC_BRANCH_FAMILIES = frozenset(
    {
        "visit_arc",
        "service_arc",
        RMP_VISIT_ARC_FAMILY,
    }
)

# Branching uses 0/1 split in build_child_constraints for these families only.
BINARY_LIKE_BRANCH_FAMILIES = frozenset(
    {
        "lambda_var",
        "schedule",
        "schedule_fix",
        "armp_schedule",  # legacy
        "successive_edges",
        "edge_driver_assign",
        "edge_day_driver_service",
        ARMP_SCHEDULE_FIX_FAMILY,
        RMP_EDGE_DRIVER_ASSIGN_FAMILY,
        RMP_EDGE_DAY_DRIVER_SERVICE_FAMILY,
    }
)


def is_whole_route_branch_family(family: Any) -> bool:
    return str(family) in WHOLE_ROUTE_BRANCH_FAMILIES


def is_daily_route_branch_family(family: Any) -> bool:
    return str(family) in DAILY_ROUTE_BRANCH_FAMILIES


def is_schedule_branch_family(family: Any) -> bool:
    return str(family) in SCHEDULE_BRANCH_FAMILIES


def is_edge_driver_assign_branch_family(family: Any) -> bool:
    return str(family) in EDGE_DRIVER_ASSIGN_BRANCH_FAMILIES


def is_edge_day_driver_service_branch_family(family: Any) -> bool:
    return str(family) in EDGE_DAY_DRIVER_SERVICE_BRANCH_FAMILIES


def is_visit_node_branch_family(family: Any) -> bool:
    return str(family) in VISIT_NODE_BRANCH_FAMILIES


def is_visit_arc_branch_family(family: Any) -> bool:
    return str(family) in VISIT_ARC_BRANCH_FAMILIES


def dist_to_nearest_integer(value: float) -> float:
    """Distance from x to the nearest integer (0 if x is almost integral)."""
    x = float(value)
    lo = math.floor(x)
    hi = math.ceil(x)
    if hi <= lo:
        return 0.0
    return min(x - lo, hi - x)


def branch_candidate_sort_key(c: BranchCandidate) -> Tuple[Any, ...]:
    """
    Full sort key for choose_branch_candidate: primary by fractional rule, then stable tie-break.

    Tie-break uses (family, target repr, day, driver) so runs are reproducible across Python versions.
    """
    fam = str(c.family)
    v = float(c.value)
    primary = branch_selection_key_from_parts(fam, v)
    tgt = c.target
    if isinstance(tgt, dict):
        tgt_s = json.dumps(tgt, sort_keys=True, default=str)
    else:
        tgt_s = str(tgt)
    day_k = c.day
    if isinstance(day_k, tuple):
        day_s = json.dumps(day_k, default=str)
    else:
        day_s = str(day_k)
    drv_k = c.driver
    return (primary[0], primary[1], fam, tgt_s, day_s, str(drv_k))


def branch_selection_key_from_parts(family: str, value: float) -> Tuple[float, float]:
    """
    Sort key (ascending = preferred first).
    Binary-like: prefer LP value closest to 0.5.
    Integer-split (whole_route, daily_route, …): prefer largest dist to nearest integer.
    """
    v = float(value)
    if str(family) in BINARY_LIKE_BRANCH_FAMILIES:
        return (abs(v - 0.5), -abs(v - round(v)))
    dni = dist_to_nearest_integer(v)
    return (-dni, abs(v - round(v)))


@dataclass
class PricingResult:
    """Result returned by pricing subproblem(s)."""

    # reduced cost < -eps 인 신규 컬럼(경로)
    new_columns: Sequence[Any] = field(default_factory=list)
    # 필요 시 로그/디버깅용 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeSolveResult:
    """Node-level summary used by the tree manager."""

    node_id: int
    status: NodeStatus
    lower_bound: float
    is_integral: bool
    best_integer_obj: Optional[float] = None


@dataclass
class _Label:
    """Internal label for RCSPP pricing."""

    node: Any
    load: float
    reduced_cost: float
    serviced: frozenset
    path_nodes: Tuple[Any, ...]
    path_arcs: Tuple[Any, ...]


class BnBNode:
    """
    One node = one RMP relaxation + node-specific branching constraints.

    Note:
    - 실제 최적화/프라이싱 구현은 다른 모듈에 위임.
    - 본 클래스는 흐름 제어와 상태 갱신 책임만 가짐.
    """

    def __init__(
        self,
        node_id: int,
        depth: int,
        master_problem: Any,
        routes: Optional[List[Any]] = None,
        constraints: Optional[Sequence[BranchConstraint]] = None,
        parent_id: Optional[int] = None,
    ) -> None:
        self.node_id = node_id
        self.depth = depth
        self.parent_id = parent_id

        self.rmp = master_problem
        self.routes: List[Any] = list(routes or [])
        self.constraints: List[BranchConstraint] = list(constraints or [])

        self.status: NodeStatus = NodeStatus.NEW
        self.is_solved: bool = False
        self.is_integral: bool = False

        self.lower_bound: float = float("inf")
        self.upper_bound: float = float("inf")

        self.lp_obj_value: Optional[float] = None
        self.fractional_solution: Dict[str, Any] = {}
        self._lp_basis_cache: Optional[Dict[str, Dict[str, int]]] = None

    @staticmethod
    def _canon_arc_key(edge_like: Any) -> Any:
        if isinstance(edge_like, tuple) and len(edge_like) >= 2:
            a, b = edge_like[0], edge_like[1]
            return (a, b) if a <= b else (b, a)
        return edge_like

    def _restore_lp_basis_if_any(self) -> None:
        cache = getattr(self, "_lp_basis_cache", None)
        if not isinstance(cache, dict):
            return
        vcache = cache.get("var", {})
        ccache = cache.get("con", {})
        if not isinstance(vcache, dict) or not isinstance(ccache, dict):
            return
        model = self._get_gurobi_model()
        restored = False
        for v in model.getVars():
            b = vcache.get(v.VarName)
            if b is None:
                continue
            try:
                v.VBasis = int(b)
                restored = True
            except Exception:
                pass
        for c in model.getConstrs():
            b = ccache.get(c.ConstrName)
            if b is None:
                continue
            try:
                c.CBasis = int(b)
                restored = True
            except Exception:
                pass
        if restored:
            model.update()

    def _capture_lp_basis(self) -> None:
        model = self._get_gurobi_model()
        var_basis: Dict[str, int] = {}
        con_basis: Dict[str, int] = {}
        try:
            for v in model.getVars():
                try:
                    var_basis[v.VarName] = int(v.VBasis)
                except Exception:
                    continue
            for c in model.getConstrs():
                try:
                    con_basis[c.ConstrName] = int(c.CBasis)
                except Exception:
                    continue
        except Exception:
            return
        self._lp_basis_cache = {"var": var_basis, "con": con_basis}

    def _apply_ub_zero_branch(self, bc: BranchConstraint, data: Dict[str, Any]) -> bool:
        """
        Branching integration in bcp2 style:
        disable incompatible existing columns by fixing their lambda UB to 0.
        """
        if bc.sense != "<=" or float(bc.rhs) > 0.0:
            return False
        model = self._get_gurobi_model()
        rmp = getattr(self, "rmp", None)
        route_cols = getattr(rmp, "route_columns", None)
        name_to_idx = getattr(rmp, "lambda_var_name_to_index", None)
        by_day = getattr(rmp, "lambda_var_names_by_day", None)
        if route_cols is None or not isinstance(name_to_idx, dict):
            return False

        disabled = 0

        def disable_var_by_name(vname: Any) -> None:
            nonlocal disabled
            if not isinstance(vname, str):
                return
            var = model.getVarByName(vname)
            if var is None:
                return
            if float(var.UB) > 0.0:
                var.UB = 0.0
                disabled += 1

        def disable_lambda_set(vnames: Iterable[Any]) -> None:
            for vn in vnames:
                disable_var_by_name(vn)

        family = str(getattr(bc, "family", ""))

        if family in {"lambda_var", "successive_edges"} or is_schedule_branch_family(family):
            target = bc.target if not isinstance(bc.target, dict) else bc.target.get("var_name", bc.target)
            var = self._resolve_model_var(target)
            if var is not None and hasattr(var, "UB"):
                var.UB = min(float(var.UB), 0.0)
                disabled += 1
            if disabled > 0:
                model.update()
            return disabled > 0

        if is_whole_route_branch_family(family):
            disable_lambda_set(name_to_idx.keys())
            if disabled > 0:
                model.update()
            return disabled > 0

        if is_daily_route_branch_family(family):
            day = bc.day
            if isinstance(bc.target, dict):
                day = bc.target.get("day", day)
            if isinstance(by_day, dict):
                disable_lambda_set(by_day.get(day, []))
            if disabled > 0:
                model.update()
            return disabled > 0

        if is_edge_day_driver_service_branch_family(family):
            key = bc.target.get("edge_day_driver_key") if isinstance(bc.target, dict) else bc.target
            if not (isinstance(key, tuple) and len(key) >= 3):
                return False
            expr_map = data.get("edge_day_driver_service_expr", {}).get(key, {})
            if isinstance(expr_map, dict):
                disable_lambda_set(expr_map.keys())
            if disabled > 0:
                model.update()
            return disabled > 0

        if is_edge_driver_assign_branch_family(family):
            key = bc.target.get("edge_driver_key") if isinstance(bc.target, dict) else bc.target
            if not (isinstance(key, tuple) and len(key) >= 2):
                return False
            expr_map = data.get("edge_driver_assign_expr", {}).get(key, {})
            if isinstance(expr_map, dict):
                disable_lambda_set(expr_map.keys())
            if disabled > 0:
                model.update()
            return disabled > 0

        if is_visit_arc_branch_family(family):
            key = bc.target.get("arc_day_key") if isinstance(bc.target, dict) else bc.target
            expr_map = data.get("arc_visit_expr", {}).get(key, {})
            if isinstance(expr_map, dict):
                disable_lambda_set(expr_map.keys())
            if disabled > 0:
                model.update()
            return disabled > 0

        if is_visit_node_branch_family(family):
            key = bc.target.get("node_day_key") if isinstance(bc.target, dict) else bc.target
            expr_map = data.get("node_visit_expr", {}).get(key, {})
            if isinstance(expr_map, dict):
                disable_lambda_set(expr_map.keys())
            if disabled > 0:
                model.update()
            return disabled > 0

        return False

    def _apply_schedule_assignment_branch(self, bc: BranchConstraint, data: Dict[str, Any]) -> bool:
        """
        Encode schedule-assignment branching directly through schedule-variable bounds.

        RMP (per-vehicle s_{e,p,k}): 아래를 스케줄 변수 경계로 전파.

        - edge_driver_assign(e,k) <= 0: forbid all s[e,p,k]
        - edge_driver_assign(e,k) >= 1: forbid all s[e,p,k'] for k' != k
        - edge_day_driver_service(e,t,k) <= 0: forbid s[e,p,k] with t in pattern p
        - edge_day_driver_service(e,t,k) >= 1: keep only s[e,p,k] with t in pattern p

        A-RMP 집계 스케줄 q_{e,p} = Σ_k s_{e,p,k} 에 대한 고정은 ``schedule_fix`` / ``is_schedule_branch_family``
        로 단일 변수 경계를 바꾸는 경로에서 처리한다 (본 함수는 k-인덱스 분기만 담당).
        """
        family = str(getattr(bc, "family", ""))
        if not (
            is_edge_driver_assign_branch_family(family)
            or is_edge_day_driver_service_branch_family(family)
        ):
            return False

        schedule_vars = data.get("schedule_vars", {})
        if not isinstance(schedule_vars, dict) or not schedule_vars:
            return False
        # A-RMP 집계 스케줄 q_{e,p}: 키가 (e,p) 만 있으면 차량별 전파 불가 → 제약 행으로 처리
        if not any(isinstance(sk, tuple) and len(sk) >= 3 for sk in schedule_vars):
            return False

        model = self._get_gurobi_model()
        changed = 0
        schedule_vars_by_edge = data.get("schedule_vars_by_edge", {})
        schedule_vars_by_edge_driver = data.get("schedule_vars_by_edge_driver", {})
        schedule_vars_by_edge_day_driver = data.get("schedule_vars_by_edge_day_driver", {})
        if not isinstance(schedule_vars_by_edge, dict):
            schedule_vars_by_edge = {}
        if not isinstance(schedule_vars_by_edge_driver, dict):
            schedule_vars_by_edge_driver = {}
        if not isinstance(schedule_vars_by_edge_day_driver, dict):
            schedule_vars_by_edge_day_driver = {}

        def disable_var_refs(var_refs: Iterable[Any], *, keep: Optional[set[str]] = None) -> None:
            nonlocal changed
            keep_names = keep or set()
            for vref in var_refs:
                if isinstance(vref, str) and vref in keep_names:
                    continue
                var = self._resolve_model_var(vref)
                if var is None or not hasattr(var, "UB"):
                    continue
                if float(var.UB) > 0.0:
                    var.UB = 0.0
                    changed += 1

        if is_edge_driver_assign_branch_family(family):
            key = bc.target.get("edge_driver_key") if isinstance(bc.target, dict) else bc.target
            if not (isinstance(key, tuple) and len(key) >= 2):
                return False
            req_id = self._canon_arc_key(key[0])
            driver_id = int(key[1])
            if bc.sense == "<=" and float(bc.rhs) <= 0.0:
                disable_var_refs(schedule_vars_by_edge_driver.get((req_id, driver_id), ()))
            elif bc.sense == ">=" and float(bc.rhs) >= 1.0:
                keep = {str(v) for v in schedule_vars_by_edge_driver.get((req_id, driver_id), ())}
                disable_var_refs(schedule_vars_by_edge.get(req_id, ()), keep=keep)
            else:
                return False
        else:
            key = bc.target.get("edge_day_driver_key") if isinstance(bc.target, dict) else bc.target
            if not (isinstance(key, tuple) and len(key) >= 3):
                return False
            req_id = self._canon_arc_key(key[0])
            day_id = int(key[1])
            driver_id = int(key[2])
            if bc.sense == "<=" and float(bc.rhs) <= 0.0:
                disable_var_refs(schedule_vars_by_edge_day_driver.get((req_id, day_id, driver_id), ()))
            elif bc.sense == ">=" and float(bc.rhs) >= 1.0:
                keep = {str(v) for v in schedule_vars_by_edge_day_driver.get((req_id, day_id, driver_id), ())}
                disable_var_refs(schedule_vars_by_edge.get(req_id, ()), keep=keep)
            else:
                return False

        if changed > 0:
            model.update()
        # Return True even when no bound changed; this family is intentionally handled
        # by schedule-variable propagation instead of adding a master row.
        return True

    @staticmethod
    def _filter_pricing_day_graph(
        depot: Any,
        req_service_meta: Dict[Any, Tuple[float, float]],
        req_service_arcs: Dict[Any, List[Tuple[Any, Any, Any, float, float, float]]],
        forbidden_edges: Iterable[Any],
    ) -> Dict[str, Any]:
        forbidden = {rid for rid in forbidden_edges}
        active_meta: Dict[Any, Tuple[float, float]] = {}
        active_arcs: Dict[Any, List[Tuple[Any, Any, Any, float, float, float]]] = {}
        pricing_nodes: set = {depot}

        for req_id, meta in req_service_meta.items():
            if req_id in forbidden:
                continue
            arcs = list(req_service_arcs.get(req_id, []))
            if not arcs:
                continue
            active_meta[req_id] = meta
            active_arcs[req_id] = arcs
            for u, v, _arc_id, _tc, _dem, _sc in arcs:
                pricing_nodes.add(u)
                pricing_nodes.add(v)

        active_req_ids = list(active_meta.keys())
        active_req_to_bit = {rid: (1 << i) for i, rid in enumerate(active_req_ids)}
        active_bit_to_req = list(active_req_ids)
        return {
            "req_service_meta": active_meta,
            "req_service_arcs": active_arcs,
            "req_ids": active_req_ids,
            "req_to_bit": active_req_to_bit,
            "bit_to_req": active_bit_to_req,
            "pricing_nodes_list": list(pricing_nodes),
        }

    def _get_node_pricing_day_graph(
        self,
        day_ctx: Any,
        depot: Any,
        req_service_meta: Dict[Any, Tuple[float, float]],
        req_service_arcs: Dict[Any, List[Tuple[Any, Any, Any, float, float, float]]],
        forbidden_edges: Iterable[Any],
        *,
        req_meta_sig: Optional[Tuple[Any, ...]] = None,
    ) -> Dict[str, Any]:
        """
        Node-local filtered pricing graph view.

        This keeps branch-induced task removals scoped to the current node instead of
        mutating any shared/global RMP structures.
        """
        cache = getattr(self, "_node_pricing_day_graph_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_node_pricing_day_graph_cache", cache)

        if req_meta_sig is not None:
            req_sig = req_meta_sig
        else:
            req_sig = tuple(sorted(self._canon_arc_key(rid) for rid in req_service_meta.keys()))
        forb_sig = tuple(sorted(self._canon_arc_key(rid) for rid in forbidden_edges))
        key = (day_ctx, depot, req_sig, forb_sig)
        cached = cache.get(key)
        if isinstance(cached, dict):
            return cached

        out = self._filter_pricing_day_graph(
            depot=depot,
            req_service_meta=req_service_meta,
            req_service_arcs=req_service_arcs,
            forbidden_edges=forbidden_edges,
        )
        cache[key] = out
        return out

    def _get_gurobi_model(self) -> Any:
        """
        Return a gurobipy.Model from either:
        - self.rmp itself
        - self.rmp.model (wrapper style)
        """
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise RuntimeError("gurobipy is required to solve the RMP.") from exc

        if isinstance(self.rmp, gp.Model):
            return self.rmp

        model = getattr(self.rmp, "model", None)
        if isinstance(model, gp.Model):
            return model

        raise TypeError(
            "Unsupported RMP type. Expected gurobipy.Model or object with `.model` as gurobipy.Model."
        )

    def solve_node(self, config: BnBConfig, incumbent_ub: float) -> NodeSolveResult:
        """Run column generation at this node until convergence/pruning."""
        self._active_config = config
        self.apply_branch_constraints()
        self._restore_lp_basis_if_any()
        self.solve_stats = {
            "cg_iterations": 0,
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
        }

        def _accumulate_addcol_stats() -> None:
            add_meta = getattr(self, "_last_addcol_stats", None)
            if not isinstance(add_meta, dict):
                return
            self.solve_stats["columns_attempted_add"] += int(add_meta.get("attempted", 0))
            self.solve_stats["columns_skipped_duplicate"] += int(add_meta.get("skipped_duplicate", 0))
            self.solve_stats["columns_skipped_dominated"] += int(add_meta.get("skipped_dominated", 0))

        # ── Dual Stabilizer 초기화 ─────────────────────────────────────────
        stabilizer: Optional[DualStabilizer] = None
        if bool(config.use_dual_stabilization):
            stabilizer = DualStabilizer(
                alpha=float(config.dual_stab_alpha),
                alpha_decay=float(config.dual_stab_alpha_decay),
                min_alpha=float(config.dual_stab_min_alpha),
            )

        for _ in range(config.max_cg_iterations_per_node):
            if config.deadline_ts is not None and time.perf_counter() >= float(config.deadline_ts):
                self.solve_stats["hit_time_limit"] = True
                break
            self.solve_stats["cg_iterations"] += 1
            t0 = time.perf_counter()
            lp_obj, dual_values = self.solve_rmp()
            self._capture_lp_basis()
            rmp_elapsed_total = time.perf_counter() - t0
            cut_sep_elapsed = float(
                dual_values.get(
                    "cut_separation_time_s",
                    getattr(self, "_last_cut_separation_time_s", 0.0),
                )
            )
            if not math.isfinite(cut_sep_elapsed):
                cut_sep_elapsed = 0.0
            cut_sep_elapsed = max(0.0, min(cut_sep_elapsed, rmp_elapsed_total))
            rmp_lp_elapsed = max(0.0, rmp_elapsed_total - cut_sep_elapsed)
            self.solve_stats["rmp_time_s"] += rmp_elapsed_total
            self.solve_stats["rmp_lp_time_s"] += rmp_lp_elapsed
            self.solve_stats["cut_separation_time_s"] += cut_sep_elapsed

            # ── Phase-I 감지: 인공변수가 양수이면 Phase-I ─────────────────
            # GlobalRMP에서는 모델이 공유되므로 인공변수를 루트에서 제거하면
            # 자식 노드 CG도 망가진다. 제거는 하지 않고 col cap으로만 제어한다.
            phase1_active = self._artificial_sum() > 1e-8
            if phase1_active:
                self.solve_stats["phase1_iters"] = int(self.solve_stats.get("phase1_iters", 0)) + 1
            effective_config = config

            # ── Dual Stabilizer: center 업데이트 ──────────────────────────
            if stabilizer is not None:
                stabilizer.update(dual_values, lp_obj)

            if config.deadline_ts is not None and time.perf_counter() >= float(config.deadline_ts):
                self.solve_stats["hit_time_limit"] = True
                break

            # ── In-step: stabilized duals로 pricing ───────────────────────
            pricing_duals = stabilizer.blend(dual_values) if stabilizer is not None else dual_values
            t1 = time.perf_counter()
            pricing_result = self.solve_subproblem(pricing_duals, effective_config)
            self.solve_stats["pricing_time_s"] += time.perf_counter() - t1
            meta = pricing_result.metadata if isinstance(pricing_result.metadata, dict) else {}
            self.solve_stats["labels_generated"] += int(meta.get("labels_generated", 0))
            self.solve_stats["labels_expanded"] += int(meta.get("labels_expanded", 0))
            self.solve_stats["backtrack_pruned"] += int(meta.get("backtrack_pruned", 0))
            self.solve_stats["shortcut_returns"] += int(meta.get("shortcut_returns", 0))
            self.solve_stats["completion_bound_pruned"] += int(meta.get("completion_bound_pruned", 0))
            self.solve_stats["existing_sig_filtered"] += int(meta.get("existing_sig_filtered", 0))
            self.solve_stats["coeff_dominated_filtered"] += int(meta.get("coeff_dominated_filtered", 0))
            self.solve_stats["columns_generated"] += int(meta.get("num_new_columns", len(pricing_result.new_columns)))
            for _pk, _pv in meta.items():
                if isinstance(_pk, str) and _pk.startswith("pricing_prof_"):
                    self.solve_stats[_pk] = self.solve_stats.get(_pk, 0.0) + float(_pv)

            if pricing_result.new_columns:
                t2 = time.perf_counter()
                added_count = self.add_columns_to_rmp(pricing_result.new_columns)
                self.solve_stats["addcol_time_s"] += time.perf_counter() - t2
                self.solve_stats["columns_added"] += int(added_count)
                _accumulate_addcol_stats()
                if int(added_count) > 0:
                    if stabilizer is not None:
                        stabilizer.switch_to_in_step()
                    continue
                self.solve_stats["zero_add_iterations"] += 1
                # fall through (duplicate columns)

            # ── In-step에서 컬럼 없음 → Out-step으로 전환 후 재시도 ──────
            if stabilizer is not None and stabilizer.is_in_phase:
                stabilizer.switch_to_out_step()
                # Out-step: pure LP dual로 즉시 재시도 (LP 재풀기 없이)
                t1b = time.perf_counter()
                pricing_result_out = self.solve_subproblem(dual_values, effective_config)
                self.solve_stats["pricing_time_s"] += time.perf_counter() - t1b
                meta_out = pricing_result_out.metadata if isinstance(pricing_result_out.metadata, dict) else {}
                self.solve_stats["labels_generated"] += int(meta_out.get("labels_generated", 0))
                self.solve_stats["labels_expanded"] += int(meta_out.get("labels_expanded", 0))
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
                    _accumulate_addcol_stats()
                    if int(added_out) > 0:
                        stabilizer.switch_to_in_step()
                        continue
                # Out-step에서도 컬럼 없음 → 진짜 수렴 (fall through)

            # ── CG 수렴 ────────────────────────────────────────────────────
            if lp_obj >= incumbent_ub - config.eps_integrality:
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
                    # Artificial-positive solution is not feasible for original problem.
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

                # ── A-RMP → 표준 RMP 전환 훅 ─────────────────────────────
                # 기본 정책:
                #   A-RMP 에서 정수 feasible 해를 만나면, 그 해를 종결로 쓰지 않고
                #   같은 branch 상태를 유지한 채 표준 SimpleSPMaster 로 전환한다.
                #   이후 동일 node 에서 CG 를 다시 수행해 exact RMP 기준으로 판단한다.
                #
                # 선택적으로 inst["armp_integral_switch_to_rmp"]=0 이면
                # 이전처럼 disaggregation MILP 를 시도하고, 실패 시에만 표준 RMP 로 전환한다.
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
                        else:
                            _need_switch = True
                    if _need_switch:
                        _rmp.switch_to_rmp_mode(_armp_vals)
                        setattr(self, "_agg_mode_switched", True)
                        self._branch_constraints_applied = False
                        self.apply_branch_constraints()
                        # 전환 후에는 표준 RMP dual 공간이 달라지므로 stabilizer 를 초기화한다.
                        if bool(config.use_dual_stabilization):
                            stabilizer = DualStabilizer(
                                alpha=float(config.dual_stab_alpha),
                                alpha_decay=float(config.dual_stab_alpha_decay),
                                min_alpha=float(config.dual_stab_min_alpha),
                            )
                        else:
                            stabilizer = None
                        continue
                # ── End A-RMP → 표준 RMP 전환 훅 ────────────────────────

                self.status = NodeStatus.INTEGRAL
                self.is_integral = True
                self.is_solved = True
                _ub_report = float(lp_obj)
                _d_sol = getattr(self, "_disaggregated_solution", None)
                _rmp_u = getattr(self, "rmp", None)
                if _d_sol is not None and _rmp_u is not None and hasattr(
                    _rmp_u, "reconcile_disaggregated_incumbent_objective"
                ):
                    try:
                        _av_u = {v.VarName: float(v.X) for v in self._get_gurobi_model().getVars()}
                        _rec_u = float(_rmp_u.reconcile_disaggregated_incumbent_objective(_av_u, _d_sol))
                        _ub_report = max(_ub_report, _rec_u)
                    except Exception:
                        pass
                self.upper_bound = _ub_report
                return NodeSolveResult(
                    node_id=self.node_id,
                    status=self.status,
                    lower_bound=_ub_report,
                    is_integral=True,
                    best_integer_obj=_ub_report,
                )

            self.solve_stats["stab_alpha"] = float(stabilizer.alpha) if stabilizer else 0.0
            self.status = NodeStatus.SOLVED_LP
            self.is_integral = False
            self.is_solved = True
            return NodeSolveResult(
                node_id=self.node_id,
                status=self.status,
                lower_bound=lp_obj,
                is_integral=False,
            )

        # Safety fallback when max CG iterations is reached
        if bool(self.solve_stats.get("hit_time_limit", False)):
            lp_obj = self.lp_obj_value if self.lp_obj_value is not None else float("inf")
            self.status = NodeStatus.SOLVED_LP
            self.is_integral = False
            self.is_solved = True
            return NodeSolveResult(
                node_id=self.node_id,
                status=self.status,
                lower_bound=lp_obj,
                is_integral=False,
            )

        # Safety fallback when max CG iterations is reached.
        # 마지막 CG 반복이 컬럼을 추가하고 continue했을 수 있으므로
        # LP를 한 번 더 풀어 fresh .X 값을 확보한다.
        # 이렇게 하지 않으면 pending 컬럼의 .X 접근 시
        # "Unable to retrieve attribute 'X'" Gurobi 에러가 발생한다.
        self.solve_stats["hit_cg_iteration_limit"] = True
        try:
            lp_obj, _ = self.solve_rmp()
        except RuntimeError:
            # 재풀기에서 infeasible이면 이 노드는 pruning
            self.status = NodeStatus.PRUNED
            self.is_solved = True
            self.is_integral = False
            return NodeSolveResult(
                node_id=self.node_id,
                status=self.status,
                lower_bound=float("inf"),
                is_integral=False,
            )

        # Bound pruning 재확인 (새 UB로 업데이트됐을 수 있음)
        if lp_obj >= incumbent_ub - config.eps_integrality:
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
        if not fractional and lp_obj < float("inf"):
            art_sum = self._artificial_sum()
            if art_sum > 1e-8:
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
            self.status = NodeStatus.INTEGRAL
            self.is_integral = True
            self.is_solved = True
            self.upper_bound = lp_obj
            return NodeSolveResult(
                node_id=self.node_id,
                status=self.status,
                lower_bound=lp_obj,
                is_integral=True,
                best_integer_obj=lp_obj,
            )

        self.status = NodeStatus.SOLVED_LP
        self.is_integral = False
        self.is_solved = True
        return NodeSolveResult(
            node_id=self.node_id,
            status=self.status,
            lower_bound=lp_obj,
            is_integral=False,
        )

    def apply_branch_constraints(self) -> None:
        """Apply `self.constraints` to this node RMP before optimization."""
        if getattr(self, "_branch_constraints_applied", False):
            return

        model = self._get_gurobi_model()
        need_route_lift = any(
            is_visit_node_branch_family(getattr(bc, "family", ""))
            or is_visit_arc_branch_family(getattr(bc, "family", ""))
            for bc in self.constraints
        )
        data = self._get_branching_data(include_route_lifting=need_route_lift)
        use_ub_zero = bool(getattr(self, "_active_config", None) and getattr(self._active_config, "use_ub_zero_branching", False))

        def build_expr_for_constraint(bc: BranchConstraint):
            family = str(bc.family)
            target = bc.target

            if family == "lambda_var":
                var_ref = target.get("var_name") if isinstance(target, dict) else target
                return self._resolve_model_var(var_ref)

            if is_whole_route_branch_family(family):
                lam = self._get_lambda_vars()
                if not lam:
                    return None
                return sum(lam)

            if is_daily_route_branch_family(family):
                day = bc.day
                if isinstance(target, dict):
                    day = target.get("day", day)
                by_day = self._get_lambda_vars_by_day()
                vars_for_day = by_day.get(day, [])
                if not vars_for_day:
                    return None
                return sum(vars_for_day)

            if is_schedule_branch_family(family):
                var_ref = None
                if isinstance(target, dict):
                    var_ref = target.get("var_name")
                    if var_ref is None and "schedule_key" in target:
                        sched = data.get("schedule_vars", {})
                        var_ref = sched.get(target["schedule_key"])
                else:
                    var_ref = target
                var = self._resolve_model_var(var_ref)
                return var

            if is_edge_driver_assign_branch_family(family):
                key = target.get("edge_driver_key") if isinstance(target, dict) else target
                return self._build_linear_expr_from_map(
                    data.get("edge_driver_assign_expr", {}).get(key, {})
                )

            if is_edge_day_driver_service_branch_family(family):
                key = target.get("edge_day_driver_key") if isinstance(target, dict) else target
                return self._build_linear_expr_from_map(
                    data.get("edge_day_driver_service_expr", {}).get(key, {})
                )

            if is_visit_node_branch_family(family):
                key = target.get("node_day_key") if isinstance(target, dict) else target
                return self._build_linear_expr_from_map(
                    data.get("node_visit_expr", {}).get(key, {})
                )

            if is_visit_arc_branch_family(family):
                key = target.get("arc_day_key") if isinstance(target, dict) else target
                return self._build_linear_expr_from_map(
                    data.get("arc_visit_expr", {}).get(key, {})
                )

            if family == "successive_edges":
                key = target.get("succ_key") if isinstance(target, dict) else target
                expr_obj = data.get("successive_expr", {}).get(key)
                if isinstance(expr_obj, dict):
                    return self._build_linear_expr_from_map(expr_obj)
                return self._resolve_model_var(expr_obj)

            return None

        for idx, bc in enumerate(self.constraints):
            cname = f"branch_n{self.node_id}_{idx}_{bc.family}"
            if model.getConstrByName(cname) is not None:
                continue

            if self._apply_schedule_assignment_branch(bc, data):
                continue

            if use_ub_zero and self._apply_ub_zero_branch(bc, data):
                continue

            expr = build_expr_for_constraint(bc)
            if expr is None:
                # 표현식을 구성할 수 없으면 이 노드에서는 해당 분기를 스킵.
                # (데이터 인터페이스가 준비되면 자동 적용됨)
                continue

            # Single-variable branching must fix bounds directly for robust propagation.
            if (
                bc.family in {"lambda_var", "successive_edges"}
                or is_schedule_branch_family(bc.family)
            ) and hasattr(expr, "LB") and hasattr(expr, "UB"):
                rhs = float(bc.rhs)
                if bc.sense == "<=":
                    expr.UB = min(float(expr.UB), rhs)
                elif bc.sense == ">=":
                    expr.LB = max(float(expr.LB), rhs)
                else:
                    raise ValueError(f"Unsupported branch sense: {bc.sense}")
                continue

            if bc.sense == "<=":
                model.addConstr(expr <= float(bc.rhs), name=cname)
            elif bc.sense == ">=":
                model.addConstr(expr >= float(bc.rhs), name=cname)
            else:
                raise ValueError(f"Unsupported branch sense: {bc.sense}")

            # ---- Register aggregate constraints in master so new columns
            #      added during CG participate with correct coefficients. ----
            rmp = getattr(self, "rmp", None)
            register = getattr(rmp, "register_aggregate_constr", None)
            if register is not None:
                family = str(bc.family)
                day_key = bc.day  # (t, k) tuple for daily_route
                target = bc.target
                if is_whole_route_branch_family(family):
                    register(("whole_route",), cname)
                elif is_daily_route_branch_family(family):
                    # Σ_{k,r} λ^{t,k}_r: BranchCandidate.day is int t (not (t,k)).
                    # Register keys must match _add_route_var / _add_agg_route_var chgCoeff.
                    day_t: Optional[int] = None
                    if isinstance(day_key, tuple) and len(day_key) == 2:
                        day_t = int(day_key[0])
                    elif day_key is not None:
                        day_t = int(day_key)
                    elif isinstance(target, dict) and target.get("day") is not None:
                        day_t = int(target["day"])
                    if day_t is not None:
                        if bool(getattr(rmp, "a_flag", False)):
                            register(("daily_route", day_t), cname)
                        else:
                            for k in getattr(rmp, "vehicles", []) or []:
                                register(("daily_route", day_t, int(k)), cname)
                elif is_visit_node_branch_family(family):
                    ndk = target.get("node_day_key") if isinstance(target, dict) else None
                    if isinstance(ndk, tuple) and len(ndk) == 3:
                        register(("visit_node", ndk[0], ndk[1], ndk[2]), cname)
                elif is_visit_arc_branch_family(family):
                    adk = target.get("arc_day_key") if isinstance(target, dict) else None
                    if isinstance(adk, tuple) and len(adk) == 3:
                        # SimpleSPMaster: (canon_edge, day t, driver k)
                        register(("visit_arc", adk[0], adk[1], adk[2]), cname)
                    elif isinstance(adk, tuple) and len(adk) == 2:
                        # AggregatedMaster A-RMP: (canon_edge, day) — no per-driver λ
                        register(("visit_arc", adk[0], adk[1]), cname)

        model.update()
        self._branch_constraints_applied = True

    def solve_rmp(self) -> Tuple[float, Dict[str, Any]]:
        """
        Solve LP relaxation of current node.

        Returns:
            lp_objective, dual_values
        """
        import gurobipy as gp
        from gurobipy import GRB

        model = self._get_gurobi_model()
        cut_separation_time_s = 0.0

        # Optional node-level separation loop (cutting-plane style).
        rmp_obj = getattr(self, "rmp", None)
        if rmp_obj is not None and hasattr(rmp_obj, "separate_cuts"):
            t_cut = time.perf_counter()
            try:
                rmp_obj.separate_cuts(node_depth=self.depth, node_id=self.node_id)
            except Exception:
                # Separation is an optional accelerator; fallback to plain LP solve.
                pass
            cut_separation_time_s = time.perf_counter() - t_cut
        setattr(self, "_last_cut_separation_time_s", float(cut_separation_time_s))

        # RMP는 LP relaxation이어야 하므로 연속 변수 설정을 강제.
        for var in model.getVars():
            if var.VType != GRB.CONTINUOUS:
                var.VType = GRB.CONTINUOUS
        model.update()

        self.status = NodeStatus.PROCESSING
        model.optimize()

        status = model.Status
        dual_values: Dict[str, Any] = {
            "model_status": status,
            "constr_pi_by_name": {},
            "constr_pi_list": [],
            "cut_separation_time_s": float(cut_separation_time_s),
        }

        if status == GRB.OPTIMAL:
            lp_objective = float(model.ObjVal)
            self.lp_obj_value = lp_objective
            self.lower_bound = lp_objective
            self.status = NodeStatus.SOLVED_LP

            for constr in model.getConstrs():
                cname = constr.ConstrName
                cpi = float(constr.Pi)
                dual_values["constr_pi_by_name"][cname] = cpi
                dual_values["constr_pi_list"].append(
                    {"name": cname, "pi": cpi}
                )

            # Optional separation by common prefixes used in RMP implementations.
            dual_values["pi_cover"] = {
                name: val
                for name, val in dual_values["constr_pi_by_name"].items()
                if name.startswith("cover_")
            }
            dual_values["sigma_vehicle"] = {
                name: val
                for name, val in dual_values["constr_pi_by_name"].items()
                if name.startswith("veh_")
            }
            dual_values["obj_value"] = lp_objective
            return lp_objective, dual_values

        if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
            self.status = NodeStatus.INFEASIBLE
            self.lp_obj_value = None
            self.lower_bound = float("inf")
            raise RuntimeError(
                f"RMP solve failed at node {self.node_id}: model status {status} "
                "(infeasible/unbounded)."
            )

        if status in (GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL):
            # LP 기준에서 확정적 dual이 필요하므로 현재는 OPTIMAL만 허용.
            raise RuntimeError(
                f"RMP solve did not reach OPTIMAL at node {self.node_id} (status={status})."
            )

        raise RuntimeError(
            f"Unhandled Gurobi status {status} at node {self.node_id} during RMP solve."
        )

    def _get_pricing_data(self) -> Dict[str, Any]:
        """
        Collect day-wise pricing inputs.

        Expected format:
        {
            "days": [0, 1, ...],
            "capacity": float,
            "depot": node_id,
            "max_columns": int (optional),
            "adjacency": {
                u: [
                    {
                        "id": arc_id,           # optional
                        "to": v,                # required
                        "travel_cost": float,   # required
                        "required": bool,       # optional (default False)
                        "required_id": req_id,  # optional (defaults to arc id)
                        "demand": float,        # optional (default 0)
                        "service_cost": float,  # optional (default 0)
                    }, ...
                ]
            },
            # optional mappings for robust dual extraction
            "cover_constr_name_by_edge_day": {(req_id, day): "cover_..."},
            "vehicle_limit_constr_name_by_day": {day: "veh_..."},
        }

        Source priority:
        1) self.rmp.get_pricing_data()
        2) self.rmp.pricing_data
        """
        if hasattr(self.rmp, "get_pricing_data"):
            data = self.rmp.get_pricing_data()
        else:
            data = getattr(self.rmp, "pricing_data", None)

        if not isinstance(data, dict):
            raise RuntimeError(
                "Pricing data missing. Provide `rmp.get_pricing_data()` or `rmp.pricing_data` dict."
            )

        required_keys = ("days", "capacity", "depot", "adjacency")
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise RuntimeError(f"Pricing data missing required keys: {missing}")
        return data

    def _extract_duals_for_day(
        self,
        day: Any,
        dual_values: Dict[str, Any],
        pricing_data: Dict[str, Any],
    ) -> Tuple[Dict[Any, float], float, Dict[Any, float]]:
        if self._is_aggregated_master_active():
            return self._extract_duals_for_day_aggregated(day, dual_values, pricing_data)
        return self._extract_duals_for_day_simple_sp(day, dual_values, pricing_data)

    def _is_aggregated_master_active(self) -> bool:
        rmp = getattr(self, "rmp", None)
        return bool(rmp is not None and getattr(rmp, "a_flag", False))

    def _extract_duals_for_day_simple_sp(
        self,
        day: Any,
        dual_values: Dict[str, Any],
        pricing_data: Dict[str, Any],
    ) -> Tuple[Dict[Any, float], float, Dict[Any, float]]:
        """
        Returns:
            edge_duals: required-edge dual price for this day
            vehicle_dual: day vehicle-limit dual (subtracted once per route)
            discount_edge_duals: nonrequired-edge duals for discount-link constraints
        """
        def _split_day_context(day_ctx: Any) -> Tuple[Any, Any]:
            if isinstance(day_ctx, tuple):
                if len(day_ctx) >= 2:
                    return day_ctx[0], day_ctx[1]
                if len(day_ctx) == 1:
                    return day_ctx[0], None
            return day_ctx, None

        day_id, driver_id = _split_day_context(day)

        # preferred direct payload:
        # {(req_id, day, driver): pi}, {(req_id, day): pi}, or {req_id: pi}
        edge_duals: Dict[Any, float] = {}
        pi_payload = dual_values.get("pi_et")
        if isinstance(pi_payload, dict):
            for key, val in pi_payload.items():
                if isinstance(key, tuple) and len(key) >= 3:
                    req_id, t, k = key[0], key[1], key[2]
                    if t == day_id and (driver_id is None or k == driver_id):
                        edge_duals[req_id] = float(val)
                elif isinstance(key, tuple) and len(key) == 2:
                    req_id, t = key[0], key[1]
                    if t == day_id:
                        edge_duals[req_id] = float(val)
                else:
                    edge_duals[key] = float(val)

        by_name = dual_values.get("constr_pi_by_name", {})
        if not edge_duals and isinstance(by_name, dict):
            cover_name_map = pricing_data.get("cover_constr_name_by_edge_day", {})
            for key, cname in cover_name_map.items():
                if cname not in by_name:
                    continue
                if isinstance(key, tuple) and len(key) >= 3:
                    req_id, t, k = key[0], key[1], key[2]
                    if t == day_id and (driver_id is None or k == driver_id):
                        edge_duals[req_id] = float(by_name[cname])
                elif isinstance(key, tuple) and len(key) == 2:
                    req_id, t = key[0], key[1]
                    if t == day_id:
                        edge_duals[req_id] = float(by_name[cname])

        vehicle_dual = 0.0
        sigma_payload = dual_values.get("sigma_t")
        if isinstance(sigma_payload, dict):
            if day in sigma_payload:
                vehicle_dual = float(sigma_payload[day])
            elif (day_id, driver_id) in sigma_payload:
                vehicle_dual = float(sigma_payload[(day_id, driver_id)])
            elif day_id in sigma_payload:
                vehicle_dual = float(sigma_payload[day_id])
        elif isinstance(by_name, dict):
            veh_name_map = pricing_data.get("vehicle_limit_constr_name_by_day", {})
            cname = veh_name_map.get(day)
            if cname is None:
                cname = veh_name_map.get((day_id, driver_id))
            if cname is None:
                cname = veh_name_map.get(day_id)
            if cname in by_name:
                vehicle_dual = float(by_name[cname])
            else:
                # fallback naming conventions
                if driver_id is None:
                    fallback_names = [f"veh_{day_id}", f"veh_t{day_id}"]
                else:
                    fallback_names = [f"veh_t{day_id}_k{driver_id}", f"veh_{day_id}_{driver_id}"]
                for fallback in fallback_names:
                    if fallback in by_name:
                        vehicle_dual = float(by_name[fallback])
                        break

        discount_edge_duals: Dict[Any, float] = {}
        discount_payload = dual_values.get("discount_link_dual")
        if isinstance(discount_payload, dict):
            for key, val in discount_payload.items():
                if not (isinstance(key, tuple) and len(key) >= 3):
                    continue
                e_key, t, k = key[0], key[1], key[2]
                if t == day_id and (driver_id is None or k == driver_id):
                    discount_edge_duals[e_key] = float(val)

        if isinstance(by_name, dict):
            discount_name_map = pricing_data.get("discount_link_constr_name_by_edge_day", {})
            if isinstance(discount_name_map, dict):
                for key, cname in discount_name_map.items():
                    if cname not in by_name:
                        continue
                    if not (isinstance(key, tuple) and len(key) >= 3):
                        continue
                    e_key, t, k = key[0], key[1], key[2]
                    if t == day_id and (driver_id is None or k == driver_id):
                        discount_edge_duals[e_key] = float(by_name[cname])

        return edge_duals, vehicle_dual, discount_edge_duals

    def _extract_duals_for_day_aggregated(
        self,
        day: Any,
        dual_values: Dict[str, Any],
        pricing_data: Dict[str, Any],
    ) -> Tuple[Dict[Any, float], float, Dict[Any, float]]:
        """
        A-RMP pricing dual extraction.

        Aggregated pricing has only a day index t (no driver k), and should use:
          - agg_cover_{e,t}
          - agg_veh_t
          - agg_disc_{e,t}
        """
        day_id = day[0] if isinstance(day, tuple) and len(day) >= 1 else day
        by_name = dual_values.get("constr_pi_by_name", {}) or {}

        edge_duals: Dict[Any, float] = {}
        pi_payload = dual_values.get("pi_et")
        if isinstance(pi_payload, dict):
            for key, val in pi_payload.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    req_id, t = key[0], key[1]
                    if t == day_id:
                        edge_duals[req_id] = float(val)
                else:
                    edge_duals[key] = float(val)
        if not edge_duals and isinstance(by_name, dict):
            cover_name_map = pricing_data.get("cover_constr_name_by_edge_day", {})
            for key, cname in cover_name_map.items():
                if cname not in by_name or not (isinstance(key, tuple) and len(key) >= 2):
                    continue
                req_id, t = key[0], key[1]
                if t == day_id:
                    edge_duals[req_id] = float(by_name[cname])

        vehicle_dual = 0.0
        sigma_payload = dual_values.get("sigma_t")
        if isinstance(sigma_payload, dict):
            if day_id in sigma_payload:
                vehicle_dual = float(sigma_payload[day_id])
            elif day in sigma_payload:
                vehicle_dual = float(sigma_payload[day])
        elif isinstance(by_name, dict):
            veh_name_map = pricing_data.get("vehicle_limit_constr_name_by_day", {})
            cname = veh_name_map.get(day_id, veh_name_map.get(day))
            if cname in by_name:
                vehicle_dual = float(by_name[cname])

        discount_edge_duals: Dict[Any, float] = {}
        discount_payload = dual_values.get("discount_link_dual")
        if isinstance(discount_payload, dict):
            for key, val in discount_payload.items():
                if not (isinstance(key, tuple) and len(key) >= 2):
                    continue
                e_key, t = key[0], key[1]
                if t == day_id:
                    discount_edge_duals[e_key] = float(val)
        if isinstance(by_name, dict):
            discount_name_map = pricing_data.get("discount_link_constr_name_by_edge_day", {})
            for key, cname in discount_name_map.items():
                if cname not in by_name or not (isinstance(key, tuple) and len(key) >= 2):
                    continue
                e_key, t = key[0], key[1]
                if t == day_id:
                    discount_edge_duals[e_key] = float(by_name[cname])

        return edge_duals, vehicle_dual, discount_edge_duals

    def _branch_rules_for_day(self, day: Any) -> Tuple[set, set]:
        if self._is_aggregated_master_active():
            return self._branch_rules_for_day_aggregated(day)
        return self._branch_rules_for_day_simple_sp(day)

    def _branch_rules_for_day_aggregated(self, day: Any) -> Tuple[set, set]:
        """
        A-RMP pricing is day-based and driver-free.

        Driver-specific schedule branching such as edge_driver_assign(e,k) must not
        directly forbid required edge e in aggregated pricing, because the edge may
        still be serviceable by another driver. Those restrictions are enforced via
        schedule-variable bounds and `forbidden_required_edges_by_day`.
        """
        forbidden: set = set()
        forced: set = set()
        day_id = day[0] if isinstance(day, tuple) and len(day) >= 1 else day

        for bc in self.constraints:
            if not is_visit_arc_branch_family(bc.family):
                continue
            if bc.day is not None and bc.day not in {day, day_id}:
                continue

            req_id = None
            target = bc.target
            if isinstance(target, dict):
                if "arc_day_key" in target:
                    key = target["arc_day_key"]
                    if isinstance(key, tuple) and len(key) >= 2:
                        req_id, t = key[0], key[1]
                        if t != day_id:
                            continue
                    else:
                        req_id = key
                elif "required_id" in target:
                    req_id = target["required_id"]
                elif "req_id" in target:
                    req_id = target["req_id"]
            elif isinstance(target, tuple) and len(target) >= 2:
                req_id, t = target[0], target[1]
                if t != day_id:
                    continue
            else:
                req_id = target

            if req_id is None:
                continue
            if bc.sense == "<=" and bc.rhs <= 0:
                forbidden.add(req_id)
            elif bc.sense == ">=" and bc.rhs >= 1:
                forced.add(req_id)
        return forbidden, forced

    def _branch_rules_for_day_simple_sp(self, day: Any) -> Tuple[set, set]:
        """
        Build simple pricing-side branch filters for required-edge servicing.

        Returns:
            forbidden_required_edges, forced_required_edges
        """
        forbidden: set = set()
        forced: set = set()

        day_id = day[0] if isinstance(day, tuple) and len(day) >= 1 else day
        driver_id = day[1] if isinstance(day, tuple) and len(day) >= 2 else None

        for bc in self.constraints:
            if not (
                is_visit_arc_branch_family(bc.family)
                or is_daily_route_branch_family(bc.family)
                or is_edge_driver_assign_branch_family(bc.family)
                or is_edge_day_driver_service_branch_family(bc.family)
            ):
                continue
            if bc.day is not None:
                if bc.day != day and bc.day != day_id:
                    continue

            req_id = None
            if is_edge_driver_assign_branch_family(bc.family):
                key = bc.target.get("edge_driver_key") if isinstance(bc.target, dict) else bc.target
                if not (isinstance(key, tuple) and len(key) >= 2):
                    continue
                req_id, k = key[0], key[1]
                if driver_id is not None and int(k) != int(driver_id):
                    continue
            elif is_edge_day_driver_service_branch_family(bc.family):
                key = bc.target.get("edge_day_driver_key") if isinstance(bc.target, dict) else bc.target
                if not (isinstance(key, tuple) and len(key) >= 3):
                    continue
                req_id, t, k = key[0], key[1], key[2]
                if int(t) != int(day_id):
                    continue
                if driver_id is not None and int(k) != int(driver_id):
                    continue
            else:
                target = bc.target
                if isinstance(target, dict):
                    # only arc-focused keys can be interpreted by pricing side
                    if "arc_day_key" in target:
                        key = target["arc_day_key"]
                        if isinstance(key, tuple) and len(key) >= 3:
                            req_id, t, k = key[0], key[1], key[2]
                            if t != day_id or (driver_id is not None and k != driver_id):
                                continue
                        elif isinstance(key, tuple) and len(key) >= 2:
                            req_id, t = key[0], key[1]
                            if t != day_id:
                                continue
                        else:
                            req_id = key
                    elif "required_id" in target:
                        req_id = target["required_id"]
                    elif "req_id" in target:
                        req_id = target["req_id"]
                    else:
                        # metric-style target dict (e.g., whole_route/daily_route) is not pricing-filterable
                        continue
                elif isinstance(target, tuple) and len(target) == 2:
                    t, req_id = target
                    if bc.day is None and t != day_id:
                        continue
                elif isinstance(target, tuple) and len(target) >= 3:
                    req_id, t, k = target[0], target[1], target[2]
                    if t != day_id:
                        continue
                    if driver_id is not None and k != driver_id:
                        continue
                else:
                    req_id = target
            if req_id is None:
                continue

            if bc.sense == "<=" and bc.rhs <= 0:
                forbidden.add(req_id)
            if bc.sense == ">=" and bc.rhs >= 1:
                forced.add(req_id)

        return forbidden, forced

    @staticmethod
    def _is_dominated(label: _Label, incumbent_labels: Iterable[_Label]) -> bool:
        """
        Simple dominance test for same (node, serviced-set):
        dominated if an incumbent has <= load and <= reduced_cost.
        """
        for other in incumbent_labels:
            if other.load <= label.load and other.reduced_cost <= label.reduced_cost:
                return True
        return False

    def _route_column_from_label(self, day: Any, label: _Label) -> Dict[str, Any]:
        day_val = day
        driver_val = None
        if isinstance(day, tuple) and len(day) >= 2:
            day_val, driver_val = day[0], day[1]
        out = {
            "day": day_val,
            "path_nodes": list(label.path_nodes),
            "path_arcs": list(label.path_arcs),
            "serviced_required_edges": list(label.serviced),
            "reduced_cost": float(label.reduced_cost),
        }
        if driver_val is not None:
            out["driver"] = driver_val
        return out

    @staticmethod
    def _is_fractional(value: float, eps: float = 1e-6) -> bool:
        return abs(value - round(value)) > eps

    def _resolve_model_var(self, ref: Any) -> Optional[Any]:
        """Resolve variable reference (Var or name string) against gurobi model."""
        if ref is None:
            return None
        if hasattr(ref, "X"):
            return ref

        if isinstance(ref, str):
            rmp = getattr(self, "rmp", None)
            for getter_name in ("_get_var_cached", "_get_agg_var_cached"):
                getter = getattr(rmp, getter_name, None)
                if callable(getter):
                    var = getter(ref)
                    if var is not None:
                        return var
            model = self._get_gurobi_model()
            return model.getVarByName(ref)
        return None

    def _build_linear_expr_from_map(self, expr_map: Any) -> Optional[Any]:
        if not isinstance(expr_map, dict) or not expr_map:
            return None
        terms = []
        for vref, coeff in expr_map.items():
            var = self._resolve_model_var(vref)
            if var is not None:
                terms.append(float(coeff) * var)
        return sum(terms) if terms else None

    def _artificial_sum(self, eps: float = 1e-8) -> float:
        """Return sum of positive artificial cover variables, when available."""
        rmp = getattr(self, "rmp", None)
        art_map = getattr(rmp, "artificial_var_name_by_cover", None)
        if not isinstance(art_map, dict) or not art_map:
            return 0.0

        model = self._get_gurobi_model()
        total = 0.0
        for aname in art_map.values():
            var = model.getVarByName(str(aname))
            if var is None:
                continue
            xv = float(var.X)
            if xv > eps:
                total += xv
        return total

    def _remove_artificials(self) -> int:
        """Phase-I 완료 후 인공변수를 모델에서 제거한다.

        제거하면 이후 BnB 자식 노드에서 LP가 진짜 비가능일 때
        Gurobi가 즉시 INFEASIBLE을 반환해 불필요한 CG 반복을 막는다.
        반환값: 제거된 인공변수 개수.
        """
        rmp = getattr(self, "rmp", None)
        art_map = getattr(rmp, "artificial_var_name_by_cover", None)
        if not isinstance(art_map, dict) or not art_map:
            return 0

        model = self._get_gurobi_model()
        removed = 0
        for aname in list(art_map.values()):
            var = model.getVarByName(str(aname))
            if var is not None:
                model.remove(var)
                removed += 1
        if removed:
            model.update()
        art_map.clear()
        return removed

    def _get_branching_data(self, *, include_route_lifting: bool = True) -> Dict[str, Any]:
        """
        Optional structured data for branching expressions.
        If absent, fallback heuristics on variable names are used.

        include_route_lifting=False skips the O(#columns·|path|) scan used for visit_arc / visit_node
        (SimpleSPMaster); pass False when only schedule / aggregate-λ branching is needed.
        """
        if hasattr(self.rmp, "get_branching_data"):
            gd = self.rmp.get_branching_data
            try:
                data = gd(include_route_lifting=include_route_lifting)
            except TypeError:
                data = gd()
            if isinstance(data, dict):
                return data
        data = getattr(self.rmp, "branching_data", None)
        return data if isinstance(data, dict) else {}

    def _get_lambda_vars(self) -> List[Any]:
        """Try to retrieve lambda route variables from structured data, else by name prefix."""
        data = self._get_branching_data(include_route_lifting=False)
        lam_refs = data.get("lambda_vars")
        lam_vars: List[Any] = []

        if isinstance(lam_refs, dict):
            for values in lam_refs.values():
                for ref in values:
                    var = self._resolve_model_var(ref)
                    if var is not None:
                        lam_vars.append(var)
        elif isinstance(lam_refs, list):
            for ref in lam_refs:
                var = self._resolve_model_var(ref)
                if var is not None:
                    lam_vars.append(var)

        if lam_vars:
            return lam_vars

        model = self._get_gurobi_model()
        for var in model.getVars():
            name = var.VarName.lower()
            if (
                name.startswith("lam_")
                or name.startswith("lambda_")
                or name.startswith("agg_lam_")
                or name.startswith("alm_")
            ):
                lam_vars.append(var)
        return lam_vars

    def _get_lambda_vars_by_day(self) -> Dict[Any, List[Any]]:
        """
        Returns day -> lambda var list.
        Priority:
        1) branching_data["lambda_vars_by_day"]
        2) parse day token from lambda variable name with `_t<day>`
        """
        data = self._get_branching_data(include_route_lifting=False)
        mapping = data.get("lambda_vars_by_day")
        out: Dict[Any, List[Any]] = {}

        if isinstance(mapping, dict):
            for day, refs in mapping.items():
                vars_for_day: List[Any] = []
                for ref in refs:
                    var = self._resolve_model_var(ref)
                    if var is not None:
                        vars_for_day.append(var)
                if vars_for_day:
                    out[day] = vars_for_day
            if out:
                return out

        # fallback parsing from variable names
        for var in self._get_lambda_vars():
            name = var.VarName
            # expected token style: ..._t3_...
            day = None
            for token in name.split("_"):
                if token.startswith("t") and token[1:].isdigit():
                    day = int(token[1:])
                    break
            if day is None:
                continue
            out.setdefault(day, []).append(var)
        return out

    @staticmethod
    def _expr_value(expr: Dict[Any, float], resolve_var) -> float:
        """
        Evaluate linear expression represented as {var_ref: coeff}.
        var_ref can be gurobi.Var or variable name string.
        """
        total = 0.0
        for ref, coeff in expr.items():
            var = resolve_var(ref)
            if var is None:
                continue
            total += float(coeff) * float(var.X)
        return total

    @staticmethod
    def _adjacency_fingerprint(adjacency: Dict[Any, List[Dict[str, Any]]]) -> Tuple[int, int, int, int]:
        """Lightweight fingerprint used to validate pricing graph precompute cache."""
        node_cnt = len(adjacency)
        arc_cnt = 0
        req_cnt = 0
        travel_micro_sum = 0
        for arcs in adjacency.values():
            arc_cnt += len(arcs)
            for arc in arcs:
                if bool(arc.get("required", False)):
                    req_cnt += 1
                travel_micro_sum += int(round(float(arc.get("travel_cost", 0.0)) * 1_000_000))
        return node_cnt, arc_cnt, req_cnt, travel_micro_sum

    def _build_deadhead_base_adjacency(
        self,
        adjacency: Dict[Any, List[Dict[str, Any]]],
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Build the sparse deadhead-only base graph used for transformed pricing.

        Preference:
        1. physical sparse road graph from instance data (best for existing/Yao inputs)
        2. fallback to the provided adjacency filtered to non-required arcs
        """
        inst = getattr(getattr(self, "rmp", None), "inst", None)
        if isinstance(inst, dict):
            edges = inst.get("road_sparse_edges") or inst.get("arc_sparse_edges")
            costs = inst.get("road_sparse_travel_cost") or inst.get("arc_sparse_travel_cost")
            if isinstance(edges, (list, tuple)) and isinstance(costs, dict):
                req_set = {self._canon_arc_key(e) for e in (inst.get("required_edges") or [])}
                out: Dict[Any, List[Dict[str, Any]]] = {}
                for ij in edges:
                    if not (isinstance(ij, (list, tuple)) and len(ij) >= 2):
                        continue
                    i, j = int(ij[0]), int(ij[1])
                    e_can = self._canon_arc_key((i, j))
                    if e_can in req_set:
                        continue
                    c = float(costs.get(e_can, float("inf")))
                    if not math.isfinite(c):
                        continue
                    out.setdefault(i, []).append({"to": j, "travel_cost": c, "id": (i, j)})
                    out.setdefault(j, []).append({"to": i, "travel_cost": c, "id": (j, i)})
                if out:
                    return out

        out: Dict[Any, List[Dict[str, Any]]] = {}
        for u, arcs in adjacency.items():
            kept: List[Dict[str, Any]] = []
            for arc in arcs:
                if bool(arc.get("required", False)):
                    continue
                if "to" not in arc or "travel_cost" not in arc:
                    continue
                kept.append(
                    {
                        "to": arc["to"],
                        "travel_cost": float(arc["travel_cost"]),
                        "id": arc.get("id", (u, arc["to"])),
                    }
                )
            if kept:
                out[u] = kept
        return out

    def _get_transformed_pricing_graph_precompute(
        self,
        adjacency: Dict[Any, List[Dict[str, Any]]],
        depot: Any,
    ) -> Dict[str, Any]:
        """
        Build the sparse-to-less-sparse transformed pricing graph.

        - nodes: depot + all required-edge endpoints
        - deadhead arcs: shortcut arcs only for pure deadhead paths
        - required arcs: explicit directed service/travel arcs for every required edge
        """
        cache = self._pricing_cache_store()
        meta = self._ensure_pricing_meta_cached(adjacency, depot)
        deadhead_adj = self._build_deadhead_base_adjacency(adjacency)
        base_fp = self._adjacency_fingerprint(deadhead_adj)
        meta_fp = meta.get("fingerprint")
        cached = cache.get("transformed_graph_precompute_v1")
        if (
            isinstance(cached, dict)
            and cached.get("base_fingerprint") == base_fp
            and cached.get("meta_fingerprint") == meta_fp
            and cached.get("depot") == depot
        ):
            return cached

        pricing_nodes = list(meta["pricing_nodes_list"])
        dead_sp_cost, dead_sp_path = self._compute_apsp_from_adjacency(deadhead_adj, pricing_nodes)

        transformed_adj: Dict[Any, List[Dict[str, Any]]] = {n: [] for n in pricing_nodes}
        for src in pricing_nodes:
            for dst in pricing_nodes:
                if src == dst:
                    continue
                c = float(dead_sp_cost.get(src, {}).get(dst, float("inf")))
                if not math.isfinite(c):
                    continue
                p = tuple(dead_sp_path.get(src, {}).get(dst, ()))
                if not p:
                    continue
                transformed_adj.setdefault(src, []).append(
                    {
                        "kind": "deadhead",
                        "to": dst,
                        "travel_cost": c,
                        "path_arcs": p,
                    }
                )

        req_service_meta = meta["req_service_meta"]
        req_service_arcs = meta["req_service_arcs"]
        req_ids = meta["req_ids"]
        req_to_bit = meta["req_to_bit"]
        bit_to_req = meta["bit_to_req"]

        for req_id, svc_arcs in req_service_arcs.items():
            for u, v, arc_id, tc, dem, sc in svc_arcs:
                transformed_adj.setdefault(u, []).append(
                    {
                        "kind": "required",
                        "to": v,
                        "travel_cost": float(tc),
                        "service_cost": float(sc),
                        "demand": float(dem),
                        "required_id": req_id,
                        "id": arc_id,
                        "path_arcs": (arc_id,),
                    }
                )

        out = {
            "base_fingerprint": base_fp,
            "meta_fingerprint": meta_fp,
            "depot": depot,
            "pricing_nodes_list": pricing_nodes,
            "adjacency": transformed_adj,
            "req_service_meta": req_service_meta,
            "req_service_arcs": req_service_arcs,
            "req_ids": req_ids,
            "req_to_bit": req_to_bit,
            "bit_to_req": bit_to_req,
        }
        cache["transformed_graph_precompute_v1"] = out
        return out

    def _pricing_cache_store(self) -> Dict[str, Any]:
        """Instance-level cache when possible; otherwise node-local cache."""
        rmp = getattr(self, "rmp", None)
        inst = getattr(rmp, "inst", None)
        if isinstance(inst, dict):
            cache = inst.get("_pricing_precompute_cache")
            if not isinstance(cache, dict):
                cache = {}
                inst["_pricing_precompute_cache"] = cache
            return cache
        cache = getattr(self, "_pricing_precompute_cache_local", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_pricing_precompute_cache_local", cache)
        return cache

    def _ensure_pricing_meta_cached(
        self,
        adjacency: Dict[Any, List[Dict[str, Any]]],
        depot: Any,
    ) -> Dict[str, Any]:
        """Cache required-edge service metadata and pricing node set (no APSP)."""
        cache = self._pricing_cache_store()
        fp = self._adjacency_fingerprint(adjacency)
        cached = cache.get("pricing_meta_only_v1")
        if isinstance(cached, dict) and cached.get("fingerprint") == fp and cached.get("depot") == depot:
            return cached

        req_service_meta: Dict[Any, Tuple[float, float]] = {}
        req_service_arcs: Dict[Any, List[Tuple[Any, Any, Any, float, float, float]]] = {}
        pricing_nodes: set = {depot}

        for u, arcs in adjacency.items():
            for arc in arcs:
                if not bool(arc.get("required", False)):
                    continue
                req_id = arc.get("required_id", arc.get("id", (u, arc.get("to"))))
                if "to" not in arc or "travel_cost" not in arc:
                    continue
                v = arc["to"]
                pricing_nodes.add(u)
                pricing_nodes.add(v)
                arc_id = arc.get("id", (u, v))
                tc = float(arc["travel_cost"])
                dem = float(arc.get("demand", 0.0))
                sc = float(arc.get("service_cost", 0.0))
                req_service_arcs.setdefault(req_id, []).append((u, v, arc_id, tc, dem, sc))
                min_total = tc + sc
                if req_id not in req_service_meta:
                    req_service_meta[req_id] = (dem, min_total)
                else:
                    old_dem, old_min = req_service_meta[req_id]
                    req_service_meta[req_id] = (old_dem, min(old_min, min_total))

        req_ids = list(req_service_meta.keys())
        req_to_bit: Dict[Any, int] = {rid: (1 << i) for i, rid in enumerate(req_ids)}
        bit_to_req: List[Any] = req_ids

        meta = {
            "fingerprint": fp,
            "depot": depot,
            "req_service_meta": req_service_meta,
            "req_service_arcs": req_service_arcs,
            "req_ids": req_ids,
            "req_to_bit": req_to_bit,
            "bit_to_req": bit_to_req,
            "pricing_nodes_list": list(pricing_nodes),
        }
        cache["pricing_meta_only_v1"] = meta
        return meta

    def _compute_apsp_from_adjacency(
        self,
        adjacency: Dict[Any, List[Dict[str, Any]]],
        pricing_nodes_list: List[Any],
    ) -> Tuple[Dict[Any, Dict[Any, float]], Dict[Any, Dict[Any, Tuple[Any, ...]]]]:
        """All-pairs shortest paths among pricing_nodes_list using given adjacency (nonnegative weights)."""
        sp_cost: Dict[Any, Dict[Any, float]] = {}
        sp_path: Dict[Any, Dict[Any, Tuple[Any, ...]]] = {}
        for src in pricing_nodes_list:
            dist_sp: Dict[Any, float] = {src: 0.0}
            prev_sp: Dict[Any, Tuple[Any, Any]] = {}
            pq_sp: List[Tuple[float, Any]] = [(0.0, src)]
            remaining = set(pricing_nodes_list)
            while pq_sp:
                cur_d, u = heapq.heappop(pq_sp)
                if cur_d > dist_sp.get(u, float("inf")) + 1e-12:
                    continue
                if u in remaining:
                    remaining.remove(u)
                    if not remaining:
                        break
                for arc in adjacency.get(u, []):
                    if "to" not in arc or "travel_cost" not in arc:
                        continue
                    v = arc["to"]
                    w = float(arc["travel_cost"])
                    nd = cur_d + w
                    if nd + 1e-12 < dist_sp.get(v, float("inf")):
                        dist_sp[v] = nd
                        prev_sp[v] = (u, arc.get("id", (u, v)))
                        heapq.heappush(pq_sp, (nd, v))
            sp_cost[src] = {}
            sp_path[src] = {}
            for dst in pricing_nodes_list:
                if src == dst:
                    sp_cost[src][dst] = 0.0
                    sp_path[src][dst] = tuple()
                    continue
                if dst not in dist_sp:
                    sp_cost[src][dst] = float("inf")
                    sp_path[src][dst] = tuple()
                    continue
                rev_arcs_sp: List[Any] = []
                cur_sp = dst
                while cur_sp != src:
                    pu, parc = prev_sp[cur_sp]
                    rev_arcs_sp.append(parc)
                    cur_sp = pu
                sp_cost[src][dst] = float(dist_sp[dst])
                sp_path[src][dst] = tuple(reversed(rev_arcs_sp))
        return sp_cost, sp_path

    def _build_sparse_modified_road_adjacency(
        self,
        inst: Dict[str, Any],
        discount_edge_duals: Dict[Any, float],
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """Lower-layer road network with Yao-style modified costs c_ij - mu_ij on discount-eligible arcs."""
        edges = inst.get("road_sparse_edges") or inst.get("arc_sparse_edges")
        costs = inst.get("road_sparse_travel_cost") or inst.get("arc_sparse_travel_cost")
        if not isinstance(edges, (list, tuple)) or not isinstance(costs, dict):
            return {}
        req_set = set(inst.get("required_edges") or [])
        adj: Dict[Any, List[Dict[str, Any]]] = {}
        for ij in edges:
            if not (isinstance(ij, (list, tuple)) and len(ij) >= 2):
                continue
            i, j = int(ij[0]), int(ij[1])
            ec = self._canon_arc_key((i, j))
            base = float(costs.get(ec, float("inf")))
            if not math.isfinite(base):
                continue
            eligible = ec not in req_set
            mu = float(discount_edge_duals.get(ec, 0.0)) if eligible else 0.0
            w = base - mu
            if w < 1e-9:
                w = 1e-9
            adj.setdefault(i, []).append({"to": j, "travel_cost": w, "id": (i, j)})
            adj.setdefault(j, []).append({"to": i, "travel_cost": w, "id": (j, i)})
        return adj

    def _get_pricing_graph_precompute(
        self,
        adjacency: Dict[Any, List[Dict[str, Any]]],
        depot: Any,
    ) -> Dict[str, Any]:
        """Build/reuse day-independent pricing precompute (req metadata + APSP on VR∪{depot})."""
        cache = self._pricing_cache_store()
        fp = self._adjacency_fingerprint(adjacency)
        cached = cache.get("graph_precompute_v2")
        if isinstance(cached, dict) and cached.get("fingerprint") == fp and cached.get("depot") == depot:
            return cached

        meta = self._ensure_pricing_meta_cached(adjacency, depot)
        sp_cost, sp_path = self._compute_apsp_from_adjacency(
            adjacency, meta["pricing_nodes_list"]
        )
        out = {
            "fingerprint": fp,
            "depot": depot,
            "req_service_meta": meta["req_service_meta"],
            "req_service_arcs": meta["req_service_arcs"],
            "req_ids": meta["req_ids"],
            "req_to_bit": meta["req_to_bit"],
            "bit_to_req": meta["bit_to_req"],
            "sp_cost": sp_cost,
            "sp_path": sp_path,
        }
        cache["graph_precompute_v2"] = out
        return out

    def _get_yao_closure_apsp_cached(
        self,
        adjacency: Dict[Any, List[Dict[str, Any]]],
        depot: Any,
        meta_only: Dict[str, Any],
    ) -> Tuple[Dict[Any, Dict[Any, float]], Dict[Any, Dict[Any, Tuple[Any, ...]]]]:
        """
        APSP on full adjacency over pricing_nodes (Yao fallback / μ-not-in-SP path).
        Independent of duals → safe to reuse across column-generation iterations.
        """
        cache = self._pricing_cache_store()
        fp = meta_only.get("fingerprint")
        if fp is None:
            fp = self._adjacency_fingerprint(adjacency)
        cached = cache.get("yao_closure_apsp_v1")
        if (
            isinstance(cached, dict)
            and cached.get("fingerprint") == fp
            and cached.get("depot") == depot
        ):
            sc = cached.get("sp_cost")
            sp = cached.get("sp_path")
            if isinstance(sc, dict) and isinstance(sp, dict):
                return sc, sp

        nodes_list = meta_only["pricing_nodes_list"]
        sp_cost, sp_path = self._compute_apsp_from_adjacency(adjacency, nodes_list)
        cache["yao_closure_apsp_v1"] = {
            "fingerprint": fp,
            "depot": depot,
            "sp_cost": sp_cost,
            "sp_path": sp_path,
        }
        return sp_cost, sp_path

    def solve_subproblem(self, dual_values: Dict[str, Any], config: BnBConfig) -> PricingResult:
        """Run pricing (labeling/RCSPP) and return negative reduced-cost columns."""
        _prof_t0 = time.perf_counter()
        _prof = {
            "pricing_prof_prep_s": 0.0,
            "pricing_prof_dual_cut_s": 0.0,
            "pricing_prof_day_graph_s": 0.0,
            "pricing_prof_yao_apsp_s": 0.0,
            "pricing_prof_cpp_core_s": 0.0,
            "pricing_prof_cpp_post_s": 0.0,
            "pricing_prof_cpp_inproc_ctx_s": 0.0,
            "pricing_prof_cpp_inproc_cand_svc_s": 0.0,
            "pricing_prof_cpp_inproc_ctypes_s": 0.0,
            "pricing_prof_cpp_inproc_native_s": 0.0,
            "pricing_prof_cpp_inproc_decode_s": 0.0,
            "pricing_prof_py_dp_s": 0.0,
            "pricing_prof_python_label_s": 0.0,
        }
        pricing_data = self._get_pricing_data()
        pricing_method_raw = str(pricing_data.get("pricing_method", "labeling")).lower()
        cut_pricing_mode = normalize_cut_pricing_mode(pricing_data.get("cut_pricing_mode", "legacy"))
        cut_pricing_dual_tol = abs(float(pricing_data.get("cut_pricing_dual_tol", 1e-15)))
        use_coeff_dominance_filter = bool(pricing_data.get("use_coeff_dominance_filter", False))
        coeff_dom_obj_tol = abs(float(pricing_data.get("coeff_dom_obj_tol", 1e-9)))
        use_lex_vehicle_dual = pricing_method_raw == "cpp_dp_lex"
        pricing_method = "cpp_dp" if use_lex_vehicle_dual else pricing_method_raw
        lex_dual_adjust = None
        if use_lex_vehicle_dual:
            from src.pricing.lex_vehicle_dual import adjust_vehicle_dual_with_lex

            lex_dual_adjust = adjust_vehicle_dual_with_lex

        adjacency: Dict[Any, List[Dict[str, Any]]] = pricing_data["adjacency"]
        capacity = float(pricing_data["capacity"])
        depot = pricing_data["depot"]
        days = list(pricing_data["days"])
        priority_store = getattr(self.rmp, "_pricing_day_priority", None)
        if not isinstance(priority_store, dict):
            priority_store = {}
            setattr(self.rmp, "_pricing_day_priority", priority_store)
        for d in days:
            priority_store.setdefault(d, 1.0)
        days = sorted(days, key=lambda d: float(priority_store.get(d, 1.0)), reverse=True)
        partial_ratio = max(0.0, min(1.0, float(getattr(config, "partial_pricing_ratio", 1.0))))
        partial_day_limit = None
        if partial_ratio < 0.999 and len(days) >= 2:
            partial_day_limit = max(1, int(math.ceil(partial_ratio * len(days))))
        dom_eps = max(1e-9, float(config.eps_reduced_cost) * 0.1)
        duplicate_rc_eps = max(1e-9, float(config.eps_reduced_cost))

        requested_max_columns = int(pricing_data.get("max_columns", 0))
        inst_dict = getattr(getattr(self, "rmp", None), "inst", None)
        use_transformed_graph = (
            isinstance(inst_dict, dict)
            and bool(int(inst_dict.get("use_transformed_pricing_graph", 1)))
            and bool(inst_dict.get("road_sparse_edges") or inst_dict.get("arc_sparse_edges"))
        )
        use_yao_sp = (
            isinstance(inst_dict, dict)
            and bool(int(inst_dict.get("yao_style_pricing", 0)))
            and bool(inst_dict.get("road_sparse_edges") or inst_dict.get("arc_sparse_edges"))
            and bool(inst_dict.get("road_sparse_travel_cost") or inst_dict.get("arc_sparse_travel_cost"))
        )
        transformed_pre = self._get_transformed_pricing_graph_precompute(adjacency, depot) if use_transformed_graph else None
        sparse_discount_edges_pre: Optional[set] = None
        if use_yao_sp and isinstance(inst_dict, dict):
            sparse_discount_edges_pre = {
                self._canon_arc_key(e)
                for e in (inst_dict.get("road_sparse_edges") or inst_dict.get("arc_sparse_edges") or [])
                if isinstance(e, tuple) and len(e) >= 2
            }
        sparse_yao_edge_canon: frozenset = (
            frozenset(sparse_discount_edges_pre) if sparse_discount_edges_pre else frozenset()
        )
        closure_sp_cost: Optional[Dict[Any, Dict[Any, float]]] = None
        closure_sp_path: Optional[Dict[Any, Dict[Any, Tuple[Any, ...]]]] = None
        if use_transformed_graph:
            pre_full = None
            meta_only = None
        elif use_yao_sp:
            meta_only = self._ensure_pricing_meta_cached(adjacency, depot)
            pre_full = None
            closure_sp_cost, closure_sp_path = self._get_yao_closure_apsp_cached(
                adjacency, depot, meta_only
            )
        else:
            pre_full = self._get_pricing_graph_precompute(adjacency, depot)
            meta_only = None

        if use_transformed_graph:
            assert transformed_pre is not None
            req_service_meta = transformed_pre["req_service_meta"]
            req_service_arcs = transformed_pre["req_service_arcs"]
            req_ids = transformed_pre["req_ids"]
            req_to_bit = transformed_pre["req_to_bit"]
            bit_to_req = transformed_pre["bit_to_req"]
        elif use_yao_sp:
            assert meta_only is not None
            req_service_meta = meta_only["req_service_meta"]
            req_service_arcs = meta_only["req_service_arcs"]
            req_ids = meta_only["req_ids"]
            req_to_bit = meta_only["req_to_bit"]
            bit_to_req = meta_only["bit_to_req"]
        else:
            assert pre_full is not None
            req_service_meta = pre_full["req_service_meta"]
            req_service_arcs = pre_full["req_service_arcs"]
            req_ids = pre_full["req_ids"]
            req_to_bit = pre_full["req_to_bit"]
            bit_to_req = pre_full["bit_to_req"]

        # sp_cost / sp_path: set each day when Yao; fixed from pre_full otherwise.
        sp_cost: Dict[Any, Dict[Any, float]] = {}
        sp_path: Dict[Any, Dict[Any, Tuple[Any, ...]]] = {}
        if not use_yao_sp:
            assert pre_full is not None
            sp_cost = pre_full["sp_cost"]
            sp_path = pre_full["sp_path"]

        num_required_for_cap = max(1, len(req_ids))
        pricing_req_meta_sig = tuple(sorted(self._canon_arc_key(rid) for rid in req_ids))

        _cache_pc = self._pricing_cache_store()
        _fp_adj = self._adjacency_fingerprint(adjacency)
        _arc_tc_key = ("min_arc_travel", _fp_adj)
        arc_travel_cost = _cache_pc.get(_arc_tc_key)
        if not isinstance(arc_travel_cost, dict):
            arc_travel_cost = {}
            for uu, outs in adjacency.items():
                try:
                    u_int = int(uu)
                except (TypeError, ValueError):
                    continue
                if not isinstance(outs, list):
                    continue
                for a in outs:
                    if not isinstance(a, dict):
                        continue
                    vv = a.get("to")
                    tc = a.get("travel_cost")
                    try:
                        v_int = int(vv)
                        c_val = float(tc)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(c_val):
                        continue
                    key = (u_int, v_int)
                    prev = arc_travel_cost.get(key)
                    if prev is None or c_val < prev:
                        arc_travel_cost[key] = c_val
            _cache_pc[_arc_tc_key] = arc_travel_cost

        if requested_max_columns > 0:
            max_columns = requested_max_columns
        else:
            # Legacy auto-cap (50*|R|) can stall LB progress by clipping too aggressively
            # (e.g., always 550 columns/iter). Use a high default cap so "auto" is
            # effectively uncapped for practical instances while staying memory-safe for C++ buffers.
            auto_max_columns = int(pricing_data.get("auto_max_columns", 100000))
            max_columns = max(200, auto_max_columns)

        # ── Phase-I column cap ────────────────────────────────────────────
        # 인공변수가 양수인 Phase-I 중에는 inflation된 dual(~1e5)로 인해
        # 거의 모든 경로가 매우 음수 reduced cost를 가져 컬럼 폭발이 발생함.
        # phase1_col_cap으로 매 pricing 호출당 추가 컬럼 수를 제한한다.
        _phase1_cap = int(getattr(config, "phase1_col_cap", 0))
        if _phase1_cap > 0 and self._artificial_sum() > 1e-8:
            max_columns = min(max_columns, _phase1_cap)
        # Safety-first default: disabling route-structure pruning avoids cutting
        # potentially optimal columns when no formal proof is attached.
        forbid_immediate_backtrack = bool(pricing_data.get("forbid_immediate_backtrack", False))
        existing_sigs_raw = pricing_data.get("existing_column_signatures", [])
        existing_sigs: set = set()
        if isinstance(existing_sigs_raw, (list, tuple, set)):
            for s in existing_sigs_raw:
                if isinstance(s, tuple) and len(s) in (3, 4):
                    existing_sigs.add(s)
        existing_coeff_best_obj_raw = pricing_data.get("existing_column_coeff_best_obj", {})
        existing_coeff_best_obj: Dict[Tuple[Any, ...], float] = {}
        if isinstance(existing_coeff_best_obj_raw, dict):
            for k, v in existing_coeff_best_obj_raw.items():
                if isinstance(k, tuple):
                    try:
                        existing_coeff_best_obj[k] = float(v)
                    except (TypeError, ValueError):
                        continue

        nonrequired_closure_edge_set: set = set()
        new_columns: List[Dict[str, Any]] = []
        new_sigs: set = set()
        new_cost_sigs: set = set()
        day_stats: Dict[Any, Dict[str, int]] = {}
        total_labels_generated = 0
        total_labels_expanded = 0
        total_backtrack_pruned = 0
        total_shortcut_returns = 0
        total_completion_bound_pruned = 0
        total_existing_sig_filtered = 0
        total_coeff_dominated_filtered = 0

        rmp_ref = getattr(self, "rmp", None)
        req_edge_set_pricing: set = set()
        if rmp_ref is not None:
            rs = getattr(rmp_ref, "required_edge_set", None)
            if rs is not None:
                req_edge_set_pricing = {self._canon_arc_key(e) for e in rs}
            nrs = getattr(rmp_ref, "nonrequired_edges", None)
            if nrs is not None:
                nonrequired_closure_edge_set = {self._canon_arc_key(e) for e in nrs}
            else:
                nonrequired_closure_edge_set = set()
        else:
            nonrequired_closure_edge_set = set()
        if not nonrequired_closure_edge_set and isinstance(inst_dict, dict):
            all_edges = inst_dict.get("edges", [])
            if isinstance(all_edges, (list, tuple, set)):
                nonrequired_closure_edge_set = {
                    self._canon_arc_key(e)
                    for e in all_edges
                    if isinstance(e, tuple) and len(e) >= 2 and self._canon_arc_key(e) not in req_edge_set_pricing
                }
        nonreq_edge_tuple_cpp = tuple(sorted(nonrequired_closure_edge_set))
        _agg_disc_nm = getattr(rmp_ref, "agg_disc_link_name", None)
        _use_unique_discount_edge = isinstance(_agg_disc_nm, dict) and len(_agg_disc_nm) > 0

        def _decode_serviced(mask: int, bit_lookup: Optional[Sequence[Any]] = None) -> List[Any]:
            lookup = bit_to_req if bit_lookup is None else bit_lookup
            out: List[Any] = []
            i = 0
            m = mask
            while m:
                if m & 1:
                    out.append(lookup[i])
                m >>= 1
                i += 1
            return out

        def _column_signature(col: Dict[str, Any]) -> Tuple[Any, ...]:
            day_val = col["day"]
            driver_val = col.get("driver", None)
            served = tuple(sorted(self._canon_arc_key(e) for e in col.get("serviced_required_edges", [])))
            arcs = tuple(col.get("path_arcs", []))
            if driver_val is None:
                return (day_val, served, arcs)
            return (day_val, driver_val, served, arcs)

        def _cost_signature(col: Dict[str, Any]) -> Tuple[Any, ...]:
            day_val = col["day"]
            driver_val = col.get("driver", None)
            served = tuple(sorted(self._canon_arc_key(e) for e in col.get("serviced_required_edges", [])))
            rc_bucket = int(round(float(col.get("reduced_cost", 0.0)) / duplicate_rc_eps))
            if driver_val is None:
                return (day_val, served, rc_bucket)
            return (day_val, driver_val, served, rc_bucket)

        req_service_cost_by_edge: Dict[Any, float] = {
            self._canon_arc_key(req_id): float(meta[1])
            for req_id, meta in req_service_meta.items()
        }

        def _column_coeff_signature_and_obj(col: Dict[str, Any]) -> Tuple[Optional[Tuple[Any, ...]], float]:
            if not use_coeff_dominance_filter:
                return None, float("inf")
            day_val = col.get("day")
            driver_val = col.get("driver", None)
            served_raw = col.get("serviced_required_edges", [])
            path_arcs = col.get("path_arcs", [])
            nonrequired_used_raw = col.get("nonrequired_edges_used", [])

            served_can: List[Any] = []
            service_cost = 0.0
            for e in served_raw:
                ec = self._canon_arc_key(e)
                served_can.append(ec)
                service_cost += float(req_service_cost_by_edge.get(ec, 0.0))

            nonreq_used: set = set()
            for e in nonrequired_used_raw:
                if not (isinstance(e, tuple) and len(e) >= 2):
                    continue
                e_can = self._canon_arc_key((e[0], e[1]))
                if e_can in req_edge_set_pricing:
                    continue
                nonreq_used.add(e_can)
            travel_cost = 0.0
            for a in path_arcs:
                if not (isinstance(a, tuple) and len(a) >= 2):
                    continue
                try:
                    u = int(a[0])
                    v = int(a[1])
                except (TypeError, ValueError):
                    continue
                c = arc_travel_cost.get((u, v))
                if c is None:
                    c = arc_travel_cost.get((v, u))
                if c is None or not math.isfinite(c):
                    return None, float("inf")
                travel_cost += float(c)

            obj_val = float(travel_cost + service_cost)
            if driver_val is None:
                sig = (day_val, tuple(sorted(served_can)), tuple(sorted(nonreq_used)))
            else:
                sig = (day_val, driver_val, tuple(sorted(served_can)), tuple(sorted(nonreq_used)))
            return sig, obj_val

        def _add_column_if_new(col: Dict[str, Any]) -> bool:
            nonlocal total_coeff_dominated_filtered
            sig = _column_signature(col)
            if sig in existing_sigs or sig in new_sigs:
                return False
            coeff_sig, obj_val = _column_coeff_signature_and_obj(col)
            if coeff_sig is not None and math.isfinite(obj_val):
                best_existing = existing_coeff_best_obj.get(coeff_sig)
                if best_existing is not None and obj_val + coeff_dom_obj_tol >= float(best_existing):
                    total_coeff_dominated_filtered += 1
                    return False
            cost_sig = _cost_signature(col)
            if cost_sig in new_cost_sigs:
                return False
            new_sigs.add(sig)
            new_cost_sigs.add(cost_sig)
            new_columns.append(col)
            if coeff_sig is not None and math.isfinite(obj_val):
                prev_obj = existing_coeff_best_obj.get(coeff_sig, float("inf"))
                if obj_val < float(prev_obj):
                    existing_coeff_best_obj[coeff_sig] = float(obj_val)
            return True

        def _discount_from_used_edges(used_edges: Sequence[Any], discount_duals: Dict[Any, float]) -> float:
            total = 0.0
            if not discount_duals:
                return total
            seen: set = set()
            for edge in used_edges:
                if not (isinstance(edge, tuple) and len(edge) >= 2):
                    continue
                e_key = self._canon_arc_key((edge[0], edge[1]))
                if e_key in req_edge_set_pricing or e_key in seen:
                    continue
                seen.add(e_key)
                total += float(discount_duals.get(e_key, 0.0))
            return total

        def _path_discount_from_arcs(path_arcs: Sequence[Any], discount_duals: Dict[Any, float]) -> float:
            total = 0.0
            if not discount_duals:
                return total
            seen: set = set()
            for arc in path_arcs:
                if not (isinstance(arc, tuple) and len(arc) >= 2):
                    continue
                e_key = self._canon_arc_key((arc[0], arc[1]))
                if e_key in req_edge_set_pricing:
                    continue
                if e_key in seen:
                    continue
                seen.add(e_key)
                total += float(discount_duals.get(e_key, 0.0))
            return total

        def _column_discount_value(col: Dict[str, Any], discount_duals: Dict[Any, float]) -> float:
            used = col.get("nonrequired_edges_used", [])
            if isinstance(used, (list, tuple)) and len(used) > 0:
                return _discount_from_used_edges(used, discount_duals)
            return _path_discount_from_arcs(col.get("path_arcs", []), discount_duals)

        def _nonrequired_edges_from_transitions(transitions: Sequence[Tuple[Any, Any]]) -> List[Any]:
            used: List[Any] = []
            seen: set = set()
            for src, dst in transitions:
                if src == dst:
                    continue
                e_key = self._canon_arc_key((src, dst))
                if e_key not in nonrequired_closure_edge_set or e_key in req_edge_set_pricing or e_key in seen:
                    continue
                seen.add(e_key)
                used.append(e_key)
            return used

        def _path_discount_between(
            src: Any,
            dst: Any,
            discount_duals: Dict[Any, float],
        ) -> float:
            e_key = self._canon_arc_key((src, dst))
            if e_key in nonrequired_closure_edge_set and e_key not in req_edge_set_pricing:
                return float(discount_duals.get(e_key, 0.0))
            arcs = sp_path.get(src, {}).get(dst, ())
            return _path_discount_from_arcs(arcs, discount_duals)

        def _nonrequired_edges_from_path(path_arcs: Sequence[Any]) -> List[Any]:
            used: List[Any] = []
            seen: set = set()
            for arc in path_arcs:
                if not (isinstance(arc, tuple) and len(arc) >= 2):
                    continue
                e_key = self._canon_arc_key((arc[0], arc[1]))
                if e_key in req_edge_set_pricing or e_key not in nonrequired_closure_edge_set or e_key in seen:
                    continue
                seen.add(e_key)
                used.append(e_key)
            return used

        def _run_transformed_day_pricing(
            *,
            day_ctx: Any,
            day_val: Any,
            driver_val: Any,
            base_rc: float,
            edge_duals: Dict[Any, float],
            discount_edge_duals: Dict[Any, float],
            forbidden_edges: set,
            cut_pricing_state: Any,
            day_req_service_meta: Dict[Any, Tuple[float, float]],
            day_req_to_bit: Dict[Any, int],
            day_bit_to_req: Sequence[Any],
            max_columns_left: int,
        ) -> Dict[str, int]:
            assert transformed_pre is not None
            graph_adj = transformed_pre["adjacency"]

            best_rc: Dict[Tuple[int, Any], float] = {(0, depot): float(base_rc)}
            best_load: Dict[Tuple[int, Any], float] = {(0, depot): 0.0}
            parent: Dict[Tuple[int, Any], Tuple[Tuple[int, Any], Tuple[Any, ...]]] = {}
            pq_state: List[Tuple[float, int, int, Any]] = []
            serial = 0
            heapq.heappush(pq_state, (float(base_rc), serial, 0, depot))
            serial += 1

            states_generated = 1
            states_expanded = 0
            negative_found = 0
            existing_sig_filtered = 0

            def _push_state(
                new_mask: int,
                new_node: Any,
                new_rc: float,
                new_load: float,
                prev_key: Tuple[int, Any],
                step_path: Tuple[Any, ...],
            ) -> None:
                nonlocal serial, states_generated
                key_new = (new_mask, new_node)
                if new_rc + dom_eps < best_rc.get(key_new, float("inf")):
                    best_rc[key_new] = float(new_rc)
                    best_load[key_new] = float(new_load)
                    parent[key_new] = (prev_key, step_path)
                    heapq.heappush(pq_state, (float(new_rc), serial, new_mask, new_node))
                    serial += 1
                    states_generated += 1

            def _reconstruct_path(final_key: Tuple[int, Any]) -> List[Any]:
                rev_steps: List[Tuple[Any, ...]] = []
                cur = final_key
                while cur in parent:
                    prev_key, step_path = parent[cur]
                    rev_steps.append(step_path)
                    cur = prev_key
                out: List[Any] = []
                for seg in reversed(rev_steps):
                    out.extend(seg)
                return out

            while pq_state and negative_found < max_columns_left:
                cur_rc, _, mask, cur_node = heapq.heappop(pq_state)
                key_cur = (mask, cur_node)
                if cur_rc > best_rc.get(key_cur, float("inf")) + dom_eps:
                    continue
                states_expanded += 1
                cur_load = best_load.get(key_cur, 0.0)

                if mask != 0 and cur_node == depot:
                    if cur_rc < -float(config.eps_reduced_cost):
                        path_arcs = _reconstruct_path(key_cur)
                        path_nodes: List[Any] = [depot]
                        for a in path_arcs:
                            if isinstance(a, tuple) and len(a) >= 2:
                                path_nodes.append(a[1])
                        col = {
                            "day": day_val,
                            "path_nodes": path_nodes,
                            "path_arcs": path_arcs,
                            "nonrequired_edges_used": _nonrequired_edges_from_path(path_arcs),
                            "serviced_required_edges": _decode_serviced(mask, day_bit_to_req),
                            "reduced_cost": float(cur_rc),
                        }
                        if driver_val is not None:
                            col["driver"] = driver_val
                        if not _add_column_if_new(col):
                            existing_sig_filtered += 1
                        else:
                            negative_found += 1
                    continue

                for arc in graph_adj.get(cur_node, []):
                    kind = str(arc.get("kind", ""))
                    nxt = arc.get("to")
                    if nxt is None:
                        continue
                    step_path = tuple(arc.get("path_arcs", ()))
                    if kind == "deadhead":
                        travel = float(arc.get("travel_cost", 0.0))
                        delta_disc = _path_discount_from_arcs(step_path, discount_edge_duals)
                        _push_state(
                            mask,
                            nxt,
                            cur_rc + travel - delta_disc,
                            cur_load,
                            key_cur,
                            step_path,
                        )
                        continue

                    if kind != "required":
                        continue

                    travel = float(arc.get("travel_cost", 0.0))
                    _push_state(
                        mask,
                        nxt,
                        cur_rc + travel,
                        cur_load,
                        key_cur,
                        step_path,
                    )

                    req_id = arc.get("required_id")
                    if req_id is None or req_id in forbidden_edges:
                        continue
                    req_bit = day_req_to_bit.get(req_id, 0)
                    if req_bit == 0 or (mask & req_bit) != 0:
                        continue
                    demand = float(arc.get("demand", 0.0))
                    new_load = cur_load + demand
                    if new_load > capacity + 1e-9:
                        continue
                    service_cost = float(arc.get("service_cost", 0.0))
                    new_mask = mask | req_bit
                    d_rb = cut_pricing_state.rb_delta(mask, new_mask, day_bit_to_req)
                    _push_state(
                        new_mask,
                        nxt,
                        cur_rc + travel + service_cost - edge_duals.get(req_id, 0.0) + d_rb,
                        new_load,
                        key_cur,
                        step_path,
                    )

            return {
                "labels_generated": int(states_generated),
                "labels_expanded": int(states_expanded),
                "negative_columns_found": int(negative_found),
                "backtrack_pruned": 0,
                "shortcut_returns": 0,
                "completion_bound_pruned": 0,
                "existing_sig_filtered": int(existing_sig_filtered),
            }

        _prof["pricing_prof_prep_s"] = time.perf_counter() - _prof_t0

        for day_idx, day_ctx in enumerate(days):
            if partial_day_limit is not None and day_idx >= partial_day_limit and len(new_columns) > 0:
                break
            effective_pricing_method = pricing_method
            if isinstance(day_ctx, tuple) and len(day_ctx) >= 2:
                day, driver = day_ctx[0], day_ctx[1]
            else:
                day, driver = day_ctx, None
            _tdc = time.perf_counter()
            edge_duals, vehicle_dual, discount_edge_duals = self._extract_duals_for_day(day_ctx, dual_values, pricing_data)
            if lex_dual_adjust is not None:
                vehicle_dual = lex_dual_adjust(
                    day_ctx=day_ctx,
                    base_vehicle_dual=vehicle_dual,
                    dual_values=dual_values,
                    pricing_data=pricing_data,
                )
            cut_pricing_state = build_cut_pricing_state(
                day_ctx=day_ctx,
                dual_values=dual_values,
                pricing_data=pricing_data,
                capacity=capacity,
                req_to_bit=req_to_bit,
                max_rb_cuts=0,
                mode=cut_pricing_mode,
                dual_tol=cut_pricing_dual_tol,
            )
            route_cut_const = float(cut_pricing_state.route_constant)
            base_route_rc = -vehicle_dual + route_cut_const
            cpp_vehicle_dual = vehicle_dual - route_cut_const
            _prof["pricing_prof_dual_cut_s"] += time.perf_counter() - _tdc
            forbidden_edges, forced_edges = self._branch_rules_for_day(day_ctx)
            sched_forbidden_map = pricing_data.get("forbidden_required_edges_by_day", {})
            use_schedule_hard_filter = bool(pricing_data.get("use_schedule_hard_filter", False))
            if use_schedule_hard_filter and isinstance(sched_forbidden_map, dict):
                forbidden_edges |= set(sched_forbidden_map.get(day_ctx, sched_forbidden_map.get(day, [])))
            # NOTE:
            # - "mandatory service on day" is an aggregate master-level condition
            #   (across all routes), not a per-route condition.
            # - Applying it as "every priced route must include edge e" over-restricts
            #   the pricing space and can invalidate the lower bound.
            # Therefore pricing only uses *forbidden* filters here.
            forced_edges = set()

            _tdg = time.perf_counter()
            day_graph = self._get_node_pricing_day_graph(
                day_ctx=day_ctx,
                depot=depot,
                req_service_meta=req_service_meta,
                req_service_arcs=req_service_arcs,
                forbidden_edges=forbidden_edges,
                req_meta_sig=pricing_req_meta_sig,
            )
            _prof["pricing_prof_day_graph_s"] += time.perf_counter() - _tdg
            day_req_service_meta = day_graph["req_service_meta"]
            day_req_service_arcs = day_graph["req_service_arcs"]
            day_req_ids = day_graph["req_ids"]
            day_req_to_bit = day_graph["req_to_bit"]
            day_bit_to_req = day_graph["bit_to_req"]
            day_pricing_nodes_list = day_graph["pricing_nodes_list"]
            if not day_req_ids:
                continue

            if use_transformed_graph:
                _tt = time.perf_counter()
                transformed_stats = _run_transformed_day_pricing(
                    day_ctx=day_ctx,
                    day_val=day,
                    driver_val=driver,
                    base_rc=base_route_rc,
                    edge_duals=edge_duals,
                    discount_edge_duals=discount_edge_duals,
                    forbidden_edges=forbidden_edges,
                    cut_pricing_state=cut_pricing_state,
                    day_req_service_meta=day_req_service_meta,
                    day_req_to_bit=day_req_to_bit,
                    day_bit_to_req=day_bit_to_req,
                    max_columns_left=max(0, max_columns - len(new_columns)),
                )
                day_stats[day_ctx] = transformed_stats
                total_labels_generated += int(transformed_stats.get("labels_generated", 0))
                total_labels_expanded += int(transformed_stats.get("labels_expanded", 0))
                total_existing_sig_filtered += int(transformed_stats.get("existing_sig_filtered", 0))
                _prof["pricing_prof_py_dp_s"] += time.perf_counter() - _tt
                if len(new_columns) >= max_columns:
                    break
                continue

            # Yao-style: μ absorbed into sparse road arc costs before APSP (TRB 153 §4.2).
            # A-RMP agg_disc_link uses one −λ per *unique* non-required edge on a route; embedding
            # μ on every arc in APSP over-counts when the same canonical edge repeats in path_arcs.
            if use_yao_sp:
                _ty = time.perf_counter()
                assert meta_only is not None and isinstance(inst_dict, dict)
                assert closure_sp_cost is not None and closure_sp_path is not None
                sparse_adj = self._build_sparse_modified_road_adjacency(
                    inst_dict, discount_edge_duals
                )
                closure_discount_outside_sparse = any(
                    self._canon_arc_key(e_key) not in sparse_yao_edge_canon
                    for e_key in discount_edge_duals.keys()
                )
                if sparse_adj and not _use_unique_discount_edge and not closure_discount_outside_sparse:
                    sp_cost, sp_path = self._compute_apsp_from_adjacency(
                        sparse_adj, day_pricing_nodes_list
                    )
                    mu_in_sp = True
                else:
                    sp_cost, sp_path = closure_sp_cost, closure_sp_path
                    mu_in_sp = False
                _prof["pricing_prof_yao_apsp_s"] += time.perf_counter() - _ty
            else:
                assert pre_full is not None
                sp_cost = pre_full["sp_cost"]
                sp_path = pre_full["sp_path"]
                mu_in_sp = False

            def _disc_seg(uu: Any, vv: Any) -> float:
                if mu_in_sp:
                    return 0.0
                return _path_discount_between(uu, vv, discount_edge_duals)

            if effective_pricing_method == "cpp_ng":
                _tcc = time.perf_counter()
                try:
                    from src.pricing.cpp_pricer import solve_day_cpp_ngroute

                    _cpp_inproc_slice: Dict[str, float] = {}
                    cpp_cols = solve_day_cpp_ngroute(
                        day=day,
                        driver=driver,
                        depot=depot,
                        capacity=capacity,
                        edge_duals=edge_duals,
                        vehicle_dual=cpp_vehicle_dual,
                        req_service_meta=day_req_service_meta,
                        req_service_arcs=day_req_service_arcs,
                        sp_cost=sp_cost,
                        sp_path=sp_path,
                        max_columns=max(0, max_columns - len(new_columns)),
                        eps_reduced_cost=float(config.eps_reduced_cost),
                        forbidden_edges=forbidden_edges,
                        ng_size=int(pricing_data.get("pricing_ng_size", 8)),
                        nonrequired_edge_set=nonreq_edge_tuple_cpp,
                        prof_out=_cpp_inproc_slice,
                    )
                    for _ik, _iv in _cpp_inproc_slice.items():
                        _pk = f"pricing_prof_cpp_inproc_{_ik}_s"
                        _prof[_pk] = _prof.get(_pk, 0.0) + float(_iv)
                except Exception:
                    cpp_cols = []
                _prof["pricing_prof_cpp_core_s"] += time.perf_counter() - _tcc

                existing_sig_filtered = 0
                negative_found = 0
                _tcp = time.perf_counter()
                for col in cpp_cols:
                    if len(new_columns) >= max_columns:
                        break
                    served = list(col.get("serviced_required_edges", []))
                    if forbidden_edges and any(req in forbidden_edges for req in served):
                        continue
                    rc_base = float(col.get("reduced_cost", 0.0))
                    if mu_in_sp:
                        rc_full = rc_base
                    else:
                        rc_adj = _column_discount_value(col, discount_edge_duals)
                        rc_full = rc_base - rc_adj
                    col["reduced_cost"] = float(rc_full)
                    if float(col["reduced_cost"]) >= -float(config.eps_reduced_cost):
                        continue
                    if not _add_column_if_new(col):
                        existing_sig_filtered += 1
                        continue
                    negative_found += 1
                _prof["pricing_prof_cpp_post_s"] += time.perf_counter() - _tcp

                if negative_found > 0:
                    day_stats[day_ctx] = {
                        "labels_generated": 0,
                        "labels_expanded": 0,
                        "negative_columns_found": int(negative_found),
                        "backtrack_pruned": 0,
                        "shortcut_returns": 0,
                        "completion_bound_pruned": 0,
                        "existing_sig_filtered": int(existing_sig_filtered),
                    }
                    total_existing_sig_filtered += int(existing_sig_filtered)
                    if len(new_columns) >= max_columns:
                        break
                    continue

                effective_pricing_method = str(
                    pricing_data.get(
                        "cpp_ng_empty_fallback",
                        pricing_data.get("cpp_empty_fallback", "labeling"),
                    )
                ).lower()
                if effective_pricing_method not in {"dp", "labeling"}:
                    day_stats[day_ctx] = {
                        "labels_generated": 0,
                        "labels_expanded": 0,
                        "negative_columns_found": 0,
                        "backtrack_pruned": 0,
                        "shortcut_returns": 0,
                        "completion_bound_pruned": 0,
                        "existing_sig_filtered": int(existing_sig_filtered),
                    }
                    total_existing_sig_filtered += int(existing_sig_filtered)
                    continue

            if effective_pricing_method == "cpp_dp":
                _tcc = time.perf_counter()
                try:
                    from src.pricing.cpp_pricer import solve_day_cpp_dp

                    _cpp_inproc_slice: Dict[str, float] = {}
                    cpp_cols = solve_day_cpp_dp(
                        day=day,
                        driver=driver,
                        depot=depot,
                        capacity=capacity,
                        edge_duals=edge_duals,
                        vehicle_dual=cpp_vehicle_dual,
                        req_service_meta=day_req_service_meta,
                        req_service_arcs=day_req_service_arcs,
                        sp_cost=sp_cost,
                        sp_path=sp_path,
                        max_columns=max(0, max_columns - len(new_columns)),
                        eps_reduced_cost=float(config.eps_reduced_cost),
                        forbidden_edges=forbidden_edges,
                        max_candidate_reqs=int(pricing_data.get("cpp_candidate_limit", 60)),
                        nonrequired_edge_set=nonreq_edge_tuple_cpp,
                        prof_out=_cpp_inproc_slice,
                    )
                    for _ik, _iv in _cpp_inproc_slice.items():
                        _pk = f"pricing_prof_cpp_inproc_{_ik}_s"
                        _prof[_pk] = _prof.get(_pk, 0.0) + float(_iv)
                except Exception:
                    cpp_cols = []
                _prof["pricing_prof_cpp_core_s"] += time.perf_counter() - _tcc

                existing_sig_filtered = 0
                negative_found = 0
                _tcp = time.perf_counter()
                for col in cpp_cols:
                    if len(new_columns) >= max_columns:
                        break
                    # Apply day-wise forbidden filter for consistency with Python pricer.
                    served_raw = col.get("serviced_required_edges", [])
                    if forbidden_edges:
                        if isinstance(served_raw, set):
                            if not served_raw.isdisjoint(forbidden_edges):
                                continue
                        elif any(req in forbidden_edges for req in served_raw):
                            continue
                    # C++ RC: μ already in sp_cost when mu_in_sp; else post-hoc discount duals.
                    rc_base = float(col.get("reduced_cost", 0.0))
                    if mu_in_sp:
                        rc_full = rc_base
                    else:
                        rc_adj = _column_discount_value(col, discount_edge_duals)
                        rc_full = rc_base - rc_adj
                    col["reduced_cost"] = float(rc_full)
                    if float(col["reduced_cost"]) >= -float(config.eps_reduced_cost):
                        continue
                    if not _add_column_if_new(col):
                        existing_sig_filtered += 1
                        continue
                    negative_found += 1
                _prof["pricing_prof_cpp_post_s"] += time.perf_counter() - _tcp

                if negative_found > 0:
                    day_stats[day_ctx] = {
                        "labels_generated": 0,
                        "labels_expanded": 0,
                        "negative_columns_found": int(negative_found),
                        "backtrack_pruned": 0,
                        "shortcut_returns": 0,
                        "completion_bound_pruned": 0,
                        "existing_sig_filtered": int(existing_sig_filtered),
                    }
                    total_existing_sig_filtered += int(existing_sig_filtered)
                    if len(new_columns) >= max_columns:
                        break
                    continue

                # Safety fallback: if cpp pricer finds no column, run a full Python pricer
                # so CG termination is not decided solely by candidate-capped cpp route search.
                effective_pricing_method = str(pricing_data.get("cpp_empty_fallback", "dp")).lower()
                if effective_pricing_method not in {"dp", "labeling"}:
                    day_stats[day_ctx] = {
                        "labels_generated": 0,
                        "labels_expanded": 0,
                        "negative_columns_found": 0,
                        "backtrack_pruned": 0,
                        "shortcut_returns": 0,
                        "completion_bound_pruned": 0,
                        "existing_sig_filtered": int(existing_sig_filtered),
                    }
                    total_existing_sig_filtered += int(existing_sig_filtered)
                    continue

            if effective_pricing_method == "dp":
                _tpd = time.perf_counter()
                # sp_cost, sp_path, req_service_arcs, _decode_serviced
                # are all precomputed before the day loop.

                start_key = (0, depot)
                best_rc = {start_key: base_route_rc}
                best_load = {start_key: 0.0}
                parent = {}
                pq_state = []
                serial = 0
                heapq.heappush(pq_state, (base_route_rc, serial, 0, depot))
                serial += 1

                states_expanded = 0
                states_generated = 1
                existing_sig_filtered = 0
                negative_found = 0

                while pq_state:
                    cur_rc, _, mask, cur_node = heapq.heappop(pq_state)
                    key_cur = (mask, cur_node)
                    if cur_rc > best_rc.get(key_cur, float("inf")) + dom_eps:
                        continue
                    states_expanded += 1
                    cur_load = best_load.get(key_cur, 0.0)

                    for req_id, svc_arcs in day_req_service_arcs.items():
                        if req_id in forbidden_edges:
                            continue
                        req_bit = day_req_to_bit.get(req_id, 0)
                        if req_bit == 0 or (mask & req_bit) != 0:
                            continue
                        for svc_from, svc_to, svc_arc_id, svc_travel, demand, service_cost in svc_arcs:
                            new_load = cur_load + demand
                            if new_load > capacity + 1e-9:
                                continue
                            dead_cost = sp_cost.get(cur_node, {}).get(svc_from, float("inf"))
                            if not math.isfinite(dead_cost):
                                continue
                            new_mask = mask | req_bit
                            d_rb = cut_pricing_state.rb_delta(mask, new_mask, day_bit_to_req)
                            new_rc = (
                                cur_rc
                                + dead_cost
                                + svc_travel
                                + service_cost
                                - edge_duals.get(req_id, 0.0)
                                - _disc_seg(cur_node, svc_from)
                                + d_rb
                            )
                            key_new = (new_mask, svc_to)
                            if new_rc + dom_eps < best_rc.get(key_new, float("inf")):
                                best_rc[key_new] = new_rc
                                best_load[key_new] = new_load
                                parent[key_new] = (mask, cur_node, svc_from, svc_arc_id)
                                heapq.heappush(pq_state, (new_rc, serial, new_mask, svc_to))
                                serial += 1
                                states_generated += 1

                def _reconstruct_arcs(mask: int, node: Any) -> List[Any]:
                    rev_steps: List[Tuple[Any, Any, Any]] = []
                    cur = (mask, node)
                    while cur in parent:
                        pmask, pnode, svc_from, svc_arc_id = parent[cur]
                        rev_steps.append((pnode, svc_from, svc_arc_id))
                        cur = (pmask, pnode)
                    out: List[Any] = []
                    for pnode, svc_from, svc_arc_id in reversed(rev_steps):
                        out.extend(list(sp_path.get(pnode, {}).get(svc_from, ())))
                        out.append(svc_arc_id)
                    return out

                terminal_candidates: List[Tuple[float, int, Any, Tuple[Any, ...]]] = []
                for (mask, node), rc_val in best_rc.items():
                    if mask == 0:
                        continue
                    back_cost = sp_cost.get(node, {}).get(depot, float("inf"))
                    if not math.isfinite(back_cost):
                        continue
                    total_rc = rc_val + back_cost - _disc_seg(node, depot)
                    if total_rc < -config.eps_reduced_cost:
                        terminal_candidates.append((total_rc, mask, node, sp_path.get(node, {}).get(depot, tuple())))
                terminal_candidates.sort(key=lambda x: x[0])

                for total_rc, mask, node, back_arcs in terminal_candidates:
                    if len(new_columns) >= max_columns:
                        break
                    path_arcs = _reconstruct_arcs(mask, node) + list(back_arcs)
                    rev_steps: List[Tuple[Any, Any, Any]] = []
                    cur_key = (mask, node)
                    while cur_key in parent:
                        pmask, pnode, svc_from, svc_arc_id = parent[cur_key]
                        rev_steps.append((pnode, svc_from, svc_arc_id))
                        cur_key = (pmask, pnode)
                    transitions = [(pnode, svc_from) for pnode, svc_from, _svc_arc in reversed(rev_steps)]
                    transitions.append((node, depot))
                    path_nodes: List[Any] = [depot]
                    for a in path_arcs:
                        if isinstance(a, tuple) and len(a) >= 2:
                            path_nodes.append(a[1])
                    col = {
                        "day": day,
                        "path_nodes": path_nodes,
                        "path_arcs": path_arcs,
                        "nonrequired_edges_used": _nonrequired_edges_from_transitions(transitions),
                        "serviced_required_edges": _decode_serviced(mask, day_bit_to_req),
                        "reduced_cost": float(total_rc),
                    }
                    if driver is not None:
                        col["driver"] = driver
                    if not _add_column_if_new(col):
                        existing_sig_filtered += 1
                        continue
                    negative_found += 1

                day_stats[day_ctx] = {
                    "labels_generated": int(states_generated),
                    "labels_expanded": int(states_expanded),
                    "negative_columns_found": int(negative_found),
                    "backtrack_pruned": 0,
                    "shortcut_returns": 0,
                    "completion_bound_pruned": 0,
                    "existing_sig_filtered": int(existing_sig_filtered),
                }
                total_labels_generated += int(states_generated)
                total_labels_expanded += int(states_expanded)
                total_existing_sig_filtered += int(existing_sig_filtered)
                _prof["pricing_prof_py_dp_s"] += time.perf_counter() - _tpd
                if len(new_columns) >= max_columns:
                    break
                continue

            _tlab = time.perf_counter()
            negative_req_candidates: List[Tuple[int, float, float]] = []
            for req_id, (req_dem, req_serv_cost) in day_req_service_meta.items():
                if req_id in forbidden_edges:
                    continue
                # If service part cannot decrease reduced cost, this req cannot help pricing.
                req_delta = req_serv_cost - edge_duals.get(req_id, 0.0)
                if req_delta < -config.eps_reduced_cost:
                    negative_req_candidates.append((day_req_to_bit[req_id], req_dem, req_delta))

            # sp_cost, sp_path, req_service_arcs, _decode_serviced
            # are all precomputed before the day loop — O(1) lookups replace Dijkstra.

            # ---- Bidirectional labeling (when enabled and beneficial) ----
            use_bidirectional = bool(pricing_data.get("use_bidirectional_labeling", False))
            half_cap = math.floor(capacity / 2.0)

            if use_bidirectional and half_cap >= 1.0 - 1e-9 and len(day_req_ids) >= 2:
                # Bidirectional label-setting algorithm:
                #   Forward labels:  depot → ... → node   (load ≤ half_cap)
                #   Backward labels: node → ... → depot    (load ≤ capacity - half_cap)
                #   Merge: forward@u + deadhead(u,v) + backward@v
                max_bwd_load = capacity - half_cap

                # --- Forward phase (depot → ... → node, load ≤ half_cap) ---
                fwd_node: List[Any] = [depot]
                fwd_load: List[float] = [0.0]
                fwd_rc: List[float] = [base_route_rc]
                fwd_mask: List[int] = [0]
                fwd_parent: List[int] = [-1]
                fwd_service_from: List[Optional[Any]] = [None]
                fwd_service_arc: List[Optional[Any]] = [None]
                fwd_nondom: Dict[Tuple[Any, int], List[int]] = {}
                fwd_expanded = 0
                fwd_generated = 1
                fwd_serial = 0
                fwd_queue: List[Tuple[float, int, int]] = []
                heapq.heappush(fwd_queue, (fwd_rc[0], fwd_serial, 0))
                fwd_serial += 1

                while fwd_queue:
                    _, _, idx = heapq.heappop(fwd_queue)
                    cur_n = fwd_node[idx]
                    cur_l = fwd_load[idx]
                    cur_r = fwd_rc[idx]
                    cur_m = fwd_mask[idx]
                    fwd_expanded += 1

                    # Labels that returned to depot with service done are complete — don't extend.
                    if cur_m > 0 and cur_n == depot:
                        continue

                    for req_id, svc_arcs in day_req_service_arcs.items():
                        if req_id in forbidden_edges:
                            continue
                        req_bit = day_req_to_bit.get(req_id, 0)
                        if req_bit == 0 or (cur_m & req_bit):
                            continue
                        for svc_from, svc_to, svc_arc_id, svc_trav, dem_v, sc_v in svc_arcs:
                            new_l = cur_l + dem_v
                            if new_l > half_cap + 1e-9:
                                continue
                            dh_c = sp_cost.get(cur_n, {}).get(svc_from, float("inf"))
                            if not math.isfinite(dh_c):
                                continue
                            new_r = (
                                cur_r
                                + dh_c
                                + svc_trav
                                + sc_v
                                - edge_duals.get(req_id, 0.0)
                                - _disc_seg(cur_n, svc_from)
                            )
                            new_m = cur_m | req_bit

                            key = (svc_to, new_m)
                            incumbents = fwd_nondom.get(key, [])
                            dominated = False
                            for oi in incumbents:
                                if fwd_load[oi] <= new_l + dom_eps and fwd_rc[oi] <= new_r + dom_eps:
                                    dominated = True
                                    break
                            if dominated:
                                continue
                            kept = [o for o in incumbents
                                    if not (new_l <= fwd_load[o] + dom_eps and new_r <= fwd_rc[o] + dom_eps)]
                            ni = len(fwd_node)
                            fwd_node.append(svc_to)
                            fwd_load.append(new_l)
                            fwd_rc.append(new_r)
                            fwd_mask.append(new_m)
                            fwd_parent.append(idx)
                            fwd_service_from.append(svc_from)
                            fwd_service_arc.append(svc_arc_id)
                            kept.append(ni)
                            fwd_nondom[key] = kept
                            heapq.heappush(fwd_queue, (new_r, fwd_serial, ni))
                            fwd_serial += 1
                            fwd_generated += 1

                # --- Backward phase (node → ... → depot, load ≤ max_bwd_load) ---
                # Backward extension of label B@w by service arc (svc_from→svc_to):
                #   Route segment (forward dir): svc_from → svc_to → deadhead(svc_to, w)
                #   New backward label at svc_from.
                bwd_node: List[Any] = [depot]
                bwd_load: List[float] = [0.0]
                bwd_rc: List[float] = [0.0]  # vehicle dual only in forward
                bwd_mask: List[int] = [0]
                bwd_parent: List[int] = [-1]
                bwd_service_to: List[Optional[Any]] = [None]
                bwd_service_arc: List[Optional[Any]] = [None]
                bwd_nondom: Dict[Tuple[Any, int], List[int]] = {}
                bwd_expanded = 0
                bwd_generated = 1
                bwd_serial = 0
                bwd_queue: List[Tuple[float, int, int]] = []
                heapq.heappush(bwd_queue, (0.0, bwd_serial, 0))
                bwd_serial += 1

                while bwd_queue:
                    _, _, idx = heapq.heappop(bwd_queue)
                    cur_n = bwd_node[idx]
                    cur_l = bwd_load[idx]
                    cur_r = bwd_rc[idx]
                    cur_m = bwd_mask[idx]
                    bwd_expanded += 1

                    for req_id, svc_arcs in day_req_service_arcs.items():
                        if req_id in forbidden_edges:
                            continue
                        req_bit = day_req_to_bit.get(req_id, 0)
                        if req_bit == 0 or (cur_m & req_bit):
                            continue
                        for svc_from, svc_to, svc_arc_id, svc_trav, dem_v, sc_v in svc_arcs:
                            new_l = cur_l + dem_v
                            if new_l > max_bwd_load + 1e-9:
                                continue
                            # Backward: deadhead from service endpoint to backward label node
                            dh_c = sp_cost.get(svc_to, {}).get(cur_n, float("inf"))
                            if not math.isfinite(dh_c):
                                continue
                            new_r = (
                                cur_r
                                + dh_c
                                + svc_trav
                                + sc_v
                                - edge_duals.get(req_id, 0.0)
                                - _disc_seg(svc_to, cur_n)
                            )
                            new_m = cur_m | req_bit

                            # New backward label at svc_from (service start in forward direction)
                            key = (svc_from, new_m)
                            incumbents = bwd_nondom.get(key, [])
                            dominated = False
                            for oi in incumbents:
                                if bwd_load[oi] <= new_l + dom_eps and bwd_rc[oi] <= new_r + dom_eps:
                                    dominated = True
                                    break
                            if dominated:
                                continue
                            kept = [o for o in incumbents
                                    if not (new_l <= bwd_load[o] + dom_eps and new_r <= bwd_rc[o] + dom_eps)]
                            ni = len(bwd_node)
                            bwd_node.append(svc_from)
                            bwd_load.append(new_l)
                            bwd_rc.append(new_r)
                            bwd_mask.append(new_m)
                            bwd_parent.append(idx)
                            bwd_service_to.append(svc_to)
                            bwd_service_arc.append(svc_arc_id)
                            kept.append(ni)
                            bwd_nondom[key] = kept
                            heapq.heappush(bwd_queue, (new_r, bwd_serial, ni))
                            bwd_serial += 1
                            bwd_generated += 1

                # --- Merge phase ---
                def _recon_fwd(fi: int) -> List[Any]:
                    """Reconstruct forward path arcs (depot → ... → node)."""
                    rev_idx: List[int] = []
                    cur = fi
                    while fwd_parent[cur] >= 0:
                        rev_idx.append(cur)
                        cur = fwd_parent[cur]
                    out: List[Any] = []
                    for ci in reversed(rev_idx):
                        pi = fwd_parent[ci]
                        pnode = fwd_node[pi]
                        svc_from = fwd_service_from[ci]
                        svc_arc = fwd_service_arc[ci]
                        if svc_from is None or svc_arc is None:
                            continue
                        out.extend(list(sp_path.get(pnode, {}).get(svc_from, ())))
                        out.append(svc_arc)
                    return out

                def _recon_bwd(bi: int) -> List[Any]:
                    """Reconstruct backward path arcs (node → ... → depot), already in forward order."""
                    out: List[Any] = []
                    cur = bi
                    while bwd_parent[cur] >= 0:
                        pi = bwd_parent[cur]
                        pnode = bwd_node[pi]
                        svc_to = bwd_service_to[cur]
                        svc_arc = bwd_service_arc[cur]
                        if svc_to is None or svc_arc is None:
                            cur = pi
                            continue
                        out.append(svc_arc)
                        out.extend(list(sp_path.get(svc_to, {}).get(pnode, ())))
                        cur = bwd_parent[cur]
                    return out

                # Collect non-dominated forward labels (including root at depot)
                fwd_set: set = {0}
                for _, indices in fwd_nondom.items():
                    fwd_set.update(indices)

                # Group non-dominated backward labels by node (including root at depot)
                bwd_by_n: Dict[Any, List[int]] = {}
                bwd_by_n.setdefault(depot, []).append(0)
                for _, indices in bwd_nondom.items():
                    for bi in indices:
                        bwd_by_n.setdefault(bwd_node[bi], []).append(bi)

                merge_cands: List[Tuple[float, int, int, Any, Any, int]] = []
                for fi in fwd_set:
                    fn = fwd_node[fi]
                    fr = fwd_rc[fi]
                    fl = fwd_load[fi]
                    fm = fwd_mask[fi]
                    for bn, bi_list in bwd_by_n.items():
                        dh = sp_cost.get(fn, {}).get(bn, float("inf"))
                        if not math.isfinite(dh):
                            continue
                        for bi in bi_list:
                            bm = bwd_mask[bi]
                            if fm & bm:
                                continue
                            bl = bwd_load[bi]
                            if fl + bl > capacity + 1e-9:
                                continue
                            mm = fm | bm
                            if mm == 0:
                                continue  # empty route
                            mrc = fr + dh + bwd_rc[bi] - _disc_seg(fn, bn)
                            if mrc < -config.eps_reduced_cost:
                                merge_cands.append((mrc, fi, bi, fn, bn, mm))

                merge_cands.sort(key=lambda x: x[0])

                negative_found = 0
                existing_sig_filtered = 0
                for mrc, fi, bi, fn, bn, mm in merge_cands:
                    if len(new_columns) >= max_columns:
                        break
                    fa = _recon_fwd(fi)
                    da = list(sp_path.get(fn, {}).get(bn, ()))
                    ba = _recon_bwd(bi)
                    full_arcs = fa + da + ba
                    pn: List[Any] = [depot]
                    for a in full_arcs:
                        if isinstance(a, tuple) and len(a) >= 2:
                            pn.append(a[1])
                    col = {
                        "day": day,
                        "path_nodes": pn,
                        "path_arcs": full_arcs,
                        "serviced_required_edges": _decode_serviced(mm, day_bit_to_req),
                        "reduced_cost": float(mrc),
                    }
                    if driver is not None:
                        col["driver"] = driver
                    if not _add_column_if_new(col):
                        existing_sig_filtered += 1
                        continue
                    negative_found += 1

                day_stats[day_ctx] = {
                    "labels_generated": fwd_generated + bwd_generated,
                    "labels_expanded": fwd_expanded + bwd_expanded,
                    "negative_columns_found": negative_found,
                    "backtrack_pruned": 0,
                    "shortcut_returns": 0,
                    "completion_bound_pruned": 0,
                    "existing_sig_filtered": existing_sig_filtered,
                }
                total_labels_generated += fwd_generated + bwd_generated
                total_labels_expanded += fwd_expanded + bwd_expanded
                total_existing_sig_filtered += existing_sig_filtered
                if len(new_columns) >= max_columns:
                    break
                _prof["pricing_prof_python_label_s"] += time.perf_counter() - _tlab
                continue  # next day_ctx

            # ---- Unidirectional labeling fallback ----
            def _fractional_relaxed_service_delta(cur_mask: int, rem_capacity: float) -> float:
                """
                q-path style completion relaxation:
                minimum additional service reduced-cost under capacity by allowing
                fractional servicing (optimistic lower bound for fathoming).
                """
                if rem_capacity <= 1e-12:
                    return 0.0

                # Zero-demand improving edges can always be taken.
                out_delta = 0.0
                items: List[Tuple[float, float, float]] = []
                for req_bit, req_dem, req_delta in negative_req_candidates:
                    if cur_mask & req_bit:
                        continue
                    if req_dem <= 1e-12:
                        out_delta += req_delta
                        continue
                    items.append((req_delta / req_dem, req_dem, req_delta))

                # Take most negative delta-per-demand first (fractional knapsack relaxation).
                items.sort(key=lambda x: x[0])
                cap_left = rem_capacity
                for _, dem, delta in items:
                    if cap_left <= 1e-12:
                        break
                    if dem <= cap_left + 1e-12:
                        out_delta += delta
                        cap_left -= dem
                    else:
                        frac = cap_left / dem
                        out_delta += delta * frac
                        cap_left = 0.0
                return out_delta

            # Label pool with parent pointers to avoid path tuple copies.
            # index -> label fields
            label_node: List[Any] = [depot]
            label_load: List[float] = [0.0]
            label_rc: List[float] = [base_route_rc]
            label_mask: List[int] = [0]
            label_parent: List[int] = [-1]
            # Per-label transition descriptor: shortest-path parent_node->svc_from, then svc_arc.
            label_service_from: List[Optional[Any]] = [None]
            label_service_arc: List[Optional[Any]] = [None]
            label_len: List[int] = [0]

            def _build_column(day_val: Any, lbl_idx: int, extra_arcs: Sequence[Any] = (), rc_override: Optional[float] = None) -> Dict[str, Any]:
                chain: List[int] = []
                cur = lbl_idx
                while label_parent[cur] >= 0:
                    chain.append(cur)
                    cur = label_parent[cur]

                path_arcs: List[Any] = []
                transitions: List[Tuple[Any, Any]] = []
                for ci in reversed(chain):
                    pi = label_parent[ci]
                    pnode = label_node[pi]
                    svc_from = label_service_from[ci]
                    svc_arc = label_service_arc[ci]
                    if svc_from is None or svc_arc is None:
                        continue
                    transitions.append((pnode, svc_from))
                    path_arcs.extend(list(sp_path.get(pnode, {}).get(svc_from, ())))
                    path_arcs.append(svc_arc)
                if extra_arcs:
                    path_arcs.extend(extra_arcs)
                transitions.append((label_node[lbl_idx], depot))

                path_nodes: List[Any] = [depot]
                for a in path_arcs:
                    if isinstance(a, tuple) and len(a) >= 2:
                        path_nodes.append(a[1])

                out = {
                    "day": day_val,
                    "path_nodes": path_nodes,
                    "path_arcs": path_arcs,
                    "nonrequired_edges_used": _nonrequired_edges_from_transitions(transitions),
                    "serviced_required_edges": _decode_serviced(label_mask[lbl_idx], day_bit_to_req),
                    "reduced_cost": float(label_rc[lbl_idx] if rc_override is None else rc_override),
                }
                if driver is not None:
                    out["driver"] = driver
                return out

            # PQ entries: (reduced_cost, serial, label_idx)
            queue: List[Tuple[float, int, int]] = []
            serial = 0
            heapq.heappush(queue, (label_rc[0], serial, 0))
            serial += 1

            nondom: Dict[Tuple[Any, int], List[int]] = {}
            labels_expanded = 0
            labels_generated = 1
            negative_found = 0
            backtrack_pruned = 0
            shortcut_returns = 0
            completion_bound_pruned = 0
            existing_sig_filtered = 0
            # Safety-first default: keep fathoming off unless explicitly enabled.
            use_q_relaxed_fathoming = bool(pricing_data.get("use_q_relaxed_fathoming", False))
            if any(abs(float(v)) > 1e-12 for v in discount_edge_duals.values()):
                # Existing relaxed completion bound does not model discount-link dual gain.
                # Keep pruning conservative when discount duals are active.
                use_q_relaxed_fathoming = False
            while queue and len(new_columns) < max_columns:
                _, _, idx = heapq.heappop(queue)
                labels_expanded += 1
                cur_node = label_node[idx]
                cur_load = label_load[idx]
                cur_rc = label_rc[idx]
                cur_mask = label_mask[idx]
                prev_idx = label_parent[idx]
                prev_node = label_node[prev_idx] if prev_idx >= 0 else None

                if label_len[idx] > 0 and cur_node == depot:
                    if cur_rc < -config.eps_reduced_cost:
                        col = _build_column(day, idx)
                        if not _add_column_if_new(col):
                            existing_sig_filtered += 1
                        else:
                            negative_found += 1
                    # depot에 복귀한 라벨은 종료라벨로만 사용
                    continue

                remaining_capacity = capacity - cur_load
                if use_q_relaxed_fathoming:
                    # Paper-inspired fathoming: optimistic completion lower bound (LB_q-like).
                    back_cost_lb = sp_cost.get(cur_node, {}).get(depot, float("inf"))
                    if math.isfinite(back_cost_lb):
                        service_delta_lb = _fractional_relaxed_service_delta(cur_mask, remaining_capacity)
                        completion_lb = cur_rc + back_cost_lb + service_delta_lb
                        if completion_lb >= -config.eps_reduced_cost:
                            completion_bound_pruned += 1
                            continue

                # Early depot return candidate from any partial label.
                if cur_mask != 0 and cur_node != depot:
                    back_cost_any = sp_cost.get(cur_node, {}).get(depot, float("inf"))
                    back_arcs_any = sp_path.get(cur_node, {}).get(depot, ())
                    if math.isfinite(back_cost_any):
                        rc_back_any = cur_rc + back_cost_any - _disc_seg(cur_node, depot)
                        if rc_back_any < -config.eps_reduced_cost:
                            col_any = _build_column(day, idx, extra_arcs=back_arcs_any, rc_override=rc_back_any)
                            if not _add_column_if_new(col_any):
                                existing_sig_filtered += 1
                            else:
                                negative_found += 1
                                if len(new_columns) >= max_columns:
                                    break

                # If no unserved feasible required edge remains, return to depot directly.
                has_feasible_service = False
                if remaining_capacity > 1e-9:
                    for req_id, (req_dem, _) in day_req_service_meta.items():
                        if req_id in forbidden_edges:
                            continue
                        req_bit = day_req_to_bit.get(req_id, 0)
                        if cur_mask & req_bit:
                            continue
                        if req_dem <= remaining_capacity + 1e-9:
                            has_feasible_service = True
                            break
                if not has_feasible_service:
                    back_cost = sp_cost.get(cur_node, {}).get(depot, float("inf"))
                    back_arcs = sp_path.get(cur_node, {}).get(depot, ())
                    if math.isfinite(back_cost):
                        rc_back = cur_rc + back_cost - _disc_seg(cur_node, depot)
                        if rc_back < -config.eps_reduced_cost:
                            col = _build_column(day, idx, extra_arcs=back_arcs, rc_override=rc_back)
                            if not _add_column_if_new(col):
                                existing_sig_filtered += 1
                            else:
                                negative_found += 1
                    shortcut_returns += 1
                    continue

                # Expand by choosing next serviced required edge.
                # Deadheading part is fixed to the shortest path, per paper assumption.
                for req_id, svc_arcs in day_req_service_arcs.items():
                    if req_id in forbidden_edges:
                        continue
                    req_bit = day_req_to_bit.get(req_id, 0)
                    if req_bit == 0 or (cur_mask & req_bit) != 0:
                        continue

                    for svc_from, svc_to, svc_arc_id, svc_travel, demand, service_cost in svc_arcs:
                        if (
                            forbid_immediate_backtrack
                            and prev_node is not None
                            and svc_from == prev_node
                            and cur_node != svc_from
                        ):
                            backtrack_pruned += 1
                            continue

                        new_load = cur_load + demand
                        if new_load > capacity + 1e-9:
                            continue

                        deadhead_cost = sp_cost.get(cur_node, {}).get(svc_from, float("inf"))
                        if not math.isfinite(deadhead_cost):
                            continue

                        new_mask = cur_mask | req_bit
                        d_rb = cut_pricing_state.rb_delta(cur_mask, new_mask, day_bit_to_req)
                        new_rc = (
                            cur_rc
                            + deadhead_cost
                            + svc_travel
                            + service_cost
                            - edge_duals.get(req_id, 0.0)
                            - _disc_seg(cur_node, svc_from)
                            + d_rb
                        )

                        key = (svc_to, new_mask)
                        incumbents = nondom.get(key, [])
                        dominated = False
                        for old_idx in incumbents:
                            if label_load[old_idx] <= new_load + dom_eps and label_rc[old_idx] <= new_rc + dom_eps:
                                dominated = True
                                break
                        if dominated:
                            continue

                        filtered = [
                            old
                            for old in incumbents
                            if not (
                                new_load <= label_load[old] + dom_eps
                                and new_rc <= label_rc[old] + dom_eps
                            )
                        ]
                        new_idx = len(label_node)
                        label_node.append(svc_to)
                        label_load.append(new_load)
                        label_rc.append(new_rc)
                        label_mask.append(new_mask)
                        label_parent.append(idx)
                        label_service_from.append(svc_from)
                        label_service_arc.append(svc_arc_id)
                        label_len.append(label_len[idx] + 1)

                        filtered.append(new_idx)
                        nondom[key] = filtered

                        heapq.heappush(queue, (new_rc, serial, new_idx))
                        serial += 1
                        labels_generated += 1

            day_stats[day_ctx] = {
                "labels_generated": labels_generated,
                "labels_expanded": labels_expanded,
                "negative_columns_found": negative_found,
                "backtrack_pruned": backtrack_pruned,
                "shortcut_returns": shortcut_returns,
                "completion_bound_pruned": completion_bound_pruned,
                "existing_sig_filtered": existing_sig_filtered,
            }
            total_labels_generated += labels_generated
            total_labels_expanded += labels_expanded
            total_backtrack_pruned += backtrack_pruned
            total_shortcut_returns += shortcut_returns
            total_completion_bound_pruned += completion_bound_pruned
            total_existing_sig_filtered += existing_sig_filtered

            _prof["pricing_prof_python_label_s"] += time.perf_counter() - _tlab

            if len(new_columns) >= max_columns:
                break

        for day_ctx, st in day_stats.items():
            old_p = float(priority_store.get(day_ctx, 1.0))
            neg = int(st.get("negative_columns_found", 0))
            if neg > 0:
                priority_store[day_ctx] = min(10.0, 0.85 * old_p + 2.0)
            else:
                priority_store[day_ctx] = max(0.1, 0.85 * old_p + 0.15)

        _meta_out = {
            "node_id": self.node_id,
            "num_new_columns": len(new_columns),
            "day_stats": day_stats,
            "labels_generated": total_labels_generated,
            "labels_expanded": total_labels_expanded,
            "backtrack_pruned": total_backtrack_pruned,
            "shortcut_returns": total_shortcut_returns,
            "completion_bound_pruned": total_completion_bound_pruned,
            "existing_sig_filtered": total_existing_sig_filtered,
            "coeff_dominated_filtered": total_coeff_dominated_filtered,
        }
        _meta_out.update(_prof)
        return PricingResult(new_columns=new_columns, metadata=_meta_out)

    def add_columns_to_rmp(self, columns: Sequence[Any]) -> int:
        """Insert generated columns/routes into RMP and internal cache."""
        self.routes.extend(list(columns))
        attempted = int(len(columns))

        def _store_add_stats(ret: Any) -> int:
            if isinstance(ret, dict):
                stats = {
                    "attempted": int(ret.get("attempted", attempted)),
                    "added": int(ret.get("added", 0)),
                    "skipped_empty": int(ret.get("skipped_empty", 0)),
                    "skipped_capacity": int(ret.get("skipped_capacity", 0)),
                    "skipped_duplicate": int(ret.get("skipped_duplicate", 0)),
                    "skipped_dominated": int(ret.get("skipped_dominated", 0)),
                }
                setattr(self, "_last_addcol_stats", stats)
                return int(stats["added"])
            if isinstance(ret, int):
                added = int(ret)
            else:
                added = attempted
            setattr(
                self,
                "_last_addcol_stats",
                {
                    "attempted": attempted,
                    "added": added,
                    "skipped_empty": 0,
                    "skipped_capacity": 0,
                    "skipped_duplicate": max(0, attempted - added),
                    "skipped_dominated": 0,
                },
            )
            return added

        # Preferred extension points on the RMP wrapper
        if hasattr(self.rmp, "add_pricing_columns"):
            ret = self.rmp.add_pricing_columns(columns)
            return _store_add_stats(ret)
        if hasattr(self.rmp, "add_columns"):
            ret = self.rmp.add_columns(columns)
            return _store_add_stats(ret)
        if hasattr(self.rmp, "column_manager") and hasattr(self.rmp.column_manager, "add_columns"):
            ret = self.rmp.column_manager.add_columns(columns)
            return _store_add_stats(ret)

        # Raw gurobi.Model cannot be updated with columns without model-specific mapping.
        try:
            import gurobipy as gp
        except ImportError:
            gp = None

        if gp is not None and isinstance(self.rmp, gp.Model):
            raise RuntimeError(
                "Cannot add pricing columns to bare gurobipy.Model automatically. "
                "Provide `rmp.add_pricing_columns(columns)` interface."
            )
        if gp is not None and isinstance(getattr(self.rmp, "model", None), gp.Model):
            raise RuntimeError(
                "Cannot add pricing columns: wrapper must provide `add_pricing_columns`/`add_columns`."
            )
        setattr(
            self,
            "_last_addcol_stats",
            {
                "attempted": attempted,
                "added": 0,
                "skipped_empty": 0,
                "skipped_capacity": 0,
                "skipped_duplicate": attempted,
                "skipped_dominated": 0,
            },
        )
        return 0

    def extract_fractional_objects(self, config: BnBConfig) -> List[BranchCandidate]:
        if self._is_aggregated_master_active():
            return self._extract_fractional_objects_aggregated(config)
        return self._extract_fractional_objects_simple_sp(config)

    def _extract_fractional_objects_aggregated(self, config: BnBConfig) -> List[BranchCandidate]:
        """
        A-RMP-specific branching hierarchy (우선순위):

        1. whole_route   — Σ_{t,r} λ^t_r (집계 경로 질량)
        2. daily_route   — 일별 Σ_r λ^t_r
        3. schedule_fix  — 집계 스케줄 q_{e,p} = Σ_k s_{e,p,k} (차량 인덱스 없음, 패턴별 0/1 고정)
        4. lambda_var      — 개별 집계 경로 변수 agg_lam_t{t}_r* (0/1 고정)

        SimpleSP와 분리한 이유: A-RMP는 일별 λ^t_r·집계 q_{e,p} 만 있고 per-(t,k) 분기 표현이 없음.
        """
        eps = config.eps_integrality
        lam_vars = self._get_lambda_vars()

        if lam_vars:
            total = float(sum(v.X for v in lam_vars))
            if self._is_fractional(total, eps):
                return [BranchCandidate(
                    family=ARMP_WHOLE_ROUTE_FAMILY,
                    target={"metric": "sum_lambda_all"},
                    value=total,
                )]

        by_day = self._get_lambda_vars_by_day()
        level_daily: List[BranchCandidate] = []
        for day_key, vars_for_day in by_day.items():
            total_t = float(sum(v.X for v in vars_for_day))
            if self._is_fractional(total_t, eps):
                level_daily.append(BranchCandidate(
                    family=ARMP_DAILY_ROUTE_FAMILY,
                    target={"metric": "sum_lambda_day", "day": day_key},
                    value=total_t,
                    day=day_key,
                ))
        if level_daily:
            return level_daily

        data = self._get_branching_data(include_route_lifting=False)

        level_schedule_fix: List[BranchCandidate] = []
        sched_vars = data.get("schedule_vars", {})
        if isinstance(sched_vars, dict):
            for sk, vref in sched_vars.items():
                var = self._resolve_model_var(vref)
                if var is None:
                    continue
                try:
                    vx = float(var.X)
                except Exception:
                    continue
                if self._is_fractional(vx, eps):
                    level_schedule_fix.append(BranchCandidate(
                        family=ARMP_SCHEDULE_FIX_FAMILY,
                        target={"schedule_key": sk, "var_name": vref},
                        value=vx,
                    ))
        if level_schedule_fix:
            return level_schedule_fix

        level_lambda_var: List[BranchCandidate] = []
        lam_by_day = data.get("lambda_vars_by_day")
        if isinstance(lam_by_day, dict):
            day_keys = sorted(
                lam_by_day.keys(),
                key=lambda d: (0, int(d)) if isinstance(d, int) else (1, str(d)),
            )
            for day_key in day_keys:
                refs = lam_by_day.get(day_key)
                if not isinstance(refs, (list, tuple)):
                    continue
                for ref in refs:
                    var = self._resolve_model_var(ref)
                    if var is None:
                        continue
                    try:
                        vx = float(var.X)
                    except Exception:
                        continue
                    if not self._is_fractional(vx, eps):
                        continue
                    vnm = str(ref) if isinstance(ref, str) else str(getattr(var, "VarName", "") or "")
                    if not vnm:
                        continue
                    day_int: Optional[int] = None
                    try:
                        day_int = int(day_key)
                    except (TypeError, ValueError):
                        day_int = None
                    level_lambda_var.append(
                        BranchCandidate(
                            family="lambda_var",
                            target={"var_name": vnm},
                            value=vx,
                            day=day_int,
                        )
                    )
        if level_lambda_var:
            return level_lambda_var

        return []

    def _extract_fractional_objects_simple_sp(self, config: BnBConfig) -> List[BranchCandidate]:
        """Collect fractional branching candidates in priority order.

        Hierarchy (aggregate-first, 6b→6a):
          Level 6b whole_route              Σ_{t,k,r} λ^{tk}_r
          Level 6a daily_route              Σ_{k,r} λ^{tk}_r per day t
          Level 5  edge_driver_assign      z_ek = Σ_p s_e^{pk} (edge e served by vehicle k; no pattern branch)
          Level 4  visit_node              w_itk expression
          Level 3  visit_arc               x+y expression
          Level 2  edge_day_driver_service x_etk
        """
        eps = config.eps_integrality
        lam_vars = self._get_lambda_vars()

        # ------------------------------------------------------------------
        # Level 6b: whole-route aggregate (first priority)
        # ------------------------------------------------------------------
        if lam_vars:
            total = float(sum(v.X for v in lam_vars))
            if self._is_fractional(total, eps):
                return [BranchCandidate(
                    family=RMP_WHOLE_ROUTE_FAMILY,
                    target={"metric": "sum_lambda_all"},
                    value=total,
                )]

        # ------------------------------------------------------------------
        # Level 6a: daily-route aggregate
        # ------------------------------------------------------------------
        by_day = self._get_lambda_vars_by_day()
        level6a: List[BranchCandidate] = []
        for day_key, vars_for_day in by_day.items():
            total_tk = float(sum(v.X for v in vars_for_day))
            if self._is_fractional(total_tk, eps):
                level6a.append(BranchCandidate(
                    family=RMP_DAILY_ROUTE_FAMILY,
                    target={"metric": "sum_lambda_day", "day": day_key},
                    value=total_tk,
                    day=day_key,
                ))
        if level6a:
            return level6a

        # Schedule-only lifting (cheap). Full λ route lifting is deferred until level 5 fails.
        data = self._get_branching_data(include_route_lifting=False)

        # ------------------------------------------------------------------
        # Level 5: z_ek = Σ_p s_e^{pk} — which vehicle covers required edge e (schedule pattern summed out).
        # ------------------------------------------------------------------
        level5: List[BranchCandidate] = []
        z_exprs = data.get("edge_driver_assign_expr", {})
        if isinstance(z_exprs, dict):
            for key, expr in z_exprs.items():
                if not isinstance(expr, dict):
                    continue
                val = float(self._expr_value(expr, self._resolve_model_var))
                if self._is_fractional(val, eps):
                    drv = key[1] if isinstance(key, tuple) and len(key) >= 2 else None
                    level5.append(BranchCandidate(
                        family=RMP_EDGE_DRIVER_ASSIGN_FAMILY,
                        target={"edge_driver_key": key},
                        value=val,
                        driver=int(drv) if drv is not None else None,
                    ))
        if level5:
            return level5

        ext = getattr(self.rmp, "extend_branching_route_expressions", None)
        if callable(ext):
            ext(data)
        else:
            data = self._get_branching_data(include_route_lifting=True)

        # ------------------------------------------------------------------
        # Level 4: node-visit expression branching
        # ------------------------------------------------------------------
        level4: List[BranchCandidate] = []
        node_exprs = data.get("node_visit_expr", {})
        if isinstance(node_exprs, dict):
            for key, expr in node_exprs.items():
                if not isinstance(expr, dict):
                    continue
                val = float(self._expr_value(expr, self._resolve_model_var))
                if self._is_fractional(val, eps):
                    # key = (node_id, t, k)
                    day = key[1] if isinstance(key, tuple) and len(key) >= 2 else None
                    level4.append(BranchCandidate(
                        family=RMP_VISIT_NODE_FAMILY,
                        target={"node_day_key": key},
                        value=val,
                        day=day,
                    ))
        if level4:
            return level4

        # ------------------------------------------------------------------
        # Level 3: edge-visit expression branching
        # ------------------------------------------------------------------
        level3: List[BranchCandidate] = []
        arc_exprs = data.get("arc_visit_expr", {})
        if isinstance(arc_exprs, dict):
            for key, expr in arc_exprs.items():
                if not isinstance(expr, dict):
                    continue
                val = float(self._expr_value(expr, self._resolve_model_var))
                if self._is_fractional(val, eps):
                    # key = (canon_edge, t, k)
                    day = key[1] if isinstance(key, tuple) and len(key) >= 2 else None
                    level3.append(BranchCandidate(
                        family=RMP_VISIT_ARC_FAMILY,
                        target={"arc_day_key": key},
                        value=val,
                        day=day,
                    ))
        if level3:
            return level3

        # ------------------------------------------------------------------
        # Level 2: x_etk = Σ_{p: t∈pat} s_e^{pk}
        # ------------------------------------------------------------------
        level2: List[BranchCandidate] = []
        x_exprs = data.get("edge_day_driver_service_expr", {})
        if isinstance(x_exprs, dict):
            for key, expr in x_exprs.items():
                if not isinstance(expr, dict):
                    continue
                val = float(self._expr_value(expr, self._resolve_model_var))
                if self._is_fractional(val, eps):
                    day = int(key[1]) if isinstance(key, tuple) and len(key) >= 3 else None
                    drv = int(key[2]) if isinstance(key, tuple) and len(key) >= 3 else None
                    level2.append(BranchCandidate(
                        family=RMP_EDGE_DAY_DRIVER_SERVICE_FAMILY,
                        target={"edge_day_driver_key": key},
                        value=val,
                        day=day,
                        driver=drv,
                    ))
        if level2:
            return level2

        return []

    def choose_branch_candidate(self, candidates: Sequence[BranchCandidate]) -> Optional[BranchCandidate]:
        """Pick one candidate for branching (simple/strong branching policy)."""
        if not candidates:
            return None
        ranked = sorted(candidates, key=branch_candidate_sort_key)
        return ranked[0]

    def build_child_constraints(self, candidate: BranchCandidate) -> Tuple[BranchConstraint, BranchConstraint]:
        """Create left/right branch constraints from one candidate."""
        family = candidate.family
        value = float(candidate.value)

        # Binary-like branching (must match BINARY_LIKE_BRANCH_FAMILIES)
        if str(family) in BINARY_LIKE_BRANCH_FAMILIES:
            left = BranchConstraint(
                family=family,
                target=candidate.target,
                sense="<=",
                rhs=0.0,
                day=candidate.day,
            )
            right = BranchConstraint(
                family=family,
                target=candidate.target,
                sense=">=",
                rhs=1.0,
                day=candidate.day,
            )
            return left, right

        # Integer split branching: x <= floor(x*) or x >= ceil(x*)
        floor_v = math.floor(value)
        ceil_v = math.ceil(value)
        left = BranchConstraint(
            family=family,
            target=candidate.target,
            sense="<=",
            rhs=float(floor_v),
            day=candidate.day,
        )
        right = BranchConstraint(
            family=family,
            target=candidate.target,
            sense=">=",
            rhs=float(ceil_v),
            day=candidate.day,
        )
        return left, right

    def get_integer_solution_if_any(self) -> Optional[Any]:
        """Return node solution when LP is integral (or None)."""
        model = self._get_gurobi_model()
        eps = 1e-6

        # Fast check from hierarchical branching expressions
        if self.extract_fractional_objects(BnBConfig(eps_integrality=eps)):
            return None

        values: Dict[str, int] = {}
        for var in model.getVars():
            xv = float(var.X)
            if abs(xv) > eps:
                values[var.VarName] = int(round(xv))

        obj_out = float(model.ObjVal)
        _dis = getattr(self, "_disaggregated_solution", None)
        _rmp_o = getattr(self, "rmp", None)
        if _dis is not None and _rmp_o is not None and hasattr(
            _rmp_o, "reconcile_disaggregated_incumbent_objective"
        ):
            try:
                armp_f = {v.VarName: float(v.X) for v in model.getVars()}
                _rec = float(_rmp_o.reconcile_disaggregated_incumbent_objective(armp_f, _dis))
                obj_out = max(obj_out, _rec)
            except Exception:
                pass

        return {
            "node_id": self.node_id,
            "objective": obj_out,
            "variables": values,
        }


class NodeSelector:
    """Policy object for selecting the next open node."""

    def __init__(self) -> None:
        self._open: List[BnBNode] = []

    def push(self, node: BnBNode) -> None:
        self._open.append(node)

    def pop(self) -> BnBNode:
        if not self._open:
            raise IndexError("No open nodes.")
        return self._open.pop(0)

    def is_empty(self) -> bool:
        return len(self._open) == 0

    def best_lower_bound(self) -> float:
        if not self._open:
            return float("inf")
        lbs: List[float] = []
        for n in self._open:
            lb = float(getattr(n, "lower_bound", float("inf")))
            lbs.append(lb)
        return min(lbs) if lbs else float("inf")


class BestBoundSelector(NodeSelector):
    """Best-bound-first selector (recommended baseline for branch-and-price)."""

    def __init__(self) -> None:
        self._open: List[Tuple[float, int, int, BnBNode]] = []
        self._serial = 0

    def push(self, node: BnBNode) -> None:
        lb = float(node.lower_bound) if node.lower_bound is not None else float("inf")
        heapq.heappush(self._open, (lb, node.depth, self._serial, node))
        self._serial += 1

    def pop(self) -> BnBNode:
        if not self._open:
            raise IndexError("No open nodes.")
        return heapq.heappop(self._open)[3]

    def is_empty(self) -> bool:
        return len(self._open) == 0

    def best_lower_bound(self) -> float:
        if not self._open:
            return float("inf")
        return float(self._open[0][0])


class DepthFirstSelector(NodeSelector):
    """Depth-first selector (LIFO stack).

    DFS 전략:
    - 가장 최근에 생성된(가장 깊은) 노드를 먼저 처리.
    - 빠르게 feasible 정수해를 찾아 UB를 낮추고 pruning 효과를 극대화.
    - 현재처럼 aggregate branching(whole/daily/schedule/node/arc)을 쓸 때
      각 분기가 강한 제약을 걸기 때문에 DFS로 빠른 UB를 얻는 게 효과적.
    - global_LB는 모든 open 노드 중 최솟값으로 계산(BFS와 동일 보장).
    """

    def __init__(self) -> None:
        self._stack: List[BnBNode] = []

    def push(self, node: BnBNode) -> None:
        self._stack.append(node)

    def pop(self) -> BnBNode:
        if not self._stack:
            raise IndexError("No open nodes.")
        return self._stack.pop()          # LIFO — 마지막에 넣은 노드(깊은 노드) 먼저

    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def best_lower_bound(self) -> float:
        if not self._stack:
            return float("inf")
        return min(
            float(getattr(n, "lower_bound", float("inf")))
            for n in self._stack
        )


class BnBTree:
    """
    Branch-and-bound tree manager.

    Responsibilities:
    - open-node 관리
    - incumbent(최적 정수해) 관리
    - node prune / branch / terminate 판단
    """

    def __init__(self, root_node: BnBNode, config: Optional[BnBConfig] = None, selector: Optional[NodeSelector] = None) -> None:
        self.root_node = root_node
        self.config = config or BnBConfig()
        self.selector = selector or BestBoundSelector()

        self.best_solution: Optional[Any] = None
        self.global_upper_bound: float = float("inf")
        self.global_lower_bound: float = float("inf")

        self.nodes_created: int = 1
        self.nodes_processed: int = 0
        self.terminated_by_node_limit: bool = False
        self.terminated_by_time_limit: bool = False
        self._solve_start_ts: Optional[float] = None
        self.profile: Dict[str, float] = {
            "rmp_time_s": 0.0,
            "rmp_lp_time_s": 0.0,
            "cut_separation_time_s": 0.0,
            "pricing_time_s": 0.0,
            "addcol_time_s": 0.0,
            "labels_generated": 0.0,
            "labels_expanded": 0.0,
            "backtrack_pruned": 0.0,
            "shortcut_returns": 0.0,
            "existing_sig_filtered": 0.0,
            "coeff_dominated_filtered": 0.0,
            "columns_generated": 0.0,
            "columns_added": 0.0,
            "columns_attempted_add": 0.0,
            "columns_skipped_duplicate": 0.0,
            "columns_skipped_dominated": 0.0,
            "zero_add_iterations": 0.0,
            "nodes_hit_cg_limit": 0.0,
        }

    def _seed_initial_upper_bound_if_available(self) -> None:
        """ALNS / root MIP incumbents on the master (see compare_arc_vs_bnp.seed_bnb_tree_initial_ub)."""
        try:
            from src.master.compare_arc_vs_bnp import seed_bnb_tree_initial_ub

            rmp = getattr(self.root_node, "rmp", None)
            if rmp is None:
                return
            inst = getattr(rmp, "inst", None)
            tl = float(inst.get("root_incumbent_time_limit_s", 3.0)) if isinstance(inst, dict) else 3.0
            seed_bnb_tree_initial_ub(self, rmp, time_limit_s=tl)
        except Exception:
            pass

    def _should_skip_open_node_by_bound(self, node: BnBNode) -> bool:
        """
        Fast pre-solve pruning for open nodes.

        Child nodes inherit parent LP lower bounds. If incumbent UB improves later,
        many open nodes can be pruned immediately without running CG/RMP again.
        """
        ub = float(self.global_upper_bound)
        if not math.isfinite(ub):
            return False

        try:
            lb = float(getattr(node, "lower_bound", float("inf")))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(lb):
            return False
        return lb >= ub - self.config.eps_integrality

    @staticmethod
    def _mark_pruned_without_solve(node: BnBNode) -> None:
        node.status = NodeStatus.PRUNED
        node.is_solved = True
        node.is_integral = False

    def solve(self) -> Optional[Any]:
        """Main branch-and-bound loop."""
        self._solve_start_ts = time.perf_counter()
        if self.config.max_time_s is not None and self.config.max_time_s > 0:
            self.config.deadline_ts = self._solve_start_ts + float(self.config.max_time_s)
        else:
            self.config.deadline_ts = None
        self.selector.push(self.root_node)
        self._refresh_global_lower_bound()
        self._seed_initial_upper_bound_if_available()

        while not self.selector.is_empty():
            if self.config.max_nodes is not None and self.nodes_processed >= self.config.max_nodes:
                self.terminated_by_node_limit = True
                break
            if self.config.max_time_s is not None and self.config.max_time_s > 0:
                elapsed = time.perf_counter() - (self._solve_start_ts or time.perf_counter())
                if elapsed >= float(self.config.max_time_s):
                    self.terminated_by_time_limit = True
                    break

            node = self.selector.pop()
            if self._should_skip_open_node_by_bound(node):
                self._mark_pruned_without_solve(node)
                self._refresh_global_lower_bound()
                continue
            node_result = self.process_node(node)
            if self.config.deadline_ts is not None and time.perf_counter() >= float(self.config.deadline_ts):
                self.terminated_by_time_limit = True
                break

            if self.should_prune_node(node_result):
                self._refresh_global_lower_bound()
                continue

            if node_result.is_integral:
                self._refresh_global_lower_bound()
                continue

            candidates = node.extract_fractional_objects(self.config)
            candidate = node.choose_branch_candidate(candidates)
            if candidate is None:
                self._refresh_global_lower_bound()
                continue

            left_child, right_child = self.create_children(node, candidate)
            self.selector.push(left_child)
            self.selector.push(right_child)
            self._refresh_global_lower_bound()

        self._refresh_global_lower_bound()
        return self.best_solution

    def process_node(self, node: BnBNode) -> NodeSolveResult:
        """Solve one node and update global bounds/incumbent."""
        try:
            result = node.solve_node(self.config, self.global_upper_bound)
        except RuntimeError:
            node.status = NodeStatus.INFEASIBLE
            node.is_solved = True
            result = NodeSolveResult(
                node_id=node.node_id,
                status=NodeStatus.INFEASIBLE,
                lower_bound=float("inf"),
                is_integral=False,
            )
        self.nodes_processed += 1

        self.update_global_bounds(result)
        stats = getattr(node, "solve_stats", {})
        self.profile["rmp_time_s"] += float(stats.get("rmp_time_s", 0.0))
        self.profile["rmp_lp_time_s"] += float(stats.get("rmp_lp_time_s", 0.0))
        self.profile["cut_separation_time_s"] += float(stats.get("cut_separation_time_s", 0.0))
        self.profile["pricing_time_s"] += float(stats.get("pricing_time_s", 0.0))
        self.profile["addcol_time_s"] += float(stats.get("addcol_time_s", 0.0))
        self.profile["labels_generated"] += float(stats.get("labels_generated", 0.0))
        self.profile["labels_expanded"] += float(stats.get("labels_expanded", 0.0))
        self.profile["backtrack_pruned"] = self.profile.get("backtrack_pruned", 0.0) + float(
            stats.get("backtrack_pruned", 0.0)
        )
        self.profile["shortcut_returns"] = self.profile.get("shortcut_returns", 0.0) + float(
            stats.get("shortcut_returns", 0.0)
        )
        self.profile["existing_sig_filtered"] = self.profile.get("existing_sig_filtered", 0.0) + float(
            stats.get("existing_sig_filtered", 0.0)
        )
        self.profile["coeff_dominated_filtered"] = self.profile.get("coeff_dominated_filtered", 0.0) + float(
            stats.get("coeff_dominated_filtered", 0.0)
        )
        self.profile["columns_generated"] += float(stats.get("columns_generated", 0.0))
        self.profile["columns_added"] += float(stats.get("columns_added", 0.0))
        self.profile["columns_attempted_add"] += float(stats.get("columns_attempted_add", 0.0))
        self.profile["columns_skipped_duplicate"] += float(stats.get("columns_skipped_duplicate", 0.0))
        self.profile["columns_skipped_dominated"] += float(stats.get("columns_skipped_dominated", 0.0))
        self.profile["zero_add_iterations"] += float(stats.get("zero_add_iterations", 0.0))
        self.profile["cg_iterations"] = self.profile.get("cg_iterations", 0.0) + float(stats.get("cg_iterations", 0.0))
        self.profile["phase1_iters"] = self.profile.get("phase1_iters", 0.0) + float(stats.get("phase1_iters", 0.0))
        for _pk, _pv in stats.items():
            if isinstance(_pk, str) and _pk.startswith("pricing_prof_"):
                self.profile[_pk] = self.profile.get(_pk, 0.0) + float(_pv)
        if bool(stats.get("hit_cg_iteration_limit", False)):
            self.profile["nodes_hit_cg_limit"] += 1.0
        if bool(stats.get("hit_time_limit", False)):
            self.terminated_by_time_limit = True
        if result.is_integral and result.best_integer_obj is not None:
            sol = node.get_integer_solution_if_any()
            if sol is not None:
                self.best_solution = sol
        if bool(self.config.verbose):
            self._print_node_progress(node, result)
        return result

    def should_prune_node(self, node_result: NodeSolveResult) -> bool:
        """Bound- or infeasibility-based pruning checks."""
        if node_result.status in {NodeStatus.INFEASIBLE, NodeStatus.PRUNED}:
            return True
        if node_result.lower_bound >= self.global_upper_bound - self.config.eps_integrality:
            return True
        return False

    def _clone_master_problem(self, master_problem: Any) -> Any:
        """Clone master problem object for child node."""
        if hasattr(master_problem, "copy_for_child"):
            return master_problem.copy_for_child()
        if hasattr(master_problem, "clone"):
            return master_problem.clone()

        try:
            import gurobipy as gp
        except ImportError:
            gp = None

        if gp is not None and isinstance(master_problem, gp.Model):
            return master_problem.copy()

        base_model = getattr(master_problem, "model", None)
        if gp is not None and isinstance(base_model, gp.Model):
            new_model = base_model.copy()
            if hasattr(master_problem, "with_model"):
                return master_problem.with_model(new_model)
            cloned = copy.copy(master_problem)
            setattr(cloned, "model", new_model)
            return cloned

        return copy.deepcopy(master_problem)

    def create_children(self, parent: BnBNode, candidate: BranchCandidate) -> Tuple[BnBNode, BnBNode]:
        """Instantiate child nodes with inherited + new branching constraints."""
        left_bc, right_bc = parent.build_child_constraints(candidate)

        left_constraints = list(parent.constraints) + [left_bc]
        right_constraints = list(parent.constraints) + [right_bc]

        left_master = self._clone_master_problem(parent.rmp)
        right_master = self._clone_master_problem(parent.rmp)

        left_node = BnBNode(
            node_id=self.nodes_created,
            depth=parent.depth + 1,
            master_problem=left_master,
            routes=list(parent.routes),
            constraints=left_constraints,
            parent_id=parent.node_id,
        )
        left_node.lower_bound = parent.lower_bound
        left_node._lp_basis_cache = getattr(parent, "_lp_basis_cache", None)
        self.nodes_created += 1

        right_node = BnBNode(
            node_id=self.nodes_created,
            depth=parent.depth + 1,
            master_problem=right_master,
            routes=list(parent.routes),
            constraints=right_constraints,
            parent_id=parent.node_id,
        )
        right_node.lower_bound = parent.lower_bound
        right_node._lp_basis_cache = getattr(parent, "_lp_basis_cache", None)
        self.nodes_created += 1

        return left_node, right_node

    def update_global_bounds(self, node_result: NodeSolveResult) -> None:
        """Maintain global LB/UB used for stopping and pruning."""
        if node_result.is_integral and node_result.best_integer_obj is not None:
            if node_result.best_integer_obj < self.global_upper_bound:
                self.global_upper_bound = node_result.best_integer_obj

    def _refresh_global_lower_bound(self) -> None:
        """
        Global LB should come from OPEN nodes (best-bound BnB semantics), not from
        the historical minimum of processed nodes.
        """
        if self.selector.is_empty():
            if math.isfinite(self.global_upper_bound):
                self.global_lower_bound = float(self.global_upper_bound)
            else:
                self.global_lower_bound = float("inf")
            return
        open_lb = float(self.selector.best_lower_bound())
        self.global_lower_bound = open_lb if math.isfinite(open_lb) else float("inf")

    def _gap_percent(self) -> Optional[float]:
        """
        Relative gap in percent: (UB-LB)/|UB| * 100.
        Returns None when UB/LB is not finite.
        """
        lb = self.global_lower_bound
        ub = self.global_upper_bound
        if not (math.isfinite(lb) and math.isfinite(ub)):
            return None
        if abs(ub) < 1e-12:
            return 0.0 if abs(ub - lb) < 1e-12 else None
        return max(0.0, (ub - lb) / abs(ub) * 100.0)

    def _print_node_progress(self, node: BnBNode, node_result: NodeSolveResult) -> None:
        lb = self.global_lower_bound
        ub = self.global_upper_bound
        gap = self._gap_percent()
        stats = getattr(node, "solve_stats", {})

        lb_s = "inf" if not math.isfinite(lb) else f"{lb:.6f}"
        ub_s = "inf" if not math.isfinite(ub) else f"{ub:.6f}"
        gap_s = "n/a" if gap is None else f"{gap:.4f}%"

        print(
            f"[BnB] node={node.node_id} depth={node.depth} status={node_result.status.value} "
            f"node_LB={node_result.lower_bound:.6f} global_LB={lb_s} global_UB={ub_s} gap={gap_s} "
            f"cg_iter={stats.get('cg_iterations', 0)} cols={stats.get('columns_generated', 0)}/{stats.get('columns_added', 0)} "
            f"zero_add_iter={stats.get('zero_add_iterations', 0)} "
            f"rmp_s={float(stats.get('rmp_time_s', 0.0)):.3f} "
            f"(lp={float(stats.get('rmp_lp_time_s', 0.0)):.3f}, cut={float(stats.get('cut_separation_time_s', 0.0)):.3f}) "
            f"pricing_s={float(stats.get('pricing_time_s', 0.0)):.3f} "
            f"addcol_s={float(stats.get('addcol_time_s', 0.0)):.3f} "
            f"labels={int(stats.get('labels_expanded', 0))}/{int(stats.get('labels_generated', 0))} "
            f"backtrack_pruned={int(stats.get('backtrack_pruned', 0))} "
            f"shortcut_returns={int(stats.get('shortcut_returns', 0))} "
            f"sig_filtered={int(stats.get('existing_sig_filtered', 0))} "
            f"coeff_dom_filtered={int(stats.get('coeff_dominated_filtered', 0))} "
            f"skip_dup={int(stats.get('columns_skipped_duplicate', 0))} "
            f"skip_dom={int(stats.get('columns_skipped_dominated', 0))}"
        )
