"""Aggregated Master Problem (A-RMP) for Branch-and-Price.

Implements the aggregation technique from:
  Yao et al. (2021) "The consistent vehicle routing problem considering path
  consistency in a road network." Transportation Research Part B 153, 21–44.
  Section 4.3 Aggregation Technique.

Adaptation for PCARP (Periodic Capacitated Arc Routing Problem):

  원본 논문 (ConVRP_RN):
    λ^s_{pk}  = 1 if path p is used by vehicle k on day s
    λ^s_p     = Σ_k λ^s_{pk}   (aggregated)
    y_{ij}    = Σ_k y_{ijk}    (aggregated arc consistency)

  본 문제 (PCARP):
    λ^{t,k}_r = 1 if route r is used by vehicle k on day t
    λ^t_r     = Σ_k λ^{t,k}_r  (aggregated, k 차원 제거)
    y_e       = Σ_k z_{e,k}    (비필수 엣지 e의 집계된 일관성 변수, y_e ∈ Z+)

A-RMP (Gurobi 이름: 일반 RMP의 lam_* / z_* 와 구분):
  - agg_lam_t{t}_r{j} : 일별 집계 경로 λ^t_r (≈ Σ_k λ^{t,k}_r)
  - agg_y_{i}_{j}     : 비필수 엣지 집계 일관성 (≈ Σ_k z_{e,k})
  - 집계 스케줄 q_{e,p} = Σ_k s_{e,p,k} (차량 인덱스 없음, 패턴 질량)
    분기: whole_route → daily_route → schedule_fix(q_{e,p}) → 개별 agg_lam (lambda_var)

A-RMP 구조:
  - λ^t_r : per-day route variable (vehicle 무관), 컬럼 풀을 day 단위로 공유
  - 집계 커버 제약: Σ_r a_{e,r} λ^t_r = Σ_{p:t∈p} q_{e,p}
  - 차량 한도: Σ_r λ^t_r ≤ |K|  (per day)
  - 집계 할인 링크: Σ_r b_{e,r} λ^t_r ≥ y_e  ∀t  (비필수 엣지 e)
  - Pricing: |T|개 서브문제 (|T|×|K| 대비 K배 감소)

정수해 발견 시 → Disaggregation MILP (Yao 17–25 + SimpleSP 커버 정합):
  - R^t* = {r : λ^t_r = 1} (day t에 선택된 경로 집합)
  - Σ_k λ^{t,k}_r = 1  ∀r ∈ R^t*        (각 경로는 정확히 한 차량에 배정)
  - Σ_r λ^{t,k}_r ≤ 1  ∀k, t            (각 차량은 하루 최대 한 경로)
  - Σ_r a_{e,r} λ^{t,k}_r = Σ_{p:t∈p} σ_{e,p,k},  Σ_k σ_{e,p,k} = q_{e,p}  (disagg MILP)
  - Σ_k z_{e,k} = y_e  ∀ non-req e       (집계 변수 분해)
  - Σ_r b_{e,r} λ^{t,k}_r ≥ z_{e,k}  ∀e,k,t
  - 성공 → global SimpleSPMaster 의 (λ,z,s) 와 정합되는 lift
  - 실패 → switch_to_rmp_mode() → 표준 SimpleSPMaster로 전환 후 재풀기
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from refactor_algorithm.core.master.compare_arc_vs_bnp import (
    SimpleSPMaster,
    _canon_edge,
    discount_objective_cost_per_edge,
    path_arcs_travel_total,
)
from refactor_algorithm.core.master.separation import (
    AggSRI3Separator,
    SeparationManager,
    SeparationRoundResult,
)
from refactor_algorithm.core.pricing.node import BnBConfig, BnBNode, BestBoundSelector, DepthFirstSelector
from refactor_algorithm.core.master.compare_global_rmp_bnp import GlobalRMPBnBTree
from refactor_algorithm.core.util.alns import run_alns_initial_solution
from refactor_algorithm.core.util.initial_heuristic import (
    add_cover_artificials,
    build_q_load_aggregated_initial_columns,
    canon_edge,
)


Edge = Tuple[int, int]


# ---------------------------------------------------------------------------
# 집계 경로 컬럼 (vehicle 인덱스 없음)
# ---------------------------------------------------------------------------

@dataclass
class AggRouteColumn:
    """A-RMP route column: per-day, vehicle-agnostic."""
    day: int
    serviced_required_edges: Tuple[Edge, ...]
    path_arcs: Tuple[Tuple[int, int], ...]
    cost: float


# ---------------------------------------------------------------------------
# AggregatedMaster
# ---------------------------------------------------------------------------

class AggregatedMaster:
    """
    Aggregated Restricted Master Problem (A-RMP).

    BnBNode.solve_node()에서 a_flag=True이면 이 모델을 사용.
    정수해 발견 시 try_disaggregate()로 차량 배정을 시도.
    실패 시 switch_to_rmp_mode()로 표준 SimpleSPMaster로 전환.

    BnBNode 호환 인터페이스:
      .model              → 현재 활성 Gurobi 모델
      .add_pricing_columns()
      .get_pricing_data()
      .get_branching_data()
      .build_executable_solution()
      .register_aggregate_constr()
      .separate_cuts()
    """

    def __init__(self, inst: Dict[str, Any]) -> None:
        self.inst = inst
        self.days: List[int] = list(inst["periods"])
        self.vehicles: List[int] = list(inst["vehicles"])
        self.edges: List[Edge] = list(inst["edges"])
        self.required_edges: List[Edge] = list(inst["required_edges"])
        self.required_edge_set: set = set(self.required_edges)
        # Discount는 실제 sparse graph 엣지(물리적 도로)에만 적용.
        # metric closure의 가상 직결 엣지에 적용하면 discount가 폭발적으로 커짐.
        _sparse_edge_set: set = set(inst.get("arc_sparse_edges", []))
        self.nonrequired_edges: List[Edge] = [
            e for e in self.edges
            if e not in self.required_edge_set and e in _sparse_edge_set
        ] if _sparse_edge_set else [e for e in self.edges if e not in self.required_edge_set]
        self.nonrequired_edge_set: set[Edge] = set(self.nonrequired_edges)
        self.capacity: float = float(inst["capacity"])
        self.depot: int = int(inst["depot"])
        self.travel_cost: Dict[Edge, float] = inst["travel_cost"]
        self.service_extra: Dict[Edge, float] = inst["service_extra"]
        self.schedule_patterns: Dict[Edge, List[frozenset]] = inst["schedule_patterns"]
        self.discount_theta: float = float(inst.get("discount_theta", 0.0))

        # ── A-RMP 상태 ─────────────────────────────────────────────────────
        self._a_flag: bool = True
        self._fallback_rmp: Optional[SimpleSPMaster] = None
        # disaggregation 실패 후 SMP로 막 전환된 뒤, 첫 separate_cuts에서만 depth를 0으로 취급해
        # cut_root_only=1일 때도 컷 분리가 한 번 돌게 함 (깊은 노드에서 전환되는 경우).
        self._pending_smp_separation_after_switch: bool = False

        # A-RMP 모델 내부 저장소
        self._armp_model: gp.Model = gp.Model("armp")
        self._armp_model.Params.OutputFlag = int(inst.get("alg_gurobi_output", inst.get("gurobi_output", 0)))

        # 변수/제약 이름 맵핑 (A-RMP 전용)
        # (required edge e, pattern index p) → Gurobi name; 집계 q_{e,p} (k 없음)
        self.schedule_var_name: Dict[Tuple[Edge, int], str] = {}
        self.agg_cover_name: Dict[Tuple[Edge, int], str] = {}   # (edge, day) → cname
        self.agg_veh_name: Dict[int, str] = {}                  # day → cname
        self.agg_y_var_name: Dict[Edge, str] = {}  # non-req edge → agg_y_* Gurobi name (집계 z)
        self.agg_disc_link_name: Dict[Tuple[Edge, int], str] = {}  # (edge, day) → cname

        # 경로 컬럼 (A-RMP 모드)
        self.agg_route_columns: List[AggRouteColumn] = []
        self.agg_column_signatures: set = set()
        self.agg_column_coeff_best_obj: Dict[Tuple[Any, ...], float] = {}
        self.use_coeff_dominance_filter: bool = bool(int(self.inst.get("use_coeff_dominance_filter", 1)))
        self.coeff_dom_obj_tol: float = abs(float(self.inst.get("coeff_dom_obj_tol", 1e-9)))
        self._last_add_route_status: str = "none"
        self.agg_lambda_var_names_by_day: Dict[int, List[str]] = {t: [] for t in self.days}
        self.agg_lambda_name_to_index: Dict[str, int] = {}
        self._agg_col_tick: int = 0
        self._agg_constr_by_name: Dict[str, Any] = {}
        self._agg_var_by_name: Dict[str, Any] = {}
        self._agg_lambda_alias_expr: Dict[str, Dict[str, float]] = {}
        self._agg_lambda_alias_drivers: Dict[str, set[int]] = {}
        self._fallback_route_exact_var_by_sig: Dict[Tuple[Any, ...], str] = {}
        self._fallback_route_coeff_var_by_sig: Dict[Tuple[Any, ...], str] = {}
        self._fallback_route_lookup_owner_id: Optional[int] = None
        self._fallback_route_lookup_size: int = 0
        self._branch_arc_visit_expr: Dict[Tuple[Any, int], Dict[str, float]] = {}
        self._branch_node_visit_expr: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._branch_ryan_foster_pair_expr: Dict[Tuple[Edge, Edge, int], Dict[str, float]] = {}
        self._schedule_vars_by_edge_day: Dict[Tuple[Edge, int], List[str]] = {}

        # Phase-I 인공변수
        self.artificial_var_name_by_cover: Dict[str, str] = {}

        # 집계 분기 제약 레지스트리 (add_pricing_columns에서 새 컬럼 반영)
        self.aggregate_branch_constrs: Dict[tuple, str] = {}
        self._aggregate_constr_handles: Dict[tuple, Any] = {}

        # 분리 관리자 (없음 – A-RMP에선 생략 가능, 필요 시 추가)
        self.separation_manager = None
        self.capacity_cuts_added: int = 0
        self.sri_cuts_added: int = 0
        self.initial_incumbent: Optional[Dict[str, Any]] = None

        self._build_armp_model()
        self._add_initial_columns()

        # A-RMP SRI-3 분리 관리자 (agg_lam 기반; enable_sri=1 일 때만 활성화)
        self._armp_sri_manager: Optional[SeparationManager] = None
        if bool(int(self.inst.get("enable_sri", self.inst.get("use_sri_cuts", 0)))):
            _sri_card = int(self.inst.get("sri_cardinality", 3))
            _sri_tol = float(self.inst.get("cut_separation_tol", 1e-7))
            self._armp_sri_manager = SeparationManager(
                separators=[AggSRI3Separator(cardinality=_sri_card)],
                tol=_sri_tol,
                max_rounds=1,
            )

    # ── 공개 속성 ────────────────────────────────────────────────────────────

    @property
    def a_flag(self) -> bool:
        return self._a_flag

    @property
    def model(self) -> gp.Model:
        if self._a_flag:
            return self._armp_model
        assert self._fallback_rmp is not None, "Fallback RMP not built yet"
        return self._fallback_rmp.model

    # ── A-RMP 모델 구성 ──────────────────────────────────────────────────────

    def _build_armp_model(self) -> None:
        m = self._armp_model
        discount_weight = self.discount_theta * float(len(self.days))

        # ── 집계 스케줄 q_{e,p} = Σ_k s_{e,p,k} (차량 차원 제거) ─────────────
        q_var: Dict[Tuple[Edge, int], Any] = {}
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            for p_idx, _ in enumerate(pats):
                v = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                             name=f"aq_{e[0]}_{e[1]}_p{p_idx}")
                q_var[(e, p_idx)] = v
        m.update()
        for (e, p_idx), v in q_var.items():
            vname = v.VarName
            self.schedule_var_name[(e, p_idx)] = vname
            e_can = _canon_edge(e[0], e[1])
            for t in self.schedule_patterns[e_can][int(p_idx)]:
                self._schedule_vars_by_edge_day.setdefault((e_can, int(t)), []).append(vname)

        # 각 필수 엣지에 대해 패턴 질량 합 = 1
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            m.addConstr(
                gp.quicksum(q_var[(e, p_idx)] for p_idx in range(len(pats))) == 1.0,
                name=f"sched_{e}",
            )

        # ── 집계 커버 제약 agg_cover_{e}_t{t} ─────────────────────────────
        # Σ_r a_{e,r} λ^t_r - Σ_{p:t∈p} q_{e,p} = 0
        # (λ 항은 컬럼 추가 시 주입)
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            for t in self.days:
                expr = -gp.quicksum(
                    q_var[(e, p_idx)]
                    for p_idx, pat in enumerate(pats) if t in pat
                )
                cname = f"agg_cover_{e[0]}_{e[1]}_t{t}"
                m.addConstr(expr == 0.0, name=cname)
                self.agg_cover_name[(e, int(t))] = cname

        # ── 차량 한도 agg_veh_t{t}: Σ_r λ^t_r ≤ |K| ─────────────────────
        K = float(len(self.vehicles))
        for t in self.days:
            cname = f"agg_veh_t{t}"
            m.addConstr(gp.LinExpr() <= K, name=cname)
            self.agg_veh_name[int(t)] = cname

        # ── y_e 변수 (집계 할인 활성화 변수) ──────────────────────────────
        # y_e = Σ_k z_{e,k}, 목적함수 기여: -discount_weight * c_e * y_e
        y_var: Dict[Edge, Any] = {}
        for e in self.nonrequired_edges:
            c_disc = discount_objective_cost_per_edge(self.inst, e, self.travel_cost)
            obj_coef = -discount_weight * float(c_disc)
            yv = m.addVar(lb=0.0, ub=float(len(self.vehicles)),
                          vtype=GRB.CONTINUOUS, obj=obj_coef,
                          name=f"agg_y_{e[0]}_{e[1]}")
            y_var[e] = yv
        m.update()
        for e, yv in y_var.items():
            self.agg_y_var_name[e] = yv.VarName

        # ── 집계 할인 링크 제약 agg_disc_link_{e}_t{t} ────────────────────
        # Σ_r b_{e,r} λ^t_r ≥ y_e  ∀t  ⇔  y_e - Σ_r b λ ≤ 0 (λ는 컬럼 추가 시 -1로 주입)
        for e in self.nonrequired_edges:
            for t in self.days:
                expr = gp.LinExpr()
                expr += 1.0 * y_var[e]
                cname = f"agg_disc_{e[0]}_{e[1]}_t{t}"
                m.addConstr(expr <= 0.0, name=cname)
                self.agg_disc_link_name[(e, int(t))] = cname

        # ── Phase-I 인공변수 ────────────────────────────────────────────────
        self.artificial_var_name_by_cover = add_cover_artificials(
            model=m,
            cover_constr_names=self.agg_cover_name.values(),
            penalty=1e5,
        )

        m.ModelSense = GRB.MINIMIZE
        m.update()

    # ── 컬럼 추가 (A-RMP 모드) ────────────────────────────────────────────────

    def _column_cost(self, path_arcs: Sequence, serviced_edges: Sequence[Edge]) -> float:
        travel = path_arcs_travel_total(self.inst, path_arcs, self.travel_cost)
        serv = sum(float(self.service_extra[e]) for e in serviced_edges)
        return travel + serv

    def _agg_column_coeff_signature(
        self,
        day: int,
        serviced_edges: Sequence[Edge],
        nonreq_used: Sequence[Edge],
    ) -> Tuple[Any, ...]:
        return (
            int(day),
            tuple(sorted(serviced_edges)),
            tuple(sorted(nonreq_used)),
        )

    @staticmethod
    def _agg_column_services_required_edge(col: AggRouteColumn, e: Edge) -> bool:
        ce = _canon_edge(int(e[0]), int(e[1]))
        for x in col.serviced_required_edges:
            if _canon_edge(int(x[0]), int(x[1])) == ce:
                return True
        return False

    def _lift_schedule_service_on_day(
        self,
        e: Edge,
        t: int,
        armp_solution: Dict[str, float],
    ) -> float:
        """집계 스케줄 기준: Σ_{p: t∈p} q_{e,p} (= 예전 Σ_k Σ_{p:t∈p} s_{e,p,k})."""
        rhs = 0.0
        pats = self.schedule_patterns[e]
        for p_idx, pat in enumerate(pats):
            if t not in pat:
                continue
            qname = self.schedule_var_name.get((e, p_idx))
            if qname is not None:
                rhs += float(armp_solution.get(qname, 0.0))
        return float(rhs)

    def _greedy_disagg_route_assignment(
        self,
        active: Dict[int, List[int]],
        armp_solution: Dict[str, float],
        K: List[int],
        *,
        tol: float = 1e-5,
    ) -> Optional[Dict[Tuple[int, int, int], float]]:
        """
        lift_cover 제약을 만족하는 (t,k,ridx) 배정을 탐욕적으로 찾아 MIP Start 힌트로 쓴다.

        집계 q_{e,p} 만 있으면 차량별 필요일 정보가 없어 탐욕 배정을 쓰지 않는다.
        """
        return None

    def _register_branching_route_var(
        self,
        *,
        day: int,
        vname: str,
        serviced_edges: Sequence[Edge],
        path_arcs: Sequence[Tuple[int, int]],
    ) -> None:
        t_int = int(day)
        arc_cnt: Dict[Edge, float] = {}
        node_cnt: Dict[int, float] = {}
        served_unique = tuple(sorted({_canon_edge(e[0], e[1]) for e in serviced_edges}))

        for arc in path_arcs:
            if not (isinstance(arc, tuple) and len(arc) >= 2):
                continue
            e_can = _canon_edge(arc[0], arc[1])
            arc_cnt[e_can] = arc_cnt.get(e_can, 0.0) + 1.0
            node_id = int(arc[1])
            node_cnt[node_id] = node_cnt.get(node_id, 0.0) + 1.0

        for e_can, coeff in arc_cnt.items():
            expr = self._branch_arc_visit_expr.setdefault((e_can, t_int), {})
            expr[vname] = expr.get(vname, 0.0) + float(coeff)

        for node_id, coeff in node_cnt.items():
            expr = self._branch_node_visit_expr.setdefault((node_id, t_int), {})
            expr[vname] = expr.get(vname, 0.0) + float(coeff)

        for i in range(len(served_unique)):
            e_i = served_unique[i]
            for j in range(i + 1, len(served_unique)):
                e_j = served_unique[j]
                expr = self._branch_ryan_foster_pair_expr.setdefault((e_i, e_j, t_int), {})
                expr[vname] = expr.get(vname, 0.0) + 1.0

    def _add_agg_route_var(
        self,
        day: int,
        serviced_edges: Sequence[Edge],
        path_arcs: Sequence,
        active_aggregate_constrs: Optional[Sequence[Tuple[tuple, Any]]] = None,
    ) -> Optional[Any]:
        """A-RMP에 day 단위 경로 컬럼을 추가. 중복이면 None 반환."""
        self._last_add_route_status = "none"
        nonreq_used: set = set()
        for a in path_arcs:
            if isinstance(a, tuple) and len(a) >= 2:
                ec = _canon_edge(int(a[0]), int(a[1]))
                if ec not in self.required_edge_set:
                    nonreq_used.add(ec)

        coeff_sig = self._agg_column_coeff_signature(day, serviced_edges, tuple(nonreq_used))
        cost = self._column_cost(path_arcs, serviced_edges)
        if self.use_coeff_dominance_filter:
            best_cost = self.agg_column_coeff_best_obj.get(coeff_sig)
            if best_cost is not None and float(cost) + self.coeff_dom_obj_tol >= float(best_cost):
                self._last_add_route_status = "dominated"
                return None

        sig = (int(day), tuple(sorted(serviced_edges)), tuple(path_arcs))
        self._agg_col_tick += 1
        if sig in self.agg_column_signatures:
            self._last_add_route_status = "duplicate"
            return None
        self.agg_column_signatures.add(sig)

        m = self._armp_model
        col = gp.Column()

        # 커버 제약 계수
        for e in serviced_edges:
            cname = self.agg_cover_name[(e, int(day))]
            c = self._get_agg_constr_cached(cname)
            if c is not None:
                col.addTerms(1.0, c)

        # 차량 한도 제약 계수
        veh_c = self._get_agg_constr_cached(self.agg_veh_name[int(day)])
        if veh_c is not None:
            col.addTerms(1.0, veh_c)

        # 할인 링크 계수 (비필수 엣지 사용 여부 binary)
        for e in nonreq_used:
            cname = self.agg_disc_link_name.get((e, int(day)))
            if cname is None:
                continue
            c = self._get_agg_constr_cached(cname)
            if c is None:
                continue
            col.addTerms(-1.0, c)

        if self.aggregate_branch_constrs:
            arc_counts: Dict[Edge, float] = {}
            node_counts: Dict[int, float] = {}
            served_set = set(serviced_edges)
            for arc in path_arcs:
                if isinstance(arc, tuple) and len(arc) >= 2:
                    ec = _canon_edge(arc[0], arc[1])
                    arc_counts[ec] = arc_counts.get(ec, 0.0) + 1.0
                    node_counts[arc[1]] = node_counts.get(arc[1], 0.0) + 1.0

            agg_items = active_aggregate_constrs
            if agg_items is None:
                agg_items = self._active_aggregate_constr_items()

            for agg_key, constr in agg_items:
                kind = agg_key[0]
                coeff = 0.0
                if kind == "whole_route":
                    coeff = 1.0
                elif kind == "daily_route":
                    if agg_key[1] == int(day):
                        coeff = 1.0
                elif kind == "visit_arc":
                    if agg_key[2] == int(day):
                        coeff = float(arc_counts.get(agg_key[1], 0.0))
                elif kind == "visit_node":
                    if len(agg_key) >= 3 and agg_key[2] == int(day):
                        coeff = float(node_counts.get(agg_key[1], 0.0))
                elif kind == "sri3_t":
                    if agg_key[1] == int(day):
                        overlap = sum(1 for e in agg_key[2] if e in served_set)
                        if overlap >= 2:
                            coeff = 1.0
                elif kind == "ryan_foster_pair":
                    if len(agg_key) >= 4 and agg_key[3] == int(day):
                        if agg_key[1] in served_set and agg_key[2] in served_set:
                            coeff = 1.0
                elif kind == "ryan_foster_pair_avg":
                    rf_days = agg_key[3] if len(agg_key) >= 4 else ()
                    if int(day) in rf_days and agg_key[1] in served_set and agg_key[2] in served_set:
                        coeff = 1.0 / float(len(rf_days)) if rf_days else 0.0
                if coeff != 0.0:
                    col.addTerms(coeff, constr)

        ridx = len(self.agg_route_columns)
        vname = f"agg_lam_t{int(day)}_r{ridx}"
        lam = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                       obj=cost, name=vname, column=col)
        self._agg_var_by_name[vname] = lam
        self.agg_lambda_name_to_index[vname] = ridx
        self.agg_lambda_var_names_by_day[int(day)].append(vname)
        self._register_branching_route_var(
            day=int(day),
            vname=vname,
            serviced_edges=serviced_edges,
            path_arcs=path_arcs,
        )
        self.agg_route_columns.append(AggRouteColumn(
            day=int(day),
            serviced_required_edges=tuple(serviced_edges),
            path_arcs=tuple(path_arcs),
            cost=float(cost),
        ))
        old_best = self.agg_column_coeff_best_obj.get(coeff_sig, float("inf"))
        if float(cost) < float(old_best):
            self.agg_column_coeff_best_obj[coeff_sig] = float(cost)
        self._last_add_route_status = "added"
        return lam

    def _active_aggregate_constr_items(self) -> List[Tuple[tuple, Any]]:
        items: List[Tuple[tuple, Any]] = []
        if not self.aggregate_branch_constrs:
            return items
        for agg_key, cname in self.aggregate_branch_constrs.items():
            constr = self._aggregate_constr_handles.get(agg_key)
            if constr is None:
                constr = self._get_agg_constr_cached(str(cname))
            if constr is None:
                self._aggregate_constr_handles.pop(agg_key, None)
                self._agg_constr_by_name.pop(str(cname), None)
                continue
            self._agg_constr_by_name[str(cname)] = constr
            self._aggregate_constr_handles[agg_key] = constr
            items.append((agg_key, constr))
        return items

    def _add_initial_columns(self) -> None:
        """ALNS로 초기 컬럼 생성 (A-RMP: driver 무관, day 단위)."""
        use_alns = bool(int(self.inst.get("use_alns_initialization", 1)))
        if use_alns:
            try:
                alns_out = run_alns_initial_solution(
                    inst=self.inst,
                    iterations=int(self.inst.get("alns_iterations", 300)),
                    destroy_fraction=float(self.inst.get("alns_destroy_fraction", 0.25)),
                    seed=(None if self.inst.get("alns_seed", None) in (None, "", -1)
                          else int(self.inst.get("alns_seed"))),
                )
                pool_cols: List[Dict[str, Any]] = list(
                    alns_out.get("column_pool") or alns_out.get("columns", [])
                )
                seen: set = set()
                for col in pool_cols:
                    day = int(col["day"])
                    served = tuple(tuple(e) for e in col.get("serviced_required_edges", []))
                    arcs = tuple(tuple(a) for a in col.get("path_arcs", []))
                    if not served or not arcs:
                        continue
                    key = (day, served, arcs)
                    if key in seen:
                        continue
                    seen.add(key)
                    self._add_agg_route_var(day, list(served), list(arcs))
                if isinstance(alns_out, dict) and math.isfinite(float(alns_out.get("objective", float("inf")))):
                    self.initial_incumbent = {
                        "objective": float(alns_out["objective"]),
                        "active_routes": list(alns_out.get("active_routes", [])),
                        "source": "alns_initial",
                        "rmp_feasible": bool(alns_out.get("rmp_feasible", True)),
                        "rmp_feasible_detail": str(alns_out.get("rmp_feasible_detail", "")),
                        "alns_column_pool_size": int(len(pool_cols)),
                    }
            except Exception:
                self.initial_incumbent = None

        # q-load seed: pick one schedule pattern per edge, then pack each day's
        # active edges into capacity-feasible route groups.
        try:
            qload_out = build_q_load_aggregated_initial_columns(
                days=self.days,
                required_edges=self.required_edges,
                schedule_patterns=self.schedule_patterns,
                demand=self.inst["demand"],
                capacity=self.capacity,
                num_vehicles=len(self.vehicles),
                depot=self.depot,
                edge_cost=self.travel_cost,
            )
        except Exception:
            qload_out = {"columns": []}
        seen_qload: set = set()
        for col in qload_out.get("columns", []):
            day = int(col["day"])
            served = tuple(tuple(e) for e in col.get("serviced_required_edges", []))
            arcs = tuple(tuple(a) for a in col.get("path_arcs", []))
            if not served or not arcs:
                continue
            key = (day, served, arcs)
            if key in seen_qload:
                continue
            seen_qload.add(key)
            self._add_agg_route_var(day, list(served), list(arcs))

        # Final safety net: 필수 엣지마다 단일 서비스 경로
        if not self.agg_route_columns:
            from refactor_algorithm.core.master.compare_arc_vs_bnp import _single_service_route
            for day in self.days:
                for e in self.required_edges:
                    arcs, _ = _single_service_route(self.depot, e, self.travel_cost)
                    self._add_agg_route_var(day, [e], list(arcs))
        self._armp_model.update()

    # ── SimpleSPMaster 호환 인터페이스 ────────────────────────────────────────

    def add_pricing_columns(self, columns: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        if not self._a_flag:
            assert self._fallback_rmp is not None
            return self._fallback_rmp.add_pricing_columns(columns)

        attempted = added = skipped_dup = skipped_empty = skipped_cap = skipped_dominated = 0
        active_aggregate_constrs = self._active_aggregate_constr_items()
        for c in columns:
            attempted += 1
            day = int(c["day"])
            served = [_canon_edge(*e) if isinstance(e, tuple) and len(e) == 2 else e
                      for e in c.get("serviced_required_edges", [])]
            arcs = [tuple(a) for a in c.get("path_arcs", [])]
            if not served:
                skipped_empty += 1
                continue
            load = sum(float(self.inst["demand"].get(e, 0.0)) for e in served)
            if load > self.capacity + 1e-9:
                skipped_cap += 1
                continue
            v = self._add_agg_route_var(day, served, arcs, active_aggregate_constrs=active_aggregate_constrs)
            if v is None:
                if self._last_add_route_status == "dominated":
                    skipped_dominated += 1
                else:
                    skipped_dup += 1
            else:
                added += 1
        self._armp_model.update()
        return {"attempted": attempted, "added": added,
                "skipped_empty": skipped_empty, "skipped_capacity": skipped_cap,
                "skipped_duplicate": skipped_dup, "skipped_dominated": skipped_dominated}

    def get_pricing_data(self) -> Dict[str, Any]:
        if not self._a_flag:
            assert self._fallback_rmp is not None
            return self._fallback_rmp.get_pricing_data()

        # A-RMP pricing: per-day contexts (driver 없음)
        adjacency: Dict[int, List[Dict[str, Any]]] = {i: [] for i in self.inst["nodes"]}
        for e in self.edges:
            i, j = e
            req = e in self.required_edge_set
            dem = float(self.inst["demand"].get(e, 0.0)) if req else 0.0
            serv = float(self.service_extra[e]) if req else 0.0
            adjacency[i].append({"id": (i, j), "to": j,
                                  "travel_cost": float(self.travel_cost[e]),
                                  "required": req, "required_id": e,
                                  "demand": dem, "service_cost": serv})
            adjacency[j].append({"id": (j, i), "to": i,
                                  "travel_cost": float(self.travel_cost[e]),
                                  "required": req, "required_id": e,
                                  "demand": dem, "service_cost": serv})

        # 스케줄 기반 금지/강제 필터 (A-RMP: driver 무관 합산)
        forbidden_by_day: Dict[int, List[Edge]] = {t: [] for t in self.days}
        for e in self.required_edges:
            for t in self.days:
                can_serve = False
                for vname in self._schedule_vars_by_edge_day.get((e, int(t)), ()):
                    var = self._get_agg_var_cached(vname)
                    if var is not None and float(var.UB) > 0.5:
                        can_serve = True
                        break
                if not can_serve:
                    forbidden_by_day[t].append(e)

        return {
            "master_mode": "aggregated",
            "days": list(self.days),
            "capacity": self.capacity,
            "depot": self.depot,
            "adjacency": adjacency,
            "pricing_method": str(self.inst.get("pricing_method", "labeling")),
            "pricing_ng_size": int(self.inst.get("pricing_ng_size", 8)),
            "cpp_empty_fallback": str(self.inst.get("cpp_empty_fallback", "dp")),
            "cpp_ng_empty_fallback": str(
                self.inst.get(
                    "cpp_ng_empty_fallback",
                    self.inst.get("cpp_empty_fallback", "labeling"),
                )
            ),
            "cut_pricing_mode": str(self.inst.get("cut_pricing_mode", "legacy")),
            "cut_pricing_dual_tol": float(self.inst.get("cut_pricing_dual_tol", 1e-15)),
            "use_schedule_hard_filter": True,
            "max_columns": int(self.inst.get("pricing_max_columns", 0)),
            "use_coeff_dominance_filter": bool(self.use_coeff_dominance_filter),
            "coeff_dom_obj_tol": float(self.coeff_dom_obj_tol),
            # 2-element key → _extract_duals_for_day len==2 분기에서 처리
            "cover_constr_name_by_edge_day": {(e, t): cname
                                               for (e, t), cname in self.agg_cover_name.items()},
            "vehicle_limit_constr_name_by_day": dict(self.agg_veh_name),
            "vehicle_lex_constr_name_by_day": {},
            # 3-element key with None: _extract_duals_for_day의 len>=3 분기 활용
            "discount_link_constr_name_by_edge_day": {(e, t, None): cname
                                                       for (e, t), cname
                                                       in self.agg_disc_link_name.items()},
            "forbidden_required_edges_by_day": {t: forbidden_by_day[t] for t in self.days},
            "existing_column_signatures": self.agg_column_signatures,
            "existing_column_coeff_best_obj": self.agg_column_coeff_best_obj,
            "aggregate_branch_constrs": dict(self.aggregate_branch_constrs),
        }

    def get_branching_data(self, *, include_route_lifting: bool = True) -> Dict[str, Any]:
        if not self._a_flag:
            assert self._fallback_rmp is not None
            data = self._fallback_rmp.get_branching_data(include_route_lifting=include_route_lifting)
            data["aggregate_lambda_alias_expr"] = self._agg_lambda_alias_expr
            return data

        # A-RMP: 집계 스케줄 q_{e,p} 만 (k-인덱스 분기용 표현 없음)
        edge_driver_assign_expr: Dict[Any, Dict[str, float]] = {}
        edge_day_driver_service_expr: Dict[Any, Dict[str, float]] = {}
        arc_visit_expr: Dict[Any, Dict[str, float]] = self._branch_arc_visit_expr if include_route_lifting else {}
        node_visit_expr: Dict[Any, Dict[str, float]] = self._branch_node_visit_expr if include_route_lifting else {}
        schedule_pattern_sum_expr: Dict[Tuple[Edge, int], Dict[str, float]] = {
            key: {vname: 1.0}
            for key, vname in self.schedule_var_name.items()
        }
        return {
            "branching_mode": "armp",
            "lambda_vars_by_day": {t: list(names) for t, names in self.agg_lambda_var_names_by_day.items()},
            "schedule_vars": self.schedule_var_name,
            "schedule_vars_by_edge_day": self._schedule_vars_by_edge_day,
            "schedule_pattern_sum_expr": schedule_pattern_sum_expr,
            "edge_driver_assign_expr": edge_driver_assign_expr,
            "edge_day_driver_service_expr": edge_day_driver_service_expr,
            "node_visit_expr": node_visit_expr,
            "arc_visit_expr": arc_visit_expr,
            "ryan_foster_pair_expr": self._branch_ryan_foster_pair_expr,
            "enable_aggregate_lambda_branching": True,
            "enable_expression_branching": False,  # A-RMP은 집계 변수 기반 분기
        }

    def register_aggregate_constr(self, key: tuple, cname: str) -> None:
        if not self._a_flag:
            if self._fallback_rmp is not None:
                self._fallback_rmp.register_aggregate_constr(key, cname)
            return
        self.aggregate_branch_constrs[key] = cname
        self._agg_constr_by_name.pop(str(cname), None)
        self._aggregate_constr_handles[key] = self._get_agg_constr_cached(str(cname))

    def _get_agg_constr_cached(self, cname: str) -> Optional[Any]:
        c = self._agg_constr_by_name.get(str(cname))
        if c is not None:
            return c
        c = self._armp_model.getConstrByName(str(cname))
        if c is None:
            self._armp_model.update()
            c = self._armp_model.getConstrByName(str(cname))
        if c is not None:
            self._agg_constr_by_name[str(cname)] = c
        return c

    def _get_agg_var_cached(self, vname: str) -> Optional[Any]:
        if not vname:
            return None
        v = self._agg_var_by_name.get(str(vname))
        if v is not None:
            return v
        v = self._armp_model.getVarByName(str(vname))
        if v is None:
            self._armp_model.update()
            v = self._armp_model.getVarByName(str(vname))
        if v is not None:
            self._agg_var_by_name[str(vname)] = v
        return v

    def separate_cuts(self, node_depth: Optional[int] = None, node_id: Optional[int] = None) -> int:
        if not self._a_flag and self._fallback_rmp is not None:
            eff_depth = node_depth
            if self._pending_smp_separation_after_switch:
                eff_depth = 0
                self._pending_smp_separation_after_switch = False
            return int(
                self._fallback_rmp.separate_cuts(node_depth=eff_depth, node_id=node_id)
            )
        # A-RMP uses aggregated λ^t_r (no per-(t,k) split): existing separators target SimpleSPMaster.
        return 0

    def separate_sri_cuts(
        self,
        node_depth: Optional[int] = None,
        node_id: Optional[int] = None,
        optimize_before: bool = False,
    ) -> SeparationRoundResult:
        if not self._a_flag and self._fallback_rmp is not None:
            eff_depth = node_depth
            if self._pending_smp_separation_after_switch:
                eff_depth = 0
                self._pending_smp_separation_after_switch = False
            res = self._fallback_rmp.separate_sri_cuts(
                node_depth=eff_depth,
                node_id=node_id,
                optimize_before=optimize_before,
            )
            self.sri_cuts_added += int(res.added_count)
            return res

        # A-RMP 모드: 집계 SRI-3 컷 (Σ_{r: overlap≥2} agg_lam^t_r ≤ 1)
        # 이 컷은 per-vehicle SRI-3 (SimpleSPMaster)를 k에 대해 합산한 것과 동등하며,
        # A-RMP LP에서 위반될 수 있고 임의의 정수 가능해에서도 유효하다.
        if self._armp_sri_manager is None:
            return SeparationRoundResult()
        if not bool(int(self.inst.get("enable_sri", self.inst.get("use_sri_cuts", 0)))):
            return SeparationRoundResult()
        if bool(int(self.inst.get("root_only_sri", 1))) and node_depth is not None and int(node_depth) > 0:
            return SeparationRoundResult()
        res = self._armp_sri_manager.separate_once(self, optimize_before=optimize_before)
        self.sri_cuts_added += int(res.added_count)
        return res

    def build_executable_solution(
        self,
        values: Dict[str, float],
        source: str = "bnb",
    ) -> Dict[str, Any]:
        if not self._a_flag:
            assert self._fallback_rmp is not None
            return self._fallback_rmp.build_executable_solution(values, source)

        eps = 1e-6
        routes = []
        for vname, ridx in self.agg_lambda_name_to_index.items():
            x = float(values.get(vname, 0.0))
            if x <= eps:
                continue
            col = self.agg_route_columns[ridx]
            routes.append({
                "var": vname,
                "value": x,
                "day": col.day,
                "driver": None,  # A-RMP: driver unknown until disaggregation
                "serviced_required_edges": [tuple(e) for e in col.serviced_required_edges],
                "path_arcs": [tuple(a) for a in col.path_arcs],
                "cost": float(col.cost),
            })

        disagg = getattr(self, "_last_disagg_result", None)
        if disagg is not None:
            return {
                "routes": routes,
                "disaggregated": disagg,
                "objective": float(values.get("__armp_obj__", 0.0)),
                "source": "armp_disaggregated",
            }
        return {
            "routes": routes,
            "objective": float(sum(r["cost"] * r["value"] for r in routes)),
            "source": source,
        }

    # ── Disaggregation ───────────────────────────────────────────────────────

    def reconcile_disaggregated_incumbent_objective(
        self,
        armp_vals: Dict[str, float],
        disagg: Dict[str, Any],
    ) -> float:
        """
        A-RMP와 동일한 목적 구조: Σ (경로비·λ) − θ|T| Σ_e c_e y_e.

        경로 항은 disaggregation 배정 (t,k,r)으로 합산; y_e는 A-RMP 스냅샷(armp_vals)에서 읽는다.
        집계 LP ObjVal이 완화/듀얼 오차로 과소평가될 때 상한 보정용.
        """
        discount_weight = float(self.discount_theta) * float(len(self.days))
        sum_route = 0.0
        assign = disagg.get("assignments") or {}
        cols = self.agg_route_columns
        for key, xv in assign.items():
            if float(xv) <= 0.5:
                continue
            if not (isinstance(key, tuple) and len(key) >= 3):
                continue
            _t, _k, ridx = int(key[0]), int(key[1]), int(key[2])
            if 0 <= ridx < len(cols):
                sum_route += float(cols[ridx].cost)

        disc_weighted = 0.0
        for e in self.nonrequired_edges:
            yn = self.agg_y_var_name.get(e)
            if not yn:
                continue
            ye = float(armp_vals.get(yn, 0.0))
            c_disc = float(discount_objective_cost_per_edge(self.inst, e, self.travel_cost))
            disc_weighted += discount_weight * c_disc * ye

        return float(sum_route - disc_weighted)

    def try_disaggregate(self, armp_solution: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Disaggregation MILP (Yao 17–22 + SimpleSP per-(e,t,k) 커버 정합).

        A-RMP 정수해 {λ^t_r ∈ {0,1}}, {y_e}, {q_{e,p}}가 주어졌을 때,
        경로→차량 배정 λ^{t,k}_r, 분해 σ_{e,p,k} (Σ_k σ=q), z_{e,k} 를 찾는다.
        lift: Σ_r a λ^{t,k}_r = Σ_{p:t∈p} σ_{e,p,k} (global RMP 커버와 동치).

        배정이 정수가 아니거나 MILP 불가능이면 None 반환 (→ switch_to_rmp_mode).
        """
        eps = 1e-6

        # 활성 경로 수집 (λ^t_r > 0.5)
        active: Dict[int, List[int]] = {t: [] for t in self.days}
        for ridx, col in enumerate(self.agg_route_columns):
            vname = f"agg_lam_t{col.day}_r{ridx}"
            if float(armp_solution.get(vname, 0.0)) > 0.5:
                active[col.day].append(ridx)

        # y_e 값 수집
        y_vals: Dict[Edge, float] = {}
        for e in self.nonrequired_edges:
            yname = self.agg_y_var_name.get(e)
            if yname:
                y_vals[e] = float(armp_solution.get(yname, 0.0))

        # 활성 경로가 없으면 (공실 해) → 모든 날 빈 배정으로 성공
        total_active = sum(len(v) for v in active.values())
        if total_active == 0:
            self._last_disagg_result = {"assignments": {}, "z_assignments": {},
                                         "source": "disaggregation_trivial"}
            return self._last_disagg_result

        K = list(self.vehicles)

        # ── Disaggregation MILP ───────────────────────────────────────────
        dm = gp.Model("disaggregate")
        dm.Params.OutputFlag = 0
        dm.Params.MIPGap = float(self.inst.get("disagg_mip_gap", 0.0))
        dm.Params.TimeLimit = max(1.0, float(self.inst.get("disagg_time_limit_s", 120.0)))
        if bool(int(self.inst.get("disagg_mip_focus_feasibility", 1))):
            dm.Params.MIPFocus = 1  # 가능해 우선

        # λ^{t,k}_r ∈ [0,1] 연속, φ^{t,k}_r ∈ {0,1}
        lam: Dict[Tuple, Any] = {}
        phi: Dict[Tuple, Any] = {}
        for t in self.days:
            for ridx in active[t]:
                for k in K:
                    lam[t, k, ridx] = dm.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                                  name=f"lam_t{t}_k{k}_r{ridx}")
                    phi[t, k, ridx] = dm.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                                  name=f"phi_t{t}_k{k}_r{ridx}")

        # z_{e,k} ∈ [0, y_e]
        z_var: Dict[Tuple, Any] = {}
        for e in self.nonrequired_edges:
            ye = float(y_vals.get(e, 0.0))
            if ye < eps:
                continue
            for k in K:
                z_var[e, k] = dm.addVar(lb=0.0, ub=ye, vtype=GRB.CONTINUOUS,
                                         name=f"z_{e[0]}_{e[1]}_k{k}")

        # σ_{e,p,k}: A-RMP 스냅샷 q_{e,p} 를 차량별로 분해
        sigma: Dict[Tuple[Any, int, int], Any] = {}
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            for p_idx in range(len(pats)):
                qname = self.schedule_var_name.get((e, p_idx))
                q_snap = float(armp_solution.get(qname, 0.0)) if qname else 0.0
                for k in K:
                    sigma[(e, p_idx, int(k))] = dm.addVar(
                        lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                        name=f"sig_e{e[0]}_{e[1]}_p{p_idx}_k{k}",
                    )
                dm.addConstr(
                    gp.quicksum(sigma[(e, p_idx, int(k))] for k in K) == q_snap,
                    name=f"sig_sum_e{e[0]}_{e[1]}_p{p_idx}",
                )
        for e in self.required_edges:
            n_pat = len(self.schedule_patterns[e])
            for k in K:
                dm.addConstr(
                    gp.quicksum(sigma[(e, p_idx, int(k))] for p_idx in range(n_pat)) <= 1.0,
                    name=f"sig_row_e{e[0]}_{e[1]}_k{k}",
                )

        dm.update()
        dm.setObjective(gp.quicksum(phi.values()), GRB.MINIMIZE)

        # (18) Σ_k λ^{t,k}_r = 1  ∀ r∈R^t*, t
        for t in self.days:
            for ridx in active[t]:
                dm.addConstr(
                    gp.quicksum(lam[t, k, ridx] for k in K) == 1.0,
                    name=f"assign_t{t}_r{ridx}",
                )

        # (19) Σ_r λ^{t,k}_r ≤ 1  ∀ k, t
        for t in self.days:
            for k in K:
                dm.addConstr(
                    gp.quicksum(lam[t, k, ridx] for ridx in active[t]) <= 1.0,
                    name=f"veh_t{t}_k{k}",
                )

        # (23) Σ_r a λ^{t,k}_r = Σ_{p:t∈p} σ_{e,p,k}
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            for t in self.days:
                for k in K:
                    rhs_e = gp.LinExpr()
                    for p_idx, pat in enumerate(pats):
                        if int(t) not in pat:
                            continue
                        rhs_e += sigma[(e, p_idx, int(k))]
                    lhs_terms = []
                    for ridx in active[t]:
                        col = self.agg_route_columns[ridx]
                        if int(col.day) != int(t):
                            continue
                        if not self._agg_column_services_required_edge(col, e):
                            continue
                        lhs_terms.append(lam[t, k, ridx])
                    if lhs_terms:
                        dm.addConstr(
                            gp.quicksum(lhs_terms) == rhs_e,
                            name=f"lift_cover_e{e[0]}_{e[1]}_t{t}_k{k}",
                        )
                    else:
                        dm.addConstr(rhs_e == 0.0, name=f"lift_cover_e{e[0]}_{e[1]}_t{t}_k{k}")

        # (20) Σ_k z_{e,k} = y_e  ∀ non-req e
        for e in self.nonrequired_edges:
            ye = float(y_vals.get(e, 0.0))
            if ye < eps:
                continue
            dm.addConstr(
                gp.quicksum(z_var[e, k] for k in K) == ye,
                name=f"y_split_{e[0]}_{e[1]}",
            )

        # (21) Σ_r b_{e,r} λ^{t,k}_r ≥ z_{e,k}  ∀ non-req e, k, t
        for e in self.nonrequired_edges:
            ye = float(y_vals.get(e, 0.0))
            if ye < eps:
                continue
            for k in K:
                for t in self.days:
                    b_terms = []
                    for ridx in active[t]:
                        col = self.agg_route_columns[ridx]
                        cnt = sum(
                            1 for a in col.path_arcs
                            if isinstance(a, tuple) and len(a) >= 2
                            and _canon_edge(int(a[0]), int(a[1])) == e
                        )
                        if cnt > 0:
                            b_terms.append((float(cnt), lam[t, k, ridx]))
                    if b_terms:
                        dm.addConstr(
                            gp.quicksum(c * v for c, v in b_terms) >= z_var[e, k],
                            name=f"disc_t{t}_k{k}_e{e[0]}_{e[1]}",
                        )

        # (22) φ ≥ λ
        for key, phi_v in phi.items():
            dm.addConstr(phi_v >= lam[key], name=f"ceil_{key}")

        # MIP Start: 스케줄 정합 탐욕 배정 (실패 시 힌트 없이 풀이)
        if bool(int(self.inst.get("disagg_greedy_warm_start", 1))):
            warm = self._greedy_disagg_route_assignment(active, armp_solution, K)
            if warm is not None:
                for (t0, k0, ridx), val in warm.items():
                    key = (t0, k0, ridx)
                    if key in lam:
                        lam[key].Start = float(val)
                for key, phi_v in phi.items():
                    phi_v.Start = 1.0 if float(lam[key].Start) > 0.5 else 0.0

        dm.update()
        dm.optimize()

        if dm.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or dm.SolCount == 0:
            return None

        # 정수성 확인: 모든 λ^{t,k}_r ∈ {0,1}이어야 성공
        for var in lam.values():
            v = float(var.X)
            if eps < v < 1.0 - eps:
                return None  # 비정수 배정 → 실패

        # 성공: 배정 정보 수집
        assignments: Dict[Tuple, float] = {}
        for (t, k, ridx), var in lam.items():
            if float(var.X) > 0.5:
                assignments[t, k, ridx] = 1.0

        z_assignments: Dict[Tuple, float] = {}
        for (e, k), var in z_var.items():
            v = float(var.X)
            if v > eps:
                z_assignments[e, k] = v

        sigma_assignments: Dict[Tuple[Any, int, int], float] = {}
        for key, var in sigma.items():
            vx = float(var.X)
            if vx > eps:
                sigma_assignments[key] = vx

        result = {
            "assignments": assignments,
            "z_assignments": z_assignments,
            "sigma_assignments": sigma_assignments,
            "source": "disaggregation",
            "agg_route_columns": self.agg_route_columns,
        }
        self._last_disagg_result = result
        return result

    def verify_disaggregation_lift_vs_global_rmp_cover(
        self,
        armp_solution: Dict[str, float],
        disagg: Dict[str, Any],
        tol: float = 1e-4,
    ) -> Tuple[bool, List[str]]:
        """
        disaggregation 배정이 lift 제약
        ``Σ_r a_{e,r} λ^{t,k}_r = Σ_{p:t∈p} σ_{e,p,k}`` 와 일치하는지 검사
        (σ 는 disagg 결과 ``sigma_assignments``).
        """
        errs: List[str] = []
        assignments = disagg.get("assignments") or {}
        lam_on: Dict[Tuple[int, int, int], float] = {}
        for key, xv in assignments.items():
            if float(xv) < 0.5:
                continue
            if not (isinstance(key, tuple) and len(key) >= 3):
                continue
            t, k, ridx = int(key[0]), int(key[1]), int(key[2])
            lam_on[(t, k, ridx)] = 1.0

        K = list(self.vehicles)
        sig = disagg.get("sigma_assignments") or {}
        for e in self.required_edges:
            pats = self.schedule_patterns[e]
            for t in self.days:
                for k in K:
                    rhs = 0.0
                    for p_idx, pat in enumerate(pats):
                        if int(t) not in pat:
                            continue
                        rhs += float(sig.get((e, p_idx, int(k)), 0.0))
                    lhs = 0.0
                    for ridx, col in enumerate(self.agg_route_columns):
                        if int(col.day) != int(t):
                            continue
                        if not self._agg_column_services_required_edge(col, e):
                            continue
                        lhs += float(lam_on.get((t, k, ridx), 0.0))
                    if abs(lhs - rhs) > tol:
                        errs.append(
                            f"lift_cover mismatch e={e} t={t} k={k} lhs_lambda_sum={lhs} rhs_from_sigma={rhs}"
                        )
        return (len(errs) == 0, errs)

    def try_disaggregate_verified(
        self,
        armp_solution: Dict[str, float],
        tol: float = 1e-4,
    ) -> Optional[Dict[str, Any]]:
        """
        try_disaggregate 후 global RMP 커버 정합을 숫자로 재검증.
        MILP 수치 오차로 verify_only 실패 시 None 을 반환해 표준 RMP 전환으로 넘긴다.
        """
        d = self.try_disaggregate(armp_solution)
        if d is None:
            return None
        ok, _errs = self.verify_disaggregation_lift_vs_global_rmp_cover(
            armp_solution, d, tol=tol
        )
        if not ok:
            self._last_disagg_result = None
            return None
        return d

    # ── 표준 RMP로 전환 ──────────────────────────────────────────────────────

    @staticmethod
    def _find_fallback_var_for_col(
        fallback: "SimpleSPMaster",
        day: int,
        driver: int,
        serviced_edges: Sequence[Edge],
        path_arcs: Sequence[Tuple[int, int]],
    ) -> Optional[str]:
        """_add_route_var가 None을 반환(duplicate/dominated)했을 때 fallback RMP에
        이미 존재하는 동등한 변수의 이름을 찾아 반환한다.

        탐색 순서:
          1. Exact match: (day, driver, sorted_edges, path_arcs) 가 완전히 일치하는 컬럼
          2. Coeff-sig match: 같은 (day, driver, sorted_edges, sorted_nonreq_used)를
             가진 컬럼 — dominated 케이스에서 더 저렴한 대체 변수를 반환.
        """
        t, k = int(day), int(driver)
        sorted_edges = tuple(sorted(_canon_edge(int(e[0]), int(e[1])) for e in serviced_edges))
        path_tup = tuple(
            (int(a[0]), int(a[1])) for a in path_arcs
            if isinstance(a, (tuple, list)) and len(a) >= 2
        )

        # nonreq_used 계산 (SimpleSPMaster._add_route_var 와 동일한 로직)
        req_set = fallback.required_edge_set
        nonreq_set = fallback.nonrequired_edge_set
        nonreq_target: frozenset = frozenset(
            _canon_edge(int(a[0]), int(a[1]))
            for a in path_arcs
            if isinstance(a, (tuple, list)) and len(a) >= 2
            and _canon_edge(int(a[0]), int(a[1])) not in req_set
            and (not nonreq_set or _canon_edge(int(a[0]), int(a[1])) in nonreq_set)
        )

        coeff_match_vname: Optional[str] = None
        for vname in fallback.lambda_var_names_by_day.get((t, k), []):
            ridx = fallback.lambda_var_name_to_index.get(vname)
            if ridx is None or ridx < 0 or ridx >= len(fallback.route_columns):
                continue
            rc = fallback.route_columns[ridx]
            if int(rc.day) != t or int(rc.driver) != k:
                continue
            rc_sorted = tuple(
                sorted(_canon_edge(int(e[0]), int(e[1])) for e in rc.serviced_required_edges)
            )
            if rc_sorted != sorted_edges:
                continue
            # (1) Exact match
            if tuple(rc.path_arcs) == path_tup:
                return vname
            # (2) Coeff-sig match (dominated) — 첫 번째 발견값을 기억
            if coeff_match_vname is None:
                rc_nonreq = frozenset(rc.nonrequired_edges_used)
                if rc_nonreq == nonreq_target:
                    coeff_match_vname = vname

        return coeff_match_vname

    @staticmethod
    def _fallback_exact_signature(
        day: int,
        driver: int,
        serviced_edges: Sequence[Edge],
        path_arcs: Sequence[Tuple[int, int]],
    ) -> Tuple[Any, ...]:
        return (
            int(day),
            int(driver),
            tuple(sorted(_canon_edge(int(e[0]), int(e[1])) for e in serviced_edges)),
            tuple(
                (int(a[0]), int(a[1]))
                for a in path_arcs
                if isinstance(a, (tuple, list)) and len(a) >= 2
            ),
        )

    @staticmethod
    def _fallback_coeff_signature(
        fallback: "SimpleSPMaster",
        day: int,
        driver: int,
        serviced_edges: Sequence[Edge],
        path_arcs: Sequence[Tuple[int, int]],
    ) -> Tuple[Any, ...]:
        req_set = fallback.required_edge_set
        nonreq_set = fallback.nonrequired_edge_set
        nonreq_target = tuple(
            sorted(
                {
                    _canon_edge(int(a[0]), int(a[1]))
                    for a in path_arcs
                    if isinstance(a, (tuple, list))
                    and len(a) >= 2
                    and _canon_edge(int(a[0]), int(a[1])) not in req_set
                    and (not nonreq_set or _canon_edge(int(a[0]), int(a[1])) in nonreq_set)
                }
            )
        )
        return (
            int(day),
            int(driver),
            tuple(sorted(_canon_edge(int(e[0]), int(e[1])) for e in serviced_edges)),
            nonreq_target,
        )

    def _reset_fallback_route_lookup(self) -> None:
        self._fallback_route_exact_var_by_sig = {}
        self._fallback_route_coeff_var_by_sig = {}
        self._fallback_route_lookup_owner_id = None
        self._fallback_route_lookup_size = 0

    def _remember_fallback_route_var(
        self,
        fallback: "SimpleSPMaster",
        vname: str,
        *,
        ridx: Optional[int] = None,
    ) -> None:
        if ridx is None:
            ridx = fallback.lambda_var_name_to_index.get(vname)
        if ridx is None or ridx < 0 or ridx >= len(fallback.route_columns):
            return
        rc = fallback.route_columns[ridx]
        exact_sig = self._fallback_exact_signature(
            rc.day,
            rc.driver,
            rc.serviced_required_edges,
            rc.path_arcs,
        )
        coeff_sig = (
            int(rc.day),
            int(rc.driver),
            tuple(sorted(_canon_edge(int(e[0]), int(e[1])) for e in rc.serviced_required_edges)),
            tuple(sorted(rc.nonrequired_edges_used)),
        )
        self._fallback_route_exact_var_by_sig.setdefault(exact_sig, vname)
        self._fallback_route_coeff_var_by_sig.setdefault(coeff_sig, vname)

    def _sync_fallback_route_lookup(self, fallback: "SimpleSPMaster") -> None:
        owner_id = id(fallback)
        if self._fallback_route_lookup_owner_id != owner_id:
            self._reset_fallback_route_lookup()
            self._fallback_route_lookup_owner_id = owner_id
        start = int(self._fallback_route_lookup_size)
        total = len(fallback.route_columns)
        if start > total:
            self._reset_fallback_route_lookup()
            self._fallback_route_lookup_owner_id = owner_id
            start = 0
        for ridx in range(start, total):
            rc = fallback.route_columns[ridx]
            vname = f"lam_t{int(rc.day)}_k{int(rc.driver)}_r{int(ridx)}"
            self._remember_fallback_route_var(fallback, vname, ridx=ridx)
        self._fallback_route_lookup_size = total

    def _find_fallback_var_for_col_cached(
        self,
        fallback: "SimpleSPMaster",
        day: int,
        driver: int,
        serviced_edges: Sequence[Edge],
        path_arcs: Sequence[Tuple[int, int]],
    ) -> Optional[str]:
        self._sync_fallback_route_lookup(fallback)
        exact_sig = self._fallback_exact_signature(day, driver, serviced_edges, path_arcs)
        vname = self._fallback_route_exact_var_by_sig.get(exact_sig)
        if vname is not None:
            return vname
        coeff_sig = self._fallback_coeff_signature(
            fallback,
            day,
            driver,
            serviced_edges,
            path_arcs,
        )
        return self._fallback_route_coeff_var_by_sig.get(coeff_sig)

    def switch_to_rmp_mode(
        self,
        armp_solution: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        A_FLAG=0: 표준 SimpleSPMaster로 전환.

        기본(inst ``armp_switch_replicate_used_lambda_only`` = 1, 기본값):
          마지막 A-RMP LP/정수해에서 ``agg_lam_t{t}_r{j}`` 값이 임계값을 넘은
          컬럼만 SimpleSP로 복제한다. (미사용 집계 컬럼은 넘기지 않음)

        ``armp_solution`` 이 없거나 위 플래그가 0이면:
          예전처럼 ``agg_route_columns`` 전체를 모든 driver에 복제한다.

        각 복제 컬럼은 (day, driver)별 ``lam_t{t}_k{k}_r…`` 로 들어간다.
        """
        if not self._a_flag:
            return  # 이미 전환됨

        fallback_inst = dict(self.inst)
        fallback_inst["use_alns_initialization"] = 0  # 컬럼 재생성 방지
        fallback_inst["alns_replicate_all_contexts"] = 0

        # Reuse existing fallback if available (preserves previously generated columns).
        # A new SimpleSPMaster is created only on the first switch.
        if self._fallback_rmp is not None:
            fallback = self._fallback_rmp
        else:
            fallback = SimpleSPMaster(fallback_inst)
            self._agg_lambda_alias_expr = {}
            self._agg_lambda_alias_drivers = {}
            self._reset_fallback_route_lookup()

        used_only = bool(int(self.inst.get("armp_switch_replicate_used_lambda_only", 1)))
        lam_eps = float(self.inst.get("armp_used_lambda_eps", 1e-6))

        cols: List[Tuple[int, AggRouteColumn]] = list(enumerate(self.agg_route_columns))
        if used_only and isinstance(armp_solution, dict) and armp_solution:
            picked: List[Tuple[int, AggRouteColumn]] = []
            for ridx, col in enumerate(self.agg_route_columns):
                vname = f"agg_lam_t{int(col.day)}_r{int(ridx)}"
                if float(armp_solution.get(vname, 0.0)) > lam_eps:
                    picked.append((ridx, col))
            if picked:
                cols = picked

        self._sync_fallback_route_lookup(fallback)
        # A-RMP 컬럼 → SimpleSPMaster: (day, driver)별 lam
        for agg_ridx, col in cols:
            agg_vname = f"agg_lam_t{int(col.day)}_r{int(agg_ridx)}"
            alias_expr = self._agg_lambda_alias_expr.setdefault(agg_vname, {})
            mapped_drivers = self._agg_lambda_alias_drivers.setdefault(agg_vname, set())
            if len(mapped_drivers) >= len(self.vehicles):
                continue
            for driver in self.vehicles:
                driver_int = int(driver)
                if driver_int in mapped_drivers:
                    continue
                pre_idx = len(fallback.route_columns)
                fallback_vname = f"lam_t{int(col.day)}_k{driver_int}_r{int(pre_idx)}"
                v = fallback._add_route_var(
                    day=col.day,
                    driver=driver_int,
                    serviced_edges=list(col.serviced_required_edges),
                    path_arcs=list(col.path_arcs),
                )
                if v is not None:
                    alias_expr[fallback_vname] = alias_expr.get(fallback_vname, 0.0) + 1.0
                    mapped_drivers.add(driver_int)
                    self._remember_fallback_route_var(fallback, fallback_vname, ridx=pre_idx)
                elif fallback._last_add_route_status in ("duplicate", "dominated"):
                    # _add_route_var가 거부했지만 fallback에는 동등한/더 좋은 변수가
                    # 이미 존재한다. 분기 제약이 전환 후 소실되지 않도록 그 변수를 alias에 등록.
                    existing = self._find_fallback_var_for_col_cached(
                        fallback, col.day, driver_int,
                        col.serviced_required_edges, col.path_arcs,
                    )
                    if existing is not None:
                        alias_expr[existing] = alias_expr.get(existing, 0.0) + 1.0
                        mapped_drivers.add(driver_int)
        fallback.model.update()
        self._fallback_route_lookup_size = len(fallback.route_columns)

        self._fallback_rmp = fallback
        self._a_flag = False
        if bool(int(self.inst.get("separate_cuts_as_root_after_armp_switch", 1))):
            self._pending_smp_separation_after_switch = True


# ---------------------------------------------------------------------------
# solve_with_aggregated_algorithm
# ---------------------------------------------------------------------------

def solve_with_aggregated_algorithm(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    A-RMP 기반 Branch-and-Price 알고리즘.

    inst["use_aggregation"] = True 일 때 AggregatedMaster 사용.
    정수해 발견 시 disaggregation, 실패 시 SimpleSPMaster로 자동 전환.
    """
    rmp = AggregatedMaster(inst)
    root = BnBNode(node_id=0, depth=0, master_problem=rmp)

    require_proof = bool(inst.get("require_proof_optimality", False))
    max_cg_iter = int(inst.get("max_cg_iterations_per_node", 80))
    max_nodes = int(inst.get("max_nodes", 200))
    max_time_s = float(inst.get("algorithm_time_limit_s", 0.0))

    strategy = str(inst.get("node_search_strategy", "dfs")).lower()
    selector = BestBoundSelector() if strategy == "best_bound" else DepthFirstSelector()

    eps_rc = float(inst.get("eps_reduced_cost", 1e-4))
    use_stab = bool(int(inst.get("use_dual_stabilization", 0)))
    stab_alpha = float(inst.get("dual_stab_alpha", 0.5))
    stab_alpha_decay = float(inst.get("dual_stab_alpha_decay", 0.9))
    stab_min_alpha = float(inst.get("dual_stab_min_alpha", 0.0))
    use_ub_zero = bool(int(inst.get("use_ub_zero_branching", 0)))
    partial_pricing_ratio = float(inst.get("partial_pricing_ratio", 1.0))
    phase1_col_cap = int(inst.get("phase1_col_cap", 3))
    enable_sri = bool(int(inst.get("enable_sri", inst.get("use_sri_cuts", 0))))
    root_only_sri = bool(int(inst.get("root_only_sri", 1)))
    max_sri_rounds = int(inst.get("max_sri_rounds", 3))

    config = BnBConfig(
        eps_integrality=1e-6,
        eps_reduced_cost=eps_rc,
        max_cg_iterations_per_node=max_cg_iter,
        max_nodes=max_nodes,
        max_time_s=(max_time_s if max_time_s > 0 else None),
        verbose=bool(inst.get("bnb_log", False)),
        use_dual_stabilization=use_stab,
        dual_stab_alpha=stab_alpha,
        dual_stab_alpha_decay=stab_alpha_decay,
        dual_stab_min_alpha=stab_min_alpha,
        use_ub_zero_branching=use_ub_zero,
        partial_pricing_ratio=partial_pricing_ratio,
        phase1_col_cap=phase1_col_cap,
        enable_sri=enable_sri,
        root_only_sri=root_only_sri,
        max_sri_rounds=max_sri_rounds,
    )

    tree = GlobalRMPBnBTree(root_node=root, config=config, selector=selector)
    best_sol = tree.solve()

    lp_gap_pct = None
    global_lb = float(tree.global_lower_bound)
    global_ub = float(tree.global_upper_bound)
    if math.isfinite(global_ub) and abs(global_ub) > 1e-12 and math.isfinite(global_lb):
        lp_gap_pct = max(0.0, (global_ub - global_lb) / abs(global_ub) * 100.0)

    art_sum = 0.0
    try:
        art_sum = float(root.solve_stats.get("artificial_sum_at_integral", 0.0))
    except Exception:
        pass

    inc_sol = None
    inc_obj = None
    if best_sol is not None:
        inc_obj = float(best_sol.get("objective", float("nan")))
        inc_sol = best_sol

    return {
        "objective": global_ub if math.isfinite(global_ub) else float("nan"),
        "mode": "aggregated_bnp",
        "nodes_processed": int(tree.nodes_processed),
        "artificial_sum": art_sum,
        "hit_node_limit": bool(tree.terminated_by_node_limit),
        "hit_time_limit": bool(tree.terminated_by_time_limit),
        "hit_cg_limit": bool(tree.profile.get("nodes_hit_cg_limit", 0) > 0),
        "gap_pct": lp_gap_pct,
        "profile": dict(tree.profile),
        "root_incumbent": (float(rmp.initial_incumbent["objective"])
                           if hasattr(rmp, "initial_incumbent") and rmp.initial_incumbent else None),
        "incumbent_objective": inc_obj,
        "incumbent_solution": inc_sol,
    }
