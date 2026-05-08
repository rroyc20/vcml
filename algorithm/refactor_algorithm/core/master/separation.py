from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gurobipy as gp


@dataclass(frozen=True)
class SeparatedCut:
    key: Tuple[Any, ...]
    cname: str
    meta: Optional[Dict[str, Any]] = None


@dataclass
class SeparationRoundResult:
    found_count: int = 0
    selected_count: int = 0
    added_count: int = 0
    max_violation: float = 0.0
    avg_selected_violation: float = 0.0
    per_day_counts: Dict[int, int] = None
    selected_cuts: List[SeparatedCut] = None

    def __post_init__(self) -> None:
        if self.per_day_counts is None:
            self.per_day_counts = {}
        if self.selected_cuts is None:
            self.selected_cuts = []

def _lookup_var(master: Any, name: Optional[str], cache: Dict[str, Any]) -> Any:
    if not name:
        return None
    v = cache.get(name)
    if v is not None:
        return v
    v = master.model.getVarByName(name)
    if v is not None:
        cache[name] = v
    return v


class MasterSeparator(ABC):
    def handles_key(self, key: Tuple[Any, ...]) -> bool:
        return False

    def select_cuts(self, master: Any, cuts: Sequence[SeparatedCut]) -> List[SeparatedCut]:
        del master
        return list(cuts)

    @abstractmethod
    def find_violated_cuts(self, master: Any, tol: float, var_cache: Dict[str, Any]) -> List[SeparatedCut]:
        raise NotImplementedError

    @abstractmethod
    def add_cut(self, master: Any, cut: SeparatedCut, var_cache: Dict[str, Any]) -> bool:
        raise NotImplementedError


class CapacityLinkSeparator(MasterSeparator):
    """Capacity-link separator for schedule-demand vs route-usage cuts."""

    def handles_key(self, key: Tuple[Any, ...]) -> bool:
        return isinstance(key, tuple) and len(key) >= 1 and key[0] in ("capacity_link_tk", "capacity_link_t")

    def _lhs_tk(self, master: Any, t: int, k: int, var_cache: Dict[str, Any]) -> float:
        lhs = 0.0
        for e in master.required_edges:
            dem = float(master.inst["demand"].get(e, 0.0))
            if dem <= 0.0:
                continue
            pats = master.schedule_patterns[e]
            for p_idx, pat in enumerate(pats):
                if int(t) not in pat:
                    continue
                vname = master.schedule_var_name.get((e, p_idx, int(k)))
                sv = _lookup_var(master, vname, var_cache)
                if sv is not None:
                    lhs += dem * float(sv.X)
        for lname in master.lambda_var_names_by_day.get((int(t), int(k)), []):
            lv = _lookup_var(master, lname, var_cache)
            if lv is not None:
                lhs += -float(master.capacity) * float(lv.X)
        return lhs

    def _lhs_t(self, master: Any, t: int, var_cache: Dict[str, Any]) -> float:
        lhs = 0.0
        for e in master.required_edges:
            dem = float(master.inst["demand"].get(e, 0.0))
            if dem <= 0.0:
                continue
            pats = master.schedule_patterns[e]
            for k in master.vehicles:
                for p_idx, pat in enumerate(pats):
                    if int(t) not in pat:
                        continue
                    vname = master.schedule_var_name.get((e, p_idx, int(k)))
                    sv = _lookup_var(master, vname, var_cache)
                    if sv is not None:
                        lhs += dem * float(sv.X)
        for k in master.vehicles:
            for lname in master.lambda_var_names_by_day.get((int(t), int(k)), []):
                lv = _lookup_var(master, lname, var_cache)
                if lv is not None:
                    lhs += -float(master.capacity) * float(lv.X)
        return lhs

    def find_violated_cuts(self, master: Any, tol: float, var_cache: Dict[str, Any]) -> List[SeparatedCut]:
        out: List[SeparatedCut] = []
        m = master.model

        for t in master.days:
            for k in master.vehicles:
                cname = f"cap_sep_t{int(t)}_k{int(k)}"
                if m.getConstrByName(cname) is not None:
                    continue
                if self._lhs_tk(master, int(t), int(k), var_cache) > tol:
                    out.append(SeparatedCut(key=("capacity_link_tk", int(t), int(k)), cname=cname))

        for t in master.days:
            cname = f"cap_sep_t{int(t)}"
            if m.getConstrByName(cname) is not None:
                continue
            if self._lhs_t(master, int(t), var_cache) > tol:
                out.append(SeparatedCut(key=("capacity_link_t", int(t)), cname=cname))

        return out

    def add_cut(self, master: Any, cut: SeparatedCut, var_cache: Dict[str, Any]) -> bool:
        m = master.model
        if m.getConstrByName(cut.cname) is not None:
            return False

        key = cut.key
        if key[0] == "capacity_link_tk":
            _, t, k = key
            expr = gp.LinExpr()
            for e in master.required_edges:
                dem = float(master.inst["demand"].get(e, 0.0))
                if dem <= 0.0:
                    continue
                pats = master.schedule_patterns[e]
                for p_idx, pat in enumerate(pats):
                    if int(t) not in pat:
                        continue
                    vname = master.schedule_var_name.get((e, p_idx, int(k)))
                    sv = _lookup_var(master, vname, var_cache)
                    if sv is not None:
                        expr += dem * sv
            for lname in master.lambda_var_names_by_day.get((int(t), int(k)), []):
                lv = _lookup_var(master, lname, var_cache)
                if lv is not None:
                    expr += -float(master.capacity) * lv
            m.addConstr(expr <= 0.0, name=cut.cname)
            master.register_aggregate_constr(("capacity_link_tk", int(t), int(k)), cut.cname)
            return True

        if key[0] == "capacity_link_t":
            _, t = key
            expr = gp.LinExpr()
            for e in master.required_edges:
                dem = float(master.inst["demand"].get(e, 0.0))
                if dem <= 0.0:
                    continue
                pats = master.schedule_patterns[e]
                for k in master.vehicles:
                    for p_idx, pat in enumerate(pats):
                        if int(t) not in pat:
                            continue
                        vname = master.schedule_var_name.get((e, p_idx, int(k)))
                        sv = _lookup_var(master, vname, var_cache)
                        if sv is not None:
                            expr += dem * sv
            for k in master.vehicles:
                for lname in master.lambda_var_names_by_day.get((int(t), int(k)), []):
                    lv = _lookup_var(master, lname, var_cache)
                    if lv is not None:
                        expr += -float(master.capacity) * lv
            m.addConstr(expr <= 0.0, name=cut.cname)
            master.register_aggregate_constr(("capacity_link_t", int(t)), cut.cname)
            return True

        return False


class SRI3Separator(MasterSeparator):
    """Subset-row inequality separator for day-aggregate required-edge triples."""

    def __init__(self, cardinality: int = 3) -> None:
        self.cardinality = int(cardinality)

    def handles_key(self, key: Tuple[Any, ...]) -> bool:
        return isinstance(key, tuple) and len(key) >= 1 and key[0] == "sri3_t"

    @staticmethod
    def _inst_flag(master: Any, key: str, default: Any) -> Any:
        inst = getattr(master, "inst", None)
        if isinstance(inst, dict):
            return inst.get(key, default)
        return default

    @staticmethod
    def _edge_token(e: Any) -> str:
        if isinstance(e, tuple) and len(e) >= 2:
            return f"{int(e[0])}_{int(e[1])}"
        return str(e)

    @staticmethod
    def _route_overlap(subset_set: set[Any], served_edges: Sequence[Any]) -> int:
        return sum(1 for e in subset_set if e in served_edges)

    @classmethod
    def _subset_lhs(cls, routes: Sequence[Tuple[float, Sequence[Any]]], subset: Sequence[Any]) -> float:
        subset_set = set(subset)
        lhs = 0.0
        for lam_val, served_edges in routes:
            if float(lam_val) <= 0.0:
                continue
            if cls._route_overlap(subset_set, served_edges) >= 2:
                lhs += float(lam_val)
        return float(lhs)

    @classmethod
    def sanity_check_examples(cls) -> Dict[str, float]:
        e1 = (1, 2)
        e2 = (2, 3)
        e3 = (3, 4)
        subset = (e1, e2, e3)
        frac_routes = [
            (0.5, (e1, e2)),
            (0.5, (e2, e3)),
            (0.5, (e1, e3)),
        ]
        lhs_frac = cls._subset_lhs(frac_routes, subset)
        if abs(lhs_frac - 1.5) > 1e-9:
            raise AssertionError(f"Unexpected fractional SRI lhs: {lhs_frac}")

        full_route = [(1.0, (e1, e2, e3))]
        lhs_full = cls._subset_lhs(full_route, subset)
        if abs(lhs_full - 1.0) > 1e-9:
            raise AssertionError(f"Unexpected full-route SRI lhs: {lhs_full}")

        split_feasible = [
            (1.0, (e1, e2)),
            (1.0, (e3,)),
        ]
        lhs_split = cls._subset_lhs(split_feasible, subset)
        if abs(lhs_split - 1.0) > 1e-9:
            raise AssertionError(f"Unexpected split-route SRI lhs: {lhs_split}")

        return {
            "fractional_triangle_lhs": float(lhs_frac),
            "fractional_triangle_rhs": 1.0,
            "fractional_triangle_violation": float(lhs_frac - 1.0),
            "full_route_lhs": float(lhs_full),
            "split_route_lhs": float(lhs_split),
        }

    @staticmethod
    def _shared_edges(cut_a: SeparatedCut, cut_b: SeparatedCut) -> int:
        return len(set(cut_a.key[2]) & set(cut_b.key[2]))

    @classmethod
    def too_similar_to_selected(cls, cut: SeparatedCut, selected: Sequence[SeparatedCut], max_shared_edges: int) -> bool:
        for old in selected:
            if cut.key[1] != old.key[1]:
                continue
            if cls._shared_edges(cut, old) > int(max_shared_edges):
                return True
        return False

    def select_cuts(self, master: Any, cuts: Sequence[SeparatedCut]) -> List[SeparatedCut]:
        min_violation = float(self._inst_flag(master, "min_sri_violation", 1e-4))
        max_cuts_per_round = max(0, int(self._inst_flag(master, "max_cuts_per_round", 20)))
        max_cuts_per_day = max(0, int(self._inst_flag(master, "max_cuts_per_day", 5)))
        use_similarity = bool(int(self._inst_flag(master, "enable_sri_similarity_filter", 1)))
        max_shared_edges = int(self._inst_flag(master, "max_shared_edges_between_sri3", 1))

        filtered = []
        for cut in cuts:
            violation = float((cut.meta or {}).get("violation", 0.0))
            if violation <= min_violation:
                continue
            filtered.append(cut)
        filtered.sort(key=lambda c: float((c.meta or {}).get("violation", 0.0)), reverse=True)

        selected: List[SeparatedCut] = []
        selected_keys: set[Tuple[Any, ...]] = set()
        count_by_day: Dict[int, int] = {}
        active_reg = getattr(master, "aggregate_branch_constrs", None)
        active_keys = set(active_reg.keys()) if isinstance(active_reg, dict) else set()

        for cut in filtered:
            if max_cuts_per_round > 0 and len(selected) >= max_cuts_per_round:
                break
            day = int(cut.key[1])
            if max_cuts_per_day > 0 and int(count_by_day.get(day, 0)) >= max_cuts_per_day:
                continue
            if cut.key in active_keys or cut.key in selected_keys:
                continue
            if use_similarity and self.too_similar_to_selected(cut, selected, max_shared_edges=max_shared_edges):
                continue
            selected.append(cut)
            selected_keys.add(cut.key)
            count_by_day[day] = int(count_by_day.get(day, 0)) + 1
        return selected

    @classmethod
    def sanity_check_selection_policy(cls) -> Dict[str, Any]:
        from types import SimpleNamespace

        sep = cls(cardinality=3)

        def mkcut(day: int, subset: Tuple[Any, Any, Any], viol: float) -> SeparatedCut:
            return SeparatedCut(
                key=("sri3_t", int(day), tuple(sorted(subset))),
                cname=f"sri3_t{day}_{viol}",
                meta={"violation": float(viol)},
            )

        e1 = (1, 2)
        e2 = (2, 3)
        e3 = (3, 4)
        e4 = (4, 5)
        e5 = (5, 6)

        cuts = [
            mkcut(0, (e1, e2, e3), 0.50),
            mkcut(0, (e1, e2, e4), 0.40),
            mkcut(0, (e1, e3, e4), 0.30),
            mkcut(1, (e1, e4, e5), 0.20),
        ]

        master_one = SimpleNamespace(
            inst={
                "min_sri_violation": 1e-4,
                "max_cuts_per_round": 1,
                "max_cuts_per_day": 5,
                "enable_sri_similarity_filter": 0,
                "max_shared_edges_between_sri3": 1,
            },
            aggregate_branch_constrs={},
        )
        sel_one = sep.select_cuts(master_one, cuts)
        if len(sel_one) != 1 or float(sel_one[0].meta["violation"]) != 0.50:
            raise AssertionError("max_cuts_per_round policy failed")

        master_daycap = SimpleNamespace(
            inst={
                "min_sri_violation": 1e-4,
                "max_cuts_per_round": 10,
                "max_cuts_per_day": 2,
                "enable_sri_similarity_filter": 0,
                "max_shared_edges_between_sri3": 1,
            },
            aggregate_branch_constrs={},
        )
        sel_daycap = sep.select_cuts(master_daycap, cuts)
        if sum(1 for c in sel_daycap if int(c.key[1]) == 0) > 2:
            raise AssertionError("max_cuts_per_day policy failed")

        master_sim = SimpleNamespace(
            inst={
                "min_sri_violation": 1e-4,
                "max_cuts_per_round": 10,
                "max_cuts_per_day": 5,
                "enable_sri_similarity_filter": 1,
                "max_shared_edges_between_sri3": 1,
            },
            aggregate_branch_constrs={},
        )
        sel_sim = sep.select_cuts(master_sim, cuts[:2])
        if len(sel_sim) != 1:
            raise AssertionError("similarity filter policy failed")

        return {
            "case_b_selected": len(sel_one),
            "case_c_day0_selected": sum(1 for c in sel_daycap if int(c.key[1]) == 0),
            "case_d_selected": len(sel_sim),
        }

    def find_violated_cuts(self, master: Any, tol: float, var_cache: Dict[str, Any]) -> List[SeparatedCut]:
        out: List[SeparatedCut] = []
        if self.cardinality != 3:
            return out

        m = master.model
        for t in master.days:
            day_routes: List[Tuple[float, Tuple[Any, ...]]] = []
            active_edge_set: set[Any] = set()
            for k in master.vehicles:
                for lname in master.lambda_var_names_by_day.get((int(t), int(k)), []):
                    lv = _lookup_var(master, lname, var_cache)
                    if lv is None:
                        continue
                    lam_val = float(lv.X)
                    if lam_val <= tol:
                        continue
                    idx = master.lambda_var_name_to_index.get(str(lname))
                    if idx is None or idx < 0 or idx >= len(master.route_columns):
                        continue
                    route_col = master.route_columns[idx]
                    served = tuple(sorted(set(route_col.serviced_required_edges)))
                    if not served:
                        continue
                    day_routes.append((lam_val, served))
                    active_edge_set.update(served)

            active_edges = sorted(active_edge_set)
            if len(active_edges) < 3 or not day_routes:
                continue

            for subset in combinations(active_edges, 3):
                cname = (
                    f"sri3_t{int(t)}_"
                    f"{self._edge_token(subset[0])}__{self._edge_token(subset[1])}__{self._edge_token(subset[2])}"
                )
                if m.getConstrByName(cname) is not None:
                    continue
                lhs = self._subset_lhs(day_routes, subset)
                if lhs > 1.0 + tol:
                    out.append(
                        SeparatedCut(
                            key=("sri3_t", int(t), tuple(subset)),
                            cname=cname,
                            meta={
                                "lhs": float(lhs),
                                "rhs": 1.0,
                                "violation": float(lhs - 1.0),
                            },
                        )
                    )
        return out

    def add_cut(self, master: Any, cut: SeparatedCut, var_cache: Dict[str, Any]) -> bool:
        m = master.model
        if m.getConstrByName(cut.cname) is not None:
            return False
        key = cut.key
        if not (isinstance(key, tuple) and len(key) == 3 and key[0] == "sri3_t"):
            return False

        _, t, subset = key
        subset_tuple = tuple(subset)
        subset_set = set(subset_tuple)
        expr = gp.LinExpr()

        for k in master.vehicles:
            for lname in master.lambda_var_names_by_day.get((int(t), int(k)), []):
                lv = _lookup_var(master, lname, var_cache)
                if lv is None:
                    continue
                idx = master.lambda_var_name_to_index.get(str(lname))
                if idx is None or idx < 0 or idx >= len(master.route_columns):
                    continue
                route_col = master.route_columns[idx]
                overlap = sum(1 for e in subset_set if e in route_col.serviced_required_edges)
                if overlap >= 2:
                    expr += lv

        m.addConstr(expr <= 1.0, name=cut.cname)
        master.register_aggregate_constr(("sri3_t", int(t), subset_tuple), cut.cname)
        return True

def _remove_separation_cuts(master: Any, m: Any, cnames: Sequence[str]) -> None:
    """Remove constraints by name and drop matching keys from aggregate_branch_constrs."""
    if not cnames:
        return
    cset = {str(cn) for cn in cnames}
    for cname in cset:
        c = m.getConstrByName(cname)
        if c is not None:
            m.remove(c)
    reg = getattr(master, "aggregate_branch_constrs", None)
    if isinstance(reg, dict):
        stale = [k for k, cn in list(reg.items()) if str(cn) in cset]
        for k in stale:
            reg.pop(k, None)
    m.update()
    inv = getattr(master, "invalidate_constr_cache_after_cut_removal", None)
    if callable(inv):
        inv(list(cset))


class SeparationManager:
    def __init__(self, separators: Sequence[MasterSeparator], tol: float = 1e-7, max_rounds: int = 50) -> None:
        self.separators = list(separators)
        self.tol = float(tol)
        self.max_rounds = int(max_rounds)
        self.total_rounds: int = 0
        self.total_cuts_added: int = 0

    def separate_once(self, master: Any, optimize_before: bool = True) -> SeparationRoundResult:
        result = SeparationRoundResult()
        if not self.separators:
            return result

        m = master.model
        if optimize_before:
            m.optimize()
            st = int(getattr(m, "Status", 0))
            if st != 2:
                return result

        var_cache: Dict[str, Any] = {}
        for vv in m.getVars():
            var_cache[vv.VarName] = vv

        selected_entries: List[Tuple[MasterSeparator, SeparatedCut]] = []
        selected_violations: List[float] = []
        per_day_counts: Dict[int, int] = {}

        for sep in self.separators:
            found = list(sep.find_violated_cuts(master, self.tol, var_cache))
            result.found_count += int(len(found))
            selected = list(sep.select_cuts(master, found))
            result.selected_count += int(len(selected))
            for cut in selected:
                selected_entries.append((sep, cut))
                result.selected_cuts.append(cut)
                violation = float((cut.meta or {}).get("violation", 0.0))
                selected_violations.append(violation)
                if isinstance(cut.key, tuple) and len(cut.key) >= 2 and cut.key[0] == "sri3_t":
                    day = int(cut.key[1])
                    per_day_counts[day] = int(per_day_counts.get(day, 0)) + 1

        for sep, cut in selected_entries:
            if sep.add_cut(master, cut, var_cache):
                result.added_count += 1

        if result.added_count > 0:
            m.update()

        result.per_day_counts = dict(per_day_counts)
        if selected_violations:
            result.max_violation = float(max(selected_violations))
            result.avg_selected_violation = float(mean(selected_violations))

        self.total_rounds += 1
        self.total_cuts_added += int(result.added_count)
        return result

    def separate(self, master: Any) -> int:
        if not self.separators:
            return 0

        m = master.model
        added_total = 0
        rounds = 0
        # Each inner list = constraint names added in one successful separation round.
        # If optimize() fails after a batch, pop batches (LIFO) until LP is OPTIMAL again.
        batches: List[List[str]] = []

        for _ in range(self.max_rounds):
            rounds += 1
            m.optimize()
            st = int(getattr(m, "Status", 0))
            did_rollback = False
            while st != 2 and batches:  # GRB.OPTIMAL == 2
                did_rollback = True
                last_batch = batches.pop()
                _remove_separation_cuts(master, m, last_batch)
                added_total -= len(last_batch)
                m.optimize()
                st = int(getattr(m, "Status", 0))
            if st != 2:
                break
            # Re-adding the same violated cuts after rollback can loop forever; stop this call.
            if did_rollback:
                break

            var_cache: Dict[str, Any] = {}
            for vv in m.getVars():
                var_cache[vv.VarName] = vv

            pending: Dict[Tuple[Any, ...], SeparatedCut] = {}
            for sep in self.separators:
                for cut in sep.find_violated_cuts(master, self.tol, var_cache):
                    pending[cut.key] = cut

            if not pending:
                break

            batch_cnames: List[str] = []
            for cut in pending.values():
                placed = False
                for sep in self.separators:
                    if sep.handles_key(cut.key) and sep.add_cut(master, cut, var_cache):
                        batch_cnames.append(cut.cname)
                        placed = True
                        break
                if not placed:
                    for sep in self.separators:
                        if sep.add_cut(master, cut, var_cache):
                            batch_cnames.append(cut.cname)
                            break

            if not batch_cnames:
                break

            m.update()
            batches.append(batch_cnames)
            added_total += len(batch_cnames)

        self.total_rounds += int(rounds)
        self.total_cuts_added += int(added_total)
        return int(added_total)
