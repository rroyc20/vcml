from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gurobipy as gp


@dataclass(frozen=True)
class SeparatedCut:
    key: Tuple[Any, ...]
    cname: str

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
