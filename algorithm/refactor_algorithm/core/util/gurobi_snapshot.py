from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def snapshot_constr_pi_by_name(model: Any) -> Dict[str, float]:
    """Fetch all constraint duals in one batched Gurobi call."""
    constrs = list(model.getConstrs())
    if not constrs:
        return {}

    names = model.getAttr("ConstrName", constrs)
    pis = model.getAttr("Pi", constrs)
    return {
        str(name): float(pi)
        for name, pi in zip(names, pis)
    }


def snapshot_var_values_by_name(
    model: Any,
    *,
    eps: Optional[float] = None,
    prefixes: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Fetch variable values in one batched call, optionally sparsified."""
    vars_ = list(model.getVars())
    if not vars_:
        return {}

    names = model.getAttr("VarName", vars_)
    values = model.getAttr("X", vars_)

    prefix_tuple = tuple(str(p) for p in prefixes) if prefixes else None
    tol = None if eps is None else abs(float(eps))
    out: Dict[str, float] = {}

    for name, value in zip(names, values):
        name_s = str(name)
        if prefix_tuple is not None and not name_s.startswith(prefix_tuple):
            continue
        value_f = float(value)
        if tol is not None and abs(value_f) <= tol:
            continue
        out[name_s] = value_f
    return out
