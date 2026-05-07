from __future__ import annotations

from typing import Any, Dict, Tuple


def _split_day_context(day_ctx: Any) -> Tuple[Any, Any]:
    if isinstance(day_ctx, tuple):
        if len(day_ctx) >= 2:
            return day_ctx[0], day_ctx[1]
        if len(day_ctx) == 1:
            return day_ctx[0], None
    return day_ctx, None


def adjust_vehicle_dual_with_lex(
    day_ctx: Any,
    base_vehicle_dual: float,
    dual_values: Dict[str, Any],
    pricing_data: Dict[str, Any],
) -> float:
    """
    Add vehicle-lex dual contributions into day-driver vehicle dual.

    SimpleSPMaster column coefficients for lex constraints are:
      +1 for k_hi, -1 for k_lo in constraint (sum λ_hi - sum λ_lo <= 0).
    So route reduced cost should include the same signed dual terms.
    """
    day_id, driver_id = _split_day_context(day_ctx)
    if driver_id is None:
        return float(base_vehicle_dual)

    by_name = dual_values.get("constr_pi_by_name", {})
    if not isinstance(by_name, dict):
        return float(base_vehicle_dual)

    lex_name_map = pricing_data.get("vehicle_lex_constr_name_by_day", {})
    if not isinstance(lex_name_map, dict) or not lex_name_map:
        return float(base_vehicle_dual)

    try:
        day_i = int(day_id)
        drv_i = int(driver_id)
    except (TypeError, ValueError):
        return float(base_vehicle_dual)

    eff = float(base_vehicle_dual)
    for key, cname in lex_name_map.items():
        if not (isinstance(key, tuple) and len(key) == 3):
            continue
        t, k_lo, k_hi = key
        try:
            if int(t) != day_i:
                continue
            lo_i = int(k_lo)
            hi_i = int(k_hi)
        except (TypeError, ValueError):
            continue
        pi = by_name.get(str(cname))
        if pi is None:
            continue
        pi_f = float(pi)
        if drv_i == hi_i:
            eff += pi_f
        elif drv_i == lo_i:
            eff -= pi_f

    return eff

