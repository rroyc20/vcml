from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple


def normalize_cut_pricing_mode(mode: Any) -> str:
    """
    Supported modes:
      - legacy  : keep previous set-touch logic
      - bitmask : precompile RB cut F-sets into bitmasks for faster delta
      - auto    : alias of bitmask
    """
    m = str(mode).strip().lower()
    if m in {"legacy", "bitmask", "auto"}:
        return m
    return "legacy"


@dataclass
class CutPricingState:
    """
    Day/context-specific cut-dual view used inside pricing.

    route_constant:
      Constant reduced-cost shift from capacity-link cuts.

    Non-capacity cut pricing adjustments are intentionally disabled. Pricing
    only receives the capacity-link route constant.
    """

    route_constant: float = 0.0

    def has_rb_specs(self) -> bool:
        return False

    def rb_delta(self, old_mask: int, new_mask: int, bit_to_req: Sequence[Any]) -> float:
        del old_mask, new_mask, bit_to_req
        return 0.0


def _extract_day_ctx(day_ctx: Any) -> Tuple[int, int | None]:
    if isinstance(day_ctx, tuple) and len(day_ctx) >= 2:
        return int(day_ctx[0]), int(day_ctx[1])
    if isinstance(day_ctx, tuple) and len(day_ctx) >= 1:
        return int(day_ctx[0]), None
    return int(day_ctx), None


def build_cut_pricing_state(
    *,
    day_ctx: Any,
    dual_values: Dict[str, Any],
    pricing_data: Dict[str, Any],
    capacity: float,
    req_to_bit: Dict[Any, int],
    max_rb_cuts: int,
    mode: str,
    dual_tol: float = 1e-15,
) -> CutPricingState:
    """
    Build per-day/per-driver cut-dual data for pricing.

    Capacity-link cuts:
      -Q on each lambda in the row -> constant +pi*Q for a priced route.

    Non-capacity cut pricing contributions are intentionally ignored.
    """
    del req_to_bit, max_rb_cuts, mode
    tol = abs(float(dual_tol))

    by_name = dual_values.get("constr_pi_by_name")
    if not isinstance(by_name, dict):
        by_name = {}
    reg = pricing_data.get("aggregate_branch_constrs")
    if not isinstance(reg, dict):
        reg = {}

    day_id, driver_id = _extract_day_ctx(day_ctx)
    route_const = 0.0
    cap_f = float(capacity)

    for agg_key, cname in reg.items():
        if not isinstance(agg_key, tuple) or len(agg_key) < 1:
            continue
        pi = float(by_name.get(str(cname), 0.0))
        if abs(pi) <= tol:
            continue

        kind = agg_key[0]
        if kind == "capacity_link_tk":
            t, k = int(agg_key[1]), int(agg_key[2])
            if t == day_id and driver_id is not None and k == driver_id:
                route_const += pi * cap_f
            continue

        if kind == "capacity_link_t":
            t = int(agg_key[1])
            if t == day_id and driver_id is not None:
                route_const += pi * cap_f
            continue

    return CutPricingState(
        route_constant=float(route_const),
    )
