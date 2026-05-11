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

    sri_specs:
      Tuple of (subset_mask, dual_pi) for SRI-3 cuts. The route coefficient is
      1 once the priced route has serviced at least two required edges from the
      tracked triple, so pricing applies a one-time reduced-cost delta of -pi
      when the serviced-mask crosses that threshold.
    """

    route_constant: float = 0.0
    sri_specs: Tuple[Tuple[int, float], ...] = ()

    def has_rb_specs(self) -> bool:
        return bool(self.sri_specs)

    def rb_delta(self, old_mask: int, new_mask: int, bit_to_req: Sequence[Any]) -> float:
        del bit_to_req
        delta = 0.0
        for subset_mask, dual_pi in self.sri_specs:
            if (old_mask & subset_mask).bit_count() < 2 <= (new_mask & subset_mask).bit_count():
                delta -= float(dual_pi)
        return float(delta)


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

    SRI-3 cuts:
      +1 on a route when it serves at least two edges from the tracked triple.
    """
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
    sri_specs: list[Tuple[int, float]] = []

    for agg_key, cname in reg.items():
        if not isinstance(agg_key, tuple) or len(agg_key) < 1:
            continue
        pi = float(by_name.get(str(cname), 0.0))
        if abs(pi) <= tol:
            continue

        kind = agg_key[0]
        if kind == "whole_route":
            route_const -= pi
            continue

        if kind == "daily_route":
            if len(agg_key) >= 3:
                t, k = int(agg_key[1]), int(agg_key[2])
                if t == day_id and driver_id is not None and k == driver_id:
                    route_const -= pi
            elif len(agg_key) >= 2:
                t = int(agg_key[1])
                if t == day_id:
                    route_const -= pi
            continue

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

        if kind == "sri3_t":
            t = int(agg_key[1])
            if t != day_id or len(agg_key) < 3:
                continue
            subset_mask = 0
            ok = True
            for req in tuple(agg_key[2]):
                bit = req_to_bit.get(req)
                if bit is None:
                    ok = False
                    break
                subset_mask |= int(bit)
            if not ok or subset_mask == 0:
                continue
            sri_specs.append((int(subset_mask), float(pi)))
            if max_rb_cuts > 0 and len(sri_specs) >= int(max_rb_cuts):
                break

        if kind == "ryan_foster_pair":
            if len(agg_key) < 4:
                continue
            t = int(agg_key[3])
            if t != day_id:
                continue
            subset_mask = 0
            ok = True
            for req in (agg_key[1], agg_key[2]):
                bit = req_to_bit.get(req)
                if bit is None:
                    ok = False
                    break
                subset_mask |= int(bit)
            if not ok or subset_mask == 0:
                continue
            sri_specs.append((int(subset_mask), float(pi)))

    return CutPricingState(
        route_constant=float(route_const),
        sri_specs=tuple(sri_specs),
    )
