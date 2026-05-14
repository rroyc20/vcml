from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    sri_specs_by_added_bit: Optional[Dict[int, Tuple[Tuple[int, float], ...]]] = None

    def __post_init__(self) -> None:
        if self.sri_specs_by_added_bit is not None:
            return

        by_bit: Dict[int, List[Tuple[int, float]]] = {}
        for subset_mask_raw, dual_pi in self.sri_specs:
            subset_mask = int(subset_mask_raw)
            remaining = subset_mask
            while remaining:
                added_bit = remaining & -remaining
                remaining -= added_bit
                other_mask = subset_mask ^ added_bit
                by_bit.setdefault(added_bit, []).append((other_mask, float(dual_pi)))

        self.sri_specs_by_added_bit = {
            int(bit): tuple(specs)
            for bit, specs in by_bit.items()
        }

    def has_rb_specs(self) -> bool:
        return bool(self.sri_specs)

    def rb_delta_added(self, old_mask: int, added_bit: int) -> float:
        """
        Reduced-cost delta when a DP transition services exactly one new edge.

        SRI/Ryan-Foster route coefficients switch from 0 to 1 only when the old
        route already contains exactly one other tracked edge from the subset.
        Pre-indexing by added bit avoids scanning every active cut per transition.
        """
        added = int(added_bit)
        old = int(old_mask)
        if added == 0 or (old & added):
            return 0.0

        specs_by_bit = self.sri_specs_by_added_bit or {}
        delta = 0.0
        for other_mask, dual_pi in specs_by_bit.get(added, ()):
            if (old & int(other_mask)).bit_count() == 1:
                delta -= float(dual_pi)
        return float(delta)

    def rb_delta(self, old_mask: int, new_mask: int, bit_to_req: Sequence[Any]) -> float:
        del bit_to_req
        added = int(new_mask) & ~int(old_mask)
        if added and (added & (added - 1)) == 0:
            return self.rb_delta_added(int(old_mask), int(added))

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
    preindexed_day_constrs: Optional[List[Tuple[tuple, str]]] = None,
) -> CutPricingState:
    """
    Build per-day/per-driver cut-dual data for pricing.

    Capacity-link cuts:
      -Q on each lambda in the row -> constant +pi*Q for a priced route.

    SRI-3 cuts:
      +1 on a route when it serves at least two edges from the tracked triple.

    ``preindexed_day_constrs``:
      Pre-filtered list of (agg_key, cname) pairs for this day (whole-route +
      day-matching entries only). When provided the full registry scan is skipped,
      reducing O(|all_constrs|) → O(|day_constrs|) per pricing call.
    """
    tol = abs(float(dual_tol))

    by_name = dual_values.get("constr_pi_by_name")
    if not isinstance(by_name, dict):
        by_name = {}

    day_id, driver_id = _extract_day_ctx(day_ctx)
    route_const = 0.0
    cap_f = float(capacity)
    sri_specs: list[Tuple[int, float]] = []

    if preindexed_day_constrs is not None:
        items_iter: Any = preindexed_day_constrs
    else:
        reg = pricing_data.get("aggregate_branch_constrs")
        if not isinstance(reg, dict):
            reg = {}
        items_iter = reg.items()

    for agg_key, cname in items_iter:
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

        if kind == "ryan_foster_pair_avg":
            if len(agg_key) < 4 or not isinstance(agg_key[3], (list, tuple)):
                continue
            rf_days = tuple(int(t) for t in agg_key[3])
            if day_id not in rf_days:
                continue
            subset_mask = 0
            ok = True
            for req in (agg_key[1], agg_key[2]):
                bit = req_to_bit.get(req)
                if bit is None:
                    ok = False
                    break
                subset_mask |= int(bit)
            if not ok or subset_mask == 0 or not rf_days:
                continue
            sri_specs.append((int(subset_mask), float(pi) / float(len(rf_days))))

    return CutPricingState(
        route_constant=float(route_const),
        sri_specs=tuple(sri_specs),
    )
