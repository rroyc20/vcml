from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


ROOT        = Path(__file__).resolve().parents[2]
NATIVE_DIR  = Path(__file__).resolve().parent / "native"


_LIB: Optional[ctypes.CDLL] = None
# Content-addressed (Blake2b) so new dict objects with identical pricing topology still hit cache.
_CPP_CTX_CACHE: Dict[bytes, Dict[str, Any]] = {}

# ── [7] Module-level output buffer cache ─────────────────────────────────────
# Reused across calls; reallocated only when the required size grows.
_OUT_BUF_MAX_COLS:  int = 0
_OUT_BUF_MAX_STEPS: int = 0
_OUT_COL_COUNT: Optional[ctypes.Array] = None
_OUT_RC:        Optional[ctypes.Array] = None
_OUT_OFFSETS:   Optional[ctypes.Array] = None
_OUT_REQ_IDX:   Optional[ctypes.Array] = None
_OUT_SVC_U:     Optional[ctypes.Array] = None
_OUT_SVC_V:     Optional[ctypes.Array] = None


def _accum_cpp_prof(prof: Optional[Dict[str, float]], key: str, t0: float) -> float:
    """Add (now - t0) to prof[key]; return perf_counter() for the next segment."""
    now = time.perf_counter()
    if prof is not None:
        prof[key] = prof.get(key, 0.0) + (now - t0)
    return now


def _collect_nonrequired_transition_edges(
    *,
    depot: Any,
    svc_seq: Sequence[Tuple[Any, Any]],
    nonrequired_edge_set: Optional[Sequence[Any]],
    nr_allowed: Optional[Set[Any]] = None,
    sp_path: Optional[Dict[Any, Dict[Any, Tuple[Any, ...]]]] = None,
) -> List[Any]:
    if nr_allowed is None:
        if not nonrequired_edge_set:
            return []
        allowed = set(nonrequired_edge_set)
    else:
        if not nr_allowed:
            return []
        allowed = nr_allowed
    used: List[Any] = []
    seen: set = set()

    def _append_transition(cur_node: Any, nxt_node: Any) -> None:
        if cur_node == nxt_node:
            return
        e = (cur_node, nxt_node) if cur_node < nxt_node else (nxt_node, cur_node)
        if e in allowed and e not in seen:
            seen.add(e)
            used.append(e)
            return
        if sp_path is None:
            return
        for arc in sp_path.get(cur_node, {}).get(nxt_node, ()):
            if not (isinstance(arc, tuple) and len(arc) >= 2):
                continue
            u, v = arc[0], arc[1]
            ae = (u, v) if u < v else (v, u)
            if ae in allowed and ae not in seen:
                seen.add(ae)
                used.append(ae)

    cur = depot
    for u, v in svc_seq:
        _append_transition(cur, u)
        cur = v
    _append_transition(cur, depot)
    return used


def _ensure_out_buffers(max_columns: int, max_steps_total: int) -> None:
    global _OUT_BUF_MAX_COLS, _OUT_BUF_MAX_STEPS
    global _OUT_COL_COUNT, _OUT_RC, _OUT_OFFSETS
    global _OUT_REQ_IDX, _OUT_SVC_U, _OUT_SVC_V

    if max_columns <= _OUT_BUF_MAX_COLS and max_steps_total <= _OUT_BUF_MAX_STEPS:
        # Existing buffers are large enough — zero out counters only.
        _OUT_COL_COUNT[0] = 0
        return

    # Need larger buffers.
    new_cols  = max(max_columns,    _OUT_BUF_MAX_COLS)
    new_steps = max(max_steps_total, _OUT_BUF_MAX_STEPS)
    _OUT_COL_COUNT = (ctypes.c_int * 1)(0)
    _OUT_RC        = (ctypes.c_double * new_cols)()
    _OUT_OFFSETS   = (ctypes.c_int * (new_cols + 1))()
    _OUT_REQ_IDX   = (ctypes.c_int * new_steps)()
    _OUT_SVC_U     = (ctypes.c_int * new_steps)()
    _OUT_SVC_V     = (ctypes.c_int * new_steps)()
    _OUT_BUF_MAX_COLS  = new_cols
    _OUT_BUF_MAX_STEPS = new_steps


# ── [5] Compile flags: -march=native -funroll-loops ──────────────────────────

def _resolve_cpp_core_paths() -> Tuple[Path, Path]:
    variant = str(os.environ.get("CPP_PRICER_VARIANT", "default")).strip().lower()
    if variant in {"", "default", "base", "orig", "original"}:
        return (
            NATIVE_DIR / "cpp_pricing_core.cpp",
            NATIVE_DIR / "libcpp_pricing_core.so",
        )
    if variant == "rc_load_dom":
        return (
            NATIVE_DIR / "cpp_pricing_core_rc_load_dom.cpp",
            NATIVE_DIR / "libcpp_pricing_core_rc_load_dom.so",
        )
    raise RuntimeError(
        f"Unknown CPP_PRICER_VARIANT={variant!r}. "
        "Supported: default, rc_load_dom."
    )


def _build_library() -> None:
    cpp_src, so_path = _resolve_cpp_core_paths()
    if so_path.exists() and so_path.stat().st_mtime >= cpp_src.stat().st_mtime:
        return
    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-march=native",
        "-funroll-loops",
        "-shared",
        "-fPIC",
        str(cpp_src),
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _load_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    _build_library()
    _cpp_src, so_path = _resolve_cpp_core_paths()
    lib = ctypes.CDLL(str(so_path))
    lib.cpp_price_dp.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.cpp_price_dp.restype = ctypes.c_int
    lib.cpp_price_ng.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.cpp_price_ng.restype = ctypes.c_int
    _LIB = lib
    return lib


def _flatten_sp_cost(node_ids: Sequence[Any], sp_cost: Dict[Any, Dict[Any, float]]) -> List[float]:
    out: List[float] = []
    for i in node_ids:
        row = sp_cost.get(i, {})
        for j in node_ids:
            v = float(row.get(j, float("inf")))
            out.append(v)
    return out


def _blake_feed_depot_meta_arcs(
    h: Any,
    depot: Any,
    req_service_meta: Dict[Any, Tuple[float, float]],
    req_service_arcs: Dict[Any, List[Tuple[Any, Any, Any, float, float, float]]],
) -> None:
    h.update(b"v1\x00")
    h.update(repr(depot).encode("utf-8", errors="surrogatepass"))
    h.update(b"\x00meta\x00")
    for k in sorted(req_service_meta.keys(), key=lambda x: repr(x)):
        dem, mc = req_service_meta[k]
        h.update(repr(k).encode("utf-8", errors="surrogatepass"))
        h.update(struct.pack("<dd", float(dem), float(mc)))
    h.update(b"\x00arcs\x00")
    for k in sorted(req_service_meta.keys(), key=lambda x: repr(x)):
        h.update(repr(k).encode("utf-8", errors="surrogatepass"))
        for t in req_service_arcs.get(k, ()):
            u, v, _aid, tc, dem, sc = t
            h.update(repr(u).encode("utf-8", errors="surrogatepass"))
            h.update(b"\x1f")
            h.update(repr(v).encode("utf-8", errors="surrogatepass"))
            h.update(struct.pack("<ddd", float(tc), float(dem), float(sc)))
    h.update(b"\x00sp\x00")


def _get_cpp_context(
    depot: Any,
    req_service_meta: Dict[Any, Tuple[float, float]],
    req_service_arcs: Dict[Any, List[Tuple[Any, Any, Any, float, float, float]]],
    sp_cost: Dict[Any, Dict[Any, float]],
    sp_path: Dict[Any, Dict[Any, Tuple[Any, ...]]],
) -> Dict[str, Any]:
    req_ids_all = list(req_service_meta.keys())

    node_id_set = set(sp_cost.keys())
    for src, row in sp_cost.items():
        node_id_set.add(src)
        node_id_set.update(row.keys())
    node_id_set.add(depot)
    node_ids = sorted(node_id_set)

    h = hashlib.blake2b(digest_size=32)
    _blake_feed_depot_meta_arcs(h, depot, req_service_meta, req_service_arcs)
    for i in node_ids:
        row = sp_cost.get(i, {})
        for j in node_ids:
            v = float(row.get(j, float("inf")))
            h.update(struct.pack("<d", v))
    cache_key = h.digest()
    cached = _CPP_CTX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    if depot not in node_to_idx:
        raise RuntimeError("depot not found in shortest-path node set")
    depot_idx = int(node_to_idx[depot])

    svc_by_req_idx: Dict[Any, List[Tuple[int, int, float, float, float]]] = {}
    for req_id in req_ids_all:
        rows: List[Tuple[int, int, float, float, float]] = []
        for u, v, _arc_id, tc, dem, sc in req_service_arcs.get(req_id, []):
            if u not in node_to_idx or v not in node_to_idx:
                continue
            rows.append((int(node_to_idx[u]), int(node_to_idx[v]), float(tc), float(sc), float(dem)))
        svc_by_req_idx[req_id] = rows

    # ── [6] Pre-build and cache the sp_flat ctypes array ─────────────────────
    # Avoids list copy + ctypes array construction on every pricing call.
    num_nodes = len(node_ids)
    sp_flat_list = _flatten_sp_cost(node_ids=node_ids, sp_cost=sp_cost)
    sp_flat_ctypes = (ctypes.c_double * (num_nodes * num_nodes))(*sp_flat_list)
    # ── [8] Pre-build and cache svc arrays for ALL req_ids ───────────────────
    # These arrays include every requirement (candidate filtering done in
    # solve_day_cpp_dp per call).  We store the full arrays and per-req
    # index ranges so callers can build subsets cheaply via slicing.
    full_svc_req:     List[int]   = []
    full_svc_from:    List[int]   = []
    full_svc_to:      List[int]   = []
    full_svc_travel:  List[float] = []
    full_svc_service: List[float] = []
    full_svc_demand:  List[float] = []
    req_svc_ranges: Dict[Any, Tuple[int, int]] = {}  # req_id -> (start, end)

    for req_id in req_ids_all:
        start = len(full_svc_req)
        for u_idx, v_idx, tc, sc, dem in svc_by_req_idx.get(req_id, []):
            full_svc_req.append(0)          # req_idx filled per-call
            full_svc_from.append(int(u_idx))
            full_svc_to.append(int(v_idx))
            full_svc_travel.append(float(tc))
            full_svc_service.append(float(sc))
            full_svc_demand.append(float(dem))
        end = len(full_svc_req)
        req_svc_ranges[req_id] = (start, end)

    # Row-wise SP path dicts aligned with node_ids (faster post-process than nested sp_path.get).
    sp_path_rows = [sp_path.get(n, {}) for n in node_ids]

    # Store base arrays (svc_req will be filled per-call based on req_ids subset)
    out = {
        "req_ids_all":    req_ids_all,
        "node_ids":       node_ids,
        "node_to_idx":    node_to_idx,
        "depot_idx":      depot_idx,
        "num_nodes":      num_nodes,
        "svc_by_req_idx": svc_by_req_idx,
        "sp_flat_ctypes": sp_flat_ctypes,   # [6] cached ctypes array
        "sp_path_rows":   sp_path_rows,
        # [8] full service arrays (from/to/travel/service/demand, req filled per-call)
        "full_svc_from":    full_svc_from,
        "full_svc_to":      full_svc_to,
        "full_svc_travel":  full_svc_travel,
        "full_svc_service": full_svc_service,
        "full_svc_demand":  full_svc_demand,
        "req_svc_ranges":   req_svc_ranges,
    }
    _CPP_CTX_CACHE[cache_key] = out
    return out


def _build_required_ng_neighbors(
    req_ids: Sequence[Any],
    req_service_arcs: Dict[Any, List[Tuple[Any, Any, Any, float, float, float]]],
    sp_cost: Dict[Any, Dict[Any, float]],
    ng_size: int,
) -> Tuple[List[int], List[int]]:
    num_reqs = len(req_ids)
    if num_reqs <= 0:
        return [0], []

    ng_size = max(1, int(ng_size))
    req_to_idx = {req_id: idx for idx, req_id in enumerate(req_ids)}
    req_offsets: List[int] = [0]
    req_neighbors: List[int] = []

    for req_id in req_ids:
        scores: List[Tuple[float, str, int]] = []
        src_arcs = req_service_arcs.get(req_id, [])
        for other_id in req_ids:
            other_idx = req_to_idx[other_id]
            if other_id == req_id:
                continue
            dst_arcs = req_service_arcs.get(other_id, [])
            best = float("inf")
            for _, src_to, _, _, _, _ in src_arcs:
                for dst_from, _, _, dst_travel, _, dst_service in dst_arcs:
                    dead = float(sp_cost.get(src_to, {}).get(dst_from, float("inf")))
                    if dead == float("inf"):
                        continue
                    cand = dead + float(dst_travel) + float(dst_service)
                    if cand < best:
                        best = cand
            scores.append((best, str(other_id), other_idx))

        scores.sort(key=lambda x: (x[0], x[1]))
        neigh = [req_to_idx[req_id]]
        neigh.extend(idx for _, _, idx in scores[: max(0, ng_size - 1)])
        neigh = sorted(set(neigh))
        req_neighbors.extend(neigh)
        req_offsets.append(len(req_neighbors))

    return req_offsets, req_neighbors


def _build_active_ng_arrays(active_gamma: Sequence[Sequence[int] | set[int]]) -> Tuple[List[int], List[int]]:
    offsets: List[int] = [0]
    neighbors: List[int] = []
    for row in active_gamma:
        vals = sorted({int(v) for v in row})
        neighbors.extend(vals)
        offsets.append(len(neighbors))
    return offsets, neighbors


def _ng_arrays_to_sets(
    num_reqs: int,
    offsets: Sequence[int],
    neighbors: Sequence[int],
) -> List[set[int]]:
    out: List[set[int]] = []
    for idx in range(num_reqs):
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        out.append(set(int(v) for v in neighbors[start:end]))
    return out


def _dssr_expand_route_violations(
    req_seq_idx: Sequence[int],
    full_ng_sets: Sequence[set[int]],
    active_gamma: List[set[int]],
) -> bool:
    if not req_seq_idx:
        return False

    pi: set[int] = set()
    last_pos: Dict[int, int] = {}
    changed = False

    for pos, req_idx in enumerate(req_seq_idx):
        if req_idx in pi:
            prev = last_pos.get(req_idx)
            if prev is not None and prev < pos:
                for mid in range(prev + 1, pos):
                    carrier = int(req_seq_idx[mid])
                    if req_idx in full_ng_sets[carrier] and req_idx not in active_gamma[carrier]:
                        active_gamma[carrier].add(req_idx)
                        changed = True
        pi = (pi & full_ng_sets[req_idx]) | {req_idx}
        last_pos[req_idx] = pos

    return changed


def solve_day_cpp_dp(
    day: Any,
    driver: Any,
    depot: Any,
    capacity: float,
    edge_duals: Dict[Any, float],
    vehicle_dual: float,
    req_service_meta: Dict[Any, Tuple[float, float]],
    req_service_arcs: Dict[Any, List[Tuple[Any, Any, Any, float, float, float]]],
    sp_cost: Dict[Any, Dict[Any, float]],
    sp_path: Dict[Any, Dict[Any, Tuple[Any, ...]]],
    max_columns: int,
    eps_reduced_cost: float,
    forbidden_edges: Optional[set] = None,
    max_candidate_reqs: int = 60,
    nonrequired_edge_set: Optional[Sequence[Any]] = None,
    discount_sp_path: Optional[Dict[Any, Dict[Any, Tuple[Any, ...]]]] = None,
    prof_out: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if max_columns <= 0:
        return []

    t_seg = time.perf_counter()
    ctx = _get_cpp_context(
        depot=depot,
        req_service_meta=req_service_meta,
        req_service_arcs=req_service_arcs,
        sp_cost=sp_cost,
        sp_path=sp_path,
    )
    t_seg = _accum_cpp_prof(prof_out, "ctx", t_seg)
    req_ids_all: List[Any] = list(ctx["req_ids_all"])
    if not req_ids_all:
        return []
    forbidden = forbidden_edges or set()

    max_candidate_reqs = int(max(1, min(61, max_candidate_reqs)))
    cand_scores: List[Tuple[Any, float]] = []
    for r in req_ids_all:
        if r in forbidden:
            continue
        svc_lb = float(req_service_meta.get(r, (0.0, float("inf")))[1])
        dual   = float(edge_duals.get(r, 0.0))
        cand_scores.append((r, svc_lb - dual))

    req_ids = [r for (r, delta) in cand_scores if delta < -float(eps_reduced_cost)]
    if not req_ids:
        cand_scores.sort(key=lambda x: x[1])
        req_ids = [r for (r, _d) in cand_scores[:max_candidate_reqs]]
    if len(req_ids) > max_candidate_reqs:
        score_map = {r: d for (r, d) in cand_scores}
        req_ids.sort(key=lambda r: score_map.get(r, float("inf")))
        req_ids = req_ids[:max_candidate_reqs]
    if not req_ids:
        _accum_cpp_prof(prof_out, "cand_svc", t_seg)
        return []

    req_to_idx  = {r: i for i, r in enumerate(req_ids)}
    node_ids: List[Any] = list(ctx["node_ids"])
    depot_idx   = int(ctx["depot_idx"])
    num_nodes   = int(ctx["num_nodes"])

    # ── [8] Build svc arrays from cached per-req data ────────────────────────
    # Only iterate over selected req_ids; avoids rebuilding from scratch each call.
    svc_req_list:     List[int]   = []
    svc_from_list:    List[int]   = []
    svc_to_list:      List[int]   = []
    svc_travel_list:  List[float] = []
    svc_service_list: List[float] = []
    svc_demand_list:  List[float] = []

    full_from    = ctx["full_svc_from"]
    full_to      = ctx["full_svc_to"]
    full_travel  = ctx["full_svc_travel"]
    full_service = ctx["full_svc_service"]
    full_demand  = ctx["full_svc_demand"]
    ranges       = ctx["req_svc_ranges"]

    for req_id in req_ids:
        req_idx = req_to_idx[req_id]
        start, end = ranges.get(req_id, (0, 0))
        for pos in range(start, end):
            svc_req_list.append(req_idx)
            svc_from_list.append(full_from[pos])
            svc_to_list.append(full_to[pos])
            svc_travel_list.append(full_travel[pos])
            svc_service_list.append(full_service[pos])
            svc_demand_list.append(full_demand[pos])

    if not svc_req_list:
        _accum_cpp_prof(prof_out, "cand_svc", t_seg)
        return []

    req_dual_list = [float(edge_duals.get(r, 0.0)) for r in req_ids]

    num_reqs   = len(req_ids)
    num_svc    = len(svc_req_list)
    max_columns_int   = int(max_columns)
    max_steps_total   = max_columns_int * max(1, num_reqs)
    t_seg = _accum_cpp_prof(prof_out, "cand_svc", t_seg)

    # Build ctypes input arrays from lists
    c_int = ctypes.c_int
    c_dbl = ctypes.c_double

    svc_req_arr     = (c_int * num_svc)(*svc_req_list)
    svc_from_arr    = (c_int * num_svc)(*svc_from_list)
    svc_to_arr      = (c_int * num_svc)(*svc_to_list)
    svc_travel_arr  = (c_dbl * num_svc)(*svc_travel_list)
    svc_service_arr = (c_dbl * num_svc)(*svc_service_list)
    svc_demand_arr  = (c_dbl * num_svc)(*svc_demand_list)
    req_dual_arr    = (c_dbl * num_reqs)(*req_dual_list)

    # ── [6] Reuse cached sp_flat ctypes array (no list copy) ─────────────────
    sp_arr = ctx["sp_flat_ctypes"]

    # ── [7] Reuse module-level output buffers ─────────────────────────────────
    _ensure_out_buffers(max_columns_int, max_steps_total)

    lib = _load_library()
    t_seg = _accum_cpp_prof(prof_out, "ctypes", t_seg)
    t_native = time.perf_counter()
    ret = int(
        lib.cpp_price_dp(
            int(num_nodes),
            int(num_reqs),
            int(num_svc),
            int(depot_idx),
            float(capacity),
            float(vehicle_dual),
            float(eps_reduced_cost),
            max_columns_int,
            svc_req_arr,
            svc_from_arr,
            svc_to_arr,
            svc_travel_arr,
            svc_service_arr,
            svc_demand_arr,
            req_dual_arr,
            sp_arr,
            int(max_steps_total),
            _OUT_COL_COUNT,
            _OUT_RC,
            _OUT_OFFSETS,
            _OUT_REQ_IDX,
            _OUT_SVC_U,
            _OUT_SVC_V,
        )
    )
    _accum_cpp_prof(prof_out, "native", t_native)
    if ret != 0:
        raise RuntimeError(f"cpp_price_dp failed with code {ret}")

    col_count = int(_OUT_COL_COUNT[0])
    cols: List[Dict[str, Any]] = []

    t_decode = time.perf_counter()
    node_to_idx_ctx: Dict[Any, int] = ctx["node_to_idx"]
    sp_path_rows: List[Dict[Any, Tuple[Any, ...]]] = ctx["sp_path_rows"]
    nr_pre: Optional[Set[Any]] = (
        set(nonrequired_edge_set) if nonrequired_edge_set else None
    )

    for ci in range(col_count):
        b = int(_OUT_OFFSETS[ci])
        e = int(_OUT_OFFSETS[ci + 1])
        if b < 0 or e < b:
            continue

        req_seq: List[Any]             = []
        svc_seq: List[Tuple[Any, Any]] = []
        for p in range(b, e):
            ridx  = int(_OUT_REQ_IDX[p])
            u_idx = int(_OUT_SVC_U[p])
            v_idx = int(_OUT_SVC_V[p])
            if 0 <= ridx < len(req_ids) and 0 <= u_idx < len(node_ids) and 0 <= v_idx < len(node_ids):
                req_seq.append(req_ids[ridx])
                svc_seq.append((node_ids[u_idx], node_ids[v_idx]))

        if not req_seq or not svc_seq:
            continue

        # Reconstruct full traversal with shortest deadheading paths
        path_arcs: List[Any] = []
        cur = depot
        for (u, v) in svc_seq:
            ri = node_to_idx_ctx[cur]
            path_arcs.extend(sp_path_rows[ri].get(u, ()))
            path_arcs.append((u, v))
            cur = v
        ri = node_to_idx_ctx[cur]
        path_arcs.extend(sp_path_rows[ri].get(depot, ()))

        path_nodes: List[Any] = [depot]
        for a in path_arcs:
            if isinstance(a, tuple) and len(a) >= 2:
                path_nodes.append(a[1])

        seen:           set       = set()
        served_ordered: List[Any] = []
        for r in req_seq:
            if r in seen:
                continue
            seen.add(r)
            served_ordered.append(r)

        col: Dict[str, Any] = {
            "day":                    day,
            "path_nodes":             path_nodes,
            "path_arcs":              path_arcs,
            "serviced_required_edges": served_ordered,
            "nonrequired_edges_used": _collect_nonrequired_transition_edges(
                depot=depot,
                svc_seq=svc_seq,
                nonrequired_edge_set=nonrequired_edge_set,
                nr_allowed=nr_pre,
                sp_path=discount_sp_path,
            ),
            "reduced_cost":           float(_OUT_RC[ci]),
        }
        if driver is not None:
            col["driver"] = driver
        cols.append(col)

    _accum_cpp_prof(prof_out, "decode", t_decode)
    return cols


def solve_day_cpp_ngroute(
    day: Any,
    driver: Any,
    depot: Any,
    capacity: float,
    edge_duals: Dict[Any, float],
    vehicle_dual: float,
    req_service_meta: Dict[Any, Tuple[float, float]],
    req_service_arcs: Dict[Any, List[Tuple[Any, Any, Any, float, float, float]]],
    sp_cost: Dict[Any, Dict[Any, float]],
    sp_path: Dict[Any, Dict[Any, Tuple[Any, ...]]],
    max_columns: int,
    eps_reduced_cost: float,
    forbidden_edges: Optional[set] = None,
    ng_size: int = 8,
    nonrequired_edge_set: Optional[Sequence[Any]] = None,
    discount_sp_path: Optional[Dict[Any, Dict[Any, Tuple[Any, ...]]]] = None,
    prof_out: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if max_columns <= 0:
        return []

    t_seg = time.perf_counter()
    ctx = _get_cpp_context(
        depot=depot,
        req_service_meta=req_service_meta,
        req_service_arcs=req_service_arcs,
        sp_cost=sp_cost,
        sp_path=sp_path,
    )
    t_seg = _accum_cpp_prof(prof_out, "ctx", t_seg)
    req_ids_all: List[Any] = list(ctx["req_ids_all"])
    if not req_ids_all:
        return []
    forbidden = forbidden_edges or set()

    req_ids = [r for r in req_ids_all if r not in forbidden]
    if not req_ids:
        _accum_cpp_prof(prof_out, "cand_svc", t_seg)
        return []

    req_to_idx = {r: i for i, r in enumerate(req_ids)}
    node_ids: List[Any] = list(ctx["node_ids"])
    depot_idx = int(ctx["depot_idx"])
    num_nodes = int(ctx["num_nodes"])

    svc_req_list: List[int] = []
    svc_from_list: List[int] = []
    svc_to_list: List[int] = []
    svc_travel_list: List[float] = []
    svc_service_list: List[float] = []
    svc_demand_list: List[float] = []

    full_from = ctx["full_svc_from"]
    full_to = ctx["full_svc_to"]
    full_travel = ctx["full_svc_travel"]
    full_service = ctx["full_svc_service"]
    full_demand = ctx["full_svc_demand"]
    ranges = ctx["req_svc_ranges"]

    for req_id in req_ids:
        req_idx = req_to_idx[req_id]
        start, end = ranges.get(req_id, (0, 0))
        for pos in range(start, end):
            svc_req_list.append(req_idx)
            svc_from_list.append(full_from[pos])
            svc_to_list.append(full_to[pos])
            svc_travel_list.append(full_travel[pos])
            svc_service_list.append(full_service[pos])
            svc_demand_list.append(full_demand[pos])

    if not svc_req_list:
        _accum_cpp_prof(prof_out, "cand_svc", t_seg)
        return []

    req_dual_list = [float(edge_duals.get(r, 0.0)) for r in req_ids]
    full_ng_offsets, full_ng_neighbors = _build_required_ng_neighbors(
        req_ids=req_ids,
        req_service_arcs=req_service_arcs,
        sp_cost=sp_cost,
        ng_size=int(ng_size),
    )
    full_ng_sets = _ng_arrays_to_sets(len(req_ids), full_ng_offsets, full_ng_neighbors)

    num_reqs = len(req_ids)
    num_svc = len(svc_req_list)
    max_columns_int = int(max_columns)
    max_steps_total = max_columns_int * max(1, 4 * num_reqs)
    t_seg = _accum_cpp_prof(prof_out, "cand_svc", t_seg)

    c_int = ctypes.c_int
    c_dbl = ctypes.c_double

    svc_req_arr = (c_int * num_svc)(*svc_req_list)
    svc_from_arr = (c_int * num_svc)(*svc_from_list)
    svc_to_arr = (c_int * num_svc)(*svc_to_list)
    svc_travel_arr = (c_dbl * num_svc)(*svc_travel_list)
    svc_service_arr = (c_dbl * num_svc)(*svc_service_list)
    svc_demand_arr = (c_dbl * num_svc)(*svc_demand_list)
    req_dual_arr = (c_dbl * num_reqs)(*req_dual_list)

    sp_arr = ctx["sp_flat_ctypes"]

    _ensure_out_buffers(max_columns_int, max_steps_total)

    lib = _load_library()
    t_seg = _accum_cpp_prof(prof_out, "ctypes", t_seg)
    if num_reqs <= max(1, int(ng_size)):
        active_gamma: List[set[int]] = [set(row) for row in full_ng_sets]
        max_dssr_iters = 1
    else:
        active_gamma = [set() for _ in range(num_reqs)]
        max_dssr_iters = max(1, num_reqs * max(1, min(int(ng_size), num_reqs)))
    best_valid_cols: List[Dict[str, Any]] = []

    for _itr in range(max_dssr_iters):
        t_ct = time.perf_counter()
        ng_offsets_list, ng_neighbors_list = _build_active_ng_arrays(active_gamma)
        ng_offsets_arr = (c_int * len(ng_offsets_list))(*ng_offsets_list)
        ng_neighbors_arr = (
            (c_int * max(1, len(ng_neighbors_list)))(*ng_neighbors_list)
            if ng_neighbors_list
            else (c_int * 1)(0)
        )
        _accum_cpp_prof(prof_out, "ctypes", t_ct)

        t_nat = time.perf_counter()
        ret = int(
            lib.cpp_price_ng(
                int(num_nodes),
                int(num_reqs),
                int(num_svc),
                int(depot_idx),
                float(capacity),
                float(vehicle_dual),
                float(eps_reduced_cost),
                max_columns_int,
                svc_req_arr,
                svc_from_arr,
                svc_to_arr,
                svc_travel_arr,
                svc_service_arr,
                svc_demand_arr,
                req_dual_arr,
                sp_arr,
                ng_offsets_arr,
                ng_neighbors_arr,
                int(max_steps_total),
                _OUT_COL_COUNT,
                _OUT_RC,
                _OUT_OFFSETS,
                _OUT_REQ_IDX,
                _OUT_SVC_U,
                _OUT_SVC_V,
            )
        )
        _accum_cpp_prof(prof_out, "native", t_nat)
        if ret != 0:
            raise RuntimeError(f"cpp_price_ng failed with code {ret}")

        col_count = int(_OUT_COL_COUNT[0])
        if col_count <= 0:
            return best_valid_cols

        cols: List[Dict[str, Any]] = []
        changed = False
        valid_cols: List[Dict[str, Any]] = []

        t_dec = time.perf_counter()
        node_to_idx_ctx: Dict[Any, int] = ctx["node_to_idx"]
        sp_path_rows: List[Dict[Any, Tuple[Any, ...]]] = ctx["sp_path_rows"]
        nr_pre: Optional[Set[Any]] = (
            set(nonrequired_edge_set) if nonrequired_edge_set else None
        )

        for ci in range(col_count):
            b = int(_OUT_OFFSETS[ci])
            e = int(_OUT_OFFSETS[ci + 1])
            if b < 0 or e < b:
                continue

            req_seq: List[Any] = []
            req_seq_idx: List[int] = []
            svc_seq: List[Tuple[Any, Any]] = []
            for p in range(b, e):
                ridx = int(_OUT_REQ_IDX[p])
                u_idx = int(_OUT_SVC_U[p])
                v_idx = int(_OUT_SVC_V[p])
                if 0 <= ridx < len(req_ids) and 0 <= u_idx < len(node_ids) and 0 <= v_idx < len(node_ids):
                    req_seq_idx.append(ridx)
                    req_seq.append(req_ids[ridx])
                    svc_seq.append((node_ids[u_idx], node_ids[v_idx]))

            if not req_seq or not svc_seq:
                continue

            path_arcs: List[Any] = []
            cur = depot
            for (u, v) in svc_seq:
                ri = node_to_idx_ctx[cur]
                path_arcs.extend(sp_path_rows[ri].get(u, ()))
                path_arcs.append((u, v))
                cur = v
            ri = node_to_idx_ctx[cur]
            path_arcs.extend(sp_path_rows[ri].get(depot, ()))

            path_nodes: List[Any] = [depot]
            for arc in path_arcs:
                if isinstance(arc, tuple) and len(arc) >= 2:
                    path_nodes.append(arc[1])

            col: Dict[str, Any] = {
                "day": day,
                "path_nodes": path_nodes,
                "path_arcs": path_arcs,
                "serviced_required_edges": list(req_seq),
                "nonrequired_edges_used": _collect_nonrequired_transition_edges(
                    depot=depot,
                    svc_seq=svc_seq,
                    nonrequired_edge_set=nonrequired_edge_set,
                    nr_allowed=nr_pre,
                    sp_path=discount_sp_path,
                ),
                "reduced_cost": float(_OUT_RC[ci]),
            }
            if driver is not None:
                col["driver"] = driver
            cols.append(col)

            if _dssr_expand_route_violations(req_seq_idx, full_ng_sets, active_gamma):
                changed = True
            else:
                valid_cols.append(col)

        _accum_cpp_prof(prof_out, "decode", t_dec)

        if valid_cols:
            return valid_cols
        if not changed:
            return cols
        best_valid_cols = valid_cols

    return best_valid_cols
