from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from refactor_algorithm.core.master.compare_arc_vs_bnp import (
    SimpleSPMaster,
    _build_executable_solution_payload,
    _solve_sp_master_exact,
)
from refactor_algorithm.core.pricing.node import (
    BnBConfig,
    BnBNode,
    BnBTree,
    BranchCandidate,
    NodeSolveResult,
    NodeStatus,
    BestBoundSelector,
    DepthFirstSelector,
    is_edge_day_driver_service_branch_family,
    is_edge_driver_assign_branch_family,
    is_schedule_branch_family,
)


class GlobalRMPBnBTree(BnBTree):
    """
    Branch-and-bound with one shared RMP model.
    Branch constraints are applied per node solve and rolled back afterwards.
    """

    def _clone_master_problem(self, master_problem: Any) -> Any:
        # Global-RMP mode: do not clone model for children.
        return master_problem

    @staticmethod
    def _is_branch_var_name(name: str) -> bool:
        lname = name.lower()
        return lname.startswith("s_") or lname.startswith("lam_") or lname.startswith("lambda_") or lname.startswith("z_")

    def _collect_bound_sensitive_var_names(self, node: BnBNode) -> set[str]:
        """
        Collect only variables whose LB/UB can be changed directly by branching.
        This avoids scanning/restoring all branch vars on every node in global-RMP mode.
        """
        names: set[str] = set()
        data: Optional[Dict[str, Any]] = None

        def ensure_data() -> Dict[str, Any]:
            nonlocal data
            if data is None:
                data = node._get_branching_data()
            return data

        for bc in list(getattr(node, "constraints", ())):
            target = bc.target
            family = str(getattr(bc, "family", ""))

            if family == "lambda_var":
                vref = target.get("var_name") if isinstance(target, dict) else target
                if isinstance(vref, str):
                    names.add(vref)
                continue

            if is_schedule_branch_family(family):
                vref = None
                if isinstance(target, dict):
                    vref = target.get("var_name")
                    if vref is None and "schedule_key" in target:
                        vref = ensure_data().get("schedule_vars", {}).get(target["schedule_key"])
                else:
                    vref = target
                if isinstance(vref, str):
                    names.add(vref)
                continue

            if family == "successive_edges":
                key = target.get("succ_key") if isinstance(target, dict) else target
                expr_obj = ensure_data().get("successive_expr", {}).get(key)
                if isinstance(expr_obj, str):
                    names.add(expr_obj)
                continue

            if is_edge_driver_assign_branch_family(family):
                key = target.get("edge_driver_key") if isinstance(target, dict) else target
                if not (isinstance(key, tuple) and len(key) >= 2):
                    continue
                req_id = key[0]
                sched = ensure_data().get("schedule_vars_by_edge", {})
                req_key = tuple(req_id) if isinstance(req_id, tuple) else req_id
                if isinstance(sched, dict):
                    for vref in sched.get(req_key, ()):
                        if isinstance(vref, str):
                            names.add(vref)
                continue

            if is_edge_day_driver_service_branch_family(family):
                key = target.get("edge_day_driver_key") if isinstance(target, dict) else target
                if not (isinstance(key, tuple) and len(key) >= 3):
                    continue
                req_id = key[0]
                sched = ensure_data().get("schedule_vars_by_edge", {})
                req_key = tuple(req_id) if isinstance(req_id, tuple) else req_id
                if isinstance(sched, dict):
                    for vref in sched.get(req_key, ()):
                        if isinstance(vref, str):
                            names.add(vref)

        return names

    def _backup_branch_var_bounds(self, model: Any, var_names: Optional[set[str]] = None) -> Dict[str, Tuple[float, float]]:
        backup: Dict[str, Tuple[float, float]] = {}
        if var_names is not None:
            for name in var_names:
                v = model.getVarByName(str(name))
                if v is not None:
                    backup[v.VarName] = (float(v.LB), float(v.UB))
            return backup

        for v in model.getVars():
            if self._is_branch_var_name(v.VarName):
                backup[v.VarName] = (float(v.LB), float(v.UB))
        return backup

    def _restore_branch_var_bounds(self, model: Any, backup: Dict[str, Tuple[float, float]]) -> None:
        for vname, b in backup.items():
            v = model.getVarByName(vname)
            if v is None:
                continue
            lb, ub = b
            v.LB = lb
            v.UB = ub

    def _remove_node_branch_constraints(self, model: Any, node_id: int) -> None:
        prefix = f"branch_n{int(node_id)}_"
        to_remove = [c for c in model.getConstrs() if c.ConstrName.startswith(prefix)]
        if to_remove:
            for c in to_remove:
                model.remove(c)
            model.update()

    @staticmethod
    def _cleanup_aggregate_registry(rmp: Any) -> None:
        """Drop registry entries whose constraints no longer exist on the active model.

        When node.rmp is AggregatedMaster with a_flag=False, branch registration goes to
        _fallback_rmp.aggregate_branch_constrs but constraints live on fallback.model.
        Cleaning only the wrapper's dict left stale (cname, Constr) pairs and a poisoned
        SimpleSPMaster._constr_by_name cache → chgCoeff on removed constraints.
        """

        def _clean_reg_on_model(reg_obj: Any, gurobi_model: Any, cache_owner: Any) -> None:
            if not isinstance(reg_obj, dict) or gurobi_model is None:
                return
            stale = [
                k
                for k, cname in reg_obj.items()
                if gurobi_model.getConstrByName(str(cname)) is None
            ]
            for k in stale:
                cname = reg_obj.pop(k, None)
                if cname is not None and cache_owner is not None:
                    cbn = getattr(cache_owner, "_constr_by_name", None)
                    if isinstance(cbn, dict):
                        cbn.pop(str(cname), None)

        model = getattr(rmp, "model", None)
        reg = getattr(rmp, "aggregate_branch_constrs", None)
        _clean_reg_on_model(reg, model, rmp)

        fb = getattr(rmp, "_fallback_rmp", None)
        if fb is not None:
            fb_model = getattr(fb, "model", None)
            fb_reg = getattr(fb, "aggregate_branch_constrs", None)
            if fb_reg is not reg:
                _clean_reg_on_model(fb_reg, fb_model, fb)

    def process_node(self, node: BnBNode) -> NodeSolveResult:
        model = node._get_gurobi_model()
        if bool(getattr(self.config, "use_ub_zero_branching", False)):
            bounds_backup = self._backup_branch_var_bounds(model)
        else:
            bounds_backup = self._backup_branch_var_bounds(
                model,
                var_names=self._collect_bound_sensitive_var_names(node),
            )
        node_solution: Optional[Any] = None

        try:
            result = node.solve_node(self.config, self.global_upper_bound)
            cached_candidates = []
            if result.status not in {NodeStatus.INFEASIBLE, NodeStatus.PRUNED} and not result.is_integral:
                try:
                    cached_candidates = node.extract_fractional_objects(self.config)
                except Exception:
                    cached_candidates = []
            elif result.is_integral and result.best_integer_obj is not None:
                # Must be captured before cleanup mutates the shared model state.
                node_solution = node.get_integer_solution_if_any()
            setattr(node, "_cached_global_candidates", cached_candidates)
        except RuntimeError:
            node.status = NodeStatus.INFEASIBLE
            node.is_solved = True
            setattr(node, "_cached_global_candidates", [])
            result = NodeSolveResult(
                node_id=node.node_id,
                status=NodeStatus.INFEASIBLE,
                lower_bound=float("inf"),
                is_integral=False,
            )
        finally:
            self._remove_node_branch_constraints(model, node.node_id)
            self._restore_branch_var_bounds(model, bounds_backup)
            model.update()
            setattr(node, "_branch_constraints_applied", False)
            self._cleanup_aggregate_registry(getattr(node, "rmp", None))

        self.nodes_processed += 1

        self.update_global_bounds(result)
        stats = getattr(node, "solve_stats", {})
        self.profile["rmp_time_s"] += float(stats.get("rmp_time_s", 0.0))
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
        self.profile["columns_generated"] += float(stats.get("columns_generated", 0.0))
        self.profile["columns_added"] += float(stats.get("columns_added", 0.0))
        self.profile["zero_add_iterations"] += float(stats.get("zero_add_iterations", 0.0))
        self.profile["cg_iterations"] = self.profile.get("cg_iterations", 0.0) + float(stats.get("cg_iterations", 0.0))
        self.profile["phase1_iters"] = self.profile.get("phase1_iters", 0.0) + float(stats.get("phase1_iters", 0.0))
        if bool(stats.get("hit_cg_iteration_limit", False)):
            self.profile["nodes_hit_cg_limit"] += 1.0
        if bool(stats.get("hit_time_limit", False)):
            self.terminated_by_time_limit = True

        if result.is_integral and result.best_integer_obj is not None:
            if node_solution is not None:
                self.best_solution = node_solution
        if bool(self.config.verbose):
            self._print_node_progress(node, result)
        return result

    def create_children(self, parent: BnBNode, candidate: BranchCandidate) -> Tuple[BnBNode, BnBNode]:
        left_bc, right_bc = parent.build_child_constraints(candidate)

        left_constraints = list(parent.constraints) + [left_bc]
        right_constraints = list(parent.constraints) + [right_bc]

        shared_master = parent.rmp

        left_node = BnBNode(
            node_id=self.nodes_created,
            depth=parent.depth + 1,
            master_problem=shared_master,
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
            master_problem=shared_master,
            routes=list(parent.routes),
            constraints=right_constraints,
            parent_id=parent.node_id,
        )
        right_node.lower_bound = parent.lower_bound
        right_node._lp_basis_cache = getattr(parent, "_lp_basis_cache", None)
        self.nodes_created += 1

        return left_node, right_node

    def solve(self) -> Optional[Any]:
        self._solve_start_ts = __import__("time").perf_counter()
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
                elapsed = __import__("time").perf_counter() - (self._solve_start_ts or __import__("time").perf_counter())
                if elapsed >= float(self.config.max_time_s):
                    self.terminated_by_time_limit = True
                    break

            node = self.selector.pop()
            if self._should_skip_open_node_by_bound(node):
                self._mark_pruned_without_solve(node)
                self._refresh_global_lower_bound()
                continue
            node_result = self.process_node(node)
            if self.config.deadline_ts is not None and __import__("time").perf_counter() >= float(self.config.deadline_ts):
                self.terminated_by_time_limit = True
                break

            if self.should_prune_node(node_result):
                self._refresh_global_lower_bound()
                continue

            if node_result.is_integral:
                self._refresh_global_lower_bound()
                continue

            candidates = getattr(node, "_cached_global_candidates", [])
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


def solve_with_global_rmp_algorithm(inst: Dict[str, Any]) -> Dict[str, Any]:
    rmp = SimpleSPMaster(inst)
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

    tree = GlobalRMPBnBTree(
        root_node=root,
        config=BnBConfig(
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
        ),
        selector=selector,
    )

    sol = tree.solve()

    alns_root = getattr(rmp, "initial_incumbent", None)
    root_incumbent_obj: Optional[float] = None
    if isinstance(alns_root, dict):
        try:
            ro = float(alns_root.get("objective", float("inf")))
            if math.isfinite(ro) and ro < float("inf"):
                root_incumbent_obj = ro
        except (TypeError, ValueError):
            root_incumbent_obj = None

    m = rmp.model
    artificial_sum = 0.0
    if int(getattr(m, "SolCount", 0)) > 0:
        for aname in rmp.artificial_var_name_by_cover.values():
            a = m.getVarByName(aname)
            if a is not None:
                artificial_sum += float(a.X)

    hit_node_limit = tree.config.max_nodes is not None and tree.nodes_processed >= tree.config.max_nodes
    hit_node_limit = bool(hit_node_limit or tree.terminated_by_node_limit)
    hit_time_limit = bool(tree.terminated_by_time_limit)
    hit_cg_limit = bool(tree.profile.get("nodes_hit_cg_limit", 0.0) > 0.0)
    final_gap_pct = tree._gap_percent()

    profile = {
        "rmp_time_s": float(tree.profile.get("rmp_time_s", 0.0)),
        "pricing_time_s": float(tree.profile.get("pricing_time_s", 0.0)),
        "addcol_time_s": float(tree.profile.get("addcol_time_s", 0.0)),
        "labels_generated": int(tree.profile.get("labels_generated", 0.0)),
        "labels_expanded": int(tree.profile.get("labels_expanded", 0.0)),
        "backtrack_pruned": int(tree.profile.get("backtrack_pruned", 0.0)),
        "shortcut_returns": int(tree.profile.get("shortcut_returns", 0.0)),
        "existing_sig_filtered": int(tree.profile.get("existing_sig_filtered", 0.0)),
        "columns_generated": int(tree.profile.get("columns_generated", 0.0)),
        "columns_added": int(tree.profile.get("columns_added", 0.0)),
        "column_pool_hits": int(getattr(rmp, "column_pool_hits", 0)),
        "column_pool_misses": int(getattr(rmp, "column_pool_misses", 0)),
        "zero_add_iterations": int(tree.profile.get("zero_add_iterations", 0.0)),
        "nodes_hit_cg_limit": int(tree.profile.get("nodes_hit_cg_limit", 0.0)),
        "cg_iterations": int(tree.profile.get("cg_iterations", 0.0)),
        "phase1_iters": int(tree.profile.get("phase1_iters", 0.0)),
    }

    incumbent_obj = float(tree.global_upper_bound) if tree.global_upper_bound < float("inf") else None
    incumbent_solution = tree.best_solution
    if incumbent_solution is None and isinstance(alns_root, dict):
        incumbent_solution = copy.deepcopy(alns_root)
    executable_solution = _build_executable_solution_payload(rmp, incumbent_solution)
    capacity_cuts_added = int(getattr(rmp, "capacity_cuts_added", 0))

    mode = "bnb_global_rmp"
    if require_proof:
        if hit_node_limit:
            raise RuntimeError("Proof mode failed: reached BnB node limit before tree exhaustion.")
        if hit_time_limit:
            raise RuntimeError("Proof mode failed: reached BnB time limit before tree exhaustion.")
        if hit_cg_limit:
            raise RuntimeError("Proof mode failed: reached CG iteration limit at least one node.")
        if not tree.selector.is_empty():
            raise RuntimeError("Proof mode failed: open nodes remain in BnB queue.")
        if tree.best_solution is None or not (tree.global_upper_bound < float("inf")):
            raise RuntimeError("Proof mode failed: no proven finite incumbent.")
        mode = "bnb_global_rmp_proven_optimal"

    if sol is not None and tree.global_upper_bound < float("inf"):
        return {
            "objective": float(tree.global_upper_bound),
            "solution": sol,
            "incumbent_objective": incumbent_obj,
            "incumbent_solution": incumbent_solution,
            "executable_solution": executable_solution,
            "nodes_processed": tree.nodes_processed,
            "mode": mode,
            "artificial_sum": artificial_sum,
            "hit_node_limit": bool(hit_node_limit),
            "hit_time_limit": bool(hit_time_limit),
            "hit_cg_limit": bool(hit_cg_limit),
            "gap_pct": final_gap_pct,
            "profile": profile,
            "root_incumbent": root_incumbent_obj,
            "capacity_cuts_added": capacity_cuts_added,
        }

    if incumbent_obj is not None:
        return {
            "objective": float(incumbent_obj),
            "solution": incumbent_solution,
            "incumbent_objective": incumbent_obj,
            "incumbent_solution": incumbent_solution,
            "executable_solution": executable_solution,
            "nodes_processed": tree.nodes_processed,
            "mode": "bnb_global_rmp_incumbent_only",
            "artificial_sum": artificial_sum,
            "hit_node_limit": bool(hit_node_limit),
            "hit_time_limit": bool(hit_time_limit),
            "hit_cg_limit": bool(hit_cg_limit),
            "gap_pct": final_gap_pct,
            "profile": profile,
            "root_incumbent": root_incumbent_obj,
            "capacity_cuts_added": capacity_cuts_added,
        }

    exact = _solve_sp_master_exact(rmp)
    exact_values: Dict[str, float] = {}
    for v in rmp.model.getVars():
        x = float(v.X)
        if x > 1e-6:
            exact_values[v.VarName] = x
    exact_solution = {"source": "sp_master_exact_fallback", "variables": exact_values}
    exact_exec = _build_executable_solution_payload(rmp, exact_solution)
    fallback_obj = float(exact["objective"])
    tree.global_upper_bound = min(float(tree.global_upper_bound), fallback_obj)
    tree._refresh_global_lower_bound()
    fallback_gap_pct = tree._gap_percent()
    return {
        "objective": fallback_obj,
        "solution": exact_solution,
        "incumbent_objective": fallback_obj,
        "incumbent_solution": exact_solution,
        "executable_solution": exact_exec,
        "nodes_processed": tree.nodes_processed,
        "mode": "sp_master_exact_fallback",
        "artificial_sum": artificial_sum,
        "hit_node_limit": bool(hit_node_limit),
        "hit_time_limit": bool(hit_time_limit),
        "hit_cg_limit": bool(hit_cg_limit),
        "gap_pct": fallback_gap_pct,
        "profile": profile,
        "root_incumbent": root_incumbent_obj,
        "capacity_cuts_added": capacity_cuts_added,
    }
