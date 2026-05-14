# Current Algorithm Pseudocode Report

본 문서는 현재 구현된 Branch-and-Price 계열 알고리즘을 default 실험 설정 기준으로 정리한다.
설명 기준은 다음과 같다.

- `use_aggregation = 1`: Aggregated Restricted Master Problem(A-RMP)에서 시작한다.
- `enable_sri = 1`, `use_sri_cuts = 1`: SRI-3 subset-row inequality cut을 사용한다.
- `yao_style_pricing = 1`: Yao et al. 방식의 pricing을 사용한다.
- `use_transformed_pricing_graph = 1`: transformed pricing graph 위에서 pricing을 수행한다.

전체 알고리즘은 A-RMP 기반 Branch-and-Price-and-Cut 구조이다. 루트 노드에서 A-RMP를 만들고,
각 노드마다 LP relaxation을 column generation으로 수렴시킨 뒤, fractional 해가 남아 있으면 cut 또는
branching으로 relaxation을 강화한다. A-RMP에서 정수해가 발견되면 먼저 disaggregation으로 차량별 해로
복원 가능한지 확인하고, 실패하면 같은 노드를 표준 SimpleSP RMP로 전환해 다시 판정한다.

```text
Algorithm 0. Default A-RMP Branch-and-Price-and-Cut

Input:
    PCARP instance I
    Default configuration:
        use_aggregation = true
        enable_sri = true
        yao_style_pricing = true
        use_transformed_pricing_graph = true

Output:
    Best feasible integer solution and bound information

1. Build AggregatedMaster(I)
       - Create schedule variables q[e,p]
       - Create aggregated route variables lambda[t,r]
       - Create aggregated discount variables y[e]
       - Add initial heuristic / artificial columns

2. Create root node N0 with no branch constraints.
3. Insert N0 into the open-node selector.
4. Initialize global lower bound LB and incumbent upper bound UB.

5. while open-node selector is not empty:
       N <- select next node

       if N can be pruned by inherited bound:
           mark N as pruned
           continue

       Apply branch constraints of N to the shared RMP.
       SolveNodeByColumnGenerationAndCuts(N)
       Restore temporary node-specific branch bounds / rows.

       Update global LB and UB.

       if N is infeasible or bound-pruned:
           continue

       if N is integral:
           update incumbent if better
           continue

       C <- ExtractBranchingCandidates(N)
       if C is empty:
           continue

       c* <- ChooseBranchCandidate(C)
       (N_left, N_right) <- CreateChildren(N, c*)
       Insert N_left and N_right into the open-node selector.

6. return incumbent solution and final bound.
```

## 1. Pricing

Pricing은 현재 RMP의 dual 값을 받아 negative reduced cost route column을 찾는 하위 문제이다.
default에서는 A-RMP를 사용하므로 route column은 차량 인덱스가 없는 `lambda[t,r]` 형태이며,
pricing subproblem도 기본적으로 day별로 수행된다. 표준 RMP로 전환된 경우에는 `(day, vehicle)`별
pricing으로 바뀐다.

핵심 목적은 다음 reduced cost가 음수인 route를 찾는 것이다.

```text
reduced_cost(route)
    = route travel/service cost
      - cover dual contribution
      - vehicle/day capacity dual contribution
      - discount-link dual contribution
      + active branch/cut dual adjustment
```

default의 Yao pricing with graph transformation은 다음 두 아이디어를 결합한다.

- Yao-style pricing: non-required sparse road edge의 discount-link dual을 shortest path layer에 반영해 하위 shortest path 비용을 조정한다.
- Transformed pricing graph: required edge service arc는 명시적으로 유지하고, 순수 deadheading subpath는 shortcut arc로 압축한다.

이 설정에서 pricing은 metric closure 전체를 그대로 탐색하지 않고, required service와 deadheading shortcut이 섞인 transformed graph에서 resource-constrained shortest path를 푼다. resource는 vehicle capacity이며, label state는 대략 `(served_required_edges_mask, current_node, load, reduced_cost)`로 볼 수 있다.

```text
Algorithm 1. Pricing With Yao-Style Transformed Graph

Input:
    Dual values pi from current RMP
    Active branch constraints B
    Active cut duals gamma
    Pricing graph data from instance
    eps_rc

Output:
    New route columns with reduced_cost < -eps_rc

1. Build or reuse transformed pricing graph:
       - Keep required service arcs explicitly.
       - Replace pure deadhead paths by shortcut arcs.
       - Keep sparse road graph information for discount dual handling.

2. For each pricing context ctx:
       In A-RMP mode:
           ctx = day t
       In SimpleSP mode:
           ctx = (day t, vehicle k)

       2.1 Extract duals relevant to ctx:
             cover duals for required-edge/day constraints
             vehicle/day route-count dual
             discount-link duals for non-required sparse edges
             branch/cut dual adjustments

       2.2 Apply branch filters:
             remove forbidden required edges for this node/day
             incorporate route-level branch constraints as pricing penalties or constants

       2.3 Modify transformed graph costs:
             service arc cost includes service/travel cost minus cover dual
             deadhead shortcut cost includes shortest path travel cost
             Yao-style discount duals are reflected on sparse-road shortest paths

       2.4 Run dynamic-programming / labeling search:
             initialize label at depot with empty served set and zero load
             expand deadhead moves and service moves
             reject labels that exceed capacity
             keep best reduced-cost state for each state key
             whenever a depot-return state has nonempty served set:
                 if reduced_cost < -eps_rc:
                     reconstruct path and create route column

       2.5 Filter generated columns:
             remove duplicate signatures
             remove coefficient-dominated columns if enabled
             stop when max column cap is reached

3. Return all accepted negative reduced cost columns.
```

Column generation loop 안에서 pricing은 다음 방식으로 호출된다.

```text
Algorithm 2. Node Column Generation Loop

Input:
    Node N
    Incumbent upper bound UB

1. repeat until CG limit or time limit:
       Solve current RMP LP and collect dual values.

       if LP lower bound >= UB:
           prune N by bound
           stop

       Run pricing with current duals.

       if negative reduced cost columns are found:
           add columns to RMP
           continue

       if dual stabilization is active and this was an in-step:
           retry pricing once with pure LP duals
           if columns are found:
               add columns and continue

       No negative reduced cost column exists under current cut set.
       declare CG convergence for this node and move to cut/integrality checks.
```

Phase-I artificial variables are used as a feasibility safety net. If artificial cover mass remains positive after CG convergence, the node is treated as infeasible for the original master and is pruned.

## 2. Branching

Branching starts only after the current node's LP relaxation has converged under column generation and no further SRI cut is added. The algorithm then extracts fractional structures from the LP solution and chooses one branching candidate.

In default A-RMP mode, branching is performed over aggregated objects:

- whole-route mass: total aggregated route usage over all days.
- daily-route mass: aggregated route usage for a specific day.
- schedule assignment: aggregated schedule variable `q[e,p]`.
- Ryan-Foster pair: whether two required edges are served together in a day-level route.
- fallback lambda branching: individual aggregated route variable when higher-level candidates are unavailable.

The implemented priority is hierarchical: the algorithm first looks for more structural fractional objects, then falls back to lower-level variable branching. This keeps the tree closer to the routing/schedule semantics instead of immediately branching on arbitrary lambda variables.

```text
Algorithm 3. Extract Branching Candidates in A-RMP Mode

Input:
    Converged LP solution at node N
    Integrality tolerance eps_int

Output:
    Candidate list C

1. Read branching data from AggregatedMaster:
       lambda variables by day
       schedule variables q[e,p]
       route-lifted arc/node/Ryan-Foster expressions

2. Search candidate families in hierarchical order:
       2.1 whole_route:
             aggregate total route mass and test integrality

       2.2 daily_route:
             aggregate route mass per day and test integrality

       2.3 schedule_fix:
             inspect q[e,p] values and find fractional schedule choices

       2.4 Ryan-Foster pair:
             inspect pair-service expressions for required-edge pairs
             find fractional together/separate decisions

       2.5 lambda fallback:
             inspect individual lambda[t,r] values

3. Return the first nonempty candidate level.
```

After candidates are found, the selected candidate is chosen by a deterministic ranking rule.

- Binary-like candidates, such as schedule or Ryan-Foster decisions, prefer values closest to `0.5`.
- Integer-split candidates, such as whole-route or daily-route mass, prefer values farthest from the nearest integer.
- Stable tie-breaking uses family, target, day, and driver information.

```text
Algorithm 4. Branch Candidate Selection and Child Creation

Input:
    Candidate list C

Output:
    Two child nodes

1. c* <- candidate with best branching rank.

2. if c* is binary-like:
       left constraint  <- expression(c*) <= 0
       right constraint <- expression(c*) >= 1

3. else:
       left constraint  <- expression(c*) <= floor(value(c*))
       right constraint <- expression(c*) >= ceil(value(c*))

4. Create child nodes:
       left child inherits parent constraints + left constraint
       right child inherits parent constraints + right constraint

5. Add children to open-node selector.
```

Branch constraints are applied in two places.

- Existing RMP columns: incompatible variables are fixed by bounds or constrained by temporary branch rows.
- Future pricing columns: the branch registry is passed to pricing so future generated routes obey the same branch decision or receive the correct reduced-cost adjustment.

Because the implementation uses a global/shared RMP model, node-specific branch bounds and rows are restored after each node is processed. This prevents a branch decision from leaking into unrelated nodes.

## 3. Cutting Plane Algorithm

The default cut component is SRI-3 separation. It is integrated with column generation rather than run as a standalone MIP cutting loop.

There are two cut hooks in the node solve process.

- General `separate_cuts`: called before each RMP LP solve. In A-RMP mode this currently returns zero because the existing general separators target the SimpleSP master. If the node switches to SimpleSP mode, this hook delegates to the fallback RMP.
- SRI separation: called after CG convergence, but before branching, when the converged LP solution is still fractional.

In A-RMP mode, SRI-3 cuts are aggregated day-level subset-row inequalities. For a required-edge subset `S` of cardinality 3 and a day `t`, the cut has the form:

```text
sum_{r: route r services at least two edges in S} lambda[t,r] <= floor(|S| / 2) = 1
```

For `|S| = 3`, this prevents the classic fractional triangle pattern in which three half-routes each cover two of the three required edges.

```text
Algorithm 5. SRI-3 Cutting Plane Loop Inside a Node

Input:
    CG-converged fractional LP solution at node N
    max_sri_rounds

Output:
    Either strengthened RMP or a node ready for branching

1. After CG convergence, inspect the LP solution.

2. if solution is integral:
       skip cut separation and handle incumbent/disaggregation.

3. if SRI cuts are disabled for this node depth:
       skip cut separation and proceed to branching.

4. if number of SRI rounds at this node reaches max_sri_rounds:
       skip cut separation and proceed to branching.

5. Separate violated SRI-3 cuts:
       for each day t:
           build route overlap information from positive lambda[t,r]
           inspect required-edge triples S
           compute lhs = sum lambda[t,r] for routes overlapping S in at least two edges
           if lhs > 1 + tolerance:
               record violated cut

6. Select a limited set of cuts:
       apply violation threshold
       apply per-round / per-day limits
       optionally apply similarity filtering

7. Add selected cuts to the RMP.

8. if at least one cut was added:
       reset stabilization state
       return to the column generation loop

9. else:
       no violated SRI cut remains under the current column pool
       proceed to branching.
```

The key point is that every time SRI cuts are added, the current column pool may no longer be sufficient. Therefore the algorithm does not branch immediately after adding cuts. It goes back to pricing and regenerates any route columns that become attractive under the new dual prices. This is why the method is best described as Branch-and-Price-and-Cut:

```text
RMP solve -> pricing until CG convergence -> SRI separation -> pricing again -> branching
```

If A-RMP finds an integral aggregated solution, the algorithm does not immediately accept it as a final vehicle-level solution. It first solves or verifies a disaggregation problem that assigns aggregated routes and schedule mass to vehicles. If disaggregation succeeds, the incumbent is updated. If it fails, the node locally switches from A-RMP to SimpleSP RMP and repeats the node logic in the exact vehicle-indexed space.

## Implementation Map

- A-RMP construction and aggregation logic: `refactor_algorithm/core/master/aggregated_master.py`
- Branch-and-bound tree and shared global RMP cleanup: `refactor_algorithm/core/master/compare_global_rmp_bnp.py`
- Node-level column generation, pricing, branching, and SRI loop: `refactor_algorithm/core/pricing/node.py`
- SRI separator implementation: `refactor_algorithm/core/master/separation.py`
- Default instance/config construction: `refactor_algorithm/engine/instances.py`
