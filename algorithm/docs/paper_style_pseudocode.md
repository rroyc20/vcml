# Paper-Style Pseudocode for the Default Algorithm

This note gives paper-style pseudocode for the current default setting:

- Aggregated master problem is used first (`use_aggregation = 1`).
- SRI-3 cuts are enabled.
- Yao-style pricing is enabled.
- Pricing is performed on the transformed pricing graph.
- Initial A-RMP columns are seeded by the q-load heuristic.
- A fast heuristic pricing routine is called before the exact DP pricing routine.

## Algorithm 1. Default A-RMP Branch-and-Price-and-Cut

**Input:** PCARP instance \(I\), maximum number of nodes \(N_{\max}\), maximum column generation iterations \(G_{\max}\), reduced-cost tolerance \(\epsilon\).  
**Output:** Best feasible solution \(x^\star\) and its objective value \(UB\).

```text
1:  Construct the aggregated restricted master problem A-RMP.
2:  Generate initial aggregated route columns by Algorithm 5.
3:  Add artificial cover variables with a large Phase-I penalty if needed.
4:  Create the root node n0 with no branching constraints.
5:  Initialize the open node list L <- {n0}, incumbent x* <- null, and UB <- +infinity.

6:  while L is not empty and the stopping criteria are not met do
7:      Select and remove a node n from L.
8:      Apply all branching constraints inherited by n to the shared RMP.

9:      Solve node n by column generation:
10:         repeat
11:             Solve the current RMP relaxation and obtain dual values.
12:             if the RMP lower bound is no smaller than UB then
13:                 Fathom n by bound and continue with the next open node.
14:             end if

15:             Run the heuristic pricing routine by Algorithm 2.
16:             if no negative reduced-cost column is found then
17:                 Run the exact DP pricing routine by Algorithm 3.
18:             end if
19:             if negative reduced-cost columns are found then
20:                 Add the generated columns to the RMP.
21:             end if
22:         until neither heuristic nor exact pricing finds a negative reduced-cost column,
            or Gmax is reached

23:      if artificial variables remain positive after convergence then
24:          Fathom n as infeasible for the original master and continue with the next open node.
25:      end if

26:      if the converged LP solution is fractional then
27:          Separate SRI-3 cuts by Algorithm 4.
28:          if at least one SRI-3 cut is added then
29:              Return to line 10 and re-run column generation.
30:          end if
31:      end if

32:      if the converged LP solution is integral then
33:          Try to disaggregate the aggregated solution into vehicle-indexed routes.
34:          if disaggregation succeeds then
35:              Update incumbent x* and UB if the solution is better.
36:              Restore node-specific branching changes and continue.
37:          else
38:              Switch this node locally to the vehicle-indexed SimpleSP RMP.
39:              Return to line 10 and solve the node in the exact RMP space.
40:          end if
41:      end if

42:      Extract fractional branching candidates from the converged LP solution.
43:      if no branching candidate exists then
44:          Restore node-specific branching changes and continue.
45:      end if

46:      Select the best branching candidate c.
47:      Create two child nodes by imposing the left and right branching constraints.
48:      Insert both child nodes into L.

49:      Restore node-specific branching bounds and temporary rows from the shared RMP.
50:  end while

51:  return x* and UB.
```

## Heuristic Pricing vs. q-load Initial Columns

The q-load heuristic and the Letchford-Oukil heuristic pricing routine serve different purposes.

- q-load is an initial column heuristic. It does not require dual values and is used before the first RMP solve to create a schedule-consistent root column scaffold.
- Letchford-Oukil-type heuristic pricing is a pricing heuristic. It requires dual values from the current RMP and is used during column generation to find negative reduced-cost columns quickly.
- Therefore, the stronger design is not to replace q-load, but to keep q-load for initialization and call heuristic pricing before exact DP pricing at each column generation iteration.

## Algorithm 2. Heuristic Pricing Before Exact DP

This heuristic follows the idea of pricing over restricted route structures before invoking the exact pricing routine. It is faster than exact pricing because it searches a smaller route family.

The first version considers routes with a three-phase structure:

1. Deadhead from the depot to the first required edge.
2. Service one or more required edges consecutively without deadheading.
3. Deadhead from the last serviced edge back to the depot.

The second version permits limited deadheading during the service phase by charging one unit of artificial load for each deadheaded edge. This expands the search space while keeping the dynamic program small.

**Input:** Day/vehicle context \((t,k)\), dual values, modified costs, capacity \(Q\), tolerance \(\epsilon\).  
**Output:** A set of heuristic route columns with negative reduced cost.

```text
1:  Compute modified deadheading costs cbar_{ij}^{tk}.
2:  Compute c_i^*, the shortest modified deadheading cost from the depot to each required-edge endpoint i.

3:  Initialize f(i,q) <- +infinity for all required-edge endpoints i and q = 1,...,Q.
4:  Set f(i,0) <- c_i^* for all required-edge endpoints i.
5:  Store predecessor information for route reconstruction.

6:  for q = 0,...,Q-1 do
7:      Optional limited-deadheading extension:
8:      for each non-service edge {i,j} in the road graph do
9:          if q + 1 <= Q then
10:             Relax f(j,q+1) using f(i,q) + cbar_{ij}^{tk}.
11:             Relax f(i,q+1) using f(j,q) + cbar_{ij}^{tk}.
12:         end if
13:     end for

14:     for each required edge e = {i,j} with demand d_e and q + d_e <= Q do
15:         Relax f(j,q+d_e) using f(i,q) + service_cost(e) - pi_e^{tk}.
16:         Relax f(i,q+d_e) using f(j,q) + service_cost(e) - pi_e^{tk}.
17:     end for
18: end for

19: Initialize Omega_h <- empty set.
20: for q = 1,...,Q do
21:     for each required-edge endpoint i do
22:         if f(i,q) + c_i^* - sigma^{tk} < -epsilon then
23:             Reconstruct the corresponding route r.
24:             Add r to Omega_h.
25:         end if
26:     end for
27: end for

28: return Omega_h.
```

If Algorithm 2 finds at least one negative reduced-cost column, those columns are added to the RMP and exact pricing is skipped for that column generation iteration. Exact pricing is called only when the heuristic fails to find an improving column.

## Algorithm 3. Exact DP Pricing on the Transformed Graph

For a route \(r\) on day \(t\) by vehicle \(k\), the pricing subproblem searches for a route with negative reduced cost. The deadheading arc cost is first modified as

\[
\bar{c}_{ij}^{tk} =
\begin{cases}
c_{ij} - \mu_{ij}^{tk}, & (i,j)\in E\setminus E^R,\\
c_{ij}, & \text{otherwise},
\end{cases}
\]

where \(\mu_{ij}^{tk}\) is the dual value associated with the discount-link constraint. Let \(P_{ij}^{tk}\) denote the shortest deadheading path between transformed graph nodes \(i\) and \(j\) under \(\bar{c}^{tk}\), and let

\[
\tilde{c}_{ij}^{tk} =
\sum_{(u,v)\in P_{ij}^{tk}} \bar{c}_{uv}^{tk}.
\]

The transformed graph contains the depot and the endpoints of required edges as nodes. Its arcs represent either shortest deadheading paths or required-edge service moves.

**Input:** Day/vehicle context \((t,k)\), dual values, transformed graph \(G^{tk}\), capacity \(Q\), tolerance \(\epsilon\).  
**Output:** A set of route columns with negative reduced cost.

```text
1:  Extract cover duals pi_e^{tk}, route-limit dual sigma^{tk},
    discount-link duals mu_{ij}^{tk}, and active cut/branching dual adjustments.

2:  Construct the transformed pricing graph G^{tk}:
3:      Use modified deadheading costs cbar_{ij}^{tk}.
4:      Connect the depot and required-edge endpoints by shortest deadheading paths.
5:      Keep required-edge service moves explicitly.

6:  Initialize the label set with the depot label:
        l0 = (v = depot, S = empty set, q = 0, rc = -sigma^{tk}).
    Here S is the set of serviced required edges, q is the accumulated load,
    and rc is the accumulated reduced cost.

7:  Initialize an empty column set Omega_new.

8:  while there exists an unprocessed label l = (v, S, q, rc) do
9:      Mark l as processed.

10:     for each deadheading transition (v,u) in G^{tk} do
11:         Create l' = (u, S, q, rc + ctilde_{vu}^{tk}).
12:         Insert l' if it is not dominated.
13:     end for

14:     for each required edge e that can be serviced from v do
15:         if e not in S and q + d_e <= Q then
16:             Let u be the endpoint reached after servicing e.
17:             rc' <- rc + deadheading cost to e
                         + service cost of e
                         - pi_e^{tk}
                         + cut/branching adjustment.
18:             Create l' = (u, S union {e}, q + d_e, rc').
19:             Insert l' if it is not dominated.
20:         end if
21:     end for

22:     if v is the depot, S is nonempty, and rc < -epsilon then
23:         Reconstruct the corresponding route r.
24:         Add r to Omega_new.
25:     end if
26:  end while

27:  Remove duplicate or coefficient-dominated columns from Omega_new.
28:  return Omega_new.
```

In A-RMP mode, the context is day-based and the vehicle index is omitted. In the vehicle-indexed fallback RMP, the same pricing logic is applied to each \((t,k)\) context.

## Algorithm 4. SRI-3 Separation

The SRI-3 cut is separated after column generation converges and before branching. For a required-edge subset \(S\subseteq E^R\) with \(|S|=3\), the aggregated day-level inequality is

\[
\sum_{r:\, |R_r\cap S|\ge 2} \lambda_r^t \le 1,
\]

where \(R_r\) is the set of required edges serviced by route \(r\).

**Input:** Converged LP solution \(\lambda\), day set \(T\), required edges \(E^R\), violation tolerance \(\tau\).  
**Output:** A set of violated SRI-3 cuts.

```text
1:  Initialize an empty cut set C.

2:  for each day t in T do
3:      Collect all route columns r with lambda_r^t > 0.

4:      for each triple S = {e1, e2, e3} of required edges do
5:          lhs <- 0.
6:          for each positive route column r on day t do
7:              if route r services at least two edges in S then
8:                  lhs <- lhs + lambda_r^t.
9:              end if
10:         end for

11:         if lhs > 1 + tau then
12:             Add the cut sum_{r: |R_r intersect S| >= 2} lambda_r^t <= 1 to C.
13:         end if
14:     end for
15: end for

16: Select a limited subset of cuts from C using violation magnitude,
    per-day limits, and similarity filtering.
17: Add the selected cuts to the RMP.
18: return the selected cuts.
```

After any SRI-3 cut is added, the algorithm returns to column generation because the new dual prices may make additional route columns attractive.

## Algorithm 5. q-load Initial Column Heuristic

The q-load heuristic constructs initial A-RMP route columns before the first pricing iteration. It is always executed when the aggregated master is initialized. ALNS columns, if enabled, are added separately before this heuristic; q-load does not depend on ALNS.

The heuristic first chooses one schedule pattern for each required edge so that aggregate day loads are balanced. Then, for each day, it packs the active required edges into capacity-feasible route groups.

**Input:** Days \(T\), required edges \(E^R\), feasible schedule patterns \(\mathcal{P}_e\), demands \(d_e\), capacity \(Q\), number of vehicles \(|K|\), depot \(o\).  
**Output:** Initial aggregated route column set \(\Omega_0\).

```text
1:  Initialize q_load(t) <- 0 for all t in T.
2:  Initialize active_edges(t) <- empty set for all t in T.
3:  Initialize selected pattern p_e <- null for all e in E^R.

4:  Sort required edges in nonincreasing order of demand.

5:  for each required edge e in the sorted order do
6:      best_score <- +infinity.
7:      best_pattern <- null.

8:      for each feasible schedule pattern p in P_e do
9:          Compute tentative day loads:
                q'_load(t) = q_load(t) + d_e if t in p,
                             q_load(t)       otherwise.

10:         Compute the score of p:
                overflow  = sum_t max{0, q'_load(t) - Q|K|}
                max_ratio = max_t q'_load(t)/(Q|K|)
                sq_ratio  = sum_t (q'_load(t)/(Q|K|))^2
                span      = max_t q'_load(t) - min_t q'_load(t)

11:         score(p) <- lexicographic tuple
                (overflow, max_ratio, sq_ratio, span, |p|, pattern index).

12:         if score(p) is better than best_score then
13:             best_score <- score(p).
14:             best_pattern <- p.
15:         end if
16:     end for

17:     Set p_e <- best_pattern.
18:     for each day t in p_e do
19:         q_load(t) <- q_load(t) + d_e.
20:         active_edges(t) <- active_edges(t) union {e}.
21:     end for
22: end for

23: Initialize Omega_0 <- empty set.

24: for each day t in T do
25:     Pack active_edges(t) into route groups by best-fit decreasing:
26:         each group must have total demand at most Q.

27:     for each route group G do
28:         Construct a depot-to-depot route that services all edges in G.
29:         Add the corresponding aggregated route column lambda_r^t to Omega_0.
30:     end for
31: end for

32: return Omega_0.
```

The q-load heuristic is a root column scaffold rather than a proof of feasibility. If the resulting columns do not fully support all cover constraints, Phase-I artificial variables keep the initial RMP feasible with a large penalty, and subsequent pricing iterations generate improving columns.

## Algorithm 6. Branching Rule

Branching is performed only after column generation converges and no violated SRI-3 cut is added. In A-RMP mode, branching uses aggregated objects.

**Input:** Converged fractional LP solution at node \(n\).  
**Output:** Two child nodes.

```text
1:  Extract candidate families in hierarchical order:
2:      whole-route mass;
3:      daily-route mass;
4:      schedule-pattern variable q_{e,p};
5:      Ryan-Foster required-edge pair expression;
6:      individual lambda variable.

7:  Let C be the first nonempty candidate family.
8:  Select c in C by the branching ranking rule:
9:      binary-like candidates: prefer values closest to 0.5;
10:     integer-split candidates: prefer values farthest from the nearest integer.

11: if c is binary-like then
12:     Create left child with expression(c) <= 0.
13:     Create right child with expression(c) >= 1.
14: else
15:     Create left child with expression(c) <= floor(value(c)).
16:     Create right child with expression(c) >= ceil(value(c)).
17: end if

18: return the two child nodes.
```
