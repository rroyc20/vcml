#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

// ── [4] Stronger hash: FNV-1a style mix ──────────────────────────────────────
static inline uint64_t pack_key(uint64_t mask, int node) {
    // FNV-1a inspired: mix mask and node without truncating high mask bits.
    uint64_t h = mask;
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= static_cast<uint64_t>(node) * UINT64_C(0x9e3779b97f4a7c15);
    h ^= h >> 33;
    return h;
}

struct State {
    uint64_t mask;
    int      node;
    double   rc;
    double   load;
    int      prev_idx;
    int      svc_idx;
};

struct Candidate {
    int    state_idx;
    double total_rc;
};

struct NgState {
    int              node;
    double           load;
    int64_t          load_key;
    double           rc;
    std::vector<int> forbid;
    int              prev_idx;
    int              svc_idx;
    int              len;
    bool             active;
};

static constexpr double kLoadKeyScale = 1e6;

static inline int64_t quantize_load(double load) {
    return static_cast<int64_t>(std::llround(load * kLoadKeyScale));
}

static inline uint64_t pack_bucket_key(int64_t load_key, int node) {
    uint64_t h = static_cast<uint64_t>(load_key);
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= static_cast<uint64_t>(node) * UINT64_C(0x9e3779b97f4a7c15);
    h ^= h >> 33;
    return h;
}

static inline bool is_subset_sorted(
    const std::vector<int>& lhs,
    const std::vector<int>& rhs
) {
    return std::includes(rhs.begin(), rhs.end(), lhs.begin(), lhs.end());
}

static inline std::vector<int> intersect_with_ng(
    const std::vector<int>& cur_forbid,
    const int*             ng_offsets,
    const int*             ng_neighbors,
    int                    req_idx
) {
    const int start = ng_offsets[req_idx];
    const int end   = ng_offsets[req_idx + 1];
    std::vector<int> out;
    out.reserve(static_cast<size_t>(end - start + 1));
    size_t i = 0;
    int j = start;
    while (i < cur_forbid.size() && j < end) {
        const int a = cur_forbid[i];
        const int b = ng_neighbors[j];
        if (a == b) {
            out.push_back(a);
            ++i;
            ++j;
        } else if (a < b) {
            ++i;
        } else {
            ++j;
        }
    }
    if (out.empty() || out.back() != req_idx) {
        auto pos = std::lower_bound(out.begin(), out.end(), req_idx);
        out.insert(pos, req_idx);
    }
    return out;
}

extern "C" int cpp_price_dp(
    int           num_nodes,
    int           num_reqs,
    int           num_svc,
    int           depot_idx,
    double        capacity,
    double        vehicle_dual,
    double        eps_rc,
    int           max_columns,
    const int*    svc_req_idx,
    const int*    svc_from,
    const int*    svc_to,
    const double* svc_travel,
    const double* svc_service,
    const double* svc_demand,
    const double* req_dual,
    const double* sp_cost,
    int           max_steps_total,
    int*          out_col_count,
    double*       out_rc,
    int*          out_offsets,
    int*          out_req_idx,
    int*          out_svc_u,
    int*          out_svc_v
) {
    if (num_nodes <= 0 || num_reqs <= 0 || num_svc <= 0 || max_columns <= 0) {
        *out_col_count = 0;
        if (out_offsets) out_offsets[0] = 0;
        return 0;
    }
    if (num_reqs >= 62) {
        return -10;  // bitmask overflow guard
    }

    // ── [3] Completion bound: precompute per-req best achievable dual saving ─
    // best_delta[r] = max(0, req_dual[r] - min service cost for req r)
    // Used to compute an upper bound on the dual savings reachable from the
    // remaining unserviced requirements.
    std::vector<double> best_delta(static_cast<size_t>(num_reqs), 0.0);
    {
        // For each req, min_svc_cost = min over service arcs of (travel + service)
        std::vector<double> min_svc(static_cast<size_t>(num_reqs),
                                    std::numeric_limits<double>::infinity());
        for (int s = 0; s < num_svc; ++s) {
            int r = svc_req_idx[s];
            if (r >= 0 && r < num_reqs) {
                double c = svc_travel[s] + svc_service[s];
                if (c < min_svc[static_cast<size_t>(r)])
                    min_svc[static_cast<size_t>(r)] = c;
            }
        }
        for (int r = 0; r < num_reqs; ++r) {
            double saving = req_dual[r] - min_svc[static_cast<size_t>(r)];
            best_delta[static_cast<size_t>(r)] = (saving > 0.0) ? saving : 0.0;
        }
    }
    // Precompute prefix sums over best_delta to quickly compute the upper bound
    // for a bitmask of remaining requirements (not in mask).
    // total_best_saving(mask) = sum of best_delta[r] for all r NOT in mask.
    // We approximate this as: total_all_saving - saving_already_collected(mask).
    double total_best_saving = 0.0;
    for (int r = 0; r < num_reqs; ++r)
        total_best_saving += best_delta[static_cast<size_t>(r)];

    std::vector<std::vector<int>> svc_by_req(static_cast<size_t>(num_reqs));
    for (int s = 0; s < num_svc; ++s) {
        int r = svc_req_idx[s];
        if (r >= 0 && r < num_reqs) {
            svc_by_req[static_cast<size_t>(r)].push_back(s);
        }
    }

    std::vector<State> states;
    states.reserve(8192);
    std::unordered_map<uint64_t, int> best_idx;
    best_idx.reserve(16384);

    auto cmp = [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first > b.first;
    };
    std::priority_queue<
        std::pair<double, int>,
        std::vector<std::pair<double, int>>,
        decltype(cmp)
    > pq(cmp);

    {
        State root{};
        root.mask     = 0ULL;
        root.node     = depot_idx;
        root.rc       = -vehicle_dual;
        root.load     = 0.0;
        root.prev_idx = -1;
        root.svc_idx  = -1;
        states.push_back(root);
        best_idx[pack_key(root.mask, root.node)] = 0;
        pq.push({root.rc, 0});
    }

    const uint64_t full_mask = (num_reqs < 64)
        ? ((1ULL << static_cast<uint64_t>(num_reqs)) - 1ULL)
        : ~0ULL;

    while (!pq.empty()) {
        auto [cur_rc, cur_idx] = pq.top();
        pq.pop();

        if (cur_idx < 0 || cur_idx >= static_cast<int>(states.size()))
            continue;

        // ── [1] Copy state fields to locals BEFORE any push_back ─────────────
        // Prevents dangling reference if states vector reallocates.
        const uint64_t cur_mask = states[static_cast<size_t>(cur_idx)].mask;
        const int      cur_node = states[static_cast<size_t>(cur_idx)].node;
        const double   cur_rc_v = states[static_cast<size_t>(cur_idx)].rc;
        const double   cur_load = states[static_cast<size_t>(cur_idx)].load;

        // Stale-entry check (lazy deletion)
        if (cur_rc > cur_rc_v + 1e-12)
            continue;

        // ── [3] Completion bound pruning ────────────────────────────────────
        // Best depot return cost from current node
        double back_min = sp_cost[cur_node * num_nodes + depot_idx];
        if (!std::isfinite(back_min))
            continue;  // unreachable depot

        // Compute already-collected dual saving from cur_mask
        double saving_collected = 0.0;
        {
            uint64_t m = cur_mask;
            while (m) {
                int r = __builtin_ctzll(m);
                saving_collected += best_delta[static_cast<size_t>(r)];
                m &= m - 1;
            }
        }
        double best_remaining_saving = total_best_saving - saving_collected;
        // Optimistic total_rc if we collect all remaining savings and return:
        // cur_rc_v - best_remaining_saving + back_min
        // If this is already >= eps_rc, this entire subtree is pruned.
        if (cur_rc_v - best_remaining_saving + back_min >= -eps_rc)
            continue;

        // ── [2] Iterate only UNVISITED required edges via bit manipulation ───
        uint64_t remaining_bits = (~cur_mask) & full_mask;
        while (remaining_bits) {
            // Extract lowest set bit position = index of next unvisited req
            int r = __builtin_ctzll(remaining_bits);
            remaining_bits &= remaining_bits - 1;  // clear this bit

            const uint64_t bit = (1ULL << static_cast<uint64_t>(r));
            const auto& svc_list = svc_by_req[static_cast<size_t>(r)];

            for (int s : svc_list) {
                double new_load = cur_load + svc_demand[s];
                if (new_load > capacity + 1e-9)
                    continue;

                int from = svc_from[s];
                int to   = svc_to[s];
                if (from < 0 || from >= num_nodes || to < 0 || to >= num_nodes)
                    continue;

                double dead = sp_cost[cur_node * num_nodes + from];
                if (!std::isfinite(dead))
                    continue;

                uint64_t new_mask = cur_mask | bit;
                double   new_rc   = cur_rc_v + dead + svc_travel[s] + svc_service[s]
                                    - req_dual[r];
                uint64_t key = pack_key(new_mask, to);
                auto it = best_idx.find(key);
                if (it == best_idx.end()) {
                    State ns{};
                    ns.mask     = new_mask;
                    ns.node     = to;
                    ns.rc       = new_rc;
                    ns.load     = new_load;
                    ns.prev_idx = cur_idx;
                    ns.svc_idx  = s;
                    int ni = static_cast<int>(states.size());
                    states.push_back(ns);   // safe: cur_* are local copies
                    best_idx[key] = ni;
                    pq.push({new_rc, ni});
                } else {
                    State& old = states[static_cast<size_t>(it->second)];
                    if (new_rc + 1e-12 < old.rc) {
                        old.rc       = new_rc;
                        old.load     = new_load;
                        old.prev_idx = cur_idx;
                        old.svc_idx  = s;
                        pq.push({new_rc, it->second});
                    }
                }
            }
        }
    }

    std::vector<Candidate> cands;
    cands.reserve(states.size());
    for (int i = 1; i < static_cast<int>(states.size()); ++i) {
        const State& st = states[static_cast<size_t>(i)];
        if (st.mask == 0ULL || st.node < 0 || st.node >= num_nodes)
            continue;
        double back = sp_cost[st.node * num_nodes + depot_idx];
        if (!std::isfinite(back))
            continue;
        double total_rc = st.rc + back;
        if (total_rc < -eps_rc)
            cands.push_back(Candidate{i, total_rc});
    }

    std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
        return a.total_rc < b.total_rc;
    });
    if (static_cast<int>(cands.size()) > max_columns)
        cands.resize(static_cast<size_t>(max_columns));

    int write_pos   = 0;
    out_offsets[0]  = 0;
    int col_count   = 0;

    for (const auto& cand : cands) {
        int idx = cand.state_idx;
        std::vector<int>               req_seq;
        std::vector<std::pair<int,int>> svc_seq;
        req_seq.reserve(static_cast<size_t>(num_reqs));
        svc_seq.reserve(static_cast<size_t>(num_reqs));

        while (idx > 0) {
            const State& st = states[static_cast<size_t>(idx)];
            int s = st.svc_idx;
            if (s < 0 || s >= num_svc)
                break;
            req_seq.push_back(svc_req_idx[s]);
            svc_seq.push_back({svc_from[s], svc_to[s]});
            idx = st.prev_idx;
            if (idx < 0)
                break;
        }

        if (req_seq.empty() || svc_seq.empty())
            continue;

        std::reverse(req_seq.begin(), req_seq.end());
        std::reverse(svc_seq.begin(), svc_seq.end());

        if (write_pos + static_cast<int>(req_seq.size()) > max_steps_total)
            return -20;  // output buffer too small

        out_rc[col_count] = cand.total_rc;
        for (int i = 0; i < static_cast<int>(req_seq.size()); ++i) {
            out_req_idx[write_pos] = req_seq[static_cast<size_t>(i)];
            out_svc_u[write_pos]   = svc_seq[static_cast<size_t>(i)].first;
            out_svc_v[write_pos]   = svc_seq[static_cast<size_t>(i)].second;
            write_pos += 1;
        }
        col_count += 1;
        out_offsets[col_count] = write_pos;
    }

    *out_col_count = col_count;
    return 0;
}

extern "C" int cpp_price_ng(
    int           num_nodes,
    int           num_reqs,
    int           num_svc,
    int           depot_idx,
    double        capacity,
    double        vehicle_dual,
    double        eps_rc,
    int           max_columns,
    const int*    svc_req_idx,
    const int*    svc_from,
    const int*    svc_to,
    const double* svc_travel,
    const double* svc_service,
    const double* svc_demand,
    const double* req_dual,
    const double* sp_cost,
    const int*    ng_offsets,
    const int*    ng_neighbors,
    int           max_steps_total,
    int*          out_col_count,
    double*       out_rc,
    int*          out_offsets,
    int*          out_req_idx,
    int*          out_svc_u,
    int*          out_svc_v
) {
    if (num_nodes <= 0 || num_reqs <= 0 || num_svc <= 0 || max_columns <= 0) {
        *out_col_count = 0;
        if (out_offsets) out_offsets[0] = 0;
        return 0;
    }

    std::vector<std::vector<int>> svc_by_req(static_cast<size_t>(num_reqs));
    std::vector<double> min_svc_cost(static_cast<size_t>(num_reqs),
                                     std::numeric_limits<double>::infinity());
    std::vector<double> req_demand_lb(static_cast<size_t>(num_reqs),
                                      std::numeric_limits<double>::infinity());
    for (int s = 0; s < num_svc; ++s) {
        const int r = svc_req_idx[s];
        if (r >= 0 && r < num_reqs) {
            svc_by_req[static_cast<size_t>(r)].push_back(s);
            const double svc_cost = svc_travel[s] + svc_service[s];
            if (svc_cost < min_svc_cost[static_cast<size_t>(r)])
                min_svc_cost[static_cast<size_t>(r)] = svc_cost;
            if (svc_demand[s] < req_demand_lb[static_cast<size_t>(r)])
                req_demand_lb[static_cast<size_t>(r)] = svc_demand[s];
        }
    }

    const double dom_eps = std::max(1e-9, eps_rc * 0.1);
    bool use_int_completion_bound =
        std::fabs(capacity - std::llround(capacity)) <= 1e-9 &&
        capacity >= 0.0 &&
        capacity <= 200000.0;
    double best_save_ratio = 0.0;
    int cap_int = -1;
    std::vector<double> best_save_by_cap;

    if (use_int_completion_bound) {
        cap_int = static_cast<int>(std::llround(capacity));
        best_save_by_cap.assign(static_cast<size_t>(cap_int + 1), 0.0);
    }

    for (int r = 0; r < num_reqs; ++r) {
        const double dem = req_demand_lb[static_cast<size_t>(r)];
        const double svc_cost = min_svc_cost[static_cast<size_t>(r)];
        if (!std::isfinite(dem) || dem <= 1e-12 || !std::isfinite(svc_cost))
            continue;

        const double delta = req_dual[r] - svc_cost;
        if (delta <= 1e-12)
            continue;

        best_save_ratio = std::max(best_save_ratio, delta / dem);
        if (!use_int_completion_bound)
            continue;
        if (std::fabs(dem - std::llround(dem)) > 1e-9) {
            use_int_completion_bound = false;
            best_save_by_cap.clear();
            continue;
        }
        const int dem_i = static_cast<int>(std::llround(dem));
        if (dem_i <= 0 || dem_i > cap_int)
            continue;
        for (int c = dem_i; c <= cap_int; ++c) {
            const double cand = best_save_by_cap[static_cast<size_t>(c - dem_i)] + delta;
            if (cand > best_save_by_cap[static_cast<size_t>(c)])
                best_save_by_cap[static_cast<size_t>(c)] = cand;
        }
    }

    auto optimistic_remaining_saving = [&](double rem_cap) -> double {
        if (rem_cap <= 1e-12)
            return 0.0;
        if (use_int_completion_bound && !best_save_by_cap.empty()) {
            int rem_i = static_cast<int>(std::floor(rem_cap + 1e-9));
            if (rem_i < 0)
                rem_i = 0;
            if (rem_i > cap_int)
                rem_i = cap_int;
            return best_save_by_cap[static_cast<size_t>(rem_i)];
        }
        return rem_cap * best_save_ratio;
    };

    std::vector<NgState> states;
    states.reserve(8192);

    std::unordered_map<uint64_t, std::vector<int>> bucket_by_qnode;
    bucket_by_qnode.reserve(16384);
    std::map<int64_t, std::vector<int>> labels_by_load;

    {
        NgState root{};
        root.node     = depot_idx;
        root.load     = 0.0;
        root.load_key = quantize_load(0.0);
        root.rc       = -vehicle_dual;
        root.prev_idx = -1;
        root.svc_idx  = -1;
        root.len      = 0;
        root.active   = true;
        states.push_back(root);
        labels_by_load[root.load_key].push_back(0);
        bucket_by_qnode[pack_bucket_key(root.load_key, root.node)].push_back(0);
    }

    for (auto it = labels_by_load.begin(); it != labels_by_load.end(); ++it) {
        const int64_t cur_load_key = it->first;
        size_t pos = 0;
        while (pos < labels_by_load[cur_load_key].size()) {
            const int cur_idx = labels_by_load[cur_load_key][pos++];
            if (cur_idx < 0 || cur_idx >= static_cast<int>(states.size()))
                continue;
            if (!states[static_cast<size_t>(cur_idx)].active)
                continue;

            const int cur_node = states[static_cast<size_t>(cur_idx)].node;
            const double cur_load = states[static_cast<size_t>(cur_idx)].load;
            const double cur_rc = states[static_cast<size_t>(cur_idx)].rc;
            const std::vector<int> cur_forbid = states[static_cast<size_t>(cur_idx)].forbid;
            const int cur_len = states[static_cast<size_t>(cur_idx)].len;

            if (cur_len > 0 && cur_node == depot_idx)
                continue;
            if (cur_len >= max_steps_total)
                continue;

            for (int r = 0; r < num_reqs; ++r) {
                if (std::binary_search(cur_forbid.begin(), cur_forbid.end(), r))
                    continue;

                const auto& svc_list = svc_by_req[static_cast<size_t>(r)];
                if (svc_list.empty())
                    continue;

                for (int s : svc_list) {
                    const double new_load = cur_load + svc_demand[s];
                    if (new_load > capacity + 1e-9)
                        continue;

                    const int from = svc_from[s];
                    const int to = svc_to[s];
                    if (from < 0 || from >= num_nodes || to < 0 || to >= num_nodes)
                        continue;

                    const double dead = sp_cost[cur_node * num_nodes + from];
                    if (!std::isfinite(dead))
                        continue;

                    const double new_rc = cur_rc + dead + svc_travel[s] + svc_service[s] - req_dual[r];
                    const int64_t new_load_key = quantize_load(new_load);
                    const double rem_after = capacity - new_load;
                    if (new_rc - optimistic_remaining_saving(rem_after) >= -eps_rc)
                        continue;
                    const std::vector<int> new_forbid = intersect_with_ng(
                        cur_forbid,
                        ng_offsets,
                        ng_neighbors,
                        r
                    );
                    const uint64_t bucket_key = pack_bucket_key(new_load_key, to);
                    auto& bucket = bucket_by_qnode[bucket_key];

                    bool insert_label = true;
                    std::vector<int> dominated_existing;
                    for (int old_idx : bucket) {
                        if (old_idx < 0 || old_idx >= static_cast<int>(states.size()))
                            continue;
                        NgState& old = states[static_cast<size_t>(old_idx)];
                        if (!old.active)
                            continue;

                        const bool old_dominates =
                            old.rc <= new_rc + dom_eps &&
                            is_subset_sorted(old.forbid, new_forbid) &&
                            (old.rc < new_rc - dom_eps || old.forbid != new_forbid);
                        if (old_dominates) {
                            insert_label = false;
                            break;
                        }

                        const bool new_dominates =
                            new_rc <= old.rc + dom_eps &&
                            is_subset_sorted(new_forbid, old.forbid) &&
                            (new_rc < old.rc - dom_eps || new_forbid != old.forbid);
                        if (new_dominates) {
                            dominated_existing.push_back(old_idx);
                        }
                    }
                    if (!insert_label)
                        continue;

                    for (int old_idx : dominated_existing) {
                        if (old_idx >= 0 && old_idx < static_cast<int>(states.size())) {
                            states[static_cast<size_t>(old_idx)].active = false;
                        }
                    }

                    NgState ns{};
                    ns.node     = to;
                    ns.load     = new_load;
                    ns.load_key = new_load_key;
                    ns.rc       = new_rc;
                    ns.forbid   = new_forbid;
                    ns.prev_idx = cur_idx;
                    ns.svc_idx  = s;
                    ns.len      = cur_len + 1;
                    ns.active   = true;
                    const int ni = static_cast<int>(states.size());
                    states.push_back(ns);
                    bucket.push_back(ni);
                    labels_by_load[new_load_key].push_back(ni);
                }
            }
        }
    }

    std::vector<Candidate> cands;
    cands.reserve(states.size());
    for (int i = 1; i < static_cast<int>(states.size()); ++i) {
        const NgState& st = states[static_cast<size_t>(i)];
        if (!st.active || st.len <= 0)
            continue;
        if (st.node < 0 || st.node >= num_nodes)
            continue;
        const double back = sp_cost[st.node * num_nodes + depot_idx];
        if (!std::isfinite(back))
            continue;
        const double total_rc = st.rc + back;
        if (total_rc < -eps_rc)
            cands.push_back(Candidate{i, total_rc});
    }

    std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
        return a.total_rc < b.total_rc;
    });
    if (static_cast<int>(cands.size()) > max_columns)
        cands.resize(static_cast<size_t>(max_columns));

    int write_pos = 0;
    int col_count = 0;
    out_offsets[0] = 0;

    for (const auto& cand : cands) {
        int idx = cand.state_idx;
        std::vector<int> req_seq;
        std::vector<std::pair<int, int>> svc_seq;
        while (idx > 0) {
            const NgState& st = states[static_cast<size_t>(idx)];
            const int s = st.svc_idx;
            if (s < 0 || s >= num_svc)
                break;
            req_seq.push_back(svc_req_idx[s]);
            svc_seq.push_back({svc_from[s], svc_to[s]});
            idx = st.prev_idx;
            if (idx < 0)
                break;
        }

        if (req_seq.empty() || svc_seq.empty())
            continue;

        std::reverse(req_seq.begin(), req_seq.end());
        std::reverse(svc_seq.begin(), svc_seq.end());

        if (write_pos + static_cast<int>(req_seq.size()) > max_steps_total)
            return -20;

        out_rc[col_count] = cand.total_rc;
        for (int i = 0; i < static_cast<int>(req_seq.size()); ++i) {
            out_req_idx[write_pos] = req_seq[static_cast<size_t>(i)];
            out_svc_u[write_pos]   = svc_seq[static_cast<size_t>(i)].first;
            out_svc_v[write_pos]   = svc_seq[static_cast<size_t>(i)].second;
            write_pos += 1;
        }
        col_count += 1;
        out_offsets[col_count] = write_pos;
    }

    *out_col_count = col_count;
    return 0;
}
