#define cpp_price_dp cpp_price_dp_original
#define cpp_price_ng cpp_price_ng_original
#include "cpp_pricing_core.cpp"
#undef cpp_price_dp
#undef cpp_price_ng

#include <cmath>
#include <unordered_map>
#include <vector>

static constexpr double kLoadDomEps = 1e-9;

static inline bool load_leq(double lhs, double rhs) {
    return lhs <= rhs + kLoadDomEps;
}

static inline bool load_lt(double lhs, double rhs) {
    return lhs < rhs - kLoadDomEps;
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
    return cpp_price_dp_original(
        num_nodes,
        num_reqs,
        num_svc,
        depot_idx,
        capacity,
        vehicle_dual,
        eps_rc,
        max_columns,
        svc_req_idx,
        svc_from,
        svc_to,
        svc_travel,
        svc_service,
        svc_demand,
        req_dual,
        sp_cost,
        max_steps_total,
        out_col_count,
        out_rc,
        out_offsets,
        out_req_idx,
        out_svc_u,
        out_svc_v
    );
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
    std::vector<double> min_svc_cost(
        static_cast<size_t>(num_reqs),
        std::numeric_limits<double>::infinity()
    );
    std::vector<double> req_demand_lb(
        static_cast<size_t>(num_reqs),
        std::numeric_limits<double>::infinity()
    );
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
            const double cand =
                best_save_by_cap[static_cast<size_t>(c - dem_i)] + delta;
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

    std::unordered_map<int, std::vector<int>> labels_by_node;
    labels_by_node.reserve(static_cast<size_t>(std::max(16, num_nodes * 2)));
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
        labels_by_node[root.node].push_back(0);
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
            const std::vector<int> cur_forbid =
                states[static_cast<size_t>(cur_idx)].forbid;
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

                    const double new_rc =
                        cur_rc + dead + svc_travel[s] + svc_service[s] - req_dual[r];
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

                    auto& node_bucket = labels_by_node[to];
                    bool insert_label = true;
                    std::vector<int> dominated_existing;

                    // Safe strengthening:
                    // same node + no larger load + no worse reduced cost +
                    // less restrictive ng forbid-set  => every feasible continuation
                    // of the new label is feasible for the old label with no worse RC.
                    for (int old_idx : node_bucket) {
                        if (old_idx < 0 || old_idx >= static_cast<int>(states.size()))
                            continue;
                        NgState& old = states[static_cast<size_t>(old_idx)];
                        if (!old.active)
                            continue;

                        const bool old_dominates =
                            load_leq(old.load, new_load) &&
                            old.rc <= new_rc + dom_eps &&
                            is_subset_sorted(old.forbid, new_forbid) &&
                            (
                                load_lt(old.load, new_load) ||
                                old.rc < new_rc - dom_eps ||
                                old.forbid != new_forbid
                            );
                        if (old_dominates) {
                            insert_label = false;
                            break;
                        }

                        const bool new_dominates =
                            load_leq(new_load, old.load) &&
                            new_rc <= old.rc + dom_eps &&
                            is_subset_sorted(new_forbid, old.forbid) &&
                            (
                                load_lt(new_load, old.load) ||
                                new_rc < old.rc - dom_eps ||
                                new_forbid != old.forbid
                            );
                        if (new_dominates)
                            dominated_existing.push_back(old_idx);
                    }
                    if (!insert_label)
                        continue;

                    for (int old_idx : dominated_existing) {
                        if (old_idx >= 0 && old_idx < static_cast<int>(states.size()))
                            states[static_cast<size_t>(old_idx)].active = false;
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
                    node_bucket.push_back(ni);
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
