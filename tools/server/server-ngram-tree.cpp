#include "server-ngram-tree.h"

#include <algorithm>
#include <map>
#include <queue>
#include <tuple>
#include <utility>

namespace {

struct working_node {
    llama_token token = LLAMA_TOKEN_NULL;
    int32_t parent = -1;
    uint32_t depth = 0;
    uint64_t weight = 0;

    // map keeps insertion/build order deterministic.
    std::map<llama_token, int32_t> children;
};

struct frontier_item {
    uint64_t weight = 0;
    uint32_t depth = 0;
    llama_token token = LLAMA_TOKEN_NULL;
    int32_t working_id = -1;
    int32_t output_parent = -1;
};

// Highest aggregate frequency first.
// For equal weight, prefer shallower nodes and then lower token/id values
// only to make the result deterministic.
struct frontier_less {
    bool operator()(
            const frontier_item & a,
            const frontier_item & b) const {
        if (a.weight != b.weight) {
            return a.weight < b.weight;
        }
        if (a.depth != b.depth) {
            return a.depth > b.depth;
        }
        if (a.token != b.token) {
            return a.token > b.token;
        }
        return a.working_id > b.working_id;
    }
};

} // namespace

server_ngram_candidate_tree server_ngram_build_candidate_tree(
        llama_token root_token,
        const server_ngram_lookup_result & lookup,
        size_t max_nodes,
        size_t max_depth,
        llama_token stop_token) {
    server_ngram_candidate_tree out;
    out.matched_prefix_length = lookup.matched_prefix_length;
    out.source_suffix_count = lookup.suffixes.size();

    // A verification round always needs the root node.
    if (root_token == LLAMA_TOKEN_NULL || max_nodes == 0) {
        return out;
    }

    out.nodes.push_back({
        root_token,
        -1,
        0,
        0,
        {}
    });

    if (max_nodes == 1 ||
            max_depth == 0 ||
            lookup.suffixes.empty()) {
        return out;
    }

    // First build the complete weighted prefix tree.
    //
    // Every occurrence contributes its n-gram frequency to every prefix
    // node along that suffix. This is the paper's cnt_map aggregation,
    // represented directly as a tree rather than as tuple keys.
    std::vector<working_node> work;
    work.push_back({
        root_token,
        -1,
        0,
        0,
        {}
    });

    for (const auto & suffix : lookup.suffixes) {
        if (suffix.tokens.empty() || suffix.frequency == 0) {
            continue;
        }

        int32_t parent = 0;
        const size_t n =
                std::min(max_depth, suffix.tokens.size());

        for (size_t i = 0; i < n; ++i) {
            const llama_token token = suffix.tokens[i];

            auto it = work[parent].children.find(token);

            int32_t node_id;
            if (it == work[parent].children.end()) {
                node_id = static_cast<int32_t>(work.size());

                working_node node;
                node.token = token;
                node.parent = parent;
                node.depth = static_cast<uint32_t>(i + 1);

                work.push_back(std::move(node));
                work[parent].children.emplace(token, node_id);
            } else {
                node_id = it->second;
            }

            work[node_id].weight += suffix.frequency;
            parent = node_id;

            // Do not spend verification budget on descendants after EOS
            // or another configured terminal token.
            if (stop_token != LLAMA_TOKEN_NULL &&
                    token == stop_token) {
                break;
            }
        }
    }

    if (work.size() <= 1) {
        return out;
    }

    // Best-first expansion gives the same intended behavior as the paper's
    // high-frequency prefix pruning, but enforces two invariants explicitly:
    //
    // 1. strict prefix closure:
    //    a child can never exist without its parent;
    //
    // 2. exact node budget:
    //    out.nodes.size() never exceeds max_nodes.
    //
    // This matters for llama.cpp because every selected node becomes an
    // actual batch/KV verification node.
    std::priority_queue<
            frontier_item,
            std::vector<frontier_item>,
            frontier_less> frontier;

    for (const auto & child : work[0].children) {
        const int32_t id = child.second;
        frontier.push({
            work[id].weight,
            work[id].depth,
            work[id].token,
            id,
            0
        });
    }

    while (!frontier.empty() &&
            out.nodes.size() < max_nodes) {
        const frontier_item item = frontier.top();
        frontier.pop();

        if (item.working_id <= 0 ||
                static_cast<size_t>(item.working_id) >= work.size() ||
                item.output_parent < 0 ||
                static_cast<size_t>(item.output_parent) >= out.nodes.size()) {
            continue;
        }

        const auto & src = work[item.working_id];

        const int32_t out_id =
                static_cast<int32_t>(out.nodes.size());

        server_ngram_tree_node node;
        node.token = src.token;
        node.parent = item.output_parent;
        node.depth = src.depth;
        node.weight = src.weight;

        out.nodes.push_back(std::move(node));
        out.nodes[item.output_parent].children.push_back(out_id);

        for (const auto & child : src.children) {
            const int32_t child_id = child.second;
            frontier.push({
                work[child_id].weight,
                work[child_id].depth,
                work[child_id].token,
                child_id,
                out_id
            });
        }
    }

    return out;
}


bool server_ngram_build_verification_plan(
        const server_ngram_candidate_tree & tree,
        size_t max_sequences,
        server_ngram_verification_plan & out) {
    out = {};

    if (tree.nodes.empty() || max_sequences < 2) {
        return false;
    }

    out.node_seq_ids.resize(tree.nodes.size());

    // One temporary sequence for every real leaf.
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        if (tree.nodes[i].children.empty()) {
            out.leaf_nodes.push_back(static_cast<int32_t>(i));
        }
    }

    if (out.leaf_nodes.empty() ||
            out.leaf_nodes.size() + 1 > max_sequences) {
        out = {};
        return false;
    }

    for (size_t leaf_index = 0;
            leaf_index < out.leaf_nodes.size();
            ++leaf_index) {
        const llama_seq_id seq_id =
                static_cast<llama_seq_id>(leaf_index + 1);

        int32_t node = out.leaf_nodes[leaf_index];

        // Add this leaf sequence to the leaf and every ancestor up to root.
        while (node >= 0) {
            if (static_cast<size_t>(node) >= tree.nodes.size()) {
                out = {};
                return false;
            }

            out.node_seq_ids[node].push_back(seq_id);
            node = tree.nodes[node].parent;
        }
    }

    // Every verification node must belong to at least one concrete leaf seq.
    for (const auto & seqs : out.node_seq_ids) {
        if (seqs.empty()) {
            out = {};
            return false;
        }
    }

    return true;
}
