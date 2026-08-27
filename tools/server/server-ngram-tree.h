#pragma once

#include "server-ngram-trie.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// One real prefix-tree node.
//
// node 0 is always the already-sampled pending/root token.
// Every other node represents one speculative token after the root.
//
// Unlike the Python reference's depth-wise token indexing, nodes are
// identified by their full parent path. Therefore the same token under
// two different parents remains two distinct verification nodes.
struct server_ngram_tree_node {
    llama_token token = LLAMA_TOKEN_NULL;

    int32_t parent = -1;
    uint32_t depth = 0;       // root=0, first speculative token=1, ...

    // Sum of source n-gram frequencies of all suffixes passing through
    // this prefix. This is the score used for frequency-first pruning.
    uint64_t weight = 0;

    std::vector<int32_t> children;
};

struct server_ngram_candidate_tree {
    std::vector<server_ngram_tree_node> nodes;

    uint32_t matched_prefix_length = 0;
    size_t source_suffix_count = 0;

    bool hit() const noexcept {
        return nodes.size() > 1;
    }

    size_t speculative_node_count() const noexcept {
        return nodes.empty() ? 0 : nodes.size() - 1;
    }
};

// Build a bounded multi-branch candidate tree.
//
// Semantics:
// - root_token is already sampled but not yet committed to Decode KV;
// - lookup suffixes begin AFTER root_token;
// - identical token prefixes are merged;
// - each prefix accumulates the frequencies of all source suffixes below it;
// - when the full tree exceeds max_nodes, nodes are selected by descending
//   aggregate prefix frequency while preserving strict prefix closure;
// - max_nodes INCLUDES the root;
// - max_depth counts speculative tokens after the root;
// - stop_token terminates a suffix path when encountered.
//
// The resulting node order is deterministic and parent-before-child, which
// is required by the later single-batch tree verifier.
server_ngram_candidate_tree server_ngram_build_candidate_tree(
        llama_token root_token,
        const server_ngram_lookup_result & lookup,
        size_t max_nodes,
        size_t max_depth,
        llama_token stop_token = LLAMA_TOKEN_NULL);

// Concrete sequence assignment for llama.cpp tree verification.
//
// seq 0 is reserved for committed history.
// Every candidate-tree leaf receives one temporary sequence id >= 1.
// Every tree node carries all leaf sequence ids in its subtree, so the
// built-in causal mask sees exactly history + that node's ancestors.
struct server_ngram_verification_plan {
    std::vector<std::vector<llama_seq_id>> node_seq_ids;
    std::vector<int32_t> leaf_nodes;

    // Number of temporary branch sequences. seq 0 is not included.
    size_t branch_count() const noexcept {
        return leaf_nodes.size();
    }
};

// max_sequences includes committed seq 0.
//
// Therefore a context with n_seq_max=32 can host at most 31 tree leaves.
bool server_ngram_build_verification_plan(
        const server_ngram_candidate_tree & tree,
        size_t max_sequences,
        server_ngram_verification_plan & out);
