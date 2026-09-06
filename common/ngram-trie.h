#pragma once

#include "llama.h"
#include "speculative.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

struct common_ngram_trie_suffix {
    llama_tokens tokens;
    uint32_t frequency = 0;
};

struct common_ngram_trie_lookup_result {
    std::vector<common_ngram_trie_suffix> suffixes;
    uint32_t matched_prefix_length = 0;
};

// Prompt-local longest-prefix N-Gram Trie draft source.
//
// B is supplied as common_speculative_tree_draft_params::n_paths and limits
// frequency-ranked suffix paths before shared-prefix merging. D is n_max and
// limits the number of speculative tokens retained from each path.
class common_ngram_trie {
public:
    common_ngram_trie(
            uint32_t ngram_size = 13,
            uint32_t prefix_length = 3,
            uint32_t lookup_window = 4);

    void clear();

    bool build(
            const llama_tokens & prompt,
            llama_token eos_token);

    common_ngram_trie_lookup_result lookup(
            const llama_tokens & history,
            llama_token pending_token) const;

    void draft_tree(
            const common_speculative_tree_draft_params & params,
            uint32_t min_prefix_length,
            llama_token stop_token,
            common_speculative_tree_draft & result) const;

    size_t entry_count() const noexcept;
    size_t node_count() const noexcept;

private:
    struct trie_value {
        llama_tokens suffix;
        uint32_t frequency = 0;
    };

    struct trie_node {
        std::map<llama_token, size_t> children;
        std::vector<size_t> value_ids;
    };

    uint32_t ngram_size_;
    uint32_t prefix_length_;
    uint32_t lookup_window_;

    std::vector<trie_node> nodes_;
    std::vector<trie_value> values_;

    void insert_full_ngram(
            const llama_tokens & gram,
            uint32_t frequency);
};
