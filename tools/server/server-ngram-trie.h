#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

// One suffix returned by the N-Gram Trie.
//
// `tokens` are the continuation after the matched prefix.
// `frequency` is the occurrence count of the source n-gram in the Prompt.
struct server_ngram_suffix {
    std::vector<llama_token> tokens;
    uint32_t frequency = 0;
};

struct server_ngram_lookup_result {
    std::vector<server_ngram_suffix> suffixes;

    // Number of newest history tokens matched by the reversed-prefix Trie.
    // The reference implementation normally reaches 3, but can fall back
    // to a shorter prefix through longest-prefix matching.
    uint32_t matched_prefix_length = 0;
};

// Prompt-local N-Gram Trie used by speculative Generation.
//
// This follows the N-Gram Trie reference implementation:
//   - n-gram size      = 13
//   - prefix length    = 3
//   - lookup window    = 4 newest tokens
//   - prefix is stored in reverse order
//   - every Trie prefix stores the suffix values below it, which allows
//     longest-prefix fallback from 3 tokens to 2/1 tokens.
//
// The Trie is static for one Generation request. It is built from the
// transferred Prompt once and is queried with Prompt + committed generated
// history + the current pending/root token.
class server_ngram_trie {
public:
    server_ngram_trie(
            uint32_t ngram_size = 13,
            uint32_t prefix_length = 3,
            uint32_t lookup_window = 4);

    void clear();

    // Build from one tokenized Prompt. `eos_token` is used by the same
    // beginning-of-sequence padding logic as the reference implementation.
    //
    // Returns false only for an invalid configuration.
    bool build(
            const std::vector<llama_token> & prompt,
            llama_token eos_token);

    // Look up suffix candidates after appending `pending_token` to `history`.
    // `history` must NOT already contain pending_token.
    server_ngram_lookup_result lookup(
            const std::vector<llama_token> & history,
            llama_token pending_token) const;

    size_t entry_count() const noexcept;
    size_t node_count() const noexcept;

    uint32_t ngram_size() const noexcept;
    uint32_t prefix_length() const noexcept;

private:
    struct trie_value {
        std::vector<llama_token> suffix;
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
            const std::vector<llama_token> & gram,
            uint32_t frequency);
};
