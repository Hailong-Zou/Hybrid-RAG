#include "server-ngram-trie.h"

#include <algorithm>
#include <map>
#include <utility>

server_ngram_trie::server_ngram_trie(
        uint32_t ngram_size,
        uint32_t prefix_length,
        uint32_t lookup_window)
    : ngram_size_(ngram_size),
      prefix_length_(prefix_length),
      lookup_window_(lookup_window) {
    clear();
}

void server_ngram_trie::clear() {
    nodes_.clear();
    nodes_.emplace_back(); // root
    values_.clear();
}

bool server_ngram_trie::build(
        const std::vector<llama_token> & prompt,
        llama_token eos_token) {
    clear();

    if (ngram_size_ == 0 ||
            prefix_length_ == 0 ||
            prefix_length_ >= ngram_size_ ||
            lookup_window_ == 0) {
        return false;
    }

    const size_t n = static_cast<size_t>(ngram_size_);
    const size_t prefix_len =
            static_cast<size_t>(prefix_length_);
    const size_t suffix_len = n - prefix_len;

    // The paper implementation treats the whole Prompt as one token
    // sequence and appends EOS before creating its n-grams.
    std::vector<llama_token> source = prompt;
    if (eos_token != LLAMA_TOKEN_NULL) {
        source.push_back(eos_token);
    }

    // Count complete n-grams first. std::map keeps construction deterministic.
    std::map<std::vector<llama_token>, uint32_t> frequencies;

    // Preserve the reference implementation's special sequence-prefix
    // entries. With n=13 and prefix_len=3 these are the first 11/12 tokens,
    // later padded with EOS into complete 13-token entries.
    for (size_t k = suffix_len + 1; k < n; ++k) {
        if (source.size() >= k) {
            std::vector<llama_token> gram(
                    source.begin(),
                    source.begin() + static_cast<ptrdiff_t>(k));
            ++frequencies[std::move(gram)];
        }
    }

    if (source.size() >= n) {
        for (size_t i = 0; i + n <= source.size(); ++i) {
            std::vector<llama_token> gram(
                    source.begin() + static_cast<ptrdiff_t>(i),
                    source.begin() + static_cast<ptrdiff_t>(i + n));
            ++frequencies[std::move(gram)];
        }
    }

    for (const auto & item : frequencies) {
        const std::vector<llama_token> & gram = item.first;
        const uint32_t frequency = item.second;

        if (gram.size() == n) {
            insert_full_ngram(gram, frequency);
            continue;
        }

        // The short-prefix padding in the Python reference requires EOS.
        if (eos_token == LLAMA_TOKEN_NULL) {
            continue;
        }

        // Direct port of trie_generate()'s prefix-padding logic.
        for (size_t k = 1; k <= prefix_len; ++k) {
            if (gram.size() <= k + 1) {
                continue;
            }

            const size_t target_suffix =
                    suffix_len + k;

            if (target_suffix < gram.size()) {
                continue;
            }

            const size_t pad_front =
                    prefix_len - k;
            const size_t pad_back =
                    target_suffix - gram.size();

            std::vector<llama_token> padded;
            padded.reserve(n);

            padded.insert(
                    padded.end(),
                    pad_front,
                    eos_token);

            padded.insert(
                    padded.end(),
                    gram.begin(),
                    gram.end());

            padded.insert(
                    padded.end(),
                    pad_back,
                    eos_token);

            if (padded.size() == n) {
                insert_full_ngram(
                        padded,
                        frequency);
            }
        }
    }

    return true;
}

void server_ngram_trie::insert_full_ngram(
        const std::vector<llama_token> & gram,
        uint32_t frequency) {
    if (gram.size() != ngram_size_) {
        return;
    }

    const size_t prefix_len =
            static_cast<size_t>(prefix_length_);

    trie_value value;
    value.frequency = frequency;
    value.suffix.assign(
            gram.begin() + static_cast<ptrdiff_t>(prefix_len),
            gram.end());

    const size_t value_id = values_.size();
    values_.push_back(std::move(value));

    size_t node_id = 0;

    // Reference implementation:
    //     key = list(reversed(kv[:prefix_length]))
    //
    // ArrayTrie.insert() associates the value with every prefix node.
    for (size_t depth = 0; depth < prefix_len; ++depth) {
        const llama_token token =
                gram[prefix_len - 1 - depth];

        auto it = nodes_[node_id].children.find(token);

        if (it == nodes_[node_id].children.end()) {
            const size_t child_id = nodes_.size();
            nodes_.emplace_back();

            nodes_[node_id].children.emplace(
                    token,
                    child_id);

            node_id = child_id;
        } else {
            node_id = it->second;
        }

        nodes_[node_id].value_ids.push_back(value_id);
    }
}

server_ngram_lookup_result server_ngram_trie::lookup(
        const std::vector<llama_token> & history,
        llama_token pending_token) const {
    server_ngram_lookup_result result;

    if (pending_token == LLAMA_TOKEN_NULL ||
            nodes_.empty()) {
        return result;
    }

    // Reference retrieve_by_ngram():
    //
    //     temp = input_ids.tolist()[-4:]
    //     temp = list(reversed(temp))
    //     trie.search_longest_prefix(temp)
    //
    // pending_token is the newest token. history excludes it.
    const size_t combined_size =
            history.size() + 1;

    const size_t max_lookup =
            std::min<size_t>(
                    lookup_window_,
                    combined_size);

    size_t node_id = 0;
    size_t best_node = 0;
    uint32_t best_depth = 0;

    for (size_t depth = 0; depth < max_lookup; ++depth) {
        llama_token token = LLAMA_TOKEN_NULL;

        if (depth == 0) {
            token = pending_token;
        } else {
            const size_t back = depth;
            if (back > history.size()) {
                break;
            }
            token = history[history.size() - back];
        }

        const auto it =
                nodes_[node_id].children.find(token);

        if (it == nodes_[node_id].children.end()) {
            break;
        }

        node_id = it->second;

        if (!nodes_[node_id].value_ids.empty()) {
            best_node = node_id;
            best_depth =
                    static_cast<uint32_t>(depth + 1);
        }
    }

    if (best_depth == 0) {
        return result;
    }

    result.matched_prefix_length = best_depth;

    const auto & ids =
            nodes_[best_node].value_ids;

    result.suffixes.reserve(ids.size());

    for (const size_t id : ids) {
        if (id >= values_.size()) {
            continue;
        }

        server_ngram_suffix suffix;
        suffix.tokens = values_[id].suffix;
        suffix.frequency = values_[id].frequency;
        result.suffixes.push_back(std::move(suffix));
    }

    // Ordering is not required for correctness, but deterministic
    // frequency-first ordering makes pruning and debugging reproducible.
    std::stable_sort(
            result.suffixes.begin(),
            result.suffixes.end(),
            [](const server_ngram_suffix & a,
               const server_ngram_suffix & b) {
                if (a.frequency != b.frequency) {
                    return a.frequency > b.frequency;
                }
                return a.tokens < b.tokens;
            });

    return result;
}

size_t server_ngram_trie::entry_count() const noexcept {
    return values_.size();
}

size_t server_ngram_trie::node_count() const noexcept {
    return nodes_.size();
}

uint32_t server_ngram_trie::ngram_size() const noexcept {
    return ngram_size_;
}

uint32_t server_ngram_trie::prefix_length() const noexcept {
    return prefix_length_;
}
