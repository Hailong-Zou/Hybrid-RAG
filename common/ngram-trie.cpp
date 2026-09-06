#include "ngram-trie.h"

#include <algorithm>
#include <map>
#include <queue>
#include <utility>

namespace {

struct common_ngram_working_node {
    llama_token token = LLAMA_TOKEN_NULL;
    int32_t parent = -1;
    uint32_t depth = 0;
    uint64_t weight = 0;
    std::map<llama_token, int32_t> children;
};

struct common_ngram_frontier_item {
    uint64_t weight = 0;
    uint32_t depth = 0;
    llama_token token = LLAMA_TOKEN_NULL;
    int32_t working_id = -1;
    int32_t output_parent = -1;
};

struct common_ngram_frontier_less {
    bool operator()(
            const common_ngram_frontier_item & a,
            const common_ngram_frontier_item & b) const {
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

common_ngram_trie::common_ngram_trie(
        uint32_t ngram_size,
        uint32_t prefix_length,
        uint32_t lookup_window)
    : ngram_size_(ngram_size),
      prefix_length_(prefix_length),
      lookup_window_(lookup_window) {
    clear();
}

void common_ngram_trie::clear() {
    nodes_.clear();
    nodes_.emplace_back();
    values_.clear();
}

bool common_ngram_trie::build(
        const llama_tokens & prompt,
        llama_token eos_token) {
    clear();

    if (ngram_size_ == 0 ||
            prefix_length_ == 0 ||
            prefix_length_ >= ngram_size_ ||
            lookup_window_ == 0) {
        return false;
    }

    const size_t n = ngram_size_;
    const size_t prefix_len = prefix_length_;
    const size_t suffix_len = n - prefix_len;

    llama_tokens source = prompt;
    if (eos_token != LLAMA_TOKEN_NULL) {
        source.push_back(eos_token);
    }

    std::map<llama_tokens, uint32_t> frequencies;

    for (size_t k = suffix_len + 1; k < n; ++k) {
        if (source.size() >= k) {
            llama_tokens gram(
                    source.begin(),
                    source.begin() + static_cast<ptrdiff_t>(k));
            ++frequencies[std::move(gram)];
        }
    }

    if (source.size() >= n) {
        for (size_t i = 0; i + n <= source.size(); ++i) {
            llama_tokens gram(
                    source.begin() + static_cast<ptrdiff_t>(i),
                    source.begin() + static_cast<ptrdiff_t>(i + n));
            ++frequencies[std::move(gram)];
        }
    }

    for (const auto & item : frequencies) {
        const llama_tokens & gram = item.first;
        const uint32_t frequency = item.second;

        if (gram.size() == n) {
            insert_full_ngram(gram, frequency);
            continue;
        }

        if (eos_token == LLAMA_TOKEN_NULL) {
            continue;
        }

        for (size_t k = 1; k <= prefix_len; ++k) {
            if (gram.size() <= k + 1) {
                continue;
            }

            const size_t target_suffix = suffix_len + k;
            if (target_suffix < gram.size()) {
                continue;
            }

            const size_t pad_front = prefix_len - k;
            const size_t pad_back = target_suffix - gram.size();

            llama_tokens padded;
            padded.reserve(n);
            padded.insert(padded.end(), pad_front, eos_token);
            padded.insert(padded.end(), gram.begin(), gram.end());
            padded.insert(padded.end(), pad_back, eos_token);

            if (padded.size() == n) {
                insert_full_ngram(padded, frequency);
            }
        }
    }

    return true;
}

void common_ngram_trie::insert_full_ngram(
        const llama_tokens & gram,
        uint32_t frequency) {
    if (gram.size() != ngram_size_) {
        return;
    }

    trie_value value;
    value.frequency = frequency;
    value.suffix.assign(
            gram.begin() + static_cast<ptrdiff_t>(prefix_length_),
            gram.end());

    const size_t value_id = values_.size();
    values_.push_back(std::move(value));

    size_t node_id = 0;
    for (size_t depth = 0; depth < prefix_length_; ++depth) {
        const llama_token token = gram[prefix_length_ - 1 - depth];
        auto it = nodes_[node_id].children.find(token);

        if (it == nodes_[node_id].children.end()) {
            const size_t child_id = nodes_.size();
            nodes_.emplace_back();
            nodes_[node_id].children.emplace(token, child_id);
            node_id = child_id;
        } else {
            node_id = it->second;
        }

        nodes_[node_id].value_ids.push_back(value_id);
    }
}

common_ngram_trie_lookup_result common_ngram_trie::lookup(
        const llama_tokens & history,
        llama_token pending_token) const {
    common_ngram_trie_lookup_result result;

    if (pending_token == LLAMA_TOKEN_NULL || nodes_.empty()) {
        return result;
    }

    const size_t combined_size = history.size() + 1;
    const size_t max_lookup = std::min<size_t>(lookup_window_, combined_size);

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

        const auto it = nodes_[node_id].children.find(token);
        if (it == nodes_[node_id].children.end()) {
            break;
        }

        node_id = it->second;
        if (!nodes_[node_id].value_ids.empty()) {
            best_node = node_id;
            best_depth = static_cast<uint32_t>(depth + 1);
        }
    }

    if (best_depth == 0) {
        return result;
    }

    result.matched_prefix_length = best_depth;
    const auto & ids = nodes_[best_node].value_ids;
    result.suffixes.reserve(ids.size());

    for (const size_t id : ids) {
        if (id >= values_.size()) {
            continue;
        }

        common_ngram_trie_suffix suffix;
        suffix.tokens = values_[id].suffix;
        suffix.frequency = values_[id].frequency;
        result.suffixes.push_back(std::move(suffix));
    }

    std::stable_sort(
            result.suffixes.begin(),
            result.suffixes.end(),
            [](const common_ngram_trie_suffix & a,
               const common_ngram_trie_suffix & b) {
                if (a.frequency != b.frequency) {
                    return a.frequency > b.frequency;
                }
                return a.tokens < b.tokens;
            });

    return result;
}

void common_ngram_trie::draft_tree(
        const common_speculative_tree_draft_params & params,
        uint32_t min_prefix_length,
        llama_token stop_token,
        common_speculative_tree_draft & result) const {
    result.clear();

    if (params.prompt == nullptr ||
            params.id_last == LLAMA_TOKEN_NULL ||
            params.n_paths <= 0 ||
            params.n_max <= 0) {
        return;
    }

    auto lookup_result = lookup(*params.prompt, params.id_last);
    result.matched_prefix_length = lookup_result.matched_prefix_length;

    if (!lookup_result.suffixes.empty() &&
            lookup_result.matched_prefix_length < min_prefix_length) {
        return;
    }

    const size_t retained_path_limit = static_cast<size_t>(params.n_paths);
    if (lookup_result.suffixes.size() > retained_path_limit) {
        lookup_result.suffixes.resize(retained_path_limit);
    }

    if (lookup_result.suffixes.empty()) {
        return;
    }

    result.retained_path_count = lookup_result.suffixes.size();
    const size_t max_depth = static_cast<size_t>(params.n_max);
    const size_t max_nodes = 1 + result.retained_path_count * max_depth;

    result.nodes.push_back({ params.id_last, -1, 0, {} });

    std::vector<common_ngram_working_node> work;
    work.push_back({ params.id_last, -1, 0, 0, {} });

    for (const auto & suffix : lookup_result.suffixes) {
        if (suffix.tokens.empty() || suffix.frequency == 0) {
            continue;
        }

        int32_t parent = 0;
        const size_t n = std::min(max_depth, suffix.tokens.size());

        for (size_t i = 0; i < n; ++i) {
            const llama_token token = suffix.tokens[i];
            auto it = work[parent].children.find(token);
            int32_t node_id;

            if (it == work[parent].children.end()) {
                node_id = static_cast<int32_t>(work.size());
                common_ngram_working_node node;
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

            if (stop_token != LLAMA_TOKEN_NULL && token == stop_token) {
                break;
            }
        }
    }

    if (work.size() <= 1) {
        result.clear();
        return;
    }

    std::priority_queue<
            common_ngram_frontier_item,
            std::vector<common_ngram_frontier_item>,
            common_ngram_frontier_less> frontier;

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

    while (!frontier.empty() && result.nodes.size() < max_nodes) {
        const common_ngram_frontier_item item = frontier.top();
        frontier.pop();

        if (item.working_id <= 0 ||
                static_cast<size_t>(item.working_id) >= work.size() ||
                item.output_parent < 0 ||
                static_cast<size_t>(item.output_parent) >= result.nodes.size()) {
            continue;
        }

        const auto & src = work[item.working_id];
        const int32_t output_id = static_cast<int32_t>(result.nodes.size());
        result.nodes.push_back({ src.token, item.output_parent, src.weight, {} });
        result.nodes[item.output_parent].children.push_back(output_id);

        for (const auto & child : src.children) {
            const int32_t child_id = child.second;
            frontier.push({
                work[child_id].weight,
                work[child_id].depth,
                work[child_id].token,
                child_id,
                output_id
            });
        }
    }
}

size_t common_ngram_trie::entry_count() const noexcept {
    return values_.size();
}

size_t common_ngram_trie::node_count() const noexcept {
    return nodes_.size();
}
