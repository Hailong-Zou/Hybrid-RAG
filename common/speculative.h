#pragma once

#include "llama.h"
#include "common.h"

#include <functional>

struct common_speculative;
struct common_sampler;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;
};

// Generic shared-prefix speculative tree.
//
// node 0 is the already-sampled root token. Every other node is one
// speculative token, identified by its full parent path. n_paths limits the
// retained candidate suffix paths, not the number of children of a node.
struct common_speculative_tree_node {
    llama_token token = LLAMA_TOKEN_NULL;
    int32_t parent = -1;
    uint64_t score = 0;
    std::vector<int32_t> children;
};

struct common_speculative_tree_draft {
    std::vector<common_speculative_tree_node> nodes;
    uint32_t matched_prefix_length = 0;
    size_t retained_path_count = 0;

    bool hit() const noexcept {
        return nodes.size() > 1;
    }

    size_t speculative_node_count() const noexcept {
        return nodes.empty() ? 0 : nodes.size() - 1;
    }

    void clear() {
        nodes.clear();
        matched_prefix_length = 0;
        retained_path_count = 0;
    }
};

struct common_speculative_tree_draft_params {
    int32_t n_paths = 1;
    int32_t n_max = 1;
    llama_pos n_past = 0;
    llama_token id_last = LLAMA_TOKEN_NULL;
    const llama_tokens * prompt = nullptr;
};

struct common_speculative_tree_accept_result {
    llama_tokens tokens;
    std::vector<int32_t> accepted_nodes;
    int32_t last_accepted_node = 0;
};

// Metadata for the most recent prompt-local ngram-trie linear draft attempt.
// This keeps server diagnostics independent of the backend implementation.
struct common_speculative_ngram_trie_draft_info {
    uint32_t matched_prefix_length = 0;
    uint32_t selected_frequency = 0;
    size_t suffix_count = 0;
};

struct common_speculative_external {
    std::string name;

    std::function<void(llama_seq_id, const llama_tokens &)> begin;
    std::function<bool(const llama_batch &)> process;
    std::function<void(llama_seq_id, const common_speculative_draft_params &, llama_tokens &)> draft;
    std::function<void(llama_seq_id, const common_speculative_tree_draft_params &, common_speculative_tree_draft &)> draft_tree;
    std::function<void(llama_seq_id, uint16_t, bool)> accept;

    bool need_embd = false;
    bool need_embd_nextn = false;
};

common_speculative * common_speculative_init_external(
        common_speculative_external callbacks,
        uint32_t n_seq);

bool common_speculative_draft_tree(
        common_speculative * spec,
        llama_seq_id seq_id,
        const common_speculative_tree_draft_params & params,
        common_speculative_tree_draft & result);

bool common_speculative_get_ngram_trie_draft_info(
        common_speculative * spec,
        llama_seq_id seq_id,
        common_speculative_ngram_trie_draft_info & result);

common_speculative_tree_accept_result common_speculative_sample_and_accept_tree(
        common_sampler * sampler,
        llama_context * ctx,
        const common_speculative_tree_draft & tree,
        const std::vector<int32_t> & node_to_batch_row);

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// true if any implementation requires target post-norm embeddings to be extracted
bool common_speculative_need_embd(common_speculative * spec);

// true if any implementation requires target nextn embeddings to be extracted
bool common_speculative_need_embd_nextn(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
