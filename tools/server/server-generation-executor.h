#pragma once

#include "server-generation-handoff.h"
#include "server-ngram-tree.h"
#include "server-ngram-trie.h"

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

class llama_conflayer;

// CPU Decode executor for the formal Generation Prefill/Decode boundary.
//
// Resource ownership:
// - model_ is non-owning; its owner must outlive this executor;
// - ctx_ is owned exclusively by this executor;
// - sampler_ is owned exclusively by this executor.
//
// One executor represents one Decode worker and is not thread-safe. It can
// process successive requests by calling begin() again.
class server_generation_decode_executor {
public:
    server_generation_decode_executor(
            llama_model * model,
            llama_context_params context_params,
            llama_seq_id sequence_id = 0,
            bool conflayer_enabled = false,
            int32_t conflayer_max_skip_layers = 10,
            bool conflayer_ngram_spec_enabled = false,
            bool reclaim_zero_kv_control = false);

    ~server_generation_decode_executor();

    server_generation_decode_executor(
            const server_generation_decode_executor &) = delete;

    server_generation_decode_executor & operator=(
            const server_generation_decode_executor &) = delete;

    // Restore Prompt state, create the request sampler, sample and accept
    // the first generated token from the transferred final Prefill logits.
    //
    // The returned token has not yet been evaluated by the Decode Context.
    llama_token begin(
            const server_generation_handoff & handoff);

    // Evaluate the current pending token and sample the next token.
    //
    // begin() must have completed successfully before this call.
    llama_token decode_next(
            int32_t remaining_output_tokens = -1);

    // Release request-local sampler state and clear the executor sequence.
    void reset();

    bool active() const noexcept;

    llama_token pending_token() const noexcept;
    llama_pos next_position() const noexcept;
    llama_seq_id sequence_id() const noexcept;

    llama_context * context() noexcept;
    const llama_context * context() const noexcept;

private:
    llama_model * model_ = nullptr; // non-owning
    bool reclaim_zero_kv_ = false;

    llama_context_ptr ctx_;
    common_sampler_ptr sampler_;

    // Optional pure-skip controller. It borrows ctx_ and never owns or
    // allocates another Decode Context.
    std::unique_ptr<llama_conflayer> conflayer_;

    // Independent switch for N-Gram speculative decoding.
    // OFF keeps the original sequential ConfLayers Decode path.
    bool ngram_spec_enabled_ = false;
    bool ngram_native_framework_ = true;

    // Per-request N-Gram runtime configuration snapshot.
    //
    // Runtime configuration files may change between requests, but one
    // Decode request keeps a consistent configuration from start to finish.
    size_t ngram_min_prefix_length_ = 1;
    size_t ngram_max_branches_ = 3;
    size_t ngram_max_draft_ = 3;

    // Prompt-local N-Gram draft source. It is built once from the transferred
    // Prompt and remains static for the whole request.
    server_ngram_trie ngram_trie_;
    std::vector<llama_token> ngram_history_;
    bool ngram_ready_ = false;

    common_speculative_ptr ngram_native_tree_spec_;
    llama_tokens ngram_native_draft_;
    common_speculative_tree_draft ngram_native_tree_draft_;
    uint32_t ngram_last_matched_prefix_ = 0;
    uint32_t ngram_last_draft_frequency_ = 0;
    size_t ngram_last_suffix_count_ = 0;

    // Tokens already determined by one speculative verification round but
    // not yet returned through the one-token-at-a-time server interface.
    //
    // The final element of every successful round is pending_token_: it has
    // been sampled/accepted but has NOT been evaluated into KV yet.
    std::deque<llama_token> ready_tokens_;

    llama_seq_id sequence_id_ = -1;
    llama_pos next_position_ = -1;
    llama_token pending_token_ = LLAMA_TOKEN_NULL;

    bool active_ = false;

    llama_token decode_next_sequential();

    // Returns true when one speculative round succeeded and ready_tokens_
    // contains at least the next externally visible token.
    bool try_ngram_speculative_round(
            int32_t remaining_output_tokens);

    bool try_ngram_tree_speculative_round(
            int32_t remaining_output_tokens);

    void draft_ngram_linear_custom(
            llama_seq_id seq_id,
            const common_speculative_draft_params & params,
            llama_tokens & result);

};
