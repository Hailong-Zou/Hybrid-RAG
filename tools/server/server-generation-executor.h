#pragma once

#include "server-generation-handoff.h"
#include "server-ngram-tree.h"
#include "server-ngram-trie.h"

#include "common.h"
#include "llama.h"
#include "sampling.h"

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
            bool conflayer_ngram_spec_enabled = false);

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

    llama_context_ptr ctx_;
    common_sampler_ptr sampler_;

    // Optional pure-skip controller. It borrows ctx_ and never owns or
    // allocates another Decode Context.
    std::unique_ptr<llama_conflayer> conflayer_;

    // Independent switch for N-Gram speculative decoding.
    // OFF keeps the original sequential ConfLayers Decode path.
    bool ngram_spec_enabled_ = false;

    // Prompt-local N-Gram draft source. It is built once from the transferred
    // Prompt and remains static for the whole request.
    server_ngram_trie ngram_trie_;
    std::vector<llama_token> ngram_history_;
    bool ngram_ready_ = false;

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
};
