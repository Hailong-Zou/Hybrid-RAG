#pragma once

#include "server-generation-handoff.h"

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <cstdint>
#include <memory>

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
            int32_t conflayer_max_skip_layers = 10);

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
    llama_token decode_next();

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

    llama_seq_id sequence_id_ = -1;
    llama_pos next_position_ = -1;
    llama_token pending_token_ = LLAMA_TOKEN_NULL;

    bool active_ = false;
};
