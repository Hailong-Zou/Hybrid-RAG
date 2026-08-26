#include "server-generation-executor.h"

#include "llama-conflayer.h"

#include <cstdio>
#include <stdexcept>

namespace {

struct generation_batch_guard {
    llama_batch batch;

    generation_batch_guard()
        : batch(llama_batch_init(1, 0, 1)) {
    }

    ~generation_batch_guard() {
        llama_batch_free(batch);
    }

    generation_batch_guard(const generation_batch_guard &) = delete;
    generation_batch_guard & operator=(
            const generation_batch_guard &) = delete;
};

} // namespace

server_generation_decode_executor::
server_generation_decode_executor(
        llama_model * model,
        llama_context_params context_params,
        llama_seq_id sequence_id,
        bool conflayer_enabled,
        int32_t conflayer_max_skip_layers)
    : model_(model),
      sequence_id_(sequence_id) {
    if (model_ == nullptr) {
        throw std::invalid_argument(
                "cannot create Decode executor with a null model");
    }

    if (sequence_id_ < 0) {
        throw std::invalid_argument(
                "cannot create Decode executor with an invalid sequence ID");
    }

    // ConfLayers periodically re-evaluates up to context_window=32 tokens
    // with logits requested for every token. The ordinary CPU Decode Context
    // only needs one output per step, so enlarge the output capacity only when
    // ConfLayers is enabled.
    if (conflayer_enabled && context_params.n_outputs_max < 32) {
        context_params.n_outputs_max = 32;
    }

    ctx_.reset(
            llama_init_from_model(
                    model_,
                    context_params));

    if (!ctx_) {
        throw std::runtime_error(
                "failed to create CPU Decode Context");
    }

    // Match the dedicated CPU decode context to the four-thread baseline.
    llama_set_n_threads(ctx_.get(), 4, 4);

    if (sequence_id_ >=
            static_cast<llama_seq_id>(
                    llama_n_seq_max(ctx_.get()))) {
        throw std::invalid_argument(
                "Decode executor sequence ID exceeds Context capacity");
    }

    if (conflayer_enabled) {
        if (sequence_id_ != 0) {
            throw std::invalid_argument(
                    "ConfLayers external Decode currently requires sequence ID 0");
        }

        if (conflayer_max_skip_layers < 0) {
            throw std::invalid_argument(
                    "ConfLayers max skip layers cannot be negative");
        }

        llama_conflayer_params conflayer_params;

        // The Context itself already exists. These fields describe that same
        // Decode configuration; the external ConFLayer path does not create
        // another Context or perform another Prompt Prefill.
        conflayer_params.n_ctx =
                context_params.n_ctx;
        conflayer_params.n_batch =
                context_params.n_batch;
        conflayer_params.n_ubatch =
                context_params.n_ubatch;
        conflayer_params.n_threads = 4;

        // Preserve the current Zhenyu pure-skip search behavior. The only
        // experiment-specific change is the requested hard cap of 10 layers
        // (or the explicitly supplied CLI value).
        conflayer_params.skip_ratio = 0.6f;
        conflayer_params.max_skip_layers =
                conflayer_max_skip_layers;
        conflayer_params.optimize = true;
        conflayer_params.search_interval = 25;
        conflayer_params.context_window = 32;
        conflayer_params.verify = false;

        // Decode-side layer streaming belongs to the borrowed Context. This
        // is independent of the Prefill producer's physical two-slot REPACK.
        conflayer_params.layer_chunks =
                context_params.layer_chunks;
        conflayer_params.keep_active_resident = 0;

        // Match the borrowed Context's cluster bookkeeping. Keep Zhenyu's
        // current pure-skip default of full attention during ConFLayer phases.
        conflayer_params.cluster_attn =
                context_params.cluster_attn;
        conflayer_params.cluster_sparse = false;

        conflayer_ =
                std::make_unique<llama_conflayer>(
                        model_,
                        ctx_.get(),
                        conflayer_params);

        fprintf(stderr,
                "CONFLAYER_DECODE_ENABLED: max_skip_layers=%d "
                "skip_ratio=0.600 search_interval=25 "
                "context_window=32 layer_chunks=%d cluster_attn=%d\n",
                conflayer_max_skip_layers,
                context_params.layer_chunks,
                (int) context_params.cluster_attn);
    }
}

server_generation_decode_executor::
~server_generation_decode_executor() = default;

llama_token server_generation_decode_executor::begin(
        const server_generation_handoff & handoff) {
    reset();

    try {
        server_generation_handoff_restore(
                ctx_.get(),
                sequence_id_,
                handoff);

        fprintf(stderr,
                "HANDOFF_NATIVE_RESTORE_DONE: context=%p seq=%d state_bytes=%zu next_position=%d\n",
                (void *) ctx_.get(), (int) sequence_id_,
                handoff.sequence_state.size(), (int) handoff.next_position);

        if (!llama_kv_cache_cluster_rebuild_after_restore(
                    ctx_.get(), sequence_id_)) {
            throw std::runtime_error(
                    "failed to rebuild cluster state after handoff restore");
        }

        if (conflayer_ &&
                !conflayer_->begin_external_request(
                        handoff.prompt_tokens,
                        handoff.next_position)) {
            throw std::runtime_error(
                    "failed to initialize ConfLayers after Generation handoff");
        }

        common_params_sampling sampling_params =
                handoff.sampling_params;

        sampler_.reset(
                common_sampler_init(
                        model_,
                        sampling_params));

        if (!sampler_) {
            throw std::runtime_error(
                    "failed to initialize Decode-side sampler");
        }

        server_generation_handoff_prepare_sampler(
                sampler_.get(),
                handoff);

        pending_token_ =
                server_generation_handoff_sample_first_token(
                        sampler_.get(),
                        handoff);

        common_sampler_accept(
                sampler_.get(),
                pending_token_,
                true);

        next_position_ = handoff.next_position;
        active_ = true;

        return pending_token_;
    } catch (...) {
        reset();
        throw;
    }
}

llama_token server_generation_decode_executor::decode_next() {
    if (!active_ ||
            !sampler_ ||
            pending_token_ == LLAMA_TOKEN_NULL ||
            next_position_ < 0) {
        throw std::logic_error(
                "Decode executor has no active Generation request");
    }

    generation_batch_guard batch;

    // This is the token that will actually be written into the Decode
    // Context at next_position_. Keep it separate from the next token that
    // will be sampled from this forward's logits.
    const llama_token evaluated_token =
            pending_token_;

    common_batch_add(
            batch.batch,
            evaluated_token,
            next_position_,
            { sequence_id_ },
            true);

    const int ret =
            llama_decode(
                    ctx_.get(),
                    batch.batch);

    if (ret != 0) {
        throw std::runtime_error(
                "CPU Decode executor failed to evaluate pending token");
    }

    llama_synchronize(ctx_.get());

    const llama_token next_token =
            common_sampler_sample(
                    sampler_.get(),
                    ctx_.get(),
                    0);

    common_sampler_accept(
            sampler_.get(),
            next_token,
            true);

    ++next_position_;
    pending_token_ = next_token;

    // Run periodic full-model optimization only after next_token has already
    // been sampled from the skip forward above. dynamic_optimization() issues
    // its own full-model decode and therefore replaces the Context logits.
    if (conflayer_ &&
            !conflayer_->advance_external_token(
                    evaluated_token)) {
        throw std::runtime_error(
                "ConfLayers failed to advance external Decode state");
    }

    return pending_token_;
}

void server_generation_decode_executor::reset() {
    sampler_.reset();

    if (conflayer_) {
        conflayer_->reset_external_request();
    }

    pending_token_ = LLAMA_TOKEN_NULL;
    next_position_ = -1;
    active_ = false;

    if (ctx_) {
        common_context_seq_rm(
                ctx_.get(),
                sequence_id_,
                -1,
                -1);
    }
}

bool server_generation_decode_executor::active() const noexcept {
    return active_;
}

llama_token
server_generation_decode_executor::pending_token() const noexcept {
    return pending_token_;
}

llama_pos
server_generation_decode_executor::next_position() const noexcept {
    return next_position_;
}

llama_seq_id
server_generation_decode_executor::sequence_id() const noexcept {
    return sequence_id_;
}

llama_context *
server_generation_decode_executor::context() noexcept {
    return ctx_.get();
}

const llama_context *
server_generation_decode_executor::context() const noexcept {
    return ctx_.get();
}
