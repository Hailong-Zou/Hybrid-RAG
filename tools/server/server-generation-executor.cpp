#include "server-generation-executor.h"

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
        llama_seq_id sequence_id)
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
}

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

    common_batch_add(
            batch.batch,
            pending_token_,
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

    return pending_token_;
}

void server_generation_decode_executor::reset() {
    sampler_.reset();

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
