#include "server-generation-executor.h"

#include "llama-conflayer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

struct generation_batch_guard {
    llama_batch batch;

    generation_batch_guard(
            int32_t n_tokens,
            int32_t n_seq_max)
        : batch(
                llama_batch_init(
                        n_tokens,
                        0,
                        n_seq_max)) {
    }

    ~generation_batch_guard() {
        llama_batch_free(batch);
    }

    generation_batch_guard(const generation_batch_guard &) = delete;
    generation_batch_guard & operator=(
            const generation_batch_guard &) = delete;
};


bool ngram_diag_enabled() {
    const char * value =
            std::getenv("LLAMA_NGRAM_DIAG");

    return value != nullptr &&
            std::strcmp(value, "1") == 0;
}

bool ngram_diag_force_seq_enabled() {
    const char * value =
            std::getenv("LLAMA_NGRAM_DIAG_FORCE_SEQ");

    return value != nullptr &&
            std::strcmp(value, "1") == 0;
}

bool ngram_diag_compare_seq_batch_enabled() {
    const char * value =
            std::getenv("LLAMA_NGRAM_DIAG_COMPARE_SEQ_BATCH");

    return value != nullptr &&
            std::strcmp(value, "1") == 0;
}

uint64_t ngram_diag_history_hash(
        const std::vector<llama_token> & history) {
    // FNV-1a over the exact token-id byte sequence.
    uint64_t hash = 1469598103934665603ULL;

    for (const llama_token token : history) {
        const uint32_t value =
                static_cast<uint32_t>(token);

        for (int shift = 0; shift < 32; shift += 8) {
            hash ^=
                    static_cast<uint8_t>(
                            (value >> shift) & 0xffU);
            hash *= 1099511628211ULL;
        }
    }

    return hash;
}

void ngram_diag_logits(
        const char * tag,
        llama_context * ctx,
        int32_t row,
        llama_pos position,
        llama_token evaluated_token,
        int32_t n_vocab,
        const std::vector<llama_token> & history) {
    if (!ngram_diag_enabled()) {
        return;
    }

    if (ctx == nullptr || n_vocab <= 0) {
        fprintf(stderr,
                "%s: position=%d evaluated=%d row=%d "
                "error=invalid_context_or_vocab\n",
                tag,
                (int) position,
                (int) evaluated_token,
                row);
        return;
    }

    const float * logits =
            llama_get_logits_ith(ctx, row);

    if (logits == nullptr) {
        fprintf(stderr,
                "%s: position=%d evaluated=%d row=%d "
                "error=null_logits\n",
                tag,
                (int) position,
                (int) evaluated_token,
                row);
        return;
    }

    std::array<llama_token, 5> top_ids;
    std::array<float, 5> top_logits;

    top_ids.fill(LLAMA_TOKEN_NULL);
    top_logits.fill(
            -std::numeric_limits<float>::infinity());

    double sum = 0.0;
    double sumsq = 0.0;

    for (int32_t token = 0;
            token < n_vocab;
            ++token) {
        const float value = logits[token];

        sum += static_cast<double>(value);
        sumsq +=
                static_cast<double>(value) *
                static_cast<double>(value);

        for (size_t rank = 0;
                rank < top_logits.size();
                ++rank) {
            if (value <= top_logits[rank]) {
                continue;
            }

            for (size_t move =
                        top_logits.size() - 1;
                    move > rank;
                    --move) {
                top_logits[move] =
                        top_logits[move - 1];
                top_ids[move] =
                        top_ids[move - 1];
            }

            top_logits[rank] = value;
            top_ids[rank] =
                    static_cast<llama_token>(token);
            break;
        }
    }

    const double margin =
            static_cast<double>(top_logits[0]) -
            static_cast<double>(top_logits[1]);

    const uint64_t history_hash =
            ngram_diag_history_hash(history);

    fprintf(stderr,
            "%s: position=%d evaluated=%d row=%d "
            "history_size=%zu history_hash=%016llx "
            "top1=%d:%.9g top2=%d:%.9g "
            "margin=%.17g "
            "top5=%d:%.9g,%d:%.9g,%d:%.9g,"
            "%d:%.9g,%d:%.9g "
            "sum=%.17g sumsq=%.17g\n",
            tag,
            (int) position,
            (int) evaluated_token,
            row,
            history.size(),
            (unsigned long long) history_hash,
            (int) top_ids[0],
            top_logits[0],
            (int) top_ids[1],
            top_logits[1],
            margin,
            (int) top_ids[0],
            top_logits[0],
            (int) top_ids[1],
            top_logits[1],
            (int) top_ids[2],
            top_logits[2],
            (int) top_ids[3],
            top_logits[3],
            (int) top_ids[4],
            top_logits[4],
            sum,
            sumsq);
}

} // namespace

server_generation_decode_executor::
server_generation_decode_executor(
        llama_model * model,
        llama_context_params context_params,
        llama_seq_id sequence_id,
        bool conflayer_enabled,
        int32_t conflayer_max_skip_layers,
        bool conflayer_ngram_spec_enabled)
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

    ngram_spec_enabled_ =
            conflayer_enabled && conflayer_ngram_spec_enabled;

    // ConfLayers dynamic optimization evaluates a bounded context window and
    // therefore needs room for multiple outputs even on the sequential path.
    if (conflayer_enabled && context_params.n_outputs_max < 32) {
        context_params.n_outputs_max = 32;
    }

    // N-Gram speculative verification deliberately keeps the same
    // single-sequence Decode Context shape as ordinary ConfLayers Decode.
    //
    // Draft tokens are verified as one linear multi-token batch on seq 0.
    // No temporary branch sequences and no unified-KV requirement are needed.
    //
    // Keeping n_seq_max == 1 avoids changing the authoritative ConfLayers
    // Decode graph merely because speculative decoding is enabled.
    if (ngram_spec_enabled_) {
        context_params.n_seq_max = 1;
    }

    ctx_.reset(
            llama_init_from_model(
                    model_,
                    context_params));

    if (!ctx_) {
        throw std::runtime_error(
                "failed to create CPU Decode Context");
    }

    if (ngram_spec_enabled_ &&
            llama_n_seq_max(ctx_.get()) != 1) {
        throw std::runtime_error(
                "N-Gram linear speculative Decode requires "
                "a single-sequence Decode Context");
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

        ngram_history_ = handoff.prompt_tokens;
        ngram_ready_ = false;

        if (conflayer_ && ngram_spec_enabled_) {
            const llama_vocab * vocab =
                    llama_model_get_vocab(model_);

            if (vocab == nullptr) {
                throw std::runtime_error(
                        "cannot initialize N-Gram Trie without vocabulary");
            }

            ngram_ready_ =
                    ngram_trie_.build(
                            ngram_history_,
                            llama_vocab_eos(vocab));

            if (!ngram_ready_) {
                throw std::runtime_error(
                        "failed to initialize request N-Gram Trie");
            }

            fprintf(stderr,
                    "NGRAM_TRIE_READY: prompt_tokens=%zu "
                    "entries=%zu nodes=%zu\n",
                    ngram_history_.size(),
                    ngram_trie_.entry_count(),
                    ngram_trie_.node_count());

            if (ngram_diag_force_seq_enabled()) {
                fprintf(stderr,
                        "NGRAM_DIAG_FORCE_SEQ: enabled=1 "
                        "n_seq_max=%u n_ctx=%u n_ctx_seq=%u\n",
                        llama_n_seq_max(ctx_.get()),
                        llama_n_ctx(ctx_.get()),
                        llama_n_ctx_seq(ctx_.get()));
            }
        }

        next_position_ = handoff.next_position;
        active_ = true;

        return pending_token_;
    } catch (...) {
        reset();
        throw;
    }
}

llama_token
server_generation_decode_executor::decode_next(
        int32_t remaining_output_tokens) {
    if (!active_ ||
            !sampler_ ||
            pending_token_ == LLAMA_TOKEN_NULL ||
            next_position_ < 0) {
        throw std::logic_error(
                "Decode executor has no active Generation request");
    }

    // A previous speculative round may already have determined several
    // externally visible tokens. Do not issue another model Decode until
    // the queue has been drained through the existing one-token API.
    if (!ready_tokens_.empty()) {
        const llama_token token =
                ready_tokens_.front();

        ready_tokens_.pop_front();
        return token;
    }

    // Diagnostic/control path:
    // keep exactly the same single-sequence Context used by N-Gram SD, but
    // force ordinary one-token sequential Decode. This provides a clean
    // same-context A/B reference for speculative verification.
    if (ngram_spec_enabled_ &&
            ngram_diag_force_seq_enabled()) {
        return decode_next_sequential();
    }

    if (ngram_spec_enabled_ &&
            conflayer_ &&
            ngram_ready_ &&
            try_ngram_speculative_round(
                    remaining_output_tokens)) {
        if (ready_tokens_.empty()) {
            throw std::logic_error(
                    "successful N-Gram round produced no output token");
        }

        const llama_token token =
                ready_tokens_.front();

        ready_tokens_.pop_front();
        return token;
    }

    return decode_next_sequential();
}

llama_token
server_generation_decode_executor::decode_next_sequential() {
    generation_batch_guard batch(1, 1);

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

    ngram_diag_logits(
            "NGRAM_DIAG_SEQ_LOGITS",
            ctx_.get(),
            0,
            next_position_,
            evaluated_token,
            llama_vocab_n_tokens(
                    llama_model_get_vocab(model_)),
            ngram_history_);

    const llama_token next_token =
            common_sampler_sample(
                    sampler_.get(),
                    ctx_.get(),
                    0);

    if (ngram_diag_enabled()) {
        fprintf(stderr,
                "NGRAM_DIAG_SEQ_SAMPLE: "
                "position=%d evaluated=%d sampled=%d\n",
                (int) next_position_,
                (int) evaluated_token,
                (int) next_token);
    }

    common_sampler_accept(
            sampler_.get(),
            next_token,
            true);

    ++next_position_;
    pending_token_ = next_token;

    ngram_history_.push_back(evaluated_token);

    // Preserve the original pure-skip ordering:
    //
    //   skip Decode
    //   -> sample next token
    //   -> periodic full-model optimization
    //
    // dynamic_optimization() performs another Decode and invalidates the
    // logits from the skip forward, so it must remain after sampling.
    if (conflayer_ &&
            !conflayer_->advance_external_token(
                    evaluated_token)) {
        throw std::runtime_error(
                "ConfLayers failed to advance external Decode state");
    }

    return pending_token_;
}

bool
server_generation_decode_executor::
try_ngram_speculative_round(
        int32_t remaining_output_tokens) {
    if (!ngram_spec_enabled_ ||
            !conflayer_ ||
            !ngram_ready_ ||
            !ctx_ ||
            !sampler_ ||
            sequence_id_ != 0 ||
            pending_token_ == LLAMA_TOKEN_NULL) {
        return false;
    }

    const uint32_t until_opt =
            conflayer_->external_tokens_until_optimization();

    // The pending/root token itself consumes one committed Decode position.
    // Do not speculate across a ConfLayers dynamic-optimization boundary.
    if (until_opt == 0 || until_opt == 1) {
        return false;
    }

    const auto lookup =
            ngram_trie_.lookup(
                    ngram_history_,
                    pending_token_);

    if (lookup.suffixes.empty()) {
        fprintf(stderr,
                "NGRAM_SPEC_MISS: position=%d history=%zu\n",
                (int) next_position_,
                ngram_history_.size());
        return false;
    }

    const llama_vocab * vocab =
            llama_model_get_vocab(model_);

    if (vocab == nullptr) {
        throw std::runtime_error(
                "N-Gram verification has no vocabulary");
    }

    // lookup.suffixes is already sorted frequency-first by the Trie.
    // Select the highest-frequency non-empty continuation as the draft path.
    const server_ngram_suffix * selected_suffix = nullptr;

    for (const auto & suffix :
            lookup.suffixes) {
        if (!suffix.tokens.empty() &&
                suffix.frequency > 0) {
            selected_suffix = &suffix;
            break;
        }
    }

    if (selected_suffix == nullptr) {
        return false;
    }

    constexpr size_t NGRAM_MAX_DRAFT_TOKENS = 3;

    const size_t suffix_depth =
            static_cast<size_t>(
                    ngram_trie_.ngram_size() -
                    ngram_trie_.prefix_length());

    size_t max_depth =
            std::min(
                    suffix_depth,
                    NGRAM_MAX_DRAFT_TOKENS);

    // remaining_output_tokens counts externally visible tokens that the
    // server may still return for this request.
    //
    // A speculative round with D draft tokens may produce at most:
    //
    //   accepted draft tokens + one next pending token
    //   <= D + 1
    //
    // Mirror llama.cpp's native speculative-decoding budget rule:
    //
    //   draft <= n_remaining - 1
    //
    // A value < 0 means unlimited generation.
    if (remaining_output_tokens >= 0) {
        if (remaining_output_tokens <= 1) {
            return false;
        }

        max_depth =
                std::min(
                        max_depth,
                        static_cast<size_t>(
                                remaining_output_tokens - 1));
    }

    if (until_opt != UINT32_MAX) {
        max_depth =
                std::min(
                        max_depth,
                        static_cast<size_t>(
                                until_opt - 1));
    }

    // The root is evaluated at next_position_. Draft token d_i is evaluated
    // at next_position_ + i + 1.
    const uint64_t ctx_capacity =
            static_cast<uint64_t>(
                    llama_n_ctx_seq(
                            ctx_.get()));

    const uint64_t root_position =
            static_cast<uint64_t>(
                    next_position_);

    if (root_position >= ctx_capacity) {
        fprintf(stderr,
                "NGRAM_SPEC_CONTEXT_FALLBACK: "
                "position=%d capacity=%llu\n",
                (int) next_position_,
                (unsigned long long) ctx_capacity);
        return false;
    }

    const size_t ctx_max_depth =
            static_cast<size_t>(
                    ctx_capacity -
                    root_position -
                    1);

    max_depth =
            std::min(
                    max_depth,
                    ctx_max_depth);

    // One Decode contains root + draft tokens.
    const size_t batch_capacity =
            std::min(
                    static_cast<size_t>(
                            llama_n_batch(
                                    ctx_.get())),
                    static_cast<size_t>(
                            llama_n_ubatch(
                                    ctx_.get())));

    if (batch_capacity <= 1) {
        return false;
    }

    max_depth =
            std::min(
                    max_depth,
                    batch_capacity - 1);

    if (max_depth == 0) {
        return false;
    }

    std::vector<llama_token> draft_tokens;
    draft_tokens.reserve(max_depth);

    for (const llama_token token :
            selected_suffix->tokens) {
        if (draft_tokens.size() >= max_depth) {
            break;
        }

        if (token == LLAMA_TOKEN_NULL ||
                llama_vocab_is_eog(
                        vocab,
                        token)) {
            // EOG is never speculatively evaluated into KV. If the verifier
            // predicts it, it becomes next_pending exactly like sequential
            // Decode.
            break;
        }

        draft_tokens.push_back(token);
    }

    if (draft_tokens.empty()) {
        return false;
    }

    llama_memory_t mem =
            llama_get_memory(ctx_.get());

    if (mem == nullptr) {
        return false;
    }

    const llama_pos round_start =
            next_position_;

    // A previous successful round leaves no speculative tail. This defensive
    // removal also guarantees that this round begins from the authoritative
    // committed prefix only.
    if (!llama_memory_seq_rm(
                mem,
                sequence_id_,
                round_start,
                -1)) {
        throw std::runtime_error(
                "N-Gram verifier cannot clear stale seq0 tail");
    }

    // Diagnostic only: evaluate the same speculative path sequentially,
    // one token per llama_decode(), then roll the whole shadow path back.
    //
    // Comparing these logits against rows of the following linear batch gives
    // an in-process reference for:
    //
    //   sequential root,d1,d2,...  vs  one batch [root,d1,d2,...]
    //
    // No sampler state or ConfLayers external lifecycle is advanced here.
    if (ngram_diag_compare_seq_batch_enabled()) {
        constexpr size_t SHADOW_COMPARE_TOKENS = 4;

        const size_t shadow_count =
                std::min(
                        SHADOW_COMPARE_TOKENS,
                        draft_tokens.size() + 1);

        for (size_t i = 0;
                i < shadow_count;
                ++i) {
            const llama_token shadow_token =
                    i == 0 ?
                    pending_token_ :
                    draft_tokens[i - 1];

            generation_batch_guard shadow_batch(1, 1);

            common_batch_add(
                    shadow_batch.batch,
                    shadow_token,
                    round_start +
                            static_cast<llama_pos>(i),
                    { sequence_id_ },
                    true);

            const int shadow_ret =
                    llama_decode(
                            ctx_.get(),
                            shadow_batch.batch);

            if (shadow_ret != 0) {
                throw std::runtime_error(
                        "N-Gram shadow sequential Decode failed");
            }

            llama_synchronize(
                    ctx_.get());

            std::vector<llama_token> diag_history =
                    ngram_history_;

            if (i > 0) {
                diag_history.push_back(
                        pending_token_);

                diag_history.insert(
                        diag_history.end(),
                        draft_tokens.begin(),
                        draft_tokens.begin() +
                                static_cast<std::ptrdiff_t>(i - 1));
            }

            ngram_diag_logits(
                    "NGRAM_DIAG_SHADOW_SEQ_LOGITS",
                    ctx_.get(),
                    0,
                    round_start +
                            static_cast<llama_pos>(i),
                    shadow_token,
                    llama_vocab_n_tokens(vocab),
                    diag_history);

            fprintf(stderr,
                    "NGRAM_DIAG_SHADOW_SEQ_DONE: "
                    "position=%d shadow_index=%zu "
                    "evaluated=%d\n",
                    (int) (
                            round_start +
                            static_cast<llama_pos>(i)),
                    i,
                    (int) shadow_token);
        }

        // Restore the exact committed prefix that existed before this
        // diagnostic chain. The real linear verifier below starts from here.
        if (!llama_memory_seq_rm(
                    mem,
                    sequence_id_,
                    round_start,
                    -1)) {
            throw std::runtime_error(
                    "N-Gram shadow Decode rollback failed");
        }
    }

    std::vector<llama_token> batch_tokens;
    batch_tokens.reserve(
            draft_tokens.size() + 1);

    batch_tokens.push_back(
            pending_token_);

    batch_tokens.insert(
            batch_tokens.end(),
            draft_tokens.begin(),
            draft_tokens.end());

    generation_batch_guard batch(
            static_cast<int32_t>(
                    batch_tokens.size()),
            1);

    for (size_t i = 0;
            i < batch_tokens.size();
            ++i) {
        common_batch_add(
                batch.batch,
                batch_tokens[i],
                round_start +
                        static_cast<llama_pos>(i),
                { sequence_id_ },
                true);
    }

    bool decode_wrote_round = false;

    try {
        const int ret =
                llama_decode(
                        ctx_.get(),
                        batch.batch);

        if (ret != 0) {
            throw std::runtime_error(
                    "N-Gram linear verifier Decode failed");
        }

        decode_wrote_round = true;

        llama_synchronize(
                ctx_.get());

        // Work on a clone. The externally visible sampler is committed only
        // after KV and ConfLayers bookkeeping have both succeeded.
        common_sampler_ptr sampler_work;
        sampler_work.reset(
                common_sampler_clone(
                        sampler_.get()));

        if (!sampler_work) {
            throw std::runtime_error(
                    "failed to clone sampler for N-Gram verification");
        }

        std::vector<llama_token> committed;
        std::vector<llama_token> produced;

        committed.reserve(
                draft_tokens.size() + 1);
        produced.reserve(
                draft_tokens.size() + 1);

        // pending_token_ was already sampled/accepted by the live sampler in
        // the previous external step. This batch evaluates it into KV.
        committed.push_back(
                pending_token_);

        llama_token next_pending =
                LLAMA_TOKEN_NULL;

        size_t accepted_draft = 0;

        for (size_t row = 0;
                row < batch_tokens.size();
                ++row) {
            if (ngram_diag_enabled()) {
                std::vector<llama_token> diag_history =
                        ngram_history_;

                diag_history.insert(
                        diag_history.end(),
                        batch_tokens.begin(),
                        batch_tokens.begin() +
                                static_cast<std::ptrdiff_t>(row));

                ngram_diag_logits(
                        "NGRAM_DIAG_LINEAR_LOGITS",
                        ctx_.get(),
                        static_cast<int32_t>(row),
                        round_start +
                                static_cast<llama_pos>(row),
                        batch_tokens[row],
                        llama_vocab_n_tokens(vocab),
                        diag_history);
            }

            const llama_token sampled =
                    common_sampler_sample(
                            sampler_work.get(),
                            ctx_.get(),
                            static_cast<int32_t>(row));

            if (ngram_diag_enabled()) {
                fprintf(stderr,
                        "NGRAM_DIAG_LINEAR_SAMPLE: "
                        "position=%d row=%zu "
                        "evaluated=%d sampled=%d\n",
                        (int) (
                                round_start +
                                static_cast<llama_pos>(row)),
                        row,
                        (int) batch_tokens[row],
                        (int) sampled);
            }

            common_sampler_accept(
                    sampler_work.get(),
                    sampled,
                    true);

            produced.push_back(
                    sampled);

            if (llama_vocab_is_eog(
                        vocab,
                        sampled)) {
                next_pending = sampled;
                break;
            }

            // row i predicts draft_tokens[i].
            //
            // When row == draft_tokens.size(), every draft token has already
            // matched and this row supplies the next normal pending token.
            if (row < draft_tokens.size() &&
                    sampled == draft_tokens[row]) {
                committed.push_back(
                        sampled);
                ++accepted_draft;
                continue;
            }

            next_pending = sampled;
            break;
        }

        if (next_pending == LLAMA_TOKEN_NULL ||
                committed.empty() ||
                produced.empty()) {
            throw std::runtime_error(
                    "N-Gram linear verifier produced an invalid result");
        }

        const llama_pos commit_end =
                round_start +
                static_cast<llama_pos>(
                        committed.size());

        // The whole draft path was evaluated on seq0. Remove every token after
        // the accepted prefix. The sampled mismatch / final next token is not
        // evaluated and therefore remains pending exactly like sequential
        // Decode.
        if (!llama_memory_seq_rm(
                    mem,
                    sequence_id_,
                    commit_end,
                    -1)) {
            throw std::runtime_error(
                    "N-Gram verifier failed to rollback rejected draft KV");
        }

        // All verifier logits have been consumed. Dynamic optimization may now
        // run without invalidating logits still needed by this round.
        if (!conflayer_->advance_external_tokens(
                    committed)) {
            throw std::runtime_error(
                    "ConfLayers rejected speculative external advance");
        }

        ngram_history_.insert(
                ngram_history_.end(),
                committed.begin(),
                committed.end());

        next_position_ =
                commit_end;

        pending_token_ =
                next_pending;

        sampler_ =
                std::move(
                        sampler_work);

        ready_tokens_.clear();

        for (const llama_token token :
                produced) {
            ready_tokens_.push_back(
                    token);
        }

        fprintf(stderr,
                "NGRAM_SPEC_ROUND: mode=linear "
                "position=%d matched_prefix=%u "
                "suffixes=%zu draft_frequency=%u "
                "draft_len=%zu evaluated=%zu "
                "accepted_draft=%zu committed=%zu "
                "queued=%zu until_opt=%u remaining=%d\n",
                (int) round_start,
                lookup.matched_prefix_length,
                lookup.suffixes.size(),
                selected_suffix->frequency,
                draft_tokens.size(),
                batch_tokens.size(),
                accepted_draft,
                committed.size(),
                produced.size(),
                until_opt,
                remaining_output_tokens);

        return true;
    } catch (...) {
        // Unlike the old tree verifier, speculative tokens are written directly
        // to seq0. Restore the pre-round authoritative KV prefix on failure.
        if (decode_wrote_round) {
            llama_memory_seq_rm(
                    mem,
                    sequence_id_,
                    round_start,
                    -1);
        }

        throw;
    }
}



void server_generation_decode_executor::reset() {
    ready_tokens_.clear();
    ngram_history_.clear();
    ngram_trie_.clear();
    ngram_ready_ = false;

    sampler_.reset();

    if (conflayer_) {
        conflayer_->reset_external_request();
    }

    pending_token_ = LLAMA_TOKEN_NULL;
    next_position_ = -1;
    active_ = false;

    if (ctx_) {
        // A speculative round may temporarily own seq 1..N. Clear every
        // sequence, not only committed seq 0, so aborted requests cannot leak
        // branch KV into the next handoff.
        const uint32_t n_seq =
                llama_n_seq_max(ctx_.get());

        for (uint32_t sid = 0;
                sid < n_seq;
                ++sid) {
            common_context_seq_rm(
                    ctx_.get(),
                    static_cast<llama_seq_id>(sid),
                    -1,
                    -1);
        }
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
