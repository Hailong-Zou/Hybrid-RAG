#include "server-generation-executor.h"

#include "llama-conflayer.h"

#include <algorithm>
#include <cstdio>
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

    // ConfLayers + N-Gram speculative verification uses a bounded
    // verification tree. seq 0 is committed history and seq 1..31 are
    // temporary leaf branches.
    //
    // Unified KV is required:
    //   1. every sequence retains the full n_ctx capacity;
    //   2. accepted branch ranges can be committed back to seq 0 through
    //      metadata-only partial llama_memory_seq_cp().
    if (conflayer_enabled) {
        context_params.n_seq_max = 32;

        if (context_params.n_outputs_max < 32) {
            context_params.n_outputs_max = 32;
        }

        context_params.kv_unified = true;
    }

    ctx_.reset(
            llama_init_from_model(
                    model_,
                    context_params));

    if (!ctx_) {
        throw std::runtime_error(
                "failed to create CPU Decode Context");
    }

    if (conflayer_enabled) {
        if (llama_n_seq_max(ctx_.get()) < 32) {
            throw std::runtime_error(
                    "Generation Decode Context does not provide "
                    "enough sequence slots for N-Gram verification");
        }

        if (llama_n_ctx_seq(ctx_.get()) !=
                llama_n_ctx(ctx_.get())) {
            throw std::runtime_error(
                    "N-Gram speculative Decode requires unified KV "
                    "with full per-sequence context capacity");
        }

        if (llama_n_batch(ctx_.get()) < 32 ||
                llama_n_ubatch(ctx_.get()) < 32) {
            throw std::runtime_error(
                    "Generation Decode batch capacity is too small "
                    "for the N-Gram verification tree");
        }
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

        if (conflayer_) {
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
        }

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

    // A previous speculative round may already have determined several
    // externally visible tokens. Do not issue another model Decode until
    // the queue has been drained through the existing one-token API.
    if (!ready_tokens_.empty()) {
        const llama_token token =
                ready_tokens_.front();

        ready_tokens_.pop_front();
        return token;
    }

    if (conflayer_ &&
            ngram_ready_ &&
            try_ngram_speculative_round()) {
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
try_ngram_speculative_round() {
    if (!conflayer_ ||
            !ngram_ready_ ||
            !ctx_ ||
            !sampler_ ||
            sequence_id_ != 0 ||
            pending_token_ == LLAMA_TOKEN_NULL) {
        return false;
    }

    const uint32_t until_opt =
            conflayer_->external_tokens_until_optimization();

    // The root itself consumes one committed Decode position. If the next
    // optimization boundary is immediately after the root, the sequential
    // path is cheaper and preserves exactly the same lifecycle.
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

    const size_t suffix_depth =
            static_cast<size_t>(
                    ngram_trie_.ngram_size() -
                    ngram_trie_.prefix_length());

    size_t max_depth = suffix_depth;

    if (until_opt != UINT32_MAX) {
        max_depth =
                std::min(
                        max_depth,
                        static_cast<size_t>(
                                until_opt - 1));
    }

    // The speculative root is evaluated at next_position_. A node at tree
    // depth d is evaluated at next_position_ + d, so the deepest node must
    // remain strictly below llama_n_ctx_seq().
    //
    // Example:
    //   n_ctx_seq     = 2048
    //   next_position = 2046
    //
    // Valid positions are 2046 (root) and 2047, therefore max_depth <= 1.
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

    if (max_depth == 0) {
        return false;
    }

    const llama_vocab * vocab =
            llama_model_get_vocab(model_);

    if (vocab == nullptr) {
        throw std::runtime_error(
                "N-Gram verification has no vocabulary");
    }

    const llama_token eos_token =
            llama_vocab_eos(vocab);

    constexpr size_t TREE_NODE_BUDGET = 32;
    constexpr size_t TREE_SEQ_BUDGET  = 32;

    const auto tree =
            server_ngram_build_candidate_tree(
                    pending_token_,
                    lookup,
                    TREE_NODE_BUDGET,
                    max_depth,
                    eos_token);

    if (!tree.hit()) {
        return false;
    }

    server_ngram_verification_plan plan;

    if (!server_ngram_build_verification_plan(
                tree,
                TREE_SEQ_BUDGET,
                plan)) {
        fprintf(stderr,
                "NGRAM_SPEC_PLAN_FALLBACK: "
                "nodes=%zu matched_prefix=%u\n",
                tree.nodes.size(),
                lookup.matched_prefix_length);
        return false;
    }

    if (plan.branch_count() == 0 ||
            plan.branch_count() + 1 >
                    llama_n_seq_max(ctx_.get())) {
        return false;
    }

    llama_memory_t mem =
            llama_get_memory(ctx_.get());

    if (mem == nullptr) {
        return false;
    }

    // Defensive cleanup. A completed round leaves only seq 0, but this also
    // recovers cleanly if an earlier request aborted after creating branches.
    llama_memory_seq_keep(
            mem,
            sequence_id_);

    // Share committed history with every temporary leaf sequence. Because the
    // Decode Context uses unified KV this is metadata-only.
    for (size_t i = 0;
            i < plan.branch_count();
            ++i) {
        const llama_seq_id branch =
                static_cast<llama_seq_id>(i + 1);

        llama_memory_seq_cp(
                mem,
                sequence_id_,
                branch,
                0,
                next_position_);
    }

    generation_batch_guard batch(
            static_cast<int32_t>(
                    tree.nodes.size()),
            static_cast<int32_t>(
                    plan.branch_count()));

    for (size_t i = 0;
            i < tree.nodes.size();
            ++i) {
        const auto & node =
                tree.nodes[i];

        common_batch_add(
                batch.batch,
                node.token,
                next_position_ +
                        static_cast<llama_pos>(
                                node.depth),
                plan.node_seq_ids[i],
                true);
    }

    int ret = 0;

    try {
        ret =
                llama_decode(
                        ctx_.get(),
                        batch.batch);

        if (ret != 0) {
            throw std::runtime_error(
                    "N-Gram tree verifier Decode failed");
        }

        llama_synchronize(ctx_.get());

        // Work on a cloned sampler so a verifier failure cannot partially
        // advance the externally visible request sampler.
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

        committed.reserve(max_depth + 1);
        produced.reserve(max_depth + 1);

        // The root has already been sampled and accepted by the live sampler
        // before entering this round. The tree Decode evaluates it now.
        committed.push_back(
                pending_token_);

        int32_t current_node = 0;
        llama_token next_pending =
                LLAMA_TOKEN_NULL;

        while (true) {
            if (current_node < 0 ||
                    static_cast<size_t>(current_node) >=
                            tree.nodes.size()) {
                throw std::runtime_error(
                        "N-Gram verifier reached invalid tree node");
            }

            // Every tree node requested logits, so output row i corresponds
            // directly to candidate-tree node i.
            const llama_token sampled =
                    common_sampler_sample(
                            sampler_work.get(),
                            ctx_.get(),
                            current_node);

            common_sampler_accept(
                    sampler_work.get(),
                    sampled,
                    true);

            produced.push_back(sampled);

            // Match normal server semantics: an EOG token is returned to the
            // caller but is not speculatively committed as another KV token.
            if (llama_vocab_is_eog(
                        vocab,
                        sampled)) {
                next_pending = sampled;
                break;
            }

            int32_t matched_child = -1;

            for (const int32_t child :
                    tree.nodes[current_node].children) {
                if (child < 0 ||
                        static_cast<size_t>(child) >=
                                tree.nodes.size()) {
                    throw std::runtime_error(
                            "N-Gram verifier tree contains invalid child");
                }

                if (tree.nodes[child].token ==
                        sampled) {
                    matched_child = child;
                    break;
                }
            }

            if (matched_child < 0) {
                // Verifier disagreed with every N-Gram branch. The sampled
                // token becomes the next normal pending token.
                next_pending = sampled;
                break;
            }

            committed.push_back(sampled);
            current_node = matched_child;
        }

        if (next_pending == LLAMA_TOKEN_NULL ||
                committed.empty() ||
                produced.empty()) {
            throw std::runtime_error(
                    "N-Gram verifier produced an invalid result");
        }

        // Any descendant leaf sequence below current_node contains the KV
        // cells for the complete accepted prefix because every shared ancestor
        // carries all descendant leaf sequence IDs.
        const auto & accepted_node_seqs =
                plan.node_seq_ids[current_node];

        if (accepted_node_seqs.empty()) {
            throw std::runtime_error(
                    "accepted N-Gram node has no branch sequence");
        }

        const llama_seq_id accepted_seq =
                accepted_node_seqs.front();

        const llama_pos commit_end =
                next_position_ +
                static_cast<llama_pos>(
                        committed.size());

        // Add seq 0 to the accepted branch cells. Under unified KV this is
        // metadata-only and supports this exact partial range.
        llama_memory_seq_cp(
                mem,
                accepted_seq,
                sequence_id_,
                next_position_,
                commit_end);

        // Delete rejected branch-only cells and remove temporary sequence IDs
        // from shared history / accepted cells.
        llama_memory_seq_keep(
                mem,
                sequence_id_);

        // All verifier logits have been consumed at this point. Only now may
        // ConFLayer trigger dynamic_optimization(), which performs another
        // Decode and invalidates those logits.
        if (!conflayer_->advance_external_tokens(
                    committed)) {
            throw std::runtime_error(
                    "ConfLayers rejected speculative external advance");
        }

        ngram_history_.insert(
                ngram_history_.end(),
                committed.begin(),
                committed.end());

        next_position_ = commit_end;
        pending_token_ = next_pending;

        // Commit sampler state only after KV commit and ConFLayer bookkeeping
        // have both succeeded.
        sampler_ = std::move(sampler_work);

        ready_tokens_.clear();

        // produced contains:
        //
        //   accepted draft tokens, if any
        //   +
        //   exactly one next pending token
        //
        // The old server API still returns only one token per decode_next().
        for (const llama_token token :
                produced) {
            ready_tokens_.push_back(token);
        }

        fprintf(stderr,
                "NGRAM_SPEC_ROUND: position=%d "
                "matched_prefix=%u suffixes=%zu "
                "tree_nodes=%zu branches=%zu "
                "accepted_draft=%zu committed=%zu "
                "queued=%zu until_opt=%u\n",
                (int) (next_position_ -
                        static_cast<llama_pos>(
                                committed.size())),
                lookup.matched_prefix_length,
                lookup.suffixes.size(),
                tree.nodes.size(),
                plan.branch_count(),
                committed.size() - 1,
                committed.size(),
                produced.size(),
                until_opt);

        return true;
    } catch (...) {
        // Tree tokens were written only to temporary branch sequences until
        // the explicit commit above. Keeping seq 0 removes all rejected /
        // partially written speculative KV on failure.
        llama_memory_seq_keep(
                mem,
                sequence_id_);
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
