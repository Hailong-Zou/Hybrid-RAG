#include "server-generation-executor.h"

#include "llama-conflayer.h"
#include "../../src/llama-ext.h"
#include "ggml-backend.h"
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

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

// Only use after every KV byte is zero and all old sequences are discarded.
static void reclaim_zero_cpu_kv(llama_context * ctx) {
#if defined(__linux__)
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) return;
    struct stats {
        uintptr_t page_size;
        size_t bytes = 0;
        size_t failures = 0;
    } result{static_cast<uintptr_t>(page_size_long)};
    llama_context_visit_memory_regions(ctx,
            [](void * userdata, const void * base, size_t size,
                    llama_memory_region_kind kind, ggml_backend_buffer_type_t buft) {
                auto & s = *static_cast<stats *>(userdata);
                if (kind != LLAMA_MEMORY_REGION_KV || base == nullptr || buft == nullptr ||
                        !ggml_backend_buft_is_host(buft) ||
                        std::strcmp(ggml_backend_buft_name(buft), "CPU") != 0) return;
                const uintptr_t first = reinterpret_cast<uintptr_t>(base);
                if (size > std::numeric_limits<uintptr_t>::max() - first ||
                        first > std::numeric_limits<uintptr_t>::max() - (s.page_size - 1)) return;
                const uintptr_t begin = (first + s.page_size - 1) & ~(s.page_size - 1);
                const uintptr_t end = (first + size) & ~(s.page_size - 1);
                if (end <= begin) return;
                if (madvise(reinterpret_cast<void *>(begin), end - begin, MADV_DONTNEED) == 0)
                    s.bytes += end - begin;
                else ++s.failures;
            }, &result);
    std::fprintf(stderr, "DECODE_ZERO_KV_RECLAIM context=%p advised_bytes=%zu failures=%zu\n",
            static_cast<void *>(ctx), result.bytes, result.failures);
#else
    (void) ctx;
#endif
}

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

bool ngram_diag_tree_enabled() {
    const char * value =
            std::getenv("LLAMA_NGRAM_DIAG_TREE");

    return value != nullptr &&
            std::strcmp(value, "1") == 0;
}



size_t ngram_tree_env_size(
        const char * name,
        size_t default_value,
        size_t min_value,
        size_t max_value) {
    const char * value = std::getenv(name);

    if (value == nullptr ||
            value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;

    const long parsed =
            std::strtol(
                    value,
                    &end,
                    10);

    if (end == value ||
            *end != '\0' ||
            parsed < static_cast<long>(min_value) ||
            parsed > static_cast<long>(max_value)) {
        fprintf(
                stderr,
                "NGRAM_TREE_CONFIG_WARNING: "
                "%s=%s invalid, using %zu\n",
                name,
                value,
                default_value);

        return default_value;
    }

    return static_cast<size_t>(parsed);
}

size_t ngram_tree_runtime_size(
        const char * env_name,
        const char * file_env_name,
        size_t default_value,
        size_t min_value,
        size_t max_value) {
    const char * path =
            std::getenv(file_env_name);

    if (path != nullptr &&
            path[0] != '\0') {
        FILE * fp = std::fopen(
                path,
                "r");

        if (fp != nullptr) {
            long parsed = 0;

            const int rc =
                    std::fscanf(
                            fp,
                            "%ld",
                            &parsed);

            std::fclose(fp);

            if (rc == 1 &&
                    parsed >=
                            static_cast<long>(
                                    min_value) &&
                    parsed <=
                            static_cast<long>(
                                    max_value)) {
                return static_cast<size_t>(
                        parsed);
            }

            fprintf(
                    stderr,
                    "NGRAM_TREE_CONFIG_WARNING: "
                    "invalid file %s, "
                    "falling back to %s\n",
                    path,
                    env_name);
        }
    }

    return ngram_tree_env_size(
            env_name,
            default_value,
            min_value,
            max_value);
}

size_t ngram_tree_max_branches() {
    return ngram_tree_runtime_size(
            "LLAMA_NGRAM_TREE_MAX_BRANCHES",
            "LLAMA_NGRAM_TREE_MAX_BRANCHES_FILE",
            3,
            1,
            8);
}

size_t ngram_tree_max_draft() {
    return ngram_tree_runtime_size(
            "LLAMA_NGRAM_TREE_MAX_DRAFT",
            "LLAMA_NGRAM_TREE_MAX_DRAFT_FILE",
            3,
            1,
            8);
}

// Context capacity is fixed when the Decode Context is created.
//
// This is intentionally independent from runtime B/D. The server reserves
// enough temporary sequences/output rows once, while each request can select
// a smaller B/D dynamically through LLAMA_NGRAM_TREE_MAX_{BRANCHES,DRAFT}_FILE.
//
// Defaults match the maximum currently accepted runtime B/D, so any valid
// runtime configuration can be selected without recreating the Context.
size_t ngram_tree_branch_capacity() {
    return ngram_tree_env_size(
            "LLAMA_NGRAM_TREE_BRANCH_CAPACITY",
            8,
            1,
            8);
}

size_t ngram_tree_draft_capacity() {
    return ngram_tree_env_size(
            "LLAMA_NGRAM_TREE_DRAFT_CAPACITY",
            8,
            1,
            8);
}


size_t ngram_runtime_size() {
    return ngram_tree_runtime_size(
            "LLAMA_NGRAM_SIZE",
            "LLAMA_NGRAM_SIZE_FILE",
            13,
            2,
            32);
}

size_t ngram_runtime_prefix_length() {
    return ngram_tree_runtime_size(
            "LLAMA_NGRAM_PREFIX_LENGTH",
            "LLAMA_NGRAM_PREFIX_LENGTH_FILE",
            3,
            1,
            16);
}

size_t ngram_tree_min_prefix_length() {
    return ngram_tree_runtime_size(
            "LLAMA_NGRAM_TREE_MIN_PREFIX_LENGTH",
            "LLAMA_NGRAM_TREE_MIN_PREFIX_LENGTH_FILE",
            1,
            1,
            16);
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

bool ngram_native_framework_enabled() {
    const char * path =
            std::getenv("LLAMA_NGRAM_NATIVE_FRAMEWORK_FILE");

    char file_value[16] = {};

    if (path != nullptr &&
            path[0] != '\0') {
        FILE * fp = std::fopen(
                path,
                "r");

        if (fp != nullptr) {
            const int rc =
                    std::fscanf(
                            fp,
                            "%15s",
                            file_value);

            std::fclose(fp);

            if (rc == 1) {
                if (std::strcmp(file_value, "native") == 0 ||
                        std::strcmp(file_value, "1") == 0) {
                    return true;
                }

                if (std::strcmp(file_value, "custom") == 0 ||
                        std::strcmp(file_value, "0") == 0) {
                    return false;
                }
            }

            fprintf(
                    stderr,
                    "NGRAM_NATIVE_FRAMEWORK_WARNING: "
                    "invalid file %s, falling back to environment\n",
                    path);
        }
    }

    const char * value =
            std::getenv("LLAMA_NGRAM_NATIVE_FRAMEWORK");

    if (value == nullptr ||
            value[0] == '\0' ||
            std::strcmp(value, "native") == 0 ||
            std::strcmp(value, "1") == 0) {
        return true;
    }

    if (std::strcmp(value, "custom") == 0 ||
            std::strcmp(value, "0") == 0) {
        return false;
    }

    fprintf(
            stderr,
            "NGRAM_NATIVE_FRAMEWORK_WARNING: "
            "invalid LLAMA_NGRAM_NATIVE_FRAMEWORK=%s, using native\n",
            value);

    return true;
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
        bool conflayer_ngram_spec_enabled,
        bool reclaim_zero_kv_control)
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
        if (ngram_diag_tree_enabled()) {
            const size_t retained_path_capacity =
                    ngram_tree_branch_capacity();

            const size_t tree_draft_capacity =
                    ngram_tree_draft_capacity();

            // seq0 = committed history.
            // seq1..N = temporary candidate leaves.
            //
            // Allocate from startup capacity, not the current runtime B.
            // Runtime B may change between requests without recreating ctx_.
            context_params.n_seq_max =
                    static_cast<uint32_t>(
                            retained_path_capacity + 1);

            context_params.kv_unified = true;

            // Keep at least 32 output rows for ConfLayer optimization.
            //
            // The maximum verification tree representable by this Context is:
            //
            //   root + branch_capacity * draft_capacity
            //
            // Runtime B/D are bounded by these capacities.
            const uint32_t tree_outputs =
                    static_cast<uint32_t>(
                            1 +
                            retained_path_capacity *
                            tree_draft_capacity);

            const uint32_t required_outputs =
                    std::max<uint32_t>(
                            32,
                            tree_outputs);

            if (context_params.n_outputs_max <
                    required_outputs) {
                context_params.n_outputs_max =
                        required_outputs;
            }

            fprintf(
                    stderr,
                    "NGRAM_TREE_CAPACITY: "
                    "branch_capacity=%zu draft_capacity=%zu "
                    "n_seq_max=%u max_nodes=%u\n",
                    retained_path_capacity,
                    tree_draft_capacity,
                    context_params.n_seq_max,
                    tree_outputs);
        } else {
            context_params.n_seq_max = 1;
        }
    }

    // Diagnostic-only Context-shape overrides for isolating the old
    // multi-sequence verification-tree correctness issue.
    //
    // These variables are intentionally independent so we can run the 2x2:
    //
    //   n_seq_max = 1 / 32
    //   kv_unified = false / true
    //
    // with N-Gram and the verification tree both disabled.
    const char * nseq_diag_max =
            std::getenv("LLAMA_NSEQ_DIAG_MAX");

    if (nseq_diag_max != nullptr &&
            nseq_diag_max[0] != '\0') {
        char * end = nullptr;
        const long value =
                std::strtol(
                        nseq_diag_max,
                        &end,
                        10);

        if (end == nseq_diag_max ||
                *end != '\0' ||
                (value != 1 && value != 32)) {
            throw std::invalid_argument(
                    "LLAMA_NSEQ_DIAG_MAX must be 1 or 32");
        }

        context_params.n_seq_max =
                static_cast<uint32_t>(value);
    }

    const char * nseq_diag_unified =
            std::getenv("LLAMA_NSEQ_DIAG_UNIFIED");

    if (nseq_diag_unified != nullptr &&
            nseq_diag_unified[0] != '\0') {
        if (std::strcmp(
                    nseq_diag_unified,
                    "0") == 0) {
            context_params.kv_unified = false;
        } else if (std::strcmp(
                       nseq_diag_unified,
                       "1") == 0) {
            context_params.kv_unified = true;
        } else {
            throw std::invalid_argument(
                    "LLAMA_NSEQ_DIAG_UNIFIED must be 0 or 1");
        }
    }

    if (nseq_diag_max != nullptr ||
            nseq_diag_unified != nullptr) {
        fprintf(stderr,
                "NSEQ_DIAG_CONTEXT_REQUEST: "
                "n_seq_max=%u kv_unified=%d "
                "ngram_spec=%d conflayer=%d\n",
                context_params.n_seq_max,
                (int) context_params.kv_unified,
                (int) ngram_spec_enabled_,
                (int) conflayer_enabled);
    }

    ctx_.reset(
            llama_init_from_model(
                    model_,
                    context_params));

    if (!ctx_) {
        throw std::runtime_error(
                "failed to create CPU Decode Context");
    }

    const char * parallel = std::getenv("LLAMA_RAG_PARALLEL_DECODE");
    reclaim_zero_kv_ = !reclaim_zero_kv_control && !context_params.cluster_attn && parallel != nullptr &&
            std::strcmp(parallel, "1") == 0;
    // Context construction has already cleared its KV buffers to zero.
    if (reclaim_zero_kv_) reclaim_zero_cpu_kv(ctx_.get());

    if (nseq_diag_max != nullptr ||
            nseq_diag_unified != nullptr) {
        fprintf(stderr,
                "NSEQ_DIAG_CONTEXT_ACTUAL: "
                "n_seq_max=%u n_ctx=%u n_ctx_seq=%u\n",
                llama_n_seq_max(ctx_.get()),
                llama_n_ctx(ctx_.get()),
                llama_n_ctx_seq(ctx_.get()));
    }

    if (ngram_spec_enabled_) {
        if (ngram_diag_tree_enabled()) {
            const size_t required_sequences =
                    ngram_tree_branch_capacity() + 1;

            if (llama_n_seq_max(ctx_.get()) <
                        required_sequences ||
                    llama_n_ctx_seq(ctx_.get()) !=
                            llama_n_ctx(ctx_.get())) {
                throw std::runtime_error(
                        "N-Gram tree diagnostic requires "
                        "sufficient sequences and unified KV");
            }
        } else if (llama_n_seq_max(ctx_.get()) != 1) {
            throw std::runtime_error(
                    "N-Gram linear speculative Decode requires "
                    "a single-sequence Decode Context");
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

    if (reclaim_zero_kv_) {
        // Preserve zero padding before restoring this request's live cells.
        llama_memory_clear(llama_get_memory(ctx_.get()), true);
        reclaim_zero_cpu_kv(ctx_.get());
    }

    try {
        server_generation_handoff_restore(
                ctx_.get(),
                sequence_id_,
                handoff);

        fprintf(stderr,
                "HANDOFF_NATIVE_RESTORE_DONE: context=%p seq=%d state_bytes=%zu next_position=%d\n",
                (void *) ctx_.get(), (int) sequence_id_,
                handoff.sequence_state.size(), (int) handoff.next_position);
        const char * trace = std::getenv("LLAMA_GEN_DECODE_TRACE");
        if (trace != nullptr && std::strcmp(trace, "1") == 0) {
            fprintf(stderr,
                    "HANDOFF_EFFECTIVE_CONFIG context=%p seed=%u temperature=%.6f state_checksum=%llu logits_checksum=%llu\n",
                    (void *) ctx_.get(), handoff.sampling_params.seed, handoff.sampling_params.temp,
                    static_cast<unsigned long long>(handoff.sequence_state_checksum),
                    static_cast<unsigned long long>(handoff.final_logits_checksum));
        }

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
            ngram_native_framework_ =
                    ngram_native_framework_enabled();

            const llama_vocab * vocab =
                    llama_model_get_vocab(model_);

            if (vocab == nullptr) {
                throw std::runtime_error(
                        "cannot initialize N-Gram Trie without vocabulary");
            }

            const size_t runtime_ngram_size =
                    ngram_runtime_size();

            const size_t runtime_prefix_length =
                    ngram_runtime_prefix_length();

            const size_t runtime_min_prefix_length =
                    ngram_tree_min_prefix_length();

            const size_t runtime_max_branches =
                    ngram_tree_max_branches();

            const size_t runtime_max_draft =
                    ngram_tree_max_draft();

            if (runtime_prefix_length >= runtime_ngram_size) {
                throw std::runtime_error(
                        "N-Gram prefix length must be smaller than ngram size");
            }

            if (runtime_min_prefix_length >
                    runtime_prefix_length) {
                throw std::runtime_error(
                        "N-Gram tree minimum prefix length must not exceed "
                        "configured prefix length");
            }

            if (runtime_max_branches >
                        ngram_tree_branch_capacity() ||
                    runtime_max_draft >
                        ngram_tree_draft_capacity()) {
                throw std::runtime_error(
                        "N-Gram tree runtime B/D exceeds Decode Context capacity");
            }

            ngram_min_prefix_length_ =
                    runtime_min_prefix_length;

            ngram_max_branches_ =
                    runtime_max_branches;

            ngram_max_draft_ =
                    runtime_max_draft;

            ngram_trie_ =
                    server_ngram_trie(
                            static_cast<uint32_t>(runtime_ngram_size),
                            static_cast<uint32_t>(runtime_prefix_length),
                            4);

            fprintf(
                    stderr,
                    "NGRAM_TRIE_CONFIG: n=%zu prefix=%zu "
                    "min_prefix=%zu lookup_window=4\n",
                    runtime_ngram_size,
                    runtime_prefix_length,
                    runtime_min_prefix_length);

            fprintf(
                    stderr,
                    "NGRAM_RUNTIME_CONFIG: "
                    "n=%zu prefix=%zu min_prefix=%zu "
                    "branches=%zu draft=%zu framework=%s "
                    "lookup_window=4\n",
                    runtime_ngram_size,
                    runtime_prefix_length,
                    ngram_min_prefix_length_,
                    ngram_max_branches_,
                    ngram_max_draft_,
                    ngram_native_framework_ ? "native" : "custom");

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

            if (ngram_native_framework_) {
                common_params_speculative tree_params;
                tree_params.types = {
                    COMMON_SPECULATIVE_TYPE_NGRAM_TRIE
                };
                tree_params.ngram_trie.size_n =
                        static_cast<uint32_t>(runtime_ngram_size);
                tree_params.ngram_trie.prefix_length =
                        static_cast<uint32_t>(runtime_prefix_length);
                tree_params.ngram_trie.min_prefix =
                        static_cast<uint32_t>(runtime_min_prefix_length);
                tree_params.ngram_trie.lookup_window = 4;
                tree_params.ngram_trie.n_max =
                        static_cast<int32_t>(runtime_max_draft);
                tree_params.ngram_trie.eos_token =
                        llama_vocab_eos(vocab);

                ngram_native_tree_spec_.reset(
                        common_speculative_init(
                                tree_params,
                                1));

                if (!ngram_native_tree_spec_) {
                    throw std::runtime_error(
                            "failed to initialize common ngram-trie backend");
                }

                common_speculative_begin(
                        ngram_native_tree_spec_.get(),
                        sequence_id_,
                        ngram_history_);
            }

            fprintf(stderr,
                    "NGRAM_LINEAR_FRAMEWORK: mode=%s "
                    "draft=%s sampler=%s verifier=linear\n",
                    ngram_native_framework_ ? "native" : "custom",
                    ngram_native_framework_ ?
                            "common_speculative" : "direct_trie",
                    ngram_native_framework_ ?
                            "sample_and_accept_n" : "per_row");

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
            ngram_diag_tree_enabled()) {
        if (conflayer_ &&
                ngram_ready_ &&
                try_ngram_tree_speculative_round(
                        remaining_output_tokens)) {
            if (ready_tokens_.empty()) {
                throw std::logic_error(
                        "successful N-Gram tree round produced no output token");
            }

            const llama_token token =
                    ready_tokens_.front();

            ready_tokens_.pop_front();
            return token;
        }

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

void
server_generation_decode_executor::
draft_ngram_linear_custom(
        llama_seq_id seq_id,
        const common_speculative_draft_params & params,
        llama_tokens & result) {
    result.clear();
    ngram_last_matched_prefix_ = 0;
    ngram_last_draft_frequency_ = 0;
    ngram_last_suffix_count_ = 0;

    if (!ngram_ready_ ||
            seq_id != sequence_id_ ||
            params.prompt == nullptr ||
            params.id_last == LLAMA_TOKEN_NULL) {
        return;
    }

    const auto lookup =
            ngram_trie_.lookup(
                    *params.prompt,
                    params.id_last);

    ngram_last_matched_prefix_ =
            lookup.matched_prefix_length;
    ngram_last_suffix_count_ =
            lookup.suffixes.size();

    const server_ngram_suffix * selected_suffix = nullptr;

    for (const auto & suffix : lookup.suffixes) {
        if (!suffix.tokens.empty() &&
                suffix.frequency > 0) {
            selected_suffix = &suffix;
            break;
        }
    }

    if (selected_suffix == nullptr) {
        return;
    }

    ngram_last_draft_frequency_ =
            selected_suffix->frequency;

    const llama_vocab * vocab =
            llama_model_get_vocab(model_);

    if (vocab == nullptr) {
        return;
    }

    size_t n_max = selected_suffix->tokens.size();
    if (params.n_max >= 0) {
        n_max = std::min(
                n_max,
                static_cast<size_t>(params.n_max));
    }

    result.reserve(n_max);

    for (const llama_token token : selected_suffix->tokens) {
        if (result.size() >= n_max ||
                token == LLAMA_TOKEN_NULL ||
                llama_vocab_is_eog(vocab, token)) {
            break;
        }

        result.push_back(token);
    }
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

    const llama_vocab * vocab =
            llama_model_get_vocab(model_);

    if (vocab == nullptr) {
        throw std::runtime_error(
                "N-Gram verification has no vocabulary");
    }

    constexpr size_t NGRAM_MAX_DRAFT_TOKENS = 3;

    size_t max_depth = NGRAM_MAX_DRAFT_TOKENS;

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

    llama_tokens custom_draft;
    llama_tokens * draft_result =
            ngram_native_framework_ ?
            &ngram_native_draft_ :
            &custom_draft;

    draft_result->clear();

    const common_speculative_draft_params draft_params = {
        /* .drafting = */ true,
        /* .n_max    = */ static_cast<int32_t>(max_depth),
        /* .n_past   = */ next_position_,
        /* .id_last  = */ pending_token_,
        /* .prompt   = */ &ngram_history_,
        /* .result   = */ draft_result,
    };

    if (ngram_native_framework_) {
        if (!ngram_native_tree_spec_) {
            return false;
        }

        common_speculative_get_draft_params(
                ngram_native_tree_spec_.get(),
                sequence_id_) = draft_params;

        common_speculative_draft(
                ngram_native_tree_spec_.get());

        // Preserve the compatibility path's stop-token semantics. The common
        // backend knows the configured EOS token; the target vocabulary may
        // classify additional control tokens as end-of-generation tokens.
        const llama_vocab * vocab =
                llama_model_get_vocab(model_);
        if (vocab == nullptr) {
            draft_result->clear();
        } else {
            for (size_t i = 0; i < draft_result->size(); ++i) {
                const llama_token token = (*draft_result)[i];
                if (token == LLAMA_TOKEN_NULL ||
                        llama_vocab_is_eog(vocab, token)) {
                    draft_result->resize(i);
                    break;
                }
            }
        }

        common_speculative_ngram_trie_draft_info draft_info;
        if (!common_speculative_get_ngram_trie_draft_info(
                    ngram_native_tree_spec_.get(),
                    sequence_id_,
                    draft_info)) {
            throw std::runtime_error(
                    "common ngram-trie backend did not expose linear draft metadata");
        }

        ngram_last_matched_prefix_ =
                draft_info.matched_prefix_length;
        ngram_last_draft_frequency_ =
                draft_info.selected_frequency;
        ngram_last_suffix_count_ =
                draft_info.suffix_count;
    } else {
        draft_ngram_linear_custom(
                sequence_id_,
                draft_params,
                *draft_result);
    }

    if (draft_result->empty()) {
        fprintf(stderr,
                "NGRAM_SPEC_MISS: position=%d history=%zu "
                "matched_prefix=%u suffixes=%zu\n",
                (int) next_position_,
                ngram_history_.size(),
                ngram_last_matched_prefix_,
                ngram_last_suffix_count_);
        return false;
    }

    const llama_tokens & draft_tokens =
            *draft_result;

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

        committed.reserve(
                draft_tokens.size() + 1);

        // pending_token_ was already sampled/accepted by the live sampler in
        // the previous external step. This batch evaluates it into KV.
        committed.push_back(
                pending_token_);

        if (ngram_diag_enabled()) {
            for (size_t row = 0;
                    row < batch_tokens.size();
                    ++row) {
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
        }

        std::vector<llama_token> produced;
        size_t accepted_draft = 0;
        llama_token next_pending =
                LLAMA_TOKEN_NULL;

        produced.reserve(
                draft_tokens.size() + 1);

        if (ngram_native_framework_) {
            std::vector<int> spec_i_batch(
                    batch_tokens.size());

            for (size_t i = 0;
                    i < spec_i_batch.size();
                    ++i) {
                spec_i_batch[i] =
                        static_cast<int>(i);
            }

            produced =
                    common_sampler_sample_and_accept_n(
                            sampler_work.get(),
                            ctx_.get(),
                            spec_i_batch,
                            draft_tokens);

            while (accepted_draft < draft_tokens.size() &&
                    accepted_draft < produced.size() &&
                    produced[accepted_draft] ==
                            draft_tokens[accepted_draft]) {
                committed.push_back(
                        produced[accepted_draft]);
                ++accepted_draft;
            }

            if (!produced.empty()) {
                next_pending =
                        produced.back();
            }
        } else {
            for (size_t row = 0;
                    row < batch_tokens.size();
                    ++row) {
                const llama_token sampled =
                        common_sampler_sample(
                                sampler_work.get(),
                                ctx_.get(),
                                static_cast<int32_t>(row));

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
        }

        if (ngram_diag_enabled()) {
            for (size_t row = 0;
                    row < produced.size();
                    ++row) {
                fprintf(stderr,
                        "NGRAM_DIAG_LINEAR_SAMPLE: "
                        "position=%d row=%zu "
                        "evaluated=%d sampled=%d\n",
                        (int) (
                                round_start +
                                static_cast<llama_pos>(row)),
                        row,
                        (int) batch_tokens[row],
                        (int) produced[row]);
            }
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

        if (ngram_native_framework_) {
            common_speculative_accept(
                    ngram_native_tree_spec_.get(),
                    sequence_id_,
                    static_cast<uint16_t>(accepted_draft));
        }

        ready_tokens_.clear();

        for (const llama_token token :
                produced) {
            ready_tokens_.push_back(
                    token);
        }

        fprintf(stderr,
                "NGRAM_SPEC_ROUND: mode=linear framework=%s "
                "position=%d matched_prefix=%u "
                "suffixes=%zu draft_frequency=%u "
                "draft_len=%zu evaluated=%zu "
                "accepted_draft=%zu committed=%zu "
                "queued=%zu until_opt=%u remaining=%d\n",
                ngram_native_framework_ ? "native" : "custom",
                (int) round_start,
                ngram_last_matched_prefix_,
                ngram_last_suffix_count_,
                ngram_last_draft_frequency_,
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




bool
server_generation_decode_executor::
try_ngram_tree_speculative_round(
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

    // The root itself consumes one committed Decode position. If the next
    // optimization boundary is immediately after the root, the sequential
    // path is cheaper and preserves exactly the same lifecycle.
    if (until_opt == 0 || until_opt == 1) {
        return false;
    }

    // B limits the number of frequency-ranked suffix paths retained before
    // shared-prefix merging. It is not a per-node child limit.
    //
    // Runtime B/D are deliberately read independently from Context capacity,
    // allowing parameter sweeps between requests without restarting server.
    const size_t retained_path_limit =
            ngram_max_branches_;

    const size_t tree_max_draft =
            ngram_max_draft_;

    const size_t suffix_depth =
            static_cast<size_t>(
                    ngram_trie_.ngram_size() -
                    ngram_trie_.prefix_length());

    size_t max_depth =
            std::min(
                    suffix_depth,
                    tree_max_draft);

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

    server_ngram_candidate_tree tree;

    if (ngram_native_framework_) {
        if (!ngram_native_tree_spec_) {
            return false;
        }

        const bool drafted =
                common_speculative_draft_tree(
                        ngram_native_tree_spec_.get(),
                        sequence_id_,
                        {
                            /* .n_paths = */ static_cast<int32_t>(retained_path_limit),
                            /* .n_max   = */ static_cast<int32_t>(max_depth),
                            /* .n_past  = */ next_position_,
                            /* .id_last = */ pending_token_,
                            /* .prompt  = */ &ngram_history_,
                        },
                        ngram_native_tree_draft_);

        if (!drafted) {
            fprintf(stderr,
                    "NGRAM_SPEC_MISS: mode=tree framework=native "
                    "position=%d history=%zu\n",
                    (int) next_position_,
                    ngram_history_.size());
            return false;
        }

        tree.matched_prefix_length =
                ngram_native_tree_draft_
                        .matched_prefix_length;
        tree.source_suffix_count =
                ngram_native_tree_draft_
                        .retained_path_count;
        tree.nodes.reserve(
                ngram_native_tree_draft_.nodes.size());

        for (const auto & native_node :
                ngram_native_tree_draft_.nodes) {
            server_ngram_tree_node node;
            node.token = native_node.token;
            node.parent = native_node.parent;
            node.weight = native_node.score;
            node.children = native_node.children;

            if (node.parent >= 0) {
                if (static_cast<size_t>(node.parent) >=
                        tree.nodes.size()) {
                    throw std::runtime_error(
                            "native N-Gram tree is not parent-before-child");
                }

                node.depth =
                        tree.nodes[node.parent].depth + 1;
            }

            tree.nodes.push_back(
                    std::move(node));
        }
    } else {
        auto lookup =
                ngram_trie_.lookup(
                        ngram_history_,
                        pending_token_);

        const size_t min_prefix_length =
                ngram_min_prefix_length_;

        if (!lookup.suffixes.empty() &&
                lookup.matched_prefix_length <
                        min_prefix_length) {
            fprintf(
                    stderr,
                    "NGRAM_SPEC_PREFIX_REJECT: "
                    "position=%d history=%zu "
                    "matched_prefix=%u min_prefix=%zu\n",
                    (int) next_position_,
                    ngram_history_.size(),
                    lookup.matched_prefix_length,
                    min_prefix_length);

            return false;
        }

        // Trie lookup is already frequency-first. Retain at most B complete
        // suffix paths before merging their shared prefixes into the tree.
        if (lookup.suffixes.size() >
                retained_path_limit) {
            lookup.suffixes.resize(
                    retained_path_limit);
        }

        if (lookup.suffixes.empty()) {
            fprintf(stderr,
                    "NGRAM_SPEC_MISS: mode=tree framework=custom "
                    "position=%d history=%zu\n",
                    (int) next_position_,
                    ngram_history_.size());
            return false;
        }

        tree =
                server_ngram_build_candidate_tree(
                        pending_token_,
                        lookup,
                        1 +
                                retained_path_limit *
                                max_depth,
                        max_depth,
                        llama_vocab_eos(vocab));
    }

    if (!tree.hit()) {
        return false;
    }

    // seq0 plus one temporary sequence per retained suffix path. Shared
    // prefixes can reduce the actual leaf/temporary-sequence count.
    const size_t TREE_SEQ_BUDGET =
            retained_path_limit + 1;

    server_ngram_verification_plan plan;

    if (!server_ngram_build_verification_plan(
                tree,
                TREE_SEQ_BUDGET,
                plan)) {
        fprintf(stderr,
                "NGRAM_SPEC_PLAN_FALLBACK: "
                "nodes=%zu matched_prefix=%u\n",
                tree.nodes.size(),
                tree.matched_prefix_length);
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

    // Match llama_conflayer::verify_tree(): shallow nodes first.
    // Parent-before-child alone is not strong enough for the in-batch
    // tree causal-mask assumptions used by the existing verifier.
    std::vector<int32_t> verification_order(
            tree.nodes.size());

    for (size_t i = 0;
            i < tree.nodes.size();
            ++i) {
        verification_order[i] =
                static_cast<int32_t>(i);
    }

    std::stable_sort(
            verification_order.begin(),
            verification_order.end(),
            [&](int32_t a, int32_t b) {
                if (tree.nodes[a].depth !=
                        tree.nodes[b].depth) {
                    return tree.nodes[a].depth <
                            tree.nodes[b].depth;
                }

                return a < b;
            });

    std::vector<int32_t> node_to_batch_row(
            tree.nodes.size(),
            -1);

    generation_batch_guard batch(
            static_cast<int32_t>(
                    tree.nodes.size()),
            static_cast<int32_t>(
                    plan.branch_count() + 1));

    for (size_t row = 0;
            row < verification_order.size();
            ++row) {
        const int32_t node_id =
                verification_order[row];

        const auto & node =
                tree.nodes[node_id];

        node_to_batch_row[node_id] =
                static_cast<int32_t>(row);

        common_batch_add(
                batch.batch,
                node.token,
                next_position_ +
                        static_cast<llama_pos>(
                                node.depth),
                plan.node_seq_ids[node_id],
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

        if (ngram_native_framework_) {
            auto native_result =
                    common_speculative_sample_and_accept_tree(
                            sampler_work.get(),
                            ctx_.get(),
                            ngram_native_tree_draft_,
                            node_to_batch_row);

            produced =
                    std::move(native_result.tokens);
            current_node =
                    native_result.last_accepted_node;

            for (const int32_t accepted_node :
                    native_result.accepted_nodes) {
                if (accepted_node <= 0 ||
                        static_cast<size_t>(accepted_node) >=
                                tree.nodes.size()) {
                    throw std::runtime_error(
                            "native N-Gram verifier returned an invalid accepted node");
                }

                committed.push_back(
                        tree.nodes[accepted_node].token);
            }

            if (!produced.empty()) {
                next_pending =
                        produced.back();
            }
        } else {
            while (true) {
                if (current_node < 0 ||
                        static_cast<size_t>(current_node) >=
                                tree.nodes.size()) {
                    throw std::runtime_error(
                            "N-Gram verifier reached invalid tree node");
                }

                // Every tree node requested logits, so output row i
                // corresponds directly to candidate-tree node i.
                const llama_token sampled =
                        common_sampler_sample(
                                sampler_work.get(),
                                ctx_.get(),
                                node_to_batch_row[current_node]);

                common_sampler_accept(
                        sampler_work.get(),
                        sampled,
                        true);

                produced.push_back(sampled);

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
                    next_pending = sampled;
                    break;
                }

                committed.push_back(sampled);
                current_node = matched_child;
            }
        }

        if (next_pending == LLAMA_TOKEN_NULL ||
                committed.empty() ||
                produced.empty()) {
            throw std::runtime_error(
                    "N-Gram verifier produced an invalid result");
        }

        if (ngram_diag_enabled()) {
            for (size_t i = 0;
                    i < produced.size();
                    ++i) {
                fprintf(
                        stderr,
                        "NGRAM_DIAG_TREE_SAMPLE: "
                        "framework=%s position=%d step=%zu "
                        "sampled=%d accepted=%d\n",
                        ngram_native_framework_ ? "native" : "custom",
                        (int) (
                                next_position_ +
                                static_cast<llama_pos>(i)),
                        i,
                        (int) produced[i],
                        i + 1 < committed.size() ? 1 : 0);
            }
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

        if (ngram_native_framework_) {
            common_speculative_accept(
                    ngram_native_tree_spec_.get(),
                    sequence_id_,
                    static_cast<uint16_t>(
                            committed.size() - 1));
        }

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
                "NGRAM_SPEC_ROUND: mode=tree framework=%s position=%d "
                "matched_prefix=%u retained_paths=%zu "
                "tree_nodes=%zu branches=%zu "
                "accepted_draft=%zu committed=%zu "
                "queued=%zu until_opt=%u remaining=%d\n",
                ngram_native_framework_ ? "native" : "custom",
                (int) (next_position_ -
                        static_cast<llama_pos>(
                                committed.size())),
                tree.matched_prefix_length,
                tree.source_suffix_count,
                tree.nodes.size(),
                plan.branch_count(),
                committed.size() - 1,
                committed.size(),
                produced.size(),
                until_opt,
                remaining_output_tokens);

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
    if (ngram_native_tree_spec_) {
        common_speculative_print_stats(
                ngram_native_tree_spec_.get());
    }
    ngram_native_tree_spec_.reset();
    ngram_native_draft_.clear();
    ngram_native_tree_draft_.clear();
    ngram_last_matched_prefix_ = 0;
    ngram_last_draft_frequency_ = 0;
    ngram_last_suffix_count_ = 0;
    ngram_native_framework_ = true;
    ngram_min_prefix_length_ = 1;
    ngram_max_branches_ = 3;
    ngram_max_draft_ = 3;
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
