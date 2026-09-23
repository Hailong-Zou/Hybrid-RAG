#pragma once

#include <cstddef>
#include <limits>
#include <cstdint>
#include <atomic>
#include <string>
#include <vector>
#include <mutex>
#include <memory>


constexpr std::size_t RAG_INVALID_INDEX =
        std::numeric_limits<std::size_t>::max();

struct RagQueryBranchResult {
    std::size_t query_index = RAG_INVALID_INDEX;
    std::string query;
    std::vector<float> embedding;
    std::vector<std::size_t> retrieval_indices;
    std::string error;
    bool success = false;
};

struct RagGenerationCandidate {
    std::size_t candidate_index = RAG_INVALID_INDEX;
    std::size_t source_query_index = RAG_INVALID_INDEX;
    std::uint32_t seed = 0;

    std::string prompt;
    std::string content;
    std::string stop_reason;
    std::string error;

    std::size_t generated_tokens = 0;

    // GenerationPrefill stores a one-shot handoff here. The matching
    // GenerationDecode node for this candidate consumes the handle.
    std::uint64_t generation_handoff_handle = 0;
    std::size_t generation_handoff_state_bytes = 0;
    std::size_t generation_handoff_logits_count = 0;
    std::string generation_handoff_producer_backend;
    bool prefill_success = false;

    bool success = false;
};

struct RagGenerationSubMetrics {
    std::size_t prefill_ms = 0;
    std::size_t export_ms = 0;
    std::size_t restore_ms = 0;
    std::size_t decode_ms = 0;
    std::size_t merge_ms = 0;
};

struct RagRequest {
    std::string doc;
    std::string query;

    std::string mode = "sequential";
    bool enable_query_expansion = false;
    bool unload_expansion_model_after_use = false;
    bool stage_model_residency = false;

    std::size_t max_expanded_queries = 3;

    std::string generation_model;
    std::string embedding_model;
    std::string rerank_model;
    std::string expansion_model;

    std::size_t top_k = 20;
    std::size_t top_n = 5;
    std::size_t max_tokens = 128;
    std::size_t generation_decode_steps = 64;
    std::size_t generation_subquery_decode_steps = 64;
    std::size_t generation_candidate_repeats = 1;

    std::uint32_t seed = 1234;

    std::string document_embedding_backend = "auto";
    std::string query_expansion_backend = "auto";
    std::string query_embedding_backend = "auto";
    std::string reranking_backend = "auto";
    std::string generation_prefill_backend = "auto";
    std::string generation_decode_backend = "auto";

    float temperature = 0.2F;
};

struct RagStageMetrics {
    std::size_t document_embedding_ms = 0;
    std::size_t query_expansion_ms = 0;
    std::size_t query_embedding_ms = 0;
    std::size_t vector_search_ms = 0;
    std::size_t retrieval_merge_ms = 0;
    std::size_t reranking_ms = 0;
    std::size_t generation_ms = 0;
    std::size_t generation_merge_ms = 0;
    std::size_t total_ms = 0;
};

struct RagRequestRuntime {
    std::uint64_t request_id = 0;
    RagRequest request;

    std::atomic<bool> cancelled { false };

    std::vector<std::string> chunks;
    std::vector<std::vector<float>> document_embeddings;
    std::vector<std::string> expanded_queries;
    bool query_expansion_used = false;
    // Transitional batch fields retained until the fan-out DAG commit.
    std::vector<std::vector<float>> query_embeddings;
    std::vector<std::vector<std::size_t>> retrieval_results;

    // Scheduler-visible per-query branch state.
    std::vector<RagQueryBranchResult> query_branches;

    std::vector<std::size_t> retrieved_indices;
    std::vector<std::string> reranked_chunks;
    std::string generation_prompt;
    std::string generation_result;

    // Scheduler-visible per-candidate generation state.
    std::vector<RagGenerationCandidate> generation_candidates;
    std::size_t selected_generation_candidate_index =
            RAG_INVALID_INDEX;

    std::string final_answer;
    RagStageMetrics stage_metrics;
    RagGenerationSubMetrics generation_sub_metrics;
    std::string error;
    std::mutex mutex;
};

std::string rag_trim(std::string value);

std::vector<std::string> rag_split_sub_queries(
        const std::string & expanded,
        std::size_t max_subqueries = 3,
        std::size_t max_subquery_chars = 128);

std::string build_query_expansion_prompt(
        const std::string & query);

std::vector<std::string> rag_split_document(
        const std::string & document);

std::vector<std::size_t> rag_search_inner_product(
        const std::vector<std::vector<float>> & doc_embeddings,
        const std::vector<std::size_t> & doc_embedding_source_indices,
        const std::vector<float> & query_embedding,
        std::size_t top_k);

std::vector<std::size_t> rag_merge_subquery_hits(
        const std::vector<std::vector<std::size_t>> & per_query_hits,
        std::size_t top_k);

std::string build_generation_prompt(
        const std::string & query,
        const std::vector<std::string> & context_chunks);

std::vector<std::string> build_generation_segments(
        const std::string & query,
        const std::vector<std::string> & sub_queries,
        const std::vector<std::string> & context_chunks);
