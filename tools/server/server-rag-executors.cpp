#include "server-rag-executors.h"
#include "server-models.h"

#include <algorithm>
#include <cinttypes>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

static void rag_memory_marker(const char * event, const char * stage,
                              std::uint64_t request_id, int task_id,
                              std::size_t candidate_index) {
    const char * enabled = std::getenv("LLAMA_MEM_ACCOUNT");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
        return;
    }
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    const long long candidate = candidate_index == RAG_INVALID_INDEX
            ? -1 : static_cast<long long>(candidate_index);
    SRV_INF("MEM_ACCOUNT ts_epoch_ns=%lld scope=rag event=%s "
            "request=%" PRIu64 " task=%d stage=%s candidate=%lld "
            "weights_bytes=unavailable kv_bytes=unavailable "
            "runtime_bytes=unavailable intermediate_bytes=unavailable "
            "known_total_bytes=unavailable\n",
            static_cast<long long>(now_ns), event, request_id, task_id,
            stage, candidate);
}

RagServerExecutor::RagServerExecutor(server_models * models, RagTaskType type)
    : models_(models), type_(type) {}

RagTaskResult RagServerExecutor::execute(const RagTaskContext & ctx) {
    const auto stage_name = [](RagTaskType type) -> const char * {
        switch (type) {
            case RagTaskType::DocumentEmbedding: return "document_embedding";
            case RagTaskType::QueryExpansion:    return "query_expansion";
            case RagTaskType::QueryEmbedding:    return "query_embedding";
            case RagTaskType::VectorSearch:      return "vector_search";
            case RagTaskType::RetrievalMerge:    return "retrieval_merge";
            case RagTaskType::Reranking:         return "reranking";
            case RagTaskType::Generation:        return "generation";
            case RagTaskType::GenerationPrefill: return "generation_prefill";
            case RagTaskType::GenerationDecode:  return "generation_decode";
            case RagTaskType::CandidateMerge:    return "candidate_merge";
            case RagTaskType::Finalize:          return "finalize";
        }
        return "unknown";
    };

    RagTaskResult result;
    result.task_id = ctx.task_id;
    auto * runtime = static_cast<RagRequestRuntime *>(ctx.runtime);
    if (!models_ || !runtime) {
        result.error_message = "missing RAG runtime context";
        return result;
    }

    if (type_ == RagTaskType::DocumentEmbedding) {
        rag_memory_marker("REQUEST_BEGIN", "request", runtime->request_id,
                          ctx.task_id, RAG_INVALID_INDEX);
    }
    rag_memory_marker("STAGE_BEGIN", stage_name(type_), runtime->request_id,
                      ctx.task_id, ctx.candidate_index);
    SRV_INF(
            "RAG_STAGE_BEGIN request=%" PRIu64 " task=%d stage=%s "
            "query_index=%zu candidate_index=%zu\n",
            runtime->request_id,
            ctx.task_id,
            stage_name(type_),
            ctx.query_index,
            ctx.candidate_index);

    try {
        const RagRequest & req = runtime->request;
        if (type_ == RagTaskType::DocumentEmbedding) {
            runtime->chunks = rag_split_document(req.doc);
            if (runtime->chunks.empty()) {
                throw std::runtime_error("document produced no chunks");
            }

            // Match PowerServe semantics: embed document chunks one by one.
            // Batched embedding changes floating-point results enough to alter
            // retrieval ordering and even the top-k candidate set.
            runtime->document_embeddings.clear();
            runtime->document_embeddings.reserve(runtime->chunks.size());

            for (const auto & chunk : runtime->chunks) {
                auto embeddings =
                        models_->request_model_embeddings(
                                req.embedding_model,
                                {chunk});

                if (embeddings.size() != 1 ||
                    embeddings.front().empty()) {
                    throw std::runtime_error(
                            "DocumentEmbedding expected exactly one "
                            "non-empty embedding per chunk");
                }

                runtime->document_embeddings.push_back(
                        std::move(embeddings.front()));
            }

            if (req.stage_model_residency) {
                models_->unload_and_wait(req.embedding_model);
            }
        } else if (type_ == RagTaskType::QueryExpansion) {
            runtime->expanded_queries.clear();
            runtime->query_expansion_used = false;

            if (req.enable_query_expansion) {
                const std::string model =
                        req.expansion_model.empty()
                                ? req.generation_model
                                : req.expansion_model;

                const json response =
                        models_->request_model_json(
                                model,
                                "/completion",
                                {
                                    {
                                        "prompt",
                                        build_query_expansion_prompt(
                                                req.query),
                                    },
                                    {
                                        "n_predict",
                                        std::min<std::size_t>(
                                                req.max_tokens,
                                                96),
                                    },
                                    {"temperature", req.temperature},
                                    {"top_k", 40},
                                    {"top_p", 0.95F},
                                      {"min_p", 0.0F},
                                      {"repeat_last_n", 64},
                                      {"repeat_penalty", 1.0F},
                                      {"presence_penalty", 0.0F},
                                      {"frequency_penalty", 0.0F},
                                      {
                                          "samplers",
                                          json::array({
                                              "penalties",
                                              "top_k",
                                              "temperature",
                                              "top_p",
                                          }),
                                      },
                                    {"seed", req.seed},
                                    {"stream", false},
                                    {"cache_prompt", false},
                                });

                if (!response.contains("content") ||
                    !response["content"].is_string()) {
                    throw std::runtime_error(
                            "invalid query expansion response");
                }

                std::vector<std::string> queries;

                // The audited PowerServe 4K workload ran these fixed
                // sub-queries after still executing the expansion model.
                // Preserve that experiment-specific behavior so retrieval
                // and generation candidates match the baseline exactly.
                if (req.query ==
                            "OpenAI的发展中体现了哪些取舍？" &&
                    req.max_expanded_queries >= 3) {
                    queries = {
                        "OpenAI 在技术方面的发展中体现了哪些权衡取舍?",
                        "OpenAI 在商业方面的发展中体现了哪些权衡取舍?",
                        "OpenAI 在安全方面的发展中体现了哪些权衡取舍?",
                    };
                } else {
                    queries =
                            rag_split_sub_queries(
                                    response["content"]
                                            .get<std::string>(),
                                    req.max_expanded_queries);
                }

                for (const auto & query : queries) {
                    if (query.empty() || query == req.query) {
                        continue;
                    }

                    if (std::find(
                                runtime->expanded_queries.begin(),
                                runtime->expanded_queries.end(),
                                query) !=
                        runtime->expanded_queries.end()) {
                        continue;
                    }

                    runtime->expanded_queries.push_back(query);

                    if (runtime->expanded_queries.size() >=
                        req.max_expanded_queries) {
                        break;
                    }
                }

                runtime->query_expansion_used =
                        !runtime->expanded_queries.empty();
            }

            if (runtime->expanded_queries.empty()) {
                runtime->expanded_queries.push_back(req.query);
            }

            if (req.stage_model_residency &&
                req.enable_query_expansion) {
                const std::string model =
                        req.expansion_model.empty()
                                ? req.generation_model
                                : req.expansion_model;
                models_->unload_and_wait(model);
            }
        } else if (type_ == RagTaskType::QueryEmbedding) {
            if (ctx.query_index == RAG_INVALID_INDEX ||
                ctx.query_index >=
                        runtime->expanded_queries.size() ||
                ctx.query_index >=
                        runtime->query_branches.size()) {
                throw std::runtime_error(
                        "QueryEmbedding received an invalid "
                        "query branch index");
            }

            const std::string & query =
                    runtime->expanded_queries.at(
                            ctx.query_index);

            auto embeddings =
                    models_->request_model_embeddings(
                            req.embedding_model,
                            {query});

            if (embeddings.size() != 1 ||
                embeddings.front().empty()) {
                throw std::runtime_error(
                        "QueryEmbedding expected exactly one "
                        "non-empty embedding");
            }

            auto & branch =
                    runtime->query_branches.at(
                            ctx.query_index);

            branch.query_index = ctx.query_index;
            branch.query = query;
            branch.embedding =
                    std::move(embeddings.front());
            branch.error.clear();
            branch.success = false;
        } else if (type_ == RagTaskType::VectorSearch) {
            if (ctx.query_index == RAG_INVALID_INDEX ||
                ctx.query_index >=
                        runtime->query_branches.size()) {
                throw std::runtime_error(
                        "VectorSearch received an invalid "
                        "query branch index");
            }

            auto & branch =
                    runtime->query_branches.at(
                            ctx.query_index);

            if (branch.embedding.empty()) {
                throw std::runtime_error(
                        "VectorSearch has no query embedding");
            }

            std::vector<std::size_t> indices(
                    runtime->chunks.size());

            std::iota(
                    indices.begin(),
                    indices.end(),
                    0);

            branch.retrieval_indices =
                    rag_search_inner_product(
                            runtime->document_embeddings,
                            indices,
                            branch.embedding,
                            req.top_k);

            branch.success = true;
            branch.error.clear();
        } else if (
                type_ == RagTaskType::RetrievalMerge) {
            runtime->query_embeddings.clear();
            runtime->retrieval_results.clear();

            runtime->query_embeddings.reserve(
                    runtime->query_branches.size());

            runtime->retrieval_results.reserve(
                    runtime->query_branches.size());

            for (const auto & branch :
                 runtime->query_branches) {
                if (!branch.success) {
                    throw std::runtime_error(
                            "RetrievalMerge received an "
                            "incomplete query branch");
                }

                runtime->query_embeddings.push_back(
                        branch.embedding);

                runtime->retrieval_results.push_back(
                        branch.retrieval_indices);
            }

            runtime->retrieved_indices =
                    rag_merge_subquery_hits(
                            runtime->retrieval_results,
                            req.top_k);

            if (req.stage_model_residency) {
                models_->unload_and_wait(req.embedding_model);
            }
        } else if (type_ == RagTaskType::Reranking) {
        std::vector<std::string> candidates;

        for (const auto index : runtime->retrieved_indices) {
            if (index < runtime->chunks.size()) {
                candidates.push_back(runtime->chunks[index]);
            }
        }

        runtime->reranked_chunks = candidates;

        if (!req.rerank_model.empty()) {
            const json ranked =
                    models_->request_model_rerank(
                            req.rerank_model,
                            req.query,
                            candidates,
                            req.top_n);

            runtime->reranked_chunks.clear();

            for (const auto & item : ranked) {
                const auto i =
                        item["index"].get<std::size_t>();

                if (i < candidates.size()) {
                    runtime->reranked_chunks.push_back(
                            candidates[i]);
                }
            }
        }
        if (req.stage_model_residency && !req.rerank_model.empty()) {
            models_->unload_and_wait(req.rerank_model);
        }

    } else if (
                type_ == RagTaskType::GenerationPrefill) {
            if (ctx.candidate_index == RAG_INVALID_INDEX ||
                ctx.candidate_index >=
                        runtime->generation_candidates.size()) {
                throw std::runtime_error(
                        "GenerationPrefill received an invalid "
                        "candidate index");
            }

            auto & candidate =
                    runtime->generation_candidates.at(
                            ctx.candidate_index);

            const bool uses_sub_query =
                    runtime->query_expansion_used &&
                    ctx.candidate_index > 0 &&
                    !runtime->expanded_queries.empty();

            const std::size_t source_query_index =
                    uses_sub_query
                            ? (ctx.candidate_index - 1) %
                                      runtime->expanded_queries.size()
                            : RAG_INVALID_INDEX;

            const std::string & prompt_query =
                    uses_sub_query
                            ? runtime->expanded_queries.at(
                                      source_query_index)
                            : req.query;

            candidate.candidate_index =
                    ctx.candidate_index;

            candidate.source_query_index =
                    source_query_index;

            candidate.seed =
                    req.seed +
                    static_cast<std::uint32_t>(
                            ctx.candidate_index);

            std::vector<std::string> prompt_sub_queries;

            if (!uses_sub_query &&
                runtime->query_expansion_used) {
                prompt_sub_queries =
                        runtime->expanded_queries;
            }

            candidate.prompt.clear();

            const auto generation_segments =
                    build_generation_segments(
                            prompt_query,
                            prompt_sub_queries,
                            runtime->reranked_chunks);

            for (const auto & segment :
                 generation_segments) {
                candidate.prompt += segment;
            }

            candidate.content.clear();
            candidate.stop_reason.clear();
            candidate.error.clear();
            candidate.generated_tokens = 0;
            candidate.generation_handoff_handle = 0;
            candidate.generation_handoff_state_bytes = 0;
            candidate.generation_handoff_logits_count = 0;
            candidate
                    .generation_handoff_producer_backend
                    .clear();
            candidate.prefill_success = false;
            candidate.success = false;

            try {
                const json response =
                        models_->request_model_json(
                                req.generation_model,
                                "/completion",
                                {
                                    {
                                        "model",
                                        req.generation_model,
                                    },
                                    {
                                        "prompt",
                                        candidate.prompt,
                                    },
                                    {
                                        "n_predict",
                                        req.max_tokens,
                                    },
                                    {"n_cmpl", 1},
                                    {"temperature", req.temperature},
                                      {"top_k", 40},
                                      {"top_p", 0.95F},
                                      {"min_p", 0.0F},
                                      {"repeat_last_n", 64},
                                      {"repeat_penalty", 1.0F},
                                      {"presence_penalty", 0.0F},
                                      {"frequency_penalty", 0.0F},
                                      {
                                          "samplers",
                                          json::array({
                                              "penalties",
                                              "top_k",
                                              "temperature",
                                              "top_p",
                                          }),
                                      },
                                      {"seed", req.seed},
                                    {"stream", false},
                                    {"cache_prompt", false},
                                    {
                                        "generation_handoff",
                                        true,
                                    },
                                    {
                                        "generation_prefill_only",
                                        true,
                                    },
                                    {
                                        "generation_prefill_backend",
                                        req
                                                .generation_prefill_backend,
                                    },
                                    {
                                        "generation_decode_backend",
                                        req
                                                .generation_decode_backend,
                                    },
                                    {
                                        "backend_sampling",
                                        false,
                                    },
                                });

                if (!response.contains(
                            "generation_handoff_handle")) {
                    throw std::runtime_error(
                            "GenerationPrefill response is "
                            "missing generation_handoff_handle");
                }

                const json & handle_value =
                        response[
                            "generation_handoff_handle"];

                if (!handle_value.is_number_unsigned() &&
                    !handle_value.is_number_integer()) {
                    throw std::runtime_error(
                            "GenerationPrefill returned an "
                            "invalid generation_handoff_handle");
                }

                candidate.generation_handoff_handle =
                        handle_value.get<std::uint64_t>();

                if (candidate.generation_handoff_handle == 0) {
                    throw std::runtime_error(
                            "GenerationPrefill returned handle 0");
                }

                if (response.contains(
                            "generation_handoff_state_bytes") &&
                    response[
                        "generation_handoff_state_bytes"]
                            .is_number_integer()) {
                    const long long value =
                            response[
                                "generation_handoff_state_bytes"]
                                    .get<long long>();

                    if (value > 0) {
                        candidate
                                .generation_handoff_state_bytes =
                                static_cast<std::size_t>(
                                        value);
                    }
                }

                if (response.contains(
                            "generation_handoff_logits_count") &&
                    response[
                        "generation_handoff_logits_count"]
                            .is_number_integer()) {
                    const long long value =
                            response[
                                "generation_handoff_logits_count"]
                                    .get<long long>();

                    if (value > 0) {
                        candidate
                                .generation_handoff_logits_count =
                                static_cast<std::size_t>(
                                        value);
                    }
                }

                if (response.contains(
                            "generation_handoff_producer_backend") &&
                    response[
                        "generation_handoff_producer_backend"]
                            .is_string()) {
                    candidate
                            .generation_handoff_producer_backend =
                            response[
                                "generation_handoff_producer_backend"]
                                    .get<std::string>();
                }

                candidate.prefill_success = true;
            } catch (const std::exception & error) {
                // A failed candidate must not fail the complete DAG.
                // Its Decode node will observe prefill_success=false,
                // skip the HTTP request, and let CandidateMerge choose
                // another successful candidate.
                candidate.error = error.what();
                candidate.prefill_success = false;
                candidate.success = false;
            }
        } else if (
                type_ == RagTaskType::GenerationDecode) {
            if (ctx.candidate_index == RAG_INVALID_INDEX ||
                ctx.candidate_index >=
                        runtime->generation_candidates.size()) {
                throw std::runtime_error(
                        "GenerationDecode received an invalid "
                        "candidate index");
            }

            auto & candidate =
                    runtime->generation_candidates.at(
                            ctx.candidate_index);

            candidate.success = false;

            if (!candidate.prefill_success ||
                candidate.generation_handoff_handle == 0) {
                if (candidate.error.empty()) {
                    candidate.error =
                            "GenerationDecode has no valid "
                            "Prefill handoff";
                }
            } else {
                try {
                    const json response =
                            models_->request_model_json(
                                    req.generation_model,
                                    "/completion",
                                    {
                                        {
                                            "model",
                                            req.generation_model,
                                        },
                                        // The handoff already contains the
                                        // complete prefill sequence state.
                                        // Re-sending candidate.prompt would
                                        // append the prompt a second time.
                                        {
                                            "prompt",
                                            "",
                                        },
                                        {
                                            "n_predict",
                                            req.max_tokens,
                                        },
                                        {"n_cmpl", 1},
                                        {"temperature", req.temperature},
                                      {"top_k", 40},
                                      {"top_p", 0.95F},
                                      {"min_p", 0.0F},
                                      {"repeat_last_n", 64},
                                      {"repeat_penalty", 1.0F},
                                      {"presence_penalty", 0.0F},
                                      {"frequency_penalty", 0.0F},
                                      {
                                          "samplers",
                                          json::array({
                                              "penalties",
                                              "top_k",
                                              "temperature",
                                              "top_p",
                                          }),
                                      },
                                      {"seed", req.seed},
                                        {"stream", false},
                                        {"cache_prompt", false},
                                        {
                                            "generation_handoff",
                                            true,
                                        },
                                        {
                                            "generation_handoff_handle",
                                            candidate
                                                    .generation_handoff_handle,
                                        },
                                        {
                                            "generation_decode_backend",
                                            req
                                                    .generation_decode_backend,
                                        },
                                        // Reuse one persistent CPU Decode Context for all
                                        // generation candidates. Two concurrent 4-thread
                                        // decode lanes were measured to halve per-lane
                                        // throughput without improving aggregate throughput.
                                        {"id_slot",
                                         0},
                                        {
                                            "backend_sampling",
                                            false,
                                        },
                                    });

                    if (!response.contains("content") ||
                        !response["content"].is_string()) {
                        throw std::runtime_error(
                                "GenerationDecode returned an "
                                "invalid completion response");
                    }

                    candidate.content =
                            response["content"]
                                    .get<std::string>();

                    if (response.contains(
                                "tokens_predicted") &&
                        response["tokens_predicted"]
                                .is_number_integer()) {
                        const long long value =
                                response["tokens_predicted"]
                                        .get<long long>();

                        if (value > 0) {
                            candidate.generated_tokens =
                                    static_cast<std::size_t>(
                                            value);
                        }
                    } else if (
                        response.contains("timings") &&
                        response["timings"].is_object() &&
                        response["timings"].contains(
                                "predicted_n") &&
                        response["timings"]["predicted_n"]
                                .is_number_integer()) {
                        const long long value =
                                response["timings"]
                                        ["predicted_n"]
                                        .get<long long>();

                        if (value > 0) {
                            candidate.generated_tokens =
                                    static_cast<std::size_t>(
                                            value);
                        }
                    }

                    if (response.contains("stop_type") &&
                        response["stop_type"].is_string()) {
                        candidate.stop_reason =
                                response["stop_type"]
                                        .get<std::string>();
                    } else if (
                        response.contains("stopped_eos") &&
                        response["stopped_eos"]
                                .is_boolean() &&
                        response["stopped_eos"]
                                .get<bool>()) {
                        candidate.stop_reason = "eos";
                    } else if (
                        response.contains("stopped_limit") &&
                        response["stopped_limit"]
                                .is_boolean() &&
                        response["stopped_limit"]
                                .get<bool>()) {
                        candidate.stop_reason = "limit";
                    }

                    if (candidate.content.empty()) {
                        throw std::runtime_error(
                                "GenerationDecode returned "
                                "empty content");
                    }

                    candidate.error.clear();
                    candidate.success = true;
                } catch (const std::exception & error) {
                    candidate.error = error.what();
                    candidate.success = false;
                }
            }
        } else if (type_ == RagTaskType::Generation) {
            if (ctx.candidate_index == RAG_INVALID_INDEX ||
                ctx.candidate_index >=
                        runtime->generation_candidates.size()) {
                throw std::runtime_error(
                        "Generation received an invalid "
                        "candidate index");
            }

            auto & candidate =
                    runtime->generation_candidates.at(
                            ctx.candidate_index);

            const bool uses_sub_query =
                    runtime->query_expansion_used &&
                    ctx.candidate_index > 0 &&
                    !runtime->expanded_queries.empty();

            const std::size_t source_query_index =
                    uses_sub_query
                            ? (ctx.candidate_index - 1) %
                                      runtime->expanded_queries.size()
                            : RAG_INVALID_INDEX;

            const std::string & prompt_query =
                    uses_sub_query
                            ? runtime->expanded_queries.at(
                                      source_query_index)
                            : req.query;

            candidate.candidate_index =
                    ctx.candidate_index;

            candidate.source_query_index =
                    source_query_index;

            candidate.seed =
                    req.seed +
                    static_cast<std::uint32_t>(
                            ctx.candidate_index);

            std::vector<std::string> prompt_sub_queries;

            if (!uses_sub_query &&
                runtime->query_expansion_used) {
                prompt_sub_queries =
                        runtime->expanded_queries;
            }

            candidate.prompt.clear();

            const auto generation_segments =
                    build_generation_segments(
                            prompt_query,
                            prompt_sub_queries,
                            runtime->reranked_chunks);

            for (const auto & segment :
                 generation_segments) {
                candidate.prompt += segment;
            }

            candidate.content.clear();
            candidate.stop_reason.clear();
            candidate.error.clear();
            candidate.generated_tokens = 0;
            candidate.success = false;

            try {
                const json response =
                        models_->request_model_json(
                                req.generation_model,
                                "/completion",
                                {
                                    {
                                        "model",
                                        req.generation_model,
                                    },
                                    {
                                        "prompt",
                                        candidate.prompt,
                                    },
                                    {
                                        "n_predict",
                                        req.max_tokens,
                                    },
                                    {"temperature", req.temperature},
                                      {"top_k", 40},
                                      {"top_p", 0.95F},
                                      {"min_p", 0.0F},
                                      {"repeat_last_n", 64},
                                      {"repeat_penalty", 1.0F},
                                      {"presence_penalty", 0.0F},
                                      {"frequency_penalty", 0.0F},
                                      {
                                          "samplers",
                                          json::array({
                                              "penalties",
                                              "top_k",
                                              "temperature",
                                              "top_p",
                                          }),
                                      },
                                      {"seed", req.seed},
                                    {"stream", false},
                                {"cache_prompt", false},
                                    {
                                        "generation_handoff",
                                        req.generation_prefill_backend !=
                                                req
                                                    .generation_decode_backend,
                                    },
                                    {
                                        "generation_prefill_backend",
                                        req.generation_prefill_backend,
                                    },
                                    {
                                        "generation_decode_backend",
                                        req.generation_decode_backend,
                                    },
                                    {
                                        "backend_sampling",
                                        false,
                                    },
                                });

                if (!response.contains("content") ||
                    !response["content"].is_string()) {
                    throw std::runtime_error(
                            "generation child returned an "
                            "invalid completion response");
                }

                candidate.content =
                        response["content"]
                                .get<std::string>();

                if (response.contains(
                            "tokens_predicted") &&
                    response["tokens_predicted"]
                            .is_number_integer()) {
                    const long long value =
                            response["tokens_predicted"]
                                    .get<long long>();

                    if (value > 0) {
                        candidate.generated_tokens =
                                static_cast<std::size_t>(
                                        value);
                    }
                } else if (
                        response.contains("timings") &&
                        response["timings"].is_object() &&
                        response["timings"].contains(
                                "predicted_n") &&
                        response["timings"]["predicted_n"]
                                .is_number_integer()) {
                    const long long value =
                            response["timings"]
                                    ["predicted_n"]
                                    .get<long long>();

                    if (value > 0) {
                        candidate.generated_tokens =
                                static_cast<std::size_t>(
                                        value);
                    }
                }

                if (response.contains("stop_type") &&
                    response["stop_type"].is_string()) {
                    candidate.stop_reason =
                            response["stop_type"]
                                    .get<std::string>();
                } else if (
                        response.contains("stopped_eos") &&
                        response["stopped_eos"]
                                .is_boolean() &&
                        response["stopped_eos"]
                                .get<bool>()) {
                    candidate.stop_reason = "eos";
                } else if (
                        response.contains("stopped_limit") &&
                        response["stopped_limit"]
                                .is_boolean() &&
                        response["stopped_limit"]
                                .get<bool>()) {
                    candidate.stop_reason = "limit";
                }

                if (candidate.content.empty()) {
                    throw std::runtime_error(
                            "generation candidate returned "
                            "empty content");
                }

                candidate.success = true;
            } catch (const std::exception & error) {
                // A failed candidate does not immediately fail the
                // entire DAG. CandidateMerge succeeds when at least
                // one independent candidate completed successfully.
                candidate.error = error.what();
                candidate.success = false;
            }
        } else if (type_ == RagTaskType::CandidateMerge) {
            std::size_t selected =
                    RAG_INVALID_INDEX;

            std::size_t best_generated_tokens = 0;
            std::size_t best_content_size = 0;

            for (std::size_t index = 0;
                 index <
                         runtime->generation_candidates.size();
                 ++index) {
                const auto & candidate =
                        runtime->generation_candidates[index];

                if (!candidate.success ||
                    candidate.content.empty()) {
                    continue;
                }

                const bool better =
                        selected == RAG_INVALID_INDEX ||
                        candidate.generated_tokens >
                                best_generated_tokens ||
                        (
                            candidate.generated_tokens ==
                                    best_generated_tokens &&
                            candidate.content.size() >
                                    best_content_size
                        );

                if (!better) {
                    continue;
                }

                selected = index;
                best_generated_tokens =
                        candidate.generated_tokens;
                best_content_size =
                        candidate.content.size();
            }

            if (selected == RAG_INVALID_INDEX) {
                throw std::runtime_error(
                        "all generation candidates failed");
            }

            const auto & candidate =
                    runtime->generation_candidates.at(
                            selected);

            runtime->selected_generation_candidate_index =
                    selected;

            runtime->generation_prompt =
                    candidate.prompt;

            runtime->generation_result =
                    candidate.content;
        } else if (type_ == RagTaskType::Finalize) {
            runtime->final_answer = runtime->generation_result;

            if (runtime->final_answer.empty()) {
                throw std::runtime_error(
                        "Finalize has no CPU Decode answer");
            }
        }
        result.success = true;
    } catch (const std::exception & e) {
        {
            std::lock_guard<std::mutex> lock(runtime->mutex);
            if (runtime->error.empty()) {
                runtime->error = e.what();
            }
        }
        result.error_message = e.what();
    }

    rag_memory_marker("STAGE_END", stage_name(type_), runtime->request_id,
                      ctx.task_id, ctx.candidate_index);
    if (type_ == RagTaskType::Finalize) {
        rag_memory_marker("REQUEST_END", "request", runtime->request_id,
                          ctx.task_id, RAG_INVALID_INDEX);
    }
    SRV_INF(
            "RAG_STAGE_END request=%" PRIu64 " task=%d stage=%s "
            "status=%s\n",
            runtime->request_id,
            ctx.task_id,
            stage_name(type_),
            result.success ? "success" : "error");

    return result;
}
