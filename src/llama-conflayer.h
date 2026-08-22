// Standalone ConfLayers self-speculative decoder.
//
// Port of conflayer/ConfLayers (Python) into llama.cpp as an independent module.
// The full model is used with a per-layer skip set as the draft model (all weights
// shared). Each round:
//   1. greedy draft along the skip-graph, collecting top-k tokens + confidences
//   2. build a candidate "bouquet" tree from the confidences
//   3. verify the whole tree with the full model in a single forward
//   4. accept the longest greedy-matching prefix, recycle the accepted KV, repeat
//
// The module is decoupled from the rest of the codebase: it only uses the public
// llama.h API and owns a single context internally. See llama_conflayer_params.
//
// KV design: like the Python original (one shared past_key_values), the module uses a
// SINGLE context whose KV cache is shared between the draft phase and the verify phase.
// `draft()` toggles the layer_skip graph on and decodes new positions through the skip
// model, reading the full-model KV history that prefill/verify wrote; `verify_tree()`
// toggles it off and re-runs the full model, overwriting the draft cells. This is what
// makes the draft see exactly the same history as the full model.

#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// Hard ceiling for the verification tree seq/output budget. Matches LLAMA_MAX_SEQ in
// llama-cparams.h: the batch allocator rejects seq_id >= n_seq_max, so tree leaves must
// stay below it. The actual budget is params.max_seq (default 32); the context is sized
// for that budget instead of LLAMA_MAX_SEQ so prefill does not pay for a 256-seq graph.
#define LLAMA_CONFLAYER_MAX_SEQ 256

// Configuration for llama_conflayer
struct llama_conflayer_params {
    uint32_t n_ctx           = 4096;   // context size of each internal context
    uint32_t n_batch         = 512;    // logical max batch (must fit the verification tree)
    uint32_t n_ubatch        = 512;    // physical max batch (must fit the verification tree)

    // CPU threads for the internal context. 0 = all cores (std::thread::hardware_concurrency).
    // The context defaults to GGML_DEFAULT_N_THREADS (4) when this is left 0, which silently
    // halves decode throughput versus a caller running the full model at -t 8.
    int32_t  n_threads       = 0;

    uint32_t max_new_tokens  = 512;    // max new tokens to generate
    uint32_t max_step_draft  = 25;     // max draft steps per round
    uint32_t topk            = 5;      // top-k candidates kept per draft step

    // Budget of sequences/outputs the verification tree may use (the context's
    // n_seq_max / n_outputs_max). Prefill and generation pay for a graph sized to this,
    // so it should be just enough to cover the practical tree: the tree uses one seq per
    // leaf plus the shared history seq, and one output per node. 32 covers ~6 draft steps
    // at the full candidate width (1 + 6*5 = 31 nodes). Bounded by LLAMA_CONFLAYER_MAX_SEQ.
    uint32_t max_seq          = 32;

    float    stop_threshold  = 0.8f;   // drafting stops below this top-1 confidence

    // dynamic_optimization layer-set search (Phase 5)
    float    skip_ratio      = 0.6f;   // target fraction of skipped layers
    int32_t  max_skip_layers = 0;      // hard cap on skipped layers (0 = unlimited, current
                                       // behavior). When set, only the N layers with the
                                       // largest decision margin (most justified to skip) are
                                       // skipped and the borderline ones are restored, which
                                       // keeps the skip graph closest to the full model.
    bool     optimize        = false;  // enable the layer-set search
    uint32_t search_interval = 25;     // rounds between layer-set optimizations
    uint32_t context_window  = 32;     // validation window used by the search
    uint32_t max_opt_iter    = 1000;   // max optimization iterations
    float    max_score       = 0.95f;  // stop optimizing once the acceptance score reaches this

    // false (default) = pure layer-skip decode: the skip graph decodes every token and its
    // top-1 is emitted directly. true = legacy self-speculative decoding (draft + verify tree).
    bool     verify          = false;
    float    stop_agree      = 0.95f;  // pure-skip: freeze the skip set once full-model agreement reaches this
    float    low_agree       = 0.6f;   // pure-skip: below this agreement, restore the best skip set

    // flash-DRAM weight offloading. layer_chunks = number of rounds to stream the whole
    // model in; each round owns a ceil(n_layer/layer_chunks)-wide window of layer indices
    // and streams only its active (non-resident, non-skip) layers, so a window with 3 of 9
    // layers skipped loads 6. 0 = disabled (full load to DRAM).
    int32_t  layer_chunks    = 0;      // 0 = disabled
    // pure-skip resident mode: after prefill + the first layer-set optimization, pin the
    // first keep_active_resident active (non-skip) layers in RAM and stream the rest (skip
    // layers + remaining active layers) once per phase. 0 = off (current full-stream pageout,
    // backward compatible); > 0 = first K active layers resident; negative or >= n_layer =
    // all active layers resident (max speed). Ignored under verify (SD uses its own resident).
    int32_t  keep_active_resident = 0;
    bool     cluster_attn    = false;  // enable cluster-based KV cache on the shared context
                                       // (cluster bookkeeping init/update/persist/evict always
                                       // runs; the attention mode is chosen below)
    bool     cluster_sparse  = false;  // true cluster attention: the verify phase computes real
                                       // sparse attention (get_k/get_v return the top-k cluster
                                       // subset); the draft phase still keeps the full KV view.
                                       // When false, sparse cluster attention stays off and the
                                       // verify phase keeps the full KV view (bookkeeping only).

    int32_t  seed            = 2024;   // RNG seed (reserved for the non-greedy path)
    uint32_t verbosity       = 1;      // 0 = quiet, 1 = per-round summary, 2 = debug

    std::string model_id     = "qwen3"; // architecture id, used by dynamic_optimization
};

// Statistics collected during a generate() call
struct llama_conflayer_metrics {
    uint64_t n_rounds       = 0;       // verification rounds
    uint64_t n_draft_tokens = 0;       // tokens drafted by the draft model
    uint64_t n_tree_nodes   = 0;       // total verification tree nodes decoded by the full model
    uint64_t n_accepted     = 0;       // accepted draft tokens (excluding the verified root)
    uint64_t n_opt_rounds   = 0;       // dynamic_optimization rounds
    float    accept_rate    = 0.0f;    // SD only: n_accepted / n_draft_tokens
    float    agree_rate     = 0.0f;    // pure-skip: mean full-model agreement over optimizations

    uint64_t prefill_us     = 0;       // prefill phase wall time (microseconds)
    uint64_t decode_us      = 0;       // draft+verify generation loop wall time (microseconds)
    uint64_t draft_us       = 0;       // draft phase wall time (skip-graph single-token decodes)
    uint64_t verify_us      = 0;       // verify phase wall time (full-model tree decode)
    uint64_t opt_us         = 0;       // dynamic_optimization wall time (validation + layer-set search)

    std::vector<uint32_t> accept_lengths; // accepted tokens per round (incl. the root)

    // pure-skip skip-set statistics over the generated tokens (0 under verify).
    uint64_t skip_count_sum = 0;       // summed skip-set size per generated token
    uint32_t skip_count_min = 0;       // min skip-set size during generation
    uint32_t skip_count_max = 0;       // max skip-set size during generation
    // entropy skip threshold per layer-set update. find_skip_layers skips a layer when its
    // z-scored confidence falls below a per-layer local threshold (lmu - lam_*lstd). Each
    // entry here is that decision boundary converted to normalized entropy units (0..1,
    // entropy / log(n_embd)); layers above the boundary get skipped. NOTE: confidence =
    // 1 - entropy/max_entropy, so the code skips relatively HIGH-entropy (low-confidence)
    // layers, the opposite of a "skip when confident" heuristic.
    std::vector<float> opt_entropy_thresholds;

    // per-layer confidence / entropy snapshot per layer-set optimization round.
    // rows = optimization rounds, columns = layers 1..n_layer-2 (the skippable range).
    // confidence = attn+ffn module confidence sum (0..2); entropy = normalized total
    // entropy = 2 - confidence (0..2, higher = noisier / lower-confidence).
    std::vector<std::vector<float>> opt_layer_confidence;
    std::vector<std::vector<float>> opt_layer_entropy;
};

class llama_conflayer {
public:
    // Creates the internal contexts. The model is not owned by the module.
    llama_conflayer(const llama_model * model, const llama_conflayer_params & params);

    ~llama_conflayer();

    llama_conflayer(const llama_conflayer &) = delete;
    llama_conflayer & operator=(const llama_conflayer &) = delete;

    // Generates up to params.max_new_tokens tokens for `prompt`, appending them to
    // `result`. Returns false on failure. If `metrics` is non-null it is filled with
    // the round statistics.
    bool generate(const std::vector<llama_token> & prompt,
                  std::vector<llama_token>       & result,
                  llama_conflayer_metrics        * metrics = nullptr);

    // Single-token answer eval for a benchmark prompt. One prefill serves both predictions:
    //   baseline_answer = full-model argmax at the answer position (the root generate()
    //                     would consume);
    //   skip_answer     = skip-graph argmax at the answer position, produced by running the
    //                     first layer-set optimization (same as generate()) and re-decoding
    //                     the LAST prompt token through the skip graph.
    // This measures the skip graph's own answer -- distinct from generate(), whose first
    // emitted token is the continuation AFTER the full-model root. Returns false on failure.
    //
    // `letter_toks` (optional) is an array of n_letter candidate token ids (one per answer
    // letter). When given, the answer-position logits of those candidates are copied into
    // `baseline_logits_out` / `skip_logits_out` (each n_letter floats), captured within the
    // same decode iteration so they are never invalidated. This supports the standard
    // multiple-choice letter-scoring (argmax over the candidate logits) as an alternative
    // to comparing the raw argmax token ids.
    bool answer_token(const std::vector<llama_token> & prompt,
                      int32_t                          n_letter,
                      const int32_t                  * letter_toks,
                      float                          * baseline_logits_out,
                      float                          * skip_logits_out,
                      llama_token                    & baseline_answer,
                      llama_token                    & skip_answer,
                      llama_conflayer_metrics        * metrics = nullptr);

    // Replaces the draft skip set. A skipped layer is bypassed entirely: residual
    // pass-through, no attention/FFN and no KV write. The draft graph is rebuilt on
    // the next decode.
    void set_skip_layers(const std::vector<int32_t> & skip_layers);
    const std::vector<int32_t> & get_skip_layers() const { return skip_layers_; }

    const llama_conflayer_params & get_params() const { return params_; }

private:
    // one greedy draft round (port of swift_draft)
    struct draft_round {
        std::vector<std::vector<llama_token>> topk_tokens; // [step][topk] candidate tokens
        std::vector<std::vector<float>>       topk_probs;  // [step][topk] softmax probs
        std::vector<float>                    top1_conf;   // [step] top-1 confidence
        std::vector<llama_token>              chain;       // [root, top1_0, top1_1, ...]
        llama_token root = -1;
    };

    // candidate tree + verification buffers (port of generate_swift_buffers)
    struct tree_buffers {
        std::vector<std::vector<int32_t>> sorted_choices; // sorted candidate paths
        std::vector<int32_t>              tree_indices;   // [node] index into the flattened top-k list
        std::vector<int32_t>              position_ids;   // [node] depth (position within the tree)
        std::vector<int32_t>              depth_counts;   // [depth] number of candidates
        std::vector<std::vector<int32_t>> retrieve_indices; // [path][depth] -> tree node (batch row)
        std::vector<llama_token>          node_token;     // [node] token id
        std::vector<std::vector<llama_seq_id>> node_seqs; // [node] descendant-leaf seq ids
        std::vector<int32_t>              node_depth;     // [node] depth
        int n_nodes = 0;
        int n_leaves = 0;
    };

    // result of verifying the tree with the full model
    struct verify_result {
        std::vector<llama_token> argmax;          // [node] argmax token of the node logits
        std::vector<const float *> logits;        // [node] logits row (valid until the next decode)
        int best_candidate = 0;                   // best retrieve path
        int accept_length  = 0;                   // accepted tokens beyond the root
    };

    // contexts
    llama_context * ctx_main_  = nullptr; // single context: prefill, skip-graph drafting, tree verification, validation capture

    const llama_model * model_ = nullptr;
    llama_conflayer_params params_;
    std::vector<int32_t> skip_layers_;   // current draft skip set
    std::vector<int32_t> opt_set_;       // best skip set found by the optimization

    int n_vocab_ = 0;
    int n_layer_ = 0;
    int n_embd_  = 0; // hidden dimension (from the model)
    int32_t n_past_        = 0; // history length in ctx_main_
    int32_t new_token_num_ = 0; // new tokens generated so far
    bool     draft_mode_   = false; // ctx_main_ currently runs the skip graph (draft phase)

    // dynamic_optimization state (see conflayer/ConfLayers/evaluation/inference_conflayer.py)
    int32_t  opt_iter_      = 0;      // optimization iterations
    float    lam_           = 0.0f;   // adaptive window scale (statistics["lambda"])
    float    best_score_    = 0.0f;   // SD only: best acceptance score found
    float    best_acc_      = 0.0f;   // SD only: best mean accepted tokens
    float    best_agree_    = 0.0f;   // pure-skip: best full-model agreement seen
    float    agree_sum_     = 0.0f;   // pure-skip: agreement accumulator (for agree_rate)
    uint32_t agree_count_   = 0;      // pure-skip: agreement samples
    bool     optimization_on_ = false; // statistics["optimization"] is not None
    bool     do_opt_        = false;  // per-round trigger (local `optimization` in the Python)
    int32_t  skip_layer_num_ = 0;    // min skip set size to keep optimization active
    uint64_t n_opt_rounds_   = 0;    // dynamic_optimization rounds actually run
    std::vector<float> origin_score_; // SD only: rolling per-round acceptance score
    std::vector<uint32_t> origin_acc_; // SD only: rolling per-round accepted tokens

    // pure-skip skip-set statistics (see llama_conflayer_metrics)
    uint64_t skip_count_sum_ = 0;   // summed skip-set size per generated token
    uint32_t skip_count_min_ = 0;   // min skip-set size during generation
    uint32_t skip_count_max_ = 0;   // max skip-set size during generation
    std::vector<float> opt_entropy_thresholds_; // entropy threshold per layer-set update
    float    last_entropy_threshold_ = 0.0f;  // mean entropy threshold of the last search
    // per-opt-round per-layer confidence / entropy snapshots (see llama_conflayer_metrics)
    std::vector<std::vector<float>> opt_layer_confidence_;
    std::vector<std::vector<float>> opt_layer_entropy_;

    // scratch buffers (avoid repeated allocations)
    std::vector<float>       softmax_scratch_;
    std::vector<std::pair<float, llama_token>> topk_scratch_;
    std::vector<float>       capture_scratch_; // raw intermediate rows [window][n_embd]

    // context setup
    bool init_contexts();
    bool init_skip_layers();

    // toggle the layer_skip graph of ctx_main_ between the draft phase (on) and the
    // full-model phases (off). No-op when the mode is already active.
    bool set_draft_mode(bool on);

    // pure-skip resident mode: pin the first K active (non-skip) layers as the resident
    // set (keep_active_resident). Called after each layer-set optimization so the resident
    // mask tracks the current skip set. No-op when keep_active_resident <= 0 or verify.
    void set_resident_active();

    // low-level decode helpers
    bool decode_seq(llama_context * ctx, const std::vector<llama_token> & tokens,
                    llama_pos pos0, bool want_logits);

    // Phase 2: greedy skip-graph drafting
    bool draft(llama_token root, draft_round & out);
    void softmax_topk(const float * logits, int k,
                      std::vector<llama_token> & out_tok, std::vector<float> & out_prob);

    // Phase 3: tree construction + verification
    void get_choices(const draft_round & d, std::vector<std::vector<int32_t>> & choices) const;
    bool build_tree(const draft_round & d, const std::vector<std::vector<int32_t>> & choices, tree_buffers & out);
    bool verify_tree(const tree_buffers & tree, verify_result & out);

    // Phase 4: greedy acceptance + KV recycle
    void evaluate_posterior(const tree_buffers & tree, verify_result & vr,
                            llama_token & sample_token, std::vector<llama_token> & accepted);
    bool recycle(const tree_buffers & tree, int best_candidate, int accept_length);

    // misc
    llama_token argmax(const float * logits) const;
    int index_in_sorted(const std::vector<std::vector<int32_t>> & sorted,
                        const std::vector<int32_t> & path) const;

    // Phase 5: dynamic_optimization (layer-set search). Runs the validation forward,
    // computes the full-model agreement (pure-skip) and updates the skip set. Returns the
    // agreement (0 when not applicable, e.g. the first iteration).
    float dynamic_optimization(const std::vector<llama_token> & generated);
    float module_confidence(const char * name, int il, int n_tokens);
    std::vector<int32_t> find_skip_layers(const std::vector<float> & confidence);
};
