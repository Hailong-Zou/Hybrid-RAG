// Standalone ConfLayers self-speculative decoder (see llama-conflayer.h).
//
// Port of conflayer/ConfLayers (Python) into llama.cpp as an independent module:
//   - greedy skip-graph drafting         (swift_draft)
//   - candidate bouquet tree + buffers   (get_choices_list, generate_swift_buffers)
//   - single-forward tree verification   (tree_decoding)
//   - greedy acceptance + KV recycle     (evaluate_posterior, update_inference_inputs)
//   - dynamic_optimization layer search  (Phase 5)
//
// The verification tree is decoded as a batch where each tree node is one token and
// each node carries the seq ids of its descendant leaves. The built-in KQ mask keeps
// only cells of the query's primary seq, so the tree mask is reproduced exactly: a
// node attends to the history plus its own ancestors (kv_unified keeps all tree seqs
// in a single stream, so ~250 seqs cost no extra KV memory).
//
// Single-context KV design (faithful to the Python shared past_key_values): ctx_main_
// owns the only KV cache. Prefill and tree verification run the full graph (skip off)
// and write full-model KV; drafting toggles the skip graph on and appends draft KV for
// new positions while reading the full-model history. Tree verification then overwrites
// the draft cells, and recycle() keeps only the accepted range. This guarantees the
// draft model sees exactly the same history the full model computed.

#include "llama-conflayer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numeric>
#include <utility>

llama_conflayer::llama_conflayer(const llama_model * model, const llama_conflayer_params & params)
    : model_(model), params_(params) {
    n_layer_ = llama_model_n_layer(model_);
    n_vocab_ = llama_vocab_n_tokens(llama_model_get_vocab(model_));
    n_embd_  = llama_model_n_embd(model_);

    init_skip_layers();

    if (!init_contexts()) {
        fprintf(stderr, "llama_conflayer: failed to initialize contexts\n");
    }

    // establish the initial resident set (only when layer_chunks > 0). draft_mode_ is off,
    // so this sets the resident mask without touching the graph's skip set.
    set_skip_layers(skip_layers_);
}

llama_conflayer::~llama_conflayer() {
    if (ctx_main_) llama_free(ctx_main_);
}

void llama_conflayer::set_skip_layers(const std::vector<int32_t> & skip_layers) {
    skip_layers_ = skip_layers;
    std::sort(skip_layers_.begin(), skip_layers_.end());

    // apply immediately if the context is currently in draft mode
    if (ctx_main_ && draft_mode_) {
        llama_set_layer_skip(ctx_main_, skip_layers_.data(), (int32_t) skip_layers_.size());
    }

    // resident = complement of the skip set: the draft's compute layers stay in RAM and are
    // shared with the verify phase; only the skipped layers stream from flash in chunks.
    if (ctx_main_ && params_.layer_chunks > 0) {
        std::vector<bool> skip_mask(n_layer_, false);
        for (int32_t il : skip_layers_) {
            if (il >= 0 && il < n_layer_) skip_mask[il] = true;
        }
        std::vector<int32_t> resident;
        resident.reserve(n_layer_);
        for (int il = 0; il < n_layer_; ++il) {
            if (!skip_mask[il]) resident.push_back(il);
        }
        llama_set_resident_layers(ctx_main_, resident.data(), (int32_t) resident.size());
    }
}

bool llama_conflayer::set_draft_mode(bool on) {
    if (draft_mode_ == on) {
        return true;
    }
    draft_mode_ = on;
    if (!on || skip_layers_.empty()) {
        llama_set_layer_skip(ctx_main_, nullptr, 0);
    } else {
        llama_set_layer_skip(ctx_main_, skip_layers_.data(), (int32_t) skip_layers_.size());
    }
    // cluster attention follows the phase: the draft phase keeps the full KV view (the
    // draft model must not use clustered attention), the verify phase uses true sparse
    // cluster attention when cluster_sparse is set, otherwise the full view again.
    if (params_.cluster_attn) {
        llama_set_cluster_full_attn(ctx_main_, on || !params_.cluster_sparse);
    }
    return true;
}

bool llama_conflayer::init_contexts() {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx         = params_.n_ctx;
    cparams.n_batch       = params_.n_batch;
    cparams.n_ubatch      = params_.n_ubatch;
    // the tree budget: the context's seq/output graph is sized to this (not LLAMA_MAX_SEQ),
    // so prefill only pays for the tree-sized reservation, not a 256-seq one.
    cparams.n_seq_max     = std::min(params_.max_seq, (uint32_t) LLAMA_CONFLAYER_MAX_SEQ);
    cparams.n_outputs_max = cparams.n_seq_max; // must be >= n_seq_max (output_reserve assert)
    cparams.kv_unified    = true;  // every tree seq shares one KV stream
    cparams.embeddings    = false;

    // flash-DRAM layered weight offloading. With the resident set (set_skip_layers), the
    // draft's compute layers stay in RAM and only the skip layers stream in chunks; without
    // it this falls back to skip-aware per-graph chunking.
    cparams.layer_chunks = params_.layer_chunks;

    // cluster-based KV cache on the shared context. Attention mode is phase-dependent:
    // the draft phase always keeps the full KV view (set_draft_mode re-applies it); the
    // verify phase uses true sparse cluster attention only when cluster_sparse is set.
    // Cluster bookkeeping init/update/persist/evict always runs.
    cparams.cluster_attn = params_.cluster_attn;

    ctx_main_ = llama_init_from_model(const_cast<llama_model *>(model_), cparams);
    if (ctx_main_ != nullptr && params_.cluster_attn) {
        llama_set_cluster_full_attn(ctx_main_, !params_.cluster_sparse);
    }
    return ctx_main_ != nullptr;
}

bool llama_conflayer::init_skip_layers() {
    // initial skip set: odd layers in [1, n_layer-1) (matches the Python _skip_layer_id_set)
    skip_layers_.clear();
    for (int il = 1; il < n_layer_ - 1; il += 2) {
        skip_layers_.push_back(il);
    }
    return true;
}

bool llama_conflayer::decode_seq(llama_context * ctx, const std::vector<llama_token> & tokens,
                                 llama_pos pos0, bool want_logits) {
    if (tokens.empty()) {
        return true;
    }
    const int32_t n = (int32_t) tokens.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = want_logits && i == n - 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

llama_token llama_conflayer::argmax(const float * logits) const {
    int32_t best = 0;
    float mx = logits[0];
    for (int i = 1; i < n_vocab_; ++i) {
        if (logits[i] > mx) {
            mx = logits[i];
            best = i;
        }
    }
    return (llama_token) best;
}

void llama_conflayer::softmax_topk(const float * logits, int k,
                                   std::vector<llama_token> & out_tok, std::vector<float> & out_prob) {
    // numerical stability: subtract the max before the exp
    float mx = logits[0];
    for (int i = 1; i < n_vocab_; ++i) {
        mx = std::max(mx, logits[i]);
    }
    softmax_scratch_.resize(n_vocab_);
    double sum = 0.0;
    for (int i = 0; i < n_vocab_; ++i) {
        softmax_scratch_[i] = (float) std::exp((double) logits[i] - mx);
        sum += softmax_scratch_[i];
    }
    topk_scratch_.resize(n_vocab_);
    for (int i = 0; i < n_vocab_; ++i) {
        topk_scratch_[i] = { (float) (softmax_scratch_[i] / sum), (llama_token) i };
    }
    const int kk = std::min(k, n_vocab_);
    std::partial_sort(topk_scratch_.begin(), topk_scratch_.begin() + kk, topk_scratch_.end(),
                      [](const std::pair<float, llama_token> & a, const std::pair<float, llama_token> & b) {
                          return a.first > b.first;
                      });
    out_tok.clear();
    out_prob.clear();
    for (int i = 0; i < kk; ++i) {
        out_tok.push_back(topk_scratch_[i].second);
        out_prob.push_back(topk_scratch_[i].first);
    }
}

// Phase 2: greedy skip-graph drafting (port of swift_draft)
bool llama_conflayer::draft(llama_token root, draft_round & out) {
    out.root = root;
    out.topk_tokens.clear();
    out.topk_probs.clear();
    out.top1_conf.clear();
    out.chain.clear();

    // run the skip graph (draft model) against the full-model history KV
    if (!set_draft_mode(true)) {
        return false;
    }

    // decode the root at n_past_ and collect the top-k of its logits
    if (!decode_seq(ctx_main_, { root }, n_past_, true)) {
        return false;
    }
    out.chain.push_back(root);

    const int k = (int) params_.topk;
    // cap the steps so the candidate tree stays within the seq/output budget:
    // each step adds at most `topk` candidates, and the tree has 1 + sum(cands) nodes,
    // so at most (max_seq - 1) / topk steps keep the whole tree under max_seq rows.
    const int max_steps = std::min((int) params_.max_step_draft, std::max(0, ((int) params_.max_seq - 1) / k));
    for (int step = 0; step < max_steps; ++step) {
        const float * logits = llama_get_logits_ith(ctx_main_, -1);

        std::vector<llama_token> toks;
        std::vector<float> probs;
        softmax_topk(logits, k, toks, probs);
        out.topk_tokens.push_back(std::move(toks));
        out.topk_probs.push_back(std::move(probs));

        const float conf = out.topk_probs.back()[0];
        out.top1_conf.push_back(conf);

        // stop when the greedy continuation is not confident enough, or near the end
        if (conf < params_.stop_threshold || new_token_num_ + step + 2 >= (int32_t) params_.max_new_tokens) {
            break;
        }

        // decode the greedy continuation at the next position
        const llama_token next = out.topk_tokens.back()[0];
        if (!decode_seq(ctx_main_, { next }, n_past_ + step + 1, true)) {
            return false;
        }
        out.chain.push_back(next);
    }
    return true;
}

// Phase 3: candidate tree construction (port of get_choices_list + generate_swift_buffers)
void llama_conflayer::get_choices(const draft_round & d, std::vector<std::vector<int32_t>> & choices) const {
    choices.clear();
    // greedy path: candidate counts {1, 2, 3, 5}. Narrower than the Python original's
    // {3, 5, 8, 10}: on CPU the verify forward cost scales with the tree width, and the
    // extra candidates only pay off when the draft is wrong, which is exactly where the
    // greedy acceptance stops anyway. Candidate 0 is always the draft's greedy token, so
    // the greedy path stays intact. (The non-greedy path uses {1, 3, 5, 10}.)
    const int32_t candidate_set[4] = { 1, 2, 3, 5 };
    for (size_t idx = 0; idx < d.top1_conf.size(); ++idx) {
        const float item = d.top1_conf[idx];
        const int32_t cand = item > 0.95f ? candidate_set[0]
                           : item > 0.8f  ? candidate_set[1]
                           : item > 0.5f  ? candidate_set[2]
                                          : candidate_set[3];
        for (int32_t i = 0; i < cand; ++i) {
            std::vector<int32_t> path(idx, 0);
            path.push_back(i);
            choices.push_back(std::move(path));
        }
    }
}

int llama_conflayer::index_in_sorted(const std::vector<std::vector<int32_t>> & sorted,
                                     const std::vector<int32_t> & path) const {
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i] == path) {
            return (int) i;
        }
    }
    return -1;
}

bool llama_conflayer::build_tree(const draft_round & d,
                                 const std::vector<std::vector<int32_t>> & choices, tree_buffers & out) {
    out.sorted_choices = choices;
    std::sort(out.sorted_choices.begin(), out.sorted_choices.end(),
              [](const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
                  return a.size() != b.size() ? a.size() < b.size() : a < b;
              });

    const int n_steps   = (int) d.top1_conf.size();
    const int TOPK      = (int) params_.topk;
    const int swift_len = (int) out.sorted_choices.size() + 1; // +1 for the root
    out.n_nodes = swift_len;

    if (swift_len > (int) params_.max_seq) {
        return false; // tree larger than the seq/output budget (draft step cap should prevent this)
    }

    // depth counts per step
    out.depth_counts.assign(n_steps, 0);
    for (const auto & path : out.sorted_choices) {
        out.depth_counts[(int) path.size() - 1] += 1;
    }

    // tree indices + node depths (port of generate_swift_buffers)
    out.tree_indices.assign(swift_len, 0);
    out.node_depth.assign(swift_len, 0);
    int start = 0;
    for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < out.depth_counts[i]; ++j) {
            const auto & cur = out.sorted_choices[start + j];
            out.tree_indices[start + j + 1] = cur.back() + TOPK * i + 1;
            out.node_depth[start + j + 1]   = i + 1;
        }
        start += out.depth_counts[i];
    }
    out.position_ids = out.node_depth;

    // node tokens: candidate list = [root, step0 topk, step1 topk, ...]
    //   tree_indices[node] is an index into that list (1 + i*TOPK + c)
    out.node_token.assign(swift_len, 0);
    out.node_token[0] = d.root;
    for (int node = 1; node < swift_len; ++node) {
        const int ci     = out.tree_indices[node];
        const int step   = (ci - 1) / TOPK;
        const int choice = (ci - 1) % TOPK;
        out.node_token[node] = d.topk_tokens[step][choice];
    }

    // parent pointers (bouquet: each node's parent is the backbone node at the previous depth)
    std::map<std::vector<int32_t>, int> path_to_node;
    path_to_node[std::vector<int32_t>()] = 0; // root = empty path
    for (int j = 0; j < (int) out.sorted_choices.size(); ++j) {
        path_to_node[out.sorted_choices[j]] = j + 1;
    }
    std::vector<int> parent(swift_len, -1);
    for (int j = 0; j < (int) out.sorted_choices.size(); ++j) {
        const auto & p = out.sorted_choices[j];
        std::vector<int32_t> pp(p.begin(), p.end() - 1);
        parent[j + 1] = path_to_node[pp];
    }
    std::vector<std::vector<int>> children(swift_len);
    std::vector<bool> has_child(swift_len, false);
    for (int j = 1; j < swift_len; ++j) {
        children[parent[j]].push_back(j);
        has_child[parent[j]] = true;
    }

    // one seq id per leaf (the seq id encodes the branch)
    out.n_leaves = 0;
    out.node_seqs.assign(swift_len, {});
    int sid = 1;
    for (int node = 0; node < swift_len; ++node) {
        if (!has_child[node]) {
            out.node_seqs[node].push_back((llama_seq_id) sid++);
            out.n_leaves += 1;
        }
    }

    // propagate leaf seqs up the tree (deepest first); the primary seq (front) is the
    // smallest descendant leaf, which the KQ mask uses as the tree-branch discriminator
    std::vector<int> order(swift_len);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return out.node_depth[a] > out.node_depth[b]; });
    for (int node : order) {
        if (children[node].empty()) {
            continue;
        }
        for (int ch : children[node]) {
            for (llama_seq_id s : out.node_seqs[ch]) {
                if (std::find(out.node_seqs[node].begin(), out.node_seqs[node].end(), s) == out.node_seqs[node].end()) {
                    out.node_seqs[node].push_back(s);
                }
            }
        }
        std::sort(out.node_seqs[node].begin(), out.node_seqs[node].end());
    }

    // retrieve indices: [path][depth] -> batch row (port of generate_swift_buffers).
    // One root-to-node path per choice, deepest first, prefixes not repeated.
    std::vector<std::vector<int32_t>> ret_nest;
    std::vector<std::vector<int32_t>> covered;
    for (int i = (int) out.sorted_choices.size() - 1; i >= 0; --i) {
        const auto & cur = out.sorted_choices[i];
        if (std::find(covered.begin(), covered.end(), cur) != covered.end()) {
            continue;
        }
        std::vector<int32_t> rpath;
        for (int c = 0; c < (int) cur.size(); ++c) {
            std::vector<int32_t> prefix(cur.begin(), cur.begin() + c + 1);
            rpath.push_back(index_in_sorted(out.sorted_choices, prefix) + 1);
            covered.push_back(std::move(prefix));
        }
        ret_nest.push_back(std::move(rpath));
    }
    int max_len = 0;
    for (const auto & r : ret_nest) {
        max_len = std::max(max_len, (int) r.size());
    }
    for (auto & r : ret_nest) {
        r.resize(max_len, -1);   // pad
    }
    for (auto & r : ret_nest) {
        r.insert(r.begin(), 0);  // root column
    }

    // sort by (len, lexicographic) with -1 treated as a large value (port of custom_sort)
    const int maxitem = swift_len + 4;
    std::stable_sort(ret_nest.begin(), ret_nest.end(),
                     [maxitem](const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
                         for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                             const int32_t xa = a[i] >= 0 ? a[i] : maxitem;
                             const int32_t xb = b[i] >= 0 ? b[i] : maxitem;
                             if (xa != xb) {
                                 return xa < xb;
                             }
                         }
                         return false;
                     });
    out.retrieve_indices = std::move(ret_nest);
    return true;
}

// Phase 3: verify the tree with the full model.
// Two modes:
//   - cluster_sparse off: SINGLE batch, all nodes at once (one forward). The history
//     cells (seq 0) are copied into every tree seq, and each tree cell carries only its
//     descendant leaves' seqs, so a node attends to history + its own ancestors through
//     the built-in causal in-batch mask (position = depth, ancestors ordered first).
//   - cluster_sparse on: one depth at a time, so every batch row is a lone token on its
//     seqs (decode semantics) and the sparse cluster attention stays active; a single
//     all-nodes batch would read as prefill and silently fall back to full attention.
bool llama_conflayer::verify_tree(const tree_buffers & tree, verify_result & out) {
    // full model: run the complete graph (skip off) to verify the tree
    if (!set_draft_mode(false)) {
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx_main_);

    for (int s = 1; s <= tree.n_leaves; ++s) {
        llama_memory_seq_cp(mem, 0, (llama_seq_id) s, 0, n_past_);
    }

    out.argmax.assign(tree.n_nodes, -1);
    out.logits.resize(tree.n_nodes);

    int max_depth = 0;
    for (int i = 0; i < tree.n_nodes; ++i) {
        max_depth = std::max(max_depth, tree.node_depth[i]);
    }

    if (!params_.cluster_sparse) {
        // single batch: all nodes, ancestors (lower depth) first so the in-batch causal
        // mask makes every node attend to history + its ancestors in one forward
        std::vector<int> order(tree.n_nodes);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return tree.node_depth[a] != tree.node_depth[b]
                                          ? tree.node_depth[a] < tree.node_depth[b] : a < b; });

        int max_seqs = 0;
        for (int i = 0; i < tree.n_nodes; ++i) {
            max_seqs = std::max(max_seqs, (int) tree.node_seqs[i].size());
        }

        llama_batch batch = llama_batch_init(tree.n_nodes, 0, max_seqs + 1);
        batch.n_tokens = tree.n_nodes;
        for (int k = 0; k < tree.n_nodes; ++k) {
            const int i = order[k];
            batch.token[k]    = tree.node_token[i];
            batch.pos[k]      = n_past_ + tree.node_depth[i];
            batch.n_seq_id[k] = (int32_t) tree.node_seqs[i].size();
            for (size_t s = 0; s < tree.node_seqs[i].size(); ++s) {
                batch.seq_id[k][s] = tree.node_seqs[i][s];
            }
            batch.logits[k] = true;
        }
        const int rc = llama_decode(ctx_main_, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            return false;
        }
        for (int k = 0; k < tree.n_nodes; ++k) {
            const int i = order[k];
            out.logits[i] = llama_get_logits_ith(ctx_main_, k);
            out.argmax[i] = argmax(out.logits[i]);
        }
        return true;
    }

    // cluster_sparse: verify one depth at a time (decode semantics per layer)
    for (int depth = 0; depth <= max_depth; ++depth) {
        std::vector<int> layer;
        for (int i = 0; i < tree.n_nodes; ++i) {
            if (tree.node_depth[i] == depth) {
                layer.push_back(i);
            }
        }
        if (layer.empty()) {
            continue;
        }

        const int n = (int) layer.size();
        int max_seqs = 0;
        for (int i : layer) {
            max_seqs = std::max(max_seqs, (int) tree.node_seqs[i].size());
        }

        llama_batch batch = llama_batch_init(n, 0, max_seqs + 1);
        batch.n_tokens = n;
        for (int k = 0; k < n; ++k) {
            const int i = layer[k];
            batch.token[k]    = tree.node_token[i];
            batch.pos[k]      = n_past_ + tree.node_depth[i];
            batch.n_seq_id[k] = (int32_t) tree.node_seqs[i].size();
            for (size_t s = 0; s < tree.node_seqs[i].size(); ++s) {
                batch.seq_id[k][s] = tree.node_seqs[i][s];
            }
            batch.logits[k] = true;
        }
        const int rc = llama_decode(ctx_main_, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            return false;
        }

        // argmax of this layer's node logits (greedy acceptance only needs the max).
        // Note: out.logits[i] pointers are only valid until the next layer's decode.
        for (int k = 0; k < n; ++k) {
            const int i = layer[k];
            out.logits[i] = llama_get_logits_ith(ctx_main_, k);
            out.argmax[i] = argmax(out.logits[i]);
        }
    }
    return true;
}

// Phase 4: greedy acceptance (port of evaluate_posterior, greedy branch)
void llama_conflayer::evaluate_posterior(const tree_buffers & tree, verify_result & vr,
                                         llama_token & sample_token, std::vector<llama_token> & accepted) {
    const int n_paths = (int) tree.retrieve_indices.size();

    // per-path leading-match count (cumprod sum in the Python port)
    int best = 0;
    int best_len = 0;
    for (int r = 0; r < n_paths; ++r) {
        const auto & path = tree.retrieve_indices[r];
        int acc = 0;
        for (int d = 0; d + 1 < (int) path.size(); ++d) {
            const int node      = path[d];
            const int next_node = path[d + 1];
            if (node < 0 || next_node < 0) {
                break;
            }
            if (tree.node_token[next_node] != vr.argmax[node]) {
                break; // cumprod stops at the first mismatch
            }
            acc += 1;
        }
        if (acc > best_len) {
            best_len = acc;
            best = r;
        }
    }
    vr.best_candidate = best;
    vr.accept_length  = best_len;

    // accepted tokens along the best path (root + accepted candidates)
    accepted.clear();
    for (int d = 0; d <= best_len; ++d) {
        const int node = tree.retrieve_indices[best][d];
        accepted.push_back(tree.node_token[node]);
    }

    // next round root: the full model's greedy continuation of the deepest accepted node
    const int node = tree.retrieve_indices[best][best_len];
    sample_token = vr.argmax[node];
}

// Phase 4: KV recycle (port of update_inference_inputs). The accepted path cells all
// carry the best path's leaf seq; with kv_unified the copy is pure cell bookkeeping.
// The draft cells written at positions beyond the accepted length are purged so the
// next round's draft starts from a clean, full-model-only history.
bool llama_conflayer::recycle(const tree_buffers & tree, int best_candidate, int accept_length) {
    const auto & path = tree.retrieve_indices[best_candidate];
    int leaf_node = -1;
    for (int d = (int) path.size() - 1; d >= 0; --d) {
        if (path[d] >= 0) {
            leaf_node = path[d];
            break;
        }
    }
    if (leaf_node < 0) {
        return false;
    }
    const llama_seq_id s_acc = tree.node_seqs[leaf_node].front();

    llama_memory_t mem = llama_get_memory(ctx_main_);
    llama_memory_seq_cp(mem, s_acc, 0, n_past_, n_past_ + accept_length + 1);
    llama_memory_seq_keep(mem, 0);
    n_past_ += accept_length + 1;
    // purge stale draft cells beyond the accepted length (they hold skip-model KV)
    llama_memory_seq_rm(mem, 0, n_past_, -1);
    return true;
}

static float vec_mean(const std::vector<float> & v) {
    if (v.empty()) {
        return 0.0f;
    }
    double s = 0.0;
    for (float x : v) {
        s += x;
    }
    return (float) (s / v.size());
}

static float vec_mean(const std::vector<uint32_t> & v) {
    if (v.empty()) {
        return 0.0f;
    }
    double s = 0.0;
    for (uint32_t x : v) {
        s += x;
    }
    return (float) (s / v.size());
}

// np.gradient 1D (central differences, forward/backward at the edges)
static std::vector<float> gradient_1d(const std::vector<float> & y) {
    std::vector<float> g(y.size(), 0.0f);
    if (y.empty()) {
        return g;
    }
    if (y.size() == 1) {
        return g;
    }
    g[0] = y[1] - y[0];
    g.back() = y.back() - y[y.size() - 2];
    for (size_t i = 1; i + 1 < y.size(); ++i) {
        g[i] = 0.5f * (y[i + 1] - y[i - 1]);
    }
    return g;
}

// Phase 5: raw pre-residual intermediate entropy confidence (compute_module_entropy port).
float llama_conflayer::module_confidence(const char * name, int il, int n_tokens) {
    capture_scratch_.resize((size_t) n_tokens * n_embd_);
    const int32_t got = llama_get_intermediate_output(ctx_main_, name, il, capture_scratch_.data(), n_tokens);
    if (got < n_tokens) {
        return 0.0f;
    }

    // logits = clamp(x, ln(float32 tiny)=-87.3, 88.7); softmax over the hidden dim
    const float log_min = -87.3f;
    const float log_max =  88.7f;
    double total_entropy = 0.0;
    for (int t = 0; t < n_tokens; ++t) {
        const float * row = capture_scratch_.data() + (size_t) t * n_embd_;
        float mx = -1e30f;
        for (int j = 0; j < n_embd_; ++j) {
            mx = std::max(mx, std::clamp(row[j], log_min, log_max));
        }
        double sum = 0.0;
        for (int j = 0; j < n_embd_; ++j) {
            sum += std::exp((double) std::clamp(row[j], log_min, log_max) - mx);
        }
        double entropy = 0.0;
        for (int j = 0; j < n_embd_; ++j) {
            const double p = std::exp((double) std::clamp(row[j], log_min, log_max) - mx) / sum;
            entropy += p * std::log(p + 1e-12);
        }
        total_entropy += -entropy;
    }
    const float mean_entropy = (float) (total_entropy / n_tokens);
    const float max_entropy  = std::log((float) n_embd_);
    return 1.0f - mean_entropy / max_entropy;
}

// Phase 5: layer selection from the per-layer confidence (find_skip_layers port).
std::vector<int32_t> llama_conflayer::find_skip_layers(const std::vector<float> & confidence) {
    // only layers 1..n_layer-2 participate (the first and last layer are always kept)
    std::vector<int32_t> layer_ids;
    std::vector<float>   vals;
    for (int il = 1; il < n_layer_ - 1; ++il) {
        layer_ids.push_back(il);
        vals.push_back(confidence[il]);
    }
    const int n = (int) vals.size();
    if (n == 0) {
        return {};
    }

    // z-score across layers
    std::vector<float> conf(n);
    {
        double mu = 0.0;
        for (float c : vals) {
            mu += c;
        }
        mu /= n;
        double var = 0.0;
        for (float c : vals) {
            var += (c - mu) * (c - mu);
        }
        const float sd = (float) std::sqrt(var / n);
        for (int i = 0; i < n; ++i) {
            conf[i] = (vals[i] - (float) mu) / (sd + 1e-6f);
        }
    }

    // second-order gradient
    std::vector<float> g1 = gradient_1d(conf);
    std::vector<float> grad = gradient_1d(g1);

    // count_consecutive_equal: first index where grad differs from grad[0], else len
    int count = n;
    for (int i = 0; i < n; ++i) {
        if (grad[i] != grad[0]) {
            count = i;
            break;
        }
    }

    // normalize grad to [0, 1], then adjust_to_mean_half
    {
        float mn = grad[0], mx = grad[0];
        for (float g : grad) {
            mn = std::min(mn, g);
            mx = std::max(mx, g);
        }
        for (float & g : grad) {
            g = (g - mn) / (mx - mn + 1e-8f);
        }
        // adjust_to_mean_half
        double mu = 0.0;
        for (float g : grad) {
            mu += g;
        }
        mu /= n;
        float gmin = grad[0], gmax = grad[0];
        for (float g : grad) {
            gmin = std::min(gmin, g);
            gmax = std::max(gmax, g);
        }
        if (std::fabs(gmax - gmin) < 1e-9f) {
            std::fill(grad.begin(), grad.end(), 0.5f);
        } else {
            const float a1 = 0.5f / (gmax - (float) mu);
            const float a2 = -0.5f / (gmin - (float) mu);
            const float a  = std::min(a1, a2);
            const float b  = 0.5f - a * (float) mu;
            for (float & g : grad) {
                g = std::clamp(a * g + b, 0.0f, 1.0f);
            }
        }
    }

    const int num_layers = n_layer_;
    float window_alpha = 5.0f * count;
    if (params_.model_id.find("qwen") != std::string::npos) {
        window_alpha = std::min((float) (num_layers / 3), window_alpha);
    }

    std::vector<int32_t> skip;
    for (int i = 0; i < n; ++i) {
        const int window_scale = (int) (window_alpha * grad[i] * ((i + 1) / (float) num_layers));
        const int window_size  = 2 + window_scale;
        const int half         = window_size / 2;
        const int start        = std::max(0, i - half);
        const int end          = std::min(n, i + half + 1);

        // local window excluding layer i
        double lmu = 0.0;
        int cnt = 0;
        for (int j = start; j < end; ++j) {
            if (j == i) {
                continue;
            }
            lmu += conf[j];
            ++cnt;
        }
        if (cnt == 0) {
            if (conf[i] < 0.0f) {
                skip.push_back(layer_ids[i]);
            }
            continue;
        }
        lmu /= cnt;
        double lvar = 0.0;
        for (int j = start; j < end; ++j) {
            if (j == i) {
                continue;
            }
            lvar += (conf[j] - lmu) * (conf[j] - lmu);
        }
        const float lstd = (float) std::sqrt(lvar / cnt);
        const float threshold = (float) lmu - lam_ * lstd;
        if (conf[i] < threshold) {
            skip.push_back(layer_ids[i]);
        }
    }

    // lambda feedback loop
    const float perc = (float) skip.size() / (float) (num_layers - 2);
    if (perc > params_.skip_ratio + 0.25f) {
        lam_ += 0.05f;
    } else if (perc < params_.skip_ratio) {
        lam_ -= 0.05f;
    }
    return skip;
}

// Phase 5: validation forward over the last `context_window` tokens, entropy confidence,
// and layer-set update. Port of dynamic_optimization in conflayer utils.
void llama_conflayer::dynamic_optimization(const std::vector<llama_token> & generated) {
    const int32_t window = (int32_t) std::min<uint32_t>(params_.context_window, (uint32_t) generated.size());
    if (window <= 0) {
        return;
    }
    n_opt_rounds_++;

    // 1. validation forward on the last window tokens through the full model (capture on).
    //    Re-decoding at already-cached positions recomputes identical cells, so the KV
    //    cache of ctx_main_ is left unchanged. The full graph is required (skip off).
    if (!set_draft_mode(false)) {
        return;
    }
    llama_set_intermediate_capture(ctx_main_, true);
    const llama_pos pos0 = n_past_ - window;
    llama_batch batch = llama_batch_init(window, 0, 1);
    batch.n_tokens = window;
    for (int32_t i = 0; i < window; ++i) {
        batch.token[i]     = generated[generated.size() - (size_t) window + i];
        batch.pos[i]       = pos0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = false;
    }
    const int rc = llama_decode(ctx_main_, batch);
    llama_batch_free(batch);
    llama_set_intermediate_capture(ctx_main_, false);
    if (rc != 0) {
        return;
    }

    // 2. per-layer total confidence for layers 1..n_layer-2 (keep the first and last layer)
    std::vector<float> confidence(n_layer_, 0.0f);
    for (int il = 1; il < n_layer_ - 1; ++il) {
        confidence[il] = module_confidence("attn_out", il, window) + module_confidence("ffn_out", il, window);
    }

    // 3. find the new skip set and apply it. No KV re-decode is needed: the shared cache
    //    already holds full-model KV for the whole history, and the next draft phase
    //    reads it through the new skip graph.
    const std::vector<int32_t> new_skip = find_skip_layers(confidence);
    set_skip_layers(new_skip);

    opt_iter_++;
    if (opt_iter_ >= (int32_t) params_.max_opt_iter) {
        set_skip_layers(opt_set_); // restore the best skip set found so far
        optimization_on_ = false;
    }
}

bool llama_conflayer::generate(const std::vector<llama_token> & prompt,
                               std::vector<llama_token>       & result,
                               llama_conflayer_metrics        * metrics) {
    if (!ctx_main_) {
        return false;
    }
    result.clear();
    n_past_ = 0;
    new_token_num_ = 0;

    // dynamic_optimization state (statistics dict from inference_conflayer.py)
    opt_iter_      = 0;
    lam_           = 0.0f;
    best_score_    = 0.0f;
    best_acc_      = 0.0f;
    optimization_on_ = params_.optimize;
    do_opt_        = true; // the Python local `optimization` starts true
    skip_layer_num_ = (int32_t) ((n_layer_ - 2) * params_.skip_ratio) + 1;
    n_opt_rounds_   = 0;
    opt_set_.clear();
    origin_score_.clear();
    origin_acc_.clear();

    if (prompt.empty()) {
        return false;
    }
    if (prompt.size() + params_.max_new_tokens + params_.max_step_draft + 2 > params_.n_ctx) {
        fprintf(stderr, "llama_conflayer: prompt + max_new_tokens exceeds n_ctx (%u)\n", params_.n_ctx);
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_);
    const llama_token eos = llama_vocab_eos(vocab);

    // prefill the single shared context through the full model (skip off)
    const int64_t t_prefill_0 = ggml_time_us();
    const int chunk = (int) params_.n_batch;
    for (size_t i0 = 0; i0 < prompt.size(); i0 += (size_t) chunk) {
        const size_t n = std::min((size_t) chunk, prompt.size() - i0);
        std::vector<llama_token> toks(prompt.begin() + i0, prompt.begin() + i0 + n);
        const bool last = (i0 + n == prompt.size());
        if (!decode_seq(ctx_main_, toks, (llama_pos) i0, last)) {
            return false;
        }
    }
    n_past_ = (int32_t) prompt.size();

    // initial root: the full model's greedy continuation of the last prompt token
    const float * lp = llama_get_logits_ith(ctx_main_, -1);
    llama_token sample_token = argmax(lp);

    llama_conflayer_metrics m;
    m.prefill_us = (uint64_t) (ggml_time_us() - t_prefill_0);

    const int64_t t_decode_0 = ggml_time_us();
    while (true) {
        // Phase 2: greedy draft along the skip graph
        draft_round d;
        const int64_t t_draft_0 = ggml_time_us();
        if (!draft(sample_token, d)) {
            return false;
        }
        m.draft_us += (uint64_t) (ggml_time_us() - t_draft_0);
        m.n_draft_tokens += d.top1_conf.size();
        m.n_rounds += 1;

        // Phase 3: candidate tree + single-forward verification
        std::vector<std::vector<int32_t>> choices;
        get_choices(d, choices);
        tree_buffers tree;
        if (!build_tree(d, choices, tree)) {
            return false;
        }
        m.n_tree_nodes += tree.n_nodes;
        verify_result vr;
        const int64_t t_verify_0 = ggml_time_us();
        if (!verify_tree(tree, vr)) {
            return false;
        }
        m.verify_us += (uint64_t) (ggml_time_us() - t_verify_0);

        // Phase 4: greedy acceptance + KV recycle
        llama_token next_root;
        std::vector<llama_token> accepted;
        evaluate_posterior(tree, vr, next_root, accepted);

        m.n_accepted += accepted.size() - 1;
        m.accept_lengths.push_back((uint32_t) accepted.size());

        if (!recycle(tree, vr.best_candidate, vr.accept_length)) {
            return false;
        }

        result.insert(result.end(), accepted.begin(), accepted.end());
        new_token_num_ += (int32_t) accepted.size();

        // Phase 5: layer-set optimization (port of the statistics loop in swift_forward)
        if (optimization_on_ && new_token_num_ > (int32_t) (params_.context_window + 1)) {
            origin_score_.push_back((float) (accepted.size() - 1) / (float) std::max<size_t>(1, d.top1_conf.size()));
            origin_acc_.push_back((uint32_t) (accepted.size() - 1));

            if (opt_iter_ > 0 && origin_score_.size() == params_.search_interval / 2) {
                if ((int) skip_layers_.size() >= skip_layer_num_) {
                    const float acc = vec_mean(origin_acc_);
                    if (acc < 1.0f) {
                        origin_score_.clear();
                        origin_acc_.clear();
                        do_opt_ = true;
                    }
                }
            }

            if (opt_iter_ > 0 && origin_score_.size() > params_.search_interval) {
                if ((int) skip_layers_.size() >= skip_layer_num_) {
                    const float ret = vec_mean(origin_score_);
                    const float acc = vec_mean(origin_acc_);
                    if (acc > 4.0f && ret > params_.max_score) {
                        optimization_on_ = false; // statistics["optimization"] = None
                        do_opt_ = false;
                    } else if (acc > best_acc_) {
                        best_acc_   = acc;
                        best_score_ = ret;
                        opt_set_    = skip_layers_;
                        do_opt_     = true;
                    } else {
                        do_opt_ = true;
                    }
                } else {
                    do_opt_ = true;
                }
                origin_score_.clear();
                origin_acc_.clear();
            }

            if (do_opt_ || (int) skip_layers_.size() < skip_layer_num_) {
                const int64_t t_opt_0 = ggml_time_us();
                dynamic_optimization(result);
                m.opt_us += (uint64_t) (ggml_time_us() - t_opt_0);
                origin_score_.clear();
                origin_acc_.clear();
                do_opt_ = false;
            }
        }

        // stop conditions
        bool stop = false;
        for (llama_token t : accepted) {
            if (t == eos) {
                stop = true;
                break;
            }
        }
        if (stop || new_token_num_ > (int32_t) params_.max_new_tokens) {
            break;
        }

        sample_token = next_root;
    }

    m.accept_rate = m.n_draft_tokens ? (float) m.n_accepted / (float) m.n_draft_tokens : 0.0f;
    m.n_opt_rounds = n_opt_rounds_;
    m.decode_us = (uint64_t) (ggml_time_us() - t_decode_0);
    if (metrics) {
        *metrics = m;
    }
    return true;
}
