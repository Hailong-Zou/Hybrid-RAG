#pragma once
#include "llama-kv-cache.h"
#include "llama-kv-cluster.h"
#include "llama-kv-pager.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_cparams;

// Read-only view of cluster K/V data for attention.
// Lifetime: valid until next acquire/release cycle.
struct ClusterView {
    int32_t cluster_id;
    std::vector<uint32_t> positions;  // cell indices, sorted ascending
    const float * k_data = nullptr;   // [n_embd_k * n_tokens]
    const float * v_data = nullptr;   // [n_embd_v * n_tokens]
    uint32_t n_tokens = 0;
    uint32_t n_embd_k = 0;
    uint32_t n_embd_v = 0;
    bool owns_data = false;           // caller must free k_data/v_data when done
    bool ready = false;               // data is valid for use
};

class llama_kv_cache_cluster : public llama_kv_cache {
public:
    llama_kv_cache_cluster(const llama_model &, const llama_hparams &, ggml_type, ggml_type,
        bool v_trans, bool offload, bool unified, uint32_t kv_size, uint32_t n_seq_max,
        uint32_t n_pad, uint32_t n_swa, llama_swa_type, llama_memory_t,
        const layer_filter_cb &, const layer_reuse_cb &, const layer_share_cb &, const llama_cparams &);
    ~llama_kv_cache_cluster() override;

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override;
    void cluster_init();
    bool rebuild_after_restore(llama_seq_id dest_seq_id);
    void observe_decode_sequences(const std::vector<llama_seq_id> & seq_ids);
    void cluster_update(int32_t il, const float * k_cur, uint32_t cell_idx);
    void cluster_update_from_cache();
    void maybe_evict();
    void ensure_resident(const std::vector<uint32_t> & needed_positions);
    void record_metadata_consistency_errors(uint64_t n) { metadata_errors_ += n; }
    bool is_cell_evicted(uint32_t ci) const { return evicted_cells_.count(ci) > 0; }
    llama_kv_cluster_manager & get_cluster_mgr(int32_t il = 0) { return *cluster_mgrs_[il]; }
    bool is_prefill_done() const { return prefill_done_; }
    void set_prefill_done(bool d) { prefill_done_ = d; }
    uint32_t get_last_cell_idx() const;
    uint32_t get_last_cell_idx(llama_seq_id seq_id) const;
    bool cell_belongs_to_seq(uint32_t cell_idx, llama_seq_id seq_id) const;
    std::vector<uint32_t> get_unclustered_positions_before(
            llama_seq_id seq_id, llama_pos before_pos) const;
    std::vector<llama_seq_id> get_active_seq_ids() const;
    bool sparse_ready_for(const std::vector<llama_seq_id> & seq_ids) const;
    int  get_cluster_k() const { return cparams_.cluster_k; }
    void inc_step() { n_steps_since_update_++; n_total_steps_++; }
    uint32_t steps_since_update() const { return n_steps_since_update_; }
    void reset_step_counter() { n_steps_since_update_ = 0; }
    void print_stats() const;

    // Token buffer: buffer K/V data for non-resident clusters
    // Item 4: keyed by (layer_id, cluster_id) per PowerServe design
    void buffer_token(int32_t il, uint32_t cell_idx, int32_t cluster_id,
                      const float * k_vec, const float * v_vec,
                      size_t n_embd_k, size_t n_embd_v);
    void flush_token_buffer(int32_t il, int32_t cluster_id);
    void flush_all_buffers();
    size_t total_buffered_tokens() const;

    // Split handling: rewrite disk data for child clusters after split
    void handle_split(int32_t old_cluster_id, int32_t new1_id, int32_t new2_id);

    // Score-based token selection (TopK equivalent)
    // Item 7: accepts optional Q vector for real attention score
    std::vector<uint32_t> select_topk_tokens(const float * query, int k_tokens,
                                              const float * q_vector = nullptr,
                                              uint32_t current_cell_idx = UINT32_MAX);
    std::vector<uint32_t> select_positions_for_seq(llama_seq_id seq_id);

    // ClusterView: unified view of cluster data (memory, disk, or buffer)
    ClusterView acquire_cluster_view(int32_t cluster_id, int32_t il);
    void release_cluster_view(ClusterView & view);

    // Persist all clusters to disk (called after prefill)
    void persist_all_clusters();

private:
    std::vector<std::unique_ptr<llama_kv_cluster_manager>> cluster_mgrs_;
    std::unique_ptr<llama_kv_pager> pager_;
    const llama_cparams & cparams_;
    bool prefill_done_ = false;
    size_t prefill_tok_count_ = 0;
    uint32_t n_steps_since_update_ = 0, n_total_steps_ = 0;
    uint64_t cluster_init_count_ = 0;
    uint64_t eviction_count_ = 0;
    uint64_t page_in_count_ = 0;
    uint64_t handoff_rebuild_count_ = 0;
    uint64_t handoff_rebuild_failures_ = 0;
    uint64_t metadata_errors_ = 0;
    std::string disk_path_;
    std::unordered_set<uint32_t> evicted_cells_;
    mutable std::vector<std::future<llama_kv_pager::write_result>> pending_writes_;
    mutable std::vector<llama_kv_pager::read_handle> prefetch_handles_;
    struct cluster_disk_info { std::vector<uint32_t> positions; std::set<uint32_t> persisted_set; size_t persisted_token_count = 0; };
    std::unordered_map<int32_t, cluster_disk_info> disk_clusters_;
    std::unordered_map<int32_t, std::set<uint32_t>> pending_tokens_; // cell indices buffered per cluster
    struct buffered_token { uint32_t cell_idx; std::vector<float> k,v; };
    // Item 4: keyed by (layer_id, cluster_id) per PowerServe design
    using buffer_key_t = uint64_t;
    static inline buffer_key_t make_buffer_key(int32_t il, int32_t cid) {
        return ((uint64_t)(uint32_t)il << 32) | (uint64_t)(uint32_t)cid;
    }
    std::unordered_map<buffer_key_t, std::vector<buffered_token>> token_buffer_;
    std::unordered_set<llama_seq_id> cluster_cohort_;
    std::unordered_set<llama_seq_id> decode_seen_;
    mutable std::recursive_mutex token_buffer_mutex_;
    static constexpr size_t DEFAULT_TOKEN_BUFFER_THRESHOLD = 64;
    size_t token_buffer_threshold_ = DEFAULT_TOKEN_BUFFER_THRESHOLD;
    size_t get_n_embd_k(int32_t il) const; size_t get_n_embd_v(int32_t il) const;
    void collect_and_flush_token_buffers();
    void write_cluster_to_disk(int32_t cluster_id, const std::vector<uint32_t> & new_positions,
                               int64_t existing_offset, int64_t existing_capacity);
};

// Context with sparse K/V via pre-copy + ggml_set_input (no ggml_get_rows)
class llama_kv_cache_cluster_context : public llama_kv_cache_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;
    void init_from_sinfos(slot_info_vec_t, int32_t);
    llama_kv_cache_cluster_context(llama_memory_status);
    llama_kv_cache_cluster_context(llama_kv_cache_cluster *);
    llama_kv_cache_cluster_context(llama_kv_cache_cluster *, llama_context *, bool);
    llama_kv_cache_cluster_context(llama_kv_cache_cluster *, std::vector<llama_ubatch>);
    ~llama_kv_cache_cluster_context() override = default;

    // Sparse K/V via pre-copied input tensors
    ggml_tensor * get_k(ggml_context *, int32_t) const override;
    ggml_tensor * get_v(ggml_context *, int32_t) const override;
    uint32_t get_n_kv() const override;

    void set_input_k_idxs(ggml_tensor *, const llama_ubatch *) const override;
    void set_input_v_idxs(ggml_tensor *, const llama_ubatch *) const override;
    void set_input_kq_mask(ggml_tensor *, const llama_ubatch *, bool) const override;

    bool use_sparse_attn() const override;
    uint32_t get_n_kv_sparse() const override;

    bool is_prefill() const { return prefill_; }
    void set_prefill(bool p) { prefill_ = p; }

private:
    llama_kv_cache_cluster * kv_cluster_;
    bool prefill_ = true;
    mutable std::vector<uint32_t> cached_positions_;
    mutable std::unordered_map<int32_t, ggml_tensor *> sparse_k_tensors_;
    mutable std::unordered_map<int32_t, ggml_tensor *> sparse_v_tensors_;
    // Item 7: cached Q vector for real attention score selection
    mutable std::vector<float> cached_q_vector_;
    void build_cluster_indices(ggml_context *, int32_t) const;
    void ensure_positions() const;
    bool current_ubatch_is_prefill() const;
    std::vector<llama_seq_id> current_seq_ids() const;
};
