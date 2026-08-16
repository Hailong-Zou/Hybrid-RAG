#pragma once

#include "llama.h"

#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

// Represents a semantic cluster of token positions in the KV cache
struct llama_kv_cluster {
    int32_t cluster_id;
    std::vector<uint32_t> token_positions;  // cell indices in the physical KV cache
    std::vector<float> center_k;            // centroid Key vector [n_embd_k_gqa] (layer 0)
    float variance;                         // max intra-cluster variance
    bool is_resident;                       // true if K/V data is in GPU/CPU memory
    int64_t disk_offset;                    // byte offset in the disk file
    size_t  disk_size;                      // size in bytes on disk
    int     access_count = 0;               // number of times this cluster has been accessed
};

// Result of a cluster update, may include split notification
struct cluster_update_result {
    int32_t assigned_cluster = -1;
    bool split_occurred = false;
    int32_t old_cluster_id = -1;
    int32_t new1_cluster_id = -1;
    int32_t new2_cluster_id = -1;
};

// Manages clustering of KV cache tokens for sparse attention and offloading
class llama_kv_cluster_manager {
public:
    struct cache_stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t disk_reads = 0;
        uint64_t disk_writes = 0;
        uint64_t splits = 0;
    };

    llama_kv_cluster_manager(int max_resident, float var_threshold);

    // Initialize clusters after prefill using K vectors from layer 0
    // token_positions: cell indices of tokens in the KV cache
    // k_data: float array of K vectors for layer 0, shape [n_embd_k_gqa * n_tokens]
    void init(
        const std::vector<uint32_t> & token_positions,
        const float * k_data,
        size_t n_embd_k,
        size_t max_initial_clusters);

    // Assign a new decode token to the nearest cluster
    // cell_idx: cell index in KV cache
    // k_vector: K vector for this token [n_embd_k]
    // Returns update result including any split that occurred
    cluster_update_result update(uint32_t cell_idx, const float * k_vector, size_t n_embd_k);

    // Add a token position to an existing cluster without re-clustering
    // Used when merging buffered tokens on disk load
    void add_position(int32_t cluster_id, uint32_t cell_idx);

    // Score all resident clusters by dot product with query vector
    // Returns (cluster_id, score) pairs sorted by score descending
    std::vector<std::pair<int32_t, float>> score_clusters(const float * q, size_t n_embd_k) const;

    // Score individual tokens within a cluster by K-vector dot product with query
    // k_storage: the K storage tensor (layer-specific)
    // Returns (cell_idx, score) pairs sorted by score descending
    std::vector<std::pair<uint32_t, float>> score_tokens_in_cluster(
            int32_t cluster_id, const float * q, size_t n_embd_k,
            const float * k_storage_data, size_t k_storage_row_bytes,
            int k_storage_type) const;

    // Get all token positions from resident clusters (for attention)
    // Returns a flat list of cell indices
    std::vector<uint32_t> get_resident_positions() const;

    // Get total number of resident positions
    size_t get_n_resident_positions() const;

    // Get candidate clusters for eviction (coldest by LRU, non-resident prioritized)
    std::vector<int32_t> get_evict_candidates() const;

    // Mark a cluster as accessed (move to front of LRU)
    void mark_accessed(int32_t cluster_id);

    // Set cluster residency status and update disk location
    void set_resident(int32_t cluster_id, bool resident, int64_t disk_offset = 0, size_t disk_size = 0);

    // Get positions for a specific cluster
    const std::vector<uint32_t> & get_positions(int32_t cluster_id) const;

    // Check memory pressure
    bool needs_eviction() const { return n_resident_ > max_resident_; }

    int get_n_resident() const { return n_resident_; }
    int get_n_clusters() const { return (int)clusters_.size(); }
    int get_max_resident() const { return max_resident_; }

    // Disk offset/size accessors for eviction
    int64_t get_disk_offset(int32_t cluster_id) const;
    size_t get_disk_size(int32_t cluster_id) const;

    // Get cluster for a cell index
    int32_t get_cluster_id(uint32_t cell_idx) const;

    // Select top-K most recently used resident clusters' token positions
    // current_cell_idx: the current decode token's cell index (its cluster always included)
    // k: max number of clusters to select
    // side effect: marks selected clusters as accessed (moves to LRU front)
    std::vector<uint32_t> get_top_cluster_positions(uint32_t current_cell_idx, int k);

    // Check if a cluster is resident by cluster_id
    bool is_cluster_resident(int32_t cluster_id) const;

    // Parallel token scoring across multiple clusters (Item 3)
    std::vector<std::pair<uint32_t, float>> score_tokens_parallel(
            const std::vector<int32_t> & cluster_ids, const float * q, size_t n_embd_k,
            const float * k_storage_data, size_t k_storage_row_bytes,
            int k_storage_type, int k_tokens) const;

    // Cache statistics (Item 6)
    const cache_stats & get_stats() const { return stats_; }
    void reset_stats() { stats_ = cache_stats{}; }

    // Callback invoked when a cluster split occurs
    // Args: old_cluster_id, new1_cluster_id, new2_cluster_id
    std::function<void(int32_t, int32_t, int32_t)> on_split;

private:
    int max_resident_;      // cache capacity (max resident clusters)
    float var_threshold_;
    int n_resident_ = 0;
    int next_cluster_id_ = 0;

    std::vector<llama_kv_cluster> clusters_;
    std::unordered_map<uint32_t, int32_t> pos_to_cluster_; // cell_idx -> cluster_id
    std::list<int32_t> lru_list_; // front = MRU, back = LRU
    mutable cache_stats stats_;

    // Split a cluster into two when variance exceeds threshold
    // Outputs the new cluster IDs via out_new1_id / out_new2_id
    void split_cluster(size_t cluster_idx, const float * k_data, size_t n_embd_k,
                       int32_t * out_new1_id = nullptr, int32_t * out_new2_id = nullptr);

    // Compute dot product similarity between two vectors
    static float dot_product(const float * a, const float * b, size_t n);

    // Compute cosine similarity between two vectors in [-1, 1]
    static float cosine_similarity(const float * a, const float * b, size_t n);

    // Normalize a vector
    static void normalize(float * v, size_t n);

    // Get internal cluster index from cluster_id
    int32_t get_cluster_idx(int32_t cluster_id) const;
};
