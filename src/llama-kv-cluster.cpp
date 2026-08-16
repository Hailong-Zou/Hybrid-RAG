#include "llama-kv-cluster.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>

float llama_kv_cluster_manager::dot_product(const float * a, const float * b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float llama_kv_cluster_manager::cosine_similarity(const float * a, const float * b, size_t n) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    return denom > 1e-8f ? dot / denom : 0.0f;
}

void llama_kv_cluster_manager::normalize(float * v, size_t n) {
    float norm = 0.0f;
    for (size_t i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
        for (size_t i = 0; i < n; i++) {
            v[i] /= norm;
        }
    }
}

llama_kv_cluster_manager::llama_kv_cluster_manager(int max_resident, float var_threshold)
    : max_resident_(max_resident), var_threshold_(var_threshold) {}

int32_t llama_kv_cluster_manager::get_cluster_idx(int32_t cluster_id) const {
    for (size_t i = 0; i < clusters_.size(); i++) {
        if (clusters_[i].cluster_id == cluster_id) {
            return (int32_t)i;
        }
    }
    return -1;
}

void llama_kv_cluster_manager::init(
        const std::vector<uint32_t> & token_positions,
        const float * k_data,
        size_t n_embd_k,
        size_t max_initial_clusters) {

    clusters_.clear();
    pos_to_cluster_.clear();
    lru_list_.clear();
    n_resident_ = 0;
    next_cluster_id_ = 0;

    size_t n_tokens = token_positions.size();
    if (n_tokens == 0) return;

    size_t n_clusters = std::min(max_initial_clusters, std::max((size_t)1, n_tokens / 16));

    // Initialize cluster centers by sampling evenly-spaced tokens
    std::vector<std::vector<float>> centers(n_clusters, std::vector<float>(n_embd_k));
    for (size_t c = 0; c < n_clusters; c++) {
        size_t idx = c * n_tokens / n_clusters;
        std::memcpy(centers[c].data(), k_data + idx * n_embd_k, n_embd_k * sizeof(float));
    }

    // K-means: assign each token to nearest center
    std::vector<int32_t> assignments(n_tokens, 0);
    for (int iter = 0; iter < 5; iter++) {
        // Assignment step
        for (size_t i = 0; i < n_tokens; i++) {
            const float * token_k = k_data + i * n_embd_k;
            float best_sim = -INFINITY;
            int32_t best_c = 0;
            for (size_t c = 0; c < n_clusters; c++) {
                float sim = dot_product(token_k, centers[c].data(), n_embd_k);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_c = (int32_t)c;
                }
            }
            assignments[i] = best_c;
        }

        // Update step
        std::vector<float> new_centers(n_clusters * n_embd_k, 0.0f);
        std::vector<int> counts(n_clusters, 0);
        for (size_t i = 0; i < n_tokens; i++) {
            int32_t c = assignments[i];
            const float * token_k = k_data + i * n_embd_k;
            float * center = new_centers.data() + c * n_embd_k;
            for (size_t j = 0; j < n_embd_k; j++) {
                center[j] += token_k[j];
            }
            counts[c]++;
        }
        for (size_t c = 0; c < n_clusters; c++) {
            if (counts[c] > 0) {
                for (size_t j = 0; j < n_embd_k; j++) {
                    centers[c][j] = new_centers[c * n_embd_k + j] / counts[c];
                }
            }
        }
    }

    // Create cluster objects
    std::vector<std::vector<uint32_t>> cluster_positions(n_clusters);
    for (size_t i = 0; i < n_tokens; i++) {
        cluster_positions[assignments[i]].push_back(token_positions[i]);
    }

    for (size_t c = 0; c < n_clusters; c++) {
        if (cluster_positions[c].empty()) continue;

        llama_kv_cluster cluster;
        cluster.cluster_id = next_cluster_id_++;
        cluster.token_positions = std::move(cluster_positions[c]);
        cluster.center_k = std::move(centers[c]);

        // Compute variance using cosine distance
        float var = 0.0f;
        for (auto cell_idx : cluster.token_positions) {
            // Find the original k_data index
            for (size_t i = 0; i < n_tokens; i++) {
                if (token_positions[i] == cell_idx) {
                    const float * token_k = k_data + i * n_embd_k;
                    float cosine_sim = cosine_similarity(token_k, cluster.center_k.data(), n_embd_k);
                    float dist = 1.0f - cosine_sim; // cosine distance in [0, 2]
                    var += dist * dist;
                    break;
                }
            }
        }
        cluster.variance = cluster.token_positions.size() > 1 ?
            var / cluster.token_positions.size() : 0.0f;

        cluster.is_resident = true;
        cluster.disk_offset = 0;
        cluster.disk_size = 0;

        clusters_.push_back(std::move(cluster));
    }

    n_resident_ = (int)clusters_.size();

    // Build pos_to_cluster map and LRU list
    for (size_t ci = 0; ci < clusters_.size(); ci++) {
        for (auto pos : clusters_[ci].token_positions) {
            pos_to_cluster_[pos] = clusters_[ci].cluster_id;
        }
        lru_list_.push_back(clusters_[ci].cluster_id);
    }
}

cluster_update_result llama_kv_cluster_manager::update(uint32_t cell_idx, const float * k_vector, size_t n_embd_k) {
    cluster_update_result result;
    if (clusters_.empty()) return result;

    // Find nearest cluster by dot product
    float best_sim = -INFINITY;
    size_t best_ci = 0;
    for (size_t ci = 0; ci < clusters_.size(); ci++) {
        float sim = dot_product(k_vector, clusters_[ci].center_k.data(), n_embd_k);
        if (sim > best_sim) {
            best_sim = sim;
            best_ci = ci;
        }
    }

    auto & cluster = clusters_[best_ci];
    result.assigned_cluster = cluster.cluster_id;

    // Add token to cluster
    cluster.token_positions.push_back(cell_idx);
    pos_to_cluster_[cell_idx] = cluster.cluster_id;

    // Update centroid (moving average)
    float weight_new = 1.0f / (float)cluster.token_positions.size();
    float weight_old = 1.0f - weight_new;
    for (size_t j = 0; j < n_embd_k; j++) {
        cluster.center_k[j] = weight_old * cluster.center_k[j] + weight_new * k_vector[j];
    }

    // Compute new variance using cosine distance in [0, 2]
    float cosine_sim = cosine_similarity(k_vector, cluster.center_k.data(), n_embd_k);
    float dist = 1.0f - cosine_sim;
    float new_var_contrib = dist * dist;
    cluster.variance = (cluster.variance * (cluster.token_positions.size() - 1) + new_var_contrib) /
                       cluster.token_positions.size();

    // Increment access count
    cluster.access_count++;

    // Capture cluster_id before any split that may invalidate the reference
    int32_t cur_cluster_id = cluster.cluster_id;

    // Check if split is needed
    if (cluster.variance > var_threshold_ && cluster.token_positions.size() > 4) {
        int32_t new1_id, new2_id;
        split_cluster(best_ci, k_vector, n_embd_k, &new1_id, &new2_id);
        result.split_occurred = true;
        result.old_cluster_id = cur_cluster_id;
        result.new1_cluster_id = new1_id;
        result.new2_cluster_id = new2_id;

        // Notify callback
        if (on_split) on_split(cur_cluster_id, new1_id, new2_id);
    }

    // Mark as accessed (use captured id — reference may be dangling after split)
    mark_accessed(cur_cluster_id);
    return result;
}

void llama_kv_cluster_manager::add_position(int32_t cluster_id, uint32_t cell_idx) {
    int32_t idx = get_cluster_idx(cluster_id);
    if (idx < 0) return;
    // Only add if not already present
    auto & positions = clusters_[idx].token_positions;
    if (std::find(positions.begin(), positions.end(), cell_idx) == positions.end()) {
        positions.push_back(cell_idx);
        pos_to_cluster_[cell_idx] = cluster_id;
    }
}

void llama_kv_cluster_manager::split_cluster(size_t cluster_idx, const float * k_data, size_t n_embd_k,
        int32_t * out_new1_id, int32_t * out_new2_id) {
    auto & old_cluster = clusters_[cluster_idx];
    if (old_cluster.token_positions.size() < 4) return;

    // Create two new clusters. Child without new token keeps old cluster_id (Item 5)
    llama_kv_cluster new1, new2;
    // new1 = child WITHOUT the new token → keeps old cluster_id
    // new2 = child WITH the new token → gets new cluster_id
    int32_t new_token_pos_idx = (int32_t)old_cluster.token_positions.size() - 1;
    new1.cluster_id = old_cluster.cluster_id;  // preserve old id
    new2.cluster_id = next_cluster_id_++;
    new1.is_resident = true;
    new2.is_resident = true;

    // Distribute positions: child without new token (new1, old_id) gets alternating
    // positions; child with new token (new2, new_id) gets the rest including the new token
    for (size_t i = 0; i < old_cluster.token_positions.size(); i++) {
        if (i % 2 == 0) {
            new1.token_positions.push_back(old_cluster.token_positions[i]);
        } else {
            new2.token_positions.push_back(old_cluster.token_positions[i]);
        }
    }
    // Ensure new token (last position) ends up in new2
    if (!new1.token_positions.empty() && new1.token_positions.back() == old_cluster.token_positions.back()) {
        // Swap last elements between new1 and new2
        auto tmp_pos = new1.token_positions.back(); new1.token_positions.pop_back();
        new1.token_positions.push_back(new2.token_positions.back());
        new2.token_positions.back() = tmp_pos;
    }

    // Copy center (approximate: use old center for both, will be refined over time)
    new1.center_k = old_cluster.center_k;
    new2.center_k = old_cluster.center_k;
    new1.variance = old_cluster.variance / 2.0f;
    new2.variance = old_cluster.variance / 2.0f;
    new1.disk_offset = 0;
    new2.disk_offset = 0;
    new1.disk_size = 0;
    new2.disk_size = 0;

    // Update pos_to_cluster_ mapping
    for (auto pos : new1.token_positions) {
        pos_to_cluster_[pos] = new1.cluster_id;
    }
    for (auto pos : new2.token_positions) {
        pos_to_cluster_[pos] = new2.cluster_id;
    }

    // Capture ids before erase/move invalidates references
    int32_t old_id = old_cluster.cluster_id;
    int32_t id1 = new1.cluster_id;
    int32_t id2 = new2.cluster_id;

    // Replace old cluster with new clusters
    clusters_.erase(clusters_.begin() + cluster_idx);
    clusters_.push_back(std::move(new1));
    clusters_.push_back(std::move(new2));

    // Update LRU: remove old, add new
    lru_list_.remove(old_id);
    lru_list_.push_back(id1);
    lru_list_.push_back(id2);

    n_resident_ += 1; // one extra cluster from split

    if (out_new1_id) *out_new1_id = id1;
    if (out_new2_id) *out_new2_id = id2;

    stats_.splits++; // Item 6: track split count
}

std::vector<uint32_t> llama_kv_cluster_manager::get_resident_positions() const {
    std::vector<uint32_t> positions;
    for (const auto & cluster : clusters_) {
        if (cluster.is_resident) {
            positions.insert(positions.end(),
                cluster.token_positions.begin(), cluster.token_positions.end());
        }
    }
    return positions;
}

size_t llama_kv_cluster_manager::get_n_resident_positions() const {
    size_t n = 0;
    for (const auto & cluster : clusters_) {
        if (cluster.is_resident) {
            n += cluster.token_positions.size();
        }
    }
    return n;
}

std::vector<int32_t> llama_kv_cluster_manager::get_evict_candidates() const {
    // Weighted eviction (Item 6): combine LRU recency + access frequency
    // Lower score = better eviction candidate
    std::vector<std::pair<int32_t, float>> scored_candidates;
    int lru_pos = 0;
    for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it, ++lru_pos) {
        int32_t cid = *it;
        int32_t idx = get_cluster_idx(cid);
        if (idx < 0) continue;
        // Non-resident clusters are cheapest to "evict" (already on disk)
        if (!clusters_[idx].is_resident) {
            scored_candidates.emplace_back(cid, -1.0f); // always evict first
        } else {
            // Score = lru_pos / (1 + access_count) — lower = colder + less used
            float score = (float)(lru_pos + 1) / (1.0f + (float)clusters_[idx].access_count);
            scored_candidates.emplace_back(cid, score);
        }
    }
    // Sort: non-resident first, then by score ascending (colder first)
    std::sort(scored_candidates.begin(), scored_candidates.end(),
              [](const auto & a, const auto & b) { return a.second < b.second; });

    std::vector<int32_t> candidates;
    for (auto & [cid, _] : scored_candidates) candidates.push_back(cid);
    return candidates;
}

void llama_kv_cluster_manager::mark_accessed(int32_t cluster_id) {
    lru_list_.remove(cluster_id);
    lru_list_.push_front(cluster_id);
    // Track cache hit (Item 6)
    int32_t idx = get_cluster_idx(cluster_id);
    if (idx >= 0 && clusters_[idx].is_resident) stats_.hits++;
    else stats_.misses++;
}

void llama_kv_cluster_manager::set_resident(int32_t cluster_id, bool resident,
                                             int64_t disk_offset, size_t disk_size) {
    int32_t idx = get_cluster_idx(cluster_id);
    if (idx < 0) return;

    if (clusters_[idx].is_resident != resident) {
        clusters_[idx].is_resident = resident;
        n_resident_ += resident ? 1 : -1;
        if (!resident) {
            clusters_[idx].disk_offset = disk_offset;
            clusters_[idx].disk_size = disk_size;
        } else {
            clusters_[idx].disk_offset = 0;
            clusters_[idx].disk_size = 0;
        }
    }
}

const std::vector<uint32_t> & llama_kv_cluster_manager::get_positions(int32_t cluster_id) const {
    for (const auto & cluster : clusters_) {
        if (cluster.cluster_id == cluster_id) {
            return cluster.token_positions;
        }
    }
    static const std::vector<uint32_t> empty;
    return empty;
}

int64_t llama_kv_cluster_manager::get_disk_offset(int32_t cluster_id) const {
    int32_t idx = get_cluster_idx(cluster_id);
    return idx >= 0 ? clusters_[idx].disk_offset : 0;
}

size_t llama_kv_cluster_manager::get_disk_size(int32_t cluster_id) const {
    int32_t idx = get_cluster_idx(cluster_id);
    return idx >= 0 ? clusters_[idx].disk_size : 0;
}

int32_t llama_kv_cluster_manager::get_cluster_id(uint32_t cell_idx) const {
    auto it = pos_to_cluster_.find(cell_idx);
    return it != pos_to_cluster_.end() ? it->second : -1;
}

bool llama_kv_cluster_manager::is_cluster_resident(int32_t cluster_id) const {
    int32_t idx = get_cluster_idx(cluster_id);
    return idx >= 0 && clusters_[idx].is_resident;
}

std::vector<uint32_t> llama_kv_cluster_manager::get_top_cluster_positions(
        uint32_t current_cell_idx, int k) {
    std::vector<uint32_t> positions;

    if (clusters_.empty() || k <= 0) {
        return positions;
    }

    // Find the cluster that contains the current cell
    int32_t current_cluster_id = get_cluster_id(current_cell_idx);

    // Collect up to k clusters from LRU front (MRU → LRU). Cold clusters
    // remain eligible: the caller resolves their positions through
    // ensure_resident() before building the sparse attention view.
    std::vector<int32_t> selected_ids;

    if (current_cluster_id >= 0) {
        selected_ids.push_back(current_cluster_id);
    }

    for (auto it = lru_list_.begin(); it != lru_list_.end() && (int)selected_ids.size() < k; ++it) {
        int32_t cid = *it;
        // Skip if already selected or stale. Do not filter non-resident
        // clusters here, otherwise ensure_resident() can never page them in.
        if (cid == current_cluster_id) continue;
        int32_t idx = get_cluster_idx(cid);
        if (idx < 0) continue;
        selected_ids.push_back(cid);
    }

    // If the current cell is not clustered, fall back to the same LRU order.
    if (current_cluster_id < 0) {
        selected_ids.clear();
        for (auto it = lru_list_.begin(); it != lru_list_.end() && (int)selected_ids.size() < k; ++it) {
            int32_t idx = get_cluster_idx(*it);
            if (idx < 0) continue;
            selected_ids.push_back(*it);
        }
    }

    // Collect all token positions from selected clusters
    for (auto cid : selected_ids) {
        int32_t idx = get_cluster_idx(cid);
        if (idx < 0) continue;
        const auto & pos = clusters_[idx].token_positions;
        positions.insert(positions.end(), pos.begin(), pos.end());
    }

    return positions;
}

std::vector<std::pair<int32_t, float>> llama_kv_cluster_manager::score_clusters(
        const float * q, size_t n_embd_k) const {
    std::vector<std::pair<int32_t, float>> scores;
    for (const auto & cluster : clusters_) {
        if (!cluster.is_resident) continue;
        float sim = dot_product(q, cluster.center_k.data(), n_embd_k);
        scores.emplace_back(cluster.cluster_id, sim);
    }
    // Sort descending by score
    std::sort(scores.begin(), scores.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });
    return scores;
}

std::vector<std::pair<uint32_t, float>> llama_kv_cluster_manager::score_tokens_in_cluster(
        int32_t cluster_id, const float * q, size_t n_embd_k,
        const float * k_storage_data, size_t k_storage_row_bytes,
        int k_storage_type) const {
    std::vector<std::pair<uint32_t, float>> scores;
    int32_t idx = get_cluster_idx(cluster_id);
    if (idx < 0) return scores;

    const auto & positions = clusters_[idx].token_positions;
    for (auto cell_idx : positions) {
        std::vector<float> token_k(n_embd_k);
        const char * row_ptr = reinterpret_cast<const char *>(k_storage_data) + cell_idx * k_storage_row_bytes;
        if (k_storage_type == GGML_TYPE_F32) {
            std::memcpy(token_k.data(), row_ptr, n_embd_k * sizeof(float));
        } else if (k_storage_type == GGML_TYPE_F16) {
            const auto * fp16_ptr = reinterpret_cast<const ggml_fp16_t *>(row_ptr);
            for (size_t j = 0; j < n_embd_k; j++) {
                token_k[j] = ggml_fp16_to_fp32(fp16_ptr[j]);
            }
        } else {
            continue;
        }
        float sim = dot_product(q, token_k.data(), n_embd_k);
        scores.emplace_back(cell_idx, sim);
    }
    // Sort descending by score (Item 2: keep full sort for standalone calls)
    std::sort(scores.begin(), scores.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });
    return scores;
}

// Item 3: Parallel token scoring across clusters
std::vector<std::pair<uint32_t, float>> llama_kv_cluster_manager::score_tokens_parallel(
        const std::vector<int32_t> & cluster_ids, const float * q, size_t n_embd_k,
        const float * k_storage_data, size_t k_storage_row_bytes,
        int k_storage_type, int k_tokens) const {
    if (cluster_ids.empty()) {
        return {};
    }
    if (cluster_ids.size() == 1) {
        // Single cluster: direct call with partial_sort
        auto result = score_tokens_in_cluster(cluster_ids[0], q, n_embd_k,
            k_storage_data, k_storage_row_bytes, k_storage_type);
        if (k_tokens > 0 && (int)result.size() > k_tokens) {
            std::partial_sort(result.begin(), result.begin() + k_tokens, result.end(),
                [](const auto & a, const auto & b) { return a.second > b.second; });
            result.resize(k_tokens);
        }
        return result;
    }

    // Multi-cluster: parallel scoring
    std::vector<std::future<std::vector<std::pair<uint32_t, float>>>> futures;
    for (auto cid : cluster_ids) {
        futures.push_back(std::async(std::launch::async, [this, cid, q, n_embd_k,
                k_storage_data, k_storage_row_bytes, k_storage_type]() {
            return score_tokens_in_cluster(cid, q, n_embd_k,
                k_storage_data, k_storage_row_bytes, k_storage_type);
        }));
    }

    // Merge results
    std::vector<std::pair<uint32_t, float>> merged;
    for (auto & f : futures) {
        auto partial = f.get();
        merged.insert(merged.end(), partial.begin(), partial.end());
    }

    // Item 2 (min-heap): use partial_sort for top-k selection
    if (k_tokens > 0 && (int)merged.size() > k_tokens) {
        std::partial_sort(merged.begin(), merged.begin() + k_tokens, merged.end(),
            [](const auto & a, const auto & b) { return a.second > b.second; });
        merged.resize(k_tokens);
    } else if (merged.size() > 1) {
        std::sort(merged.begin(), merged.end(),
            [](const auto & a, const auto & b) { return a.second > b.second; });
    }
    return merged;
}
