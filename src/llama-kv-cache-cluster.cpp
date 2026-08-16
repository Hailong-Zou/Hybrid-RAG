#include "llama-kv-cache-cluster.h"
#include "llama-kv-cache.h"
#include "llama-cparams.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <set>
#include <sys/mman.h>
#include <unistd.h>
#include "llama-impl.h"

namespace {

// MADV_DONTNEED operates on whole VM pages, not individual KV rows. A KV row
// is commonly 2048 bytes on Qwen3-4B, so dropping one aligned row can also
// discard an adjacent resident token that shares the same 4096-byte page.
// Release a page only when every KV row intersecting it is already evicted.
static size_t madvise_fully_evicted_kv_pages(
        ggml_tensor * tensor,
        const std::vector<uint32_t> & candidate_positions,
        const std::unordered_set<uint32_t> & evicted_cells) {
    if (!tensor || !tensor->data || !tensor->buffer ||
            !ggml_backend_buffer_is_host(tensor->buffer) ||
            tensor->ne[1] <= 0 || tensor->nb[1] == 0) {
        return 0;
    }

    long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        page_size_long = 4096;
    }
    const uintptr_t page_size = static_cast<uintptr_t>(page_size_long);
    const uintptr_t data_begin = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t row_stride = tensor->nb[1];
    const size_t n_rows = static_cast<size_t>(tensor->ne[1]);
    if (n_rows > SIZE_MAX / row_stride) {
        return 0;
    }
    const size_t tensor_span = n_rows * row_stride;
    if (data_begin > UINTPTR_MAX - tensor_span) {
        return 0;
    }
    const uintptr_t data_end = data_begin + tensor_span;

    std::set<uintptr_t> candidate_pages;
    for (uint32_t cell : candidate_positions) {
        if (cell >= n_rows) {
            continue;
        }
        const uintptr_t row_begin = data_begin + static_cast<size_t>(cell) * row_stride;
        const uintptr_t row_end = row_begin + row_stride;
        for (uintptr_t page = row_begin - row_begin % page_size;
                page < row_end;
                page += page_size) {
            candidate_pages.insert(page);
        }
    }

    size_t released_bytes = 0;
    for (uintptr_t page : candidate_pages) {
        if (page < data_begin || page > UINTPTR_MAX - page_size ||
                page + page_size > data_end) {
            continue;
        }
        const size_t first_row = static_cast<size_t>((page - data_begin) / row_stride);
        const size_t last_row = std::min(
                n_rows - 1,
                static_cast<size_t>((page + page_size - 1 - data_begin) / row_stride));
        bool safe = true;
        for (size_t row = first_row; row <= last_row; ++row) {
            if (evicted_cells.count(static_cast<uint32_t>(row)) == 0) {
                safe = false;
                break;
            }
        }
        if (safe && madvise(reinterpret_cast<void *>(page), page_size, MADV_DONTNEED) == 0) {
            released_bytes += page_size;
        }
    }
    return released_bytes;
}

} // namespace

llama_kv_cache_cluster::llama_kv_cache_cluster(
        const llama_model & model, const llama_hparams & hparams,
        ggml_type type_k, ggml_type type_v, bool v_trans,
        bool offload, bool unified, uint32_t kv_size, uint32_t n_seq_max,
        uint32_t n_pad, uint32_t n_swa, llama_swa_type swa_type,
        llama_memory_t mem_other, const layer_filter_cb & filter,
        const layer_reuse_cb & reuse, const layer_share_cb & share,
        const llama_cparams & cparams)
    : llama_kv_cache(model, hparams, type_k, type_v, v_trans, offload, unified,
                     kv_size, n_seq_max, n_pad, n_swa, swa_type, mem_other,
                     filter, reuse, share), cparams_(cparams) {
    const char * pager_root_env = getenv("LLAMA_KV_PAGER_DIR");
    std::string pager_root =
            pager_root_env && pager_root_env[0] != '\0'
                ? pager_root_env
                : "/tmp";

    while (pager_root.size() > 1 && pager_root.back() == '/') {
        pager_root.pop_back();
    }

    // Namespace pager files by process, model and Context. This prevents
    // multiple Generation models / server slots / candidate Contexts from
    // overwriting each other's on-disk KV data.
    disk_path_ =
            pager_root +
            "/llama_kv_cluster_pid_" +
            std::to_string(static_cast<long long>(getpid())) +
            "_model_" +
            std::to_string(reinterpret_cast<uintptr_t>(&model)) +
            "_ctx_" +
            std::to_string(reinterpret_cast<uintptr_t>(this));

    LLAMA_LOG_INFO(
            "%s: KV pager path: %s\n",
            __func__,
            disk_path_.c_str());

    pager_ = std::make_unique<llama_kv_pager>(
            disk_path_,
            (int) layers.size());
    for (size_t i = 0; i < layers.size(); i++)
        cluster_mgrs_.push_back(std::make_unique<llama_kv_cluster_manager>(cparams_.max_resident_clusters, cparams_.cluster_var_threshold));
}

llama_kv_cache_cluster::~llama_kv_cache_cluster() {
    const auto * mgr = cluster_mgrs_.empty() ? nullptr : cluster_mgrs_[0].get();
    fprintf(stderr,
            "KV_CLUSTER_SUMMARY: context=%p init_count=%llu clusters=%d resident=%d "
            "eviction_count=%llu page_in_count=%llu handoff_rebuild_count=%llu "
            "handoff_rebuild_failures=%llu metadata_errors=%llu disk_clusters=%zu evicted_cells=%zu\n",
            (void *) this,
            (unsigned long long) cluster_init_count_,
            mgr ? mgr->get_n_clusters() : 0,
            mgr ? mgr->get_n_resident() : 0,
            (unsigned long long) eviction_count_,
            (unsigned long long) page_in_count_,
            (unsigned long long) handoff_rebuild_count_,
            (unsigned long long) handoff_rebuild_failures_,
            (unsigned long long) metadata_errors_,
            disk_clusters_.size(), evicted_cells_.size());
}

llama_memory_context_ptr llama_kv_cache_cluster::init_batch(llama_batch_allocr & b, uint32_t n, bool e) {
    GGML_UNUSED(e);
    if (prefill_done_) {
        // Residency is resolved after the current ubatch sequence IDs are
        // known.  Doing it here used the global last cell and could page in a
        // different candidate's cluster.
        collect_and_flush_token_buffers();
    }
    do {
        b.split_reset(); std::vector<llama_ubatch> ubs;
        while (true) { auto ub = b.split_simple(n); if (ub.n_tokens == 0) break; ubs.push_back(std::move(ub)); }
        if (b.get_n_used() < b.get_n_tokens()) break;
        auto sinfos = prepare(ubs); if (sinfos.empty()) break;
        auto ctx = std::make_unique<llama_kv_cache_cluster_context>(this, std::move(ubs));
        ctx->init_from_sinfos(std::move(sinfos), (int32_t)get_size());
        return ctx;
    } while (false);
    return std::make_unique<llama_kv_cache_cluster_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

size_t llama_kv_cache_cluster::get_n_embd_k(int32_t il) const { int32_t ik=il; auto it=map_layer_ids.find(il); if(it!=map_layer_ids.end())ik=it->second; if(ik<0||(size_t)ik>=layers.size())return 0; return (size_t)layers[ik].k->ne[0]; }
size_t llama_kv_cache_cluster::get_n_embd_v(int32_t il) const { int32_t ik=il; auto it=map_layer_ids.find(il); if(it!=map_layer_ids.end())ik=it->second; if(ik<0||(size_t)ik>=layers.size())return 0; return (size_t)layers[ik].v->ne[0]; }

bool llama_kv_cache_cluster::rebuild_after_restore(llama_seq_id dest_seq_id) {
    handoff_rebuild_count_++;
    // A decode executor is persistent across requests. Native sequence-state
    // restore replaces its KV cells, but none of the following derived state
    // is part of the handoff format. Drop all request-local metadata before
    // deriving a fresh resident cluster view from the restored KV cache.
    if (dest_seq_id < 0 || (uint32_t) dest_seq_id >= n_seq_max) {
        handoff_rebuild_failures_++;
        fprintf(stderr,
                "KV_HANDOFF_REBUILD_FAIL: context=%p seq=%d reason=invalid_seq n_seq_max=%u\n",
                (void *) this, (int) dest_seq_id, n_seq_max);
        return false;
    }

    if (pager_) {
        pager_->sync();
    }
    pending_writes_.clear();
    prefetch_handles_.clear();
    pager_.reset();
    pager_ = std::make_unique<llama_kv_pager>(disk_path_, (int) layers.size());

    disk_clusters_.clear();
    evicted_cells_.clear();
    pending_tokens_.clear();
    token_buffer_.clear();

    cluster_mgrs_.clear();
    cluster_mgrs_.reserve(layers.size());
    for (size_t il = 0; il < layers.size(); ++il) {
        cluster_mgrs_.push_back(std::make_unique<llama_kv_cluster_manager>(
                cparams_.max_resident_clusters,
                cparams_.cluster_var_threshold));
    }

    prefill_done_ = false;
    prefill_tok_count_ = 0;
    n_steps_since_update_ = 0;
    n_total_steps_ = 0;
    cluster_cohort_.clear();
    decode_seen_.clear();

    const auto active = get_active_seq_ids();
    if (std::find(active.begin(), active.end(), dest_seq_id) == active.end()) {
        handoff_rebuild_failures_++;
        fprintf(stderr,
                "KV_HANDOFF_REBUILD_FAIL: context=%p seq=%d reason=restored_seq_empty active=%zu\n",
                (void *) this, (int) dest_seq_id, active.size());
        return false;
    }

    cluster_cohort_.insert(active.begin(), active.end());
    // A restored handoff is already at the prefill/decode boundary. Mark the
    // restored cohort as observed so the first single-token decode cannot be
    // mistaken for a fresh-context transition or add the current token twice.
    decode_seen_.insert(active.begin(), active.end());

    size_t restored_cells = 0;
    uint32_t physical_min = std::numeric_limits<uint32_t>::max();
    uint32_t physical_max = 0;
    llama_pos logical_min = std::numeric_limits<llama_pos>::max();
    llama_pos logical_max = std::numeric_limits<llama_pos>::min();
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t ci = 0; ci < cells.size(); ++ci) {
            if (cells.is_empty(ci) || !cells.seq_has(ci, dest_seq_id)) continue;
            restored_cells++;
            physical_min = std::min(physical_min, ci);
            physical_max = std::max(physical_max, ci);
            logical_min = std::min(logical_min, cells.pos_get(ci));
            logical_max = std::max(logical_max, cells.pos_get(ci));
        }
    }

    fprintf(stderr,
            "KV_HANDOFF_REBUILD_BEGIN: context=%p seq=%d streams=%u active=%zu cells=%zu physical_min=%u physical_max=%u logical_min=%d logical_max=%d disk_clusters=0 evicted_cells=0 prefill_done=0\n",
            (void *) this, (int) dest_seq_id, n_stream, active.size(), restored_cells,
            restored_cells ? physical_min : 0, restored_cells ? physical_max : 0,
            restored_cells ? (int) logical_min : -1,
            restored_cells ? (int) logical_max : -1);

    cluster_init();

    const int n_clusters = cluster_mgrs_.empty() ? 0 : cluster_mgrs_[0]->get_n_clusters();
    const int n_resident = cluster_mgrs_.empty() ? 0 : cluster_mgrs_[0]->get_n_resident();
    const bool ok = restored_cells > 0 && prefill_done_ && n_clusters > 0;
    if (!ok) handoff_rebuild_failures_++;
    fprintf(stderr,
            "KV_HANDOFF_REBUILD_DONE: context=%p seq=%d ok=%d cells=%zu clusters=%d resident=%d cohort=%zu decode_seen=%zu disk_clusters=%zu evicted_cells=%zu prefill_done=%d\n",
            (void *) this, (int) dest_seq_id, ok ? 1 : 0, restored_cells,
            n_clusters, n_resident, cluster_cohort_.size(), decode_seen_.size(),
            disk_clusters_.size(), evicted_cells_.size(), prefill_done_ ? 1 : 0);
    return ok;
}

void llama_kv_cache_cluster::cluster_init() {
    if (layers.empty()) return;
    auto & cells = v_cells[0];
    std::vector<uint32_t> positions;
    positions.reserve(cells.get_used());
    for (uint32_t ci = 0; ci < cells.size(); ++ci) {
        if (!cells.is_empty(ci)) {
            positions.push_back(ci);
        }
    }
    if (positions.empty()) return;
    // Skip re-clustering if token count hasn't increased (e.g., repeated calls during prefill)
    size_t prev_count = prefill_tok_count_;
    if ((size_t)positions.size() <= prev_count) return;
    prefill_tok_count_ = positions.size();

    // K-means seed count is driven by data size, not by max_resident.
    // max_resident controls how many clusters stay in RAM after init;
    // it must not limit how many semantic groups K-means can find.
    // Cap at 64 to bound K-means time on long contexts.
    size_t max_initial = std::min((size_t)64,
                                  std::max((size_t)cparams_.cluster_k * 2,
                                           std::max((size_t)1, positions.size() / 16)));
    for (size_t il_kv = 0; il_kv < layers.size(); il_kv++) {
        ggml_tensor * k0 = layers[il_kv].k; if (!k0) continue;
        size_t n_embd_k = (size_t)k0->ne[0], type_size = ggml_type_size(k0->type);
        std::vector<float> k_data(positions.size() * n_embd_k);
        for (size_t i = 0; i < positions.size(); i++) {
            size_t cell_idx = positions[i], offset = cell_idx * n_embd_k * type_size;
            if (k0->type == GGML_TYPE_F32) { ggml_backend_tensor_get(k0, k_data.data()+i*n_embd_k, offset, n_embd_k*sizeof(float)); }
            else if (k0->type == GGML_TYPE_F16) { std::vector<ggml_fp16_t> t(n_embd_k); ggml_backend_tensor_get(k0,t.data(),offset,n_embd_k*sizeof(ggml_fp16_t)); for(size_t j=0;j<n_embd_k;j++)k_data[i*n_embd_k+j]=ggml_fp16_to_fp32(t[j]); }
            else continue;
        }
        cluster_mgrs_[il_kv]->init(positions, k_data.data(), n_embd_k, max_initial);
    }
    prefill_done_ = true;
    cluster_init_count_++;

    fprintf(stderr,
            "KV_MULTI_INIT: context=%p streams=%u cohort=%zu positions=%zu clusters=%d prefill_done=1\n",
            (void *) this,
            n_stream,
            cluster_cohort_.size(),
            positions.size(),
            cluster_mgrs_.empty() ? 0 : cluster_mgrs_[0]->get_n_clusters());

}

bool llama_kv_cache_cluster::cell_belongs_to_seq(uint32_t cell_idx, llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) return false;
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        if (cell_idx < cells.size() && !cells.is_empty(cell_idx) && cells.seq_has(cell_idx, seq_id)) {
            return true;
        }
    }
    return false;
}

std::vector<uint32_t> llama_kv_cache_cluster::get_unclustered_positions_before(
        llama_seq_id seq_id, llama_pos before_pos) const {
    std::vector<uint32_t> result;
    if (cluster_mgrs_.empty() || !cluster_mgrs_[0]) return result;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t ci = 0; ci < cells.size(); ++ci) {
            if (cells.is_empty(ci) || !cells.seq_has(ci, seq_id)) continue;
            if (cells.pos_get(ci) >= before_pos) continue;
            if (cluster_mgrs_[0]->get_cluster_id(ci) >= 0) continue;
            result.push_back(ci);
        }
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<llama_seq_id> llama_kv_cache_cluster::get_active_seq_ids() const {
    std::vector<llama_seq_id> result;
    for (uint32_t seq = 0; seq < n_seq_max; ++seq) {
        bool active = false;
        for (uint32_t s = 0; s < n_stream && !active; ++s) {
            const auto & cells = v_cells[s];
            for (uint32_t ci = 0; ci < cells.size(); ++ci) {
                if (!cells.is_empty(ci) && cells.seq_has(ci, (llama_seq_id) seq)) {
                    active = true;
                    break;
                }
            }
        }
        if (active) result.push_back((llama_seq_id) seq);
    }
    return result;
}

uint32_t llama_kv_cache_cluster::get_last_cell_idx(llama_seq_id seq_id) const {
    uint32_t best_cell = 0;
    llama_pos best_pos = std::numeric_limits<llama_pos>::min();
    bool found = false;
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t ci = 0; ci < cells.size(); ++ci) {
            if (cells.is_empty(ci) || !cells.seq_has(ci, seq_id)) continue;
            const llama_pos pos = cells.pos_get(ci);
            if (!found || pos > best_pos) {
                found = true;
                best_pos = pos;
                best_cell = ci;
            }
        }
    }
    return best_cell;
}

bool llama_kv_cache_cluster::sparse_ready_for(const std::vector<llama_seq_id> & seq_ids) const {
    if (!prefill_done_ || seq_ids.empty() || n_stream != 1) return false;
    for (llama_seq_id seq_id : seq_ids) {
        if (cluster_cohort_.count(seq_id) == 0) return false;
    }
    return true;
}

void llama_kv_cache_cluster::observe_decode_sequences(const std::vector<llama_seq_id> & seq_ids) {
    if (seq_ids.empty()) return;

    for (llama_seq_id seq_id : seq_ids) {
        decode_seen_.insert(seq_id);
    }

    if (prefill_done_) return;

    const auto active = get_active_seq_ids();
    cluster_cohort_.insert(active.begin(), active.end());
    cluster_cohort_.insert(seq_ids.begin(), seq_ids.end());

    bool ready = !cluster_cohort_.empty();
    for (llama_seq_id seq_id : cluster_cohort_) {
        if (decode_seen_.count(seq_id) == 0) {
            ready = false;
            break;
        }
    }

    fprintf(stderr,
            "KV_MULTI_GATE: context=%p observed=%zu active=%zu cohort=%zu decode_seen=%zu ready=%d prefill_done=%d\n",
            (void *) this,
            seq_ids.size(),
            active.size(),
            cluster_cohort_.size(),
            decode_seen_.size(),
            ready ? 1 : 0,
            prefill_done_ ? 1 : 0);

    if (!ready) return;

    cluster_init();
    if (prefill_done_) {
        persist_all_clusters();
        print_stats();
        maybe_evict();
    }
}

void llama_kv_cache_cluster::cluster_update(int32_t il, const float * k_cur, uint32_t cell_idx) {
    if (!prefill_done_ || il < 0 || (size_t)il >= cluster_mgrs_.size()) return;
    auto result = cluster_mgrs_[il]->update(cell_idx, k_cur, (size_t)layers[il].k->ne[0]);

    // Handle split: rewrite disk data for child clusters
    if (result.split_occurred && il == 0) {
        handle_split(result.old_cluster_id, result.new1_cluster_id, result.new2_cluster_id);
    }

    // Buffer token if assigned to a non-resident cluster
    int32_t cid = result.assigned_cluster;
    if (cid >= 0 && !cluster_mgrs_[il]->is_cluster_resident(cid)) {
        // Read V vector for the same token
        size_t n_embd_k = (size_t)layers[il].k->ne[0];
        size_t n_embd_v = (size_t)layers[il].v->ne[0];
        size_t v_type_size = ggml_type_size(layers[il].v->type);

        std::vector<float> v_cur(n_embd_v);
        size_t v_offset = (size_t)cell_idx * n_embd_v * v_type_size;
        if (layers[il].v->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(layers[il].v, v_cur.data(), v_offset, n_embd_v * sizeof(float));
        } else if (layers[il].v->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(n_embd_v);
            ggml_backend_tensor_get(layers[il].v, tmp.data(), v_offset, n_embd_v * sizeof(ggml_fp16_t));
            for (size_t j = 0; j < n_embd_v; j++) v_cur[j] = ggml_fp16_to_fp32(tmp[j]);
        }

        buffer_token(il, cell_idx, cid, k_cur, v_cur.data(), n_embd_k, n_embd_v);
    }

    // Track pending token: add cell_idx to its cluster's pending set
    if (cid >= 0) pending_tokens_[cid].insert(cell_idx);
}
void llama_kv_cache_cluster::maybe_evict() {
    bool needs = false; for (auto & m : cluster_mgrs_) if (m && m->needs_eviction()) { needs = true; break; }
    if (!needs) return;
    auto & mgr = *cluster_mgrs_[0]; auto candidates = mgr.get_evict_candidates();
    int n_to_evict = mgr.get_n_resident() - mgr.get_max_resident(); if (n_to_evict <= 0) return;
    fprintf(stderr,
            "KV_EVICT_PLAN: clusters=%d resident=%d max_resident=%d to_evict=%d disk_clusters=%zu evicted_cells=%zu\n",
            mgr.get_n_clusters(), mgr.get_n_resident(), mgr.get_max_resident(),
            n_to_evict, disk_clusters_.size(), evicted_cells_.size());
    int evicted = 0;
    for (auto cluster_id : candidates) {
        if (evicted >= n_to_evict) break;
        auto & positions = mgr.get_positions(cluster_id); if (positions.empty()) continue;

        // Get existing disk info (for incremental tracking)
        auto existing = disk_clusters_.find(cluster_id);
        bool has_existing = (existing != disk_clusters_.end());

        size_t persisted_bytes = 0;
        int persisted_layers = 0;
        for (size_t il_kv = 0; il_kv < layers.size(); ++il_kv) {
            const auto * desc = pager_->get_descriptor((int32_t) il_kv, cluster_id);
            if (desc && desc->offset >= 0 && (desc->flags & PAGER_DESC_FLAG_VALID)) {
                persisted_bytes += (size_t) desc->size;
                persisted_layers++;
            }
        }

        // Skip write if cluster is already fully persisted (Phase 1.3)
        if (has_existing && existing->second.persisted_token_count >= positions.size()) {
            // Already fully persisted — just evict without writing
            for (auto pos : positions) evicted_cells_.insert(pos);
            for (auto & m : cluster_mgrs_) m->set_resident(cluster_id, false);
            size_t released_bytes = 0;
            for (size_t il_kv = 0; il_kv < layers.size(); il_kv++) {
                auto *ks=layers[il_kv].k, *vs=layers[il_kv].v;
                if(!ks||!vs||!ks->data||!vs->data) continue;
                released_bytes += madvise_fully_evicted_kv_pages(ks, positions, evicted_cells_);
                released_bytes += madvise_fully_evicted_kv_pages(vs, positions, evicted_cells_);
            }
            evicted++;
            eviction_count_++;
            fprintf(stderr,
                    "KV_EVICT_COMPLETE: cluster=%d tokens=%zu layers=%d bytes=%zu released_bytes=%zu reused_persisted=1 resident=%d max_resident=%d evicted_cells=%zu disk_clusters=%zu\n",
                    cluster_id, positions.size(), persisted_layers, persisted_bytes, released_bytes,
                    mgr.get_n_resident(), mgr.get_max_resident(), evicted_cells_.size(),
                    disk_clusters_.size());
            continue;
        }

        // Compute new positions (not yet persisted) — incremental write (Phase 1.3)
        std::vector<uint32_t> new_positions;
        size_t persisted_count = has_existing ? existing->second.persisted_token_count : 0;
        for (size_t i = persisted_count; i < positions.size(); i++) {
            new_positions.push_back(positions[i]);
        }

        if (new_positions.empty()) {
            // No new data, but still mark evicted if we have disk info
            if (has_existing) {
                for (auto pos : positions) evicted_cells_.insert(pos);
                for (auto & m : cluster_mgrs_) m->set_resident(cluster_id, false);
                evicted++;
                eviction_count_++;
            }
            continue;
        }

        // Token Buffer: include pending tokens in the write
        auto pit = pending_tokens_.find(cluster_id);
        if (pit != pending_tokens_.end()) {
            for (auto p : pit->second) {
                bool already = has_existing && existing->second.persisted_set.count(p);
                if (!already && std::find(new_positions.begin(), new_positions.end(), p) == new_positions.end())
                    new_positions.push_back(p);
            }
            pending_tokens_.erase(pit);
        }

        // write_cluster_at() replaces the descriptor payload; it is not an append API.
        // Re-write the complete cluster so descriptor.n_tokens and disk contents stay
        // aligned with info.positions after a resident cluster gains new tokens.
        size_t written_bytes = 0;
        int written_layers = 0;
        for (size_t il_kv = 0; il_kv < layers.size(); il_kv++) {
            auto *ks=layers[il_kv].k,*vs=layers[il_kv].v; if(!ks||!vs)continue;
            const auto wr = pager_->write_cluster_at((int32_t)il_kv, cluster_id, positions,
                ks, vs, (size_t)ks->ne[0], (size_t)vs->ne[0]);
            written_bytes += wr.disk_size;
            written_layers++;
        }

        // Update disk tracking (no offset/size — pager owns that)
        cluster_disk_info info;
        info.positions = positions;
        if (has_existing) {
            info.persisted_set = existing->second.persisted_set;
        }
        info.persisted_token_count = positions.size();
        for (auto p : positions) info.persisted_set.insert(p);
        for (auto p : new_positions) info.persisted_set.insert(p);

        for (auto pos : positions) evicted_cells_.insert(pos);
        for (auto & m : cluster_mgrs_) m->set_resident(cluster_id, false);
        disk_clusters_[cluster_id] = info;

        // Release only pages that contain no resident KV rows.
        size_t released_bytes = 0;
        for (size_t il_kv = 0; il_kv < layers.size(); il_kv++) {
            auto *ks=layers[il_kv].k, *vs=layers[il_kv].v;
            if(!ks||!vs||!ks->data||!vs->data) continue;
            released_bytes += madvise_fully_evicted_kv_pages(ks, positions, evicted_cells_);
            released_bytes += madvise_fully_evicted_kv_pages(vs, positions, evicted_cells_);
        }
        evicted++;
        eviction_count_++;
        fprintf(stderr,
                "KV_EVICT_COMPLETE: cluster=%d tokens=%zu layers=%d bytes=%zu released_bytes=%zu reused_persisted=0 resident=%d max_resident=%d evicted_cells=%zu disk_clusters=%zu\n",
                cluster_id, positions.size(), written_layers, written_bytes, released_bytes,
                mgr.get_n_resident(), mgr.get_max_resident(), evicted_cells_.size(),
                disk_clusters_.size());
    } // end for loop over candidates
}

void llama_kv_cache_cluster::ensure_resident(const std::vector<uint32_t> & needed) {
    if (!prefill_done_) return;
    // Resolve completed prefetch handles
    for (auto & h : prefetch_handles_) {
        if (h.pending) {
            auto*ks=layers[h.il].k,*vs=layers[h.il].v;
            if(ks&&vs) pager_->resolve_read(h,ks,vs);
        }
    }
    // Collect clusters that need loading
    std::set<int32_t> to_reload;
    for (auto p : needed) {
        if (evicted_cells_.count(p)) {
            int32_t c = cluster_mgrs_[0]->get_cluster_id(p);
            if (c >= 0 && disk_clusters_.count(c)) to_reload.insert(c);
        }
    }
    // Reload clusters from disk
    for (auto cid : to_reload) {
        const size_t disk_clusters_before = disk_clusters_.size();
        auto it = disk_clusters_.find(cid);
        if (it == disk_clusters_.end()) continue;
        auto & info = it->second;
        size_t evicted_before = 0;
        for (auto p : info.positions) {
            evicted_before += evicted_cells_.count(p) ? 1 : 0;
        }
        size_t expected_bytes = 0;
        int descriptor_layers = 0;
        bool metadata_ok = info.persisted_token_count == info.positions.size();
        for (size_t il = 0; il < layers.size(); ++il) {
            const auto * desc = pager_->get_descriptor((int32_t) il, cid);
            if (!desc || desc->offset < 0 ||
                    desc->n_tokens != info.positions.size() ||
                    desc->size < 0 || desc->capacity < desc->size) {
                metadata_ok = false;
                continue;
            }
            expected_bytes += (size_t) desc->size;
            descriptor_layers++;
        }
        fprintf(stderr,
                "KV_PAGE_IN_BEGIN: cluster=%d tokens=%zu layers=%d bytes=%zu metadata_ok=%d evicted_before=%zu disk_clusters=%zu needed_positions=%zu\n",
                cid, info.positions.size(), descriptor_layers, expected_bytes,
                metadata_ok ? 1 : 0, evicted_before, disk_clusters_before, needed.size());
        if (!metadata_ok) {
            metadata_errors_++;
            fprintf(stderr,
                    "KV_CLUSTER_METADATA_ERROR: op=page_in cluster=%d tokens=%zu descriptor_layers=%d expected_layers=%zu\n",
                    cid, info.positions.size(), descriptor_layers, layers.size());
        }
        GGML_ASSERT(metadata_ok);
        // Read all layers from disk using per-cluster-ID API
        for (size_t il = 0; il < layers.size(); il++) {
            auto * ks = layers[il].k, * vs = layers[il].v;
            if (!ks || !vs) continue;
            pager_->read_cluster_by_id((int32_t)il, cid, info.positions, ks, vs,
                (size_t)ks->ne[0], (size_t)vs->ne[0]);
        }
        // Clear evicted flags
        for (auto p : info.positions) evicted_cells_.erase(p);
        // Mark cluster as resident in all layer managers
        for (auto & m : cluster_mgrs_) m->set_resident(cid, true);

        // Item 4: Merge buffered tokens for this cluster from ALL layers
        {
            std::lock_guard<std::recursive_mutex> lock(token_buffer_mutex_);
            size_t total_merged = 0;
            for (size_t il2 = 0; il2 < layers.size(); il2++) {
                auto key = make_buffer_key((int32_t)il2, cid);
                auto buf_it = token_buffer_.find(key);
                if (buf_it == token_buffer_.end() || buf_it->second.empty()) continue;
                for (auto & bt : buf_it->second) {
                    auto * ks = layers[il2].k, * vs = layers[il2].v;
                    if (!ks || !vs) continue;
                    size_t n_k = (size_t)ks->ne[0], n_v = (size_t)vs->ne[0];
                    size_t k_sz = ggml_type_size(ks->type);
                    size_t v_sz = ggml_type_size(vs->type);
                    if (ks->type == GGML_TYPE_F32) {
                        ggml_backend_tensor_set(ks, bt.k.data(), bt.cell_idx * n_k * k_sz, n_k * sizeof(float));
                    } else if (ks->type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> tmp(n_k);
                        for (size_t j = 0; j < n_k; j++) tmp[j] = ggml_fp32_to_fp16(bt.k[j]);
                        ggml_backend_tensor_set(ks, tmp.data(), bt.cell_idx * n_k * k_sz, n_k * sizeof(ggml_fp16_t));
                    }
                    if (vs->type == GGML_TYPE_F32) {
                        ggml_backend_tensor_set(vs, bt.v.data(), bt.cell_idx * n_v * v_sz, n_v * sizeof(float));
                    } else if (vs->type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> tmp(n_v);
                        for (size_t j = 0; j < n_v; j++) tmp[j] = ggml_fp32_to_fp16(bt.v[j]);
                        ggml_backend_tensor_set(vs, tmp.data(), bt.cell_idx * n_v * v_sz, n_v * sizeof(ggml_fp16_t));
                    }
                    evicted_cells_.erase(bt.cell_idx);
                    cluster_mgrs_[il2]->add_position(cid, bt.cell_idx);
                }
                total_merged += buf_it->second.size();
                token_buffer_.erase(buf_it);
            }
            GGML_UNUSED(total_merged);
        }

        size_t evicted_after = 0;
        for (auto p : info.positions) {
            evicted_after += evicted_cells_.count(p) ? 1 : 0;
        }
        fprintf(stderr,
                "KV_PAGE_IN_COMPLETE: cluster=%d tokens=%zu layers=%d bytes=%zu metadata_ok=%d evicted_after=%zu resident=%d disk_clusters_before=%zu disk_clusters_after=%zu\n",
                cid, info.positions.size(), descriptor_layers, expected_bytes,
                metadata_ok ? 1 : 0, evicted_after,
                cluster_mgrs_[0]->is_cluster_resident(cid) ? 1 : 0,
                disk_clusters_before, disk_clusters_before - 1);
        page_in_count_++;
        disk_clusters_.erase(it);
    }

    // Memory-pressure-aware prefetch (Phase 5)
    prefetch_handles_.clear();
    if (!cluster_mgrs_.empty() && cluster_mgrs_[0]->get_n_clusters() > get_cluster_k()) {
        auto & mgr = *cluster_mgrs_[0];
        int free_slots = mgr.get_max_resident() - mgr.get_n_resident();
        int prefetch_budget = std::min(free_slots, cparams_.cluster_max_prefetch);
        if (prefetch_budget > 0) {
            uint32_t last = get_last_cell_idx();
            auto wanted = mgr.get_top_cluster_positions(last, get_cluster_k() + prefetch_budget);
            auto needed_set = mgr.get_top_cluster_positions(last, get_cluster_k());
            std::set<int32_t> prefetch_candidates;
            for (auto p : wanted) {
                if (std::find(needed_set.begin(), needed_set.end(), p) == needed_set.end()) {
                    int32_t c = mgr.get_cluster_id(p);
                    if (c >= 0 && disk_clusters_.count(c) && !mgr.is_cluster_resident(c)) {
                        prefetch_candidates.insert(c);
                    }
                }
            }
            for (auto cid : prefetch_candidates) {
                auto it = disk_clusters_.find(cid);
                if (it == disk_clusters_.end()) continue;
                for (size_t il = 0; il < layers.size(); il++) {
                    const auto * desc = pager_->get_descriptor((int32_t)il, cid);
                    if (!desc || desc->offset < 0) continue;
                    prefetch_handles_.push_back(pager_->prefetch_cluster(
                        (int32_t)il, it->second.positions, desc->offset, (size_t)desc->size,
                        (size_t)layers[il].k->ne[0], (size_t)layers[il].v->ne[0]));
                }
            }
        }
    }
}

uint32_t llama_kv_cache_cluster::get_last_cell_idx() const { if(v_cells.empty())return 0; uint32_t u=v_cells[0].used_max_p1(); return u>0?u-1:0; }

void llama_kv_cache_cluster::cluster_update_from_cache() {
    if(layers.empty()||!prefill_done_)return; uint32_t ci=get_last_cell_idx();
    for(size_t il=0;il<layers.size();il++){ggml_tensor*k0=layers[il].k;if(!k0)continue;size_t n=(size_t)k0->ne[0];std::vector<float>kc(n);
        if(k0->type==GGML_TYPE_F32){ggml_backend_tensor_get(k0,kc.data(),(size_t)ci*n*sizeof(float),n*sizeof(float));}
        else if(k0->type==GGML_TYPE_F16){std::vector<ggml_fp16_t>t(n);ggml_backend_tensor_get(k0,t.data(),(size_t)ci*n*sizeof(ggml_fp16_t),n*sizeof(ggml_fp16_t));for(size_t j=0;j<n;j++)kc[j]=ggml_fp16_to_fp32(t[j]);}else continue;
        cluster_update((int32_t)il,kc.data(),ci);}
}

void llama_kv_cache_cluster::print_stats() const {
    if(cluster_mgrs_.empty()||!cluster_mgrs_[0])return; auto&m=*cluster_mgrs_[0];
    fprintf(stderr,"KV_CLUSTER_STATE: context=%p steps=%u layers=%zu clusters=%d resident=%d max_resident=%d capacity=%u used=%u used_pct=%.1f evicted_cells=%zu resident_tokens=%zu resident_pct=%.1f cluster_k=%d buffered=%zu\n",
        (void *) this,
        n_total_steps_,cluster_mgrs_.size(),m.get_n_clusters(),m.get_n_resident(),m.get_max_resident(),get_size(),get_last_cell_idx()+1,
        100.0*(get_last_cell_idx()+1)/(double)get_size(),evicted_cells_.size(),m.get_n_resident_positions(),
        (get_last_cell_idx()+1)>0?100.0*m.get_n_resident_positions()/(double)(get_last_cell_idx()+1):0.0,get_cluster_k(),token_buffer_.size());
}

// ===== TOKEN BUFFER =====

void llama_kv_cache_cluster::buffer_token(int32_t il, uint32_t cell_idx, int32_t cluster_id,
        const float * k_vec, const float * v_vec, size_t n_embd_k, size_t n_embd_v) {
    std::lock_guard<std::recursive_mutex> lock(token_buffer_mutex_);
    buffered_token bt;
    bt.cell_idx = cell_idx;
    bt.k.assign(k_vec, k_vec + n_embd_k);
    bt.v.assign(v_vec, v_vec + n_embd_v);
    auto key = make_buffer_key(il, cluster_id); // Item 4
    token_buffer_[key].push_back(std::move(bt));

    // Auto-flush if threshold exceeded
    if (token_buffer_[key].size() >= token_buffer_threshold_) {
        flush_token_buffer(il, cluster_id);
    }
}

void llama_kv_cache_cluster::flush_token_buffer(int32_t il, int32_t cluster_id) {
    std::lock_guard<std::recursive_mutex> lock(token_buffer_mutex_);
    auto key = make_buffer_key(il, cluster_id); // Item 4
    auto it = token_buffer_.find(key);
    if (it == token_buffer_.end() || it->second.empty()) return;

    auto & tokens = it->second;
    // Sort by cell_idx for sequential disk write
    std::sort(tokens.begin(), tokens.end(),
              [](const buffered_token & a, const buffered_token & b) { return a.cell_idx < b.cell_idx; });

    // Item 4: Write buffered tokens into only this layer's storage
    auto disk_it = disk_clusters_.find(cluster_id);
    auto * ks = layers[il].k, * vs = layers[il].v;
    for (auto & bt : tokens) {
        if (ks && vs) {
            size_t n_embd_k = (size_t)ks->ne[0];
            size_t n_embd_v = (size_t)vs->ne[0];
            if (ks->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(ks, bt.k.data(), bt.cell_idx * n_embd_k * sizeof(float), n_embd_k * sizeof(float));
            } else if (ks->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(n_embd_k);
                for (size_t j = 0; j < n_embd_k; j++) tmp[j] = ggml_fp32_to_fp16(bt.k[j]);
                ggml_backend_tensor_set(ks, tmp.data(), bt.cell_idx * n_embd_k * sizeof(ggml_fp16_t), n_embd_k * sizeof(ggml_fp16_t));
            }
            if (vs->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(vs, bt.v.data(), bt.cell_idx * n_embd_v * sizeof(float), n_embd_v * sizeof(float));
            } else if (vs->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(n_embd_v);
                for (size_t j = 0; j < n_embd_v; j++) tmp[j] = ggml_fp32_to_fp16(bt.v[j]);
                ggml_backend_tensor_set(vs, tmp.data(), bt.cell_idx * n_embd_v * sizeof(ggml_fp16_t), n_embd_v * sizeof(ggml_fp16_t));
            }
        }
        evicted_cells_.erase(bt.cell_idx);
    }

    // If cluster has disk data, append new tokens to disk
    if (disk_it != disk_clusters_.end()) {
        std::vector<uint32_t> new_positions;
        for (auto & bt : tokens) new_positions.push_back(bt.cell_idx);
        write_cluster_to_disk(cluster_id, new_positions, -1, 0);
        disk_it->second.persisted_token_count += tokens.size();
        for (auto p : new_positions) disk_it->second.persisted_set.insert(p);
    }

    token_buffer_.erase(it);
}

void llama_kv_cache_cluster::flush_all_buffers() {
    std::vector<std::pair<int32_t,int32_t>> ids;
    {
        std::lock_guard<std::recursive_mutex> lock(token_buffer_mutex_);
        for (auto & [key, _] : token_buffer_) {
            int32_t il = (int32_t)(key >> 32);
            int32_t cid = (int32_t)(key & 0xFFFFFFFF);
            ids.emplace_back(il, cid);
        }
    }
    for (auto [il, cid] : ids) flush_token_buffer(il, cid);
}

void llama_kv_cache_cluster::collect_and_flush_token_buffers() {
    size_t total = total_buffered_tokens();
    if (total >= token_buffer_threshold_) {
        bool found = false;
        buffer_key_t max_key = 0;
        size_t max_sz = 0;
        {
            std::lock_guard<std::recursive_mutex> lock(token_buffer_mutex_);
            for (auto & [key, tokens] : token_buffer_) {
                if (tokens.size() > max_sz) {
                    max_sz = tokens.size();
                    max_key = key;
                    found = true;
                }
            }
        }
        // Use `found` flag instead of `max_key != 0` — key (il=0,cid=0) legitimately maps to 0
        if (found && max_sz >= token_buffer_threshold_ / 2) {
            int32_t il = (int32_t)(max_key >> 32);
            int32_t cid = (int32_t)(max_key & 0xFFFFFFFF);
            flush_token_buffer(il, cid);
        }
    }
}

size_t llama_kv_cache_cluster::total_buffered_tokens() const {
    size_t total = 0;
    std::lock_guard<std::recursive_mutex> lock(token_buffer_mutex_);
    for (auto & [_, tokens] : token_buffer_) total += tokens.size();
    return total;
}

// ===== PREFILL PERSIST =====

void llama_kv_cache_cluster::persist_all_clusters() {
    if (!prefill_done_ || layers.empty()) return;

    // Outer loop over clusters, inner over layers — ensures per-layer offsets are tracked
    // correctly by the pager's cluster-ID-keyed descriptor API.
    auto & mgr0 = *cluster_mgrs_[0];
    for (int ci = 0; ci < mgr0.get_n_clusters(); ci++) {
        const auto & positions = mgr0.get_positions(ci);
        if (positions.empty()) continue;

        for (size_t il = 0; il < layers.size(); il++) {
            auto * ks = layers[il].k, * vs = layers[il].v;
            if (!ks || !vs) continue;
            pager_->write_cluster_at((int32_t)il, ci, positions, ks, vs,
                (size_t)ks->ne[0], (size_t)vs->ne[0]);
        }

        cluster_disk_info info;
        info.positions = positions;
        info.persisted_token_count = positions.size();
        for (auto p : positions) info.persisted_set.insert(p);
        disk_clusters_[ci] = info;
    }
}

// ===== SPLIT HANDLING =====

void llama_kv_cache_cluster::handle_split(int32_t old_cluster_id, int32_t new1_id, int32_t new2_id) {
    fprintf(stderr, "KV_CLUSTER_SPLIT: old_cluster=%d child_1=%d child_2=%d\n",
            old_cluster_id, new1_id, new2_id);

    // Invalidate old disk data in pager descriptors and remove from tracking
    auto old_it = disk_clusters_.find(old_cluster_id);
    if (old_it != disk_clusters_.end()) {
        for (size_t il = 0; il < layers.size(); il++) {
            pager_->invalidate_cluster((int32_t)il, old_cluster_id);
        }
        disk_clusters_.erase(old_it);
    }

    // Write both child clusters to disk (all layers, per-cluster-ID API)
    for (int32_t child_id : {new1_id, new2_id}) {
        const auto & positions = cluster_mgrs_[0]->get_positions(child_id);
        if (positions.empty()) continue;

        for (size_t il = 0; il < layers.size(); il++) {
            auto * ks = layers[il].k, * vs = layers[il].v;
            if (!ks || !vs) continue;
            pager_->write_cluster_at((int32_t)il, child_id, positions, ks, vs,
                (size_t)ks->ne[0], (size_t)vs->ne[0]);
        }

        cluster_disk_info info;
        info.positions = positions;
        info.persisted_token_count = positions.size();
        for (auto p : positions) info.persisted_set.insert(p);
        disk_clusters_[child_id] = info;
    }

    // Item 4: Also move token buffer entries from old cluster to appropriate child (all layers)
    {
        std::lock_guard<std::recursive_mutex> lock(token_buffer_mutex_);
        for (size_t il = 0; il < layers.size(); il++) {
            auto key_old = make_buffer_key((int32_t)il, old_cluster_id);
            auto buf_it = token_buffer_.find(key_old);
            if (buf_it == token_buffer_.end()) continue;
            auto tokens = std::move(buf_it->second);
            token_buffer_.erase(buf_it);
            for (auto & bt : tokens) {
                int32_t assigned = cluster_mgrs_[il]->get_cluster_id(bt.cell_idx);
                if (assigned == new1_id || assigned == new2_id) {
                    token_buffer_[make_buffer_key((int32_t)il, assigned)].push_back(std::move(bt));
                }
            }
        }
    }
}

// ===== SCORE-BASED TOKEN SELECTION =====

std::vector<uint32_t> llama_kv_cache_cluster::select_topk_tokens(const float * query, int k_tokens,
                                                                  const float * q_vector) {
    // Guard: if managers not initialized, return empty
    if (cluster_mgrs_.empty() || !cluster_mgrs_[0]) {
        return {};
    }
    // Guard: if k_tokens<=0, fall back to LRU-based cluster selection
    if (k_tokens <= 0) {
        uint32_t last = get_last_cell_idx();
        return cluster_mgrs_[0]->get_top_cluster_positions(last, get_cluster_k());
    }

    auto & mgr = *cluster_mgrs_[0];

    // Item 7: use real Q vector for scoring when available, otherwise use K proxy
    const float * score_query = q_vector ? q_vector : query;

    uint32_t last = get_last_cell_idx();
    int32_t cur_cid = mgr.get_cluster_id(last);

    // Phase 1: Score resident clusters by dot product with query
    auto cluster_scores = mgr.score_clusters(score_query, (size_t)layers[0].k->ne[0]);
    if (cluster_scores.empty()) {
        return mgr.get_resident_positions();
    }

    // Ensure current cluster is first
    std::vector<int32_t> selected_clusters;
    if (cur_cid >= 0 && mgr.is_cluster_resident(cur_cid)) {
        selected_clusters.push_back(cur_cid);
    }
    for (auto & [cid, score] : cluster_scores) {
        if ((int)selected_clusters.size() >= get_cluster_k()) break;
        if (cid == cur_cid) continue;
        selected_clusters.push_back(cid);
    }

    // Ensure selected clusters are resident
    {
        std::vector<uint32_t> needed_positions;
        for (auto cid : selected_clusters) {
            if (!mgr.is_cluster_resident(cid)) {
                const auto & pos = mgr.get_positions(cid);
                needed_positions.insert(needed_positions.end(), pos.begin(), pos.end());
            }
        }
        if (!needed_positions.empty()) ensure_resident(needed_positions);
    }

    // Phase 2: Score individual tokens within selected clusters (Item 3: parallel)
    std::vector<std::pair<uint32_t, float>> token_scores;
    auto * k_storage = layers[0].k;
    if (k_storage && ggml_backend_buffer_is_host(k_storage->buffer)) {
        const float * k_data = (const float *)k_storage->data;
        size_t k_row_bytes = ggml_type_size(k_storage->type) * (size_t)k_storage->ne[0];

        // Item 3: use parallel scoring across clusters
        token_scores = mgr.score_tokens_parallel(
            selected_clusters, score_query, (size_t)k_storage->ne[0],
            k_data, k_row_bytes, k_storage->type, k_tokens);
    } else {
        for (auto cid : selected_clusters) {
            for (auto p : mgr.get_positions(cid)) {
                token_scores.emplace_back(p, 0.0f);
            }
        }
    }

    // Select top k_tokens (already sorted by score_tokens_parallel)
    if (k_tokens > 0 && (int)token_scores.size() > k_tokens) {
        token_scores.resize(k_tokens);
    }

    std::vector<uint32_t> positions;
    for (auto & [cell_idx, _] : token_scores) {
        positions.push_back(cell_idx);
    }

    // Mark selected clusters as accessed (Item 6: tracks cache hits)
    for (auto cid : selected_clusters) mgr.mark_accessed(cid);

    return positions;
}

// ===== CLUSTER VIEW =====

ClusterView llama_kv_cache_cluster::acquire_cluster_view(int32_t cluster_id, int32_t il) {
    ClusterView view;
    view.cluster_id = cluster_id;
    view.ready = false;

    if (il < 0 || (size_t)il >= layers.size() || !cluster_mgrs_[il]) return view;

    auto & mgr = *cluster_mgrs_[il];
    view.positions = mgr.get_positions(cluster_id);
    if (view.positions.empty()) return view;

    view.n_tokens = (uint32_t)view.positions.size();
    auto * ks = layers[il].k;
    auto * vs = layers[il].v;
    if (!ks || !vs) return view;  // guard: shared/filtered layers have null tensors
    view.n_embd_k = (uint32_t)ks->ne[0];
    view.n_embd_v = (uint32_t)vs->ne[0];

    if (mgr.is_cluster_resident(cluster_id)) {
        // Check if positions are contiguous for zero-copy
        bool contiguous = true;
        for (size_t i = 1; i < view.positions.size(); i++) {
            if (view.positions[i] != view.positions[i-1] + 1) {
                contiguous = false;
                break;
            }
        }

        if (contiguous && ggml_backend_buffer_is_host(ks->buffer)) {
            // Zero-copy: point directly to storage tensor data
            size_t k_per_token = view.n_embd_k * ggml_type_size(ks->type);
            size_t v_per_token = view.n_embd_v * ggml_type_size(vs->type);

            // Need to convert to float32 for attention
            float * k_buf = new float[view.n_tokens * view.n_embd_k];
            float * v_buf = new float[view.n_tokens * view.n_embd_v];

            for (uint32_t i = 0; i < view.n_tokens; i++) {
                uint32_t pos = view.positions[i];
                if (ks->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(ks, k_buf + i * view.n_embd_k,
                        (size_t)pos * k_per_token, view.n_embd_k * sizeof(float));
                } else if (ks->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> tmp(view.n_embd_k);
                    ggml_backend_tensor_get(ks, tmp.data(), (size_t)pos * k_per_token,
                        view.n_embd_k * sizeof(ggml_fp16_t));
                    for (uint32_t j = 0; j < view.n_embd_k; j++)
                        k_buf[i * view.n_embd_k + j] = ggml_fp16_to_fp32(tmp[j]);
                }
                if (vs->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(vs, v_buf + i * view.n_embd_v,
                        (size_t)pos * v_per_token, view.n_embd_v * sizeof(float));
                } else if (vs->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> tmp(view.n_embd_v);
                    ggml_backend_tensor_get(vs, tmp.data(), (size_t)pos * v_per_token,
                        view.n_embd_v * sizeof(ggml_fp16_t));
                    for (uint32_t j = 0; j < view.n_embd_v; j++)
                        v_buf[i * view.n_embd_v + j] = ggml_fp16_to_fp32(tmp[j]);
                }
            }
            view.k_data = k_buf;
            view.v_data = v_buf;
            view.owns_data = true;
        } else {
            // Non-contiguous or non-host: compact copy
            float * k_buf = new float[view.n_tokens * view.n_embd_k];
            float * v_buf = new float[view.n_tokens * view.n_embd_v];
            size_t k_per_token = view.n_embd_k * ggml_type_size(ks->type);
            size_t v_per_token = view.n_embd_v * ggml_type_size(vs->type);

            for (uint32_t i = 0; i < view.n_tokens; i++) {
                uint32_t pos = view.positions[i];
                if (ks->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(ks, k_buf + i * view.n_embd_k,
                        (size_t)pos * k_per_token, view.n_embd_k * sizeof(float));
                } else if (ks->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> tmp(view.n_embd_k);
                    ggml_backend_tensor_get(ks, tmp.data(), (size_t)pos * k_per_token,
                        view.n_embd_k * sizeof(ggml_fp16_t));
                    for (uint32_t j = 0; j < view.n_embd_k; j++)
                        k_buf[i * view.n_embd_k + j] = ggml_fp16_to_fp32(tmp[j]);
                }
                if (vs->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(vs, v_buf + i * view.n_embd_v,
                        (size_t)pos * v_per_token, view.n_embd_v * sizeof(float));
                } else if (vs->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> tmp(view.n_embd_v);
                    ggml_backend_tensor_get(vs, tmp.data(), (size_t)pos * v_per_token,
                        view.n_embd_v * sizeof(ggml_fp16_t));
                    for (uint32_t j = 0; j < view.n_embd_v; j++)
                        v_buf[i * view.n_embd_v + j] = ggml_fp16_to_fp32(tmp[j]);
                }
            }
            view.k_data = k_buf;
            view.v_data = v_buf;
            view.owns_data = true;
        }
    } else {
        // Cluster is on disk: load into temporary buffers
        auto disk_it = disk_clusters_.find(cluster_id);
        if (disk_it == disk_clusters_.end()) return view;

        float * k_buf = new float[view.n_tokens * view.n_embd_k];
        float * v_buf = new float[view.n_tokens * view.n_embd_v];

        pager_->read_cluster_by_id(il, cluster_id, view.positions, ks, vs,
                                   view.n_embd_k, view.n_embd_v);

        // Read back from storage after pager wrote it (supports F32 and F16)
        size_t k_per_token = view.n_embd_k * ggml_type_size(ks->type);
        size_t v_per_token = view.n_embd_v * ggml_type_size(vs->type);
        for (uint32_t i = 0; i < view.n_tokens; i++) {
            uint32_t pos = view.positions[i];
            if (ks->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(ks, k_buf + i * view.n_embd_k,
                    (size_t)pos * k_per_token, view.n_embd_k * sizeof(float));
            } else if (ks->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(view.n_embd_k);
                ggml_backend_tensor_get(ks, tmp.data(), (size_t)pos * k_per_token,
                    view.n_embd_k * sizeof(ggml_fp16_t));
                for (uint32_t j = 0; j < view.n_embd_k; j++)
                    k_buf[i * view.n_embd_k + j] = ggml_fp16_to_fp32(tmp[j]);
            }
            if (vs->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(vs, v_buf + i * view.n_embd_v,
                    (size_t)pos * v_per_token, view.n_embd_v * sizeof(float));
            } else if (vs->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(view.n_embd_v);
                ggml_backend_tensor_get(vs, tmp.data(), (size_t)pos * v_per_token,
                    view.n_embd_v * sizeof(ggml_fp16_t));
                for (uint32_t j = 0; j < view.n_embd_v; j++)
                    v_buf[i * view.n_embd_v + j] = ggml_fp16_to_fp32(tmp[j]);
            }
        }

        // Merge buffered tokens for this cluster
        {
            std::lock_guard<std::recursive_mutex> lock(token_buffer_mutex_);
            auto buf_it = token_buffer_.find(make_buffer_key(il, cluster_id));
            if (buf_it != token_buffer_.end() && !buf_it->second.empty()) {
                // Extend buffers to accommodate buffered tokens
                size_t old_n = view.n_tokens;
                size_t extra = buf_it->second.size();
                float * new_k = new float[(old_n + extra) * view.n_embd_k];
                float * new_v = new float[(old_n + extra) * view.n_embd_v];
                std::memcpy(new_k, k_buf, old_n * view.n_embd_k * sizeof(float));
                std::memcpy(new_v, v_buf, old_n * view.n_embd_v * sizeof(float));
                delete[] k_buf; delete[] v_buf;

                for (size_t i = 0; i < extra; i++) {
                    auto & bt = buf_it->second[i];
                    std::memcpy(new_k + (old_n + i) * view.n_embd_k, bt.k.data(),
                                view.n_embd_k * sizeof(float));
                    std::memcpy(new_v + (old_n + i) * view.n_embd_v, bt.v.data(),
                                view.n_embd_v * sizeof(float));
                    view.positions.push_back(bt.cell_idx);
                }
                view.n_tokens += (uint32_t)extra;
                k_buf = new_k; v_buf = new_v;
                token_buffer_.erase(buf_it);
            }
        }

        view.k_data = k_buf;
        view.v_data = v_buf;
        view.owns_data = true;
    }

    view.ready = true;
    return view;
}

void llama_kv_cache_cluster::release_cluster_view(ClusterView & view) {
    if (view.owns_data) {
        delete[] view.k_data;
        delete[] view.v_data;
    }
    view.k_data = nullptr;
    view.v_data = nullptr;
    view.owns_data = false;
    view.ready = false;
}

// ===== DISK WRITE HELPER =====

void llama_kv_cache_cluster::write_cluster_to_disk(int32_t cluster_id,
        const std::vector<uint32_t> & new_positions,
        int64_t existing_offset, int64_t existing_capacity) {
    GGML_UNUSED(existing_offset); GGML_UNUSED(existing_capacity);
    // Write all layers using per-cluster-ID API (pager tracks per-layer offsets internally)
    for (size_t il = 0; il < layers.size(); il++) {
        auto * ks = layers[il].k, * vs = layers[il].v;
        if (!ks || !vs) continue;
        pager_->write_cluster_at((int32_t)il, cluster_id, new_positions, ks, vs,
            (size_t)ks->ne[0], (size_t)vs->ne[0]);
    }
}

// ===== CONTEXT WITH PRE-COPY SPARSE K/V =====

llama_kv_cache_cluster_context::llama_kv_cache_cluster_context(llama_memory_status s):llama_kv_cache_context(s),kv_cluster_(nullptr),prefill_(true){}
llama_kv_cache_cluster_context::llama_kv_cache_cluster_context(llama_kv_cache_cluster*kv):llama_kv_cache_context(kv),kv_cluster_(kv),prefill_(false){n_kv=kv->get_size();}
llama_kv_cache_cluster_context::llama_kv_cache_cluster_context(llama_kv_cache_cluster*kv,llama_context*l,bool o):llama_kv_cache_context(kv,l,false,stream_copy_info()),kv_cluster_(kv),prefill_(false){(void)o;}
llama_kv_cache_cluster_context::llama_kv_cache_cluster_context(llama_kv_cache_cluster*kv,std::vector<llama_ubatch>ub):llama_kv_cache_context(LLAMA_MEMORY_STATUS_SUCCESS),kv_cluster_(kv),prefill_(false){
    this->kv=kv;
    this->ubatches=std::move(ub);
    if(!this->ubatches.empty()){
        const auto&u=this->ubatches[0];
        // split_simple() can represent a prompt chunk as n_seq_tokens=1
        // with one sequence-set per token. Decode is the stricter condition:
        // exactly one token for every unique sequence in the ubatch.
        prefill_=(u.n_tokens != u.n_seqs_unq);
        if (!prefill_) {
            kv_cluster_->observe_decode_sequences(current_seq_ids());
        }
    }
}

bool llama_kv_cache_cluster_context::current_ubatch_is_prefill() const {
    if (ubatches.empty() || i_cur >= ubatches.size()) return true;
    const auto & u = ubatches[i_cur];
    return u.n_tokens != u.n_seqs_unq;
}

std::vector<llama_seq_id> llama_kv_cache_cluster_context::current_seq_ids() const {
    std::vector<llama_seq_id> result;
    if (ubatches.empty() || i_cur >= ubatches.size()) return result;
    const auto & u = ubatches[i_cur];
    for (uint32_t i = 0; i < u.n_tokens; ++i) {
        for (int32_t j = 0; j < u.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = u.seq_id[i][j];
            if (std::find(result.begin(), result.end(), seq_id) == result.end()) {
                result.push_back(seq_id);
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

void llama_kv_cache_cluster_context::build_cluster_indices(ggml_context *, int32_t) const {
    const auto seq_ids = current_seq_ids();
    if (!kv_cluster_ || !kv_cluster_->sparse_ready_for(seq_ids)) {
        return;
    }

    std::vector<uint32_t> needed;

    for (llama_seq_id seq_id : seq_ids) {
        const uint32_t last = kv_cluster_->get_last_cell_idx(seq_id);
        const auto selected = kv_cluster_->get_cluster_mgr(0).get_top_cluster_positions(
                last, kv_cluster_->get_cluster_k());
        for (uint32_t pos : selected) {
            if (!kv_cluster_->cell_belongs_to_seq(pos, seq_id)) {
                continue;
            }
            needed.push_back(pos);
        }

        llama_pos current_pos = std::numeric_limits<llama_pos>::max();
        if (!ubatches.empty() && i_cur < ubatches.size()) {
            const auto & u = ubatches[i_cur];
            for (uint32_t token = 0; token < u.n_tokens; ++token) {
                for (int32_t sid = 0; sid < u.n_seq_id[token]; ++sid) {
                    if (u.seq_id[token][sid] == seq_id) {
                        current_pos = std::min(current_pos, u.pos[token]);
                    }
                }
            }
        }
        const auto recent_tail = current_pos == std::numeric_limits<llama_pos>::max()
                ? std::vector<uint32_t>{}
                : kv_cluster_->get_unclustered_positions_before(seq_id, current_pos);
        needed.insert(needed.end(), recent_tail.begin(), recent_tail.end());
    }

    std::sort(needed.begin(), needed.end());
    needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
    kv_cluster_->ensure_resident(needed);
    cached_positions_ = std::move(needed);
}
void llama_kv_cache_cluster_context::ensure_positions()const{if(!cached_positions_.empty())return;build_cluster_indices(nullptr,0);}

// Framework hooks
bool llama_kv_cache_cluster_context::use_sparse_attn() const {
    return kv_cluster_ && !current_ubatch_is_prefill() &&
           kv_cluster_->sparse_ready_for(current_seq_ids());
}

uint32_t llama_kv_cache_cluster_context::get_n_kv_sparse() const {
    ensure_positions();
    if (cached_positions_.empty()) {
        return llama_kv_cache_context::get_n_kv();
    }

    // Sparse KV contains exactly the selected historical positions.
    //
    // Do not append alignment padding here. Padding with duplicate KV rows,
    // even when masked, was observed to change decode results. CPU flash
    // attention accepts arbitrary KV lengths, so the sparse view can use the
    // exact number of selected positions.
    return (uint32_t) cached_positions_.size();
}

// get_k: create sparse K input tensor, return 4D view for attention
ggml_tensor * llama_kv_cache_cluster_context::get_k(ggml_context * ctx, int32_t il) const {
    if (!use_sparse_attn()) return llama_kv_cache_context::get_k(ctx, il);
    ensure_positions(); if (cached_positions_.empty()) return llama_kv_cache_context::get_k(ctx, il);

    uint32_t n_sparse = get_n_kv_sparse();
    uint32_t n_embd_k_gqa = kv->get_n_embd_head_k(il) * kv->get_n_head_kv(il);

    // Create input tensor that will be filled during set_input phase
    auto * sparse_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_k_gqa, n_sparse);
    ggml_set_input(sparse_k);
    const_cast<llama_kv_cache_cluster_context*>(this)->sparse_k_tensors_[il] = sparse_k;

    return ggml_reshape_4d(ctx, sparse_k, kv->get_n_embd_head_k(il), kv->get_n_head_kv(il), n_sparse, 1);
}

ggml_tensor * llama_kv_cache_cluster_context::get_v(ggml_context * ctx, int32_t il) const {
    if (!use_sparse_attn()) {
        return llama_kv_cache_context::get_v(ctx, il);
    }

    ensure_positions();

    if (cached_positions_.empty()) {
        return llama_kv_cache_context::get_v(ctx, il);
    }

    const uint32_t n_sparse =
            get_n_kv_sparse();

    const uint32_t n_head =
            kv->get_n_head_kv(il);

    const uint32_t n_embd_head =
            kv->get_n_embd_head_v(il);

    const uint32_t n_embd_v_gqa =
            n_head * n_embd_head;

    //
    // Sparse V is intentionally exposed in NON-transposed form:
    //
    //     [head_dim, n_head_kv, n_sparse, 1]
    //
    // Even when the underlying llama KV cache uses transposed V storage,
    // set_input_v_idxs() gathers/reorders the selected values into this
    // canonical layout.
    //
    // This avoids carrying the original cache's special V strides into
    // the independently allocated sparse input tensor.
    //
    auto * sparse_v =
            ggml_new_tensor_2d(
                    ctx,
                    GGML_TYPE_F32,
                    n_embd_v_gqa,
                    n_sparse);

    ggml_set_input(sparse_v);

    const_cast<llama_kv_cache_cluster_context *>(this)
            ->sparse_v_tensors_[il] = sparse_v;

    return ggml_reshape_4d(
            ctx,
            sparse_v,
            n_embd_head,
            n_head,
            n_sparse,
            1);
}

uint32_t llama_kv_cache_cluster_context::get_n_kv() const {
    if (!use_sparse_attn()) return llama_kv_cache_context::get_n_kv();
    ensure_positions();
    if (cached_positions_.empty()) return llama_kv_cache_context::get_n_kv();

    // Sparse storage contains historical selected KV rows. build_attn()
    // appends every current token in this decode ubatch, including true
    // multi-sequence batches.
    const uint32_t n_cur = ubatches.empty() || i_cur >= ubatches.size()
            ? 1u : ubatches[i_cur].n_tokens;
    return get_n_kv_sparse() + n_cur;
}

// set_input: pre-copy selected rows from storage into sparse input tensors
void llama_kv_cache_cluster_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    llama_kv_cache_context::set_input_k_idxs(dst, ubatch);
    if (!use_sparse_attn() || sparse_k_tensors_.empty()) return;

    const_cast<llama_kv_cache_cluster_context*>(this)->ensure_positions();
    if (cached_positions_.empty()) return;

    // Pre-copy sparse K data from each layer's storage into its sparse input tensor
    for (auto & [il_k, sparse_tensor] : sparse_k_tensors_) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sparse_tensor->buffer));
        uint32_t n_sparse = (uint32_t)sparse_tensor->ne[1];
        uint32_t n_embd = (uint32_t)sparse_tensor->ne[0];

        // Use the correct per-layer K storage, not always layer 0
        auto * k_storage = kv->get_k_storage(il_k);
        if (!k_storage) continue;

        std::vector<float> host_buf(n_sparse * n_embd);

        for (uint32_t i = 0; i < n_sparse; i++) {
            uint32_t pos = i < (uint32_t)cached_positions_.size() ? cached_positions_[i] : cached_positions_[0];
            size_t src_offset = (size_t)pos * n_embd * ggml_type_size(k_storage->type);
            float * dst_ptr = host_buf.data() + i * n_embd;
            if (k_storage->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(k_storage, dst_ptr, src_offset, n_embd * sizeof(float));
            } else if (k_storage->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(n_embd);
                ggml_backend_tensor_get(k_storage, tmp.data(), src_offset, n_embd * sizeof(ggml_fp16_t));
                for (uint32_t j = 0; j < n_embd; j++) dst_ptr[j] = ggml_fp16_to_fp32(tmp[j]);
            }
        }
        ggml_backend_tensor_set(sparse_tensor, host_buf.data(), 0, n_sparse * n_embd * sizeof(float));
    }
}

void llama_kv_cache_cluster_context::set_input_v_idxs(
        ggml_tensor * dst,
        const llama_ubatch * ubatch) const {

    llama_kv_cache_context::set_input_v_idxs(dst, ubatch);

    if (!use_sparse_attn() ||
        sparse_v_tensors_.empty() ||
        cached_positions_.empty()) {
        return;
    }

    for (auto & [il_v, sparse_tensor] : sparse_v_tensors_) {
        GGML_ASSERT(
                ggml_backend_buffer_is_host(sparse_tensor->buffer));

        //
        // Sparse tensor always has canonical layout:
        //
        //     raw 2D: [n_embd_v_gqa, n_sparse]
        //     view:   [head_dim, n_head_kv, n_sparse, 1]
        //
        const uint32_t n_embd =
                (uint32_t) sparse_tensor->ne[0];

        const uint32_t n_sparse =
                (uint32_t) sparse_tensor->ne[1];

        auto * v_storage =
                kv->get_v_storage(il_v);

        if (!v_storage) {
            continue;
        }

        const uint32_t n_head =
                kv->get_n_head_kv(il_v);

        const uint32_t head_dim =
                kv->get_n_embd_head_v(il_v);

        GGML_ASSERT(
                n_embd == n_head * head_dim);

        std::vector<float> host_buf(
                (size_t) n_sparse * n_embd,
                0.0f);

        //
        // Normal V cache:
        //
        //     storage is token-major for the selected vector:
        //     [n_embd_v_gqa, kv_size, ...]
        //
        if (!kv->is_v_trans()) {
            const size_t type_size =
                    ggml_type_size(v_storage->type);

            for (uint32_t i = 0; i < n_sparse; ++i) {
                const uint32_t pos =
                        i < cached_positions_.size()
                        ? cached_positions_[i]
                        : cached_positions_[0];

                const size_t src_offset =
                        (size_t) pos *
                        n_embd *
                        type_size;

                float * dst_ptr =
                        host_buf.data() +
                        (size_t) i * n_embd;

                if (v_storage->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(
                            v_storage,
                            dst_ptr,
                            src_offset,
                            (size_t) n_embd * sizeof(float));

                } else if (v_storage->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> tmp(n_embd);

                    ggml_backend_tensor_get(
                            v_storage,
                            tmp.data(),
                            src_offset,
                            (size_t) n_embd *
                                    sizeof(ggml_fp16_t));

                    for (uint32_t j = 0; j < n_embd; ++j) {
                        dst_ptr[j] =
                                ggml_fp16_to_fp32(tmp[j]);
                    }
                } else {
                    GGML_ABORT(
                            "unsupported sparse V cache type");
                }
            }

        } else {
            //
            // Transposed native V cache.
            //
            // llama_kv_cache::get_v() interprets the underlying memory as:
            //
            //     [token, head, head_dim]
            //
            // with strides equivalent to:
            //
            //     src(token, head, d)
            //       = token
            //       + kv_size * (d + head_dim * head)
            //
            // Therefore a token's complete V vector is NOT contiguous.
            //
            // Read storage once, then gather selected token values into
            // canonical non-transposed sparse layout:
            //
            //     dst(i, head, d)
            //       = d + head_dim * head
            //
            const size_t kv_size =
                    kv->get_size();

            const size_t n_storage =
                    ggml_nelements(v_storage);

            if (v_storage->type == GGML_TYPE_F32) {
                std::vector<float> raw(n_storage);

                ggml_backend_tensor_get(
                        v_storage,
                        raw.data(),
                        0,
                        n_storage * sizeof(float));

                for (uint32_t i = 0; i < n_sparse; ++i) {
                    const uint32_t pos =
                            i < cached_positions_.size()
                            ? cached_positions_[i]
                            : cached_positions_[0];

                    for (uint32_t h = 0; h < n_head; ++h) {
                        for (uint32_t d = 0; d < head_dim; ++d) {
                            const size_t src_idx =
                                    (size_t) pos +
                                    kv_size *
                                    ((size_t) d +
                                     (size_t) head_dim * h);

                            const size_t dst_idx =
                                    (size_t) i * n_embd +
                                    (size_t) h * head_dim +
                                    d;

                            GGML_ASSERT(src_idx < raw.size());

                            host_buf[dst_idx] =
                                    raw[src_idx];
                        }
                    }
                }

            } else if (v_storage->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> raw(n_storage);

                ggml_backend_tensor_get(
                        v_storage,
                        raw.data(),
                        0,
                        n_storage * sizeof(ggml_fp16_t));

                for (uint32_t i = 0; i < n_sparse; ++i) {
                    const uint32_t pos =
                            i < cached_positions_.size()
                            ? cached_positions_[i]
                            : cached_positions_[0];

                    for (uint32_t h = 0; h < n_head; ++h) {
                        for (uint32_t d = 0; d < head_dim; ++d) {
                            const size_t src_idx =
                                    (size_t) pos +
                                    kv_size *
                                    ((size_t) d +
                                     (size_t) head_dim * h);

                            const size_t dst_idx =
                                    (size_t) i * n_embd +
                                    (size_t) h * head_dim +
                                    d;

                            GGML_ASSERT(src_idx < raw.size());

                            host_buf[dst_idx] =
                                    ggml_fp16_to_fp32(
                                            raw[src_idx]);
                        }
                    }
                }

            } else {
                GGML_ABORT(
                        "unsupported transposed sparse V cache type");
            }
        }

        ggml_backend_tensor_set(
                sparse_tensor,
                host_buf.data(),
                0,
                host_buf.size() * sizeof(float));
    }
}

void llama_kv_cache_cluster_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal) const {
    if (use_sparse_attn() && !sparse_k_tensors_.empty()) {
        GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

        // Sparse K/V contains the union of selected historical entries for
        // every sequence in this ubatch. build_attn() then appends all current
        // tokens. Each query row must expose only cells owned by that row's
        // sequence, mirroring llama_kv_cache::set_input_kq_mask().
        const size_t n_kv   = (size_t) dst->ne[0];
        const size_t n_hist = (size_t) get_n_kv_sparse();
        const size_t n_cur  = ubatch ? (size_t) ubatch->n_tokens : 0;
        const size_t n_rows = ggml_nelements(dst) / n_kv;

        // Layout: [historical union][all current ubatch tokens].
        GGML_ASSERT(n_kv == n_hist + n_cur);

        auto token_has_seq = [ubatch](size_t token_idx, llama_seq_id seq_id) {
            if (!ubatch || token_idx >= ubatch->n_tokens) return false;
            for (int32_t j = 0; j < ubatch->n_seq_id[token_idx]; ++j) {
                if (ubatch->seq_id[token_idx][j] == seq_id) return true;
            }
            return false;
        };

        size_t total_hist_allowed = 0;
        size_t total_current_allowed = 0;
        size_t cross_allowed = 0;

        if (dst->type == GGML_TYPE_F32) {
            auto * data = (float *) dst->data;

            std::fill_n(data, ggml_nelements(dst), -INFINITY);

            for (size_t r = 0; r < n_rows && r < n_cur; ++r) {
                float * row = data + r*n_kv;
                const llama_seq_id seq_id = ubatch->seq_id[r][0];
                for (size_t h = 0; h < n_hist && h < cached_positions_.size(); ++h) {
                    if (kv_cluster_->cell_belongs_to_seq(cached_positions_[h], seq_id)) {
                        row[h] = 0.0f;
                        total_hist_allowed++;
                    }
                }
                for (size_t c = 0; c < n_cur; ++c) {
                    const bool same_seq = token_has_seq(c, seq_id);
                    const bool causal_ok = !causal || ubatch->pos[c] <= ubatch->pos[r];
                    if (same_seq && causal_ok) {
                        row[n_hist + c] = 0.0f;
                        total_current_allowed++;
                    }
                }
            }
        } else if (dst->type == GGML_TYPE_F16) {
            auto * data = (ggml_fp16_t *) dst->data;

            const ggml_fp16_t mask_drop = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t mask_keep = ggml_fp32_to_fp16(0.0f);

            std::fill_n(data, ggml_nelements(dst), mask_drop);

            for (size_t r = 0; r < n_rows && r < n_cur; ++r) {
                ggml_fp16_t * row = data + r*n_kv;
                const llama_seq_id seq_id = ubatch->seq_id[r][0];
                for (size_t h = 0; h < n_hist && h < cached_positions_.size(); ++h) {
                    if (kv_cluster_->cell_belongs_to_seq(cached_positions_[h], seq_id)) {
                        row[h] = mask_keep;
                        total_hist_allowed++;
                    }
                }
                for (size_t c = 0; c < n_cur; ++c) {
                    const bool same_seq = token_has_seq(c, seq_id);
                    const bool causal_ok = !causal || ubatch->pos[c] <= ubatch->pos[r];
                    if (same_seq && causal_ok) {
                        row[n_hist + c] = mask_keep;
                        total_current_allowed++;
                    }
                }
            }
        } else {
            GGML_ABORT("unsupported sparse KQ mask type");
        }

        // Re-audit every visible historical cell against its row sequence.
        // This is deliberately independent of the construction loops above.
        for (size_t r = 0; r < n_rows && r < n_cur; ++r) {
            const llama_seq_id seq_id = ubatch->seq_id[r][0];
            for (size_t h = 0; h < n_hist && h < cached_positions_.size(); ++h) {
                bool visible = false;
                if (dst->type == GGML_TYPE_F32) {
                    visible = ((float *) dst->data)[r*n_kv + h] == 0.0f;
                } else {
                    visible = ggml_fp16_to_fp32(((ggml_fp16_t *) dst->data)[r*n_kv + h]) == 0.0f;
                }
                if (visible && !kv_cluster_->cell_belongs_to_seq(cached_positions_[h], seq_id)) {
                    cross_allowed++;
                }
            }
        }

        if (cross_allowed > 0) {
            kv_cluster_->record_metadata_consistency_errors(cross_allowed);
            fprintf(stderr,
                    "KV_CLUSTER_METADATA_ERROR: op=sparse_mask context=%p rows=%zu seqs=%zu historical_union=%zu current_tokens=%zu hist_allowed=%zu current_allowed=%zu cross_allowed=%zu\n",
                    (void *) kv_cluster_,
                    std::min(n_rows, n_cur),
                    current_seq_ids().size(),
                    n_hist,
                    n_cur,
                    total_hist_allowed,
                    total_current_allowed,
                    cross_allowed);
        }

        GGML_ASSERT(cross_allowed == 0);

        return;
    }

    llama_kv_cache_context::set_input_kq_mask(dst, ubatch, causal);
}

void llama_kv_cache_cluster_context::init_from_sinfos(slot_info_vec_t s, int32_t n) {
    this->sinfos=std::move(s);
    this->n_kv=n;
}
