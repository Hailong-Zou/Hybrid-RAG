#pragma once

#include "llama.h"
#include <atomic>
#include <cstdint>
#include <future>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

// Per-cluster disk descriptor. Each cluster gets one contiguous region.
// Layout is exactly PAGER_DESC_SIZE (64) bytes — verified by static_assert below.
struct pager_cluster_descriptor {
    int32_t  cluster_id;       // -1 = free slot
    int32_t  _pad0;            // explicit padding to align next int64_t
    int64_t  offset;           // -1 = invalid/free
    int64_t  size;             // actual data size
    int64_t  capacity;         // allocated size (>= size)
    uint32_t n_tokens;
    uint32_t flags;            // bit0=valid, bit1=needs_relocate
    uint8_t  _reserved[24];    // padding to reach 64 bytes total

    pager_cluster_descriptor() : cluster_id(-1), _pad0(0), offset(-1), size(0), capacity(0), n_tokens(0), flags(0) {
        __builtin_memset(_reserved, 0, sizeof(_reserved));
    }
};

// File header for per-layer KV cache files.
struct pager_file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t descriptor_count;
    uint32_t free_count;
    uint64_t data_section_end;
    uint64_t reserved[4];

    // data_section_end starts after the fixed descriptor table:
    // PAGER_HEADER_SIZE(4096) + 256 slots * PAGER_DESC_SIZE(64) = 20480
    pager_file_header() : magic(0x4B56434C), version(2), descriptor_count(0), free_count(0), data_section_end(20480) {
        reserved[0] = reserved[1] = reserved[2] = reserved[3] = 0;
    }
};
static constexpr uint64_t PAGER_HEADER_SIZE = 4096;
static constexpr uint64_t PAGER_DESC_SIZE = 64;
static_assert(sizeof(pager_cluster_descriptor) == PAGER_DESC_SIZE,
              "pager_cluster_descriptor must be exactly PAGER_DESC_SIZE bytes");
static constexpr uint32_t PAGER_DESC_FLAG_VALID = 1;
static constexpr uint32_t PAGER_DESC_FLAG_RELOCATE = 2;

// Async KVPager: hierarchical KV cache offloading with async I/O + prefetch
//
// File format per layer (layer_N.bin):
//   [0..4095]:             File header (pager_file_header)
//   [4096..4096+N*64]:     Descriptor array (pager_cluster_descriptor)
//   [4096+N*64..]:         Data section (per-cluster contiguous blocks)
//
// Hot  = resident in CPU memory (K/V tensor cells)
// Cold = on disk; async reload on demand, async eviction in background
class llama_kv_pager {
public:
    llama_kv_pager(const std::string & base_path, int n_layer);
    ~llama_kv_pager();

    struct write_result { int64_t disk_offset; size_t disk_size; };

    // ---- Legacy Sync API (compatibility) ----
    write_result write_cluster(int32_t il, const std::vector<uint32_t> & positions,
        struct ggml_tensor * k_storage, struct ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa);
    void read_cluster(int32_t il, const std::vector<uint32_t> & positions,
        struct ggml_tensor * k_storage, struct ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa, int64_t disk_offset, size_t disk_size);

    // ---- Per-Cluster API (new format) ----
    // Write cluster data at its allocated descriptor slot. Relocates if needed.
    write_result write_cluster_at(int32_t il, int32_t cluster_id,
        const std::vector<uint32_t> & positions,
        struct ggml_tensor * k_storage, struct ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa);

    // Read cluster by cluster_id (looks up descriptor)
    void read_cluster_by_id(int32_t il, int32_t cluster_id,
        const std::vector<uint32_t> & positions,
        struct ggml_tensor * k_storage, struct ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa);

    // Mark a cluster descriptor as invalid (for split/delete)
    void invalidate_cluster(int32_t il, int32_t cluster_id);

    // Get a cluster's descriptor
    const pager_cluster_descriptor * get_descriptor(int32_t il, int32_t cluster_id) const;

    // ---- Async Write (fire-and-forget) ----
    std::future<write_result> async_write_cluster(int32_t il,
        const std::vector<uint32_t> & positions,
        struct ggml_tensor * k_storage, struct ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa);

    // ---- Async Read (with resolve) ----
    struct read_handle {
        int32_t il; std::vector<uint32_t> positions;
        int64_t disk_offset; size_t disk_size;
        size_t n_embd_k_gqa, n_embd_v_gqa;
        std::future<std::vector<uint8_t>> future;
        bool pending = false;
        bool ready = false;
    };

    read_handle async_read_cluster(int32_t il, const std::vector<uint32_t> & positions,
        int64_t disk_offset, size_t disk_size, size_t n_embd_k_gqa, size_t n_embd_v_gqa);
    bool resolve_read(read_handle & h, struct ggml_tensor * k_storage, struct ggml_tensor * v_storage);

    // ---- Prefetch ----
    read_handle prefetch_cluster(int32_t il, const std::vector<uint32_t> & positions,
        int64_t disk_offset, size_t disk_size, size_t n_embd_k_gqa, size_t n_embd_v_gqa);

    size_t pending_ops() const { return pending_count_.load(); }
    void sync();

private:
    std::string base_path_;
    int n_layer_;
    std::vector<int> fds_;
    std::vector<size_t> file_sizes_;   // deprecated: kept for legacy API
    std::vector<uint8_t> buf_;
    size_t buf_size_;
    mutable std::atomic<size_t> pending_count_{0};
    mutable std::atomic<uint64_t> read_calls_{0};
    mutable std::atomic<uint64_t> read_bytes_{0};
    mutable std::atomic<uint64_t> write_calls_{0};
    mutable std::atomic<uint64_t> write_bytes_{0};
    mutable std::atomic<uint64_t> metadata_errors_{0};

    // New format: per-layer metadata
    std::vector<pager_file_header> headers_;
    std::vector<std::vector<pager_cluster_descriptor>> descriptors_; // indexed by cluster_id
    std::deque<std::mutex> layer_mutexes_;

    std::vector<uint8_t> do_disk_read(int32_t il, int64_t offset, size_t size);
    void init_or_load_file(int32_t il, bool create);
    void write_header(int32_t il);
    void write_descriptors(int32_t il);
    int32_t find_or_add_descriptor_slot(int32_t il, int32_t cluster_id);
    int64_t allocate_space(int32_t il, int32_t cluster_id, size_t required);
    void mark_descriptor_free(int32_t il, int32_t cluster_id);
    size_t per_token_size(int32_t il, struct ggml_tensor * k_storage, struct ggml_tensor * v_storage) const;
};
