#include "llama-kv-pager.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "llama-impl.h"

static std::string get_layer_path(const std::string & base_path, int32_t il) {
    return base_path + "/layer_" + std::to_string(il) + ".bin";
}

static void ensure_directory_tree(const std::string & path) {
    if (path.empty()) {
        GGML_ABORT("KVPager: empty pager directory path");
    }

    size_t pos = path.front() == '/' ? 1 : 0;

    while (true) {
        pos = path.find('/', pos);

        const std::string part =
                pos == std::string::npos
                    ? path
                    : path.substr(0, pos);

        if (!part.empty() && part != "/") {
            errno = 0;

            if (mkdir(part.c_str(), 0755) != 0 &&
                    errno != EEXIST) {
                GGML_ABORT(
                        "KVPager: mkdir(%s) failed: errno=%d (%s)",
                        part.c_str(),
                        errno,
                        strerror(errno));
            }

            struct stat st = {};
            if (stat(part.c_str(), &st) != 0) {
                GGML_ABORT(
                        "KVPager: stat(%s) failed after mkdir: errno=%d (%s)",
                        part.c_str(),
                        errno,
                        strerror(errno));
            }

            if (!S_ISDIR(st.st_mode)) {
                GGML_ABORT(
                        "KVPager: path exists but is not a directory: %s",
                        part.c_str());
            }
        }

        if (pos == std::string::npos) {
            break;
        }

        ++pos;
    }

    errno = 0;
    if (access(path.c_str(), W_OK | X_OK) != 0) {
        GGML_ABORT(
                "KVPager: pager directory is not writable: %s, "
                "errno=%d (%s)",
                path.c_str(),
                errno,
                strerror(errno));
    }
}

static int get_fd(const std::string & path, bool create) {
    int flags = O_RDWR;

    if (create) {
        flags |= O_CREAT;
    }

    errno = 0;
    const int fd = open(path.c_str(), flags, 0644);

    if (fd < 0) {
        GGML_ABORT(
                "KVPager: open(%s) failed: errno=%d (%s)",
                path.c_str(),
                errno,
                strerror(errno));
    }

    return fd;
}

static void pread_exact_or_abort(
        int fd,
        void * data,
        size_t size,
        off_t offset,
        const char * what) {
    size_t done = 0;
    auto * dst = static_cast<uint8_t *>(data);

    while (done < size) {
        errno = 0;

        const ssize_t n = pread(
                fd,
                dst + done,
                size - done,
                offset + static_cast<off_t>(done));

        if (n > 0) {
            done += static_cast<size_t>(n);
            continue;
        }

        if (n < 0 && errno == EINTR) {
            continue;
        }

        if (n == 0) {
            GGML_ABORT(
                    "KVPager: unexpected EOF while reading %s: "
                    "offset=%lld requested=%zu completed=%zu",
                    what,
                    static_cast<long long>(offset),
                    size,
                    done);
        }

        GGML_ABORT(
                "KVPager: pread failed for %s: "
                "offset=%lld requested=%zu completed=%zu "
                "errno=%d (%s)",
                what,
                static_cast<long long>(offset),
                size,
                done,
                errno,
                strerror(errno));
    }
}

static void pwrite_exact_or_abort(
        int fd,
        const void * data,
        size_t size,
        off_t offset,
        const char * what) {
    size_t done = 0;
    const auto * src = static_cast<const uint8_t *>(data);

    while (done < size) {
        errno = 0;

        const ssize_t n = pwrite(
                fd,
                src + done,
                size - done,
                offset + static_cast<off_t>(done));

        if (n > 0) {
            done += static_cast<size_t>(n);
            continue;
        }

        if (n < 0 && errno == EINTR) {
            continue;
        }

        if (n == 0) {
            GGML_ABORT(
                    "KVPager: zero-length write for %s: "
                    "offset=%lld requested=%zu completed=%zu",
                    what,
                    static_cast<long long>(offset),
                    size,
                    done);
        }

        GGML_ABORT(
                "KVPager: pwrite failed for %s: "
                "offset=%lld requested=%zu completed=%zu "
                "errno=%d (%s)",
                what,
                static_cast<long long>(offset),
                size,
                done,
                errno,
                strerror(errno));
    }
}

llama_kv_pager::llama_kv_pager(
        const std::string & base_path,
        int n_layer)
    : base_path_(base_path),
      n_layer_(n_layer) {
    ensure_directory_tree(base_path_);

    LLAMA_LOG_INFO(
            "%s: initialized pager directory %s for %d layers\n",
            __func__,
            base_path_.c_str(),
            n_layer_);

    fds_.resize(n_layer_, -1);
    file_sizes_.resize(n_layer_, 0);
    headers_.resize(n_layer_);
    descriptors_.resize(n_layer_);

    for (int i = 0; i < n_layer_; ++i) {
        layer_mutexes_.emplace_back();
    }
}

llama_kv_pager::~llama_kv_pager() {
    while (pending_count_.load() > 0) {
        usleep(1000);
    }

    fprintf(stderr,
            "KV_PAGER_SUMMARY: pager=%p read_calls=%llu read_bytes=%llu "
            "write_calls=%llu write_bytes=%llu metadata_errors=%llu pending_ops=%zu\n",
            (void *) this,
            (unsigned long long) read_calls_.load(),
            (unsigned long long) read_bytes_.load(),
            (unsigned long long) write_calls_.load(),
            (unsigned long long) write_bytes_.load(),
            (unsigned long long) metadata_errors_.load(),
            pending_count_.load());

    int cleanup_failures = 0;

    for (size_t i = 0; i < fds_.size(); ++i) {
        if (fds_[i] < 0) {
            continue;
        }

        errno = 0;
        if (close(fds_[i]) != 0) {
            LLAMA_LOG_WARN(
                    "%s: close(layer=%zu) failed: errno=%d (%s)\n",
                    __func__,
                    i,
                    errno,
                    strerror(errno));
        }

        fds_[i] = -1;
    }

    int files_remaining = 0;
    for (int il = 0; il < n_layer_; ++il) {
        const std::string path =
                get_layer_path(base_path_, il);

        errno = 0;
        if (unlink(path.c_str()) != 0 &&
                errno != ENOENT) {
            cleanup_failures++;
            LLAMA_LOG_WARN(
                    "%s: unlink(%s) failed: errno=%d (%s)\n",
                    __func__,
                    path.c_str(),
                    errno,
                    strerror(errno));
        }
        if (access(path.c_str(), F_OK) == 0) {
            files_remaining++;
        }
    }

    // Remove only this Context's directory. Never remove
    // LLAMA_KV_PAGER_DIR itself because it can be shared by other
    // models / Contexts / processes.
    errno = 0;
    if (rmdir(base_path_.c_str()) != 0 &&
            errno != ENOENT) {
        cleanup_failures++;
        LLAMA_LOG_WARN(
                "%s: rmdir(%s) failed: errno=%d (%s)\n",
                __func__,
                base_path_.c_str(),
                errno,
                strerror(errno));
    } else {
        LLAMA_LOG_INFO(
                "%s: cleaned pager directory %s\n",
                __func__,
                base_path_.c_str());
    }

    fprintf(stderr,
            "KV_PAGER_CLEANUP: pager=%p cleanup_failures=%d files_remaining=%d\n",
            (void *) this, cleanup_failures, files_remaining);
}

// ===== File Format Helpers =====

void llama_kv_pager::init_or_load_file(int32_t il, bool create) {
    if (fds_[il] >= 0) return; // fast path: already initialized
    std::lock_guard<std::mutex> lock(layer_mutexes_[il]);
    if (fds_[il] >= 0) return; // double-check after lock: another thread may have initialized

    std::string path = get_layer_path(base_path_, il);
    bool exists = (access(path.c_str(), F_OK) == 0);

    if (!exists || create) {
        fds_[il] = get_fd(path, true);
        // Write initial header
        headers_[il] = pager_file_header{};
        write_header(il);
        // Pre-allocate descriptor space for up to 256 clusters
        descriptors_[il].resize(256);
        write_descriptors(il);
    } else {
        fds_[il] = get_fd(path, false);
        // Read header
        pread_exact_or_abort(
                fds_[il],
                &headers_[il],
                sizeof(pager_file_header),
                0,
                "pager header");
        if (headers_[il].magic != 0x4B56434C) {
            // Old/missing format: reinitialize
            headers_[il] = pager_file_header{};
            write_header(il);
            descriptors_[il].resize(256);
            write_descriptors(il);
            return;
        }
        // Read descriptors
        uint32_t n = headers_[il].descriptor_count;
        if (n > 0) {
            descriptors_[il].resize(std::max(n, 256u));
            pread_exact_or_abort(
                    fds_[il],
                    descriptors_[il].data(),
                    n * PAGER_DESC_SIZE,
                    PAGER_HEADER_SIZE,
                    "pager descriptors");
        } else {
            descriptors_[il].resize(256);
        }
    }
}

void llama_kv_pager::write_header(int32_t il) {
    pwrite_exact_or_abort(
            fds_[il],
            &headers_[il],
            sizeof(pager_file_header),
            0,
            "pager header");
}

void llama_kv_pager::write_descriptors(int32_t il) {
    uint32_t n = headers_[il].descriptor_count;
    if (n == 0) return;
    pwrite_exact_or_abort(
            fds_[il],
            descriptors_[il].data(),
            n * PAGER_DESC_SIZE,
            PAGER_HEADER_SIZE,
            "pager descriptors");
}

int32_t llama_kv_pager::find_or_add_descriptor_slot(int32_t il, int32_t cluster_id) {
    auto & descs = descriptors_[il];
    // Search for existing
    for (size_t i = 0; i < descs.size(); i++) {
        if (descs[i].cluster_id == cluster_id && (descs[i].flags & PAGER_DESC_FLAG_VALID))
            return (int32_t)i;
    }
    // Search for free slot
    for (size_t i = 0; i < descs.size(); i++) {
        if (!(descs[i].flags & PAGER_DESC_FLAG_VALID) && descs[i].cluster_id == -1) {
            descs[i].cluster_id = cluster_id;
            descs[i].flags = PAGER_DESC_FLAG_VALID;
            headers_[il].descriptor_count = std::max(headers_[il].descriptor_count, (uint32_t)(i + 1));
            return (int32_t)i;
        }
    }
    // Expand
    size_t idx = descs.size();
    descs.resize(idx * 2);
    descs[idx].cluster_id = cluster_id;
    descs[idx].flags = PAGER_DESC_FLAG_VALID;
    headers_[il].descriptor_count = (uint32_t)(idx + 1);
    write_header(il);
    return (int32_t)idx;
}

int64_t llama_kv_pager::allocate_space(int32_t il, int32_t cluster_id, size_t required) {
    auto & hdr = headers_[il];
    int64_t offset = (int64_t)hdr.data_section_end;
    hdr.data_section_end += required;
    write_header(il);
    return offset;
}

void llama_kv_pager::mark_descriptor_free(int32_t il, int32_t cluster_id) {
    for (auto & d : descriptors_[il]) {
        if (d.cluster_id == cluster_id && (d.flags & PAGER_DESC_FLAG_VALID)) {
            d.flags = 0;
            d.cluster_id = -1;
            d.offset = -1;
            d.size = 0;
            d.capacity = 0;
            d.n_tokens = 0;
            headers_[il].free_count++;
            write_descriptors(il);
            return;
        }
    }
}

size_t llama_kv_pager::per_token_size(int32_t, ggml_tensor * k_storage, ggml_tensor * v_storage) const {
    return ggml_type_size(k_storage->type) * k_storage->ne[0] +
           ggml_type_size(v_storage->type) * v_storage->ne[0];
}

// ===== Legacy Sync API (kept for backward compat) =====

llama_kv_pager::write_result llama_kv_pager::write_cluster(
        int32_t il, const std::vector<uint32_t> & positions,
        ggml_tensor * k_storage, ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa) {
    init_or_load_file(il, true);

    size_t n_tokens = positions.size();
    size_t k_per_token = n_embd_k_gqa * ggml_type_size(k_storage->type);
    size_t v_per_token = n_embd_v_gqa * ggml_type_size(v_storage->type);
    size_t per_token = k_per_token + v_per_token;
    size_t total_size = n_tokens * per_token;
    if (buf_.size() < total_size) buf_.resize(total_size);

    for (size_t i = 0; i < n_tokens; i++) {
        size_t cell_idx = positions[i];
        ggml_backend_tensor_get(k_storage, buf_.data() + i * per_token,
            (size_t)cell_idx * k_per_token, k_per_token);
    }
    for (size_t i = 0; i < n_tokens; i++) {
        size_t cell_idx = positions[i];
        ggml_backend_tensor_get(v_storage, buf_.data() + i * per_token + k_per_token,
            (size_t)cell_idx * v_per_token, v_per_token);
    }

    int64_t offset = allocate_space(il, -1, total_size);
    pwrite_exact_or_abort(
            fds_[il],
            buf_.data(),
            total_size,
            offset,
            "legacy cluster data");
    write_calls_.fetch_add(1);
    write_bytes_.fetch_add(total_size);

    write_result result; result.disk_offset = offset; result.disk_size = total_size;
    return result;
}

void llama_kv_pager::read_cluster(int32_t il, const std::vector<uint32_t> & positions,
        ggml_tensor * k_storage, ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa, int64_t disk_offset, size_t disk_size) {
    init_or_load_file(il, false);

    size_t n_tokens = positions.size();
    size_t k_per_token = n_embd_k_gqa * ggml_type_size(k_storage->type);
    size_t v_per_token = n_embd_v_gqa * ggml_type_size(v_storage->type);
    size_t per_token = k_per_token + v_per_token;
    if (buf_.size() < disk_size) buf_.resize(disk_size);

    pread_exact_or_abort(
            fds_[il],
            buf_.data(),
            disk_size,
            disk_offset,
            "legacy cluster data");
    read_calls_.fetch_add(1);
    read_bytes_.fetch_add(disk_size);

    for (size_t i = 0; i < n_tokens; i++) {
        size_t cell_idx = positions[i];
        ggml_backend_tensor_set(k_storage, buf_.data() + i * per_token, (size_t)cell_idx * k_per_token, k_per_token);
        ggml_backend_tensor_set(v_storage, buf_.data() + i * per_token + k_per_token, (size_t)cell_idx * v_per_token, v_per_token);
    }
}

// ===== Per-Cluster API (new format) =====

llama_kv_pager::write_result llama_kv_pager::write_cluster_at(
        int32_t il, int32_t cluster_id,
        const std::vector<uint32_t> & positions,
        ggml_tensor * k_storage, ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa) {
    init_or_load_file(il, true);

    size_t n_tokens = positions.size();
    size_t k_per_token = n_embd_k_gqa * ggml_type_size(k_storage->type);
    size_t v_per_token = n_embd_v_gqa * ggml_type_size(v_storage->type);
    size_t per_token = k_per_token + v_per_token;
    size_t total_size = n_tokens * per_token;
    if (buf_.size() < total_size) buf_.resize(total_size);

    // Serialize data
    for (size_t i = 0; i < n_tokens; i++) {
        size_t cell_idx = positions[i];
        ggml_backend_tensor_get(k_storage, buf_.data() + i * per_token,
            (size_t)cell_idx * k_per_token, k_per_token);
    }
    for (size_t i = 0; i < n_tokens; i++) {
        size_t cell_idx = positions[i];
        ggml_backend_tensor_get(v_storage, buf_.data() + i * per_token + k_per_token,
            (size_t)cell_idx * v_per_token, v_per_token);
    }

    // Find or create descriptor slot
    int32_t slot = find_or_add_descriptor_slot(il, cluster_id);
    auto & desc = descriptors_[il][slot];

    // Check if relocation needed
    bool need_relocate = (desc.capacity < (int64_t)total_size);
    int64_t write_offset;

    if (need_relocate) {
        // Allocate new space at EOF
        write_offset = allocate_space(il, cluster_id, total_size);
        // Mark old space as free if any
        if (desc.offset >= 0) {
            headers_[il].free_count++;
        }
        desc.capacity = (int64_t)total_size;
    } else {
        write_offset = desc.offset;
    }

    // Write data
    pwrite_exact_or_abort(
            fds_[il],
            buf_.data(),
            total_size,
            write_offset,
            "cluster data");

    // Update descriptor
    desc.offset = write_offset;
    desc.size = (int64_t)total_size;
    desc.n_tokens = (uint32_t)n_tokens;
    desc.flags |= PAGER_DESC_FLAG_VALID;
    write_descriptors(il);

    const bool metadata_ok =
            desc.n_tokens == n_tokens && desc.size == (int64_t) total_size &&
            desc.capacity >= desc.size && desc.offset == write_offset;
    write_calls_.fetch_add(1);
    write_bytes_.fetch_add(total_size);
    if (!metadata_ok) {
        metadata_errors_.fetch_add(1);
        fprintf(stderr,
                "KV_PAGER_METADATA_ERROR: op=write layer=%d cluster=%d tokens=%zu bytes=%zu offset=%lld desc_tokens=%u desc_size=%lld capacity=%lld\n",
                il, cluster_id, n_tokens, total_size, (long long) write_offset,
                desc.n_tokens, (long long) desc.size, (long long) desc.capacity);
    }
    GGML_ASSERT(metadata_ok);
    write_result result; result.disk_offset = write_offset; result.disk_size = total_size;
    return result;
}

void llama_kv_pager::read_cluster_by_id(int32_t il, int32_t cluster_id,
        const std::vector<uint32_t> & positions,
        ggml_tensor * k_storage, ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa) {
    init_or_load_file(il, false);

    // Find descriptor
    const auto * desc = get_descriptor(il, cluster_id);
    if (!desc || desc->offset < 0) {
        metadata_errors_.fetch_add(1);
        fprintf(stderr, "KV_PAGER_METADATA_ERROR: op=read layer=%d cluster=%d reason=descriptor_missing\n", il, cluster_id);
        return;
    }

    size_t n_tokens = positions.size();
    size_t k_per_token = n_embd_k_gqa * ggml_type_size(k_storage->type);
    size_t v_per_token = n_embd_v_gqa * ggml_type_size(v_storage->type);
    size_t per_token = k_per_token + v_per_token;
    size_t disk_size = n_tokens * per_token;
    const bool metadata_ok =
            desc->n_tokens == n_tokens && desc->size == (int64_t) disk_size &&
            desc->capacity >= desc->size;
    if (!metadata_ok) {
        metadata_errors_.fetch_add(1);
        fprintf(stderr,
                "KV_PAGER_METADATA_ERROR: op=read layer=%d cluster=%d tokens=%zu bytes=%zu offset=%lld desc_tokens=%u desc_size=%lld capacity=%lld\n",
                il, cluster_id, n_tokens, disk_size, (long long) desc->offset,
                desc->n_tokens, (long long) desc->size, (long long) desc->capacity);
    }
    GGML_ASSERT(metadata_ok);
    if (buf_.size() < disk_size) buf_.resize(disk_size);

    pread_exact_or_abort(
            fds_[il],
            buf_.data(),
            disk_size,
            desc->offset,
            "cluster data");
    read_calls_.fetch_add(1);
    read_bytes_.fetch_add(disk_size);

    for (size_t i = 0; i < n_tokens; i++) {
        size_t cell_idx = positions[i];
        ggml_backend_tensor_set(k_storage, buf_.data() + i * per_token, (size_t)cell_idx * k_per_token, k_per_token);
        ggml_backend_tensor_set(v_storage, buf_.data() + i * per_token + k_per_token, (size_t)cell_idx * v_per_token, v_per_token);
    }
}

void llama_kv_pager::invalidate_cluster(int32_t il, int32_t cluster_id) {
    init_or_load_file(il, false);
    std::lock_guard<std::mutex> lock(layer_mutexes_[il]);
    mark_descriptor_free(il, cluster_id);
}

const pager_cluster_descriptor * llama_kv_pager::get_descriptor(int32_t il, int32_t cluster_id) const {
    for (const auto & d : descriptors_[il]) {
        if (d.cluster_id == cluster_id && (d.flags & PAGER_DESC_FLAG_VALID))
            return &d;
    }
    return nullptr;
}

// ===== Async API =====

std::vector<uint8_t> llama_kv_pager::do_disk_read(int32_t il, int64_t offset, size_t size) {
    // The fd must be initialized on the caller thread by init_or_load_file().
    // Never lazily mutate fds_ from concurrent prefetch worker threads.
    GGML_ASSERT(il >= 0 && il < n_layer_);
    GGML_ASSERT(fds_[il] >= 0);

    std::vector<uint8_t> data(size);
    pread_exact_or_abort(
            fds_[il],
            data.data(),
            size,
            offset,
            "async cluster data");
    read_calls_.fetch_add(1);
    read_bytes_.fetch_add(size);
    pending_count_--;
    return data;
}

std::future<llama_kv_pager::write_result> llama_kv_pager::async_write_cluster(
        int32_t il, const std::vector<uint32_t> & positions,
        ggml_tensor * k_storage, ggml_tensor * v_storage,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa) {
    init_or_load_file(il, true);

    size_t n_tokens = positions.size();
    size_t k_per_token = n_embd_k_gqa * ggml_type_size(k_storage->type);
    size_t v_per_token = n_embd_v_gqa * ggml_type_size(v_storage->type);
    size_t per_token = k_per_token + v_per_token;
    size_t total_size = n_tokens * per_token;

    auto buf = std::make_shared<std::vector<uint8_t>>(total_size);
    for (size_t i = 0; i < n_tokens; i++) {
        size_t cell_idx = positions[i];
        ggml_backend_tensor_get(k_storage, buf->data() + i * per_token,
            (size_t)cell_idx * k_per_token, k_per_token);
    }
    for (size_t i = 0; i < n_tokens; i++) {
        size_t cell_idx = positions[i];
        ggml_backend_tensor_get(v_storage, buf->data() + i * per_token + k_per_token,
            (size_t)cell_idx * v_per_token, v_per_token);
    }

    int64_t offset = allocate_space(il, -1, total_size);
    int fd = fds_[il];

    pending_count_++;
    return std::async(std::launch::async, [buf, fd, offset, total_size, this]() {
        pwrite_exact_or_abort(
                fd,
                buf->data(),
                total_size,
                offset,
                "async cluster data");
        write_calls_.fetch_add(1);
        write_bytes_.fetch_add(total_size);
        pending_count_--;
        write_result r; r.disk_offset = offset; r.disk_size = total_size;
        return r;
    });
}

llama_kv_pager::read_handle llama_kv_pager::async_read_cluster(
        int32_t il, const std::vector<uint32_t> & positions,
        int64_t disk_offset, size_t disk_size,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa) {
    // Initialize/open the layer file under layer_mutexes_ before spawning
    // concurrent prefetch workers. Workers then only read the stable fd.
    init_or_load_file(il, false);

    read_handle h;
    h.il = il; h.positions = positions; h.disk_offset = disk_offset; h.disk_size = disk_size;
    h.n_embd_k_gqa = n_embd_k_gqa; h.n_embd_v_gqa = n_embd_v_gqa;
    pending_count_++;
    h.future = std::async(std::launch::async, [this, il, disk_offset, disk_size]() {
        return do_disk_read(il, disk_offset, disk_size);
    });
    h.pending = true;
    return h;
}

bool llama_kv_pager::resolve_read(read_handle & h,
        ggml_tensor * k_storage, ggml_tensor * v_storage) {
    if (!h.pending) return true;
    if (!h.ready) {
        if (h.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        h.ready = true;
    }
    auto data = h.future.get();
    size_t n_tokens = h.positions.size();
    size_t k_per_token = h.n_embd_k_gqa * ggml_type_size(k_storage->type);
    size_t v_per_token = h.n_embd_v_gqa * ggml_type_size(v_storage->type);
    size_t per_token = k_per_token + v_per_token;
    for (size_t i = 0; i < n_tokens; i++) {
        size_t cell_idx = h.positions[i];
        ggml_backend_tensor_set(k_storage, data.data() + i * per_token, (size_t)cell_idx * k_per_token, k_per_token);
        ggml_backend_tensor_set(v_storage, data.data() + i * per_token + k_per_token, (size_t)cell_idx * v_per_token, v_per_token);
    }
    h.pending = false;
    return true;
}

llama_kv_pager::read_handle llama_kv_pager::prefetch_cluster(
        int32_t il, const std::vector<uint32_t> & positions,
        int64_t disk_offset, size_t disk_size,
        size_t n_embd_k_gqa, size_t n_embd_v_gqa) {
    return async_read_cluster(il, positions, disk_offset, disk_size, n_embd_k_gqa, n_embd_v_gqa);
}

void llama_kv_pager::sync() {
    while (pending_count_.load() > 0) { usleep(1000); }
    for (int il = 0; il < n_layer_; il++) { if (fds_[il] >= 0) fsync(fds_[il]); }
}
