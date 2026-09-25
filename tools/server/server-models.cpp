#include "server-common.h"
#include "http.h"
#include "server-models.h"
#include "server-context.h"
#include "server-stream.h"
#include "server-rag.h"
#include "server-rag-executors.h"
#include "rag-scheduler.h"

#include "build-info.h"
#include "preset.h"
#include "download.h"
#include "http.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <optional>
#include <sheredom/subprocess.h>

#include <functional>
#include <optional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <random>
#include <sstream>
#include <cstring>

#ifndef _WIN32
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

static int64_t server_models_steady_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_STATE "cmd_child_to_router:state:" // followed by json string

// address for child process, this is needed because router may run on 0.0.0.0
// ref: https://github.com/ggml-org/llama.cpp/issues/17862
#define CHILD_ADDR "127.0.0.1"

struct server_subproc {
    std::optional<subprocess_s> sproc; // empty while in DOWNLOADING state
    std::atomic<bool> stopped{false}; // set to cancel a download or signal child process exit

    subprocess_s & get() {
        GGML_ASSERT(sproc.has_value() && "subprocess not initialized");
        return sproc.value();
    }

    bool is_alive() {
        return sproc.has_value() && subprocess_alive(&sproc.value());
    }

    void request_exit() {
        if (sproc.has_value()) {
            FILE * stdin_file = subprocess_stdin(&sproc.value());
            if (stdin_file) {
                fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
                fflush(stdin_file);
            }
        }
        stopped.store(true, std::memory_order_relaxed);
    }

    void terminate() {
        if (!sproc.has_value()) {
            return;
        }
#if defined(_WIN32)
        if (sproc->hProcess == NULL) {
            return;
        }
#else
        if (sproc->child <= 0) {
            return;
        }
#endif
        subprocess_terminate(&sproc.value());
    }
};

// short loopback budget for the resumable stream router to child JSON calls (probe, lookup,
// delete). distinct from params.timeout_read/write which only applies to the generation proxy
static constexpr int STREAM_LOOKUP_TIMEOUT_MS = 250;

static std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}

static void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_ALIAS");
        preset.unset_option("LLAMA_ARG_HF_REPO");
    }
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) {
        return {};
    }

    const int len = static_cast<int>(std::wcslen(ws));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, ws, len, nullptr, 0, nullptr, nullptr);
    if (bytes == 0) {
        return {};
    }

    std::string utf8(bytes, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, len, utf8.data(), bytes, nullptr, nullptr);

    return utf8;
}
#endif

static std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wide_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}

void server_model_meta::update_args(common_preset_context & ctx_preset, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);
    preset.set_option(ctx_preset, "LLAMA_ARG_HOST",  CHILD_ADDR);
    preset.set_option(ctx_preset, "LLAMA_ARG_PORT",  std::to_string(port));
    preset.set_option(ctx_preset, "LLAMA_ARG_ALIAS", name);
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);

    // unified binary dispatches by subcommand, re-inject it right after the
    // binary path so the child starts as 'llama serve ...' not 'llama ...'
    const char * app_cmd = std::getenv("LLAMA_APP_CMD");
    if (app_cmd != nullptr && app_cmd[0] != '\0' && !bin_path.empty()) {
        args.insert(args.begin() + 1, app_cmd);
    }
}

void server_model_meta::update_caps() {
    try {
        common_params params;
        preset.apply_to_params(params, {
            "LLAMA_ARG_MODEL",
            "LLAMA_ARG_MODEL_URL",
            "LLAMA_ARG_MMPROJ",
            "LLAMA_ARG_MMPROJ_URL",
            "LLAMA_ARG_MMPROJ_AUTO",
            "LLAMA_ARG_HF_REPO",
            "LLAMA_ARG_HF_REPO_FILE",
        });
        params.offline = true;
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        common_models_handler_apply(handler, params); // note: this won't download the model because offline=true
        if (params.no_mmproj || params.mmproj.path.empty()) {
            multimodal = { false, false };
        } else {
            multimodal = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to initialize common_params for multimodal capability detection: %s\n", e.what());
        multimodal = { false, false };
    }
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params),
              base_env(get_environment()),
              base_preset(ctx_preset.load_from_args(argc, argv)) {
    // clean up base preset
    unset_reserved_args(base_preset, true);
    // set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = argv[0];
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using original argv[0] as fallback: %s\n", argv[0]);
    }
    load_models();
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    meta.update_args(ctx_preset, bin_path); // render args
    meta.update_caps();
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<server_subproc>(),
        /* th      */ std::thread(),
        /* meta    */ std::move(meta)
    };
}

void server_models::notify_sse(const std::string & event, const std::string & model_id, const json & data) {
    std::unique_ptr<server_task_result_router> result = std::make_unique<server_task_result_router>();
    result->data = {
        {"model", model_id},
        {"event", event},
    };
    if (!data.is_null()) {
        result->data["data"] = data;
    }
    SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", event.c_str(), model_id.c_str(), safe_json_to_str(result->data).c_str());
    sse.broadcast(std::move(result));
}

void server_models::load_models() {
    // Phase 1: load presets from all sources - pure I/O, no lock needed
    // 1. cached models
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_INF("Loaded %zu cached model presets\n", cached_models.size());
    // 2. local models from --models-dir
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_INF("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }
    // 3. custom-path models from presets
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        SRV_INF("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }

    // cascade, apply global preset first
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // note: if a model exists in both cached and local, local takes precedence
    common_presets final_presets;
    std::unordered_map<std::string, server_model_source> source_map;
    for (const auto & [name, preset] : cached_models) {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_CACHE;
    }
    for (const auto & [name, preset] : local_models)  {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_MODELS_DIR;
    }
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            final_presets[name].merge(custom);
        } else {
            final_presets[name] = custom;
        }
        source_map[name] = SERVER_MODEL_SOURCE_PRESET;
    }

    // overlay router's own CLI args on top of every model preset so that
    // e.g. `llama-server --temp 0` is honoured by all child processes
    for (auto & [name, preset] : final_presets) {
        preset.merge(base_preset);
    }

    auto get_source = [&](const std::string & name) {
        return source_map.count(name) ? source_map.at(name) : SERVER_MODEL_SOURCE_PRESET;
    };

    // Helpers that read `mapping` - must be called while holding the lock.
    std::unordered_set<std::string> custom_names;
    for (const auto & [name, preset] : custom_presets) custom_names.insert(name);
    auto join_set = [](const std::set<std::string> & s) {
        std::string result;
        for (const auto & v : s) {
            if (!result.empty()) result += ", ";
            result += v;
        }
        return result;
    };
    auto log_available_models = [&]() {
        SRV_INF("Available models (%zu) (*: custom preset)\n", mapping.size());
        for (const auto & [name, inst] : mapping) {
            bool has_custom = custom_names.find(name) != custom_names.end();
            std::string info;
            if (!inst.meta.aliases.empty()) info += " (aliases: " + join_set(inst.meta.aliases) + ")";
            if (!inst.meta.tags.empty())    info += " [tags: "    + join_set(inst.meta.tags)    + "]";
            SRV_INF("  %c %s%s\n", has_custom ? '*' : ' ', name.c_str(), info.c_str());
        }
    };
    auto apply_stop_timeout = [&]() {
        for (auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val)) {
                try {
                    inst.meta.stop_timeout = std::stoi(val);
                } catch (...) {
                    SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n",
                        val.c_str(), name.c_str(), DEFAULT_STOP_TIMEOUT);
                    inst.meta.stop_timeout = DEFAULT_STOP_TIMEOUT;
                }
            }
        }
    };
    // update_args() injects HOST/PORT/ALIAS, so strip them before comparing presets
    auto preset_options_for_compare = [](common_preset p) {
        p.unset_option("LLAMA_ARG_HOST");
        p.unset_option("LLAMA_ARG_PORT");
        p.unset_option("LLAMA_ARG_ALIAS");
        return p.options;
    };

    // Phase 2: acquire the lock once for all mapping mutations.
    // We temporarily release it only when calling functions that acquire it internally
    // (unload, load) or when joining threads (the monitoring thread calls update_status
    // which locks the mutex, so joining while holding it would deadlock).
    std::unique_lock<std::mutex> lk(mutex);

    need_reload = false;
    bool is_first_load = mapping.empty();

    if (is_first_load) {
        // FIRST LOAD: add all models, then unlock for autoloading
        for (const auto & [name, preset] : final_presets) {
            server_model_meta meta{
                /* source        */ get_source(name),
                /* preset        */ preset,
                /* name          */ name,
                /* aliases       */ {},
                /* tags          */ {},
                /* port          */ 0,
                /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used     */ 0,
                /* args          */ std::vector<std::string>(),
                /* loaded_info   */ {},
                /* progress      */ {},
                /* exit_code     */ 0,
                /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                /* multimodal    */ mtmd_caps{false, false},
                // /* need_download */ false,
            };
            add_model(std::move(meta));
        }
        apply_stop_timeout();
        log_available_models();

        std::vector<std::string> models_to_load;
        for (const auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                models_to_load.push_back(name);
            }
        }
        if ((int)models_to_load.size() > base_params.models_max) {
            throw std::runtime_error(string_format(
                "number of models to load on startup (%zu) exceeds models_max (%d)",
                models_to_load.size(), base_params.models_max));
        }

        lk.unlock();
        for (const auto & name : models_to_load) {
            SRV_INF("(startup) loading model %s\n", name.c_str());
            load(name);
        }
    } else {
        // RELOAD: diff the new preset list against the current mapping and reconcile
        is_reloading = true;

        // find running models whose source was removed or whose preset changed
        std::vector<std::string> to_unload;
        for (const auto & [name, inst] : mapping) {
            if (!inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) {
                to_unload.push_back(name); // removed from source
            } else if (preset_options_for_compare(inst.meta.preset) != preset_options_for_compare(it->second)) {
                to_unload.push_back(name); // preset changed
            }
        }

        // unload() acquires the lock internally, so release before each call
        for (const auto & name : to_unload) {
            SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", name.c_str());
            lk.unlock();
            unload(name);
            lk.lock();
        }

        // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
        cv.wait(lk, [&]() {
            for (const auto & name : to_unload) {
                auto it = mapping.find(name);
                if (it != mapping.end() && it->second.meta.is_running()) return false;
            }
            return true;
        });

        // collect all threads to join in one pass while the lock is held:
        // - monitoring threads from just-unloaded models (to_unload)
        // - threads of finished downloads (DOWNLOADED), they acquire the mutex on exit
        // - threads of already-UNLOADED models that are being removed from source
        std::vector<std::thread> threads_to_join;
        for (const auto & name : to_unload) {
            auto it = mapping.find(name);
            if (it != mapping.end() && it->second.th.joinable()) {
                threads_to_join.push_back(std::move(it->second.th));
            }
        }
        for (auto & [name, inst] : mapping) {
            if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                continue; // downloading models are not from config sources, leave them alone
            }
            if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // joining this thread under the lock deadlocks: it locks the mutex on its way out
                if (inst.th.joinable()) {
                    threads_to_join.push_back(std::move(inst.th));
                }
                continue;
            }
            if (final_presets.find(name) == final_presets.end() && !inst.meta.is_running() && inst.th.joinable()) {
                threads_to_join.push_back(std::move(inst.th));
            }
        }

        // join outside the lock - monitoring thread calls update_status (needs lock)
        lk.unlock();
        for (auto & th : threads_to_join) th.join();
        lk.lock();

        // erase models no longer in any source
        for (auto it = mapping.begin(); it != mapping.end(); ) {
            if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                ++it; // download thread is still busy, skip
            } else if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // download finished, thread is joined above, safe to erase
                GGML_ASSERT(!it->second.th.joinable());
                it = mapping.erase(it);
            } else if (final_presets.find(it->first) == final_presets.end()) {
                SRV_INF("(reload) removing model name=%s (no longer in source)\n", it->first.c_str());
                GGML_ASSERT(!it->second.th.joinable()); // must have been joined above
                it = mapping.erase(it);
            } else {
                ++it;
            }
        }

        // update presets for non-running models still in source
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) continue; // erased above

            inst.meta.preset = it->second;

            // re-parse aliases, then validate against other models
            std::set<std::string> new_aliases;
            std::string alias_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
                for (auto & alias : string_split<std::string>(alias_str, ',')) {
                    alias = string_strip(alias);
                    if (!alias.empty()) new_aliases.insert(alias);
                }
            }
            inst.meta.aliases.clear();
            for (const auto & alias : new_aliases) {
                bool conflict = false;
                for (const auto & [other_name, other_inst] : mapping) {
                    if (other_name == name) continue;
                    if (other_name == alias || other_inst.meta.aliases.count(alias)) {
                        SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n",
                            alias.c_str(), name.c_str(), other_name.c_str());
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) inst.meta.aliases.insert(alias);
            }

            // re-parse tags
            inst.meta.tags.clear();
            std::string tags_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
                for (auto & tag : string_split<std::string>(tags_str, ',')) {
                    tag = string_strip(tag);
                    if (!tag.empty()) inst.meta.tags.insert(tag);
                }
            }

            inst.meta.exit_code = 0; // clear failed state so the model can be reloaded
            inst.meta.update_args(ctx_preset, bin_path);
            inst.meta.update_caps();
        }

        // add models that are new in this reload
        std::vector<std::string> newly_added;
        for (const auto & [name, preset] : final_presets) {
            if (mapping.find(name) == mapping.end()) {
                server_model_meta meta{
                    /* source        */ get_source(name),
                    /* preset        */ preset,
                    /* name          */ name,
                    /* aliases       */ {},
                    /* tags          */ {},
                    /* port          */ 0,
                    /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                    /* last_used     */ 0,
                    /* args          */ std::vector<std::string>(),
                    /* loaded_info   */ {},
                    /* progress      */ {},
                    /* exit_code     */ 0,
                    /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                    /* multimodal    */ mtmd_caps{false, false},
                    // /* need_download */ false,
                };
                add_model(std::move(meta));
                newly_added.push_back(name);
            }
        }

        apply_stop_timeout();

        // clear reload flag before unlocking for autoload - load() blocks on !is_reloading,
        // so clearing it here (while still locked) prevents a deadlock in the autoload calls below
        is_reloading = false;
        cv.notify_all();

        log_available_models();

        // collect autoload candidates while still under the lock
        std::vector<std::string> to_autoload;
        for (const auto & name : newly_added) {
            auto it = mapping.find(name);
            if (it != mapping.end()) {
                std::string val;
                if (it->second.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                    to_autoload.push_back(name);
                }
            }
        }

        lk.unlock();
        for (const auto & name : to_autoload) {
            SRV_INF("(reload) loading new model %s\n", name.c_str());
            load(name);
        }

        notify_sse("models_reload", "*");
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;
        }
    }
    return std::nullopt;
}

std::optional<server_models::model_load_timing_snapshot>
server_models::get_model_load_timing_snapshot(const std::string & name) {
    const auto meta = get_meta(name);
    if (!meta.has_value()) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = mapping.find(meta->name);
    if (it == mapping.end()) {
        return std::nullopt;
    }
    const auto & instance = it->second;
    return model_load_timing_snapshot{
            instance.meta.status,
            instance.load_generation,
            instance.last_load_started_at_ms,
            instance.last_load_duration_ms,
            instance.last_ready_at_ms,
    };
}

// helper to convert vector<string> to char **
// pointers are only valid as long as the original vector is valid
static std::vector<char *> to_char_ptr_array(const std::vector<std::string> & vec) {
    std::vector<char *> result;
    result.reserve(vec.size() + 1);
    for (const auto & s : vec) {
        result.push_back(const_cast<char*>(s.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // no limit
    }
    // remove one of the servers if we passed the models_max (least recently used - LRU)
    std::string lru_model_name = "";
    int64_t lru_last_used = ggml_time_ms();
    size_t count_active = 0;
    {
        std::unique_lock<std::mutex> lk(mutex);
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                count_active++;
                if (m.second.meta.last_used < lru_last_used) {
                    lru_model_name = m.first;
                    lru_last_used = m.second.meta.last_used;
                }
            }
        }
    }
    if (!lru_model_name.empty() && count_active >= (size_t)base_params.models_max) {
        SRV_INF("models_max limit reached, removing LRU name=%s\n", lru_model_name.c_str());
        unload(lru_model_name);
        // wait for unload to complete
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this, &lru_model_name]() {
                return mapping[lru_model_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
    }
}

void server_models::load(const std::string & name) {
    load(name, load_options{});
}

void server_models::load(const std::string & name, const load_options & opts) {
    if (!opts.custom_meta.has_value()) {
        if (!has_model(name)) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        unload_lru();
    }

    std::unique_lock<std::mutex> lk(mutex);
    // edge case: block until any in-progress reload has finished so we always load
    // against the freshest preset and a consistent mapping state
    cv.wait(lk, [this]() { return !is_reloading; });

    auto meta = opts.custom_meta.has_value() ? *opts.custom_meta : mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // Re-check capacity under the lock to prevent concurrent loads from
    // exceeding models_max. Without this, the window between unload_lru()
    // releasing its lock and this lock_guard acquiring allows multiple
    // threads to each observe capacity and all proceed to load.
    if (base_params.models_max > 0) {
        size_t count_active = 0;
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                count_active++;
            }
        }
        if (count_active >= (size_t)base_params.models_max) {
            throw std::runtime_error("model limit reached, try again later");
        }
    }
    const int64_t load_started_at_ms = server_models_steady_time_ms();
    SRV_INF(
            "MODEL_LOAD_BEGIN model=%s weight_bytes=unavailable "
            "backend_buffer_bytes=unavailable kv_bytes=unavailable "
            "runtime_bytes=unavailable\n",
            meta.name.c_str());


    // prepare new instance info
    instance_t inst;
    inst.meta             = meta;
    inst.meta.port        = common_http_get_free_port();
    inst.meta.status      = SERVER_MODEL_STATUS_LOADING;
    inst.meta.loaded_info = json{};
    inst.meta.last_used   = ggml_time_ms();
    inst.load_started_at_ms = load_started_at_ms;
    inst.last_load_started_at_ms = load_started_at_ms;
    if (const auto previous = mapping.find(meta.name); previous != mapping.end()) {
        inst.load_generation = previous->second.load_generation;
        inst.last_load_duration_ms = previous->second.last_load_duration_ms;
        inst.last_ready_at_ms = previous->second.last_ready_at_ms;
    }

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<server_subproc>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        inst.meta.update_args(ctx_preset, bin_path); // render args

        std::vector<std::string> child_args = inst.meta.args; // copy
        std::vector<std::string> child_env  = base_env; // copy
        child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));

        if (opts.mode == SERVER_CHILD_MODE_DOWNLOAD) {
            inst.meta.status = SERVER_MODEL_STATUS_DOWNLOADING;
            child_env.push_back("LLAMA_SERVER_CHILD_MODE=download");
            child_env.push_back("LLAMA_ARG_HF_REPO=" + name);
        }

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args; // save for debugging

        std::vector<char *> argv = to_char_ptr_array(child_args);
        std::vector<char *> envp = to_char_ptr_array(child_env);

        // TODO @ngxson : maybe separate stdout and stderr in the future
        //                so that we can use stdout for commands and stderr for logging
        int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        inst.subproc->sproc.emplace();
        int result = subprocess_create_ex(argv.data(), options, envp.data(), &inst.subproc->get());
        if (result != 0) {
            throw std::runtime_error("failed to spawn server instance");
        }
    }

    // start a thread to manage the child process
    // captured variables are guaranteed to be destroyed only after the thread is joined
    inst.th = std::thread([
        this, name,
        child_proc = inst.subproc,
        port = inst.meta.port,
        stop_timeout = inst.meta.stop_timeout,
        child_mode = opts.mode
    ]() {
        FILE * stdin_file = subprocess_stdin(&child_proc->get());
        FILE * stdout_file = subprocess_stdout(&child_proc->get()); // combined stdout/stderr

        std::thread log_thread([&]() {
            // read stdout/stderr and forward to main server log
            // also handle status report from child process
            std::vector<char> vec_buf(128 * 1024); // large buffer for storing info
            char * buffer = vec_buf.data();
            if (stdout_file) {
                while (fgets(buffer, vec_buf.size(), stdout_file) != nullptr) {
                    LOG("[%5d] %s", port, buffer);
                    std::string str(buffer);
                    if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_STATE)) {
                        this->handle_child_state(name, str);
                    }
                }
            } else {
                SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
            }
        });

        std::thread stopping_thread([&]() {
            // thread to monitor explicit stop requests; child crash is signalled via child_proc->stopped
            auto is_stopping = [this, &name]() {
                return this->stopping_models.find(name) != this->stopping_models.end();
            };
            {
                std::unique_lock<std::mutex> lk(this->mutex);
                this->cv_stop.wait(lk, [&]() {
                    return is_stopping() || child_proc->stopped.load(std::memory_order_acquire);
                });
            }
            // child crashed or finished on its own, skip graceful shutdown sequence
            if (child_proc->stopped.load(std::memory_order_acquire)) {
                return;
            }
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
            fflush(stdin_file);
            int64_t start_time = ggml_time_ms();
            while (true) {
                std::unique_lock<std::mutex> lk(this->mutex);
                if (!is_stopping() || child_proc->stopped.load(std::memory_order_acquire)) {
                    return;
                }
                int64_t elapsed = ggml_time_ms() - start_time;
                if (elapsed >= stop_timeout * 1000) {
                    lk.unlock();
                    SRV_WRN("force-killing model instance name=%s after %d seconds timeout\n", name.c_str(), stop_timeout);
                    child_proc->terminate();
                    return;
                }
                this->cv_stop.wait_for(lk, std::chrono::seconds(1), [&]() {
                    return !is_stopping() || child_proc->stopped.load(std::memory_order_acquire);
                });
            }
        });

        // we reach here when the child process exits (stdout EOF)
        // note: we cannot join() prior to this point because it will close stdin_file
        if (log_thread.joinable()) {
            log_thread.join();
        }

        child_proc->stopped.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(this->mutex);
            stopping_models.erase(name);
            cv_stop.notify_all();
        }
        if (stopping_thread.joinable()) {
            stopping_thread.join();
        }

        // get the exit code
        int exit_code = 0;
        subprocess_join(&child_proc->get(), &exit_code);
        subprocess_destroy(&child_proc->get());

        // update status and exit code
        if (child_mode == SERVER_CHILD_MODE_DOWNLOAD) {
            // instance will be cleaned up on next load_models() call
        } else {
            this->update_status(name, {
                SERVER_MODEL_STATUS_UNLOADED,
                exit_code
            });
        }
        SRV_INF("instance name=%s exited with status %d\n", name.c_str(), exit_code);
    });

    // clean up old process/thread if exists
    {
        auto & old_instance = mapping[name];
        // old process should have exited already, but just in case, we clean it up here
        if (old_instance.subproc && old_instance.subproc->is_alive()) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            old_instance.subproc->terminate(); // force kill
        }
        if (old_instance.th.joinable()) {
            old_instance.th.join();
        }
    }

    notify_sse("model_status", name, {
        {"status", server_model_status_to_string(inst.meta.status)},
    });

    mapping[name] = std::move(inst);
    cv.notify_all();
}

void server_models::unload(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            it->second.subproc->request_exit();
            // for convenience, we wait the status change here
            wait(lk, name, [](const server_model_meta & new_meta) {
                return new_meta.status != SERVER_MODEL_STATUS_DOWNLOADING;
            });
        } else if (it->second.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            stopping_models.insert(name);
            if (it->second.meta.status == SERVER_MODEL_STATUS_LOADING) {
                // special case: if model is in loading state, unloading means force-killing it
                SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
                it->second.subproc->terminate();
            }
            cv_stop.notify_all();
            // status change will be handled by the managing thread
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}

bool server_models::unload_and_wait(const std::string & name) {
    const auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }

    const std::string canonical_name = meta->name;
    if (!meta->is_running()) {
        return false;
    }

    SRV_INF(
            "MODEL_UNLOAD_BEGIN model=%s weight_bytes=unavailable "
            "backend_buffer_bytes=unavailable kv_bytes=unavailable "
            "runtime_bytes=unavailable\n",
            canonical_name.c_str());

    unload(canonical_name);
    wait(canonical_name, [](const server_model_meta & current) {
        return current.status == SERVER_MODEL_STATUS_UNLOADED;
    });

    const auto final_meta = get_meta(canonical_name);
    const int exit_code = final_meta.has_value()
            ? final_meta->exit_code
            : 0;

    SRV_INF(
            "MODEL_UNLOAD_END model=%s status=unloaded exit_code=%d "
            "weight_bytes=unavailable backend_buffer_bytes=unavailable "
            "kv_bytes=unavailable runtime_bytes=unavailable\n",
            canonical_name.c_str(),
            exit_code);

    if (exit_code != 0) {
        throw std::runtime_error(
                "model name=" + canonical_name +
                " exited with status " + std::to_string(exit_code));
    }

    return true;
}

void server_models::unload_all() {
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lk(mutex);
        for (auto & [name, inst] : mapping) {
            if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                SRV_INF("cancelling download for model name=%s\n", name.c_str());
                inst.subproc->stopped.store(true, std::memory_order_relaxed);
            } else if (inst.meta.is_running()) {
                SRV_INF("stopping model instance name=%s\n", name.c_str());
                stopping_models.insert(name);
                cv_stop.notify_all();
                // status change will be handled by the managing thread
            }
            // moving the thread to join list to avoid deadlock
            to_join.push_back(std::move(inst.th));
        }
    }
    for (auto & th : to_join) {
        if (th.joinable()) {
            th.join();
        }
    }
}

void server_models::update_status(const std::string & name, const update_status_args & args) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & instance = it->second;
        auto & meta = instance.meta;
        const auto previous_status = meta.status;
        if (args.status == SERVER_MODEL_STATUS_LOADED) {
            const int64_t ready_at_ms = server_models_steady_time_ms();
            instance.last_ready_at_ms = ready_at_ms;
            if (previous_status == SERVER_MODEL_STATUS_LOADING &&
                instance.load_started_at_ms > 0) {
                instance.last_load_duration_ms =
                        std::max<int64_t>(0, ready_at_ms - instance.load_started_at_ms);
                instance.load_generation++;
                instance.load_started_at_ms = 0;
            }
        }
        meta.status      = args.status;
        meta.exit_code   = args.exit_code;
        if (!args.loaded_info.is_null()) {
            meta.loaded_info = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            meta.progress = args.progress;
        }
    }
    // broadcast status change to SSE
    {
        json data = {
            {"status", server_model_status_to_string(args.status)},
        };
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            data["exit_code"] = args.exit_code;
        }
        if (!args.loaded_info.is_null()) {
            data["info"] = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            data["progress"] = args.progress;
        }
        // note: notify_sse doesn't acquire the lock, so no deadlock here
        notify_sse("status_change", name, data);
    }
    cv.notify_all();
}

void server_models::update_download_progress(const std::string & name, const common_download_progress & progress, bool done, bool ok) {
    json curr;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            if (done) {
                // mark the instance to be erased on next load_models() call
                it->second.meta.status = SERVER_MODEL_STATUS_DOWNLOADED;
                need_reload = true;
            } else {
                json & info = it->second.meta.loaded_info;
                if (!info.contains("progress")) {
                    info["progress"] = json{};
                }
                info["progress"][progress.url] = {
                    {"done",  progress.downloaded},
                    {"total", progress.total},
                };
                curr = it->second.meta.loaded_info; // copy
            }
        }
    }
    if (done) {
        cv.notify_all(); // notify in case unload() is waiting for download to be cancelled
        notify_sse(ok ? "download_finished" : "download_failed", name, {});
    } else {
        notify_sse("download_progress", name, curr);
    }
}

bool server_models::remove(const std::string & name) {
    // do everything under one lock acquisition; avoid get_meta() /
    // unload() because they can trigger load_models() which erases
    // transient DOWNLOADING / DOWNLOADED entries as a side-effect
    std::unique_lock<std::mutex> lk(mutex);

    auto it = mapping.find(name);
    if (it == mapping.end()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (it->second.meta.source != SERVER_MODEL_SOURCE_CACHE) {
        throw std::runtime_error("model name=" + name + " is not removable (not from cache)");
    }

    if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
        // cancel in-flight download
        SRV_INF("cancelling download for model name=%s\n", name.c_str());
        it->second.subproc->request_exit();
    } else if (it->second.meta.is_running()) {
        // stop running instance
        SRV_INF("stopping model instance name=%s\n", name.c_str());
        stopping_models.insert(name);
        if (it->second.meta.status == SERVER_MODEL_STATUS_LOADING) {
            it->second.subproc->terminate();
        }
        cv_stop.notify_all();
    }

    // wait until the monitoring thread finishes
    wait(lk, name, [](const server_model_meta & meta) {
        return meta.status == SERVER_MODEL_STATUS_UNLOADED
            || meta.status == SERVER_MODEL_STATUS_DOWNLOADED;
    });

    // re-find after wait - load_models() may have erased the entry during the wait
    it = mapping.find(name);
    if (it == mapping.end()) {
        // load_models() already joined the thread and erased the entry;
        // we just need to clean up the cached files on disk
        lk.unlock();
        bool ok = common_download_remove(name);
        SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
        notify_sse("model_remove", name, {});
        return true;
    }

    // join before erasing - thread no longer acquires this mutex
    if (it->second.th.joinable()) {
        it->second.th.join();
    }

    // remove from disk (best-effort: cancelled downloads may have no cached files)
    bool ok = common_download_remove(name);
    mapping.erase(name);
    if (!ok) {
        SRV_WRN("removing model name=%s from disk returned false (no cached files?)\n", name.c_str());
    }
    SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
    notify_sse("model_remove", name, {});
    return true;
}

void server_models::wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    std::unique_lock<std::mutex> lk(mutex);
    wait(lk, name, predicate);
}

void server_models::wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    cv.wait(lk, [this, &name, &predicate]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return predicate(it->second.meta);

        }
        // model was removed from mapping by another code path (e.g. load_models()).
        // nothing left to wait for - tell the caller to proceed.
        return true;
    });
}

bool server_models::ensure_model_ready(const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->is_ready()) {
        return false; // ready for taking requests
    }

    if (meta->status == SERVER_MODEL_STATUS_SLEEPING) {
        return false; // child is sleeping but still running; new request will wake it up
    }

    if (meta->status == SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model name=%s is not loaded, loading...\n", name.c_str());
        load(name);
    }

    // wait for loading to complete
    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    wait(name, [&meta](const server_model_meta & new_meta) {
        if (new_meta.status != SERVER_MODEL_STATUS_LOADING) {
            meta = new_meta; // update meta for final check after wait
            return true;
        }
        return false;
    });

    // check final status
    if (!meta.has_value() || meta->is_failed()) {
        throw std::runtime_error("model name=" + name + " failed to load");
    }

    return true;
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (!meta->is_running()) {
        throw std::invalid_argument("model name=" + name + " is not running");
    }
    if (update_last_used) {
        std::unique_lock<std::mutex> lk(mutex);
        mapping[name].meta.last_used = ggml_time_ms();
    }
    SRV_INF("proxying request to model %s on port %d\n", name.c_str(), meta->port);
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += '?' + req.query_string;
    }
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            "http",
            CHILD_ADDR,
            meta->port,
            proxy_path,
            req.headers,
            req.body,
            req.files,
            req.should_stop,
            base_params.timeout_read,
            base_params.timeout_write
            );
    return proxy;
}

json server_models::request_model_json(
        const std::string & model,
        const std::string & path,
        const json & request_body) {
    if (model.empty()) {
        throw std::invalid_argument("internal model request is missing model name");
    }

    if (path.empty() || path.front() != '/') {
        throw std::invalid_argument(
                "internal model request path must start with '/': " + path);
    }

    // Resolve an alias to the canonical model name.
    auto meta = get_meta(model);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + model + " is not found");
    }

    const std::string name = meta->name;

    // Start the child process when necessary and wait until loading completes.
    ensure_model_ready(name);

    // Metadata must be refreshed after ensure_model_ready(), because loading
    // can create the child process and assign its listening port.
    meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error(
                "model name=" + name + " disappeared after loading");
    }

    if (!meta->is_running()) {
        throw std::runtime_error(
                "model name=" + name + " is not running after loading");
    }

    // Match proxy_request() behavior so that router LRU accounting remains
    // correct for requests made by the RAG orchestrator.
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            it->second.meta.last_used = ggml_time_ms();
        }
    }

    json body = request_body;
    body["model"] = name;

    SRV_INF(
            "sending internal JSON request to model %s on port %d, path=%s\n",
            name.c_str(),
            meta->port,
            path.c_str());

    httplib::Client cli(CHILD_ADDR, meta->port);

    // Use the same router timeout configuration as ordinary model requests.
    cli.set_read_timeout(base_params.timeout_read, 0);
    cli.set_write_timeout(base_params.timeout_write, 0);

    const auto response = cli.Post(
            path,
            body.dump(),
            "application/json");

    if (!response) {
        throw std::runtime_error(
                "internal request to model=" + name +
                ", path=" + path +
                " failed before receiving an HTTP response");
    }

    if (response->status < 200 || response->status >= 300) {
        throw std::runtime_error(
                "internal request to model=" + name +
                ", path=" + path +
                " returned HTTP " + std::to_string(response->status) +
                ": " + response->body);
    }

    if (response->body.empty()) {
        throw std::runtime_error(
                "internal request to model=" + name +
                ", path=" + path +
                " returned an empty body");
    }

    try {
        return json::parse(response->body);
    } catch (const json::parse_error & e) {
        throw std::runtime_error(
                "internal request to model=" + name +
                ", path=" + path +
                " returned invalid JSON: " + std::string(e.what()));
    }
}


std::vector<std::vector<float>> server_models::request_model_embeddings(
        const std::string & model,
        const std::vector<std::string> & inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument(
                "embedding request contains no input texts");
    }

    const json request_body = {
        {"input", inputs},
        {"encoding_format", "float"},
    };

    const json response = request_model_json(
            model,
            "/v1/embeddings",
            request_body);

    if (!response.contains("data") ||
        !response["data"].is_array()) {
        throw std::runtime_error(
                "embedding model returned a response without a data array");
    }

    const json & data = response["data"];

    if (data.size() != inputs.size()) {
        throw std::runtime_error(
                "embedding response count does not match input count");
    }

    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(data.size());

    std::size_t expected_dimension = 0;

    for (std::size_t index = 0; index < data.size(); ++index) {
        const json & item = data.at(index);

        if (!item.is_object() ||
            !item.contains("embedding") ||
            !item["embedding"].is_array()) {
            throw std::runtime_error(
                    "embedding response item " +
                    std::to_string(index) +
                    " does not contain an embedding array");
        }

        std::vector<float> embedding;

        try {
            embedding =
                    item["embedding"].get<std::vector<float>>();
        } catch (const json::exception & e) {
            throw std::runtime_error(
                    "failed to parse embedding response item " +
                    std::to_string(index) +
                    ": " + std::string(e.what()));
        }

        if (embedding.empty()) {
            throw std::runtime_error(
                    "embedding response item " +
                    std::to_string(index) +
                    " is empty");
        }

        if (expected_dimension == 0) {
            expected_dimension = embedding.size();
        } else if (embedding.size() != expected_dimension) {
            throw std::runtime_error(
                    "embedding response contains inconsistent dimensions");
        }

        embeddings.push_back(std::move(embedding));
    }

    return embeddings;
}

json server_models::request_model_rerank(
        const std::string & model,
        const std::string & query,
        const std::vector<std::string> & documents,
        std::size_t top_n) {
    if (query.empty()) {
        throw std::invalid_argument(
                "reranking request contains an empty query");
    }

    if (documents.empty()) {
        throw std::invalid_argument(
                "reranking request contains no candidate documents");
    }

    if (top_n == 0) {
        throw std::invalid_argument(
                "reranking request top_n must be greater than zero");
    }

    const std::size_t requested_top_n =
            std::min(top_n, documents.size());

    const json request_body = {
        {"query", query},
        {"documents", documents},
        {"top_n", requested_top_n},
    };

    const json response = request_model_json(
            model,
            "/v1/rerank",
            request_body);

    if (!response.contains("results") ||
        !response["results"].is_array()) {
        throw std::runtime_error(
                "reranking model returned a response without "
                "a results array");
    }

    const json & results = response["results"];

    if (results.empty()) {
        throw std::runtime_error(
                "reranking model returned no ranked results");
    }

    if (results.size() > requested_top_n) {
        throw std::runtime_error(
                "reranking model returned more results than requested");
    }

    json normalized_results = json::array();
    std::vector<bool> seen_indices(documents.size(), false);

    for (std::size_t rank = 0; rank < results.size(); ++rank) {
        const json & item = results.at(rank);

        if (!item.is_object() ||
            !item.contains("index") ||
            !item["index"].is_number_integer()) {
            throw std::runtime_error(
                    "reranking response item " +
                    std::to_string(rank) +
                    " does not contain an integer index");
        }

        if (!item.contains("relevance_score") ||
            !item["relevance_score"].is_number()) {
            throw std::runtime_error(
                    "reranking response item " +
                    std::to_string(rank) +
                    " does not contain a relevance_score");
        }

        const long long signed_index =
                item["index"].get<long long>();

        if (signed_index < 0) {
            throw std::runtime_error(
                    "reranking response contains a negative index");
        }

        const std::size_t candidate_index =
                static_cast<std::size_t>(signed_index);

        if (candidate_index >= documents.size()) {
            throw std::runtime_error(
                    "reranking response contains an out-of-range index");
        }

        if (seen_indices[candidate_index]) {
            throw std::runtime_error(
                    "reranking response contains a duplicate index");
        }

        seen_indices[candidate_index] = true;

        normalized_results.push_back({
            {"index", candidate_index},
            {"relevance_score",
             item["relevance_score"].get<float>()},
        });
    }

    return normalized_results;
}


void server_models::handle_child_state(const std::string & name, const std::string & raw_input) {
    server_state state;
    json payload;

    try {
        json data = json::parse(raw_input.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
        state = server_state_from_str(json_value(data, "state", std::string()));
        payload = json_value(data, "payload", json{});
    } catch (const std::exception & e) {
        SRV_ERR("failed to parse child state update for name=%s: %s\n", name.c_str(), e.what());
        return;
    }

    switch (state) {
        case SERVER_STATE_DOWNLOADING:
            {
                std::string result = json_value(payload, "result", std::string());
                std::string url    = json_value(payload, "url",    std::string());
                auto request_exit = [&]() {
                    std::lock_guard<std::mutex> lk(mutex);
                    auto it = mapping.find(name);
                    if (it != mapping.end()) {
                        return it->second.subproc->request_exit();
                    }
                };
                if (result == "download_finished") {
                    update_download_progress(name, {}, true, true);
                    request_exit();
                } else if (result == "download_failed") {
                    update_download_progress(name, {}, true, false);
                    request_exit();
                } else if (!url.empty()) {
                    common_download_progress p;
                    p.url        = url;
                    p.downloaded = json_value(payload, "downloaded", (size_t)0);
                    p.total      = json_value(payload, "total", (size_t)0);
                    update_download_progress(name, p, false);
                }
            } break;
        case SERVER_STATE_LOADING:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADING,
                    0,
                    nullptr, // no loaded_info yet
                    payload,
                });
            } break;
        case SERVER_STATE_READY:
            {
                const auto before = get_meta(name);
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADED,
                    0,
                    // note: payload can be empty if this is a wakeup from sleep
                    payload.size() > 0 ? payload : nullptr,
                    {}, // reset progress info
                });
                if (before.has_value() &&
                    before->status == SERVER_MODEL_STATUS_LOADING) {
                    const auto timing = get_model_load_timing_snapshot(name);
                    SRV_INF(
                            "MODEL_LOAD_END model=%s status=loaded "
                            "duration_ms=%lld load_generation=%llu "
                            "weight_bytes=unavailable "
                            "backend_buffer_bytes=unavailable "
                            "kv_bytes=unavailable runtime_bytes=unavailable\n",
                            before->name.c_str(),
                            static_cast<long long>(timing.has_value() ? timing->last_load_duration_ms : 0),
                            static_cast<unsigned long long>(timing.has_value() ? timing->load_generation : 0));
                }
            } break;
        case SERVER_STATE_SLEEPING:
            {
                update_status(name, { SERVER_MODEL_STATUS_SLEEPING });
            } break;
        default:
            // should never happen, but just in case
            GGML_ASSERT(false && "unexpected state from child server");
    }
}

//
// server_child
//

bool server_child::is_child() {
    const char * router_port = std::getenv("LLAMA_SERVER_ROUTER_PORT");
    return router_port != nullptr;
}

server_child_mode server_child::get_mode() {
    const char * mode = std::getenv("LLAMA_SERVER_CHILD_MODE");
    std::string mode_str(mode ? mode : "");
    if (mode_str == "download") {
        return SERVER_CHILD_MODE_DOWNLOAD;
    } else {
        return SERVER_CHILD_MODE_NORMAL;
    }
}

struct server_download_state : public common_download_callback {
    server_child * self;
    std::function<bool()> should_stop;
    std::atomic<int64_t> last_progress_time{0}; // multiple files downloading in different threads
    bool is_ok = false;

    server_download_state(server_child * s) : self(s) {}

    bool run(common_params & params) {
        try {
            common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            common_models_handler_apply(handler, params, this);
            is_ok = true;
        } catch (const std::exception & e) {
            auto model_name = params.model.get_name();
            SRV_ERR("download failed for model name=%s: %s\n", model_name.c_str(), e.what());
            is_ok = false;
        }
        return is_ok;
    }
    void on_progress(const common_download_progress & p) {
        json data = {
            {"url", p.url},
            {"downloaded", p.downloaded},
            {"total", p.total},
        };
        self->notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), data);
    }
    void on_start(const common_download_progress & p) override {
        on_progress(p);
    }
    void on_update(const common_download_progress & p) override {
        int64_t now = ggml_time_ms();
        // throttle progress updates to avoid flooding logs
        if (now - last_progress_time.load(std::memory_order_relaxed) >= 100) {
            on_progress(p);
            last_progress_time.store(now, std::memory_order_relaxed);
        }
    }
    void on_done(const common_download_progress & p, bool) override {
        on_progress(p);
    }
    bool is_cancelled() const override {
        return should_stop ? should_stop() : false;
    }
};

int server_child::run_download(common_params & params) {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);

    // monitor stdin for cancellation command from the router
    std::thread signal_thread = setup([cancelled](int) {
        cancelled->store(true, std::memory_order_relaxed);
    });

    server_download_state dl(this);
    dl.should_stop = [cancelled]() {
        return cancelled->load(std::memory_order_relaxed);
    };

    bool ok = dl.run(params);

    notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), {
        {"result", ok ? "download_finished" : "download_failed"},
    });

    // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
    if (signal_thread.joinable()) {
        signal_thread.join();
    }

    SRV_INF("download completed %s\n", ok ? "successfully" : "with errors");
    return 0;
}

std::thread server_child::setup(const std::function<void(int)> & shutdown_handler) {
    // setup thread for monitoring stdin
    return std::thread([shutdown_handler]() {
        // wait for EOF on stdin
        SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
        bool eof = false;
        while (true) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                // EOF detected, that means the router server is unexpectedly exit or killed
                eof = true;
                break;
            }
            if (line.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
                SRV_INF("%s", "exit command received, exiting...\n");
                shutdown_handler(0);
                break;
            }
        }
        if (eof) {
            SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
            exit(1);
        }
    });
}

void server_child::notify_to_router(const std::string & state, const json & payload) {
    json data = {
        {"state", state},
        {"payload", payload},
    };
    std::lock_guard<std::mutex> lk(mtx_stdout);
    common_log_pause(common_log_main());
    fflush(stdout);
    fprintf(stdout, "%s%s\n", CMD_CHILD_TO_ROUTER_STATE, safe_json_to_str(data).c_str());
    fflush(stdout);
    common_log_resume(common_log_main());
}


//
// server_models_routes
//

// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> sse_client_id_counter = 0;
struct server_models_sse_client {
    server_response & queue_results;
    int client_id;
    server_models_sse_client(server_response & q)
            : queue_results(q), client_id(sse_client_id_counter.fetch_add(1, std::memory_order_relaxed)) {
        SRV_DBG("new SSE client connected, assigned client_id=%d\n", client_id);
        queue_results.add_waiting_task_id(client_id);
    }
    ~server_models_sse_client() {
        SRV_DBG("SSE client disconnected, removing client_id=%d\n", client_id);
        queue_results.remove_waiting_task_id(client_id);
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (true) {
            static const int http_polling_seconds = 1; // check should_stop every 1 second
            server_task_result_ptr result = queue_results.recv_with_timeout({client_id}, http_polling_seconds);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    return nullptr;
                }
                // continue waiting otherwise
            } else {
                SRV_DBG("recv result for client_id=%d: %s\n", client_id, safe_json_to_str(result->to_json()).c_str());
                return result;
            }
        }
        // should not reach here
    }
};

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    // resolve alias to canonical model name
    name = meta->name;
    if (models_autoload) {
        models.ensure_model_ready(name);
    } else {
        if (!meta->is_running()) {
            res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return false;
        }
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_http_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}

// percent encode one query or path component, covers reserved chars without pulling in
// httplib::detail. used by the stream routes to forward conversation_id to children safely
static std::string encode_qs(const std::string & in) {
    std::string out;
    out.reserve(in.size() * 3);
    for (unsigned char c : in) {
        bool safe = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                 || c == '-' || c == '_' || c == '.' || c == '~';
        if (safe) {
            out.push_back(char(c));
        } else {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out.append(buf, 3);
        }
    }
    return out;
}

// resolve the child that owns a conversation's stream session via the conv_id -> model map
// populated when the POST was routed. single map lookup then a meta lookup, no polling, no
// parsing of the conv id. returns nullopt when nothing maps, the caller answers not found and
// the client recovers
static std::optional<server_model_meta> resolve_child_for_conv(
        server_models & models, const std::string & conversation_id) {
    if (conversation_id.empty()) {
        return std::nullopt;
    }
    auto tracked = models.conv_models.lookup(conversation_id);
    if (!tracked.has_value()) {
        return std::nullopt;
    }
    auto meta = models.get_meta(*tracked);
    if (meta.has_value() && meta->is_ready()) {
        return meta;
    }
    return std::nullopt;
}

void server_models_routes::init_routes() {
    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_http_res>();
            res_ok(res, {
                // TODO: add support for this on web UI
                {"role",                 "router"},
                {"max_instances",        params.models_max},
                {"models_autoload",      params.models_autoload},
                // this is a dummy response to make sure the UI doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
                // New key
                {"ui_settings",          ui_settings},
                {"build_info",           std::string(llama_build_info())},
                {"cors_proxy_enabled",   params.ui_mcp_proxy},
            });
            return res;
        }
        return proxy_get(req);
    };

    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        // remember which child serves this conversation so the stream routes can route straight
        // to it without polling, keyed on the exact conv id from the header
        std::string conv_id = server_stream_conv_id_from_headers(req.headers);
        if (!conv_id.empty()) {
            models.conv_models.remember(conv_id, name);
        }
        return models.proxy_request(req, method, name, true); // update last usage for POST request only
    };

        this->post_rag = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        const auto total_start_ms = ggml_time_ms();
        RagStageMetrics stage_metrics;

        const auto elapsed_ms = [](const auto start_ms) -> std::size_t {
            const auto end_ms = ggml_time_ms();
            return end_ms > start_ms
                    ? static_cast<std::size_t>(end_ms - start_ms)
                    : 0;
        };

        const json body = json::parse(req.body);

        const auto parse_schedule_policy =
                [](const std::string & mode)
                        -> std::optional<RagSchedulePolicy> {
            if (mode == "sequential") {
                return RagSchedulePolicy::Sequential;
            }

            if (mode == "carrier_baseline") {
                return RagSchedulePolicy::CarrierBaseline;
            }

            if (mode == "hetero_parallel") {
                return RagSchedulePolicy::HeteroParallel;
            }

            if (mode == "npu_cpu_cs") {
                return RagSchedulePolicy::NpuCpuCriticalScore;
            }

            return std::nullopt;
        };

        const auto critical_score_for_task =
                [](RagTaskType type) -> double {
            constexpr double kProfile4kIndexingMs = 5086.0;
            constexpr double kProfile4kQueryExpandMs = 1115.0;
            constexpr double kProfile4kQueryEmbeddingTotalMs = 1242.0;
            constexpr double kProfile4kRetrievalBranchCount = 3.0;
            constexpr double kProfile4kQueryEmbeddingNodeMs =
                    kProfile4kQueryEmbeddingTotalMs /
                    kProfile4kRetrievalBranchCount;
            constexpr double kProfile4kSearchingMs = 16.0;
            constexpr double kProfile4kRerankingMs = 4401.0;
            constexpr double kProfile4kGenerationMs = 26973.0;
            constexpr double kProfile4kGenerationPrefillMs = 2283.0;
            constexpr double kProfile4kGenerationDecodeMs = 26179.0;

            constexpr double kScoreCandidateMerge = 1.0;
            constexpr double kScoreGenerationDecode =
                    kProfile4kGenerationDecodeMs;
            constexpr double kScoreGenerationPrefill =
                    kProfile4kGenerationPrefillMs +
                    kScoreGenerationDecode;
            constexpr double kScoreGeneration =
                    kProfile4kGenerationMs;
            constexpr double kScoreReranking =
                    kProfile4kRerankingMs +
                    kScoreGenerationPrefill;
            constexpr double kScoreSearching =
                    kProfile4kSearchingMs +
                    kScoreReranking;
            constexpr double kScoreQueryEmbedding =
                    kProfile4kQueryEmbeddingNodeMs +
                    kScoreSearching;
            constexpr double kScoreQueryExpansion =
                    kProfile4kQueryExpandMs +
                    kProfile4kQueryEmbeddingTotalMs +
                    kScoreSearching;
            constexpr double kScoreDocumentEmbedding =
                    kProfile4kIndexingMs +
                    kScoreSearching;

            switch (type) {
                case RagTaskType::DocumentEmbedding:
                    return kScoreDocumentEmbedding;
                case RagTaskType::QueryExpansion:
                    return kScoreQueryExpansion;
                case RagTaskType::QueryEmbedding:
                    return kScoreQueryEmbedding;
                case RagTaskType::VectorSearch:
                    return kScoreSearching;
                case RagTaskType::RetrievalMerge:
                    return kScoreReranking;
                case RagTaskType::Reranking:
                    return kScoreReranking;
                case RagTaskType::Generation:
                    return kScoreGeneration;
                case RagTaskType::GenerationPrefill:
                    return kScoreGenerationPrefill;
                case RagTaskType::GenerationDecode:
                    return kScoreGenerationDecode;
                case RagTaskType::CandidateMerge:
                    return kScoreCandidateMerge;
                case RagTaskType::Finalize:
                    return 0.0;
            }

            return 0.0;
        };

        const std::string requested_mode =
                json_value(
                        body,
                        "mode",
                        std::string("sequential"));

        const std::optional<RagSchedulePolicy>
                schedule_policy =
                        parse_schedule_policy(
                                requested_mode);

        if (!schedule_policy.has_value()) {
            res_err(
                    res,
                    format_error_response(
                            "unsupported RAG mode '" +
                            requested_mode +
                            "'; supported modes: "
                            "sequential, carrier_baseline, "
                            "hetero_parallel, npu_cpu_cs",
                            ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        {
            auto runtime =
                    std::make_shared<RagRequestRuntime>();

            runtime->request_id =
                    static_cast<std::uint64_t>(
                            ggml_time_us());

            runtime->request.mode =
                    requested_mode;

            runtime->request.doc =
                    json_value(
                            body,
                            "doc",
                            std::string());

            runtime->request.query =
                    json_value(
                            body,
                            "query",
                            std::string());

            runtime->request.generation_model =
                    json_value(
                            body,
                            "generation_model",
                            std::string());

            runtime->request.embedding_model =
                    json_value(
                            body,
                            "embedding_model",
                            std::string());

            runtime->request.rerank_model =
                    json_value(
                            body,
                            "rerank_model",
                            std::string());

            runtime->request.expansion_model =
                    json_value(
                            body,
                            "expansion_model",
                            std::string());

            runtime->request.enable_query_expansion =
                    json_value(
                            body,
                            "enable_query_expansion",
                            false);

            const char * stage_residency_env =
                    std::getenv("LLAMA_RAG_STAGE_MODEL_RESIDENCY");
            const bool stage_residency_default =
                    stage_residency_env != nullptr &&
                    std::strcmp(stage_residency_env, "1") == 0;

            runtime->request.stage_model_residency =
                    json_value(
                            body,
                            "stage_model_residency",
                            stage_residency_default);

            runtime->request.max_expanded_queries =
                    json_value(
                            body,
                            "max_expanded_queries",
                            std::size_t(3));

            runtime->request.top_k =
                    json_value(
                            body,
                            "top_k",
                            std::size_t(20));

            runtime->request.top_n =
                    json_value(
                            body,
                            "top_n",
                            std::size_t(5));

            runtime->request.max_tokens =
                    json_value(
                            body,
                            "max_tokens",
                            std::size_t(64));

            runtime->request.generation_candidate_repeats =
                    json_value(
                            body,
                            "generation_candidate_repeats",
                            std::size_t(1));

            runtime->request.temperature =
                    json_value(
                            body,
                            "temperature",
                            0.1F);

            runtime->request.seed =
                    json_value(
                            body,
                            "seed",
                            std::uint32_t(1234));

            runtime->request.generation_prefill_backend =
                    json_value(
                            body,
                            "generation_prefill_backend",
                            std::string("npu"));

            runtime->request.generation_decode_backend =
                    json_value(
                            body,
                            "generation_decode_backend",
                            std::string("cpu"));

            const RagRequest & rag_request =
                    runtime->request;

            if (rag_request.doc.empty() ||
                rag_request.query.empty() ||
                rag_request.generation_model.empty() ||
                rag_request.embedding_model.empty()) {
                res_err(
                        res,
                        format_error_response(
                                "doc, query, generation_model "
                                "and embedding_model are required",
                                ERROR_TYPE_INVALID_REQUEST));
                return res;
            }

            if (rag_request.max_expanded_queries == 0 ||
                rag_request.top_k == 0 ||
                rag_request.top_n == 0 ||
                rag_request.max_tokens == 0 ||
                rag_request.generation_candidate_repeats == 0 ||
                rag_request.generation_candidate_repeats > 8) {
                res_err(
                        res,
                        format_error_response(
                                "max_expanded_queries, top_k, "
                                "top_n, max_tokens and "
                                "generation_candidate_repeats must "
                                "be greater than zero; "
                                "generation_candidate_repeats must "
                                "not exceed 8",
                                ERROR_TYPE_INVALID_REQUEST));
                return res;
            }

            RagScheduler scheduler;

            const RagTaskType executor_types[] = {
                RagTaskType::DocumentEmbedding,
                RagTaskType::QueryExpansion,
                RagTaskType::QueryEmbedding,
                RagTaskType::VectorSearch,
                RagTaskType::RetrievalMerge,
                RagTaskType::Reranking,
                RagTaskType::Generation,
                RagTaskType::GenerationPrefill,
                RagTaskType::GenerationDecode,
                RagTaskType::CandidateMerge,
                RagTaskType::Finalize,
            };

            std::vector<
                    std::unique_ptr<RagServerExecutor>>
                    executors;

            executors.reserve(
                    sizeof(executor_types) /
                    sizeof(executor_types[0]));

            for (const RagTaskType type :
                 executor_types) {
                executors.emplace_back(
                        new RagServerExecutor(
                                &models,
                                type));

                scheduler.register_executor(
                        type,
                        RagBackend::CPU,
                        executors.back().get());
            }

            // ----------------------------------------------------------
            // Phase A:
            // DocumentEmbedding and QueryExpansion have no dependency
            // and are therefore dispatched concurrently.
            // ----------------------------------------------------------

            RagExecutionPlan phase_a;
            phase_a.request_id =
                    static_cast<int>(
                            runtime->request_id &
                            0x7fffffffU);
            phase_a.policy =
                    *schedule_policy;
            phase_a.runtime = runtime.get();

            phase_a.add_node(
                    0,
                    RagTaskType::DocumentEmbedding,
                    RagBackend::CPU);

            phase_a.add_node(
                    1,
                    RagTaskType::QueryExpansion,
                    RagBackend::CPU);

            // The memory-baseline mode intentionally trades preparation
            // latency for a single-model working set.
            if (runtime->request.stage_model_residency) {
                phase_a.add_dependency(0, 1);
            }

            for (auto & node : phase_a.nodes) {
                node.critical_score =
                        critical_score_for_task(node.type);
            }

            if (!scheduler.run(phase_a)) {
                const std::string error =
                        runtime->error.empty()
                                ? "RAG preparation phase failed"
                                : runtime->error;

                res_err(
                        res,
                        format_error_response(
                                error,
                                ERROR_TYPE_SERVER));
                return res;
            }

            if (runtime->expanded_queries.empty()) {
                res_err(
                        res,
                        format_error_response(
                                "query expansion produced no "
                                "queries",
                                ERROR_TYPE_SERVER));
                return res;
            }

            runtime->query_branches.clear();
            runtime->query_branches.resize(
                    runtime->expanded_queries.size());

            for (std::size_t index = 0;
                 index <
                         runtime->query_branches.size();
                 ++index) {
                runtime->query_branches[index]
                        .query_index = index;

                runtime->query_branches[index]
                        .query =
                        runtime->expanded_queries[index];
            }

            std::size_t generation_candidate_count = 1;

            if (runtime->query_expansion_used) {
                generation_candidate_count +=
                        runtime->expanded_queries.size() *
                        runtime->request
                                .generation_candidate_repeats;
            }

            runtime->generation_candidates.clear();
            runtime->generation_candidates.resize(
                    generation_candidate_count);

            runtime->selected_generation_candidate_index =
                    RAG_INVALID_INDEX;

            // ----------------------------------------------------------
            // Phase B:
            // Each expanded query gets an independent
            // QueryEmbedding -> VectorSearch branch.
            // RetrievalMerge waits for every branch.
            // ----------------------------------------------------------

            RagExecutionPlan phase_b;
            phase_b.request_id =
                    phase_a.request_id;
            phase_b.policy =
                    *schedule_policy;
            phase_b.runtime = runtime.get();

            constexpr int branch_base_id = 100;
            constexpr int merge_id = 100000;
            constexpr int rerank_id = 100001;
            constexpr int generation_base_id = 190000;
            constexpr int generation_prefill_base_id = 200000;
            constexpr int generation_decode_base_id = 210000;
            constexpr int candidate_merge_id = 300000;
            constexpr int finalize_id = 300001;

            for (std::size_t query_index = 0;
                 query_index <
                         runtime->expanded_queries.size();
                 ++query_index) {
                const int embedding_id =
                        branch_base_id +
                        static_cast<int>(
                                query_index * 2);

                const int search_id =
                        embedding_id + 1;

                phase_b.add_node(
                        embedding_id,
                        RagTaskType::QueryEmbedding,
                        RagBackend::CPU);

                phase_b.add_node(
                        search_id,
                        RagTaskType::VectorSearch,
                        RagBackend::CPU);

                auto * embedding_node =
                        phase_b.find_node(
                                embedding_id);

                auto * search_node =
                        phase_b.find_node(
                                search_id);

                if (!embedding_node ||
                    !search_node) {
                    res_err(
                            res,
                            format_error_response(
                                    "failed to construct "
                                    "query branch nodes",
                                    ERROR_TYPE_SERVER));
                    return res;
                }

                embedding_node->query_index =
                        query_index;

                search_node->query_index =
                        query_index;

                phase_b.add_dependency(
                        embedding_id,
                        search_id);
            }

            phase_b.add_node(
                    merge_id,
                    RagTaskType::RetrievalMerge,
                    RagBackend::CPU);

            phase_b.add_node(
                    rerank_id,
                    RagTaskType::Reranking,
                    RagBackend::CPU);

            for (std::size_t candidate_index = 0;
                 candidate_index <
                         runtime->generation_candidates.size();
                 ++candidate_index) {
                if (runtime->request
                            .generation_prefill_backend ==
                    runtime->request
                            .generation_decode_backend) {
                    const int generation_id =
                            generation_base_id +
                            static_cast<int>(candidate_index);

                    phase_b.add_node(
                            generation_id,
                            RagTaskType::Generation,
                            RagBackend::CPU);

                    auto * generation_node =
                            phase_b.find_node(generation_id);

                    if (!generation_node) {
                        res_err(
                                res,
                                format_error_response(
                                        "failed to construct "
                                        "generation candidate node",
                                        ERROR_TYPE_SERVER));
                        return res;
                    }

                    generation_node->candidate_index =
                            candidate_index;
                    generation_node->allowed_backends = {
                        RagBackend::CPU,
                    };
                    generation_node->stealable = false;
                    continue;
                }

                const int prefill_id =
                        generation_prefill_base_id +
                        static_cast<int>(candidate_index);

                const int decode_id =
                        generation_decode_base_id +
                        static_cast<int>(candidate_index);

                phase_b.add_node(
                        prefill_id,
                        RagTaskType::GenerationPrefill,
                        RagBackend::CPU);

                phase_b.add_node(
                        decode_id,
                        RagTaskType::GenerationDecode,
                        RagBackend::CPU);

                auto * prefill_node =
                        phase_b.find_node(prefill_id);

                auto * decode_node =
                        phase_b.find_node(decode_id);

                if (!prefill_node || !decode_node) {
                    res_err(
                            res,
                            format_error_response(
                                    "failed to construct split "
                                    "generation candidate nodes",
                                    ERROR_TYPE_SERVER));
                    return res;
                }

                prefill_node->candidate_index =
                        candidate_index;

                decode_node->candidate_index =
                        candidate_index;

                // Both nodes perform HTTP orchestration on CPU.
                // The child completion request selects the actual
                // Prefill and Decode compute backends.
                prefill_node->allowed_backends = {
                    RagBackend::CPU,
                };

                decode_node->allowed_backends = {
                    RagBackend::CPU,
                };

                prefill_node->stealable = false;
                decode_node->stealable = false;
            }

            phase_b.add_node(
                    candidate_merge_id,
                    RagTaskType::CandidateMerge,
                    RagBackend::CPU);

            phase_b.add_node(
                    finalize_id,
                    RagTaskType::Finalize,
                    RagBackend::CPU);

            for (std::size_t query_index = 0;
                 query_index <
                         runtime->expanded_queries.size();
                 ++query_index) {
                const int search_id =
                        branch_base_id +
                        static_cast<int>(
                                query_index * 2) +
                        1;

                phase_b.add_dependency(
                        search_id,
                        merge_id);
            }

            phase_b.add_dependency(
                    merge_id,
                    rerank_id);

            for (std::size_t candidate_index = 0;
                 candidate_index <
                         runtime->generation_candidates.size();
                 ++candidate_index) {
                if (runtime->request
                            .generation_prefill_backend ==
                    runtime->request
                            .generation_decode_backend) {
                    const int generation_id =
                            generation_base_id +
                            static_cast<int>(candidate_index);

                    phase_b.add_dependency(
                            rerank_id,
                            generation_id);

                    phase_b.add_dependency(
                            generation_id,
                            candidate_merge_id);

                    continue;
                }

                const int prefill_id =
                        generation_prefill_base_id +
                        static_cast<int>(candidate_index);

                const int decode_id =
                        generation_decode_base_id +
                        static_cast<int>(candidate_index);

                phase_b.add_dependency(
                        rerank_id,
                        prefill_id);

                phase_b.add_dependency(
                        prefill_id,
                        decode_id);

                // Keep NPU prefills independent, but serialize CPU decode.
                // One persistent 4-thread Decode Context is reused across
                // all candidates to avoid memory-bandwidth contention.
                if (candidate_index > 0) {
                    const int previous_decode_id =
                            generation_decode_base_id +
                            static_cast<int>(candidate_index - 1);

                    phase_b.add_dependency(
                            previous_decode_id,
                            decode_id);
                }

                phase_b.add_dependency(
                        decode_id,
                        candidate_merge_id);
            }

            phase_b.add_dependency(
                    candidate_merge_id,
                    finalize_id);

            for (auto & node : phase_b.nodes) {
                node.critical_score =
                        critical_score_for_task(node.type);
            }

            if (!scheduler.run(phase_b)) {
                const std::string error =
                        runtime->error.empty()
                                ? "RAG fan-out execution failed"
                                : runtime->error;

                res_err(
                        res,
                        format_error_response(
                                error,
                                ERROR_TYPE_SERVER));
                return res;
            }

            if (runtime->request.stage_model_residency) {
                models.unload_and_wait(runtime->request.generation_model);
            }

            const auto task_type_name =
                    [](RagTaskType type)
                            -> const char * {
                switch (type) {
                    case RagTaskType::DocumentEmbedding:
                        return "document_embedding";
                    case RagTaskType::QueryExpansion:
                        return "query_expansion";
                    case RagTaskType::QueryEmbedding:
                        return "query_embedding";
                    case RagTaskType::VectorSearch:
                        return "vector_search";
                    case RagTaskType::RetrievalMerge:
                        return "retrieval_merge";
                    case RagTaskType::Reranking:
                        return "reranking";
                    case RagTaskType::Generation:
                        return "generation";
                    case RagTaskType::GenerationPrefill:
                        return "generation_prefill";
                    case RagTaskType::GenerationDecode:
                        return "generation_decode";
                    case RagTaskType::CandidateMerge:
                        return "candidate_merge";
                    case RagTaskType::Finalize:
                        return "finalize";
                }

                return "unknown";
            };

            const auto backend_name =
                    [](RagBackend backend)
                            -> const char * {
                switch (backend) {
                    case RagBackend::CPU:
                        return "cpu";
                    case RagBackend::NPU:
                        return "npu";
                    case RagBackend::Auto:
                        return "auto";
                }

                return "unknown";
            };

            const auto task_state_name =
                    [](RagTaskState state)
                            -> const char * {
                switch (state) {
                    case RagTaskState::Pending:
                        return "pending";
                    case RagTaskState::Ready:
                        return "ready";
                    case RagTaskState::Running:
                        return "running";
                    case RagTaskState::Completed:
                        return "completed";
                    case RagTaskState::Failed:
                        return "failed";
                    case RagTaskState::Cancelled:
                        return "cancelled";
                }

                return "unknown";
            };

            json node_metrics = json::array();

            const auto append_metrics =
                    [&](const RagExecutionPlan & plan) {
                for (const auto & entry :
                     plan.metrics) {
                    const auto & metric =
                            entry.second;

                    json item = {
                        {"node_id", entry.first},
                        {
                            "task_type",
                            task_type_name(
                                    metric.task_type),
                        },
                        {
                            "queue_wait_ms",
                            metric.queue_wait_ms,
                        },
                        {
                            "execution_ms",
                            metric.execution_ms,
                        },
                        {
                            "start_ms",
                            metric.start_ms,
                        },
                        {
                            "end_ms",
                            metric.end_ms,
                        },
                        {
                            "preferred_backend",
                            backend_name(
                                    metric
                                            .preferred_backend),
                        },
                        {
                            "actual_backend",
                            backend_name(
                                    metric
                                            .selected_backend),
                        },
                        {
                            "worker_id",
                            metric.worker_id,
                        },
                        {
                            "stolen",
                            metric.stolen,
                        },
                        {
                            "status",
                            task_state_name(
                                    metric.final_state),
                        },
                    };

                    if (metric.query_index !=
                        RAG_INVALID_INDEX) {
                        item["query_index"] =
                                metric.query_index;
                    }

                    if (metric.candidate_index !=
                        RAG_INVALID_INDEX) {
                        item["candidate_index"] =
                                metric.candidate_index;
                    }

                    if (!metric.error_message.empty()) {
                        item["error"] =
                                metric.error_message;
                    }

                    node_metrics.push_back(
                            std::move(item));
                }
            };

            append_metrics(phase_a);
            append_metrics(phase_b);

            // Report wall-clock spans rather than summing node execution
            // times. Multiple candidate Prefill and Decode nodes may run
            // concurrently, so summing their durations would double-count
            // overlapping intervals.
            const auto task_wall_span_ms =
                    [&](RagTaskType first_type,
                        RagTaskType second_type)
                            -> std::size_t {
                bool found = false;
                int64_t earliest_start_ms = 0;
                int64_t latest_end_ms = 0;

                for (const auto & entry :
                     phase_b.metrics) {
                    const auto & metric =
                            entry.second;

                    if (metric.task_type != first_type &&
                        metric.task_type != second_type) {
                        continue;
                    }

                    if (metric.final_state !=
                                RagTaskState::Completed ||
                        metric.end_ms < metric.start_ms) {
                        continue;
                    }

                    if (!found ||
                        metric.start_ms <
                                earliest_start_ms) {
                        earliest_start_ms =
                                metric.start_ms;
                    }

                    if (!found ||
                        metric.end_ms >
                                latest_end_ms) {
                        latest_end_ms =
                                metric.end_ms;
                    }

                    found = true;
                }

                return found &&
                               latest_end_ms >
                                       earliest_start_ms
                        ? static_cast<std::size_t>(
                                  latest_end_ms -
                                  earliest_start_ms)
                        : 0;
            };

            runtime->generation_sub_metrics.prefill_ms =
                    task_wall_span_ms(
                            RagTaskType::GenerationPrefill,
                            RagTaskType::GenerationPrefill);
            runtime->generation_sub_metrics.prefill_stage_ms =
                    runtime->generation_sub_metrics.prefill_ms;

            int64_t latest_prefill_end_ms = 0;
            std::size_t completed_prefill_candidates = 0;
            for (const auto & entry : phase_b.metrics) {
                const auto & metric = entry.second;
                if (metric.task_type != RagTaskType::GenerationPrefill ||
                    metric.final_state != RagTaskState::Completed ||
                    metric.end_ms < metric.start_ms) {
                    continue;
                }
                latest_prefill_end_ms = std::max(latest_prefill_end_ms, metric.end_ms);
                completed_prefill_candidates++;
            }

            const int64_t hot_start_ms =
                    runtime->generation_prefill_hot_start_ms.load(std::memory_order_relaxed);
            runtime->generation_sub_metrics.prefill_hot_ms =
                    hot_start_ms < latest_prefill_end_ms
                            ? static_cast<std::size_t>(latest_prefill_end_ms - hot_start_ms)
                            : 0;
            runtime->generation_sub_metrics.model_load_ms =
                    runtime->generation_model_load_ms.load(std::memory_order_relaxed);
            runtime->generation_sub_metrics.model_load_triggered =
                    runtime->generation_model_load_triggered.load(std::memory_order_relaxed);
            SRV_INF(
                    "GEN_PREFILL_STAGE_SUMMARY request=%llu stage_ms=%zu "
                    "model_load_ms=%zu model_load_triggered=%s hot_ms=%zu "
                    "candidate_count=%zu completed_candidates=%zu "
                    "span=earliest_start_to_latest_handoff\n",
                    static_cast<unsigned long long>(runtime->request_id),
                    runtime->generation_sub_metrics.prefill_stage_ms,
                    runtime->generation_sub_metrics.model_load_ms,
                    runtime->generation_sub_metrics.model_load_triggered ? "true" : "false",
                    runtime->generation_sub_metrics.prefill_hot_ms,
                    runtime->generation_candidates.size(),
                    completed_prefill_candidates);

            runtime->generation_sub_metrics.decode_ms =
                    task_wall_span_ms(
                            RagTaskType::GenerationDecode,
                            RagTaskType::GenerationDecode);

            stage_metrics.generation_ms =
                    task_wall_span_ms(
                            RagTaskType::GenerationPrefill,
                            RagTaskType::GenerationDecode);

            stage_metrics.total_ms =
                    elapsed_ms(total_start_ms);

            runtime->stage_metrics = stage_metrics;

            json branch_results = json::array();

            for (const auto & branch :
                 runtime->query_branches) {
                branch_results.push_back({
                    {
                        "query_index",
                        branch.query_index,
                    },
                    {"query", branch.query},
                    {
                        "retrieval_indices",
                        branch.retrieval_indices,
                    },
                    {"success", branch.success},
                });
            }

            json generation_candidate_results =
                    json::array();

            for (const auto & candidate :
                 runtime->generation_candidates) {
                json item = {
                    {
                        "candidate_index",
                        candidate.candidate_index,
                    },
                    {
                        "source_query_index",
                        candidate.source_query_index ==
                                RAG_INVALID_INDEX
                                ? json(nullptr)
                                : json(
                                      candidate
                                              .source_query_index),
                    },
                    {"seed", candidate.seed},
                    {"content", candidate.content},
                    {
                        "generated_tokens",
                        candidate.generated_tokens,
                    },
                    {
                        "stop_reason",
                        candidate.stop_reason,
                    },
                    {
                        "prefill_success",
                        candidate.prefill_success,
                    },
                    {
                        "generation_handoff_handle",
                        candidate.generation_handoff_handle,
                    },
                    {
                        "generation_handoff_state_bytes",
                        candidate
                                .generation_handoff_state_bytes,
                    },
                    {
                        "generation_handoff_logits_count",
                        candidate
                                .generation_handoff_logits_count,
                    },
                    {
                        "generation_handoff_producer_backend",
                        candidate
                                .generation_handoff_producer_backend,
                    },
                    {"success", candidate.success},
                };

                if (!candidate.error.empty()) {
                    item["error"] = candidate.error;
                }

                generation_candidate_results.push_back(
                        std::move(item));
            }

            json selected_candidate_index = nullptr;

            if (runtime
                        ->selected_generation_candidate_index !=
                RAG_INVALID_INDEX) {
                selected_candidate_index =
                        runtime
                                ->selected_generation_candidate_index;
            }

            res_ok(
                    res,
                    {
                        {
                            "answer",
                            runtime->final_answer,
                        },
                        {
                            "generation_candidate_repeats",
                            runtime->request
                                    .generation_candidate_repeats,
                        },
                        {
                            "selected_generation_candidate_index",
                            selected_candidate_index,
                        },
                        {
                            "generation_candidates",
                            generation_candidate_results,
                        },
                        {
                            "sub_queries",
                            runtime->expanded_queries,
                        },
                        {
                            "query_branches",
                            branch_results,
                        },
                        {
                            "context_chunks",
                            runtime->reranked_chunks,
                        },
                        {
                            "mode_requested",
                            requested_mode,
                        },
                        {
                            "mode_used",
                            requested_mode,
                        },
                        {
                            "scheduler_metrics",
                            node_metrics,
                        },
                        {
                            "stage_metrics",
                            {
                                {
                                    "generation_ms",
                                    stage_metrics.generation_ms,
                                },
                                {
                                    "generation_prefill_ms",
                                    runtime
                                            ->generation_sub_metrics
                                            .prefill_ms,
                                },
                                {
                                    "generation_prefill_stage_ms",
                                    runtime
                                            ->generation_sub_metrics
                                            .prefill_stage_ms,
                                },
                                {
                                    "generation_model_load_ms",
                                    runtime
                                            ->generation_sub_metrics
                                            .model_load_ms,
                                },
                                {
                                    "generation_model_load_triggered",
                                    runtime
                                            ->generation_sub_metrics
                                            .model_load_triggered,
                                },
                                {
                                    "generation_prefill_hot_ms",
                                    runtime
                                            ->generation_sub_metrics
                                            .prefill_hot_ms,
                                },
                                {
                                    "generation_decode_ms",
                                    runtime
                                            ->generation_sub_metrics
                                            .decode_ms,
                                },
                                {
                                    "total_ms",
                                    stage_metrics.total_ms,
                                },
                            },
                        },
                    });

            return res;
        }

        const std::string doc =
                json_value(body, "doc", std::string());

        const std::string query =
                json_value(body, "query", std::string());

        const std::string generation_model =
                json_value(body, "generation_model", std::string());

        const std::string embedding_model =
                json_value(body, "embedding_model", std::string());

        const std::string rerank_model =
                json_value(body, "rerank_model", std::string());

        const bool enable_query_expansion =
                json_value(body, "enable_query_expansion", false);

        const std::string expansion_model =
                json_value(body, "expansion_model", std::string());

        const int top_k =
                json_value(body, "top_k", 20);

        const int top_n =
                json_value(body, "top_n", 5);

        const int max_tokens =
                json_value(body, "max_tokens", 64);

        const float temperature =
                json_value(body, "temperature", 0.1F);

        if (doc.empty()) {
            res_err(
                    res,
                    format_error_response(
                            "doc is missing from the request",
                            ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (query.empty()) {
            res_err(
                    res,
                    format_error_response(
                            "query is missing from the request",
                            ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (generation_model.empty()) {
            res_err(
                    res,
                    format_error_response(
                            "generation_model is missing from the request",
                            ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (embedding_model.empty()) {
            res_err(
                    res,
                    format_error_response(
                            "embedding_model is missing from the request",
                            ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (top_k <= 0) {
            res_err(
                    res,
                    format_error_response(
                            "top_k must be greater than zero",
                            ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (top_n <= 0) {
            res_err(
                    res,
                    format_error_response(
                            "top_n must be greater than zero",
                            ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (max_tokens <= 0) {
            res_err(
                    res,
                    format_error_response(
                            "max_tokens must be greater than zero",
                            ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        // Split the source document before computing embeddings.
        const std::vector<std::string> source_chunks =
                rag_split_document(doc);

        if (source_chunks.empty()) {
            res_err(
                    res,
                    format_error_response(
                            "doc produced no context chunks",
                            ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        // Compute document embeddings through the Router's synchronous
        // internal model interface.
        const auto document_embedding_start_ms = ggml_time_ms();

        const std::vector<std::vector<float>> doc_embeddings =
                models.request_model_embeddings(
                        embedding_model,
                        source_chunks);

        stage_metrics.document_embedding_ms =
                elapsed_ms(document_embedding_start_ms);

        std::vector<std::string> retrieval_queries = {query};
        std::vector<std::string> sub_queries;

        std::string expansion_output;
        std::string expansion_model_used;

        if (enable_query_expansion) {
            const auto query_expansion_start_ms = ggml_time_ms();

            expansion_model_used = expansion_model.empty()
                    ? generation_model
                    : expansion_model;

            const json expansion_request = {
                {"prompt", build_query_expansion_prompt(query)},
                {"n_predict", std::min(max_tokens, 96)},
                {"temperature", 0.2F},
                {"stream", false},
            };

            const json expansion_response =
                    models.request_model_json(
                            expansion_model_used,
                            "/completion",
                            expansion_request);

            if (!expansion_response.contains("content") ||
                !expansion_response["content"].is_string()) {
                throw std::runtime_error(
                        "query expansion model returned an invalid "
                        "completion response");
            }

            expansion_output =
                    expansion_response["content"].get<std::string>();

            sub_queries =
                    rag_split_sub_queries(expansion_output);

            // Match the original pipeline's fallback behavior: when the
            // expansion result cannot produce usable sub-queries, retrieve
            // using the original query.
            if (!sub_queries.empty()) {
                retrieval_queries = sub_queries;
            }

            stage_metrics.query_expansion_ms =
                    elapsed_ms(query_expansion_start_ms);
        }

        const auto query_embedding_start_ms = ggml_time_ms();

        const std::vector<std::vector<float>> query_embeddings =
                models.request_model_embeddings(
                        embedding_model,
                        retrieval_queries);

        stage_metrics.query_embedding_ms =
                elapsed_ms(query_embedding_start_ms);

        if (query_embeddings.size() != retrieval_queries.size()) {
            throw std::runtime_error(
                    "query embedding response count does not match "
                    "retrieval query count");
        }

        const auto vector_search_start_ms = ggml_time_ms();

        std::vector<std::size_t> source_indices;
        source_indices.reserve(source_chunks.size());

        for (std::size_t index = 0;
             index < source_chunks.size();
             ++index) {
            source_indices.push_back(index);
        }

        std::vector<std::vector<std::size_t>> per_query_hits;
        per_query_hits.reserve(query_embeddings.size());

        for (const auto & query_embedding : query_embeddings) {
            per_query_hits.push_back(
                    rag_search_inner_product(
                            doc_embeddings,
                            source_indices,
                            query_embedding,
                            static_cast<std::size_t>(top_k)));
        }

        const std::vector<std::size_t> retrieved_indices =
                rag_merge_subquery_hits(
                        per_query_hits,
                        static_cast<std::size_t>(top_k));

        stage_metrics.vector_search_ms =
                elapsed_ms(vector_search_start_ms);

        std::vector<std::string> retrieved_chunks;
        retrieved_chunks.reserve(retrieved_indices.size());

        for (const std::size_t index : retrieved_indices) {
            if (index >= source_chunks.size()) {
                throw std::runtime_error(
                        "retrieval returned an out-of-range chunk index");
            }

            retrieved_chunks.push_back(source_chunks[index]);
        }

        if (retrieved_chunks.empty()) {
            throw std::runtime_error(
                    "retrieval produced no candidate chunks");
        }

        const bool enable_reranking =
                !rerank_model.empty();

        std::vector<std::size_t> reranked_indices =
                retrieved_indices;

        json rerank_results = json::array();

        if (enable_reranking) {
            const auto reranking_start_ms = ggml_time_ms();

            const json ranked_candidates =
                    models.request_model_rerank(
                            rerank_model,
                            query,
                            retrieved_chunks,
                            static_cast<std::size_t>(top_n));

            reranked_indices.clear();
            reranked_indices.reserve(ranked_candidates.size());

            for (const json & ranked_item : ranked_candidates) {
                const std::size_t candidate_index =
                        ranked_item["index"].get<std::size_t>();

                if (candidate_index >= retrieved_indices.size()) {
                    throw std::runtime_error(
                            "reranker returned an out-of-range "
                            "candidate index");
                }

                const std::size_t source_index =
                        retrieved_indices[candidate_index];

                reranked_indices.push_back(source_index);

                rerank_results.push_back({
                    {"candidate_index", candidate_index},
                    {"source_index", source_index},
                    {"relevance_score",
                     ranked_item["relevance_score"]},
                });
            }

            stage_metrics.reranking_ms =
                    elapsed_ms(reranking_start_ms);
        }

        std::vector<std::string> context_chunks;
        context_chunks.reserve(reranked_indices.size());

        for (const std::size_t source_index : reranked_indices) {
            if (source_index >= source_chunks.size()) {
                throw std::runtime_error(
                        "reranking produced an out-of-range "
                        "source chunk index");
            }

            context_chunks.push_back(source_chunks[source_index]);
        }

        if (context_chunks.empty()) {
            throw std::runtime_error(
                    "reranking produced no context chunks");
        }

        // The current baseline uses a Base model rather than an
        // instruction/chat model. Therefore, use llama.cpp's raw
        // completion endpoint instead of /v1/chat/completions.
        const auto generation_start_ms = ggml_time_ms();

        const std::string prompt =
                build_generation_prompt(query, context_chunks);

        json generation_request = {
            {"model",       generation_model},
            {"prompt",      prompt},
            {"n_predict",   max_tokens},
            {"temperature", temperature},
            {"stream",      false},
        };

        const json generation_response = models.request_model_json(
                generation_model,
                "/completion",
                generation_request);

        if (!generation_response.contains("content") ||
            !generation_response["content"].is_string()) {
            throw std::runtime_error(
                    "generation model returned an invalid completion response");
        }

        const std::string answer =
                generation_response["content"].get<std::string>();

        stage_metrics.generation_ms =
                elapsed_ms(generation_start_ms);

        const std::string mode =
                enable_query_expansion
                        ? (enable_reranking
                                   ? "query_expanded_reranked_"
                                     "retrieval_augmented_sequential"
                                   : "query_expanded_"
                                     "retrieval_augmented_sequential")
                        : (enable_reranking
                                   ? "reranked_"
                                     "retrieval_augmented_sequential"
                                   : "retrieval_augmented_sequential");

        stage_metrics.total_ms = elapsed_ms(total_start_ms);

        const json stage_metrics_json = {
            {"document_embedding_ms", stage_metrics.document_embedding_ms},
            {"query_expansion_ms", stage_metrics.query_expansion_ms},
            {"query_embedding_ms", stage_metrics.query_embedding_ms},
            {"vector_search_ms", stage_metrics.vector_search_ms},
            {"reranking_ms", stage_metrics.reranking_ms},
            {"generation_ms", stage_metrics.generation_ms},
            {"total_ms", stage_metrics.total_ms},
        };

        json response_data = {
            {"answer", answer},
            {"mode", mode},
            {"stage_metrics", stage_metrics_json},
            {"generation_model", generation_model},
            {"embedding_model", embedding_model},
            {"enable_reranking", enable_reranking},
            {"rerank_model", rerank_model},
            {"enable_query_expansion", enable_query_expansion},
            {"expansion_model", expansion_model_used},
            {"expansion_output", expansion_output},
            {"sub_queries", sub_queries},
            {"retrieval_queries", retrieval_queries},
            {"query", query},
            {"top_k", top_k},
            {"top_n", top_n},
            {"source_chunk_count", source_chunks.size()},
            {"retrieved_chunk_count", retrieved_chunks.size()},
            {"retrieved_indices", retrieved_indices},
            {"retrieved_chunks", retrieved_chunks},
            {"reranked_chunk_count", context_chunks.size()},
            {"reranked_indices", reranked_indices},
            {"rerank_results", rerank_results},
            {"context_chunks", context_chunks},
            {"embedding_dimension", doc_embeddings.front().size()},
        };

        // Preserve llama.cpp timing information when available. This will
        // later be useful when comparing the migrated pipeline by stage.
        if (generation_response.contains("timings")) {
            response_data["generation_timings"] =
                    generation_response["timings"];
        }

        if (generation_response.contains("tokens_predicted")) {
            response_data["tokens_predicted"] =
                    generation_response["tokens_predicted"];
        }

        if (generation_response.contains("tokens_evaluated")) {
            response_data["tokens_evaluated"] =
                    generation_response["tokens_evaluated"];
        }

        res_ok(res, response_data);
        return res;
    };

    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (meta->is_running()) {
            res_err(res, format_error_response("model is already running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(meta->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models = [this](const server_http_req & req) {
        bool reload = !req.get_param("reload", "").empty();
        if (reload) {
            models.load_models();
        }
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        for (const auto & meta : all_models) {
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                preset_copy.unset_option("LLAMA_ARG_HOST");
                preset_copy.unset_option("LLAMA_ARG_PORT");
                preset_copy.unset_option("LLAMA_ARG_ALIAS");
                preset_copy.unset_option("LLAMA_ARG_TAGS");
                status["preset"] = preset_copy.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
            }

            // pi coding agent multimodal compatibility
            json input_modalities = json::array({"text"});
            if (meta.multimodal.inp_vision) {
                input_modalities.push_back("image");
            }
            if (meta.multimodal.inp_audio) {
                input_modalities.push_back("audio");
            }
            json architecture {
                {"input_modalities",  input_modalities},
                {"output_modalities", json::array({"text"})},
            };

            json model_info = json {
                {"id",            meta.name},
                {"aliases",       meta.aliases},
                {"tags",          meta.tags},
                {"object",        "model"},    // for OAI-compat
                {"owned_by",      "llamacpp"}, // for OAI-compat
                {"created",       t},          // for OAI-compat
                {"status",        status},
                {"architecture",  architecture},
                {"source",        server_model_source_to_string(meta.source)},
                {"can_remove",    meta.source == SERVER_MODEL_SOURCE_CACHE},
                // {"need_download", meta.need_download},
                // TODO: add other fields, may require reading GGUF metadata
            };

            // merge with loaded_info from the child process if available
            if (meta.is_running()) {
                for (auto it = meta.loaded_info.begin(); it != meta.loaded_info.end(); ++it) {
                    if (!model_info.contains(it.key())) {
                        model_info[it.key()] = it.value();
                    }
                }
            }
            models_json.push_back(model_info);
        }
        res_ok(res, {
            {"data", models_json},
            {"object", "list"},
        });
        return res;
    };

    this->post_router_models_unload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!model->is_running() && model->status != SERVER_MODEL_STATUS_DOWNLOADING) {
            res_err(res, format_error_response("model is not running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(model->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models_sse = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        auto sse_client = std::make_shared<server_models_sse_client>(models.sse);
        res->next = [this, sse_client, &req](std::string & output) -> bool {
            auto result = sse_client->next([&]() {
                return stopping.load(std::memory_order_relaxed) || req.should_stop();
            });
            if (result == nullptr) {
                return false; // client disconnected or should_stop
            }
            output = "data: " + safe_json_to_str(result->to_json()) + "\n\n";
            return true; // listen for the next event
        };
        return res;
    };

    this->post_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        common_params p;
        p.model.hf_repo  = name;
        p.hf_token       = params.hf_token;

        // validate by fetching metadata
        bool ok = false;
        try {
            common_models_handler_init(p, LLAMA_EXAMPLE_SERVER);
            ok = true;
        } catch (...) {
            SRV_ERR("unknown error while validating model '%s'\n", name.c_str());
            // other exceptions will be handled by the outer ex_wrapper()
            throw;
        }

        if (!ok) {
            throw std::invalid_argument("model validation failed, unable to download");
        }

        // reject if model already exists
        if (models.has_model(name)) {
            throw std::invalid_argument("model '" + name + "' already exists");
        }

        // then, proceed with the actual download
        SRV_INF("starting download for model '%s'\n", name.c_str());
        {
            server_models::load_options load_opts;
            load_opts.mode = SERVER_CHILD_MODE_DOWNLOAD;
            load_opts.custom_meta = server_model_meta{};
            load_opts.custom_meta->source = SERVER_MODEL_SOURCE_CACHE;
            load_opts.custom_meta->name   = name;
            models.load(name, load_opts);
        }

        res_ok(res, {{"success", true}});
        return res;
    };

    this->del_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        std::string name = req.get_param("model");
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        models.remove(name); // throws on error

        res_ok(res, {{"success", true}});
        return res;
    };

    this->router_stream_get = [this](const server_http_req & req) {
        // GET /v1/stream/<conv_id>?from=N. resolve the owning child from the conv_id -> model
        // map, 404 when nothing maps
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::optional<server_model_meta> owner = resolve_child_for_conv(models, conv_id);
        if (!owner.has_value()) {
            res_err(res, format_error_response("Stream not found or expired", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        std::string from = req.get_param("from");
        std::string child_path = "/v1/stream/" + encode_qs(conv_id);
        if (!from.empty()) {
            child_path += "?from=" + from;
        }
        SRV_TRC("proxying stream resume to model %s on port %d, path=%s\n",
                owner->name.c_str(), owner->port, child_path.c_str());
        auto proxy = std::make_unique<server_http_proxy>(
                "GET",
                "http",
                CHILD_ADDR,
                owner->port,
                child_path,
                req.headers,
                req.body,
                req.files,
                req.should_stop,
                params.timeout_read,
                params.timeout_write);
        return std::unique_ptr<server_http_res>(std::move(proxy));
    };

    this->router_streams_lookup = [this](const server_http_req & req) {
        // POST /v1/streams/lookup. resolve each requested conv id to its owning child via the
        // map, group the ids per child, and query only the children that actually own some of
        // them instead of fanning out to every ready child. a child only answers for the ids
        // it owns, never lists anything else
        auto res = std::make_unique<server_http_res>();
        std::vector<std::string> requested;
        try {
            json body = json::parse(req.body);
            if (body.contains("conversation_ids") && body["conversation_ids"].is_array()) {
                for (const auto & v : body["conversation_ids"]) {
                    if (v.is_string() && !v.get<std::string>().empty()) {
                        requested.push_back(v.get<std::string>());
                    }
                }
            }
        } catch (const std::exception &) {
            res_ok(res, json::array());
            return res;
        }

        // group requested ids by the child port that owns them, drop ids that map to nothing
        std::unordered_map<int, json> per_child;
        for (const auto & cid : requested) {
            auto owner = resolve_child_for_conv(models, cid);
            if (!owner.has_value()) {
                continue;
            }
            per_child[owner->port].push_back(cid);
        }

        json aggregated = json::array();
        for (auto & [port, ids] : per_child) {
            json child_body = {{"conversation_ids", ids}};
            httplib::Client cli(CHILD_ADDR, port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Post("/v1/streams/lookup", child_body.dump(), "application/json");
            if (!resp || resp->status != 200) {
                continue;
            }
            try {
                json child_arr = json::parse(resp->body);
                if (!child_arr.is_array()) {
                    continue;
                }
                for (auto & entry : child_arr) {
                    if (entry.is_object()) {
                        aggregated.push_back(entry);
                    }
                }
            } catch (const std::exception &) {
                continue;
            }
        }
        res_ok(res, aggregated);
        return res;
    };

    this->router_stream_delete = [this](const server_http_req & req) {
        // DELETE /v1/stream/<conv_id>. resolve the owning child via the map and forward only to
        // it, evict_and_cancel is idempotent on the child
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string child_path = "/v1/stream/" + encode_qs(conv_id);
        auto owner = resolve_child_for_conv(models, conv_id);
        if (owner.has_value()) {
            httplib::Client cli(CHILD_ADDR, owner->port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Delete(child_path.c_str());
            (void) resp; // the child logs its own miss when the session is unknown there
        } else {
            SRV_WRN("router stop for unknown conv_id=%s, no owning child in the conv map\n",
                    conv_id.c_str());
        }
        // drop the tracking entry, the session is being torn down
        models.conv_models.forget(conv_id);
        res->status = 204;
        res->content_type = "application/json";
        return res;
    };
}



//
// server_http_proxy
//

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct pipe_t {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};
    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop()) {
                close_read(); // signal broken pipe to writer
                return false; // cancelled / reader no longer alive
            }
            cv.wait_for(lk, poll_interval);
        }
    }
    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static std::string generate_multipart_boundary() {
    thread_local std::mt19937 gen(std::random_device{}());
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
    std::string boundary = "----llama-cpp-proxy-";
    for (int i = 0; i < 16; i++) {
        boundary += chars[dis(gen)];
    }
    return boundary;
}

static std::string build_multipart_body(
        const json & form_fields,
        const std::map<std::string, uploaded_file> & files,
        const std::string & boundary) {
    static auto sanitize_field = [](const std::string & text) {
        std::string result;
        result.reserve(text.size());
        for (char c : text) {
            if (c != '\n' && c != '\r' && c != '"') {
                result += c;
            }
        }
        return result;
    };

    std::ostringstream body;

    for (const auto & [key, value] : form_fields.items()) {
        if (value.is_array()) {
            for (const auto & item : value) {
                body << "--" << boundary << "\r\n";
                body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
                body << "\r\n";
                if (!item.is_string()) {
                    throw std::invalid_argument("expected string");
                }
                body << item.get<std::string>() << "\r\n";
            }
        } else {
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
            body << "\r\n";
            if (!value.is_string()) {
                throw std::invalid_argument("expected string");
            }
            body << value.get<std::string>() << "\r\n";
        }
    }

    for (const auto & [key, file] : files) {
        body << "--" << boundary << "\r\n";
        body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"";
        if (!file.filename.empty()) {
            body << "; filename=\"" << sanitize_field(file.filename) << "\"";
        }
        body << "\r\n";
        if (!file.content_type.empty()) {
            body << "Content-Type: " << sanitize_field(file.content_type) << "\r\n";
        } else {
            body << "Content-Type: application/octet-stream\r\n";
        }
        body << "\r\n";
        body.write(reinterpret_cast<const char*>(file.data.data()), file.data.size());
        body << "\r\n";
    }

    body << "--" << boundary << "--\r\n";
    return body.str();
}

server_http_proxy::server_http_proxy(
        const std::string & method,
        const std::string & scheme,
        const std::string & host,
        int port,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        const std::map<std::string, uploaded_file> & files,
        const std::function<bool()> should_stop,
        int32_t timeout_read,
        int32_t timeout_write
        ) {
    // shared between reader and writer threads
    auto cli  = std::make_shared<httplib::ClientImpl>(host, port);
    auto pipe = std::make_shared<pipe_t<msg_t>>();

    if (scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        cli.reset(new httplib::SSLClient(host, port));
#else
        throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
    }

    // setup Client
    cli->set_follow_location(true);
    cli->set_connection_timeout(timeout_read, 0); // use --timeout value instead of hardcoded 5 s
    cli->set_write_timeout(timeout_read, 0); // reversed for cli (client) vs srv (server)
    cli->set_read_timeout(timeout_write, 0);
    this->status = 500; // to be overwritten upon response
    this->cleanup = [pipe]() {
        pipe->close_read();
        pipe->close_write();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, cli](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return pipe->write(std::move(msg)); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    // when files are present, the body was converted from multipart form data to JSON
    // we need to reconstruct the multipart body for the downstream server
    std::string effective_body = body;
    std::string override_content_type;
    bool has_files = !files.empty();

    if (has_files) {
        json form_fields = json::parse(body, nullptr, false);
        if (!form_fields.is_discarded()) {
            auto boundary = generate_multipart_boundary();
            effective_body = build_multipart_body(form_fields, files, boundary);
            override_content_type = "multipart/form-data; boundary=" + boundary;
        } else {
            throw std::runtime_error("failed to parse multipart form fields JSON");
        }
    }

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            const auto lowered = to_lower_copy(key);
            if (lowered == "accept-encoding") {
                // disable Accept-Encoding to avoid compressed responses
                continue;
            }
            if (lowered == "transfer-encoding") {
                // the body is already decoded
                continue;
            }
            if (lowered == "content-length") {
                // let httplib calculate Content-Length from the actual body
                continue;
            }
            if (lowered == "content-type") {
                if (has_files) {
                    // we set our own Content-Type with the new boundary
                    continue;
                }
                // when no files but the original request was multipart,
                // the body is now JSON, so correct the Content-Type
                if (value.find("multipart/form-data") != std::string::npos) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (lowered == "host") {
                bool is_default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
                const std::string url_host = common_http_format_host(host);
                req.set_header(key, is_default_port ? url_host : url_host + ":" + std::to_string(port));
            } else {
                req.set_header(key, value);
            }
        }
        req.body = effective_body;
        if (!override_content_type.empty()) {
            req.set_header("Content-Type", override_content_type);
        }
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    SRV_DBG("start proxy thread %s %s\n", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            pipe->write({{}, 500, "", ""}); // header
            pipe->write({{}, 0, "proxy error: " + err_str, ""}); // body
        }
        pipe->close_write(); // signal EOF to reader
        SRV_DBG("%s", "client request thread ended\n");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            SRV_DBG("%s", "received response headers\n");
            this->status  = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        } else {
            SRV_DBG("%s", "no response headers received (request cancelled?)\n");
        }
    }
}
