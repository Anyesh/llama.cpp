#include "llama-expert-streamer.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace {

struct stream_file {
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif

    void open(const std::string & path) {
#ifdef _WIN32
        h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("expert streamer: failed to open " + path);
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("expert streamer: failed to open " + path);
        }
#endif
    }

    void close() {
#ifdef _WIN32
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
            h = INVALID_HANDLE_VALUE;
        }
#else
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
#endif
    }

    bool read_at(void * dst, size_t size, size_t offs) const {
        uint8_t * out = (uint8_t *) dst;
        while (size > 0) {
#ifdef _WIN32
            OVERLAPPED ov = {};
            ov.Offset     = (DWORD) (offs & 0xffffffff);
            ov.OffsetHigh = (DWORD) (offs >> 32);
            DWORD chunk = (DWORD) std::min<size_t>(size, 1u << 30);
            DWORD got   = 0;
            if (!ReadFile(h, out, chunk, &got, &ov) || got == 0) {
                return false;
            }
#else
            ssize_t got = pread(fd, out, size, (off_t) offs);
            if (got <= 0) {
                return false;
            }
#endif
            out  += got;
            offs += got;
            size -= (size_t) got;
        }
        return true;
    }
};

} // namespace

struct llama_expert_streamer::impl {
    struct slot_info {
        int32_t  expert_id = -1;
        uint32_t freq      = 0;
        uint64_t last_use  = 0;
        uint64_t round     = 0; // plan() call that last assigned or hit this slot; slots
                                // touched in the current round must not be evicted by it
        bool     in_flight = false;
    };

    struct tensor_entry {
        uint16_t file_idx;
        size_t   offs;
        size_t   slab_nbytes;
        int32_t  n_expert;
        ggml_tensor * pool;
    };

    struct layer_state {
        std::vector<slot_info>    slots;
        std::vector<tensor_entry> tensors;
        std::vector<int32_t>      in_flight_slots;
        size_t   n_pending = 0; // guarded by q_mtx
        uint64_t tick  = 0;
        uint64_t round = 0;
    };

    struct io_task {
        const stream_file * file;
        size_t offs;
        size_t size;
        ggml_tensor * pool;
        size_t pool_offs;
        bool host;
        layer_state * layer;
    };

    int32_t n_slots;
    int32_t n_io_threads;

    std::vector<stream_file>        files;
    std::map<int32_t, layer_state>  layers;
    mutable std::mutex              mtx;

    llama_expert_streamer_stats stats;

    std::vector<std::thread>  workers;
    std::deque<io_task>       queue;
    std::mutex                q_mtx;
    std::condition_variable   q_cv;
    std::condition_variable   done_cv;
    bool                      stop = false;
    std::atomic<bool>         io_error{false};

    impl(int32_t n_slots, int32_t n_io_threads) : n_slots(n_slots), n_io_threads(std::max(1, n_io_threads)) {
        GGML_ASSERT(n_slots > 0);
        workers.reserve(this->n_io_threads);
        for (int32_t i = 0; i < this->n_io_threads; i++) {
            workers.emplace_back([this]() { worker_loop(); });
        }
    }

    ~impl() {
        {
            std::lock_guard<std::mutex> lock(q_mtx);
            stop = true;
        }
        q_cv.notify_all();
        for (auto & w : workers) {
            w.join();
        }
        for (auto & f : files) {
            f.close();
        }
    }

    void worker_loop() {
        std::vector<uint8_t> bounce; // only grows when a non-host pool needs staging
        while (true) {
            io_task task;
            {
                std::unique_lock<std::mutex> lock(q_mtx);
                q_cv.wait(lock, [this]() { return stop || !queue.empty(); });
                if (stop && queue.empty()) {
                    return;
                }
                task = queue.front();
                queue.pop_front();
            }
            bool ok = false;
            if (task.host) {
                void * dst = (uint8_t *) task.pool->data + task.pool_offs;
                ok = task.file->read_at(dst, task.size, task.offs);
            } else {
                if (bounce.size() < task.size) {
                    bounce.resize(task.size);
                }
                ok = task.file->read_at(bounce.data(), task.size, task.offs);
                if (ok) {
                    ggml_backend_tensor_set(task.pool, bounce.data(), task.pool_offs, task.size);
                }
            }
            if (!ok) {
                io_error.store(true);
            }
            {
                std::lock_guard<std::mutex> lock(q_mtx);
                task.layer->n_pending--;
                if (task.layer->n_pending == 0) {
                    done_cv.notify_all();
                }
            }
        }
    }

    void dispatch_tasks(int32_t il, const std::vector<miss> & misses) {
        std::vector<io_task> tasks;
        layer_state * layer = nullptr;
        {
            std::lock_guard<std::mutex> lock(mtx);

            auto it = layers.find(il);
            GGML_ASSERT(it != layers.end());
            layer = &it->second;

            tasks.reserve(misses.size() * layer->tensors.size());
            for (const auto & m : misses) {
                layer->slots[m.slot].in_flight = true;
                layer->in_flight_slots.push_back(m.slot);
                for (const auto & t : layer->tensors) {
                    GGML_ASSERT(m.expert_id >= 0 && m.expert_id < t.n_expert);
                    GGML_ASSERT(t.file_idx < files.size());
                    const bool host = t.pool->buffer == nullptr || ggml_backend_buffer_is_host(t.pool->buffer);
                    tasks.push_back({
                        &files[t.file_idx],
                        t.offs + (size_t) m.expert_id * t.slab_nbytes,
                        t.slab_nbytes,
                        t.pool,
                        (size_t) m.slot * t.slab_nbytes,
                        host,
                        layer,
                    });
                    stats.n_bytes += t.slab_nbytes;
                }
            }
        }
        if (tasks.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(q_mtx);
            layer->n_pending += tasks.size();
            for (auto & t : tasks) {
                queue.push_back(t);
            }
        }
        q_cv.notify_all();
    }

    void join_layer(int32_t il) {
        layer_state * layer = nullptr;
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = layers.find(il);
            GGML_ASSERT(it != layers.end());
            layer = &it->second;
        }
        {
            std::unique_lock<std::mutex> lock(q_mtx);
            done_cv.wait(lock, [layer]() { return layer->n_pending == 0; });
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (int32_t s : layer->in_flight_slots) {
                layer->slots[s].in_flight = false;
            }
            layer->in_flight_slots.clear();
        }
        if (io_error.load()) {
            io_error.store(false);
            throw std::runtime_error("expert streamer: read failed");
        }
    }
};

llama_expert_streamer::llama_expert_streamer(int32_t n_slots, int32_t n_io_threads) :
    pimpl(new impl(n_slots, n_io_threads)) {}

llama_expert_streamer::~llama_expert_streamer() = default;

void llama_expert_streamer::set_files(const std::vector<std::string> & paths) {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    GGML_ASSERT(pimpl->files.empty());
    pimpl->files.resize(paths.size());
    for (size_t i = 0; i < paths.size(); i++) {
        // empty entries come from FILE-handle loads; they only fail if a streamed
        // tensor actually references that file index
        if (!paths[i].empty()) {
            pimpl->files[i].open(paths[i]);
        }
    }
}

void llama_expert_streamer::add_tensor(int32_t il, uint16_t file_idx, size_t offs, size_t slab_nbytes, int32_t n_expert, ggml_tensor * pool) {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    GGML_ASSERT(pool != nullptr);
    GGML_ASSERT(pool->ne[2] >= pimpl->n_slots);
    GGML_ASSERT(pool->nb[2] == slab_nbytes);
    GGML_ASSERT(ggml_is_contiguous(pool));

    auto & layer = pimpl->layers[il];
    if (layer.slots.empty()) {
        layer.slots.resize(pimpl->n_slots);
    }
    layer.tensors.push_back({file_idx, offs, slab_nbytes, n_expert, pool});
}

void llama_expert_streamer::plan(int32_t il, const int32_t * ids, int32_t n, std::vector<miss> & misses, int32_t * slot_ids_out) {
    const auto t_start = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(pimpl->mtx);

    auto it = pimpl->layers.find(il);
    GGML_ASSERT(it != pimpl->layers.end() && "plan() for a layer with no registered tensors");
    auto & layer = it->second;

    GGML_ASSERT(n <= pimpl->n_slots && "more experts requested than slots");

    misses.clear();
    layer.round++;

    for (int32_t i = 0; i < n; i++) {
        const int32_t id = ids[i];

        int32_t slot = -1;
        for (int32_t s = 0; s < pimpl->n_slots; s++) {
            if (layer.slots[s].expert_id == id) {
                slot = s;
                break;
            }
        }

        if (slot >= 0) {
            if (layer.slots[slot].round != layer.round) {
                pimpl->stats.n_hits++;
            }
        } else {
            // LFU victim, LRU tie-break; slots claimed this round or still being
            // filled by in-flight reads are excluded
            int32_t victim = -1;
            for (int32_t s = 0; s < pimpl->n_slots; s++) {
                const auto & si = layer.slots[s];
                if (si.round == layer.round || si.in_flight) {
                    continue;
                }
                if (si.expert_id < 0) {
                    victim = s;
                    break;
                }
                if (victim < 0 ||
                    si.freq <  layer.slots[victim].freq ||
                    (si.freq == layer.slots[victim].freq && si.last_use < layer.slots[victim].last_use)) {
                    victim = s;
                }
            }
            GGML_ASSERT(victim >= 0);
            slot = victim;
            layer.slots[slot].expert_id = id;
            layer.slots[slot].freq      = 0;
            misses.push_back({id, slot});
            pimpl->stats.n_misses++;
        }

        layer.slots[slot].freq++;
        layer.slots[slot].last_use = ++layer.tick;
        layer.slots[slot].round    = layer.round;

        slot_ids_out[i] = slot;
    }

    pimpl->stats.t_plan_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_start).count();
}

void llama_expert_streamer::execute(int32_t il, const std::vector<miss> & misses) {
    if (misses.empty()) {
        return;
    }
    const auto t_start = std::chrono::steady_clock::now();

    pimpl->dispatch_tasks(il, misses);
    pimpl->join_layer(il);

    std::lock_guard<std::mutex> lock(pimpl->mtx);
    pimpl->stats.t_read_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_start).count();
}

void llama_expert_streamer::dispatch(int32_t il, const std::vector<miss> & misses) {
    if (misses.empty()) {
        return;
    }
    pimpl->dispatch_tasks(il, misses);
}

void llama_expert_streamer::join(int32_t il) {
    const auto t_start = std::chrono::steady_clock::now();

    pimpl->join_layer(il);

    std::lock_guard<std::mutex> lock(pimpl->mtx);
    pimpl->stats.t_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_start).count();
}

llama_expert_streamer_stats llama_expert_streamer::get_stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    return pimpl->stats;
}

void llama_expert_streamer::reset_stats() {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    pimpl->stats = {};
}

int32_t llama_expert_streamer::n_slots() const {
    return pimpl->n_slots;
}
