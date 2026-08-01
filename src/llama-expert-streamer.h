#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// streams routed-expert weight slabs from disk into small per-layer slot pools,
// so that MoE decode does not require the full expert set to be resident

struct llama_expert_streamer_stats {
    uint64_t n_hits      = 0;
    uint64_t n_misses    = 0;
    uint64_t n_bytes     = 0;
    uint64_t t_plan_us   = 0;
    uint64_t t_read_us   = 0; // wall time of synchronous execute() calls
    uint64_t t_wait_us   = 0; // wall time spent in join() after async dispatch
};

struct llama_expert_streamer {
    llama_expert_streamer(int32_t n_slots, int32_t n_io_threads);
    ~llama_expert_streamer();

    llama_expert_streamer(const llama_expert_streamer &) = delete;
    llama_expert_streamer & operator=(const llama_expert_streamer &) = delete;

    // open the model files that expert slabs are read from; indices must match
    // the file_idx values passed to add_tensor
    void set_files(const std::vector<std::string> & paths);

    // register one streamed expert tensor; entries sharing il share one slot table,
    // so a slot always holds the same expert across all pools of a layer.
    // offs is the file offset of expert 0, slab_nbytes the size of one expert slab (nb[2]),
    // pool a persistent [ne0, ne1, n_slots] tensor the slabs are loaded into
    void add_tensor(int32_t il, uint16_t file_idx, size_t offs, size_t slab_nbytes, int32_t n_expert, ggml_tensor * pool);

    struct miss {
        int32_t expert_id;
        int32_t slot;
    };

    // map n expert ids of layer il to slots; misses receive the loads that execute()
    // must perform before the slots are valid; slot_ids_out[i] is the slot for ids[i].
    // no IO happens here; safe to call from an eval callback
    void plan(int32_t il, const int32_t * ids, int32_t n, std::vector<miss> & misses, int32_t * slot_ids_out);

    // read the missed expert slabs into their slots for every pool of layer il;
    // blocks until all reads (and uploads, for non-host pools) are complete
    void execute(int32_t il, const std::vector<miss> & misses);

    // async variant: dispatch() starts the reads and pins the slots so a
    // concurrent plan() cannot evict them; join() must run before the layer's
    // pools are consumed
    void dispatch(int32_t il, const std::vector<miss> & misses);
    void join(int32_t il);

    llama_expert_streamer_stats get_stats() const;
    void reset_stats();

    int32_t n_slots() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
